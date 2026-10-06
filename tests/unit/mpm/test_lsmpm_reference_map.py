import numpy as np
import pytest
from types import SimpleNamespace


@pytest.mark.parametrize("shape_type", [0, 1, 2])
def test_reference_map_preserves_affine_motion_without_sdf_accumulation(taichi_runtime, shape_type):
    ti = taichi_runtime
    from src.mpm.soft_particle.ReferenceMap import (
        initialize_soft_reference_sdf_,
        reconstruct_soft_levelset,
    )
    from src.mpm.soft_particle.Structs import SoftBody, SoftMaterialPoint
    from src.mpm.Simulation import Simulation
    from src.mpm.soft_particle.DEMPMBridge import bind_explicit_engine

    n, h, scale = 9, 0.25, 1.7
    count = n**3
    vec3 = ti.types.vector(3, ti.f64)
    box = ti.types.struct(
        xmin=vec3, xmax=vec3, scale=ti.f64, grid_space=ti.f64, gnum=ti.types.vector(3, ti.i32), startGrid=ti.i32
    ).field(shape=1)
    rigid = ti.types.struct(softID=ti.i32, mass_center=vec3, q=ti.types.vector(4, ti.f64)).field(shape=1)
    soft = SoftBody.field(shape=1)
    points = SoftMaterialPoint.field(shape=8)
    grid = ti.types.struct(distance_field=ti.f64).field(shape=count)
    reference = ti.field(ti.f64, shape=count)
    initialized = ti.field(ti.i32, shape=1)
    displacement = ti.Vector.field(3, ti.f64, shape=count)
    derivative = ti.Matrix.field(3, 3, ti.f64, shape=count)
    weight = ti.field(ti.f64, shape=count)
    band = ti.field(ti.i32, shape=())
    uncovered = ti.field(ti.i32, shape=())
    loss = ti.field(ti.f64, shape=())
    initial_angle = 0.7

    def rotation(a):
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])

    R0 = rotation(initial_angle)
    center0 = np.array([2.0, 3.0, 4.0])
    soft.bodyID[0] = 0
    soft.gridNum[0] = count
    soft.gridStart[0] = 0
    soft.scale[0] = scale
    soft.mass_center0[0] = center0
    soft.referenceRotation[0] = R0
    box[0] = dict(xmin=[-scale] * 3, xmax=[scale] * 3, scale=scale, grid_space=h * scale, gnum=[n] * 3, startGrid=0)
    xyz = np.array(
        [[x, y, z] for z in np.linspace(-1, 1, n) for y in np.linspace(-1, 1, n) for x in np.linspace(-1, 1, n)]
    )
    initial = xyz[:, 0].copy()
    grid.distance_field.from_numpy(initial)
    initialize_soft_reference_sdf_(1, soft, grid, reference, initialized)
    sims = Simulation.__new__(Simulation)
    sims.initialize_soft_particle_options()
    sims.soft_levelset_domain_check = False
    sims.soft_shape_function_type = shape_type
    sims.soft_levelset_projection_monitor_band = 2.0
    sims.soft_levelset_projection_extension_iterations = 12
    sims.dt = ti.field(ti.f64, shape=())
    sims.dt[None] = 0.001
    scene = SimpleNamespace(
        softNum=np.array([1]),
        softMaxLogicalGridNum=np.array([count]),
        softPointNum=np.array([8]),
        soft=soft,
        rigid_grid=grid,
        soft_point=points,
        soft_grid=None,
        soft_shape_node=None,
        soft_dshape=None,
        soft_shape_count=None,
        soft_levelset_velocity=displacement,
        soft_levelset_inverse_derivative=derivative,
        soft_levelset_projection_weight=weight,
        soft_levelset_projection_band_nodes=band,
        soft_levelset_projection_uncovered_nodes=uncovered,
        soft_levelset_projection_max_support_loss=loss,
        rigid=rigid,
        box=box,
        soft_levelset_initial_sdf=reference,
        soft_levelset_initial_sdf_initialized=initialized,
    )
    engine = SimpleNamespace()
    bind_explicit_engine(engine, sims, scene)
    X = np.array(list(np.ndindex(2, 2, 2)), dtype=float) * 0.5 - 0.25
    for p in range(8):
        points.active[p] = 1
        points.bodyID[p] = 0
        points.m[p] = 1.0
        points.x0[p] = center0 + scale * (R0 @ X[p])

    # Each motion is total motion, including a return to the initial geometry.
    for angle, shift, axial in ((1.4, 0.07, 0.8), (2.4, -0.06, 1.3), (0.7, 0.0, 1.0)):
        R = rotation(angle)
        A = np.diag([axial, 1.1, 1.2])
        center = np.array([5.0, -2.0, 1.0])
        rigid.softID[0] = 0
        rigid.mass_center[0] = center
        rigid.q[0] = [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]
        for p in range(8):
            points.x[p] = center + scale * (R @ (A @ X[p] + [shift, 0, 0]))
            points.F[p] = R @ A @ R0.T
        # Corrupt the preceding SDF and attempt to reinitialize the reference.
        grid.distance_field.fill(0.0)
        initialize_soft_reference_sdf_(1, soft, grid, reference, initialized)
        assert np.array_equal(reference.to_numpy(), initial)
        engine.initialize_soft_levelset_transport(sims, scene)
        engine.advance_soft_levelset_transport(sims, scene)
        assert sims.soft_levelset_domain_max_departure_excess_cells < 1e-10
        assert uncovered[None] == 0
        valid = np.abs(xyz[:, 0]) <= 0.5
        assert np.allclose(grid.distance_field.to_numpy()[valid], xyz[valid, 0] - shift, atol=1e-10)
        assert np.allclose(derivative.to_numpy()[weight.to_numpy() > 0], np.linalg.inv(A) - np.eye(3), atol=1e-10)

    # Invalid material kinematics must not silently turn into a usable SDF.
    points.F[0] = np.diag([-1.0, 1.0, 1.0])
    with pytest.raises(RuntimeError, match="nonpositive particle F"):
        reconstruct_soft_levelset(
            1,
            count,
            8,
            shape_type,
            2.0,
            12,
            soft,
            grid,
            points,
            displacement,
            derivative,
            weight,
            band,
            uncovered,
            loss,
            rigid,
            box,
            reference,
        )
