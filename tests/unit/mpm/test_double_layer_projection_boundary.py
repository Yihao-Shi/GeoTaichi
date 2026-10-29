"""Repeated pressure correction must retain prescribed moving-wall velocities."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    kernel_capture_double_layer_solid_boundary,
    kernel_restore_double_layer_solid_boundary,
    kernel_mark_double_layer_solid_cell_region3d,
    kernel_constrain_double_layer_particles_to_solid_region2d,
    kernel_constrain_double_layer_particles_to_solid_region3d,
)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("phase", [1, 2])
@pytest.mark.parametrize("upper_wall", [False, True])
def test_wall_contact_cannot_escape_along_a_domain_spanning_tangent(taichi_runtime, dimension, phase, upper_wall):
    from src.mpm.structs.Particle import ParticleCloudTwoPhase, ParticleCloudTwoPhase2D

    particle = (ParticleCloudTwoPhase if dimension == 3 else ParticleCloudTwoPhase2D).field(shape=1)
    position = np.full(dimension, 0.5)
    position[2 if dimension == 3 else 0] = 1 - 1e-8
    position[1] = 0.95 if upper_wall else 0.05
    velocity = np.full(dimension, 0.4)
    velocity[1] = 0.3 if upper_wall else -0.3
    particle.x[0], particle.v[0], particle.vf[0], particle.vs[0] = position, velocity, velocity, velocity
    particle.phase[0], particle.active[0], particle.materialID[0] = phase, 1, 1
    lower, upper = np.zeros(dimension), np.ones(dimension)
    if upper_wall:
        lower[1] = 0.9
    else:
        upper[1] = 0.1
    constrain = (
        kernel_constrain_double_layer_particles_to_solid_region3d
        if dimension == 3
        else kernel_constrain_double_layer_particles_to_solid_region2d
    )
    constrain(
        1, ti.Vector([1.0] * dimension), ti.Vector([0.1] * dimension), ti.Vector(lower), ti.Vector(upper), particle
    )
    expected = position.copy()
    expected[1] = 0.899999 if upper_wall else 0.100001
    np.testing.assert_allclose(particle.x.to_numpy()[0], expected)
    expected_velocity = velocity.copy()
    expected_velocity[1] = 0
    np.testing.assert_allclose(particle.v.to_numpy()[0], expected_velocity)
    np.testing.assert_allclose((particle.vs if phase == 1 else particle.vf).to_numpy()[0], expected_velocity)


@pytest.mark.parametrize(
    "case_name",
    [
        "examples.mmpm.TwoPhaseWavemaker3D.two_layer_two_phase_wavemaker_3d",
        "examples.mmpm.TwoPhaseLSDEMCoupling.wavemaker_lsdem_particles_3d.wavemaker_lsdem_particles_3d",
    ],
)
def test_moving_piston_constrains_the_velocity_used_by_each_phase(taichi_runtime, case_name):
    import importlib
    from src.mpm.structs.Particle import ParticleCloudTwoPhase

    particle = ParticleCloudTwoPhase.field(shape=5)
    phase = np.array([2, 1, 2, 2, 1], np.uint8)
    active = np.array([1, 1, 1, 0, 1], np.uint8)
    position = np.array([[0.08, 0.1, 0.1]] * 4 + [[0.12, 0.1, 0.1]], float)
    velocity = np.array([[-0.3, 0.5, -0.2], [-0.1, -0.2, 0.4], [0.4, 0.1, 0.2], [-0.3, 0.5, -0.2], [-0.1, 0.2, 0.3]])
    particle.phase.from_numpy(phase)
    particle.active.from_numpy(active)
    particle.x.from_numpy(position)
    particle.v.from_numpy(velocity)
    particle.vf.from_numpy(velocity)
    particle.vs.from_numpy(velocity)
    importlib.import_module(case_name).enforce_moving_piston_particles(5, 0.1, 0.2, 0.02, particle)
    expected = velocity.copy()
    expected[:2, 0] = 0.2
    np.testing.assert_allclose(particle.v.to_numpy(), expected)
    np.testing.assert_allclose(particle.vf.to_numpy()[phase == 2], expected[phase == 2])
    np.testing.assert_allclose(particle.vs.to_numpy()[phase == 1], expected[phase == 1])
    np.testing.assert_allclose(particle.vs.to_numpy()[phase == 2], velocity[phase == 2])
    np.testing.assert_allclose(particle.vf.to_numpy()[phase == 1], velocity[phase == 1])
    expected_position = position.copy()
    expected_position[[0, 1, 2], 0] = 0.100002
    np.testing.assert_allclose(particle.x.to_numpy(), expected_position)


@pytest.mark.parametrize(
    "case_name",
    [
        "examples.mmpm.TwoPhaseWavemaker3D.two_layer_two_phase_wavemaker_3d",
        "examples.mmpm.TwoPhaseLSDEMCoupling.wavemaker_lsdem_particles_3d.wavemaker_lsdem_particles_3d",
    ],
)
def test_two_phase_wavemaker_solid_walls_cover_edges_and_corners(taichi_runtime, case_name):
    import importlib
    from types import SimpleNamespace
    from src.mpm.boundaries.BoundaryConstraint import BoundaryConstraints

    case = importlib.import_module(case_name)
    TANK, solid_cell_boundaries = case.TANK, case.solid_cell_boundaries
    dx = 0.02
    shape = tuple(np.rint(np.asarray(TANK) / dx).astype(int))
    cells = ti.field(ti.i32, shape=shape)
    cells.fill(0)
    sims = SimpleNamespace(dimension=3, domain=TANK)
    element = SimpleNamespace(grid_size=[dx] * 3, ghost_cell=0)
    boundaries = BoundaryConstraints()
    for boundary in solid_cell_boundaries(dx):
        boundaries.set_solid_cell_boundary(sims, element, boundary, boundary["StartPoint"], boundary["EndPoint"])
    for lower, upper in boundaries.solid_cell_regions:
        kernel_mark_double_layer_solid_cell_region3d(ti.Vector([dx] * 3), ti.Vector(lower), ti.Vector(upper), cells)
    kinds = cells.to_numpy()
    assert np.all(kinds[:, :, 0] == 2)
    assert np.all(kinds[:, 0, :] == 2)
    assert np.all(kinds[:, -1, :] == 2)
    assert np.all(kinds[-1, :, :] == 2)
    assert np.all(kinds[1:-1, 1:-1, 1:] == 0)


@pytest.mark.parametrize("dimension", [2, 3])
def test_pressure_reprojection_preserves_moving_and_stationary_boundaries(taichi_runtime, dimension):
    cells = ti.field(ti.i32, shape=(4,) * dimension)
    kinds = np.ones(cells.shape, dtype=np.int32)
    kinds[(0,) + (slice(None),) * (dimension - 1)] = 2
    cells.from_numpy(kinds)
    shapes = [tuple(4 + int(axis == d) for d in range(dimension)) for axis in range(dimension)]
    velocities = tuple(ti.field(ti.f64, shape=s) for s in shapes)
    saved = tuple(ti.field(ti.f64, shape=s) for s in shapes)
    masks = tuple(ti.field(ti.i32, shape=s) for s in shapes)
    moving = (2,) * dimension
    velocities[0][moving] = 0.37
    kernel_capture_double_layer_solid_boundary(cells, velocities, saved, masks)
    for f in velocities:
        f.fill(-0.52)
    kernel_restore_double_layer_solid_boundary(velocities, saved, masks)
    assert velocities[0][moving] == pytest.approx(0.37, abs=1e-14)
    wall = (1,) + (2,) * (dimension - 1)
    assert velocities[0][wall] == 0.0
    interior = (3,) * dimension
    for f in velocities:
        assert f[interior] == pytest.approx(-0.52, abs=1e-14)
