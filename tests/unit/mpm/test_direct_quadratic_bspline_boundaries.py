"""Boundary B-splines reproduce affine fields, including axisymmetric hoop stretch."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.cpu, pytest.mark.serial]


@pytest.mark.parametrize("shape_name,node_count", [("bspline", 3), ("bspline", 4), ("linear", 3)])
def test_boundary_grid_requires_distinct_first_and_last_two_nodes(shape_name, node_count):
    from src.mpm.engines.direct.MPMSolver import MPMSolver

    engine = object.__new__(MPMSolver)
    engine.shape_function_name = shape_name
    engine.bodies = SimpleNamespace(bodies={"soil": dict(grid_size=1.0, xmin=[0.0, 0.0], xmax=[node_count - 1.0] * 2)})
    if shape_name == "bspline" and node_count < 4:
        with pytest.raises(ValueError, match="at least four grid nodes per axis"):
            engine.add_body_info()
    else:
        engine.add_body_info()
        assert engine.total_background_grid_num == node_count**2


def _basis(dimension):
    from src.mpm.engines.direct.ImplicitMPM import ImplicitMPM
    from src.mpm.engines.direct.MPMSolver import _make_shape_function

    engine = object.__new__(ImplicitMPM)
    engine.shape_func, engine.shape_function_name = _make_shape_function("QuadBSpline")
    points = np.array([[0.0, 0.0], [0.025, 0.075], [0.235, 0.457], [0.999, 0.725], [1.0, 1.0]])
    if dimension == 3:
        points = np.column_stack((points, [0.025, 0.457, 0.73, 0.999, 1.0]))
    count, support = len(points), 3**dimension
    body_type = ti.types.struct(
        goffset=ti.i32,
        grid_num=ti.types.vector(dimension, ti.i32),
        xmin=ti.types.vector(dimension, ti.f64),
        grid_size=ti.f64,
    )
    particle_type = ti.types.struct(x=ti.types.vector(dimension, ti.f64), bodyID=ti.i32)
    engine.body = body_type.field(shape=1)
    engine.body[0] = dict(goffset=0, grid_num=[11] * dimension, xmin=[0.0] * dimension, grid_size=0.1)
    engine.particle = particle_type.field(shape=count)
    engine.particle.x.from_numpy(points)
    engine.particleNum = ti.field(ti.i32, shape=1)
    engine.particleNum[0] = count
    engine.offset = ti.field(ti.i32, shape=count)
    engine.invalid_stencil_particle = ti.field(ti.i32, shape=())
    engine.LnID = ti.field(ti.i32, shape=(count, support))
    engine.shape = ti.field(ti.f64, shape=(count, support))
    engine.dshape = ti.Vector.field(dimension, ti.f64, shape=(count, support))
    hessians = ti.Matrix.field(dimension, dimension, ti.f64, shape=(count, support))

    @ti.kernel
    def evaluate_hessians():
        for i, j in hessians:
            hessians[i, j] = engine.shape_hessian(i, j)

    engine.compute_shapefn()
    evaluate_hessians()
    ids = engine.LnID.to_numpy()
    nodes = np.stack([(ids // 11**d) % 11 * 0.1 for d in range(dimension)], axis=-1)
    shape, gradient = engine.shape.to_numpy(), engine.dshape.to_numpy()
    np.testing.assert_allclose(shape.sum(axis=1), 1, atol=2e-15)
    np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=2e-14)
    np.testing.assert_allclose(np.einsum("ij,ijk->ik", shape, nodes), points, atol=2e-15)
    np.testing.assert_allclose(
        np.einsum("ijk,ijl->ikl", gradient, nodes),
        np.broadcast_to(np.eye(dimension), (count, dimension, dimension)),
        atol=2e-14,
    )
    np.testing.assert_allclose(hessians.to_numpy().sum(axis=1), 0, atol=2e-13)
    # Both basis gradients and contact-position Hessians must use the boundary polynomial.
    for axis in range(dimension):
        values = []
        for sign in [-1, 1]:
            trial = points.copy()
            trial[1:4, axis] += sign * 1e-6
            engine.particle.x.from_numpy(trial)
            engine.compute_shapefn()
            values.append(engine.dshape.to_numpy()[1:4].copy())
        np.testing.assert_allclose((values[1] - values[0]) / 2e-6, hessians.to_numpy()[1:4, :, :, axis], atol=2e-7)
    engine.particle.x.from_numpy(points)
    engine.compute_shapefn()
    return engine, points


@pytest.mark.isolated_dimension(2)
def test_boundary_basis_and_axisymmetric_hoop_reproduce_affine_expansion(taichi_runtime):
    engine, points = _basis(2)
    engine.axis_offset = 0.0
    engine.node2dof = ti.field(ti.i32, shape=121)
    engine.node2dof.from_numpy(np.arange(1, 122, dtype=np.int32))
    displacement = ti.field(ti.f64, shape=242)
    nodes = np.column_stack((np.arange(121) % 11, np.arange(121) // 11)) * 0.1
    displacement.from_numpy((nodes * np.array([0.03, -0.02])).reshape(-1))
    maps = ti.Matrix.field(3, 3, ti.f64, shape=len(points))

    @ti.kernel
    def evaluate():
        for i in range(1, len(points)):
            maps[i] = engine.get_axisymmetric_incremental_map(i, displacement)

    evaluate()
    np.testing.assert_allclose(
        maps.to_numpy()[1:], np.broadcast_to(np.diag([1.03, 0.98, 1.03]), (len(points) - 1, 3, 3)), atol=2e-14
    )


@pytest.mark.isolated_dimension(3)
def test_boundary_basis_in_three_dimensions(taichi_runtime):
    _basis(3)
