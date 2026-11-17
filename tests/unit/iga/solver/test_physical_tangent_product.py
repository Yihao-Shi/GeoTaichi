"""Recover the physical merit derivative after nonzero Dirichlet elimination."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.iga, pytest.mark.isolated_dimension(2)]


@pytest.mark.parametrize("backend,symmetric", [("HashTriplet", True), ("HashTriplet", False), ("COO", False)])
def test_physical_product_from_constrained_matrix(taichi_runtime, backend, symmetric):
    import src.igampm.config as config

    config.set_dimension(2)
    from src.igampm.engines.ContactEngine import ContactEngineMixin
    from src.igampm.engines.FrictionEngine import FrictionEngineMixin
    from src.linear_solver.BuildTriplet import BuildTriplet
    from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix

    @ti.data_oriented
    class Engine(ContactEngineMixin, FrictionEngineMixin):
        pass

    engine = Engine()
    engine.assemble_type = backend
    engine.iga = SimpleNamespace(degree_of_freedom=4)
    rng = np.random.default_rng(103)
    physical = rng.normal(size=(6, 6))
    if symmetric:
        physical = physical + physical.T
    physical += 10 * np.eye(6)
    direction, rhs = rng.normal(size=(2, 6))
    fixed = np.array([1, 0, 0, 1, 0, 0], dtype=np.int32)
    ids = np.flatnonzero(fixed)
    eliminated_rhs = rhs - physical[:, ids] @ direction[ids]
    eliminated_rhs[ids] = direction[ids]
    constrained = physical.copy()
    constrained[:, ids] = constrained[ids, :] = 0
    constrained[ids, ids] = 1
    for name, values in (
        ("monolithic_correction", direction),
        ("monolithic_rhs", eliminated_rhs),
        ("monolithic_physical_rhs", rhs),
    ):
        field = ti.field(ti.f64, shape=6)
        field.from_numpy(values)
        setattr(engine, name, field)
    engine.monolithic_fixed = ti.field(ti.i32, shape=6)
    engine.monolithic_fixed.from_numpy(fixed)
    engine.monolithic_tangent_product = ti.field(ti.f64, shape=6)
    rows, columns = np.indices((6, 6), dtype=np.int32)
    if backend == "HashTriplet":
        matrix = BuildTriplet(
            dim=2,
            max_pairs_num=40,
            max_nonzeros=6,
            max_active_nodes=3,
            symmetric=False,
            matrix_symmetric=symmetric,
            full_symmetric_input=symmetric,
        )
        matrix.install_fixed_pattern(np.array([[0, 1]], dtype=np.int32))
        matrix.assemble_scalar_triplets(rows.ravel(), columns.ravel(), constrained.ravel())
        matrix.finalize_taichi_assembly()
        engine.monolithic_hash_matrix = matrix
    else:
        matrix = CoordinateSparseMatrix(36, 6, symmetry=False)
        matrix.rows.from_numpy(rows.ravel())
        matrix.cols.from_numpy(columns.ravel())
        matrix.data.from_numpy(constrained.ravel())
        engine.monolithic_coo_matrix = matrix
    slope = engine._assemble_device_physical_tangent_product(active_mpm_dof=2, include_friction=not symmetric)
    expected = physical @ direction
    expected[ids] = 0
    np.testing.assert_allclose(engine.monolithic_tangent_product.to_numpy(), expected, rtol=1e-13, atol=1e-12)
    assert slope == pytest.approx(-rhs @ expected, rel=1e-13, abs=1e-12)
