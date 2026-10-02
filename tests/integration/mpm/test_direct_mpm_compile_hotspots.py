import os

import numpy as np
import pytest


pytestmark = pytest.mark.serial


def _single_particle_body(position):
    from src.mpm.generator.Body import Body

    body = Body()
    body.add_particles(
        np.asarray([position], dtype=np.float64),
        volume=1.0e-3,
        init_v=[0.0, 0.0, 0.0],
        xmin=[0.0, 0.0, 0.0],
        xmax=[1.0, 1.0, 1.0],
    )
    return body


def _direct_kwargs(tmp_path, shape_function):
    return {
        "domain": [1.0, 1.0, 1.0],
        "dx": 0.1,
        "dt": 1.0e-3,
        "bodies": _single_particle_body([0.395, 0.395, 0.395]),
        "young_modulus": 2.0e4,
        "poisson_ratio": 0.3,
        "density": 1.0,
        "gravity": [0.0, 0.0, 0.0],
        "residual": 1.0e-10,
        "interval": 1,
        "step": 1,
        "scale": 1.0,
        "line_search": False,
        "shape_function": shape_function,
        "visualize": False,
        "path": os.fspath(tmp_path / shape_function),
    }


def _assert_shape_contract(solver, expected_count):
    solver.compute_shapefn()

    count = int(solver.offset.to_numpy()[0])
    nodes = solver.LnID.to_numpy()[0, :count]
    shape = solver.shape.to_numpy()[0, :count]
    gradient = solver.dshape.to_numpy()[0, :count]

    assert count == expected_count
    assert np.unique(nodes).size == count
    assert np.all((0 <= nodes) & (nodes < solver.total_background_grid_num))
    assert np.isfinite(shape).all()
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(shape.sum(), 1.0, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(gradient.sum(axis=0), np.zeros(3), rtol=0.0, atol=2.0e-11)


@pytest.mark.isolated_dimension(3)
def test_mpmsolver_runtime_27_point_stencil_preserves_partition(taichi_runtime, tmp_path):
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

    solver = ImplicitULMPM(
        material="neoHookean",
        **_direct_kwargs(tmp_path, "bspline"),
    )
    _assert_shape_contract(solver, 27)


@pytest.mark.isolated_dimension(3)
def test_static_twophase_runtime_stencils_and_assembly_are_finite(taichi_runtime, tmp_path):
    from src.mpm.engines.direct.StaticTwoPhaseULMPM import (
        StaticTwoPhaseULMPM,
    )

    solvers = {}
    for shape_function, expected_count in (
        ("linear", 8),
        ("gimp", 27),
        ("bspline", 27),
    ):
        solver = StaticTwoPhaseULMPM(
            material="linearElastic",
            fluid_density=1.0,
            porosity=0.4,
            mobility=1.0e-6,
            ppd=2,
            **_direct_kwargs(tmp_path, shape_function),
        )
        _assert_shape_contract(solver, expected_count)

        count = int(solver.offset.to_numpy()[0])
        shape_avg = solver.shape_avg.to_numpy()[0, :count]
        dshape_ref = solver.dshape_ref.to_numpy()[0, :count]
        assert np.isfinite(shape_avg).all()
        np.testing.assert_allclose(shape_avg.sum(), 1.0, rtol=0.0, atol=2.0e-12)
        np.testing.assert_allclose(
            dshape_ref,
            solver.dshape.to_numpy()[0, :count],
            rtol=0.0,
            atol=0.0,
        )
        solvers[shape_function] = solver

    solver = solvers["bspline"]
    solver.refresh_active_dofs()
    solver.rhs.fill(0.0)
    solver.hash_matrix.reset_system()
    solver.assemble_system(solver.dt, [0.0, 0.0, 0.0])

    active_nodes = solver.active_dof // solver.component
    raw_count = int(solver.hash_matrix.raw_non_diag_count[0])
    support = int(solver.offset.to_numpy()[0])
    assert raw_count == support * (support - 1)
    assert int(solver.hash_matrix.overflow[0]) == 0
    assert np.isfinite(solver.rhs.to_numpy()[: solver.active_dof]).all()
    assert np.isfinite(solver.hash_matrix.diag.to_numpy()[:active_nodes]).all()
    assert np.isfinite(solver.hash_matrix.non_diag.blockH.to_numpy()[:raw_count]).all()

    solver.assemble_pressure_projection()
    pressure = solver.pressure_projection.to_numpy()
    assert np.isfinite(pressure).all()
    np.testing.assert_allclose(pressure, 0.0, atol=0.0)
    solver.reset_step_solution()
    solver.apply_pressure_projection_to_solution()
    solver.solve_current_step(verbose=False)
    assert solver.last_converged
    assert solver.linear_solver == "BiCGSTAB"


@pytest.mark.isolated_dimension(3)
def test_static_twophase_bound_line_search_converges_without_mode_redispatch(taichi_runtime, tmp_path):
    from src.mpm.engines.direct.StaticTwoPhaseULMPM import (
        StaticTwoPhaseULMPM,
    )

    options = _direct_kwargs(tmp_path, "linear")
    options["line_search"] = True
    solver = StaticTwoPhaseULMPM(
        material="linearElastic",
        fluid_density=1.0,
        porosity=0.4,
        mobility=1.0e-6,
        ppd=2,
        **options,
    )
    solver.refresh_active_dofs()
    solver.reset_step_solution()
    solver.assemble_pressure_projection()
    solver.apply_pressure_projection_to_solution()
    solver.solve_current_step(verbose=False)

    assert solver.last_converged
    assert solver.last_line_search_alpha == 1.0
    assert solver.accept_newton_increment_step.__func__ is StaticTwoPhaseULMPM._accept_line_search_increment


@pytest.mark.isolated_dimension(3)
def test_static_twophase_runtime_dp_newton_matches_reference(
    taichi_runtime,
):
    import taichi as ti

    from src.mpm.engines.direct.StaticTwoPhaseULMPM import (
        StaticTwoPhaseULMPM,
    )
    from tools.diagnostics.mpm.static_twophase.verify_static_twophase_local_jacobian_3d import (
        dp_local_update,
    )

    young_modulus = 2.0e4
    poisson_ratio = 0.3
    lambda_ = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = 0.5 * young_modulus / (1.0 + poisson_ratio)

    solver = StaticTwoPhaseULMPM.__new__(StaticTwoPhaseULMPM)
    solver.lambda_ = lambda_
    solver.mu_ = mu
    solver.bulk_ = lambda_ + 2.0 * mu / 3.0
    solver.dp_friction_angle = 25.0
    solver.dp_dilation_angle = 5.0
    solver.dp_cohesion = 1.0
    solver.dp_shape_factor = 0.0
    solver.dp_local_tol = 1.0e-10
    solver.dp_local_max_iters = 25

    sigma_out = ti.Matrix.field(3, 3, dtype=ti.f64, shape=())
    strain_out = ti.Matrix.field(3, 3, dtype=ti.f64, shape=())
    t_dev_out = ti.field(ti.f64, shape=())
    delta_lambda_out = ti.field(ti.f64, shape=())

    @ti.kernel
    def evaluate(eps: ti.types.ndarray()):
        trial = ti.Matrix([[eps[i, j] for j in ti.static(range(3))] for i in ti.static(range(3))])
        (
            sigma,
            elastic_strain,
            _,
            _,
            _,
            _,
            t_dev,
            delta_lambda,
        ) = solver.dp_local_update(trial)
        sigma_out[None] = sigma
        strain_out[None] = elastic_strain
        t_dev_out[None] = t_dev
        delta_lambda_out[None] = delta_lambda

    eps_trial = np.asarray(
        [
            [-0.02, 0.03, 0.0],
            [0.03, -0.015, 0.0],
            [0.0, 0.0, -0.01],
        ],
        dtype=np.float64,
    )
    evaluate(eps_trial)
    expected = dp_local_update(
        eps_trial,
        lambda_,
        mu,
        solver.dp_friction_angle,
        solver.dp_dilation_angle,
        solver.dp_cohesion,
        solver.dp_shape_factor,
        tol=solver.dp_local_tol,
        max_iters=solver.dp_local_max_iters,
    )

    assert float(delta_lambda_out[None]) > 0.0
    assert np.isfinite(float(t_dev_out[None]))
    np.testing.assert_allclose(sigma_out.to_numpy(), expected[0], rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(strain_out.to_numpy(), expected[1], rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        float(delta_lambda_out[None]),
        expected[6],
        rtol=2.0e-12,
        atol=2.0e-12,
    )


@pytest.mark.isolated_dimension(3)
def test_static_twophase_runtime_dim4_dirichlet_matches_dense_elimination(
    taichi_runtime,
):
    import taichi as ti

    from src.linear_solver.BuildTriplet import BuildTriplet
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary
    from src.mpm.engines.direct.StaticTwoPhaseULMPM import (
        StaticTwoPhaseULMPM,
    )

    solver = StaticTwoPhaseULMPM.__new__(StaticTwoPhaseULMPM)
    solver.component = 4
    solver.active_dof = 8
    solver.dof2node = ti.field(ti.i32, shape=2)
    solver.dof2node.from_numpy(np.asarray([1, 2], dtype=np.int32))
    solver.rhs = ti.field(ti.f64, shape=8)
    solver.new_solution = ti.field(ti.f64, shape=8)
    solver.hash_matrix = BuildTriplet(
        dim=4,
        max_pairs_num=2,
        max_nonzeros=2,
        max_active_nodes=2,
        symmetric=False,
    )

    fixed = np.asarray([1, 6], dtype=np.int32)
    targets = np.asarray([0.75, -0.4], dtype=np.float64)
    full_fixed = np.asarray([4 * 1 + 1, 4 * 2 + 2], dtype=np.int32)
    solver.dirichlet = DirichletBoundary()
    solver.dirichlet.append(
        [[int(full_fixed[0])], [int(full_fixed[1])]],
        targets.tolist(),
    )
    solver.dirichlet.finalize(12)

    diagonal = np.asarray(
        [
            [
                [4.0, 0.2, -0.1, 0.3],
                [0.4, 5.0, 0.5, -0.2],
                [0.1, -0.3, 6.0, 0.6],
                [-0.4, 0.2, 0.7, 7.0],
            ],
            [
                [8.0, -0.2, 0.3, 0.1],
                [0.6, 9.0, -0.4, 0.2],
                [-0.5, 0.7, 10.0, -0.3],
                [0.2, 0.4, 0.5, 11.0],
            ],
        ],
        dtype=np.float64,
    )
    block_01 = np.arange(1.0, 17.0, dtype=np.float64).reshape(4, 4) / 10.0
    block_10 = -np.arange(17.0, 33.0, dtype=np.float64).reshape(4, 4) / 20.0

    solver.hash_matrix.reset_system()
    solver.hash_matrix.diag.from_numpy(diagonal.reshape(2, 16))
    solver.hash_matrix.non_diag.blockI.from_numpy(np.asarray([0, 1], dtype=np.int32))
    solver.hash_matrix.non_diag.blockJ.from_numpy(np.asarray([1, 0], dtype=np.int32))
    solver.hash_matrix.non_diag.blockH.from_numpy(np.stack([block_01.ravel(), block_10.ravel()]))
    solver.hash_matrix.raw_non_diag_count[0] = 2

    rhs = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
    solution = np.linspace(0.2, 0.9, 8, dtype=np.float64)
    solver.rhs.from_numpy(rhs)
    solver.new_solution.from_numpy(solution)

    dense = np.block([[diagonal[0], block_01], [block_10, diagonal[1]]])
    expected_rhs = rhs - dense[:, fixed] @ targets
    expected_rhs[fixed] = solution[fixed] - targets
    expected_matrix = dense.copy()
    expected_matrix[:, fixed] = 0.0
    expected_matrix[fixed, :] = 0.0
    expected_matrix[fixed, fixed] = 1.0

    solver.apply_dirichlet_hash()
    actual_matrix = solver.current_hash_matrix().toarray()
    actual_rhs = solver.rhs.to_numpy()

    np.testing.assert_allclose(actual_matrix, expected_matrix, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(actual_rhs, expected_rhs, rtol=0.0, atol=2.0e-12)
