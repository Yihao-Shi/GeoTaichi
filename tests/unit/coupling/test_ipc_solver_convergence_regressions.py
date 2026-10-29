from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti
from scipy.sparse import csr_matrix

import src.mpm.config as mpm_config
from src.igampm.engines import Engine as IGAEngine
from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix
from src.mpdem.engines.SoftAffineIPCEngine import SoftAffineIPCEngine
from src.mpdem.engines.SoftAffineIPCOperator import SoftAffineIPCOperator
from src.mpm.soft_particle.IPCMPM import IPCMPM


def _force_settings():
    return SimpleNamespace(
        affine_newton_tolerance=1.0e-7,
        affine_fully_implicit_force_atol=1.0e-12,
        affine_fully_implicit_force_rtol=1.0e-8,
        affine_max_newton_iteration=2,
        affine_max_step=0.0,
    )


@pytest.mark.parametrize("dt", [1.0, 1.0e-2, 1.0e-3])
@pytest.mark.parametrize("newmark", [(1.0, 0.5, 1.0), (0.5, 0.25, 0.5)])
def test_soft_affine_force_tolerance_is_invariant_under_timestep_scaling(dt, newmark):
    physical_residual = 2.0e-7
    settings = _force_settings()
    settings.affine_newmark = newmark
    scale = 2 * newmark[0] * newmark[1] * dt**2
    target = SoftAffineIPCEngine._fully_implicit_force_target(settings, scale * physical_residual, dt)
    assert target / scale == pytest.approx(1.0e-12 + 1.0e-8 * physical_residual, rel=1.0e-12, abs=0.0)


def test_soft_affine_small_incremental_residual_does_not_skip_newton():
    engine = SoftAffineIPCEngine()
    residual = np.array([2.0e-13])

    def solve(*_args):
        raise RuntimeError("nonconverged force reached linear solve")

    engine.operator = SimpleNamespace(
        dt=1.0e-3,
        residual_inf_norm_device=lambda: float(residual[0]),
        solve_direction_device=solve,
    )
    with pytest.raises(RuntimeError, match="nonconverged force reached linear solve"):
        engine._solve_fully_implicit_newton_device(_force_settings())
    assert not engine.last_inner_converged


@pytest.mark.parametrize("dt", [0.0, float("nan"), float("inf")])
def test_soft_affine_force_tolerance_rejects_invalid_timestep(dt):
    with pytest.raises(RuntimeError, match="positive time step"):
        SoftAffineIPCEngine._fully_implicit_force_target(_force_settings(), 1.0, dt)


@pytest.mark.parametrize("direction_norm", [float("nan"), float("inf")])
def test_soft_affine_device_rejects_nonfinite_newton_correction(direction_norm):
    engine = SoftAffineIPCEngine()
    engine.operator = SimpleNamespace(
        dt=0.01,
        residual_inf_norm_device=lambda: 1.0,
        solve_direction_device=lambda sims: {"solution_inf_norm": direction_norm},
    )
    with pytest.raises(RuntimeError, match="correction is non-finite"):
        engine._solve_fully_implicit_newton_device(_force_settings())
    assert engine.last_inner_failure_reason == "non_finite_direction"


def test_soft_affine_applies_small_exact_newton_correction():
    engine = SoftAffineIPCEngine()
    residual = np.array([1.0e-10])
    direction = -residual / 2.0
    applied = []
    engine.operator = SimpleNamespace(
        dt=0.01,
        affine_dof=1,
        total_dof=1,
        max_soft_dof=1,
        soft_direction=SimpleNamespace(from_numpy=lambda values: None),
        residual_inf_norm_device=lambda: float(residual[0]),
        solve_direction_device=lambda sims: {"solution_inf_norm": float(abs(direction[0]))},
        solve_direction=lambda sims, values: (direction.copy(), np.empty(0), direction.copy()),
        apply_jacobian=lambda values: 2.0 * values,
        residual_merit_directional_derivative_device=lambda: float(residual @ (2.0 * direction)),
    )

    def line_search(*args):
        applied.append(direction.copy())
        residual[:] += 2.0 * direction
        return 1.0, residual.copy()

    engine._fully_implicit_line_search_device = line_search
    iterations = engine._solve_fully_implicit_newton_device(_force_settings())
    assert iterations == 1
    assert len(applied) == 1
    assert engine.last_inner_converged


def _identity_hash(dimension):
    matrix = BuildTriplet(
        dim=dimension,
        max_pairs_num=1,
        max_nonzeros=1,
        max_active_nodes=1,
        symmetric=False,
        matrix_symmetric=False,
        solver="BiCGSTAB",
        device_reduction=True,
    )
    matrix.load_from_scipy_blocks(csr_matrix(np.eye(dimension)), active_nodes=1)
    return matrix


@pytest.mark.parametrize("scale", [1.0e-11, 1.0e-16])
def test_mpm_fully_implicit_linear_solve_resolves_rhs_below_configured_tolerance(taichi_runtime, scale):
    dimension = mpm_config.DIM
    matrix = _identity_hash(dimension)
    rhs = ti.field(ti.f64, shape=dimension)
    rhs.fill(scale)
    correction = ti.field(ti.f64, shape=dimension)
    contact = object.__new__(IPCMPM)
    contact.friction_mode = "fully_implicit"
    contact.cuda_monolithic_solver = True
    contact.mpm = SimpleNamespace(
        active_dof=dimension,
        rhs=rhs,
        incre_resolution=correction,
        linear_solver_tolerance=1.0e-10,
        linear_solver_relative_tolerance=0.0,
        linear_solver_max_iters=50,
    )
    contact.assemble_current_system = lambda disp: matrix
    result = contact.solve_current_system()
    assert result["converged"]
    np.testing.assert_allclose(correction.to_numpy(), scale, rtol=1.0e-12, atol=0.0)


def test_soft_affine_linear_solve_resolves_small_incremental_force(taichi_runtime):
    operator = object.__new__(SoftAffineIPCOperator)
    operator.fully_implicit = True
    operator.cuda_hot_loop = True
    operator.profile = False
    operator.max_dof = operator.total_dof = operator.affine_dof = 3
    operator.max_soft_dof = 1
    operator.hash_triplet = _identity_hash(3)
    operator.global_grad = ti.field(ti.f64, shape=3)
    operator.global_grad.fill(2.0e-13)
    operator.soft_direction = ti.field(ti.f64, shape=1)
    operator.device_status = ti.field(ti.i32, shape=())
    operator.affine = SimpleNamespace(control_num=1, direction_y=ti.Vector.field(3, ti.f64, shape=1))
    sims = SimpleNamespace(
        affine_linear_tolerance=1.0e-9,
        affine_linear_relative_tolerance=0.0,
        affine_linear_max_iteration=50,
        affine_max_step=0.0,
    )
    result = operator.solve_direction_device(sims)
    assert result["converged"]
    direction = operator.affine.direction_y.to_numpy().ravel()
    np.testing.assert_allclose(direction, -2.0e-13, rtol=1.0e-12, atol=0.0)


@pytest.mark.parametrize("backend", ["HashTriplet", "COO"])
def test_iga_fully_implicit_newton_resolves_small_force_with_coarse_linear_settings(taichi_runtime, backend):
    engine = object.__new__(IGAEngine)
    engine.assemble_type = backend
    engine.monolithic_solver_name = "BiCGSTAB"
    engine.monolithic_linear_solver_tolerance = 1.0e-10
    engine.monolithic_linear_solver_relative_tolerance = 1.0
    engine.monolithic_linear_solver_max_iters = 50
    engine.fully_implicit_force_rtol = 1.0e-8
    engine.monolithic_rhs = ti.field(ti.f64, shape=3)
    engine.monolithic_rhs.fill(1.0e-11)
    engine.monolithic_correction = ti.field(ti.f64, shape=3)
    if backend == "HashTriplet":
        engine.monolithic_hash_matrix = _identity_hash(3)
    else:
        matrix = CoordinateSparseMatrix(3, 3, preconditioned=True, symmetry=False)
        matrix.rows.from_numpy(np.arange(3, dtype=np.int32))
        matrix.cols.from_numpy(np.arange(3, dtype=np.int32))
        matrix.data.fill(1.0)
        matrix.linear_operator.update_nnz(3)
        engine.monolithic_coo_matrix = matrix
        engine.monolithic_coo_diagonal = ti.field(ti.f64, shape=3)
        engine.monolithic_coo_diagonal.fill(1.0)
    system = {"active_nodes": 1, "active_dof": 3, "active_mpm_dof": 3}
    engine.assemble_monolithic_newton_system = lambda **kwargs: system.copy()
    engine._save_device_monolithic_entry_displacements = lambda: None
    engine._restore_device_monolithic_entry_displacements = lambda: None
    engine._initialize_fully_implicit_velocity_guess_device = lambda: None
    engine._device_physical_residual_squared = lambda dofs: float(
        engine.monolithic_rhs.to_numpy() @ engine.monolithic_rhs.to_numpy()
    )
    engine._device_dirichlet_residual = lambda dofs: 0.0
    engine._split_device_monolithic_correction = lambda dofs: None
    engine._assemble_device_physical_tangent_product = lambda **kwargs: -float(
        engine.monolithic_rhs.to_numpy() @ engine.monolithic_correction.to_numpy()
    )
    engine._material_feasible_step_device = lambda: 1.0
    engine.mpm = engine.iga = SimpleNamespace(grid_disp=None)
    engine.initialize_barrier = lambda *args: None

    def line_search(*args, **kwargs):
        remaining = engine.monolithic_rhs.to_numpy() - engine.monolithic_correction.to_numpy()
        engine.monolithic_rhs.from_numpy(remaining)
        return {"accepted": True, "step": 1.0, "residual": float(np.linalg.norm(remaining)), "system": system.copy()}

    engine.fully_implicit_residual_armijo_device = line_search
    result = engine._solve_fully_implicit_newton_device(
        include_friction=True,
        max_iterations=2,
        absolute_tolerance=1.0e-12,
        dirichlet_tolerance=1.0e-12,
        verbose=False,
    )
    assert result["converged"]
    assert result["iterations"] == 1
    assert result["residual"] <= 1.0e-12
    np.testing.assert_allclose(engine.monolithic_correction.to_numpy(), 1.0e-11, rtol=1.0e-12, atol=0.0)
