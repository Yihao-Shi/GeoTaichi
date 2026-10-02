from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_matrix

pytestmark = [pytest.mark.unit, pytest.mark.dem, pytest.mark.ipc, pytest.mark.contact]

from src.dem.Simulation import Simulation
from src.dem.engines.AffineBodyEngine import AffineBodyEngine


class _State:
    def __init__(self):
        self.hat_y = np.zeros((1, 4, 3), dtype=np.float64)
        self.accepted = None

    def begin_step(self, _dt):
        pass

    def pack(self):
        return np.zeros(12, dtype=np.float64)

    def unpack(self, values):
        return np.asarray(values).reshape((1, 4, 3))

    def accept_step(self, values, _dt):
        self.accepted = np.asarray(values).copy()


def _simulation(iterations=1, tolerance=1.0e-7):
    return SimpleNamespace(
        dt=_ScalarField(1.0e-3),
        current_time=0.0,
        current_step=0,
        delta=1.0e-3,
        affine_friction_mode="lagged",
        affine_friction_iterations=iterations,
        affine_friction_max_iterations=10,
        affine_friction_tolerance=tolerance,
    )


class _ScalarField:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, _index):
        return self.value


def test_affine_outer_iteration_rebuilds_only_after_complete_inner_solve():
    engine = AffineBodyEngine()
    engine.state = _State()
    events = []
    engine.operator = SimpleNamespace(
        fully_implicit=False,
        initialize_contact_damping=lambda y, hat_y: events.append(
            ("refresh", float(np.asarray(y)[0]), hat_y is engine.state.hat_y)
        ),
    )
    assembled_gradients = iter(
        [
            np.full(12, 2.0),
            np.full(12, 0.5),
            np.full(12, 1.0e-11),
        ]
    )

    def assemble(_sims, y, need_matrix=True):
        events.append(("assemble", float(np.asarray(y)[0]), need_matrix))
        return 1.0, next(assembled_gradients)

    def inner(_sims, y, energy, gradient):
        events.append(("inner", float(np.asarray(y)[0]), float(gradient[0])))
        return np.asarray(y) + 1.0, energy, gradient, 3

    engine._assemble_system = assemble
    engine._solve_lagged_inner = inner
    engine._solve_direction = lambda _sims, gradient: -np.asarray(gradient)
    engine.step(_simulation(iterations=4), scene=None)

    assert [event[0] for event in events] == [
        "refresh",
        "assemble",
        "inner",
        "refresh",
        "assemble",
        "inner",
        "refresh",
        "assemble",
    ]
    assert engine.last_friction_iterations == 2
    assert engine.last_newton_iterations == 6
    assert engine.last_friction_converged
    # Fixed-point residuals use reference IPC's correction-velocity units.
    assert engine.last_friction_residual == pytest.approx(1.0e-8)
    np.testing.assert_allclose(engine.state.accepted.reshape(-1), 2.0)


def test_affine_default_is_one_lagged_solve_with_updated_system_probe():
    engine = AffineBodyEngine()
    engine.state = _State()
    refreshes = []
    engine.operator = SimpleNamespace(
        fully_implicit=False, initialize_contact_damping=lambda y, hat_y: refreshes.append(1)
    )
    assembled_gradients = iter([np.ones(12), np.zeros(12)])
    engine._assemble_system = lambda _sims, y, need_matrix=True: (
        1.0,
        next(assembled_gradients),
    )
    engine._solve_lagged_inner = lambda _sims, y, energy, gradient: (
        y,
        energy,
        gradient,
        1,
    )
    probes = []
    engine._solve_direction = lambda _sims, gradient: probes.append(np.asarray(gradient).copy()) or -np.asarray(
        gradient
    )

    engine.step(_simulation(), scene=None)

    assert len(refreshes) == 2
    assert len(probes) == 1
    assert engine.last_friction_iterations == 1
    assert engine.last_friction_converged


def test_affine_unbounded_outer_mode_raises_at_safety_cap_without_accepting():
    engine = AffineBodyEngine()
    engine.state = _State()
    engine.operator = SimpleNamespace(fully_implicit=False, initialize_contact_damping=lambda _y, _hat_y: None)
    engine._assemble_system = lambda _sims, _y, need_matrix=True: (
        1.0,
        np.ones(12),
    )
    engine._solve_lagged_inner = lambda _sims, y, energy, gradient: (
        np.asarray(y) + 1.0,
        energy,
        gradient,
        1,
    )
    engine._solve_direction = lambda _sims, gradient: -np.asarray(gradient)
    sims = _simulation(iterations=-1, tolerance=1.0e-12)
    sims.affine_friction_max_iterations = 2

    with pytest.raises(RuntimeError, match="safety cap 2"):
        engine.step(sims, scene=None)

    assert engine.last_friction_iterations == 2
    assert engine.last_friction_terminated_by_cap
    assert engine.state.accepted is None


def test_affine_inner_commits_the_descent_direction_checked_by_line_search():
    engine = AffineBodyEngine()
    sims = SimpleNamespace(
        affine_max_newton_iteration=2,
        affine_newton_tolerance=1.0e-12,
        affine_max_step=10.0,
    )
    engine._solve_direction = lambda _sims, gradient: (np.ones(1) if np.any(gradient) else np.zeros(1))
    checked_directions = []

    def line_search(_sims, _y, _energy, _gradient, direction):
        checked_directions.append(np.asarray(direction).copy())
        return 1.0, 0.0, np.zeros(1)

    engine._line_search = line_search
    result, _, gradient, iterations = engine._solve_lagged_inner(
        sims,
        np.zeros(1),
        1.0,
        np.ones(1),
    )

    np.testing.assert_array_equal(checked_directions[0], -np.ones(1))
    np.testing.assert_array_equal(result, -np.ones(1))
    np.testing.assert_array_equal(gradient, np.zeros(1))
    assert iterations == 1
    assert engine.last_inner_converged


def test_affine_lagged_inner_rejects_nonfinite_barrier_energy():
    engine = AffineBodyEngine()
    engine._solve_direction = lambda *_args: pytest.fail(
        "an infeasible IPC state must be rejected before the linear solve"
    )
    sims = SimpleNamespace(
        affine_max_newton_iteration=2,
        affine_newton_tolerance=1.0e-12,
        affine_max_step=1.0,
    )

    with pytest.raises(RuntimeError, match="not strictly feasible"):
        engine._solve_lagged_inner(sims, np.zeros(1), np.inf, np.ones(1))
    assert engine.last_inner_failure_reason == "non_finite_energy"


def test_affine_lagged_inner_applies_first_nonzero_correction_before_convergence():
    engine = AffineBodyEngine()
    engine.operator = SimpleNamespace(dt=0.25)
    directions = iter([np.full(1, 2.0e-8), np.zeros(1, dtype=np.float64)])
    engine._solve_direction = lambda _sims, _gradient: next(directions)
    applied = []

    def line_search(_sims, _y, _energy, _gradient, direction):
        applied.append(np.asarray(direction).copy())
        return 1.0, 0.0, np.zeros(1)

    engine._line_search = line_search
    sims = SimpleNamespace(
        affine_max_newton_iteration=2,
        affine_newton_tolerance=1.0e-6,
        affine_max_step=1.0,
    )

    result, _, _, iterations = engine._solve_lagged_inner(sims, np.zeros(1), 1.0, -np.ones(1))

    # Reference IPC uses ``k && gradVanish``: the nonzero k=0 correction is
    # applied even though 8e-8 m/s is below tol.  The next zero correction
    # establishes convergence.  This prevents an epsv-scale stick correction
    # from hiding an over-threshold sliding load.
    np.testing.assert_array_equal(applied, [np.full(1, 2.0e-8)])
    np.testing.assert_array_equal(result, np.full(1, 2.0e-8))
    assert iterations == 1
    assert engine.last_inner_converged
    assert engine.last_inner_failure_reason == ""
    assert engine.last_inner_residual == 0.0
    np.testing.assert_allclose(engine.last_inner_correction_history, [8.0e-8, 0.0])


def test_affine_lagged_inner_accepts_exactly_stationary_initial_state():
    engine = AffineBodyEngine()
    engine.operator = SimpleNamespace(dt=0.25)
    engine._solve_direction = lambda _sims, _gradient: np.zeros(1)
    engine._line_search = lambda *_args: pytest.fail("an exactly zero direction must not enter CCD/Armijo")
    sims = SimpleNamespace(
        affine_max_newton_iteration=2,
        affine_newton_tolerance=1.0e-6,
        affine_max_step=1.0,
    )

    result, _, _, iterations = engine._solve_lagged_inner(sims, np.zeros(1), 0.0, np.zeros(1))

    np.testing.assert_array_equal(result, np.zeros(1))
    assert iterations == 0
    assert engine.last_inner_converged
    assert engine.last_inner_correction_history == [0.0]


def test_affine_lagged_cuda_applies_first_nonzero_small_correction():
    engine = AffineBodyEngine()
    direction_norms = iter([2.0e-8, 0.0])
    engine.operator = SimpleNamespace(
        dt=0.25,
        device_gradient_has_nonfinite=lambda: 0,
        device_direction_has_nonfinite=lambda: 0,
        device_direction_inf_norm=lambda: next(direction_norms),
        device_gradient_direction_dot=lambda: -1.0,
        device_scale_direction=lambda _scale: pytest.fail("the mocked direction is already a descent direction"),
    )
    engine._solve_hash_direction_device = lambda _sims: None
    engine._clamp_device_direction = lambda _max_step: None
    applied = []
    engine._line_search_cuda = lambda _sims, energy: applied.append(energy) or energy - 1.0
    sims = SimpleNamespace(
        affine_max_newton_iteration=2,
        affine_newton_tolerance=1.0e-6,
        affine_max_step=1.0,
    )

    energy, iterations = engine._solve_lagged_inner_cuda(sims, energy=2.0)

    assert applied == [2.0]
    assert energy == 1.0
    assert iterations == 1
    assert engine.last_inner_converged
    np.testing.assert_allclose(engine.last_inner_correction_history, [8.0e-8, 0.0])


def test_affine_lagged_inner_uses_official_strict_tolerance_boundary():
    engine = AffineBodyEngine()
    engine.operator = SimpleNamespace(dt=0.25)
    directions = iter([np.full(1, 2.5e-7), np.zeros(1, dtype=np.float64)])
    engine._solve_direction = lambda _sims, _gradient: next(directions)
    applied = []

    def line_search(_sims, y, _energy, _gradient, direction):
        applied.append(np.asarray(direction).copy())
        return 1.0, 0.0, np.zeros(1)

    engine._line_search = line_search
    sims = SimpleNamespace(
        affine_max_newton_iteration=2,
        affine_newton_tolerance=1.0e-6,
        affine_max_step=1.0,
    )

    result, _, _, iterations = engine._solve_lagged_inner(sims, np.zeros(1), 1.0, np.ones(1))

    # Reference IPC uses res < tol, not res <= tol.  A correction exactly on
    # the boundary is applied, and only the following zero probe converges.
    assert len(applied) == 1
    # The mock returned an ascent direction, so the production safeguard
    # reverses it before both line search and commit.
    np.testing.assert_allclose(applied[0], np.full(1, -2.5e-7))
    np.testing.assert_allclose(result, np.full(1, -2.5e-7))
    assert iterations == 1
    assert engine.last_inner_converged
    assert engine.last_inner_residual == 0.0


def test_affine_configured_small_coo_system_uses_mass_whitened_eigensolve():
    matrix = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    mass = np.asarray([[1.5, 0.25], [0.25, 1.25]], dtype=np.float64)
    noninertial = matrix - mass
    assert np.linalg.eigvalsh(noninertial).min() > 0.0
    rhs = np.asarray([1.0, -2.0], dtype=np.float64)
    engine = AffineBodyEngine()
    engine.operator = SimpleNamespace(
        fully_implicit=False,
        dof=2,
        control_mass_matrix_np=mass,
        _add_hessian_shift=lambda *_args: pytest.fail("an SPD direct solve must not need regularization"),
    )
    engine.coo_matrix = SimpleNamespace(_to_scipy=lambda: coo_matrix(noninertial))
    engine._ensure_linear_solver = lambda _sims: None
    engine._solve_coo_pcg = lambda *_args: pytest.fail("the configured direct threshold must bypass PCG")
    sims = SimpleNamespace(
        affine_assemble_type="MatrixFree",
        affine_hessian_shift=1.0e-9,
        affine_direct_hessian_dofs=2,
        affine_linear_tolerance=1.0e-12,
    )

    direction = engine._solve_direction(sims, -rhs)

    np.testing.assert_allclose(direction, np.linalg.solve(matrix, rhs), rtol=1.0e-14, atol=1.0e-15)
    assert engine.last_linear_backend == "MatrixFree"
    assert engine.last_linear_method == "DenseMassWhitenedProjectedEigen"
    assert engine.last_linear_converged
    assert engine.last_linear_iterations == 1
    assert engine.last_linear_failure_reason == ""


def test_affine_direct_solve_restores_erased_inertia_floor():
    # A nearly rank-one PSD contact block has one exact null direction.
    # Independent atomic sums may round its off-diagonal one ulp above its
    # diagonal while the O(1) mass contribution is itself lost next to 1e20,
    # producing a tiny negative mode in an otherwise physically reachable
    # ``M + K_psd`` assembly.  Restore the known mass floor for that mode.
    barrier_scale = 1.0e20
    rounded_off_diagonal = np.nextafter(barrier_scale, np.inf)
    noninertial = np.asarray(
        [
            [barrier_scale, rounded_off_diagonal],
            [rounded_off_diagonal, barrier_scale],
        ],
        dtype=np.float64,
    )
    mass = np.asarray([[2.0, 0.2], [0.2, 3.0]], dtype=np.float64)
    rhs = np.asarray([1.0, 0.5], dtype=np.float64)
    engine = AffineBodyEngine()
    engine.operator = SimpleNamespace(
        fully_implicit=False,
        dof=2,
        control_mass_matrix_np=mass,
        _add_hessian_shift=lambda *_args: pytest.fail("round-off PSD restoration must not use trial shifts"),
    )
    engine.coo_matrix = SimpleNamespace(_to_scipy=lambda: coo_matrix(noninertial))
    engine._ensure_linear_solver = lambda _sims: None
    sims = SimpleNamespace(
        affine_assemble_type="COO",
        affine_hessian_shift=1.0e-9,
        affine_direct_hessian_dofs=2,
        affine_linear_tolerance=1.0e-12,
    )

    direction = engine._solve_direction(sims, -rhs)

    assert np.isfinite(direction).all()
    assert np.dot(rhs, direction) > 0.0
    assert engine.last_linear_converged
    assert engine.last_linear_min_eigenvalue <= 0.0
    assert engine.last_linear_projection_correction > 0.0
    assert -engine.last_linear_min_eigenvalue <= engine.last_linear_spectral_resolution
    assert engine.last_linear_unresolved_modes >= 1
    assert engine.last_linear_original_residual > engine.last_linear_residual


def test_affine_direct_solve_treats_both_ulp_signs_as_unresolved():
    """A null mode must not depend on the sign of one rounded contact entry."""

    barrier_scale = 1.0e20
    mass = np.asarray([[2.0, 0.2], [0.2, 3.0]], dtype=np.float64)
    rhs = np.asarray([1.0, -1.0], dtype=np.float64)
    directions = []

    for rounded_off_diagonal in (
        np.nextafter(barrier_scale, -np.inf),
        np.nextafter(barrier_scale, np.inf),
    ):
        noninertial = np.asarray(
            [
                [barrier_scale, rounded_off_diagonal],
                [rounded_off_diagonal, barrier_scale],
            ],
            dtype=np.float64,
        )
        engine = AffineBodyEngine()
        engine.operator = SimpleNamespace(
            fully_implicit=False,
            dof=2,
            control_mass_matrix_np=mass,
            _add_hessian_shift=lambda *_args: pytest.fail("round-off classification must not use trial shifts"),
        )
        engine.coo_matrix = SimpleNamespace(_to_scipy=lambda matrix=noninertial: coo_matrix(matrix))
        engine._ensure_linear_solver = lambda _sims: None
        sims = SimpleNamespace(
            affine_assemble_type="COO",
            affine_hessian_shift=1.0e-9,
            affine_direct_hessian_dofs=2,
            affine_linear_tolerance=1.0e-12,
        )

        directions.append(engine._solve_direction(sims, -rhs))
        assert engine.last_linear_unresolved_modes >= 1

    np.testing.assert_allclose(directions[0], directions[1], rtol=1.0e-12, atol=1.0e-12)


def test_affine_lagged_convergence_uses_physical_surface_correction():
    engine = AffineBodyEngine()
    engine.operator = SimpleNamespace(
        dt=0.5,
        surface_direction_inf_norm=lambda direction: float(np.linalg.norm(direction, ord=np.inf) * 0.25),
    )
    engine.evaluate_surface_direction_norm = engine.operator.surface_direction_inf_norm
    directions = iter([np.asarray([2.0e-6]), np.asarray([1.6e-6])])
    engine._solve_direction = lambda _sims, _gradient: next(directions)
    engine._line_search = lambda *_args: (
        1.0,
        0.0,
        np.zeros(1, dtype=np.float64),
    )
    sims = SimpleNamespace(
        affine_max_newton_iteration=1,
        affine_newton_tolerance=1.0e-6,
        affine_max_step=1.0,
    )

    _, _, _, updates = engine._solve_lagged_inner(sims, np.zeros(1), 1.0, np.ones(1))

    assert updates == 1
    assert engine.last_inner_control_residual == pytest.approx(3.2e-6)
    assert engine.last_inner_residual == pytest.approx(8.0e-7)
    assert engine.last_inner_converged


def test_affine_lagged_cpu_and_device_line_search_reject_zero_ccd():
    sims = SimpleNamespace(affine_line_search_max_iteration=2)

    cpu_engine = AffineBodyEngine()
    cpu_engine._ccd_step_size = lambda *_args: 0.0
    cpu_engine._assemble_system = lambda *_args, **_kwargs: pytest.fail(
        "alpha=0 must not be accepted as a monotone trial"
    )
    with pytest.raises(RuntimeError, match="no strictly feasible step"):
        cpu_engine._line_search(sims, np.zeros(1), 1.0, np.ones(1), -np.ones(1))

    device_engine = AffineBodyEngine()
    device_engine.operator = SimpleNamespace(
        device_gradient_direction_dot=lambda: -1.0,
        device_backup_line_search_base=lambda: pytest.fail(
            "alpha=0 must be rejected before mutating the device iterate"
        ),
    )
    device_engine._ccd_step_size_device = lambda _sims: 0.0
    with pytest.raises(RuntimeError, match="no strictly feasible step"):
        device_engine._line_search_cuda(sims, 1.0)

    assert cpu_engine.last_inner_failure_reason == "ccd_step_size"
    assert device_engine.last_inner_failure_reason == "ccd_step_size"


def test_affine_lagged_line_search_uses_reference_monotone_energy_rule():
    """Lagged IPC requires E_new <= E_old, not sufficient-decrease Armijo."""

    sims = SimpleNamespace(affine_line_search_max_iteration=3)
    base_energy = 1.0
    trial_energy = 0.99995

    cpu_engine = AffineBodyEngine()
    cpu_engine._ccd_step_size = lambda *_args: 1.0
    cpu_assemblies = []

    def assemble_cpu(_sims, _trial, need_matrix):
        cpu_assemblies.append(bool(need_matrix))
        return trial_energy, np.zeros(1)

    cpu_engine._assemble_system = assemble_cpu
    alpha, accepted_energy, _ = cpu_engine._line_search(
        sims,
        np.zeros(1),
        base_energy,
        np.ones(1),
        -np.ones(1),
    )
    assert alpha == 1.0
    assert accepted_energy == trial_energy
    assert cpu_assemblies == [False, True]
    assert cpu_engine.last_line_search_trials == [(1.0, trial_energy)]
    assert cpu_engine.last_line_search_accepted
    assert cpu_engine.last_line_search_alpha == 1.0
    assert cpu_engine.last_line_search_backtracks == 0

    device_engine = AffineBodyEngine()
    trial_alphas = []
    device_engine.operator = SimpleNamespace(
        device_gradient_direction_dot=lambda: -1.0,
        device_backup_line_search_base=lambda: None,
        device_set_line_search_trial=lambda alpha: trial_alphas.append(alpha),
        device_gradient_has_nonfinite=lambda: 0,
    )
    device_engine._ccd_step_size_device = lambda _sims: 1.0
    device_assemblies = []

    def assemble_device(_sims, need_matrix):
        device_assemblies.append(bool(need_matrix))
        return trial_energy

    device_engine._assemble_system_device = assemble_device
    accepted_device_energy = device_engine._line_search_cuda(sims, base_energy)
    assert accepted_device_energy == trial_energy
    assert trial_alphas == [1.0]
    assert device_assemblies == [False, True]
    assert device_engine.last_line_search_trials == [(1.0, trial_energy)]
    assert device_engine.last_line_search_accepted
    assert device_engine.step_line_search_calls == 1
    assert device_engine.step_line_search_min_alpha == 1.0
    assert device_engine.step_line_search_max_backtracks == 0


def test_affine_line_search_diagnostics_report_the_worst_step_backtrack():
    engine = AffineBodyEngine()
    engine._ccd_step_size = lambda *_args: 1.0
    trial_energies = iter([1.1, 0.9])

    def assemble(_sims, _trial, need_matrix):
        return (0.9 if need_matrix else next(trial_energies)), np.zeros(1)

    engine._assemble_system = assemble
    alpha, _, _ = engine._line_search(
        SimpleNamespace(affine_line_search_max_iteration=3),
        np.zeros(1),
        1.0,
        np.ones(1),
        -np.ones(1),
    )

    assert alpha == 0.5
    assert engine.last_line_search_backtracks == 1
    assert engine.step_line_search_calls == 1
    assert engine.step_line_search_min_alpha == 0.5
    assert engine.step_line_search_max_backtracks == 1
    assert engine.diagnostics_snapshot()["line_search"]["converged"]


def test_affine_device_hash_linear_solve_records_convergence_diagnostics():
    engine = AffineBodyEngine()
    engine._ensure_linear_solver = lambda _sims: None
    copied = []
    engine.operator = SimpleNamespace(
        control_num=1,
        fully_implicit=False,
        device_load_negative_gradient=lambda _rhs: None,
        device_copy_hash_solution_to_direction=lambda solution: copied.append(solution),
    )
    solution = object()
    engine.hash_triplet = SimpleNamespace(
        rhs=object(),
        x=solution,
        finalize_taichi_assembly=lambda: None,
        solve=lambda **_kwargs: {
            "converged": True,
            "iterations": 7,
            "initial_residual": 3.0,
            "residual": 2.0e-10,
        },
    )
    sims = SimpleNamespace(
        affine_assemble_type="HashTriplet",
        affine_hessian_shift=1.0e-9,
        affine_linear_tolerance=1.0e-9,
        affine_linear_max_iteration=50,
    )

    engine._solve_hash_direction_device(sims)

    assert copied == [solution]
    assert engine.last_linear_method == "PCG"
    assert engine.last_linear_converged
    assert engine.last_linear_initial_residual == 3.0
    assert engine.last_linear_residual == 2.0e-10
    assert engine.last_linear_iterations == 7


def test_affine_friction_configuration_validation():
    simulation = object.__new__(Simulation)
    simulation.affine_friction_mode = "lagged"
    simulation.affine_friction_iterations = 1
    simulation.affine_friction_tolerance = 1.0e-7
    simulation.affine_friction_max_iterations = 50
    simulation.affine_dynamic_friction = -1.0
    simulation.affine_static_friction = -1.0
    simulation.affine_viscous_friction = 0.0
    simulation.affine_stribeck_velocity = -1.0
    simulation.affine_friction_profile = "quadratic"
    simulation.set_affine_body_parameters(
        friction_mode="lagged",
        friction_iterations=-1,
        friction_tolerance=1.0e-8,
        friction_max_iterations=7,
    )
    assert simulation.affine_friction_iterations == -1
    assert simulation.affine_friction_tolerance == pytest.approx(1.0e-8)
    assert simulation.affine_friction_max_iterations == 7
    simulation.set_affine_body_parameters(friction_iterations=0)
    assert simulation.affine_friction_iterations == -1
    simulation.set_affine_body_parameters(friction_iterations=-9)
    assert simulation.affine_friction_iterations == -1
    simulation.set_affine_body_parameters(fully_implicit_jacobian_shift=2.0e-8)
    assert simulation.affine_fully_implicit_jacobian_shift == pytest.approx(2.0e-8)
    with pytest.raises(RuntimeError, match="jacobian_shift"):
        simulation.set_affine_body_parameters(fully_implicit_jacobian_shift=np.inf)
    simulation.set_affine_body_parameters(
        fully_implicit_force_atol=2.0e-11,
        fully_implicit_force_rtol=3.0e-8,
    )
    assert simulation.affine_fully_implicit_force_atol == pytest.approx(2.0e-11)
    assert simulation.affine_fully_implicit_force_rtol == pytest.approx(3.0e-8)
    for key in ("fully_implicit_force_atol", "fully_implicit_force_rtol"):
        with pytest.raises(RuntimeError, match=key):
            simulation.set_affine_body_parameters(**{key: np.nan})
    with pytest.raises(RuntimeError, match="friction_iterations"):
        simulation.set_affine_body_parameters(friction_iterations=1.5)
    with pytest.raises(RuntimeError, match="friction_tolerance"):
        simulation.set_affine_body_parameters(friction_tolerance=np.nan)
    with pytest.raises(RuntimeError, match="friction_max_iterations"):
        simulation.set_affine_body_parameters(friction_max_iterations=1.5)
    simulation.set_affine_body_parameters(
        linear_tolerance=1.0e-10,
        linear_max_iteration=17,
        direct_hessian_dofs=0,
        line_search_max_iteration=23,
    )
    assert simulation.affine_linear_tolerance == pytest.approx(1.0e-10)
    assert simulation.affine_linear_max_iteration == 17
    assert simulation.affine_direct_hessian_dofs == 0
    assert simulation.affine_line_search_max_iteration == 23
    simulation.set_affine_body_parameters(
        mixed_contact_capacity=4096,
        mixed_friction_capacity=2048,
    )
    assert simulation.soft_affine_mixed_contact_capacity == 4096
    assert simulation.soft_affine_mixed_friction_capacity == 2048
    for invalid in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(RuntimeError, match="linear_tolerance"):
            simulation.set_affine_body_parameters(linear_tolerance=invalid)
    for invalid in (0, -1, 1.5, np.nan, np.inf):
        with pytest.raises(RuntimeError, match="linear_max_iteration"):
            simulation.set_affine_body_parameters(linear_max_iteration=invalid)
    for invalid in (-1, 1.5, np.nan, np.inf):
        with pytest.raises(RuntimeError, match="direct_hessian_dofs"):
            simulation.set_affine_body_parameters(direct_hessian_dofs=invalid)
    for invalid in (0, -1, 1.5, np.nan, np.inf):
        with pytest.raises(RuntimeError, match="line_search_max_iteration"):
            simulation.set_affine_body_parameters(line_search_max_iteration=invalid)
        with pytest.raises(RuntimeError, match="mixed_contact_capacity"):
            simulation.set_affine_body_parameters(mixed_contact_capacity=invalid)
    with pytest.raises(RuntimeError, match="stribeck_velocity"):
        simulation.set_affine_body_parameters(
            dynamic_friction=0.3,
            static_friction=0.5,
            stribeck_velocity=0.0,
        )


def test_affine_friction_configuration_accepts_only_documented_fallback_sentinel():
    simulation = object.__new__(Simulation)
    simulation.affine_dynamic_friction = 0.2
    simulation.affine_static_friction = 0.2
    simulation.affine_viscous_friction = 0.0
    simulation.affine_stribeck_velocity = 0.1

    simulation.set_affine_body_parameters(
        dynamic_friction=-1.0,
        static_friction=-1.0,
        stribeck_velocity=-1.0,
    )
    assert simulation.affine_dynamic_friction == -1.0
    assert simulation.affine_static_friction == -1.0
    assert simulation.affine_stribeck_velocity == -1.0

    for key in ("dynamic_friction", "static_friction", "stribeck_velocity"):
        with pytest.raises(RuntimeError, match=key):
            simulation.set_affine_body_parameters(**{key: -2.0})
    with pytest.raises(RuntimeError, match="viscous_friction"):
        simulation.set_affine_body_parameters(viscous_friction=-1.0)
