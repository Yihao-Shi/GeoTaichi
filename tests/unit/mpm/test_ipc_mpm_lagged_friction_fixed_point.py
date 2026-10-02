from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.mpm.soft_particle.IPCMPM import (
    IPCMPM,
    _normalize_contact_search,
    _normalize_friction_iterations,
    _positive_float,
    _positive_integer,
)
from src.mpm.soft_particle.IPCTLMPM import IPCTLMPM
from src.mpm.soft_particle.IPCULMPM import IPCULMPM


class _ScalarField:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, _):
        return self.value

    def __setitem__(self, _, value):
        self.value = value


class _FillField:
    def __init__(self):
        self.values = []

    def fill(self, value):
        self.values.append(value)


def test_standalone_system_finalizes_reusable_raw_sources():
    ipc = object.__new__(IPCMPM)
    ipc.cuda_monolithic_solver = True
    calls = []
    ipc.assemble_current_sources = lambda *args, **kwargs: (calls.append((args, kwargs)) or 12)
    ipc._assemble_cuda_monolithic_matrix = lambda active_dof: (calls.append(("finalize", active_dof)) or "matrix")

    assert ipc.assemble_current_system("disp", project_spd=False) == "matrix"
    assert calls == [
        (
            ("disp",),
            {
                "need_matrix": True,
                "project_spd": False,
                "exact_plastic_tangent": False,
            },
        ),
        ("finalize", 12),
    ]
    calls.clear()
    assert ipc.assemble_current_system("trial", need_matrix=False) is None
    assert calls == [
        (
            ("trial",),
            {
                "need_matrix": False,
                "project_spd": None,
                "exact_plastic_tangent": False,
            },
        )
    ]


def test_raw_source_assembly_stops_before_matrix_finalize():
    ipc = object.__new__(IPCMPM)
    events = []
    marker = object()
    ipc.mpm = SimpleNamespace(
        active_dof=6,
        grid_disp=marker,
        rhs=MagicMock(),
        hash_matrix=SimpleNamespace(reset_system=lambda: events.append("reset_mpm")),
        damping=0.0,
        gravity=np.zeros(3),
        integration=(0.5, 0.25, 0.5),
        assemble_inertia_force=lambda *args: events.append("inertia"),
        assemble_material_force=lambda *args: events.append("material"),
        assemble_stiffness_matrix_hash=lambda *args, **kwargs: events.append(("stiffness", kwargs)),
        assemble_mass_matrix_hash=lambda: events.append("mass"),
        neumann=SimpleNamespace(num=0),
    )
    ipc.activate_barrier = False
    ipc.activate_fric = False
    ipc.friction_mode = "lagged"
    ipc.is_semi = False
    ipc.pbarrierNum = _ScalarField(0)
    ipc.gbarrierNum = _ScalarField(0)
    ipc.curr_friction_contact_num = 0
    ipc.update_particle_pos = lambda value: events.append(("positions", value))
    ipc.rebuild_barrier_contacts = lambda: events.append("contacts") or True

    assert ipc.assemble_current_sources(marker, project_spd=False) == 6
    ipc.mpm.rhs.fill.assert_called_once_with(0)
    assert events[:5] == [
        "reset_mpm",
        "inertia",
        "material",
        ("stiffness", {"project_spd": False, "exact_plastic_tangent": False}),
        "mass",
    ]
    assert events[5:] == [("positions", marker), "contacts"]


def test_external_coupled_adjoint_reuses_device_plastic_pullback():
    ipc = object.__new__(IPCMPM)
    ipc.friction_mode = "lagged"
    ipc.activate_fric = True
    ipc.mpm = SimpleNamespace(material=type("FiniteStrainVonMisesModel", (), {})())
    events = []
    ipc.gravity_vjp = object()
    ipc.friction_parameter_vjp = _FillField()
    ipc._differentiate_elastic_gravity = lambda: events.append("gravity")
    ipc._differentiate_plastic_input_history = lambda: events.append("history")
    ipc._differentiate_lagged_friction_parameters = lambda: events.append("friction")

    ipc.pullback_plastic_equilibrium_from_current_adjoint_device()

    assert ipc.friction_parameter_vjp.values == [0.0]
    assert events == ["gravity", "history", "friction"]


def test_external_coupled_adjoint_splits_particle_prepare_and_finish():
    ipc = object.__new__(IPCMPM)
    events = []
    ipc._validate_plastic_particle_adjoint_configuration = lambda: events.append("validate")
    ipc._differentiate_plastic_commit_state = lambda: events.append("commit")
    ipc._differentiate_particle_advection = lambda: events.append("advection")
    ipc.trajectory_grid_vjp = object()

    assert ipc.prepare_plastic_particle_step_adjoint_device() is ipc.trajectory_grid_vjp
    assert events == ["validate", "commit", "advection"]

    events.clear()
    ipc.pullback_plastic_equilibrium_from_current_adjoint_device = lambda: events.append("equilibrium")
    ipc._differentiate_inertia_input_state = lambda: events.append("inertia")
    ipc._differentiate_material_position = lambda: events.append("material")
    ipc.curr_barrier_contact_num = 1
    ipc._differentiate_barrier_position = lambda: events.append("barrier")
    ipc.activate_fric = True
    ipc.curr_friction_contact_num = 1
    ipc._differentiate_lagged_friction_position = lambda: events.append("friction")
    ipc.mpm = SimpleNamespace(tractionNum=_ScalarField(1))
    ipc._differentiate_traction_position = lambda: events.append("traction")
    ipc._differentiate_p2g_input_state = lambda: events.append("p2g")
    ipc._combine_plastic_step_vjp = lambda: events.append("combine")

    ipc.finish_plastic_particle_step_adjoint_device()

    assert events == [
        "equilibrium",
        "inertia",
        "material",
        "barrier",
        "friction",
        "traction",
        "p2g",
        "combine",
    ]


def test_direct_ipc_contact_search_never_falls_back_on_a_typo():
    assert _normalize_contact_search("linked_cell") == "LinkedCell"
    assert _normalize_contact_search("Brust") == "Brust"
    with pytest.raises(ValueError, match="contact_search"):
        _normalize_contact_search("linkd-cell")


def _controller(iterations, residuals, tolerance=1.0e-3, safety_cap=50):
    ipc = object.__new__(IPCMPM)
    ipc.activate_fric = True
    ipc.friction_iterations = iterations
    ipc.friction_max_iterations = safety_cap
    ipc.friction_tolerance = tolerance
    ipc.pending_adjoint_seed = object()
    ipc.mpm = SimpleNamespace(grid_disp=object())

    events = []
    residuals = iter(residuals)

    def begin(grid_disp):
        events.append(("begin", grid_disp))

    def inner(_verbose):
        events.append("inner")
        return 2, 0.25, 3

    def refresh(grid_disp):
        events.append(("refresh", grid_disp))

    def updated_residual():
        events.append("updated_residual")
        return next(residuals)

    ipc.begin_friction_step = begin
    ipc.solve_frozen_friction_newton = inner
    ipc._backup_lagged_friction_for_adjoint = lambda: events.append("backup")
    ipc.refresh_friction_cache = refresh
    ipc.updated_friction_system_residual = updated_residual
    return ipc, events


@pytest.mark.parametrize("value", [1.5, np.inf, "bad"])
def test_friction_iteration_validation_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="friction_iterations"):
        _normalize_friction_iterations(value)


def test_friction_iteration_validation_accepts_reference_modes():
    assert _normalize_friction_iterations(1) == 1
    assert _normalize_friction_iterations(4.0) == 4
    assert _normalize_friction_iterations(-1) == -1
    assert _normalize_friction_iterations(0) == -1
    assert _normalize_friction_iterations(-7) == -1
    assert _positive_integer(50, "cap") == 50
    assert _positive_float(1.0e-8, "tolerance") == pytest.approx(1.0e-8)
    with pytest.raises(ValueError, match="cap"):
        _positive_integer(0, "cap")
    with pytest.raises(ValueError, match="tolerance"):
        _positive_float(0.0, "tolerance")


def test_begin_step_freezes_hat_x_once_and_cache_refresh_does_not_touch_it():
    ipc = object.__new__(IPCMPM)
    ipc.activate_fric = True
    marker = object()
    ipc.mpm = SimpleNamespace(grid_disp=marker)
    ipc.pfrictionNum = _ScalarField(2)
    ipc.gfrictionNum = _ScalarField(3)
    ipc.adjoint_friction_valid = _ScalarField(1)
    events = []
    ipc.initialize_hat_x = lambda: events.append("hat_x")
    ipc.update_particle_pos = lambda grid_disp: events.append(("particle_position", grid_disp))
    ipc.particle_friction_initialize = lambda: events.append("particle_cache")
    ipc.ground_friction_initialize = lambda: events.append("ground_cache")

    assert ipc.begin_friction_step() == 5
    assert ipc.refresh_friction_cache() == 5
    assert events == [
        "hat_x",
        ("particle_position", marker),
        "particle_cache",
        "ground_cache",
        ("particle_position", marker),
        "particle_cache",
        "ground_cache",
    ]
    assert ipc.friction_reference_initialized
    assert ipc.adjoint_friction_valid[None] == 0


def test_default_one_iteration_keeps_one_inner_solve_but_checks_updated_cache():
    ipc, events = _controller(1, residuals=[10.0])

    iter_num, inner_residual = ipc.solve_lagged_friction_fixed_point(False)

    assert iter_num == 2
    assert inner_residual == pytest.approx(0.25)
    assert ipc.last_newton_iterations == 3
    assert ipc.last_friction_iterations == 1
    assert ipc.last_friction_residual == pytest.approx(10.0)
    assert not ipc.last_friction_converged
    assert events == [
        ("begin", ipc.mpm.grid_disp),
        "inner",
        "backup",
        ("refresh", ipc.mpm.grid_disp),
        "updated_residual",
    ]


def test_regular_forward_step_skips_adjoint_cache_copy():
    ipc, events = _controller(1, residuals=[0.0])
    ipc.pending_adjoint_seed = None

    ipc.solve_lagged_friction_fixed_point(False)

    assert "backup" not in events


def test_positive_iteration_count_stops_on_updated_system_residual():
    ipc, events = _controller(4, residuals=[0.5, 1.0e-4, 0.0])

    iter_num, _ = ipc.solve_lagged_friction_fixed_point(False)

    assert iter_num == 4
    assert ipc.last_newton_iterations == 6
    assert ipc.last_friction_iterations == 2
    assert ipc.last_friction_residual == pytest.approx(1.0e-4)
    assert ipc.last_friction_converged
    assert events.count("inner") == 2
    assert events.count("backup") == 2
    assert events.count("updated_residual") == 2
    assert sum(event == ("refresh", ipc.mpm.grid_disp) for event in events) == 2


def test_minus_one_iteration_mode_obeys_safety_cap():
    ipc, events = _controller(
        -1,
        residuals=[1.0, 1.0, 1.0, 1.0],
        tolerance=1.0e-6,
        safety_cap=4,
    )

    with pytest.raises(RuntimeError, match="did not converge within safety cap"):
        ipc.solve_lagged_friction_fixed_point(False)

    assert ipc.last_friction_iterations == 4
    assert ipc.last_newton_iterations == 12
    assert not ipc.last_friction_converged
    assert events.count("inner") == 4
    assert events.count("backup") == 4
    assert events.count("updated_residual") == 4


def test_nonconverged_inner_newton_fails_before_cache_refresh():
    ipc, events = _controller(3, residuals=[0.0])

    def failed_inner(_verbose):
        events.append("inner_failed")
        ipc.last_inner_converged = False
        ipc.last_inner_failure_reason = "line_search_failed"
        return 1, 2.5, 1

    ipc.solve_frozen_friction_newton = failed_inner
    with pytest.raises(RuntimeError, match="inner Newton solve did not converge"):
        ipc.solve_lagged_friction_fixed_point(False)

    assert events == [("begin", ipc.mpm.grid_disp), "inner_failed"]
    assert ipc.last_friction_iterations == 0


def test_updated_residual_is_an_unapplied_newton_correction_norm():
    ipc = object.__new__(IPCMPM)
    ipc.mpm = SimpleNamespace(dt=0.25)
    ipc.solve_current_system = lambda: np.asarray([-0.2, 0.1, 0.0])

    assert ipc.updated_friction_system_residual() == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("friction_mode", "activate_fric", "expected_solver", "expected_backend"),
    [
        ("lagged", True, "PCG", "taichi_cuda_monolithic_pcg"),
        ("lagged", False, "PCG", "taichi_cuda_monolithic_pcg"),
        (
            "fully_implicit",
            True,
            "BiCGSTAB",
            "taichi_cuda_monolithic_bicgstab",
        ),
        (
            "fully_implicit",
            False,
            "BiCGSTAB",
            "taichi_cuda_monolithic_bicgstab",
        ),
    ],
)
def test_cuda_monolithic_krylov_routing_preserves_matrix_form(
    friction_mode, activate_fric, expected_solver, expected_backend
):
    import src.mpm.config as config

    ipc = object.__new__(IPCMPM)
    ipc.cuda_monolithic_solver = True
    ipc.friction_mode = friction_mode
    ipc.activate_fric = activate_fric
    ipc.mpm = SimpleNamespace(
        active_dof=6,
        rhs=object(),
        incre_resolution=object(),
        linear_solver_tolerance=1.0e-11,
        linear_solver_max_iters=123,
    )

    matrix = SimpleNamespace(solver="unset")

    def solve_flat_system(rhs, solution, **kwargs):
        assert matrix.solver == expected_solver
        assert rhs is ipc.mpm.rhs
        assert solution is ipc.mpm.incre_resolution
        assert kwargs == {
            "active_nodes": 6 // config.DIM,
            "tol": 1.0e-11,
            "maxiter": 123,
            "return_solution": False,
        }
        return {
            "converged": True,
            "iterations": 3,
            "residual": 1.0e-13,
            "solution_inf_norm": 0.25,
        }

    matrix.solve_flat_system = solve_flat_system
    ipc.assemble_current_system = lambda _grid_disp=None: matrix

    result = ipc.solve_current_system()

    assert result["backend"] == expected_backend
    assert matrix.solver == expected_solver


def test_trial_at_or_below_dmin_has_infinite_energy_instead_of_disappearing():
    ipc = object.__new__(IPCMPM)
    marker = object()
    ipc.mpm = SimpleNamespace(
        energy={},
        damping=0.0,
        integration=(),
        gravity=np.zeros(3),
        grid_disp=marker,
        neumann=SimpleNamespace(num=0),
        get_material_energy=lambda _grid_disp: None,
        get_inertia_energy=lambda *_args: None,
    )
    ipc.update_particle_pos = lambda grid_disp: None
    ipc.rebuild_barrier_contacts = lambda: False

    assert np.isinf(ipc.total_energy(marker))


def test_armijo_reuses_the_accepted_trial_energy(monkeypatch):
    import src.mpm.soft_particle.IPCMPM as ipc_module

    ipc = object.__new__(IPCMPM)
    base = object()
    trial = object()
    ipc.mpm = SimpleNamespace(
        grid_disp=base,
        grid_disp_temp=trial,
        incre_resolution=object(),
        rhs=object(),
        do_line_search=True,
        calc_g0=lambda _active_dof: 1.0,
        update_grid_disp=lambda _active_dof, _alpha: None,
    )
    ipc.line_search_descent_fallback = False
    ipc.line_search_work_tol = 0.0
    ipc.line_search_energy_atol = 0.0
    ipc.line_search_energy_rtol = 0.0
    ipc.line_search_armijo = 1.0e-4
    ipc.line_search_min_alpha = 1.0e-12
    ipc.line_search_max_backtracks = 8
    ipc.line_search_stagnation_alpha = 1.0e-12
    ipc.line_search_stagnation_energy_tol = 0.0
    ipc.ccd = lambda _active_dof: 1.0
    evaluations = []

    def total_energy(displacement):
        evaluations.append(displacement)
        return 10.0 if displacement is base else 9.0

    ipc.total_energy = total_energy
    monkeypatch.setattr(ipc_module, "copy_field", lambda *_args: None)

    assert ipc.line_search(1, verbose=False)
    assert evaluations == [base, trial]
    assert ipc.line_search_last_accepted_energy == pytest.approx(9.0)


@pytest.mark.parametrize("solver_type", [IPCULMPM, IPCTLMPM])
def test_ul_and_tl_substeps_delegate_to_shared_fixed_point_solver(solver_type):
    # Call the ordinary Python substep function with a lightweight receiver;
    # this avoids Taichi's data-oriented attribute wrapper treating mocks as
    # kernels while still exercising each class's actual orchestration code.
    solver = SimpleNamespace()
    solver.mpm = MagicMock()
    solver.mpm.integration = [0.5, 0.25, 0.5]
    solver.mpm.coeffPIC = 0.0
    solver.mpm.dt = 1.0e-3
    solver.mpm.set_active_dof.return_value = 6
    solver.ipc = MagicMock()
    solver.ipc.solve_friction_step.return_value = (0, 1.0e-8)
    events = []
    solver.ipc.solve_friction_step.side_effect = lambda _verbose: (events.append("solve") or (0, 1.0e-8))
    solver.ipc.differentiate_before_commit.side_effect = lambda: events.append("adjoint")
    solver.mpm.update_nodal_acc.side_effect = lambda _integration: events.append("commit")

    solver_type.substep(solver, verbose=False)

    solver.ipc.solve_friction_step.assert_called_once_with(False)
    solver.ipc.differentiate_before_commit.assert_called_once_with()
    solver.mpm.update_nodal_acc.assert_called_once_with(solver.mpm.integration)
    solver.mpm.advent_particles.assert_called_once_with(solver.mpm.coeffPIC)
    solver.ipc.ground.move.assert_called_once_with(solver.mpm.dt)
    assert events == ["solve", "adjoint", "commit"]


def test_direct_ipc_mpm_failed_adjoint_restores_pre_solve_displacement():
    ipc = object.__new__(IPCMPM)
    ipc.pending_adjoint_seed = object()
    ipc.pending_adjoint_mode = "elastic"
    ipc.last_elastic_differentiation = None
    ipc.device_grid_disp_snapshot = object()
    events = []
    ipc.differentiate_elastic_parameters = lambda _seed: (_ for _ in ()).throw(RuntimeError("adjoint failed"))
    ipc._restore_grid_displacement = lambda snapshot: events.append(snapshot)

    with pytest.raises(RuntimeError, match="adjoint failed"):
        ipc.differentiate_before_commit()

    assert events == [ipc.device_grid_disp_snapshot]


def test_direct_ipc_mpm_dispatches_plastic_equilibrium_before_commit():
    ipc = object.__new__(IPCMPM)
    seed = object()
    expected = {"plastic_history": "input_vjp"}
    ipc.pending_adjoint_seed = seed
    ipc.pending_adjoint_mode = "plastic"
    ipc.last_elastic_differentiation = None
    ipc.differentiate_plastic_equilibrium_parameters = lambda actual: expected if actual is seed else None

    ipc.differentiate_before_commit()

    assert ipc.last_elastic_differentiation is expected


def test_direct_ipc_mpm_dispatches_accepted_plastic_state_before_commit():
    ipc = object.__new__(IPCMPM)
    seed = object()
    state_vjp = object()
    expected = {"plastic_history": "accepted_step_input_vjp"}
    ipc.pending_adjoint_seed = seed
    ipc.pending_adjoint_mode = "plastic_state"
    ipc.pending_plastic_state_vjp = state_vjp
    ipc.last_elastic_differentiation = None
    ipc.differentiate_plastic_step_parameters = lambda actual_seed, actual_state: (
        expected if actual_seed is seed and actual_state is state_vjp else None
    )

    ipc.differentiate_before_commit()

    assert ipc.last_elastic_differentiation is expected


def test_direct_ipc_mpm_dispatches_device_particle_pullback_before_commit():
    ipc = object.__new__(IPCMPM)
    ipc.pending_adjoint_seed = object()
    ipc.pending_adjoint_mode = "plastic_particle_device"
    ipc.last_elastic_differentiation = None
    events = []
    ipc.pullback_plastic_particle_step_device = lambda: events.append("pullback")

    ipc.differentiate_before_commit()

    assert events == ["pullback"]
    assert ipc.last_elastic_differentiation is True
