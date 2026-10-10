from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from src.fedem.AffineIPCEngine import FEMAffineIPCEngine
from src.fedem.contact.AffineIPC import AffineIPCModel
from src.fem.contact.ContactModel import FEMContact
from src.fem.engines.ImplicitFEM import ImplicitFEM, NewtonConvergenceError
from src.fempm.ImplicitEngine import FEMPMImplicitEngine
from src.fempm.contact.IPC import IPCModel

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact]


@pytest.mark.parametrize("kind", ["fem", "fempm", "fedem"])
def test_force_reference_survives_friction_refresh(kind):
    engine, _, _ = _mock_outer_solver(kind, (0.0,))
    engine.max_iterations = 0
    engine.absolute_tolerance = 0.0
    engine.residual_tolerance = 1e-4
    engine.raise_on_nonconvergence = False
    engine._friction_force_reference = None
    force = [1000.0]
    engine.dt = 0.001
    if kind == "fem":
        engine.contact_assembler = None
        engine.damping, engine.beta, engine.gamma, engine.quasi_static = 0.0, 0.25, 0.5, False
        engine._prepare_contact_iteration_device = lambda position: None
        engine._assemble_internal_device = lambda **kwargs: None
        engine._current_stiffness = None
        engine.state.assemble_implicit_residual = lambda *args: None
        engine.state.residual_norm = lambda: force[0]
        engine.state.external_norm = lambda: 0.0
        engine._contact_converged_device = lambda: True
        solve = lambda: ImplicitFEM._solve_frozen_friction_newton_device(engine, 0.1)
    else:
        engine.residual_squared = {None: 0.0}
        engine.contact.diagnostics = lambda: {}
        engine.contact.contact_converged = lambda: True
        engine.assemble_system = lambda **kwargs: {"active_dof": 3}
        if kind == "fempm":
            engine.inexact_newton = False
            engine.correction_velocity_tolerance = 1e-7
            engine.constraint_inf_norm = {None: 0.0}
            engine._reduce_system_metrics = lambda dof: engine.residual_squared.update({None: force[0] ** 2})
            solve = lambda: FEMPMImplicitEngine._solve_newton(engine, True, False)[:2]
        else:
            engine.simulation = SimpleNamespace(timer=SimpleNamespace(section=lambda name: nullcontext()))
            engine._reduce_metrics = lambda: engine.residual_squared.update({None: force[0] ** 2})
            solve = lambda: FEMAffineIPCEngine._solve_frozen_friction_newton(engine, False)
    assert not solve()[0]
    force[0] = 0.01
    assert solve()[0]
    assert engine._friction_force_reference == 1000.0


def test_nonassociated_fempm_updated_probe_also_requires_force_balance():
    engine, run, _ = _mock_outer_solver("fempm", (1e-12, 1e-12), maximum=2)
    engine.nonassociated_newton = True
    engine.absolute_tolerance, engine.residual_tolerance = 1e-10, 1e-8
    engine.dt, engine.correction_velocity_tolerance = 0.001, 1e-7
    engine.constraint_inf_norm = {None: 0.0}
    engine.residual_squared = {None: 1.0}
    engine.contact.contact_converged = lambda: True
    inner = engine._solve_newton

    def solve(include_friction, verbose):
        engine._friction_force_reference = 1.0
        return inner(include_friction, verbose)

    engine._solve_newton = solve
    with pytest.raises(NewtonConvergenceError, match="fixed point did not converge"):
        run()
    assert not engine.last_friction_converged


@pytest.mark.parametrize("model_type", (IPCModel, AffineIPCModel))
def test_coupled_ipc_accepts_strict_friction_fixed_point_settings(model_type):
    model = model_type(
        SimpleNamespace(search="LinkedCell"),
        friction_iterations=-1,
        friction_max_iterations=17,
        friction_tolerance=2.0e-8,
    )

    assert model.friction_iterations == -1
    assert model.friction_max_iterations == 17
    assert model.friction_tolerance == pytest.approx(2.0e-8)


def test_fem_contact_accepts_strict_friction_fixed_point_settings():
    contact = FEMContact.create(
        "IPC",
        friction_iterations=-1,
        friction_max_iterations=17,
        friction_tolerance=2.0e-8,
    )

    assert contact.friction_iterations == -1
    assert contact.friction_max_iterations == 17
    assert contact.friction_tolerance == pytest.approx(2.0e-8)


def test_fem_pair_contact_inherits_fixed_point_settings():
    contact = FEMContact.create(
        "IPC",
        friction_iterations=-1,
        friction_max_iterations=17,
        friction_tolerance=2.0e-8,
    )

    pair = contact.add_property(0, 1, {"friction_coefficient": 0.4})

    assert pair.friction_iterations == -1
    assert pair.friction_max_iterations == 17
    assert pair.friction_tolerance == pytest.approx(2.0e-8)


@pytest.mark.parametrize("value", (0, -2, 1.5, float("nan"), "many"))
def test_fem_contact_rejects_invalid_friction_iterations(value):
    with pytest.raises(ValueError, match="friction_iterations"):
        FEMContact.create("IPC", friction_iterations=value)


@pytest.mark.parametrize("model_type", (IPCModel, AffineIPCModel))
@pytest.mark.parametrize("value", (0, -2, 1.5, float("nan"), "many"))
def test_coupled_ipc_rejects_invalid_friction_iterations(model_type, value):
    with pytest.raises(ValueError, match="friction_iterations"):
        model_type(SimpleNamespace(search="LinkedCell"), friction_iterations=value)


def _mock_outer_solver(kind, residuals, *, iterations=-1, maximum=4, tolerance=0.05):
    settings = SimpleNamespace(
        friction_iterations=iterations,
        friction_max_iterations=maximum,
        friction_tolerance=tolerance,
    )
    values = iter(residuals)
    events = []

    if kind == "fem":
        engine = object.__new__(ImplicitFEM)
        engine.state = SimpleNamespace(position="fem_position")
        engine.contact_assembler = SimpleNamespace(
            is_ipc=True,
            activate_friction=True,
            contact=settings,
            refresh_friction_device=lambda position: events.append(("refresh", position)),
        )
        engine._solve_frozen_friction_newton_device = lambda next_time: (
            events.append(("solve", next_time)) or True,
            [{"inner": True}],
        )
        engine._updated_friction_residual_device = lambda: (events.append(("probe", None)) or next(values))
        run = lambda: engine.solve_lagged_friction_fixed_point(0.1)
    elif kind == "fempm":
        engine = object.__new__(FEMPMImplicitEngine)
        engine.nonassociated_newton = False
        engine.contact_model = settings
        engine.fem = SimpleNamespace(state=SimpleNamespace(position="fem_position"))
        engine.mpm = SimpleNamespace(
            grid_disp="mpm_displacement", has_lagged_material=False, begin_lagged_material_state=lambda: None
        )
        engine.contact = SimpleNamespace(
            activate_friction=True,
            refresh_friction=lambda fem, mpm: events.append(("refresh", (fem, mpm))),
        )
        engine._solve_newton = lambda include_friction, verbose: (
            events.append(("solve", include_friction)) or True,
            [{"inner": True}],
            {"system": True},
        )
        engine._updated_friction_residual = lambda include_friction: (
            events.append(("probe", include_friction)) or next(values)
        )
        run = lambda: engine.solve_lagged_friction_fixed_point(False)[:2]
    else:
        engine = object.__new__(FEMAffineIPCEngine)
        engine.contact_model = settings
        engine.raise_on_nonconvergence = True
        engine.fem_contact = None
        engine.affine = SimpleNamespace(
            initialize_contact_damping_device=lambda: None, device_backup_lagged_friction_for_adjoint=lambda: None
        )
        engine.fem = SimpleNamespace(state=SimpleNamespace(position="fem_position"))
        engine.contact = SimpleNamespace(
            activate_friction=True,
            backup_lagged_friction_for_adjoint_device=lambda: None,
            refresh_friction=lambda position: events.append(("refresh", position)),
        )
        engine._solve_frozen_friction_newton = lambda verbose: (
            events.append(("solve", verbose)) or True,
            [{"residual_norm": 0.0, "convergence_tolerance": 1.0}],
        )
        engine._updated_friction_residual = lambda: (events.append(("probe", None)) or next(values))
        run = lambda: engine.solve_lagged_friction_fixed_point(0.1, False)

    return engine, run, events


@pytest.mark.parametrize("kind", ("fem", "fempm", "fedem"))
def test_fem_ipc_paths_refresh_then_probe_until_fixed_point_converges(kind):
    engine, run, events = _mock_outer_solver(kind, (0.2, 0.01))

    converged, records = run()

    assert converged
    assert len(records) == 2
    assert [event[0] for event in events] == ["solve", "refresh", "probe", "solve", "refresh", "probe"]
    assert engine.last_friction_iterations == 2
    assert engine.last_friction_residual == pytest.approx(0.01)
    assert engine.last_friction_converged


@pytest.mark.parametrize("kind", ("fem", "fempm", "fedem"))
def test_fem_ipc_strict_mode_rejects_unconverged_fixed_point(kind):
    _, run, events = _mock_outer_solver(kind, (0.2, 0.1), maximum=2)

    with pytest.raises(NewtonConvergenceError, match="fixed point did not converge"):
        run()

    assert [event[0] for event in events] == ["solve", "refresh", "probe", "solve", "refresh", "probe"]


@pytest.mark.parametrize("kind", ("fem", "fempm", "fedem"))
def test_fem_ipc_fixed_count_reports_failed_probe_without_claiming_convergence(kind):
    engine, run, _ = _mock_outer_solver(kind, (0.2,), iterations=1)

    converged, _ = run()

    assert converged
    assert not engine.last_friction_converged
    assert engine.last_friction_residual == pytest.approx(0.2)


def test_fem_affine_newton_applies_first_nonzero_small_correction():
    engine = object.__new__(FEMAffineIPCEngine)
    engine.max_iterations = 2
    engine.absolute_tolerance = 0.0
    engine.residual_tolerance = 0.0
    engine.correction_velocity_tolerance = 1.0e-3
    engine.dt = 1.0e-2
    engine.last_line_search_limits = None
    engine.residual_squared = {None: 1.0}
    engine.physical_correction_inf_norm = {None: 5.0e-6}
    engine.contact = SimpleNamespace(diagnostics=lambda: {}, contact_converged=lambda: True)
    engine.simulation = SimpleNamespace(timer=SimpleNamespace(section=lambda _name: nullcontext()))
    engine.assemble_system = lambda need_matrix: {"energy": 1.0}
    engine._reduce_metrics = lambda: None
    engine._solve_linear_system = lambda _system: None
    engine._split_direction = lambda: None
    engine._reduce_physical_correction = lambda: None
    applied = []

    def line_search(_energy):
        applied.append(True)
        engine.physical_correction_inf_norm[None] = 0.0
        return 1.0, 0, 0.0

    engine._line_search = line_search
    engine.fem = SimpleNamespace(apply_boundary_step=lambda _dt, _step: None)

    converged, records = engine._solve_frozen_friction_newton()

    assert converged
    assert applied == [True]
    assert len(records) == 2
    assert records[-1]["convergence_reason"] == "physical_correction_velocity"
