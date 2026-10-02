import ast
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np

from src.fem.engines.ExplicitFEM import ExplicitFEM
from src.fem.engines.FEMSolver import FEMSolver
from src.fem.engines.ImplicitFEM import ImplicitFEM


class _State:
    def __init__(self):
        self.calls = []

    def apply_boundary(self, *args):
        self.calls.append(("apply", args))

    def build_external_force(self):
        self.calls.append(("force", ()))

    def prepare_explicit_step(self, constrained):
        self.calls.append(("prepare", (constrained,)))


def _has_call(function, name):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
        for node in ast.walk(tree)
    )


def test_explicit_substep_uses_bound_operations_without_runtime_discovery():
    assert not _has_call(ExplicitFEM.substep, "getattr")
    source = inspect.getsource(ExplicitFEM.substep)
    assert "advance_constitutive_state()" in source
    assert "assemble_explicit_internal_force()" in source
    assert "resolve_soft_particle_contact_step" in source
    assert "prepare_explicit_step" in source


def test_static_boundary_binding_selects_fixed_device_operations():
    solver = object.__new__(FEMSolver)
    solver.state = _State()
    solver.boundary_data = SimpleNamespace(
        dirichlet_dynamic=False,
        force_dynamic=False,
    )
    solver._boundary_dof_count = 0

    solver.bind_boundary_functions()

    assert solver.set_boundary_data_step(3.0, 4) == 0
    solver.prepare_explicit_step(3.0, 4)
    solver.update_external_force_step(3.0, 4)
    solver.apply_boundary_step(1.0, 1)
    assert solver.state.calls == [("prepare", (0,)), ("force", ())]


def test_retry_boundary_binding_evaluates_the_exact_attempt_time():
    solver = object.__new__(FEMSolver)
    solver.boundary_data = SimpleNamespace(
        dirichlet_dynamic=True,
        force_dynamic=True,
    )
    solver._adaptive_boundary_evaluation = True
    solver._boundary_dof_count = 1
    solver._boundary_dofs = np.array([2], dtype=np.int32)
    solver.mesh = object()
    solver.is_axisymmetric = False
    solver.axis_offset = 0.0
    calls = []

    class State(_State):
        def set_boundary_data(self, dofs, values):
            calls.append(("dirichlet", dofs.copy(), values.copy()))

        def set_boundary_force(self, values):
            calls.append(("neumann", values.copy()))

    solver.state = State()
    solver._dirichlet_values = lambda time: (
        np.array([2], dtype=np.int32),
        np.array([time], dtype=np.float64),
    )
    solver.neumann = SimpleNamespace(force=lambda _mesh, time, **_kwargs: np.array([[time, 0.0, 0.0]]))

    solver.bind_boundary_functions()
    assert solver.set_boundary_data_step(0.375, 99) == 1
    solver.update_external_force_step(0.375, 99)

    np.testing.assert_array_equal(calls[0][1], [2])
    np.testing.assert_allclose(calls[0][2], [0.375])
    np.testing.assert_allclose(calls[1][1], [[0.375, 0.0, 0.0]])


def test_soft_particle_contact_is_bound_once_and_receives_step_policy():
    calls = []

    class _Contact:
        def resolve(self, dt, **kwargs):
            calls.append((dt, kwargs))

    solver = object.__new__(ExplicitFEM)
    solver.dt = 0.125
    solver.track_energy = False
    solver.soft_particle_contact = _Contact()
    solver._advance_constitutive_state = lambda: None
    solver._assemble_internal_force_device = lambda: "force"
    solver._assemble_internal_device = lambda **_kwargs: "fallback"

    solver.bind_runtime_functions()
    result = solver.resolve_soft_particle_contact_step(advance_history=False, check_rebuild=False)

    assert result is None
    assert calls == [
        (
            0.125,
            {"advance_history": False, "check_rebuild": False},
        )
    ]
    assert solver.assemble_explicit_internal_force() == "force"


def test_implicit_fem_keeps_last_step_but_schedules_long_term_history():
    source = inspect.getsource(ImplicitFEM._substep_once)
    assert "last_step_record" in source
    assert "step_schedule.append_history" in source
    assert "self.history.append" not in source
