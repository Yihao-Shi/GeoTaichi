from src.utils.SolverDiagnostics import (
    SolverDiagnosticsMixin,
    solver_diagnostics_snapshot,
)


class _Engine:
    time = 1.25
    step_count = 7
    dt = 0.05
    history = [{"step": 7, "residual": 1.0e-8}]
    last_failure = None


class _Simulation:
    time = 4.0


class _Facade(SolverDiagnosticsMixin):
    def __init__(self):
        self.enginer = _Engine()
        self.sims = _Simulation()


def test_common_solver_diagnostics_reads_progress_without_backend_imports():
    snapshot = solver_diagnostics_snapshot(_Facade())

    assert snapshot["schema_version"] == 1
    assert snapshot["solver"] == "_Facade"
    assert snapshot["engine"] == "_Engine"
    assert snapshot["state"] == {
        "time": 1.25,
        "step": 7,
        "timestep": 0.05,
        "target_time": 4.0,
    }
    assert snapshot["last_step"]["residual"] == 1.0e-8


def test_diagnostics_mixin_delegates_to_specialized_engine_snapshot():
    facade = _Facade()
    facade.enginer.diagnostics_snapshot = lambda: {
        "subsystem": "specialized",
        "last_failure": {"kind": "newton_nonconvergence"},
    }

    snapshot = facade.diagnostics_snapshot()

    assert snapshot["subsystem"] == "specialized"
    assert snapshot["solver"] == "_Facade"
    assert snapshot["engine"] == "_Engine"
    assert snapshot["schema_version"] == 1
