from types import SimpleNamespace

from src.dem.DEMBase import AffineBodySolver, Solver
from src.utils.SolverRuntime import python_callback


class _Timer:
    def begin(self, _name):
        pass

    def end(self, _name):
        pass


def test_preintegration_callback_runs_after_contact_and_before_integration(monkeypatch):
    monkeypatch.setattr(
        "src.dem.DEMBase.normalize_callbacks",
        lambda functions, _transform: (functions,),
    )
    events = []

    class _Engine:
        def reset_wall_message(self, _scene):
            pass

        def reset_particle_message(self, _scene):
            pass

        def reset_contact_energy(self):
            pass

        def update_neighbor_lists(self, _sims, _scene, _neighbor):
            events.append("contact")

        def integration(self, _sims, _scene, _neighbor):
            events.append("integration")

        def adaptive_timestep(self, _sims, _scene):
            pass

    sims = SimpleNamespace(timer=_Timer())
    contact = SimpleNamespace(neighbor=object())
    solver = Solver(sims, None, contact, _Engine(), None)
    solver.set_preintegration_callback_function(lambda: events.append("diagnostic"))

    solver.core(object())

    assert events == ["contact", "diagnostic", "integration"]


def test_clearing_preintegration_callback_rebinds_no_operation():
    sims = SimpleNamespace(timer=_Timer())
    contact = SimpleNamespace(neighbor=object())
    solver = Solver(sims, None, contact, object(), None)
    solver.set_preintegration_callback_function(lambda: None)

    solver.clear_preintegration_callback_functions()

    assert solver.preintegration == []
    assert solver.run_preintegration_callbacks() is None


def test_python_preintegration_callback_bypasses_taichi_kernel_wrapping():
    callback = python_callback(lambda: None)
    solver = Solver.__new__(Solver)
    solver.preintegration = []

    solver.set_preintegration_callback_function(callback)

    assert solver.preintegration == [callback]


def test_affine_body_postprocess_callback_remains_a_python_callable():
    callback = python_callback(lambda: None)
    solver = AffineBodySolver(None, None, None, None, None)

    solver.set_callback_function(callback)

    assert solver.postprocess == [callback]


def test_servo_python_callback_bypasses_taichi_kernel_wrapping():
    from src.dem.engines.ExplicitEngine import ExplicitEngine

    engine = ExplicitEngine.__new__(ExplicitEngine)
    callback = python_callback(lambda: None)
    sims = SimpleNamespace(
        max_servo_wall_num=1,
        servo_status="On",
        servo_type="GainControl",
        scheme="DEM",
        max_particle_num=0,
    )

    engine.set_servo_mechanism(sims, callback)

    assert engine.callback is callback


def test_soft_affine_solver_rejects_an_undefined_preintegration_stage():
    import pytest

    from src.mpdem.engines.SoftAffineIPCBase import SoftAffineIPCSolver

    solver = SoftAffineIPCSolver.__new__(SoftAffineIPCSolver)
    solver.postprocess = []

    solver.set_preintegration_callback_function(None)
    solver.clear_preintegration_callback_functions()
    with pytest.raises(ValueError, match="does not expose a state"):
        solver.set_preintegration_callback_function(lambda: None)
