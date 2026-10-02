"""Unit checks for lagged-friction fixed-point orchestration."""

import numpy as np
import pytest

from src.igampm.engines import Engine


class _ArrayField:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64).copy()

    def to_numpy(self):
        return self.values.copy()

    def from_numpy(self, values):
        self.values = np.asarray(values, dtype=np.float64).copy()


def _mock_engine(iterations=3, tolerance=0.05, max_iterations=10):
    engine = object.__new__(Engine)
    engine.friction_mode = "lagged"
    engine.friction_iterations = iterations
    engine.friction_tolerance = tolerance
    engine.friction_max_iterations = max_iterations
    engine.activate_fric = False
    engine.iga = type(
        "IGA",
        (),
        {
            "grid_disp": _ArrayField([1.0, 2.0]),
            "grid_disp_temp": _ArrayField([1.0, 2.0]),
        },
    )()
    engine.mpm = type(
        "MPM",
        (),
        {
            "grid_disp": _ArrayField([3.0, 4.0]),
            "grid_disp_temp": _ArrayField([3.0, 4.0]),
        },
    )()
    engine.events = []
    engine.refresh_lagged_friction_cache = lambda grid_disp: engine.events.append(
        ("refresh", grid_disp)
    )
    engine.assemble_barrier_system = lambda: engine.events.append(("barrier",))
    engine.assemble_friction_system = lambda: engine.events.append(("friction",))

    def restore(iga_displacement, mpm_displacement):
        engine.iga.grid_disp.from_numpy(iga_displacement)
        engine.mpm.grid_disp.from_numpy(mpm_displacement)
        engine.iga.grid_disp_temp.from_numpy(iga_displacement)
        engine.mpm.grid_disp_temp.from_numpy(mpm_displacement)
        engine.events.append(("restore",))

    engine._restore_accepted_displacements = restore
    engine._test_external_correction = np.empty(0, dtype=np.float64)

    def load_external_correction(correction):
        engine._test_external_correction = np.asarray(
            correction, dtype=np.float64
        ).reshape(-1)
        return engine._test_external_correction

    engine._load_external_correction = load_external_correction
    engine._monolithic_correction_inf_norm = lambda active_dof: float(
        np.max(np.abs(engine._test_external_correction[:active_dof]))
    )
    return engine


def test_iga_lagged_outer_refreshes_only_between_complete_inner_solves():
    engine = _mock_engine()
    corrections = iter((0.5, 0.01))
    inner_refresh_counts = []

    def inner_solve(current, outer_iteration):
        inner_refresh_counts.append(
            sum(event[0] == "refresh" for event in current.events)
        )
        current.events.append(("inner", outer_iteration))
        return {"outer": outer_iteration}

    def updated_system(current):
        current.events.append(("probe",))
        return {"correction": np.array([next(corrections), 0.0])}

    result = engine.solve_lagged_friction_fixed_point(
        inner_solve, updated_system
    )

    assert inner_refresh_counts == [1, 2]
    assert [event[0] for event in engine.events] == [
        "refresh",
        "inner", "refresh", "probe",
        "inner", "refresh", "probe",
    ]
    assert result["iterations"] == 2
    assert result["residual"] == 0.01
    assert result["converged"] is True


def test_iga_lagged_outer_probe_uses_velocity_units_for_each_subsystem():
    engine = _mock_engine(iterations=1, tolerance=0.3)
    engine.iga.degree_of_freedom = 2
    engine.iga.dt = 0.5
    engine.mpm.dt = 0.25

    result = engine.solve_lagged_friction_fixed_point(
        lambda current, iteration: {"outer": iteration},
        lambda current: {
            # IGA: 0.1 / 0.5 = 0.2; MPM: 0.1 / 0.25 = 0.4.
            "correction": np.array([0.1, 0.0, 0.1, 0.0]),
        },
    )

    assert result["residual"] == pytest.approx(0.4)
    assert result["converged"] is False


def test_iga_unbounded_outer_mode_respects_safety_cap_and_never_applies_probe():
    engine = _mock_engine(iterations=-1, tolerance=0.0, max_iterations=3)
    iga_entry = engine.iga.grid_disp.to_numpy()
    mpm_entry = engine.mpm.grid_disp.to_numpy()

    def inner_solve(current, iteration):
        current.events.append(("inner", iteration))
        current.iga.grid_disp.from_numpy(np.array([9.0, 9.0]))
        current.mpm.grid_disp.from_numpy(np.array([8.0, 8.0]))

    with pytest.raises(RuntimeError, match="exhausted its safety cap"):
        engine.solve_lagged_friction_fixed_point(
            inner_solve,
            lambda current: {"correction": np.array([1.0])},
        )

    np.testing.assert_array_equal(engine.iga.grid_disp.to_numpy(), iga_entry)
    np.testing.assert_array_equal(engine.mpm.grid_disp.to_numpy(), mpm_entry)
    assert sum(event[0] == "inner" for event in engine.events) == 3
    assert engine.events[-1] == ("restore",)


def test_iga_builtin_lagged_driver_rejects_host_fallback():
    engine = _mock_engine(iterations=2, tolerance=0.05)
    engine.activate_fric = True
    engine._device_monolithic_available = lambda include_friction: False

    with pytest.raises(RuntimeError, match="no NumPy/SciPy fallback"):
        engine.solve_lagged_friction_fixed_point()


def test_iga_device_builtin_probe_keeps_full_correction_on_device():
    engine = _mock_engine(iterations=1, tolerance=0.05)
    engine.activate_fric = True
    engine.monolithic_linear_solver_tolerance = 1.0e-10
    engine.monolithic_linear_solver_max_iters = 123
    engine._device_monolithic_available = lambda include_friction: True
    engine._save_device_monolithic_entry_displacements = lambda: None
    engine._restore_device_monolithic_entry_displacements = lambda: None
    engine.solve_monolithic_newton = lambda **kwargs: {
        "converged": True,
        "residual": 0.0,
        "backend": "taichi_device_lagged_pcg",
    }

    class DeviceCorrection:
        def __array__(self, *_args, **_kwargs):
                raise AssertionError("device convergence probe copied correction to NumPy")

    class DeviceMatrix:
        def solve_flat_system(self, rhs, correction, **kwargs):
            assert rhs == "device_rhs"
            assert isinstance(correction, DeviceCorrection)
            assert kwargs == {
                "active_nodes": 4,
                "tol": 1.0e-10,
                "maxiter": 123,
                "return_solution": False,
            }
            return {
                "converged": True,
                "residual": 1.0e-12,
                "iterations": 3,
                "solution_inf_norm": 0.01,
            }

    engine.assemble_monolithic_newton_system = (
        lambda include_friction: {
            "matrix": DeviceMatrix(),
            "rhs": "device_rhs",
            "correction": DeviceCorrection(),
            "active_nodes": 4,
        }
    )
    engine._solve_monolithic_linear_system = lambda system, linear_solve=None: (
        system["matrix"].solve_flat_system(
            system["rhs"],
            system["correction"],
            active_nodes=system["active_nodes"],
            tol=engine.monolithic_linear_solver_tolerance,
            maxiter=engine.monolithic_linear_solver_max_iters,
            return_solution=False,
        )
    )

    result = engine.solve_lagged_friction_fixed_point()

    assert result["converged"] is True
    assert result["residual"] == 0.01


def test_iga_builtin_inner_failure_is_hard_error_and_rolls_back():
    engine = _mock_engine(iterations=2, tolerance=0.05)
    engine.activate_fric = True
    engine._device_monolithic_available = lambda include_friction: True
    iga_entry = engine.iga.grid_disp.to_numpy()
    mpm_entry = engine.mpm.grid_disp.to_numpy()
    engine._save_device_monolithic_entry_displacements = lambda: None

    def restore_device_entry():
        engine._restore_accepted_displacements(iga_entry, mpm_entry)

    engine._restore_device_monolithic_entry_displacements = restore_device_entry

    def builtin_inner(**kwargs):
        engine.iga.grid_disp.from_numpy(np.array([11.0, 12.0]))
        engine.mpm.grid_disp.from_numpy(np.array([13.0, 14.0]))
        return {"converged": False, "residual": 2.5}

    engine.solve_monolithic_newton = builtin_inner
    engine.assemble_monolithic_newton_system = lambda **kwargs: pytest.fail(
        "the fixed-point probe must not run after a failed inner solve"
    )

    with pytest.raises(RuntimeError, match="Newton solve did not converge"):
        engine.solve_lagged_friction_fixed_point()

    np.testing.assert_array_equal(engine.iga.grid_disp.to_numpy(), iga_entry)
    np.testing.assert_array_equal(engine.mpm.grid_disp.to_numpy(), mpm_entry)
    assert engine.events[-1] == ("restore",)


def test_iga_finite_outer_budget_reports_explicit_approximation():
    engine = _mock_engine(iterations=2, tolerance=0.0)

    result = engine.solve_lagged_friction_fixed_point(
        lambda current, iteration: {"outer": iteration},
        lambda current: {"correction": np.array([1.0])},
    )

    assert result["iterations"] == 2
    assert result["converged"] is False
    assert result["approximate"] is True
