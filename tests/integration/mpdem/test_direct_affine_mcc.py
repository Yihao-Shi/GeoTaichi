"""Coupled physical MCC: hardening, friction and history are differentiated."""

import numpy as np
import pytest

from tests.integration.mpdem.test_direct_affine_nonassociated import _case, runtime
from tests.integration.mpdem.test_direct_affine_trajectory_gradients import _fields, _signature

pytestmark = [
    pytest.mark.integration,
    pytest.mark.mpdem,
    pytest.mark.ipc,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(3),
]


@pytest.mark.parametrize("mu", [0.0, 0.2])
def test_mcc_three_step_state_hardening_and_friction_vjp(tmp_path, mu, monkeypatch):
    coupling = _case(tmp_path, material="ModifiedCamClay")
    system, mpm = coupling.enginer, coupling.enginer.mpm
    assert not mpm.has_lagged_material
    assert system.matrix.solver == "BiCGSTAB" and not system.matrix.matrix_symmetric
    system.mixed.pair_friction.fill(mu)
    mpm.F0.from_numpy(np.array([np.diag(np.exp([0.01, -0.02, -0.04]))]))
    trajectory = coupling.differentiable(steps=3)
    fields = _fields(trajectory)
    initial = {name: field.to_numpy().copy() for name, field in fields.items()}
    rng = np.random.default_rng(835)
    seeds = {name: 0.1 * rng.normal(size=value.shape) for name, value in initial.items()}
    seeds["acceleration"] *= 1e-6
    seeds["preconsolidation_pressure"] *= 1e-3
    signatures = []
    for _ in range(3):
        step = trajectory.step(verbose=False)
        assert step["friction_converged"]
        assert "material_lagged_iterations" not in step
        signatures.append(_signature(trajectory, step))
    assert mpm.material.preconsolidation_pressure[0] > initial["preconsolidation_pressure"][0] + 1e-3
    terminal = {name: field.to_numpy().copy() for name, field in fields.items()}
    result = trajectory.backward(seeds)
    assert result["material_parameters"].shape == (5,)
    assert all(key in result for key in ("stress_ratio", "lambda_star", "kappa_star", "initial_pressure"))
    assert "young_modulus" not in result and "yield_stress" not in result
    for name, field in fields.items():
        np.testing.assert_array_equal(field.to_numpy(), terminal[name])
    directions = {
        name: rng.normal(size=value.shape) * (0.02 if name != "preconsolidation_pressure" else 10.0)
        for name, value in initial.items()
    }
    analytic = 0.0
    for name, direction in directions.items():
        key = "affine_initial_" + name[7:] if name.startswith("affine_") else "initial_" + name
        analytic += np.sum(result[key] * direction)

    def objective(shift, mu_shift=0.0):
        for name, field in fields.items():
            field.from_numpy(initial[name] + shift * directions[name])
        system.mixed.pair_friction.fill(mu + mu_shift)
        mpm.time, mpm.step_count = 0.0, 0
        for record in range(3):
            step = trajectory._advance(False)
            assert step["friction_converged"]
            assert _signature(trajectory, step) == signatures[record]
        return sum(np.sum(seeds[name] * field.to_numpy()) for name, field in fields.items())

    for h in (1e-5, 2e-6):
        numerical = (objective(h) - objective(-h)) / (2 * h)
        assert analytic == pytest.approx(numerical, rel=3e-3, abs=2e-7), (h, analytic, numerical)
    if mu > 0.0:
        for h in (1e-5, 2e-6):
            numerical = (objective(0.0, h) - objective(0.0, -h)) / (2 * h)
            assert result["mixed_friction_coefficient"][0, 0] == pytest.approx(numerical, rel=3e-3, abs=2e-7)
    for name, field in fields.items():
        field.from_numpy(terminal[name])
    system.mixed.pair_friction.fill(mu)
    mpm.time, mpm.step_count = 0.003, 3

    def fail(*args, **kwargs):
        raise RuntimeError("forced MCC backward failure")

    with monkeypatch.context() as patch:
        patch.setattr(system, "solve_adjoint_device", fail)
        with pytest.raises(RuntimeError, match="forced MCC"):
            trajectory.backward(seeds)
    for name, field in fields.items():
        np.testing.assert_array_equal(field.to_numpy(), terminal[name])
