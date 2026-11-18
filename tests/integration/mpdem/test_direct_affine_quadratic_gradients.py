"""MPM transfer curvature and constrained coupled adjoints with friction."""

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


def test_quadratic_transfer_prescribed_displacement_and_be_vjp(tmp_path):
    coupling = _case(
        tmp_path,
        material="VonMises",
        shape_function="QuadBSpline",
        fixed_displacement=2e-6,
        mpm_newmark=(1.0, 0.5, 1.0),
    )
    trajectory = coupling.differentiable(steps=2)
    mpm = trajectory.mpm
    fields = _fields(trajectory)
    initial = {name: field.to_numpy().copy() for name, field in fields.items()}
    assert mpm.shape_func.max_node_per_particle == 27
    assert mpm.dirichlet.num == 1
    assert mpm.node2dof[5 + 11 * (5 + 11 * 3)] > 0
    np.testing.assert_array_equal(mpm.integration, [1.0, 0.5, 1.0])
    rng = np.random.default_rng(221)
    seeds = {name: 0.1 * rng.normal(size=value.shape) for name, value in initial.items()}
    seeds["acceleration"] *= 1e-6
    directions = {name: 0.02 * rng.normal(size=value.shape) for name, value in initial.items()}
    signatures = []
    for _ in range(2):
        result = trajectory.step(verbose=False)
        assert result["friction_converged"]
        signatures.append(_signature(trajectory, result))
    gradient = trajectory.backward(seeds)
    analytic = 0.0
    for name, direction in directions.items():
        key = "affine_initial_" + name[7:] if name.startswith("affine_") else "initial_" + name
        analytic += np.sum(gradient[key] * direction)

    def objective(shift):
        for name, field in fields.items():
            field.from_numpy(initial[name] + shift * directions[name])
        mpm.time, mpm.step_count = 0.0, 0
        for record in range(2):
            result = trajectory._advance(False)
            assert _signature(trajectory, result) == signatures[record]
        return sum(np.sum(seeds[name] * field.to_numpy()) for name, field in fields.items())

    for step in (1e-5, 2e-6):
        numerical = (objective(step) - objective(-step)) / (2 * step)
        assert analytic == pytest.approx(numerical, rel=3e-3, abs=2e-7), (step, analytic, numerical)
