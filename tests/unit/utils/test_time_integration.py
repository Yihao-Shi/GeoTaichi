"""Three-parameter kinematics against analytical linear dynamics."""

import numpy as np
import pytest

from src.utils.TimeIntegration import newmark_parameters
from src.fem.engines.FEMState import FEMState


@pytest.mark.parametrize("values", [(0.25, 0.5), (1, 2, 3, 4), (0, 0.25, 0.5), (0.5, np.nan, 0.5), "123"])
def test_newmark_requires_exactly_three_finite_positive_parameters(values):
    with pytest.raises(ValueError, match="alpha, beta, gamma"):
        newmark_parameters(values)


@pytest.mark.parametrize("integration", [(1.0, 0.5, 1.0), (0.5, 0.25, 0.5), (0.8, 0.3, 0.6)])
def test_fem_newmark_kinematics_and_equilibrium_use_alpha(taichi_runtime, integration):
    dt, mass, stiffness = 0.08, 2.0, 7.0
    position = np.array([[0.7, -0.2, 0.1]])
    velocity = np.array([[0.3, 0.1, -0.4]])
    acceleration = -stiffness * position / mass
    state = FEMState(position, velocity, np.array([mass]))
    state.acceleration.from_numpy(acceleration)
    state.save_step_state()
    alpha, beta, gamma = integration
    prediction = position + dt * velocity + alpha * (1 - 2 * beta) * dt**2 * acceleration
    scale = 2 * alpha * beta * dt**2
    endpoint = mass * prediction / (mass + scale * stiffness)
    expected_acceleration = (endpoint - prediction) / scale
    expected_velocity = velocity + dt * ((1 - gamma) * acceleration + gamma * expected_acceleration)
    state.build_newmark_prediction(dt, integration)
    np.testing.assert_allclose(state.predicted_position.to_numpy(), prediction, atol=1e-14)
    state.position.from_numpy(endpoint)
    force = taichi_runtime.Vector.field(3, taichi_runtime.f64, shape=1)
    force.from_numpy(stiffness * endpoint)
    state.assemble_implicit_residual(force, 0.0, dt, integration, 0)
    np.testing.assert_allclose(state.residual.to_numpy(), 0, atol=1e-12)
    state.finalize_newmark(dt, integration, 0)
    np.testing.assert_allclose(state.acceleration.to_numpy(), expected_acceleration, atol=1e-13)
    np.testing.assert_allclose(state.velocity.to_numpy(), expected_velocity, atol=1e-14)
    if integration == (0.5, 0.25, 0.5):
        before = 0.5 * mass * np.sum(velocity**2) + 0.5 * stiffness * np.sum(position**2)
        after = 0.5 * mass * np.sum(expected_velocity**2) + 0.5 * stiffness * np.sum(endpoint**2)
        assert after == pytest.approx(before, rel=1e-13)
