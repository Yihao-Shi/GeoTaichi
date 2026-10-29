"""Check the coupled forcing and pressure response against a direct 2x2 solve."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import _drag_rates_props, _implicit_phase_accelerations


@pytest.mark.parametrize("dimension", [2, 3])
def test_forced_drag_and_pressure_use_the_same_inverse(taichi_runtime, dimension):
    state = ti.Vector.field(dimension, ti.f64, shape=3)
    result = ti.Vector.field(dimension, ti.f64, shape=2)
    rates = ti.field(ti.f64, shape=2)
    phi, rho_s, rho_f = 0.25, 2650.0, 1000.0
    mass_s, mass_f = (1.0 - phi) * rho_s, phi * rho_f

    @ti.kernel
    def solve(timestep: ti.f64, model: ti.f64):
        a, b = _drag_rates_props(state[0], phi, rho_s, rho_f, 0.001, 0.0005, 0.001, 9800.0, model)
        rates[0], rates[1] = a, b
        result[0], result[1] = _implicit_phase_accelerations(state[0], state[1], state[2], a, b, timestep)

    for model in [0.0, 1.0, 2.0]:
        for timestep in [0.0, 1e-5, 0.1]:
            for mode in ["rest_forcing", "moving_forcing", "gravity", "pressure"]:
                slip = np.zeros(dimension) if mode != "moving_forcing" else np.arange(1.0, dimension + 1.0) * 0.03
                force_s, force_f = np.ones(dimension), -np.ones(dimension) * 3.0
                if mode == "gravity":
                    force_s = force_f = np.full(dimension, -9.81)
                elif mode == "pressure":
                    force_s = np.ones(dimension) / rho_s
                    force_f = np.ones(dimension) / rho_f
                state.from_numpy(np.array([slip, force_s, force_f]))
                solve(timestep, model)
                a, b = rates.to_numpy()
                matrix = np.array([[1.0 + timestep * a, -timestep * a], [-timestep * b, 1.0 + timestep * b]])
                expected = np.linalg.solve(matrix, np.array([force_s + a * slip, force_f - b * slip]))
                actual = result.to_numpy()
                np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-12)
                np.testing.assert_allclose(
                    mass_s * actual[0] + mass_f * actual[1], mass_s * force_s + mass_f * force_f, rtol=2e-11, atol=1e-9
                )
                if mode == "pressure":
                    mobility = (1.0 - phi) * actual[0] + phi * actual[1]
                    assert np.all(mobility > 0.0)

    # A steady forced slip must survive one backward-Euler step exactly.
    state.from_numpy(np.array([np.zeros(dimension), np.ones(dimension), np.zeros(dimension)]))
    solve(1e-5, 1.0)
    a, b = rates.to_numpy()
    steady_slip = np.full(dimension, -1.0 / (a + b))
    state.from_numpy(np.array([steady_slip, np.ones(dimension), np.zeros(dimension)]))
    solve(1e-5, 1.0)
    actual = result.to_numpy()
    np.testing.assert_allclose(actual[0], actual[1], rtol=1e-13, atol=1e-13)
