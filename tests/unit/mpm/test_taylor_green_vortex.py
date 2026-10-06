import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.mpm.IncompressibleFluid.wavemaker_tank_3d.wavemaker_tank_3d import WATER_ORIGIN, WATER_SIZE
from examples.mpm.IncompressibleFluid.wavemaker_tank_3d.draw.evaluate_wavemaker_tank_3d import (
    harmonic_fit,
    linear_piston_wave,
    piston_penetration,
    sdf_surface_profile_metrics,
)
from examples.mmpm.TwoPhaseWavemaker3D.two_layer_two_phase_wavemaker_3d_parameters import (
    SLOPE_HEIGHT,
    SLOPE_RUN,
    SLOPE_TOE,
    TANK as TWO_PHASE_TANK,
    WATER_DEPTH as TWO_PHASE_WATER_DEPTH,
)
from examples.mmpm.TwoPhaseWavemaker3D.draw.evaluate_two_layer_two_phase_wavemaker_3d import evaluate_metrics
from examples.mpm.IncompressibleFluid.wavemaker_tank_3d.draw.evaluate_wavemaker_tank_3d import (
    sdf_mean_surface,
    sdf_surface_gauge,
)

SCRIPT = (
    Path(__file__).parents[3]
    / "examples/mpm/IncompressibleFluid/taylor_green_vortex_2d/draw/evaluate_taylor_green_vortex_2d.py"
)
SPEC = importlib.util.spec_from_file_location("taylor_green_vortex_2d", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_taylor_green_analytic_velocity_pressure_and_decay():
    points = np.array([[0.5 * np.pi, 0.0], [0.0, 0.5 * np.pi], [0.25 * np.pi, 0.25 * np.pi]])
    velocity = MODULE.analytical_velocity(points, time=0.0, viscosity=0.05)
    pressure = MODULE.analytical_pressure(points, time=0.0, viscosity=0.05)

    np.testing.assert_allclose(velocity[:2], [[1.0, 0.0], [0.0, -1.0]], atol=1.0e-15)
    np.testing.assert_allclose(pressure, [0.0, 0.0, 0.0], atol=1.0e-15)
    np.testing.assert_allclose(
        MODULE.analytical_velocity(points, time=1.0, viscosity=0.05),
        velocity * np.exp(-0.1),
        atol=1.0e-15,
    )
    assert (
        MODULE.analytical_pressure(np.array([[0.0, 0.0]]), time=0.0, viscosity=0.05, density=3.0, amplitude=2.0)[0]
        == 6.0
    )
    steady_time = MODULE.decay_time_to_fraction(0.05, 0.01)
    assert np.exp(-2.0 * 0.05 * steady_time) == pytest.approx(0.01)


def test_linear_wavemaker_reference_and_harmonic_fit():
    np.testing.assert_array_equal(WATER_ORIGIN, [0.08, 0.0, 0.0])
    np.testing.assert_allclose(WATER_SIZE, [2.32, 0.32, 0.36])
    frequency = 1.0
    wavenumber, amplitude, group_speed = linear_piston_wave(frequency, 0.36, 0.03)
    assert 9.81 * wavenumber * np.tanh(wavenumber * 0.36) == pytest.approx((2.0 * np.pi) ** 2)
    assert amplitude > 0.0 and group_speed > 0.0
    time = np.linspace(0.0, 2.0, 81)
    fitted_amplitude, mean, r2 = harmonic_fit(time, 0.007 * np.sin(2.0 * np.pi * time) + 0.36, frequency)
    assert fitted_amplitude == pytest.approx(0.007)
    assert mean == pytest.approx(0.36)
    assert r2 == pytest.approx(1.0)


def test_piston_penetration_uses_cut_cell_scale_tolerance():
    positions = np.array([[0.0995, 0.1, 0.1], [0.12, 0.1, 0.1]])
    assert piston_penetration(positions, 0.1, 0.1) == (0, pytest.approx(0.0005))
    assert piston_penetration(positions, 0.101, 0.1)[0] == 1


def test_two_layer_wave_gauge_uses_sdf_zero_crossing():
    cell_type = np.zeros((5, 5, 4), dtype=np.int32)
    cell_type[:, :, :2] = 1
    fluid_sdf = np.full((5, 5, 4), 0.015)
    fluid_sdf[:, :, 1] = -0.005
    assert np.isclose(sdf_surface_gauge(cell_type, fluid_sdf, 0.03, 0.06), 0.0525)
    assert np.isclose(sdf_mean_surface(cell_type, fluid_sdf, 0.03), 0.0525)
    assert sdf_surface_profile_metrics(cell_type, fluid_sdf, 0.03) == pytest.approx((0.0, 0.0, 0.0))


def test_two_phase_wavemaker_has_one_third_depth_trapezoid_and_long_platform():
    assert SLOPE_HEIGHT == pytest.approx(TWO_PHASE_WATER_DEPTH / 3.0)
    assert TWO_PHASE_TANK[0] - SLOPE_TOE - SLOPE_RUN >= 1.0


def test_two_phase_wavemaker_metrics_reject_loss_escape_and_missing_wave():
    args = SimpleNamespace(
        frequency=1.1,
        ramp_time=1.0,
        wave_velocity=0.18,
        time=3.0,
        dt=2.0e-4,
        dx=0.02,
        maximum_porosity=0.56,
        soil_young_modulus=12.0e6,
        soil_cohesion=300.0,
        soil_friction=26.0,
    )
    time = np.linspace(0.0, args.time, 61)
    _, amplitude, _ = linear_piston_wave(args.frequency, TWO_PHASE_WATER_DEPTH, args.wave_velocity)
    wave = amplitude * np.sin(2.0 * np.pi * args.frequency * time)
    rows = np.zeros((len(time), 24))
    rows[:, 0] = time
    rows[:, 1:3] = [1000, 200]
    rows[:, 3] = TWO_PHASE_WATER_DEPTH + wave
    rows[:, 4:6] = [0.2, 0.1]
    rows[:, 6:8] = [0.0, 5000.0]
    rows[:, 8] = TWO_PHASE_WATER_DEPTH + wave
    rows[:, 9] = TWO_PHASE_WATER_DEPTH + 0.8 * wave
    rows[:, 10] = TWO_PHASE_WATER_DEPTH + 0.5 * wave
    rows[:, 15] = TWO_PHASE_WATER_DEPTH + 0.2 * wave
    rows[:, 17:19] = [0.39, 0.41]
    rows[:, 20:23] = [0.001, 0.001, 2.0 * amplitude]
    rows[:, 23] = 0.002

    assert evaluate_metrics(rows, 1000, 200, args)["passed"]
    rows[-1, 11] = 1
    assert not evaluate_metrics(rows, 1000, 200, args)["passed"]
    rows[-1, 11] = 0
    rows[-1, 1] -= 1
    assert not evaluate_metrics(rows, 1000, 200, args)["passed"]
    rows[-1, 1] += 1
    rows[-1, 16] = 1
    assert not evaluate_metrics(rows, 1000, 200, args)["passed"]
    rows[-1, 16] = 0
    rows[-1, 18] = 0.57
    assert not evaluate_metrics(rows, 1000, 200, args)["passed"]
    rows[-1, 18] = 0.41
    rows[-1, 19] = 1
    assert not evaluate_metrics(rows, 1000, 200, args)["passed"]
    rows[-1, 19] = 0
    rows[:, 8] = TWO_PHASE_WATER_DEPTH
    assert not evaluate_metrics(rows, 1000, 200, args)["passed"]
