"""Geometry and output acceptance for the ABD cradle, without a runtime import."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load_case():
    path = Path(__file__).resolve().parents[3] / "examples/dem/AffineBody/newtons_cradle/newtons_cradle.py"
    spec = importlib.util.spec_from_file_location("newtons_cradle_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initial_geometry_is_on_pivot_circles_and_strictly_separated():
    case = load_case()
    rest, pivots, centers = case.geometry()
    np.testing.assert_allclose(np.linalg.norm(centers - pivots, axis=1), case.LENGTH, atol=1e-14)
    np.testing.assert_allclose(centers[1:], rest[1:])
    assert centers[0, 0] < rest[0, 0] and centers[0, 2] > rest[0, 2]
    for i in range(case.COUNT):
        for j in range(i + 1, case.COUNT):
            assert np.linalg.norm(centers[i] - centers[j]) - 2 * case.RADIUS > case.DHAT


def test_saved_output_uses_mass_weighted_centers_and_exact_affine_kinetic_energy():
    case = load_case()
    _, _, centers = case.geometry()
    y = np.repeat(centers[:, None, :], 4, axis=1)
    y[:, :, 0] += np.arange(4)[None, :]
    v = np.ones_like(y) * 2
    mass = np.tile(np.diag([1.0, 2.0, 3.0, 4.0]), (case.COUNT, 1, 1))
    c, velocity, anchors, energy = case.frame_values(
        dict(y=y, v_y=v, mass_matrix=mass), np.tile([1.0, 0, 0, 0], (case.COUNT, 1))
    )
    np.testing.assert_allclose(c[:, 0], centers[:, 0] + 2)
    np.testing.assert_allclose(velocity, 2)
    np.testing.assert_allclose(anchors, y[:, 0])
    expected_potential = 10 * case.GRAVITY * (centers[0, 2] - case.geometry()[0][0, 2])
    assert energy == pytest.approx(0.5 * case.COUNT * 10 * 12 + expected_potential)


def test_finite_stationary_balls_do_not_pass_cradle_transmission(tmp_path):
    case = load_case()
    _, _, centers = case.geometry()
    origin = centers - 0.25
    y = np.stack((origin, origin + [1, 0, 0], origin + [0, 1, 0], origin + [0, 0, 1]), axis=1)
    particles = tmp_path / "particles"
    particles.mkdir()
    (tmp_path / "vtks").mkdir()
    for i, t in enumerate((0.0, 0.6)):
        np.savez(
            particles / f"AffineBody{i:06d}.npz",
            t_current=t,
            y=y,
            v_y=np.zeros_like(y),
            vertices=centers,
            bodyID=np.arange(case.COUNT),
            volume=np.ones(case.COUNT),
            young_modulus=np.ones(case.COUNT),
            mass_matrix=np.tile(np.eye(4), (case.COUNT, 1, 1)),
        )
    with pytest.raises(RuntimeError, match="physical acceptance failed"):
        case.analyze(tmp_path, 0.6, False, "cpu")
    summary = json.loads((tmp_path / "validation_summary.json").read_text())
    assert summary["finite"] and not summary["passed"]
    assert summary["outgoing_speed_ratio"] == 0


def test_energy_includes_penalties_and_does_not_omit_active_ipc():
    case = load_case()
    _, pivots, centers = case.geometry()
    y = np.stack((centers, centers + [1, 0, 0], centers + [0, 1, 0], centers + [0, 0, 1]), axis=1)
    weights = np.array([np.linalg.solve(np.column_stack((q, np.ones(4))).T, np.r_[p, 1.0]) for q, p in zip(y, pivots)])
    state = dict(
        y=y.copy(),
        v_y=np.zeros_like(y),
        mass_matrix=np.tile(np.eye(4), (case.COUNT, 1, 1)),
        vertices=centers.copy(),
        bodyID=np.arange(case.COUNT),
        volume=np.ones(case.COUNT),
        young_modulus=np.full(case.COUNT, 8.0),
    )
    state["y"][0, 1, 0] += 0.01
    strain_energy = (1.01**2 - 1) ** 2
    anchor_error = 0.01 * weights[0, 1]
    expected = case.frame_values(state, weights)[3] + strain_energy + 0.5 * case.HINGE * anchor_error**2
    assert case.contact_free_energy(state, y, weights) == pytest.approx(expected)
    state["vertices"][2] = state["vertices"][1] + [case.DHAT / 2, 0, 0]
    assert case.contact_free_energy(state, y, weights) is None


@pytest.mark.parametrize("axis", [0, 2])
def test_transfer_gate_checks_horizontal_motion_separately_from_energy(tmp_path, axis):
    """A large outgoing pulse alone cannot hide residual striker/middle motion."""
    case = load_case()
    rest, pivots, centers = case.geometry()
    origin = centers - 0.25
    y0 = np.stack((origin, origin + [1, 0, 0], origin + [0, 1, 0], origin + [0, 0, 1]), axis=1)
    particles = tmp_path / "particles"
    particles.mkdir()
    (tmp_path / "vtks").mkdir()
    incoming = np.sqrt(2 * case.GRAVITY * case.LENGTH * (1 - np.cos(case.ANGLE)))
    for i, t in enumerate((0.0, 0.38, 0.4)):
        y = y0.copy()
        v = np.zeros_like(y)
        vertices = centers.copy()
        if i:
            y[-1, :, 0] += 0.02
            vertices[-1, 0] += 0.02
            v[-1, :, 0] = 0.98 * incoming
        if i == 2:
            v[0, :, axis] = -0.2 * incoming
            v[-2, :, axis] = 0.3 * incoming
            v[-1, :, 0] = 0.9 * incoming
        np.savez(
            particles / f"AffineBody{i:06d}.npz",
            t_current=t,
            y=y,
            v_y=v,
            vertices=vertices,
            bodyID=np.arange(case.COUNT),
            volume=np.ones(case.COUNT),
            young_modulus=np.zeros(case.COUNT),
            mass_matrix=np.tile(np.eye(4), (case.COUNT, 1, 1)),
        )
    with pytest.raises(RuntimeError, match="physical acceptance failed"):
        case.analyze(tmp_path, 0.4, False, "cpu")
    summary = json.loads((tmp_path / "validation_summary.json").read_text())
    assert summary["outgoing_speed_ratio"] == pytest.approx(0.98)
    assert summary["first_transfer"]["passed"] == (axis == 2)
    assert not summary["passed"]  # The displaced-anchor energy check remains independent.
    assert summary["first_transfer"]["striker_speed_ratio"] == pytest.approx(0.2 if axis == 0 else 0)
    assert summary["first_transfer"]["middle_speed_ratio"] == pytest.approx(0.3 if axis == 0 else 0)

    # A good first outgoing snapshot must not hide residual motion in later frames.
    assert summary["first_transfer"]["time"] == pytest.approx(0.38)
    assert summary["first_transfer"]["outgoing_speed_ratio"] == pytest.approx(0.98)
    check = json.loads((tmp_path / "validation-evidence.json").read_text())["checks"][0]
    assert check["expected"] == [0, 1] and check["observed"] == pytest.approx(6 if axis == 0 else 0)


def test_eight_round_trips_require_large_returns_not_residual_wobble():
    case = load_case()
    period = case.reference_period()
    times = np.linspace(0, 8.1 * period, 4001)
    phase = 2 * np.pi * times / period
    amplitude = case.LENGTH * np.sin(case.ANGLE)
    centre = case.geometry()[0][0, 0]
    history = np.column_stack(
        (times, centre - amplitude * np.cos(phase), amplitude * 2 * np.pi / period * np.sin(phase))
    )
    returns = case.completed_round_trips(history)
    assert len(returns) == 8
    np.testing.assert_allclose(returns, np.arange(1, 9) * period, atol=times[1] - times[0])
    history[:, 1] = centre + 0.01 * np.cos(phase * 30)
    history[:, 2] = -0.01 * 30 * 2 * np.pi / period * np.sin(phase * 30)
    assert case.completed_round_trips(history) == []


@pytest.mark.parametrize("outgoing_ratio,expected_pass", [(0.6, True), (1.2, False)])
def test_correct_transfer_allows_decay_but_rejects_energy_creation(tmp_path, outgoing_ratio, expected_pass):
    case = load_case()
    rest, pivots, centres = case.geometry()
    origin = centres - 0.25
    initial = np.stack((origin, origin + [1, 0, 0], origin + [0, 1, 0], origin + [0, 0, 1]), axis=1)
    weights_mass = np.ones((4, 4)) / 16 + np.eye(4) * 1e-6
    mass = np.tile(weights_mass, (case.COUNT, 1, 1))
    particles = tmp_path / "particles"
    particles.mkdir()
    (tmp_path / "vtks").mkdir()
    incoming = np.sqrt(2 * case.GRAVITY * case.LENGTH * (1 - np.cos(case.ANGLE)))

    def rotation(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    for index, timestamp in enumerate((0.0, 0.4)):
        y = initial.copy()
        velocity = np.zeros_like(y)
        if index:
            y[0] = (y[0] - pivots[0]) @ rotation(-case.ANGLE).T + pivots[0]
            angle = -np.arcsin(0.02 / case.LENGTH)
            y[-1] = (y[-1] - pivots[-1]) @ rotation(angle).T + pivots[-1]
            centre = y[-1].mean(axis=0)
            vx = outgoing_ratio * incoming
            angular = np.array([0, -vx / np.sqrt(case.LENGTH**2 - 0.02**2), 0])
            tangent = np.cross(angular, centre - pivots[-1])
            velocity[-1] = tangent + np.cross(angular, y[-1] - centre)
        np.savez(
            particles / f"AffineBody{index:06d}.npz",
            t_current=timestamp,
            y=y,
            v_y=velocity,
            vertices=y.mean(axis=1),
            bodyID=np.arange(case.COUNT),
            volume=np.ones(case.COUNT),
            young_modulus=np.zeros(case.COUNT),
            mass_matrix=mass,
        )
    if expected_pass:
        summary = case.analyze(tmp_path, 0.4, False, "cpu")
    else:
        with pytest.raises(RuntimeError, match="physical acceptance failed"):
            case.analyze(tmp_path, 0.4, False, "cpu")
        summary = json.loads((tmp_path / "validation_summary.json").read_text())
    assert summary["passed"] == expected_pass
    assert summary["first_transfer"]["passed"]
    assert (summary["maximum_certified_total_energy_ratio"] <= 1 + case.ENERGY_GROWTH_TOLERANCE) == expected_pass


def test_round_trip_count_accepts_visible_damped_returns():
    case = load_case()
    period = case.reference_period()
    times = np.linspace(0, 8.1 * period, 5001)
    omega = 2 * np.pi / period
    amplitude = case.LENGTH * np.sin(case.ANGLE) * np.exp(-0.15 * times)
    position = case.geometry()[0][0, 0] - amplitude * np.cos(omega * times)
    velocity = amplitude * (0.15 * np.cos(omega * times) + omega * np.sin(omega * times))
    returns = case.completed_round_trips(np.column_stack((times, position, velocity)))
    assert len(returns) == 8
    assert amplitude[-1] < 0.5 * amplitude[0] and amplitude[-1] > case.MINIMUM_RETURN_EXCURSION
