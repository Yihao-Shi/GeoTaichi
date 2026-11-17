"""Joint freedom, feedback, and time-dependent motor trajectory gradients."""

import numpy as np
import pytest

from tests.unit.dem.test_affine_time_integration import build_body

pytestmark = pytest.mark.isolated_dimension(3)


def test_batch_targets_validate_before_mutating_scene():
    from src.dem.SceneManager import SceneManager

    scene = SceneManager.__new__(SceneManager)
    scene.affine_joints = [{"TargetAngle": 1.0}, {"TargetAngle": 2.0}, {"JointType": "Fixed"}]
    for ids, angles, error in (
        ([0, 1], [3.0, np.nan], ValueError),
        ([0, 3], [3.0, 4.0], IndexError),
        ([0.5], [3.0], IndexError),
        ([0, 0], [3.0, 4.0], ValueError),
        ([0, 2], [3.0, 4.0], ValueError),
        ([0], [[3.0]], ValueError),
    ):
        with pytest.raises(error):
            scene.set_affine_joint_target_angles(ids, angles)
        assert scene.affine_joints == [{"TargetAngle": 1.0}, {"TargetAngle": 2.0}, {"JointType": "Fixed"}]
    scene.set_affine_joint_target_angles([1, 0], [-4.0, 5.0])
    assert [joint["TargetAngle"] for joint in scene.affine_joints[:2]] == [5.0, -4.0]


@pytest.mark.parametrize("kind", ["Revolute", "Fixed", "Spherical"])
@pytest.mark.parametrize("pair", [False, True])
def test_joint_freedoms_and_physical_energy_gradient(tmp_path, taichi_runtime, kind, pair):
    from src.dem.engines.AffineBodyOperator import MATRIX_COO

    anchor = np.array([1.0, 1.0, 1.0])
    spec = dict(JointType=kind, BodyID1=0, BodyID2=1 if pair else -1, WorldAnchor=anchor)
    if kind == "Revolute":
        spec["WorldAxis"] = [0.0, 0.0, 1.0]
    dem = build_body(tmp_path, (1.0, 0.5, 1.0), pair=pair, joint=spec)
    op = dem.enginer.operator
    initial = op.y.to_numpy()

    def terms(positions):
        op.y.from_numpy(np.ascontiguousarray(positions))
        op._clear_system(False, MATRIX_COO)
        op._assemble_joint_constraints(False, MATRIX_COO)
        return float(op.energy[None]), op.grad.to_numpy()

    theta = 0.3
    c, s = np.cos(theta), np.sin(theta)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    for rotation, free in ((rz, kind != "Fixed"), (rx, kind == "Spherical")):
        rotated = initial.copy()
        rotated[:4] = anchor + (initial[:4] - anchor) @ rotation.T
        energy, _ = terms(rotated)
        if free:
            assert energy == pytest.approx(0.0, abs=1e-20)
        else:
            assert energy > 0.0
    moved = initial.copy()
    moved[:4] += [0.01, -0.02, 0.03]
    energy, gradient = terms(moved)
    assert energy > 0.0
    direction = np.random.default_rng(54).normal(size=initial.shape)
    epsilon = 1e-6
    numerical = (terms(moved + epsilon * direction)[0] - terms(moved - epsilon * direction)[0]) / (2 * epsilon)
    assert np.sum(gradient * direction) == pytest.approx(numerical, rel=1e-8, abs=1e-11)
    if pair:
        np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=1e-15)


@pytest.mark.parametrize("pair", [False, True])
def test_joint_feedback_uses_angle_derivative_and_public_degrees(tmp_path, taichi_runtime, pair):
    dem = build_body(
        tmp_path,
        (1.0, 0.5, 1.0),
        pair=pair,
        joint=dict(BodyID1=0, BodyID2=1 if pair else -1, WorldAnchor=[1, 1, 1], WorldAxis=[0, 0, 1]),
    )
    op = dem.enginer.operator
    y = op.y.to_numpy()
    theta = np.deg2rad(31.0)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    y[:4] = (y[:4] - 1.0) @ rotation.T + 1.0
    velocity = np.zeros_like(y)
    velocity[:4] = np.cross([0, 0, 0.7], y[:4] - 1.0) + [0.3, -0.2, 0.1]
    if pair:
        velocity[4:] = np.cross([0, 0, -0.2], y[4:] - 1.0)
    op.y.from_numpy(y)
    op.velocity_y.from_numpy(velocity)
    dem.set_joint_target_angles([0], [-20.0])
    state = dem.get_joint_states()
    sign = -1.0 if pair else 1.0
    assert state["angle"][0] == pytest.approx(sign * 31.0)
    assert state["angular_velocity"][0] == pytest.approx(sign * np.rad2deg(0.9 if pair else 0.7))
    assert state["target_angle"][0] == pytest.approx(-20.0)
    assert state["anchor_error"][0] < 1e-14
    assert state["axis_error"][0] < 1e-14


def test_motor_sequence_vjp_matches_complete_forward_and_restores_controls(tmp_path, taichi_runtime, monkeypatch):
    dem = build_body(
        tmp_path,
        (1.0, 0.5, 1.0),
        joint=dict(BodyID1=0, WorldAnchor=[1, 1, 1], WorldAxis=[0, 0, 1], MotorStiffness=20.0, Damping=0.1),
    )
    op = dem.enginer.operator
    initial_y, initial_v = op.y.to_numpy(), op.velocity_y.to_numpy()
    trajectory = dem.differentiable_affine(steps=3)
    commands = np.array([10.0, -15.0, 20.0])
    for angle in commands:
        dem.set_joint_target_angle(0, angle)
        trajectory.step()
    terminal_y, terminal_v = op.y.to_numpy(), op.velocity_y.to_numpy()
    seed = np.random.default_rng(43).normal(size=terminal_y.shape)
    dem.set_joint_target_angle(0, 37.0)
    gradient = trajectory.backward(seed)
    repeated = trajectory.backward(seed)
    for key in gradient:
        np.testing.assert_allclose(gradient[key], repeated[key], atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(op.y.to_numpy(), terminal_y)
    np.testing.assert_array_equal(op.velocity_y.to_numpy(), terminal_v)
    assert dem.sims.current_step == 3
    assert dem.get_joint_states()["target_angle"][0] == pytest.approx(37.0)
    with monkeypatch.context() as patch:

        def fail_step(*args):
            raise RuntimeError("forced replay failure")

        patch.setattr(dem.enginer, "step", fail_step)
        with pytest.raises(RuntimeError, match="forced replay failure"):
            trajectory.backward(seed)
    np.testing.assert_array_equal(op.y.to_numpy(), terminal_y)
    np.testing.assert_array_equal(op.velocity_y.to_numpy(), terminal_v)
    assert dem.sims.current_step == 3
    assert dem.get_joint_states()["target_angle"][0] == pytest.approx(37.0)
    per_step = gradient["joint_target_angle_degrees_per_step"]
    assert per_step.shape == (3, 1)
    np.testing.assert_allclose(per_step.sum(axis=0), gradient["joint_target_angle_degrees"], atol=1e-12)

    def objective(targets):
        op.y.from_numpy(initial_y)
        op.velocity_y.from_numpy(initial_v)
        dem.sims.current_time, dem.sims.current_step = 0.0, 0
        for angle in targets:
            dem.set_joint_target_angle(0, angle)
            dem.enginer.step(dem.sims, dem.scene)
            dem.sims.current_time += dem.sims.delta
            dem.sims.current_step += 1
        return np.sum(seed * op.y.to_numpy())

    epsilon = 1e-3
    for step in range(3):
        direction = np.eye(3)[step]
        numerical = (objective(commands + epsilon * direction) - objective(commands - epsilon * direction)) / (
            2 * epsilon
        )
        assert abs(per_step[step, 0]) > 1e-8
        assert per_step[step, 0] == pytest.approx(numerical, rel=3e-3, abs=1e-9)


@pytest.mark.parametrize("kind", ["Fixed", "Spherical"])
def test_new_joint_trajectory_state_vjp(tmp_path, taichi_runtime, kind):
    dem = build_body(
        tmp_path,
        (1.0, 0.5, 1.0),
        joint=dict(JointType=kind, BodyID1=0, WorldAnchor=[1, 1, 1]),
    )
    op = dem.enginer.operator
    initial_y, initial_v = op.y.to_numpy(), op.velocity_y.to_numpy()
    trajectory = dem.differentiable_affine(steps=2)
    trajectory.step()
    trajectory.step()
    seed = np.random.default_rng(21).normal(size=initial_y.shape)
    gradient = trajectory.backward(seed)
    assert np.count_nonzero(gradient["joint_target_angle_degrees_per_step"]) == 0
    assert np.count_nonzero(gradient["joint_damping"]) == 0
    direction = np.random.default_rng(25).normal(size=initial_y.shape)

    def objective(shift):
        op.y.from_numpy(initial_y + shift * direction)
        op.velocity_y.from_numpy(initial_v)
        dem.sims.current_step, dem.sims.current_time = 0, 0.0
        for _ in range(2):
            dem.enginer.step(dem.sims, dem.scene)
            dem.sims.current_time += dem.sims.delta
            dem.sims.current_step += 1
        return np.sum(seed * op.y.to_numpy())

    epsilon = 2e-6
    numerical = (objective(epsilon) - objective(-epsilon)) / (2 * epsilon)
    assert np.sum(gradient["initial_position"].reshape(-1, 3) * direction) == pytest.approx(
        numerical, rel=2e-4, abs=2e-6
    )
