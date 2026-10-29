"""Check exact top-face rotation and reject invalid example inputs."""

import json

import numpy as np
import pytest

from examples.iga.twisting_column_3d.twisting_column_3d import parse_arguments, top_displacement
from examples.iga.twisting_column_3d.draw.evaluate_twisting_column_3d import evaluate

pytestmark = [pytest.mark.unit, pytest.mark.iga]


def test_top_rotation_preserves_face_and_uses_total_displacement():
    reference = np.array([[0.3, 0.3, 3.0], [-0.3, 0.3, 3.0], [0.0, 0.0, 3.0]])
    np.testing.assert_array_equal(top_displacement(reference, 90, 0), 0)
    final = reference + top_displacement(reference, 90, 1)
    np.testing.assert_allclose(final, [[-0.3, 0.3, 3], [-0.3, -0.3, 3], [0, 0, 3]], atol=1e-15)
    for fraction in np.linspace(0, 1, 11):
        current = reference + top_displacement(reference, 90, fraction)
        np.testing.assert_allclose(np.linalg.norm(current[:, :2], axis=1), np.linalg.norm(reference[:, :2], axis=1))
        np.testing.assert_array_equal(current[:, 2], reference[:, 2])
    np.testing.assert_allclose(top_displacement(reference, 90, 0.5), top_displacement(reference, 45, 1))
    # 450 degrees has the same endpoint as 90, but must traverse the extra turn.
    np.testing.assert_allclose(top_displacement(reference, 450, 0.5), top_displacement(reference, 225, 1))
    assert not np.allclose(top_displacement(reference, 450, 0.5), top_displacement(reference, 90, 0.5))


def test_solver_defaults_preserve_loading_duration_and_separate_outputs():
    explicit = parse_arguments([])
    implicit = parse_arguments(["--solver", "implicit"])
    for args in (explicit, implicit):
        assert args.steps * args.dt == pytest.approx(1.5)
        assert args.angle == 450
        assert np.prod(args.control_points) * 3 == 1944
    assert explicit.output_dir != implicit.output_dir
    custom = parse_arguments(["--solver", "implicit", "--dt", "0.001", "--steps", "17"])
    assert custom.dt == 0.001 and custom.steps == 17


@pytest.mark.parametrize(
    "argv",
    [
        ["--steps", "0"],
        ["--output-interval", "0"],
        ["--dt", "nan"],
        ["--dt", "0"],
        ["--angle", "inf"],
        ["--control-points", "2", "4", "10"],
    ],
)
def test_invalid_arguments(argv):
    with pytest.raises(SystemExit):
        parse_arguments(argv)


@pytest.fixture
def saved_rotation(tmp_path):
    reference = np.array([[0.3, 0.0, 0.0], [0.3, 0.0, 1.5], [0.3, 0.0, 3.0]])
    np.savez(tmp_path / "reference.npz", control_points=reference, bottom=[0], top=[2])
    summary = dict(
        steps=4,
        output_interval=1,
        dt=0.25,
        angle_degrees=450,
        elements=1,
        solver="implicit",
        backend="test",
        precision="float64",
    )
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    for step, angle in enumerate((0, 70.3125, 225, 379.6875, 450)):
        current = reference.copy()
        current[2, :2] = 0.3 * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        np.savez(
            tmp_path / f"state_{step:06d}.npz",
            step=step,
            time=step * 0.25,
            control_points=current,
            deformation_gradient=np.tile(np.eye(3), (1, 27, 1, 1)),
            stress=np.zeros(3),
        )
    return tmp_path


def test_evaluator_tracks_extra_turn(saved_rotation):
    report = evaluate(saved_rotation)
    assert report["passed"]
    assert report["measured_rotation_degrees"] == pytest.approx(450)


@pytest.mark.parametrize(
    "damage,check",
    [
        ("bottom", "fixed_base"),
        ("top", "rotating_top"),
        ("inversion", "positive_quadrature_jacobian"),
        ("time", "physical_time"),
    ],
)
def test_evaluator_rejects_invalid_results(saved_rotation, damage, check):
    path = saved_rotation / "state_000004.npz"
    with np.load(path) as archive:
        data = dict(archive)
    if damage == "bottom":
        data["control_points"][0, 0] += 0.01
    elif damage == "top":
        data["control_points"][2, 2] += 0.01
    elif damage == "inversion":
        data["deformation_gradient"][0, 0, 0, 0] = -1
    else:
        data["time"] = 0.5
    np.savez(path, **data)
    report = evaluate(saved_rotation)
    assert not report["passed"]
    assert not report["checks"][check]


def test_evaluator_rejects_missing_frames(saved_rotation):
    (saved_rotation / "state_000002.npz").unlink()
    with pytest.raises(ValueError, match="missing or stale"):
        evaluate(saved_rotation)
