"""Input checks for the excavation-to-truck cycle."""

import csv
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation


@pytest.fixture
def cycle(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "examples/mpdem/AffineBody/excavator_dig"))
    import excavator_cycle

    return excavator_cycle


def test_pose_interpolates_pin_and_continuous_quaternion(cycle, tmp_path):
    path = tmp_path / "trajectory.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s", "pin_x_m", "pin_y_m", "pin_z_m", "qx", "qy", "qz", "qw"])
        for i in range(145):
            t = i / 24
            quaternion = (-1) ** i * Rotation.from_euler("z", 10 * t, degrees=True).as_quat()
            writer.writerow([i + 1, f"{t:.6f}", t, 0, 1, *quaternion])
    pose = cycle.load_motion(path)
    time = 37.5 / 24
    pin, rotation = pose(time)
    np.testing.assert_allclose(pin, [time, 0, 1])
    np.testing.assert_allclose(rotation.as_euler("xyz", degrees=True), [0, 0, time * 10], atol=1e-12)
    np.testing.assert_allclose(pose(6.0 + 1e-14)[0], [6, 0, 1])


def test_sample_soil_stays_in_volume_and_above_plane_gap(cycle):
    mesh = trimesh.creation.box()
    points = cycle.sample_soil(mesh, 180, 0.01)
    assert points.shape == (180, 3)
    assert mesh.contains(points).all() and points[:, 2].min() > 0.01
    np.testing.assert_array_equal(points, cycle.sample_soil(mesh, 180, 0.01))
    with pytest.raises(ValueError, match="above the ground clearance"):
        cycle.sample_soil(mesh, 180, 0.5)


def test_bundled_scene_contains_complete_motion_and_contact_solids(cycle):
    scene = Path(cycle.__file__).parent / "inputs"
    pose = cycle.load_motion(scene / "trajectory.csv")
    assert np.isfinite(pose(6)[0]).all()
    bucket = cycle.load_solid(scene / "contact/bucket_contact.obj")
    assert len(bucket.faces) == 84
    assert len(cycle.load_solid(scene / "contact/truck_bed_contact.obj").faces) == 92
    soil = cycle.load_solid(scene / "contact/soil_initial_volume.obj")
    np.testing.assert_allclose(soil.extents, [6.4, 4.9, 4.808], atol=0.001)
    assert soil.volume == pytest.approx(48.196, abs=0.001)
    pin, rotation = pose(0)
    assert (pin + rotation.apply(bucket.vertices))[:, 2].min() > soil.bounds[1, 2]
