"""Input checks for the excavation-to-truck cycle."""

import csv
import json
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
    bucket_surface = np.vstack([bucket.vertices, bucket.triangles_center])
    tooth = np.array([-0.05, 1.49, 0.065])
    for t in np.arange(16) / 24:
        pin, rotation = pose(t)
        surface = pin + rotation.apply(bucket_surface)
        assert trimesh.proximity.signed_distance(soil, surface).max() < -0.024

    far_pin, far_rotation = pose(15 / 24)
    end_pin, end_rotation = pose(1.0)
    far_tooth = far_pin + far_rotation.apply(tooth)
    end_tooth = end_pin + end_rotation.apply(tooth)
    assert far_tooth[1] < -7.0 and end_tooth[1] > -6.2
    assert end_tooth[2] == pytest.approx(far_tooth[2], abs=0.03)
    for t in np.arange(15, 36, 0.5) / 24:
        pin, rotation = pose(t)
        surface = pin + rotation.apply(bucket_surface)
        inside = trimesh.proximity.signed_distance(soil, surface) > 0
        assert np.all(surface[inside, 2] > 4.4)
        if t == 1.0:
            assert inside.any()
    pin, rotation = pose(36 / 24)
    assert trimesh.proximity.signed_distance(soil, pin + rotation.apply(bucket_surface)).max() < -0.024


def test_soil_overburden_uses_local_surface_and_recovers_kirchhoff_stress(cycle):
    soil = trimesh.creation.box()
    soil.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [1, 0, 0]))
    points = np.array([[0, 0.25, 0], [0, 0.25, 0.25], [0, 0, 0.25]])
    material = dict(density=1700.0, young_modulus=2e5, poisson_ratio=0.3, FrictionAngle=30.0)
    gravity = 9.81
    k0, stress, gradient = cycle.soil_overburden(soil, points, material, gravity)
    expected_depth = np.sqrt(0.5) - np.abs(points[:, 1]) - points[:, 2]
    expected_stress = -1700.0 * gravity * expected_depth[:, None] * [0.5, 0.5, 1.0]
    assert k0 == pytest.approx(0.5)
    np.testing.assert_allclose(stress, expected_stress)
    assert np.all(np.linalg.det(gradient) > 0.0)
    np.testing.assert_allclose(gradient, gradient * np.eye(3))
    strain = np.log(np.diagonal(gradient, axis1=1, axis2=2))
    np.testing.assert_allclose(
        2 * (2e5 / 2.6) * strain + (2e5 * 0.3 / (1.3 * 0.4)) * strain.sum(axis=1, keepdims=True),
        expected_stress,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "case,count,volume,settle", [("cone", 150000, 77.571077, 1.8), ("trench", 149454, 24.4140625, 1.1)]
)
def test_paired_excavation_preserves_soil_volume_and_motion(cycle, case, count, volume, settle):
    inputs = cycle.EXPLICIT / f"explicit_{case}/inputs"
    config = json.loads((inputs / f"{case}.json").read_text())
    points, point_volume, depth, container = cycle.paired_soil(case, config, 150000)
    assert len(points) == count and count * point_volume == pytest.approx(volume)
    assert (depth > 0).all()
    if case == "trench":
        minimum, maximum, rim = container
        assert np.all((minimum < points) & (points < maximum)) and rim == 1.5
        assert [len(np.unique(points[:, i])) for i in range(3)] == [57, 114, 23]
    pose, phases, duration = cycle.paired_motion(inputs / "trajectory.csv", settle, case)
    np.testing.assert_allclose(pose(0)[0], pose(settle)[0])
    assert np.linalg.norm(pose(phases["swing"])[0] - pose(0)[0]) > 1
    assert duration == pytest.approx(settle + 15) and phases["retract"] == duration


def test_coarse_trench_grid_fixes_floor_and_only_wall_normal_below_rim():
    from examples.mpdem.soil_setup import soil_boundary_dofs

    h = 0.3125
    lo = np.array([-1.5, -1.25, -0.625])
    hi = lo + 12 * h
    minimum, maximum = np.array([-0.875, -0.625, 0]), np.array([1, 1.25, 1.25])
    fixed = [set(a) for a in soil_boundary_dofs(lo, hi, h, (minimum, maximum, 1.5625))]

    def components(x, y, z):
        node = x + 13 * (y + 13 * z)
        return [axis for axis in range(3) if 3 * node + axis in fixed[axis]]

    assert components(4, 4, 2) == [0, 1, 2]
    assert components(2, 4, 4) == [0]
    assert components(4, 8, 4) == [1]
    assert components(2, 4, 10) == []
    assert components(4, 4, 4) == []
