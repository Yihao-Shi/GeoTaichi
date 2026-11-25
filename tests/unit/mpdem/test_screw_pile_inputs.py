"""Screw-pile mesh, soil sampling, and inclined kinematic drive."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from examples.mpdem.LevelSet.ScrewPile import pile_archive


@pytest.fixture(scope="module")
def pile():
    folder = Path(__file__).resolve().parents[3] / "examples/mpdem/AffineBody/ScrewPile"
    spec = importlib.util.spec_from_file_location("screw_pile", folder / "screw_pile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, json.loads((folder / "model-contract.json").read_text())


@pytest.mark.parametrize("name,count,faces", [("low", 196, 3000), ("full", 149940, 14012)])
def test_pile_profiles_preserve_geometry_and_soil_volume(pile, name, count, faces):
    module, contract = pile
    profile = contract["discretization"]["profiles"][name]
    mesh = trimesh.load_mesh(module.INPUTS / profile["mesh"], merge_tex=True, merge_norm=True)
    assert mesh.is_volume and mesh.body_count == 1 and mesh.euler_number == 0
    assert len(mesh.faces) == faces
    np.testing.assert_allclose(mesh.extents, [2.8, 2.8, 13], atol=0.005)
    size = np.array(contract["geometry"]["soil_size_m"])
    points, volume = module.box_points([-size[0] / 2, -size[1] / 2, 0], size, profile["soil_counts"])
    assert points.shape == (count, 3)
    assert np.abs(points[:, :2]).max() < size[0] / 2
    assert 0 < points[:, 2].min() < points[:, 2].max() < size[2]
    assert volume * count == pytest.approx(size.prod())
    assert size.tolist() == [16.0, 16.0, 16.0]
    material = contract["materials"][0]["parameters"]
    k0, stress, deformation = module.geostatic_state(size[2] - points[:, 2], material, 9.81)
    assert k0 == pytest.approx(0.5)
    assert np.all(stress < 0.0) and np.all(np.linalg.det(deformation) > 0.0)


def test_pile_rotates_clockwise_while_advancing_along_its_axis(pile):
    module, contract = pile
    geometry, timing = contract["geometry"], contract["time"]
    tilt = Rotation.from_euler("y", geometry["inclination_deg"], degrees=True)
    tip = np.array([1.8570266486, 0.0, 16.0290586666])
    pitch = geometry["pitch_m"]
    settle, duration = timing["settle_s"], timing["installation_s"]
    axis = tilt.apply([0, 0, 1])
    progress = 0.25
    position, orientation, advance, turns = module.command(
        settle + progress * duration, tilt, tip, pitch, 1, settle, duration
    )
    assert 0 < turns < 0.5
    np.testing.assert_allclose(orientation.apply([0, 0, 1]), axis)
    np.testing.assert_allclose(position, tip - advance * axis)
    assert (tilt.inv() * orientation).as_rotvec()[2] == pytest.approx(-2 * np.pi * turns)
    assert advance == pytest.approx(pitch * turns)
    target_turns = geometry["turns"]
    end, orientation, advance, turns = module.command(
        settle + duration, tilt, tip, pitch, target_turns, settle, duration
    )
    np.testing.assert_allclose(end, tip - pitch * target_turns * axis)
    assert 0 < end[2] < tip[2] - 13.0
    np.testing.assert_allclose(orientation.as_matrix(), tilt.as_matrix(), atol=1e-14)
    assert turns == 7 and advance == pytest.approx(14.35)


def test_explicit_pile_profiles_limit_initial_cell_occupancy():
    folder = Path(__file__).resolve().parents[3] / "examples/mpdem/LevelSet/ScrewPile"
    config = json.loads((folder / "inputs/pile.json").read_text())
    size = np.asarray(config["geometry"]["soil_size_m"])
    np.testing.assert_array_equal(size, [16.0, 16.0, 16.0])
    for name, profile in config["profiles"].items():
        h = profile["grid_size_m"]
        counts = profile["soil_counts"]
        max_axis_counts = []
        for width, count in zip(size, counts):
            cells = round(width / h)
            assert cells * h == pytest.approx(width)
            coordinates = (np.arange(count) + 0.5) * (width / count)
            occupied_cells = np.floor(coordinates / h).astype(int)
            max_axis_counts.append(np.bincount(occupied_cells).max())
        assert np.prod(max_axis_counts) <= 8
        if name == "h200":
            assert np.prod(max_axis_counts) == 8
            assert np.prod(counts) == 201_230_056
            assert profile["dt_s"] == pytest.approx(1e-4)
            assert config["settle_s"] + config["geometry"]["turns"] * profile["seconds_per_turn"] == pytest.approx(18.0)


def test_pile_npz_frames_keep_all_points_and_publish_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(pile_archive, "CHUNK", 2)
    path = tmp_path / "000000.npz"
    origin = np.array([-8.0, -8.0, 0.0])
    scale = np.array([16.0, 16.0, 32.0]) / 65535
    coordinates = np.array([[0, 65535, 32768], [456, 789, 123], [654, 987, 321]], dtype=np.uint16)

    def copy(start, _, block):
        block[:] = coordinates[start : start + len(block)]

    pile_archive.save_frame(path, len(coordinates), origin, scale, copy)
    np.testing.assert_allclose(pile_archive.load_frame(path, 3), origin + scale * coordinates, atol=2e-6)
    with pytest.raises(OSError, match="disk full"):
        with pile_archive.atomic_npz(path) as archive:
            pile_archive.write_array(archive, "step", np.array(12))
            raise OSError("disk full")
    assert pile_archive.frame_paths(tmp_path) == [path]
    assert not path.with_suffix(".partial").exists()
    np.testing.assert_allclose(pile_archive.load_frame(path, 3), origin + scale * coordinates, atol=2e-6)
