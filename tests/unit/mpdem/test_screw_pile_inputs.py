"""Screw-pile mesh, soil sampling, and inclined kinematic drive."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation


@pytest.fixture(scope="module")
def pile():
    folder = Path(__file__).resolve().parents[3] / "examples/mpdem/AffineBody/screw_pile"
    spec = importlib.util.spec_from_file_location("screw_pile", folder / "screw_pile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, json.loads((folder / "model-contract.json").read_text())


@pytest.mark.parametrize("name,count,faces", [("low", 196, 3000), ("full", 18032, 14012)])
def test_pile_profiles_preserve_geometry_and_soil_volume(pile, name, count, faces):
    module, contract = pile
    profile = contract["discretization"]["profiles"][name]
    mesh = trimesh.load_mesh(module.INPUTS / profile["mesh"], merge_tex=True, merge_norm=True)
    assert mesh.is_volume and mesh.body_count == 1 and mesh.euler_number == 0
    assert len(mesh.faces) == faces
    np.testing.assert_allclose(mesh.extents, [2.8, 2.8, 13], atol=0.005)
    size = np.array(contract["geometry"]["soil_size_m"])
    points, volume = module.soil_points(size, np.array(profile["soil_counts"]))
    assert points.shape == (count, 3)
    assert np.abs(points[:, :2]).max() < size[0] / 2
    assert 0 < points[:, 2].min() < points[:, 2].max() < size[2]
    assert volume * count == pytest.approx(size.prod())


def test_pile_rotates_clockwise_while_advancing_along_its_axis(pile):
    module, contract = pile
    geometry, timing = contract["geometry"], contract["time"]
    tilt = Rotation.from_euler("y", geometry["inclination_deg"], degrees=True)
    tip = np.array([0.0, 0.0, 3.22])
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
    end, orientation, advance, turns = module.command(settle + duration, tilt, tip, pitch, 1, settle, duration)
    np.testing.assert_allclose(end, tip - pitch * axis)
    np.testing.assert_allclose(orientation.as_matrix(), tilt.as_matrix(), atol=1e-14)
    assert (advance, turns) == (pitch, 1)
