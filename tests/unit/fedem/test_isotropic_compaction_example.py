"""The compression example owns its setup and geometry without another script."""

import builtins
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.fedem, pytest.mark.cpu]


def load_example(monkeypatch):
    original_import = builtins.__import__

    def without_other_examples(name, *args, **kwargs):
        if name == "examples" or name.startswith("examples."):
            raise ImportError("the example must construct its own scene")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_other_examples)
    path = Path(__file__).resolve().parents[3] / "examples/fedem/IsotropicCompaction/isotropic_compaction.py"
    spec = importlib.util.spec_from_file_location("isotropic_compaction_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("soft_percent,soft_count", [(0, 0), (50, 63), (100, 125)])
def test_standalone_composition_keeps_every_body_and_box_clearance(monkeypatch, soft_percent, soft_count):
    example = load_example(monkeypatch)
    example._set_composition(soft_percent)
    centers, soft_ids, rigid_ids, orientations = example._packing()
    assert len(soft_ids) == soft_count
    np.testing.assert_array_equal(np.sort(np.concatenate((soft_ids, rigid_ids))), np.arange(125))
    assert orientations.shape == centers.shape == (125, 3)
    lower = np.array([example.LEFT, example.FRONT, example.BOTTOM])
    upper = np.array([example.RIGHT, example.BACK, example.TOP])
    assert np.all(centers - example.RADIUS > lower)
    assert np.all(centers + example.RADIUS < upper)


def test_standalone_reference_geometry_preserves_volume_under_placement(monkeypatch):
    example = load_example(monkeypatch)
    from src.fem.generator.Mesh import FEMMesh

    reference = FEMMesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([[0, 1, 2, 3]]), "TET4"
    )
    normalized, quality = example.normalize_reference(reference)
    placed = example.place_reference(normalized, [0.2, 0.3, 0.4], 0.02, [20, 40, 60], name="grain")
    tetra = placed.points[placed.cells][0]
    volume = abs(np.linalg.det((tetra[1:] - tetra[0]).T)) / 6
    assert volume == pytest.approx(quality["minimum_tetra_volume"] * 0.02**3)
    np.testing.assert_allclose(example._volume_centroid(placed.points, placed.cells), [0.2, 0.3, 0.4], atol=1e-14)
