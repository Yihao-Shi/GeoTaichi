"""Mesh extraction distinguishes empty cells from failed surface construction."""

import numpy as np
import pytest

from src.sdf import mesh

pytestmark = [pytest.mark.unit, pytest.mark.geometry, pytest.mark.cpu]


@pytest.mark.parametrize("offset", [0.0, 0.5])
def test_crossing_or_lower_endpoint_retains_the_plane(offset):
    axis = np.array([0.0, 1.0])
    points = mesh._worker(lambda p: p[:, 0] - offset, (axis, axis, axis), sparse=False)
    np.testing.assert_allclose(points[:, 0], offset, atol=1e-7)
    assert len(points) == 6


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_uniform_cells_have_no_surface(value):
    axis = np.array([0.0, 1.0])
    assert mesh._worker(lambda p: np.full(len(p), value), (axis, axis, axis), sparse=False) == []


@pytest.mark.parametrize("error", [MemoryError, RuntimeError, ValueError])
def test_surface_extraction_errors_propagate(monkeypatch, error):
    def fail(_volume):
        raise error("surface extraction failed")

    monkeypatch.setattr(mesh, "_marching_cubes", fail)
    axis = np.array([0.0, 1.0])
    with pytest.raises(error, match="surface extraction failed"):
        mesh._worker(lambda p: p[:, 0] - 0.5, (axis, axis, axis), sparse=False)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_samples_fail(value):
    axis = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        mesh._worker(lambda p: np.full(len(p), value), (axis, axis, axis), sparse=False)
