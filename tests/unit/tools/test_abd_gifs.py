"""Container rendering shares all five planes' exact edges and native geometry."""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
from tools.abd_gifs import container_panels, read_mesh


def test_open_top_container_has_five_adjoining_panels():
    bounds = np.asarray([[0.30, 0.20, 0.06], [2.93516, 2.83516, 2.69516]])
    panels = container_panels(bounds)
    assert set(panels) == {"bottom", "left", "right", "front", "back"}
    for panel in panels.values():
        assert panel.n_points == 4 and panel.n_cells == 1
        assert np.all((panel.points == bounds[0]) | (panel.points == bounds[1]))
    for axis, low, high in [(0, "left", "right"), (1, "front", "back")]:
        assert np.all(panels[low].points[:, axis] == bounds[0, axis])
        assert np.all(panels[high].points[:, axis] == bounds[1, axis])
        expected_area = np.prod((bounds[1] - bounds[0])[[1 - axis, 2]])
        assert np.isclose(panels[low].area, expected_area)
        assert np.isclose(panels[high].area, expected_area)
    assert np.all(panels["bottom"].points[:, 2] == bounds[0, 2])
    with pytest.raises(AssertionError):
        container_panels(bounds[::-1])


def test_triangle_vtu_retains_native_points_and_rejects_other_cells(tmp_path):
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = pv.UnstructuredGrid([3, 0, 1, 2], [pv.CellType.TRIANGLE], points)
    file = tmp_path / "native.vtu"
    mesh.save(file)
    np.testing.assert_array_equal(read_mesh(file).points, points)
    with pytest.raises(ValueError, match="native VTU"):
        read_mesh(tmp_path / "._native.vtu")
    quad = pv.UnstructuredGrid([4, 0, 1, 2, 3], [pv.CellType.QUAD], np.r_[points, [[1.0, 1.0, 0.0]]])
    quad.save(file)
    with pytest.raises(ValueError, match="triangle"):
        read_mesh(file)
