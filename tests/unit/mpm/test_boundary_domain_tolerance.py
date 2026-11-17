from types import SimpleNamespace

import numpy as np
import pytest

from src.mpm.boundaries.BoundaryConstraint import BoundaryConstraints


def test_boundary_domain_accepts_decimal_endpoint_after_float32_rounding():
    boundary = BoundaryConstraints()
    simulation = SimpleNamespace(dimension=2, domain=np.asarray([0.4, 0.12], dtype=np.float32))

    boundary.check_boundary_domain(simulation, [0.0, 0.0], [0.4, 0.12])

    with pytest.raises(RuntimeError, match="EndPoint"):
        boundary.check_boundary_domain(simulation, [0.0, 0.0], [0.4, 0.121])


def test_zero_thickness_solid_plane_expands_toward_its_normal():
    boundary = BoundaryConstraints()
    simulation = SimpleNamespace(dimension=3, domain=np.asarray([0.36, 0.16, 0.36]))
    element = SimpleNamespace(grid_size=np.asarray([0.005] * 3), ghost_cell=1)

    boundary.set_solid_plane_cell_boundary(
        simulation,
        element,
        {
            "BoundaryType": "SolidPlaneCell",
            "StartPoint": [0.03, 0.05, 0.125],
            "EndPoint": [0.03, 0.11, 0.325],
            "Norm": [-1.0, 0.0, 0.0],
            "CellThickness": 1,
        },
        [0.03, 0.05, 0.125],
        [0.03, 0.11, 0.325],
    )

    lower, upper, point, normal = boundary.solid_cell_plane_regions[0]
    assert np.allclose(lower, [0.025, 0.05, 0.125])
    assert np.allclose(upper, [0.03, 0.11, 0.325])
    assert np.allclose(point, [0.03, 0.05, 0.125])
    assert np.allclose(normal, [-1.0, 0.0, 0.0])


@pytest.mark.parametrize("dimension", [2, 3])
def test_solid_cell_retains_physical_plane_for_solid_nodes(dimension):
    boundary = BoundaryConstraints()
    simulation = SimpleNamespace(dimension=dimension, domain=np.ones(dimension))
    element = SimpleNamespace(grid_size=np.full(dimension, 0.1), ghost_cell=1)
    start, end, normal = np.zeros(dimension), np.ones(dimension), np.zeros(dimension)
    start[-1] = end[-1] = 0.1
    normal[-1] = -1.0
    boundary.set_solid_cell_boundary(
        simulation, element, {"BoundaryType": "SolidCell", "Norm": normal, "CellThickness": 1}, start, end
    )
    lower, upper, point, direction = boundary.solid_cell_node_planes[0]
    np.testing.assert_allclose(point, start)
    np.testing.assert_array_equal(direction, normal)
    np.testing.assert_allclose(lower[-1], 0.0)
    np.testing.assert_allclose(upper[-1], 0.1)
    assert not boundary.solid_cell_plane_regions
