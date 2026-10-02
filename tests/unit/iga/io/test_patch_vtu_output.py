"""Unit checks for sampled IGA patch VTU output."""

import numpy as np
import pyvista as pv
import pytest

from third_party.pyevtk.hl import unstructuredGridToVTK
from src.iga.elements.Patch import _sampled_grid_vtu_data


@pytest.mark.parametrize(
    ("dimension", "resolution", "expected_connectivity", "vtk_cell_type"),
    (
        (
            2,
            (3, 2),
            np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.int64),
            9,
        ),
        (
            3,
            (2, 3, 2),
            np.array(
                [
                    [0, 1, 3, 2, 6, 7, 9, 8],
                    [2, 3, 5, 4, 8, 9, 11, 10],
                ],
                dtype=np.int64,
            ),
            12,
        ),
    ),
)
def test_sampled_grid_vtu_round_trip(
    tmp_path, dimension, resolution, expected_connectivity, vtk_cell_type
):
    indices = np.indices(resolution, dtype=np.float64)
    sampled_points = np.stack(
        tuple(indices[d] for d in range(dimension)), axis=-1
    )
    displacement = sampled_points + np.arange(
        1, dimension + 1, dtype=np.float64
    )
    stress = sum((10.0 ** d) * indices[d] for d in range(dimension))

    points, connectivity, offsets, cell_types, point_data = (
        _sampled_grid_vtu_data(sampled_points, displacement, stress)
    )
    output = tmp_path / f"sampled_{dimension}d"
    unstructuredGridToVTK(
        str(output),
        *points,
        connectivity,
        offsets,
        cell_types,
        pointData=point_data,
    )

    mesh = pv.read(output.with_suffix(".vtu"))
    expected_points = np.column_stack(
        tuple(
            sampled_points[..., d].reshape(-1, order="F")
            for d in range(dimension)
        )
    )
    if dimension == 2:
        expected_points = np.column_stack(
            (expected_points, np.zeros(expected_points.shape[0]))
        )
    expected_displacement = np.column_stack(point_data["displacement"])

    assert mesh.n_points == int(np.prod(resolution))
    assert mesh.n_cells == expected_connectivity.shape[0]
    assert np.all(mesh.celltypes == vtk_cell_type)
    assert np.array_equal(
        mesh.cell_connectivity.reshape(expected_connectivity.shape),
        expected_connectivity,
    )
    assert np.allclose(mesh.points, expected_points)
    assert np.allclose(mesh.point_data["displacement"], expected_displacement)
    assert np.allclose(
        mesh.point_data["stress"], stress.reshape(-1, order="F")
    )
