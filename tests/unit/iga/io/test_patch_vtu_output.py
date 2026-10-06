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
def test_sampled_grid_vtu_round_trip(tmp_path, dimension, resolution, expected_connectivity, vtk_cell_type):
    indices = np.indices(resolution, dtype=np.float64)
    sampled_points = np.stack(tuple(indices[d] for d in range(dimension)), axis=-1)
    displacement = sampled_points + np.arange(1, dimension + 1, dtype=np.float64)
    stress = sum((10.0**d) * indices[d] for d in range(dimension))

    points, connectivity, offsets, cell_types, point_data = _sampled_grid_vtu_data(sampled_points, displacement, stress)
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
    expected_points = np.column_stack(tuple(sampled_points[..., d].reshape(-1, order="F") for d in range(dimension)))
    if dimension == 2:
        expected_points = np.column_stack((expected_points, np.zeros(expected_points.shape[0])))
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
    assert np.allclose(mesh.point_data["stress"], stress.reshape(-1, order="F"))


def test_patch_visualize_writes_one_vtu_per_frame(tmp_path):
    from types import SimpleNamespace
    from src.iga.elements.Patch import Patch
    from src.nurbs.BasicSurface import Rectangle

    rectangle = Rectangle()
    rectangle.set_parameters(start_point=[0.0, 0.0], size=[1.0, 0.5])
    rectangle.generate_knot_u(degree=1, num_ctrlpts=2)
    rectangle.generate_knot_v(degree=1, num_ctrlpts=2)
    rectangle.generate_ctrlpts()
    rectangle.generate_weights()

    def field(array):
        return SimpleNamespace(to_numpy=lambda: array.copy())

    patch = SimpleNamespace(
        current_print=0,
        primitive=SimpleNamespace(body={"test": {"primitive": rectangle}}),
        num_ctrlpts=np.array([[0, 0], [2, 2]], dtype=np.int32),
        control_points=field(rectangle.control_points + [0.1, 0.2]),
        stress=field(np.ones(4)),
    )
    Patch.visualize(patch, str(tmp_path), res=3)
    assert patch.current_print == 1
    assert {p.name for p in tmp_path.iterdir() if not p.name.startswith("._")} == {"NurbsVolumetest000000.vtu"}
    mesh = pv.read(tmp_path / "NurbsVolumetest000000.vtu")
    assert mesh.n_points == 9 and mesh.n_cells == 4
    np.testing.assert_allclose(mesh.point_data["displacement"], np.tile([0.1, 0.2, 0.0], (9, 1)))
    np.testing.assert_allclose(mesh.point_data["stress"], 1.0)
