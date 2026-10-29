"""The cloth gallery preserves native geometry and actual recorder step times."""

import json

import numpy as np
import pytest

from tools.fem_iga_gifs import elastic_bar_summary, prepare_cloth


def test_elastic_bar_requires_completed_implicit_result(tmp_path):
    summary = dict(
        solver="implicit", converged=True, dt=0.005, output_interval=2, steps=200, physical_seconds=1.0, frames=101
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    assert elastic_bar_summary(tmp_path) == summary
    for key, value in (
        ("solver", "explicit"),
        ("converged", False),
        ("frames", 100),
        ("physical_seconds", 0.99),
        ("dt", float("nan")),
    ):
        path.write_text(json.dumps(dict(summary, **{key: value})))
        with pytest.raises(ValueError):
            elastic_bar_summary(tmp_path)


def test_cloth_geometry_timing_and_incomplete_run_rejection(tmp_path):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import numpy_to_vtk

    source = tmp_path / "source"
    (source / "vtks").mkdir(parents=True)
    summary = dict(
        case="newton_cloth_twist",
        converged=True,
        finite_positions=True,
        maximum_active_contacts=3,
        nodes=3,
        triangles=1,
        steps=60,
        dt=1 / 60,
        final_time=1,
    )
    summary_path = source / "validation_summary.json"
    summary_path.write_text(json.dumps(summary))
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for step in (0, 30, 60):
        grid = vtk.vtkUnstructuredGrid()
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(vertices + [0, 0, step * 0.001], deep=True))
        grid.SetPoints(points)
        triangle = vtk.vtkTriangle()
        for index in range(3):
            triangle.GetPointIds().SetId(index, index)
        grid.InsertNextCell(triangle.GetCellType(), triangle.GetPointIds())
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(source / "vtks" / f"FEM{step:06d}.vtu"))
        writer.SetInputData(grid)
        assert writer.Write() == 1
    folder = tmp_path / "render"
    config = prepare_cloth(source, folder)
    assert [frame["time"] for frame in config["frames"]] == [0, 0.5, 1]
    assert config["cloth_display"] == "single_color_native_mesh"
    with np.load(folder / config["frames"][-1]["mesh"]) as data:
        np.testing.assert_allclose(data["soft"] / config["display_scale"] + config["origin_m"], vertices + [0, 0, 0.06])
        np.testing.assert_array_equal(data["soft_faces"], [[0, 1, 2]])
    for key, value in (("converged", False), ("maximum_active_contacts", 0), ("final_time", 2)):
        summary_path.write_text(json.dumps(dict(summary, **{key: value})))
        with pytest.raises(ValueError):
            prepare_cloth(source, folder)
    summary_path.write_text(json.dumps(summary))
    (source / "vtks/FEM000060.vtu").unlink()
    with pytest.raises(ValueError, match="cover the completed simulation"):
        prepare_cloth(source, folder)
