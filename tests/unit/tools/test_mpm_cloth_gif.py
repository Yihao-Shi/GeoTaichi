"""Run directly: python tests/unit/tools/test_mpm_cloth_gif.py."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from mpm_cloth_gif import prepare


class ClothGalleryCheck(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("vtk"), "VTK is not installed")
    def test_native_geometry_single_color_and_aligned_frames(self):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / "vtks").mkdir(parents=True)
            summary = dict(
                case="ordinary_mpm_cloth_fem_ipc",
                converged=True,
                finite=True,
                completed_time=0.6,
                fem_nodes=3,
                fem_elements=1,
                mpm_particles=2,
            )
            (source / "validation_summary.json").write_text(json.dumps(summary))
            cloth = np.array([[0.0, 0.0, 0.1], [1.0, 0.0, 0.1], [0.0, 1.0, 0.1]])
            positions = np.array([[0.2, 0.2, 0.2], [0.3, 0.2, 0.2]])
            for step in range(2):
                for name, values in (("FEM", cloth), ("GraphicMPMParticle", positions + [0, 0, step * 0.1])):
                    grid = vtk.vtkUnstructuredGrid()
                    points = vtk.vtkPoints()
                    points.SetData(numpy_to_vtk(values, deep=True))
                    grid.SetPoints(points)
                    cell = vtk.vtkTriangle() if name == "FEM" else vtk.vtkVertex()
                    for i in range(cell.GetNumberOfPoints()):
                        cell.GetPointIds().SetId(i, i)
                    grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
                    writer = vtk.vtkXMLUnstructuredGridWriter()
                    writer.SetFileName(str(source / "vtks" / f"{name}{step:06d}.vtu"))
                    writer.SetInputData(grid)
                    self.assertEqual(writer.Write(), 1)
            folder = Path(temporary) / "render"
            _, config = prepare(source, folder)
            self.assertEqual(config["cloth_display"], "single_color_native_mesh")
            self.assertFalse(config["show_tank"])
            self.assertEqual([frame["time"] for frame in config["frames"]], [0.0, 0.6])
            self.assertAlmostEqual(config["particle_marker_radius_m"], 0.045)
            with np.load(folder / config["frames"][1]["mesh"]) as data:
                origin, scale = np.array(config["origin_m"]), config["display_scale"]
                np.testing.assert_allclose(data["soft"] / scale + origin, cloth)
                np.testing.assert_allclose(data["solid_positions"] / scale + origin, positions + [0, 0, 0.1])
                self.assertEqual(data["soft_faces"].shape, (1, 3))
                self.assertEqual(data["solid_radii"].shape, (2,))
            (source / "vtks/GraphicMPMParticle000001.vtu").unlink()
            with self.assertRaises(ValueError):
                prepare(source, folder)


if __name__ == "__main__":
    unittest.main()
