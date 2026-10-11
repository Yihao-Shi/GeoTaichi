"""Run directly: python tests/unit/tools/test_blender_cfdem_gif.py."""

from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.blender_cfdem_gif import (
    close_enclosed_voids,
    grain_surface,
    water_surface,
)
from tools.wavy_plate_gif import particle_displacement, render as render_iga_gallery

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import incompressible_gifs
import mmpm_gifs
import fedem_gifs
import two_phase_3d_gifs

with patch.dict(sys.modules, {"paraview.simple": MagicMock()}):
    from tools.vtu2gif import symmetric_color_range


class GeometryCheck(unittest.TestCase):
    def test_blender_fluid_preparation_needs_no_pressure_and_keeps_single_color(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / "particles").mkdir(parents=True)
            (source / "grids").mkdir()
            np.savez(source / "particles/MPMParticle000000.npz", t_current=0.0)
            np.savez(source / "grids/MPMGrid000000.npz", t_current=0.0)
            case = (source, "tank.gif", "blender", [1, 1, 1], "Tank")
            vertices = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 0, 1.0]])
            folder = Path(temporary) / "render"
            with (
                patch.dict(incompressible_gifs.CASES, {"large_tank": case}),
                patch.object(incompressible_gifs, "water_surface", return_value=(vertices, np.array([[0, 1, 2]]))),
            ):
                _, config = incompressible_gifs.prepare("large_tank", folder, True)
            self.assertNotIn("water_scalar", config)
            self.assertEqual(config["water_color"], [0.06, 0.38, 0.66])
            self.assertEqual(config["water_visibility"], 0.72)
            self.assertEqual(config["water_transmission"], 0.55)
            with np.load(folder / config["frames"][0]["mesh"]) as data:
                self.assertNotIn("water_pressure", data)

    @unittest.skipUnless(importlib.util.find_spec("vtk"), "VTK is not installed")
    def test_axisymmetric_cpt_uses_displacement_and_independent_color_maps(self):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / "vtks").mkdir(parents=True)
            (source / "parameters.json").write_text(json.dumps({"save_interval": 0.2}))
            (source / "cpt_summary.json").write_text(json.dumps({"axisymmetric": True}))
            for step in range(2):
                for name in ("NurbsVolumecpt_penetrator", "GraphicMPMParticle"):
                    grid = vtk.vtkUnstructuredGrid()
                    points = vtk.vtkPoints()
                    points.SetData(
                        numpy_to_vtk(np.array([[0.0, step * 0.1, 0.0], [1.0, 1.0 + step * 0.1, 0.0]]), deep=True)
                    )
                    grid.SetPoints(points)
                    values = (
                        np.tile([0.0, step * 0.2, 0.0], (2, 1)) if name.startswith("Nurbs") else np.zeros(2, dtype=int)
                    )
                    array = numpy_to_vtk(values, deep=True)
                    array.SetName("displacement" if name.startswith("Nurbs") else "bodyID")
                    grid.GetPointData().AddArray(array)
                    writer = vtk.vtkXMLUnstructuredGridWriter()
                    writer.SetFileName(str(source / "vtks" / f"{name}{step:06d}.vtu"))
                    writer.SetInputData(grid)
                    self.assertEqual(writer.Write(), 1)
            pv = MagicMock()
            pv.Show.side_effect = lambda *args, **kwargs: MagicMock()
            pv.GetColorTransferFunction.side_effect = lambda *args, **kwargs: MagicMock()
            folder = Path(temporary) / "render"
            with patch.dict(sys.modules, {"paraview": MagicMock(simple=pv)}):
                render_iga_gallery(source, folder, folder / "cpt.gif", 10, True)
            config = json.loads((folder / "manifest.json").read_text())
            self.assertTrue(config["axisymmetric"])
            self.assertEqual(config["dimension"], 2)
            self.assertEqual(config["camera_direction"], [0, 0, 1])
            self.assertEqual(config["iga_scalar"], "displacement")
            np.testing.assert_allclose(config["iga_color_range"], [0, 0.2])
            np.testing.assert_allclose(config["mpm_displacement_range"], [0, 0.1])
            self.assertEqual(pv.GetColorTransferFunction.call_count, 2)
            for call in pv.GetColorTransferFunction.call_args_list:
                self.assertEqual(call.args[0], "displacement")
                self.assertTrue(call.kwargs["separate"])
            self.assertIsNot(
                pv.GetColorTransferFunction.call_args_list[0].args[1],
                pv.GetColorTransferFunction.call_args_list[1].args[1],
            )

    def test_mpm_displacement_is_position_change_from_frame_zero(self):
        initial = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
        current = initial + [[0, 3, 4], [-1, 0, 0]]
        ids = np.array([0, 1])
        values = particle_displacement(current, initial, ids, ids)
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), [5, 1])
        np.testing.assert_array_equal(particle_displacement(initial, initial, ids, ids), np.zeros((2, 3)))
        for positions, body_ids in ((current[:1], ids[:1]), (current, ids[::-1])):
            with self.assertRaises(ValueError):
                particle_displacement(positions, initial, body_ids, ids)

    def test_particle_surface_void_closure_preserves_outer_air(self):
        sdf = np.ones((7, 7, 7), dtype=np.float32)
        sdf[1:6, 1:6, 1:6] = -1
        sdf[3, 3, 3] = 0.2
        repaired = close_enclosed_voids(sdf, 0.1)
        self.assertLess(repaired[3, 3, 3], 0)
        np.testing.assert_array_equal(repaired[[0, -1]], sdf[[0, -1]])

    def test_3d_two_phase_render_uses_saved_solid_volume_and_separates_water(self):
        fluid = np.stack(np.meshgrid(*([np.arange(0.25, 0.85, 0.1)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
        positions = np.vstack((fluid, [1.4, 0.5, 0.5], [np.nan] * 3))
        particles = {
            "position": positions,
            "volume": np.full(len(positions), 0.001),
            "active": np.r_[np.ones(len(fluid) + 1), 0],
            "phase": np.r_[np.full(len(fluid), 2), 1, 2],
            "porosity": np.full(len(positions), 0.4),
        }
        coords = np.stack(np.meshgrid(*([np.arange(21) * 0.1] * 3), indexing="ij"), axis=-1)
        grid = {"dims": np.array([21] * 3), "coords": coords.reshape(-1, 3), "cell_type": np.zeros((20, 20, 20, 1))}
        first = two_phase_3d_gifs.phase_geometry(particles, grid, [2] * 3, 2)
        np.testing.assert_allclose(first["solid_positions"], [[2.8, 1, 1]])
        np.testing.assert_allclose(first["solid_radii"], [2 * np.cbrt(3 * 0.0006 / (4 * np.pi))])
        particles["position"][len(fluid), 0] += 0.1
        moved = two_phase_3d_gifs.phase_geometry(particles, grid, [2] * 3, 2)
        np.testing.assert_array_equal(first["water"], moved["water"])
        particles["porosity"][len(fluid)] = 1
        with self.assertRaises(ValueError):
            two_phase_3d_gifs.phase_geometry(particles, grid, [2] * 3, 2)

    def test_fedem_wall_edges_omit_planar_triangle_diagonals(self):
        vertices, faces = incompressible_gifs.box_mesh([0, 0, 0], [1, 1, 1])
        segments = fedem_gifs.wall_segments(vertices, faces)
        self.assertEqual(segments.shape, (12, 2, 3))
        np.testing.assert_allclose(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1), 1)

    def test_fedem_stress_colorbar_caps_global_maximum_at_seventy_percent(self):
        np.testing.assert_allclose(fedem_gifs.stress_color_range(20000), [0, 14000])
        np.testing.assert_allclose(fedem_gifs.stress_color_range(144384, 50000), [0, 50000])
        for maximum in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                fedem_gifs.stress_color_range(maximum)
            with self.assertRaises(ValueError):
                fedem_gifs.stress_color_range(20000, maximum)

    def test_irregular_dam_obstacle_is_extruded_without_changing_its_outline(self):
        outline = incompressible_gifs.DAM_OBSTACLE_VERTICES
        vertices, faces = incompressible_gifs.polygon_prism_mesh(outline, 0.06)
        np.testing.assert_allclose(vertices[: len(outline), ::2], outline)
        np.testing.assert_allclose(np.unique(vertices[:, 1]), [0.0, 0.06])
        self.assertEqual(faces.shape, (4 * len(outline), 3))
        cap = faces[np.all(vertices[faces, 1] == 0.0, axis=1)]
        triangles = vertices[cap][:, :, ::2]
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        cap_area = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]).sum()
        following = np.roll(outline, -1, axis=0)
        polygon_area = 0.5 * abs(np.sum(outline[:, 0] * following[:, 1] - outline[:, 1] * following[:, 0]))
        self.assertAlmostEqual(cap_area, polygon_area)

    def test_compaction_times_follow_saved_schedule_not_uniform_animation_frames(self):
        config = {
            "loading_protocol": {"compression_step_count": 901},
            "execution": {"command": ["run", "--snapshot-count", "10"]},
            "parameters": {"effective_dt": 0.01},
        }
        metrics = {"compression_step_count": 250, "final": {"time": 3.5}}
        np.testing.assert_allclose(fedem_gifs.compaction_times(config, metrics, 5, 1), [1, 1.01, 2.01, 3.01, 3.5])
        with self.assertRaises(ValueError):
            fedem_gifs.compaction_times(config, metrics, 4, 1)
        np.testing.assert_allclose(
            fedem_gifs.compaction_times(config, metrics, 6, 0, compression_start=1), [0, 1, 1.01, 2.01, 3.01, 3.5]
        )

    def test_two_phase_render_keeps_solid_particles_out_of_water_reconstruction(self):
        fluid = np.stack(
            np.meshgrid(np.arange(0.25, 0.85, 0.1), np.arange(0.25, 0.85, 0.1), indexing="ij"), axis=-1
        ).reshape(-1, 2)
        positions = np.vstack((fluid, [1.4, 0.5], [np.nan, np.nan]))
        particles = {
            "position": positions,
            "volume": np.full(len(positions), 0.01),
            "active": np.r_[np.ones(len(fluid) + 1), 0],
            "phase": np.r_[np.full(len(fluid), 2), 1, 2],
            "porosity": np.full(len(positions), 0.4),
        }
        grid = mmpm_gifs.reconstruction_grid("porous_dam", [2, 2], 0.1)
        first = mmpm_gifs.phase_geometry(particles, grid, [2, 2], 0.2, 2)
        np.testing.assert_allclose(first["solid_positions"], [[2.8, 0, 1.0]])
        np.testing.assert_allclose(first["solid_radii"], [2 * np.sqrt(0.006 / np.pi)])
        particles["position"][len(fluid)] += [0.1, 0]
        moved = mmpm_gifs.phase_geometry(particles, grid, [2, 2], 0.2, 2)
        np.testing.assert_array_equal(first["water"], moved["water"])
        np.testing.assert_allclose(moved["solid_positions"] - first["solid_positions"], [[0.2, 0, 0]])
        particles["volume"][0] = -1
        with self.assertRaises(ValueError):
            mmpm_gifs.phase_geometry(particles, grid, [2, 2], 0.2, 2)

    def test_pressure_percentiles_ignore_inactive_values_and_keep_zero_centered(self):
        pressure = np.r_[np.linspace(-1, 1, 1000), -100, 100, np.nan]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / "particles").mkdir(parents=True)
            np.savez(
                source / "particles/MPMParticle000000.npz",
                pressure=pressure,
                active=np.r_[np.ones(1002), 0],
                t_current=0,
            )
            np.savez(
                source / "particles/MPMParticle000001.npz",
                pressure=pressure * 0.001,
                active=np.r_[np.ones(1002), 0],
                t_current=1,
            )
            for case in ("lid_n64", "cylinder", "taylor_green"):
                settings = (source, "check.gif", "pressure", [1, 1], "check")
                with patch.dict(incompressible_gifs.CASES, {case: settings}):
                    _, config = incompressible_gifs.prepare(case, Path(temporary) / case)
                low, high = config["color_range"]
                self.assertAlmostEqual(low, -high)
                self.assertLess(high, 1)
                if case == "cylinder":
                    self.assertEqual((low, high), (-0.6, 0.6))
                else:
                    self.assertGreater(high, 0.9)
                self.assertEqual(config["color_range_mode"], "fixed")
                self.assertEqual(len(config["frames"]), 2)

    def test_frame_pressure_color_range_tracks_decay_and_handles_zero(self):
        self.assertEqual(symmetric_color_range((-0.3, 0.4), (-1, 1)), (-0.4, 0.4))
        self.assertEqual(symmetric_color_range((-0.003, 0.004), (-1, 1)), (-0.004, 0.004))
        self.assertEqual(symmetric_color_range((0, 0), (-0.2, 0.2)), (-0.2, 0.2))
        with self.assertRaises(ValueError):
            symmetric_color_range((float("nan"), 1), (-1, 1))

    def test_cell_centers_and_outward_water_normals(self):
        coords = np.stack(np.meshgrid(*([np.arange(-5, 8)] * 3), indexing="ij"), axis=-1)
        positions = np.stack(np.meshgrid(*([[-0.5, 0.5]] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
        vertices, faces = water_surface(
            {"dims": np.array([13, 13, 13]), "coords": coords.reshape(-1, 3), "cell_type": np.zeros((12, 12, 12, 1))},
            {"position": positions, "volume": np.ones(8), "active": np.ones(8)},
        )
        np.testing.assert_allclose(vertices.min(0), -vertices.max(0), atol=1e-6)
        triangles = vertices[faces]
        volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
        self.assertGreater(volume, 0)

    def test_shared_template_scale_rotation_and_translation(self):
        surface = {
            "master": np.array([0, 0, 0, 1, 1, 1]),
            "vertices": np.eye(3),
            "connectivity": np.array([[0, 1, 2], [3, 4, 5]]),
        }
        rigid = {
            "startNode": np.array([0, 3]),
            "localNode": np.array([0, 0]),
            "scale": np.array([1, 2]),
            "mass_center": np.array([[0, 0, 0], [10, 0, 0]]),
            "quanternion": np.array([[0, 0, 0, 1], [0, 0, 2**-0.5, 2**-0.5]]),
        }
        vertices, _ = grain_surface(surface, rigid)
        np.testing.assert_allclose(vertices[:3], np.eye(3))
        np.testing.assert_allclose(vertices[3:], [[10, 2, 0], [8, 0, 0], [10, 0, 2]], atol=1e-14)

    def test_particles_move_water_despite_a_stale_grid_sdf(self):
        coords = np.stack(np.meshgrid(*([np.arange(-5, 8)] * 3), indexing="ij"), axis=-1)
        grid = {
            "dims": np.array([13, 13, 13]),
            "coords": coords.reshape(-1, 3),
            "cell_fluid_sdf": np.ones((12, 12, 12, 1)),
            "cell_type": np.zeros((12, 12, 12, 1)),
        }
        positions = np.stack(np.meshgrid(*([[-0.5, 0.5]] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
        particles = {"position": positions, "volume": np.ones(8), "active": np.ones(8)}
        first, _ = water_surface(grid, particles)
        particles["position"] = positions + [1, 0, 0]
        moved, _ = water_surface(grid, particles)
        np.testing.assert_allclose(moved.min(0) - first.min(0), [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(moved.max(0) - first.max(0), [1, 0, 0], atol=1e-6)

    def test_subcell_particle_motion_does_not_snap_water_to_histogram_bins(self):
        coords = np.stack(np.meshgrid(*([np.arange(-5, 8)] * 3), indexing="ij"), axis=-1)
        grid = {"dims": np.array([13, 13, 13]), "coords": coords.reshape(-1, 3), "cell_type": np.zeros((12, 12, 12, 1))}
        positions = np.stack(np.meshgrid(*([[-0.5, 0.5]] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
        particles = {"position": positions, "volume": np.ones(8), "active": np.ones(8)}
        first, _ = water_surface(grid, particles)
        particles["position"] = positions + [0.2, 0, 0]
        moved, _ = water_surface(grid, particles)
        shift = (moved.min(0) + moved.max(0) - first.min(0) - first.max(0)) / 2
        self.assertGreater(shift[0], 0.1)
        self.assertLess(shift[0], 0.3)
        np.testing.assert_allclose(shift[1:], 0, atol=1e-6)

    def test_2d_extrusion_maps_vertical_axis_and_keeps_outward_normals(self):
        coords = np.stack(np.meshgrid(*([np.arange(-5, 8)] * 2), indexing="ij"), axis=-1)
        positions = np.array([[-0.5, 1.5], [-0.5, 2.5], [0.5, 1.5], [0.5, 2.5]])
        vertices, faces = water_surface(
            {"dims": np.array([13, 13]), "coords": coords.reshape(-1, 2), "cell_type": np.zeros((12, 12, 1))},
            {"position": positions, "volume": np.ones(4), "active": np.ones(4)},
            depth=0.3,
        )
        self.assertGreaterEqual(vertices[:, 1].min(), 0)
        self.assertLessEqual(vertices[:, 1].max(), 0.3)
        self.assertAlmostEqual((vertices[:, 2].min() + vertices[:, 2].max()) / 2, 2, places=6)
        triangles = vertices[faces]
        self.assertGreater(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum(), 0)


if __name__ == "__main__":
    unittest.main()
