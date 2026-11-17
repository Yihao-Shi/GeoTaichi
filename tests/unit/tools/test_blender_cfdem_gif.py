"""Gallery geometry and saved-frame tests; run with pytest."""

from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from tools.blender_cfdem_gif import (
    close_enclosed_voids,
    close_surface_holes,
    export,
    grain_surface,
    water_surface,
)
from tools.wavy_plate_gif import particle_displacement, render as render_iga_gallery

from tools import incompressible_gifs, mmpm_gifs, fedem_gifs, two_phase_3d_gifs

with patch.dict(sys.modules, {"paraview.simple": MagicMock()}):
    from tools.vtu2gif import magnitude_color_range, symmetric_color_range


class GeometryCheck(unittest.TestCase):
    def test_export_ignores_cached_pngs_after_the_selected_interval(self):
        import shutil
        from PIL import Image

        if not shutil.which("ffmpeg"):
            self.skipTest("FFmpeg is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "render"
            folder.mkdir()
            for index, color in enumerate(("white", "blue", "red")):
                Image.new("RGB", (16, 16), color).save(folder / f"frame_{index:06d}.png")
            output = Path(temporary) / "selected.gif"
            export(dict(output=str(output), frames=[{}, {}], fps=10), folder)
            with Image.open(output) as gif:
                self.assertEqual(gif.n_frames, 2)

    def test_porous_dam_gallery_uses_first_hundred_saved_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / "particles").mkdir(parents=True)
            for index in range(101):
                np.savez(source / f"particles/MPMParticle{index:06d}.npz", t_current=index * 0.2)
            with patch.object(mmpm_gifs, "phase_geometry", return_value=dict(solid_positions=np.zeros((1, 3)))):
                _, config = mmpm_gifs.prepare("porous_dam", Path(temporary) / "render", source=source)
            self.assertEqual(len(config["frames"]), 100)
            self.assertEqual(config["frames"][0]["step"], "000000")
            self.assertEqual(config["frames"][-1]["step"], "000099")
            self.assertAlmostEqual(config["frames"][-1]["time"], 19.8)

    def test_missing_triangle_is_repaired_without_moving_vertices_or_existing_faces(self):
        import trimesh

        original = trimesh.creation.icosphere(subdivisions=1)
        faces = original.faces[1:].copy()
        vertices, repaired = close_surface_holes(original.vertices.copy(), faces)
        np.testing.assert_array_equal(vertices, original.vertices)
        np.testing.assert_array_equal(repaired[: len(faces)], faces)
        result = trimesh.Trimesh(vertices, repaired, process=False)
        self.assertTrue(result.is_watertight)
        self.assertAlmostEqual(result.volume, original.volume)

    def test_closed_surfaces_touching_along_an_edge_are_preserved(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1]], dtype=float)
        tetra = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        faces = np.vstack((tetra, np.array([0, 1, 4, 5])[tetra]))
        actual_vertices, actual_faces = close_surface_holes(vertices, faces)
        np.testing.assert_array_equal(actual_vertices, vertices)
        np.testing.assert_array_equal(actual_faces, faces)

    def test_water_touching_domain_walls_is_closed_and_keeps_a_resolved_air_channel(self):
        import trimesh

        coords = np.stack(np.meshgrid(*([np.arange(11) * 0.1] * 3), indexing="ij"), axis=-1)
        centers = coords[:-1, :-1, :-1] + 0.05
        positions = np.stack(
            np.meshgrid(
                np.arange(0.025, 1, 0.05), np.arange(0.025, 1, 0.05), np.arange(0.125, 0.65, 0.05), indexing="ij"
            ),
            axis=-1,
        ).reshape(-1, 3)
        positions = positions[~((abs(positions[:, 0] - 0.5) < 0.15) & (abs(positions[:, 1] - 0.5) < 0.15))]
        grid = {
            "dims": np.array([11] * 3),
            "coords": coords.reshape(-1, 3),
            "cell_type": np.zeros((10, 10, 10, 1)),
            "cell_solid_sdf": (centers[..., 2] - 0.1)[..., None],
        }
        particles = {
            "position": positions.copy(),
            "volume": np.full(len(positions), 0.05**3),
            "active": np.ones(len(positions)),
        }
        vertices, faces = water_surface(grid, particles, occupancy_threshold=0.35)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        self.assertTrue(mesh.is_watertight)
        np.testing.assert_array_equal(mesh.contains([[0.5, 0.5, 0.4], [0.05, 0.5, 0.4]]), [False, True])
        np.testing.assert_allclose(mesh.bounds[:, :2], [[0, 0], [1, 1]], atol=1e-6)
        self.assertGreaterEqual(vertices[:, 2].min(), 0.1 - 1e-6)
        np.testing.assert_array_equal(particles["position"], positions)
        particles.update(wet_soil_position=positions.copy(), wet_soil_volume=particles["volume"].copy())
        supported_vertices, supported_faces = water_surface(grid, particles, occupancy_threshold=0.35)
        np.testing.assert_array_equal(supported_vertices, vertices)
        np.testing.assert_array_equal(supported_faces, faces)

    def test_lsdem_wavemaker_outline_uses_inner_walls_and_leaves_the_top_open(self):
        dims = np.array([61, 13, 19])
        coords = np.stack(np.meshgrid(*[np.arange(n) * 0.02 for n in dims], indexing="ij"), axis=-1)
        grid = {"dims": dims, "coords": coords.reshape(-1, 3), "cell_type": np.zeros((60, 12, 18, 1))}
        clipped, segments = two_phase_3d_gifs.wavemaker_wall_geometry(grid, [1.2, 0.24, 0.36], 0.08)
        np.testing.assert_allclose(segments.min((0, 1)), [0.08, 0.02, 0.02])
        np.testing.assert_allclose(segments.max((0, 1)), [1.18, 0.22, 0.36])
        self.assertEqual(len(segments), 12)
        sdf = clipped["cell_solid_sdf"][..., 0]
        self.assertLess(sdf[10, 0, 5], 0)
        self.assertLess(sdf[59, 5, 5], 0)
        self.assertGreater(sdf[10, 5, -1], 0)
        self.assertNotIn("cell_solid_sdf", grid)

    def test_exact_wall_planes_close_a_full_tank_without_shrinking_its_walls(self):
        import trimesh

        dims = np.array([11] * 3)
        coords = np.stack(np.meshgrid(*([np.arange(11) * 0.1] * 3), indexing="ij"), axis=-1)
        positions = np.stack(np.meshgrid(*([np.arange(0.025, 1, 0.05)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
        grid = {
            "dims": dims,
            "coords": coords.reshape(-1, 3),
            "cell_type": np.zeros((10, 10, 10, 1)),
            "solid_planes": [[1, 1, 0, 1.4]],
        }
        particles = {
            "position": positions,
            "volume": np.full(len(positions), 0.05**3),
            "active": np.ones(len(positions)),
        }
        vertices, faces = water_surface(grid, particles, occupancy_threshold=0.35)
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        self.assertTrue(mesh.is_watertight)
        self.assertLessEqual((vertices[:, 0] + vertices[:, 1]).max(), 1.4 + 1e-6)
        np.testing.assert_allclose(mesh.bounds, [[0, 0, 0], [1, 1, 1]], atol=1e-6)
        grid["solid_planes"] = [[0, 0, 0, 1]]
        with self.assertRaisesRegex(ValueError, "Solid planes"):
            water_surface(grid, particles)

    def test_continuum_soil_uses_saved_bulk_volume_and_does_not_add_water(self):
        positions = np.stack(np.meshgrid(*([np.arange(0.25, 0.85, 0.1)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
        count = len(positions)
        particles = {
            "position": np.vstack((positions, positions)),
            "volume": np.full(2 * count, 0.001),
            "active": np.ones(2 * count),
            "phase": np.r_[np.full(count, 2), np.full(count, 1)],
            "porosity": np.full(2 * count, 0.4),
        }
        coords = np.stack(np.meshgrid(*([np.arange(21) * 0.1] * 3), indexing="ij"), axis=-1)
        grid = {"dims": np.array([21] * 3), "coords": coords.reshape(-1, 3), "cell_type": np.zeros((20, 20, 20, 1))}
        first = two_phase_3d_gifs.phase_geometry(particles, grid, [2] * 3, 1, solid_surface=True)
        self.assertTrue(np.all(first["soil"].min(0) >= first["water"].min(0)))
        self.assertTrue(np.all(first["soil"].max(0) <= first["water"].max(0)))
        self.assertNotIn("solid_positions", first)
        particles["position"][count:, 0] += 1.0
        moved = two_phase_3d_gifs.phase_geometry(particles, grid, [2] * 3, 1, solid_surface=True)
        np.testing.assert_array_equal(first["water"], moved["water"])
        np.testing.assert_allclose(moved["soil"].mean(0) - first["soil"].mean(0), [1, 0, 0], atol=1e-5)

    def test_impact_outline_is_cylindrical_and_present_in_every_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / "particles").mkdir(parents=True)
            (source / "grids").mkdir()
            coords = np.stack(np.meshgrid(*([np.arange(41) * 0.005] * 3), indexing="ij"), axis=-1)
            for frame, time in enumerate((0.0, 0.12)):
                for name in ("MPMParticle", "LSDEMRigid", "LSDEMSurface"):
                    np.savez(source / f"particles/{name}{frame:06d}.npz", t_current=time)
                np.savez(
                    source / f"grids/MPMGrid{frame:06d}.npz",
                    t_current=time,
                    dims=[41] * 3,
                    coords=coords.reshape(-1, 3),
                )
            (source / "metrics.json").write_text(json.dumps({"passed": True, "final_time_s": 0.12}))
            geometry = {"water": np.array([[0.1, 0.1, 0.15]]), "water_faces": np.empty((0, 3), int)}
            with (
                patch.object(two_phase_3d_gifs, "phase_geometry", side_effect=lambda *args, **kwargs: dict(geometry)),
                patch.object(
                    two_phase_3d_gifs, "grain_surface", return_value=(np.zeros((3, 3)), np.array([[0, 1, 2]]))
                ),
            ):
                folder = Path(temporary) / "prepared"
                _, config = two_phase_3d_gifs.prepare("bunny_impact", folder, source=source)
            self.assertFalse(config["show_tank"])
            with np.load(folder / "mesh_000000.npz") as first, np.load(folder / "mesh_000001.npz") as last:
                np.testing.assert_array_equal(first["wall_segments"], last["wall_segments"])
                self.assertAlmostEqual(last["wall_segments"][..., 2].min() / 18, 0.005)

    def test_export_rejects_black_background_without_replacing_existing_gif(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            output = folder / "gallery.gif"
            output.write_bytes(b"existing verified GIF")
            Image.new("RGB", (16, 16), (160, 160, 160)).save(folder / "frame_000000.png")
            Image.new("RGB", (16, 16), (0, 0, 0)).save(folder / "frame_000001.png")
            with patch("tools.blender_cfdem_gif.subprocess.run") as command:
                with self.assertRaisesRegex(ValueError, "Black studio background in frame 1"):
                    export({"output": str(output), "frames": [{}, {}]}, folder)
                command.assert_not_called()
            self.assertEqual(output.read_bytes(), b"existing verified GIF")

    def test_gallery_rejects_a_restart_tail_without_initial_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "tail"
            (source / "particles").mkdir(parents=True)
            np.savez(source / "particles/MPMParticle000020.npz", t_current=2.0, pressure=[1.0])
            with self.assertRaisesRegex(ValueError, "history starting at time zero"):
                incompressible_gifs.prepare("lid_n64", Path(temporary) / "prepared", source=source)

    def test_leakage_walls_follow_the_model_and_leave_the_slit_open(self):
        dims = np.array([73, 33, 73])
        coords = np.stack(np.meshgrid(*[np.arange(n) * 0.005 for n in dims], indexing="ij"), axis=-1)
        types = np.zeros((*tuple(dims - 1), 1), dtype=np.uint8)
        grid = {"dims": dims, "coords": coords.reshape(-1, 3), "cell_type": types}
        clipped, vertices, faces = two_phase_3d_gifs.leakage_wall_geometry(grid, 0.005)
        sdf = clipped["cell_solid_sdf"][..., 0]
        self.assertLess(sdf[20, 15, 24], 0)
        self.assertGreater(sdf[36, 15, 24], 0)
        self.assertGreater(sdf[20, 15, 40], 0)
        self.assertGreater(sdf[20, 15, 10], 0)
        self.assertLess(sdf[20, 15, 0], 0)
        self.assertTrue(np.isfinite(vertices).all())
        self.assertLess(faces.max(), len(vertices))
        np.testing.assert_array_equal(grid["coords"], coords.reshape(-1, 3))
        np.testing.assert_array_equal(grid["cell_type"], np.zeros_like(types))

    def test_leakage_gallery_needs_no_lsdem_files_and_rejects_short_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / "particles").mkdir(parents=True)
            (source / "grids").mkdir()
            coords = np.stack(np.meshgrid(*([np.arange(3) * 0.005] * 3), indexing="ij"), axis=-1)
            for step, time in enumerate((0.0, 6.0)):
                np.savez(source / f"particles/MPMParticle{step:06d}.npz", t_current=time)
                np.savez(
                    source / f"grids/MPMGrid{step:06d}.npz",
                    t_current=time,
                    dims=[3, 3, 3],
                    coords=coords.reshape(-1, 3),
                    cell_type=np.zeros((2, 2, 2, 1)),
                )
            metrics = {"passed": True, "duration_complete": True, "final_time_s": 6, "represented_crack_width_m": 0.005}
            (source / "metrics.json").write_text(json.dumps(metrics))
            water = {
                "water": np.array([[0.1, 0.08, 0.2], [0.11, 0.08, 0.2], [0.1, 0.08, 0.21]]),
                "water_faces": np.array([[0, 1, 2]]),
            }
            folder = Path(temporary) / "prepared"
            with patch.object(two_phase_3d_gifs, "phase_geometry", return_value=water):
                _, config = two_phase_3d_gifs.prepare("leakage", folder, source=source)
            self.assertNotIn("LSDEM", config["legend"])
            self.assertFalse(config["show_tank"])
            with np.load(folder / "mesh_000001.npz") as data:
                self.assertIn("walls", data)
                self.assertNotIn("rigid", data)
            metrics["final_time_s"] = 0.12
            (source / "metrics.json").write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "six-second"):
                two_phase_3d_gifs.prepare("leakage", Path(temporary) / "short", source=source)

    def test_leakage_reconstructs_outflow_and_receiver_water_as_surfaces(self):
        particles = {
            "position": np.array([[0.18, 0.08, 0.2], [0.18, 0.08, 0.06], [0.1, 0.08, 0.16], [np.nan] * 3]),
            "volume": np.array([2e-8, 1e-8, 2e-8, np.nan]),
            "active": np.array([1, 1, 1, 0]),
            "phase": np.array([2, 2, 1, 2]),
            "porosity": np.array([1, 0.8, 0.4, np.nan]),
        }
        before = particles["position"].copy()
        with patch.object(
            two_phase_3d_gifs, "water_surface", return_value=(np.array([[0.18, 0.08, 0.12]]), np.empty((0, 3), int))
        ) as surface:
            data = two_phase_3d_gifs.phase_geometry(particles, {}, [0.36, 0.16, 0.36], 10)
        np.testing.assert_array_equal(surface.call_args.args[1]["position"], before[:2])
        np.testing.assert_array_equal(surface.call_args.args[1]["volume"], particles["volume"][:2])
        self.assertNotIn("water_positions", data)
        self.assertNotIn("water_radii", data)
        self.assertLess(data["water"][:, 2].max(), 1.25)
        np.testing.assert_array_equal(particles["position"], before)
        particles["position"][0, 2] = 0.04
        with patch.object(
            two_phase_3d_gifs, "water_surface", return_value=(np.array([[0.18, 0.08, 0.12]]), np.empty((0, 3), int))
        ) as surface:
            empty_tank = two_phase_3d_gifs.phase_geometry(particles, {}, [0.36, 0.16, 0.36], 10)
            surface.assert_called_once()
        self.assertEqual(empty_tank["water"].shape, (1, 3))
        self.assertNotIn("water_positions", empty_tank)

    def test_utube_gallery_requires_frames_covering_the_passed_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "saved_50s"
            (source / "particles").mkdir(parents=True)
            for step, time in enumerate((0.0, 50.0)):
                np.savez(source / f"particles/MPMParticle{step:06d}.npz", t_current=time)
            for passed, complete in ((False, True), (True, False)):
                (source / "metrics.json").write_text(
                    json.dumps(
                        {
                            "passed": passed,
                            "duration_complete": complete,
                            "simulation_time_s": 50,
                        }
                    )
                )
                with self.assertRaisesRegex(ValueError, "passed, completed"):
                    mmpm_gifs.prepare("utube", Path(temporary) / "prepared", source=source)
            (source / "metrics.json").write_text(
                json.dumps({"passed": True, "duration_complete": True, "simulation_time_s": 50})
            )
            geometry = {"solid_positions": np.zeros((1, 3))}
            with patch.object(mmpm_gifs, "phase_geometry", return_value=geometry):
                _, config = mmpm_gifs.prepare("utube", Path(temporary) / "prepared", source=source)
                self.assertEqual([frame["time"] for frame in config["frames"]], [0, 50])
                self.assertEqual(config["source"], str(source.resolve()))
                np.savez(source / "particles/MPMParticle000001.npz", t_current=48.0)
                with self.assertRaisesRegex(ValueError, "cover the validated interval"):
                    mmpm_gifs.prepare("utube", Path(temporary) / "short", source=source)

    def test_clustered_quadrature_keeps_bulk_water_and_a_resolved_air_hole(self):
        axes = [np.repeat(0.125 + 0.1 * np.arange(n), 2) + np.tile([-0.002, 0.002], n) for n in (20, 15)]
        positions = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
        grid = mmpm_gifs.reconstruction_grid("porous_dam", [2.4, 2], 0.1)
        for hole in (False, True):
            with self.subTest(hole=hole):
                keep = np.ones(len(positions), dtype=bool)
                if hole:
                    keep &= ~((abs(positions[:, 0] - 1) < 0.22) & (abs(positions[:, 1] - 1) < 0.22))
                particles = {
                    "position": positions[keep],
                    "volume": np.full(keep.sum(), 0.05**2),
                    "active": np.ones(keep.sum(), dtype=bool),
                }
                vertices, faces = water_surface(grid, particles, 0.2)
                triangles = vertices[faces]
                volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
                self.assertAlmostEqual(volume / (0.2 * particles["volume"].sum()), 1, delta=0.05)
                edges = np.unique(
                    np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0
                )
                self.assertEqual(len(vertices) - len(edges) + len(faces), 0 if hole else 2)
                np.testing.assert_array_equal(particles["position"], positions[keep])

    def test_particle_reconstruction_respects_non_aligned_physical_walls(self):
        axes = [np.arange(-1, 12) * 0.1, np.arange(-1, 9) * 0.1]
        coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        centers = coords[:-1, :-1] + 0.05
        x, z = centers[..., 0], centers[..., 1]
        distance = np.minimum.reduce([x - 0.023, 0.83 - x, z - 0.037, 0.71 - z])
        grid = {
            "dims": np.array(coords.shape[:2]),
            "coords": coords.reshape(-1, 2),
            "cell_type": ((distance < 0).astype(np.uint8) * 2)[..., None],
            "cell_solid_sdf": distance[..., None],
        }
        positions = np.stack(
            np.meshgrid(np.arange(0.025, 0.83, 0.05), np.arange(0.0625, 0.52, 0.05), indexing="ij"), axis=-1
        ).reshape(-1, 2)
        particles = {
            "position": positions.copy(),
            "volume": np.full(len(positions), 0.05**2),
            "active": np.ones(len(positions)),
        }
        vertices, _ = water_surface(grid, particles, 0.06)
        self.assertGreaterEqual(vertices[:, 0].min(), 0.023 - 1e-6)
        self.assertLessEqual(vertices[:, 0].max(), 0.83 + 1e-6)
        self.assertGreaterEqual(vertices[:, 2].min(), 0.037 - 1e-6)
        self.assertLessEqual(vertices[:, 2].max(), 0.71 + 1e-6)
        self.assertAlmostEqual(vertices[:, 0].min(), 0.023, places=5)
        np.testing.assert_array_equal(particles["position"], positions)

    def test_porous_gallery_uses_physical_flume_extent(self):
        self.assertEqual(mmpm_gifs.CASES["porous_dam"][3], [0.892, 0.37])
        self.assertEqual(mmpm_gifs.CASES["utube"][3], [3, 3.35])
        grid = mmpm_gifs.reconstruction_grid("porous_dam", [0.892, 0.37], 0.0025)
        coords = grid["coords"].reshape(*grid["dims"], 2)[:-1, :-1] + 0.00125
        phi = grid["cell_solid_sdf"][..., 0]
        self.assertTrue(np.all(phi[coords[..., 0] > 0.892] < 0))
        self.assertTrue(np.all(phi[coords[..., 1] > 0.37] < 0))

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
                expected = {"cylinder": 0.2, "lid_n64": 0.05, "taylor_green": 0.05}[case]
                self.assertEqual((low, high), (-expected, expected))
                self.assertEqual(config["color_range_mode"], "fixed")
                self.assertEqual(len(config["frames"]), 2)

    def test_taylor_green_view_uses_saved_domain_for_one_or_four_vortices(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / "particles").mkdir(parents=True)
            np.savez(source / "particles/MPMParticle000000.npz", velocity=[[0.3, 0.4], [-0.3, -0.4]], t_current=0)
            for extent in (np.pi, 2 * np.pi):
                (source / "metrics.json").write_text(json.dumps({"domain_m": [extent, extent]}))
                _, config = incompressible_gifs.prepare("taylor_green", Path(temporary) / "render", source=source)
                np.testing.assert_allclose(config["domain"], [extent, extent])
                self.assertEqual(config["color_range"], [0, 0.5])
            (source / "metrics.json").write_text(json.dumps({"domain_m": [float("nan"), 1]}))
            with self.assertRaises(ValueError):
                incompressible_gifs.prepare("taylor_green", Path(temporary) / "render", source=source)

    def test_taylor_green_speed_uses_active_velocities_and_tracks_late_decay(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / "particles").mkdir(parents=True)
            for frame, factor in enumerate((1.0, 0.01)):
                np.savez(
                    source / f"particles/MPMParticle{frame:06d}.npz",
                    velocity=np.array([[0, 0], [-3, 4], [np.nan, np.nan]]) * factor,
                    active=[1, 1, 0],
                    t_current=frame,
                )
            _, config = incompressible_gifs.prepare("taylor_green", Path(temporary) / "render", source=source)
            self.assertEqual(config["scalar"], "velocity")
            self.assertEqual(config["scalar_mode"], "magnitude")
            self.assertEqual(config["color_range_mode"], "frame-magnitude")
            self.assertEqual(config["color_range"], [0, 5])
            np.testing.assert_allclose([f["speed_range"] for f in config["frames"]], [[0, 5], [0, 0.05]])
            np.savez(source / "particles/MPMParticle000001.npz", velocity=[[np.nan, 0]], t_current=1)
            with self.assertRaisesRegex(ValueError, "non-finite particle velocity"):
                incompressible_gifs.prepare("taylor_green", Path(temporary) / "invalid", source=source)

    def test_vorticity_has_signed_curl_and_does_not_bridge_a_resolved_cylinder(self):
        positions = np.stack(np.meshgrid(np.linspace(-1, 1, 33), np.linspace(-1, 1, 33)), axis=-1).reshape(-1, 2)
        positions = positions[np.linalg.norm(positions, axis=1) > 0.35]
        for velocity, expected in (
            (np.column_stack((-2 * positions[:, 1], 2 * positions[:, 0])), 4),
            (np.column_stack((3 * positions[:, 1], np.zeros(len(positions)))), -3),
        ):
            particles = dict(
                position=np.vstack((positions, [np.nan, np.nan])),
                velocity=np.vstack((velocity, [np.nan, np.nan])),
                active=np.r_[np.ones(len(positions)), 0],
            )
            points, triangles, curl = incompressible_gifs.particle_vorticity(particles, 1 / 16)
            np.testing.assert_allclose(curl, expected, atol=1e-12)
            self.assertTrue((np.linalg.norm(points[triangles].mean(axis=1), axis=1) > 0.30).all())
        particles["active"][-1] = 1
        with self.assertRaisesRegex(ValueError, "finite 2D particle velocities"):
            incompressible_gifs.particle_vorticity(particles, 1 / 16)

    def test_vorticity_preparation_uses_velocity_without_requiring_pressure(self):
        import meshio

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / "particles").mkdir(parents=True)
            (source / "grids").mkdir()
            coords = np.stack(np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5), indexing="ij"), axis=-1)
            points = coords.reshape(-1, 2)
            np.savez(
                source / "particles/MPMParticle000000.npz",
                t_current=0,
                position=points,
                velocity=np.column_stack((-points[:, 1], points[:, 0])),
                active=np.ones(len(points)),
            )
            grid_path = source / "grids/MPMGrid000000.npz"
            np.savez(grid_path, t_current=0, dims=[5, 5], coords=points)
            _, config = incompressible_gifs.prepare("cylinder_vorticity", Path(temporary) / "render", source=source)
            mesh = meshio.read(Path(config["vtk_source"]) / "GraphicMPMParticle000000.vtu")
            np.testing.assert_allclose(mesh.point_data["vorticity_z"], 2)
            self.assertEqual(config["scalar"], "vorticity_z")
            self.assertEqual(config["color_range"], [-20, 20])
            np.savez(grid_path, t_current=1, dims=[5, 5], coords=points)
            with self.assertRaisesRegex(ValueError, "Velocity/grid time mismatch"):
                incompressible_gifs.prepare("cylinder_vorticity", Path(temporary) / "mismatch", source=source)

    def test_frame_pressure_color_range_tracks_decay_and_handles_zero(self):
        self.assertEqual(symmetric_color_range((-0.3, 0.4), (-1, 1)), (-0.4, 0.4))
        self.assertEqual(symmetric_color_range((-0.003, 0.004), (-1, 1)), (-0.004, 0.004))
        self.assertEqual(symmetric_color_range((0, 0), (-0.2, 0.2)), (-0.2, 0.2))
        with self.assertRaises(ValueError):
            symmetric_color_range((float("nan"), 1), (-1, 1))

    def test_frame_speed_color_range_stays_nonnegative_as_velocity_decays(self):
        self.assertEqual(magnitude_color_range((0.01, 1), (0, 1)), (0, 1))
        self.assertEqual(magnitude_color_range((0.0001, 0.01), (0, 1)), (0, 0.01))
        self.assertEqual(magnitude_color_range((0, 0), (0, 1)), (0, 1))
        for values in ((-1, 1), (float("nan"), 1), None):
            with self.assertRaises(ValueError):
                magnitude_color_range(values, (0, 1))

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
