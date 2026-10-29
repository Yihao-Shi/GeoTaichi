"""Check that the water-impact display uses synchronized saved DEM state."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools import semi_resolved_gifs


class SavedSphereImpact(unittest.TestCase):
    def source(self, root, sphere_time=0.06, radius=0.0127):
        for part in ("particles", "grids"):
            (root / part).mkdir(exist_ok=True)
        for i, time in enumerate((0.0, 0.06)):
            np.savez(root / f"particles/MPMParticle{i:06d}.npz", t_current=time)
            np.savez(root / f"grids/MPMGrid{i:06d}.npz", t_current=time)
            np.savez(
                root / f"particles/DEMParticle{i:06d}.npz",
                t_current=0 if i == 0 else sphere_time,
                active=[1],
                position=[[0.05, 0.05, 0.1327 - 0.01 * i]],
                radius=[radius],
            )

    def test_saved_sphere_position_and_radius_are_scaled_with_the_water(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.source(root)
            vertices, faces = np.array([[0.01, 0.02, 0.03]]), np.array([[0, 0, 0]])
            with patch.object(semi_resolved_gifs, "water_surface", return_value=(vertices, faces)):
                _, config = semi_resolved_gifs.prepare(root, root / "display")
            self.assertEqual([f["time"] for f in config["frames"]], [0.0, 0.06])
            np.testing.assert_allclose(config["frames"][1]["centers"], [[1.2, 1.2, 0.1227 * 24]])
            self.assertAlmostEqual(config["radius"], 0.0127 * 24)
            with np.load(root / "display/mesh_000001.npz") as z:
                np.testing.assert_allclose(z["water"], vertices * 24)

    def test_mismatched_dem_time_and_invalid_radius_are_rejected(self):
        for sphere_time, radius, error in [
            (0.05, 0.0127, "Mismatched"),
            (np.nan, 0.0127, "Mismatched"),
            (0.06, 0.0, "positive radius"),
        ]:
            with self.subTest(sphere_time=sphere_time, radius=radius), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.source(root, sphere_time, radius)
                with patch.object(
                    semi_resolved_gifs, "water_surface", return_value=(np.zeros((1, 3)), np.zeros((1, 3), dtype=int))
                ):
                    with self.assertRaisesRegex(ValueError, error):
                        semi_resolved_gifs.prepare(root, root / "display")
