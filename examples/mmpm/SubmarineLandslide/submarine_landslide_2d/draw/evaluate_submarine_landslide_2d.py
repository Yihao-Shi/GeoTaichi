"""Evaluation and postprocessing for submarine_landslide_2d."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import numpy as np

from examples.mmpm.SubmarineLandslide.submarine_landslide_2d.submarine_landslide_2d_parameters import (
    FREE_SURFACE_BIN_COUNT,
    SAND_POLYGON,
    SAVE_INTERVAL,
    SAVE_PATH,
    SLOPE_NORMAL,
    SLOPE_POINT,
    TANK_LENGTH,
    WATER_POLYGON,
)


def _latest_particle_files(save_path):
    particle_dir = os.path.join(save_path, "particles")
    if not os.path.isdir(particle_dir):
        return []
    return sorted(
        os.path.join(particle_dir, name)
        for name in os.listdir(particle_dir)
        if name.startswith("MPMParticle") and name.endswith(".npz")
    )


def postprocess_landslide(save_path=SAVE_PATH):
    files = _latest_particle_files(save_path)
    if len(files) < 2:
        return

    diagnostics = []
    free_surface_profiles = []
    surface_edges = np.linspace(0.0, TANK_LENGTH, FREE_SURFACE_BIN_COUNT + 1)
    for save_id, file_name in enumerate(files):
        data = np.load(file_name)
        time_value = float(data["t_current"]) if "t_current" in data.files else save_id * SAVE_INTERVAL
        position = data["position"]
        solid_velocity = data["solid_velocity"]
        phase = data["phase"]
        solid = phase == 1
        fluid = phase == 2

        solid_pos = position[solid]
        solid_vel = solid_velocity[solid]
        if solid_pos.size == 0:
            continue
        solid_speed = np.linalg.norm(solid_vel, axis=1)
        centroid = solid_pos.mean(axis=0)
        front_x = float(np.max(solid_pos[:, 0]))
        max_speed = float(np.max(solid_speed))
        mean_speed = float(np.mean(solid_speed))

        surface = np.full(FREE_SURFACE_BIN_COUNT, -np.inf, dtype=np.float64)
        if np.any(fluid):
            fluid_pos = position[fluid]
            ids = np.clip(
                np.searchsorted(surface_edges, fluid_pos[:, 0], side="right") - 1,
                0,
                FREE_SURFACE_BIN_COUNT - 1,
            )
            np.maximum.at(surface, ids, fluid_pos[:, 1])
        surface[~np.isfinite(surface)] = np.nan
        free_surface_profiles.append(surface)

        diagnostics.append(
            [
                save_id,
                time_value,
                centroid[0],
                centroid[1],
                front_x,
                max_speed,
                mean_speed,
                float(np.min(solid_pos[:, 1])),
                float(np.max(solid_pos[:, 1])),
            ]
        )

    if not diagnostics:
        return

    diagnostics = np.asarray(diagnostics, dtype=np.float64)
    os.makedirs(save_path, exist_ok=True)
    np.savetxt(
        os.path.join(save_path, "landslide_diagnostics.csv"),
        diagnostics,
        delimiter=",",
        header="save_id,time_s,solid_centroid_x,solid_centroid_y,solid_front_x,solid_max_speed,solid_mean_speed,solid_min_y,solid_max_y",
        comments="",
    )
    np.savez(
        os.path.join(save_path, "landslide_diagnostics.npz"),
        diagnostics=diagnostics,
        free_surface_profiles=np.asarray(free_surface_profiles, dtype=np.float64),
        free_surface_x=0.5 * (surface_edges[:-1] + surface_edges[1:]),
        sand_polygon=np.asarray(SAND_POLYGON, dtype=np.float64),
        water_polygon=np.asarray(WATER_POLYGON, dtype=np.float64),
        slope_point=SLOPE_POINT,
        slope_normal=SLOPE_NORMAL,
    )

    final = diagnostics[-1]
    print(
        "landslide diagnostics: "
        f"time={final[1]:.4f}s, solid_front_x={final[4]:.4f}m, "
        f"solid_max_speed={final[5]:.4f}m/s, solid_mean_speed={final[6]:.4f}m/s"
    )
