"""Evaluation and postprocessing for DoublePointConsolidation3D."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import numpy as np

from examples.mmpm.Consolidation.DoublePointConsolidation3D.DoublePointConsolidation3D_parameters import (
    COLUMN_DEPTH,
    COLUMN_HEIGHT,
    COLUMN_ORIGIN,
    COLUMN_WIDTH,
    DT,
    FLUID_DENSITY,
    GRAVITY,
    PERMEABILITY,
    POISSON_RATIO,
    PROFILE_BINS,
    SAVE_INTERVAL,
    SAVE_PATH,
    SIMULATION_TIME,
    SURCHARGE,
    YOUNG_MODULUS,
)


def terzaghi_series(depth_from_top, time_value, height, load, cv, terms=400):
    series = np.zeros_like(depth_from_top, dtype=np.float64)
    for m in range(terms):
        n = 2 * m + 1
        series += (
            4.0
            / (n * np.pi)
            * np.sin(n * np.pi * depth_from_top / (2.0 * height))
            * np.exp(-(n * n) * np.pi * np.pi * cv * time_value / (4.0 * height * height))
        )
    return load * series


def consolidation_coefficient():
    constrained = YOUNG_MODULUS * (1.0 - POISSON_RATIO) / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO))
    return PERMEABILITY * constrained / (GRAVITY * FLUID_DENSITY)


def average_pressure_profile(position, pressure, phase, time_value, cv, n_bins=PROFILE_BINS):
    fluid_mask = phase == 2
    position = position[fluid_mask]
    pressure = pressure[fluid_mask]
    column_min = COLUMN_ORIGIN
    column_max = COLUMN_ORIGIN + np.array([COLUMN_WIDTH, COLUMN_DEPTH, COLUMN_HEIGHT], dtype=np.float64)
    column_bottom = COLUMN_ORIGIN[2]
    column_top = column_bottom + COLUMN_HEIGHT
    depth = column_top - position[:, 2]
    valid = (
        (position[:, 0] >= column_min[0])
        & (position[:, 0] <= column_max[0])
        & (position[:, 1] >= column_min[1])
        & (position[:, 1] <= column_max[1])
        & (depth >= 0.0)
        & (depth <= COLUMN_HEIGHT)
        & np.isfinite(pressure)
    )
    depth = depth[valid]
    pressure = pressure[valid]

    bins = np.linspace(0.0, COLUMN_HEIGHT, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    values = np.zeros(n_bins, dtype=np.float64)
    analytical = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int32)
    ids = np.clip(np.digitize(depth, bins) - 1, 0, n_bins - 1)
    particle_theory = terzaghi_series(depth, time_value, COLUMN_HEIGHT, SURCHARGE, cv)
    for pid, bid in enumerate(ids):
        values[bid] += pressure[pid]
        analytical[bid] += particle_theory[pid]
        counts[bid] += 1
    mask = counts > 0
    values[mask] /= counts[mask]
    analytical[mask] /= counts[mask]
    return centers, values, analytical, mask


def postprocess_consolidation(save_path=SAVE_PATH):
    particle_dir = os.path.join(save_path, "particles")
    file_names = sorted(
        name for name in os.listdir(particle_dir) if name.startswith("MPMParticle") and name.endswith(".npz")
    )
    expected_save_count = int(round(SIMULATION_TIME / SAVE_INTERVAL))
    file_names = file_names[: expected_save_count + 1]
    if len(file_names) < 2:
        raise RuntimeError(f"Not enough particle files in {particle_dir}")

    cv = consolidation_coefficient()
    errors = []
    profiles = []
    times = []
    for save_id, file_name in enumerate(file_names[1:], start=1):
        data = np.load(os.path.join(particle_dir, file_name))
        time_value = float(data["t_current"]) if "t_current" in data.files else save_id * SAVE_INTERVAL
        depth, numerical, analytical, mask = average_pressure_profile(
            data["position"],
            data["pressure"],
            data["phase"],
            time_value,
            cv,
        )
        if not np.any(mask):
            raise RuntimeError(f"No valid fluid pressure bins found in {file_name}")
        abs_err = np.linalg.norm(numerical[mask] - analytical[mask])
        rel_err = abs_err / max(np.linalg.norm(analytical[mask]), 1.0e-12)
        profiles.append(np.column_stack([depth, numerical, analytical]))
        times.append(time_value)
        errors.append([save_id, time_value, cv * time_value / (COLUMN_HEIGHT * COLUMN_HEIGHT), abs_err, rel_err])
        print(
            f"save={save_id:03d}, time={time_value:10.4e}, "
            f"Tv={cv * time_value / (COLUMN_HEIGHT * COLUMN_HEIGHT):8.4f}, "
            f"relative_profile_error={rel_err:10.4e}"
        )

    np.savez(
        os.path.join(save_path, "terzaghi_profiles_3d.npz"),
        profiles=np.array(profiles, dtype=object),
        times=np.array(times, dtype=np.float64),
        dt=DT,
        cv=cv,
        height=COLUMN_HEIGHT,
        surcharge=SURCHARGE,
        column_width=COLUMN_WIDTH,
        column_depth=COLUMN_DEPTH,
    )
    np.savetxt(
        os.path.join(save_path, "terzaghi_profile_errors_3d.csv"),
        np.asarray(errors, dtype=np.float64),
        delimiter=",",
        header="save_id,time_s,Tv,abs_profile_error,relative_profile_error",
        comments="",
    )
