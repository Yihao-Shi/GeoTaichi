"""Evaluation and postprocessing for SaturatedSoilColumnCollapseSemiImplicit2D."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glob
import os
import numpy as np

from examples.mmpm.ColumnCollapse.SaturatedSoilColumnCollapseSemiImplicit2D.SaturatedSoilColumnCollapseSemiImplicit2D_parameters import (
    COLUMN,
    DOMAIN,
    SAVE_PATH,
    SIMULATION_TIME,
    SOLVER_TYPE,
)


def postprocess(save_path=SAVE_PATH):
    files = sorted(glob.glob(os.path.join(save_path, "particles", "MPMParticle*.npz")))
    if not files:
        return

    rows = []
    for file_name in files:
        data = np.load(file_name)
        active = data["active"] > 0 if "active" in data.files else np.ones(len(data["position"]), dtype=bool)
        position = data["position"][active]
        solid_velocity = data["solid_velocity"][active]
        fluid_velocity = data["fluid_velocity"][active]
        pressure = data["pressure"][active]
        porosity = data["porosity"][active]
        volume = data["volume"][active]
        fluid_mass = data["fluid_mass"][active]
        rows.append(
            [
                float(data["t_current"]),
                float(position[:, 0].max()),
                float(position[:, 1].max()),
                float(position[:, 0].mean()),
                float(position[:, 1].mean()),
                float(np.linalg.norm(solid_velocity, axis=1).max()),
                float(np.linalg.norm(fluid_velocity, axis=1).max()),
                float(pressure.min()),
                float(pressure.max()),
                float(porosity.min()),
                float(porosity.max()),
                float(volume.sum()),
                float(((1.0 - porosity) * volume).sum()),
                float(fluid_mass.sum()),
                float(np.median(pressure)),
                float(np.mean(pressure > 1.0)),
            ]
        )

    rows = np.asarray(rows)
    np.savetxt(
        os.path.join(save_path, "column_collapse_diagnostics.csv"),
        rows,
        delimiter=",",
        header=(
            "time_s,front_x_m,height_m,centroid_x_m,centroid_y_m,solid_vmax_mps,fluid_vmax_mps,"
            "pressure_min_pa,pressure_max_pa,porosity_min,porosity_max,total_volume_m2"
            ",solid_volume_m2,fluid_mass_kg_per_m,pressure_median_pa,pressure_above_1pa_fraction"
        ),
        comments="",
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(rows[:, 0], rows[:, 1] / COLUMN[0], marker="o", label="front x/L0")
    axes[0].plot(rows[:, 0], rows[:, 2] / COLUMN[1], marker="s", label="top y/H0")
    axes[0].set(xlabel="time (s)", ylabel="normalized coordinate")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(rows[:, 0], rows[:, 7] / 1000.0, label="min")
    axes[1].plot(rows[:, 0], rows[:, 14] / 1000.0, label="median")
    axes[1].plot(rows[:, 0], rows[:, 8] / 1000.0, label="max")
    axes[1].set(xlabel="time (s)", ylabel="pore pressure (kPa)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("Single-point saturated column collapse")
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "column_collapse_diagnostics.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), layout="constrained")
    for axis, target_time in zip(axes.flat, (0.0, 0.005, 0.01, 0.025, 0.1, SIMULATION_TIME)):
        index = int(np.argmin(np.abs(rows[:, 0] - target_time)))
        with np.load(files[index]) as data:
            points = data["position"]
            scatter = axis.scatter(
                points[:, 0],
                points[:, 1],
                c=data["pressure"] / 1000.0,
                s=2,
                vmin=min(float(rows[:, 7].min()) / 1000.0, 0.0),
                vmax=max(float(rows[:, 8].max()) / 1000.0, 1.0e-12),
            )
        axis.set(title=f"t = {rows[index, 0]:.3f} s", xlabel="x (m)", ylabel="y (m)")
        axis.set_xlim(0.0, min(DOMAIN[0], float(rows[:, 1].max()) * 1.05))
        axis.set_ylim(0.0, float(rows[:, 2].max()) * 1.05)
        axis.set_aspect("equal")
    fig.colorbar(scatter, ax=axes, label="pore pressure (kPa)")
    fig.suptitle(f"Saturated column collapse — {SOLVER_TYPE}")
    fig.savefig(os.path.join(save_path, "column_collapse_pressure_snapshots.png"), dpi=180)
    plt.close(fig)
