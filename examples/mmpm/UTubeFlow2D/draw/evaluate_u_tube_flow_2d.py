"""Evaluation and postprocessing for u_tube_flow_2d."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glob
import json
import math
from pathlib import Path
import numpy as np

from examples.mmpm.UTubeFlow2D.u_tube_flow_2d_parameters import (
    DOMAIN,
    DOMAIN_CELLS,
    DT,
    DX,
    EXPECTED_FLUID_PARTICLES,
    EXPECTED_SOLID_PARTICLES,
    HYDRAULIC_CONDUCTIVITY,
    INITIAL_HEAD_DIFFERENCE,
    PHYSICAL_HEIGHT,
    POROSITY,
    POROUS_ORIGIN,
    POROUS_SIZE,
    PPC,
    SIMULATION_TIME,
    WIDTH,
)


def postprocess(output: Path):
    files = sorted(glob.glob(str(output / "particles" / "MPMParticle*.npz")))
    if len(files) < 2:
        raise RuntimeError("U-tube recorder produced fewer than two particle states")

    rows = []
    max_solid_displacement = 0.0
    max_cavity_particles = 0
    initial_solid_position = None
    for file_name in files:
        data = np.load(file_name)
        active = data["active"] > 0
        phase = data["phase"]
        position = data["position"]
        fluid = active & (phase == 2)
        solid = active & (phase == 1)
        left = fluid & (position[:, 0] < POROUS_ORIGIN[0])
        right = fluid & (position[:, 0] > POROUS_ORIGIN[0] + POROUS_SIZE[0])
        cavity = (
            fluid
            & (position[:, 0] >= POROUS_ORIGIN[0])
            & (position[:, 0] <= POROUS_ORIGIN[0] + POROUS_SIZE[0])
            & (position[:, 1] > POROUS_ORIGIN[1] + POROUS_SIZE[1] + DX)
        )
        max_cavity_particles = max(max_cavity_particles, int(np.count_nonzero(cavity)))
        if not np.any(left) or not np.any(right):
            raise RuntimeError(f"lost a water column in {file_name}")
        left_surface = float(np.quantile(position[left, 1], 0.99))
        right_surface = float(np.quantile(position[right, 1], 0.99))
        time = float(data["t_current"])
        numerical_head = left_surface - right_surface
        analytical_head = INITIAL_HEAD_DIFFERENCE * math.exp(-2.0 * HYDRAULIC_CONDUCTIVITY * time / POROUS_SIZE[0])
        if initial_solid_position is None:
            initial_solid_position = position[solid].copy()
        max_solid_displacement = max(
            max_solid_displacement,
            float(np.max(np.linalg.norm(position[solid] - initial_solid_position, axis=1))),
        )
        speed = np.linalg.norm(data["fluid_velocity"][fluid], axis=1)
        rows.append(
            [
                time,
                left_surface,
                right_surface,
                numerical_head,
                analytical_head,
                numerical_head - analytical_head,
                float(np.quantile(speed, 0.99)),
                float(np.min(data["pressure"][fluid])),
                float(np.max(data["pressure"][fluid])),
            ]
        )

    rows = np.asarray(rows, dtype=np.float64)
    np.savetxt(
        output / "head_decay.csv",
        rows,
        delimiter=",",
        header=(
            "time_s,left_surface_m,right_surface_m,numerical_head_difference_m,"
            "analytical_head_difference_m,error_m,fluid_speed_p99_mps,pressure_min_pa,pressure_max_pa"
        ),
        comments="",
    )
    relative_l2 = float(np.linalg.norm(rows[:, 3] - rows[:, 4]) / max(np.linalg.norm(rows[:, 4]), 1.0e-12))
    metrics = {
        "case": "Zhang et al. (2027), Section 4.2, two-dimensional U-tube flow",
        "doi": "10.1016/j.cma.2026.119401",
        "domain_m": DOMAIN,
        "physical_container_height_m": PHYSICAL_HEIGHT,
        "cell_counts": DOMAIN_CELLS,
        "element_size_m": DX,
        "timestep_s": DT,
        "simulation_time_s": SIMULATION_TIME,
        "particle_spacing_m": DX / PPC,
        "fluid_particle_count": EXPECTED_FLUID_PARTICLES,
        "solid_particle_count": EXPECTED_SOLID_PARTICLES,
        "porosity": POROSITY,
        "hydraulic_conductivity_mps": HYDRAULIC_CONDUCTIVITY,
        "initial_head_difference_m": INITIAL_HEAD_DIFFERENCE,
        "analytical_time_scale_s": POROUS_SIZE[0] / (2.0 * HYDRAULIC_CONDUCTIVITY),
        "relative_l2_head_error": relative_l2,
        "final_numerical_head_difference_m": float(rows[-1, 3]),
        "final_analytical_head_difference_m": float(rows[-1, 4]),
        "maximum_solid_displacement_m": max_solid_displacement,
        "maximum_fluid_particles_in_closed_cavity": max_cavity_particles,
        "finite": bool(np.isfinite(rows).all()),
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and max_solid_displacement <= 1.0e-10
        and max_cavity_particles == 0
        and abs(metrics["final_numerical_head_difference_m"]) <= DX
        and relative_l2 <= 0.35
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(json.dumps(metrics, sort_keys=True))
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(rows[:, 0], rows[:, 4], "k-", label="analytical")
    ax.plot(rows[:, 0], rows[:, 3], "o", ms=3, label="MPM")
    ax.set(xlabel="time (s)", ylabel="head difference (m)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "head_decay.png", dpi=180)
    plt.close(fig)

    targets = (0.0, 10.0, 30.0, 100.0)
    snapshots = []
    for target in targets:
        file_name = min(files, key=lambda name: abs(float(np.load(name)["t_current"]) - target))
        snapshots.append(np.load(file_name))
    pressure_max = max(
        float(np.quantile(data["pressure"][(data["active"] > 0) & (data["phase"] == 2)], 0.99)) for data in snapshots
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.5), sharex=True, sharey=True)
    for ax, data in zip(axes.flat, snapshots):
        active = data["active"] > 0
        fluid = active & (data["phase"] == 2)
        solid = active & (data["phase"] == 1)
        position = data["position"]
        ax.scatter(position[solid, 0], position[solid, 1], s=1.0, color="0.65")
        points = ax.scatter(
            position[fluid, 0],
            position[fluid, 1],
            s=1.5,
            c=data["pressure"][fluid],
            cmap="viridis",
            vmin=0.0,
            vmax=pressure_max,
        )
        ax.set(
            title=f"t = {float(data['t_current']):g} s", aspect="equal", xlim=(0.0, WIDTH), ylim=(0.0, PHYSICAL_HEIGHT)
        )
    fig.supxlabel("x (m)")
    fig.supylabel("y (m)")
    fig.colorbar(points, ax=axes, label="pressure (Pa)", shrink=0.8)
    fig.savefig(output / "pressure_snapshots.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(metrics, sort_keys=True))
