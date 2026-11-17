"""Compare saved MAC velocities with Ghia et al. (1982), tables I/II, Re=100.

Reference: https://doi.org/10.1016/0021-9991(82)90058-4
This is a numerical Navier--Stokes benchmark, not a closed-form exact solution.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def centerlines(data):
    g = int(data["ghost_cell"])
    dx, dy = data["grid_size"]
    lx, ly = data["domain"]
    u, v = data["velocity_x"], data["velocity_y"]
    ux = (np.arange(u.shape[0]) - g) * dx
    uy = (np.arange(u.shape[1]) - g + 0.5) * dy
    vx = (np.arange(v.shape[0]) - g + 0.5) * dx
    vy = (np.arange(v.shape[1]) - g) * dy
    return (
        uy / ly,
        np.array([np.interp(lx / 2, ux, row) for row in u.T]),
        vx / lx,
        np.array([np.interp(ly / 2, vy, row) for row in v]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frames = sorted(args.output.glob("mac_flow_*.npz"))
    if not frames:
        parser.error("no mac_flow_*.npz frames found")
    data = np.load(frames[-1])
    reference = np.genfromtxt(Path(__file__).with_name("ghia_re100.csv"), delimiter=",", names=True)
    y, u, x, v = centerlines(data)
    speed = float(data["lid_velocity"])
    re = speed * float(data["domain"][0]) / float(data["viscosity"])
    if not np.isclose(re, 100):
        parser.error(f"reference is Re=100, but this run is Re={re}")
    ui = np.interp(reference["u_y"], y, u / speed)
    vi = np.interp(reference["v_x"], x, v / speed)
    error_u, error_v = ui - reference["u_reference"], vi - reference["v_reference"]
    metrics = {
        "reference": "Ghia et al. (1982), tables I/II; numerical reference, not exact analytic solution",
        "reference_doi": "10.1016/0021-9991(82)90058-4",
        "Re": re,
        "time": float(data["time"]),
        "grid_size": data["grid_size"].tolist(),
        "u_rmse_over_lid_speed": float(np.sqrt(np.mean(error_u**2))),
        "v_rmse_over_lid_speed": float(np.sqrt(np.mean(error_v**2))),
        "u_max_error_over_lid_speed": float(np.max(np.abs(error_u))),
        "v_max_error_over_lid_speed": float(np.max(np.abs(error_v))),
    }
    history = []
    previous = None
    max_air_cells = 0
    for frame in frames:
        with np.load(frame) as current:
            fields = np.concatenate([current[key].ravel() for key in ("velocity_x", "velocity_y")])
            change = (
                np.nan if previous is None else np.linalg.norm(fields - previous) / max(np.linalg.norm(fields), 1e-30)
            )
            history.append([float(current["time"]), change])
            previous = fields
            ghost = int(current["ghost_cell"])
            max_air_cells = max(
                max_air_cells, int(np.count_nonzero(current["cell_type"][ghost:-ghost, ghost:-ghost] == 0))
            )
    metrics["last_snapshot_relative_velocity_change"] = float(history[-1][1])
    metrics["max_interior_air_cells_over_saved_frames"] = max_air_cells
    metrics["closed_cavity_phase_check_passed"] = max_air_cells == 0
    if "max_interior_air_cells_over_steps" in data:
        metrics["max_interior_air_cells_over_steps"] = int(data["max_interior_air_cells_over_steps"])
        metrics["steps_with_interior_air"] = int(data["steps_with_interior_air"])
        metrics["closed_cavity_phase_check_passed"] &= metrics["max_interior_air_cells_over_steps"] == 0
    metrics["all_saved_mac_velocities_finite"] = bool(
        np.isfinite(previous).all() and np.isfinite(np.asarray(history)[1:, 1]).all()
    )
    g = int(data["ghost_cell"])
    nx, ny = np.rint(data["domain"] / data["grid_size"]).astype(int)
    grid_u = data["velocity_x"][g : g + nx + 1, g : g + ny]
    grid_v = data["velocity_y"][g : g + nx, g : g + ny + 1]
    divergence = np.diff(grid_u, axis=0) / data["grid_size"][0] + np.diff(grid_v, axis=1) / data["grid_size"][1]
    metrics["max_mac_divergence"] = float(np.max(np.abs(divergence)))
    types, counts = np.unique(data["cell_type"][g : g + nx, g : g + ny], return_counts=True)
    metrics["interior_cell_types"] = dict(zip(map(str, types), map(int, counts)))
    particle_files = sorted((args.output / "particles").glob("MPMParticle*.npz"))
    if particle_files:
        initial, final = np.load(particle_files[0]), np.load(particle_files[-1])
        metrics["initial_particle_count"] = len(initial["position"])
        metrics["final_particle_count"] = len(final["position"])
        metrics["relative_particle_mass_change"] = float(np.sum(final["mass"]) / np.sum(initial["mass"]) - 1)
    (args.output / "comparison.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savetxt(
        args.output / "centerline_comparison.csv",
        np.column_stack(
            [
                reference["u_y"],
                reference["u_reference"],
                ui,
                error_u,
                reference["v_x"],
                reference["v_reference"],
                vi,
                error_v,
            ]
        ),
        delimiter=",",
        header="y,u_ref,u_mpm,u_error,x,v_ref,v_mpm,v_error",
        comments="",
    )
    np.savetxt(
        args.output / "convergence.csv",
        history,
        delimiter=",",
        header="time,snapshot_relative_velocity_change",
        comments="",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(u / speed, y, label="Incompressible MPM")
    axes[0].plot(reference["u_reference"], reference["u_y"], "o", label="Ghia, Re=100")
    axes[0].set(xlabel="u / U (x/L=0.5)", ylabel="y / L", ylim=(0, 1))
    axes[1].plot(x, v / speed, label="Incompressible MPM")
    axes[1].plot(reference["v_x"], reference["v_reference"], "o", label="Ghia, Re=100")
    axes[1].set(xlabel="x / L", ylabel="v / U (y/L=0.5)", xlim=(0, 1))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    status = " — DIAGNOSTIC: internal AIR cells" if not metrics["closed_cavity_phase_check_passed"] else ""
    fig.suptitle(f"Lid-driven cavity, Re=100, t={metrics['time']:.3f}{status}")
    fig.tight_layout()
    fig.savefig(args.output / "centerlines.png", dpi=160)
    plt.close(fig)
    uc, vc = 0.5 * (grid_u[:-1] + grid_u[1:]), 0.5 * (grid_v[:, :-1] + grid_v[:, 1:])
    fig, ax = plt.subplots(figsize=(6, 5))
    centers_x = (np.arange(nx) + 0.5) * data["grid_size"][0]
    centers_y = (np.arange(ny) + 0.5) * data["grid_size"][1]
    ax.streamplot(centers_x, centers_y, uc.T, vc.T, density=1.5, color="white", linewidth=0.6)
    field = ax.pcolormesh(centers_x, centers_y, np.hypot(uc, vc).T, shading="nearest")
    air = np.argwhere(data["cell_type"][g : g + nx, g : g + ny] == 0)
    if len(air):
        ax.scatter(
            centers_x[air[:, 0]],
            centers_y[air[:, 1]],
            marker="x",
            color="red",
            s=12,
            label="AIR (invalid for filled cavity)",
        )
        ax.legend(fontsize=8)
    fig.colorbar(field, ax=ax, label="Speed")
    ax.set(xlabel="x", ylabel="y", aspect="equal", title=f"Re=100, t={metrics['time']:.3f}")
    fig.tight_layout()
    fig.savefig(args.output / "flow.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
