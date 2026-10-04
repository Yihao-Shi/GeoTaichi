"""Compare the complete E4 settling trajectory with the repository CSV data."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def load_velocity_experiment(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all():
        raise ValueError(f"invalid Ten Cate velocity data in {path}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path(__file__).parent / "OutputData/semi_resolved_E4")
    parser.add_argument("--experiment", type=Path, default=Path(__file__).with_name("experiment.csv"))
    args = parser.parse_args()
    experiment = load_velocity_experiment(args.experiment)
    trajectory = args.output / "trajectory.npz"
    if trajectory.exists():
        data = np.load(trajectory)
    else:
        # Failed runs still have their on-disk frames; do not invent an endpoint.
        frames = sorted(args.output.rglob("DEMParticle*.npz"))
        if not frames:
            parser.error("neither trajectory.npz nor DEMParticle frames exist")
        times, positions, velocities = [], [], []
        for frame in frames:
            with np.load(frame) as saved:
                if not len(saved["position"]):
                    break
                times.append(float(saved["t_current"]))
                positions.append(saved["position"][:1])
                velocities.append(saved["velocity"][:1])
        data = {"time": np.array(times), "center": np.array(positions), "velocity": np.array(velocities)}
    time, velocity = data["time"], data["velocity"][:, 0, 2]
    covered = (experiment[:, 0] >= time[0]) & (experiment[:, 0] <= time[-1])
    samples = experiment[covered]
    if not len(samples):
        parser.error("simulation does not overlap the experimental time window")
    predicted = np.interp(samples[:, 0], time, velocity)
    error = predicted - samples[:, 1]
    peak = float(np.max(np.abs(experiment[:, 1])))
    pre_near_wall = samples[:, 0] <= 1.0
    configuration = json.loads((args.output / "configuration.json").read_text())
    particle_density = float(configuration["particle_density"])
    fluid_density = float(configuration["fluid_density"])
    viscosity = float(configuration["viscosity"])
    diameter = float(configuration["diameter"])
    added_mass_coefficient = float(configuration.get("added_mass_coefficient", 0.0))
    effective_density = particle_density + added_mass_coefficient * fluid_density

    # Diagnostic limit of the implemented quasi-steady drag closure; this is
    # neither the confined experiment nor a substitute for the coupled MPM run.
    def quasi_steady_drag(_, state):
        speed = abs(state[1])
        reynolds = fluid_density * diameter * speed / viscosity
        acceleration = -9.81 * (particle_density - fluid_density) / effective_density
        acceleration -= 18.0 * viscosity / (effective_density * diameter**2) * (1.0 + 0.15 * reynolds**0.687) * state[1]
        return [state[1], acceleration]

    release_height = float(configuration["centers"][0][2])
    published_bottom_clearance = release_height - 0.5 * diameter
    experimental_displacement = -float(np.trapz(np.r_[0.0, experiment[:, 1]], np.r_[0.0, experiment[:, 0]]))
    ode = solve_ivp(
        quasi_steady_drag, (0.0, float(time[-1])), [release_height, 0.0], t_eval=time, rtol=1e-9, atol=1e-12
    )
    prewall = ode.y[0] > 0.03
    # Keep every workbook row, including the two different values at its final time.
    metrics = {
        "workbook": args.experiment.name,
        "experimental_samples": len(experiment),
        "trajectory_source": "callback" if trajectory.exists() else "saved DEM frames (run incomplete)",
        "covered_samples": int(covered.sum()),
        "complete_time_window": bool(covered.all()),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(error**2))),
        "velocity_rmse_over_experimental_peak": float(np.sqrt(np.mean(error**2)) / peak),
        "pre_near_wall_cutoff_s": 1.0,
        "pre_near_wall_samples": int(pre_near_wall.sum()),
        "pre_near_wall_velocity_rmse_over_experimental_peak": float(np.sqrt(np.mean(error[pre_near_wall] ** 2)) / peak),
        "velocity_max_error_m_s": float(np.max(np.abs(error))),
        "experimental_peak_speed_m_s": peak,
        "simulation_peak_speed_m_s": float(np.max(-velocity)),
        "final_time": float(time[-1]),
        "final_center_z": float(data["center"][-1, 0, 2]),
        "minimum_bottom_clearance_m": float(np.min(data["center"][:, 0, 2]) - 0.0075),
        "time_or_velocity_rescaled": False,
        "configured_release_center_z_m": release_height,
        "published_initial_bottom_clearance_m": published_bottom_clearance,
        "experimental_integrated_downward_displacement_m": experimental_displacement,
        "late_workbook_geometry_consistent": bool(experimental_displacement <= published_bottom_clearance),
        "prewall_rmse_to_quasi_steady_drag_ode_m_s": float(
            np.sqrt(np.mean((velocity[prewall] - ode.y[1, prewall]) ** 2))
        ),
    }
    (args.output / "experiment_comparison.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savetxt(
        args.output / "experiment_comparison.csv",
        np.column_stack([samples[:, 0], samples[:, 1], predicted, error]),
        delimiter=",",
        header="time_s,experiment_vz_m_s,simulation_vz_m_s,error_m_s",
        comments="",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(time, velocity, label="Semi-resolved incompressible MPM")
    axes[0].plot(time[prewall], ode.y[1, prewall], "--", label="Quasi-steady drag ODE (diagnostic)")
    axes[0].scatter(*experiment.T, label="experiment.xls (ten Cate E4)", s=18, color="black")
    axes[0].set(xlabel="Time (s)", ylabel="Vertical velocity (m/s)")
    axes[0].legend(fontsize=8)
    axes[1].plot(time, data["center"][:, 0, 2], label="Sphere center")
    axes[1].axhline(0.0075, color="black", linestyle="--", label="Bottom contact height")
    axes[1].set(xlabel="Time (s)", ylabel="Height above tank bottom (m)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output / "experiment_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
