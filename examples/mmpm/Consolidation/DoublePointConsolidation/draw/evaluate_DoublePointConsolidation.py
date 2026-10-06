"""Terzaghi profiles and plots from saved double-point MPM frames."""

import os

import numpy as np


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


def average_pressure_profile(
    position, pressure, phase, height, time_value=None, cv=None, load=None, n_bins=None, *, column_bottom
):
    fluid_mask = phase == 2
    position = position[fluid_mask]
    pressure = pressure[fluid_mask]
    column_top = column_bottom + height
    depth = column_top - position[:, 1]
    valid = (depth >= 0.0) & (depth <= height) & np.isfinite(pressure)
    depth = depth[valid]
    pressure = pressure[valid]

    bins = np.linspace(0.0, height, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    values = np.zeros(n_bins, dtype=np.float64)
    analytical = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int32)
    ids = np.clip(np.digitize(depth, bins) - 1, 0, n_bins - 1)
    particle_theory = None
    if time_value is not None and cv is not None and load is not None:
        particle_theory = terzaghi_series(depth, time_value, height, load, cv)
    for pid, bid in enumerate(ids):
        values[bid] += pressure[pid]
        if particle_theory is not None:
            analytical[bid] += particle_theory[pid]
        counts[bid] += 1
    mask = counts > 0
    values[mask] /= counts[mask]
    if particle_theory is not None:
        analytical[mask] /= counts[mask]
    else:
        analytical[:] = np.nan
    return centers, values, analytical, mask


def postprocess_consolidation(
    save_path,
    *,
    simulation_time,
    save_interval,
    cv,
    column_height,
    surcharge,
    dt,
    column_bottom,
    profile_bins,
):
    particle_dir = os.path.join(save_path, "particles")
    file_names = sorted(
        name for name in os.listdir(particle_dir) if name.startswith("MPMParticle") and name.endswith(".npz")
    )
    expected_save_count = int(round(simulation_time / save_interval))
    file_names = file_names[: expected_save_count + 1]
    if len(file_names) < 2:
        raise RuntimeError(f"Not enough particle files in {particle_dir}")

    profiles = []
    times = []
    errors = []
    for save_id, file_name in enumerate(file_names[1:], start=1):
        data = np.load(os.path.join(particle_dir, file_name))
        time_value = float(data["t_current"]) if "t_current" in data.files else save_id * save_interval
        position = data["position"]
        pressure = data["pressure"]
        phase = data["phase"]
        fluid_mask = phase == 2
        if not np.any(fluid_mask):
            raise RuntimeError(f"No fluid particles found in {file_name}")
        finite_fluid_pressure = np.isfinite(pressure[fluid_mask])
        if not np.all(finite_fluid_pressure):
            bad = int(np.count_nonzero(~finite_fluid_pressure))
            raise RuntimeError(f"{bad} non-finite fluid particle pressures in {file_name}")
        depth, numerical, analytical, mask = average_pressure_profile(
            position,
            pressure,
            phase,
            column_height,
            time_value,
            cv,
            surcharge,
            n_bins=profile_bins,
            column_bottom=column_bottom,
        )
        if not np.any(mask):
            raise RuntimeError(f"No valid fluid pressure bins found in {file_name}")
        abs_err = np.linalg.norm(numerical[mask] - analytical[mask])
        rel_err = abs_err / max(np.linalg.norm(analytical[mask]), 1.0e-12)
        profiles.append(np.column_stack([depth, numerical, analytical]))
        times.append(time_value)
        errors.append([save_id, time_value, cv * time_value / (column_height * column_height), abs_err, rel_err])
        print(
            f"save={save_id:03d}, time={time_value:10.4e}, "
            f"Tv={cv * time_value / (column_height * column_height):8.4f}, "
            f"relative_profile_error={rel_err:10.4e}"
        )

    profile_path = os.path.join(save_path, "terzaghi_profiles.npz")
    np.savez(
        profile_path,
        profiles=np.array(profiles, dtype=object),
        times=np.array(times, dtype=np.float64),
        dt=dt,
        cv=cv,
        height=column_height,
        surcharge=surcharge,
    )
    np.savetxt(
        os.path.join(save_path, "terzaghi_profile_errors.csv"),
        np.asarray(errors, dtype=np.float64),
        delimiter=",",
        header="save_id,time_s,Tv,abs_profile_error,relative_profile_error",
        comments="",
    )
    plot_terzaghi_profiles(profile_path, os.path.join(save_path, "terzaghi_profiles_plot.png"))


def plot_terzaghi_profiles(npz_path, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(npz_path, allow_pickle=True)
    profiles = [np.asarray(profile, dtype=np.float64) for profile in data["profiles"]]
    times = data["times"]
    cv = float(data["cv"])
    height = float(data["height"])
    surcharge = float(data["surcharge"])

    colors = ["#202020", "#e74c3c", "#2e6bd1", "#8e44ad", "#16a085", "#f39c12", "#7f8c8d"]
    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=200)
    sim_handle = None
    ana_handle = None
    for idx, (profile, time_value) in enumerate(zip(profiles, times)):
        color = colors[idx % len(colors)]
        depth = profile[:, 0] / height
        numerical = profile[:, 1] / surcharge
        analytical = profile[:, 2] / surcharge
        ana_handle = ax.plot(analytical, depth, color=color, lw=1.6, zorder=2)[0]
        sim_handle = ax.plot(
            numerical,
            depth,
            linestyle="none",
            marker="o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            zorder=3,
        )[0]
        label_id = min(len(depth) - 1, max(2, len(depth) // 3))
        ax.text(
            analytical[label_id] + 0.025,
            depth[label_id],
            rf"$T_v={cv * time_value / (height * height):.2f}$",
            color=color,
            fontsize=11,
        )

    ax.set_xlabel(r"$u/u_0$")
    ax.set_ylabel(r"$z/H$")
    ax.set_xlim(0.0, 1.15)
    ax.set_ylim(0.0, 1.0)
    ax.invert_yaxis()
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(
        [sim_handle, ana_handle],
        ["Double-point MPM", "Terzaghi theory"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
