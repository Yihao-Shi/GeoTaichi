import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import detrend, periodogram


def dominant_frequency(time, signal):
    sample_dt = float(np.median(np.diff(time)))
    frequency, power = periodogram(detrend(signal), fs=1.0 / sample_dt, window="hann", nfft=max(4096, 8 * len(time)))
    usable = (frequency >= 0.3) & (frequency <= 0.4 / sample_dt)
    peak = np.flatnonzero(usable)[np.argmax(power[usable])]
    return float(frequency[peak]), frequency, power


def analyze(model_path, output_path, diameter, reference_velocity, viscosity, dx):
    files = sorted((model_path / "particles").glob("MPMParticle*.npz"))
    if len(files) < 10:
        raise ValueError(f"need at least 10 particle frames, found {len(files)}")

    rows = []
    probe = np.array([0.35 + 3.0 * diameter, 0.20])
    for path in files:
        with np.load(path) as data:
            position, velocity = data["position"], data["velocity"]
            near = np.all(np.abs(position - probe) <= 1.5 * dx, axis=1)
            if not np.any(near):
                raise ValueError(f"empty wake probe in {path.name}")
            rows.append(
                (
                    float(data["t_current"]),
                    int(np.sum(near)),
                    float(np.mean(velocity[near, 0])),
                    float(np.mean(velocity[near, 1])),
                    float(np.mean(velocity[:, 0])),
                )
            )

    history = np.asarray(rows)
    transient_end = history[0, 0] + 0.3 * (history[-1, 0] - history[0, 0])
    sample = history[:, 0] >= transient_end
    frequency, frequencies, power = dominant_frequency(history[sample, 0], history[sample, 3])
    mean_velocity = float(np.mean(history[sample, 4]))
    strouhal = frequency * diameter / mean_velocity
    rms_crossflow = float(np.sqrt(np.mean(detrend(history[sample, 3]) ** 2)))
    metrics = {
        "frames": len(files),
        "analysis_start_time": float(history[sample, 0][0]),
        "final_time": float(history[-1, 0]),
        "probe_x": float(probe[0]),
        "probe_y": float(probe[1]),
        "mean_bulk_velocity": mean_velocity,
        "nominal_reynolds_number": reference_velocity * diameter / viscosity,
        "dominant_frequency": frequency,
        "strouhal_number": strouhal,
        "crossflow_velocity_rms": rms_crossflow,
        "shedding_detected": bool(rms_crossflow > 1.0e-3 * abs(mean_velocity) and 0.08 < strouhal < 0.30),
    }

    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "wake_probe.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "sample_count", "probe_u", "probe_v", "mean_u"))
        writer.writerows(history)
    (output_path / "vortex_shedding_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    axes[0].plot(history[:, 0], history[:, 3], lw=1.2)
    axes[0].axvline(transient_end, color="0.5", ls="--", lw=0.8)
    axes[0].set(xlabel="time (s)", ylabel=r"wake $v_y$", title="Cylinder-wake cross-flow probe")
    axes[0].grid(alpha=0.25)
    positive = frequencies > 0.0
    axes[1].plot(frequencies[positive], power[positive], lw=1.2)
    axes[1].axvline(frequency, color="tab:red", ls="--", lw=0.9, label=f"f={frequency:.3f} Hz, St={strouhal:.3f}")
    axes[1].set(xlim=(0.0, 5.0), xlabel="frequency (Hz)", ylabel="PSD")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.savefig(output_path / "wake_probe_and_spectrum.png", dpi=180)
    plt.close(fig)

    with np.load(files[-1]) as data:
        position, velocity = data["position"], data["velocity"]
    x = np.arange(dx / 2.0, 1.0, dx)
    y = np.arange(dx / 2.0, 0.4, dx)
    xx, yy = np.meshgrid(x, y)
    ux = griddata(position, velocity[:, 0], (xx, yy), method="linear")
    uy = griddata(position, velocity[:, 1], (xx, yy), method="linear")
    dvdx = np.gradient(uy, dx, axis=1)
    dudy = np.gradient(ux, dx, axis=0)
    vorticity = dvdx - dudy
    solid = (xx - 0.35) ** 2 + (yy - 0.20) ** 2 <= (0.5 * diameter) ** 2
    vorticity[solid] = np.nan
    limit = max(1.0, float(np.nanpercentile(np.abs(vorticity), 98)))
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    image = ax.contourf(xx, yy, vorticity, levels=np.linspace(-limit, limit, 81), cmap="RdBu_r", extend="both")
    ax.add_patch(plt.Circle((0.35, 0.20), 0.5 * diameter, color="0.2"))
    ax.set(xlabel="x", ylabel="y", title=f"Vorticity at t={history[-1, 0]:.3f} s", aspect="equal")
    fig.colorbar(image, ax=ax, label=r"$\omega_z$")
    fig.savefig(output_path / "vorticity_final.png", dpi=180)
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--diameter", type=float, default=0.1)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--viscosity", type=float, default=1.0e-3)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        t = np.linspace(0.0, 5.0, 251)
        assert abs(dominant_frequency(t, np.sin(2.0 * np.pi * 1.6 * t))[0] - 1.6) < 0.02
    else:
        if args.model is None:
            parser.error("model path is required")
        print(
            json.dumps(
                analyze(args.model, args.output or args.model, args.diameter, args.velocity, args.viscosity, args.dx),
                indent=2,
                sort_keys=True,
            )
        )
