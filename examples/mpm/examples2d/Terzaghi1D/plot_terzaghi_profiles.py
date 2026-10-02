import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def load_profiles(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    profiles = [np.asarray(arr, dtype=np.float64) for arr in data["profiles"]]
    dt = float(data["dt"])
    height = float(data["height"]) if "height" in data else None
    surcharge = float(data["surcharge"]) if "surcharge" in data else None
    steps = data["steps"].tolist() if "steps" in data else None
    return profiles, dt, height, surcharge, steps


def infer_times(n_profiles, dt):
    default_steps = {
        6: [55, 111, 222, 333, 555, 600],
        7: [1, 5, 10, 15, 20, 25, 40],
        4: [1, 10, 20, 40],
    }
    steps = default_steps.get(n_profiles, list(range(1, n_profiles + 1)))
    if len(steps) != n_profiles:
        steps = list(range(1, n_profiles + 1))
    return [dt * step for step in steps]


def main():
    default_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Plot Terzaghi 1D consolidation profiles.")
    parser.add_argument(
        "--input",
        default=os.path.join(default_dir, "terzaghi_profiles.npz"),
        help="Path to terzaghi_profiles.npz",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(default_dir, "terzaghi_profiles_plot.png"),
        help="Output figure path",
    )
    parser.add_argument(
        "--times",
        nargs="*",
        type=float,
        default=None,
        help="Optional time labels matching the stored profiles",
    )
    parser.add_argument(
        "--xlabel",
        default=r"$p/p_0$",
        help="X-axis label",
    )
    parser.add_argument(
        "--ylabel",
        default=r"$z/H$",
        help="Y-axis label",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor applied to pore pressure before plotting",
    )
    parser.add_argument(
        "--normalize-pressure",
        action="store_true",
        help="Normalize pore pressure by surcharge p0 stored in the npz file",
    )
    parser.add_argument(
        "--normalize-depth",
        action="store_true",
        help="Normalize depth by specimen height H",
    )
    parser.set_defaults(normalize_depth=True)
    args = parser.parse_args()

    profiles, dt, height_ref, surcharge, steps = load_profiles(args.input)
    if args.times is not None and len(args.times) == len(profiles):
        times = args.times
    elif steps is not None:
        times = [dt * step for step in steps]
    else:
        times = infer_times(len(profiles), dt)

    colors = [
        "#202020",
        "#ff2d2d",
        "#4a69bd",
        "#d633b6",
        "#43c6f5",
        "#ff9f1a",
        "#6f3f1a",
        "#9e9e9e",
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 18,
            "axes.linewidth": 1.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.minor.width": 1.2,
            "ytick.minor.width": 1.2,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=200)

    sim_handle = None
    ana_handle = None

    for idx, (profile, time_value) in enumerate(zip(profiles, times)):
        depth = profile[:, 0]
        numerical = profile[:, 1].copy()
        analytical = profile[:, 2].copy()
        depth_axis = depth.copy()
        color = colors[idx % len(colors)]

        if args.normalize_pressure:
            if surcharge is None or abs(surcharge) < 1.0e-14:
                raise ValueError("Cannot normalize pressure because surcharge is missing or zero in the npz file.")
            numerical /= surcharge
            analytical /= surcharge
        numerical *= args.scale
        analytical *= args.scale

        if args.normalize_depth:
            current_height = height_ref if height_ref is not None else depth.max()
            if current_height <= 0.0:
                raise ValueError("Invalid specimen height for depth normalization.")
            depth_axis /= current_height

        line, = ax.plot(
            analytical,
            depth_axis,
            color=color,
            lw=1.6,
            zorder=2,
        )
        pts = ax.plot(
            numerical,
            depth_axis,
            linestyle="none",
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            zorder=3,
        )[0]

        label_x = analytical[min(len(analytical) - 1, max(2, len(analytical) // 3))]
        label_y = depth_axis[min(len(depth_axis) - 1, max(2, len(depth_axis) // 3))]
        ax.text(label_x + 0.03 * max(1.0, analytical.max()), label_y, rf"$T={time_value:.2f}$", color=color, fontsize=15)

        if idx == 0:
            sim_handle = pts
            ana_handle = line

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.minorticks_on()
    ax.invert_yaxis()
    ax.legend(
        [sim_handle, ana_handle],
        ["MPM simulation", "Analytical solution"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    max_depth = max(profile[:, 0].max() for profile in profiles)
    if args.normalize_depth:
        ref_height = height_ref if height_ref is not None else max_depth
        ax.set_ylim(0.0, max_depth / ref_height)
    else:
        ax.set_ylim(0.0, 1.05 * max_depth)
    xmax = max(np.max(profile[:, 1]) for profile in profiles) * args.scale
    xmax = max(xmax, max(np.max(profile[:, 2]) for profile in profiles) * args.scale)
    if args.normalize_pressure:
        xmax /= surcharge
    ax.set_xlim(0.0, 1.15 * xmax)
    fig.tight_layout()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
