"""Plot the saved 2D states at the paper's times and report distance differences."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

from evaluate_sensitive_clay import PAPER_TIMES, equivalent_plastic_strain


def draw_state(ax, output, row, initial=False):
    with np.load(output / "particles" / row["file"], allow_pickle=True) as data:
        pos = data["position"]
        half_width = 0.5 * np.sqrt(data["volume"])
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        vertices = pos[:, None, :] + half_width[:, None, None] * corners
        field = data["stress"][:, 1] / 1000 if initial else equivalent_plastic_strain(data["state_vars"].item())
        collection = PolyCollection(
            vertices,
            array=field,
            cmap="coolwarm",
            linewidth=0,
            norm=Normalize(-100, 0) if initial else Normalize(0, 2),
            rasterized=True,
        )
        ax.add_collection(collection)
    ax.plot([0, 20, 25], [5, 5, 0], "k--", linewidth=0.7, alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set(xlim=(0, 50), ylim=(-0.15, 6), aspect="equal", ylabel="y (m)")
    ax.text(
        0.01,
        0.86,
        "Elastic initial stress" if initial else f"t = {row['time']:.2f} s",
        transform=ax.transAxes,
        fontsize=10,
    )
    return collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--paper-figure", type=Path, help="Optional local image of the supplied PDF's Fig.17")
    args = parser.parse_args()
    output = args.output.resolve()
    report = json.loads((output / "evaluation.json").read_text())
    if not report["passed"] or report["paper_comparison"] is None:
        raise ValueError("Evaluate the completed 10 s simulation before paper comparison")
    render = output / "render"
    render.mkdir(exist_ok=True)
    rows = [report["initial"]] + [min(report["history"], key=lambda r: abs(r["time"] - t)) for t in PAPER_TIMES]
    fig, axes = plt.subplots(7, 1, figsize=(13, 12), sharex=True, layout="constrained")
    for i, (ax, row) in enumerate(zip(axes, rows)):
        colors = draw_state(ax, output, row, initial=i == 0)
        ax.set_xlim(0, max(50.0, 5.0 * np.ceil((report["bounds"][1] + 1.0) / 5.0)))
        if i in (0, 6):
            fig.colorbar(
                colors,
                ax=ax,
                shrink=0.55,
                pad=0.01,
                label="Vertical stress (kPa)" if i == 0 else "Equivalent plastic strain",
            )
    axes[-1].set_xlabel("x (m)")
    comparison = report["paper_comparison"]
    model = report["run"].get("material_model", "MohrCoulomb")
    grid = "adaptive" if report["run"]["adaptive"] else "uniform"
    fig.suptitle(
        f"2D {grid} MPM, "
        f"{model}, section 4.3, ground friction = {comparison['friction']:g}\n"
        "Dashed: undeformed section; strain range 0–2 matches Fig.17",
        fontsize=13,
    )
    fig.savefig(render / "paper_times_2d.png", dpi=180)
    plt.close(fig)
    if args.paper_figure:
        reference = plt.imread(args.paper_figure)
        # Panel boundaries in the supplied Fig.17 crop, including its distance arrows.
        cuts = np.array([0, 140, 249, 369, 492, 618, 734, 922]) / 950 * len(reference)
        fig, axes = plt.subplots(7, 2, figsize=(18, 13), layout="constrained", gridspec_kw={"width_ratios": [1, 1.4]})
        for i, row in enumerate(rows):
            axes[i, 0].imshow(reference[int(cuts[i]) : int(cuts[i + 1])])
            axes[i, 0].axis("off")
            colors = draw_state(axes[i, 1], output, row, initial=i == 0)
            axes[i, 1].set_xlim(0, max(50.0, 5.0 * np.ceil((report["bounds"][1] + 1.0) / 5.0)))
            if i < 6:
                axes[i, 1].tick_params(labelbottom=False)
            if i in (0, 6):
                fig.colorbar(
                    colors,
                    ax=axes[i, 1],
                    shrink=0.75,
                    pad=0.01,
                    label="Vertical stress (kPa)" if i == 0 else "Equivalent plastic strain",
                )
        axes[0, 0].set_title("Paper Fig.17: constrained 3D reference")
        axes[0, 1].set_title(f"Computed 2D {model}, {grid} grid (physical x:y aspect)")
        axes[-1, 1].set_xlabel("x (m)")
        fig.suptitle("Zhang et al. (2021), section 4.3; strain range 0–2; dashed line = initial section", fontsize=13)
        fig.savefig(render / "paper_comparison.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, key, title in zip(axes, ["runout_m", "retrogression_m"], ["Run-out distance", "Retrogression distance"]):
        bars = ax.bar(
            ["Paper", "2D MPM"], [comparison["reference"][key], comparison["mpm"][key]], color=["#777777", "#3182bd"]
        )
        ax.bar_label(bars, fmt="%.2f m")
        ax.set(title=title, ylabel="Distance (m)")
        ax.set_ylim(0, max(comparison["reference"][key], comparison["mpm"][key]) * 1.2)
    fig.savefig(render / "distance_comparison.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Section 4.3 comparison",
        "",
        "Source: Zhang et al., Computers and Geotechnics 129 (2021), 103856, section 4.3, pp.10–12, Figs.15–18.",
        "",
        f"Computed: 2D {model}, adaptive={report['run']['adaptive']}, cell size={report['run']['cell_size']:g} m.",
        "",
        "| Observable | Paper (m) | 2D MPM (m) | Difference |",
        "|---|---:|---:|---:|",
    ]
    for key in comparison["reference"]:
        lines.append(
            f"| {key} | {comparison['reference'][key]:.2f} | {comparison['mpm'][key]:.2f} | "
            f"{comparison['relative_error_percent'][key]:+.1f}% |"
        )
    lines += [
        "",
        "The distances are measured at 10 s, not extrapolated to a resting deposit.",
        "",
        "Retrogression is an envelope-based estimate; the threshold sensitivity is recorded in paper_comparison.json.",
        "",
        "## Numerical checks",
        "",
        f"Checks passed: {report['passed']}; relative mass drift: {report['max_relative_mass_error']:.3g}.",
        f"Final maximum speed: {report['final']['max_speed']:.3g} m/s.",
        "",
        "## Limits",
        "",
    ]
    lines += ["- " + item for item in comparison["limitations"]]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {"comparison": comparison["mpm"], "relative_error_percent": comparison["relative_error_percent"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
