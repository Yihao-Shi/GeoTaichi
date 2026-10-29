"""Add the current 2D uniform and adaptive L2 results to the large-strain benchmark."""

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xlrd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE.parent / "strip_footing_adaptive/OutputData/verification_20261022"


def main():
    original = HERE / "large_strain.png"
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    output = HERE / "OutputData/large_strain_uniform_adaptive_l2.png"
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    plt.rcParams.update({"font.family": "serif", "font.size": 17, "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.27)
    ax.axhline(5.14, color="black", linestyle="--", linewidth=2, label="Lower bound (Prandtl, 1921)")
    ax.axhline(8.28, color="black", linestyle="-.", linewidth=2, label="Upper bound (Meyerhof, 1951)")
    sheet = xlrd.open_workbook(str(HERE / "comparison.xls")).sheet_by_index(0)
    headers = sheet.row_values(0)
    for index, label, color, style in [
        (1, r"RITSS (Wang $et\ al.$, 2013)", "green", "-."),
        (2, r"PFEM (Monforte $et\ al.$, 2017)", "#5e72ff", "--"),
        (3, r"Moving mesh MPM (Bisht $et\ al.$, 2021)", "red", ":"),
    ]:
        xcol, ycol = headers.index(f"Line #{index}"), headers.index(f"data{index}")
        pairs = np.array(
            [
                [sheet.cell_value(row, xcol), sheet.cell_value(row, ycol)]
                for row in range(2, sheet.nrows)
                if all(isinstance(sheet.cell_value(row, col), (int, float)) for col in (xcol, ycol))
            ]
        )
        assert pairs.ndim == 2 and pairs.shape[1] == 2 and np.isfinite(pairs).all()
        ax.plot(pairs[:, 0], pairs[:, 1], color=color, linestyle=style, linewidth=2.4, label=label)
    reference = np.genfromtxt(RESULTS.parent / "historical_normal.csv", delimiter=",", names=True)
    ax.plot(
        reference["settlement_m"] / 2,
        reference["footing_fy_N_per_m"] / 1000,
        color="grey",
        linewidth=2.2,
        label="MPM (GeoTaichi, original)",
    )
    curves = {}
    for name, color, style, label in [
        ("uniform", "#084594", "-", "2D uniform grid (h = 0.1 m)"),
        ("adaptive_l2", "#e67700", "--", "2D adaptive L2 (h = 0.4 to 0.1 m)"),
    ]:
        data = np.genfromtxt(RESULTS / name / "contact_history.csv", delimiter=",", names=True)
        x, y = data["settlement_m"] / 2, data["footing_fy_N_per_m"] / 1000
        assert np.isfinite(x).all() and np.isfinite(y).all() and np.all(np.diff(x) >= -1e-8)
        curves[name] = (x, y, color, style)
        ax.plot(x, y, color=color, linestyle=style, linewidth=1.5, label=label)
    ax.set(xlabel=r"Normalized displacement, $d/B$", ylabel=r"Normalized load, $q/s_u$", xlim=(0, 1), ylim=(0, 9))
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=12.5)
    # Retain the original axes while showing the isolated L2 peak beyond their range.
    inset = ax.inset_axes([0.65, 0.08, 0.31, 0.30])
    for x, y, color, style in curves.values():
        mask = (x >= 0.095) & (x <= 0.105)
        inset.plot(x[mask], y[mask], color=color, linestyle=style, linewidth=1.2)
    peak = float(curves["adaptive_l2"][1].max())
    inset.set(xlim=(0.095, 0.105), ylim=(0, max(9, peak * 1.08)), title="L2 transient (raw data)")
    inset.set_xticks([0.095, 0.100, 0.105])
    inset.tick_params(labelsize=10)
    inset.title.set_fontsize(11)
    inset.text(0.97, 0.95, f"Peak = {peak:.2f}", ha="right", va="top", transform=inset.transAxes, fontsize=10)
    fig.savefig(output, dpi=250)
    plt.close(fig)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == original_hash
    print(output)


if __name__ == "__main__":
    main()
