import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np


def list_particle_files(path):
    files = sorted(glob.glob(os.path.join(path, "particles", "MPMParticle*.npz")))
    if not files:
        raise FileNotFoundError(f"No particle files found under {path}/particles")
    return files


def frame_id(filename):
    stem = Path(filename).stem
    return int(stem.replace("MPMParticle", ""))


def read_particle_file(filename):
    data = np.load(filename, allow_pickle=True)
    position = np.asarray(data["position"], dtype=np.float64)
    active = (
        np.asarray(data["active"], dtype=np.uint8)
        if "active" in data.files
        else np.ones(position.shape[0], dtype=np.uint8)
    )
    pressure = read_pressure(data)
    time = float(data["t_current"]) if "t_current" in data.files else np.nan
    mask = active.astype(bool) & np.isfinite(pressure)
    return time, position[mask], pressure[mask]


def read_pressure(data):
    if "pressure" in data.files:
        return np.asarray(data["pressure"], dtype=np.float64)
    if "state_vars" in data.files:
        state_vars = data["state_vars"]
        if state_vars.shape == ():
            state_vars = state_vars.item()
        if isinstance(state_vars, dict) and "pressure" in state_vars:
            return np.asarray(state_vars["pressure"], dtype=np.float64)
    if "stress" in data.files:
        stress = np.asarray(data["stress"], dtype=np.float64)
        return np.mean(stress[:, :3], axis=1)
    raise KeyError("pressure is not found in pressure, state_vars['pressure'], or stress")


def infer_2d_domain(path, position):
    grid_files = sorted(glob.glob(os.path.join(path, "grids", "MPMGrid*.npz")))
    if grid_files:
        grid = np.load(grid_files[0], allow_pickle=True)
        if "coords" in grid.files:
            coords = np.asarray(grid["coords"], dtype=np.float64)
            return np.nanmax(coords[:, 0]), np.nanmax(coords[:, 1])
    return np.nanmax(position[:, 0]), np.nanmax(position[:, 1])


def binned_average(coord, value, domain, bins, min_count=1):
    coord = np.asarray(coord, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    domain = np.asarray(domain, dtype=np.float64)
    scaled = coord / domain
    valid = np.all(np.isfinite(scaled), axis=1)
    valid &= np.all(scaled >= 0.0, axis=1) & np.all(scaled <= 1.0, axis=1)
    scaled = scaled[valid]
    value = value[valid]

    ij = np.floor(scaled * np.asarray(bins)).astype(np.int64)
    ij = np.minimum(np.maximum(ij, 0), np.asarray(bins) - 1)
    linear = ij[:, 0] * bins[1] + ij[:, 1]

    count = np.bincount(linear, minlength=bins[0] * bins[1]).reshape(bins)
    total = np.bincount(linear, weights=value, minlength=bins[0] * bins[1]).reshape(bins)
    total2 = np.bincount(linear, weights=value * value, minlength=bins[0] * bins[1]).reshape(bins)

    mean = np.full(bins, np.nan, dtype=np.float64)
    std = np.full(bins, np.nan, dtype=np.float64)
    mask = count >= min_count
    mean[mask] = total[mask] / count[mask]
    variance = total2[mask] / count[mask] - mean[mask] * mean[mask]
    std[mask] = np.sqrt(np.maximum(variance, 0.0))
    return mean, std, count


def closest_time_file(files, target_time):
    best = None
    best_error = np.inf
    for filename in files:
        data = np.load(filename, allow_pickle=True)
        time = float(data["t_current"]) if "t_current" in data.files else np.nan
        error = abs(time - target_time)
        if error < best_error:
            best = filename
            best_error = error
    return best, best_error


def region_y_profile(position, pressure, region, y_bins):
    x0, x1, z0, z1 = region
    mask = (position[:, 0] >= x0) & (position[:, 0] <= x1) & (position[:, 2] >= z0) & (position[:, 2] <= z1)
    if not np.any(mask):
        return [], {}
    y = position[mask, 1]
    p = pressure[mask]
    bins = np.linspace(np.nanmin(y), np.nanmax(y), y_bins + 1)
    rows = []
    for i in range(y_bins):
        sub = (y >= bins[i]) & (y < bins[i + 1])
        if i == y_bins - 1:
            sub = (y >= bins[i]) & (y <= bins[i + 1])
        if np.any(sub):
            rows.append(
                {
                    "y_center": 0.5 * (bins[i] + bins[i + 1]),
                    "count": int(np.count_nonzero(sub)),
                    "pressure_mean": float(np.nanmean(p[sub])),
                    "pressure_std": float(np.nanstd(p[sub])),
                    "pressure_min": float(np.nanmin(p[sub])),
                    "pressure_max": float(np.nanmax(p[sub])),
                }
            )
    summary = {
        "count": int(np.count_nonzero(mask)),
        "pressure_min": float(np.nanmin(p)),
        "pressure_mean": float(np.nanmean(p)),
        "pressure_max": float(np.nanmax(p)),
        "pressure_std": float(np.nanstd(p)),
        "y_mean_span": (
            float(max(row["pressure_mean"] for row in rows) - min(row["pressure_mean"] for row in rows))
            if rows
            else np.nan
        ),
    }
    return rows, summary


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_png(path, map3d, map2d, diff):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"PNG skipped: matplotlib import failed: {exc}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, title, field in zip(axes, ["3D y-projected pressure", "2D pressure", "3D - 2D"], [map3d, map2d, diff]):
        im = ax.imshow(field.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("x / L")
        ax.set_ylabel("vertical / H")
        fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    output_root = Path(__file__).with_name("OutputData")
    parser = argparse.ArgumentParser(
        description="Project 3D large-tank pressure along y and compare it with 2D pressure."
    )
    parser.add_argument("--path3d", default=str(output_root / "large_tank_incompressible_3d"))
    parser.add_argument("--path2d", default=str(output_root / "large_tank_incompressible_2d"))
    parser.add_argument("--frame3d", type=int, default=5)
    parser.add_argument("--frame2d", type=int, default=None)
    parser.add_argument("--match-time", action="store_true", default=True)
    parser.add_argument("--bins-x", type=int, default=160)
    parser.add_argument("--bins-z", type=int, default=120)
    parser.add_argument("--min-count", type=int, default=4)
    parser.add_argument("--domain3d", type=float, nargs=3, default=[6.0, 1.42, 6.0])
    parser.add_argument("--domain2d", type=float, nargs=2, default=None)
    parser.add_argument(
        "--y-range3d",
        type=float,
        nargs=2,
        default=None,
        metavar=("Y0", "Y1"),
        help="Optional 3D thickness slab used for x-z projection. The y-profile diagnostic still uses all particles.",
    )
    parser.add_argument(
        "--region3d", type=float, nargs=4, default=[1.0, 1.5, 0.0, 0.5], metavar=("X0", "X1", "Z0", "Z1")
    )
    parser.add_argument("--y-bins", type=int, default=14)
    parser.add_argument("--output-dir", default=str(output_root / "pressure_projection_compare"))
    args = parser.parse_args()

    path3d = Path(args.path3d)
    path2d = Path(args.path2d)
    files3d = list_particle_files(path3d)
    files2d = list_particle_files(path2d)

    file3d = path3d / "particles" / f"MPMParticle{args.frame3d:06d}.npz"
    if not file3d.exists():
        raise FileNotFoundError(file3d)
    time3d, pos3d, pressure3d = read_particle_file(file3d)

    if args.frame2d is not None:
        file2d = path2d / "particles" / f"MPMParticle{args.frame2d:06d}.npz"
        if not file2d.exists():
            raise FileNotFoundError(file2d)
        match_error = np.nan
    elif args.match_time:
        file2d, match_error = closest_time_file(files2d, time3d)
    else:
        file2d = path2d / "particles" / f"MPMParticle{args.frame3d:06d}.npz"
        match_error = np.nan
    time2d, pos2d, pressure2d = read_particle_file(file2d)

    domain3d = np.asarray(args.domain3d, dtype=np.float64)
    domain2d = np.asarray(
        args.domain2d if args.domain2d is not None else infer_2d_domain(path2d, pos2d), dtype=np.float64
    )
    bins = (args.bins_x, args.bins_z)
    project_pos3d = pos3d
    project_pressure3d = pressure3d
    if args.y_range3d is not None:
        y0, y1 = args.y_range3d
        y_mask = (pos3d[:, 1] >= y0) & (pos3d[:, 1] <= y1)
        project_pos3d = pos3d[y_mask]
        project_pressure3d = pressure3d[y_mask]
        if project_pos3d.shape[0] == 0:
            raise RuntimeError(f"No 3D particles found in y-range [{y0}, {y1}]")

    map3d, std3d, count3d = binned_average(
        project_pos3d[:, [0, 2]], project_pressure3d, domain3d[[0, 2]], bins, args.min_count
    )
    map2d, std2d, count2d = binned_average(pos2d[:, [0, 1]], pressure2d, domain2d, bins, args.min_count)

    both = np.isfinite(map3d) & np.isfinite(map2d)
    diff = map3d - map2d
    comparison = {
        "frame3d": frame_id(file3d),
        "time3d": time3d,
        "frame2d": frame_id(file2d),
        "time2d": time2d,
        "time_error": float(abs(time3d - time2d)) if np.isfinite(time3d) and np.isfinite(time2d) else match_error,
        "projected_3d_particles": int(project_pos3d.shape[0]),
        "overlap_bins": int(np.count_nonzero(both)),
        "mae": float(np.nanmean(np.abs(diff[both]))) if np.any(both) else np.nan,
        "rmse": float(np.sqrt(np.nanmean(diff[both] * diff[both]))) if np.any(both) else np.nan,
        "max_abs": float(np.nanmax(np.abs(diff[both]))) if np.any(both) else np.nan,
        "corr": float(np.corrcoef(map3d[both], map2d[both])[0, 1]) if np.count_nonzero(both) > 2 else np.nan,
    }

    y_rows, region_summary = region_y_profile(pos3d, pressure3d, args.region3d, args.y_bins)
    top = np.argsort(pressure3d)[-20:][::-1]
    top_rows = [
        {
            "rank": i + 1,
            "x": float(pos3d[idx, 0]),
            "y": float(pos3d[idx, 1]),
            "z": float(pos3d[idx, 2]),
            "pressure": float(pressure3d[idx]),
        }
        for i, idx in enumerate(top)
    ]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        outdir / f"pressure_projection_3d{frame_id(file3d):06d}_2d{frame_id(file2d):06d}.npz",
        map3d=map3d,
        std3d=std3d,
        count3d=count3d,
        map2d=map2d,
        std2d=std2d,
        count2d=count2d,
        diff=diff,
        domain3d=domain3d,
        domain2d=domain2d,
        comparison=comparison,
        region_summary=region_summary,
    )
    write_csv(outdir / "comparison_summary.csv", [comparison], list(comparison.keys()))
    if y_rows:
        write_csv(outdir / "region_y_profile.csv", y_rows, list(y_rows[0].keys()))
    write_csv(outdir / "top_3d_pressure_particles.csv", top_rows, list(top_rows[0].keys()))
    save_png(outdir / f"pressure_projection_3d{frame_id(file3d):06d}_2d{frame_id(file2d):06d}.png", map3d, map2d, diff)

    print("3D file:", file3d, "time=", time3d, "particles=", pos3d.shape[0])
    print("2D file:", file2d, "time=", time2d, "particles=", pos2d.shape[0])
    print("comparison:", comparison)
    print("region3d x/z summary:", region_summary)
    if y_rows:
        print("region y-profile first/last:", y_rows[0], y_rows[-1])
    print("top 3D pressure particle:", top_rows[0])
    print("output:", outdir)


if __name__ == "__main__":
    main()
