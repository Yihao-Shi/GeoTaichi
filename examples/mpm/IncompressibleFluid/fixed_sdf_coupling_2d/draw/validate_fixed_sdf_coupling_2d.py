import argparse
import glob
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the 2D incompressible SDF coupling NPZ output.")
    parser.add_argument("--path", default="fixed_sdf_coupling_2d")
    parser.add_argument("--domain", nargs=2, type=float, default=(0.50, 0.30))
    parser.add_argument("--center", nargs=2, type=float, default=(0.19, 0.12))
    parser.add_argument("--radius", type=float, default=0.035)
    parser.add_argument("--ghost-cell", type=int, default=1)
    parser.add_argument("--sdf-tol", type=float, default=1.0e-6)
    parser.add_argument("--penetration-tol", type=float, default=1.0e-9)
    parser.add_argument("--near-band-cells", type=float, default=2.0)
    parser.add_argument("--allow-no-near-particles", action="store_true")
    return parser.parse_args()


def latest_npz(path, subdir, prefix):
    files = sorted(glob.glob(os.path.join(path, subdir, f"{prefix}*.npz")))
    if not files:
        raise FileNotFoundError(f"No {prefix}*.npz files found under {os.path.join(path, subdir)}")
    return files[-1]


def cell_array(grid_output, key, dtype=None):
    array = np.asarray(grid_output[key], dtype=dtype)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def interpolate_cell_sdf(position, solid_sdf, grid_size, ghost_cell):
    cnum = np.asarray(solid_sdf.shape[:2], dtype=np.int64)
    coord = position[:, :2] / grid_size - 0.5
    base = np.floor(coord).astype(np.int64)
    frac = coord - base
    phi = np.zeros(position.shape[0], dtype=np.float64)
    for i in (0, 1):
        for j in (0, 1):
            logical = base + np.asarray([i, j], dtype=np.int64)
            logical = np.minimum(np.maximum(logical, -ghost_cell), cnum - ghost_cell - 1)
            storage = logical + ghost_cell
            weight = (frac[:, 0] if i else 1.0 - frac[:, 0]) * (frac[:, 1] if j else 1.0 - frac[:, 1])
            phi += weight * solid_sdf[storage[:, 0], storage[:, 1]]
    return phi


def validate(args):
    particle_file = latest_npz(args.path, "particles", "MPMParticle")
    grid_file = latest_npz(args.path, "grids", "MPMGrid")
    particle = np.load(particle_file, allow_pickle=True)
    grid = np.load(grid_file, allow_pickle=True)

    position = np.asarray(particle["position"], dtype=np.float64)
    velocity = np.asarray(particle["velocity"], dtype=np.float64)
    pressure = np.asarray(particle["pressure"], dtype=np.float64)
    active = np.asarray(particle["active"])
    volume = np.asarray(particle["volume"], dtype=np.float64)
    mass = np.asarray(particle["mass"], dtype=np.float64)

    cell_type = cell_array(grid, "cell_type", np.int32)
    solid_sdf = cell_array(grid, "cell_solid_sdf", np.float64)
    fluid_sdf = cell_array(grid, "cell_fluid_sdf", np.float64)
    cell_pressure = cell_array(grid, "cell_pressure", np.float64)

    center = np.asarray(args.center, dtype=np.float64)
    domain = np.asarray(args.domain, dtype=np.float64)
    ghost = int(args.ghost_cell)
    cnum = np.asarray(cell_type.shape[:2], dtype=np.int64)
    active_cnum = cnum - 2 * ghost
    grid_size = domain / active_cnum.astype(np.float64)
    min_dx = float(np.min(grid_size))

    ii, jj = np.meshgrid(np.arange(cnum[0]), np.arange(cnum[1]), indexing="ij")
    logical = np.stack((ii - ghost, jj - ghost), axis=-1).astype(np.float64)
    centers = (logical + 0.5) * grid_size
    analytic_phi = np.linalg.norm(centers - center, axis=-1) - args.radius

    active_mask = (ii >= ghost) & (ii < cnum[0] - ghost) & (jj >= ghost) & (jj < cnum[1] - ghost)
    inside_circle = active_mask & (analytic_phi < 0.0)
    near_circle = active_mask & (np.abs(analytic_phi) < args.near_band_cells * min_dx)
    solid_cells = active_mask & (cell_type == 2)
    solid_near_circle = solid_cells & (np.linalg.norm(centers - center, axis=-1) < 2.0 * args.radius)
    negative_sdf_near_circle = near_circle & (solid_sdf < 0.0)

    analytic_particle_phi = np.linalg.norm(position[:, :2] - center, axis=1) - args.radius
    grid_particle_phi = interpolate_cell_sdf(position, solid_sdf, grid_size, ghost)
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    sdf_error = np.abs(solid_sdf - analytic_phi)

    finite = {
        "position": bool(np.isfinite(position).all()),
        "velocity": bool(np.isfinite(velocity).all()),
        "particle_pressure": bool(np.isfinite(pressure).all()),
        "volume": bool(np.isfinite(volume).all()),
        "mass": bool(np.isfinite(mass).all()),
        "cell_solid_sdf": bool(np.isfinite(solid_sdf).all()),
        "cell_fluid_sdf": bool(np.isfinite(fluid_sdf).all()),
        "cell_pressure": bool(np.isfinite(cell_pressure).all()),
    }

    particles_near = int(np.count_nonzero(grid_particle_phi < args.near_band_cells * min_dx))
    report = {
        "latest_particle_file": particle_file,
        "latest_grid_file": grid_file,
        "time": float(particle["t_current"]),
        "particle_count": int(position.shape[0]),
        "active_particle_count": int(np.count_nonzero(active)),
        "grid_cell_shape": tuple(int(x) for x in cnum),
        "active_cell_shape": tuple(int(x) for x in active_cnum),
        "grid_size": tuple(float(x) for x in grid_size),
        "finite": finite,
        "position_min": position[:, :2].min(axis=0).tolist(),
        "position_max": position[:, :2].max(axis=0).tolist(),
        "max_speed": float(speed.max()),
        "mean_speed": float(speed.mean()),
        "pressure_minmax": [float(pressure.min()), float(pressure.max())],
        "cell_pressure_minmax": [float(cell_pressure.min()), float(cell_pressure.max())],
        "solid_cell_count": int(np.count_nonzero(solid_cells)),
        "analytic_circle_inside_cell_count": int(np.count_nonzero(inside_circle)),
        "solid_cells_near_circle": int(np.count_nonzero(solid_near_circle)),
        "negative_sdf_near_circle": int(np.count_nonzero(negative_sdf_near_circle)),
        "max_abs_sdf_error_near_circle": float(sdf_error[near_circle].max()) if np.any(near_circle) else None,
        "mean_abs_sdf_error_near_circle": float(sdf_error[near_circle].mean()) if np.any(near_circle) else None,
        "min_particle_grid_solid_sdf": float(grid_particle_phi.min()),
        "particles_inside_grid_solid_sdf": int(np.count_nonzero(grid_particle_phi < -args.penetration_tol)),
        "min_particle_phi_to_analytic_circle": float(analytic_particle_phi.min()),
        "particles_inside_analytic_circle": int(np.count_nonzero(analytic_particle_phi < -args.penetration_tol)),
        "particles_near_circle_band": particles_near,
        "fluid_cell_count": int(np.count_nonzero(active_mask & (cell_type == 1))),
        "air_cell_count": int(np.count_nonzero(active_mask & (cell_type == 0))),
    }

    failures = []
    if not all(finite.values()):
        failures.append("nonfinite output field")
    if np.any(position[:, 0] < -args.penetration_tol) or np.any(position[:, 0] > domain[0] + args.penetration_tol):
        failures.append("particle x position outside domain")
    if np.any(position[:, 1] < -args.penetration_tol) or np.any(position[:, 1] > domain[1] + args.penetration_tol):
        failures.append("particle y position outside domain")
    if report["particles_inside_grid_solid_sdf"] != 0:
        failures.append("particle penetrated saved grid solid SDF")
    if report["negative_sdf_near_circle"] == 0:
        failures.append("saved cell_solid_sdf does not contain the circular obstacle")
    if report["max_abs_sdf_error_near_circle"] is not None and report["max_abs_sdf_error_near_circle"] > args.sdf_tol:
        failures.append("saved cell_solid_sdf near circle does not match analytic circle SDF")
    if not args.allow_no_near_particles and particles_near == 0:
        failures.append("no particles are near the circular SDF obstacle; the run does not exercise SDF coupling")

    return report, failures


def main():
    report, failures = validate(parse_args())
    print(json.dumps(report, indent=2))
    if failures:
        print("VALIDATION FAIL")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)
    print("VALIDATION PASS")


if __name__ == "__main__":
    main()
