#!/usr/bin/env python3
"""Render the requested incompressible MPM gallery, excluding n32 and large-tank 2D.

python tools/incompressible_gifs.py --case all --work-dir /tmp/incompressible_gifs
Uses the installed Blender, ParaView, numpy, scipy, scikit-image and FFmpeg.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

if __package__:
    from .blender_cfdem_gif import ROOT, water_surface, export
else:
    from blender_cfdem_gif import ROOT, water_surface, export

if not __package__:
    sys.path.insert(0, str(ROOT))
from examples.mpm.IncompressibleFluid.dam_break_visible_sdf_2d.obstacle import VERTICES as DAM_OBSTACLE_VERTICES

OUTPUTS = ROOT / "examples/mpm/IncompressibleFluid/OutputData"
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
PVPYTHON = "/Applications/ParaView-5.11.2.app/Contents/bin/pvpython"
CASES = {
    "lid_n64": (
        ROOT
        / "examples/mpm/IncompressibleFluid/lid_driven_cavity_2d/OutputData/remote_20261019/queue_utube_stage_20261019__lid_driven_cavity_2d",
        "lid_driven_cavity_n64.gif",
        "pressure",
        [1, 1],
        "Lid-driven cavity (n64)",
    ),
    "taylor_green": (
        ROOT / "examples/mpm/IncompressibleFluid/taylor_green_vortex_2d/OutputData/four_vortices_n128_ppc4_20261020",
        "taylor_green_vortex.gif",
        "velocity",
        [2 * np.pi, 2 * np.pi],
        "Taylor-Green vortex",
    ),
    "cylinder": (
        ROOT / "examples/cfdem/FullyResolved/IBMFixedCylinder2D/OutputData/ibm2d_karman_postfix_20260907/model",
        "cylinder_fluid.gif",
        "pressure",
        [1, 0.4],
        "Flow around a fixed cylinder",
    ),
    "cylinder_vorticity": (
        ROOT / "examples/cfdem/FullyResolved/IBMFixedCylinder2D/OutputData/ibm2d_karman_postfix_20260907/model",
        "cylinder_vorticity.gif",
        "vorticity",
        [1, 0.4],
        "Flow around a fixed cylinder: signed vorticity",
    ),
    "dam_2d": (
        ROOT
        / "examples/mpm/IncompressibleFluid/dam_break_visible_sdf_2d/OutputData/dam_break_star_sdf_2d_dx005_ppc2_current_20261014",
        "dam_break_star_obstacle_2d.gif",
        "blender",
        [0.9, 0.06, 0.45],
        "2D dam break around a five-point star",
    ),
    "affine": (
        ROOT
        / "examples/mpm/IncompressibleFluid/affine_body_coupling_3d/OutputData/incompressible_affine_gpu_20step_final",
        "affine_body_fluid_3d.gif",
        "blender",
        [0.32, 0.2, 0.28],
        "Submerged affine body",
    ),
    "large_tank": (
        OUTPUTS / "large_tank_incompressible_3d_complete_ppc2",
        "large_tank_3d.gif",
        "blender",
        [6, 1.42, 6],
        "3D large-tank dam break",
    ),
    "wavemaker": (
        OUTPUTS / "wavemaker_tank_3d_wave_train_no_sponge_v2",
        "wavemaker_3d.gif",
        "blender",
        [2.4, 0.32, 0.64],
        "3D piston wavemaker",
    ),
}


def box_mesh(lower, upper):
    vertices = np.array(
        [[x, y, z] for x in [lower[0], upper[0]] for y in [lower[1], upper[1]] for z in [lower[2], upper[2]]]
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ]
    )
    return vertices, faces


def polygon_prism_mesh(vertices_2d, thickness):
    count = len(vertices_2d)
    front = np.column_stack((vertices_2d[:, 0], np.zeros(count), vertices_2d[:, 1]))
    # A center fan preserves the concave star; a fan from its first tip fills notches.
    # ponytail: these gallery polygons are star-shaped about their mean, not arbitrary polygons.
    front = np.vstack((front, front.mean(axis=0)))
    back = front.copy()
    back[:, 1] = thickness
    faces = []
    stride = count + 1
    for index in range(count):
        next_index = (index + 1) % count
        faces.extend(((count, index, next_index), (stride + count, stride + next_index, stride + index)))
        faces.extend(((index, stride + next_index, next_index), (index, stride + index, stride + next_index)))
    return np.vstack((front, back)), np.asarray(faces, dtype=np.int32)


def particle_vorticity(particles, grid_spacing):
    """Area-average the signed curl of linear velocity on a particle triangulation."""
    from scipy.spatial import Delaunay

    active = np.asarray(particles["active"]).reshape(-1) != 0
    positions = np.asarray(particles["position"], dtype=float)[active]
    velocity = np.asarray(particles["velocity"], dtype=float)[active]
    if (
        positions.ndim != 2
        or positions.shape[1] != 2
        or len(positions) < 3
        or velocity.shape != positions.shape
        or not np.isfinite(positions).all()
        or not np.isfinite(velocity).all()
        or not np.isfinite(grid_spacing)
        or grid_spacing <= 0
    ):
        raise ValueError("Vorticity requires finite 2D particle velocities and positive grid spacing")
    triangles = Delaunay(positions).simplices
    edges = positions[triangles[:, 1:]] - positions[triangles[:, 0, None]]
    area = np.abs(np.linalg.det(edges))
    radius = np.linalg.norm(edges, axis=2).prod(axis=1) * np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1) / (2 * area)
    # shortcut: obstacle radii must exceed two grid cells; use the saved SDF for smaller obstacles.
    triangles = triangles[(area > 0) & (radius <= 2 * grid_spacing)]
    if not len(triangles):
        raise ValueError("No resolved fluid triangles for vorticity")
    edges = positions[triangles[:, 1:]] - positions[triangles[:, 0, None]]
    area = np.abs(np.linalg.det(edges))
    differences = velocity[triangles[:, 1:]] - velocity[triangles[:, 0, None]]
    gradient = np.linalg.solve(edges, differences)
    curl = gradient[:, 0, 1] - gradient[:, 1, 0]
    values, weights = np.zeros(len(positions)), np.zeros(len(positions))
    np.add.at(values, triangles.ravel(), np.repeat(curl * area, 3))
    np.add.at(weights, triangles.ravel(), np.repeat(area, 3))
    used, indices = np.unique(triangles, return_inverse=True)
    vorticity = values[used] / weights[used]
    if not np.isfinite(vorticity).all():
        raise ValueError("Non-finite reconstructed vorticity")
    return positions[used], indices.reshape(-1, 3), vorticity


def prepare(case, folder, preview=False, source=None):
    default_source, filename, mode, domain, title = CASES[case]
    source = Path(source).resolve() if source else default_source
    if case == "taylor_green" and (source / "metrics.json").is_file():
        domain = np.asarray(json.loads((source / "metrics.json").read_text())["domain_m"], dtype=float)
        if domain.shape != (2,) or not np.isfinite(domain).all() or np.any(domain <= 0):
            raise ValueError("Taylor-Green rendering requires a finite positive saved 2D domain")
    folder.mkdir(parents=True, exist_ok=True)
    files = sorted((source / "particles").glob("MPMParticle[0-9]*.npz"))
    if not files:
        raise FileNotFoundError(f"No saved particles in {source}")
    scale = 3.6 / max(domain) if mode == "blender" else 1.0
    frames = []
    pressure_values = []
    peak_water_height = 0.0
    for index, path in enumerate(files[:1] if preview else files):
        step = path.stem[-6:]
        with np.load(path) as particles:
            frame = {"time": float(particles["t_current"]), "step": step}
            if mode == "blender":
                with np.load(source / f"grids/MPMGrid{step}.npz") as grid:
                    if abs(frame["time"] - float(grid["t_current"])) > 1e-8:
                        raise ValueError(f"Fluid/grid time mismatch at {path}")
                    vertices, faces = water_surface(grid, particles, domain[1] if case == "dam_2d" else None)
                    vertices = np.clip(vertices, 0, domain)
                    data = {
                        "water": vertices * scale,
                        "water_faces": faces,
                    }
                    peak_water_height = max(peak_water_height, float(data["water"][:, 2].max()))
                    if case == "dam_2d":
                        solid, triangles = polygon_prism_mesh(DAM_OBSTACLE_VERTICES, domain[1])
                    elif case == "affine":
                        with np.load(source / f"particles/AffineBody{step}.npz") as body:
                            if abs(frame["time"] - float(body["t_current"])) > 1e-8:
                                raise ValueError(f"Affine body time mismatch at {step}")
                            solid, triangles = body["vertices"], body["faces"]
                    elif case == "wavemaker":
                        dims = grid["dims"]
                        coords = grid["coords"].reshape(*dims, 3)
                        dx = coords[1, 0, 0, 0] - coords[0, 0, 0, 0]
                        phi = grid["cell_solid_sdf"][:, dims[1] // 2, dims[2] // 2, 0]
                        crossing = np.flatnonzero((phi[:-1] < 0) & (phi[1:] >= 0))[0]
                        x = coords[crossing, 0, 0, 0] + dx / 2
                        piston = x + dx * (-phi[crossing]) / (phi[crossing + 1] - phi[crossing])
                        solid, triangles = box_mesh([0, 0, 0], [piston, domain[1], domain[2]])
                    if case in ("dam_2d", "affine", "wavemaker"):
                        data.update(grains=solid * scale, grain_faces=triangles)
                    frame["mesh"] = f"mesh_{index:06d}.npz"
                    np.savez(folder / frame["mesh"], **data)
            elif mode == "vorticity":
                import meshio

                with np.load(source / f"grids/MPMGrid{step}.npz") as grid:
                    if abs(frame["time"] - float(grid["t_current"])) > 1e-8:
                        raise ValueError(f"Velocity/grid time mismatch at {path}")
                    coords = grid["coords"].reshape(*grid["dims"], 2)
                    spacing = float(coords[1, 0, 0] - coords[0, 0, 0])
                    positions, triangles, vorticity = particle_vorticity(particles, spacing)
                derived = folder / "vorticity_vtks"
                derived.mkdir(exist_ok=True)
                meshio.write_points_cells(
                    derived / f"GraphicMPMParticle{step}.vtu",
                    np.column_stack((positions, np.zeros(len(positions)))),
                    [("triangle", triangles)],
                    point_data={"vorticity_z": vorticity},
                )
            elif mode == "velocity":
                velocity = np.asarray(particles["velocity"], dtype=float)
                if "active" in particles:
                    velocity = velocity[particles["active"].reshape(-1) != 0]
                if (
                    velocity.ndim != 2
                    or velocity.shape[1] not in (2, 3)
                    or not len(velocity)
                    or not np.isfinite(velocity).all()
                ):
                    raise ValueError(f"Empty or non-finite particle velocity in {path}")
                speed = np.linalg.norm(velocity, axis=1)
                frame["speed_range"] = [float(speed.min()), float(speed.max())]
            else:
                pressure = particles["pressure"].reshape(-1)
                if "active" in particles:
                    pressure = pressure[particles["active"].reshape(-1) != 0]
                if not len(pressure) or not np.isfinite(pressure).all():
                    raise ValueError(f"Empty or non-finite pressure in {path}")
                pressure_values.append(pressure)
            frames.append(frame)
    if not preview and not np.isclose(frames[0]["time"], 0.0):
        raise ValueError("Gallery rendering requires saved history starting at time zero")
    config = {
        "case": case,
        "source": str(source),
        "output": str(ROOT / "images" / filename),
        "title": title,
        "domain": (np.array(domain) * scale).tolist(),
        "frames": frames,
        "radius": None,
        "fps": 10 if len(frames) > 20 else 5 if len(frames) > 2 else 2,
        "width": 1200 if case.startswith("cylinder") else 960,
        "height": 560 if case.startswith("cylinder") else 720,
        "solid_material": "metal" if case in ("affine", "wavemaker") else "soil",
        "camera_direction": [0.15, -1, 0.2] if case == "dam_2d" else [0.25, -1, 0.4],
        "fluid_surface_method": "particle_position_and_volume" if mode == "blender" else "pressure_points",
    }
    if case == "large_tank":
        config.update(
            display_domain=[domain[0] * scale, domain[1] * scale, min(domain[2] * scale, peak_water_height * 1.12)]
        )
    if case in ("dam_2d", "large_tank", "wavemaker"):
        config.update(water_color=[0.06, 0.38, 0.66], water_visibility=0.72, water_transmission=0.55)
    if mode == "pressure":
        # ponytail: fixed percentiles saturate rare peaks; keep raw data for extrema analysis.
        quantiles = np.percentile(np.concatenate(pressure_values), [1, 99])
        extent = float(np.abs(quantiles).max()) or 1.0
        extent = {"cylinder": 0.2, "lid_n64": 0.05, "taylor_green": 0.05}.get(case, extent)
        config.update(color_range=[-extent, extent], pressure_percentiles=quantiles.tolist(), color_range_mode="fixed")
    if mode != "blender":
        config.update(scalar="pressure", vtk_source=str(source / "vtks"), colorbar_title="pressure (fixed scale)")
    if mode == "vorticity":
        config.update(
            scalar="vorticity_z",
            vtk_source=str(folder / "vorticity_vtks"),
            fluid_surface_method="particle_velocity_triangulation",
            color_range=[-20.0, 20.0],
            color_range_mode="fixed",
            colorbar_title="vorticity z [1/s]",
            vorticity_method="Area-weighted curl of linear saved particle velocities; circumradius cutoff 2*grid_spacing",
        )
    if mode == "velocity":
        config.update(
            scalar="velocity",
            scalar_mode="magnitude",
            fluid_surface_method="velocity_points",
            color_range=[0.0, max(frame["speed_range"][1] for frame in frames) or 1.0],
            color_range_mode="frame-magnitude",
            color_preset="Viridis (matplotlib)",
            colorbar_title="speed [m/s]",
        )
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    return manifest, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--source", type=Path, help="Saved result directory for a single case")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.source and args.case == "all":
        parser.error("--source requires a single --case")
    work = (args.work_dir or Path(tempfile.mkdtemp(prefix="incompressible_gifs_"))).resolve()
    for case in CASES if args.case == "all" else [args.case]:
        folder = work / case
        manifest, config = prepare(case, folder, args.preview, args.source)
        if args.prepare_only:
            continue
        if CASES[case][2] == "blender":
            command = [
                BLENDER,
                "-b",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(ROOT / "tools/blender_cfdem_gif.py"),
                "--",
                "--render",
                str(manifest),
                "--samples",
                str(args.samples),
            ]
        else:
            domain = config["domain"]
            focal = [(0.58 if case.startswith("cylinder") else 0.6) * domain[0], 0.5 * domain[1], 0]
            parallel_scale = max(0.55 * domain[1], domain[0] * config["height"] / (1.68 * config["width"]))
            command = [
                PVPYTHON,
                "--force-offscreen-rendering",
                str(ROOT / "tools/vtu2gif.py"),
                config["vtk_source"],
                "--scalar",
                config["scalar"],
                "--scalar-mode",
                config.get("scalar_mode", "auto"),
                "--camera-view",
                "custom",
                "--camera-position",
                str(focal[0]),
                str(focal[1]),
                str(3 * max(domain)),
                "--camera-focal-point",
                *map(str, focal),
                "--parallel-scale",
                str(parallel_scale),
                "--camera-view-up",
                "0",
                "1",
                "0",
                "--mpm-point-size",
                "3",
                "--width",
                str(config["width"]),
                "--height",
                str(config["height"]),
                "--frames-dir",
                str(folder),
                "--skip-gif",
                "--keep-frames",
            ]
            command.extend(
                [
                    "--color-min",
                    str(config["color_range"][0]),
                    "--color-max",
                    str(config["color_range"][1]),
                    "--color-preset",
                    config.get("color_preset", "Cool to Warm"),
                    "--color-range-mode",
                    config["color_range_mode"],
                    "--colorbar-title",
                    config["colorbar_title"],
                ]
            )
            if CASES[case][2] == "vorticity":
                command.extend(["--mpm-representation", "Surface"])
            if args.preview:
                command.extend(["--end-frame", "0"])
        subprocess.run(command, check=True)
        if not args.preview:
            export(config, folder)


if __name__ == "__main__":
    main()
