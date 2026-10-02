#!/usr/bin/env python3
"""Blender gallery for saved two-phase 2-D MPM particles; no simulation reruns.

python tools/mmpm_gifs.py --case all --work-dir /tmp/mmpm_gifs
The thin extrusion is display geometry, not a simulated third dimension.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from blender_cfdem_gif import ROOT, export, water_surface
from incompressible_gifs import BLENDER, box_mesh

CASES = {
    "porous_dam": (
        "DamBreakPorousElastic2D/OutputData/paper_case1_beetstra_rigid_final",
        "dam_break_porous_elastic_2d.gif",
        "Dam break through a porous column",
        [0.9, 0.38],
        0.01,
        0.06,
    ),
    "landslide": (
        "SubmarineLandslide/OutputData/rzadkiewicz_postfix_20260907/model",
        "submarine_landslide.gif",
        "Submarine granular landslide",
        [4.0, 1.8],
        0.01,
        0.18,
    ),
    "utube": (
        "UTubeFlow2D/OutputData/paper_section_4_2_2d",
        "u_tube_flow_2d.gif",
        "U-tube flow through a porous bed",
        [3.0, 3.4],
        0.05,
        0.18,
    ),
}


def reconstruction_grid(case, domain, dx):
    # Display grid only: occupancy is rebuilt from saved fluid particles every frame.
    axes = [np.arange(-2, round(length / dx) + 3) * dx for length in domain]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    centers = coords[:-1, :-1] + dx / 2
    x, z = centers[..., 0], centers[..., 1]
    outside = (x < 0) | (x > domain[0]) | (z < 0) | (z > domain[1])
    if case == "landslide":
        outside |= z < x - 2.4
    elif case == "utube":
        outside |= (x >= 1) & (x <= 2) & (z >= 1)
    return {
        "dims": np.array(coords.shape[:2]),
        "coords": coords.reshape(-1, 2),
        "cell_type": (outside.astype(np.uint8) * 2)[..., None],
    }


def boundary_mesh(case, domain, depth):
    if case == "landslide":
        vertices = np.array([[x, y, z] for y in [0, depth] for x, z in [(2.4, 0), (domain[0], 0), (domain[0], 1.6)]])
        faces = np.array([[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]])
        return vertices, faces
    if case == "utube":
        boxes = [
            box_mesh([1 - 0.008, 0, 1], [1, depth, 3.35]),
            box_mesh([2, 0, 1], [2 + 0.008, depth, 3.35]),
            box_mesh([1, 0, 1], [2, depth, 1 + 0.008]),
        ]
        return np.concatenate([v for v, _ in boxes]), np.concatenate([f + 8 * i for i, (_, f) in enumerate(boxes)])
    return None


def phase_geometry(particles, grid, domain, depth, scale):
    active = particles["active"].reshape(-1) != 0
    phase = particles["phase"].reshape(-1)
    position = particles["position"]
    volume = particles["volume"].reshape(-1)
    if not np.isfinite(position[active]).all() or not np.isfinite(volume[active]).all() or np.any(volume[active] <= 0):
        raise ValueError("Invalid saved particle positions or volumes")
    fluid, solid = active & (phase == 2), active & (phase == 1)
    if not fluid.any() or not solid.any():
        raise ValueError("Both saved fluid and solid phases are required")
    water, faces = water_surface(
        grid, {"position": position[fluid], "volume": volume[fluid], "active": np.ones(fluid.sum(), dtype=bool)}, depth
    )
    # Expose the 2-D solid cross-section at the front, so water does not hide the soil.
    centers = np.column_stack((position[solid, 0], np.zeros(solid.sum()), position[solid, 1]))
    porosity = particles["porosity"].reshape(-1)[solid]
    if not np.isfinite(porosity).all() or np.any((porosity < 0) | (porosity >= 1)):
        raise ValueError("Invalid solid porosity")
    # Cross-sectional bead area follows the saved solid fraction; no invented grains/layers.
    radii = np.sqrt(volume[solid] * (1 - porosity) / np.pi)
    return {
        "water": np.clip(water, 0, [domain[0], depth, domain[1]]) * scale,
        "water_faces": faces,
        "solid_positions": centers * scale,
        "solid_radii": radii * scale,
    }


def prepare(case, folder, preview=False):
    relative, filename, title, domain, dx, depth = CASES[case]
    source = ROOT / "examples/mmpm" / relative
    files = sorted((source / "particles").glob("MPMParticle[0-9]*.npz"))
    if not files:
        raise FileNotFoundError(f"No saved particles in {source}")
    folder.mkdir(parents=True, exist_ok=True)
    scale = 3.6 / max(domain)
    grid = reconstruction_grid(case, domain, dx)
    boundary = boundary_mesh(case, domain, depth)
    frames = []
    for index, path in enumerate(files[:1] if preview else files):
        with np.load(path) as particles:
            time = float(particles["t_current"])
            if not np.isfinite(time) or (frames and time <= frames[-1]["time"]):
                raise ValueError(f"Invalid/non-increasing saved time in {path}")
            data = phase_geometry(particles, grid, domain, depth, scale)
        if boundary is not None and index == 0:
            data.update(boundary=boundary[0] * scale, boundary_faces=boundary[1])
        mesh = f"mesh_{index:06d}.npz"
        np.savez(folder / mesh, **data)
        frames.append(
            {"time": time, "step": path.stem[-6:], "mesh": mesh, "solid_particle_count": len(data["solid_positions"])}
        )
        print(f"PREPARE {case}: {index + 1}/{1 if preview else len(files)}", flush=True)
    config = {
        "case": case,
        "source": str(source),
        "output": str(ROOT / "images" / filename),
        "title": title,
        "domain": (np.array([domain[0], depth, domain[1]]) * scale).tolist(),
        "frames": frames,
        "radius": None,
        "fps": 10 if len(frames) > 20 else 5,
        "width": 960,
        "height": 880 if case == "utube" else 640,
        "camera_direction": [0.07, -1, 0.12],
        "water_color": [0.06, 0.38, 0.66],
        "water_visibility": 0.55,
        "water_transmission": 0.65,
        "fluid_surface_method": "particle_position_and_volume",
        "display_extrusion_m": depth,
        "solid_display_method": "one_bead_per_saved_solid_particle_front_section",
    }
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    return manifest, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    work = (args.work_dir or Path(tempfile.mkdtemp(prefix="mmpm_blender_"))).resolve()
    print(f"Render assets / PNGs / backups: {work}", flush=True)
    for case in CASES if args.case == "all" else [args.case]:
        manifest, config = prepare(case, work / case, args.preview)
        if args.prepare_only:
            continue
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
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True)
        if not args.preview:
            export(config, manifest.parent)


if __name__ == "__main__":
    main()
