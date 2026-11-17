"""Render the saved DEM sphere and MPM water in the semi-resolved impact case."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

if __package__:
    from .blender_cfdem_gif import ROOT, export, water_surface
    from .incompressible_gifs import BLENDER
else:
    from blender_cfdem_gif import ROOT, export, water_surface
    from incompressible_gifs import BLENDER


def prepare(source, folder):
    source, folder = Path(source).resolve(), Path(folder).resolve()
    paths = sorted((source / "particles").glob("MPMParticle[0-9]*.npz"))
    if not paths:
        raise FileNotFoundError(f"No saved fluid particles in {source}")
    folder.mkdir(parents=True, exist_ok=True)
    domain = np.array([0.1, 0.1, 0.15])
    scale, frames, radius = 3.6 / max(domain), [], None
    for index, path in enumerate(paths):
        step = path.stem[-6:]
        with (
            np.load(path) as fluid,
            np.load(source / f"grids/MPMGrid{step}.npz") as grid,
            np.load(source / f"particles/DEMParticle{step}.npz") as sphere,
        ):
            times = np.array([float(z["t_current"]) for z in (fluid, grid, sphere)])
            time = float(times[0])
            if not np.isfinite(times).all() or np.any(np.abs(times - time) > 1e-8):
                raise ValueError(f"Mismatched fluid/grid/DEM times at {step}")
            if frames and time <= frames[-1]["time"]:
                raise ValueError("Saved times must increase")
            active = sphere["active"].reshape(-1) != 0
            centers, radii = sphere["position"][active], sphere["radius"].reshape(-1)[active]
            if (
                centers.shape != (1, 3)
                or radii.shape != (1,)
                or not np.isfinite(centers).all()
                or not np.isfinite(radii).all()
                or radii[0] <= 0
            ):
                raise ValueError("Exactly one finite saved DEM sphere with positive radius is required")
            if radius is not None and not np.isclose(radii[0] * scale, radius, rtol=1e-10, atol=1e-12):
                raise ValueError("Saved sphere radius changes between frames")
            radius = float(radii[0] * scale)
            vertices, faces = water_surface(grid, fluid)
            mesh = f"mesh_{index:06d}.npz"
            np.savez(folder / mesh, water=np.clip(vertices, 0, domain) * scale, water_faces=faces)
            frames.append(dict(time=time, mesh=mesh, centers=(centers * scale).tolist()))
    if abs(frames[0]["time"]) > 1e-10:
        raise ValueError("Gallery rendering requires saved history starting at time zero")
    config = dict(
        case="water_impact",
        source=str(source),
        output=str(ROOT / "images/sphere_impact_water.gif"),
        title="Sphere impacting water (semi-resolved CFDEM)",
        domain=(domain * scale).tolist(),
        frames=frames,
        radius=radius,
        fps=10,
        width=960,
        height=800,
        fluid_surface_method="particle_position_and_volume",
        solid_position_method="saved_DEMParticle_position_and_radius",
    )
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    return manifest, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["water_impact"], default="water_impact")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="semi_resolved_gif_"))
    manifest, config = prepare(args.source, work / args.case)
    if args.prepare_only:
        return
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
    export(config, manifest.parent)


if __name__ == "__main__":
    main()
