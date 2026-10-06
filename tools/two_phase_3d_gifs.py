#!/usr/bin/env python3
"""Blender GIFs from saved 3-D two-phase particles and synchronized LSDEM meshes."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from blender_cfdem_gif import ROOT, export, grain_surface, water_surface
from incompressible_gifs import BLENDER, box_mesh

CASES = {
    "sphere_impact": (
        "TwoPhaseLSDEMCoupling/OutputData/sphere_impact_section_5_2_formal_priority_v1",
        "sphere_impact_saturated_soil.gif",
        "Sphere impact into saturated soil",
        [0.2, 0.2, 0.2],
        None,
    ),
    "lsdem_wavemaker": (
        "TwoPhaseLSDEMCoupling/OutputData/wavemaker_lsdem_particles_3d_formal_v13",
        "two_phase_lsdem_wavemaker.gif",
        "Two-phase wavemaker with LSDEM spheres",
        [1.2, 0.24, 0.36],
        [0.06, 0.8, 0.10, 0.75],
    ),
    "wavemaker": (
        "TwoPhaseWavemaker3D/OutputData/two_layer_two_phase_wavemaker_3d_wave_train_no_sponge_v2",
        "two_phase_wavemaker.gif",
        "Two-phase wavemaker over a saturated slope",
        [2.4, 0.48, 0.72],
        [0.08, 0.9, 0.24, 1.2],
    ),
}


def phase_geometry(
    particles,
    grid,
    domain,
    scale,
    close_particle_voids=False,
    radial_limit=None,
    occupancy_threshold=0.35,
):
    active = particles["active"].reshape(-1) != 0
    phase = particles["phase"].reshape(-1)
    positions = particles["position"]
    volumes = particles["volume"].reshape(-1)
    if (
        not np.isfinite(positions[active]).all()
        or not np.isfinite(volumes[active]).all()
        or np.any(volumes[active] <= 0)
    ):
        raise ValueError("Invalid saved positions or volumes")
    fluid, solid = active & (phase == 2), active & (phase == 1)
    if not fluid.any() or not solid.any():
        raise ValueError("Both fluid and solid phases are required")
    vertices, faces = water_surface(
        grid,
        {"position": positions[fluid], "volume": volumes[fluid], "active": np.ones(fluid.sum(), dtype=bool)},
        close_particle_voids=close_particle_voids,
        occupancy_threshold=occupancy_threshold,
    )
    if radial_limit is not None:
        center = 0.5 * np.asarray(domain[:2], dtype=float)
        radial = vertices[:, :2] - center
        distance = np.linalg.norm(radial, axis=1)
        outside = distance > radial_limit
        vertices[outside, :2] = center + radial[outside] * (radial_limit / distance[outside])[:, None]
    porosity = particles["porosity"].reshape(-1)[solid]
    if not np.isfinite(porosity).all() or np.any((porosity < 0) | (porosity >= 1)):
        raise ValueError("Invalid solid porosity")
    return {
        "water": np.clip(vertices, 0, domain) * scale,
        "water_faces": faces,
        "solid_positions": positions[solid] * scale,
        "solid_radii": np.cbrt(3 * volumes[solid] * (1 - porosity) / (4 * np.pi)) * scale,
    }


def cylinder_outline(domain, scale):
    # Match the saved case's 32 tangent-plane walls, not a rectangular display tank.
    angle = np.arange(32) * 2 * np.pi / 32 + np.pi / 32
    radius = domain[0] / 2 / np.cos(np.pi / 32)
    ring = np.column_stack((domain[0] / 2 + radius * np.cos(angle), domain[1] / 2 + radius * np.sin(angle)))
    segments = []
    for height in (0, domain[2]):
        points = np.column_stack((ring, np.full(32, height)))
        segments.extend(np.stack((points, np.roll(points, -1, axis=0)), axis=1))
    for i in (3, 11, 19, 27):
        segments.append([[*ring[i], 0], [*ring[i], domain[2]]])
    return np.asarray(segments) * scale


def prepare(case, folder, preview=False, allow_failed=False, source=None):
    relative, filename, title, domain, piston = CASES[case]
    source = source or ROOT / "examples/mmpm" / relative
    metrics = json.loads((source / "metrics.json").read_text())
    if not metrics.get("passed") and not allow_failed:
        raise ValueError(f"Source validation failed: {source}")
    paths = sorted((source / "particles").glob("MPMParticle[0-9]*.npz"))
    if not paths:
        raise FileNotFoundError(f"No saved particles in {source}")
    grid_steps = {p.stem[-6:] for p in (source / "grids").glob("MPMGrid[0-9]*.npz")}
    if {p.stem[-6:] for p in paths} != grid_steps:
        raise ValueError(f"Incomplete particle/grid frame pairs: {source}")
    with np.load(paths[-1]) as last:
        if abs(float(last["t_current"]) - metrics["final_time_s"]) > 1e-7:
            raise ValueError(f"Incomplete local particle data: {source}")
    folder.mkdir(parents=True, exist_ok=True)
    scale = 3.6 / max(domain)
    frames = []
    for index, path in enumerate(paths[:1] if preview else paths):
        step = path.stem[-6:]
        with np.load(path) as particles, np.load(source / f"grids/MPMGrid{step}.npz") as grid:
            time = float(particles["t_current"])
            if (
                not np.isfinite(time)
                or abs(time - float(grid["t_current"])) > 1e-8
                or (frames and time <= frames[-1]["time"])
            ):
                raise ValueError(f"Invalid or mismatched frame time: {step}")
            data = phase_geometry(
                particles,
                grid,
                domain,
                scale,
                close_particle_voids=True,
                radial_limit=0.5 * domain[0] if case == "sphere_impact" else None,
                occupancy_threshold=0.35,
            )
        if case != "wavemaker":
            with (
                np.load(source / f"particles/LSDEMRigid{step}.npz") as rigid,
                np.load(source / f"particles/LSDEMSurface{step}.npz") as surface,
            ):
                if any(abs(time - float(d["t_current"])) > 1e-8 for d in (rigid, surface)):
                    raise ValueError(f"LSDEM time mismatch: {step}")
                vertices, faces = grain_surface(surface, rigid)
                data.update(rigid=vertices * scale, rigid_faces=faces)
        if piston:
            mean, frequency, velocity, ramp_time = piston
            omega = 2 * np.pi * frequency
            ramp = 1.0 if time >= ramp_time else 0.5 * (1 - np.cos(np.pi * time / ramp_time))
            position = mean + ramp * velocity / omega * np.sin(omega * time)
            vertices, faces = box_mesh([0, 0, 0], [position, domain[1], domain[2]])
            data.update(grains=vertices * scale, grain_faces=faces)
        elif index == 0:
            data["wall_segments"] = cylinder_outline(domain, scale)
        mesh = f"mesh_{index:06d}.npz"
        np.savez_compressed(folder / mesh, **data)
        frames.append({"time": time, "step": step, "mesh": mesh})
        print(f"PREPARE {case}: {index + 1}/{len(paths[:1] if preview else paths)}", flush=True)
    config = {
        "case": case,
        "source": str(source),
        "output": str(ROOT / "images" / filename),
        "title": title,
        "domain": (np.array(domain) * scale).tolist(),
        "frames": frames,
        "fps": 10,
        "width": 960,
        "height": 800 if case == "sphere_impact" else 720,
        "camera_direction": [0.8, -1.8, 0.8] if case == "sphere_impact" else [0.25, -1, 0.4],
        "show_tank": case != "sphere_impact",
        "solid_material": "metal",
        "water_color": [0.06, 0.38, 0.66],
        "water_visibility": 0.55,
        "water_transmission": 0.55,
        "fluid_surface_method": "particle_position_and_volume_with_case_topology_guard",
        "solid_display_method": "one_volume_equivalent_bead_per_saved_solid_particle",
        "legend": "Blue: water | Brown: saturated soil" + (" | Dark blue: LSDEM" if case != "wavemaker" else ""),
    }
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    return manifest, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failed", action="store_true", help="Render a failed run for diagnosis")
    parser.add_argument("--source", type=Path, help="Override the saved result directory")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="two_phase_3d_"))
    manifest, config = prepare(args.case, work / args.case, args.preview, args.allow_failed, args.source)
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
    if not args.preview:
        export(config, manifest.parent)


if __name__ == "__main__":
    main()
