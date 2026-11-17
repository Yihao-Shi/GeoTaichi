#!/usr/bin/env python3
"""Blender GIFs from saved 3-D two-phase particles and synchronized LSDEM meshes."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

if __package__:
    from .blender_cfdem_gif import ROOT, export, grain_surface, water_surface
    from .incompressible_gifs import BLENDER, box_mesh
else:
    from blender_cfdem_gif import ROOT, export, grain_surface, water_surface
    from incompressible_gifs import BLENDER, box_mesh
from examples.mmpm.GranularWaterLeakage3D.granular_water_leakage_3d_parameters import (
    DOMAIN as LEAKAGE_DOMAIN,
)

CASES = {
    "leakage": (
        "GranularWaterLeakage3D/OutputData/section_5_1_current_formal_20261014",
        "granular_water_leakage_3d.gif",
        "Granular-water leakage without an elastic plate",
        LEAKAGE_DOMAIN,
        None,
    ),
    "sphere_impact": (
        "TwoPhaseLSDEMCoupling/OutputData/sphere_impact_section_5_2_formal_priority_v1",
        "sphere_impact_saturated_soil.gif",
        "Sphere impact into saturated soil",
        [0.2, 0.2, 0.2],
        None,
    ),
    "lsdem_wavemaker": (
        "TwoPhaseLSDEMCoupling/OutputData/lsdem_trunk_wavemaker_ppc2_formal_pressure_wall_20261011",
        "two_phase_lsdem_wavemaker.gif",
        "Two-phase wavemaker with LSDEM trunk and irregular grains",
        [1.2, 0.24, 0.36],
        [0.06, 0.8, 0.10, 0.75],
    ),
    "wavemaker": (
        "TwoPhaseWavemaker3D/OutputData/two_phase_wavemaker_ppc2_formal_pressure_wall_20261011",
        "two_phase_wavemaker.gif",
        "Two-phase wavemaker over a saturated slope",
        [2.4, 0.48, 0.72],
        [0.08, 0.9, 0.24, 1.2],
    ),
    "bunny_impact": (
        "TwoPhaseLSDEMCoupling/OutputData/bunny_impact_ppc2_formal_pressure_wall_20261011",
        "bunny_impact_saturated_soil.gif",
        "Bunny impact into saturated soil",
        [0.2, 0.2, 0.2],
        None,
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
    solid_surface=False,
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
    if radial_limit is not None:
        grid = dict(grid)
        dims = grid["dims"]
        coords = grid["coords"].reshape(*dims, 3)
        spacing = np.array([coords[tuple(int(j == d) for j in range(3))][d] - coords[0, 0, 0, d] for d in range(3)])
        grid["solid_planes"] = cylinder_planes(domain, spacing[2], radial_limit)
    if fluid.any():
        fluid_support = {
            "position": positions[fluid],
            "volume": volumes[fluid],
            "active": np.ones(fluid.sum(), dtype=bool),
        }
        if solid_surface:
            from scipy.spatial import cKDTree

            # A quadratic MPM basis supports fluid/soil overlap within 1.5 native cells.
            support_radius = 1.5 * min(np.asarray(domain) / (grid["dims"] - 1))
            wet = cKDTree(positions[fluid]).query(positions[solid])[0] <= support_radius
            if wet.any():
                fluid_support.update(wet_soil_position=positions[solid][wet], wet_soil_volume=volumes[solid][wet])
        vertices, faces = water_surface(
            grid,
            fluid_support,
            close_particle_voids=close_particle_voids,
            occupancy_threshold=occupancy_threshold,
        )
    else:
        vertices, faces = np.empty((0, 3)), np.empty((0, 3), dtype=int)
    porosity = particles["porosity"].reshape(-1)[solid]
    if not np.isfinite(porosity).all() or np.any((porosity < 0) | (porosity >= 1)):
        raise ValueError("Invalid solid porosity")
    vertices = np.clip(vertices, 0, domain)
    if radial_limit is not None:
        for angle in np.arange(32) * 2 * np.pi / 32:
            normal = np.array([np.cos(angle), np.sin(angle)])
            overlap = ((vertices[:, :2] - np.asarray(domain[:2]) / 2) * normal).sum(axis=1) - radial_limit
            vertices[:, :2] -= np.maximum(overlap, 0)[:, None] * normal
    data = {
        "water": vertices * scale,
        "water_faces": faces,
        "solid_positions": positions[solid] * scale,
        "solid_radii": np.cbrt(3 * volumes[solid] * (1 - porosity) / (4 * np.pi)) * scale,
    }
    if solid_surface:
        soil, soil_faces = water_surface(
            grid,
            {"position": positions[solid], "volume": volumes[solid], "active": np.ones(solid.sum(), dtype=bool)},
            occupancy_threshold=min(0.95, occupancy_threshold + 0.05),
        )
        if "solid_planes" in grid:
            # Separate coincident phase faces without changing the deposition support.
            inset = 0.05 * min(np.asarray(domain) / (grid["dims"] - 1))
            for plane in grid["solid_planes"]:
                overlap = (soil * plane[:3]).sum(axis=1) - plane[-1] + inset
                soil -= np.maximum(overlap, 0)[:, None] * plane[:3]
        data["soil"], data["soil_faces"] = soil * scale, soil_faces
        del data["solid_positions"], data["solid_radii"]
    return data


def cylinder_planes(domain, floor, radius=None):
    angles = np.arange(32) * 2 * np.pi / 32
    normals = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(32)))
    offsets = (normals[:, :2] * (np.asarray(domain[:2]) / 2)).sum(axis=1) + (
        domain[0] / 2 if radius is None else radius
    )
    return np.vstack((np.column_stack((normals, offsets)), [0, 0, -1, -floor]))


def cylinder_outline(domain, scale, floor=0):
    # Match the saved case's 32 tangent-plane walls, not a rectangular display tank.
    angle = np.arange(32) * 2 * np.pi / 32 + np.pi / 32
    radius = domain[0] / 2 / np.cos(np.pi / 32)
    ring = np.column_stack((domain[0] / 2 + radius * np.cos(angle), domain[1] / 2 + radius * np.sin(angle)))
    segments = []
    for height in (floor, domain[2]):
        points = np.column_stack((ring, np.full(32, height)))
        segments.extend(np.stack((points, np.roll(points, -1, axis=0)), axis=1))
    for i in (3, 11, 19, 27):
        segments.append([[*ring[i], floor], [*ring[i], domain[2]]])
    return np.asarray(segments) * scale


def wavemaker_wall_geometry(grid, domain, piston_position):
    dims = grid["dims"]
    coords = grid["coords"].reshape(*dims, 3)
    spacing = np.array([coords[tuple(int(j == d) for j in range(3))][d] - coords[0, 0, 0, d] for d in range(3)])
    centers = coords[:-1, :-1, :-1] + spacing / 2
    lower = np.array([piston_position, spacing[1], spacing[2]])
    upper = np.asarray(domain) - [spacing[0], spacing[1], 0]
    # The top is open; only the four saved side/bottom walls and piston bound the water.
    sdf = np.minimum.reduce(
        [
            centers[..., 0] - lower[0],
            upper[0] - centers[..., 0],
            centers[..., 1] - lower[1],
            upper[1] - centers[..., 1],
            centers[..., 2] - lower[2],
        ]
    )
    clipped = dict(grid)
    clipped["cell_solid_sdf"] = sdf[..., None]
    clipped["solid_planes"] = np.array(
        [[-1, 0, 0, -lower[0]], [1, 0, 0, upper[0]], [0, -1, 0, -lower[1]], [0, 1, 0, upper[1]], [0, 0, -1, -lower[2]]]
    )
    vertices, _ = box_mesh(lower, upper)
    segments = np.array(
        [[a, b] for i, a in enumerate(vertices) for b in vertices[i + 1 :] if np.count_nonzero(a != b) == 1]
    )
    return clipped, segments


def leakage_wall_geometry(grid, dx, crack_width=None):
    from examples.mmpm.GranularWaterLeakage3D.granular_water_leakage_3d import boundaries

    dims = grid["dims"]
    coords = grid["coords"].reshape(*dims, 3)
    spacing = np.array([coords[tuple(int(j == d) for j in range(3))][d] - coords[0, 0, 0, d] for d in range(3)])
    if not np.allclose(spacing, dx):
        raise ValueError("Leakage grid spacing differs from the accepted wall geometry")
    centers = coords[:-1, :-1, :-1] + spacing / 2
    solid_sdf = np.full(tuple(dims - 1), np.inf)
    vertices, faces = [], []
    for boundary in boundaries(dx, crack_width):
        lower, upper = np.array(boundary["StartPoint"]), np.array(boundary["EndPoint"])
        normal = np.array(boundary["Norm"])
        lower += np.minimum(normal, 0) * dx
        upper += np.maximum(normal, 0) * dx
        v, f = box_mesh(lower, upper)
        faces.append(f + 8 * len(vertices))
        vertices.append(v)
        distance = np.abs(centers - (lower + upper) / 2) - (upper - lower) / 2
        box_sdf = np.linalg.norm(np.maximum(distance, 0), axis=-1) + np.minimum(distance.max(axis=-1), 0)
        solid_sdf = np.minimum(solid_sdf, box_sdf)
    clipped = {key: grid[key] for key in ("dims", "coords", "cell_type")}
    clipped["cell_solid_sdf"] = solid_sdf[..., None]
    return clipped, np.concatenate(vertices), np.concatenate(faces)


def prepare(case, folder, preview=False, allow_failed=False, source=None):
    relative, filename, title, domain, piston = CASES[case]
    source = source or ROOT / "examples/mmpm" / relative
    metrics = json.loads((source / "metrics.json").read_text())
    if not metrics.get("passed") and not allow_failed:
        raise ValueError(f"Source validation failed: {source}")
    if (
        case == "leakage"
        and not allow_failed
        and (not metrics.get("duration_complete") or abs(metrics["final_time_s"] - 6) > 1e-7)
    ):
        raise ValueError("Leakage rendering requires the complete six-second validation")
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
    has_lsdem = case in ("sphere_impact", "bunny_impact", "lsdem_wavemaker")
    leakage_geometry = None
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
            render_grid = grid
            if piston:
                mean, frequency, velocity, ramp_time = piston
                omega = 2 * np.pi * frequency
                ramp = 1.0 if time >= ramp_time else 0.5 * (1 - np.cos(np.pi * time / ramp_time))
                piston_position = mean + ramp * velocity / omega * np.sin(omega * time)
            if case == "lsdem_wavemaker":
                render_grid, wall_segments = wavemaker_wall_geometry(grid, domain, piston_position)
            elif case in ("sphere_impact", "bunny_impact"):
                coords = grid["coords"].reshape(*grid["dims"], 3)
                floor = coords[0, 0, 1, 2] - coords[0, 0, 0, 2]
                wall_segments = cylinder_outline(domain, 1, floor)
            if case == "leakage":
                if leakage_geometry is None:
                    leakage_geometry = leakage_wall_geometry(
                        grid,
                        metrics.get("grid_spacing_m", metrics["represented_crack_width_m"]),
                        metrics.get("outlet_width_override_m"),
                    )
                render_grid = leakage_geometry[0]
            data = phase_geometry(
                particles,
                render_grid,
                domain,
                scale,
                close_particle_voids=False,
                radial_limit=0.5 * domain[0] if case in ("sphere_impact", "bunny_impact") else None,
                occupancy_threshold=0.35,
                solid_surface=has_lsdem,
            )
        if has_lsdem:
            with (
                np.load(source / f"particles/LSDEMRigid{step}.npz") as rigid,
                np.load(source / f"particles/LSDEMSurface{step}.npz") as surface,
            ):
                if any(abs(time - float(d["t_current"])) > 1e-8 for d in (rigid, surface)):
                    raise ValueError(f"LSDEM time mismatch: {step}")
                vertices, faces = grain_surface(surface, rigid)
                data.update(rigid=vertices * scale, rigid_faces=faces)
        if piston:
            vertices, faces = box_mesh([0, 0, 0], [piston_position, domain[1], domain[2]])
            data.update(grains=vertices * scale, grain_faces=faces)
            if case == "lsdem_wavemaker":
                data["wall_segments"] = wall_segments * scale
        elif case == "leakage":
            data.update(walls=leakage_geometry[1] * scale, walls_faces=leakage_geometry[2])
        else:
            data["wall_segments"] = wall_segments * scale
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
        "show_tank": case == "wavemaker",
        "solid_material": "metal",
        "water_color": [0.06, 0.38, 0.66],
        "water_visibility": 0.55,
        "water_transmission": 0.55,
        "fluid_surface_method": "particle_position_and_volume_with_case_topology_guard",
        "solid_display_method": (
            "continuum_surface_from_saved_solid_positions_and_bulk_volumes_with_0.05_grid_wall_inset"
            if has_lsdem
            else "one_volume_equivalent_bead_per_saved_solid_particle"
        ),
        "legend": "Blue: water | Brown: saturated soil" + (" | Dark blue: LSDEM" if has_lsdem else ""),
    }
    if has_lsdem:
        config["fluid_surface_method"] = "maximum_bulk_support_of_fluid_and_nearby_wet_soil_without_void_filling"
    if case == "leakage":
        if metrics.get("outlet_width_override_m") is not None:
            config["title"] = f"Granular-water discharge ({1000 * metrics['outlet_width_override_m']:g} mm outlet)"
        config["fluid_surface_method"] = "saved_fluid_bulk_volume_surface_including_outlet_stream_and_receiver"
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
