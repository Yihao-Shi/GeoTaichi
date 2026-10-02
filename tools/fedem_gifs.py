#!/usr/bin/env python3
"""FEDEM gallery: paper Hertz stress PNG, mixed funnel and 100% compaction GIFs.

python tools/fedem_gifs.py --case all --work-dir /tmp/fedem_gallery
ParaView extracts native VTU surfaces; Blender renders their unamplified geometry.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from blender_cfdem_gif import ROOT, export
from incompressible_gifs import BLENDER, PVPYTHON, box_mesh

CASES = {
    "hertz": (
        "HertzContact/OutputData/hertz_mesh_convergence_v6_projected_pressure/mesh_ultrafine",
        "hertz_contact.png",
        "Hertz contact: von Mises stress",
    ),
    "funnel": (
        "MixedFunnel/OutputData/mixed_funnel_800_gate_v12_fresh_dt1us/run",
        "mixed_funnel.gif",
        "Mixed funnel: 400 soft + 400 rigid grains",
    ),
    "compaction": (
        "IsotropicCompaction/OutputData/isotropic_compaction_spherical_disordered_barrier_v1/soft_100",
        "isotropic_compaction_100.gif",
        "Isotropic compaction: 100% soft grains",
    ),
}


def read_surface(path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    if reader.GetErrorCode() or not reader.GetOutput().GetNumberOfCells():
        raise ValueError(f"Empty/unreadable native mesh: {path}")
    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputConnection(reader.GetOutputPort())
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(surface.GetOutputPort())
    triangles.Update()
    data = triangles.GetOutput()
    vertices = vtk_to_numpy(data.GetPoints().GetData()).copy()
    connectivity = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)
    if not np.isfinite(vertices).all() or not np.all(connectivity[:, 0] == 3):
        raise ValueError(f"Invalid native surface: {path}")
    return vertices, connectivity[:, 1:].copy()


def wall_segments(vertices, faces):
    edges = {}
    for face in faces:
        a, b, c = vertices[face]
        normal = np.cross(b - a, c - a)
        normal = normal / max(np.linalg.norm(normal), 1e-30)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((tuple(np.round(vertices[face[i]], 12)), tuple(np.round(vertices[face[j]], 12)))))
            edges.setdefault(edge, []).append(normal)
    # Do not show triangulation diagonals on planar hopper/press walls.
    return np.array(
        [
            edge
            for edge, normals in edges.items()
            if len(normals) == 1 or any(abs(np.dot(normals[0], n)) < 0.999 for n in normals[1:])
        ]
    )


def compaction_times(config, metrics, count, first_time, compression_start=None):
    steps = config["loading_protocol"]["compression_step_count"]
    command = config["execution"]["command"]
    snapshots = int(command[command.index("--snapshot-count") + 1])
    saved_steps = np.linspace(1, steps, snapshots, dtype=np.int64)
    completed = metrics["compression_step_count"]
    local_steps = [0, *[int(step) for step in saved_steps if step <= completed]]
    if local_steps[-1] != completed:
        local_steps.append(completed)
    start = first_time if compression_start is None else compression_start
    times = start + np.array(local_steps) * config["parameters"]["effective_dt"]
    if first_time < start - 1e-8:
        times = np.r_[first_time, times]
    if len(times) != count:
        raise ValueError("Saved compaction frame count disagrees with the recorded snapshot schedule")
    if abs(times[-1] - metrics["final"]["time"]) > 1e-7:
        raise ValueError("Compaction snapshot schedule disagrees with recorded final time")
    return times


def prepare(case, folder, preview=False):
    relative, filename, title = CASES[case]
    source = ROOT / "examples/fedem" / relative
    config = json.loads((source / "config.json").read_text())
    metrics = json.loads((source / "metrics.json").read_text())
    files = sorted((source / "native/vtks").glob("FEM[0-9]*.vtu"))
    if not files:
        raise FileNotFoundError(f"No saved FEM meshes in {source}")
    folder.mkdir(parents=True, exist_ok=True)
    if case == "hertz":
        files = files[-1:]
        times = [metrics["actual_end_time"]]
        origin, domain = np.array([0.22, 0.22, 0.388]), np.array([0.16, 0.16, 0.13])
    elif case == "funnel":
        times = np.arange(len(files)) * config["parameters"]["output_interval"]
        origin, domain = np.zeros(3), np.array([0.6, 0.4, 0.86])
    else:
        if config["parameters"]["requested_soft_percent"] != 100:
            raise ValueError("The gallery requires the 100% soft-particle compaction run")
        with np.load(source / "native/walls/DEMWall000000.npz") as p:
            first_time = float(p["t_current"])
        times = compaction_times(config, metrics, len(files), first_time, float(metrics["final_consolidation"]["time"]))
        with (source / "history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        history_times = np.array([float(row["time"]) for row in history])
        initial_walls = json.loads((source / "state.json").read_text())["initial_wall_positions"]
        lower = np.array([initial_walls[k] for k in ("left", "front", "bottom")])
        upper = np.array([initial_walls[k] for k in ("right", "back", "top")])
        wall_center = (lower + upper) / 2
        initial_dimensions = upper - lower
        origin, domain = np.full(3, 0.09), np.full(3, 0.29)
    scale = 3.6 / max(domain)
    frames = []
    for index, (path, time) in enumerate(zip(files[:1] if preview else files, times)):
        step = path.stem[-6:]
        data = {}
        for key, prefix in (("soft", "FEM"), ("rigid", "GraphicLSDEMSurface")):
            if key == "rigid" and config["parameters"].get("rigid_particle_count") == 0:
                continue
            vertices, faces = read_surface(path.parent / f"{prefix}{step}.vtu")
            if case == "hertz" and key == "rigid":
                # Crop only the oversized rigid support; never magnify FEM deformation.
                vertices = np.clip(vertices, [0.22, 0.22, 0.388], [0.38, 0.38, 0.4])
            data[key], data[f"{key}_faces"] = (vertices - origin) * scale, faces
        if case != "hertz":
            if case == "funnel":
                wall, faces = read_surface(path.parent / f"TriangleWall{step}.vtu")
                with np.load(source / "native/walls/DEMWall000000.npz") as w:
                    active = w["active"].astype(bool)
                if time >= config["parameters"]["gate_open_time"]:
                    active[config["parameters"]["realized_wall_geometry"]["outlet_gate"]["facet_ids"]] = False
                if len(faces) != len(active):
                    raise ValueError("Hopper wall facets disagree with the saved active-wall mask")
                faces = faces[active]
            else:
                # Native wall vertices are static references; history records the moving six-wall box.
                if time < history_times[0]:
                    dimensions = initial_dimensions
                else:
                    dimensions = np.array(
                        [
                            np.interp(time, history_times, [float(row[k]) for row in history])
                            for k in ("width", "depth", "height")
                        ]
                    )
                wall, faces = box_mesh(wall_center - dimensions / 2, wall_center + dimensions / 2)
            data.update(
                walls=(wall - origin) * scale,
                walls_faces=faces,
                wall_segments=(wall_segments(wall, faces) - origin) * scale,
            )
        mesh = f"mesh_{index:06d}.npz"
        np.savez(folder / mesh, **data)
        frames.append({"mesh": mesh, "step": step, "time": float(time)})
        print(f"PREPARE {case}: {index + 1}/{1 if preview else len(files)}", flush=True)
    manifest = {
        "case": case,
        "source": str(source),
        "output": str(ROOT / "images" / filename),
        "title": title,
        "domain": (domain * scale).tolist(),
        "frames": frames,
        "radius": None,
        "width": 960,
        "height": 880 if case == "funnel" else 720,
        "rigid_flat_shading": case == "hertz",
        "fps": 2 if case == "compaction" else 5,
        "show_tank": False,
        "camera_direction": [0.8, -1.7, 0.8],
        "legend": (
            "Orange: deformable FEM particles" if case == "compaction" else "Orange: deformable FEM   Blue: rigid LSDEM"
        ),
        "geometry_method": "native_deformed_FEM_and_LSDEM_surfaces",
        "deformation_scale": 1,
        "origin_m": origin.tolist(),
        "display_scale": scale,
        "simulation_passed": metrics.get("passed"),
        "simulation_stop_reason": metrics.get("stop_reason"),
        "wall_geometry_method": "recorded_box_dimensions" if case == "compaction" else "native_surface",
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--prepare-case", choices=list(CASES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    work = (args.work_dir or Path(tempfile.mkdtemp(prefix="fedem_blender_"))).resolve()
    if args.prepare_case:
        prepare(args.prepare_case, work / args.prepare_case, args.preview)
        return
    print(f"Render assets / PNGs / backups: {work}", flush=True)
    for case in CASES if args.case == "all" else [args.case]:
        if case == "hertz":
            paper_png = ROOT / "research/llm_assist/fem_lsdem/figures/docs/hertz.png"
            folder = work / case
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(paper_png, folder / "frame_000000.png")
            if not args.preview:
                output = ROOT / "images" / CASES[case][1]
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists():
                    shutil.copy2(output, folder / f"previous_{output.name}")
                shutil.copy2(paper_png, output)
                print(f"Copied paper von Mises stress PNG: {output}", flush=True)
            continue
        command = [PVPYTHON, str(Path(__file__).resolve()), "--prepare-case", case, "--work-dir", str(work)]
        if args.preview:
            command.append("--preview")
        subprocess.run(command, check=True)
        manifest = work / case / "manifest.json"
        subprocess.run(
            [
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
            ],
            check=True,
        )
        config = json.loads(manifest.read_text())
        if not args.preview:
            export(config, manifest.parent)


if __name__ == "__main__":
    main()
