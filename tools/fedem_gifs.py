#!/usr/bin/env python3
"""FEDEM gallery: paper Hertz stress PNG, mixed funnel and 100% compaction GIFs.

python tools/fedem_gifs.py --case all --work-dir /tmp/fedem_gallery
Add --renderer paraview for native FEM von Mises stress (0.7 of the global maximum).
Both renderers preserve the native deformation and recorded wall motion.

python tools/fedem_gifs.py --case funnel --renderer paraview --fem-color-max 10000
python tools/fedem_gifs.py --case compaction --renderer paraview
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


def stress_color_range(maximum, color_max=None):
    upper = 0.7 * float(maximum) if color_max is None else float(color_max)
    if not np.isfinite([maximum, upper]).all() or min(maximum, upper) <= 0:
        raise ValueError("von Mises stress must have a finite positive maximum")
    return [0.0, upper]


def render_paraview(path, color_max=None):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from paraview import simple as pv

    config = json.loads(path.read_text())
    folder = path.parent
    source = Path(config["source"]) / "native/vtks"
    pv._DisableFirstRenderCameraReset()
    view = pv.CreateView("RenderView")
    view.ViewSize = [config["width"], config["height"]]
    view.UseColorPaletteForBackground = 0
    view.Background = [1, 1, 1]
    view.OrientationAxesVisibility = 0
    view.UseFXAA = 1
    readers = {}
    displays = {}
    for key, prefix in (("soft", "FEM"), ("rigid", "GraphicLSDEMSurface")):
        files = [str(p) for p in sorted(source.glob(f"{prefix}[0-9]*.vtu"))]
        if key == "rigid" and not files:
            continue
        readers[key] = pv.XMLUnstructuredGridReader(FileName=files)
        displays[key] = pv.Show(readers[key], view)
        displays[key].Representation = "Surface"
        displays[key].Scale = [config["display_scale"]] * 3
        displays[key].Position = (-np.array(config["origin_m"]) * config["display_scale"]).tolist()
    maximum = 0.0
    for index in range(len(readers["soft"].FileName)):
        readers["soft"].UpdatePipeline(index)
        array = readers["soft"].PointData.GetArray("von_mises")
        if array is None:
            raise ValueError("FEM output is missing the von_mises point array")
        bounds = array.GetRange()
        if not np.isfinite(bounds).all() or bounds[0] < 0:
            raise ValueError(f"Invalid von Mises stress range: {bounds}")
        maximum = max(maximum, bounds[1])
    color_range = stress_color_range(maximum, color_max)
    config.update(renderer="ParaView", fem_scalar="von_mises", fem_scalar_units="Pa",
                  fem_stress_maximum=maximum, fem_color_range=color_range,
                  fem_color_max_factor=color_range[1] / maximum)
    path.write_text(json.dumps(config, indent=2) + "\n")
    pv.ColorBy(displays["soft"], ("POINTS", "von_mises"))
    lut = pv.GetColorTransferFunction("von_mises")
    lut.ApplyPreset("Viridis (matplotlib)", True)
    lut.RescaleTransferFunction(*color_range)
    lut.AutomaticRescaleRangeMode = "Never"
    displays["soft"].SetScalarBarVisibility(view, True)
    bar = pv.GetScalarBar(lut, view)
    bar.Title = "von Mises stress (Pa)"
    bar.ComponentTitle = ""
    bar.Orientation = "Horizontal"
    bar.WindowLocation = "Any Location"
    bar.Position = [0.22, 0.06]
    bar.ScalarBarLength = 0.56
    bar.TitleFontSize = 17
    bar.LabelFontSize = 15
    bar.LabelFormat = "%.3g"
    bar.RangeLabelFormat = "%.3g"
    bar.AutomaticLabelFormat = 0
    bar.TitleColor = bar.LabelColor = [0, 0, 0]
    bar.UseCustomLabels = 1
    bar.CustomLabels = np.linspace(*color_range, 5).tolist()
    if "rigid" in displays:
        pv.ColorBy(displays["rigid"], None)
        displays["rigid"].DiffuseColor = [0.18, 0.42, 0.72]
    walls = pv.TrivialProducer()
    walls.GetClientSideObject().SetOutput(vtk.vtkPolyData())
    wall_display = pv.Show(walls, view)
    wall_display.DiffuseColor = [0.60, 0.66, 0.72]
    wall_display.Opacity = 0.10
    edges = pv.TrivialProducer()
    edges.GetClientSideObject().SetOutput(vtk.vtkPolyData())
    edge_display = pv.Show(edges, view)
    edge_display.DiffuseColor = [0.30, 0.34, 0.39]
    edge_display.LineWidth = 1.5
    caption = pv.Text()
    caption.Text = config["title"]
    label = pv.Show(caption, view)
    label.WindowLocation = "Upper Center"
    label.FontSize = 19
    label.Color = [0.10, 0.12, 0.15]
    clock = pv.Text()
    clock_display = pv.Show(clock, view)
    clock_display.WindowLocation = "Lower Left Corner"
    clock_display.FontSize = 16
    clock_display.Color = [0.10, 0.12, 0.15]
    center = np.array(config["domain"]) / 2
    direction = np.array(config["camera_direction"])
    view.CameraFocalPoint = center.tolist()
    view.CameraPosition = (center + direction * max(config["domain"]) * 2).tolist()
    view.CameraViewUp = [0, 0, 1]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = max(config["domain"]) * (0.80 if config["case"] == "compaction" else 0.68)
    for index, frame in enumerate(config["frames"]):
        for reader in readers.values():
            reader.UpdatePipeline(index)
        with np.load(folder / frame["mesh"]) as data:
            for producer, vertices, cells, triangles in (
                (walls, data["walls"], data["walls_faces"], True),
                (edges, data["wall_segments"].reshape(-1, 3),
                 np.arange(data["wall_segments"].size // 3).reshape(-1, 2), False),
            ):
                points = vtk.vtkPoints()
                points.SetData(numpy_to_vtk(vertices, deep=True))
                connectivity = np.c_[np.full(len(cells), cells.shape[1]), cells].astype(np.int64)
                topology = vtk.vtkCellArray()
                topology.SetCells(len(cells), numpy_to_vtkIdTypeArray(connectivity.ravel(), deep=True))
                mesh = vtk.vtkPolyData()
                mesh.SetPoints(points)
                (mesh.SetPolys if triangles else mesh.SetLines)(topology)
                producer.GetClientSideObject().SetOutput(mesh)
                producer.MarkModified(producer)
                producer.UpdatePipeline()
        clock.Text = f"t = {frame['time']:.3f} s"
        view.ViewTime = index
        pv.Render(view)
        pv.SaveScreenshot(str(folder / f"frame_{index:06d}.png"), view,
                          ImageResolution=[config["width"], config["height"]])
        print(f"PARAVIEW {config['case']}: {index + 1}/{len(config['frames'])}", flush=True)
    print(f"von Mises maximum = {maximum:.6g} Pa; colorbar = {color_range}", flush=True)


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
    parser.add_argument("--renderer", choices=("blender", "paraview"), default="blender")
    parser.add_argument("--fem-color-max", type=float, help="ParaView stress upper limit in Pa; default: 0.7 of maximum")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--prepare-case", choices=list(CASES), help=argparse.SUPPRESS)
    parser.add_argument("--render-case", choices=("funnel", "compaction"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    work = (args.work_dir or Path(tempfile.mkdtemp(prefix="fedem_blender_"))).resolve()
    if args.render_case:
        path, _ = prepare(args.render_case, work / args.render_case, args.preview)
        render_paraview(path, args.fem_color_max)
        return
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
        if args.renderer == "paraview":
            command[2] = "--render-case"
            if args.fem_color_max is not None:
                command.extend(["--fem-color-max", str(args.fem_color_max)])
        if args.preview:
            command.append("--preview")
        subprocess.run(command, check=True)
        manifest = work / case / "manifest.json"
        if args.renderer == "blender":
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
