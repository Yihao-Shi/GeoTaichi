#!/usr/bin/env python3
"""Render native IGA bars with ParaView and self-contacting FEM cloth with Blender.

See tools/README.md for source runs and commands. No deformation magnification.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.blender_cfdem_gif import ROOT, export
from tools.fedem_gifs import read_surface
from tools.incompressible_gifs import BLENDER, PVPYTHON


def elastic_bar_summary(source):
    summary = json.loads((source / "summary.json").read_text())
    if summary["solver"] != "implicit" or not summary.get("converged"):
        raise ValueError("The elastic-bar gallery requires a completed implicit IGA run")
    dt, interval, steps = summary["dt"], summary["output_interval"], summary["steps"]
    if (
        not np.isfinite(dt)
        or dt <= 0
        or interval <= 0
        or steps <= 0
        or steps % interval
        or summary["frames"] != steps // interval + 1
        or not np.isclose(summary["physical_seconds"], steps * dt)
    ):
        raise ValueError("Implicit bar frame count/time disagrees with its run summary")
    return summary


def prepare_cloth(source, folder):
    summary = json.loads((source / "validation_summary.json").read_text())
    if (
        summary["case"] != "newton_cloth_twist"
        or not summary["converged"]
        or not summary["finite_positions"]
        or summary["maximum_active_contacts"] <= 0
    ):
        raise ValueError("Expected completed, finite twisting cloth with active self-contact")
    if (
        not np.isfinite(summary["dt"])
        or summary["dt"] <= 0
        or not np.isclose(summary["final_time"], summary["steps"] * summary["dt"])
    ):
        raise ValueError("Cloth completion time disagrees with its timestep")
    files = sorted((source / "vtks").glob("FEM[0-9]*.vtu"))
    steps = [int(p.stem[-6:]) for p in files]
    if len(steps) < 2 or steps[0] != 0 or steps[-1] != summary["steps"]:
        raise ValueError("Cloth frames must cover the completed simulation")
    intervals = np.diff(steps)
    if not np.all(intervals == intervals[0]):
        raise ValueError("Cloth recorder frames must be uniformly spaced")
    geometry = [read_surface(p) for p in files]
    for vertices, faces in geometry:
        if len(vertices) != summary["nodes"] or len(faces) != summary["triangles"]:
            raise ValueError("Saved cloth topology differs from its validation summary")
        if not np.array_equal(faces, geometry[0][1]):
            raise ValueError("Saved cloth connectivity changed")
    points = np.concatenate([vertices for vertices, _ in geometry])
    origin = points.min(axis=0) - [0, 0, 0.05]
    extent = points.max(axis=0) - origin
    scale = 3.6 / extent.max()
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, (step, (vertices, faces)) in enumerate(zip(steps, geometry)):
        mesh = f"mesh_{index:06d}.npz"
        np.savez(folder / mesh, soft=(vertices - origin) * scale, soft_faces=faces)
        frames.append(dict(time=step * summary["dt"], step=step, mesh=mesh))
    return dict(
        case="cloth_twisting",
        source=str(source),
        output=str(ROOT / "images/fem_cloth_twisting.gif"),
        title="FEM cloth twisting: self-contact",
        frames=frames,
        fps=5,
        width=960,
        height=640,
        domain=(extent * scale).tolist(),
        origin_m=origin.tolist(),
        display_scale=scale,
        camera_direction=[0.6, -1.1, 1.5],
        show_tank=False,
        cloth_display="single_color_native_mesh",
        timing_source="VTU_step_times_dt",
    )


def sample_twist(source, folder):
    from src.nurbs.BasicVolume import Cube
    from src.nurbs.NurbsBasis import NurbsBasisInterpolations3d
    from third_party.pyevtk.hl import gridToVTK

    summary = json.loads((source / "summary.json").read_text())
    reference = np.load(source / "reference.npz")["control_points"]
    counts = [len(np.unique(reference[:, axis])) for axis in range(3)]
    if np.prod(counts) != summary["control_points"] or len(reference) != np.prod(counts):
        raise ValueError("Expected the twisting example's tensor-product control lattice")
    cube = Cube()
    cube.generate_knot_u(2, counts[0])
    cube.generate_knot_v(2, counts[1])
    cube.generate_knot_w(2, counts[2])
    resolution = (9, 9, 81)
    coordinates = list(np.ndindex(resolution))
    identity = np.eye(len(reference))
    weights = np.ones(len(reference))
    basis = np.array(
        [
            NurbsBasisInterpolations3d(
                *(np.array(index) / (np.array(resolution) - 1)),
                2,
                2,
                2,
                cube.knot_vector_u,
                cube.knot_vector_v,
                cube.knot_vector_w,
                identity,
                weights,
            )
            for index in coordinates
        ]
    )
    # Display sampling must preserve constants and the reference cube's corners.
    np.testing.assert_allclose(basis.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(np.einsum("ij,jk->ik", basis, reference)[[0, -1]], reference[[0, -1]], atol=1e-12)
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for index in range(summary["steps"] // summary["output_interval"] + 1):
        with np.load(source / f"state_{index:06d}.npz") as state:
            if int(state["step"]) != index * summary["output_interval"] or not np.isclose(
                state["time"], index * summary["output_interval"] * summary["dt"]
            ):
                raise ValueError("Saved twist state time/step disagrees with its run summary")
            values = np.einsum("ij,jk->ik", basis, np.column_stack((state["control_points"], state["stress"])))
        if not np.isfinite(values).all() or np.any(values[:, 3] < 0):
            raise ValueError("Invalid sampled NURBS geometry/stress")
        values = values.reshape((*resolution, 4))
        path = folder / f"NurbsVolumetwisting_column{index:06d}"
        gridToVTK(
            str(path),
            *(np.ascontiguousarray(values[..., axis]) for axis in range(3)),
            pointData={"stress": np.ascontiguousarray(values[..., 3])},
        )
        files.append(path.with_suffix(".vts"))
    return files


def render_bar(source, folder, case, frame_interval, preview):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from paraview import simple as pv

    files = sorted((source / "vtks").glob("NurbsVolume*[0-9].vtu"))
    if case == "iga_twisting":
        files = sorted((folder / "sampled").glob("NurbsVolume*[0-9].vts"))
    elif len(files) != elastic_bar_summary(source)["frames"]:
        raise ValueError("Saved implicit bar frames are incomplete")
    if len(files) < 2:
        raise ValueError("Expected at least two native IGA frames")
    maximum, points = 0.0, []
    for index, path in enumerate(files):
        if int(path.stem[-6:]) != index:
            raise ValueError("IGA output indices must be consecutive, starting at zero")
        reader = vtk.vtkXMLStructuredGridReader() if case == "iga_twisting" else vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(path))
        reader.Update()
        grid = reader.GetOutput()
        stress = grid.GetPointData().GetArray("stress")
        if reader.GetErrorCode() or not grid.GetNumberOfCells() or stress is None:
            raise ValueError(f"Missing native IGA geometry/stress: {path}")
        values = vtk_to_numpy(stress)
        vertices = vtk_to_numpy(grid.GetPoints().GetData()).copy()
        if not np.isfinite(vertices).all() or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid IGA geometry/stress: {path}")
        maximum = max(maximum, float(values.max()))
        if case == "iga_twisting":
            vertices = np.column_stack((vertices[:, 2], vertices[:, 1], -vertices[:, 0]))
        points.append(vertices)
    if maximum <= 0:
        raise ValueError("IGA frames must contain nonzero von Mises stress")
    bounds = np.concatenate(points)
    center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    extent = np.ptp(bounds, axis=0)
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(
        case=case,
        source=str(source),
        renderer="ParaView",
        deformation_scale=1,
        solver=json.loads((source / "summary.json").read_text())["solver"],
        display_rotation_degrees=[0, 90, 0] if case == "iga_twisting" else [0, 0, 0],
        scalar="stress",
        scalar_meaning="von Mises stress",
        scalar_units="Pa",
        color_range=[0, maximum],
        output=str(ROOT / "images" / f"{case}.gif"),
        fps=15,
        width=960,
        height=640,
        frames=[dict(time=i * frame_interval, source=str(p)) for i, p in enumerate(files)],
    )
    (folder / "manifest.json").write_text(json.dumps(config, indent=2) + "\n")
    pv._DisableFirstRenderCameraReset()
    view = pv.CreateView("RenderView")
    view.ViewSize = [config["width"], config["height"]]
    view.UseColorPaletteForBackground = 0
    view.Background = [1, 1, 1]
    view.OrientationAxesVisibility = 0
    view.UseFXAA = 1
    reader_type = pv.XMLStructuredGridReader if case == "iga_twisting" else pv.XMLUnstructuredGridReader
    reader = reader_type(FileName=[str(p) for p in files])
    surface = pv.ExtractSurface(Input=reader)
    normals = pv.GenerateSurfaceNormals(Input=surface)
    normals.FeatureAngle = 45
    display = pv.Show(normals, view)
    display.Representation = "Surface"
    display.Orientation = config["display_rotation_degrees"]
    pv.ColorBy(display, ("POINTS", "stress"))
    lut = pv.GetColorTransferFunction("stress")
    lut.ApplyPreset("Viridis (matplotlib)", True)
    lut.RescaleTransferFunction(0, maximum)
    lut.AutomaticRescaleRangeMode = "Never"
    display.SetScalarBarVisibility(view, True)
    bar = pv.GetScalarBar(lut, view)
    bar.Title, bar.ComponentTitle = "von Mises stress (Pa)", ""
    bar.Orientation, bar.WindowLocation = "Horizontal", "Any Location"
    bar.Position, bar.ScalarBarLength = [0.22, 0.08], 0.56
    bar.TitleFontSize, bar.LabelFontSize = 18, 15
    bar.TitleColor = bar.LabelColor = [0, 0, 0]
    bar.AutomaticLabelFormat = 0
    bar.LabelFormat = bar.RangeLabelFormat = "%.2g"
    bar.UseCustomLabels = 1
    bar.CustomLabels = np.linspace(0, maximum, 5).tolist()
    caption = pv.Text()
    caption.Text = "IGA bar: 450-degree twist" if case == "iga_twisting" else "IGA implicit elastic bar"
    label = pv.Show(caption, view)
    label.WindowLocation, label.FontSize, label.Color = "Upper Center", 20, [0.1, 0.1, 0.1]
    clock = pv.Text()
    label = pv.Show(clock, view)
    label.WindowLocation, label.FontSize, label.Color = "Lower Left Corner", 16, [0.1, 0.1, 0.1]
    view.CameraFocalPoint = center.tolist()
    direction = np.array([0, -2, 1.1] if case == "iga_twisting" else [0.35, -2, 1.1])
    view.CameraPosition = (center + direction * max(extent) * 2).tolist()
    view.CameraViewUp = [0, 0, 1]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = max(0.5 * extent[0], extent[1], extent[2]) * 0.9
    for index in [len(files) - 1] if preview else range(len(files)):
        reader.UpdatePipeline(index)
        view.ViewTime = index
        clock.Text = f"t = {config['frames'][index]['time']:.3f} s"
        pv.SaveScreenshot(str(folder / f"frame_{index:06d}.png"), view, ImageResolution=view.ViewSize)
        print(f"RENDER {case}: {index + 1}/{len(files)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("iga_elastic_bar", "iga_twisting", "cloth_twisting"))
    parser.add_argument("source", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--render-bar", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--frame-interval", type=float, help="Optional saved-frame spacing check (s)")
    args = parser.parse_args()
    source = args.source.resolve()
    folder = (args.work_dir or Path(tempfile.mkdtemp(prefix=f"{args.case}_"))).resolve()
    if args.case == "cloth_twisting":
        config = prepare_cloth(source, folder)
        if args.preview:
            config["frames"] = config["frames"][-1:]
        manifest = folder / ("preview.json" if args.preview else "manifest.json")
        manifest.write_text(json.dumps(config, indent=2) + "\n")
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
                "64",
            ],
            check=True,
        )
        if not args.preview:
            export(config, folder)
    else:
        if args.case == "iga_twisting":
            summary = json.loads((source / "summary.json").read_text())
            if not json.loads((source / "evaluation.json").read_text())["passed"]:
                parser.error("IGA twist must pass its saved-state evaluation")
        else:
            summary = elastic_bar_summary(source)
        saved_interval = summary["dt"] * summary["output_interval"]
        if args.frame_interval is not None and not np.isclose(args.frame_interval, saved_interval):
            parser.error("--frame-interval disagrees with the saved run")
        args.frame_interval = saved_interval
        if not np.isfinite(args.frame_interval) or args.frame_interval <= 0:
            parser.error("--frame-interval must be finite and positive")
        if args.render_bar:
            render_bar(source, folder, args.case, args.frame_interval, args.preview)
        else:
            if args.case == "iga_twisting":
                sample_twist(source, folder / "sampled")
            command = [
                PVPYTHON,
                "--force-offscreen-rendering",
                str(Path(__file__).resolve()),
                args.case,
                str(source),
                "--work-dir",
                str(folder),
                "--render-bar",
                "--frame-interval",
                str(args.frame_interval),
            ]
            if args.preview:
                command.append("--preview")
            subprocess.run(command, check=True)
            if not args.preview:
                export(json.loads((folder / "manifest.json").read_text()), folder)
