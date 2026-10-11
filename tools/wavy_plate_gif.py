#!/usr/bin/env pvpython
"""Render saved IGA--MPM gallery frames with IGA stress and MPM displacement.

pvpython tools/wavy_plate_gif.py examples/igampm/wavy_plate_collapse/OutputData/ipc_formal_20261006
Color native IGA von Mises stress and MPM displacement from frame zero with fixed ranges.
For flexible_barrier, also pass --output images/flexible_barrier.gif.
For cpt_dp, also pass --output images/cpt_dp.gif; both bodies use displacement.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.blender_cfdem_gif import export


def particle_displacement(positions, initial, body_ids, initial_body_ids):
    # shortcut: GalleryRecorder keeps particle order, require particle IDs for exports that reorder particles.
    if positions.shape != initial.shape or not np.array_equal(body_ids, initial_body_ids):
        raise ValueError("MPM particle count/order changed relative to frame zero")
    if not np.isfinite(positions).all() or not np.isfinite(initial).all():
        raise ValueError("MPM positions must be finite")
    return positions - initial


def render(source, folder, output, fps, preview):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    from paraview import simple as pv

    folder.mkdir(parents=True, exist_ok=True)
    parameters = json.loads((source / "parameters.json").read_text())
    summary = source / "cpt_summary.json"
    axisymmetric = summary.exists() and json.loads(summary.read_text()).get("axisymmetric", False)
    dimension = parameters.get("dimension", 2 if axisymmetric else 3)
    # CPT has recovered frames without stress; prescribed probe displacement is complete.
    iga_scalar = "displacement" if axisymmetric else "stress"
    iga_files = sorted((source / "vtks").glob("NurbsVolume*.vtu"))
    iga = {p.stem[-6:]: p for p in iga_files if not p.name.startswith("._")}
    mpm = {p.stem[-6:]: p for p in (source / "vtks").glob("GraphicMPMParticle[0-9]*.vtu")}
    if not iga or len(iga) != len(iga_files) or iga.keys() != mpm.keys() or "000000" not in mpm:
        raise ValueError("Expected aligned native IGA and MPM frames")
    frames = []
    iga_max = displacement_max = 0.0
    bounds = []
    for index, step in enumerate(sorted(iga)):
        time = int(step) * parameters["save_interval"]
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(mpm[step]))
        reader.Update()
        grid = reader.GetOutput()
        if reader.GetErrorCode() or not grid.GetNumberOfPoints() or grid.GetPointData().GetArray("bodyID") is None:
            raise ValueError(f"Invalid MPM frame {step}")
        positions = vtk_to_numpy(grid.GetPoints().GetData())
        body_ids = vtk_to_numpy(grid.GetPointData().GetArray("bodyID"))
        if index == 0:
            initial, initial_body_ids = positions.copy(), body_ids.copy()
        values = particle_displacement(positions, initial, body_ids, initial_body_ids)
        displacement_max = max(displacement_max, float(np.linalg.norm(values, axis=1).max()))
        array = numpy_to_vtk(values, deep=True)
        array.SetName("displacement")
        grid.GetPointData().AddArray(array)
        derived = folder / f"MPMDisplacement{step}.vtu"
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(derived))
        writer.SetInputData(grid)
        if writer.Write() != 1:
            raise OSError(f"Could not save displacement frame {step}")
        bounds.append(grid.GetBounds())
        reader.SetFileName(str(iga[step]))
        reader.Update()
        grid = reader.GetOutput()
        array = grid.GetPointData().GetArray(iga_scalar)
        if reader.GetErrorCode() or array is None:
            raise ValueError(f"Missing IGA {iga_scalar} in frame {step}")
        values = vtk_to_numpy(array)
        if iga_scalar == "displacement":
            values = np.linalg.norm(values, axis=1)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid IGA {iga_scalar}")
        iga_max = max(iga_max, float(values.max()))
        bounds.append(grid.GetBounds())
        frames.append(dict(step=step, time=time, iga=str(iga[step]), mpm=str(derived)))
        print(f"PREPARE {index + 1}/{len(iga)}", flush=True)
    if iga_max <= 0 or displacement_max <= 0:
        raise ValueError(f"Saved frames must contain nonzero IGA {iga_scalar} and MPM displacement")
    config = dict(
        source=str(source),
        output=str(output),
        renderer="ParaView",
        dimension=dimension,
        axisymmetric=axisymmetric,
        frames=frames,
        fps=fps,
        camera_direction=[0, 0, 1] if dimension == 2 else [1, -1, 0.75],
        deformation_scale=1,
        iga_opacity=1 if axisymmetric else 0.75 if dimension == 2 else 0.45,
        iga_scalar=iga_scalar,
        iga_color_range=[0, iga_max if axisymmetric else 0.7 * iga_max],
        iga_scalar_units="m" if axisymmetric else "Pa",
        mpm_displacement_range=[0, displacement_max],
        mpm_displacement_units="m",
        mpm_scalar="displacement",
        mpm_displacement_reference="frame_zero_particle_positions",
        mpm_component="Magnitude",
    )
    pv._DisableFirstRenderCameraReset()
    view = pv.CreateView("RenderView")
    view.ViewSize = [720, 1080] if axisymmetric else [960, 640 if dimension == 2 else 800]
    view.UseColorPaletteForBackground = 0
    view.Background = [1, 1, 1]
    view.OrientationAxesVisibility = 0
    view.UseFXAA = 1
    readers = []
    for key, scalar, title, preset, position, value_range in (
        (
            "iga",
            iga_scalar,
            "IGA: displacement (m)" if axisymmetric else "IGA: von Mises (Pa)",
            "Viridis (matplotlib)",
            [0.08, 0.06],
            config["iga_color_range"],
        ),
        (
            "mpm",
            "displacement",
            "MPM: displacement (m)",
            "Plasma (matplotlib)",
            [0.56, 0.06],
            config["mpm_displacement_range"],
        ),
    ):
        reader = pv.XMLUnstructuredGridReader(FileName=[frame[key] for frame in frames])
        readers.append(reader)
        display = pv.Show(reader, view)
        display.Representation = ("Surface With Edges" if axisymmetric else "Surface") if key == "iga" else "Points"
        if key == "iga" and axisymmetric:
            display.EdgeColor = [0.15, 0.15, 0.15]
        display.Opacity = config["iga_opacity"] if key == "iga" else 1
        if key == "mpm":
            if axisymmetric:
                display.Position = [0, 0, -0.001]
            display.PointSize = 4
            display.RenderPointsAsSpheres = 0 if axisymmetric else 1
            display.Ambient = 1
            display.Diffuse = 0
        pv.ColorBy(
            display, ("POINTS", scalar, "Magnitude") if scalar == "displacement" else ("POINTS", scalar), separate=True
        )
        lut = pv.GetColorTransferFunction(scalar, display, separate=True)
        if scalar == "displacement":
            lut.VectorMode = "Magnitude"
        if not lut.ApplyPreset(preset, True):
            raise ValueError(f"ParaView color preset is unavailable: {preset}")
        lut.RescaleTransferFunction(*value_range)
        lut.AutomaticRescaleRangeMode = "Never"
        display.SetScalarBarVisibility(view, True)
        bar = pv.GetScalarBar(lut, view)
        bar.Title, bar.ComponentTitle = title, ""
        bar.Orientation, bar.WindowLocation = "Horizontal", "Any Location"
        bar.Position, bar.ScalarBarLength = position, 0.36
        bar.TitleFontSize, bar.LabelFontSize = 17, 14
        bar.TitleColor = bar.LabelColor = [0, 0, 0]
        bar.AutomaticLabelFormat = 0
        bar.LabelFormat = bar.RangeLabelFormat = "%.2g"
        bar.UseCustomLabels = 1
        bar.CustomLabels = np.linspace(*value_range, 3).tolist()
    bounds = np.array(bounds)
    lower, upper = bounds[:, ::2].min(axis=0), bounds[:, 1::2].max(axis=0)
    center = (lower + upper) / 2
    if dimension == 3:
        center[2] -= 0.25
    else:
        center[1] -= 0.07 * (upper[1] - lower[1])
    direction = np.array(config["camera_direction"], dtype=float)
    view.CameraFocalPoint = center.tolist()
    view.CameraPosition = (center + direction / np.linalg.norm(direction) * 4 * max(upper - lower)).tolist()
    view.CameraViewUp = [0, 1, 0] if dimension == 2 else [0, 0, 1]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = (
        max(0.7 * (upper[1] - lower[1]), 0.58 * (upper[0] - lower[0]) * view.ViewSize[1] / view.ViewSize[0])
        if dimension == 2
        else 0.65 * max(upper - lower)
    )
    config.update(camera_position=list(view.CameraPosition), camera_focal_point=list(view.CameraFocalPoint))
    (folder / "manifest.json").write_text(json.dumps(config, indent=2) + "\n")
    title = pv.Text()
    title.Text = (
        "CPT penetration (IGA-MPM) | axisymmetric (r, z)"
        if axisymmetric
        else (
            "Flexible barrier (IGA-MPM) | plane strain (x, z)"
            if dimension == 2
            else "Wavy plate collapse (IGA-MPM) | view from +x/-y/+z"
        )
    )
    label = pv.Show(title, view)
    label.WindowLocation, label.FontSize, label.Color = "Upper Center", 19, [0.1, 0.1, 0.1]
    clock = pv.Text()
    label = pv.Show(clock, view)
    label.WindowLocation, label.FontSize, label.Color = "Lower Left Corner", 16, [0.1, 0.1, 0.1]
    for index in [len(frames) - 1] if preview else range(len(frames)):
        view.ViewTime = index
        for reader in readers:
            reader.UpdatePipeline(index)
        clock.Text = f"t = {frames[index]['time']:.3f} s"
        pv.Render(view)
        pv.SaveScreenshot(str(folder / f"frame_{index:06d}.png"), view, ImageResolution=view.ViewSize)
        print(f"RENDER {index + 1}/{len(frames)}", flush=True)
    if not preview:
        export(config, folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "images/wavy_plate_collapse.gif")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    folder = args.work_dir or Path(tempfile.mkdtemp(prefix="wavy_plate_paraview_"))
    render(args.source.resolve(), folder.resolve(), args.output.resolve(), args.fps, args.preview)
