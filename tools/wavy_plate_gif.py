#!/usr/bin/env pvpython
"""Render saved IGA--MPM wavy-plate frames from +x/-y/+z with stress and speed.

pvpython tools/wavy_plate_gif.py examples/igampm/wavy_plate_collapse/OutputData/ipc_formal_20261006
Color native IGA von Mises stress and MPM velocity magnitude with fixed ranges.
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


def render(source, folder, output, fps, preview):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from paraview import simple as pv

    folder.mkdir(parents=True, exist_ok=True)
    parameters = json.loads((source / "parameters.json").read_text())
    iga = {p.stem[-6:]: p for p in (source / "vtks").glob("NurbsVolumewavy_plate[0-9]*.vtu")}
    mpm = {p.stem[-6:]: p for p in (source / "vtks").glob("GraphicMPMParticle[0-9]*.vtu")}
    if not iga or iga.keys() != mpm.keys():
        raise ValueError("Expected aligned native IGA and MPM frames")
    frames = []
    iga_max = speed_max = 0.0
    bounds = []
    for index, step in enumerate(sorted(iga)):
        time = int(step) * parameters["save_interval"]
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(mpm[step]))
        reader.Update()
        grid = reader.GetOutput()
        velocity = grid.GetPointData().GetArray("velocity")
        if velocity is None or velocity.GetNumberOfComponents() != 3:
            raise ValueError(f"Expected MPM velocity vectors in frame {step}")
        values = vtk_to_numpy(velocity)
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid MPM velocity in frame {step}")
        speed_max = max(speed_max, float(np.linalg.norm(values, axis=1).max()))
        bounds.append(grid.GetBounds())
        reader.SetFileName(str(iga[step]))
        reader.Update()
        grid = reader.GetOutput()
        values = vtk_to_numpy(grid.GetPointData().GetArray("stress"))
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Invalid IGA von Mises stress")
        iga_max = max(iga_max, float(values.max()))
        bounds.append(grid.GetBounds())
        frames.append(dict(step=step, time=time, iga=str(iga[step]), mpm=str(mpm[step])))
        print(f"PREPARE {index + 1}/{len(iga)}", flush=True)
    if iga_max <= 0 or speed_max <= 0:
        raise ValueError("Saved frames must contain nonzero IGA stress and MPM speed")
    config = dict(
        source=str(source),
        output=str(output),
        renderer="ParaView",
        frames=frames,
        fps=fps,
        camera_direction=[1, -1, 0.75],
        deformation_scale=1,
        iga_opacity=0.45,
        iga_von_mises_range=[0, 0.7 * iga_max],
        iga_stress_units="Pa",
        mpm_speed_range=[0, speed_max],
        mpm_speed_units="m/s",
        mpm_scalar="velocity",
        mpm_component="Magnitude",
    )
    pv._DisableFirstRenderCameraReset()
    view = pv.CreateView("RenderView")
    view.ViewSize = [960, 800]
    view.UseColorPaletteForBackground = 0
    view.Background = [1, 1, 1]
    view.OrientationAxesVisibility = 0
    view.UseFXAA = 1
    readers = []
    for key, scalar, title, preset, position, value_range in (
        ("iga", "stress", "IGA: von Mises (Pa)", "Viridis (matplotlib)", [0.08, 0.06], config["iga_von_mises_range"]),
        ("mpm", "velocity", "MPM: speed (m/s)", "Plasma (matplotlib)", [0.56, 0.06], config["mpm_speed_range"]),
    ):
        reader = pv.XMLUnstructuredGridReader(FileName=[frame[key] for frame in frames])
        readers.append(reader)
        display = pv.Show(reader, view)
        display.Representation = "Surface" if key == "iga" else "Points"
        display.Opacity = config["iga_opacity"] if key == "iga" else 1
        if key == "mpm":
            display.PointSize = 4
            display.RenderPointsAsSpheres = 1
            display.Ambient = 1
            display.Diffuse = 0
        pv.ColorBy(display, ("POINTS", scalar, "Magnitude") if key == "mpm" else ("POINTS", scalar))
        lut = pv.GetColorTransferFunction(scalar)
        if key == "mpm":
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
    center[2] -= 0.25
    direction = np.array(config["camera_direction"], dtype=float)
    view.CameraFocalPoint = center.tolist()
    view.CameraPosition = (center + direction / np.linalg.norm(direction) * 4 * max(upper - lower)).tolist()
    view.CameraViewUp = [0, 0, 1]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = 0.65 * max(upper - lower)
    config.update(camera_position=list(view.CameraPosition), camera_focal_point=list(view.CameraFocalPoint))
    (folder / "manifest.json").write_text(json.dumps(config, indent=2) + "\n")
    title = pv.Text()
    title.Text = "Wavy plate collapse (IGA-MPM) | view from +x/-y/+z"
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
