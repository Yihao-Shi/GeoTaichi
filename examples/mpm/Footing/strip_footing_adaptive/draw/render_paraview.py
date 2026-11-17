"""Render exported L2 footing data with pvpython; no simulation is run here."""

import argparse
import json
from pathlib import Path
import subprocess

import paraview.simple as pvs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    render = args.output.resolve() / "render"
    run = json.loads((args.output.resolve() / "run.json").read_text())
    spacing = {"soil": run["initial_spacing"], "footing": run.get("rigid_spacing", run["initial_spacing"])}
    inputs = json.loads((render / "inputs.json").read_text())
    frames = render / "frames"
    frames.mkdir(exist_ok=True)
    pvs._DisableFirstRenderCameraReset()
    view = pvs.CreateView("RenderView")
    view.ViewSize = [1000, 1120]
    view.UseColorPaletteForBackground = 0
    view.Background = [1, 1, 1]
    view.OrientationAxesVisibility = 0
    view.InteractionMode = "2D"
    view.CameraParallelProjection = 1
    view.CameraPosition = [5, 5.7, 100]
    view.CameraFocalPoint = [5, 5.7, 0]
    view.CameraViewUp = [0, 1, 0]
    view.CameraParallelScale = 7.0
    sources = {}
    for name in ("soil", "footing"):
        source = pvs.PVDReader(FileName=str(render / f"{name}.pvd"))
        surface = pvs.Delaunay2D(Input=source)
        # The initial coarse cloud must form a surface before its particles split.
        surface.Alpha = 1.1 * spacing[name]
        display = pvs.Show(surface, view)
        display.Representation = "Surface"
        display.Ambient = 1.0
        display.Diffuse = 0.0
        if name == "soil":
            pvs.ColorBy(display, ("POINTS", "epdstrain"))
            lut = pvs.GetColorTransferFunction("epdstrain")
            lut.ApplyPreset("Cool to Warm", True)
            lut.RescaleTransferFunction(0, 1)
            lut.AutomaticRescaleRangeMode = "Never"
            display.SetScalarBarVisibility(view, True)
            bar = pvs.GetScalarBar(lut, view)
            bar.Title = "Equivalent plastic strain"
            bar.ComponentTitle = ""
            bar.Orientation = "Horizontal"
            bar.WindowLocation = "Any Location"
            bar.Position = [0.25, 0.012]
            bar.ScalarBarLength = 0.5
            bar.TitleColor = [0.12, 0.12, 0.12]
            bar.LabelColor = [0.12, 0.12, 0.12]
            bar.TitleFontSize = 20
            bar.LabelFontSize = 17
            bar.RangeLabelFormat = "%.1f"
            bar.LabelFormat = "%.1f"
        else:
            pvs.ColorBy(display, None)
            display.DiffuseColor = [0.28, 0.30, 0.34]
            display.AmbientColor = [0.28, 0.30, 0.34]
        sources[name] = surface
    grid = pvs.PVDReader(FileName=str(render / "grid.pvd"))
    mesh = pvs.Show(grid, view)
    pvs.ColorBy(mesh, None)
    mesh.Representation = "Wireframe"
    mesh.AmbientColor = [0.35, 0.38, 0.42]
    mesh.DiffuseColor = [0.35, 0.38, 0.42]
    mesh.LineWidth = 0.6
    mesh.Opacity = 0.3
    sources["grid"] = grid
    title = pvs.Text()
    label = pvs.Show(title, view)
    label.WindowLocation = "Upper Left Corner"
    label.FontSize = 20
    label.Color = [0.12, 0.12, 0.12]
    selected = inputs["frames"][-1:] if args.preview else inputs["frames"]
    for index, frame in enumerate(selected):
        time = frame["time_s"]
        view.ViewTime = time
        for source in sources.values():
            source.UpdatePipeline(time)
        title.Text = f"Adaptive MPM strip footing - 2D\nL2: h = 0.4 / 0.2 / 0.1 m    t = {time:.1f} s"
        pvs.Render(view)
        pvs.SaveScreenshot(str(frames / f"frame_{index:04d}.png"), view, ImageResolution=view.ViewSize)
        print(f"Rendered t = {time:.1f} s", flush=True)
    pvs.SaveScreenshot(str(render / "final.png"), view, ImageResolution=view.ViewSize)
    pvs.SaveState(str(render / "footing_l2.pvsm"))
    if not args.preview:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "4",
                "-i",
                str(frames / "frame_%04d.png"),
                "-vf",
                "tpad=stop_mode=clone:stop_duration=1,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                "-loop",
                "0",
                str(render / "footing_l2.gif"),
            ],
            check=True,
        )
        (render / "render.json").write_text(
            json.dumps(
                {
                    "application": "ParaView",
                    "frames": len(selected),
                    "saved_interval_s": 1.0,
                    "final_interval_s": 0.5,
                    "field": "epdstrain",
                    "color_range": [0, 1],
                    "above_range": "saturated",
                    "grid": "actual saved adaptive leaf mesh",
                    "fps": 4,
                    "time_interpolation": False,
                    "surface": "Delaunay2D of saved particle centers; alpha = 1.1 times each body's initial spacing",
                    "surface_alpha_m": {name: 1.1 * value for name, value in spacing.items()},
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
