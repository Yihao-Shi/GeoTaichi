"""Run with pvpython: fixed-view, fixed-scale equivalent plastic strain rendering."""

import argparse
import json
from pathlib import Path
import subprocess

import paraview.simple as pvs

from paraview.simple import (
    Calculator,
    ColorBy,
    CreateView,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    GetScalarBar,
    Glyph,
    PVDReader,
    Render,
    SaveScreenshot,
    SaveState,
    Show,
    Text,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--color-max", type=float, default=1.0)
    parser.add_argument("--stride", type=int, default=5, help="Render every Nth saved state for the animation")
    parser.add_argument("--preview", action="store_true", help="Render the final frame only")
    args = parser.parse_args()
    if args.stride < 1 or args.fps < 1 or args.color_max <= 0:
        parser.error("stride, fps and color maximum must be positive")
    pvs._DisableFirstRenderCameraReset()
    root = args.output.resolve()
    report = json.loads((root / "evaluation.json").read_text())
    if not report["passed"]:
        raise ValueError("Evaluate the completed simulation successfully before rendering")
    frames = root / "render" / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    source = PVDReader(FileName=str(root / "sensitive_clay.pvd"))
    size = Calculator(Input=source)
    size.ResultArrayName = "particle_width"
    size.Function = "sqrt(volume)"
    glyph = Glyph(Input=size, GlyphType="2D Glyph")
    glyph.GlyphType.GlyphType = "Square"
    glyph.GlyphType.Filled = 1
    glyph.OrientationArray = ["POINTS", "No orientation array"]
    glyph.ScaleArray = ["POINTS", "particle_width"]
    glyph.ScaleFactor = 1.0
    glyph.GlyphMode = "All Points"
    view = CreateView("RenderView")
    view.ViewSize = [1920, 720]
    view.UseColorPaletteForBackground = 0
    view.Background = [0.98, 0.98, 0.98]
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.InteractionMode = "2D"
    xmin, xmax, ymin, ymax = report["bounds"]
    width = max(30.0, xmax - xmin + 2.0)
    center = [(xmin + xmax) / 2, 2.5, 0]
    view.CameraPosition = [center[0], center[1], 100]
    view.CameraFocalPoint = center
    view.CameraViewUp = [0, 1, 0]
    view.CameraParallelScale = max(4.5, width * 720 / 1920 / 2)
    display = Show(glyph, view)
    display.Representation = "Surface"
    display.Ambient = 1.0
    display.Diffuse = 0.0
    ColorBy(display, ("POINTS", "epdstrain"))
    lut = GetColorTransferFunction("epdstrain")
    lut.ApplyPreset("Cool to Warm (Extended)", True)
    lut.RescaleTransferFunction(0, args.color_max)
    lut.AutomaticRescaleRangeMode = "Never"
    GetOpacityTransferFunction("epdstrain").RescaleTransferFunction(0, args.color_max)
    display.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(lut, view)
    bar.Title = "Equivalent plastic strain"
    bar.ComponentTitle = ""
    bar.Orientation = "Horizontal"
    bar.WindowLocation = "Any Location"
    bar.Position = [0.30, 0.08]
    bar.ScalarBarLength = 0.40
    bar.TitleColor = [0.12, 0.12, 0.12]
    bar.LabelColor = [0.12, 0.12, 0.12]
    bar.TitleFontSize = 19
    bar.LabelFontSize = 17
    bar.RangeLabelFormat = "%.2f"
    bar.LabelFormat = "%.2f"
    title = Text()
    title_display = Show(title, view)
    title_display.WindowLocation = "Upper Left Corner"
    title_display.FontSize = 20
    title_display.Color = [0.12, 0.12, 0.12]
    last = len(report["history"]) - 1
    model = report["run"].get("material_model", "MohrCoulomb")
    grid = "adaptive" if report["run"]["adaptive"] else "uniform grid"
    selected = [last] if args.preview else sorted(set(range(0, last + 1, args.stride)) | {last})
    for frame_index, index in enumerate(selected):
        row = report["history"][index]
        view.ViewTime = row["time"]
        source.UpdatePipeline(row["time"])
        glyph.UpdatePipeline(row["time"])
        title.Text = (
            "Sensitive clay - 2D %s MPM, %s\nt = %.2f s - %s particles\n"
            "epdstrain - fixed range 0 - %.2f (values above range saturated)"
            % (model, grid, row["time"], format(row["particles"], ","), args.color_max)
        )
        Render(view)
        SaveScreenshot(str(frames / ("frame_%04d.png" % frame_index)), view, ImageResolution=[1920, 720])
        print("Rendered frame %d at %.2f s" % (index, row["time"]), flush=True)
    SaveScreenshot(str(root / "render" / "final.png"), view, ImageResolution=[1920, 720])
    SaveState(str(root / "render" / "sensitive_clay.pvsm"))
    if not args.preview:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(args.fps),
                "-i",
                str(frames / "frame_%04d.png"),
                "-frames:v",
                str(len(selected)),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(root / "render" / "sensitive_clay_epdstrain.mp4"),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(root / "render" / "sensitive_clay_epdstrain.mp4"),
                "-vf",
                "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                "-loop",
                "0",
                str(root / "render" / "sensitive_clay_epdstrain.gif"),
            ],
            check=True,
        )
    (root / "render" / "render.json").write_text(
        json.dumps(
            {
                "renderer": "ParaView",
                "field": "epdstrain",
                "color_range": [0, args.color_max],
                "glyph": "square with width sqrt(particle volume), unit plane-strain thickness",
                "fps": args.fps,
                "preview": args.preview,
                "stride": args.stride,
                "rendered_times": [report["history"][i]["time"] for i in selected],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
