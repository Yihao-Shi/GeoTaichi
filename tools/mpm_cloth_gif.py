#!/usr/bin/env python3
"""Render saved FEM cloth--MPM IPC contact with Blender and single-color cloth.

python tools/mpm_cloth_gif.py --work-dir /tmp/mpm_cloth_gallery
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from blender_cfdem_gif import ROOT, export
from fedem_gifs import read_surface
from incompressible_gifs import BLENDER

SOURCE = (
    ROOT / "examples/fempm/implicit_ipc_elastic_contact/OutputData/fem_mpm_cloth_dp_spread_t060_gridpad_v2_20260917"
)


def prepare(source, folder):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from scipy.spatial import cKDTree

    summary = json.loads((source / "validation_summary.json").read_text())
    if summary["case"] != "ordinary_mpm_cloth_fem_ipc" or not summary["converged"] or not summary["finite"]:
        raise ValueError("Expected a completed, finite cloth--MPM IPC result")
    cloth = {p.stem[-6:]: p for p in (source / "vtks").glob("FEM[0-9]*.vtu")}
    particles = {p.stem[-6:]: p for p in (source / "vtks").glob("GraphicMPMParticle[0-9]*.vtu")}
    steps = sorted(cloth)
    if len(steps) < 2 or cloth.keys() != particles.keys() or steps != [f"{i:06d}" for i in range(len(steps))]:
        raise ValueError("Expected aligned, consecutive cloth and MPM snapshots starting at zero")
    duration = float(summary["completed_time"])
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Expected a finite positive simulation duration")
    folder.mkdir(parents=True, exist_ok=True)
    frames, geometry = [], []
    for index, step in enumerate(steps):
        vertices, faces = read_surface(cloth[step])
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(particles[step]))
        reader.Update()
        grid = reader.GetOutput()
        if reader.GetErrorCode() or grid.GetNumberOfPoints() != summary["mpm_particles"]:
            raise ValueError(f"Invalid saved MPM frame {step}")
        positions = vtk_to_numpy(grid.GetPoints().GetData()).copy()
        if (
            not np.isfinite(positions).all()
            or len(vertices) != summary["fem_nodes"]
            or len(faces) != summary["fem_elements"]
        ):
            raise ValueError(f"Invalid cloth/particle geometry in frame {step}")
        if index == 0:
            spacing = float(np.median(cKDTree(positions).query(positions, k=2)[0][:, 1]))
            if not np.isfinite(spacing) or spacing <= 0:
                raise ValueError("Initial particle marker spacing must be positive")
        geometry.append((vertices, faces, positions))
        # shortcut: these VTUs lack times; use the uniform recorder schedule ending at completed_time.
        frames.append(dict(time=index * duration / (len(steps) - 1), step=step, mesh=f"mesh_{index:06d}.npz"))
    bounds = np.concatenate([points for vertices, _, positions in geometry for points in (vertices, positions)])
    origin = bounds.min(axis=0)
    origin[2] = min(0.0, origin[2])
    extent = bounds.max(axis=0) - origin
    scale = 3.6 / extent.max()
    for frame, (vertices, faces, positions) in zip(frames, geometry):
        np.savez(
            folder / frame["mesh"],
            soft=(vertices - origin) * scale,
            soft_faces=faces,
            solid_positions=(positions - origin) * scale,
            solid_radii=np.full(len(positions), 0.45 * spacing * scale),
        )
    config = dict(
        case="mpm_cloth",
        source=str(source),
        output=str(ROOT / "images/mpm_cloth_ipc.gif"),
        title="MPM block on FEM cloth (IPC)",
        legend="Orange: cloth | Brown: MPM particles",
        frames=frames,
        domain=(extent * scale).tolist(),
        origin_m=origin.tolist(),
        display_scale=scale,
        fps=8,
        width=960,
        height=800,
        camera_direction=[1, -1.5, 0.8],
        show_tank=False,
        cloth_display="single_color_native_mesh",
        particle_display="markers_at_saved_positions",
        particle_marker_radius_m=0.45 * spacing,
        timing_source="uniform_snapshots_and_completed_time",
    )
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    return manifest, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=SOURCE)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--preview", action="store_true", help="Render the last saved frame without replacing the GIF")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    folder = (args.work_dir or Path(tempfile.mkdtemp(prefix="mpm_cloth_blender_"))).resolve()
    manifest, config = prepare(args.source.resolve(), folder)
    if args.preview:
        preview = dict(config, frames=config["frames"][-1:])
        manifest = folder / "preview.json"
        manifest.write_text(json.dumps(preview, indent=2) + "\n")
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
    if not args.preview:
        export(config, folder)
