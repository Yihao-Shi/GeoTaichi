"""Export saved two-point MPM/LSDEM snapshots without rerunning the solver."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import meshio
import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dem.PostPlot import write_lsdem_vtk_file
from src.mpm.PostPlot import write_vtk_file
from src.mpm.Recorder import WriteFile
import src.utils.GlobalVariable as GlobalVariable
from tools.postprocessing.mpdem.reprocess_cfdem_vtk import digest


def export(folder):
    folder = Path(folder).resolve()
    particles = sorted((folder / "particles").glob("MPMParticle*.npz"))
    if not particles or [int(f.stem[-6:]) for f in particles] != list(range(len(particles))):
        raise ValueError(f"Missing or non-contiguous particle frames: {folder}")
    with np.load(particles[0]) as first:
        dimension = first["position"].shape[1]
    if dimension not in (2, 3):
        raise ValueError(f"Unsupported particle dimension: {dimension}")
    destination = folder / "vtks"
    if any(destination.glob("Graphic*")):
        raise FileExistsError(f"Refusing to overwrite existing VTK data: {destination}")
    lsdem = bool(list((folder / "particles").glob("LSDEMRigid*.npz")))
    before = {str(f.relative_to(folder)): digest(f) for f in (particles[0], particles[-1])}
    times = []
    for frame, particle in enumerate(particles):
        with np.load(particle) as info:
            time = float(info["t_current"])
            if set(np.unique(info["phase"])) != {1, 2}:
                raise ValueError(f"Missing solid or fluid phase: {particle}")
        sources = [folder / "grids" / f"MPMGrid{frame:06d}.npz"]
        if lsdem:
            sources.extend(folder / "particles" / f"{name}{frame:06d}.npz" for name in ("LSDEMRigid", "LSDEMSurface"))
        for source in sources:
            with np.load(source) as info:
                if not np.isclose(float(info["t_current"]), time, rtol=0, atol=1e-8):
                    raise ValueError(f"Snapshot times differ: {source}")
        if times and time <= times[-1]:
            raise ValueError(f"Non-increasing snapshot time: {particle}")
        times.append(time)
    destination.mkdir(exist_ok=True)
    sims = SimpleNamespace(
        material_type="TwoPhaseDoubleLayer", solver_type="SemiImplicit", dimension=dimension, current_print=0
    )
    previous_dimension = GlobalVariable.DIMENSION
    GlobalVariable.DIMENSION = dimension
    try:
        write_vtk_file(sims, 0, len(particles), str(folder), str(destination), {})
    finally:
        GlobalVariable.DIMENSION = previous_dimension
    if lsdem:
        write_lsdem_vtk_file(sims, 0, len(particles), str(folder), str(destination), {})
    recorder = WriteFile.__new__(WriteFile)
    recorder.vtk_path = str(destination)
    for frame in range(len(particles)):
        sims.current_print = frame
        with np.load(folder / "grids" / f"MPMGrid{frame:06d}.npz") as grid:
            recorder.VisualizeGrid(
                sims, grid["coords"], cell_data={k: grid[k] for k in grid.files if k.startswith("cell_")}
            )
    prefixes = ["GraphicMPMFluidParticle", "GraphicMPMSolidParticle", "GraphicMPMGrid"]
    if lsdem:
        prefixes.append("GraphicLSDEMSurface")
    for prefix in prefixes:
        if len(list(destination.glob(f"{prefix}*.vtu"))) != len(particles):
            raise RuntimeError(f"Incomplete export: {prefix}")
    for frame in sorted({0, len(particles) - 1}):
        with np.load(particles[frame]) as source:
            for phase, prefix, velocity in (
                (1, "GraphicMPMSolidParticle", "solid_velocity"),
                (2, "GraphicMPMFluidParticle", "fluid_velocity"),
            ):
                mask = source["phase"] == phase
                mesh = meshio.read(destination / f"{prefix}{frame:06d}.vtu")
                np.testing.assert_array_equal(mesh.points[:, :dimension], source["position"][mask])
                np.testing.assert_array_equal(mesh.point_data["pressure"].reshape(-1), source["pressure"][mask])
                np.testing.assert_array_equal(mesh.point_data["velocity"][:, :dimension], source[velocity][mask])
        mesh = meshio.read(destination / f"GraphicMPMGrid{frame:06d}.vtu")
        with np.load(folder / "grids" / f"MPMGrid{frame:06d}.npz") as source:
            np.testing.assert_array_equal(
                np.concatenate(mesh.cell_data["cell_pressure"]).reshape(-1), source["cell_pressure"].reshape(-1)
            )
        if lsdem:
            mesh = meshio.read(destination / f"GraphicLSDEMSurface{frame:06d}.vtu")
            if not np.isfinite(mesh.points).all() or not mesh.cells:
                raise RuntimeError(f"Invalid LSDEM mesh: frame {frame}")
    for relative, expected in before.items():
        if digest(folder / relative) != expected:
            raise RuntimeError(f"Raw snapshot changed: {relative}")
    report = {
        "passed": True,
        "frames": len(particles),
        "dimension": dimension,
        "final_time_s": times[-1],
        "lsdem": lsdem,
        "vtk_files": len(prefixes) * len(particles),
        "raw_pressure_position_velocity_preserved": True,
        "raw_sha256_checked": before,
    }
    (folder / "vtu_export_20261013.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"folder": str(folder), **report}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=4)
    export(args.folder)
