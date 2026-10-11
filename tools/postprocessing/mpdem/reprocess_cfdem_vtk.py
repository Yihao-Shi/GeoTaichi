"""Re-export the five adopted CFDEM results without changing simulation NPZs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dem.PostPlot import write_dem_vtk_file, write_lsdem_vtk_file
from src.mpm.PostPlot import write_vtk_file
from src.mpm.Recorder import (
    WriteFile,
    _closed_domain_pressure_offset,
    _has_resolved_air_reference,
    _normalize_closed_domain_pressure,
)
import src.utils.GlobalVariable as GlobalVariable

CASES = (
    ("FullyResolved/IBMDraftingKissingTumbling/OutputData/ibm_dkt_strict_final_d8p35", 3, "LSDEM"),
    ("FullyResolved/IBMResolvedSphereSettling/OutputData/sphere_settling_strict_d10_t0p8_fixed", 3, "LSDEM"),
    ("SemiResolved/SphereFallingOil/OutputData/E4_CA2_wall_lubrication_formal_20260910", 3, "DEM"),
    ("FullyResolved/IBMLevelSetDamBreak3D/OutputData/dam3d_highres_sdf_vf/output", 3, "LSDEM"),
    ("FullyResolved/IBMFixedCylinder2D/OutputData/ibm2d_karman_postfix_20260907/model", 2, None),
)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def reprocess_case(folder, backup_root, dimension, scheme):
    files = sorted((folder / "particles").glob("MPMParticle*.npz"))
    if not files or [int(f.stem[-6:]) for f in files] != list(range(len(files))):
        raise ValueError(f"Missing or non-contiguous particle snapshots: {folder}")
    checked_files = {
        files[0],
        files[-1],
        folder / "grids/MPMGrid000000.npz",
        folder / "grids" / f"MPMGrid{len(files)-1:06d}.npz",
    }
    checked_files.update(
        f
        for f in (
            folder / "configuration.json",
            folder / "trajectory.npz",
            folder / "metrics.json",
            folder / "acceptance_pre_wall_20261003.json",
        )
        if f.exists()
    )
    before = {str(f.relative_to(folder)): digest(f) for f in checked_files}
    destination = folder / "vtks"
    backup = backup_root / folder.name / "vtks"
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite backup: {backup}")
    staging = Path(tempfile.mkdtemp(prefix="vtks.reprocess-", dir=folder))
    GlobalVariable.DIMENSION = dimension
    sims = SimpleNamespace(material_type="Fluid", solver_type="Implicit", dimension=dimension, current_print=0)
    write_vtk_file(sims, 0, len(files), str(folder), str(staging), {})
    if scheme == "LSDEM":
        write_lsdem_vtk_file(sims, 0, len(files), str(folder), str(staging), {})
    elif scheme == "DEM":
        write_dem_vtk_file(sims, 0, len(files), str(folder), str(staging), {})
    recorder = WriteFile.__new__(WriteFile)
    recorder.vtk_path = str(staging)
    rows = []
    for step, particle_file in enumerate(files):
        with np.load(particle_file) as particles, np.load(folder / "grids" / f"MPMGrid{step:06d}.npz") as grid:
            if abs(float(particles["t_current"]) - float(grid["t_current"])) > 1e-8:
                raise ValueError(f"Particle/grid snapshot times differ: {particle_file}")
            types = grid["cell_type"]
            offset = _closed_domain_pressure_offset(grid["cell_pressure"], types)
            cell_data = {k: grid[k] for k in grid.files if k.startswith("cell_")}
            cell_data["cell_pressure"] = _normalize_closed_domain_pressure(grid["cell_pressure"], types)
            pressure = particles["pressure"] - offset
            fluid = types == 1
            if not np.isfinite(pressure).all() or not np.isfinite(cell_data["cell_pressure"][fluid]).all():
                raise ValueError(f"Non-finite pressure: {particle_file}")
            sims.current_print = step
            recorder.VisualizeGrid(sims, grid["coords"], cell_data=cell_data)
            rows.append(
                {
                    "frame": step,
                    "time_s": float(particles["t_current"]),
                    "pressure_offset_pa": offset,
                    "has_air_cells": bool(np.any(types == 0)),
                    "has_air_reference": _has_resolved_air_reference(types),
                    "fluid_grid_pressure_mean_pa": float(cell_data["cell_pressure"][fluid].mean()),
                    "particle_pressure_mean_pa": float(pressure.mean()),
                }
            )
    for name, expected in before.items():
        if digest(folder / name) != expected:
            raise RuntimeError(f"Source data changed during postprocessing: {folder / name}")
    if len(list(staging.glob("GraphicMPMParticle*.vtu"))) != len(files) or len(
        list(staging.glob("GraphicMPMGrid*.vtu"))
    ) != len(files):
        raise RuntimeError(f"Incomplete VTK export: {staging}")
    # Read the actual exported pressure, not just the pre-export arrays.
    import meshio

    changed_gauge_frames = {row["frame"] for row in rows if row["has_air_cells"] and not row["has_air_reference"]}
    for step in sorted({0, min(1, len(files) - 1), len(files) - 1} | changed_gauge_frames):
        mesh = meshio.read(staging / f"GraphicMPMParticle{step:06d}.vtu")
        with np.load(files[step]) as source:
            np.testing.assert_allclose(
                mesh.point_data["pressure"].reshape(-1), source["pressure"] - rows[step]["pressure_offset_pa"]
            )
            np.testing.assert_array_equal(mesh.points[:, :dimension], source["position"][:, :dimension])
            np.testing.assert_array_equal(mesh.point_data["velocity"][:, :dimension], source["velocity"][:, :dimension])
        mesh = meshio.read(staging / f"GraphicMPMGrid{step:06d}.vtu")
        with np.load(folder / "grids" / f"MPMGrid{step:06d}.npz") as source:
            expected = _normalize_closed_domain_pressure(source["cell_pressure"], source["cell_type"])
            np.testing.assert_allclose(
                np.concatenate(mesh.cell_data["cell_pressure"]).reshape(-1), expected.reshape(-1)
            )
    backup.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.move(str(destination), str(backup))
    try:
        staging.rename(destination)
    except OSError:
        if backup.exists():
            shutil.move(str(backup), str(destination))
        raise
    report = {
        "passed": True,
        "frames": rows,
        "vtk_files": len(list(destination.glob("Graphic*.vtu"))),
        "raw_data_unchanged_sha256": before,
        "previous_vtks_backup": str(backup),
        "gauge": "fluid-cell zero mean for closed domains; unchanged atmospheric gauge otherwise",
    }
    for manifest in (folder / "SHA256SUMS", folder.parent / "SHA256SUMS"):
        if not manifest.exists():
            continue
        lines = manifest.read_text().splitlines()
        derived = [
            (manifest.parent / line.split("  ", 1)[1]).resolve().is_relative_to(destination.resolve()) for line in lines
        ]
        if any(derived):
            shutil.copy2(manifest, backup.parent / f"{manifest.parent.name}_SHA256SUMS.original")
            retained = [line for line, is_derived in zip(lines, derived) if not is_derived]
            retained.extend(
                f"{digest(f)}  ./{f.relative_to(manifest.parent)}" for f in sorted(destination.glob("Graphic*.vtu"))
            )
            manifest.write_text("\n".join(retained) + "\n")
    (folder / "vtk_postprocess_pressure_report_20261011.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"case": str(folder), "snapshots": len(files), "vtk_files": report["vtk_files"], "passed": True}),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, required=True)
    args = parser.parse_args()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=4)
    for relative, dimension, scheme in CASES:
        reprocess_case(ROOT / "examples/cfdem" / relative, args.backup_root.expanduser().resolve(), dimension, scheme)


if __name__ == "__main__":
    main()
