import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytestmark = [
    pytest.mark.verification,
    pytest.mark.mpdem,
    pytest.mark.cpu,
    pytest.mark.slow,
    pytest.mark.serial,
]


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples/mpm/IncompressibleFluid/lsdem_coupling_3d/lsdem_coupling_3d.py"


def run_solver(tmp_path, solver):
    output = tmp_path / solver.lower()
    environment = os.environ.copy()
    environment.update(
        {
            "GEOTAICHI_REAL_DTYPE": "float64",
            "GT_LSDEM_ARCH": "cpu",
            "GT_LSDEM_LINEAR_SOLVER": solver,
            "GT_LSDEM_SIMULATION_TIME": "4e-4",
            "GT_LSDEM_SAVE_INTERVAL": "4e-4",
            "GT_LSDEM_SAVE_PATH": str(output),
            "GT_LSDEM_POSTPROCESS": "0",
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
            "PYTHONWARNINGS": "error::RuntimeWarning",
        }
    )
    runner = f"""import json, math, runpy
import numpy as np
model = runpy.run_path({str(EXAMPLE)!r}, run_name="__main__")
dempm = model["dempm"]
pressure_solver = dempm.mpm.enginer.poisson_solver
if hasattr(pressure_solver, "last_converged"):
    converged = pressure_solver.last_converged
    initial_residual = pressure_solver.last_initial_residual
    final_residual = pressure_solver.last_residual
else:
    initial_residual = pressure_solver.initial_residual
    final_residual = pressure_solver.final_residual
    converged = final_residual <= max(1.0e-14, initial_residual * 1.0e-10)
coupler = dempm.enginer.incompressible_coupler
grid_size = np.array([float(dempm.mpm.scene.element.grid_size[d]) for d in range(3)])
mapped_solid_volume = float(np.sum(coupler.solid_fraction.to_numpy()) * np.prod(grid_size))
equivalent_radius = float(dempm.dem.scene.rigid[0].equi_r)
rigid_volume = 4.0 * math.pi * equivalent_radius**3 / 3.0
print("GT_SOLVER_EVIDENCE=" + json.dumps({{
    "converged": bool(converged),
    "initial_residual": float(initial_residual),
    "final_residual": float(final_residual),
    "iterations": int(pressure_solver.last_iterations),
    "mapped_solid_volume": mapped_solid_volume,
    "rigid_volume": rigid_volume,
    "solid_volume_relative_error": abs(mapped_solid_volume - rigid_volume) / rigid_volume,
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", runner],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-8000:]
    assert "RuntimeWarning" not in completed.stdout + completed.stderr
    solver_evidence = json.loads(
        next(
            line.removeprefix("GT_SOLVER_EVIDENCE=")
            for line in completed.stdout.splitlines()
            if line.startswith("GT_SOLVER_EVIDENCE=")
        )
    )

    particle = np.load(sorted((output / "particles").glob("MPMParticle*.npz"))[-1])
    grid = np.load(sorted((output / "grids").glob("MPMGrid*.npz"))[-1])
    rigid = np.load(sorted((output / "particles").glob("LSDEMRigid*.npz"))[-1])
    return {
        "time": float(particle["t_current"]),
        "position": particle["position"],
        "velocity": particle["velocity"],
        "pressure": particle["pressure"],
        "cell_pressure": grid["cell_pressure"],
        "cell_type": grid["cell_type"],
        "force": rigid["contact_force"],
        "torque": rigid["contact_torque"],
        "solver": solver_evidence,
    }


def relative_l2(left, right):
    scale = max(np.linalg.norm(left), np.linalg.norm(right), 1.0e-30)
    return np.linalg.norm(left - right) / scale


def test_fully_resolved_pcg_and_mgpcg_agree_end_to_end(tmp_path):
    pcg = run_solver(tmp_path, "PCG")
    mgpcg = run_solver(tmp_path, "MGPCG")

    assert pcg["solver"]["converged"]
    assert mgpcg["solver"]["converged"]
    assert pcg["solver"]["solid_volume_relative_error"] < 0.02
    assert mgpcg["solver"]["solid_volume_relative_error"] < 0.02
    assert pcg["time"] == pytest.approx(mgpcg["time"], abs=1.0e-15)
    np.testing.assert_array_equal(pcg["cell_type"], mgpcg["cell_type"])
    for name in ("position", "velocity", "pressure", "cell_pressure", "force", "torque"):
        assert np.isfinite(pcg[name]).all()
        assert np.isfinite(mgpcg[name]).all()
        assert relative_l2(pcg[name], mgpcg[name]) < 1.0e-4
