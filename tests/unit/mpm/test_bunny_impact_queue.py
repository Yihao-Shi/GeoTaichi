import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from examples.mmpm.TwoPhaseLSDEMCoupling.sphere_impact_submerged_bed_3d.draw.evaluate_sphere_impact_submerged_bed_3d import (
    fluid_surface_coverage,
    write_metrics,
)

ROOT = Path(__file__).resolve().parents[3]


def test_sphere_defaults_are_unchanged_and_bunny_has_separate_output(monkeypatch):
    from examples.mmpm.TwoPhaseLSDEMCoupling.sphere_impact_submerged_bed_3d.sphere_impact_submerged_bed_3d import (
        parse_args,
    )

    monkeypatch.setattr(sys, "argv", ["impact"])
    sphere = parse_args()
    assert sphere.impactor == "sphere" and sphere.ppc == 2
    assert sphere.output.name == "sphere_impact_section_5_2"
    monkeypatch.setattr(sys, "argv", ["impact", "--impactor", "bunny"])
    assert parse_args().output.name == "bunny_impact_section_5_2"


def test_bunny_mesh_is_closed_without_changing_outline():
    original = trimesh.load(ROOT / "assets/bunny_sparse.obj", force="mesh")
    closed = trimesh.load(ROOT / "assets/mesh/LSDEM/bunny_impact_watertight.ply", force="mesh")
    assert closed.is_watertight and closed.is_winding_consistent and closed.volume > 0
    assert len(closed.vertices) == len(original.vertices) < 5000
    np.testing.assert_allclose(closed.bounds, original.bounds, atol=1e-7)
    assert abs(closed.volume / original.volume - 1) < 0.01


def test_bunny_metrics_use_actual_surface_and_require_synchronized_frames(tmp_path):
    directory = tmp_path / "particles"
    directory.mkdir()
    local = np.array([[-0.01, -0.01, -0.025], [0.01, -0.01, -0.025], [0, 0.01, -0.025], [0, 0, 0.02]])
    final_surface = local + [0.1, 0.1, 0.11]
    assert fluid_surface_coverage(final_surface, final_surface, 0.0025) == 1
    assert fluid_surface_coverage(np.empty((0, 3)), final_surface, 0.0025) == 0
    position = np.vstack([final_surface, [0.05, 0.05, 0.05]])
    for step, (time, center_z, velocity, force) in enumerate(((0, 0.175, -4.429, 0), (0.12, 0.11, -0.2, 100))):
        np.savez(
            directory / f"MPMParticle{step:06d}.npz",
            t_current=time,
            active=np.ones(5),
            phase=[2, 2, 2, 2, 1],
            position=position,
            fluid_velocity=np.zeros((5, 3)),
            solid_velocity=np.zeros((5, 3)),
            pressure=np.zeros(5),
        )
        np.savez(
            directory / f"LSDEMRigid{step:06d}.npz",
            t_current=time,
            mass_center=[[0.1, 0.1, center_z]],
            velocity=[[0, 0, velocity]],
            contact_force=[[0, 0, force]],
            quanternion=[[0, 0, 0, 1]],
            omega=[[0, 0, 0]],
            contact_torque=[[0, 0, 0]],
            startNode=[0],
            localNode=[0],
            scale=[1.0],
        )
        np.savez(
            directory / f"LSDEMSurface{step:06d}.npz",
            t_current=time,
            vertices=local,
            master=np.zeros(4, dtype=int),
            connectivity=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        )
    args = SimpleNamespace(dx=0.005, dt=1e-5, time=0.12, drop_height=1.0, ppc=2, strict=True, impactor="bunny")
    write_metrics(tmp_path, 4, 1, args)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["passed"]
    assert metrics["impactor_maximum_bed_penetration_m"] == pytest.approx(0.015)
    assert metrics["minimum_submerged_fluid_surface_coverage"] == 1
    assert (tmp_path / "bunny_impact_trajectory.csv").exists()
    (directory / "LSDEMRigid000000.npz").unlink()
    with pytest.raises(RuntimeError):
        write_metrics(tmp_path, 4, 1, args)


def test_queue_runs_later_cases_after_bunny_failure(tmp_path):
    fake_python = tmp_path / "fake_python"
    calls = tmp_path / "calls"
    fake_python.write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$*" >> "{calls}"\n' '[[ "$*" == *"--impactor bunny"* ]] && exit 7\nexit 0\n'
    )
    fake_python.chmod(0o755)
    logroot = tmp_path / "logs"
    repo = tmp_path / "repo"
    (repo / "research/remote_validation").mkdir(parents=True)
    env = dict(
        os.environ,
        GEOTAICHI_QUEUE_REPO=str(repo),
        GEOTAICHI_QUEUE_PYTHON=str(fake_python),
        GEOTAICHI_QUEUE_LOGROOT=str(logroot),
    )
    result = subprocess.run(
        ["bash", str(ROOT / "research/remote_validation/run_semi_implicit_revision_queue_20261010.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    commands = calls.read_text().splitlines()
    assert len(commands) == 4
    assert "--impactor bunny" in commands[0]
    assert all("--impactor sphere" not in command for command in commands)
    records = (logroot / "results.tsv").read_text().splitlines()
    assert records[0].split("\t")[1] == "7"
    assert all(row.split("\t")[1] == "0" for row in records[1:])
    assert "complete\tfailures=1" in (logroot / "queue.status").read_text()
