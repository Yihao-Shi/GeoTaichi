import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from examples.mmpm.TwoPhaseLSDEMCoupling.sphere_impact_submerged_bed_3d.draw.evaluate_sphere_impact_submerged_bed_3d import (
    fluid_surface_coverage,
    wet_surface_coverage,
    write_metrics,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "relative",
    [
        "examples/mmpm/TwoPhaseLSDEMCoupling/sphere_impact_submerged_bed_3d/sphere_impact_submerged_bed_3d.py",
        "examples/mmpm/TwoPhaseLSDEMCoupling/wavemaker_lsdem_particles_3d/wavemaker_lsdem_particles_3d.py",
        "examples/mmpm/TwoPhaseWavemaker3D/two_layer_two_phase_wavemaker_3d.py",
        "examples/mmpm/GranularWaterLeakage3D/granular_water_leakage_3d.py",
    ],
)
def test_no_post_disables_live_vtk_without_disabling_raw_recorder(relative):
    tree = ast.parse((ROOT / relative).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    visualization = [keyword.value for call in calls for keyword in call.keywords if keyword.arg == "visualize"]
    assert visualization
    for no_post in (True, False):
        for expression in visualization:
            value = eval(
                compile(ast.Expression(expression), relative, "eval"), {"args": SimpleNamespace(no_post=no_post)}
            )
            assert value is not no_post
    raw = [call for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == "select_save_data"]
    assert any(
        {keyword.arg: isinstance(keyword.value, ast.Constant) and keyword.value.value for keyword in call.keywords}.get(
            "particle"
        )
        for call in raw
    )


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
    (tmp_path / "grids").mkdir()
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
        np.savez(tmp_path / "grids" / f"MPMGrid{step:06d}.npz", t_current=time, cell_type=np.ones((50, 50, 50)))
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


def test_wet_coverage_excludes_only_open_air_not_enclosed_voids_or_missing_fluid():
    types = np.ones((5, 5, 5), dtype=int)
    types[0, 1, 1] = 0
    types[2, 2, 2] = 0
    vertices = np.array([[0.5, 1.5, 1.5], [2.5, 2.5, 2.5], [3.5, 3.5, 3.5]])
    wet, dry = wet_surface_coverage(vertices[-1:], vertices, 0.1, types, 1.0)
    assert wet == 0.5 and dry == pytest.approx(1 / 3)
    assert wet_surface_coverage(np.empty((0, 3)), vertices, 0.1, types, 1.0)[0] == 0
    assert wet_surface_coverage(vertices[:1], vertices[:1], 0.1, types, 1.0)[0] == 0


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
        GEOTAICHI_QUEUE_RESULT_SUFFIX="repaired_20261011",
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
    assert all("_formal_repaired_20261011" in command for command in commands)
    assert "--impactor bunny" in commands[0]
    assert all("--impactor sphere" not in command for command in commands)
    records = (logroot / "results.tsv").read_text().splitlines()
    assert records[0].split("\t")[1] == "7"
    assert all(row.split("\t")[1] == "0" for row in records[1:])
    assert "complete\tfailures=1" in (logroot / "queue.status").read_text()

    env["GEOTAICHI_QUEUE_RESULT_SUFFIX"] = "../invalid"
    invalid = subprocess.run(
        ["bash", str(ROOT / "research/remote_validation/run_semi_implicit_revision_queue_20261010.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 2
    assert calls.read_text().splitlines() == commands

    fake_df = tmp_path / "df"
    fake_df.write_text(
        '#!/bin/sh\nprintf "Filesystem 1024-blocks Used Available Capacity Mounted\\ntest 2000 1999 1 100%% /\\n"\n'
    )
    fake_df.chmod(0o755)
    env.update(
        GEOTAICHI_QUEUE_RESULT_SUFFIX="low_space",
        GEOTAICHI_QUEUE_LOGROOT=str(tmp_path / "low_space_logs"),
        PATH=f"{tmp_path}:{os.environ['PATH']}",
    )
    low_space = subprocess.run(
        ["bash", str(ROOT / "research/remote_validation/run_semi_implicit_revision_queue_20261010.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert low_space.returncode == 1
    assert calls.read_text().splitlines() == commands
    space_records = (tmp_path / "low_space_logs/queue.status").read_text()
    assert space_records.count("insufficient disk space") == 4
    assert "complete\tfailures=4" in space_records

    env.update(
        PATH=os.environ["PATH"],
        GEOTAICHI_QUEUE_RESULT_SUFFIX="with_utube",
        GEOTAICHI_QUEUE_LOGROOT=str(tmp_path / "utube_logs"),
        GEOTAICHI_QUEUE_INCLUDE_UTUBE="1",
    )
    with_utube = subprocess.run(
        ["bash", str(ROOT / "research/remote_validation/run_semi_implicit_revision_queue_20261010.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert with_utube.returncode == 1
    added = calls.read_text().splitlines()[len(commands) :]
    assert len(added) == 5
    assert added[-2] == "examples/mmpm/UTubeFlow2D/u_tube_flow_2d.py"
    assert "granular_water_leakage_3d.py" in added[-1]


@pytest.mark.skipif(sys.platform != "linux", reason="Running Bash input position is checked through /proc")
def test_queue_pending_tail_can_be_reordered_without_restarting_running_case(tmp_path):
    template = (ROOT / "research/remote_validation/run_semi_implicit_revision_queue_20261010.sh").read_text()
    plate_start = template.index("\nrun_case section_5_1_no_plate")
    utube_start = template.index('\nif [ "${GEOTAICHI_QUEUE_INCLUDE_UTUBE:-0}"')
    end = template.index("\nrecord complete", max(plate_start, utube_start))
    if plate_start < utube_start:
        plate, utube = template[plate_start:utube_start], template[utube_start:end]
    else:
        utube, plate = template[utube_start:plate_start], template[plate_start:end]
    prefix, suffix = template[: min(plate_start, utube_start)], template[end:]
    original = prefix + plate + utube + suffix
    reordered = prefix + utube + plate + suffix
    assert len(original.encode()) == len(reordered.encode())
    script = tmp_path / "queue.sh"
    script.write_text(original)
    inode = script.stat().st_ino
    calls, child_file, release = (tmp_path / name for name in ("calls", "child_pid", "release"))
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$*" >> "{calls}"\n'
        f'if [[ "$*" == *wavemaker_lsdem_particles_3d.py* ]]; then\n'
        f'  printf "%s\\n" "$$" > "{child_file}"\n'
        f'  while [ ! -e "{release}" ]; do sleep 0.02; done\nfi\nexit 0\n'
    )
    fake_python.chmod(0o755)
    repo, logroot = tmp_path / "repo", tmp_path / "logs"
    (repo / "research/remote_validation").mkdir(parents=True)
    env = dict(
        os.environ,
        GEOTAICHI_QUEUE_REPO=str(repo),
        GEOTAICHI_QUEUE_PYTHON=str(fake_python),
        GEOTAICHI_QUEUE_LOGROOT=str(logroot),
        GEOTAICHI_QUEUE_RESULT_SUFFIX="pending_tail",
        GEOTAICHI_QUEUE_INCLUDE_UTUBE="1",
    )
    process = subprocess.Popen(
        ["bash", str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 10
        while not child_file.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_file.exists(), process.poll()
        child_pid = int(child_file.read_text())
        info = Path(f"/proc/{process.pid}/fdinfo/255").read_text()
        position = int(next(line.split()[1] for line in info.splitlines() if line.startswith("pos:")))
        assert position <= len(prefix.encode())
        with script.open("r+") as stream:
            stream.write(reordered)
            stream.truncate()
        assert script.stat().st_ino == inode
        os.kill(child_pid, 0)
        assert process.poll() is None
        release.touch()
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)
        commands = calls.read_text().splitlines()
        assert len(commands) == 5
        assert commands[-2] == "examples/mmpm/UTubeFlow2D/u_tube_flow_2d.py"
        assert "granular_water_leakage_3d.py" in commands[-1]
        assert "complete\tfailures=0" in (logroot / "queue.status").read_text()
    finally:
        release.touch()
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=10)
