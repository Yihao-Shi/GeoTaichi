import time
from pathlib import Path

import pytest


for dependency in ("taichi", "open3d", "meshio", "scipy", "pyevtk"):
    pytest.importorskip(dependency)

from geotaichi_mcp.execution.task_manager import TaskManager


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def wait_for_terminal(manager, task_id, timeout=120.0):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = manager.status(task_id, max_output_chars=20000)
        if latest["task"]["status"] in {"completed", "failed", "interrupted"}:
            return latest
        time.sleep(0.05)
    raise AssertionError("GeoTaichi smoke task did not finish: %s" % latest)


@pytest.mark.integration
@pytest.mark.slow
def test_reduced_mpm_run_handles_live_code_at_a_real_solver_checkpoint(tmp_path):
    script = tmp_path / "reduced_mpm_live.py"
    script.write_text(
        """\
import os

os.environ["GEOTAICHI_FORCE_CPU"] = "1"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ["GEOTAICHI_MCP_TASK_DIR"], "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.environ["GEOTAICHI_MCP_TASK_DIR"], "cache"))

import numpy as np
from geotaichi import MPM, init
from geotaichi_mcp.execution.live import publish

init(dim=3, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
model = MPM(title="MCP reduced MPM", log=False)
model.set_configuration(
    domain=[0.2, 0.2, 0.2],
    gravity=[0.0, 0.0, -9.8],
    alphaPIC=0.0,
    mapping="MUSL",
    shape_function="Linear",
    configuration="ULMPM",
    solver_type="Explicit",
    material_type="Solid",
    visualize=False,
    log=False,
)
model.set_solver(
    solver={
        "Timestep": 1.0e-5,
        "SimulationTime": 1.0e-5,
        "SaveInterval": 1.0e-5,
        "SavePath": os.environ["GEOTAICHI_MCP_OUTPUT_DIR"],
    },
    log=False,
)
model.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 64,
        "max_constraint_number": {
            "max_displacement_constraint": 64,
            "max_velocity_constraint": 64,
        },
    },
    log=False,
)
model.add_material(
    model="LinearElastic",
    material={
        "MaterialID": 1,
        "Density": 1800.0,
        "YoungModulus": 2.0e5,
        "PoissonRatio": 0.3,
    },
)
model.add_element(element={"ElementType": "R8N3D", "ElementSize": [0.1, 0.1, 0.1]})
model.add_region(
    region={
        "Name": "block",
        "Type": "Rectangle",
        "BoundingBoxPoint": [0.05, 0.05, 0.05],
        "BoundingBoxSize": [0.1, 0.1, 0.1],
    }
)
model.add_body(
    body={
        "Template": {
            "RegionName": "block",
            "nParticlesPerCell": 1,
            "BodyID": 0,
            "MaterialID": 1,
            "InitialVelocity": [0.0, 0.0, 0.0],
            "FixVelocity": ["Free", "Free", "Free"],
        }
    }
)
model.add_boundary_condition()
model.select_save_data(particle=True, grid=False, object=False)
publish(model=model)
model.run()

particle_num = int(model.scene.particleNum[0])
positions = model.scene.particle.x.to_numpy()[:particle_num]
print("reduced-mpm-finite=%s particles=%s probe=%s" % (
    bool(np.isfinite(positions).all()), particle_num, getattr(model, "_mcp_probe", None)
), flush=True)
""",
        encoding="utf-8",
    )
    manager = TaskManager(workspace=tmp_path / "tasks", repo_root=REPOSITORY_ROOT)
    submitted = manager.submit(str(script), "reduced MPM live checkpoint smoke")

    live = manager.execute_code_in_task(
        submitted["task_id"],
        "model._mcp_probe = 42\n_geotaichi_mcp_result = {"
        "'particles': int(model.scene.particleNum[0]), "
        "'step': int(model.sims.current_step), 'probe': model._mcp_probe}",
        timeout=120,
    )
    terminal = wait_for_terminal(manager, submitted["task_id"])

    assert live["status"] == "completed", live
    assert live["result"]["particles"] > 0
    assert live["result"]["step"] >= 1
    assert live["result"]["probe"] == 42
    assert terminal["task"]["status"] == "completed", terminal
    assert "reduced-mpm-finite=True" in terminal["stdout"]["text"]
    assert "probe=42" in terminal["stdout"]["text"]
