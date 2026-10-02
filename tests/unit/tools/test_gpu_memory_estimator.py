"""Static GPU-pool estimates used by the GeoTaichi model-building agent."""

import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPOSITORY_ROOT / "agent/geotaichi-model-builder/scripts/estimate_gpu_memory.py"


def _write(tmp_path, source):
    path = tmp_path / "model.py"
    path.write_text(source, encoding="utf-8")
    return path


def _estimate(path, *options):
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            str(path),
            "--platform",
            "linux",
            "--machine",
            "x86_64",
            "--no-nvml",
            *options,
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)["data"]


def test_resolves_project_style_environment_helper_without_executing_script(tmp_path):
    path = _write(
        tmp_path,
        '''\
import os
from geotaichi import init

PREFIX = "GT_CASE_"

def env_float(name, default):
    return float(os.environ.get(PREFIX + name, str(default)))

init(arch="gpu", device_memory_GB=env_float("DEVICE_MEMORY_GB", 3.5))
raise RuntimeError("the estimator must not execute this script")
''',
    )

    result = _estimate(path, "--env", "GT_CASE_DEVICE_MEMORY_GB=7.25")

    assert result["estimated_preallocated_pool_gib"] == 7.25
    assert result["init_calls"][0]["confidence"] == "exact-configured-pool"


def test_converts_device_fraction_when_gpu_capacity_is_known(tmp_path):
    path = _write(
        tmp_path,
        "import taichi as ti\nti.init(arch=ti.cuda, device_memory_fraction=0.375)\n",
    )

    result = _estimate(path, "--gpu-total-gib", "32")

    assert result["estimated_preallocated_pool_gib"] == 12.0
    assert result["init_calls"][0]["confidence"] == "exact-fraction-of-reported-total"


def test_reports_zero_for_cpu_and_taichi_default_for_gpu(tmp_path):
    cpu_path = _write(tmp_path, "from geotaichi import init\ninit(arch='cpu', device_memory_GB=9)\n")
    cpu_result = _estimate(cpu_path, "--env", "GEOTAICHI_FORCE_CPU=1")
    assert cpu_result["estimated_preallocated_pool_gib"] == 0.0

    gpu_path = tmp_path / "gpu.py"
    gpu_path.write_text("from geotaichi import init\ninit(arch='gpu')\n", encoding="utf-8")
    gpu_result = _estimate(gpu_path)
    assert gpu_result["estimated_preallocated_pool_gib"] == 1.0
    assert gpu_result["init_calls"][0]["configured_source"] == "Taichi 1.7 default"


def test_follows_reachable_main_and_reports_multiple_conditional_initializers(tmp_path):
    path = _write(
        tmp_path,
        '''\
from geotaichi import init

def unused():
    init(arch="gpu", device_memory_GB=99)

def main(arch="gpu"):
    if arch == "gpu":
        init(arch=arch, device_memory_GB=5)
    else:
        init(arch=arch, device_memory_GB=2)

if __name__ == "__main__":
    main()
''',
    )

    result = _estimate(path)

    assert result["estimated_preallocated_pool_gib"] == 5.0
    assert [item["line"] for item in result["init_calls"]] == [8]
