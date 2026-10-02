#!/usr/bin/env python3
"""Run dependency-free self-tests for the GeoTaichi model-builder tools."""

from __future__ import annotations

import tempfile
from pathlib import Path

from audit_api_docs import DEFAULT_DOC, audit
from build_capability_index import build_index
from common import build_error, build_ok, emit, find_repo_root, load_json
from estimate_gpu_memory import estimate_script
from inspect_example import inspect_example
from query_capabilities import browse, query
from validate_model_contract import validate_contract

from geotaichi_mcp.core.resources import load_physics_validation_rubric
from geotaichi_mcp.knowledge.physics_validation import score_physics_validation


SMOKE_EXAMPLE = '''\
import os

os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"

from geotaichi import MPM, init

init(dim=2, arch="cpu", default_fp="float64", log=False)
model = MPM(log=False)
model.set_configuration({"domain": [1.0, 1.0]})
model.set_solver({"Timestep": 1.0e-4, "SimulationTime": 1.0e-3, "SaveInterval": 1.0e-3})
model.memory_allocate({"max_material_number": 1, "max_particle_number": 100})
model.run()
'''

MEMORY_EXAMPLE = '''\
import os

from geotaichi import init

PREFIX = "GT_MEMORY_TEST_"

def env_float(name, default):
    return float(os.environ.get(PREFIX + name, str(default)))

init(arch=os.environ.get(PREFIX + "ARCH", "gpu"), device_memory_GB=env_float("DEVICE_GIB", 4.5))
'''

FRACTION_EXAMPLE = '''\
import taichi as ti

ti.init(arch=ti.cuda, device_memory_fraction=0.25)
'''


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(repo: Path) -> dict:
    checks: list[dict[str, str]] = []
    index = build_index(repo)
    require(
        set(index["categories"])
        == {
            "mpm",
            "dem",
            "mpdem",
            "cfdem",
            "fem",
            "fedem",
            "fempm",
            "iga",
            "igampm",
        },
        "category set",
    )
    require(all(item["summary"]["method_count"] > 0 for item in index["categories"].values()), "method extraction")
    fedem_keys = {
        item["name"]: item for item in index["categories"]["fedem"]["configuration_keys"]
    }
    require("max_levelset_cell_pairs" in fedem_keys, "mapping.get configuration extraction")
    require(
        any(
            location["source"] == "src/fedem/Simulation.py"
            and location["reader"] == "get"
            for location in fedem_keys["max_levelset_cell_pairs"]["locations"]
        ),
        "mapping.get configuration source",
    )
    checks.append({"name": "index_build", "status": "passed"})

    browse_result = browse(index, "mpm/methods/set_configuration")
    require(browse_result["ok"], "known browse path")
    require(browse_result["data"]["entries"][0]["source"] == "src/mpm/mainMPM.py", "browse source")
    query_result = query(index, "soft levelset advection", 10)
    require(query_result["ok"] and query_result["data"]["entries"], "capability query")
    natural_query = query(index, "deformable soft particles colliding with rigid level-set bodies", 10)
    require(
        natural_query["ok"] and natural_query["data"]["entries"][0]["path"] == "fedem",
        "natural-language coupling route",
    )
    checks.append({"name": "browse_and_query", "status": "passed"})

    contract_path = repo / "agent/geotaichi-model-builder/assets/model-contract.json"
    contract_result = validate_contract(load_json(contract_path))
    require(contract_result["valid"], "bundled model contract")
    checks.append({"name": "contract_validation", "status": "passed"})

    physics_score = score_physics_validation(
        {
            "module": "mpm",
            "unresolved": [],
            "validation": {
                "observable": "mass drift",
                "expectation": 0.0,
                "tolerance": {"absolute": 1.0e-6},
                "invariants": [
                    {
                        "kind": "mass_balance",
                        "expectation": 0.0,
                        "tolerance": {"absolute": 1.0e-6},
                        "basis": "self-test analytical input",
                    }
                ],
            },
        },
        {
            "task_status": "completed",
            "finite_state": True,
            "capacity_overflow": False,
            "solver_converged": "not_applicable",
            "timestep_consistent": True,
            "command": "python reduced_model.py",
            "backend": "cpu",
            "precision": "float64",
            "production_parameters": False,
            "checks": [
                {
                    "name": "mass drift observable",
                    "kind": "contract_observable",
                    "observed": 1.0e-8,
                    "expected": 0.0,
                    "tolerance": {"absolute": 1.0e-6},
                    "evidence": "self-test analytical input",
                },
                {
                    "name": "mass balance",
                    "kind": "mass_balance",
                    "observed": 1.0e-8,
                    "expected": 0.0,
                    "tolerance": {"absolute": 1.0e-6},
                    "evidence": "self-test analytical input",
                },
            ],
        },
        load_physics_validation_rubric(repo),
    )
    require(physics_score["decision"] == "accept_reduced", "physics evidence scorer")
    checks.append({"name": "physics_evidence_scorer", "status": "passed"})

    with tempfile.TemporaryDirectory(prefix="geotaichi_model_builder_") as temporary:
        example_path = Path(temporary) / "model.py"
        example_path.write_text(SMOKE_EXAMPLE, encoding="utf-8")
        inspection = inspect_example(example_path, index)
    require(inspection["valid"], f"static example inspection: {inspection['errors']}")
    checks.append({"name": "example_inspection", "status": "passed"})

    with tempfile.TemporaryDirectory(prefix="geotaichi_memory_estimator_") as temporary:
        memory_path = Path(temporary) / "memory_model.py"
        memory_path.write_text(MEMORY_EXAMPLE, encoding="utf-8")
        memory_result = estimate_script(
            memory_path,
            environment={"GT_MEMORY_TEST_DEVICE_GIB": "6.25"},
            target_platform="Linux",
            target_machine="x86_64",
        )
        require(memory_result["estimated_preallocated_pool_gib"] == 6.25, "environment-backed GPU pool")
        require(memory_result["init_calls"][0]["confidence"] == "exact-configured-pool", "GPU pool confidence")

        fraction_path = Path(temporary) / "fraction_model.py"
        fraction_path.write_text(FRACTION_EXAMPLE, encoding="utf-8")
        fraction_result = estimate_script(
            fraction_path,
            environment={},
            target_platform="Linux",
            target_machine="x86_64",
            gpu_total_gib=24.0,
        )
        require(fraction_result["estimated_preallocated_pool_gib"] == 6.0, "fractional GPU pool")
    checks.append({"name": "gpu_memory_estimator", "status": "passed"})

    asset_root = repo / "agent/geotaichi-model-builder/assets"
    required_templates = {
        "mpm_model_template.py",
        "dem_model_template.py",
        "mpdem_model_template.py",
        "fem_model_template.py",
        "fedem_model_template.py",
        "fempm_model_template.py",
        "iga_model_template.py",
        "igampm_model_template.py",
    }
    available_templates = {
        path.name
        for path in asset_root.glob("*_model_template.py")
        if not path.name.startswith("._")
    }
    require(required_templates <= available_templates, "facade template coverage")
    for template_name in sorted(available_templates):
        template_result = inspect_example(asset_root / template_name, index)
        require(template_result["valid"], f"{template_name}: {template_result['errors']}")
    checks.append({"name": "asset_template_inspection", "status": "passed"})

    docs_result = audit(repo, repo / DEFAULT_DOC, list(index["categories"]), "methods")
    require(docs_result["ok"], "public facade method documentation coverage")
    checks.append({"name": "facade_method_docs", "status": "passed"})

    return {"checks": checks, "summary": {"passed": len(checks), "failed": 0}}


def main() -> int:
    try:
        payload = build_ok(run(find_repo_root()))
        emit(payload)
        return 0
    except Exception as exc:
        emit(build_error("self_test_failed", str(exc)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
