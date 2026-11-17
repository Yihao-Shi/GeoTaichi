"""Contract-driven MPM lifecycle template.

Copy this file into the requested example location. Populate the contract from
current source and a nearby maintained example before running it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_contract(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def publish_live_namespace(**values: object) -> None:
    try:
        from geotaichi_mcp.execution.live import publish
    except ImportError:
        return
    publish(**values)


def apply_import_environment(contract: dict) -> None:
    for name, value in contract["api"].get("environment", {}).items():
        os.environ[str(name)] = str(value)


def task_runtime(api: dict) -> dict:
    runtime = dict(api["runtime"])
    if os.environ.get("GEOTAICHI_MCP_TASK_ID"):
        runtime.setdefault("log", False)
    return runtime


def task_solver(api: dict, output_directory: str | None = None) -> dict:
    solver = dict(api["solver"])
    if output_directory:
        solver.setdefault("SavePath", output_directory)
    return solver


def build_and_run(contract: dict, output_directory: str | None = None) -> None:
    apply_import_environment(contract)
    from geotaichi import MPM, init

    api = contract["api"]
    init(**task_runtime(api))
    model = MPM(title=contract["title"])
    model.set_configuration(**api["configuration"])
    if api.get("implicit_solver"):
        model.set_implicit_solver_parameters(**api["implicit_solver"])
    if api.get("semi_implicit_solver"):
        model.set_semi_implicit_solver_parameters(api["semi_implicit_solver"])
    model.set_solver(task_solver(api, output_directory))
    model.memory_allocate(api["memory"])

    for entry in api.get("materials", []):
        model.add_material(model=entry["model"], material=entry["material"])
    for element in api.get("elements", []):
        model.add_element(element)
    for region in api.get("regions", []):
        model.add_region(region)
    for body in api.get("bodies", []):
        model.add_body(body)
    if api.get("boundaries"):
        model.add_boundary_condition(api["boundaries"])

    model.select_save_data(**api.get("save_data", {}))
    publish_live_namespace(model=model, contract=contract)
    model.run(**api.get("run", {}))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, help="Model contract JSON")
    parser.add_argument("--scene-manifest", help="Optional staged SceneManifest JSON")
    parser.add_argument("--output-dir", help="Override solver SavePath")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    build_and_run(load_contract(arguments.contract), arguments.output_dir)
