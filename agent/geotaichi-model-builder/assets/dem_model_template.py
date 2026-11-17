"""Contract-driven DEM/LSDEM/LSMPM/AffineBody lifecycle template."""

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
    api = contract["api"]
    for name, value in api.get("environment", {}).items():
        os.environ[str(name)] = str(value)

    from geotaichi import DEM, init

    init(**task_runtime(api))
    model = DEM(title=contract["title"])
    model.set_configuration(**api["configuration"])
    model.set_solver(task_solver(api, output_directory))

    for template in api.get("preallocation_templates", []):
        model.add_template(template)
    preprocessing = []
    for request in api.get("soft_grid_preprocessing", []):
        preprocessing.append(model.preprocess_soft_grid_template(**request))
    model.memory_allocate(api["memory"])

    for entry in api.get("attributes", []):
        model.add_attribute(entry["material_id"], entry["attribute"])
    for template in api.get("templates", []):
        model.add_template(template)
    for region in api.get("regions", []):
        model.add_region(region)
    for body in api.get("bodies", []):
        model.create_body(body)
    for wall in api.get("walls", []):
        model.add_wall(wall)

    if api.get("contact_models"):
        model.choose_contact_model(**api["contact_models"])
    for pair in api.get("contact_properties", []):
        model.add_property(
            pair["material_id_1"],
            pair["material_id_2"],
            pair["property"],
            pair.get("d_type", "all"),
        )
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
