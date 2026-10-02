"""Contract-driven Lagrangian MPDEM lifecycle template.

This template is intentionally distinct from the generic coupled template so
an agent selecting ``module="mpdem"`` has an unambiguous output asset.
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
    if contract.get("module") != "mpdem":
        raise ValueError("mpdem_model_template.py requires contract.module='mpdem'")

    api = contract["api"]
    for name, value in api.get("environment", {}).items():
        os.environ[str(name)] = str(value)

    from geotaichi import DEMPM, init

    configuration = dict(api["configuration"])
    configuration.setdefault("coupling_scheme", "MPDEM")
    if configuration["coupling_scheme"] != "MPDEM":
        raise ValueError("The MPDEM template requires coupling_scheme='MPDEM'")

    init(**task_runtime(api))
    model = DEMPM(title=contract["title"], coupling="Lagrangian")
    model.set_configuration(**configuration)
    model.dem.set_configuration(**api["dem"]["configuration"])
    model.mpm.set_configuration(**api["mpm"]["configuration"])
    model.set_solver(task_solver(api, output_directory))
    model.dem.memory_allocate(api["dem"]["memory"])
    model.mpm.memory_allocate(api["mpm"]["memory"])
    model.memory_allocate(api["memory"])

    for entry in api["dem"].get("attributes", []):
        model.dem.add_attribute(entry["material_id"], entry["attribute"])
    for template in api["dem"].get("templates", []):
        model.dem.add_template(template)
    for region in api["dem"].get("regions", []):
        model.dem.add_region(region)
    for body in api["dem"].get("bodies", []):
        model.dem.create_body(body)
    for wall in api["dem"].get("walls", []):
        model.dem.add_wall(wall)

    for entry in api["mpm"].get("materials", []):
        model.mpm.add_material(model=entry["model"], material=entry["material"])
    for element in api["mpm"].get("elements", []):
        model.mpm.add_element(element)
    for region in api["mpm"].get("regions", []):
        model.mpm.add_region(region)
    for body in api["mpm"].get("bodies", []):
        model.mpm.add_body(body)
    if api["mpm"].get("boundaries"):
        model.mpm.add_boundary_condition(api["mpm"]["boundaries"])

    model.choose_contact_model(**api["contact_models"])
    for pair in api.get("contact_properties", []):
        model.add_property(
            pair["dem_material"],
            pair["mpm_material"],
            pair["property"],
            pair.get("d_type", "all"),
        )

    model.dem.select_save_data(**api["dem"].get("save_data", {}))
    model.mpm.select_save_data(**api["mpm"].get("save_data", {}))
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
