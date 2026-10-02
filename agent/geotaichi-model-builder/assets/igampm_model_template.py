"""Contract-driven IGA-MPM lifecycle template."""

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


def task_solver(solver: dict, output_directory: str | None = None) -> dict:
    resolved = dict(solver)
    if output_directory:
        resolved.setdefault("SavePath", output_directory)
    return resolved


def build_and_run(contract: dict, output_directory: str | None = None) -> None:
    api = contract["api"]
    for name, value in api.get("environment", {}).items():
        os.environ[str(name)] = str(value)

    from geotaichi import IGA, IGAMPM, MPM, init

    init(**task_runtime(api))
    iga = IGA(log=False)
    mpm = MPM(log=False)
    contact = dict(api.get("contact_model", {}))
    contact_name = contact.get(
        "contact_model",
        api.get("configuration", {}).get("contact_model", "IPC"),
    )
    # Explicit contact must select ParticleCoupling before MPM allocation.
    model = IGAMPM(
        iga=iga,
        mpm=mpm,
        title=contract["title"],
        contact_model=contact_name,
    )

    iga.set_configuration(**api["iga"]["configuration"])
    iga.add_primitives(api["iga"]["primitives"])
    iga.add_material(**api["iga"]["material"])
    iga.add_element(api["iga"]["degree"])
    iga.add_boundary_condition(**api["iga"].get("boundary", {}))
    iga.set_solver(**api["iga"].get("solver", {}))

    mpm.set_configuration(**api["mpm"]["configuration"])
    if api["mpm"].get("implicit_solver"):
        mpm.set_implicit_solver_parameters(**api["mpm"]["implicit_solver"])
    mpm.set_solver(task_solver(api["mpm"]["solver"], output_directory))
    mpm.memory_allocate(api["mpm"]["memory"])
    for entry in api["mpm"].get("materials", []):
        mpm.add_material(model=entry["model"], material=entry["material"])
    for element in api["mpm"].get("elements", []):
        mpm.add_element(element)
    for region in api["mpm"].get("regions", []):
        mpm.add_region(region)
    for body in api["mpm"].get("bodies", []):
        mpm.add_body(body)
    if api["mpm"].get("boundaries"):
        mpm.add_boundary_condition(api["mpm"]["boundaries"])

    model.set_configuration(**api["configuration"])
    model.choose_contact_model(**contact)
    for entry in api.get("properties", []):
        model.add_property(
            MPMmaterial=entry["MPMmaterial"],
            IGAbody=entry["IGAbody"],
            property=entry["property"],
        )
    model.build()
    publish_live_namespace(model=model, iga=iga, mpm=mpm, contract=contract)
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
