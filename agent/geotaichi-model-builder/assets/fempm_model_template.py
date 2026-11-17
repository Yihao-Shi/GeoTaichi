"""Contract-driven explicit FEMPM lifecycle template."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path


def load_contract(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _build_fem_problem(contract, fem):
    factory = contract["api"]["fem"].get("problem_factory")
    if not factory:
        raise ValueError("api.fem.problem_factory is required for FEMPM")
    module = importlib.import_module(factory["module"])
    function = getattr(module, factory.get("function", "build_fem_problem"))
    function(contract, fem)


def build_and_run(contract: dict, output_directory: str | None = None) -> None:
    if contract.get("module") != "fempm":
        raise ValueError(
            "fempm_model_template.py requires contract.module='fempm'"
        )
    api = contract["api"]
    for name, value in api.get("environment", {}).items():
        os.environ[str(name)] = str(value)

    from geotaichi import FEMPM, init

    init(**api["runtime"])
    model = FEMPM(title=contract["title"], log=api.get("log", True))
    model.mpm.set_configuration(**api["mpm"]["configuration"])
    model.mpm.memory_allocate(api["mpm"]["memory"])
    for entry in api["mpm"].get("materials", []):
        model.mpm.add_material(**entry)
    model.mpm.add_element(api["mpm"]["element"])
    for region in api["mpm"].get("regions", []):
        model.mpm.add_region(region)
    for body in api["mpm"].get("bodies", []):
        model.mpm.add_body(body)
    if api["mpm"].get("boundaries"):
        model.mpm.add_boundary_condition(api["mpm"]["boundaries"])

    model.fem.set_configuration(**api["fem"]["configuration"])
    _build_fem_problem(contract, model.fem)
    model.set_configuration(**api["configuration"])
    solver = dict(api["solver"])
    if output_directory:
        solver.setdefault("SavePath", output_directory)
    model.set_solver(solver)
    model.add_surface(**api.get("surface", {}))
    model.memory_allocate(api["memory"])
    model.choose_contact_model(api["contact_model"])
    for pair in api.get("contact_properties", []):
        model.add_property(
            pair["mpm_material"], pair["fem_body"], pair["property"]
        )
    model.select_save_data(**api.get("save_data", {}))
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
