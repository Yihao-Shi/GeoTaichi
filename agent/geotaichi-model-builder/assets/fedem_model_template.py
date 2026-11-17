"""Contract-driven FEDEM/LSDEM/AffineBody coupling lifecycle template."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_contract(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def build_and_run(contract: dict, output_directory: str | None = None) -> None:
    if contract.get("module") != "fedem":
        raise ValueError(
            "fedem_model_template.py requires contract.module='fedem'"
        )
    api = contract["api"]
    for name, value in api.get("environment", {}).items():
        os.environ[str(name)] = str(value)

    from geotaichi import FEDEM, init

    init(**api["runtime"])
    model = FEDEM(title=contract["title"], log=api.get("log", True))
    model.dem.set_configuration(**api["dem"]["configuration"])
    model.dem.memory_allocate(api["dem"]["memory"])
    for entry in api["dem"].get("attributes", []):
        model.dem.add_attribute(entry["material_id"], entry["attribute"])
    for body in api["dem"].get("bodies", []):
        model.dem.create_body(body)
    model.dem.choose_contact_model(**api["dem"].get("contact_models", {}))

    model.fem.set_configuration(**api["fem"]["configuration"])
    if api["fem"].get("soft_particles"):
        for particle in api["fem"]["soft_particles"]:
            if isinstance(particle, dict) and "mesh" in particle:
                model.fem.add_soft_particle(
                    particle["mesh"], **particle.get("mesh_options", {})
                )
            else:
                model.fem.add_soft_particle(particle)
    else:
        model.fem.add_mesh(api["fem"]["mesh"])
    model.fem.add_material(**api["fem"]["material"])
    if api["fem"].get("boundaries"):
        model.fem.add_boundary_condition(api["fem"]["boundaries"])

    model.set_configuration(**api["configuration"])
    solver = dict(api["solver"])
    if output_directory:
        solver.setdefault("SavePath", output_directory)
    model.set_solver(solver)
    model.add_surface(**api.get("surface", {}))
    model.memory_allocate(api["memory"])
    model.choose_contact_model(
        api["contact_model"], **api.get("contact_parameters", {})
    )
    ipc = str(api["contact_model"]).replace("_", "").lower() == "ipc"
    for pair in api.get("contact_properties", []):
        if ipc:
            model.add_ipc_property(
                pair["affine_body"], pair["fem_body"], pair["property"]
            )
        else:
            model.add_property(
                pair["dem_material"], pair["fem_body"], pair["property"]
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
