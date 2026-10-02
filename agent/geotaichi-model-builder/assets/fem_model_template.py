"""Contract-driven volume or cloth FEM lifecycle template."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Callable


def load_contract(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def publish_live_namespace(**values: object) -> None:
    try:
        from geotaichi_mcp.execution.live import publish
    except ImportError:
        return
    publish(**values)


def load_problem_factory(specification: dict) -> Callable[[dict], dict]:
    module_name = specification.get("module")
    if not module_name:
        raise ValueError("api.problem_factory.module is required")
    function_name = specification.get("function", "build_fem_problem")
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise TypeError(f"{module_name}.{function_name} must be callable")
    return factory


def build_and_run(contract: dict, output_directory: str | None = None) -> None:
    if contract.get("module") != "fem":
        raise ValueError("fem_model_template.py requires contract.module='fem'")

    api = contract["api"]
    for name, value in api.get("environment", {}).items():
        os.environ[str(name)] = str(value)

    from geotaichi import FEM, init

    runtime = dict(api["runtime"])
    if os.environ.get("GEOTAICHI_MCP_TASK_ID"):
        runtime.setdefault("log", False)
    init(**runtime)
    problem = load_problem_factory(api["problem_factory"])(contract)
    if (
        not isinstance(problem, dict)
        or ("mesh" not in problem and not problem.get("soft_particles"))
        or "material" not in problem
    ):
        raise TypeError(
            "FEM problem factory must return material and either mesh or soft_particles"
        )

    model = FEM(title=contract["title"])
    model.set_configuration(**api["configuration"])
    soft_particles = problem.get("soft_particles")
    if soft_particles:
        for particle in soft_particles:
            if isinstance(particle, dict) and "mesh" in particle:
                options = dict(particle.get("mesh_options", {}))
                model.add_soft_particle(particle["mesh"], **options)
            else:
                model.add_soft_particle(particle)
    else:
        model.add_mesh(problem["mesh"], **problem.get("mesh_options", {}))
    material = problem["material"]
    if isinstance(material, dict):
        model.add_material(**material)
    else:
        model.add_material(material=material)
    if problem.get("boundary"):
        model.add_boundary_condition(**problem["boundary"])
    if problem.get("bending_model") is not None:
        model.set_bending_model(problem["bending_model"])
    for energy in problem.get("cloth_energies", []):
        model.add_cloth_energy(energy)
    if problem.get("contact"):
        contact = dict(problem["contact"])
        model.add_contact(contact.pop("model", "IPC"), **contact)
    if problem.get("soft_particle_contact"):
        contact = dict(problem["soft_particle_contact"])
        pairs = contact.pop("pairs", [])
        model.add_soft_particle_contact(
            contact.pop("model", "Linear"), **contact
        )
        for pair in pairs:
            model.add_soft_particle_property(
                pair["body1"], pair["body2"], pair.get("property", {})
            )
    solver = dict(api["solver"])
    if output_directory:
        solver.setdefault("path", output_directory)
    model.set_solver(**solver)
    publish_live_namespace(model=model, problem=problem, contract=contract)
    model.run(**api.get("run", {}))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, help="Model contract JSON")
    parser.add_argument("--scene-manifest", help="Optional staged SceneManifest JSON")
    parser.add_argument("--output-dir", help="Override solver output path")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    build_and_run(load_contract(arguments.contract), arguments.output_dir)
