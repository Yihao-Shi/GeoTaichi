"""Contract-driven pure IGA lifecycle template.

IGA geometry, primitive collections, and boundary objects are Python objects,
not JSON values. The contract therefore names a user model-factory module.
Its factory receives the full contract and returns this mapping::

    {
        "primitives": primitives,
        "degree": [2, 2],
        "material": {"young_modulus": 1.0e5, ...},
        "boundary": {"dirichlet": dirichlet, "neumann": neumann},
    }
"""

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


def task_runtime(api: dict) -> dict:
    runtime = dict(api["runtime"])
    if os.environ.get("GEOTAICHI_MCP_TASK_ID"):
        runtime.setdefault("log", False)
    return runtime


def load_problem_factory(specification: dict) -> Callable[[dict], dict]:
    module_name = specification.get("module")
    if not module_name:
        raise ValueError("api.problem_factory.module is required")
    function_name = specification.get("function", "build_iga_problem")
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise TypeError(f"{module_name}.{function_name} must be callable")
    return factory


def validate_problem(problem: object) -> dict:
    if not isinstance(problem, dict):
        raise TypeError("The IGA problem factory must return a dictionary")
    missing = [name for name in ("primitives", "degree", "material") if name not in problem]
    if missing:
        raise ValueError(f"The IGA problem factory omitted: {', '.join(missing)}")
    if not isinstance(problem["material"], dict):
        raise TypeError("IGA problem.material must be a dictionary")
    if not isinstance(problem.get("boundary", {}), dict):
        raise TypeError("IGA problem.boundary must be a dictionary")
    return problem


def build_and_run(contract: dict, output_directory: str | None = None) -> None:
    if contract.get("module") != "iga":
        raise ValueError("iga_model_template.py requires contract.module='iga'")

    api = contract["api"]
    for name, value in api.get("environment", {}).items():
        os.environ[str(name)] = str(value)

    from geotaichi import IGA, init

    init(**task_runtime(api))
    problem = validate_problem(load_problem_factory(api["problem_factory"])(contract))

    model = IGA(title=contract["title"])
    model.set_configuration(**api["configuration"])
    model.add_primitives(problem["primitives"])
    model.add_boundary_condition(**problem.get("boundary", {}))
    model.add_element(problem["degree"])
    model.add_material(**problem["material"])
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
