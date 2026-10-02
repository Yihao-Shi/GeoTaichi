#!/usr/bin/env python3
"""Build a hierarchical GeoTaichi facade and configuration capability index."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import build_error, build_ok, emit, find_repo_root, write_json


CATEGORIES: dict[str, dict[str, Any]] = {
    "mpm": {
        "facade": "MPM",
        "description": "Material point continuum solid, fluid, porous, and direct-backend workflows.",
        "facade_source": "src/mpm/mainMPM.py",
        "key_sources": [
            "src/mpm/mainMPM.py",
            "src/mpm/Simulation.py",
            "src/mpm/GenerateManager.py",
            "src/mpm/MaterialManager.py",
            "src/mpm/generator/*.py",
            "src/mpm/boundaries/*.py",
            "src/physics_model/consititutive_model/**/*.py",
        ],
        "example_roots": ["examples/mpm"],
        "reference": "references/workflow-mpm.md",
        "related": ["mpdem", "cfdem", "igampm", "fempm"],
    },
    "dem": {
        "facade": "DEM",
        "description": "Discrete particles, level sets, AffineBody, and LSMPM soft-body workflows.",
        "facade_source": "src/dem/mainDEM.py",
        "key_sources": [
            "src/dem/mainDEM.py",
            "src/dem/Simulation.py",
            "src/dem/GenerateManager.py",
            "src/dem/generator/*.py",
            "src/dem/contact/*.py",
            "src/physics_model/contact_model/*.py",
        ],
        "example_roots": ["examples/dem"],
        "reference": "references/workflow-dem.md",
        "related": ["mpdem", "cfdem", "fedem"],
    },
    "mpdem": {
        "facade": "DEMPM",
        "description": "Lagrangian DEM-MPM contact and coupled orchestration.",
        "facade_source": "src/mpdem/mainDEMPM.py",
        "key_sources": [
            "src/mpdem/mainDEMPM.py",
            "src/mpdem/Simulation.py",
            "src/mpdem/contact/*.py",
            "src/mpdem/engines/*.py",
            "src/physics_model/contact_model/*.py",
        ],
        "example_roots": ["examples/mpdem"],
        "reference": "references/workflow-coupling.md",
        "related": ["mpm", "dem", "cfdem"],
    },
    "cfdem": {
        "facade": "DEMPM",
        "description": "Fluid-particle coupling routed through the DEMPM facade.",
        "facade_source": "src/mpdem/mainDEMPM.py",
        "key_sources": [
            "src/mpdem/mainDEMPM.py",
            "src/mpdem/Simulation.py",
            "src/mpdem/fluid_dynamics/*.py",
            "src/mpdem/engines/*.py",
        ],
        "example_roots": ["examples/cfdem"],
        "reference": "references/workflow-coupling.md",
        "related": ["mpm", "dem", "mpdem"],
    },
    "fem": {
        "facade": "FEM",
        "description": "Volume and cloth finite elements with explicit/implicit solvers and contact.",
        "facade_source": "src/fem/mainFEM.py",
        "key_sources": [
            "src/fem/mainFEM.py",
            "src/fem/Simulation.py",
            "src/fem/MaterialManager.py",
            "src/fem/generator/*.py",
            "src/fem/boundaries/*.py",
            "src/fem/contact/*.py",
            "src/fem/cloth/*.py",
            "src/fem/engines/*.py",
            "src/physics_model/consititutive_model/finite_strain/*.py",
            "src/physics_model/consititutive_model/infinitesimal_strain/*.py",
        ],
        "example_roots": ["examples/fem"],
        "reference": "references/workflow-fem.md",
        "related": ["iga", "fedem", "fempm"],
    },
    "fedem": {
        "facade": "FEDEM",
        "description": "Explicit DEM/LSDEM penalty contact or implicit AffineBody IPC with deforming FEM boundaries.",
        "facade_source": "src/fedem/mainFEDEM.py",
        "key_sources": [
            "src/fedem/*.py",
            "src/fedem/contact/*.py",
            "src/fedem/neighbor/*.py",
            "src/fedem/structs/*.py",
        ],
        "example_roots": ["examples/fedem"],
        "reference": "references/workflow-coupling.md",
        "related": ["fem", "dem"],
    },
    "fempm": {
        "facade": "FEMPM",
        "description": "Three-dimensional explicit DEM-law contact or 2D, axisymmetric, and 3D monolithic IPC between MPM material points and deforming FEM boundaries.",
        "facade_source": "src/fempm/mainFEMPM.py",
        "key_sources": [
            "src/fempm/*.py",
            "src/fempm/contact/*.py",
            "src/fempm/neighbor/*.py",
            "src/fempm/structs/*.py",
        ],
        "example_roots": ["examples/fempm"],
        "reference": "references/workflow-coupling.md",
        "related": ["fem", "mpm", "fedem", "mpdem"],
    },
    "iga": {
        "facade": "IGA",
        "description": "NURBS/isogeometric analysis workflows.",
        "facade_source": "src/iga/mainIGA.py",
        "key_sources": ["src/iga/mainIGA.py", "src/iga/*.py"],
        "example_roots": ["examples/iga"],
        "reference": "references/workflow-iga.md",
        "related": ["igampm"],
    },
    "igampm": {
        "facade": "IGAMPM",
        "description": (
            "Three-dimensional explicit shared-DEM-law point--NURBS contact, "
            "or Cartesian 2D/3D and axisymmetric monolithic implicit IPC, "
            "between elastic IGA and MPM."
        ),
        "facade_source": "src/igampm/mainIGAMPM.py",
        "key_sources": [
            "src/igampm/*.py",
            "src/igampm/contact/*.py",
            "src/igampm/engines/*.py",
        ],
        "example_roots": ["examples/igampm"],
        "reference": "references/workflow-iga.md",
        "related": ["iga", "mpm"],
    },
}

DICT_READERS = {"GetEssential", "GetAlternative", "GetOptional", "GetSwitch"}
MAPPING_READERS = {"get"}
CONFIGURATION_MAPPING_NAMES = {
    "coupling_parameters",
    "implicit_parameters",
    "kwargs",
    "memory",
    "parameters",
    "semi_implicit_parameters",
    "solver",
}


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a compact source signature without importing GeoTaichi."""
    arguments: list[str] = []
    positional = [*node.args.posonlyargs, *node.args.args]
    default_offset = len(positional) - len(node.args.defaults)
    for index, argument in enumerate(positional):
        value = argument.arg
        if index >= default_offset:
            default = node.args.defaults[index - default_offset]
            value += f"={ast.unparse(default)}"
        arguments.append(value)
    if node.args.vararg:
        arguments.append(f"*{node.args.vararg.arg}")
    elif node.args.kwonlyargs:
        arguments.append("*")
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        value = argument.arg
        if default is not None:
            value += f"={ast.unparse(default)}"
        arguments.append(value)
    if node.args.kwarg:
        arguments.append(f"**{node.args.kwarg.arg}")
    return f"{node.name}({', '.join(arguments)})"


def _facade_methods(repo: Path, relative_path: str, class_name: str) -> list[dict[str, Any]]:
    path = repo / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            methods = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                    methods.append(
                        {
                            "name": child.name,
                            "signature": _signature(child),
                            "source": relative_path,
                            "line": child.lineno,
                            "doc": ast.get_docstring(child) or "",
                        }
                    )
            inherited_names = {
                base.id
                for base in node.bases
                if isinstance(base, ast.Name)
            }
            if "SolverDiagnosticsMixin" in inherited_names:
                methods.append(
                    {
                        "name": "diagnostics_snapshot",
                        "signature": "diagnostics_snapshot(self)",
                        "source": "src/utils/SolverDiagnostics.py",
                        "line": 111,
                        "doc": "Return the common JSON-friendly solver diagnostics snapshot.",
                    }
                )
            return sorted(methods, key=lambda item: item["name"].lower())
    raise RuntimeError(f"Facade class {class_name!r} was not found in {relative_path}")


def _expand_sources(repo: Path, patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            paths.update(
                path
                for path in repo.glob(pattern)
                if path.is_file() and not path.name.startswith("._")
            )
        else:
            path = repo / pattern
            if path.is_file():
                paths.add(path)
    return sorted(paths)


def _configuration_keys(repo: Path, patterns: Iterable[str]) -> list[dict[str, Any]]:
    locations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in _expand_sources(repo, patterns):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in DICT_READERS and len(node.args) >= 2:
                key_node = node.args[1]
            elif (
                node.func.attr in MAPPING_READERS
                and len(node.args) >= 1
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in CONFIGURATION_MAPPING_NAMES
            ):
                key_node = node.args[0]
            else:
                continue
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                relative = path.relative_to(repo).as_posix()
                record = {"source": relative, "line": node.lineno, "reader": node.func.attr}
                if record not in locations[key_node.value]:
                    locations[key_node.value].append(record)
    return [
        {"name": name, "locations": sorted(records, key=lambda item: (item["source"], item["line"]))}
        for name, records in sorted(locations.items(), key=lambda item: item[0].lower())
    ]


def _examples(repo: Path, roots: Iterable[str]) -> list[str]:
    examples: list[str] = []
    for root in roots:
        path = repo / root
        if path.is_dir():
            examples.extend(
                item.relative_to(repo).as_posix()
                for item in path.rglob("*.py")
                if not item.name.startswith("._")
            )
    return sorted(set(examples))


def build_index(repo: Path) -> dict[str, Any]:
    """Build the complete capability index from the working tree."""
    categories: dict[str, Any] = {}
    method_ref: dict[str, list[str]] = defaultdict(list)
    key_ref: dict[str, list[str]] = defaultdict(list)

    for name, metadata in CATEGORIES.items():
        methods = _facade_methods(repo, metadata["facade_source"], metadata["facade"])
        keys = _configuration_keys(repo, metadata["key_sources"])
        examples = _examples(repo, metadata["example_roots"])
        category = {
            "facade": metadata["facade"],
            "description": metadata["description"],
            "facade_source": metadata["facade_source"],
            "reference": metadata["reference"],
            "related": metadata["related"],
            "public_methods": methods,
            "configuration_keys": keys,
            "examples": examples,
            "summary": {
                "method_count": len(methods),
                "configuration_key_count": len(keys),
                "example_count": len(examples),
            },
        }
        categories[name] = category
        for method in methods:
            method_ref[method["name"]].append(f"{name}/methods/{method['name']}")
        for key in keys:
            key_ref[key["name"]].append(f"{name}/keys/{key['name']}")

    return {
        "schema_version": 1,
        "description": "Working-tree GeoTaichi capability locator; source remains authoritative.",
        "navigation": {
            "root": "List facade categories",
            "category": "Show one facade overview",
            "methods": "List public facade methods",
            "method": "Show one method signature and source",
            "keys": "List configuration keys consumed through DictIO or recognized configuration-mapping get calls",
            "key": "Show one key and all consuming source locations",
            "examples": "List examples under the facade's example roots",
        },
        "categories": categories,
        "quick_ref": {
            "methods": dict(sorted(method_ref.items(), key=lambda item: item[0].lower())),
            "keys": dict(sorted(key_ref.items(), key=lambda item: item[0].lower())),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, help="GeoTaichi repository root")
    parser.add_argument("--output", type=Path, help="Write the index to this JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo = (args.repo_root or find_repo_root()).resolve()
        index = build_index(repo)
        if args.output:
            output = args.output if args.output.is_absolute() else repo / args.output
            write_json(output, index)
            try:
                output_label = output.relative_to(repo).as_posix()
            except ValueError:
                output_label = str(output)
            emit(
                build_ok(
                    {
                        "output": output_label,
                        "summary": {
                            name: data["summary"]
                            for name, data in index["categories"].items()
                        },
                    }
                )
            )
        else:
            emit(build_ok(index))
        return 0
    except Exception as exc:
        emit(build_error("index_build_failed", str(exc)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
