"""Static lifecycle and public-API inspection for GeoTaichi model scripts."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


FACADE_TO_CATEGORY = {
    "MPM": "mpm",
    "DEM": "dem",
    "DEMPM": "mpdem",
    "MPDEM": "mpdem",
    "CFDEM": "cfdem",
    "FEM": "fem",
    "FEDEM": "fedem",
    "DEMFEM": "fedem",
    "FEMPM": "fempm",
    "MPMFEM": "fempm",
    "IGA": "iga",
    "IGAMPM": "igampm",
}
IMPORT_ENV = {"GEOTAICHI_REAL_DTYPE"}
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _call_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _string_constant(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _dict_keys(node: ast.AST) -> List[str]:
    if not isinstance(node, ast.Dict):
        return []
    values = []
    for key in node.keys:
        if key is not None:
            value = _string_constant(key)
            if value is not None:
                values.append(value)
    return values


def _environment_assignment(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute):
                owner = target.value
                if isinstance(owner.value, ast.Name) and owner.value.id == "os" and owner.attr == "environ":
                    return _string_constant(target.slice)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
        if isinstance(call.func, ast.Attribute) and call.func.attr in {"setdefault", "__setitem__"}:
            owner = call.func.value
            if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name):
                if owner.value.id == "os" and owner.attr == "environ" and call.args:
                    return _string_constant(call.args[0])
    return None


def _render_expression(source: str, node: ast.AST) -> str:
    unparse = getattr(ast, "unparse", None)
    if unparse is not None:
        return unparse(node)
    return ast.get_source_segment(source, node) or "<expression>"


def inspect_model(
    script_path: Path,
    index: Dict[str, Any],
    contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Inspect one script without importing GeoTaichi or initializing Taichi."""
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    facts: Dict[str, Any] = {
        "geotaichi_import_line": None,
        "init_calls": [],
        "environment_assignments": [],
        "facades": {},
        "delegated_facades": [],
        "method_calls": [],
        "dictionary_keys": [],
    }

    def error(code: str, message: str, line: Optional[int] = None) -> None:
        entry: Dict[str, Any] = {"code": code, "message": message}
        if line is not None:
            entry["line"] = line
        errors.append(entry)

    def warning(code: str, message: str, line: Optional[int] = None) -> None:
        entry: Dict[str, Any] = {"code": code, "message": message}
        if line is not None:
            entry["line"] = line
        warnings.append(entry)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "geotaichi":
            facts["geotaichi_import_line"] = facts["geotaichi_import_line"] or node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "geotaichi" or alias.name.startswith("geotaichi."):
                    facts["geotaichi_import_line"] = facts["geotaichi_import_line"] or node.lineno
                if alias.name == "src" or alias.name.startswith("src."):
                    warning("internal_import", "User model imports internal module %r." % alias.name, node.lineno)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "src" or node.module.startswith("src."))
        ):
            warning("internal_import", "User model imports internal module %r." % node.module, node.lineno)
        env_name = _environment_assignment(node)
        if env_name:
            facts["environment_assignments"].append({"name": env_name, "line": node.lineno})

    import_line = facts["geotaichi_import_line"]
    if import_line is None:
        error("missing_geotaichi_import", "No geotaichi import was found.")
    for assignment in facts["environment_assignments"]:
        if assignment["name"] in IMPORT_ENV and import_line and assignment["line"] > import_line:
            error(
                "late_import_environment",
                "%s must be set before importing geotaichi." % assignment["name"],
                assignment["line"],
            )

    facade_variables: Dict[str, Dict[str, str]] = {}
    absolute_paths: List[Tuple[int, str]] = []
    delegated_facades: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            constructor = _call_name(node.value)
            if constructor in FACADE_TO_CATEGORY:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        category = FACADE_TO_CATEGORY[constructor]
                        if contract and contract.get("module") == "cfdem" and constructor in {"DEMPM", "CFDEM"}:
                            category = "cfdem"
                        facade_variables[target.id] = {"facade": constructor, "category": category}
                        facts["facades"][target.id] = facade_variables[target.id]
                if constructor in {
                    "DEMPM",
                    "MPDEM",
                    "CFDEM",
                    "FEDEM",
                    "DEMFEM",
                    "FEMPM",
                    "MPMFEM",
                    "IGAMPM",
                }:
                    supplied = [*node.value.args, *(keyword.value for keyword in node.value.keywords)]
                    delegated_facades.update(value.id for value in supplied if isinstance(value, ast.Name))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("/") or WINDOWS_ABSOLUTE.match(node.value):
                absolute_paths.append((node.lineno, node.value))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name == "init":
            facts["init_calls"].append(
                {
                    "line": node.lineno,
                    "keywords": {
                        keyword.arg: _render_expression(source, keyword.value)
                        for keyword in node.keywords
                        if keyword.arg is not None
                    },
                }
            )
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            owner = node.func.value.id
            if owner in facade_variables:
                method = node.func.attr
                record = {"owner": owner, "method": method, "line": node.lineno}
                facts["method_calls"].append(record)
                for argument in node.args:
                    for key in _dict_keys(argument):
                        facts["dictionary_keys"].append({**record, "key": key})

    if len(facts["init_calls"]) != 1:
        error("init_count", "Expected exactly one init() call; found %d." % len(facts["init_calls"]))
    if not facade_variables:
        error("missing_facade", "No supported GeoTaichi facade construction was found.")
    facts["delegated_facades"] = sorted(delegated_facades)
    if facts["init_calls"] and facade_variables:
        first_facade_line = min(
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _call_name(node.value) in FACADE_TO_CATEGORY
        )
        if facts["init_calls"][0]["line"] > first_facade_line:
            error("late_init", "Call init() before constructing a facade.", facts["init_calls"][0]["line"])

    method_sets = {
        category: {item["name"] for item in data["public_methods"]} for category, data in index["categories"].items()
    }
    key_sets = {
        category: {item["name"] for item in data["configuration_keys"]}
        for category, data in index["categories"].items()
    }
    for call in facts["method_calls"]:
        category = facade_variables[call["owner"]]["category"]
        if call["method"] not in method_sets.get(category, set()):
            error(
                "unknown_facade_method",
                "%s.%s is not indexed as a public %s method." % (call["owner"], call["method"], category),
                call["line"],
            )
    for item in facts["dictionary_keys"]:
        category = facade_variables[item["owner"]]["category"]
        if item["key"] not in key_sets.get(category, set()):
            warning(
                "unindexed_dictionary_key",
                "Key %r was not found in the broad %s DictIO index; trace its signature and consumer."
                % (item["key"], category),
                item["line"],
            )

    for variable in facade_variables:
        calls = sorted(
            (item for item in facts["method_calls"] if item["owner"] == variable),
            key=lambda item: item["line"],
        )
        run_calls = [item for item in calls if item["method"] == "run"]
        if not run_calls:
            if variable not in delegated_facades:
                warning("missing_run", "Facade variable %r has no run() call." % variable)
        elif any(item["line"] > run_calls[0]["line"] for item in calls if item["method"] != "postprocessing"):
            warning(
                "calls_after_run",
                "Facade %r has setup or update calls after its first run()." % variable,
                run_calls[0]["line"],
            )
        configure = [item for item in calls if item["method"] == "set_configuration"]
        if run_calls and configure and configure[0]["line"] > run_calls[0]["line"]:
            error("configuration_after_run", "Configure %r before run()." % variable, configure[0]["line"])

    for line, value in absolute_paths:
        if value.startswith("/private/tmp") or value.startswith("/tmp"):
            continue
        warning("absolute_path", "Machine-specific absolute path: %s" % value, line)

    if contract:
        expected_module = contract.get("module")
        actual_categories = {metadata["category"] for metadata in facade_variables.values()}
        if expected_module and expected_module not in actual_categories:
            if not (expected_module in {"mpdem", "cfdem"} and "mpdem" in actual_categories):
                warning(
                    "contract_module_mismatch",
                    "Contract module %r does not match detected facades %s."
                    % (expected_module, sorted(actual_categories)),
                )
        init_keywords = facts["init_calls"][0]["keywords"] if facts["init_calls"] else {}
        if "dim" in init_keywords and isinstance(contract.get("dimension"), int):
            if init_keywords["dim"] != str(contract["dimension"]):
                error(
                    "contract_dimension_mismatch",
                    "Contract dimension %s differs from init(dim=%s)." % (contract["dimension"], init_keywords["dim"]),
                    facts["init_calls"][0]["line"],
                )

    return {"valid": not errors, "errors": errors, "warnings": warnings, "facts": facts}
