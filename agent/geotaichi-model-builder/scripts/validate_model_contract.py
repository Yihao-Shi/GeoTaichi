#!/usr/bin/env python3
"""Validate a GeoTaichi model contract JSON file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import build_error, build_ok, emit, load_json


REQUIRED_FIELDS = (
    "schema_version",
    "title",
    "module",
    "dimension",
    "coordinate_assumption",
    "units",
    "physics",
    "geometry",
    "discretization",
    "materials",
    "initial_conditions",
    "boundary_conditions",
    "contact_or_coupling",
    "time",
    "capacities",
    "outputs",
    "validation",
    "assumptions",
    "unresolved",
    "api",
)

MODULES = {
    "mpm",
    "dem",
    "mpdem",
    "cfdem",
    "fem",
    "fedem",
    "fempm",
    "iga",
    "igampm",
}
COORDINATES = {"3d", "plane_strain", "plane_stress", "axisymmetric", "2d"}


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def validate_contract(contract: Any) -> dict[str, Any]:
    """Return errors and warnings for one decoded contract."""
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    def error(path: str, message: str) -> None:
        errors.append({"path": path, "message": message})

    def warning(path: str, message: str) -> None:
        warnings.append({"path": path, "message": message})

    if not isinstance(contract, dict):
        return {"valid": False, "errors": [{"path": "$", "message": "Contract must be a JSON object."}], "warnings": []}

    for field in REQUIRED_FIELDS:
        if field not in contract:
            error(field, "Required field is missing.")

    if contract.get("schema_version") != 1:
        error("schema_version", "Expected schema_version 1.")
    module = contract.get("module")
    if module not in MODULES:
        error("module", f"Expected one of: {', '.join(sorted(MODULES))}.")
    dimension = contract.get("dimension")
    if dimension not in {2, 3}:
        error("dimension", "Expected integer 2 or 3.")
    coordinate = contract.get("coordinate_assumption")
    if coordinate not in COORDINATES:
        warning("coordinate_assumption", "Use a supported explicit assumption and confirm it in source.")
    if dimension == 3 and coordinate not in {None, "3d"}:
        error("coordinate_assumption", "Three-dimensional models must use coordinate_assumption '3d'.")

    units = contract.get("units")
    if not isinstance(units, dict):
        error("units", "Expected an object with length, mass, and time base units.")
    else:
        for unit in ("length", "mass", "time"):
            if not isinstance(units.get(unit), str) or not units[unit].strip():
                error(f"units.{unit}", "Expected a non-empty unit string.")

    for field in ("materials", "boundary_conditions", "assumptions", "unresolved"):
        value = contract.get(field)
        if not isinstance(value, list):
            error(field, "Expected a list.")
    if isinstance(contract.get("materials"), list) and not contract["materials"]:
        error("materials", "At least one material is required.")

    time = contract.get("time")
    if not isinstance(time, dict):
        error("time", "Expected an object.")
    else:
        if not _positive_number(time.get("duration")):
            error("time.duration", "Expected a positive duration.")
        timestep = time.get("timestep")
        if not (_positive_number(timestep) or timestep == "adaptive"):
            error("time.timestep", "Expected a positive number or 'adaptive'.")
        if not _positive_number(time.get("save_interval")):
            error("time.save_interval", "Expected a positive save interval.")
        if _positive_number(time.get("duration")) and _positive_number(time.get("save_interval")):
            if time["save_interval"] > time["duration"]:
                warning(
                    "time.save_interval",
                    "Save interval exceeds total duration; only terminal output may be produced.",
                )

    capacities = contract.get("capacities")
    if not isinstance(capacities, dict) or not capacities:
        error("capacities", "Expected a non-empty object with estimates and allocations.")
    else:
        margin = capacities.get("safety_margin")
        if not _positive_number(margin) or margin < 1:
            error("capacities.safety_margin", "Expected a multiplicative margin >= 1.")

    validation = contract.get("validation")
    if not isinstance(validation, dict):
        error("validation", "Expected an object.")
    else:
        for field in ("observable", "expectation", "tolerance", "evidence"):
            if field not in validation or validation[field] in (None, "", []):
                error(f"validation.{field}", "Required validation field is empty.")
        invariants = validation.get("invariants")
        if invariants is None:
            warning(
                "validation.invariants",
                "Declare machine-readable invariant criteria before physics scoring.",
            )
        elif not isinstance(invariants, list) or not invariants:
            error("validation.invariants", "Expected a non-empty list when provided.")
        else:
            for index, invariant in enumerate(invariants):
                path = f"validation.invariants[{index}]"
                if not isinstance(invariant, dict):
                    error(path, "Expected an object.")
                    continue
                for field in ("kind", "expectation", "tolerance", "basis"):
                    if field not in invariant or invariant[field] in (None, "", []):
                        error(f"{path}.{field}", "Required invariant criterion is empty.")

    outputs = contract.get("outputs")
    if not isinstance(outputs, dict):
        error("outputs", "Expected an object.")
    else:
        output_path = outputs.get("path")
        if isinstance(output_path, str) and output_path.startswith("/"):
            warning("outputs.path", "Use a repository-relative production path or /private/tmp for smoke runs.")

    api = contract.get("api")
    if not isinstance(api, dict):
        error("api", "Expected an object containing source-backed facade dictionaries.")
    elif module and api.get("facade", "").lower() not in {module, "dempm" if module in {"mpdem", "cfdem"} else module}:
        warning("api.facade", "Facade label does not obviously match module; verify coupled aliases and ownership.")

    unresolved = contract.get("unresolved")
    if isinstance(unresolved, list) and unresolved:
        warning("unresolved", "Unresolved choices prevent a fully validated production handoff.")

    return {"valid": not errors, "errors": errors, "warnings": warnings}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path, help="Model contract JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract = load_json(args.contract)
        result = validate_contract(contract)
        payload = build_ok(result) if result["valid"] else build_error(
            "invalid_model_contract",
            "The model contract failed validation.",
            result,
        )
        emit(payload)
        return 0 if result["valid"] else 1
    except FileNotFoundError as exc:
        emit(build_error("contract_not_found", str(exc)))
        return 1
    except ValueError as exc:
        emit(build_error("contract_json_invalid", str(exc)))
        return 1
    except Exception as exc:
        emit(build_error("contract_validation_failed", str(exc)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
