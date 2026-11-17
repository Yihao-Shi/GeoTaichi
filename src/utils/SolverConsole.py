"""Consistent terminal summaries for public GeoTaichi solvers."""

from __future__ import annotations

import re


CONSOLE_WIDTH = 71
CONTENT_WIDTH = 67


_CONSTITUTIVE_MODEL_NAMES = {
    "stvk": "St. Venant-Kirchhoff",
    "stvenantkirchhoff": "St. Venant-Kirchhoff",
    "stvenantkirchhoffmodel": "St. Venant-Kirchhoff",
    "neohookean": "Neo-Hookean",
    "neohookeanmodel": "Neo-Hookean",
    "clotharap": "Cloth ARAP",
    "clothneohookean": "Cloth Neo-Hookean",
}


def constitutive_model_name(model):
    """Return a readable name for a model alias, class, or instance."""
    raw_name = model if isinstance(model, str) else type(model).__name__
    key = (
        str(raw_name)
        .strip()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .lower()
    )
    if key in _CONSTITUTIVE_MODEL_NAMES:
        return _CONSTITUTIVE_MODEL_NAMES[key]
    readable = re.sub(r"(?<!^)(?=[A-Z])", " ", str(raw_name)).strip()
    if readable.endswith(" Model"):
        readable = readable[:-6]
    return readable or "Unknown"


def print_material_info(model_name, material_id, entries=(), solver_name=None):
    """Print material identity first, followed by optional properties."""
    owner = f"{solver_name} " if solver_name else ""
    print(f" {owner}Constitutive Model Information ".center(CONSOLE_WIDTH, "-"))
    print(f"Constitutive model: {constitutive_model_name(model_name)}")
    print(f"Material ID: {material_id}")
    for label, value in entries:
        display_value = "Not configured" if value is None else value
        print(f"{label}: {display_value}")
    print()


def print_solver_section(solver_name, section_name, entries):
    """Print one compact solver information section."""
    title = f" {solver_name} {section_name} "
    print(title.center(CONSOLE_WIDTH, "-"))
    for label, value in entries:
        display_value = "Not configured" if value is None else value
        print(f"{label}: {display_value}".ljust(CONTENT_WIDTH))


def runtime_architecture():
    """Return the active Taichi architecture without making console output fail."""
    try:
        from taichi.lang.impl import current_cfg

        return current_cfg().arch
    except (AttributeError, RuntimeError, TypeError):
        return None


def print_simulation_start(solver_name):
    print("#", f" Start {solver_name} Simulation ".center(CONTENT_WIDTH, "="), "#")


def print_simulation_end(solver_name, elapsed_seconds=None):
    if elapsed_seconds is not None:
        print(f"{solver_name} simulation-loop time = {elapsed_seconds:.6g} s")
    print("#", f" End {solver_name} Simulation ".center(CONTENT_WIDTH, "="), "#", "\n")


def print_save_file_info(solver_name, step, save_number, simulation_time, path):
    # Save Path belongs to Solver Information and is intentionally not
    # repeated for every frame.
    del path
    print(
        f"# {solver_name} Save | Step = {step} | Save Number = {save_number} | "
        f"Simulation Time = {simulation_time}"
    )


__all__ = [
    "constitutive_model_name",
    "print_material_info",
    "print_save_file_info",
    "print_simulation_end",
    "print_simulation_start",
    "print_solver_section",
    "runtime_architecture",
]
