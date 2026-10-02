"""Shared FEM--MPM contact-property handling."""

from __future__ import annotations

import math


def property_value(mapping, name, *aliases, default=None, required=False):
    normalized = {str(key).replace("_", "").replace("-", "").lower(): value for key, value in mapping.items()}
    for candidate in (name, *aliases):
        key = str(candidate).replace("_", "").replace("-", "").lower()
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(f"FEMPM contact property {name!r} is required")
    return default


class ContactModelBase:
    model_name = "ContactModel"

    def __init__(self, simulation):
        self.simulation = simulation
        self.surface_properties = None

    def property_index(self, mpm_material, fem_body):
        mpm_material = int(mpm_material)
        fem_body = int(fem_body)
        if not 0 <= mpm_material < self.simulation.max_mpm_material_num:
            raise IndexError("MPM material id is outside allocated FEMPM range")
        if not 0 <= fem_body < self.simulation.max_fem_body_num:
            raise IndexError("FEM body id is outside allocated FEMPM range")
        return mpm_material, fem_body

    def critical_timestep(self, minimum_mass, maximum_radius=None):
        del minimum_mass, maximum_radius
        return math.inf


__all__ = ["ContactModelBase", "property_value"]
