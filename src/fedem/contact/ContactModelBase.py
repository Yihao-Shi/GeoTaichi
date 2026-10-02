"""Shared contact-model allocation and property handling."""

from __future__ import annotations

import math

import taichi as ti


def property_value(mapping, name, *aliases, default=None, required=False):
    normalized = {str(key).replace("_", "").replace("-", "").lower(): value for key, value in mapping.items()}
    for candidate in (name, *aliases):
        key = str(candidate).replace("_", "").replace("-", "").lower()
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(f"FEDEM contact property {name!r} is required")
    return default


class ContactModelBase:
    model_name = "ContactModel"

    def __init__(self, simulation):
        self.simulation = simulation
        self.surface_properties = None
        self.resolve_kernel = None
        self.elastic_energy = ti.field(dtype=ti.lang.impl.current_cfg().default_fp, shape=())
        self.friction_dissipation = ti.field(dtype=ti.lang.impl.current_cfg().default_fp, shape=())
        self.damping_dissipation = ti.field(dtype=ti.lang.impl.current_cfg().default_fp, shape=())
        self.elastic_energy.fill(0.0)
        self.friction_dissipation.fill(0.0)
        self.damping_dissipation.fill(0.0)

    def reset_elastic_energy(self):
        self.elastic_energy.fill(0.0)

    def energy_diagnostics(self):
        return {
            "elastic_energy": float(self.elastic_energy[None]),
            "friction_dissipation": float(self.friction_dissipation[None]),
            "damping_dissipation": float(self.damping_dissipation[None]),
        }

    def property_index(self, dem_material, fem_body):
        dem_material = int(dem_material)
        fem_body = int(fem_body)
        if not 0 <= dem_material < self.simulation.max_dem_material_num:
            raise IndexError("DEM material id is outside allocated FEDEM range")
        if not 0 <= fem_body < self.simulation.max_fem_body_num:
            raise IndexError("FEM body id is outside allocated FEDEM range")
        return dem_material, fem_body

    def critical_timestep(self, minimum_mass, maximum_radius=None):
        del maximum_radius
        return math.inf


__all__ = ["ContactModelBase", "property_value"]
