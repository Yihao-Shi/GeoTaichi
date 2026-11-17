from __future__ import annotations
from src.utils.errors import SimulationStateError
import taichi as ti


class DEMContactModelBase:
    @staticmethod
    def property_value(parameters, *names, default=None, required=False):
        for name in names:
            if name in parameters:
                return parameters[name]
        if required:
            raise KeyError(f"missing contact property: {' or '.join(names)}")
        return default

    """Host-side property ownership; all force evaluation remains in Taichi."""

    model_name = None

    def __init__(self):
        self.pending_properties = []
        self.surface_properties = None
        self.active_properties = None
        self.max_mpm_material_num = 0
        self.max_iga_body_num = 0
        self.initialized = False

    def add_property(self, mpm_material, iga_body, parameters):
        if self.initialized:
            raise SimulationStateError("explicit IGA-MPM contact properties are frozen after build()")
        material = int(mpm_material)
        body = int(iga_body)
        if material < 0 or body < 0:
            raise ValueError("contact material/body ids must be non-negative")
        self.pending_properties.append((material, body, dict(parameters)))

    def _property_id(self, material, body):
        if material >= self.max_mpm_material_num:
            raise ValueError(f"MPM material {material} exceeds allocated capacity " f"{self.max_mpm_material_num}")
        if body >= self.max_iga_body_num:
            raise ValueError(f"IGA body {body} exceeds allocated capacity " f"{self.max_iga_body_num}")
        return material * self.max_iga_body_num + body

    def initialize(self, max_mpm_material_num, max_iga_body_num):
        self.max_mpm_material_num = max(1, int(max_mpm_material_num))
        self.max_iga_body_num = max(1, int(max_iga_body_num))
        capacity = self.max_mpm_material_num * self.max_iga_body_num
        self.surface_properties = self._allocate_properties(capacity)
        self.active_properties = ti.field(ti.i32, shape=capacity)
        for material, body, parameters in self.pending_properties:
            property_id = self._property_id(material, body)
            self._write_property(property_id, parameters)
            self.active_properties[property_id] = 1
        self.initialized = True

    def critical_timestep(self, minimum_mass, maximum_radius):
        raise NotImplementedError

    def reset_instantaneous_energy(self):
        """Reset stored contact energy without touching cumulative losses."""
        self.surface_properties.elastic_energy.fill(0.0)

    def energy_diagnostics(self):
        """Download the contact ledger only at a scheduled sample point."""
        return {
            "contact_elastic_energy": float(self.surface_properties.elastic_energy.to_numpy().sum()),
            "contact_friction_energy": float(self.surface_properties.friction_energy.to_numpy().sum()),
            "contact_viscous_energy": float(self.surface_properties.damp_energy.to_numpy().sum()),
        }
