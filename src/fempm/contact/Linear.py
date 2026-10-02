"""DEM linear spring-dashpot law for FEM--MPM surface contact."""

import math

import taichi as ti

from src.fempm.contact.ContactModelBase import ContactModelBase, property_value


@ti.dataclass
class LinearSurfaceProperty:
    active: ti.i32
    kn: float
    ks: float
    friction: float
    normal_damping: float
    tangential_damping: float


class LinearModel(ContactModelBase):
    model_name = "Linear"

    def __init__(self, simulation):
        super().__init__(simulation)
        self.surface_properties = LinearSurfaceProperty.field(
            shape=(
                simulation.max_mpm_material_num,
                simulation.max_fem_body_num,
            )
        )
        self.maximum_normal_stiffness = 0.0

    def add_property(self, mpm_material, fem_body, parameters):
        mpm_material, fem_body = self.property_index(mpm_material, fem_body)
        kn = float(property_value(parameters, "NormalStiffness", "kn", required=True))
        ks = float(property_value(parameters, "TangentialStiffness", "ks", required=True))
        if kn <= 0.0 or ks <= 0.0:
            raise ValueError("FEMPM linear stiffnesses must be positive")
        values = {
            "active": 1,
            "kn": kn,
            "ks": ks,
            "friction": float(property_value(parameters, "Friction", "mu", default=0.0)),
            "normal_damping": float(property_value(parameters, "NormalViscousDamping", "ndratio", default=0.0)),
            "tangential_damping": float(
                property_value(
                    parameters,
                    "TangentialViscousDamping",
                    "sdratio",
                    default=0.0,
                )
            ),
        }
        if (
            min(
                values["friction"],
                values["normal_damping"],
                values["tangential_damping"],
            )
            < 0.0
        ):
            raise ValueError("FEMPM friction and damping ratios must be non-negative")
        self.surface_properties[mpm_material, fem_body] = values
        self.maximum_normal_stiffness = max(self.maximum_normal_stiffness, kn)

    def critical_timestep(self, minimum_mass, maximum_radius=None):
        del maximum_radius
        if self.maximum_normal_stiffness <= 0.0:
            return math.inf
        return math.sqrt(float(minimum_mass) / self.maximum_normal_stiffness)


__all__ = ["LinearModel", "LinearSurfaceProperty"]
