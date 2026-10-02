"""DEM barrier law reused by explicit FEM--DEM surface contact."""

import math

import taichi as ti

from src.fedem.contact.ContactModelBase import ContactModelBase, property_value


@ti.dataclass
class BarrierSurfaceProperty:
    active: ti.i32
    kappa: float
    normal_cutoff: float
    stiffness_ratio: float
    friction: float
    normal_damping: float
    tangential_damping: float


class BarrierModel(ContactModelBase):
    model_name = "Barrier"

    def __init__(self, simulation):
        super().__init__(simulation)
        self.surface_properties = BarrierSurfaceProperty.field(
            shape=(
                simulation.max_dem_material_num,
                simulation.max_fem_body_num,
            )
        )
        self.maximum_kappa = 0.0
        self.maximum_normal_cutoff = 0.0

    def add_property(self, dem_material, fem_body, parameters):
        dem_material, fem_body = self.property_index(dem_material, fem_body)
        kappa = float(property_value(parameters, "Stiffness", "kappa", required=True))
        normal_cutoff = float(
            property_value(
                parameters,
                "NormalCutOff",
                "NormalCutoff",
                "ncut",
                required=True,
            )
        )
        stiffness_ratio = float(
            property_value(
                parameters,
                "StiffnessRatio",
                "ratio",
                default=1.0,
            )
        )
        values = {
            "active": 1,
            "kappa": kappa,
            "normal_cutoff": normal_cutoff,
            "stiffness_ratio": stiffness_ratio,
            "friction": float(property_value(parameters, "Friction", "mu", default=0.0)),
            "normal_damping": float(
                property_value(
                    parameters,
                    "NormalViscousDamping",
                    "ndratio",
                    default=0.0,
                )
            ),
            "tangential_damping": float(
                property_value(
                    parameters,
                    "TangentialViscousDamping",
                    "sdratio",
                    default=0.0,
                )
            ),
        }
        if kappa <= 0.0 or normal_cutoff <= 0.0 or stiffness_ratio <= 0.0:
            raise ValueError("FEDEM barrier stiffness, cutoff, and stiffness ratio must be positive")
        if (
            min(
                values["friction"],
                values["normal_damping"],
                values["tangential_damping"],
            )
            < 0.0
        ):
            raise ValueError("FEDEM barrier friction and damping ratios must be non-negative")
        self.surface_properties[dem_material, fem_body] = values
        self.maximum_kappa = max(self.maximum_kappa, kappa)
        self.maximum_normal_cutoff = max(self.maximum_normal_cutoff, normal_cutoff)

    def critical_timestep(self, minimum_mass, maximum_radius=None):
        del maximum_radius
        if self.maximum_kappa <= 0.0:
            return math.inf
        zero_gap_tangent = (4.0 + 2.0 * math.log(2.0)) * self.maximum_kappa
        return math.sqrt(float(minimum_mass) / zero_gap_tangent)


__all__ = ["BarrierModel", "BarrierSurfaceProperty"]
