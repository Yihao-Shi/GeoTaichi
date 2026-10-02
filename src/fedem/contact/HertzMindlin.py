"""Hertz--Mindlin FEM--DEM surface contact."""

import math

import taichi as ti

from src.fedem.contact.ContactModelBase import ContactModelBase, property_value


@ti.dataclass
class HertzMindlinSurfaceProperty:
    active: ti.i32
    effective_young: float
    effective_shear: float
    friction: float
    damping: float


class HertzMindlinModel(ContactModelBase):
    model_name = "HertzMindlin"

    def __init__(self, simulation):
        super().__init__(simulation)
        self.surface_properties = HertzMindlinSurfaceProperty.field(
            shape=(
                simulation.max_dem_material_num,
                simulation.max_fem_body_num,
            )
        )
        self.maximum_effective_young = 0.0
        self.maximum_effective_shear = 0.0

    def add_property(self, dem_material, fem_body, parameters):
        dem_material, fem_body = self.property_index(dem_material, fem_body)
        modulus = float(property_value(parameters, "ShearModulus", "Modulus", required=True))
        poisson = float(property_value(parameters, "Poisson", "PoissonRatio", required=True))
        if modulus <= 0.0 or not -1.0 < poisson < 0.5:
            raise ValueError("invalid Hertz--Mindlin modulus/Poisson ratio")
        effective_shear = 0.5 * modulus / (2.0 - poisson)
        effective_young = (4.0 * effective_shear - 2.0 * effective_shear * poisson) / (1.0 - poisson)
        restitution = float(property_value(parameters, "Restitution", required=True))
        if not 0.0 <= restitution <= 1.0:
            raise ValueError("FEDEM restitution must be in [0, 1]")
        damping = 0.0
        if restitution >= 1.0e-16:
            logarithm = math.log(restitution)
            damping = -logarithm / math.sqrt(math.pi * math.pi + logarithm * logarithm)
        friction = float(property_value(parameters, "Friction", "mu", default=0.0))
        if friction < 0.0:
            raise ValueError("FEDEM friction must be non-negative")
        self.surface_properties[dem_material, fem_body] = {
            "active": 1,
            "effective_young": effective_young,
            "effective_shear": effective_shear,
            "friction": friction,
            "damping": damping,
        }
        self.maximum_effective_young = max(self.maximum_effective_young, effective_young)
        self.maximum_effective_shear = max(self.maximum_effective_shear, effective_shear)

    def critical_timestep(self, minimum_mass, maximum_radius=None):
        if maximum_radius is None or float(maximum_radius) <= 0.0:
            return math.inf
        radius = float(maximum_radius)
        # Active one-sided contact has 0 <= penetration <= radius.  The
        # source Hertz law therefore has dFn/ddelta <= 2 E* R and
        # tangential stiffness <= 8 G* R over its admissible section.
        maximum_stiffness = max(
            2.0 * self.maximum_effective_young * radius,
            8.0 * self.maximum_effective_shear * radius,
        )
        if maximum_stiffness <= 0.0:
            return math.inf
        return math.sqrt(float(minimum_mass) / maximum_stiffness)


__all__ = ["HertzMindlinModel", "HertzMindlinSurfaceProperty"]
