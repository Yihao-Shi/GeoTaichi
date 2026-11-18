from __future__ import annotations
from src.igampm.contact.DEMContactModelBase import DEMContactModelBase
from src.utils.linalg import positive
import math
from src.physics_model.contact_model.HertzMindlinModel import HertzMindlinSurfaceProperty


class HertzMindlinDEMContactModel(DEMContactModelBase):
    """The repository's DEM Hertz--Mindlin/Coulomb law."""

    model_name = "HertzMindlin"

    def __init__(self):
        super().__init__()
        self.maximum_effective_young = 0.0
        self.maximum_effective_shear = 0.0

    def _allocate_properties(self, capacity):
        return HertzMindlinSurfaceProperty.field(shape=capacity)

    def _write_property(self, property_id, parameters):
        modulus = float(self.property_value(parameters, "ShearModulus", "Modulus", required=True))
        poisson = float(self.property_value(parameters, "Poisson", "PoissonRatio", required=True))
        if not math.isfinite(modulus) or modulus <= 0.0:
            raise ValueError("ShearModulus must be finite and positive")
        if not math.isfinite(poisson) or not -1.0 < poisson < 0.5:
            raise ValueError("Poisson must lie in (-1, 0.5)")
        effective_shear = 0.5 * modulus / (2.0 - poisson)
        effective_young = (4.0 * effective_shear - 2.0 * effective_shear * poisson) / (1.0 - poisson)
        friction = positive("Friction", self.property_value(parameters, "Friction", "mu", default=0.0), allow_zero=True)
        mus = positive(
            "StaticFriction", self.property_value(parameters, "StaticFriction", default=friction), allow_zero=True
        )
        mud = positive(
            "DynamicFriction", self.property_value(parameters, "DynamicFriction", default=friction), allow_zero=True
        )
        rolling = positive(
            "RollingFriction", self.property_value(parameters, "RollingFriction", default=0.0), allow_zero=True
        )
        if rolling > 0.0:
            raise ValueError(
                "IGA control points have no rotational DOFs; explicit "
                "IGA-MPM contact does not support RollingFriction"
            )
        restitution = float(self.property_value(parameters, "Restitution", required=True))
        if not math.isfinite(restitution) or not 0.0 <= restitution <= 1.0:
            raise ValueError("Restitution must lie in [0, 1]")
        damping = 0.0
        if restitution >= 1.0e-16:
            logarithm = math.log(restitution)
            damping = -logarithm / math.sqrt(math.pi * math.pi + logarithm * logarithm)
        self.surface_properties[property_id] = {
            "YoungModulus": effective_young,
            "ShearModulus": effective_shear,
            "damping": damping,
            "mus": mus,
            "mud": mud,
            "rmu": 0.0,
            "ncut": 0.0,
            "elastic_energy": 0.0,
            "friction_energy": 0.0,
            "damp_energy": 0.0,
        }
        self.maximum_effective_young = max(self.maximum_effective_young, effective_young)
        self.maximum_effective_shear = max(self.maximum_effective_shear, effective_shear)

    def critical_timestep(self, minimum_mass, maximum_radius):
        radius = float(maximum_radius)
        stiffness = max(
            2.0 * self.maximum_effective_young * radius,
            8.0 * self.maximum_effective_shear * radius,
        )
        if stiffness <= 0.0:
            return math.inf
        return math.sqrt(float(minimum_mass) / stiffness)
