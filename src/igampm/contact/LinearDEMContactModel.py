from __future__ import annotations
from src.igampm.contact.DEMContactModelBase import DEMContactModelBase
from src.utils.linalg import positive
import math
import src.utils.GlobalVariable as GlobalVariable
from src.physics_model.contact_model.LinearModel import LinearSurfaceProperty


class LinearDEMContactModel(DEMContactModelBase):
    """The repository's DEM linear spring--dashpot/Coulomb law."""

    model_name = "Linear"

    def __init__(self):
        super().__init__()
        self.maximum_stiffness = 0.0
        self.adaptive_stiffness = None

    def _allocate_properties(self, capacity):
        return LinearSurfaceProperty.field(shape=capacity)

    def _write_property(self, property_id, parameters):
        kn = positive(
            "NormalStiffness", self.property_value(parameters, "NormalStiffness", "kn", default=0.0), allow_zero=True
        )
        ks = positive(
            "TangentialStiffness",
            self.property_value(parameters, "TangentialStiffness", "ks", default=0.0),
            allow_zero=True,
        )
        emod = positive(
            "EffectiveModulus",
            self.property_value(parameters, "EffectiveModulus", "emod", default=0.0),
            allow_zero=True,
        )
        kratio = positive(
            "NormalToShearRatio",
            self.property_value(parameters, "NormalToShearRatio", "kratio", default=0.0),
            allow_zero=True,
        )
        adaptive = emod > 0.0 or kratio > 0.0
        if adaptive and not (emod > 0.0 and kratio > 0.0):
            raise ValueError("adaptive linear contact requires both EffectiveModulus and " "NormalToShearRatio")
        if not adaptive and not (kn > 0.0 and ks > 0.0):
            raise ValueError("linear contact requires positive NormalStiffness and " "TangentialStiffness")
        if adaptive and (kn > 0.0 or ks > 0.0):
            raise ValueError("choose fixed stiffnesses or EffectiveModulus/" "NormalToShearRatio, not both")
        if self.adaptive_stiffness is not None and adaptive != self.adaptive_stiffness:
            raise ValueError("one explicit IGA-MPM Linear model cannot mix fixed and " "adaptive-stiffness properties")
        self.adaptive_stiffness = adaptive
        GlobalVariable.ADAPTIVESTIFF = adaptive

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
        normal_damping = positive(
            "NormalViscousDamping",
            self.property_value(parameters, "NormalViscousDamping", "ndratio", default=0.0),
            allow_zero=True,
        )
        tangential_damping = positive(
            "TangentialViscousDamping",
            self.property_value(parameters, "TangentialViscousDamping", "sdratio", default=0.0),
            allow_zero=True,
        )
        self.surface_properties[property_id] = {
            "kn": kn,
            "ks": ks,
            "emod": emod,
            "kratio": kratio,
            "mus": mus,
            "mud": mud,
            "rmu": 0.0,
            "ndratio": normal_damping,
            "sdratio": tangential_damping,
            "ncut": 0.0,
            "elastic_energy": 0.0,
            "friction_energy": 0.0,
            "damp_energy": 0.0,
        }
        if not adaptive:
            self.maximum_stiffness = max(self.maximum_stiffness, kn, ks)
        else:
            self.maximum_stiffness = max(self.maximum_stiffness, math.pi * emod, math.pi * emod / kratio)

    def critical_timestep(self, minimum_mass, maximum_radius):
        stiffness = self.maximum_stiffness
        if self.adaptive_stiffness:
            stiffness *= float(maximum_radius)
        if stiffness <= 0.0:
            return math.inf
        return math.sqrt(float(minimum_mass) / stiffness)
