"""DEM constitutive laws for explicit point--NURBS IGA--MPM contact."""

from __future__ import annotations

import math

import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.physics_model.contact_model.HertzMindlinModel import (
    HertzMindlinSurfaceProperty,
)
from src.physics_model.contact_model.LinearModel import LinearSurfaceProperty


def _value(parameters, *names, default=None, required=False):
    for name in names:
        if name in parameters:
            return parameters[name]
    if required:
        raise KeyError(f"missing contact property: {' or '.join(names)}")
    return default


def _finite_nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


class DEMContactModelBase:
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
            raise RuntimeError("explicit IGA-MPM contact properties are frozen after build()")
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
        kn = _finite_nonnegative(
            _value(parameters, "NormalStiffness", "kn", default=0.0),
            "NormalStiffness",
        )
        ks = _finite_nonnegative(
            _value(parameters, "TangentialStiffness", "ks", default=0.0),
            "TangentialStiffness",
        )
        emod = _finite_nonnegative(
            _value(parameters, "EffectiveModulus", "emod", default=0.0),
            "EffectiveModulus",
        )
        kratio = _finite_nonnegative(
            _value(parameters, "NormalToShearRatio", "kratio", default=0.0),
            "NormalToShearRatio",
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

        friction = _finite_nonnegative(_value(parameters, "Friction", "mu", default=0.0), "Friction")
        mus = _finite_nonnegative(
            _value(parameters, "StaticFriction", default=friction),
            "StaticFriction",
        )
        mud = _finite_nonnegative(
            _value(parameters, "DynamicFriction", default=friction),
            "DynamicFriction",
        )
        rolling = _finite_nonnegative(
            _value(parameters, "RollingFriction", default=0.0),
            "RollingFriction",
        )
        if rolling > 0.0:
            raise ValueError(
                "IGA control points have no rotational DOFs; explicit "
                "IGA-MPM contact does not support RollingFriction"
            )
        normal_damping = _finite_nonnegative(
            _value(
                parameters,
                "NormalViscousDamping",
                "ndratio",
                default=0.0,
            ),
            "NormalViscousDamping",
        )
        tangential_damping = _finite_nonnegative(
            _value(
                parameters,
                "TangentialViscousDamping",
                "sdratio",
                default=0.0,
            ),
            "TangentialViscousDamping",
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
        modulus = float(_value(parameters, "ShearModulus", "Modulus", required=True))
        poisson = float(_value(parameters, "Poisson", "PoissonRatio", required=True))
        if not math.isfinite(modulus) or modulus <= 0.0:
            raise ValueError("ShearModulus must be finite and positive")
        if not math.isfinite(poisson) or not -1.0 < poisson < 0.5:
            raise ValueError("Poisson must lie in (-1, 0.5)")
        effective_shear = 0.5 * modulus / (2.0 - poisson)
        effective_young = (4.0 * effective_shear - 2.0 * effective_shear * poisson) / (1.0 - poisson)
        friction = _finite_nonnegative(_value(parameters, "Friction", "mu", default=0.0), "Friction")
        mus = _finite_nonnegative(
            _value(parameters, "StaticFriction", default=friction),
            "StaticFriction",
        )
        mud = _finite_nonnegative(
            _value(parameters, "DynamicFriction", default=friction),
            "DynamicFriction",
        )
        rolling = _finite_nonnegative(
            _value(parameters, "RollingFriction", default=0.0),
            "RollingFriction",
        )
        if rolling > 0.0:
            raise ValueError(
                "IGA control points have no rotational DOFs; explicit "
                "IGA-MPM contact does not support RollingFriction"
            )
        restitution = float(_value(parameters, "Restitution", required=True))
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


__all__ = [
    "DEMContactModelBase",
    "LinearDEMContactModel",
    "HertzMindlinDEMContactModel",
]
