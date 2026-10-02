"""Finite-strain associated von Mises plasticity.

The model is parallel to Drucker--Prager and shares only the internal
multiplicative Hencky plasticity base. ``YieldStress`` uses the standard
equivalent-stress convention ``sqrt(3 J2)``. Optional ``HardeningModulus``
supplies linear isotropic hardening in equivalent plastic strain.
"""

from __future__ import annotations

import math

import taichi as ti

from src.physics_model.consititutive_model.finite_strain.HenckyPlasticity import (
    HenckyAssociatedPlasticityModel,
)


@ti.data_oriented
class FiniteStrainVonMisesModel(HenckyAssociatedPlasticityModel):
    """Multiplicative finite-strain J2 plasticity with associated flow."""

    def __init__(
        self,
        material_type="Solid",
        configuration="UL",
        solver_type="Implicit",
    ):
        super().__init__(material_type, configuration, solver_type)
        self.yield_stress_equivalent = 0.0
        self.hardening_modulus = 0.0

    def model_initialize(self, material):
        self.material = material
        density = float(self._parameter(material, ("Density", "density"), 2650.0))
        young = float(
            self._parameter(
                material,
                ("YoungModulus", "young_modulus", "ElasticModulus"),
                required=True,
            )
        )
        poisson = float(self._parameter(material, ("PoissonRatio", "poisson_ratio"), 0.3))
        yield_stress = float(
            self._parameter(
                material,
                ("YieldStress", "yield_stress", "YieldStrength"),
                required=True,
            )
        )
        hardening = float(
            self._parameter(
                material,
                ("HardeningModulus", "hardening_modulus", "IsotropicHardening"),
                0.0,
            )
        )
        self.validate_elastic_parameters(density, young, poisson)
        self.validate_nonnegative_parameters(
            YieldStress=yield_stress,
            HardeningModulus=hardening,
        )
        self.add_material(
            density,
            young,
            poisson,
            yield_stress,
            hardening,
        )
        self.add_coupling_material(material)

    def add_material(
        self,
        density,
        young,
        poisson,
        yield_stress,
        hardening_modulus=0.0,
    ):
        self.density = float(density)
        self.young = float(young)
        self.poisson = float(poisson)
        self.yield_stress_equivalent = float(yield_stress)
        self.hardening_modulus = float(hardening_modulus)
        self.shear = 0.5 * self.young / (1.0 + self.poisson)
        self.bulk = self.young / (3.0 * (1.0 - 2.0 * self.poisson))
        self.alpha = 0.0
        # ||dev(tau)|| = sqrt(2/3) sigma_y when sqrt(3 J2)=sigma_y.
        self.reference_yield_intercept = math.sqrt(2.0 / 3.0) * self.yield_stress_equivalent
        self.trace_apex = 1.0e30
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    @ti.func
    def _material_parameter_partials(self, particle_id, parameter_id):
        """Device partials for ``(E, nu, yield_stress, hardening_modulus)``."""
        dmu = 0.0
        dbulk = 0.0
        dalpha = 0.0
        dyield = 0.0
        dhardening = 0.0
        if parameter_id == 0:
            dmu = 1.0 / (2.0 * (1.0 + self.poisson))
            dbulk = 1.0 / (3.0 * (1.0 - 2.0 * self.poisson))
        elif parameter_id == 1:
            dmu = -self.young / (2.0 * (1.0 + self.poisson) ** 2)
            dbulk = 2.0 * self.young / (3.0 * (1.0 - 2.0 * self.poisson) ** 2)
        elif parameter_id == 2:
            dyield = ti.sqrt(2.0 / 3.0)
        elif parameter_id == 3:
            dyield = ti.sqrt(2.0 / 3.0) * self.equivalent_plastic_strain[particle_id]
            dhardening = 2.0 / 3.0
        return dmu, dbulk, dalpha, dyield, dhardening

    @ti.func
    def current_yield_intercept(self, particle_id):
        return ti.sqrt(2.0 / 3.0) * (
            self.yield_stress_equivalent + self.hardening_modulus * self.equivalent_plastic_strain[particle_id]
        )

    @ti.func
    def plastic_hardening_modulus(self, particle_id):
        # d(tau_y)/d(gamma) = (2/3) H because
        # d(ep_eq) = sqrt(2/3) d(gamma).
        return (2.0 / 3.0) * self.hardening_modulus

    @ti.func
    def yield_intercept_equivalent_plastic_strain_derivative(self, particle_id):
        return ti.sqrt(2.0 / 3.0) * self.hardening_modulus

    def print_message(self, materialID):
        self.print_console_header()
        print("Constitutive model: finite-strain von Mises")
        print("Material ID: ", materialID)
        print("Density: ", self.density)
        print("Young Modulus: ", self.young)
        print("Poisson Ratio: ", self.poisson)
        print("Yield stress: ", self.yield_stress_equivalent)
        print("Linear isotropic hardening modulus: ", self.hardening_modulus, "\n")


VonMisesModel = FiniteStrainVonMisesModel


__all__ = ["FiniteStrainVonMisesModel", "VonMisesModel"]
