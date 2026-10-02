"""Classical finite-strain associated Drucker--Prager plasticity."""

from __future__ import annotations

import math

import taichi as ti

from src.physics_model.consititutive_model.finite_strain.HenckyPlasticity import (
    HenckyAssociatedPlasticityModel,
)


@ti.data_oriented
class FiniteStrainDruckerPragerModel(HenckyAssociatedPlasticityModel):
    """Perfect Drucker--Prager cone with physical cohesion and associated flow."""

    def __init__(
        self,
        material_type="Solid",
        configuration="UL",
        solver_type="Implicit",
    ):
        super().__init__(material_type, configuration, solver_type)
        self.friction_angle = 0.0
        self.dilation_angle = 0.0
        self.cohesion = 0.0
        self.dp_type = "Circumscribed"
        self.q_friction = 0.0
        self.k_cohesion = 0.0
        self.cohesive_yield_stress = 0.0
        self.dp_type_code = 0

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
        cohesion = float(self._parameter(material, ("Cohesion", "cohesion"), 0.0))
        friction = float(
            self._parameter(
                material,
                ("FrictionAngle", "Friction", "StaticFriction", "friction_angle"),
                required=True,
            )
        )
        dilation_value = self._parameter(
            material,
            ("DilationAngle", "Dilation", "dilation_angle"),
            None,
        )
        dilation = friction if dilation_value is None else float(dilation_value)
        dp_type = str(
            self._parameter(
                material,
                ("dpType", "DPType", "yield_surface_type"),
                "Circumscribed",
            )
        )
        self.validate_elastic_parameters(density, young, poisson)
        self.validate_nonnegative_parameters(Cohesion=cohesion)
        self.validate_friction_angle("FrictionAngle", friction)
        self.validate_friction_angle("DilationAngle", dilation)
        if abs(dilation - friction) > 1.0e-12:
            raise ValueError(
                "Finite-strain Drucker-Prager currently implements associated "
                "flow only, so DilationAngle must equal FrictionAngle."
            )
        self.add_material(
            density,
            young,
            poisson,
            cohesion,
            math.radians(friction),
            math.radians(dilation),
            dp_type,
        )
        self.add_coupling_material(material)

    def add_material(
        self,
        density,
        young,
        poisson,
        cohesion,
        friction_angle,
        dilation_angle=None,
        dp_type="Circumscribed",
    ):
        self.density = float(density)
        self.young = float(young)
        self.poisson = float(poisson)
        self.cohesion = float(cohesion)
        self.friction_angle = float(friction_angle)
        self.dilation_angle = self.friction_angle if dilation_angle is None else float(dilation_angle)
        if abs(self.dilation_angle - self.friction_angle) > 1.0e-12:
            raise ValueError(
                "Finite-strain Drucker-Prager currently implements associated "
                "flow only, so dilation_angle must equal friction_angle."
            )
        self.shear = 0.5 * self.young / (1.0 + self.poisson)
        self.bulk = self.young / (3.0 * (1.0 - 2.0 * self.poisson))
        self.dp_type = self._canonical_dp_type(dp_type)
        self.dp_type_code = {
            "Circumscribed": 0,
            "MiddleCircumscribed": 1,
            "Inscribed": 2,
        }[self.dp_type]
        self.q_friction, self.k_cohesion = self._cone_parameters(self.cohesion, self.friction_angle, self.dp_type)

        # sqrt(J2) + q I1/3 - k <= 0 is multiplied by sqrt(2) to
        # match ||dev(tau)|| + alpha I1 - tau_c <= 0.
        self.alpha = math.sqrt(2.0) * self.q_friction / 3.0
        self.cohesive_yield_stress = math.sqrt(2.0) * self.k_cohesion
        self.reference_yield_intercept = self.cohesive_yield_stress
        elastic_trace_modulus = 3.0 * self.lame_lambda + 2.0 * self.shear
        if self.alpha > 1.0e-14:
            self.trace_apex = self.cohesive_yield_stress / (self.alpha * elastic_trace_modulus)
        else:
            self.trace_apex = 1.0e30
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    @ti.func
    def _material_parameter_partials(self, particle_id, parameter_id):
        """Device partials for ``(E, nu, cohesion, friction_angle_deg)``."""
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
            sine = ti.sin(self.friction_angle)
            cosine = ti.cos(self.friction_angle)
            root3 = ti.sqrt(3.0)
            dk = 0.0
            if self.dp_type_code == 0:
                denominator = root3 * (3.0 - sine)
                dk = 6.0 * cosine / denominator
            elif self.dp_type_code == 1:
                denominator = root3 * (3.0 + sine)
                dk = 6.0 * cosine / denominator
            else:
                tangent = ti.tan(self.friction_angle)
                denominator = ti.sqrt(9.0 + 12.0 * tangent * tangent)
                dk = 3.0 / denominator
            dyield = ti.sqrt(2.0) * dk
        elif parameter_id == 3:
            sine = ti.sin(self.friction_angle)
            cosine = ti.cos(self.friction_angle)
            root3 = ti.sqrt(3.0)
            dq = 0.0
            dk = 0.0
            if self.dp_type_code == 0:
                denominator = 3.0 - sine
                dq = 18.0 * cosine / (root3 * denominator * denominator)
                dk = 6.0 * self.cohesion * (1.0 - 3.0 * sine) / (root3 * denominator * denominator)
            elif self.dp_type_code == 1:
                denominator = 3.0 + sine
                dq = 18.0 * cosine / (root3 * denominator * denominator)
                dk = -6.0 * self.cohesion * (1.0 + 3.0 * sine) / (root3 * denominator * denominator)
            else:
                tangent = ti.tan(self.friction_angle)
                denominator = ti.sqrt(9.0 + 12.0 * tangent * tangent)
                dq = 27.0 * (1.0 + tangent * tangent) / (denominator**3)
                dk = -36.0 * self.cohesion * tangent * (1.0 + tangent * tangent) / (denominator**3)
            # User-facing friction/dilation angles are specified in degrees.
            degree_to_radian = 0.017453292519943295
            dalpha = ti.sqrt(2.0) * dq / 3.0 * degree_to_radian
            dyield = ti.sqrt(2.0) * dk * degree_to_radian
        return dmu, dbulk, dalpha, dyield, dhardening

    @staticmethod
    def _canonical_dp_type(dp_type):
        key = str(dp_type).replace("-", "").replace("_", "").replace(" ", "").lower()
        aliases = {
            "circumscribed": "Circumscribed",
            "outer": "Circumscribed",
            "triaxialcompression": "Circumscribed",
            "middlecircumscribed": "MiddleCircumscribed",
            "inner": "MiddleCircumscribed",
            "triaxialextension": "MiddleCircumscribed",
            "inscribed": "Inscribed",
        }
        if key not in aliases:
            raise ValueError("dpType must be Circumscribed, MiddleCircumscribed, or Inscribed")
        return aliases[key]

    @staticmethod
    def _cone_parameters(cohesion, friction_angle, dp_type):
        sine = math.sin(friction_angle)
        cosine = math.cos(friction_angle)
        if dp_type == "Circumscribed":
            denominator = math.sqrt(3.0) * (3.0 - sine)
            q_friction = 6.0 * sine / denominator
            k_cohesion = 6.0 * cohesion * cosine / denominator
        elif dp_type == "MiddleCircumscribed":
            denominator = math.sqrt(3.0) * (3.0 + sine)
            q_friction = 6.0 * sine / denominator
            k_cohesion = 6.0 * cohesion * cosine / denominator
        else:
            tangent = math.tan(friction_angle)
            denominator = math.sqrt(9.0 + 12.0 * tangent * tangent)
            q_friction = 3.0 * tangent / denominator
            k_cohesion = 3.0 * cohesion / denominator
        return q_friction, k_cohesion

    def print_message(self, materialID):
        self.print_console_header()
        print("Constitutive model: finite-strain associated Drucker-Prager")
        print("Material ID: ", materialID)
        print("Density: ", self.density)
        print("Young Modulus: ", self.young)
        print("Poisson Ratio: ", self.poisson)
        print("Cohesion (stress): ", self.cohesion)
        print("Friction angle (radian): ", self.friction_angle)
        print("Dilation angle (radian): ", self.dilation_angle)
        print("Yield surface type: ", self.dp_type, "\n")


DruckerPragerModel = FiniteStrainDruckerPragerModel


__all__ = ["FiniteStrainDruckerPragerModel", "DruckerPragerModel"]
