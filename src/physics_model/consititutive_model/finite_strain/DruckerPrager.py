"""Finite-strain Drucker--Prager with independent friction and dilation."""

from __future__ import annotations

import math

import taichi as ti

from src.physics_model.consititutive_model.finite_strain.HenckyPlasticity import (
    HenckyAssociatedPlasticityModel,
)
from src.utils.MatrixFunction import flatten_matrix, unflatten_matrix


@ti.data_oriented
class FiniteStrainDruckerPragerModel(HenckyAssociatedPlasticityModel):
    """Perfect DP cone; a frozen flow correction gives symmetric inner solves.

    The physical return uses alpha(phi) and beta(psi). During a material
    fixed point, the associated potential sees e + delta I, where
    delta = alpha * gamma - Delta(vp)/3. At convergence its stress equals
    the nonassociated return exactly. Its Hessian is symmetric; the true
    physical tangent outside the inner solve generally is not.
    """

    def __init__(
        self,
        material_type="Solid",
        configuration="UL",
        solver_type="Implicit",
    ):
        super().__init__(material_type, configuration, solver_type)
        self.requires_lagged_incremental_potential = True
        self.friction_angle = 0.0
        self.dilation_angle = 0.0
        self.cohesion = 0.0
        self.dp_type = "Circumscribed"
        self.q_friction = 0.0
        self.k_cohesion = 0.0
        self.cohesive_yield_stress = 0.0
        self.dp_type_code = 0
        self.beta = 0.0
        self.is_nonassociated = False
        self.dilation_is_tied = True
        self.lagged_flow_shift = None
        self.lagged_predictor_residual = None
        self.lagged_predictor_relaxation = None
        # Coupled FEM/IGA IPC enables the physical nonsymmetric Newton map.
        # Other consumers retain their existing frozen-potential contract.
        self.use_direct_nonassociated_solve = False

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
        self.add_material(
            density,
            young,
            poisson,
            cohesion,
            math.radians(friction),
            None if dilation_value is None else math.radians(dilation),
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
        self.dilation_is_tied = dilation_angle is None
        self.dilation_angle = self.friction_angle if dilation_angle is None else float(dilation_angle)
        self.validate_elastic_parameters(self.density, self.young, self.poisson)
        self.validate_nonnegative_parameters(Cohesion=self.cohesion)
        self.validate_friction_angle("FrictionAngle", math.degrees(self.friction_angle))
        self.validate_friction_angle("DilationAngle", math.degrees(self.dilation_angle))
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
        q_dilation = self._cone_parameters(0.0, self.dilation_angle, self.dp_type)[0]
        self.beta = math.sqrt(2.0) * q_dilation / 3.0
        self.is_nonassociated = abs(self.beta - self.alpha) > 1.0e-14
        self.has_symmetric_tangent = not self.is_nonassociated
        # has_incremental_potential describes the frozen INNER solve only.
        self.has_physical_incremental_potential = not self.is_nonassociated
        if self.is_nonassociated:
            self.lagged_max_iterations = 100
        self.cohesive_yield_stress = math.sqrt(2.0) * self.k_cohesion
        self.reference_yield_intercept = self.cohesive_yield_stress
        elastic_trace_modulus = 3.0 * self.lame_lambda + 2.0 * self.shear
        if self.alpha > 1.0e-14:
            self.trace_apex = self.cohesive_yield_stress / (self.alpha * elastic_trace_modulus)
        else:
            self.trace_apex = 1.0e30
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def allocate_state(self, particle_count):
        super().allocate_state(particle_count)
        if self.is_nonassociated:
            self.lagged_flow_shift = ti.field(ti.f64, shape=int(particle_count))
            self.lagged_flow_shift.fill(0.0)
            self.lagged_predictor_residual = ti.Vector.field(2, ti.f64, shape=int(particle_count))
            self.lagged_predictor_relaxation = ti.field(ti.f64, shape=int(particle_count))
            self.lagged_predictor_residual.fill(0.0)
            self.lagged_predictor_relaxation.fill(1.0)

    @ti.func
    def plastic_flow_alpha(self):
        return self.beta

    @ti.func
    def _flow_parameter_partial(self, particle_id, parameter_id):
        derivative = 0.0
        if ti.static(self.dilation_is_tied or not self.is_nonassociated):
            derivative = self._material_parameter_partials(particle_id, parameter_id)[2]
        return derivative

    @ti.func
    def begin_lagged_incremental_potential(self, particle_id):
        self.begin_lagged_plastic_volume(particle_id)
        if ti.static(self.is_nonassociated):
            self.lagged_flow_shift[particle_id] = 0.0
            self.lagged_predictor_residual[particle_id] = ti.Vector.zero(ti.f64, 2)
            self.lagged_predictor_relaxation[particle_id] = 1.0

    @ti.func
    def refresh_lagged_incremental_potential(self, particle_id, total_deformation_gradient):
        previous_volume = self.lagged_plastic_jacobian[particle_id]
        error = self.refresh_lagged_plastic_volume(particle_id, total_deformation_gradient)
        if ti.static(self.is_nonassociated):
            trial = self.trial_elastic_deformation(particle_id, total_deformation_gradient)
            state = self._principal_trial_state(trial)
            returned = self._associated_principal_return(particle_id, state[3], state[4], state[5], state[6])
            candidate = self.alpha * returned[1] - returned[2] / 3.0
            if ti.static(self.beta <= 1.0e-14):
                if returned[3] == 2:
                    # The tensile cap has no deviatoric flow beyond norm(dev e).
                    # Keep the inner return inside the apex branch, rather than
                    # exactly at its nonsmooth cone/apex boundary.
                    candidate = self.alpha * returned[1]
            # Check the actual inner/physical stress mismatch. The apex has a
            # flat stress response: a shift change there need not change force.
            shift = self.lagged_flow_shift[particle_id]
            inner = self._principal_return_with_flow(
                particle_id,
                state[3] + shift,
                state[4] + 3.0 * shift,
                state[5],
                state[6],
                self.alpha,
            )
            physical_stress = self._elastic_principal_response(returned[0])[0]
            inner_stress = self._elastic_principal_response(inner[0])[0]
            error = ti.max(error, (physical_stress - inner_stress).norm() / (2.0 * self.shear))
            residual = ti.Vector(
                [
                    ti.log(self.lagged_plastic_jacobian[particle_id] / previous_volume),
                    candidate - shift,
                ]
            )
            previous = self.lagged_predictor_residual[particle_id]
            difference = residual - previous
            relaxation = self.lagged_predictor_relaxation[particle_id]
            # Aitken secant damping leaves the physical fixed point unchanged.
            # ponytail: particle-local damping; global acceleration if coupled modes still stall.
            if previous.norm_sqr() > 1.0e-30 and difference.norm_sqr() > 1.0e-30:
                relaxation = ti.min(
                    1.0,
                    ti.max(
                        1.0e-3,
                        -relaxation * previous.dot(difference) / difference.norm_sqr(),
                    ),
                )
            self.lagged_flow_shift[particle_id] = shift + relaxation * residual[1]
            self.lagged_plastic_jacobian[particle_id] = previous_volume * ti.exp(relaxation * residual[0])
            self.lagged_predictor_residual[particle_id] = residual
            self.lagged_predictor_relaxation[particle_id] = relaxation
            # The stopping error above is unrelaxed: small updates cannot fake
            # convergence of the physical stress or predicted plastic volume.
        return error

    @ti.func
    def _potential_trial_state(self, particle_id, deformation_gradient):
        state = self._principal_trial_state(deformation_gradient)
        shift = 0.0
        if ti.static(self.is_nonassociated and not self.use_direct_nonassociated_solve):
            if self.lagged_plastic_volume_active[particle_id] != 0:
                shift = self.lagged_flow_shift[particle_id]
        return state[0], state[1], state[2], state[3] + shift, state[4] + 3.0 * shift, state[5], state[6]

    @ti.func
    def _potential_principal_return(self, particle_id, strain, trace, dev, norm):
        return self._principal_return_with_flow(
            particle_id, strain, trace, dev, norm, self._potential_flow_alpha(particle_id)
        )

    @ti.func
    def _potential_flow_alpha(self, particle_id):
        flow_alpha = self.beta
        if ti.static(self.is_nonassociated and not self.use_direct_nonassociated_solve):
            if self.lagged_plastic_volume_active[particle_id] != 0:
                flow_alpha = self.alpha
        return flow_alpha

    @ti.func
    def _potential_flow_parameter_partial(self, particle_id, parameter_id):
        derivative = self._flow_parameter_partial(particle_id, parameter_id)
        if ti.static(self.is_nonassociated and not self.use_direct_nonassociated_solve):
            if self.lagged_plastic_volume_active[particle_id] != 0:
                derivative = self._material_parameter_partials(particle_id, parameter_id)[2]
        return derivative

    @ti.func
    def _potential_strain_jacobian(self, particle_id, strain, trace, dev, norm, multiplier, region):
        return self._strain_return_jacobian(
            particle_id, strain, trace, dev, norm, multiplier, region, self._potential_flow_alpha(particle_id)
        )

    @ti.func
    def strain_energy_density_at(self, particle_id, deformation_gradient):
        if ti.static(self.is_nonassociated):
            assert (
                self.lagged_plastic_volume_active[particle_id] != 0
            ), "Nonassociated DP energy requires a frozen inner potential"
        return HenckyAssociatedPlasticityModel.strain_energy_density_at(self, particle_id, deformation_gradient)

    @ti.func
    def _physical_plastic_jacobian_from_response(self, total_deformation_gradient, response):
        projected_trace = response[7].sum()
        return total_deformation_gradient.determinant() * ti.exp(-projected_trace)

    @ti.func
    def total_stress_from_response(self, particle_id, total_deformation_gradient, response):
        stress = HenckyAssociatedPlasticityModel.total_stress_from_response(
            self, particle_id, total_deformation_gradient, response
        )
        if ti.static(self.use_direct_nonassociated_solve):
            stress *= self._physical_plastic_jacobian_from_response(
                total_deformation_gradient, response
            ) / self.incremental_reference_jacobian(particle_id)
        return stress

    @ti.func
    def total_tangent_from_response(self, particle_id, total_deformation_gradient, response):
        tangent = HenckyAssociatedPlasticityModel.total_tangent_from_response(
            self, particle_id, total_deformation_gradient, response
        )
        if ti.static(self.use_direct_nonassociated_solve):
            jacobian = self._physical_plastic_jacobian_from_response(total_deformation_gradient, response)
            tangent *= jacobian / self.incremental_reference_jacobian(particle_id)
            returned_jacobian = self._projected_strain_jacobian(
                particle_id, response[3], response[4], response[5], response[6], response[8], response[9]
            )
            diagonal = ti.Matrix.zero(ti.f64, 3, 3)
            for column in ti.static(range(3)):
                trace_derivative = 0.0
                for row in ti.static(range(3)):
                    trace_derivative += returned_jacobian[row, column]
                diagonal[column, column] = (1.0 - trace_derivative) / response[2][column]
            # dP/dF includes dJp: P = Jp(F) Pe(Fe) Fp_n^{-T}.
            log_jacobian_gradient = flatten_matrix(
                response[0]
                @ diagonal
                @ response[1].transpose()
                @ self.plastic_deformation_inverse[particle_id].transpose()
            )
            physical_stress = self.total_stress_from_response(particle_id, total_deformation_gradient, response)
            tangent += physical_stress.outer_product(log_jacobian_gradient)
        return tangent

    @ti.func
    def total_first_piola_stress_at(self, particle_id, total_deformation_gradient):
        elastic = self.trial_elastic_deformation(particle_id, total_deformation_gradient)
        response = self._principal_response(particle_id, elastic)
        return unflatten_matrix(
            self.total_stress_from_response(particle_id, total_deformation_gradient, response),
            total_deformation_gradient,
        )

    @ti.func
    def total_first_piola_tangent_at(self, particle_id, total_deformation_gradient):
        elastic = self.trial_elastic_deformation(particle_id, total_deformation_gradient)
        response = self._principal_response(particle_id, elastic)
        return self.total_tangent_from_response(particle_id, total_deformation_gradient, response)

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
        print(
            "Constitutive model: finite-strain Drucker-Prager ("
            + ("nonassociated" if self.is_nonassociated else "associated")
            + ")"
        )
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
