"""Shared multiplicative Hencky machinery for associated plasticity."""

from __future__ import annotations

import taichi as ti

from src.physics_model.consititutive_model.finite_strain.FiniteStrainModel import (
    FiniteStrainModel,
)
from src.utils.MatrixFunction import contraction, flatten_matrix, unflatten_matrix


@ti.data_oriented
class HenckyAssociatedPlasticityModel(FiniteStrainModel):
    """Internal base for isotropic associated principal-space returns."""

    def __init__(
        self,
        material_type="Solid",
        configuration="UL",
        solver_type="Implicit",
    ):
        super().__init__(material_type, configuration, solver_type)
        self.is_elastic = False
        self.is_elastoplastic = True
        self.is_finite_strain_plastic = True
        self.has_symmetric_tangent = True
        self.has_incremental_potential = True
        self.requires_lagged_incremental_potential = False
        self.lagged_tolerance = 1.0e-8
        self.lagged_max_iterations = 20
        # Scalars followed by the column-major plastic inverse Fp^{-1}.
        # Engines keep the total deformation gradient; the constitutive
        # model owns the plastic part needed to form Fe = F Fp^{-1}.
        self.history_state_size = 11
        self.alpha = 0.0
        self.reference_yield_intercept = 0.0
        self.trace_apex = 1.0e30
        self.equivalent_plastic_strain = None
        self.volumetric_plastic_strain = None
        self.plastic_deformation_inverse = None
        self.lagged_plastic_jacobian = None
        self.lagged_plastic_volume_active = None

    @staticmethod
    def _parameter(material, names, default=None, required=False):
        for name in names:
            if name in material:
                return material[name]
        if required:
            raise KeyError(f"Missing essential material parameter: {names[0]}")
        return default

    def allocate_state(self, particle_count):
        particle_count = int(particle_count)
        self.equivalent_plastic_strain = ti.field(ti.f64, shape=particle_count)
        self.volumetric_plastic_strain = ti.field(ti.f64, shape=particle_count)
        self.plastic_deformation_inverse = ti.Matrix.field(3, 3, ti.f64, shape=particle_count)
        self.equivalent_plastic_strain.fill(0.0)
        self.volumetric_plastic_strain.fill(0.0)
        self._initialize_plastic_deformation_inverse()
        if self.requires_lagged_incremental_potential:
            self.lagged_plastic_jacobian = ti.field(ti.f64, shape=particle_count)
            self.lagged_plastic_volume_active = ti.field(ti.i32, shape=particle_count)
            self.lagged_plastic_jacobian.fill(1.0)
            self.lagged_plastic_volume_active.fill(0)

    @ti.kernel
    def _initialize_plastic_deformation_inverse(self):
        for particle_id in self.plastic_deformation_inverse:
            self.plastic_deformation_inverse[particle_id] = ti.Matrix.identity(ti.f64, 3)

    def define_state_vars(self):
        return {}

    @ti.func
    def get_history_state(self, particle_id):
        state = ti.Vector.zero(float, 11)
        state[0] = self.equivalent_plastic_strain[particle_id]
        state[1] = self.volumetric_plastic_strain[particle_id]
        for column, row in ti.static(ti.ndrange(3, 3)):
            state[2 + 3 * column + row] = self.plastic_deformation_inverse[particle_id][row, column]
        return state

    @ti.func
    def set_history_state(self, particle_id, history_state):
        self.equivalent_plastic_strain[particle_id] = history_state[0]
        self.volumetric_plastic_strain[particle_id] = history_state[1]
        for column, row in ti.static(ti.ndrange(3, 3)):
            self.plastic_deformation_inverse[particle_id][row, column] = history_state[2 + 3 * column + row]
        self.end_lagged_plastic_volume(particle_id)

    @ti.func
    def trial_elastic_deformation(self, particle_id, total_deformation_gradient):
        plastic_inverse = self.plastic_deformation_inverse[particle_id]
        assert plastic_inverse.determinant() > 0.0, "finite-strain plasticity requires det(Fp^{-1}) > 0"
        return total_deformation_gradient @ plastic_inverse

    @ti.func
    def plastic_reference_jacobian(self, particle_id):
        return 1.0 / self.plastic_deformation_inverse[particle_id].determinant()

    @ti.func
    def incremental_reference_jacobian(self, particle_id):
        """One frozen volume weight for the inner energy, force and tangent.

        The material fixed point predicts the accepted plastic volume without
        changing committed history. Outside that solve use the actual volume.
        The tangent here differentiates the INNER potential at fixed weight.
        """
        jacobian = self.plastic_reference_jacobian(particle_id)
        if ti.static(self.requires_lagged_incremental_potential):
            if self.lagged_plastic_volume_active[particle_id] != 0:
                jacobian = self.lagged_plastic_jacobian[particle_id]
        return jacobian

    @ti.func
    def begin_lagged_plastic_volume(self, particle_id):
        if ti.static(self.requires_lagged_incremental_potential):
            self.lagged_plastic_jacobian[particle_id] = self.plastic_reference_jacobian(particle_id)
            self.lagged_plastic_volume_active[particle_id] = 1

    @ti.func
    def refresh_lagged_plastic_volume(self, particle_id, total_deformation_gradient):
        elastic_trial = self.trial_elastic_deformation(particle_id, total_deformation_gradient)
        projected_elastic = self.project_elastic_deformation(particle_id, elastic_trial)
        candidate = total_deformation_gradient.determinant() / projected_elastic.determinant()
        assert candidate > 0.0, "finite-strain plasticity requires positive predicted plastic volume"
        error = ti.abs(ti.log(candidate / self.lagged_plastic_jacobian[particle_id]))
        self.lagged_plastic_jacobian[particle_id] = candidate
        return error

    @ti.func
    def end_lagged_plastic_volume(self, particle_id):
        if ti.static(self.requires_lagged_incremental_potential):
            self.lagged_plastic_volume_active[particle_id] = 0

    @ti.func
    def begin_lagged_incremental_potential(self, particle_id):
        self.begin_lagged_plastic_volume(particle_id)

    @ti.func
    def refresh_lagged_incremental_potential(self, particle_id, total_deformation_gradient):
        return self.refresh_lagged_plastic_volume(particle_id, total_deformation_gradient)

    @ti.func
    def total_strain_energy_density_at(self, particle_id, total_deformation_gradient):
        elastic_trial = self.trial_elastic_deformation(particle_id, total_deformation_gradient)
        return self.incremental_reference_jacobian(particle_id) * self.strain_energy_density_at(
            particle_id, elastic_trial
        )

    @ti.func
    def total_first_piola_stress_at(self, particle_id, total_deformation_gradient):
        plastic_inverse = self.plastic_deformation_inverse[particle_id]
        elastic_trial = total_deformation_gradient @ plastic_inverse
        return (
            self.incremental_reference_jacobian(particle_id)
            * self.first_piola_stress_at(particle_id, elastic_trial)
            @ plastic_inverse.transpose()
        )

    @ti.func
    def total_first_piola_tangent_at(self, particle_id, total_deformation_gradient):
        plastic_inverse = self.plastic_deformation_inverse[particle_id]
        elastic_trial = total_deformation_gradient @ plastic_inverse
        elastic_tangent = self.first_piola_tangent_at(particle_id, elastic_trial)
        return self._total_tangent_from_elastic(particle_id, elastic_tangent)

    @ti.func
    def _total_tangent_from_elastic(self, particle_id, elastic_tangent):
        plastic_inverse = self.plastic_deformation_inverse[particle_id]
        reference_jacobian = self.incremental_reference_jacobian(particle_id)
        tangent = ti.Matrix.zero(float, 9, 9)
        row = 0
        while row < 9:
            total_column = row // 3
            spatial_row = row - 3 * total_column
            column = 0
            while column < 9:
                total_direction = column // 3
                spatial_direction = column - 3 * total_direction
                value = 0.0
                elastic_column = 0
                while elastic_column < 3:
                    elastic_direction = 0
                    while elastic_direction < 3:
                        value += (
                            elastic_tangent[
                                3 * elastic_column + spatial_row,
                                3 * elastic_direction + spatial_direction,
                            ]
                            * plastic_inverse[total_column, elastic_column]
                            * plastic_inverse[total_direction, elastic_direction]
                        )
                        elastic_direction += 1
                    elastic_column += 1
                tangent[row, column] = reference_jacobian * value
                column += 1
            row += 1
        return tangent

    @ti.func
    def total_stress_from_response(self, particle_id, total_deformation_gradient, response):
        inverse = self.plastic_deformation_inverse[particle_id]
        elastic = total_deformation_gradient @ inverse
        return flatten_matrix(
            self.incremental_reference_jacobian(particle_id)
            * self._stress_from_response(elastic, response)
            @ inverse.transpose()
        )

    @ti.func
    def total_tangent_from_response(self, particle_id, total_deformation_gradient, response):
        elastic = self.trial_elastic_deformation(particle_id, total_deformation_gradient)
        tangent = self._tangent_from_response(particle_id, elastic, response)
        return self._total_tangent_from_elastic(particle_id, tangent)

    @ti.func
    def total_dPsi_div_dF_at(self, particle_id, total_deformation_gradient):
        return flatten_matrix(self.total_first_piola_stress_at(particle_id, total_deformation_gradient))

    @ti.func
    def total_d2Psi_div_d2F_at(self, particle_id, total_deformation_gradient):
        return self.total_first_piola_tangent_at(particle_id, total_deformation_gradient)

    @ti.func
    def commit_total_state(self, particle_id, total_deformation_gradient):
        elastic_trial = self.trial_elastic_deformation(particle_id, total_deformation_gradient)
        committed_elastic = self.commit_state(particle_id, elastic_trial)
        self.plastic_deformation_inverse[particle_id] = total_deformation_gradient.inverse() @ committed_elastic
        self.end_lagged_plastic_volume(particle_id)
        return total_deformation_gradient

    @ti.func
    def total_von_mises_at(self, particle_id, total_deformation_gradient):
        elastic_trial = self.trial_elastic_deformation(particle_id, total_deformation_gradient)
        elastic_pk1 = self.first_piola_stress_at(particle_id, elastic_trial)
        return self.von_mises_from_pk1(elastic_trial, elastic_pk1)

    @ti.func
    def _principal_trial_state(self, deformation_gradient):
        ti.static_assert(deformation_gradient.n == 3 and deformation_gradient.m == 3)
        assert deformation_gradient.determinant() > 0.0, "finite-strain plasticity requires det(F_e,tr) > 0"
        matrix_u, singular_matrix, matrix_v = ti.svd(deformation_gradient)
        dimension = ti.static(deformation_gradient.n)
        singular_values = ti.Vector.zero(float, dimension)
        strain = ti.Vector.zero(float, dimension)
        trace_strain = 0.0
        for i in ti.static(range(dimension)):
            singular_values[i] = singular_matrix[i, i]
            strain[i] = ti.log(singular_values[i])
            trace_strain += strain[i]
        # The volumetric strain is exactly log(det F). Avoid amplification of
        # iterative SVD roundoff in the pressure and hydrostatic apex energy.
        trace_strain = ti.log(deformation_gradient.determinant())
        deviatoric_strain = strain - (trace_strain / dimension) * ti.Vector.one(float, dimension)
        deviatoric_norm = deviatoric_strain.norm()
        return (
            matrix_u,
            matrix_v,
            singular_values,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
        )

    @ti.func
    def plastic_flow_alpha(self):
        return self.alpha

    @ti.func
    def _flow_parameter_partial(self, particle_id, parameter_id):
        return self._material_parameter_partials(particle_id, parameter_id)[2]

    @ti.func
    def _potential_flow_alpha(self, particle_id):
        return self.plastic_flow_alpha()

    @ti.func
    def _potential_flow_parameter_partial(self, particle_id, parameter_id):
        return self._flow_parameter_partial(particle_id, parameter_id)

    @ti.func
    def _potential_trial_state(self, particle_id, deformation_gradient):
        return self._principal_trial_state(deformation_gradient)

    @ti.func
    def _potential_principal_return(self, particle_id, strain, trace, dev, norm):
        return self._associated_principal_return(particle_id, strain, trace, dev, norm)

    @ti.func
    def _potential_strain_jacobian(self, particle_id, strain, trace, dev, norm, multiplier, region):
        return self._projected_strain_jacobian(particle_id, strain, trace, dev, norm, multiplier, region)

    @ti.func
    def current_yield_intercept(self, particle_id):
        return self.reference_yield_intercept

    @ti.func
    def plastic_hardening_modulus(self, particle_id):
        return 0.0

    @ti.func
    def yield_intercept_equivalent_plastic_strain_derivative(self, particle_id):
        return 0.0

    @ti.func
    def _material_parameter_partials(self, particle_id, parameter_id):
        """Return ``d(mu), d(K), d(alpha), d(tau_y), d(Hp)`` on device.

        The four slots are intentionally shared by the DP and von-Mises
        models: ``(E, nu, cohesion/friction-or-yield, hardening)``.  Concrete
        models override this tiny hook; keeping the return map here means the
        equilibrium and commit VJPs use exactly the same branch decisions as
        the forward constitutive update.
        """
        return 0.0, 0.0, 0.0, 0.0, 0.0

    @ti.func
    def _return_parameter_derivatives_from_response(
        self,
        particle_id,
        strain,
        trace_strain,
        deviatoric_strain,
        deviatoric_norm,
        projected,
        plastic_multiplier,
        return_region,
        parameter_id,
        use_potential=False,
    ):
        """Same return-map derivative with a caller-owned principal response.

        Keeping the SVD/return response outside the parameter loop is
        important on CUDA: four material slots should not instantiate four
        independent spectral decompositions in the hot adjoint kernel.
        """
        dmu, dbulk, dalpha, dyield, dhardening = self._material_parameter_partials(particle_id, parameter_id)
        flow_alpha = self.plastic_flow_alpha()
        dflow = self._flow_parameter_partial(particle_id, parameter_id)
        if use_potential:
            flow_alpha = self._potential_flow_alpha(particle_id)
            dflow = self._potential_flow_parameter_partial(particle_id, parameter_id)
        dimension = ti.static(3)
        dprojected = ti.Vector.zero(float, dimension)
        dplastic_multiplier = 0.0
        dvolume_increment = 0.0
        trace_modulus = 3.0 * self.bulk
        dtrace_modulus = 3.0 * dbulk
        if return_region == 1 and deviatoric_norm > 1.0e-14:
            denominator = (
                2.0 * self.shear
                + dimension * self.alpha * flow_alpha * trace_modulus
                + self.plastic_hardening_modulus(particle_id)
            )
            numerator = (
                2.0 * self.shear * deviatoric_norm
                + self.alpha * trace_modulus * trace_strain
                - self.current_yield_intercept(particle_id)
            )
            ddenominator = (
                2.0 * dmu
                + dimension
                * (
                    (dalpha * flow_alpha + self.alpha * dflow) * trace_modulus
                    + self.alpha * flow_alpha * dtrace_modulus
                )
                + dhardening
            )
            dnumerator = (
                2.0 * dmu * deviatoric_norm
                + (dalpha * trace_modulus + self.alpha * dtrace_modulus) * trace_strain
                - dyield
            )
            dplastic_multiplier = (dnumerator * denominator - numerator * ddenominator) / (denominator * denominator)
            direction = deviatoric_strain / deviatoric_norm
            dvolumetric_projection = dflow * plastic_multiplier + flow_alpha * dplastic_multiplier
            for i in ti.static(range(dimension)):
                dprojected[i] = -dplastic_multiplier * direction[i] - dvolumetric_projection
            dvolume_increment = dimension * (dflow * plastic_multiplier + flow_alpha * dplastic_multiplier)
        elif return_region == 2 and self.alpha > 1.0e-14:
            apex_denominator = self.alpha * trace_modulus
            dtrace_apex = (
                dyield * apex_denominator
                - self.current_yield_intercept(particle_id) * (dalpha * trace_modulus + self.alpha * dtrace_modulus)
            ) / (apex_denominator * apex_denominator)
            if flow_alpha > 1.0e-14:
                dplastic_multiplier = (
                    -dtrace_apex * dimension * flow_alpha - (trace_strain - self.trace_apex) * dimension * dflow
                ) / (dimension * dimension * flow_alpha * flow_alpha)
            for i in ti.static(range(dimension)):
                dprojected[i] = dtrace_apex / dimension
            dvolume_increment = -dtrace_apex
        return dprojected, dplastic_multiplier, dvolume_increment

    @ti.func
    def first_piola_parameter_derivatives_at(self, particle_id, deformation_gradient):
        """All four PK1 parameter derivatives sharing one device SVD."""
        response = self._principal_response(particle_id, deformation_gradient)
        matrix_u = response[0]
        matrix_v = response[1]
        singular_values = response[2]
        strain = response[3]
        trace_strain = response[4]
        deviatoric_strain = response[5]
        deviatoric_norm = response[6]
        projected = response[7]
        plastic_multiplier = response[8]
        return_region = response[9]
        derivatives = ti.Matrix.zero(float, 9, 4)
        for parameter_id in ti.static(range(4)):
            dprojected, _, _ = self._return_parameter_derivatives_from_response(
                particle_id,
                strain,
                trace_strain,
                deviatoric_strain,
                deviatoric_norm,
                projected,
                plastic_multiplier,
                return_region,
                parameter_id,
                use_potential=True,
            )
            dmu, dbulk, _, _, _ = self._material_parameter_partials(particle_id, parameter_id)
            projected_trace = projected[0] + projected[1] + projected[2]
            dtrace = dprojected[0] + dprojected[1] + dprojected[2]
            ddev = dprojected - (dtrace / 3.0) * ti.Vector.one(float, 3)
            dev = projected - (projected_trace / 3.0) * ti.Vector.one(float, 3)
            dkirchhoff = 2.0 * dmu * dev + 2.0 * self.shear * ddev
            dkirchhoff += (dbulk * projected_trace + self.bulk * dtrace) * ti.Vector.one(float, 3)
            diagonal = ti.Matrix.zero(float, 3, 3)
            for i in ti.static(range(3)):
                diagonal[i, i] = dkirchhoff[i] / singular_values[i]
            derivative = matrix_u @ diagonal @ matrix_v.transpose()
            for column, row in ti.static(ti.ndrange(3, 3)):
                derivatives[3 * column + row, parameter_id] = derivative[row, column]
        return derivatives

    @ti.func
    def first_piola_parameter_derivative_at(self, particle_id, deformation_gradient, parameter_id):
        """PK1 derivative for one material parameter, evaluated on device."""
        derivatives = self.first_piola_parameter_derivatives_at(particle_id, deformation_gradient)
        result = ti.Matrix.zero(float, 3, 3)
        for column, row in ti.static(ti.ndrange(3, 3)):
            result[row, column] = derivatives[3 * column + row, parameter_id]
        return result

    @ti.func
    def total_first_piola_parameter_vjp_at(self, particle_id, total_deformation_gradient, adjoint):
        """VJP of total PK1 wrt all four material parameters."""
        result = ti.Vector.zero(float, 4)
        plastic_inverse = self.plastic_deformation_inverse[particle_id]
        elastic_trial = total_deformation_gradient @ plastic_inverse
        reference_jacobian = self.incremental_reference_jacobian(particle_id)
        elastic_derivatives = self.first_piola_parameter_derivatives_at(particle_id, elastic_trial)
        for parameter_id in ti.static(range(4)):
            elastic_derivative = ti.Matrix.zero(float, 3, 3)
            for column, row in ti.static(ti.ndrange(3, 3)):
                elastic_derivative[row, column] = elastic_derivatives[3 * column + row, parameter_id]
            total_derivative = reference_jacobian * elastic_derivative @ plastic_inverse.transpose()
            result[parameter_id] = contraction(adjoint, total_derivative)
        return result

    @ti.func
    def commit_total_state_parameter_vjp(
        self,
        particle_id,
        total_deformation_gradient,
        plastic_inverse_output_vjp,
        equivalent_plastic_strain_output_vjp,
        volumetric_plastic_strain_output_vjp,
    ):
        """VJP of the accepted return-map state wrt material parameters."""
        result = ti.Vector.zero(float, 4)
        plastic_inverse = self.plastic_deformation_inverse[particle_id]
        elastic_trial = total_deformation_gradient @ plastic_inverse
        principal = self._principal_trial_state(elastic_trial)
        matrix_u = principal[0]
        matrix_v = principal[1]
        returned = self._associated_principal_return(
            particle_id,
            principal[3],
            principal[4],
            principal[5],
            principal[6],
        )
        projected = returned[0]
        plastic_multiplier = returned[1]
        return_region = returned[3]
        total_inverse = total_deformation_gradient.inverse()
        for parameter_id in ti.static(range(4)):
            dprojected, dplastic_multiplier, dvolume_increment = self._return_parameter_derivatives_from_response(
                particle_id,
                principal[3],
                principal[4],
                principal[5],
                principal[6],
                projected,
                plastic_multiplier,
                return_region,
                parameter_id,
            )
            dcommitted_elastic = ti.Matrix.zero(float, 3, 3)
            for principal_id in ti.static(range(3)):
                value = ti.exp(projected[principal_id]) * dprojected[principal_id]
                for row, column in ti.static(ti.ndrange(3, 3)):
                    dcommitted_elastic[row, column] += (
                        value * matrix_u[row, principal_id] * matrix_v[column, principal_id]
                    )
            dplastic_inverse = total_inverse @ dcommitted_elastic
            result[parameter_id] = (
                contraction(plastic_inverse_output_vjp, dplastic_inverse)
                + ti.sqrt(2.0 / 3.0) * equivalent_plastic_strain_output_vjp * dplastic_multiplier
                + volumetric_plastic_strain_output_vjp * dvolume_increment
            )
        return result

    @ti.func
    def _elastic_principal_response(self, projected):
        """Kirchhoff stress and ``d tau / d log(lambda_e)``."""
        dimension = ti.static(projected.n)
        projected_trace = 0.0
        for i in ti.static(range(dimension)):
            projected_trace += projected[i]
        projected_deviatoric = projected - (projected_trace / dimension) * ti.Vector.one(float, dimension)
        kirchhoff = 2.0 * self.shear * projected_deviatoric + self.bulk * projected_trace
        stress_operator = ti.Matrix.zero(float, dimension, dimension)
        for i in ti.static(range(dimension)):
            for j in ti.static(range(dimension)):
                identity = 1.0 if ti.static(i == j) else 0.0
                stress_operator[i, j] = 2.0 * self.shear * (identity - 1.0 / dimension) + self.bulk
        return kirchhoff, stress_operator

    @ti.func
    def _projected_strain_jacobian(
        self,
        particle_id,
        strain,
        trace_strain,
        deviatoric_strain,
        deviatoric_norm,
        plastic_multiplier,
        return_region,
    ):
        return self._strain_return_jacobian(
            particle_id,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
            plastic_multiplier,
            return_region,
            self.plastic_flow_alpha(),
        )

    @ti.func
    def _strain_return_jacobian(
        self,
        particle_id,
        strain,
        trace_strain,
        deviatoric_strain,
        deviatoric_norm,
        plastic_multiplier,
        return_region,
        flow_alpha,
    ):
        """Exact principal Jacobian; nonassociated flow generally is nonsymmetric."""
        dimension = ti.static(strain.n)
        derivative = ti.Matrix.identity(float, dimension)
        if return_region == 1 and deviatoric_norm > 1.0e-14:
            trace_modulus = ti.static(dimension * self.lame_lambda + 2.0 * self.shear)
            denominator = (
                2.0 * self.shear
                + dimension * self.alpha * flow_alpha * trace_modulus
                + self.plastic_hardening_modulus(particle_id)
            )
            deviatoric_direction = deviatoric_strain / deviatoric_norm
            flow_direction = deviatoric_direction + flow_alpha
            yield_gradient = (2.0 * self.shear * deviatoric_direction + self.alpha * trace_modulus) / denominator
            for i in ti.static(range(dimension)):
                for j in ti.static(range(dimension)):
                    identity = 1.0 if ti.static(i == j) else 0.0
                    deviatoric_projector = identity - 1.0 / dimension
                    derivative[i, j] = (
                        identity
                        - flow_direction[i] * yield_gradient[j]
                        - plastic_multiplier
                        / deviatoric_norm
                        * (deviatoric_projector - deviatoric_direction[i] * deviatoric_direction[j])
                    )
        elif return_region == 2:
            derivative = ti.Matrix.zero(float, dimension, dimension)
        return derivative

    @ti.func
    def _associated_principal_return(
        self,
        particle_id,
        strain,
        trace_strain,
        deviatoric_strain,
        deviatoric_norm,
    ):
        return self._principal_return_with_flow(
            particle_id,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
            self.plastic_flow_alpha(),
        )

    @ti.func
    def _principal_return_with_flow(
        self, particle_id, strain, trace_strain, deviatoric_strain, deviatoric_norm, flow_alpha
    ):
        dimension = ti.static(strain.n)
        projected = strain
        plastic_multiplier = 0.0
        plastic_volume_increment = 0.0
        return_region = 0
        trace_modulus = ti.static(dimension * self.lame_lambda + 2.0 * self.shear)
        trial_yield = (
            2.0 * self.shear * deviatoric_norm
            + self.alpha * trace_modulus * trace_strain
            - self.current_yield_intercept(particle_id)
        )
        if trial_yield > 0.0:
            denominator = (
                2.0 * self.shear
                + dimension * self.alpha * flow_alpha * trace_modulus
                + self.plastic_hardening_modulus(particle_id)
            )
            plastic_multiplier = trial_yield / denominator
            if ti.static(self.alpha > 1.0e-14):
                if plastic_multiplier >= deviatoric_norm:
                    return_region = 2
                    if flow_alpha > 1.0e-14:
                        plastic_multiplier = (trace_strain - self.trace_apex) / (dimension * flow_alpha)
                    else:
                        # Zero-dilation cone cannot return hydrostatic tension.
                        # At its apex use a tensile cap, with only the deviatoric
                        # multiplier contributing to equivalent plastic strain.
                        plastic_multiplier = deviatoric_norm
                    projected = (self.trace_apex / dimension) * ti.Vector.one(float, dimension)
                    plastic_volume_increment = trace_strain - self.trace_apex
            if return_region != 2 and deviatoric_norm > 1.0e-14:
                return_region = 1
                projected = strain - (plastic_multiplier / deviatoric_norm) * deviatoric_strain
                projected -= (flow_alpha * plastic_multiplier) * ti.Vector.one(float, dimension)
                plastic_volume_increment = dimension * flow_alpha * plastic_multiplier
        return (
            projected,
            plastic_multiplier,
            plastic_volume_increment,
            return_region,
        )

    @ti.func
    def project_elastic_deformation(self, particle_id, deformation_gradient):
        (
            matrix_u,
            matrix_v,
            singular_values,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
        ) = self._principal_trial_state(deformation_gradient)
        projected_state = self._associated_principal_return(
            particle_id,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
        )
        projected = projected_state[0]
        projected_stretch = ti.Matrix.zero(float, deformation_gradient.n, deformation_gradient.m)
        for i in ti.static(range(deformation_gradient.n)):
            projected_stretch[i, i] = ti.exp(projected[i])
        return matrix_u @ projected_stretch @ matrix_v.transpose()

    @ti.func
    def _spectral_matrix_tangent(
        self,
        matrix_u,
        matrix_v,
        singular_values,
        principal_values,
        principal_jacobian,
    ):
        """Derivative of ``U diag(value(sigma)) V^T`` in column-major form."""
        dimension = ti.static(singular_values.n)
        size = ti.static(dimension * dimension)
        tangent = ti.Matrix.zero(float, size, size)
        spectral_column = 0
        while spectral_column < size:
            input_column = spectral_column // dimension
            input_row = spectral_column - input_column * dimension
            transformed_direction = ti.Matrix.zero(float, dimension, dimension)
            for row in ti.static(range(dimension)):
                for column in ti.static(range(dimension)):
                    transformed_direction[row, column] = matrix_u[input_row, row] * matrix_v[input_column, column]

            transformed_response = ti.Matrix.zero(float, dimension, dimension)
            for row in ti.static(range(dimension)):
                for column in ti.static(range(dimension)):
                    transformed_response[row, row] += (
                        principal_jacobian[row, column] * transformed_direction[column, column]
                    )
            for row in ti.static(range(dimension)):
                for column in ti.static(range(dimension)):
                    if ti.static(row != column):
                        scale = ti.max(
                            1.0,
                            ti.max(singular_values[row], singular_values[column]),
                        )
                        difference_quotient = 0.0
                        if ti.abs(singular_values[row] - singular_values[column]) <= 1.0e-9 * scale:
                            difference_quotient = principal_jacobian[row, row] - principal_jacobian[row, column]
                        else:
                            difference_quotient = (principal_values[row] - principal_values[column]) / (
                                singular_values[row] - singular_values[column]
                            )
                        sum_quotient = (principal_values[row] + principal_values[column]) / (
                            singular_values[row] + singular_values[column]
                        )
                        transformed_response[row, column] = 0.5 * (
                            (difference_quotient + sum_quotient) * transformed_direction[row, column]
                            + (difference_quotient - sum_quotient) * transformed_direction[column, row]
                        )

            for output_column in ti.static(range(dimension)):
                for output_row in ti.static(range(dimension)):
                    value = 0.0
                    for row, column in ti.static(ti.ndrange(dimension, dimension)):
                        value += (
                            matrix_u[output_row, row]
                            * transformed_response[row, column]
                            * matrix_v[output_column, column]
                        )
                    tangent[
                        output_column * dimension + output_row,
                        spectral_column,
                    ] = value
            spectral_column += 1
        return tangent

    @ti.func
    def _projected_elastic_deformation_tangent(
        self,
        particle_id,
        matrix_u,
        matrix_v,
        singular_values,
        strain,
        trace_strain,
        deviatoric_strain,
        deviatoric_norm,
        projected,
        plastic_multiplier,
        return_region,
    ):
        """Exact derivative of the accepted elastic projection."""
        projected_jacobian = self._projected_strain_jacobian(
            particle_id,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
            plastic_multiplier,
            return_region,
        )
        projected_stretch = ti.Vector.zero(float, singular_values.n)
        principal_jacobian = ti.Matrix.zero(float, 3, 3)
        for row in ti.static(range(3)):
            projected_stretch[row] = ti.exp(projected[row])
            for column in ti.static(range(3)):
                principal_jacobian[row, column] = (
                    projected_stretch[row] * projected_jacobian[row, column] / singular_values[column]
                )
        return self._spectral_matrix_tangent(
            matrix_u,
            matrix_v,
            singular_values,
            projected_stretch,
            principal_jacobian,
        )

    @ti.func
    def commit_total_state_vjp(
        self,
        particle_id,
        total_deformation_gradient,
        total_deformation_output_vjp,
        plastic_inverse_output_vjp,
        equivalent_plastic_strain_output_vjp,
        volumetric_plastic_strain_output_vjp,
    ):
        """Pull an accepted DP/VM state seed back to its pre-commit inputs.

        The material fields must contain the history at the beginning of the
        step.  All matrix entries use the same column-major convention as the
        implicit MPM tangents.
        """
        plastic_inverse = self.plastic_deformation_inverse[particle_id]
        elastic_trial = total_deformation_gradient @ plastic_inverse
        principal = self._principal_trial_state(elastic_trial)
        matrix_u = principal[0]
        matrix_v = principal[1]
        singular_values = principal[2]
        strain = principal[3]
        trace_strain = principal[4]
        deviatoric_strain = principal[5]
        deviatoric_norm = principal[6]
        returned = self._associated_principal_return(
            particle_id,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
        )
        projected = returned[0]
        return_region = returned[3]

        committed_stretch = ti.Matrix.zero(float, 3, 3)
        for i in ti.static(range(3)):
            committed_stretch[i, i] = ti.exp(projected[i])
        committed_elastic = matrix_u @ committed_stretch @ matrix_v.transpose()
        total_inverse = total_deformation_gradient.inverse()
        committed_plastic_inverse = total_inverse @ committed_elastic

        committed_elastic_vjp = total_inverse.transpose() @ plastic_inverse_output_vjp
        total_deformation_vjp = (
            total_deformation_output_vjp
            - total_inverse.transpose() @ plastic_inverse_output_vjp @ committed_plastic_inverse.transpose()
        )
        projection_tangent = self._projected_elastic_deformation_tangent(
            particle_id,
            matrix_u,
            matrix_v,
            singular_values,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
            projected,
            returned[1],
            return_region,
        )
        elastic_trial_vjp = unflatten_matrix(
            projection_tangent.transpose() @ flatten_matrix(committed_elastic_vjp),
            elastic_trial,
        )

        dimension = ti.static(3)
        trace_modulus = ti.static(dimension * self.lame_lambda + 2.0 * self.shear)
        flow_alpha = self.plastic_flow_alpha()
        denominator = (
            2.0 * self.shear
            + dimension * self.alpha * flow_alpha * trace_modulus
            + self.plastic_hardening_modulus(particle_id)
        )
        multiplier_strain_gradient = ti.Vector.zero(float, dimension)
        volume_strain_gradient = ti.Vector.zero(float, dimension)
        multiplier_equivalent_gradient = 0.0
        volume_equivalent_gradient = 0.0
        projected_equivalent_gradient = ti.Vector.zero(float, dimension)
        yield_history_derivative = self.yield_intercept_equivalent_plastic_strain_derivative(particle_id)
        if return_region == 1 and deviatoric_norm > 1.0e-14:
            deviatoric_direction = deviatoric_strain / deviatoric_norm
            flow_direction = deviatoric_direction + flow_alpha
            multiplier_strain_gradient = (
                2.0 * self.shear * deviatoric_direction + self.alpha * trace_modulus
            ) / denominator
            multiplier_equivalent_gradient = -yield_history_derivative / denominator
            volume_strain_gradient = dimension * flow_alpha * multiplier_strain_gradient
            volume_equivalent_gradient = dimension * flow_alpha * multiplier_equivalent_gradient
            projected_equivalent_gradient = yield_history_derivative * flow_direction / denominator
        elif return_region == 2:
            if flow_alpha > 1.0e-14:
                multiplier_strain_gradient = ti.Vector.one(float, dimension) / (dimension * flow_alpha)
            elif deviatoric_norm > 1.0e-14:
                multiplier_strain_gradient = deviatoric_strain / deviatoric_norm
            volume_strain_gradient = ti.Vector.one(float, dimension)

        multiplier_vjp = ti.sqrt(2.0 / 3.0) * equivalent_plastic_strain_output_vjp
        for principal_id in ti.static(range(3)):
            principal_seed = (
                multiplier_vjp * multiplier_strain_gradient[principal_id]
                + volumetric_plastic_strain_output_vjp * volume_strain_gradient[principal_id]
            ) / singular_values[principal_id]
            for row, column in ti.static(ti.ndrange(3, 3)):
                elastic_trial_vjp[row, column] += (
                    principal_seed * matrix_u[row, principal_id] * matrix_v[column, principal_id]
                )

        projected_equivalent_matrix = ti.Matrix.zero(float, 3, 3)
        for principal_id in ti.static(range(3)):
            for row, column in ti.static(ti.ndrange(3, 3)):
                projected_equivalent_matrix[row, column] += (
                    ti.exp(projected[principal_id])
                    * projected_equivalent_gradient[principal_id]
                    * matrix_u[row, principal_id]
                    * matrix_v[column, principal_id]
                )
        equivalent_plastic_strain_vjp = (
            equivalent_plastic_strain_output_vjp
            + multiplier_vjp * multiplier_equivalent_gradient
            + volumetric_plastic_strain_output_vjp * volume_equivalent_gradient
            + contraction(
                committed_elastic_vjp,
                projected_equivalent_matrix,
            )
        )
        total_deformation_vjp += elastic_trial_vjp @ plastic_inverse.transpose()
        plastic_inverse_vjp = total_deformation_gradient.transpose() @ elastic_trial_vjp
        return (
            total_deformation_vjp,
            plastic_inverse_vjp,
            equivalent_plastic_strain_vjp,
            volumetric_plastic_strain_output_vjp,
        )

    @ti.func
    def commit_state(self, particle_id, deformation_gradient):
        (
            matrix_u,
            matrix_v,
            singular_values,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
        ) = self._principal_trial_state(deformation_gradient)
        projected, plastic_multiplier, volume_increment, return_region = self._associated_principal_return(
            particle_id,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
        )
        self.equivalent_plastic_strain[particle_id] += ti.sqrt(2.0 / 3.0) * plastic_multiplier
        self.volumetric_plastic_strain[particle_id] += volume_increment
        projected_stretch = ti.Matrix.zero(float, deformation_gradient.n, deformation_gradient.m)
        for i in ti.static(range(deformation_gradient.n)):
            projected_stretch[i, i] = ti.exp(projected[i])
        return matrix_u @ projected_stretch @ matrix_v.transpose()

    @ti.func
    def strain_energy_density_at(self, particle_id, deformation_gradient):
        principal_state = self._potential_trial_state(particle_id, deformation_gradient)
        strain = principal_state[3]
        trace_strain = principal_state[4]
        deviatoric_strain = principal_state[5]
        deviatoric_norm = principal_state[6]
        return_state = self._potential_principal_return(
            particle_id,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
        )
        projected = return_state[0]
        plastic_multiplier = return_state[1]
        return_region = return_state[3]
        projected_trace = 0.0
        for i in ti.static(range(strain.n)):
            projected_trace += projected[i]
        projected_deviatoric = projected - (projected_trace / strain.n) * ti.Vector.one(float, strain.n)
        energy = (
            self.shear * projected_deviatoric.dot(projected_deviatoric)
            + 0.5 * self.bulk * projected_trace * projected_trace
            + self.current_yield_intercept(particle_id) * plastic_multiplier
            + 0.5 * self.plastic_hardening_modulus(particle_id) * plastic_multiplier * plastic_multiplier
        )
        if return_region == 2:
            # At the cone apex the associated dissipation reduces to a
            # linear volumetric continuation.  This is C1 and retains the
            # classical hydrostatic tensile apex stress.
            energy = 0.5 * self.bulk * self.trace_apex**2 + self.bulk * self.trace_apex * (
                trace_strain - self.trace_apex
            )
        elif return_region == 0:
            energy = self.shear * deviatoric_norm * deviatoric_norm + 0.5 * self.bulk * trace_strain * trace_strain
        return energy

    @ti.func
    def _principal_response(self, particle_id, deformation_gradient):
        (
            matrix_u,
            matrix_v,
            singular_values,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
        ) = self._potential_trial_state(particle_id, deformation_gradient)
        dimension = ti.static(deformation_gradient.n)
        return_state = self._potential_principal_return(
            particle_id,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
        )
        projected = return_state[0]
        plastic_multiplier = return_state[1]
        return_region = return_state[3]
        kirchhoff, stress_operator = self._elastic_principal_response(projected)
        principal_pk1 = ti.Vector.zero(float, dimension)
        for i in ti.static(range(dimension)):
            principal_pk1[i] = kirchhoff[i] / singular_values[i]
        return (
            matrix_u,
            matrix_v,
            singular_values,
            strain,
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
            projected,
            plastic_multiplier,
            return_region,
            kirchhoff,
            principal_pk1,
        )

    @ti.func
    def first_piola_stress_at(self, particle_id, deformation_gradient):
        response = self._principal_response(particle_id, deformation_gradient)
        return self._stress_from_response(deformation_gradient, response)

    @ti.func
    def _stress_from_response(self, deformation_gradient, response):
        matrix_u, matrix_v, principal_pk1 = response[0], response[1], response[11]
        diagonal = ti.Matrix.zero(float, deformation_gradient.n, deformation_gradient.m)
        for i in ti.static(range(deformation_gradient.n)):
            diagonal[i, i] = principal_pk1[i]
        stress = matrix_u @ diagonal @ matrix_v.transpose()
        if ti.static(self.alpha > 1.0e-14):
            if response[9] == 2:
                stress = self.bulk * self.trace_apex * deformation_gradient.inverse().transpose()
        return stress

    @ti.func
    def first_piola_equivalent_plastic_strain_derivative_at(self, particle_id, deformation_gradient):
        """PK1 derivative with respect to the accepted hardening history."""
        response = self._principal_response(particle_id, deformation_gradient)
        matrix_u = response[0]
        matrix_v = response[1]
        singular_values = response[2]
        deviatoric_strain = response[5]
        deviatoric_norm = response[6]
        return_region = response[9]
        derivative = ti.Matrix.zero(float, deformation_gradient.n, deformation_gradient.m)
        yield_derivative = self.yield_intercept_equivalent_plastic_strain_derivative(particle_id)
        if return_region == 1 and deviatoric_norm > 1.0e-14 and yield_derivative != 0.0:
            dimension = ti.static(deformation_gradient.n)
            trace_modulus = ti.static(dimension * self.lame_lambda + 2.0 * self.shear)
            denominator = (
                2.0 * self.shear
                + dimension * self.alpha * self.alpha * trace_modulus
                + self.plastic_hardening_modulus(particle_id)
            )
            flow_direction = deviatoric_strain / deviatoric_norm + self.alpha
            projected_derivative = yield_derivative * flow_direction / denominator
            stress_operator = self._elastic_principal_response(response[7])[1]
            kirchhoff_derivative = stress_operator @ projected_derivative
            diagonal = ti.Matrix.zero(float, deformation_gradient.n, deformation_gradient.m)
            for i in ti.static(range(deformation_gradient.n)):
                diagonal[i, i] = kirchhoff_derivative[i] / singular_values[i]
            derivative = matrix_u @ diagonal @ matrix_v.transpose()
        return derivative

    @ti.func
    def first_piola_tangent_at(self, particle_id, deformation_gradient):
        response = self._principal_response(particle_id, deformation_gradient)
        return self._tangent_from_response(particle_id, deformation_gradient, response)

    @ti.func
    def _tangent_from_response(self, particle_id, deformation_gradient, response):
        matrix_u = response[0]
        matrix_v = response[1]
        singular_values = response[2]
        trace_strain = response[4]
        deviatoric_strain = response[5]
        deviatoric_norm = response[6]
        plastic_multiplier = response[8]
        return_region = response[9]
        kirchhoff = response[10]
        principal_pk1 = response[11]
        dimension = ti.static(deformation_gradient.n)
        inverse_stretch = ti.Vector.zero(float, dimension)
        inverse_stretch_squared = ti.Vector.zero(float, dimension)
        for i in ti.static(range(dimension)):
            inverse_stretch[i] = 1.0 / singular_values[i]
            inverse_stretch_squared[i] = inverse_stretch[i] ** 2

        dprojected_dstrain = self._potential_strain_jacobian(
            particle_id,
            response[3],
            trace_strain,
            deviatoric_strain,
            deviatoric_norm,
            plastic_multiplier,
            return_region,
        )

        dprojected_dstretch = ti.Matrix.zero(float, dimension, dimension)
        for i in ti.static(range(dimension)):
            for j in ti.static(range(dimension)):
                dprojected_dstretch[i, j] = dprojected_dstrain[i, j] * inverse_stretch[j]

        stress_operator = self._elastic_principal_response(response[7])[1]

        principal_jacobian = ti.Matrix.zero(float, dimension, dimension)
        for i in ti.static(range(dimension)):
            for j in ti.static(range(dimension)):
                derivative = 0.0
                for k in ti.static(range(dimension)):
                    derivative += stress_operator[i, k] * dprojected_dstretch[k, j]
                principal_jacobian[i, j] = inverse_stretch[i] * derivative
                if ti.static(i == j):
                    principal_jacobian[i, j] -= kirchhoff[i] * inverse_stretch_squared[i]

        tangent = self._spectral_matrix_tangent(
            matrix_u,
            matrix_v,
            singular_values,
            principal_pk1,
            principal_jacobian,
        )
        if ti.static(self.alpha > 1.0e-14):
            if return_region == 2:
                inverse = deformation_gradient.inverse()
                row = 0
                while row < 9:
                    column = 0
                    while column < 9:
                        tangent[row, column] = (
                            -self.bulk * self.trace_apex * inverse[row // 3, column % 3] * inverse[column // 3, row % 3]
                        )
                        column += 1
                    row += 1
        return tangent

    @ti.func
    def Psi_at(self, particle_id, deformation_gradient):
        return self.strain_energy_density_at(particle_id, deformation_gradient)

    @ti.func
    def dPsi_div_dF_at(self, particle_id, deformation_gradient):
        return flatten_matrix(self.first_piola_stress_at(particle_id, deformation_gradient))

    @ti.func
    def d2Psi_div_d2F_at(self, particle_id, deformation_gradient):
        return self.first_piola_tangent_at(particle_id, deformation_gradient)

    @ti.func
    def VonMises_at(self, particle_id, deformation_gradient):
        stress = self.first_piola_stress_at(particle_id, deformation_gradient)
        return self.von_mises_from_pk1(deformation_gradient, stress)


__all__ = ["HenckyAssociatedPlasticityModel"]
