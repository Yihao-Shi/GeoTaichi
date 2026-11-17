"""Finite-strain isotropic Modified Cam-Clay plasticity.

The model uses ``F = Fe Fp``, principal elastic logarithmic strains, the
classical MCC ellipse, associated flow, and exponential preconsolidation
hardening.  Its pressure-dependent elastic moduli come from one hyperelastic
potential rather than from a hypoelastic stress-rate approximation.
"""

import math

import taichi as ti
from src.utils.MatrixFunction import contraction, flatten_matrix, unflatten_matrix

from src.physics_model.constitutive_model.finite_strain.HenckyPlasticity import (
    HenckyAssociatedPlasticityModel,
)


@ti.data_oriented
class FiniteStrainModifiedCamClayModel(HenckyAssociatedPlasticityModel):
    """Multiplicative MCC with coupled or lagged pressure hardening.

    Continuum Direct MPM--ABD couples return and hardening in one local solve
    and differentiates physical plastic volume. Other consumers retain the
    incremental potential with frozen pressure and a material fixed point.
    """

    def __init__(
        self,
        material_type="Solid",
        configuration="UL",
        solver_type="Implicit",
    ):
        super().__init__(material_type, configuration, solver_type)
        self.has_symmetric_tangent = True
        self.has_incremental_potential = True
        self.requires_lagged_incremental_potential = True
        self.history_state_size = 12
        self.stress_ratio = 0.0
        self.compression_index = 0.0
        self.swelling_index = 0.0
        self.lambda_star = 0.0
        self.kappa_star = 0.0
        self.plastic_compressibility = 0.0
        self.reference_void_ratio = 0.0
        self.initial_pressure = 0.0
        self.initial_preconsolidation_pressure = 0.0
        self.reference_shear_modulus = 0.0
        self.local_tolerance = 1.0e-10
        self.local_max_iterations = 40
        self.lagged_tolerance = 1.0e-8
        self.lagged_max_iterations = 20
        self.preconsolidation_pressure = None
        self.lagged_preconsolidation_pressure = None
        self.use_direct_consistent_solve = False
        self.parameter_count = 5

    def model_initialize(self, material):
        self.material = material
        density = float(self._parameter(material, ("Density", "density"), 2650.0))
        poisson = float(self._parameter(material, ("PoissonRatio", "poisson_ratio"), 0.3))
        stress_ratio = float(
            self._parameter(
                material,
                ("StressRatio", "CriticalStateStressRatio", "M", "stress_ratio"),
                required=True,
            )
        )
        void_ratio = float(
            self._parameter(
                material,
                ("void_ratio_ref", "ReferenceVoidRatio", "InitialVoidRatio"),
                required=True,
            )
        )
        lambda_star_value = self._parameter(material, ("lambda_star", "LambdaStar"), None)
        kappa_star_value = self._parameter(material, ("kappa_star", "KappaStar"), None)
        compression_index = self._parameter(material, ("lambda", "CompressionIndex", "compression_index"), None)
        swelling_index = self._parameter(material, ("kappa", "SwellingIndex", "swelling_index"), None)
        if lambda_star_value is None or kappa_star_value is None:
            if compression_index is None or swelling_index is None:
                raise KeyError("Finite-strain MCC requires lambda/kappa or " "lambda_star/kappa_star")
            compression_index = float(compression_index)
            swelling_index = float(swelling_index)
            lambda_star = compression_index / (1.0 + void_ratio)
            kappa_star = swelling_index / (1.0 + void_ratio)
        else:
            lambda_star = float(lambda_star_value)
            kappa_star = float(kappa_star_value)
            compression_index = lambda_star * (1.0 + void_ratio)
            swelling_index = kappa_star * (1.0 + void_ratio)

        pc0 = float(
            self._parameter(
                material,
                (
                    "pc",
                    "pc0",
                    "ConsolidationPressure",
                    "PreconsolidationPressure",
                ),
                required=True,
            )
        )
        ocr = float(self._parameter(material, ("OverConsolidationRatio", "OCR", "ocr"), 1.0))
        if not math.isfinite(ocr) or ocr < 1.0:
            raise ValueError("OverConsolidationRatio must be at least one")
        initial_pressure_value = self._parameter(
            material,
            ("InitialPressure", "initial_pressure", "pressure_initial"),
            None,
        )
        initial_pressure = pc0 / ocr if initial_pressure_value is None else float(initial_pressure_value)
        local_tolerance = float(
            self._parameter(
                material,
                ("LocalTolerance", "local_tolerance"),
                self.local_tolerance,
            )
        )
        local_iterations = int(
            self._parameter(
                material,
                ("LocalMaxIterations", "local_max_iterations"),
                self.local_max_iterations,
            )
        )
        lagged_tolerance = float(
            self._parameter(
                material,
                ("LaggedTolerance", "lagged_tolerance"),
                self.lagged_tolerance,
            )
        )
        lagged_iterations = int(
            self._parameter(
                material,
                ("LaggedMaxIterations", "lagged_max_iterations"),
                self.lagged_max_iterations,
            )
        )

        if not math.isfinite(density) or density <= 0.0:
            raise ValueError("Density must be finite and positive")
        if not math.isfinite(poisson) or not -1.0 < poisson < 0.5:
            raise ValueError("PoissonRatio must satisfy -1 < nu < 0.5")
        positive = {
            "StressRatio": stress_ratio,
            "lambda_star": lambda_star,
            "kappa_star": kappa_star,
            "pc0": pc0,
            "InitialPressure": initial_pressure,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(void_ratio) or void_ratio <= -1.0:
            raise ValueError("void_ratio_ref must be finite and greater than -1")
        if lambda_star <= kappa_star:
            raise ValueError("MCC requires lambda_star > kappa_star")
        if initial_pressure > pc0 * (1.0 + 1.0e-12):
            raise ValueError("InitialPressure must not exceed pc0")
        if not math.isfinite(local_tolerance) or local_tolerance <= 0.0:
            raise ValueError("LocalTolerance must be finite and positive")
        if local_iterations <= 0:
            raise ValueError("LocalMaxIterations must be positive")
        if not math.isfinite(lagged_tolerance) or lagged_tolerance <= 0.0:
            raise ValueError("LaggedTolerance must be finite and positive")
        if lagged_iterations <= 0:
            raise ValueError("LaggedMaxIterations must be positive")

        self.density = density
        self.poisson = poisson
        self.stress_ratio = stress_ratio
        self.reference_void_ratio = void_ratio
        self.compression_index = compression_index
        self.swelling_index = swelling_index
        self.lambda_star = lambda_star
        self.kappa_star = kappa_star
        self.plastic_compressibility = lambda_star - kappa_star
        self.initial_pressure = initial_pressure
        self.initial_preconsolidation_pressure = pc0
        self.local_tolerance = local_tolerance
        self.local_max_iterations = local_iterations
        self.lagged_tolerance = lagged_tolerance
        self.lagged_max_iterations = lagged_iterations

        self.bulk = initial_pressure / kappa_star
        self.shear = 3.0 * self.bulk * (1.0 - 2.0 * poisson) / (2.0 * (1.0 + poisson))
        self.reference_shear_modulus = self.shear
        self.young = 9.0 * self.bulk * self.shear / (3.0 * self.bulk + self.shear)
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)
        self.add_coupling_material(material)

    def allocate_state(self, particle_count):
        super().allocate_state(particle_count)
        self.preconsolidation_pressure = ti.field(ti.f64, shape=int(particle_count))
        self.lagged_preconsolidation_pressure = ti.field(ti.f64, shape=int(particle_count))
        self.preconsolidation_pressure.fill(self.initial_preconsolidation_pressure)
        self.lagged_preconsolidation_pressure.fill(self.initial_preconsolidation_pressure)

    @ti.func
    def begin_lagged_incremental_potential(self, particle_id):
        self.lagged_preconsolidation_pressure[particle_id] = self.preconsolidation_pressure[particle_id]
        self.begin_lagged_plastic_volume(particle_id)

    @ti.func
    def refresh_lagged_incremental_potential(self, particle_id, total_deformation_gradient):
        elastic_trial = self.trial_elastic_deformation(particle_id, total_deformation_gradient)
        principal = self._principal_trial_state(elastic_trial)
        solution = self._solve_return_variables(
            principal[4],
            principal[6],
            self.lagged_preconsolidation_pressure[particle_id],
        )
        plastic_compression = solution[0] - principal[4]
        candidate = self.preconsolidation_pressure[particle_id] * ti.exp(
            ti.max(-50.0, ti.min(50.0, plastic_compression / self.plastic_compressibility))
        )
        previous = self.lagged_preconsolidation_pressure[particle_id]
        error = ti.abs(ti.log(candidate / previous))
        volume_error = self.refresh_lagged_plastic_volume(particle_id, total_deformation_gradient)
        self.lagged_preconsolidation_pressure[particle_id] = candidate
        return ti.max(error, volume_error)

    @ti.func
    def get_history_state(self, particle_id):
        state = ti.Vector.zero(float, 12)
        state[0] = self.equivalent_plastic_strain[particle_id]
        state[1] = self.volumetric_plastic_strain[particle_id]
        state[2] = self.preconsolidation_pressure[particle_id]
        for column, row in ti.static(ti.ndrange(3, 3)):
            state[3 + 3 * column + row] = self.plastic_deformation_inverse[particle_id][row, column]
        return state

    @ti.func
    def set_history_state(self, particle_id, history_state):
        self.equivalent_plastic_strain[particle_id] = history_state[0]
        self.volumetric_plastic_strain[particle_id] = history_state[1]
        self.preconsolidation_pressure[particle_id] = history_state[2]
        self.lagged_preconsolidation_pressure[particle_id] = history_state[2]
        for column, row in ti.static(ti.ndrange(3, 3)):
            self.plastic_deformation_inverse[particle_id][row, column] = history_state[3 + 3 * column + row]
        self.end_lagged_plastic_volume(particle_id)

    @ti.func
    def _elastic_invariants(self, trace_strain, deviatoric_norm):
        exponent = (
            -trace_strain / self.kappa_star
            + self.reference_shear_modulus
            * deviatoric_norm
            * deviatoric_norm
            / (self.kappa_star * self.initial_pressure)
        )
        scale = ti.exp(ti.max(-50.0, ti.min(50.0, exponent)))
        pressure = self.initial_pressure * scale
        shear_modulus = self.reference_shear_modulus * scale
        q = ti.sqrt(6.0) * shear_modulus * deviatoric_norm
        return scale, pressure, shear_modulus, q

    @ti.func
    def _local_system(
        self,
        trace_strain,
        deviatoric_norm,
        plastic_multiplier,
        trial_trace,
        trial_deviatoric_norm,
        pc,
    ):
        scale, pressure, shear_modulus, q = self._elastic_invariants(trace_strain, deviatoric_norm)
        flow_volume = 2.0 * pressure - pc
        shear_factor = 6.0 * shear_modulus / (self.stress_ratio**2)
        residual = ti.Vector.zero(float, 3)
        residual[0] = trace_strain - trial_trace - plastic_multiplier * flow_volume
        residual[1] = deviatoric_norm * (1.0 + shear_factor * plastic_multiplier) - trial_deviatoric_norm
        residual[2] = q * q / (self.stress_ratio**2) + pressure * (pressure - pc)

        pressure_trace = (
            -pressure
            / self.kappa_star
            * (1.0 if not self.use_direct_consistent_solve or ti.exp(-50.0) < scale < ti.exp(50.0) else 0.0)
        )
        pressure_deviatoric = (
            pressure * 2.0 * self.reference_shear_modulus * deviatoric_norm / (self.kappa_star * self.initial_pressure)
        ) * (1.0 if not self.use_direct_consistent_solve or ti.exp(-50.0) < scale < ti.exp(50.0) else 0.0)
        shear_trace = (
            -shear_modulus
            / self.kappa_star
            * (1.0 if not self.use_direct_consistent_solve or ti.exp(-50.0) < scale < ti.exp(50.0) else 0.0)
        )
        shear_deviatoric = (
            shear_modulus
            * 2.0
            * self.reference_shear_modulus
            * deviatoric_norm
            / (self.kappa_star * self.initial_pressure)
        ) * (1.0 if not self.use_direct_consistent_solve or ti.exp(-50.0) < scale < ti.exp(50.0) else 0.0)
        q_trace = (
            -q
            / self.kappa_star
            * (1.0 if not self.use_direct_consistent_solve or ti.exp(-50.0) < scale < ti.exp(50.0) else 0.0)
        )
        q_deviatoric = ti.sqrt(6.0) * (shear_deviatoric * deviatoric_norm + shear_modulus)
        flow_trace = 2.0 * pressure_trace
        flow_deviatoric = 2.0 * pressure_deviatoric
        shear_factor_trace = 6.0 * shear_trace / (self.stress_ratio**2)
        shear_factor_deviatoric = 6.0 * shear_deviatoric / (self.stress_ratio**2)

        jacobian = ti.Matrix.zero(float, 3, 3)
        jacobian[0, 0] = 1.0 - plastic_multiplier * flow_trace
        jacobian[0, 1] = -plastic_multiplier * flow_deviatoric
        jacobian[0, 2] = -flow_volume
        jacobian[1, 0] = deviatoric_norm * plastic_multiplier * shear_factor_trace
        jacobian[1, 1] = (
            1.0 + shear_factor * plastic_multiplier + deviatoric_norm * plastic_multiplier * shear_factor_deviatoric
        )
        jacobian[1, 2] = deviatoric_norm * shear_factor
        jacobian[2, 0] = 2.0 * q * q_trace / (self.stress_ratio**2) + flow_volume * pressure_trace
        jacobian[2, 1] = 2.0 * q * q_deviatoric / (self.stress_ratio**2) + flow_volume * pressure_deviatoric
        return residual, jacobian, pressure, q

    @ti.func
    def _local_residual_norm(self, residual, pc):
        norm = ti.max(
            ti.abs(residual[0]),
            ti.abs(residual[1]),
        )
        return ti.max(norm, ti.abs(residual[2]) / (pc * pc))

    @ti.func
    def _consistent_local_system(self, state, trial_trace, trial_norm, accepted_pc):
        residual3, jacobian3, pressure, q = self._local_system(
            state[0], state[1], state[2], trial_trace, trial_norm, state[3]
        )
        residual = ti.Vector(
            [
                residual3[0],
                residual3[1],
                residual3[2],
                ti.log(state[3] / accepted_pc)
                - ti.max(-50.0, ti.min(50.0, (state[0] - trial_trace) / self.plastic_compressibility)),
            ]
        )
        jacobian = ti.Matrix.zero(float, 4, 4)
        for i, j in ti.static(ti.ndrange(3, 3)):
            jacobian[i, j] = jacobian3[i, j]
        jacobian[0, 3] = state[2]
        jacobian[2, 3] = -pressure
        jacobian[3, 0] = (
            -1.0 / self.plastic_compressibility
            if -50.0 < (state[0] - trial_trace) / self.plastic_compressibility < 50.0
            else 0.0
        )
        jacobian[3, 3] = 1.0 / state[3]
        return residual, jacobian

    @ti.func
    def _consistent_residual_norm(self, residual, pc):
        return ti.max(self._local_residual_norm(residual, pc), ti.abs(residual[3]))

    @ti.func
    def _solve_consistent_return(self, trial_trace, trial_norm, accepted_pc):
        _, p, _, q = self._elastic_invariants(trial_trace, trial_norm)
        region = 0
        state = ti.Vector([trial_trace, trial_norm, 0.0, accepted_pc])
        if q * q / self.stress_ratio**2 + p * (p - accepted_pc) > self.local_tolerance * accepted_pc**2:
            region = 1
            iteration = 0
            while iteration < ti.static(self.local_max_iterations):
                residual, jacobian = self._consistent_local_system(state, trial_trace, trial_norm, accepted_pc)
                norm = self._consistent_residual_norm(residual, accepted_pc)
                if norm <= self.local_tolerance:
                    break
                correction = jacobian.inverse() @ (-residual)
                alpha = 1.0
                accepted = False
                for backtrack in range(24):
                    candidate = state + alpha * correction
                    if candidate[1] >= 0.0 and candidate[2] >= 0.0 and candidate[3] > 0.0:
                        candidate_residual, _ = self._consistent_local_system(
                            candidate, trial_trace, trial_norm, accepted_pc
                        )
                        if self._consistent_residual_norm(candidate_residual, accepted_pc) < norm:
                            accepted = True
                            break
                    alpha *= 0.5
                assert accepted, "finite-strain MCC coupled return line search failed"
                state += alpha * correction
                iteration += 1
            residual, _ = self._consistent_local_system(state, trial_trace, trial_norm, accepted_pc)
            assert (
                self._consistent_residual_norm(residual, accepted_pc) <= 10.0 * self.local_tolerance
            ), "finite-strain MCC coupled return mapping failed"
        return state[0], state[1], state[3], state[2], region

    @ti.func
    def _solve_return_variables(self, trial_trace, trial_deviatoric_norm, pc):
        trial_scale, trial_pressure, trial_shear, trial_q = self._elastic_invariants(trial_trace, trial_deviatoric_norm)
        trial_yield = trial_q * trial_q / (self.stress_ratio**2) + trial_pressure * (trial_pressure - pc)
        trace_strain = trial_trace
        deviatoric_norm = trial_deviatoric_norm
        plastic_multiplier = 0.0
        return_region = 0
        if trial_yield > self.local_tolerance * pc * pc and ti.static(not self.use_direct_consistent_solve):
            return_region = 1
            converged = 0
            residual = ti.Vector.zero(float, 3)
            local_iteration = 0
            while local_iteration < ti.static(self.local_max_iterations):
                residual, jacobian, current_pressure, current_q = self._local_system(
                    trace_strain,
                    deviatoric_norm,
                    plastic_multiplier,
                    trial_trace,
                    trial_deviatoric_norm,
                    pc,
                )
                residual_norm = self._local_residual_norm(residual, pc)
                if residual_norm <= self.local_tolerance:
                    converged = 1
                    break
                increment = jacobian.inverse() @ (-residual)
                step = 1.0
                accepted = 0
                backtrack = 0
                while backtrack < 16:
                    candidate_trace = trace_strain + step * increment[0]
                    candidate_deviatoric = deviatoric_norm + step * increment[1]
                    candidate_multiplier = plastic_multiplier + step * increment[2]
                    if accepted == 0 and candidate_deviatoric >= 0.0 and candidate_multiplier >= 0.0:
                        candidate_residual = self._local_system(
                            candidate_trace,
                            candidate_deviatoric,
                            candidate_multiplier,
                            trial_trace,
                            trial_deviatoric_norm,
                            pc,
                        )[0]
                        if self._local_residual_norm(candidate_residual, pc) < residual_norm:
                            accepted = 1
                    if accepted == 0:
                        step *= 0.5
                    backtrack += 1
                trace_strain += step * increment[0]
                deviatoric_norm += step * increment[1]
                plastic_multiplier += step * increment[2]
                local_iteration += 1
            if converged == 0:
                residual = self._local_system(
                    trace_strain,
                    deviatoric_norm,
                    plastic_multiplier,
                    trial_trace,
                    trial_deviatoric_norm,
                    pc,
                )[0]
                assert (
                    self._local_residual_norm(residual, pc) <= 10.0 * self.local_tolerance
                ), "finite-strain MCC local return mapping failed"
        if ti.static(self.use_direct_consistent_solve):
            trace_strain, deviatoric_norm, pc, plastic_multiplier, return_region = self._solve_consistent_return(
                trial_trace, trial_deviatoric_norm, pc
            )
        return (
            trace_strain,
            deviatoric_norm,
            pc,
            plastic_multiplier,
            return_region,
        )

    @ti.func
    def _associated_principal_return(
        self,
        particle_id,
        strain,
        trace_strain,
        deviatoric_strain,
        deviatoric_norm,
    ):
        solution = self._solve_return_variables(
            trace_strain,
            deviatoric_norm,
            (
                self.preconsolidation_pressure[particle_id]
                if ti.static(self.use_direct_consistent_solve)
                else self.lagged_preconsolidation_pressure[particle_id]
            ),
        )
        projected_trace = solution[0]
        projected_norm = solution[1]
        projected = (projected_trace / 3.0) * ti.Vector.one(float, 3)
        if deviatoric_norm > 1.0e-14:
            projected += (projected_norm / deviatoric_norm) * deviatoric_strain
        plastic_volume_increment = projected_trace - trace_strain
        return (
            projected,
            solution[3],
            plastic_volume_increment,
            solution[4],
        )

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
        derivative = ti.Matrix.identity(float, 3)
        if return_region == 1:
            solution = self._solve_return_variables(
                trace_strain,
                deviatoric_norm,
                (
                    self.preconsolidation_pressure[particle_id]
                    if ti.static(self.use_direct_consistent_solve)
                    else self.lagged_preconsolidation_pressure[particle_id]
                ),
            )
            projected_trace = solution[0]
            projected_norm = solution[1]
            local_residual, jacobian, current_pressure, current_q = self._local_system(
                projected_trace,
                projected_norm,
                solution[3],
                trace_strain,
                deviatoric_norm,
                solution[2],
            )
            inverse_jacobian = jacobian.inverse()
            trace_from_trace = inverse_jacobian[0, 0]
            norm_from_trace = inverse_jacobian[1, 0]
            trace_from_norm = inverse_jacobian[0, 1]
            norm_from_norm = inverse_jacobian[1, 1]
            if ti.static(self.use_direct_consistent_solve):
                state = ti.Vector([projected_trace, projected_norm, solution[3], solution[2]])
                _, coupled_jacobian = self._consistent_local_system(
                    state, trace_strain, deviatoric_norm, self.preconsolidation_pressure[particle_id]
                )
                inverse = coupled_jacobian.inverse()
                trace_from_trace = inverse[0, 0] + inverse[0, 3] * (
                    -1.0 / self.plastic_compressibility
                    if -50.0 < (projected_trace - trace_strain) / self.plastic_compressibility < 50.0
                    else 0.0
                )
                norm_from_trace = inverse[1, 0] + inverse[1, 3] * (
                    -1.0 / self.plastic_compressibility
                    if -50.0 < (projected_trace - trace_strain) / self.plastic_compressibility < 50.0
                    else 0.0
                )
                trace_from_norm = inverse[0, 1]
                norm_from_norm = inverse[1, 1]
            derivative = ti.Matrix.zero(float, 3, 3)
            if deviatoric_norm > 1.0e-14:
                direction = deviatoric_strain / deviatoric_norm
                scale = projected_norm / deviatoric_norm
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        identity = 1.0 if ti.static(i == j) else 0.0
                        deviatoric_projector = identity - 1.0 / 3.0
                        derivative[i, j] = (
                            (trace_from_trace + trace_from_norm * direction[j]) / 3.0
                            + direction[i] * (norm_from_trace + norm_from_norm * direction[j])
                            + scale * (deviatoric_projector - direction[i] * direction[j])
                        )
            else:
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        identity = 1.0 if ti.static(i == j) else 0.0
                        derivative[i, j] = trace_from_trace / 3.0 + norm_from_norm * (identity - 1.0 / 3.0)
        return derivative

    @ti.func
    def _elastic_principal_response(self, projected):
        trace_strain = projected[0] + projected[1] + projected[2]
        deviatoric = projected - (trace_strain / 3.0)
        deviatoric_norm = deviatoric.norm()
        scale, pressure, shear_modulus, q = self._elastic_invariants(trace_strain, deviatoric_norm)
        base = 2.0 * self.reference_shear_modulus * deviatoric - pressure / scale
        kirchhoff = scale * base
        operator = ti.Matrix.zero(float, 3, 3)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                identity = 1.0 if ti.static(i == j) else 0.0
                operator[i, j] = scale * (
                    2.0 * self.reference_shear_modulus * (identity - 1.0 / 3.0)
                    + base[i]
                    * base[j]
                    / (self.kappa_star * self.initial_pressure)
                    * (1.0 if not self.use_direct_consistent_solve or ti.exp(-50.0) < scale < ti.exp(50.0) else 0.0)
                )
        return kirchhoff, operator

    @ti.func
    def _elastic_differential(self, trace, norm, dtrace, dnorm, parameter: ti.template()):
        _, pressure, shear, _ = self._elastic_invariants(trace, norm)
        dnu, dkappa, dp0 = 0.0, 0.0, 0.0
        if ti.static(parameter == 0):
            dnu = 1.0
        elif ti.static(parameter == 3):
            dkappa = 1.0
        elif ti.static(parameter == 4):
            dp0 = 1.0
        dG0 = self.reference_shear_modulus * (
            dp0 / self.initial_pressure
            - dkappa / self.kappa_star
            - (2.0 / (1.0 - 2.0 * self.poisson) + 1.0 / (1.0 + self.poisson)) * dnu
        )
        term = self.reference_shear_modulus * norm * norm / (self.kappa_star * self.initial_pressure)
        exponent = -trace / self.kappa_star + term
        dexponent = 0.0
        if -50.0 < exponent < 50.0:
            dexponent = (
                -dtrace / self.kappa_star
                + trace * dkappa / self.kappa_star**2
                + dG0 * norm * norm / (self.kappa_star * self.initial_pressure)
                + 2.0 * self.reference_shear_modulus * norm * dnorm / (self.kappa_star * self.initial_pressure)
                - term * (dkappa / self.kappa_star + dp0 / self.initial_pressure)
            )
        dp = pressure * (dp0 / self.initial_pressure + dexponent)
        dG = shear * (dG0 / self.reference_shear_modulus + dexponent)
        return dp, dG, ti.sqrt(6.0) * (dG * norm + shear * dnorm)

    @ti.func
    def _consistent_return_derivatives(self, particle_id, trace, norm, solution):
        """Columns: trial trace/norm, accepted pc, nu, M, lambda*, kappa*, p0."""
        derivative = ti.Matrix.zero(float, 4, 8)
        derivative[0, 0], derivative[1, 1], derivative[3, 2] = 1.0, 1.0, 1.0
        if solution[4] == 1:
            state = ti.Vector([solution[0], solution[1], solution[3], solution[2]])
            _, jacobian = self._consistent_local_system(state, trace, norm, self.preconsolidation_pressure[particle_id])
            rhs = ti.Matrix.zero(float, 4, 8)
            rhs[0, 0], rhs[1, 1] = 1.0, 1.0
            rhs[3, 0] = (
                -1.0 / self.plastic_compressibility
                if -50.0 < (solution[0] - trace) / self.plastic_compressibility < 50.0
                else 0.0
            )
            rhs[3, 2] = 1.0 / self.preconsolidation_pressure[particle_id]
            _, pressure, shear, q = self._elastic_invariants(solution[0], solution[1])
            for parameter in ti.static(range(5)):
                dp, dG, dq = self._elastic_differential(solution[0], solution[1], 0.0, 0.0, parameter)
                dM = 1.0 if ti.static(parameter == 1) else 0.0
                dcompressibility = 0.0
                if ti.static(parameter == 2):
                    dcompressibility = 1.0
                elif ti.static(parameter == 3):
                    dcompressibility = -1.0
                rhs[0, 3 + parameter] = 2.0 * solution[3] * dp
                rhs[1, 3 + parameter] = (
                    -solution[1]
                    * solution[3]
                    * 6.0
                    * shear
                    / self.stress_ratio**2
                    * (dG / shear - 2.0 * dM / self.stress_ratio)
                )
                rhs[2, 3 + parameter] = -(
                    2.0 * q * dq / self.stress_ratio**2
                    - 2.0 * q * q * dM / self.stress_ratio**3
                    + (2.0 * pressure - solution[2]) * dp
                )
                rhs[3, 3 + parameter] = (
                    -(solution[0] - trace) * dcompressibility / self.plastic_compressibility**2
                    if -50.0 < (solution[0] - trace) / self.plastic_compressibility < 50.0
                    else 0.0
                )
            derivative = jacobian.inverse() @ rhs
        return derivative

    @ti.func
    def _return_differentials(self, particle_id, elastic_trial):
        """Exact local linearization; no differentiation through Newton iterations."""
        principal = self._principal_trial_state(elastic_trial)
        solution = self._solve_return_variables(principal[4], principal[6], self.preconsolidation_pressure[particle_id])
        dz = self._consistent_return_derivatives(particle_id, principal[4], principal[6], solution)
        direction = ti.Vector.zero(float, 3)
        if principal[6] > 1e-14:
            direction = principal[5] / principal[6]
        projected = solution[0] / 3.0 + solution[1] * direction
        stretch = ti.Matrix.zero(float, 3, 3)
        for i in ti.static(range(3)):
            stretch[i, i] = ti.exp(projected[i])
        projected_elastic = principal[0] @ stretch @ principal[1].transpose()
        _, _, _, q = self._elastic_invariants(solution[0], solution[1])
        scalar_derivative = ti.Matrix.zero(float, 3, 8)
        external_stress = ti.Matrix.zero(float, 9, 6)
        external_projection = ti.Matrix.zero(float, 9, 6)
        for column in ti.static(range(8)):
            parameter = ti.static(column - 3)
            dp, dG, dq = self._elastic_differential(solution[0], solution[1], dz[0, column], dz[1, column], parameter)
            dM = 1.0 if ti.static(column == 4) else 0.0
            scalar_derivative[0, column] = (
                2.0 * (q * dz[2, column] + solution[3] * dq) / self.stress_ratio**2
                - 4.0 * solution[3] * q * dM / self.stress_ratio**3
            )
            scalar_derivative[1, column] = dz[0, column] - (1.0 if ti.static(column == 0) else 0.0)
            scalar_derivative[2, column] = dz[3, column]
            if ti.static(column >= 2):
                dprojected = dz[0, column] / 3.0 + dz[1, column] * direction
                dtau = (
                    2.0 * dG * solution[1] * direction
                    + 2.0 * self._elastic_invariants(solution[0], solution[1])[2] * dz[1, column] * direction
                    - dp
                )
                dP, dFe = ti.Matrix.zero(float, 3, 3), ti.Matrix.zero(float, 3, 3)
                for i, row, col in ti.static(ti.ndrange(3, 3, 3)):
                    basis = principal[0][row, i] * principal[1][col, i]
                    dP[row, col] += basis * dtau[i] / principal[2][i]
                    dFe[row, col] += basis * ti.exp(projected[i]) * dprojected[i]
                for row in ti.static(range(9)):
                    external_stress[row, column - 2] = flatten_matrix(dP)[row]
                    external_projection[row, column - 2] = flatten_matrix(dFe)[row]
        return (
            principal,
            solution,
            projected,
            projected_elastic,
            scalar_derivative,
            external_stress,
            external_projection,
        )

    @ti.func
    def strain_energy_density_at(self, particle_id, deformation_gradient):
        principal = self._principal_trial_state(deformation_gradient)
        returned = self._associated_principal_return(
            particle_id,
            principal[3],
            principal[4],
            principal[5],
            principal[6],
        )
        projected = returned[0]
        trace_strain = projected[0] + projected[1] + projected[2]
        deviatoric = projected - trace_strain / 3.0
        exponent = -trace_strain / self.kappa_star + self.reference_shear_modulus * deviatoric.dot(deviatoric) / (
            self.kappa_star * self.initial_pressure
        )
        elastic_energy = self.kappa_star * self.initial_pressure * (ti.exp(ti.max(-50.0, ti.min(50.0, exponent))) - 1.0)
        plastic_compression = trace_strain - principal[4]
        plastic_deviatoric = ti.max(0.0, principal[6] - deviatoric.norm())
        # Support function of q^2 / M^2 + p (p - pc) <= 0 evaluated at
        # the plastic logarithmic-strain increment.
        support_norm = ti.sqrt(
            plastic_compression * plastic_compression
            + (2.0 / 3.0) * self.stress_ratio**2 * plastic_deviatoric * plastic_deviatoric
        )
        dissipation = 0.5 * self.lagged_preconsolidation_pressure[particle_id] * (plastic_compression + support_norm)
        return elastic_energy + dissipation

    @ti.func
    def total_stress_from_response(self, particle_id, deformation, response):
        value = ti.Vector.zero(float, 9)
        if ti.static(self.use_direct_consistent_solve):
            value = self._physical_total_stress_from_response(particle_id, deformation, response)
        else:
            value = HenckyAssociatedPlasticityModel.total_stress_from_response(self, particle_id, deformation, response)
        return value

    @ti.func
    def total_tangent_from_response(self, particle_id, deformation, response):
        value = ti.Matrix.zero(float, 9, 9)
        if ti.static(self.use_direct_consistent_solve):
            value = self._physical_total_tangent_from_response(particle_id, deformation, response)
        else:
            value = HenckyAssociatedPlasticityModel.total_tangent_from_response(
                self, particle_id, deformation, response
            )
        return value

    @ti.func
    def total_first_piola_stress_at(self, particle_id, deformation):
        response = self._principal_response(particle_id, self.trial_elastic_deformation(particle_id, deformation))
        return unflatten_matrix(self.total_stress_from_response(particle_id, deformation, response), deformation)

    @ti.func
    def total_first_piola_tangent_at(self, particle_id, deformation):
        response = self._principal_response(particle_id, self.trial_elastic_deformation(particle_id, deformation))
        return self.total_tangent_from_response(particle_id, deformation, response)

    @ti.func
    def physical_return_vjp(
        self,
        particle_id,
        deformation,
        deformation_seed,
        inverse_seed,
        equivalent_seed,
        volumetric_seed,
        pressure_seed,
        stress_seed,
        equilibrium: ti.template(),
    ):
        """VJP of physical total stress or the accepted MCC state."""
        ti.static_assert(self.use_direct_consistent_solve)
        inverse = self.plastic_deformation_inverse[particle_id]
        elastic = deformation @ inverse
        (
            principal,
            solution,
            projected,
            projected_elastic,
            scalar_derivative,
            external_stress,
            external_projection,
        ) = self._return_differentials(particle_id, elastic)
        total_inverse = deformation.inverse()
        committed_inverse = total_inverse @ projected_elastic
        projected_seed = total_inverse.transpose() @ inverse_seed
        total_seed = deformation_seed - total_inverse.transpose() @ inverse_seed @ committed_inverse.transpose()
        scalar_seed = ti.Vector([equivalent_seed, volumetric_seed, pressure_seed])
        local_seed = scalar_derivative.transpose() @ scalar_seed
        elastic_seed = ti.Matrix.zero(float, 3, 3)
        inverse_input_seed = ti.Matrix.zero(float, 3, 3)
        external_seed = ti.Vector.zero(float, 6)
        if ti.static(equilibrium):
            jacobian = deformation.determinant() * ti.exp(-solution[0])
            pk1 = self.first_piola_stress_at(particle_id, elastic)
            stress_input_seed = jacobian * stress_seed @ inverse
            elastic_seed = unflatten_matrix(
                self.first_piola_tangent_at(particle_id, elastic).transpose() @ flatten_matrix(stress_input_seed),
                elastic,
            )
            inverse_input_seed = jacobian * stress_seed.transpose() @ pk1
            volume_seed = jacobian * contraction(stress_seed, pk1 @ inverse.transpose())
            total_seed += volume_seed * total_inverse.transpose()
            for column in ti.static(range(8)):
                local_seed[column] -= volume_seed * (
                    scalar_derivative[1, column] + (1.0 if ti.static(column == 0) else 0.0)
                )
            external_seed += external_stress.transpose() @ flatten_matrix(stress_input_seed)
        else:
            projection = self._projected_elastic_deformation_tangent(
                particle_id,
                principal[0],
                principal[1],
                principal[2],
                principal[3],
                principal[4],
                principal[5],
                principal[6],
                projected,
                solution[3],
                solution[4],
            )
            elastic_seed += unflatten_matrix(projection.transpose() @ flatten_matrix(projected_seed), elastic)
            external_seed += external_projection.transpose() @ flatten_matrix(projected_seed)
        trace_gradient = elastic.inverse().transpose()
        norm_gradient = ti.Matrix.zero(float, 3, 3)
        if principal[6] > 1e-14:
            for i, row, column in ti.static(ti.ndrange(3, 3, 3)):
                norm_gradient[row, column] += (
                    principal[5][i] / (principal[6] * principal[2][i]) * principal[0][row, i] * principal[1][column, i]
                )
        elastic_seed += local_seed[0] * trace_gradient + local_seed[1] * norm_gradient
        total_seed += elastic_seed @ inverse.transpose()
        inverse_input_seed += deformation.transpose() @ elastic_seed
        for column in ti.static(range(6)):
            external_seed[column] += local_seed[column + 2]
        parameters = ti.Vector([external_seed[i + 1] for i in ti.static(range(5))])
        return total_seed, inverse_input_seed, equivalent_seed, volumetric_seed, external_seed[0], parameters

    @ti.func
    def first_piola_equivalent_plastic_strain_derivative_at(self, particle_id, deformation):
        return ti.Matrix.zero(float, 3, 3)

    @ti.func
    def total_first_piola_parameter_vjp_at(self, particle_id, deformation, adjoint):
        zero = ti.Matrix.zero(float, 3, 3)
        return self.physical_return_vjp(particle_id, deformation, zero, zero, 0.0, 0.0, 0.0, adjoint, True)[5]

    @ti.func
    def commit_state(self, particle_id, deformation_gradient):
        principal = self._principal_trial_state(deformation_gradient)
        solution = self._solve_return_variables(
            principal[4],
            principal[6],
            (
                self.preconsolidation_pressure[particle_id]
                if ti.static(self.use_direct_consistent_solve)
                else self.lagged_preconsolidation_pressure[particle_id]
            ),
        )
        projected_trace = solution[0]
        projected_norm = solution[1]
        projected = (projected_trace / 3.0) * ti.Vector.one(float, 3)
        if principal[6] > 1.0e-14:
            projected += (projected_norm / principal[6]) * principal[5]
        plastic_volume_increment = projected_trace - principal[4]
        scale, pressure, shear_modulus, q = self._elastic_invariants(projected_trace, projected_norm)
        self.volumetric_plastic_strain[particle_id] += plastic_volume_increment
        self.equivalent_plastic_strain[particle_id] += 2.0 * solution[3] * q / (self.stress_ratio**2)
        self.preconsolidation_pressure[particle_id] *= ti.exp(
            ti.max(-50.0, ti.min(50.0, plastic_volume_increment / self.plastic_compressibility))
        )
        self.lagged_preconsolidation_pressure[particle_id] = self.preconsolidation_pressure[particle_id]
        projected_stretch = ti.Matrix.zero(float, 3, 3)
        for i in ti.static(range(3)):
            projected_stretch[i, i] = ti.exp(projected[i])
        return principal[0] @ projected_stretch @ principal[1].transpose()

    def print_message(self, materialID):
        self.print_console_header()
        print("Constitutive model: finite-strain Modified Cam-Clay")
        print("Material ID: ", materialID)
        print("Critical-state stress ratio: ", self.stress_ratio)
        print("lambda*: ", self.lambda_star)
        print("kappa*: ", self.kappa_star)
        print("Initial mean pressure: ", self.initial_pressure)
        print(
            "Initial preconsolidation pressure: ",
            self.initial_preconsolidation_pressure,
            "\n",
        )


ModifiedCamClayModel = FiniteStrainModifiedCamClayModel


__all__ = ["FiniteStrainModifiedCamClayModel", "ModifiedCamClayModel"]
