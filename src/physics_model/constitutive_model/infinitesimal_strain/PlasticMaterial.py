from src.physics_model.constitutive_model.infinitesimal_strain.ElasticMaterial import ElasticMaterial

from src.utils.errors import ConfigurationError, SimulationStateError
import taichi as ti
from src.physics_model.constitutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.constitutive_model.SoftenModel import *
from src.physics_model.constitutive_model.infinitesimal_strain.RateDependent import *
from src.utils.constants import LTOL, MAXITS, NSUB, EPS, STOL, MIN_DT, FTOL
from src.utils.VectorFunction import voigt_tensor_dot
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefinition import REAL_DTYPE_NAME
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class PlasticMaterial(ElasticMaterial):
    def __init__(self, material_type, configuration, solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type)
        self.stress_integration = stress_integration
        self.rate_dependent_function = False
        self.is_rate_dependent = False
        self._select_stress_integration()
        self.soft_function = None
        self.is_soft = False
        self.cirtial_state_model = False

    def _select_stress_integration(self):
        if self.stress_integration in ("ReturnMapping", "ImplicitIntegration"):
            self.core = self.implicit_integration
            if not self.is_rate_dependent:
                self.core = getattr(self, "closed_form_integration", self.implicit_integration)
        elif self.stress_integration == "SubStepping":
            # The rate consistency loop requires a complete local return mapping.
            self.core = self.implicit_integration if self.is_rate_dependent else self.explicit_integration
        elif self.stress_integration == "ImplicitIntegrationAL":
            if not hasattr(self, "implicit_integration_al"):
                raise SimulationStateError(
                    f"{type(self).__name__} does not support stress_integration='ImplicitIntegrationAL'"
                )
            self.core = self.implicit_integration_al
        else:
            raise ConfigurationError(f"Unsupported stress_integration='{self.stress_integration}'")

    def choose_soft_function(self, material):
        soft_type = DictIO.get_alternative(material, "SoftType", None)
        if soft_type == "Linear":
            self.soft_function = LinearSoft()
            self.is_soft = True
        elif soft_type == "Exponential":
            alpha = DictIO.get_alternative(material, "alpha", 1.0)
            beta = DictIO.get_alternative(material, "beta", 5.0)
            self.soft_function = ExponentialSoft(alpha, beta)
            self.is_soft = True
        else:
            self.soft_function = None

    def set_rate_dependent_model(self, material):
        enabled = self.validate_rate_dependent(material)
        rate_model = RateDependent(material) if enabled else False
        self.is_rate_dependent = enabled
        self.rate_dependent_function = rate_model
        self._select_stress_integration()

    @ti.func
    def integrate_rate_dependent(self, np, previous_stress, de, dw, state_variables, dt):
        assert dt[None] > 0.0, "mu(I) requires a positive time step"
        initial_state = state_variables[np]
        parameters = self.get_material_parameter(previous_stress, initial_state)
        bulk_modulus, shear_modulus = self.compute_elastic_modulus(previous_stress, parameters)
        trial_stress = self.compute_elastic_stress(1.0, de, previous_stress, parameters) + self.compute_sigrot_stress(
            dw, previous_stress
        )
        pressure = -spherical_tensor(trial_stress)
        rate = 0.0
        updated_stress = trial_stress
        converged = False
        tolerance = 1.0e-9
        if ti.static(REAL_DTYPE_NAME == "float32"):
            tolerance = 1.0e-5
        relaxation = 0.5
        previous_residual = ti.Vector([0.0, 0.0])
        pressure_scale = ti.max(1.0, ti.abs(pressure))
        trial_dev = deviatoric_tensor(trial_stress)
        rate_bound = ti.max(1.0, ti.sqrt(2.0 * voigt_tensor_dot(trial_dev, trial_dev)) / (shear_modulus * dt[None]))
        lower_rate, upper_rate = 0.0, rate_bound
        iteration = 0
        while iteration < 100:
            # Each trial starts from the same history; failed iterations must not accumulate plastic strain or energy.
            state_variables[np] = initial_state
            state_variables[np].strain_rate = rate
            state_variables[np].rate_pressure = pressure
            updated_stress = self.core(np, previous_stress, de, dw, state_variables)
            plastic_de = deviatoric_tensor(trial_stress - updated_stress) / (2.0 * shear_modulus)
            new_rate = ti.sqrt(ti.max(2.0 * voigt_tensor_dot(plastic_de, plastic_de), 0.0)) / dt[None]
            new_pressure = -spherical_tensor(updated_stress)
            error = ti.abs(new_rate - rate) / ti.max(1.0, new_rate)
            error = ti.max(error, ti.abs(new_pressure - pressure) / ti.max(1.0, ti.abs(new_pressure)))
            if error < tolerance:
                converged = True
                break
            residual = ti.Vector([new_rate - rate, (new_pressure - pressure) / pressure_scale])
            # Aitken relaxation handles the stiff friction feedback near initial yielding.
            if iteration > 0:
                delta_residual = residual - previous_residual
                denominator = delta_residual.dot(delta_residual)
                if denominator > 1.0e-30:
                    relaxation = ti.min(
                        1.0, ti.max(1.0e-6, -relaxation * previous_residual.dot(delta_residual) / denominator)
                    )
            if ti.abs(residual[1]) < tolerance:
                if residual[0] > 0.0:
                    lower_rate = rate
                    upper_rate = ti.max(upper_rate, new_rate)
                else:
                    upper_rate = rate
            else:
                lower_rate, upper_rate = 0.0, ti.max(rate_bound, new_rate)
            candidate_rate = rate + relaxation * residual[0]
            if candidate_rate <= lower_rate or candidate_rate >= upper_rate:
                candidate_rate = 0.5 * (lower_rate + upper_rate)
            rate = candidate_rate
            pressure += relaxation * residual[1] * pressure_scale
            previous_residual = residual
            iteration += 1
        if not converged:
            state_variables[np] = initial_state
        assert converged, "mu(I) stress/rate consistency solve failed"
        return updated_stress

    @ti.func
    def compute_yield_function(self, stress, material_params):
        raise NotImplementedError

    @ti.func
    def compute_yield_state(self, stress, internal_vars, material_params):
        raise NotImplementedError

    @ti.func
    def compute_df_dsigma(self, yield_state, stress, internal_vars, material_params):
        raise NotImplementedError

    @ti.func
    def compute_dg_dsigma(self, yield_state, stress, internal_vars, material_params):
        raise NotImplementedError

    @ti.func
    def compute_plastic_modulus(self, yield_state, dgdsigma, stress, internal_vars, state_vars, material_params):
        raise NotImplementedError

    @ti.func
    def get_internal_variables(self, state_vars):
        raise NotImplementedError

    @ti.func
    def compute_internal_variables(self, dlambda, dgdsigma, internal_vars, material_params):
        raise NotImplementedError

    @ti.func
    def update_internal_variables(self, np, internal_vars, state_variables):
        raise NotImplementedError

    @ti.func
    def update_state_variables(self, np, stress, internal_vars, state_variables):
        pass

    @ti.func
    def update_material_parameter(self, stress, internal_vars, state_vars, material_params):
        return material_params

    @ti.func
    def get_df_dlambda(
        self, yield_state, dstrain, bulk_modulus, shear_modulus, stress, internal_vars, state_vars, material_params
    ):
        df_dsigma = self.compute_df_dsigma(yield_state, stress, internal_vars, material_params)
        dg_dsigma = self.compute_dg_dsigma(yield_state, stress, internal_vars, material_params)
        dsigma_dlambda = -dsigma_dlambda(dg_dsigma, bulk_modulus, shear_modulus)
        df_dlambda = voigt_tensor_dot(dsigma_dlambda, df_dsigma) - self.compute_plastic_modulus(
            yield_state, dg_dsigma, stress, internal_vars, state_vars, material_params
        )
        return df_dlambda

    @ti.func
    def get_dlambda_linearize(
        self, yield_state, dstrain, bulk_modulus, shear_modulus, stress, internal_vars, state_vars, material_params
    ):
        df_dsigma = self.compute_df_dsigma(yield_state, stress, internal_vars, material_params)
        dg_dsigma = self.compute_dg_dsigma(yield_state, stress, internal_vars, material_params)
        dsigma_dlambda = -dsigma_dlambda(dg_dsigma, bulk_modulus, shear_modulus)
        df_dlambda = voigt_tensor_dot(dsigma_dlambda, df_dsigma) - self.compute_plastic_modulus(
            yield_state, dg_dsigma, stress, internal_vars, state_vars, material_params
        )
        dlambda = voigt_tensor_dot(dsigma_dlambda, dstrain) / df_dlambda
        return dlambda

    @ti.func
    def get_dlambda(
        self, yield_state, f_function, bulk_modulus, shear_modulus, stress, internal_vars, state_vars, material_params
    ):
        df_dsigma = self.compute_df_dsigma(yield_state, stress, internal_vars, material_params)
        dg_dsigma = self.compute_dg_dsigma(yield_state, stress, internal_vars, material_params)
        dsigma_dlambda = -dsigma_dlambda(dg_dsigma, bulk_modulus, shear_modulus)
        df_dlambda = voigt_tensor_dot(dsigma_dlambda, df_dsigma) - self.compute_plastic_modulus(
            yield_state, dg_dsigma, stress, internal_vars, state_vars, material_params
        )
        dlambda = f_function / df_dlambda
        return dlambda

    @ti.func
    def substepping(self, dT, dstrain, stress, internal_vars, state_vars, material_params):
        material_params = self.update_material_parameter(stress, internal_vars, state_vars, material_params)
        yield_state, _ = self.compute_yield_state(stress, internal_vars, material_params)
        bulk_modulus, shear_modulus = self.compute_elastic_modulus(stress, material_params)
        dsig_e = elastic_tensor_multiply_vector(dT * dstrain, bulk_modulus, shear_modulus)

        dfdsigma = self.compute_df_dsigma(yield_state, stress, internal_vars, material_params)
        dgdsigma = self.compute_dg_dsigma(yield_state, stress, internal_vars, material_params)
        dsigma_dlambda = -dsigma_dlambda(dgdsigma, bulk_modulus, shear_modulus)
        df_dlambda = voigt_tensor_dot(dfdsigma, dsigma_dlambda) - self.compute_plastic_modulus(
            yield_state, dgdsigma, stress, internal_vars, state_vars, material_params
        )
        safe_den = df_dlambda if ti.abs(df_dlambda) > THRESHOLD else THRESHOLD
        dlambda = ti.max(voigt_tensor_dot(dfdsigma, dsig_e) / safe_den, 0)
        dstress = dsig_e - dlambda * dsigma_dlambda
        dinternal_vars = self.compute_internal_variables(dlambda, dgdsigma, internal_vars, material_params)
        return dstress, dinternal_vars

    @ti.func
    def consistent_correction(self, yield_state, f_function, stress, internal_vars, state_vars, material_params):
        material_params = self.update_material_parameter(stress, internal_vars, state_vars, material_params)
        bulk_modulus, shear_modulus = self.compute_elastic_modulus(stress, material_params)
        dfdsigma = self.compute_df_dsigma(yield_state, stress, internal_vars, material_params)
        dgdsigma = self.compute_dg_dsigma(yield_state, stress, internal_vars, material_params)
        dsigma_dlambda = -dsigma_dlambda(dgdsigma, bulk_modulus, shear_modulus)
        df_dlambda = voigt_tensor_dot(dfdsigma, dsigma_dlambda) - self.compute_plastic_modulus(
            yield_state, dgdsigma, stress, internal_vars, state_vars, material_params
        )
        safe_den = df_dlambda if ti.abs(df_dlambda) > THRESHOLD else THRESHOLD
        dlambda = f_function / safe_den
        dstress = dlambda * dsigma_dlambda
        dinternal_vars = self.compute_internal_variables(dlambda, dgdsigma, internal_vars, material_params)
        return stress - dstress, internal_vars + dinternal_vars

    @ti.func
    def normal_correction(self, yield_state, f_function, stress, internal_vars, material_params):
        dfdsigma = self.compute_df_dsigma(yield_state, stress, internal_vars, material_params)
        dfdsigmadfdsigma = voigt_tensor_dot(dfdsigma, dfdsigma)
        safe_den = dfdsigmadfdsigma if ti.abs(dfdsigmadfdsigma) > THRESHOLD else THRESHOLD
        dlambda = f_function / safe_den
        dstress = dlambda * dfdsigma
        return stress - dstress

    @ti.func
    def drift_correct(self, yield_state, f_function, stress, internal_vars, state_vars, material_params):
        for _ in range(MAXITS):
            material_params = self.update_material_parameter(stress, internal_vars, state_vars, material_params)
            stress_new, internal_vars_new = self.consistent_correction(
                yield_state, f_function, stress, internal_vars, state_vars, material_params
            )
            material_params_new = self.update_material_parameter(
                stress_new, internal_vars_new, state_vars, material_params
            )
            yield_state_new, f_function_new = self.compute_yield_state(
                stress_new, internal_vars_new, material_params_new
            )

            if ti.abs(f_function_new) > ti.abs(f_function):
                stress_new = self.normal_correction(yield_state, f_function, stress, internal_vars, material_params)
                material_params_new = self.update_material_parameter(
                    stress_new, internal_vars, state_vars, material_params
                )
                yield_state_new, f_function_new = self.compute_yield_state(
                    stress_new, internal_vars, material_params_new
                )
                internal_vars_new = internal_vars

            stress = stress_new
            internal_vars = internal_vars_new
            material_params = material_params_new
            yield_state = yield_state_new
            f_function = f_function_new
            if ti.abs(f_function_new) <= FTOL:
                break
        assert f_function < 1e-1, f"Stress Integration failed, current F is {f_function}"
        return stress, internal_vars

    @ti.func
    def calculate_elastic_factor(self, np, dstrain, stress, internal_vars, material_params, state_variables):
        trial_stress = self.compute_elastic_stress(1.0, dstrain, stress, material_params)
        yield_state0, f_function0 = self.compute_yield_state(stress, internal_vars, material_params)
        yield_state_trial, f_function1 = self.compute_yield_state(trial_stress, internal_vars, material_params)
        if ti.static(self.solver_type == 1):
            state_variables[np].yield_state = ti.u8(yield_state_trial)

        alpha = 0.0
        if f_function1 <= FTOL:
            alpha = 1.0
        elif f_function0 < -FTOL and f_function1 > FTOL:
            alpha = self.modified_regula_falsi(dstrain, stress, f_function0, internal_vars, material_params, 0.0, 1.0)
        elif ti.abs(f_function0) <= FTOL and f_function1 > FTOL:
            dstress = self.compute_elastic_stress_increment(dstrain, stress, material_params)
            dfdsigma = self.compute_df_dsigma(yield_state0, stress, internal_vars, material_params)
            df_norm = ti.sqrt(ti.max(voigt_tensor_dot(dfdsigma, dfdsigma), THRESHOLD))
            dsig_norm = ti.sqrt(ti.max(voigt_tensor_dot(dstress, dstress), THRESHOLD))
            cos_theta = voigt_tensor_dot(dfdsigma, dstress) / df_norm / dsig_norm

            if cos_theta >= -LTOL:
                alpha = 0.0
            else:
                alpha = self.regula_falsi_negative_plastic_multiplier(
                    dstrain, stress, f_function0, internal_vars, material_params
                )
        else:
            alpha = 0.0
        return alpha

    @ti.func
    def modified_regula_falsi(self, dstrain, stress, f_function_save, internal_vars, material_params, alpha0, alpha1):
        alpha = 0.0
        stress0 = self.compute_elastic_stress(alpha0, dstrain, stress, material_params)
        stress1 = self.compute_elastic_stress(alpha1, dstrain, stress, material_params)
        _, f_function0 = self.compute_yield_state(stress0, internal_vars, material_params)
        _, f_function1 = self.compute_yield_state(stress1, internal_vars, material_params)
        for __ in range(MAXITS):
            alpha = alpha1 - (alpha1 - alpha0) * f_function1 / (f_function1 - f_function0)
            stress_new = self.compute_elastic_stress(alpha, dstrain, stress, material_params)
            _, f_function_new = self.compute_yield_state(stress_new, internal_vars, material_params)

            if ti.abs(f_function_new) <= FTOL:
                break
            if f_function_new * f_function0 < 0.0:
                alpha1 = alpha
                f_function1 = f_function_new
                if f_function_new * f_function_save > 0.0:
                    f_function0 *= 0.5
            else:
                alpha0 = alpha
                f_function0 = f_function_new
                if f_function_new * f_function_save > 0.0:
                    f_function1 *= 0.5
            f_function_save = f_function_new
        return alpha

    @ti.func
    def regula_falsi_negative_plastic_multiplier(self, dstrain, stress, f_function0, internal_vars, material_params):
        unloading_factor = 0.0
        alpha0, alpha1 = 0.0, 1.0
        f_function_save = f_function0
        for _ in range(MAXITS):
            dalpha = (alpha1 - alpha0) / NSUB
            flag = 0
            for __ in range(NSUB):
                alpha = alpha0 + dalpha
                stress_new = self.compute_elastic_stress(alpha, dstrain, stress, material_params)
                ___, f_function_new = self.compute_yield_state(stress_new, internal_vars, material_params)
                if f_function_new > FTOL:
                    alpha1 = alpha
                    if f_function0 < -FTOL:
                        unloading_factor = self.modified_regula_falsi(
                            dstrain, stress, f_function0, internal_vars, material_params, alpha0, alpha1
                        )
                        flag = 1
                        break
                    else:
                        alpha0 = 0
                        f_function0 = f_function_save
                        break
                else:
                    alpha0 = alpha
                    f_function0 = f_function_new
            if flag == 1:
                break
        return unloading_factor

    @ti.func
    def explicit_integration(self, np, previous_stress, de, dw, state_variables):
        state_vars = state_variables[np]
        internal_vars = self.get_internal_variables(state_vars)
        material_params = self.get_material_parameter(previous_stress, state_vars)
        ############################## STEP2 ##############################
        alpha = self.calculate_elastic_factor(np, de, previous_stress, internal_vars, material_params, state_variables)

        ############################## STEP5 ##############################
        stress = self.compute_elastic_stress(alpha, de, previous_stress, material_params)
        trial_stress = self.compute_elastic_stress(1.0, de, previous_stress, material_params)
        if ti.abs(1.0 - alpha) > THRESHOLD:
            stress, internal_vars = self.nburkdp2(
                (1.0 - alpha) * de, stress, internal_vars, state_vars, material_params
            )
        update_stress = stress

        if ti.static(GlobalVariable.TRACKENERGY):
            bulk_modulus, shear_modulus = self.compute_elastic_modulus(previous_stress, material_params)
            a1 = bulk_modulus + (4.0 / 3.0) * shear_modulus
            a2 = bulk_modulus - (2.0 / 3.0) * shear_modulus
            b1 = (a1 + a2) / ((a1 - a2) * (a1 + 2.0 * a2))
            b2 = -a2 / ((a1 - a2) * (a1 + 2.0 * a2))
            dstress = trial_stress - update_stress
            dep = ZEROVEC6F
            dep[0] = b1 * dstress[0] + b2 * (dstress[1] + dstress[2])
            dep[1] = b1 * dstress[1] + b2 * (dstress[0] + dstress[2])
            dep[2] = b1 * dstress[2] + b2 * (dstress[0] + dstress[1])
            dep[3] = dstress[3] / shear_modulus
            dep[4] = dstress[4] / shear_modulus
            dep[5] = dstress[5] / shear_modulus
            state_variables[np].plastic_energy += voigt_tensor_dot(update_stress, dep)

        update_stress += self.compute_sigrot_stress(dw, previous_stress)
        self.update_internal_variables(np, internal_vars, state_variables)
        self.update_state_variables(np, update_stress, internal_vars, state_variables)
        return update_stress

    @ti.func
    def nburkdp2(self, dstrain, sig, internal_vars, state_vars, material_params):
        T = 0.0
        dT = 1.0
        while T < 1:
            sig1 = sig
            internal_vars1 = internal_vars
            dsig1, dinternal_vars1 = self.substepping(dT, dstrain, sig1, internal_vars1, state_vars, material_params)

            iter = 0
            dm = 1.0
            while iter < 50:
                iter += 1

                sig2 = sig + dsig1 * dm
                internal_vars2 = internal_vars1 + dinternal_vars1 * dm
                dsig2, dinternal_vars2 = self.substepping(
                    dT, dstrain, sig2, internal_vars2, state_vars, material_params
                )

                sig_temp = sig + 0.5 * (dsig1 * dm + dsig2)
                E_sigma = 0.5 * (-dsig1 * dm + dsig2)
                dr = sig_temp.norm()
                de = E_sigma.norm()
                dg = 0.0
                if dr > 1e-10:
                    dg = de / dr
                if dg < 1e-6 or dT < 5e-4:
                    sig = sig_temp
                    T += dT
                    internal_vars = internal_vars + 0.5 * (dinternal_vars1 * dm + dinternal_vars2)
                    iter = 50
                db = ti.sqrt(1e-6 / dg)
                if iter < 50:
                    db = 0.8 * db
                if db > 2.0:
                    db = 2.0
                if db < 0.1:
                    db = 0.1
                if dT * db < 5e-4:
                    db = 5e-4 / dT - 1e-6
                if dT * db + T > 1.0:
                    db = (1.0 - T) / dT
                dT = dT * db
                dm = dm * db

            material_params = self.update_material_parameter(sig, internal_vars, state_vars, material_params)
            yield_state, f_function = self.compute_yield_state(sig, internal_vars, material_params)
            if ti.abs(f_function) > FTOL:
                sig, internal_vars = self.drift_correct(
                    yield_state, f_function, sig, internal_vars, state_vars, material_params
                )
        return sig, internal_vars

    @ti.func
    def rkdp2(self, dstrain, sig, internal_vars, state_vars, material_params):
        ############################## STEP6 ##############################
        T = 0.0
        dT = 1.0
        ############################## STEP6 ##############################

        ############################## STEP7 ##############################
        iter = 0
        while T < 1:
            ############################## STEP7 ##############################

            ############################## STEP8 ##############################
            # Substep1 #
            sig1 = sig
            internal_vars1 = internal_vars
            dsig1, dinternal_vars1 = self.substepping(dT, dstrain, sig1, internal_vars1, state_vars, material_params)
            # Substep1 #

            # Substep2 #
            sig2 = sig + dsig1
            internal_vars2 = internal_vars1 + dinternal_vars1
            dsig2, dinternal_vars2 = self.substepping(dT, dstrain, sig2, internal_vars2, state_vars, material_params)
            # Substep2 #
            ############################## STEP8 ##############################

            ############################## STEP9 ##############################
            sig_temp = sig + 0.5 * (dsig1 + dsig2)
            internal_vars_temp = internal_vars + 0.5 * (dinternal_vars1 + dinternal_vars2)
            ############################## STEP9 ##############################

            ############################## STEP10 ##############################
            E_sigma = dsig2 - dsig1
            vars_err = 0.5 * E_sigma.norm() / sig_temp.norm()
            for i in ti.static(range(internal_vars.n)):
                E_internal_vars = dinternal_vars2[i] - dinternal_vars1[i]
                vars_err = ti.max(vars_err, 0.5 * ti.abs(E_internal_vars) / ti.abs(internal_vars_temp[i]))
            err = ti.max(vars_err, EPS)
            ############################## STEP10 ##############################

            ############################## STEP11 ##############################
            iter += 1
            if err > STOL and dT > MIN_DT:
                Q = ti.max(0.9 * ti.sqrt(STOL / err), 0.1)
                dT = ti.max(MIN_DT, Q * dT)
                continue
            ############################## STEP11 ##############################

            ############################## STEP12 ##############################
            sig = sig_temp
            internal_vars = internal_vars_temp
            ############################## STEP12 ##############################

            ############################## STEP13 ##############################
            if ti.static(GlobalVariable.DRIFTCORRECT):
                material_params = self.update_material_parameter(sig, internal_vars, state_vars, material_params)
                yield_state, f_function = self.compute_yield_state(sig, internal_vars, material_params)
                if ti.abs(f_function) > FTOL:
                    sig, internal_vars = self.drift_correct(
                        yield_state, f_function, sig, internal_vars, state_vars, material_params
                    )
            ############################## STEP13 ##############################

            ############################## STEP14 ##############################
            T += dT
            Q = ti.max(0.9 * ti.sqrt(STOL / err), 1.0)
            dT = ti.min(ti.max(MIN_DT, Q * dT), 1.0 - T)
            ############################## STEP14 ##############################
        return sig, internal_vars

    @ti.func
    def rkdp6(self, dstrain, sig, internal_vars, state_vars, material_params):
        ############################## STEP6 ##############################
        T = 0.0
        dT = 1.0
        ############################## STEP6 ##############################

        ############################## STEP7 ##############################
        while T < 1:
            ############################## STEP7 ##############################

            ############################## STEP8 ##############################
            # Substep1 #
            sig1 = sig
            internal_vars1 = internal_vars
            dsig1, dinternal_vars1 = self.substepping(dT, dstrain, sig1, internal_vars1, state_vars, material_params)
            # Substep1 #

            # Substep2 #
            sig2 = sig + 0.2 * dsig1
            internal_vars2 = internal_vars + 0.2 * dinternal_vars1
            dsig2, dinternal_vars2 = self.substepping(dT, dstrain, sig2, internal_vars2, state_vars, material_params)
            # Substep2 #

            # Substep3 #
            sig3 = sig + 3.0 / 40.0 * dsig1 + 9.0 / 40.0 * dsig2
            internal_vars3 = internal_vars + 3.0 / 40.0 * dinternal_vars1 + 9.0 / 40.0 * dinternal_vars2
            dsig3, dinternal_vars3 = self.substepping(dT, dstrain, sig3, internal_vars3, state_vars, material_params)
            # Substep3 #

            # Substep4 #
            sig4 = sig + 0.3 * dsig1 - 0.9 * dsig2 + 6.0 / 5.0 * dsig3
            internal_vars4 = internal_vars + 0.3 * dinternal_vars1 - 0.9 * dinternal_vars2 + 6.0 / 5.0 * dinternal_vars3
            dsig4, dinternal_vars4 = self.substepping(dT, dstrain, sig4, internal_vars4, state_vars, material_params)
            # Substep4 #

            # Substep5 #
            sig5 = sig + 226.0 / 729.0 * dsig1 - 25.0 / 27.0 * dsig2 + 880.0 / 729.0 * dsig3 + 55.0 / 729.0 * dsig4
            internal_vars5 = (
                internal_vars
                + 226.0 / 729.0 * dinternal_vars1
                - 25.0 / 27.0 * dinternal_vars2
                + 880.0 / 729.0 * dinternal_vars3
                + 55.0 / 729.0 * dinternal_vars4
            )
            dsig5, dinternal_vars5 = self.substepping(dT, dstrain, sig5, internal_vars5, state_vars, material_params)
            # Substep5 #

            # Substep6 #
            sig6 = (
                sig
                - 181.0 / 270.0 * dsig1
                + 5.0 / 2.0 * dsig2
                - 226.0 / 297.0 * dsig3
                - 91.0 / 27.0 * dsig4
                + 189.0 / 55.0 * dsig5
            )
            internal_vars6 = (
                internal_vars
                - 181.0 / 270.0 * dinternal_vars1
                + 5.0 / 2.0 * dinternal_vars2
                - 226.0 / 297.0 * dinternal_vars3
                - 91.0 / 27.0 * dinternal_vars4
                + 189.0 / 55.0 * dinternal_vars5
            )
            dsig6, dinternal_vars6 = self.substepping(dT, dstrain, sig6, internal_vars6, state_vars, material_params)
            # Substep6 #
            ############################## STEP8 ##############################

            ############################## STEP9 ##############################
            sig_temp = (
                sig
                + 19.0 / 216.0 * dsig1
                + 1000.0 / 2079.0 * dsig3
                - 125.0 / 216.0 * dsig4
                + 81.0 / 88.0 * dsig5
                + 5.0 / 56.0 * dsig6
            )
            internal_vars_temp = (
                internal_vars
                + 19.0 / 216.0 * dinternal_vars1
                + 1000.0 / 2079.0 * dinternal_vars3
                - 125.0 / 216.0 * dinternal_vars4
                + 81.0 / 88.0 * dinternal_vars5
                + 5.0 / 56.0 * dinternal_vars6
            )
            ############################## STEP9 ##############################

            ############################## STEP10 ##############################
            E_sigma = (
                11.0 / 360.0 * dsig1
                - 10.0 / 63.0 * dsig3
                + 55.0 / 72.0 * dsig4
                - 27.0 / 40.0 * dsig5
                + 11.0 / 280.0 * dsig6
            )
            vars_err = E_sigma.norm() / sig_temp.norm()
            for i in ti.static(range(internal_vars.n)):
                E_internal_vars = (
                    11.0 / 360.0 * dinternal_vars1[i]
                    - 10.0 / 63.0 * dinternal_vars3[i]
                    + 55.0 / 72.0 * dinternal_vars4[i]
                    - 27.0 / 40.0 * dinternal_vars5[i]
                    + 11.0 / 280.0 * dinternal_vars6[i]
                )
                vars_err = ti.max(vars_err, 0.5 * ti.abs(E_internal_vars) / ti.abs(internal_vars_temp[i]))
            err = ti.max(vars_err, EPS)
            ############################## STEP10 ##############################

            ############################## STEP11 ##############################
            if err > STOL and dT > MIN_DT:
                Q = ti.max(0.9 * (STOL / err) ** 0.2, 0.1)
                dT = ti.max(MIN_DT, Q * dT)
                continue
            ############################## STEP11 ##############################

            ############################## STEP12 ##############################
            sig = sig_temp
            internal_vars = internal_vars_temp
            ############################## STEP12 ##############################

            ############################## STEP13 ##############################
            if ti.static(GlobalVariable.DRIFTCORRECT):
                material_params = self.update_material_parameter(sig, internal_vars, state_vars, material_params)
                yield_state, f_function = self.compute_yield_state(sig, internal_vars, material_params)
                if ti.abs(f_function) > FTOL:
                    sig, internal_vars = self.drift_correct(
                        yield_state, f_function, sig, internal_vars, state_vars, material_params
                    )
            ############################## STEP13 ##############################

            ############################## STEP14 ##############################
            T += dT
            Q = ti.max(0.9 * (STOL / err) ** 0.2, 1.0)
            dT = ti.max(MIN_DT, ti.min(Q * dT, 1.0 - T))
            ############################## STEP14 ##############################
        return sig, internal_vars

    @ti.func
    def implicit_integration(self, np, previous_stress, de, dw, state_variables):
        state_vars = state_variables[np]
        internal_vars = self.get_internal_variables(state_vars)
        material_params = self.get_material_parameter(previous_stress, state_vars)
        bulk_modulus, shear_modulus = self.compute_elastic_modulus(previous_stress, material_params)

        # !---- trial elastic stresses ----!
        trial_stress = self.compute_elastic_stress(1.0, de, previous_stress, material_params)

        # !---- compute trial stress invariants ----!
        update_stress = trial_stress
        yield_state_trial, f_function_trial = self.compute_yield_state(trial_stress, internal_vars, material_params)
        if ti.static(self.solver_type == 1):
            state_variables[np].yield_state = ti.u8(yield_state_trial)

        if f_function_trial > FTOL:
            tolerance = 1e-4

            df_dsigma_trial = self.compute_df_dsigma(yield_state_trial, trial_stress, internal_vars, material_params)
            dg_dsigma_trial = self.compute_dg_dsigma(yield_state_trial, trial_stress, internal_vars, material_params)
            dsigma_dlambda_trial = -dsigma_dlambda(dg_dsigma_trial, bulk_modulus, shear_modulus)
            df_dlambda_trial = voigt_tensor_dot(dsigma_dlambda_trial, df_dsigma_trial) - self.compute_plastic_modulus(
                yield_state_trial, dg_dsigma_trial, trial_stress, internal_vars, state_vars, material_params
            )
            safe_den = df_dlambda_trial if ti.abs(df_dlambda_trial) > THRESHOLD else THRESHOLD
            lambda_trial = f_function_trial / safe_den
            update_stress -= lambda_trial * dsigma_dlambda_trial
            internal_vars += self.compute_internal_variables(
                lambda_trial, dg_dsigma_trial, internal_vars, material_params
            )
            material_params = self.update_material_parameter(update_stress, internal_vars, state_vars, material_params)

            if ti.static(GlobalVariable.DRIFTCORRECT):
                yield_state, f_function = self.compute_yield_state(update_stress, internal_vars, material_params)
                if ti.abs(f_function) > tolerance:
                    update_stress, internal_vars = self.drift_correct(
                        yield_state, f_function, update_stress, internal_vars, state_vars, material_params
                    )

        if ti.static(GlobalVariable.TRACKENERGY):
            a1 = bulk_modulus + (4.0 / 3.0) * shear_modulus
            a2 = bulk_modulus - (2.0 / 3.0) * shear_modulus
            b1 = (a1 + a2) / ((a1 - a2) * (a1 + 2.0 * a2))
            b2 = -a2 / ((a1 - a2) * (a1 + 2.0 * a2))
            dstress = trial_stress - update_stress
            dep = ZEROVEC6F
            dep[0] = b1 * dstress[0] + b2 * (dstress[1] + dstress[2])
            dep[1] = b1 * dstress[1] + b2 * (dstress[0] + dstress[2])
            dep[2] = b1 * dstress[2] + b2 * (dstress[0] + dstress[1])
            dep[3] = dstress[3] / shear_modulus
            dep[4] = dstress[4] / shear_modulus
            dep[5] = dstress[5] / shear_modulus
            state_variables[np].plastic_energy += voigt_tensor_dot(update_stress, dep)

        update_stress += self.compute_sigrot_stress(dw, previous_stress)
        self.update_internal_variables(np, internal_vars, state_variables)
        self.update_state_variables(np, update_stress, internal_vars, state_variables)
        return update_stress

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, state_variables):
        stiffness_matrix = self.compute_elastic_tensor(np, current_stress, state_variables)
        state_vars = state_variables[np]
        yield_state = int(state_vars.yield_state)
        internal_vars = self.get_internal_variables(state_vars)
        material_params = self.get_material_parameter(current_stress, state_vars)

        if yield_state > 0:
            bulk_modulus, shear_modulus = self.compute_elastic_modulus(current_stress, material_params)
            dfdsigma = self.compute_df_dsigma(yield_state, current_stress, internal_vars, material_params)
            dgdsigma = self.compute_dg_dsigma(yield_state, current_stress, internal_vars, material_params)
            temp_matf = elastic_tensor_multiply_vector(dfdsigma, bulk_modulus, shear_modulus)
            temp_matg = elastic_tensor_multiply_vector(dgdsigma, bulk_modulus, shear_modulus)
            dfdsigma_dedgdsigma = voigt_tensor_dot(dfdsigma, temp_matg) - self.compute_plastic_modulus(
                yield_state, dgdsigma, current_stress, internal_vars, state_vars, material_params
            )
            stiffness_matrix -= 1.0 / dfdsigma_dedgdsigma * (temp_matg.outer_product(temp_matf))
        return stiffness_matrix
