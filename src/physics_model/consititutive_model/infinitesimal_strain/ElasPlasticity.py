import taichi as ti

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.SoftenModel import *
from src.physics_model.consititutive_model.infinitesimal_strain.RateDependent import *
from src.utils.constants import LTOL, MAXITS, NSUB, EPS, STOL, dTmin, FTOL
from src.physics_model.consititutive_model.infinitesimal_strain.InfinitesimalStrainModel import InfinitesimalStrainModel
from src.utils.VectorFunction import voigt_tensor_dot
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import REAL_DTYPE_NAME

import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class ElasticMaterial(InfinitesimalStrainModel):
    def __init__(self, material_type, configuration, solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.core = self.cores

    def calculate_lame_parameter(self, young, poisson):
        shear = 0.5 * young / (1.0 + poisson)
        bulk = young / (3.0 * (1 - 2.0 * poisson))
        return shear, bulk

    def read_random_field(self, start_particle, end_particle, stateVars):
        raise NotImplementedError

    @ti.kernel
    def kernel_add_random_material(self):
        raise NotImplementedError

    @ti.func
    def cores(self, np, previous_stress, de, dw, stateVars):
        raise NotImplementedError

    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        raise NotImplementedError

    @ti.func
    def ComputeSigrotStress(self, dsigrot, stress):
        return Sigrot(stress, dsigrot)

    @ti.func
    def ComputeElasticModulus(self, stress, material_params):
        return material_params[0], material_params[1]

    @ti.func
    def ComputeElasticStress(self, alpha, dstrain, stress, material_params):
        return stress + self.ComputeElasticStressIncrement(alpha * dstrain, stress, material_params)

    @ti.func
    def ComputeElasticStressIncrement(self, dstrain, stress, material_params):
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(stress, material_params)

        # !-- trial elastic stresses ----!
        dstress = ElasticTensorMultiplyVector(dstrain, bulk_modulus, shear_modulus)
        return dstress

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        material_params = self.GetMaterialParameter(current_stress, stateVars[np])
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(current_stress, material_params)
        return ComputeElasticStiffnessTensor(bulk_modulus, shear_modulus)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stateVars):
        return self.compute_elastic_tensor(np, current_stress, stateVars)


@ti.data_oriented
class PlasticMaterial(ElasticMaterial):
    def __init__(self, material_type, configuration, solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type)
        self.stress_integration = stress_integration
        if stress_integration in ("ReturnMapping", "ImplicitIntegration"):
            self.core = self.ImplicitIntegration
        elif stress_integration == "SubStepping":
            self.core = self.ExplicitIntegration
        elif stress_integration == "ImplicitIntegrationAL":
            if not hasattr(self, "ImplicitIntegrationAL"):
                raise RuntimeError(f"{type(self).__name__} does not support stress_integration='ImplicitIntegrationAL'")
            self.core = self.ImplicitIntegrationAL
        else:
            raise RuntimeError(f"Unsupported stress_integration='{stress_integration}'")
        self.rate_dependent_function = False
        self.is_rate_dependent = False
        self.soft_function = None
        self.is_soft = False
        self.cirtial_state_model = False

    def choose_soft_function(self, material):
        soft_type = DictIO.GetAlternative(material, "SoftType", None)
        if soft_type == "Linear":
            params = DictIO.GetAlternative(material, "SoftenParameter", 1.0)
            self.soft_function = LinearSoft(params)
            self.is_soft = True
        elif soft_type == "Exponential":
            alpha = DictIO.GetAlternative(material, "alpha", 1.0)
            beta = DictIO.GetAlternative(material, "beta", 5.0)
            self.soft_function = ExponentialSoft(alpha, beta)
            self.is_soft = True
        else:
            self.soft_function = None

    def set_rate_dependent_model(self, material):
        rate_dependent = DictIO.GetAlternative(material, "RateDependent", False)
        if not isinstance(rate_dependent, (bool, np.bool_)):
            raise ValueError("RateDependent must be a boolean")
        self.is_rate_dependent = bool(rate_dependent)
        if self.is_rate_dependent:
            if self.solver_type != 0:
                raise ValueError("mu(I) RateDependent currently requires an Explicit solver")
            self.rate_dependent_function = RateDependent(material)
            # The coupled rate solve surrounds a complete local return mapping.
            if self.stress_integration == "SubStepping":
                self.core = self.ImplicitIntegration
        else:
            self.rate_dependent_function = False

    @ti.func
    def IntegrateRateDependent(self, np, previous_stress, de, dw, stateVars, dt):
        assert dt[None] > 0.0, "mu(I) requires a positive time step"
        initial_state = stateVars[np]
        parameters = self.GetMaterialParameter(previous_stress, initial_state)
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(previous_stress, parameters)
        trial_stress = self.ComputeElasticStress(1.0, de, previous_stress, parameters) + self.ComputeSigrotStress(
            dw, previous_stress
        )
        pressure = -SphericalTensor(trial_stress)
        rate = 0.0
        updated_stress = trial_stress
        converged = False
        tolerance = 1.0e-9
        if ti.static(REAL_DTYPE_NAME == "float32"):
            tolerance = 1.0e-5
        relaxation = 0.5
        previous_residual = ti.Vector([0.0, 0.0])
        pressure_scale = ti.max(1.0, ti.abs(pressure))
        trial_dev = DeviatoricTensor(trial_stress)
        rate_bound = ti.max(1.0, ti.sqrt(2.0 * voigt_tensor_dot(trial_dev, trial_dev)) / (shear_modulus * dt[None]))
        lower_rate, upper_rate = 0.0, rate_bound
        iteration = 0
        while iteration < 100:
            # Each trial starts from the same history; failed iterations must not accumulate plastic strain or energy.
            stateVars[np] = initial_state
            stateVars[np].strain_rate = rate
            stateVars[np].rate_pressure = pressure
            updated_stress = self.core(np, previous_stress, de, dw, stateVars)
            plastic_de = DeviatoricTensor(trial_stress - updated_stress) / (2.0 * shear_modulus)
            new_rate = ti.sqrt(ti.max(2.0 * voigt_tensor_dot(plastic_de, plastic_de), 0.0)) / dt[None]
            new_pressure = -SphericalTensor(updated_stress)
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
            stateVars[np] = initial_state
        assert converged, "mu(I) stress/rate consistency solve failed"
        return updated_stress

    @ti.func
    def ComputeYieldFunction(self, stress, material_params):
        raise NotImplementedError

    @ti.func
    def ComputeYieldState(self, stress, internal_vars, material_params):
        raise NotImplementedError

    @ti.func
    def ComputeDfDsigma(self, yield_state, stress, internal_vars, material_params):
        raise NotImplementedError

    @ti.func
    def ComputeDgDsigma(self, yield_state, stress, internal_vars, material_params):
        raise NotImplementedError

    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma, stress, internal_vars, state_vars, material_params):
        raise NotImplementedError

    @ti.func
    def GetInternalVariables(self, state_vars):
        raise NotImplementedError

    @ti.func
    def ComputeInternalVariables(self, dlambda, dgdsigma, internal_vars, material_params):
        raise NotImplementedError

    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        raise NotImplementedError

    @ti.func
    def UpdateStateVariables(self, np, stress, internal_vars, stateVars):
        pass

    @ti.func
    def UpdateMaterialParameter(self, stress, internal_vars, state_vars, material_params):
        return material_params

    @ti.func
    def GetDfDlambda(
        self, yield_state, dstrain, bulk_modulus, shear_modulus, stress, internal_vars, state_vars, material_params
    ):
        df_dsigma = self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)
        dg_dsigma = self.ComputeDgDsigma(yield_state, stress, internal_vars, material_params)
        dsigma_dlambda = -DsigmaDlambda(dg_dsigma, bulk_modulus, shear_modulus)
        df_dlambda = voigt_tensor_dot(dsigma_dlambda, df_dsigma) - self.ComputePlasticModulus(
            yield_state, dg_dsigma, stress, internal_vars, state_vars, material_params
        )
        return df_dlambda

    @ti.func
    def GetDlambdaLinearize(
        self, yield_state, dstrain, bulk_modulus, shear_modulus, stress, internal_vars, state_vars, material_params
    ):
        df_dsigma = self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)
        dg_dsigma = self.ComputeDgDsigma(yield_state, stress, internal_vars, material_params)
        dsigma_dlambda = -DsigmaDlambda(dg_dsigma, bulk_modulus, shear_modulus)
        df_dlambda = voigt_tensor_dot(dsigma_dlambda, df_dsigma) - self.ComputePlasticModulus(
            yield_state, dg_dsigma, stress, internal_vars, state_vars, material_params
        )
        dlambda = voigt_tensor_dot(dsigma_dlambda, dstrain) / df_dlambda
        return dlambda

    @ti.func
    def GetDlambda(
        self, yield_state, f_function, bulk_modulus, shear_modulus, stress, internal_vars, state_vars, material_params
    ):
        df_dsigma = self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)
        dg_dsigma = self.ComputeDgDsigma(yield_state, stress, internal_vars, material_params)
        dsigma_dlambda = -DsigmaDlambda(dg_dsigma, bulk_modulus, shear_modulus)
        df_dlambda = voigt_tensor_dot(dsigma_dlambda, df_dsigma) - self.ComputePlasticModulus(
            yield_state, dg_dsigma, stress, internal_vars, state_vars, material_params
        )
        dlambda = f_function / df_dlambda
        return dlambda

    @ti.func
    def Substepping(self, dT, dstrain, stress, internal_vars, state_vars, material_params):
        material_params = self.UpdateMaterialParameter(stress, internal_vars, state_vars, material_params)
        yield_state, _ = self.ComputeYieldState(stress, internal_vars, material_params)
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(stress, material_params)
        dsig_e = ElasticTensorMultiplyVector(dT * dstrain, bulk_modulus, shear_modulus)

        dfdsigma = self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)
        dgdsigma = self.ComputeDgDsigma(yield_state, stress, internal_vars, material_params)
        dsigma_dlambda = -DsigmaDlambda(dgdsigma, bulk_modulus, shear_modulus)
        df_dlambda = voigt_tensor_dot(dfdsigma, dsigma_dlambda) - self.ComputePlasticModulus(
            yield_state, dgdsigma, stress, internal_vars, state_vars, material_params
        )
        safe_den = df_dlambda if ti.abs(df_dlambda) > Threshold else Threshold
        dlambda = ti.max(voigt_tensor_dot(dfdsigma, dsig_e) / safe_den, 0)
        dstress = dsig_e - dlambda * dsigma_dlambda
        dinternal_vars = self.ComputeInternalVariables(dlambda, dgdsigma, internal_vars, material_params)
        return dstress, dinternal_vars

    @ti.func
    def ConsistentCorrection(self, yield_state, f_function, stress, internal_vars, state_vars, material_params):
        material_params = self.UpdateMaterialParameter(stress, internal_vars, state_vars, material_params)
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(stress, material_params)
        dfdsigma = self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)
        dgdsigma = self.ComputeDgDsigma(yield_state, stress, internal_vars, material_params)
        dsigma_dlambda = -DsigmaDlambda(dgdsigma, bulk_modulus, shear_modulus)
        df_dlambda = voigt_tensor_dot(dfdsigma, dsigma_dlambda) - self.ComputePlasticModulus(
            yield_state, dgdsigma, stress, internal_vars, state_vars, material_params
        )
        safe_den = df_dlambda if ti.abs(df_dlambda) > Threshold else Threshold
        dlambda = f_function / safe_den
        dstress = dlambda * dsigma_dlambda
        dinternal_vars = self.ComputeInternalVariables(dlambda, dgdsigma, internal_vars, material_params)
        return stress - dstress, internal_vars + dinternal_vars

    @ti.func
    def NormalCorrection(self, yield_state, f_function, stress, internal_vars, material_params):
        dfdsigma = self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)
        dfdsigmadfdsigma = voigt_tensor_dot(dfdsigma, dfdsigma)
        safe_den = dfdsigmadfdsigma if ti.abs(dfdsigmadfdsigma) > Threshold else Threshold
        dlambda = f_function / safe_den
        dstress = dlambda * dfdsigma
        return stress - dstress

    @ti.func
    def DriftCorrect(self, yield_state, f_function, stress, internal_vars, state_vars, material_params):
        for _ in range(MAXITS):
            material_params = self.UpdateMaterialParameter(stress, internal_vars, state_vars, material_params)
            stress_new, internal_vars_new = self.ConsistentCorrection(
                yield_state, f_function, stress, internal_vars, state_vars, material_params
            )
            material_params_new = self.UpdateMaterialParameter(
                stress_new, internal_vars_new, state_vars, material_params
            )
            yield_state_new, f_function_new = self.ComputeYieldState(stress_new, internal_vars_new, material_params_new)

            if ti.abs(f_function_new) > ti.abs(f_function):
                stress_new = self.NormalCorrection(yield_state, f_function, stress, internal_vars, material_params)
                material_params_new = self.UpdateMaterialParameter(
                    stress_new, internal_vars, state_vars, material_params
                )
                yield_state_new, f_function_new = self.ComputeYieldState(stress_new, internal_vars, material_params_new)
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
    def CalculateElasticFactor(self, np, dstrain, stress, internal_vars, material_params, stateVars):
        trial_stress = self.ComputeElasticStress(1.0, dstrain, stress, material_params)
        yield_state0, f_function0 = self.ComputeYieldState(stress, internal_vars, material_params)
        yield_state_trial, f_function1 = self.ComputeYieldState(trial_stress, internal_vars, material_params)
        if ti.static(self.solver_type == 1):
            stateVars[np].yield_state = ti.u8(yield_state_trial)

        alpha = 0.0
        if f_function1 <= FTOL:
            alpha = 1.0
        elif f_function0 < -FTOL and f_function1 > FTOL:
            alpha = self.ModifiedRegulaFalsi(dstrain, stress, f_function0, internal_vars, material_params, 0.0, 1.0)
        elif ti.abs(f_function0) <= FTOL and f_function1 > FTOL:
            dstress = self.ComputeElasticStressIncrement(dstrain, stress, material_params)
            dfdsigma = self.ComputeDfDsigma(yield_state0, stress, internal_vars, material_params)
            df_norm = ti.sqrt(ti.max(voigt_tensor_dot(dfdsigma, dfdsigma), Threshold))
            dsig_norm = ti.sqrt(ti.max(voigt_tensor_dot(dstress, dstress), Threshold))
            cos_theta = voigt_tensor_dot(dfdsigma, dstress) / df_norm / dsig_norm

            if cos_theta >= -LTOL:
                alpha = 0.0
            else:
                alpha = self.RegulaFalsiNegativePlasticMultiplier(
                    dstrain, stress, f_function0, internal_vars, material_params
                )
        else:
            alpha = 0.0
        return alpha

    @ti.func
    def ModifiedRegulaFalsi(self, dstrain, stress, f_function_save, internal_vars, material_params, alpha0, alpha1):
        alpha = 0.0
        stress0 = self.ComputeElasticStress(alpha0, dstrain, stress, material_params)
        stress1 = self.ComputeElasticStress(alpha1, dstrain, stress, material_params)
        _, f_function0 = self.ComputeYieldState(stress0, internal_vars, material_params)
        _, f_function1 = self.ComputeYieldState(stress1, internal_vars, material_params)
        for __ in range(MAXITS):
            alpha = alpha1 - (alpha1 - alpha0) * f_function1 / (f_function1 - f_function0)
            stress_new = self.ComputeElasticStress(alpha, dstrain, stress, material_params)
            _, f_function_new = self.ComputeYieldState(stress_new, internal_vars, material_params)

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
    def RegulaFalsiNegativePlasticMultiplier(self, dstrain, stress, f_function0, internal_vars, material_params):
        unloading_factor = 0.0
        alpha0, alpha1 = 0.0, 1.0
        f_function_save = f_function0
        for _ in range(MAXITS):
            dalpha = (alpha1 - alpha0) / NSUB
            flag = 0
            for __ in range(NSUB):
                alpha = alpha0 + dalpha
                stress_new = self.ComputeElasticStress(alpha, dstrain, stress, material_params)
                ___, f_function_new = self.ComputeYieldState(stress_new, internal_vars, material_params)
                if f_function_new > FTOL:
                    alpha1 = alpha
                    if f_function0 < -FTOL:
                        unloading_factor = self.ModifiedRegulaFalsi(
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
    def ExplicitIntegration(self, np, previous_stress, de, dw, stateVars):
        state_vars = stateVars[np]
        internal_vars = self.GetInternalVariables(state_vars)
        material_params = self.GetMaterialParameter(previous_stress, state_vars)
        ############################## STEP2 ##############################
        alpha = self.CalculateElasticFactor(np, de, previous_stress, internal_vars, material_params, stateVars)

        ############################## STEP5 ##############################
        stress = self.ComputeElasticStress(alpha, de, previous_stress, material_params)
        trial_stress = self.ComputeElasticStress(1.0, de, previous_stress, material_params)
        if ti.abs(1.0 - alpha) > Threshold:
            stress, internal_vars = self.NBURKDP2(
                (1.0 - alpha) * de, stress, internal_vars, state_vars, material_params
            )
        update_stress = stress

        if ti.static(GlobalVariable.TRACKENERGY):
            bulk_modulus, shear_modulus = self.ComputeElasticModulus(previous_stress, material_params)
            a1 = bulk_modulus + (4.0 / 3.0) * shear_modulus
            a2 = bulk_modulus - (2.0 / 3.0) * shear_modulus
            b1 = (a1 + a2) / ((a1 - a2) * (a1 + 2.0 * a2))
            b2 = -a2 / ((a1 - a2) * (a1 + 2.0 * a2))
            dstress = trial_stress - update_stress
            dep = ZEROVEC6f
            dep[0] = b1 * dstress[0] + b2 * (dstress[1] + dstress[2])
            dep[1] = b1 * dstress[1] + b2 * (dstress[0] + dstress[2])
            dep[2] = b1 * dstress[2] + b2 * (dstress[0] + dstress[1])
            dep[3] = dstress[3] / shear_modulus
            dep[4] = dstress[4] / shear_modulus
            dep[5] = dstress[5] / shear_modulus
            stateVars[np].plastic_energy += voigt_tensor_dot(update_stress, dep)

        update_stress += self.ComputeSigrotStress(dw, previous_stress)
        self.UpdateInternalVariables(np, internal_vars, stateVars)
        self.UpdateStateVariables(np, update_stress, internal_vars, stateVars)
        return update_stress

    @ti.func
    def NBURKDP2(self, dstrain, sig, internal_vars, state_vars, material_params):
        T = 0.0
        dT = 1.0
        while T < 1:
            sig1 = sig
            internal_vars1 = internal_vars
            dsig1, dinternal_vars1 = self.Substepping(dT, dstrain, sig1, internal_vars1, state_vars, material_params)

            iter = 0
            dm = 1.0
            while iter < 50:
                iter += 1

                sig2 = sig + dsig1 * dm
                internal_vars2 = internal_vars1 + dinternal_vars1 * dm
                dsig2, dinternal_vars2 = self.Substepping(
                    dT, dstrain, sig2, internal_vars2, state_vars, material_params
                )

                sigTemp = sig + 0.5 * (dsig1 * dm + dsig2)
                E_sigma = 0.5 * (-dsig1 * dm + dsig2)
                dr = sigTemp.norm()
                de = E_sigma.norm()
                dg = 0.0
                if dr > 1e-10:
                    dg = de / dr
                if dg < 1e-6 or dT < 5e-4:
                    sig = sigTemp
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

            yield_state, f_function = self.ComputeYieldState(sig, internal_vars, material_params)
            if ti.abs(f_function) > FTOL:
                sig, internal_vars = self.DriftCorrect(
                    yield_state, f_function, sig, internal_vars, state_vars, material_params
                )
        return sig, internal_vars

    @ti.func
    def RKDP2(self, dstrain, sig, internal_vars, state_vars, material_params):
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
            dsig1, dinternal_vars1 = self.Substepping(dT, dstrain, sig1, internal_vars1, state_vars, material_params)
            # Substep1 #

            # Substep2 #
            sig2 = sig + dsig1
            internal_vars2 = internal_vars1 + dinternal_vars1
            dsig2, dinternal_vars2 = self.Substepping(dT, dstrain, sig2, internal_vars2, state_vars, material_params)
            # Substep2 #
            ############################## STEP8 ##############################

            ############################## STEP9 ##############################
            sigTemp = sig + 0.5 * (dsig1 + dsig2)
            internal_varsTemp = internal_vars + 0.5 * (dinternal_vars1 + dinternal_vars2)
            ############################## STEP9 ##############################

            ############################## STEP10 ##############################
            E_sigma = dsig2 - dsig1
            vars_err = 0.5 * E_sigma.norm() / sigTemp.norm()
            for i in ti.static(range(internal_vars.n)):
                E_internal_vars = dinternal_vars2[i] - dinternal_vars1[i]
                vars_err = ti.max(vars_err, 0.5 * ti.abs(E_internal_vars) / ti.abs(internal_varsTemp[i]))
            err = ti.max(vars_err, EPS)
            ############################## STEP10 ##############################

            ############################## STEP11 ##############################
            iter += 1
            if err > STOL and dT > dTmin:
                Q = ti.max(0.9 * ti.sqrt(STOL / err), 0.1)
                dT = ti.max(dTmin, Q * dT)
                continue
            ############################## STEP11 ##############################

            ############################## STEP12 ##############################
            sig = sigTemp
            internal_vars = internal_varsTemp
            ############################## STEP12 ##############################

            ############################## STEP13 ##############################
            if ti.static(GlobalVariable.DRIFTCORRECT):
                yield_state, f_function = self.ComputeYieldState(sig, internal_vars, material_params)
                if ti.abs(f_function) > FTOL:
                    sig, internal_vars = self.DriftCorrect(
                        yield_state, f_function, sig, internal_vars, state_vars, material_params
                    )
            ############################## STEP13 ##############################

            ############################## STEP14 ##############################
            T += dT
            Q = ti.max(0.9 * ti.sqrt(STOL / err), 1.0)
            dT = ti.min(ti.max(dTmin, Q * dT), 1.0 - T)
            ############################## STEP14 ##############################
        return sig, internal_vars

    @ti.func
    def RKDP6(self, dstrain, sig, internal_vars, state_vars, material_params):
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
            dsig1, dinternal_vars1 = self.Substepping(dT, dstrain, sig1, internal_vars1, state_vars, material_params)
            # Substep1 #

            # Substep2 #
            sig2 = sig + 0.2 * dsig1
            internal_vars2 = internal_vars + 0.2 * dinternal_vars1
            dsig2, dinternal_vars2 = self.Substepping(dT, dstrain, sig2, internal_vars2, state_vars, material_params)
            # Substep2 #

            # Substep3 #
            sig3 = sig + 3.0 / 40.0 * dsig1 + 9.0 / 40.0 * dsig2
            internal_vars3 = internal_vars + 3.0 / 40.0 * dinternal_vars1 + 9.0 / 40.0 * dinternal_vars2
            dsig3, dinternal_vars3 = self.Substepping(dT, dstrain, sig3, internal_vars3, state_vars, material_params)
            # Substep3 #

            # Substep4 #
            sig4 = sig + 0.3 * dsig1 - 0.9 * dsig2 + 6.0 / 5.0 * dsig3
            internal_vars4 = internal_vars + 0.3 * dinternal_vars1 - 0.9 * dinternal_vars2 + 6.0 / 5.0 * dinternal_vars3
            dsig4, dinternal_vars4 = self.Substepping(dT, dstrain, sig4, internal_vars4, state_vars, material_params)
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
            dsig5, dinternal_vars5 = self.Substepping(dT, dstrain, sig5, internal_vars5, state_vars, material_params)
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
            dsig6, dinternal_vars6 = self.Substepping(dT, dstrain, sig6, internal_vars6, state_vars, material_params)
            # Substep6 #
            ############################## STEP8 ##############################

            ############################## STEP9 ##############################
            sigTemp = (
                sig
                + 19.0 / 216.0 * dsig1
                + 1000.0 / 2079.0 * dsig3
                - 125.0 / 216.0 * dsig4
                + 81.0 / 88.0 * dsig5
                + 5.0 / 56.0 * dsig6
            )
            internal_varsTemp = (
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
            vars_err = E_sigma.norm() / sigTemp.norm()
            for i in ti.static(range(internal_vars.n)):
                E_internal_vars = (
                    11.0 / 360.0 * dinternal_vars1[i]
                    - 10.0 / 63.0 * dinternal_vars3[i]
                    + 55.0 / 72.0 * dinternal_vars4[i]
                    - 27.0 / 40.0 * dinternal_vars5[i]
                    + 11.0 / 280.0 * dinternal_vars6[i]
                )
                vars_err = ti.max(vars_err, 0.5 * ti.abs(E_internal_vars) / ti.abs(internal_varsTemp[i]))
            err = ti.max(vars_err, EPS)
            ############################## STEP10 ##############################

            ############################## STEP11 ##############################
            if err > STOL and dT > dTmin:
                Q = ti.max(0.9 * (STOL / err) ** 0.2, 0.1)
                dT = ti.max(dTmin, Q * dT)
                continue
            ############################## STEP11 ##############################

            ############################## STEP12 ##############################
            sig = sigTemp
            internal_vars = internal_varsTemp
            ############################## STEP12 ##############################

            ############################## STEP13 ##############################
            if ti.static(GlobalVariable.DRIFTCORRECT):
                yield_state, f_function = self.ComputeYieldState(sig, internal_vars, material_params)
                if ti.abs(f_function) > FTOL:
                    sig, internal_vars = self.DriftCorrect(
                        yield_state, f_function, sig, internal_vars, state_vars, material_params
                    )
            ############################## STEP13 ##############################

            ############################## STEP14 ##############################
            T += dT
            Q = ti.max(0.9 * (STOL / err) ** 0.2, 1.0)
            dT = ti.max(dTmin, ti.min(Q * dT, 1.0 - T))
            ############################## STEP14 ##############################
        return sig, internal_vars

    @ti.func
    def ImplicitIntegration(self, np, previous_stress, de, dw, stateVars):
        state_vars = stateVars[np]
        internal_vars = self.GetInternalVariables(state_vars)
        material_params = self.GetMaterialParameter(previous_stress, state_vars)
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(previous_stress, material_params)

        # !---- trial elastic stresses ----!
        trial_stress = self.ComputeElasticStress(1.0, de, previous_stress, material_params)

        # !---- compute trial stress invariants ----!
        update_stress = trial_stress
        yield_state_trial, f_function_trial = self.ComputeYieldState(trial_stress, internal_vars, material_params)
        if ti.static(self.solver_type == 1):
            stateVars[np].yield_state = ti.u8(yield_state_trial)

        if f_function_trial > FTOL:
            Tolerance = 1e-4

            df_dsigma_trial = self.ComputeDfDsigma(yield_state_trial, trial_stress, internal_vars, material_params)
            dg_dsigma_trial = self.ComputeDgDsigma(yield_state_trial, trial_stress, internal_vars, material_params)
            dsigma_dlambda_trial = -DsigmaDlambda(dg_dsigma_trial, bulk_modulus, shear_modulus)
            df_dlambda_trial = voigt_tensor_dot(dsigma_dlambda_trial, df_dsigma_trial) - self.ComputePlasticModulus(
                yield_state_trial, dg_dsigma_trial, trial_stress, internal_vars, state_vars, material_params
            )
            safe_den = df_dlambda_trial if ti.abs(df_dlambda_trial) > Threshold else Threshold
            lambda_trial = f_function_trial / safe_den
            update_stress -= lambda_trial * dsigma_dlambda_trial
            internal_vars += self.ComputeInternalVariables(
                lambda_trial, dg_dsigma_trial, internal_vars, material_params
            )

            if ti.static(GlobalVariable.DRIFTCORRECT):
                yield_state, f_function = self.ComputeYieldState(update_stress, internal_vars, material_params)
                if ti.abs(f_function) > Tolerance:
                    update_stress, internal_vars = self.DriftCorrect(
                        yield_state, f_function, update_stress, internal_vars, state_vars, material_params
                    )

        if ti.static(GlobalVariable.TRACKENERGY):
            a1 = bulk_modulus + (4.0 / 3.0) * shear_modulus
            a2 = bulk_modulus - (2.0 / 3.0) * shear_modulus
            b1 = (a1 + a2) / ((a1 - a2) * (a1 + 2.0 * a2))
            b2 = -a2 / ((a1 - a2) * (a1 + 2.0 * a2))
            dstress = trial_stress - update_stress
            dep = ZEROVEC6f
            dep[0] = b1 * dstress[0] + b2 * (dstress[1] + dstress[2])
            dep[1] = b1 * dstress[1] + b2 * (dstress[0] + dstress[2])
            dep[2] = b1 * dstress[2] + b2 * (dstress[0] + dstress[1])
            dep[3] = dstress[3] / shear_modulus
            dep[4] = dstress[4] / shear_modulus
            dep[5] = dstress[5] / shear_modulus
            stateVars[np].plastic_energy += voigt_tensor_dot(update_stress, dep)

        update_stress += self.ComputeSigrotStress(dw, previous_stress)
        self.UpdateInternalVariables(np, internal_vars, stateVars)
        self.UpdateStateVariables(np, update_stress, internal_vars, stateVars)
        return update_stress

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stateVars):
        stiffness_matrix = self.compute_elastic_tensor(np, current_stress, stateVars)
        state_vars = stateVars[np]
        yield_state = int(state_vars.yield_state)
        internal_vars = self.GetInternalVariables(state_vars)
        material_params = self.GetMaterialParameter(current_stress, state_vars)

        if yield_state > 0:
            bulk_modulus, shear_modulus = self.ComputeElasticModulus(current_stress, material_params)
            dfdsigma = self.ComputeDfDsigma(yield_state, current_stress, internal_vars, material_params)
            dgdsigma = self.ComputeDgDsigma(yield_state, current_stress, internal_vars, material_params)
            tempMatf = ElasticTensorMultiplyVector(dfdsigma, bulk_modulus, shear_modulus)
            tempMatg = ElasticTensorMultiplyVector(dgdsigma, bulk_modulus, shear_modulus)
            dfdsigmaDedgdsigma = voigt_tensor_dot(dfdsigma, tempMatg) - self.ComputePlasticModulus(
                yield_state, dgdsigma, current_stress, internal_vars, state_vars, material_params
            )
            stiffness_matrix -= 1.0 / dfdsigmaDedgdsigma * (tempMatg.outer_product(tempMatf))
        return stiffness_matrix
