import taichi as ti

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.SoftenModel import *
from src.physics_model.consititutive_model.infinitesimal_strain.RateDependent import *
from src.utils.constants import LTOL, MAXITS, NSUB, EPS, STOL, dTmin, FTOL
from src.physics_model.consititutive_model.infinitesimal_strain.InfinitesimalStrainModel import InfinitesimalStrainModel
from src.utils.VectorFunction import voigt_tensor_dot, Squared
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class ElasticMaterial(InfinitesimalStrainModel):
    def __init__(self, material_type, configuration, solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.core = self.cores

    def calculate_lame_parameter(self, young, poisson):
        shear = 0.5 * young / (1. + poisson)
        bulk = young / (3. * (1 - 2. * poisson))
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


@ti.data_oriented
class PlasticMaterial(ElasticMaterial):
    def __init__(self, material_type, configuration, solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type)
        if stress_integration == "ReturnMapping":
            self.core = self.ImplicitIntegration
        elif stress_integration == "SubStepping":
            self.core = self.ExplicitIntegration
        self.rate_dependent_function = False
        self.is_rate_dependent = False
        self.soft_function = False
        self.is_soft = False
        self.soft_param = 1.
        self.cirtial_state_model = False

    def choose_soft_function(self, material):
        soft_type = DictIO.GetAlternative(material, "SoftType", None)
        if soft_type == "Linear":
            self.soft_param = DictIO.GetAlternative(material, 'SoftenParameter', 1.) 
            self.soft_function = LinearSoft()
            self.is_soft = True
        elif soft_type == "Exponential":
            self.soft_param = DictIO.GetAlternative(material, 'SoftenParameter', 5.) 
            self.soft_function = ExponentialSoft()
            self.is_soft = True
        else:
            self.soft_function = False
        
    def set_rate_dependent_model(self, material):
        rate_dependent = DictIO.GetAlternative(material, "RateDependent", None)
        if rate_dependent is not None:
            self.rate_dependent_function = RateDependent(rate_dependent)
            self.is_rate_dependent = True
        else:
            self.rate_dependent_function = False
    
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
    def Substepping(self, dT, dstrain, stress, internal_vars, state_vars, material_params):
        yield_state, _ = self.ComputeYieldState(stress, internal_vars, material_params)
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(stress, material_params)
        dsig_e = ElasticTensorMultiplyVector(dT * dstrain, bulk_modulus, shear_modulus)

        dfdsigma = self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)
        dgdsigma = self.ComputeDgDsigma(yield_state, stress, internal_vars, material_params)
        tempMat = ElasticTensorMultiplyVector(dgdsigma, bulk_modulus, shear_modulus)
        den = voigt_tensor_dot(dfdsigma, tempMat) - self.ComputePlasticModulus(yield_state, dgdsigma, stress, internal_vars, state_vars, material_params)
        abeta = 1. / den if ti.abs(den) > Threshold else 0.
        dlambda = ti.max(voigt_tensor_dot(dfdsigma, dsig_e) * abeta, 0)
        dstress = dsig_e - dlambda * tempMat
        dinternal_vars = self.ComputeInternalVariables(dlambda, dgdsigma, internal_vars, material_params)
        return dstress, dinternal_vars
    
    @ti.func
    def line_search(self, stress, dstress, f_function, internal_vars, material_params):
        alpha = 1.
        while True:
            _, f_function_new = self.ComputeYieldState(stress - alpha * dstress, internal_vars, material_params)
            if ti.abs(f_function_new) < ti.abs(f_function) or alpha < 1e-5: 
                break
            alpha /= 2.
        return alpha
    
    @ti.func
    def ConsistentCorrection(self, yield_state, f_function, stress, internal_vars, state_vars, material_params):
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(stress, material_params)
        dfdsigma = self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)
        dgdsigma = self.ComputeDgDsigma(yield_state, stress, internal_vars, material_params)
        tempMat = ElasticTensorMultiplyVector(dgdsigma, bulk_modulus, shear_modulus)
        den = voigt_tensor_dot(dfdsigma, tempMat) - self.ComputePlasticModulus(yield_state, dgdsigma, stress, internal_vars, state_vars, material_params)
        abeta = 1. / den if ti.abs(den) > Threshold else 0.
        dlambda = f_function * abeta
        dstress = dlambda * tempMat
        dinternal_vars = self.ComputeInternalVariables(dlambda, dgdsigma, internal_vars, material_params)
        alpha = 1. # self.line_search(stress, dstress, f_function, internal_vars, material_params)
        return stress - alpha * dstress, internal_vars + alpha * dinternal_vars
    
    @ti.func
    def NormalCorrection(self, yield_state, f_function, stress, internal_vars, material_params):
        dfdsigma = self.ComputeDfDsigma(yield_state, stress, internal_vars, material_params)
        dfdsigmadfdsigma = voigt_tensor_dot(dfdsigma, dfdsigma)
        abeta = 1. / dfdsigmadfdsigma if ti.abs(dfdsigmadfdsigma) > Threshold else 0.
        dlambda = f_function * abeta
        dstress = dlambda * dfdsigma
        alpha = 1. # self.line_search(stress, dstress, f_function, internal_vars, material_params)
        return stress - alpha * dstress
    
    @ti.func
    def DriftCorrect(self, yield_state, f_function, stress, internal_vars, state_vars, material_params):
        for _ in range(MAXITS):
            stress_new, internal_vars_new = self.ConsistentCorrection(yield_state, f_function, stress, internal_vars, state_vars, material_params)
            yield_state_new, f_function_new = self.ComputeYieldState(stress_new, internal_vars_new, material_params)

            if ti.abs(f_function_new) > ti.abs(f_function):
                stress_new = self.NormalCorrection(yield_state, f_function, stress, internal_vars, material_params)
                yield_state_new, f_function_new = self.ComputeYieldState(stress_new, internal_vars, material_params)
                internal_vars_new = internal_vars

            stress = stress_new
            internal_vars = internal_vars_new
            yield_state = yield_state_new
            f_function = f_function_new
            if ti.abs(f_function_new) <= FTOL:
                break
        return stress, internal_vars

    @ti.func
    def CalculateElasticFactor(self, np, dstrain, stress, internal_vars, material_params, stateVars):
        trial_stress = self.ComputeElasticStress(1., dstrain, stress, material_params)
        yield_state0, f_function0 = self.ComputeYieldState(stress, internal_vars, material_params)
        yield_state_trial, f_function1 = self.ComputeYieldState(trial_stress, internal_vars, material_params)
        if ti.static(self.solver_type == 1):
            stateVars[np].yield_state = ti.u8(yield_state_trial)
        
        alpha = 0.
        if f_function1 <= FTOL:
            alpha = 1.
        elif f_function0 < -FTOL and f_function1 > FTOL:
            alpha = self.ModifiedRegulaFalsi(dstrain, stress, f_function0, internal_vars, material_params, 0., 1.)
        elif ti.abs(f_function0) <= FTOL and f_function1 > FTOL:
            dstress = self.ComputeElasticStressIncrement(dstrain, stress, material_params)
            dfdsigma = self.ComputeDfDsigma(yield_state0, stress, internal_vars, material_params)
            cos_theta = voigt_tensor_dot(dfdsigma, dstress) / ti.sqrt(Squared(dfdsigma)) / ti.sqrt(Squared(dstress))

            if cos_theta >= -LTOL:
                alpha = 0.
            else:
                alpha = self.RegulaFalsiNegativePlasticMultiplier(dstrain, stress, f_function0, internal_vars, material_params)
        else:
            alpha = 0.
        return alpha

    @ti.func
    def ModifiedRegulaFalsi(self, dstrain, stress, f_function_save, internal_vars, material_params, alpha0, alpha1):
        alpha = 0. 
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
            if f_function_new * f_function0 < 0.:                    
                alpha1 = alpha
                f_function1 = f_function_new
                if f_function_new * f_function_save > 0.: f_function0 *= 0.5
            else:
                alpha0 = alpha
                f_function0 = f_function_new
                if f_function_new * f_function_save > 0.: f_function1 *= 0.5
            f_function_save = f_function_new
        return alpha

    @ti.func
    def RegulaFalsiNegativePlasticMultiplier(self, dstrain, stress, f_function0, internal_vars, material_params):
        unloading_factor = 0.
        alpha0, alpha1 = 0., 1.
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
                        unloading_factor = self.ModifiedRegulaFalsi(dstrain, stress, f_function0, internal_vars, material_params, alpha0, alpha1)
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
        if ti.abs(1. - alpha) > Threshold:
            stress, internal_vars = self.NBURKDP2(np, (1. - alpha) * de, stress, internal_vars, state_vars, material_params)

        update_stress = stress + self.ComputeSigrotStress(dw, previous_stress)
        self.UpdateInternalVariables(np, internal_vars, stateVars)
        self.UpdateStateVariables(np, update_stress, internal_vars, stateVars)
        return update_stress
    
    @ti.func
    def NBURKDP2(self, np, dstrain, sig, internal_vars, state_vars, material_params):
        T = 0.
        dT = 1.
        while(T < 1):
            sig1 = sig
            internal_vars1 = internal_vars
            dsig1, dinternal_vars1 = self.Substepping(dT, dstrain, sig1, internal_vars1, state_vars, material_params)

            iter = 0
            dm = 1.
            while iter < 50:
                iter += 1

                sig2 = sig + dsig1 * dm
                internal_vars2 = internal_vars1 + dinternal_vars1 * dm
                dsig2, dinternal_vars2 = self.Substepping(dT, dstrain, sig2, internal_vars2, state_vars, material_params)

                sigTemp = sig + 0.5 * (dsig1 * dm + dsig2)
                E_sigma = 0.5 * (-dsig1 * dm + dsig2)
                dr = sigTemp.norm()
                de = E_sigma.norm()
                dg = 0.
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
                if db > 2.: db = 2.
                if db < 0.1: db = 0.1
                if dT * db < 5e-4: db = 5e-4 / dT - 1e-6
                if dT * db + T > 1.: db = (1. - T) / dT
                dT = dT * db
                dm = dm * db

            yield_state, f_function = self.ComputeYieldState(sig, internal_vars, material_params)
            if ti.abs(f_function) > FTOL:
                sig, internal_vars = self.DriftCorrect(yield_state, f_function, sig, internal_vars, state_vars, material_params)
        return sig, internal_vars
    
    @ti.func
    def RKDP2(self, np, dstrain, sig, internal_vars, state_vars, material_params):
        ############################## STEP6 ##############################
        T = 0.
        dT = 1.
        ############################## STEP6 ##############################
        
        ############################## STEP7 ##############################
        iter = 0
        while(T < 1):
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
                Q = ti.max(.9 * ti.sqrt(STOL / err), .1)
                dT = ti.max(dTmin, Q * dT)
                continue
            ############################## STEP11 ##############################

            ############################## STEP12 ##############################
            sig = sigTemp
            internal_vars = internal_varsTemp
            ############################## STEP12 ##############################

            ############################## STEP13 ##############################
            yield_state, f_function = self.ComputeYieldState(sig, internal_vars, material_params)
            if ti.abs(f_function) > FTOL:
                sig, internal_vars = self.DriftCorrect(yield_state, f_function, sig, internal_vars, state_vars, material_params)
            ############################## STEP13 ##############################
            
            ############################## STEP14 ##############################
            T += dT
            Q = ti.max(.9 * ti.sqrt(STOL / err), 1.)
            dT = ti.min(ti.max(dTmin, Q * dT), 1. - T)
            ############################## STEP14 ##############################  
        return sig, internal_vars
    
    @ti.func
    def RKDP6(self, dstrain, sig, internal_vars, state_vars, material_params):
        ############################## STEP6 ##############################
        T = 0.
        dT = 1.
        ############################## STEP6 ##############################
        
        ############################## STEP7 ##############################
        while(T < 1):
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
            sig3 = sig + 3./40. * dsig1 + 9./40. * dsig2
            internal_vars3 = internal_vars + 3./40. * dinternal_vars1 + 9./40. * dinternal_vars2
            dsig3, dinternal_vars3 = self.Substepping(dT, dstrain, sig3, internal_vars3, state_vars, material_params)
            # Substep3 #
            
            # Substep4 #
            sig4 = sig + 0.3 * dsig1 - 0.9 * dsig2 + 6./5. * dsig3
            internal_vars4 = internal_vars + 0.3 * dinternal_vars1 - 0.9 * dinternal_vars2 + 6./5. * dinternal_vars3
            dsig4, dinternal_vars4 = self.Substepping(dT, dstrain, sig4, internal_vars4, state_vars, material_params)
            # Substep4 #
            
            # Substep5 #
            sig5 = sig + 226./729. * dsig1 - 25./27. * dsig2 + 880./729. * dsig3 + 55./729. * dsig4 
            internal_vars5 = internal_vars + 226./729. * dinternal_vars1 - 25./27. * dinternal_vars2 + 880./729. * dinternal_vars3 + 55./729. * dinternal_vars4 
            dsig5, dinternal_vars5 = self.Substepping(dT, dstrain, sig5, internal_vars5, state_vars, material_params)
            # Substep5 #
            
            # Substep6 #
            sig6 = sig - 181./270. * dsig1 + 5./2. * dsig2 - 226./297. * dsig3 - 91./27. * dsig4 + 189./55. * dsig5
            internal_vars6 = internal_vars - 181./270. * dinternal_vars1 + 5./2. * dinternal_vars2 - 226./297. * dinternal_vars3 - 91./27. * dinternal_vars4 + 189./55. * dinternal_vars5
            dsig6, dinternal_vars6 = self.Substepping(dT, dstrain, sig6, internal_vars6, state_vars, material_params)
            # Substep6 #
            ############################## STEP8 ##############################

            ############################## STEP9 ##############################
            sigTemp = sig + 19./216. * dsig1 + 1000./2079. * dsig3 - 125./216. * dsig4 + 81./88. * dsig5 + 5./56. * dsig6
            internal_varsTemp = internal_vars + 19./216. * dinternal_vars1 + 1000./2079. * dinternal_vars3 - 125./216. * dinternal_vars4 + 81./88. * dinternal_vars5 + 5./56. * dinternal_vars6
            ############################## STEP9 ##############################

            ############################## STEP10 ##############################
            E_sigma = 11./360. * dsig1 - 10./63. * dsig3 + 55./72. * dsig4 - 27./40. * dsig5 + 11./280. * dsig6
            vars_err = E_sigma.norm() / sigTemp.norm()
            for i in ti.static(range(internal_vars.n)):
                E_internal_vars = 11./360. * dinternal_vars1[i] - 10./63. * dinternal_vars3[i] + 55./72. * dinternal_vars4[i] - 27./40. * dinternal_vars5[i] + 11./280. * dinternal_vars6[i]
                vars_err = ti.max(vars_err, 0.5 * ti.abs(E_internal_vars) / ti.abs(internal_varsTemp[i]))
            err = ti.max(vars_err, EPS)
            ############################## STEP10 ##############################

            ############################## STEP11 ##############################
            if err > STOL and dT > dTmin:
                Q = ti.max(.9 * (STOL / err) ** 0.2, .1)
                dT = ti.max(dTmin, Q * dT)
                continue
            ############################## STEP11 ##############################

            ############################## STEP12 ##############################
            sig = sigTemp
            internal_vars = internal_varsTemp
            ############################## STEP12 ##############################

            ############################## STEP13 ##############################
            yield_state, f_function = self.ComputeYieldState(sig, internal_vars, material_params)
            if ti.abs(f_function) > FTOL:
                sig, internal_vars = self.DriftCorrect(yield_state, f_function, sig, internal_vars, state_vars, material_params)
            ############################## STEP13 ##############################
            
            ############################## STEP14 ##############################
            T += dT
            Q = ti.max(.9 * (STOL / err) ** 0.2, 1.)
            dT = ti.max(dTmin, ti.min(Q * dT, 1. - T))
            ############################## STEP14 ##############################   
        return sig, internal_vars
    
    @ti.func
    def ImplicitIntegration(self, np, previous_stress, de, dw, stateVars):
        state_vars = stateVars[np]
        internal_vars = self.GetInternalVariables(state_vars)
        material_params = self.GetMaterialParameter(previous_stress, state_vars)
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(previous_stress, material_params)

        # !---- trial elastic stresses ----!
        trial_stress = self.ComputeElasticStress(1., de, previous_stress, material_params)

        # !---- compute trial stress invariants ----!
        update_stress = trial_stress
        yield_state_trial, f_function_trial = self.ComputeYieldState(trial_stress, internal_vars, material_params)
        if ti.static(self.solver_type == 1):
            stateVars[np].yield_state = ti.u8(yield_state_trial)
        
        if yield_state_trial > 0:
            Tolerance = 1e-4
        
            yield_state, f_function = self.ComputeYieldState(previous_stress, internal_vars, material_params)
            if yield_state > 0:
                df_dsigma_trial = self.ComputeDfDsigma(yield_state_trial, trial_stress, internal_vars, material_params)
                dg_dsigma_trial = self.ComputeDgDsigma(yield_state_trial, trial_stress, internal_vars, material_params)
                temp_matrix = ElasticTensorMultiplyVector(dg_dsigma_trial, bulk_modulus, shear_modulus)
                den = voigt_tensor_dot(temp_matrix, df_dsigma_trial) - self.ComputePlasticModulus(yield_state_trial, dg_dsigma_trial, trial_stress, internal_vars, state_vars, material_params)
                lambda_trial = f_function_trial / den if ti.abs(den) > Tolerance else 0.

                update_stress -= lambda_trial * temp_matrix
                internal_vars += self.ComputeInternalVariables(lambda_trial, dg_dsigma_trial, internal_vars, material_params)
            else:
                df_dsigma = self.ComputeDfDsigma(yield_state, previous_stress, internal_vars, material_params)
                dg_dsigma = self.ComputeDgDsigma(yield_state, previous_stress, internal_vars, material_params)
                temp_matrix = ElasticTensorMultiplyVector(dg_dsigma, bulk_modulus, shear_modulus)
                den = voigt_tensor_dot(temp_matrix, df_dsigma) - self.ComputePlasticModulus(yield_state, dg_dsigma, previous_stress, internal_vars, state_vars, material_params)
                lambda_ = voigt_tensor_dot(temp_matrix, de) / den if ti.abs(den) > Tolerance else 0.

                update_stress -= lambda_ * temp_matrix
                internal_vars += self.ComputeInternalVariables(lambda_, dg_dsigma, internal_vars, material_params)

            yield_state, f_function = self.ComputeYieldState(update_stress, internal_vars, material_params)
            if ti.abs(f_function) > Tolerance:
                update_stress, internal_vars = self.DriftCorrect(yield_state, f_function, update_stress, internal_vars, state_vars, material_params)

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
            dfdsigmaDedgdsigma = voigt_tensor_dot(dfdsigma, tempMatg) - self.ComputePlasticModulus(yield_state, dgdsigma, current_stress, internal_vars, state_vars, material_params)
            stiffness_matrix -= 1. / dfdsigmaDedgdsigma * (tempMatg.outer_product(tempMatf))
        return stiffness_matrix