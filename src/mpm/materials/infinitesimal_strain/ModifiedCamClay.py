import numpy as np
import taichi as ti

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.MaterialKernel import *
from src.utils.constants import (DELTA, EPS, EYE, FTOL, ITS, LTOL, MAXITS, NSUB, STOL, ZEROVEC6f, dTmin)
from src.utils.MatrixFunction import matrix_form
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import mat3x3
from src.utils.VectorFunction import voigt_form, voigt_tensor_dot, voigt_tensor_trace, Squared


class ModifiedCamClay(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_particle_num, configuration="ULMPM", solver_type="Explicit"):
        super().__init__()
        self.matProps = ModifiedCamClayModel.field(shape=max_material_num)
        if configuration == "ULMPM":
            self.stateVars = ULStateVariable.field(shape=max_particle_num) 
        elif configuration == "TLMPM":
            self.stateVars = TLStateVariable.field(shape=max_particle_num) 

        if solver_type == "Implicit":
            self.stiffness_matrix = ti.Matrix.field(6, 6, float, shape=max_particle_num)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID')
        self.check_materialID(materialID, self.matProps.shape[0])
        
        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, 'Density', 2650)
        possion = DictIO.GetAlternative(material, 'PossionRatio', 0.3)
        m_theta = DictIO.GetEssential(material, 'StressRatio')
        lambda_ = DictIO.GetEssential(material, 'lambda')
        kappa = DictIO.GetEssential(material, 'kappa')
        p0 = DictIO.GetEssential(material, 'ConsolidationPressure')

        if 'initial_void_ratio' in material and 'OCR' in material:
            initial_void_ratio = DictIO.GetEssential(material, 'initial_void_ratio')
            ocr = DictIO.GetEssential(material, 'OCR')
            p_ref = 1000.
            e_ref = initial_void_ratio + lambda_ * ti.log(p0 / p_ref / ocr) - kappa * ti.log(ocr)
        else:
            e_ref = DictIO.GetEssential(material, 'void_ratio_ref')
            p_ref = DictIO.GetAlternative(material, 'pressure_ref', 1000.)
        
        self.matProps[materialID].add_material(density, possion, m_theta, kappa, lambda_, e_ref, p_ref, p0)
        self.matProps[materialID].print_message(materialID)

    def get_state_vars_dict(self, start_particle, end_particle):
        pc = np.ascontiguousarray(self.stateVars.pc.to_numpy()[start_particle:end_particle])
        epstrain = np.ascontiguousarray(self.stateVars.epstrain.to_numpy()[start_particle:end_particle])
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        void_ratio = np.ascontiguousarray(self.stateVars.void_ratio.to_numpy()[start_particle:end_particle])
        return {'pc': pc, 'epstrain': epstrain, 'estress': estress, 'void_ratio': void_ratio}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        epstrain = state_vars.item()['epstrain']
        pc = state_vars.item()['pc']
        void_ratio = state_vars.item()['void_ratio']
        kernel_reload_state_variables(estress, epstrain, pc, void_ratio, self.stateVars)
    
    def get_lateral_coefficient(self, materialID):
        return 0.9


@ti.dataclass
class ULStateVariable:
    epstrain: float
    estress: float
    pc: float
    void_ratio: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        p = -MeanStress(stress)
        materialID = int(particle[np].materialID)
        self.pc = matProps[materialID].p0
        self.estress = VonMisesStress(stress)
        self.void_ratio = matProps[materialID].e_ref - matProps[materialID].lambda_ * ti.log(self.pc / matProps[materialID].p_ref) + matProps[materialID].kappa * ti.log(self.pc / p)

    @ti.func
    def _update_vars(self, stress, epstrain, pc, void_ratio):
        self.estress = VonMisesStress(-stress)
        self.epstrain = epstrain
        self.pc = pc
        self.void_ratio = void_ratio

    @ti.func
    def _get_internal_vars(self):
        return self.pc
    
    @ti.func
    def _set_internal_vars(self, var):
        self.pc = var


@ti.dataclass
class TLStateVariable:
    epstrain: float
    estress: float
    pc: float
    void_ratio: float
    deformation_gradient: mat3x3
    stress: mat3x3

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        p = -MeanStress(stress)
        materialID = int(particle[np].materialID)
        self.pc = matProps[materialID].p0
        self.estress = VonMisesStress(stress)
        self.void_ratio = matProps[materialID].e_ref - matProps[materialID].lambda_ * ti.log(self.pc / matProps[materialID].p_ref) + matProps[materialID].kappa * ti.log(self.pc / p)
        self.deformation_gradient = DELTA
        self.stress = matrix_form(stress)

    @ti.func
    def _update_deformation_gradient(self, deformation_gradient_rate, dt):
        self.deformation_gradient += deformation_gradient_rate * dt[None]

    @ti.func
    def _update_vars(self, stress, epstrain, pc, void_ratio):
        self.estress = VonMisesStress(-stress)
        self.epstrain = epstrain
        self.pc = pc
        self.void_ratio = void_ratio

    @ti.func
    def _get_internal_vars(self):
        return self.pc
    
    @ti.func
    def _set_internal_vars(self, var):
        self.pc = var


@ti.dataclass
class ModifiedCamClayModel:
    density: float
    possion: float
    m_theta: float
    kappa: float
    lambda_: float
    e_ref: float
    p_ref: float
    p0: float

    def add_material(self, density, possion, m_theta, kappa, lambda_, e_ref, p_ref, p0):
        self.density = density
        self.possion = possion
        self.m_theta = m_theta
        self.kappa = kappa
        self.lambda_ = lambda_
        self.e_ref = e_ref
        self.p_ref = p_ref
        self.p0 = p0

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model = Modified Cam-Clay Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Possion Ratio: ', self.possion)
        print('Critical Stress Ratio = ', self.m_theta)
        print('Compression index = ', self.lambda_)
        print('Swelling index = ', self.kappa)
        print('Initial void ratio = ', self.e_ref)
        print('Initial pressure = ', self.p0, '\n')

    @ti.func
    def _get_sound_speed(self):
        sound_speed = 0.
        if self.density > 0.:
            bulk_modulus = (1. + self.e_ref) / self.kappa * self.p0
            shear_modulus = 3. * bulk_modulus * (1 - 2 * self.possion) / (2 * (1 + self.possion))
            young = 2. * (1. + self.possion) * shear_modulus
            sound_speed = ti.sqrt(young * (1 - self.possion) / (1 + self.possion) / (1 - 2 * self.possion) / self.density)
        return sound_speed
    
    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        return (DELTA + velocity_gradient * dt[None]).determinant()
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        return 1. + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])
    
    @ti.func
    def PK2CauchyStress(self, np, stateVars):
        inv_j = 1. / stateVars[np].deformation_gradient.determinant()
        return voigt_form(stateVars[np].stress @ stateVars[np].deformation_gradient.transpose() * inv_j)

    @ti.func
    def Cauchy2PKStress(self, np, stateVars, stress):
        j = stateVars[np].deformation_gradient.determinant()
        return matrix_form(stress) @ stateVars[np].deformation_gradient.inverse().transpose() * j
    
    # ============================================================================= #
    #               explicit stress integration with Runge Kutta                    #
    # ============================================================================= #
    @ti.func
    def ComputeElasticModulus(self, p, void_ratio):
        bulk_modulus = (1. + void_ratio) / self.kappa * p
        shear_modulus = 3. * bulk_modulus * (1 - 2 * self.possion) / (2 * (1 + self.possion))
        return bulk_modulus, shear_modulus
    
    @ti.func
    def ComputeDfDp(self, p, pc):
        return self.m_theta * self.m_theta * (2 * p - pc)
    
    @ti.func
    def ComputePlasticModulus(self, p, pc, void_ratio):
        upslion = (1 + void_ratio) / (self.lambda_ - self.kappa)
        dfdpc = -self.m_theta * self.m_theta * p
        dpdmul = pc * upslion
        dfdp = self.ComputeDfDp(p, pc)
        return -dfdpc * dpdmul * dfdp
        
    @ti.func
    def ComputeDfDsigma(self, p, devsig, pc):
        return self.ComputeDfDp(p, pc) * DpDsigma() + 3. * devsig
    
    @ti.func
    def ComputeVoidRatio(self, p, pc):
        return self.e_ref - self.lambda_ * ti.log(pc / self.p_ref) + self.kappa * ti.log(pc / p) 

    @ti.func
    def ComputeInternalVariableIncrement(self, void_ratio, pc, depstrain):
        depvstrain = voigt_tensor_trace(depstrain) 
        upslion = (1 + void_ratio) / (self.lambda_ - self.kappa)
        return pc * upslion * depvstrain

    @ti.func
    def ComputeTrialStress(self, dstrain, stress, void_ratio):
        p = MeanStress(stress)
        
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(p, void_ratio)
        return ElasticTensorMultiplyVector(dstrain, bulk_modulus, shear_modulus)
    
    @ti.func
    def ComputeStressInvariants(self, stress):
        p = MeanStress(stress)
        q = EquivalentStress(stress)
        return p, q
    
    @ti.func
    def ComputeYieldFunction(self, stress, pc):
        p, q = self.ComputeStressInvariants(stress)

        return (q / self.m_theta) * (q / self.m_theta) + p * (p - pc)

    @ti.func
    def is_yielding(self, stress, pc):
        return self.ComputeYieldFunction(stress, pc) < -FTOL

    @ti.func
    def ComputeElasticStress(self, dstrain, stress, void_ratio):
        ds = 0.
        di = 1.
        while ds < 1.:
            des = di * dstrain
            trial_dstress1 = self.ComputeTrialStress(des, stress, void_ratio)
            
            dm = 1.0
            flag = 0
            for niter in range(50):
                if niter > 0: des = di * dstrain
                trial_dstress2 = self.ComputeTrialStress(des, stress + dm * trial_dstress1, void_ratio)

                dstress_add = 0.5 * (trial_dstress1 * dm + trial_dstress2) + stress
                dstress_minus = 0.5 * (-trial_dstress1 * dm + trial_dstress2)
                dr = dstress_add.norm()
                de = dstress_minus.norm()

                dg = 0.
                if dr < 1e-10:
                    dg = 0.
                else:
                    dg = de / dr

                if dg < 1e-8 or di <= 1e-4:
                    stress = dstress_add
                    ds += di
                    flag = 1
                
                if flag == 1:
                    db = clamp(0.1, 1.5, ti.sqrt(1e-8 / dg))
                    if di * db < 1e-4:
                        db = 1e-4 / di - 1e-10
                    if di * db + ds > 1.0:
                        db = (1. - ds) / di
                    di *= db
                    break
                else:
                    db = clamp(0.1, 1.5, 0.8 * ti.sqrt(1e-8 / dg))
                    if di * db < 1e-4:
                        db = 1e-4 / di - 1e-10
                    if di * db + ds > 1.0:
                        db = (1. - ds) / di
                    di *= db
                    dm *= db
        return stress
    
    @ti.func
    def Substepping(self, dsig_e, stress, pc, void_ratio):
        p = MeanStress(stress)
        devsig = DeviatoricStress(stress)
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(p, void_ratio)

        dfdsigma = self.ComputeDfDsigma(p, devsig, pc)
        Kp = self.ComputePlasticModulus(p, pc, void_ratio)
        tempMat = ElasticTensorMultiplyVector(dfdsigma, bulk_modulus, shear_modulus)
        dfdsigmaDedgdsigma = voigt_tensor_dot(dfdsigma, tempMat)
        dlambda = ti.max(voigt_tensor_dot(dfdsigma, dsig_e) / (dfdsigmaDedgdsigma + Kp), 0)
        dsig = dsig_e - dlambda * tempMat

        depstrain = dlambda * dfdsigma
        dpc = self.ComputeInternalVariableIncrement(void_ratio, pc, depstrain)
        dpdstrain = EquivalentStress(depstrain)
        return dsig, dpc, dpdstrain
    
    @ti.func
    def ConsistentCorrection(self, f_function, stress, pc, dpdstrain):
        p = MeanStress(stress) 
        devsig = DeviatoricStress(stress)
        void_ratio = self.ComputeVoidRatio(p, pc)
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(p, void_ratio)

        Kp = self.ComputePlasticModulus(p, pc, void_ratio)
        dfdsigma = self.ComputeDfDsigma(p, devsig, pc)
        tempMat = ElasticTensorMultiplyVector(dfdsigma, bulk_modulus, shear_modulus)
        dfdsigmaDedgdsigma = voigt_tensor_dot(dfdsigma, tempMat) 
        dlambda = f_function / (dfdsigmaDedgdsigma + Kp)
        stress_new = stress - dlambda * tempMat
        depstrain = dlambda * dfdsigma
        pc_new = pc + self.ComputeInternalVariableIncrement(void_ratio, pc, depstrain)
        dpdstrain_new = dpdstrain + EquivalentStress(depstrain)
        return stress_new, pc_new, dpdstrain_new
    
    @ti.func
    def NormalCorrection(self, f_function, stress, pc):
        p = MeanStress(stress) 
        devsig = DeviatoricStress(stress)
        dfdsigma =self.ComputeDfDsigma(p, devsig, pc)

        dlambda = f_function / voigt_tensor_dot(dfdsigma, dfdsigma)
        stress_new = stress - dlambda * dfdsigma
        return stress_new
    
    @ti.func
    def UpdateInternalVariables(self, np, pc, dpdstrain, stateVars):
        stateVars[np]._set_internal_vars(pc)
        stateVars[np].epstrain += dpdstrain

    @ti.func
    def UpdateStateVariables(self, np, stress, stateVars):
        p = -MeanStress(stress)
        pc = stateVars[np].pc
        stateVars[np].void_ratio = self.ComputeVoidRatio(p, pc)
    
    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        ############################## STEP1 ##############################
        de = calculate_strain_increment2D(-velocity_gradient, dt)
        dw = calculate_vorticity_increment2D(velocity_gradient, dt)

        void_ratio = stateVars[np].void_ratio
        pc = stateVars[np].pc

        ############################## STEP2 ##############################
        alpha = self.CalculateElasticFactor(de, -previous_stress, void_ratio, pc)

        ############################## STEP5 ##############################
        stress = self.ComputeElasticStress(alpha * de, -previous_stress, void_ratio)

        if ti.abs(1. - alpha) > Threshold:
            dsig_e = self.ComputeElasticStress((1. - alpha) * de, -previous_stress, void_ratio) + previous_stress
            stress = self.core(np, dsig_e, stress, void_ratio, pc, stateVars)

        dsigrot = Sigrot(previous_stress, dw)
        update_stress = -stress + dsigrot
        self.UpdateStateVariables(np, update_stress, stateVars)
        return update_stress

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        ############################## STEP1 ##############################
        de = calculate_strain_increment(-velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)

        void_ratio = stateVars[np].void_ratio
        pc = stateVars[np].pc

        ############################## STEP2 ##############################
        alpha = self.CalculateElasticFactor(de, -previous_stress, void_ratio, pc)

        ############################## STEP5 ##############################
        stress = self.ComputeElasticStress(alpha * de, -previous_stress, void_ratio)

        if ti.abs(1. - alpha) > Threshold:
            dsig_e = self.ComputeElasticStress((1. - alpha) * de, -previous_stress, void_ratio) + previous_stress
            stress = self.core(np, dsig_e, stress, void_ratio, pc, stateVars)

        dsigrot = Sigrot(previous_stress, dw)
        update_stress = -stress + dsigrot
        self.UpdateStateVariables(np, update_stress, stateVars)
        return update_stress
    
    @ti.func
    def core(self, np, dsig_e, sig, void_ratio, pc, stateVars):
        ############################## STEP6 ##############################
        T = 0.
        dT = 1.
        dpdstrain = 0.
        ############################## STEP6 ##############################
        
        ############################## STEP7 ##############################
        while(T < 1):
        ############################## STEP7 ##############################

            ############################## STEP8 ##############################
            # Substep1 #
            sig1 = sig
            pc1 = pc
            dsig1, dpc1, dpdstrain1 = self.Substepping(dT * dsig_e, sig1, pc1, void_ratio)
            # Substep1 #

            # Substep2 #
            sig2 = sig + 0.2 * dsig1
            pc2 = pc + 0.2 * dpc1
            dsig2, dpc2, dpdstrain2 = self.Substepping(dT * dsig_e, sig2, pc2, void_ratio)
            # Substep2 #
            
            # Substep3 #
            sig3 = sig + 3./40. * dsig1 + 9./40.*dsig2
            pc3 = pc + 3./40. * dpc1 + 9./40.*dpc2
            dsig3, dpc3, dpdstrain3 = self.Substepping(dT * dsig_e, sig3, pc3, void_ratio)
            # Substep3 #
            
            # Substep4 #
            sig4 = sig + 0.3 * dsig1 - 0.9 * dsig2 + 6./5. * dsig3
            pc4 = pc + 0.3 * dpc1 - 0.9 * dpc2 + 6./5. * dpc3
            dsig4, dpc4, dpdstrain4 = self.Substepping(dT * dsig_e, sig4, pc4, void_ratio)
            # Substep4 #
            
            # Substep5 #
            sig5 = sig + 226./729. * dsig1 - 25./27. * dsig2 + 880./729. * dsig3 + 55./729. * dsig4 
            pc5 = pc + 226./729. * dpc1 - 25./27. * dpc2 + 880./729. * dpc3 + 55./729. * dpc4 
            dsig5, dpc5, dpdstrain5 = self.Substepping(dT * dsig_e, sig5, pc5, void_ratio)
            # Substep5 #
            
            # Substep6 #
            sig6 = sig - 181./270. * dsig1 + 5./2. * dsig2 - 226./297. * dsig3 - 91./27. * dsig4 + 189./55. * dsig5
            pc6 = pc - 181./270. * dpc1 + 5./2. * dpc2 - 226./297. * dpc3 - 91./27. * dpc4 + 189./55. * dpc5 
            dsig6, dpc6, dpdstrain6 = self.Substepping(dT * dsig_e, sig6, pc6, void_ratio)
            # Substep6 #
            ############################## STEP8 ##############################

            ############################## STEP9 ##############################
            sigTemp = sig + 19./216. * dsig1 + 1000./2079. * dsig3 - 125./216. * dsig4 + 81./88. * dsig5 + 5./56. * dsig6
            pcTemp = pc + 19./216. * dpc1 + 1000./2079. * dpc3 - 125./216. * dpc4 + 81./88. * dpc5 + 5./56. * dpc6
            dpdstrainTemp = dpdstrain + 19./216. * dpdstrain1 + 1000./2079. * dpdstrain3 - 125./216. * dpdstrain4 + 81./88. * dpdstrain5 + 5./56. * dpdstrain6
            ############################## STEP9 ##############################

            ############################## STEP10 ##############################
            E_sigma = 11./360. * dsig1 - 10./63. * dsig3 + 55./72. * dsig4 - 27./40. * dsig5 + 11./280. * dsig6
            E_pc = 11./360. * dpc1 - 10./63. * dpc3 + 55./72. * dpc4 - 27./40. * dpc5 + 11./280. * dpc6
            err = ti.max(ti.max(E_sigma.norm() / sigTemp.norm(), ti.abs(E_pc) / ti.abs(pcTemp)), EPS)
            ############################## STEP10 ##############################

            ############################## STEP11 ##############################
            if err > STOL and dT > dTmin:
                Q = ti.max(.9 * (STOL / err) ** 0.2, .1)
                dT = ti.max(dTmin, Q * dT)
                continue
            ############################## STEP11 ##############################

            ############################## STEP12 ##############################
            sig = sigTemp
            pc = pcTemp
            dpdstrain = dpdstrainTemp
            ############################## STEP12 ##############################

            ############################## STEP13 ##############################
            f_function = self.ComputeYieldFunction(sig, pc)
            if ti.abs(f_function) > FTOL:
                sig, pc, dpdstrain = self.DriftCorrect(f_function, sig, pc, dpdstrain)
            ############################## STEP13 ##############################
            
            ############################## STEP14 ##############################
            Q = ti.max(.9 * (STOL / err) ** 0.2, 1.)
            dT = ti.max(dTmin, ti.min(Q * dT, 1. - T))
            T += dT
            ############################## STEP14 ##############################   

        self.UpdateInternalVariables(np, pc, dpdstrain, stateVars)
        return sig

    @ti.func
    def DriftCorrect(self, f_function, stress, pc, dpdstrain):
        for _ in range(MAXITS):
            stress_new, pc_new, dpdstrain_new = self.ConsistentCorrection(f_function, stress, pc, dpdstrain)
            f_function_new = self.ComputeYieldFunction(stress_new, pc_new)

            if ti.abs(f_function_new) > ti.abs(f_function):
                stress_new = self.NormalCorrection(f_function, stress, pc)
                f_function_new = self.ComputeYieldFunction(stress_new, pc_new)
                pc_new = pc
                dpdstrain_new = dpdstrain

            if ti.abs(f_function_new) <= FTOL:
                stress = stress_new
                pc = pc_new
                dpdstrain = dpdstrain_new
                break

            stress = stress_new
            pc = pc_new
            dpdstrain = dpdstrain_new
            f_function = f_function_new
        return stress, pc, dpdstrain

    @ti.func
    def CalculateElasticFactor(self, dstrain, stress, void_ratio, pc):
        dstress = self.ComputeTrialStress(dstrain, stress, void_ratio)
        f_function0 = self.ComputeYieldFunction(stress, pc)
        f_function1 = self.ComputeYieldFunction(stress + dstress, pc)

        alpha = 0.
        if f_function1 <= FTOL:
            alpha = 1.
        elif f_function0 < -FTOL and f_function1 > FTOL:
            alpha = self.ModifiedRegulaFalsi(dstrain, stress, f_function0, f_function1, void_ratio, pc)
        elif ti.abs(f_function0) <= FTOL and f_function1 > FTOL:
            p = MeanStress(stress)
            devsig = DeviatoricStress(stress)
            dfdsigma = self.ComputeDfDsigma(p, devsig, stress)
            cos_theta = voigt_tensor_dot(dfdsigma, dstress) / ti.sqrt(Squared(dfdsigma)) / ti.sqrt(Squared(dstress))

            if cos_theta >= -LTOL:
                alpha = 0.
            else:
                alpha = self.RegulaFalsiNegativePlasticMultiplier(dstress, stress, f_function0, void_ratio, pc)
        else:
            alpha = 0.
        return alpha

    @ti.func
    def ModifiedRegulaFalsi(self, dstrain, stress, f_function0, f_function1, void_ratio, pc):
        alpha = 0. 
        alpha0, alpha1 = 0., 1.
        for _ in range(MAXITS):
            alpha = alpha1 - (alpha1 - alpha0) * f_function1 / (f_function1 - f_function0)
            des = dstrain * alpha
            stress_new = self.ComputeElasticStress(des, stress, void_ratio)
            f_function_new = self.ComputeYieldFunction(stress_new, pc)
            
            if ti.abs(f_function_new) <= FTOL:
                break
            if f_function_new * f_function0 < 0.:                    
                alpha1 = alpha
                f_function1 = f_function0
            else:
                f_function1 = f_function1 * f_function0 / (f_function0 + f_function_new)
            
            f_function0 = f_function_new
            alpha0 = alpha
        return alpha

    @ti.func
    def RegulaFalsiNegativePlasticMultiplier(self, dstrain, stress, f_function0, void_ratio, pc):
        alpha = 0.
        alpha0, alpha1 = 0., 1.
        f_function_save = f_function0
        for _ in range(ITS):
            dalpha = (alpha1 - alpha0) / NSUB
            flag = 0
            for __ in range(NSUB):
                alpha = alpha0 + dalpha
                des = alpha * dstrain
                dstress = self.ComputeTrialStress(des, stress, void_ratio)
                f_function_new = self.ComputeYieldFunction(stress + dstress, pc)
                if f_function_new > FTOL:
                    alpha1 = alpha
                    if f_function0 < -FTOL:
                        for ___ in range(MAXITS):
                            unloading_factor = alpha1 - f_function_new * (alpha1 - alpha0) / (f_function_new - f_function0)
                            dstr = dstrain * unloading_factor
                            dstress = self.ComputeTrialStress(dstr, stress, void_ratio)
                            f_function2 = self.ComputeYieldFunction(stress + dstress, pc)
                            
                            if ti.abs(f_function2) < 1e-8: break

                            if f_function2 * f_function0 < 0.:
                                alpha1 = alpha0
                                f_function_new = f_function0
                            else:
                                f_function_new = f_function_new * f_function0 / (f_function0 + f_function2)

                            alpha0 = unloading_factor
                            f_function0 = f_function2
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
        return alpha
    
    @ti.func
    def ComputePKStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        previous_stress = self.PK2CauchyStress(np, stateVars, previous_stress)
        stress = self.ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
        return self.Cauchy2PKStress(np, stateVars, stress)
    
    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stiffness, stateVars):
        kappa = self.kappa
        possion = self.possion
        void_ratio = stateVars[np].void_ratio
        stress = -current_stress
        p = MeanStress(stress)
        
        bulk_modulus = ti.max((1. + void_ratio) / kappa * p, 100)
        shear_modulus = 3. * bulk_modulus * (1 - 2 * possion) / (2 * (1 + possion))
        ComputeElasticStiffnessTensor(np, bulk_modulus, shear_modulus, stiffness)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stiffness, stateVars):
        pass

@ti.kernel
def find_max_bulk_modulus(materialID: int, particleNum: ti.types.ndarray(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template()) -> float:
    bulk_modulus = 100.
    for np in range(particleNum[0]):
        materialID = particle[np].materialID
        if materialID == materialID:
            kappa = matProps[materialID].kappa
            lambda_ = matProps[materialID].lambda_
            e_ini = matProps[materialID].e0
            pc = stateVars.pc[np]
            stress = -particle[np].stress
            p = -MeanStress(stress)
            void_ratio = e_ini - lambda_ * ti.log(pc / matProps[materialID].p_ref) + kappa * ti.log(pc / p)
            ti.atomic_max(bulk_modulus, (1. + void_ratio) / kappa * p)
    return bulk_modulus


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), epstrain: ti.types.ndarray(), pc: ti.types.ndarray(), void_ratio: ti.types.ndarray(), strain: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].epstrain = epstrain[np]
        state_vars[np].pc = pc[np]
        state_vars[np].void_ratio = void_ratio[np]
        state_vars[np].strain = strain[np]