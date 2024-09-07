import numpy as np
import taichi as ti

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.MaterialKernel import *
from src.utils.constants import (DELTA, PI, Threshold, ZEROVEC6f, itrstep, substep, Ftolerance, Gtolerance)
from src.utils.MatrixFunction import matrix_form
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import mat3x3
from src.utils.VectorFunction import voigt_form


class CohesiveModifiedCamClay(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_particle_num, configuration="ULMPM", solver_type="Explicit"):
        super().__init__()
        self.matProps = CohesiveModifiedCamClayModel.field(shape=max_material_num)
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
        lambda_ = DictIO.GetEssential(material, 'lambda')
        kappa = DictIO.GetEssential(material, 'kappa')
        
        if 'StressRatio' in material:
            m_theta = DictIO.GetEssential(material, 'StressRatio')
        elif 'CSFriction' in material:
            fric = DictIO.GetEssential(material, 'CSFriction') * PI / 180.
            sinc_friction_cs = ti.sin(fric)
            m_theta = (6. * sinc_friction_cs) / (3. - sinc_friction_cs)
        else:
            raise RuntimeError("Critical state ratio is not defined")
        
        e_ref = DictIO.GetEssential(material, 'void_ratio_ref')
        p_ref = DictIO.GetAlternative(material, 'pressure_ref', 1000.)
        p0 = DictIO.GetEssential(material, 'ConsolidationPressure')

        subloading = DictIO.GetAlternative(material, 'Subloading', False)
        subloading_u = DictIO.GetAlternative(material, 'SubloadingSurfaceRatio', 0.)

        bonding = DictIO.GetAlternative(material, 'Bonding', False)
        s_h = DictIO.GetAlternative(material, 'HydrateSaturation', 0.)
        mc_a = DictIO.GetAlternative(material, 'Coefficient1', 0.)
        mc_b = DictIO.GetAlternative(material, 'Coefficient2', 0.)
        mc_c = DictIO.GetAlternative(material, 'Coefficient3', 0.)
        mc_d = DictIO.GetAlternative(material, 'Coefficient4', 0.)
        m_degradation = DictIO.GetAlternative(material, 'Degradation', 0.)
        m_shear = DictIO.GetAlternative(material, 'ModulusIncrement', 0.)

        self.matProps[materialID].add_material(density, possion, m_theta, kappa, lambda_, e_ref, p_ref, p0, subloading, subloading_u,
                                               bonding, s_h, mc_a, mc_b, mc_c, mc_d, m_degradation, m_shear)
        self.matProps[materialID].print_message(materialID)

    def get_state_vars_dict(self, start_particle, end_particle):
        pc = np.ascontiguousarray(self.stateVars.pc.to_numpy()[start_particle:end_particle])
        pcc = np.ascontiguousarray(self.stateVars.pcc.to_numpy()[start_particle:end_particle])
        pcd = np.ascontiguousarray(self.stateVars.pcd.to_numpy()[start_particle:end_particle])
        epstrain = np.ascontiguousarray(self.stateVars.epstrain.to_numpy()[start_particle:end_particle])
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        void_ratio0 = np.ascontiguousarray(self.stateVars.void_ratio0.to_numpy()[start_particle:end_particle])
        void_ratio = np.ascontiguousarray(self.stateVars.void_ratio.to_numpy()[start_particle:end_particle])
        chi = np.ascontiguousarray(self.stateVars.chi.to_numpy()[start_particle:end_particle])
        subloading = np.ascontiguousarray(self.stateVars.subloading.to_numpy()[start_particle:end_particle])
        return {'pc': pc, 'pcc': pcc, 'pcd': pcd, 'epstrain': epstrain, 'estress': estress, 'void_ratio': void_ratio, 'void_ratio0': void_ratio0, 'chi': chi, 'subloading': subloading}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        epstrain = state_vars.item()['epstrain']
        void_ratio0 = state_vars.item()['void_ratio0']
        void_ratio = state_vars.item()['void_ratio']
        chi = state_vars.item()['chi']
        subloading = state_vars.item()['csubloadinghi']
        pc = state_vars.item()['pc']
        pcc = state_vars.item()['pcc']
        pcd = state_vars.item()['pcd']
        kernel_reload_state_variables(estress, epstrain, void_ratio0, void_ratio, pc, pcc, pcd, chi, subloading, self.stateVars)
    
    def get_lateral_coefficient(self, materialID):
        return 0.9


@ti.dataclass
class ULStateVariable:
    epstrain: float
    estress: float
    pc: float
    pcc: float
    pcd: float
    void_ratio0: float
    void_ratio: float
    chi: float
    subloading: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        p = ti.max(100, -MeanStress(stress))
        materialID = int(particle[np].materialID)
        self.pc = matProps[materialID].p0
        self.estress = VonMisesStress(stress)
        self.void_ratio = matProps[materialID].e_ref - matProps[materialID].lambda_ * ti.log(self.pc / matProps[materialID].p_ref) + matProps[materialID].kappa * ti.log(self.pc / p)
        self.void_ratio0 = self.void_ratio
        self.chi = 1.
        self.subloading = 1.

    @ti.func
    def _update_vars(self, stress, epstrain, pc, void_ratio):
        self.estress = VonMisesStress(-stress)
        self.epstrain = epstrain
        self.pc = pc
        self.void_ratio = void_ratio


@ti.dataclass
class TLStateVariable:
    epstrain: float
    estress: float
    pc: float
    pcc: float
    pcd: float
    void_ratio0: float
    void_ratio: float
    chi: float
    deformation_gradient: mat3x3
    stress: mat3x3
    subloading: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        p = ti.max(100, -MeanStress(stress))
        materialID = int(particle[np].materialID)
        self.pc = matProps[materialID].p0
        self.estress = VonMisesStress(stress)
        self.void_ratio = matProps[materialID].e_ref - matProps[materialID].lambda_ * ti.log(self.pc / matProps[materialID].p_ref) + matProps[materialID].kappa * ti.log(self.pc / p)
        self.void_ratio0 = self.void_ratio
        self.chi = 1.
        self.subloading = 1.
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


@ti.dataclass
class CohesiveModifiedCamClayModel:
    density: float
    possion: float
    m_theta: float
    kappa: float
    lambda_: float
    e_ref: float
    p_ref: float
    p0: float
    subloading: ti.u8
    subloading_u: float
    bonding: ti.u8
    s_h: float
    mc_a: float
    mc_b: float
    mc_c: float
    mc_d: float
    m_degradation: float
    m_shear: float

    def add_material(self, density, possion, m_theta, kappa, lambda_, e_ref, p_ref, p0, subloading, subloading_u, bonding, s_h, mc_a, mc_b, mc_c, mc_d, m_degradation, m_shear):
        self.density = density
        self.possion = possion
        self.m_theta = m_theta
        self.kappa = kappa
        self.lambda_ = lambda_
        self.e_ref = e_ref
        self.p_ref = p_ref
        self.p0 = p0
        self.subloading = subloading
        self.subloading_u = subloading_u
        self.bonding = bonding
        self.s_h = s_h
        self.mc_a = mc_a
        self.mc_b = mc_b
        self.mc_c = mc_c
        self.mc_d = mc_d
        self.m_degradation = m_degradation
        self.m_shear = m_shear

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model = Cohesive Modified Cam-Clay Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Possion Ratio: ', self.possion)
        print('Critical Stress Ratio = ', self.m_theta)
        print('Compression index = ', self.lambda_)
        print('Swelling index = ', self.kappa)
        print('Initial void ratio = ', self.e_ref)
        print('Hydrate saturation = ', self.s_h)
        print('Coefficient1 = ', self.mc_a)
        print('Coefficient2 = ', self.mc_b)
        print('Coefficient3 = ', self.mc_c)
        print('Coefficient4 = ', self.mc_d)
        print('Degradation = ', self.m_degradation)
        print('Increment in shear modulus = ', self.m_shear, '\n')

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
    #                        Implicit stress integration                            #
    # ============================================================================= #
    @ti.func
    def ComputeStressInvariants(self, stress):
        p = -MeanStress(stress)
        q = EquivalentStress(stress)
        return p, q
    
    @ti.func
    def ComputeBondingParameters(self, chi_n, dpdstrain, stateVars):
        chi = chi_n - self.m_degradation * chi_n * dpdstrain
        chi = clamp(0., 1., chi)
        stateVars.chi = chi
        pcd = self.mc_a * (chi * self.s_h) ** self.mc_b
        pcc = self.mc_c * (chi * self.s_h) ** self.mc_d
        return pcd, pcc

    @ti.func
    def yield_function(self, p, q, mtheta, pc, pcd, pcc, subloading_r):
        return (q / mtheta) * (q / mtheta) + (p + pcc) * (p - subloading_r * (pc + pcd + pcc))
    
    @ti.func
    def ComputeYieldState(self, p, q, mtheta, pc, pcd, pcc, stateVars):
        subloading_r = stateVars.subloading
        yield_shear = self.yield_function(p, q, mtheta, pc, pcd, pcc, subloading_r)
        return yield_shear > -Threshold, yield_shear
    
    @ti.func
    def ComputeElasticParameter(self, stress, stateVars):
        p = -MeanStress(stress)
        bulk_modulus = ti.max((1 + stateVars.void_ratio) / self.kappa * p, 100)
        shear_modulus = 3 * bulk_modulus * (1 - 2. * self.possion) / (2 * (1 + self.possion))
        if int(self.bonding) == 1:
            shear_modulus += self.m_shear * stateVars.chi * self.s_h
            bulk_modulus += shear_modulus * (2 * (1 + self.possion)) / (1 - 2 * self.possion) / 3.
        return bulk_modulus, shear_modulus

    @ti.func
    def ComputeSubloadingParameters(self, p, pc, pcd, pcc, subloading_n, dpvstrain, dpdstrain, stateVars):
        subloading = stateVars.subloading
        if ti.abs(subloading - 1.0) < Threshold:
            subloading = p / (pc + pcd + pcc)
        else:
            subloading = subloading_n - self.subloading_u * (1 + (pcd + pcc) / pc) * ti.log(subloading_n) * ti.sqrt(dpvstrain * dpvstrain + dpdstrain * dpdstrain)

        if subloading < Threshold:
            subloading = 1e-5
        elif subloading > 1.:
            subloading = 1.
        stateVars.subloading = subloading

    @ti.func
    def ComputeDfDmul(self, p, q, mtheta, pc, pcd, pcc, delta_phi, bulk_modulus, shear_modulus, stateVars):
        void_ratio = stateVars.void_ratio

        df_dp = 2 * p - pc - pcd
        df_dq = 2 * q / (mtheta * mtheta)
        df_dpc = - p - pcc
        upsilon = (1 + void_ratio) / (self.lambda_ - self.kappa)
        a_den = 1 + (2 * bulk_modulus + upsilon * (pc + pcd)) * delta_phi
        dp_dmul = -bulk_modulus * (2 * p - pc - pcd) / a_den
        dpc_dmul = upsilon * (pc + pcd) * (2 * p - pc - pcd) / a_den
        dq_dmul = -q / (delta_phi + (mtheta * mtheta) / (6 * shear_modulus))

        dfdmul = 0.
        if int(self.bonding) == 0:
            dfdmul = df_dp * dp_dmul + df_dq * dq_dmul + df_dpc * dpc_dmul 
        elif int(self.bonding) == 1:
            df_dpcd = -p - pcc
            df_dpcc = -2. * pcc - pc - pcd

            dpcd_dmul, dpcc_dmul = 0., 0.
            if pcd > Threshold: 
                dpcd_dmul = -ti.sqrt(6) * self.mc_b * self.m_degradation * q / (6. * shear_modulus * delta_phi + mtheta * mtheta) * pcd
            if pcc > Threshold:
                dpcc_dmul = -ti.sqrt(6) * self.mc_d * self.m_degradation * q / (6. * shear_modulus * delta_phi + mtheta * mtheta) * pcc
            dpc_dmul -= dpcd_dmul
            dfdmul = df_dp * dp_dmul + df_dq * dq_dmul + df_dpc * dpc_dmul + df_dpcd * dpcd_dmul + df_dpcc * dpcc_dmul
        return dfdmul

    @ti.func
    def ComputeDgDpc(self, p_trial, pc, pc_n, pcd, delta_phi, bulk_modulus, stateVars):
        void_ratio = stateVars.void_ratio

        upsilon = (1 + void_ratio) / (self.lambda_ - self.kappa)
        a_den = 1 + 2 * delta_phi * bulk_modulus
        sine = upsilon * delta_phi
        e_index = sine * (2 * p_trial - pc - pcd) / a_den
        scalar = pc_n * ti.exp(e_index)
        g_func = scalar - pc
        dg_dpc = scalar * (-sine / a_den) - 1.
        return g_func, dg_dpc

    @ti.func
    def ComputeDeviatoricStressTensor(self, p, q, stress):
        dev_stress = stress
        for d in ti.static(range(3)):
            dev_stress[d] += p

        n = ZEROVEC6f
        if q > Threshold: 
            n = dev_stress / q
        return n 

    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        de = calculate_strain_increment2D(velocity_gradient, dt)
        dw = calculate_vorticity_increment2D(velocity_gradient, dt)
        return self.core(np, previous_stress, de, dw, stateVars)

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        de = calculate_strain_increment(velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)
        return self.core(np, previous_stress, de, dw, stateVars)
    
    @ti.func
    def ComputeMTheta(self, stress):
        lode = ComputeLodeAngle(stress)
        return self.m_theta - (self.m_theta * self.m_theta) / (3 + self.m_theta) * ti.cos(1.5 * lode)
    
    @ti.func
    def core(self, np, stress, de, dw, stateVars): 
        # !--Initialise state variables----!
        dpvstrain, dpdstrain = 0., 0.
        pc = stateVars[np].pc
        chi_n = stateVars[np].chi
        subloading_n = stateVars[np].subloading

        # !-- trial elastic stresses ----!
        bulk_modulus, shear_modulus = self.ComputeElasticParameter(stress, stateVars[np])
        sigrot = Sigrot(stress, dw)
        dstress = ElasticTensorMultiplyVector(de, bulk_modulus, shear_modulus)
        trial_stress = stress + dstress 
        # !-- compute trial stress invariants ----!
        p_trial, q_trial = self.ComputeStressInvariants(trial_stress)

        pcd, pcc = 0., 0.
        if int(self.bonding) == 1:
            pcd, pcc = self.ComputeBondingParameters(chi_n, dpdstrain, stateVars[np])
        if int(self.subloading) == 1:
            self.ComputeSubloadingParameters(p_trial, pc, pcd, pcc, subloading_n, dpvstrain, dpdstrain, stateVars[np])
        mtheta_trial = self.ComputeMTheta(trial_stress)
        mtheta = mtheta_trial
        updated_stress = trial_stress
        yield_state, f_func = self.ComputeYieldState(p_trial, q_trial, mtheta_trial, pc, pcd, pcc, stateVars[np])
        if yield_state == 1:
            counter_f = 0
            delta_phi = 0.
            p = p_trial
            q = q_trial
            pc_n = pc
            n_trial = self.ComputeDeviatoricStressTensor(p_trial, q_trial, trial_stress)

            while ti.abs(f_func) > Ftolerance and counter_f < itrstep:
                mtheta = mtheta_trial
                df_dmul = self.ComputeDfDmul(p, q, mtheta, pc, pcd, pcc, delta_phi, bulk_modulus, shear_modulus, stateVars[np])
                delta_phi -= f_func / df_dmul

                counter_g = 0
                g_function, dg_dpc = self.ComputeDgDpc(p_trial, pc, pc_n, pcd, delta_phi, bulk_modulus, stateVars[np])
                while ti.abs(g_function) > Gtolerance and counter_g < substep:
                    pc -= g_function / dg_dpc
                    g_function, dg_dpc = self.ComputeDgDpc(p_trial, pc, pc_n, pcd, delta_phi, bulk_modulus, stateVars[np])
                    counter_g += 1

                p = (p_trial + bulk_modulus * delta_phi * pc) / (1 + 2 * bulk_modulus * delta_phi)
                q = q_trial / (1 + 6 * shear_modulus * delta_phi / (mtheta ** 2))
                dpvstrain = delta_phi * (2. * p - pc - pcd)
                dpdstrain = delta_phi * (ti.sqrt(6) * q / (mtheta * mtheta))

                if int(self.bonding) == 1:
                    pcd, pcc = self.ComputeBondingParameters(chi_n, dpdstrain, stateVars[np])
                if int(self.subloading) == 1:
                    self.ComputeSubloadingParameters(p, pc, pcd, pcc, subloading_n, dpvstrain, dpdstrain, stateVars[np])

                updated_stress = q * n_trial
                for j in ti.static(range(3)):  updated_stress[j] -= p
                mtheta = self.ComputeMTheta(updated_stress)
                _, f_func = self.ComputeYieldState(p, q, mtheta, pc, pcd, pcc, stateVars[np])
                counter_f +=1
            updated_stress = q * n_trial
            for j in ti.static(range(3)):  updated_stress[j] -= p
            stateVars[np].void_ratio += (de[0] + de[1] + de[2]) * (1 + stateVars[np].void_ratio0)

        updated_stress += sigrot
        stateVars[np].pc = pc
        stateVars[np].pcc = pcc
        stateVars[np].pcd = pcd
        stateVars[np].estress = VonMisesStress(updated_stress)
        stateVars[np].epstrain += ti.sqrt(dpvstrain * dpvstrain + dpdstrain * dpdstrain)
        return updated_stress

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        bulk_modulus, shear_modulus = self.ComputeElasticParameter(current_stress, stateVars[np])
        return ComputeElasticStiffnessTensor(bulk_modulus, shear_modulus)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stateVars):
        pc = stateVars[np].pc
        pcc = stateVars[np].pcc
        pcd = stateVars[np].pcd
        epstrain = stateVars[np].epstrain
        void_ratio = stateVars[np].void_ratio
        subloading = stateVars[np].subloading
        p, q = self.ComputeStressInvariants(current_stress)
        bulk_modulus, shear_modulus = self.ComputeElasticParameter(current_stress, stateVars[np])
        
        mtheta = self.ComputeMTheta(current_stress)
        df_dp = 2 * p - pc - pcd
        df_dq = 2 * q / (mtheta * mtheta)
        df_dpc = -p - pcc
        df_dpcd = -p - pcc
        df_dpcc = -2 * pcc - pc - pcd
        if int(self.subloading) == 1:
            df_dp = 2 * p + pcc - subloading * (pc + pcc + pcd)
            df_dpc *= subloading
            df_dpcd *= subloading
            df_dpcc = p - subloading * (p + pc + pcd + 2 * pcc)

        upsilon = (1 + void_ratio) / (self.lambda_ - self.kappa)
        a1 = (bulk_modulus * df_dp) * (bulk_modulus * df_dp)
        a2 = -ti.sqrt(6) * bulk_modulus * df_dp * shear_modulus * df_dq
        a3 = 6 * (shear_modulus * df_dq) * (shear_modulus * df_dq)
        num = bulk_modulus * (df_dp * df_dp) + 3 * shear_modulus * (df_dq * df_dq)

        hardening_par = upsilon * pc * df_dp * df_dpc
        if int(self.bonding) == 1: 
            hardening_pcd = df_dpcd * (-self.m_degradation * self.mc_b * pcd) * df_dq
            hardening_pcc = df_dpcc * (-self.m_degradation * self.mc_d * pcc) * df_dq
            hardening_par += (hardening_pcd + hardening_pcc)
        if int(self.subloading) == 1:
            df_dr = -(p + pcc) * (pc + pcd + pcc)
            hardening_subloading = -df_dr * self.subloading_u * (1 + (pcd + pcc) / pc) * ti.log(subloading) * epstrain
            hardening_par += hardening_subloading

        dev_stress = vec6f(current_stress[0], current_stress[1], current_stress[2], current_stress[3], current_stress[4], current_stress[5])
        for i in ti.static(range(3)): dev_stress[i] += p
        xi = q / ti.sqrt(1.5)
        n_l = ZEROMAT6x6
        l_n = ZEROMAT6x6
        l_l = ZEROMAT6x6
        n_n = ZEROMAT6x6
        if xi > Threshold:
            for i, j in ti.static(ti.ndrange((0, 3), (0, 6))):
                l_n[i, j] = dev_stress[j] / xi
            for i, j in ti.static(ti.ndrange((0, 6), (0, 3))):
                n_l[i, j] = dev_stress[i] / xi
            for i, j in ti.static(ti.ndrange((0, 6), (0, 6))):
                n_n[i, j] = dev_stress[i] * dev_stress[j] / (xi * xi)
            for i, j in ti.static(ti.ndrange((0, 3), (0, 3))):
                l_l[i, j] = 1
        stiffness_matrix = self.compute_elastic_tensor(np, current_stress, stateVars)
        pstiffness = (a1 * l_l + a2 * (n_l + l_n) + a3 * (n_n)) / (num - hardening_par)
        return stiffness_matrix - pstiffness
    
    @ti.func
    def ComputePKStress(self, np, velocity_gradient, stateVars, dt):  
        previous_stress = self.PK2CauchyStress(np, stateVars)
        cauchy_stress = self.ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
        PKstress = self.Cauchy2PKStress(np, stateVars, cauchy_stress)
        stateVars[np].stress = PKstress
        return PKstress


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
def kernel_reload_state_variables(estress: ti.types.ndarray(), epstrain: ti.types.ndarray(), void_ratio0: ti.types.ndarray(), void_ratio: ti.types.ndarray(), 
                                  pc: ti.types.ndarray(), pcc: ti.types.ndarray(), pcd: ti.types.ndarray(), chi: ti.types.ndarray(), subloading: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].epstrain = epstrain[np]
        state_vars[np].void_ratio0 = void_ratio0[np]
        state_vars[np].void_ratio = void_ratio[np]
        state_vars[np].pc = pc[np]
        state_vars[np].pcc = pcc[np]
        state_vars[np].pcd = pcd[np]
        state_vars[np].chi = chi[np]
        state_vars[np].subloading = subloading[np]