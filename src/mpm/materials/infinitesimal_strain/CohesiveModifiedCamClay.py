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
        subloading_u = DictIO.GetEssential(material, 'SubloadingSurfaceRatio')

        bonding = DictIO.GetAlternative(material, 'Bonging', False)
        s_h = DictIO.GetEssential(material, 'HydrateSaturation')
        mc_a = DictIO.GetEssential(material, 'Coefficient1')
        mc_b = DictIO.GetEssential(material, 'Coefficient2')
        mc_c = DictIO.GetEssential(material, 'Coefficient3')
        mc_d = DictIO.GetEssential(material, 'Coefficient4')
        m_degradation = DictIO.GetEssential(material, 'Degradtion')
        m_shear = DictIO.GetEssential(material, 'ModulusIncrement')

        self.matProps[materialID].add_material(density, possion, m_theta, kappa, lambda_, e_ref, p_ref, p0, subloading, subloading_u,
                                               bonding, s_h, mc_a, mc_b, mc_c, mc_d, m_degradation, m_shear)
        self.matProps[materialID].print_message(materialID)

    def get_state_vars_dict(self, start_particle, end_particle):
        pc = np.ascontiguousarray(self.stateVars.pc.to_numpy()[start_particle:end_particle])
        dpdstrain = np.ascontiguousarray(self.stateVars.dpdstrain.to_numpy()[start_particle:end_particle])
        dpvstrain = np.ascontiguousarray(self.stateVars.dpvstrain.to_numpy()[start_particle:end_particle])
        pdstrain = np.ascontiguousarray(self.stateVars.pdstrain.to_numpy()[start_particle:end_particle])
        pvstrain = np.ascontiguousarray(self.stateVars.pvstrain.to_numpy()[start_particle:end_particle])
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        void_ratio = np.ascontiguousarray(self.stateVars.void_ratio.to_numpy()[start_particle:end_particle])
        chi = np.ascontiguousarray(self.stateVars.chi.to_numpy()[start_particle:end_particle])
        subloading = np.ascontiguousarray(self.stateVars.subloading.to_numpy()[start_particle:end_particle])
        return {'pc': pc, 'pdstrain': pdstrain, 'pvstrain': pvstrain, 'dpdstrain': dpdstrain, 'dpvstrain': dpvstrain, 'estress': estress, 
                'void_ratio': void_ratio, 'chi': chi, 'subloading': subloading}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        dpdstrain = state_vars.item()['dpdstrain']
        dpvstrain = state_vars.item()['dpvstrain']
        pdstrain = state_vars.item()['pdstrain']
        pvstrain = state_vars.item()['pvstrain']
        void_ratio = state_vars.item()['void_ratio']
        chi = state_vars.item()['chi']
        subloading = state_vars.item()['csubloadinghi']
        pc = state_vars.item()['pc']
        kernel_reload_state_variables(estress, dpvstrain, dpdstrain, pvstrain, pdstrain, void_ratio, pc, chi, subloading, self.stateVars)
    
    def get_lateral_coefficient(self, materialID):
        return 0.9


@ti.dataclass
class ULStateVariable:
    dpdstrain: float
    dpvstrain: float
    pdstrain: float
    pvstrain: float
    estress: float
    pc: float
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
        self.chi = 1.
        self.subloading = 1.

    @ti.func
    def _update_vars(self, stress, dpdstrain, devstrain, pc, void_ratio):
        self.estress = VonMisesStress(-stress)
        self.dpdstrain = dpdstrain
        self.devstrain = devstrain
        self.pc = pc
        self.void_ratio = void_ratio


@ti.dataclass
class TLStateVariable:
    dpdstrain: float
    dpvstrain: float
    pdstrain: float
    pvstrain: float
    estress: float
    pc: float
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
        self.chi = 1.
        self.subloading = 1.
        self.deformation_gradient = DELTA
        self.stress = matrix_form(stress)

    @ti.func
    def _update_deformation_gradient(self, deformation_gradient_rate, dt):
        self.deformation_gradient += deformation_gradient_rate * dt[None]

    @ti.func
    def _update_vars(self, stress, dpdstrain, devstrain, pc, void_ratio):
        self.estress = VonMisesStress(-stress)
        self.dpdstrain = dpdstrain
        self.devstrain = devstrain
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
        lode = ComputeLodeAngle(stress)
        p = -MeanStress(stress)
        q = EquivalentStress(stress)
        return p, q, lode
    
    @ti.func
    def ComputeBondingParameters(self, np, stateVars):
        chi_n = stateVars[np].chi
        chi = chi_n - self.m_degradation * chi_n * stateVars[np].dpdstrain
        chi = clamp(0., 1., chi)
        stateVars[np].chi = chi
        pcd = self.mc_a * (chi * self.s_h) ** self.mc_b
        pcc = self.mc_c * (chi * self.s_h) ** self.mc_d
        return pcd, pcc

    @ti.func
    def yield_function(self, p, q, m, pc, pcd, pcc, subloading_r):
        return (q / m) * (q / m) + (p + pcc) * (p - subloading_r * (pc + pcd + pcc))
    
    @ti.func
    def ComputeElasticParameter(self, np, stress, stateVars):
        p = -MeanStress(stress)
        bulk_modulus = ti.max((1 + stateVars[np].void_ratio) / self.kappa * p, 100)
        shear_modulus = 3 * bulk_modulus * (1 - 2. * self.possion) / (2 * (1 + self.possion))
        if int(self.bonding) == 1:
            shear_modulus += self.m_shear * stateVars[np].chi * self.s_h
            bulk_modulus += shear_modulus * (2 * (1 + self.possion)) / (1 - 2 * self.possion) / 3.
        return bulk_modulus, shear_modulus
    
    @ti.func
    def ComputeBondingParameters(self, dpdstrain, chi_n):
        chi = chi_n - self.m_degradation * chi_n * dpdstrain
        chi = clamp(0., 1., chi)
        pcd = self.mc_a * (chi * self.s_h) ** self.mc_b
        pcc = self.mc_c * (chi * self.s_h) ** self.mc_d
        return chi, pcd, pcc

    @ti.func
    def ComputeSubloadingParameters(self, p, pc, pcd, pcc, dpvstrain, dpdstrain, subloading_n, subloading):
        if ti.abs(subloading - 1.0) < Threshold:
            subloading = p / (pc + pcd + pcc)
        else:
            subloading = subloading_n - self.subloading_u * (1 + (pcd + pcc) / pc) * ti.log(subloading) * ti.sqrt(dpvstrain * dpvstrain + dpdstrain * dpdstrain)

        if subloading < Threshold:
            subloading = 1e-5
        elif subloading > 1.:
            subloading = 1.
        return subloading

    @ti.func
    def ComputeDfDmul(self, p, q, pc, delta_phi, pcd, pcc, void_ratio, bulk_modulus, shear_modulus):
        df_dp = 2 * p - pc - pcd
        df_dq = 2 * q / (self.m_theta * self.m_theta)
        df_dpc = - p - pcc
        upsilon = (1 + void_ratio) / (self.lambda_ - self.kappa)
        a_den = 1 + (2 * bulk_modulus + upsilon * (pc + pcd)) * delta_phi
        dp_dmul = -bulk_modulus * (2 * p - pc) / a_den
        dpc_dmul = upsilon * (pc) * (2 * p - pc - pcd) / a_den
        dq_dmul = -q / (delta_phi + self.m_theta * self.m_theta / (6 * shear_modulus))
        df_dpcd = -p - pcc
        df_dpcc = -2. * pcc - pc - pcd

        dpcd_dmul = 0.
        dpcc_dmul = 0.
        if int(self.subloading) == 1:
            if pcd > Threshold: 
                dpcd_dmul = -ti.sqrt(6) * self.mc_b * self.m_degradation * q / (6. * shear_modulus * delta_phi + self.m_theta * self.m_theta) * pcc
            
            if pcc > Threshold:
                dpcc_dmul = -ti.sqrt(6) * self.mc_d * self.m_degradation * q / (6. * shear_modulus * delta_phi + self.m_theta * self.m_theta) * pcc
            dpc_dmul -= dpcd_dmul
        return df_dp * dp_dmul + df_dq * dq_dmul + df_dpc * dpc_dmul + df_dpcd * dpcd_dmul + df_dpcc * dpcc_dmul

    @ti.func
    def ComputeDgDpc(self, p_trial, pc_n, pc, pcd, void_ratio, delta_phi, bulk_modulus):
        upsilon = (1 + void_ratio) / (self.lambda_ - self.kappa)
        a_den = 1 + 2 * delta_phi * bulk_modulus
        scalar = pc_n * ti.exp(e_index)
        sine = upsilon * delta_phi
        e_index = sine * (2 * p_trial - pc - pcd) / a_den
        g_func = scalar - pc
        dg_dpc = scalar * (-sine / a_den) - 1
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
    def core(self, np, previous_stress, de, dw, stateVars): 
        # !--Initialise state variables----!
        m_theta = self.m_theta
        
        stress = previous_stress
        pc_n = stateVars[np].pc
        void_ratio = self.e_ref - self.lambda_ * ti.log(pc_n / self.p_ref) + self.kappa * ti.log(pc_n / p)
        chi_n = stateVars[np].chi
        subloading_n = stateVars[np].subloading
        dpdstrain = stateVars[np].dpdstrain
        dpvstrain = stateVars[np].dpvstrain

        # !-- trial elastic stresses ----!
        bulk_modulus, shear_modulus = self.ComputeElasticParameter(np, stress, stateVars)
        sigrot = Sigrot(stress, dw)
        dstress = ElasticTensorMultiplyVector(de, bulk_modulus, shear_modulus)
        trial_stress = stress + dstress 

        # !-- compute trial stress invariants ----!
        p_trial, q_trial, theta_trial = self.ComputeStressInvariants(trial_stress)

        pcd, pcc = 0., 0., 0.
        if int(self.bonding) == 1:
            chi, pcd, pcc = self.ComputeBondingParameters(dpdstrain, chi_n)
        
        if int(self.subloading) == 1:
            subloading = self.ComputeSubloadingParameters(p_trial, pc_n, pcd, pcc, dpvstrain, dpdstrain, subloading_n, subloading_n)

        f_func = self.yield_function(p_trial, q_trial, m_theta, pc_n, pcd, pcc, stateVars[np].subloading)
        
        updated_stress = trial_stress
        if f_func < Threshold:
            updated_stress += sigrot
            stateVars[np].estress = VonMisesStress(trial_stress + sigrot)
        else:
            counter_f = 0
            delta_phi = 0.
            p = p_trial
            q = q_trial
            pc = pc_n
            n_trial = self.ComputeDeviatoricStressTensor(p_trial, q_trial, trial_stress)

            while ti.abs(f_func) > Ftolerance and counter_f < itrstep:
                df_dmul = self.ComputeDfDmul(p, q, pc, delta_phi, pcd, pcc, void_ratio, bulk_modulus, shear_modulus)
                delta_phi -= f_func / df_dmul

                counter_g = 0
                g_function, dg_dpc = self.ComputeDgDpc(p_trial, pc_n, pc, pcd, void_ratio, delta_phi, bulk_modulus)
                while ti.abs(g_function) > Gtolerance and counter_g < itrstep:
                    pc -= g_function / dg_dpc
                    g_function, dg_dpc = self.ComputeDgDpc(p_trial, pc_n, pc, pcd, void_ratio, delta_phi, bulk_modulus)
                    counter_g += 1

                p = (p_trial + bulk_modulus * delta_phi * pc) / (1 + 2 * bulk_modulus * delta_phi)
                q = q_trial / (1 + 6 * shear_modulus * delta_phi / (m_theta ** 2))
                dpvstrain = delta_phi * (2. * p - pc - pcd)
                dpdstrain = delta_phi * (ti.sqrt(6) * q / (m_theta * m_theta))

                if int(self.bonding) == 1:
                    chi, pcd, pcc = self.ComputeBondingParameters(dpdstrain, chi_n)
                
                if int(self.subloading) == 1:
                    subloading = self.ComputeSubloadingParameters(p_trial, pc_n, pcd, pcc, dpvstrain, dpdstrain, subloading_n, subloading)

                f_func = self.yield_function(p, q, m_theta, pc, pcd, pcc, subloading)
                counter_f +=1
            
            updated_stress = q * n_trial
            for j in range(3):
                updated_stress[j] -= p
            updated_stress += sigrot

            stateVars[np].void_ratio = self.e_ref - self.lambda_ * ti.log(pc_n / self.p_ref) + self.kappa * ti.log(pc_n / p)
            stateVars[np].chi = chi
            stateVars[np].subloading = subloading
            stateVars[np].pc = pc
            stateVars[np].estress = VonMisesStress(updated_stress)
            stateVars[np].dpvstrain = dpvstrain
            stateVars[np].dpdstrain = dpdstrain
            stateVars[np].pvstrain += dpvstrain
            stateVars[np].pdstrain += dpdstrain
        return updated_stress

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stiffness, stateVars):
        bulk_modulus, shear_modulus = self.ComputeElasticParameter(np, current_stress, stateVars)
        ComputeElasticStiffnessTensor(np, bulk_modulus, shear_modulus, stiffness)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stiffness, stateVars):
        lambda_ = self.lambda_
        kappa = self.kappa
        possion = self.possion
        m_theta = self.m_theta
        e_ini = self.e0

        pc = stateVars.pc[np]
        stress = -current_stress
        p = -MeanStress(stress)
        
        e0 = e_ini - lambda_ * ti.log(pc / self.p_ref) + kappa * ti.log(pc / p) # initial void ratio
        bulk_modulus, shear_modulus = self.update_modulus(stress, e0, kappa, possion)
        
        q = EquivalentStress(stress)
        df_dp = 2 * p - pc
        df_dq = 2 * q / (m_theta * m_theta)
        df_dpc = -p 
        upsilon = (1 + e0) / (lambda_ - kappa)
        a1 = (bulk_modulus * df_dp) * (bulk_modulus * df_dp)
        a2 = -ti.sqrt(6) * bulk_modulus * df_dp * shear_modulus * df_dq
        a3 = 6 * (shear_modulus * df_dq) * (shear_modulus * df_dq)
        num = bulk_modulus * (df_dp * df_dp) + 3 * shear_modulus * (df_dq * df_dq)
        hardening_par = upsilon * pc * df_dp * df_dpc

        dev_stress = ti.Vector([stress[0], stress[1], stress[2], stress[3], stress[4], stress[5]])
        xi = q / ti.sqrt(1.5)
        n_l = ZEROVEC6f
        l_n = ZEROVEC6f
        l_l = ZEROVEC6f
        n_n = ZEROVEC6f
        if xi > Threshold:
            for i, j in range((0, 3), (0, 6)):
                l_n[i, j] = dev_stress[j] / xi
            for i, j in range((0, 6), (0, 3)):
                n_l[i, j] = dev_stress[i] / xi
            for i, j in range((0, 6), (0, 6)):
                n_n[i, j] = dev_stress[i] * dev_stress[j] / (xi * xi)
            for i, j in range((0, 3), (0, 3)):
                l_l[i, j] = 1

        pstiffness = (a1 * l_l + a2 * (n_l + l_n) + a3 * (n_n)) / (num - hardening_par)
        return pstiffness
    
    @ti.func
    def ComputePKStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        previous_stress = self.PK2CauchyStress(np, stateVars, previous_stress)
        stress = self.ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
        return self.Cauchy2PKStress(np, stateVars, stress)


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
def kernel_reload_state_variables(estress: ti.types.ndarray(), dpvstrain: ti.types.ndarray(), dpdstrain: ti.types.ndarray(), pvstrain: ti.types.ndarray(), pdstrain: ti.types.ndarray(), 
                                  void_ratio: ti.types.ndarray(), pc: ti.types.ndarray(), chi: ti.types.ndarray(), subloading: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].dpvstrain = dpvstrain[np]
        state_vars[np].dpdstrain = dpdstrain[np]
        state_vars[np].pvstrain = pvstrain[np]
        state_vars[np].pdstrain = pdstrain[np]
        state_vars[np].void_ratio = void_ratio[np]
        state_vars[np].pc = pc[np]
        state_vars[np].chi = chi[np]
        state_vars[np].subloading = subloading[np]