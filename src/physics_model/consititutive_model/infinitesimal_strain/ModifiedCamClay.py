import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.SoftenModel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.utils.constants import FTOL, Ftolerance, Gtolerance, itrstep, substep
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec2f
from src.utils.VectorFunction import voigt_tensor_trace, voigt_tensor_dot
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class ModifiedCamClayModel(PlasticMaterial):
    def __init__(self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type, stress_integration)
        self.m_theta = 0.
        self.kappa = 0.
        self.lambda_ = 0.
        self.e_ref = 0.
        self.p_ref = 0.
        self.ocr = 0.
        self.pc = 0.
        self.e0 = -1.
        self.three_invariants = False
        self.subloading = False
        self.subloading_u = 0.
        self.bonding = False
        self.mc_a = 0.
        self.mc_b = 0.
        self.mc_c = 0.
        self.mc_d = 0.
        self.m_degradation = 0.
        self.m_shear = 0.
        self.s_h = 0.

    def model_initialize(self, material):
        self.material = material
        density = DictIO.GetAlternative(material, 'Density', 2650)
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        m_theta = DictIO.GetEssential(material, 'StressRatio')
        lambda_ = DictIO.GetEssential(material, 'lambda')
        kappa = DictIO.GetEssential(material, 'kappa')
        ocr = DictIO.GetAlternative(material, 'OverConsolidationRatio', -1)
        pc = 1000.
        if ocr < 0:
            pc = DictIO.GetEssential(material, 'pc', 'pc0', 'ConsolidationPressure')
        e_ref = DictIO.GetEssential(material, 'void_ratio_ref')
        p_ref = DictIO.GetAlternative(material, 'pressure_ref', 1000.)
        e0 = DictIO.GetAlternative(material, 'void_ratio_initial', DictIO.GetAlternative(material, 'InitialVoidRatio', -1.))
        porosity = DictIO.GetAlternative(material, 'porosity', -1.)
        if porosity > 0.:
            e0 = porosity / (1. - porosity)
        three_invariants = DictIO.GetAlternative(material, 'three_invariants', DictIO.GetAlternative(material, 'ThreeInvariants', False))
        subloading = DictIO.GetAlternative(material, 'subloading', DictIO.GetAlternative(material, 'Subloading', False))
        subloading_u = DictIO.GetAlternative(material, 'subloading_u', DictIO.GetAlternative(material, 'SubloadingU', 0.))
        bonding = DictIO.GetAlternative(material, 'bonding', DictIO.GetAlternative(material, 'Bonding', False))
        bonding = DictIO.GetAlternative(material, 'bounding_surface', DictIO.GetAlternative(material, 'BoundingSurface', bonding))
        mc_a = DictIO.GetAlternative(material, 'mc_a', DictIO.GetAlternative(material, 'MCA', 0.))
        mc_b = DictIO.GetAlternative(material, 'mc_b', DictIO.GetAlternative(material, 'MCB', 0.))
        mc_c = DictIO.GetAlternative(material, 'mc_c', DictIO.GetAlternative(material, 'MCC', 0.))
        mc_d = DictIO.GetAlternative(material, 'mc_d', DictIO.GetAlternative(material, 'MCD', 0.))
        m_degradation = DictIO.GetAlternative(material, 'm_degradation', DictIO.GetAlternative(material, 'Degradation', 0.))
        m_shear = DictIO.GetAlternative(material, 'm_shear', DictIO.GetAlternative(material, 'BondedShearModulus', 0.))
        s_h = DictIO.GetAlternative(material, 's_h', DictIO.GetAlternative(material, 'HydrateSaturation', 0.))
        self.soft_function = None
        self.set_rate_dependent_model(material)
        self.add_material(density, poisson, m_theta, kappa, lambda_, e_ref, p_ref, ocr, pc, e0, 
                          three_invariants, subloading, subloading_u, bonding, mc_a, mc_b, mc_c, mc_d, m_degradation, m_shear, s_h)
        self.add_coupling_material(material)

    def add_material(self, density, poisson, m_theta, kappa, lambda_, e_ref, p_ref, ocr, pc, e0,
                     three_invariants, subloading, subloading_u, bonding, mc_a, mc_b, mc_c, mc_d, m_degradation, m_shear, s_h):
        self.density = density
        self.poisson = poisson
        self.m_theta = m_theta
        self.kappa = kappa
        self.lambda_ = lambda_
        self.e_ref = e_ref
        self.p_ref = p_ref
        self.ocr = ocr
        self.pc = pc
        self.e0 = e0
        self.three_invariants = three_invariants
        self.subloading = subloading
        self.subloading_u = subloading_u
        self.bonding = bonding
        self.mc_a = mc_a
        self.mc_b = mc_b
        self.mc_c = mc_c
        self.mc_d = mc_d
        self.m_degradation = m_degradation
        self.m_shear = m_shear
        self.s_h = s_h
        self.max_sound_speed = self.get_sound_speed()
        self.is_soft = True

    def print_message(self, materialID):
        self.print_console_header()
        print('Constitutive model: Modified Cam-Clay')
        print("Material ID: ", materialID)
        print('Density: ', self.density)
        print('Poisson Ratio: ', self.poisson)
        if self.is_rate_dependent:
            self.rate_dependent_function.print_message()
        print('Critical Stress Ratio = ', self.m_theta)
        print('Compression index = ', self.lambda_)
        print('Swelling index = ', self.kappa)
        print('Three invariants = ', self.three_invariants)
        print('Subloading = ', self.subloading)
        print('Bonding = ', self.bonding)
        print('Initial void ratio = ', self.e_ref)
        print('Overconsolidation ratio = ', self.ocr, '\n')

    def define_state_vars(self):
        state_vars = {
            'pc': float,
            'epdstrain': float,
            'bulk_modulus': float,
            'shear_modulus': float,
            'p': float,
            'q': float,
            'void_ratio': float,
            'delta_phi': float,
            'f_function': float,
            'dpvstrain': float,
            'dpdstrain': float,
            'pvstrain': float,
            'pdstrain': float,
        }
        if self.three_invariants:
            state_vars.update({'theta': float})
            if GlobalVariable.RANDOMFIELD is False:
                state_vars.update({'m_theta': float})
        if self.bonding:
            state_vars.update({'chi': float, 'pcd': float, 'pcc': float})
        if self.subloading:
            state_vars.update({'subloading_r': float})
        if GlobalVariable.RANDOMFIELD:
            state_vars.update({'density': float, 'poisson': float, 'lambda_': float, 'kappa': float, 'm_theta': float})
        return state_vars
    
    def get_sound_speed(self):
        return 0.

    def choose_soft_function(self, material):
        raise RuntimeError("")

    def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        return np.repeat(0.9, end_index - start_index)
        '''if GlobalVariable.RANDOMFIELD:
            particle_index = np.ascontiguousarray(materialID.to_numpy()[start_index:end_index])
            m_theta = np.ascontiguousarray(stateVars.m_theta.to_numpy()[particle_index])
            return 1. - (3 * m_theta) / (6. + m_theta)
        else:
            m_theta = self.m_theta
            return np.repeat(1. - (3 * m_theta) / (6. + m_theta), end_index - start_index)'''

    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        stress = particle[np].stress
        pressure = ti.max(-SphericalTensor(stress), 100.)
        if ti.static(self.ocr > 0):
            pc = self.ocr * pressure
            stateVars[np].pc = ti.max(100, pc)
        else:
            stateVars[np].pc = ti.max(100, self.pc)

        void_ratio = self.e_ref - self.lambda_ * ti.log(stateVars[np].pc / self.p_ref)
        if ti.static(self.e0 > 0.):
            void_ratio = self.e0
        elif ti.static(self.ocr > 0.):
            void_ratio = self.e_ref - self.lambda_ * ti.log(pressure / self.p_ref) - self.kappa * ti.log(ti.max(self.ocr, 1.e-12))
        stateVars[np].epdstrain = 0.
        stateVars[np].p = pressure
        stateVars[np].q = EquivalentDeviatoricStress(stress)
        stateVars[np].void_ratio = void_ratio
        stateVars[np].delta_phi = 0.
        stateVars[np].f_function = 0.
        stateVars[np].dpvstrain = 0.
        stateVars[np].dpdstrain = 0.
        stateVars[np].pvstrain = 0.
        stateVars[np].pdstrain = 0.
        if ti.static(self.three_invariants):
            stateVars[np].theta = ComputeLodeAngle(stress)
            if ti.static(not GlobalVariable.RANDOMFIELD):
                stateVars[np].m_theta = self.ComputeMTheta(stateVars[np].theta, self.m_theta)
        if ti.static(self.bonding):
            stateVars[np].chi = 1.
            stateVars[np].pcd = 0.
            stateVars[np].pcc = 0.
        if ti.static(self.subloading):
            stateVars[np].subloading_r = 1.
        material_params = self.GetMaterialParameter(stress, stateVars[np])
        stateVars[np].bulk_modulus, stateVars[np].shear_modulus = self.ComputeElasticModulus(stress, material_params)

    # ==================================================== Modified cam-clay Model ==================================================== #
    @ti.func
    def ComputeElasticModulus(self, stress, material_params):
        poisson, kappa, void_ratio, chi = material_params[0], material_params[2], material_params[4], material_params[5]
        p = ti.max(-SphericalTensor(stress), 100)
        bulk_modulus = (1 + void_ratio) / kappa * p
        shear_modulus = 3. * bulk_modulus * (1 - 2 * poisson) / (2 * (1 + poisson))
        if ti.static(self.bonding):
            shear_modulus += self.m_shear * chi * self.s_h
            bulk_modulus = shear_modulus * 2. * (1. + poisson) / (3. * (1. - 2. * poisson))
        return bulk_modulus, shear_modulus
    
    @ti.func
    def ComputeNonLinearElasticModulus(self, dvolumetric_strain, stress, material_params):
        poisson, kappa, void_ratio, chi = material_params[0], material_params[2], material_params[4], material_params[5]
        p = ti.max(-SphericalTensor(stress), 100)
        safe_dvolumetric_strain = dvolumetric_strain if ti.abs(dvolumetric_strain) > Threshold else Threshold
        bulk_modulus = -p / safe_dvolumetric_strain * (ti.exp(-(1. + void_ratio) * safe_dvolumetric_strain / kappa) - 1.)
        shear_modulus = 3. * bulk_modulus * (1 - 2 * poisson) / (2 * (1 + poisson))
        if ti.static(self.bonding):
            shear_modulus += self.m_shear * chi * self.s_h
            bulk_modulus = shear_modulus * 2. * (1. + poisson) / (3. * (1. - 2. * poisson))
        return bulk_modulus, shear_modulus
    
    @ti.func
    def ComputeElasticStress(self, alpha, dstrain, stress, material_params):
        strain_increment = alpha * dstrain
        dvolumetric_strain = voigt_tensor_trace(strain_increment)
        if ti.abs(dvolumetric_strain) > Threshold:
            bulk_modulus, shear_modulus = self.ComputeNonLinearElasticModulus(dvolumetric_strain, stress, material_params)
            stress += ElasticTensorMultiplyVector(strain_increment, bulk_modulus, shear_modulus)
        else: 
            bulk_modulus, shear_modulus = self.ComputeElasticModulus(stress, material_params)
            stress += ElasticTensorMultiplyVector(strain_increment, bulk_modulus, shear_modulus)
        return stress
    
    @ti.func
    def ComputeStressInvariants(self, stress):
        p = -SphericalTensor(stress)
        q = EquivalentDeviatoricStress(stress)
        lode = ComputeLodeAngle(stress) if ti.static(self.three_invariants) else 0.
        return p, q, lode

    @ti.func
    def ComputeStressInvariants1(self, stress):
        return self.ComputeStressInvariants(stress)

    @ti.func
    def ComputeMTheta(self, lode, mtheta):
        m_theta = mtheta
        if ti.static(self.three_invariants):
            m_theta = mtheta - (mtheta * mtheta) / (3. + mtheta) * ti.cos(1.5 * lode)
        return m_theta

    @ti.func
    def ComputeBondingParameters(self, chi):
        chi_new = clamp(0., 1., chi)
        pcd = 0.
        pcc = 0.
        if ti.static(self.bonding):
            bonded_state = ti.max(chi_new * self.s_h, 0.)
            pcd = self.mc_a * ti.pow(bonded_state, self.mc_b)
            pcc = self.mc_c * ti.pow(bonded_state, self.mc_d)
        return chi_new, pcd, pcc

    @ti.func
    def UpdateBondingParameters(self, chi_n, dpdstrain):
        chi = chi_n
        if ti.static(self.bonding):
            chi = chi_n - self.m_degradation * chi_n * dpdstrain
        return self.ComputeBondingParameters(chi)

    @ti.func
    def ComputeSubloadingParameter(self, pressure, pc, pcd, pcc, subloading_r, dpvstrain, dpdstrain):
        subloading = 1.
        if ti.static(self.subloading):
            pc_safe = ti.max(pc, 100.)
            surface_size = ti.max(pc + pcd + pcc, 100.)
            if ti.abs(subloading_r - 1.) < Threshold:
                subloading = pressure / surface_size
            else:
                r_safe = ti.max(subloading_r, 1.e-12)
                plastic_increment = ti.sqrt(dpvstrain * dpvstrain + dpdstrain * dpdstrain)
                subloading = subloading_r - self.subloading_u * (1. + (pcd + pcc) / pc_safe) * ti.log(r_safe) * plastic_increment
            subloading = clamp(1.e-5, 1., subloading)
        return subloading

    @ti.func
    def ComputeYieldFunctionInvariant(self, pressure, q, lode, pc, pcd, pcc, subloading_r, material_params):
        m_theta = self.ComputeMTheta(lode, material_params[3])
        return q * q / (m_theta * m_theta) + (pressure + pcc) * (pressure - subloading_r * (pc + pcd + pcc))
    
    @ti.func
    def ComputeYieldFunction(self, stress, internal_vars, material_params):
        if ti.static(self.is_rate_dependent):
            pass
        pc = internal_vars[0]
        pcd, pcc, subloading_r = internal_vars[9], internal_vars[10], internal_vars[11]
        pressure, q, lode = self.ComputeStressInvariants(stress)
        return self.ComputeYieldFunctionInvariant(pressure, q, lode, pc, pcd, pcc, subloading_r, material_params)

    @ti.func
    def ComputeYieldState(self, stress, internal_vars, material_params):
        f_function = self.ComputeYieldFunction(stress, internal_vars, material_params)
        return f_function > -FTOL, f_function

    @ti.func
    def ComputeDfDsigma(self, yield_state, stress, internal_vars, material_params):
        m_theta0 = material_params[3]
        pc, pcd, pcc, subloading_r = internal_vars[0], internal_vars[9], internal_vars[10], internal_vars[11]
        pressure, q, lode = self.ComputeStressInvariants(stress)
        m_theta = self.ComputeMTheta(lode, m_theta0)
        dfdp = 2. * pressure + pcc - subloading_r * (pc + pcd + pcc)
        dpdsigma = -DpDsigma()
        dfdq = 2. * q / (m_theta * m_theta)
        dqdsigma = DqDsigma(stress)
        dfdsigma = dfdp * dpdsigma + dfdq * dqdsigma
        if ti.static(self.three_invariants):
            dfdmtheta = -2. * q * q / (m_theta * m_theta * m_theta)
            dmthetadtheta = 1.5 * m_theta0 * m_theta0 / (3. + m_theta0) * ti.sin(1.5 * lode)
            dthetadsigma = DlodeDsigma(stress)
            dfdsigma += dfdmtheta * dmthetadtheta * dthetadsigma
        if ti.static(self.is_rate_dependent):
            pass
        return dfdsigma
    
    @ti.func
    def ComputeDgDsigma(self, yield_state, stress, internal_vars, material_params):
        pc, pcd = internal_vars[0], internal_vars[9]
        pressure, q, lode = self.ComputeStressInvariants(stress)
        m_theta = self.ComputeMTheta(lode, material_params[3])
        dgdp = 2. * pressure - pc - pcd
        dpdsigma = -DpDsigma()
        dgdq = 2. * q / (m_theta * m_theta)
        dqdsigma = DqDsigma(stress)
        dgdsigma = dgdp * dpdsigma + dgdq * dqdsigma
        if ti.static(self.is_rate_dependent):
            pass
        return dgdsigma
    
    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma, stress, internal_vars, state_vars, material_params):
        lambda_, kappa, void_ratio = material_params[1], material_params[2], material_params[4]
        pc, pcd, pcc, subloading_r = internal_vars[0], internal_vars[9], internal_vars[10], internal_vars[11]
        pressure, q, lode = self.ComputeStressInvariants(stress)
        df_dp = 2. * pressure - pc - pcd
        df_dpc = -(pressure + pcc)
        upsilon = (1 + void_ratio) / (lambda_ - kappa)
        hardening = upsilon * pc * df_dp * df_dpc
        if ti.static(self.bonding):
            df_dpcd = -pressure - pcc
            df_dpcc = -2. * pcc - pc - pcd
            df_dq = 2. * q / (self.ComputeMTheta(lode, material_params[3]) ** 2)
            hardening += df_dpcd * (-self.m_degradation * self.mc_b * pcd) * df_dq
            hardening += df_dpcc * (-self.m_degradation * self.mc_d * pcc) * df_dq
        if ti.static(self.subloading):
            df_dr = -(pressure + pcc) * (pc + pcd + pcc)
            plastic_increment = ti.sqrt(state_vars.dpvstrain * state_vars.dpvstrain + state_vars.dpdstrain * state_vars.dpdstrain)
            hardening += -df_dr * self.subloading_u * (1. + (pcd + pcc) / ti.max(pc, 100.)) * ti.log(ti.max(subloading_r, 1.e-12)) * plastic_increment
        return hardening

    @ti.func
    def ComputeInternalVariables(self, dlambda, dgdsigma, internal_vars, material_params):
        plastic_strain = dlambda * dgdsigma
        dpvstrain = -ComputeStrainInvariantI1(plastic_strain)
        dpdstrain = self.ComputeEquivalentPlasticStrainIncrement(dlambda, dgdsigma)
        lambda_, kappa, void_ratio = material_params[1], material_params[2], material_params[4]
        pc, pcd = internal_vars[0], internal_vars[9]
        dpc = dpvstrain * (pc + pcd) * (1. + void_ratio) / (lambda_ - kappa)
        chi, pcd_new, pcc_new = self.UpdateBondingParameters(internal_vars[8], dpdstrain)
        subloading_r = self.ComputeSubloadingParameter(100., pc + dpc, pcd_new, pcc_new, internal_vars[11], dpvstrain, dpdstrain)
        return ti.Vector([dpc, dpdstrain, 0., dlambda, dpvstrain, dpdstrain, dpvstrain, dpdstrain, 
                          chi - internal_vars[8], pcd_new - internal_vars[9], pcc_new - internal_vars[10], subloading_r - internal_vars[11]])

    @ti.func
    def GetBondingState(self, state_vars):
        chi = 1.
        pcd = 0.
        pcc = 0.
        if ti.static(self.bonding):
            chi = state_vars.chi
            pcd = state_vars.pcd
            pcc = state_vars.pcc
        return chi, pcd, pcc

    @ti.func
    def GetSubloadingState(self, state_vars):
        subloading_r = 1.
        if ti.static(self.subloading):
            subloading_r = state_vars.subloading_r
        return subloading_r

    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        poisson, lambda_, kappa, m_theta0 = self.get_current_material_parameter(state_vars)
        void_ratio = state_vars.void_ratio
        if void_ratio <= 0.:
            pc = ti.max(state_vars.pc, 100)
            pressure = ti.max(-SphericalTensor(stress), 100)
            void_ratio = self.e_ref - lambda_ * ti.log(pc / self.p_ref) + kappa * ti.log(pc / pressure)
        chi, pcd, pcc = self.GetBondingState(state_vars)
        subloading_r = self.GetSubloadingState(state_vars)
        return ti.Vector([poisson, lambda_, kappa, m_theta0, void_ratio, chi, pcd, pcc, subloading_r])

    @ti.func
    def UpdateMaterialParameter(self, stress, internal_vars, state_vars, material_params):
        poisson, lambda_, kappa, m_theta0 = self.get_current_material_parameter(state_vars)
        return ti.Vector([poisson, lambda_, kappa, m_theta0, internal_vars[2], internal_vars[8], internal_vars[9], internal_vars[10], internal_vars[11]])
    
    @ti.func
    def GetInternalVariables(self, state_vars):
        chi, pcd, pcc = self.GetBondingState(state_vars)
        subloading_r = self.GetSubloadingState(state_vars)
        return ti.Vector([state_vars.pc, state_vars.epdstrain, state_vars.void_ratio, state_vars.delta_phi,
                          state_vars.dpvstrain, state_vars.dpdstrain, state_vars.pvstrain, state_vars.pdstrain,
                          chi, pcd, pcc, subloading_r])
    
    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        stateVars[np].pc = ti.max(100, internal_vars[0])
        stateVars[np].epdstrain = ti.max(0., internal_vars[1])
        stateVars[np].void_ratio = internal_vars[2]
        stateVars[np].delta_phi = internal_vars[3]
        stateVars[np].dpvstrain = internal_vars[4]
        stateVars[np].dpdstrain = internal_vars[5]
        stateVars[np].pvstrain = internal_vars[6]
        stateVars[np].pdstrain = internal_vars[7]
        if ti.static(self.bonding):
            stateVars[np].chi = clamp(0., 1., internal_vars[8])
            stateVars[np].pcd = ti.max(0., internal_vars[9])
            stateVars[np].pcc = ti.max(0., internal_vars[10])
        if ti.static(self.subloading):
            stateVars[np].subloading_r = clamp(1.e-5, 1., internal_vars[11])

    @ti.func
    def ComputeEquivalentPlasticStrainIncrement(self, dlambda, dgdsigma):
        plastic_strain = dlambda * dgdsigma
        strain_norm = voigt_tensor_dot(plastic_strain, plastic_strain)
        return ti.sqrt(ti.max(2. / 3. * strain_norm, 0.))

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stateVars):
        stiffness_matrix = self.compute_elastic_tensor(np, current_stress, stateVars)
        state_vars = stateVars[np]
        yield_state = int(state_vars.yield_state)
        if yield_state > 0:
            material_params = self.GetMaterialParameter(current_stress, state_vars)
            bulk_modulus, shear_modulus = self.ComputeElasticModulus(current_stress, material_params)
            pressure, q, lode = self.ComputeStressInvariants(current_stress)
            pc = ti.max(state_vars.pc, 100.)
            _, pcd, pcc = self.GetBondingState(state_vars)
            subloading_r = self.GetSubloadingState(state_vars)
            m_theta = self.ComputeMTheta(lode, material_params[3])
            void_ratio = material_params[4]

            df_dp = 2. * pressure - pc - pcd
            df_dq = 2. * q / (m_theta * m_theta)
            df_dpc = -pressure - pcc
            df_dpcd = -pressure - pcc
            df_dpcc = -2. * pcc - pc - pcd
            if ti.static(self.subloading):
                df_dp = 2. * pressure + pcc - subloading_r * (pc + pcc + pcd)
                df_dpc *= subloading_r
                df_dpcd *= subloading_r
                df_dpcc = pressure - subloading_r * (pressure + pc + pcd + 2. * pcc)

            lambda_, kappa = material_params[1], material_params[2]
            upsilon = (1. + void_ratio) / (lambda_ - kappa)
            hardening = upsilon * pc * df_dp * df_dpc
            if ti.static(self.bonding):
                hardening += df_dpcd * (-self.m_degradation * self.mc_b * pcd) * df_dq
                hardening += df_dpcc * (-self.m_degradation * self.mc_d * pcc) * df_dq
            if ti.static(self.subloading):
                df_dr = -(pressure + pcc) * (pc + pcd + pcc)
                plastic_increment = ti.sqrt(state_vars.dpvstrain * state_vars.dpvstrain + state_vars.dpdstrain * state_vars.dpdstrain)
                hardening += -df_dr * self.subloading_u * (1. + (pcd + pcc) / ti.max(pc, 100.)) * ti.log(ti.max(subloading_r, 1.e-12)) * plastic_increment

            a1 = (bulk_modulus * df_dp) * (bulk_modulus * df_dp)
            a2 = -ti.sqrt(6.) * bulk_modulus * df_dp * shear_modulus * df_dq
            a3 = 6. * (shear_modulus * df_dq) * (shear_modulus * df_dq)
            numerator = bulk_modulus * df_dp * df_dp + 3. * shear_modulus * df_dq * df_dq
            denominator = numerator - hardening

            if ti.abs(denominator) > Threshold:
                dev_stress = vec6f(current_stress[0], current_stress[1], current_stress[2], current_stress[3], current_stress[4], current_stress[5])
                for i in ti.static(range(3)):
                    dev_stress[i] += pressure
                xi = q / ti.sqrt(1.5)

                l_l = ti.Matrix.zero(float, 6, 6)
                n_l = ti.Matrix.zero(float, 6, 6)
                l_n = ti.Matrix.zero(float, 6, 6)
                n_n = ti.Matrix.zero(float, 6, 6)
                if xi > Threshold:
                    volumetric_direction = ti.Vector(
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
                    )
                    deviatoric_direction = dev_stress / xi
                    l_l = volumetric_direction.outer_product(
                        volumetric_direction
                    )
                    l_n = volumetric_direction.outer_product(
                        deviatoric_direction
                    )
                    n_l = deviatoric_direction.outer_product(
                        volumetric_direction
                    )
                    n_n = deviatoric_direction.outer_product(
                        deviatoric_direction
                    )
                stiffness_matrix = (a1 * l_l + a2 * (n_l + l_n) + a3 * n_n) / denominator
        return stiffness_matrix

    @ti.func
    def get_current_material_parameter(self, state_vars):
        if ti.static(GlobalVariable.RANDOMFIELD):
            return state_vars.poisson, state_vars.lambda_, state_vars.kappa, state_vars.m_theta
        else:
            return self.poisson, self.lambda_, self.kappa, self.m_theta

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        material_params = self.GetMaterialParameter(current_stress, stateVars[np])
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(current_stress, material_params)
        return ComputeElasticStiffnessTensor(bulk_modulus, shear_modulus)

    @ti.func
    def UpdateStateVariables(self, np, stress, internal_vars, stateVars):
        pressure, q, lode = self.ComputeStressInvariants(stress)
        stateVars[np].p = pressure
        stateVars[np].q = q
        material_params = self.GetMaterialParameter(stress, stateVars[np])
        if ti.static(self.three_invariants):
            stateVars[np].theta = lode
            if ti.static(not GlobalVariable.RANDOMFIELD):
                stateVars[np].m_theta = self.ComputeMTheta(lode, material_params[3])
        stateVars[np].f_function = self.ComputeYieldFunction(stress, internal_vars, material_params)
        stateVars[np].bulk_modulus, stateVars[np].shear_modulus = self.ComputeElasticModulus(stress, material_params)

    @ti.func
    def ComputeLocalResidualNorm(self, residual_vector, pressure_trial, q_trial, pc_trial, f_trial, material_params):
        m_theta0 = material_params[3]
        scale_p = ti.max(ti.max(ti.abs(pressure_trial), ti.abs(pc_trial)), 1.)
        scale_q = ti.max(ti.abs(q_trial), 1.)
        scale_f = ti.max(ti.max(ti.abs(f_trial), pc_trial * pc_trial + q_trial * q_trial / ti.max(m_theta0 * m_theta0, Threshold)), 1.)
        residual = ti.abs(residual_vector[0]) / scale_p
        residual = ti.max(residual, ti.abs(residual_vector[1]) / scale_q)
        residual = ti.max(residual, ti.abs(residual_vector[2]))
        residual = ti.max(residual, ti.abs(residual_vector[3]) / scale_f)
        return residual

    @ti.func
    def AssembleStressFromPQ(self, pressure, q, n_trial):
        stress = q * n_trial
        for i in ti.static(range(3)):
            stress[i] -= pressure
        return stress

    @ti.func
    def ComputePlasticIncrements(self, pressure, q, pc, pcd, m_theta, dlambda):
        dpvstrain = dlambda * (2. * pressure - pc - pcd)
        dpdstrain = dlambda * (ti.sqrt(6.) * q / (m_theta * m_theta))
        return dpvstrain, dpdstrain

    @ti.func
    def ComputeUpdatedVoidRatio(self, previous_void_ratio, strain_increment):
        reference_void_ratio = previous_void_ratio
        if ti.static(self.e0 > 0.):
            reference_void_ratio = self.e0
        return previous_void_ratio + voigt_tensor_trace(strain_increment) * (1. + reference_void_ratio)

    @ti.func
    def ComputeDfDmul(self, bulk_modulus, shear_modulus, pressure, q, pc, pcd, pcc, lode, material_params, mul):
        lambda_, kappa, m_theta0, void_ratio = material_params[1], material_params[2], material_params[3], material_params[4]
        m_theta = self.ComputeMTheta(lode, m_theta0)
        df_dp = 2. * pressure - pc - pcd
        df_dq = 2. * q / (m_theta * m_theta)
        df_dpc = -(pressure + pcc)
        upsilon = (1. + void_ratio) / (lambda_ - kappa)
        a_den = 1. + (2. * bulk_modulus + upsilon * (pc + pcd)) * mul
        dpdmul = -bulk_modulus * (2. * pressure - pc - pcd) / a_den
        dpcdmul = upsilon * (pc + pcd) * (2. * pressure - pc - pcd) / a_den
        dqdmul = -q / (mul + m_theta * m_theta / (6. * shear_modulus))
        dfdmul = df_dp * dpdmul + df_dq * dqdmul + df_dpc * dpcdmul
        if ti.static(self.bonding):
            df_dpcd = -pressure - pcc
            df_dpcc = -2. * pcc - pc - pcd
            denominator = 6. * shear_modulus * mul + m_theta * m_theta
            dpcd_dmul = 0.
            dpcc_dmul = 0.
            if pcd > Threshold:
                dpcd_dmul = -ti.sqrt(6.) * self.mc_b * self.m_degradation * q / denominator * pcd
            if pcc > Threshold:
                dpcc_dmul = -ti.sqrt(6.) * self.mc_d * self.m_degradation * q / denominator * pcc
            dpcdmul -= dpcd_dmul
            dfdmul = df_dp * dpdmul + df_dq * dqdmul + df_dpc * dpcdmul + df_dpcd * dpcd_dmul + df_dpcc * dpcc_dmul
        return dfdmul
    
    @ti.func
    def ComputeDgDpc(self, bulk_modulus, pressure_trial, pc, pcd, pc_n, material_params, mul):
        lambda_, kappa, void_ratio = material_params[1], material_params[2], material_params[4]
        upsilon = (1. + void_ratio) / (lambda_ - kappa)
        denominator = 1. + 2. * mul * bulk_modulus
        e_index = upsilon * mul * (2. * pressure_trial - pc - pcd) / denominator
        exp_index = ti.exp(e_index)
        g_function = pc_n * exp_index - pc
        dgdpc = pc_n * exp_index * (-upsilon * mul / denominator) - 1.
        return g_function, dgdpc

    @ti.func
    def UpdateReturnMappingAuxiliaries(self, pressure, q, pc, lode, material_params, chi_n, subloading_n, dlambda):
        m_theta = self.ComputeMTheta(lode, material_params[3])
        _, pcd_old, pcc_old = self.ComputeBondingParameters(chi_n)
        dpvstrain, dpdstrain = self.ComputePlasticIncrements(pressure, q, pc, pcd_old, m_theta, dlambda)
        chi, pcd, pcc = self.UpdateBondingParameters(chi_n, dpdstrain)
        subloading_r = self.ComputeSubloadingParameter(pressure, pc, pcd, pcc, subloading_n, dpvstrain, dpdstrain)
        return chi, pcd, pcc, subloading_r, dpvstrain, dpdstrain

    @ti.func
    def SolveLocalReturnMapping(self, pressure_trial, q_trial, lode_trial, pc_trial, chi_n, subloading_n, bulk_modulus, shear_modulus, material_params, n_trial, f_trial):
        m_theta_trial = self.ComputeMTheta(lode_trial, material_params[3])
        chi, pcd, pcc = self.ComputeBondingParameters(chi_n)
        subloading_r = self.ComputeSubloadingParameter(pressure_trial, pc_trial, pcd, pcc, subloading_n, 0., 0.)

        pressure = ti.max(pressure_trial, 100.)
        q = ti.max(q_trial, 0.)
        pc = ti.max(pc_trial, 100.)
        lode = lode_trial
        m_theta = m_theta_trial
        dlambda = 0.
        dpvstrain = 0.
        dpdstrain = 0.
        converged = 0
        f_function = self.ComputeYieldFunctionInvariant(pressure, q, lode, pc, pcd, pcc, subloading_r, material_params)

        counter_f = 0
        while ti.abs(f_function) > Ftolerance and counter_f < itrstep:
            dfdmul = self.ComputeDfDmul(bulk_modulus, shear_modulus, pressure, q, pc, pcd, pcc, lode, material_params, dlambda)
            if ti.abs(dfdmul) <= Threshold:
                break
            dlambda = ti.max(0., dlambda - f_function / dfdmul)
            
            g_function, dgdpc = self.ComputeDgDpc(bulk_modulus, pressure_trial, pc, pcd, pc_trial, material_params, dlambda)
            counter_g = 0
            while ti.abs(g_function) > Gtolerance and counter_g < substep:
                if ti.abs(dgdpc) <= Threshold:
                    break
                pc = ti.max(100., pc - g_function / dgdpc)
                g_function, dgdpc = self.ComputeDgDpc(bulk_modulus, pressure_trial, pc, pcd, pc_trial, material_params, dlambda)
                counter_g += 1

            pressure = (pressure_trial + bulk_modulus * dlambda * pc) / (1. + 2. * bulk_modulus * dlambda)
            q = q_trial / (1. + 6. * shear_modulus * dlambda / (m_theta * m_theta))
            chi, pcd, pcc, subloading_r, dpvstrain, dpdstrain = self.UpdateReturnMappingAuxiliaries(pressure, q, pc, lode, material_params, chi_n, subloading_n, dlambda)
            if ti.static(self.three_invariants):
                updated_stress = self.AssembleStressFromPQ(pressure, q, n_trial)
                _, _, lode = self.ComputeStressInvariants(updated_stress)
                m_theta = self.ComputeMTheta(lode, material_params[3])
            f_function = self.ComputeYieldFunctionInvariant(pressure, q, lode, pc, pcd, pcc, subloading_r, material_params)
            if ti.abs(f_function) <= Ftolerance:
                converged = 1
                break
            counter_f += 1
        return pressure, q, pc, dlambda, chi, pcd, pcc, subloading_r, dpvstrain, dpdstrain, converged

    @ti.func
    def ComputeALResidual(self, unknown, pressure_trial, q_trial, lode_trial, pc_trial, chi_n, subloading_n, bulk_modulus, shear_modulus, material_params, n_trial, f_trial):
        pressure = ti.max(unknown[0], 100.)
        q = ti.max(unknown[1], 0.)
        pc = ti.max(unknown[2], 100.)
        dlambda = ti.max(unknown[3], 0.)
        lode = lode_trial
        m_theta = self.ComputeMTheta(lode, material_params[3])
        chi, pcd, pcc, subloading_r, dpvstrain, dpdstrain = self.UpdateReturnMappingAuxiliaries(pressure, q, pc, lode, material_params, chi_n, subloading_n, dlambda)
        if ti.static(self.three_invariants):
            updated_stress = self.AssembleStressFromPQ(pressure, q, n_trial)
            _, _, lode = self.ComputeStressInvariants(updated_stress)
            m_theta = self.ComputeMTheta(lode, material_params[3])
            chi, pcd, pcc, subloading_r, dpvstrain, dpdstrain = self.UpdateReturnMappingAuxiliaries(pressure, q, pc, lode, material_params, chi_n, subloading_n, dlambda)
        lambda_, kappa, void_ratio = material_params[1], material_params[2], material_params[4]
        upsilon = (1. + void_ratio) / (lambda_ - kappa)
        residual = ti.Vector.zero(float, 4)
        residual[0] = pressure - pressure_trial - bulk_modulus * dlambda * (-2. * pressure + pc)
        residual[1] = q * (1. + 6. * shear_modulus * dlambda / (m_theta * m_theta)) - q_trial
        residual[2] = ti.log(pc / pc_trial) - upsilon * dlambda * (2. * pressure_trial - pc - pcd) / (1. + 2. * dlambda * bulk_modulus)
        residual[3] = self.ComputeYieldFunctionInvariant(pressure, q, lode, pc, pcd, pcc, subloading_r, material_params)
        return residual

    @ti.func
    def SolveLocalReturnMappingAL(self, pressure_trial, q_trial, lode_trial, pc_trial, chi_n, subloading_n, bulk_modulus, shear_modulus, material_params, n_trial, f_trial):
        unknown = ti.Vector([ti.max(pressure_trial, 100.), ti.max(q_trial, 0.), ti.max(pc_trial, 100.), 0.])
        converged = 0
        for _ in range(itrstep):
            residual_vector = self.ComputeALResidual(unknown, pressure_trial, q_trial, lode_trial, pc_trial, chi_n, subloading_n, bulk_modulus, shear_modulus, material_params, n_trial, f_trial)
            residual = self.ComputeLocalResidualNorm(residual_vector, pressure_trial, q_trial, pc_trial, f_trial, material_params)
            if residual < Ftolerance:
                converged = 1
                break
            jacobian = ti.Matrix.zero(float, 4, 4)
            col = 0
            while col < 4:
                perturbed = unknown
                eps = ti.max(ti.abs(unknown[col]) * 1.e-6, 1.e-4)
                perturbed[col] += eps
                residual_perturbed = self.ComputeALResidual(perturbed, pressure_trial, q_trial, lode_trial, pc_trial, chi_n, subloading_n, bulk_modulus, shear_modulus, material_params, n_trial, f_trial)
                row = 0
                while row < 4:
                    jacobian[row, col] = (residual_perturbed[row] - residual_vector[row]) / eps
                    row += 1
                col += 1
            delta = ti.Vector.zero(float, 4)
            if ti.abs(jacobian.determinant()) > Threshold:
                delta = jacobian.inverse() @ residual_vector
            else:
                break

            alpha = 1.
            accepted = 0
            for __ in range(substep):
                unknown_new = unknown - alpha * delta
                unknown_new[0] = ti.max(100., unknown_new[0])
                unknown_new[1] = ti.max(0., unknown_new[1])
                unknown_new[2] = ti.max(100., unknown_new[2])
                unknown_new[3] = ti.max(0., unknown_new[3])
                residual_new_vector = self.ComputeALResidual(unknown_new, pressure_trial, q_trial, lode_trial, pc_trial, chi_n, subloading_n, bulk_modulus, shear_modulus, material_params, n_trial, f_trial)
                residual_new = self.ComputeLocalResidualNorm(residual_new_vector, pressure_trial, q_trial, pc_trial, f_trial, material_params)
                if residual_new < residual or alpha < 1e-4:
                    unknown = unknown_new
                    accepted = 1
                    break
                alpha *= 0.5
            if accepted == 0:
                break
        pressure, q, pc, dlambda = unknown[0], unknown[1], unknown[2], unknown[3]
        lode = lode_trial
        if ti.static(self.three_invariants):
            updated_stress = self.AssembleStressFromPQ(pressure, q, n_trial)
            _, _, lode = self.ComputeStressInvariants(updated_stress)
        chi, pcd, pcc, subloading_r, dpvstrain, dpdstrain = self.UpdateReturnMappingAuxiliaries(pressure, q, pc, lode, material_params, chi_n, subloading_n, dlambda)
        return pressure, q, pc, dlambda, chi, pcd, pcc, subloading_r, dpvstrain, dpdstrain, converged

    @ti.func
    def ImplicitIntegrationAL(self, np, previous_stress, de, dw, stateVars):
        state_vars = stateVars[np]
        internal_vars = self.GetInternalVariables(state_vars)
        material_params = self.GetMaterialParameter(previous_stress, state_vars)
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(previous_stress, material_params)

        trial_stress = self.ComputeElasticStress(1., de, previous_stress, material_params)
        update_stress = trial_stress
        pressure_trial0, _, _ = self.ComputeStressInvariants(trial_stress)
        _, internal_vars[9], internal_vars[10] = self.ComputeBondingParameters(internal_vars[8])
        internal_vars[11] = self.ComputeSubloadingParameter(pressure_trial0, internal_vars[0], internal_vars[9], internal_vars[10], internal_vars[11], 0., 0.)
        material_params = self.UpdateMaterialParameter(previous_stress, internal_vars, state_vars, material_params)
        yield_state_trial, f_function_trial = self.ComputeYieldState(trial_stress, internal_vars, material_params)
        if ti.static(self.solver_type == 1):
            stateVars[np].yield_state = ti.u8(yield_state_trial)

        if f_function_trial > FTOL:
            pressure_trial, q_trial, lode_trial = self.ComputeStressInvariants1(trial_stress)
            n_trial = ComputeDeviatoricStressTensor(trial_stress)
            pc_trial = ti.max(internal_vars[0], 100.)
            pressure, q, pc, dlambda, chi, pcd, pcc, subloading_r, dpvstrain, dpdstrain, converged = self.SolveLocalReturnMappingAL(
                pressure_trial, q_trial, lode_trial, pc_trial, internal_vars[8], internal_vars[11], bulk_modulus, shear_modulus, material_params, n_trial, f_function_trial)
            update_stress = self.AssembleStressFromPQ(pressure, q, n_trial)
            internal_vars[0] = pc
            internal_vars[2] = self.ComputeUpdatedVoidRatio(internal_vars[2], de)
            internal_vars[3] = dlambda
            internal_vars[4] = dpvstrain
            internal_vars[5] = dpdstrain
            internal_vars[6] += dpvstrain
            internal_vars[7] += dpdstrain
            internal_vars[1] += ti.abs(dpdstrain)
            internal_vars[8] = chi
            internal_vars[9] = pcd
            internal_vars[10] = pcc
            internal_vars[11] = subloading_r

            if converged == 0:
                material_params = self.UpdateMaterialParameter(update_stress, internal_vars, state_vars, material_params)
                yield_state, f_function = self.ComputeYieldState(update_stress, internal_vars, material_params)
                if ti.abs(f_function) > FTOL:
                    update_stress, internal_vars = self.DriftCorrect(yield_state, f_function, update_stress, internal_vars, state_vars, material_params)

        update_stress += self.ComputeSigrotStress(dw, previous_stress)
        self.UpdateInternalVariables(np, internal_vars, stateVars)
        self.UpdateStateVariables(np, update_stress, internal_vars, stateVars)
        return update_stress
    
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
        pressure_trial0, _, _ = self.ComputeStressInvariants(trial_stress)
        _, internal_vars[9], internal_vars[10] = self.ComputeBondingParameters(internal_vars[8])
        internal_vars[11] = self.ComputeSubloadingParameter(pressure_trial0, internal_vars[0], internal_vars[9], internal_vars[10], internal_vars[11], 0., 0.)
        material_params = self.UpdateMaterialParameter(previous_stress, internal_vars, state_vars, material_params)
        yield_state_trial, f_function_trial = self.ComputeYieldState(trial_stress, internal_vars, material_params)
        if ti.static(self.solver_type == 1):
            stateVars[np].yield_state = ti.u8(yield_state_trial)

        if f_function_trial > FTOL:
            pressure_trial, q_trial, lode_trial = self.ComputeStressInvariants1(trial_stress)
            n_trial = ComputeDeviatoricStressTensor(trial_stress)
            pressure_trial = ti.max(pressure_trial, 100.)
            q_trial = ti.max(q_trial, 0.)
            pc_n = ti.max(internal_vars[0], 100.)
            pressure, q, pc, mul, chi, pcd, pcc, subloading_r, dpvstrain, dpdstrain, converged = self.SolveLocalReturnMapping(
                pressure_trial, q_trial, lode_trial, pc_n, internal_vars[8], internal_vars[11], bulk_modulus, shear_modulus, material_params, n_trial, f_function_trial)
            update_stress = self.AssembleStressFromPQ(pressure, q, n_trial)
            internal_vars[0] = pc
            internal_vars[2] = self.ComputeUpdatedVoidRatio(internal_vars[2], de)
            internal_vars[3] = mul
            internal_vars[4] = dpvstrain
            internal_vars[5] = dpdstrain
            internal_vars[6] += dpvstrain
            internal_vars[7] += dpdstrain
            internal_vars[1] += ti.abs(dpdstrain)
            internal_vars[8] = chi
            internal_vars[9] = pcd
            internal_vars[10] = pcc
            internal_vars[11] = subloading_r

            if converged == 0:
                material_params = self.UpdateMaterialParameter(update_stress, internal_vars, state_vars, material_params)
                yield_state, f_function_full = self.ComputeYieldState(update_stress, internal_vars, material_params)
                if ti.abs(f_function_full) > FTOL:
                    update_stress, internal_vars = self.DriftCorrect(yield_state, f_function_full, update_stress, internal_vars, state_vars, material_params)
        update_stress += self.ComputeSigrotStress(dw, previous_stress)
        self.UpdateInternalVariables(np, internal_vars, stateVars)
        self.UpdateStateVariables(np, update_stress, internal_vars, stateVars)
        return update_stress

    @ti.func
    def energy_val(self):
        return 0.

    @ti.func
    def energy_grad(self):
        return 0.

    @ti.func
    def energy_hess(self):
        return 0.
