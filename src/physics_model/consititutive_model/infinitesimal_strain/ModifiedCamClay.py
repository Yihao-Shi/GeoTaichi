import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.SoftenModel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.utils.constants import FTOL, Ftolerance, Gtolerance, itrstep, substep
from src.utils.ObjectIO import DictIO
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
        self.beta = 0.
        self.ocr = 0.

    def model_initialize(self, material):
        density = DictIO.GetAlternative(material, 'Density', 2650)
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        m_theta = DictIO.GetEssential(material, 'StressRatio')
        lambda_ = DictIO.GetEssential(material, 'lambda')
        kappa = DictIO.GetEssential(material, 'kappa')
        ocr = DictIO.GetEssential(material, 'OverConsolidationRatio', 0)
        e_ref = DictIO.GetEssential(material, 'void_ratio_ref')
        p_ref = DictIO.GetAlternative(material, 'pressure_ref', 1000.)
        beta = DictIO.GetAlternative(material, 'beta', 0.)
        three_invariants = DictIO.GetAlternative(material, 'ThreeInvariants', False)
        self.soft_function = ExponentialSoft()
        self.set_rate_dependent_model(material)
        self.add_material(density, poisson, m_theta, kappa, lambda_, e_ref, p_ref, ocr, beta, three_invariants)
        self.add_coupling_material(material)

    def add_material(self, density, poisson, m_theta, kappa, lambda_, e_ref, p_ref, ocr, beta, three_invariants):
        self.density = density
        self.poisson = poisson
        self.m_theta = m_theta
        self.kappa = kappa
        self.lambda_ = lambda_
        self.e_ref = e_ref
        self.p_ref = p_ref
        self.ocr = ocr
        self.beta = beta
        self.three_invariants = three_invariants
        self.max_sound_speed = self.get_sound_speed()
        self.is_soft = True

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model = Modified Cam-Clay Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Poisson Ratio: ', self.poisson)
        print('Critical Stress Ratio = ', self.m_theta)
        print('Compression index = ', self.lambda_)
        print('Swelling index = ', self.kappa)
        if self.beta > 0:
            print('Cohension index = ', self.beta)
        print('Initial void ratio = ', self.e_ref)
        print('Overconsolidation ratio = ', self.ocr, '\n')

    def define_state_vars(self):
        return {'pc': float}
    
    def get_sound_speed(self):
        return 0.

    def choose_soft_function(self, material):
        soft_type = DictIO.GetAlternative(material, "SoftType", None)
        if soft_type == "Exponential":
            self.is_soft = True
            self.soft_function = ExponentialSoft()
        elif soft_type == "Sinh":
            self.is_soft = True
            self.soft_function = SinhSoft()
        else:
            self.is_soft = False

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
        p = -SphericalTensor(stress)
        stateVars[np].pc = self.ocr * p

    # ==================================================== Modified cam-clay Model ==================================================== #
    @ti.func
    def ComputeElasticModulus(self, stress, material_params):
        poisson, kappa, void_ratio = material_params[0], material_params[2], material_params[4]
        p = -SphericalTensor(stress)
        bulk_modulus = ti.max((1 + void_ratio) / kappa * p, 100)
        shear_modulus = 3. * bulk_modulus * (1 - 2 * poisson) / (2 * (1 + poisson))
        return bulk_modulus, shear_modulus
    
    @ti.func
    def ComputeNonLinearElasticModulus(self, dvolumetric_strain, stress, material_params):
        poisson, kappa, void_ratio = material_params[0], material_params[2], material_params[4]
        p = -SphericalTensor(stress)
        bulk_modulus = -p / dvolumetric_strain * (ti.exp(-(1. + void_ratio) * dvolumetric_strain / kappa) - 1.)
        shear_modulus = 3. * bulk_modulus * (1 - 2 * poisson) / (2 * (1 + poisson))
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
        p = SphericalTensor(stress)
        q = EquivalentDeviatoricStress(stress)
        lode = ComputeLodeAngle(stress)
        return p, q, lode
    
    @ti.func
    def ComputeYieldFunction(self, stress, internal_vars, material_params):
        m_theta0, beta = material_params[3], material_params[5]
        pc = internal_vars[0]
        p, q, lode = self.ComputeStressInvariants(stress)
        m_theta = self.ComputeMTheta(lode, m_theta0)
        return m_theta * m_theta * (-p + beta * pc) * (-p - pc) + (1. + 2. * beta) * q * q 

    @ti.func
    def ComputeYieldState(self, stress, internal_vars, material_params):
        f_function = self.ComputeYieldFunction(stress, internal_vars, material_params)
        return f_function > -FTOL, f_function
    
    @ti.func
    def ComputeMTheta(self, lode, mtheta):
        if ti.static(self.three_invariants):
            return mtheta - (mtheta * mtheta) / (3 + mtheta) * ti.cos(1.5 * lode)
        else:
            return mtheta
        
    @ti.func
    def ComputeDfDsigma(self, yield_state, stress, internal_vars, material_params):
        m_theta0, beta = material_params[3], material_params[5]
        pc = internal_vars[0]
        p, q, lode = self.ComputeStressInvariants(stress)
        m_theta = self.ComputeMTheta(lode, m_theta0)
        dfdp = (m_theta * m_theta) * (2 * p + (1. - beta) * pc)
        dpdsigma = DpDsigma()
        dfdq = 2. * (1. + 2. * beta) * q
        dqdsigma = DqDsigma(stress)
        dfdsigma = dfdp * dpdsigma + dfdq * dqdsigma
        if ti.static(self.three_invariants):
            dfdmtheta = 2. * m_theta * (-p + beta * pc) * (-p - pc)
            dmthetadtheta = 1.5 * m_theta0 * m_theta0 / (3. + m_theta0) * ti.sin(1.5 * lode)
            dthetadsigma = DlodeDsigma(stress)
            dfdsigma += dfdmtheta * dmthetadtheta * dthetadsigma
        return dfdsigma
    
    @ti.func
    def ComputeDgDsigma(self, yield_state, stress, internal_vars, material_params):
        m_theta0, beta = material_params[3], material_params[5]
        pc = internal_vars[0]
        p, q, lode = self.ComputeStressInvariants(stress)
        m_theta = self.ComputeMTheta(lode, m_theta0)
        dgdp = (m_theta * m_theta) * (2 * p + pc)
        dpdsigma = DpDsigma()
        dgdq = 2. * q
        dqdsigma = DqDsigma(stress)
        dgdsigma = dgdp * dpdsigma + dgdq * dqdsigma
        return dgdsigma
    
    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma, stress, internal_vars, state_vars, material_params):
        lambda_, kappa, m_theta0, void_ratio, beta = material_params[1], material_params[2], material_params[3], material_params[4], material_params[5]
        pc = internal_vars[0]
        p = SphericalTensor(stress)
        lode = ComputeLodeAngle(stress)
        m_theta = self.ComputeMTheta(lode, m_theta0)
        dfdpc = m_theta * m_theta * ((1. - beta) * p - 2. * beta * pc)
        upsilon = (1 + void_ratio) / (lambda_ - kappa)
        dpcdpvstrain = -pc * upsilon
        dfdpvstrain = dfdpc * dpcdpvstrain
        r_func = voigt_tensor_dot(DeqepsilonvDepsilon(), dgdsigma)
        return dfdpvstrain * r_func
    
    @ti.func
    def ComputeInternalVariables(self, dlambda, dgdsigma, internal_vars, material_params):
        dpvstrain = -ComputeStrainInvariantI1(dlambda * dgdsigma)
        lambda_, kappa, void_ratio = material_params[1], material_params[2], material_params[4]
        return ti.Vector([dpvstrain * internal_vars[0] * (1. + void_ratio) / (lambda_ - kappa)])
    
    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        pc = state_vars.pc
        p = -SphericalTensor(stress)
        poisson, lambda_, kappa, m_theta0, beta = self.get_current_material_parameter(state_vars)
        void_ratio = self.e_ref - lambda_ * ti.log(pc / self.p_ref) + kappa * ti.log(pc / p)
        return ti.Vector([poisson, lambda_, kappa, m_theta0, void_ratio, beta])
    
    @ti.func
    def GetInternalVariables(self, state_vars):
        return ti.Vector([state_vars.pc])
    
    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        stateVars[np].pc = internal_vars[0]

    @ti.func
    def get_current_material_parameter(self, state_vars):
        if ti.static(GlobalVariable.RANDOMFIELD):
            return state_vars.poisson, state_vars.lambda_, state_vars.kappa, state_vars.m_theta, state_vars.beta
        else:
            return self.poisson, self.lambda_, self.kappa, self.m_theta, self.beta
    
    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(current_stress, stateVars[np])
        return ComputeElasticStiffnessTensor(bulk_modulus, shear_modulus)
        
    '''@ti.func
    def ComputeDfDmul(self, bulk_modulus, shear_modulus, p, q, pc, void_ratio, mul):
        dfdp = 2 * p - pc
        dfdq = 2. * q / (self.m_theta * self.m_theta)
        dfdpc = -p
        upsilon = (1 + void_ratio) / (self.lambda_ - self.kappa)
        a_den = 1 + (2 * bulk_modulus + upsilon * pc) * mul
        dpdmul = -bulk_modulus * (2 * p - pc) / a_den
        dpcdmul = upsilon * pc * (2 * p - pc) / a_den
        dqdmul = -q / (mul + self.m_theta * self.m_theta / (6 * shear_modulus))
        dfdmul = (dfdp * dpdmul) + (dfdq * dqdmul) + (dfdpc * dpcdmul)
        return dfdmul
    
    @ti.func
    def ComputeDgDpc(self, bulk_modulus, p, pc, pc_n, void_ratio, mul):
        upsilon = (1 + void_ratio) / (self.lambda_ - self.kappa)
        e_index = upsilon * mul * (2. * p - pc) / (1 + 2. * mul * bulk_modulus)
        g_function = pc_n * ti.exp(e_index) - pc
        dgdpc = pc_n * ti.exp(e_index) * (-upsilon * mul / (1. + 2. * mul * bulk_modulus)) - 1.
        return g_function, dgdpc
    
    @ti.func
    def ImplicitIntegration(self, np, previous_stress, de, dw, stateVars):
        # !--Initialise state variables----!
        state_vars = stateVars[np]
        internal_vars = self.GetInternalVariables(state_vars)
        material_params = self.GetMaterialParameter(previous_stress, state_vars)

        # !---- trial elastic stresses ----!
        trial_stress = self.ComputeElasticStress(1., de, previous_stress, material_params)

        # !---- compute trial stress invariants ----!
        updated_stress = trial_stress
        yield_state_trial, f_function_trial = self.ComputeYieldState(trial_stress, internal_vars, material_params)
        if ti.static(self.solver_type == 1):
            stateVars[np].yield_state = ti.u8(yield_state_trial)

        p_trial, q_trial = self.ComputeStressInvariants(trial_stress)
        n_trial = ComputeDeviatoricStressTensor(trial_stress)
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(trial_stress, material_params)

        # !-- implicit return mapping ----!
        if f_function_trial > -FTOL:
            mul = 0.
            counter_f = counter_g = 0
            p_n, q_n, pc_n = p_trial, q_trial, pc
            while ti.abs(f_function_trial) > Ftolerance and counter_f < itrstep:
                dfdmul = self.ComputeDfDmul(bulk_modulus, shear_modulus, p_trial, q_trial, pc, void_ratio, mul)
                mul -= f_function_trial / dfdmul
                
                g_function, dgdpc = self.ComputeDgDpc(bulk_modulus, p_n, pc, pc_n, void_ratio, mul)
                while ti.abs(g_function) > Gtolerance and counter_g < substep:
                    pc -= g_function / dgdpc
                    g_function, dgdpc = self.ComputeDgDpc(bulk_modulus, p_n, pc, pc_n, void_ratio, mul)
                    counter_g += 1

                p_trial = (p_n + bulk_modulus * mul * pc) / (1. + 2. * bulk_modulus * mul)
                q_trial = q_n / (1. + 6. * shear_modulus * mul / self.m_theta / self.m_theta)

                yield_state_trial, f_function_trial = self.ComputeYieldState(trial_stress, internal_vars, material_params)
                counter_f += 1
            updated_stress = q_trial * n_trial
            for i in range(3): updated_stress[i] -= p_trial
        update_stress += self.ComputeSigrotStress(dw, previous_stress)
        self.UpdateInternalVariables(np, internal_vars, stateVars)
        self.UpdateStateVariables(np, update_stress, internal_vars, stateVars)
        return update_stress'''