import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.utils.constants import PI, FTOL
from src.utils.ObjectIO import DictIO
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class DruckerPragerModel(PlasticMaterial):
    def __init__(self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type, stress_integration)
        self.c_peak = 0.
        self.fai_peak = 0.
        self.psi_peak = 0.
        self.q_fai = 0.
        self.k_fai = 0.
        self.q_psi = 0.
        self.tensile = 0.
        self.yield_surface_type = 0

    def model_initialize(self, material):
        self.material = material
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        c_peak = DictIO.GetAlternative(material, 'Cohesion', 0.)
        fai_peak = DictIO.GetEssential(material, 'Friction', 'StaticFriction')
        psi_peak = DictIO.GetAlternative(material, 'Dilation', 0.)
        c_residual = DictIO.GetAlternative(material, 'ResidualCohesion', c_peak)
        fai_residual = DictIO.GetAlternative(material, 'ResidualFriction', fai_peak)
        psi_residual = DictIO.GetAlternative(material, 'ResidualDilation', psi_peak)
        pdstrain_peak = DictIO.GetAlternative(material, 'PlasticDevStrain', 0.) 
        pdstrain_residual = DictIO.GetAlternative(material, 'ResidualPlasticDevStrain', 0.)
        tensile = DictIO.GetAlternative(material, 'Tensile', 1e22)
        dpType = DictIO.GetAlternative(material, 'dpType', "MiddleCircumscribed")
        self.validate_elastic_parameters(density, young, poisson)
        self.validate_nonnegative_parameters(
            Cohesion=c_peak,
            ResidualCohesion=c_residual,
            Tensile=tensile,
            PlasticDevStrain=pdstrain_peak,
            ResidualPlasticDevStrain=pdstrain_residual,
        )
        self.validate_friction_angle("Friction", fai_peak)
        self.validate_friction_angle(
            "ResidualFriction", fai_residual
        )
        self.validate_friction_angle("Dilation", psi_peak)
        self.validate_friction_angle(
            "ResidualDilation", psi_residual
        )
        self.choose_soft_function(material)
        self.set_rate_dependent_model(material)
        self.add_material(density, young, poisson, c_peak, fai_peak * PI / 180., psi_peak * PI / 180., c_residual, fai_residual * PI / 180., psi_residual * PI / 180., pdstrain_peak, pdstrain_residual, tensile, dpType)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson, c_peak, fai_peak, psi_peak, c_residual, fai_residual, psi_residual, pdstrain_peak, pdstrain_residual, tensile, dpType="MiddleCircumscribed"):
        fai_peak = max(1e-6, fai_peak)
        self.density = density
        self.young = young
        self.poisson = poisson
        self.c_peak = c_peak
        self.fai_peak = fai_peak
        self.psi_peak = psi_peak
        self.c_residual = c_residual
        self.fai_residual = fai_residual
        self.psi_residual = psi_residual
        self.pdstrain_peak = pdstrain_peak
        self.pdstrain_residual = pdstrain_residual

        self.shear, self.bulk = self.calculate_lame_parameter(self.young, self.poisson) 
        yield_surface_type = {"Circumscribed": 0, "MiddleCircumscribed": 1, "Inscribed": 2}
        self.yield_surface_type = DictIO.GetAlternative(yield_surface_type, dpType, 2)
        self.q_fai, self.k_fai, self.q_psi, self.tensile = self.choose_yield_surface_type(self.c_peak, self.fai_peak, self.psi_peak, tensile)
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def choose_yield_surface_type(self, c, fai, psi, tensile):
        if self.yield_surface_type == 0:
            q_fai = 6. * np.sin(fai) / (np.sqrt(3) * (3 - np.sin(fai)))
            k_fai = 6. * np.cos(fai) * c / (np.sqrt(3) * (3 - np.sin(fai)))
            q_psi = 6. * np.sin(psi) / (np.sqrt(3) * (3 - np.sin(psi)))
        elif self.yield_surface_type == 1:
            q_fai = 6. * np.sin(fai) / (np.sqrt(3) * (3 + np.sin(fai)))
            k_fai = 6. * np.cos(fai) * c / (np.sqrt(3) * (3 + np.sin(fai)))
            q_psi = 6. * np.sin(psi) / (np.sqrt(3) * (3 + np.sin(psi)))
        elif self.yield_surface_type == 2:
            q_fai = 3. * np.tan(fai) / np.sqrt(9. + 12 * np.tan(fai) ** 2)
            k_fai = 3. * c / np.sqrt(9. + 12 * np.tan(fai) ** 2)
            q_psi = 3. * np.tan(psi) / np.sqrt(9. + 12 * np.tan(psi) ** 2)
        if fai > 0.:
            tensile = np.where(tensile < k_fai / q_fai, tensile, k_fai / q_fai)
        tensile = np.where(tensile > 1e-15, tensile, 1e-15)
        return q_fai, k_fai, q_psi, tensile

    def print_message(self, materialID):
        self.print_console_header()
        print('Constitutive model: Drucker-Prager Model')
        print("Material ID: ", materialID)
        print('Density: ', self.density)
        if GlobalVariable.RANDOMFIELD is False:
            print('Young Modulus: ', self.young)
            print('Poisson Ratio: ', self.poisson)
            if self.is_rate_dependent:
                self.rate_dependent_function.print_message()
            if self.soft_function:
                print('Peak Cohesion Coefficient = ', self.c_peak)
                print('Peak Internal Friction (in radian) = ', self.fai_peak)
                print('Peak Dilatation (in radian) = ', self.psi_peak)
                print('Residual Cohesion Coefficient = ', self.c_residual)
                print('Residual Internal Friction (in radian) = ', self.fai_residual)
                print('Residual Dilatation (in radian) = ', self.psi_residual)
                print('Peak Plastic Deviartoric Strain = ', self.pdstrain_peak)
                print('Residual Plastic Deviartoric Strain = ', self.pdstrain_residual)
            else:
                print('Cohesion Coefficient = ', self.c_peak)
                print('Internal Friction (in radian) = ', self.fai_peak)
                print('Dilatation (in radian) = ', self.psi_peak)
            print('Tensile = ', self.tensile)
        yield_surface_type = {0: "Circumscribed", 1: "MiddleCircumscribed", 2: "Inscribed"}
        print('Yield Surface Type: ', yield_surface_type.get(self.yield_surface_type), '\n')

    def define_state_vars(self):
        state_vars = {'epdstrain': float}
        if self.is_rate_dependent:
            state_vars.update({"strain_rate": float})
        if GlobalVariable.RANDOMFIELD:
            state_vars.update({'density': float, 'shear': float, 'bulk': float, 'fai_peak': float, 'psi_peak': float, 'c_peak': float, 'tensile': float})
        return state_vars

    def random_field_initialize(self, parameter):
        super().random_field_initialize(parameter)
        self.c_residual = DictIO.GetAlternative(parameter, 'ResidualCohesion', 0.)
        self.fai_residual = DictIO.GetAlternative(parameter, 'ResidualFriction', 0.) * PI / 180.
        self.psi_residual = DictIO.GetAlternative(parameter, 'ResidualDilation', 0.) * PI / 180.
        self.pdstrain_peak = DictIO.GetAlternative(parameter, 'PlasticDevStrain', 0.)
        self.pdstrain_residual = DictIO.GetAlternative(parameter, 'ResidualPlasticDevStrain', 0.)
        dpType = DictIO.GetAlternative(parameter, 'dpType', "Inscribed")
        yield_surface_type = {"Circumscribed": 0, "MiddleCircumscribed": 1, "Inscribed": 2}
        self.yield_surface_type = DictIO.GetAlternative(yield_surface_type, dpType, 1)
        self.choose_soft_function(parameter)
        
    def read_random_field(self, start_particle, end_particle, stateVars):
        random_field = np.loadtxt(self.random_field_file, unpack=True, comments='#').transpose()
        if random_field.shape[0] < end_particle - start_particle:
            raise RuntimeError("Shape error for the random field file")
        density = np.ascontiguousarray(random_field[0:, 0])
        young = np.ascontiguousarray(random_field[0:, 1])
        poisson = np.ascontiguousarray(random_field[0:, 2])
        cohesion = np.ascontiguousarray(random_field[0:, 3])
        friction = np.ascontiguousarray(random_field[0:, 4]) * np.pi / 180.
        dilation = np.ascontiguousarray(random_field[0:, 5]) * np.pi / 180.
        tensile = np.ascontiguousarray(random_field[0:, 6])
        shear, bulk = self.calculate_lame_parameter(young, poisson)
        self.kernel_add_random_material(start_particle, end_particle, density, shear, bulk, friction, dilation, cohesion, tensile, stateVars)
        self.max_sound_speed = np.max(self.get_sound_speed(density, young, poisson))

    '''def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        if GlobalVariable.RANDOMFIELD:
            particle_index = np.ascontiguousarray(materialID.to_numpy()[start_index:end_index])
            fai = np.ascontiguousarray(stateVars.fai.to_numpy()[particle_index])
            return 1. - np.sin(fai)
        else:
            fai = self.fai_peak
            return np.repeat(0.5, end_index - start_index)'''

    @ti.kernel
    def kernel_add_random_material(self, start_particle: int, end_particle: int, density: ti.types.ndarray(), shear: ti.types.ndarray(), bulk: ti.types.ndarray(), 
                                   friction: ti.types.ndarray(), dilation: ti.types.ndarray(), cohesion: ti.types.ndarray(), tensile: ti.types.ndarray(), stateVars: ti.template()):
        for np in range(start_particle, end_particle):
            stateVars[np].density = density[np - start_particle]
            stateVars[np].shear = shear[np - start_particle]
            stateVars[np].bulk = bulk[np - start_particle]
            stateVars[np].fai_peak = friction[np - start_particle]
            stateVars[np].psi_peak = dilation[np - start_particle]
            stateVars[np].c_peak = cohesion[np - start_particle]
            stateVars[np].tensile = tensile[np - start_particle]

    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        if ti.static(self.is_rate_dependent):
            stateVars[np].strain_rate = 0.
        stateVars[np].epdstrain = 0.
    
    # ==================================================== Drucker-Parger Model ==================================================== #
    @ti.func
    def ComputeStressInvariant(self, stress):
        return SphericalTensor(stress), ti.sqrt(ti.max(ComputeStressInvariantJ2(stress), 0.))
    
    @ti.func
    def ComputeTensileFunction(self, sigma, tensile):
        return sigma - tensile
    
    @ti.func
    def ComputeShearFunction(self, sigma, J2sqrt, q_fai, k_fai):
        return J2sqrt + q_fai * sigma - k_fai
    
    @ti.func
    def ComputeYieldFunction(self, stress, internal_vars, material_params):
        q_fai, k_fai, tensile = material_params[2], material_params[3], material_params[5]
        if ti.static(self.is_rate_dependent):
            c_peak, strain_rate = material_params[7], material_params[9]
            pressure = -SphericalTensor(stress) + c_peak
            fai_peak = self.rate_dependent_function.GetMuI(pressure, strain_rate)
            q_fai, k_fai = self.get_q_fai(fai_peak), self.get_k_fai(c_peak, fai_peak)
        sigma, J2sqrt = self.ComputeStressInvariant(stress)
        yield_shear = self.ComputeShearFunction(sigma, J2sqrt, q_fai, k_fai)
        yield_tensile = self.ComputeTensileFunction(sigma, tensile)
        return yield_shear, yield_tensile

    @ti.func
    def ComputeYieldState(self, stress, internal_vars, material_params):
        q_fai, k_fai, tensile = material_params[2], material_params[3], material_params[5]
        
        if ti.static(self.is_rate_dependent):
            c_peak, strain_rate = material_params[7], material_params[9]
            pressure = -SphericalTensor(stress) + c_peak
            fai_peak = self.rate_dependent_function.GetMuI(pressure, strain_rate)
            q_fai, k_fai = self.get_q_fai(fai_peak), self.get_k_fai(c_peak, fai_peak)
        
        yield_shear, yield_tensile = self.ComputeYieldFunction(stress, internal_vars, material_params)
        yield_state = 0
        if yield_tensile > -1e-1 and yield_shear > -1e-1:
            _, J2sqrt = self.ComputeStressInvariant(stress)
            alphap = ti.sqrt(1 + q_fai ** 2) - q_fai
            J2sqrtp = k_fai - q_fai * tensile
            dp_hfai = J2sqrt - J2sqrtp - alphap * yield_tensile
            if dp_hfai > Threshold:
                yield_state = 1
            else:
                yield_state = 2
        if yield_tensile < -1e-1 and yield_shear > -1e-1:
            yield_state = 1
        if yield_tensile > -1e-1 and yield_shear < -1e-1:
            yield_state = 2

        f_function = 0.
        if yield_state == 1:
            f_function = yield_shear
        elif yield_state == 2:
            f_function = yield_tensile
        return yield_state, f_function
    
    @ti.func
    def ComputeDfDsigma(self, yield_state, stress, internal_vars, material_params):
        q_fai = material_params[2]
        pressure, strain_rate = 0., 0.
        if ti.static(self.is_rate_dependent):
            pressure = -SphericalTensor(stress) + material_params[7]
            strain_rate = material_params[9]

        if ti.static(self.is_rate_dependent):
            fai_peak = self.rate_dependent_function.GetMuI(pressure, strain_rate)
            q_fai = self.get_q_fai(fai_peak)

        df_dp, df_dq = 0., 0.
        if yield_state == 2:
            df_dp = 1.
            df_dq = 0.
        else:
            df_dp = q_fai
            df_dq = ti.sqrt(3.) / 3.
        
        dp_dsigma = DpDsigma() 
        dq_dsigma = DqDsigma(stress) 
        df_dsigma = df_dp * dp_dsigma + df_dq * dq_dsigma 
        if ti.static(self.is_rate_dependent):
            df_dsigma += self.ComputeDfDmuI(stress, internal_vars, material_params) * self.rate_dependent_function.DMuIDsigma(pressure, strain_rate, dp_dsigma)
        return df_dsigma
    
    @ti.func
    def ComputeDgDsigma(self, yield_state, stress, internal_vars, material_params):
        dg_dp, dg_dq = 0., 0.
        if yield_state == 2:
            dg_dp = 1.
            dg_dq = 0.
        else:
            dg_dp = material_params[4]
            dg_dq = ti.sqrt(3.) / 3.
        
        dp_dsigma = DpDsigma() 
        dq_dsigma = DqDsigma(stress) 
        dg_dsigma = dg_dp * dp_dsigma + dg_dq * dq_dsigma 
        return dg_dsigma
    
    @ti.func
    def ComputeDfDmuI(self, stress, internal_vars, material_params):
        return -SphericalTensor(stress)
    
    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma, stress, internal_vars, state_vars, material_params):
        if ti.static(self.is_soft):
            plastic_modulus = 0.
            if yield_state == 1:
                fai, c = material_params[6], material_params[7]
                pdstrain = internal_vars[0]
                sigma = SphericalTensor(stress)

                bulk, shear, fai_peak, psi_peak, c_peak, tensile = self.get_current_material_parameter(state_vars)
                fai_residual, c_residual, pdstrain_peak, pdstrain_residual = self.fai_residual, self.c_residual, self.pdstrain_peak, self.pdstrain_residual
                pdstrain_deriv = pdstrain
                if pdstrain >= pdstrain_peak and pdstrain < pdstrain_residual:
                    pdstrain_deriv = ti.max(pdstrain, pdstrain_peak + Threshold)
                dfai_dpstrain = self.soft_function.soft_deriv(fai_peak, fai_residual, pdstrain_peak, pdstrain_residual, pdstrain_deriv)
                dc_dpstrain = self.soft_function.soft_deriv(c_peak, c_residual, pdstrain_peak, pdstrain_residual, pdstrain_deriv)
                dqfaidfai, dkfaidfai, dkfaidc = self.get_strength_derivative(c, fai)
                df_dqfai = sigma
                df_dqkai = -1
                dfdpdstrain = (df_dqfai * dqfaidfai + df_dqkai * dkfaidfai) * dfai_dpstrain + df_dqkai * dkfaidc * dc_dpstrain
                plastic_modulus = -dfdpdstrain

                if ti.static(self.is_rate_dependent):
                    pass
            return plastic_modulus
        else:
            return 0.
    
    @ti.func
    def ComputeInternalVariables(self, dlambda, dgdsigma, internal_vars, material_params):
        q_psi = material_params[4]
        dpdstrain = dlambda * ti.sqrt(1./3. + (2./9.) * q_psi ** 2)
        return ti.Vector([dpdstrain])
        
    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        bulk, shear, fai_peak, psi_peak, c_peak, tensile = self.get_current_material_parameter(state_vars)
        q_fai, k_fai, q_psi = self.q_fai, self.k_fai, self.q_psi
        if ti.static(self.is_soft):
            pdstrain = state_vars.epdstrain
            fai_residual, psi_residual, c_residual, pdstrain_peak, pdstrain_residual = self.fai_residual, self.psi_residual, self.c_residual, self.pdstrain_peak, self.pdstrain_residual
            if ti.static(GlobalVariable.RANDOMFIELD):
                fai_residual, psi_residual, c_residual = self.residual_frac * fai_peak, self.residual_frac * psi_peak, self.residual_frac * c_peak
            fai_peak = self.soft_function.soft(fai_peak, fai_residual, pdstrain_peak, pdstrain_residual, pdstrain)
            psi_peak = self.soft_function.soft(psi_peak, psi_residual, pdstrain_peak, pdstrain_residual, pdstrain)
            c_peak = self.soft_function.soft(c_peak, c_residual, pdstrain_peak, pdstrain_residual, pdstrain)
            q_fai, k_fai, q_psi, tensile = self.get_strength_parameter(c_peak, fai_peak, psi_peak, tensile)
        else:
            if ti.static(GlobalVariable.RANDOMFIELD):
                q_fai, k_fai, q_psi, tensile = self.get_strength_parameter(c_peak, fai_peak, psi_peak, tensile)
        strain_rate = 0.
        if ti.static(self.is_rate_dependent):
            strain_rate = state_vars.strain_rate
        return ti.Vector([bulk, shear, q_fai, k_fai, q_psi, tensile, fai_peak, c_peak, psi_peak, strain_rate])

    @ti.func
    def GetInternalVariables(self, state_vars):
        return ti.Vector([state_vars.epdstrain])
    
    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        stateVars[np].epdstrain = internal_vars[0]

    @ti.func
    def get_current_material_parameter(self, state_vars):
        if ti.static(GlobalVariable.RANDOMFIELD):
            return state_vars.bulk, state_vars.shear, state_vars.fai_peak, state_vars.psi_peak, state_vars.c_peak, state_vars.tensile
        else:
            return self.bulk, self.shear, self.fai_peak, self.psi_peak, self.c_peak, self.tensile
        
    @ti.func
    def get_strength_parameter0(self, c, fai, psi):
        q_fai, k_fai, q_psi = 0., 0., 0.
        if ti.static(self.yield_surface_type == 0):
            q_fai = 6. * ti.sin(fai) / (ti.sqrt(3) * (3 - ti.sin(fai)))
            k_fai = 6. * ti.cos(fai) * c / (ti.sqrt(3) * (3 - ti.sin(fai)))
            q_psi = 6. * ti.sin(psi) / (ti.sqrt(3) * (3 - ti.sin(psi)))
        elif ti.static(self.yield_surface_type == 1):
            q_fai = 6. * ti.sin(fai) / (ti.sqrt(3) * (3 + ti.sin(fai)))
            k_fai = 6. * ti.cos(fai) * c / (ti.sqrt(3) * (3 + ti.sin(fai)))
            q_psi = 6. * ti.sin(psi) / (ti.sqrt(3) * (3 + ti.sin(psi)))
        elif ti.static(self.yield_surface_type == 2):
            q_fai = 3. * ti.tan(fai) / ti.sqrt(9. + 12 * ti.tan(fai) ** 2)
            k_fai = 3. * c / ti.sqrt(9. + 12 * ti.tan(fai) ** 2)
            q_psi = 3. * ti.tan(psi) / ti.sqrt(9. + 12 * ti.tan(psi) ** 2)
        return q_fai, k_fai, q_psi
        
    @ti.func
    def get_q_fai(self, fai):
        q_fai = 0.
        if ti.static(self.yield_surface_type == 0):
            q_fai = 6. * ti.sin(fai) / (ti.sqrt(3) * (3 - ti.sin(fai)))
        elif ti.static(self.yield_surface_type == 1):
            q_fai = 6. * ti.sin(fai) / (ti.sqrt(3) * (3 + ti.sin(fai)))
        elif ti.static(self.yield_surface_type == 2):
            q_fai = 3. * ti.tan(fai) / ti.sqrt(9. + 12 * ti.tan(fai) ** 2)
        return q_fai
        
    @ti.func
    def get_k_fai(self, c, fai):
        k_fai = 0.
        if ti.static(self.yield_surface_type == 0):
            k_fai = 6. * ti.cos(fai) * c / (ti.sqrt(3) * (3 - ti.sin(fai)))
        elif ti.static(self.yield_surface_type == 1):
            k_fai = 6. * ti.cos(fai) * c / (ti.sqrt(3) * (3 + ti.sin(fai)))
        elif ti.static(self.yield_surface_type == 2):
            k_fai = 3. * c / ti.sqrt(9. + 12 * ti.tan(fai) ** 2)
        return k_fai
        
    @ti.func
    def get_q_psi(self, psi):
        q_psi = 0.
        if ti.static(self.yield_surface_type == 0):
            q_psi = 6. * ti.sin(psi) / (ti.sqrt(3) * (3 - ti.sin(psi)))
        elif ti.static(self.yield_surface_type == 1):
            q_psi = 6. * ti.sin(psi) / (ti.sqrt(3) * (3 + ti.sin(psi)))
        elif ti.static(self.yield_surface_type == 2):
            q_psi = 3. * ti.tan(psi) / ti.sqrt(9. + 12 * ti.tan(psi) ** 2)
        return q_psi

    @ti.func
    def get_strength_parameter(self, c, fai, psi, tensile):
        q_fai, k_fai, q_psi = self.get_strength_parameter0(c, fai, psi)
        if fai > 0.:
            tensile = tensile if tensile < k_fai / q_fai else k_fai / q_fai
        tensile = ti.max(1e-15, tensile)
        return q_fai, k_fai, q_psi, tensile

    @ti.func
    def get_strength_derivative(self, c, fai):
        dqfaidfai, dkfaidfai, dkfaidc = 0., 0., 0.
        sin_fai = ti.sin(fai)
        cos_fai = ti.cos(fai)
        tan_fai = ti.tan(fai)
        
        if ti.static(self.yield_surface_type == 0):
            denom = 3. - sin_fai
            # dq_fai / dphi
            dqfaidfai = 6. / ti.sqrt(3.) * (3. * cos_fai + sin_fai * cos_fai) / (denom * denom)
            # dk_fai / dphi
            dkfaidfai = 6. / ti.sqrt(3.) * c * (-sin_fai / denom + cos_fai * cos_fai / (denom * denom))
            # dk_fai / dc
            dkfaidc = 6. * cos_fai / (ti.sqrt(3.) * denom)
        elif ti.static(self.yield_surface_type == 1):
            denom = 3. + sin_fai
            # dq_fai / dphi
            dqfaidfai = 6. / ti.sqrt(3.) * (3. * cos_fai - sin_fai * cos_fai) / (denom * denom)
            # dk_fai / dphi
            dkfaidfai = 6. / ti.sqrt(3.) * c * (-sin_fai / denom - cos_fai * cos_fai / (denom * denom))
            # dk_fai / dc
            dkfaidc = 6. * cos_fai / (ti.sqrt(3.) * denom)
        elif ti.static(self.yield_surface_type == 2):
            denom = ti.sqrt(9. + 12. * tan_fai * tan_fai)
            cos_fai2 = cos_fai * cos_fai
            # dq_fai = d/dphi [ 3 * tan(phi) / denom ]
            # first compute d tan(phi) / d phi = 1 / cos^2(phi)
            dtan_dphi = 1. / cos_fai2
            # dq_fai/dphi = 3 / denom * dtan_dphi - 3 * tan(phi) / denom^2 * d denom/dphi
            # d denom/dphi = (1/2)(9 + 12 t^2)^(-1/2) * 24 t * dtan_dphi = 12 t dtan_dphi / denom
            ddenom_dphi = 12. * tan_fai * dtan_dphi / denom
            dqfaidfai = 3. / denom * dtan_dphi - 3. * tan_fai / (denom * denom) * ddenom_dphi
            # dk_fai = d/dphi [ 3 c / denom ] = -3 c / denom^2 * d denom/dphi
            dkfaidfai = -3. * c / (denom * denom) * ddenom_dphi
            # dk_fai / dc
            dkfaidc = 3. / denom
        return dqfaidfai, dkfaidfai, dkfaidc
    
    @ti.func
    def GetPlasticStrainRate(self, trial_stress, update_stress):
        pass

    @ti.func
    def ComputeClosedFormShearDenominator(self, trial_stress, internal_vars, state_vars, material_params, bulk_modulus, shear_modulus, q_fai, q_psi):
        denominator = shear_modulus + bulk_modulus * q_fai * q_psi
        if ti.static(self.is_soft):
            dgdsigma = q_psi * DpDsigma() + ti.sqrt(3.) / 3. * DqDsigma(trial_stress)
            denominator -= self.ComputePlasticModulus(1, dgdsigma, trial_stress, internal_vars, state_vars, material_params)
        return denominator

    @ti.func
    def ImplicitIntegration(self, np, previous_stress, de, dw, stateVars):
        state_vars = stateVars[np]
        internal_vars = self.GetInternalVariables(state_vars)
        material_params = self.GetMaterialParameter(previous_stress, state_vars)
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(previous_stress, material_params)
        q_fai, k_fai, q_psi, tensile = material_params[2], material_params[3], material_params[4], material_params[5]

        # !-- trial elastic stresses ----!
        stress = previous_stress
        sigrot = Sigrot(stress, dw)
        stress += sigrot
        dstress = ElasticTensorMultiplyVector(de, bulk_modulus, shear_modulus)
        trial_stress = stress + dstress 

        sigma = SphericalTensor(trial_stress)
        sd = DeviatoricTensor(trial_stress)
        J2sqrt = ti.sqrt(ti.max(ComputeStressInvariantJ2(trial_stress), 0.))
        safe_J2sqrt = ti.max(J2sqrt, Threshold)

        dpFi = J2sqrt + q_fai * sigma - k_fai
        dpsig = sigma - tensile
        dpdstrain = 0.
        updated_stress = trial_stress
        yield_state = 0

        if dpsig < 0.:
            if dpFi > 0.:
                yield_state = 1
                denominator = self.ComputeClosedFormShearDenominator(trial_stress, internal_vars, state_vars, material_params, bulk_modulus, shear_modulus, q_fai, q_psi)
                safe_den = denominator if ti.abs(denominator) > Threshold else Threshold
                dlamd = dpFi / safe_den
                sigma -= bulk_modulus * q_psi * dlamd
                ratio = (k_fai - q_fai * sigma) / safe_J2sqrt

                sd *= ratio
                updated_stress = AssembleStress(sigma, sd)
                dpdstrain += dlamd * ti.sqrt(1./3. + (2./9.) * q_psi ** 2)
        else:
            alphap = ti.sqrt(1 + q_fai ** 2) - q_fai
            J2sqrtp = k_fai - q_fai * tensile
            dp_hfai = J2sqrt - J2sqrtp - alphap * dpsig

            if dp_hfai > 0.:
                yield_state = 1
                denominator = self.ComputeClosedFormShearDenominator(trial_stress, internal_vars, state_vars, material_params, bulk_modulus, shear_modulus, q_fai, q_psi)
                safe_den = denominator if ti.abs(denominator) > Threshold else Threshold
                dlamd = dpFi / safe_den
                sigma -= bulk_modulus * q_psi * dlamd
                ratio = (k_fai - q_fai * sigma) / safe_J2sqrt
                sd *= ratio
                updated_stress = AssembleStress(sigma, sd)
                dpdstrain += dlamd * ti.sqrt(1./3. + (2./9.) * q_psi ** 2)
            else:
                yield_state = 2
                dlamd = (sigma - tensile) / bulk_modulus
                sigma = tensile
                updated_stress = AssembleStress(sigma, sd)
                dpdstrain += dlamd * (1./3.) * ti.sqrt(2)

        internal_vars[0] += dpdstrain
        if ti.static(self.solver_type == 1):
            stateVars[np].yield_state = ti.u8(yield_state)
        self.UpdateInternalVariables(np, internal_vars, stateVars)
        return updated_stress
