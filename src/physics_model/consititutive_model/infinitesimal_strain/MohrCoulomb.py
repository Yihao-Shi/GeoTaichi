import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.utils.constants import PI, DBL_EPSILON
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_dot
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class MohrCoulombModel(PlasticMaterial):
    def __init__(self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type, stress_integration)
        self.c_peak = 0.
        self.fai_peak = 0.
        self.psi_peak = 0.
        self.c_residual = 0.
        self.fai_residual = 0.
        self.psi_residual = 0.
        self.pdstrain_peak = 0.
        self.pdstrain_residual = 0.
        self.tensile = 0.

    def model_initialize(self, material):
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        tensile = DictIO.GetAlternative(material, 'Tensile', 1e22)
        c_peak = DictIO.GetAlternative(material, 'Cohesion', 0.)
        fai_peak = DictIO.GetAlternative(material, 'Friction', 0.)
        psi_peak = DictIO.GetAlternative(material, 'Dilation', 0.)
        c_residual = DictIO.GetAlternative(material, 'ResidualCohesion', c_peak)
        fai_residual = DictIO.GetAlternative(material, 'ResidualFriction', fai_peak)
        psi_residual = DictIO.GetAlternative(material, 'ResidualDilation', psi_peak)
        pdstrain_peak = DictIO.GetAlternative(material, 'PlasticDevStrain', 0.) 
        pdstrain_residual = DictIO.GetAlternative(material, 'ResidualPlasticDevStrain', 0.)
        self.choose_soft_function(material)
        self.add_material(density, young, poisson, c_peak, fai_peak * PI / 180., psi_peak * PI / 180., c_residual, fai_residual * PI / 180., psi_residual * PI / 180., pdstrain_peak, pdstrain_residual, tensile)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson, c_peak, fai_peak, psi_peak, c_residual, fai_residual, psi_residual, pdstrain_peak, pdstrain_residual, tensile):
        self.density = density
        self.young = young
        self.poisson = poisson

        self.shear = 0.5 * self.young / (1. + self.poisson)
        self.bulk = self.young / (3. * (1 - 2. * self.poisson))
        self.c_peak = c_peak
        self.fai_peak = fai_peak
        self.psi_peak = psi_peak
        self.c_residual = c_residual
        self.fai_residual = fai_residual
        self.psi_residual = psi_residual
        self.pdstrain_peak = pdstrain_peak
        self.pdstrain_residual = pdstrain_residual
        if self.fai_peak > 0.:
            tensile = np.where(tensile < self.c_peak / np.tan(fai_peak), tensile, self.c_peak / np.tan(fai_peak))
        self.tensile = np.where(tensile > 1e-15, tensile, 1e-15)
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Mohr-Coulomb Model')
        print("Model ID: ", materialID)
        if GlobalVariable.RANDOMFIELD is False:
            print('Density: ', self.density)
            print('Young Modulus: ', self.young)
            print('Poisson Ratio: ', self.poisson)
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
        print('\n')

    def define_state_vars(self):
        state_vars = {}
        if self.soft_function:
            state_vars.update({'strain': vec6f})
        else:
            state_vars.update({'epdstrain': float})
        if GlobalVariable.RANDOMFIELD:
            state_vars.update({'density': float, 'shear': float, 'bulk': float, 'c_peak': float, 'fai_peak': float, 'psi_peak': float, 'tensile': float})
        return state_vars

    '''def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        if GlobalVariable.RANDOMFIELD:
            particle_index = np.ascontiguousarray(materialID.to_numpy()[start_index:end_index])
            fai = np.ascontiguousarray(stateVars.fai.to_numpy()[particle_index])
            return 1. - np.sin(fai)
        else:
            fai = self.fai_peak
            return np.repeat(1. - np.sin(fai), end_index - start_index)'''

    def random_field_initialize(self, parameter):
        super().random_field_initialize(parameter)
        self.c_residual = DictIO.GetAlternative(parameter, 'ResidualCohesion', 0.)
        self.fai_residual = DictIO.GetAlternative(parameter, 'ResidualFriction', 0.) * PI / 180.
        self.psi_residual = DictIO.GetAlternative(parameter, 'ResidualDilation', 0.) * PI / 180.
        self.pdstrain_peak = DictIO.GetAlternative(parameter, 'PlasticDevStrain', 0.)
        self.pdstrain_residual = DictIO.GetAlternative(parameter, 'ResidualPlasticDevStrain', 0.)
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
        self.kernel_add_random_material(start_particle, end_particle, density, shear, bulk, cohesion, friction, dilation, tensile, stateVars)
        self.max_sound_speed = np.max(self.get_sound_speed(density, young, poisson))

    @ti.kernel
    def kernel_add_random_material(self, start_particle: int, end_particle: int, density: ti.types.ndarray(), shear: ti.types.ndarray(), bulk: ti.types.ndarray(), cohesion: ti.types.ndarray(), friction: ti.types.ndarray(), dilation: ti.types.ndarray(), tensile: ti.types.ndarray(), stateVars: ti.template()):
        for np in range(start_particle, end_particle):
            stateVars[np].density = density[np - start_particle]
            stateVars[np].shear = shear[np - start_particle]
            stateVars[np].bulk = bulk[np - start_particle]
            stateVars[np].c_peak = cohesion[np - start_particle]
            stateVars[np].fai_peak = friction[np - start_particle]
            stateVars[np].psi_peak = dilation[np - start_particle]
            stateVars[np].tensile = tensile[np - start_particle]

    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        if ti.static(self.is_soft):
            stateVars[np].strain = vec6f(0, 0, 0, 0, 0, 0)
        else:
            stateVars[np].epdstrain = 0.

    # ==================================================== Mohr-Coulomb Model ==================================================== #
    @ti.func
    def ComputeStressInvariant(self, stress):
        return ti.sqrt(3) * SphericalTensor(stress), \
               ti.sqrt(2 * ComputeStressInvariantJ2(stress)), \
               ComputeLodeAngle(stress)
    
    @ti.func
    def ComputeShearFunction(self, epsilon, sqrt2J2, lode, fai, cohesion):
        cos_fai, tan_fai = ti.cos(fai), ti.tan(fai)
        yield_shear = ti.sqrt(1.5) * sqrt2J2 * (ti.sin(lode + PI/3.) / (ti.sqrt(3.) * cos_fai) + \
                      ti.cos(lode + PI/3.) * tan_fai / 3.) + epsilon * ti.sqrt(1./3.) * tan_fai - cohesion
        return yield_shear
    
    @ti.func
    def ComputeTensileFunction(self, epsilon, sqrt2J2, lode, tensile):
        cos_lode = ti.cos(lode)
        yield_tensile = ti.sqrt(2./3.) * cos_lode * sqrt2J2 + epsilon * ti.sqrt(1./3.) - tensile
        return yield_tensile
    
    @ti.func
    def ComputeYieldFunction(self, stress, internal_vars, material_params):
        fai, cohesion, tensile = material_params[2], material_params[4], material_params[5]
        epsilon, sqrt2J2, lode = self.ComputeStressInvariant(stress)
        yield_shear = self.ComputeShearFunction(epsilon, sqrt2J2, lode, fai, cohesion)
        yield_tensile = self.ComputeTensileFunction(epsilon, sqrt2J2, lode, tensile)
        return yield_shear, yield_tensile

    @ti.func
    def ComputeYieldState(self, stress, internal_vars, material_params):
        Tolerance = -1e-1
        fai, cohesion, tensile = material_params[2], material_params[4], material_params[5]
        sin_fai = ti.sin(fai)
        epsilon, sqrt2J2, lode = self.ComputeStressInvariant(stress)
        yield_shear, yield_tensile = self.ComputeYieldFunction(stress, internal_vars, material_params)
        yield_state = 0
        '''f_function = 0.
        if yield_shear > 1e-8:
            yield_state = 1
            f_function = yield_shear
        '''
        if yield_tensile > Tolerance and yield_shear > Tolerance:
            n_fai = (1. + sin_fai) / (1. - sin_fai)
            sigma_p = tensile * n_fai - 2. * cohesion * ti.sqrt(n_fai)
            alpha_p = ti.sqrt(1. + n_fai * n_fai) + n_fai
            h = yield_tensile + alpha_p * (ti.sqrt(2./3.) * ti.cos(lode - 4.*PI/3.) * sqrt2J2 + epsilon * ti.sqrt(1./3.) - sigma_p)
            if h > DBL_EPSILON:
                yield_state = 2
            else:
                yield_state = 1
        if yield_tensile < Tolerance and yield_shear > Tolerance:
            yield_state = 1
        if yield_tensile > Tolerance and yield_shear < Tolerance:
            yield_state = 2

        f_function = 0.
        if yield_state == 1:
            f_function = yield_shear
        elif yield_state == 2:
            f_function = yield_tensile
        return yield_state, f_function
    
    @ti.func
    def ComputeDfDsigma(self, yield_state, stress, internal_vars, material_params):
        fai = material_params[2]
        sqrt2J2 = ti.sqrt(2 * ComputeStressInvariantJ2(stress))
        lode = ComputeLodeAngle(stress)

        df_depsilon, df_dsqrt2J2, df_dlode = 0., 0., 0.
        if yield_state == 2:
            sin_lode, cos_lode = ti.sin(lode), ti.cos(lode)
            df_depsilon = ti.sqrt(1./3.)
            df_dsqrt2J2 = ti.sqrt(2./3.) * cos_lode
            df_dlode = -ti.sqrt(2./3.) * sqrt2J2 * sin_lode
        else:
            sin_lode_PI_3, cos_lode_PI_3 = ti.sin(lode + PI/3.), ti.cos(lode + PI/3.)
            cos_fai, tan_fai = ti.cos(fai), ti.tan(fai)
            df_depsilon = tan_fai * ti.sqrt(1./3.)
            df_dsqrt2J2 = ti.sqrt(1.5) * (sin_lode_PI_3 / (ti.sqrt(3.) * cos_fai) + cos_lode_PI_3 * tan_fai / 3.)
            df_dlode = ti.sqrt(1.5) * sqrt2J2 * (cos_lode_PI_3 / (ti.sqrt(3.) * cos_fai) - sin_lode_PI_3 * tan_fai / 3.)
        
        depsilon_dsigma = DpDsigma() * ti.sqrt(3.)
        dsqrt2J2_dsigma = DqDsigma(stress) * ti.sqrt(2./3.)
        dlode_dsigma = DlodeDsigma(stress)
        df_dsigma = df_depsilon * depsilon_dsigma + df_dsqrt2J2 * dsqrt2J2_dsigma + df_dlode * dlode_dsigma
        return df_dsigma
    
    @ti.func
    def ComputeDgDsigma(self, yield_state, stress, internal_vars, material_params):
        fai, psi, cohesion, tensile = material_params[2], material_params[3], material_params[4], material_params[5]
        sqrt2J2 = ti.sqrt(2 * ComputeStressInvariantJ2(stress))
        lode = ComputeLodeAngle(stress)

        xi, xit = 0.1, 0.1
        sin_lode, cos_lode = ti.sin(lode), ti.cos(lode)
        sin_fai, cos_fai, tan_psi = ti.sin(fai), ti.cos(fai), ti.tan(psi)

        depsilon_dsigma = DpDsigma() * ti.sqrt(3.)
        dsqrt2J2_dsigma = DqDsigma(stress) * ti.sqrt(2./3.)
        dlode_dsigma = DlodeDsigma(stress)
        
        dg_dp, dg_dq = 0., 0.
        dg_depsilon, dg_dsqrt2J2, dg_dlode = 0., 0., 0.
        if yield_state == 2:
            et_value = 0.6
            sqpart = 4. * (1 - et_value * et_value) * cos_lode * cos_lode + 5. * et_value * et_value - 4. * et_value
            if sqpart < Threshold: sqpart = 1e-5
            rt_den = 2. * (1 - et_value * et_value) * cos_lode + (2. * et_value - 1) * ti.sqrt(sqpart)
            rt_num = 4. * (1 - et_value * et_value) * cos_lode * cos_lode + (2. * et_value - 1) * (2. * et_value - 1)
            if ti.abs(rt_den) < Threshold: rt_den = 1e-5
            rt = rt_num / (3. * rt_den)
            temp_den = ti.sqrt(xit * xit * tensile * tensile + 1.5 * rt * rt * sqrt2J2 * sqrt2J2)
            if temp_den < Threshold: temp_den = Threshold
            dg_dp = 1.
            dg_dq = ti.sqrt(1.5) * sqrt2J2 * rt * rt / temp_den
            dp_drt = 1.5 * sqrt2J2 * sqrt2J2 * rt / temp_den
            drtden_dlode = -2. * (1 - et_value * et_value) * sin_lode - (2. * et_value - 1) * 4. * (1 - et_value * et_value) * cos_lode * \
                            sin_lode / ti.sqrt(4. * (1 - et_value * et_value) * cos_lode * cos_lode + 5. * et_value * et_value - 4. * et_value)
            drtnum_dlode = -8. * (1 - et_value * et_value) * cos_lode * sin_lode
            drt_dlode = (drtnum_dlode * rt_den - drtden_dlode * rt_num) / (3. * rt_den * rt_den)
            dg_dlode = dp_drt * drt_dlode 
        else:
            r_mc = (3. - sin_fai) / (6. * cos_fai)
            e_val = (3. - sin_fai) / (3. + sin_fai)
            e_val = clamp(0.5 + 1e-10, 1., e_val)
            sqpart = 4. * (1 - e_val * e_val) * cos_lode * cos_lode + 5 * e_val * e_val - 4 * e_val
            if sqpart < Threshold: sqpart = 1e-10
            m = 2. * (1 - e_val * e_val) * cos_lode + (2. * e_val - 1) * ti.sqrt(sqpart)
            if ti.abs(m) < Threshold: m = 1e-10
            l = 4. * (1 - e_val * e_val) * cos_lode * cos_lode + (2. * e_val - 1) * (2. * e_val - 1)
            r_mw = (l / m) * r_mc
            omega = (xi * cohesion * tan_psi) * (xi * cohesion * tan_psi) + (r_mw * ti.sqrt(1.5) * sqrt2J2) * (r_mw * ti.sqrt(1.5) * sqrt2J2)
            if omega < Threshold: omega = 1e-10
            dl_dlode = -8. * (1. - e_val * e_val) * cos_lode * sin_lode
            dm_dlode = -2. * (1. - e_val * e_val) * sin_lode + (0.5 * (2. * e_val - 1.) * dl_dlode) / ti.sqrt(sqpart)
            drmw_dlode = ((m * dl_dlode) - (l * dm_dlode)) / (m * m)
            dg_dp = tan_psi
            dg_dq = sqrt2J2 * r_mw * r_mw / (2. * ti.sqrt(omega)) * ti.sqrt(6.)
            dg_dlode = (3. * sqrt2J2 * sqrt2J2 * r_mw * r_mc * drmw_dlode) / (2. * ti.sqrt(omega))
        
        dg_depsilon = dg_dp / ti.sqrt(3.)
        dg_dsqrt2J2 = dg_dq * ti.sqrt(1.5)
        dg_dsigma = (dg_depsilon * depsilon_dsigma) + (dg_dsqrt2J2 * dsqrt2J2_dsigma) + (dg_dlode * dlode_dsigma)
        return dg_dsigma

    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma, stress, internal_vars, state_vars, material_params):
        if ti.static(self.is_soft):
            plastic_modulus = 0.
            if yield_state == 1:
                fai = material_params[2]
                strain = vec6f(internal_vars[0], internal_vars[1], internal_vars[2], internal_vars[3], internal_vars[4], internal_vars[5])
                pdstrain = EquivalentDeviatoricStrain(strain)
                sqrt2J2 = ti.sqrt(2 * ComputeStressInvariantJ2(stress))
                lode = ComputeLodeAngle(stress)

                sin_fai, cos_fai = ti.sin(fai), ti.cos(fai)
                bulk, shear, fai_peak, psi_peak, c_peak, tensile = self.get_current_material_parameter(state_vars)
                fai_residual, c_residual, pdstrain_peak, pdstrain_residual = self.fai_residual, self.c_residual, self.pdstrain_peak, self.pdstrain_residual
                dfai_dpstrain = self.soft_function.soft_deriv(self.soft_param, fai_peak, fai_residual, pdstrain_peak, pdstrain_residual, pdstrain)
                dc_dpstrain = self.soft_function.soft_deriv(self.soft_param, c_peak, c_residual, pdstrain_peak, pdstrain_residual, pdstrain)
                df_dfai = ti.sqrt(1.5) * sqrt2J2 * (sin_fai * ti.sin(lode + PI / 3.) / (ti.sqrt(3.) * cos_fai * cos_fai) + ti.cos(lode + PI / 3.) / (3. * cos_fai * cos_fai)) + SphericalTensor(stress) / (cos_fai * cos_fai)
                df_dc = -1
                dfdpdstrain = df_dfai * dfai_dpstrain + df_dc * dc_dpstrain
                r_func = voigt_tensor_dot(DeqepsilonqDepsilon(strain), dgdsigma)
                plastic_modulus = dfdpdstrain * r_func
            return plastic_modulus
        else:
            return 0.
    
    @ti.func
    def ComputeInternalVariables(self, dlambda, dgdsigma, internal_vars, material_params):
        if ti.static(self.is_soft):
            return dlambda * dgdsigma
        else:
            dpdstrain = EquivalentDeviatoricStrain(dlambda * dgdsigma)
            return ti.Vector([dpdstrain])
        
    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        bulk, shear, fai_peak, psi_peak, c_peak, tensile = self.get_current_material_parameter(state_vars)
        if ti.static(self.is_soft):
            pdstrain = EquivalentDeviatoricStrain(state_vars.strain)
            fai_residual, psi_residual, c_residual, pdstrain_peak, pdstrain_residual = self.fai_residual, self.psi_residual, self.c_residual, self.pdstrain_peak, self.pdstrain_residual
            fai_peak = self.soft_function.soft(self.soft_param, fai_peak, fai_residual, pdstrain_peak, pdstrain_residual, pdstrain)
            psi_peak = self.soft_function.soft(self.soft_param, psi_peak, psi_residual, pdstrain_peak, pdstrain_residual, pdstrain)
            c_peak = self.soft_function.soft(self.soft_param, c_peak, c_residual, pdstrain_peak, pdstrain_residual, pdstrain)
            apex = c_peak / ti.max(ti.tan(fai_peak), Threshold)
            if tensile > apex: tensile = ti.max(apex, Threshold)
        return ti.Vector([bulk, shear, fai_peak, psi_peak, c_peak, tensile])
    
    @ti.func
    def GetInternalVariables(self, state_vars):
        if ti.static(self.is_soft):
            strain = state_vars.strain
            return ti.Vector([strain[0], strain[1], strain[2], strain[3], strain[4], strain[5]])
        else:
            return ti.Vector([state_vars.epdstrain])
    
    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        if ti.static(self.is_soft):
            stateVars[np].strain = vec6f(internal_vars[0], internal_vars[1], internal_vars[2], internal_vars[3], internal_vars[4], internal_vars[5])
        else:
            stateVars[np].epdstrain = internal_vars[0]

    @ti.func
    def get_current_material_parameter(self, state_vars):
        if ti.static(GlobalVariable.RANDOMFIELD):
            return state_vars.bulk, state_vars.shear, state_vars.fai_peak, state_vars.psi_peak, state_vars.c_peak, state_vars.tensile
        else:
            return self.bulk, self.shear, self.fai_peak, self.psi_peak, self.c_peak, self.tensile
