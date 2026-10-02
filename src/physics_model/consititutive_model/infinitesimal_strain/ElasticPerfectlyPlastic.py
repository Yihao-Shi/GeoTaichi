import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.utils.constants import FTOL
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_dot
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class ElasticPerfectlyPlasticModel(PlasticMaterial):
    def __init__(self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type, stress_integration)
        self._yield_peak = 0.
        self._yield_residual = 0.

    def model_initialize(self, material):
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        _yield = DictIO.GetEssential(material, 'YieldStress')
        _yield_residual = DictIO.GetAlternative(material, 'ResidualYieldStress', _yield)
        pdstrain_peak = DictIO.GetAlternative(material, 'PlasticDevStrain', 0.) 
        pdstrain_residual = DictIO.GetAlternative(material, 'ResidualPlasticDevStrain', 0.)
        self.choose_soft_function(material)
        self.add_material(density, young, poisson, _yield, _yield_residual, pdstrain_peak, pdstrain_residual)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson, yield_stress, yield_stress_residual, pdstrain_peak, pdstrain_residual):
        self.density = density
        self.young = young
        self.poisson = poisson
        self.shear = 0.5 * self.young / (1. + self.poisson)
        self.bulk = self.young / (3. * (1 - 2. * self.poisson))
        self._yield_peak = yield_stress
        self._yield_residual = yield_stress_residual
        self.pdstrain_peak = pdstrain_peak
        self.pdstrain_residual = pdstrain_residual
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Elastic Perfectly Plastic Model')
        print("Model ID: ", materialID)
        if GlobalVariable.RANDOMFIELD is False:
            print('Density: ', self.density)
            print('Young Modulus: ', self.young)
            print('Poisson Ratio: ', self.poisson)
            if self.soft_function:
                print('Peak Yield Stress = ', self._yield_peak)
                print('Peak Plastic Deviartoric Strain = ', self.pdstrain_peak)
                print('Residual Plastic Deviartoric Strain = ', self.pdstrain_residual)
            else:
                print('Yield Stress: ', self._yield_peak)
        print('\n')

    def define_state_vars(self):
        state_vars = {}
        if self.soft_function:
            state_vars.update({'strain': vec6f})
        else:
            state_vars.update({'epstrain': float})
        if GlobalVariable.RANDOMFIELD:
            state_vars.update({'density': float, 'shear': float, 'bulk': float, '_yield_peak': float})
        return state_vars
    
    def random_field_initialize(self, parameter):
        super().random_field_initialize(parameter)
        self._yield_residual = DictIO.GetEssential(parameter, 'ResidualYieldStress')
        self.choose_soft_function(parameter)
        
    def read_random_field(self, start_particle, end_particle, stateVars):
        random_field = np.loadtxt(self.random_field_file, unpack=True, comments='#').transpose()
        if random_field.shape[0] < end_particle - start_particle:
            raise RuntimeError("Shape error for the random field file")
        density = np.ascontiguousarray(random_field[0:, 0])
        young = np.ascontiguousarray(random_field[0:, 1])
        poisson = np.ascontiguousarray(random_field[0:, 2])
        yield_stress = np.ascontiguousarray(random_field[0:, 3])
        shear, bulk = self.calculate_lame_parameter(young, poisson)
        self.kernel_add_random_material(start_particle, end_particle, density, shear, bulk, yield_stress, stateVars)
        self.max_sound_speed = np.max(self.get_sound_speed(density, young, poisson))

    @ti.kernel
    def kernel_add_random_material(self, start_particle: int, end_particle: int, density: ti.types.ndarray(), shear: ti.types.ndarray(), bulk: ti.types.ndarray(), yield_stress: ti.types.ndarray(), stateVars: ti.template()):
        for np in range(start_particle, end_particle):
            stateVars[np].density = density[np - start_particle]
            stateVars[np].shear = shear[np - start_particle]
            stateVars[np].bulk = bulk[np - start_particle]
            stateVars[np]._yield_peak = yield_stress[np - start_particle]
    
    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        if ti.static(self.is_soft):
            stateVars[np].strain = vec6f(0, 0, 0, 0, 0, 0)
        else:
            stateVars[np].epdstrain = 0.
    
    # ==================================================== Von-Mises Model ==================================================== #
    @ti.func
    def ComputeStressInvariant(self, stress):
        return EquivalentDeviatoricStress(stress)
    
    @ti.func
    def ComputeShearFunction(self, seqv, yield_stress):
        return seqv - yield_stress
    
    @ti.func
    def ComputeYieldFunction(self, stress, internal_vars, material_params):
        seqv = self.ComputeStressInvariant(stress)
        yield_stress = material_params[1]
        yield_shear = self.ComputeShearFunction(seqv, yield_stress)
        return yield_shear

    @ti.func
    def ComputeYieldState(self, stress, internal_vars, material_params):
        yield_shear = self.ComputeYieldFunction(stress, internal_vars, material_params)

        yield_state = 0
        if yield_shear > -FTOL:
            yield_state = 1
        return yield_state, yield_shear
    
    @ti.func
    def ComputeDfDsigma(self, stress, internal_vars, material_params):
        df_dp = 0.
        df_dq = 1.
        
        dp_dsigma = DpDsigma() 
        dq_dsigma = DqDsigma(stress) 
        df_dsigma = df_dp * dp_dsigma + df_dq * dq_dsigma 
        return df_dsigma
    
    @ti.func
    def ComputeDgDsigma(self, stress, internal_vars, material_params):
        dg_dp = 0.
        dg_dq = 1.
        
        dp_dsigma = DpDsigma() 
        dq_dsigma = DqDsigma(stress) 
        dg_dsigma = dg_dp * dp_dsigma + dg_dq * dq_dsigma 
        return dg_dsigma
    
    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma, stress, internal_vars, state_vars, material_params):
        if ti.static(self.is_soft):
            strain = vec6f(internal_vars[0], internal_vars[1], internal_vars[2], internal_vars[3], internal_vars[4], internal_vars[5])
            pdstrain = EquivalentStrain(strain)

            df_dyield = -1
            dfyield_dpstrain = self.soft_function.soft_deriv(self.soft_param, self._yield_peak, self._yield_residual, self.pdstrain_peak, self.pdstrain_residual, pdstrain)
            dfdpdstrain = df_dyield * dfyield_dpstrain
            r_func = voigt_tensor_dot(DeqepsilonqDepsilon(strain), dgdsigma)
            return dfdpdstrain * r_func
        else:
            return 0.
    
    @ti.func
    def ComputeInternalVariables(self, dlambda, dgdsigma, internal_vars, material_params):
        if ti.static(self.is_soft):
            return dlambda * dgdsigma
        else:
            depstrain = EquivalentStrain(dlambda * dgdsigma)
            return ti.Vector([depstrain])
        
    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        bulk, shear, _yield_peak = self.get_current_material_parameter(state_vars)
        if ti.static(self.is_soft):
            pdstrain = EquivalentStrain(state_vars.strain)
            _yield_peak = self.soft_function.soft(self.soft_param, _yield_peak, self._yield_residual, self.pdstrain_peak, self.pdstrain_residual, pdstrain)
        return ti.Vector([bulk, shear, _yield_peak])

    @ti.func
    def GetInternalVariables(self, state_vars):
        if ti.static(self.is_soft):
            strain = state_vars.strain
            return ti.Vector([strain[0], strain[1], strain[2], strain[3], strain[4], strain[5]])
        else:
            return ti.Vector([state_vars.epstrain])
    
    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        if ti.static(self.is_soft):
            stateVars[np].strain = vec6f(internal_vars[0], internal_vars[1], internal_vars[2], internal_vars[3], internal_vars[4], internal_vars[5])
        else:
            stateVars[np].epstrain = internal_vars[0]

    @ti.func
    def get_current_material_parameter(self, state_vars):
        if ti.static(GlobalVariable.RANDOMFIELD):
            return state_vars.bulk, state_vars.shear, state_vars._yield_peak
        else:
            return self.bulk, self.shear, self._yield_peak