from src.utils.errors import SimulationStateError
import taichi as ti
import numpy as np

from src.physics_model.constitutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.constitutive_model.infinitesimal_strain.PlasticMaterial import PlasticMaterial
from src.utils.constants import FTOL
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_dot
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class ElasticPerfectlyPlasticModel(PlasticMaterial):
    def __init__(
        self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"
    ):
        super().__init__(material_type, configuration, solver_type, stress_integration)
        self._yield_peak = 0.0
        self._yield_residual = 0.0

    def model_initialize(self, material):
        self.material = material
        density = DictIO.get_alternative(material, "Density", 2650)
        young = DictIO.get_essential(material, "YoungModulus")
        poisson = DictIO.get_alternative(material, "PoissonRatio", 0.3)
        _yield = DictIO.get_essential(material, "YieldStress")
        _yield_residual = DictIO.get_alternative(material, "ResidualYieldStress", _yield)
        pdstrain_peak = DictIO.get_alternative(material, "PlasticDevStrain", 0.0)
        pdstrain_residual = DictIO.get_alternative(material, "ResidualPlasticDevStrain", 0.0)
        self.validate_elastic_parameters(density, young, poisson)
        self.validate_nonnegative_parameters(
            YieldStress=_yield,
            ResidualYieldStress=_yield_residual,
            PlasticDevStrain=pdstrain_peak,
            ResidualPlasticDevStrain=pdstrain_residual,
        )
        self.choose_soft_function(material)
        self.add_material(density, young, poisson, _yield, _yield_residual, pdstrain_peak, pdstrain_residual)
        self.add_coupling_material(material)

    def add_material(
        self, density, young, poisson, yield_stress, yield_stress_residual, pdstrain_peak, pdstrain_residual
    ):
        self.density = density
        self.young = young
        self.poisson = poisson
        self.shear = 0.5 * self.young / (1.0 + self.poisson)
        self.bulk = self.young / (3.0 * (1 - 2.0 * self.poisson))
        self._yield_peak = yield_stress
        self._yield_residual = yield_stress_residual
        self.pdstrain_peak = pdstrain_peak
        self.pdstrain_residual = pdstrain_residual
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def print_message(self, materialID):
        self.print_console_header()
        print("Constitutive model: Elastic Perfectly Plastic")
        print("Material ID: ", materialID)
        if GlobalVariable.RANDOMFIELD is False:
            print("Density: ", self.density)
            print("Young Modulus: ", self.young)
            print("Poisson Ratio: ", self.poisson)
            if self.soft_function:
                print("Peak Yield Stress = ", self._yield_peak)
                print("Peak Plastic Deviartoric Strain = ", self.pdstrain_peak)
                print("Residual Plastic Deviartoric Strain = ", self.pdstrain_residual)
            else:
                print("Yield Stress: ", self._yield_peak)
        print("\n")

    def define_state_vars(self):
        state_vars = {}
        if self.soft_function:
            state_vars.update({"strain": vec6f})
        else:
            state_vars.update({"epstrain": float})
        if GlobalVariable.RANDOMFIELD:
            state_vars.update({"density": float, "shear": float, "bulk": float, "_yield_peak": float})
        return state_vars

    def random_field_initialize(self, parameter):
        super().random_field_initialize(parameter)
        self._yield_residual = DictIO.get_essential(parameter, "ResidualYieldStress")
        self.pdstrain_peak = DictIO.get_alternative(parameter, "PlasticDevStrain", 0.0)
        self.pdstrain_residual = DictIO.get_alternative(parameter, "ResidualPlasticDevStrain", 0.0)
        self.choose_soft_function(parameter)

    def read_random_field(self, start_particle, end_particle, stateVars):
        random_field = np.loadtxt(self.random_field_file, unpack=True, comments="#").transpose()
        if random_field.shape[0] < end_particle - start_particle:
            raise SimulationStateError("Shape error for the random field file")
        density = np.ascontiguousarray(random_field[0:, 0])
        young = np.ascontiguousarray(random_field[0:, 1])
        poisson = np.ascontiguousarray(random_field[0:, 2])
        yield_stress = np.ascontiguousarray(random_field[0:, 3])
        shear, bulk = self.calculate_lame_parameter(young, poisson)
        self.kernel_add_random_material(start_particle, end_particle, density, shear, bulk, yield_stress, stateVars)
        self.max_sound_speed = np.max(self.get_sound_speed(density, young, poisson))

    @ti.kernel
    def kernel_add_random_material(
        self,
        start_particle: int,
        end_particle: int,
        density: ti.types.ndarray(),
        shear: ti.types.ndarray(),
        bulk: ti.types.ndarray(),
        yield_stress: ti.types.ndarray(),
        state_variables: ti.template(),
    ):
        for np in range(start_particle, end_particle):
            state_variables[np].density = density[np - start_particle]
            state_variables[np].shear = shear[np - start_particle]
            state_variables[np].bulk = bulk[np - start_particle]
            state_variables[np]._yield_peak = yield_stress[np - start_particle]

    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, state_variables):
        if ti.static(self.is_soft):
            state_variables[np].strain = vec6f(0, 0, 0, 0, 0, 0)
        else:
            state_variables[np].epstrain = 0.0

    # ==================================================== Von-Mises Model ==================================================== #
    @ti.func
    def compute_stress_invariant(self, stress):
        return equivalent_deviatoric_stress(stress)

    @ti.func
    def compute_shear_function(self, seqv, yield_stress):
        return seqv - yield_stress

    @ti.func
    def compute_yield_function(self, stress, internal_vars, material_params):
        seqv = self.compute_stress_invariant(stress)
        yield_stress = material_params[2]
        yield_shear = self.compute_shear_function(seqv, yield_stress)
        return yield_shear

    @ti.func
    def compute_yield_state(self, stress, internal_vars, material_params):
        yield_shear = self.compute_yield_function(stress, internal_vars, material_params)

        yield_state = 0
        if yield_shear > -FTOL:
            yield_state = 1
        return yield_state, yield_shear

    @ti.func
    def compute_df_dsigma(self, yield_state, stress, internal_vars, material_params):
        df_dp = 0.0
        df_dq = 1.0

        dp_dsigma = dp_dsigma()
        dq_dsigma = dq_dsigma(stress)
        df_dsigma = df_dp * dp_dsigma + df_dq * dq_dsigma
        return df_dsigma

    @ti.func
    def compute_dg_dsigma(self, yield_state, stress, internal_vars, material_params):
        dg_dp = 0.0
        dg_dq = 1.0

        dp_dsigma = dp_dsigma()
        dq_dsigma = dq_dsigma(stress)
        dg_dsigma = dg_dp * dp_dsigma + dg_dq * dq_dsigma
        return dg_dsigma

    @ti.func
    def compute_plastic_modulus(self, yield_state, dgdsigma, stress, internal_vars, state_vars, material_params):
        if ti.static(self.is_soft):
            strain = vec6f(
                internal_vars[0],
                internal_vars[1],
                internal_vars[2],
                internal_vars[3],
                internal_vars[4],
                internal_vars[5],
            )
            pdstrain = equivalent_deviatoric_strain(strain)

            df_dyield = -1
            bulk, shear, yield_peak = self.get_current_material_parameter(state_vars)
            yield_residual = self._yield_residual
            if ti.static(GlobalVariable.RANDOMFIELD):
                yield_residual = self.residual_frac * yield_peak
            dfyield_dpstrain = self.soft_function.soft_deriv(
                yield_peak, yield_residual, self.pdstrain_peak, self.pdstrain_residual, pdstrain
            )
            dfdpdstrain = df_dyield * dfyield_dpstrain
            r_func = voigt_tensor_dot(deqepsilonq_depsilon(strain), dgdsigma)
            return dfdpdstrain * r_func
        else:
            return 0.0

    @ti.func
    def compute_internal_variables(self, dlambda, dgdsigma, internal_vars, material_params):
        if ti.static(self.is_soft):
            return dlambda * dgdsigma
        else:
            depstrain = equivalent_deviatoric_strain(dlambda * dgdsigma)
            return ti.Vector([depstrain])

    @ti.func
    def get_material_parameter(self, stress, state_vars):
        strain = ZEROVEC6F
        if ti.static(self.is_soft):
            strain = state_vars.strain
        return self._material_parameter_from_history(state_vars, strain)

    @ti.func
    def update_material_parameter(self, stress, internal_vars, state_vars, material_params):
        if ti.static(self.is_soft):
            return self._material_parameter_from_history(state_vars, internal_vars)
        else:
            return material_params

    @ti.func
    def _material_parameter_from_history(self, state_vars, strain):
        bulk, shear, _yield_peak = self.get_current_material_parameter(state_vars)
        if ti.static(self.is_soft):
            pdstrain = equivalent_deviatoric_strain(strain)
            _yield_residual = self._yield_residual
            if ti.static(GlobalVariable.RANDOMFIELD):
                _yield_residual = self.residual_frac * _yield_peak
            _yield_peak = self.soft_function.soft(
                _yield_peak, _yield_residual, self.pdstrain_peak, self.pdstrain_residual, pdstrain
            )
        return ti.Vector([bulk, shear, _yield_peak])

    @ti.func
    def get_internal_variables(self, state_vars):
        if ti.static(self.is_soft):
            strain = state_vars.strain
            return ti.Vector([strain[0], strain[1], strain[2], strain[3], strain[4], strain[5]])
        else:
            return ti.Vector([state_vars.epstrain])

    @ti.func
    def update_internal_variables(self, np, internal_vars, state_variables):
        if ti.static(self.is_soft):
            state_variables[np].strain = vec6f(
                internal_vars[0],
                internal_vars[1],
                internal_vars[2],
                internal_vars[3],
                internal_vars[4],
                internal_vars[5],
            )
        else:
            state_variables[np].epstrain = internal_vars[0]

    @ti.func
    def get_current_material_parameter(self, state_vars):
        if ti.static(GlobalVariable.RANDOMFIELD):
            return state_vars.bulk, state_vars.shear, state_vars._yield_peak
        else:
            return self.bulk, self.shear, self._yield_peak
