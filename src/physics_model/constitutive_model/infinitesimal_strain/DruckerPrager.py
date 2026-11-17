from src.utils.errors import SimulationStateError
import taichi as ti
import numpy as np

from src.physics_model.constitutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.constitutive_model.infinitesimal_strain.PlasticMaterial import PlasticMaterial
from src.utils.constants import PI, FTOL
from src.utils.ObjectIO import DictIO
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class DruckerPragerModel(PlasticMaterial):
    supports_rate_dependent = True

    def __init__(
        self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"
    ):
        super().__init__(material_type, configuration, solver_type, stress_integration)
        self.c_peak = 0.0
        self.fai_peak = 0.0
        self.psi_peak = 0.0
        self.q_fai = 0.0
        self.k_fai = 0.0
        self.q_psi = 0.0
        self.tensile = 0.0
        self.yield_surface_type = 0

    def model_initialize(self, material):
        self.material = material
        density = DictIO.get_alternative(material, "Density", 2650)
        young = DictIO.get_essential(material, "YoungModulus")
        poisson = DictIO.get_alternative(material, "PoissonRatio", 0.3)
        c_peak = DictIO.get_alternative(material, "Cohesion", 0.0)
        fai_peak = DictIO.get_essential(material, "Friction", "StaticFriction")
        psi_peak = DictIO.get_alternative(material, "Dilation", 0.0)
        c_residual = DictIO.get_alternative(material, "ResidualCohesion", c_peak)
        fai_residual = DictIO.get_alternative(material, "ResidualFriction", fai_peak)
        psi_residual = DictIO.get_alternative(material, "ResidualDilation", psi_peak)
        pdstrain_peak = DictIO.get_alternative(material, "PlasticDevStrain", 0.0)
        pdstrain_residual = DictIO.get_alternative(material, "ResidualPlasticDevStrain", 0.0)
        tensile = DictIO.get_alternative(material, "Tensile", 1e22)
        dp_type = DictIO.get_alternative(material, "dpType", "MiddleCircumscribed")
        self.validate_elastic_parameters(density, young, poisson)
        self.validate_nonnegative_parameters(
            Cohesion=c_peak,
            ResidualCohesion=c_residual,
            Tensile=tensile,
            PlasticDevStrain=pdstrain_peak,
            ResidualPlasticDevStrain=pdstrain_residual,
        )
        self.validate_friction_angle("Friction", fai_peak)
        self.validate_friction_angle("ResidualFriction", fai_residual)
        self.validate_friction_angle("Dilation", psi_peak)
        self.validate_friction_angle("ResidualDilation", psi_residual)
        self.choose_soft_function(material)
        self.set_rate_dependent_model(material)
        self.add_material(
            density,
            young,
            poisson,
            c_peak,
            fai_peak * PI / 180.0,
            psi_peak * PI / 180.0,
            c_residual,
            fai_residual * PI / 180.0,
            psi_residual * PI / 180.0,
            pdstrain_peak,
            pdstrain_residual,
            tensile,
            dp_type,
        )
        self.add_coupling_material(material)

    def add_material(
        self,
        density,
        young,
        poisson,
        c_peak,
        fai_peak,
        psi_peak,
        c_residual,
        fai_residual,
        psi_residual,
        pdstrain_peak,
        pdstrain_residual,
        tensile,
        dpType="MiddleCircumscribed",
    ):
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
        self.yield_surface_type = DictIO.get_alternative(yield_surface_type, dpType, 2)
        self.q_fai, self.k_fai, self.q_psi, self.tensile = self.choose_yield_surface_type(
            self.c_peak, self.fai_peak, self.psi_peak, tensile
        )
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def choose_yield_surface_type(self, c, fai, psi, tensile):
        if self.yield_surface_type == 0:
            q_fai = 6.0 * np.sin(fai) / (np.sqrt(3) * (3 - np.sin(fai)))
            k_fai = 6.0 * np.cos(fai) * c / (np.sqrt(3) * (3 - np.sin(fai)))
            q_psi = 6.0 * np.sin(psi) / (np.sqrt(3) * (3 - np.sin(psi)))
        elif self.yield_surface_type == 1:
            q_fai = 6.0 * np.sin(fai) / (np.sqrt(3) * (3 + np.sin(fai)))
            k_fai = 6.0 * np.cos(fai) * c / (np.sqrt(3) * (3 + np.sin(fai)))
            q_psi = 6.0 * np.sin(psi) / (np.sqrt(3) * (3 + np.sin(psi)))
        elif self.yield_surface_type == 2:
            q_fai = 3.0 * np.tan(fai) / np.sqrt(9.0 + 12 * np.tan(fai) ** 2)
            k_fai = 3.0 * c / np.sqrt(9.0 + 12 * np.tan(fai) ** 2)
            q_psi = 3.0 * np.tan(psi) / np.sqrt(9.0 + 12 * np.tan(psi) ** 2)
        if fai > 0.0:
            tensile = np.where(tensile < k_fai / q_fai, tensile, k_fai / q_fai)
        tensile = np.maximum(tensile, 0.0)
        return q_fai, k_fai, q_psi, tensile

    def print_message(self, materialID):
        self.print_console_header()
        print("Constitutive model: Drucker-Prager Model")
        print("Material ID: ", materialID)
        print("Density: ", self.density)
        if GlobalVariable.RANDOMFIELD is False:
            print("Young Modulus: ", self.young)
            print("Poisson Ratio: ", self.poisson)
            if self.is_rate_dependent:
                self.rate_dependent_function.print_message()
            if self.soft_function:
                print("Peak Cohesion Coefficient = ", self.c_peak)
                print("Peak Internal Friction (in radian) = ", self.fai_peak)
                print("Peak Dilatation (in radian) = ", self.psi_peak)
                print("Residual Cohesion Coefficient = ", self.c_residual)
                print("Residual Internal Friction (in radian) = ", self.fai_residual)
                print("Residual Dilatation (in radian) = ", self.psi_residual)
                print("Peak Plastic Deviartoric Strain = ", self.pdstrain_peak)
                print("Residual Plastic Deviartoric Strain = ", self.pdstrain_residual)
            else:
                print("Cohesion Coefficient = ", self.c_peak)
                print("Internal Friction (in radian) = ", self.fai_peak)
                print("Dilatation (in radian) = ", self.psi_peak)
            print("Tensile = ", self.tensile)
        yield_surface_type = {0: "Circumscribed", 1: "MiddleCircumscribed", 2: "Inscribed"}
        print("Yield Surface Type: ", yield_surface_type.get(self.yield_surface_type), "\n")

    def define_state_vars(self):
        state_vars = {"epdstrain": float}
        if self.is_rate_dependent:
            state_vars.update({"strain_rate": float})
        if GlobalVariable.RANDOMFIELD:
            state_vars.update(
                {
                    "density": float,
                    "shear": float,
                    "bulk": float,
                    "fai_peak": float,
                    "psi_peak": float,
                    "c_peak": float,
                    "tensile": float,
                }
            )
        return state_vars

    def random_field_initialize(self, parameter):
        super().random_field_initialize(parameter)
        self.set_rate_dependent_model(parameter)
        self.c_residual = DictIO.get_alternative(parameter, "ResidualCohesion", 0.0)
        self.fai_residual = DictIO.get_alternative(parameter, "ResidualFriction", 0.0) * PI / 180.0
        self.psi_residual = DictIO.get_alternative(parameter, "ResidualDilation", 0.0) * PI / 180.0
        self.pdstrain_peak = DictIO.get_alternative(parameter, "PlasticDevStrain", 0.0)
        self.pdstrain_residual = DictIO.get_alternative(parameter, "ResidualPlasticDevStrain", 0.0)
        dp_type = DictIO.get_alternative(parameter, "dpType", "Inscribed")
        yield_surface_type = {"Circumscribed": 0, "MiddleCircumscribed": 1, "Inscribed": 2}
        self.yield_surface_type = DictIO.get_alternative(yield_surface_type, dp_type, 1)
        self.choose_soft_function(parameter)

    def read_random_field(self, start_particle, end_particle, stateVars):
        random_field = np.loadtxt(self.random_field_file, unpack=True, comments="#").transpose()
        if random_field.shape[0] < end_particle - start_particle:
            raise SimulationStateError("Shape error for the random field file")
        density = np.ascontiguousarray(random_field[0:, 0])
        young = np.ascontiguousarray(random_field[0:, 1])
        poisson = np.ascontiguousarray(random_field[0:, 2])
        cohesion = np.ascontiguousarray(random_field[0:, 3])
        friction = np.ascontiguousarray(random_field[0:, 4]) * np.pi / 180.0
        dilation = np.ascontiguousarray(random_field[0:, 5]) * np.pi / 180.0
        tensile = np.ascontiguousarray(random_field[0:, 6])
        shear, bulk = self.calculate_lame_parameter(young, poisson)
        self.kernel_add_random_material(
            start_particle, end_particle, density, shear, bulk, friction, dilation, cohesion, tensile, stateVars
        )
        self.max_sound_speed = np.max(self.get_sound_speed(density, young, poisson))

    """def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        if GlobalVariable.RANDOMFIELD:
            particle_index = np.ascontiguousarray(materialID.to_numpy()[start_index:end_index])
            fai = np.ascontiguousarray(stateVars.fai.to_numpy()[particle_index])
            return 1. - np.sin(fai)
        else:
            fai = self.fai_peak
            return np.repeat(0.5, end_index - start_index)"""

    @ti.kernel
    def kernel_add_random_material(
        self,
        start_particle: int,
        end_particle: int,
        density: ti.types.ndarray(),
        shear: ti.types.ndarray(),
        bulk: ti.types.ndarray(),
        friction: ti.types.ndarray(),
        dilation: ti.types.ndarray(),
        cohesion: ti.types.ndarray(),
        tensile: ti.types.ndarray(),
        state_variables: ti.template(),
    ):
        for np in range(start_particle, end_particle):
            state_variables[np].density = density[np - start_particle]
            state_variables[np].shear = shear[np - start_particle]
            state_variables[np].bulk = bulk[np - start_particle]
            state_variables[np].fai_peak = friction[np - start_particle]
            state_variables[np].psi_peak = dilation[np - start_particle]
            state_variables[np].c_peak = cohesion[np - start_particle]
            state_variables[np].tensile = tensile[np - start_particle]

    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, state_variables):
        if ti.static(self.is_rate_dependent):
            state_variables[np].strain_rate = 0.0
            state_variables[np].rate_pressure = -spherical_tensor(particle[np].stress)
        state_variables[np].epdstrain = 0.0

    # ==================================================== Drucker-Parger Model ==================================================== #
    @ti.func
    def compute_stress_invariant(self, stress):
        return spherical_tensor(stress), ti.sqrt(ti.max(compute_stress_invariant_j2(stress), 0.0))

    @ti.func
    def compute_tensile_function(self, sigma, tensile):
        return sigma - tensile

    @ti.func
    def compute_shear_function(self, sigma, J2sqrt, q_fai, k_fai):
        return J2sqrt + q_fai * sigma - k_fai

    @ti.func
    def compute_yield_function(self, stress, internal_vars, material_params):
        q_fai, k_fai, tensile = material_params[2], material_params[3], material_params[5]
        sigma, J2sqrt = self.compute_stress_invariant(stress)
        yield_shear = self.compute_shear_function(sigma, J2sqrt, q_fai, k_fai)
        yield_tensile = self.compute_tensile_function(sigma, tensile)
        return yield_shear, yield_tensile

    @ti.func
    def compute_yield_state(self, stress, internal_vars, material_params):
        q_fai, k_fai, tensile = material_params[2], material_params[3], material_params[5]

        yield_shear, yield_tensile = self.compute_yield_function(stress, internal_vars, material_params)
        yield_state = 0
        if yield_tensile > -1e-1 and yield_shear > -1e-1:
            _, J2sqrt = self.compute_stress_invariant(stress)
            alphap = ti.sqrt(1 + q_fai**2) - q_fai
            J2sqrtp = k_fai - q_fai * tensile
            dp_hfai = J2sqrt - J2sqrtp - alphap * yield_tensile
            if dp_hfai > THRESHOLD:
                yield_state = 1
            else:
                yield_state = 2
        if yield_tensile < -1e-1 and yield_shear > -1e-1:
            yield_state = 1
        if yield_tensile > -1e-1 and yield_shear < -1e-1:
            yield_state = 2

        # Elastic/plastic intersection searches need the signed interior distance.
        f_function = ti.max(yield_shear, yield_tensile)
        if yield_state == 1:
            f_function = yield_shear
        elif yield_state == 2:
            f_function = yield_tensile
        return yield_state, f_function

    @ti.func
    def compute_df_dsigma(self, yield_state, stress, internal_vars, material_params):
        q_fai = material_params[2]
        df_dp, df_dq = 0.0, 0.0
        if yield_state == 2:
            df_dp = 1.0
            df_dq = 0.0
        else:
            df_dp = q_fai
            df_dq = ti.sqrt(3.0) / 3.0

        dp_dsigma = dp_dsigma()
        dq_dsigma = dq_dsigma(stress)
        df_dsigma = df_dp * dp_dsigma + df_dq * dq_dsigma
        return df_dsigma

    @ti.func
    def compute_dg_dsigma(self, yield_state, stress, internal_vars, material_params):
        dg_dp, dg_dq = 0.0, 0.0
        if yield_state == 2:
            dg_dp = 1.0
            dg_dq = 0.0
        else:
            dg_dp = material_params[4]
            dg_dq = ti.sqrt(3.0) / 3.0

        dp_dsigma = dp_dsigma()
        dq_dsigma = dq_dsigma(stress)
        dg_dsigma = dg_dp * dp_dsigma + dg_dq * dq_dsigma
        return dg_dsigma

    @ti.func
    def compute_plastic_modulus(self, yield_state, dgdsigma, stress, internal_vars, state_vars, material_params):
        if ti.static(self.is_soft):
            plastic_modulus = 0.0
            if yield_state == 1 or yield_state == 2:
                fai, c = material_params[6], material_params[7]
                pdstrain = internal_vars[0]
                sigma = spherical_tensor(stress)

                bulk, shear, fai_peak, psi_peak, c_peak, tensile = self.get_current_material_parameter(state_vars)
                fai_residual, c_residual, pdstrain_peak, pdstrain_residual = (
                    self.fai_residual,
                    self.c_residual,
                    self.pdstrain_peak,
                    self.pdstrain_residual,
                )
                if ti.static(GlobalVariable.RANDOMFIELD):
                    fai_residual = self.residual_frac * fai_peak
                    c_residual = self.residual_frac * c_peak
                pdstrain_deriv = pdstrain
                if pdstrain >= pdstrain_peak and pdstrain < pdstrain_residual:
                    pdstrain_deriv = ti.max(pdstrain, pdstrain_peak + THRESHOLD)
                dfai_dpstrain = self.soft_function.soft_deriv(
                    fai_peak, fai_residual, pdstrain_peak, pdstrain_residual, pdstrain_deriv
                )
                dc_dpstrain = self.soft_function.soft_deriv(
                    c_peak, c_residual, pdstrain_peak, pdstrain_residual, pdstrain_deriv
                )
                if ti.static(self.is_rate_dependent):
                    static_fai = self.soft_function.soft(
                        fai_peak, fai_residual, pdstrain_peak, pdstrain_residual, pdstrain_deriv
                    )
                    dmu_dp = self.rate_dependent_function.d_mu_i_dpressure(
                        state_vars.rate_pressure + c, state_vars.strain_rate
                    )
                    dfai_dpstrain = ti.cos(fai) ** 2 * (dfai_dpstrain / ti.cos(static_fai) ** 2 + dmu_dp * dc_dpstrain)
                dqfaidfai, dkfaidfai, dkfaidc = self.get_strength_derivative(c, fai)
                df_dqfai = sigma
                df_dqkai = -1
                dfdpdstrain = (
                    df_dqfai * dqfaidfai + df_dqkai * dkfaidfai
                ) * dfai_dpstrain + df_dqkai * dkfaidc * dc_dpstrain
                if yield_state == 2:
                    dfdpdstrain = 0.0
                    apex = c / ti.max(ti.tan(fai), THRESHOLD)
                    if fai > 0.0 and tensile >= apex:
                        dfdpdstrain = -dc_dpstrain / ti.tan(fai) + c * dfai_dpstrain / ti.sin(fai) ** 2
                plastic_modulus = (
                    dfdpdstrain * self.compute_internal_variables(1.0, dgdsigma, internal_vars, material_params)[0]
                )

            return plastic_modulus
        else:
            return 0.0

    @ti.func
    def compute_internal_variables(self, dlambda, dgdsigma, internal_vars, material_params):
        # The actual flow gives both the shear rate and the distinct hydrostatic tensile rate.
        dpdstrain = dlambda * ti.sqrt(2.0 / 3.0 * voigt_tensor_dot(dgdsigma, dgdsigma))
        return ti.Vector([dpdstrain])

    @ti.func
    def get_material_parameter(self, stress, state_vars):
        return self._material_parameter_from_history(state_vars, state_vars.epdstrain)

    @ti.func
    def update_material_parameter(self, stress, internal_vars, state_vars, material_params):
        if ti.static(self.is_soft):
            return self._material_parameter_from_history(state_vars, internal_vars[0])
        else:
            return material_params

    @ti.func
    def _material_parameter_from_history(self, state_vars, pdstrain):
        bulk, shear, fai_peak, psi_peak, c_peak, tensile = self.get_current_material_parameter(state_vars)
        q_fai, k_fai, q_psi = self.q_fai, self.k_fai, self.q_psi
        if ti.static(self.is_soft):
            fai_residual, psi_residual, c_residual, pdstrain_peak, pdstrain_residual = (
                self.fai_residual,
                self.psi_residual,
                self.c_residual,
                self.pdstrain_peak,
                self.pdstrain_residual,
            )
            if ti.static(GlobalVariable.RANDOMFIELD):
                fai_residual, psi_residual, c_residual = (
                    self.residual_frac * fai_peak,
                    self.residual_frac * psi_peak,
                    self.residual_frac * c_peak,
                )
            fai_peak = self.soft_function.soft(fai_peak, fai_residual, pdstrain_peak, pdstrain_residual, pdstrain)
            psi_peak = self.soft_function.soft(psi_peak, psi_residual, pdstrain_peak, pdstrain_residual, pdstrain)
            c_peak = self.soft_function.soft(c_peak, c_residual, pdstrain_peak, pdstrain_residual, pdstrain)
            q_fai, k_fai, q_psi, tensile = self.get_strength_parameter(c_peak, fai_peak, psi_peak, tensile)
        else:
            if ti.static(GlobalVariable.RANDOMFIELD):
                q_fai, k_fai, q_psi, tensile = self.get_strength_parameter(c_peak, fai_peak, psi_peak, tensile)
        strain_rate = 0.0
        if ti.static(self.is_rate_dependent):
            strain_rate = state_vars.strain_rate
            fai_peak = self.rate_dependent_function.get_friction_angle(
                state_vars.rate_pressure + c_peak, strain_rate, fai_peak
            )
            q_fai, k_fai, q_psi, tensile = self.get_strength_parameter(c_peak, fai_peak, psi_peak, tensile)
        return ti.Vector([bulk, shear, q_fai, k_fai, q_psi, tensile, fai_peak, c_peak, psi_peak, strain_rate])

    @ti.func
    def get_internal_variables(self, state_vars):
        return ti.Vector([state_vars.epdstrain])

    @ti.func
    def update_internal_variables(self, np, internal_vars, state_variables):
        state_variables[np].epdstrain = internal_vars[0]

    @ti.func
    def get_current_material_parameter(self, state_vars):
        if ti.static(GlobalVariable.RANDOMFIELD):
            return (
                state_vars.bulk,
                state_vars.shear,
                state_vars.fai_peak,
                state_vars.psi_peak,
                state_vars.c_peak,
                state_vars.tensile,
            )
        else:
            return self.bulk, self.shear, self.fai_peak, self.psi_peak, self.c_peak, self.tensile

    @ti.func
    def get_strength_parameter0(self, c, fai, psi):
        q_fai, k_fai, q_psi = 0.0, 0.0, 0.0
        if ti.static(self.yield_surface_type == 0):
            q_fai = 6.0 * ti.sin(fai) / (ti.sqrt(3) * (3 - ti.sin(fai)))
            k_fai = 6.0 * ti.cos(fai) * c / (ti.sqrt(3) * (3 - ti.sin(fai)))
            q_psi = 6.0 * ti.sin(psi) / (ti.sqrt(3) * (3 - ti.sin(psi)))
        elif ti.static(self.yield_surface_type == 1):
            q_fai = 6.0 * ti.sin(fai) / (ti.sqrt(3) * (3 + ti.sin(fai)))
            k_fai = 6.0 * ti.cos(fai) * c / (ti.sqrt(3) * (3 + ti.sin(fai)))
            q_psi = 6.0 * ti.sin(psi) / (ti.sqrt(3) * (3 + ti.sin(psi)))
        elif ti.static(self.yield_surface_type == 2):
            q_fai = 3.0 * ti.tan(fai) / ti.sqrt(9.0 + 12 * ti.tan(fai) ** 2)
            k_fai = 3.0 * c / ti.sqrt(9.0 + 12 * ti.tan(fai) ** 2)
            q_psi = 3.0 * ti.tan(psi) / ti.sqrt(9.0 + 12 * ti.tan(psi) ** 2)
        return q_fai, k_fai, q_psi

    @ti.func
    def get_q_fai(self, fai):
        q_fai = 0.0
        if ti.static(self.yield_surface_type == 0):
            q_fai = 6.0 * ti.sin(fai) / (ti.sqrt(3) * (3 - ti.sin(fai)))
        elif ti.static(self.yield_surface_type == 1):
            q_fai = 6.0 * ti.sin(fai) / (ti.sqrt(3) * (3 + ti.sin(fai)))
        elif ti.static(self.yield_surface_type == 2):
            q_fai = 3.0 * ti.tan(fai) / ti.sqrt(9.0 + 12 * ti.tan(fai) ** 2)
        return q_fai

    @ti.func
    def get_k_fai(self, c, fai):
        k_fai = 0.0
        if ti.static(self.yield_surface_type == 0):
            k_fai = 6.0 * ti.cos(fai) * c / (ti.sqrt(3) * (3 - ti.sin(fai)))
        elif ti.static(self.yield_surface_type == 1):
            k_fai = 6.0 * ti.cos(fai) * c / (ti.sqrt(3) * (3 + ti.sin(fai)))
        elif ti.static(self.yield_surface_type == 2):
            k_fai = 3.0 * c / ti.sqrt(9.0 + 12 * ti.tan(fai) ** 2)
        return k_fai

    @ti.func
    def get_q_psi(self, psi):
        q_psi = 0.0
        if ti.static(self.yield_surface_type == 0):
            q_psi = 6.0 * ti.sin(psi) / (ti.sqrt(3) * (3 - ti.sin(psi)))
        elif ti.static(self.yield_surface_type == 1):
            q_psi = 6.0 * ti.sin(psi) / (ti.sqrt(3) * (3 + ti.sin(psi)))
        elif ti.static(self.yield_surface_type == 2):
            q_psi = 3.0 * ti.tan(psi) / ti.sqrt(9.0 + 12 * ti.tan(psi) ** 2)
        return q_psi

    @ti.func
    def get_strength_parameter(self, c, fai, psi, tensile):
        q_fai, k_fai, q_psi = self.get_strength_parameter0(c, fai, psi)
        if fai > 0.0:
            apex = k_fai / ti.select(q_fai > 0.0, q_fai, 1.0)
            tensile = ti.min(tensile, apex)
        tensile = ti.max(0.0, tensile)
        return q_fai, k_fai, q_psi, tensile

    @ti.func
    def get_strength_derivative(self, c, fai):
        dqfaidfai, dkfaidfai, dkfaidc = 0.0, 0.0, 0.0
        sin_fai = ti.sin(fai)
        cos_fai = ti.cos(fai)
        tan_fai = ti.tan(fai)

        if ti.static(self.yield_surface_type == 0):
            denom = 3.0 - sin_fai
            # dq_fai / dphi
            dqfaidfai = 6.0 / ti.sqrt(3.0) * 3.0 * cos_fai / (denom * denom)
            # dk_fai / dphi
            dkfaidfai = 6.0 / ti.sqrt(3.0) * c * (-sin_fai / denom + cos_fai * cos_fai / (denom * denom))
            # dk_fai / dc
            dkfaidc = 6.0 * cos_fai / (ti.sqrt(3.0) * denom)
        elif ti.static(self.yield_surface_type == 1):
            denom = 3.0 + sin_fai
            # dq_fai / dphi
            dqfaidfai = 6.0 / ti.sqrt(3.0) * 3.0 * cos_fai / (denom * denom)
            # dk_fai / dphi
            dkfaidfai = 6.0 / ti.sqrt(3.0) * c * (-sin_fai / denom - cos_fai * cos_fai / (denom * denom))
            # dk_fai / dc
            dkfaidc = 6.0 * cos_fai / (ti.sqrt(3.0) * denom)
        elif ti.static(self.yield_surface_type == 2):
            denom = ti.sqrt(9.0 + 12.0 * tan_fai * tan_fai)
            cos_fai2 = cos_fai * cos_fai
            # dq_fai = d/dphi [ 3 * tan(phi) / denom ]
            # first compute d tan(phi) / d phi = 1 / cos^2(phi)
            dtan_dphi = 1.0 / cos_fai2
            # dq_fai/dphi = 3 / denom * dtan_dphi - 3 * tan(phi) / denom^2 * d denom/dphi
            # d denom/dphi = (1/2)(9 + 12 t^2)^(-1/2) * 24 t * dtan_dphi = 12 t dtan_dphi / denom
            ddenom_dphi = 12.0 * tan_fai * dtan_dphi / denom
            dqfaidfai = 3.0 / denom * dtan_dphi - 3.0 * tan_fai / (denom * denom) * ddenom_dphi
            # dk_fai = d/dphi [ 3 c / denom ] = -3 c / denom^2 * d denom/dphi
            dkfaidfai = -3.0 * c / (denom * denom) * ddenom_dphi
            # dk_fai / dc
            dkfaidc = 3.0 / denom
        return dqfaidfai, dkfaidfai, dkfaidc

    @ti.func
    def compute_closed_form_shear_denominator(
        self, trial_stress, internal_vars, state_vars, material_params, bulk_modulus, shear_modulus, q_fai, q_psi
    ):
        denominator = shear_modulus + bulk_modulus * q_fai * q_psi
        if ti.static(self.is_soft):
            dgdsigma = q_psi * dp_dsigma() + ti.sqrt(3.0) / 3.0 * dq_dsigma(trial_stress)
            denominator -= self.compute_plastic_modulus(
                1, dgdsigma, trial_stress, internal_vars, state_vars, material_params
            )
        return denominator

    @ti.func
    def closed_form_integration(self, np, previous_stress, de, dw, state_variables):
        state_vars = state_variables[np]
        internal_vars = self.get_internal_variables(state_vars)
        material_params = self.get_material_parameter(previous_stress, state_vars)
        bulk_modulus, shear_modulus = self.compute_elastic_modulus(previous_stress, material_params)
        q_fai, k_fai, q_psi, tensile = material_params[2], material_params[3], material_params[4], material_params[5]

        # !-- trial elastic stresses ----!
        stress = previous_stress
        sigrot = sigrot(stress, dw)
        stress += sigrot
        dstress = elastic_tensor_multiply_vector(de, bulk_modulus, shear_modulus)
        trial_stress = stress + dstress

        trial_sigma = spherical_tensor(trial_stress)
        sigma = trial_sigma
        sd = deviatoric_tensor(trial_stress)
        J2sqrt = ti.sqrt(ti.max(compute_stress_invariant_j2(trial_stress), 0.0))
        dp_fi = J2sqrt + q_fai * sigma - k_fai
        returned_J2sqrt = J2sqrt
        shear_multiplier = 0.0
        updated_stress = trial_stress
        yield_state = 0

        if dp_fi > 0.0:
            yield_state = 1
            denominator = self.compute_closed_form_shear_denominator(
                trial_stress, internal_vars, state_vars, material_params, bulk_modulus, shear_modulus, q_fai, q_psi
            )
            safe_den = denominator if ti.abs(denominator) > THRESHOLD else THRESHOLD
            shear_multiplier = dp_fi / safe_den
            sigma -= bulk_modulus * q_psi * shear_multiplier
            returned_J2sqrt = ti.max(0.0, k_fai - q_fai * sigma)

        if sigma > tensile:
            # A shear return beyond the cap requires both plastic flow directions.
            sigma = tensile
            cap_J2sqrt = ti.max(0.0, k_fai - q_fai * tensile)
            returned_J2sqrt = ti.min(J2sqrt, cap_J2sqrt)
            shear_multiplier = (J2sqrt - returned_J2sqrt) / shear_modulus
            yield_state = 2
            if shear_multiplier > 0.0 or cap_J2sqrt <= THRESHOLD:
                yield_state = 3

        if yield_state > 0:
            sd *= returned_J2sqrt / ti.max(J2sqrt, THRESHOLD)
            updated_stress = assemble_stress(sigma, sd)
            plastic_volume = (trial_sigma - sigma) / bulk_modulus
            internal_vars[0] += ti.sqrt(shear_multiplier**2 / 3.0 + 2.0 * plastic_volume**2 / 9.0)
        if ti.static(self.solver_type == 1):
            state_variables[np].yield_state = ti.u8(yield_state)
        self.update_internal_variables(np, internal_vars, state_variables)
        return updated_stress

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, state_variables):
        stiffness = ti.Matrix.zero(float, 6, 6)
        if int(state_variables[np].yield_state) == 3:
            _, J2sqrt = self.compute_stress_invariant(current_stress)
            if J2sqrt > THRESHOLD:
                material_params = self.get_material_parameter(current_stress, state_variables[np])
                _, shear_modulus = self.compute_elastic_modulus(current_stress, material_params)
                direction = deviatoric_tensor(current_stress) / J2sqrt
                # shortcut: corner tangent freezes strength; use a return Jacobian for implicit softening convergence.
                stiffness = compute_elastic_stiffness_tensor(0.0, shear_modulus)
                stiffness -= shear_modulus * direction.outer_product(direction)
        else:
            stiffness = PlasticMaterial.compute_stiffness_tensor(self, np, current_stress, state_variables)
        return stiffness
