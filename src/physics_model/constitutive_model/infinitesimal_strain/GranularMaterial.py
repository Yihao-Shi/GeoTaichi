from src.utils.errors import SimulationStateError
import taichi as ti
import numpy as np

from src.physics_model.constitutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.constitutive_model.infinitesimal_strain.InfinitesimalStrainModel import InfinitesimalStrainModel
from src.physics_model.constitutive_model.infinitesimal_strain.RateDependent import *
from src.utils.ObjectIO import DictIO
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class GranularMaterial(InfinitesimalStrainModel):
    supports_rate_dependent = True

    def __init__(
        self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"
    ):
        super().__init__(material_type, configuration, solver_type)
        self.yield_surface_type = 0

    def model_initialize(self, material):
        self.material = material
        density = DictIO.get_alternative(material, "Density", 2650)
        young = DictIO.get_essential(material, "YoungModulus")
        poisson = DictIO.get_alternative(material, "PoissonRatio", 0.3)
        self.validate_elastic_parameters(density, young, poisson)
        self.rate_dependent_function = RateDependent(material)
        self.add_material(density, young, poisson)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson):
        self.density = density
        self.young = young
        self.poisson = poisson

        self.shear = 0.5 * self.young / (1.0 + self.poisson)
        self.bulk = self.young / (3.0 * (1 - 2.0 * self.poisson))
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def print_message(self, materialID):
        self.print_console_header()
        print("Constitutive model: Granular Material")
        print("Material ID: ", materialID)
        print("Density: ", self.density)
        if GlobalVariable.RANDOMFIELD is False:
            print("Young Modulus: ", self.young)
            print("Poisson Ratio: ", self.poisson)
        self.rate_dependent_function.print_message()
        print("\n")

    def define_state_vars(self):
        state_vars = {"epdstrain": float}
        if GlobalVariable.RANDOMFIELD:
            state_vars.update({"density": float, "shear": float, "bulk": float})
        return state_vars

    def read_random_field(self, start_particle, end_particle, stateVars):
        random_field = np.loadtxt(self.random_field_file, unpack=True, comments="#").transpose()
        if random_field.shape[0] < end_particle - start_particle:
            raise SimulationStateError("Shape error for the random field file")
        density = np.ascontiguousarray(random_field[0:, 0])
        young = np.ascontiguousarray(random_field[0:, 1])
        poisson = np.ascontiguousarray(random_field[0:, 2])
        shear = 0.5 * young / (1.0 + poisson)
        bulk = young / (3.0 * (1 - 2.0 * poisson))
        self.kernel_add_random_material(start_particle, end_particle, density, shear, bulk, stateVars)
        self.max_sound_speed = np.max(self.get_sound_speed(density, young, poisson))

    @ti.kernel
    def kernel_add_random_material(
        self,
        start_particle: int,
        end_particle: int,
        density: ti.types.ndarray(),
        shear: ti.types.ndarray(),
        bulk: ti.types.ndarray(),
        state_variables: ti.template(),
    ):
        for np in range(start_particle, end_particle):
            state_variables[np].density = density[np - start_particle]
            state_variables[np].shear = shear[np - start_particle]
            state_variables[np].bulk = bulk[np - start_particle]

    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, state_variables):
        state_variables[np].epdstrain = 0.0

    # ==================================================== Drucker-Parger Model ==================================================== #
    @ti.func
    def compute_stress2d(self, np, previous_stress, velocity_gradient, state_variables, dt):
        ############################## STEP1 ##############################
        de = calculate_strain_increment2d(velocity_gradient, dt)
        dw = calculate_vorticity_increment2d(velocity_gradient, dt)
        previous_stress = self.implicit_integration(np, previous_stress, de, dw, state_variables, dt)
        return previous_stress

    @ti.func
    def compute_stress(self, np, previous_stress, velocity_gradient, state_variables, dt):
        de = calculate_strain_increment(velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)
        previous_stress = self.implicit_integration(np, previous_stress, de, dw, state_variables, dt)
        return previous_stress

    @ti.func
    def implicit_integration(self, np, previous_stress, de, dw, state_variables, dt):
        bulk_mod = self.bulk
        shear_mod = self.shear

        dstress = elastic_tensor_multiply_vector(de, bulk_mod, shear_mod)
        trial_stress = previous_stress + dstress + sigrot(previous_stress, dw)

        plastic_strain, updated_stress = self.rate_dependent_function.solve(shear_mod, trial_stress, dt)
        state_variables[np].epdstrain += plastic_strain
        return updated_stress
