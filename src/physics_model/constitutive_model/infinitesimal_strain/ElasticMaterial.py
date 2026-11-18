import taichi as ti
from src.physics_model.constitutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.constitutive_model.SoftenModel import *
from src.physics_model.constitutive_model.infinitesimal_strain.RateDependent import *
from src.physics_model.constitutive_model.infinitesimal_strain.InfinitesimalStrainModel import InfinitesimalStrainModel


@ti.data_oriented
class ElasticMaterial(InfinitesimalStrainModel):
    def __init__(self, material_type, configuration, solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.core = self.cores

    def calculate_lame_parameter(self, young, poisson):
        shear = 0.5 * young / (1.0 + poisson)
        bulk = young / (3.0 * (1 - 2.0 * poisson))
        return shear, bulk

    def read_random_field(self, start_particle, end_particle, stateVars):
        raise NotImplementedError

    @ti.kernel
    def kernel_add_random_material(self):
        raise NotImplementedError

    @ti.func
    def cores(self, np, previous_stress, de, dw, state_variables):
        raise NotImplementedError

    @ti.func
    def get_material_parameter(self, stress, state_vars):
        raise NotImplementedError

    @ti.func
    def compute_sigrot_stress(self, dsigrot, stress):
        return sigrot(stress, dsigrot)

    @ti.func
    def compute_elastic_modulus(self, stress, material_params):
        return material_params[0], material_params[1]

    @ti.func
    def compute_elastic_stress(self, alpha, dstrain, stress, material_params):
        return stress + self.compute_elastic_stress_increment(alpha * dstrain, stress, material_params)

    @ti.func
    def compute_elastic_stress_increment(self, dstrain, stress, material_params):
        bulk_modulus, shear_modulus = self.compute_elastic_modulus(stress, material_params)

        # !-- trial elastic stresses ----!
        dstress = elastic_tensor_multiply_vector(dstrain, bulk_modulus, shear_modulus)
        return dstress

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, state_variables):
        material_params = self.get_material_parameter(current_stress, state_variables[np])
        bulk_modulus, shear_modulus = self.compute_elastic_modulus(current_stress, material_params)
        return compute_elastic_stiffness_tensor(bulk_modulus, shear_modulus)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, state_variables):
        return self.compute_elastic_tensor(np, current_stress, state_variables)
