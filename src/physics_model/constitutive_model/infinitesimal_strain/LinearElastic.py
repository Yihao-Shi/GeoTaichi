import taichi as ti

from src.physics_model.constitutive_model.infinitesimal_strain.MaterialKernel import von_mises_stress
from src.physics_model.constitutive_model.infinitesimal_strain.ElasticMaterial import ElasticMaterial
from src.utils.ObjectIO import DictIO
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class LinearElasticModel(ElasticMaterial):
    def __init__(
        self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"
    ):
        super().__init__(material_type, configuration, solver_type)
        self.is_elastic = True

    def model_initialize(self, material):
        self.material = material
        density = DictIO.get_alternative(material, "Density", 2650)
        young = DictIO.get_essential(material, "YoungModulus")
        poisson = DictIO.get_alternative(material, "PoissonRatio", 0.3)
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
        print("Constitutive model: Linear Elastic")
        print("Material ID: ", materialID)
        print("Density: ", self.density)
        print("Young Modulus: ", self.young)
        print("Poisson Ratio: ", self.poisson, "\n")

    def define_state_vars(self):
        return {"estress": float}

    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, state_variables):
        pass

    # ==================================================== Linear elastic Model ==================================================== #
    @ti.func
    def cores(self, np, previous_stress, de, dw, state_variables):
        material_params = self.get_material_parameter(previous_stress, state_variables[np])
        stress = self.compute_elastic_stress(1.0, de, previous_stress, material_params)
        current_stress = stress + self.compute_sigrot_stress(dw, stress)
        state_variables[np].estress = von_mises_stress(current_stress)
        return current_stress

    @ti.func
    def get_material_parameter(self, stress, state_vars):
        bulk, shear = self.get_current_material_parameter(state_vars)
        return ti.Vector([bulk, shear])

    @ti.func
    def get_current_material_parameter(self, state_vars):
        if ti.static(GlobalVariable.RANDOMFIELD):
            return state_vars.bulk, state_vars.shear
        else:
            return self.bulk, self.shear
