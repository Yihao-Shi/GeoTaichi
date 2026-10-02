import taichi as ti

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import VonMisesStress
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import ElasticMaterial
from src.utils.ObjectIO import DictIO
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class LinearElasticModel(ElasticMaterial):
    def __init__(self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type)
        self.is_elastic = True

    def model_initialize(self, material):
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        self.add_material(density, young, poisson)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson):
        self.density = density
        self.young = young
        self.poisson = poisson

        self.shear = 0.5 * self.young / (1. + self.poisson)
        self.bulk = self.young / (3. * (1 - 2. * self.poisson))
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Elastic Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Poisson Ratio: ', self.poisson, '\n')

    def define_state_vars(self):
        return {'estress': float}

    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        pass

    # ==================================================== Linear elastic Model ==================================================== #
    @ti.func
    def cores(self, np, previous_stress, de, dw, stateVars): 
        material_params = self.GetMaterialParameter(previous_stress, stateVars[np])
        stress = self.ComputeElasticStress(1., de, previous_stress, material_params)
        current_stress = stress + self.ComputeSigrotStress(dw, stress)
        stateVars[np].estress = VonMisesStress(current_stress)
        return current_stress
        
    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        bulk, shear = self.get_current_material_parameter(state_vars)
        return ti.Vector([bulk, shear])

    @ti.func
    def get_current_material_parameter(self, state_vars):
        if ti.static(GlobalVariable.RANDOMFIELD):
            return state_vars.bulk, state_vars.shear
        else:
            return self.bulk, self.shear


