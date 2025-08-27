import taichi as ti

from src.physics_model.consititutive_model.finite_strain.FiniteStrainModel import FiniteStrainModel
import src.utils.GlobalVariable as GlobalVariable
from src.utils.constants import ZEROMAT6x6
from src.utils.MatrixFunction import principal_tensor, Diagonal
from src.utils.TypeDefination import mat3x3
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class HenckyElasticModel(FiniteStrainModel):
    def __init__(self, material_type="Solid", configuration="TL", solver_type="Explicit"):
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
        self.get_sound_speed(self.density, self.young, self.poisson)

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Hencky Elastic Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Poisson Ratio: ', self.poisson, '\n')

    def define_state_vars(self):
        return {'stress0': mat3x3, 'deformation_gradient': ti.types.matrix(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION, float)}

    @ti.func
    def _initialize_vars_(self, np, particle, stateVars):
        stateVars[np].deformation_gradient = ti.Matrix.identity(float, GlobalVariable.DIMENSION)    
        stateVars[np].stress0 = particle[np].stress
    
    @ti.func
    def elastic_part(self):
        a1 = self.bulk + (4./3.) * self.shear
        a2 = self.bulk - (2./3.) * self.shear
        return mat3x3([[a1, a2, a2], [a2, a1, a2], [a2, a2, a1]])
    
    @ti.func
    def corePK(self, np, stateVars):  
        deformation_gradient = stateVars[np].deformation_gradient
        left_cauchy_green_tensor = deformation_gradient @ deformation_gradient.transpose()
        principal_left_cauchy_green_strain, directors = principal_tensor(left_cauchy_green_tensor)
        principal_hencky_strain = 0.5 * ti.log(principal_left_cauchy_green_strain)
        principal_kirchhoff_stress = self.elastic_part() @ principal_hencky_strain
        kirchhoff_stress = stateVars[np].stress0 + directors @ Diagonal(principal_kirchhoff_stress) @ directors.transpose()
        return kirchhoff_stress
    
    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        deformation_gradient = stateVars[np].deformation_gradient
        jacobian = deformation_gradient.determinant()
        lambda_ = 3. * self.bulk * self.poisson / (1 + self.poisson) 
        modified_shear = (self.shear - lambda_ * ti.log(jacobian)) / jacobian
        modified_lambda = lambda_ / jacobian

        a1 = modified_lambda + 2. * modified_shear
        a2 = modified_lambda
        stiffness = ZEROMAT6x6
        stiffness[0, 0] = stiffness[1, 1] = stiffness[2, 2] = a1
        stiffness[0, 1] = stiffness[0, 2] = stiffness[1, 2] = a2
        stiffness[1, 0] = stiffness[2, 0] = stiffness[2, 1] = a2
        stiffness[3, 3] = stiffness[4, 4] = stiffness[5, 5] = modified_shear
        return stiffness

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stateVars):
        return self.compute_elastic_tensor(np, current_stress, stateVars)