import numpy as np
import taichi as ti

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.MaterialKernel import *
from src.utils.constants import DELTA
from src.utils.MatrixFunction import trace, matrix_form
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import mat3x3
from src.utils.VectorFunction import voigt_form


class NeoHookean(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_particle_num, configuration="ULMPM", solver_type="Explicit"):
        super().__init__()
        self.matProps = NeoHookeanModel.field(shape=max_material_num)
        self.stateVars = StateVariable.field(shape=max_particle_num)

    def get_state_vars_dict(self, start_particle, end_particle):
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        deformation_gradient = np.ascontiguousarray(self.stateVars.deformation_gradient.to_numpy()[start_particle:end_particle])
        return {'estress': estress, 'deformation_gradient': deformation_gradient}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        deformation_gradient = state_vars.item()['deformation_gradient']
        kernel_reload_state_variables(estress, deformation_gradient, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID') 
        self.check_materialID(materialID, self.matProps.shape[0])

        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        possion = DictIO.GetAlternative(material, 'PossionRatio', 0.3)

        self.matProps[materialID].add_material(density, young, possion)
        self.matProps[materialID].print_message(materialID)

    def get_lateral_coefficient(self, materialID):
        mu = self.matProps[materialID].possion
        return 0.3#mu / (1. - mu)


@ti.dataclass
class StateVariable:
    estress: float
    deformation_gradient: mat3x3
    stress0: mat3x3
    stress: mat3x3

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.estress = VonMisesStress(stress)
        self.deformation_gradient = DELTA
        self.stress0 = matrix_form(stress)

    @ti.func
    def _update_deformation_gradient(self, deformation_gradient_rate, dt):
        self.deformation_gradient += deformation_gradient_rate * dt[None]

    @ti.func
    def _update_vars(self, stress):
        self.estress = VonMisesStress(stress)


@ti.dataclass
class NeoHookeanModel:
    density: float
    young: float
    possion: float
    shear: float
    bulk: float

    def add_material(self, density, young, possion):
        self.density = density
        self.young = young
        self.possion = possion

        self.shear = 0.5 * self.young / (1. + self.possion)
        self.bulk = self.young / (3. * (1 - 2. * self.possion))
        
    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Neo-Hookean')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Possion Ratio: ', self.possion, '\n')

    @ti.func
    def _get_sound_speed(self):
        sound_speed = 0.
        if self.density > 0.:
            sound_speed = ti.sqrt(self.young * (1 - self.possion) / (1 + self.possion) / (1 - 2 * self.possion) / self.density)
        return sound_speed
    
    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        stateVars[np].deformation_gradient = (DELTA + velocity_gradient * dt[None]) @ stateVars[np].deformation_gradient
        return (DELTA + velocity_gradient * dt[None]).determinant()
    
    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        return self.core(np, previous_stress, stateVars)

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        return self.core(np, stateVars)
    
    @ti.func
    def core(self, np, stateVars): 
        deformation_gradient = stateVars[np].deformation_gradient
        left_cauchy_green_tensor = deformation_gradient @ deformation_gradient.transpose()
        jacobian = deformation_gradient.determinant()
        ijacobian = 1. / jacobian
        para1 = self.shear * ijacobian ** (5./3.) * left_cauchy_green_tensor
        para2 = self.bulk * (jacobian - 1.) * DELTA
        para3 = 1./3. * self.shear * trace(left_cauchy_green_tensor) * ijacobian ** (5./3.)
        cauchy_stress = stateVars[np].stress0 + para1 + para2 + para3
        voigt_stress = voigt_form(cauchy_stress)
        stateVars[np].estress = VonMisesStress(voigt_stress)
        return cauchy_stress

    @ti.func
    def ComputePKStress2D(self, np, velocity_gradient, stateVars, dt):  
        PKstress = self.corePK(np, stateVars)
        stateVars[np].stress = PKstress
        return PKstress

    @ti.func
    def ComputePKStress(self, np, velocity_gradient, stateVars, dt):  
        PKstress = self.corePK(np, stateVars)
        stateVars[np].stress = PKstress
        return PKstress

    @ti.func
    def corePK(self, np, stateVars):  
        shear_modulus = self.shear
        la = 3. * self.bulk * self.possion / (1 + self.possion) 
        F = stateVars[np].deformation_gradient
        det_F = F.determinant()
        F_inverse_T = F.inverse().transpose()
        PKstress = stateVars[np].stress0 + shear_modulus * (F - F_inverse_T) + la * ti.log(det_F) * F_inverse_T
        voigt_stress = voigt_form(PKstress)
        stateVars[np].estress = VonMisesStress(voigt_stress)
        return PKstress
    
    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        return self.compute_stiffness_tensor(np, current_stress, stateVars)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stateVars):
        deformation_gradient = stateVars[np].deformation_gradient
        jacobian = deformation_gradient.determinant()
        lambda_ = 3. * self.bulk * self.possion / (1 + self.possion) 
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


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), deformation_gradient: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].deformation_gradient = deformation_gradient[np]