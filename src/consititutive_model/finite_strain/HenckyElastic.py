import taichi as ti

from src.consititutive_model.MaterialKernel import *
from src.utils.constants import DELTA, DELTA2D, ZEROMAT6x6
from src.utils.MatrixFunction import principal_tensor, Diagonal, matrix_form
from src.utils.TypeDefination import mat3x3
from src.utils.VectorFunction import voigt_form


@ti.dataclass
class StateVariable:
    estress: float
    deformation_gradient: mat3x3
    stress: mat3x3
    stress0: mat3x3

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
class HenckyElasticModel:
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

    def add_contact_parameter(self, friction, kn, kt):
        self.friction = friction
        self.kn = kn
        self.kt = kt
        
    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Elastic Model')
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
        deformation_gradient_rate = DELTA + velocity_gradient * dt[None]
        stateVars[np].deformation_gradient = deformation_gradient_rate @ stateVars[np].deformation_gradient
        return deformation_gradient_rate.determinant()
    
    @ti.func
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        deformation_gradient_rate = DELTA2D + velocity_gradient * dt[None]
        stateVars[np].deformation_gradient = deformation_gradient_rate @ stateVars[np].deformation_gradient
        return deformation_gradient_rate.determinant()
    
    @ti.func
    def elastic_part(self):
        a1 = self.bulk + (4./3.) * self.shear
        a2 = self.bulk - (2./3.) * self.shear
        return mat3x3([[a1, a2, a2], [a2, a1, a2], [a2, a2, a1]])
    
    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        return self.core(np, stateVars)

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        return self.core(np, stateVars)
    
    @ti.func
    def core(self, np, stateVars): 
        kirchhoff_stress = self.corePK(np, stateVars)
        deformation_gradient = stateVars[np].deformation_gradient
        ijacobian = 1. / deformation_gradient.determinant()
        cauchy_stress = ijacobian * kirchhoff_stress
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
        deformation_gradient = stateVars[np].deformation_gradient
        left_cauchy_green_tensor = deformation_gradient @ deformation_gradient.transpose()
        principal_left_cauchy_green_strain, directors = principal_tensor(left_cauchy_green_tensor)
        principal_hencky_strain = 0.5 * ti.log(principal_left_cauchy_green_strain)
        principal_kirchhoff_stress = self.elastic_part() @ principal_hencky_strain
        kirchhoff_stress = stateVars[np].stress0 + directors @ Diagonal(principal_kirchhoff_stress) @ directors.transpose()
        voigt_stress = voigt_form(kirchhoff_stress)
        stateVars[np].estress = VonMisesStress(voigt_stress)
        return kirchhoff_stress
    
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