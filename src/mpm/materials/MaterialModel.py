import numpy as np
import taichi as ti

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.MaterialKernel import *
from src.utils.constants import DELTA
from src.utils.TypeDefination import mat3x3


class UserDefined(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_particle_num, configuration, solver_type):
        self.matProps = MaterialModel.field(shape=max_material_num)
        if configuration == "ULMPM":
            self.stateVars = ULStateVariable.field(shape=max_particle_num) 
        elif configuration == "TLMPM":
            self.stateVars = TLStateVariable.field(shape=max_particle_num) 

        if solver_type == "Implicit":
            self.stiffness_matrix = ti.Matrix.field(6, 6, float, shape=max_particle_num)

    def model_initialize(self, material):
        pass

    def reload_state_variables(self, state_vars):
        pass

    def get_state_vars_dict(self, start_particle, end_particle):
        pass


@ti.dataclass
class ULStateVariable:
    # TODO: add essential state variable for constitutive model
    estress: float

    @ti.func
    def _initialize_vars(self, stress):
        pass

    @ti.func
    def _update_vars(self, stress, epstrain):
        pass    


@ti.dataclass
class TLStateVariable:
    # TODO: add essential state variable for constitutive model
    estress: float
    deformation_gradient: mat3x3

    @ti.func
    def _initialize_vars(self, stress):
        self.deformation_gradient = DELTA
        pass

    @ti.func
    def _update_deformation_gradient(self, deformation_gradient_rate, dt):
        self.deformation_gradient += deformation_gradient_rate * dt[None]

    @ti.func
    def _update_vars(self, stress, epstrain):
        pass  


@ti.dataclass
class MaterialModel:
    density: float
    # TODO: Add essential material properties

    def add_material(self, density):
        self.density = density
        # TODO: Add essential material properties

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model = Material Model')
        print("Model ID: ", materialID)
        print("Model density = ",  self.density)

    @ti.func
    def _get_sound_speed(self):
        sound_speed = 0.
        if self.density > 0.:
            pass
        return sound_speed
        # TODO: Add proporiate equations of sound speed

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, particle, dt):
        particle[np].vol *= (DELTA + velocity_gradient * dt[None]).determinant()
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, particle, dt):
        particle[np].vol *= 1. + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])

    @ti.func
    def ComputeStress(self,
                      np,                                                      # particle id 
                      stateVars,                                               # state variables
                      particle,                                                # particle pointer
                      dt                                                       # time step
                     ):                  
        pass


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]

