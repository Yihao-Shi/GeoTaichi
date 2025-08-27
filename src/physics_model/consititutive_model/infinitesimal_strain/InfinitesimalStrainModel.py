import taichi as ti

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
import src.utils.GlobalVariable as GlobalVariable
from src.physics_model.consititutive_model.MaterialModel import Solid


@ti.data_oriented
class InfinitesimalStrainModel(Solid):
    def __init__(self, material_type, configuration, solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)

    def get_state_vars(self):
        state_vars = self.define_state_vars()
        if self.is_elastic is False and self.solver_type == 1:
            state_vars.update({"yield_state": ti.u8})
        if "UL" in self.configuration:
            self._initialize_vars = self._initialize_vars_update_lagrangian
            return state_vars
        elif "TL" in self.configuration:
            self._initialize_vars = self._initialize_vars_total_lagrangian
            state_vars.update({'deformation_gradient': ti.types.matrix(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION, float)})
            return state_vars

    @ti.func
    def _initialize_vars_update_lagrangian(np, particle, stateVars):
        raise NotImplementedError
    
    @ti.func
    def _initialize_vars_total_lagrangian(self, np, particle, stateVars):
        self._initialize_vars_update_lagrangian(np, particle, stateVars)
        stateVars[np].deformation_gradient = ti.Matrix.identity(float, GlobalVariable.DIMENSION)    

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        return (ti.Matrix.identity(float, 3) + velocity_gradient * dt[None]).determinant()
    
    @ti.func
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        return (ti.Matrix.identity(float, 2) + velocity_gradient * dt[None]).determinant()
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        return 1. + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])

    @ti.func
    def ComputePKStress(self, np, previous_PKstress, velocity_gradient, stateVars, dt):  
        previous_cauchy_stress = self.PK2CauchyStress(np, previous_PKstress, stateVars)
        cauchy_stress = self.ComputeStress(np, previous_cauchy_stress, velocity_gradient, stateVars, dt)
        PKstress = self.Cauchy2PKStress(np, stateVars, cauchy_stress)
        return PKstress
    
    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        de = calculate_strain_increment2D(velocity_gradient, dt)
        dw = calculate_vorticity_increment2D(velocity_gradient, dt)
        return self.core(np, previous_stress, de, dw, stateVars)

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        de = calculate_strain_increment(velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)
        return self.core(np, previous_stress, de, dw, stateVars)
