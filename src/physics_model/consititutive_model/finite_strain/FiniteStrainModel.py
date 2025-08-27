import taichi as ti

from src.physics_model.consititutive_model.MaterialModel import Solid
from src.utils.constants import DELTA, DELTA2D
from src.utils.TypeDefination import vec3f
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class FiniteStrainModel(Solid):
    def __init__(self, material_type, configuration, solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.young = 0.
        self.shear = 0.
        self.bulk = 0.

    def get_state_vars(self):
        self._initialize_vars = self._initialize_vars_
        return self.define_state_vars()
    
    @ti.func
    def _initialize_vars_(self, np, particle, stateVars):
        raise NotImplementedError   
    
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
    def ComputeStress2D(self, np, previous_cauchy_stress, velocity_gradient, stateVars, dt):  
        previous_PKstress = self.Cauchy2PKStress(np, stateVars, previous_cauchy_stress)
        PKstress = self.ComputePKStress2D(np, previous_PKstress, velocity_gradient, stateVars, dt)
        return self.PK2CauchyStress(np, PKstress, stateVars)

    @ti.func
    def ComputeStress(self, np, previous_cauchy_stress, velocity_gradient, stateVars, dt):  
        previous_PKstress = self.Cauchy2PKStress(np, stateVars, previous_cauchy_stress)
        PKstress = self.ComputePKStress(np, previous_PKstress, velocity_gradient, stateVars, dt)
        return self.PK2CauchyStress(np, PKstress, stateVars)

    @ti.func
    def ComputePKStress2D(self, np, presvious_stress, velocity_gradient, stateVars, dt):  
        PKstress = self.corePK(np, stateVars)
        return PKstress

    @ti.func
    def ComputePKStress(self, np, presvious_stress, velocity_gradient, stateVars, dt):  
        PKstress = self.corePK(np, stateVars)
        return PKstress
    
    @ti.func
    def corePK(self, np, stateVars):  
        raise NotImplementedError
    
    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        pass


@ti.data_oriented
class ElasticModel(FiniteStrainModel):
    def __init__(self, material_type, configuration, solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)


@ti.data_oriented
class PlasticModel(ElasticModel):
    def __init__(self, material_type, configuration, solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)

    @ti.func
    def core(self, np, stateVars):
        trial_deformation_gradient = stateVars[np].deformation_gradient
        matrixU, sigma, matrixVT = ti.svd(trial_deformation_gradient)
        hencky_strain = ti.log(vec3f(ti.max(1e-4, ti.abs(sigma[0])), ti.max(1e-4, ti.abs(sigma[1])), ti.max(1e-4, ti.abs(sigma[2]))))
        hencky_trace_trace = hencky_strain[0] + hencky_strain[1] + hencky_strain[2]
        hencky_deviatoric = hencky_strain - (hencky_trace_trace / GlobalVariable.DIMENSION) * ti.Vector.one(float, GlobalVariable.DIMENSION)
        hencky_deviatoric_norm = hencky_deviatoric.norm()
        if hencky_deviatoric_norm > 0.: hencky_deviatoric /= hencky_deviatoric_norm
        return self.plastic_process(matrixU, matrixVT, hencky_strain, hencky_trace_trace, hencky_deviatoric, hencky_deviatoric_norm, stateVars[np])

    @ti.func
    def plastic_process(self, matrixU, matrixVT, hencky_strain, hencky_trace_trace, hencky_deviatoric, hencky_deviatoric_norm, state_vars):
        raise NotImplementedError
