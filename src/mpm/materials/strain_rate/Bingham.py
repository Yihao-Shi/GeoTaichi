import taichi as ti
import numpy as np

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.MaterialKernel import *
from src.utils.constants import ZEROVEC6f, EYE, Threshold
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_trace


class Bingham(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_particle_num):
        super().__init__()
        self.matProps = BinghamModel.field(shape=max_material_num)
        self.stateVars = ULStateVariable.field(shape=max_particle_num) 

    def get_state_vars_dict(self, start_particle, end_particle):
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        p0 = np.ascontiguousarray(self.stateVars.p0.to_numpy()[start_particle:end_particle])
        return {'estress': estress, 'p0': p0}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        p0 = state_vars.item()['p0']
        kernel_reload_state_variables(estress, p0, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID')
        self.check_materialID(materialID, self.matProps.shape[0])
        
        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, 'Density', 1000)
        modulus = DictIO.GetEssential(material, 'Modulus')
        viscosity = DictIO.GetEssential(material, 'Viscosity')
        _yield = DictIO.GetEssential(material, 'YieldStress')
        critical_rate = DictIO.GetEssential(material, 'CriticalStrainRate')
        gamma = DictIO.GetAlternative(material, 'gamma', 7.)
        
        self.matProps[materialID].add_material(density, modulus, viscosity, _yield, critical_rate, gamma)
        self.matProps[materialID].print_message(materialID)

    def get_lateral_coefficient(self, materialID):
        return 1.


@ti.dataclass
class ULStateVariable:
    estress: float
    p0: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.estress = -MeanStress(stress)
        self.p0 = -MeanStress(stress)

    @ti.func
    def _update_vars(self, stress):
        self.estress = -MeanStress(stress)    


@ti.dataclass
class BinghamModel:
    density: float
    modulus: float
    viscosity: float
    _yield: float
    critical_rate: float
    gamma: float

    def add_material(self, density, modulus, viscosity, _yield, critical_rate, gamma):
        self.density = density
        self.modulus = modulus
        self.viscosity = viscosity
        self._yield = _yield
        self.critical_rate = critical_rate
        self.gamma = gamma

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model = Bingham Model')
        print("Model ID: ", materialID)
        print("Model density = ",  self.density)
        print('Bulk Modulus = ', self.modulus)
        print('Viscosity = ', self.viscosity)
        print('Yield Stress = ', self._yield)
        print('Critical Shear Rate = ', self.critical_rate, '\n')

    @ti.func
    def _get_sound_speed(self):
        sound_speed = 0.
        if self.density > 0.:
            sound_speed = ti.sqrt(self.modulus / self.density)
        return sound_speed
    
    @ti.func
    def _set_modulus(self, velocity):
        velocity = 1000 * velocity
        ti.atomic_max(self.modulus, self.density * velocity / self.gamma)
    
    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        delta_jacobian = 1. + dt[None] * (velocity_gradient[0, 0] + velocity_gradient[1, 1] + velocity_gradient[2, 2])
        return delta_jacobian
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        delta_jacobian = 1. + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])
        return delta_jacobian

    @ti.func
    def thermodynamic_pressure(self, dvolumertic_strain):
        pressure = -self.modulus * dvolumertic_strain 
        return pressure
    
    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):
        strain_rate = calculate_strain_rate2D(velocity_gradient)
        return self.core(np, strain_rate, stateVars, dt)

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):
        strain_rate = calculate_strain_rate(velocity_gradient)
        return self.core(np, strain_rate, stateVars, dt)

    @ti.func
    def core(self, np, strain_rate, stateVars, dt):   
        _yield = self._yield
        viscosity = self.viscosity
        critical_shear_rate = self.critical_rate

        # Convert strain rate to rate of deformation tensor
        volumetric_strain_rate = voigt_tensor_trace(strain_rate)
        for d in ti.static(range(3, 6)):
            strain_rate[d] *= 0.5
        
        # critical yielding shear rate
        critical_shear_rate = ti.min(Threshold, critical_shear_rate)

        # Rate of shear = sqrt(2 * D_ij * D_ij)
        strain_rate_tail = strain_rate[3] * strain_rate[3] + strain_rate[4] * strain_rate[4] + strain_rate[5] * strain_rate[5]
        shear_rate = ti.sqrt(2. * (strain_rate.dot(strain_rate) + strain_rate_tail))
        
        apparent_viscosity = 0.
        if shear_rate * shear_rate > critical_shear_rate * critical_shear_rate:
            apparent_viscosity = 2. * (_yield / shear_rate + viscosity)

        tau = apparent_viscosity * strain_rate

        trace_invariant2 = 0.5 * (tau[0] * tau[0] + tau[1] * tau[1] + tau[2] * tau[2])
        if trace_invariant2 < _yield * _yield: tau = ZEROVEC6f

        pressure = stateVars[np].p0 + self.thermodynamic_pressure(volumetric_strain_rate * dt[None])
        updated_stress = -pressure * EYE + tau 
        stateVars[np].estress = -MeanStress(updated_stress)
        stateVars[np].p0 = pressure
        return updated_stress


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), p0: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].p0 = p0[np]