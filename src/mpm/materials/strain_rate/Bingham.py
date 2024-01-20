import taichi as ti
import numpy as np

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.MaterialKernel import *
from src.utils.constants import ZEROVEC6f, EYE, Threshold, DELTA
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import mat3x3


class Bingham(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_particle_num):
        super().__init__()
        self.matProps = BinghamModel.field(shape=max_material_num)
        self.stateVars = ULStateVariable.field(shape=max_particle_num) 

    def get_state_vars_dict(self, start_particle, end_particle):
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        dvolumetric_strain = np.ascontiguousarray(self.stateVars.dvolumetric_strain.to_numpy()[start_particle:end_particle])
        return {'estress': estress, 'dvolumetric_strain': dvolumetric_strain}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        dvolumetric_strain = state_vars.item()['dvolumetric_strain']
        kernel_reload_state_variables(estress, dvolumetric_strain, self.stateVars)

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
        
        self.matProps[materialID].add_material(density, modulus, viscosity, _yield, critical_rate)
        self.matProps[materialID].print_message(materialID)

    def get_lateral_coefficient(self, materialID):
        return 1.


@ti.dataclass
class ULStateVariable:
    estress: float
    dvolumetric_strain: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.estress = MeanStress(stress)
        self.dvolumetric_strain = 1.

    @ti.func
    def _update_vars(self, stress):
        self.estress = MeanStress(stress)    


@ti.dataclass
class BinghamModel:
    density: float
    modulus: float
    viscosity: float
    _yield: float
    critical_rate: float

    def add_material(self, density, modulus, viscosity, _yield, critical_rate):
        self.density = density
        self.modulus = modulus
        self.viscosity = viscosity
        self._yield = _yield
        self.critical_rate = critical_rate

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
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        delta_jacobian = 1. + dt[None] * (velocity_gradient[0, 0] + velocity_gradient[1, 1] + velocity_gradient[2, 2])
        stateVars[np].dvolumetric_strain *= delta_jacobian
        return delta_jacobian
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        delta_jacobian = 1. + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])
        stateVars[np].dvolumetric_strain *= delta_jacobian
        return delta_jacobian

    @ti.func
    def thermodynamic_pressure(self, dvolumertic):
        gamma = 7
        pressure = -self.modulus * (dvolumertic ** gamma - 1)
        return pressure
    
    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):
        strain_rate = calculate_strain_rate2D(velocity_gradient)
        return self.core(np, strain_rate, stateVars)

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):
        strain_rate = calculate_strain_rate(velocity_gradient)
        return self.core(np, strain_rate, stateVars)

    @ti.func
    def core(self, np, strain_rate, stateVars):   
        _yield = self._yield
        viscosity = self.viscosity
        critical_shear_rate = self.critical_rate

        # Convert strain rate to rate of deformation tensor
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

        pressure = self.thermodynamic_pressure(stateVars[np].dvolumetric_strain)
        stateVars[np].estress = pressure
        return -pressure * EYE + tau 


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), dvolumetric_strain: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].dvolumetric_strain = dvolumetric_strain[np]