import numpy as np
import taichi as ti

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.MaterialKernel import *
from src.utils.constants import ZEROVEC6f
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_trace


class Newtonian(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_particle_num):
        super().__init__()
        self.matProps = NewtonianModel.field(shape=max_material_num)
        self.stateVars = ULStateVariable.field(shape=max_particle_num) 

    def get_state_vars_dict(self, start_particle, end_particle):
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        dvolumertic_strain = np.ascontiguousarray(self.stateVars.dvolumertic_strain.to_numpy()[start_particle:end_particle])
        rho = np.ascontiguousarray(self.stateVars.rho.to_numpy()[start_particle:end_particle])
        return {'estress': estress, 'dvolumertic_strain': dvolumertic_strain, "rho": rho}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        dvolumertic_strain = state_vars.item()['dvolumertic_strain']
        rho = state_vars.item()['rho']
        kernel_reload_state_variables(estress, dvolumertic_strain, rho, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID')
        self.check_materialID(materialID, self.matProps.shape[0])
        
        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, 'Density', 1000)
        modulus = DictIO.GetAlternative(material, 'Modulus', 3.6e5)
        viscosity = DictIO.GetAlternative(material, 'Viscosity', 1e-3)
        gamma = DictIO.GetAlternative(material, 'gamma', 1.)
        cl = DictIO.GetAlternative(material, 'cL', 1.)
        cq = DictIO.GetAlternative(material, 'cQ', 2.)
        element_length = DictIO.GetAlternative(material, 'ElementLength', 0)

        self.matProps[materialID].add_material(density, modulus, viscosity, gamma, element_length, cl, cq)
        self.matProps[materialID].print_message(materialID)

    def get_lateral_coefficient(self, materialID):
        return 1.


@ti.dataclass
class ULStateVariable:
    estress: float
    dvolumertic_strain: float
    rho: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.estress = -MeanStress(stress)
        self.dvolumertic_strain = 0.
        self.rho = matProps[int(particle[np].materialID)].density

    @ti.func
    def _update_vars(self, stress):
        self.estress = -MeanStress(stress)    


@ti.dataclass
class NewtonianModel:
    density: float
    modulus: float
    viscosity: float
    gamma: float
    element_length: float
    cl: float
    cq: float

    def add_material(self, density, modulus, viscosity, gamma, element_length, cl, cq):
        self.density = density
        self.modulus = modulus
        self.viscosity = viscosity
        self.gamma = gamma
        self.element_length = element_length
        self.cl = cl
        self.cq = cq

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model = Newtonian Model')
        print("Model ID: ", materialID)
        print("Model density = ",  self.density)
        print('Bulk Modulus = ', self.modulus)
        print('Viscosity = ', self.viscosity)
        print('Power Index = ', self.gamma)
        print("Characteristic Element Length = ", self.element_length)
        print("Artifical Viscosity Parameter = ", self.cl, self.cq, '\n')

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
        stateVars[np].rho /= delta_jacobian
        return delta_jacobian
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        delta_jacobian = 1. + dt[None] * voigt_tensor_trace(strain_rate)
        stateVars[np].rho /= delta_jacobian
        return delta_jacobian

    @ti.func
    def thermodynamic_pressure(self, rho, dvolumertic):
        pressure = -self.modulus / self.density * rho * dvolumertic ** self.gamma 
        return pressure
    
    @ti.func
    def artifical_viscosity(self, np, volumetric_strain_rate, stateVars):
        # VonNeumann J. 1950, A method for the numerical calculation of hydrodynamic shocks. J. Appl. Phys.
        q = 0.
        if volumetric_strain_rate < 0.:
            q = -stateVars[np].rho * self.cl * self.element_length * volumetric_strain_rate + \
                stateVars[np].rho * self.cq * self.element_length * self.element_length * volumetric_strain_rate * volumetric_strain_rate
        return q
    
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
        viscosity = self.viscosity
        volumetric_strain_rate = voigt_tensor_trace(strain_rate) 
        volumetric_strain = stateVars[np].dvolumertic_strain + volumetric_strain_rate * dt[None]
        
        pressure = self.thermodynamic_pressure(stateVars[np].rho, volumetric_strain)
        artifical_pressure = self.artifical_viscosity(np, volumetric_strain_rate, stateVars)
        pressureAV = pressure + artifical_pressure
        volumetric_component = -pressureAV - 2. * viscosity * volumetric_strain_rate / 3.
        
        pstress = ZEROVEC6f
        pstress[0] = volumetric_component + 2. * viscosity * strain_rate[0]
        pstress[1] = volumetric_component + 2. * viscosity * strain_rate[1]
        pstress[2] = volumetric_component + 2. * viscosity * strain_rate[2]
        pstress[3] = viscosity * strain_rate[3]
        pstress[4] = viscosity * strain_rate[4]
        pstress[5] = viscosity * strain_rate[5]
        stateVars[np].estress = pressure
        stateVars[np].dvolumertic_strain = volumetric_strain
        return pstress


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), dvolumertic_strain: ti.types.ndarray(), rho: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].dvolumertic_strain = dvolumertic_strain[np]
        state_vars[np].rho = rho[np]