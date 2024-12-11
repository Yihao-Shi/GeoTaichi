import taichi as ti

from src.consititutive_model.MaterialKernel import *
from src.utils.constants import ZEROVEC6f, EYE
from src.utils.VectorFunction import voigt_tensor_trace


@ti.dataclass
class ULStateVariable:
    rho: float
    pressure: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.rho = matProps[int(particle[np].materialID)].density
        self.pressure = MeanStress(stress)

    @ti.func
    def _update_vars(self, stress):
        self.pressure = MeanStress(stress)    


@ti.dataclass
class NewtonianModel:
    density: float
    atmospheric_pressure: float
    modulus: float
    viscosity: float
    element_length: float
    cl: float
    cq: float

    def add_material(self, density, modulus, viscosity, element_length, cl, cq, atmospheric_pressure):
        self.density = density
        self.modulus = modulus
        self.viscosity = viscosity
        self.element_length = element_length
        self.cl = cl
        self.cq = cq
        self.atmospheric_pressure = atmospheric_pressure

    def add_contact_parameter(self, friction, kn, kt):
        self.friction = friction
        self.kn = kn
        self.kt = kt

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model = Newtonian Model')
        print("Model ID: ", materialID)
        print("Model density = ",  self.density)
        print('Bulk Modulus = ', self.modulus)
        print('Viscosity = ', self.viscosity)
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
        delta_jacobian = (1. + dt[None] * (velocity_gradient[0, 0] + velocity_gradient[1, 1] + velocity_gradient[2, 2]))
        stateVars[np].rho /= delta_jacobian
        return delta_jacobian
    
    @ti.func
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        delta_jacobian = 1. + dt[None] * (velocity_gradient[0, 0] + velocity_gradient[1, 1])
        stateVars[np].rho /= delta_jacobian
        return delta_jacobian
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        delta_jacobian = 1. + dt[None] * voigt_tensor_trace(strain_rate)
        stateVars[np].rho /= delta_jacobian
        return delta_jacobian

    @ti.func
    def thermodynamic_pressure(self, rho, volumertic_strain):
        pressure = -rho * self.modulus / self.density * volumertic_strain
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
    def ComputePressure2D(self, np, stateVars, velocity_gradient, dt):
        strain_rate = calculate_strain_rate2D(velocity_gradient)
        return self.fluid_pressure(np, stateVars, strain_rate, dt)

    @ti.func
    def ComputePressure(self, np, stateVars, velocity_gradient, dt):
        strain_rate = calculate_strain_rate(velocity_gradient)
        return self.fluid_pressure(np, stateVars, strain_rate, dt)
    
    @ti.func
    def fluid_pressure(self, np, stateVars, strain_rate, dt):
        volumetric_strain_rate = voigt_tensor_trace(strain_rate) 
        volumetric_strain_increment = volumetric_strain_rate * dt[None]
        pressure = -stateVars[np].pressure + self.thermodynamic_pressure(stateVars[np].rho, volumetric_strain_increment)
        artifical_pressure = self.artifical_viscosity(np, volumetric_strain_rate, stateVars)
        pressureAV = pressure + artifical_pressure
        stateVars[np].pressure = -pressure
        return pressureAV
    
    @ti.func
    def ComputeShearStress2D(self, velocity_gradient):
        strain_rate = calculate_strain_rate2D(velocity_gradient)
        return self.shear_stress(strain_rate)

    @ti.func
    def ComputeShearStress(self, velocity_gradient):
        strain_rate = calculate_strain_rate(velocity_gradient)
        return self.shear_stress(strain_rate)
    
    @ti.func
    def shear_stress(self, strain_rate):
        viscosity = self.viscosity
        mean_volumetric_strain_rate = voigt_tensor_trace(strain_rate) / 3.

        sstress = ZEROVEC6f
        sstress[0] = 2. * viscosity * (strain_rate[0] - mean_volumetric_strain_rate)
        sstress[1] = 2. * viscosity * (strain_rate[1] - mean_volumetric_strain_rate)
        sstress[2] = 2. * viscosity * (strain_rate[2] - mean_volumetric_strain_rate)
        sstress[3] = viscosity * strain_rate[3]
        sstress[4] = viscosity * strain_rate[4]
        sstress[5] = viscosity * strain_rate[5]
        return sstress

    @ti.func
    def core(self, np, strain_rate, stateVars, dt): 
        pressureAV = self.fluid_pressure(np, stateVars, strain_rate, dt)
        shear_stress = self.shear_stress(strain_rate)
        return -pressureAV * EYE + shear_stress


@ti.kernel
def kernel_reload_state_variables(rho: ti.types.ndarray(), pressure: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(rho.shape[0]):
        state_vars[np].rho = rho[np]
        state_vars[np].pressure = pressure[np]