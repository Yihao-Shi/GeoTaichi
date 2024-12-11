import taichi as ti

from src.consititutive_model.MaterialKernel import *
from src.utils.constants import ZEROVEC6f, EYE, Threshold
from src.utils.VectorFunction import voigt_tensor_trace


@ti.dataclass
class ULStateVariable:
    pressure: float
    rho: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.pressure = MeanStress(stress)
        self.rho = matProps[int(particle[np].materialID)].density

    @ti.func
    def _update_vars(self, stress):
        self.pressure = MeanStress(stress)    


@ti.dataclass
class BinghamModel:
    density: float
    atmospheric_pressure: float
    modulus: float
    viscosity: float
    _yield: float
    critical_rate: float
    gamma: float

    def add_material(self, density, modulus, viscosity, _yield, critical_rate, gamma, atmospheric_pressure):
        self.density = density
        self.modulus = modulus
        self.viscosity = viscosity
        self._yield = _yield
        self.critical_rate = critical_rate
        self.gamma = gamma
        self.atmospheric_pressure = atmospheric_pressure

    def add_contact_parameter(self, friction, kn, kt):
        self.friction = friction
        self.kn = kn
        self.kt = kt

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
        stateVars[np].rho /= delta_jacobian
        return delta_jacobian
    
    @ti.func
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        delta_jacobian = 1. + dt[None] * (velocity_gradient[0, 0] + velocity_gradient[1, 1])
        stateVars[np].rho /= delta_jacobian
        return delta_jacobian
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        delta_jacobian = 1. + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])
        stateVars[np].rho /= delta_jacobian
        return delta_jacobian

    @ti.func
    def thermodynamic_pressure(self, rho, volumertic_strain):
        pressure = -rho * self.modulus / self.density * volumertic_strain
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
    def fluid_pressure(self, np, stateVars, velocity_gradient, dt):
        strain_rate = calculate_strain_rate(velocity_gradient)
        volumetric_strain_rate = voigt_tensor_trace(strain_rate) 
        volumetric_strain_increment = volumetric_strain_rate * dt[None]
        pressure = -stateVars[np].pressure + self.thermodynamic_pressure(stateVars[np].rho, volumetric_strain_increment)
        artifical_pressure = self.artifical_viscosity(np, volumetric_strain_rate, stateVars)
        pressureAV = pressure + artifical_pressure
        stateVars[np].pressure = -pressure
        return pressureAV
    
    @ti.func
    def shear_stress(self, velocity_gradient):
        _yield = self._yield
        viscosity = self.viscosity
        critical_shear_rate = self.critical_rate
        strain_rate = calculate_strain_rate(velocity_gradient)

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
        return tau
    
    @ti.func
    def correct_pressure(self, pressure_bar):
        return -pressure_bar * EYE

    @ti.func
    def core(self, np, strain_rate, stateVars, dt):   
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

        volumetric_strain_rate = voigt_tensor_trace(strain_rate)
        pressure = -stateVars[np].pressure + self.thermodynamic_pressure(stateVars[np].rho, volumetric_strain_rate * dt[None])
        updated_stress = -pressure * EYE + tau 
        stateVars[np].pressure = -pressure
        return updated_stress


@ti.kernel
def kernel_reload_state_variables(pressure: ti.types.ndarray(), rho: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(pressure.shape[0]):
        state_vars[np].pressure = pressure[np]
        state_vars[np].rho = rho[np]