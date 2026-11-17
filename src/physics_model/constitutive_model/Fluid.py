from src.utils.errors import SimulationStateError
from src.physics_model.constitutive_model.MaterialModel import MaterialModel

import taichi as ti
import numpy as np
from src.physics_model.constitutive_model.infinitesimal_strain.MaterialKernel import *
from src.utils.MatrixFunction import trace
from src.utils.VectorFunction import voigt_tensor_trace


@ti.data_oriented
class Fluid(MaterialModel):
    def __init__(self, material_type, configuration, solver_type):
        super().__init__(material_type, configuration, solver_type)
        self.atmospheric_pressure = 0.0
        self.surface_tension = 0.0
        self.modulus = None if self.incompressible_fluid else 0.0
        self.viscosity = 0.0
        self.element_length = 0.0
        self.cl = 0.0
        self.cq = 0.0
        self.gamma = 0.0
        if "TL" in self.configuration:
            raise SimulationStateError("Fluid materials do not support total Lagrangian simulation")

    def add_material(self, *args, **kwargs):
        pass

    def initialize_coupling(self):
        pass

    def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        return np.repeat(1.0, end_index - start_index)

    def get_state_vars(self):
        self._initialize_vars = self._initialize_vars
        return self.define_state_vars()

    def get_sound_speed(self, density, modulus):
        return np.where(density > 0, np.sqrt(modulus / density), 0)

    @ti.func
    def _initialize_vars(np, particle, state_variables):
        raise NotImplementedError

    @ti.func
    def _set_modulus(self, velocity):
        velocity = 1000 * velocity
        ti.atomic_max(self.modulus, self.density * velocity / self.gamma)

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, state_variables, dt):
        delta_jacobian = 1.0 + dt[None] * trace(velocity_gradient)
        state_variables[np].rho /= delta_jacobian
        return delta_jacobian

    @ti.func
    def update_particle_volume_2d(self, np, velocity_gradient, state_variables, dt):
        return self.update_particle_volume(np, velocity_gradient, state_variables, dt)

    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, state_variables, dt):
        delta_jacobian = 1.0 + dt[None] * voigt_tensor_trace(strain_rate)
        state_variables[np].rho /= delta_jacobian
        return delta_jacobian

    @ti.func
    def thermodynamic_pressure(self, rho, volumertic_strain):
        pressure = -rho * self.modulus / self.density * volumertic_strain
        return pressure

    @ti.func
    def artifical_viscosity(self, np, volumetric_strain_rate, state_variables):
        # VonNeumann J. 1950, A method for the numerical calculation of hydrodynamic shocks. J. Appl. Phys.
        q = 0.0
        if volumetric_strain_rate < 0.0:
            q = (
                -state_variables[np].rho * self.cl * self.element_length * volumetric_strain_rate
                + state_variables[np].rho
                * self.cq
                * self.element_length
                * self.element_length
                * volumetric_strain_rate
                * volumetric_strain_rate
            )
        return q

    @ti.func
    def fluid_pressure(self, np, state_variables, strain_rate, dt):
        volumetric_strain_rate = voigt_tensor_trace(strain_rate)
        volumetric_strain_increment = volumetric_strain_rate * dt[None]
        pressure = ti.max(
            0.0,
            -state_variables[np].pressure
            + self.thermodynamic_pressure(state_variables[np].rho, volumetric_strain_increment),
        )
        artifical_pressure = self.artifical_viscosity(np, volumetric_strain_rate, state_variables)
        pressure_av = pressure + artifical_pressure
        state_variables[np].pressure = -pressure
        return pressure_av

    @ti.func
    def compute_stress2d(self, np, previous_stress, velocity_gradient, state_variables, dt):
        strain_rate = calculate_strain_rate2d(velocity_gradient)
        return self.core(np, strain_rate, state_variables, dt)

    @ti.func
    def compute_stress(self, np, previous_stress, velocity_gradient, state_variables, dt):
        strain_rate = calculate_strain_rate(velocity_gradient)
        return self.core(np, strain_rate, state_variables, dt)

    @ti.func
    def compute_pressure2d(self, np, state_variables, velocity_gradient, dt):
        strain_rate = calculate_strain_rate2d(velocity_gradient)
        return self.fluid_pressure(np, state_variables, strain_rate, dt)

    @ti.func
    def compute_pressure(self, np, state_variables, velocity_gradient, dt):
        strain_rate = calculate_strain_rate(velocity_gradient)
        return self.fluid_pressure(np, state_variables, strain_rate, dt)

    @ti.func
    def compute_shear_stress2d(self, velocity_gradient):
        strain_rate = calculate_strain_rate2d(velocity_gradient)
        return self.shear_stress(strain_rate)

    @ti.func
    def compute_shear_stress(self, velocity_gradient):
        strain_rate = calculate_strain_rate(velocity_gradient)
        return self.shear_stress(strain_rate)

    @ti.func
    def shear_stress(self, strain_rate):
        raise NotImplementedError

    @ti.func
    def core(self, np, strain_rate, state_variables, dt):
        raise NotImplementedError
