import taichi as ti

from src.utils.constants import PI


@ti.data_oriented
class VirtualMassModel:
    def __init__(self):
        pass

    @ti.func
    def coefficient(self, particle_radius, relative_velocity, relative_accerlate):
        Ac = relative_velocity.dot(relative_velocity) / (relative_accerlate.norm() * 2. * particle_radius)
        Cvm = 2.1 - 0.132 / (0.12 + Ac * Ac)
        return Cvm

    @ti.func
    def virtual_mass_force(self, fluid_density, particle_radius, relative_velocity, relative_accerlate):
        particle_volume = 4./3. * PI * particle_radius * particle_radius * particle_radius
        Cvm = self.coefficient(particle_radius, relative_velocity, relative_accerlate)
        virtual_mass_force = 0.5 * Cvm * fluid_density * particle_volume * relative_accerlate
        return virtual_mass_force