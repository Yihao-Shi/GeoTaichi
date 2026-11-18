import taichi as ti

from src.physics_model.constitutive_model.MaterialModel import MaterialModel
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class RigidModel(MaterialModel):
    def __init__(self, material_type="Solid", configuration="UL", solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.density = 0.0

    def add_material(self, density):
        self.density = density

    def model_initialize(self, material):
        density = DictIO.get_essential(material, "Density")
        self.add_material(density)

    def print_message(self, materialID):
        self.print_console_header()
        print("Constitutive model: Rigid Body")
        print("Material ID: ", materialID)
        print("Density: ", self.density, "\n")

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, state_variables, dt):
        pass

    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, state_variables, dt):
        pass

    @ti.func
    def compute_stress2d(self, np, previous_stress, velocity_gradient, state_variables, dt):
        pass

    @ti.func
    def compute_stress(self, np, state_variables, particle, dt):
        pass

    @ti.func
    def core(self, np, previous_stress, de, dw, state_variables):
        pass

    @ti.func
    def _get_sound_speed(self):
        return 0.0
