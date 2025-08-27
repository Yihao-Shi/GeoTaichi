import taichi as ti

from src.physics_model.consititutive_model.MaterialModel import MaterialModel


@ti.data_oriented
class UserDefined(MaterialModel):
    def __init__(self, material_type, configuration):
        super().__init__(material_type, configuration)
        self.density = 0.

    def add_material(self, density):
        self.density = density
        
    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Rigid Body')
        print("Model ID: ", materialID)
        print('Density: ', self.density, '\n')

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        pass
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        pass

    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        pass

    @ti.func
    def ComputeStress(self, np, stateVars, particle, dt):
        pass

    @ti.func
    def core(self, np, previous_stress, de, dw, stateVars): 
        pass

    @ti.func
    def _get_sound_speed(self):
        return 0.