import taichi as ti

from src.utils.constants import DELTA
from src.utils.TypeDefination import mat3x3


@ti.dataclass
class ULStateVariable:
    # TODO: add essential state variable for constitutive model
    estress: float

    @ti.func
    def _initialize_vars(self, stress):
        pass

    @ti.func
    def _update_vars(self, stress, epstrain):
        pass    


@ti.dataclass
class TLStateVariable:
    # TODO: add essential state variable for constitutive model
    estress: float
    deformation_gradient: mat3x3

    @ti.func
    def _initialize_vars(self, stress):
        self.deformation_gradient = DELTA
        pass

    @ti.func
    def _update_deformation_gradient(self, deformation_gradient_rate, dt):
        self.deformation_gradient += deformation_gradient_rate * dt[None]

    @ti.func
    def _update_vars(self, stress, epstrain):
        pass  


@ti.dataclass
class MaterialModel:
    density: float
    # TODO: Add essential material properties

    def add_material(self, density):
        self.density = density
        # TODO: Add essential material properties

    def add_contact_parameter(self, friction, kn, kt):
        self.friction = friction
        self.kn = kn
        self.kt = kt

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model = Material Model')
        print("Model ID: ", materialID)
        print("Model density = ",  self.density)

    @ti.func
    def _get_sound_speed(self):
        sound_speed = 0.
        if self.density > 0.:
            pass
        return sound_speed
        # TODO: Add proporiate equations of sound speed

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        return (DELTA + velocity_gradient * dt[None]).determinant()
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        return 1. + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])

    @ti.func
    def ComputeStress(self,
                      np,                                                      # particle id 
                      stateVars,                                               # state variables
                      particle,                                                # particle pointer
                      dt                                                       # time step
                     ):                  
        pass


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]

