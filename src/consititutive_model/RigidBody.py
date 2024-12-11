import taichi as ti

from src.utils.constants import DELTA
from src.utils.TypeDefination import mat3x3


@ti.dataclass
class ULStateVariable:
    estress: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        self.estress = 0.

    @ti.func
    def _update_vars(self, stress):
        self.estress = 0.


@ti.dataclass
class TLStateVariable:
    estress: float
    deformation_gradient: mat3x3

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        self.estress = 0.
        self.deformation_gradient = DELTA

    @ti.func
    def _update_deformation_gradient(self, deformation_gradient_rate, dt):
        self.deformation_gradient += deformation_gradient_rate * dt[None]

    @ti.func
    def _update_vars(self, stress):
        self.estress = 0.

    
@ti.dataclass
class Rigid:
    density: float

    def add_material(self, density):
        self.density = density

    def add_contact_parameter(self, friction, kn, kt):
        self.friction = 100.
        self.kn = kn
        self.kt = kt
        
    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Rigid Body')
        print("Model ID: ", materialID)
        print('Density: ', self.density, '\n')

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, particle, dt):
        pass
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, particle, dt):
        pass

    @ti.func
    def ComputeStress(self, np, stateVars, particle, dt):
        pass

    @ti.func
    def _get_sound_speed(self):
        return 0.

@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]