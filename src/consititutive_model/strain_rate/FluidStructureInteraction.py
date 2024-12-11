import taichi as ti

from src.consititutive_model.MaterialKernel import *
from src.utils.constants import ZEROVEC6f
from src.utils.VectorFunction import voigt_tensor_trace


@ti.dataclass
class ULStateVariable:
    rho: float
    estress: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.rho = matProps[int(particle[np].materialID)].density
        if int(matProps[int(particle[np].materialID)].is_structure) == 1:
            self.estress = -MeanStress(stress)
        elif int(matProps[int(particle[np].materialID)].is_structure) == 0:
            self.estress = MeanStress(stress)

    @ti.func
    def _update_vars(self, stress, np, particle, matProps):
        if int(matProps[int(particle[np].materialID)].is_structure) == 1:
            self.estress = -MeanStress(stress)
        elif int(matProps[int(particle[np].materialID)].is_structure) == 0:
            self.estress = VonMisesStress(stress)


@ti.dataclass
class FSIModel:
    is_structure: ti.u8
    density: float
    atmospheric_pressure: float
    modulus: float
    viscosity: float
    gamma: float
    young: float
    possion: float
    shear: float
    bulk: float

    def add_structure_material(self, density, young, possion):
        self.is_structure = 1
        self.density = density
        self.young = young
        self.possion = possion
        self.shear = 0.5 * self.young / (1. + self.possion)
        self.bulk = self.young / (3. * (1 - 2. * self.possion))

    def add_fluid_material(self, density, modulus, viscosity, gamma, atmospheric_pressure):
        self.is_structure = 0
        self.density = density
        self.modulus = modulus
        self.viscosity = viscosity
        self.gamma = gamma
        self.atmospheric_pressure = atmospheric_pressure

    def add_contact_parameter(self, friction, kn, kt):
        self.friction = friction
        self.kn = kn
        self.kt = kt

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        if self.is_structure == 0:
            print('Constitutive model = Fluid Model (Newtonian Model)')
            print("Model ID: ", materialID)
            print("Model density = ",  self.density)
            print('Bulk Modulus = ', self.modulus)
            print('Viscosity = ', self.viscosity, '\n')
        elif self.is_structure == 1:
            print('Constitutive model: Structure Model (Linear Elastic)')
            print("Model ID: ", materialID)
            print('Density: ', self.density)
            print('Young Modulus: ', self.young)
            print('Possion Ratio: ', self.possion, '\n')

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
        updated_stress = previous_stress
        if int(self.is_structure) == 0:
            strain_rate = calculate_strain_rate2D(velocity_gradient)
            updated_stress = self.core1(np, strain_rate, stateVars, dt)
        else:
            de = calculate_strain_increment2D(velocity_gradient, dt)
            dw = calculate_vorticity_increment2D(velocity_gradient, dt)
            updated_stress = self.core2(np, previous_stress, de, dw, stateVars)
        return updated_stress

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):
        updated_stress = previous_stress
        if int(self.is_structure) == 0:
            strain_rate = calculate_strain_rate(velocity_gradient)
            updated_stress = self.core1(np, strain_rate, stateVars, dt)
        else:
            de = calculate_strain_increment(velocity_gradient, dt)
            dw = calculate_vorticity_increment(velocity_gradient, dt)
            updated_stress = self.core2(np, previous_stress, de, dw, stateVars)
        return updated_stress
    
    @ti.func
    def fluid_pressure(self, np, stateVars, velocity_gradient, dt):
        strain_rate = calculate_strain_rate(velocity_gradient)
        volumetric_strain_rate = voigt_tensor_trace(strain_rate) 
        volumetric_strain_increment = volumetric_strain_rate * dt[None]
        stateVars[np].pressure = -stateVars[np].pressure + self.thermodynamic_pressure(stateVars[np].rho, volumetric_strain_increment)

    @ti.func
    def shear_stress(self, velocity_gradient):
        viscosity = self.viscosity
        strain_rate = calculate_strain_rate(velocity_gradient)
        volumetric_strain_rate = voigt_tensor_trace(strain_rate) 
        dila = viscosity * volumetric_strain_rate / 3.

        sstress = ZEROVEC6f
        sstress[0] = 2. * (viscosity * strain_rate[0] - dila)
        sstress[1] = 2. * (viscosity * strain_rate[1] - dila)
        sstress[2] = 2. * (viscosity * strain_rate[2] - dila)
        sstress[3] = viscosity * strain_rate[3]
        sstress[4] = viscosity * strain_rate[4]
        sstress[5] = viscosity * strain_rate[5]
        return sstress
    
    @ti.func
    def correct_pressure(self, pressure_bar):
        return -pressure_bar * EYE

    @ti.func
    def core1(self, np, strain_rate, stateVars, dt):   
        viscosity = self.viscosity
        volumetric_strain_rate = voigt_tensor_trace(strain_rate)
        
        pressure = -stateVars[np].pressure + self.thermodynamic_pressure(stateVars[np].rho, volumetric_strain_rate * dt[None])
        volumetric_component = -pressure - 2. * viscosity * volumetric_strain_rate / 3.
        
        pstress = ZEROVEC6f
        pstress[0] = volumetric_component + 2. * viscosity * strain_rate[0]
        pstress[1] = volumetric_component + 2. * viscosity * strain_rate[1]
        pstress[2] = volumetric_component + 2. * viscosity * strain_rate[2]
        pstress[3] = viscosity * strain_rate[3]
        pstress[4] = viscosity * strain_rate[4]
        pstress[5] = viscosity * strain_rate[5]
        stateVars[np].estress = -pressure
        return pstress
    
    @ti.func
    def core2(self, np, previous_stress, de, dw, stateVars):
        # !-- trial elastic stresses ----!
        bulk_modulus = self.bulk
        shear_modulus = self.shear

        # !-- trial elastic stresses ----!
        stress = previous_stress
        sigrot = Sigrot(stress, dw)
        dstress = ElasticTensorMultiplyVector(de, bulk_modulus, shear_modulus)
        trial_stress = stress + dstress 
        updated_stress = trial_stress

        updated_stress += sigrot
        stateVars[np].estress = VonMisesStress(updated_stress)
        return updated_stress


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), p0: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].p0 = p0[np]