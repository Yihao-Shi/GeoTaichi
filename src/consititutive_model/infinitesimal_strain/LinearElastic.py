import taichi as ti

from src.consititutive_model.MaterialKernel import *
from src.utils.constants import DELTA2D, DELTA
from src.utils.MatrixFunction import matrix_form
from src.utils.TypeDefination import mat3x3
from src.utils.VectorFunction import voigt_form


@ti.dataclass
class ULStateVariable:
    estress: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.estress = VonMisesStress(stress)

    @ti.func
    def _update_vars(self, stress):
        self.estress = VonMisesStress(stress)

@ti.dataclass
class TLStateVariable:
    estress: float
    deformation_gradient: mat3x3
    stress: mat3x3

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.estress = VonMisesStress(stress)
        self.deformation_gradient = DELTA
        self.stress = matrix_form(stress)

    @ti.func
    def _update_deformation_gradient(self, deformation_gradient_rate, dt):
        self.deformation_gradient += deformation_gradient_rate * dt[None]

    @ti.func
    def _update_vars(self, stress):
        self.estress = VonMisesStress(stress)


@ti.dataclass
class LinearElasticModel:
    density: float
    young: float
    possion: float
    shear: float
    bulk: float

    def add_material(self, density, young, possion):
        self.density = density
        self.young = young
        self.possion = possion

        self.shear = 0.5 * self.young / (1. + self.possion)
        self.bulk = self.young / (3. * (1 - 2. * self.possion))

    def add_coupling_material(self, porosity, solid_density, fluid_density, young, possion, fluid_bulk, permeability):
        self.density = fluid_density * porosity + solid_density * ( 1.- porosity)
        self.solid_density = solid_density
        self.fluid_density = fluid_density
        self.porosity = porosity
        self.young = young
        self.possion = possion
        self.fluid_bulk = fluid_bulk
        self.permeability = permeability

        self.shear = 0.5 * self.young / (1. + self.possion)
        self.bulk = self.young / (3. * (1 - 2. * self.possion))

    def add_contact_parameter(self, friction, kn, kt):
        self.friction = friction
        self.kn = kn
        self.kt = kt
        
    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Elastic Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Possion Ratio: ', self.possion, '\n')

    @ti.func
    def _get_sound_speed(self):
        sound_speed = 0.
        if self.density > 0.:
            sound_speed = ti.sqrt(self.young * (1 - self.possion) / (1 + self.possion) / (1 - 2 * self.possion) / self.density)
        return sound_speed
    
    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        return (DELTA + velocity_gradient * dt[None]).determinant()
    
    @ti.func
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        return (DELTA2D + velocity_gradient * dt[None]).determinant()
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        return 1. + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])
    
    @ti.func    # two phase
    def update_particle_porosity(self, np, velocity_gradient, stateVars, particle, dt):
        particle[np].porosity = 1.0 - (1.0 - particle[np].porosity) / (DELTA + velocity_gradient * dt[None]).determinant()

    @ti.func    # two phase
    def ComputePressure(self, np, velocity_gradients, velocity_gradientf, stateVars, particle, dt):
        vs = ((DELTA + velocity_gradients * dt[None]).determinant() - 1.0)
        vf = ((DELTA + velocity_gradientf * dt[None]).determinant() - 1.0)
        Kf = self.fluid_bulk
        particle[np].pressure -= Kf/particle[np].porosity * ( (1.0-particle[np].porosity)*vs + particle[np].porosity * vf)

    @ti.func    # two phase
    def update_particle_porosity_2D(self, np, velocity_gradient, stateVars, particle, dt):
        particle[np].porosity = 1.0 - (1.0 - particle[np].porosity) / (DELTA2D + velocity_gradient * dt[None]).determinant()

    @ti.func    # two phase
    def ComputePressure2D(self, np, velocity_gradients, velocity_gradientf, stateVars, particle, dt):
        vs = ((DELTA2D + velocity_gradients * dt[None]).determinant() - 1.0)
        vf = ((DELTA2D + velocity_gradientf * dt[None]).determinant() - 1.0)
        Kf = self.fluid_bulk
        particle[np].pressure -= Kf/particle[np].porosity * ( (1.0-particle[np].porosity)*vs + particle[np].porosity * vf)

    @ti.func    # two phase
    def update_particle_massf(self, np, stateVars, particle):
        particle[np].mf = particle[np].vol * particle[np].porosity * self.fluid_density

    @ti.func
    def PK2CauchyStress(self, np, stateVars):
        inv_j = 1. / stateVars[np].deformation_gradient.determinant()
        return voigt_form(stateVars[np].stress @ stateVars[np].deformation_gradient.transpose() * inv_j)

    @ti.func
    def Cauchy2PKStress(self, np, stateVars, stress):
        j = stateVars[np].deformation_gradient.determinant()
        return matrix_form(stress) @ stateVars[np].deformation_gradient.inverse().transpose() * j
    
    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        de = calculate_strain_increment2D(velocity_gradient, dt)
        dw = calculate_vorticity_increment2D(velocity_gradient, dt)
        return self.core(np, previous_stress, de, dw, stateVars)

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        de = calculate_strain_increment(velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)
        return self.core(np, previous_stress, de, dw, stateVars)
    
    @ti.func
    def core(self, np, previous_stress, de, dw, stateVars): 
        shear_modulus, bulk_modulus = self.shear, self.bulk
        stress = previous_stress

        sigrot = Sigrot(stress, dw)
        dstress = ElasticTensorMultiplyVector(de, bulk_modulus, shear_modulus)
        
        stress += dstress + sigrot
        stateVars[np].estress = VonMisesStress(stress)
        return stress

    @ti.func
    def ComputePKStress(self, np, velocity_gradient, stateVars, dt):  
        previous_stress = self.PK2CauchyStress(np, stateVars)
        cauchy_stress = self.ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
        PKstress = self.Cauchy2PKStress(np, stateVars, cauchy_stress)
        stateVars[np].stress = PKstress
        return PKstress
    
    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        return ComputeElasticStiffnessTensor(self.bulk, self.shear)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stateVars):
        return ComputeElasticStiffnessTensor(self.bulk, self.shear)


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]