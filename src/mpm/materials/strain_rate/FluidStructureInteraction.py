import numpy as np
import taichi as ti

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.MaterialKernel import *
from src.utils.constants import ZEROVEC6f, DELTA
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import mat3x3


class FluidStructureInteraction(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_particle_num):
        super().__init__()
        self.matProps = FSIModel.field(shape=max_material_num)
        self.stateVars = ULStateVariable.field(shape=max_particle_num) 

    def get_state_vars_dict(self, start_particle, end_particle):
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        dvolumetric_strain = np.ascontiguousarray(self.stateVars.dvolumetric_strain.to_numpy()[start_particle:end_particle])
        return {'estress': estress, 'dvolumetric_strain': dvolumetric_strain}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        dvolumetric_strain = state_vars.item()['dvolumetric_strain']
        kernel_reload_state_variables(estress, dvolumetric_strain, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID')
        self.check_materialID(materialID, self.matProps.shape[0])
        
        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        is_structure = DictIO.GetEssential(material, 'IsStructure')
        if is_structure is False:
            density = DictIO.GetAlternative(material, 'Density', 1000)
            modulus = DictIO.GetAlternative(material, 'Modulus', 2e8)
            viscosity = DictIO.GetAlternative(material, 'Viscosity', 1e-3)
            self.matProps[materialID].add_fluid_material(density, modulus, viscosity)
            self.matProps[materialID].print_message(materialID)
        else:
            density = DictIO.GetAlternative(material, 'Density', 2650)
            young = DictIO.GetEssential(material, 'YoungModulus')
            possion = DictIO.GetAlternative(material, 'PossionRatio', 0.3)
            self.matProps[materialID].add_material(density, young, possion)
            self.matProps[materialID].print_message(materialID)

    def get_lateral_coefficient(self, materialID):
        return 1.


@ti.dataclass
class ULStateVariable:
    estress: float
    dvolumetric_strain: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.estress = MeanStress(stress)
        self.dvolumetric_strain = 1.

    @ti.func
    def _update_vars(self, stress):
        self.estress = MeanStress(stress)    


@ti.dataclass
class TLStateVariable:
    estress: float
    dvolumetric_strain: float
    deformation_gradient: mat3x3

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.estress = MeanStress(stress)
        self.dvolumetric_strain = 1.
        self.deformation_gradient = DELTA

    @ti.func
    def _update_deformation_gradient(self, deformation_gradient_rate, dt):
        self.deformation_gradient += deformation_gradient_rate * dt[None]

    @ti.func
    def _update_vars(self, stress):
        self.estress = MeanStress(stress)  

@ti.dataclass
class FSIModel:
    is_structure: ti.u8
    density: float
    modulus: float
    viscosity: float
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

    def add_fluid_material(self, density, modulus, viscosity):
        self.is_structure = 0
        self.density = density
        self.modulus = modulus
        self.viscosity = viscosity

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
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        delta_jacobian = 1. + dt[None] * (velocity_gradient[0, 0] + velocity_gradient[1, 1] + velocity_gradient[2, 2])
        stateVars[np].dvolumetric_strain *= delta_jacobian
        return delta_jacobian
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        delta_jacobian = 1. + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])
        stateVars[np].dvolumetric_strain *= delta_jacobian
        return delta_jacobian

    @ti.func
    def thermodynamic_pressure(self, np, stateVars):
        gamma = 1
        bulk_modulus = self.modulus
        dvolumertic = stateVars[np].dvolumetric_strain

        pressure = -bulk_modulus * (dvolumertic ** gamma - 1)
        return pressure
    
    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):
        updated_stress = previous_stress
        if int(self.is_structure) == 0:
            strain_rate = calculate_strain_rate2D(velocity_gradient)
            updated_stress = self.core1(np, strain_rate, stateVars)
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
            updated_stress = self.core1(np, strain_rate, stateVars)
        else:
            de = calculate_strain_increment(velocity_gradient, dt)
            dw = calculate_vorticity_increment(velocity_gradient, dt)
            updated_stress = self.core2(np, previous_stress, de, dw, stateVars)
        return updated_stress

    @ti.func
    def core1(self, np, strain_rate, stateVars):   
        viscosity = self.viscosity
        volumetric_strain_rate = strain_rate[0] + strain_rate[1] + strain_rate[2]
        old_pressure = stateVars[np].estress
        
        pressure = old_pressure + self.thermodynamic_pressure(np, stateVars)
        volumetric_component = -pressure - 2. * viscosity * volumetric_strain_rate / 3.
        
        pstress = ZEROVEC6f
        pstress[0] = volumetric_component + 2. * viscosity * strain_rate[0]
        pstress[1] = volumetric_component + 2. * viscosity * strain_rate[1]
        pstress[2] = volumetric_component + 2. * viscosity * strain_rate[2]
        pstress[3] = viscosity * strain_rate[3]
        pstress[4] = viscosity * strain_rate[4]
        pstress[5] = viscosity * strain_rate[5]
        
        updated_stress = pstress
        stateVars[np].estress = pressure
        return updated_stress
    
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
def kernel_reload_state_variables(estress: ti.types.ndarray(), dvolumetric_strain: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].dvolumetric_strain = dvolumetric_strain[np]