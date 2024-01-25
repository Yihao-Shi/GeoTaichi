import numpy as np
import taichi as ti

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.MaterialKernel import *
from src.utils.constants import DELTA, PI
from src.utils.MatrixFunction import matrix_form
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import mat3x3
from src.utils.VectorFunction import voigt_form


class SoilStructureInteraction(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_particle_num, configuration="ULMPM", solver_type="Explicit"):
        super().__init__()
        self.matProps = SSIModel.field(shape=max_material_num)
        if configuration == "ULMPM":
            self.stateVars = ULStateVariable.field(shape=max_particle_num) 
        elif configuration == "TLMPM":
            self.stateVars = TLStateVariable.field(shape=max_particle_num) 

        if solver_type == "Implicit":
            self.stiffness_matrix = ti.Matrix.field(6, 6, float, shape=max_particle_num)

    def get_state_vars_dict(self, start_particle, end_particle):
        epstrain = np.ascontiguousarray(self.stateVars.epstrain.to_numpy()[start_particle:end_particle])
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        return {'epstrain': epstrain, 'estress': estress}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        epstrain = state_vars.item()['epstrain']
        kernel_reload_state_variables(estress, epstrain, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID') 
        self.check_materialID(materialID, self.matProps.shape[0])
        
        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        is_structure = DictIO.GetEssential(material, 'IsStructure')
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        possion = DictIO.GetAlternative(material, 'PossionRatio', 0.3)
        if is_structure is False:
            c = DictIO.GetAlternative(material, 'Cohesion', 0.)
            fai = DictIO.GetAlternative(material, 'Friction', 0.) * PI / 180.
            psi = DictIO.GetAlternative(material, 'Dilation', 0.) * PI / 180.
            tensile = DictIO.GetAlternative(material, 'Tensile', 0.)
            dpType = DictIO.GetAlternative(material, 'dpType', "Inscribed")
            self.matProps[materialID].add_soil_material(density, young, possion, c, fai, psi, tensile, dpType)
            self.matProps[materialID].print_message(materialID)
        else:
            self.matProps[materialID].add_structure_material(density, young, possion)
            self.matProps[materialID].print_message(materialID)
    
    def get_lateral_coefficient(self, materialID):
        mu = self.matProps[materialID].possion
        return mu / (1. - mu)


@ti.dataclass
class ULStateVariable:
    epstrain: float
    estress: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.estress = VonMisesStress(stress)

    @ti.func
    def _update_vars(self, stress, epstrain):
        self.estress = VonMisesStress(stress)
        self.epstrain = epstrain

@ti.dataclass
class TLStateVariable:
    epstrain: float
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
    def _update_vars(self, stress, epstrain):
        self.estress = VonMisesStress(stress)
        self.epstrain = epstrain


@ti.dataclass
class SSIModel:
    is_structure: ti.u8
    density: float
    young: float
    possion: float
    shear: float
    bulk: float
    c: float
    fai: float
    psi: float
    q_fai: float
    k_fai: float
    q_psi: float
    tensile: float

    def add_structure_material(self, density, young, possion):
        self.is_structure = 1
        self.density = density
        self.young = young
        self.possion = possion
        self.shear = 0.5 * self.young / (1. + self.possion)
        self.bulk = self.young / (3. * (1 - 2. * self.possion))
    
    def add_soil_material(self, density, young, possion, c, fai, psi, tensile, dpType):
        self.is_structure = 0
        self.density = density
        self.young = young
        self.possion = possion

        self.shear = 0.5 * self.young / (1. + self.possion)
        self.bulk = self.young / (3. * (1 - 2. * self.possion))
        self.c = c
        self.fai = fai
        self.psi = psi
        
        if dpType == "Circumscribed":
            self.q_fai = 6. * ti.sin(self.fai) / ti.sqrt(3) * (3 + ti.sin(self.fai))
            self.k_fai = 6. * ti.cos(self.fai) * self.c / ti.sqrt(3) * (3 + ti.sin(self.fai))
            self.q_psi = 6. * ti.sin(self.psi) / ti.sqrt(3) * (3 + ti.sin(self.psi))
        elif dpType == "MiddleCircumscribed":
            self.q_fai = 6. * ti.sin(self.fai) / ti.sqrt(3) * (3 - ti.sin(self.fai))
            self.k_fai = 6. * ti.cos(self.fai) * self.c / ti.sqrt(3) * (3 - ti.sin(self.fai))
            self.q_psi = 6. * ti.sin(self.psi) / ti.sqrt(3) * (3 - ti.sin(self.psi))
        elif dpType == "Inscribed":
            self.q_fai = 3. * ti.tan(self.fai) / ti.sqrt(9. + 12 * ti.tan(self.fai) ** 2)
            self.k_fai = 3. * self.c / ti.sqrt(9. + 12 * ti.tan(self.fai) ** 2)
            self.q_psi = 3. * ti.tan(self.psi) / ti.sqrt(9. + 12 * ti.tan(self.psi) ** 2)

        if self.fai == 0:
            self.tensile = 0.
        else:
            self.tensile = ti.min(tensile, self.k_fai / self.q_fai)

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        if self.is_structure == 0:
            print('Constitutive model: Soil Model (Drucker-Prager Model)')
        elif self.is_structure == 1:
            print('Constitutive model: Structure Model (Linear Elastic)')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Possion Ratio: ', self.possion)
        if self.is_structure == 0:
            print('Cohesion Coefficient = ', self.c)
            print('Angle of Internal Friction (in radian) = ', self.fai)
            print('Angle of Dilatation (in radian) = ', self.psi)
            print('Tensile = ', self.tensile, '\n')
        else:
            print('\n')

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
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        return 1. + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])
    
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
        updated_stress = previous_stress
        if int(self.is_structure) == 0:
            de = calculate_strain_increment2D(velocity_gradient, dt)
            dw = calculate_vorticity_increment2D(velocity_gradient, dt)
            updated_stress = self.core1(np, previous_stress, de, dw, stateVars)
        else:
            de = calculate_strain_increment2D(velocity_gradient, dt)
            dw = calculate_vorticity_increment2D(velocity_gradient, dt)
            updated_stress = self.core2(np, previous_stress, de, dw, stateVars)
        return updated_stress

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):
        updated_stress = previous_stress
        if int(self.is_structure) == 0:
            de = calculate_strain_increment(velocity_gradient, dt)
            dw = calculate_vorticity_increment(velocity_gradient, dt)
            updated_stress = self.core1(np, previous_stress, de, dw, stateVars)
        else:
            de = calculate_strain_increment(velocity_gradient, dt)
            dw = calculate_vorticity_increment(velocity_gradient, dt)
            updated_stress = self.core2(np, previous_stress, de, dw, stateVars)
        return updated_stress

    @ti.func
    def core1(self, np, previous_stress, de, dw, stateVars):   
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

        q_fai = self.q_fai
        k_fai = self.k_fai
        q_psi = self.q_psi
        tensile = self.tensile
        
        sigma = MeanStress(trial_stress)
        sd = DeviatoricStress(trial_stress)
        seqv = EquivalentStress(trial_stress)
        J2sqrt = ti.sqrt(ComputeInvariantJ2(trial_stress))

        dpFi = J2sqrt + q_fai * sigma - k_fai
        dpsig = sigma - tensile
        dpdstrain = 0.
        if dpsig < 0.:
            if dpFi > 0.:
                dlamd = dpFi / (shear_modulus + bulk_modulus * q_fai * q_psi)
                sigma -= bulk_modulus * q_psi * dlamd
                ratio = (k_fai - q_fai * sigma) / J2sqrt

                sd *= ratio
                seqv *= ratio
                updated_stress = AssembleStress(sigma, sd)
                dpdstrain += dlamd * ti.sqrt(1./3. + (2./9.) * q_psi ** 2)
        else:
            alphap = ti.sqrt(1 + q_fai ** 2) - q_fai
            J2sqrtp = k_fai - q_fai - tensile
            dp_hfai = J2sqrt - J2sqrtp - alphap * dpsig

            if dp_hfai > 0.:
                dlamd = dpFi / (shear_modulus + bulk_modulus * q_fai * q_psi)
                sigma -=bulk_modulus * q_psi * dlamd
                ratio = (k_fai - q_fai * sigma) / J2sqrt
                sd *= ratio
                seqv *= ratio
                updated_stress = AssembleStress(sigma, sd)
                dpdstrain += dlamd * ti.sqrt(1./3. + (2./9.) * q_psi ** 2)
            else:
                dlamd = (sigma - tensile) / bulk_modulus
                for d in ti.static(range(3)):
                    updated_stress[d] += tensile - sigma
                dpdstrain += dlamd * (1./3.) * ti.sqrt(2)
        stateVars[np].epstrain += dpdstrain
        
        updated_stress += sigrot
        stateVars[np].estress = VonMisesStress(updated_stress)
        return updated_stress

    @ti.func
    def ComputePKStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        previous_stress = self.PK2CauchyStress(np, stateVars, previous_stress)
        stress = self.ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
        return self.Cauchy2PKStress(np, stateVars, stress)
    
    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stiffness, stateVars):
        ComputeElasticStiffnessTensor(np, self.bulk, self.shear, stiffness)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stiffness, stateVars):
        pass


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), epstrain: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].epstrain = epstrain[np]