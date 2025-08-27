import taichi as ti

from src.physics_model.consititutive_model.MaterialKernel import SphericalTensor
from src.physics_model.consititutive_model.MaterialModel import Fluid
from src.utils.constants import ZEROVEC6f, EYE
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_trace


@ti.data_oriented
class NewtonianModel(Fluid):
    def __init__(self, material_type="Fluid", configuration="UL", solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)

    def model_initialize(self, material):
        density = DictIO.GetAlternative(material, 'Density', 1000)
        modulus = DictIO.GetAlternative(material, 'Modulus', 3.6e5)
        viscosity = DictIO.GetAlternative(material, 'Viscosity', 1e-3)
        cl = DictIO.GetAlternative(material, 'cL', 1.)
        cq = DictIO.GetAlternative(material, 'cQ', 2.)
        element_length = DictIO.GetAlternative(material, 'ElementLength', 0.)
        atmospheric_pressure = DictIO.GetAlternative(material, 'atmospheric_pressure', 0.)
        self.add_material(density, modulus, viscosity, element_length, cl, cq, atmospheric_pressure)

    def add_material(self, density, modulus, viscosity, element_length, cl, cq, atmospheric_pressure, gamma=1.):
        self.density = density
        self.modulus = modulus
        self.viscosity = viscosity
        self.element_length = element_length
        self.cl = cl
        self.cq = cq
        self.atmospheric_pressure = atmospheric_pressure
        self.gamma = gamma
        self.max_sound_speed = self.get_sound_speed(self.density, self.modulus)

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model = Newtonian Model')
        print("Model ID: ", materialID)
        print("Model density = ",  self.density)
        print('Bulk Modulus = ', self.modulus)
        print('Viscosity = ', self.viscosity)
        print("Characteristic Element Length = ", self.element_length)
        print("Artifical Viscosity Parameter = ", self.cl, self.cq, '\n')

    def define_state_vars(self):
        if self.solver_type == 0:
            return {'pressure': float, 'rho': float}
        elif self.solver_type == 1:
            return {}
    
    @ti.func
    def _initialize_vars_(self, np, particle, stateVars):
        stress = particle[np].stress
        stateVars[np].pressure = SphericalTensor(stress)
        stateVars[np].rho = self.density

    @ti.func
    def shear_stress(self, strain_rate):
        viscosity = self.viscosity
        mean_volumetric_strain_rate = voigt_tensor_trace(strain_rate) / 3.

        sstress = ZEROVEC6f
        sstress[0] = 2. * viscosity * (strain_rate[0] - mean_volumetric_strain_rate)
        sstress[1] = 2. * viscosity * (strain_rate[1] - mean_volumetric_strain_rate)
        sstress[2] = 2. * viscosity * (strain_rate[2] - mean_volumetric_strain_rate)
        sstress[3] = 2. * viscosity * strain_rate[3]
        sstress[4] = 2. * viscosity * strain_rate[4]
        sstress[5] = 2. * viscosity * strain_rate[5]
        return sstress
    
    @ti.func
    def core(self, np, strain_rate, stateVars, dt): 
        pressureAV = self.fluid_pressure(np, stateVars, strain_rate, dt)
        shear_stress = self.shear_stress(strain_rate)
        return -pressureAV * EYE + shear_stress