import taichi as ti

from src.physics_model.consititutive_model.finite_strain.MaterialKernel import *
from src.physics_model.consititutive_model.finite_strain.FiniteStrainModel import PlasticModel
import src.utils.GlobalVariable as GlobalVariable
from src.utils.TypeDefination import mat3x3
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class VonMises(PlasticModel):
    def __init__(self, material_type, configuration, solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)

    def model_initialize(self, material):
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        _yield = DictIO.GetEssential(material, 'YieldStress')
        self.add_material(density, young, poisson, _yield)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson, yield_stress):
        self.density = density
        self.young = young
        self.poisson = poisson
        self.shear = 0.5 * self.young / (1. + self.poisson)
        self.bulk = self.young / (3. * (1 - 2. * self.poisson))
        self._yield = yield_stress
        self.get_sound_speed(self.density, self.young, self.poisson)

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Von Mises Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Poisson Ratio: ', self.poisson)
        print('Yield Stress: ', self._yield, '\n')

    def define_state_vars(self):
        return {'stress0': mat3x3, 'deformation_gradient': ti.types.matrix(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION, float)}

    @ti.func
    def _initialize_vars_(self, np, particle, stateVars):
        stateVars[np].deformation_gradient = ti.Matrix.identity(float, GlobalVariable.DIMENSION)    
        stateVars[np].stress0 = particle[np].stress
        
    # ============================================================================= #
    #                              Return Mapping                                   #
    # ============================================================================= #
    @ti.func
    def plastic_process(self, matrixU, matrixVT, hencky_strain, hencky_trace_trace, hencky_deviatoric, hencky_deviatoric_norm, state_vars):
        delta_gamma = hencky_deviatoric_norm - self._yield / e_mu_prefac
        if delta_gamma > 0:
            delta_gamma = d_prefac * delta_gamma / dt
            hencky -= delta_gamma * hencky_deviatoric
            state_vars.deformation_gradient = matrixU @ hencky.array().exp().matrix().asDiagonal() @ matrixVT
            state_vars.elstrain += delta_gamma

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stateVars):
        pass