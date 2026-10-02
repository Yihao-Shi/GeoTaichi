import numpy as np
import taichi as ti

from src.physics_model.consititutive_model.finite_strain.MaterialKernel import *
from src.physics_model.consititutive_model.finite_strain.FiniteStrainModel import FiniteStrainModel, InvertedElementError
from src.utils.constants import Threshold
import src.utils.GlobalVariable as GlobalVariable
from src.utils.TypeDefination import mat3x3
from src.utils.MatrixFunction import matrix_form_3d_from_stress
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class NeoHookeanModel(FiniteStrainModel):
    def __init__(self, material_type="Solid", configuration="TL", solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.is_elastic = True
        self.thickness = 1.
        self.minimum_jacobian = 1.e-10

    def model_initialize(self, material):
        self.material = material
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        self.thickness = float(DictIO.GetAlternative(material, 'Thickness', 1.))
        self.minimum_jacobian = float(
            DictIO.GetAlternative(material, 'MinimumJacobian', 1.e-10)
        )
        self.validate_elastic_parameters(density, young, poisson)
        if not np.isfinite(self.thickness) or self.thickness <= 0.:
            raise ValueError('Thickness must be finite and positive')
        if not np.isfinite(self.minimum_jacobian) or self.minimum_jacobian <= 0.:
            raise ValueError('MinimumJacobian must be finite and positive')
        self.add_material(density, young, poisson)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson):
        self.density = density
        self.young = young
        self.poisson = poisson
        self.shear = 0.5 * self.young / (1. + self.poisson)
        self.bulk = self.young / (3. * (1 - 2. * self.poisson))
        self.max_sound_speed = self.get_sound_speed(
            self.density, self.young, self.poisson
        )
        
    def print_message(self, materialID):
        self.print_console_header()
        print('Constitutive model: Neo-Hookean')
        print("Material ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Poisson Ratio: ', self.poisson, '\n')

    @property
    def lame_parameters(self):
        return self.lame_lambda, self.shear

    def evaluate(self, deformation_gradient, need_tangent=True):
        deformation_gradient = np.asarray(
            deformation_gradient, dtype=np.float64
        )
        if deformation_gradient.ndim != 2 or (
            deformation_gradient.shape[0] != deformation_gradient.shape[1]
        ):
            raise ValueError(
                'NeoHookeanModel requires a square volume deformation gradient'
            )
        dimension = deformation_gradient.shape[0]
        jacobian = float(np.linalg.det(deformation_gradient))
        if jacobian <= self.minimum_jacobian:
            raise InvertedElementError(
                f'Neo-Hookean deformation Jacobian {jacobian:.3e} is not positive'
            )
        lame_lambda, shear = self.lame_parameters
        log_jacobian = np.log(jacobian)
        inverse_transpose = np.linalg.inv(deformation_gradient).T
        first_piola = (
            shear * (deformation_gradient - inverse_transpose)
            + lame_lambda * log_jacobian * inverse_transpose
        )
        energy = (
            0.5
            * shear
            * (np.sum(deformation_gradient * deformation_gradient) - dimension)
            - shear * log_jacobian
            + 0.5 * lame_lambda * log_jacobian**2
        )
        tangent = None
        if need_tangent:
            tangent = np.empty(
                (dimension, dimension, dimension, dimension),
                dtype=np.float64,
            )
            for i in range(dimension):
                for j in range(dimension):
                    for k in range(dimension):
                        for ell in range(dimension):
                            tangent[i, j, k, ell] = (
                                shear
                                * (1. if i == k and j == ell else 0.)
                                + (shear - lame_lambda * log_jacobian)
                                * inverse_transpose[i, ell]
                                * inverse_transpose[k, j]
                                + lame_lambda
                                * inverse_transpose[i, j]
                                * inverse_transpose[k, ell]
                            )
        return float(energy), first_piola, tangent

    def cauchy_stress(self, deformation_gradient):
        deformation_gradient = np.asarray(
            deformation_gradient, dtype=np.float64
        )
        _, first_piola, _ = self.evaluate(
            deformation_gradient, need_tangent=False
        )
        return (
            first_piola
            @ deformation_gradient.T
            / np.linalg.det(deformation_gradient)
        )

    def define_state_vars(self):
        return {'stress0': mat3x3, 'deformation_gradient': ti.types.matrix(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION, float)}

    @ti.func
    def _initialize_vars_(self, np, particle, stateVars):
        stateVars[np].deformation_gradient = ti.Matrix.identity(float, stateVars[np].deformation_gradient.n) 
        stress = particle[np].stress
        stateVars[np].stress0 = matrix_form_3d_from_stress(stress)

    @ti.func
    def strain_energy_density(self, deformation_gradient):
        la = 3. * self.bulk * self.poisson / (1. + self.poisson)
        jacobian = ti.max(deformation_gradient.determinant(), Threshold)
        log_j = ti.log(jacobian)
        i1 = 0.
        for i in ti.static(range(deformation_gradient.n)):
            for j in ti.static(range(deformation_gradient.m)):
                i1 += deformation_gradient[i, j] * deformation_gradient[i, j]
        return 0.5 * self.shear * (i1 - deformation_gradient.n) - self.shear * log_j + 0.5 * la * log_j * log_j

    @ti.func
    def first_piola_stress(self, deformation_gradient):
        la = 3. * self.bulk * self.poisson / (1. + self.poisson)
        det_F = ti.max(deformation_gradient.determinant(), Threshold)
        inverse_transpose = deformation_gradient.inverse().transpose()
        return self.shear * (deformation_gradient - inverse_transpose) + la * ti.log(det_F) * inverse_transpose

    @ti.func
    def first_piola_tangent(self, deformation_gradient):
        la = 3. * self.bulk * self.poisson / (1. + self.poisson)
        det_F = ti.max(deformation_gradient.determinant(), Threshold)
        log_j = ti.log(det_F)
        inverse_transpose = deformation_gradient.inverse().transpose()
        size = ti.static(
            deformation_gradient.n * deformation_gradient.m
        )
        tangent = ti.Matrix.zero(float, size, size)
        tangent_entry = 0
        while tangent_entry < size * size:
            row = tangent_entry // size
            column = tangent_entry - row * size
            a = row // deformation_gradient.n
            i = row - a * deformation_gradient.n
            b = column // deformation_gradient.n
            j = column - b * deformation_gradient.n
            diagonal = 0.
            if i == j and a == b:
                diagonal = self.shear
            tangent[row, column] = (
                diagonal
                + la
                * inverse_transpose[i, a]
                * inverse_transpose[j, b]
                - (la * log_j - self.shear)
                * inverse_transpose[i, b]
                * inverse_transpose[j, a]
            )
            tangent_entry += 1
        return tangent
