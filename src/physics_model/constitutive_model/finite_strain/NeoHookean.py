import numpy as np
import taichi as ti

from src.physics_model.constitutive_model.finite_strain.MaterialKernel import *
from src.physics_model.constitutive_model.finite_strain.FiniteStrainModel import FiniteStrainModel
from src.utils.constants import THRESHOLD
import src.utils.GlobalVariable as GlobalVariable
from src.utils.TypeDefinition import mat3x3
from src.utils.MatrixFunction import matrix_form_3d_from_stress
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class NeoHookeanModel(FiniteStrainModel):
    def __init__(self, material_type="Solid", configuration="TL", solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.is_elastic = True
        self.thickness = 1.0
        self.minimum_jacobian = 1.0e-10

    def model_initialize(self, material):
        self.material = material
        density = DictIO.get_alternative(material, "Density", 2650)
        young = DictIO.get_essential(material, "YoungModulus")
        poisson = DictIO.get_alternative(material, "PoissonRatio", 0.3)
        self.thickness = float(DictIO.get_alternative(material, "Thickness", 1.0))
        self.minimum_jacobian = float(DictIO.get_alternative(material, "MinimumJacobian", 1.0e-10))
        self.validate_elastic_parameters(density, young, poisson)
        if not np.isfinite(self.thickness) or self.thickness <= 0.0:
            raise ValueError("Thickness must be finite and positive")
        if not np.isfinite(self.minimum_jacobian) or self.minimum_jacobian <= 0.0:
            raise ValueError("MinimumJacobian must be finite and positive")
        self.add_material(density, young, poisson)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson):
        self.density = density
        self.young = young
        self.poisson = poisson
        self.shear = 0.5 * self.young / (1.0 + self.poisson)
        self.bulk = self.young / (3.0 * (1 - 2.0 * self.poisson))
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def print_message(self, materialID):
        self.print_console_header()
        print("Constitutive model: Neo-Hookean")
        print("Material ID: ", materialID)
        print("Density: ", self.density)
        print("Young Modulus: ", self.young)
        print("Poisson Ratio: ", self.poisson, "\n")

    @property
    def lame_parameters(self):
        return self.lame_lambda, self.shear

    @ti.func
    def cauchy_stress(self, deformation_gradient):
        jacobian = deformation_gradient.determinant()
        return self.first_piola_stress(deformation_gradient) @ deformation_gradient.transpose() / jacobian

    def define_state_vars(self):
        return {
            "stress0": mat3x3,
            "deformation_gradient": ti.types.matrix(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION, float),
        }

    @ti.func
    def _initialize_vars(self, np, particle, state_variables):
        state_variables[np].deformation_gradient = ti.Matrix.identity(float, state_variables[np].deformation_gradient.n)
        stress = particle[np].stress
        state_variables[np].stress0 = matrix_form_3d_from_stress(stress)

    @ti.func
    def strain_energy_density(self, deformation_gradient):
        la = 3.0 * self.bulk * self.poisson / (1.0 + self.poisson)
        jacobian = ti.max(deformation_gradient.determinant(), THRESHOLD)
        log_j = ti.log(jacobian)
        i1 = 0.0
        for i in ti.static(range(deformation_gradient.n)):
            for j in ti.static(range(deformation_gradient.m)):
                i1 += deformation_gradient[i, j] * deformation_gradient[i, j]
        return 0.5 * self.shear * (i1 - deformation_gradient.n) - self.shear * log_j + 0.5 * la * log_j * log_j

    @ti.func
    def first_piola_stress(self, deformation_gradient):
        la = 3.0 * self.bulk * self.poisson / (1.0 + self.poisson)
        det_F = ti.max(deformation_gradient.determinant(), THRESHOLD)
        inverse_transpose = deformation_gradient.inverse().transpose()
        return self.shear * (deformation_gradient - inverse_transpose) + la * ti.log(det_F) * inverse_transpose

    @ti.func
    def first_piola_tangent(self, deformation_gradient):
        la = 3.0 * self.bulk * self.poisson / (1.0 + self.poisson)
        det_F = ti.max(deformation_gradient.determinant(), THRESHOLD)
        log_j = ti.log(det_F)
        inverse_transpose = deformation_gradient.inverse().transpose()
        size = ti.static(deformation_gradient.n * deformation_gradient.m)
        tangent = ti.Matrix.zero(float, size, size)
        tangent_entry = 0
        while tangent_entry < size * size:
            row = tangent_entry // size
            column = tangent_entry - row * size
            a = row // deformation_gradient.n
            i = row - a * deformation_gradient.n
            b = column // deformation_gradient.n
            j = column - b * deformation_gradient.n
            diagonal = 0.0
            if i == j and a == b:
                diagonal = self.shear
            tangent[row, column] = (
                diagonal
                + la * inverse_transpose[i, a] * inverse_transpose[j, b]
                - (la * log_j - self.shear) * inverse_transpose[i, b] * inverse_transpose[j, a]
            )
            tangent_entry += 1
        return tangent
