"""St. Venant--Kirchhoff finite-strain elasticity."""

from __future__ import annotations

import numpy as np
import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.physics_model.consititutive_model.finite_strain.FiniteStrainModel import (
    FiniteStrainModel,
    InvertedElementError,
)
from src.utils.MatrixFunction import matrix_form_3d_from_stress
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import mat3x3


@ti.data_oriented
class StVenantKirchhoffModel(FiniteStrainModel):
    def __init__(
        self,
        material_type='Solid',
        configuration='TL',
        solver_type='Explicit',
    ):
        super().__init__(material_type, configuration, solver_type)
        self.is_elastic = True
        self.thickness = 1.

    def model_initialize(self, material):
        self.material = material
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        self.thickness = float(
            DictIO.GetAlternative(material, 'Thickness', 1.)
        )
        self.validate_elastic_parameters(density, young, poisson)
        if not np.isfinite(self.thickness) or self.thickness <= 0.:
            raise ValueError('Thickness must be finite and positive')
        self.add_material(density, young, poisson)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson):
        self.density = density
        self.young = young
        self.poisson = poisson
        self.shear = 0.5 * self.young / (1. + self.poisson)
        self.bulk = self.young / (3. * (1. - 2. * self.poisson))
        self.max_sound_speed = self.get_sound_speed(
            self.density, self.young, self.poisson
        )

    def print_message(self, materialID):
        self.print_console_header()
        print('Constitutive model: St. Venant-Kirchhoff')
        print('Material ID: ', materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Poisson Ratio: ', self.poisson, '\n')

    def lame_parameters(self, material_dimension):
        if material_dimension == 2:
            lame_lambda = (
                self.young
                * self.poisson
                / (1. - self.poisson * self.poisson)
            )
        else:
            lame_lambda = self.lame_lambda
        return lame_lambda, self.shear

    def define_state_vars(self):
        return {
            'stress0': mat3x3,
            'deformation_gradient': ti.types.matrix(
                GlobalVariable.DIMENSION,
                GlobalVariable.DIMENSION,
                float,
            ),
        }

    @ti.func
    def _initialize_vars_(self, np, particle, stateVars):
        stateVars[np].deformation_gradient = ti.Matrix.identity(
            float, stateVars[np].deformation_gradient.n
        )
        stateVars[np].stress0 = matrix_form_3d_from_stress(
            particle[np].stress
        )

    @ti.func
    def _device_lame_lambda(self, deformation_gradient):
        lame_lambda = self.lame_lambda
        if ti.static(deformation_gradient.n == 2):
            lame_lambda = (
                self.young
                * self.poisson
                / (1. - self.poisson * self.poisson)
            )
        return lame_lambda

    @ti.func
    def strain_energy_density(self, deformation_gradient):
        identity = ti.Matrix.identity(float, deformation_gradient.n)
        strain = 0.5 * (
            deformation_gradient.transpose()
            @ deformation_gradient
            - identity
        )
        lame_lambda = self._device_lame_lambda(deformation_gradient)
        return (
            0.5 * lame_lambda * strain.trace() * strain.trace()
            + self.shear * (strain.transpose() @ strain).trace()
        )

    @ti.func
    def first_piola_stress(self, deformation_gradient):
        identity = ti.Matrix.identity(float, deformation_gradient.n)
        strain = 0.5 * (
            deformation_gradient.transpose()
            @ deformation_gradient
            - identity
        )
        lame_lambda = self._device_lame_lambda(deformation_gradient)
        second_piola = (
            lame_lambda * strain.trace() * identity
            + 2. * self.shear * strain
        )
        return deformation_gradient @ second_piola

    @ti.func
    def first_piola_tangent(self, deformation_gradient):
        identity = ti.Matrix.identity(float, deformation_gradient.n)
        strain = 0.5 * (
            deformation_gradient.transpose()
            @ deformation_gradient
            - identity
        )
        lame_lambda = self._device_lame_lambda(deformation_gradient)
        second_piola = (
            lame_lambda * strain.trace() * identity
            + 2. * self.shear * strain
        )
        left_cauchy = deformation_gradient @ deformation_gradient.transpose()
        dimension = ti.static(deformation_gradient.n)
        size = ti.static(dimension * dimension)
        tangent = ti.Matrix.zero(float, size, size)
        tangent_entry = 0
        while tangent_entry < size * size:
            row = tangent_entry // size
            column = tangent_entry - row * size
            material_i = row // dimension
            spatial_i = row - material_i * dimension
            material_j = column // dimension
            spatial_j = column - material_j * dimension
            value = (
                lame_lambda
                * deformation_gradient[spatial_i, material_i]
                * deformation_gradient[spatial_j, material_j]
                + self.shear
                * deformation_gradient[spatial_i, material_j]
                * deformation_gradient[spatial_j, material_i]
            )
            if spatial_i == spatial_j:
                value += second_piola[material_j, material_i]
            if material_i == material_j:
                value += self.shear * left_cauchy[spatial_i, spatial_j]
            tangent[row, column] = value
            tangent_entry += 1
        return tangent

    @ti.func
    def surface_strain_energy_density(self, deformation_gradient):
        """Plane-stress StVK energy for an embedded 3x2 surface map."""
        identity = ti.Matrix.identity(float, 2)
        strain = 0.5 * (
            deformation_gradient.transpose() @ deformation_gradient
            - identity
        )
        lame_lambda = (
            self.young
            * self.poisson
            / (1. - self.poisson * self.poisson)
        )
        return (
            0.5 * lame_lambda * strain.trace() ** 2
            + self.shear * (strain.transpose() @ strain).trace()
        )

    @ti.func
    def surface_first_piola_stress(self, deformation_gradient):
        identity = ti.Matrix.identity(float, 2)
        strain = 0.5 * (
            deformation_gradient.transpose() @ deformation_gradient
            - identity
        )
        lame_lambda = (
            self.young
            * self.poisson
            / (1. - self.poisson * self.poisson)
        )
        second_piola = (
            lame_lambda * strain.trace() * identity
            + 2. * self.shear * strain
        )
        return deformation_gradient @ second_piola

    @ti.func
    def surface_first_piola_tangent(self, deformation_gradient):
        identity = ti.Matrix.identity(float, 2)
        strain = 0.5 * (
            deformation_gradient.transpose() @ deformation_gradient
            - identity
        )
        lame_lambda = (
            self.young
            * self.poisson
            / (1. - self.poisson * self.poisson)
        )
        second_piola = (
            lame_lambda * strain.trace() * identity
            + 2. * self.shear * strain
        )
        left_cauchy = deformation_gradient @ deformation_gradient.transpose()
        tangent = ti.Matrix.zero(float, 6, 6)
        for material_i, spatial_i, material_j, spatial_j in ti.ndrange(
            2, 3, 2, 3
        ):
            value = (
                lame_lambda
                * deformation_gradient[spatial_i, material_i]
                * deformation_gradient[spatial_j, material_j]
                + self.shear
                * deformation_gradient[spatial_i, material_j]
                * deformation_gradient[spatial_j, material_i]
            )
            if spatial_i == spatial_j:
                value += second_piola[material_j, material_i]
            if material_i == material_j:
                value += self.shear * left_cauchy[spatial_i, spatial_j]
            tangent[
                3 * material_i + spatial_i,
                3 * material_j + spatial_j,
            ] = value
        return 0.5 * (tangent + tangent.transpose())

    def evaluate(self, deformation_gradient, need_tangent=True):
        deformation_gradient = np.asarray(
            deformation_gradient, dtype=np.float64
        )
        if deformation_gradient.ndim != 2:
            raise ValueError('deformation_gradient must be a matrix')
        spatial_dimension, material_dimension = deformation_gradient.shape
        lame_lambda, shear = self.lame_parameters(material_dimension)
        identity = np.eye(material_dimension)
        strain = 0.5 * (
            deformation_gradient.T @ deformation_gradient - identity
        )
        second_piola = (
            lame_lambda * np.trace(strain) * identity
            + 2. * shear * strain
        )
        first_piola = deformation_gradient @ second_piola
        energy = (
            0.5 * lame_lambda * np.trace(strain) ** 2
            + shear * np.sum(strain * strain)
        )
        tangent = None
        if need_tangent:
            left_cauchy = deformation_gradient @ deformation_gradient.T
            tangent = np.empty(
                (
                    spatial_dimension,
                    material_dimension,
                    spatial_dimension,
                    material_dimension,
                ),
                dtype=np.float64,
            )
            for i in range(spatial_dimension):
                for j in range(material_dimension):
                    for k in range(spatial_dimension):
                        for ell in range(material_dimension):
                            tangent[i, j, k, ell] = (
                                (1. if i == k else 0.)
                                * second_piola[ell, j]
                                + lame_lambda
                                * deformation_gradient[i, j]
                                * deformation_gradient[k, ell]
                                + shear
                                * deformation_gradient[i, ell]
                                * deformation_gradient[k, j]
                                + shear
                                * (1. if j == ell else 0.)
                                * left_cauchy[i, k]
                            )
        return float(energy), first_piola, tangent

    def cauchy_stress(self, deformation_gradient):
        deformation_gradient = np.asarray(
            deformation_gradient, dtype=np.float64
        )
        _, first_piola, _ = self.evaluate(
            deformation_gradient, need_tangent=False
        )
        if deformation_gradient.shape[1] == 2:
            jacobian = np.sqrt(
                np.linalg.det(
                    deformation_gradient.T @ deformation_gradient
                )
            )
        else:
            jacobian = np.linalg.det(deformation_gradient)
        if jacobian <= 1.e-14:
            raise InvertedElementError(
                'cannot compute stress for a collapsed element'
            )
        return first_piola @ deformation_gradient.T / jacobian


__all__ = ['StVenantKirchhoffModel']
