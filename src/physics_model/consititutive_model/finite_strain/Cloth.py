"""Finite-strain constitutive laws for embedded triangular surfaces."""

from __future__ import annotations

import numpy as np
import taichi as ti

from src.physics_model.consititutive_model.finite_strain.FiniteStrainModel import (
    InvertedElementError,
)


def _positive(name, value, allow_zero=False):
    value = float(value)
    valid = value >= 0.0 if allow_zero else value > 0.0
    if not np.isfinite(value) or not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


def normalize_cloth_bending_model(value):
    key = str(value).strip().replace("_", "").replace("-", "").lower()
    aliases = {
        "none": "None",
        "off": "None",
        "disabled": "None",
        "quadratic": "Quadratic",
        "quad": "Quadratic",
        "cotangent": "Quadratic",
        "dihedral": "Dihedral",
        "dihedralangle": "Dihedral",
        "ipc": "Dihedral",
    }
    if key not in aliases:
        raise ValueError("cloth bending_model must be 'None', 'Quadratic', or 'Dihedral'")
    return aliases[key]


def _surface_jacobian(deformation_gradient, minimum_jacobian):
    metric = deformation_gradient.T @ deformation_gradient
    determinant = float(np.linalg.det(metric))
    if determinant <= float(minimum_jacobian) ** 2:
        raise InvertedElementError("cloth triangle is inverted or collapsed")
    return np.sqrt(determinant), metric


def _spectral_arap_response(deformation_gradient, stretch_stiffness, compression_stiffness):
    """Return ARAP energy, PK1 and its analytic Frechet derivative."""
    metric = deformation_gradient.T @ deformation_gradient
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    eigenvalues = np.maximum(eigenvalues, np.finfo(np.float64).tiny)
    stretches = np.sqrt(eigenvalues)
    stiffnesses = np.where(stretches <= 1.0, compression_stiffness, stretch_stiffness)
    offsets = stretches - 1.0
    energy = float(np.dot(stiffnesses, offsets * offsets))
    responses = 2.0 * stiffnesses * offsets / stretches
    response_derivatives = stiffnesses / (stretches**3)
    response_matrix = (eigenvectors * responses[None, :]) @ eigenvectors.T
    first_piola = deformation_gradient @ response_matrix

    tangent = np.empty((3, 2, 3, 2), dtype=np.float64)
    eigenvalue_gap = eigenvalues[0] - eigenvalues[1]
    if abs(eigenvalue_gap) > 1.0e-12 * max(float(eigenvalues[-1]), 1.0):
        divided_difference = (responses[0] - responses[1]) / eigenvalue_gap
    else:
        divided_difference = 0.5 * np.sum(response_derivatives)
    for spatial_column in range(3):
        for material_column in range(2):
            increment = np.zeros((3, 2), dtype=np.float64)
            increment[spatial_column, material_column] = 1.0
            metric_increment = increment.T @ deformation_gradient + deformation_gradient.T @ increment
            local_increment = eigenvectors.T @ metric_increment @ eigenvectors
            response_increment = np.empty((2, 2), dtype=np.float64)
            response_increment[0, 0] = response_derivatives[0] * local_increment[0, 0]
            response_increment[1, 1] = response_derivatives[1] * local_increment[1, 1]
            response_increment[0, 1] = divided_difference * local_increment[0, 1]
            response_increment[1, 0] = divided_difference * local_increment[1, 0]
            tangent[:, :, spatial_column, material_column] = (
                increment @ response_matrix + deformation_gradient @ eigenvectors @ response_increment @ eigenvectors.T
            )
    return energy, first_piola, tangent


@ti.data_oriented
class ClothARAP:
    is_cloth = True
    cloth_model = "arap"
    constitutive_id = 0

    def __init__(
        self,
        stretch_stiffness,
        density=1.0,
        thickness=1.0,
        compression_stiffness=None,
        bending_stiffness=0.0,
        bending_poisson_ratio=0.0,
        bending_model="Quadratic",
        minimum_jacobian=1.0e-10,
        **parameters,
    ):
        # ``tangent_epsilon`` belonged to the removed finite-difference
        # implementation.  Accept and discard it for input compatibility.
        parameters.pop("tangent_epsilon", None)
        if parameters:
            raise TypeError(f"unexpected ClothARAP parameters: {', '.join(parameters)}")
        self.stretch_stiffness = _positive("stretch_stiffness", stretch_stiffness)
        self.compression_stiffness = _positive(
            "compression_stiffness",
            self.stretch_stiffness if compression_stiffness is None else compression_stiffness,
        )
        self.density = _positive("density", density)
        self.thickness = _positive("thickness", thickness)
        self.bending_stiffness = _positive("bending_stiffness", bending_stiffness, allow_zero=True)
        self.bending_poisson_ratio = float(bending_poisson_ratio)
        self.bending_model = normalize_cloth_bending_model(bending_model)
        self.minimum_jacobian = _positive("minimum_jacobian", minimum_jacobian)
        if not np.isfinite(self.bending_poisson_ratio) or abs(self.bending_poisson_ratio) >= 1.0:
            raise ValueError("bending_poisson_ratio must satisfy |nu| < 1")

    @property
    def young_modulus(self):
        return self.stretch_stiffness

    @property
    def quadratic_bending_modulus(self):
        return self.bending_stiffness * self.thickness**3 / (24.0 * (1.0 - self.bending_poisson_ratio**2))

    def lame_parameters(self, material_dimension=2):
        del material_dimension
        return 0.0, max(self.stretch_stiffness, self.compression_stiffness)

    @ti.func
    def _principal_response(self, eigenvalue):
        stretch = ti.sqrt(ti.max(eigenvalue, self.minimum_jacobian**2))
        stiffness = self.stretch_stiffness
        if stretch <= 1.0:
            stiffness = self.compression_stiffness
        response = 2.0 * stiffness * (stretch - 1.0) / stretch
        derivative = stiffness / (stretch * stretch * stretch)
        energy = stiffness * (stretch - 1.0) ** 2
        return energy, response, derivative

    @ti.func
    def _spectral_data(self, deformation_gradient):
        metric = deformation_gradient.transpose() @ deformation_gradient
        difference = metric[0, 0] - metric[1, 1]
        discriminant = ti.sqrt(
            ti.max(
                difference * difference + 4.0 * metric[0, 1] ** 2,
                0.0,
            )
        )
        eigenvalue0 = ti.max(
            0.5 * (metric.trace() + discriminant),
            self.minimum_jacobian**2,
        )
        eigenvalue1 = ti.max(
            0.5 * (metric.trace() - discriminant),
            self.minimum_jacobian**2,
        )
        energy0, response0, derivative0 = self._principal_response(eigenvalue0)
        energy1, response1, derivative1 = self._principal_response(eigenvalue1)
        identity = ti.Matrix.identity(float, 2)
        projector0 = 0.5 * identity
        projector1 = 0.5 * identity
        gap = eigenvalue0 - eigenvalue1
        if ti.abs(gap) > 1.0e-12 * ti.max(eigenvalue0, 1.0):
            projector0 = (metric - eigenvalue1 * identity) / gap
            projector1 = identity - projector0
        response_matrix = response0 * projector0 + response1 * projector1
        return (
            energy0 + energy1,
            metric,
            response_matrix,
            projector0,
            projector1,
            eigenvalue0,
            eigenvalue1,
            response0,
            response1,
            derivative0,
            derivative1,
        )

    @ti.func
    def strain_energy_density(self, deformation_gradient):
        energy, _, _, _, _, _, _, _, _, _, _ = self._spectral_data(deformation_gradient)
        return energy

    @ti.func
    def first_piola_stress(self, deformation_gradient):
        _, _, response, _, _, _, _, _, _, _, _ = self._spectral_data(deformation_gradient)
        return deformation_gradient @ response

    @ti.func
    def first_piola_tangent(self, deformation_gradient):
        (
            _,
            _,
            response_matrix,
            projector0,
            projector1,
            eigenvalue0,
            eigenvalue1,
            response0,
            response1,
            derivative0,
            derivative1,
        ) = self._spectral_data(deformation_gradient)
        gap = eigenvalue0 - eigenvalue1
        divided_difference = 0.5 * (derivative0 + derivative1)
        if ti.abs(gap) > 1.0e-12 * ti.max(eigenvalue0, 1.0):
            divided_difference = (response0 - response1) / gap
        tangent = ti.Matrix.zero(float, 6, 6)
        for column in ti.static(range(6)):
            spatial_column = column % 3
            material_column = column // 3
            increment = ti.Matrix.zero(float, 3, 2)
            increment[spatial_column, material_column] = 1.0
            metric_increment = (
                increment.transpose() @ deformation_gradient + deformation_gradient.transpose() @ increment
            )
            response_increment = (
                derivative0 * projector0 @ metric_increment @ projector0
                + derivative1 * projector1 @ metric_increment @ projector1
                + divided_difference
                * (projector0 @ metric_increment @ projector1 + projector1 @ metric_increment @ projector0)
            )
            piola_increment = increment @ response_matrix + deformation_gradient @ response_increment
            for material_row, spatial_row in ti.static(ti.ndrange(2, 3)):
                tangent[3 * material_row + spatial_row, column] = piola_increment[spatial_row, material_row]
        return 0.5 * (tangent + tangent.transpose())

    @ti.func
    def Psi(self, deformation_gradient):
        return self.strain_energy_density(deformation_gradient)

    @ti.func
    def dPsi_div_dF(self, deformation_gradient):
        piola = self.first_piola_stress(deformation_gradient)
        result = ti.Vector.zero(float, 6)
        for material, spatial in ti.static(ti.ndrange(2, 3)):
            result[3 * material + spatial] = piola[spatial, material]
        return result

    @ti.func
    def d2Psi_div_d2F(self, deformation_gradient):
        return self.first_piola_tangent(deformation_gradient)

    def evaluate(self, deformation_gradient, need_tangent=True):
        deformation_gradient = np.asarray(deformation_gradient, dtype=np.float64)
        if deformation_gradient.shape != (3, 2):
            raise ValueError("ClothARAP requires a TRI3 surface deformation gradient " "with shape (3, 2)")
        _surface_jacobian(deformation_gradient, self.minimum_jacobian)
        energy, first_piola, tangent = _spectral_arap_response(
            deformation_gradient,
            self.stretch_stiffness,
            self.compression_stiffness,
        )
        return energy, first_piola, tangent if need_tangent else None

    def cauchy_stress(self, deformation_gradient):
        deformation_gradient = np.asarray(deformation_gradient, dtype=np.float64)
        jacobian, _ = _surface_jacobian(deformation_gradient, self.minimum_jacobian)
        _, first_piola, _ = self.evaluate(deformation_gradient, need_tangent=False)
        return first_piola @ deformation_gradient.T / jacobian


@ti.data_oriented
class ClothNeoHookean:
    """Metric-based compressible Neo-Hookean cloth membrane."""

    is_cloth = True
    cloth_model = "neo_hookean"
    constitutive_id = 1

    def __init__(
        self,
        young_modulus,
        poisson_ratio,
        density=1.0,
        thickness=1.0,
        bending_stiffness=0.0,
        bending_poisson_ratio=0.0,
        bending_model="Quadratic",
        minimum_jacobian=1.0e-10,
        **parameters,
    ):
        parameters.pop("tangent_epsilon", None)
        if parameters:
            raise TypeError("unexpected ClothNeoHookean parameters: " + ", ".join(parameters))
        self.young_modulus = _positive("young_modulus", young_modulus)
        self.poisson_ratio = float(poisson_ratio)
        if not np.isfinite(self.poisson_ratio) or not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must satisfy -1 < nu < 0.5")
        self.density = _positive("density", density)
        self.thickness = _positive("thickness", thickness)
        self.bending_stiffness = _positive("bending_stiffness", bending_stiffness, allow_zero=True)
        self.bending_poisson_ratio = float(bending_poisson_ratio)
        self.bending_model = normalize_cloth_bending_model(bending_model)
        self.minimum_jacobian = _positive("minimum_jacobian", minimum_jacobian)
        if not np.isfinite(self.bending_poisson_ratio) or abs(self.bending_poisson_ratio) >= 1.0:
            raise ValueError("bending_poisson_ratio must satisfy |nu| < 1")
        self.lambda_, self.mu_ = self.lame_parameters(2)

    @property
    def quadratic_bending_modulus(self):
        return self.bending_stiffness * self.thickness**3 / (24.0 * (1.0 - self.bending_poisson_ratio**2))

    def lame_parameters(self, material_dimension=2):
        if material_dimension != 2:
            raise ValueError("ClothNeoHookean is a two-dimensional material")
        shear = self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))
        lame_lambda = self.young_modulus * self.poisson_ratio / (1.0 - self.poisson_ratio**2)
        return lame_lambda, shear

    @ti.func
    def strain_energy_density(self, deformation_gradient):
        metric = deformation_gradient.transpose() @ deformation_gradient
        jacobian = ti.sqrt(ti.max(metric.determinant(), self.minimum_jacobian**2))
        log_jacobian = ti.log(jacobian)
        return 0.5 * self.mu_ * (metric.trace() - 2.0 - 2.0 * log_jacobian) + 0.5 * self.lambda_ * log_jacobian**2

    @ti.func
    def first_piola_stress(self, deformation_gradient):
        metric = deformation_gradient.transpose() @ deformation_gradient
        jacobian = ti.sqrt(ti.max(metric.determinant(), self.minimum_jacobian**2))
        log_jacobian = ti.log(jacobian)
        return (
            self.mu_ * deformation_gradient
            + (-self.mu_ + self.lambda_ * log_jacobian) * deformation_gradient @ metric.inverse()
        )

    @ti.func
    def first_piola_tangent(self, deformation_gradient):
        metric = deformation_gradient.transpose() @ deformation_gradient
        inverse_metric = metric.inverse()
        jacobian = ti.sqrt(ti.max(metric.determinant(), self.minimum_jacobian**2))
        coefficient = -self.mu_ + self.lambda_ * ti.log(jacobian)
        metric_gradient = deformation_gradient @ inverse_metric
        projection = metric_gradient @ deformation_gradient.transpose()
        tangent = ti.Matrix.zero(float, 6, 6)
        for material_row, spatial_row, material_column, spatial_column in ti.ndrange(2, 3, 2, 3):
            value = self.lambda_ * metric_gradient[spatial_row, material_row] * metric_gradient[
                spatial_column, material_column
            ] + coefficient * (
                ((1.0 if spatial_row == spatial_column else 0.0) - projection[spatial_row, spatial_column])
                * inverse_metric[material_column, material_row]
                - metric_gradient[spatial_row, material_column] * metric_gradient[spatial_column, material_row]
            )
            if spatial_row == spatial_column and material_row == material_column:
                value += self.mu_
            tangent[
                3 * material_row + spatial_row,
                3 * material_column + spatial_column,
            ] = value
        return 0.5 * (tangent + tangent.transpose())

    @ti.func
    def Psi(self, deformation_gradient):
        return self.strain_energy_density(deformation_gradient)

    @ti.func
    def dPsi_div_dF(self, deformation_gradient):
        piola = self.first_piola_stress(deformation_gradient)
        result = ti.Vector.zero(float, 6)
        for material, spatial in ti.static(ti.ndrange(2, 3)):
            result[3 * material + spatial] = piola[spatial, material]
        return result

    @ti.func
    def d2Psi_div_d2F(self, deformation_gradient):
        return self.first_piola_tangent(deformation_gradient)

    def evaluate(self, deformation_gradient, need_tangent=True):
        deformation_gradient = np.asarray(deformation_gradient, dtype=np.float64)
        if deformation_gradient.shape != (3, 2):
            raise ValueError("ClothNeoHookean requires a TRI3 surface deformation " "gradient with shape (3, 2)")
        jacobian, metric = _surface_jacobian(deformation_gradient, self.minimum_jacobian)
        log_jacobian = np.log(jacobian)
        inverse_metric = np.linalg.inv(metric)
        energy = 0.5 * self.mu_ * (np.trace(metric) - 2.0 - 2.0 * log_jacobian) + 0.5 * self.lambda_ * log_jacobian**2
        metric_gradient = deformation_gradient @ inverse_metric
        first_piola = self.mu_ * deformation_gradient + (-self.mu_ + self.lambda_ * log_jacobian) * metric_gradient
        tangent = None
        if need_tangent:
            projection = metric_gradient @ deformation_gradient.T
            tangent = np.empty((3, 2, 3, 2), dtype=np.float64)
            coefficient = -self.mu_ + self.lambda_ * log_jacobian
            for i in range(3):
                for material_i in range(2):
                    for k in range(3):
                        for material_k in range(2):
                            tangent[i, material_i, k, material_k] = (
                                self.mu_ * float(i == k and material_i == material_k)
                                + self.lambda_ * metric_gradient[i, material_i] * metric_gradient[k, material_k]
                                + coefficient
                                * (
                                    (float(i == k) - projection[i, k]) * inverse_metric[material_k, material_i]
                                    - metric_gradient[i, material_k] * metric_gradient[k, material_i]
                                )
                            )
        return float(energy), first_piola, tangent

    def cauchy_stress(self, deformation_gradient):
        deformation_gradient = np.asarray(deformation_gradient, dtype=np.float64)
        jacobian, _ = _surface_jacobian(deformation_gradient, self.minimum_jacobian)
        _, first_piola, _ = self.evaluate(deformation_gradient, need_tangent=False)
        return first_piola @ deformation_gradient.T / jacobian


__all__ = [
    "ClothARAP",
    "ClothNeoHookean",
    "normalize_cloth_bending_model",
]
