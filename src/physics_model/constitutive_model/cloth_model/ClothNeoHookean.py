"""ClothNeoHookean membrane response for embedded triangular surfaces."""

from __future__ import annotations

from math import isfinite
import taichi as ti

from src.physics_model.constitutive_model.cloth_model.ClothBending import ClothBending
from src.utils.linalg import positive


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
        self.young_modulus = positive("young_modulus", young_modulus)
        self.poisson_ratio = float(poisson_ratio)
        if not isfinite(self.poisson_ratio) or not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must satisfy -1 < nu < 0.5")
        self.density = positive("density", density)
        self.thickness = positive("thickness", thickness)
        self.bending_stiffness = positive("bending_stiffness", bending_stiffness, allow_zero=True)
        self.bending_poisson_ratio = float(bending_poisson_ratio)
        self.bending_model = ClothBending.validate_model(bending_model)
        self.minimum_jacobian = positive("minimum_jacobian", minimum_jacobian)
        if not isfinite(self.bending_poisson_ratio) or abs(self.bending_poisson_ratio) >= 1.0:
            raise ValueError("bending_poisson_ratio must satisfy |nu| < 1")
        self.lambda_, self.mu_ = self.lame_parameters(2)

    @property
    def quadratic_bending_modulus(self):
        return ClothBending.quadratic_modulus(self.bending_stiffness, self.thickness, self.bending_poisson_ratio)

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
    def psi(self, deformation_gradient):
        return self.strain_energy_density(deformation_gradient)

    @ti.func
    def d_psi_div_d_f(self, deformation_gradient):
        piola = self.first_piola_stress(deformation_gradient)
        result = ti.Vector.zero(float, 6)
        for material, spatial in ti.static(ti.ndrange(2, 3)):
            result[3 * material + spatial] = piola[spatial, material]
        return result

    @ti.func
    def d2_psi_div_d2_f(self, deformation_gradient):
        return self.first_piola_tangent(deformation_gradient)

    @ti.func
    def cauchy_stress(self, deformation_gradient):
        metric = deformation_gradient.transpose() @ deformation_gradient
        jacobian = ti.sqrt(ti.max(metric.determinant(), self.minimum_jacobian**2))
        first_piola = self.first_piola_stress(deformation_gradient)
        return first_piola @ deformation_gradient.transpose() / jacobian
