"""ClothARAP membrane response for embedded triangular surfaces."""

from __future__ import annotations

from math import isfinite
import taichi as ti

from src.physics_model.constitutive_model.cloth_model.ClothBending import ClothBending
from src.utils.linalg import positive


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
        self.stretch_stiffness = positive("stretch_stiffness", stretch_stiffness)
        self.compression_stiffness = positive(
            "compression_stiffness",
            self.stretch_stiffness if compression_stiffness is None else compression_stiffness,
        )
        self.density = positive("density", density)
        self.thickness = positive("thickness", thickness)
        self.bending_stiffness = positive("bending_stiffness", bending_stiffness, allow_zero=True)
        self.bending_poisson_ratio = float(bending_poisson_ratio)
        self.bending_model = ClothBending.validate_model(bending_model)
        self.minimum_jacobian = positive("minimum_jacobian", minimum_jacobian)
        if not isfinite(self.bending_poisson_ratio) or abs(self.bending_poisson_ratio) >= 1.0:
            raise ValueError("bending_poisson_ratio must satisfy |nu| < 1")

    @property
    def young_modulus(self):
        return self.stretch_stiffness

    @property
    def quadratic_bending_modulus(self):
        return ClothBending.quadratic_modulus(self.bending_stiffness, self.thickness, self.bending_poisson_ratio)

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
        for column in range(6):
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
