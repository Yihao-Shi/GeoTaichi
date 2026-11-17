import taichi as ti
import numpy as np
from src.utils.MatrixFunction import flatten_matrix
from src.utils.constants import THRESHOLD


@ti.data_oriented
class SoftParticleNeoHookeanMaterialTable(object):
    """Material-indexed Neo-Hookean parameters for mixed soft bodies."""

    def __init__(self, capacity, models):
        self.shear = ti.field(float, shape=capacity)
        self.lame = ti.field(float, shape=capacity)
        shear = np.zeros(capacity, dtype=np.float64)
        lame = np.zeros(capacity, dtype=np.float64)
        for material_id, model in models.items():
            shear[material_id] = model.shear
            lame[material_id] = model.lame_lambda
        self.shear.from_numpy(shear)
        self.lame.from_numpy(lame)

    @ti.func
    def soft_particle_pk1(self, material_id, deformation_gradient):
        mu = self.shear[material_id]
        la = self.lame[material_id]
        det_f = ti.max(deformation_gradient.determinant(), THRESHOLD)
        inverse_transpose = deformation_gradient.inverse().transpose()
        return mu * (deformation_gradient - inverse_transpose) + la * ti.log(det_f) * inverse_transpose

    @ti.func
    def psi(self, material_id, deformation_gradient):
        mu = self.shear[material_id]
        la = self.lame[material_id]
        det_f = ti.max(deformation_gradient.determinant(), THRESHOLD)
        log_j = ti.log(det_f)
        i1 = 0.0
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                i1 += deformation_gradient[i, j] * deformation_gradient[i, j]
        return 0.5 * mu * (i1 - 3.0) - mu * log_j + 0.5 * la * log_j * log_j

    @ti.func
    def d_psi_div_d_f(self, material_id, deformation_gradient):
        return flatten_matrix(self.soft_particle_pk1(material_id, deformation_gradient))

    @ti.func
    def d2_psi_div_d2_f(self, material_id, deformation_gradient):
        mu = self.shear[material_id]
        la = self.lame[material_id]
        det_f = ti.max(deformation_gradient.determinant(), THRESHOLD)
        log_j = ti.log(det_f)
        inverse_transpose = deformation_gradient.inverse().transpose()
        tangent = ti.Matrix.zero(float, 9, 9)
        for row in range(9):
            for column in range(9):
                a = row // 3
                i = row - 3 * a
                b = column // 3
                j = column - 3 * b
                diagonal = 0.0
                if i == j and a == b:
                    diagonal = mu
                tangent[row, column] = (
                    diagonal
                    + la * inverse_transpose[i, a] * inverse_transpose[j, b]
                    - (la * log_j - mu) * inverse_transpose[i, b] * inverse_transpose[j, a]
                )
        return tangent

    @ti.func
    def update_soft_particle_state(self, np, material_id, deformation_gradient, pk1_stress, state_vars):
        det_f = ti.max(deformation_gradient.determinant(), THRESHOLD)
        cauchy = pk1_stress @ deformation_gradient.transpose() / det_f
        state_vars[np].estress = ti.sqrt(
            ti.max(
                0.5
                * (
                    (cauchy[0, 0] - cauchy[1, 1]) ** 2
                    + (cauchy[1, 1] - cauchy[2, 2]) ** 2
                    + (cauchy[0, 0] - cauchy[2, 2]) ** 2
                )
                + 3.0 * (cauchy[0, 1] ** 2 + cauchy[1, 2] ** 2 + cauchy[0, 2] ** 2),
                0.0,
            )
        )

    @ti.func
    def trial_deformation_gradient(self, particle_id, material_id, deformation_gradient, displacement_gradient):
        return deformation_gradient + displacement_gradient

    @ti.func
    def psi_at(self, particle_id, material_id, deformation_gradient):
        return self.psi(material_id, deformation_gradient)

    @ti.func
    def d_psi_div_d_f_at(self, particle_id, material_id, deformation_gradient):
        return self.d_psi_div_d_f(material_id, deformation_gradient)

    @ti.func
    def d2_psi_div_d2_f_at(self, particle_id, material_id, deformation_gradient):
        return self.d2_psi_div_d2_f(material_id, deformation_gradient)

    @ti.func
    def commit_soft_particle_state(self, particle_id, material_id, deformation_gradient, state_vars):
        stress = self.soft_particle_pk1(material_id, deformation_gradient)
        self.update_soft_particle_state(particle_id, material_id, deformation_gradient, stress, state_vars)
        return deformation_gradient, stress
