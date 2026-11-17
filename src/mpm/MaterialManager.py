import taichi as ti
import numpy as np


@ti.data_oriented
class SoftParticleSingleMaterialAdapter(object):
    """Give legacy soft constitutive models the material-indexed interface."""

    def __init__(self, model):
        self.model = model
        self.is_finite_strain_plastic = bool(getattr(model, "is_finite_strain_plastic", False))

    @ti.func
    def soft_particle_pk1(self, material_id, deformation_gradient):
        return self.model.soft_particle_pk1(deformation_gradient)

    @ti.func
    def psi(self, material_id, deformation_gradient):
        return self.model.psi(deformation_gradient)

    @ti.func
    def d_psi_div_d_f(self, material_id, deformation_gradient):
        return self.model.d_psi_div_d_f(deformation_gradient)

    @ti.func
    def d2_psi_div_d2_f(self, material_id, deformation_gradient):
        return self.model.d2_psi_div_d2_f(deformation_gradient)

    @ti.func
    def update_soft_particle_state(self, np, material_id, deformation_gradient, pk1_stress, state_vars):
        self.model.update_soft_particle_state(np, deformation_gradient, pk1_stress, state_vars)

    @ti.func
    def trial_deformation_gradient(self, particle_id, material_id, deformation_gradient, displacement_gradient):
        return deformation_gradient + displacement_gradient

    @ti.func
    def psi_at(self, particle_id, material_id, deformation_gradient):
        value = 0.0
        if ti.static(self.is_finite_strain_plastic):
            value = self.model.total_strain_energy_density_at(particle_id, deformation_gradient)
        else:
            value = self.model.psi(deformation_gradient)
        return value

    @ti.func
    def d_psi_div_d_f_at(self, particle_id, material_id, deformation_gradient):
        value = ti.Vector.zero(float, 9)
        if ti.static(self.is_finite_strain_plastic):
            value = self.model.total_d_psi_div_d_f_at(particle_id, deformation_gradient)
        else:
            value = self.model.d_psi_div_d_f(deformation_gradient)
        return value

    @ti.func
    def d2_psi_div_d2_f_at(self, particle_id, material_id, deformation_gradient):
        value = ti.Matrix.zero(float, 9, 9)
        if ti.static(self.is_finite_strain_plastic):
            value = self.model.total_d2_psi_div_d2_f_at(particle_id, deformation_gradient)
        else:
            value = self.model.d2_psi_div_d2_f(deformation_gradient)
        return value

    @ti.func
    def commit_soft_particle_state(self, particle_id, material_id, deformation_gradient, state_vars):
        committed = ti.Matrix.zero(float, 3, 3)
        stress = ti.Matrix.zero(float, 3, 3)
        if ti.static(self.is_finite_strain_plastic):
            committed = self.model.commit_total_state(particle_id, deformation_gradient)
            stress = self.model.total_first_piola_stress_at(particle_id, committed)
        else:
            committed = deformation_gradient
            stress = self.model.soft_particle_pk1(deformation_gradient)
        self.model.update_soft_particle_state(particle_id, committed, stress, state_vars)
        return committed, stress
