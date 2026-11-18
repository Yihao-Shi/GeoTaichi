import taichi as ti

from src.physics_model.constitutive_model.infinitesimal_strain.MaterialKernel import *
import src.utils.GlobalVariable as GlobalVariable
from src.physics_model.constitutive_model.Solid import Solid


@ti.data_oriented
class InfinitesimalStrainModel(Solid):
    def __init__(self, material_type, configuration, solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)

    def get_state_vars(self):
        state_vars = self.define_state_vars()
        if getattr(self, "is_rate_dependent", False):
            state_vars.update({"strain_rate": float, "rate_pressure": float})
        if GlobalVariable.TRACKENERGY:
            state_vars.update({"plastic_energy": float})
        if self.is_elastic is False and self.solver_type == 1:
            state_vars.update({"yield_state": ti.u8})
        if "UL" in self.configuration:
            self._initialize_vars = self._initialize_vars_update_lagrangian
            return state_vars
        elif "TL" in self.configuration:
            self._initialize_vars = self._initialize_vars_total_lagrangian
            state_vars.update(
                {"deformation_gradient": ti.types.matrix(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION, float)}
            )
            return state_vars

    @ti.func
    def _initialize_vars_update_lagrangian(np, particle, state_variables):
        raise NotImplementedError

    @ti.func
    def _initialize_vars_total_lagrangian(self, np, particle, state_variables):
        self._initialize_vars_update_lagrangian(np, particle, state_variables)
        state_variables[np].deformation_gradient = ti.Matrix.identity(float, GlobalVariable.DIMENSION)

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, state_variables, dt):
        return (ti.Matrix.identity(float, 3) + velocity_gradient * dt[None]).determinant()

    @ti.func
    def update_particle_volume_2d(self, np, velocity_gradient, state_variables, dt):
        return (ti.Matrix.identity(float, 2) + velocity_gradient * dt[None]).determinant()

    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, state_variables, dt):
        return 1.0 + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])

    @ti.func
    def update_particle_volume_bbar_2d(self, np, velocity_gradient, state_variables, dt):
        return (ti.Matrix.identity(float, 2) + velocity_gradient * dt[None]).determinant()

    @ti.func
    def compute_pk_stress2d(self, np, previous_PKstress, velocity_gradient, state_variables, dt):
        previous_cauchy_stress = self.pk2_cauchy_stress2d(np, previous_PKstress, state_variables)
        cauchy_stress = self.compute_stress2d(np, previous_cauchy_stress, velocity_gradient, state_variables, dt)
        PKstress2D = self.cauchy2_pk_stress2d(np, state_variables, cauchy_stress)
        if ti.static(previous_PKstress.n == 3):
            deformation_gradient = state_variables[np].deformation_gradient
            j = deformation_gradient.determinant()
            PKstress = ti.Matrix.zero(float, 3, 3)
            PKstress[0, 0] = PKstress2D[0, 0]
            PKstress[0, 1] = PKstress2D[0, 1]
            PKstress[1, 0] = PKstress2D[1, 0]
            PKstress[1, 1] = PKstress2D[1, 1]
            PKstress[2, 2] = cauchy_stress[2] * j
            return PKstress
        else:
            return PKstress2D

    @ti.func
    def compute_pk_stress(self, np, previous_PKstress, velocity_gradient, state_variables, dt):
        previous_cauchy_stress = self.pk2_cauchy_stress(np, previous_PKstress, state_variables)
        cauchy_stress = self.compute_stress(np, previous_cauchy_stress, velocity_gradient, state_variables, dt)
        PKstress = self.cauchy2_pk_stress(np, state_variables, cauchy_stress)
        return PKstress

    @ti.func
    def compute_stress2d(self, np, previous_stress, velocity_gradient, state_variables, dt):
        de = calculate_strain_increment2d(velocity_gradient, dt)
        dw = calculate_vorticity_increment2d(velocity_gradient, dt)
        if ti.static(getattr(self, "is_rate_dependent", False)):
            return self.integrate_rate_dependent(np, previous_stress, de, dw, state_variables, dt)
        else:
            return self.core(np, previous_stress, de, dw, state_variables)

    @ti.func
    def compute_stress(self, np, previous_stress, velocity_gradient, state_variables, dt):
        de = calculate_strain_increment(velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)
        if ti.static(getattr(self, "is_rate_dependent", False)):
            return self.integrate_rate_dependent(np, previous_stress, de, dw, state_variables, dt)
        else:
            return self.core(np, previous_stress, de, dw, state_variables)
