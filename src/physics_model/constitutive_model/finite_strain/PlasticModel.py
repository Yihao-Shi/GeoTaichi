from src.physics_model.constitutive_model.finite_strain.FiniteStrainModel import FiniteStrainModel

import taichi as ti
from src.utils.TypeDefinition import vec3f
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class PlasticModel(FiniteStrainModel):

    @ti.func
    def core(self, np, state_variables):
        trial_deformation_gradient = state_variables[np].deformation_gradient
        matrix_u, sigma, matrix_vt = ti.svd(trial_deformation_gradient)
        hencky_strain = ti.log(
            vec3f(ti.max(1e-4, ti.abs(sigma[0])), ti.max(1e-4, ti.abs(sigma[1])), ti.max(1e-4, ti.abs(sigma[2])))
        )
        hencky_trace_trace = hencky_strain[0] + hencky_strain[1] + hencky_strain[2]
        hencky_deviatoric = hencky_strain - (hencky_trace_trace / GlobalVariable.DIMENSION) * ti.Vector.one(
            float, GlobalVariable.DIMENSION
        )
        hencky_deviatoric_norm = hencky_deviatoric.norm()
        if hencky_deviatoric_norm > 0.0:
            hencky_deviatoric /= hencky_deviatoric_norm
        return self.plastic_process(
            matrix_u,
            matrix_vt,
            hencky_strain,
            hencky_trace_trace,
            hencky_deviatoric,
            hencky_deviatoric_norm,
            state_variables[np],
        )

    @ti.func
    def plastic_process(
        self,
        matrix_u,
        matrix_vt,
        hencky_strain,
        hencky_trace_trace,
        hencky_deviatoric,
        hencky_deviatoric_norm,
        state_vars,
    ):
        raise NotImplementedError
