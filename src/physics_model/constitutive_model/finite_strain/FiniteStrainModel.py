import taichi as ti
from src.physics_model.constitutive_model.Solid import Solid
from src.physics_model.constitutive_model.finite_strain.MaterialKernel import f3d
from src.utils.constants import DELTA, DELTA2D, THRESHOLD
from src.utils.MatrixFunction import flatten_matrix
from src.utils.TypeDefinition import mat2x2


@ti.data_oriented
class FiniteStrainModel(Solid):
    def __init__(self, material_type, configuration, solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.young = 0.0
        self.shear = 0.0
        self.bulk = 0.0
        self.young_modulus = 0.0
        self.poisson_ratio = 0.0
        self.mu_ = 0.0
        self.lambda_ = 0.0

    def initialize_from_kwargs(self, **kwargs):
        material = self._material_dict_from_kwargs(kwargs)
        self.model_initialize(material)
        self.young_modulus = self.young
        self.poisson_ratio = self.poisson
        self.mu_ = self.shear
        self.lambda_ = self.lame_lambda
        return self

    def _material_dict_from_kwargs(self, kwargs):
        material_parameters = kwargs.get("material_parameters")
        material = dict(material_parameters) if isinstance(material_parameters, dict) else {}
        material.update(kwargs)
        aliases = {
            "Density": ("density", "Density"),
            "YoungModulus": ("young_modulus", "YoungModulus", "ElasticModulus"),
            "PoissonRatio": ("poisson_ratio", "PoissonRatio"),
            "Thickness": ("thickness", "Thickness"),
            "MinimumJacobian": ("minimum_jacobian", "MinimumJacobian"),
        }
        for target, keys in aliases.items():
            if target in material:
                continue
            for key in keys:
                if key in kwargs:
                    material[target] = kwargs[key]
                    break
        if "Density" not in material:
            material["Density"] = 2650
        return material

    @property
    def lame_lambda(self):
        return 3.0 * self.bulk * self.poisson / (1.0 + self.poisson) if (1.0 + self.poisson) != 0.0 else 0.0

    def get_state_vars(self):
        self._initialize_vars = self._initialize_vars
        return self.define_state_vars()

    def define_soft_particle_state_vars(self):
        return {"estress": float}

    @ti.func
    def _initialize_vars(self, np, particle, state_variables):
        raise NotImplementedError

    @ti.func
    def strain_energy_density(self, deformation_gradient):
        raise NotImplementedError

    @ti.func
    def first_piola_stress(self, deformation_gradient):
        raise NotImplementedError

    @ti.func
    def first_piola_tangent(self, deformation_gradient):
        raise NotImplementedError

    @ti.func
    def soft_particle_pk1(self, deformation_gradient):
        return self.first_piola_stress(deformation_gradient)

    @ti.func
    def von_mises(self, deformation_gradient):
        pk1_stress = self.first_piola_stress(deformation_gradient)
        return self.von_mises_from_pk1(deformation_gradient, pk1_stress)

    @ti.func
    def psi(self, deformation_gradient):
        return self.strain_energy_density(deformation_gradient)

    @ti.func
    def d_psi_div_d_f(self, deformation_gradient):
        return flatten_matrix(self.first_piola_stress(deformation_gradient))

    @ti.func
    def d2_psi_div_d2_f(self, deformation_gradient):
        return self.first_piola_tangent(deformation_gradient)

    @ti.func
    def d_psi_div_dx(self, deformation_gradient, dF_dx):
        return dF_dx @ self.d_psi_div_d_f(deformation_gradient)

    @ti.func
    def d2_psi_div_d2x(self, deformation_gradient, dF_dx):
        return dF_dx @ self.d2_psi_div_d2_f(deformation_gradient) @ dF_dx.transpose()

    @ti.func
    def von_mises_from_pk1(self, deformation_gradient, pk1_stress):
        jacobian = ti.max(deformation_gradient.determinant(), THRESHOLD)
        cauchy_stress = pk1_stress @ deformation_gradient.transpose() / jacobian
        von_mises = 0.0
        if ti.static(deformation_gradient.n == 3):
            von_mises = ti.sqrt(
                ti.max(
                    0.5
                    * (
                        (cauchy_stress[0, 0] - cauchy_stress[1, 1]) ** 2
                        + (cauchy_stress[1, 1] - cauchy_stress[2, 2]) ** 2
                        + (cauchy_stress[0, 0] - cauchy_stress[2, 2]) ** 2
                    )
                    + 3.0 * (cauchy_stress[0, 1] ** 2 + cauchy_stress[1, 2] ** 2 + cauchy_stress[0, 2] ** 2),
                    0.0,
                )
            )
        else:
            mean_stress = 0.5 * (cauchy_stress[0, 0] + cauchy_stress[1, 1])
            s00 = cauchy_stress[0, 0] - mean_stress
            s11 = cauchy_stress[1, 1] - mean_stress
            s01 = cauchy_stress[0, 1]
            von_mises = ti.sqrt(ti.max(1.5 * (s00 * s00 + s11 * s11 + 2.0 * s01 * s01), 0.0))
        return von_mises

    @ti.func
    def soft_particle_von_mises(self, deformation_gradient, pk1_stress):
        return self.von_mises_from_pk1(deformation_gradient, pk1_stress)

    @ti.func
    def update_soft_particle_state(self, np, deformation_gradient, pk1_stress, state_variables):
        state_variables[np].estress = self.soft_particle_von_mises(deformation_gradient, pk1_stress)

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, state_variables, dt):
        deformation_gradient_rate = DELTA + velocity_gradient * dt[None]
        state_variables[np].deformation_gradient = deformation_gradient_rate @ state_variables[np].deformation_gradient
        return deformation_gradient_rate.determinant()

    @ti.func
    def update_particle_volume_2d(self, np, velocity_gradient, state_variables, dt):
        deformation_gradient_rate = DELTA2D + velocity_gradient * dt[None]
        state_variables[np].deformation_gradient = deformation_gradient_rate @ state_variables[np].deformation_gradient
        return deformation_gradient_rate.determinant()

    @ti.func
    def compute_stress2d(self, np, previous_cauchy_stress, velocity_gradient, state_variables, dt):
        # Keep the out-of-plane stress in the plane-strain Cauchy response.
        PKstress = self.core_pk(np, state_variables)
        return self.pk2_cauchy_stress2d(np, PKstress, state_variables)

    @ti.func
    def compute_stress(self, np, previous_cauchy_stress, velocity_gradient, state_variables, dt):
        previous_PKstress = self.cauchy2_pk_stress(np, state_variables, previous_cauchy_stress)
        PKstress = self.compute_pk_stress(np, previous_PKstress, velocity_gradient, state_variables, dt)
        return self.pk2_cauchy_stress(np, PKstress, state_variables)

    @ti.func
    def compute_pk_stress2d(self, np, presvious_stress, velocity_gradient, state_variables, dt):
        PKstress = self.core_pk(np, state_variables)
        return mat2x2([[PKstress[0, 0], PKstress[0, 1]], [PKstress[1, 0], PKstress[1, 1]]])

    @ti.func
    def compute_pk_stress(self, np, presvious_stress, velocity_gradient, state_variables, dt):
        PKstress = self.core_pk(np, state_variables)
        return PKstress

    @ti.func
    def core_pk(self, np, state_variables):
        return state_variables[np].stress0 + self.first_piola_stress(f3d(state_variables[np].deformation_gradient))

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, state_variables):
        lambda_ = self.lame_lambda
        factor = lambda_ + 2.0 * self.shear
        stiffness = ti.Matrix.zero(float, 6, 6)
        stiffness[0, 0] = stiffness[1, 1] = stiffness[2, 2] = factor
        stiffness[0, 1] = stiffness[0, 2] = stiffness[1, 2] = lambda_
        stiffness[1, 0] = stiffness[2, 0] = stiffness[2, 1] = lambda_
        stiffness[3, 3] = stiffness[4, 4] = stiffness[5, 5] = self.shear
        return stiffness

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, state_variables):
        return self.compute_elastic_tensor(np, current_stress, state_variables)
