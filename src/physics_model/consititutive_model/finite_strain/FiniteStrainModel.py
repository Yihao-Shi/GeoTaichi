import taichi as ti

from src.physics_model.consititutive_model.MaterialModel import Solid
from src.physics_model.consititutive_model.finite_strain.MaterialKernel import F3d
from src.utils.constants import DELTA, DELTA2D, Threshold
from src.utils.MatrixFunction import flatten_matrix
from src.utils.TypeDefination import vec3f, mat2x2
import src.utils.GlobalVariable as GlobalVariable


class InvertedElementError(RuntimeError):
    """Raised when a finite-strain map is collapsed or orientation reversing."""


@ti.data_oriented
class FiniteStrainModel(Solid):
    def __init__(self, material_type, configuration, solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.young = 0.
        self.shear = 0.
        self.bulk = 0.
        self.young_modulus = 0.
        self.poisson_ratio = 0.
        self.mu_ = 0.
        self.lambda_ = 0.

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
        return 3. * self.bulk * self.poisson / (1. + self.poisson) if (1. + self.poisson) != 0. else 0.

    def get_state_vars(self):
        self._initialize_vars = self._initialize_vars_
        return self.define_state_vars()

    def define_soft_particle_state_vars(self):
        return {'estress': float}
    
    @ti.func
    def _initialize_vars_(self, np, particle, stateVars):
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
    def VonMises(self, deformation_gradient):
        pk1_stress = self.first_piola_stress(deformation_gradient)
        return self.von_mises_from_pk1(deformation_gradient, pk1_stress)

    @ti.func
    def Psi(self, deformation_gradient):
        return self.strain_energy_density(deformation_gradient)

    @ti.func
    def dPsi_div_dF(self, deformation_gradient):
        return flatten_matrix(self.first_piola_stress(deformation_gradient))

    @ti.func
    def d2Psi_div_d2F(self, deformation_gradient):
        return self.first_piola_tangent(deformation_gradient)

    @ti.func
    def dPsi_div_dx(self, deformation_gradient, dF_dx):
        return dF_dx @ self.dPsi_div_dF(deformation_gradient)

    @ti.func
    def d2Psi_div_d2x(self, deformation_gradient, dF_dx):
        return dF_dx @ self.d2Psi_div_d2F(deformation_gradient) @ dF_dx.transpose()

    @ti.func
    def von_mises_from_pk1(self, deformation_gradient, pk1_stress):
        jacobian = ti.max(deformation_gradient.determinant(), Threshold)
        cauchy_stress = pk1_stress @ deformation_gradient.transpose() / jacobian
        von_mises = 0.
        if ti.static(deformation_gradient.n == 3):
            von_mises = ti.sqrt(ti.max(0.5 * ((cauchy_stress[0, 0] - cauchy_stress[1, 1]) ** 2 +
                                              (cauchy_stress[1, 1] - cauchy_stress[2, 2]) ** 2 +
                                              (cauchy_stress[0, 0] - cauchy_stress[2, 2]) ** 2) +
                                       3. * (cauchy_stress[0, 1] ** 2 +
                                             cauchy_stress[1, 2] ** 2 +
                                             cauchy_stress[0, 2] ** 2), 0.))
        else:
            mean_stress = 0.5 * (cauchy_stress[0, 0] + cauchy_stress[1, 1])
            s00 = cauchy_stress[0, 0] - mean_stress
            s11 = cauchy_stress[1, 1] - mean_stress
            s01 = cauchy_stress[0, 1]
            von_mises = ti.sqrt(ti.max(1.5 * (s00 * s00 + s11 * s11 + 2. * s01 * s01), 0.))
        return von_mises

    @ti.func
    def soft_particle_von_mises(self, deformation_gradient, pk1_stress):
        return self.von_mises_from_pk1(deformation_gradient, pk1_stress)

    @ti.func
    def update_soft_particle_state(self, np, deformation_gradient, pk1_stress, stateVars):
        stateVars[np].estress = self.soft_particle_von_mises(deformation_gradient, pk1_stress)
    
    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        deformation_gradient_rate = DELTA + velocity_gradient * dt[None]
        stateVars[np].deformation_gradient = deformation_gradient_rate @ stateVars[np].deformation_gradient
        return deformation_gradient_rate.determinant()
    
    @ti.func
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        deformation_gradient_rate = DELTA2D + velocity_gradient * dt[None]
        stateVars[np].deformation_gradient = deformation_gradient_rate @ stateVars[np].deformation_gradient
        return deformation_gradient_rate.determinant()
    
    @ti.func
    def ComputeStress2D(self, np, previous_cauchy_stress, velocity_gradient, stateVars, dt):  
        previous_PKstress = self.Cauchy2PKStress2D(np, stateVars, previous_cauchy_stress)
        PKstress = self.ComputePKStress2D(np, previous_PKstress, velocity_gradient, stateVars, dt)
        return self.PK2CauchyStress2D(np, PKstress, stateVars)

    @ti.func
    def ComputeStress(self, np, previous_cauchy_stress, velocity_gradient, stateVars, dt):  
        previous_PKstress = self.Cauchy2PKStress(np, stateVars, previous_cauchy_stress)
        PKstress = self.ComputePKStress(np, previous_PKstress, velocity_gradient, stateVars, dt)
        return self.PK2CauchyStress(np, PKstress, stateVars)

    @ti.func
    def ComputePKStress2D(self, np, presvious_stress, velocity_gradient, stateVars, dt):  
        PKstress = self.corePK(np, stateVars)
        return mat2x2([[PKstress[0, 0], PKstress[0, 1]],
                       [PKstress[1, 0], PKstress[1, 1]]])

    @ti.func
    def ComputePKStress(self, np, presvious_stress, velocity_gradient, stateVars, dt):  
        PKstress = self.corePK(np, stateVars)
        return PKstress
    
    @ti.func
    def corePK(self, np, stateVars):  
        return stateVars[np].stress0 + F3d(self.first_piola_stress(stateVars[np].deformation_gradient))
    
    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        lambda_ = self.lame_lambda
        factor = lambda_ + 2. * self.shear
        stiffness = ti.Matrix.zero(float, 6, 6)
        stiffness[0, 0] = stiffness[1, 1] = stiffness[2, 2] = factor
        stiffness[0, 1] = stiffness[0, 2] = stiffness[1, 2] = lambda_
        stiffness[1, 0] = stiffness[2, 0] = stiffness[2, 1] = lambda_
        stiffness[3, 3] = stiffness[4, 4] = stiffness[5, 5] = self.shear
        return stiffness

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stateVars):
        return self.compute_elastic_tensor(np, current_stress, stateVars)


@ti.data_oriented
class ElasticModel(FiniteStrainModel):
    def __init__(self, material_type, configuration, solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)


@ti.data_oriented
class PlasticModel(ElasticModel):
    def __init__(self, material_type, configuration, solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)

    @ti.func
    def core(self, np, stateVars):
        trial_deformation_gradient = stateVars[np].deformation_gradient
        matrixU, sigma, matrixVT = ti.svd(trial_deformation_gradient)
        hencky_strain = ti.log(vec3f(ti.max(1e-4, ti.abs(sigma[0])), ti.max(1e-4, ti.abs(sigma[1])), ti.max(1e-4, ti.abs(sigma[2]))))
        hencky_trace_trace = hencky_strain[0] + hencky_strain[1] + hencky_strain[2]
        hencky_deviatoric = hencky_strain - (hencky_trace_trace / GlobalVariable.DIMENSION) * ti.Vector.one(float, GlobalVariable.DIMENSION)
        hencky_deviatoric_norm = hencky_deviatoric.norm()
        if hencky_deviatoric_norm > 0.: hencky_deviatoric /= hencky_deviatoric_norm
        return self.plastic_process(matrixU, matrixVT, hencky_strain, hencky_trace_trace, hencky_deviatoric, hencky_deviatoric_norm, stateVars[np])

    @ti.func
    def plastic_process(self, matrixU, matrixVT, hencky_strain, hencky_trace_trace, hencky_deviatoric, hencky_deviatoric_norm, state_vars):
        raise NotImplementedError
