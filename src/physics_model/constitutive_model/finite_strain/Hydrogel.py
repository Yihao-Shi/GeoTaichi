import taichi as ti
import numpy as np

from src.physics_model.constitutive_model.finite_strain.MaterialKernel import *
from src.physics_model.constitutive_model.finite_strain.FiniteStrainModel import FiniteStrainModel
from src.utils.constants import THRESHOLD
import src.utils.GlobalVariable as GlobalVariable
from src.utils.TypeDefinition import mat3x3
from src.utils.MatrixFunction import matrix_form_3d_from_stress
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class Hydrogel(FiniteStrainModel):
    def __init__(self, material_type="Solid", configuration="TL", solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.Jm = 0.0
        self.is_elastic = True

    def model_initialize(self, material):
        self.material = material
        density = DictIO.get_alternative(material, "Density", 2650)
        young = DictIO.get_essential(material, "YoungModulus")
        poisson = DictIO.get_alternative(material, "PoissonRatio", 0.3)
        Jm = DictIO.get_essential(material, "Tensile")
        self.validate_elastic_parameters(density, young, poisson)
        if not np.isfinite(Jm) or Jm <= 0.0:
            raise ValueError(f"Tensile must be finite and positive, got {Jm!r}")
        self.add_material(density, young, poisson, Jm)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson, Jm):
        self.density = density
        self.young = young
        self.poisson = poisson
        self.shear = 0.5 * self.young / (1.0 + self.poisson)
        self.bulk = self.young / (3.0 * (1 - 2.0 * self.poisson))
        self.Jm = Jm
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def print_message(self, materialID):
        self.print_console_header()
        print("Constitutive model: Hydrogel")
        print("Material ID: ", materialID)
        print("Density: ", self.density)
        print("Young Modulus: ", self.young)
        print("Poisson Ratio: ", self.poisson)
        print("Finite Tensile: ", self.Jm, "\n")

    def define_state_vars(self):
        return {
            "stress0": mat3x3,
            "deformation_gradient": ti.types.matrix(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION, float),
        }

    @ti.func
    def _initialize_vars(self, np, particle, state_variables):
        state_variables[np].deformation_gradient = ti.Matrix.identity(float, state_variables[np].deformation_gradient.n)
        stress = particle[np].stress
        state_variables[np].stress0 = matrix_form_3d_from_stress(stress)

    @ti.func
    def strain_energy_density(self, deformation_gradient):
        det_F = deformation_gradient.determinant()
        assert det_F > 0.0, "Hydrogel requires det(F) > 0"
        I1_bar = det_F ** (-2.0 / 3.0) * get_i1(deformation_gradient)
        denominator = self.Jm - I1_bar + 3.0
        assert denominator > 0.0, "Hydrogel locking limit exceeded"
        la = self.lame_lambda
        return (
            -0.5 * self.shear * self.Jm * ti.log(denominator / self.Jm)
            + 0.25 * la * (det_F * det_F - 1.0)
            - 0.5 * la * ti.log(det_F)
        )

    @ti.func
    def _invariant_energy_derivatives(self, deformation_gradient):
        det_F = deformation_gradient.determinant()
        assert det_F > 0.0, "Hydrogel requires det(F) > 0"
        I1 = get_i1(deformation_gradient)
        det_dev = det_F ** (-2.0 / 3.0)
        I1_bar = I1 * det_dev
        denominator = self.Jm - I1_bar + 3.0
        assert denominator > 0.0, "Hydrogel locking limit exceeded"
        first_derivative = 0.5 * self.shear * self.Jm / denominator
        second_derivative = 0.5 * self.shear * self.Jm / (denominator * denominator)
        exponent = -2.0 / 3.0
        q_j = exponent * I1_bar / det_F
        dUdI1 = first_derivative * det_dev
        dUdI2 = 0.0
        la = self.lame_lambda
        dUdJ = first_derivative * q_j + 0.5 * la * (det_F - 1.0 / det_F)
        d2UdI1I1 = second_derivative * det_dev * det_dev
        d2UdI1J = exponent * det_dev / det_F * (second_derivative * I1_bar + first_derivative)
        d2UdJJ = (
            second_derivative * q_j * q_j
            + first_derivative * exponent * (exponent - 1.0) * I1_bar / (det_F * det_F)
            + 0.5 * la * (1.0 + 1.0 / (det_F * det_F))
        )
        return (
            dUdI1,
            dUdI2,
            dUdJ,
            d2UdI1I1,
            0.0,
            d2UdI1J,
            0.0,
            0.0,
            d2UdJJ,
        )

    @ti.func
    def first_piola_stress(self, deformation_gradient):
        (
            dUdI1,
            dUdI2,
            dUdJ,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self._invariant_energy_derivatives(deformation_gradient)
        return get_pk1(dUdI1, dUdI2, dUdJ, deformation_gradient)

    @ti.func
    def first_piola_tangent(self, deformation_gradient):
        derivatives = self._invariant_energy_derivatives(deformation_gradient)
        return get_invariant_hessian(deformation_gradient, *derivatives)
