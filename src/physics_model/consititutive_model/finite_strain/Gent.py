import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.finite_strain.MaterialKernel import *
from src.physics_model.consititutive_model.finite_strain.FiniteStrainModel import FiniteStrainModel
from src.utils.constants import Threshold
import src.utils.GlobalVariable as GlobalVariable
from src.utils.TypeDefination import mat3x3
from src.utils.MatrixFunction import matrix_form_3d_from_stress
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class Gent(FiniteStrainModel):
    def __init__(self, material_type="Solid", configuration="TL", solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.Jm1 = 0.
        self.Jm2 = 0.
        self.is_elastic = True

    def model_initialize(self, material):
        self.material = material
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        Jm1 = DictIO.GetEssential(material, 'Tensile1')
        Jm2 = DictIO.GetAlternative(material, 'Tensile2', 0.)
        self.validate_elastic_parameters(density, young, poisson)
        if not np.isfinite(Jm1) or Jm1 <= 0.:
            raise ValueError(
                f"Tensile1 must be finite and positive, got {Jm1!r}"
            )
        self.validate_nonnegative_parameters(Tensile2=Jm2)
        self.add_material(density, young, poisson, Jm1, Jm2)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson, Jm1, Jm2):
        self.density = density
        self.young = young
        self.poisson = poisson
        self.shear = 0.5 * self.young / (1. + self.poisson)
        self.bulk = self.young / (3. * (1 - 2. * self.poisson))
        self.Jm1 = Jm1
        self.Jm2 = Jm2
        self.max_sound_speed = self.get_sound_speed(
            self.density, self.young, self.poisson
        )
        
    def print_message(self, materialID):
        self.print_console_header()
        print('Constitutive model: Gent')
        print("Material ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Poisson Ratio: ', self.poisson)
        print('Finite Tensile: ', self.Jm1, self.Jm2, '\n')

    def define_state_vars(self):
        return {'stress0': mat3x3, 'deformation_gradient': ti.types.matrix(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION, float)}

    @ti.func
    def _initialize_vars_(self, np, particle, stateVars):
        stateVars[np].deformation_gradient = ti.Matrix.identity(float, stateVars[np].deformation_gradient.n)    
        stress = particle[np].stress
        stateVars[np].stress0 = matrix_form_3d_from_stress(stress)

    @ti.func
    def strain_energy_density(self, deformation_gradient):
        det_F = deformation_gradient.determinant()
        assert det_F > 0., "Gent requires det(F) > 0"
        I1 = getI1(deformation_gradient)
        I2 = getI2(deformation_gradient)
        I1_bar = det_F ** (-2. / 3.) * I1
        I2_bar = det_F ** (-4. / 3.) * I2
        energy = 0.
        if ti.static(self.Jm1 > Threshold):
            denominator1 = self.Jm1 - I1_bar + 3.
            assert denominator1 > 0., "Gent I1 locking limit exceeded"
            energy -= (
                0.5
                * self.shear
                * self.Jm1
                * ti.log(denominator1 / self.Jm1)
            )
        if ti.static(self.Jm2 > Threshold):
            denominator2 = self.Jm2 - I2_bar + 3.
            assert denominator2 > 0., "Gent I2 locking limit exceeded"
            energy -= (
                0.5
                * self.shear
                * self.Jm2
                * ti.log(denominator2 / self.Jm2)
            )
        la = self.lame_lambda
        return (
            energy
            + 0.25 * la * (det_F * det_F - 1.)
            - 0.5 * la * ti.log(det_F)
        )

    @ti.func
    def _invariant_energy_derivatives(self, deformation_gradient):
        det_F = deformation_gradient.determinant()
        assert det_F > 0., "Gent requires det(F) > 0"
        I1 = getI1(deformation_gradient)
        I2 = getI2(deformation_gradient)
        det_dev1 = det_F ** (-2. / 3.)
        det_dev2 = det_F ** (-4. / 3.)
        I1_bar = det_dev1 * I1
        I2_bar = det_dev2 * I2
        dUdI1, dUdI2, dUdJ = 0., 0., 0.
        d2UdI1I1, d2UdI2I2 = 0., 0.
        d2UdI1J, d2UdI2J, d2UdJJ = 0., 0., 0.
        if ti.static(self.Jm1 > Threshold):
            denominator1 = self.Jm1 - I1_bar + 3.
            assert denominator1 > 0., "Gent I1 locking limit exceeded"
            first_derivative1 = (
                0.5 * self.shear * self.Jm1 / denominator1
            )
            second_derivative1 = (
                0.5
                * self.shear
                * self.Jm1
                / (denominator1 * denominator1)
            )
            exponent1 = -2. / 3.
            q1_j = exponent1 * I1_bar / det_F
            dUdI1 += first_derivative1 * det_dev1
            dUdJ += first_derivative1 * q1_j
            d2UdI1I1 += second_derivative1 * det_dev1 * det_dev1
            d2UdI1J += (
                exponent1
                * det_dev1
                / det_F
                * (
                    second_derivative1 * I1_bar
                    + first_derivative1
                )
            )
            d2UdJJ += (
                second_derivative1 * q1_j * q1_j
                + first_derivative1
                * exponent1
                * (exponent1 - 1.)
                * I1_bar
                / (det_F * det_F)
            )
        if ti.static(self.Jm2 > Threshold):
            denominator2 = self.Jm2 - I2_bar + 3.
            assert denominator2 > 0., "Gent I2 locking limit exceeded"
            first_derivative2 = (
                0.5 * self.shear * self.Jm2 / denominator2
            )
            second_derivative2 = (
                0.5
                * self.shear
                * self.Jm2
                / (denominator2 * denominator2)
            )
            exponent2 = -4. / 3.
            q2_j = exponent2 * I2_bar / det_F
            dUdI2 += first_derivative2 * det_dev2
            dUdJ += first_derivative2 * q2_j
            d2UdI2I2 += second_derivative2 * det_dev2 * det_dev2
            d2UdI2J += (
                exponent2
                * det_dev2
                / det_F
                * (
                    second_derivative2 * I2_bar
                    + first_derivative2
                )
            )
            d2UdJJ += (
                second_derivative2 * q2_j * q2_j
                + first_derivative2
                * exponent2
                * (exponent2 - 1.)
                * I2_bar
                / (det_F * det_F)
            )
        la = self.lame_lambda
        dUdJ += 0.5 * la * (det_F - 1. / det_F)
        d2UdJJ += 0.5 * la * (
            1. + 1. / (det_F * det_F)
        )
        return (
            dUdI1,
            dUdI2,
            dUdJ,
            d2UdI1I1,
            0.,
            d2UdI1J,
            d2UdI2I2,
            d2UdI2J,
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
        return getPK1(dUdI1, dUdI2, dUdJ, deformation_gradient)

    @ti.func
    def first_piola_tangent(self, deformation_gradient):
        derivatives = self._invariant_energy_derivatives(
            deformation_gradient
        )
        return get_invariant_hessian(
            deformation_gradient, *derivatives
        )
