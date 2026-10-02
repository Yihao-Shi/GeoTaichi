import taichi as ti

from src.physics_model.consititutive_model.finite_strain.MaterialKernel import *
from src.physics_model.consititutive_model.finite_strain.FiniteStrainModel import FiniteStrainModel
from src.utils.constants import Threshold
import src.utils.GlobalVariable as GlobalVariable
from src.utils.MatrixFunction import matrix_form_3d_from_stress
from src.utils.TypeDefination import mat3x3
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class MooneyRivlin(FiniteStrainModel):
    def __init__(self, material_type="Solid", configuration="TL", solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.coeff = []
        self.is_elastic = True

    def model_initialize(self, material):
        self.material = material
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        coeff = DictIO.GetEssential(material, 'Coefficient')
        if (
            len(list(coeff)) != 2
            or any(len(list(row)) != 2 for row in coeff)
        ):
            raise ValueError("The dimension of /Exponent/ must be 2.")
        self.validate_elastic_parameters(density, young, poisson)
        self.add_material(density, young, poisson, coeff)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson, coeff):
        self.density = density
        self.young = young
        self.poisson = poisson
        self.shear = 0.5 * self.young / (1. + self.poisson)
        self.bulk = self.young / (3. * (1 - 2. * self.poisson))
        self.coeff = ti.Matrix(coeff)
        self.max_sound_speed = self.get_sound_speed(
            self.density, self.young, self.poisson
        )
        
    def print_message(self, materialID):
        self.print_console_header()
        print('Constitutive model: Mooney-Rivlin')
        print("Material ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Poisson Ratio: ', self.poisson)
        print('Coefficient: ', self.coeff, '\n')

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
        assert det_F > 0., "Mooney-Rivlin requires det(F) > 0"
        I1, I2 = getI1(deformation_gradient), getI2(
            deformation_gradient
        )
        energy = 0.
        for i in ti.static(range(self.coeff.n)):
            for j in ti.static(range(self.coeff.m)):
                energy += (
                    self.coeff[i, j]
                    * (I1 - 3.) ** i
                    * (I2 - 3.) ** j
                )
        log_j = ti.log(det_F)
        la = self.lame_lambda
        return (
            energy
            + 0.5 * la * log_j * log_j
            - self.shear * log_j
        )

    @ti.func
    def _invariant_energy_derivatives(self, deformation_gradient):
        det_F = deformation_gradient.determinant()
        assert det_F > 0., "Mooney-Rivlin requires det(F) > 0"
        I1, I2 = getI1(deformation_gradient), getI2(deformation_gradient)
        dUdI1, dUdI2 = 0., 0.
        d2UdI1I1, d2UdI1I2, d2UdI2I2 = 0., 0., 0.
        for i in ti.static(range(1, self.coeff.n)):
            for j in ti.static(range(self.coeff.m)):
                dUdI1 += i * self.coeff[i, j] * (I1 - 3.) ** (i - 1) * (I2 - 3.) ** j
        for i in ti.static(range(2, self.coeff.n)):
            for j in ti.static(range(self.coeff.m)):
                d2UdI1I1 += i * (i - 1) * self.coeff[i, j] * (I1 - 3.) ** (i - 2) * (I2 - 3.) ** j

        for i in ti.static(range(self.coeff.n)):
            for j in ti.static(range(1, self.coeff.m)):
                dUdI2 += j * self.coeff[i, j] * (I1 - 3.) ** i * (I2 - 3.) ** (j - 1)
        for i in ti.static(range(self.coeff.n)):
            for j in ti.static(range(2, self.coeff.m)):
                d2UdI2I2 += j * (j - 1) * self.coeff[i, j] * (I1 - 3.) ** i * (I2 - 3.) ** (j - 2)
        for i in ti.static(range(1, self.coeff.n)):
            for j in ti.static(range(1, self.coeff.m)):
                d2UdI1I2 += i * j * self.coeff[i, j] * (I1 - 3.) ** (i - 1) * (I2 - 3.) ** (j - 1)

        la = self.lame_lambda
        log_j = ti.log(det_F)
        dUdJ = (la * log_j - self.shear) / det_F
        d2UdJJ = (
            la + self.shear - la * log_j
        ) / (det_F * det_F)
        return (
            dUdI1,
            dUdI2,
            dUdJ,
            d2UdI1I1,
            d2UdI1I2,
            0.,
            d2UdI2I2,
            0.,
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
