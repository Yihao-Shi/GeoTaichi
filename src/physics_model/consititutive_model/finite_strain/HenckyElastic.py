import taichi as ti

from src.physics_model.consititutive_model.finite_strain.FiniteStrainModel import FiniteStrainModel
from src.physics_model.consititutive_model.finite_strain.MaterialKernel import *
import src.utils.GlobalVariable as GlobalVariable
from src.utils.MatrixFunction import matrix_form_3d_from_stress
from src.utils.TypeDefination import mat3x3
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class HenckyElasticModel(FiniteStrainModel):
    def __init__(self, material_type="Solid", configuration="TL", solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.is_elastic = True

    def model_initialize(self, material):
        self.material = material
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        self.validate_elastic_parameters(density, young, poisson)
        self.add_material(density, young, poisson)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson):
        self.density = density
        self.young = young
        self.poisson = poisson
        self.shear = 0.5 * self.young / (1. + self.poisson)
        self.bulk = self.young / (3. * (1 - 2. * self.poisson))
        self.max_sound_speed = self.get_sound_speed(
            self.density, self.young, self.poisson
        )

    def print_message(self, materialID):
        self.print_console_header()
        print('Constitutive model: Hencky Elastic')
        print("Material ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Poisson Ratio: ', self.poisson, '\n')

    def define_state_vars(self):
        return {'stress0': mat3x3, 'deformation_gradient': ti.types.matrix(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION, float)}

    @ti.func
    def _initialize_vars_(self, np, particle, stateVars):
        stateVars[np].deformation_gradient = ti.Matrix.identity(float, stateVars[np].deformation_gradient.n)    
        stress = particle[np].stress
        stateVars[np].stress0 = matrix_form_3d_from_stress(stress)
    
    @ti.func
    def elastic_part(self):
        a1 = self.bulk + (4./3.) * self.shear
        a2 = self.bulk - (2./3.) * self.shear
        return mat3x3([[a1, a2, a2], [a2, a1, a2], [a2, a2, a1]])

    @ti.func
    def strain_energy_density(self, deformation_gradient):
        det_F = deformation_gradient.determinant()
        assert det_F > 0., "Hencky elasticity requires det(F) > 0"
        _, singular_values, _ = ti.svd(deformation_gradient)
        trace_strain = 0.
        squared_norm = 0.
        for i in ti.static(range(deformation_gradient.n)):
            principal_strain = ti.log(singular_values[i, i])
            trace_strain += principal_strain
            squared_norm += principal_strain * principal_strain
        return (
            self.shear * squared_norm
            + 0.5 * self.lame_lambda * trace_strain * trace_strain
        )
    
    @ti.func
    def first_piola_stress(self, deformation_gradient):
        det_F = deformation_gradient.determinant()
        assert det_F > 0., "Hencky elasticity requires det(F) > 0"
        matrix_u, singular_matrix, matrix_v = ti.svd(
            deformation_gradient
        )
        log_j = 0.
        for i in ti.static(range(deformation_gradient.n)):
            log_j += ti.log(singular_matrix[i, i])
        principal_pk1 = ti.Matrix.zero(
            float,
            deformation_gradient.n,
            deformation_gradient.m,
        )
        for i in ti.static(range(deformation_gradient.n)):
            singular_value = singular_matrix[i, i]
            principal_kirchhoff = (
                2. * self.shear * ti.log(singular_value)
                + self.lame_lambda * log_j
            )
            principal_pk1[i, i] = (
                principal_kirchhoff / singular_value
            )
        return matrix_u @ principal_pk1 @ matrix_v.transpose()

    @ti.func
    def first_piola_tangent(self, deformation_gradient):
        """Exact spectral derivative of quadratic Hencky energy.

        The repeated-singular-value branch evaluates the analytic divided-
        difference limit.  It is not a finite-difference regularization.
        """
        det_F = deformation_gradient.determinant()
        assert det_F > 0., "Hencky elasticity requires det(F) > 0"
        matrix_u, singular_matrix, matrix_v = ti.svd(
            deformation_gradient
        )
        dimension = ti.static(deformation_gradient.n)
        singular_values = ti.Vector.zero(float, dimension)
        principal_kirchhoff = ti.Vector.zero(float, dimension)
        principal_pk1 = ti.Vector.zero(float, dimension)
        log_j = 0.
        for i in ti.static(range(dimension)):
            singular_values[i] = singular_matrix[i, i]
            log_j += ti.log(singular_values[i])
        for i in ti.static(range(dimension)):
            principal_kirchhoff[i] = (
                2. * self.shear * ti.log(singular_values[i])
                + self.lame_lambda * log_j
            )
            principal_pk1[i] = (
                principal_kirchhoff[i] / singular_values[i]
            )

        principal_jacobian = ti.Matrix.zero(
            float, dimension, dimension
        )
        for i in ti.static(range(dimension)):
            for j in ti.static(range(dimension)):
                principal_jacobian[i, j] = (
                    self.lame_lambda
                    / (singular_values[i] * singular_values[j])
                )
                if ti.static(i == j):
                    principal_jacobian[i, j] += (
                        2. * self.shear - principal_kirchhoff[i]
                    ) / (singular_values[i] * singular_values[i])

        size = ti.static(dimension * dimension)
        tangent = ti.Matrix.zero(float, size, size)
        spectral_column = 0
        while spectral_column < size:
            b = spectral_column // dimension
            j = spectral_column - b * dimension
            transformed_direction = ti.Matrix.zero(
                float, dimension, dimension
            )
            for r in ti.static(range(dimension)):
                for s in ti.static(range(dimension)):
                    transformed_direction[r, s] = (
                        matrix_u[j, r] * matrix_v[b, s]
                    )

            transformed_response = ti.Matrix.zero(
                float, dimension, dimension
            )
            for r in ti.static(range(dimension)):
                for s in ti.static(range(dimension)):
                    transformed_response[r, r] += (
                        principal_jacobian[r, s]
                        * transformed_direction[s, s]
                    )
            for r in ti.static(range(dimension)):
                for s in ti.static(range(dimension)):
                    if ti.static(r != s):
                        difference_quotient = 0.
                        if singular_values[r] == singular_values[s]:
                            difference_quotient = (
                                2. * self.shear
                                - principal_kirchhoff[r]
                            ) / (
                                singular_values[r]
                                * singular_values[r]
                            )
                        else:
                            difference_quotient = (
                                principal_pk1[r]
                                - principal_pk1[s]
                            ) / (
                                singular_values[r]
                                - singular_values[s]
                            )
                        sum_quotient = (
                            principal_pk1[r] + principal_pk1[s]
                        ) / (
                            singular_values[r]
                            + singular_values[s]
                        )
                        transformed_response[r, s] = 0.5 * (
                            (
                                difference_quotient
                                + sum_quotient
                            )
                            * transformed_direction[r, s]
                            + (
                                difference_quotient
                                - sum_quotient
                            )
                            * transformed_direction[s, r]
                        )

            for a in ti.static(range(dimension)):
                for i in ti.static(range(dimension)):
                    row = a * dimension + i
                    value = 0.
                    contraction_entry = 0
                    while contraction_entry < size:
                        r = contraction_entry // dimension
                        s = contraction_entry - r * dimension
                        value += (
                            matrix_u[i, r]
                            * transformed_response[r, s]
                            * matrix_v[a, s]
                        )
                        contraction_entry += 1
                    tangent[row, spectral_column] = value
            spectral_column += 1
        return tangent
