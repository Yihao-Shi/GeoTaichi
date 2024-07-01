import numpy as np
import taichi as ti

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.MaterialKernel import *
from src.utils.constants import DELTA, FTOL, MAXITS
from src.utils.MatrixFunction import matrix_form
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import mat3x3
from src.utils.VectorFunction import voigt_form, equivalent_voigt, voigt_tensor_dot


class ElasticPerfectlyPlastic(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_particle_num, configuration="ULMPM", solver_type="Explicit"):
        super().__init__()
        self.matProps = ElasticPerfectlyPlasticModel.field(shape=max_material_num)
        if configuration == "ULMPM":
            self.stateVars = ULStateVariable.field(shape=max_particle_num) 
        elif configuration == "TLMPM":
            self.stateVars = TLStateVariable.field(shape=max_particle_num) 

        if solver_type == "Implicit":
            self.stiffness_matrix = ti.Matrix.field(6, 6, float, shape=max_particle_num)

    def get_state_vars_dict(self, start_particle, end_particle):
        epstrain = np.ascontiguousarray(self.stateVars.epstrain.to_numpy()[start_particle:end_particle])
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        return {'epstrain': epstrain, 'estress': estress}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        epstrain = state_vars.item()['epstrain']
        kernel_reload_state_variables(estress, epstrain, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID') 
        self.check_materialID(materialID, self.matProps.shape[0])
        
        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        possion = DictIO.GetAlternative(material, 'PossionRatio', 0.3)
        _yield = DictIO.GetEssential(material, 'YieldStress')

        self.matProps[materialID].add_material(density, young, possion, _yield)
        self.matProps[materialID].print_message(materialID)
    
    def get_lateral_coefficient(self, materialID):
        mu = self.matProps[materialID].possion
        return mu / (1. - mu)


@ti.dataclass
class ULStateVariable:
    epstrain: float
    estress: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.estress = VonMisesStress(stress)

    @ti.func
    def _update_vars(self, stress, epstrain):
        self.estress = VonMisesStress(stress)
        self.epstrain = epstrain

@ti.dataclass
class TLStateVariable:
    epstrain: float
    estress: float
    deformation_gradient: mat3x3
    stress: mat3x3

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        self.estress = VonMisesStress(stress)
        self.deformation_gradient = DELTA
        self.stress = matrix_form(stress)

    @ti.func
    def _update_deformation_gradient(self, deformation_gradient_rate, dt):
        self.deformation_gradient += deformation_gradient_rate * dt[None]

    @ti.func
    def _update_vars(self, stress, epstrain):
        self.estress = VonMisesStress(stress)
        self.epstrain = epstrain


@ti.dataclass
class ElasticPerfectlyPlasticModel:
    density: float
    young: float
    possion: float
    shear: float
    bulk: float
    _yield: float

    def add_material(self, density, young, possion, yield_stress):
        self.density = density
        self.young = young
        self.possion = possion

        self.shear = 0.5 * self.young / (1. + self.possion)
        self.bulk = self.young / (3. * (1 - 2. * self.possion))
        self._yield = yield_stress

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Elastic Perfectly Plastic Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Possion Ratio: ', self.possion)
        print('Yield Stress: ', self._yield, '\n')

    @ti.func
    def _get_sound_speed(self):
        sound_speed = 0.
        if self.density > 0.:
            sound_speed = ti.sqrt(self.young * (1 - self.possion) / (1 + self.possion) / (1 - 2 * self.possion) / self.density)
        return sound_speed
    
    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        return (DELTA + velocity_gradient * dt[None]).determinant()
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        return 1. + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])

    @ti.func
    def PK2CauchyStress(self, np, stateVars):
        inv_j = 1. / stateVars[np].deformation_gradient.determinant()
        return voigt_form(stateVars[np].stress @ stateVars[np].deformation_gradient.transpose() * inv_j)

    @ti.func
    def Cauchy2PKStress(self, np, stateVars, stress):
        j = stateVars[np].deformation_gradient.determinant()
        return matrix_form(stress) @ stateVars[np].deformation_gradient.inverse().transpose() * j
    
    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        de = calculate_strain_increment2D(velocity_gradient, dt)
        dw = calculate_vorticity_increment2D(velocity_gradient, dt)
        return self.ImplicitIntegration(np, previous_stress, de, dw, stateVars)

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        de = calculate_strain_increment(velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)
        return self.ImplicitIntegration(np, previous_stress, de, dw, stateVars)

    @ti.func
    def ComputeStressInvariant(self, stress):
        return EquivalentStress(stress)
    
    @ti.func
    def ComputeShearFunction(self, seqv):
        return seqv - self._yield
    
    @ti.func
    def ComputeYieldFunction(self, stress):
        seqv = self.ComputeStressInvariant(stress)
        yield_shear = self.ComputeShearFunction(seqv)
        return yield_shear

    @ti.func
    def ComputeYieldState(self, stress):
        tolerance = -1e-8
        yield_shear = self.ComputeYieldFunction(stress)

        yield_state = 0
        if yield_shear > tolerance:
            yield_state = 1
        return yield_state, yield_shear
    
    @ti.func
    def ComputeDfDsigma(self, stress):
        df_dp = 0.
        df_dq = 1.
        
        dp_dsigma = DpDsigma() 
        dq_dsigma = DqDsigma(stress) 
        df_dsigma = df_dp * dp_dsigma + df_dq * dq_dsigma 
        return df_dsigma
    
    @ti.func
    def ComputeDfDvariable(self, sqrt2J2, lode, stress):
        return 0.

    @ti.func
    def ComputeDfDpvstrain(self, sqrt2J2, lode, stress):
        return 0.
    
    @ti.func
    def ComputeDfDpdstrain(self, sqrt2J2, lode, stress):
        return 0.
    
    @ti.func
    def ComputeDgDsigma(self, stress):
        dg_dp = 0.
        dg_dq = 1.
        
        dp_dsigma = DpDsigma() 
        dq_dsigma = DqDsigma(stress) 
        dg_dsigma = dg_dp * dp_dsigma + dg_dq * dq_dsigma 
        return dg_dp, dg_dq, dg_dsigma
    
    @ti.func
    def ComputeElasticStress(self, dstrain, stress):
        return stress + self.ComputeElasticStressIncrement(dstrain)
    
    @ti.func
    def ComputeElasticStressIncrement(self, dstrain):
        bulk_modulus = self.bulk
        shear_modulus = self.shear

        # !-- trial elastic stresses ----!
        dstress = ElasticTensorMultiplyVector(dstrain, bulk_modulus, shear_modulus)
        return dstress

    @ti.func
    def line_search(self, stress, dstress, f_function):
        alpha = 1.
        while True:
            _, f_function_new = self.ComputeYieldState(stress - alpha * dstress)
            if ti.abs(f_function_new) < ti.abs(f_function) or alpha < 1e-5: 
                break
            alpha /= 2.
        return alpha
    
    @ti.func
    def ConsistentCorrection(self, yield_state, f_function, stress, pdstrain):
        bulk_modulus, shear_modulus = self.bulk, self.shear

        dfdsigma = self.ComputeDfDsigma(yield_state, stress)
        _, _, dgdsigma = self.ComputeDgDsigma(yield_state, stress)
        tempMat = ElasticTensorMultiplyVector(dgdsigma, bulk_modulus, shear_modulus)
        dfdsigmaDedgdsigma = voigt_tensor_dot(dfdsigma, tempMat)
        abeta = 1. / dfdsigmaDedgdsigma if ti.abs(dfdsigmaDedgdsigma) > Threshold else 0.
        dlambda = f_function * abeta
        dstress = dlambda * tempMat
        dpdstrain = dlambda * equivalent_voigt(dgdsigma)
        alpha = self.line_search(stress, dstress, f_function)
        return stress - alpha * dstress, pdstrain + alpha * dpdstrain
    
    @ti.func
    def NormalCorrection(self, yield_state, f_function, stress):
        dfdsigma = self.ComputeDfDsigma(yield_state, stress)
        dfdsigmadfdsigma = voigt_tensor_dot(dfdsigma, dfdsigma)
        abeta = 1. / dfdsigmadfdsigma if ti.abs(dfdsigmadfdsigma) > Threshold else 0.
        dlambda = f_function * abeta
        dstress = dlambda * dfdsigma
        alpha = self.line_search(stress, dstress, f_function)
        return stress - alpha * dstress
    
    @ti.func
    def UpdateInternalVariables(self, np, dpdstrain, stateVars):
        stateVars[np].epstrain += dpdstrain

    @ti.func
    def UpdateStateVariables(self, np, stress, stateVars):
        stateVars[np].estress = VonMisesStress(stress)
    
    @ti.func
    def DriftCorrect(self, yield_state, f_function, stress, pdstrain):
        for _ in range(MAXITS):
            stress_new, pdstrain_new = self.ConsistentCorrection(yield_state, f_function, stress, pdstrain)
            yield_state_new, f_function_new = self.ComputeYieldState(stress_new)

            if ti.abs(f_function_new) > ti.abs(f_function):
                stress_new = self.NormalCorrection(yield_state, f_function, stress)
                yield_state_new, f_function_new = self.ComputeYieldState(stress_new)
                pdstrain_new = pdstrain

            stress = stress_new
            pdstrain = pdstrain_new
            yield_state = yield_state_new
            f_function = f_function_new
            if ti.abs(f_function_new) <= FTOL:
                break
        return stress, pdstrain
    
    @ti.func
    def ImplicitIntegration(self, np, previous_stress, de, dw, stateVars):
        bulk_modulus = self.bulk
        shear_modulus = self.shear

        # !---- trial elastic stresses ----!
        stress = previous_stress
        sigrot = Sigrot(stress, dw)
        dstress = ElasticTensorMultiplyVector(de, bulk_modulus, shear_modulus)
        trial_stress = stress + dstress

        # !---- compute trial stress invariants ----!
        pdstrain = 0.
        updated_stress = trial_stress
        yield_state_trial, f_function_trial = self.ComputeYieldState(trial_stress)
        
        if yield_state_trial > 0:
            Tolerance = 1e-1

            df_dsigma_trial = self.ComputeDfDsigma(yield_state_trial, trial_stress)
            __, _, dp_dsigma_trial = self.ComputeDgDsigma(yield_state_trial, trial_stress)
            temp_matrix = ElasticTensorMultiplyVector(df_dsigma_trial, bulk_modulus, shear_modulus)
            lambda_trial = f_function_trial / ti.max(((temp_matrix).dot(dp_dsigma_trial)), Threshold)

            yield_state, f_function = self.ComputeYieldState(stress)
            df_dsigma = self.ComputeDfDsigma(yield_state, stress)
            __, _, dg_dsigma = self.ComputeDgDsigma(yield_state, stress)
            temp_matrix = ElasticTensorMultiplyVector(df_dsigma, bulk_modulus, shear_modulus)
            lambda_ = temp_matrix.dot(de) / ti.max(((temp_matrix).dot(dg_dsigma)), Threshold)

            pdstrain = 0.
            if ti.abs(f_function) < Tolerance:
                temp_matrix = ElasticTensorMultiplyVector(dg_dsigma, bulk_modulus, shear_modulus)
                updated_stress -= lambda_ * temp_matrix
                pdstrain = lambda_ * equivalent_voigt(dg_dsigma)
            else:
                temp_matrix = ElasticTensorMultiplyVector(dp_dsigma_trial, bulk_modulus, shear_modulus)
                updated_stress -= lambda_trial * temp_matrix
                pdstrain = lambda_trial * equivalent_voigt(dp_dsigma_trial)

            yield_state, f_function = self.ComputeYieldState(updated_stress)
            if ti.abs(f_function) > FTOL:
                updated_stress, pdstrain = self.DriftCorrect(yield_state, f_function, updated_stress, pdstrain)
            
        updated_stress += sigrot
        stateVars[np].estress = VonMisesStress(updated_stress)
        stateVars[np].epstrain += pdstrain
        return updated_stress

    @ti.func
    def ComputePKStress(self, np, velocity_gradient, stateVars, dt):  
        previous_stress = self.PK2CauchyStress(np, stateVars)
        cauchy_stress = self.ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
        PKstress = self.Cauchy2PKStress(np, stateVars, cauchy_stress)
        stateVars[np].stress = PKstress
        return PKstress
    
    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        return ComputeElasticStiffnessTensor(self.bulk, self.shear)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stateVars):
        stiffness_matrix = self.ComputeStressInvariant(np, current_stress, stateVars)
        yield_state, f_function = self.ComputeYieldState(current_stress, stateVars[np])
        if yield_state > 0:
            bulk_modulus = self.bulk
            shear_modulus = self.shear

            dfdsigma = self.ComputeDfDsigma(yield_state, current_stress, stateVars[np])
            _, _, dgdsigma = self.ComputeDgDsigma(yield_state, current_stress, stateVars[np])
            tempMatf = ElasticTensorMultiplyVector(dfdsigma, bulk_modulus, shear_modulus)
            tempMatg = ElasticTensorMultiplyVector(dgdsigma, bulk_modulus, shear_modulus)
            dfdsigmaDedgdsigma = voigt_tensor_dot(dgdsigma, tempMatf)
            stiffness_matrix -= 1. / (dfdsigmaDedgdsigma) * (tempMatg.outer_product(tempMatf))
        return stiffness_matrix


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), epstrain: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].epstrain = epstrain[np]