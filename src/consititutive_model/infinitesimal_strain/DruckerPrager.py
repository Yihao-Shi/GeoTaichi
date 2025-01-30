import taichi as ti

from src.consititutive_model.MaterialKernel import *
from src.utils.constants import DELTA2D, DELTA, FTOL, MAXITS
from src.utils.MatrixFunction import matrix_form
from src.utils.TypeDefination import mat3x3
from src.utils.VectorFunction import voigt_form, voigt_tensor_dot, equivalent_voigt


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
class DruckerPragerModel:
    density: float
    young: float
    possion: float
    shear: float
    bulk: float
    c: float
    fai: float
    psi: float
    q_fai: float
    k_fai: float
    q_psi: float
    tensile: float

    def add_material(self, density, young, possion, c, fai, psi, tensile, dpType="Inscribed"):
        self.density = density
        self.young = young
        self.possion = possion

        self.shear = 0.5 * self.young / (1. + self.possion)
        self.bulk = self.young / (3. * (1 - 2. * self.possion))
        self.c = c
        self.fai = fai
        self.psi = psi
        
        if dpType == "Circumscribed":
            self.q_fai = 6. * ti.sin(self.fai) / (ti.sqrt(3) * (3 + ti.sin(self.fai)))
            self.k_fai = 6. * ti.cos(self.fai) * self.c / (ti.sqrt(3) * (3 + ti.sin(self.fai)))
            self.q_psi = 6. * ti.sin(self.psi) / (ti.sqrt(3) * (3 + ti.sin(self.psi)))
        elif dpType == "MiddleCircumscribed":
            self.q_fai = 6. * ti.sin(self.fai) / (ti.sqrt(3) * (3 - ti.sin(self.fai)))
            self.k_fai = 6. * ti.cos(self.fai) * self.c / (ti.sqrt(3) * (3 - ti.sin(self.fai)))
            self.q_psi = 6. * ti.sin(self.psi) / (ti.sqrt(3) * (3 - ti.sin(self.psi)))
        elif dpType == "Inscribed":
            self.q_fai = 3. * ti.tan(self.fai) / ti.sqrt(9. + 12 * ti.tan(self.fai) ** 2)
            self.k_fai = 3. * self.c / ti.sqrt(9. + 12 * ti.tan(self.fai) ** 2)
            self.q_psi = 3. * ti.tan(self.psi) / ti.sqrt(9. + 12 * ti.tan(self.psi) ** 2)

        if self.fai == 0:
            self.tensile = 0.
        else:
            self.tensile = ti.min(tensile, self.k_fai / self.q_fai)

    def add_contact_parameter(self, friction, kn, kt):
        self.friction = friction
        self.kn = kn
        self.kt = kt

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Drucker-Prager Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Possion Ratio: ', self.possion)
        print('Cohesion Coefficient = ', self.c)
        print('Angle of Internal Friction (in radian) = ', self.fai)
        print('Angle of Dilatation (in radian) = ', self.psi)
        print('Tensile = ', self.tensile, '\n')

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
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        return (DELTA2D + velocity_gradient * dt[None]).determinant()
    
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
        previous_stress = self.ImplicitIntegration(np, previous_stress, de, dw, stateVars)
        return previous_stress

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        de = calculate_strain_increment(velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)
        previous_stress = self.ImplicitIntegration(np, previous_stress, de, dw, stateVars)
        return previous_stress

    @ti.func
    def ComputeStressInvariant(self, stress):
        return MeanStress(stress), ti.sqrt(ComputeInvariantJ2(stress)) + Threshold
    
    @ti.func
    def ComputeTensileFunction(self, sigma):
        return sigma - self.tensile
    
    @ti.func
    def ComputeShearFunction(self, sigma, J2sqrt):
        return J2sqrt + self.q_fai * sigma - self.k_fai
    
    @ti.func
    def ComputeYieldFunction(self, stress):
        sigma, J2sqrt = self.ComputeStressInvariant(stress)
        yield_shear = self.ComputeShearFunction(sigma, J2sqrt)
        yield_tensile = self.ComputeTensileFunction(sigma)
        return yield_shear, yield_tensile

    @ti.func
    def ComputeYieldState(self, stress):
        tolerance = -1e-8
        yield_shear, yield_tensile = self.ComputeYieldFunction(stress)

        yield_state = 0
        if yield_tensile > tolerance and yield_shear > tolerance:
            _, J2sqrt = self.ComputeStressInvariant(stress)
            alphap = ti.sqrt(1 + self.q_fai ** 2) - self.q_fai
            J2sqrtp = self.k_fai - self.q_fai * self.tensile
            dp_hfai = J2sqrt - J2sqrtp - alphap * yield_tensile
            if dp_hfai > Threshold:
                yield_state = 1
            else:
                yield_state = 2
        if yield_tensile < tolerance and yield_shear > tolerance:
            yield_state = 1
        if yield_tensile > tolerance and yield_shear < tolerance:
            yield_state = 2

        f_function = 0.
        if yield_state == 1:
            f_function = yield_shear
        elif yield_state == 2:
            f_function = yield_tensile
        return yield_state, f_function
    
    @ti.func
    def ComputeDfDsigma(self, yield_state, stress):
        df_dp, df_dq = 0., 0.
        if yield_state == 2:
            df_dp = 1.
            df_dq = 0.
        else:
            df_dp = self.q_fai
            df_dq = ti.sqrt(3.) / 3.
        
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
    def ComputeDgDsigma(self, yield_state, stress):
        dg_dp, dg_dq = 0., 0.
        if yield_state == 2:
            dg_dp = 1.
            dg_dq = 0.
        else:
            dg_dp = self.q_psi
            dg_dq = ti.sqrt(3.) / 3.
        
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
            __, _, dg_dsigma_trial = self.ComputeDgDsigma(yield_state_trial, trial_stress)
            temp_matrix = ElasticTensorMultiplyVector(df_dsigma_trial, bulk_modulus, shear_modulus)
            den = (temp_matrix).dot(dg_dsigma_trial)
            lambda_trial = ti.max(0., f_function_trial / den if ti.abs(den) > Tolerance else 0.)

            yield_state, f_function = self.ComputeYieldState(stress)
            df_dsigma = self.ComputeDfDsigma(yield_state, stress)
            __, _, dg_dsigma = self.ComputeDgDsigma(yield_state, stress)
            temp_matrix = ElasticTensorMultiplyVector(df_dsigma, bulk_modulus, shear_modulus)
            den = (temp_matrix).dot(dg_dsigma)
            lambda_ = ti.max(0., f_function / den if ti.abs(den) > Tolerance else 0.)

            pdstrain = 0.
            if ti.abs(f_function) > Tolerance or yield_state == 0:
                temp_matrix = ElasticTensorMultiplyVector(dg_dsigma, bulk_modulus, shear_modulus)
                updated_stress -= lambda_ * temp_matrix
                pdstrain = lambda_ * equivalent_voigt(dg_dsigma)
            else:
                temp_matrix = ElasticTensorMultiplyVector(dg_dsigma_trial, bulk_modulus, shear_modulus)
                updated_stress -= lambda_trial * temp_matrix
                pdstrain = lambda_trial * equivalent_voigt(dg_dsigma_trial)

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
        stiffness_matrix = self.compute_elastic_tensor(np, current_stress, stateVars)
        yield_state, f_function = self.ComputeYieldState(current_stress)
        if yield_state > 0:
            bulk_modulus = self.bulk
            shear_modulus = self.shear

            dfdsigma = self.ComputeDfDsigma(yield_state, current_stress)
            _, _, dgdsigma = self.ComputeDgDsigma(yield_state, current_stress)
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