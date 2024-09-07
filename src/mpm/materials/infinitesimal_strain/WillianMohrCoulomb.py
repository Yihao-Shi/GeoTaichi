import numpy as np
import taichi as ti

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.MaterialKernel import *
from src.utils.constants import DELTA, PI, FTOL, LTOL, MAXITS, NSUB, EPS, STOL, dTmin
from src.utils.MatrixFunction import matrix_form
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import mat3x3
from src.utils.VectorFunction import voigt_form, voigt_tensor_dot, Squared


class WillianMohrCoulomb(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_particle_num, configuration="ULMPM", solver_type="Explicit"):
        super().__init__()
        self.matProps = WillianMohrCoulombModel.field(shape=max_material_num)
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
        c = DictIO.GetAlternative(material, 'Cohesion', 0.)
        fai = DictIO.GetAlternative(material, 'Friction', 0.) * np.pi / 180.
        psi = DictIO.GetAlternative(material, 'Dilation', 0.) * np.pi / 180.
        tensile = DictIO.GetAlternative(material, 'Tensile', 0.)
        self.matProps[materialID].add_material(density, young, possion, c, fai, psi, tensile)
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
class WillianMohrCoulombModel:
    density: float
    young: float
    possion: float
    shear: float
    bulk: float
    c: float
    fai: float
    psi: float
    tensile: float

    def add_material(self, density, young, possion, c, fai, psi, tensile):
        self.density = density
        self.young = young
        self.possion = possion

        self.shear = 0.5 * self.young / (1. + self.possion)
        self.bulk = self.young / (3. * (1 - 2. * self.possion))
        self.c = c
        self.fai = fai
        self.psi = psi

        if self.fai == 0:
            self.tensile = 0.
        else:
            self.tensile = ti.min(tensile, self.c / ti.tan(self.fai))

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Willam Mohr-Coulomb Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Possion Ratio: ', self.possion)
        print('Cohesion Coefficient = ', self.c)
        print('Internal Friction (in radian) = ', self.fai)
        print('Dilatation (in radian) = ', self.psi, '\n')

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
    
    # ==================================================== Mohr-Coulomb Model ==================================================== #
    @ti.func
    def get_epsilon(self, stress):
        return ti.sqrt(3) * MeanStress(stress)
    
    @ti.func
    def get_sqrt2J2(self, stress):
        return ti.sqrt(2 * ComputeInvariantJ2(stress))
    
    @ti.func
    def get_lode(self, stress):
        return ComputeLodeAngle(stress)

    @ti.func
    def ComputeStressInvariant(self, stress):
        return self.get_epsilon(stress), self.get_sqrt2J2(stress), self.get_lode(stress)
    
    @ti.func
    def ComputeShearFunction(self, epsilon, sqrt2J2, lode):
        fai, cohesion = self.fai, self.c
        cos_fai, tan_fai = ti.cos(fai), ti.tan(fai)
        yield_shear = ti.sqrt(1.5) * sqrt2J2 * (ti.sin(lode + PI/3.) / (ti.sqrt(3.) * cos_fai) + \
                      ti.cos(lode + PI/3.) * tan_fai / 3.) + epsilon * ti.sqrt(1./3.) * tan_fai - cohesion
        return yield_shear
    
    @ti.func
    def ComputeTensileFunction(self, epsilon, sqrt2J2, lode):
        tensile = self.tensile
        cos_lode = ti.cos(lode)
        yield_tensile = ti.sqrt(2./3.) * cos_lode * sqrt2J2 + epsilon * ti.sqrt(1./3.) - tensile
        return yield_tensile
    
    @ti.func
    def ComputeYieldFunction(self, stress):
        epsilon, sqrt2J2, lode = self.ComputeStressInvariant(stress)
        yield_shear = self.ComputeShearFunction(epsilon, sqrt2J2, lode)
        yield_tensile = self.ComputeTensileFunction(epsilon, sqrt2J2, lode)
        return yield_shear, yield_tensile

    @ti.func
    def ComputeYieldState(self, stress):
        tolerance = -1e-8
        fai, cohesion, tensile = self.fai, self.c, self.tensile
        sin_fai = ti.sin(fai)
        epsilon, sqrt2J2, lode = self.ComputeStressInvariant(stress)
        yield_shear, yield_tensile = self.ComputeYieldFunction(stress)

        yield_state = 0
        if yield_tensile > tolerance and yield_shear > tolerance:
            n_fai = (1. + sin_fai) / (1. - sin_fai)
            sigma_p = tensile * n_fai - 2. * cohesion * ti.sqrt(n_fai)
            alpha_p = ti.sqrt(1. + n_fai * n_fai) + n_fai
            h = yield_tensile + alpha_p * (ti.sqrt(2./3.) * ti.cos(lode - 4.*PI/3.) * sqrt2J2 + epsilon * ti.sqrt(1./3.) - sigma_p)
            if h > Threshold:
                yield_state = 2
            else:
                yield_state = 1
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
        sqrt2J2 = self.get_sqrt2J2(stress)
        lode = self.get_lode(stress)

        fai = self.fai
        df_depsilon, df_dsqrt2J2, df_dlode = 0., 0., 0.
        if yield_state == 2:
            sin_lode, cos_lode = ti.sin(lode), ti.cos(lode)
            df_depsilon = ti.sqrt(1./3.)
            df_dsqrt2J2 = ti.sqrt(2./3.) * cos_lode
            df_dlode = -ti.sqrt(2./3.) * sqrt2J2 * sin_lode
        else:
            sin_lode_PI_3, cos_lode_PI_3 = ti.sin(lode + PI/3.), ti.cos(lode + PI/3.)
            cos_fai, tan_fai = ti.cos(fai), ti.tan(fai)
            df_depsilon = tan_fai * ti.sqrt(1./3.)
            df_dsqrt2J2 = ti.sqrt(1.5) * (sin_lode_PI_3 / (ti.sqrt(3.) * cos_fai) + cos_lode_PI_3 * tan_fai / 3.)
            df_dlode = ti.sqrt(1.5) * sqrt2J2 * (cos_lode_PI_3 / (ti.sqrt(3.) * cos_fai) - sin_lode_PI_3 * tan_fai / 3.)
        
        depsilon_dsigma = DpDsigma() * ti.sqrt(3.)
        dsqrt2J2_dsigma = DqDsigma(stress) * ti.sqrt(2./3.)
        dlode_dsigma = DlodeDsigma(stress)
        df_dsigma = df_depsilon * depsilon_dsigma + df_dsqrt2J2 * dsqrt2J2_dsigma + df_dlode * dlode_dsigma
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
        sqrt2J2 = self.get_sqrt2J2(stress)
        lode = self.get_lode(stress)

        xi, xit = 0.1, 0.1
        fai, psi, cohesion, tensile = self.fai, self.psi, self.c, self.tensile
        sin_lode, cos_lode = ti.sin(lode), ti.cos(lode)
        sin_fai, cos_fai, tan_psi = ti.sin(fai), ti.cos(fai), ti.tan(psi)

        depsilon_dsigma = DpDsigma() * ti.sqrt(3.)
        dsqrt2J2_dsigma = DqDsigma(stress) * ti.sqrt(2./3.)
        dlode_dsigma = DlodeDsigma(stress)
        
        dg_dp, dg_dq = 0., 0.
        dg_depsilon, dg_dsqrt2J2, dg_dlode = 0., 0., 0.
        if yield_state == 2:
            et_value = 0.6
            sqpart = 4. * (1 - et_value * et_value) * cos_lode * cos_lode + 5. * et_value * et_value - 4. * et_value
            if sqpart < Threshold: sqpart = 1e-5
            rt_den = 2. * (1 - et_value * et_value) * cos_lode + (2. * et_value - 1) * ti.sqrt(sqpart)
            rt_num = 4. * (1 - et_value * et_value) * cos_lode * cos_lode + (2. * et_value - 1) * (2. * et_value - 1)
            if ti.abs(rt_den) < Threshold: rt_den = 1e-5
            rt = rt_num / (3. * rt_den)
            temp_den = ti.sqrt(xit * xit * tensile * tensile + 1.5 * rt * rt * sqrt2J2 * sqrt2J2)
            if temp_den < Threshold: temp_den = Threshold
            dg_dp = 1.
            dg_dq = ti.sqrt(1.5) * sqrt2J2 * rt * rt / temp_den
            dp_drt = 1.5 * sqrt2J2 * sqrt2J2 * rt / temp_den
            drtden_dlode = -2. * (1 - et_value * et_value) * sin_lode - (2. * et_value - 1) * 4. * (1 - et_value * et_value) * cos_lode * \
                            sin_lode / ti.sqrt(4. * (1 - et_value * et_value) * cos_lode * cos_lode + 5. * et_value * et_value - 4. * et_value)
            drtnum_dlode = -8. * (1 - et_value * et_value) * cos_lode * sin_lode
            drt_dlode = (drtnum_dlode * rt_den - drtden_dlode * rt_num) / (3. * rt_den * rt_den)
            dg_dlode = dp_drt * drt_dlode 
        else:
            r_mc = (3. - sin_fai) / (6. * cos_fai)
            e_val = (3. - sin_fai) / (3. + sin_fai)
            e_val = clamp(0.5 + 1e-10, 1., e_val)
            sqpart = 4. * (1 - e_val * e_val) * cos_lode * cos_lode + 5 * e_val * e_val - 4 * e_val
            if sqpart < Threshold: sqpart = 1e-10
            m = 2. * (1 - e_val * e_val) * cos_lode + (2. * e_val - 1) * ti.sqrt(sqpart)
            if ti.abs(m) < Threshold: m = 1e-10
            l = 4. * (1 - e_val * e_val) * cos_lode * cos_lode + (2. * e_val - 1) * (2. * e_val - 1)
            r_mw = (l / m) * r_mc
            omega = (xi * cohesion * tan_psi) * (xi * cohesion * tan_psi) + (r_mw * ti.sqrt(1.5) * sqrt2J2) * (r_mw * ti.sqrt(1.5) * sqrt2J2)
            if omega < Threshold: omega = 1e-10
            dl_dlode = -8. * (1. - e_val * e_val) * cos_lode * sin_lode
            dm_dlode = -2. * (1. - e_val * e_val) * sin_lode + (0.5 * (2. * e_val - 1.) * dl_dlode) / ti.sqrt(sqpart)
            drmw_dlode = ((m * dl_dlode) - (l * dm_dlode)) / (m * m)
            dg_dp = tan_psi
            dg_dq = sqrt2J2 * r_mw * r_mw / (2. * ti.sqrt(omega)) * ti.sqrt(6.)
            dg_dlode = (3. * sqrt2J2 * sqrt2J2 * r_mw * r_mc * drmw_dlode) / (2. * ti.sqrt(omega))
        
        dg_depsilon = dg_dp / ti.sqrt(3.)
        dg_dsqrt2J2 = dg_dq * ti.sqrt(1.5)
        dg_dsigma = (dg_depsilon * depsilon_dsigma) + (dg_dsqrt2J2 * dsqrt2J2_dsigma) + (dg_dlode * dlode_dsigma)
        return dg_dp, dg_dq, dg_dsigma
    
    @ti.func
    def ComputeElasticStress(self, dstrain, stress):
        return stress + self.ComputeElasticStressIncrement(dstrain, stress)
    
    @ti.func
    def ComputeElasticStressIncrement(self, dstrain, stress):
        bulk_modulus = self.bulk
        shear_modulus = self.shear

        # !-- trial elastic stresses ----!
        dstress = ElasticTensorMultiplyVector(dstrain, bulk_modulus, shear_modulus)
        return dstress
        
    @ti.func
    def Substepping(self, yield_state, dsig_e, stress):
        bulk_modulus = self.bulk
        shear_modulus = self.shear

        dfdsigma = self.ComputeDfDsigma(yield_state, stress)
        _, dg_dq, dgdsigma = self.ComputeDgDsigma(yield_state, stress)
        tempMat = ElasticTensorMultiplyVector(dfdsigma, bulk_modulus, shear_modulus)
        dfdsigmaDedgdsigma = voigt_tensor_dot(dgdsigma, tempMat)
        abeta = 1. / dfdsigmaDedgdsigma if ti.abs(dfdsigmaDedgdsigma) > Threshold else 0.
        dlambda = ti.max(voigt_tensor_dot(dfdsigma, dsig_e) * abeta, 0)
        dstress = dsig_e - dlambda * ElasticTensorMultiplyVector(dgdsigma, bulk_modulus, shear_modulus)
        dpdstrain = dlambda * dg_dq
        return dstress, dpdstrain
    
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
        _, dg_dq, dgdsigma = self.ComputeDgDsigma(yield_state, stress)
        tempMat = ElasticTensorMultiplyVector(dgdsigma, bulk_modulus, shear_modulus)
        dfdsigmaDedgdsigma = voigt_tensor_dot(dfdsigma, tempMat)
        abeta = 1. / dfdsigmaDedgdsigma if ti.abs(dfdsigmaDedgdsigma) > Threshold else 0.
        dlambda = f_function * abeta
        dstress = dlambda * tempMat
        dpdstrain = dlambda * dg_dq
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
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        ############################## STEP1 ##############################
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
    def ExplicitIntegration(self, np, previous_stress, de, dw, stateVars):
        ############################## STEP2 ##############################
        alpha = self.CalculateElasticFactor(de, previous_stress)

        ############################## STEP5 ##############################
        stress = self.ComputeElasticStress(alpha * de, previous_stress)

        if ti.abs(1. - alpha) > Threshold:
            dsig_e = self.ComputeElasticStressIncrement((1. - alpha) * de, previous_stress)
            stress = self.core(np, dsig_e, stress, stateVars)

        dsigrot = Sigrot(previous_stress, dw)
        update_stress = stress + dsigrot
        self.UpdateStateVariables(np, update_stress, stateVars)
        return update_stress
    
    @ti.func
    def core(self, np, dsig_e, sig, stateVars):  
        ############################## STEP6 ##############################
        T = 0.
        dT = 1.
        pdstrain = 0.
        ############################## STEP6 ##############################
        
        ############################## STEP7 ##############################
        while(T < 1):
        ############################## STEP7 ##############################

            ############################## STEP8 ##############################
            # Substep1 #
            sig1 = sig
            dsig1, dpdstrain1 = self.Substepping(dT * dsig_e, sig1)
            # Substep1 #

            # Substep2 #
            sig2 = sig + dsig1
            dsig2, dpdstrain2 = self.Substepping(dT * dsig_e, sig2)
            # Substep2 #
            ############################## STEP8 ##############################

            ############################## STEP9 ##############################
            sigTemp = sig + 0.5 * (dsig1 + dsig2)
            pdstrainTemp = pdstrain + 0.5 * (dpdstrain1 + dpdstrain2)
            ############################## STEP9 ##############################

            ############################## STEP10 ##############################
            E_sigma = dsig2 - dsig1
            err = ti.max(0.5 * E_sigma.norm() / sigTemp.norm(), EPS)
            ############################## STEP10 ##############################

            ############################## STEP11 ##############################
            if err > STOL and dT > dTmin:
                Q = ti.max(.9 * ti.sqrt(STOL / err), .1)
                dT = ti.max(dTmin, Q * dT)
                continue
            ############################## STEP11 ##############################

            ############################## STEP12 ##############################
            sig = sigTemp
            pdstrain = pdstrainTemp
            ############################## STEP12 ##############################

            ############################## STEP13 ##############################
            f_function = self.ComputeYieldFunction(sig)
            if ti.abs(f_function) > FTOL:
                sig, pdstrain = self.DriftCorrect(f_function, sig, pdstrain)
            ############################## STEP13 ##############################
            
            ############################## STEP14 ##############################
            T += dT
            Q = ti.max(.9 * ti.sqrt(STOL / err), 1.)
            dT = ti.max(dTmin, ti.min(Q * dT, 1. - T))
            ############################## STEP14 ##############################   
        self.UpdateInternalVariables(np, pdstrain, stateVars)
        return sig
    
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
    def CalculateElasticFactor(self, dstrain, stress):
        dstress = self.ComputeElasticStressIncrement(dstrain, stress)
        f_function0 = self.ComputeYieldFunction(stress)
        f_function1 = self.ComputeYieldFunction(stress + dstress)

        alpha = 0.
        if f_function1 <= FTOL:
            alpha = 1.
        elif f_function0 < -FTOL and f_function1 > FTOL:
            alpha = self.ModifiedRegulaFalsi(dstrain, stress, f_function0, 0., 1.)
        elif ti.abs(f_function0) <= FTOL and f_function1 > FTOL:
            dfdsigma = self.ComputeDfDsigma(stress)
            cos_theta = voigt_tensor_dot(dfdsigma, dstress) / ti.sqrt(Squared(dfdsigma)) / ti.sqrt(Squared(dstress))

            if cos_theta >= -LTOL:
                alpha = 0.
            else:
                alpha0, alpha1 = self.RegulaFalsiNegativePlasticMultiplier(dstress, stress, f_function0)
                alpha = self.ModifiedRegulaFalsi(dstrain, stress, f_function0, alpha0, alpha1)
        else:
            alpha = 0.
        return alpha

    @ti.func
    def ModifiedRegulaFalsi(self, dstrain, stress, f_function_save, alpha0, alpha1):
        alpha = 0. 
        dstress0 = self.ComputeElasticStressIncrement(alpha0 * dstrain, stress)
        dstress1 = self.ComputeElasticStressIncrement(alpha1 * dstrain, stress)
        f_function0 = self.ComputeYieldFunction(stress + dstress0)
        f_function1 = self.ComputeYieldFunction(stress + dstress1)
        for _ in range(MAXITS):
            alpha = alpha1 - (alpha1 - alpha0) * f_function1 / (f_function1 - f_function0)
            des = dstrain * alpha
            stress_new = self.ComputeElasticStress(des, stress)
            f_function_new = self.ComputeYieldFunction(stress_new)
            
            if ti.abs(f_function_new) <= FTOL:
                break
            if f_function_new * f_function0 < 0.:                    
                alpha1 = alpha
                f_function1 = f_function_new
                if f_function_new * f_function_save > 0.: f_function0 *= 0.5
            else:
                alpha0 = alpha
                f_function0 = f_function_new
                if f_function_new * f_function_save > 0.: f_function1 *= 0.5
            f_function_save = f_function_new
        return alpha

    @ti.func
    def RegulaFalsiNegativePlasticMultiplier(self, dstrain, stress, f_function0):
        alpha = 0.
        alpha0, alpha1 = 0., 1.
        f_function_save = f_function0
        for _ in range(MAXITS):
            dalpha = (alpha1 - alpha0) / NSUB
            flag = 0
            for __ in range(NSUB):
                alpha = alpha0 + dalpha
                des = alpha * dstrain
                dstress = self.ComputeElasticStressIncrement(des, stress)
                f_function_new = self.ComputeYieldFunction(stress + dstress)
                if f_function_new > FTOL:
                    alpha1 = alpha
                    if f_function0 < -FTOL:
                        flag = 1
                        break
                    else:
                        alpha0 = 0
                        f_function0 = f_function_save
                        break
                else:
                    alpha0 = alpha
                    f_function0 = f_function_new
            if flag == 1:
                break
        return alpha0, alpha1

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
            __, dp_dq_trial, dp_dsigma_trial = self.ComputeDgDsigma(yield_state_trial, trial_stress)
            temp_matrix = ElasticTensorMultiplyVector(df_dsigma_trial, bulk_modulus, shear_modulus)
            lambda_trial = f_function_trial / ti.max(((temp_matrix).dot(dp_dsigma_trial)), Threshold)
            
            yield_state, f_function = self.ComputeYieldState(stress)
            df_dsigma = self.ComputeDfDsigma(yield_state, stress)
            __, dp_dq, dp_dsigma = self.ComputeDgDsigma(yield_state, stress)
            temp_matrix = ElasticTensorMultiplyVector(df_dsigma, bulk_modulus, shear_modulus)
            lambda_ = temp_matrix.dot(de) / ti.max(((temp_matrix).dot(dp_dsigma)), Threshold)
            
            pdstrain = 0.
            if ti.abs(f_function) < Tolerance:
                temp_matrix = ElasticTensorMultiplyVector(dp_dsigma, bulk_modulus, shear_modulus)
                updated_stress -= lambda_ * temp_matrix
                pdstrain = lambda_ * dp_dq
            else:
                temp_matrix = ElasticTensorMultiplyVector(dp_dsigma_trial, bulk_modulus, shear_modulus)
                updated_stress -= lambda_trial * temp_matrix
                pdstrain = lambda_trial * dp_dq_trial

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
        stiffness_matrix = ComputeElasticStiffnessTensor(self.bulk, self.shear)
        yield_state, _ = self.ComputeYieldState(current_stress)
        if yield_state > 0:
            bulk_modulus = self.bulk
            shear_modulus = self.shear

            dfdsigma = self.ComputeDfDsigma(current_stress)
            _, _, dgdsigma = self.ComputeDgDsigma(current_stress)
            tempMatf = ElasticTensorMultiplyVector(dfdsigma, bulk_modulus, shear_modulus)
            tempMatg = ElasticTensorMultiplyVector(dgdsigma, bulk_modulus, shear_modulus)
            dfdsigmaDedgdsigma = voigt_tensor_dot(dgdsigma, tempMatf)
            stiffness_matrix -= 1. / dfdsigmaDedgdsigma * (tempMatg.outer_product(tempMatf))
        return stiffness_matrix

@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), epstrain: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].epstrain = epstrain[np]