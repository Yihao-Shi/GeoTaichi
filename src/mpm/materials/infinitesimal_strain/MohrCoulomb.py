import numpy as np
import taichi as ti

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.MaterialKernel import *
from src.utils.constants import DELTA, PI, MThreshold
from src.utils.MatrixFunction import matrix_form
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import mat3x3
from src.utils.VectorFunction import voigt_form


class MohrCoulomb(ConstitutiveModelBase):
    def __init__(self, max_material_num, max_particle_num, configuration="ULMPM", solver_type="Explicit"):
        super().__init__()
        self.matProps = MohrCoulombModel.field(shape=max_material_num)
        if configuration == "ULMPM":
            self.stateVars = ULStateVariable.field(shape=max_particle_num) 
        elif configuration == "TLMPM":
            self.stateVars = TLStateVariable.field(shape=max_particle_num) 

        if solver_type == "Implicit":
            self.stiffness_matrix = ti.Matrix.field(6, 6, float, shape=max_particle_num)

    def get_state_vars_dict(self, start_particle, end_particle):
        epstrain = np.ascontiguousarray(self.stateVars.epstrain.to_numpy()[start_particle:end_particle])
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        cohesion = np.ascontiguousarray(self.stateVars.c.to_numpy()[start_particle:end_particle])
        friction = np.ascontiguousarray(self.stateVars.fai.to_numpy()[start_particle:end_particle])
        dilation = np.ascontiguousarray(self.stateVars.psi.to_numpy()[start_particle:end_particle])
        return {'epstrain': epstrain, 'estress': estress, 'cohesion': cohesion, 'friction': friction, 'dilation': dilation}        
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        epstrain = state_vars.item()['epstrain']
        cohesion = state_vars.item()['cohesion']
        friction = state_vars.item()['friction']
        dilation = state_vars.item()['dilation']
        kernel_reload_state_variables(estress, epstrain, cohesion, friction, dilation, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID')
        self.check_materialID(materialID, self.matProps.shape[0])
        
        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        possion = DictIO.GetAlternative(material, 'PossionRatio', 0.3)
        tensile = DictIO.GetAlternative(material, 'Tensile', 0.)
        c_peak = DictIO.GetAlternative(material, 'Cohesion', 0.)
        fai_peak = DictIO.GetAlternative(material, 'Friction', 0.) * PI / 180.
        psi_peak = DictIO.GetAlternative(material, 'Dilation', 0.) * PI / 180.
        c_residual = DictIO.GetAlternative(material, 'ResidualCohesion', 0.)
        fai_residual = DictIO.GetAlternative(material, 'ResidualFriction', 0.) * PI / 180.
        psi_residual = DictIO.GetAlternative(material, 'ResidualDilation', 0.) * PI / 180.
        pdstrain_peak = DictIO.GetAlternative(material, 'PlasticDevStrain', 0.) 
        pdstrain_residual = DictIO.GetAlternative(material, 'ResidualPlasticDevStrain', 0.)
        
        if fai_peak == 0.:
            tensile = 0.
        elif tensile >= c_peak / ti.tan(fai_peak): tensile = c_peak / ti.tan(fai_peak)
        
        self.matProps[materialID].add_material(density, young, possion, c_peak, fai_peak, psi_peak, c_residual, fai_residual, psi_residual, pdstrain_peak, pdstrain_residual, tensile)
        self.matProps[materialID].print_message(materialID)


    def get_lateral_coefficient(self, materialID):
        mu = self.matProps[materialID].possion
        return mu / (1. - mu)


@ti.dataclass
class ULStateVariable:
    epstrain: float
    estress: float
    fai: float
    psi: float
    c: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        materialID = int(particle[np].materialID)
        self.estress = VonMisesStress(stress)
        self.fai = matProps[materialID].fai_peak
        self.psi = matProps[materialID].psi_peak
        self.c = matProps[materialID].c_peak

    @ti.func
    def _update_vars(self, stress, epstrain):
        self.estress = VonMisesStress(stress)
        self.epstrain = epstrain


@ti.dataclass
class TLStateVariable:
    epstrain: float
    estress: float
    fai: float
    psi: float
    c: float
    deformation_gradient: mat3x3
    stress: mat3x3

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        stress = particle[np].stress
        materialID = int(particle[np].materialID)
        self.estress = VonMisesStress(stress)
        self.fai = matProps[materialID].fai_peak
        self.psi = matProps[materialID].psi_peak
        self.c = matProps[materialID].c_peak
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
class MohrCoulombModel:
    density: float
    young: float
    possion: float
    shear: float
    bulk: float
    c_peak: float
    fai_peak: float
    psi_peak: float
    c_residual: float
    fai_residual: float
    psi_residual: float
    pdstrain_peak: float
    pdstrain_residual: float
    tensile: float

    def add_material(self, density, young, possion, c_peak, fai_peak, psi_peak, c_residual, fai_residual, psi_residual, pdstrain_peak, pdstrain_residual, tensile):
        self.density = density
        self.young = young
        self.possion = possion

        self.shear = 0.5 * self.young / (1. + self.possion)
        self.bulk = self.young / (3. * (1 - 2. * self.possion))
        self.c_peak = c_peak
        self.fai_peak = fai_peak
        self.psi_peak = psi_peak
        self.c_residual = c_residual
        self.fai_residual = fai_residual
        self.psi_residual = psi_residual
        self.pdstrain_peak = pdstrain_peak
        self.pdstrain_residual = pdstrain_residual
        self.tensile = tensile

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        if self.fai_peak > Threshold:
            print('Constitutive model: Mohr-Coulomb Model')
        else:
            print('Constitutive model: Tresca Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Possion Ratio: ', self.possion)
        print('Peak Cohesion Coefficient = ', self.c_peak)
        print('Peak Internal Friction (in radian) = ', self.fai_peak)
        print('Peak Dilatation (in radian) = ', self.psi_peak)
        print('Residual Cohesion Coefficient = ', self.c_residual)
        print('Residual Internal Friction (in radian) = ', self.fai_residual)
        print('Residual Dilatation (in radian) = ', self.psi_residual)
        print('Peak Plastic Deviartoric Strain = ', self.pdstrain_peak)
        print('Residual Plastic Deviartoric Strain = ', self.pdstrain_residual)
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
    # cb-geo /include/materials/infinitesimal_strain/mohr_coulomb.tcc
    @ti.func
    def YieldState(self, lode, sqrt2J2, epsilon, variables):
        tolerance = -1e-6
        fai, cohesion, tensile = variables.fai, variables.c, self.tensile
        cos_lode = ti.cos(lode)
        sin_fai, cos_fai, tan_fai = ti.sin(fai), ti.cos(fai), ti.tan(fai)

        yield_tension = ti.sqrt(2./3.) * cos_lode * sqrt2J2 + epsilon * ti.sqrt(1./3.) - tensile
        yield_shear = ti.sqrt(1.5) * sqrt2J2 * (ti.sin(lode + PI/3.) / (ti.sqrt(3.) * cos_fai) + \
                                        ti.cos(lode + PI/3.) * tan_fai / 3.) + \
                                        epsilon * ti.sqrt(1./3.) * tan_fai - cohesion

        yield_state = 0
        if yield_tension > tolerance and yield_shear > tolerance:
            n_fai = (1. + sin_fai) / (1. - sin_fai)
            sigma_p = tensile * n_fai - 2. * cohesion * ti.sqrt(n_fai)
            alpha_p = ti.sqrt(1. + n_fai * n_fai) + n_fai
            h = yield_tension + alpha_p * (ti.sqrt(2./3.) * ti.cos(lode - 4.*PI/3.) * sqrt2J2 \
                                        + epsilon * ti.sqrt(1./3.) - sigma_p)
            if h > Threshold:
                yield_state = 2
            else:
                yield_state = 1
        if yield_tension < tolerance and yield_shear > tolerance:
            yield_state = 1
        if yield_tension > tolerance and yield_shear < tolerance:
            yield_state = 2

        return yield_state, yield_tension, yield_shear
        
    @ti.func
    def Compute_DfDp(self, yield_state, sqrt2J2, lode, stress, variables):
        df_dsigma = dp_dsigma = ZEROVEC6f
        dp_dq = 0.
        
        pdstrain = variables.epstrain
        pdstrain_peak, pdstrain_residual = self.pdstrain_peak, self.pdstrain_residual
        fai, psi, cohesion, tensile = variables.fai, variables.psi, variables.c, self.tensile
        sin_lode, cos_lode = ti.sin(lode), ti.cos(lode)
        sin_fai, cos_fai, tan_fai, tan_psi = ti.sin(fai), ti.cos(fai), ti.tan(fai), ti.tan(psi)

        df_depsilon, df_dsqrt2J2, df_dlode = 0., 0., 0.
        if yield_state == 2:
            df_depsilon = ti.sqrt(1./3.)
            df_dsqrt2J2 = ti.sqrt(2./3.) * cos_lode
            df_dlode = -ti.sqrt(2./3.) * sqrt2J2 * sin_lode
        else:
            df_depsilon = tan_fai * ti.sqrt(1./3.)
            df_dsqrt2J2 = ti.sqrt(1.5) * (ti.sin(lode + PI/3.) / (ti.sqrt(3.) * cos_fai) + \
                                            ti.cos(lode + PI/3.) * tan_fai / 3.)
            df_dlode = ti.sqrt(1.5) * sqrt2J2 * (ti.cos(lode + PI/3.) / (ti.sqrt(3.) * cos_fai) - \
                                                ti.sin(lode + PI/3.) * tan_fai / 3.)
        
        depsilon_dsigma = DpDsigma() * ti.sqrt(3.)
        dsqrt2J2_dsigma = DqDsigma(stress) * ti.sqrt(2./3.)
        dlode_dsigma = DlodeDsigma(stress)
        df_dsigma = df_depsilon * depsilon_dsigma + df_dsqrt2J2 * dsqrt2J2_dsigma + df_dlode * dlode_dsigma
        
        if yield_state == 2:
            et_value = 0.6
            xit = 0.1
            sqpart = 4. * (1 - et_value * et_value) * cos_lode * cos_lode + 5. * et_value * et_value - 4. * et_value
            if sqpart < Threshold: sqpart = 1e-5
            rt_den = 2. * (1 - et_value * et_value) * cos_lode + (2. * et_value - 1) * ti.sqrt(sqpart)
            rt_num = 4. * (1 - et_value * et_value) * cos_lode * cos_lode + (2. * et_value - 1) * (2. * et_value - 1)
            if ti.abs(rt_den) < Threshold: rt_den = 1e-5
            rt = rt_num / (3. * rt_den)

            temp_den = ti.sqrt(xit * xit * tensile * tensile + 1.5 * rt * rt * sqrt2J2 * sqrt2J2)
            if temp_den < Threshold: temp_den = Threshold
            dp_drt = 1.5 * sqrt2J2 * sqrt2J2 * rt / temp_den
            dp_dsqrt2J2 = 1.5 * sqrt2J2 * rt * rt / temp_den

            dp_depsilon = 1. / ti.sqrt(3.)
            drtden_dlode = -2. * (1 - et_value * et_value) * sin_lode - (2. * et_value - 1) * 4. * (1 - et_value * et_value) * cos_lode * \
                            sin_lode / ti.sqrt(4. * (1 - et_value * et_value) * cos_lode * cos_lode + 5. * et_value * et_value - 4. * et_value)
            drtnum_dlode = -8. * (1 - et_value * et_value) * cos_lode * sin_lode
            drt_dlode = (drtnum_dlode * rt_den - drtden_dlode * rt_num) / (3. * rt_den * rt_den)
            dp_dsigma = dp_depsilon * depsilon_dsigma + dp_dsqrt2J2 * dsqrt2J2_dsigma + dp_drt * drt_dlode * dlode_dsigma
            dp_dq = dp_dsqrt2J2 * ti.sqrt(2. / 3.)
        else:
            r_mc = (3. - sin_fai) / (6. * cos_fai)
            e_val = (3. - sin_fai) / (3. + sin_fai)
            e_val = clamp(0.5 + 1e-10, 1., e_val)
            sqpart = 4. * (1 - e_val * e_val) * cos_lode * cos_lode + 5 * e_val * e_val - 4 * e_val
            if sqpart < Threshold: sqpart = 1e-5
            m = 2. * (1 - e_val * e_val) * cos_lode + (2. * e_val - 1) * ti.sqrt(sqpart)
            if ti.abs(m) < Threshold: m = 1e-5
            l = 4. * (1 - e_val * e_val) * cos_lode * cos_lode + (2. * e_val - 1) * (2. * e_val - 1)
            r_mw = (l / m) * r_mc
            xi = 0.1
            omega = (xi * cohesion * tan_psi) * (xi * cohesion * tan_psi) + (r_mw * ti.sqrt(1.5) * sqrt2J2) * (r_mw * ti.sqrt(1.5) * sqrt2J2)
            if omega < Threshold: omega = 1e-5
            dl_dlode = -8. * (1. - e_val * e_val) * cos_lode * sin_lode
            dm_dlode = -2. * (1. - e_val * e_val) * sin_lode + (0.5 * (2. * e_val - 1.) * dl_dlode) / ti.sqrt(sqpart)
            drmw_dlode = ((m * dl_dlode) - (l * dm_dlode)) / (m * m)
            dp_depsilon = tan_psi / ti.sqrt(3.)
            dp_dsqrt2J2 = 3. * sqrt2J2 * r_mw * r_mw / (2. * ti.sqrt(omega))
            dp_dlode = (3. * sqrt2J2 * sqrt2J2 * r_mw * r_mc * drmw_dlode) / (2. * ti.sqrt(omega))
            dp_dsigma = (dp_depsilon * depsilon_dsigma) + (dp_dsqrt2J2 * dsqrt2J2_dsigma) + (dp_dlode * dlode_dsigma)
            dp_dq = dp_dsqrt2J2 * ti.sqrt(2./3.)
        
        c_peak, c_residual = self.c_peak, self.c_residual
        fai_peak, fai_residual = self.fai_peak, self.fai_residual

        softening = 0.
        if pdstrain > pdstrain_peak and pdstrain < pdstrain_residual:
            dfai_dpstrain = (fai_residual - fai_peak) / (pdstrain_residual - pdstrain_peak)
            dc_dpstrain = (c_residual - c_peak) / (pdstrain_residual - pdstrain_peak)
            df_dfai = ti.sqrt(1.5) * sqrt2J2 * (sin_fai * ti.sin(lode + PI / 3.) / (ti.sqrt(3.) * cos_fai * cos_fai) + ti.cos(lode + PI / 3.) / (3. * cos_fai * cos_fai)) + MeanStress(stress) / (cos_fai * cos_fai)
            df_dc = -1
            softening = -1. * (df_dfai * dfai_dpstrain + df_dc * dc_dpstrain) * dp_dq
        return df_dsigma, dp_dsigma, dp_dq, softening

    @ti.func
    def ComputeStressInvariant(self, stress):
        epsilon = ti.sqrt(3) * MeanStress(stress)
        sqrt2J2 = ti.sqrt(2 * ComputeInvariantJ2(stress))
        lode = ComputeLodeAngle(stress)
        return epsilon, sqrt2J2, lode

    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        de = calculate_strain_increment2D(velocity_gradient, dt)
        dw = calculate_vorticity_increment2D(velocity_gradient, dt)
        return self.core(np, previous_stress, de, dw, stateVars)

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        de = calculate_strain_increment(velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)
        return self.core(np, previous_stress, de, dw, stateVars)
    
    @ti.func
    def core(self, np, previous_stress, de, dw, stateVars): 
        bulk_modulus = self.bulk
        shear_modulus = self.shear

        pdstrain = stateVars[np].epstrain
        pdstrain_peak, pdstrain_residual = self.pdstrain_peak, self.pdstrain_residual
        c_peak, c_residual = self.c_peak, self.c_residual
        fai_peak, fai_residual = self.fai_peak, self.fai_residual
        psi_peak, psi_residual = self.psi_peak, self.psi_residual
        if pdstrain > pdstrain_peak:
            if pdstrain < pdstrain_residual:
                stateVars[np].fai = fai_residual + (fai_peak - fai_residual) * (pdstrain - pdstrain_residual) / (pdstrain_peak - pdstrain_residual)
                stateVars[np].psi = psi_residual + (psi_peak - psi_residual) * (pdstrain - pdstrain_residual) / (pdstrain_peak - pdstrain_residual)
                stateVars[np].c = c_residual + (c_peak - c_residual) * (pdstrain - pdstrain_residual) / (pdstrain_peak - pdstrain_residual)
            else:
                stateVars[np].fai = fai_residual
                stateVars[np].psi = psi_residual
                stateVars[np].c = c_residual

            apex = stateVars[np].c / ti.max(ti.tan(stateVars[np].fai), Threshold)
            if self.tensile > apex: self.tensile = ti.max(apex, Threshold)
        

        # !-- trial elastic stresses ----!
        stress = previous_stress
        sigrot = Sigrot(stress, dw)
        dstress = ElasticTensorMultiplyVector(de, shear_modulus, bulk_modulus)
        trial_stress = stress + dstress 

        # !-- compute trial stress invariants ----!
        epsilon, sqrt2J2, lode = self.ComputeStressInvariant(trial_stress)
        yield_state, yield_tension_trial, yield_shear_trial = self.YieldState(lode, sqrt2J2, epsilon, stateVars[np])
        updated_stress = trial_stress
        if yield_state == 0:
            updated_stress += sigrot
            stateVars[np].estress = VonMisesStress(updated_stress)
        elif yield_state > 0:
            Tolerance = 1e-1

            df_dsigma_trial, dp_dsigma_trial, dp_dq_trial, softening_trial = self.Compute_DfDp(yield_state, sqrt2J2, lode, trial_stress, stateVars[np])
            yield_trial = 0.
            if yield_state == 1:
                yield_trial = yield_shear_trial
            elif yield_state == 2:
                yield_trial = yield_tension_trial

            temp_matrix = ElasticTensorMultiplyVector(df_dsigma_trial, bulk_modulus, shear_modulus)
            lambda_trial = yield_trial / ((temp_matrix).dot(dp_dsigma_trial) + softening_trial)
            
            epsilon, sqrt2J2, lode = self.ComputeStressInvariant(stress)
            yield_state, yield_tension, yield_shear = self.YieldState(lode, sqrt2J2, epsilon, stateVars[np])
            _yield = MThreshold
            if yield_state == 1:
                _yield = yield_shear
            elif yield_state == 2:
                _yield = yield_tension

            df_dsigma, dp_dsigma, dp_dq, softening = self.Compute_DfDp(yield_state, sqrt2J2, lode, stress, stateVars[np])
            temp_matrix = ElasticTensorMultiplyVector(df_dsigma, bulk_modulus, shear_modulus)
            _lambda = temp_matrix.dot(de) / ((temp_matrix).dot(dp_dsigma) + softening)
            dpdstrain = 0.

            if ti.abs(_yield) < Tolerance:
                temp_matrix = ElasticTensorMultiplyVector(dp_dsigma, bulk_modulus, shear_modulus)
                updated_stress -= _lambda * temp_matrix
                dpdstrain = _lambda * dp_dq
            else:
                temp_matrix = ElasticTensorMultiplyVector(dp_dsigma_trial, bulk_modulus, shear_modulus)
                updated_stress -= lambda_trial * temp_matrix
                dpdstrain = lambda_trial * dp_dq_trial

            itr_max = 100
            for _ in range(itr_max):
                epsilon, sqrt2J2, lode = self.ComputeStressInvariant(updated_stress)

                yield_state, yield_tension_trial, yield_shear_trial = self.YieldState(lode, sqrt2J2, epsilon, stateVars[np])
                
                if yield_tension_trial < Tolerance and yield_shear_trial < Tolerance:
                    break
                
                df_dsigma_trial, dp_dsigma_trial, dp_dq_trial, softening_trial = self.Compute_DfDp(yield_state, sqrt2J2, lode, updated_stress, stateVars[np])

                if yield_state == 1:
                    yield_trial = yield_shear_trial
                elif yield_state == 2:
                    yield_trial = yield_tension_trial

                temp_matrix = ElasticTensorMultiplyVector(df_dsigma_trial, bulk_modulus, shear_modulus)
                lambda_trial = yield_trial / ((temp_matrix).dot(dp_dsigma_trial) + softening_trial)
                
                temp_matrix = ElasticTensorMultiplyVector(dp_dsigma_trial, bulk_modulus, shear_modulus)
                updated_stress -= lambda_trial * temp_matrix
                dpdstrain += lambda_trial * dp_dq_trial
            
            updated_stress += sigrot
            stateVars[np].estress = VonMisesStress(updated_stress)
            stateVars[np].epstrain += dpdstrain
        return updated_stress
    
    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stiffness, stateVars):
        ComputeElasticStiffnessTensor(np, self.bulk, self.shear, stiffness)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stiffness, stateVars):
        bulk_modulus = self.bulk
        shear_modulus = self.shear

        a1 = bulk_modulus + (4./3.) * shear_modulus
        a2 = bulk_modulus - (2./3.) * shear_modulus
        stiffness[np][0, 0] = stiffness[np][1, 1] = stiffness[np][2, 2] = a1
        stiffness[np][0, 1] = stiffness[np][0, 2] = stiffness[np][1, 2] = a2
        stiffness[np][1, 0] = stiffness[np][2, 0] = stiffness[np][2, 1] = a2
        stiffness[np][3, 3] = stiffness[np][4, 4] = stiffness[np][5, 5] = shear_modulus

        epsilon, sqrt2J2, lode = self.ComputeStressInvariant(current_stress)
        yield_state, _, _ = self.YieldState(lode, sqrt2J2, epsilon, stateVars[np])
        if yield_state > 0:
            df_dsigma, dp_dsigma, dp_dq, softening = self.Compute_DfDp(yield_state, sqrt2J2, lode, current_stress, stateVars[np])
            de_dpdsigma = ElasticTensorMultiplyVector(dp_dsigma, bulk_modulus, shear_modulus)
            de_dfdsigma = ElasticTensorMultiplyVector(df_dsigma, bulk_modulus, shear_modulus)
            dfdsigma_de_dpdsigma = df_dsigma.dot(de_dpdsigma)
            stiffness[np] -= 1. / (dfdsigma_de_dpdsigma + softening) * (de_dpdsigma.outer_product(de_dfdsigma))
    
    @ti.func
    def ComputePKStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        previous_stress = self.PK2CauchyStress(np, stateVars, previous_stress)
        stress = self.ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
        return self.Cauchy2PKStress(np, stateVars, stress)


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), epstrain: ti.types.ndarray(), cohesion: ti.types.ndarray(), friction: ti.types.ndarray(), dilation: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].epstrain = epstrain[np]
        state_vars[np].cohesion = cohesion[np]
        state_vars[np].friction = friction[np]
        state_vars[np].dilation = dilation[np]