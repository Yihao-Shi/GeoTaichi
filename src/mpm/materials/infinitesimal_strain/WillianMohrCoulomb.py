import numpy as np
import taichi as ti

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.MaterialKernel import *
from src.utils.constants import DELTA, PI, ZEROMAT3x3, ZEROVEC3f
from src.utils.MatrixFunction import Diagonal, matrix_form
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import mat3x3
from src.utils.VectorFunction import voigt_form


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
        fai = DictIO.GetAlternative(material, 'Friction', 0.) * PI / 180.
        psi = DictIO.GetAlternative(material, 'Dilation', 0.) * PI / 180.
        self.matProps[materialID].add_material(density, young, possion, c, fai, psi)
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
    soften: ti.u8
    young: float
    possion: float
    shear: float
    bulk: float
    c: float
    fai: float
    psi: float

    def add_material(self, density, young, possion, c, fai, psi):
        self.density = density
        self.young = young
        self.possion = possion

        self.shear = 0.5 * self.young / (1. + self.possion)
        self.bulk = self.young / (3. * (1 - 2. * self.possion))
        self.c = c
        self.fai = fai
        self.psi = psi

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
    # cb-geo /include/materials/infinitesimal_strain/mohr_coulomb.tcc
    @ti.func
    def YieldState(self, lode, sqrt2J2, epsilon):
        Tolerance = -1e-8
        fai, cohesion = self.fai, self.c
        cos_fai, tan_fai = ti.cos(fai), ti.tan(fai)

        yield_shear = ti.sqrt(1.5) * sqrt2J2 * (ti.sin(lode + PI/3.) / (ti.sqrt(3.) * cos_fai) + \
                                        ti.cos(lode + PI/3.) * tan_fai / 3.) + \
                                        epsilon * ti.sqrt(1./3.) * tan_fai - cohesion

        yield_state = 0
        if yield_shear > Tolerance:
            yield_state = 1
        return yield_state, yield_shear
        
    @ti.func
    def Compute_DfDp(self, yield_state, sqrt2J2, lode, stress):
        df_dsigma = dp_dsigma = ZEROVEC6f
        dp_dq = 0.
        
        fai, psi, cohesion = self.fai, self.psi, self.c
        sin_lode, cos_lode, sin_lode_PI_3, cos_lode_PI_3 = ti.sin(lode), ti.cos(lode), ti.sin(lode + PI/3.), ti.cos(lode + PI/3.)
        sin_fai, cos_fai, tan_fai, tan_psi = ti.sin(fai), ti.cos(fai), ti.tan(fai), ti.tan(psi)

        df_depsilon = tan_fai * ti.sqrt(1./3.)
        df_dsqrt2J2 = ti.sqrt(1.5) * (sin_lode_PI_3 / (ti.sqrt(3.) * cos_fai) + cos_lode_PI_3 * tan_fai / 3.)
        df_dlode = ti.sqrt(1.5) * sqrt2J2 * (cos_lode_PI_3 / (ti.sqrt(3.) * cos_fai) - sin_lode_PI_3 * tan_fai / 3.)
        
        depsilon_dsigma = DpDsigma() * ti.sqrt(3.)
        dsqrt2J2_dsigma = DqDsigma(stress) * ti.sqrt(2./3.)
        dlode_dsigma = DlodeDsigma(stress)
        df_dsigma = df_depsilon * depsilon_dsigma + df_dsqrt2J2 * dsqrt2J2_dsigma + df_dlode * dlode_dsigma
    
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
        
        return df_dsigma, dp_dsigma, dp_dq

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

        # !-- trial elastic stresses ----!
        stress = previous_stress
        sigrot = Sigrot(stress, dw)
        dstress = ElasticTensorMultiplyVector(de, bulk_modulus, shear_modulus)
        trial_stress = stress + dstress

        # !-- compute trial stress invariants ----!
        updated_stress = trial_stress
        epsilon, sqrt2J2, lode = self.ComputeStressInvariant(trial_stress)
        yield_state, yield_shear_trial = self.YieldState(lode, sqrt2J2, epsilon)
        if yield_state == 0:
            updated_stress += sigrot
            stateVars[np].estress = VonMisesStress(updated_stress)
        elif yield_state > 0:
            Tolerance = 1e-1

            df_dsigma_trial, dp_dsigma_trial, dp_dq_trial = self.Compute_DfDp(yield_state, sqrt2J2, lode, trial_stress)

            temp_matrix = ElasticTensorMultiplyVector(df_dsigma_trial, bulk_modulus, shear_modulus)
            lambda_trial = yield_shear_trial / ti.max(((temp_matrix).dot(dp_dsigma_trial)), Threshold)
            
            epsilon, sqrt2J2, lode = self.ComputeStressInvariant(stress)
            yield_state, yield_shear = self.YieldState(lode, sqrt2J2, epsilon)
            _yield = yield_shear

            df_dsigma, dp_dsigma, dp_dq = self.Compute_DfDp(yield_state, sqrt2J2, lode, stress)
            temp_matrix = ElasticTensorMultiplyVector(df_dsigma, bulk_modulus, shear_modulus)
            _lambda = temp_matrix.dot(de) / ti.max(((temp_matrix).dot(dp_dsigma)), Threshold)

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

                yield_state, yield_shear_trial = self.YieldState(lode, sqrt2J2, epsilon)
                
                if ti.abs(yield_shear_trial) < Tolerance:
                    break
                
                df_dsigma_trial, dp_dsigma_trial, dp_dq_trial = self.Compute_DfDp(yield_state, sqrt2J2, lode, updated_stress)
            
                temp_matrix = ElasticTensorMultiplyVector(df_dsigma_trial, bulk_modulus, shear_modulus)
                lambda_trial = yield_shear_trial / ((temp_matrix).dot(dp_dsigma_trial))
                
                temp_matrix = ElasticTensorMultiplyVector(dp_dsigma_trial, bulk_modulus, shear_modulus)
                updated_stress -= lambda_trial * temp_matrix
                dpdstrain += lambda_trial * dp_dq_trial
            
            updated_stress += sigrot
            stateVars[np].estress = VonMisesStress(updated_stress)
            stateVars[np].epstrain += dpdstrain
        return updated_stress

    @ti.func
    def ImplicitComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):
        de = calculate_strain_increment(velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)

        stress = previous_stress
        sigrot = Sigrot(stress, dw)
        dstress = ElasticTensorMultiplyVector(de, bulk_modulus, shear_modulus)
        trial_stress = stress + dstress + sigrot
        
        phi = stateVars[np].fai
        c = stateVars[np].c
        psi = stateVars[np].psi
        shear_modulus = self.shear
        bulk_modulus = self.bulk
        updated_stress = VigotVec2Tensor(trial_stress)

        eig_val, eig_vec = ti.sym_eig(updated_stress)
        s1, s2, s3 = eig_val[2], eig_val[1], eig_val[0]
        sin0 = ti.sin(phi)
        cos0 = ti.cos(phi)
        sin1 = ti.sin(psi)
        
        f = (s1 - s3) + (s1 + s3) * sin0 - 2. * c * cos0
        sc = ZEROVEC3f
        v0 = ZEROMAT3x3
        if f > Threshold:
            v0 = mat3x3([[eig_vec[0, 2], eig_vec[0, 1], eig_vec[0, 0]],
                        [eig_vec[1, 2], eig_vec[1, 1], eig_vec[1, 0]],
                        [eig_vec[2, 2], eig_vec[2, 1], eig_vec[2, 0]]])

            sin01 = sin0 * sin1
            qA0 = (8. * shear_modulus / 3. - 4. * bulk_modulus) * sin01
            qA1 = shear_modulus * (1. + sin0) * (1. + sin1)
            qA2 = shear_modulus * (1. - sin0) * (1. - sin1)
            qB0 = 2. * c * cos0

            gsl = 0.5 * (s1 - s2) / (shear_modulus * (1. + sin1))
            gsr = 0.5 * s2 / (shear_modulus * (1. - sin1))
            gla = 0.5 * (s1 + s2) / (shear_modulus * (3. - sin1))
            gra = 0.5 * (2. * s1 - s2) / (shear_modulus * (3. + sin1))

            qsA = qA0 - 4. * shear_modulus * (1. + sin01)
            qsB = f
            qlA = qA0 - qA1 - 2. * qA2
            qlB = 0.5 * (1. + sin0) * (s1 + s2) - qB0
            qrA = qA0 - 2. * qA1 - qA2
            qrB = (1. + sin0) * s1 - 0.5 * (1. - sin0) * s2 - qB0
            qaA = -4. * bulk_modulus * sin01
            qaB = 2. * (s1 + s2) / 3. * sin0 - qB0

            minslsr = ti.min(gsl, gsr)
            maxlara = ti.max(gla, gra)

            if minslsr > 0 and qsA * minslsr + qsB < 0:
                dl = -qsB / qsA
                ds0 = -dl * (2. * bulk_modulus - 4. * shear_modulus / 3.) * sin1
                sc[0] = s1 + ds0 - dl * (2. * shear_modulus * (1. + sin1))
                sc[1] = s2 + ds0
            elif 0 < gsl <= gla and qlA * gsl + qlB >= 0 and qlA * gla + qlB <= 0:
                dl = -qlB / qlA
                ds0 = dl * (4. * shear_modulus / 3. - 2. * bulk_modulus) * sin1
                sc[0] = sc[1] = 0.5 * (s1 + s2) + ds0 - dl * shear_modulus * (1. + sin1)
            elif 0 < gsr <= gra and qrA * gsr + qrB >= 0 and qrA * gra + qrB <= 0:
                dl = -qrB / qrA
                ds0 = dl * (4. * shear_modulus / 3. - 2. * bulk_modulus) * sin1
                sc[0] = s1 + ds0 - 2. * dl * shear_modulus * (1. + sin1)
                sc[1] = 0.5 * s2 + ds0 + dl * shear_modulus * (1.-sin1)
            elif maxlara > 0 and qaA * maxlara + qaB >= -1.e-24:
                sc[0] = sc[1] = c / ti.tan(phi)
            updated_stress = Tensor2VigotVec(v0 @ Diagonal(sc) @ v0.transpose())
        return updated_stress

    @ti.func
    def ComputePKStress(self, np, previous_stress, velocity_gradient, stateVars, dt):  
        previous_stress = self.PK2CauchyStress(np, stateVars, previous_stress)
        stress = self.ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
        return self.Cauchy2PKStress(np, stateVars, stress)

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stiffness, stateVars):
        ComputeElasticStiffnessTensor(np, self.bulk, self.shear, stiffness)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stiffness, stateVars):
        ComputeElasticStiffnessTensor(np, self.bulk, self.shear, stiffness)

        epsilon, sqrt2J2, lode = self.ComputeStressInvariant(current_stress)
        yield_state, _ = self.YieldState(lode, sqrt2J2, epsilon)
        if yield_state > 0:
            df_dsigma, dp_dsigma, _ = self.Compute_DfDp(yield_state, sqrt2J2, lode, current_stress)
            de_dpdsigma = ElasticTensorMultiplyVector(dp_dsigma, self.bulk, self.shear)
            de_dfdsigma = ElasticTensorMultiplyVector(df_dsigma, self.bulk, self.shear)
            dfdsigma_de_dpdsigma = ti.max(df_dsigma.dot(de_dpdsigma), Threshold)
            stiffness[np] -= 1. / dfdsigma_de_dpdsigma * (de_dpdsigma.outer_product(de_dfdsigma))

@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), epstrain: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].epstrain = epstrain[np]