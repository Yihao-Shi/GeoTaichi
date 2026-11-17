from src.utils.errors import SimulationStateError
import taichi as ti
import numpy as np

from src.physics_model.constitutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.constitutive_model.infinitesimal_strain.InfinitesimalStrainModel import InfinitesimalStrainModel
from src.utils.constants import DELTA, EYE, DELTA2D, ZEROVEC6F
from src.utils.MatrixFunction import eig3_symmetric
from src.utils.ObjectIO import DictIO
from src.utils.ScalarFunction import macauley, macauley_index
from src.utils.VectorFunction import voigt_tensor_dot

SMALL = 1e-10
M_P_ATM = 101.3
M_PMIN = 1e-4 * M_P_ATM
M_TOL_F = 1.0e-4
M_TOL_R = 1.0e-4
M_I1 = EYE  # 待确认
ROOT23 = ti.sqrt(2.0 / 3.0)
ONE3 = 1.0 / 3.0
TWO3 = 2.0 / 3.0


@ti.data_oriented
class SanisandMSModel(InfinitesimalStrainModel):
    density: float
    G0: float
    nu: float
    Mc: float
    c: float
    lambda_c: float
    e0: float
    ksi: float
    m: float
    h0: float
    ch: float
    nb: float
    A0: float
    nd: float
    zeta: float
    mu0: float
    beta: float

    e_init: float
    emax: float
    emin: float

    def __init__(
        self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"
    ):
        super().__init__(material_type, configuration, solver_type)
        self.stress_integration = stress_integration
        self.G0 = 0.0
        self.nu = 0.0
        self.Mc = 0.0
        self.c = 0.0
        self.lambda_c = 0.0
        self.e0 = 0.0
        self.ksi = 0.0
        self.m = 0.0
        self.h0 = 0.0
        self.ch = 0.0
        self.nb = 0.0
        self.A0 = 0.0
        self.nd = 0.0
        self.zeta = 0.0
        self.mu0 = 0.0
        self.beta = 0.0
        self.e_init = 0.0
        self.emax = 0.0
        self.emin = 0.0

    def model_initialize(self, material):
        self.material = material
        density = DictIO.get_alternative(material, "Density", 2650.0)
        G0 = DictIO.get_essential(material, "G0")
        nu = DictIO.get_alternative(material, "PoissonRatio", DictIO.get_alternative(material, "nu", 0.05))
        Mc = DictIO.get_essential(material, "Mc", "CriticalStateRatio")
        c = DictIO.get_essential(material, "c", "LodeRatio")
        lambda_c = DictIO.get_essential(material, "lambda_c", "LambdaC")
        e0 = DictIO.get_essential(material, "e0", "ReferenceVoidRatio")
        ksi = DictIO.get_essential(material, "ksi", "xi")
        m = DictIO.get_essential(material, "m", "YieldSurfaceSize")
        h0 = DictIO.get_essential(material, "h0")
        ch = DictIO.get_essential(material, "ch")
        nb = DictIO.get_essential(material, "nb")
        A0 = DictIO.get_essential(material, "A0")
        nd = DictIO.get_essential(material, "nd")
        zeta = DictIO.get_essential(material, "zeta")
        mu0 = DictIO.get_essential(material, "mu0")
        beta = DictIO.get_essential(material, "beta")
        e_init = DictIO.get_essential(material, "e_init", "InitialVoidRatio")
        emax = DictIO.get_essential(material, "emax", "MaximumVoidRatio")
        emin = DictIO.get_essential(material, "emin", "MinimumVoidRatio")
        self.add_material(
            density, G0, nu, Mc, c, lambda_c, e0, ksi, m, h0, ch, nb, A0, nd, zeta, mu0, beta, e_init, emax, emin
        )
        self.add_coupling_material(material)

    def add_material(
        self, density, G0, nu, Mc, c, lambda_c, e0, ksi, m, h0, ch, nb, A0, nd, zeta, mu0, beta, e_init, emax, emin
    ):
        self.density = density
        self.G0 = G0
        self.nu = nu
        self.Mc = Mc
        self.c = c
        self.lambda_c = lambda_c
        self.e0 = e0
        self.ksi = ksi
        self.m = m
        self.h0 = h0
        self.ch = ch
        self.nb = nb
        self.A0 = A0
        self.nd = nd
        self.zeta = zeta
        self.mu0 = mu0
        self.beta = beta

        self.e_init = e_init
        self.emax = emax
        self.emin = emin
        reference_shear_kpa = self.G0 * M_P_ATM * (2.97 - self.e_init) ** 2 / (1.0 + self.e_init)
        reference_bulk_kpa = (1.0 + self.nu) / (1.0 - 2.0 * self.nu) * reference_shear_kpa * 2.0 / 3.0
        self.shear = reference_shear_kpa * 1.0e3
        self.bulk = reference_bulk_kpa * 1.0e3
        self.young = 2.0 * self.shear * (1.0 + self.nu)
        self.poisson = self.nu
        self.max_sound_speed = self.get_sound_speed()

    def define_state_vars(self):
        return {
            "epstrain": float,
            "estress": float,
            "mVoidRatio": float,
            "mAlpha": vec6f,
            "mAlphaM": vec6f,
            "malpha_in": vec6f,
            "mMM_plus": float,
            "mMM_minus": float,
        }

    def random_field_initialize(self, parameter):
        raise SimulationStateError("SanisandMS does not support random-field material parameters yet")

    def get_sound_speed(self):
        if self.density <= 0.0:
            return 0.0
        constrained_modulus = self.bulk + 4.0 * self.shear / 3.0
        return np.sqrt(max(constrained_modulus, 0.0) / self.density)

    def add_contact_parameter(self, friction, kn, kt):
        self.friction = friction
        self.kn = kn
        self.kt = kt

    def print_message(self, materialID):
        self.print_console_header()
        print("Constitutive model: SANISAND-MS")
        print("Material ID: ", materialID)
        print("Density: ", self.density)
        print("G0", self.G0)
        print("nu: ", self.nu)
        print("Mc: ", self.Mc)
        print("c: ", self.c)
        print("lambda_c: ", self.lambda_c)
        print("e0: ", self.e0)
        print("ksi: ", self.ksi)
        print("m: ", self.m)
        print("h0: ", self.h0)
        print("ch: ", self.ch)
        print("nb: ", self.nb)
        print("A0: ", self.A0)
        print("nd: ", self.nd)
        print("zeta: ", self.zeta)
        print("mu0: ", self.mu0)
        print("beta: ", self.beta)

    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, state_variables):
        stress = -particle[np].stress * 1.0e-3
        state_variables[np].estress = von_mises_stress(stress)
        p = compute_stress_invariant_i1(stress) / 3.0
        rr = ZEROVEC6F
        if p > M_PMIN:
            rr = deviatoric_tensor(stress) / p

        state_variables[np].mAlpha = rr
        state_variables[np].mAlphaM = rr
        state_variables[np].malpha_in = rr
        state_variables[np].mMM_plus = self.m
        state_variables[np].mMM_minus = 0.0
        state_variables[np].mVoidRatio = clamp(self.emin, self.emax, self.e_init)
        state_variables[np].epstrain = 0.0

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, state_variables, dt):
        return (DELTA + velocity_gradient * dt[None]).determinant()

    @ti.func
    def update_particle_volume_2d(self, np, velocity_gradient, state_variables, dt):
        return (DELTA2D + velocity_gradient * dt[None]).determinant()

    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, state_variables, dt):
        return 1.0 + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])

    @ti.func
    def update_internal_variables(self, np, dpdstrain, state_variables):
        state_variables[np].epstrain += dpdstrain

    @ti.func
    def update_state_variables(self, np, stress, state_variables):
        state_variables[np].estress = von_mises_stress(stress)

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, state_variables):
        compression_stress_kpa = -current_stress * 1.0e-3
        shear, bulk = self.get_elastic_moduli(compression_stress_kpa, state_variables[np].mVoidRatio, 1)
        return compute_elastic_stiffness_tensor(bulk * 1.0e3, shear * 1.0e3)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, state_variables):
        stiffness_matrix = self.compute_elastic_tensor(np, current_stress, state_variables)
        compression_stress_kpa = -current_stress * 1.0e-3
        state_vars = state_variables[np]
        yield_function = self.get_f(compression_stress_kpa, state_vars.mAlpha)
        if yield_function > -M_TOL_F:
            shear, bulk = self.get_elastic_moduli(compression_stress_kpa, state_vars.mVoidRatio, 1)
            n, d, b, bM, cos3theta, h, hM, psi, r_btheta, r_dtheta, b0, A, D, B, C, R, Z = self.get_state_dependent(
                compression_stress_kpa,
                state_vars.mAlpha,
                state_vars.mAlphaM,
                state_vars.mMM_plus,
                state_vars.mMM_minus,
                state_vars.malpha_in,
                state_vars.mVoidRatio,
            )
            stiffness_matrix = (
                self.get_elasto_plastic_tangent(
                    compression_stress_kpa,
                    1.0,
                    ZEROVEC6F,
                    ZEROVEC6F,
                    shear,
                    bulk,
                    B,
                    C,
                    D,
                    h,
                    n,
                    d,
                    b,
                )
                * 1.0e3
            )
        return stiffness_matrix

    @ti.func
    def compute_stress2d(self, np, previous_stress, velocity_gradient, state_variables, dt):
        ############################## STEP1 ##############################
        strain_rate = calculate_strain_rate2d(velocity_gradient)
        de = calculate_strain_increment2d(velocity_gradient, dt)
        dw = calculate_vorticity_increment2d(velocity_gradient, dt)
        previous_stress = self.explicit_integration(np, previous_stress, de, strain_rate, dw, state_variables)
        return previous_stress

    @ti.func
    def compute_stress(self, np, previous_stress, velocity_gradient, state_variables, dt):
        strain_rate = calculate_strain_rate(velocity_gradient)
        de = calculate_strain_increment(velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)
        previous_stress = self.explicit_integration(np, previous_stress, de, strain_rate, dw, state_variables)
        return previous_stress

    @ti.func
    def explicit_integration(self, np, previous_stress, strain_inc, strain_rate, dw, state_variables):
        m_sigma_n = -previous_stress * 1.0e-3
        # mSigma_n += stateVars[np].stress_visc   # 阻尼项
        # mEpsilon_n = - curstrain
        # mEpsilon = - nextstrain
        dstrain = -strain_inc

        m_alpha_n = state_variables[np].mAlpha
        m_alpha_m_n = state_variables[np].mAlphaM
        malpha_in_n = state_variables[np].malpha_in
        m_mm_plus_n = state_variables[np].mMM_plus
        m_mm_minus_n = state_variables[np].mMM_minus

        # 按照Fortran版本的理解来的
        cur_void_ratio = state_variables[np].mVoidRatio
        mG, mK = self.get_elastic_moduli(m_sigma_n, cur_void_ratio, 1)
        mCe = compute_elastic_stiffness_tensor(mK, mG)

        trial_direction = mCe @ dstrain
        # trialDirection = ElasticTensorMultiplyVector(mEpsilon - mEpsilon_n, mK, mG)
        malpha_in = ZEROVEC6F
        if voigt_tensor_dot(m_alpha_n - malpha_in_n, trial_direction) < 0.0:
            malpha_in = m_alpha_n
        else:
            malpha_in = malpha_in_n

        m_sigma = m_sigma_n
        m_alpha = m_alpha_n
        m_alpha_m = m_alpha_m_n
        m_mm_plus = m_mm_plus_n
        m_mm_minus = m_mm_minus_n
        if voigt_tensor_norm(dstrain) > 1.0e-16:
            m_sigma, m_alpha, m_alpha_m, m_mm_plus, m_mm_minus = self.explicit_integrator(
                m_sigma_n,
                dstrain,
                cur_void_ratio,
                m_alpha_n,
                m_alpha_m_n,
                m_mm_plus_n,
                m_mm_minus_n,
                malpha_in,
                mG,
                mK,
            )

        # (
        #     xN1,
        #     xN2,
        #     xN3,
        #     sNextStress1a,
        #     sNextStress2a,
        #     sNextStress3a,
        #     PNextStress,
        #     QNextStress,
        # ) = Eig3Symmetric(mSigma)
        # if min(sNextStress1a, sNextStress2a, sNextStress3a) >= 0.0:
        (
            xN1,
            xN2,
            xN3,
            s_next_stress1a,
            s_next_stress2a,
            s_next_stress3a,
            p_next_stress,
            q_next_stress,
        ) = eig3_symmetric(m_sigma)
        if min(s_next_stress1a, s_next_stress2a, s_next_stress3a) < 0.0:
            m_sigma = M_PMIN * M_I1
            # p = ComputeStressInvariantI1(mSigma) / 3.
            # rr = DeviatoricTensor(mSigma) / p
            # q_Norm_New = VoigtTensorNorm(rr)
            # q_Norm_New = ti.sqrt(3./ 2.) * q_Norm_New
            m_alpha = ZEROVEC6F
            m_alpha_m = ZEROVEC6F
            malpha_in = ZEROVEC6F
            m_mm_plus = self.m
            m_mm_minus = 0.0

        # p = ComputeStressInvariantI1(mSigma) / 3.0
        # if p >= 0.0:
        m_void_ratio = cur_void_ratio - (1 + cur_void_ratio) * compute_stress_invariant_i1(dstrain)
        m_void_ratio = clamp(self.emin, self.emax, m_void_ratio)

        state_variables[np].malpha_in = malpha_in
        state_variables[np].mAlpha = m_alpha
        state_variables[np].mAlphaM = m_alpha_m
        state_variables[np].mMM_plus = m_mm_plus
        state_variables[np].mMM_minus = m_mm_minus
        state_variables[np].mVoidRatio = m_void_ratio

        # sigrot = Sigrot(mSigma_n, -dw)
        # mSigma += sigrot

        # p = ComputeStressInvariantI1(mSigma) / 3.
        # if p < m_Pmin:
        #     mSigma = DeviatoricTensor(mSigma) + m_Pmin * mI1

        updated_stress = -m_sigma * 1.0e3
        state_variables[np].estress = von_mises_stress(updated_stress)
        return updated_stress

    @ti.func
    def explicit_integrator(
        self,
        cur_stress,
        strain_inc,
        cur_void_ratio,
        cur_alpha,
        cur_alpha_m,
        cur_mm_plus,
        cur_mm_minus,
        curalpha_in,
        G,
        K,
    ):
        p_tr_pos = True

        m_e_init = self.e_init

        # NextVoidRatio = m_e_init - (1 + m_e_init) * ComputeStressInvariantI1(NextStrain)
        dStrain = strain_inc
        aC = compute_elastic_stiffness_tensor(K, G)
        dSigma = aC @ dStrain
        # dSigma = ElasticTensorMultiplyVector(dStrain, K, G)
        next_stress = cur_stress + dSigma
        f = self.get_f(next_stress, cur_alpha)
        p = compute_stress_invariant_i1(next_stress) / 3.0

        # 局部变量初始化
        next_alpha = ZEROVEC6F
        next_alpha_m = ZEROVEC6F
        next_mm_plus = 0.0
        next_mm_minus = 0.0
        runge_kutta = False
        rk_stress = cur_stress
        rk_strain = dStrain

        # if VoigtTensorNorm(dStrain) >= small:
        #     return

        if p < M_PMIN:
            p_tr_pos = False

        if p_tr_pos and f <= M_TOL_F:  # This is a pure elastic loading/unloading
            next_alpha = cur_alpha
            next_alpha_m = cur_alpha_m
            next_mm_plus = cur_mm_plus
            next_mm_minus = cur_mm_minus
        else:
            fn = self.get_f(cur_stress, cur_alpha)
            pn = compute_stress_invariant_i1(cur_stress) / 3.0
            if fn > M_TOL_F:  # This is an illegal stress state! This shouldn't happen
                runge_kutta = True

            elif fn < -M_TOL_F:  # This is a transition from elastic to plastic
                elastic_ratio = self.intersection_factor(cur_stress, dStrain, cur_void_ratio, cur_alpha, 0.0, 1.0)
                dSigma = aC @ (elastic_ratio * dStrain)
                # dSigma = ElasticTensorMultiplyVector(elasticRatio * (NextStrain - CurStrain), K, G)
                rk_stress = cur_stress + dSigma
                rk_strain = (1.0 - elastic_ratio) * dStrain
                runge_kutta = True

            elif abs(fn) < M_TOL_F:
                denom = voigt_tensor_norm(dSigma)
                if denom == 0:
                    denom = 1.0
                ratio = voigt_tensor_dot(self.get_normal_to_yield(cur_stress, cur_alpha), dSigma) / denom
                if ratio > -ti.sqrt(M_TOL_F):
                    # This is a pure plastic step
                    runge_kutta = True
                else:
                    # This is an elastic unloding followed by plastic loading
                    elastic_ratio = self.intersection_factor_unloading(cur_stress, dStrain, cur_void_ratio, cur_alpha)
                    dSigma = aC @ (elastic_ratio * dStrain)
                    # dSigma = ElasticTensorMultiplyVector(elasticRatio * (NextStrain - CurStrain), K, G)
                    rk_stress = cur_stress + dSigma
                    rk_strain = (1.0 - elastic_ratio) * dStrain
                    runge_kutta = True

        # One call site avoids inlining the complete adaptive RK integrator four times.
        if runge_kutta:
            (
                next_stress,
                next_alpha,
                next_alpha_m,
                next_mm_plus,
                next_mm_minus,
                next_void_ratio,
            ) = self.runge_kutta4(
                rk_stress,
                rk_strain,
                cur_void_ratio,
                cur_alpha,
                cur_alpha_m,
                cur_mm_plus,
                cur_mm_minus,
                curalpha_in,
            )
        return next_stress, next_alpha, next_alpha_m, next_mm_plus, next_mm_minus

    @ti.func
    def get_elastic_moduli(self, sigma, en, elast_flag):
        pn = compute_stress_invariant_i1(sigma) / 3.0
        if pn <= M_PMIN:
            pn = M_PMIN

        # G, K = 0., 0.
        # if ElastFlag == 0:
        #     G = self.G0 * m_P_atm * pow((2.97 - en), 2) / (1 + en)
        # else:
        G = self.G0 * M_P_ATM * pow((2.97 - en), 2) / (1 + en) * ti.sqrt(pn / M_P_ATM)
        K = (1 + self.nu) / (1 - 2 * self.nu) * G * 2.0 / 3.0
        return G, K

    @ti.func
    def get_f(self, stress, alpha):
        s = deviatoric_tensor(stress)
        p = compute_stress_invariant_i1(stress) / 3.0
        s = s - p * alpha
        return voigt_tensor_norm(s) - ROOT23 * self.m * p

    @ti.func
    def get_psi(self, e, p):
        return e - (self.e0 - self.lambda_c * pow((p / M_P_ATM), self.ksi))

    @ti.func
    def get_lode_angle(self, stress):
        cos3_theta = ti.sqrt(6.0) * compute_stress_invariant_i1(
            symmetric_tensor_single_dot(stress, symmetric_tensor_single_dot(stress, stress))
        )
        cos3_theta = clamp(-1.0, 1.0, cos3_theta)
        return cos3_theta

    @ti.func
    def g(self, cos3theta, c):
        return 2 * c / ((1 + c) - (1 - c) * cos3theta)

    @ti.func
    def get_state_dependent(self, stress, alpha, alpha_m, MM_plus, MM_minus, alpha_in, e):
        zerozo = 1.0e-2
        zero_plus_tol = 1.0e-30
        tol = 1.0e-15

        D_factor = 1.0
        p = compute_stress_invariant_i1(stress) / 3.0
        stress_use = stress
        if p < M_PMIN:
            stress_use = deviatoric_tensor(stress) + M_PMIN * M_I1
            p = compute_stress_invariant_i1(stress_use) / 3.0

        r = deviatoric_tensor(stress_use) / p

        n = self.get_normal_to_yield(stress_use, alpha)

        alpha_alpha_in_dot_n = voigt_tensor_dot(alpha - alpha_in, n)

        psi = self.get_psi(e, p)

        cos3Theta = self.get_lode_angle(n)

        # numgeo对Mb的修改 Mb = nb * <-psi>
        r_btheta = self.g(cos3Theta, self.c) * self.Mc * ti.exp(-self.nb * psi) - self.m
        r_btheta_plu_spi = self.g(-cos3Theta, self.c) * self.Mc * ti.exp(-self.nb * psi) - self.m

        alpha_btheta = ROOT23 * r_btheta * n

        alpha_btheta_plu_spi = -ROOT23 * r_btheta_plu_spi * n

        r_dtheta = self.g(cos3Theta, self.c) * self.Mc * ti.exp(self.nd * psi) - self.m

        alpha_dtheta = ROOT23 * r_dtheta * n

        b0 = self.G0 * self.h0 * (1.0 - self.ch * e) / ti.sqrt(p / M_P_ATM)

        d = alpha_dtheta - alpha

        b = alpha_btheta - alpha

        # the memory surface and image points on it
        MM = max(self.m, MM_plus + MM_minus)
        # MM = max(self.m, MM_plus + MM_minus) # 来源于Fortran

        r_alpha_m = alpha_m + ROOT23 * (MM - self.m) * n
        r_m_tilde = alpha_m - ROOT23 * MM * n
        r_m_alphatilde = alpha_m - ROOT23 * (MM - self.m) * n
        rM = alpha_m + ROOT23 * MM * n

        b_m_distance = voigt_tensor_dot(r_alpha_m - alpha, n)
        b_m_distance_tilde = voigt_tensor_dot(alpha - r_m_alphatilde, n)

        if b_m_distance < 0.0:
            r_alpha_m = alpha
            rM = r
            b_m_distance = 0.0

        r_tilde = alpha - ROOT23 * self.m * n
        x2 = 0.0
        f_shr = 0.0
        if b_m_distance_tilde < 0.0:
            r_m_tilde = r_tilde
            r_m_alphatilde = alpha
            b_m_distance = 2 * ROOT23 * (MM - self.m)

        # snM = rM - r
        # res100 = VoigtTensorNorm(snM)
        # if ti.abs(res100) < 0.01:
        #     snM = n
        # else:
        #     snM = snM / res100

        x3 = voigt_tensor_dot(n, rM - r_m_tilde)

        x2 = voigt_tensor_dot(n, r_tilde - r_m_tilde)

        if x2 < 0.0:
            f_shr = 0.0
        else:
            f_shr = x2 / x3

        f_shr = clamp(0.0, 1.0, f_shr)

        # Change
        gtheta_plus_pi = 2 * self.c / ((1 + self.c) + (1 - self.c) * cos3Theta)
        alphad_tilde = ROOT23 * (gtheta_plus_pi * self.Mc * ti.exp(self.nd * psi) - self.m) * ((-1) * n)

        b_d_m_tilde = voigt_tensor_dot(alphad_tilde - alpha_in, n)
        bref = max(2 * ROOT23 * self.m, voigt_tensor_dot(alpha_btheta - alpha_btheta_plu_spi, n))
        bref_D = max(2 * ROOT23 * self.m, voigt_tensor_norm(alpha_in))

        b_d_r = voigt_tensor_dot(d, n)

        A = self.A0 * ti.exp(self.beta * macauley(b_d_m_tilde) / bref_D)

        D = A * b_d_r

        if p < 0.001 * M_P_ATM:
            D_factor = 1.0 / (1.0 + (ti.exp(7.6349 - 7.2713 * p)))
        else:
            D_factor = 1.0

        D *= D_factor

        B = 1.0 + 1.5 * (1 - self.c) / self.c * self.g(cos3Theta, self.c) * cos3Theta

        C = 3.0 * ti.sqrt(1.5) * (1 - self.c) / self.c * self.g(cos3Theta, self.c)

        R = B * n - C * (symmetric_tensor_single_dot(n, n) - ONE3 * M_I1) + ONE3 * D * M_I1

        Z = MM / self.zeta * f_shr

        bM = alpha_btheta - r_alpha_m
        b_b_m = voigt_tensor_dot(bM, n)

        b_r_m_rin = voigt_tensor_dot(r_alpha_m - alpha_in, n)
        # if b_rM_rin < 1.0e-7:
        #     b_rM_rin = 1.0e-7
        b_r_m_rin = abs(b_r_m_rin) + 0.001
        hM = min(1.0e10, 0.5 * b0 / b_r_m_rin + 0.5 / ROOT23 * Z * macauley(-D) / b_b_m)

        # if AlphaAlphaInDotN < small:
        #     AlphaAlphaInDotN = small

        alpha_alpha_in_dot_n = abs(alpha_alpha_in_dot_n) + 0.001
        mem = self.mu0 * pow(p / M_P_ATM, 0.5) * pow(b_m_distance / bref, 2)
        h = 0.0
        if mem > 9.2103:
            h = b0 / alpha_alpha_in_dot_n * ti.exp(9.2103)
        else:
            h = b0 / alpha_alpha_in_dot_n * ti.exp(mem)
        h = min(1.0e7, h)

        return n, d, b, bM, cos3Theta, h, hM, psi, r_btheta, r_dtheta, b0, A, D, B, C, R, Z

    @ti.func
    def get_elasto_plastic_tangent(self, next_stress, next_d_gamma, cur_strain, next_strain, G, K, B, C, D, h, n, d, b):
        p = compute_stress_invariant_i1(next_stress) / 3.0
        if p < SMALL:
            p = SMALL
        r = deviatoric_tensor(next_stress) / p
        Kp = TWO3 * p * h * voigt_tensor_dot(b, n)

        elastic_stiffness = compute_elastic_stiffness_tensor(K, G)
        flow_direction = voigt_to_covariant(
            (B * n) - (C * (symmetric_tensor_single_dot(n, n) - ONE3 * M_I1)) + (ONE3 * D * M_I1)
        )

        temp1 = elastic_stiffness @ voigt_to_covariant(flow_direction)
        temp2 = elastic_stiffness.transpose() @ voigt_to_covariant(n - ONE3 * voigt_tensor_dot(n, r) * M_I1)
        denominator = voigt_tensor_dot(temp2, flow_direction) + Kp

        tangent = elastic_stiffness
        if ti.abs(denominator) >= SMALL:
            tangent = elastic_stiffness - (macauley_index(next_d_gamma) / denominator) * temp1.outer_product(temp2)
        return tangent

    @ti.func
    def runge_kutta4(
        self, cur_stress, dStrain, cur_void_ratio, cur_alpha, cur_alpha_m, cur_mm_plus, cur_mm_minus, curalpha_in
    ):
        T = 0.0
        dT = 1.0
        dT_min = 1.0e-3
        tol_e = M_TOL_R

        m_e_init = self.e_init

        # CurVoidRatio = m_e_init - (1 + m_e_init) * ComputeStressInvariantI1(CurStrain)
        next_void_ratio = cur_void_ratio - (1 + cur_void_ratio) * compute_stress_invariant_i1(dStrain)
        next_void_ratio = clamp(self.emin, self.emax, next_void_ratio)

        # G, K = self.GetElasticModuli(CurStress, CurVoidRatio, 1)

        # aC = ComputeElasticStiffnessTensor(K, G)

        next_stress = cur_stress
        next_alpha = cur_alpha
        next_alpha_m = cur_alpha_m
        next_mm_plus = cur_mm_plus
        next_mm_minus = cur_mm_minus

        p = compute_stress_invariant_i1(next_stress) / 3.0
        if p < M_PMIN:
            next_stress = deviatoric_tensor(next_stress) + M_PMIN * M_I1
            p = compute_stress_invariant_i1(next_stress) / 3.0

        a_cep_consistent = ti.Matrix.zero(float, 6, 6)
        n_trial = self.get_normal_to_yield(next_stress, next_alpha)

        MM_trial = next_mm_plus + next_mm_minus
        r_alpha_m_trial = next_alpha_m + ROOT23 * (MM_trial - self.m) * n_trial
        r_m_alphatilde_trial = next_alpha_m - ROOT23 * (MM_trial - self.m) * n_trial

        b_m_distance_trial = voigt_tensor_dot(r_alpha_m_trial - next_alpha, n_trial)
        b_m_distance_tilde_trial = voigt_tensor_dot(next_alpha - r_m_alphatilde_trial, n_trial)

        done = False
        while T < 1.0 and not done:
            next_void_ratio = cur_void_ratio - (1.0 + cur_void_ratio) * compute_stress_invariant_i1(T * dStrain)
            next_void_ratio = clamp(self.emin, self.emax, next_void_ratio)

            dVolStrain = dT * compute_stress_invariant_i1(dStrain)
            dDevStrain = dT * deviatoric_tensor(dStrain)

            # Calc Delta 1
            this_sigma = next_stress
            this_alpha = next_alpha
            this_alpha_m = next_alpha_m
            this_mm_plus = next_mm_plus
            this_mm_minus = next_mm_minus
            this_void_ratio = next_void_ratio
            p = compute_stress_invariant_i1(this_sigma) / 3.0
            if p < M_PMIN:
                this_sigma = deviatoric_tensor(this_sigma) + M_PMIN * M_I1
                p = compute_stress_invariant_i1(this_sigma) / 3.0

            r = deviatoric_tensor(this_sigma) / p

            G, K = self.get_elastic_moduli(this_sigma, this_void_ratio, 1.0)
            n, d, b, bM, cos3_theta, h, hM, psi, r_btheta, r_dtheta, b0, A, D, B, C, R, Z = self.get_state_dependent(
                this_sigma, this_alpha, this_alpha_m, this_mm_plus, this_mm_minus, curalpha_in, this_void_ratio
            )

            # r = DeviatoricTensor(NextStress) / p
            Kp = TWO3 * p * h * voigt_tensor_dot(b, n)
            temp4 = (
                Kp
                + 2.0
                * G
                * (
                    B
                    - C * compute_stress_invariant_i1(symmetric_tensor_single_dot(n, symmetric_tensor_single_dot(n, n)))
                )
                - K * D * voigt_tensor_dot(n, r)
            )
            if abs(temp4) < SMALL:
                temp4 = SMALL
            next_d_gamma = (2.0 * G * voigt_tensor_dot(n, dDevStrain) - K * dVolStrain * voigt_tensor_dot(n, r)) / temp4
            dSigma1 = (
                2.0 * G * voigt_to_contravariant(dDevStrain)
                + K * dVolStrain * M_I1
                - macauley(next_d_gamma)
                * (2.0 * G * (B * n - C * (symmetric_tensor_single_dot(n, n) - 1.0 / 3.0 * M_I1)) + K * D * M_I1)
            )
            dAlpha1 = macauley(next_d_gamma) * TWO3 * h * b
            dAlphaM1 = macauley(next_d_gamma) * TWO3 * hM * bM
            dPStrain1 = next_d_gamma * voigt_to_covariant(R)
            depsilon_pv = compute_stress_invariant_i1(dPStrain1)
            dMM1_plus = 1 / ROOT23 * voigt_tensor_dot(dAlphaM1, n)
            dMM1_minus = -Z * macauley(-depsilon_pv)

            # Calc Delta 2
            this_sigma = next_stress + 0.5 * dSigma1
            this_alpha = next_alpha + 0.5 * dAlpha1
            this_alpha_m = next_alpha_m + 0.5 * dAlphaM1
            this_mm_plus = next_mm_plus + 0.5 * dMM1_plus
            this_mm_minus = next_mm_minus + 0.5 * dMM1_minus
            # thisVoidRatio = m_e_init - (1.0 + m_e_init) * ComputeStressInvariantI1(NextStrain + 0.5 * dPStrain1)
            # thisVoidRatio = clamp(self.emin, self.emax, thisVoidRatio)

            p = compute_stress_invariant_i1(this_sigma) / 3.0
            if p < M_PMIN:
                this_sigma = deviatoric_tensor(this_sigma) + M_PMIN * M_I1
                p = compute_stress_invariant_i1(this_sigma) / 3.0

            r = deviatoric_tensor(this_sigma) / p

            G, K = self.get_elastic_moduli(this_sigma, this_void_ratio, 1.0)
            n, d, b, bM, cos3_theta, h, hM, psi, r_btheta, r_dtheta, b0, A, D, B, C, R, Z = self.get_state_dependent(
                this_sigma, this_alpha, this_alpha_m, this_mm_plus, this_mm_minus, curalpha_in, this_void_ratio
            )

            Kp = TWO3 * p * h * voigt_tensor_dot(b, n)
            temp4 = (
                Kp
                + 2.0
                * G
                * (
                    B
                    - C * compute_stress_invariant_i1(symmetric_tensor_single_dot(n, symmetric_tensor_single_dot(n, n)))
                )
                - K * D * voigt_tensor_dot(n, r)
            )
            if abs(temp4) < SMALL:
                temp4 = SMALL
            next_d_gamma = (2.0 * G * voigt_tensor_dot(n, dDevStrain) - K * dVolStrain * voigt_tensor_dot(n, r)) / temp4
            dSigma2 = (
                2.0 * G * voigt_to_contravariant(dDevStrain)
                + K * dVolStrain * M_I1
                - macauley(next_d_gamma)
                * (2.0 * G * (B * n - C * (symmetric_tensor_single_dot(n, n) - 1.0 / 3.0 * M_I1)) + K * D * M_I1)
            )
            dAlpha2 = macauley(next_d_gamma) * TWO3 * h * b
            dAlphaM2 = macauley(next_d_gamma) * TWO3 * hM * bM
            dPStrain2 = next_d_gamma * voigt_to_covariant(R)
            depsilon_pv = compute_stress_invariant_i1(dPStrain2)
            dMM2_plus = 1 / ROOT23 * voigt_tensor_dot(dAlphaM2, n)
            dMM2_minus = -Z * macauley(-depsilon_pv)

            # Calc Delta 3
            this_sigma = next_stress + 0.25 * (dSigma1 + dSigma2)
            this_alpha = next_alpha + 0.25 * (dAlpha1 + dAlpha2)
            this_alpha_m = next_alpha_m + 0.25 * (dAlphaM1 + dAlphaM2)
            this_mm_plus = next_mm_plus + 0.25 * (dMM1_plus + dMM2_plus)
            this_mm_minus = next_mm_minus + 0.25 * (dMM1_minus + dMM2_minus)
            # thisVoidRatio = m_e_init - (1.0 + m_e_init) * ComputeStressInvariantI1(NextStrain + 0.25 * (dPStrain1 + dPStrain2))
            # thisVoidRatio = clamp(self.emin, self.emax, thisVoidRatio)

            p = compute_stress_invariant_i1(this_sigma) / 3.0
            if p < M_PMIN:
                this_sigma = deviatoric_tensor(this_sigma) + M_PMIN * M_I1
                p = compute_stress_invariant_i1(this_sigma) / 3.0

            r = deviatoric_tensor(this_sigma) / p

            G, K = self.get_elastic_moduli(this_sigma, this_void_ratio, 1.0)
            n, d, b, bM, cos3_theta, h, hM, psi, r_btheta, r_dtheta, b0, A, D, B, C, R, Z = self.get_state_dependent(
                this_sigma, this_alpha, this_alpha_m, this_mm_plus, this_mm_minus, curalpha_in, this_void_ratio
            )

            Kp = TWO3 * p * h * voigt_tensor_dot(b, n)
            temp4 = (
                Kp
                + 2.0
                * G
                * (
                    B
                    - C * compute_stress_invariant_i1(symmetric_tensor_single_dot(n, symmetric_tensor_single_dot(n, n)))
                )
                - K * D * voigt_tensor_dot(n, r)
            )
            if abs(temp4) < SMALL:
                temp4 = SMALL
            next_d_gamma = (2.0 * G * voigt_tensor_dot(n, dDevStrain) - K * dVolStrain * voigt_tensor_dot(n, r)) / temp4
            dSigma3 = (
                2.0 * G * voigt_to_contravariant(dDevStrain)
                + K * dVolStrain * M_I1
                - macauley(next_d_gamma)
                * (2.0 * G * (B * n - C * (symmetric_tensor_single_dot(n, n) - 1.0 / 3.0 * M_I1)) + K * D * M_I1)
            )
            dAlpha3 = macauley(next_d_gamma) * TWO3 * h * b
            dAlphaM3 = macauley(next_d_gamma) * TWO3 * hM * bM
            dPStrain3 = next_d_gamma * voigt_to_covariant(R)
            depsilon_pv = compute_stress_invariant_i1(dPStrain3)
            dMM3_plus = 1 / ROOT23 * voigt_tensor_dot(dAlphaM3, n)
            dMM3_minus = -Z * macauley(-depsilon_pv)

            # Calc Delta 4
            this_sigma = next_stress - dSigma2 + 2 * dSigma3
            this_alpha = next_alpha - dAlpha2 + 2 * dAlpha3
            this_alpha_m = next_alpha_m - dAlphaM2 + 2 * dAlphaM3
            this_mm_plus = next_mm_plus - dMM2_plus + 2 * dMM3_plus
            this_mm_minus = next_mm_minus - dMM2_minus + 2 * dMM3_minus
            # thisVoidRatio = m_e_init - (1.0 + m_e_init) * ComputeStressInvariantI1(NextStrain - dPStrain2 + 2 * dPStrain3)
            # thisVoidRatio = clamp(self.emin, self.emax, thisVoidRatio)

            p = compute_stress_invariant_i1(this_sigma) / 3.0
            if p < M_PMIN:
                this_sigma = deviatoric_tensor(this_sigma) + M_PMIN * M_I1
                p = compute_stress_invariant_i1(this_sigma) / 3.0

            r = deviatoric_tensor(this_sigma) / p

            G, K = self.get_elastic_moduli(this_sigma, this_void_ratio, 1.0)
            n, d, b, bM, cos3_theta, h, hM, psi, r_btheta, r_dtheta, b0, A, D, B, C, R, Z = self.get_state_dependent(
                this_sigma, this_alpha, this_alpha_m, this_mm_plus, this_mm_minus, curalpha_in, this_void_ratio
            )

            Kp = TWO3 * p * h * voigt_tensor_dot(b, n)
            temp4 = (
                Kp
                + 2.0
                * G
                * (
                    B
                    - C * compute_stress_invariant_i1(symmetric_tensor_single_dot(n, symmetric_tensor_single_dot(n, n)))
                )
                - K * D * voigt_tensor_dot(n, r)
            )
            if abs(temp4) < SMALL:
                temp4 = SMALL
            next_d_gamma = (2.0 * G * voigt_tensor_dot(n, dDevStrain) - K * dVolStrain * voigt_tensor_dot(n, r)) / temp4
            dSigma4 = (
                2.0 * G * voigt_to_contravariant(dDevStrain)
                + K * dVolStrain * M_I1
                - macauley(next_d_gamma)
                * (2.0 * G * (B * n - C * (symmetric_tensor_single_dot(n, n) - 1.0 / 3.0 * M_I1)) + K * D * M_I1)
            )
            dAlpha4 = macauley(next_d_gamma) * TWO3 * h * b
            dAlphaM4 = macauley(next_d_gamma) * TWO3 * hM * bM
            dPStrain4 = next_d_gamma * voigt_to_covariant(R)
            depsilon_pv = compute_stress_invariant_i1(dPStrain4)
            dMM4_plus = 1 / ROOT23 * voigt_tensor_dot(dAlphaM4, n)
            dMM4_minus = -Z * macauley(-depsilon_pv)

            # Calc Delta 5
            this_sigma = next_stress + (7 * dSigma1 + 10 * dSigma2 + dSigma4) / 27
            this_alpha = next_alpha + (7 * dAlpha1 + 10 * dAlpha2 + dAlpha4) / 27
            this_alpha_m = next_alpha_m + (7 * dAlphaM1 + 10 * dAlphaM2 + dAlphaM4) / 27
            this_mm_plus = next_mm_plus + (7 * dMM1_plus + 10 * dMM2_plus + dMM4_plus) / 27
            this_mm_minus = next_mm_minus + (7 * dMM1_minus + 10 * dMM2_minus + dMM4_minus) / 27
            # thisVoidRatio = m_e_init - (1.0 + m_e_init) * ComputeStressInvariantI1(NextStrain + (7 * dPStrain1 + 10 * dPStrain2 + dPStrain4) / 27)
            # thisVoidRatio = clamp(self.emin, self.emax, thisVoidRatio)

            p = compute_stress_invariant_i1(this_sigma) / 3.0
            if p < M_PMIN:
                this_sigma = deviatoric_tensor(this_sigma) + M_PMIN * M_I1
                p = compute_stress_invariant_i1(this_sigma) / 3.0

            r = deviatoric_tensor(this_sigma) / p

            G, K = self.get_elastic_moduli(this_sigma, this_void_ratio, 1.0)
            n, d, b, bM, cos3_theta, h, hM, psi, r_btheta, r_dtheta, b0, A, D, B, C, R, Z = self.get_state_dependent(
                this_sigma, this_alpha, this_alpha_m, this_mm_plus, this_mm_minus, curalpha_in, this_void_ratio
            )

            Kp = TWO3 * p * h * voigt_tensor_dot(b, n)
            temp4 = (
                Kp
                + 2.0
                * G
                * (
                    B
                    - C * compute_stress_invariant_i1(symmetric_tensor_single_dot(n, symmetric_tensor_single_dot(n, n)))
                )
                - K * D * voigt_tensor_dot(n, r)
            )
            if abs(temp4) < SMALL:
                temp4 = SMALL
            next_d_gamma = (2.0 * G * voigt_tensor_dot(n, dDevStrain) - K * dVolStrain * voigt_tensor_dot(n, r)) / temp4
            dSigma5 = (
                2.0 * G * voigt_to_contravariant(dDevStrain)
                + K * dVolStrain * M_I1
                - macauley(next_d_gamma)
                * (2.0 * G * (B * n - C * (symmetric_tensor_single_dot(n, n) - 1.0 / 3.0 * M_I1)) + K * D * M_I1)
            )
            dAlpha5 = macauley(next_d_gamma) * TWO3 * h * b
            dAlphaM5 = macauley(next_d_gamma) * TWO3 * hM * bM
            dPStrain5 = next_d_gamma * voigt_to_covariant(R)
            depsilon_pv = compute_stress_invariant_i1(dPStrain5)
            dMM5_plus = 1 / ROOT23 * voigt_tensor_dot(dAlphaM5, n)
            dMM5_minus = -Z * macauley(-depsilon_pv)

            # Calc Delta 6
            this_sigma = (
                next_stress + (28 * dSigma1 - 125 * dSigma2 + 546 * dSigma3 + 54 * dSigma4 - 378 * dSigma5) / 625
            )
            this_alpha = (
                next_alpha + (28 * dAlpha1 - 125 * dAlpha2 + 546 * dAlpha3 + 54 * dAlpha4 - 378 * dAlpha5) / 625
            )
            this_alpha_m = (
                next_alpha_m + (28 * dAlphaM1 - 125 * dAlphaM2 + 546 * dAlphaM3 + 54 * dAlphaM4 - 378 * dAlphaM5) / 625
            )
            this_mm_plus = (
                next_mm_plus
                + (28 * dMM1_plus - 125 * dMM2_plus + 546 * dMM3_plus + 54 * dMM4_plus - 378 * dMM5_plus) / 625
            )
            this_mm_minus = (
                next_mm_minus
                + (28 * dMM1_minus - 125 * dMM2_minus + 546 * dMM3_minus + 54 * dMM4_minus - 378 * dMM5_minus) / 625
            )
            # thisVoidRatio = m_e_init - (1.0 + m_e_init) * ComputeStressInvariantI1(NextStrain + (28 * dPStrain1 - 125 * dPStrain2 + 546 * dPStrain3 + 54 * dPStrain4 - 378 * dPStrain5) / 625)
            # thisVoidRatio = clamp(self.emin, self.emax, thisVoidRatio)

            p = compute_stress_invariant_i1(this_sigma) / 3.0
            if p < M_PMIN:
                this_sigma = deviatoric_tensor(this_sigma) + M_PMIN * M_I1
                p = compute_stress_invariant_i1(this_sigma) / 3.0

            r = deviatoric_tensor(this_sigma) / p

            G, K = self.get_elastic_moduli(this_sigma, this_void_ratio, 1.0)
            n, d, b, bM, cos3_theta, h, hM, psi, r_btheta, r_dtheta, b0, A, D, B, C, R, Z = self.get_state_dependent(
                this_sigma, this_alpha, this_alpha_m, this_mm_plus, this_mm_minus, curalpha_in, this_void_ratio
            )

            Kp = TWO3 * p * h * voigt_tensor_dot(b, n)
            temp4 = (
                Kp
                + 2.0
                * G
                * (
                    B
                    - C * compute_stress_invariant_i1(symmetric_tensor_single_dot(n, symmetric_tensor_single_dot(n, n)))
                )
                - K * D * voigt_tensor_dot(n, r)
            )
            if abs(temp4) < SMALL:
                temp4 = SMALL
            next_d_gamma = (2.0 * G * voigt_tensor_dot(n, dDevStrain) - K * dVolStrain * voigt_tensor_dot(n, r)) / temp4
            dSigma6 = (
                2.0 * G * voigt_to_contravariant(dDevStrain)
                + K * dVolStrain * M_I1
                - macauley(next_d_gamma)
                * (2.0 * G * (B * n - C * (symmetric_tensor_single_dot(n, n) - 1.0 / 3.0 * M_I1)) + K * D * M_I1)
            )
            dAlpha6 = macauley(next_d_gamma) * TWO3 * h * b
            dAlphaM6 = macauley(next_d_gamma) * TWO3 * hM * bM
            dPStrain6 = next_d_gamma * voigt_to_covariant(R)
            depsilon_pv = compute_stress_invariant_i1(dPStrain6)

            # Update
            dSigma = (dSigma1 + 4 * dSigma3 + dSigma4) / 6
            dAlpha = (dAlpha1 + 4 * dAlpha3 + dAlpha4) / 6
            dAlphaM = (dAlphaM1 + 4 * dAlphaM3 + dAlphaM4) / 6

            dMM_plus = (dMM1_plus + 4 * dMM3_plus + dMM4_plus) / 6
            dMM_minus = (dMM1_minus + 4 * dMM3_minus + dMM4_minus) / 6

            dPStrain = (dPStrain1 + 4 * dPStrain3 + dPStrain4) / 6

            n_stress = next_stress + dSigma
            n_alpha = next_alpha + dAlpha
            n_alpha_m = next_alpha_m + dAlphaM
            n_mm_plus = next_mm_plus + dMM_plus
            n_mm_minus = next_mm_minus + dMM_minus

            n_trial = self.get_normal_to_yield(n_stress, n_alpha)

            if (n_mm_plus + n_mm_minus) < self.m:
                n_mm_minus = self.m - n_mm_plus

            MM_trial = n_mm_plus + n_mm_minus
            r_alpha_m_trial = n_alpha_m + ROOT23 * (MM_trial - self.m) * n_trial

            b_m_distance_trial = voigt_tensor_dot(r_alpha_m_trial - n_alpha, n_trial)

            if b_m_distance_trial < -1.0e-4:
                n_alpha_m = n_alpha - ROOT23 * (MM_trial - self.m) * n_trial

            r_m_alphatilde_trial = n_alpha_m - ROOT23 * (MM_trial - self.m) * n_trial
            b_m_distance_tilde_trial = voigt_tensor_dot(n_alpha - r_m_alphatilde_trial, n_trial)

            if b_m_distance_tilde_trial < -1.0e-4:
                n_mm_plus += abs(b_m_distance_tilde_trial) / ROOT23

            # Compute error
            p = compute_stress_invariant_i1(n_stress) / 3.0
            if p < M_PMIN:
                if dT == dT_min:
                    done = True
                dT = max(0.1 * dT, dT_min)
                continue

            if done == False:
                stress_norm = voigt_tensor_norm(next_stress)
                alpha_norm = voigt_tensor_norm(next_alpha)

                cur_step_error1 = (
                    voigt_tensor_norm(-42 * dSigma1 - 224 * dSigma3 - 21 * dSigma4 + 162 * dSigma5 + 125 * dSigma6)
                    / 336
                )
                if stress_norm >= 0.5:
                    cur_step_error1 /= 2.0 * stress_norm

                cur_step_error2 = (
                    voigt_tensor_norm(-42 * dAlpha1 - 224 * dAlpha3 - 21 * dAlpha4 + 162 * dAlpha5 + 125 * dAlpha6)
                    / 336
                )
                if alpha_norm >= 0.5:
                    cur_step_error2 /= 2.0 * alpha_norm

                cur_step_error = max(cur_step_error1, cur_step_error2)

                if cur_step_error > tol_e:
                    q = max(0.8 * pow(tol_e / cur_step_error, 0.2), 0.1)

                    if dT == dT_min:
                        next_stress = n_stress
                        next_alpha = n_alpha
                        next_alpha_m = next_alpha
                        next_mm_plus = cur_mm_plus
                        next_mm_minus = self.m - cur_mm_plus

                        T += dT
                    dT = max(q * dT, dT_min)
                else:
                    next_stress = n_stress
                    next_alpha = n_alpha
                    next_alpha_m = n_alpha_m
                    next_mm_plus = n_mm_plus
                    next_mm_minus = n_mm_minus

                    q = min(0.8 * pow(tol_e / cur_step_error, 0.2), 2.0)
                    T += dT
                    dT = max(q * dT, dT_min)
                    dT = min(dT, 1 - T)

        return next_stress, next_alpha, next_alpha_m, next_mm_plus, next_mm_minus, next_void_ratio

    @ti.func
    def intersection_factor(self, cur_stress, dStrain, cur_void_ratio, cur_alpha, a0, a1):
        a = a0

        m_e_init = self.e_init

        strain_inc = dStrain

        vR = cur_void_ratio - (1 + cur_void_ratio) * compute_stress_invariant_i1(a0 * strain_inc)
        vR = clamp(self.emin, self.emax, vR)
        G, K = self.get_elastic_moduli(cur_stress, vR, 1)
        dSigma0 = a0 * (compute_elastic_stiffness_tensor(K, G) @ strain_inc)
        f0 = self.get_f(cur_stress + dSigma0, cur_alpha)

        vR = cur_void_ratio - (1 + cur_void_ratio) * compute_stress_invariant_i1(a1 * strain_inc)
        vR = clamp(self.emin, self.emax, vR)
        G, K = self.get_elastic_moduli(cur_stress, vR, 1)
        dSigma1 = a1 * (compute_elastic_stiffness_tensor(K, G) @ strain_inc)
        f1 = self.get_f(cur_stress + dSigma1, cur_alpha)

        for i in range(10):
            a = a1 - f1 * (a1 - a0) / (f1 - f0)
            dSigma = a * (compute_elastic_stiffness_tensor(K, G) @ strain_inc)
            f = self.get_f(cur_stress + dSigma, cur_alpha)
            if abs(f) < M_TOL_F:
                break

            if f * f0 < 0:
                a1 = a
                f1 = f
            else:
                f1 = f1 * f0 / (f0 + f)
                a0 = a
                f0 = f

            if i == 9:
                a = 0.0
                break

        if a > (1 - SMALL):
            a = 1.0
        if a < SMALL:
            a = 0.0
        return a

    @ti.func
    def intersection_factor_unloading(self, cur_stress, dStrain, curvoid_ratio, cur_alpha):
        a, a0, a1 = 0.0, 0.0, 1.0
        n_sub = 20

        m_e_init = self.e_init

        strain_inc = dStrain

        vR = curvoid_ratio
        vR = clamp(self.emin, self.emax, vR)
        G, K = self.get_elastic_moduli(cur_stress, vR, 1)
        dSigma = compute_elastic_stiffness_tensor(K, G) @ strain_inc

        done = False
        a_result = 0.0
        for i in range(n_sub):
            if not done:
                da = (a1 - a0) / 2.0
                a = a1 - da
                f = self.get_f(cur_stress + a * dSigma, cur_alpha)
                if f > M_TOL_F:
                    a1 = a

                elif f < -M_TOL_F:
                    a0 = a
                    done = True
                    a_result = self.intersection_factor(cur_stress, strain_inc, vR, cur_alpha, a0, a1)
                else:
                    done = True
                    a_result = a
                if i == (n_sub - 1) and not done:
                    done = True
                    a_result = 0.0

        return a_result

    @ti.func
    def get_normal_to_yield(self, stress, alpha):
        dev_stress = deviatoric_tensor(stress)
        p = compute_stress_invariant_i1(stress) * (1.0 / 3.0)

        n = ZEROVEC6F
        if abs(p) < M_PMIN:
            n = ZEROVEC6F
        else:
            n = dev_stress - p * alpha
            norm_n = voigt_tensor_norm(n)
            if norm_n < SMALL:
                norm_n = SMALL
            n = n / norm_n
        return n

    @ti.func
    def stress_correction300(
        self, pre_stress, pre_alpha, cur_stress, cur_alpha, curalpha_m, cur_mm_plus, cur_mm_minus, curalpha_in, e
    ):
        s_newstress = ZEROVEC6F
        s_new_alpha = ZEROVEC6F

        sqrt_two3 = ti.sqrt(TWO3)
        small_adim = 1.0e-12

        switch = 0

        PAR_m = self.m

        SI1 = EYE
        small = 1.0e-7
        i = 0

        s_interstress = cur_stress
        s_inter_alpha = cur_alpha

        f2 = self.get_f(pre_stress, pre_alpha)

        f0 = 0.0
        f1 = 0.0
        temp2 = ZEROVEC6F
        res = ZEROVEC6F
        done = False
        while i <= 50 and not done:
            if switch == 0:
                p = compute_stress_invariant_i1(s_interstress) / 3.0

                rr = deviatoric_tensor(s_interstress) / p

                # TRACE_RR = rr[0] + rr[1] + rr[2]
                # TRACE_ALFA = sInterAlpha[0] + sInterAlpha[1] + sInterAlpha[2]

                f0 = self.get_f(s_interstress, s_inter_alpha)

                sn, dd, bb, bM, cos3_theta, h, hM, psi, r_btheta, r_dtheta, b0, A, D, B, C, R, Z = (
                    self.get_state_dependent(
                        s_interstress, s_inter_alpha, curalpha_m, cur_mm_plus, cur_mm_minus, curalpha_in, e
                    )
                )

                PM = voigt_tensor_dot(bb, sn)
                PM = TWO3 * p * h * PM

                E_G, E_K = self.get_elastic_moduli(s_interstress, e, 1)
                aC = compute_elastic_stiffness_tensor(E_K, E_G)

                temp1 = symmetric_tensor_single_dot(sn, sn)

                res = (B * sn) - (C * (temp1 - 1.0 / 3.0 * SI1)) + (1.0 / 3.0 * D * SI1)
                R = voigt_to_covariant(res)

                temp1 = aC @ R

                res2 = voigt_tensor_dot(sn, rr)
                res = sn - 1.0 / 3.0 * res2 * SI1
                temp2 = voigt_to_covariant(res)
                temp2 = aC @ temp2

                temp3 = voigt_tensor_dot(temp2, R)
                temp3 += PM

                dlambda = f0 / temp3

                temp4 = aC @ R
                s_newstress = s_interstress - dlambda * temp4
                s_new_alpha = s_inter_alpha + dlambda * TWO3 * h * bb

                f1 = self.get_f(s_newstress, s_new_alpha)

                if abs(f1) < M_TOL_F:
                    done = True
                if abs(f1) > abs(f2):
                    switch = 1
                else:
                    s_interstress = s_newstress
                    s_inter_alpha = s_new_alpha
            else:
                temp5 = voigt_tensor_dot(temp2, res)
                dlambda = f0 / temp5
                s_newstress = s_interstress - dlambda * res
                s_new_alpha = s_inter_alpha
                f1 = self.get_f(s_newstress, s_new_alpha)

                s_interstress = s_newstress
                s_inter_alpha = s_new_alpha

                f0 = self.get_f(s_newstress, s_new_alpha)

                if abs(f0) < M_TOL_F:
                    done = True

            if i == 50 and not done:
                if f1 > 0.0:
                    s_newstress_bisection = self.bisection_of_ys_crossing(s_newstress, s_new_alpha)
                    f0 = self.get_f(s_newstress_bisection, s_new_alpha)
                    if f0 < M_TOL_F:
                        s_newstress = s_newstress_bisection
                        done = True

                if not done:
                    p = compute_stress_invariant_i1(s_newstress)
                    p_s_new_alpha = compute_stress_invariant_i1(s_new_alpha)
                    s_new_alpha = restore_dviatoric_property(s_new_alpha, p_s_new_alpha)

                    sn = self.get_normal_to_yield(s_newstress, s_new_alpha)

                    p = 1.0 / 3.0 * p
                    rr = s_new_alpha + sqrt_two3 * sn * PAR_m
                    ss = rr * p
                    rr_alfa = rr - s_new_alpha

                    s_newstress = rr * p
                    s_newstress[0] += p
                    s_newstress[1] += p
                    s_newstress[2] += p

                    f1 = self.get_f(s_newstress, s_new_alpha)

                    if f1 > M_TOL_F:
                        done = True
            i += 1
        return s_newstress, s_new_alpha

    @ti.func
    def bisection_of_ys_crossing(self, cur_stress, cur_alpha):
        s_newstress = ZEROVEC6F

        beta0 = 0.0
        beta1 = 1.0

        p0 = compute_stress_invariant_i1(cur_stress) / 3.0
        s0 = deviatoric_tensor(cur_stress)
        s1 = cur_alpha * p0

        pa = p0
        pC = p0

        done = False
        for it in range(50):
            if not done:
                beta = (beta0 + beta1) / 2.0
                sa = (s1 - s0) * beta0 + s0
                stress_a = sa
                stress_a[0] += p0
                stress_a[1] += p0
                stress_a[2] += p0

                sC = (s1 - s0) * beta + s0
                stress_c = sC
                stress_c[0] += p0
                stress_c[1] += p0
                stress_c[2] += p0

                Fa = self.get_f(stress_a, cur_alpha)
                Fc = self.get_f(stress_c, cur_alpha)

                if abs(Fc) <= M_TOL_F:
                    s_newstress = stress_c
                    done = True
                else:
                    if (Fc > 0.0 and Fa > 0.0) or (Fc < 0.0 and Fa < 0.0):
                        beta0 = beta
                    else:
                        beta1 = beta

        return s_newstress

    @ti.func
    def proj_bounding(self, T, phi):
        sig0 = -0.01
        pi = 3.141592653589793
        tcorrected = T

        tr_sig = T[0] + T[1] + T[2]
        if tr_sig / 3.0 > sig0:
            tcorrected[0] += sig0 - tr_sig / 3.0
            tcorrected[1] += sig0 - tr_sig / 3.0
            tcorrected[2] += sig0 - tr_sig / 3.0

            tr_sig = 3.0 * sig0

        w = ti.sin(1.4 * phi) ** 2
        rc = (1.0 - w) / (9.0 - w)

        s11 = tcorrected[0] / tr_sig - 1.0 / 3.0
        s22 = tcorrected[1] / tr_sig - 1.0 / 3.0
        s33 = tcorrected[2] / tr_sig - 1.0 / 3.0
        s12 = tcorrected[3] / tr_sig
        s23 = tcorrected[4] / tr_sig
        s13 = tcorrected[5] / tr_sig

        norm = ti.sqrt(s11**2 + s22**2 + s33**2 + 2.0 * (s12**2 + s13**2 + s23**2))

        do_projection = norm / abs(tr_sig) >= 1.0e-3
        root = 0.0

        if do_projection:
            s11 /= norm
            s22 /= norm
            s33 /= norm
            s12 /= norm
            s13 /= norm
            s23 /= norm
            det = s11 * s22 * s33 + 2.0 * s12 * s23 * s13 - s13 * s13 * s22 - s23 * s23 * s11 - s12 * s12 * s33
            r = 0.5 * rc - 1.0 / 6.0
            dummy = 1.0 / 27.0 - rc / 3.0
            if abs(det) < 1.0e-6 * abs(r):
                dummy2 = -dummy / r
                root = 0.0
                if dummy2 > 0.0:
                    root = ti.sqrt(dummy2)
            else:
                r /= det
                dummy /= det
                p = -r * r / 3.0
                q = 2.0 * r * r * r / 27.0 + dummy
                rr = ti.sqrt(abs(p) / 3.0)
                if q < 0.0:
                    rr = -rr
                w = ti.acos(q / (2.0 * rr**3))
                w1 = -2.0 * rr * ti.cos(w / 3.0) - r / 3.0
                w2 = -2.0 * rr * ti.cos(w / 3.0 + 2.0 * pi / 3.0) - r / 3.0
                w3 = -2.0 * rr * ti.cos(w / 3.0 + 4.0 * pi / 3.0) - r / 3.0

                w_max = 0.0
                if w1 > 0.0 and 1.0 / w1 > w_max:
                    w_max = 1.0 / w1
                if w2 > 0.0 and 1.0 / w2 > w_max:
                    w_max = 1.0 / w2
                if w3 > 0.0 and 1.0 / w3 > w_max:
                    w_max = 1.0 / w3
                root = 1.0 / w_max

        if do_projection and root < norm:
            tcorrected[0] = (s11 * root + 1.0 / 3.0) * tr_sig
            tcorrected[1] = (s22 * root + 1.0 / 3.0) * tr_sig
            tcorrected[2] = (s33 * root + 1.0 / 3.0) * tr_sig
            tcorrected[3] = s12 * root * tr_sig
            tcorrected[4] = s23 * root * tr_sig
            tcorrected[5] = s13 * root * tr_sig

        return tcorrected
