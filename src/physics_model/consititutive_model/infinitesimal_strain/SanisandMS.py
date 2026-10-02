import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.infinitesimal_strain.InfinitesimalStrainModel import InfinitesimalStrainModel
from src.utils.constants import DELTA, EYE, DELTA2D, ZEROVEC6f
from src.utils.MatrixFunction import Eig_3
from src.utils.ObjectIO import DictIO
from src.utils.ScalarFunction import macauley, macauley_index
from src.utils.VectorFunction import voigt_tensor_dot

small = 1e-10
m_P_atm = 101.3
m_Pmin = 1e-4 * m_P_atm
mTolF = 1.0e-4
mTolR = 1.0e-4
mI1 = EYE  # 待确认
root23 = ti.sqrt(2.0 / 3.0)
one3 = 1.0 / 3.0
two3 = 2.0 / 3.0


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
        density = DictIO.GetAlternative(material, "Density", 2650.0)
        G0 = DictIO.GetEssential(material, "G0")
        nu = DictIO.GetAlternative(material, "PoissonRatio", DictIO.GetAlternative(material, "nu", 0.05))
        Mc = DictIO.GetEssential(material, "Mc", "CriticalStateRatio")
        c = DictIO.GetEssential(material, "c", "LodeRatio")
        lambda_c = DictIO.GetEssential(material, "lambda_c", "LambdaC")
        e0 = DictIO.GetEssential(material, "e0", "ReferenceVoidRatio")
        ksi = DictIO.GetEssential(material, "ksi", "xi")
        m = DictIO.GetEssential(material, "m", "YieldSurfaceSize")
        h0 = DictIO.GetEssential(material, "h0")
        ch = DictIO.GetEssential(material, "ch")
        nb = DictIO.GetEssential(material, "nb")
        A0 = DictIO.GetEssential(material, "A0")
        nd = DictIO.GetEssential(material, "nd")
        zeta = DictIO.GetEssential(material, "zeta")
        mu0 = DictIO.GetEssential(material, "mu0")
        beta = DictIO.GetEssential(material, "beta")
        e_init = DictIO.GetEssential(material, "e_init", "InitialVoidRatio")
        emax = DictIO.GetEssential(material, "emax", "MaximumVoidRatio")
        emin = DictIO.GetEssential(material, "emin", "MinimumVoidRatio")
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
        reference_shear_kpa = self.G0 * m_P_atm * (2.97 - self.e_init) ** 2 / (1.0 + self.e_init)
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
        raise RuntimeError("SanisandMS does not support random-field material parameters yet")

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
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        stress = -particle[np].stress * 1.0e-3
        stateVars[np].estress = VonMisesStress(stress)
        p = ComputeStressInvariantI1(stress) / 3.0
        rr = ZEROVEC6f
        if p > m_Pmin:
            rr = DeviatoricTensor(stress) / p

        stateVars[np].mAlpha = rr
        stateVars[np].mAlphaM = rr
        stateVars[np].malpha_in = rr
        stateVars[np].mMM_plus = self.m
        stateVars[np].mMM_minus = 0.0
        stateVars[np].mVoidRatio = clamp(self.emin, self.emax, self.e_init)
        stateVars[np].epstrain = 0.0

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        return (DELTA + velocity_gradient * dt[None]).determinant()

    @ti.func
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        return (DELTA2D + velocity_gradient * dt[None]).determinant()

    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        return 1.0 + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])

    @ti.func
    def UpdateInternalVariables(self, np, dpdstrain, stateVars):
        stateVars[np].epstrain += dpdstrain

    @ti.func
    def UpdateStateVariables(self, np, stress, stateVars):
        stateVars[np].estress = VonMisesStress(stress)

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        compression_stress_kpa = -current_stress * 1.0e-3
        shear, bulk = self.GetElasticModuli(compression_stress_kpa, stateVars[np].mVoidRatio, 1)
        return ComputeElasticStiffnessTensor(bulk * 1.0e3, shear * 1.0e3)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stateVars):
        stiffness_matrix = self.compute_elastic_tensor(np, current_stress, stateVars)
        compression_stress_kpa = -current_stress * 1.0e-3
        state_vars = stateVars[np]
        yield_function = self.GetF(compression_stress_kpa, state_vars.mAlpha)
        if yield_function > -mTolF:
            shear, bulk = self.GetElasticModuli(compression_stress_kpa, state_vars.mVoidRatio, 1)
            n, d, b, bM, cos3theta, h, hM, psi, rBtheta, rDtheta, b0, A, D, B, C, R, Z = self.GetStateDependent(
                compression_stress_kpa,
                state_vars.mAlpha,
                state_vars.mAlphaM,
                state_vars.mMM_plus,
                state_vars.mMM_minus,
                state_vars.malpha_in,
                state_vars.mVoidRatio,
            )
            stiffness_matrix = (
                self.GetElastoPlasticTangent(
                    compression_stress_kpa,
                    1.0,
                    ZEROVEC6f,
                    ZEROVEC6f,
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
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):
        ############################## STEP1 ##############################
        strain_rate = calculate_strain_rate2D(velocity_gradient)
        de = calculate_strain_increment2D(velocity_gradient, dt)
        dw = calculate_vorticity_increment2D(velocity_gradient, dt)
        previous_stress = self.Explicit_Integration(np, previous_stress, de, strain_rate, dw, stateVars)
        return previous_stress

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):
        strain_rate = calculate_strain_rate(velocity_gradient)
        de = calculate_strain_increment(velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)
        previous_stress = self.Explicit_Integration(np, previous_stress, de, strain_rate, dw, stateVars)
        return previous_stress

    @ti.func
    def Explicit_Integration(self, np, previous_stress, strain_inc, strain_rate, dw, stateVars):
        mSigma_n = -previous_stress * 1.0e-3
        # mSigma_n += stateVars[np].stress_visc   # 阻尼项
        # mEpsilon_n = - curstrain
        # mEpsilon = - nextstrain
        dstrain = -strain_inc

        mAlpha_n = stateVars[np].mAlpha
        mAlphaM_n = stateVars[np].mAlphaM
        malpha_in_n = stateVars[np].malpha_in
        mMM_plus_n = stateVars[np].mMM_plus
        mMM_minus_n = stateVars[np].mMM_minus

        # 按照Fortran版本的理解来的
        CurVoidRatio = stateVars[np].mVoidRatio
        mG, mK = self.GetElasticModuli(mSigma_n, CurVoidRatio, 1)
        mCe = ComputeElasticStiffnessTensor(mK, mG)

        trialDirection = mCe @ dstrain
        # trialDirection = ElasticTensorMultiplyVector(mEpsilon - mEpsilon_n, mK, mG)
        malpha_in = ZEROVEC6f
        if voigt_tensor_dot(mAlpha_n - malpha_in_n, trialDirection) < 0.0:
            malpha_in = mAlpha_n
        else:
            malpha_in = malpha_in_n

        mSigma = mSigma_n
        mAlpha = mAlpha_n
        mAlphaM = mAlphaM_n
        mMM_plus = mMM_plus_n
        mMM_minus = mMM_minus_n
        if VoigtTensorNorm(dstrain) > 1.0e-16:
            mSigma, mAlpha, mAlphaM, mMM_plus, mMM_minus = self.explicit_integrator(
                mSigma_n,
                dstrain,
                CurVoidRatio,
                mAlpha_n,
                mAlphaM_n,
                mMM_plus_n,
                mMM_minus_n,
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
        # ) = Eig_3(mSigma)
        # if min(sNextStress1a, sNextStress2a, sNextStress3a) >= 0.0:
        (
            xN1,
            xN2,
            xN3,
            sNextStress1a,
            sNextStress2a,
            sNextStress3a,
            PNextStress,
            QNextStress,
        ) = Eig_3(mSigma)
        if min(sNextStress1a, sNextStress2a, sNextStress3a) < 0.0:
            mSigma = m_Pmin * mI1
            # p = ComputeStressInvariantI1(mSigma) / 3.
            # rr = DeviatoricTensor(mSigma) / p
            # q_Norm_New = VoigtTensorNorm(rr)
            # q_Norm_New = ti.sqrt(3./ 2.) * q_Norm_New
            mAlpha = ZEROVEC6f
            mAlphaM = ZEROVEC6f
            malpha_in = ZEROVEC6f
            mMM_plus = self.m
            mMM_minus = 0.0

        # p = ComputeStressInvariantI1(mSigma) / 3.0
        # if p >= 0.0:
        mVoidRatio = CurVoidRatio - (1 + CurVoidRatio) * ComputeStressInvariantI1(dstrain)
        mVoidRatio = clamp(self.emin, self.emax, mVoidRatio)

        stateVars[np].malpha_in = malpha_in
        stateVars[np].mAlpha = mAlpha
        stateVars[np].mAlphaM = mAlphaM
        stateVars[np].mMM_plus = mMM_plus
        stateVars[np].mMM_minus = mMM_minus
        stateVars[np].mVoidRatio = mVoidRatio

        # sigrot = Sigrot(mSigma_n, -dw)
        # mSigma += sigrot

        # p = ComputeStressInvariantI1(mSigma) / 3.
        # if p < m_Pmin:
        #     mSigma = DeviatoricTensor(mSigma) + m_Pmin * mI1

        updated_stress = -mSigma * 1.0e3
        stateVars[np].estress = VonMisesStress(updated_stress)
        return updated_stress

    @ti.func
    def explicit_integrator(
        self, CurStress, StrainInc, CurVoidRatio, CurAlpha, CurAlphaM, CurMM_plus, CurMM_minus, Curalpha_in, G, K
    ):
        p_tr_pos = True

        m_e_init = self.e_init

        # NextVoidRatio = m_e_init - (1 + m_e_init) * ComputeStressInvariantI1(NextStrain)
        dStrain = StrainInc
        aC = ComputeElasticStiffnessTensor(K, G)
        dSigma = aC @ dStrain
        # dSigma = ElasticTensorMultiplyVector(dStrain, K, G)
        NextStress = CurStress + dSigma
        f = self.GetF(NextStress, CurAlpha)
        p = ComputeStressInvariantI1(NextStress) / 3.0

        # 局部变量初始化
        NextAlpha = ZEROVEC6f
        NextAlphaM = ZEROVEC6f
        NextMM_plus = 0.0
        NextMM_minus = 0.0

        # if VoigtTensorNorm(dStrain) >= small:
        #     return

        if p < m_Pmin:
            p_tr_pos = False

        if p_tr_pos and f <= mTolF:  # This is a pure elastic loading/unloading
            NextAlpha = CurAlpha
            NextAlphaM = CurAlphaM
            NextMM_plus = CurMM_plus
            NextMM_minus = CurMM_minus
        else:
            fn = self.GetF(CurStress, CurAlpha)
            pn = ComputeStressInvariantI1(CurStress) / 3.0
            if fn > mTolF:  # This is an illegal stress state! This shouldn't happen
                (
                    NextStress,
                    NextAlpha,
                    NextAlphaM,
                    NextMM_plus,
                    NextMM_minus,
                    NextVoidRatio,
                ) = self.RungeKutta4(
                    CurStress,
                    dStrain,
                    CurVoidRatio,
                    CurAlpha,
                    CurAlphaM,
                    CurMM_plus,
                    CurMM_minus,
                    Curalpha_in,
                )

            elif fn < -mTolF:  # This is a transition from elastic to plastic
                elasticRatio = self.IntersectionFactor(CurStress, dStrain, CurVoidRatio, CurAlpha, 0.0, 1.0)
                dSigma = aC @ (elasticRatio * dStrain)
                # dSigma = ElasticTensorMultiplyVector(elasticRatio * (NextStrain - CurStrain), K, G)
                (
                    NextStress,
                    NextAlpha,
                    NextAlphaM,
                    NextMM_plus,
                    NextMM_minus,
                    NextVoidRatio,
                ) = self.RungeKutta4(
                    CurStress + dSigma,
                    (1.0 - elasticRatio) * dStrain,
                    CurVoidRatio,
                    CurAlpha,
                    CurAlphaM,
                    CurMM_plus,
                    CurMM_minus,
                    Curalpha_in,
                )

            elif abs(fn) < mTolF:
                denom = VoigtTensorNorm(dSigma)
                if denom == 0:
                    denom = 1.0
                ratio = voigt_tensor_dot(self.GetNormalToYield(CurStress, CurAlpha), dSigma) / denom
                if ratio > -ti.sqrt(mTolF):
                    # This is a pure plastic step
                    (
                        NextStress,
                        NextAlpha,
                        NextAlphaM,
                        NextMM_plus,
                        NextMM_minus,
                        NextVoidRatio,
                    ) = self.RungeKutta4(
                        CurStress,
                        dStrain,
                        CurVoidRatio,
                        CurAlpha,
                        CurAlphaM,
                        CurMM_plus,
                        CurMM_minus,
                        Curalpha_in,
                    )
                else:
                    # This is an elastic unloding followed by plastic loading
                    elasticRatio = self.IntersectionFactor_Unloading(CurStress, dStrain, CurVoidRatio, CurAlpha)
                    dSigma = aC @ (elasticRatio * dStrain)
                    # dSigma = ElasticTensorMultiplyVector(elasticRatio * (NextStrain - CurStrain), K, G)
                    (
                        NextStress,
                        NextAlpha,
                        NextAlphaM,
                        NextMM_plus,
                        NextMM_minus,
                        NextVoidRatio,
                    ) = self.RungeKutta4(
                        CurStress + dSigma,
                        (1.0 - elasticRatio) * dStrain,
                        CurVoidRatio,
                        CurAlpha,
                        CurAlphaM,
                        CurMM_plus,
                        CurMM_minus,
                        Curalpha_in,
                    )
        return NextStress, NextAlpha, NextAlphaM, NextMM_plus, NextMM_minus

    @ti.func
    def GetElasticModuli(self, sigma, en, ElastFlag):
        pn = ComputeStressInvariantI1(sigma) / 3.0
        if pn <= m_Pmin:
            pn = m_Pmin

        # G, K = 0., 0.
        # if ElastFlag == 0:
        #     G = self.G0 * m_P_atm * pow((2.97 - en), 2) / (1 + en)
        # else:
        G = self.G0 * m_P_atm * pow((2.97 - en), 2) / (1 + en) * ti.sqrt(pn / m_P_atm)
        K = (1 + self.nu) / (1 - 2 * self.nu) * G * 2.0 / 3.0
        return G, K

    @ti.func
    def GetF(self, stress, Alpha):
        s = DeviatoricTensor(stress)
        p = ComputeStressInvariantI1(stress) / 3.0
        s = s - p * Alpha
        return VoigtTensorNorm(s) - root23 * self.m * p

    @ti.func
    def GetPSI(self, e, p):
        return e - (self.e0 - self.lambda_c * pow((p / m_P_atm), self.ksi))

    @ti.func
    def GetLodeAngle(self, stress):
        Cos3Theta = ti.sqrt(6.0) * ComputeStressInvariantI1(
            SymmetricTensorSingleDot(stress, SymmetricTensorSingleDot(stress, stress))
        )
        Cos3Theta = clamp(-1.0, 1.0, Cos3Theta)
        return Cos3Theta

    @ti.func
    def g(self, cos3theta, c):
        return 2 * c / ((1 + c) - (1 - c) * cos3theta)

    @ti.func
    def GetStateDependent(self, stress, alpha, alphaM, MM_plus, MM_minus, alpha_in, e):
        zerozo = 1.0e-2
        zero_plus_tol = 1.0e-30
        tol = 1.0e-15

        D_factor = 1.0
        p = ComputeStressInvariantI1(stress) / 3.0
        stress_use = stress
        if p < m_Pmin:
            stress_use = DeviatoricTensor(stress) + m_Pmin * mI1
            p = ComputeStressInvariantI1(stress_use) / 3.0

        r = DeviatoricTensor(stress_use) / p

        n = self.GetNormalToYield(stress_use, alpha)

        AlphaAlphaInDotN = voigt_tensor_dot(alpha - alpha_in, n)

        psi = self.GetPSI(e, p)

        cos3Theta = self.GetLodeAngle(n)

        # numgeo对Mb的修改 Mb = nb * <-psi>
        rBtheta = self.g(cos3Theta, self.c) * self.Mc * ti.exp(-self.nb * psi) - self.m
        rBthetaPLUSpi = self.g(-cos3Theta, self.c) * self.Mc * ti.exp(-self.nb * psi) - self.m

        alphaBtheta = root23 * rBtheta * n

        alphaBthetaPLUSpi = -root23 * rBthetaPLUSpi * n

        rDtheta = self.g(cos3Theta, self.c) * self.Mc * ti.exp(self.nd * psi) - self.m

        alphaDtheta = root23 * rDtheta * n

        b0 = self.G0 * self.h0 * (1.0 - self.ch * e) / ti.sqrt(p / m_P_atm)

        d = alphaDtheta - alpha

        b = alphaBtheta - alpha

        # the memory surface and image points on it
        MM = max(self.m, MM_plus + MM_minus)
        # MM = max(self.m, MM_plus + MM_minus) # 来源于Fortran

        r_alphaM = alphaM + root23 * (MM - self.m) * n
        rM_tilde = alphaM - root23 * MM * n
        rM_Alphatilde = alphaM - root23 * (MM - self.m) * n
        rM = alphaM + root23 * MM * n

        bM_distance = voigt_tensor_dot(r_alphaM - alpha, n)
        bM_distance_tilde = voigt_tensor_dot(alpha - rM_Alphatilde, n)

        if bM_distance < 0.0:
            r_alphaM = alpha
            rM = r
            bM_distance = 0.0

        r_tilde = alpha - root23 * self.m * n
        x2 = 0.0
        f_shr = 0.0
        if bM_distance_tilde < 0.0:
            rM_tilde = r_tilde
            rM_Alphatilde = alpha
            bM_distance = 2 * root23 * (MM - self.m)

        # snM = rM - r
        # res100 = VoigtTensorNorm(snM)
        # if ti.abs(res100) < 0.01:
        #     snM = n
        # else:
        #     snM = snM / res100

        x3 = voigt_tensor_dot(n, rM - rM_tilde)

        x2 = voigt_tensor_dot(n, r_tilde - rM_tilde)

        if x2 < 0.0:
            f_shr = 0.0
        else:
            f_shr = x2 / x3

        f_shr = clamp(0.0, 1.0, f_shr)

        # Change
        gthetaPlusPi = 2 * self.c / ((1 + self.c) + (1 - self.c) * cos3Theta)
        alphad_tilde = root23 * (gthetaPlusPi * self.Mc * ti.exp(self.nd * psi) - self.m) * ((-1) * n)

        b_dM_tilde = voigt_tensor_dot(alphad_tilde - alpha_in, n)
        bref = max(2 * root23 * self.m, voigt_tensor_dot(alphaBtheta - alphaBthetaPLUSpi, n))
        bref_D = max(2 * root23 * self.m, VoigtTensorNorm(alpha_in))

        b_d_r = voigt_tensor_dot(d, n)

        A = self.A0 * ti.exp(self.beta * macauley(b_dM_tilde) / bref_D)

        D = A * b_d_r

        if p < 0.001 * m_P_atm:
            D_factor = 1.0 / (1.0 + (ti.exp(7.6349 - 7.2713 * p)))
        else:
            D_factor = 1.0

        D *= D_factor

        B = 1.0 + 1.5 * (1 - self.c) / self.c * self.g(cos3Theta, self.c) * cos3Theta

        C = 3.0 * ti.sqrt(1.5) * (1 - self.c) / self.c * self.g(cos3Theta, self.c)

        R = B * n - C * (SymmetricTensorSingleDot(n, n) - one3 * mI1) + one3 * D * mI1

        Z = MM / self.zeta * f_shr

        bM = alphaBtheta - r_alphaM
        b_bM = voigt_tensor_dot(bM, n)

        b_rM_rin = voigt_tensor_dot(r_alphaM - alpha_in, n)
        # if b_rM_rin < 1.0e-7:
        #     b_rM_rin = 1.0e-7
        b_rM_rin = abs(b_rM_rin) + 0.001
        hM = min(1.0e10, 0.5 * b0 / b_rM_rin + 0.5 / root23 * Z * macauley(-D) / b_bM)

        # if AlphaAlphaInDotN < small:
        #     AlphaAlphaInDotN = small

        AlphaAlphaInDotN = abs(AlphaAlphaInDotN) + 0.001
        mem = self.mu0 * pow(p / m_P_atm, 0.5) * pow(bM_distance / bref, 2)
        h = 0.0
        if mem > 9.2103:
            h = b0 / AlphaAlphaInDotN * ti.exp(9.2103)
        else:
            h = b0 / AlphaAlphaInDotN * ti.exp(mem)
        h = min(1.0e7, h)

        return n, d, b, bM, cos3Theta, h, hM, psi, rBtheta, rDtheta, b0, A, D, B, C, R, Z

    @ti.func
    def GetElastoPlasticTangent(self, NextStress, NextDGamma, CurStrain, NextStrain, G, K, B, C, D, h, n, d, b):
        p = ComputeStressInvariantI1(NextStress) / 3.0
        if p < small:
            p = small
        r = DeviatoricTensor(NextStress) / p
        Kp = two3 * p * h * voigt_tensor_dot(b, n)

        elastic_stiffness = ComputeElasticStiffnessTensor(K, G)
        flow_direction = VoigtToCovariant(
            (B * n) - (C * (SymmetricTensorSingleDot(n, n) - one3 * mI1)) + (one3 * D * mI1)
        )

        temp1 = elastic_stiffness @ VoigtToCovariant(flow_direction)
        temp2 = elastic_stiffness.transpose() @ VoigtToCovariant(n - one3 * voigt_tensor_dot(n, r) * mI1)
        denominator = voigt_tensor_dot(temp2, flow_direction) + Kp

        tangent = elastic_stiffness
        if ti.abs(denominator) >= small:
            tangent = elastic_stiffness - (macauley_index(NextDGamma) / denominator) * temp1.outer_product(temp2)
        return tangent

    @ti.func
    def RungeKutta4(self, CurStress, dStrain, CurVoidRatio, CurAlpha, CurAlphaM, CurMM_plus, CurMM_minus, Curalpha_in):
        T = 0.0
        dT = 1.0
        dT_min = 1.0e-3
        TolE = mTolR

        m_e_init = self.e_init

        # CurVoidRatio = m_e_init - (1 + m_e_init) * ComputeStressInvariantI1(CurStrain)
        NextVoidRatio = CurVoidRatio - (1 + CurVoidRatio) * ComputeStressInvariantI1(dStrain)
        NextVoidRatio = clamp(self.emin, self.emax, NextVoidRatio)

        # G, K = self.GetElasticModuli(CurStress, CurVoidRatio, 1)

        # aC = ComputeElasticStiffnessTensor(K, G)

        NextStress = CurStress
        NextAlpha = CurAlpha
        NextAlphaM = CurAlphaM
        NextMM_plus = CurMM_plus
        NextMM_minus = CurMM_minus

        p = ComputeStressInvariantI1(NextStress) / 3.0
        if p < m_Pmin:
            NextStress = DeviatoricTensor(NextStress) + m_Pmin * mI1
            p = ComputeStressInvariantI1(NextStress) / 3.0

        aCep_Consistent = ti.Matrix.zero(float, 6, 6)
        n_trial = self.GetNormalToYield(NextStress, NextAlpha)

        MM_trial = NextMM_plus + NextMM_minus
        r_alphaM_trial = NextAlphaM + root23 * (MM_trial - self.m) * n_trial
        rM_Alphatilde_trial = NextAlphaM - root23 * (MM_trial - self.m) * n_trial

        bM_distance_trial = voigt_tensor_dot(r_alphaM_trial - NextAlpha, n_trial)
        bM_distance_tilde_trial = voigt_tensor_dot(NextAlpha - rM_Alphatilde_trial, n_trial)

        done = False
        while T < 1.0 and not done:
            NextVoidRatio = CurVoidRatio - (1.0 + CurVoidRatio) * ComputeStressInvariantI1(T * dStrain)
            NextVoidRatio = clamp(self.emin, self.emax, NextVoidRatio)

            dVolStrain = dT * ComputeStressInvariantI1(dStrain)
            dDevStrain = dT * DeviatoricTensor(dStrain)

            # Calc Delta 1
            thisSigma = NextStress
            thisAlpha = NextAlpha
            thisAlphaM = NextAlphaM
            thisMM_plus = NextMM_plus
            thisMM_minus = NextMM_minus
            thisVoidRatio = NextVoidRatio
            p = ComputeStressInvariantI1(thisSigma) / 3.0
            if p < m_Pmin:
                thisSigma = DeviatoricTensor(thisSigma) + m_Pmin * mI1
                p = ComputeStressInvariantI1(thisSigma) / 3.0

            r = DeviatoricTensor(thisSigma) / p

            G, K = self.GetElasticModuli(thisSigma, thisVoidRatio, 1.0)
            n, d, b, bM, Cos3Theta, h, hM, psi, rBtheta, rDtheta, b0, A, D, B, C, R, Z = self.GetStateDependent(
                thisSigma, thisAlpha, thisAlphaM, thisMM_plus, thisMM_minus, Curalpha_in, thisVoidRatio
            )

            # r = DeviatoricTensor(NextStress) / p
            Kp = two3 * p * h * voigt_tensor_dot(b, n)
            temp4 = (
                Kp
                + 2.0
                * G
                * (B - C * ComputeStressInvariantI1(SymmetricTensorSingleDot(n, SymmetricTensorSingleDot(n, n))))
                - K * D * voigt_tensor_dot(n, r)
            )
            if abs(temp4) < small:
                temp4 = small
            NextDGamma = (2.0 * G * voigt_tensor_dot(n, dDevStrain) - K * dVolStrain * voigt_tensor_dot(n, r)) / temp4
            dSigma1 = (
                2.0 * G * VoigtToContravariant(dDevStrain)
                + K * dVolStrain * mI1
                - macauley(NextDGamma)
                * (2.0 * G * (B * n - C * (SymmetricTensorSingleDot(n, n) - 1.0 / 3.0 * mI1)) + K * D * mI1)
            )
            dAlpha1 = macauley(NextDGamma) * two3 * h * b
            dAlphaM1 = macauley(NextDGamma) * two3 * hM * bM
            dPStrain1 = NextDGamma * VoigtToCovariant(R)
            depsilon_pv = ComputeStressInvariantI1(dPStrain1)
            dMM1_plus = 1 / root23 * voigt_tensor_dot(dAlphaM1, n)
            dMM1_minus = -Z * macauley(-depsilon_pv)

            # Calc Delta 2
            thisSigma = NextStress + 0.5 * dSigma1
            thisAlpha = NextAlpha + 0.5 * dAlpha1
            thisAlphaM = NextAlphaM + 0.5 * dAlphaM1
            thisMM_plus = NextMM_plus + 0.5 * dMM1_plus
            thisMM_minus = NextMM_minus + 0.5 * dMM1_minus
            # thisVoidRatio = m_e_init - (1.0 + m_e_init) * ComputeStressInvariantI1(NextStrain + 0.5 * dPStrain1)
            # thisVoidRatio = clamp(self.emin, self.emax, thisVoidRatio)

            p = ComputeStressInvariantI1(thisSigma) / 3.0
            if p < m_Pmin:
                thisSigma = DeviatoricTensor(thisSigma) + m_Pmin * mI1
                p = ComputeStressInvariantI1(thisSigma) / 3.0

            r = DeviatoricTensor(thisSigma) / p

            G, K = self.GetElasticModuli(thisSigma, thisVoidRatio, 1.0)
            n, d, b, bM, Cos3Theta, h, hM, psi, rBtheta, rDtheta, b0, A, D, B, C, R, Z = self.GetStateDependent(
                thisSigma, thisAlpha, thisAlphaM, thisMM_plus, thisMM_minus, Curalpha_in, thisVoidRatio
            )

            Kp = two3 * p * h * voigt_tensor_dot(b, n)
            temp4 = (
                Kp
                + 2.0
                * G
                * (B - C * ComputeStressInvariantI1(SymmetricTensorSingleDot(n, SymmetricTensorSingleDot(n, n))))
                - K * D * voigt_tensor_dot(n, r)
            )
            if abs(temp4) < small:
                temp4 = small
            NextDGamma = (2.0 * G * voigt_tensor_dot(n, dDevStrain) - K * dVolStrain * voigt_tensor_dot(n, r)) / temp4
            dSigma2 = (
                2.0 * G * VoigtToContravariant(dDevStrain)
                + K * dVolStrain * mI1
                - macauley(NextDGamma)
                * (2.0 * G * (B * n - C * (SymmetricTensorSingleDot(n, n) - 1.0 / 3.0 * mI1)) + K * D * mI1)
            )
            dAlpha2 = macauley(NextDGamma) * two3 * h * b
            dAlphaM2 = macauley(NextDGamma) * two3 * hM * bM
            dPStrain2 = NextDGamma * VoigtToCovariant(R)
            depsilon_pv = ComputeStressInvariantI1(dPStrain2)
            dMM2_plus = 1 / root23 * voigt_tensor_dot(dAlphaM2, n)
            dMM2_minus = -Z * macauley(-depsilon_pv)

            # Calc Delta 3
            thisSigma = NextStress + 0.25 * (dSigma1 + dSigma2)
            thisAlpha = NextAlpha + 0.25 * (dAlpha1 + dAlpha2)
            thisAlphaM = NextAlphaM + 0.25 * (dAlphaM1 + dAlphaM2)
            thisMM_plus = NextMM_plus + 0.25 * (dMM1_plus + dMM2_plus)
            thisMM_minus = NextMM_minus + 0.25 * (dMM1_minus + dMM2_minus)
            # thisVoidRatio = m_e_init - (1.0 + m_e_init) * ComputeStressInvariantI1(NextStrain + 0.25 * (dPStrain1 + dPStrain2))
            # thisVoidRatio = clamp(self.emin, self.emax, thisVoidRatio)

            p = ComputeStressInvariantI1(thisSigma) / 3.0
            if p < m_Pmin:
                thisSigma = DeviatoricTensor(thisSigma) + m_Pmin * mI1
                p = ComputeStressInvariantI1(thisSigma) / 3.0

            r = DeviatoricTensor(thisSigma) / p

            G, K = self.GetElasticModuli(thisSigma, thisVoidRatio, 1.0)
            n, d, b, bM, Cos3Theta, h, hM, psi, rBtheta, rDtheta, b0, A, D, B, C, R, Z = self.GetStateDependent(
                thisSigma, thisAlpha, thisAlphaM, thisMM_plus, thisMM_minus, Curalpha_in, thisVoidRatio
            )

            Kp = two3 * p * h * voigt_tensor_dot(b, n)
            temp4 = (
                Kp
                + 2.0
                * G
                * (B - C * ComputeStressInvariantI1(SymmetricTensorSingleDot(n, SymmetricTensorSingleDot(n, n))))
                - K * D * voigt_tensor_dot(n, r)
            )
            if abs(temp4) < small:
                temp4 = small
            NextDGamma = (2.0 * G * voigt_tensor_dot(n, dDevStrain) - K * dVolStrain * voigt_tensor_dot(n, r)) / temp4
            dSigma3 = (
                2.0 * G * VoigtToContravariant(dDevStrain)
                + K * dVolStrain * mI1
                - macauley(NextDGamma)
                * (2.0 * G * (B * n - C * (SymmetricTensorSingleDot(n, n) - 1.0 / 3.0 * mI1)) + K * D * mI1)
            )
            dAlpha3 = macauley(NextDGamma) * two3 * h * b
            dAlphaM3 = macauley(NextDGamma) * two3 * hM * bM
            dPStrain3 = NextDGamma * VoigtToCovariant(R)
            depsilon_pv = ComputeStressInvariantI1(dPStrain3)
            dMM3_plus = 1 / root23 * voigt_tensor_dot(dAlphaM3, n)
            dMM3_minus = -Z * macauley(-depsilon_pv)

            # Calc Delta 4
            thisSigma = NextStress - dSigma2 + 2 * dSigma3
            thisAlpha = NextAlpha - dAlpha2 + 2 * dAlpha3
            thisAlphaM = NextAlphaM - dAlphaM2 + 2 * dAlphaM3
            thisMM_plus = NextMM_plus - dMM2_plus + 2 * dMM3_plus
            thisMM_minus = NextMM_minus - dMM2_minus + 2 * dMM3_minus
            # thisVoidRatio = m_e_init - (1.0 + m_e_init) * ComputeStressInvariantI1(NextStrain - dPStrain2 + 2 * dPStrain3)
            # thisVoidRatio = clamp(self.emin, self.emax, thisVoidRatio)

            p = ComputeStressInvariantI1(thisSigma) / 3.0
            if p < m_Pmin:
                thisSigma = DeviatoricTensor(thisSigma) + m_Pmin * mI1
                p = ComputeStressInvariantI1(thisSigma) / 3.0

            r = DeviatoricTensor(thisSigma) / p

            G, K = self.GetElasticModuli(thisSigma, thisVoidRatio, 1.0)
            n, d, b, bM, Cos3Theta, h, hM, psi, rBtheta, rDtheta, b0, A, D, B, C, R, Z = self.GetStateDependent(
                thisSigma, thisAlpha, thisAlphaM, thisMM_plus, thisMM_minus, Curalpha_in, thisVoidRatio
            )

            Kp = two3 * p * h * voigt_tensor_dot(b, n)
            temp4 = (
                Kp
                + 2.0
                * G
                * (B - C * ComputeStressInvariantI1(SymmetricTensorSingleDot(n, SymmetricTensorSingleDot(n, n))))
                - K * D * voigt_tensor_dot(n, r)
            )
            if abs(temp4) < small:
                temp4 = small
            NextDGamma = (2.0 * G * voigt_tensor_dot(n, dDevStrain) - K * dVolStrain * voigt_tensor_dot(n, r)) / temp4
            dSigma4 = (
                2.0 * G * VoigtToContravariant(dDevStrain)
                + K * dVolStrain * mI1
                - macauley(NextDGamma)
                * (2.0 * G * (B * n - C * (SymmetricTensorSingleDot(n, n) - 1.0 / 3.0 * mI1)) + K * D * mI1)
            )
            dAlpha4 = macauley(NextDGamma) * two3 * h * b
            dAlphaM4 = macauley(NextDGamma) * two3 * hM * bM
            dPStrain4 = NextDGamma * VoigtToCovariant(R)
            depsilon_pv = ComputeStressInvariantI1(dPStrain4)
            dMM4_plus = 1 / root23 * voigt_tensor_dot(dAlphaM4, n)
            dMM4_minus = -Z * macauley(-depsilon_pv)

            # Calc Delta 5
            thisSigma = NextStress + (7 * dSigma1 + 10 * dSigma2 + dSigma4) / 27
            thisAlpha = NextAlpha + (7 * dAlpha1 + 10 * dAlpha2 + dAlpha4) / 27
            thisAlphaM = NextAlphaM + (7 * dAlphaM1 + 10 * dAlphaM2 + dAlphaM4) / 27
            thisMM_plus = NextMM_plus + (7 * dMM1_plus + 10 * dMM2_plus + dMM4_plus) / 27
            thisMM_minus = NextMM_minus + (7 * dMM1_minus + 10 * dMM2_minus + dMM4_minus) / 27
            # thisVoidRatio = m_e_init - (1.0 + m_e_init) * ComputeStressInvariantI1(NextStrain + (7 * dPStrain1 + 10 * dPStrain2 + dPStrain4) / 27)
            # thisVoidRatio = clamp(self.emin, self.emax, thisVoidRatio)

            p = ComputeStressInvariantI1(thisSigma) / 3.0
            if p < m_Pmin:
                thisSigma = DeviatoricTensor(thisSigma) + m_Pmin * mI1
                p = ComputeStressInvariantI1(thisSigma) / 3.0

            r = DeviatoricTensor(thisSigma) / p

            G, K = self.GetElasticModuli(thisSigma, thisVoidRatio, 1.0)
            n, d, b, bM, Cos3Theta, h, hM, psi, rBtheta, rDtheta, b0, A, D, B, C, R, Z = self.GetStateDependent(
                thisSigma, thisAlpha, thisAlphaM, thisMM_plus, thisMM_minus, Curalpha_in, thisVoidRatio
            )

            Kp = two3 * p * h * voigt_tensor_dot(b, n)
            temp4 = (
                Kp
                + 2.0
                * G
                * (B - C * ComputeStressInvariantI1(SymmetricTensorSingleDot(n, SymmetricTensorSingleDot(n, n))))
                - K * D * voigt_tensor_dot(n, r)
            )
            if abs(temp4) < small:
                temp4 = small
            NextDGamma = (2.0 * G * voigt_tensor_dot(n, dDevStrain) - K * dVolStrain * voigt_tensor_dot(n, r)) / temp4
            dSigma5 = (
                2.0 * G * VoigtToContravariant(dDevStrain)
                + K * dVolStrain * mI1
                - macauley(NextDGamma)
                * (2.0 * G * (B * n - C * (SymmetricTensorSingleDot(n, n) - 1.0 / 3.0 * mI1)) + K * D * mI1)
            )
            dAlpha5 = macauley(NextDGamma) * two3 * h * b
            dAlphaM5 = macauley(NextDGamma) * two3 * hM * bM
            dPStrain5 = NextDGamma * VoigtToCovariant(R)
            depsilon_pv = ComputeStressInvariantI1(dPStrain5)
            dMM5_plus = 1 / root23 * voigt_tensor_dot(dAlphaM5, n)
            dMM5_minus = -Z * macauley(-depsilon_pv)

            # Calc Delta 6
            thisSigma = NextStress + (28 * dSigma1 - 125 * dSigma2 + 546 * dSigma3 + 54 * dSigma4 - 378 * dSigma5) / 625
            thisAlpha = NextAlpha + (28 * dAlpha1 - 125 * dAlpha2 + 546 * dAlpha3 + 54 * dAlpha4 - 378 * dAlpha5) / 625
            thisAlphaM = (
                NextAlphaM + (28 * dAlphaM1 - 125 * dAlphaM2 + 546 * dAlphaM3 + 54 * dAlphaM4 - 378 * dAlphaM5) / 625
            )
            thisMM_plus = (
                NextMM_plus
                + (28 * dMM1_plus - 125 * dMM2_plus + 546 * dMM3_plus + 54 * dMM4_plus - 378 * dMM5_plus) / 625
            )
            thisMM_minus = (
                NextMM_minus
                + (28 * dMM1_minus - 125 * dMM2_minus + 546 * dMM3_minus + 54 * dMM4_minus - 378 * dMM5_minus) / 625
            )
            # thisVoidRatio = m_e_init - (1.0 + m_e_init) * ComputeStressInvariantI1(NextStrain + (28 * dPStrain1 - 125 * dPStrain2 + 546 * dPStrain3 + 54 * dPStrain4 - 378 * dPStrain5) / 625)
            # thisVoidRatio = clamp(self.emin, self.emax, thisVoidRatio)

            p = ComputeStressInvariantI1(thisSigma) / 3.0
            if p < m_Pmin:
                thisSigma = DeviatoricTensor(thisSigma) + m_Pmin * mI1
                p = ComputeStressInvariantI1(thisSigma) / 3.0

            r = DeviatoricTensor(thisSigma) / p

            G, K = self.GetElasticModuli(thisSigma, thisVoidRatio, 1.0)
            n, d, b, bM, Cos3Theta, h, hM, psi, rBtheta, rDtheta, b0, A, D, B, C, R, Z = self.GetStateDependent(
                thisSigma, thisAlpha, thisAlphaM, thisMM_plus, thisMM_minus, Curalpha_in, thisVoidRatio
            )

            Kp = two3 * p * h * voigt_tensor_dot(b, n)
            temp4 = (
                Kp
                + 2.0
                * G
                * (B - C * ComputeStressInvariantI1(SymmetricTensorSingleDot(n, SymmetricTensorSingleDot(n, n))))
                - K * D * voigt_tensor_dot(n, r)
            )
            if abs(temp4) < small:
                temp4 = small
            NextDGamma = (2.0 * G * voigt_tensor_dot(n, dDevStrain) - K * dVolStrain * voigt_tensor_dot(n, r)) / temp4
            dSigma6 = (
                2.0 * G * VoigtToContravariant(dDevStrain)
                + K * dVolStrain * mI1
                - macauley(NextDGamma)
                * (2.0 * G * (B * n - C * (SymmetricTensorSingleDot(n, n) - 1.0 / 3.0 * mI1)) + K * D * mI1)
            )
            dAlpha6 = macauley(NextDGamma) * two3 * h * b
            dAlphaM6 = macauley(NextDGamma) * two3 * hM * bM
            dPStrain6 = NextDGamma * VoigtToCovariant(R)
            depsilon_pv = ComputeStressInvariantI1(dPStrain6)

            # Update
            dSigma = (dSigma1 + 4 * dSigma3 + dSigma4) / 6
            dAlpha = (dAlpha1 + 4 * dAlpha3 + dAlpha4) / 6
            dAlphaM = (dAlphaM1 + 4 * dAlphaM3 + dAlphaM4) / 6

            dMM_plus = (dMM1_plus + 4 * dMM3_plus + dMM4_plus) / 6
            dMM_minus = (dMM1_minus + 4 * dMM3_minus + dMM4_minus) / 6

            dPStrain = (dPStrain1 + 4 * dPStrain3 + dPStrain4) / 6

            nStress = NextStress + dSigma
            nAlpha = NextAlpha + dAlpha
            nAlphaM = NextAlphaM + dAlphaM
            nMM_plus = NextMM_plus + dMM_plus
            nMM_minus = NextMM_minus + dMM_minus

            n_trial = self.GetNormalToYield(nStress, nAlpha)

            if (nMM_plus + nMM_minus) < self.m:
                nMM_minus = self.m - nMM_plus

            MM_trial = nMM_plus + nMM_minus
            r_alphaM_trial = nAlphaM + root23 * (MM_trial - self.m) * n_trial

            bM_distance_trial = voigt_tensor_dot(r_alphaM_trial - nAlpha, n_trial)

            if bM_distance_trial < -1.0e-4:
                nAlphaM = nAlpha - root23 * (MM_trial - self.m) * n_trial

            rM_Alphatilde_trial = nAlphaM - root23 * (MM_trial - self.m) * n_trial
            bM_distance_tilde_trial = voigt_tensor_dot(nAlpha - rM_Alphatilde_trial, n_trial)

            if bM_distance_tilde_trial < -1.0e-4:
                nMM_plus += abs(bM_distance_tilde_trial) / root23

            # Compute error
            p = ComputeStressInvariantI1(nStress) / 3.0
            if p < m_Pmin:
                if dT == dT_min:
                    done = True
                dT = max(0.1 * dT, dT_min)
                continue

            if done == False:
                stressNorm = VoigtTensorNorm(NextStress)
                alphaNorm = VoigtTensorNorm(NextAlpha)

                curStepError1 = (
                    VoigtTensorNorm(-42 * dSigma1 - 224 * dSigma3 - 21 * dSigma4 + 162 * dSigma5 + 125 * dSigma6) / 336
                )
                if stressNorm >= 0.5:
                    curStepError1 /= 2.0 * stressNorm

                curStepError2 = (
                    VoigtTensorNorm(-42 * dAlpha1 - 224 * dAlpha3 - 21 * dAlpha4 + 162 * dAlpha5 + 125 * dAlpha6) / 336
                )
                if alphaNorm >= 0.5:
                    curStepError2 /= 2.0 * alphaNorm

                curStepError = max(curStepError1, curStepError2)

                if curStepError > TolE:
                    q = max(0.8 * pow(TolE / curStepError, 0.2), 0.1)

                    if dT == dT_min:
                        NextStress = nStress
                        NextAlpha = nAlpha
                        NextAlphaM = NextAlpha
                        NextMM_plus = CurMM_plus
                        NextMM_minus = self.m - CurMM_plus

                        T += dT
                    dT = max(q * dT, dT_min)
                else:
                    NextStress = nStress
                    NextAlpha = nAlpha
                    NextAlphaM = nAlphaM
                    NextMM_plus = nMM_plus
                    NextMM_minus = nMM_minus

                    q = min(0.8 * pow(TolE / curStepError, 0.2), 2.0)
                    T += dT
                    dT = max(q * dT, dT_min)
                    dT = min(dT, 1 - T)

        return NextStress, NextAlpha, NextAlphaM, NextMM_plus, NextMM_minus, NextVoidRatio

    @ti.func
    def IntersectionFactor(self, CurStress, dStrain, CurVoidRatio, CurAlpha, a0, a1):
        a = a0

        m_e_init = self.e_init

        strainInc = dStrain

        vR = CurVoidRatio - (1 + CurVoidRatio) * ComputeStressInvariantI1(a0 * strainInc)
        vR = clamp(self.emin, self.emax, vR)
        G, K = self.GetElasticModuli(CurStress, vR, 1)
        dSigma0 = a0 * (ComputeElasticStiffnessTensor(K, G) @ strainInc)
        f0 = self.GetF(CurStress + dSigma0, CurAlpha)

        vR = CurVoidRatio - (1 + CurVoidRatio) * ComputeStressInvariantI1(a1 * strainInc)
        vR = clamp(self.emin, self.emax, vR)
        G, K = self.GetElasticModuli(CurStress, vR, 1)
        dSigma1 = a1 * (ComputeElasticStiffnessTensor(K, G) @ strainInc)
        f1 = self.GetF(CurStress + dSigma1, CurAlpha)

        for i in range(10):
            a = a1 - f1 * (a1 - a0) / (f1 - f0)
            dSigma = a * (ComputeElasticStiffnessTensor(K, G) @ strainInc)
            f = self.GetF(CurStress + dSigma, CurAlpha)
            if abs(f) < mTolF:
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

        if a > (1 - small):
            a = 1.0
        if a < small:
            a = 0.0
        return a

    @ti.func
    def IntersectionFactor_Unloading(self, CurStress, dStrain, CurvoidRatio, CurAlpha):
        a, a0, a1 = 0.0, 0.0, 1.0
        nSub = 20

        m_e_init = self.e_init

        strainInc = dStrain

        vR = CurvoidRatio
        vR = clamp(self.emin, self.emax, vR)
        G, K = self.GetElasticModuli(CurStress, vR, 1)
        dSigma = ComputeElasticStiffnessTensor(K, G) @ strainInc

        done = False
        a_result = 0.0
        for i in range(nSub):
            if not done:
                da = (a1 - a0) / 2.0
                a = a1 - da
                f = self.GetF(CurStress + a * dSigma, CurAlpha)
                if f > mTolF:
                    a1 = a

                elif f < -mTolF:
                    a0 = a
                    done = True
                    a_result = self.IntersectionFactor(CurStress, strainInc, vR, CurAlpha, a0, a1)
                else:
                    done = True
                    a_result = a
                if i == (nSub - 1) and not done:
                    done = True
                    a_result = 0.0

        return a_result

    @ti.func
    def GetNormalToYield(self, stress, alpha):
        devStress = DeviatoricTensor(stress)
        p = ComputeStressInvariantI1(stress) * (1.0 / 3.0)

        n = ZEROVEC6f
        if abs(p) < m_Pmin:
            n = ZEROVEC6f
        else:
            n = devStress - p * alpha
            normN = VoigtTensorNorm(n)
            if normN < small:
                normN = small
            n = n / normN
        return n

    @ti.func
    def StressCorrection300(
        self, PreStress, PreAlpha, CurStress, CurAlpha, CuralphaM, Cur_MM_plus, Cur_MM_minus, Curalpha_in, e
    ):
        sNewstress = ZEROVEC6f
        sNewAlpha = ZEROVEC6f

        sqrt_two3 = ti.sqrt(two3)
        small_adim = 1.0e-12

        switch = 0

        PAR_m = self.m

        SI1 = EYE
        small = 1.0e-7
        i = 0

        sInterstress = CurStress
        sInterAlpha = CurAlpha

        f2 = self.GetF(PreStress, PreAlpha)

        f0 = 0.0
        f1 = 0.0
        temp2 = ZEROVEC6f
        res = ZEROVEC6f
        done = False
        while i <= 50 and not done:
            if switch == 0:
                p = ComputeStressInvariantI1(sInterstress) / 3.0

                rr = DeviatoricTensor(sInterstress) / p

                # TRACE_RR = rr[0] + rr[1] + rr[2]
                # TRACE_ALFA = sInterAlpha[0] + sInterAlpha[1] + sInterAlpha[2]

                f0 = self.GetF(sInterstress, sInterAlpha)

                sn, dd, bb, bM, Cos3Theta, h, hM, psi, rBtheta, rDtheta, b0, A, D, B, C, R, Z = self.GetStateDependent(
                    sInterstress, sInterAlpha, CuralphaM, Cur_MM_plus, Cur_MM_minus, Curalpha_in, e
                )

                PM = voigt_tensor_dot(bb, sn)
                PM = two3 * p * h * PM

                E_G, E_K = self.GetElasticModuli(sInterstress, e, 1)
                aC = ComputeElasticStiffnessTensor(E_K, E_G)

                temp1 = SymmetricTensorSingleDot(sn, sn)

                res = (B * sn) - (C * (temp1 - 1.0 / 3.0 * SI1)) + (1.0 / 3.0 * D * SI1)
                R = VoigtToCovariant(res)

                temp1 = aC @ R

                res2 = voigt_tensor_dot(sn, rr)
                res = sn - 1.0 / 3.0 * res2 * SI1
                temp2 = VoigtToCovariant(res)
                temp2 = aC @ temp2

                temp3 = voigt_tensor_dot(temp2, R)
                temp3 += PM

                dlambda = f0 / temp3

                temp4 = aC @ R
                sNewstress = sInterstress - dlambda * temp4
                sNewAlpha = sInterAlpha + dlambda * two3 * h * bb

                f1 = self.GetF(sNewstress, sNewAlpha)

                if abs(f1) < mTolF:
                    done = True
                if abs(f1) > abs(f2):
                    switch = 1
                else:
                    sInterstress = sNewstress
                    sInterAlpha = sNewAlpha
            else:
                temp5 = voigt_tensor_dot(temp2, res)
                dlambda = f0 / temp5
                sNewstress = sInterstress - dlambda * res
                sNewAlpha = sInterAlpha
                f1 = self.GetF(sNewstress, sNewAlpha)

                sInterstress = sNewstress
                sInterAlpha = sNewAlpha

                f0 = self.GetF(sNewstress, sNewAlpha)

                if abs(f0) < mTolF:
                    done = True

            if i == 50 and not done:
                if f1 > 0.0:
                    sNewstressBisection = self.bisectionOfYSCrossing(sNewstress, sNewAlpha)
                    f0 = self.GetF(sNewstressBisection, sNewAlpha)
                    if f0 < mTolF:
                        sNewstress = sNewstressBisection
                        done = True

                if not done:
                    p = ComputeStressInvariantI1(sNewstress)
                    p_sNewAlpha = ComputeStressInvariantI1(sNewAlpha)
                    sNewAlpha = RestoreDviatoricProperty(sNewAlpha, p_sNewAlpha)

                    sn = self.GetNormalToYield(sNewstress, sNewAlpha)

                    p = 1.0 / 3.0 * p
                    rr = sNewAlpha + sqrt_two3 * sn * PAR_m
                    ss = rr * p
                    rr_alfa = rr - sNewAlpha

                    sNewstress = rr * p
                    sNewstress[0] += p
                    sNewstress[1] += p
                    sNewstress[2] += p

                    f1 = self.GetF(sNewstress, sNewAlpha)

                    if f1 > mTolF:
                        done = True
            i += 1
        return sNewstress, sNewAlpha

    @ti.func
    def bisectionOfYSCrossing(self, CurStress, CurAlpha):
        sNewstress = ZEROVEC6f

        beta0 = 0.0
        beta1 = 1.0

        p0 = ComputeStressInvariantI1(CurStress) / 3.0
        s0 = DeviatoricTensor(CurStress)
        s1 = CurAlpha * p0

        pa = p0
        pC = p0

        done = False
        for it in range(50):
            if not done:
                beta = (beta0 + beta1) / 2.0
                sa = (s1 - s0) * beta0 + s0
                stressA = sa
                stressA[0] += p0
                stressA[1] += p0
                stressA[2] += p0

                sC = (s1 - s0) * beta + s0
                stressC = sC
                stressC[0] += p0
                stressC[1] += p0
                stressC[2] += p0

                Fa = self.GetF(stressA, CurAlpha)
                Fc = self.GetF(stressC, CurAlpha)

                if abs(Fc) <= mTolF:
                    sNewstress = stressC
                    done = True
                else:
                    if (Fc > 0.0 and Fa > 0.0) or (Fc < 0.0 and Fa < 0.0):
                        beta0 = beta
                    else:
                        beta1 = beta

        return sNewstress

    @ti.func
    def proj_bounding(self, T, phi):
        sig0 = -0.01
        pi = 3.141592653589793
        Tcorrected = T

        tr_sig = T[0] + T[1] + T[2]
        if tr_sig / 3.0 > sig0:
            Tcorrected[0] += sig0 - tr_sig / 3.0
            Tcorrected[1] += sig0 - tr_sig / 3.0
            Tcorrected[2] += sig0 - tr_sig / 3.0

            tr_sig = 3.0 * sig0

        w = ti.sin(1.4 * phi) ** 2
        rc = (1.0 - w) / (9.0 - w)

        s11 = Tcorrected[0] / tr_sig - 1.0 / 3.0
        s22 = Tcorrected[1] / tr_sig - 1.0 / 3.0
        s33 = Tcorrected[2] / tr_sig - 1.0 / 3.0
        s12 = Tcorrected[3] / tr_sig
        s23 = Tcorrected[4] / tr_sig
        s13 = Tcorrected[5] / tr_sig

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
            Tcorrected[0] = (s11 * root + 1.0 / 3.0) * tr_sig
            Tcorrected[1] = (s22 * root + 1.0 / 3.0) * tr_sig
            Tcorrected[2] = (s33 * root + 1.0 / 3.0) * tr_sig
            Tcorrected[3] = s12 * root * tr_sig
            Tcorrected[4] = s23 * root * tr_sig
            Tcorrected[5] = s13 * root * tr_sig

        return Tcorrected
