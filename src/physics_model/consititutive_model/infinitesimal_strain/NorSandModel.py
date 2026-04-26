import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.utils.constants import FTOL
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_trace, voigt_tensor_dot
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class NorSandModel(PlasticMaterial):
    """
    NorSand constitutive model — Jefferies (1993); Jefferies & Been (2006)
    Integrado ao framework PlasticMaterial do GeoTaichi.

    Convenção GeoTaichi: tração positiva.
        p_GT = SphericalTensor(σ) ≤ 0  (compressão → p_GT negativo)
        p_c  = -p_GT > 0               (compressão positiva, uso interno)

    Variável interna plástica (única): pi  (pressão de imagem)
    void_ratio: armazenado em stateVars, fixo durante substepping
                (evita instabilidade por acumulação de de dentro do NBURKDP2)

    material_params (vetor Taichi):
        [0] = void_ratio  (do início do passo — FIXO durante substepping)
        [1] = pi          (do início do passo)
        [2] = p_c         (= -p_GT do início do passo)

    internal_vars (vetor Taichi — único elemento):
        [0] = pi          (evolui durante substepping)
    """

    def __init__(self, material_type="Solid", configuration="UL",
                 solver_type="Explicit", stress_integration="SubStepping"):
        super().__init__(material_type, configuration, solver_type, stress_integration)
        self.G0       = 100.
        self.kappa    = 0.01
        self.lmbda    = 0.05
        self.M        = 1.25
        self.N        = 0.3
        self.beta_dil = 3.5
        self.vc0      = 1.9
        self.v0       = 1.8
        self.h        = 150.
        self.p_ref    = 100.
        self.density  = 1800.
        self.is_soft  = True
        self.max_sound_speed = 0.

    # =========================================================================
    # Inicialização Python
    # =========================================================================
    def model_initialize(self, material):
        self.density  = DictIO.GetAlternative(material, 'Density',  1800.)
        self.G0       = DictIO.GetEssential(material,   'G0')
        self.kappa    = DictIO.GetEssential(material,   'kappa')
        self.lmbda    = DictIO.GetEssential(material,   'lambda')
        self.M        = DictIO.GetEssential(material,   'M')
        self.N        = DictIO.GetEssential(material,   'N')
        self.beta_dil = DictIO.GetEssential(material,   'beta')
        self.vc0      = DictIO.GetEssential(material,   'vc0')
        self.v0       = DictIO.GetEssential(material,   'v0')
        self.h        = DictIO.GetEssential(material,   'h')
        self.p_ref    = DictIO.GetAlternative(material, 'p_ref', 100.)
        self.add_coupling_material(material)

    def get_sound_speed(self):
        return 0.

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print("Constitutive model = NorSand")
        print(f"Model ID:           {materialID}")
        print(f"Density:            {self.density} kg/m³")
        print(f"G0:                 {self.G0}")
        print(f"kappa (κ):          {self.kappa}")
        print(f"lambda (λ):         {self.lmbda}")
        print(f"M:                  {self.M}")
        print(f"N:                  {self.N}")
        print(f"beta (dilatancy):   {self.beta_dil}")
        print(f"vc0:                {self.vc0}")
        print(f"v0 (= 1+e0):       {self.v0}  →  e0 = {self.v0-1:.4f}")
        print(f"h (hardening):      {self.h}")
        print(f"p_ref:              {self.p_ref} kPa\n")

    # =========================================================================
    # Variáveis de estado por partícula
    # pi:         pressão de imagem (variável interna plástica)
    # void_ratio: índice de vazios (fixo durante substepping, atualizado no fim)
    # =========================================================================
    def define_state_vars(self):
        return {
            'pi':         float,
            'void_ratio': float,
        }

    def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        sin_phi = 3. * self.M / (6. + self.M)
        return np.repeat(1. - sin_phi, end_index - start_index)

    # =========================================================================
    # Inicialização pós-campo gravitacional
    # NC: pi0 = p_c0;  void_ratio = v0 - 1
    # =========================================================================
    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        p_GT = SphericalTensor(particle[np].stress)
        p_c  = ti.max(-p_GT, 1e-2)
        stateVars[np].pi         = p_c
        stateVars[np].void_ratio = self.v0 - 1.0

    # =========================================================================
    # Helpers internos
    # =========================================================================
    @ti.func
    def _eta_image(self, p_c, pi):
        """Razão de tensões de imagem η_i(p_c, pi)"""
        p_eff  = ti.max(p_c, 1e-4)
        pi_eff = ti.max(pi,  1e-4)
        result = 0.0
        if ti.abs(self.N) < 1e-10:
            result = self.M * (1.0 + ti.log(pi_eff / p_eff))
        else:
            result = (self.M / self.N) * (
                1.0 - (1.0 - self.N) * (p_eff / pi_eff) ** (self.N / (1.0 - self.N))
            )
        return result

    @ti.func
    def _pi_star(self, p_c, void_ratio):
        """Pressão de imagem na CSL: pi* = p_c * exp(ᾱ·ψ/M)"""
        v   = 1.0 + void_ratio
        psi = v - self.vc0 + self.lmbda * ti.log(ti.max(p_c, 1e-4))
        return ti.max(p_c * ti.exp(-3.5 / self.beta_dil * psi / self.M), 1e-4)

    # =========================================================================
    # Módulos elásticos
    # K = v·p_c/κ    G = G0·√(p_c/p_ref)
    # material_params[0] = void_ratio (FIXO do início do passo)
    # =========================================================================
    @ti.func
    def ComputeElasticModulus(self, stress, material_params):
        void_ratio = material_params[0]
        p_GT = SphericalTensor(stress)
        p_c  = ti.max(-p_GT, 1e-4)
        v    = 1.0 + void_ratio
        K    = ti.max(v * p_c / self.kappa, 100.0)
        G    = self.G0 * ti.sqrt(p_c / self.p_ref)
        return K, G

    @ti.func
    def ComputeElasticStress(self, alpha, dstrain, stress, material_params):
        K, G = self.ComputeElasticModulus(stress, material_params)
        stress += ElasticTensorMultiplyVector(alpha * dstrain, K, G)
        return stress

    # =========================================================================
    # Invariantes
    # =========================================================================
    @ti.func
    def ComputeStressInvariants(self, stress):
        p = SphericalTensor(stress)
        q = EquivalentDeviatoricStress(stress)
        return p, q

    # =========================================================================
    # Função de plastificação
    # f = q - η_i · p_c = q + η_i · p_GT
    # internal_vars[0] = pi (único)
    # =========================================================================
    @ti.func
    def ComputeYieldFunction(self, stress, internal_vars, material_params):
        pi   = internal_vars[0]
        p_GT = SphericalTensor(stress)
        p_c  = ti.max(-p_GT, 1e-4)
        q    = EquivalentDeviatoricStress(stress)
        eta  = self._eta_image(p_c, pi)
        return q - eta * p_c

    @ti.func
    def ComputeYieldState(self, stress, internal_vars, material_params):
        f = self.ComputeYieldFunction(stress, internal_vars, material_params)
        return f > -FTOL, f

    # =========================================================================
    # ∂f/∂σ  (gradiente da superfície de plastificação)
    #
    # f = q - η_i(p_c,pi)·p_c,   p_c = -p_GT
    #
    # ∂f/∂p_GT = η_i - M·(p_c/pi)^(N/(1-N))
    # ∂f/∂σ   = DqDsigma(σ) + dfdp_GT · DpDsigma()
    # =========================================================================
    @ti.func
    def ComputeDfDsigma(self, yield_state, stress, internal_vars, material_params):
        pi    = internal_vars[0]
        p_GT  = SphericalTensor(stress)
        p_c   = ti.max(-p_GT, 1e-4)
        pi_e  = ti.max(pi, 1e-4)
        eta   = self._eta_image(p_c, pi_e)
        exp_N = self.N / (1.0 - self.N + 1e-12)
        dfdp_GT = eta - self.M * ti.pow(p_c / pi_e, exp_N)
        return DqDsigma(stress) + dfdp_GT * DpDsigma()

    # =========================================================================
    # ∂g/∂σ  (gradiente do potencial plástico — regra de fluxo)
    #
    # Dilatância de Rowe:  D = M - η_curr  (η_curr = q/p_c atual)
    # ∂g/∂p_GT = η_curr - M = -D
    # ∂g/∂σ   = DqDsigma(σ) + (η_curr - M) · DpDsigma()
    #
    # Solo fofo (η < M):  ∂g/∂p_GT < 0 → dεv_p_GT < 0 → compressão ✓
    # =========================================================================
    @ti.func
    def ComputeDgDsigma(self, yield_state, stress, internal_vars, material_params):
        p_GT = SphericalTensor(stress)
        p_c  = ti.max(-p_GT, 1e-4)
        q    = EquivalentDeviatoricStress(stress)
        eta_curr = q / ti.max(p_c, 1e-4)
        dgdp_GT  = eta_curr - self.M
        return DqDsigma(stress) + dgdp_GT * DpDsigma()

    # =========================================================================
    # Módulo de hardening H  (negativo para hardening — padrão do framework)
    #
    # Framework usa:  den = ∂f:C:∂g - H
    # Para dlambda > 0:  H < 0 → hardening (den maior) ✓
    #
    # H = (∂f/∂pi) · (dpi/dλ)
    #   = (-M·p_c·(p_c/pi)^exp_N / pi) · h·(pi*-pi)
    #
    # Solo fofo: pi* > pi → (pi*-pi) > 0; ∂f/∂pi < 0 → H < 0 ✓ (hardening)
    # Solo denso: pi* < pi → H > 0 ✓ (softening)
    #
    # state_vars é stateVars[np] → acesso direto ao void_ratio armazenado
    # (ESTÁVEL: não o void_ratio acumulado no substepping)
    # =========================================================================
    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma, stress,
                               internal_vars, state_vars, material_params):
        pi         = internal_vars[0]
        # CRÍTICO: usa state_vars.void_ratio (fixo, do início do passo)
        # NÃO usa internal_vars[1] (que causava instabilidade por acumulação)
        void_ratio = state_vars.void_ratio
        p_c        = ti.max(material_params[2], 1e-4)
        pi_e       = ti.max(pi, 1e-4)

        pi_star = self._pi_star(p_c, void_ratio)
        exp_N   = self.N / (1.0 - self.N + 1e-12)
        dfdpi   = -self.M * p_c * ti.pow(p_c / pi_e, exp_N) / pi_e

        # H = (∂f/∂pi) · h · (pi* - pi)  →  negativo para hardening
        H = dfdpi * self.h * (pi_star - pi)

        # Limite físico: impede snap-back numérico
        K, G = self.ComputeElasticModulus(stress, material_params)
        H_min = -0.9 * (3.0 * G + K)
        return ti.max(H, H_min)

    # =========================================================================
    # Incremento da variável interna plástica (apenas pi)
    #
    # CORREÇÃO CENTRAL: retorna ti.Vector([dpi]) com UM único elemento.
    # Remover void_ratio do loop de substepping elimina a instabilidade:
    #   void_ratio não acumula de = v·dlambda·(-M) dentro do NBURKDP2,
    #   o que levava K = v·p/κ → 0 e explodia a simulação.
    # =========================================================================
    @ti.func
    def ComputeInternalVariables(self, dlambda, dgdsigma, internal_vars, material_params):
        pi         = internal_vars[0]
        void_ratio = material_params[0]   # fixo do início do passo
        p_c        = ti.max(material_params[2], 1e-4)

        pi_star = self._pi_star(p_c, void_ratio)
        dpi     = self.h * (pi_star - pi) * dlambda
        return ti.Vector([dpi])

    # =========================================================================
    # Empacotamento do estado para os kernels Taichi
    # =========================================================================
    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        void_ratio = state_vars.void_ratio
        pi         = state_vars.pi
        p_GT       = SphericalTensor(stress)
        p_c        = ti.max(-p_GT, 1e-4)
        return ti.Vector([void_ratio, pi, p_c])

    @ti.func
    def GetInternalVariables(self, state_vars):
        # ÚNICO elemento: pi
        return ti.Vector([state_vars.pi])

    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        stateVars[np].pi = ti.max(internal_vars[0], 1e-4)

    # =========================================================================
    # Atualização do void_ratio ao fim de cada passo (fora do substepping)
    # Usa relação elástica logarítmica:
    #   Δe = -κ · ln(p_c_new / p_c_old)
    # Aproximação consistente com a rigidez K = v·p/κ usada no modelo.
    # =========================================================================
    @ti.func
    def UpdateStateVariables(self, np, stress, internal_vars, stateVars):
        p_GT    = SphericalTensor(stress)
        p_c_new = ti.max(-p_GT, 1e-4)
        pi_new  = stateVars[np].pi
        e_old   = stateVars[np].void_ratio
        v_old   = 1.0 + e_old

        # Aproximação: variação de e via linha elástica κ
        # de ≈ -κ · ln(p_c_new / p_c_ref),  p_c_ref estimado via pi atual
        # Na linha NC: e_NC = vc0 - λ·ln(pi) → referência
        e_nc    = self.vc0 - 1.0 - self.lmbda * ti.log(pi_new)
        # e atual = e_NC + κ·ln(pi/p_c)  (swelling line)
        e_new   = e_nc + self.kappa * ti.log(pi_new / p_c_new)
        e_new   = ti.max(e_new, 0.05)   # físico: e > 0
        stateVars[np].void_ratio = e_new

    @ti.func
    def get_current_material_parameter(self, state_vars):
        return state_vars.pi, state_vars.void_ratio

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        material_params = self.GetMaterialParameter(current_stress, stateVars[np])
        K, G = self.ComputeElasticModulus(current_stress, material_params)
        return ComputeElasticStiffnessTensor(K, G)
