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

    Convenção de sinais GeoTaichi: tração positiva.
        p_GT  = SphericalTensor(σ) ≤ 0   (tensão média, tração positiva)
        p_c   = -p_GT             > 0   (compressão positiva, uso interno)

    Parâmetros obrigatórios (dict material):
        G0      : coeficiente de rigidez cisalhante  [G = G0·√(p_c/p_ref)]
        kappa   : índice de recompressão elástica κ
        lambda  : índice de compressão virgem λ (CSL)
        M       : razão de tensões no estado crítico (triaxial compressão)
        N       : parâmetro de forma da superfície (0=log, 0<N<1=cap generalizado)
        beta    : parâmetro de dilatância plástica  (sempre > 0)
        vc0     : volume específico da CSL em p=1 kPa (escala ln natural)
        v0      : volume específico inicial = 1 + e0
        h       : módulo de hardening

    Parâmetros opcionais:
        p_ref   : pressão de referência para G (default 100 kPa)

    Variáveis de estado por partícula (stateVars, criadas em activate_state_variables):
        pi         : pressão de imagem (compressão positiva) > 0
        void_ratio : índice de vazios e = v - 1

    Vetor material_params (uso interno nos kernels Taichi):
        [0] = void_ratio   (estado atual do sub-passo)
        [1] = pi           (estado atual do sub-passo)
        [2] = p_c          (pressão no início do sub-passo = -p_GT)

    Vetor internal_vars (uso interno nos kernels Taichi):
        [0] = pi
        [1] = void_ratio
    """

    def __init__(self, material_type="Solid", configuration="UL",
                 solver_type="Explicit", stress_integration="SubStepping"):
        super().__init__(material_type, configuration, solver_type, stress_integration)
        # Parâmetros do modelo (inicializados em model_initialize)
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
    # Inicialização Python — chamada pelo framework antes da simulação
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
        print(f"G0:                 {self.G0}  [G=G0·√(p/p_ref)]")
        print(f"kappa (κ):          {self.kappa}")
        print(f"lambda (λ):         {self.lmbda}")
        print(f"M (CSL slope):      {self.M}")
        print(f"N (surface shape):  {self.N}")
        print(f"beta (dilatancy):   {self.beta_dil}")
        print(f"vc0 (CSL @ 1 kPa): {self.vc0}")
        print(f"v0 (= 1+e0):       {self.v0}  →  e0 = {self.v0-1:.4f}")
        print(f"h (hardening):      {self.h}")
        print(f"p_ref:              {self.p_ref} kPa\n")

    # =========================================================================
    # Declaração das variáveis de estado por partícula
    # O framework cria ti.Struct.field({'pi': float, 'void_ratio': float})
    # via activate_state_variables — seguindo o mesmo padrão do MCC {'pc': float}
    # =========================================================================
    def define_state_vars(self):
        return {
            'pi':         float,
            'void_ratio': float,
        }

    # =========================================================================
    # Coeficiente K0 para inicialização do campo de tensões geostáticas
    # Fórmula de Jaky:  K0 = 1 - sin(φ'),  sin(φ') = 3M/(6+M)
    # =========================================================================
    def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        sin_phi = 3. * self.M / (6. + self.M)
        return np.repeat(1. - sin_phi, end_index - start_index)

    # =========================================================================
    # Inicialização das variáveis de estado após campo gravitacional
    # Chamado pelo engine após as tensões geostáticas serem aplicadas
    # pi0 ≈ p_c (condição normalmente consolidada)
    # =========================================================================
    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        p_GT = SphericalTensor(particle[np].stress)   # ≤ 0 em compressão
        p_c  = ti.max(-p_GT, 1e-2)                    # compressão positiva
        stateVars[np].pi         = p_c                 # NC: pi0 = p0
        stateVars[np].void_ratio = self.v0 - 1.0      # e0 uniforme

    # =========================================================================
    # Razão de tensões de imagem η_i(p_c, pi)
    #
    # N = 0  → superfície logarítmica clássica (Jefferies 1993):
    #           η_i = M · (1 + ln(pi/p_c))
    #
    # N ≠ 0  → cap generalizado (Jefferies & Been 2006):
    #           η_i = (M/N) · [1 - (1-N)·(p_c/pi)^(N/(1-N))]
    # =========================================================================
    @ti.func
    def _eta_image(self, p_c, pi):
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

    # =========================================================================
    # Pressão de imagem na CSL: pi* = p_c · exp(ᾱ · ψ / M)
    # ᾱ = -3.5/β;  ψ = v - vc0 + λ·ln(p_c)
    # Solo fofo (ψ > 0): pi* > p_c → superfície se expande (hardening) ✓
    # Solo denso (ψ < 0): pi* < p_c → superfície se contrai (softening) ✓
    # =========================================================================
    @ti.func
    def _pi_star(self, p_c, void_ratio):
        v   = 1.0 + void_ratio
        psi = v - self.vc0 + self.lmbda * ti.log(ti.max(p_c, 1e-4))
        return ti.max(p_c * ti.exp(-3.5 / self.beta_dil * psi / self.M), 1e-4)

    # =========================================================================
    # Módulos elásticos dependentes do estado
    #   K = v · p_c / κ       (NorSand: rigidez volumétrica inclui volume esp.)
    #   G = G0 · √(p_c/p_ref) (módulo cisalhante dependente de pressão)
    #
    # material_params[0] = void_ratio;  material_params[2] = p_c (início do passo)
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
        """Atualização elástica: σ_trial = σ + Cₑ : (α·Δε)"""
        K, G = self.ComputeElasticModulus(stress, material_params)
        stress += ElasticTensorMultiplyVector(alpha * dstrain, K, G)
        return stress

    # =========================================================================
    # Invariantes de tensão (2-invariantes: p_GT e q)
    # =========================================================================
    @ti.func
    def ComputeStressInvariants(self, stress):
        p = SphericalTensor(stress)             # p_GT ≤ 0 para solo
        q = EquivalentDeviatoricStress(stress)  # q ≥ 0
        return p, q

    # =========================================================================
    # Função de plastificação NorSand
    #
    # Em termos de p_c (compressão +):  f = q - η_i · p_c
    # Em termos de p_GT (GeoTaichi):    f = q + η_i · p_GT   (p_GT = -p_c)
    #
    # f > 0 → estado plástico (fora da superfície de plastificação)
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
    # Gradiente da função de plastificação: ∂f/∂σ
    #
    # Derivação em convenção GeoTaichi (p_GT = SphericalTensor ≤ 0):
    #
    #   ∂f/∂σ = ∂f/∂q · DqDsigma(σ) + ∂f/∂p_GT · DpDsigma()
    #
    #   ∂f/∂q    = 1   (sempre)
    #
    #   ∂f/∂p_GT = η_i - M·(p_c/pi)^(N/(1-N))
    #
    # Verificação (N=0):  ∂f/∂p_GT = η_i - M  (= -(M-η_i))
    #   Solo fofo (η_i < M): ∂f/∂p_GT < 0 → aumento de compressão tende a fechar f ✓
    # =========================================================================
    @ti.func
    def ComputeDfDsigma(self, yield_state, stress, internal_vars, material_params):
        pi   = internal_vars[0]
        p_GT = SphericalTensor(stress)
        p_c  = ti.max(-p_GT, 1e-4)
        pi_e = ti.max(pi, 1e-4)
        eta  = self._eta_image(p_c, pi_e)
        exp_N     = self.N / (1.0 - self.N + 1e-12)   # N/(1-N)
        dfdp_GT   = eta - self.M * ti.pow(p_c / pi_e, exp_N)
        return DqDsigma(stress) + dfdp_GT * DpDsigma()

    # =========================================================================
    # Gradiente do potencial plástico: ∂g/∂σ  (regra de fluxo não-associada)
    #
    # Dilatância de Rowe:  D = dεv_p_c / dεs_p = M - η_curr
    #   (positivo → contrativo para solo fofo com η_curr < M)
    #
    # Em GeoTaichi (compressão negativa):
    #   dεv_p_GT / dεs_p = -(M - η_curr) = η_curr - M
    #
    # Logo: ∂g/∂p_GT = η_curr - M
    #   ∂g/∂σ = DqDsigma(σ) + (η_curr - M) · DpDsigma()
    #
    # Verificação:
    #   Solo fofo (η_curr < M):  ∂g/∂p_GT < 0 → tr(dε_p_GT) < 0 (compressão) ✓
    #   Estado crítico (η = M):  ∂g/∂p_GT = 0 → sem variação volumétrica ✓
    #   Solo denso (η > M):      ∂g/∂p_GT > 0 → dilatação ✓
    # =========================================================================
    @ti.func
    def ComputeDgDsigma(self, yield_state, stress, internal_vars, material_params):
        pi   = internal_vars[0]
        p_GT = SphericalTensor(stress)
        p_c  = ti.max(-p_GT, 1e-4)
        pi_e = ti.max(pi, 1e-4)
        q    = EquivalentDeviatoricStress(stress)
        # η_curr = q/p_c (razão de tensões atual, não a de imagem)
        eta_curr = q / ti.max(p_c, 1e-4)
        dgdp_GT  = eta_curr - self.M
        return DqDsigma(stress) + dgdp_GT * DpDsigma()

    # =========================================================================
    # Módulo de hardening H
    #
    # Regra de hardening NorSand (acionada por deformação plástica cisalhante):
    #   dpi = h · (pi* - pi) · dλ
    #
    # Gradiente da superfície em relação a pi:
    #   ∂f/∂pi = -M · p_c · (p_c/pi)^(N/(1-N)) / pi
    #   (sempre ≤ 0: aumento de pi expande a superfície, reduz f)
    #
    # Módulo de hardening (negativo para hardening — padrão GeoTaichi):
    #   H = ∂f/∂pi · h · (pi* - pi)
    #
    # Verificação de sinais:
    #   Solo fofo: pi* > pi → (pi*-pi) > 0; ∂f/∂pi < 0 → H < 0 (hardening) ✓
    #   Solo denso: pi* < pi → (pi*-pi) < 0; ∂f/∂pi < 0 → H > 0 (softening) ✓
    #   (Consistente com o padrão do MCC no GeoTaichi)
    # =========================================================================
    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma, stress,
                               internal_vars, state_vars, material_params):
        pi         = internal_vars[0]
        void_ratio = internal_vars[1]
        p_c        = ti.max(material_params[2], 1e-4)
        pi_e       = ti.max(pi, 1e-4)

        pi_star = self._pi_star(p_c, void_ratio)
        exp_N   = self.N / (1.0 - self.N + 1e-12)
        dfdpi   = -self.M * p_c * ti.pow(p_c / pi_e, exp_N) / pi_e
        H       = dfdpi * self.h * (pi_star - pi)
        return H

    # =========================================================================
    # Incremento das variáveis internas por sub-passo plástico
    #
    # Retorna [dpi, de] — incrementos que o framework acumula antes de
    # chamar UpdateInternalVariables (mesmo padrão do MCC que retorna [dpc])
    #
    # dpi = h · (pi* - pi) · dλ           [hardening de pi]
    # de  = (1+e) · tr(dλ · ∂g/∂σ_GT)    [variação de e via deformação plástica]
    #
    # Nota sobre de:
    #   ψ = v - vc0 + λ·ln(p_c) captura efeitos elásticos via p_c e plásticos via v.
    #   Aqui atualizamos e apenas via deformação PLÁSTICA; a mudança elástica de
    #   pressão (e portanto de ψ) é capturada pelo ln(p_c) no próximo sub-passo.
    # =========================================================================
    @ti.func
    def ComputeInternalVariables(self, dlambda, dgdsigma, internal_vars, material_params):
        pi         = internal_vars[0]
        void_ratio = internal_vars[1]
        p_c        = ti.max(material_params[2], 1e-4)
        v          = 1.0 + void_ratio
        pi_e       = ti.max(pi, 1e-4)

        pi_star = self._pi_star(p_c, void_ratio)

        # Incremento de pi
        dpi = self.h * (pi_star - pi) * dlambda

        # Incremento de e via deformação volumétrica plástica (convenção GT)
        dv_p_GT = voigt_tensor_trace(dlambda * dgdsigma)  # < 0 para compressão
        de      = v * dv_p_GT                              # (1+e)·dεv_p_GT

        return ti.Vector([dpi, de])

    # =========================================================================
    # Empacotamento do estado para os kernels Taichi
    # Seguindo o padrão exato do MCC:
    #   GetMaterialParameter → ti.Vector usado em todos os Compute*
    #   GetInternalVariables → ti.Vector dos escalares que evoluem plasticamente
    #   UpdateInternalVariables → escreve de volta em stateVars (recebe NEW values)
    # =========================================================================
    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        void_ratio = state_vars.void_ratio
        pi         = state_vars.pi
        p_GT       = SphericalTensor(stress)
        p_c        = ti.max(-p_GT, 1e-4)
        # [0] = void_ratio, [1] = pi, [2] = p_c no início do sub-passo
        return ti.Vector([void_ratio, pi, p_c])

    @ti.func
    def GetInternalVariables(self, state_vars):
        return ti.Vector([state_vars.pi, state_vars.void_ratio])

    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        # O framework acumula os deltas de ComputeInternalVariables e passa
        # os valores NOVOS aqui (mesmo comportamento do MCC com pc)
        stateVars[np].pi         = ti.max(internal_vars[0], 1e-4)
        stateVars[np].void_ratio = ti.max(internal_vars[1], 0.01)

    @ti.func
    def get_current_material_parameter(self, state_vars):
        return state_vars.pi, state_vars.void_ratio

    # =========================================================================
    # Para o solver implícito (não acionado pelo SubStepping explícito)
    # Implementado por completude — segue o mesmo padrão do MCC
    # =========================================================================
    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        material_params = self.GetMaterialParameter(current_stress, stateVars[np])
        K, G = self.ComputeElasticModulus(current_stress, material_params)
        return ComputeElasticStiffnessTensor(K, G)
