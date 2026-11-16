import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.utils.constants import FTOL, Threshold
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_trace, voigt_tensor_dot
import src.utils.GlobalVariable as GlobalVariable

# Tensao minima (Pa) para clamps de p' e p0_ref: evita divisao por zero e log(0).
# Unico valor de clamp da elasticidade, negligenciavel frente as tensoes de trabalho (~1e5 Pa).
P_MIN = 1.0e-4


@ti.data_oriented
class NorSandModel(PlasticMaterial):
    """
    NorSand monotonico (Jefferies 1993; Jefferies & Been 2015), alinhado a sub-rotina
    NorTxl_r2 da planilha NorTxlDSS (r12) de Jefferies, Shuttle & Been, que gerou a base NorSand4AI
    (Ozelim et al. 2024). Os numeros [Lnnn] sao linhas da extracao do VBA
    da planilha feita com olevba 0.60.2.

    Convencao do GeoTaichi: tracao positiva; internamente p' = -tr(sigma)/3 > 0 (Pa); a CSL usa p' em kPa.

    Formulacao (compressao positiva):
      CSL e estado:      e_c = Gamma - lambda ln(p'/1 kPa);  psi = e - e_c(p')
      Elasticidade:      G = Gmax_p0 (p'/p0)^Gexp,  K = 2(1+nu)/(3(1-2nu)) G   (p0 capturado por particula)
      Dilatancia minima: Dmin = min(chi_tc psi, Mtc)                         [L682-683]
      Razao de imagem:   Mi = Mtc (1 + N Dmin/Mtc) se Dmin <= 0; Mtc se Dmin > 0; Mi >= 0,3   [L1793-1800]
      Superficie:        F = q - Mi (1 + ln(p_i/p')) p'   (q de von Mises; no modo de verificacao TriaxialSignedQ,
                         q com sinal -sign(J3) q_vM, como a planilha)
      Modulo plastico:   sem limitador
      Fluxo:             Dp = Mi - eta (Rowe-Jefferies; associado no plano p'-q)
      Teto:              p_i,max = p' exp(-Dmin/Mi)                          [L706]
      Endurecimento:     dp_i/deps_q^p = H (p'/p_i)(p_i,max - p_i) + p_i S_soft,  H = max(H0 - Hpsi psi, 0,5/lambda)   [L707, L1833-1835]
      Amolecimento:      S_soft = -(1 - chi_tc lambda/Mi) Dp eta (K_tot/p')/etaL,  etaL = Mi - Dmin,
                         K_tot = K_w K/(K_w + K); K_w = 0 no drenado          [L379-398, L670, L699-700]
      Inicializacao:     OCR <- min(max(OCR,1), exp(-chi_tc psi0/Mi0)/exp(eta0/Mi0 - 1));  p_i/p' = OCR exp(eta0/Mi0 - 1),
                         eta0 = q_vM/p' por padrao; com TriaxialInitSign, -sign(J3) q_vM/p' [L315, L325-328, L539]

    Arquitetura:
      p_i        -> variavel interna plastica (internal_vars[0]), integrada pelo NBURKDP2 do arcabouco.
      void_ratio -> variavel cinematica, atualizada uma vez por passo: de = (1+e) tr(deps) (tracao positiva).
      material_params (congelados no passo): [0] e, [1] p_i, [2] p', [3] p0_ref, [4:15] parametros do material
                                             lidos do estado da particula (PerParticleParams).
      Atributos inertes mantidos por compatibilidade: G0, kappa, vc0, h; estado p_imax_prev nao usado.
    """

    def __init__(self, material_type="Solid", configuration="UL",
                 solver_type="Explicit", stress_integration="SubStepping"):
        super().__init__(material_type, configuration, solver_type, stress_integration)
        # --- Elasticidade nu-based: G = Gmax_p0*(p'/p0_ref)^Gexp ---
        self.Gmax_p0  = 60.0e6   # G na tensao media inicial p0 da particula [Pa]
        self.Gexp     = 0.5      # expoente da lei de potencia [-]
        self.nu       = 0.20     # coeficiente de Poisson, constante [-]
        # --- Inertes, mantidos por compatibilidade; nao participam da elasticidade ---
        self.G0       = 1.0e7    # inerte: modulo cisalhante da forma em kappa
        self.kappa    = 0.005    # inerte: indice de descompressao
        self.lmbda    = 0.05
        self.Gamma    = 0.90     # [CSL] intercepto em p'=1 kPa: e_c = Gamma - lambda*ln(p'[kPa])
        self.M        = 1.30
        self.N        = 0.3
        self.chi      = 3.5      # parametro material de dilatancia (Jefferies)
        self.vc0      = 1.85
        self.v0       = 1.80
        self.h        = 100.     # inerte: o endurecimento usa H0 e Hpsi
        self.H0       = 200.     # modulo de hardening em psi=0 [-]
        self.Hpsi     = 300.     # coef. de dependencia de estado (atributo .h5 'Hy') [-]
        self.S        = 1.0      # liga (S > 0) e escala o termo S_soft da planilha (BookSoft [L201])
        self.Kw       = 0.0      # modulo volumetrico do fluido K_w [Pa]; 0 = drenado (K_tot = 0) [L544-549]
        self.TriaxialInitSign = False   # eta0 com sinal (-sign J3) so no modo de verificacao triaxial
        self.PerParticleParams = False  # parametros lidos do estado da particula, preenchidos fora do init
        self.OCR      = 1.0      # razao de sobreconsolidacao p/ init de pi
        self.p_ref    = 1.0e5
        self.density  = 1900.
        self.is_soft  = True
        self.max_sound_speed = 0.

    # =========================================================================
    # Inicializacao Python
    # =========================================================================
    def model_initialize(self, material):
        self.density  = DictIO.GetAlternative(material, 'Density',  1900.)
        # Elasticidade nu-based (SI/Pa). A conversao de unidades dos atributos .h5
        # (Gmax_p0: MPa->Pa, p0: kPa->Pa) e feita na fronteira de dados,
        # nao aqui: o modelo opera sempre em SI.
        self.Gmax_p0  = DictIO.GetEssential(material,   'Gmax_p0')
        self.Gexp     = DictIO.GetEssential(material,   'Gexp')
        self.nu       = DictIO.GetEssential(material,   'nu')
        # G0 e kappa: retidos como alternativos (inertes na elasticidade nu-based).
        self.G0       = DictIO.GetAlternative(material, 'G0',    1.0e7)
        self.kappa    = DictIO.GetAlternative(material, 'kappa', 0.005)
        self.lmbda    = DictIO.GetEssential(material,   'lambda')
        # CSL canonica: e_c = Gamma - lambda*ln(p'[kPa]). Gamma e o intercepto
        # em p'=1 kPa (atributo .h5 'Gamma'); p' convertido Pa->kPa na avaliacao da CSL.
        self.Gamma    = DictIO.GetEssential(material,   'Gamma')
        self.M        = DictIO.GetEssential(material,   'M')
        self.N        = DictIO.GetEssential(material,   'N')
        self.chi      = DictIO.GetEssential(material,   'chi')
        # vc0: inerte; a CSL usa Gamma.
        self.vc0      = DictIO.GetAlternative(material,  'vc0', 0.0)
        self.v0       = DictIO.GetEssential(material,   'v0')
        self.H0   = DictIO.GetEssential(material,   'H0')
        self.Hpsi = DictIO.GetEssential(material,   'Hpsi')
        self.h    = DictIO.GetAlternative(material,  'h', 100.0)   # inerte
        self.S    = DictIO.GetAlternative(material,  'S', 1.0)     # S_soft ligado/escala
        self.Kw   = DictIO.GetAlternative(material,  'Kw', 0.0)    # K_w [Pa]; NorTxl_r2 usa 2e6 kPa nao drenado [L109]
        # Modo de verificacao triaxial com q com sinal (-sign J3) na inicializacao, na
        # superficie, no gradiente, no potencial e no endurecimento; TriaxialInitSign e o nome antigo da mesma chave.
        self.TriaxialInitSign = bool(DictIO.GetAlternative(material, 'TriaxialInitSign', False)) or \
            bool(DictIO.GetAlternative(material, 'TriaxialSignedQ', False))
        self.PerParticleParams = bool(DictIO.GetAlternative(material, 'PerParticleParams', False))
        self.OCR  = DictIO.GetAlternative(material,  'OCR', 1.0)   # init de pi
        self.p_ref    = DictIO.GetAlternative(material, 'p_ref', 1.0e5)
        self.add_coupling_material(material)

    def get_sound_speed(self):
        return 0.

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print("Constitutive model = NorSand (Jefferies-Rowe, 1993)")
        print(f"Model ID:           {materialID}")
        print(f"Density:            {self.density} kg/m3")
        print(f"Gmax_p0 (nu-based): {self.Gmax_p0:.3e} Pa  (G na tensao media inicial p0)")
        print(f"Gexp:               {self.Gexp}")
        print(f"nu (Poisson):       {self.nu}")
        print(f"G0    [RETIDO/INERTE]: {self.G0:.3e} Pa")
        print(f"kappa [RETIDO/INERTE]: {self.kappa}")
        print(f"lambda:             {self.lmbda}")
        print(f"M:                  {self.M}")
        print(f"N:                  {self.N}")
        print(f"chi (dilatancy):    {self.chi}")
        print(f"Gamma (CSL@1kPa):   {self.Gamma}")
        print(f"vc0 [RETIDO/INERTE]: {self.vc0}")
        print(f"v0 (= 1+e0):        {self.v0}  ->  e0 = {self.v0-1:.4f}")
        print(f"H0 (hardening@psi=0): {self.H0}")
        print(f"Hpsi (=.h5 'Hy'):     {self.Hpsi}")
        print(f"S (softening flag):   {self.S}")
        print(f"h [LEGADO/retido]:    {self.h}")
        print(f"p_ref:              {self.p_ref:.3e} Pa")
        print(f"e0 (= v0 - 1):      {self.v0 - 1.0:.4f}")
        # psi inicial canonico (e - e_c(p0)) depende do p0 da particula; calculado em runtime.
        print()

    # =========================================================================
    # Variaveis de estado por particula
    # =========================================================================
    def define_state_vars(self):
        return {
            'pi':          float,
            'void_ratio':  float,
            'p0_ref':      float,   # tensao media inicial p0, capturada por particula no init
            'p_imax_prev': float,   # gravado na inicializacao, nunca lido
            # parametros do material por particula (material_params[4:15]) e do estado inicial
            'c_Gamma': float, 'c_lambda': float, 'c_M': float, 'c_N': float, 'c_chi': float,
            'c_H0': float, 'c_Hpsi': float, 'c_Gmax_p0': float, 'c_Gexp': float, 'c_nu': float, 'c_Kw': float,
            'c_OCR': float, 'c_v0': float,
        }

    def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        sin_phi = 3. * self.M / (6. + self.M)
        return np.repeat(1. - sin_phi, end_index - start_index)

    # =========================================================================
    # Inicializacao canonica de pi, apos campo gravitacional / consolidacao:
    #   p_i/p' = OCR_ef exp(eta0/Mi0 - 1),  eta0 = q0/p0,
    #   OCR_ef = min(max(OCR,1), exp(-chi_tc psi0/Mi0)/exp(eta0/Mi0 - 1)).
    # Nao ha piso p_i >= p': o estado inicial fica sobre a superficie para OCR <= 1 e dentro
    # dela, com trecho elastico, para OCR > 1, como na planilha.
    # =========================================================================
    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        # Forma de NorTxl_r2 / CheckInputParameters (planilha r12).
        #   eta0 com sinal na planilha [L506-511]; aqui, so com TriaxialInitSign. Em 3D o sinal vem do terceiro invariante do desviador; na
        #   convencao do GeoTaichi (tracao positiva) J3 < 0 em compressao triaxial, logo sign(eta0) = -sign(J3);
        #   em cisalhamento puro (J3 = 0) adota-se +1. Exato no triaxial; fora dele so tem sentido
        #   fisico pleno quando a dependencia do angulo de Lode for ativada.
        #   Mi0 = M_psi_v3(Mtc, N, chi_tc*psi0) sem o trap de Dmin [L319, L534-536].
        #   OCR <- max(OCR, 1) [L315];  OCR <- min(OCR, exp(-chi_tc psi0/Mi0)/exp(eta0/Mi0 - 1)) [L325-328].
        #   pi/p' = OCR * exp(eta0/Mi0 - 1) [L539].
        # Sem PerParticleParams, os parametros por particula recebem os valores do material.
        if ti.static(not self.PerParticleParams):
            stateVars[np].c_Gamma = self.Gamma; stateVars[np].c_lambda = self.lmbda; stateVars[np].c_M = self.M
            stateVars[np].c_N = self.N; stateVars[np].c_chi = self.chi; stateVars[np].c_H0 = self.H0
            stateVars[np].c_Hpsi = self.Hpsi; stateVars[np].c_Gmax_p0 = self.Gmax_p0; stateVars[np].c_Gexp = self.Gexp
            stateVars[np].c_nu = self.nu; stateVars[np].c_Kw = self.Kw; stateVars[np].c_OCR = self.OCR
            stateVars[np].c_v0 = self.v0
        stress0 = particle[np].stress
        p_GT = SphericalTensor(stress0)
        p_c  = ti.max(-p_GT, P_MIN)
        q    = EquivalentDeviatoricStress(stress0)
        # Por padrao eta0 = q_vM/p' >= 0, coerente com a superficie
        # simetrica do kernel; com TriaxialInitSign = True (verificacao contra a NorTxl_r2) o sinal de
        # q = sigma_a - sigma_r vem de -sign(J3) em tracao positiva, J3 = (2/27)(sigma_a - sigma_r)^3 no triaxial.
        sgn  = 1.0
        if ti.static(self.TriaxialInitSign):
            J3 = ComputeStressInvariantJ3(stress0)
            if J3 > 0.0:
                sgn = -1.0
        eta0 = sgn * q / p_c

        e_init  = stateVars[np].c_v0 - 1.0
        stateVars[np].void_ratio = e_init
        stateVars[np].p0_ref     = p_c
        mp      = self.GetMaterialParameter(stress0, stateVars[np])
        chi     = mp[8]
        psi0    = self._psi(e_init, p_c, mp)
        Mi0     = self._Mi_de_Dmin(chi * psi0, mp)
        ocr_eff = ti.max(stateVars[np].c_OCR, 1.0)
        ocr_max = ti.exp(-chi * psi0 / Mi0) / ti.exp(eta0 / Mi0 - 1.0)
        ocr_eff = ti.min(ocr_eff, ocr_max)
        pi_init = ocr_eff * p_c * ti.exp(eta0 / Mi0 - 1.0)

        stateVars[np].pi         = ti.max(pi_init, P_MIN)
        stateVars[np].p_imax_prev = self._p_i_max(p_c, ti.max(pi_init, P_MIN), e_init, mp)


    @ti.func
    def ExplicitIntegration(self, np, previous_stress, de, dw, stateVars):
        # ---- Atualizacao cinematica do indice de vazios ----
        # Convencao GT tracao-positiva: tr(de) > 0 = expansao -> e aumenta.
        # Em trajetoria nao-drenada: tr(de) ~ 0 -> de_void ~ 0.
        e_old      = stateVars[np].void_ratio
        deps_v_tot = voigt_tensor_trace(de)
        de_void    = (1.0 + e_old) * deps_v_tot
        stateVars[np].void_ratio = ti.max(e_old + de_void, 0.05)
        # ------------------------------------------------------

        state_vars      = stateVars[np]
        internal_vars   = self.GetInternalVariables(state_vars)
        material_params = self.GetMaterialParameter(previous_stress, state_vars)

        alpha = self.CalculateElasticFactor(np, de, previous_stress, internal_vars,
                                             material_params, stateVars)

        stress = self.ComputeElasticStress(alpha, de, previous_stress, material_params)
        if ti.abs(1. - alpha) > Threshold:
            stress, internal_vars = self.NBURKDP2(np, (1. - alpha) * de, stress,
                                                   internal_vars, state_vars,
                                                   material_params)


        update_stress = stress + self.ComputeSigrotStress(dw, previous_stress)
        self.UpdateInternalVariables(np, internal_vars, stateVars)
        self.UpdateStateVariables(np, update_stress, internal_vars, stateVars)
        return update_stress

    # =========================================================================
    # Helpers internos
    # =========================================================================
    @ti.func
    def _eta_image(self, p_c, pi, void_ratio, mp):
        # Superficie log canonica: eta_image = Mi*(1 + ln(pi/p')), com Mi = razao de imagem
        # dependente de estado, identica a usada no gradiente e no fluxo.
        p_eff  = ti.max(p_c, P_MIN)
        pi_eff = ti.max(pi,  P_MIN)
        Mi     = self._Mi(void_ratio, p_eff, mp)
        return Mi * (1.0 + ti.log(pi_eff / p_eff))

    # =========================================================================
    # CSL canonica: linha de estado critico semi-log em p' [kPa].
    #   e_c(p') = Gamma - lambda*ln(p'[kPa]),  Gamma = e_c em p'=1 kPa
    #   psi(e,p') = e - e_c(p')                 (parametro de estado)
    # p' chega em Pa (SI); converte-se Pa->kPa (divide por 1000) na avaliacao.
    # Jefferies & Been (2015), cap. 3.
    # =========================================================================
    @ti.func
    def _e_csl(self, p_Pa, mp):
        p_kPa = ti.max(p_Pa, P_MIN) / 1000.0
        return mp[4] - mp[5] * ti.log(p_kPa)

    @ti.func
    def _psi(self, void_ratio, p_Pa, mp):
        return void_ratio - self._e_csl(p_Pa, mp)

    @ti.func
    def _chi_i(self, mp):
        # chi_i = chi_tc / (1 - lambda*chi_tc/Mtc) Nao usado.
        return mp[8] / (1.0 - mp[5] * mp[8] / mp[6])

    @ti.func
    def _Mi_de_Dmin(self, Dmin, mp):
        # M_psi_v3 da planilha r12 em compressao triaxial [L1793-1800]:
        #   Dmin > 0: Mi = Mtc;  Dmin <= 0: Mi = Mtc (1 + N Dmin / Mtc);  piso Mi >= 0,3.
        Mi = mp[6]
        if Dmin <= 0.0:
            Mi = mp[6] * (1.0 + mp[7] * Dmin / mp[6])
        return ti.max(Mi, 0.3)

    @ti.func
    def _Dmin(self, void_ratio, p_c, mp):
        # Dilatancia minima com o parametro de estado corrente (NorTxl_r2 [L682]),
        # com o trap Dmin <= Mi_tc [L683]; para Dmin > 0, Mi_tc = Mtc [L1794-1795], logo o trap e Dmin <= Mtc.
        psi = self._psi(void_ratio, p_c, mp)
        return ti.min(mp[8] * psi, mp[6])

    @ti.func
    def _dMi_dp(self, void_ratio, p_c, mp):
        # dMi/dp' com e fixo: psi = e - Gamma + lambda ln(p'/1 kPa) => dpsi/dp' = lambda/p'.
        # Mi = Mtc + N Dmin no ramo Dmin <= 0 (acima do piso 0,3); derivada nula no ramo Dmin > 0,
        # no trap Dmin = Mtc e no piso de Mi.
        Dmin = self._Dmin(void_ratio, p_c, mp)
        d = 0.0
        if Dmin <= 0.0 and mp[6] + mp[7] * Dmin > 0.3:
            d = mp[7] * mp[8] * mp[5] / p_c
        return d

    @ti.func
    def _Mi(self, void_ratio, p_c, mp):
        # Razao de imagem Mi = M_psi_v3(Mtc, N, Dmin) [L688, L1793-1800],
        # com Dmin = min(chi_tc*psi, Mtc) e psi = e - e_c(p') corrente, nao o psi_i de imagem.
        # Segundo argumento: p' (Pa).
        return self._Mi_de_Dmin(self._Dmin(void_ratio, p_c, mp), mp)

    @ti.func
    def _p_i_max(self, p_c, pi, void_ratio, mp):
        # Teto do endurecimento da NorTxl_r2,
        #   p_i,max / p' = exp(-Dmin / Mi_tc) [L706], com Dmin = min(chi_tc*psi, Mtc) e psi corrente [L682-683]
        #   e Mi_tc = M_psi_v3(Mtc, N, Dmin) [L688].
        #   O argumento pi fica na assinatura por compatibilidade com os chamadores; nao entra no calculo.
        Dmin = self._Dmin(void_ratio, p_c, mp)
        Mi   = self._Mi_de_Dmin(Dmin, mp)
        return ti.max(p_c * ti.exp(-Dmin / Mi), P_MIN)

    @ti.func
    def _K_el(self, p_c, mp):
        G = mp[11] * ti.pow(p_c / ti.max(mp[3], P_MIN), mp[12])
        return 2.0 * (1.0 + mp[13]) / (3.0 * (1.0 - 2.0 * mp[13])) * G

    @ti.func
    def _hardening_rate(self, void_ratio, p_c, pi, eta, K_el, mp):
        # dp_i/deps_q^p da NorTxl_r2 [L707]:
        #   dp_i/p_i = [ H(psi) (p'/p_i)^2 (p_i,max/p' - p_i/p') + S_soft ] deps_q^p
        #   => dp_i/deps_q^p = H (p'/p_i)(p_i,max - p_i) + p_i S_soft
        # S_soft = SoftTerm(K_tot/p', lambda, chi_tc, Mi_tc, Dp, eta, etaL) [L379-398, L699-700],
        #   K_tot = K_w K / (K_w + K) [L670] (nulo no drenado, K_w = 0), Dp = Mi - eta [L693], etaL = Mi_tc - Dmin [L699].
        Dmin = self._Dmin(void_ratio, p_c, mp)
        Mi   = self._Mi_de_Dmin(Dmin, mp)
        pmx  = p_c * ti.exp(-Dmin / Mi)
        H    = self._H_mod(void_ratio, p_c, mp)
        rate = H * (p_c / pi) * (pmx - pi)
        Kw   = mp[14]
        if self.S > 0.0 and Kw > 0.0:
            Ktot = Kw * K_el / (Kw + K_el)
            Ik   = Ktot / p_c
            lf   = 1.0 - mp[8] * mp[5] / Mi
            Dp   = Mi - eta
            etaL = Mi - Dmin
            soft = -lf * Dp * eta * Ik / etaL
            if etaL < 0.05 and eta > etaL:
                soft = -lf * Dp * Ik
            rate += self.S * pi * soft
        return rate

    @ti.func
    def _H_mod(self, void_ratio, p_c, mp):
        # Modulo de hardening dependente de estado:
        #   H = H0 - Hpsi*psi,  piso Hmin = 0.5/lambda.
        # Atributo .h5 'Hy' = Hpsi (positivo); o sinal de menos faz denso (psi<0)
        # => H maior => mais rigido.
        psi  = self._psi(void_ratio, p_c, mp)
        H    = mp[9] - mp[10] * psi
        Hmin = 0.5 / mp[5]
        return ti.max(H, Hmin)

    # =========================================================================
    # Modulos elasticos nu-based:
    #   G = Gmax_p0 * (p'/p0_ref)^Gexp   (referencia = tensao media inicial p0)
    #   K = 2(1+nu)/(3(1-2nu)) * G
    # Lei de potencia da NorTxl_r2 [L1736] e K/G com Poisson constante [L341].
    # Sem piso em K/G: o unico clamp e P_MIN em p' e p0_ref.
    # =========================================================================
    @ti.func
    def ComputeElasticModulus(self, stress, material_params):
        p_GT   = SphericalTensor(stress)
        p_c    = ti.max(-p_GT, P_MIN)
        p0_ref = ti.max(material_params[3], P_MIN)
        G      = material_params[11] * ti.pow(p_c / p0_ref, material_params[12])
        nu     = material_params[13]
        K      = 2.0 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu)) * G
        return K, G

    @ti.func
    def ComputeElasticStress(self, alpha, dstrain, stress, material_params):
        K, G = self.ComputeElasticModulus(stress, material_params)
        stress += ElasticTensorMultiplyVector(alpha * dstrain, K, G)
        return stress

    @ti.func
    def ComputeStressInvariants(self, stress):
        p = SphericalTensor(stress)
        q = EquivalentDeviatoricStress(stress)
        return p, q

    # =========================================================================
    # Funcao de escoamento (superficie log):
    #   f = q - eta_image(p_c, pi, e) * p_c,  eta_image = Mi*(1 + ln(pi/p'))
    # =========================================================================
    @ti.func
    def ComputeYieldFunction(self, stress, internal_vars, material_params):
        pi         = internal_vars[0]
        void_ratio = material_params[0]
        p_GT = SphericalTensor(stress)
        p_c  = ti.max(-p_GT, P_MIN)
        q    = self._sgn_q(stress) * EquivalentDeviatoricStress(stress)
        eta  = self._eta_image(p_c, pi, void_ratio, material_params)
        return q - eta * p_c

    @ti.func
    def ComputeYieldState(self, stress, internal_vars, material_params):
        f = self.ComputeYieldFunction(stress, internal_vars, material_params)
        return f > -FTOL, f

    # =========================================================================
    # df/dsigma da superficie log; a expressao, com o termo dMi/dp', esta no corpo da funcao.
    # =========================================================================
    @ti.func
    def ComputeDfDsigma(self, yield_state, stress, internal_vars, material_params):
        pi         = internal_vars[0]
        void_ratio = material_params[0]
        p_GT  = SphericalTensor(stress)
        p_c   = ti.max(-p_GT, P_MIN)
        pi_e  = ti.max(pi, P_MIN)
        Mi    = self._Mi(void_ratio, p_c, material_params)
        eta_i = self._eta_image(p_c, pi_e, void_ratio, material_params)   # Mi*(1+ln(pi/p'))
        # Gradiente exato de F = q - Mi(psi(p')) (1 + ln(pi/p')) p'.
        #   dF/dp_GT = Mi ln(pi/p') + (dMi/dp') (1 + ln(pi/p')) p'   (p_GT = -p'; e congelado no passo)
        dfdp_GT = (eta_i - Mi) + self._dMi_dp(void_ratio, p_c, material_params) * (eta_i / Mi) * p_c
        return self._sgn_q(stress) * DqDsigma(stress) + dfdp_GT * DpDsigma()

    # =========================================================================
    # dg/dsigma: dilatancia de Rowe na forma de Jefferies (1993). Difere de df/dsigma so pelo termo dMi/dp'.
    #   D = Mi - eta_curr  (dilatancia de Rowe; eta instantaneo)
    #   dg/dp_GT = eta_curr - Mi
    # O "M" desta regra e a razao de imagem Mi dependente de estado, identica a da
    # superficie e do gradiente, e nao o Mtc constante.
    # =========================================================================
    @ti.func
    def ComputeDgDsigma(self, yield_state, stress, internal_vars, material_params):
        pi         = internal_vars[0]
        void_ratio = material_params[0]
        p_GT     = SphericalTensor(stress)
        p_c      = ti.max(-p_GT, P_MIN)
        sgn      = self._sgn_q(stress)
        q        = sgn * EquivalentDeviatoricStress(stress)
        eta_curr = q / ti.max(p_c, P_MIN)
        Mi       = self._Mi(void_ratio, p_c, material_params)
        dgdp_GT  = eta_curr - Mi
        return sgn * DqDsigma(stress) + dgdp_GT * DpDsigma()

    # =========================================================================
    # Modulo plastico H = (dF/dp_i) (dp_i/deps_q^p); negativo no endurecimento
    # (convencao do arcabouco: den = dF:De:dg - H).
    # =========================================================================
    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma, stress,
                               internal_vars, state_vars, material_params):
        pi         = internal_vars[0]
        void_ratio = material_params[0]
        p_c        = ti.max(material_params[2], P_MIN)
        pi_e       = ti.max(pi, P_MIN)

        Mi      = self._Mi(void_ratio, p_c, material_params)
        dfdpi   = -Mi * p_c / pi_e                  # dF/dpi da superficie log (Mi congelado)
        K, G    = self.ComputeElasticModulus(stress, material_params)
        eta     = self._sgn_q(stress) * EquivalentDeviatoricStress(stress) / p_c
        # dp_i/deps_q^p da NorTxl_r2 [L707], com S_soft; ver _hardening_rate.
        hard    = self._hardening_rate(void_ratio, p_c, pi_e, eta, K, material_params)

        # Sem limite inferior em H, como na planilha; abs(den) <= Threshold resulta em abeta = 0 e, em Substepping,
        # dlambda e limitado a >= 0.
        H = dfdpi * hard
        return H

    # =========================================================================
    # Incremento da unica variavel interna plastica:
    #   dp_i = (dp_i/deps_q^p) dlambda, ver _hardening_rate.
    # void_ratio nao e variavel interna; e atualizado cinematicamente.
    # =========================================================================
    @ti.func
    def ComputeInternalVariables(self, dlambda, dgdsigma, internal_vars, material_params):
        pi             = internal_vars[0]
        void_ratio_fix = material_params[0]
        p_c            = ti.max(material_params[2], P_MIN)
        pi_e           = ti.max(pi, P_MIN)

        Mi     = self._Mi(void_ratio_fix, p_c, material_params)
        # eta corrente a partir do gradiente do potencial: tr(dg/dsigma) = dg/dp_GT = eta - Mi
        eta    = Mi + (dgdsigma[0] + dgdsigma[1] + dgdsigma[2])
        K_el   = self._K_el(p_c, material_params)
        # dpi = [dp_i/deps_q^p da NorTxl_r2] * deps_q^p, deps_q^p = dlambda (dg/dq = 1) [L707]
        dpi    = self._hardening_rate(void_ratio_fix, p_c, pi_e, eta, K_el, material_params) * dlambda
        return ti.Vector([dpi])

    # =========================================================================
    # Substepping e ConsistentCorrection do arcabouco, sobrescritos so para
    # avaliar o endurecimento, o modulo plastico e K na p' corrente do subpasso (material_params[2]),
    # em vez da p' congelada no inicio do passo global.
    # =========================================================================
    @ti.func
    def _sgn_q(self, stress):
        # Modo de verificacao: sinal de q = sigma_a - sigma_r no triaxial, -sign(J3) em tracao positiva
        # (J3 = (2/27)(sigma_a - sigma_r)^3); fora do modo, superficie simetrica (sinal +1).
        sgn = 1.0
        if ti.static(self.TriaxialInitSign):
            if ComputeStressInvariantJ3(stress) > 0.0:
                sgn = -1.0
        return sgn

    @ti.func
    def _mp_corrente(self, stress, material_params):
        mp = material_params
        mp[2] = ti.max(-SphericalTensor(stress), P_MIN)
        return mp

    @ti.func
    def Substepping(self, dT, dstrain, stress, internal_vars, state_vars, material_params):
        mp = self._mp_corrente(stress, material_params)
        yield_state, _ = self.ComputeYieldState(stress, internal_vars, mp)
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(stress, mp)
        dsig_e = ElasticTensorMultiplyVector(dT * dstrain, bulk_modulus, shear_modulus)
        dfdsigma = self.ComputeDfDsigma(yield_state, stress, internal_vars, mp)
        dgdsigma = self.ComputeDgDsigma(yield_state, stress, internal_vars, mp)
        tempMat = ElasticTensorMultiplyVector(dgdsigma, bulk_modulus, shear_modulus)
        den = voigt_tensor_dot(dfdsigma, tempMat) - self.ComputePlasticModulus(yield_state, dgdsigma, stress, internal_vars, state_vars, mp)
        abeta = 1. / den if ti.abs(den) > Threshold else 0.
        dlambda = ti.max(voigt_tensor_dot(dfdsigma, dsig_e) * abeta, 0)
        dstress = dsig_e - dlambda * tempMat
        dinternal_vars = self.ComputeInternalVariables(dlambda, dgdsigma, internal_vars, mp)
        return dstress, dinternal_vars

    @ti.func
    def ConsistentCorrection(self, yield_state, f_function, stress, internal_vars, state_vars, material_params):
        mp = self._mp_corrente(stress, material_params)
        bulk_modulus, shear_modulus = self.ComputeElasticModulus(stress, mp)
        dfdsigma = self.ComputeDfDsigma(yield_state, stress, internal_vars, mp)
        dgdsigma = self.ComputeDgDsigma(yield_state, stress, internal_vars, mp)
        tempMat = ElasticTensorMultiplyVector(dgdsigma, bulk_modulus, shear_modulus)
        den = voigt_tensor_dot(dfdsigma, tempMat) - self.ComputePlasticModulus(yield_state, dgdsigma, stress, internal_vars, state_vars, mp)
        abeta = 1. / den if ti.abs(den) > Threshold else 0.
        dlambda = f_function * abeta
        dstress = dlambda * tempMat
        dinternal_vars = self.ComputeInternalVariables(dlambda, dgdsigma, internal_vars, mp)
        return stress - dstress, internal_vars + dinternal_vars

    # =========================================================================
    # Empacotamento / desempacotamento de estado
    # =========================================================================
    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        # state_vars.void_ratio aqui ja contem a atualizacao cinematica do
        # ExplicitIntegration sobrescrito.
        void_ratio = state_vars.void_ratio
        pi         = state_vars.pi
        p_GT       = SphericalTensor(stress)
        p_c        = ti.max(-p_GT, P_MIN)
        p0_ref     = state_vars.p0_ref
        # [0] e, [1] p_i, [2] p', [3] p0_ref, [4] Gamma, [5] lambda, [6] Mtc, [7] N, [8] chi_tc,
        #      [9] H0, [10] Hpsi, [11] Gmax_p0, [12] Gexp, [13] nu, [14] K_w
        return ti.Vector([void_ratio, pi, p_c, p0_ref,
                          state_vars.c_Gamma, state_vars.c_lambda, state_vars.c_M, state_vars.c_N, state_vars.c_chi,
                          state_vars.c_H0, state_vars.c_Hpsi, state_vars.c_Gmax_p0, state_vars.c_Gexp,
                          state_vars.c_nu, state_vars.c_Kw])

    @ti.func
    def GetInternalVariables(self, state_vars):
        return ti.Vector([state_vars.pi])

    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        stateVars[np].pi = ti.max(internal_vars[0], 1e-4)

    @ti.func
    def UpdateStateVariables(self, np, stress, internal_vars, stateVars):
        # Vazia: void_ratio ja atualizado em ExplicitIntegration (cinematica),
        # pi ja atualizado em UpdateInternalVariables (plasticidade).
        pass

    @ti.func
    def get_current_material_parameter(self, state_vars):
        return state_vars.pi, state_vars.void_ratio

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        material_params = self.GetMaterialParameter(current_stress, stateVars[np])
        K, G = self.ComputeElasticModulus(current_stress, material_params)
        return ComputeElasticStiffnessTensor(K, G)