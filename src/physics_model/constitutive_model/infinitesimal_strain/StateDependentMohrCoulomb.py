import taichi as ti
import numpy as np

from src.physics_model.constitutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.constitutive_model.infinitesimal_strain.InfinitesimalStrainModel import InfinitesimalStrainModel
from src.utils.constants import PI
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class StateDependentMohrCoulombModel(InfinitesimalStrainModel):
    def __init__(
        self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"
    ):
        super().__init__(material_type, configuration, solver_type)
        self.e0 = 0.0
        self.e_Tao = 0.0
        self.lambda_c = 0.0
        self.ksi = 0.0
        self.nd = 0.0
        self.nf = 0.0
        self.fai_c = 0.0
        self.c = 0.0
        self.fai = 0.0
        self.psi = 0.0
        self.tensile = 0.0

    def model_initialize(self, material):
        self.material = material
        density = DictIO.get_alternative(material, "Density", 2650)
        young = DictIO.get_essential(material, "YoungModulus")
        poisson = DictIO.get_alternative(material, "PoissonRatio", 0.3)
        e0 = DictIO.get_alternative(material, "e0", 0.3)
        e_Tao = DictIO.get_alternative(material, "e_Tao", 0.3)
        lambda_c = DictIO.get_alternative(material, "lambda_c", 0.3)
        ksi = DictIO.get_alternative(material, "ksi", 0.3)
        nd = DictIO.get_alternative(material, "nd", 0.3)
        nf = DictIO.get_alternative(material, "nf", 0.3)
        fai_c = DictIO.get_alternative(material, "fai_c", 0.3) * PI / 180.0
        c = DictIO.get_alternative(material, "Cohesion", 0.0)
        tensile = DictIO.get_alternative(material, "Tensile", 0.0)
        self.add_material(density, young, poisson, e0, e_Tao, lambda_c, ksi, nd, nf, fai_c, c, tensile)
        self.add_coupling_material(material)

    def add_material(self, density, young, poisson, e0, e_Tao, lambda_c, ksi, nd, nf, fai_c, c, tensile):
        self.density = density
        self.young = young
        self.poisson = poisson
        self.e0 = e0
        self.e_Tao = e_Tao
        self.lambda_c = lambda_c
        self.ksi = ksi
        self.nd = nd
        self.nf = nf
        self.fai_c = fai_c
        self.c = c
        self.shear = 0.5 * self.young / (1.0 + self.poisson)
        self.bulk = self.young / (3.0 * (1 - 2.0 * self.poisson))
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def print_message(self, materialID):
        self.print_console_header()
        print("Constitutive model: State-Dependent Mohr-Coulomb")
        print("Material ID: ", materialID)
        print("Density: ", self.density)
        print("Young Modulus: ", self.young)
        print("Poisson Ratio: ", self.poisson)
        print("Cohesion Coefficient = ", self.c)
        print("Internal Friction (in radian) = ", self.fai)
        print("Dilatation (in radian) = ", self.psi, "\n")

    def define_state_vars(self):
        return {"epstrain": float, "void_ratio": float}

    def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        fai = self.fai_c
        return np.repeat(1.0 - np.sin(fai), end_index - start_index)

    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, state_variables):
        state_variables[np].epstrain = 0.0
        state_variables[np].void_ratio = self.e0

    # ==================================================== Mohr-Coulomb Model ==================================================== #
    @ti.func
    def compute_stress2d(self, np, previous_stress, velocity_gradient, state_variables, dt):
        ############################## STEP1 ##############################
        de = calculate_strain_increment2d(velocity_gradient, dt)
        dw = calculate_vorticity_increment2d(velocity_gradient, dt)
        previous_stress = self.implicit_integration(np, previous_stress, de, dw, state_variables)
        return previous_stress

    @ti.func
    def compute_stress(self, np, previous_stress, velocity_gradient, state_variables, dt):
        de = calculate_strain_increment(velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)
        previous_stress = self.implicit_integration(np, previous_stress, de, dw, state_variables)
        return previous_stress

    @ti.func
    def implicit_integration(self, np, previous_stress, de, dw, state_variables):
        e_Tao = self.e_Tao
        lambda_c = self.lambda_c
        ksi = self.ksi
        fai_c = self.fai_c
        p = -(previous_stress[0] + previous_stress[1] + previous_stress[2]) / 3.0
        if p < 1000.0:
            p = 1000.0
        dv = de[0] + de[1] + de[2]
        e_c = e_Tao - lambda_c * (p / 101000.0) ** ksi
        e = state_variables[np].void_ratio
        state_variables[np].void_ratio = e + (1.0 + e) * dv
        if state_variables[np].void_ratio > 1.5:
            state_variables[np].void_ratio = 1.5
        elif state_variables[np].void_ratio < 0.1:
            state_variables[np].void_ratio = 0.1
        SP = state_variables[np].void_ratio - e_c
        nd, nf = self.nd, self.nf
        psi = ti.atan2(-nd * SP, 1.0)
        fai = ti.tan(fai_c) * ti.exp(-nf * SP)
        fai = ti.atan2(fai, 1.0)
        if psi < 0.0:
            psi = 0.0
        if fai <= fai_c:
            fai = fai_c

        sfai = ti.sin(fai)
        cfai = ti.cos(fai)
        spsi = ti.sin(psi)
        qfai = 6.0 * sfai / ti.sqrt(3.0) / (3.0 + sfai)
        kfai = 6.0 * self.c * cfai / ti.sqrt(3.0) / (3.0 + sfai)
        qpsi = 6.0 * spsi / ti.sqrt(3.0) / (3.0 + spsi)
        tenf = self.tensile
        bulk_mod = self.bulk
        shear_mod = self.shear
        if qfai == 0.0:
            tenf = 0.0
        else:
            tenf = ti.min(tenf, kfai / qfai)

        stress = previous_stress
        sigrot = sigrot(stress, dw)
        stress += sigrot
        dstress = elastic_tensor_multiply_vector(de, bulk_mod, shear_mod)
        trial_stress = stress + dstress
        sm = spherical_tensor(trial_stress)
        sd = deviatoric_tensor(trial_stress)
        epeff_ = state_variables[np].epstrain

        iplas = 0  # elastic calculation
        J2 = 0.5 * (sd[0] ** 2 + sd[1] ** 2 + sd[2] ** 2) + sd[3] ** 2 + sd[4] ** 2 + sd[5] ** 2
        Tau = ti.sqrt(J2)
        seqv = Tau * ti.sqrt(3.0)
        dp_fi = Tau + qfai * sm - kfai
        dpsig = sm - tenf

        if dpsig < 0.0:
            if dp_fi > 0.0:
                iplas = 1
                dlamd = dp_fi / (shear_mod + bulk_mod * qfai * qpsi)
                sm = sm - bulk_mod * qpsi * dlamd
                new_tau = kfai - qfai * sm
                ratio = new_tau / Tau
                # correct deviatoric stress
                sd = sd * ratio
                seqv = seqv * ratio
                # calculate the effective plastic strain
                depeff = dlamd * ti.sqrt(1.0 / 3.0 + (2.0 / 9.0) * (qpsi**2))
                epeff_ += depeff

        else:  # (dpsig >= 0.0)
            alphap = ti.sqrt(1.0 + qfai**2) - qfai
            taup = kfai - qfai * tenf
            dp_hfai = Tau - taup - alphap * dpsig

            if dp_hfai > 0.0:
                iplas = 1  # shear plastic flow
                # plastic flow coefficient
                dlamd = dp_fi / (shear_mod + bulk_mod * qfai * qpsi)
                # correct spherical stress
                sm = sm - bulk_mod * qpsi * dlamd
                # correct shear stress
                new_tau = kfai - qfai * sm
                ratio = new_tau / Tau
                # correct deviatoric stress
                sd = sd * ratio
                seqv = seqv * ratio  # correct the mises stress
                # calculate the effective plastic strain
                depeff = dlamd * ti.sqrt(1.0 / 3.0 + (2.0 / 9.0) * (qpsi**2))
                epeff_ += depeff
            else:
                iplas = 2  # tension plastic flow
                dlamd = (sm - tenf) / bulk_mod
                sm = tenf
                depeff = dlamd * (1.0 / 3.0) * ti.sqrt(2.0)
                epeff_ += depeff

        updated_stress = assemble_mean_devia_stress(sd, sm)
        # stateVars[np].estress = VonMisesStress(updated_stress)
        state_variables[np].epstrain = epeff_
        return updated_stress
