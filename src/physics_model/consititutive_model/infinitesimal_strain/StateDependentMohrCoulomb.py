import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.infinitesimal_strain.InfinitesimalStrainModel import InfinitesimalStrainModel
from src.utils.constants import PI
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class StateDependentMohrCoulombModel(InfinitesimalStrainModel):
    def __init__(self, material_type="Solid", configuration="UL", solver_type="Explicit", stress_integration="ReturnMapping"):
        super().__init__(material_type, configuration, solver_type)
        self.e0 = 0.
        self.e_Tao = 0.
        self.lambda_c = 0.
        self.ksi = 0.
        self.nd = 0.
        self.nf = 0.
        self.fai_c = 0.
        self.c = 0.
        self.fai = 0.
        self.psi = 0.
        self.tensile = 0.

    def model_initialize(self, material):
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        poisson = DictIO.GetAlternative(material, 'PoissonRatio', 0.3)
        e0 = DictIO.GetAlternative(material, 'e0', 0.3)
        e_Tao = DictIO.GetAlternative(material, 'e_Tao', 0.3)
        lambda_c = DictIO.GetAlternative(material, 'lambda_c', 0.3)
        ksi = DictIO.GetAlternative(material, 'ksi', 0.3)
        nd = DictIO.GetAlternative(material, 'nd', 0.3)
        nf = DictIO.GetAlternative(material, 'nf', 0.3)
        fai_c = DictIO.GetAlternative(material, 'fai_c', 0.3) * PI / 180.
        c = DictIO.GetAlternative(material, 'Cohesion', 0.)
        tensile = DictIO.GetAlternative(material, 'Tensile', 0.)
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
        self.shear = 0.5 * self.young / (1. + self.poisson)
        self.bulk = self.young / (3. * (1 - 2. * self.poisson))
        self.max_sound_speed = self.get_sound_speed(self.density, self.young, self.poisson)

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: Willam Mohr-Coulomb Model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Poisson Ratio: ', self.poisson)
        print('Cohesion Coefficient = ', self.c)
        print('Internal Friction (in radian) = ', self.fai)
        print('Dilatation (in radian) = ', self.psi, '\n')

    def define_state_vars(self):
        return {'epstrain': float, 'void_ratio': float}

    def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        fai = self.fai_c
        return np.repeat(1. - np.sin(fai), end_index - start_index)

    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        stateVars[np].epstrain = 0.
        stateVars[np].void_ratio = self.e0

    # ==================================================== Mohr-Coulomb Model ==================================================== #
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
    def ImplicitIntegration(self, np, previous_stress, de, dw, stateVars):
        e_Tao = self.e_Tao
        lambda_c = self.lambda_c
        ksi = self.ksi
        fai_c = self.fai_c
        p = -(previous_stress[0] + previous_stress[1] + previous_stress[2]) / 3.
        if p < 1000.:
            p = 1000.
        dv = de[0] + de[1] + de[2]
        e_c = e_Tao - lambda_c * (p/101000.)**ksi
        e = stateVars[np].void_ratio
        stateVars[np].void_ratio = e + (1.0 + e) * dv
        if stateVars[np].void_ratio > 1.5:
            stateVars[np].void_ratio = 1.5
        elif stateVars[np].void_ratio < 0.1:
            stateVars[np].void_ratio = 0.1
        SP = stateVars[np].void_ratio - e_c
        nd, nf = self.nd, self.nf
        psi = ti.atan2(-nd * SP, 1.)
        fai = ti.tan(fai_c) * ti.exp(-nf * SP)
        fai = ti.atan2(fai, 1.)
        if psi < 0.:
            psi = 0.
        if fai <= fai_c:
            fai = fai_c

        sfai = ti.sin(fai)
        cfai = ti.cos(fai)
        spsi = ti.sin(psi)
        qfai = 6. * sfai/ ti.sqrt(3.)/ (3.+sfai)
        kfai = 6.* self.c *cfai/ti.sqrt(3.)/(3.+sfai)
        qpsi = 6.* spsi/ ti.sqrt(3.)/ (3.+spsi)
        tenf = self.tensile
        bulk_mod = self.bulk
        shear_mod = self.shear
        if qfai ==0.:
            tenf =0.
        else:
            tenf = ti.min(tenf, kfai/qfai)

        stress = previous_stress
        sigrot = Sigrot(stress, dw)
        stress += sigrot
        dstress = ElasticTensorMultiplyVector(de, bulk_mod, shear_mod)
        trial_stress = stress + dstress
        sm = SphericalTensor(trial_stress)
        sd = DeviatoricTensor(trial_stress)
        epeff_ = stateVars[np].epstrain
        
        iplas = 0  # elastic calculation
        J2 = 0.5 * (sd[0]**2 + sd[1]**2 + sd[2]**2) + sd[3]**2 + sd[4]**2 + sd[5]**2
        Tau = ti.sqrt(J2)
        seqv = Tau * ti.sqrt(3.0)
        dpFi = Tau + qfai * sm - kfai
        dpsig = sm - tenf
        
        if dpsig < 0.0:
            if dpFi > 0.0:
                iplas = 1
                dlamd = dpFi / (shear_mod + bulk_mod * qfai * qpsi)
                sm = sm - bulk_mod * qpsi * dlamd
                newTau = kfai - qfai * sm
                ratio = newTau / Tau
                # correct deviatoric stress
                sd = sd * ratio
                seqv = seqv * ratio                
                # calculate the effective plastic strain
                depeff = dlamd * ti.sqrt(1./3. + (2./9.) * (qpsi**2))
                epeff_ += depeff
                
        else:  # (dpsig >= 0.0)
            alphap = ti.sqrt(1. + qfai**2) - qfai
            Taup = kfai - qfai * tenf
            dp_hfai = Tau - Taup - alphap * dpsig
                
            if dp_hfai > 0.0:
                iplas = 1  # shear plastic flow
                # plastic flow coefficient
                dlamd = dpFi / (shear_mod + bulk_mod * qfai * qpsi)
                # correct spherical stress
                sm = sm - bulk_mod * qpsi * dlamd
                # correct shear stress
                newTau = kfai - qfai * sm
                ratio = newTau / Tau
                 # correct deviatoric stress
                sd = sd * ratio
                seqv = seqv * ratio  # correct the mises stress
                # calculate the effective plastic strain
                depeff = dlamd * ti.sqrt(1./3. + (2./9.) * (qpsi**2))
                epeff_ += depeff
            else:
                iplas = 2  # tension plastic flow
                dlamd = (sm - tenf) / bulk_mod
                sm = tenf
                depeff = dlamd * (1./3.) * ti.sqrt(2.)
                epeff_ += depeff
        
        updated_stress = AssembleMeanDeviaStress(sd, sm)
        #stateVars[np].estress = VonMisesStress(updated_stress)
        stateVars[np].epstrain = epeff_
        return updated_stress