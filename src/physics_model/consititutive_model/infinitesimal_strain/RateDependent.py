import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import SphericalTensor, ComputeStressInvariantJ2, DeviatoricTensor, AssembleMeanDeviaStress
from src.utils.constants import PI
from src.utils.TypeDefination import vec6f
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class RateDependent:
    def __init__(self, materials):
        friction1 = DictIO.GetEssential(materials, "StaticFriction", "Friction") * PI / 180.
        friction2 = DictIO.GetAlternative(materials, "DynamicFriction", friction1) * PI / 180.
        self.phi1 = np.tan(friction1)
        self.phi2 = np.tan(friction2)
        self.diameter_avg = DictIO.GetEssential(materials, "AverageDiameter")
        self.inertial_number = DictIO.GetEssential(materials, "InertialNumber")
        self.inertial_density = DictIO.GetAlternative(materials, "GrainDensity", DictIO.GetEssential(materials, "Density"))
        self.eps = DictIO.GetAlternative(materials, "eps", 0.)
    
    def print_message(self):
        print('Grain density: ', self.inertial_density)
        print('Static friction coefficient: ', self.phi1)
        print('Dynamic friction coefficient: ', self.phi2)
        print('Average diameter: ', self.diameter_avg)
        print('Inertial number: ', self.inertial_number)

    @ti.func
    def GetInertialNumber(self, pressure, shear_rate):
        pressure = ti.max(100, pressure)
        return shear_rate * self.diameter_avg * ti.sqrt(self.inertial_density / pressure)
    
    @ti.func
    def DInertialNumberDpressure(self, pressure, shear_rate):
        return -0.5 * shear_rate * self.diameter_avg * ti.sqrt(self.inertial_density) * pressure ** (-1.5)
    
    @ti.func
    def DInertialNumberDlambda(self, pressure, shear_rate, dt, dp_dlambda):
        # I(\lambda) = d / \Delta t \lambda \sqrt(\rho / p(\lambda))
        dI_dp = self.DInertialNumberDpressure(pressure, shear_rate)
        return self.diameter_avg / dt[None] * ti.sqrt(self.inertial_density / pressure) + dI_dp * dp_dlambda

    @ti.func
    def GetMuI(self, pressure, shear_rate):
        return self.phi1 + shear_rate * (self.phi2 - self.phi1) / (self.inertial_number * ti.sqrt(pressure / (self.inertial_density * self.diameter_avg * self.diameter_avg)) + ti.sqrt(shear_rate * shear_rate + self.eps * self.eps)) \
               if pressure > 0. else 0.
    
    @ti.func
    def DMuIDpressure(self, pressure, shear_rate):
        dmu_dp = 0.
        if pressure > 0.:
            inertial_number = self.GetInertialNumber(pressure, shear_rate)
            B = inertial_number * ti.sqrt(pressure / (self.inertial_density * self.diameter_avg * self.diameter_avg)) + ti.sqrt(shear_rate*shear_rate + self.eps*self.eps)
            A = -0.5 * shear_rate * (self.phi2 - self.phi1) * inertial_number * pressure ** (-0.5) / (ti.sqrt(self.inertial_density) * self.diameter_avg)
            dmu_dp = A / B / B
        return dmu_dp
    
    @ti.func
    def DMuIDsigma(self, pressure, shear_rate, dp_dsigma):
        return self.DMuIDpressure(pressure, shear_rate) * dp_dsigma
    
    @ti.func
    def solve(self, shear_mod, trial_stress, dt):
        # ref: Continuum modelling and simulation of granular flows through their many phases
        const_para = (shear_mod * dt[None])
        gap_mu = self.phi2 - self.phi1

        pressure_trial = -SphericalTensor(trial_stress)
        C = self.inertial_number * ti.sqrt(pressure_trial / (self.inertial_density * self.diameter_avg * self.diameter_avg))

        updated_stress = vec6f(0, 0, 0, 0, 0, 0)
        plastic_strain_rate = 0.
        if pressure_trial >= 0.:
            tau_trial = ti.sqrt(ComputeStressInvariantJ2(trial_stress))
            dev_stress_trial = DeviatoricTensor(trial_stress)

            plastic_strain_rate = (tau_trial - self.phi1 * pressure_trial) / (const_para + pressure_trial * gap_mu / (C + self.eps))
            plastic_strain_rate = ti.max(0., plastic_strain_rate)
            for _ in range(10):
                r = ti.sqrt(self.eps * self.eps + plastic_strain_rate * plastic_strain_rate)
                iden = 1. / (C + r)
                yield_func = tau_trial - const_para * plastic_strain_rate - pressure_trial * self.phi1 - pressure_trial * gap_mu * plastic_strain_rate * iden
                grad = -const_para - pressure_trial * gap_mu * (C + r - plastic_strain_rate * plastic_strain_rate / r) * iden * iden
                plastic_strain_rate_new = plastic_strain_rate - yield_func / grad
                plastic_strain_rate_new = ti.max(0., plastic_strain_rate_new)

                if ti.abs(plastic_strain_rate_new - plastic_strain_rate) < 1e-8:
                    break
                else:
                    plastic_strain_rate = plastic_strain_rate_new

            dev_stress = vec6f(0, 0, 0, 0, 0, 0)
            if tau_trial > 1e-12:
                dev_stress = dev_stress_trial * ti.max(0., (1. - const_para * plastic_strain_rate / tau_trial))
            updated_stress = AssembleMeanDeviaStress(dev_stress, -pressure_trial)
        return plastic_strain_rate * dt[None], updated_stress




