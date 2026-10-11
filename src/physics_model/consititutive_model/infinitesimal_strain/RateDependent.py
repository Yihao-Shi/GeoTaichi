import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import (
    SphericalTensor,
    ComputeStressInvariantJ2,
    DeviatoricTensor,
    AssembleMeanDeviaStress,
)
from src.utils.constants import PI
from src.utils.TypeDefination import vec6f
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class RateDependent:
    def __init__(self, materials):
        friction1 = DictIO.GetEssential(materials, "StaticFriction", "Friction")
        friction2 = DictIO.GetAlternative(materials, "DynamicFriction", friction1)
        for name, angle in (("StaticFriction", friction1), ("DynamicFriction", friction2)):
            if not np.isfinite(angle) or not 0.0 <= angle < 90.0:
                raise ValueError(f"{name} must lie in [0, 90) degrees")
        if friction2 < friction1:
            raise ValueError("DynamicFriction must be >= StaticFriction for mu(I)")
        self.phi1 = np.tan(friction1 * PI / 180.0)
        self.phi2 = np.tan(friction2 * PI / 180.0)
        self.diameter_avg = DictIO.GetEssential(materials, "AverageDiameter")
        self.inertial_number = DictIO.GetEssential(materials, "InertialNumber")
        self.inertial_density = DictIO.GetAlternative(
            materials, "GrainDensity", DictIO.GetAlternative(materials, "Density", 2650.0)
        )
        self.eps = DictIO.GetAlternative(materials, "eps", 0.0)
        for name, value in (
            ("AverageDiameter", self.diameter_avg),
            ("InertialNumber", self.inertial_number),
            ("GrainDensity", self.inertial_density),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.eps) or self.eps < 0.0:
            raise ValueError("eps must be finite and nonnegative")

    def print_message(self):
        print("Grain density: ", self.inertial_density)
        print("Static friction coefficient: ", self.phi1)
        print("Dynamic friction coefficient: ", self.phi2)
        print("Average diameter: ", self.diameter_avg)
        print("Inertial number: ", self.inertial_number)

    @ti.func
    def GetInertialNumber(self, pressure, shear_rate):
        inertial_number = 0.0
        if pressure > 0.0:
            inertial_number = ti.abs(shear_rate) * self.diameter_avg * ti.sqrt(self.inertial_density / pressure)
        return inertial_number

    @ti.func
    def DInertialNumberDpressure(self, pressure, shear_rate):
        derivative = 0.0
        if pressure > 0.0:
            derivative = -0.5 * self.GetInertialNumber(pressure, shear_rate) / pressure
        return derivative

    @ti.func
    def DInertialNumberDlambda(self, pressure, shear_rate, dt, dp_dlambda):
        # I(\lambda) = d / \Delta t \lambda \sqrt(\rho / p(\lambda))
        assert dt[None] > 0.0, "mu(I) requires a positive time step"
        derivative = 0.0
        if pressure > 0.0:
            derivative = (
                self.diameter_avg / dt[None] * ti.sqrt(self.inertial_density / pressure)
                + self.DInertialNumberDpressure(pressure, shear_rate) * dp_dlambda
            )
        return derivative

    @ti.func
    def GetMuI(self, pressure, shear_rate):
        mu = self.phi1
        if pressure > 0.0:
            rate = ti.abs(shear_rate)
            C = self.inertial_number * ti.sqrt(
                pressure / (self.inertial_density * self.diameter_avg * self.diameter_avg)
            )
            mu += rate * (self.phi2 - self.phi1) / (C + ti.sqrt(rate * rate + self.eps * self.eps))
        return mu

    @ti.func
    def GetFrictionAngle(self, pressure, shear_rate, static_angle):
        # Add the rate increment to the current static strength, preserving softening and random fields.
        mu = ti.tan(static_angle) + self.GetMuI(pressure, shear_rate) - self.phi1
        return ti.atan2(mu, 1.0)

    @ti.func
    def GetStressRatio(self, pressure, shear_rate, static_ratio):
        sin_static = 3.0 * static_ratio / (6.0 + static_ratio)
        static_angle = ti.asin(sin_static)
        angle = self.GetFrictionAngle(pressure, shear_rate, static_angle)
        sin_angle = ti.sin(angle)
        return 6.0 * sin_angle / (3.0 - sin_angle)

    @ti.func
    def DMuIDpressure(self, pressure, shear_rate):
        dmu_dp = 0.0
        if pressure > 0.0:
            rate = ti.abs(shear_rate)
            C = self.inertial_number * ti.sqrt(
                pressure / (self.inertial_density * self.diameter_avg * self.diameter_avg)
            )
            B = C + ti.sqrt(rate * rate + self.eps * self.eps)
            dmu_dp = -0.5 * rate * (self.phi2 - self.phi1) * C / (pressure * B * B)
        return dmu_dp

    @ti.func
    def DMuIDsigma(self, pressure, shear_rate, dp_dsigma):
        return self.DMuIDpressure(pressure, shear_rate) * dp_dsigma

    @ti.func
    def solve(self, shear_mod, trial_stress, dt):
        # ref: Continuum modelling and simulation of granular flows through their many phases
        assert shear_mod > 0.0 and dt[None] > 0.0, "mu(I) requires positive shear modulus and time step"
        const_para = shear_mod * dt[None]
        pressure_trial = -SphericalTensor(trial_stress)
        updated_stress = vec6f(0, 0, 0, 0, 0, 0)
        plastic_strain_rate = 0.0
        if pressure_trial >= 0.0:
            tau_trial = ti.sqrt(ti.max(ComputeStressInvariantJ2(trial_stress), 0.0))
            dev_stress_trial = DeviatoricTensor(trial_stress)
            updated_stress = trial_stress
            if tau_trial > self.phi1 * pressure_trial:
                lower, upper = 0.0, tau_trial / const_para
                for _ in range(60):
                    plastic_strain_rate = 0.5 * (lower + upper)
                    residual = (
                        tau_trial
                        - const_para * plastic_strain_rate
                        - pressure_trial * self.GetMuI(pressure_trial, plastic_strain_rate)
                    )
                    if residual > 0.0:
                        lower = plastic_strain_rate
                    else:
                        upper = plastic_strain_rate
                plastic_strain_rate = 0.5 * (lower + upper)
                ratio = ti.max(0.0, 1.0 - const_para * plastic_strain_rate / tau_trial)
                updated_stress = AssembleMeanDeviaStress(dev_stress_trial * ratio, -pressure_trial)
        return plastic_strain_rate * dt[None], updated_stress
