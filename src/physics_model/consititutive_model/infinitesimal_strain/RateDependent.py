import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import DpDsigma
from src.utils.constants import PI
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class RateDependent:
    def __init__(self, materials):
        friction1 = DictIO.GetAlternative(materials, "StaticFriction") * PI / 180.
        friction2 = DictIO.GetAlternative(materials, "DynamicFriction") * PI / 180.
        self.phi1 = np.tan(friction1)
        self.phi2 = np.tan(friction2)
        self.diameter_avg = DictIO.GetAlternative(materials, "AverageDiameter")
        self.inertial_number = DictIO.GetAlternative(materials, "InertialNumber")
        self.inertial_density = DictIO.GetAlternative(materials, "InertialDensity")

    @ti.func
    def GetInertialNumber(self, cohesion, pressure, shear_rate):
        new_pressure = pressure + cohesion / self.phi1
        return shear_rate * self.diameter_avg / ti.sqrt(new_pressure / self.inertial_density)

    @ti.func
    def GetMuI(self, cohesion, pressure, shear_rate):
        inertial_number = self.GetInertialNumber(cohesion, pressure, shear_rate)
        return self.phi1 + (self.phi2 - self.phi1) / (self.inertial_number / inertial_number + 1.)
    
    @ti.func
    def DMuIDsigma(self, cohesion, pressure, shear_rate):
        inertial_number = self.GetInertialNumber(cohesion, pressure, shear_rate)
        den = 3. * (self.inertial_number + inertial_number) * (self.inertial_number + inertial_number) * shear_rate * self.diameter_avg * ti.sqrt(pressure * self.inertial_density)
        return (self.phi2 - self.phi1) * self.inertial_number / den * DpDsigma()




