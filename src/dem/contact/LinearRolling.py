import taichi as ti
import math

from src.dem.contact.ContactKernel import *
from src.dem.contact.RollingModelBase import RollingModelBase
from src.dem.SceneManager import myScene
from src.physics_model.contact_model.LinearRollingModel import LinearRollingSurfaceProperty
from src.utils.ObjectIO import DictIO


# refers to Luding 2008 Introduction to discrete element method
class LinearRollingModel(RollingModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.surfaceProps = LinearRollingSurfaceProperty.field(shape=self.sims.max_material_num * self.sims.max_material_num)

    def calcu_critical_timestep(self, scene: myScene):
        mass = scene.find_particle_min_mass(self.sims)
        radius = scene.find_particle_max_radius(self.sims)
        stiffness = self.find_max_stiffness(radius)
        return ti.sqrt(mass / stiffness)

    def find_max_stiffness(self, radius):
        maxstiff = 0.
        for materialID1 in range(self.sims.max_material_num):
            for materialID2 in range(self.sims.max_material_num):
                componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                if not GlobalVariable.ADAPTIVESTIFF:
                    if self.surfaceProps[componousID].kn > 0.:
                        maxstiff = ti.max(ti.max(maxstiff, self.surfaceProps[componousID].kn), self.surfaceProps[componousID].ks)
                else:
                    if self.surfaceProps[componousID].kratio > 0.:
                        kn = math.pi * 0.5 * radius * self.surfaceProps[componousID].emod
                        maxstiff = ti.max(ti.max(maxstiff, kn), kn / self.surfaceProps[componousID].kratio)
        return maxstiff
            
    def add_surface_property(self, materialID1, materialID2, property):
        kn = DictIO.GetAlternative(property, 'NormalStiffness', 0.)
        ks = DictIO.GetAlternative(property, 'TangentialStiffness', 0.)
        kr = DictIO.GetAlternative(property, 'RollingStiffness', 0.)
        kt = DictIO.GetAlternative(property, 'TwistingStiffness', 0.)
        emod = DictIO.GetAlternative(property, 'EffectiveModulus', 0.)
        kratio = DictIO.GetAlternative(property, 'NormalToShearRatio', 0.)

        if kn == 0. and ks == 0. and kr == 0. and kt == 0. and emod == 0. and kratio == 0.:
            raise RuntimeError("Input error")
        if emod > 0. and kratio > 0.:
            GlobalVariable.ADAPTIVESTIFF = True
        if GlobalVariable.ADAPTIVESTIFF and (kn > 0. or ks > 0. or kr > 0. or kt > 0.):
            raise RuntimeError("Using Effective Modulus and NormalToShearRatio instead")

        mu = DictIO.GetEssential(property, 'Friction')
        rmu = DictIO.GetAlternative(property, 'RollingFriction', 0.)
        tmu = DictIO.GetAlternative(property, 'TwistingFriction', 0.)
        ndratio = DictIO.GetAlternative(property, 'NormalViscousDamping', 0.)
        sdratio = DictIO.GetAlternative(property, 'TangentialViscousDamping', 0.)
        rdratio = DictIO.GetAlternative(property, 'RollingViscousDamping', 0.)
        tdratio = DictIO.GetAlternative(property, 'TwistingViscousDamping', 0.)
        componousID = 0
        if materialID1 == materialID2:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, kr, kt, emod, kratio, mu, rmu, tmu, ndratio, sdratio, rdratio, tdratio)
        else:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, kr, kt, emod, kratio, mu, rmu, tmu, ndratio, sdratio, rdratio, tdratio)
            componousID = self.get_componousID(self.sims.max_material_num, materialID2, materialID1)
            self.surfaceProps[componousID].add_surface_property(kn, ks, kr, kt, emod, kratio, mu, rmu, tmu, ndratio, sdratio, rdratio, tdratio)
        return componousID
    
    def update_property(self, componousID, property_name, value, override):
        factor = 0
        if not override:
            factor = 1

        if property_name == "NormalStiffness":
            self.surfaceProps[componousID].kn = factor * self.surfaceProps[componousID].kn + value
        elif property_name == "TangentialStiffness":
            self.surfaceProps[componousID].ks = factor * self.surfaceProps[componousID].ks + value
        elif property_name == "Friction":
            self.surfaceProps[componousID].mu = factor * self.surfaceProps[componousID].mu + value
        elif property_name == "NormalViscousDamping":
            self.surfaceProps[componousID].ndratio = factor * self.surfaceProps[componousID].ndratio + value
        elif property_name == "TangentialViscousDamping":
            self.surfaceProps[componousID].sdratio = factor * self.surfaceProps[componousID].sdratio + value
