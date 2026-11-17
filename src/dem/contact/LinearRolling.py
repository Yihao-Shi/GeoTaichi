from src.utils.errors import SimulationStateError
import taichi as ti
import math

from src.dem.contact.ContactKernel import *
from src.dem.contact.RollingModelBase import RollingModelBase
from src.dem.SceneManager import SceneManager
from src.physics_model.contact_model.LinearRollingModel import LinearRollingSurfaceProperty
from src.utils.ObjectIO import DictIO


# refers to Luding 2008 Introduction to discrete element method
class LinearRollingModel(RollingModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.surfaceProps = LinearRollingSurfaceProperty.field(
            shape=self.sims.max_material_num * self.sims.max_material_num
        )

    def calcu_critical_timestep(self, scene: SceneManager):
        mass = scene.find_particle_min_mass(self.sims)
        radius = scene.find_particle_max_radius(self.sims)
        stiffness = self.find_max_stiffness(radius)
        return ti.sqrt(mass / stiffness)

    def find_max_stiffness(self, radius):
        maxstiff = 0.0
        for material_id1 in range(self.sims.max_material_num):
            for material_id2 in range(self.sims.max_material_num):
                componous_id = self.get_componous_id(self.sims.max_material_num, material_id1, material_id2)
                if not GlobalVariable.ADAPTIVESTIFF:
                    if self.surfaceProps[componous_id].kn > 0.0:
                        maxstiff = ti.max(
                            ti.max(maxstiff, self.surfaceProps[componous_id].kn), self.surfaceProps[componous_id].ks
                        )
                else:
                    if self.surfaceProps[componous_id].kratio > 0.0:
                        kn = math.pi * 0.5 * radius * self.surfaceProps[componous_id].emod
                        maxstiff = ti.max(ti.max(maxstiff, kn), kn / self.surfaceProps[componous_id].kratio)
        return maxstiff

    def add_surface_property(self, materialID1, materialID2, property):
        kn = DictIO.get_alternative(property, "NormalStiffness", 0.0)
        ks = DictIO.get_alternative(property, "TangentialStiffness", 0.0)
        kr = DictIO.get_alternative(property, "RollingStiffness", 0.0)
        kt = DictIO.get_alternative(property, "TwistingStiffness", 0.0)
        emod = DictIO.get_alternative(property, "EffectiveModulus", 0.0)
        kratio = DictIO.get_alternative(property, "NormalToShearRatio", 0.0)

        if kn == 0.0 and ks == 0.0 and kr == 0.0 and kt == 0.0 and emod == 0.0 and kratio == 0.0:
            raise SimulationStateError("Input error")
        if emod > 0.0 and kratio > 0.0:
            GlobalVariable.ADAPTIVESTIFF = True
        if GlobalVariable.ADAPTIVESTIFF and (kn > 0.0 or ks > 0.0 or kr > 0.0 or kt > 0.0):
            raise SimulationStateError("Using Effective Modulus and NormalToShearRatio instead")

        mu = DictIO.get_essential(property, "Friction")
        rmu = DictIO.get_alternative(property, "RollingFriction", 0.0)
        tmu = DictIO.get_alternative(property, "TwistingFriction", 0.0)
        ndratio = DictIO.get_alternative(property, "NormalViscousDamping", 0.0)
        sdratio = DictIO.get_alternative(property, "TangentialViscousDamping", 0.0)
        rdratio = DictIO.get_alternative(property, "RollingViscousDamping", 0.0)
        tdratio = DictIO.get_alternative(property, "TwistingViscousDamping", 0.0)
        componous_id = 0
        if materialID1 == materialID2:
            componous_id = self.get_componous_id(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(
                kn, ks, kr, kt, emod, kratio, mu, rmu, tmu, ndratio, sdratio, rdratio, tdratio
            )
        else:
            componous_id = self.get_componous_id(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(
                kn, ks, kr, kt, emod, kratio, mu, rmu, tmu, ndratio, sdratio, rdratio, tdratio
            )
            componous_id = self.get_componous_id(self.sims.max_material_num, materialID2, materialID1)
            self.surfaceProps[componous_id].add_surface_property(
                kn, ks, kr, kt, emod, kratio, mu, rmu, tmu, ndratio, sdratio, rdratio, tdratio
            )
        return componous_id

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
