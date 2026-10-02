import taichi as ti
import math

from src.dem.contact.ContactKernel import *
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.SceneManager import myScene
from src.physics_model.contact_model.LinearModel import LinearSurfaceProperty
from src.utils.ObjectIO import DictIO
import src.utils.GlobalVariable as GlobalVariable


class LinearModel(ContactModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.surfaceProps = LinearSurfaceProperty.field(shape=self.sims.max_material_num * self.sims.max_material_num)
        self.null_model = False
        self.model_type = 1

    def calcu_critical_timestep(self, scene: myScene):
        mass = scene.find_particle_min_mass(self.sims.scheme)
        stiffness = self.find_max_stiffness(scene)
        return ti.sqrt(mass / stiffness)

    def find_max_stiffness(self, scene: myScene):
        maxstiff = 0.
        radius = scene.find_particle_max_radius(self.sims.scheme)
        if self.sims.scheme == "LSDEM":
            maxstiff = kernel_find_max_stiffness(int(scene.particleNum[0]), scene.rigid, scene.surface, scene.vertice, self.surfaceProps)
        else:
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
        emod = DictIO.GetAlternative(property, 'EffectiveModulus', 0.)
        kratio = DictIO.GetAlternative(property, 'NormalToShearRatio', 0.)

        if kn == 0. and ks == 0. and emod == 0. and kratio == 0.:
            raise RuntimeError("Input error")
        if emod > 0. and kratio > 0.:
            GlobalVariable.ADAPTIVESTIFF = True
        if GlobalVariable.ADAPTIVESTIFF and (kn > 0. or ks > 0.):
            raise RuntimeError("Using Effective Modulus and NormalToShearRatio instead")
        
        mus = DictIO.GetEssential(property, 'StaticFriction', 'Friction')
        mud = DictIO.GetEssential(property, 'DynamicFriction', 'Friction')
        rmu = DictIO.GetAlternative(property, 'RollingFriction', 0.)
        ndratio = DictIO.GetAlternative(property, 'NormalViscousDamping', 0)
        sdratio = DictIO.GetAlternative(property, 'TangentialViscousDamping', 0)
        if rmu > 0.:
            GlobalVariable.CONSTANTORQUEMODEL = True
        componousID = 0
        if materialID1 == materialID2:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, emod, kratio, mus, mud, rmu, ndratio, sdratio)
        else:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, emod, kratio, mus, mud, rmu, ndratio, sdratio)
            componousID = self.get_componousID(self.sims.max_material_num, materialID2, materialID1)
            self.surfaceProps[componousID].add_surface_property(kn, ks, emod, kratio, mus, mud, rmu, ndratio, sdratio)
        return componousID

    def update_property(self, componousID, property_name, value, override):
        factor = 0
        if not override:
            factor = 1

        if property_name == "NormalStiffness":
            self.surfaceProps[componousID].kn = factor * self.surfaceProps[componousID].kn + value
        elif property_name == "TangentialStiffness":
            self.surfaceProps[componousID].ks = factor * self.surfaceProps[componousID].ks + value
        elif property_name == "StaticFriction" or property_name == "Friction":
            self.surfaceProps[componousID].mus = factor * self.surfaceProps[componousID].mus + value
        elif property_name == "DynamicFriction" or property_name == "Friction":
            self.surfaceProps[componousID].mud = factor * self.surfaceProps[componousID].mud + value
        elif property_name == "NormalViscousDamping":
            self.surfaceProps[componousID].ndratio = factor * self.surfaceProps[componousID].ndratio + value
        elif property_name == "TangentialViscousDamping":
            self.surfaceProps[componousID].sdratio = factor * self.surfaceProps[componousID].sdratio + value
