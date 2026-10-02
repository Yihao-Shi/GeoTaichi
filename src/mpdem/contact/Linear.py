import taichi as ti
import math

from src.physics_model.contact_model.LinearModel import LinearSurfaceProperty
from src.dem.Simulation import Simulation as DEMSimulation
from src.dem.SceneManager import myScene as DEMScene
from src.dem.contact.ContactKernel import *
from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpdem.contact.MultiLinkedCell import MultiLinkedCell
from src.mpm.SceneManager import myScene as MPMScene
from src.utils.ObjectIO import DictIO


class LinearModel(ContactModelBase):
    def __init__(self, max_material_num) -> None:
        super().__init__()
        self.surfaceProps = LinearSurfaceProperty.field(shape=max_material_num * max_material_num)
        self.null_model = False
        self.model_type = 1

    def calcu_critical_timestep(self, mscene: MPMScene, dsims: DEMSimulation, dscene: DEMScene, max_material_num):
        mass = min(mscene.find_particle_min_mass(), dscene.find_particle_min_mass(dsims.scheme))
        stiffness = self.find_max_stiffness(max_material_num, mscene, dsims, dscene)
        return ti.sqrt(mass / stiffness)

    def find_max_stiffness(self, max_material_num, mscene: MPMScene, dsims: DEMSimulation, dscene: DEMScene):
        maxstiff = 0.
        radius = max(mscene.find_particle_max_radius(), dscene.find_particle_max_radius(dsims.scheme))
        for materialID1 in range(max_material_num):
            for materialID2 in range(max_material_num):
                componousID = self.get_componousID(max_material_num, materialID1, materialID2)
                if not GlobalVariable.ADAPTIVESTIFF:
                    if self.surfaceProps[componousID].kn > 0.:
                        maxstiff = ti.max(ti.max(maxstiff, self.surfaceProps[componousID].kn), self.surfaceProps[componousID].ks)
                else:
                    if self.surfaceProps[componousID].kratio > 0.:
                        kn = math.pi * 0.5 * radius * self.surfaceProps[componousID].emod
                        maxstiff = ti.max(ti.max(maxstiff, kn), kn / self.surfaceProps[componousID].kratio)
        return maxstiff
    
    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
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
        ndratio = DictIO.GetEssential(property, 'NormalViscousDamping')
        sdratio = DictIO.GetEssential(property, 'TangentialViscousDamping')
        componousID = 0
        if materialID1 == materialID2:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, emod, kratio, mus, mud, rmu, ndratio, sdratio)
        else:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, emod, kratio, mus, mud, rmu, ndratio, sdratio)
            componousID = self.get_componousID(max_material_num, materialID2, materialID1)
            self.surfaceProps[componousID].add_surface_property(kn, ks, emod, kratio, mus, mud, rmu, ndratio, sdratio)
        return componousID

