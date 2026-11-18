from src.utils.errors import SimulationStateError
import taichi as ti
import math

from src.physics_model.contact_model.LinearModel import LinearSurfaceProperty
from src.dem.Simulation import Simulation as DEMSimulation
from src.dem.SceneManager import SceneManager as DEMScene
from src.dem.contact.ContactKernel import *
from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpm.SceneManager import SceneManager as MPMScene
from src.utils.ObjectIO import DictIO


class LinearModel(ContactModelBase):
    def __init__(self, max_material_num) -> None:
        super().__init__()
        self.surfaceProps = LinearSurfaceProperty.field(shape=max_material_num * max_material_num)
        self.null_model = False
        self.model_type = 1

    def calcu_critical_timestep(self, mscene: MPMScene, dsims: DEMSimulation, dscene: DEMScene, max_material_num):
        mass = min(mscene.find_particle_min_mass(), dscene.find_particle_min_mass(dsims))
        stiffness = self.find_max_stiffness(max_material_num, mscene, dsims, dscene)
        return ti.sqrt(mass / stiffness)

    def find_max_stiffness(self, max_material_num, mscene: MPMScene, dsims: DEMSimulation, dscene: DEMScene):
        maxstiff = 0.0
        radius = max(mscene.find_particle_max_radius(), dscene.find_particle_max_radius(dsims))
        for material_id1 in range(max_material_num):
            for material_id2 in range(max_material_num):
                componous_id = self.get_componous_id(max_material_num, material_id1, material_id2)
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

    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        kn = DictIO.get_alternative(property, "NormalStiffness", 0.0)
        ks = DictIO.get_alternative(property, "TangentialStiffness", 0.0)
        emod = DictIO.get_alternative(property, "EffectiveModulus", 0.0)
        kratio = DictIO.get_alternative(property, "NormalToShearRatio", 0.0)

        if kn == 0.0 and ks == 0.0 and emod == 0.0 and kratio == 0.0:
            raise SimulationStateError("Input error")
        if emod > 0.0 and kratio > 0.0:
            GlobalVariable.ADAPTIVESTIFF = True
        if GlobalVariable.ADAPTIVESTIFF and (kn > 0.0 or ks > 0.0):
            raise SimulationStateError("Using Effective Modulus and NormalToShearRatio instead")

        mus = DictIO.get_essential(property, "StaticFriction", "Friction")
        mud = DictIO.get_essential(property, "DynamicFriction", "Friction")
        rmu = DictIO.get_alternative(property, "RollingFriction", 0.0)
        ndratio = DictIO.get_essential(property, "NormalViscousDamping")
        sdratio = DictIO.get_essential(property, "TangentialViscousDamping")
        componous_id = 0
        if materialID1 == materialID2:
            componous_id = self.get_componous_id(max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(kn, ks, emod, kratio, mus, mud, rmu, ndratio, sdratio)
        else:
            componous_id = self.get_componous_id(max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(kn, ks, emod, kratio, mus, mud, rmu, ndratio, sdratio)
            componous_id = self.get_componous_id(max_material_num, materialID2, materialID1)
            self.surfaceProps[componous_id].add_surface_property(kn, ks, emod, kratio, mus, mud, rmu, ndratio, sdratio)
        return componous_id
