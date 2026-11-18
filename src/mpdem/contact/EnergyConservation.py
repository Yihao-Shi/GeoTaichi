import taichi as ti
import numpy as np

from src.physics_model.contact_model.EnergyConservingModel import PenaltyProperty
from src.dem.Simulation import Simulation as DEMSimulation
from src.dem.SceneManager import SceneManager as DEMScene
from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpm.SceneManager import SceneManager as MPMScene
from src.utils.ObjectIO import DictIO


class EnergyConservationModel(ContactModelBase):
    def __init__(self, max_material_num, types) -> None:
        super().__init__()
        self.surfaceProps = PenaltyProperty.field(shape=max_material_num * max_material_num)
        self.null_model = False
        self.model_type = 0

    def calcu_critical_timestep(self, mscene: MPMScene, dsims: DEMSimulation, dscene: DEMScene, max_material_num):
        mass = min(mscene.find_particle_min_mass(), dscene.find_particle_min_mass(dsims))
        stiffness = self.find_max_stiffness(max_material_num)
        return ti.sqrt(mass / stiffness)

    def find_max_stiffness(self, max_material_num):
        maxstiff = 0.0
        for material_id1 in range(max_material_num):
            for material_id2 in range(max_material_num):
                componous_id = self.get_componous_id(max_material_num, material_id1, material_id2)
                if self.surfaceProps[componous_id].kn > 0.0:
                    maxstiff = ti.max(
                        ti.max(maxstiff, self.surfaceProps[componous_id].kn), self.surfaceProps[componous_id].ks
                    )
        return maxstiff

    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        kn = DictIO.get_essential(property, "NormalStiffness")
        ks = DictIO.get_essential(property, "TangentialStiffness")
        mu = DictIO.get_essential(property, "Friction")
        theta = DictIO.get_alternative(property, "FreeParameter", 1.5)
        ndratio = DictIO.get_essential(property, "NormalViscousDamping")
        sdratio = DictIO.get_essential(property, "TangentialViscousDamping")
        componous_id = 0
        if materialID1 == materialID2:
            componous_id = self.get_componous_id(max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
        else:
            componous_id = self.get_componous_id(max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
            componous_id = self.get_componous_id(max_material_num, materialID2, materialID1)
            self.surfaceProps[componous_id].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
        return componous_id
