import taichi as ti
import numpy as np

from src.physics_model.contact_model.EnergyConservingModel import PenaltyProperty
from src.dem.Simulation import Simulation as DEMSimulation
from src.dem.SceneManager import myScene as DEMScene
from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpm.SceneManager import myScene as MPMScene
from src.utils.ObjectIO import DictIO


class EnergyConservationModel(ContactModelBase):
    def __init__(self, max_material_num, types) -> None:
        super().__init__()
        self.surfaceProps = PenaltyProperty.field(shape=max_material_num * max_material_num)
        self.null_model = False
        self.model_type = 0

    def calcu_critical_timestep(self, mscene: MPMScene, dsims: DEMSimulation, dscene: DEMScene, max_material_num):
        mass = min(mscene.find_particle_min_mass(), dscene.find_particle_min_mass(dsims.scheme))
        stiffness = self.find_max_stiffness(max_material_num)
        return ti.sqrt(mass / stiffness)

    def find_max_stiffness(self, max_material_num):
        maxstiff = 0.
        for materialID1 in range(max_material_num):
            for materialID2 in range(max_material_num):
                componousID = self.get_componousID(max_material_num, materialID1, materialID2)
                if self.surfaceProps[componousID].kn > 0.:
                    maxstiff = ti.max(ti.max(maxstiff, self.surfaceProps[componousID].kn), self.surfaceProps[componousID].ks)
        return maxstiff
    
    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        kn = DictIO.GetEssential(property, 'NormalStiffness')
        ks = DictIO.GetEssential(property, 'TangentialStiffness')
        mu = DictIO.GetEssential(property, 'Friction')
        theta = DictIO.GetAlternative(property, 'FreeParameter', 1.5)
        ndratio = DictIO.GetEssential(property, 'NormalViscousDamping')
        sdratio = DictIO.GetEssential(property, 'TangentialViscousDamping')
        componousID = 0
        if materialID1 == materialID2:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
        else:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
            componousID = self.get_componousID(max_material_num, materialID2, materialID1)
            self.surfaceProps[componousID].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
        return componousID

    
