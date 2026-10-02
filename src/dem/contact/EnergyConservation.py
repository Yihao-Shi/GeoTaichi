import taichi as ti

from src.dem.contact.ContactKernel import *
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.SceneManager import myScene
from src.physics_model.contact_model.EnergyConservingModel import PenaltyProperty
from src.physics_model.contact_model.BarrierModel import BarrierProperty
from src.utils.constants import ZEROVEC3f
from src.utils.ObjectIO import DictIO


class EnergyConservation(ContactModelBase):
    def __init__(self, sims, types) -> None:
        super().__init__(sims)
        if types == "Penalty":
            self.types = 1
            self.surfaceProps = PenaltyProperty.field(shape=self.sims.max_material_num * self.sims.max_material_num)
        elif types == "Barrier":
            self.types = 2
            self.surfaceProps = BarrierProperty.field(shape=self.sims.max_material_num * self.sims.max_material_num)

        self.null_model = False
        self.model_type = 0

    def calcu_critical_timestep(self, scene: myScene):
        mass = scene.find_particle_min_mass(self.sims.scheme)
        stiffness = self.find_max_stiffness(scene)
        return ti.sqrt(mass / stiffness)

    def find_max_stiffness(self, scene: myScene):
        maxstiff = 0.
        if self.types == 1:
            for materialID1 in range(self.sims.max_material_num):
                for materialID2 in range(self.sims.max_material_num):
                    componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                    if self.surfaceProps[componousID].kn > 0.:
                        maxstiff = ti.max(ti.max(maxstiff, self.surfaceProps[componousID].kn), self.surfaceProps[componousID].ks)
        elif self.types == 2:
            for materialID1 in range(self.sims.max_material_num):
                for materialID2 in range(self.sims.max_material_num):
                    componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                    if self.surfaceProps[componousID].kappa > 0.:
                        kappa = self.surfaceProps[componousID].kappa
                        ratio = kernel_get_min_ratio(componousID, int(scene.particleNum[0]), self.surfaceProps, scene.rigid)
                        kn = -kappa * (2. * ti.log(ratio) + ((ratio - 1) * (3 * ratio + 1)) / ratio ** 2)
                        maxstiff = ti.max(maxstiff, kn)
        return maxstiff
    
    def find_max_penetration(self):
        max_penetration = 0.
        if self.types == 2:
            for materialID1 in range(self.sims.max_material_num):
                for materialID2 in range(self.sims.max_material_num):
                    componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                    if self.surfaceProps[componousID].kappa > 0.:
                        max_penetration = ti.max(max_penetration, self.surfaceProps[componousID].ncut)
        return max_penetration
    
    def add_surface_property(self, materialID1, materialID2, property):
        if self.types == 1:
            kn = DictIO.GetEssential(property, 'NormalStiffness')
            ks = DictIO.GetEssential(property, 'TangentialStiffness')
            mu = DictIO.GetEssential(property, 'Friction')
            theta = DictIO.GetAlternative(property, 'FreeParameter', 2.5)
            ndratio = DictIO.GetAlternative(property, 'NormalViscousDamping', 0.)
            sdratio = DictIO.GetAlternative(property, 'TangentialViscousDamping', 0.)
            componousID = 0
            if materialID1 == materialID2:
                componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                self.surfaceProps[componousID].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
            else:
                componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                self.surfaceProps[componousID].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
                componousID = self.get_componousID(self.sims.max_material_num, materialID2, materialID1)
                self.surfaceProps[componousID].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
            return componousID
        elif self.types == 2:
            kappa = DictIO.GetEssential(property, 'Stiffness')
            ncut = DictIO.GetEssential(property, 'NormalCutOff')
            ratio = DictIO.GetAlternative(property, 'StiffnessRatio', 1.)
            mu = DictIO.GetEssential(property, 'Friction')
            ndratio = DictIO.GetAlternative(property, 'NormalViscousDamping', 0.)
            sdratio = DictIO.GetAlternative(property, 'TangentialViscousDamping', 0.)
            componousID = 0
            if materialID1 == materialID2:
                componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                self.surfaceProps[componousID].add_surface_property(kappa, ncut, ratio, mu, ndratio, sdratio)
            else:
                componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                self.surfaceProps[componousID].add_surface_property(kappa, ncut, ratio, mu, ndratio, sdratio)
                componousID = self.get_componousID(self.sims.max_material_num, materialID2, materialID1)
                self.surfaceProps[componousID].add_surface_property(kappa, ncut, ratio, mu, ndratio, sdratio)
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
    