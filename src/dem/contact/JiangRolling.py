import taichi as ti

from src.dem.contact.ContactKernel import *
from src.dem.contact.RollingModelBase import RollingModelBase
from src.dem.SceneManager import myScene
from src.physics_model.contact_model.RollingModel import JiangRollingSurfaceProperty
from src.utils.ObjectIO import DictIO


# Refers to Jiang et. al (2015) A novel three-dimensional contact model for granulates incorporating rolling and twisting resistances. Computer and Geotechnics
class JiangRollingResistanceModel(RollingModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.surfaceProps = JiangRollingSurfaceProperty.field(shape=self.sims.max_material_num * self.sims.max_material_num)

    def calcu_critical_timestep(self, scene: myScene):
        mass = scene.find_particle_min_mass(self.sims)
        radius = scene.find_particle_min_radius(self.sims)
        modulus = self._find_max_modulus_()
        stiffness = 2 * radius * modulus
        return ti.sqrt(mass / stiffness)

    def _find_max_modulus_(self):
        maxmodulus = 0.
        for materialID1 in range(self.sims.max_material_num):
            for materialID2 in range(self.sims.max_material_num):
                componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                if self.surfaceProps[componousID].YoungModulus > 0.:
                    maxmodulus = ti.max(maxmodulus, self.surfaceProps[componousID].YoungModulus)
        return maxmodulus

    def add_surface_property(self, materialID1, materialID2, property):
        YoungModulus = DictIO.GetEssential(property, 'YoungModulus')
        stiffness_ratio = DictIO.GetEssential(property, 'StiffnessRatio')
        mu = DictIO.GetEssential(property, 'Friction')
        shape_factor = DictIO.GetEssential(property, 'ShapeFactor')
        crush_factor = DictIO.GetEssential(property, 'CrushFactor')
        ndratio = DictIO.GetEssential(property, 'NormalViscousDamping')
        sdratio = DictIO.GetEssential(property, 'TangentialViscousDamping')
        componousID = 0
        if materialID1 == materialID2:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio)
        else:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio)
            componousID = self.get_componousID(self.sims.max_material_num, materialID2, materialID1)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio)
        return componousID
    
    def update_property(self, componousID, property_name, value, override):
        factor = 0
        if not override:
            factor = 1

        if property_name == "YoungModulus":
            self.surfaceProps[componousID].YoungModulus = factor * self.surfaceProps[componousID].YoungModulus + value
        elif property_name == "StiffnessRatio":
            self.surfaceProps[componousID].stiffness_ratio = factor * self.surfaceProps[componousID].stiffness_ratio + value
        elif property_name == "Friction":
            self.surfaceProps[componousID].mu = factor * self.surfaceProps[componousID].mu + value
        elif property_name == "NormalViscousDamping":
            self.surfaceProps[componousID].ndratio = factor * self.surfaceProps[componousID].ndratio + value
        elif property_name == "TangentialViscousDamping":
            self.surfaceProps[componousID].sdratio = factor * self.surfaceProps[componousID].sdratio + value
        elif property_name == "ShapeFactor":
            self.surfaceProps[componousID].shape_factor = factor * self.surfaceProps[componousID].shape_factor + value
        elif property_name == "CrushFactor":
            self.surfaceProps[componousID].crush_factor = factor * self.surfaceProps[componousID].crush_factor + value
   