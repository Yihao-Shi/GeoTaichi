import taichi as ti

from src.dem.contact.ContactKernel import *
from src.dem.contact.RollingModelBase import RollingModelBase
from src.dem.SceneManager import SceneManager
from src.physics_model.contact_model.RollingModel import JiangRollingSurfaceProperty
from src.utils.ObjectIO import DictIO


# Refers to Jiang et. al (2015) A novel three-dimensional contact model for granulates incorporating rolling and twisting resistances. Computer and Geotechnics
class JiangRollingResistanceModel(RollingModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.surfaceProps = JiangRollingSurfaceProperty.field(
            shape=self.sims.max_material_num * self.sims.max_material_num
        )

    def calcu_critical_timestep(self, scene: SceneManager):
        mass = scene.find_particle_min_mass(self.sims)
        radius = scene.find_particle_min_radius(self.sims)
        modulus = self._find_max_modulus()
        stiffness = 2 * radius * modulus
        return ti.sqrt(mass / stiffness)

    def _find_max_modulus(self):
        maxmodulus = 0.0
        for material_id1 in range(self.sims.max_material_num):
            for material_id2 in range(self.sims.max_material_num):
                componous_id = self.get_componous_id(self.sims.max_material_num, material_id1, material_id2)
                if self.surfaceProps[componous_id].YoungModulus > 0.0:
                    maxmodulus = ti.max(maxmodulus, self.surfaceProps[componous_id].YoungModulus)
        return maxmodulus

    def add_surface_property(self, materialID1, materialID2, property):
        young_modulus = DictIO.get_essential(property, "YoungModulus")
        stiffness_ratio = DictIO.get_essential(property, "StiffnessRatio")
        mu = DictIO.get_essential(property, "Friction")
        shape_factor = DictIO.get_essential(property, "ShapeFactor")
        crush_factor = DictIO.get_essential(property, "CrushFactor")
        ndratio = DictIO.get_essential(property, "NormalViscousDamping")
        sdratio = DictIO.get_essential(property, "TangentialViscousDamping")
        componous_id = 0
        if materialID1 == materialID2:
            componous_id = self.get_componous_id(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(
                young_modulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio
            )
        else:
            componous_id = self.get_componous_id(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(
                young_modulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio
            )
            componous_id = self.get_componous_id(self.sims.max_material_num, materialID2, materialID1)
            self.surfaceProps[componous_id].add_surface_property(
                young_modulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio
            )
        return componous_id

    def update_property(self, componousID, property_name, value, override):
        factor = 0
        if not override:
            factor = 1

        if property_name == "YoungModulus":
            self.surfaceProps[componousID].YoungModulus = factor * self.surfaceProps[componousID].YoungModulus + value
        elif property_name == "StiffnessRatio":
            self.surfaceProps[componousID].stiffness_ratio = (
                factor * self.surfaceProps[componousID].stiffness_ratio + value
            )
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
