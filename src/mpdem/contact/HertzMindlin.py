import taichi as ti
import numpy as np

from src.physics_model.contact_model.HertzMindlinModel import HertzMindlinSurfaceProperty
from src.dem.Simulation import Simulation as DEMSimulation
from src.dem.SceneManager import SceneManager as DEMScene
from src.dem.contact.ContactKernel import *
from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpm.SceneManager import SceneManager as MPMScene
from src.utils.constants import PI
from src.utils.ObjectIO import DictIO


class HertzMindlinModel(ContactModelBase):
    def __init__(self, max_material_num) -> None:
        super().__init__()
        self.surfaceProps = HertzMindlinSurfaceProperty.field(shape=max_material_num * max_material_num)
        self.null_model = False
        self.model_type = 1

    def calcu_critical_timestep(self, mscene: MPMScene, dsims: DEMSimulation, dscene: DEMScene, max_material_num):
        radius = min(mscene.find_particle_min_radius(), dscene.find_particle_min_radius(dsims))
        density = min(mscene.find_min_density(), dscene.find_min_density())
        modulus, poisson = self.find_max_mparas(max_material_num)
        return PI * radius * ti.sqrt(density / modulus) / (0.1631 * poisson + 0.8766)

    def find_max_mparas(self, max_material_num):
        maxmodulus, maxpoisson = 0.0, 0.0
        for material_id1 in range(max_material_num):
            for material_id2 in range(max_material_num):
                componous_id = self.get_componous_id(max_material_num, material_id1, material_id2)
                if self.surfaceProps[componous_id].ShearModulus > 0.0:
                    poisson = (
                        4 * self.surfaceProps[componous_id].ShearModulus - self.surfaceProps[componous_id].YoungModulus
                    ) / (
                        2 * self.surfaceProps[componous_id].ShearModulus - self.surfaceProps[componous_id].YoungModulus
                    )
                    modulus = 2 * self.surfaceProps[componous_id].ShearModulus * (2 - poisson)
                    maxpoisson = ti.max(maxpoisson, poisson)
                    maxmodulus = ti.max(maxpoisson, modulus)
        return maxmodulus, maxpoisson

    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        modulus = DictIO.get_essential(property, "ShearModulus")
        poisson = DictIO.get_essential(property, "Poisson")
        shear_modulus = 0.5 * modulus / (2.0 - poisson)
        young_modulus = (4.0 * shear_modulus - 2.0 * shear_modulus * poisson) / (1.0 - poisson)
        mus = DictIO.get_essential(property, "StaticFriction", "Friction")
        mud = DictIO.get_essential(property, "DynamicFriction", "Friction")
        rmu = DictIO.get_alternative(property, "RollingFriction", 0.0)
        restitution = DictIO.get_essential(property, "Restitution")
        componous_id = 0
        if restitution < 1e-16:
            restitution = 0.0
        else:
            restitution = -ti.log(restitution) / ti.sqrt(PI * PI + ti.log(restitution) * ti.log(restitution))
        if materialID1 == materialID2:
            componous_id = self.get_componous_id(max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(
                young_modulus, shear_modulus, mus, mud, rmu, restitution
            )
        else:
            componous_id = self.get_componous_id(max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(
                young_modulus, shear_modulus, mus, mud, rmu, restitution
            )
            componous_id = self.get_componous_id(max_material_num, materialID2, materialID1)
            self.surfaceProps[componous_id].add_surface_property(
                young_modulus, shear_modulus, mus, mud, rmu, restitution
            )
        return componous_id
