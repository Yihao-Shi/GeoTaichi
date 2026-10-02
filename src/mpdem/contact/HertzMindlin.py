import taichi as ti
import numpy as np

from src.physics_model.contact_model.HertzMindlinModel import HertzMindlinSurfaceProperty
from src.dem.Simulation import Simulation as DEMSimulation
from src.dem.SceneManager import myScene as DEMScene
from src.dem.contact.ContactKernel import *
from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpm.SceneManager import myScene as MPMScene
from src.utils.constants import PI
from src.utils.ObjectIO import DictIO


class HertzMindlinModel(ContactModelBase):
    def __init__(self, max_material_num) -> None:
        super().__init__()
        self.surfaceProps = HertzMindlinSurfaceProperty.field(shape=max_material_num * max_material_num)
        self.null_model = False
        self.model_type = 1

    def calcu_critical_timestep(self, mscene: MPMScene, dsims:DEMSimulation, dscene: DEMScene, max_material_num):
        radius = min(mscene.find_particle_min_radius(), dscene.find_particle_min_radius(dsims.scheme))
        density = min(mscene.find_min_density(), dscene.find_min_density())
        modulus, poisson = self.find_max_mparas(max_material_num)
        return PI * radius * ti.sqrt(density / modulus) / (0.1631 * poisson + 0.8766)

    def find_max_mparas(self, max_material_num):
        maxmodulus, maxpoisson = 0., 0.
        for materialID1 in range(max_material_num):
            for materialID2 in range(max_material_num):
                componousID = self.get_componousID(max_material_num, materialID1, materialID2)
                if self.surfaceProps[componousID].ShearModulus > 0.:
                    poisson = (4 * self.surfaceProps[componousID].ShearModulus - self.surfaceProps[componousID].YoungModulus) / \
                              (2 * self.surfaceProps[componousID].ShearModulus - self.surfaceProps[componousID].YoungModulus)
                    modulus = 2 * self.surfaceProps[componousID].ShearModulus * (2 - poisson)
                    maxpoisson = ti.max(maxpoisson, poisson)
                    maxmodulus = ti.max(maxpoisson, modulus)
        return maxmodulus, maxpoisson
    
    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        modulus = DictIO.GetEssential(property, 'ShearModulus')
        poisson = DictIO.GetEssential(property, 'Poisson')
        ShearModulus = 0.5 * modulus / (2. - poisson)
        YoungModulus = (4. * ShearModulus - 2. * ShearModulus * poisson) / (1. - poisson)
        mus = DictIO.GetEssential(property, 'StaticFriction', 'Friction')
        mud = DictIO.GetEssential(property, 'DynamicFriction', 'Friction')
        rmu = DictIO.GetAlternative(property, 'RollingFriction', 0.)
        restitution = DictIO.GetEssential(property, 'Restitution')
        componousID = 0
        if restitution < 1e-16:
            restitution = 0.
        else:
            restitution = -ti.log(restitution) / ti.sqrt(PI * PI + ti.log(restitution) * ti.log(restitution))
        if materialID1 == materialID2:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mus, mud, rmu, restitution)
        else:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mus, mud, rmu, restitution)
            componousID = self.get_componousID(max_material_num, materialID2, materialID1)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mus, mud, rmu, restitution)
        return componousID



