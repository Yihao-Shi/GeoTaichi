import taichi as ti

from src.dem.contact.ContactKernel import *
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.SceneManager import myScene
from src.physics_model.contact_model.HertzMindlinModel import HertzMindlinSurfaceProperty
from src.utils.constants import PI
from src.utils.ObjectIO import DictIO
import src.utils.GlobalVariable as GlobalVariable


class HertzMindlinModel(ContactModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.surfaceProps = HertzMindlinSurfaceProperty.field(shape=self.sims.max_material_num * self.sims.max_material_num)
        self.null_model = False
        self.model_type = 1

    def calcu_critical_timestep(self, scene: myScene):
        radius = scene.find_particle_min_radius(self.sims.scheme)
        density = scene.find_min_density()
        modulus, Poisson = self.find_max_mparas()
        return PI * radius * ti.sqrt(density / modulus) / (0.1631 * Poisson + 0.8766)

    def find_max_mparas(self):
        maxmodulus, maxpoisson = 0., 0.
        for materialID1 in range(self.sims.max_material_num):
            for materialID2 in range(self.sims.max_material_num):
                componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                if self.surfaceProps[componousID].ShearModulus > 0.:
                    Poisson = (4 * self.surfaceProps[componousID].ShearModulus - self.surfaceProps[componousID].YoungModulus) / \
                              (2 * self.surfaceProps[componousID].ShearModulus - self.surfaceProps[componousID].YoungModulus)
                    modulus = 2 * self.surfaceProps[componousID].ShearModulus * (2 - Poisson)
                    maxpoisson = ti.max(maxpoisson, Poisson)
                    maxmodulus = ti.max(maxpoisson, modulus)
        return maxmodulus, maxpoisson
    
    def add_surface_property(self, materialID1, materialID2, property):
        modulus = DictIO.GetEssential(property, 'ShearModulus')
        poisson = DictIO.GetEssential(property, 'Poisson')
        ShearModulus = 0.5 * modulus / (2. - poisson)
        YoungModulus = (4. * ShearModulus - 2. * ShearModulus * poisson) / (1. - poisson)
        mus = DictIO.GetEssential(property, 'StaticFriction', 'Friction')
        mud = DictIO.GetEssential(property, 'DynamicFriction', 'Friction')
        rmu = DictIO.GetAlternative(property, 'RollingFriction', 0.)
        restitution = DictIO.GetAlternative(property, 'Restitution', 1.)
        if rmu > 0.:
            GlobalVariable.CONSTANTORQUEMODEL = True
        componousID = 0
        if restitution < 1e-16:
            restitution = 0.
        else:
            restitution = -ti.log(restitution) / ti.sqrt(PI * PI + ti.log(restitution) * ti.log(restitution))
        if materialID1 == materialID2:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mus, mud, rmu, restitution)
        else:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mus, mud, rmu, restitution)
            componousID = self.get_componousID(self.sims.max_material_num, materialID2, materialID1)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mus, mud, rmu, restitution)
        return componousID

    
    def update_property(self, componousID, property_name, value, override):
        factor = 0
        if not override:
            factor = 1

        if property_name == "ShearModulus":
            E, G = self.surfaceProps[componousID].YoungModulus, self.surfaceProps[componousID].ShearModulus
            poisson = (4. * G - E) / (2. * G - E)
            modulus = 2. * G * (2. - poisson)
            modulus = factor * modulus + value
            ShearModulus = 0.5 * modulus / (2. - poisson)
            YoungModulus = (4. * ShearModulus - 2. * ShearModulus * poisson) / (1. - poisson)
            self.surfaceProps[componousID].YoungModulus = YoungModulus
            self.surfaceProps[componousID].ShearModulus = ShearModulus
        elif property_name == "Poisson":
            E, G = self.surfaceProps[componousID].YoungModulus, self.surfaceProps[componousID].ShearModulus
            poisson = (4. * G - E) / (2. * G - E)
            modulus = 2. * G * (2. - poisson)
            poisson = factor * poisson + value
            ShearModulus = 0.5 * modulus / (2. - poisson)
            YoungModulus = (4. * ShearModulus - 2. * ShearModulus * poisson) / (1. - poisson)
            self.surfaceProps[componousID].YoungModulus = YoungModulus
            self.surfaceProps[componousID].ShearModulus = ShearModulus
        elif property_name == "Friction":
            self.surfaceProps[componousID].mu = factor * self.surfaceProps[componousID].mu + value
        elif property_name == "Restitution":
            self.surfaceProps[componousID].restitution = factor * self.surfaceProps[componousID].restitution + value