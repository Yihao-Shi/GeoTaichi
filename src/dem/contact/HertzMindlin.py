import taichi as ti

from src.dem.contact.ContactKernel import *
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.SceneManager import SceneManager
from src.physics_model.contact_model.HertzMindlinModel import HertzMindlinSurfaceProperty
from src.utils.constants import PI
from src.utils.ObjectIO import DictIO
import src.utils.GlobalVariable as GlobalVariable


class HertzMindlinModel(ContactModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.surfaceProps = HertzMindlinSurfaceProperty.field(
            shape=self.sims.max_material_num * self.sims.max_material_num
        )
        self.null_model = False
        self.model_type = 1

    def calcu_critical_timestep(self, scene: SceneManager):
        radius = scene.find_particle_min_radius(self.sims)
        density = scene.find_min_density()
        modulus, poisson = self.find_max_mparas()
        wave_timestep = PI * radius * ti.sqrt(density / modulus) / (0.1631 * poisson + 0.8766)
        if self.sims.scheme == "LSMPM":
            penetration_bound = max(
                float(self.sims.point_verlet_distance),
                float(scene.find_min_grid_space(self.sims)),
            )
            stiffness = kernel_find_max_hertz_stiffness_lsmpm(
                self.sims.max_material_num,
                int(scene.surfaceNum[0]),
                penetration_bound,
                scene.rigid,
                scene.surface,
                scene.vertice,
                scene.box,
                self.surfaceProps,
            )
            contact_mass = scene.find_particle_min_mass(self.sims)
            quadrature_timestep = ti.sqrt(contact_mass / stiffness)
            return ti.min(wave_timestep, quadrature_timestep)
        return wave_timestep

    def find_max_mparas(self):
        maxmodulus, maxpoisson = 0.0, 0.0
        for material_id1 in range(self.sims.max_material_num):
            for material_id2 in range(self.sims.max_material_num):
                componous_id = self.get_componous_id(self.sims.max_material_num, material_id1, material_id2)
                if self.surfaceProps[componous_id].ShearModulus > 0.0:
                    poisson = (
                        4 * self.surfaceProps[componous_id].ShearModulus - self.surfaceProps[componous_id].YoungModulus
                    ) / (
                        2 * self.surfaceProps[componous_id].ShearModulus - self.surfaceProps[componous_id].YoungModulus
                    )
                    modulus = 2 * self.surfaceProps[componous_id].ShearModulus * (2 - poisson)
                    maxpoisson = ti.max(maxpoisson, poisson)
                    maxmodulus = ti.max(maxmodulus, modulus)
        return maxmodulus, maxpoisson

    def add_surface_property(self, materialID1, materialID2, property):
        modulus = DictIO.get_essential(property, "ShearModulus")
        poisson = DictIO.get_essential(property, "Poisson")
        shear_modulus = 0.5 * modulus / (2.0 - poisson)
        young_modulus = (4.0 * shear_modulus - 2.0 * shear_modulus * poisson) / (1.0 - poisson)
        mus = DictIO.get_essential(property, "StaticFriction", "Friction")
        mud = DictIO.get_essential(property, "DynamicFriction", "Friction")
        rmu = DictIO.get_alternative(property, "RollingFriction", 0.0)
        restitution = DictIO.get_alternative(property, "Restitution", 1.0)
        if rmu > 0.0:
            GlobalVariable.CONSTANTORQUEMODEL = True
        componous_id = 0
        if restitution < 1e-16:
            restitution = 0.0
        else:
            restitution = -ti.log(restitution) / ti.sqrt(PI * PI + ti.log(restitution) * ti.log(restitution))
        if materialID1 == materialID2:
            componous_id = self.get_componous_id(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(
                young_modulus, shear_modulus, mus, mud, rmu, restitution
            )
        else:
            componous_id = self.get_componous_id(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(
                young_modulus, shear_modulus, mus, mud, rmu, restitution
            )
            componous_id = self.get_componous_id(self.sims.max_material_num, materialID2, materialID1)
            self.surfaceProps[componous_id].add_surface_property(
                young_modulus, shear_modulus, mus, mud, rmu, restitution
            )
        return componous_id

    def update_property(self, componousID, property_name, value, override):
        factor = 0
        if not override:
            factor = 1

        if property_name == "ShearModulus":
            E, G = self.surfaceProps[componousID].YoungModulus, self.surfaceProps[componousID].ShearModulus
            poisson = (4.0 * G - E) / (2.0 * G - E)
            modulus = 2.0 * G * (2.0 - poisson)
            modulus = factor * modulus + value
            shear_modulus = 0.5 * modulus / (2.0 - poisson)
            young_modulus = (4.0 * shear_modulus - 2.0 * shear_modulus * poisson) / (1.0 - poisson)
            self.surfaceProps[componousID].YoungModulus = young_modulus
            self.surfaceProps[componousID].ShearModulus = shear_modulus
        elif property_name == "Poisson":
            E, G = self.surfaceProps[componousID].YoungModulus, self.surfaceProps[componousID].ShearModulus
            poisson = (4.0 * G - E) / (2.0 * G - E)
            modulus = 2.0 * G * (2.0 - poisson)
            poisson = factor * poisson + value
            shear_modulus = 0.5 * modulus / (2.0 - poisson)
            young_modulus = (4.0 * shear_modulus - 2.0 * shear_modulus * poisson) / (1.0 - poisson)
            self.surfaceProps[componousID].YoungModulus = young_modulus
            self.surfaceProps[componousID].ShearModulus = shear_modulus
        elif property_name == "Friction":
            self.surfaceProps[componousID].mu = factor * self.surfaceProps[componousID].mu + value
        elif property_name == "Restitution":
            self.surfaceProps[componousID].restitution = factor * self.surfaceProps[componousID].restitution + value
