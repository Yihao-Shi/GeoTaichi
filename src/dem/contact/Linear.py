from src.utils.errors import SimulationStateError
import taichi as ti
import math

from src.dem.contact.ContactKernel import *
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.SceneManager import SceneManager
from src.physics_model.contact_model.LinearModel import LinearSurfaceProperty
from src.utils.ObjectIO import DictIO
import src.utils.GlobalVariable as GlobalVariable


class LinearModel(ContactModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.surfaceProps = LinearSurfaceProperty.field(shape=self.sims.max_material_num * self.sims.max_material_num)
        self.wall_elastic_energy_scratch = ti.field(dtype=ti.lang.impl.current_cfg().default_fp, shape=())
        self.null_model = False
        self.model_type = 1

    def lsparticle_wall_elastic_energy(self, scene, neighbor, wall_id):
        """Return current elastic energy supported by one facet-wall group."""

        if self.sims.scheme != "LSDEM":
            raise SimulationStateError("wall-group elastic energy is currently available for LSDEM")
        kernel_accumulate_linear_lsparticle_wall_elastic_energy(
            int(scene.surfaceNum[0]),
            int(wall_id),
            self.sims.max_material_num,
            self.surfaceProps,
            scene.rigid,
            scene.vertice,
            scene.surface,
            scene.box,
            scene.wall,
            self.cplist,
            neighbor.hist_lsparticle_wall,
            self.wall_elastic_energy_scratch,
        )
        return float(self.wall_elastic_energy_scratch[None])

    def calcu_critical_timestep(self, scene: SceneManager):
        mass = scene.find_particle_min_mass(self.sims)
        stiffness = self.find_max_stiffness(scene)
        return ti.sqrt(mass / stiffness)

    def find_max_stiffness(self, scene: SceneManager):
        maxstiff = 0.0
        radius = scene.find_particle_max_radius(self.sims)
        if self.sims.scheme == "LSMPM":
            maxstiff = kernel_find_max_stiffness_lsmpm(
                self.sims.max_material_num,
                int(scene.surfaceNum[0]),
                scene.rigid,
                scene.surface,
                scene.vertice,
                scene.box,
                self.surfaceProps,
            )
        elif self.sims.scheme == "LSDEM":
            maxstiff = kernel_find_max_stiffness(
                self.sims.max_material_num,
                int(scene.surfaceNum[0]),
                scene.rigid,
                scene.surface,
                scene.vertice,
                self.surfaceProps,
            )
        else:
            for material_id1 in range(self.sims.max_material_num):
                for material_id2 in range(self.sims.max_material_num):
                    componous_id = self.get_componous_id(self.sims.max_material_num, material_id1, material_id2)
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

    def add_surface_property(self, materialID1, materialID2, property):
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
        ndratio = DictIO.get_alternative(property, "NormalViscousDamping", 0)
        sdratio = DictIO.get_alternative(property, "TangentialViscousDamping", 0)
        if rmu > 0.0:
            GlobalVariable.CONSTANTORQUEMODEL = True
        componous_id = 0
        if materialID1 == materialID2:
            componous_id = self.get_componous_id(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(kn, ks, emod, kratio, mus, mud, rmu, ndratio, sdratio)
        else:
            componous_id = self.get_componous_id(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componous_id].add_surface_property(kn, ks, emod, kratio, mus, mud, rmu, ndratio, sdratio)
            componous_id = self.get_componous_id(self.sims.max_material_num, materialID2, materialID1)
            self.surfaceProps[componous_id].add_surface_property(kn, ks, emod, kratio, mus, mud, rmu, ndratio, sdratio)
        return componous_id

    def update_property(self, componousID, property_name, value, override):
        factor = 0
        if not override:
            factor = 1

        if property_name == "NormalStiffness":
            self.surfaceProps[componousID].kn = factor * self.surfaceProps[componousID].kn + value
        elif property_name == "TangentialStiffness":
            self.surfaceProps[componousID].ks = factor * self.surfaceProps[componousID].ks + value
        elif property_name == "StaticFriction" or property_name == "Friction":
            self.surfaceProps[componousID].mus = factor * self.surfaceProps[componousID].mus + value
        elif property_name == "DynamicFriction" or property_name == "Friction":
            self.surfaceProps[componousID].mud = factor * self.surfaceProps[componousID].mud + value
        elif property_name == "NormalViscousDamping":
            self.surfaceProps[componousID].ndratio = factor * self.surfaceProps[componousID].ndratio + value
        elif property_name == "TangentialViscousDamping":
            self.surfaceProps[componousID].sdratio = factor * self.surfaceProps[componousID].sdratio + value
