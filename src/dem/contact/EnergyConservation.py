import taichi as ti

from src.dem.contact.ContactKernel import *
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.SceneManager import SceneManager
from src.physics_model.contact_model.EnergyConservingModel import PenaltyProperty
from src.physics_model.contact_model.BarrierModel import BarrierProperty
from src.utils.constants import ZEROVEC3F
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

    def calcu_critical_timestep(self, scene: SceneManager):
        mass = scene.find_particle_min_mass(self.sims)
        stiffness = self.find_max_stiffness(scene)
        return ti.sqrt(mass / stiffness)

    def find_max_stiffness(self, scene: SceneManager):
        maxstiff = 0.0
        if self.types == 1:
            if self.sims.scheme == "LSMPM":
                penetration_bound = max(
                    float(self.sims.point_verlet_distance),
                    float(scene.find_min_grid_space(self.sims)),
                )
                gradient_bound = 1.0 + max(float(self.sims.soft_levelset_reinit_grad_threshold), 0.0)
                maxstiff = kernel_find_max_penalty_stiffness_lsmpm(
                    self.sims.max_material_num,
                    int(scene.surfaceNum[0]),
                    penetration_bound,
                    gradient_bound,
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
                        if self.surfaceProps[componous_id].kn > 0.0:
                            maxstiff = ti.max(
                                ti.max(
                                    maxstiff,
                                    self.surfaceProps[componous_id].kn,
                                ),
                                self.surfaceProps[componous_id].ks,
                            )
        elif self.types == 2:
            if self.sims.scheme == "LSMPM":
                gradient_bound = 1.0 + max(float(self.sims.soft_levelset_reinit_grad_threshold), 0.0)
                maxstiff = kernel_find_max_barrier_stiffness_lsmpm(
                    self.sims.max_material_num,
                    int(scene.surfaceNum[0]),
                    gradient_bound,
                    scene.rigid,
                    scene.surface,
                    scene.vertice,
                    scene.box,
                    self.surfaceProps,
                )
            else:
                for material_id1 in range(self.sims.max_material_num):
                    for material_id2 in range(self.sims.max_material_num):
                        componous_id = self.get_componous_id(self.sims.max_material_num, material_id1, material_id2)
                        if self.surfaceProps[componous_id].kappa > 0.0:
                            kappa = self.surfaceProps[componous_id].kappa
                            ratio = kernel_get_min_ratio(
                                componous_id, int(scene.particleNum[0]), self.surfaceProps, scene.rigid
                            )
                            kn = -kappa * (2.0 * ti.log(ratio) + ((ratio - 1) * (3 * ratio + 1)) / ratio**2)
                            maxstiff = ti.max(maxstiff, kn)
        return maxstiff

    def find_max_penetration(self):
        max_penetration = 0.0
        if self.types == 2:
            for material_id1 in range(self.sims.max_material_num):
                for material_id2 in range(self.sims.max_material_num):
                    componous_id = self.get_componous_id(self.sims.max_material_num, material_id1, material_id2)
                    if self.surfaceProps[componous_id].kappa > 0.0:
                        max_penetration = ti.max(max_penetration, self.surfaceProps[componous_id].ncut)
        return max_penetration

    def add_surface_property(self, materialID1, materialID2, property):
        if self.types == 1:
            kn = DictIO.get_essential(property, "NormalStiffness")
            ks = DictIO.get_essential(property, "TangentialStiffness")
            mu = DictIO.get_essential(property, "Friction")
            theta = DictIO.get_alternative(property, "FreeParameter", 2.5)
            if theta < 2.0:
                raise ValueError(
                    "Energy Conserving Model requires FreeParameter >= 2 "
                    "for a finite explicit-contact tangent at zero gap"
                )
            ndratio = DictIO.get_alternative(property, "NormalViscousDamping", 0.0)
            sdratio = DictIO.get_alternative(property, "TangentialViscousDamping", 0.0)
            componous_id = 0
            if materialID1 == materialID2:
                componous_id = self.get_componous_id(self.sims.max_material_num, materialID1, materialID2)
                self.surfaceProps[componous_id].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
            else:
                componous_id = self.get_componous_id(self.sims.max_material_num, materialID1, materialID2)
                self.surfaceProps[componous_id].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
                componous_id = self.get_componous_id(self.sims.max_material_num, materialID2, materialID1)
                self.surfaceProps[componous_id].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
            return componous_id
        elif self.types == 2:
            kappa = DictIO.get_essential(property, "Stiffness")
            ncut = DictIO.get_essential(property, "NormalCutOff")
            ratio = DictIO.get_alternative(property, "StiffnessRatio", 1.0)
            mu = DictIO.get_essential(property, "Friction")
            ndratio = DictIO.get_alternative(property, "NormalViscousDamping", 0.0)
            sdratio = DictIO.get_alternative(property, "TangentialViscousDamping", 0.0)
            componous_id = 0
            if materialID1 == materialID2:
                componous_id = self.get_componous_id(self.sims.max_material_num, materialID1, materialID2)
                self.surfaceProps[componous_id].add_surface_property(kappa, ncut, ratio, mu, ndratio, sdratio)
            else:
                componous_id = self.get_componous_id(self.sims.max_material_num, materialID1, materialID2)
                self.surfaceProps[componous_id].add_surface_property(kappa, ncut, ratio, mu, ndratio, sdratio)
                componous_id = self.get_componous_id(self.sims.max_material_num, materialID2, materialID1)
                self.surfaceProps[componous_id].add_surface_property(kappa, ncut, ratio, mu, ndratio, sdratio)
            return componous_id

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
