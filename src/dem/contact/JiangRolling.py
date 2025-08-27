import taichi as ti
import numpy as np

from src.dem.contact.ContactKernel import *
from src.dem.structs.BaseStruct import (HistoryRollingContactTable, HistoryRollingISContactTable, RollingContactTable, RollingISContactTable, DigitalRollingContactTable)
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.physics_model.contact_model.RollingModel import JiangRollingSurfaceProperty
from src.utils.ObjectIO import DictIO
from src.utils.linalg import round32
from src.utils.TypeDefination import u1


# Refers to Jiang et. al (2015) A novel three-dimensional contact model for granulates incorporating rolling and twisting resistances. Computer and Geotechnics
class JiangRollingResistanceModel(ContactModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.surfaceProps = JiangRollingSurfaceProperty.field(shape=self.sims.max_material_num * self.sims.max_material_num)
        self.null_model = False
        self.model_type = 2

    def calcu_critical_timestep(self, scene: myScene):
        mass = scene.find_particle_min_mass(self.sims.scheme)
        radius = scene.find_particle_min_radius(self.sims.scheme)
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
    
    def collision_initialize(self, object_type, work_type, max_object_pairs, object_num1, object_num2):
        if not self.null_model and self.first_run:
            if object_type == 'particle' or object_type == 'wall':
                if self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics" and object_type == 'particle':
                    self.cplist = RollingISContactTable.field(shape=max_object_pairs)
                else:
                    self.cplist = RollingContactTable.field(shape=max_object_pairs)
                if work_type == 0 or work_type == 1:
                    self.deactivate_exist = ti.field(ti.u8, shape=())
                    self.contact_active = ti.field(u1)
                    ti.root.dense(ti.i, round32(object_num1 * object_num2)//32).quant_array(ti.i, dimensions=32, max_num_bits=32).place(self.contact_active)
                if self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics" and object_type == 'particle':
                    self.hist_cplist = HistoryRollingISContactTable.field(shape=max_object_pairs)
                else:
                    self.hist_cplist = HistoryRollingContactTable.field(shape=max_object_pairs)
            elif object_type == 'wall' and self.sims.wall_type == 3:
                if self.sims.scheme == "DEM":
                    self.cplist = DigitalRollingContactTable.field(shape=int(self.sims.max_particle_num))
                elif self.sims.scheme == "LSDEM":
                    self.cplist = DigitalRollingContactTable.field(shape=int(self.sims.max_surface_node_num * self.sims.max_rigid_body_num))
        self.first_run = False

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
    
    def get_contact_output(self, scene: myScene, neighbor_list):
        end1 = np.ascontiguousarray(self.cplist.endID1.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        end2 = np.ascontiguousarray(self.cplist.endID2.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        normal_force = np.ascontiguousarray(self.cplist.cnforce.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        tangential_force = np.ascontiguousarray(self.cplist.csforce.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        oldTangentialOverlap = np.ascontiguousarray(self.cplist.oldTangOverlap.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        oldRollAngle = np.ascontiguousarray(self.cplist.oldRollAngle.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        oldTwistAngle = np.ascontiguousarray(self.cplist.oldTwistAngle.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        return end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle
    
    def get_ppcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        particleParticle = np.ascontiguousarray(pcontact.hist_particle_particle.to_numpy()[0:scene.particleNum[0] + 1])
        end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle = self.get_contact_output(scene, particleParticle)
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleParticle, end1=end1, end2=end2, normal_force=normal_force, 
                                                      tangential_force=tangential_force, oldTangentialOverlap=oldTangentialOverlap, oldRollAngle=oldRollAngle, oldTwistAngle=oldTwistAngle)
        
    def get_pwcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        particleWall = np.ascontiguousarray(pcontact.hist_particle_wall.to_numpy()[0:scene.particleNum[0] + 1])
        end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle = self.get_contact_output(scene, particleWall)
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleWall, end1=end1, end2=end2, normal_force=normal_force, 
                                                      tangential_force=tangential_force, oldTangentialOverlap=oldTangentialOverlap, oldRollAngle=oldRollAngle, oldTwistAngle=oldTwistAngle)
        
    def rebuild_contact_list(self, contact_info):
        object_object = DictIO.GetEssential(contact_info, "contact_num")
        DstID = DictIO.GetEssential(contact_info, "end2")
        normal_force = DictIO.GetEssential(contact_info, "normal_force")
        tangential_force = DictIO.GetEssential(contact_info, "tangential_force")
        oldTangOverlap = DictIO.GetEssential(contact_info, "oldTangentialOverlap")
        oldRollAngle = DictIO.GetEssential(contact_info, "oldRollAngle")
        oldTwistAngle = DictIO.GetEssential(contact_info, "oldTwistAngle")
        return object_object, DstID, normal_force, tangential_force, oldTangOverlap, oldRollAngle, oldTwistAngle
    
    def rebuild_ppcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, DstID, normal_force, tangential_force, oldTangOverlap, oldRollAngle, oldTwistAngle = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_addition_history_contact_list(self.cplist, pcontact.hist_particle_particle, object_object, DstID, normal_force, tangential_force, oldTangOverlap, oldRollAngle, oldTwistAngle)

    def rebuild_pwcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, DstID, normal_force, tangential_force, oldTangOverlap, oldRollAngle, oldTwistAngle = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_addition_history_contact_list(self.cplist, pcontact.hist_particle_wall, object_object, DstID, normal_force, tangential_force, oldTangOverlap, oldRollAngle, oldTwistAngle)
    
    # ========================================================= #
    #              Particle Contact Matrix Resolve              #
    # ========================================================= # 
    def update_particle_particle_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_addition_contact_table(pcontact.hist_particle_particle, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
        self.update_ppcontact_table(sims, scene, pcontact)
        kernel_inherit_rolling_history(int(scene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_particle, pcontact.hist_particle_particle)

    def update_particle_wall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_addition_contact_table(pcontact.hist_particle_wall, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
        self.update_pwcontact_table(sims, scene, pcontact)
        kernel_inherit_rolling_history(int(scene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_wall, pcontact.hist_particle_wall)
