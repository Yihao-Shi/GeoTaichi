import taichi as ti
import numpy as np
import math

from src.dem.contact.ContactKernel import *
from src.dem.structs.BaseStruct import (HistoryRollingContactTable, HistoryRollingISContactTable, RollingContactTable, RollingISContactTable, DigitalRollingContactTable)
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.FieldIO import field_to_numpy_prefix
from src.utils.linalg import round32
from src.utils.TypeDefination import u1


class RollingModelBase(ContactModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.null_model = False
        self.model_type = 2
    
    def collision_initialize(self, object_type, work_type, max_object_pairs, object_num1, object_num2):
        if not self.null_model and self.first_run:
            if object_type == 'wall' and self.sims.use_digital_elevation_heightfield():
                if self.sims.scheme == "DEM":
                    self.cplist = DigitalRollingContactTable.field(shape=int(self.sims.max_particle_num))
                else:
                    raise RuntimeError(f"DigitalElevation heightfield rolling contact does not support DEM scheme {self.sims.scheme}")
            elif object_type == 'particle' or object_type == 'wall':
                if (self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics") and object_type == 'particle':
                    self.cplist = RollingISContactTable.field(shape=max_object_pairs)
                else:
                    self.cplist = RollingContactTable.field(shape=max_object_pairs)

                if work_type == 0 or work_type == 1:
                    self.deactivate_exist = ti.field(ti.u8, shape=())
                    self.contact_active = ti.field(u1)
                    ti.root.dense(ti.i, round32(object_num1 * object_num2)//32).quant_array(ti.i, dimensions=32, max_num_bits=32).place(self.contact_active)
                if (self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics") and object_type == 'particle':
                    self.hist_cplist = HistoryRollingISContactTable.field(shape=max_object_pairs)
                else:
                    self.hist_cplist = HistoryRollingContactTable.field(shape=max_object_pairs)
            elif object_type == 'wall' and self.sims.wall_type == 3:
                if self.sims.scheme == "DEM":
                    self.cplist = DigitalRollingContactTable.field(shape=int(self.sims.max_particle_num))
                elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                    self.cplist = DigitalRollingContactTable.field(shape=int(self.sims.max_ls_contact_node_num))
        self.first_run = False

    def get_contact_output(self, scene: myScene, neighbor_list, total_len):
        contact_count = int(neighbor_list[total_len])
        end1 = field_to_numpy_prefix(self.cplist.endID1, contact_count)
        end2 = field_to_numpy_prefix(self.cplist.endID2, contact_count)
        normal_force = field_to_numpy_prefix(self.cplist.cnforce, contact_count)
        tangential_force = field_to_numpy_prefix(self.cplist.csforce, contact_count)
        oldTangentialOverlap = field_to_numpy_prefix(self.cplist.oldTangOverlap, contact_count)
        oldRollAngle = field_to_numpy_prefix(self.cplist.oldRollAngle, contact_count)
        oldTwistAngle = field_to_numpy_prefix(self.cplist.oldTwistAngle, contact_count)
        return end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle
    
    def get_ppcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        output = self.get_contact_energy_output()
        if self.sims.scheme == "DEM":
            particleParticle = field_to_numpy_prefix(pcontact.hist_particle_particle, scene.particleNum[0] + 1)
            end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle = self.get_contact_output(scene, particleParticle, scene.particleNum[0])
            output.update({"t_current": current_time, "contact_num": particleParticle, "end1": end1, "end2": end2, "normal_force": normal_force, 
                        "tangential_force": tangential_force, "oldTangentialOverlap": oldTangentialOverlap, "oldRollAngle": oldRollAngle, "oldTwistAngle": oldTwistAngle})
            np.savez(contact_path+f'{current_print:06d}', **output)
        elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
            contact_node_num = self.get_ls_contact_node_num(scene)
            particleParticle = field_to_numpy_prefix(pcontact.hist_lsparticle_lsparticle, contact_node_num + 1)
            end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle = self.get_contact_output(scene, particleParticle, contact_node_num)
            output.update({"t_current": current_time, "contact_num": particleParticle, "end1": end1, "end2": end2, "normal_force": normal_force, 
                           "tangential_force": tangential_force, "oldTangentialOverlap": oldTangentialOverlap, "oldRollAngle": oldRollAngle, "oldTwistAngle": oldTwistAngle})
            np.savez(contact_path+f'{current_print:06d}', **output)
        
    def get_pwcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        output = self.get_contact_energy_output()
        if self.sims.scheme == "DEM":
            particleWall = field_to_numpy_prefix(pcontact.hist_particle_wall, scene.particleNum[0] + 1)
            end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle = self.get_contact_output(scene, particleWall, scene.particleNum[0])
            output.update({"t_current": current_time, "contact_num": particleWall, "end1": end1, "end2": end2, "normal_force": normal_force, 
                        "tangential_force": tangential_force, "oldTangentialOverlap": oldTangentialOverlap, "oldRollAngle": oldRollAngle, "oldTwistAngle": oldTwistAngle})
            np.savez(contact_path+f'{current_print:06d}', **output)
        elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
            contact_node_num = self.get_ls_contact_node_num(scene)
            particleWall = field_to_numpy_prefix(pcontact.hist_lsparticle_wall, contact_node_num + 1)
            end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle = self.get_contact_output(scene, particleWall, contact_node_num)
            output.update({"t_current": current_time, "contact_num": particleWall, "end1": end1, "end2": end2, "normal_force": normal_force, 
                           "tangential_force": tangential_force, "oldTangentialOverlap": oldTangentialOverlap, "oldRollAngle": oldRollAngle, "oldTwistAngle": oldTwistAngle})
            np.savez(contact_path+f'{current_print:06d}', **output)
        
    def rebuild_contact_list(self, contact_info):
        object_object = DictIO.GetEssential(contact_info, "contact_num")
        DstID = DictIO.GetEssential(contact_info, "end2")
        normal_force = DictIO.GetEssential(contact_info, "normal_force")
        tangential_force = DictIO.GetEssential(contact_info, "tangential_force")
        oldTangOverlap = DictIO.GetEssential(contact_info, "oldTangentialOverlap")
        oldRollAngle = DictIO.GetAlternative(contact_info, "oldRollAngle", np.zeros_like(oldTangOverlap))
        oldTwistAngle = DictIO.GetAlternative(contact_info, "oldTwistAngle", np.zeros_like(oldTangOverlap))
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
