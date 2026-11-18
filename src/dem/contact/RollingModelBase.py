from src.utils.errors import SimulationStateError
import taichi as ti
import numpy as np
import math

from src.dem.contact.ContactKernel import *
from src.dem.structs.BaseStruct import (
    HistoryRollingContactTable,
    HistoryRollingISContactTable,
    RollingContactTable,
    RollingISContactTable,
    DigitalRollingContactTable,
)
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.SceneManager import SceneManager
from src.dem.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.FieldIO import field_to_numpy_prefix
from src.utils.linalg import round32
from src.utils.TypeDefinition import u1


class RollingModelBase(ContactModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.null_model = False
        self.model_type = 2

    def collision_initialize(self, object_type, work_type, max_object_pairs, object_num1, object_num2):
        if not self.null_model and self.first_run:
            if object_type == "wall" and self.sims.use_digital_elevation_heightfield():
                if self.sims.scheme == "DEM":
                    self.cplist = DigitalRollingContactTable.field(shape=int(self.sims.max_particle_num))
                else:
                    raise SimulationStateError(
                        f"DigitalElevation heightfield rolling contact does not support DEM scheme {self.sims.scheme}"
                    )
            elif object_type == "particle" or object_type == "wall":
                if (
                    self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics"
                ) and object_type == "particle":
                    self.cplist = RollingISContactTable.field(shape=max_object_pairs)
                else:
                    self.cplist = RollingContactTable.field(shape=max_object_pairs)

                if work_type == 0 or work_type == 1:
                    self.deactivate_exist = ti.field(ti.u8, shape=())
                    self.contact_active = ti.field(u1)
                    ti.root.dense(ti.i, round32(object_num1 * object_num2) // 32).quant_array(
                        ti.i, dimensions=32, max_num_bits=32
                    ).place(self.contact_active)
                if (
                    self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics"
                ) and object_type == "particle":
                    self.hist_cplist = HistoryRollingISContactTable.field(shape=max_object_pairs)
                else:
                    self.hist_cplist = HistoryRollingContactTable.field(shape=max_object_pairs)
            elif object_type == "wall" and self.sims.wall_type == 3:
                if self.sims.scheme == "DEM":
                    self.cplist = DigitalRollingContactTable.field(shape=int(self.sims.max_particle_num))
                elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                    self.cplist = DigitalRollingContactTable.field(shape=int(self.sims.max_ls_contact_node_num))
        self.first_run = False

    def get_contact_output(self, scene: SceneManager, neighbor_list, total_len):
        contact_count = int(neighbor_list[total_len])
        end1 = field_to_numpy_prefix(self.cplist.endID1, contact_count)
        end2 = field_to_numpy_prefix(self.cplist.endID2, contact_count)
        normal_force = field_to_numpy_prefix(self.cplist.cnforce, contact_count)
        tangential_force = field_to_numpy_prefix(self.cplist.csforce, contact_count)
        old_tangential_overlap = field_to_numpy_prefix(self.cplist.oldTangOverlap, contact_count)
        old_roll_angle = field_to_numpy_prefix(self.cplist.oldRollAngle, contact_count)
        old_twist_angle = field_to_numpy_prefix(self.cplist.oldTwistAngle, contact_count)
        return end1, end2, normal_force, tangential_force, old_tangential_overlap, old_roll_angle, old_twist_angle

    def get_ppcontact_output(
        self, contact_path, current_time, current_print, scene: SceneManager, pcontact: NeighborBase
    ):
        output = self.get_contact_energy_output()
        if self.sims.scheme == "DEM":
            particle_particle = field_to_numpy_prefix(pcontact.hist_particle_particle, scene.particleNum[0] + 1)
            end1, end2, normal_force, tangential_force, old_tangential_overlap, old_roll_angle, old_twist_angle = (
                self.get_contact_output(scene, particle_particle, scene.particleNum[0])
            )
            output.update(
                {
                    "t_current": current_time,
                    "contact_num": particle_particle,
                    "end1": end1,
                    "end2": end2,
                    "normal_force": normal_force,
                    "tangential_force": tangential_force,
                    "oldTangentialOverlap": old_tangential_overlap,
                    "oldRollAngle": old_roll_angle,
                    "oldTwistAngle": old_twist_angle,
                }
            )
            np.savez(contact_path + f"{current_print:06d}", **output)
        elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
            contact_node_num = self.get_ls_contact_node_num(scene)
            particle_particle = field_to_numpy_prefix(pcontact.hist_lsparticle_lsparticle, contact_node_num + 1)
            end1, end2, normal_force, tangential_force, old_tangential_overlap, old_roll_angle, old_twist_angle = (
                self.get_contact_output(scene, particle_particle, contact_node_num)
            )
            output.update(
                {
                    "t_current": current_time,
                    "contact_num": particle_particle,
                    "end1": end1,
                    "end2": end2,
                    "normal_force": normal_force,
                    "tangential_force": tangential_force,
                    "oldTangentialOverlap": old_tangential_overlap,
                    "oldRollAngle": old_roll_angle,
                    "oldTwistAngle": old_twist_angle,
                }
            )
            np.savez(contact_path + f"{current_print:06d}", **output)

    def get_pwcontact_output(
        self, contact_path, current_time, current_print, scene: SceneManager, pcontact: NeighborBase
    ):
        output = self.get_contact_energy_output()
        if self.sims.scheme == "DEM":
            particle_wall = field_to_numpy_prefix(pcontact.hist_particle_wall, scene.particleNum[0] + 1)
            end1, end2, normal_force, tangential_force, old_tangential_overlap, old_roll_angle, old_twist_angle = (
                self.get_contact_output(scene, particle_wall, scene.particleNum[0])
            )
            output.update(
                {
                    "t_current": current_time,
                    "contact_num": particle_wall,
                    "end1": end1,
                    "end2": end2,
                    "normal_force": normal_force,
                    "tangential_force": tangential_force,
                    "oldTangentialOverlap": old_tangential_overlap,
                    "oldRollAngle": old_roll_angle,
                    "oldTwistAngle": old_twist_angle,
                }
            )
            np.savez(contact_path + f"{current_print:06d}", **output)
        elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
            contact_node_num = self.get_ls_contact_node_num(scene)
            particle_wall = field_to_numpy_prefix(pcontact.hist_lsparticle_wall, contact_node_num + 1)
            end1, end2, normal_force, tangential_force, old_tangential_overlap, old_roll_angle, old_twist_angle = (
                self.get_contact_output(scene, particle_wall, contact_node_num)
            )
            output.update(
                {
                    "t_current": current_time,
                    "contact_num": particle_wall,
                    "end1": end1,
                    "end2": end2,
                    "normal_force": normal_force,
                    "tangential_force": tangential_force,
                    "oldTangentialOverlap": old_tangential_overlap,
                    "oldRollAngle": old_roll_angle,
                    "oldTwistAngle": old_twist_angle,
                }
            )
            np.savez(contact_path + f"{current_print:06d}", **output)

    def rebuild_contact_list(self, contact_info):
        object_object = DictIO.get_essential(contact_info, "contact_num")
        dst_id = DictIO.get_essential(contact_info, "end2")
        normal_force = DictIO.get_essential(contact_info, "normal_force")
        tangential_force = DictIO.get_essential(contact_info, "tangential_force")
        old_tang_overlap = DictIO.get_essential(contact_info, "oldTangentialOverlap")
        old_roll_angle = DictIO.get_alternative(contact_info, "oldRollAngle", np.zeros_like(old_tang_overlap))
        old_twist_angle = DictIO.get_alternative(contact_info, "oldTwistAngle", np.zeros_like(old_tang_overlap))
        return object_object, dst_id, normal_force, tangential_force, old_tang_overlap, old_roll_angle, old_twist_angle

    def rebuild_ppcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, dst_id, normal_force, tangential_force, old_tang_overlap, old_roll_angle, old_twist_angle = (
            self.rebuild_contact_list(contact_info)
        )
        if dst_id.shape[0] > self.cplist.shape[0]:
            raise SimulationStateError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_addition_history_contact_list(
            self.cplist,
            pcontact.hist_particle_particle,
            object_object,
            dst_id,
            normal_force,
            tangential_force,
            old_tang_overlap,
            old_roll_angle,
            old_twist_angle,
        )

    def rebuild_pwcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, dst_id, normal_force, tangential_force, old_tang_overlap, old_roll_angle, old_twist_angle = (
            self.rebuild_contact_list(contact_info)
        )
        if dst_id.shape[0] > self.cplist.shape[0]:
            raise SimulationStateError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_addition_history_contact_list(
            self.cplist,
            pcontact.hist_particle_wall,
            object_object,
            dst_id,
            normal_force,
            tangential_force,
            old_tang_overlap,
            old_roll_angle,
            old_twist_angle,
        )

    # ========================================================= #
    #              Particle Contact Matrix Resolve              #
    # ========================================================= #
    def update_particle_particle_contact_table(self, sims: Simulation, scene: SceneManager, pcontact: NeighborBase):
        copy_addition_contact_table(
            pcontact.hist_particle_particle, int(scene.particleNum[0]), self.cplist, self.hist_cplist
        )
        self.update_ppcontact_table(sims, scene, pcontact)
        kernel_inherit_rolling_history(
            int(scene.particleNum[0]),
            self.cplist,
            self.hist_cplist,
            pcontact.particle_particle,
            pcontact.hist_particle_particle,
        )

    def update_particle_wall_contact_table(self, sims: Simulation, scene: SceneManager, pcontact: NeighborBase):
        copy_addition_contact_table(
            pcontact.hist_particle_wall, int(scene.particleNum[0]), self.cplist, self.hist_cplist
        )
        self.update_pwcontact_table(sims, scene, pcontact)
        kernel_inherit_rolling_history(
            int(scene.particleNum[0]),
            self.cplist,
            self.hist_cplist,
            pcontact.particle_wall,
            pcontact.hist_particle_wall,
        )
