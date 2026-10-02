import taichi as ti
import os
import numpy as np

from src.dem.structs.BaseStruct import (
    ContactTable,
    ISContactTable,
    HistoryContactTable,
    HistoryISContactTable,
    DigitalContactTable,
)
from src.dem.contact.ContactKernel import (
    ISparticle_contact_model_type1,
    ISparticle_contact_model_type2,
    ISparticle_wall_contact_model_type1,
    ISparticle_wall_contact_model_type2,
    LSparticle_contact_model_type0,
    LSparticle_contact_model_type1,
    LSparticle_wall_contact_model,
    LSparticle_wall_contact_model_type0,
    copy_contact_table,
    copy_lsmpm_contact_table,
    find_history,
    kernel_ISparticle_ISparticle_force_assemble_,
    kernel_ISparticle_wall_force_assemble_,
    kernel_LSparticle_LSparticle_force_assemble_,
    kernel_LSparticle_wall_force_assemble_,
    kernel_inherit_IScontact_history,
    kernel_inherit_contact_history,
    kernel_inherit_lsmpm_contact_history,
    kernel_particle_particle_force_assemble_,
    kernel_particle_wall_force_assemble_,
    kernel_rebulid_history_contact_list,
    particle_contact_model_type1,
    particle_contact_model_type2,
    update_LScontact_table_,
    update_contact_bit_table_,
    update_contact_table_,
    update_contact_table_hierarchical_,
    update_contact_wall_bit_table_,
    update_wall_contact_table_,
    update_wall_contact_table_hierarchical_,
    wall_contact_model_type1,
    wall_contact_model_type2,
)
from src.dem.contact.DigitalElevation import (
    kernel_LSparticle_digital_elevation_heightfield_force_assemble_,
    kernel_particle_digital_elevation_heightfield_force_assemble_,
)
from src.dem.contact.contact_point_root import GJKiteration, LagrangianMultiplieriteration, PCNiteration
from src.dem.Simulation import Simulation
from src.dem.SceneManager import myScene
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.neighbor.HierarchicalLinkedCell import HierarchicalLinkedCell
from src.utils.ObjectIO import DictIO
from src.utils.FieldIO import field_to_numpy_prefix
from src.utils.linalg import round32
from src.utils.TypeDefination import u1
from src.utils import GlobalVariable


class ContactModelBase(object):
    sims: Simulation

    def __init__(self, sims):
        self.sims = sims
        self.name = "Base"
        self.contact_list_initialize = None
        self.resolve = None
        self.update_contact_table = None
        self.cplist = None
        self.contact_type = None
        self.hist_cplist = None
        self.contact_active = None
        self.deactivate_exist = None
        self.surfaceProps = None
        self.contact_model = None
        self.null_model = True
        self.iterative_model = None
        self.model_type = -1
        self.first_run = True

    def _soft_particle_backend(self, scene):
        backend = getattr(scene, "soft_particle_backend", None)
        if backend is None:
            raise RuntimeError(
                "LSMPM contact force assembly is an MPM soft-particle mode. Use MPDEM/MPM soft_particle so the coupling layer installs the backend."
            )
        return backend

    def manage_function(self, object_type, work_type):
        self.resolve = self.no_operation
        self.update_contact_table = self.no_operation
        self.add_surface_properties = self.no_add_property
        self.calcu_critical_timesteps = self.no_critical_timestep
        self.update_verlet_particle_particle_tables = self.no_operation
        self.update_verlet_particle_wall_tables = self.no_operation
        if not self.null_model:
            self.add_surface_properties = self.add_surface_property
            self.calcu_critical_timesteps = self.calcu_critical_timestep
            if object_type == "particle":
                if self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                    if self.sims.iterative_model == "LagrangianMultiplier":
                        self.iterative_model = LagrangianMultiplieriteration
                    elif self.sims.iterative_model == "GJK":
                        self.iterative_model = GJKiteration
                    elif self.sims.iterative_model == "PCN":
                        self.iterative_model = PCNiteration
                        raise RuntimeError("PCN iterative model is not implemented yet.")
                if work_type == 0 or work_type == 1:
                    self.resolve = self.tackle_particle_particle_contact_bit_table
                elif work_type == 2:
                    if self.sims.scheme == "DEM":
                        self.resolve = self.tackle_particle_particle_contact_cplist
                        self.update_contact_table = self.update_particle_particle_contact_table
                    elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                        self.resolve = self.tackle_LSparticle_LSparticle_contact_cplist
                        self.update_contact_table = self.update_LSparticle_LSparticle_contact_table
                    elif self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                        self.resolve = self.tackle_ISparticle_ISparticle_contact_cplist
                        self.update_contact_table = self.update_ISparticle_ISparticle_contact_table
            elif object_type == "wall":
                if self.sims.use_digital_elevation_heightfield():
                    self.resolve = self.tackle_digital_elevation_heightfield_contact
                    self.update_contact_table = self.no_operation
                    self.update_verlet_particle_wall_tables = self.no_operation
                elif work_type == 0 or work_type == 1:
                    self.resolve = self.tackle_particle_wall_contact_bit_table
                elif work_type == 2:
                    if self.sims.scheme == "DEM":
                        self.resolve = self.tackle_particle_wall_contact_cplist
                        self.update_contact_table = self.update_particle_wall_contact_table
                    elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                        self.resolve = self.tackle_LSparticle_wall_contact_cplist
                        self.update_contact_table = self.update_LSparticle_wall_contact_table
                    elif self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                        self.resolve = self.tackle_ISparticle_wall_contact_cplist
                        self.update_contact_table = self.update_particle_wall_contact_table

            self.update_ppcontact_table = self.update_particle_contact_table
            if self.sims.scheme == "DEM":
                if self.sims.search == "HierarchicalLinkedCell":
                    self.update_ppcontact_table = self.update_particle_contact_table_hierarchical
            elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                self.update_ppcontact_table = self.update_LSparticle_contact_table

            self.update_pwcontact_table = self.update_wall_contact_table
            if self.sims.scheme == "DEM":
                if self.sims.search == "HierarchicalLinkedCell":
                    self.update_pwcontact_table = self.update_wall_contact_table_hierarchical
            elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                self.update_pwcontact_table = self.update_LSwall_contact_table
            elif self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                if self.sims.wall_type == 1 or self.sims.wall_type == 2:
                    self.update_pwcontact_table = self.update_ISwall_contact_table

            if self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                if self.sims.search == "HierarchicalLinkedCell":
                    self.update_verlet_particle_particle_tables = self.update_particle_verlet_table_hierarchical
                    self.update_verlet_particle_wall_tables = self.update_wall_verlet_table_hierarchical
                else:
                    self.update_verlet_particle_particle_tables = self.update_particle_verlet_table
                    self.update_verlet_particle_wall_tables = self.update_wall_verlet_table

            if object_type == "wall" and self.sims.use_digital_elevation_heightfield():
                self.update_contact_table = self.no_operation
                self.update_verlet_particle_wall_tables = self.no_operation

            if object_type == "particle":
                if self.sims.scheme == "DEM":
                    if self.model_type == 1:
                        self.contact_model = particle_contact_model_type1
                    elif self.model_type == 2:
                        self.contact_model = particle_contact_model_type2
                elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                    if self.model_type == 0:
                        self.contact_model = LSparticle_contact_model_type0
                    elif self.model_type == 1:
                        self.contact_model = LSparticle_contact_model_type1
                elif self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                    if self.model_type == 1:
                        self.contact_model = ISparticle_contact_model_type1
                    elif self.model_type == 2:
                        self.contact_model = ISparticle_contact_model_type2
            elif object_type == "wall":
                if self.sims.scheme == "DEM":
                    if self.model_type == 1:
                        self.contact_model = wall_contact_model_type1
                    elif self.model_type == 2:
                        self.contact_model = wall_contact_model_type2
                elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                    if self.model_type == 0:
                        self.contact_model = LSparticle_wall_contact_model_type0
                    elif self.model_type == 1:
                        self.contact_model = LSparticle_wall_contact_model
                elif self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                    if self.model_type == 1:
                        self.contact_model = ISparticle_wall_contact_model_type1
                    elif self.model_type == 2:
                        self.contact_model = ISparticle_wall_contact_model_type2

            if self.resolve is None:
                raise RuntimeError("Internal error!")

    def collision_initialize(self, object_type, work_type, max_object_pairs, object_num1, object_num2):
        if not self.null_model and self.first_run:
            if object_type == "wall" and self.sims.use_digital_elevation_heightfield():
                if self.sims.scheme == "DEM":
                    self.cplist = DigitalContactTable.field(shape=int(self.sims.max_particle_num))
                elif self.sims.scheme == "LSDEM":
                    self.cplist = DigitalContactTable.field(shape=int(self.sims.max_ls_contact_node_num))
                else:
                    raise RuntimeError(
                        f"DigitalElevation heightfield contact does not support DEM scheme {self.sims.scheme}"
                    )
            elif object_type == "particle" or object_type == "wall":
                if (
                    self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics"
                ) and object_type == "particle":
                    self.cplist = ISContactTable.field(shape=max_object_pairs)
                else:
                    self.cplist = ContactTable.field(shape=max_object_pairs)
                if work_type == 0 or work_type == 1:
                    self.deactivate_exist = ti.field(ti.u8, shape=())
                    self.contact_active = ti.field(u1)
                    ti.root.dense(ti.i, round32(object_num1 * object_num2) // 32).quant_array(
                        ti.i, dimensions=32, max_num_bits=32
                    ).place(self.contact_active)

                if (
                    self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics"
                ) and object_type == "particle":
                    self.hist_cplist = HistoryISContactTable.field(shape=max_object_pairs)
                else:
                    self.hist_cplist = HistoryContactTable.field(shape=max_object_pairs)

                if self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                    self.contact_type = ti.field(ti.u8, shape=max_object_pairs)
                    if self.sims.wall_type == 0:
                        self.contact_type.fill(1)
            elif object_type == "wall" and self.sims.wall_type == 3:
                if self.sims.scheme == "DEM":
                    self.cplist = DigitalContactTable.field(shape=int(self.sims.max_particle_num))
                elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                    self.cplist = DigitalContactTable.field(shape=int(self.sims.max_ls_contact_node_num))
        self.first_run = False

    def get_componousID(self, max_material_num, materialID1, materialID2):
        return int(materialID1 * max_material_num + materialID2)

    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        raise NotImplementedError

    def no_add_property(self, materialID1, materialID2, property):
        return self.get_componousID(self.sims.max_material_num, materialID1, materialID2)

    def calcu_critical_timestep(self, sims, scene):
        raise NotImplementedError

    def no_critical_timestep(self, scene):
        return np.inf

    def find_max_penetration(self):
        return 0.0

    def get_ls_contact_node_num(self, scene: myScene):
        if self.sims.scheme == "LSMPM":
            return int(scene.lsContactNodeNum[0])
        return int(scene.surfaceNum[0])

    def get_contact_output(self, scene: myScene, neighbor_list, total_len):
        contact_count = int(neighbor_list[total_len])
        end1 = field_to_numpy_prefix(self.cplist.endID1, contact_count)
        end2 = field_to_numpy_prefix(self.cplist.endID2, contact_count)
        normal_force = field_to_numpy_prefix(self.cplist.cnforce, contact_count)
        tangential_force = field_to_numpy_prefix(self.cplist.csforce, contact_count)
        oldTangentialOverlap = field_to_numpy_prefix(self.cplist.oldTangOverlap, contact_count)
        normal_overlap = field_to_numpy_prefix(self.cplist.normalOverlap, contact_count)
        normal_overlap_active = field_to_numpy_prefix(self.cplist.normalOverlapActive, contact_count)
        return (
            end1,
            end2,
            normal_force,
            tangential_force,
            oldTangentialOverlap,
            normal_overlap,
            normal_overlap_active,
        )

    def update_properties(self, materialID1, materialID2, property_name, value, override):
        if materialID1 == materialID2:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
        else:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
            self.update_property(componousID, property_name, value, override)
            componousID = self.get_componousID(self.sims.max_material_num, materialID2, materialID1)
            self.update_property(componousID, property_name, value, override)
        return componousID

    def update_property(self, componousID, property_name, value, override):
        raise NotImplementedError

    def restart(self, pcontact, file_number, contact, is_particle_particle=True):
        if not contact is None:
            if not os.path.exists(contact):
                raise EOFError("Invaild contact path")
            if is_particle_particle:
                contact_info = np.load(contact + "/DEMContactPP{0:06d}.npz".format(file_number), allow_pickle=True)
                self.rebuild_ppcontact_list(pcontact, contact_info)
            else:
                contact_info = np.load(contact + "/DEMContactPW{0:06d}.npz".format(file_number), allow_pickle=True)
                self.rebuild_pwcontact_list(pcontact, contact_info)

    def rebuild_contact_list(self, contact_info):
        object_object = DictIO.GetEssential(contact_info, "contact_num")
        DstID = DictIO.GetEssential(contact_info, "end2")
        normal_force = DictIO.GetEssential(contact_info, "normal_force")
        tangential_force = DictIO.GetEssential(contact_info, "tangential_force")
        oldTangOverlap = DictIO.GetEssential(contact_info, "oldTangentialOverlap")
        return object_object, DstID, normal_force, tangential_force, oldTangOverlap

    def get_contact_energy_output(self):
        output = {}
        if self.sims.energy_tracking:
            elastic_energy = np.ascontiguousarray(self.surfaceProps.elastic_energy.to_numpy())
            friction_energy = np.ascontiguousarray(self.surfaceProps.friction_energy.to_numpy())
            viscous_damping_energy = np.ascontiguousarray(self.surfaceProps.damp_energy.to_numpy())
            output.update(
                {
                    "elastic_energy": elastic_energy,
                    "friction_energy": friction_energy,
                    "viscous_damping_energy": viscous_damping_energy,
                }
            )
        return output

    def get_ppcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        output = self.get_contact_energy_output()
        if self.sims.scheme == "DEM":
            particleParticle = field_to_numpy_prefix(pcontact.hist_particle_particle, scene.particleNum[0] + 1)
            end1, end2, normal_force, tangential_force, oldTangentialOverlap, normal_overlap, normal_overlap_active = (
                self.get_contact_output(scene, particleParticle, scene.particleNum[0])
            )
            output.update(
                {
                    "t_current": current_time,
                    "contact_num": particleParticle,
                    "end1": end1,
                    "end2": end2,
                    "normal_force": normal_force,
                    "tangential_force": tangential_force,
                    "oldTangentialOverlap": oldTangentialOverlap,
                    "normal_overlap": normal_overlap,
                    "normal_overlap_active": normal_overlap_active,
                }
            )
            np.savez(contact_path + f"{current_print:06d}", **output)
        elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
            contact_node_num = self.get_ls_contact_node_num(scene)
            particleParticle = field_to_numpy_prefix(pcontact.hist_lsparticle_lsparticle, contact_node_num + 1)
            end1, end2, normal_force, tangential_force, oldTangentialOverlap, normal_overlap, normal_overlap_active = (
                self.get_contact_output(scene, particleParticle, contact_node_num)
            )
            output.update(
                {
                    "t_current": current_time,
                    "contact_num": particleParticle,
                    "end1": end1,
                    "end2": end2,
                    "normal_force": normal_force,
                    "tangential_force": tangential_force,
                    "oldTangentialOverlap": oldTangentialOverlap,
                    "normal_overlap": normal_overlap,
                    "normal_overlap_active": normal_overlap_active,
                }
            )
            np.savez(contact_path + f"{current_print:06d}", **output)

    def get_pwcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        output = self.get_contact_energy_output()
        if self.sims.scheme == "DEM":
            particleWall = field_to_numpy_prefix(pcontact.hist_particle_wall, scene.particleNum[0] + 1)
            end1, end2, normal_force, tangential_force, oldTangentialOverlap, normal_overlap, normal_overlap_active = (
                self.get_contact_output(scene, particleWall, scene.particleNum[0])
            )
            output.update(
                {
                    "t_current": current_time,
                    "contact_num": particleWall,
                    "end1": end1,
                    "end2": end2,
                    "normal_force": normal_force,
                    "tangential_force": tangential_force,
                    "oldTangentialOverlap": oldTangentialOverlap,
                    "normal_overlap": normal_overlap,
                    "normal_overlap_active": normal_overlap_active,
                }
            )
            np.savez(contact_path + f"{current_print:06d}", **output)
        elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
            contact_node_num = self.get_ls_contact_node_num(scene)
            particleWall = field_to_numpy_prefix(pcontact.hist_lsparticle_wall, contact_node_num + 1)
            end1, end2, normal_force, tangential_force, oldTangentialOverlap, normal_overlap, normal_overlap_active = (
                self.get_contact_output(scene, particleWall, contact_node_num)
            )
            output.update(
                {
                    "t_current": current_time,
                    "contact_num": particleWall,
                    "end1": end1,
                    "end2": end2,
                    "normal_force": normal_force,
                    "tangential_force": tangential_force,
                    "oldTangentialOverlap": oldTangentialOverlap,
                    "normal_overlap": normal_overlap,
                    "normal_overlap_active": normal_overlap_active,
                }
            )
            np.savez(contact_path + f"{current_print:06d}", **output)

    def rebuild_ppcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, DstID, normal_force, tangential_force, oldTangOverlap = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        if self.sims.scheme == "DEM":
            kernel_rebulid_history_contact_list(
                self.cplist,
                pcontact.hist_particle_particle,
                object_object,
                DstID,
                normal_force,
                tangential_force,
                oldTangOverlap,
            )
        elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
            kernel_rebulid_history_contact_list(
                self.cplist,
                pcontact.hist_lsparticle_lsparticle,
                object_object,
                DstID,
                normal_force,
                tangential_force,
                oldTangOverlap,
            )

    def rebuild_pwcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, DstID, normal_force, tangential_force, oldTangOverlap = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        if self.sims.scheme == "DEM":
            kernel_rebulid_history_contact_list(
                self.cplist,
                pcontact.hist_particle_wall,
                object_object,
                DstID,
                normal_force,
                tangential_force,
                oldTangOverlap,
            )
        elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
            kernel_rebulid_history_contact_list(
                self.cplist,
                pcontact.hist_lsparticle_wall,
                object_object,
                DstID,
                normal_force,
                tangential_force,
                oldTangOverlap,
            )

    def reset(self):
        if GlobalVariable.TRACKENERGY and self.null_model is False:
            self.surfaceProps.elastic_energy.fill(0)

    # ========================================================= #
    #                   Bit Table Resolve                       #
    # ========================================================= #
    def tackle_particle_particle_contact_bit_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_contact_bit_table_(
            pcontact.particle_particle,
            sims.max_particle_num,
            pcontact.potential_list_particle_particle,
            scene.particle,
            self.cplist,
            self.active_contactNum,
            self.contact_active,
        )
        kernel_particle_particle_force_assemble_(
            int(scene.particleNum[0]),
            sims.dt,
            sims.max_material_num,
            self.surfaceProps,
            scene.particle,
            self.cplist,
            self.hist_cplist,
            pcontact.particle_particle,
            pcontact.hist_particle_particle,
            find_history,
        )
        copy_contact_table(pcontact.particle_particle, int(scene.particleNum[0]), self.cplist, self.hist_cplist)

    def tackle_particle_wall_contact_bit_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_contact_wall_bit_table_(
            pcontact.particle_wall,
            sims.max_wall_num,
            pcontact.potential_list_particle_wall,
            scene.particle,
            scene.wall,
            self.cplist,
            self.active_contactNum,
            self.contact_active,
        )
        kernel_particle_wall_force_assemble_(
            int(scene.particleNum[0]),
            sims.dt,
            sims.max_material_num,
            self.surfaceProps,
            scene.particle,
            scene.wall,
            self.cplist,
            self.hist_cplist,
            pcontact.particle_wall,
            pcontact.hist_particle_wall,
            find_history,
        )
        copy_contact_table(pcontact.particle_wall, int(scene.particleNum[0]), self.cplist, self.hist_cplist)

    # ========================================================= #
    #              Particle Contact Matrix Resolve              #
    # ========================================================= #
    def update_particle_particle_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_contact_table(pcontact.hist_particle_particle, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
        self.update_ppcontact_table(sims, scene, pcontact)
        kernel_inherit_contact_history(
            int(scene.particleNum[0]),
            self.cplist,
            self.hist_cplist,
            pcontact.particle_particle,
            pcontact.hist_particle_particle,
        )

    def update_particle_wall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_contact_table(pcontact.hist_particle_wall, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
        self.update_pwcontact_table(sims, scene, pcontact)
        kernel_inherit_contact_history(
            int(scene.particleNum[0]),
            self.cplist,
            self.hist_cplist,
            pcontact.particle_wall,
            pcontact.hist_particle_wall,
        )

    def tackle_particle_particle_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_particle_particle_force_assemble_(
            int(scene.particleNum[0]),
            sims.dt,
            sims.max_material_num,
            self.surfaceProps,
            scene.particle,
            scene.particle,
            self.cplist,
            pcontact.hist_particle_particle,
            self.contact_model,
        )

    def tackle_particle_wall_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_particle_wall_force_assemble_(
            int(scene.particleNum[0]),
            sims.dt,
            sims.max_material_num,
            self.surfaceProps,
            scene.particle,
            scene.wall,
            self.cplist,
            pcontact.hist_particle_wall,
            self.contact_model,
        )

    def tackle_digital_elevation_heightfield_contact(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        if sims.scheme == "DEM":
            kernel_particle_digital_elevation_heightfield_force_assemble_(
                int(scene.particleNum[0]),
                sims.dt,
                sims.max_material_num,
                self.surfaceProps,
                scene.particle,
                int(scene.digital_elevation.materialID),
                scene.digital_elevation.digital_size,
                scene.digital_elevation.idigital_size,
                scene.digital_elevation.digital_dim,
                scene.digital_elevation.height_dim,
                scene.digital_elevation.no_data,
                scene.digital_elevation.height,
                self.cplist,
                self.model_type,
            )
        elif sims.scheme == "LSDEM":
            kernel_LSparticle_digital_elevation_heightfield_force_assemble_(
                int(scene.surfaceNum[0]),
                sims.dt,
                sims.max_material_num,
                self.surfaceProps,
                scene.rigid,
                scene.vertice,
                scene.surface,
                scene.box,
                int(scene.digital_elevation.materialID),
                scene.digital_elevation.digital_size,
                scene.digital_elevation.idigital_size,
                scene.digital_elevation.digital_dim,
                scene.digital_elevation.height_dim,
                scene.digital_elevation.no_data,
                scene.digital_elevation.height,
                self.cplist,
            )
        else:
            raise RuntimeError(f"DigitalElevation heightfield contact does not support DEM scheme {sims.scheme}")

    def update_LSparticle_LSparticle_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        contact_node_num = self.get_ls_contact_node_num(scene)
        copy_lsmpm_contact_table(pcontact.hist_lsparticle_lsparticle, contact_node_num, self.cplist, self.hist_cplist)
        self.update_ppcontact_table(sims, scene, pcontact)
        kernel_inherit_lsmpm_contact_history(
            contact_node_num,
            self.cplist,
            self.hist_cplist,
            pcontact.lsparticle_lsparticle,
            pcontact.hist_lsparticle_lsparticle,
        )

    def update_LSparticle_wall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        contact_node_num = self.get_ls_contact_node_num(scene)
        copy_lsmpm_contact_table(pcontact.hist_lsparticle_wall, contact_node_num, self.cplist, self.hist_cplist)
        self.update_pwcontact_table(sims, scene, pcontact)
        kernel_inherit_lsmpm_contact_history(
            contact_node_num, self.cplist, self.hist_cplist, pcontact.lsparticle_wall, pcontact.hist_lsparticle_wall
        )

    def update_ISparticle_ISparticle_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_contact_table(pcontact.hist_particle_particle, int(scene.rigidNum[0]), self.cplist, self.hist_cplist)
        self.update_ppcontact_table(sims, scene, pcontact)
        kernel_inherit_IScontact_history(
            int(scene.rigidNum[0]),
            self.cplist,
            self.hist_cplist,
            pcontact.particle_particle,
            pcontact.hist_particle_particle,
        )

    def update_ISparticle_wall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_contact_table(pcontact.hist_particle_wall, int(scene.rigidNum[0]), self.cplist, self.hist_cplist)
        self.update_pwcontact_table(sims, scene, pcontact)
        kernel_inherit_IScontact_history(
            int(scene.rigidNum[0]), self.cplist, self.hist_cplist, pcontact.particle_wall, pcontact.hist_particle_wall
        )

    def tackle_LSparticle_LSparticle_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        if sims.scheme == "LSMPM":
            self._soft_particle_backend(scene).tackle_lsparticle_lsparticle_contact(self, sims, scene, pcontact)
        else:
            kernel_LSparticle_LSparticle_force_assemble_(
                int(scene.surfaceNum[0]),
                sims.dt,
                sims.max_material_num,
                self.surfaceProps,
                scene.rigid,
                scene.rigid_grid,
                scene.vertice,
                scene.surface,
                scene.box,
                self.cplist,
                pcontact.hist_lsparticle_lsparticle,
                self.contact_model,
            )

    def tackle_ISparticle_ISparticle_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_ISparticle_ISparticle_force_assemble_(
            int(scene.particleNum[0]),
            sims.dt,
            sims.max_material_num,
            self.surfaceProps,
            scene.particle,
            scene.rigid,
            scene.surface,
            self.cplist,
            pcontact.hist_particle_particle,
            self.contact_model,
            self.iterative_model,
        )

    def tackle_LSparticle_wall_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        if sims.scheme == "LSMPM":
            self._soft_particle_backend(scene).tackle_lsparticle_wall_contact(self, sims, scene, pcontact)
        else:
            kernel_LSparticle_wall_force_assemble_(
                int(scene.surfaceNum[0]),
                sims.dt,
                sims.max_material_num,
                self.surfaceProps,
                scene.rigid,
                scene.vertice,
                scene.surface,
                scene.box,
                scene.wall,
                self.cplist,
                pcontact.hist_lsparticle_wall,
                self.contact_model,
            )

    def tackle_ISparticle_wall_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_ISparticle_wall_force_assemble_(
            int(scene.particleNum[0]),
            sims.wall_type,
            sims.dt,
            sims.max_material_num,
            self.surfaceProps,
            scene.rigid,
            scene.surface,
            scene.wall,
            self.cplist,
            pcontact.hist_particle_wall,
            self.contact_type,
            self.contact_model,
        )

    def update_particle_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_contact_table_(
            sims.potential_particle_num,
            int(scene.particleNum[0]),
            pcontact.particle_particle,
            pcontact.potential_list_particle_particle,
            self.cplist,
        )

    def update_particle_contact_table_hierarchical(
        self, sims: Simulation, scene: myScene, pcontact: HierarchicalLinkedCell
    ):
        update_contact_table_hierarchical_(
            int(scene.particleNum[0]),
            pcontact.particle_particle,
            pcontact.potential_list_particle_particle,
            self.cplist,
            pcontact.body,
        )

    def update_particle_verlet_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_contact_table_(
            sims.potential_particle_num,
            int(scene.particleNum[0]),
            pcontact.particle_particle,
            pcontact.potential_list_particle_particle,
            pcontact.pplist,
        )

    def update_particle_verlet_table_hierarchical(
        self, sims: Simulation, scene: myScene, pcontact: HierarchicalLinkedCell
    ):
        update_contact_table_hierarchical_(
            int(scene.particleNum[0]),
            pcontact.particle_particle,
            pcontact.potential_list_particle_particle,
            pcontact.pplist,
            pcontact.body,
        )

    def update_LSparticle_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_LScontact_table_(
            sims.point_particle_coordination_number,
            self.get_ls_contact_node_num(scene),
            pcontact.lsparticle_lsparticle,
            pcontact.potential_list_point_particle,
            self.cplist,
        )

    def update_wall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_contact_table_(
            sims.wall_coordination_number,
            int(scene.particleNum[0]),
            pcontact.particle_wall,
            pcontact.potential_list_particle_wall,
            self.cplist,
        )

    def update_ISwall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_wall_contact_table_(
            sims.wall_coordination_number,
            int(scene.particleNum[0]),
            scene.rigid,
            scene.surface,
            scene.wall,
            pcontact.particle_wall,
            pcontact.potential_list_particle_wall,
            self.cplist,
            self.contact_type,
        )

    def update_wall_contact_table_hierarchical(
        self, sims: Simulation, scene: myScene, pcontact: HierarchicalLinkedCell
    ):
        update_wall_contact_table_hierarchical_(
            int(scene.particleNum[0]),
            pcontact.particle_wall,
            pcontact.potential_list_particle_wall,
            self.cplist,
            pcontact.body,
        )

    def update_wall_verlet_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_contact_table_(
            sims.wall_coordination_number,
            int(scene.particleNum[0]),
            pcontact.particle_wall,
            pcontact.potential_list_particle_wall,
            pcontact.pwlist,
        )

    def update_wall_verlet_table_hierarchical(self, sims: Simulation, scene: myScene, pcontact: HierarchicalLinkedCell):
        update_wall_contact_table_hierarchical_(
            int(scene.particleNum[0]),
            pcontact.particle_wall,
            pcontact.potential_list_particle_wall,
            pcontact.pwlist,
            pcontact.body,
        )

    def update_LSwall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_LScontact_table_(
            sims.point_wall_coordination_number,
            self.get_ls_contact_node_num(scene),
            pcontact.lsparticle_wall,
            pcontact.potential_list_point_wall,
            self.cplist,
        )

    def no_operation(self, sims, scene, pcontact):
        return
