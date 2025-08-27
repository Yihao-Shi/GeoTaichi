import taichi as ti
import os
import numpy as np

from src.dem.structs.BaseStruct import (ContactTable, ISContactTable, HistoryContactTable, HistoryISContactTable, DigitalContactTable)
from src.dem.contact.ContactKernel import *
from src.dem.Simulation import Simulation
from src.dem.SceneManager import myScene
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.neighbor.HierarchicalLinkedCell import HierarchicalLinkedCell
from src.utils.ObjectIO import DictIO
from src.utils.linalg import round32
from src.utils.TypeDefination import u1

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
                if work_type == 0 or work_type == 1:
                    self.resolve = self.tackle_particle_particle_contact_bit_table
                elif work_type == 2:
                    if self.sims.scheme == "DEM":
                        self.resolve = self.tackle_particle_particle_contact_cplist
                        self.update_contact_table = self.update_particle_particle_contact_table
                    elif self.sims.scheme == "LSDEM":
                        self.resolve = self.tackle_LSparticle_LSparticle_contact_cplist
                        self.update_contact_table = self.update_LSparticle_LSparticle_contact_table
                    elif self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                        self.resolve = self.tackle_ISparticle_ISparticle_contact_cplist
                        self.update_contact_table = self.update_ISparticle_ISparticle_contact_table
            elif object_type == "wall":
                if work_type == 0 or work_type == 1:
                    self.resolve = self.tackle_particle_wall_contact_bit_table
                elif work_type == 2:
                    if self.sims.scheme == "DEM":
                        self.resolve = self.tackle_particle_wall_contact_cplist
                        self.update_contact_table = self.update_particle_wall_contact_table
                    elif self.sims.scheme == "LSDEM":
                        self.resolve = self.tackle_LSparticle_wall_contact_cplist
                        self.update_contact_table = self.update_LSparticle_wall_contact_table
                    elif self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                        self.resolve = self.tackle_ISparticle_wall_contact_cplist
                        self.update_contact_table = self.update_particle_wall_contact_table

            self.update_ppcontact_table = self.update_particle_contact_table
            if self.sims.scheme == "DEM":
                if self.sims.search == "HierarchicalLinkedCell":
                    self.update_ppcontact_table = self.update_particle_contact_table_hierarchical
            elif self.sims.scheme == "LSDEM":
                self.update_ppcontact_table = self.update_LSparticle_contact_table
                
            self.update_pwcontact_table = self.update_wall_contact_table
            if self.sims.scheme == "DEM":
                if self.sims.search == "HierarchicalLinkedCell":
                    self.update_pwcontact_table = self.update_wall_contact_table_hierarchical
            elif self.sims.scheme == "LSDEM":
                self.update_pwcontact_table = self.update_LSwall_contact_table
            elif self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                if self.sims.wall_type == 1 or self.sims.wall_type == 2:
                    self.update_pwcontact_table = self.update_ISwall_contact_table

            if self.sims.scheme == "LSDEM":
                if self.sims.search == "HierarchicalLinkedCell":
                    self.update_verlet_particle_particle_tables = self.update_particle_verlet_table_hierarchical
                    self.update_verlet_particle_wall_tables = self.update_wall_verlet_table_hierarchical
                else:
                    self.update_verlet_particle_particle_tables = self.update_particle_verlet_table
                    self.update_verlet_particle_wall_tables = self.update_wall_verlet_table

            if object_type == "particle":
                if self.sims.scheme == "DEM":
                    if self.model_type == 1:
                        self.contact_model = particle_contact_model_type1
                    elif self.model_type == 2:
                        self.contact_model = particle_contact_model_type2
                elif self.sims.scheme == "LSDEM":
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
                elif self.sims.scheme == "LSDEM":
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
            if object_type == 'particle' or object_type == 'wall':
                if (self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics") and object_type == 'particle':
                    self.cplist = ISContactTable.field(shape=max_object_pairs)
                else:
                    self.cplist = ContactTable.field(shape=max_object_pairs)
                if work_type == 0 or work_type == 1:
                    self.deactivate_exist = ti.field(ti.u8, shape=())
                    self.contact_active = ti.field(u1)
                    ti.root.dense(ti.i, round32(object_num1 * object_num2)//32).quant_array(ti.i, dimensions=32, max_num_bits=32).place(self.contact_active)
                    
                if self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics" and object_type == 'particle':
                    self.hist_cplist = HistoryISContactTable.field(shape=max_object_pairs)
                else:
                    self.hist_cplist = HistoryContactTable.field(shape=max_object_pairs)

                if (self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics"):
                    self.contact_type = ti.field(ti.u8, shape=max_object_pairs)
                    if self.sims.wall_type == 0:
                        self.contact_type.fill(1)
            elif object_type == 'wall' and self.sims.wall_type == 3:
                if self.sims.scheme == "DEM":
                    self.cplist = DigitalContactTable.field(shape=int(self.sims.max_particle_num))
                elif self.sims.scheme == "LSDEM":
                    self.cplist = DigitalContactTable.field(shape=int(self.sims.max_surface_node_num * self.sims.max_rigid_body_num))
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
        return 1e-3
    
    def find_max_penetration(self):
        return 0.
    
    def get_contact_output(self, scene: myScene, neighbor_list):
        end1 = np.ascontiguousarray(self.cplist.endID1.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        end2 = np.ascontiguousarray(self.cplist.endID2.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        normal_force = np.ascontiguousarray(self.cplist.cnforce.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        tangential_force = np.ascontiguousarray(self.cplist.csforce.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        oldTangentialOverlap = np.ascontiguousarray(self.cplist.oldTangOverlap.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        return end1, end2, normal_force, tangential_force, oldTangentialOverlap
    
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

    def get_ppcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        particleParticle = np.ascontiguousarray(pcontact.hist_particle_particle.to_numpy()[0:scene.particleNum[0] + 1])
        end1, end2, normal_force, tangential_force, oldTangentialOverlap = self.get_contact_output(scene, particleParticle)
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleParticle, end1=end1, end2=end2, normal_force=normal_force, 
                                                      tangential_force=tangential_force, oldTangentialOverlap=oldTangentialOverlap)
        
    def get_pwcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        particleWall = np.ascontiguousarray(pcontact.hist_particle_wall.to_numpy()[0:scene.particleNum[0] + 1])
        end1, end2, normal_force, tangential_force, oldTangentialOverlap = self.get_contact_output(scene, particleWall)
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleWall, end1=end1, end2=end2, normal_force=normal_force, 
                                                      tangential_force=tangential_force, oldTangentialOverlap=oldTangentialOverlap)
    
    def rebuild_ppcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, DstID, normal_force, tangential_force, oldTangOverlap = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_history_contact_list(self.cplist, pcontact.hist_particle_particle, object_object, DstID, normal_force, tangential_force, oldTangOverlap)

    def rebuild_pwcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, DstID, normal_force, tangential_force, oldTangOverlap = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_history_contact_list(self.cplist, pcontact.hist_particle_wall, object_object, DstID, normal_force, tangential_force, oldTangOverlap)
    
    # ========================================================= #
    #                   Bit Table Resolve                       #
    # ========================================================= # 
    def tackle_particle_particle_contact_bit_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_contact_bit_table_(pcontact.particle_particle, sims.max_particle_num, pcontact.potential_list_particle_particle, scene.particle, self.cplist, self.active_contactNum, self.contact_active)
        kernel_particle_particle_force_assemble_(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.particle, self.cplist, self.hist_cplist, 
                                                 pcontact.particle_particle, pcontact.hist_particle_particle, find_history)
        copy_contact_table(pcontact.particle_particle, int(scene.particleNum[0]), self.cplist, self.hist_cplist)

    def tackle_particle_wall_contact_bit_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_contact_wall_bit_table_(pcontact.particle_wall, sims.max_wall_num, pcontact.potential_list_particle_wall, scene.particle, scene.wall, self.cplist, self.active_contactNum, self.contact_active)
        kernel_particle_wall_force_assemble_(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.particle, scene.wall, self.cplist, self.hist_cplist, 
                                             pcontact.particle_wall, pcontact.hist_particle_wall, find_history)
        copy_contact_table(pcontact.particle_wall, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
    
    # ========================================================= #
    #              Particle Contact Matrix Resolve              #
    # ========================================================= # 
    def update_particle_particle_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_contact_table(pcontact.hist_particle_particle, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
        self.update_ppcontact_table(sims, scene, pcontact)
        kernel_inherit_contact_history(int(scene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_particle, pcontact.hist_particle_particle)
         
    def update_particle_wall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_contact_table(pcontact.hist_particle_wall, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
        self.update_pwcontact_table(sims, scene, pcontact)
        kernel_inherit_contact_history(int(scene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_wall, pcontact.hist_particle_wall)
        
    def tackle_particle_particle_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_particle_particle_force_assemble_(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.particle, scene.particle, self.cplist, pcontact.hist_particle_particle, self.contact_model)
        
    def tackle_particle_wall_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_particle_wall_force_assemble_(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.particle, scene.wall, self.cplist, pcontact.hist_particle_wall, self.contact_model)
    
    def update_LSparticle_LSparticle_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_contact_table(pcontact.hist_lsparticle_lsparticle, int(scene.surfaceNum[0]), self.cplist, self.hist_cplist)
        self.update_ppcontact_table(sims, scene, pcontact)
        kernel_inherit_contact_history(int(scene.surfaceNum[0]), self.cplist, self.hist_cplist, pcontact.lsparticle_lsparticle, pcontact.hist_lsparticle_lsparticle)
        
    def update_LSparticle_wall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_contact_table(pcontact.hist_lsparticle_wall, int(scene.surfaceNum[0]), self.cplist, self.hist_cplist)
        self.update_pwcontact_table(sims, scene, pcontact)
        kernel_inherit_contact_history(int(scene.surfaceNum[0]), self.cplist, self.hist_cplist, pcontact.lsparticle_wall, pcontact.hist_lsparticle_wall)
    
    def update_ISparticle_ISparticle_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_contact_table(pcontact.hist_particle_particle, int(scene.rigidNum[0]), self.cplist, self.hist_cplist)
        self.update_ppcontact_table(sims, scene, pcontact)
        kernel_inherit_IScontact_history(int(scene.rigidNum[0]), self.cplist, self.hist_cplist, pcontact.particle_particle, pcontact.hist_particle_particle)
        
    def update_ISparticle_wall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_contact_table(pcontact.hist_particle_wall, int(scene.rigidNum[0]), self.cplist, self.hist_cplist)
        self.update_pwcontact_table(sims, scene, pcontact)
        kernel_inherit_IScontact_history(int(scene.rigidNum[0]), self.cplist, self.hist_cplist, pcontact.particle_wall, pcontact.hist_particle_wall)
        
    def tackle_LSparticle_LSparticle_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_LSparticle_LSparticle_force_assemble_(int(scene.surfaceNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.rigid, scene.rigid_grid, 
                                                     scene.vertice, scene.surface, scene.box, self.cplist, pcontact.hist_lsparticle_lsparticle, self.contact_model)
        
    def tackle_ISparticle_ISparticle_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_ISparticle_ISparticle_force_assemble_(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.particle, scene.rigid, scene.surface, self.cplist, pcontact.hist_particle_particle, self.contact_model, self.iterative_model)
        
    def tackle_LSparticle_wall_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_LSparticle_wall_force_assemble_(int(scene.surfaceNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.rigid, scene.vertice, scene.surface, 
                                               scene.box, scene.wall, self.cplist, pcontact.hist_lsparticle_wall, self.contact_model)
        
    def tackle_ISparticle_wall_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_ISparticle_wall_force_assemble_(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.rigid, scene.surface, 
                                               scene.wall, self.cplist, pcontact.hist_particle_wall, self.contact_type, self.contact_model)
    
    def update_particle_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_contact_table_(sims.potential_particle_num, int(scene.particleNum[0]), pcontact.particle_particle, pcontact.potential_list_particle_particle, self.cplist)

    def update_particle_contact_table_hierarchical(self, sims: Simulation, scene: myScene, pcontact: HierarchicalLinkedCell):
        update_contact_table_hierarchical_(int(scene.particleNum[0]), pcontact.particle_particle, pcontact.potential_list_particle_particle, self.cplist, pcontact.body)

    def update_particle_verlet_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_contact_table_(sims.potential_particle_num, int(scene.particleNum[0]), pcontact.particle_particle, pcontact.potential_list_particle_particle, pcontact.pplist)

    def update_particle_verlet_table_hierarchical(self, sims: Simulation, scene: myScene, pcontact: HierarchicalLinkedCell):
        update_contact_table_hierarchical_(int(scene.particleNum[0]), pcontact.particle_particle, pcontact.potential_list_particle_particle, pcontact.pplist, pcontact.body)

    def update_LSparticle_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_LScontact_table_(sims.point_particle_coordination_number, int(scene.surfaceNum[0]), pcontact.lsparticle_lsparticle, pcontact.potential_list_point_particle, self.cplist)

    def update_wall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_contact_table_(sims.wall_coordination_number, int(scene.particleNum[0]), pcontact.particle_wall, pcontact.potential_list_particle_wall, self.cplist)

    def update_ISwall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_wall_contact_table_(sims.wall_coordination_number, int(scene.particleNum[0]), scene.rigid, scene.surface, scene.wall, pcontact.particle_wall, pcontact.potential_list_particle_wall, self.cplist, self.contact_type)

    def update_wall_contact_table_hierarchical(self, sims: Simulation, scene: myScene, pcontact: HierarchicalLinkedCell):
        update_wall_contact_table_hierarchical_(int(scene.particleNum[0]), pcontact.particle_wall, pcontact.potential_list_particle_wall, self.cplist, pcontact.body)

    def update_wall_verlet_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_contact_table_(sims.wall_coordination_number, int(scene.particleNum[0]), pcontact.particle_wall, pcontact.potential_list_particle_wall, pcontact.pwlist)

    def update_wall_verlet_table_hierarchical(self, sims: Simulation, scene: myScene, pcontact: HierarchicalLinkedCell):
        update_wall_contact_table_hierarchical_(int(scene.particleNum[0]), pcontact.particle_wall, pcontact.potential_list_particle_wall, pcontact.pwlist, pcontact.body)

    def update_LSwall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        update_LScontact_table_(sims.point_wall_coordination_number, int(scene.surfaceNum[0]), pcontact.lsparticle_wall, pcontact.potential_list_point_wall, self.cplist)
    
    def no_operation(self, sims, scene, pcontact):
        pass