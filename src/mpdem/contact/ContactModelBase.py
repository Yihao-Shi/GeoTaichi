import os
import numpy as np

from src.mpdem.contact.MultiLinkedCell import MultiLinkedCell
from src.mpdem.Simulation import Simulation
from src.mpm.SceneManager import myScene as MPMScene
from src.dem.BaseStruct import CoupledContactTable, HistoryContactTable
from src.dem.SceneManager import myScene as DEMScene
from src.dem.contact.ContactKernel import *
from src.mpm.SceneManager import myScene as MPMScene
from src.utils.ObjectIO import DictIO


class ContactModelBase(object):
    def __init__(self):
        self.resolve = None
        self.update_contact_table = None
        self.cplist = None
        self.hist_cplist = None
        self.surfaceProps = None
        self.null_model = True
        self.contact_model = None
        self.model_type = -1

    def manage_function(self, object_type, dem_scheme=None, wall_type=None):
        self.resolve = self.no_operation
        self.update_contact_table = self.no_operation
        self.add_surface_properties = self.no_add_property
        self.calcu_critical_timesteps = self.no_critical_timestep
        if not self.null_model:
            self.add_surface_properties = self.add_surface_property
            self.calcu_critical_timesteps = self.calcu_critical_timestep
            if object_type == "particle":
                if dem_scheme == "DEM":
                    self.resolve = self.tackle_particle_particle_contact_cplist
                elif dem_scheme == "LSDEM":
                    self.resolve = self.tackle_particle_LSparticle_contact_cplist
                self.update_contact_table = self.update_particle_particle_contact_table
            elif object_type == "wall":
                if wall_type != 3:
                    self.resolve = self.tackle_particle_wall_contact_cplist
                    self.update_contact_table = self.update_particle_wall_contact_table
                elif wall_type == 3:
                    self.resolve = self.tackle_particle_digital_elevation_contact_cplist

            if object_type == "particle":
                if dem_scheme == "DEM":
                    if self.model_type == 1:
                        self.contact_model = particle_contact_model_type1
                    elif self.model_type == 2:
                        self.contact_model = particle_contact_model_type2
                    elif self.model_type == 3:
                        self.contact_model = fluid_particle_contact_model
                elif dem_scheme == "LSDEM":
                    if self.model_type == 0:
                        self.contact_model = LSparticle_contact_model_type0
                    elif self.model_type == 1:
                        self.contact_model = LSparticle_contact_model_type1
                    elif self.model_type == 3:
                        self.contact_model = fluid_LSparticle_contact_model
            elif object_type == "wall":
                if self.model_type == 1:
                    self.contact_model = wall_contact_model_type1
                elif self.model_type == 2:
                    self.contact_model = wall_contact_model_type2
                elif self.model_type == 3:
                    self.contact_model = fluid_wall_contact_model

            if self.resolve is None:
                raise RuntimeError("Internal error!")

    def collision_initialize(self, max_object_pairs):
        if not self.null_model:
            self.cplist = CoupledContactTable.field(shape=int(max_object_pairs))
            self.hist_cplist = HistoryContactTable.field(shape=int(max_object_pairs))

    def get_componousID(self, max_material_num, materialID1, materialID2):
        return int(materialID1 * max_material_num + materialID2)
    
    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        raise NotImplementedError
    
    def no_add_property(self, max_material_num, materialID1, materialID2, property):
        return self.get_componousID(max_material_num, materialID1, materialID2)
    
    def calcu_critical_timestep(self, mscene, dsims, dscene, max_material_num):
        raise NotImplementedError
    
    def no_critical_timestep(self, mscene, dsims, dscene, max_material_num):
        return 1e-3
    
    def get_contact_output(self, scene: MPMScene, neighbor_list):
        end1 = np.ascontiguousarray(self.cplist.endID1.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        end2 = np.ascontiguousarray(self.cplist.endID2.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        oldTangentialOverlap = np.ascontiguousarray(self.cplist.oldTangOverlap.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        return end1, end2, oldTangentialOverlap
    
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
        particle_number = object_object[object_object.shape[0] - 1]
        DstID = DictIO.GetEssential(contact_info, "end2")
        oldTangOverlap = DictIO.GetEssential(contact_info, "oldTangentialOverlap")
        return object_object, particle_number, DstID, oldTangOverlap

    def get_ppcontact_output(self, contact_path, current_time, current_print, scene: MPMScene, pcontact: MultiLinkedCell):
        end1, end2, oldTangentialOverlap = self.get_contact_output(scene, pcontact.particle_particle)
        end1 = np.ascontiguousarray(self.cplist.endID1.to_numpy()[0:pcontact.particle_particle[scene.particleNum[0]]])
        end2 = np.ascontiguousarray(self.cplist.endID2.to_numpy()[0:pcontact.particle_particle[scene.particleNum[0]]])
        oldTangentialOverlap = np.ascontiguousarray(self.cplist.oldTangOverlap.to_numpy()[0:pcontact.particle_particle[scene.particleNum[0]]])
        particleParticle = np.ascontiguousarray(pcontact.hist_particle_particle.to_numpy()[0:pcontact.particle_particle[scene.particleNum[0]]])
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleParticle, end1=end1, end2=end2, oldTangentialOverlap=oldTangentialOverlap)
        
    def get_pwcontact_output(self, contact_path, current_time, current_print, scene: MPMScene, pcontact: MultiLinkedCell):
        end1, end2, oldTangentialOverlap = self.get_contact_output(scene, pcontact.particle_wall)
        end1 = np.ascontiguousarray(self.cplist.endID1.to_numpy()[0:pcontact.particle_wall[scene.particleNum[0]]])
        end2 = np.ascontiguousarray(self.cplist.endID2.to_numpy()[0:pcontact.particle_wall[scene.particleNum[0]]])
        oldTangentialOverlap = np.ascontiguousarray(self.cplist.oldTangOverlap.to_numpy()[0:pcontact.particle_wall[scene.particleNum[0]]])
        particleWall = np.ascontiguousarray(pcontact.hist_particle_wall.to_numpy()[0:pcontact.particle_wall[scene.particleNum[0]]])
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleWall, end1=end1, end2=end2, oldTangentialOverlap=oldTangentialOverlap)
    
    def rebuild_ppcontact_list(self, pcontact: MultiLinkedCell, contact_info):
        object_object, particle_number, DstID, oldTangOverlap = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_history_contact_list(self.cplist, pcontact.hist_particle_particle, object_object, DstID, oldTangOverlap)

    def rebuild_pwcontact_list(self, pcontact: MultiLinkedCell, contact_info):
        object_object, particle_number, DstID, oldTangOverlap = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_history_contact_list(self.cplist, pcontact.hist_particle_wall, object_object, DstID, oldTangOverlap)
    
    # ========================================================= #
    #              Particle Contact Matrix Resolve              #
    # ========================================================= # 
    def update_particle_particle_contact_table(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        copy_contact_table(pcontact.hist_particle_particle, int(mscene.particleNum[0]), self.cplist, self.hist_cplist)
        update_contact_table_(sims.potential_particle_num, int(mscene.particleNum[0]), pcontact.particle_particle, pcontact.potential_list_particle_particle, self.cplist)
        kernel_inherit_contact_history(int(mscene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_particle, pcontact.hist_particle_particle)

    def update_particle_wall_contact_table(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        copy_contact_table(pcontact.hist_particle_wall, int(mscene.particleNum[0]), self.cplist, self.hist_cplist)
        update_contact_table_(sims.wall_coordination_number, int(mscene.particleNum[0]), pcontact.particle_wall, pcontact.potential_list_particle_wall, self.cplist)
        kernel_inherit_contact_history(int(mscene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_wall, pcontact.hist_particle_wall)

    def tackle_particle_particle_contact_cplist(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        kernel_particle_particle_force_assemble_(int(mscene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, mscene.particle, dscene.particle, 
                                                 self.cplist, pcontact.hist_particle_particle, self.contact_model)

    def tackle_particle_wall_contact_cplist(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        kernel_particle_wall_force_assemble_(int(mscene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, mscene.particle, dscene.wall, 
                                             self.cplist, pcontact.hist_particle_wall, self.contact_model)
        
    def tackle_particle_LSparticle_contact_cplist(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        kernel_particle_LSparticle_force_assemble_(int(mscene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, mscene.particle, dscene.rigid, dscene.rigid_grid, 
                                                   dscene.box, self.cplist, pcontact.hist_particle_particle, self.contact_model)
        
    def tackle_particle_digital_elevation_contact_cplist(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        kernel_particle_digital_elevation_force_assemble_(int(mscene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, mscene.particle, dscene.wall, 
                                                          dscene.digital_elevation.idigital_size, dscene.digital_elevation.digital_dim, pcontact.digital_wall, self.cplist, self.contact_model)
    
    def no_operation(self, sims, mscene, dscene, pcontact):
        pass