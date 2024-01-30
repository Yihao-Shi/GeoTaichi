import taichi as ti
import os, math
import numpy as np

from src.dem.BaseStruct import (ContactTable, HistoryContactTable)
from src.dem.contact.ContactKernel import *
from src.dem.SceneManager import myScene
from src.utils.ObjectIO import DictIO
from src.utils.ScalarFunction import round32
from src.utils.TypeDefination import u1
# from src.utils.PrefixSum import PrefixSumExecutor

class ContactModelBase(object):
    def __init__(self):
        self.contact_list_initialize = None
        self.resolve = None
        self.update_contact_table = None
        self.cplist = None
        self.hist_cplist = None
        self.contact_active = None
        self.deactivate_exist = None
        self.surfaceProps = None
        self.null_mode = True

    def manage_function(self, object_type, work_type):
        self.contact_list_initialize = self.no_contact_list_initial
        self.resolve = self.no_operation
        self.update_contact_table = self.no_operation
        self.add_surface_properties = self.no_add_property
        self.calcu_critical_timesteps = self.no_critical_timestep
        if not self.null_mode:
            self.contact_list_initialize = self.contact_list_initial
            self.add_surface_properties = self.add_surface_property
            self.calcu_critical_timesteps = self.calcu_critical_timestep
            if object_type == "particle":
                if work_type == 0 or work_type == 1:
                    self.resolve = self.tackle_particle_particle_contact_bit_table
                elif work_type == 2:
                    self.resolve = self.tackle_particle_particle_contact_cplist
                    self.update_contact_table = self.update_particle_particle_contact_table
            elif object_type == "wall":
                if work_type == 0 or work_type == 1:
                    self.resolve = self.tackle_particle_wall_contact_bit_table
                elif work_type == 2:
                    self.resolve = self.tackle_particle_wall_contact_cplist
                    self.update_contact_table = self.update_particle_wall_contact_table
            if self.resolve is None:
                raise RuntimeError("Internal error!")

    def collision_initialize(self, parameter, work_type, max_object_pairs, object_num1, object_num2):
        if not self.null_mode:
            self.cplist = ContactTable.field(shape=int(math.ceil(parameter * max_object_pairs)))
            '''
            self.active_contact = ti.field(int, shape=int(parameter * max_object_pairs) + 1)
            self.active_pse = PrefixSumExecutor(int(parameter * max_object_pairs) + 1)
            self.compact_table = ti.field(int, shape=int(parameter * max_object_pairs))
            '''
            if work_type == 0 or work_type == 1:
                self.deactivate_exist = ti.field(ti.u8, shape=())
                self.contact_active = ti.field(u1)
                ti.root.dense(ti.i, round32(object_num1 * object_num2)//32).quant_array(ti.i, dimensions=32, max_num_bits=32).place(self.contact_active)
                self.hist_cplist = ContactTable.field(shape=int(math.ceil(parameter * max_object_pairs)))
            elif work_type == 2:
                self.hist_cplist = HistoryContactTable.field(shape=int(math.ceil(parameter * max_object_pairs)))
                #self.active_index = ti.field(int, shape=int(0.5 * max_object_pairs))

    def get_componousID(self, max_material_num, materialID1, materialID2):
        return int(materialID1 * max_material_num + materialID2)
    
    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        raise NotImplementedError
    
    def no_add_property(self, max_material_num, materialID1, materialID2, property):
        return self.get_componousID(max_material_num, materialID1, materialID2)
    
    def calcu_critical_timestep(self, scene, max_material_num):
        raise NotImplementedError
    
    def no_critical_timestep(self, scene, max_material_num):
        return 1e-3
    
    def get_ppcontact_output(self, contact_path, current_time, current_print, scene, pcontact):
        raise NotImplementedError
    
    def get_pwcontact_output(self, contact_path, current_time, current_print, scene, pcontact):
        raise NotImplementedError
    
    def get_contact_output(self, scene: myScene, neighbor_list):
        end1 = np.ascontiguousarray(self.cplist.endID1.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        end2 = np.ascontiguousarray(self.cplist.endID2.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        normal_force = np.ascontiguousarray(self.cplist.cnforce.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        tangential_force = np.ascontiguousarray(self.cplist.csforce.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        oldTangentialOverlap = np.ascontiguousarray(self.cplist.oldTangOverlap.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        return end1, end2, normal_force, tangential_force, oldTangentialOverlap
    
    def update_properties(self, max_material_num, materialID1, materialID2, property_name, value, override):
        if materialID1 == materialID2:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
        else:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.update_property(componousID, property_name, value, override)
            componousID = self.get_componousID(max_material_num, materialID2, materialID1)
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
        particle_number = object_object[object_object.shape[0] - 1]
        DstID = DictIO.GetEssential(contact_info, "end2")
        oldTangOverlap = DictIO.GetEssential(contact_info, "oldTangentialOverlap")
        return object_object, particle_number, DstID, oldTangOverlap

    def rebuild_ppcontact_list(self, pcontact, contact_info):
        raise NotImplementedError
    
    def rebuild_pwcontact_list(self, pcontact, contact_info):
        raise NotImplementedError
    
    def contact_list_initial(self, particleNum, object_object, hist_object_object):
        copy_histcp2cp(int(particleNum[0]), object_object, hist_object_object, self.cplist, self.hist_cplist)
        copy_contact_table(object_object, int(particleNum[0]), self.cplist, self.hist_cplist)

    def no_contact_list_initial(self, particleNum, object_object, hist_object_object):
        pass

    
    # ========================================================= #
    #                   Bit Table Resolve                       #
    # ========================================================= # 
    def tackle_particle_particle_contact_bit_table(self, sims, scene, pcontact):
        raise NotImplementedError
    
    def tackle_particle_wall_contact_bit_table(self, sims, scene, pcontact):
        raise NotImplementedError
    
    
    # ========================================================= #
    #              Particle Contact Matrix Resolve              #
    # ========================================================= # 
    def update_particle_particle_contact_table(self, sims, scene, pcontact):
        raise NotImplementedError

    def update_particle_wall_contact_table(self, sims, scene, pcontact):
        raise NotImplementedError

    def tackle_particle_particle_contact_cplist(self, sims, scene, pcontact):
        raise NotImplementedError
    
    def tackle_particle_wall_contact_cplist(self, sims, scene, pcontact):
        raise NotImplementedError

    def no_operation(self, sims, scene, pcontact):
        pass
