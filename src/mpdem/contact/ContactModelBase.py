import os
import math
import numpy as np

from src.dem.BaseStruct import CoupledContactTable, HistoryContactTable
from src.dem.contact.ContactKernel import *
from src.mpm.SceneManager import myScene as MPMScene
from src.utils.ObjectIO import DictIO


class ContactModelBase(object):
    def __init__(self):
        self.contact_list_initialize = None
        self.resolve = None
        self.update_contact_table = None
        self.cplist = None
        self.hist_cplist = None
        self.surfaceProps = None
        self.null_mode = True

    def manage_function(self, object_type, contact_method=None):
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
                if contact_method == 'P2P':
                    self.resolve = self.tackle_particle_particle_contact_cplist
                elif contact_method == 'implicitP2S':
                    self.resolve = self.tackle_modified_particle_particle_contact_cplist
                self.update_contact_table = self.update_particle_particle_contact_table
            elif object_type == "wall":
                self.resolve = self.tackle_particle_wall_contact_cplist
                self.update_contact_table = self.update_particle_wall_contact_table
            if self.resolve is None:
                raise RuntimeError("Internal error!")

    def collision_initialize(self, parameter, max_object_pairs):
        if not self.null_mode:
            self.cplist = CoupledContactTable.field(shape=int(math.ceil(parameter * max_object_pairs)))
            self.hist_cplist = HistoryContactTable.field(shape=int(math.ceil(parameter * max_object_pairs)))

    def get_componousID(self, max_material_num, materialID1, materialID2):
        return int(materialID1 * max_material_num + materialID2)
    
    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        raise NotImplementedError
    
    def no_add_property(self, max_material_num, materialID1, materialID2, property):
        return self.get_componousID(max_material_num, materialID1, materialID2)
    
    def calcu_critical_timestep(self, mscene, dscene, max_material_num):
        raise NotImplementedError
    
    def no_critical_timestep(self, mscene, dscene, max_material_num):
        return 1e-3
    
    def get_ppcontact_output(self, contact_path, current_time, current_print, scene, pcontact):
        raise NotImplementedError
    
    def get_pwcontact_output(self, contact_path, current_time, current_print, scene, pcontact):
        raise NotImplementedError
    
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
    #              Particle Contact Matrix Resolve              #
    # ========================================================= # 
    def update_particle_particle_contact_table(self, sims, mscene, dscene, pcontact):
        raise NotImplementedError

    def update_particle_wall_contact_table(self, sims, mscene, dscene, pcontact):
        raise NotImplementedError

    def tackle_particle_particle_contact_cplist(self, sims, mscene, dscene, pcontact):
        raise NotImplementedError
    
    def tackle_modified_particle_particle_contact_cplist(self, sims, mscene, dscene, pcontact):
        raise NotImplementedError
    
    def tackle_particle_wall_contact_cplist(self, sims, mscene, dscene, pcontact):
        raise NotImplementedError
    
    def no_operation(self, sims, mscene, dscene, pcontact):
        pass
