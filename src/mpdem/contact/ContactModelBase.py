import os
import numpy as np

from src.mpdem.contact.MultiLinkedCell import MultiLinkedCell
from src.mpdem.Simulation import Simulation
from src.mpm.SceneManager import myScene as MPMScene
from src.dem.structs.BaseStruct import CoupledContactTable, HistoryContactTable
from src.dem.SceneManager import myScene as DEMScene
from src.dem.contact.ContactKernel import *
from src.mpm.SceneManager import myScene as MPMScene
from src.utils.TypeDefination import u1
from src.utils.ObjectIO import DictIO
from src.utils.linalg import round32
from src.utils.linalg import no_operation


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
        self.first_run = True

    def manage_function(self, object_type, enhanced_coupling, dem_scheme=None):
        self.resolve = no_operation
        self.update_contact_table = no_operation
        self.add_surface_properties = self.no_add_property
        self.calcu_critical_timesteps = self.no_critical_timestep
        if not self.null_model:
            self.add_surface_properties = self.add_surface_property
            self.calcu_critical_timesteps = self.calcu_critical_timestep
            if object_type == "particle":
                if dem_scheme == "DEM":
                    if enhanced_coupling:
                        self.resolve = self.tackle_enhanced_coupling_particle_particle_contact_cplist
                    else:
                        self.resolve = self.tackle_particle_particle_contact_cplist
                elif dem_scheme == "LSDEM":
                    self.resolve = self.tackle_particle_LSparticle_contact_cplist
                self.update_contact_table = self.update_particle_particle_contact_table
                if enhanced_coupling:
                    if dem_scheme == "DEM":
                        self.update_contact_table = self.update_enhanced_particle_particle_contact_table
            elif object_type == "wall":
                self.resolve = self.tackle_particle_wall_contact_cplist
                self.update_contact_table = self.update_particle_wall_contact_table

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

    def collision_initialize(self, enhanced_coupling, max_object_pairs):
        if not self.null_model and self.first_run:
            self.cplist = CoupledContactTable.field(shape=int(max_object_pairs))
            self.hist_cplist = HistoryContactTable.field(shape=int(max_object_pairs))
            if enhanced_coupling:
                self.contact_flag = ti.field(u1)
                ti.root.dense(ti.i, round32(max_object_pairs)//32).quant_array(ti.i, dimensions=32, max_num_bits=32).place(self.contact_flag)  
        self.first_run = False

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
        end1 = np.ascontiguousarray(self.cplist.endID1.to_numpy()[0:neighbor_list[scene.couplingNum[0]]])
        end2 = np.ascontiguousarray(self.cplist.endID2.to_numpy()[0:neighbor_list[scene.couplingNum[0]]])
        oldTangentialOverlap = np.ascontiguousarray(self.cplist.oldTangOverlap.to_numpy()[0:neighbor_list[scene.couplingNum[0]]])
        return end1, end2, oldTangentialOverlap
    
    def restart(self, pcontact, file_number, contact, is_particle_particle=True):
        if not contact is None:
            if not os.path.exists(contact):
                raise EOFError("Invaild contact path")
            
            if is_particle_particle:
                contact_info = np.load(contact + "/DEMPMContactPP{0:06d}.npz".format(file_number), allow_pickle=True) 
                self.rebuild_ppcontact_list(pcontact, contact_info)
            else:
                contact_info = np.load(contact + "/DEMPMContactPW{0:06d}.npz".format(file_number), allow_pickle=True) 
                self.rebuild_pwcontact_list(pcontact, contact_info)
            
    def rebuild_contact_list(self, contact_info):
        object_object = DictIO.GetEssential(contact_info, "contact_num")
        DstID = DictIO.GetEssential(contact_info, "end2")
        oldTangOverlap = DictIO.GetEssential(contact_info, "oldTangentialOverlap")
        return object_object, DstID, oldTangOverlap

    def get_ppcontact_output(self, contact_path, current_time, current_print, scene: MPMScene, pcontact: MultiLinkedCell):
        particleParticle = np.ascontiguousarray(pcontact.hist_particle_particle.to_numpy()[0:scene.couplingNum[0] + 1])
        end1, end2, oldTangentialOverlap = self.get_contact_output(scene, particleParticle)
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleParticle, end1=end1, end2=end2, oldTangentialOverlap=oldTangentialOverlap)
        
    def get_pwcontact_output(self, contact_path, current_time, current_print, scene: MPMScene, pcontact: MultiLinkedCell):
        particleWall = np.ascontiguousarray(pcontact.hist_particle_wall.to_numpy()[0:scene.couplingNum[0] + 1])
        end1, end2, oldTangentialOverlap = self.get_contact_output(scene, particleWall)
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleWall, end1=end1, end2=end2, oldTangentialOverlap=oldTangentialOverlap)
    
    def rebuild_ppcontact_list(self, pcontact: MultiLinkedCell, contact_info):
        object_object, DstID, oldTangOverlap = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_history_contact_list(self.cplist, pcontact.hist_particle_particle, object_object, DstID, oldTangOverlap)

    def rebuild_pwcontact_list(self, pcontact: MultiLinkedCell, contact_info):
        object_object, DstID, oldTangOverlap = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_history_contact_list(self.cplist, pcontact.hist_particle_wall, object_object, DstID, oldTangOverlap)
    
    # ========================================================= #
    #              Particle Contact Matrix Resolve              #
    # ========================================================= # 
    def update_particle_particle_contact_table(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        copy_contact_table(pcontact.hist_particle_particle, int(mscene.couplingNum[0]), self.cplist, self.hist_cplist)
        update_contact_table_(sims.potential_particle_num, int(mscene.couplingNum[0]), pcontact.particle_particle, pcontact.potential_list_particle_particle, self.cplist)
        kernel_inherit_contact_history(int(mscene.couplingNum[0]), self.cplist, self.hist_cplist, pcontact.particle_particle, pcontact.hist_particle_particle)

    def update_enhanced_particle_particle_contact_table(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        copy_contact_table(pcontact.hist_particle_particle, int(dscene.particleNum[0]), self.cplist, self.hist_cplist)
        update_contact_table_(sims.potential_particle_num, int(dscene.particleNum[0]), pcontact.particle_particle, pcontact.potential_list_particle_particle, self.cplist)
        kernel_inherit_contact_history(int(dscene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_particle, pcontact.hist_particle_particle)

    def update_particle_wall_contact_table(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        copy_contact_table(pcontact.hist_particle_wall, int(mscene.couplingNum[0]), self.cplist, self.hist_cplist)
        update_contact_table_(sims.wall_coordination_number, int(mscene.couplingNum[0]), pcontact.particle_wall, pcontact.potential_list_particle_wall, self.cplist)
        kernel_inherit_contact_history(int(mscene.couplingNum[0]), self.cplist, self.hist_cplist, pcontact.particle_wall, pcontact.hist_particle_wall)

    def tackle_particle_particle_contact_cplist(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        kernel_particle_particle_force_assemble_(int(mscene.couplingNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, mscene.particle, dscene.particle, 
                                                 self.cplist, pcontact.hist_particle_particle, self.contact_model)

    def tackle_enhanced_coupling_particle_particle_contact_cplist(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        update_point_particle_flag(int(dscene.particleNum[0]), mscene.particle, pcontact.hist_particle_particle, self.cplist, self.contact_flag)
        kernel_enhanced_coupling_particle_particle_force_assemble_(int(dscene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, dscene.particle, mscene.particle, 
                                                 self.cplist, pcontact.hist_particle_particle, self.contact_flag, self.contact_model)

    def tackle_particle_wall_contact_cplist(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        kernel_particle_wall_force_assemble_(int(mscene.couplingNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, mscene.particle, dscene.wall, 
                                             self.cplist, pcontact.hist_particle_wall, self.contact_model)
        
    def tackle_particle_ISparticle_contact_cplist(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        kernel_particle_ISparticle_force_assemble_(int(mscene.couplingNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, mscene.particle, dscene.rigid, dscene.surface, self.cplist, pcontact.hist_particle_particle, self.contact_model, self.iterative)
        
    def tackle_particle_LSparticle_contact_cplist(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        kernel_particle_LSparticle_force_assemble_(int(mscene.couplingNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, mscene.particle, dscene.rigid, dscene.rigid_grid, 
                                                   dscene.box, self.cplist, pcontact.hist_particle_particle, self.contact_model)