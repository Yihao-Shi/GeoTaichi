import taichi as ti
import numpy as np

from src.dem.contact.Linear import LinearSurfaceProperty
from src.dem.SceneManager import myScene as DEMScene
from src.mpdem.contact.ContactKernel import *
from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpdem.contact.MultiLinkedCell import MultiLinkedCell
from src.mpdem.Simulation import Simulation
from src.mpm.SceneManager import myScene as MPMScene
from src.utils.ObjectIO import DictIO
from src.utils.ScalarFunction import EffectiveValue


class LinearModel(ContactModelBase):
    def __init__(self, max_material_num) -> None:
        super().__init__()
        self.surfaceProps = LinearSurfaceProperty.field(shape=max_material_num * max_material_num)
        self.null_mode = False

    def calcu_critical_timestep(self, mscene: MPMScene, dscene: DEMScene, max_material_num):
        mass = min(mscene.find_particle_min_mass(), dscene.find_particle_min_mass())
        stiffness = self.find_max_stiffness(max_material_num)
        return ti.sqrt(mass / stiffness)

    def find_max_stiffness(self, max_material_num):
        maxstiff = 0.
        for materialID1 in range(max_material_num):
            for materialID2 in range(max_material_num):
                componousID = self.get_componousID(max_material_num, materialID1, materialID2)
                if self.surfaceProps[componousID].kn > 0.:
                    maxstiff = ti.max(ti.max(maxstiff, self.surfaceProps[componousID].kn), self.surfaceProps[componousID].ks)
        return maxstiff
    
    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        kn = DictIO.GetEssential(property, 'NormalStiffness')
        ks = DictIO.GetEssential(property, 'TangentialStiffness')
        mu = DictIO.GetEssential(property, 'Friction')
        ndratio = DictIO.GetEssential(property, 'NormalViscousDamping')
        sdratio = DictIO.GetEssential(property, 'TangentialViscousDamping')
        componousID = 0
        if materialID1 == materialID2:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, mu, ndratio, sdratio)
        else:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, mu, ndratio, sdratio)
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, mu, ndratio, sdratio)
        return componousID


    def inherit_surface_property(self, max_material_num, materialID1,  materialID2, property1, property2):
        kn1 = DictIO.GetEssential(property1, 'NormalStiffness')
        ks1 = DictIO.GetEssential(property1, 'TangentialStiffness')
        mu1 = DictIO.GetEssential(property1, 'Friction')
        dnratio1 = DictIO.GetEssential(property1, 'NormalViscousDamping')
        dsratio1 = DictIO.GetEssential(property1, 'TangentialViscousDamping')

        kn2 = DictIO.GetEssential(property2, 'NormalStiffness')
        ks2 = DictIO.GetEssential(property2, 'TangentialStiffness')
        mu2 = DictIO.GetEssential(property2, 'Friction')
        dnratio2 = DictIO.GetEssential(property2, 'NormalViscousDamping')
        dsratio2 = DictIO.GetEssential(property2, 'TangentialViscousDamping')
        
        kn = EffectiveValue(kn1, kn2)
        ks = EffectiveValue(ks1, ks2)
        mu = ti.min(mu1, mu2)
        ndratio = ti.min(dnratio1, dnratio2)
        sdratio = ti.min(dsratio1, dsratio2)
        componousID = 0
        if materialID1 == materialID2:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, mu, ndratio, sdratio)
        else:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, mu, ndratio, sdratio)
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, mu, ndratio, sdratio)
        return componousID
    
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
        oldTangentialOverlap = np.ascontiguousarray(self.cplist.oldTangOverlap.to_numpy()[0:pcontact.particle_particle[scene.particleNum[0]]])
        particleWall = np.ascontiguousarray(pcontact.hist_particle_wall.to_numpy()[0:pcontact.particle_particle[scene.particleNum[0]]])
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleWall, end1=end1, end2=end2, oldTangentialOverlap=oldTangentialOverlap)
    
    def rebuild_ppcontact_list(self, pcontact: MultiLinkedCell, contact_info):
        object_object, particle_number, DstID, oldTangOverlap = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_history_contact_list(self.hist_cplist, pcontact.hist_particle_particle, object_object, DstID, oldTangOverlap)

    def rebuild_pwcontact_list(self, pcontact: MultiLinkedCell, contact_info):
        object_object, particle_number, DstID, oldTangOverlap = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_history_contact_list(self.hist_cplist, pcontact.hist_particle_wall, object_object, DstID, oldTangOverlap)

    # ========================================================= #
    #              Particle Contact Matrix Resolve              #
    # ========================================================= # 
    def update_particle_particle_contact_table(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        copy_contact_table(pcontact.particle_particle, int(mscene.particleNum[0]), self.cplist, self.hist_cplist)
        update_contact_table_(sims.potential_particle_num, pcontact.particle_particle, pcontact.potential_list_particle_particle, self.cplist, int(mscene.particleNum[0]))
        kernel_inherit_contact_history(int(mscene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_particle, pcontact.hist_particle_particle)

    def update_particle_wall_contact_table(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        copy_contact_table(pcontact.particle_wall, int(mscene.particleNum[0]), self.cplist, self.hist_cplist)
        update_wall_contact_table_(sims.wall_coordination_number, pcontact.particle_wall, pcontact.potential_list_particle_wall, self.cplist, int(mscene.particleNum[0]))
        kernel_inherit_contact_history(int(mscene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_wall, pcontact.hist_particle_wall)

    def tackle_particle_particle_contact_cplist(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        kernel_particle_particle_force_assemble_(int(mscene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, mscene.particle, dscene.particle, 
                                                 self.cplist, pcontact.particle_particle)

    def tackle_particle_wall_contact_cplist(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        kernel_particle_wall_force_assemble_(int(mscene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, mscene.particle, dscene.wall, 
                                             self.cplist, pcontact.particle_wall)


@ti.kernel
def kernel_rebulid_history_contact_list(hist_cplist: ti.template(), hist_object_object: ti.template(), object_object: ti.types.ndarray(), 
                                        dst: ti.types.ndarray(), oldTangOverlap: ti.types.ndarray()):
    for i in range(object_object.shape[0]):
        hist_object_object[i] = object_object[i]

    for cp in range(object_object[object_object.shape[0] - 1]):
        hist_cplist[cp].DstID = dst[cp]
        hist_cplist[cp].oldTangOverlap = oldTangOverlap[cp]
