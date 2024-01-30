import taichi as ti
import numpy as np

from src.dem.contact.HertzMindlin import HertzMindlinSurfaceProperty
from src.dem.SceneManager import myScene as DEMScene
from src.mpdem.contact.ContactKernel import *
from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpdem.contact.MultiLinkedCell import MultiLinkedCell
from src.mpdem.Simulation import Simulation
from src.mpm.SceneManager import myScene as MPMScene
from src.utils.constants import PI
from src.utils.ObjectIO import DictIO


class HertzMindlinModel(ContactModelBase):
    def __init__(self, max_material_num) -> None:
        super().__init__()
        self.surfaceProps = HertzMindlinSurfaceProperty.field(shape=max_material_num)
        self.null_mode = False

    def calcu_critical_timestep(self, mscene: MPMScene, dscene: DEMScene, max_material_num):
        radius = min(mscene.find_particle_min_mass(), dscene.find_particle_min_mass())
        density = min(mscene.find_min_density(), dscene.find_min_density())
        modulus, possion = self.find_max_mparas(max_material_num)
        return PI * radius * ti.sqrt(density / modulus) / (0.01631 * possion + 0.8766)

    def find_max_mparas(self, max_material_num):
        maxmodulus, maxpossion = 0., 0.
        for materialID1 in range(max_material_num):
            for materialID2 in range(max_material_num):
                componousID = self.get_componousID(max_material_num, materialID1, materialID2)
                if self.surfaceProps[componousID].ShearModulus > 0.:
                    possion = (4 * self.surfaceProps[componousID].ShearModulus - self.surfaceProps[componousID].YoungModulus) / \
                              (2 * self.surfaceProps[componousID].ShearModulus - self.surfaceProps[componousID].YoungModulus)
                    modulus = 2 * self.surfaceProps[componousID].ShearModulus * (2 - possion)
                    maxpossion = ti.max(maxpossion, possion)
                    maxmodulus = ti.max(maxpossion, modulus)
        return maxmodulus, maxpossion
    
    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        modulus = DictIO.GetEssential(property, 'ShearModulus')
        possion = DictIO.GetEssential(property, 'Possion')
        YoungModulus = modulus * 2 * (1 + possion)
        ShearModulus = modulus
        mu = DictIO.GetEssential(property, 'Friction')
        restitution = DictIO.GetEssential(property, 'Restitution')
        componousID = 0
        if restitution < 1e-16:
            restitution = 0.
        else:
            restitution = -ti.log(restitution) / ti.sqrt(PI * PI + ti.log(restitution) * ti.log(restitution))
        if materialID1 == materialID2:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mu, restitution)
        else:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mu, restitution)
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mu, restitution)
        return componousID


    def inherit_surface_property(self, max_material_num, materialID1, materialID2, property1, property2):
        modulus1 = DictIO.GetEssential(property1, 'ShearModulus')
        possion1 = DictIO.GetEssential(property1, 'Possion')
        mu1 = DictIO.GetEssential(property1, 'Friction')
        restitution1 = DictIO.GetEssential(property1, 'Restitution')

        modulus2 = DictIO.GetEssential(property2, 'ShearModulus')
        possion2 = DictIO.GetEssential(property2, 'Possion')
        mu2 = DictIO.GetEssential(property2, 'Friction')
        restitution2 = DictIO.GetEssential(property2, 'Restitution')
        
        YoungModulus = 1. / ((1 - possion1) / (2. * modulus1) + (1 - possion2) / (2. * modulus2))
        ShearModulus = 1. / ((2 - possion1) / modulus1 + (2 - possion2) / modulus2)
        restitution = ti.min(restitution1, restitution2)
        componousID = 0
        if restitution < 1e-16:
            restitution = 0.
        else:
            restitution = -ti.log(restitution) / ti.sqrt(PI * PI + ti.log(restitution) * ti.log(restitution))
        mu = ti.min(mu1, mu2)
        if materialID1 == materialID2:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mu, restitution)
        else:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mu, restitution)
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mu, restitution)
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
        if particle_number > pcontact.hist_particle_particle.shape[0]:
            raise RuntimeError("/max_particle_num/ should be enlarged")
        kernel_rebulid_history_contact_list(self.hist_cplist, pcontact.hist_particle_particle, object_object, DstID, oldTangOverlap)

    def rebuild_pwcontact_list(self, pcontact: MultiLinkedCell, contact_info):
        object_object, particle_number, DstID, oldTangOverlap = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        if particle_number > pcontact.hist_particle_wall.shape[0]:
            raise RuntimeError("/max_particle_num/ should be enlarged")
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
