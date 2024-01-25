import taichi as ti
import numpy as np

from geotaichi.dem.contact.ContactKernel import *
from geotaichi.dem.contact.ContactModelBase import ContactModelBase
from geotaichi.dem.neighbor.NeighborBase import NeighborBase
from geotaichi.dem.SceneManager import myScene
from geotaichi.dem.Simulation import Simulation
from geotaichi.utils.constants import ZEROVEC3f
from geotaichi.utils.ObjectIO import DictIO
from geotaichi.utils.ScalarFunction import EffectiveValue
from geotaichi.utils.TypeDefination import vec3f
from geotaichi.utils.VectorFunction import Normalize


class LinearModel(ContactModelBase):
    def __init__(self, max_material_num) -> None:
        super().__init__()
        self.surfaceProps = LinearSurfaceProperty.field(shape=max_material_num * max_material_num)

    def calcu_critical_timestep(self, scene: myScene, max_material_num):
        mass = scene.find_particle_min_mass()
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
            componousID = self.get_componousID(max_material_num, materialID2, materialID1)
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
            componousID = self.get_componousID(max_material_num, materialID2, materialID1)
            self.surfaceProps[componousID].add_surface_property(kn, ks, mu, ndratio, sdratio)
        return componousID
    
    def update_property(self, componousID, property_name, value, override):
        if override:
            override = 1
        else:
            override = 0

        if property_name == "NormalStiffness":
            self.surfaceProps[componousID].kn = override * self.surfaceProps[componousID].kn + value
        elif property_name == "TangentialStiffness":
            self.surfaceProps[componousID].ks = override * self.surfaceProps[componousID].ks + value
        elif property_name == "Friction":
            self.surfaceProps[componousID].mu = override * self.surfaceProps[componousID].mu + value
        elif property_name == "NormalViscousDamping":
            self.surfaceProps[componousID].ndratio = override * self.surfaceProps[componousID].ndratio + value
        elif property_name == "TangentialViscousDamping":
            self.surfaceProps[componousID].sdratio = override * self.surfaceProps[componousID].sdratio + value
    
    def get_ppcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        end1, end2, normal_force, tangential_force, oldTangentialOverlap = self.get_contact_output(scene, pcontact.particle_particle)
        particleParticle = np.ascontiguousarray(pcontact.particle_particle.to_numpy()[0:scene.particleNum[0] + 1])
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleParticle, end1=end1, end2=end2, normal_force=normal_force, 
                                                      tangential_force=tangential_force, oldTangentialOverlap=oldTangentialOverlap)
        
    def get_pwcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        end1, end2, normal_force, tangential_force, oldTangentialOverlap = self.get_contact_output(scene, pcontact.particle_wall)
        particleWall = np.ascontiguousarray(pcontact.particle_wall.to_numpy()[0:scene.particleNum[0] + 1])
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleWall, end1=end1, end2=end2, normal_force=normal_force, 
                                                      tangential_force=tangential_force, oldTangentialOverlap=oldTangentialOverlap)
    
    def rebuild_ppcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, particle_number, DstID, oldTangOverlap = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_history_contact_list(self.hist_cplist, pcontact.hist_particle_particle, object_object, DstID, oldTangOverlap)

    def rebuild_pwcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, particle_number, DstID, oldTangOverlap = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_history_contact_list(self.hist_cplist, pcontact.hist_particle_wall, object_object, DstID, oldTangOverlap)

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
        copy_contact_table(pcontact.particle_particle, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
        update_contact_table_(sims.potential_particle_num, int(scene.particleNum[0]), pcontact.particle_particle, pcontact.potential_list_particle_particle, self.cplist)
        kernel_inherit_contact_history(int(scene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_particle, pcontact.hist_particle_particle)
         
    def update_particle_wall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_contact_table(pcontact.particle_wall, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
        update_wall_contact_table_(sims.wall_coordination_number, int(scene.particleNum[0]), pcontact.particle_wall, pcontact.potential_list_particle_wall, self.cplist)
        kernel_inherit_contact_history(int(scene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_wall, pcontact.hist_particle_wall)
        
    def tackle_particle_particle_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_particle_particle_force_assemble_(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.particle, self.cplist, pcontact.particle_particle)
        
    def tackle_particle_wall_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_particle_wall_force_assemble_(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.particle, scene.wall, self.cplist, pcontact.particle_wall)


@ti.dataclass
class LinearSurfaceProperty:
    kn: float
    ks: float
    mu: float
    ndratio: float
    sdratio: float

    def add_surface_property(self, kn, ks, mu, ndratio, sdratio):
        self.kn = kn
        self.ks = ks
        self.mu = mu
        self.ndratio = ndratio
        self.sdratio = sdratio

    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, '-'))
        print('Contact model: Linear Contact Model')
        print(f'MaterialID{matID1} < --- > MaterialID{matID2}')
        print('Contact normal stiffness: = ', self.kn)
        print('Contact tangential stiffness: = ', self.ks)
        print('Friction coefficient = ', self.mu)
        print('Viscous damping coefficient = ', self.ndratio)
        print('Viscous damping coefficient = ', self.sdratio, '\n')

    @ti.func
    def _get_equivalent_stiffness(self, end1, end2, particle, wall):
        pos1, pos2 = particle[end1].x, wall[end2]._get_center()
        particle_rad, norm = particle[end1].rad, wall[end2].norm
        distance = (pos1 - pos2).dot(norm)
        gapn = distance - particle_rad
        fraction = ti.abs(wall[end2].processCircleShape(pos1, distance, -gapn))
        return fraction * self.kn

    # ========================================================= #
    #                   Particle-Particle                       #
    # ========================================================= # 
    @ti.func
    def _particle_particle_force_assemble(self, nc, dt, particle, cplist):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        pos1, pos2 = particle[end1].x, particle[end2].x
        rad1, rad2 = particle[end1].rad, particle[end2].rad
        gapn = (pos1 - pos2).norm() - (rad1 + rad2)
        if gapn < 0.:
            norm = (pos1 - pos2).normalized()
            
            mass1, mass2 = particle[end1].m, particle[end2].m
            vel1, vel2 = particle[end1].v, particle[end2].v
            w1, w2 = particle[end1].w, particle[end2].w

            m_eff = EffectiveValue(mass1, mass2)
            kn, ks = self.kn, self.ks
            ndratio, sdratio = self.ndratio, self.sdratio
            miu = self.mu

            cpos = pos2 + (rad2 + 0.5 * gapn) * norm
            v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))
            vn = v_rel.dot(norm) 
            vs = v_rel - v_rel.dot(norm) * norm

            normal_contact_force = -kn * gapn 
            normal_damping_force = -2 * ndratio * ti.sqrt(m_eff * kn) * vn
            normal_force = (normal_contact_force + normal_damping_force) * norm

            tangOverlapOld = cplist[nc].oldTangOverlap
            tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
            tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
            trial_ft = -ks * tangOverTemp
            tang_damping_force = -2 * sdratio * ti.sqrt(m_eff * ks) * vs
            
            fric = miu * ti.abs(normal_contact_force + normal_damping_force)
            tangential_force = ZEROVEC3f
            if trial_ft.norm() > fric:
                tangential_force = fric * trial_ft.normalized()
                tangOverTemp = -tangential_force / ks
            else:
                tangential_force = trial_ft + tang_damping_force
            
            Ftotal = normal_force + tangential_force
            moment = tangential_force.cross(norm)

            cplist[nc].cnforce = normal_force
            cplist[nc].csforce = tangential_force
            cplist[nc].oldTangOverlap = tangOverTemp

            particle[end1]._update_contact_interaction(Ftotal, moment * (rad1 + 0.5 * gapn))
            particle[end2]._update_contact_interaction(-Ftotal, moment * (rad2 + 0.5 * gapn))
        else:
            cplist[nc].cnforce = ZEROVEC3f
            cplist[nc].csforce = ZEROVEC3f
            cplist[nc].oldTangOverlap = ZEROVEC3f

    @ti.func
    def _coupled_particle_force_assemble(self, nc, dt, particle1, particle2, cplist):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        pos1, pos2 = particle1[end1].x, particle2[end2].x
        rad1, rad2 = particle1[end1].rad, particle2[end2].rad
        gapn = (pos1 - pos2).norm() - (rad1 + rad2)
        if gapn < 0.:
            norm = (pos1 - pos2).normalized()
            
            mass1, mass2 = particle1[end1].m, particle2[end2].m
            vel1, vel2 = particle1[end1].v, particle2[end2].v
            w1, w2 = ZEROVEC3f, particle2[end2].w

            m_eff = EffectiveValue(mass1, mass2)
            kn, ks = self.kn, self.ks
            ndratio, sdratio = self.ndratio, self.sdratio
            miu = self.mu

            cpos = pos2 + (rad2 + 0.5 * gapn) * norm
            v_rel = vel1 - (vel2 + w2.cross(cpos - pos2))
            vn = v_rel.dot(norm) 
            vs = v_rel - v_rel.dot(norm) * norm

            normal_contact_force = -kn * gapn 
            normal_damping_force = -2 * ndratio * ti.sqrt(m_eff * kn) * vn
            normal_force = (normal_contact_force + normal_damping_force) * norm

            tangOverlapOld = cplist[nc].oldTangOverlap
            tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
            tangOverTemp = vs * dt[None]  + tangOverlapOld.norm() * Normalize(tangOverlapRot)
            trial_ft = -ks * tangOverTemp
            tang_damping_force = -2 * sdratio * ti.sqrt(m_eff * ks) * vs
            
            fric = miu * ti.abs(normal_contact_force + normal_damping_force)
            tangential_force = ZEROVEC3f
            if trial_ft.norm() > fric:
                tangential_force = fric * trial_ft.normalized()
                tangOverTemp = -tangential_force / ks
            else:
                tangential_force = trial_ft + tang_damping_force
            
            Ftotal = normal_force + tangential_force
            resultant_momentum = Ftotal.cross(cpos - pos2) 

            cplist[nc].oldTangOverlap = tangOverTemp
            particle1[end1]._update_contact_interaction(Ftotal)
            particle2[end2]._update_contact_interaction(-Ftotal, resultant_momentum)
        else:
            cplist[nc].oldTangOverlap = ZEROVEC3f


    # ========================================================= #
    #                      Particle-Wall                        #
    # ========================================================= # 
    @ti.func
    def _particle_wall_force_assemble(self, nc, dt, particle, wall, cplist):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        pos1 = particle[end1].x
        particle_rad, norm = particle[end1].rad, wall[end2].norm
        distance = wall[end2]._get_norm_distance(pos1)
        gapn = distance - particle_rad
        if gapn < 0.:
            vel1, vel2 = particle[end1].v, wall[end2]._get_velocity()
            w1 = particle[end1].w

            m_eff = particle[end1].m
            kn, ks = self.kn, self.ks
            ndratio, sdratio = self.ndratio, self.sdratio
            miu = self.mu

            cpos = wall[end2]._point_projection(pos1) - 0.5 * gapn * norm
            v_rel = vel1 + w1.cross(cpos - pos1) - vel2 
            vn = v_rel.dot(norm) 
            vs = v_rel - v_rel.dot(norm) * norm

            normal_contact_force = -kn * gapn 
            normal_damping_force = -2 * ndratio * ti.sqrt(m_eff * kn) * vn
            normal_force = (normal_contact_force + normal_damping_force) * norm
            
            tangOverlapOld = cplist[nc].oldTangOverlap
            tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
            tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
            trial_ft = -ks * tangOverTemp
            tang_damping_force = -2 * sdratio * ti.sqrt(m_eff * ks) * vs
            
            fric = miu * ti.abs(normal_contact_force + normal_damping_force)
            tangential_force = ZEROVEC3f
            if trial_ft.norm() > fric:
                tangential_force = fric * trial_ft.normalized()
                tangOverTemp = -tangential_force / ks
            else:
                tangential_force = trial_ft + tang_damping_force
                
            fraction = ti.abs(wall[end2].processCircleShape(pos1, distance, -gapn))
            Ftotal = fraction * (normal_force + tangential_force)
            resultant_momentum = fraction * Ftotal.cross(pos1 - cpos)

            cplist[nc].cnforce = fraction * normal_force
            cplist[nc].csforce = fraction * tangential_force
            cplist[nc].oldTangOverlap = tangOverTemp
            particle[end1]._update_contact_interaction(Ftotal, resultant_momentum)
        else:
            cplist[nc].cnforce = ZEROVEC3f
            cplist[nc].csforce = ZEROVEC3f
            cplist[nc].oldTangOverlap = ZEROVEC3f

    @ti.func
    def _mpm_wall_force_assemble(self, nc, dt, particle, wall, cplist):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        pos1 = particle[end1].x
        particle_rad, norm = particle[end1].rad, wall[end2].norm
        distance = wall[end2]._get_norm_distance(pos1)
        gapn = distance - particle_rad
        if gapn < 0.:
            vel1, vel2 = particle[end1].v, wall[end2]._get_velocity()

            m_eff = particle[end1].m
            kn, ks = self.kn, self.ks
            ndratio, sdratio = self.ndratio, self.sdratio
            miu = self.mu

            v_rel = vel1 - vel2 
            vn = v_rel.dot(norm) 
            vs = v_rel - v_rel.dot(norm) * norm

            normal_contact_force = -kn * gapn 
            normal_damping_force = -2 * ndratio * ti.sqrt(m_eff * kn) * vn
            normal_force = (normal_contact_force + normal_damping_force) * norm
            
            tangOverlapOld = cplist[nc].oldTangOverlap
            tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
            tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
            trial_ft = -ks * tangOverTemp
            tang_damping_force = -2 * sdratio * ti.sqrt(m_eff * ks) * vs
            
            fric = miu * ti.abs(normal_contact_force + normal_damping_force)
            tangential_force = ZEROVEC3f
            if trial_ft.norm() > fric:
                tangential_force = fric * trial_ft.normalized()
                tangOverTemp = -tangential_force / ks
            else:
                tangential_force = trial_ft + tang_damping_force
            
            fraction = wall[end2].processCircleShape(pos1, distance, -gapn)
            Ftotal = fraction * (normal_force + tangential_force)

            cplist[nc].oldTangOverlap = tangOverTemp
            particle[end1]._update_contact_interaction(Ftotal)
        else:
            cplist[nc].oldTangOverlap = ZEROVEC3f


@ti.kernel
def kernel_rebulid_history_contact_list(hist_cplist: ti.template(), hist_object_object: ti.template(), object_object: ti.types.ndarray(), 
                                        dst: ti.types.ndarray(), oldTangOverlap: ti.types.ndarray()):
    for i in range(object_object.shape[0]):
        hist_object_object[i] = object_object[i]
     
    for cp in range(object_object[object_object.shape[0] - 1]):
        hist_cplist[cp].DstID = dst[cp]
        hist_cplist[cp].oldTangOverlap = vec3f(oldTangOverlap[cp, 0], oldTangOverlap[cp, 1], oldTangOverlap[cp, 2])