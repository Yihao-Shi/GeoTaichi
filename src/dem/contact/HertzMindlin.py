import taichi as ti
import numpy as np

from src.dem.contact.ContactKernel import *
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.constants import PI, ZEROVEC3f
from src.utils.ObjectIO import DictIO
from src.utils.ScalarFunction import EffectiveValue, sgn
from src.utils.TypeDefination import vec3f
from src.utils.VectorFunction import Normalize


class HertzMindlinModel(ContactModelBase):
    def __init__(self, max_material_num) -> None:
        super().__init__()
        self.surfaceProps = HertzMindlinSurfaceProperty.field(shape=max_material_num * max_material_num)
        self.null_mode = False

    def calcu_critical_timestep(self, scene: myScene, max_material_num):
        radius = scene.find_particle_min_radius()
        density = scene.find_min_density()
        modulus, Possion = self.find_max_mparas(max_material_num)
        return PI * radius * ti.sqrt(density / modulus) / (0.01631 * Possion + 0.8766)

    def find_max_mparas(self, max_material_num):
        maxmodulus, maxpossion = 0., 0.
        for materialID1 in range(max_material_num):
            for materialID2 in range(max_material_num):
                componousID = self.get_componousID(max_material_num, materialID1, materialID2)
                if self.surfaceProps[componousID].ShearModulus > 0.:
                    Possion = (4 * self.surfaceProps[componousID].ShearModulus - self.surfaceProps[componousID].YoungModulus) / \
                              (2 * self.surfaceProps[componousID].ShearModulus - self.surfaceProps[componousID].YoungModulus)
                    modulus = 2 * self.surfaceProps[componousID].ShearModulus * (2 - Possion)
                    maxpossion = ti.max(maxpossion, Possion)
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
            componousID = self.get_componousID(max_material_num, materialID2, materialID1)
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
            componousID = self.get_componousID(max_material_num, materialID2, materialID1)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mu, restitution)
        return componousID
    
    def update_property(self, componousID, property_name, value, override):
        if override:
            override = 1
        else:
            override = 0

        if property_name == "ShearModulus":
            shear_modulus = override * self.surfaceProps[componousID].ShearModulus + value
            Possion = 0.5 * (self.surfaceProps[componousID].YoungModulus / self.surfaceProps[componousID].ShearModulus) - 1.
            YoungModulus = shear_modulus * 2 * (1 + Possion)
            self.surfaceProps[componousID].YoungModulus = YoungModulus
            self.surfaceProps[componousID].ShearModulus = shear_modulus
        elif property_name == "Possion":
            shear_modulus = self.surfaceProps[componousID].ShearModulus 
            Possion = override * (0.5 * (self.surfaceProps[componousID].YoungModulus / self.surfaceProps[componousID].ShearModulus) - 1.) + value
            self.surfaceProps[componousID].YoungModulus = shear_modulus * 2 * (1 + Possion)
        elif property_name == "Friction":
            self.surfaceProps[componousID].mu = override * self.surfaceProps[componousID].mu + value
        elif property_name == "Restitution":
            self.surfaceProps[componousID].restitution = override * self.surfaceProps[componousID].restitution + value
    
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
        copy_contact_table(pcontact.hist_particle_particle, int(scene.particleNum[0]), self.cplist, self.hist_cplist) 
        update_contact_table_(sims.potential_particle_num, int(scene.particleNum[0]), pcontact.particle_particle, pcontact.potential_list_particle_particle, self.cplist)
        kernel_inherit_contact_history(int(scene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_particle, pcontact.hist_particle_particle)

    def update_particle_wall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_contact_table(pcontact.hist_particle_wall, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
        update_wall_contact_table_(sims.wall_coordination_number, int(scene.particleNum[0]), pcontact.particle_wall, pcontact.potential_list_particle_wall, self.cplist)
        kernel_inherit_contact_history(int(scene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_wall, pcontact.hist_particle_wall)

    def tackle_particle_particle_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_particle_particle_force_assemble_(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.particle, self.cplist, pcontact.particle_particle)
        '''
        kernel_particle_particle_narrow_detection_(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.particle, self.cplist, self.hist_cplist, 
                                                   pcontact.particle_particle, pcontact.hist_particle_particle, find_history, self.active_contact)
        self.active_pse.run(self.active_contact, total_num + 1)
        kernel_compact_contact_table(total_num, self.compact_table, self.active_contact)
        kernel_calculate_contact_force(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.particle, self.cplist, self.hist_cplist, 
                                                 pcontact.particle_particle, pcontact.hist_particle_particle, find_history, self.compact_table, self.active_contact)
        '''
        
    def tackle_particle_wall_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_particle_wall_force_assemble_(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.particle, scene.wall, self.cplist, pcontact.particle_wall)


@ti.dataclass
class HertzMindlinSurfaceProperty:
    YoungModulus: float
    ShearModulus: float
    restitution: float
    mu: float

    def add_surface_property(self, YoungModulus, ShearModulus, mu, restitution):
        self.YoungModulus = YoungModulus
        self.ShearModulus = ShearModulus
        self.mu = mu
        self.restitution = restitution


    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, '-'))
        print('Contact model: Hertz contact Model')
        print(f'MaterialID{matID1} < --- > MaterialID{matID2}')
        print('Effecitive Youngs Modulus: = ', self.YoungModulus)
        print('Effecitive Shear Modulus: = ', self.ShearModulus)
        print('Friction coefficient = ', self.mu)
        print('Restitution = ', self.restitution, '\n')

    @ti.func
    def _get_equivalent_stiffness(self, end1, end2, particle, wall):
        pos1, pos2 = particle[end1].x, wall[end2]._get_center()
        particle_rad, norm = particle[end1].rad, wall[end2].norm
        distance = (pos1 - pos2).dot(norm)
        gapn = distance - particle_rad
        contactAreaRadius = ti.sqrt(-gapn * particle_rad)
        fraction = ti.abs(wall[end2].processCircleShape(pos1, distance, -gapn))
        return 2 * fraction * self.YoungModulus * contactAreaRadius

    # ========================================================= #
    #                   Particle-Particle                       #
    # ========================================================= # 
    @ti.func
    def _particle_particle_force_assemble(self, nc, end1, end2, gapn, norm, cpos, dt, particle, cplist):
        pos1, pos2 = particle[end1].x, particle[end2].x
        rad1, rad2 = particle[end1].rad, particle[end2].rad 
        mass1, mass2 = particle[end1].m, particle[end2].m
        vel1, vel2 = particle[end1].v, particle[end2].v
        w1, w2 = particle[end1].w, particle[end2].w
        
        m_eff = EffectiveValue(mass1, mass2)
        rad_eff = EffectiveValue(rad1, rad2)
        contactAreaRadius = ti.sqrt(-gapn * rad_eff)
        effective_E, effective_G = self.YoungModulus, self.ShearModulus
        miu, restitution = self.mu, self.restitution
        kn = 2 * effective_E * contactAreaRadius
        ks = 8 * effective_G * contactAreaRadius

        v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))
        vn = v_rel.dot(norm)
        vs = v_rel - vn * norm
        
        normal_contact_force = -2./3. * kn * gapn 
        normal_damping_force = -1.8257 * restitution * vn * ti.sqrt(kn * m_eff) 
        normal_force = (normal_contact_force + normal_damping_force) * norm

        tangOverlapOld = cplist[nc].oldTangOverlap
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        trial_ft = -ks * tangOverTemp
        tang_damping_force = -1.8257 * restitution * vs * ti.sqrt(ks * m_eff)
        
        fric = miu * ti.abs(normal_contact_force + normal_damping_force)
        tangential_force = ZEROVEC3f
        if trial_ft.norm() > fric:
            tangential_force = fric * trial_ft.normalized()
            tangOverTemp = -tangential_force / ks
        else:
            tangential_force = trial_ft + tang_damping_force
        
        Ftotal = normal_force + tangential_force
        moment = tangential_force.cross(norm)

        cplist[nc]._set_contact(normal_force, tangential_force, tangOverTemp)
        particle[end1]._update_contact_interaction(Ftotal, moment * (rad1 + 0.5 * gapn))
        particle[end2]._update_contact_interaction(-Ftotal, moment * (rad2 + 0.5 * gapn))


    @ti.func
    def _coupled_particle_force_assemble(self, nc, end1, end2, gapn, norm, cpos, dt, particle1, particle2, cplist):
        pos2, w2 = particle2[end2].x, particle2[end2].w
        rad1, rad2 = particle1[end1].rad, particle2[end2].rad 
        mass1, mass2 = particle1[end1].m, particle2[end2].m
        vel1, vel2 = particle1[end1].v, particle2[end2].v
        
        m_eff = EffectiveValue(mass1, mass2)
        rad_eff = EffectiveValue(rad1, rad2)
        contactAreaRadius = ti.sqrt(-gapn * rad_eff)
        effective_E, effective_G = self.YoungModulus, self.ShearModulus
        miu, restitution = self.mu, self.restitution
        kn = 2 * effective_E * contactAreaRadius
        ks = 8 * effective_G * contactAreaRadius

        v_rel = vel1 - (vel2 + w2.cross(cpos - pos2))
        vn = v_rel.dot(norm)
        vs = v_rel - vn * norm
        
        normal_contact_force = -2./3. * kn * gapn 
        normal_damping_force = -1.8257 * restitution * vn * ti.sqrt(kn * m_eff) 
        normal_force = (normal_contact_force + normal_damping_force) * norm

        tangOverlapOld = cplist[nc].oldTangOverlap
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        trial_ft = -ks * tangOverTemp
        tang_damping_force = -1.8257 * restitution * vs * ti.sqrt(ks * m_eff)
        
        fric = miu * ti.abs(normal_contact_force + normal_damping_force)
        tangential_force = ZEROVEC3f
        if trial_ft.norm() > fric:
            tangential_force = fric * trial_ft.normalized()
            tangOverTemp = -tangential_force / ks
        else:
            tangential_force = trial_ft + tang_damping_force
        
        Ftotal = normal_force + tangential_force
        resultant_momentum = Ftotal.cross(cpos - pos2) 

        cplist[nc]._set_contact(tangOverTemp)
        particle1[end1]._update_contact_interaction(Ftotal)
        particle2[end2]._update_contact_interaction(-Ftotal, resultant_momentum)


    # ========================================================= #
    #                      Particle-Wall                        #
    # ========================================================= # 
    @ti.func
    def _particle_wall_force_assemble(self, nc, end1, end2, distance, gapn, norm, cpos, dt, particle, wall, cplist):
        pos1, particle_rad = particle[end1].x, particle[end1].rad
        vel1, vel2 = particle[end1].v, wall[end2]._get_velocity()
        w1 = particle[end1].w
            
        m_eff = particle[end1].m
        rad_eff = particle_rad
        contactAreaRadius = ti.sqrt(-gapn * rad_eff)
        effective_E, effective_G = self.YoungModulus, self.ShearModulus
        miu, restitution = self.mu, self.restitution
        kn = 2 * effective_E * contactAreaRadius
        ks = 8 * effective_G * contactAreaRadius

        v_rel = vel1 + w1.cross(cpos - pos1) - vel2 
        vn = v_rel.dot(norm)
        vs = v_rel - vn * norm
        
        normal_contact_force = -2./3. * kn * gapn 
        normal_damping_force = -1.8257 * restitution * vn * ti.sqrt(kn * m_eff) 
        normal_force = (normal_contact_force + normal_damping_force) * norm
        
        tangOverlapOld = cplist[nc].oldTangOverlap
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        trial_ft = -ks * tangOverTemp
        tang_damping_force = -1.8257 * restitution * vs * ti.sqrt(ks * m_eff)
        
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

        cplist[nc]._set_contact(fraction * normal_force, fraction * tangential_force, tangOverTemp)
        particle[end1]._update_contact_interaction(Ftotal, resultant_momentum)

    @ti.func
    def _mpm_wall_force_assemble(self, nc, end1, end2, distance, gapn, norm, dt, particle, wall, cplist):
        pos1, particle_rad = particle[end1].x, particle[end1].rad
        vel1, vel2 = particle[end1].v, wall[end2]._get_velocity()
        
        m_eff = particle[end1].m
        contactAreaRadius = ti.sqrt(-gapn * particle_rad)
        effective_E, effective_G = self.YoungModulus, self.ShearModulus
        miu, restitution = self.mu, self.restitution
        kn = 2 * effective_E * contactAreaRadius
        ks = 8 * effective_G * contactAreaRadius

        v_rel = vel1 - vel2 
        vn = v_rel.dot(norm)
        vs = v_rel - vn * norm
        
        normal_contact_force = -2./3. * kn * gapn 
        normal_damping_force = -1.8257 * restitution * vn * ti.sqrt(kn * m_eff) 
        normal_force = (normal_contact_force + normal_damping_force) * norm
        
        tangOverlapOld = cplist[nc].oldTangOverlap
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        trial_ft = -ks * tangOverTemp
        tang_damping_force = -1.8257 * restitution * vs * ti.sqrt(ks * m_eff)
        
        fric = miu * ti.abs(normal_contact_force + normal_damping_force)
        tangential_force = ZEROVEC3f
        if trial_ft.norm() > fric:
            tangential_force = fric * trial_ft.normalized()
            tangOverTemp = -tangential_force / ks
        else:
            tangential_force = trial_ft + tang_damping_force
        
        fraction = wall[end2].processCircleShape(pos1, distance, -gapn)
        Ftotal = fraction * (normal_force + tangential_force)

        cplist[nc]._set_contact(tangOverTemp)
        particle[end1]._update_contact_interaction(Ftotal)


@ti.kernel
def kernel_rebulid_history_contact_list(hist_cplist: ti.template(), hist_object_object: ti.template(), object_object: ti.types.ndarray(), 
                                        dst: ti.types.ndarray(), oldTangOverlap: ti.types.ndarray()):
    for i in range(object_object.shape[0]):
        hist_object_object[i] = object_object[i]

    for cp in range(object_object[object_object.shape[0] - 1]):
        hist_cplist[cp].DstID = dst[cp]
        hist_cplist[cp].oldTangOverlap = vec3f(oldTangOverlap[cp, 0], oldTangOverlap[cp, 1], oldTangOverlap[cp, 2])
