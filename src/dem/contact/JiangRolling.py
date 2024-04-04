import taichi as ti
import numpy as np
import math

from src.dem.contact.ContactKernel import *
from src.dem.BaseStruct import (HistoryRollingContactTable, RollingContactTable)
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.constants import ZEROVEC3f
from src.utils.ObjectIO import DictIO
from src.utils.ScalarFunction import EffectiveValue, round32
from src.utils.TypeDefination import u1, vec3f
from src.utils.VectorFunction import Normalize


# Refers to Jiang et. al (2015) A novel three-dimensional contact model for granulates incorporating rolling and twisting resistances. Computer and Geotechnics
class JiangRollingResistanceModel(ContactModelBase):
    def __init__(self, max_material_num) -> None:
        super().__init__()
        self.surfaceProps = JiangRollingSurfaceProperty.field(shape=max_material_num * max_material_num)
        self.null_mode = False

    def calcu_critical_timestep(self, scene: myScene, max_material_num):
        mass = scene.find_particle_min_mass()
        radius = scene.find_particle_min_radius()
        modulus = self._find_max_modulus_(max_material_num)
        stiffness = 2 * radius * modulus
        return ti.sqrt(mass / stiffness)

    def _find_max_modulus_(self, max_material_num):
        maxmodulus = 0.
        for materialID1 in range(max_material_num):
            for materialID2 in range(max_material_num):
                componousID = self.get_componousID(max_material_num, materialID1, materialID2)
                if self.surfaceProps[componousID].YoungModulus > 0.:
                    maxmodulus = ti.max(maxmodulus, self.surfaceProps[componousID].YoungModulus)
        return maxmodulus
    
    def collision_initialize(self, parameter, work_type, max_particle_pairs, object_num1, object_num2):
        if not self.null_mode:
            self.cplist = RollingContactTable.field(shape=int(math.ceil(parameter * max_particle_pairs)))
            if work_type == 0 or work_type == 1:
                self.deactivate_exist = ti.field(ti.u8, shape=())
                self.contact_active = ti.field(u1)
                ti.root.dense(ti.i, round32(object_num1 * object_num2)//32).quant_array(ti.i, dimensions=32, max_num_bits=32).place(self.contact_active)
                self.old_cplist = RollingContactTable.field(shape=int(math.ceil(parameter * max_particle_pairs)))
            elif work_type == 2:
                self.hist_cplist = HistoryRollingContactTable.field(shape=int(math.ceil(parameter * max_particle_pairs)))

    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        YoungModulus = DictIO.GetEssential(property, 'YoungModulus')
        stiffness_ratio = DictIO.GetEssential(property, 'StiffnessRatio')
        mu = DictIO.GetEssential(property, 'Friction')
        shape_factor = DictIO.GetEssential(property, 'ShapeFactor')
        crush_factor = DictIO.GetEssential(property, 'CrushFactor')
        ndratio = DictIO.GetEssential(property, 'NormalViscousDamping')
        sdratio = DictIO.GetEssential(property, 'TangentialViscousDamping')
        componousID = 0
        if materialID1 == materialID2:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio)
        else:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio)
            componousID = self.get_componousID(max_material_num, materialID2, materialID1)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio)
        return componousID

    def inherit_surface_property(self, max_material_num, materialID1, materialID2, property1, property2):
        YoungModulus1 = DictIO.GetEssential(property1, 'YoungModulus')
        stiffness_ratio1 = DictIO.GetEssential(property1, 'StiffnessRatio')
        mu1 = DictIO.GetEssential(property1, 'Friction')
        shape_factor1 = DictIO.GetEssential(property1, 'ShapeFactor')
        crush_factor1 = DictIO.GetEssential(property1, 'CrushFactor')
        ndratio1 = DictIO.GetEssential(property1, 'NormalViscousDamping')
        sdratio1 = DictIO.GetEssential(property1, 'TangentialViscousDamping')

        YoungModulus2 = DictIO.GetEssential(property2, 'YoungModulus')
        stiffness_ratio2 = DictIO.GetEssential(property2, 'StiffnessRatio')
        mu2 = DictIO.GetEssential(property2, 'Friction')
        shape_factor2 = DictIO.GetEssential(property2, 'ShapeFactor')
        crush_factor2 = DictIO.GetEssential(property2, 'CrushFactor')
        ndratio2 = DictIO.GetEssential(property2, 'NormalViscousDamping')
        sdratio2 = DictIO.GetEssential(property2, 'TangentialViscousDamping')
        
        YoungModulus = EffectiveValue(YoungModulus1, YoungModulus2)
        stiffness_ratio = ti.min(stiffness_ratio1, stiffness_ratio2)
        mu = ti.min(mu1, mu2)
        shape_factor = ti.min(shape_factor1, shape_factor2)
        crush_factor = ti.min(crush_factor1, crush_factor2)
        ndratio = ti.min(ndratio1, ndratio2)
        sdratio = ti.min(sdratio1, sdratio2)
        componousID = 0
        if materialID1 == materialID2:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio)
        else:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio)
            componousID = self.get_componousID(max_material_num, materialID2, materialID1)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio)
        return componousID
    
    def update_property(self, componousID, property_name, value, override):
        if override:
            override = 1
        else:
            override = 0

        if property_name == "YoungModulus":
            self.surfaceProps[componousID].YoungModulus = override * self.surfaceProps[componousID].YoungModulus + value
        elif property_name == "StiffnessRatio":
            self.surfaceProps[componousID].stiffness_ratio = override * self.surfaceProps[componousID].stiffness_ratio + value
        elif property_name == "Friction":
            self.surfaceProps[componousID].mu = override * self.surfaceProps[componousID].mu + value
        elif property_name == "NormalViscousDamping":
            self.surfaceProps[componousID].ndratio = override * self.surfaceProps[componousID].ndratio + value
        elif property_name == "TangentialViscousDamping":
            self.surfaceProps[componousID].sdratio = override * self.surfaceProps[componousID].sdratio + value
        elif property_name == "ShapeFactor":
            self.surfaceProps[componousID].shape_factor = override * self.surfaceProps[componousID].shape_factor + value
        elif property_name == "CrushFactor":
            self.surfaceProps[componousID].crush_factor = override * self.surfaceProps[componousID].crush_factor + value
    
    def get_ppcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle = self.get_contact_output(scene, pcontact.particle_particle)
        particleParticle = np.ascontiguousarray(pcontact.hist_particle_particle.to_numpy()[0:scene.particleNum[0] + 1])
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleParticle, end1=end1, end2=end2, normal_force=normal_force, 
                                                      tangential_force=tangential_force, oldTangentialOverlap=oldTangentialOverlap, oldRollAngle=oldRollAngle, oldTwistAngle=oldTwistAngle)
        
    def get_pwcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle = self.get_contact_output(scene, pcontact.particle_wall)
        particleWall = np.ascontiguousarray(pcontact.hist_particle_wall.to_numpy()[0:scene.particleNum[0] + 1])
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleWall, end1=end1, end2=end2, normal_force=normal_force, 
                                                      tangential_force=tangential_force, oldTangentialOverlap=oldTangentialOverlap, oldRollAngle=oldRollAngle, oldTwistAngle=oldTwistAngle)
        
    def get_contact_output(self, scene: myScene, neighbor_list):
        end1 = np.ascontiguousarray(self.cplist.endID1.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        end2 = np.ascontiguousarray(self.cplist.endID2.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        normal_force = np.ascontiguousarray(self.cplist.cnforce.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        tangential_force = np.ascontiguousarray(self.cplist.csforce.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        oldTangentialOverlap = np.ascontiguousarray(self.cplist.oldTangOverlap.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        oldRollAngle = np.ascontiguousarray(self.cplist.oldRollAngle.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        oldTwistAngle = np.ascontiguousarray(self.cplist.oldTwistAngle.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        return end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle

    def rebuild_contact_list(self, contact_info):
        object_object = DictIO.GetEssential(contact_info, "contact_num")
        particle_number = object_object[object_object.shape[0] - 1]
        DstID = DictIO.GetEssential(contact_info, "end2")
        oldTangOverlap = DictIO.GetEssential(contact_info, "oldTangentialOverlap")
        oldRollAngle = DictIO.GetEssential(contact_info, "oldRollAngle")
        oldTwistAngle = DictIO.GetEssential(contact_info, "oldTwistAngle")
        return object_object, particle_number, DstID, oldTangOverlap, oldRollAngle, oldTwistAngle

    def rebuild_ppcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, particle_number, DstID, oldTangOverlap, oldRollAngle, oldTwistAngle = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_history_contact_list(self.hist_cplist, pcontact.hist_particle_particle, object_object, DstID, oldTangOverlap, oldRollAngle, oldTwistAngle)

    def rebuild_pwcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, particle_number, DstID, oldTangOverlap, oldRollAngle, oldTwistAngle = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_history_contact_list(self.hist_cplist, pcontact.hist_particle_wall, object_object, DstID, oldTangOverlap, oldRollAngle, oldTwistAngle)

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
        copy_addition_contact_table(pcontact.hist_particle_particle, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
        update_contact_table_(sims.potential_particle_num, int(scene.particleNum[0]), pcontact.particle_particle, pcontact.potential_list_particle_particle, self.cplist)
        kernel_inherit_rolling_history(int(scene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_particle, pcontact.hist_particle_particle)

    def update_particle_wall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_addition_contact_table(pcontact.hist_particle_wall, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
        update_wall_contact_table_(sims.wall_coordination_number, int(scene.particleNum[0]), pcontact.particle_wall, pcontact.potential_list_particle_wall, self.cplist)
        kernel_inherit_rolling_history(int(scene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_wall, pcontact.hist_particle_wall)

    def tackle_particle_particle_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_particle_particle_force_assemble_(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.particle, self.cplist, pcontact.particle_particle)
        
    def tackle_particle_wall_contact_cplist(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        kernel_particle_wall_force_assemble_(int(scene.particleNum[0]), sims.dt, sims.max_material_num, self.surfaceProps, scene.particle, scene.wall, self.cplist, pcontact.particle_wall)


@ti.dataclass
class JiangRollingSurfaceProperty:
    YoungModulus: float
    stiffness_ratio: float
    shape_factor: float
    crush_factor: float
    mu: float
    ndratio: float
    sdratio: float

    def add_surface_property(self, YoungModulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio):
        self.YoungModulus = YoungModulus
        self.stiffness_ratio = stiffness_ratio
        self.mu = mu
        self.shape_factor = shape_factor
        self.crush_factor = crush_factor
        self.ndratio = ndratio
        self.sdratio = sdratio

    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, '-'))
        print('Contact model: Linear Rolling Resistance Contact Model')
        print(f'MaterialID{matID1} < --- > MaterialID{matID2}')
        print('Youngs Modulus: = ', self.YoungModulus)
        print('Stiffness Ratio: = ', self.stiffness_ratio)
        print('Shape Factor = ', self.shape_factor)
        print('Crush Factor = ', self.crush_factor)
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
        return 2 * fraction * particle_rad * self.YoungModulus

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
        rad_eff = 2  * EffectiveValue(rad1, rad2)
        YoungModulus, stiffness_ratio = self.YoungModulus, self.stiffness_ratio
        shape_factor, crush_factor = self.shape_factor, self.crush_factor
        ndratio = self.ndratio * 2 * ti.sqrt(m_eff * kn),
        sdratio = self.sdratio * 2 * ti.sqrt(m_eff * ks)
        miu = self.mu
        kn = 2 * rad_eff * YoungModulus
        ks = kn * stiffness_ratio

        RBar = shape_factor * rad_eff
        SquareR = RBar * RBar
        kr = 0.25 * kn * SquareR
        kt = 0.5 * ks * SquareR
        rdratio = 0.25 * ndratio * SquareR
        tdratio = 0.5 * sdratio * SquareR
        rmiu = 0.25 * RBar * crush_factor
        tmiu = 0.65 * RBar * miu

        cpos = pos2 + (rad2 + 0.5 * gapn) * norm
        v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))
        w_rel = w1 - w2
        vn = v_rel.dot(norm) 
        vs = v_rel - vn * norm
        wt = (w_rel).dot(norm) * norm
        wr = w_rel - wt 

        tangOverlapOld, tangRollingOld, tangTwistingOld = cplist[nc].oldTangOverlap, cplist[nc].oldRollAngle, cplist[nc].oldTwistAngle
        
        normal_contact_force = -kn * gapn 
        normal_damping_force = -ndratio * vn
        normal_force = (normal_contact_force + normal_damping_force) * norm

        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        trial_ft = -ks * tangOverTemp
        tang_damping_force = -sdratio * vs
        
        fric = miu * ti.abs(normal_contact_force + normal_damping_force)
        tangential_force = ZEROVEC3f
        if trial_ft.norm() > fric:
            tangential_force = fric * trial_ft.normalized()
            tangOverTemp = -tangential_force / ks
        else:
            tangential_force = trial_ft + tang_damping_force
        
        tangRollingRot = tangRollingOld - tangRollingOld.dot(norm) * norm
        tangRollingTemp = wr * dt[None] + tangRollingOld.norm() * Normalize(tangRollingRot)
        trial_fr = -kr * tangRollingTemp
        rolling_damping_force = -rdratio * wr
        
        fricRoll = rmiu * ti.abs(normal_contact_force + normal_damping_force)
        rolling_momentum = ZEROVEC3f
        if trial_fr.norm() > fricRoll:
            rolling_momentum = fricRoll * trial_fr.normalized()
            tangRollingTemp = -rolling_momentum / kr
        else:
            rolling_momentum = trial_fr + rolling_damping_force

        tangTwistingTemp = wt * dt[None] + tangTwistingOld.norm() * Normalize(norm)
        trial_ft = -kt * tangTwistingTemp
        twisting_damping_force = -tdratio * wt
        
        fricTwist = tmiu * ti.abs(normal_contact_force + normal_damping_force)
        twisting_momentum = ZEROVEC3f
        if trial_ft.norm() > fricTwist:
            twisting_momentum = fricTwist * trial_ft.normalized()
            tangTwistingTemp = -twisting_momentum / kt
        else:
            twisting_momentum = trial_ft + twisting_damping_force
        
        Ftotal = normal_force + tangential_force
        resultant_momentum1 = Ftotal.cross(pos1 - cpos) + rolling_momentum + twisting_momentum
        resultant_momentum2 = Ftotal.cross(pos2 - cpos) + rolling_momentum + twisting_momentum

        cplist[nc]._set_contact(normal_force, tangential_force, tangOverTemp, tangRollingTemp, tangTwistingTemp)
        particle[end1]._update_contact_interaction(Ftotal, resultant_momentum1)
        particle[end2]._update_contact_interaction(-Ftotal, -resultant_momentum2)


    # ========================================================= #
    #                      Particle-Wall                        #
    # ========================================================= # 
    @ti.func
    def _particle_wall_force_assemble(self, nc, end1, end2, distance, gapn, norm, cpos, dt, particle, wall, cplist):
        pos1, particle_rad = particle[end1].x, particle[end1].rad
        vel1, vel2 = particle[end1].v, wall[end2]._get_velocity()
        w1 = particle[end1].w
        m_eff = particle[end1].m

        rad_eff = 2 * particle_rad
        YoungModulus, stiffness_ratio = self.YoungModulus, self.stiffness_ratio
        shape_factor, crush_factor = self.shape_factor, self.crush_factor
        ndratio = self.ndratio * 2 * ti.sqrt(m_eff * kn),
        sdratio = self.sdratio * 2 * ti.sqrt(m_eff * ks)
        miu = self.mu
        kn = 2 * rad_eff * YoungModulus
        ks = kn * stiffness_ratio

        RBar = shape_factor * rad_eff
        SquareR = RBar * RBar
        kr = 0.25 * kn * SquareR
        kt = 0.5 * ks * SquareR
        rdratio = 0.25 * ndratio * SquareR
        tdratio = 0.5 * sdratio * SquareR
        rmiu = 0.25 * RBar * crush_factor
        tmiu = 0.65 * RBar * miu

        v_rel = vel1 + w1.cross(cpos - pos1) - vel2 
        w_rel = w1 
        vn = v_rel.dot(norm) 
        vs = v_rel - vn * norm
        wt = (w_rel).dot(norm) * norm
        wr = w_rel - wt 

        tangOverlapOld, tangRollingOld, tangTwistingOld = cplist[nc].oldTangOverlap, cplist[nc].oldRollAngle, cplist[nc].oldTwistAngle

        normal_contact_force = -kn * gapn 
        normal_damping_force = -ndratio * vn
        normal_force = (normal_contact_force + normal_damping_force) * norm

        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        trial_ft = -ks * tangOverTemp
        tang_damping_force = -sdratio * vs
        
        fric = miu * ti.abs(normal_contact_force + normal_damping_force)
        tangential_force = ZEROVEC3f
        if trial_ft.norm() > fric:
            tangential_force = fric * trial_ft.normalized()
            tangOverTemp = -tangential_force / ks
        else:
            tangential_force = trial_ft + tang_damping_force
        
        tangRollingRot = tangRollingOld - tangRollingOld.dot(norm) * norm
        tangRollingTemp = wr * dt[None] + tangRollingOld.norm() * Normalize(tangRollingRot)
        trial_fr = -kr * tangRollingTemp
        rolling_damping_force = -rdratio * wr
        
        fricRoll = rmiu * ti.abs(normal_contact_force + normal_damping_force)
        rolling_momentum = ZEROVEC3f
        if trial_fr.norm() > fricRoll:
            rolling_momentum = fricRoll * trial_fr.normalized()
            tangRollingTemp = -rolling_momentum / kr
        else:
            rolling_momentum = trial_fr + rolling_damping_force

        tangTwistingTemp = wt * dt[None] + tangTwistingOld.norm() * Normalize(norm)
        trial_ft = -kt * tangTwistingTemp
        twisting_damping_force = -tdratio * wt
        
        fricTwist = tmiu * ti.abs(normal_contact_force + normal_damping_force)
        twisting_momentum = ZEROVEC3f
        if trial_ft.norm() > fricTwist:
            twisting_momentum = fricTwist * trial_ft.normalized()
            tangTwistingTemp = -twisting_momentum / kt
        else:
            twisting_momentum = trial_ft + twisting_damping_force
        
        fraction = ti.abs(wall[end2].processCircleShape(pos1, distance, -gapn))
        Ftotal = fraction * (normal_force + tangential_force)
        resultant_momentum1 = fraction * (Ftotal.cross(pos1 - cpos) + rolling_momentum + twisting_momentum)

        cplist[nc]._set_contact(fraction * normal_force, fraction * tangential_force, tangOverTemp, tangRollingTemp, tangTwistingTemp)
        particle[end1]._update_contact_interaction(Ftotal, resultant_momentum1)

 
@ti.kernel
def kernel_rebulid_history_contact_list(hist_cplist: ti.template(), hist_object_object: ti.template(), object_object: ti.types.ndarray(), 
                                        dst: ti.types.ndarray(), oldTangOverlap: ti.types.ndarray(), oldRollAngle: ti.types.ndarray(), oldTwistAngle: ti.types.ndarray()):
    for i in range(object_object.shape[0]):
        hist_object_object[i] = object_object[i]

    for cp in range(object_object[object_object.shape[0] - 1]):
        hist_cplist[cp].DstID = dst[cp]
        hist_cplist[cp].oldTangOverlap = vec3f(oldTangOverlap[cp, 0], oldTangOverlap[cp, 1], oldTangOverlap[cp, 2])
        hist_cplist[cp].oldRollAngle = vec3f(oldRollAngle[cp, 0], oldRollAngle[cp, 1], oldRollAngle[cp, 2])
        hist_cplist[cp].oldTwistAngle = vec3f(oldTwistAngle[cp, 0], oldTwistAngle[cp, 1], oldTwistAngle[cp, 2])
