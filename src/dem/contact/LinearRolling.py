import taichi as ti
import numpy as np
import math

from src.dem.contact.ContactKernel import *
from src.dem.BaseStruct import (HistoryRollingContactTable, RollingContactTable, DigitalRollingContactTable)
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.constants import ZEROVEC3f
from src.utils.ObjectIO import DictIO
from src.utils.linalg import round32
from src.utils.TypeDefination import u1
from src.utils.VectorFunction import Normalize


# refers to Luding 2008 Introduction to discrete element method
class LinearRollingModel(ContactModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.surfaceProps = LinearRollingSurfaceProperty.field(shape=self.sims.max_material_num * self.sims.max_material_num)
        self.null_model = False
        self.model_type = 2

    def calcu_critical_timestep(self, scene: myScene):
        mass = scene.find_particle_min_mass(self.sims.scheme)
        stiffness = self.find_max_stiffness()
        return ti.sqrt(mass / stiffness)

    def find_max_stiffness(self):
        maxstiff = 0.
        for materialID1 in range(self.sims.max_material_num):
            for materialID2 in range(self.sims.max_material_num):
                componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                if self.surfaceProps[componousID].kn > 0.:
                    maxstiff = ti.max(ti.max(maxstiff, self.surfaceProps[componousID].kn), self.surfaceProps[componousID].ks)
        return maxstiff
    
    def collision_initialize(self, object_type, work_type, max_object_pairs, object_num1, object_num2):
        if not self.null_model:
            if object_type == 'particle' or (object_type == 'wall' and self.sims.wall_type != 3):
                self.cplist = RollingContactTable.field(shape=max_object_pairs)

                if work_type == 0 or work_type == 1:
                    self.deactivate_exist = ti.field(ti.u8, shape=())
                    self.contact_active = ti.field(u1)
                    ti.root.dense(ti.i, round32(object_num1 * object_num2)//32).quant_array(ti.i, dimensions=32, max_num_bits=32).place(self.contact_active)
                    self.old_cplist = RollingContactTable.field(shape=max_object_pairs)
                elif work_type == 2:
                    self.hist_cplist = HistoryRollingContactTable.field(shape=max_object_pairs)
            elif object_type == 'wall' and self.sims.wall_type == 3:
                if self.sims.scheme == "DEM":
                    self.cplist = DigitalRollingContactTable.field(shape=int(self.sims.max_particle_num))
                elif self.sims.scheme == "LSDEM":
                    self.cplist = DigitalRollingContactTable.field(shape=int(self.sims.max_surface_node_num * self.sims.max_rigid_body_num))
            
    
    def add_surface_property(self, materialID1, materialID2, property):
        kn = DictIO.GetEssential(property, 'NormalStiffness')
        ks = DictIO.GetEssential(property, 'TangentialStiffness')
        kr = DictIO.GetEssential(property, 'RollingStiffness')
        kt = DictIO.GetEssential(property, 'TwistingStiffness')
        mu = DictIO.GetEssential(property, 'Friction')
        rmu = DictIO.GetEssential(property, 'RollingFriction')
        tmu = DictIO.GetEssential(property, 'TwistingFriction')
        ndratio = DictIO.GetEssential(property, 'NormalViscousDamping')
        sdratio = DictIO.GetEssential(property, 'TangentialViscousDamping')
        rdratio = DictIO.GetEssential(property, 'RollingViscousDamping')
        tdratio = DictIO.GetEssential(property, 'TwistingViscousDamping')
        componousID = 0
        if materialID1 == materialID2:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, kr, kt, mu, rmu, tmu, ndratio, sdratio, rdratio, tdratio)
        else:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(kn, ks, kr, kt, mu, rmu, tmu, ndratio, sdratio, rdratio, tdratio)
            componousID = self.get_componousID(self.sims.max_material_num, materialID2, materialID1)
            self.surfaceProps[componousID].add_surface_property(kn, ks, kr, kt, mu, rmu, tmu, ndratio, sdratio, rdratio, tdratio)
        return componousID
    
    def update_property(self, componousID, property_name, value, override):
        factor = 0
        if not override:
            factor = 1

        if property_name == "NormalStiffness":
            self.surfaceProps[componousID].kn = factor * self.surfaceProps[componousID].kn + value
        elif property_name == "TangentialStiffness":
            self.surfaceProps[componousID].ks = factor * self.surfaceProps[componousID].ks + value
        elif property_name == "Friction":
            self.surfaceProps[componousID].mu = factor * self.surfaceProps[componousID].mu + value
        elif property_name == "NormalViscousDamping":
            self.surfaceProps[componousID].ndratio = factor * self.surfaceProps[componousID].ndratio + value
        elif property_name == "TangentialViscousDamping":
            self.surfaceProps[componousID].sdratio = factor * self.surfaceProps[componousID].sdratio + value

    def get_contact_output(self, scene: myScene, neighbor_list):
        end1 = np.ascontiguousarray(self.cplist.endID1.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        end2 = np.ascontiguousarray(self.cplist.endID2.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        normal_force = np.ascontiguousarray(self.cplist.cnforce.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        tangential_force = np.ascontiguousarray(self.cplist.csforce.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        oldTangentialOverlap = np.ascontiguousarray(self.cplist.oldTangOverlap.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        oldRollAngle = np.ascontiguousarray(self.cplist.oldRollAngle.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        oldTwistAngle = np.ascontiguousarray(self.cplist.oldTwistAngle.to_numpy()[0:neighbor_list[scene.particleNum[0]]])
        return end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle
    
    def get_ppcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle = self.get_contact_output(scene, pcontact.particle_particle)
        particleParticle = np.ascontiguousarray(pcontact.particle_particle.to_numpy()[0:scene.particleNum[0] + 1])
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleParticle, end1=end1, end2=end2, normal_force=normal_force, 
                                                      tangential_force=tangential_force, oldTangentialOverlap=oldTangentialOverlap, oldRollAngle=oldRollAngle, oldTwistAngle=oldTwistAngle)
        
    def get_pwcontact_output(self, contact_path, current_time, current_print, scene: myScene, pcontact: NeighborBase):
        end1, end2, normal_force, tangential_force, oldTangentialOverlap, oldRollAngle, oldTwistAngle = self.get_contact_output(scene, pcontact.particle_wall)
        particleWall = np.ascontiguousarray(pcontact.particle_wall.to_numpy()[0:scene.particleNum[0] + 1])
        np.savez(contact_path+f'{current_print:06d}', t_current=current_time, contact_num=particleWall, end1=end1, end2=end2, normal_force=normal_force, 
                                                      tangential_force=tangential_force, oldTangentialOverlap=oldTangentialOverlap, oldRollAngle=oldRollAngle, oldTwistAngle=oldTwistAngle)
        
    def rebuild_contact_list(self, contact_info):
        object_object = DictIO.GetEssential(contact_info, "contact_num")
        DstID = DictIO.GetEssential(contact_info, "end2")
        oldTangOverlap = DictIO.GetEssential(contact_info, "oldTangentialOverlap")
        oldRollAngle = DictIO.GetEssential(contact_info, "oldRollAngle")
        oldTwistAngle = DictIO.GetEssential(contact_info, "oldTwistAngle")
        return object_object, DstID, oldTangOverlap, oldRollAngle, oldTwistAngle
    
    def rebuild_ppcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, DstID, oldTangOverlap, oldRollAngle, oldTwistAngle = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_addition_history_contact_list(self.cplist, pcontact.hist_particle_particle, object_object, DstID, oldTangOverlap, oldRollAngle, oldTwistAngle)

    def rebuild_pwcontact_list(self, pcontact: NeighborBase, contact_info):
        object_object, DstID, oldTangOverlap, oldRollAngle, oldTwistAngle = self.rebuild_contact_list(contact_info)
        if DstID.shape[0] > self.cplist.shape[0]:
            raise RuntimeError("/body_coordination_number/ should be enlarged")
        kernel_rebulid_addition_history_contact_list(self.cplist, pcontact.hist_particle_wall, object_object, DstID, oldTangOverlap, oldRollAngle, oldTwistAngle)

    # ========================================================= #
    #              Particle Contact Matrix Resolve              #
    # ========================================================= # 
    def update_particle_particle_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_addition_contact_table(pcontact.hist_particle_particle, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
        self.update_ppcontact_table(sims, scene, pcontact)
        kernel_inherit_rolling_history(int(scene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_particle, pcontact.hist_particle_particle)

    def update_particle_wall_contact_table(self, sims: Simulation, scene: myScene, pcontact: NeighborBase):
        copy_addition_contact_table(pcontact.hist_particle_wall, int(scene.particleNum[0]), self.cplist, self.hist_cplist)
        self.update_pwcontact_table(sims, scene, pcontact)
        kernel_inherit_rolling_history(int(scene.particleNum[0]), self.cplist, self.hist_cplist, pcontact.particle_wall, pcontact.hist_particle_wall)


        
@ti.dataclass
class LinearRollingSurfaceProperty:
    kn: float
    ks: float
    kr: float
    kt: float
    mu: float
    rmu: float
    tmu: float
    ndratio: float
    sdratio: float
    rdratio: float
    tdratio: float

    def add_surface_property(self, kn, ks, kr, kt, mu, rmu, tmu, ndratio, sdratio, rdratio, tdratio):
        self.kn = kn
        self.ks = ks
        self.kr = kr
        self.kt = kt
        self.mu = mu
        self.rmu = rmu
        self.tmu = tmu
        self.ndratio = ndratio
        self.sdratio = sdratio
        self.rdratio = rdratio
        self.tdratio = tdratio

    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, '-'))
        print('Contact model: Linear Contact Model')
        print(f'MaterialID{matID1} < --- > MaterialID{matID2}')
        print('Contact normal stiffness: = ', self.kn)
        print('Contact tangential stiffness: = ', self.ks)
        print('Contact rolling stiffness: = ', self.kr)
        print('Contact twisting stiffness: = ', self.kt)
        print('Friction coefficient = ', self.mu)
        print('Rolling friction coefficient = ', self.rmu)
        print('Twisting friction coefficient = ', self.tmu)
        print('Viscous damping coefficient = ', self.ndratio)
        print('Viscous damping coefficient = ', self.sdratio)
        print('Viscous damping coefficient = ', self.rdratio)
        print('Viscous damping coefficient = ', self.tdratio, '\n')

    @ti.func
    def _get_equivalent_stiffness(self, end1, end2, particle, wall):
        pos1, pos2 = particle[end1].x, wall[end2]._get_center()
        particle_rad, norm = particle[end1].rad, wall[end2].norm
        distance = (pos1 - pos2).dot(norm)
        fraction = ti.abs(wall[end2].processCircleShape(pos1, particle_rad, distance))
        return fraction * self.kn
    
    @ti.func
    def _normal_force(self, kn, ndratio, m_eff, gapn, vn):
        normal_contact_force = -kn * gapn 
        normal_damping_force = -2 * ndratio * ti.sqrt(m_eff * kn) * vn
        return normal_contact_force + normal_damping_force
    
    @ti.func
    def _tangential_force(self, ks, miu, sdratio, m_eff, normal_force, vs, norm, tangOverlapOld, dt):
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        trial_ft = -ks * tangOverTemp
        tang_damping_force = -2 * sdratio * ti.sqrt(m_eff * ks) * vs
        
        fric = miu * ti.abs(normal_force)
        tangential_force = ZEROVEC3f
        if trial_ft.norm() > fric:
            tangential_force = fric * trial_ft.normalized()
            tangOverTemp = -tangential_force / ks
        else:
            tangential_force = trial_ft + tang_damping_force
        return tangential_force, tangOverTemp
    
    @ti.func
    def _rolling_force(self, kr, rmiu, rdratio, m_eff, rad_eff, normal_force, vr, norm, tangRollingOld, dt):
        tangRollingRot = tangRollingOld - tangRollingOld.dot(norm) * norm
        tangRollingTemp = vr * dt[None] + tangRollingOld.norm() * Normalize(tangRollingRot)
        trial_fr = -kr * tangRollingTemp
        rolling_damping_force = -2 * rdratio * ti.sqrt(m_eff * kr) * vr
        
        fricRoll = rmiu * ti.abs(normal_force)
        rolling_force = ZEROVEC3f
        if trial_fr.norm() > fricRoll:
            rolling_force = fricRoll * trial_fr.normalized()
            tangRollingTemp = -rolling_force / kr
        else:
            rolling_force = trial_fr + rolling_damping_force
        rolling_momentum = rad_eff * norm.cross(rolling_force)
        return rolling_momentum, tangRollingTemp

    @ti.func
    def _twisting_force(self, kt, tmiu, tdratio, m_eff, rad_eff, normal_force, vt, norm, tangTwistingOld, dt):
        tangTwistingTemp = vt * dt[None] + tangTwistingOld.norm() * Normalize(norm)
        trial_ft = -kt * tangTwistingTemp
        twisting_damping_force = -2 * tdratio * ti.sqrt(m_eff * kt) * vt
        
        fricTwist = tmiu * ti.abs(normal_force)
        twisting_force = ZEROVEC3f
        if trial_ft.norm() > fricTwist:
            twisting_force = fricTwist * trial_ft.normalized()
            tangTwistingTemp = -twisting_force / kt
        else:
            twisting_force = trial_ft + twisting_damping_force
        twisting_momentum = rad_eff * twisting_force
        return twisting_momentum, tangTwistingTemp
    
    @ti.func
    def _force_assemble(self, m_eff, rad_eff, gapn, coeff, norm, v_rel, w_rel, wr_rel, tangOverlapOld, tangRollingOld, tangTwistingOld, dt):
        kn, ks = self.kn * coeff, self.ks * coeff
        ndratio, sdratio = self.ndratio, self.sdratio
        miu = self.mu
        kr, kt = self.kr * coeff, self.kt * coeff
        rdratio, tdratio = self.rdratio, self.tdratio
        rmiu, tmiu = self.rmu, self.tmu

        vn = v_rel.dot(norm) 
        vs = v_rel - vn * norm
        vt = rad_eff * (w_rel).dot(norm) * norm
        vr = -rad_eff * wr_rel

        normal_force = self._normal_force(kn, ndratio, m_eff, gapn, vn)
        tangential_force, tangOverTemp = self._tangential_force(ks, miu, sdratio, m_eff, normal_force, vs, norm, tangOverlapOld, dt)
        rolling_momentum, tangRollingTemp = self._rolling_force(kr, rmiu, rdratio, m_eff, rad_eff, normal_force, vr, norm, tangRollingOld, dt)
        twisting_momentum, tangTwistingTemp = self._twisting_force(kt, tmiu, tdratio, m_eff, rad_eff, normal_force, vt, norm, tangTwistingOld, dt)
        return normal_force * norm, tangential_force, rolling_momentum + twisting_momentum, tangOverTemp, tangRollingTemp, tangTwistingTemp


