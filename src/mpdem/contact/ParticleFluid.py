import taichi as ti

from src.dem.BaseStruct import SFContactTable
from src.dem.SceneManager import myScene as DEMScene
from src.mpdem.contact.ContactKernel import *
from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpdem.contact.MultiLinkedCell import MultiLinkedCell
from src.mpdem.Simulation import Simulation
from src.mpm.SceneManager import myScene as MPMScene
from src.utils.ObjectIO import DictIO
from src.utils.ScalarFunction import EffectiveValue
from src.utils.TypeDefination import vec3f


class ParticleFluid(ContactModelBase):
    def __init__(self, max_material_num) -> None:
        super().__init__()
        self.surfaceProps = LiquidSurfaceProperty.field(shape=max_material_num * max_material_num)
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
                    maxstiff = ti.max(maxstiff, self.surfaceProps[componousID].kn)
        return maxstiff
    
    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        componousID = 0
        if materialID1 == materialID2:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property()
        else:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property()
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property()
        return componousID


    def inherit_surface_property(self, max_material_num, materialID1,  materialID2, property1, property2):
        componousID = 0
        if materialID1 == materialID2:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property()
        else:
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property()
            componousID = self.get_componousID(max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property()
        return componousID
    
    def collision_initialize(self, parameter, max_object_pairs):
        if not self.null_mode:
            self.cplist = SFContactTable.field(shape=int(parameter * max_object_pairs))
    
    def get_contact_output(self, scene: MPMScene, neighbor_list):
        pass
    
    def get_ppcontact_output(self, contact_path, current_time, current_print, scene: DEMScene, pcontact: MultiLinkedCell):
        pass
        
    def get_pwcontact_output(self, contact_path, current_time, current_print, scene: MPMScene, pcontact: MultiLinkedCell):
        pass
    
    def rebuild_ppcontact_list(self, pcontact: MultiLinkedCell, contact_info):
        pass

    def rebuild_pwcontact_list(self, pcontact: MultiLinkedCell, contact_info):
        pass

    # ========================================================= #
    #              Particle Contact Matrix Resolve              #
    # ========================================================= # 
    def update_particle_particle_contact_table(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        update_contact_table_(sims.potential_particle_num, pcontact.particle_particle, pcontact.potential_list_particle_particle, self.cplist, int(mscene.particleNum[0]))

    def update_particle_wall_contact_table(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        update_wall_contact_table_(sims.wall_coordination_number, pcontact.particle_wall, pcontact.potential_list_particle_wall, self.cplist, int(mscene.particleNum[0]))

    def tackle_particle_particle_contact_cplist(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        kernel_fluid_particle_force_assemble_(int(mscene.particleNum[0]), sims.max_material_num, self.surfaceProps, mscene.particle, dscene.particle, 
                                              self.cplist, pcontact.particle_particle, sims.dt)

    def tackle_particle_wall_contact_cplist(self, sims: Simulation, mscene: MPMScene, dscene: DEMScene, pcontact: MultiLinkedCell):
        kernel_fluid_wall_force_assemble_(int(mscene.particleNum[0]), sims.max_material_num, self.surfaceProps, mscene.particle, dscene.wall, 
                                          self.cplist, pcontact.particle_wall, sims.dt)

@ti.dataclass
class LiquidSurfaceProperty:
    def add_surface_property(self):
        pass

    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, '-'))
        print('Contact model: Fluid-Particle Model')
        print(f'MaterialID{matID1} < --- > MaterialID{matID2}')
        print('\n')

    # ========================================================= #
    #                   Particle-Particle                       #
    # ========================================================= # 
    @ti.func
    def _coupled_particle_force_assemble(self, nc, end1, end2, gapn, norm, cpos, dt, particle1, particle2, cplist):
        pos2, w2 = particle2[end2].x, particle2[end2].w
        vel1, vel2 = particle1[end1].v, particle2[end2].v

        v_rel = vel1 - (vel2 + w2.cross(cpos - pos2))
        Ftotal = v_rel / dt[None] * particle1[end1].m
        
        particle1[end1]._update_contact_interaction(Ftotal)
        particle2[end2]._update_contact_interaction(-Ftotal, vec3f(0, 0, 0))


    # ========================================================= #
    #                      Particle-Wall                        #
    # ========================================================= # 
    @ti.func
    def _mpm_wall_force_assemble(self, nc, end1, end2, distance, gapn, norm, dt, particle, wall, cplist):
        pos1 = particle[end1].x
        vel1, vel2 = particle[end1].v, wall[end2]._get_velocity()

        v_rel = vel1 - vel2 
        normal_force = v_rel / dt[None] * particle[end1].m

        fraction = wall[end2].processCircleShape(pos1, distance, -gapn)
        Ftotal = fraction * normal_force 
        particle[end1]._update_contact_interaction(Ftotal)

