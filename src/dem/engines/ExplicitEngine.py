import numpy as np

from src.dem.contact.ContactModelBase import ContactModelBase 
from src.dem.ContactManager import ContactManager
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.engines.EngineKernel import *
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation


class ExplicitEngine(object):
    physpp: ContactModelBase
    physpw: ContactModelBase

    def __init__(self, contactor: ContactManager):
        self.physpp = contactor.physpp
        self.physpw = contactor.physpw

        self.update_servo_wall = None
        self.callback = None
        self.calm = None

        self.limit = 0.

    def choose_engine(self, sims: Simulation, scene: myScene):
        self.reset_particle_message = self.reset_particle
        self.reset_wall_message = self.no_operation_rest
        if sims.max_wall_num > 0 and sims.wall_type == 1 and sims.servo_status == "On":
            self.reset_wall_message = self.reset_wall
               
        self.calcu_sphere_position = self.no_operation
        self.calcu_clump_position = self.no_operation
        self.calcu_wall_position = self.no_operation       
        if sims.engine == "SymplecticEuler":
            if sims.max_sphere_num > 0:
                self.calcu_sphere_position = self.euler_sphere_integration
            if sims.max_clump_num > 0:
                self.calcu_clump_position = self.euler_clump_integration
        elif sims.engine == "VelocityVerlet":
            if sims.max_sphere_num > 0:
                self.calcu_sphere_position = self.verlet_sphere_integration
            if sims.max_clump_num > 0:
                self.calcu_clump_position = self.verlet_clump_integration
        elif sims.engine == "PredictCorrector":
            pass
        else:
            raise RuntimeError("Engine Type is error")
        
        if sims.max_wall_num > 0 and sims.wall_type != 0:
            self.calcu_wall_position = self.wall_integration

        self.calm = self.launch_calm1
        if sims.max_clump_num > 0:
            self.calm = self.launch_calm2

        self.is_verlet_update = self.no_operation_rest
        if sims.max_particle_num > 0 and (sims.max_wall_num == 0 or sims.wall_type == 0):
            self.is_verlet_update = scene.is_particle_need_update_verlet_table
        elif sims.max_particle_num == 0 and sims.max_wall_num > 0 and sims.wall_type != 0:
            self.is_verlet_update = scene.is_wall_need_update_verlet_table
        elif sims.max_particle_num > 0 and sims.max_wall_num > 0 and sims.wall_type != 0:
            self.is_verlet_update = scene.is_need_update_verlet_table

    def set_servo_mechanism(self, sims: Simulation, callback=None):  
        if sims.max_servo_wall_num > 0 and sims.servo_status == "On":
            self.update_servo_wall = self.update_servo_motion
            if callback is None:
                self.callback = self.no_operation_none
            else:
                self.callback = callback
        else:
            self.update_servo_wall = self.no_operation_other

    def no_operation_none(self):
        pass

    def no_operation(self, sims, scene):
        pass

    def no_operation_rest(self, scene):
        pass

    def no_operation_other(self, sims, scene, neighbor):
        pass

    def launch_calm1(self, scene: myScene):
        scene.particle_calm()

    def launch_calm2(self, scene: myScene):
        scene.particle_calm()
        scene.clump_calm()

    def reset_particle(self, scene: myScene):
        particle_force_reset_(int(scene.particleNum[0]), scene.particle)

    def reset_wall(self, scene: myScene):
        wall_force_reset_(int(scene.wallNum[0]), scene.wall)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        scene.check_particle_in_domain(sims)
        neighbor.pre_neighbor(scene)
        self.physpp.update_contact_table(sims, scene, neighbor)
        self.physpw.update_contact_table(sims, scene, neighbor)
        self.physpp.contact_list_initialize(scene.particleNum, neighbor.particle_particle, neighbor.hist_particle_particle)
        neighbor.update_particle_particle_auxiliary_lists()
        self.physpw.contact_list_initialize(scene.particleNum, neighbor.particle_wall, neighbor.hist_particle_wall)
        neighbor.update_particle_wall_auxiliary_lists()
        self.limit = sims.verlet_distance * sims.verlet_distance
        sims.max_particle_radius = scene.find_particle_max_radius()

    def update_neighbor_list(self, sims, scene: myScene, neighbor: NeighborBase):
        if self.is_verlet_update(self.limit) == 1:
            self.update_verlet_table(sims, scene, neighbor)
        else:
            self.physpp.resolve(sims, scene, neighbor)
            self.physpw.resolve(sims, scene, neighbor)

    def update_verlet_table(self, sims, scene: myScene, neighbor: NeighborBase):
        scene.check_particle_in_domain(sims)
        neighbor.update_verlet_table(scene)
        self.physpp.update_contact_table(sims, scene, neighbor)
        self.physpw.update_contact_table(sims, scene, neighbor)
        self.physpp.resolve(sims, scene, neighbor)
        self.physpw.resolve(sims, scene, neighbor)
        neighbor.update_particle_particle_auxiliary_lists()
        neighbor.update_particle_wall_auxiliary_lists()

    def system_resolve(self, sims, scene: myScene, neighbor: NeighborBase):
        self.physpp.resolve(sims, scene, neighbor)
        self.physpw.resolve(sims, scene, neighbor)

    def update_servo_motion(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        get_contact_stiffness(sims.max_material_num, int(scene.particleNum[0]), scene.particle, scene.wall, self.physpw.surfaceProps, self.physpw.cplist, neighbor.particle_wall)
        self.callback()
        get_gain(sims.dt, int(scene.servoNum[0]), scene.servo, scene.wall)
        servo(int(scene.servoNum[0]), scene.wall, scene.servo)

    def integration(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        self.update_servo_wall(sims, scene, neighbor)
        self.calcu_sphere_position(sims, scene)
        self.calcu_clump_position(sims, scene)
        self.calcu_wall_position(sims, scene)

    def euler_sphere_integration(self, sims: Simulation, scene: myScene):
        move_spheres_euler_(int(scene.sphereNum[0]), sims.dt, scene.sphere, scene.particle, scene.material, sims.gravity)

    def euler_clump_integration(self, sims: Simulation, scene: myScene):
        move_clumps_euler_(int(scene.clumpNum[0]), sims.dt, scene.clump, scene.particle, scene.material, sims.gravity)
        
    def wall_integration(self, sims: Simulation, scene: myScene):
        move_walls_euler_(int(scene.wallNum[0]), sims.dt, scene.wall)

    def verlet_sphere_integration(self, sims: Simulation, scene: myScene):
        move_spheres_verlet_(int(scene.sphereNum[0]), sims.dt, scene.sphere, scene.particle, scene.material, sims.gravity)

    def verlet_clump_integration(self, sims: Simulation, scene: myScene):
        move_clumps_verlet_(int(scene.clumpNum[0]), sims.dt, scene.clump, scene.particle, scene.material, sims.gravity)
        
    def compute_aratio(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        currF = 0.
        if int(scene.sphereNum[0]) > 0:
            currF += calculate_sphere_total_unbalance_force(sims.gravity, int(scene.sphereNum[0]), scene.sphere, scene.particle)
        if int(scene.clumpNum[0]) > 0:
            currF += calculate_clump_total_unbalance_force(sims.gravity, int(scene.clumpNum[0]), scene.clump, scene.particle)

        contactF, contactC = calculate_total_contact_force(int(scene.particleNum[0]), neighbor.particle_particle, self.physpp.cplist)
        if sims.max_wall_num > 0:
            contactFw, contactCw = calculate_total_contact_force(int(scene.particleNum[0]), neighbor.particle_wall, self.physpw.cplist)
            contactF += contactFw
            contactC += contactCw
        return currF * contactC / (contactF * (int(scene.sphereNum[0]) + int(scene.clumpNum[0])))
    
    def compute_mratio(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        currF = 0.
        if int(scene.sphereNum[0]) > 0:
            currF += calculate_sphere_maximum_unbalance_force(sims.gravity, int(scene.sphereNum[0]), scene.sphere, scene.particle)
        if int(scene.clumpNum[0]) > 0:
            currF += calculate_clump_maximum_unbalance_force(sims.gravity, int(scene.clumpNum[0]), scene.clump, scene.particle)

        contactF, contactC = calculate_total_contact_force(int(scene.particleNum[0]), neighbor.particle_particle, self.physpp.cplist)
        if sims.max_wall_num > 0:
            contactFw, contactCw = calculate_total_contact_force(int(scene.particleNum[0]), neighbor.particle_wall, self.physpw.cplist)
            contactF += contactFw
            contactC += contactCw
        return currF * contactC / contactF 


