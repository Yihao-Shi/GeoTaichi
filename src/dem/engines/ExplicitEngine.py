import numpy as np

from src.dem.contact.ContactModelBase import ContactModelBase 
from src.dem.ContactManager import ContactManager
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.engines.EngineKernel import *
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation


class ExplicitEngine(object):
    scene: myScene
    neighbor: NeighborBase
    physpp: ContactModelBase
    physpw: ContactModelBase

    def __init__(self, scene, contactor: ContactManager):
        self.scene = scene
        self.neighbor = contactor.neighbor
        self.physpp = contactor.physpp
        self.physpw = contactor.physpw

        self.update_servo_wall = None
        self.callback = None
        self.calm = None

        self.limit1 = 0.
        self.limit2 = 0.

    def choose_engine(self, sims: Simulation, scene: myScene):
        self.update_neighbor_lists = self.update_neighbor_list
        self.reset_particle_message = self.reset_particle
        if sims.scheme == "LSDEM":
            self.update_neighbor_lists = self.update_LSneighbor_list
            self.reset_particle_message = self.reset_level_set_particle

        self.reset_wall_message = self.no_operation_rest
        if sims.max_wall_num > 0 and sims.wall_type == 1 and sims.servo_status == "On":
            self.reset_wall_message = self.reset_wall
               
        self.calcu_sphere_position = self.no_operation
        self.calcu_clump_position = self.no_operation
        self.calcu_wall_position = self.no_operation       
        if sims.engine == "SymplecticEuler":
            if sims.scheme == "DEM" and sims.max_sphere_num > 0:
                self.calcu_sphere_position = self.euler_sphere_integration
            elif sims.scheme == "LSDEM" and sims.max_rigid_body_num > 0:
                self.calcu_sphere_position = self.euler_level_set_integration
            if sims.scheme == "DEM" and sims.max_clump_num > 0:
                self.calcu_clump_position = self.euler_clump_integration
        elif sims.engine == "VelocityVerlet":
            if sims.scheme == "DEM" and sims.max_sphere_num > 0:
                self.calcu_sphere_position = self.verlet_sphere_integration
            elif sims.scheme == "LSDEM" and sims.max_rigid_body_num > 0:
                self.calcu_sphere_position = self.verlet_level_set_integration
            if sims.scheme == "DEM" and sims.max_clump_num > 0:
                self.calcu_clump_position = self.verlet_clump_integration
        elif sims.engine == "PredictCorrector":
            pass
        else:
            raise RuntimeError("Engine Type is error")
        
        if sims.max_wall_num > 0 and sims.static_wall is False:
            self.calcu_wall_position = self.wall_integration

        self.calm = self.launch_calm1
        if sims.max_clump_num > 0:
            self.calm = self.launch_calm2

        self.is_verlet_update = self.no_operation_rest
        if sims.max_particle_num > 0 and (sims.max_wall_num == 0 or sims.wall_type == 0 or sims.wall_type == 3):
            self.is_verlet_update = scene.is_particle_need_update_verlet_table
        elif sims.max_particle_num == 0 and sims.max_wall_num > 0 and sims.wall_type != 0:
            self.is_verlet_update = scene.is_wall_need_update_verlet_table
        elif sims.max_particle_num > 0 and sims.max_wall_num > 0 and sims.wall_type != 0:
            self.is_verlet_update = scene.is_need_update_verlet_table
        if sims.scheme == "LSDEM":
            self.is_verlet_update_point = self.no_operation_rest
            if sims.max_particle_num > 1:
                if sims.max_wall_num == 0 or sims.wall_type == 3:
                    self.is_verlet_update_point = self.is_point_particle_need_update_verlet_table
                elif sims.max_wall_num > 0:
                    self.is_verlet_update_point = self.is_point_need_update_verlet_table
            elif sims.max_particle_num == 1 and sims.max_wall_num > 0:
                self.is_verlet_update_point = self.is_point_wall_need_update_verlet_table

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

    def launch_calm1(self, current_step, calm_interval, scene: myScene):
        if current_step % calm_interval == 0:
            scene.particle_calm()

    def launch_calm2(self, current_step, calm_interval, scene: myScene):
        if current_step % calm_interval == 0:
            scene.particle_calm()
            scene.clump_calm()

    def reset_particle(self, scene: myScene):
        particle_force_reset_(int(scene.particleNum[0]), scene.particle)

    def reset_level_set_particle(self, scene: myScene):
        particle_force_reset_(int(scene.particleNum[0]), scene.rigid)

    def reset_wall(self, scene: myScene):
        wall_force_reset_(int(scene.wallNum[0]), scene.wall)

    def is_point_need_update_verlet_table(self, limit):
        return self.neighbor.is_particle_particle_point_need_update_verlet_table(limit, self.scene) or self.neighbor.is_particle_wall_point_need_update_verlet_table(limit, self.scene)

    def is_point_particle_need_update_verlet_table(self, limit):
        return self.neighbor.is_particle_particle_point_need_update_verlet_table(limit, self.scene)

    def is_point_wall_need_update_verlet_table(self, limit):
        return self.neighbor.is_particle_wall_point_need_update_verlet_table(limit, self.scene)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        scene.apply_boundary_conditions()
        if sims.scheme == "DEM":
            neighbor.pre_neighbor(scene)
            self.physpp.update_contact_table(sims, scene, neighbor)
            self.physpw.update_contact_table(sims, scene, neighbor)
            neighbor.update_particle_particle_auxiliary_lists()
            neighbor.update_particle_wall_auxiliary_lists()
            self.limit1 = sims.verlet_distance * sims.verlet_distance
        elif sims.scheme == "LSDEM":
            neighbor.pre_neighbor(scene)
            self.physpp.update_verlet_particle_particle_tables(sims, scene, neighbor)
            self.physpw.update_verlet_particle_wall_tables(sims, scene, neighbor)
            neighbor.update_point_verlet_table(scene)
            self.physpp.update_contact_table(sims, scene, neighbor)
            self.physpw.update_contact_table(sims, scene, neighbor)
            neighbor.update_particle_particle_auxiliary_lists()
            neighbor.update_particle_wall_auxiliary_lists()
            self.limit1 = sims.verlet_distance * sims.verlet_distance
            self.limit2 = sims.point_verlet_distance * sims.point_verlet_distance

    def update_neighbor_list(self, sims, scene: myScene, neighbor: NeighborBase):
        if self.is_verlet_update(self.limit1) == 1:
            self.update_verlet_table(sims, scene, neighbor)
        self.physpp.resolve(sims, scene, neighbor)
        self.physpw.resolve(sims, scene, neighbor)

    def update_verlet_table(self, sims, scene: myScene, neighbor: NeighborBase):
        scene.apply_boundary_conditions()
        neighbor.update_verlet_table(scene)
        self.physpp.update_contact_table(sims, scene, neighbor)
        self.physpw.update_contact_table(sims, scene, neighbor)
        neighbor.update_particle_particle_auxiliary_lists()
        neighbor.update_particle_wall_auxiliary_lists()

    def update_LSneighbor_list(self, sims, scene: myScene, neighbor: NeighborBase):
        if self.is_verlet_update(self.limit1) == 1:
            self.update_LSDEM_verlet_table1(sims, scene, neighbor)
            self.update_LSDEM_verlet_table2(sims, scene, neighbor)
        elif self.is_verlet_update_point(self.limit2) == 1:
            self.update_LSDEM_verlet_table2(sims, scene, neighbor)
        self.physpp.resolve(sims, scene, neighbor)
        self.physpw.resolve(sims, scene, neighbor)

    def update_LSDEM_verlet_table1(self, sims, scene: myScene, neighbor: NeighborBase):
        scene.apply_boundary_conditions()
        neighbor.update_verlet_table(scene)
        self.physpp.update_verlet_particle_particle_tables(sims, scene, neighbor)
        self.physpw.update_verlet_particle_wall_tables(sims, scene, neighbor)

    def update_LSDEM_verlet_table2(self, sims, scene: myScene, neighbor: NeighborBase):
        neighbor.update_point_verlet_table(scene)
        self.physpp.update_contact_table(sims, scene, neighbor)
        self.physpw.update_contact_table(sims, scene, neighbor)
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

    def euler_level_set_integration(self, sims: Simulation, scene: myScene):
        move_level_set_euler_(int(scene.particleNum[0]), sims.dt, scene.particle, scene.rigid, scene.material, sims.gravity)
        
    def wall_integration(self, sims: Simulation, scene: myScene):
        move_walls_euler_(int(scene.wallNum[0]), sims.dt, scene.wall)

    def verlet_sphere_integration(self, sims: Simulation, scene: myScene):
        move_spheres_verlet_(int(scene.sphereNum[0]), sims.dt, scene.sphere, scene.particle, scene.material, sims.gravity)

    def verlet_clump_integration(self, sims: Simulation, scene: myScene):
        move_clumps_verlet_(int(scene.clumpNum[0]), sims.dt, scene.clump, scene.particle, scene.material, sims.gravity)

    def verlet_level_set_integration(self, sims: Simulation, scene: myScene):
        move_level_set_verlet_(int(scene.particleNum[0]), sims.dt, scene.particle, scene.rigid, scene.material, sims.gravity)
        
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


