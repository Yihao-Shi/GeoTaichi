import taichi as ti
import numpy as np

from src.dem.BaseStruct import *
from src.dem.Simulation import Simulation
from src.dem.BaseKernel import *


class myScene(object):
    clump: ClumpFamily
    particle: ParticleFamily
    sphere: SphereFamily
    wall: PlaneFamily or FacetFamily or PatchFamily
    servo: ServoWall
    material: Material

    def __init__(self) -> None:
        self.particle = None
        self.clump = None
        self.sphere = None
        self.wall = None
        self.grid = None
        self.rigid = None
        self.soft = None
        self.box = None
        self.surface_node = None
        self.relation = None
        self.material = None
        self.neighbor = None
        self.servo = None
        self.vispts = []
        self.vistri = []

        self.sphereNum = np.zeros(1, dtype=np.int32)
        self.clumpNum = np.zeros(1, dtype=np.int32)
        self.rigidNum = np.zeros(1, dtype=np.int32)
        self.particleNum = np.zeros(1, dtype=np.int32)
        self.wallNum = np.zeros(1, dtype=np.int32)
        self.servoNum = np.zeros(1, dtype=np.int32)
        self.gridNum = np.zeros(1, dtype=np.int32)
        self.surfaceNum = np.zeros(1, dtype=np.int32)
    
    def activate_basic_class(self, sims: Simulation):
        if sims.max_material_num > 0:
            self.activate_material(sims)
            
        if sims.max_sphere_num > 0: 
            self.activate_sphere(sims)
        if sims.max_clump_num > 0: 
            self.activate_clump(sims)

        if sims.max_level_grid_num > 0:
            self.activate_levelset_grid(sims)
        if sims.max_rigid_body_num > 0:
            self.activate_rigid_body(sims)
            self.active_boundings(sims)

        if sims.wall_type == 0: 
            self.activate_plane(sims)
        elif sims.wall_type == 1: 
            self.activate_facet(sims)
            self.activate_servo_control(sims)
        elif sims.wall_type == 2: 
            self.activate_patch(sims)
            self.activate_servo_control(sims)

    def activate_material(self, sims: Simulation):
        self.material = Material.field(shape=sims.max_material_num)

    def activate_particle(self, sims: Simulation):
        if self.particle is None:
            if sims.max_particle_num > 0:
                self.particle = ParticleFamily.field(shape=sims.max_particle_num)
            else:
                raise RuntimeError(f"KeyWord:: /max_particle_num/ should be larger than 0")

    def activate_sphere(self, sims: Simulation):    
        if self.sphere is None:
            if self.particle is None:
                self.activate_particle(sims)
            self.sphere = SphereFamily.field(shape=sims.max_sphere_num)

    def activate_clump(self, sims: Simulation):
        if self.clump is None:
            if self.particle is None:
                self.activate_particle(sims)
            self.clump = ClumpFamily.field(shape=sims.max_clump_num)

    def activate_levelset_grid(self, sims: Simulation):
        if self.grid is None and sims.max_level_grid_num > 0:
            self.grid = LevelSetGrid.field(shape=sims.max_level_grid_num)

    def activate_rigid_body(self, sims: Simulation):
        if self.rigid is None and sims.max_rigid_body_num > 0:
            self.rigid = RigidBody.field(shape=sims.max_rigid_body_num)

            if self.grid is None: 
                raise RuntimeError("Level set grid has not been set!")
            
            if sims.max_level_grid_num <= 0:
                raise RuntimeError("Keyword:: /max_levelset_grid_num/ should be larger than 0")
            
            if self.surface_node is None:
                if sims.max_surface_node_num > 0:
                    self.surface_node = SurfaceNode.field(shape=sims.max_surface_node_num)
                else:
                    raise RuntimeError("Keyword:: /max_surface_node_num/ should be larger than 0")

    def active_boundings(self, sims: Simulation):
        if self.particle is None and sims.max_rigid_body_num > 0:
            self.particle = BoundingSphere.field(shape=sims.max_rigid_body_num)
            if sims.max_particle_num > 0.:
                raise RuntimeError("Level set DEM cannot run ")
            
        if self.box is None and sims.max_rigid_body_num > 0:
            self.box = BoundingBox.field(shape=sims.max_rigid_body_num)
                
    def activate_plane(self, sims: Simulation):
        if self.wall is None and not sims.wall_type is None:
            self.wall = PlaneFamily.field(shape=sims.max_wall_num)

    def activate_facet(self, sims: Simulation):
        if self.wall is None and not sims.wall_type is None:
            self.wall = FacetFamily.field(shape=sims.max_wall_num)

    def activate_patch(self, sims: Simulation):
        if self.wall is None and not sims.wall_type is None:
            self.wall = PatchFamily.field(shape=sims.max_wall_num)

    def activate_servo_control(self, sims: Simulation):
        if self.servo is None and sims.max_servo_wall_num > 0:
            self.servo = ServoWall.field(shape=sims.max_servo_wall_num)

    def get_material_ptr(self):
        return self.material
    
    def get_particle_ptr(self):
        return self.particle
    
    def get_sphere_ptr(self):
        return self.sphere
    
    def get_clump_ptr(self):
        return self.clump
    
    def get_grid_ptr(self):
        return self.grid
    
    def get_rigid_ptr(self):
        return self.rigid
    
    def get_wall_ptr(self):
        return self.wall
    
    def get_bounding_box(self):
        return self.box
    
    def get_bounding_sphere(self):
        return self.particle
    
    def get_surface_node(self):
        return self.surface_node
    
    def check_particle_num(self, sims: Simulation, particle_number):
        if self.particleNum[0] + particle_number > sims.max_particle_num:
            raise ValueError ("The DEM particles should be set as: ", self.particleNum[0] + particle_number)
    
    def check_sphere_number(self, sims: Simulation, body_number):
        if self.sphereNum[0] + body_number > sims.max_sphere_num:
            raise ValueError ("The DEM spheres should be set as: ", self.sphereNum[0] + body_number)
        
    def check_clump_number(self, sims: Simulation, body_number):
        if self.clumpNum[0] + body_number > sims.max_clump_num:
            raise ValueError ("The DEM clumps should be set as: ", self.clumpNum[0] + body_number)
        
    def check_grid_number(self, sims: Simulation, grid_number):
        if self.gridNum[0] + grid_number > sims.max_level_grid_num:
            raise ValueError ("The level set grid should be set as: ", self.gridNum[0] + grid_number)
        
    def check_rigid_number(self, sims: Simulation, body_number):
        if self.rigidNum[0] + body_number > sims.max_rigid_body_num:
            raise ValueError ("The rigid bodies should be set as: ", self.rigidNum[0] + body_number)
        
    def check_surface_node_number(self, sims: Simulation, surface_node_number):
        if self.surfaceNum[0] + surface_node_number > sims.max_surface_node_num:
            raise ValueError ("The surface nodes should be set as: ", self.surfaceNum[0] + surface_node_number)

    def check_wall_number(self, sims: Simulation, body_number):
        if self.wallNum[0] + body_number > sims.max_wall_num:
            raise ValueError ("The DEM walls should be set as: ", self.wallNum[0] + body_number)
        
    def check_servo_number(self, sims: Simulation, body_number):
        if self.servoNum[0] + body_number > sims.max_servo_wall_num:
            raise ValueError ("The DEM servos should be set as: ", self.servoNum[0] + body_number)

    def add_attribute(self, sims: Simulation, materialID, attribute_dict):
        print(" Body Attribute Information ".center(71, '-'))
        if materialID > sims.max_material_num - 1:
            raise RuntimeError("Material ID is out of the scope!")
        else:
            self.material[materialID].add_attribute(attribute_dict)
            self.material[materialID].print_info(materialID)

    def find_min_density(self):
        mindensity = 1e15
        for nm in range(self.material.shape[0]):
            if self.material[nm].density > 0:
                mindensity = ti.min(mindensity, self.material[nm].density)
        return mindensity
    
    def find_bounding_sphere_radius(self, sims: Simulation):
        rad_max = 0.
        rad_min = MThreshold

        if self.particleNum[0] > 0:
            rad_max = max(rad_max, find_particle_max_radius_(int(self.particleNum[0]), self.particle))
            rad_min = min(rad_min, find_particle_min_radius_(int(self.particleNum[0]), self.particle))

        if sims.wall_type == 2 and self.wallNum[0] > 0:
            rad_max = max(rad_max, find_patch_max_radius_(int(self.wallNum[0]), self.wall))
            rad_min = min(rad_min, find_patch_min_radius_(int(self.wallNum[0]), self.wall))
        return rad_min, rad_max
    
    def reset_verlet_disp(self):
        reset_verlet_disp_(int(self.particleNum[0]), self.particle)
        reset_verlet_disp_(int(self.wallNum[0]), self.wall)
    
    def reset_particle_verlet_disp(self):
        reset_verlet_disp_(int(self.particleNum[0]), self.particle)

    def reset_wall_verlet_disp(self):
        reset_verlet_disp_(int(self.wallNum[0]), self.wall)

    def is_need_update_verlet_table(self, limit):
        return validate_displacement_(limit, int(self.wallNum[0]), self.wall) or validate_displacement_(limit, int(self.particleNum[0]), self.particle)

    def is_wall_need_update_verlet_table(self, limit):
        return validate_displacement_(limit, int(self.wallNum[0]), self.wall)

    def is_particle_need_update_verlet_table(self, limit):
        return validate_displacement_(limit, int(self.particleNum[0]), self.particle)

    def find_particle_min_radius(self):
        return find_particle_min_radius_(int(self.particleNum[0]), self.particle)
    
    def find_particle_max_radius(self):
        return find_particle_max_radius_(int(self.particleNum[0]), self.particle)
    
    def find_min_radius(self, sims: Simulation):
        rad_min = MThreshold

        if self.particleNum[0] > 0:
            rad_min = min(rad_min, find_particle_min_radius_(int(self.particleNum[0]), self.particle))

        if sims.wall_type == 2:
            rad_min = min(rad_min, find_patch_min_radius_(int(self.wallNum[0]), self.wall))
        return rad_min
    
    def find_max_radius(self, sims: Simulation):
        rad_max = 0.

        if self.particleNum[0] > 0:
            rad_max = max(rad_max, find_particle_max_radius_(int(self.particleNum[0]), self.particle))
        
        if sims.wall_type == 2 and self.wallNum[0] > 0:
            rad_max = max(rad_max, find_patch_max_radius_(int(self.wallNum[0]), self.wall))
        return rad_max

    def find_particle_min_mass(self):
        return find_particle_min_mass_(int(self.particleNum[0]), self.particle)
    
    def check_particle_in_domain(self, sims: Simulation):
        check_in_domain(sims.domain, int(self.particleNum[0]), self.particle)

    def update_particle_properties(self, override, particle_type, property_name, value, bodyID):
        print(" Modify Particle Information ".center(71, '-'))
        print("Target BodyID =", bodyID)
        print("Target Property =", property_name)
        print("Target Value =", value)
        print("Override =", override, '\n')

        override = 1 if not override else 0
        if particle_type == "Sphere":
            if property_name == "bodyID":
                modify_sphere_bodyID(value, int(self.particleNum[0]), self.particle, bodyID)
            elif property_name == "groupID":
                modify_sphere_groupID(value, int(self.particleNum[0]), self.particle, bodyID)
            elif property_name == "materialID":
                modify_sphere_materialID(value, int(self.particleNum[0]), self.particle, self.material, bodyID)
            elif property_name == "radius":
                modify_sphere_radius(value, int(self.particleNum[0]), self.particle, bodyID)
            elif property_name == "position":
                modify_sphere_position(override, value, int(self.particleNum[0]), self.particle, bodyID)
            elif property_name == "velocity":
                modify_sphere_velocity(override, value, int(self.particleNum[0]), self.particle, bodyID)
            elif property_name == "angular_velocity":
                modify_sphere_angular_velocity(override, value, int(self.particleNum[0]), self.particle, bodyID)
            elif property_name == "orientation":
                modify_sphere_orientation(override, value, int(self.particleNum[0]), self.particle, bodyID)
            elif property_name == "fix_velocity":
                FIX = {
                        "Free": 0,
                        "Fix": 1
                    }
                fix_v = vec3u8([DictIO.GetEssential(FIX, is_fix) for is_fix in value])
                modify_sphere_fix_v(fix_v, int(self.particleNum[0]), self.particle, bodyID)
            elif property_name == "fix_angular_velocity":
                FIX = {
                        "Free": 0,
                        "Fix": 1
                    }
                fix_v = vec3u8([DictIO.GetEssential(FIX, is_fix) for is_fix in value])
                modify_sphere_fix_w(fix_v, int(self.particleNum[0]), self.particle, bodyID)
            else:
                valid_list = ["bodyID", "materialID", "position", "velocity", "traction", "stress", "fix_velocity"]
                raise KeyError(f"Invalid property_name: {property_name}! Only the following keywords is valid: {valid_list}")
        elif particle_type == "Clump":
            pass

    def update_particle_properties_in_region(self, override, particle_type, property_name, value, is_in_region):
        print(" Modify Particle Information ".center(71, '-'))
        print("Target Property =", property_name)
        print("Target Value =", value)
        print("Override =", override, '\n')

        override = 1 if not override else 0
        if particle_type == "Sphere":
            if property_name == "bodyID":
                modify_sphere_bodyID_in_region(value, int(self.particleNum[0]), self.particle, is_in_region)
            elif property_name == "groupID":
                modify_sphere_groupID_in_region(value, int(self.particleNum[0]), self.particle, is_in_region)
            elif property_name == "materialID":
                modify_sphere_materialID_in_region(value, int(self.particleNum[0]), self.particle, self.material, is_in_region)
            elif property_name == "radius":
                modify_sphere_radius_in_region(value, int(self.particleNum[0]), self.particle, is_in_region)
            elif property_name == "position":
                modify_sphere_position_in_region(override, value, int(self.particleNum[0]), self.particle, is_in_region)
            elif property_name == "velocity":
                modify_sphere_velocity_in_region(override, value, int(self.particleNum[0]), self.particle, is_in_region)
            elif property_name == "angular_velocity":
                modify_sphere_angular_velocity_in_region(override, value, int(self.particleNum[0]), self.particle, is_in_region)
            elif property_name == "orientation":
                modify_sphere_orientation_in_region(override, value, int(self.particleNum[0]), self.particle, is_in_region)
            elif property_name == "fix_velocity":
                FIX = {
                        "Free": 0,
                        "Fix": 1
                    }
                fix_v = vec3u8([DictIO.GetEssential(FIX, is_fix) for is_fix in value])
                modify_sphere_fix_v_in_region(fix_v, int(self.particleNum[0]), self.particle, is_in_region)
            elif property_name == "fix_angular_velocity":
                FIX = {
                        "Free": 0,
                        "Fix": 1
                    }
                fix_v = vec3u8([DictIO.GetEssential(FIX, is_fix) for is_fix in value])
                modify_sphere_fix_w_in_region(fix_v, int(self.particleNum[0]), self.particle, is_in_region)
            else:
                valid_list = ["bodyID", "materialID", "position", "velocity", "traction", "stress", "fix_velocity"]
                raise KeyError(f"Invalid property_name: {property_name}! Only the following keywords is valid: {valid_list}")
        elif particle_type == "Clump":
            pass

    def update_wall_properties(self, sims: Simulation, override, property_name, value, wallID):
        print(" Modify Wall Information ".center(71, '-'))
        print("Target Wall ID =", wallID)
        print("Target Property =", property_name)
        print("Target Value =", value)
        print("Override =", override, '\n')

        override = 1 if not override else 0
        if property_name == "Status":
            value = 1 if value == "On" else 0
            modify_wall_activate_status(wallID, value, int(self.wallNum[0]), self.wall)
        elif property_name == "MaterialID":
            modify_wall_materialID(wallID, value, int(self.wallNum[0]), self.wall)

        if sims.wall_type == 0:
            if property_name == "Position":
                modify_plane_position(override, wallID, vec3f(value), int(self.wallNum[0]), self.wall)
            elif property_name == "Orientation":
                modify_plane_orientation(override, wallID, vec3f(value), int(self.wallNum[0]), self.wall)
        elif sims.wall_type == 1 or sims.wall_type == 2:
            if property_name == "Position":
                modify_triangle_position(override, wallID, vec3f(value), int(self.wallNum[0]), self.wall)
            elif property_name == "Velocity":
                modify_triangle_velocity(override, wallID, vec3f(value), int(self.wallNum[0]), self.wall)
            elif property_name == "Orientation":
                mode = 0
                new_direction = vec3f(0, 0, 0)
                rotation_center = vec3f(0, 0, 0)
                if isinstance(value, dict):
                    mode = 0
                    new_direction = vec3f(DictIO.GetEssential(value, "Orientation"))
                    rotation_center = vec3f(DictIO.GetEssential(value, "RotateCenter"))
                elif isinstance(value, np.ndarray, tuple, list, ti.types.vector):
                    mode = 1
                    new_direction = vec3f(value)
                else:
                    raise KeyError("Input error:: /value/ should be input as dict, np.ndarray, tuple, list or ti.types.vector")

                modify_triangle_orientation(mode, override, wallID, new_direction, rotation_center, int(self.wallNum[0]), self.wall)

    def particle_calm(self):
        particle_calm(int(self.particleNum[0]), self.particle)

    def clump_calm(self):
        clump_calm(int(self.clumpNum[0]), self.clump)
    

    
