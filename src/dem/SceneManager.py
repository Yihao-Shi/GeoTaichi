import taichi as ti
import numpy as np

from src.dem.BaseStruct import *
from src.dem.Simulation import Simulation
from src.dem.BaseKernel import *
from src.sdf.BasicShape import BasicShape
from src.utils.DomainBoundary import DomainBoundary
from src.utils.linalg import no_operation

from functools import partial
from multiprocessing.pool import ThreadPool as Pool


class myScene(object):
    domain_boundary: DomainBoundary
    clump: ClumpFamily
    particle: ParticleFamily
    sphere: SphereFamily
    wall: PlaneFamily or FacetFamily or PatchFamily
    servo: ServoWall
    material: Material

    def __init__(self) -> None:
        self.domain_boundary = None
        self.particle = None
        self.clump = None
        self.sphere = None
        self.wall = None
        self.rigid_grid = None
        self.soft_grid = None
        self.rigid = None
        self.soft = None
        self.box = None
        self.wallbody = None
        self.vertice = None
        self.relation = None
        self.material = None
        self.neighbor = None
        self.servo = None
        self.surface = None
        self.levelset = None
        self.digital_elevation = None
        self.apply_boundary_conditions = None
        self.connectivity = np.array([], dtype=np.int32)
        self.prefixID = {}
        self.distance_field = []
        self.vertice_paras = []
        self.gridID = []
        self.verticeID = []

        self.sphereNum = np.zeros(1, dtype=np.int32)
        self.clumpNum = np.zeros(1, dtype=np.int32)
        self.rigidNum = np.zeros(1, dtype=np.int32)
        self.softNum = np.zeros(1, dtype=np.int32)
        self.particleNum = np.zeros(1, dtype=np.int32)
        self.wallNum = np.zeros(1, dtype=np.int32)
        self.servoNum = np.zeros(1, dtype=np.int32)
        self.surfaceNum = np.zeros(1, dtype=np.int32)
        self.gridNum = np.zeros(1, dtype=np.int32)
    
    def activate_basic_class(self, sims: Simulation):
        if sims.max_material_num > 0:
            self.activate_material(sims)
            
        if sims.scheme == "DEM":
            self.activate_particle(sims)
            if sims.max_sphere_num > 0: 
                self.activate_sphere(sims)
            if sims.max_clump_num > 0: 
                self.activate_clump(sims)
        
        if sims.scheme == "LSDEM":
            if sims.max_level_grid_num > 0:
                self.activate_levelset_grid(sims)
            self.activate_levelset_body(sims)
            self.active_boundings(sims)

        if sims.wall_type == 0: 
            self.activate_plane(sims)
        elif sims.wall_type == 1: 
            self.activate_facet(sims)
            self.activate_servo_control(sims)
        elif sims.wall_type == 2: 
            self.activate_patch(sims)
            self.activate_servo_control(sims)
        elif sims.wall_type == 3:
            self.activate_digital_elevation(sims)

    def activate_material(self, sims: Simulation):
        self.material = Material.field(shape=sims.max_material_num)

    def activate_particle(self, sims: Simulation):
        if self.particle is None:
            if sims.max_particle_num >= 0:
                ptemp = ParticleFamily
                if sims.energy_tracking:
                    ptemp.members.update({"elastic_energy": float, "friction_energy": float, "damp_energy": float})
                self.particle = ptemp.field(shape=max(sims.max_particle_num, 1))
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
        if sims.max_level_grid_num > 0:
            if sims.scheme == "LSDEM" or sims.scheme == "LSMPM":
                if sims.max_rigid_template_num > 0:
                    self.rigid_grid = LevelSetGrid.field(shape=sims.max_level_grid_num * sims.max_rigid_template_num)
                if sims.max_rigid_template_num <= 0 :
                    if sims.scheme == "LSDEM":
                        raise RuntimeError("Keyword:: /max_template_number/ must be larger than 0")

    def activate_levelset_body(self, sims: Simulation):
        if self.rigid is None and sims.max_rigid_body_num > 0:
            ptemp = RigidBody
            if sims.energy_tracking:
                ptemp.members.update({"elastic_energy": float, "friction_energy": float, "damp_energy": float})
            self.rigid = ptemp.field(shape=max(sims.max_rigid_body_num, 1))

            if self.rigid_grid is None: 
                raise RuntimeError("Level set grid has not been set!")
            
            if self.soft_grid is None and sims.scheme == "LSMPM": 
                raise RuntimeError("Deformable level set grid has not been set!")
            
            if sims.max_level_grid_num <= 0:
                raise RuntimeError("Keyword:: /max_levelset_grid_num/ should be larger than 0")
            
            if self.vertice is None:
                if sims.max_rigid_template_num > 0:
                    if sims.max_surface_node_num > 0:
                        self.vertice = VerticeNode.field(shape=sims.max_surface_node_num * sims.max_rigid_template_num)
                        self.surface = ti.field(int, shape=sims.max_surface_node_num * sims.max_rigid_body_num)
                    else:
                        raise RuntimeError("Keyword:: /max_surface_node_num/ should be larger than 0")

            if sims.max_level_grid_num <= 0:
                raise RuntimeError("Keyword:: /max_levelset_grid_num/ should be larger than 0")
                
    def activate_levelset_visualization(self, sims: Simulation):
        if sims.scheme == "LSDEM":
            if self.vertice is None:
                raise RuntimeError("Surface node should be activated first")
            else:
                if "surface" in sims.monitor_type and sims.visualize:
                    self.visualzie_surface_node = ti.Vector.field(3, float, shape=sims.max_surface_node_num * (sims.max_rigid_body_num))

    def active_boundings(self, sims: Simulation):
        if self.particle is None and (sims.max_rigid_body_num > 0 ):
            self.particle = BoundingSphere.field(shape=(sims.max_rigid_body_num))

            if sims.max_sphere_num > 0. or sims.max_clump_num > 0.:
                raise RuntimeError("Level set DEM cannot run. Bacause spheres or clumps are included in the simulation")
            
        if self.box is None and (sims.max_rigid_body_num > 0):
            self.box = BoundingBox.field(shape=(sims.max_rigid_body_num))
                
    def activate_plane(self, sims: Simulation):
        if self.wall is None and not sims.wall_type is None:
            self.wall = PlaneFamily.field(shape=sims.max_wall_num)

    def activate_facet(self, sims: Simulation):
        if self.wall is None and not sims.wall_type is None:
            self.wall = FacetFamily.field(shape=sims.max_wall_num)

    def activate_patch(self, sims: Simulation):
        if self.wall is None and not sims.wall_type is None:
            self.wall = PatchFamily.field(shape=sims.max_wall_num)
            if sims.search == "HierarchicalLinkedCell":
                self.wallbody = ti.field(ti.u8, shape=sims.max_wall_num)

    def activate_servo_control(self, sims: Simulation):
        if self.servo is None and sims.max_servo_wall_num > 0:
            self.servo = ServoWall.field(shape=sims.max_servo_wall_num)

    def activate_digital_elevation(self, sims: Simulation):
        if self.wall is None and sims.max_wall_num > 0.:
            self.digital_elevation = DigitalElevationModel()
            self.wall = PatchFamily.field(shape=sims.max_wall_num)

    def get_material_ptr(self):
        return self.material
    
    def get_particle_ptr(self):
        return self.particle
    
    def get_sphere_ptr(self):
        return self.sphere
    
    def get_clump_ptr(self):
        return self.clump
    
    def get_grid_ptr(self):
        return self.rigid_grid
    
    def get_rigid_ptr(self):
        return self.rigid
    
    def get_wall_ptr(self):
        return self.wall
    
    def get_bounding_box(self):
        return self.box
    
    def get_bounding_sphere(self):
        return self.particle
    
    def get_vertice(self):
        return self.vertice
    
    def get_surface(self):
        return self.surface
    
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
        if grid_number > sims.max_level_grid_num:
            raise ValueError ("The level set grid should be set as: ", grid_number)
        
    def check_rigid_body_number(self, sims: Simulation, rigid_body_number):
        if self.rigidNum[0] + rigid_body_number > sims.max_rigid_body_num:
            raise ValueError ("The rigid bodies should be set as: ", self.rigidNum[0] + rigid_body_number)
        
    def check_surface_node_number(self, sims: Simulation, surface_node_number):
        if surface_node_number > sims.max_surface_node_num :
            raise ValueError ("The surface nodes should be set as: ", surface_node_number)
        
    def check_wall_number(self, sims: Simulation, body_number):
        if self.wallNum[0] + body_number > sims.max_wall_num:
            raise ValueError ("The DEM walls should be set as: ", self.wallNum[0] + body_number)
        
    def check_servo_number(self, sims: Simulation, body_number):
        if self.servoNum[0] + body_number > sims.max_servo_wall_num:
            raise ValueError ("The DEM servos should be set as: ", self.servoNum[0] + body_number)
        
    def check_rigid_template_number(self, sims: Simulation, rigid_template_number):
        if rigid_template_number > sims.max_rigid_template_num:
            raise ValueError ("The number of rigid LSDEM templates should be set as: ", rigid_template_number)

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
    
    def find_expect_extent(self, sims: Simulation, distance):
        if sims.scheme == "LSDEM":
            temp = find_expect_extent_(distance, int(self.particleNum[0]), self.box)
            return temp[0], temp[1]
        return None
    
    def find_min_grid_space(self, sims: Simulation):
        if sims.scheme == "LSDEM":
            return find_min_grid_space_(int(self.particleNum[0]), self.box)
        return None
    
    def find_min_extent(self, sims: Simulation):
        if sims.scheme == "LSDEM":
            return find_min_extent_(int(self.particleNum[0]), self.box)
        return None
    
    def find_particle_radius(self, scheme):
        return self.find_particle_min_radius(scheme), self.find_particle_max_radius(scheme)

    def find_particle_min_radius(self, scheme):
        if scheme == "LSDEM":
            return find_particle_min_radius_(int(self.particleNum[0]), self.rigid)
        elif scheme == "DEM":
            return find_particle_min_radius_(int(self.particleNum[0]), self.particle)
    
    def find_particle_max_radius(self, scheme):
        if scheme == "LSDEM":
            return find_particle_max_radius_(int(self.particleNum[0]), self.rigid)
        elif scheme == "DEM":
            return find_particle_max_radius_(int(self.particleNum[0]), self.particle)
    
    def find_bounding_sphere_radius(self, sims: Simulation):
        return self.find_bounding_sphere_min_radius(sims), self.find_bounding_sphere_max_radius(sims)
    
    def find_bounding_sphere_min_radius(self, sims: Simulation):
        rad_min = MThreshold

        if self.particleNum[0] > 0:
            rad_min = min(rad_min, find_particle_min_radius_(int(self.particleNum[0]), self.particle))
        else:
            rad_min = 0.

        if sims is not None:
            if sims.wall_type == 2 and self.wallNum[0] > 0:
                rad_min = min(rad_min, find_patch_min_radius_(int(self.wallNum[0]), self.wall))
        return rad_min
    
    def find_bounding_sphere_max_radius(self, sims: Simulation):
        rad_max = 0.

        if self.particleNum[0] > 0:
            rad_max = max(rad_max, find_particle_max_radius_(int(self.particleNum[0]), self.particle))
        
        if sims is not None:
            if sims.wall_type == 2 and self.wallNum[0] > 0:
                rad_max = max(rad_max, find_patch_max_radius_(int(self.wallNum[0]), self.wall))
        return rad_max

    def find_particle_min_mass(self, scheme):
        if scheme == "LSDEM":
            return find_particle_min_mass_(int(self.particleNum[0]), self.rigid)
        elif scheme == "DEM":
            return find_particle_min_mass_(int(self.particleNum[0]), self.particle)
        
    def print_delete_particles(self, sphere_num, clump_num):
        if sphere_num > 0:
            print(f"Total {sphere_num} spheres have been deleted! Remaining {self.particleNum[0]} particles and {self.sphereNum[0]} spheres.")
        if clump_num > 0:
            print(f"Total {clump_num} clumps have been deleted! Remaining {self.particleNum[0]} particles and {self.clumpNum[0]} clumps.")

    def update_material_properties(self, override, materialID, property_name, value):
        print(" Modify Material Information ".center(71, '-'))
        print("Target materialID =", materialID)
        print("Target Property =", property_name)
        print("Target Value =", value)
        print("Override =", override, '\n')

        factor = 1 if not override else 0
        if property_name == "Density":
            modify_material_density(factor, value, materialID, self.material)
        elif property_name == "ForceLocalDamping":
            modify_material_fdamp(factor, value, materialID, self.material)
        elif property_name == "TorqueLocalDamping":
            modify_material_tdamp(factor, value, materialID, self.material)

    def update_particle_properties(self, override, particle_type, property_name, value, bodyID):
        print(" Modify Particle Information ".center(71, '-'))
        print("Target BodyID =", bodyID)
        print("Target Property =", property_name)
        print("Target Value =", value)
        print("Override =", override, '\n')

        factor = 1 if not override else 0
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
                modify_sphere_position(factor, value, int(self.particleNum[0]), self.particle, bodyID)
            elif property_name == "velocity":
                modify_sphere_velocity(factor, value, int(self.particleNum[0]), self.particle, bodyID)
            elif property_name == "angular_velocity":
                modify_sphere_angular_velocity(factor, value, int(self.particleNum[0]), self.particle, bodyID)
            elif property_name == "orientation":
                modify_sphere_orientation(factor, value, int(self.particleNum[0]), self.particle, bodyID)
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

        factor = 1 if not override else 0
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
                modify_sphere_position_in_region(factor, value, int(self.particleNum[0]), self.particle, is_in_region)
            elif property_name == "velocity":
                modify_sphere_velocity_in_region(factor, value, int(self.particleNum[0]), self.particle, is_in_region)
            elif property_name == "angular_velocity":
                modify_sphere_angular_velocity_in_region(factor, value, int(self.particleNum[0]), self.particle, is_in_region)
            elif property_name == "orientation":
                modify_sphere_orientation_in_region(factor, value, int(self.particleNum[0]), self.particle, is_in_region)
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

        factor = 1 if not override else 0
        if property_name == "Status":
            value = 1 if value == "On" else 0
            modify_wall_activate_status(wallID, value, int(self.wallNum[0]), self.wall)
        elif property_name == "MaterialID":
            modify_wall_materialID(wallID, value, int(self.wallNum[0]), self.wall)

        if sims.wall_type == 0:
            if property_name == "Position":
                modify_plane_position(factor, wallID, vec3f(value), int(self.wallNum[0]), self.wall)
            elif property_name == "Orientation":
                modify_plane_orientation(factor, wallID, vec3f(value), int(self.wallNum[0]), self.wall)
        elif sims.wall_type == 1 or sims.wall_type == 2:
            if property_name == "Position":
                modify_triangle_position(factor, wallID, vec3f(value), int(self.wallNum[0]), self.wall)
            elif property_name == "Velocity":
                modify_triangle_velocity(factor, wallID, vec3f(value), int(self.wallNum[0]), self.wall)
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

                modify_triangle_orientation(mode, factor, wallID, new_direction, rotation_center, int(self.wallNum[0]), self.wall)

    def add_rigid_levelset_template(self, name, distance_field, vertice, parameter):
        currID = len(self.distance_field)
        DictIO.append(self.prefixID, name, currID)
        self.distance_field.append(distance_field)
        self.vertice_paras.append([vertice, parameter])

    def add_rigid_template_grid_field(self, sims):
        self.gridID = []
        self.verticeID = []
        gridSums, nodeSums = 0, 0
        for name in self.prefixID:
            index = DictIO.GetEssential(self.prefixID, name)
            distance_field = self.distance_field[index]
            vertice, parameter = self.vertice_paras[index]
            assert vertice.shape[0] == parameter.shape[0], "The dimension of vertice and parameter is wrong!"
            gridSum = distance_field.shape[0]
            nodeSum = parameter.shape[0]
            self.check_grid_number(sims, gridSum)
            self.check_surface_node_number(sims, nodeSum)
            create_level_set_grids_(gridSums, gridSum, self.rigid_grid, distance_field)
            create_level_set_surface(nodeSums, nodeSum, self.vertice, vertice, parameter)
            self.gridID.append(gridSums)
            self.verticeID.append(nodeSums)
            gridSums += gridSum
            nodeSums += nodeSum
        self.gridID.append(gridSums)
        self.verticeID.append(nodeSums)
        self.check_rigid_template_number(sims, len(self.prefixID))

    def add_connectivity(self, body_count, surface_node_number, objects: BasicShape):
        pool = Pool()
        func = partial(GetConnectivity, surface_node_number, objects.mesh.faces)
        faces = np.array(pool.map(func, range(body_count)), dtype=np.int32).reshape(-1, 3)
        pool.terminate()
        pool.join()
        self.connectivity = np.append(self.connectivity, faces + self.surfaceNum[0]).reshape(-1, 3)
        return faces

    def particle_calm(self):
        particle_calm(int(self.particleNum[0]), self.particle)

    def clump_calm(self):
        clump_calm(int(self.clumpNum[0]), self.clump)

    def visualize_surface(self):
        kernel_visualize_surface_(int(self.particleNum[0]), self.vertice, self.visualzie_surface_node, self.rigid, self.box)

    def check_radius(self):
        check = check_radius_(int(self.particleNum[0]), self.rigid)
        if check is False:
            raise RuntimeError("The equivalent radius should be sorted in ascending order. Please double check create_body and add_body")
        
    def delete_particles(self, bodyID):
        initial_particle = self.particleNum[0]
        kernel_delete_particles(self.particleNum[0], self.particle, bodyID)
        self.particleNum[0] = update_particle_storage_(self.particleNum[0], self.particle)
        finial_particle = self.particleNum[0]
        print(f"Total {-finial_particle + initial_particle} particles has been deleted", '\n')

    def delete_particles_in_region(self, is_in_region):
        initial_particle = self.particleNum[0]
        kernel_delete_particles_in_region(self.particleNum[0], self.particle, is_in_region)
        self.particleNum[0] = update_particle_storage_(self.particleNum[0], self.particle)
        finial_particle = self.particleNum[0]
        print(f"Total {-finial_particle + initial_particle} particles has been deleted", '\n')

    def set_boundary_condition(self, sims: Simulation):
        self.domain_boundary = DomainBoundary(sims.domain)
        self.domain_boundary.set_boundary_condition(sims.boundary)
        if self.domain_boundary.need_run and sims.scheme == "DEM":
            self.apply_boundary_conditions = self.apply_boundary_condition
        else:
            self.apply_boundary_conditions = no_operation

    def apply_boundary_condition(self):
        if self.domain_boundary.apply_boundary_conditions(int(self.particleNum[0]), self.particle):
            update_particle_storage_(self.particleNum, self.sphereNum, self.clumpNum, self.particle, self.sphere, self.clump)
