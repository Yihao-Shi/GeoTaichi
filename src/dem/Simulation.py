import taichi as ti
import math, warnings

import src.utils.GlobalVariable as GlobalVariable
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec3i, vec3f
from src.utils.TimeTicker import Timer


class Simulation(object):
    def __init__(self) -> None:
        self.dimension = 3
        self.domain = vec3f([0, 0, 0])
        self.boundary = vec3i([0, 0, 0])
        self.gravity = vec3f([0, 0, 0])
        self.engine = None
        self.search = None
        self.is_continue = False
        self.coupling = False
        self.scheme = None
        self.static_wall = False
        self.sparse_grid = False
        self.energy_tracking = False
        self.iterative_model = None
        self.search_direction = "Up"
        self.history_contact_path = {}
        self.timer = Timer()

        self.dt = ti.field(float, shape=())
        self.delta = 0.
        self.current_time = 0.
        self.current_step = 0
        self.current_print = 0
        self.CurrentTime = ti.field(float, shape=())
        
        self.bvh_rebuild_interval = 1000
        self.max_material_num = 0
        self.max_particle_num = 0
        self.max_sphere_num = 0
        self.max_clump_num = 0
        self.max_level_grid_num = 0
        self.max_rigid_body_num = 0
        self.max_soft_body_num = 0
        self.max_surface_node_num = 0
        self.max_rigid_template_num = 0
        self.max_wall_num = 0
        self.max_servo_wall_num = 0
        self.max_digital_elevation_grid_number = [0, 0]
        self.compaction_ratio = 0.5
        self.point_particle_coordination_number = 2
        self.point_wall_coordination_number = 1
        self.xpbc = False
        self.ypbc = False
        self.zpbc = False
        self.wall_type = None
        self.particle_work = None
        self.wall_work = None
        self.servo_status = "Off"

        self.body_coordination_number = 0
        self.wall_coordination_number = 0

        self.hierarchical_level = 1
        self.hierarchical_size = []

        self.verlet_distance = 0.
        self.verlet_distance_multiplier = [0., 0.]
        self.max_potential_particle_pairs = 0
        self.max_potential_wall_pairs = 0
        self.wall_per_cell = 0
        self.max_bounding_sphere_radius = 0.
        self.min_bounding_sphere_radius = 0.

        self.potential_particle_num = 0
        self.potential_contact_points_particle = 0
        self.potential_contact_points_wall = 0
        self.particle_contact_list_length = 0
        self.wall_contact_list_length = 0
        self.particle_particle_contact_model = None
        self.particle_wall_contact_model = None

        self.time = 0.
        self.CFL = 0.2
        self.isadaptive = False
        self.visualize = True
        self.save_interval = 1e6
        self.path = None
        self.verlet_distance = 0.
        self.point_verlet_distance = 0.

        self.visualize_interval = 0.
        self.window_size = (1024, 1024)
        self.camera_up = (0.0, 1.0, 0.0)
        self.look_at = (0.0, 1.0, 0.0)
        self.look_from = (0.0, 0.0, 0.0)
        self.particle_color = (1, 1, 1)
        self.background_color = (0, 0, 0)
        self.point_light = (0, 0, 0)
        self.view_angle = 45.
        self.move_velocity = 0.

        self.monitor_type = []

    def get_simulation_domain(self):
        return self.domain

    def set_domain(self, domain):
        self.domain = domain
        if isinstance(domain, (list, tuple)):
            self.domain = vec3f(domain)
            
    def set_boundary(self, boundary):
        BOUNDARY = {
                        None: -1,
                        "Reflect": 0,
                        "Destroy": 1,
                        "Period": 2
                   }
        self.boundary = [DictIO.GetEssential(BOUNDARY, b) for b in boundary]
        if self.boundary[0] == 2:
            self.xpbc = True
            GlobalVariable.DEMXPBC = True
            GlobalVariable.DEMXSIZE = self.domain[0]
        if self.boundary[1] == 2:
            self.ypbc = True
            GlobalVariable.DEMYPBC = True
            GlobalVariable.DEMYSIZE = self.domain[0]
        if self.dimension == 3:
            if self.boundary[2] == 2:
                self.zpbc = True
                GlobalVariable.DEMZPBC = True
                GlobalVariable.DEMZSIZE = self.domain[0]

    def set_gravity(self, gravity):
        self.gravity = gravity
        if isinstance(gravity, (list, tuple)):
            self.gravity = vec3f(gravity)

    def set_engine(self, engine):
        self.engine = engine
        valid = ["SymplecticEuler", "VelocityVerlet", "PredictCorrector"]
        if not engine in valid:
            raise RuntimeError(f"Keyword:: /engine/ is wrong, Only the following is valid: {valid}")

    def set_search(self, search):
        self.search = search
        valid = ["Brust", "LinkedCell", "HierarchicalLinkedCell", "BVH"]
        if not search in valid:
            raise RuntimeError(f"Keyword:: /search/ is wrong, Only the following is valid: {valid}")
        
    def set_search_direction(self, search_direction):
        self.search_direction = search_direction
        valid = ["Up", "Down"]
        if not search_direction in valid:
            raise RuntimeError(f"Keyword:: /search_direction/ is wrong, Only the following is valid: {valid}")
        
    def set_track_energy(self, track_energy):
        if track_energy and self.coupling:
            raise RuntimeError("Coupling MPDEM do not support energy tracking yet!")
        self.energy_tracking = track_energy
        GlobalVariable.TRACKENERGY = track_energy

    def update_critical_timestep(self, dt):
        print("The time step is corrected as:", dt, '\n')
        self.dt[None] = dt
        self.delta = dt

    def set_material_num(self, material_num):
        if material_num <= 0:
            raise ValueError("Material number should be larger than 0!")
        self.max_material_num = int(material_num)

    def set_particle_num(self, particle_num):
        if particle_num < 0:
            raise ValueError("Particle number should be larger than 0!")
        self.max_particle_num = int(max(particle_num, 0))

    def set_sphere_num(self, sphere_num):
        if sphere_num < 0:
            raise ValueError("Sphere number should be larger than 0!")
        self.max_sphere_num = int(sphere_num)

    def set_clump_num(self, clump_num):
        if clump_num < 0:
            raise ValueError("Clump number should be larger than 0!")
        self.max_clump_num = int(clump_num)

    def set_level_grid_num(self, level_grid_num):
        if level_grid_num < 0:
            raise ValueError("Level grid number should be larger than 0!")
        self.max_level_grid_num = int(level_grid_num)

        if level_grid_num > 0 and (self.max_sphere_num > 0 or self.max_clump_num > 0):
            raise RuntimeError("Sphere/Multisphere particles are not supported when using level set method")

    def set_rigid_body_num(self, rigid_body_num):
        if rigid_body_num < 0:
            raise ValueError("Rigid body number should be larger than 0!")
        self.max_rigid_body_num = int(rigid_body_num)
        self.max_particle_num += int(rigid_body_num)

        if rigid_body_num > 0 and (self.max_sphere_num > 0 or self.max_clump_num > 0):
            raise RuntimeError("Sphere/Multisphere particles are not supported when using level set method")
        
    def set_soft_body_num(self, soft_body_num):
        if soft_body_num < 0:
            raise ValueError("Soft body number should be larger than 0!")
        self.max_soft_body_num = int(soft_body_num)
        self.max_particle_num += int(soft_body_num)

        if soft_body_num > 0 and (self.max_sphere_num > 0 or self.max_clump_num > 0):
            raise RuntimeError("Sphere/Multisphere particles are not supported when using level set method")
        
    def set_material_point_num(self, material_point_num):
        if material_point_num < 0:
            raise ValueError("Material point number should be larger than 0!")
        
    def set_surface_node_num(self, surface_node_num):
        if surface_node_num < 0:
            raise ValueError("Surface node number should be larger than 0!")
        self.max_surface_node_num = int(surface_node_num)

    def set_rigid_template_num(self, rigid_template_num):
        self.max_rigid_template_num = rigid_template_num

    def set_patch_num(self, patch_num):
        if patch_num > 0:
            if self.wall_type is None:
                self.max_wall_num = int(patch_num)
                self.wall_type = 2
            elif not self.wall_type is None:
                self.raise_wall_error_info(curr_wall_type=2)

    def set_servo_wall_num(self, servo_wall_num):
        if servo_wall_num > 0:
            if self.wall_type == 1:
                self.max_servo_wall_num = int(servo_wall_num)
            else:
                raise RuntimeError("Facet Number has not been set")

    def set_facet_num(self, facet_num):
        if facet_num > 0:
            if self.wall_type is None:
                self.max_wall_num = int(facet_num)
                self.wall_type = 1
            elif not self.wall_type is None:
                self.raise_wall_error_info(curr_wall_type=1)

    def set_plane_num(self, plane_num):
        if plane_num > 0:
            if self.wall_type is None:
                self.max_wall_num = int(plane_num)
                self.wall_type = 0
                self.static_wall = True
            elif not self.wall_type is None:
                self.raise_wall_error_info(curr_wall_type=0)

    def set_digital_elevation_grid_num(self, digital_elevation_grid_number):
        if isinstance(digital_elevation_grid_number, (int, float)):
            self.max_digital_elevation_grid_number = [int(digital_elevation_grid_number), int(digital_elevation_grid_number)]
        elif isinstance(digital_elevation_grid_number, (list, tuple)):
            self.max_digital_elevation_grid_number = [int(i) for i in digital_elevation_grid_number]

        expect_cell_num = 1
        for i in self.max_digital_elevation_grid_number:
            expect_cell_num *= max(i - 1, 0)
        if self.max_wall_num > 2 * expect_cell_num:
            warnings.warn(f"Keyword:: /max_digital_elevation_facet_num/ {self.max_wall_num} is large enough, which may unnecessarily occupy more GPU memory")

    def set_digital_elevation_facet_num(self, digital_elevation_facet_number):
        if digital_elevation_facet_number > 0:
            if self.wall_type is None:
                self.max_wall_num = int(digital_elevation_facet_number)
                self.wall_type = 3
            elif not self.wall_type is None:
                self.raise_wall_error_info(curr_wall_type=3)

    def set_dem_scheme(self, scheme):
        self.scheme = scheme

        valid = ["DEM", "LSDEM", "LSMPM", "PolySuperEllipsoid", "PolySuperQuadrics"]
        if not scheme in valid:
            raise RuntimeError(f"Keyword:: /scheme/ error. Only the following {valid} is support")
        
    def set_visualize(self, visualize):
        self.visualize = visualize

    def set_dem_coupling(self, coupling):
        self.coupling = coupling

    def set_static_wall(self, static_wall):
        self.static_wall = static_wall

    def update_servo_status(self, status):
        self.servo_status = status
        if status == "On":
            self.static_wall = False

    def set_body_coordination_number(self, body_coordination_number):
        if self.search == "HierarchicalLinkedCell":
            if isinstance(body_coordination_number, (int, float)):
                self.body_coordination_number = [int(max(body_coordination_number, 1)) for _ in range(self.hierarchical_level)]
            else:
                self.body_coordination_number = [int(max(i, 1)) for i in body_coordination_number]
            
            if len(self.body_coordination_number) != self.hierarchical_level:
                raise RuntimeError(f"Keyword:: /body_coordination_number/ should have a size of {self.hierarchical_level}")
        else:
            self.body_coordination_number = max(body_coordination_number, 1)

    def set_wall_coordination_number(self, wall_coordination_number):
        if self.search == "HierarchicalLinkedCell":
            if isinstance(wall_coordination_number, (float, int)):
                self.wall_coordination_number = [int(wall_coordination_number) for _ in range(self.hierarchical_level)]
            else:
                self.wall_coordination_number = wall_coordination_number
            
            if len(self.wall_coordination_number) != self.hierarchical_level:
                raise RuntimeError(f"Keyword:: /wall_coordination_number/ should have a size of {self.hierarchical_level}")
        else:
            self.wall_coordination_number = wall_coordination_number

    def set_verlet_distance_multiplier(self, verlet_distance_multiplier):
        if isinstance(verlet_distance_multiplier, float):
            self.verlet_distance_multiplier = [verlet_distance_multiplier, verlet_distance_multiplier]
        else:
            self.verlet_distance_multiplier = verlet_distance_multiplier

    def set_wall_per_cell(self, wall_per_cell):
        if self.search == "HierarchicalLinkedCell":
            if isinstance(wall_per_cell, (int, float)):
                self.wall_per_cell = [int(max(wall_per_cell, 1)) for _ in range(self.hierarchical_level)]
            else:
                self.wall_per_cell = [int(max(i, 1)) for i in wall_per_cell]
            if len(self.wall_per_cell) != self.hierarchical_level:
                raise RuntimeError(f"Keyword:: /wall_per_cell/ should have a size of {self.hierarchical_level}")
        else:
            self.wall_per_cell = wall_per_cell

    def set_iterative_model(self, model):
        self.iterative_model = model
        valid_list = ["LagrangianMultiplier", "PCN", "GJK"]
        if model not in valid_list:
            raise RuntimeError(f"Keyword:: /iterative_model/ error. Only the following {valid_list} is support")

    def set_particle_particle_contact_model(self, model):
        self.particle_particle_contact_model = model

    def set_particle_wall_contact_model(self, model):
        if model is None and self.max_wall_num > 0:
            warnings.warn("Particle-Wall contact model have not been assigned!")
        if self.max_wall_num == 0: model = None
        self.particle_wall_contact_model = model

    def set_save_data(self, particle, sphere, clump, surface, grid, bounding, wall, ppcontact, pwcontact):
        if particle: self.monitor_type.append('particle')
        if sphere: self.monitor_type.append('sphere')
        if clump: self.monitor_type.append('clump')
        if surface: self.monitor_type.append('surface')
        if grid: self.monitor_type.append('grid')
        if bounding: self.monitor_type.append('bounding')
        if wall: self.monitor_type.append('wall')
        if ppcontact: self.monitor_type.append('ppcontact')
        if pwcontact: self.monitor_type.append('pwcontact')

    def get_wall_type(self, wall_type):
        if wall_type == 0:
            return "Infinitesimal Plane"
        elif wall_type == 1:
            return "Polygon Wall"
        elif wall_type == 2:
            return "Triangle Patch"
        elif wall_type == 3:
            return "digital Elevation Facet"
        else:
            raise ValueError("Wall Type error!")

    def raise_wall_error_info(self, curr_wall_type):
        Type1 = self.get_wall_type(curr_wall_type)
        Type2 = self.get_wall_type(self.wall_type)
        raise ValueError(f"Wall Type: {Type1} and Wall Type: {Type2} are activated simultaneously")
    
    def set_hierarchical_level(self, hierarchical_level):
        if hierarchical_level > 8:
            raise RuntimeError("The maximum level of grid is 8")
        self.hierarchical_level = int(hierarchical_level)

    def set_rebuild_interval(self, rebuild_interval):
        if self.search == "BVH":
            self.bvh_rebuild_interval = rebuild_interval
    
    def set_hierarchical_size(self, hierarchical_size):
        self.hierarchical_size = list(hierarchical_size)
        self.hierarchical_size.sort()
        if len(self.hierarchical_size) != self.hierarchical_level:
            warnings.warn(f"KeyWord:: /hierarchical_level/ should be set as {len(self.hierarchical_size)}")
            self.set_hierarchical_level(len(self.hierarchical_size))

    def set_timestep(self, timestep):
        self.dt[None] = timestep
        self.delta = timestep

    def set_simulation_time(self, time):
        self.time = time

    def set_CFL(self, CFL):
        self.CFL = CFL

    def set_adaptive_timestep(self, isadaptive):
        self.isadaptive = isadaptive

    def set_save_interval(self, save_interval):
        self.save_interval = save_interval

    def set_is_continue(self, is_continue):
        self.is_continue = is_continue

    def set_point_coordination_number(self, point_coordination_number):
        if isinstance(point_coordination_number, (list, tuple)):
            self.point_particle_coordination_number = int(point_coordination_number[0])
            self.point_wall_coordination_number = int(point_coordination_number[1])
        elif isinstance(point_coordination_number, (int, float)):
            self.point_particle_coordination_number = int(point_coordination_number)
            self.point_wall_coordination_number = int(point_coordination_number)

    def set_compaction_ratio(self, compaction_ratio):
        if isinstance(compaction_ratio, float):
            if self.scheme == "LSDEM" or self.scheme == "LSMPM":
                self.compaction_ratio = [compaction_ratio, compaction_ratio, compaction_ratio, compaction_ratio]
            else:
                self.compaction_ratio = [compaction_ratio, compaction_ratio]
        elif isinstance(compaction_ratio, (tuple, list)):
            if self.scheme == "LSDEM" or self.scheme == "LSMPM":
                if len(list(compaction_ratio)) == 2:
                    self.compaction_ratio = [compaction_ratio[0], compaction_ratio[1], compaction_ratio[0], compaction_ratio[1]]
                elif len(list(compaction_ratio)) == 4:
                    self.compaction_ratio = list(compaction_ratio)
                else:
                    raise RuntimeError("Keyword:: /compaction_ratio/ dimension error")
            else:
                self.compaction_ratio = list(compaction_ratio)

    def set_window_parameters(self, windows):
        self.visualize_interval = DictIO.GetAlternative(windows, "VisualizeInterval", self.save_interval)
        self.window_size = DictIO.GetAlternative(windows, "WindowSize", self.window_size)
        self.camera_up = DictIO.GetAlternative(windows, "CameraUp", self.camera_up)
        self.look_at = DictIO.GetAlternative(windows, "LookAt", (0.7*self.domain[0], -1.5*self.domain[1], 0.4*self.domain[2]))
        self.look_from = DictIO.GetAlternative(windows, "LookFrom", (0.7*self.domain[0], -1.5*self.domain[1], 0.5*self.domain[2]))
        self.particle_color = DictIO.GetAlternative(windows, "ParticleColor", (1, 1, 1))
        self.background_color = DictIO.GetAlternative(windows, "BackgroundColor", (0, 0, 0))
        self.point_light = DictIO.GetAlternative(windows, "PointLight", (0.5*self.domain[0], 0.5*self.domain[1], 1.0*self.domain[2]))
        self.view_angle = DictIO.GetAlternative(windows, "ViewAngle", 70.)
        self.move_velocity = DictIO.GetAlternative(windows, "MoveVelocity", 0.01 * (self.domain[0] + self.domain[1] + self.domain[2]))

    def set_save_path(self, path):
        self.path = path

    def define_work_load(self):
        if self.max_particle_num <= 1000:
            self.particle_work = 0
        elif 1000 < self.max_particle_num < 46338:
            self.particle_work = 1
        else:
            self.particle_work = 2
        self.particle_work = 2

        if self.max_particle_num * self.max_wall_num<= 1000000:
            self.wall_work = 0
        elif 1000000 < self.max_particle_num * self.max_wall_num < 46338 * 46338:
            self.wall_work = 1
        else:
            self.wall_work = 2
        self.wall_work = 2

    def set_verlet_distance(self, rad_min):
        if self.verlet_distance < 1e-15:
            self.verlet_distance = self.verlet_distance_multiplier[0] * rad_min

    def set_point_verlet_distance(self, rad_min):
        if self.point_verlet_distance < 1e-15:
            self.point_verlet_distance = self.verlet_distance_multiplier[1] * rad_min

    def check_grid_extent(self, pid, extent):
        if pid != -1:
            raise RuntimeError(f"Keyword:: /extent/ is not large enough, Particle {pid} need extent = {extent}")
        
    def compute_potential_ratios(self, rad_max):
        return ((3 * rad_max + 2 * self.verlet_distance) ** 3 - rad_max ** 3) / (26. * rad_max ** 3)
    
    def update_hierarchical_size(self, rad_max):
        self.hierarchical_size[-1] = max(rad_max, self.hierarchical_size[-1])

    def set_potential_list_size(self, rad_max):
        potential_particle_ratio = self.compute_potential_ratios(rad_max)
        self.potential_particle_num = math.ceil(potential_particle_ratio * self.body_coordination_number) # next_pow2(int(potential_particle_ratio * self.body_coordination_number))
        self.max_potential_particle_pairs = math.ceil(self.potential_particle_num * self.max_particle_num)
        if self.max_wall_num > 0:
            self.max_potential_wall_pairs = math.ceil(self.wall_coordination_number * self.max_particle_num) 
        self.set_levelset_contact_list_size()
        self.set_contact_list_size()

    def set_levelset_contact_list_size(self):
        self.potential_contact_points_particle = int(self.point_particle_coordination_number * self.max_surface_node_num)
        self.potential_contact_points_wall = int(self.point_wall_coordination_number * self.max_surface_node_num)

    def set_contact_list_size(self): 
        if self.scheme == "LSDEM" or self.scheme == "LSMPM":
            self.particle_verlet_length = int(math.ceil(self.compaction_ratio[0] * self.max_potential_particle_pairs))
            self.wall_verlet_length = int(math.ceil(self.compaction_ratio[1] * self.max_potential_wall_pairs))
            self.particle_contact_list_length = int(math.ceil(self.compaction_ratio[2] * self.potential_contact_points_particle * self.max_rigid_body_num))
            self.wall_contact_list_length = int(math.ceil(self.compaction_ratio[3] * self.potential_contact_points_wall * self.max_rigid_body_num))   
        else:
            self.particle_contact_list_length = int(math.ceil(self.compaction_ratio[0] * self.max_potential_particle_pairs))
            self.wall_contact_list_length = int(math.ceil(self.compaction_ratio[1] * self.max_potential_wall_pairs))

    def set_hierarchical_list_size(self, potential_particle_num, max_potential_wall_pairs):
        self.max_potential_particle_pairs = potential_particle_num
        self.max_potential_wall_pairs = max_potential_wall_pairs
        self.set_levelset_contact_list_size()
        self.set_contact_list_size()

    def set_max_bounding_sphere_radius(self, max_rad):
        self.max_bounding_sphere_radius = max_rad

    def set_min_bounding_sphere_radius(self, min_rad):
        self.min_bounding_sphere_radius = min_rad

    def check_multiplier(self, penetration_depth):
        if self.scheme == "LSDEM":
            if penetration_depth > self.point_verlet_distance:
                raise RuntimeError(f"Keyword:: /verlet_distance_multiplier[1]/ should larger, at least {penetration_depth / self.point_verlet_distance * self.verlet_distance_multiplier[1]}")