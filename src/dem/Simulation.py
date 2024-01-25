import taichi as ti

from src.utils.ObjectIO import DictIO
from src.utils.ScalarFunction import next_pow2
from src.utils.TypeDefination import vec3i, vec3f


class Simulation(object):
    def __init__(self) -> None:
        self.domain = vec3f([0, 0, 0])
        self.boundary = vec3i([0, 0, 0])
        self.gravity = vec3f([0, 0, 0])
        self.engine = None
        self.search = None
        self.is_continue = True
        self.coupling = False

        self.dt = ti.field(float, shape=())
        self.delta = 0.
        self.current_time = 0.
        self.current_step = 0
        self.current_print = 0
        self.CurrentTime = ti.field(float, shape=())

        self.max_material_num = 0
        self.max_particle_num = 0
        self.max_clump_num = 0
        self.max_level_grid_num = 0
        self.max_rigid_body_num = 0
        self.max_surface_node_num = 0
        self.max_wall_num = 0
        self.max_servo_wall_num = 0
        self.compaction_ratio = 0.5
        self.pbc = False
        self.wall_type = None
        self.particle_work = None
        self.wall_work = None
        self.servo_status = "Off"

        self.body_coordination_number = 0
        self.wall_coordination_number = 0

        self.verlet_distance = 0.
        self.verlet_distance_multiplier = 0.
        self.max_potential_particle_pairs = 0
        self.max_potential_wall_pairs = 0
        self.wall_per_cell = 0
        self.max_particle_radius = 0.

        self.potential_particle_num = 0.
        self.particle_particle_contact_model = None
        self.particle_wall_contact_model = None

        self.time = 0.
        self.CFL = 0.2
        self.isadaptive = False
        self.save_interval = 1e6
        self.path = None
        self.verlet_distance = 0.

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
            
    def set_boundary(self, boundary):
        BOUNDARY = {
                        "Reflect": 0,
                        "Destroy": 1,
                        "Period": 2
                   }
        self.boundary = vec3i([DictIO.GetEssential(BOUNDARY, b) for b in boundary])
        if self.boundary[0] == 2 or self.boundary[1] == 2 or self.boundary[2] == 2:
            self.activate_period_boundary()

    def set_gravity(self, gravity):
        self.gravity = gravity

    def set_engine(self, engine):
        self.engine = engine

    def set_search(self, search):
        self.search = search

    def activate_period_boundary(self):
        self.pbc = True

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
        self.max_particle_num = int(particle_num)

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

        if level_grid_num > 0 and self.max_particle_num > 0:
            raise RuntimeError("Sphere/Multisphere particles are not supported when using level set method")

    def set_rigid_body_num(self, rigid_body_num):
        if rigid_body_num < 0:
            raise ValueError("Rigid body number should be larger than 0!")
        self.max_rigid_body_num = int(rigid_body_num)

        if rigid_body_num > 0 and self.max_particle_num > 0:
            raise RuntimeError("Sphere/Multisphere particles are not supported when using level set method")
        
    def set_surface_node_num(self, surface_node_num):
        if surface_node_num < 0:
            raise ValueError("Surface node number should be larger than 0!")
        self.max_surface_node_num = int(surface_node_num)

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
            elif not self.wall_type is None:
                self.raise_wall_error_info(curr_wall_type=0)

    def set_dem_coupling(self, coupling):
        self.coupling = coupling

    def update_servo_status(self, status):
        self.servo_status = status

    def set_body_coordination_number(self, body_coordination_number):
        self.body_coordination_number = body_coordination_number

    def set_wall_coordination_number(self, wall_coordination_number):
        self.wall_coordination_number = wall_coordination_number

    def set_verlet_distance_multiplier(self, verlet_distance_multiplier):
        self.verlet_distance_multiplier = verlet_distance_multiplier

    def set_wall_per_cell(self, wall_per_cell):
        if wall_per_cell is None:
            if self.wall_type == 0:
                self.wall_per_cell = 4
            elif self.wall_type == 1:
                self.wall_per_cell = 8
        else:
            self.wall_per_cell = wall_per_cell

    def set_particle_particle_contact_model(self, model):
        self.particle_particle_contact_model = model

    def set_particle_wall_contact_model(self, model):
        if model is None and self.max_wall_num > 0:
            raise ValueError("Particle-Wall contact model have not been assigned")
        if self.max_wall_num == 0: model = None
        self.particle_wall_contact_model = model

    def set_save_data(self, particle, sphere, clump, wall, ppcontact, pwcontact):
        if particle: self.monitor_type.append('particle')
        if sphere: self.monitor_type.append('sphere')
        if clump: self.monitor_type.append('clump')
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
        else:
            raise ValueError("Wall Type error!")

    def raise_wall_error_info(self, curr_wall_type):
        Type1 = self.get_wall_type(curr_wall_type)
        Type2 = self.get_wall_type(self.wall_type)
        raise ValueError(f"Wall Type: {Type1} and Wall Type: {Type2} are activated simultaneously")

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

    def set_compaction_ratio(self, compaction_ratio):
        self.compaction_ratio = compaction_ratio

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
        if self.verlet_distance < 1e-16:
            self.verlet_distance = self.verlet_distance_multiplier * rad_min

    def set_potential_list_size(self, rad_max):
        potential_particle_ratio = ((rad_max + self.verlet_distance) / (rad_max)) ** 3
        self.potential_particle_num = next_pow2(int(potential_particle_ratio * self.body_coordination_number))
        self.max_potential_particle_pairs = int(self.potential_particle_num * self.max_particle_num)
        if self.max_wall_num > 0:
            self.max_potential_wall_pairs = int(self.wall_coordination_number * self.max_particle_num)    