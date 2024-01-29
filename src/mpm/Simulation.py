import taichi as ti

from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec3i, vec3f


class Simulation(object):
    def __init__(self) -> None:
        self.dimension = 3
        self.domain = vec3f([0, 0, 0])
        self.boundary = vec3i([0, 0, 0])
        self.gravity = vec3f([0, 0, 0])
        self.background = 0.
        self.alphaPIC = 0.
        self.coupling = False
        self.neighbor_detection = False
        self.free_surface_detection = False
        self.boundary_direction_detection = False
        self.stabilize = None
        self.stress_smoothing = False
        self.strain_smoothing = False
        self.mapping = None
        self.shape_function = None
        self.wall_type = None
        self.monitor_type = []
        self.gauss_number = 0
        self.mls_order = 0
        self.order = 2.
        self.update = None

        self.dt = ti.field(float, shape=())
        self.delta = 0.
        self.current_time = 0.
        self.current_step = 0
        self.current_print = 0
        self.CurrentTime = ti.field(float, shape=())
        
        self.max_body_num = 2
        self.max_material_num = 0
        self.max_particle_num = 0
        self.verlet_distance_multiplier = 0
        self.verlet_distance = 0.
        self.nvelocity = 0
        self.nfriction = 0
        self.nreflection = 0
        self.nabsorbing = 0
        self.ntraction = 0
        self.ndisplacement = 0
        self.pbc = False
        self.is_continue = True

        self.time = 0.
        self.CFL = 0.2
        self.isadaptive = False
        self.save_interval = 1e6
        self.path = None
        self.contact_detection = False

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

        self.calculate_reaction_force = False
        self.displacement_tolerance = 1e-4
        self.residual_tolerance = 1e-7
        self.relative_residual_tolerance = 1e-6
        self.quasi_static = False
        self.max_iteration = 10000
        self.newmark_gamma = 0.5
        self.newmark_beta = 0.25
        self.iter_max = 100
        self.dof_multiplier = 2
        self.update = "Newmark"
        self.configuration = "ULMPM"
        self.material_type = "Solid"
        self.solver_type = "Explicit"
        
    def get_simulation_domain(self):
        return self.domain
    
    def set_dimension(self, dimension):
        DIMENSION = ["2-Dimension", "3-Dimension"]
        if not dimension in DIMENSION:
            raise ValueError(f"Keyword:: /dimension/ should choose as follows: {DIMENSION}")
        self.dimension = 3 if dimension == "3-Dimension" else 2

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

    def set_background_damping(self, background_damping):
        self.background_damping = background_damping

    def set_alpha(self, alphaPIC):
        self.alphaPIC = alphaPIC

    def set_stabilize_technique(self, stabilize):
        typelist = [None, "B-Bar Method", "F-Bar Method"]
        if not stabilize in typelist:
            raise RuntimeError(f"KeyWord:: /stabilize: {stabilize}/ is invalid. The valid type are given as follows: {typelist}")
        self.stabilize = stabilize

    def set_stress_smoothing(self, stress_smoothing):
        self.stress_smoothing = stress_smoothing

    def set_strain_smoothing(self, strain_smoothing):
        self.strain_smoothing = strain_smoothing

    def set_configuration(self, configuration):
        config = ["TLMPM", "ULMPM"]
        if not configuration in config:
            raise RuntimeError(f"Keyword:: /configuration/ error. Only {config} is valid!")
        self.configuration = configuration

    def set_material_type(self, material_type):
        mt = ["Solid", "Fluid", "TwoPhase"]
        if not material_type in mt:
            raise RuntimeError(f"Keyword:: /material_type/ error. Only {mt} is valid!")
        self.material_type = material_type

    def set_gauss_integration(self, gauss_number):
        self.gauss_number = gauss_number
    
    def set_moving_least_square_order(self, mls_order):
        self.mls_order = mls_order

    def set_mapping_scheme(self, mapping):
        typelist = ["USL", "USF", "MUSL", "APIC", "Newmark"]
        if not mapping in typelist:
            raise RuntimeError(f"KeyWord:: /mapping: {mapping}/ is invalid. The valid type are given as follows: {typelist}")
        self.mapping = mapping

    def set_shape_function(self, shape_function):
        typelist = ["Linear", "GIMP", "QuadBSpline", "CubicBSpline"]
        if not shape_function in typelist:
            raise RuntimeError(f"KeyWord:: /mapping: {shape_function}/ is invalid. The valid type are given as follows: {typelist}")
        self.shape_function = shape_function

    def set_mpm_coupling(self, coupling):
        self.coupling = coupling

    def set_free_surface_detection(self, free_surface_detection):
        self.free_surface_detection = free_surface_detection
        if free_surface_detection is True:
            self.neighbor_detection = True
            self.boundary_direction_detection = True

    def set_boundary_direction(self, boundary_direction_detection):
        self.boundary_direction_detection = boundary_direction_detection
        if boundary_direction_detection is True:
            self.neighbor_detection = True

    def set_solver_type(self, solver_type):
        typelist = ["Explicit", "Implicit", "SimiImplicit"]
        if not solver_type in typelist:
            raise RuntimeError(f"KeyWord:: /solver_type: {solver_type}/ is invalid. The valid type are given as follows: {typelist}")
        self.solver_type = solver_type

        if self.solver_type != "Explicit" and self.dimension == 2:
            raise RuntimeError("Only Explicit type supports 2-Dimensional condition!")

    def set_is_continue(self, is_continue):
        self.is_continue = is_continue

    def activate_period_boundary(self):
        self.pbc = True

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

    def set_visualize_interval(self, visualize_interval):
        self.visualize_interval = visualize_interval

    def set_window_size(self, window_size):
        self.window_size = window_size

    def set_save_path(self, path):
        self.path = path

    def set_material_num(self, material_num):
        if material_num <= 0:
            raise ValueError("Max material number should be larger than 0!")
        self.max_material_num = int(material_num + 1)

    def set_body_num(self, body_num):
        if body_num <= 0:
            raise ValueError("Max Baterial number should be larger than 0!")
        self.max_body_num = int(body_num)

    def set_particle_num(self, particle_num):
        if particle_num <= 0:
            raise ValueError("Max particle number should be larger than 0!")
        self.max_particle_num = int(particle_num)

    def set_verlet_distance_multiplier(self, verlet_distance_multiplier):
        self.verlet_distance_multiplier = verlet_distance_multiplier

    def set_verlet_distance(self, rad_min):
        if self.verlet_distance < 1e-16:
            self.verlet_distance = self.verlet_distance_multiplier * rad_min 

    def set_window_parameters(self, windows):
        self.visualize_interval = DictIO.GetAlternative(windows, "VisualizeInterval", self.save_interval)
        self.window_size = DictIO.GetAlternative(windows, "WindowSize", self.window_size)
        self.camera_up = DictIO.GetAlternative(windows, "CameraUp", self.camera_up)
        self.look_at = DictIO.GetAlternative(windows, "LookAt", self.look_at)
        self.look_from = DictIO.GetAlternative(windows, "LookFrom", (0.7*self.domain[0], -0.4*self.domain[1], 1.5*self.domain[2]))
        self.particle_color = DictIO.GetAlternative(windows, "ParticleColor", (1, 1, 1))
        self.background_color = DictIO.GetAlternative(windows, "BackgroundColor", (0, 0, 0))
        self.point_light = DictIO.GetAlternative(windows, "PointLight", (0.5*self.domain[0], 0.5*self.domain[1], 1.0*self.domain[2]))
        self.view_angle = DictIO.GetAlternative(windows, "ViewAngle", 45.)
        self.move_velocity = DictIO.GetAlternative(windows, "MoveVelocity", 0.01 * (self.domain[0] + self.domain[1] + self.domain[2]))

    def set_constraint_num(self, constraint):
        self.nvelocity = int(DictIO.GetAlternative(constraint, "max_velocity_constraint", 0))
        self.nreflection = int(DictIO.GetAlternative(constraint, "max_reflection_constraint", 0))
        self.nfriction = int(DictIO.GetAlternative(constraint, "max_friction_constraint", 0))
        self.nabsorbing = int(DictIO.GetAlternative(constraint, "max_absorbing_constraint", 0))
        self.ntraction = int(DictIO.GetAlternative(constraint, "max_traction_constraint", 0))
        if self.solver_type == "Implicit":
            self.ndisplacement = int(DictIO.GetAlternative(constraint, "max_displacement_constraint", 0))
        
    def set_save_data(self, particle, grid):
        if particle: self.monitor_type.append('particle')
        if grid: self.monitor_type.append('grid')

    def update_critical_timestep(self, dt):
        print("The time step is corrected as:", dt, '\n')
        self.dt[None] = dt
        self.delta = dt

    def set_contact_detection(self, contact_detection):
        self.contact_detection = contact_detection

    def set_implicit_parameters(self, implicit_parameters):
        if self.solver_type != "Implicit":
            raise RuntimeError("KeyError:: /solver_type/ should be set as Implicit")
        
        self.calculate_reaction_force = DictIO.GetAlternative(implicit_parameters, "calculate_reaction_force", False)
        self.update = DictIO.GetAlternative(implicit_parameters, "update_scheme", "Newmark")
        self.displacement_tolerance = DictIO.GetAlternative(implicit_parameters, "displacement_tolerance", self.displacement_tolerance)
        self.residual_tolerance = DictIO.GetAlternative(implicit_parameters, "residual_tolerance", self.residual_tolerance)
        self.relative_residual_tolerance = DictIO.GetAlternative(implicit_parameters, "relative_residual_tolerance", self.relative_residual_tolerance)
        self.quasi_static = DictIO.GetAlternative(implicit_parameters, "quasi_static", self.quasi_static)
        self.max_iteration = DictIO.GetAlternative(implicit_parameters, "max_iteration", self.max_iteration)
        self.newmark_gamma = DictIO.GetAlternative(implicit_parameters, "newmark_gamma", self.newmark_gamma)
        self.newmark_beta = DictIO.GetAlternative(implicit_parameters, "newmark_beta", self.newmark_beta)
        self.iter_max = DictIO.GetAlternative(implicit_parameters, "max_iteration_number", self.iter_max)
        self.dof_multiplier = DictIO.GetAlternative(implicit_parameters, "multiplier", self.dof_multiplier)
        
