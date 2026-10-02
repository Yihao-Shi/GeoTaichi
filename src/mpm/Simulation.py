import taichi as ti
import warnings

from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec2f, vec3i, vec3f
import src.utils.GlobalVariable as GlobalVariable
from src.utils.TimeTicker import Timer


class Simulation(object):
    def __init__(self) -> None:
        self.dimension = 3
        self.is_2DAxisy = False
        self.mode = "Normal"
        self.domain = [0., 0., 0.]
        self.boundary = [0, 0, 0]
        self.gravity = [0., 0., 0.]
        self.block_size = [128, 4]
        self.background = 0.
        self.alphaPIC = 0.
        self.shape_smooth = 0.
        self.fbar_fraction = 0.99
        self.max_radius = 0.
        self.coupling = False
        self.neighbor_detection = False
        self.free_surface_detection = False
        self.sparse_grid = False
        self.boundary_direction_detection = False
        self.stress_integration = None
        self.stabilize = None
        self.pressure_smoothing = False
        self.strain_smoothing = False
        self.random_field = False
        self.mapping = None
        self.shape_function = None
        self.wall_type = None
        self.monitor_type = []
        self.gauss_number = 0
        self.mls = False
        self.order = 2.
        self.integration_scheme = None
        self.visualize = True
        self.particle_shifting = False
        self.isTHB = False
        self.AOSOA = False
        self.norm_adaptivity = False
        self.THBparameter = {}
        self.grid_layer = 0
        self.timer = Timer()

        self.dt = ti.field(float, shape=())
        self.delta = 0.
        self.current_time = 0.
        self.current_step = 0
        self.current_print = 0
        self.CurrentTime = ti.field(float, shape=())
        
        self.max_body_num = 2
        self.max_material_num = 0
        self.max_particle_num = 1
        self.max_coupling_particle_num = 1
        self.verlet_distance_multiplier = 0
        self.verlet_distance = 0.
        self.nvelocity = 0
        self.nfriction = 0
        self.nreflection = 0
        self.nabsorbing = 0
        self.ntraction = 0
        self.nptraction = 0
        self.ndisplacement = 0
        self.xpbc = False
        self.ypbc = False
        self.zpbc = False
        self.is_continue = True

        self.time = 0.
        self.CFL = 0.2
        self.isadaptive = False
        self.save_interval = 1e6
        self.path = None
        self.contact_detection = None

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
        self.quasi_static = False
        self.newmark_gamma = 0.5
        self.newmark_beta = 0.25
        self.iter_max = 50
        self.dof_multiplier = 2
        self.multilevel = 1
        self.pre_and_post_smoothing = 0
        self.bottom_smoothing = 0
        self.linear_solver = "PCG"
        self.assemble_type = "MatrixFree"
        self.integration_scheme = "Newmark"
        self.configuration = "ULMPM"
        self.material_type = "Solid"
        self.solver_type = "Explicit"
        self.discretization = "FEM"

        self.TESTMODE = False
        
    def get_simulation_domain(self):
        return self.domain
    
    def set_dimension(self):
        self.dimension = GlobalVariable.DIMENSION

    def set_is_2DAxisy(self, is_2DAxisy):
        self.is_2DAxisy = is_2DAxisy

    def set_mode(self, mode):
        valid_list = ['Normal', 'Lightweight']
        if mode not in valid_list:
            raise RuntimeError(f"Keyword:: /mode/ must be {valid_list}")
        self.mode = mode

    def set_domain(self, domain):
        self.domain = domain
        if isinstance(domain, (list, tuple)):
            if self.dimension == 3:
                self.domain = vec3f(domain)
            elif self.dimension == 2:
                self.domain = vec2f(domain)
        
    def set_boundary(self, boundary):
        BOUNDARY = {
                        'None': -1,
                        "Reflect": 0,
                        "Destroy": 1,
                        "Period": 2
                   }
        self.boundary = [DictIO.GetEssential(BOUNDARY, b) for b in boundary]
        if self.boundary[0] == 2:
            self.xpbc = True
            GlobalVariable.MPMXPBC = True
            GlobalVariable.MPMXSIZE = self.domain[0]
        if self.boundary[1] == 2:
            self.ypbc = True
            GlobalVariable.MPMYPBC = True
            GlobalVariable.MPMYSIZE = self.domain[1]
        if self.dimension == 3:
            if self.boundary[2] == 2:
                self.zpbc = True
                GlobalVariable.MPMZPBC = True
                GlobalVariable.MPMZSIZE = self.domain[2]

    def set_gravity(self, gravity):
        self.gravity = gravity
        if len(gravity) == 2:
            gravity = [gravity[0], gravity[1], 0.]
        if isinstance(gravity, (list, tuple)):
            self.gravity = vec3f(gravity)

    def set_background_damping(self, background_damping):
        self.background_damping = background_damping

    def set_alpha(self, alphaPIC):
        self.alphaPIC = alphaPIC

    def set_stabilize_technique(self, stabilize):
        typelist = [None, "B-Bar Method", "F-Bar Method"]
        if not stabilize in typelist:
            raise RuntimeError(f"KeyWord:: /stabilize: {stabilize}/ is invalid. The valid type are given as follows: {typelist}")
        self.stabilize = stabilize
        
        if self.stabilize == "B-Bar Method":
            GlobalVariable.BBAR = True
        elif self.stabilize == "F-Bar Method":
            GlobalVariable.FBAR = True

    def set_shape_smoothing(self, shape_smooth):
        if self.shape_function == "SmoothLinear":
            self.shape_smooth = shape_smooth

    def set_pressure_smoothing(self, pressure_smoothing):
        self.pressure_smoothing = pressure_smoothing

    def set_strain_smoothing(self, strain_smoothing):
        self.strain_smoothing = strain_smoothing

    def set_configuration(self, configuration):
        config = ["TLMPM", "ULMPM"]
        if not configuration in config:
            raise RuntimeError(f"Keyword:: /configuration/ error. Only {config} is valid!")
        self.configuration = configuration

    def set_material_type(self, material_type):
        valid_list = ["Solid", "Fluid", "TwoPhaseSingleLayer", "TwoPhaseDoubleLayer"]
        if not material_type in valid_list:
            raise RuntimeError(f"Keyword:: /material_type/ error. Only {valid_list} is valid!")
        self.material_type = material_type

        if self.material_type == "TwoPhaseSingleLayer":
            GlobalVariable.TWOPHASESINGLELAYER = True

    def set_visualize(self, visualize):
        self.visualize = visualize

    def set_THB(self, THBparameter):
        if THBparameter:
            self.isTHB = True
            self.THBparameter = THBparameter
            self.grid_layer = THBparameter['grid_layer']

    def set_sparse_grid(self, sparse_grid):
        if sparse_grid:
            self.sparse_grid = True

    def set_gauss_integration(self, gauss_number):
        self.gauss_number = gauss_number
    
    def set_particle_shifting(self, particle_shifting):
        self.particle_shifting = particle_shifting
        GlobalVariable.PARTICLESHIFTING = particle_shifting

    def set_stress_integration(self, stress_integration):
        typelist = ["ReturnMapping", "SubStepping"]
        if not stress_integration in typelist:
            raise RuntimeError(f"KeyWord:: /stress_integration: {stress_integration}/ is invalid. The valid type are given as follows: {typelist}")
        self.stress_integration = stress_integration

    def set_discretization(self, discretization):
        typelist = ["FEM", "FDM"]
        if not discretization in typelist:
            raise RuntimeError(f"KeyWord:: /discretization: {discretization}/ is invalid. The valid type are given as follows: {typelist}")
        self.discretization = discretization

    def set_moving_least_square(self, mls):
        self.mls = mls
        if self.mapping == "G2P2G":
            self.mls = True
        if mls is True:
            self.set_velocity_projection_scheme("Affine")
            self.alphaPIC = 1.

    def set_mapping_scheme(self, mapping):
        typelist = ["USL", "USF", "MUSL", "G2P2G"]
        if not mapping in typelist:
            raise RuntimeError(f"KeyWord:: /mapping: {mapping}/ is invalid. The valid type are given as follows: {typelist}")
        self.mapping = mapping

    def set_shape_function(self, shape_function):
        typelist = ["Linear", "SmoothLinear", "GIMP", "QuadBSpline", "CubicBSpline"]
        if not shape_function in typelist:
            raise RuntimeError(f"KeyWord:: /mapping: {shape_function}/ is invalid. The valid type are given as follows: {typelist}")
        self.shape_function = shape_function
        if self.mapping == "G2P2G":
            self.shape_function == "QuadBSpline"
        if self.mode == "Lightweight":
            if self.shape_function == "Linear":
                GlobalVariable.SHAPEFUNCTION = 0
            elif self.shape_function == "GIMP":
                GlobalVariable.SHAPEFUNCTION = 1
            elif self.shape_function == "QuadBSpline":
                GlobalVariable.SHAPEFUNCTION = 2
            elif self.shape_function == "CubicBSpline":
                GlobalVariable.SHAPEFUNCTION = 3

    def set_mpm_coupling(self, coupling):
        typelist = ["Lagrangian", "Eulerian", False]
        if not coupling in typelist:
            raise RuntimeError(f"KeyWord:: /coupling: {coupling}/ is invalid. The valid type are given as follows: {typelist}")
        self.coupling = coupling

    def set_free_surface_detection(self, free_surface_detection):
        if self.norm_adaptivity: free_surface_detection = True
        self.free_surface_detection = free_surface_detection
        if free_surface_detection is True:
            self.neighbor_detection = True
            self.boundary_direction_detection = True

    def set_boundary_direction(self, boundary_direction_detection):
        self.boundary_direction_detection = boundary_direction_detection
        if boundary_direction_detection is True:
            self.neighbor_detection = True

    def set_solver_type(self, solver_type):
        typelist = ["Explicit", "Implicit", "SemiImplicit"]
        if not solver_type in typelist:
            raise RuntimeError(f"KeyWord:: /solver_type: {solver_type}/ is invalid. The valid type are given as follows: {typelist}")
        self.solver_type = solver_type

        if self.solver_type == "SemiImplicit" and self.dimension == 2:
            raise RuntimeError("SemiImplicit type supports 2-Dimensional condition!")

    def set_is_continue(self, is_continue):
        self.is_continue = is_continue

    def set_norm_adaptivity(self, is_adaptivity):
        self.norm_adaptivity = is_adaptivity
        if is_adaptivity:
            self.set_free_surface_detection(True)

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

    def set_max_radius(self, max_radius):
        self.max_radius = max(self.max_radius, max_radius)

    def set_coupling_particles(self, coupling_particles):
        if coupling_particles == 0:
            raise RuntimeError("It is unnecessary to using coupling modules because there is no materail points are considered in coupling process")
        self.max_coupling_particle_num = coupling_particles

    def set_velocity_projection_scheme(self, velocity_projection_scheme: str):
        self.velocity_projection_scheme = velocity_projection_scheme
        valid_type = ["PIC", "FLIP", "PIC/FLIP", "Affine", "Taylor"]
        if velocity_projection_scheme == "PIC": 
            self.alphaPIC = 1.
        elif velocity_projection_scheme == "FLIP": 
            self.alphaPIC = 0.

        if velocity_projection_scheme not in valid_type:
            raise RuntimeError(f"Keyword:: /velocity_projection_scheme/ is error, followings are valid {valid_type}")
        
        if self.velocity_projection_scheme == "Affine":
            GlobalVariable.APIC = True
        elif self.velocity_projection_scheme == "Taylor":
            GlobalVariable.TPIC = True

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
        if not self.ptraction_method == "Virtual":
            self.nptraction = int(DictIO.GetAlternative(constraint, "max_particle_traction_constraint", 0))
        if self.solver_type == "Implicit":
            self.ndisplacement = int(DictIO.GetAlternative(constraint, "max_displacement_constraint", 0))

    def set_particle_traction_method(self, particle_traction_method):
        self.ptraction_method = particle_traction_method
        valid_list = [None, "Stable", "Nanson", "Virtual"]
        if self.ptraction_method not in valid_list:
            warnings.warn("We choose stable version of particle traction by default!")
            self.ptraction_method = "Stable"

        if self.ptraction_method == "Virtual":
            self.nptraction = 1

    def set_AOSOA(self, AOSOA):
        if AOSOA:
            self.AOSOA = True
            if isinstance(AOSOA, (list, tuple)):
                if len(AOSOA) >= 1:
                    self.block_size = list(AOSOA)
                else:
                    raise ValueError(f"Keyword:: /AOSOA/ is empty. The input {AOSOA} is invalid. For example, you can input [grid_block_size, leaf_block_size].")
                
    def set_random_field(self, random_field):
        if random_field:
            self.random_field = True
            GlobalVariable.RANDOMFIELD = True
        
    def set_save_data(self, particle, grid, object):
        if particle: self.monitor_type.append('particle')
        if grid: self.monitor_type.append('grid')
        if object: self.monitor_type.append('object')

    def update_critical_timestep(self, dt):
        print("The time step is corrected as:", dt, '\n')
        self.dt[None] = dt
        self.delta = dt

    def set_contact_detection(self, contact_detection):
        self.contact_detection = contact_detection
        if contact_detection and not contact_detection in ["MPMContact", "GeoContact", "DEMContact"]:
            valid = ["MPMContact", "GeoContact", "DEMContact"]
            raise RuntimeError(f"Keyword:: /contact_detection/ is wrong. Only the following is valid: {valid}")
        if self.dimension == 3 and contact_detection == "DEMContact":
            raise RuntimeError("Three-dimension model do not support DEMContact!")
        
    def set_calculate_reaction_force(self, calculate_reaction_force):
        self.calculate_reaction_force = calculate_reaction_force

    def set_integration_scheme(self, integration_scheme):
        self.integration_scheme = integration_scheme

    def set_displacement_tolerance(self, displacement_tolerance):
        self.displacement_tolerance = displacement_tolerance

    def set_residual_tolerance(self, residual_tolerance):
        self.residual_tolerance = residual_tolerance

    def set_quasi_static(self, quasi_static):
        self.quasi_static = quasi_static

    def set_max_iteration(self, iter_max):
        self.iter_max = iter_max

    def set_newmark_parameter(self, newmark_parameter):
        newmark_parameter = list(newmark_parameter)
        if len(newmark_parameter) != 2: 
            raise RuntimeError("The size of newmark parameter should follow [gamma, beta]")
        self.newmark_gamma = newmark_parameter[0]
        self.newmark_beta = newmark_parameter[1]

    def set_dof_multiplier(self, dof_multiplier):
        self.dof_multiplier = dof_multiplier

    def set_linear_solver(self, linear_solver):
        self.linear_solver = linear_solver

        valid_list = ["CG", "PCG", "BiCG", "MGPCG"]
        if linear_solver not in valid_list:
            raise RuntimeError(f"Keyword:: /linear_solver/ is error, followings are valid {valid_list}")
        
    def set_assemble_type(self, assemble_type):
        self.assemble_type = assemble_type
        valid_list = ["MatrixFree", "CSR"]
        if assemble_type not in valid_list:
            raise RuntimeError(f"Keyword:: /assemble_type/ is error, followings are valid {valid_list}")
        
    def set_multigrid_paramter(self, multilevel, pre_and_post_smoothing, bottom_smoothing):
        self.multilevel = int(multilevel)
        self.pre_and_post_smoothing = int(pre_and_post_smoothing)
        self.bottom_smoothing = int(bottom_smoothing)

