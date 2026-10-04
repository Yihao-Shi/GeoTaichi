import math

import taichi as ti
import warnings

from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec2f, vec3i, vec3f
import src.utils.GlobalVariable as GlobalVariable
from src.utils.TimeTicker import Timer


def _normalize_solver_keyword(value):
    return str(value).strip().replace("_", "").replace("-", "").replace(" ", "").lower()


class Simulation(object):
    def __init__(self) -> None:
        self.dimension = 3
        self.is_2DAxisy = False
        self.axis_offset = 0.0
        self.mode = "Normal"
        self.mpm_backend = "Native"
        self.soft_particle = False
        self.soft_particle_backend = "LevelSetTLMPM"
        self.initialize_soft_particle_options()
        self.ipc_contact = False
        self.ipc_contact_parameters = {}
        self.direct_static_twophase = False
        self.domain = [0.0, 0.0, 0.0]
        self.boundary = [0, 0, 0]
        self.gravity = [0.0, 0.0, 0.0]
        self.block_size = [128, 4]
        self.background = 0.0
        self.alphaPIC = 0.0
        self.shape_smooth = 0.0
        self.fbar_fraction = 0.99
        self.max_radius = 0.0
        self.coupling = False
        self.neighbor_detection = False
        self.free_surface_detection = False
        self.fluid_level_set = False
        self.fluid_domain_volume_fraction = 0.1
        self.energy_tracking = False
        self.sparse_grid = False
        self.sparse_grid_backend = "BlockScan"
        self.sparse_grid_block_size = 4
        self.sparse_grid_capacity_factor = 1.25
        self.sparse_grid_max_blocks = 0
        self.sparse_grid_visualize_active_blocks = False
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
        self.order = 2.0
        self.integration_scheme = None
        self.visualize = True
        self.particle_shifting = False
        self.particle_shifting_scale = 1.0
        self.density_projection = False
        self.density_projection_tolerance = 0.01
        self.density_projection_error_clamp = 0.01
        self.density_projection_max_shift_ratio = 0.05
        self.density_projection_interior_only = True
        self.delayed_fluid_advection = True
        self.solid_sdf_cut_cell = False
        self.fluid_wall_no_slip = False
        self.solid_cut_cell_min_fraction = 0.01
        self.isTHB = False
        self.AOSOA = False
        self.particle_traction_update_area = True
        self.norm_adaptivity = False
        self.THBparameter = {}
        self.grid_layer = 0
        self.timer = Timer()

        self.dt = ti.field(float, shape=())
        self.delta = 0.0
        self.init_delta = 0.0
        self.current_time = 0.0
        self.current_step = 0
        self.current_print = 0
        self.CurrentTime = ti.field(float, shape=())

        self.max_body_num = 2
        self.max_material_num = 0
        self.max_particle_num = 1
        self.max_coupling_particle_num = 1
        self.verlet_distance_multiplier = 0
        self.verlet_distance = 0.0
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
        self.is_continue = False

        self.time = 0.0
        self.CFL = 0.2
        self.adaptive_timestep = 0
        self.save_interval = 1e6
        self.path = None
        self.contact_detection = None

        self.visualize_interval = 0.0
        self.window_size = (1024, 1024)
        self.camera_up = (0.0, 1.0, 0.0)
        self.look_at = (0.0, 1.0, 0.0)
        self.look_from = (0.0, 0.0, 0.0)
        self.particle_color = (1, 1, 1)
        self.background_color = (0, 0, 0)
        self.point_light = (0, 0, 0)
        self.view_angle = 45.0
        self.move_velocity = 0.0

        self.calculate_reaction_force = False
        self.displacement_tolerance = 1e-4
        self.residual_tolerance = 1e-7
        self.linear_solver_relative_tolerance = 0.0
        self.symmetrize_matrix_free_tangent = False
        self.use_elastic_matrix_free_tangent = False
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
        self.assemble_type_input = "MatrixFree"
        self.pressure_solver = "PCG"
        self.hash_triplet_matrix_symmetric = None
        self.integration_scheme = "Newmark"
        self.configuration = "ULMPM"
        self.material_type = "Solid"
        self.solver_type = "Explicit"
        self.discretization = "FEM"

        self.fracture_analysis = False
        self.pressure_stabilize = None
        self.poisson_equation = False
        self.poisson_equation_u_p = False
        self.pressure_beta = 1.0
        self.npressure = None
        self.pressure_constraint_list = None

        self.TESTMODE = False

    def get_simulation_domain(self):
        return self.domain

    def set_dimension(self):
        self.dimension = GlobalVariable.DIMENSION

    def set_is_2DAxisy(self, is_2DAxisy):
        self.is_2DAxisy = is_2DAxisy

    def set_mode(self, mode):
        aliases = {
            "normal": "Normal",
            "lightweight": "Lightweight",
            "softparticle": "SoftParticle",
            "softparticles": "SoftParticle",
            "lsmpm": "SoftParticle",
            "levelsetmpm": "SoftParticle",
        }
        key = str(mode).strip().replace("_", "").replace("-", "").replace(" ", "").lower()
        mode = aliases.get(key)
        valid_list = ["Normal", "Lightweight", "SoftParticle"]
        if mode not in valid_list:
            raise RuntimeError(f"Keyword:: /mode/ must be {valid_list}")
        self.mode = mode
        if self.mode == "SoftParticle":
            self.soft_particle = True

    def set_soft_particle_mode(self, soft_particle=False, backend="LevelSetTLMPM"):
        if isinstance(soft_particle, dict):
            backend = DictIO.GetAlternative(soft_particle, "Backend", backend)
            soft_particle = DictIO.GetAlternative(soft_particle, "Enabled", True)
        self.soft_particle = bool(soft_particle)
        if self.soft_particle:
            self.mode = "SoftParticle"
            self.soft_particle_backend = str(backend)
        elif self.mode == "SoftParticle":
            self.mode = "Normal"

    def initialize_soft_particle_options(self):
        self.soft_shape_function = "QuadBSpline"
        self.soft_shape_function_type = 1
        self.soft_influenced_node = 3
        self.soft_shape_nodes = 27
        self.soft_grid_type = "Hexahedron"
        self.soft_grid_type_id = 0
        self.soft_mechanical_grid_spacing_ratio = 0.15
        self.soft_grid_storage = "Dense"
        self.soft_grid_compact = False
        self.soft_levelset_transport = True
        self.soft_levelset_reinitialization = True
        self.soft_levelset_advection_scheme = "SemiLagrangian"
        self.soft_levelset_advection_cfl = 1.0
        self.soft_levelset_advection_interval = 1
        self.soft_levelset_advection_elapsed = 0.0
        self.soft_levelset_advection_steps = 0
        self.soft_levelset_reinit_grad_threshold = 0.2
        self.soft_levelset_reinit_check_interval = 5
        self.soft_levelset_reinit_monitor_band = 4.0
        self.soft_levelset_reinit_band = 8.0
        self.soft_levelset_reinit_cfl = 0.3
        self.soft_levelset_reinit_iterations = 0
        self.soft_levelset_max_grad_error = 0.0
        self.soft_levelset_reinitialization_count = 0
        self.soft_levelset_last_reinitialization_step = -1
        self.soft_levelset_last_grad_error_before = 0.0
        self.soft_levelset_last_grad_error_after = 0.0
        self.soft_levelset_reinitialization_events = []
        self.soft_levelset_volume_correction = False
        self.soft_levelset_volume_epsilon_cells = 1.5
        self.soft_levelset_volume_tolerance = 1.0e-6
        self.soft_levelset_volume_max_shift_cells = 0.25
        self.soft_levelset_volume_iterations = 6
        self.soft_levelset_volume_interval = 1
        self.soft_levelset_volume_reference_initialized = False
        self.soft_levelset_volume_reference_count = 0
        self.soft_levelset_volume_update_count = 0
        self.soft_levelset_volume_correction_count = 0
        self.soft_levelset_volume_max_error = 0.0
        self.soft_levelset_domain_check = True
        self.soft_levelset_domain_tolerance_cells = 1.0e-5
        self.soft_levelset_domain_audit_count = 0
        self.soft_levelset_domain_min_margin_cells = math.inf
        self.soft_levelset_domain_max_departure_excess_cells = 0.0
        self.soft_levelset_projection_monitor_band = 1.5
        self.soft_levelset_projection_extension_iterations = 3
        self.soft_levelset_projection_band_nodes = 0
        self.soft_levelset_projection_uncovered_nodes = 0
        self.soft_levelset_projection_max_uncovered_fraction = 0.0
        self.soft_levelset_projection_max_support_loss = 0.0
        self.max_soft_body_num = 0
        self.max_soft_grid_num = 0
        self.max_material_point_num = 0
        self.max_soft_template_point_num = 0
        self.max_soft_template_surface_num = 0
        self.max_soft_template_sdf_num = 0
        self.max_soft_velocity_constraint_num = 0
        self.max_ls_contact_node_num = 0

    def set_soft_body_num(self, soft_body_num):
        if soft_body_num < 0:
            raise ValueError("Soft body number should be larger than 0!")
        self.max_soft_body_num = int(soft_body_num)
        self.max_particle_num += int(soft_body_num)

        if soft_body_num > 0 and (getattr(self, "max_sphere_num", 0) > 0 or getattr(self, "max_clump_num", 0) > 0):
            raise RuntimeError("Sphere/Multisphere particles are not supported when using level set method")

    def set_material_point_num(self, material_point_num):
        if material_point_num < 0:
            raise ValueError("Material point number should be larger than 0!")
        self.max_material_point_num = int(material_point_num)

    def set_soft_grid_num(self, soft_grid_num):
        if soft_grid_num < 0:
            raise ValueError("Soft grid number should be larger than 0!")
        self.max_soft_grid_num = int(soft_grid_num)

    def set_soft_velocity_constraint_num(self, constraint_num):
        if constraint_num < 0:
            raise ValueError("Soft velocity constraint number should be non-negative")
        self.max_soft_velocity_constraint_num = int(constraint_num)

    def set_soft_template_support_num(self, point_num, surface_num, sdf_num):
        if min(point_num, surface_num, sdf_num) < 0:
            raise ValueError("Soft template support capacities should be non-negative")
        self.max_soft_template_point_num = int(point_num)
        self.max_soft_template_surface_num = int(surface_num)
        self.max_soft_template_sdf_num = int(sdf_num)

    def set_soft_shape_function(self, shape_function):
        key = str(shape_function).replace("_", "").replace("-", "").replace(" ", "").lower()
        aliases = {
            "linear": "Linear",
            "quadbspline": "QuadBSpline",
            "quadraticbspline": "QuadBSpline",
            "cubicbspline": "CubicBSpline",
        }
        if key not in aliases:
            raise RuntimeError("LSMPM soft shape_function must be one of ['Linear', 'QuadBSpline', 'CubicBSpline']")
        self.soft_shape_function = aliases[key]
        if self.soft_shape_function == "Linear":
            self.soft_shape_function_type = 0
            self.soft_influenced_node = 2
            self.soft_shape_nodes = 8
        elif self.soft_shape_function == "QuadBSpline":
            self.soft_shape_function_type = 1
            self.soft_influenced_node = 3
            self.soft_shape_nodes = 27
        elif self.soft_shape_function == "CubicBSpline":
            self.soft_shape_function_type = 2
            self.soft_influenced_node = 4
            self.soft_shape_nodes = 64

    def set_soft_grid_storage(self, storage="Dense"):
        key = str(storage).replace("_", "").replace("-", "").replace(" ", "").lower()
        aliases = {
            "dense": "Dense",
            "compact": "Compact",
            "compacted": "Compact",
            "supportcompact": "Compact",
            "supportcompacted": "Compact",
            # Compatibility aliases now select contiguous support compaction;
            # no pointer SNode is created.
            "sparse": "Compact",
            "blocksparse": "Compact",
            "fixedsparse": "Compact",
        }
        if key not in aliases:
            raise RuntimeError("LSMPM soft_grid_storage must be one of ['Dense', 'Compact']")
        self.soft_grid_storage = aliases[key]
        self.soft_grid_compact = self.soft_grid_storage == "Compact"

    def set_soft_grid_type(self, grid_type="Hexahedron"):
        from src.mpm.soft_particle.TemplateSupport import (
            TETRAHEDRON,
            normalize_soft_grid_type,
        )

        name, grid_type_id = normalize_soft_grid_type(grid_type)
        self.soft_grid_type = name
        self.soft_grid_type_id = grid_type_id
        if grid_type_id == TETRAHEDRON:
            self.soft_shape_function = "Linear"
            self.soft_shape_function_type = 0
            self.soft_influenced_node = 2
            self.soft_shape_nodes = 4

    def set_soft_mechanical_grid_spacing_ratio(self, spacing_ratio=0.15):
        spacing_ratio = float(spacing_ratio)
        if not math.isfinite(spacing_ratio) or spacing_ratio <= 0.0:
            raise ValueError("LSMPM soft mechanical-grid spacing ratio must be positive")
        self.soft_mechanical_grid_spacing_ratio = spacing_ratio

    def set_soft_levelset_reinitialization(
        self,
        enabled=None,
        grad_threshold=None,
        check_interval=None,
        monitor_band=None,
        reinit_band=None,
        cfl=None,
        iterations=None,
        threshold=None,
        monitor_band_cells=None,
        reinit_band_cells=None,
        pseudo_time_cfl=None,
        advection_interval=None,
        advection_scheme=None,
        advection_cfl=None,
        transport_enabled=None,
        volume_correction=None,
        preserve_volume=None,
        volume_epsilon_cells=None,
        volume_tolerance=None,
        volume_max_shift_cells=None,
        volume_iterations=None,
        volume_interval=None,
        domain_check=None,
        domain_tolerance_cells=None,
    ):
        if threshold is not None and grad_threshold is None:
            grad_threshold = threshold
        if monitor_band_cells is not None and monitor_band is None:
            monitor_band = monitor_band_cells
        if reinit_band_cells is not None and reinit_band is None:
            reinit_band = reinit_band_cells
        if pseudo_time_cfl is not None and cfl is None:
            cfl = pseudo_time_cfl
        if transport_enabled is not None:
            self.soft_levelset_transport = bool(transport_enabled)
        if enabled is not None:
            self.soft_levelset_reinitialization = bool(enabled)
        if preserve_volume is not None and volume_correction is None:
            volume_correction = preserve_volume
        if volume_correction is not None:
            self.soft_levelset_volume_correction = bool(volume_correction)
            self.soft_levelset_volume_reference_initialized = False
            self.soft_levelset_volume_reference_count = 0
        if advection_scheme is not None:
            key = str(advection_scheme).replace("_", "").replace("-", "").replace(" ", "").lower()
            aliases = {
                "weno5": "WENO5",
                "wenofive": "WENO5",
                "semilagrangian": "SemiLagrangian",
                "maccormack": "SemiLagrangian",
            }
            if key not in aliases:
                raise RuntimeError("LSMPM soft level-set /advection_scheme/ must be " "'WENO5' or 'SemiLagrangian'")
            self.soft_levelset_advection_scheme = aliases[key]
            if advection_cfl is None:
                self.soft_levelset_advection_cfl = 0.20 if self.soft_levelset_advection_scheme == "WENO5" else 1.0
        if advection_cfl is not None:
            if not math.isfinite(advection_cfl) or advection_cfl <= 0.0:
                raise RuntimeError("LSMPM soft level-set /advection_cfl/ should be positive")
            self.soft_levelset_advection_cfl = float(advection_cfl)
        if advection_interval is not None:
            if advection_interval < 1:
                raise RuntimeError("LSMPM soft level-set /advection_interval/ should be larger than 0")
            self.soft_levelset_advection_interval = int(advection_interval)
        if grad_threshold is not None:
            if grad_threshold < 0.0:
                raise RuntimeError("LSMPM soft level-set reinitialization /grad_threshold/ should be non-negative")
            self.soft_levelset_reinit_grad_threshold = float(grad_threshold)
        if check_interval is not None:
            if check_interval < 1:
                raise RuntimeError("LSMPM soft level-set reinitialization /check_interval/ should be larger than 0")
            self.soft_levelset_reinit_check_interval = int(check_interval)
        if monitor_band is not None:
            if monitor_band < 1.0:
                raise RuntimeError("LSMPM soft level-set reinitialization /monitor_band/ should be at least 1 cell")
            self.soft_levelset_reinit_monitor_band = float(monitor_band)
        if reinit_band is not None:
            if reinit_band < 2.0:
                raise RuntimeError("LSMPM soft level-set reinitialization /reinit_band/ should be at least 2 cells")
            self.soft_levelset_reinit_band = float(reinit_band)
        if cfl is not None:
            if cfl <= 0.0:
                raise RuntimeError("LSMPM soft level-set reinitialization /cfl/ should be positive")
            self.soft_levelset_reinit_cfl = float(cfl)
        if iterations is not None:
            if iterations < 0:
                raise RuntimeError("LSMPM soft level-set reinitialization /iterations/ should be non-negative")
            self.soft_levelset_reinit_iterations = int(iterations)
        if volume_epsilon_cells is not None:
            if volume_epsilon_cells <= 0.0:
                raise RuntimeError("LSMPM soft level-set /volume_epsilon_cells/ should be positive")
            self.soft_levelset_volume_epsilon_cells = float(volume_epsilon_cells)
        if volume_tolerance is not None:
            if volume_tolerance < 0.0:
                raise RuntimeError("LSMPM soft level-set /volume_tolerance/ should be non-negative")
            self.soft_levelset_volume_tolerance = float(volume_tolerance)
        if volume_max_shift_cells is not None:
            if volume_max_shift_cells <= 0.0:
                raise RuntimeError("LSMPM soft level-set /volume_max_shift_cells/ should be positive")
            self.soft_levelset_volume_max_shift_cells = float(volume_max_shift_cells)
        if volume_iterations is not None:
            if volume_iterations < 1:
                raise RuntimeError("LSMPM soft level-set /volume_iterations/ should be at least 1")
            self.soft_levelset_volume_iterations = int(volume_iterations)
        if volume_interval is not None:
            if volume_interval < 1:
                raise RuntimeError("LSMPM soft level-set /volume_interval/ should be at least 1")
            self.soft_levelset_volume_interval = int(volume_interval)
        if domain_check is not None:
            self.soft_levelset_domain_check = bool(domain_check)
        if domain_tolerance_cells is not None:
            if domain_tolerance_cells < 0.0:
                raise RuntimeError("LSMPM soft level-set /domain_tolerance_cells/ should " "be non-negative")
            self.soft_levelset_domain_tolerance_cells = float(domain_tolerance_cells)
        if self.soft_levelset_reinit_band < self.soft_levelset_reinit_monitor_band + 2.0:
            self.soft_levelset_reinit_band = self.soft_levelset_reinit_monitor_band + 2.0

    def set_levelset_contact_list_size(self):
        if getattr(self, "scheme", None) == "LSMPM":
            level_body_num = self.max_rigid_body_num + self.max_soft_body_num
            self.max_ls_contact_node_num = self.max_surface_node_num * level_body_num
            self.potential_contact_points_particle = int(
                self.point_particle_coordination_number * self.max_ls_contact_node_num
            )
            self.potential_contact_points_wall = int(self.point_wall_coordination_number * self.max_ls_contact_node_num)
        else:
            self.max_ls_contact_node_num = self.max_surface_node_num * self.max_particle_num
            self.potential_contact_points_particle = int(
                self.point_particle_coordination_number * self.max_surface_node_num
            )
            self.potential_contact_points_wall = int(self.point_wall_coordination_number * self.max_surface_node_num)

    def validate_soft_particle_configuration(self, require_memory=False):
        if require_memory and getattr(self, "scheme", None) == "LSMPM":
            if self.max_soft_body_num <= 0:
                raise RuntimeError("LSMPM requires /max_soft_body_number/ larger than 0")
            if self.max_material_point_num <= 0:
                raise RuntimeError("LSMPM requires /max_material_point_number/ larger than 0")
            if self.max_soft_grid_num <= 0:
                raise RuntimeError("LSMPM requires /soft_grid_number/ larger than 0")

    def soft_particle_contact_list_length(self):
        particle_contact_list_length = int(math.ceil(self.compaction_ratio[2] * self.potential_contact_points_particle))
        use_heightfield = getattr(self, "use_digital_elevation_heightfield", lambda: False)
        wall_contact_list_length = (
            0 if use_heightfield() else int(math.ceil(self.compaction_ratio[3] * self.potential_contact_points_wall))
        )
        return particle_contact_list_length, wall_contact_list_length

    def set_mpm_backend(self, backend="Native"):
        aliases = {
            "native": "Native",
            "geotaichi": "Native",
            "mpm": "Native",
            "direct": "Direct",
            "standalone": "Direct",
            "classic": "Direct",
        }
        key = str(backend).strip().replace(" ", "").lower()
        if key not in aliases:
            raise RuntimeError("Keyword:: /mpm_backend/ must be 'Native' or 'Direct'")
        self.mpm_backend = aliases[key]

    def is_direct_backend(self):
        return self.mpm_backend == "Direct"

    def set_ipc_contact(self, ipc_contact=False, parameters=None):
        self.ipc_contact = bool(ipc_contact)
        if parameters is not None:
            self.ipc_contact_parameters = dict(parameters)

    def set_direct_static_twophase(self, static_twophase=False):
        self.direct_static_twophase = bool(static_twophase)

    def set_domain(self, domain):
        self.domain = domain
        if isinstance(domain, (list, tuple)):
            if self.dimension == 3:
                self.domain = vec3f(domain)
            elif self.dimension == 2:
                self.domain = vec2f(domain)

    def set_boundary(self, boundary):
        BOUNDARY = {None: -1, "Reflect": 0, "Destroy": 1, "Period": 2}
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
            gravity = [gravity[0], gravity[1], 0.0]
        if isinstance(gravity, (list, tuple)):
            self.gravity = vec3f(gravity)

    def set_background_damping(self, background_damping):
        self.background_damping = background_damping

    def set_alpha(self, alphaPIC):
        self.alphaPIC = alphaPIC

    def set_stabilize_technique(self, stabilize):
        typelist = [None, "B-Bar Method", "F-Bar Method", "Displacement F-Bar Method"]
        if not stabilize in typelist:
            raise RuntimeError(
                f"KeyWord:: /stabilize: {stabilize}/ is invalid. The valid type are given as follows: {typelist}"
            )
        self.stabilize = stabilize

        if self.stabilize == "B-Bar Method":
            GlobalVariable.BBAR = True
        elif self.stabilize == "F-Bar Method":
            GlobalVariable.FBAR = True

    def set_pressure_stabilize_technique(self, stabilize):
        typelist = [None, "FIC"]
        if not stabilize in typelist:
            raise RuntimeError(
                f"KeyWord:: /stabilize: {stabilize}/ is invalid. The valid type are given as follows: {typelist}"
            )
        self.pressure_stabilize = stabilize

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
            self.grid_layer = THBparameter["grid_layer"]

    def set_sparse_grid(self, sparse_grid):
        if not sparse_grid:
            self.sparse_grid = False
            self.sparse_grid_visualize_active_blocks = False
            return

        self.sparse_grid = True
        if isinstance(sparse_grid, dict):
            enabled = DictIO.GetAlternative(sparse_grid, "Enabled", True)
            if not enabled:
                self.sparse_grid = False
                self.sparse_grid_visualize_active_blocks = False
                return
            backend = DictIO.GetAlternative(sparse_grid, "Backend", "BlockScan")
            backend_key = str(backend).strip().replace("_", "").replace("-", "").replace(" ", "").lower()
            if backend_key not in ("blockscan", "scan", "parallelblockscan"):
                raise RuntimeError(
                    "MPM sparse_grid now uses the block-level scan backend. "
                    "Valid sparse_grid/Backend values are 'BlockScan' or 'Scan'."
                )
            self.sparse_grid_backend = "BlockScan"
            block_size = int(DictIO.GetAlternative(sparse_grid, "BlockSize", self.sparse_grid_block_size))
            if block_size <= 1:
                raise ValueError("sparse_grid/BlockSize must be larger than 1")
            self.sparse_grid_block_size = block_size
            capacity_factor = float(
                DictIO.GetAlternative(sparse_grid, "CapacityFactor", self.sparse_grid_capacity_factor)
            )
            if capacity_factor <= 0.0:
                raise ValueError("sparse_grid/CapacityFactor must be positive")
            self.sparse_grid_capacity_factor = capacity_factor
            self.sparse_grid_max_blocks = int(
                DictIO.GetAlternative(sparse_grid, "MaxActiveBlocks", self.sparse_grid_max_blocks)
            )
            if self.sparse_grid_max_blocks < 0:
                raise ValueError("sparse_grid/MaxActiveBlocks must be non-negative")
            self.sparse_grid_visualize_active_blocks = bool(
                DictIO.GetAlternative(
                    sparse_grid,
                    "VisualizeActiveBlocks",
                    self.sparse_grid_visualize_active_blocks,
                )
            )

    def set_gauss_integration(self, gauss_number):
        self.gauss_number = gauss_number

    def set_particle_shifting(self, particle_shifting):
        self.particle_shifting = particle_shifting
        GlobalVariable.PARTICLESHIFTING = particle_shifting

    def set_particle_shifting_scale(self, particle_shifting_scale):
        particle_shifting_scale = float(particle_shifting_scale)
        if not 0.0 <= particle_shifting_scale <= 1.0:
            raise ValueError("particle_shifting_scale must be in [0, 1]")
        self.particle_shifting_scale = particle_shifting_scale

    def set_density_projection(
        self, density_projection, tolerance=0.01, error_clamp=0.1, max_shift_ratio=0.05, interior_only=True
    ):
        self.density_projection = density_projection
        self.density_projection_tolerance = tolerance
        self.density_projection_error_clamp = error_clamp
        self.density_projection_max_shift_ratio = max_shift_ratio
        self.density_projection_interior_only = interior_only

    def set_delayed_fluid_advection(self, delayed_fluid_advection):
        self.delayed_fluid_advection = bool(delayed_fluid_advection)

    def set_solid_sdf_cut_cell(self, solid_sdf_cut_cell, min_fraction=0.01, no_slip=False):
        self.solid_sdf_cut_cell = solid_sdf_cut_cell
        self.solid_cut_cell_min_fraction = min_fraction
        self.fluid_wall_no_slip = bool(no_slip)

    def set_stress_integration(self, stress_integration):
        typelist = ["ReturnMapping", "ImplicitIntegration", "ImplicitIntegrationAL", "SubStepping"]
        if not stress_integration in typelist:
            raise RuntimeError(
                f"KeyWord:: /stress_integration: {stress_integration}/ is invalid. The valid type are given as follows: {typelist}"
            )
        self.stress_integration = stress_integration

    def set_discretization(self, discretization):
        typelist = ["FEM", "FDM"]
        if not discretization in typelist:
            raise RuntimeError(
                f"KeyWord:: /discretization: {discretization}/ is invalid. The valid type are given as follows: {typelist}"
            )
        self.discretization = discretization

    def set_moving_least_square(self, mls):
        self.mls = mls
        if self.mapping == "G2P2G":
            self.mls = True
        if mls is True:
            self.set_velocity_projection_scheme("Affine")
            self.alphaPIC = 1.0

    def set_mapping_scheme(self, mapping):
        typelist = ["USL", "USF", "MUSL", "G2P2G"]
        if not mapping in typelist:
            raise RuntimeError(
                f"KeyWord:: /mapping: {mapping}/ is invalid. The valid type are given as follows: {typelist}"
            )
        self.mapping = mapping

    def set_shape_function(self, shape_function):
        typelist = ["Linear", "SmoothLinear", "GIMP", "QuadBSpline", "CubicBSpline"]
        if not shape_function in typelist:
            raise RuntimeError(
                f"KeyWord:: /mapping: {shape_function}/ is invalid. The valid type are given as follows: {typelist}"
            )
        self.shape_function = shape_function
        if self.mapping == "G2P2G":
            self.shape_function == "QuadBSpline"
        if self.shape_function == "Linear" or self.shape_function == "SmoothLinear":
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
            raise RuntimeError(
                f"KeyWord:: /coupling: {coupling}/ is invalid. The valid type are given as follows: {typelist}"
            )
        self.coupling = coupling

    def set_free_surface_detection(self, free_surface_detection):
        if self.norm_adaptivity:
            free_surface_detection = True
        self.free_surface_detection = free_surface_detection
        if free_surface_detection is True:
            self.neighbor_detection = True
            self.boundary_direction_detection = True

    def set_fluid_level_set(self, fluid_level_set):
        self.fluid_level_set = fluid_level_set

    def set_fluid_domain_volume_fraction(self, volume_fraction):
        volume_fraction = float(volume_fraction)
        if not 0.0 <= volume_fraction <= 1.0:
            raise ValueError("fluid_domain_volume_fraction must be in [0, 1]")
        self.fluid_domain_volume_fraction = volume_fraction

    def set_boundary_direction(self, boundary_direction_detection):
        self.boundary_direction_detection = boundary_direction_detection
        if boundary_direction_detection is True:
            self.neighbor_detection = True

    def set_solver_type(self, solver_type):
        typelist = ["Explicit", "Implicit", "SemiImplicit", "SemiImplicit_u_p"]
        if not solver_type in typelist:
            raise RuntimeError(
                f"KeyWord:: /solver_type: {solver_type}/ is invalid. The valid type are given as follows: {typelist}"
            )
        self.solver_type = solver_type

        if self.solver_type == "SemiImplicit":
            self.poisson_equation = True
        if self.solver_type == "SemiImplicit_u_p":
            self.poisson_equation_u_p = True

    def set_is_continue(self, is_continue):
        self.is_continue = is_continue

    def set_track_energy(self, track_energy):
        self.energy_tracking = track_energy
        GlobalVariable.TRACKENERGY = track_energy

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

    def set_adaptive_timestep(self, adaptive_timestep):
        self.adaptive_timestep = int(adaptive_timestep)

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
            raise RuntimeError(
                "It is unnecessary to using coupling modules because no material points are considered in coupling process"
            )
        self.max_coupling_particle_num = coupling_particles

    def set_velocity_projection_scheme(self, velocity_projection_scheme: str):
        valid_type = ["PIC", "FLIP", "PIC/FLIP", "Affine", "Taylor"]
        if velocity_projection_scheme not in valid_type:
            raise RuntimeError(
                f"Keyword:: /velocity_projection_scheme/ is error, followings are valid {valid_type}. Use velocity_projection='Affine' with alphaPIC=1.0 for affine PIC transfer."
            )

        self.velocity_projection_scheme = velocity_projection_scheme
        if velocity_projection_scheme == "PIC":
            self.alphaPIC = 1.0
        elif velocity_projection_scheme == "FLIP":
            self.alphaPIC = 0.0

        GlobalVariable.APIC = False
        GlobalVariable.TPIC = False
        if self.velocity_projection_scheme == "Affine":
            GlobalVariable.APIC = True
        elif self.velocity_projection_scheme == "Taylor":
            GlobalVariable.TPIC = True

    def set_window_parameters(self, windows):
        self.visualize_interval = DictIO.GetAlternative(windows, "VisualizeInterval", self.save_interval)
        self.window_size = DictIO.GetAlternative(windows, "WindowSize", self.window_size)
        self.camera_up = DictIO.GetAlternative(windows, "CameraUp", self.camera_up)
        self.look_at = DictIO.GetAlternative(windows, "LookAt", self.look_at)
        self.look_from = DictIO.GetAlternative(
            windows, "LookFrom", (0.7 * self.domain[0], -0.4 * self.domain[1], 1.5 * self.domain[2])
        )
        self.particle_color = DictIO.GetAlternative(windows, "ParticleColor", (1, 1, 1))
        self.background_color = DictIO.GetAlternative(windows, "BackgroundColor", (0, 0, 0))
        self.point_light = DictIO.GetAlternative(
            windows, "PointLight", (0.5 * self.domain[0], 0.5 * self.domain[1], 1.0 * self.domain[2])
        )
        self.view_angle = DictIO.GetAlternative(windows, "ViewAngle", 45.0)
        self.move_velocity = DictIO.GetAlternative(
            windows, "MoveVelocity", 0.01 * (self.domain[0] + self.domain[1] + self.domain[2])
        )

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
        npressure = int(DictIO.GetAlternative(constraint, "max_node_pressure_constraint", 0))
        if self.solver_type == "SemiImplicit" and npressure >= 1:
            self.npressure = ti.field(dtype=ti.i32, shape=())
            self.npressure[None] = npressure
            self.pressure_constraint_list = ti.field(dtype=ti.i32, shape=npressure)

    def set_particle_traction_method(self, particle_traction_method):
        self.ptraction_method = particle_traction_method
        valid_list = [None, "Stable", "Nanson", "Virtual"]
        if self.ptraction_method not in valid_list:
            warnings.warn("We choose stable version of particle traction by default!")
            self.ptraction_method = "Stable"

        if self.ptraction_method == "Virtual":
            self.nptraction = 1

    def set_particle_traction_update_area(self, update_area):
        self.particle_traction_update_area = bool(update_area)

    def set_AOSOA(self, AOSOA):
        if AOSOA:
            self.AOSOA = True
            if isinstance(AOSOA, (list, tuple)):
                if len(AOSOA) >= 1:
                    self.block_size = list(AOSOA)
                else:
                    raise ValueError(
                        f"Keyword:: /AOSOA/ is empty. The input {AOSOA} is invalid. For example, you can input [grid_block_size, leaf_block_size]."
                    )

    def set_random_field(self, random_field):
        if random_field:
            self.random_field = True
            GlobalVariable.RANDOMFIELD = True

    def set_drift_correct(self, drift_correct):
        if drift_correct is False:
            GlobalVariable.DRIFTCORRECT = drift_correct

    def set_save_data(self, particle, grid, object):
        if particle:
            self.monitor_type.append("particle")
        if grid:
            self.monitor_type.append("grid")
        if object:
            self.monitor_type.append("object")

    def update_critical_timestep(self, critical_timestep):
        dt = self.CFL * critical_timestep
        if dt < self.dt[None]:
            print("The time step is corrected as:", dt, "\n")
            self.dt[None] = dt
            self.delta = dt
        else:
            dt = self.dt[None]
            print("The prescribed time step is sufficiently small\n")
        self.init_delta = dt

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
        displacement_tolerance = float(displacement_tolerance)
        if not math.isfinite(displacement_tolerance) or displacement_tolerance <= 0.0:
            raise ValueError("implicit MPM displacement_tolerance must be finite and positive")
        self.displacement_tolerance = displacement_tolerance

    def set_residual_tolerance(self, residual_tolerance):
        residual_tolerance = float(residual_tolerance)
        if not math.isfinite(residual_tolerance) or residual_tolerance < 0.0:
            raise ValueError("implicit MPM residual_tolerance must be finite and non-negative")
        self.residual_tolerance = residual_tolerance

    def set_linear_solver_relative_tolerance(self, relative_tolerance):
        relative_tolerance = float(relative_tolerance)
        if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
            raise ValueError("implicit MPM linear solver relative tolerance must be finite " "and non-negative")
        self.linear_solver_relative_tolerance = relative_tolerance

    def set_symmetrize_matrix_free_tangent(self, symmetrize):
        self.symmetrize_matrix_free_tangent = bool(symmetrize)

    def set_use_elastic_matrix_free_tangent(self, use_elastic_tangent):
        self.use_elastic_matrix_free_tangent = bool(use_elastic_tangent)

    def set_quasi_static(self, quasi_static):
        self.quasi_static = quasi_static

    def set_max_iteration(self, iter_max):
        numeric = float(iter_max)
        integer = int(numeric)
        if not math.isfinite(numeric) or numeric != integer or integer <= 0:
            raise ValueError("implicit MPM max_iteration_number must be a positive integer")
        self.iter_max = integer

    def set_newmark_parameter(self, newmark_parameter):
        newmark_parameter = list(newmark_parameter)
        if len(newmark_parameter) != 2:
            raise RuntimeError("The size of newmark parameter should follow [gamma, beta]")
        self.newmark_gamma = newmark_parameter[0]
        self.newmark_beta = newmark_parameter[1]

    def set_pressure_parameter(self, pressure_beta):
        pressure_beta = float(pressure_beta)
        if not math.isfinite(pressure_beta) or not 0.0 <= pressure_beta <= 1.0:
            raise ValueError("pressure_beta must lie in [0, 1]")
        self.pressure_beta = pressure_beta

    def set_dof_multiplier(self, dof_multiplier):
        self.dof_multiplier = dof_multiplier

    def set_linear_solver(self, linear_solver):
        self.linear_solver = linear_solver

        valid_list = ["CG", "PCG", "BiCG", "MGPCG"]
        if linear_solver not in valid_list:
            raise RuntimeError(f"Keyword:: /linear_solver/ is error, followings are valid {valid_list}")
        if self.solver_type in ("SemiImplicit", "SemiImplicit_u_p") and linear_solver == "MGPCG":
            self.pressure_solver = "MGPCG"

    def set_pressure_solver(self, pressure_solver):
        if pressure_solver is None:
            return
        aliases = {
            "pcg": "PCG",
            "matrixfree": "PCG",
            "matrixfreepcg": "PCG",
            "mgpcg": "MGPCG",
            "matrixfreemgpcg": "MGPCG",
        }
        solver = aliases.get(_normalize_solver_keyword(pressure_solver))
        if solver is None:
            valid_list = ["PCG", "MGPCG"]
            raise RuntimeError(f"Keyword:: /pressure_solver/ is error, followings are valid {valid_list}")
        self.pressure_solver = solver

    def use_mgpcg_pressure_solver(self):
        return self.pressure_solver == "MGPCG" or (
            self.solver_type in ("SemiImplicit", "SemiImplicit_u_p") and self.linear_solver == "MGPCG"
        )

    def set_hash_triplet_matrix_symmetric(self, matrix_symmetric):
        if matrix_symmetric is None:
            self.hash_triplet_matrix_symmetric = None
        else:
            self.hash_triplet_matrix_symmetric = bool(matrix_symmetric)

    def set_assemble_type(self, assemble_type):
        aliases = {
            "matrixfree": "MatrixFree",
            "mf": "MatrixFree",
            "coo": "COO",
            "hashtriplet": "HashTriplet",
            "hash": "HashTriplet",
            "csr": "COO",
            "matrixfreemgpcg": "MatrixFreeMGPCG",
        }
        self.assemble_type_input = assemble_type
        canonical = aliases.get(_normalize_solver_keyword(assemble_type))
        if canonical is None:
            valid_list = ["MatrixFree", "COO", "HashTriplet", "MatrixFreeMGPCG"]
            raise RuntimeError(f"Keyword:: /assemble_type/ is error, followings are valid {valid_list}")

        if canonical == "MatrixFreeMGPCG":
            if self.solver_type not in ("SemiImplicit", "SemiImplicit_u_p"):
                raise RuntimeError(
                    "assemble_type='MatrixFreeMGPCG' is a legacy SemiImplicit pressure-solver alias. "
                    "Use assemble_type='MatrixFree', 'COO', or 'HashTriplet' for Implicit solid MPM."
                )
            self.assemble_type = "MatrixFree"
            self.pressure_solver = "MGPCG"
            return

        if self.solver_type == "Implicit" and self.material_type == "Solid":
            valid_list = ["MatrixFree", "COO", "HashTriplet"]
            if canonical not in valid_list:
                raise RuntimeError(
                    f"Implicit solid MPM supports assemble_type {valid_list}; "
                    "use COO or HashTriplet for assembled matrices."
                )
        elif self.solver_type in ("SemiImplicit", "SemiImplicit_u_p"):
            valid_list = ["MatrixFree", "COO"]
            if canonical not in valid_list:
                raise RuntimeError(
                    f"{self.solver_type} MPM supports assemble_type {valid_list}; "
                    "use pressure_solver='MGPCG' for the multigrid pressure path."
                )
        self.assemble_type = canonical

    def set_multigrid_paramter(self, multilevel, pre_and_post_smoothing, bottom_smoothing):
        self.multilevel = int(multilevel)
        self.pre_and_post_smoothing = int(pre_and_post_smoothing)
        self.bottom_smoothing = int(bottom_smoothing)

    def is_incompressible_fluid_fdm(self):
        return self.solver_type == "Implicit" and self.material_type == "Fluid" and self.discretization == "FDM"

    def _raise_unsupported(self, feature, requirements):
        raise RuntimeError(f"{feature} currently requires: {', '.join(requirements)}")

    def validate_configuration(self, require_solver_parameters=True):
        if self.dimension not in (2, 3):
            raise RuntimeError("MPM currently supports dimension=2 or dimension=3")
        if self.pressure_stabilize == "FIC" and (
            self.solver_type != "SemiImplicit" or self.material_type != "TwoPhaseSingleLayer"
        ):
            self._raise_unsupported("FIC pressure stabilization", ["SemiImplicit TwoPhaseSingleLayer"])
        if self.is_direct_backend():
            if self.solver_type not in ("Explicit", "Implicit"):
                self._raise_unsupported("Direct MPM backend", ["solver_type='Explicit' or 'Implicit'"])
            if self.configuration not in ("ULMPM", "TLMPM"):
                self._raise_unsupported("Direct MPM backend", ["configuration='ULMPM' or 'TLMPM'"])
            if self.material_type != "Solid":
                self._raise_unsupported("Direct MPM backend", ["material_type='Solid'"])
            if self.soft_particle:
                self._raise_unsupported("Direct MPM backend", ["soft_particle=False"])
            if self.coupling:
                self._raise_unsupported("Direct MPM backend", ["coupling=False"])
            return
        if self.is_2DAxisy and self.dimension != 2:
            self._raise_unsupported("2D axisymmetric MPM", ["dimension=2"])
        if self.isTHB and self.dimension != 2:
            self._raise_unsupported("THB MPM", ["dimension=2"])

        if self.configuration == "TLMPM":
            if self.solver_type != "Explicit":
                self._raise_unsupported("TLMPM", ["solver_type='Explicit'"])
            if self.material_type != "Solid":
                self._raise_unsupported("TLMPM", ["material_type='Solid'"])
            if self.discretization != "FEM":
                self._raise_unsupported("TLMPM", ["discretization='FEM'"])
            if self.is_2DAxisy:
                self._raise_unsupported("TLMPM", ["is_2DAxisy=False"])
            if self.stabilize is not None:
                self._raise_unsupported("TLMPM", ["stabilize=None"])
            if self.mls:
                self._raise_unsupported("TLMPM", ["moving_least_square=False"])
            if self.gauss_number > 0:
                self._raise_unsupported("TLMPM", ["gauss_number=0"])
            if self.dimension == 2 and self.coupling:
                self._raise_unsupported(
                    "2D TLMPM coupling", ["coupling=False or a 2D coupling particle implementation"]
                )

        if self.soft_particle:
            if self.mode != "SoftParticle":
                self._raise_unsupported("MPM soft_particle mode", ["mode='SoftParticle'"])
            if self.dimension != 3:
                self._raise_unsupported("MPM soft_particle mode", ["dimension=3"])
            if self.configuration != "TLMPM":
                self._raise_unsupported("MPM soft_particle mode", ["configuration='TLMPM'"])
            if self.solver_type != "Explicit":
                self._raise_unsupported("MPM soft_particle mode", ["solver_type='Explicit'"])
            if self.material_type != "Solid":
                self._raise_unsupported("MPM soft_particle mode", ["material_type='Solid'"])
            if self.discretization != "FEM":
                self._raise_unsupported("MPM soft_particle mode", ["discretization='FEM'"])
            if self.stabilize is not None:
                self._raise_unsupported("MPM soft_particle mode", ["stabilize=None"])
            if self.gauss_number > 0:
                self._raise_unsupported("MPM soft_particle mode", ["gauss_number=0"])

        if self.mode == "Lightweight":
            if self.configuration != "ULMPM" or self.solver_type != "Explicit":
                self._raise_unsupported("Lightweight MPM", ["ULMPM", "solver_type='Explicit'"])

        if self.sparse_grid:
            if self.sparse_grid_backend != "BlockScan":
                self._raise_unsupported("MPM sparse_grid", ["Backend='BlockScan'"])
            if self.AOSOA:
                self._raise_unsupported("BlockScan sparse_grid", ["AOSOA=False"])
            if self.solver_type not in ("Explicit", "Implicit"):
                self._raise_unsupported(
                    "BlockScan sparse_grid",
                    ["solver_type='Explicit' or implicit solid MPM"],
                )
            if self.solver_type == "Implicit" and self.material_type != "Solid":
                self._raise_unsupported("Implicit BlockScan sparse_grid", ["material_type='Solid'"])
            if self.mode != "Normal":
                self._raise_unsupported("BlockScan sparse_grid", ["mode='Normal'"])
            if self.mapping == "G2P2G":
                self._raise_unsupported("BlockScan sparse_grid", ["mapping='USL', 'USF', or 'MUSL'"])
            if self.coupling not in (False, "Lagrangian"):
                self._raise_unsupported(
                    "BlockScan sparse_grid",
                    ["coupling=False or coupling='Lagrangian'"],
                )
            if self.mls:
                self._raise_unsupported("BlockScan sparse_grid", ["moving_least_square=False"])
            if self.gauss_number > 0:
                self._raise_unsupported("BlockScan sparse_grid", ["gauss_number=0"])
            if self.stabilize == "Displacement F-Bar Method":
                self._raise_unsupported("BlockScan sparse_grid", ["stabilize=None, 'B-Bar Method', or 'F-Bar Method'"])
            if self.isTHB:
                self._raise_unsupported("BlockScan sparse_grid", ["set_THB=False"])
            if self.is_2DAxisy:
                self._raise_unsupported("BlockScan sparse_grid", ["is_2DAxisy=False"])
            if self.contact_detection:
                self._raise_unsupported("BlockScan sparse_grid", ["contact_detection=None"])
            if self.xpbc or self.ypbc or self.zpbc:
                self._raise_unsupported("BlockScan sparse_grid", ["non-periodic domain boundaries"])
            if self.nabsorbing > 0:
                self._raise_unsupported("BlockScan sparse_grid", ["max_absorbing_constraint=0"])
        incompressible_fdm_features = []
        if self.fluid_level_set:
            incompressible_fdm_features.append("fluid_level_set")
        if self.density_projection:
            incompressible_fdm_features.append("density_projection")
        if self.solid_sdf_cut_cell:
            incompressible_fdm_features.append("solid_sdf_cut_cell")
        if self.fluid_wall_no_slip and not self.solid_sdf_cut_cell:
            raise RuntimeError("fluid_wall_no_slip requires solid_sdf_cut_cell=True")
        if incompressible_fdm_features and not self.is_incompressible_fluid_fdm():
            self._raise_unsupported(
                ", ".join(incompressible_fdm_features),
                ["solver_type='Implicit'", "material_type='Fluid'", "discretization='FDM'"],
            )

        if self.solver_type == "Explicit":
            if self.discretization != "FEM":
                self._raise_unsupported("Explicit MPM", ["discretization='FEM'"])
            if self.material_type == "TwoPhaseSingleLayer":
                if self.is_2DAxisy:
                    self._raise_unsupported("Explicit TwoPhaseSingleLayer", ["Cartesian 2D or 3D"])
                if self.stabilize in ("F-Bar Method", "Displacement F-Bar Method"):
                    self._raise_unsupported("Explicit TwoPhaseSingleLayer", ["stabilize=None, or 2D B-Bar Method"])
                if self.dimension == 3 and self.stabilize == "B-Bar Method":
                    self._raise_unsupported("3D Explicit TwoPhaseSingleLayer", ["stabilize=None"])
                if self.dimension == 3 and self.velocity_projection_scheme in ("Affine", "Taylor"):
                    self._raise_unsupported("3D Explicit TwoPhaseSingleLayer", ["PIC/FLIP velocity projection"])
                if self.mls:
                    self._raise_unsupported("Explicit TwoPhaseSingleLayer", ["moving_least_square=False"])
                if self.gauss_number > 0:
                    self._raise_unsupported("Explicit TwoPhaseSingleLayer", ["gauss_number=0"])
                if self.contact_detection:
                    self._raise_unsupported("Explicit TwoPhaseSingleLayer", ["contact_detection=None"])
            elif self.material_type == "TwoPhaseDoubleLayer":
                self._raise_unsupported("Explicit TwoPhaseDoubleLayer", ["solver_type='SemiImplicit'"])
            if self.assemble_type == "HashTriplet":
                self._raise_unsupported("HashTriplet assemble_type", ["Implicit solid MPM"])
            return

        if self.solver_type == "Implicit":
            if self.material_type == "Solid":
                if self.discretization != "FEM":
                    self._raise_unsupported("Implicit solid MPM", ["discretization='FEM'"])
                if self.integration_scheme != "Newmark":
                    self._raise_unsupported("Implicit solid MPM", ["integration_scheme='Newmark'"])
                if self.assemble_type not in ("MatrixFree", "COO", "HashTriplet"):
                    self._raise_unsupported(
                        "Implicit solid MPM",
                        ["assemble_type='MatrixFree', 'COO', or 'HashTriplet'"],
                    )
            elif self.material_type == "Fluid":
                if self.discretization != "FDM":
                    self._raise_unsupported("Implicit incompressible fluid MPM", ["discretization='FDM'"])
                if self.linear_solver not in ("PCG", "MGPCG"):
                    self._raise_unsupported("Implicit incompressible fluid MPM", ["linear_solver='PCG' or 'MGPCG'"])
            else:
                self._raise_unsupported("Implicit MPM", ["material_type='Solid' or 'Fluid'"])
            return

        if self.solver_type == "SemiImplicit":
            if self.discretization != "FEM":
                self._raise_unsupported("SemiImplicit MPM", ["discretization='FEM'"])
            if self.material_type == "TwoPhaseSingleLayer":
                if self.mapping not in ("USL", "USF"):
                    self._raise_unsupported("SemiImplicit TwoPhaseSingleLayer", ["USL or USF mapping"])
                if require_solver_parameters and self.use_mgpcg_pressure_solver() and self.pressure_beta != 0.0:
                    self._raise_unsupported(
                        "MGPCG SemiImplicit TwoPhaseSingleLayer",
                        ["pressure_beta=0 (non-incremental cell-centred projection)"],
                    )
                if require_solver_parameters and self.dimension == 3 and not self.use_mgpcg_pressure_solver():
                    self._raise_unsupported("3D SemiImplicit TwoPhaseSingleLayer", ["pressure_solver='MGPCG'"])
                if self.dimension == 3 and self.pressure_stabilize == "FIC":
                    self._raise_unsupported("3D SemiImplicit TwoPhaseSingleLayer", ["pressure_stabilize=None"])
                if self.dimension == 3 and self.velocity_projection_scheme in ("Affine", "Taylor"):
                    self._raise_unsupported("3D SemiImplicit TwoPhaseSingleLayer", ["PIC/FLIP velocity projection"])
                if self.is_2DAxisy and self.pressure_stabilize == "FIC":
                    self._raise_unsupported(
                        "Axisymmetric SemiImplicit TwoPhaseSingleLayer", ["pressure_stabilize=None"]
                    )
            elif self.material_type == "TwoPhaseDoubleLayer":
                if self.is_2DAxisy:
                    self._raise_unsupported("SemiImplicit TwoPhaseDoubleLayer", ["Cartesian coordinates"])
                if self.mapping not in ("USL", "USF", "MUSL"):
                    self._raise_unsupported("SemiImplicit TwoPhaseDoubleLayer", ["USL, USF, or MUSL mapping"])
                if require_solver_parameters and not self.use_mgpcg_pressure_solver():
                    self._raise_unsupported("SemiImplicit TwoPhaseDoubleLayer", ["pressure_solver='MGPCG'"])
            else:
                self._raise_unsupported(
                    "SemiImplicit MPM",
                    ["material_type='TwoPhaseSingleLayer' or 'TwoPhaseDoubleLayer'"],
                )
            return

        if self.solver_type == "SemiImplicit_u_p":
            if self.discretization != "FEM":
                self._raise_unsupported("SemiImplicit_u_p MPM", ["discretization='FEM'"])
            if self.material_type != "TwoPhaseSingleLayer":
                self._raise_unsupported("SemiImplicit_u_p MPM", ["material_type='TwoPhaseSingleLayer'"])
            if self.dimension != 2:
                self._raise_unsupported("SemiImplicit_u_p MPM", ["dimension=2"])
            if self.mapping != "USF":
                self._raise_unsupported("SemiImplicit_u_p MPM", ["USF mapping"])
            if self.use_mgpcg_pressure_solver():
                self._raise_unsupported("SemiImplicit_u_p MPM", ["pressure_solver='PCG'", "linear_solver!='MGPCG'"])
