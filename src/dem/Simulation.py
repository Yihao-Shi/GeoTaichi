import taichi as ti
import math, warnings

import src.utils.GlobalVariable as GlobalVariable
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec3i, vec3f
from src.utils.TimeTicker import Timer
from src.utils.constants import Threshold


class Simulation(object):
    def __init__(self) -> None:
        self.dimension = 3
        self.domain = vec3f([0, 0, 0])
        self.boundary = vec3i([0, 0, 0])
        self.gravity = vec3f([0, 0, 0])
        self.engine = None
        self.search = "LinkedCell"
        self.is_continue = False
        self.coupling = False
        self.scheme = None
        self.static_wall = False
        self.sparse_grid = False
        self.energy_tracking = False
        self.enable_shell = False
        self.iterative_model = None
        self.search_direction = "Up"
        self.history_contact_path = {}
        self.timer = Timer()
        self.affine_body = False
        self.enable_step_retry = False
        self.step_retry_max_retries = 2
        self.step_retry_reduction = 0.5
        self.step_retry_minimum_timestep = 0.0
        self._affine_body_parameters_frozen = False
        self._lsmpm_soft_rigid_contact_frozen = False
        self.affine_assemble_type = "MatrixFree"
        self.affine_young_modulus = 1.0e6
        self.affine_dhat = 1.0e-2
        self.affine_barrier_stiffness = 1.0e6
        self.affine_penalty = None
        self.affine_constraint_tolerance = 1.0e-6
        self.affine_contact_model = "BarrierIPC"
        self.affine_contact_damping_stiffness = 0.0
        self.affine_contact_block_capacity = 256
        self.affine_force_local_damping = None
        self.affine_torque_local_damping = None
        self.affine_friction_epsv = 1.0e-4
        # Negative Coulomb overrides mean "use the per-material PP/PW value".
        # This keeps the legacy material table authoritative while exposing the
        # additional Stribeck/viscous parameters required by fully implicit
        # friction.
        self.affine_dynamic_friction = -1.0
        self.affine_static_friction = -1.0
        self.affine_viscous_friction = 0.0
        self.affine_stribeck_velocity = -1.0
        self.affine_friction_profile = "quadratic"
        # IPC lagged friction solves one frozen-friction problem by default,
        # matching the reference IPC ``fricIterAmt=1`` behaviour.  Values
        # greater than one enable the outer fixed-point update; ``-1`` means
        # iterate to tolerance, subject to the safety cap below.
        self.affine_friction_mode = "lagged"
        self.affine_friction_iterations = 1
        # Lagged nonlinear tolerances have velocity units (m/s), matching
        # reference IPC's ``max(abs(newton_direction)) / dt`` criterion.
        # The separate outer tolerance is a GeoTaichi control extension; its
        # default equals the reference solver's single ``tol`` value.
        self.affine_friction_tolerance = 1.0e-7
        self.affine_friction_max_iterations = 50
        self.affine_max_newton_iteration = 8
        self.affine_newton_tolerance = 1.0e-7
        self.affine_linear_tolerance = 1.0e-9
        self.affine_linear_max_iteration = 500
        # Reference lagged IPC backtracks until the conservative incremental
        # potential is non-increasing. Keep a generous finite safety cap;
        # fully implicit friction uses residual-merit Armijo globalization.
        self.affine_line_search_max_iteration = 50
        self.affine_direct_hessian_dofs = 240
        self.affine_finite_difference = 1.0e-6
        self.affine_hessian_shift = 1.0e-9
        # Optional regularization for the nonsymmetric fully implicit
        # residual Jacobian.  Zero preserves the exact paper Jacobian.
        self.affine_fully_implicit_jacobian_shift = 0.0
        # Fully implicit friction solves a nonconservative force residual.
        # Keep its force-unit stopping controls separate from the lagged
        # correction-velocity tolerance above.
        self.affine_fully_implicit_force_atol = 1.0e-12
        self.affine_fully_implicit_force_rtol = 1.0e-8
        self.affine_fully_implicit_armijo = 1.0e-4
        self.affine_fully_implicit_line_search_contraction = 0.5
        self.affine_max_step = 0.2
        self.affine_ccd = True
        self.affine_ccd_type = "ccd"
        self.affine_ccd_eta = 0.2
        self.affine_accd_tolerance = 1.0e-7
        self.affine_ccd_max_iteration = 10000
        self.affine_levelset_auto_initialize = False
        self.affine_levelset_initial_anchor_stiffness = 1.0
        self.affine_levelset_initial_continuation_ratio = 0.1
        self.affine_levelset_initial_gap_tolerance = 1.0e-8
        self.affine_levelset_initial_maximum_stages = 8
        self.affine_levelset_initial_stiffness_growth = 10.0
        self.affine_levelset_initial_maximum_iterations = 300
        self.lsmpm_soft_rigid_contact = "DEM"
        self.soft_background_damping = 0.0
        self.soft_affine_contact_triplet_budget = 4_000_000
        self.soft_affine_hash_triplet_capacity = 0
        self.soft_affine_soft_friction_capacity = None
        self.soft_affine_mixed_friction_capacity = None
        self.soft_affine_soft_contact_capacity = None
        self.soft_affine_mixed_contact_capacity = None
        self.soft_pic_fraction = 0.0
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

        self.dt = ti.field(float, shape=())
        self.delta = 0.0
        self.init_delta = 0.0
        self.current_time = 0.0
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
        self.max_soft_grid_num = 0
        self.max_material_point_num = 0
        self.max_soft_template_point_num = 0
        self.max_soft_template_surface_num = 0
        self.max_soft_template_sdf_num = 0
        self.max_soft_velocity_constraint_num = 0
        self.max_surface_node_num = 0
        # Fixed primitive-contact capacities for AffineBody triangle-mesh IPC.
        # They are independent of coarse body-pair coordination.
        self.max_point_triangle_pairs = 0
        self.max_edge_edge_pairs = 0
        self.max_ls_contact_node_num = 0
        self.max_rigid_template_num = 0
        self.max_wall_num = 0
        self.max_servo_wall_num = 0
        self.max_digital_elevation_grid_number = [0, 0]
        self.digital_elevation_contact_mode = "heightfield"
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
        self.servo_type = "StiffnessControl"

        self.body_coordination_number = 0
        self.wall_coordination_number = 0

        self.hierarchical_level = 1
        self.hierarchical_size = []

        self.verlet_distance = 0.0
        self.verlet_distance_multiplier = [0.0, 0.0]
        self.max_potential_particle_pairs = 0
        self.max_potential_wall_pairs = 0
        self.wall_per_cell = 0
        self.max_bounding_sphere_radius = 0.0
        self.min_bounding_sphere_radius = 0.0

        self.potential_particle_num = 0
        self.potential_contact_points_particle = 0
        self.potential_contact_points_wall = 0
        self.particle_contact_list_length = 0
        self.wall_contact_list_length = 0
        self.refit_number = 1000
        self.particle_particle_contact_model = None
        self.particle_wall_contact_model = None

        self.time = 0.0
        self.CFL = 0.2
        self.adaptive_timestep = 0
        self.visualize = True
        self.save_interval = 1e6
        self.path = None
        self.verlet_distance = 0.0
        self.point_verlet_distance = 0.0

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

        self.monitor_type = []
        self.save_data_selected = False

    def _soft_particle_backend(self):
        backend = getattr(self, "soft_particle_backend", None)
        if backend is None:
            raise RuntimeError(
                "SoftBody/LSMPM is an MPM soft-particle mode. Use MPDEM/MPM soft_particle so the coupling layer installs the backend."
            )
        return backend

    def get_simulation_domain(self):
        return self.domain

    def set_domain(self, domain):
        self.domain = domain
        if isinstance(domain, (list, tuple)):
            self.domain = vec3f(domain)

    def set_boundary(self, boundary):
        BOUNDARY = {None: -1, "Reflect": 0, "Destroy": 1, "Period": 2}
        self.boundary = [DictIO.GetEssential(BOUNDARY, b) for b in boundary]
        if self.boundary[0] == 2:
            self.xpbc = True
            GlobalVariable.DEMXPBC = True
            GlobalVariable.DEMXSIZE = self.domain[0]
        if self.boundary[1] == 2:
            self.ypbc = True
            GlobalVariable.DEMYPBC = True
            GlobalVariable.DEMYSIZE = self.domain[1]
        if self.dimension == 3:
            if self.boundary[2] == 2:
                self.zpbc = True
                GlobalVariable.DEMZPBC = True
                GlobalVariable.DEMZSIZE = self.domain[2]

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
        self.energy_tracking = track_energy
        GlobalVariable.TRACKENERGY = track_energy

    def set_enable_shell(self, enable_shell):
        self.enable_shell = enable_shell
        GlobalVariable.ENABLESHELL = enable_shell

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
        self._soft_particle_backend().set_soft_body_num(self, soft_body_num)

    def set_material_point_num(self, material_point_num):
        self._soft_particle_backend().set_material_point_num(self, material_point_num)

    def set_soft_grid_num(self, soft_grid_num):
        self._soft_particle_backend().set_soft_grid_num(self, soft_grid_num)

    def set_soft_velocity_constraint_num(self, constraint_num):
        self._soft_particle_backend().set_soft_velocity_constraint_num(self, constraint_num)

    def set_soft_template_support_num(self, point_num, surface_num, sdf_num):
        self._soft_particle_backend().set_soft_template_support_num(self, point_num, surface_num, sdf_num)

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
                if self.enable_shell:
                    raise RuntimeError("Polygon wall is not supported when enable_shell is True")
            elif not self.wall_type is None:
                self.raise_wall_error_info(curr_wall_type=1)

    def set_plane_num(self, plane_num):
        if plane_num > 0:
            if self.wall_type is None:
                self.max_wall_num = int(plane_num)
                self.wall_type = 0
                self.static_wall = True
                if self.enable_shell:
                    raise RuntimeError("Half space plane is not supported when enable_shell is True")
            elif not self.wall_type is None:
                self.raise_wall_error_info(curr_wall_type=0)

    def set_digital_elevation_grid_num(self, digital_elevation_grid_number):
        if isinstance(digital_elevation_grid_number, (int, float)):
            self.max_digital_elevation_grid_number = [
                int(digital_elevation_grid_number),
                int(digital_elevation_grid_number),
            ]
        elif isinstance(digital_elevation_grid_number, (list, tuple)):
            self.max_digital_elevation_grid_number = [int(i) for i in digital_elevation_grid_number]

        expect_cell_num = 1
        for i in self.max_digital_elevation_grid_number:
            expect_cell_num *= max(i - 1, 0)
        if self.max_wall_num > 2 * expect_cell_num:
            warnings.warn(
                f"Keyword:: /max_digital_elevation_facet_num/ {self.max_wall_num} is large enough, which may unnecessarily occupy more GPU memory"
            )

    def set_digital_elevation_facet_num(self, digital_elevation_facet_number):
        if digital_elevation_facet_number > 0:
            if self.wall_type is None:
                if self.digital_elevation_contact_mode == "heightfield":
                    self.max_wall_num = 1
                else:
                    self.max_wall_num = int(digital_elevation_facet_number)
                self.wall_type = 3
            elif not self.wall_type is None:
                self.raise_wall_error_info(curr_wall_type=3)

    def set_digital_elevation_contact_mode(self, contact_mode):
        if contact_mode is not None and str(contact_mode).lower() != "heightfield":
            warnings.warn("DEM DigitalElevation contact is forced to heightfield; the requested mode is ignored")
        self.digital_elevation_contact_mode = "heightfield"

    def use_digital_elevation_heightfield(self):
        return self.wall_type == 3 and self.digital_elevation_contact_mode == "heightfield"

    def set_dem_scheme(self, scheme):
        self.scheme = scheme

        valid = ["DEM", "LSDEM", "LSMPM", "PolySuperEllipsoid", "PolySuperQuadrics", "AffineBody"]
        if not scheme in valid:
            raise RuntimeError(f"Keyword:: /scheme/ error. Only the following {valid} is support")
        self.affine_body = scheme == "AffineBody"

    def set_soft_shape_function(self, shape_function):
        self._soft_particle_backend().set_soft_shape_function(self, shape_function)

    def set_soft_grid_storage(self, storage="Dense"):
        self._soft_particle_backend().set_soft_grid_storage(self, storage=storage)

    def set_soft_grid_type(self, grid_type="Hexahedron"):
        self._soft_particle_backend().set_soft_grid_type(self, grid_type=grid_type)

    def set_soft_mechanical_grid_spacing_ratio(self, spacing_ratio=0.15):
        self._soft_particle_backend().set_soft_mechanical_grid_spacing_ratio(self, spacing_ratio=spacing_ratio)

    def set_soft_pic_fraction(self, fraction):
        fraction = float(fraction)
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("LSMPM soft PIC fraction must lie in [0, 1]")
        self.soft_pic_fraction = fraction

    def set_soft_levelset_reinitialization(self, *args, **kwargs):
        self._soft_particle_backend().set_soft_levelset_reinitialization(self, *args, **kwargs)

    def freeze_affine_body_parameters(self):
        """Freeze settings captured by affine IPC operators at initialization."""
        self._affine_body_parameters_frozen = True

    def freeze_lsmpm_soft_rigid_contact(self):
        """Freeze the LSMPM soft--rigid engine route after it is selected."""
        self._lsmpm_soft_rigid_contact_frozen = True

    def set_lsmpm_soft_rigid_contact(self, contact):
        if getattr(self, "_lsmpm_soft_rigid_contact_frozen", False):
            raise RuntimeError(
                "LSMPM soft-rigid contact cannot be changed after engine "
                "initialization; configure it before add_essentials()"
            )
        key = str(contact).replace("_", "").replace("-", "").replace(" ", "").lower()
        aliases = {
            "dem": "DEM",
            "explicit": "DEM",
            "explicitdem": "DEM",
            "ipc": "IPC",
            "implicit": "IPC",
            "implicitipc": "IPC",
            "barrieripc": "IPC",
            "semi": "IPC",
            "semiipc": "IPC",
        }
        if key not in aliases:
            raise RuntimeError("LSMPM soft-rigid contact must be one of ['DEM', 'IPC']")
        self.lsmpm_soft_rigid_contact = aliases[key]
        if key in ("semi", "semiipc"):
            self.affine_contact_model = "SemiIPC"

    def set_affine_body_parameters(self, **kwargs):
        if getattr(self, "_affine_body_parameters_frozen", False):
            raise RuntimeError(
                "AffineBody IPC parameters cannot be changed after engine "
                "initialization; configure them before add_essentials()"
            )
        if "assemble_type" in kwargs:
            key = str(kwargs["assemble_type"]).replace("_", "").replace("-", "").lower()
            aliases = {
                "matrixfree": "MatrixFree",
                "coo": "COO",
                "csr": "COO",
                "coordinate": "COO",
                "coordinatematrix": "COO",
                "hashtriplet": "HashTriplet",
                "triplet": "HashTriplet",
            }
            if key not in aliases:
                raise RuntimeError("AffineBody assemble_type must be one of ['MatrixFree', 'COO', 'HashTriplet']")
            self.affine_assemble_type = aliases[key]
        if "YoungModulus" in kwargs:
            self.affine_young_modulus = float(kwargs["YoungModulus"])
        if "young_modulus" in kwargs:
            self.affine_young_modulus = float(kwargs["young_modulus"])
        if "dhat" in kwargs:
            self.affine_dhat = float(kwargs["dhat"])
        if "barrier_stiffness" in kwargs:
            self.affine_barrier_stiffness = float(kwargs["barrier_stiffness"])
        if "penalty" in kwargs:
            penalty = float(kwargs["penalty"])
            if not math.isfinite(penalty) or penalty <= 0.0:
                raise RuntimeError("AffineBody penalty must be finite and positive")
            self.affine_penalty = penalty
        if "constraint_tolerance" in kwargs:
            tolerance = float(kwargs["constraint_tolerance"])
            if not math.isfinite(tolerance) or tolerance < 0.0:
                raise RuntimeError("AffineBody constraint_tolerance must be finite and non-negative")
            self.affine_constraint_tolerance = tolerance
        for key in ("contact_model", "ipc_model", "contact_form"):
            if key in kwargs:
                from src.physics_model.contact_model.ipc.IPC import normalize_ipc_model

                self.affine_contact_model = normalize_ipc_model(kwargs[key])
        for key in ("contact_damping_stiffness", "contact_damping", "k_CD", "k_cd"):
            if key in kwargs:
                self.affine_contact_damping_stiffness = float(kwargs[key])
        for key in ("local_damping", "LocalDamping"):
            if key in kwargs:
                self.affine_force_local_damping = float(kwargs[key])
                self.affine_torque_local_damping = float(kwargs[key])
        for key in ("force_local_damping", "ForceLocalDamping"):
            if key in kwargs:
                self.affine_force_local_damping = float(kwargs[key])
        for key in ("torque_local_damping", "TorqueLocalDamping"):
            if key in kwargs:
                self.affine_torque_local_damping = float(kwargs[key])
        if "friction_epsv" in kwargs:
            epsv = float(kwargs["friction_epsv"])
            if not math.isfinite(epsv) or epsv <= 0.0:
                raise RuntimeError("AffineBody friction_epsv must be finite and positive")
            self.affine_friction_epsv = epsv
        for key, attribute in (
            ("dynamic_friction", "affine_dynamic_friction"),
            ("static_friction", "affine_static_friction"),
            ("viscous_friction", "affine_viscous_friction"),
            ("stribeck_velocity", "affine_stribeck_velocity"),
        ):
            if key in kwargs:
                value = float(kwargs[key])
                accepts_fallback = key in (
                    "dynamic_friction",
                    "static_friction",
                    "stribeck_velocity",
                )
                if not math.isfinite(value) or (value < 0.0 and not (accepts_fallback and value == -1.0)):
                    suffix = " or -1" if accepts_fallback else ""
                    raise RuntimeError(f"AffineBody {key} must be finite and non-negative{suffix}")
                setattr(self, attribute, value)
        if "friction_profile" in kwargs:
            profile = str(kwargs["friction_profile"]).strip().replace("-", "_").replace(" ", "_").lower()
            aliases = {
                "quadratic": "quadratic",
                "c1": "quadratic",
                "ipc": "quadratic",
                "stabilized": "stabilized",
                "stabilised": "stabilized",
                "cinfinity": "stabilized",
                "c_infinity": "stabilized",
            }
            if profile not in aliases:
                raise RuntimeError("AffineBody friction_profile must be 'quadratic' or 'stabilized'")
            self.affine_friction_profile = aliases[profile]
        if (
            self.affine_static_friction >= 0.0
            and (self.affine_dynamic_friction < 0.0 or self.affine_static_friction != self.affine_dynamic_friction)
            and self.affine_stribeck_velocity == 0.0
        ):
            raise RuntimeError(
                "AffineBody stribeck_velocity must be positive when " "static_friction differs from dynamic_friction"
            )
        if "friction_mode" in kwargs:
            mode = str(kwargs["friction_mode"]).strip().replace("-", "_").lower()
            aliases = {
                "lagged": "lagged",
                "lag": "lagged",
                "fully_implicit": "fully_implicit",
                "fullyimplicit": "fully_implicit",
            }
            if mode not in aliases:
                raise RuntimeError("AffineBody friction_mode must be one of " "['lagged', 'fully_implicit']")
            self.affine_friction_mode = aliases[mode]
        for key in ("friction_iterations", "friction_fixed_point_iterations"):
            if key in kwargs:
                try:
                    numeric_iterations = float(kwargs[key])
                    iterations = int(numeric_iterations)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError("AffineBody friction_iterations must be an integer") from exc
                if not math.isfinite(numeric_iterations) or numeric_iterations != iterations:
                    raise RuntimeError("AffineBody friction_iterations must be an integer")
                self.affine_friction_iterations = -1 if iterations <= 0 else iterations
        for key in ("friction_tolerance", "friction_fixed_point_tolerance"):
            if key in kwargs:
                try:
                    tolerance = float(kwargs[key])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError("AffineBody friction_tolerance must be finite and positive") from exc
                if not math.isfinite(tolerance) or tolerance <= 0.0:
                    raise RuntimeError("AffineBody friction_tolerance must be positive")
                self.affine_friction_tolerance = tolerance
        if "friction_max_iterations" in kwargs:
            try:
                numeric_max_iterations = float(kwargs["friction_max_iterations"])
                max_iterations = int(numeric_max_iterations)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("AffineBody friction_max_iterations must be a positive integer") from exc
            if (
                not math.isfinite(numeric_max_iterations)
                or numeric_max_iterations != max_iterations
                or max_iterations <= 0
            ):
                raise RuntimeError("AffineBody friction_max_iterations must be a positive integer")
            self.affine_friction_max_iterations = max_iterations
        if "max_newton_iteration" in kwargs:
            try:
                numeric_iterations = float(kwargs["max_newton_iteration"])
                iterations = int(numeric_iterations)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("AffineBody max_newton_iteration must be a positive integer") from exc
            if not math.isfinite(numeric_iterations) or numeric_iterations != iterations or iterations <= 0:
                raise RuntimeError("AffineBody max_newton_iteration must be a positive integer")
            self.affine_max_newton_iteration = iterations
        if "newton_tolerance" in kwargs:
            tolerance = float(kwargs["newton_tolerance"])
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise RuntimeError("AffineBody newton_tolerance must be finite and positive " "(in m/s)")
            self.affine_newton_tolerance = tolerance
        if "linear_tolerance" in kwargs:
            tolerance = float(kwargs["linear_tolerance"])
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise RuntimeError("AffineBody linear_tolerance must be finite and positive")
            self.affine_linear_tolerance = tolerance
        if "linear_max_iteration" in kwargs:
            try:
                numeric_iterations = float(kwargs["linear_max_iteration"])
                iterations = int(numeric_iterations)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("AffineBody linear_max_iteration must be a positive integer") from exc
            if not math.isfinite(numeric_iterations) or numeric_iterations != iterations or iterations <= 0:
                raise RuntimeError("AffineBody linear_max_iteration must be a positive integer")
            self.affine_linear_max_iteration = iterations
        if "line_search_max_iteration" in kwargs:
            try:
                numeric_iterations = float(kwargs["line_search_max_iteration"])
                iterations = int(numeric_iterations)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("AffineBody line_search_max_iteration must be a " "positive integer") from exc
            if not math.isfinite(numeric_iterations) or numeric_iterations != iterations or iterations <= 0:
                raise RuntimeError("AffineBody line_search_max_iteration must be a " "positive integer")
            self.affine_line_search_max_iteration = iterations
        if "direct_hessian_dofs" in kwargs:
            try:
                numeric_dofs = float(kwargs["direct_hessian_dofs"])
                direct_dofs = int(numeric_dofs)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("AffineBody direct_hessian_dofs must be a non-negative integer") from exc
            if not math.isfinite(numeric_dofs) or numeric_dofs != direct_dofs or direct_dofs < 0:
                raise RuntimeError("AffineBody direct_hessian_dofs must be a non-negative integer")
            self.affine_direct_hessian_dofs = direct_dofs
        if "finite_difference" in kwargs:
            self.affine_finite_difference = float(kwargs["finite_difference"])
        if "hessian_shift" in kwargs:
            self.affine_hessian_shift = float(kwargs["hessian_shift"])
        if "fully_implicit_jacobian_shift" in kwargs:
            shift = float(kwargs["fully_implicit_jacobian_shift"])
            if not math.isfinite(shift) or shift < 0.0:
                raise RuntimeError("AffineBody fully_implicit_jacobian_shift must be " "finite and non-negative")
            self.affine_fully_implicit_jacobian_shift = shift
        if "fully_implicit_force_atol" in kwargs:
            tolerance = float(kwargs["fully_implicit_force_atol"])
            if not math.isfinite(tolerance) or tolerance < 0.0:
                raise RuntimeError("AffineBody fully_implicit_force_atol must be finite and " "non-negative")
            self.affine_fully_implicit_force_atol = tolerance
        if "fully_implicit_force_rtol" in kwargs:
            tolerance = float(kwargs["fully_implicit_force_rtol"])
            if not math.isfinite(tolerance) or tolerance < 0.0:
                raise RuntimeError("AffineBody fully_implicit_force_rtol must be finite and " "non-negative")
            self.affine_fully_implicit_force_rtol = tolerance
        if "fully_implicit_armijo" in kwargs:
            armijo = float(kwargs["fully_implicit_armijo"])
            if not 0.0 < armijo < 1.0:
                raise RuntimeError("AffineBody fully_implicit_armijo must lie in (0, 1)")
            self.affine_fully_implicit_armijo = armijo
        if "fully_implicit_line_search_contraction" in kwargs:
            contraction = float(kwargs["fully_implicit_line_search_contraction"])
            if not 0.0 < contraction < 1.0:
                raise RuntimeError("AffineBody fully_implicit_line_search_contraction must lie in (0, 1)")
            self.affine_fully_implicit_line_search_contraction = contraction
        if "max_step" in kwargs:
            self.affine_max_step = float(kwargs["max_step"])
        if "ccd" in kwargs:
            self.affine_ccd = bool(kwargs["ccd"])
        if "ccd_type" in kwargs:
            ccd_type = str(kwargs["ccd_type"]).lower()
            valid = ["none", "off", "ccd", "accd"]
            if ccd_type not in valid:
                raise RuntimeError(f"Keyword:: /ccd_type/ is wrong, Only the following is valid: {valid}")
            self.affine_ccd_type = ccd_type
            self.affine_ccd = ccd_type not in ["none", "off"]
        if "ccd_eta" in kwargs:
            self.affine_ccd_eta = float(kwargs["ccd_eta"])
        if "accd_tolerance" in kwargs:
            self.affine_accd_tolerance = float(kwargs["accd_tolerance"])
        if "ccd_max_iteration" in kwargs:
            self.affine_ccd_max_iteration = int(kwargs["ccd_max_iteration"])
        for key in (
            "levelset_auto_initialize",
            "levelset_initialization",
            "resolve_initial_overlap",
        ):
            if key in kwargs:
                self.affine_levelset_auto_initialize = bool(kwargs[key])
        scalar_initialization_parameters = (
            (
                "levelset_initial_anchor_stiffness",
                "affine_levelset_initial_anchor_stiffness",
                True,
            ),
            (
                "levelset_initial_continuation_ratio",
                "affine_levelset_initial_continuation_ratio",
                True,
            ),
            (
                "levelset_initial_gap_tolerance",
                "affine_levelset_initial_gap_tolerance",
                True,
            ),
            (
                "levelset_initial_stiffness_growth",
                "affine_levelset_initial_stiffness_growth",
                True,
            ),
        )
        for key, attribute, positive in scalar_initialization_parameters:
            if key in kwargs:
                value = float(kwargs[key])
                if not math.isfinite(value) or (positive and value <= 0.0):
                    raise RuntimeError(f"AffineBody {key} must be finite and positive")
                setattr(self, attribute, value)
        continuation_ratio = getattr(self, "affine_levelset_initial_continuation_ratio", 0.1)
        if not (0.0 < continuation_ratio < 1.0):
            raise RuntimeError("AffineBody levelset_initial_continuation_ratio must lie " "in (0, 1)")
        stiffness_growth = getattr(self, "affine_levelset_initial_stiffness_growth", 10.0)
        if stiffness_growth <= 1.0:
            raise RuntimeError("AffineBody levelset_initial_stiffness_growth must be " "larger than one")
        for key, attribute in (
            (
                "levelset_initial_maximum_stages",
                "affine_levelset_initial_maximum_stages",
            ),
            (
                "levelset_initial_maximum_iterations",
                "affine_levelset_initial_maximum_iterations",
            ),
        ):
            if key in kwargs:
                try:
                    numeric = float(kwargs[key])
                    value = int(numeric) if math.isfinite(numeric) else 0
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(f"AffineBody {key} must be a positive integer") from exc
                if not math.isfinite(numeric) or numeric != value or value <= 0:
                    raise RuntimeError(f"AffineBody {key} must be a positive integer")
                setattr(self, attribute, value)
        for key in ("soft_background_damping", "background_damping", "BackgroundDamping"):
            if key in kwargs:
                self.soft_background_damping = float(kwargs[key])
        for key, attribute in (
            ("contact_triplet_budget", "soft_affine_contact_triplet_budget"),
            ("hash_triplet_capacity", "soft_affine_hash_triplet_capacity"),
            ("soft_friction_capacity", "soft_affine_soft_friction_capacity"),
            ("mixed_friction_capacity", "soft_affine_mixed_friction_capacity"),
            ("soft_contact_capacity", "soft_affine_soft_contact_capacity"),
            ("mixed_contact_capacity", "soft_affine_mixed_contact_capacity"),
        ):
            if key in kwargs:
                try:
                    numeric = float(kwargs[key])
                    value = int(numeric) if math.isfinite(numeric) else 0
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(f"AffineBody {key} must be a positive integer") from exc
                if not math.isfinite(numeric) or numeric != value or value <= 0:
                    raise RuntimeError(f"AffineBody {key} must be a positive integer")
                setattr(self, attribute, value)

    def set_visualize(self, visualize):
        self.visualize = visualize

    def set_dem_coupling(self, coupling):
        self.coupling = coupling

    def set_static_wall(self, static_wall):
        self.static_wall = static_wall

    def update_servo_status(self, status):
        valid = ["On", "Off", "StiffnessControl", "GainControl"]
        if not status in valid:
            raise RuntimeError(f"Keyword:: /status/ error. Only the following {valid} is support")

        if status != "Off":
            self.static_wall = False
            self.servo_status = "On"
            if status == "On":
                self.servo_type = "StiffnessControl"
            else:
                self.servo_type = status
        else:
            self.servo_status = status

    def set_body_coordination_number(self, body_coordination_number):
        if self.search == "HierarchicalLinkedCell":
            if isinstance(body_coordination_number, (int, float)):
                self.body_coordination_number = [
                    int(max(body_coordination_number, 1)) for _ in range(self.hierarchical_level)
                ]
            else:
                self.body_coordination_number = [int(max(i, 1)) for i in body_coordination_number]

            if len(self.body_coordination_number) != self.hierarchical_level:
                raise RuntimeError(
                    f"Keyword:: /body_coordination_number/ should have a size of {self.hierarchical_level}"
                )
        else:
            self.body_coordination_number = max(body_coordination_number, 1)

    def set_affine_primitive_pair_capacity(self, point_triangle_pairs, edge_edge_pairs):
        self.max_point_triangle_pairs = int(point_triangle_pairs)
        self.max_edge_edge_pairs = int(edge_edge_pairs)
        if self.max_point_triangle_pairs < 0:
            raise ValueError("AffineBody max_point_triangle_pairs must be non-negative")
        if self.max_edge_edge_pairs < 0:
            raise ValueError("AffineBody max_edge_edge_pairs must be non-negative")

    def set_affine_contact_block_capacity(self, capacity):
        numeric_capacity = float(capacity)
        block_capacity = int(numeric_capacity)
        if not math.isfinite(numeric_capacity) or numeric_capacity != block_capacity or block_capacity <= 0:
            raise ValueError("AffineBody contact block capacity must be a positive integer")
        self.affine_contact_block_capacity = block_capacity

    def set_wall_coordination_number(self, wall_coordination_number):
        if self.search == "HierarchicalLinkedCell":
            if isinstance(wall_coordination_number, (float, int)):
                self.wall_coordination_number = [int(wall_coordination_number) for _ in range(self.hierarchical_level)]
            else:
                self.wall_coordination_number = wall_coordination_number

            if len(self.wall_coordination_number) != self.hierarchical_level:
                raise RuntimeError(
                    f"Keyword:: /wall_coordination_number/ should have a size of {self.hierarchical_level}"
                )
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
        if self.max_wall_num == 0:
            model = None
        self.particle_wall_contact_model = model

    def set_save_data(self, particle, sphere, clump, surface, grid, bounding, wall, ppcontact, pwcontact):
        self.save_data_selected = True
        if particle:
            self.monitor_type.append("particle")
        if sphere:
            self.monitor_type.append("sphere")
        if clump:
            self.monitor_type.append("clump")
        if surface:
            self.monitor_type.append("surface")
        if grid:
            self.monitor_type.append("grid")
        if bounding:
            self.monitor_type.append("bounding")
        if wall:
            self.monitor_type.append("wall")
        if ppcontact:
            self.monitor_type.append("ppcontact")
        if pwcontact:
            self.monitor_type.append("pwcontact")

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
            self.bvh_rebuild_interval = int(rebuild_interval)
            self.refit_number = max(1, self.bvh_rebuild_interval)

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

    def set_adaptive_timestep(self, adaptive_timestep):
        self.adaptive_timestep = int(adaptive_timestep)

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
                    self.compaction_ratio = [
                        compaction_ratio[0],
                        compaction_ratio[1],
                        compaction_ratio[0],
                        compaction_ratio[1],
                    ]
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
        self.look_at = DictIO.GetAlternative(
            windows, "LookAt", (0.7 * self.domain[0], -1.5 * self.domain[1], 0.4 * self.domain[2])
        )
        self.look_from = DictIO.GetAlternative(
            windows, "LookFrom", (0.7 * self.domain[0], -1.5 * self.domain[1], 0.5 * self.domain[2])
        )
        self.particle_color = DictIO.GetAlternative(windows, "ParticleColor", (1, 1, 1))
        self.background_color = DictIO.GetAlternative(windows, "BackgroundColor", (0, 0, 0))
        self.point_light = DictIO.GetAlternative(
            windows, "PointLight", (0.5 * self.domain[0], 0.5 * self.domain[1], 1.0 * self.domain[2])
        )
        self.view_angle = DictIO.GetAlternative(windows, "ViewAngle", 70.0)
        self.move_velocity = DictIO.GetAlternative(
            windows, "MoveVelocity", 0.01 * (self.domain[0] + self.domain[1] + self.domain[2])
        )

    def set_save_path(self, path):
        self.path = path

    def define_work_load(self):
        # Restart, persistent history, and energy output share the compact
        # contact-list representation, so both contact paths use work mode 2.
        self.particle_work = 2
        self.wall_work = 2

    def set_verlet_distance(self, rad_min):
        if self.verlet_distance < 1e-15:
            self.verlet_distance = max(self.verlet_distance, self.verlet_distance_multiplier[0] * rad_min)

    def set_point_verlet_distance(self, rad_min):
        if self.point_verlet_distance < 1e-15:
            self.point_verlet_distance = self.verlet_distance_multiplier[1] * rad_min

    def check_grid_extent(self, pid, extent):
        if pid != -1:
            raise RuntimeError(f"Keyword:: /extent/ is not large enough, Particle {pid} need extent = {extent}")

    def compute_potential_ratios(self, rad_max):
        if rad_max <= Threshold:
            return 0.0
        return ((3 * rad_max + 2 * self.verlet_distance) ** 3 - rad_max**3) / (26.0 * rad_max**3)

    def update_hierarchical_size(self, rad_max):
        self.hierarchical_size[-1] = max(rad_max, self.hierarchical_size[-1])

    def set_potential_list_size(self, rad_max):
        self.potential_particle_num = 0
        self.max_potential_particle_pairs = 0
        if self.max_particle_num > 0 and rad_max > Threshold:
            potential_particle_ratio = self.compute_potential_ratios(rad_max)
            self.potential_particle_num = math.ceil(
                potential_particle_ratio * self.body_coordination_number
            )  # next_pow2(int(potential_particle_ratio * self.body_coordination_number))
            self.max_potential_particle_pairs = math.ceil(self.potential_particle_num * self.max_particle_num)
        if self.max_wall_num > 0 and self.use_digital_elevation_heightfield():
            self.max_potential_wall_pairs = 0
        elif self.max_wall_num > 0:
            self.max_potential_wall_pairs = math.ceil(self.wall_coordination_number * self.max_particle_num)
        self.set_levelset_contact_list_size()
        self.set_contact_list_size()

    def set_contact_list_size(self):
        if self.scheme == "LSDEM" or self.scheme == "LSMPM":
            level_body_num = max(self.max_particle_num, self.max_rigid_body_num + self.max_soft_body_num)
            self.particle_verlet_length = int(math.ceil(self.compaction_ratio[0] * self.max_potential_particle_pairs))
            self.wall_verlet_length = (
                0
                if self.use_digital_elevation_heightfield()
                else int(math.ceil(self.compaction_ratio[1] * self.max_potential_wall_pairs))
            )
            if self.scheme == "LSMPM":
                self.particle_contact_list_length, self.wall_contact_list_length = (
                    self.soft_particle_contact_list_length()
                )
            else:
                self.particle_contact_list_length = int(
                    math.ceil(self.compaction_ratio[2] * self.potential_contact_points_particle * level_body_num)
                )
                self.wall_contact_list_length = (
                    0
                    if self.use_digital_elevation_heightfield()
                    else int(math.ceil(self.compaction_ratio[3] * self.potential_contact_points_wall * level_body_num))
                )
        else:
            self.particle_contact_list_length = int(
                math.ceil(self.compaction_ratio[0] * self.max_potential_particle_pairs)
            )
            self.wall_contact_list_length = (
                0
                if self.use_digital_elevation_heightfield()
                else int(math.ceil(self.compaction_ratio[1] * self.max_potential_wall_pairs))
            )

    def set_levelset_contact_list_size(self):
        if self.scheme == "LSMPM":
            self._soft_particle_backend().set_levelset_contact_list_size(self)
            return
        self.max_ls_contact_node_num = self.max_surface_node_num * self.max_particle_num
        self.potential_contact_points_particle = int(
            self.point_particle_coordination_number * self.max_surface_node_num
        )
        self.potential_contact_points_wall = int(self.point_wall_coordination_number * self.max_surface_node_num)

    def soft_particle_contact_list_length(self):
        return self._soft_particle_backend().soft_particle_contact_list_length(self)

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
        if self.scheme == "LSDEM" or self.scheme == "LSMPM":
            if penetration_depth > self.point_verlet_distance:
                raise RuntimeError(
                    f"Keyword:: /verlet_distance_multiplier[1]/ should larger, at least {penetration_depth / self.point_verlet_distance * self.verlet_distance_multiplier[1]}"
                )

    def validate_configuration(self, require_memory=False):
        if self.enable_shell and self.wall_type in (0, 1):
            raise RuntimeError("enable_shell=True is incompatible with plane and polygon walls")
        if self.scheme == "LSMPM":
            self._soft_particle_backend().validate_soft_particle_configuration(self, require_memory)
        if self.max_servo_wall_num > 0 and self.wall_type != 1:
            raise RuntimeError("Servo walls require polygon/facet wall_type")
        if require_memory and self.servo_status == "On" and self.max_servo_wall_num <= 0:
            raise RuntimeError("Servo wall is enabled but max_servo_wall_number is zero")
        if (
            require_memory
            and self.search == "HierarchicalLinkedCell"
            and len(self.hierarchical_size) != self.hierarchical_level
        ):
            raise RuntimeError(
                f"HierarchicalLinkedCell requires hierarchical_size with " f"{self.hierarchical_level} entries"
            )
