"""Public IGA--MPM engine composed from focused responsibility mixins."""

import numpy as np
import taichi as ti

import src.igampm.config as config
import src.mpm.config as mpm_config
from .ContactEngine import ContactEngineMixin
from .FrictionEngine import FrictionEngineMixin
from .ImplicitEngine import ImplicitEngineMixin
from src.iga.engines.ImplicitIGA import ImplicitIGA
from src.igampm.ContactManager import ContactManager
from src.igampm.contact.ContactSurface import CouplingContactSurface
from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix
from src.mpm.engines.direct.ImplicitMPM import ImplicitMPM
from src.physics_model.contact_model.ipc.NurbsContact import PointNurbsDerivative
from src.utils.StepRetry import StepRetryPolicy
from src.utils.SolverRuntime import StepSchedule
from src.utils.TimeTicker import Timer
from src.utils.linalg import no_operation
from src.utils.PrefixSum import PrefixSumExecutor


@ti.data_oriented
class Engine(
    FrictionEngineMixin,
    ImplicitEngineMixin,
    ContactEngineMixin,
):
    def __init__(self, iga: ImplicitIGA, mpm: ImplicitMPM, contactor: ContactManager = None, **kwargs):
        self.iga = iga
        self.mpm = mpm
        self.is_axisymmetric = bool(self.iga.is_axisymmetric)
        if self.is_axisymmetric != bool(self.mpm.is_axisymmetric):
            raise ValueError("IGA-MPM requires matching planar or axisymmetric child modes")
        self.axis_offset = float(self.iga.axis_offset)
        mpm_config.set_dimension(config.DIM)
        self.contactor = contactor if contactor is not None else ContactManager(**kwargs)
        if self.contactor.contact_model != "IPC":
            raise ValueError("IGA-MPM Engine supports only implicit IPC contact")
        self.barrier = self.contactor.barrier
        self.is_semi = self.barrier.is_semi
        self.friction = self.contactor.friction
        self.derivative = PointNurbsDerivative()
        self.activate_barrier = self.contactor.activate_barrier
        self.activate_fric = self.contactor.activate_friction
        self.friction_mode = self.contactor.friction_mode
        if self.is_semi and self.friction_mode != "lagged":
            raise ValueError("IGA-MPM SemiIPC currently requires friction_mode='lagged'")
        self.project_lagged_hessians = self.friction_mode == "lagged" and not self.is_semi
        self.friction_iterations = self.contactor.friction_iterations
        self.friction_tolerance = self.contactor.friction_tolerance
        self.friction_max_iterations = self.contactor.friction_max_iterations
        self.last_friction_iterations = 0
        self.last_friction_residual = np.inf
        self.last_friction_converged = False
        self.strict_feasibility_tolerance = float(kwargs.get("strict_feasibility_tolerance", 1.0e-14))
        if self.strict_feasibility_tolerance < 0.0:
            raise ValueError("strict_feasibility_tolerance must be non-negative")
        self.contact_ccd_safety = float(kwargs.get("contact_ccd_safety", 0.9))
        self.contact_ccd_min_step = float(kwargs.get("contact_ccd_min_step", 1.0e-12))
        self.contact_ccd_max_iterations = int(kwargs.get("contact_ccd_max_iterations", 100))
        self.armijo_c1 = float(kwargs.get("armijo_c1", 1.0e-4))
        self.armijo_max_backtracks = int(kwargs.get("armijo_max_backtracks", 40))
        self.line_search_energy_rtol = float(kwargs.get("line_search_energy_rtol", 1.0e-12))
        self.line_search_energy_atol = float(kwargs.get("line_search_energy_atol", 1.0e-14))
        self.monolithic_max_iterations = int(
            kwargs.get(
                "monolithic_max_iterations",
                min(
                    int(self.iga.max_iters),
                    int(self.mpm.max_iters),
                ),
            )
        )
        self.monolithic_tolerance = float(
            kwargs.get(
                "monolithic_tolerance",
                min(
                    float(self.iga.tol),
                    float(self.mpm.tol),
                ),
            )
        )
        self.monolithic_force_atol = float(kwargs.get("monolithic_force_atol", 1.0e-10))
        self.monolithic_force_rtol = float(kwargs.get("monolithic_force_rtol", 5.0e-4))
        self.monolithic_dirichlet_tolerance = float(kwargs.get("monolithic_dirichlet_tolerance", 1.0e-12))
        self.step_retry = StepRetryPolicy(
            enabled=kwargs.get("enable_step_retry", False),
            maximum_retries=kwargs.get("step_retry_max_retries", 2),
            reduction=kwargs.get("step_retry_reduction", 0.5),
            minimum_timestep=kwargs.get("step_retry_minimum_timestep", 0.0),
        )
        if self.is_semi and self.step_retry.enabled:
            raise ValueError(
                "IGA-MPM step retry does not support SemiIPC because its multipliers advance inside Newton iterations"
            )
        self.time = 0.0
        self.history = []
        self.last_step_record = None
        self.track_energy = bool(kwargs.get("track_energy", False))
        self.add_implicit_energy_record = (
            self._add_implicit_energy_record
            if self.track_energy and self.friction_mode != "fully_implicit"
            else no_operation
        )
        if self.activate_fric and self.friction_mode == "fully_implicit":
            self.solve_configured_implicit_system = self._solve_configured_fully_implicit_system
        elif self.activate_fric:
            self.solve_configured_implicit_system = self._solve_configured_lagged_friction_system
        else:
            self.solve_configured_implicit_system = self._solve_configured_conservative_system
        self.output_interval = max(
            1,
            min(
                int(self.iga.output_interval),
                int(self.mpm.output_interval),
            ),
        )
        self.step_schedule = StepSchedule.from_options(
            kwargs,
            output_interval=self.output_interval,
        )
        self.timer = Timer()
        self.compile_seconds = None
        self.last_failure = None
        if not 0.0 < self.contact_ccd_safety < 1.0:
            raise ValueError("contact_ccd_safety must lie strictly between 0 and 1")
        if self.contact_ccd_min_step <= 0.0:
            raise ValueError("contact_ccd_min_step must be positive")
        if self.contact_ccd_max_iterations <= 0:
            raise ValueError("contact_ccd_max_iterations must be positive")
        if not 0.0 < self.armijo_c1 < 1.0:
            raise ValueError("armijo_c1 must lie strictly between 0 and 1")
        if self.armijo_max_backtracks <= 0:
            raise ValueError("armijo_max_backtracks must be positive")
        if (
            not np.isfinite(self.line_search_energy_rtol)
            or self.line_search_energy_rtol < 0.0
            or not np.isfinite(self.line_search_energy_atol)
            or self.line_search_energy_atol < 0.0
        ):
            raise ValueError("line-search energy tolerances must be finite and non-negative")
        if self.monolithic_max_iterations <= 0:
            raise ValueError("monolithic_max_iterations must be positive")
        if self.monolithic_tolerance < 0.0:
            raise ValueError("monolithic_tolerance must be non-negative")
        if (
            not np.isfinite(self.monolithic_force_atol)
            or self.monolithic_force_atol < 0.0
            or not np.isfinite(self.monolithic_force_rtol)
            or self.monolithic_force_rtol < 0.0
            or not np.isfinite(self.monolithic_dirichlet_tolerance)
            or self.monolithic_dirichlet_tolerance < 0.0
        ):
            raise ValueError("monolithic force/Dirichlet tolerances must be finite and non-negative")
        self.last_contact_ccd_step = 1.0
        self.last_contact_ccd_min_distance = np.inf
        self.last_armijo_step = 0.0
        self.last_armijo_backtracks = 0
        self.last_monolithic_iterations = 0
        self.last_monolithic_residual = np.inf
        self.last_monolithic_force_residual = np.inf
        self.last_monolithic_dirichlet_residual = np.inf
        self.last_monolithic_converged = False
        if self.mpm.configuration not in ("TL", "UL"):
            raise ValueError("IGA-MPM Direct MPM configuration must be TL or UL")
        self.total_lagrangian_mpm = self.mpm.configuration == "TL"
        self.prepare_mpm_step = (
            self._prepare_total_lagrangian_mpm_step
            if self.total_lagrangian_mpm
            else self._prepare_updated_lagrangian_mpm_step
        )
        self.compute_mpm_traction = self.mpm.traction_p2g if bool(self.mpm.compute_traction) else no_operation
        self.compute_dynamic_mass_list = self._compute_dynamic_mass_list if mpm_config.DYNAMIC else no_operation
        self.assemble_mpm_mass_matrix = self.mpm.assemble_mass_matrix_hash if mpm_config.DYNAMIC else no_operation
        self.apply_mpm_neumann = self.mpm.apply_neumann if self.mpm.neumann.num > 0 else no_operation
        self.iga_material_ccd = self.iga.ccd
        self.mpm_material_ccd = self._mpm_material_ccd
        self.fully_implicit_force_atol = float(kwargs.get("fully_implicit_force_atol", 1.0e-12))
        self.fully_implicit_force_rtol = float(kwargs.get("fully_implicit_force_rtol", 1.0e-8))
        self.fully_implicit_dirichlet_atol = float(kwargs.get("fully_implicit_dirichlet_atol", 1.0e-12))
        self.fully_implicit_armijo_reduction = float(kwargs.get("fully_implicit_armijo_reduction", 0.5))
        self.fully_implicit_velocity_predictor = bool(kwargs.get("fully_implicit_velocity_predictor", True))
        if (
            not np.isfinite(self.fully_implicit_force_atol)
            or self.fully_implicit_force_atol < 0.0
            or not np.isfinite(self.fully_implicit_force_rtol)
            or self.fully_implicit_force_rtol < 0.0
            or not np.isfinite(self.fully_implicit_dirichlet_atol)
            or self.fully_implicit_dirichlet_atol < 0.0
        ):
            raise ValueError("fully implicit force and Dirichlet tolerances must be finite " "and non-negative")
        if not 0.0 < self.fully_implicit_armijo_reduction < 1.0:
            raise ValueError("fully_implicit_armijo_reduction must lie strictly between 0 and 1")
        self.mpm.build_surface_node(all_particles=bool(kwargs.get("contact_all_mpm_particles", False)))
        self.contact_surface = CouplingContactSurface(self.iga, **kwargs)

        contact_dtype = ti.types.struct(
            active=ti.i32,
            surface_id=ti.i32,
            sample_id=ti.i32,
            particle_id=ti.i32,
            distance=ti.f64,
            knot_value=ti.types.vector(2, ti.f64),
        )
        friction_contact_dtype = ti.types.struct(
            active=ti.i32,
            surface_id=ti.i32,
            sample_id=ti.i32,
            particle_id=ti.i32,
            mu_lambda=ti.f64,
            normal=ti.types.vector(config.DIM, ti.f64),
            knot_value=ti.types.vector(2, ti.f64),
        )
        # IPC keeps every active point--boundary-primitive pair.  Retaining
        # only the nearest IGA face drops constraints at patch/solid corners.
        self.contact_capacity = max(
            1,
            self.mpm.total_surface_num * max(1, self.contact_surface.num_surfaces),
        )
        self.contacts = contact_dtype.field(shape=self.contact_capacity)
        self.contact_point_direction = ti.Vector.field(config.DIM, ti.f64, shape=max(1, self.mpm.total_surface_num))
        # Seeds are acceleration hints, not accepted contact state. Any finite
        # parameter remains valid after sample remapping or step rollback.
        self.contact_projection_seed = (
            ti.Vector.field(2, ti.f64, shape=self.contact_capacity) if config.DIM == 3 else None
        )
        self.semi_multiplier = ti.field(ti.f64, shape=self.contact_capacity)
        self.semi_constraint_violation = ti.field(ti.f64, shape=())
        self.friction_contacts = friction_contact_dtype.field(shape=self.contact_capacity)
        self.contact_active_buffer = ti.field(ti.i32, shape=self.contact_capacity)
        self.contact_surface_id_buffer = ti.field(ti.i32, shape=self.contact_capacity)
        self.contact_sample_id_buffer = ti.field(ti.i32, shape=self.contact_capacity)
        self.contact_particle_id_buffer = ti.field(ti.i32, shape=self.contact_capacity)
        self.contact_distance_buffer = ti.field(ti.f64, shape=self.contact_capacity)
        self.contact_knot_value_buffer = ti.Vector.field(2, ti.f64, shape=self.contact_capacity)
        self.curr_barrier_contact_num = 0
        self.curr_friction_contact_num = 0
        self.contact_num = ti.field(ti.i32, shape=1)
        self.friction_contact_num = ti.field(ti.i32, shape=1)

        contact_ctrlpts = int(self.contact_surface.max_support_size)
        contact_mpm_nodes = int(
            getattr(
                self.mpm.shape_func,
                "max_node_per_particle",
                self.mpm.shape_func.influenced_node,
            )
        )
        per_contact_blocks = contact_ctrlpts + contact_mpm_nodes
        # Every possible point--NURBS contact owns one fixed square block
        # stencil.  Matrix kernels index this stencil directly instead of
        # atomically appending triplets, so an unchanged topology has exactly
        # the same raw order on device and can reuse HashReduction's pattern.
        self.contact_ctrlpts_capacity = contact_ctrlpts
        self.contact_mpm_nodes_capacity = contact_mpm_nodes
        self.contact_stencil_capacity = per_contact_blocks
        self.contact_pair_capacity = per_contact_blocks * per_contact_blocks
        self.contact_pair_count = int(self.mpm.total_surface_num * self.contact_surface.num_surfaces)
        self.compact_contact_slots = kwargs.get("compact_contact_slots", False)
        if not isinstance(self.compact_contact_slots, (bool, np.bool_)):
            raise TypeError("compact_contact_slots must be a boolean")
        if self.compact_contact_slots:
            self.contact_slot_prefix = PrefixSumExecutor(self.contact_pair_count)
            slot_length = self.contact_slot_prefix.get_length()
            self.barrier_contact_slots = ti.field(ti.i32, shape=slot_length)
            self.friction_contact_slots = ti.field(ti.i32, shape=slot_length)
        default_contact_blocks = max(
            1,
            self.contact_pair_count * self.contact_pair_capacity,
        )
        # ``BuildTriplet`` stores dense DIM x DIM blocks.  Capacity is thus
        # counted in coupled node pairs, not scalar Hessian entries.
        barrier_nnz = kwargs.get("barrier_nnz", default_contact_blocks)
        friction_nnz = kwargs.get(
            "friction_nnz",
            default_contact_blocks if self.activate_fric else 1,
        )
        total_dofs = self.iga.degree_of_freedom + self.mpm.degree_of_freedom
        active_node_capacity = max(1, total_dofs // config.DIM)
        contact_reduced_nnz = active_node_capacity * max(0, active_node_capacity - 1)
        self.barrier_hash_matrix = BuildTriplet(
            dim=config.DIM,
            max_pairs_num=max(1, barrier_nnz),
            max_nonzeros=max(1, min(barrier_nnz, contact_reduced_nnz)),
            max_active_nodes=active_node_capacity,
            symmetric=False,
            device_reduction=True,
            reduction="bucket",
        )
        self.friction_hash_matrix = BuildTriplet(
            dim=config.DIM,
            max_pairs_num=max(1, friction_nnz),
            max_nonzeros=max(1, min(friction_nnz, contact_reduced_nnz)),
            max_active_nodes=active_node_capacity,
            symmetric=False,
            device_reduction=True,
            reduction="bucket",
        )
        self.barrier_grad = ti.field(ti.f64, shape=total_dofs)
        self.friction_grad = ti.field(ti.f64, shape=total_dofs)
        self.fully_implicit_endpoint_velocity = ti.Vector.field(
            config.DIM,
            ti.f64,
            shape=max(1, total_dofs // config.DIM),
        )
        self.fully_implicit_contact_status = ti.field(ti.i32, shape=())
        self.barrier_nnz_count = ti.field(ti.i32, shape=1)
        self.barrier_nnz_overflow = ti.field(ti.i32, shape=1)
        self.barrier_projection_status = ti.field(ti.i32, shape=())
        if config.DIM == 3:
            reduced_dimension = config.DIM + 2
            projection_shape = (self.contact_capacity, self.contact_ctrlpts_capacity)
            self.barrier_projection_active = ti.field(ti.i32, shape=self.contact_capacity)
            self.barrier_projection_distance = ti.field(ti.f64, shape=self.contact_capacity)
            self.barrier_projection_span = ti.Vector.field(2, ti.i32, shape=self.contact_capacity)
            self.barrier_projection_free = ti.Vector.field(2, ti.i32, shape=self.contact_capacity)
            self.barrier_projection_pointer = ti.Vector.field(config.DIM, ti.f64, shape=self.contact_capacity)
            self.barrier_projection_tangent_u = ti.Vector.field(config.DIM, ti.f64, shape=self.contact_capacity)
            self.barrier_projection_tangent_v = ti.Vector.field(config.DIM, ti.f64, shape=self.contact_capacity)
            self.barrier_projection_shape = ti.field(ti.f64, shape=projection_shape)
            self.barrier_projection_derivative_u = ti.field(ti.f64, shape=projection_shape)
            self.barrier_projection_derivative_v = ti.field(ti.f64, shape=projection_shape)
            self.barrier_projection_metric = ti.Matrix.field(
                reduced_dimension,
                reduced_dimension,
                ti.f64,
                shape=self.contact_capacity,
            )
            self.barrier_projection_point_jacobian = ti.Matrix.field(
                reduced_dimension,
                config.DIM,
                ti.f64,
                shape=self.contact_capacity,
            )
        self.friction_nnz_count = ti.field(ti.i32, shape=1)
        self.friction_nnz_overflow = ti.field(ti.i32, shape=1)
        self.barrier_nnz_capacity = int(barrier_nnz)
        self.friction_nnz_capacity = int(friction_nnz)

        # Official IPC lagged friction is a projected-Newton method: local
        # elastic, barrier, and frozen-friction Hessians are PSD before global
        # scattering, so the Newmark/Dirichlet-constrained system uses PCG.
        # Fully implicit friction retains the exact generally nonsymmetric
        # Jacobian and therefore uses BiCGSTAB without any PSD projection.
        self.monolithic_solver_name = "BiCGSTAB" if self.friction_mode == "fully_implicit" else "PCG"
        assembly_key = (
            str(kwargs.get("assemble_type", kwargs.get("assembly", "HashTriplet")))
            .strip()
            .replace("_", "")
            .replace("-", "")
            .lower()
        )
        assembly_aliases = {
            "hash": "HashTriplet",
            "hashtriplet": "HashTriplet",
            "triplet": "HashTriplet",
            "buildtriplet": "HashTriplet",
            "coo": "COO",
            "coordinate": "COO",
            "coordinatesparse": "COO",
            "coordinatesparsematrix": "COO",
        }
        if assembly_key not in assembly_aliases:
            raise ValueError("IGA-MPM assemble_type must be 'COO' or 'HashTriplet'")
        self.assemble_type = assembly_aliases[assembly_key]
        if not hasattr(self.iga, "hash_matrix") or not hasattr(self.mpm, "hash_matrix"):
            raise ValueError(
                "IGA-MPM requires Taichi block sources from both implicit "
                "sub-solvers before scattering to coupled COO or HashTriplet"
            )

        self.monolithic_hash_matrix = None
        self.monolithic_coo_matrix = None
        self.contact_step_alpha = ti.field(ti.f64, shape=())
        self.contact_query_status = ti.field(ti.i32, shape=())
        self.contact_accd_toc = ti.field(ti.f64, shape=self.contact_capacity)
        self.contact_accd_distance = ti.field(ti.f64, shape=self.contact_capacity)
        self.contact_accd_motion_bound = ti.field(ti.f64, shape=self.contact_capacity)
        self.contact_accd_active = ti.field(ti.i32, shape=self.contact_capacity)
        self.contact_accd_active_count = ti.field(ti.i32, shape=())
        self.monolithic_linear_solver_tolerance = float(kwargs.get("monolithic_linear_solver_tolerance", 1.0e-10))
        self.monolithic_linear_solver_relative_tolerance = float(
            kwargs.get("monolithic_linear_solver_relative_tolerance", 0.0)
        )
        self.monolithic_linear_solver_max_iters = int(
            kwargs.get(
                "monolithic_linear_solver_max_iters",
                max(500, 5 * total_dofs),
            )
        )
        if not np.isfinite(self.monolithic_linear_solver_tolerance) or self.monolithic_linear_solver_tolerance <= 0.0:
            raise ValueError("monolithic_linear_solver_tolerance must be finite and positive")
        if (
            not np.isfinite(self.monolithic_linear_solver_relative_tolerance)
            or self.monolithic_linear_solver_relative_tolerance < 0.0
        ):
            raise ValueError("monolithic_linear_solver_relative_tolerance must be finite and non-negative")
        if self.monolithic_linear_solver_max_iters <= 0:
            raise ValueError("monolithic_linear_solver_max_iters must be positive")
        body_pair_capacity = int(self.iga.hash_matrix.non_diag.max_pairs_num) + int(
            self.mpm.hash_matrix.non_diag.max_pairs_num
        )
        body_nnz_capacity = int(self.iga.hash_matrix.max_nonzeros) + int(self.mpm.hash_matrix.max_nonzeros)
        device_friction_capacity = self.activate_fric
        coupled_pair_capacity = (
            body_pair_capacity
            + int(self.barrier_hash_matrix.non_diag.max_pairs_num)
            + (int(self.friction_hash_matrix.non_diag.max_pairs_num) if device_friction_capacity else 0)
        )
        coupled_nnz_capacity = (
            body_nnz_capacity
            + int(self.barrier_hash_matrix.max_nonzeros)
            + (int(self.friction_hash_matrix.max_nonzeros) if device_friction_capacity else 0)
        )
        lagged_symmetric_system = self.friction_mode != "fully_implicit"
        if lagged_symmetric_system:
            active_nodes = max(1, total_dofs // config.DIM)
            coupled_nnz_capacity = min(
                coupled_nnz_capacity,
                active_nodes * max(0, active_nodes - 1) // 2,
            )
        if self.assemble_type == "HashTriplet":
            self.monolithic_hash_matrix = BuildTriplet(
                dim=config.DIM,
                max_pairs_num=max(1, coupled_pair_capacity),
                max_nonzeros=max(1, coupled_nnz_capacity),
                max_active_nodes=max(1, total_dofs // config.DIM),
                symmetric=False,
                solver=self.monolithic_solver_name,
                matrix_symmetric=lagged_symmetric_system,
                full_symmetric_input=lagged_symmetric_system,
                device_reduction=True,
            )
            coordinates, slots = self.iga.fixed_block_coordinates(upper_triangle=lagged_symmetric_system)
            self.monolithic_hash_matrix.install_fixed_pattern(coordinates)
            self.iga_fixed_slots = ti.field(ti.i32, shape=slots.shape)
            self.iga_fixed_slots.from_numpy(slots)
        else:
            source_hashes = (
                self.iga.hash_matrix,
                self.mpm.hash_matrix,
                self.barrier_hash_matrix,
                self.friction_hash_matrix if device_friction_capacity else None,
            )
            coo_capacity = total_dofs
            for source in source_hashes:
                if source is not None:
                    coo_capacity += (
                        config.DIM * config.DIM * (int(source.max_active_nodes) + int(source.non_diag.max_pairs_num))
                    )
            self.monolithic_coo_matrix = CoordinateSparseMatrix(
                max(1, coo_capacity),
                total_dofs,
                preconditioned=True,
                symmetry=lagged_symmetric_system,
            )
        self.monolithic_coo_count = ti.field(ti.i32, shape=())
        self.monolithic_coo_overflow = ti.field(ti.i32, shape=())
        self.monolithic_coo_diagonal = ti.field(ti.f64, shape=total_dofs)
        self.monolithic_rhs = ti.field(ti.f64, shape=total_dofs)
        self.monolithic_physical_rhs = ti.field(ti.f64, shape=total_dofs)
        self.monolithic_correction = ti.field(ti.f64, shape=total_dofs)
        self.monolithic_fixed = ti.field(ti.i32, shape=total_dofs)
        self.monolithic_fixed_correction = ti.field(ti.f64, shape=total_dofs)
        self.monolithic_tangent_product = ti.field(ti.f64, shape=total_dofs)
        self.monolithic_entry_iga_displacement = ti.field(ti.f64, shape=self.iga.degree_of_freedom)
        self.monolithic_entry_mpm_displacement = ti.field(ti.f64, shape=self.mpm.degree_of_freedom)
        self.implicit_state_status = ti.field(ti.i32, shape=7)
        iga_node_capacity = int(self.iga.patch.control_points.shape[0])
        particle_capacity = int(self.mpm.particle.shape[0])
        grid_capacity = int(self.mpm.grid.shape[0])
        self.implicit_snapshot_iga_control_points = ti.Vector.field(config.DIM, ti.f64, shape=iga_node_capacity)
        self.implicit_snapshot_iga_velocity = ti.Vector.field(config.DIM, ti.f64, shape=iga_node_capacity)
        self.implicit_snapshot_iga_acceleration = ti.Vector.field(config.DIM, ti.f64, shape=iga_node_capacity)
        self.implicit_snapshot_mpm_position = ti.Vector.field(config.DIM, ti.f64, shape=particle_capacity)
        self.implicit_snapshot_mpm_velocity = ti.Vector.field(config.DIM, ti.f64, shape=particle_capacity)
        self.implicit_snapshot_mpm_acceleration = ti.Vector.field(config.DIM, ti.f64, shape=particle_capacity)
        self.implicit_snapshot_mpm_deformation = ti.Matrix.field(
            self.mpm.material_dimension,
            self.mpm.material_dimension,
            ti.f64,
            shape=particle_capacity,
        )
        self.mpm_has_plastic_history = bool(self.mpm.is_finite_strain_plastic)
        self.implicit_snapshot_mpm_plastic_history = None
        if self.mpm_has_plastic_history:
            history_state_size = int(self.mpm.material.history_state_size)
            if history_state_size <= 0:
                raise RuntimeError(
                    "IGA-MPM finite-strain plasticity requires a material-owned " "device history-state interface"
                )
            self.implicit_snapshot_mpm_plastic_history = ti.Vector.field(
                history_state_size, ti.f64, shape=particle_capacity
            )
        self.implicit_snapshot_grid_mass = ti.field(ti.f64, shape=grid_capacity)
        self.implicit_snapshot_grid_velocity = ti.Vector.field(config.DIM, ti.f64, shape=grid_capacity)
        self.implicit_snapshot_grid_acceleration = ti.Vector.field(config.DIM, ti.f64, shape=grid_capacity)
        self.implicit_initialized = False
        self.implicit_step_in_progress = False
        self.implicit_step_index = 0
        self._implicit_step_iga_displacement = None
        self._implicit_step_mpm_displacement = None

        if self.friction_mode == "fully_implicit":
            self._validate_fully_implicit_time_integration()
        self.contactor.freeze_configuration()

    @staticmethod
    def _full_material_step():
        return 1.0

    def _mpm_material_ccd(self):
        return self.mpm.ccd(self.mpm.active_dof)

    def _compute_dynamic_mass_list(self):
        self.mpm.compute_mass_list(self.mpm.integration)

    @property
    def total_degree_of_freedom(self):
        return self.iga.degree_of_freedom + self.mpm.degree_of_freedom


__all__ = ["Engine"]
