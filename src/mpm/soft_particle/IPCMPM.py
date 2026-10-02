import math

import taichi as ti
import numpy as np
from taichi.lang.impl import current_cfg

import src.mpm.config as config
from src.contact_detection.continuous_contact_detection import (
    linear_gap_accd,
    point_point_accd,
)
from src.mpm.engines.direct.ImplicitMPM import ImplicitMPM
from src.mpm.generator.Ground import Ground
from src.physics_model.contact_model.ipc.IPC import (
    Barrier,
    Friction,
    ipc_fully_implicit_point_plane_stribeck_force,
    ipc_fully_implicit_point_plane_stribeck_friction,
    semi_ipc_find,
    semi_ipc_find_or_insert,
    semi_ipc_terms,
    semi_ipc_update_multiplier,
)
from src.physics_model.contact_model.ipc.ContactAssembly import (
    psd_project_nd,
    scatter_hash_scalar,
)
from src.physics_model.contact_model.ipc.ContactMeasure import (
    point_contact_measure,
    symmetric_contact_measure,
)
from src.mpm.soft_particle.IPCFunction import PointPointDerivative, PointGroundDerivative
from src.mpm.utils import add_field, copy_field
from src.linear_solver.BuildTriplet import BuildTriplet
from src.utils.MatrixFunction import contraction, flatten_matrix, unflatten_matrix


def _first_float(value, default):
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[0]) if len(value) > 0 else float(default)
    if value is None:
        return float(default)
    return float(value)


def _normalize_contact_search(value):
    key = str(value).strip().replace("_", "").replace("-", "").replace(" ", "").lower()
    if key in ("linkedcell", "cell", "lc"):
        return "LinkedCell"
    if key in ("hierarchicallinkedcell", "hierarchicalcell", "hlc"):
        return "HierarchicalLinkedCell"
    if key in ("bvh", "boundingvolumehierarchy"):
        return "BVH"
    if key in ("brust", "brute", "bruteforce"):
        return "Brust"
    raise ValueError("contact_search must be 'LinkedCell', 'HierarchicalLinkedCell', " "'BVH', or explicit 'Brust'")


def _normalize_friction_mode(value):
    key = str(value).strip().replace("-", "_").replace(" ", "_").lower()
    aliases = {
        "lag": "lagged",
        "lagged": "lagged",
        "fullyimplicit": "fully_implicit",
        "fully_implicit": "fully_implicit",
        "fully_implicit_experimental": "fully_implicit",
    }
    if key not in aliases:
        raise ValueError("friction_mode must be 'lagged' or 'fully_implicit'")
    return aliases[key]


def _normalize_friction_iterations(value):
    try:
        numeric_value = float(value)
        iterations = int(numeric_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("friction_iterations must be an integer") from exc
    if not np.isfinite(numeric_value) or numeric_value != iterations:
        raise ValueError("friction_iterations must be an integer")
    # Original IPC's fricIterAmt treats every non-positive value as
    # iterate-to-convergence. Canonicalize those aliases to one sentinel.
    return -1 if iterations <= 0 else iterations


def _positive_integer(value, name):
    try:
        numeric_value = float(value)
        integer_value = int(numeric_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric_value) or numeric_value != integer_value or integer_value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer_value


def _positive_float(value, name):
    try:
        float_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if not np.isfinite(float_value) or float_value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return float_value


def _nonnegative_float(value, name):
    try:
        float_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not np.isfinite(float_value) or float_value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float_value


class IPCInfeasibleContactState(RuntimeError):
    """A trial configuration crossed the IPC minimum-distance boundary."""


@ti.data_oriented
class IPCMPM:
    @property
    def is_semi(self):
        return getattr(self, "_is_semi", False)

    @is_semi.setter
    def is_semi(self, value):
        self._is_semi = bool(value)

    def __init__(self, mpm: ImplicitMPM, ground: Ground = None, **kwargs):
        self.mpm = mpm
        self.ground = ground
        self.ground.finalize()
        self.barrier = Barrier(**kwargs)
        self.is_semi = self.barrier.is_semi
        self.friction = Friction(**kwargs)
        self.pderivative = PointPointDerivative()
        self.gderivative = PointGroundDerivative(self.ground)
        self.activate_barrier = True
        self.activate_fric = True
        self.mpm.build_surface_node()
        integration = np.asarray(self.mpm.integration, dtype=np.float64)
        if integration.size < 3 or not np.all(np.isfinite(integration[:3])) or np.any(integration[:3] <= 0.0):
            raise ValueError("newmark must contain positive [alpha, beta, gamma]")
        damping = float(self.mpm.damping)
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("implicit IPC MPM damping must be finite and non-negative")
        self.endpoint_velocity_displacement_scale = float(
            0.5 * integration[2] / integration[0] / integration[1] / self.mpm.dt
        )
        self.endpoint_velocity_previous_scale = float(-(0.5 * integration[2] / integration[0] / integration[1] - 1.0))
        self.endpoint_velocity_acceleration_scale = float(-0.5 * self.mpm.dt * (integration[2] / integration[1] - 2.0))
        self.line_search_work_tol = kwargs.get("line_search_work_tol", 1.0e-8)
        self.line_search_energy_rtol = kwargs.get("line_search_energy_rtol", 1.0e-12)
        self.line_search_energy_atol = kwargs.get("line_search_energy_atol", 1.0e-14)
        self.line_search_min_alpha = kwargs.get("line_search_min_alpha", 1.0e-12)
        self.line_search_stagnation_alpha = kwargs.get("line_search_stagnation_alpha", 1.0e-6)
        self.line_search_stagnation_energy_tol = kwargs.get("line_search_stagnation_energy_tol", 1.0e-6)
        self.line_search_max_backtracks = kwargs.get("line_search_max_backtracks", 25)
        self.line_search_descent_fallback = kwargs.get("line_search_descent_fallback", True)
        self.line_search_armijo = _positive_float(
            kwargs.get("line_search_armijo", 1.0e-4),
            "line_search_armijo",
        )
        if self.line_search_armijo >= 1.0:
            raise ValueError("line_search_armijo must lie in (0, 1)")
        self.line_search_stop_iter = False
        self.line_search_stop_reason = ""
        self.line_search_failed = False
        self.line_search_used_fallback = False
        self.line_search_last_g0 = 0.0
        self.line_search_last_alpha0 = 1.0
        self.line_search_last_alpha = 1.0
        self.line_search_last_backtracks = 0
        self.line_search_last_previous_energy = 0.0
        self.line_search_last_trial_energy = 0.0
        self.line_search_last_accepted_energy = 0.0
        self.friction_reference_initialized = False
        self.friction_mode = _normalize_friction_mode(kwargs.get("friction_mode", "lagged"))
        if self.is_semi and self.friction_mode != "lagged":
            raise ValueError("MPM SemiIPC currently requires friction_mode='lagged'")
        self.friction_iterations = _normalize_friction_iterations(
            kwargs.get(
                "friction_iterations",
                kwargs.get("friction_fixed_point_iterations", 1),
            )
        )
        self.friction_max_iterations = _positive_integer(
            kwargs.get("friction_max_iterations", 50),
            "friction_max_iterations",
        )
        self.friction_tolerance = _positive_float(
            kwargs.get(
                "friction_residual",
                kwargs.get(
                    "friction_tolerance",
                    kwargs.get("friction_fixed_point_tolerance", self.mpm.tol),
                ),
            ),
            "friction_residual",
        )
        self.last_newton_iterations = 0
        self.last_newton_residual = np.inf
        self.last_inner_converged = False
        self.last_inner_failure_reason = ""
        self.last_friction_iterations = 0
        self.last_friction_residual = np.inf
        self.last_friction_converged = False
        self.fully_implicit_armijo = _positive_float(
            kwargs.get("fully_implicit_armijo", 1.0e-4),
            "fully_implicit_armijo",
        )
        if self.fully_implicit_armijo >= 1.0:
            raise ValueError("fully_implicit_armijo must lie in (0, 1)")
        self.fully_implicit_backtrack = float(kwargs.get("fully_implicit_backtrack", 0.5))
        if not 0.0 < self.fully_implicit_backtrack < 1.0:
            raise ValueError("fully_implicit_backtrack must lie in (0, 1)")
        self.fully_implicit_max_backtracks = _positive_integer(
            kwargs.get("fully_implicit_max_backtracks", 30),
            "fully_implicit_max_backtracks",
        )
        self.fully_implicit_residual_atol = _nonnegative_float(
            kwargs.get("fully_implicit_residual_atol", 1.0e-12),
            "fully_implicit_residual_atol",
        )
        self.fully_implicit_residual_rtol = _nonnegative_float(
            kwargs.get("fully_implicit_residual_rtol", 1.0e-8),
            "fully_implicit_residual_rtol",
        )
        self.last_fully_implicit_residual = np.inf
        self.last_fully_implicit_initial_residual = np.inf
        self.last_fully_implicit_backtracks = 0
        self.fully_implicit_velocity_predictor = bool(kwargs.get("fully_implicit_velocity_predictor", True))
        self.contact_search = _normalize_contact_search(
            kwargs.get("contact_search", kwargs.get("search", "LinkedCell"))
        )
        self.last_particle_search_backend = self.contact_search

        ground_barrier_dtype = ti.types.struct(surfaceID=ti.i32, wallID=ti.i32, distance=ti.f64, semi_slot=ti.i32)
        particle_barrier_dtype = ti.types.struct(masterID=ti.i32, slaveID=ti.i32, distance=ti.f64, semi_slot=ti.i32)
        body_pair_dtype = ti.types.struct(master=ti.i32, slave=ti.i32)
        surface_body_contact_dtype = ti.types.struct(surfaceID=ti.i32, bodyID=ti.i32)
        ground_friction_dtype = ti.types.struct(surfaceID=ti.i32, wallID=ti.i32, mu_lambda=ti.f64)
        particle_friction_dtype = ti.types.struct(
            masterID=ti.i32, slaveID=ti.i32, mu_lambda=ti.f64, normal=ti.types.vector(config.DIM, ti.f64)
        )
        self._ground_barrier_dtype = ground_barrier_dtype
        self._particle_barrier_dtype = particle_barrier_dtype
        self._ground_friction_dtype = ground_friction_dtype
        self._particle_friction_dtype = particle_friction_dtype

        coordination_number = kwargs.get("coordination_number", [self.mpm.n_body, self.ground.num])
        barrier_set = kwargs.get(
            "barrier_set",
            [
                int(coordination_number[0] * self.mpm.total_surface_num),
                int(coordination_number[1] * self.mpm.total_surface_num),
            ],
        )
        friction_set = kwargs.get(
            "friction_set",
            [
                int(coordination_number[0] * self.mpm.total_surface_num),
                int(coordination_number[1] * self.mpm.total_surface_num),
            ],
        )
        if not self.friction.has_friction and not kwargs.get("activate_friction", False):
            self.activate_fric = False
        self.enable_particle_contact = self.mpm.n_body > 1
        self.curr_barrier_contact_num = 0
        self.curr_friction_contact_num = 0
        # Contact kernels scatter one dense node block per support-node pair.
        # The previous scalar-COO bound used ``influenced_node`` (which also
        # includes coordinate components) and allocated/atomically appended
        # DIM^2 mostly-empty blocks for every actual dense block.
        support_nodes = int(self.mpm.shape_func.max_node_per_particle)
        support_block_pairs = support_nodes * support_nodes
        self.barrier_nnz = 4 * support_block_pairs * barrier_set[0] + support_block_pairs * barrier_set[1]
        self.friction_nnz = 4 * support_block_pairs * friction_set[0] + support_block_pairs * friction_set[1]
        active_node_capacity = max(1, self.mpm.degree_of_freedom // config.DIM)
        maximum_unique_off_diagonal = active_node_capacity * max(0, active_node_capacity - 1)
        barrier_reduced_nnz = min(self.barrier_nnz, maximum_unique_off_diagonal)
        friction_reduced_nnz = min(self.friction_nnz, maximum_unique_off_diagonal)
        if self.activate_barrier:
            self.barrier_hash_matrix = BuildTriplet(
                config.DIM,
                max(1, self.barrier_nnz),
                max(1, barrier_reduced_nnz),
                active_node_capacity,
                symmetric=False,
                device_reduction=True,
            )
        if self.activate_fric:
            self.friction_hash_matrix = BuildTriplet(
                config.DIM,
                max(1, self.friction_nnz),
                max(1, friction_reduced_nnz),
                active_node_capacity,
                symmetric=False,
                device_reduction=True,
            )

        # Every Taichi architecture uses one monolithic device matrix. Body, barrier and
        # friction blocks are appended device-to-device and reduced once;
        # neither CSR addition nor a full correction vector crosses PCIe.
        self.cuda_monolithic_solver = True
        self.system_hash_matrix = None
        self.system_physical_rhs = None
        if self.cuda_monolithic_solver:
            system_raw_capacity = self.mpm.hash_matrix.non_diag.max_pairs_num
            system_reduced_capacity = self.mpm.hash_matrix.max_nonzeros
            if self.activate_barrier:
                system_raw_capacity += self.barrier_hash_matrix.non_diag.max_pairs_num
                system_reduced_capacity += self.barrier_hash_matrix.max_nonzeros
            if self.activate_fric:
                system_raw_capacity += self.friction_hash_matrix.non_diag.max_pairs_num
                system_reduced_capacity += self.friction_hash_matrix.max_nonzeros
            system_reduced_capacity = min(system_reduced_capacity, maximum_unique_off_diagonal)
            monolithic_solver = self._cuda_monolithic_krylov_solver()
            lagged_symmetric_system = monolithic_solver == "PCG"
            if lagged_symmetric_system:
                system_reduced_capacity = min(
                    system_reduced_capacity,
                    maximum_unique_off_diagonal // 2,
                )
            self.system_hash_matrix = BuildTriplet(
                config.DIM,
                max(1, system_raw_capacity),
                max(1, system_reduced_capacity),
                max(1, self.mpm.degree_of_freedom // config.DIM),
                symmetric=False,
                matrix_symmetric=lagged_symmetric_system,
                full_symmetric_input=lagged_symmetric_system,
                solver=monolithic_solver,
                device_reduction=True,
            )
            # Dirichlet elimination overwrites ``mpm.rhs``.  Keep the
            # unconstrained physical residual on device so the
            # nonconservative fully implicit line search can evaluate the
            # exact merit slope -R^T K p without rebuilding or downloading
            # the matrix.
            self.system_physical_rhs = ti.field(ti.f64, shape=self.mpm.degree_of_freedom)

        if self.activate_barrier:
            self.gbarrier = ground_barrier_dtype.field(shape=max(1, int(barrier_set[1])))
            self.gbarrierNum = ti.field(ti.i32, shape=1)
            self.gbarrier_overflow = ti.field(ti.i32, shape=1)
            self.barrier_infeasible = ti.field(ti.i32, shape=1)
            self.pbarrier = particle_barrier_dtype.field(shape=barrier_set[0])
            self.pbarrierNum = ti.field(ti.i32, shape=1)
            self.pbarrier_overflow = ti.field(ti.i32, shape=1)
            semi_capacity = 1 << (max(2 * (int(barrier_set[0]) + int(barrier_set[1])), 1) - 1).bit_length()
            self.semi_capacity = semi_capacity
            self.semi_state = ti.field(ti.i32, shape=semi_capacity)
            self.semi_key = ti.Vector.field(4, ti.i32, shape=semi_capacity)
            self.semi_multiplier = ti.field(ti.f64, shape=semi_capacity)
            self.semi_normal = ti.Vector.field(config.DIM, ti.f64, shape=semi_capacity)
            self.semi_count = ti.field(ti.i32, shape=())
            self.semi_overflow = ti.field(ti.i32, shape=())
            self.semi_constraint_violation = ti.field(ti.f64, shape=())
            self.semi_state.fill(2)
        if self.activate_fric:
            ground_friction_capacity = max(1, int(friction_set[1]))
            self.gfriction = ground_friction_dtype.field(shape=ground_friction_capacity)
            self.gfrictionNum = ti.field(ti.i32, shape=1)
            self.gfriction_overflow = ti.field(ti.i32, shape=1)
            self.pfriction = particle_friction_dtype.field(shape=friction_set[0])
            self.pfrictionNum = ti.field(ti.i32, shape=1)
            self.pfriction_overflow = ti.field(ti.i32, shape=1)
            self.adjoint_gfriction = ground_friction_dtype.field(shape=ground_friction_capacity)
            self.adjoint_pfriction = particle_friction_dtype.field(shape=friction_set[0])
            self.adjoint_gfriction_num = ti.field(ti.i32, shape=())
            self.adjoint_pfriction_num = ti.field(ti.i32, shape=())
            self.adjoint_friction_valid = ti.field(ti.i32, shape=())
            self.friction_grad = ti.field(ti.f64, shape=self.mpm.degree_of_freedom)
            self.hat_x = ti.Vector.field(config.DIM, ti.f64, shape=self.mpm.total_surface_num)
            self.fully_implicit_point_force = ti.Vector.field(config.DIM, ti.f64, shape=max(1, int(friction_set[1])))

        self.normal_force = ti.Vector.field(config.DIM, ti.f64, shape=self.mpm.n_body)
        self.tangential_force = ti.Vector.field(config.DIM, ti.f64, shape=self.mpm.n_body)
        self.last_adjoint_result = None
        self.pending_adjoint_seed = None
        self.pending_adjoint_mode = None
        self.pending_plastic_state_vjp = None
        self.last_elastic_differentiation = None
        self.trajectory_capture_active = False
        self.adjoint_trajectory_record = -1
        self.gravity_vjp = ti.Vector.field(config.DIM, ti.f64, shape=())
        self.young_vjp = ti.field(ti.f64, shape=())
        # (Young, Poisson, DP cohesion/friction angle or VM yield/hardening)
        self.material_parameter_vjp = ti.Vector.field(4, ti.f64, shape=())
        self.friction_parameter_vjp = ti.Vector.field(4, ti.f64, shape=())
        history_capacity = max(
            self.mpm.n_particles if self.mpm.is_finite_strain_plastic else 1,
            1,
        )
        self.plastic_inverse_vjp = ti.Matrix.field(3, 3, ti.f64, shape=history_capacity)
        self.plastic_equivalent_strain_vjp = ti.field(ti.f64, shape=history_capacity)
        self.plastic_volumetric_strain_vjp = ti.field(ti.f64, shape=history_capacity)
        self.plastic_deformation_vjp = ti.Matrix.field(3, 3, ti.f64, shape=history_capacity)
        self.plastic_commit_deformation_vjp = ti.Matrix.field(3, 3, ti.f64, shape=history_capacity)
        self.plastic_trial_deformation_vjp = ti.Matrix.field(3, 3, ti.f64, shape=history_capacity)
        self.plastic_commit_inverse_vjp = ti.Matrix.field(3, 3, ti.f64, shape=history_capacity)
        self.plastic_commit_equivalent_vjp = ti.field(ti.f64, shape=history_capacity)
        self.plastic_commit_volumetric_vjp = ti.field(ti.f64, shape=history_capacity)
        self.plastic_output_deformation_vjp = ti.Matrix.field(3, 3, ti.f64, shape=history_capacity)
        self.plastic_output_inverse_vjp = ti.Matrix.field(3, 3, ti.f64, shape=history_capacity)
        self.plastic_output_equivalent_vjp = ti.field(ti.f64, shape=history_capacity)
        self.plastic_output_volumetric_vjp = ti.field(ti.f64, shape=history_capacity)
        plastic_dof_capacity = self.mpm.degree_of_freedom if self.mpm.is_finite_strain_plastic else 1
        self.plastic_commit_grid_vjp = ti.field(ti.f64, shape=plastic_dof_capacity)
        self.plastic_step_rhs = ti.field(ti.f64, shape=plastic_dof_capacity)
        self.particle_position_vjp = ti.Vector.field(config.DIM, ti.f64, shape=history_capacity)
        # Keep contact-position scatter scalar on CUDA.  Atomic writes into a
        # vector-field component can be emitted as misaligned transactions by
        # Taichi's CUDA backend when the target is dynamically indexed.
        self.barrier_position_vjp_flat = ti.field(ti.f64, shape=history_capacity * config.DIM)
        self.particle_velocity_vjp = ti.Vector.field(config.DIM, ti.f64, shape=history_capacity)
        self.particle_acceleration_vjp = ti.Vector.field(config.DIM, ti.f64, shape=history_capacity)
        self.particle_output_position_vjp = ti.Vector.field(config.DIM, ti.f64, shape=history_capacity)
        self.particle_output_velocity_vjp = ti.Vector.field(config.DIM, ti.f64, shape=history_capacity)
        self.particle_output_acceleration_vjp = ti.Vector.field(config.DIM, ti.f64, shape=history_capacity)
        trajectory_node_capacity = self.mpm.total_background_grid_num if self.mpm.is_finite_strain_plastic else 1
        self.trajectory_grid_vjp = ti.field(ti.f64, shape=plastic_dof_capacity)
        self.trajectory_node_mass_vjp = ti.field(ti.f64, shape=trajectory_node_capacity)
        self.trajectory_node_velocity_vjp = ti.Vector.field(config.DIM, ti.f64, shape=trajectory_node_capacity)
        self.trajectory_node_acceleration_vjp = ti.Vector.field(config.DIM, ti.f64, shape=trajectory_node_capacity)
        self.minimum_candidate_distance = ti.field(ti.f64, shape=1)
        self.candidate_state_invalid = ti.field(ti.i32, shape=1)
        # Transactional rollback remains device-resident on every backend.
        self.device_grid_disp_snapshot = ti.field(ti.f64, shape=self.mpm.degree_of_freedom)

        dhat_value = _first_float(kwargs.get("dhat", 1.0e-3), 1.0e-3)
        surface_starts = [0 for _ in range(max(self.mpm.n_body, 1))]
        surface_counts = [0 for _ in range(max(self.mpm.n_body, 1))]
        cursor = 0
        for body_id, body in enumerate(self.mpm.bodies.bodies.values()):
            count = int(len(body.get("surface_id", [])))
            if count > 0 and body_id < self.mpm.n_body:
                surface_starts[body_id] = cursor
                surface_counts[body_id] = count
            cursor += count
        self.body_surface_start = ti.field(ti.i32, shape=max(self.mpm.n_body, 1))
        self.body_surface_count = ti.field(ti.i32, shape=max(self.mpm.n_body, 1))
        self.body_surface_start.from_numpy(np.asarray(surface_starts, dtype=np.int32))
        self.body_surface_count.from_numpy(np.asarray(surface_counts, dtype=np.int32))
        self.max_body_surface_count = max(surface_counts, default=0)
        surface_bodies = np.empty(self.mpm.total_surface_num, dtype=np.int32)
        for body_id, (start, count) in enumerate(zip(surface_starts, surface_counts)):
            surface_bodies[start : start + count] = body_id
        self.surface_body = ti.field(ti.i32, shape=max(self.mpm.total_surface_num, 1))
        if self.mpm.total_surface_num > 0:
            self.surface_body.from_numpy(surface_bodies)

        coord0 = (
            coordination_number[0]
            if isinstance(coordination_number, (list, tuple, np.ndarray)) and len(coordination_number) > 0
            else coordination_number
        )
        max_body_pairs = max(self.mpm.n_body * max(self.mpm.n_body - 1, 1) // 2, 1)
        self.body_pair = body_pair_dtype.field(shape=int(kwargs.get("body_pair_capacity", max_body_pairs)))
        self.body_pair_num = ti.field(ti.i32, shape=1)
        self.body_pair_overflow = ti.field(ti.i32, shape=1)
        # ponytail: dense marks make dedup collision-free; replace with a
        # device hash only if O(body_count^2) storage becomes material.
        self.body_pair_mark = ti.field(ti.i32, shape=max(self.mpm.n_body * self.mpm.n_body, 1))
        point_body_capacity = int(
            kwargs.get("point_body_contact_capacity", max(1, int(coord0) * self.mpm.total_surface_num))
        )
        self.surface_body_contact = surface_body_contact_dtype.field(shape=max(point_body_capacity, 1))
        self.surface_body_contact_num = ti.field(ti.i32, shape=1)
        self.surface_body_contact_overflow = ti.field(ti.i32, shape=1)
        self.surface_body_prefix = ti.field(ti.i32, shape=max(int(self.mpm.total_surface_num), 1))
        self.surface_body_cursor = ti.field(ti.i32, shape=max(int(self.mpm.total_surface_num), 1))
        self.body_min = ti.Vector.field(config.DIM, ti.f64, shape=max(self.mpm.n_body, 1))
        self.body_max = ti.Vector.field(config.DIM, ti.f64, shape=max(self.mpm.n_body, 1))
        self.surface_sweep_disp = ti.Vector.field(config.DIM, ti.f64, shape=max(int(self.mpm.total_surface_num), 1))
        cell_size = _first_float(kwargs.get("contact_cell_size", None), max(2.0 * dhat_value, 1.0e-5))
        domain = np.asarray(self.mpm.domain, dtype=np.float64).reshape(-1)
        if domain.size < config.DIM:
            domain = np.pad(domain, (0, config.DIM - domain.size), mode="constant", constant_values=cell_size)
        domain = np.maximum(domain[: config.DIM], cell_size)
        max_cells = int(kwargs.get("contact_max_cells", 200000))
        cell_num = np.maximum(np.floor(domain / cell_size).astype(np.int32), 1)
        while int(np.prod(cell_num)) > max_cells:
            cell_size *= 1.25
            cell_num = np.maximum(np.floor(domain / cell_size).astype(np.int32), 1)
        self.body_cell_size = float(cell_size)
        self.body_inv_cell_size = 1.0 / self.body_cell_size
        self.body_cell_nx = int(cell_num[0])
        self.body_cell_ny = int(cell_num[1]) if config.DIM >= 2 else 1
        self.body_cell_nz = int(cell_num[2]) if config.DIM >= 3 else 1
        self.body_cell_sum = int(self.body_cell_nx * self.body_cell_ny * self.body_cell_nz)
        self.body_per_cell = int(kwargs.get("body_per_cell", max(8, int(coord0) if coord0 is not None else 8)))
        self.body_cell_count = ti.field(ti.i32, shape=max(self.body_cell_sum, 1))
        self.body_cell = ti.field(ti.i32, shape=max(self.body_cell_sum * self.body_per_cell, 1))
        self.body_cell_overflow = ti.field(ti.i32, shape=1)

        # Active contact lists must have a stable order on CUDA.  An atomic
        # compact append gives the same *set* of contacts but can assign a
        # different contact id on every Newton iteration; because contact id
        # owns the raw sparse-matrix slot, that permutation defeats the fast
        # raw->reduced mapping even when the geometry and active set are
        # unchanged.  Count contacts per surface, scan these counts entirely
        # on the device, then let each surface write its own contiguous range
        # in the fixed (surface, wall/body, target-surface) order.
        #
        # One field/executor is shared by all four contact lists because their
        # construction is sequential.  This keeps the extra storage O(number
        # of surface points), rather than allocating an O(N^2) candidate map.
        self.contact_surface_prefix = ti.field(ti.i32, shape=max(int(self.mpm.total_surface_num), 1))
        self.contact_prefix_sum_executor = None
        if current_cfg().arch == ti.cuda and int(self.mpm.total_surface_num) > 1:
            self.contact_prefix_sum_executor = ti.algorithms.PrefixSumExecutor(int(self.mpm.total_surface_num))

        if self.friction_mode == "fully_implicit" and self.activate_fric:
            self._validate_fully_implicit_configuration()
        elif self.friction_mode == "lagged" and self.activate_fric:
            self._validate_lagged_friction_configuration()

    def allocate_trajectory_contact_tape(self, capacity):
        """Allocate a device-resident accepted contact-set tape.

        The tape stores the contact relation used by the converged equilibrium
        solve of each time step.  It intentionally stores no Newton trial or
        CCD branch; the adjoint is conditional on this accepted relation.
        """
        capacity = _positive_integer(capacity, "trajectory contact tape capacity")
        if getattr(self, "trajectory_contact_capacity", 0) >= capacity:
            return
        if not self.activate_barrier:
            raise RuntimeError("trajectory contact tape requires IPC barrier contacts")

        self.trajectory_contact_capacity = capacity
        self.trajectory_gbarrier_capacity = int(self.gbarrier.shape[0])
        self.trajectory_pbarrier_capacity = int(self.pbarrier.shape[0])
        self.trajectory_gbarrier = self._ground_barrier_dtype.field(shape=capacity * self.trajectory_gbarrier_capacity)
        self.trajectory_pbarrier = self._particle_barrier_dtype.field(
            shape=capacity * self.trajectory_pbarrier_capacity
        )
        self.trajectory_gbarrier_num = ti.field(ti.i32, shape=capacity)
        self.trajectory_pbarrier_num = ti.field(ti.i32, shape=capacity)
        self.trajectory_contact_valid = ti.field(ti.i32, shape=capacity)
        self.trajectory_hat_x = None
        self.trajectory_gfriction = None
        self.trajectory_pfriction = None
        self.trajectory_gfriction_num = None
        self.trajectory_pfriction_num = None
        if self.activate_fric:
            self.trajectory_gfriction_capacity = int(self.gfriction.shape[0])
            self.trajectory_pfriction_capacity = int(self.pfriction.shape[0])
            self.trajectory_gfriction = self._ground_friction_dtype.field(
                shape=capacity * self.trajectory_gfriction_capacity
            )
            self.trajectory_pfriction = self._particle_friction_dtype.field(
                shape=capacity * self.trajectory_pfriction_capacity
            )
            self.trajectory_gfriction_num = ti.field(ti.i32, shape=capacity)
            self.trajectory_pfriction_num = ti.field(ti.i32, shape=capacity)
            self.trajectory_hat_x = ti.Vector.field(
                config.DIM,
                ti.f64,
                shape=capacity * self.mpm.total_surface_num,
            )

    @ti.kernel
    def _record_trajectory_contact_state(self, record: ti.i32, use_adjoint_friction: ti.i32):
        gbarrier_count = self.gbarrierNum[0]
        pbarrier_count = self.pbarrierNum[0]
        self.trajectory_gbarrier_num[record] = gbarrier_count
        self.trajectory_pbarrier_num[record] = pbarrier_count
        self.trajectory_contact_valid[record] = 1
        for contact in range(self.gbarrier.shape[0]):
            if contact < gbarrier_count:
                self.trajectory_gbarrier[record * self.trajectory_gbarrier_capacity + contact] = self.gbarrier[contact]
        for contact in range(self.pbarrier.shape[0]):
            if contact < pbarrier_count:
                self.trajectory_pbarrier[record * self.trajectory_pbarrier_capacity + contact] = self.pbarrier[contact]

        if ti.static(self.activate_fric):
            ground_count = self.gfrictionNum[0]
            particle_count = self.pfrictionNum[0]
            if use_adjoint_friction != 0 and self.adjoint_friction_valid[None] != 0:
                ground_count = self.adjoint_gfriction_num[None]
                particle_count = self.adjoint_pfriction_num[None]
            self.trajectory_gfriction_num[record] = ground_count
            self.trajectory_pfriction_num[record] = particle_count
            for contact in range(self.gfriction.shape[0]):
                if contact < ground_count:
                    if use_adjoint_friction != 0 and self.adjoint_friction_valid[None] != 0:
                        self.trajectory_gfriction[record * self.trajectory_gfriction_capacity + contact] = (
                            self.adjoint_gfriction[contact]
                        )
                    else:
                        self.trajectory_gfriction[record * self.trajectory_gfriction_capacity + contact] = (
                            self.gfriction[contact]
                        )
            for contact in range(self.pfriction.shape[0]):
                if contact < particle_count:
                    if use_adjoint_friction != 0 and self.adjoint_friction_valid[None] != 0:
                        self.trajectory_pfriction[record * self.trajectory_pfriction_capacity + contact] = (
                            self.adjoint_pfriction[contact]
                        )
                    else:
                        self.trajectory_pfriction[record * self.trajectory_pfriction_capacity + contact] = (
                            self.pfriction[contact]
                        )
            for surface in range(self.mpm.total_surface_num):
                self.trajectory_hat_x[record * self.mpm.total_surface_num + surface] = self.hat_x[surface]

    def record_trajectory_contact_state(self, record):
        if not hasattr(self, "trajectory_contact_capacity"):
            raise RuntimeError("allocate the trajectory contact tape before recording")
        record = int(record)
        if record < 0 or record >= self.trajectory_contact_capacity:
            raise IndexError("trajectory contact tape record is out of range")
        self._record_trajectory_contact_state(record, int(self.trajectory_capture_active))

    @ti.kernel
    def _load_trajectory_contact_state(self, record: ti.i32):
        gbarrier_count = self.trajectory_gbarrier_num[record]
        pbarrier_count = self.trajectory_pbarrier_num[record]
        self.gbarrierNum[0] = gbarrier_count
        self.pbarrierNum[0] = pbarrier_count
        self.gbarrier_overflow[0] = 0
        self.pbarrier_overflow[0] = 0
        for contact in range(self.gbarrier.shape[0]):
            if contact < gbarrier_count:
                self.gbarrier[contact] = self.trajectory_gbarrier[record * self.trajectory_gbarrier_capacity + contact]
        for contact in range(self.pbarrier.shape[0]):
            if contact < pbarrier_count:
                self.pbarrier[contact] = self.trajectory_pbarrier[record * self.trajectory_pbarrier_capacity + contact]

        if ti.static(self.activate_fric):
            ground_count = self.trajectory_gfriction_num[record]
            particle_count = self.trajectory_pfriction_num[record]
            self.gfrictionNum[0] = ground_count
            self.pfrictionNum[0] = particle_count
            self.gfriction_overflow[0] = 0
            self.pfriction_overflow[0] = 0
            for contact in range(self.gfriction.shape[0]):
                if contact < ground_count:
                    self.gfriction[contact] = self.trajectory_gfriction[
                        record * self.trajectory_gfriction_capacity + contact
                    ]
            for contact in range(self.pfriction.shape[0]):
                if contact < particle_count:
                    self.pfriction[contact] = self.trajectory_pfriction[
                        record * self.trajectory_pfriction_capacity + contact
                    ]
            for surface in range(self.mpm.total_surface_num):
                self.hat_x[surface] = self.trajectory_hat_x[record * self.mpm.total_surface_num + surface]

    def begin_trajectory_adjoint(self, record):
        if not hasattr(self, "trajectory_contact_capacity"):
            raise RuntimeError("allocate the trajectory contact tape before replay")
        record = int(record)
        if record < 0 or record >= self.trajectory_contact_capacity:
            raise IndexError("trajectory contact tape record is out of range")
        if int(self.trajectory_contact_valid[record]) == 0:
            raise RuntimeError(f"trajectory contact record {record} is not committed")
        self.adjoint_trajectory_record = record

    def end_trajectory_adjoint(self):
        self.adjoint_trajectory_record = -1

    @ti.kernel
    def _serial_surface_prefix_sum(self, prefix: ti.template()):
        """Device-only fallback for backends without Taichi's parallel scan."""
        ti.loop_config(serialize=True)
        for surface in range(self.mpm.total_surface_num):
            if surface > 0:
                prefix[surface] += prefix[surface - 1]

    def _scan_surface_prefix(self, prefix):
        if int(self.mpm.total_surface_num) <= 1:
            return
        if current_cfg().arch == ti.cuda:
            if self.contact_prefix_sum_executor is None:
                raise RuntimeError("CUDA contact compaction requires a device prefix-sum executor")
            self.contact_prefix_sum_executor.run(prefix)
        else:
            self._serial_surface_prefix_sum(prefix)

    def _scan_contact_surface_counts(self):
        """Inclusive scan without downloading the per-surface count array."""
        # CPU/Metal use the same device kernel; no host sort or full-array copy.
        self._scan_surface_prefix(self.contact_surface_prefix)

    @ti.kernel
    def _finalize_stable_contact_count(
        self,
        contact_count: ti.template(),
        overflow: ti.template(),
        capacity: ti.i32,
    ):
        required = 0
        if ti.static(self.mpm.total_surface_num > 0):
            required = self.contact_surface_prefix[self.mpm.total_surface_num - 1]
        contact_count[0] = required
        overflow[0] = ti.cast(required > capacity, ti.i32)

    def _validate_lagged_friction_configuration(self):
        """Reject parameters the lagged IPC potential does not represent."""
        mu = float(self.friction.mu[0])
        mu_dynamic = float(self.friction.mu_dynamic[0])
        mu_static = float(self.friction.mu_static[0])
        mu_viscous = float(self.friction.mu_viscous[0])
        if mu != mu_dynamic or mu_dynamic != mu_static or mu_viscous != 0.0 or self.friction.profile_id != 0:
            raise RuntimeError(
                "lagged IPC friction supports only the equal-static/dynamic "
                "quadratic Coulomb potential; Stribeck, viscous, or "
                "stabilized-profile friction requires friction_mode="
                "'fully_implicit'"
            )

    def _cuda_monolithic_krylov_solver(self):
        """Match the two friction formulations' mathematical operators.

        Official lagged IPC uses projected Newton: elasticity, barrier, and
        frozen-friction local Hessians are PSD-projected before symmetric
        assembly, while dynamic inertia supplies the positive mass diagonal.
        Its CUDA solve therefore uses PCG.  Fully implicit friction retains
        the exact, generally nonsymmetric Jacobian and must use BiCGSTAB.
        """
        if self.friction_mode == "lagged":
            return "PCG"
        return "BiCGSTAB"

    def _validate_fully_implicit_configuration(self):
        if not config.DYNAMIC:
            raise RuntimeError("fully implicit IPC friction currently requires dynamic MPM")
        integration = np.asarray(self.mpm.integration, dtype=np.float64)
        if integration.size < 3 or not np.all(np.isfinite(integration[:3])) or np.any(integration[:3] <= 0.0):
            raise RuntimeError(
                "fully implicit IPC friction requires finite positive " "Newmark [alpha, beta, gamma] parameters"
            )
        # These are the two schemes derived explicitly in the paper.  The
        # surrounding MPM implementation does not currently expose the
        # paper's BDF2/SDIRK state history, so accepting arbitrary Newmark
        # triples here would make the advertised residual ambiguous.
        if np.allclose(integration[:3], [1.0, 0.5, 1.0], rtol=0.0, atol=1.0e-14):
            self.fully_implicit_integrator = "backward_euler"
        elif np.allclose(integration[:3], [0.5, 0.25, 0.5], rtol=0.0, atol=1.0e-14):
            self.fully_implicit_integrator = "trapezoidal"
        else:
            raise RuntimeError(
                "fully implicit IPC friction currently implements the "
                "paper's backward Euler [1, 0.5, 1] and trapezoidal "
                "[0.5, 0.25, 0.5] schemes only"
            )
        parameter_values = {
            "dt": float(self.mpm.dt),
            "dhat": float(self.barrier.dhat[0]),
            "kappa": float(self.barrier.kappa[0]),
            "epsv": float(self.friction.epsv[0]),
            "mu": float(self.friction.mu[0]),
            "dynamic_friction": float(self.friction.mu_dynamic[0]),
            "static_friction": float(self.friction.mu_static[0]),
            "viscous_friction": float(self.friction.mu_viscous[0]),
            "stribeck_velocity": float(self.friction.stribeck_velocity[0]),
        }
        for name in ("dt", "dhat", "kappa", "epsv"):
            value = parameter_values[name]
            if not np.isfinite(value) or value <= 0.0:
                raise RuntimeError(f"fully implicit IPC friction requires finite positive {name}")
        stribeck_velocity = parameter_values["stribeck_velocity"]
        if not np.isfinite(stribeck_velocity) or stribeck_velocity < 0.0:
            raise RuntimeError("fully implicit IPC friction requires finite nonnegative " "stribeck_velocity")
        if parameter_values["static_friction"] != parameter_values["dynamic_friction"] and stribeck_velocity <= 0.0:
            raise RuntimeError(
                "fully implicit IPC friction requires positive "
                "stribeck_velocity when static_friction differs from "
                "dynamic_friction"
            )
        for name in (
            "mu",
            "dynamic_friction",
            "static_friction",
            "viscous_friction",
        ):
            value = parameter_values[name]
            if not np.isfinite(value) or value < 0.0:
                raise RuntimeError("fully implicit IPC friction requires finite " f"nonnegative {name}")
        if self.mpm.n_body != 1 or self.enable_particle_contact:
            raise RuntimeError(
                "fully implicit IPC friction currently supports one MPM body " "against planar ground only"
            )
        if self.friction_iterations != 1:
            raise RuntimeError(
                "friction_iterations is only defined for lagged friction and " "must equal 1 in fully implicit mode"
            )
        if self.ground.num <= 0:
            raise RuntimeError("fully implicit IPC friction requires at least one planar ground")
        if self.mpm.dirichlet.num > 0:
            raise RuntimeError(
                "fully implicit IPC friction currently rejects Dirichlet "
                "constraints because contact blocks require coupled elimination"
            )
        normals = np.asarray(self.ground.np_norm, dtype=np.float64)
        velocities = np.asarray(self.ground.np_vel, dtype=np.float64)
        if not np.all(np.isfinite(normals)) or not np.all(np.isfinite(velocities)):
            raise RuntimeError("fully implicit IPC friction requires finite ground data")
        if not np.allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=1.0e-10, atol=1.0e-12):
            raise RuntimeError("fully implicit IPC friction requires unit planar-ground normals")
        if not np.allclose(velocities, 0.0, rtol=0.0, atol=1.0e-14):
            raise RuntimeError("fully implicit IPC friction currently supports stationary ground only")
        maximum_ground_contacts = int(self.mpm.total_surface_num * self.ground.num)
        if self.gbarrier.shape[0] < maximum_ground_contacts or self.gfriction.shape[0] < maximum_ground_contacts:
            raise RuntimeError(
                "fully implicit IPC friction requires ground barrier/friction "
                "buffers sized for every surface-point/ground pair"
            )

    @ti.func
    def _normal_terms(self, value, slot):
        energy = 0.0
        gradient = 0.0
        hessian = 0.0
        if ti.static(self.is_semi):
            multiplier = 0.0
            if slot >= 0:
                multiplier = self.semi_multiplier[slot]
            energy, gradient, hessian = semi_ipc_terms(value, multiplier, self.barrier.penalty[0])
        else:
            energy, gradient, hessian = self.barrier._terms(value)
        return energy, gradient, hessian

    @ti.func
    def compute_mu_lambda(self, value, measure, slot):
        """Frozen Coulomb weight for one geometrically weighted constraint."""
        _, gradient, _ = self._normal_terms(value, slot)
        return -point_contact_measure(measure) * self.friction.mu[0] * gradient

    @ti.func
    def add_barrier_triplet(self, row, col, value):
        scatter_hash_scalar(self.barrier_hash_matrix, row, col, value, config.DIM)

    @ti.func
    def add_friction_triplet(self, row, col, value):
        scatter_hash_scalar(self.friction_hash_matrix, row, col, value, config.DIM)

    def barrier_matrix(self, active_dof=None):
        active_dof = self.mpm.degree_of_freedom if active_dof is None else int(active_dof)
        self.barrier_hash_matrix.finalize_taichi_assembly()
        return self.barrier_hash_matrix.to_scipy(active_dof // config.DIM).tocsr()

    def friction_matrix(self, active_dof=None):
        active_dof = self.mpm.degree_of_freedom if active_dof is None else int(active_dof)
        self.friction_hash_matrix.finalize_taichi_assembly()
        return self.friction_hash_matrix.to_scipy(active_dof // config.DIM).tocsr()

    @ti.kernel
    def update_particle_pos(self, grid_disp: ti.template()):
        for s in range(self.mpm.total_surface_num):
            i = self.mpm.surface_id[s]
            disp = ti.Vector.zero(ti.f64, config.DIM)
            for j in range(self.mpm.offset[i]):
                grid_id = self.mpm.LnID[i, j]
                dofs = config.DIM * (self.mpm.node2dof[grid_id] - 1)
                disp += self.mpm.shape[i, j] * ti.Vector([grid_disp[dofs + d] for d in ti.static(range(config.DIM))])
            self.mpm.p_temp[s] = self.mpm.particle[i].x + disp

    @ti.kernel
    def initialize_hat_x(self):
        for s in range(self.mpm.total_surface_num):
            i = self.mpm.surface_id[s]
            self.hat_x[s] = self.mpm.particle[i].x

    def friction_outer_iteration_limit(self):
        if self.friction_iterations == -1:
            return self.friction_max_iterations
        return self.friction_iterations

    def refresh_friction_cache(self, grid_disp=None):
        """Rebuild lagged normals and normal-force weights at ``grid_disp``.

        ``hat_x`` deliberately is not touched here: it is the start-of-time-step
        reference and must remain fixed across all outer friction iterations.
        """
        if not self.activate_fric:
            self.curr_friction_contact_num = 0
            return 0
        if grid_disp is None:
            grid_disp = self.mpm.grid_disp
        self.update_particle_pos(grid_disp)
        self.particle_friction_initialize()
        self.ground_friction_initialize()
        self.curr_friction_contact_num = self.pfrictionNum[0] + self.gfrictionNum[0]
        return int(self.curr_friction_contact_num)

    def begin_friction_step(self, grid_disp=None):
        """Freeze the incremental-motion reference once, then seed the cache."""
        if not self.activate_fric:
            return 0
        self.adjoint_friction_valid[None] = 0
        self.initialize_hat_x()
        self.friction_reference_initialized = True
        return self.refresh_friction_cache(grid_disp)

    @ti.kernel
    def _backup_lagged_friction_for_adjoint(self):
        ground_count = self.gfrictionNum[0]
        particle_count = self.pfrictionNum[0]
        self.adjoint_gfriction_num[None] = ground_count
        self.adjoint_pfriction_num[None] = particle_count
        self.adjoint_friction_valid[None] = 1
        for contact in range(ground_count):
            self.adjoint_gfriction[contact] = self.gfriction[contact]
        for contact in range(particle_count):
            self.adjoint_pfriction[contact] = self.pfriction[contact]

    @ti.kernel
    def _restore_lagged_friction_for_adjoint(self):
        ground_count = self.adjoint_gfriction_num[None]
        particle_count = self.adjoint_pfriction_num[None]
        self.gfrictionNum[0] = ground_count
        self.pfrictionNum[0] = particle_count
        self.gfriction_overflow[0] = 0
        self.pfriction_overflow[0] = 0
        for contact in range(ground_count):
            self.gfriction[contact] = self.adjoint_gfriction[contact]
        for contact in range(particle_count):
            self.pfriction[contact] = self.adjoint_pfriction[contact]

    @ti.func
    def _body_cell_coord(self, pos):
        ix = ti.min(ti.max(int(ti.floor(pos[0] * self.body_inv_cell_size)), 0), self.body_cell_nx - 1)
        iy = 0
        iz = 0
        if ti.static(config.DIM >= 2):
            iy = ti.min(ti.max(int(ti.floor(pos[1] * self.body_inv_cell_size)), 0), self.body_cell_ny - 1)
        if ti.static(config.DIM >= 3):
            iz = ti.min(ti.max(int(ti.floor(pos[2] * self.body_inv_cell_size)), 0), self.body_cell_nz - 1)
        return ix, iy, iz

    @ti.func
    def _body_cell_id(self, ix, iy, iz):
        return ix + self.body_cell_nx * (iy + self.body_cell_ny * iz)

    @ti.func
    def _body_aabb_overlap(self, body_i, body_j):
        overlap = 1
        for d in ti.static(range(config.DIM)):
            if (
                self.body_max[body_i][d] < self.body_min[body_j][d]
                or self.body_max[body_j][d] < self.body_min[body_i][d]
            ):
                overlap = 0
        return overlap == 1

    @ti.func
    def _store_body_pair(self, body_i, body_j):
        master = ti.min(body_i, body_j)
        slave = ti.max(body_i, body_j)
        if master != slave:
            key = master * self.mpm.n_body + slave
            old = ti.atomic_add(self.body_pair_mark[key], 1)
            if old == 0:
                cur_pos = ti.atomic_add(self.body_pair_num[0], 1)
                if cur_pos < self.body_pair.shape[0]:
                    self.body_pair[cur_pos].master = master
                    self.body_pair[cur_pos].slave = slave
                else:
                    self.body_pair_overflow[0] = 1

    @ti.func
    def _point_in_body_aabb(self, pos, body):
        inside = 1
        for d in ti.static(range(config.DIM)):
            if pos[d] < self.body_min[body][d] or pos[d] > self.body_max[body][d]:
                inside = 0
        return inside == 1

    @ti.func
    def _store_particle_barrier(self, s, t, dist):
        cur_pos = ti.atomic_add(self.pbarrierNum[0], 1)
        if cur_pos < self.pbarrier.shape[0]:
            self.pbarrier[cur_pos].masterID = s
            self.pbarrier[cur_pos].slaveID = t
            self.pbarrier[cur_pos].distance = dist
            self.pbarrier[cur_pos].semi_slot = -1
        else:
            self.pbarrier_overflow[0] = 1

    @ti.func
    def _store_particle_friction(self, s, t, dist, normal):
        cur_pos = ti.atomic_add(self.pfrictionNum[0], 1)
        if cur_pos < self.pfriction.shape[0]:
            self.pfriction[cur_pos].masterID = s
            self.pfriction[cur_pos].slaveID = t
            measure = symmetric_contact_measure(self.mpm.surface_measure[s], self.mpm.surface_measure[t])
            slot = -1
            value = dist
            if ti.static(self.is_semi):
                key = ti.Vector([s, t, 0, -1])
                slot = semi_ipc_find(self.semi_state, self.semi_key, key, ti.static(self.semi_capacity))
                value = normal.dot(self.mpm.p_temp[s] - self.mpm.p_temp[t]) - self.barrier.activation_distance_term()
            self.pfriction[cur_pos].mu_lambda = self.compute_mu_lambda(value, measure, slot)
            self.pfriction[cur_pos].normal = normal
        else:
            self.pfriction_overflow[0] = 1

    @ti.kernel
    def _reset_body_bounds(self):
        expand = self.barrier.activation_distance_term()
        for b in range(self.mpm.n_body):
            for d in ti.static(range(config.DIM)):
                self.body_min[b][d] = 1.0e30
                self.body_max[b][d] = -1.0e30

    @ti.kernel
    def _accumulate_body_bounds(self):
        expand = self.barrier.activation_distance_term()
        for surface in range(self.mpm.total_surface_num):
            body = self.surface_body[surface]
            pos = self.mpm.p_temp[surface]
            for d in ti.static(range(config.DIM)):
                ti.atomic_min(self.body_min[body][d], pos[d] - expand)
                ti.atomic_max(self.body_max[body][d], pos[d] + expand)

    def update_body_bounds(self):
        self._reset_body_bounds()
        self._accumulate_body_bounds()

    @ti.kernel
    def _prepare_swept_body_bounds(self, grid_disp: ti.template()):
        clearance = 0.0
        if ti.static(not self.is_semi):
            clearance = self.barrier.dmin[0]
        for body in range(self.mpm.n_body):
            for d in ti.static(range(config.DIM)):
                self.body_min[body][d] = 1.0e30
                self.body_max[body][d] = -1.0e30
        for surface in range(self.mpm.total_surface_num):
            particle = self.mpm.surface_id[surface]
            body = self.surface_body[surface]
            disp = ti.Vector.zero(ti.f64, config.DIM)
            for local in range(self.mpm.offset[particle]):
                grid_id = self.mpm.LnID[particle, local]
                dofs = config.DIM * (self.mpm.node2dof[grid_id] - 1)
                disp += self.mpm.shape[particle, local] * ti.Vector(
                    [grid_disp[dofs + d] for d in ti.static(range(config.DIM))]
                )
            self.surface_sweep_disp[surface] = disp
            current = self.mpm.p_temp[surface]
            trial = current + disp
            for d in ti.static(range(config.DIM)):
                ti.atomic_min(self.body_min[body][d], ti.min(current[d], trial[d]) - clearance)
                ti.atomic_max(self.body_max[body][d], ti.max(current[d], trial[d]) + clearance)

    @ti.kernel
    def _body_pair_search_brust(self):
        previous_count = self.body_pair_num[0]
        for pair_id in range(previous_count):
            master = self.body_pair[pair_id].master
            slave = self.body_pair[pair_id].slave
            self.body_pair_mark[master * self.mpm.n_body + slave] = 0
        self.body_pair_num[0] = 0
        self.body_pair_overflow[0] = 0
        self.body_cell_overflow[0] = 0
        if ti.static(not self.enable_particle_contact):
            return
        # ponytail: explicit small-body/overflow fallback; LinkedCell is the
        # production default and avoids this quadratic candidate scan.
        for master in range(self.mpm.n_body):
            if self.body_surface_count[master] > 0:
                for slave in range(master + 1, self.mpm.n_body):
                    if self.body_surface_count[slave] > 0 and self._body_aabb_overlap(master, slave):
                        self._store_body_pair(master, slave)
        if self.body_pair_num[0] > self.body_pair.shape[0]:
            self.body_pair_num[0] = self.body_pair.shape[0]

    @ti.kernel
    def _body_pair_search_linked_cell(self):
        previous_count = self.body_pair_num[0]
        for pair_id in range(previous_count):
            master = self.body_pair[pair_id].master
            slave = self.body_pair[pair_id].slave
            self.body_pair_mark[master * self.mpm.n_body + slave] = 0
        self.body_pair_num[0] = 0
        self.body_pair_overflow[0] = 0
        self.body_cell_overflow[0] = 0
        for cell_id in range(self.body_cell_count.shape[0]):
            self.body_cell_count[cell_id] = 0
        if ti.static(not self.enable_particle_contact):
            return
        for body in range(self.mpm.n_body):
            if self.body_surface_count[body] > 0:
                min_x, min_y, min_z = self._body_cell_coord(self.body_min[body])
                max_x, max_y, max_z = self._body_cell_coord(self.body_max[body])
                for cx in range(min_x, max_x + 1):
                    for cy in range(min_y, max_y + 1):
                        for cz in range(min_z, max_z + 1):
                            cell_id = self._body_cell_id(cx, cy, cz)
                            local = ti.atomic_add(self.body_cell_count[cell_id], 1)
                            if local < self.body_per_cell:
                                self.body_cell[cell_id * self.body_per_cell + local] = body
                            else:
                                self.body_cell_overflow[0] = 1
        for cell_id in range(self.body_cell_count.shape[0]):
            stored = ti.min(self.body_cell_count[cell_id], self.body_per_cell)
            for i in range(stored):
                body_i = self.body_cell[cell_id * self.body_per_cell + i]
                for j in range(i + 1, stored):
                    body_j = self.body_cell[cell_id * self.body_per_cell + j]
                    if body_i != body_j and self._body_aabb_overlap(body_i, body_j):
                        self._store_body_pair(body_i, body_j)
        if self.body_pair_num[0] > self.body_pair.shape[0]:
            self.body_pair_num[0] = self.body_pair.shape[0]

    def update_body_pair_table(self):
        self.update_body_bounds()
        self._search_body_pairs()

    def _search_body_pairs(self):
        if self.contact_search == "Brust":
            self.last_particle_search_backend = "Brust"
            self._body_pair_search_brust()
        else:
            self.last_particle_search_backend = self.contact_search
            self._body_pair_search_linked_cell()
            if int(self.body_cell_overflow[0]) != 0:
                self.last_particle_search_backend = "Brust"
                self._body_pair_search_brust()
        if int(self.body_pair_overflow[0]) != 0:
            raise RuntimeError("body-pair contact capacity exceeded; increase body_pair_capacity")

    def update_swept_body_pair_table(self, grid_disp):
        self._prepare_swept_body_bounds(grid_disp)
        self._search_body_pairs()

    @ti.kernel
    def _count_surface_body_contacts(self):
        self.surface_body_contact_num[0] = 0
        self.surface_body_contact_overflow[0] = 0
        for surface in range(self.mpm.total_surface_num):
            self.surface_body_prefix[surface] = 0
            self.surface_body_cursor[surface] = 0
        if ti.static(not self.enable_particle_contact):
            return
        for pair_id, local, side in ti.ndrange(self.body_pair_num[0], self.max_body_surface_count, 2):
            master = self.body_pair[pair_id].master
            slave = self.body_pair[pair_id].slave
            source = master
            target = slave
            if side == 1:
                source = slave
                target = master
            if local < self.body_surface_count[source]:
                surface = self.body_surface_start[source] + local
                if self._point_in_body_aabb(self.mpm.p_temp[surface], target):
                    ti.atomic_add(self.surface_body_prefix[surface], 1)

    @ti.kernel
    def _finalize_surface_body_contact_count(self, capacity: ti.i32):
        required = 0
        if ti.static(self.mpm.total_surface_num > 0):
            required = self.surface_body_prefix[self.mpm.total_surface_num - 1]
        self.surface_body_contact_num[0] = required
        self.surface_body_contact_overflow[0] = ti.cast(required > capacity, ti.i32)

    @ti.kernel
    def _fill_surface_body_contacts(self):
        if ti.static(not self.enable_particle_contact):
            return
        for pair_id, local, side in ti.ndrange(self.body_pair_num[0], self.max_body_surface_count, 2):
            master = self.body_pair[pair_id].master
            slave = self.body_pair[pair_id].slave
            source = master
            target = slave
            if side == 1:
                source = slave
                target = master
            if local < self.body_surface_count[source]:
                surface = self.body_surface_start[source] + local
                if self._point_in_body_aabb(self.mpm.p_temp[surface], target):
                    begin = 0
                    if surface > 0:
                        begin = self.surface_body_prefix[surface - 1]
                    slot = begin + ti.atomic_add(self.surface_body_cursor[surface], 1)
                    self.surface_body_contact[slot].surfaceID = surface
                    self.surface_body_contact[slot].bodyID = target

    @ti.kernel
    def _sort_surface_body_contacts(self):
        for surface in range(self.mpm.total_surface_num):
            begin = 0
            if surface > 0:
                begin = self.surface_body_prefix[surface - 1]
            count = self.surface_body_prefix[surface] - begin
            # ponytail: coordination is intentionally small; replace this
            # per-surface insertion sort only if profiling shows large degree.
            for local in range(1, count):
                body = self.surface_body_contact[begin + local].bodyID
                cursor = local
                while cursor > 0 and self.surface_body_contact[begin + cursor - 1].bodyID > body:
                    self.surface_body_contact[begin + cursor].surfaceID = surface
                    self.surface_body_contact[begin + cursor].bodyID = self.surface_body_contact[
                        begin + cursor - 1
                    ].bodyID
                    cursor -= 1
                self.surface_body_contact[begin + cursor].surfaceID = surface
                self.surface_body_contact[begin + cursor].bodyID = body

    def build_surface_body_contact_table(self):
        self._count_surface_body_contacts()
        self._scan_surface_prefix(self.surface_body_prefix)
        self._finalize_surface_body_contact_count(self.surface_body_contact.shape[0])
        if int(self.surface_body_contact_overflow[0]) != 0:
            raise RuntimeError("surface/body contact capacity exceeded; increase " "point_body_contact_capacity")
        self._fill_surface_body_contacts()
        self._sort_surface_body_contacts()

    @ti.kernel
    def _compute_minimum_candidate_distance(self):
        """Measure the actual gap over every currently relevant constraint."""
        self.minimum_candidate_distance[0] = 1.0e30
        self.candidate_state_invalid[0] = 0
        for surface in range(self.mpm.total_surface_num):
            point = self.mpm.p_temp[surface]
            for component in ti.static(range(config.DIM)):
                if not (point[component] == point[component] and ti.abs(point[component]) < 1.0e300):
                    ti.atomic_max(self.candidate_state_invalid[0], 1)
            for wall in range(self.ground.num):
                distance = self.ground.distance(wall, point)
                if distance == distance and ti.abs(distance) < 1.0e300:
                    ti.atomic_min(self.minimum_candidate_distance[0], distance)
                else:
                    ti.atomic_max(self.candidate_state_invalid[0], 1)
        if ti.static(self.enable_particle_contact):
            for contact in range(self.surface_body_contact_num[0]):
                surface = self.surface_body_contact[contact].surfaceID
                target_body = self.surface_body_contact[contact].bodyID
                target_start = self.body_surface_start[target_body]
                target_count = self.body_surface_count[target_body]
                for local_target in range(target_count):
                    target = target_start + local_target
                    if surface < target:
                        distance = (self.mpm.p_temp[surface] - self.mpm.p_temp[target]).norm()
                        if distance == distance and distance < 1.0e300:
                            ti.atomic_min(self.minimum_candidate_distance[0], distance)
                        else:
                            ti.atomic_max(self.candidate_state_invalid[0], 1)

    def validate_strict_feasibility(self, grid_disp=None):
        """Reject an initially intersecting IPC state instead of hiding it.

        IPC's barrier is only defined above ``dmin``. Subsequent Newton trial
        steps remain in that domain through CCD; this check establishes the
        invariant at the beginning of a time step.
        """
        if grid_disp is None:
            grid_disp = self.mpm.grid_disp
        self.update_particle_pos(grid_disp)
        if self.enable_particle_contact:
            self.update_body_pair_table()
            self.build_surface_body_contact_table()
        self._compute_minimum_candidate_distance()
        minimum_distance = float(self.minimum_candidate_distance[0])
        dmin = 0.0 if self.is_semi else float(self.barrier.dmin[0])
        safe_floor = max(
            1.0e-12 * float(self.barrier.dhat[0]),
            1.0e-14,
        )
        if (
            int(self.candidate_state_invalid[0]) != 0
            or not np.isfinite(minimum_distance)
            or minimum_distance <= dmin + safe_floor
        ):
            raise RuntimeError(
                "IPC requires a strictly feasible positive starting gap "
                f"above dmin (minimum={minimum_distance:.6e}, "
                f"dmin={dmin:.6e}, margin={safe_floor:.6e})"
            )
        return minimum_distance

    @ti.kernel
    def _count_particle_barrier_contacts_stable(self):
        """Count PP constraints per source surface in a canonical order."""
        for s in range(self.mpm.total_surface_num):
            count = 0
            if ti.static(self.enable_particle_contact):
                current_ipos = self.mpm.p_temp[s]
                begin = 0
                if s > 0:
                    begin = self.surface_body_prefix[s - 1]
                end = self.surface_body_prefix[s]
                for contact in range(begin, end):
                    target_body = self.surface_body_contact[contact].bodyID
                    target_start = self.body_surface_start[target_body]
                    target_count = self.body_surface_count[target_body]
                    for local_t in range(target_count):
                        t = target_start + local_t
                        if s < t:
                            dist = (current_ipos - self.mpm.p_temp[t]).norm()
                            if ti.static(self.is_semi):
                                key = ti.Vector([s, t, 0, -1])
                                slot = semi_ipc_find(
                                    self.semi_state,
                                    self.semi_key,
                                    key,
                                    ti.static(self.semi_capacity),
                                )
                                multiplier = 0.0
                                if slot >= 0:
                                    multiplier = self.semi_multiplier[slot]
                                gap = dist - self.barrier.activation_distance_term()
                                if multiplier - self.barrier.penalty[0] * gap >= 0.0:
                                    count += 1
                            else:
                                if dist <= self.barrier.dmin[0]:
                                    ti.atomic_max(self.barrier_infeasible[0], 1)
                                elif dist < self.barrier.activation_distance_term():
                                    count += 1
            self.contact_surface_prefix[s] = count

    @ti.kernel
    def _fill_particle_barrier_contacts_stable(self):
        for s in range(self.mpm.total_surface_num):
            output = 0
            if s > 0:
                output = self.contact_surface_prefix[s - 1]
            if ti.static(self.enable_particle_contact):
                current_ipos = self.mpm.p_temp[s]
                begin = 0
                if s > 0:
                    begin = self.surface_body_prefix[s - 1]
                end = self.surface_body_prefix[s]
                for contact in range(begin, end):
                    target_body = self.surface_body_contact[contact].bodyID
                    target_start = self.body_surface_start[target_body]
                    target_count = self.body_surface_count[target_body]
                    for local_t in range(target_count):
                        t = target_start + local_t
                        if s < t:
                            delta = current_ipos - self.mpm.p_temp[t]
                            dist = delta.norm()
                            active = dist > self.barrier.dmin[0] and dist < self.barrier.activation_distance_term()
                            key = ti.Vector([s, t, 0, -1])
                            if ti.static(self.is_semi):
                                slot = semi_ipc_find(
                                    self.semi_state,
                                    self.semi_key,
                                    key,
                                    ti.static(self.semi_capacity),
                                )
                                multiplier = 0.0
                                if slot >= 0:
                                    multiplier = self.semi_multiplier[slot]
                                gap = dist - self.barrier.activation_distance_term()
                                active = multiplier - self.barrier.penalty[0] * gap >= 0.0
                            if active:
                                self.pbarrier[output].masterID = s
                                self.pbarrier[output].slaveID = t
                                self.pbarrier[output].distance = dist
                                self.pbarrier[output].semi_slot = -1
                                if ti.static(self.is_semi):
                                    slot = semi_ipc_find_or_insert(
                                        self.semi_state,
                                        self.semi_key,
                                        self.semi_multiplier,
                                        self.semi_count,
                                        key,
                                        ti.static(self.semi_capacity),
                                    )
                                    if slot >= 0:
                                        normal = delta / ti.max(dist, 1.0e-15)
                                        previous = self.semi_normal[slot]
                                        if previous.norm_sqr() > 0.0 and previous.dot(normal) < 0.0:
                                            normal = -normal
                                        self.semi_normal[slot] = normal
                                    else:
                                        self.semi_overflow[None] = 1
                                    self.pbarrier[output].semi_slot = slot
                                output += 1

    def point_point_distance(self):
        self.update_body_pair_table()
        # Preserve the configured broad-phase capacity contract and the table
        # consumed by strict-feasibility diagnostics.  Its atomic order is not
        # used to assign final contact ids.
        self.build_surface_body_contact_table()
        self._count_particle_barrier_contacts_stable()
        self._scan_contact_surface_counts()
        self._finalize_stable_contact_count(
            self.pbarrierNum,
            self.pbarrier_overflow,
            self.pbarrier.shape[0],
        )
        if int(self.pbarrier_overflow[0]) != 0:
            raise RuntimeError("particle barrier contact capacity exceeded; increase barrier_set[0]")
        self._fill_particle_barrier_contacts_stable()

    @ti.kernel
    def _count_ground_barrier_contacts_stable(self):
        for s in range(self.mpm.total_surface_num):
            count = 0
            current_pos = self.mpm.p_temp[s]
            for wall in range(self.ground.num):
                dist = self.ground.distance(wall, current_pos)
                if ti.static(self.is_semi):
                    key = ti.Vector([s, wall, 1, -1])
                    slot = semi_ipc_find(self.semi_state, self.semi_key, key, ti.static(self.semi_capacity))
                    multiplier = 0.0
                    if slot >= 0:
                        multiplier = self.semi_multiplier[slot]
                    gap = dist - self.barrier.activation_distance_term()
                    if multiplier - self.barrier.penalty[0] * gap >= 0.0:
                        count += 1
                else:
                    if dist <= self.barrier.dmin[0]:
                        ti.atomic_max(self.barrier_infeasible[0], 1)
                    elif dist < self.barrier.activation_distance_term():
                        count += 1
            self.contact_surface_prefix[s] = count

    @ti.kernel
    def _fill_ground_barrier_contacts_stable(self):
        for s in range(self.mpm.total_surface_num):
            output = 0
            if s > 0:
                output = self.contact_surface_prefix[s - 1]
            current_pos = self.mpm.p_temp[s]
            for wall in range(self.ground.num):
                dist = self.ground.distance(wall, current_pos)
                active = dist > self.barrier.dmin[0] and dist < self.barrier.activation_distance_term()
                key = ti.Vector([s, wall, 1, -1])
                if ti.static(self.is_semi):
                    slot = semi_ipc_find(self.semi_state, self.semi_key, key, ti.static(self.semi_capacity))
                    multiplier = 0.0
                    if slot >= 0:
                        multiplier = self.semi_multiplier[slot]
                    gap = dist - self.barrier.activation_distance_term()
                    active = multiplier - self.barrier.penalty[0] * gap >= 0.0
                if active:
                    self.gbarrier[output].surfaceID = s
                    self.gbarrier[output].wallID = wall
                    self.gbarrier[output].distance = dist
                    self.gbarrier[output].semi_slot = -1
                    if ti.static(self.is_semi):
                        slot = semi_ipc_find_or_insert(
                            self.semi_state,
                            self.semi_key,
                            self.semi_multiplier,
                            self.semi_count,
                            key,
                            ti.static(self.semi_capacity),
                        )
                        if slot >= 0:
                            self.semi_normal[slot] = self.ground.norm[wall]
                        else:
                            self.semi_overflow[None] = 1
                        self.gbarrier[output].semi_slot = slot
                    output += 1

    def point_ground_distance(self):
        self._count_ground_barrier_contacts_stable()
        self._scan_contact_surface_counts()
        self._finalize_stable_contact_count(
            self.gbarrierNum,
            self.gbarrier_overflow,
            self.gbarrier.shape[0],
        )
        if int(self.gbarrier_overflow[0]) != 0:
            raise RuntimeError("ground barrier contact capacity exceeded; increase barrier_set[1]")
        self._fill_ground_barrier_contacts_stable()

    def rebuild_barrier_contacts(self):
        """Build the active set and report strict IPC feasibility.

        Contacts at or below ``dmin`` are not silently discarded. They make
        the trial invalid, matching the infinite domain boundary of the
        barrier potential.
        """
        self.barrier_infeasible[0] = 0
        if self.is_semi:
            self.semi_overflow[None] = 0
        self.point_ground_distance()
        self.point_point_distance()
        if self.is_semi and int(self.semi_overflow[None]) != 0:
            raise RuntimeError("MPM SemiIPC multiplier hash capacity is too small")
        return int(self.barrier_infeasible[0]) == 0

    @ti.kernel
    def _update_semi_multipliers(self):
        self.semi_constraint_violation[None] = 0.0
        for slot in range(self.semi_capacity):
            if self.semi_state[slot] == 0:
                key = self.semi_key[slot]
                gap = 0.0
                if key[2] == 0:
                    gap = (
                        self.mpm.p_temp[key[0]] - self.mpm.p_temp[key[1]]
                    ).norm() - self.barrier.activation_distance_term()
                else:
                    gap = (
                        self.ground.distance(key[1], self.mpm.p_temp[key[0]]) - self.barrier.activation_distance_term()
                    )
                self.semi_multiplier[slot] = semi_ipc_update_multiplier(
                    gap, self.semi_multiplier[slot], self.barrier.penalty[0]
                )
                ti.atomic_max(self.semi_constraint_violation[None], ti.max(-gap, 0.0))

    def accept_update(self):
        if not self.is_semi:
            return
        self.update_particle_pos(self.mpm.grid_disp)
        self.rebuild_barrier_contacts()
        self._update_semi_multipliers()

    def contact_converged(self):
        return not self.is_semi or float(self.semi_constraint_violation[None]) <= self.barrier.constraint_tolerance

    @ti.kernel
    def _count_ground_friction_contacts_stable(self):
        for s in range(self.mpm.total_surface_num):
            count = 0
            current_pos = self.mpm.p_temp[s]
            for wall in range(self.ground.num):
                dist = self.ground.distance(wall, current_pos)
                active = dist > self.barrier.dmin[0] and dist < self.barrier.activation_distance_term()
                if ti.static(self.is_semi):
                    key = ti.Vector([s, wall, 1, -1])
                    slot = semi_ipc_find(self.semi_state, self.semi_key, key, ti.static(self.semi_capacity))
                    multiplier = 0.0
                    if slot >= 0:
                        multiplier = self.semi_multiplier[slot]
                    gap = dist - self.barrier.activation_distance_term()
                    active = multiplier - self.barrier.penalty[0] * gap > 0.0
                if active:
                    count += 1
            self.contact_surface_prefix[s] = count

    @ti.kernel
    def _fill_ground_friction_contacts_stable(self):
        for s in range(self.mpm.total_surface_num):
            output = 0
            if s > 0:
                output = self.contact_surface_prefix[s - 1]
            current_pos = self.mpm.p_temp[s]
            for wall in range(self.ground.num):
                dist = self.ground.distance(wall, current_pos)
                active = dist > self.barrier.dmin[0] and dist < self.barrier.activation_distance_term()
                slot = -1
                value = dist
                if ti.static(self.is_semi):
                    key = ti.Vector([s, wall, 1, -1])
                    slot = semi_ipc_find(self.semi_state, self.semi_key, key, ti.static(self.semi_capacity))
                    value = dist - self.barrier.activation_distance_term()
                    multiplier = 0.0
                    if slot >= 0:
                        multiplier = self.semi_multiplier[slot]
                    active = multiplier - self.barrier.penalty[0] * value > 0.0
                if active:
                    self.gfriction[output].surfaceID = s
                    self.gfriction[output].wallID = wall
                    self.gfriction[output].mu_lambda = self.compute_mu_lambda(value, self.mpm.surface_measure[s], slot)
                    output += 1

    def ground_friction_initialize(self):
        self._count_ground_friction_contacts_stable()
        self._scan_contact_surface_counts()
        self._finalize_stable_contact_count(
            self.gfrictionNum,
            self.gfriction_overflow,
            self.gfriction.shape[0],
        )
        if int(self.gfriction_overflow[0]) != 0:
            raise RuntimeError("ground friction contact capacity exceeded; increase friction_set[1]")
        self._fill_ground_friction_contacts_stable()

    @ti.kernel
    def _count_particle_friction_contacts_stable(self):
        for s in range(self.mpm.total_surface_num):
            count = 0
            if ti.static(self.enable_particle_contact):
                current_ipos = self.mpm.p_temp[s]
                begin = 0
                if s > 0:
                    begin = self.surface_body_prefix[s - 1]
                end = self.surface_body_prefix[s]
                for contact in range(begin, end):
                    target_body = self.surface_body_contact[contact].bodyID
                    target_start = self.body_surface_start[target_body]
                    target_count = self.body_surface_count[target_body]
                    for local_t in range(target_count):
                        t = target_start + local_t
                        if s < t:
                            delta = current_ipos - self.mpm.p_temp[t]
                            dist = delta.norm()
                            active = dist > self.barrier.dmin[0] and dist < self.barrier.activation_distance_term()
                            if ti.static(self.is_semi):
                                key = ti.Vector([s, t, 0, -1])
                                slot = semi_ipc_find(
                                    self.semi_state,
                                    self.semi_key,
                                    key,
                                    ti.static(self.semi_capacity),
                                )
                                multiplier = 0.0
                                if slot >= 0:
                                    multiplier = self.semi_multiplier[slot]
                                normal = delta / ti.max(dist, 1.0e-12)
                                if slot >= 0:
                                    normal = self.semi_normal[slot]
                                gap = (
                                    normal.dot(current_ipos - self.mpm.p_temp[t])
                                    - self.barrier.activation_distance_term()
                                )
                                active = multiplier - self.barrier.penalty[0] * gap > 0.0
                            if active:
                                count += 1
            self.contact_surface_prefix[s] = count

    @ti.kernel
    def _fill_particle_friction_contacts_stable(self):
        for s in range(self.mpm.total_surface_num):
            output = 0
            if s > 0:
                output = self.contact_surface_prefix[s - 1]
            if ti.static(self.enable_particle_contact):
                current_ipos = self.mpm.p_temp[s]
                begin = 0
                if s > 0:
                    begin = self.surface_body_prefix[s - 1]
                end = self.surface_body_prefix[s]
                for contact in range(begin, end):
                    target_body = self.surface_body_contact[contact].bodyID
                    target_start = self.body_surface_start[target_body]
                    target_count = self.body_surface_count[target_body]
                    for local_t in range(target_count):
                        t = target_start + local_t
                        if s < t:
                            delta = current_ipos - self.mpm.p_temp[t]
                            dist = delta.norm()
                            active = dist > self.barrier.dmin[0] and dist < self.barrier.activation_distance_term()
                            slot = -1
                            normal = delta / ti.max(dist, 1.0e-12)
                            value = dist
                            if ti.static(self.is_semi):
                                key = ti.Vector([s, t, 0, -1])
                                slot = semi_ipc_find(
                                    self.semi_state,
                                    self.semi_key,
                                    key,
                                    ti.static(self.semi_capacity),
                                )
                                multiplier = 0.0
                                if slot >= 0:
                                    multiplier = self.semi_multiplier[slot]
                                    normal = self.semi_normal[slot]
                                value = normal.dot(delta) - self.barrier.activation_distance_term()
                                active = multiplier - self.barrier.penalty[0] * value > 0.0
                            if active:
                                self.pfriction[output].masterID = s
                                self.pfriction[output].slaveID = t
                                measure = symmetric_contact_measure(
                                    self.mpm.surface_measure[s],
                                    self.mpm.surface_measure[t],
                                )
                                self.pfriction[output].mu_lambda = self.compute_mu_lambda(value, measure, slot)
                                self.pfriction[output].normal = normal
                                output += 1

    def particle_friction_initialize(self):
        self.update_body_pair_table()
        self.build_surface_body_contact_table()
        self._count_particle_friction_contacts_stable()
        self._scan_contact_surface_counts()
        self._finalize_stable_contact_count(
            self.pfrictionNum,
            self.pfriction_overflow,
            self.pfriction.shape[0],
        )
        if int(self.pfriction_overflow[0]) != 0:
            raise RuntimeError("particle friction contact capacity exceeded; increase friction_set[0]")
        self._fill_particle_friction_contacts_stable()

    def assemble_current_system(
        self,
        grid_disp=None,
        need_matrix=True,
        project_spd=None,
        exact_plastic_tangent=False,
        rebuild_contacts=True,
    ):
        """Assemble and finalize the standalone MPM system."""
        source_kwargs = {
            "need_matrix": need_matrix,
            "project_spd": project_spd,
            "exact_plastic_tangent": exact_plastic_tangent,
        }
        if not rebuild_contacts:
            source_kwargs["rebuild_contacts"] = False
        active_dof = self.assemble_current_sources(grid_disp, **source_kwargs)
        if not bool(need_matrix):
            return None
        if self.cuda_monolithic_solver:
            return self._assemble_cuda_monolithic_matrix(active_dof)
        raise RuntimeError(
            "SoftParticle IPC requires the monolithic Taichi HashTriplet "
            "path; host CSR assembly is not a runtime backend"
        )

    def assemble_current_sources(
        self,
        grid_disp=None,
        need_matrix=True,
        project_spd=None,
        exact_plastic_tangent=False,
        rebuild_contacts=True,
    ):
        """Assemble unfinalized MPM residual and matrix sources.

        Fully implicit Armijo trials use ``need_matrix=False``.  In that mode
        all force/contact state is refreshed exactly as in a Newton assembly,
        while the existing GPU HashTriplet buffers are neither reset nor
        written, finalized, reduced, or copied to SciPy.

        ``project_spd=False`` is the unprojected residual Jacobian used by an
        implicit adjoint.  The default preserves the forward solver's
        projected-Newton behavior.
        """
        if grid_disp is None:
            grid_disp = self.mpm.grid_disp
        need_matrix = bool(need_matrix)
        if project_spd is None:
            project_spd = self.friction_mode == "lagged"
        project_spd = bool(project_spd)
        active_dof = self.mpm.active_dof
        self.mpm.rhs.fill(0)
        if need_matrix:
            self.mpm.hash_matrix.reset_system()
            if self.activate_barrier:
                self.barrier_hash_matrix.reset_system()
        if self.activate_fric:
            self.friction_grad.fill(0)
            if need_matrix:
                self.friction_hash_matrix.reset_system()

        self.mpm.assemble_inertia_force(
            active_dof,
            self.mpm.damping,
            self.mpm.gravity,
            self.mpm.integration,
            grid_disp,
        )
        self.mpm.assemble_material_force(active_dof, grid_disp)
        if need_matrix:
            self.mpm.assemble_stiffness_matrix_hash(
                active_dof,
                grid_disp,
                project_spd=project_spd,
                exact_plastic_tangent=exact_plastic_tangent,
            )
            if config.DYNAMIC:
                self.mpm.assemble_mass_matrix_hash()

        self.update_particle_pos(grid_disp)
        if rebuild_contacts and not self.rebuild_barrier_contacts():
            raise IPCInfeasibleContactState("IPC trial crossed the configured minimum distance dmin")
        if self.is_semi:
            self.semi_constraint_violation[None] = 0.0
        self.curr_barrier_contact_num = self.pbarrierNum[0] + self.gbarrierNum[0]
        if self.curr_barrier_contact_num > 0:
            self._assemble_ground_barrier_system(need_matrix)
            self._assemble_particle_barrier_system(
                need_matrix,
                project_spd,
            )
        if self.activate_fric and self.friction_mode == "fully_implicit":
            # Current contact set and normal-force magnitude are part of the
            # residual, so every Newton and line-search evaluation rebuilds it.
            self.ground_friction_initialize()
            self.pfrictionNum[0] = 0
            self.curr_friction_contact_num = self.gfrictionNum[0]
        if self.curr_friction_contact_num > 0:
            if self.friction_mode == "fully_implicit":
                self._assemble_ground_fully_implicit_friction_system(grid_disp, need_matrix)
            else:
                self.assemble_ground_friction_matrix()
                self.assemble_particle_friction_matrix()
            add_field(
                active_dof,
                self.mpm.rhs,
                self.mpm.rhs,
                self.friction_grad,
            )

        if self.mpm.neumann.num > 0:
            self.mpm.apply_neumann()
        return active_dof

    def _solve_equilibrium_adjoint(self, loss_gradient, grid_disp, allow_plastic):
        if self.is_semi:
            raise ValueError("Direct IPC-MPM adjoint currently requires BarrierIPC")
        self._validate_differentiable_friction()
        is_plastic = bool(self.mpm.is_finite_strain_plastic)
        if is_plastic and not allow_plastic:
            raise ValueError("elastic IPC adjoint does not include plastic history variables")
        if is_plastic and type(self.mpm.material).__name__ not in {
            "FiniteStrainDruckerPragerModel",
            "FiniteStrainVonMisesModel",
        }:
            raise ValueError(
                "plastic IPC-MPM equilibrium adjoint supports Drucker-Prager "
                "and von Mises; MCC is intentionally excluded"
            )
        replay_record = int(self.__dict__.get("adjoint_trajectory_record", -1))
        if replay_record >= 0:
            self._load_trajectory_contact_state(replay_record)
            self.curr_barrier_contact_num = int(
                self.trajectory_gbarrier_num[replay_record] + self.trajectory_pbarrier_num[replay_record]
            )
            if self.activate_fric:
                self.curr_friction_contact_num = int(
                    self.trajectory_gfriction_num[replay_record] + self.trajectory_pfriction_num[replay_record]
                )
        elif self.activate_fric and int(self.adjoint_friction_valid[None]) != 0:
            self._restore_lagged_friction_for_adjoint()
            self.curr_friction_contact_num = int(self.adjoint_gfriction_num[None] + self.adjoint_pfriction_num[None])
        matrix = self.assemble_current_system(
            grid_disp,
            need_matrix=True,
            project_spd=False,
            exact_plastic_tangent=is_plastic,
            rebuild_contacts=replay_record < 0,
        )
        transpose = not matrix.matrix_symmetric
        matrix.solver = "PCG" if matrix.matrix_symmetric else "BiCGSTAB"
        self.mpm.incre_resolution.fill(0.0)
        result = matrix.solve_flat_system(
            loss_gradient,
            self.mpm.incre_resolution,
            active_nodes=self.mpm.active_dof // config.DIM,
            tol=self.mpm.linear_solver_tolerance,
            maxiter=self.mpm.linear_solver_max_iters,
            return_solution=False,
            transpose=transpose,
            fallback_to_bicgstab=matrix.matrix_symmetric,
        )
        self.last_adjoint_result = result
        if not result["converged"]:
            raise RuntimeError(
                "Direct IPC-MPM adjoint solve did not converge: "
                f"residual={result['residual']:.6e}, iterations={result['iterations']}"
            )
        return self.mpm.incre_resolution

    def _validate_differentiable_friction(self):
        if self.friction_mode == "fully_implicit":
            raise ValueError(
                "Direct differentiable IPC-MPM supports friction_mode='lagged' "
                "only; fully implicit friction is not implemented"
            )

    def solve_elastic_adjoint(self, loss_gradient, grid_disp=None):
        """Solve the exact elastic pre-commit equilibrium adjoint."""
        return self._solve_equilibrium_adjoint(
            loss_gradient,
            grid_disp,
            allow_plastic=False,
        )

    def solve_plastic_equilibrium_adjoint(self, loss_gradient, grid_disp=None):
        """Solve one exact equilibrium adjoint with frozen prior history."""
        if not self.mpm.is_finite_strain_plastic:
            raise ValueError("plastic equilibrium adjoint requires a finite-strain plastic material")
        return self._solve_equilibrium_adjoint(
            loss_gradient,
            grid_disp,
            allow_plastic=True,
        )

    @ti.kernel
    def _differentiate_elastic_gravity(self):
        self.gravity_vjp[None] = ti.Vector.zero(ti.f64, config.DIM)
        for grid_id in self.mpm.grid:
            mass = self.mpm.grid[grid_id].m
            block = self.mpm.node2dof[grid_id] - 1
            if mass > self.mpm.val_lim and block >= 0:
                for component in ti.static(range(config.DIM)):
                    free = True
                    if ti.static(self.mpm.dirichlet.num > 0):
                        free = self.mpm.dirichlet.node[config.DIM * grid_id + component] == 0
                    if free:
                        ti.atomic_add(
                            self.gravity_vjp[None][component],
                            mass * self.mpm.incre_resolution[config.DIM * block + component],
                        )

    @ti.kernel
    def _differentiate_lagged_friction_parameters(self):
        self.friction_parameter_vjp[None] = ti.Vector.zero(ti.f64, 4)
        mu = self.friction.mu[0]
        for contact in range(self.gfrictionNum[0]):
            surface = self.gfriction[contact].surfaceID
            particle = self.mpm.surface_id[surface]
            wall = self.gfriction[contact].wallID
            mu_lambda = self.gfriction[contact].mu_lambda
            if mu > 0.0 and mu_lambda > 0.0:
                point_adjoint = ti.Vector.zero(ti.f64, config.DIM)
                for local in range(self.mpm.offset[particle]):
                    node = self.mpm.LnID[particle, local]
                    block = self.mpm.node2dof[node] - 1
                    if block >= 0:
                        dofs = config.DIM * block
                        point_adjoint += self.mpm.shape[particle, local] * ti.Vector(
                            [self.mpm.incre_resolution[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                tangent = self.ground.tangent_operator(wall)
                relative_displacement = (
                    self.mpm.p_temp[surface] - self.hat_x[surface] - self.ground.vel[wall] * self.mpm.dt
                )
                tangent_velocity = tangent.transpose() @ relative_displacement / self.mpm.dt
                point_force = mu_lambda * self.friction.grad_term(tangent_velocity.norm()) * tangent @ tangent_velocity
                ti.atomic_add(
                    self.friction_parameter_vjp[None][0],
                    -point_adjoint.dot(point_force) / mu,
                )

        for contact in range(self.pfrictionNum[0]):
            master_surface = self.pfriction[contact].masterID
            slave_surface = self.pfriction[contact].slaveID
            master = self.mpm.surface_id[master_surface]
            slave = self.mpm.surface_id[slave_surface]
            mu_lambda = self.pfriction[contact].mu_lambda
            if mu > 0.0 and mu_lambda > 0.0:
                relative_adjoint = ti.Vector.zero(ti.f64, config.DIM)
                for local in range(self.mpm.offset[master]):
                    node = self.mpm.LnID[master, local]
                    block = self.mpm.node2dof[node] - 1
                    if block >= 0:
                        dofs = config.DIM * block
                        relative_adjoint += self.mpm.shape[master, local] * ti.Vector(
                            [self.mpm.incre_resolution[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                for local in range(self.mpm.offset[slave]):
                    node = self.mpm.LnID[slave, local]
                    block = self.mpm.node2dof[node] - 1
                    if block >= 0:
                        dofs = config.DIM * block
                        relative_adjoint -= self.mpm.shape[slave, local] * ti.Vector(
                            [self.mpm.incre_resolution[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                normal = self.pfriction[contact].normal
                tangent = ti.Matrix.identity(ti.f64, config.DIM) - normal.outer_product(normal)
                relative_displacement = (
                    self.mpm.p_temp[master_surface]
                    - self.hat_x[master_surface]
                    - self.mpm.p_temp[slave_surface]
                    + self.hat_x[slave_surface]
                )
                tangent_velocity = tangent.transpose() @ relative_displacement / self.mpm.dt
                point_force = mu_lambda * self.friction.grad_term(tangent_velocity.norm()) * tangent @ tangent_velocity
                ti.atomic_add(
                    self.friction_parameter_vjp[None][0],
                    -relative_adjoint.dot(point_force) / mu,
                )

    def _friction_parameter_gradients(self):
        values = np.zeros(4, dtype=np.float64)
        if self.activate_fric:
            self._differentiate_lagged_friction_parameters()
            values = np.asarray(self.friction_parameter_vjp[None], dtype=np.float64)
        gradients = {
            "friction_coefficient": float(values[0]),
        }
        return gradients

    @ti.kernel
    def _differentiate_elastic_young(self, active_dof: ti.i32, young: ti.f64):
        self.young_vjp[None] = 0.0
        for dof in range(active_dof):
            block = dof // config.DIM
            component = dof - config.DIM * block
            grid_id = self.mpm.dof2node[block]
            free = True
            if ti.static(self.mpm.dirichlet.num > 0):
                free = self.mpm.dirichlet.node[config.DIM * grid_id + component] == 0
            if free:
                ti.atomic_add(
                    self.young_vjp[None],
                    self.mpm.incre_resolution[dof] * self.mpm.rhs[dof] / young,
                )

    @ti.kernel
    def _differentiate_plastic_input_history(self):
        for particle_id in range(self.mpm.particleNum[0]):
            adjoint_deformation = ti.Matrix.zero(ti.f64, 3, 3)
            for local_id in range(self.mpm.offset[particle_id]):
                grid_id = self.mpm.LnID[particle_id, local_id]
                block = self.mpm.node2dof[grid_id] - 1
                if block >= 0:
                    for component in ti.static(range(config.DIM)):
                        free = True
                        if ti.static(self.mpm.dirichlet.num > 0):
                            free = self.mpm.dirichlet.node[config.DIM * grid_id + component] == 0
                        if free:
                            derivative = ti.Matrix.zero(ti.f64, 3, 3)
                            if ti.static(self.mpm.is_axisymmetric):
                                derivative = self.mpm.axisymmetric_dF_du(particle_id, local_id, component)
                            elif ti.static(self.mpm.is_plane_strain):
                                derivative = self.mpm.plane_strain_dF_du(particle_id, local_id, component)
                            else:
                                direction = (
                                    self.mpm.F0[particle_id].transpose() @ self.mpm.dshape[particle_id, local_id]
                                )
                                for column in ti.static(range(3)):
                                    derivative[component, column] = direction[column]
                            adjoint_deformation += (
                                self.mpm.incre_resolution[config.DIM * block + component] * derivative
                            )

            total_deformation = ti.Matrix.identity(ti.f64, 3)
            if ti.static(self.mpm.is_axisymmetric):
                total_deformation = (
                    self.mpm.get_axisymmetric_incremental_map(particle_id, self.mpm.grid_disp)
                    @ self.mpm.F0[particle_id]
                )
            elif ti.static(self.mpm.is_plane_strain):
                total_deformation = (
                    self.mpm.get_plane_strain_incremental_map(particle_id, self.mpm.grid_disp)
                    @ self.mpm.F0[particle_id]
                )
            else:
                total_deformation = (
                    ti.Matrix.identity(ti.f64, 3) + self.mpm.get_displacement_incre(particle_id, self.mpm.grid_disp)
                ) @ self.mpm.F0[particle_id]

            plastic_inverse = self.mpm.material.plastic_deformation_inverse[particle_id]
            elastic_trial = total_deformation @ plastic_inverse
            elastic_stress = self.mpm.material.first_piola_stress_at(particle_id, elastic_trial)
            elastic_tangent = self.mpm.material.first_piola_tangent_at(particle_id, elastic_trial)
            reference_jacobian = 1.0 / plastic_inverse.determinant()
            pulled_stress = elastic_stress @ plastic_inverse.transpose()
            particle_volume = self.mpm.particle[particle_id].vol0
            inverse = plastic_inverse.inverse()

            history_vjp = ti.Matrix.zero(ti.f64, 3, 3)
            for history_row, history_column in ti.static(ti.ndrange(3, 3)):
                elastic_stress_derivative = ti.Matrix.zero(ti.f64, 3, 3)
                for stress_column, stress_row in ti.static(ti.ndrange(3, 3)):
                    value = 0.0
                    for elastic_row in ti.static(range(3)):
                        value += (
                            elastic_tangent[
                                3 * stress_column + stress_row,
                                3 * history_column + elastic_row,
                            ]
                            * total_deformation[elastic_row, history_row]
                        )
                    elastic_stress_derivative[stress_row, stress_column] = value
                total_stress_derivative = reference_jacobian * (
                    elastic_stress_derivative @ plastic_inverse.transpose()
                    - inverse[history_column, history_row] * pulled_stress
                )
                for row in ti.static(range(3)):
                    total_stress_derivative[row, history_row] += (
                        reference_jacobian * elastic_stress[row, history_column]
                    )
                history_vjp[history_row, history_column] = -particle_volume * contraction(
                    adjoint_deformation, total_stress_derivative
                )
            self.plastic_inverse_vjp[particle_id] = history_vjp

            hardening_stress_derivative = self.mpm.material.first_piola_equivalent_plastic_strain_derivative_at(
                particle_id, elastic_trial
            )
            total_hardening_derivative = reference_jacobian * hardening_stress_derivative @ plastic_inverse.transpose()
            self.plastic_equivalent_strain_vjp[particle_id] = -particle_volume * contraction(
                adjoint_deformation, total_hardening_derivative
            )
            self.plastic_volumetric_strain_vjp[particle_id] = 0.0

            stress_vjp = (
                unflatten_matrix(
                    elastic_tangent.transpose()
                    @ flatten_matrix(reference_jacobian * adjoint_deformation @ plastic_inverse),
                    elastic_trial,
                )
                @ plastic_inverse.transpose()
            )
            total_stress = reference_jacobian * pulled_stress
            parameter_vjp = self.mpm.material.total_first_piola_parameter_vjp_at(
                particle_id, total_deformation, adjoint_deformation
            )
            for parameter_id in ti.static(range(4)):
                ti.atomic_add(
                    self.material_parameter_vjp[None][parameter_id],
                    -particle_volume * parameter_vjp[parameter_id],
                )
            previous_total = self.mpm.F0[particle_id]
            incremental_map = total_deformation @ previous_total.inverse()
            adjoint_incremental_map = adjoint_deformation @ previous_total.inverse()
            self.plastic_deformation_vjp[particle_id] = -particle_volume * (
                incremental_map.transpose() @ stress_vjp + adjoint_incremental_map.transpose() @ total_stress
            )

    @ti.kernel
    def _differentiate_plastic_commit_state(self):
        for dof in self.plastic_commit_grid_vjp:
            self.plastic_commit_grid_vjp[dof] = 0.0
        for particle_id in range(self.mpm.particleNum[0]):
            total_deformation = ti.Matrix.identity(ti.f64, 3)
            if ti.static(self.mpm.is_axisymmetric):
                total_deformation = (
                    self.mpm.get_axisymmetric_incremental_map(particle_id, self.mpm.grid_disp)
                    @ self.mpm.F0[particle_id]
                )
            elif ti.static(self.mpm.is_plane_strain):
                total_deformation = (
                    self.mpm.get_plane_strain_incremental_map(particle_id, self.mpm.grid_disp)
                    @ self.mpm.F0[particle_id]
                )
            else:
                total_deformation = (
                    ti.Matrix.identity(ti.f64, 3) + self.mpm.get_displacement_incre(particle_id, self.mpm.grid_disp)
                ) @ self.mpm.F0[particle_id]
            (
                trial_vjp,
                plastic_vjp,
                equivalent_vjp,
                volumetric_vjp,
            ) = self.mpm.material.commit_total_state_vjp(
                particle_id,
                total_deformation,
                self.plastic_output_deformation_vjp[particle_id],
                self.plastic_output_inverse_vjp[particle_id],
                self.plastic_output_equivalent_vjp[particle_id],
                self.plastic_output_volumetric_vjp[particle_id],
            )
            parameter_vjp = self.mpm.material.commit_total_state_parameter_vjp(
                particle_id,
                total_deformation,
                self.plastic_output_inverse_vjp[particle_id],
                self.plastic_output_equivalent_vjp[particle_id],
                self.plastic_output_volumetric_vjp[particle_id],
            )
            for parameter_id in ti.static(range(4)):
                ti.atomic_add(
                    self.material_parameter_vjp[None][parameter_id],
                    parameter_vjp[parameter_id],
                )
            self.plastic_trial_deformation_vjp[particle_id] = trial_vjp
            self.plastic_commit_inverse_vjp[particle_id] = plastic_vjp
            self.plastic_commit_equivalent_vjp[particle_id] = equivalent_vjp
            self.plastic_commit_volumetric_vjp[particle_id] = volumetric_vjp
            incremental_map = total_deformation @ self.mpm.F0[particle_id].inverse()
            self.plastic_commit_deformation_vjp[particle_id] = incremental_map.transpose() @ trial_vjp
            for local_id in range(self.mpm.offset[particle_id]):
                grid_id = self.mpm.LnID[particle_id, local_id]
                block = self.mpm.node2dof[grid_id] - 1
                if block >= 0:
                    for component in ti.static(range(config.DIM)):
                        free = True
                        if ti.static(self.mpm.dirichlet.num > 0):
                            free = self.mpm.dirichlet.node[config.DIM * grid_id + component] == 0
                        if free:
                            derivative = ti.Matrix.zero(ti.f64, 3, 3)
                            if ti.static(self.mpm.is_axisymmetric):
                                derivative = self.mpm.axisymmetric_dF_du(particle_id, local_id, component)
                            elif ti.static(self.mpm.is_plane_strain):
                                derivative = self.mpm.plane_strain_dF_du(particle_id, local_id, component)
                            else:
                                direction = (
                                    self.mpm.F0[particle_id].transpose() @ self.mpm.dshape[particle_id, local_id]
                                )
                                for column in ti.static(range(3)):
                                    derivative[component, column] = direction[column]
                            ti.atomic_add(
                                self.plastic_commit_grid_vjp[config.DIM * block + component],
                                contraction(trial_vjp, derivative),
                            )

    @ti.kernel
    def _combine_plastic_step_vjp(self):
        for particle_id in range(self.mpm.particleNum[0]):
            self.plastic_deformation_vjp[particle_id] += self.plastic_commit_deformation_vjp[particle_id]
            self.plastic_inverse_vjp[particle_id] += self.plastic_commit_inverse_vjp[particle_id]
            self.plastic_equivalent_strain_vjp[particle_id] += self.plastic_commit_equivalent_vjp[particle_id]
            self.plastic_volumetric_strain_vjp[particle_id] += self.plastic_commit_volumetric_vjp[particle_id]

    def _load_particle_output_state_vjp(self, state_vjp):
        particle_num = int(self.mpm.particleNum[0])
        capacity = int(self.particle_output_position_vjp.shape[0])

        def load(name, field):
            values = (
                np.asarray(state_vjp[name], dtype=np.float64)
                if name in state_vjp
                else np.zeros((particle_num, config.DIM), dtype=np.float64)
            )
            if values.shape != (particle_num, config.DIM) or not np.all(np.isfinite(values)):
                raise ValueError(f"state_vjp['{name}'] must be finite with shape " f"({particle_num}, {config.DIM})")
            padded = np.zeros((capacity, config.DIM), dtype=np.float64)
            padded[:particle_num] = values
            field.from_numpy(padded)

        load("position", self.particle_output_position_vjp)
        load("velocity", self.particle_output_velocity_vjp)
        load("acceleration", self.particle_output_acceleration_vjp)

    @ti.kernel
    def _differentiate_particle_advection(self):
        for dof in self.trajectory_grid_vjp:
            self.trajectory_grid_vjp[dof] = self.plastic_commit_grid_vjp[dof]
        for node in self.trajectory_node_mass_vjp:
            self.trajectory_node_mass_vjp[node] = 0.0
            self.trajectory_node_velocity_vjp[node] = ti.Vector.zero(ti.f64, config.DIM)
            self.trajectory_node_acceleration_vjp[node] = ti.Vector.zero(ti.f64, config.DIM)

        dt = self.mpm.TIdt[None]
        alpha = ti.static(float(self.mpm.integration[0]))
        beta = ti.static(float(self.mpm.integration[1]))
        gamma = ti.static(float(self.mpm.integration[2]))
        acceleration_displacement_scale = 1.0 / (2.0 * dt * dt * alpha * beta)
        acceleration_velocity_scale = -1.0 / (2.0 * dt * alpha * beta)
        acceleration_acceleration_scale = 1.0 - 0.5 / beta
        velocity_displacement_scale = 0.5 * gamma / (alpha * beta * dt)
        velocity_velocity_scale = 1.0 - 0.5 * gamma / (alpha * beta)
        velocity_acceleration_scale = dt * (1.0 - 0.5 * gamma / beta)
        pic = ti.static(float(self.mpm.coeffPIC))

        for particle_id in range(self.mpm.particleNum[0]):
            output_position_vjp = self.particle_output_position_vjp[particle_id]
            output_velocity_vjp = self.particle_output_velocity_vjp[particle_id]
            acceleration_vjp = (
                self.particle_output_acceleration_vjp[particle_id] + (1.0 - pic) * dt * output_velocity_vjp
            )
            interpolated_velocity_vjp = pic * output_velocity_vjp
            position_vjp = output_position_vjp
            velocity_vjp = (1.0 - pic) * output_velocity_vjp
            trial_increment_vjp = self.plastic_trial_deformation_vjp[particle_id] @ self.mpm.F0[particle_id].transpose()

            for local_id in range(self.mpm.offset[particle_id]):
                node = self.mpm.LnID[particle_id, local_id]
                block = self.mpm.node2dof[node] - 1
                if block >= 0:
                    dofs = config.DIM * block
                    weight = self.mpm.shape[particle_id, local_id]
                    gradient = self.mpm.dshape[particle_id, local_id]
                    displacement = ti.Vector(
                        [self.mpm.grid_disp[dofs + component] for component in ti.static(range(config.DIM))]
                    )
                    old_velocity = self.mpm.grid[node].v
                    old_acceleration = self.mpm.grid[node].a
                    new_velocity = (
                        velocity_displacement_scale * displacement
                        + velocity_velocity_scale * old_velocity
                        + velocity_acceleration_scale * old_acceleration
                    )
                    new_acceleration = (
                        acceleration_displacement_scale * displacement
                        + acceleration_velocity_scale * old_velocity
                        + acceleration_acceleration_scale * old_acceleration
                    )
                    free = ti.Vector.one(ti.f64, config.DIM)
                    if ti.static(self.mpm.dirichlet.num > 0):
                        for component in ti.static(range(config.DIM)):
                            if self.mpm.dirichlet.node[config.DIM * node + component] != 0:
                                free[component] = 0.0
                    grid_vjp = weight * (
                        output_position_vjp
                        + acceleration_displacement_scale * acceleration_vjp
                        + velocity_displacement_scale * interpolated_velocity_vjp
                    )
                    for component in ti.static(range(config.DIM)):
                        ti.atomic_add(
                            self.trajectory_grid_vjp[dofs + component],
                            free[component] * grid_vjp[component],
                        )
                    self.trajectory_node_velocity_vjp[node] += weight * (
                        acceleration_velocity_scale * acceleration_vjp
                        + velocity_velocity_scale * interpolated_velocity_vjp
                    )
                    self.trajectory_node_acceleration_vjp[node] += weight * (
                        acceleration_acceleration_scale * acceleration_vjp
                        + velocity_acceleration_scale * interpolated_velocity_vjp
                    )
                    position_vjp += gradient * (
                        output_position_vjp.dot(displacement)
                        + acceleration_vjp.dot(new_acceleration)
                        + interpolated_velocity_vjp.dot(new_velocity)
                    )

                    hessian_argument = ti.Vector.zero(ti.f64, config.DIM)
                    for material_axis, spatial in ti.static(ti.ndrange(config.DIM, config.DIM)):
                        hessian_argument[material_axis] += (
                            trial_increment_vjp[spatial, material_axis] * displacement[spatial]
                        )
                    position_vjp += self.mpm.shape_hessian(particle_id, local_id).transpose() @ hessian_argument

            self.particle_position_vjp[particle_id] = position_vjp
            self.particle_velocity_vjp[particle_id] = velocity_vjp
            self.particle_acceleration_vjp[particle_id] = ti.Vector.zero(ti.f64, config.DIM)

    @ti.kernel
    def _differentiate_inertia_input_state(self):
        dt = self.mpm.TIdt[None]
        alpha = ti.static(float(self.mpm.integration[0]))
        beta = ti.static(float(self.mpm.integration[1]))
        gamma = ti.static(float(self.mpm.integration[2]))
        acceleration_velocity_scale = -1.0 / (2.0 * dt * alpha * beta)
        acceleration_acceleration_scale = 1.0 - 0.5 / beta
        velocity_velocity_scale = 1.0 - 0.5 * gamma / (alpha * beta)
        velocity_acceleration_scale = dt * (1.0 - 0.5 * gamma / beta)
        gravity = ti.Vector(
            [ti.static(float(self.mpm.gravity[component])) for component in ti.static(range(config.DIM))]
        )
        for node in self.mpm.grid:
            mass = self.mpm.grid[node].m
            block = self.mpm.node2dof[node] - 1
            if mass > self.mpm.val_lim and block >= 0:
                dofs = config.DIM * block
                displacement = ti.Vector(
                    [self.mpm.grid_disp[dofs + component] for component in ti.static(range(config.DIM))]
                )
                old_velocity = self.mpm.grid[node].v
                old_acceleration = self.mpm.grid[node].a
                new_velocity = (
                    self.endpoint_velocity_displacement_scale * displacement
                    + velocity_velocity_scale * old_velocity
                    + velocity_acceleration_scale * old_acceleration
                )
                acceleration_displacement_scale = 1.0 / (2.0 * dt * dt * alpha * beta)
                new_acceleration = (
                    acceleration_displacement_scale * displacement
                    + acceleration_velocity_scale * old_velocity
                    + acceleration_acceleration_scale * old_acceleration
                )
                adjoint = ti.Vector(
                    [self.mpm.incre_resolution[dofs + component] for component in ti.static(range(config.DIM))]
                )
                self.trajectory_node_mass_vjp[node] += adjoint.dot(
                    gravity - new_acceleration - self.mpm.damping * new_velocity
                )
                self.trajectory_node_velocity_vjp[node] += (
                    -mass * (acceleration_velocity_scale + self.mpm.damping * velocity_velocity_scale) * adjoint
                )
                self.trajectory_node_acceleration_vjp[node] += (
                    -mass * (acceleration_acceleration_scale + self.mpm.damping * velocity_acceleration_scale) * adjoint
                )

    @ti.kernel
    def _differentiate_material_position(self):
        for particle_id in range(self.mpm.particleNum[0]):
            adjoint_increment = ti.Matrix.zero(ti.f64, 3, 3)
            for local_id in range(self.mpm.offset[particle_id]):
                node = self.mpm.LnID[particle_id, local_id]
                block = self.mpm.node2dof[node] - 1
                if block >= 0:
                    dofs = config.DIM * block
                    for spatial in ti.static(range(config.DIM)):
                        adjoint = self.mpm.incre_resolution[dofs + spatial]
                        for material_axis in ti.static(range(config.DIM)):
                            adjoint_increment[spatial, material_axis] += (
                                adjoint * self.mpm.dshape[particle_id, local_id][material_axis]
                            )
            adjoint_deformation = adjoint_increment @ self.mpm.F0[particle_id]
            total_deformation = ti.Matrix.identity(ti.f64, 3)
            if ti.static(config.DIM == 2):
                total_deformation = (
                    self.mpm.get_plane_strain_incremental_map(particle_id, self.mpm.grid_disp)
                    @ self.mpm.F0[particle_id]
                )
            else:
                total_deformation = (
                    ti.Matrix.identity(ti.f64, 3) + self.mpm.get_displacement_incre(particle_id, self.mpm.grid_disp)
                ) @ self.mpm.F0[particle_id]
            stress = unflatten_matrix(
                self.mpm.material.total_dPsi_div_dF_at(particle_id, total_deformation),
                total_deformation,
            )
            tangent = self.mpm.material.total_d2Psi_div_d2F_at(particle_id, total_deformation)
            deformation_vjp = unflatten_matrix(
                -self.mpm.particle[particle_id].vol0 * tangent.transpose() @ flatten_matrix(adjoint_deformation),
                total_deformation,
            )
            increment_vjp = deformation_vjp @ self.mpm.F0[particle_id].transpose()
            adjoint_increment_vjp = -self.mpm.particle[particle_id].vol0 * stress @ self.mpm.F0[particle_id].transpose()
            position_vjp = ti.Vector.zero(ti.f64, config.DIM)
            for local_id in range(self.mpm.offset[particle_id]):
                node = self.mpm.LnID[particle_id, local_id]
                block = self.mpm.node2dof[node] - 1
                if block >= 0:
                    dofs = config.DIM * block
                    displacement = ti.Vector(
                        [self.mpm.grid_disp[dofs + component] for component in ti.static(range(config.DIM))]
                    )
                    adjoint = ti.Vector(
                        [self.mpm.incre_resolution[dofs + component] for component in ti.static(range(config.DIM))]
                    )
                    displacement_argument = ti.Vector.zero(ti.f64, config.DIM)
                    adjoint_argument = ti.Vector.zero(ti.f64, config.DIM)
                    for material_axis, spatial in ti.static(ti.ndrange(config.DIM, config.DIM)):
                        displacement_argument[material_axis] += (
                            increment_vjp[spatial, material_axis] * displacement[spatial]
                        )
                        adjoint_argument[material_axis] += (
                            adjoint_increment_vjp[spatial, material_axis] * adjoint[spatial]
                        )
                    hessian = self.mpm.shape_hessian(particle_id, local_id)
                    position_vjp += hessian.transpose() @ (displacement_argument + adjoint_argument)
            self.particle_position_vjp[particle_id] += position_vjp

    @ti.kernel
    def _differentiate_barrier_position(self):
        for contact in range(self.gbarrierNum[0]):
            surface = self.gbarrier[contact].surfaceID
            particle = self.mpm.surface_id[surface]
            wall = self.gbarrier[contact].wallID
            value = self.gbarrier[contact].distance
            measure = point_contact_measure(self.mpm.surface_measure[surface])
            _, first, second = self._normal_terms(value, -1)
            distance_gradient = self.gderivative.Ddistance_div_Dpoint(wall)
            energy_gradient = measure * first * distance_gradient
            point_adjoint = ti.Vector.zero(ti.f64, config.DIM)
            # Flatten these tiny matrices explicitly.  This avoids a CUDA
            # misaligned local-matrix transaction in Taichi 1.7 while keeping
            # all contact work on device.
            adjoint_gradient = ti.Vector.zero(ti.f64, 9)
            point_jacobian = ti.Vector.zero(ti.f64, 9)
            for row, column in ti.static(ti.ndrange(config.DIM, config.DIM)):
                point_jacobian[row * config.DIM + column] = 1.0 if row == column else 0.0
            for local_id in ti.static(range(self.mpm.shape_func.max_node_per_particle)):
                if local_id < self.mpm.offset[particle]:
                    node = self.mpm.LnID[particle, local_id]
                    block = self.mpm.node2dof[node] - 1
                    if block >= 0:
                        dofs = config.DIM * block
                        adjoint = ti.Vector(
                            [self.mpm.incre_resolution[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                        displacement = ti.Vector(
                            [self.mpm.grid_disp[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                        weight = self.mpm.shape[particle, local_id]
                        gradient = self.mpm.dshape[particle, local_id]
                        point_adjoint += weight * adjoint
                        for row, column in ti.static(ti.ndrange(config.DIM, config.DIM)):
                            adjoint_gradient[row * config.DIM + column] += adjoint[row] * gradient[column]
                            point_jacobian[row * config.DIM + column] += displacement[row] * gradient[column]
            position_vjp = ti.Vector.zero(ti.f64, config.DIM)
            for row in ti.static(range(config.DIM)):
                value_vjp = 0.0
                for column in ti.static(range(config.DIM)):
                    value_vjp -= adjoint_gradient[column * config.DIM + row] * energy_gradient[column]
                    hessian_adjoint = 0.0
                    for axis in ti.static(range(config.DIM)):
                        hessian_adjoint += (
                            measure * second * distance_gradient[column] * distance_gradient[axis] * point_adjoint[axis]
                        )
                    value_vjp -= point_jacobian[column * config.DIM + row] * hessian_adjoint
                position_vjp[row] = value_vjp
            for component in ti.static(range(config.DIM)):
                ti.atomic_add(
                    self.barrier_position_vjp_flat[particle * config.DIM + component],
                    position_vjp[component],
                )

        for contact in range(self.pbarrierNum[0]):
            master_surface = self.pbarrier[contact].masterID
            slave_surface = self.pbarrier[contact].slaveID
            master = self.mpm.surface_id[master_surface]
            slave = self.mpm.surface_id[slave_surface]
            master_position = self.mpm.p_temp[master_surface]
            slave_position = self.mpm.p_temp[slave_surface]
            measure = symmetric_contact_measure(
                self.mpm.surface_measure[master_surface],
                self.mpm.surface_measure[slave_surface],
            )
            distance_gradient, _ = self.pderivative.Ddistance_div_Dpoint(master_position, slave_position)
            _, first, second = self._normal_terms(self.pbarrier[contact].distance, -1)
            energy_gradient = measure * first * distance_gradient
            relative_position = master_position - slave_position
            inverse_distance = 1.0 / ti.max(relative_position.norm(), 1.0e-12)
            distance_hessian = ti.Vector.zero(ti.f64, 9)
            for row, column in ti.static(ti.ndrange(config.DIM, config.DIM)):
                distance_hessian[row * config.DIM + column] = inverse_distance * (
                    (1.0 if row == column else 0.0)
                    - inverse_distance * inverse_distance * relative_position[row] * relative_position[column]
                )
            energy_hessian = ti.Vector.zero(ti.f64, 9)
            for row, column in ti.static(ti.ndrange(config.DIM, config.DIM)):
                energy_hessian[row * config.DIM + column] = (
                    measure * second * distance_gradient[row] * distance_gradient[column]
                    + measure * first * distance_hessian[row * config.DIM + column]
                )
            master_adjoint = ti.Vector.zero(ti.f64, config.DIM)
            slave_adjoint = ti.Vector.zero(ti.f64, config.DIM)
            master_adjoint_gradient = ti.Vector.zero(ti.f64, 9)
            slave_adjoint_gradient = ti.Vector.zero(ti.f64, 9)
            master_jacobian = ti.Vector.zero(ti.f64, 9)
            slave_jacobian = ti.Vector.zero(ti.f64, 9)
            for row, column in ti.static(ti.ndrange(config.DIM, config.DIM)):
                master_jacobian[row * config.DIM + column] = 1.0 if row == column else 0.0
                slave_jacobian[row * config.DIM + column] = 1.0 if row == column else 0.0
            for local_id in ti.static(range(self.mpm.shape_func.max_node_per_particle)):
                if local_id < self.mpm.offset[master]:
                    node = self.mpm.LnID[master, local_id]
                    block = self.mpm.node2dof[node] - 1
                    if block >= 0:
                        dofs = config.DIM * block
                        adjoint = ti.Vector(
                            [self.mpm.incre_resolution[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                        displacement = ti.Vector(
                            [self.mpm.grid_disp[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                        weight = self.mpm.shape[master, local_id]
                        gradient = self.mpm.dshape[master, local_id]
                        master_adjoint += weight * adjoint
                        for row, column in ti.static(ti.ndrange(config.DIM, config.DIM)):
                            master_adjoint_gradient[row * config.DIM + column] += adjoint[row] * gradient[column]
                            master_jacobian[row * config.DIM + column] += displacement[row] * gradient[column]
            for local_id in ti.static(range(self.mpm.shape_func.max_node_per_particle)):
                if local_id < self.mpm.offset[slave]:
                    node = self.mpm.LnID[slave, local_id]
                    block = self.mpm.node2dof[node] - 1
                    if block >= 0:
                        dofs = config.DIM * block
                        adjoint = ti.Vector(
                            [self.mpm.incre_resolution[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                        displacement = ti.Vector(
                            [self.mpm.grid_disp[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                        weight = self.mpm.shape[slave, local_id]
                        gradient = self.mpm.dshape[slave, local_id]
                        slave_adjoint += weight * adjoint
                        for row, column in ti.static(ti.ndrange(config.DIM, config.DIM)):
                            slave_adjoint_gradient[row * config.DIM + column] += adjoint[row] * gradient[column]
                            slave_jacobian[row * config.DIM + column] += displacement[row] * gradient[column]
            relative_adjoint = master_adjoint - slave_adjoint
            master_vjp = ti.Vector.zero(ti.f64, config.DIM)
            slave_vjp = ti.Vector.zero(ti.f64, config.DIM)
            for row in ti.static(range(config.DIM)):
                master_value = 0.0
                slave_value = 0.0
                for column in ti.static(range(config.DIM)):
                    master_value -= master_adjoint_gradient[column * config.DIM + row] * energy_gradient[column]
                    slave_value += slave_adjoint_gradient[column * config.DIM + row] * energy_gradient[column]
                    hessian_adjoint = 0.0
                    for axis in ti.static(range(config.DIM)):
                        hessian_adjoint += energy_hessian[axis * config.DIM + column] * relative_adjoint[axis]
                    master_value -= master_jacobian[column * config.DIM + row] * hessian_adjoint
                    slave_value += slave_jacobian[column * config.DIM + row] * hessian_adjoint
                master_vjp[row] = master_value
                slave_vjp[row] = slave_value
            for component in ti.static(range(config.DIM)):
                ti.atomic_add(
                    self.barrier_position_vjp_flat[master * config.DIM + component],
                    master_vjp[component],
                )
                ti.atomic_add(
                    self.barrier_position_vjp_flat[slave * config.DIM + component],
                    slave_vjp[component],
                )

    @ti.kernel
    def _accumulate_barrier_position_vjp(self):
        for particle in range(self.particle_position_vjp.shape[0]):
            for component in ti.static(range(config.DIM)):
                self.particle_position_vjp[particle][component] += self.barrier_position_vjp_flat[
                    particle * config.DIM + component
                ]

    @ti.kernel
    def _differentiate_lagged_friction_position(self):
        # ponytail: the accepted lagged normal, tangent frame, active set, and
        # normal-force magnitude are stop-gradient state; differentiate their
        # update only if the outer fixed-point map itself becomes an objective.
        for contact in range(self.gfrictionNum[0]):
            surface = self.gfriction[contact].surfaceID
            particle = self.mpm.surface_id[surface]
            wall = self.gfriction[contact].wallID
            mu_lambda = self.gfriction[contact].mu_lambda
            if mu_lambda > 0.0:
                tangent = self.ground.tangent_operator(wall)
                relative_displacement = (
                    self.mpm.p_temp[surface] - self.hat_x[surface] - self.ground.vel[wall] * self.mpm.dt
                )
                tangent_velocity = tangent.transpose() @ relative_displacement / self.mpm.dt
                speed = tangent_velocity.norm()
                friction_gradient = self.friction.grad_term(speed)
                inner = friction_gradient * ti.Matrix.identity(ti.f64, config.DIM)
                if speed != 0.0:
                    inner += self.friction.hess_term(speed) / speed * tangent_velocity.outer_product(tangent_velocity)
                point_force = mu_lambda * friction_gradient * tangent @ tangent_velocity
                point_jacobian = mu_lambda * tangent @ psd_project_nd(inner) @ tangent.transpose() / self.mpm.dt
                point_adjoint = ti.Vector.zero(ti.f64, config.DIM)
                adjoint_gradient = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                displacement_gradient = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                for local_id in range(self.mpm.offset[particle]):
                    node = self.mpm.LnID[particle, local_id]
                    block = self.mpm.node2dof[node] - 1
                    if block >= 0:
                        dofs = config.DIM * block
                        adjoint = ti.Vector(
                            [self.mpm.incre_resolution[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                        displacement = ti.Vector(
                            [self.mpm.grid_disp[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                        weight = self.mpm.shape[particle, local_id]
                        gradient = self.mpm.dshape[particle, local_id]
                        point_adjoint += weight * adjoint
                        adjoint_gradient += adjoint.outer_product(gradient)
                        displacement_gradient += displacement.outer_product(gradient)
                position_vjp = (
                    -adjoint_gradient.transpose() @ point_force
                    - displacement_gradient.transpose() @ point_jacobian.transpose() @ point_adjoint
                )
                for component in ti.static(range(config.DIM)):
                    ti.atomic_add(
                        self.particle_position_vjp[particle][component],
                        position_vjp[component],
                    )

        for contact in range(self.pfrictionNum[0]):
            master_surface = self.pfriction[contact].masterID
            slave_surface = self.pfriction[contact].slaveID
            master = self.mpm.surface_id[master_surface]
            slave = self.mpm.surface_id[slave_surface]
            mu_lambda = self.pfriction[contact].mu_lambda
            if mu_lambda > 0.0:
                normal = self.pfriction[contact].normal
                tangent = ti.Matrix.identity(ti.f64, config.DIM) - normal.outer_product(normal)
                relative_displacement = (
                    self.mpm.p_temp[master_surface]
                    - self.hat_x[master_surface]
                    - self.mpm.p_temp[slave_surface]
                    + self.hat_x[slave_surface]
                )
                tangent_velocity = tangent.transpose() @ relative_displacement / self.mpm.dt
                speed = tangent_velocity.norm()
                friction_gradient = self.friction.grad_term(speed)
                inner = friction_gradient * ti.Matrix.identity(ti.f64, config.DIM)
                if speed != 0.0:
                    inner += self.friction.hess_term(speed) / speed * tangent_velocity.outer_product(tangent_velocity)
                point_force = mu_lambda * friction_gradient * tangent @ tangent_velocity
                point_jacobian = mu_lambda * tangent @ psd_project_nd(inner) @ tangent.transpose() / self.mpm.dt
                relative_adjoint = ti.Vector.zero(ti.f64, config.DIM)
                master_adjoint_gradient = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                slave_adjoint_gradient = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                master_displacement_gradient = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                slave_displacement_gradient = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                for local_id in range(self.mpm.offset[master]):
                    node = self.mpm.LnID[master, local_id]
                    block = self.mpm.node2dof[node] - 1
                    if block >= 0:
                        dofs = config.DIM * block
                        adjoint = ti.Vector(
                            [self.mpm.incre_resolution[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                        displacement = ti.Vector(
                            [self.mpm.grid_disp[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                        weight = self.mpm.shape[master, local_id]
                        gradient = self.mpm.dshape[master, local_id]
                        relative_adjoint += weight * adjoint
                        master_adjoint_gradient += adjoint.outer_product(gradient)
                        master_displacement_gradient += displacement.outer_product(gradient)
                for local_id in range(self.mpm.offset[slave]):
                    node = self.mpm.LnID[slave, local_id]
                    block = self.mpm.node2dof[node] - 1
                    if block >= 0:
                        dofs = config.DIM * block
                        adjoint = ti.Vector(
                            [self.mpm.incre_resolution[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                        displacement = ti.Vector(
                            [self.mpm.grid_disp[dofs + component] for component in ti.static(range(config.DIM))]
                        )
                        weight = self.mpm.shape[slave, local_id]
                        gradient = self.mpm.dshape[slave, local_id]
                        relative_adjoint -= weight * adjoint
                        slave_adjoint_gradient += adjoint.outer_product(gradient)
                        slave_displacement_gradient += displacement.outer_product(gradient)
                point_vjp = point_jacobian.transpose() @ relative_adjoint
                master_vjp = (
                    -master_adjoint_gradient.transpose() @ point_force
                    - master_displacement_gradient.transpose() @ point_vjp
                )
                slave_vjp = (
                    slave_adjoint_gradient.transpose() @ point_force
                    + slave_displacement_gradient.transpose() @ point_vjp
                )
                for component in ti.static(range(config.DIM)):
                    ti.atomic_add(
                        self.particle_position_vjp[master][component],
                        master_vjp[component],
                    )
                    ti.atomic_add(
                        self.particle_position_vjp[slave][component],
                        slave_vjp[component],
                    )

    @ti.kernel
    def _differentiate_traction_position(self):
        for traction_id in range(self.mpm.tractionNum[0]):
            particle = self.mpm.traction[traction_id].particleID
            traction = self.mpm.traction[traction_id].traction
            position_vjp = ti.Vector.zero(ti.f64, config.DIM)
            for local_id in range(self.mpm.offset[particle]):
                node = self.mpm.LnID[particle, local_id]
                block = self.mpm.node2dof[node] - 1
                if block >= 0:
                    dofs = config.DIM * block
                    adjoint = ti.Vector(
                        [self.mpm.incre_resolution[dofs + component] for component in ti.static(range(config.DIM))]
                    )
                    position_vjp += adjoint.dot(traction) * self.mpm.dshape[particle, local_id]
            for component in ti.static(range(config.DIM)):
                ti.atomic_add(
                    self.particle_position_vjp[particle][component],
                    position_vjp[component],
                )

    @ti.kernel
    def _differentiate_p2g_input_state(self):
        for particle_id in range(self.mpm.particleNum[0]):
            mass = self.mpm.particle[particle_id].m
            particle_velocity = self.mpm.particle[particle_id].v
            particle_acceleration = self.mpm.particle[particle_id].a
            position_vjp = ti.Vector.zero(ti.f64, config.DIM)
            velocity_vjp = ti.Vector.zero(ti.f64, config.DIM)
            acceleration_vjp = ti.Vector.zero(ti.f64, config.DIM)
            for local_id in range(self.mpm.offset[particle_id]):
                node = self.mpm.LnID[particle_id, local_id]
                node_mass = self.mpm.grid[node].m
                if node_mass > self.mpm.val_lim:
                    weight = self.mpm.shape[particle_id, local_id]
                    node_velocity_vjp = self.trajectory_node_velocity_vjp[node]
                    node_acceleration_vjp = self.trajectory_node_acceleration_vjp[node]
                    velocity_vjp += mass * weight / node_mass * node_velocity_vjp
                    acceleration_vjp += mass * weight / node_mass * node_acceleration_vjp
                    weight_vjp = mass * (
                        self.trajectory_node_mass_vjp[node]
                        + node_velocity_vjp.dot(particle_velocity - self.mpm.grid[node].v) / node_mass
                        + node_acceleration_vjp.dot(particle_acceleration - self.mpm.grid[node].a) / node_mass
                    )
                    position_vjp += weight_vjp * self.mpm.dshape[particle_id, local_id]
            self.particle_position_vjp[particle_id] += position_vjp
            self.particle_velocity_vjp[particle_id] += velocity_vjp
            self.particle_acceleration_vjp[particle_id] += acceleration_vjp

    def differentiate_elastic_parameters(self, loss_gradient):
        """Return one converged pre-commit BarrierIPC equilibrium VJP."""
        young = float(getattr(self.mpm.material, "young", 0.0))
        if type(self.mpm.material).__name__ != "NeoHookeanModel" or not np.isfinite(young) or young <= 0.0:
            # ponytail: add per-model parameter kernels only when another
            # direct elastic material is exposed by ImplicitMPM.
            raise ValueError("Direct IPC-MPM Young-modulus VJP currently requires the " "NeoHookeanModel")
        adjoint = self.solve_elastic_adjoint(loss_gradient)
        self._differentiate_elastic_gravity()
        self.mpm.rhs.fill(0.0)
        self.mpm.assemble_material_force(self.mpm.active_dof, self.mpm.grid_disp)
        self._differentiate_elastic_young(self.mpm.active_dof, young)
        return {
            "gravity": np.asarray(self.gravity_vjp[None], dtype=np.float64),
            "young_modulus": float(self.young_vjp[None]),
            "adjoint": adjoint,
            **self._friction_parameter_gradients(),
        }

    def differentiate_plastic_equilibrium_parameters(self, loss_gradient):
        """Return current-equilibrium parameter and input-history VJPs."""
        self.material_parameter_vjp.fill(0.0)
        adjoint = self.solve_plastic_equilibrium_adjoint(loss_gradient)
        self.pullback_plastic_equilibrium_from_current_adjoint_device()
        particle_num = int(self.mpm.particleNum[0])
        material_values = np.asarray(self.material_parameter_vjp[None], dtype=np.float64)
        if type(self.mpm.material).__name__ == "FiniteStrainDruckerPragerModel":
            material_names = {
                "cohesion": float(material_values[2]),
                "friction_angle_degrees": float(material_values[3]),
            }
        else:
            material_names = {
                "yield_stress": float(material_values[2]),
                "hardening_modulus": float(material_values[3]),
            }
        friction_values = np.asarray(self.friction_parameter_vjp[None], dtype=np.float64)
        return {
            "gravity": np.asarray(self.gravity_vjp[None], dtype=np.float64),
            "material_parameters": material_values,
            "young_modulus": float(material_values[0]),
            "poisson_ratio": float(material_values[1]),
            **material_names,
            "adjoint": adjoint,
            "plastic_deformation_inverse": self.plastic_inverse_vjp.to_numpy()[:particle_num].copy(),
            "equivalent_plastic_strain": self.plastic_equivalent_strain_vjp.to_numpy()[:particle_num].copy(),
            "volumetric_plastic_strain": self.plastic_volumetric_strain_vjp.to_numpy()[:particle_num].copy(),
            "deformation_gradient": self.plastic_deformation_vjp.to_numpy()[:particle_num].copy(),
            "plastic_history": "input_vjp",
            "friction_coefficient": float(friction_values[0]),
        }

    def pullback_plastic_equilibrium_from_current_adjoint_device(self, reset_material_parameters=True):
        """Pull back with an adjoint already stored in ``incre_resolution``."""
        self._validate_differentiable_friction()
        if type(self.mpm.material).__name__ not in {
            "FiniteStrainDruckerPragerModel",
            "FiniteStrainVonMisesModel",
        }:
            raise ValueError(
                "plastic equilibrium pullback supports Drucker-Prager and " "von Mises; MCC is intentionally excluded"
            )
        if reset_material_parameters:
            self.material_parameter_vjp.fill(0.0)
        self._differentiate_elastic_gravity()
        self._differentiate_plastic_input_history()
        self.friction_parameter_vjp.fill(0.0)
        if self.activate_fric:
            self._differentiate_lagged_friction_parameters()

    def _load_plastic_output_state_vjp(self, state_vjp):
        if not isinstance(state_vjp, dict):
            raise TypeError("state_vjp must be a dictionary of per-particle seeds")
        particle_num = int(self.mpm.particleNum[0])
        capacity = int(self.plastic_output_equivalent_vjp.shape[0])

        def load_matrix(name, field):
            values = (
                np.asarray(state_vjp[name], dtype=np.float64)
                if name in state_vjp
                else np.zeros((particle_num, 3, 3), dtype=np.float64)
            )
            if values.shape != (particle_num, 3, 3) or not np.all(np.isfinite(values)):
                raise ValueError(f"state_vjp['{name}'] must be finite with shape " f"({particle_num}, 3, 3)")
            padded = np.zeros((capacity, 3, 3), dtype=np.float64)
            padded[:particle_num] = values
            field.from_numpy(padded)

        def load_scalar(name, field):
            values = (
                np.asarray(state_vjp[name], dtype=np.float64)
                if name in state_vjp
                else np.zeros(particle_num, dtype=np.float64)
            )
            if values.shape != (particle_num,) or not np.all(np.isfinite(values)):
                raise ValueError(f"state_vjp['{name}'] must be finite with shape " f"({particle_num},)")
            padded = np.zeros(capacity, dtype=np.float64)
            padded[:particle_num] = values
            field.from_numpy(padded)

        load_matrix("deformation_gradient", self.plastic_output_deformation_vjp)
        load_matrix("plastic_deformation_inverse", self.plastic_output_inverse_vjp)
        load_scalar("equivalent_plastic_strain", self.plastic_output_equivalent_vjp)
        load_scalar("volumetric_plastic_strain", self.plastic_output_volumetric_vjp)

    def differentiate_plastic_step_parameters(self, loss_gradient, state_vjp):
        """Pull one accepted plastic step back to its material input state."""
        self._validate_differentiable_friction()
        # ponytail: checkpoint/replay stays external until particle transfer
        # exposes its own x/v/a VJP; this method is the exact material-state step.
        if type(self.mpm.material).__name__ not in {
            "FiniteStrainDruckerPragerModel",
            "FiniteStrainVonMisesModel",
        }:
            raise ValueError(
                "accepted plastic-step differentiation supports "
                "Drucker-Prager and von Mises; MCC is intentionally excluded"
            )
        self._load_plastic_output_state_vjp(state_vjp)
        self.material_parameter_vjp.fill(0.0)
        self._differentiate_plastic_commit_state()
        active_dof = int(self.mpm.active_dof)
        if isinstance(loss_gradient, ti.ScalarField):
            if len(loss_gradient.shape) != 1 or int(loss_gradient.shape[0]) < active_dof:
                raise ValueError("loss_gradient field must cover all active MPM degrees of freedom")
            copy_field(active_dof, self.plastic_step_rhs, loss_gradient)
        else:
            values = np.asarray(loss_gradient, dtype=np.float64).reshape(-1)
            if values.size != active_dof or not np.all(np.isfinite(values)):
                raise ValueError("loss_gradient must be finite and match the active MPM degrees of freedom")
            padded = np.zeros(self.mpm.degree_of_freedom, dtype=np.float64)
            padded[:active_dof] = values
            self.plastic_step_rhs.from_numpy(padded)
        add_field(
            active_dof,
            self.plastic_step_rhs,
            self.plastic_step_rhs,
            self.plastic_commit_grid_vjp,
        )
        adjoint = self.solve_plastic_equilibrium_adjoint(self.plastic_step_rhs)
        self.pullback_plastic_equilibrium_from_current_adjoint_device(reset_material_parameters=False)
        particle_num = int(self.mpm.particleNum[0])
        material_values = np.asarray(self.material_parameter_vjp[None], dtype=np.float64)
        if type(self.mpm.material).__name__ == "FiniteStrainDruckerPragerModel":
            material_names = {
                "cohesion": float(material_values[2]),
                "friction_angle_degrees": float(material_values[3]),
            }
        else:
            material_names = {
                "yield_stress": float(material_values[2]),
                "hardening_modulus": float(material_values[3]),
            }
        friction_values = np.asarray(self.friction_parameter_vjp[None], dtype=np.float64)
        result = {
            "gravity": np.asarray(self.gravity_vjp[None], dtype=np.float64),
            "material_parameters": material_values,
            "young_modulus": float(material_values[0]),
            "poisson_ratio": float(material_values[1]),
            **material_names,
            "adjoint": adjoint,
            "plastic_deformation_inverse": self.plastic_inverse_vjp.to_numpy()[:particle_num].copy(),
            "equivalent_plastic_strain": self.plastic_equivalent_strain_vjp.to_numpy()[:particle_num].copy(),
            "volumetric_plastic_strain": self.plastic_volumetric_strain_vjp.to_numpy()[:particle_num].copy(),
            "deformation_gradient": self.plastic_deformation_vjp.to_numpy()[:particle_num].copy(),
            "plastic_history": "accepted_step_input_vjp",
            "friction_coefficient": float(friction_values[0]),
        }
        self._combine_plastic_step_vjp()
        result.update(
            {
                "deformation_gradient": self.plastic_deformation_vjp.to_numpy()[:particle_num].copy(),
                "plastic_deformation_inverse": self.plastic_inverse_vjp.to_numpy()[:particle_num].copy(),
                "equivalent_plastic_strain": self.plastic_equivalent_strain_vjp.to_numpy()[:particle_num].copy(),
                "volumetric_plastic_strain": self.plastic_volumetric_strain_vjp.to_numpy()[:particle_num].copy(),
                "plastic_history": "accepted_step_input_vjp",
            }
        )
        return result

    def _validate_plastic_particle_adjoint_configuration(self):
        self._validate_differentiable_friction()
        if type(self.mpm.material).__name__ not in {
            "FiniteStrainDruckerPragerModel",
            "FiniteStrainVonMisesModel",
        }:
            raise ValueError(
                "particle-state differentiation supports Drucker-Prager " "and von Mises; MCC is intentionally excluded"
            )
        if not config.DYNAMIC or self.mpm.configuration != "UL":
            raise ValueError("particle-state differentiation requires dynamic Direct ULMPM")
        if self.mpm.is_axisymmetric:
            raise ValueError("axisymmetric particle-state differentiation is not implemented")
        if self.mpm.velocity_proj:
            raise ValueError("particle-state differentiation currently requires " "velocity_projection=False")
        if self.mpm.shape_function_name == "gimp":
            # ponytail: add the GIMP second derivative only when Direct GIMP
            # supplies a nonzero particle domain instead of lp=0.
            raise ValueError(
                "particle-state differentiation supports linear and " "quadratic B-spline transfer, not GIMP"
            )

    def prepare_plastic_particle_step_adjoint_device(self):
        """Build the MPM equilibrium RHS from already loaded output seeds."""
        self._validate_plastic_particle_adjoint_configuration()
        self.material_parameter_vjp.fill(0.0)
        self._differentiate_plastic_commit_state()
        self._differentiate_particle_advection()
        return self.trajectory_grid_vjp

    def finish_plastic_particle_step_adjoint_device(self):
        """Finish a particle-step pullback from the current MPM adjoint."""
        self.pullback_plastic_equilibrium_from_current_adjoint_device(reset_material_parameters=False)
        self._differentiate_inertia_input_state()
        self._differentiate_material_position()
        if self.curr_barrier_contact_num > 0:
            if hasattr(self, "barrier_position_vjp_flat"):
                self.barrier_position_vjp_flat.fill(0.0)
                self._differentiate_barrier_position()
                self._accumulate_barrier_position_vjp()
            else:
                # Lightweight external-coupling test doubles may only expose
                # the original contact hook; preserve that protocol.
                self._differentiate_barrier_position()
        if self.activate_fric and self.curr_friction_contact_num > 0:
            self._differentiate_lagged_friction_position()
        if self.mpm.tractionNum[0] > 0:
            self._differentiate_traction_position()
        self._differentiate_p2g_input_state()
        self._combine_plastic_step_vjp()

    def pullback_plastic_particle_step_device(self):
        """Reverse one loaded particle step without host gradient staging."""
        equilibrium_rhs = self.prepare_plastic_particle_step_adjoint_device()
        self.solve_plastic_equilibrium_adjoint(equilibrium_rhs)
        self.finish_plastic_particle_step_adjoint_device()

    @ti.kernel
    def promote_plastic_particle_input_vjp_device(self):
        """Use this step's input VJP as the preceding step's output seed."""
        for particle in range(self.mpm.particleNum[0]):
            self.particle_output_position_vjp[particle] = self.particle_position_vjp[particle]
            self.particle_output_velocity_vjp[particle] = self.particle_velocity_vjp[particle]
            self.particle_output_acceleration_vjp[particle] = self.particle_acceleration_vjp[particle]
            self.plastic_output_deformation_vjp[particle] = self.plastic_deformation_vjp[particle]
            self.plastic_output_inverse_vjp[particle] = self.plastic_inverse_vjp[particle]
            self.plastic_output_equivalent_vjp[particle] = self.plastic_equivalent_strain_vjp[particle]
            self.plastic_output_volumetric_vjp[particle] = self.plastic_volumetric_strain_vjp[particle]

    def differentiate_plastic_particle_step_parameters(self, state_vjp):
        """Pull one accepted DP/VM step through transfer, solve, and commit."""
        self._validate_plastic_particle_adjoint_configuration()
        self._load_particle_output_state_vjp(state_vjp)
        self._load_plastic_output_state_vjp(state_vjp)
        equilibrium_rhs = self.prepare_plastic_particle_step_adjoint_device()
        adjoint = self.solve_plastic_equilibrium_adjoint(equilibrium_rhs)
        self.finish_plastic_particle_step_adjoint_device()
        particle_num = int(self.mpm.particleNum[0])
        friction_values = np.asarray(self.friction_parameter_vjp[None], dtype=np.float64)
        material_values = np.asarray(self.material_parameter_vjp[None], dtype=np.float64)
        if type(self.mpm.material).__name__ == "FiniteStrainDruckerPragerModel":
            material_names = {
                "cohesion": float(material_values[2]),
                "friction_angle_degrees": float(material_values[3]),
            }
        else:
            material_names = {
                "yield_stress": float(material_values[2]),
                "hardening_modulus": float(material_values[3]),
            }
        return {
            "gravity": np.asarray(self.gravity_vjp[None], dtype=np.float64),
            "material_parameters": material_values,
            "young_modulus": float(material_values[0]),
            "poisson_ratio": float(material_values[1]),
            **material_names,
            "friction_coefficient": float(friction_values[0]),
            "adjoint": adjoint,
            "position": self.particle_position_vjp.to_numpy()[:particle_num].copy(),
            "velocity": self.particle_velocity_vjp.to_numpy()[:particle_num].copy(),
            "acceleration": self.particle_acceleration_vjp.to_numpy()[:particle_num].copy(),
            "deformation_gradient": self.plastic_deformation_vjp.to_numpy()[:particle_num].copy(),
            "plastic_deformation_inverse": self.plastic_inverse_vjp.to_numpy()[:particle_num].copy(),
            "equivalent_plastic_strain": self.plastic_equivalent_strain_vjp.to_numpy()[:particle_num].copy(),
            "volumetric_plastic_strain": self.plastic_volumetric_strain_vjp.to_numpy()[:particle_num].copy(),
            "plastic_history": "accepted_particle_step_input_vjp",
        }

    def _differentiate_step(self, step, loss_gradient, mode, verbose, plastic_state_vjp=None):
        self._validate_differentiable_friction()
        if self.pending_adjoint_seed is not None:
            raise RuntimeError("A Direct IPC-MPM differentiation step is already active")
        self.pending_adjoint_seed = loss_gradient
        self.pending_adjoint_mode = mode
        self.pending_plastic_state_vjp = plastic_state_vjp
        self.last_elastic_differentiation = None
        try:
            step(verbose=verbose)
            if self.last_elastic_differentiation is None:
                raise RuntimeError("Direct IPC-MPM step completed without its pre-commit adjoint")
            return self.last_elastic_differentiation
        finally:
            self.pending_adjoint_seed = None
            self.pending_adjoint_mode = None
            self.pending_plastic_state_vjp = None

    def differentiate_elastic_step(self, step, loss_gradient, verbose=True):
        """Run ``step`` and evaluate its elastic adjoint before commit."""
        return self._differentiate_step(step, loss_gradient, "elastic", verbose)

    def differentiate_plastic_equilibrium_step(self, step, loss_gradient, verbose=True):
        """Run one plastic step and differentiate equilibrium before commit."""
        return self._differentiate_step(step, loss_gradient, "plastic", verbose)

    def differentiate_plastic_step(self, step, loss_gradient, state_vjp, verbose=True):
        """Run and reverse one accepted DP/VM update, including history."""
        return self._differentiate_step(
            step,
            loss_gradient,
            "plastic_state",
            verbose,
            plastic_state_vjp=state_vjp,
        )

    def differentiate_plastic_particle_step(self, step, state_vjp, verbose=True):
        """Run and reverse one complete Direct ULMPM particle update."""
        return self._differentiate_step(
            step,
            self.trajectory_grid_vjp,
            "plastic_particle_state",
            verbose,
            plastic_state_vjp=state_vjp,
        )

    def differentiate_before_commit(self):
        """Consume a pending seed while the converged iterate is uncommitted."""
        if self.pending_adjoint_seed is None:
            return
        try:
            if self.pending_adjoint_mode == "plastic_particle_state":
                self.last_elastic_differentiation = self.differentiate_plastic_particle_step_parameters(
                    self.pending_plastic_state_vjp
                )
            elif self.pending_adjoint_mode == "plastic_particle_device":
                self.pullback_plastic_particle_step_device()
                self.last_elastic_differentiation = True
            elif self.pending_adjoint_mode == "plastic_state":
                self.last_elastic_differentiation = self.differentiate_plastic_step_parameters(
                    self.pending_adjoint_seed,
                    self.pending_plastic_state_vjp,
                )
            elif self.pending_adjoint_mode == "plastic":
                self.last_elastic_differentiation = self.differentiate_plastic_equilibrium_parameters(
                    self.pending_adjoint_seed
                )
            else:
                self.last_elastic_differentiation = self.differentiate_elastic_parameters(self.pending_adjoint_seed)
        except BaseException:
            self._restore_grid_displacement(self.device_grid_disp_snapshot)
            raise

    def _assemble_cuda_monolithic_matrix(self, active_dof):
        """Merge all current raw blocks and impose constraints on the GPU."""
        active_nodes = int(active_dof) // config.DIM
        matrix = self.system_hash_matrix
        matrix.reset_system()
        matrix.append_raw_from(
            self.mpm.hash_matrix,
            active_nodes=active_nodes,
        )
        if self.activate_barrier and self.curr_barrier_contact_num > 0:
            matrix.append_raw_from(
                self.barrier_hash_matrix,
                active_nodes=active_nodes,
            )
        if self.activate_fric and self.curr_friction_contact_num > 0:
            matrix.append_raw_from(
                self.friction_hash_matrix,
                active_nodes=active_nodes,
            )
        # Lagged IPC sources assemble both physical triangles so the same
        # fields can also feed the exact fully implicit path.  Canonicalize the
        # lagged destination before constraints: its DBC kernel below is
        # mirror-aware and therefore uses exactly the operator PCG will solve.
        if matrix.full_symmetric_input:
            matrix.canonicalize_full_symmetric_input()
        if self.system_physical_rhs is None:
            # CPU-backend tests deliberately exercise the production device
            # dataflow by enabling it after construction.
            self.system_physical_rhs = ti.field(ti.f64, shape=self.mpm.degree_of_freedom)
        self._copy_cuda_physical_rhs(int(active_dof))
        if self.mpm.dirichlet.num > 0:
            self._apply_cuda_monolithic_dirichlet(int(active_dof))
        matrix.finalize_taichi_assembly()
        return matrix

    @ti.kernel
    def _copy_cuda_physical_rhs(self, active_dof: ti.i32):
        for dof in range(active_dof):
            self.system_physical_rhs[dof] = self.mpm.rhs[dof]

    @ti.kernel
    def _apply_cuda_monolithic_dirichlet(self, active_dof: int):
        """Apply elimination after body and every contact block are merged."""
        active_nodes = active_dof // config.DIM
        for block in range(active_nodes):
            base_grid = self.mpm.dof2node[block]
            for d1 in ti.static(range(config.DIM)):
                row = config.DIM * block + d1
                row_dof = config.DIM * base_grid + d1
                for d2 in ti.static(range(config.DIM)):
                    col_dof = config.DIM * base_grid + d2
                    component = d1 * config.DIM + d2
                    value = self.system_hash_matrix.diag[block][component]
                    if self.mpm.dirichlet.node[col_dof] == 1 or self.mpm.dirichlet.node[row_dof] == 1:
                        fixed_correction = (
                            self.mpm.dirichlet.value[col_dof] - self.mpm.grid_disp[config.DIM * block + d2]
                        )
                        self.mpm.rhs[row] -= value * fixed_correction
                        self.system_hash_matrix.diag[block][component] = 0.0
                if self.mpm.dirichlet.node[row_dof] == 1:
                    self.system_hash_matrix.diag[block][d1 * config.DIM + d1] = 1.0

        raw_nnz = self.system_hash_matrix.raw_non_diag_count[0]
        for entry in range(raw_nnz):
            if entry < self.system_hash_matrix.non_diag.blockI.shape[0]:
                block_i = self.system_hash_matrix.non_diag.blockI[entry]
                block_j = self.system_hash_matrix.non_diag.blockJ[entry]
                if 0 <= block_i and block_i < active_nodes and 0 <= block_j and block_j < active_nodes:
                    row_grid = self.mpm.dof2node[block_i]
                    col_grid = self.mpm.dof2node[block_j]
                    for d1 in ti.static(range(config.DIM)):
                        row = config.DIM * block_i + d1
                        row_dof = config.DIM * row_grid + d1
                        for d2 in ti.static(range(config.DIM)):
                            col = config.DIM * block_j + d2
                            col_dof = config.DIM * col_grid + d2
                            component = d1 * config.DIM + d2
                            value = self.system_hash_matrix.non_diag.blockH[entry][component]
                            column_fixed = self.mpm.dirichlet.node[col_dof] == 1
                            row_fixed = self.mpm.dirichlet.node[row_dof] == 1
                            if column_fixed:
                                column_correction = (
                                    self.mpm.dirichlet.value[col_dof] - self.mpm.grid_disp[config.DIM * block_j + d2]
                                )
                                ti.atomic_add(
                                    self.mpm.rhs[row],
                                    -value * column_correction,
                                )
                            if ti.static(self.system_hash_matrix.matrix_symmetric):
                                # Only the upper block is stored.  Its
                                # transposed mirror contributes to the other
                                # RHS orientation when the row DOF is fixed.
                                if row_fixed:
                                    row_correction = (
                                        self.mpm.dirichlet.value[row_dof]
                                        - self.mpm.grid_disp[config.DIM * block_i + d1]
                                    )
                                    ti.atomic_add(
                                        self.mpm.rhs[col],
                                        -value * row_correction,
                                    )
                            if column_fixed or row_fixed:
                                self.system_hash_matrix.non_diag.blockH[entry][component] = 0.0

        for dof in range(active_dof):
            grid_dof = config.DIM * self.mpm.dof2node[dof // config.DIM] + dof % config.DIM
            if self.mpm.dirichlet.node[grid_dof] == 1:
                self.mpm.rhs[dof] = self.mpm.dirichlet.value[grid_dof] - self.mpm.grid_disp[dof]

    def solve_current_system(self, grid_disp=None):
        """Return a Newton correction without applying it to ``grid_disp``."""
        matrix = self.assemble_current_system(grid_disp)
        active_dof = self.mpm.active_dof
        if self.cuda_monolithic_solver:
            # Keep this assignment adjacent to the solve so a changed contact
            # mode cannot accidentally reuse the wrong Krylov method. It does
            # not alter, mirror, or project any matrix block.
            krylov_solver = self._cuda_monolithic_krylov_solver()
            matrix.solver = krylov_solver
            result = matrix.solve_flat_system(
                self.mpm.rhs,
                self.mpm.incre_resolution,
                active_nodes=active_dof // config.DIM,
                tol=self.mpm.linear_solver_tolerance,
                maxiter=self.mpm.linear_solver_max_iters,
                return_solution=False,
            )
            result["backend"] = (
                "taichi_cuda_monolithic_pcg" if krylov_solver == "PCG" else "taichi_cuda_monolithic_bicgstab"
            )
            if not result["converged"]:
                raise RuntimeError(
                    f"SoftParticle IPC CUDA {krylov_solver} did not converge: "
                    f"residual={result['residual']:.6e}, "
                    f"iterations={result['iterations']}"
                )
            return result
        raise RuntimeError("SoftParticle IPC linear solves must use the device monolithic " "HashTriplet backend")

    def _correction_inf_norm(self, correction, *, load_solution=False):
        if isinstance(correction, dict):
            return float(correction["solution_inf_norm"])
        if load_solution:
            self.mpm.incre_resolution.from_numpy(correction)
        return float(np.linalg.norm(correction, np.inf))

    def _current_rhs_l2_norm(self):
        if self.cuda_monolithic_solver:
            return self.system_hash_matrix.flat_l2_norm(self.mpm.rhs, self.mpm.active_dof)
        raise RuntimeError("SoftParticle IPC residual norms require the device monolithic " "backend")

    def updated_friction_system_residual(self):
        """Probe convergence with the system assembled from the updated cache."""
        correction = self.solve_current_system()
        return self._correction_inf_norm(correction) / self.mpm.dt

    def solve_frozen_friction_newton(self, verbose=True):
        """Run one complete inner Newton solve with friction data held fixed."""
        iter_num = 0
        linear_solve_count = 0
        residual = np.inf
        semi_progress = 0.0
        self.last_inner_converged = False
        self.last_inner_failure_reason = "maximum_newton_iterations"
        while iter_num < self.mpm.max_iters:
            if self.is_semi and iter_num > 1 and semi_progress > 0.999:
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break
            self.mpm.incre_resolution.fill(0)
            result = self.solve_current_system()
            linear_solve_count += 1
            residual = self._correction_inf_norm(result, load_solution=True) / self.mpm.dt
            if residual < self.mpm.tol:
                self.mpm.update_grid_disp(self.mpm.active_dof, 1.0)
                copy_field(
                    self.mpm.active_dof,
                    self.mpm.grid_disp,
                    self.mpm.grid_disp_temp,
                )
                self.accept_update()
                if self.contact_converged():
                    self.last_inner_converged = True
                    self.last_inner_failure_reason = ""
                    break
                iter_num += 1
                continue

            success = self.line_search(self.mpm.active_dof, verbose)
            if self.line_search_stop_iter or not success:
                self.last_inner_failure_reason = self.line_search_stop_reason or "line_search_failed"
                break
            if self.is_semi:
                semi_progress += (1.0 - semi_progress) * self.line_search_last_alpha
            iter_num += 1
        return iter_num, residual, linear_solve_count

    def _snapshot_grid_displacement(self):
        """Best-effort snapshot used to make a failed time step transactional."""
        device_snapshot = getattr(self, "device_grid_disp_snapshot", None)
        if device_snapshot is not None:
            copy_field(
                self.mpm.degree_of_freedom,
                device_snapshot,
                self.mpm.grid_disp,
            )
            return device_snapshot
        try:
            return self.mpm.grid_disp.to_numpy().copy()
        except (AttributeError, TypeError):
            # Lightweight orchestration tests use opaque sentinels instead of
            # Taichi fields; no mutation occurs in those test controllers.
            return None

    def _restore_grid_displacement(self, displacement):
        if displacement is None:
            return
        if displacement is getattr(self, "device_grid_disp_snapshot", None):
            copy_field(
                self.mpm.degree_of_freedom,
                self.mpm.grid_disp,
                displacement,
            )
            copy_field(
                self.mpm.degree_of_freedom,
                self.mpm.grid_disp_temp,
                displacement,
            )
            return
        self.mpm.grid_disp.from_numpy(displacement)
        self.mpm.grid_disp_temp.from_numpy(displacement)

    @ti.kernel
    def _active_field_inf_norm(self, active_dof: ti.i32, values: ti.template()) -> ti.f64:
        maximum = 0.0
        for dof in range(active_dof):
            ti.atomic_max(maximum, ti.abs(values[dof]))
        return maximum

    @ti.kernel
    def _active_field_is_finite(self, active_dof: ti.i32, values: ti.template()) -> ti.i32:
        invalid = 0
        for dof in range(active_dof):
            value = values[dof]
            if not (value == value and ti.abs(value) < 1.0e300):
                ti.atomic_max(invalid, 1)
        return 1 - invalid

    @ti.kernel
    def _fully_implicit_constrained_squared_residual(
        self,
        active_dof: ti.i32,
        physical_residual: ti.template(),
        displacement: ti.template(),
    ) -> ti.f64:
        """Squared residual of the free equations plus exact constraints."""
        value = 0.0
        for dof in range(active_dof):
            residual = physical_residual[dof]
            if ti.static(self.mpm.dirichlet.num > 0):
                block = dof // config.DIM
                component = dof % config.DIM
                grid_dof = config.DIM * self.mpm.dof2node[block] + component
                if self.mpm.dirichlet.node[grid_dof] == 1:
                    residual = self.mpm.dirichlet.value[grid_dof] - displacement[dof]
            value += residual * residual
        return value

    @ti.kernel
    def _fully_implicit_cuda_merit_slope_kernel(self, active_dof: ti.i32) -> ti.f64:
        """Return exact ``d(0.5 ||R||^2)[p] = -R^T Kp`` on device."""
        slope = 0.0
        for dof in range(active_dof):
            block = dof // config.DIM
            component = dof % config.DIM
            residual = self.system_physical_rhs[dof]
            matrix_direction = (
                self.system_hash_matrix.Ax[block][component] + self.system_physical_rhs[dof] - self.mpm.rhs[dof]
            )
            if ti.static(self.mpm.dirichlet.num > 0):
                grid_dof = config.DIM * self.mpm.dof2node[block] + component
                if self.mpm.dirichlet.node[grid_dof] == 1:
                    residual = self.mpm.dirichlet.value[grid_dof] - self.mpm.grid_disp[dof]
                    # The eliminated row is the exact constraint Jacobian.
                    matrix_direction = self.system_hash_matrix.Ax[block][component]
            slope -= residual * matrix_direction
        return slope

    def _fully_implicit_cuda_merit_slope(self, active_dof):
        active_dof = int(active_dof)
        active_nodes = active_dof // config.DIM
        matrix = self.system_hash_matrix
        matrix.matvec(
            active_nodes,
            int(matrix.non_diag.element_pair_num[0]),
            matrix.x,
            matrix.Ax,
        )
        return float(self._fully_implicit_cuda_merit_slope_kernel(active_dof))

    def _fully_implicit_cuda_residual_norm(self, physical_residual, displacement):
        squared = float(
            self._fully_implicit_constrained_squared_residual(
                int(self.mpm.active_dof),
                physical_residual,
                displacement,
            )
        )
        return float(math.sqrt(max(squared, 0.0)))

    def _fully_implicit_residual_line_search(self, current_residual, directional_derivative, verbose):
        """CCD-filtered exact residual-merit Armijo backtracking."""
        active_dof = self.mpm.active_dof
        current_merit = 0.5 * current_residual * current_residual
        directional_derivative = float(directional_derivative)
        if not np.isfinite(current_merit) or not np.isfinite(directional_derivative):
            raise RuntimeError("fully implicit residual merit or directional derivative " "is non-finite")
        if directional_derivative >= 0.0:
            raise RuntimeError("fully implicit Newton correction is not a descent " "direction for 0.5 * ||R||^2")
        alpha = float(self.ccd(active_dof))
        alpha = min(max(alpha, 0.0), 1.0)
        self.last_fully_implicit_backtracks = 0
        for backtrack in range(self.fully_implicit_max_backtracks + 1):
            if alpha < self.line_search_min_alpha:
                break
            self.mpm.update_grid_disp(active_dof, alpha)
            try:
                self.assemble_current_system(self.mpm.grid_disp_temp, need_matrix=False)
            except IPCInfeasibleContactState:
                trial_residual = float("inf")
                trial_merit = float("inf")
            else:
                if self.cuda_monolithic_solver:
                    trial_residual = self._fully_implicit_cuda_residual_norm(self.mpm.rhs, self.mpm.grid_disp_temp)
                else:
                    trial_residual = self._current_rhs_l2_norm()
                trial_merit = 0.5 * trial_residual * trial_residual
            target = current_merit + self.fully_implicit_armijo * alpha * directional_derivative
            if np.isfinite(trial_merit) and trial_merit <= target:
                copy_field(
                    active_dof,
                    self.mpm.grid_disp,
                    self.mpm.grid_disp_temp,
                )
                self.last_fully_implicit_residual = trial_residual
                self.last_fully_implicit_backtracks = backtrack
                return True
            alpha *= self.fully_implicit_backtrack
            if verbose:
                print(
                    "fully implicit friction residual backtrack: "
                    f"alpha={alpha:.6e}, merit={trial_merit:.6e}, "
                    f"Armijo bound={target:.6e}"
                )
        # Restore current contact/residual state; the accepted displacement was
        # never overwritten when all trial steps failed.
        self.assemble_current_system(self.mpm.grid_disp, need_matrix=False)
        self.last_fully_implicit_backtracks = self.fully_implicit_max_backtracks + 1
        return False

    def _validate_fully_implicit_feasible_state(self):
        """Require point-plane gaps strictly above the configured ``dmin``."""
        self.update_particle_pos(self.mpm.grid_disp)
        surface_count = int(self.mpm.total_surface_num)
        if surface_count <= 0:
            raise RuntimeError("fully implicit IPC friction requires at least one surface point")
        self._compute_minimum_candidate_distance()
        minimum_gap = float(self.minimum_candidate_distance[0])
        dmin = float(self.barrier.dmin[0])
        safe_floor = max(
            1.0e-12 * float(self.barrier.dhat[0]),
            1.0e-14,
        )
        if (
            int(self.candidate_state_invalid[0]) != 0
            or not np.isfinite(minimum_gap)
            or minimum_gap <= dmin + safe_floor
        ):
            raise RuntimeError(
                "fully implicit IPC friction requires a strictly feasible "
                f"positive starting gap above dmin (minimum={minimum_gap:.6e}, "
                f"dmin={dmin:.6e}, margin={safe_floor:.6e})"
            )

    def _restore_fully_implicit_displacement(self, displacement):
        self._restore_grid_displacement(displacement)

    @ti.kernel
    def _build_fully_implicit_velocity_predictor(self):
        """Map the paper's initial guess v_(n+1)^0 = v_n to displacement."""
        for block in range(self.mpm.active_dof // config.DIM):
            grid_id = self.mpm.dof2node[block]
            for component in ti.static(range(config.DIM)):
                dof = config.DIM * block + component
                target_displacement = (
                    (1.0 - self.endpoint_velocity_previous_scale) * self.mpm.grid[grid_id].v[component]
                    - self.endpoint_velocity_acceleration_scale * self.mpm.grid[grid_id].a[component]
                ) / self.endpoint_velocity_displacement_scale
                self.mpm.incre_resolution[dof] = target_displacement - self.mpm.grid_disp[dof]

    def _initialize_fully_implicit_velocity_guess(self):
        """Use the paper's velocity guess, capped by IPC/material CCD.

        A nonzero caller-supplied displacement is respected. For the usual
        zero initial displacement, the mapped ``v_(n+1)=v_n`` predictor is
        truncated along its segment whenever IPC or material feasibility
        requires it.
        """
        if not self.fully_implicit_velocity_predictor:
            return
        active_dof = self.mpm.active_dof
        current_norm = float(self._active_field_inf_norm(active_dof, self.mpm.grid_disp))
        if current_norm > 1.0e-14:
            return

        self.update_particle_pos(self.mpm.grid_disp)
        if not self.rebuild_barrier_contacts():
            raise RuntimeError("fully implicit velocity predictor requires a feasible " "starting configuration")
        self._build_fully_implicit_velocity_predictor()
        if int(self._active_field_is_finite(active_dof, self.mpm.incre_resolution)) == 0:
            raise RuntimeError("fully implicit velocity predictor produced a non-finite step")
        predictor_norm = float(self._active_field_inf_norm(active_dof, self.mpm.incre_resolution))
        if predictor_norm <= 1.0e-30:
            return

        alpha = min(max(float(self.ccd(active_dof)), 0.0), 1.0)
        while alpha >= self.line_search_min_alpha:
            self.mpm.update_grid_disp(active_dof, alpha)
            self.update_particle_pos(self.mpm.grid_disp_temp)
            if self.rebuild_barrier_contacts():
                copy_field(
                    active_dof,
                    self.mpm.grid_disp,
                    self.mpm.grid_disp_temp,
                )
                return
            alpha *= self.fully_implicit_backtrack

        # The zero displacement was already checked feasible and is a valid
        # fallback initial iterate if every positive predictor step is blocked.
        copy_field(
            self.mpm.degree_of_freedom,
            self.mpm.grid_disp_temp,
            self.mpm.grid_disp,
        )
        self.update_particle_pos(self.mpm.grid_disp)
        self.rebuild_barrier_contacts()

    def solve_fully_implicit_friction_newton(self, verbose=True):
        """Solve the restricted current point-plane friction residual."""
        self._validate_fully_implicit_configuration()
        self._validate_fully_implicit_feasible_state()
        initial_displacement = self._snapshot_grid_displacement()
        self.line_search_stop_iter = False
        self.line_search_stop_reason = ""
        self.last_newton_iterations = 0
        self.last_newton_residual = np.inf
        self.last_friction_iterations = 1
        self.last_friction_residual = np.inf
        self.last_friction_converged = False
        self.last_fully_implicit_residual = np.inf
        self.last_fully_implicit_initial_residual = np.inf

        iteration = 0
        correction_residual = np.inf
        initial_residual = None
        residual_tolerance = np.inf
        try:
            self._initialize_fully_implicit_velocity_guess()
            while iteration < self.mpm.max_iters:
                self.mpm.incre_resolution.fill(0)
                correction = self.solve_current_system(self.mpm.grid_disp)
                self.last_newton_iterations += 1
                if self.cuda_monolithic_solver:
                    current_residual = self._fully_implicit_cuda_residual_norm(
                        self.system_physical_rhs,
                        self.mpm.grid_disp,
                    )
                    merit_slope = self._fully_implicit_cuda_merit_slope(self.mpm.active_dof)
                else:
                    current_residual = self._current_rhs_l2_norm()
                    # The CPU oracle uses a direct solve.  With its exact
                    # Newton correction Kp=R, the residual-merit slope is
                    # -R^T R.  The CUDA production route above evaluates Kp
                    # explicitly so an inexact Krylov solve is also handled.
                    merit_slope = -current_residual * current_residual
                correction_residual = self._correction_inf_norm(correction, load_solution=True) / self.mpm.dt
                if not np.isfinite(current_residual) or not np.isfinite(correction_residual):
                    raise RuntimeError(
                        "fully implicit IPC friction produced a non-finite " "residual or Newton correction"
                    )
                if initial_residual is None:
                    initial_residual = current_residual
                    self.last_fully_implicit_initial_residual = current_residual
                residual_tolerance = (
                    self.fully_implicit_residual_atol + self.fully_implicit_residual_rtol * initial_residual
                )
                self.last_fully_implicit_residual = current_residual
                self.last_friction_residual = current_residual
                self.last_newton_residual = correction_residual
                if current_residual <= residual_tolerance:
                    self.last_friction_converged = True
                    break

                if not self._fully_implicit_residual_line_search(current_residual, merit_slope, verbose):
                    self.line_search_stop_iter = True
                    self.line_search_stop_reason = "fully_implicit_residual_backtracking"
                    break
                iteration += 1
        except Exception:
            self._restore_fully_implicit_displacement(initial_displacement)
            raise

        if not self.last_friction_converged:
            self._restore_fully_implicit_displacement(initial_displacement)
            reason = self.line_search_stop_reason or "maximum_newton_iterations"
            raise RuntimeError(
                "fully implicit IPC friction did not converge; the time-step "
                f"displacement was rolled back (reason={reason}, "
                f"residual={self.last_fully_implicit_residual:.6e}, "
                f"tolerance={residual_tolerance:.6e})"
            )

        return iteration, correction_residual

    def solve_friction_step(self, verbose=True):
        if self.is_semi:
            self.semi_state.fill(2)
            self.semi_multiplier.fill(0.0)
            self.semi_count[None] = 0
            self.semi_overflow[None] = 0
        self.validate_strict_feasibility(self.mpm.grid_disp)
        if self.friction_mode == "fully_implicit" and self.activate_fric:
            return self.solve_fully_implicit_friction_newton(verbose)
        return self.solve_lagged_friction_fixed_point(verbose)

    def solve_lagged_friction_fixed_point(self, verbose=True):
        """Solve a time step with outer updates of lagged IPC friction data.

        Every outer iteration fully converges (or exhausts) the conservative
        inner Newton problem.  Only then are the contact normal and
        ``mu_lambda`` rebuilt.  The updated system is solved once as a residual
        probe, but that correction is not applied; this keeps the formulation
        lagged rather than turning it into a fully implicit friction Hessian.
        """
        self.last_newton_iterations = 0
        self.last_newton_residual = np.inf
        self.last_friction_iterations = 0
        self.last_friction_residual = np.inf
        self.last_friction_converged = False
        initial_displacement = self._snapshot_grid_displacement()

        try:
            if not self.activate_fric:
                iter_num, residual, solve_count = self.solve_frozen_friction_newton(verbose)
                self.last_newton_iterations = solve_count
                self.last_newton_residual = residual
                if not bool(getattr(self, "last_inner_converged", True)):
                    raise RuntimeError(
                        "lagged IPC inner Newton solve did not converge "
                        f"(reason={self.last_inner_failure_reason}, "
                        f"residual={residual:.6e})"
                    )
                return iter_num, residual

            self.begin_friction_step(self.mpm.grid_disp)
            total_iter_num = 0
            residual = np.inf
            for outer_iteration in range(self.friction_outer_iteration_limit()):
                iter_num, residual, solve_count = self.solve_frozen_friction_newton(verbose)
                if not bool(getattr(self, "last_inner_converged", True)):
                    raise RuntimeError(
                        "lagged IPC inner Newton solve did not converge "
                        f"(outer_iteration={outer_iteration + 1}, "
                        f"reason={self.last_inner_failure_reason}, "
                        f"residual={residual:.6e})"
                    )
                total_iter_num += iter_num
                self.last_newton_iterations += solve_count
                self.last_newton_residual = residual
                self.last_friction_iterations = outer_iteration + 1

                # The reference position remains frozen, while normals, active
                # contacts, and normal-force weights are refreshed at the current
                # accepted grid displacement.
                if self.pending_adjoint_seed is not None or self.__dict__.get("trajectory_capture_active", False):
                    self._backup_lagged_friction_for_adjoint()
                self.refresh_friction_cache(self.mpm.grid_disp)
                self.last_friction_residual = self.updated_friction_system_residual()
                if not np.isfinite(self.last_friction_residual):
                    raise RuntimeError("lagged IPC updated friction residual is non-finite")
                if self.last_friction_residual <= self.friction_tolerance:
                    self.last_friction_converged = True
                    break

            if self.friction_iterations == -1 and not self.last_friction_converged:
                raise RuntimeError(
                    "lagged IPC friction fixed-point iteration did not converge "
                    f"within safety cap {self.friction_max_iterations} "
                    f"(residual={self.last_friction_residual:.6e}, "
                    f"tolerance={self.friction_tolerance:.6e})"
                )
            return total_iter_num, residual
        except Exception:
            self._restore_grid_displacement(initial_displacement)
            raise

    def assemble_ground_barrier_matrix(self):
        self._assemble_ground_barrier_system(True)

    @ti.kernel
    def _assemble_ground_barrier_system(self, need_matrix: ti.template()):
        support_capacity = ti.static(self.mpm.shape_func.max_node_per_particle)
        pair_capacity = ti.static(support_capacity * support_capacity)
        raw_base = 0
        raw_end = 0
        stored_end = 0
        if ti.static(need_matrix):
            raw_base = self.barrier_hash_matrix.raw_non_diag_count[0]
            raw_end = raw_base + self.gbarrierNum[0] * pair_capacity
            stored_end = ti.min(
                raw_end,
                self.barrier_hash_matrix.non_diag.blockI.shape[0],
            )
            self.barrier_hash_matrix.raw_non_diag_count[0] = stored_end
            if raw_end > self.barrier_hash_matrix.non_diag.blockI.shape[0]:
                self.barrier_hash_matrix.overflow[0] = 1
            for slot in range(raw_base, stored_end):
                self.barrier_hash_matrix.non_diag.blockI[slot] = -1
                self.barrier_hash_matrix.non_diag.blockJ[slot] = -1
                for component in ti.static(range(config.DIM * config.DIM)):
                    self.barrier_hash_matrix.non_diag.blockH[slot][component] = 0.0

        for c in range(self.gbarrierNum[0]):
            s = self.gbarrier[c].surfaceID
            i = self.mpm.surface_id[s]
            wallID = self.gbarrier[c].wallID
            dist = self.gbarrier[c].distance
            slot = self.gbarrier[c].semi_slot
            value = dist
            if ti.static(self.is_semi):
                value -= self.barrier.activation_distance_term()
                ti.atomic_max(self.semi_constraint_violation[None], ti.max(-value, 0.0))
            measure = point_contact_measure(self.mpm.surface_measure[s])
            _, first, second = self._normal_terms(value, slot)
            barrier_gradient = measure * first
            ddistance_dpoint = self.gderivative.Ddistance_div_Dpoint(wallID)

            for j in range(self.mpm.offset[i]):
                base_jnode = self.mpm.LnID[i, j]
                base_joffset = self.mpm.node2dof[base_jnode] - 1
                dofs = config.DIM * (self.mpm.node2dof[base_jnode] - 1)
                dpoint_dx1 = self.mpm.shape[i, j] * ti.Matrix.identity(ti.f64, config.DIM)
                ddistance_dx1 = ddistance_dpoint @ dpoint_dx1
                dPsi_dx = barrier_gradient * ddistance_dx1
                for d in ti.static(range(config.DIM)):
                    self.mpm.rhs[dofs + d] -= dPsi_dx[d]
                if ti.static(need_matrix):
                    barrier_hessian = measure * second
                    for k in range(self.mpm.offset[i]):
                        base_knode = self.mpm.LnID[i, k]
                        base_koffset = self.mpm.node2dof[base_knode] - 1
                        dpoint_dx2 = self.mpm.shape[i, k] * ti.Matrix.identity(ti.f64, config.DIM)
                        ddistance_dx2 = ddistance_dpoint @ dpoint_dx2
                        d2Psi_dx1dx2 = barrier_hessian * ddistance_dx1.outer_product(ddistance_dx2)
                        slot = raw_base + c * pair_capacity + j * support_capacity + k
                        if base_joffset >= 0 and base_koffset >= 0:
                            if base_joffset == base_koffset:
                                for row in ti.static(range(config.DIM)):
                                    for column in ti.static(range(config.DIM)):
                                        component = row * config.DIM + column
                                        ti.atomic_add(
                                            self.barrier_hash_matrix.diag[base_joffset][component],
                                            d2Psi_dx1dx2[row, column],
                                        )
                            elif slot < stored_end:
                                self.barrier_hash_matrix.non_diag.blockI[slot] = base_joffset
                                self.barrier_hash_matrix.non_diag.blockJ[slot] = base_koffset
                                for row in ti.static(range(config.DIM)):
                                    for column in ti.static(range(config.DIM)):
                                        component = row * config.DIM + column
                                        self.barrier_hash_matrix.non_diag.blockH[slot][component] = d2Psi_dx1dx2[
                                            row, column
                                        ]

    def assemble_particle_barrier_matrix(self):
        self._assemble_particle_barrier_system(
            True,
            self.friction_mode == "lagged",
        )

    @ti.func
    def _write_barrier_block_fixed_slot(self, slot, stored_end, block_i, block_j, block):
        if block_i >= 0 and block_j >= 0:
            if block_i == block_j:
                for row in ti.static(range(config.DIM)):
                    for column in ti.static(range(config.DIM)):
                        component = row * config.DIM + column
                        ti.atomic_add(
                            self.barrier_hash_matrix.diag[block_i][component],
                            block[row, column],
                        )
            elif slot < stored_end:
                self.barrier_hash_matrix.non_diag.blockI[slot] = block_i
                self.barrier_hash_matrix.non_diag.blockJ[slot] = block_j
                for row in ti.static(range(config.DIM)):
                    for column in ti.static(range(config.DIM)):
                        component = row * config.DIM + column
                        self.barrier_hash_matrix.non_diag.blockH[slot][component] = block[row, column]

    @ti.kernel
    def _assemble_particle_barrier_system(
        self,
        need_matrix: ti.template(),
        project_spd: ti.template(),
    ):
        support_capacity = ti.static(self.mpm.shape_func.max_node_per_particle)
        pair_capacity = ti.static(support_capacity * support_capacity)
        contact_pair_capacity = ti.static(4 * pair_capacity)
        raw_base = 0
        raw_end = 0
        stored_end = 0
        if ti.static(need_matrix):
            raw_base = self.barrier_hash_matrix.raw_non_diag_count[0]
            raw_end = raw_base + self.pbarrierNum[0] * contact_pair_capacity
            stored_end = ti.min(
                raw_end,
                self.barrier_hash_matrix.non_diag.blockI.shape[0],
            )
            self.barrier_hash_matrix.raw_non_diag_count[0] = stored_end
            if raw_end > self.barrier_hash_matrix.non_diag.blockI.shape[0]:
                self.barrier_hash_matrix.overflow[0] = 1
            for slot in range(raw_base, stored_end):
                self.barrier_hash_matrix.non_diag.blockI[slot] = -1
                self.barrier_hash_matrix.non_diag.blockJ[slot] = -1
                for component in ti.static(range(config.DIM * config.DIM)):
                    self.barrier_hash_matrix.non_diag.blockH[slot][component] = 0.0

        for c in range(self.pbarrierNum[0]):
            sp = self.pbarrier[c].masterID
            tp = self.pbarrier[c].slaveID

            ip = self.mpm.surface_id[sp]
            jp = self.mpm.surface_id[tp]

            ipos = self.mpm.p_temp[sp]
            jpos = self.mpm.p_temp[tp]

            dist = self.pbarrier[c].distance
            slot = self.pbarrier[c].semi_slot
            measure = symmetric_contact_measure(self.mpm.surface_measure[sp], self.mpm.surface_measure[tp])
            # 1st derivatives
            ddist_ip, ddist_jp = self.pderivative.Ddistance_div_Dpoint(ipos, jpos)
            value = dist
            if ti.static(self.is_semi):
                ddist_ip = self.semi_normal[slot]
                ddist_jp = -ddist_ip
                value = ddist_ip.dot(ipos - jpos) - self.barrier.activation_distance_term()
                ti.atomic_max(self.semi_constraint_violation[None], ti.max(-value, 0.0))
            _, first, second = self._normal_terms(value, slot)
            barrier_grad = measure * first
            relative_hessian = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            if ti.static(need_matrix):
                barrier_hess = measure * second
                relative_hessian = barrier_hess * ddist_ip.outer_product(ddist_ip)
                if ti.static(not self.is_semi):
                    d2dist = self.pderivative.D2distance_div_Dpoint2(ipos, jpos)
                    # For a PP stencil H_local = [I,-I]^T H_relative [I,-I].
                    relative_hessian += barrier_grad * d2dist
                    if ti.static(project_spd):
                        relative_hessian = psd_project_nd(relative_hessian)

            # ----------------------
            # i-i block
            # ----------------------
            for j in range(self.mpm.offset[ip]):
                base_jnode = self.mpm.LnID[ip, j]
                base_joffset = self.mpm.node2dof[base_jnode] - 1
                dpoint_dx1 = self.mpm.shape[ip, j] * ti.Matrix.identity(ti.f64, config.DIM)
                ddist_dx1 = ddist_ip @ dpoint_dx1
                dPsi_dx1 = barrier_grad * ddist_dx1
                for d in ti.static(range(config.DIM)):
                    self.mpm.rhs[config.DIM * base_joffset + d] -= dPsi_dx1[d]

                if ti.static(need_matrix):
                    for k in range(self.mpm.offset[ip]):
                        base_knode = self.mpm.LnID[ip, k]
                        base_koffset = self.mpm.node2dof[base_knode] - 1
                        dpoint_dx2 = self.mpm.shape[ip, k] * ti.Matrix.identity(ti.f64, config.DIM)
                        d2Psi = dpoint_dx1 @ relative_hessian @ dpoint_dx2
                        slot = raw_base + c * contact_pair_capacity + j * support_capacity + k
                        self._write_barrier_block_fixed_slot(
                            slot,
                            stored_end,
                            base_joffset,
                            base_koffset,
                            d2Psi,
                        )

            # ----------------------
            # j-j block
            # ----------------------
            for j in range(self.mpm.offset[jp]):
                base_jnode = self.mpm.LnID[jp, j]
                base_joffset = self.mpm.node2dof[base_jnode] - 1
                dpoint_dx1 = self.mpm.shape[jp, j] * ti.Matrix.identity(ti.f64, config.DIM)
                ddist_dx1 = ddist_jp @ dpoint_dx1
                dPsi_dx1 = barrier_grad * ddist_dx1
                for d in ti.static(range(config.DIM)):
                    self.mpm.rhs[config.DIM * base_joffset + d] -= dPsi_dx1[d]

                if ti.static(need_matrix):
                    for k in range(self.mpm.offset[jp]):
                        base_knode = self.mpm.LnID[jp, k]
                        base_koffset = self.mpm.node2dof[base_knode] - 1
                        dpoint_dx2 = self.mpm.shape[jp, k] * ti.Matrix.identity(ti.f64, config.DIM)
                        d2Psi = dpoint_dx1 @ relative_hessian @ dpoint_dx2
                        slot = raw_base + c * contact_pair_capacity + pair_capacity + j * support_capacity + k
                        self._write_barrier_block_fixed_slot(
                            slot,
                            stored_end,
                            base_joffset,
                            base_koffset,
                            d2Psi,
                        )

            # ----------------------
            # i-j block
            # ----------------------
            if ti.static(need_matrix):
                for j in range(self.mpm.offset[ip]):
                    base_jnode = self.mpm.LnID[ip, j]
                    base_joffset = self.mpm.node2dof[base_jnode] - 1
                    dpoint_dx1 = self.mpm.shape[ip, j] * ti.Matrix.identity(ti.f64, config.DIM)

                    for k in range(self.mpm.offset[jp]):
                        base_knode = self.mpm.LnID[jp, k]
                        base_koffset = self.mpm.node2dof[base_knode] - 1
                        dpoint_dx2 = self.mpm.shape[jp, k] * ti.Matrix.identity(ti.f64, config.DIM)
                        d2Psi = -(dpoint_dx1 @ relative_hessian @ dpoint_dx2)
                        slot = raw_base + c * contact_pair_capacity + 2 * pair_capacity + j * support_capacity + k
                        self._write_barrier_block_fixed_slot(
                            slot,
                            stored_end,
                            base_joffset,
                            base_koffset,
                            d2Psi,
                        )

                # ----------------------
                # j-i block
                # ----------------------
                for j in range(self.mpm.offset[jp]):
                    base_jnode = self.mpm.LnID[jp, j]
                    base_joffset = self.mpm.node2dof[base_jnode] - 1
                    dpoint_dx1 = self.mpm.shape[jp, j] * ti.Matrix.identity(ti.f64, config.DIM)

                    for k in range(self.mpm.offset[ip]):
                        base_knode = self.mpm.LnID[ip, k]
                        base_koffset = self.mpm.node2dof[base_knode] - 1
                        dpoint_dx2 = self.mpm.shape[ip, k] * ti.Matrix.identity(ti.f64, config.DIM)
                        d2Psi = -(dpoint_dx1 @ relative_hessian @ dpoint_dx2)
                        slot = raw_base + c * contact_pair_capacity + 3 * pair_capacity + j * support_capacity + k
                        self._write_barrier_block_fixed_slot(
                            slot,
                            stored_end,
                            base_joffset,
                            base_koffset,
                            d2Psi,
                        )

    @ti.func
    def _write_friction_block_fixed_slot(self, slot, stored_end, block_i, block_j, block):
        if block_i >= 0 and block_j >= 0:
            if block_i == block_j:
                for row in ti.static(range(config.DIM)):
                    for column in ti.static(range(config.DIM)):
                        component = row * config.DIM + column
                        ti.atomic_add(
                            self.friction_hash_matrix.diag[block_i][component],
                            block[row, column],
                        )
            elif slot < stored_end:
                self.friction_hash_matrix.non_diag.blockI[slot] = block_i
                self.friction_hash_matrix.non_diag.blockJ[slot] = block_j
                for row in ti.static(range(config.DIM)):
                    for column in ti.static(range(config.DIM)):
                        component = row * config.DIM + column
                        self.friction_hash_matrix.non_diag.blockH[slot][component] = block[row, column]

    @ti.kernel
    def assemble_ground_friction_matrix(self):
        support_capacity = ti.static(self.mpm.shape_func.max_node_per_particle)
        pair_capacity = ti.static(support_capacity * support_capacity)
        raw_base = self.friction_hash_matrix.raw_non_diag_count[0]
        raw_end = raw_base + self.gfrictionNum[0] * pair_capacity
        stored_end = ti.min(
            raw_end,
            self.friction_hash_matrix.non_diag.blockI.shape[0],
        )
        self.friction_hash_matrix.raw_non_diag_count[0] = stored_end
        if raw_end > self.friction_hash_matrix.non_diag.blockI.shape[0]:
            self.friction_hash_matrix.overflow[0] = 1
        for slot in range(raw_base, stored_end):
            self.friction_hash_matrix.non_diag.blockI[slot] = -1
            self.friction_hash_matrix.non_diag.blockJ[slot] = -1
            for component in ti.static(range(config.DIM * config.DIM)):
                self.friction_hash_matrix.non_diag.blockH[slot][component] = 0.0

        for c in range(self.gfrictionNum[0]):
            s = self.gfriction[c].surfaceID
            i = self.mpm.surface_id[s]
            wid = self.gfriction[c].wallID
            mu_lambda = self.gfriction[c].mu_lambda
            if mu_lambda > 0.0:
                T = self.ground.tangent_operator(wid)
                rel_disp = self.mpm.p_temp[s] - self.hat_x[s] - self.ground.vel[wid] * self.mpm.dt
                vbar = T.transpose() @ rel_disp / self.mpm.dt
                vbarnorm = vbar.norm()
                friction_gradient = self.friction.grad_term(vbarnorm)
                friction_hessian = self.friction.hess_term(vbarnorm)
                ddispbar_dpoint = mu_lambda * friction_gradient * T @ vbar
                inner_term = friction_gradient * ti.Matrix.identity(ti.f64, config.DIM)
                if vbarnorm != 0:
                    inner_term += friction_hessian / vbarnorm * vbar.outer_product(vbar)
                d2dispbar_d2point = mu_lambda * T @ psd_project_nd(inner_term) @ T.transpose() / self.mpm.dt

                for j in range(self.mpm.offset[i]):
                    base_jnode = self.mpm.LnID[i, j]
                    base_joffset = self.mpm.node2dof[base_jnode] - 1
                    dofs = config.DIM * base_joffset
                    dpoint_dx1 = self.mpm.shape[i, j] * ti.Matrix.identity(ti.f64, config.DIM)
                    dPsi_dx = ddispbar_dpoint @ dpoint_dx1
                    for d in ti.static(range(config.DIM)):
                        self.friction_grad[dofs + d] -= dPsi_dx[d]
                    for k in range(self.mpm.offset[i]):
                        base_knode = self.mpm.LnID[i, k]
                        base_koffset = self.mpm.node2dof[base_knode] - 1
                        dpoint_dx2 = self.mpm.shape[i, k] * ti.Matrix.identity(ti.f64, config.DIM)
                        d2Psi_dx1dx2 = dpoint_dx1 @ d2dispbar_d2point @ dpoint_dx2.transpose()
                        slot = raw_base + c * pair_capacity + j * support_capacity + k
                        self._write_friction_block_fixed_slot(
                            slot,
                            stored_end,
                            base_joffset,
                            base_koffset,
                            d2Psi_dx1dx2,
                        )

    @ti.func
    def _surface_endpoint_velocity(self, particle_id, grid_disp: ti.template()):
        velocity = ti.Vector.zero(ti.f64, config.DIM)
        for local_node in range(self.mpm.offset[particle_id]):
            grid_id = self.mpm.LnID[particle_id, local_node]
            dofs = config.DIM * (self.mpm.node2dof[grid_id] - 1)
            displacement = ti.Vector([grid_disp[dofs + d] for d in ti.static(range(config.DIM))])
            nodal_velocity = (
                self.endpoint_velocity_displacement_scale * displacement
                + self.endpoint_velocity_previous_scale * self.mpm.grid[grid_id].v
                + self.endpoint_velocity_acceleration_scale * self.mpm.grid[grid_id].a
            )
            velocity += self.mpm.shape[particle_id, local_node] * nodal_velocity
        return velocity

    def assemble_ground_fully_implicit_friction_matrix(self, grid_disp):
        self._assemble_ground_fully_implicit_friction_system(grid_disp, True)

    def _assemble_ground_fully_implicit_friction_system(self, grid_disp, need_matrix):
        self._evaluate_ground_fully_implicit_friction_law(grid_disp)
        self._scatter_ground_fully_implicit_friction_force()
        if need_matrix:
            self._scatter_ground_fully_implicit_friction_matrix(grid_disp)

    @ti.kernel
    def _evaluate_ground_fully_implicit_friction_law(self, grid_disp: ti.template()):
        """Assemble the exact current point-plane paper friction system."""
        for contact in range(self.gfrictionNum[0]):
            self.fully_implicit_point_force[contact] = ti.Vector.zero(ti.f64, config.DIM)
            surface = self.gfriction[contact].surfaceID
            particle = self.mpm.surface_id[surface]
            wall = self.gfriction[contact].wallID
            distance = self.ground.distance(wall, self.mpm.p_temp[surface])
            measure = point_contact_measure(self.mpm.surface_measure[surface])
            normal_force = -measure * self.barrier.grad_term(distance)
            if normal_force > 0.0:
                normal = self.ground.norm[wall]
                point_velocity = self._surface_endpoint_velocity(particle, grid_disp)
                relative_velocity = point_velocity - self.ground.vel[wall]
                point_force = ti.Vector.zero(ti.f64, config.DIM)
                point_force = ipc_fully_implicit_point_plane_stribeck_force(
                    relative_velocity,
                    normal,
                    normal_force,
                    self.friction.mu_dynamic[0],
                    self.friction.mu_static[0],
                    self.friction.mu_viscous[0],
                    self.friction.stribeck_velocity[0],
                    self.friction.epsv[0],
                    ti.static(self.friction.profile_id),
                )

                self.fully_implicit_point_force[contact] = point_force

    @ti.kernel
    def _scatter_ground_fully_implicit_friction_force(self):
        for contact in range(self.gfrictionNum[0]):
            surface = self.gfriction[contact].surfaceID
            particle = self.mpm.surface_id[surface]
            point_force = self.fully_implicit_point_force[contact]
            for local_j in range(self.mpm.offset[particle]):
                node_j = self.mpm.LnID[particle, local_j]
                block_j = self.mpm.node2dof[node_j] - 1
                dofs_j = config.DIM * block_j
                shape_j = self.mpm.shape[particle, local_j]
                for component in ti.static(range(config.DIM)):
                    self.friction_grad[dofs_j + component] -= shape_j * point_force[component]

    @ti.kernel
    def _scatter_ground_fully_implicit_friction_matrix(self, grid_disp: ti.template()):
        # One raw slot is owned by each (contact, local_j, local_k) tuple.
        # This removes the append atomic entirely and gives the pattern cache
        # a stable raw-to-reduced map across Newton/Armijo iterations.
        support_capacity = ti.static(self.mpm.shape_func.max_node_per_particle)
        pair_capacity = ti.static(support_capacity * support_capacity)
        raw_count = self.gfrictionNum[0] * pair_capacity
        stored_count = ti.min(
            raw_count,
            self.friction_hash_matrix.non_diag.blockI.shape[0],
        )
        self.friction_hash_matrix.raw_non_diag_count[0] = stored_count
        if raw_count > self.friction_hash_matrix.non_diag.blockI.shape[0]:
            self.friction_hash_matrix.overflow[0] = 1
        for slot in range(stored_count):
            self.friction_hash_matrix.non_diag.blockI[slot] = -1
            self.friction_hash_matrix.non_diag.blockJ[slot] = -1
            for component in ti.static(range(config.DIM * config.DIM)):
                self.friction_hash_matrix.non_diag.blockH[slot][component] = 0.0

        for contact in range(self.gfrictionNum[0]):
            surface = self.gfriction[contact].surfaceID
            particle = self.mpm.surface_id[surface]
            wall = self.gfriction[contact].wallID
            distance = self.ground.distance(wall, self.mpm.p_temp[surface])
            measure = point_contact_measure(self.mpm.surface_measure[surface])
            normal_force = -measure * self.barrier.grad_term(distance)
            if normal_force > 0.0:
                normal = self.ground.norm[wall]
                relative_velocity = self._surface_endpoint_velocity(particle, grid_disp) - self.ground.vel[wall]
                normal_force_gradient = -measure * self.barrier.hess_term(distance) * normal
                _, point_jacobian = ipc_fully_implicit_point_plane_stribeck_friction(
                    relative_velocity,
                    normal,
                    normal_force,
                    normal_force_gradient,
                    self.friction.mu_dynamic[0],
                    self.friction.mu_static[0],
                    self.friction.mu_viscous[0],
                    self.friction.stribeck_velocity[0],
                    self.friction.epsv[0],
                    ti.static(self.friction.profile_id),
                    self.endpoint_velocity_displacement_scale,
                )
                for local_j in range(self.mpm.offset[particle]):
                    node_j = self.mpm.LnID[particle, local_j]
                    block_j = self.mpm.node2dof[node_j] - 1
                    shape_j = self.mpm.shape[particle, local_j]
                    for local_k in range(self.mpm.offset[particle]):
                        node_k = self.mpm.LnID[particle, local_k]
                        block_k = self.mpm.node2dof[node_k] - 1
                        shape_k = self.mpm.shape[particle, local_k]
                        slot = contact * pair_capacity + local_j * support_capacity + local_k
                        if block_j >= 0 and block_k >= 0:
                            if block_j == block_k:
                                for row, column in ti.static(ti.ndrange(config.DIM, config.DIM)):
                                    component = row * config.DIM + column
                                    ti.atomic_add(
                                        self.friction_hash_matrix.diag[block_j][component],
                                        shape_j * shape_k * point_jacobian[row, column],
                                    )
                            elif slot < stored_count:
                                self.friction_hash_matrix.non_diag.blockI[slot] = block_j
                                self.friction_hash_matrix.non_diag.blockJ[slot] = block_k
                                for row, column in ti.static(ti.ndrange(config.DIM, config.DIM)):
                                    component = row * config.DIM + column
                                    self.friction_hash_matrix.non_diag.blockH[slot][component] = (
                                        shape_j * shape_k * point_jacobian[row, column]
                                    )

    @ti.kernel
    def assemble_particle_friction_matrix(self):
        support_capacity = ti.static(self.mpm.shape_func.max_node_per_particle)
        pair_capacity = ti.static(support_capacity * support_capacity)
        contact_pair_capacity = ti.static(4 * pair_capacity)
        raw_base = self.friction_hash_matrix.raw_non_diag_count[0]
        raw_end = raw_base + self.pfrictionNum[0] * contact_pair_capacity
        stored_end = ti.min(
            raw_end,
            self.friction_hash_matrix.non_diag.blockI.shape[0],
        )
        self.friction_hash_matrix.raw_non_diag_count[0] = stored_end
        if raw_end > self.friction_hash_matrix.non_diag.blockI.shape[0]:
            self.friction_hash_matrix.overflow[0] = 1
        for slot in range(raw_base, stored_end):
            self.friction_hash_matrix.non_diag.blockI[slot] = -1
            self.friction_hash_matrix.non_diag.blockJ[slot] = -1
            for component in ti.static(range(config.DIM * config.DIM)):
                self.friction_hash_matrix.non_diag.blockH[slot][component] = 0.0

        for c in range(self.pfrictionNum[0]):
            sp = self.pfriction[c].masterID
            tp = self.pfriction[c].slaveID

            ip = self.mpm.surface_id[sp]
            jp = self.mpm.surface_id[tp]

            ipos = self.mpm.p_temp[sp]
            jpos = self.mpm.p_temp[tp]

            mu_lambda = self.pfriction[c].mu_lambda
            if mu_lambda > 0.0:
                norm = self.pfriction[c].normal
                T = ti.Matrix.identity(ti.f64, config.DIM) - norm.outer_product(norm)
                rel_disp = (self.mpm.p_temp[sp] - self.hat_x[sp]) - (self.mpm.p_temp[tp] - self.hat_x[tp])
                vbar = T.transpose() @ rel_disp / self.mpm.dt
                vbarnorm = vbar.norm()
                friction_gradient = self.friction.grad_term(vbarnorm)
                friction_hessian = self.friction.hess_term(vbarnorm)
                ddispbar_dpoint = mu_lambda * friction_gradient * T @ vbar
                inner_term = friction_gradient * ti.Matrix.identity(ti.f64, config.DIM)
                if vbarnorm != 0:
                    inner_term += friction_hessian / vbarnorm * vbar.outer_product(vbar)
                d2dispbar_d2point = mu_lambda * T @ psd_project_nd(inner_term) @ T.transpose() / self.mpm.dt

                # ----------------------
                # i-i block
                # ----------------------
                for j in range(self.mpm.offset[ip]):
                    base_jnode = self.mpm.LnID[ip, j]
                    base_joffset = self.mpm.node2dof[base_jnode] - 1
                    dofs = config.DIM * base_joffset
                    dpoint_dx1 = self.mpm.shape[ip, j] * ti.Matrix.identity(ti.f64, config.DIM)
                    dPsi_dx = ddispbar_dpoint @ dpoint_dx1
                    for d in ti.static(range(config.DIM)):
                        self.friction_grad[dofs + d] -= dPsi_dx[d]
                    for k in range(self.mpm.offset[ip]):
                        base_knode = self.mpm.LnID[ip, k]
                        base_koffset = self.mpm.node2dof[base_knode] - 1
                        dpoint_dx2 = self.mpm.shape[ip, k] * ti.Matrix.identity(ti.f64, config.DIM)
                        d2Psi = (
                            dpoint_dx1 @ d2dispbar_d2point @ dpoint_dx2.transpose()
                        )  # + ddispbar_dpoint @ (ddist_dx1 @ d2_ipip @ ddist_dx2)
                        slot = raw_base + c * contact_pair_capacity + j * support_capacity + k
                        self._write_friction_block_fixed_slot(
                            slot,
                            stored_end,
                            base_joffset,
                            base_koffset,
                            d2Psi,
                        )

                # ----------------------
                # j-j block
                # ----------------------
                for j in range(self.mpm.offset[jp]):
                    base_jnode = self.mpm.LnID[jp, j]
                    base_joffset = self.mpm.node2dof[base_jnode] - 1
                    dofs = config.DIM * base_joffset
                    dpoint_dx1 = self.mpm.shape[jp, j] * ti.Matrix.identity(ti.f64, config.DIM)
                    # ddist_dx1 = ddist_jp @ dpoint_dx1
                    dPsi_dx1 = ddispbar_dpoint @ dpoint_dx1
                    for d in ti.static(range(config.DIM)):
                        self.friction_grad[config.DIM * base_joffset + d] += dPsi_dx1[d]

                    for k in range(self.mpm.offset[jp]):
                        base_knode = self.mpm.LnID[jp, k]
                        base_koffset = self.mpm.node2dof[base_knode] - 1
                        dpoint_dx2 = self.mpm.shape[jp, k] * ti.Matrix.identity(ti.f64, config.DIM)
                        d2Psi = dpoint_dx1 @ d2dispbar_d2point @ dpoint_dx2.transpose()
                        slot = raw_base + c * contact_pair_capacity + pair_capacity + j * support_capacity + k
                        self._write_friction_block_fixed_slot(
                            slot,
                            stored_end,
                            base_joffset,
                            base_koffset,
                            d2Psi,
                        )

                # ----------------------
                # i-j block
                # ----------------------
                for j in range(self.mpm.offset[ip]):
                    base_jnode = self.mpm.LnID[ip, j]
                    base_joffset = self.mpm.node2dof[base_jnode] - 1
                    dpoint_dx1 = self.mpm.shape[ip, j] * ti.Matrix.identity(ti.f64, config.DIM)
                    # ddist_dx1 = ddist_ip @ dpoint_dx1

                    for k in range(self.mpm.offset[jp]):
                        base_knode = self.mpm.LnID[jp, k]
                        base_koffset = self.mpm.node2dof[base_knode] - 1
                        dpoint_dx2 = self.mpm.shape[jp, k] * ti.Matrix.identity(ti.f64, config.DIM)
                        d2Psi = -(dpoint_dx1 @ d2dispbar_d2point @ dpoint_dx2.transpose())
                        slot = raw_base + c * contact_pair_capacity + 2 * pair_capacity + j * support_capacity + k
                        self._write_friction_block_fixed_slot(
                            slot,
                            stored_end,
                            base_joffset,
                            base_koffset,
                            d2Psi,
                        )

                # ----------------------
                # j-i block
                # ----------------------
                for j in range(self.mpm.offset[jp]):
                    base_jnode = self.mpm.LnID[jp, j]
                    base_joffset = self.mpm.node2dof[base_jnode] - 1
                    dpoint_dx1 = self.mpm.shape[jp, j] * ti.Matrix.identity(ti.f64, config.DIM)
                    # ddist_dx1 = ddist_jp @ dpoint_dx1

                    for k in range(self.mpm.offset[ip]):
                        base_knode = self.mpm.LnID[ip, k]
                        base_koffset = self.mpm.node2dof[base_knode] - 1
                        dpoint_dx2 = self.mpm.shape[ip, k] * ti.Matrix.identity(ti.f64, config.DIM)
                        d2Psi = -(dpoint_dx1 @ d2dispbar_d2point @ dpoint_dx2.transpose())
                        slot = raw_base + c * contact_pair_capacity + 3 * pair_capacity + j * support_capacity + k
                        self._write_friction_block_fixed_slot(
                            slot,
                            stored_end,
                            base_joffset,
                            base_koffset,
                            d2Psi,
                        )

    @ti.kernel
    def compute_normal_contact_force(self):
        for c in range(self.gbarrierNum[0]):
            s = self.gbarrier[c].surfaceID
            i = self.mpm.surface_id[s]
            wid = self.gbarrier[c].wallID
            dist = self.gbarrier[c].distance
            slot = self.gbarrier[c].semi_slot
            value = dist
            if ti.static(self.is_semi):
                value -= self.barrier.activation_distance_term()
            _, first, _ = self._normal_terms(value, slot)
            barrier_gradient = point_contact_measure(self.mpm.surface_measure[s]) * first
            bodyID = self.mpm.particle[i].bodyID
            self.normal_force[bodyID] -= barrier_gradient * self.ground.norm[wid]

    @ti.kernel
    def compute_tangential_contact_force(self):
        for c in range(self.gfrictionNum[0]):
            s = self.gfriction[c].surfaceID
            i = self.mpm.surface_id[s]
            wid = self.gfriction[c].wallID
            mu_lambda = self.gfriction[c].mu_lambda
            T = self.ground.tangent_operator(wid)
            rel_disp = self.mpm.p_temp[s] - self.hat_x[s] - self.ground.vel[wid] * self.mpm.dt
            vbar = T.transpose() @ rel_disp / self.mpm.dt
            vbarnorm = vbar.norm()
            friction_gradient = self.friction.grad_term(vbarnorm)
            if vbarnorm != 0:
                bodyID = self.mpm.particle[i].bodyID
                ddispbar_dpoint = mu_lambda * friction_gradient * T @ vbar
                self.tangential_force[bodyID] -= ddispbar_dpoint

    @ti.kernel
    def ground_ccd(self, slackness: ti.f64, grid_disp: ti.template()) -> ti.f64:
        toc = 1.0
        clearance = 0.0
        if ti.static(not self.is_semi):
            clearance = self.barrier.dmin[0]
        for c in range(self.gbarrierNum[0]):
            s = self.gbarrier[c].surfaceID
            i = self.mpm.surface_id[s]
            wid = self.gbarrier[c].wallID
            dist = self.gbarrier[c].distance
            disp = ti.Vector.zero(ti.f64, config.DIM)
            for j in range(self.mpm.offset[i]):
                grid_id = self.mpm.LnID[i, j]
                dofs = config.DIM * (self.mpm.node2dof[grid_id] - 1)
                disp += self.mpm.shape[i, j] * ti.Vector([grid_disp[dofs + d] for d in ti.static(range(config.DIM))])
            gdisp = self.ground.vel[wid] * self.mpm.dt
            rel_disp = disp - gdisp
            cur_disp = self.ground.norm[wid].dot(rel_disp)
            ptoc = linear_gap_accd(
                dist,
                cur_disp,
                slackness,
                clearance,
            )
            ti.atomic_min(toc, ptoc)
        return toc

    @ti.kernel
    def particle_ccd(self, slackness: float, grid_disp: ti.template()) -> ti.f64:
        toc = 1.0
        clearance = 0.0
        if ti.static(not self.is_semi):
            clearance = self.barrier.dmin[0]
        if ti.static(not self.enable_particle_contact):
            return toc
        for contact_id in range(self.pbarrierNum[0]):
            s = self.pbarrier[contact_id].masterID
            t = self.pbarrier[contact_id].slaveID
            i = self.mpm.surface_id[s]
            j = self.mpm.surface_id[t]
            ipos = self.mpm.p_temp[s]
            jpos = self.mpm.p_temp[t]
            idisp = ti.Vector.zero(ti.f64, config.DIM)
            for k in range(self.mpm.offset[i]):
                grid_id = self.mpm.LnID[i, k]
                dofs = config.DIM * (self.mpm.node2dof[grid_id] - 1)
                idisp += self.mpm.shape[i, k] * ti.Vector([grid_disp[dofs + d] for d in ti.static(range(config.DIM))])
            jdisp = ti.Vector.zero(ti.f64, config.DIM)
            for k in range(self.mpm.offset[j]):
                grid_id = self.mpm.LnID[j, k]
                dofs = config.DIM * (self.mpm.node2dof[grid_id] - 1)
                jdisp += self.mpm.shape[j, k] * ti.Vector([grid_disp[dofs + d] for d in ti.static(range(config.DIM))])
            solution = point_point_accd(
                ipos,
                jpos,
                idisp,
                jdisp,
                1.0 - slackness,
                clearance,
                100,
            )
            ti.atomic_min(toc, solution)
        return toc

    @ti.kernel
    def ground_full_ccd(self, slackness: ti.f64, grid_disp: ti.template()) -> ti.f64:
        toc = 1.0
        clearance = 0.0
        if ti.static(not self.is_semi):
            clearance = self.barrier.dmin[0]
        for s in range(self.mpm.total_surface_num):
            i = self.mpm.surface_id[s]
            current_pos = self.mpm.p_temp[s]
            disp = ti.Vector.zero(ti.f64, config.DIM)
            for j in range(self.mpm.offset[i]):
                grid_id = self.mpm.LnID[i, j]
                dofs = config.DIM * (self.mpm.node2dof[grid_id] - 1)
                disp += self.mpm.shape[i, j] * ti.Vector([grid_disp[dofs + d] for d in ti.static(range(config.DIM))])
            ptoc = 1.0
            for k in range(self.ground.num):
                dist = self.ground.distance(k, current_pos)
                cur_disp = self.ground.norm[k].dot(disp - self.ground.vel[k] * self.mpm.dt)
                ptoc = ti.min(
                    ptoc,
                    linear_gap_accd(
                        dist,
                        cur_disp,
                        slackness,
                        clearance,
                    ),
                )
            ti.atomic_min(toc, ptoc)
        return toc

    @ti.kernel
    def _particle_swept_ccd(self, slackness: float) -> ti.f64:
        toc = 1.0
        clearance = 0.0
        if ti.static(not self.is_semi):
            clearance = self.barrier.dmin[0]
        if ti.static(not self.enable_particle_contact):
            return toc
        for pair_id, local_s, local_t in ti.ndrange(
            self.body_pair_num[0], self.max_body_surface_count, self.max_body_surface_count
        ):
            body_i = self.body_pair[pair_id].master
            body_j = self.body_pair[pair_id].slave
            if local_s < self.body_surface_count[body_i] and local_t < self.body_surface_count[body_j]:
                s = self.body_surface_start[body_i] + local_s
                t = self.body_surface_start[body_j] + local_t
                solution = point_point_accd(
                    self.mpm.p_temp[s],
                    self.mpm.p_temp[t],
                    self.surface_sweep_disp[s],
                    self.surface_sweep_disp[t],
                    1.0 - slackness,
                    clearance,
                    100,
                )
                ti.atomic_min(toc, solution)
        return toc

    def particle_full_ccd(self, slackness, grid_disp):
        """Conservative point--point ACCD over swept broad-phase body pairs."""
        if not self.enable_particle_contact:
            return 1.0
        self.update_swept_body_pair_table(grid_disp)
        return self._particle_swept_ccd(float(slackness))

    @ti.kernel
    def initial_surface_temp(self):
        for s in range(self.mpm.total_surface_num):
            i = self.mpm.surface_id[s]
            self.mpm.p_temp[s] = self.mpm.particle[i].x
            self.mpm.pv_temp[s] = self.mpm.particle[i].v

    @ti.kernel
    def get_barrier_energy(self):
        for c in range(self.gbarrierNum[0]):
            s = self.gbarrier[c].surfaceID
            dist = self.gbarrier[c].distance
            value = dist
            if ti.static(self.is_semi):
                value -= self.barrier.activation_distance_term()
            energy, _, _ = self._normal_terms(value, self.gbarrier[c].semi_slot)
            self.mpm.energy[None] += point_contact_measure(self.mpm.surface_measure[s]) * energy
        for c in range(self.pbarrierNum[0]):
            sp = self.pbarrier[c].masterID
            tp = self.pbarrier[c].slaveID
            dist = self.pbarrier[c].distance
            value = dist
            if ti.static(self.is_semi):
                slot = self.pbarrier[c].semi_slot
                value = (
                    self.semi_normal[slot].dot(self.mpm.p_temp[sp] - self.mpm.p_temp[tp])
                    - self.barrier.activation_distance_term()
                )
            energy, _, _ = self._normal_terms(value, self.pbarrier[c].semi_slot)
            self.mpm.energy[None] += (
                symmetric_contact_measure(self.mpm.surface_measure[sp], self.mpm.surface_measure[tp]) * energy
            )

    @ti.kernel
    def get_friction_energy(self):
        for c in range(self.gfrictionNum[0]):
            s = self.gfriction[c].surfaceID
            wid = self.gfriction[c].wallID
            mu_lambda = self.gfriction[c].mu_lambda
            if mu_lambda > 0.0:
                T = self.ground.tangent_operator(wid)
                rel_disp = self.mpm.p_temp[s] - self.hat_x[s] - self.ground.vel[wid] * self.mpm.dt
                vbar = T.transpose() @ rel_disp / self.mpm.dt
                vbarnorm = vbar.norm()
                self.mpm.energy[None] += mu_lambda * self.friction.energy_term(vbarnorm, self.mpm.dt)
        for c in range(self.pfrictionNum[0]):
            sp = self.pfriction[c].masterID
            tp = self.pfriction[c].slaveID
            mu_lambda = self.pfriction[c].mu_lambda
            if mu_lambda > 0.0:
                norm = self.pfriction[c].normal
                T = ti.Matrix.identity(ti.f64, config.DIM) - norm.outer_product(norm)
                rel_disp = (self.mpm.p_temp[sp] - self.hat_x[sp]) - (self.mpm.p_temp[tp] - self.hat_x[tp])
                vbar = T.transpose() @ rel_disp / self.mpm.dt
                vbarnorm = vbar.norm()
                self.mpm.energy[None] += mu_lambda * self.friction.energy_term(vbarnorm, self.mpm.dt)

    def total_energy(self, grid_disp):
        self.mpm.energy[None] = 0.0
        self.mpm.get_material_energy(grid_disp)
        self.mpm.get_inertia_energy(self.mpm.damping, self.mpm.integration, self.mpm.gravity, grid_disp)
        self.update_particle_pos(grid_disp)
        if not self.rebuild_barrier_contacts():
            return float("inf")
        self.get_barrier_energy()
        # Friction is evaluated with the same lagged state used in assembly:
        # hat_x is frozen at the beginning of the time step, while the contact
        # normal and mu_lambda are frozen for the current outer fixed-point solve.
        if self.friction_mode == "lagged" and self.curr_friction_contact_num > 0:
            self.get_friction_energy()
        if self.mpm.neumann.num > 0:
            self.mpm.get_neumann_energy(grid_disp)
        return self.mpm.energy[None]

    def record_normal_force(self):
        self.normal_force.fill(0)
        self.initial_surface_temp()
        if not self.rebuild_barrier_contacts():
            raise IPCInfeasibleContactState("cannot evaluate contact force at or below dmin")
        self.compute_normal_contact_force()
        return self.normal_force.to_numpy()

    def record_tangential_force(self):
        self.tangential_force.fill(0)
        self.initial_surface_temp()
        if self.activate_fric:
            if not self.friction_reference_initialized:
                self.initialize_hat_x()
                self.friction_reference_initialized = True
            self.ground_friction_initialize()
            self.compute_tangential_contact_force()
        return self.tangential_force.to_numpy()

    def ccd(self, active_dof):
        slackness_a = 0.9
        slackness_b = 0.8
        slackness_m = 0.8
        self.mpm.update_grid_disp(active_dof, 1.0)
        alpha_ground = self.ground_ccd(slackness_a, self.mpm.incre_resolution)
        alpha_particle = self.particle_ccd(slackness_b, self.mpm.incre_resolution)
        alpha_material = self.mpm.material_ccd(slackness_m)
        alpha_CCD = min(alpha_ground, alpha_particle, alpha_material)

        max_disp = self.mpm.compute_particle_disp(self.mpm.incre_resolution)
        alpha_CFL = 1.0
        if np.isfinite(max_disp) and max_disp > 1.0e-30:
            alpha_CFL = min(1.0, 0.5 * float(self.barrier.dhat[0]) / max_disp)

        alpha = min(alpha_CCD, alpha_CFL)
        if alpha_CCD > 2 * alpha_CFL:
            alpha_full_ground = self.ground_full_ccd(slackness_a, self.mpm.incre_resolution)
            alpha_full_particle = self.particle_full_ccd(slackness_b, self.mpm.incre_resolution)
            alpha_full = min(alpha_full_ground, alpha_full_particle)
            alpha = min(alpha_CCD, alpha_full * 1.0)
            alpha = max(alpha, alpha_CFL)
        return alpha

    def line_search(self, active_dof, verbose=False):
        self.line_search_stop_iter = False
        self.line_search_stop_reason = ""
        self.line_search_failed = False
        self.line_search_used_fallback = False
        g0 = -self.mpm.calc_g0(active_dof)
        if g0 > 0:
            if not self.line_search_descent_fallback:
                raise AssertionError(f"Warning: Not a descent direction! g0: {g0}")
            copy_field(active_dof, self.mpm.incre_resolution, self.mpm.rhs)
            g0 = -self.mpm.calc_g0(active_dof)
            self.line_search_used_fallback = True
            if g0 > 0:
                self.line_search_failed = True
                return False
        previous_energy = self.total_energy(self.mpm.grid_disp)
        energy_scale = max(1.0, abs(previous_energy))
        self.line_search_last_g0 = float(g0)
        self.line_search_last_previous_energy = float(previous_energy)
        self.line_search_last_trial_energy = float(previous_energy)
        self.line_search_last_accepted_energy = float(previous_energy)
        self.line_search_last_alpha0 = 0.0
        self.line_search_last_alpha = 0.0
        self.line_search_last_backtracks = 0
        if abs(g0) <= self.line_search_work_tol * energy_scale and verbose:
            print(
                "line search: tiny directional work; continuing the "
                f"feasible Armijo step because Newton has not converged "
                f"(g0={g0}, energy={previous_energy})"
            )

        alpha = self.ccd(active_dof)
        initial_alpha = alpha
        self.mpm.update_grid_disp(active_dof, alpha)
        current_energy = None
        if self.mpm.do_line_search:
            energy_tol = self.line_search_energy_atol + self.line_search_energy_rtol * energy_scale
            backtracks = 0
            while alpha > self.line_search_min_alpha:
                current_energy = self.total_energy(self.mpm.grid_disp_temp)
                if backtracks == 0:
                    self.line_search_last_trial_energy = float(current_energy)
                armijo_bound = previous_energy + self.line_search_armijo * alpha * g0 + energy_tol
                if current_energy <= armijo_bound:
                    break
                if backtracks >= self.line_search_max_backtracks:
                    if (
                        alpha <= self.line_search_stagnation_alpha
                        and current_energy <= previous_energy + self.line_search_stagnation_energy_tol * energy_scale
                    ):
                        self.line_search_stop_iter = True
                        self.line_search_stop_reason = "stagnation"
                        self.line_search_last_alpha0 = float(initial_alpha)
                        self.line_search_last_alpha = float(alpha)
                        self.line_search_last_backtracks = int(backtracks)
                        self.line_search_last_accepted_energy = float(previous_energy)
                        self.accept_update()
                        return True
                    self.line_search_failed = True
                    self.line_search_last_alpha0 = float(initial_alpha)
                    self.line_search_last_alpha = float(alpha)
                    self.line_search_last_backtracks = int(backtracks)
                    self.line_search_last_accepted_energy = float(current_energy)
                    return False
                alpha *= 0.5
                backtracks += 1
                if verbose:
                    print(
                        f"current alpha: {alpha}, current energy: {current_energy}, previous energy: {previous_energy}"
                    )
                self.mpm.update_grid_disp(active_dof, alpha)
            self.line_search_last_backtracks = int(backtracks)
        self.line_search_last_alpha0 = float(initial_alpha)
        self.line_search_last_alpha = float(alpha)
        if alpha <= self.line_search_min_alpha:
            self.line_search_failed = True
            copy_field(
                active_dof,
                self.mpm.grid_disp_temp,
                self.mpm.grid_disp,
            )
            return False
        # The accepted Armijo trial has already rebuilt all barrier contacts
        # and evaluated the complete potential.  Reuse that scalar instead of
        # repeating the material/contact energy pipeline solely for logging.
        accepted_energy = current_energy
        if accepted_energy is None:
            accepted_energy = self.total_energy(self.mpm.grid_disp_temp)
        copy_field(active_dof, self.mpm.grid_disp, self.mpm.grid_disp_temp)
        self.accept_update()
        self.line_search_last_accepted_energy = float(accepted_energy)
        if self.mpm.do_line_search and alpha <= self.line_search_stagnation_alpha:
            self.line_search_stop_iter = True
            self.line_search_stop_reason = "stagnation"
        return True
