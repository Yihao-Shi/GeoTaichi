"""Monolithic implicit FEM--MPM solver with IPC contact and friction."""

import math
import time

import numpy as np
import taichi as ti

import src.mpm.config as mpm_config
from src.fem.engines.ImplicitFEM import NewtonConvergenceError
from src.fempm.contact.IPCAssembler import FEMPMIPCAssembler
from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.StepRetry import StepRetryPolicy, nonlinear_failure_kind
from src.utils.linalg import no_operation


def _normalize_assembly(value):
    key = str(value).replace("_", "").replace("-", "").lower()
    if key in ("hash", "hashtriplet", "triplet", "buildtriplet"):
        return "HashTriplet"
    if key in ("coo", "coordinate", "coordinatesparse"):
        return "COO"
    raise ValueError("FEMPM implicit assemble_type must be COO or HashTriplet")


def _normalize_solver(value):
    key = str(value).replace("_", "").replace("-", "").lower()
    if key in ("pcg", "cg", "taichipcg"):
        return "PCG"
    if key in ("scipy", "spsolve", "cpu", "direct"):
        return "Scipy"
    raise ValueError("FEMPM lagged IPC linear_solver must be PCG or Scipy")


@ti.data_oriented
class FEMPMImplicitEngine:
    """Projected-Newton solve over FEM nodes and active MPM grid nodes."""

    def __init__(
        self,
        simulation,
        fem,
        mpm,
        contact_model,
        faces,
        face_body,
        **kwargs,
    ):
        self.simulation = simulation
        self.fem_wrapper = fem
        self.mpm_wrapper = mpm
        self.fem = fem.engine
        self.mpm = mpm.enginer
        self.mpm_dimension = int(getattr(self.mpm, "dimension", 3))
        self.contact_model = contact_model
        self.dt = float(simulation.delta)
        self.time = float(simulation.current_time)
        self.step_count = int(simulation.current_step)
        self.compile_seconds = None
        self.max_iterations = int(kwargs.get("max_iterations", kwargs.get("max_iters", self.fem.max_iterations)))
        self.residual_tolerance = float(kwargs.get("residual_tolerance", kwargs.get("residual", 1.0e-8)))
        self.absolute_tolerance = float(kwargs.get("absolute_tolerance", 1.0e-10))
        self.correction_velocity_tolerance = float(
            kwargs.get(
                "correction_velocity_tolerance",
                kwargs.get("newton_velocity_tolerance", 1.0e-7),
            )
        )
        self.raise_on_nonconvergence = bool(kwargs.get("raise_on_nonconvergence", True))
        self.line_search_reduction = float(kwargs.get("line_search_reduction", 0.5))
        self.line_search_c1 = float(kwargs.get("line_search_sufficient_decrease", 1.0e-4))
        self.line_search_max_backtracks = int(kwargs.get("line_search_max_backtracks", 24))
        self.line_search_minimum_step = float(kwargs.get("line_search_minimum_step", 1.0e-10))
        self.line_search_energy_rtol = float(kwargs.get("line_search_energy_rtol", 1.0e-12))
        self.line_search_energy_atol = float(kwargs.get("line_search_energy_atol", 1.0e-14))
        self.step_retry = StepRetryPolicy(
            enabled=kwargs.get("enable_step_retry", False),
            maximum_retries=kwargs.get("step_retry_max_retries", 2),
            reduction=kwargs.get("step_retry_reduction", 0.5),
            minimum_timestep=kwargs.get("step_retry_minimum_timestep", 0.0),
        )
        self.assemble_type = _normalize_assembly(
            kwargs.get("assemble_type", kwargs.get("assembly", self.fem.assemble_type))
        )
        self.linear_solver = _normalize_solver(kwargs.get("linear_solver", self.fem.linear_solver))
        self.linear_solver_tolerance = float(kwargs.get("linear_solver_tolerance", 1.0e-10))
        self.linear_solver_relative_tolerance = float(kwargs.get("linear_solver_relative_tolerance", 0.0))
        self.linear_solver_max_iters = int(
            kwargs.get(
                "linear_solver_max_iters",
                max(500, 5 * (self.fem.degree_of_freedom + self.mpm.degree_of_freedom)),
            )
        )
        if self.max_iterations <= 0:
            raise ValueError("FEMPM implicit max_iterations must be positive")
        if not math.isfinite(self.residual_tolerance) or self.residual_tolerance < 0.0:
            raise ValueError("FEMPM residual_tolerance must be finite and non-negative")
        if not math.isfinite(self.absolute_tolerance) or self.absolute_tolerance < 0.0:
            raise ValueError("FEMPM absolute_tolerance must be finite and non-negative")
        if not math.isfinite(self.correction_velocity_tolerance) or self.correction_velocity_tolerance <= 0.0:
            raise ValueError("FEMPM correction_velocity_tolerance must be finite and positive")
        if not math.isfinite(self.linear_solver_tolerance) or self.linear_solver_tolerance < 0.0:
            raise ValueError("FEMPM linear_solver_tolerance must be finite and non-negative")
        if not math.isfinite(self.linear_solver_relative_tolerance) or self.linear_solver_relative_tolerance < 0.0:
            raise ValueError("FEMPM linear_solver_relative_tolerance must be finite and non-negative")
        if self.linear_solver_max_iters <= 0:
            raise ValueError("FEMPM linear_solver_max_iters must be positive")
        if not 0.0 < self.line_search_reduction < 1.0:
            raise ValueError("FEMPM line_search_reduction must be in (0, 1)")
        if not 0.0 < self.line_search_c1 < 1.0:
            raise ValueError("FEMPM Armijo coefficient must be in (0, 1)")
        if (
            not math.isfinite(self.line_search_energy_rtol)
            or self.line_search_energy_rtol < 0.0
            or not math.isfinite(self.line_search_energy_atol)
            or self.line_search_energy_atol < 0.0
        ):
            raise ValueError("FEMPM line-search energy tolerances must be finite and non-negative")
        self._mpm_initialized = False
        self._mpm_step_prepared = False
        self.prepare_mpm_transfer = (
            self._prepare_mass_mpm_transfer
            if hasattr(self.mpm, "mass_vel_acc_p2g")
            else self._prepare_legacy_mpm_transfer
        )
        self.transfer_mpm_traction = self.mpm.traction_p2g if self.mpm.compute_traction else no_operation
        self.update_mpm_mass_list = self._update_mpm_mass_list if mpm_config.DYNAMIC else no_operation
        self.mpm_reference_status = ti.field(dtype=ti.i32, shape=3)
        self._initialize_mpm_state()
        contact_all_mpm_particles = bool(kwargs.get("contact_all_mpm_particles", False))
        if contact_all_mpm_particles or not hasattr(self.mpm, "surface_id"):
            self.mpm.build_surface_node(all_particles=contact_all_mpm_particles)

        self.contact = FEMPMIPCAssembler(
            self.fem,
            self.mpm,
            faces,
            face_body,
            contact_model,
            simulation,
        )
        if self.step_retry.enabled and self.contact.is_semi:
            raise ValueError(
                "FEMPM step retry does not support SemiIPC because its multipliers advance inside Newton iterations"
            )
        self.fem_nodes = int(self.fem.mesh.number_of_nodes)
        self.mpm_node_capacity = int(self.mpm.degree_of_freedom // self.mpm_dimension)
        self.node_capacity = self.fem_nodes + self.mpm_node_capacity
        self.dof_capacity = 3 * self.node_capacity
        self.rhs = ti.field(dtype=ti.f64, shape=self.dof_capacity)
        self.physical_rhs = ti.field(dtype=ti.f64, shape=self.dof_capacity)
        self.correction = ti.field(dtype=ti.f64, shape=self.dof_capacity)
        self.energy_scratch_rhs = ti.field(dtype=ti.f64, shape=self.dof_capacity)
        self.fixed = ti.field(dtype=ti.i32, shape=self.dof_capacity)
        self.fixed_correction = ti.field(dtype=ti.f64, shape=self.dof_capacity)
        self.residual_squared = ti.field(dtype=ti.f64, shape=())
        self.directional_derivative = ti.field(dtype=ti.f64, shape=())
        self.correction_inf_norm = ti.field(dtype=ti.f64, shape=())
        self.history = []
        self.last_step_record = None
        self.record_history_step = True
        self.last_linear_solve = None
        self.last_failure = None
        self.last_friction_iterations = 0
        self.last_friction_residual = math.inf
        self.last_friction_converged = False
        self.step_schedule = simulation.step_schedule
        self.track_energy = bool(self.fem.track_energy)
        self.advance_constitutive_state = getattr(self.fem, "_advance_constitutive_state", no_operation)
        self.sample_energy_step = self.fem.state.reduce_kinetic_energy if self.track_energy else no_operation
        self.add_energy_record = self._add_energy_record if self.track_energy else no_operation

        particle_capacity = int(self.mpm.particle.shape[0])
        grid_capacity = int(self.mpm.grid.shape[0])
        self.snapshot_mpm_position = ti.Vector.field(self.mpm_dimension, ti.f64, shape=particle_capacity)
        self.snapshot_mpm_velocity = ti.Vector.field(self.mpm_dimension, ti.f64, shape=particle_capacity)
        self.snapshot_mpm_acceleration = ti.Vector.field(self.mpm_dimension, ti.f64, shape=particle_capacity)
        self.snapshot_mpm_deformation = ti.Matrix.field(
            self.mpm.material_dimension,
            self.mpm.material_dimension,
            ti.f64,
            shape=particle_capacity,
        )
        self.mpm_has_plastic_history = bool(self.mpm.is_finite_strain_plastic)
        self.mpm_uses_residual_merit = self.mpm_has_plastic_history and not bool(
            getattr(self.mpm.material, "has_incremental_potential", False)
        )
        self.snapshot_mpm_plastic_history = None
        if self.mpm_has_plastic_history:
            history_state_size = int(self.mpm.material.history_state_size)
            if history_state_size <= 0:
                raise RuntimeError(
                    "FEM-MPM finite-strain plasticity requires a " "material-owned device history-state interface"
                )
            self.snapshot_mpm_plastic_history = ti.Vector.field(history_state_size, ti.f64, shape=particle_capacity)
        self.snapshot_grid_mass = ti.field(ti.f64, shape=grid_capacity)
        self.snapshot_grid_velocity = ti.Vector.field(self.mpm_dimension, ti.f64, shape=grid_capacity)
        self.snapshot_grid_acceleration = ti.Vector.field(self.mpm_dimension, ti.f64, shape=grid_capacity)

        self.fem_hash = self._build_fem_source_matrix()
        self.mpm_embedded_hash = self._build_mpm_embedded_source_matrix()
        self.contact_hash = None
        self.monolithic_hash = None
        self.monolithic_coo = None
        self.coo_count = ti.field(dtype=ti.i32, shape=())
        self.coo_overflow = ti.field(dtype=ti.i32, shape=())
        self.coo_diagonal = ti.field(dtype=ti.f64, shape=self.dof_capacity)
        self._matrix_contact_capacity = 0
        initial_candidates = self.contact.prepare(self.fem.state.position, self.mpm.grid_disp)
        self._ensure_matrices(initial_candidates)

    @property
    def minimum_jacobian(self):
        return float(self.fem._minimum_jacobian_ratio_device(self.fem.state.position))

    @ti.kernel
    def _inspect_mpm_reference_state(self):
        for status in ti.static(range(3)):
            self.mpm_reference_status[status] = 0
        for particle in range(self.mpm.particleNum[0]):
            deformation_nonzero = 0
            for row, column in ti.static(
                ti.ndrange(
                    self.mpm.material_dimension,
                    self.mpm.material_dimension,
                )
            ):
                value = self.mpm.F0[particle][row, column]
                if not (-ti.math.inf < value and value < ti.math.inf):
                    self.mpm_reference_status[1] = 1
                if value != 0.0:
                    deformation_nonzero = 1
            if deformation_nonzero != 0:
                ti.atomic_max(self.mpm_reference_status[0], 1)
            if not (self.mpm.F0[particle].determinant() > 0.0):
                ti.atomic_max(self.mpm_reference_status[2], 1)

    def _initialize_mpm_state(self):
        if self._mpm_initialized:
            return
        self._inspect_mpm_reference_state()
        if int(self.mpm_reference_status[1]) != 0:
            raise RuntimeError("FEM-MPM MPM deformation gradients contain non-finite values")
        if int(self.mpm_reference_status[0]) == 0:
            self.mpm.init_F0()
        elif int(self.mpm_reference_status[2]) != 0:
            raise RuntimeError("FEM-MPM MPM deformation gradients must all have positive " "determinants")
        if hasattr(self.mpm, "mass_p2g"):
            self.mpm.compute_shapefn()
            self.mpm.mass_p2g()
            self.mpm.find_active_node()
            self.mpm.prefix_sum_executor.run(self.mpm.node2dof)
            self.mpm.active_dof = self.mpm.set_active_dof()
            if mpm_config.DYNAMIC:
                self.mpm.compute_mass_list(self.mpm.integration)
        self._mpm_initialized = True
        self._prepare_mpm_step()

    def _prepare_mpm_step(self):
        if self._mpm_step_prepared:
            return
        self.prepare_mpm_transfer()
        self.mpm.grid_disp.fill(0.0)
        self.mpm.grid_disp_temp.fill(0.0)
        self._mpm_step_prepared = True

    def _update_mpm_mass_list(self):
        self.mpm.compute_mass_list(self.mpm.integration)

    def _prepare_mass_mpm_transfer(self):
        self.mpm.mass_vec.fill(0.0)
        self.mpm.grid_reset()
        self.mpm.compute_shapefn()
        self.mpm.mass_vel_acc_p2g()
        self.mpm.find_active_node()
        self.mpm.prefix_sum_executor.run(self.mpm.node2dof)
        self.mpm.active_dof = self.mpm.set_active_dof()
        self.transfer_mpm_traction()
        self.mpm.compute_nodal_vel_acc()
        self.update_mpm_mass_list()

    def _prepare_legacy_mpm_transfer(self):
        self.mpm.grid_reset()
        self.mpm.vel_acc_p2g()
        self.transfer_mpm_traction()
        self.mpm.compute_nodal_vel_acc()

    def _fem_assembler(self):
        assembler = getattr(
            self.fem,
            "classical_assembler",
            getattr(self.fem, "cloth_assembler", None),
        )
        if assembler is None:
            raise RuntimeError("FEMPM IPC requires a Taichi elastic FEM assembler")
        return assembler

    def _fem_material_step(self):
        """Return the admissible FEM step for both volume and cloth FEM."""
        return self._fem_assembler().maximum_material_step_device(
            self.fem.state.position,
            self.fem.state.direction,
        )

    def _build_fem_source_matrix(self):
        assembler = self._fem_assembler()
        capacity = max(1, int(assembler.stiffness_block_pair_count))
        reduced_capacity = max(
            1,
            int(
                getattr(
                    assembler,
                    "stiffness_unique_block_pair_count",
                    capacity,
                )
            ),
        )
        self.fem_source_reduced_capacity = reduced_capacity
        return BuildTriplet(
            dim=3,
            max_pairs_num=capacity,
            max_nonzeros=reduced_capacity,
            max_active_nodes=int(self.fem.mesh.number_of_nodes),
            symmetric=False,
            solver="BiCGSTAB",
            matrix_symmetric=False,
            device_reduction=True,
            raw_only=True,
        )

    def _build_mpm_embedded_source_matrix(self):
        source = self.mpm.hash_matrix
        self.mpm_source_reduced_capacity = max(1, int(source.max_nonzeros))
        return BuildTriplet(
            dim=3,
            max_pairs_num=max(1, int(source.non_diag.max_pairs_num)),
            max_nonzeros=max(1, int(source.max_nonzeros)),
            max_active_nodes=max(1, self.mpm_node_capacity),
            symmetric=False,
            solver="BiCGSTAB",
            matrix_symmetric=False,
            device_reduction=True,
            raw_only=True,
        )

    @staticmethod
    def _coo_scalar_capacity(node_capacity, raw_block_pairs, dof_capacity):
        """Scalar COO bound for FEM, MPM and contact block sources."""
        return 18 * int(node_capacity) + 9 * int(raw_block_pairs) + int(dof_capacity)

    @ti.kernel
    def _embed_mpm_source(self, active_nodes: ti.i32):
        self.mpm_embedded_hash.raw_non_diag_count[0] = self.mpm.hash_matrix.raw_non_diag_count[0]
        for block in range(active_nodes):
            for entry in ti.static(range(9)):
                self.mpm_embedded_hash.diag[block][entry] = 0.0
            for row, column in ti.static(ti.ndrange(self.mpm_dimension, self.mpm_dimension)):
                self.mpm_embedded_hash.diag[block][3 * row + column] = self.mpm.hash_matrix.diag[block][
                    row * self.mpm_dimension + column
                ]
        for entry in range(self.mpm.hash_matrix.raw_non_diag_count[0]):
            self.mpm_embedded_hash.non_diag.blockI[entry] = self.mpm.hash_matrix.non_diag.blockI[entry]
            self.mpm_embedded_hash.non_diag.blockJ[entry] = self.mpm.hash_matrix.non_diag.blockJ[entry]
            for component in ti.static(range(9)):
                self.mpm_embedded_hash.non_diag.blockH[entry][component] = 0.0
            for row, column in ti.static(ti.ndrange(self.mpm_dimension, self.mpm_dimension)):
                self.mpm_embedded_hash.non_diag.blockH[entry][3 * row + column] = self.mpm.hash_matrix.non_diag.blockH[
                    entry
                ][row * self.mpm_dimension + column]

    def _ensure_matrices(self, candidate_count):
        required = self.contact.contact_block_capacity(candidate_count)
        if required <= self._matrix_contact_capacity:
            return
        contact_capacity = self.contact.configured_contact_block_capacity()
        if required > contact_capacity:
            raise RuntimeError(
                "FEMPM contact block capacity is too small: " f"need {required}, configured bound {contact_capacity}"
            )
        self._matrix_contact_capacity = contact_capacity
        self.contact_hash = BuildTriplet(
            dim=3,
            max_pairs_num=contact_capacity,
            max_nonzeros=contact_capacity,
            max_active_nodes=self.node_capacity,
            symmetric=False,
            solver="BiCGSTAB",
            matrix_symmetric=False,
            device_reduction=True,
            raw_only=True,
        )
        body_pairs = int(self.fem_hash.non_diag.max_pairs_num) + int(self.mpm_embedded_hash.non_diag.max_pairs_num)
        total_pairs = body_pairs + contact_capacity
        total_nonzeros = self.fem_source_reduced_capacity + self.mpm_source_reduced_capacity + contact_capacity
        if self.assemble_type == "HashTriplet":
            self.monolithic_hash = BuildTriplet(
                dim=3,
                max_pairs_num=max(1, total_pairs),
                max_nonzeros=max(1, total_nonzeros),
                max_active_nodes=self.node_capacity,
                symmetric=False,
                solver="PCG",
                matrix_symmetric=True,
                full_symmetric_input=True,
                device_reduction=True,
            )
        else:
            # FEM and MPM together own one dense diagonal block per coupled
            # node; contact owns another.  Each raw off-diagonal source entry
            # is a dense 3x3 block.  Constraint elimination may append one
            # scalar identity per DOF.
            scalar_capacity = self._coo_scalar_capacity(
                self.node_capacity,
                total_pairs,
                self.dof_capacity,
            )
            self.monolithic_coo = CoordinateSparseMatrix(
                max(1, scalar_capacity),
                self.dof_capacity,
                preconditioned=True,
                symmetry=True,
            )

    @ti.kernel
    def _add_fem_dynamic_diagonal(self, factor: ti.f64):
        for node in range(self.fem_nodes):
            for component in ti.static(range(3)):
                self.fem_hash.diag[node][component * 3 + component] += factor * self.fem.state.mass[node]

    def _assemble_fem_source(self, need_matrix=True):
        internal_force = self.fem._assemble_internal_device(need_stiffness=need_matrix)
        self.fem.state.assemble_implicit_residual(
            internal_force,
            self.fem.damping,
            self.dt,
            self.fem.beta,
            self.fem.gamma,
            int(self.fem.quasi_static),
        )
        if need_matrix:
            stiffness = self.fem._current_stiffness
            if stiffness.values.size:
                raise RuntimeError(
                    "FEMPM IPC cannot consume host FEM triplets; move the "
                    "contribution to a Taichi scatter_stiffness_to_hash kernel"
                )
            self.fem_hash.reset_system()
            if stiffness.base_assembler is not None:
                stiffness.base_assembler.scatter_stiffness_to_hash(self.fem_hash)
            for contribution in stiffness.device_contributions:
                contribution.scatter_stiffness_to_hash(self.fem_hash)
            factor = self.fem._dynamic_diagonal_factor()
            if factor:
                self._add_fem_dynamic_diagonal(factor)
        return internal_force

    def _assemble_mpm_source(self, need_matrix=True):
        active_dof = int(self.mpm.active_dof)
        self.mpm.matrix_reset()
        if need_matrix:
            self.mpm.hash_matrix.reset_system()
        self.mpm.assemble_inertia_force(
            active_dof,
            self.mpm.damping,
            self.mpm.gravity,
            self.mpm.integration,
            self.mpm.grid_disp,
        )
        self.mpm.assemble_material_force(active_dof, self.mpm.grid_disp)
        if need_matrix:
            self.mpm.assemble_stiffness_matrix_hash(active_dof, self.mpm.grid_disp, project_spd=True)
            if mpm_config.DYNAMIC:
                self.mpm.assemble_mass_matrix_hash()
        if self.mpm.neumann.num > 0:
            self.mpm.apply_neumann()
        return active_dof

    @ti.kernel
    def _load_mechanical_rhs(self, active_mpm_dof: ti.i32):
        active_mpm_nodes = active_mpm_dof // ti.static(self.mpm_dimension)
        active_dof = ti.static(3 * self.fem_nodes) + 3 * active_mpm_nodes
        for dof in range(self.dof_capacity):
            value = 0.0
            if dof < ti.static(3 * self.fem_nodes):
                node = dof // 3
                component = dof % 3
                value = -self.fem.state.residual[node][component]
            elif dof < active_dof:
                local = dof - ti.static(3 * self.fem_nodes)
                block = local // 3
                component = local % 3
                if component < ti.static(self.mpm_dimension):
                    value = self.mpm.rhs[ti.static(self.mpm_dimension) * block + component]
            self.rhs[dof] = value
            self.correction[dof] = 0.0
            self.fixed[dof] = 0
            self.fixed_correction[dof] = 0.0

    @ti.kernel
    def _load_trial_mechanical_residual(
        self,
        active_mpm_dof: ti.i32,
        residual: ti.template(),
    ):
        active_mpm_nodes = active_mpm_dof // ti.static(self.mpm_dimension)
        active_dof = ti.static(3 * self.fem_nodes) + 3 * active_mpm_nodes
        for dof in range(self.dof_capacity):
            value = 0.0
            if dof < ti.static(3 * self.fem_nodes):
                node = dof // 3
                component = dof % 3
                value = -self.fem.state.residual[node][component]
            elif dof < active_dof:
                local = dof - ti.static(3 * self.fem_nodes)
                block = local // 3
                component = local % 3
                if component < ti.static(self.mpm_dimension):
                    value = self.mpm.rhs[ti.static(self.mpm_dimension) * block + component]
            residual[dof] = value

    @ti.kernel
    def _free_residual_squared(
        self,
        active_dof: ti.i32,
        residual: ti.template(),
    ) -> ti.f64:
        value = 0.0
        for dof in range(active_dof):
            if self.fixed[dof] == 0:
                value += residual[dof] * residual[dof]
        return value

    @ti.func
    def _append_coo_value(self, row, column, value):
        slot = ti.atomic_add(self.coo_count[None], 1)
        if slot < self.monolithic_coo.data.shape[0]:
            self.monolithic_coo.rows[slot] = row
            self.monolithic_coo.cols[slot] = column
            self.monolithic_coo.data[slot] = value
        else:
            self.coo_overflow[None] = 1

    @ti.kernel
    def _append_hash_to_coo(
        self,
        source: ti.template(),
        active_nodes: ti.i32,
        block_offset: ti.i32,
    ):
        for block in range(active_nodes):
            for row, column in ti.static(ti.ndrange(3, 3)):
                self._append_coo_value(
                    3 * (block + block_offset) + row,
                    3 * (block + block_offset) + column,
                    source.diag[block][3 * row + column],
                )
        for entry in range(source.raw_non_diag_count[0]):
            first = source.non_diag.blockI[entry]
            second = source.non_diag.blockJ[entry]
            if 0 <= first < active_nodes and 0 <= second < active_nodes:
                for row, column in ti.static(ti.ndrange(3, 3)):
                    self._append_coo_value(
                        3 * (first + block_offset) + row,
                        3 * (second + block_offset) + column,
                        source.non_diag.blockH[entry][3 * row + column],
                    )

    @ti.kernel
    def _load_fem_constraints(self):
        for dof in range(ti.static(3 * self.fem_nodes)):
            if self.fem.state.constrained[dof] != 0:
                node = dof // 3
                component = dof % 3
                target = (
                    self.fem.state.reference_position[node][component] + self.fem.state.prescribed_displacement[dof]
                )
                self.fixed[dof] = 1
                self.fixed_correction[dof] = target - self.fem.state.position[node][component]

    @ti.kernel
    def _load_mpm_constraints(self, active_mpm_nodes: ti.i32):
        offset = ti.static(3 * self.fem_nodes)
        for block in range(active_mpm_nodes):
            grid = self.mpm.dof2node[block]
            for component in ti.static(range(self.mpm_dimension)):
                source_dof = ti.static(self.mpm_dimension) * grid + component
                if self.mpm.dirichlet.node[source_dof] != 0:
                    local_dof = 3 * block + component
                    coupled_dof = offset + local_dof
                    self.fixed[coupled_dof] = 1
                    self.fixed_correction[coupled_dof] = (
                        self.mpm.dirichlet.value[source_dof]
                        - self.mpm.grid_disp[ti.static(self.mpm_dimension) * block + component]
                    )

    @ti.kernel
    def _fix_virtual_mpm_components(self, active_mpm_nodes: ti.i32):
        if ti.static(self.mpm_dimension < 3):
            offset = ti.static(3 * self.fem_nodes)
            for block in range(active_mpm_nodes):
                for component in ti.static(range(self.mpm_dimension, 3)):
                    dof = offset + 3 * block + component
                    self.fixed[dof] = 1
                    self.fixed_correction[dof] = 0.0

    @ti.kernel
    def _eliminate_hash_constraints(self, active_nodes: ti.i32):
        for block in range(active_nodes):
            diagonal = self.monolithic_hash.diag[block]
            for row, column in ti.static(ti.ndrange(3, 3)):
                row_dof = 3 * block + row
                column_dof = 3 * block + column
                value = diagonal[3 * row + column]
                if self.fixed[column_dof] != 0:
                    ti.atomic_add(
                        self.rhs[row_dof],
                        -value * self.fixed_correction[column_dof],
                    )
                if self.fixed[row_dof] != 0 or self.fixed[column_dof] != 0:
                    diagonal[3 * row + column] = 0.0
            for component in ti.static(range(3)):
                dof = 3 * block + component
                if self.fixed[dof] != 0:
                    diagonal[component * 3 + component] = 1.0
                    self.rhs[dof] = self.fixed_correction[dof]
            self.monolithic_hash.diag[block] = diagonal
        for entry in range(self.monolithic_hash.raw_non_diag_count[0]):
            first = self.monolithic_hash.non_diag.blockI[entry]
            second = self.monolithic_hash.non_diag.blockJ[entry]
            if 0 <= first < active_nodes and 0 <= second < active_nodes:
                block = self.monolithic_hash.non_diag.blockH[entry]
                for row, column in ti.static(ti.ndrange(3, 3)):
                    row_dof = 3 * first + row
                    column_dof = 3 * second + column
                    value = block[3 * row + column]
                    if self.fixed[column_dof] != 0:
                        ti.atomic_add(
                            self.rhs[row_dof],
                            -value * self.fixed_correction[column_dof],
                        )
                    if self.fixed[row_dof] != 0:
                        ti.atomic_add(
                            self.rhs[column_dof],
                            -value * self.fixed_correction[row_dof],
                        )
                    if self.fixed[row_dof] != 0 or self.fixed[column_dof] != 0:
                        block[3 * row + column] = 0.0
                self.monolithic_hash.non_diag.blockH[entry] = block

    @ti.kernel
    def _eliminate_coo_constraints(self, active_dof: ti.i32):
        count = self.coo_count[None]
        for entry in range(count):
            row = self.monolithic_coo.rows[entry]
            column = self.monolithic_coo.cols[entry]
            value = self.monolithic_coo.data[entry]
            if 0 <= row < active_dof and 0 <= column < active_dof:
                if self.fixed[column] != 0:
                    ti.atomic_add(self.rhs[row], -value * self.fixed_correction[column])
                if self.fixed[row] != 0 or self.fixed[column] != 0:
                    self.monolithic_coo.data[entry] = 0.0
        for dof in range(active_dof):
            if self.fixed[dof] != 0:
                self._append_coo_value(dof, dof, 1.0)
                self.rhs[dof] = self.fixed_correction[dof]

    @ti.kernel
    def _build_coo_diagonal(self, active_dof: ti.i32):
        for dof in range(self.dof_capacity):
            self.coo_diagonal[dof] = 0.0
        for entry in range(self.coo_count[None]):
            row = self.monolithic_coo.rows[entry]
            column = self.monolithic_coo.cols[entry]
            if row == column and 0 <= row < active_dof:
                ti.atomic_add(self.coo_diagonal[row], self.monolithic_coo.data[entry])
        for dof in range(active_dof):
            if ti.abs(self.coo_diagonal[dof]) < 1.0e-14:
                self.coo_diagonal[dof] = 1.0

    @ti.kernel
    def _copy_physical_rhs(self, active_dof: ti.i32):
        for dof in range(self.dof_capacity):
            self.physical_rhs[dof] = self.rhs[dof] if dof < active_dof else 0.0

    def assemble_system(self, include_friction=True, need_matrix=True):
        count = self.contact.prepare(self.fem.state.position, self.mpm.grid_disp)
        self._ensure_matrices(count)
        self._assemble_fem_source(need_matrix=need_matrix)
        active_mpm_dof = self._assemble_mpm_source(need_matrix=need_matrix)
        active_mpm_nodes = active_mpm_dof // self.mpm_dimension
        active_nodes = self.fem_nodes + active_mpm_nodes
        active_dof = 3 * active_nodes
        self._load_mechanical_rhs(active_mpm_dof)
        if need_matrix:
            self.contact_hash.reset_system()
        self.contact.assemble(
            count,
            self.contact.candidate_field(),
            self.contact_hash,
            self.rhs,
            bool(need_matrix),
            bool(include_friction and self.contact.activate_friction),
            int(self.contact.friction_count),
        )
        self._copy_physical_rhs(active_dof)

        matrix = None
        if need_matrix:
            if self.assemble_type == "HashTriplet":
                matrix = self.monolithic_hash
                matrix.reset_system()
                matrix.append_raw_from(self.fem_hash, active_nodes=self.fem_nodes, block_offset=0)
                self.mpm_embedded_hash.reset_system()
                self._embed_mpm_source(active_mpm_nodes)
                matrix.append_raw_from(
                    self.mpm_embedded_hash,
                    active_nodes=active_mpm_nodes,
                    block_offset=self.fem_nodes,
                )
                matrix.append_raw_from(self.contact_hash, active_nodes=active_nodes, block_offset=0)
                matrix.canonicalize_full_symmetric_input()
            else:
                matrix = self.monolithic_coo
                matrix.reset()
                self.coo_count[None] = 0
                self.coo_overflow[None] = 0
                self._append_hash_to_coo(self.fem_hash, self.fem_nodes, 0)
                self.mpm_embedded_hash.reset_system()
                self._embed_mpm_source(active_mpm_nodes)
                self._append_hash_to_coo(self.mpm_embedded_hash, active_mpm_nodes, self.fem_nodes)
                self._append_hash_to_coo(self.contact_hash, active_nodes, 0)

        self._load_fem_constraints()
        self._fix_virtual_mpm_components(active_mpm_nodes)
        if self.mpm.dirichlet.num > 0:
            self._load_mpm_constraints(active_mpm_nodes)
        if need_matrix:
            if self.assemble_type == "HashTriplet":
                self._eliminate_hash_constraints(active_nodes)
                matrix.finalize_taichi_assembly()
            else:
                self._eliminate_coo_constraints(active_dof)
                if int(self.coo_overflow[None]) != 0:
                    raise RuntimeError("FEMPM COO capacity exceeded while assembling IPC")
                matrix.linear_operator.update_active_dofs(active_dof)
                matrix.linear_operator.update_nnz(int(self.coo_count[None]))
                self._build_coo_diagonal(active_dof)
        return {
            "matrix": matrix,
            "active_mpm_dof": active_mpm_dof,
            "active_nodes": active_nodes,
            "active_dof": active_dof,
            "contact_count": count,
        }

    @ti.kernel
    def _load_host_solution(self, active_dof: ti.i32, values: ti.types.ndarray(dtype=ti.f64, ndim=1)):
        for dof in range(self.dof_capacity):
            self.correction[dof] = values[dof] if dof < active_dof else 0.0

    def _solve_linear_system(self, system):
        active_dof = int(system["active_dof"])
        active_nodes = int(system["active_nodes"])
        self.correction.fill(0.0)
        if self.linear_solver == "Scipy":
            if self.assemble_type == "HashTriplet":
                matrix = self.monolithic_hash.to_scipy(active_nodes).tocsr()
            else:
                matrix = self.monolithic_coo._to_scipy().tocsr()
                matrix = matrix[:active_dof, :active_dof]
            from scipy.sparse.linalg import spsolve

            solution = np.asarray(spsolve(matrix, self.rhs.to_numpy()[:active_dof]), dtype=np.float64)
            if not np.all(np.isfinite(solution)):
                raise RuntimeError("FEMPM Scipy linear solve returned non-finite values")
            self._load_host_solution(active_dof, solution)
            result = {
                "converged": True,
                "residual": 0.0,
                "iterations": -1,
                "backend": "scipy_spsolve",
            }
        elif self.assemble_type == "HashTriplet":
            result = self.monolithic_hash.solve_flat_system(
                self.rhs,
                self.correction,
                active_nodes=active_nodes,
                tol=self.linear_solver_tolerance,
                rel_tol=self.linear_solver_relative_tolerance,
                maxiter=self.linear_solver_max_iters,
                return_solution=False,
            )
            result["backend"] = "taichi_hash_pcg"
        else:
            converged = self.monolithic_coo.solve(
                self.rhs,
                self.correction,
                self.coo_diagonal,
                tol=self.linear_solver_tolerance,
                rel_tol=self.linear_solver_relative_tolerance,
                maxiter=self.linear_solver_max_iters,
            )
            result = {
                "converged": bool(converged),
                "residual": float(self.monolithic_coo.linear_solver.last_residual),
                "iterations": int(self.monolithic_coo.linear_solver.last_iterations),
                "backend": "taichi_coo_pcg",
            }
        if not result["converged"]:
            raise NewtonConvergenceError(
                "FEMPM monolithic linear solve did not converge: "
                f"residual={result['residual']:.6e}, "
                f"initial={result.get('initial_residual', math.nan):.6e}, "
                f"target={result.get('convergence_tolerance', math.nan):.6e}, "
                f"iterations={result['iterations']}"
            )
        self.last_linear_solve = result
        return result

    @ti.kernel
    def _split_direction(self, active_mpm_dof: ti.i32):
        for node, component in ti.ndrange(self.fem_nodes, 3):
            dof = 3 * node + component
            self.fem.state.direction[node][component] = self.correction[dof]
        offset = ti.static(3 * self.fem_nodes)
        for dof in range(self.mpm.degree_of_freedom):
            value = 0.0
            if dof < active_mpm_dof:
                block = dof // ti.static(self.mpm_dimension)
                component = dof % ti.static(self.mpm_dimension)
                value = self.correction[offset + 3 * block + component]
            self.mpm.incre_resolution[dof] = value

    @ti.kernel
    def _reduce_system_metrics(self, active_dof: ti.i32):
        self.residual_squared[None] = 0.0
        self.directional_derivative[None] = 0.0
        self.correction_inf_norm[None] = 0.0
        for dof in range(active_dof):
            ti.atomic_add(self.residual_squared[None], self.rhs[dof] ** 2)
            ti.atomic_add(
                self.directional_derivative[None],
                -self.rhs[dof] * self.correction[dof],
            )
            ti.atomic_max(self.correction_inf_norm[None], ti.abs(self.correction[dof]))

    @ti.kernel
    def _accept_mpm_trial(self, active_mpm_dof: ti.i32):
        for dof in range(active_mpm_dof):
            self.mpm.grid_disp[dof] = self.mpm.grid_disp_temp[dof]

    @ti.kernel
    def _snapshot_mpm_physical_state(self):
        for particle in self.mpm.particle:
            self.snapshot_mpm_position[particle] = self.mpm.particle[particle].x
            self.snapshot_mpm_velocity[particle] = self.mpm.particle[particle].v
            self.snapshot_mpm_acceleration[particle] = self.mpm.particle[particle].a
            self.snapshot_mpm_deformation[particle] = self.mpm.F0[particle]
            if ti.static(self.mpm_has_plastic_history):
                self.snapshot_mpm_plastic_history[particle] = self.mpm.material.get_history_state(particle)
        for node in self.mpm.grid:
            self.snapshot_grid_mass[node] = self.mpm.grid[node].m
            self.snapshot_grid_velocity[node] = self.mpm.grid[node].v
            self.snapshot_grid_acceleration[node] = self.mpm.grid[node].a

    @ti.kernel
    def _restore_failed_step_state(self):
        for node in self.fem.state.position:
            self.fem.state.position[node] = self.fem.state.old_position[node]
            self.fem.state.velocity[node] = self.fem.state.old_velocity[node]
            self.fem.state.acceleration[node] = self.fem.state.old_acceleration[node]
            self.fem.state.trial_position[node] = self.fem.state.old_position[node]
            self.fem.state.predicted_position[node] = self.fem.state.old_position[node]
        for particle in self.mpm.particle:
            self.mpm.particle[particle].x = self.snapshot_mpm_position[particle]
            self.mpm.particle[particle].v = self.snapshot_mpm_velocity[particle]
            self.mpm.particle[particle].a = self.snapshot_mpm_acceleration[particle]
            self.mpm.F0[particle] = self.snapshot_mpm_deformation[particle]
            if ti.static(self.mpm_has_plastic_history):
                self.mpm.material.set_history_state(
                    particle,
                    self.snapshot_mpm_plastic_history[particle],
                )
        for node in self.mpm.grid:
            self.mpm.grid[node].m = self.snapshot_grid_mass[node]
            self.mpm.grid[node].v = self.snapshot_grid_velocity[node]
            self.mpm.grid[node].a = self.snapshot_grid_acceleration[node]

    def _total_energy(self, fem_positions, mpm_displacement, include_friction):
        self.fem._assemble_internal_device_at(fem_positions, need_stiffness=False)
        self.fem.state.reduce_dynamic_potential(
            fem_positions,
            self.fem.damping,
            self.dt,
            self.fem.beta,
            self.fem.gamma,
            int(self.fem.quasi_static),
        )
        fem_energy = self.fem._internal_energy_device() + float(self.fem.state.dynamic_potential[None])
        mpm_energy = float(self.mpm.total_energy(mpm_displacement))
        count = self.contact.prepare(fem_positions, mpm_displacement)
        self.energy_scratch_rhs.fill(0.0)
        self.contact.assemble(
            count,
            self.contact.candidate_field(),
            self.contact_hash,
            self.energy_scratch_rhs,
            False,
            bool(include_friction and self.contact.activate_friction),
            int(self.contact.friction_count),
        )
        return fem_energy + mpm_energy + float(self.contact.total_energy[None])

    def _trial_physical_residual(self, fem_positions, mpm_displacement, include_friction):
        count = self.contact.prepare(fem_positions, mpm_displacement)
        internal_force = self.fem._assemble_internal_device_at(
            fem_positions,
            need_stiffness=False,
        )
        self.fem.state.assemble_implicit_residual_at(
            fem_positions,
            internal_force,
            self.fem.damping,
            self.dt,
            self.fem.beta,
            self.fem.gamma,
            int(self.fem.quasi_static),
        )
        active_mpm_dof = int(self.mpm.active_dof)
        self.mpm.rhs.fill(0.0)
        self.mpm.assemble_inertia_force(
            active_mpm_dof,
            self.mpm.damping,
            self.mpm.gravity,
            self.mpm.integration,
            mpm_displacement,
        )
        self.mpm.assemble_material_force(active_mpm_dof, mpm_displacement)
        if self.mpm.neumann.num > 0:
            self.mpm.apply_neumann()
        self._load_trial_mechanical_residual(
            active_mpm_dof,
            self.energy_scratch_rhs,
        )
        self.contact.assemble(
            count,
            self.contact.candidate_field(),
            self.contact_hash,
            self.energy_scratch_rhs,
            False,
            bool(include_friction and self.contact.activate_friction),
            int(self.contact.friction_count),
        )
        active_mpm_nodes = active_mpm_dof // self.mpm_dimension
        active_dof = 3 * (self.fem_nodes + active_mpm_nodes)
        return math.sqrt(
            max(
                float(
                    self._free_residual_squared(
                        active_dof,
                        self.energy_scratch_rhs,
                    )
                ),
                0.0,
            )
        )

    def _residual_merit_line_search(
        self,
        system,
        include_friction,
        current_residual,
    ):
        active_mpm_dof = int(system["active_mpm_dof"])
        current_merit = 0.5 * current_residual * current_residual
        merit_slope = -current_residual * current_residual
        mpm_material_step = float(self.mpm.ccd(active_mpm_dof))
        fem_material_step = self._fem_material_step()
        contact_step = self.contact.maximum_step(
            self.fem.state.position,
            self.fem.state.direction,
            self.mpm.grid_disp,
            self.mpm.incre_resolution,
        )
        alpha = min(1.0, mpm_material_step, fem_material_step, contact_step)
        for backtrack in range(self.line_search_max_backtracks + 1):
            if alpha < self.line_search_minimum_step:
                break
            self.fem.state.set_trial_position(alpha)
            self.mpm.update_grid_disp(active_mpm_dof, alpha)
            trial_residual = math.inf
            try:
                if self.fem._minimum_jacobian_ratio_device(self.fem.state.trial_position) > self.fem.minimum_jacobian:
                    trial_residual = self._trial_physical_residual(
                        self.fem.state.trial_position,
                        self.mpm.grid_disp_temp,
                        include_friction,
                    )
            except (FloatingPointError, RuntimeError, ValueError):
                trial_residual = math.inf
            trial_merit = 0.5 * trial_residual * trial_residual
            if math.isfinite(trial_merit) and trial_merit <= (
                current_merit + self.line_search_c1 * alpha * merit_slope
            ):
                self.fem.state.accept_trial_position()
                self._accept_mpm_trial(active_mpm_dof)
                self.contact.accept_update(self.fem.state.position, self.mpm.grid_disp)
                return alpha, backtrack, trial_merit
            alpha *= self.line_search_reduction
        raise NewtonConvergenceError(
            "FEMPM IPC residual-merit Armijo line search failed; reduce the "
            "timestep or increase contact/linear solver accuracy"
        )

    def _line_search(self, system, include_friction, current_residual):
        if self.mpm_uses_residual_merit:
            return self._residual_merit_line_search(
                system,
                include_friction,
                current_residual,
            )
        active_mpm_dof = int(system["active_mpm_dof"])
        active_dof = int(system["active_dof"])
        self._split_direction(active_mpm_dof)
        self._reduce_system_metrics(active_dof)
        slope = float(self.directional_derivative[None])
        if not math.isfinite(slope) or slope >= 0.0:
            raise NewtonConvergenceError("FEMPM projected Newton direction is not a descent direction")
        previous = self._total_energy(self.fem.state.position, self.mpm.grid_disp, include_friction)
        energy_tolerance = self.line_search_energy_atol + self.line_search_energy_rtol * max(1.0, abs(previous))
        mpm_material_step = float(self.mpm.ccd(active_mpm_dof))
        fem_material_step = self._fem_material_step()
        contact_step = self.contact.maximum_step(
            self.fem.state.position,
            self.fem.state.direction,
            self.mpm.grid_disp,
            self.mpm.incre_resolution,
        )
        alpha = min(
            1.0,
            mpm_material_step,
            fem_material_step,
            contact_step,
        )
        for backtrack in range(self.line_search_max_backtracks + 1):
            if alpha < self.line_search_minimum_step:
                break
            self.fem.state.set_trial_position(alpha)
            self.mpm.update_grid_disp(active_mpm_dof, alpha)
            minimum_jacobian = self.fem._minimum_jacobian_ratio_device(self.fem.state.trial_position)
            trial = math.inf
            if minimum_jacobian > self.fem.minimum_jacobian:
                trial = self._total_energy(
                    self.fem.state.trial_position,
                    self.mpm.grid_disp_temp,
                    include_friction,
                )
            if math.isfinite(trial) and trial <= (previous + self.line_search_c1 * alpha * slope + energy_tolerance):
                self.fem.state.accept_trial_position()
                self._accept_mpm_trial(active_mpm_dof)
                self.contact.accept_update(self.fem.state.position, self.mpm.grid_disp)
                return alpha, backtrack, trial
            alpha *= self.line_search_reduction
        raise NewtonConvergenceError(
            "FEMPM IPC Armijo line search failed; reduce the timestep or " "increase contact/linear solver accuracy"
        )

    @ti.kernel
    def _store_fem_equilibrium(self):
        for node, component in ti.ndrange(self.fem_nodes, 3):
            dof = 3 * node + component
            value = -self.physical_rhs[dof]
            self.fem.state.residual[node][component] = value
            self.fem.state.reaction[node][component] = value if self.fem.state.constrained[dof] != 0 else 0.0

    def _solve_newton(self, include_friction, verbose):
        initial_norm = None
        records = []
        converged = False
        last_system = None
        semi_progress = 0.0
        for iteration in range(self.max_iterations + 1):
            if getattr(self.contact, "is_semi", False) and iteration > 1 and semi_progress > 0.999:
                converged = True
                break
            last_system = self.assemble_system(include_friction=include_friction, need_matrix=True)
            self._reduce_system_metrics(int(last_system["active_dof"]))
            residual_norm = math.sqrt(max(float(self.residual_squared[None]), 0.0))
            if initial_norm is None:
                initial_norm = max(residual_norm, 1.0)
            record = {
                "iteration": iteration,
                "residual_norm": residual_norm,
                "relative_residual": residual_norm / initial_norm,
                "residual_tolerance": (self.absolute_tolerance + self.residual_tolerance * initial_norm),
                "convergence_reason": None,
                "line_search_step": 0.0,
                **self.contact.diagnostics(),
            }
            records.append(record)
            if (
                residual_norm <= (self.absolute_tolerance + self.residual_tolerance * initial_norm)
                and self.contact.contact_converged()
            ):
                record["convergence_reason"] = "force_residual"
                converged = True
                break
            if iteration == self.max_iterations:
                break
            linear_solve = self._solve_linear_system(last_system)
            self._split_direction(int(last_system["active_mpm_dof"]))
            self._reduce_system_metrics(int(last_system["active_dof"]))
            correction_norm = float(self.correction_inf_norm[None])
            correction_velocity = correction_norm / self.dt
            record["correction_inf_norm"] = correction_norm
            record["correction_velocity"] = correction_velocity
            record["correction_velocity_tolerance"] = self.correction_velocity_tolerance
            record["linear_solve"] = dict(linear_solve)
            if correction_velocity <= self.correction_velocity_tolerance and self.contact.contact_converged():
                record["convergence_reason"] = "physical_correction_velocity"
                converged = True
                if verbose:
                    print(
                        f"FEMPM implicit Newton {iteration + 1}: "
                        f"residual={residual_norm:.4e}, "
                        f"correction_velocity={correction_velocity:.4e} "
                        "(converged)"
                    )
                break
            alpha, backtracks, _ = self._line_search(
                last_system,
                include_friction,
                residual_norm,
            )
            record["line_search_step"] = alpha
            record["line_search_backtracks"] = backtracks
            self.fem.apply_boundary_step(self.dt, 0)
            if getattr(self.contact, "is_semi", False):
                semi_progress += (1.0 - semi_progress) * alpha
            if verbose:
                print(f"FEMPM implicit Newton {iteration + 1}: " f"residual={residual_norm:.4e}, alpha={alpha:.4e}")
        if not converged and self.raise_on_nonconvergence:
            correction = records[-1].get("correction_velocity", math.inf)
            raise NewtonConvergenceError(
                "FEMPM implicit solve did not converge: "
                f"residual={records[-1]['residual_norm']:.3e}, "
                f"correction_velocity={correction:.3e} after "
                f"{self.max_iterations} iterations"
            )
        return converged, records, last_system

    def _updated_friction_residual(self, include_friction):
        system = self.assemble_system(include_friction=include_friction, need_matrix=True)
        self._reduce_system_metrics(int(system["active_dof"]))
        if math.sqrt(max(float(self.residual_squared[None]), 0.0)) <= self.absolute_tolerance:
            return 0.0
        self._solve_linear_system(system)
        self._split_direction(int(system["active_mpm_dof"]))
        self._reduce_system_metrics(int(system["active_dof"]))
        return float(self.correction_inf_norm[None]) / self.dt

    def _solve_lagged_friction_fixed_point(self, verbose=True):
        include_friction = self.contact.activate_friction
        requested_outer = self.contact_model.friction_iterations if include_friction else 1
        outer_limit = self.contact_model.friction_max_iterations if requested_outer == -1 else requested_outer
        outer_records = []
        converged = True
        last_system = None
        self.last_friction_iterations = 0
        self.last_friction_residual = 0.0 if not include_friction else math.inf
        self.last_friction_converged = not include_friction
        for outer in range(outer_limit):
            outer_converged, records, last_system = self._solve_newton(include_friction, verbose)
            converged = converged and outer_converged
            outer_records.append(records)
            if not outer_converged:
                break
            if include_friction:
                self.last_friction_iterations = outer + 1
                self.contact.refresh_friction(self.fem.state.position, self.mpm.grid_disp)
                self.last_friction_residual = self._updated_friction_residual(include_friction=True)
                if not math.isfinite(self.last_friction_residual):
                    raise NewtonConvergenceError("FEMPM lagged IPC friction residual is non-finite")
                if self.last_friction_residual <= self.contact_model.friction_tolerance:
                    self.last_friction_converged = True
                    break

        if requested_outer == -1 and not self.last_friction_converged:
            raise NewtonConvergenceError(
                "FEMPM lagged IPC friction fixed point did not converge within "
                f"{outer_limit} iterations (residual={self.last_friction_residual:.6e}, "
                f"tolerance={self.contact_model.friction_tolerance:.6e})"
            )
        return converged, outer_records, last_system

    def solve_lagged_friction_fixed_point(self, verbose=True):
        self.mpm.begin_lagged_material_state()
        if not self.mpm.has_lagged_material:
            return self._solve_lagged_friction_fixed_point(verbose)

        all_outer_records = []
        last_system = None
        for _ in range(self.mpm.material_lagged_max_iterations):
            converged, outer_records, last_system = self._solve_lagged_friction_fixed_point(verbose)
            all_outer_records.extend(outer_records)
            if not converged:
                return False, all_outer_records, last_system
            if self.mpm.refresh_lagged_material_state(self.mpm.grid_disp) <= self.mpm.material_lagged_tolerance:
                return True, all_outer_records, last_system
        raise NewtonConvergenceError(
            "FEMPM lagged MCC hardening did not converge: "
            f"error={self.mpm.last_material_lagged_error:.6e} after "
            f"{self.mpm.material_lagged_max_iterations} iterations"
        )

    def _advance_prepared_substep(self, verbose=True):
        next_time = self.time + self.dt
        self.fem.set_boundary_data_step(next_time, self.step_count + 1)
        self.fem.apply_boundary_step(self.dt, 0)
        self.fem.update_external_force_step(next_time, self.step_count + 1)
        self.fem.state.build_newmark_prediction(self.dt, self.fem.beta)
        self.contact.begin_step(self.fem.state.position, self.mpm.grid_disp, self.dt)

        converged, outer_records, last_system = self.solve_lagged_friction_fixed_point(verbose)
        include_friction = self.contact.activate_friction

        # Reassemble the accepted state so reactions include IPC and friction.
        last_system = self.assemble_system(include_friction=include_friction, need_matrix=False)
        self._store_fem_equilibrium()
        self.fem.state.finalize_newmark(
            self.dt,
            self.fem.beta,
            self.fem.gamma,
            int(self.fem.quasi_static),
        )
        self.mpm.update_nodal_acc(self.mpm.integration)
        self.mpm.advent_particles(self.mpm.coeffPIC)
        self.advance_constitutive_state()
        self.time = next_time
        self.step_count += 1
        self.simulation.current_time = self.time
        self.simulation.current_step = self.step_count
        self.fem.time = self.time
        self.fem.step_count = self.step_count
        self.mpm_wrapper.sims.current_time = self.time
        self.mpm_wrapper.sims.current_step = self.step_count
        self._mpm_step_prepared = False
        record = {
            "step": self.step_count,
            "time": self.time,
            "converged": converged,
            "friction_iterations": outer_records,
            "friction_iteration_count": self.last_friction_iterations,
            "friction_residual": self.last_friction_residual,
            "friction_converged": self.last_friction_converged,
            "material_lagged_iterations": self.mpm.last_material_lagged_iterations,
            "material_lagged_error": self.mpm.last_material_lagged_error,
            "material": type(self.mpm.material).__name__,
            "minimum_jacobian": self.minimum_jacobian,
            "contact": self.contact.diagnostics(),
            "active_mpm_dof": int(last_system["active_mpm_dof"]),
            "assembly": self.assemble_type,
            "linear_solver": self.linear_solver,
        }
        self.last_step_record = record
        if self.record_history_step:
            self.sample_energy_step()
            self.add_energy_record(record)
            self.step_schedule.append_history(self.history, record)
        return converged

    def _add_energy_record(self, record):
        record["fem_kinetic_energy"] = float(self.fem.state.kinetic_energy[None])

    def _set_timestep(self, timestep):
        value = float(timestep)
        self.dt = value
        self.simulation.set_timestep(value)
        self.mpm_wrapper.sims.set_timestep(value)
        self.fem.dt = value
        self.mpm.dt = value

    def _failure_diagnostics(self, exception, attempt, timestep):
        try:
            contact = self.contact.diagnostics()
        except Exception as diagnostic_error:
            contact = {"unavailable": str(diagnostic_error)}
        return {
            "kind": nonlinear_failure_kind(exception),
            "exception": type(exception).__name__,
            "message": str(exception),
            "attempt": int(attempt),
            "timestep": float(timestep),
            "time": float(self.time),
            "step": int(self.step_count),
            "contact": contact,
            "linear_solver": self.last_linear_solve,
        }

    def diagnostics_snapshot(self):
        try:
            contact = self.contact.diagnostics()
        except Exception as diagnostic_error:
            contact = {"unavailable": str(diagnostic_error)}
        return {
            "schema_version": 1,
            "subsystem": "fempm_implicit_ipc",
            "time": float(self.time),
            "step": int(self.step_count),
            "timestep": float(self.dt),
            "contact": contact,
            "friction": {
                "iterations": int(self.last_friction_iterations),
                "residual": float(self.last_friction_residual),
                "converged": bool(self.last_friction_converged),
            },
            "linear_solver": self.last_linear_solve,
            "last_failure": self.last_failure,
            "last_step": self.history[-1] if self.history else None,
        }

    def _substep_once(self, verbose=True):
        entry_time = self.time
        entry_step = self.step_count
        with self.simulation.timer.section("Step setup"):
            self.fem.state.save_step_state()
            self._snapshot_mpm_physical_state()
        try:
            with self.simulation.timer.section("MPM preparation"):
                self._prepare_mpm_step()
            with self.simulation.timer.section("Coupled nonlinear solve"):
                return self._advance_prepared_substep(verbose)
        except BaseException as exception:
            try:
                self._restore_failed_step_state()
                self.fem.state.direction.fill(0.0)
                self.mpm.grid_disp.fill(0.0)
                self.mpm.grid_disp_temp.fill(0.0)
                self.mpm.incre_resolution.fill(0.0)
            except BaseException as rollback_error:
                raise exception from rollback_error
            finally:
                self._mpm_step_prepared = False
                self.time = entry_time
                self.step_count = entry_step
                self.simulation.current_time = entry_time
                self.simulation.current_step = entry_step
                self.fem.time = entry_time
                self.fem.step_count = entry_step
                self.mpm_wrapper.sims.current_time = entry_time
                self.mpm_wrapper.sims.current_step = entry_step
            raise

    def substep(self, verbose=True, record_history=True):
        self.record_history_step = bool(record_history)
        original_timestep = self.dt
        attempt_timestep = original_timestep
        attempts = []
        for attempt in range(self.step_retry.maximum_retries + 1):
            self._set_timestep(attempt_timestep)
            try:
                result = self._substep_once(verbose=verbose)
            except NewtonConvergenceError as exception:
                failure = self._failure_diagnostics(exception, attempt, attempt_timestep)
                attempts.append(failure)
                next_timestep = self.step_retry.next_timestep(attempt_timestep, attempt)
                if next_timestep is None:
                    self.last_failure = {
                        **failure,
                        "original_timestep": float(original_timestep),
                        "attempts": attempts,
                    }
                    self._set_timestep(original_timestep)
                    raise
                if verbose:
                    print(
                        "FEMPM implicit step retry: "
                        f"{attempt_timestep:.6e} -> {next_timestep:.6e} "
                        f"after {failure['kind']}"
                    )
                attempt_timestep = next_timestep
                continue

            retry_record = {
                "enabled": bool(self.step_retry.enabled),
                "original_timestep": float(original_timestep),
                "accepted_timestep": float(attempt_timestep),
                "retry_count": int(attempt),
                "attempts": attempts,
            }
            if self.last_step_record is not None:
                self.last_step_record["step_retry"] = retry_record
            self.last_failure = None
            return result

        raise AssertionError("unreachable FEMPM retry state")

    def run(self, steps=None, verbose=True, postprocessing=()):
        target_time = float(self.simulation.time)
        target_step = None if steps is None else self.step_count + int(steps)
        time_tolerance = max(1.0e-14, 64.0 * math.ulp(max(1.0, abs(target_time))))
        overall = True

        def has_remaining_step():
            return self.time < target_time - time_tolerance if target_step is None else self.step_count < target_step

        def clip_final_timestep():
            if target_step is None:
                remaining = target_time - self.time
                if time_tolerance < remaining < self.dt:
                    self._set_timestep(remaining)

        def advance_one():
            nonlocal overall
            clip_final_timestep()
            next_step = self.step_count + 1
            final_step = (
                next_step >= target_step
                if target_step is not None
                else self.time + self.dt >= target_time - time_tolerance
            )
            overall = (
                self.substep(
                    verbose=verbose,
                    record_history=self.step_schedule.history_due(next_step, final=final_step),
                )
                and overall
            )
            runtime_checkpoint()
            with self.simulation.timer.section("Postprocess"):
                for callback in postprocessing:
                    callback(self)

        if self.compile_seconds is None and has_remaining_step():
            print("Compiling first ... ...")
            compile_start = time.perf_counter()
            clip_final_timestep()
            next_step = self.step_count + 1
            final_step = (
                next_step >= target_step
                if target_step is not None
                else self.time + self.dt >= target_time - time_tolerance
            )
            overall = (
                self.substep(
                    verbose=verbose,
                    record_history=self.step_schedule.history_due(next_step, final=final_step),
                )
                and overall
            )
            ti.sync()
            self.compile_seconds = time.perf_counter() - compile_start
            print(f"Compiling time = {self.compile_seconds} \n")
            self.simulation.timer.profile1()
            runtime_checkpoint()
            with self.simulation.timer.section("Postprocess"):
                for callback in postprocessing:
                    callback(self)
        while has_remaining_step():
            advance_one()
        if target_step is None and abs(self.time - target_time) <= time_tolerance:
            self.time = target_time
            self.simulation.current_time = target_time
            self.fem.time = target_time
            self.mpm_wrapper.sims.current_time = target_time
        return {
            "converged": overall,
            "time": self.time,
            "step": self.step_count,
            "history": list(self.history),
        }


__all__ = ["FEMPMImplicitEngine"]
