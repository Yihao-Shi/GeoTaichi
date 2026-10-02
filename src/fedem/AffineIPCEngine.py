"""Monolithic implicit FEM--AffineBody solver with mixed IPC and friction."""

import math
import time

import numpy as np
import taichi as ti

from src.dem.engines.AffineBodyOperator import MATRIX_HASH_TRIPLET
from src.fedem.contact.AffineIPCAssembler import FEMAffineIPCAssembler
from src.fem.engines.ImplicitFEM import NewtonConvergenceError
from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.StepRetry import StepRetryPolicy, nonlinear_failure_kind
from src.utils.linalg import no_operation


def _assembly_name(value):
    key = str(value).replace("_", "").replace("-", "").lower()
    if key in ("hash", "hashtriplet", "triplet", "buildtriplet"):
        return "HashTriplet"
    if key in ("coo", "coordinate", "coordinatesparse"):
        return "COO"
    raise ValueError("FEM--AffineBody assemble_type must be COO or HashTriplet")


def _solver_name(value):
    key = str(value).replace("_", "").replace("-", "").lower()
    if key in ("pcg", "cg", "taichipcg"):
        return "PCG"
    if key in ("scipy", "spsolve", "cpu", "direct"):
        return "Scipy"
    raise ValueError("FEM--AffineBody linear_solver must be PCG or Scipy")


@ti.data_oriented
class FEMAffineIPCEngine:
    """Projected Newton solve over affine controls and FEM nodal DOFs."""

    def __init__(
        self,
        simulation,
        fem,
        dem,
        contact_model,
        fem_faces=None,
        **kwargs,
    ):
        self.simulation = simulation
        self.fem_wrapper = fem
        self.dem_wrapper = dem
        self.fem = fem.engine
        self.dem_engine = dem.enginer
        self.dem_engine.initialize(dem.sims, dem.scene)
        self.affine = self.dem_engine.operator
        if self.affine.fully_implicit:
            raise RuntimeError("FEM--AffineBody IPC currently requires AffineBody " "self-friction_mode='lagged'")
        requested_friction_iterations = int(getattr(dem.sims, "affine_friction_iterations", 1))
        if requested_friction_iterations != 1:
            raise RuntimeError("FEM--AffineBody IPC currently requires AffineBody " "friction_iterations=1")
        self.contact_model = contact_model
        self.dt = float(simulation.delta)
        self.time = float(simulation.current_time)
        self.step_count = int(simulation.current_step)
        self.compile_seconds = None
        self.max_iterations = int(kwargs.get("max_iterations", self.fem.max_iterations))
        self.residual_tolerance = float(kwargs.get("residual_tolerance", self.fem.residual_tolerance))
        self.absolute_tolerance = float(kwargs.get("absolute_tolerance", self.fem.absolute_tolerance))
        self.correction_velocity_tolerance = float(
            kwargs.get(
                "correction_velocity_tolerance",
                kwargs.get(
                    "newton_velocity_tolerance",
                    getattr(dem.sims, "affine_newton_tolerance", 1.0e-7),
                ),
            )
        )
        self.raise_on_nonconvergence = bool(kwargs.get("raise_on_nonconvergence", True))
        self.line_search_reduction = float(kwargs.get("line_search_reduction", 0.5))
        self.line_search_c1 = float(kwargs.get("line_search_sufficient_decrease", 1.0e-4))
        self.line_search_max_backtracks = int(kwargs.get("line_search_max_backtracks", 24))
        self.line_search_minimum_step = float(kwargs.get("line_search_minimum_step", 1.0e-10))
        self.step_retry = StepRetryPolicy(
            enabled=kwargs.get("enable_step_retry", False),
            maximum_retries=kwargs.get("step_retry_max_retries", 2),
            reduction=kwargs.get("step_retry_reduction", 0.5),
            minimum_timestep=kwargs.get("step_retry_minimum_timestep", 0.0),
        )
        self.assemble_type = _assembly_name(kwargs.get("assemble_type", self.fem.assemble_type))
        self.linear_solver = _solver_name(kwargs.get("linear_solver", self.fem.linear_solver))
        self.linear_solver_tolerance = float(kwargs.get("linear_solver_tolerance", 1.0e-10))
        self.linear_solver_relative_tolerance = float(kwargs.get("linear_solver_relative_tolerance", 0.0))
        if not math.isfinite(self.linear_solver_relative_tolerance) or self.linear_solver_relative_tolerance < 0.0:
            raise ValueError("FEM--AffineBody linear_solver_relative_tolerance must be " "finite and non-negative")
        self.linear_solver_max_iters = int(
            kwargs.get(
                "linear_solver_max_iters",
                max(
                    500,
                    5 * (self.affine.dof + self.fem.degree_of_freedom),
                ),
            )
        )
        if self.max_iterations <= 0:
            raise ValueError("FEM--AffineBody max_iterations must be positive")
        if not math.isfinite(self.correction_velocity_tolerance) or self.correction_velocity_tolerance <= 0.0:
            raise ValueError("FEM--AffineBody correction_velocity_tolerance must be " "finite and positive")
        if not 0.0 < self.line_search_reduction < 1.0:
            raise ValueError("line_search_reduction must be in (0, 1)")

        self.affine_controls = int(self.affine.control_num)
        self.fem_nodes = int(self.fem.mesh.number_of_nodes)
        self.node_count = self.affine_controls + self.fem_nodes
        self.dof_count = 3 * self.node_count
        self.rhs = ti.field(dtype=ti.f64, shape=self.dof_count)
        self.physical_rhs = ti.field(dtype=ti.f64, shape=self.dof_count)
        self.affine_source_rhs = ti.field(dtype=ti.f64, shape=max(1, 3 * self.affine_controls))
        self.affine_contact_resultant = ti.Vector.field(3, dtype=ti.f64, shape=max(1, self.affine.body_num))
        self.affine_prescribed_body = ti.field(dtype=ti.i32, shape=max(1, self.affine.body_num))
        self.correction = ti.field(dtype=ti.f64, shape=self.dof_count)
        self.energy_rhs = ti.field(dtype=ti.f64, shape=self.dof_count)
        self.fixed = ti.field(dtype=ti.i32, shape=self.dof_count)
        self.residual_squared = ti.field(dtype=ti.f64, shape=())
        self.directional_derivative = ti.field(dtype=ti.f64, shape=())
        self.correction_inf_norm = ti.field(dtype=ti.f64, shape=())
        self.physical_correction_inf_norm = ti.field(dtype=ti.f64, shape=())
        self.minimum_jacobian = 1.0
        self.history = []
        self.last_step_record = None
        self.record_history_step = True
        self.last_linear_solve = None
        self.last_failure = None
        self.last_friction_iterations = 0
        self.last_friction_residual = math.inf
        self.last_friction_converged = False
        self.last_line_search_limits = None
        self.affine_velocity_body_ids = set()
        self.affine_pressure_servos = {}
        self.affine_pressure_history = []
        self.step_schedule = simulation.step_schedule
        self.track_energy = bool(self.fem.track_energy)
        self.advance_constitutive_state = getattr(self.fem, "_advance_constitutive_state", no_operation)
        self.sample_energy_step = self.fem.state.reduce_kinetic_energy if self.track_energy else no_operation
        self.add_energy_record = self._add_energy_record if self.track_energy else no_operation

        self.contact = FEMAffineIPCAssembler(
            self.fem,
            self.affine,
            contact_model,
            simulation,
            fem_faces=fem_faces,
        )
        if self.step_retry.enabled and self.contact.is_semi:
            raise ValueError(
                "FEM--AffineBody step retry does not support SemiIPC because its multipliers advance inside Newton iterations"
            )
        self.fem_contact = getattr(self.fem, "contact_assembler", None)
        self.fem_hash = self._build_fem_source_matrix()
        self.affine_source_capacity = self._affine_source_raw_capacity()
        self.affine_source_reduced_capacity = max(
            1,
            min(
                self.affine_source_capacity,
                self.affine_controls * max(0, self.affine_controls - 1),
            ),
        )
        self.affine_hash = BuildTriplet(
            dim=3,
            max_pairs_num=self.affine_source_capacity,
            # Affine assembly appends scalar entries, but reduction keys are
            # 3x3 control blocks.  At most A(A-1) directed off-diagonal block
            # coordinates can survive regardless of scalar scatter count.
            max_nonzeros=self.affine_source_reduced_capacity,
            max_active_nodes=max(1, self.affine_controls),
            symmetric=False,
            solver="BiCGSTAB",
            matrix_symmetric=False,
            device_reduction=True,
            raw_only=True,
        )
        self.affine.bind_hash_triplet(self.affine_hash, full_symmetric_input=True)
        self.contact_hash = None
        self.monolithic_hash = None
        self.monolithic_coo = None
        self._contact_capacity = 0
        self.coo_count = ti.field(dtype=ti.i32, shape=())
        self.coo_overflow = ti.field(dtype=ti.i32, shape=())
        self.coo_diagonal = ti.field(dtype=ti.f64, shape=self.dof_count)
        initial_pt, initial_ee = self.contact.prepare(self.fem.state.position)
        self._ensure_matrices(initial_pt, initial_ee)

    def _affine_source_raw_capacity(self):
        """Bound affine raw scatters from fixed PT/EE capacities."""
        vf_candidates = int(getattr(self.dem_wrapper.sims, "max_point_triangle_pairs", 0))
        ee_candidates = int(getattr(self.dem_wrapper.sims, "max_edge_edge_pairs", 0))
        # The coupled source emits both global triangles before monolithic
        # canonicalization.  Ask the affine operator for that exact storage
        # mode directly; do not double a standalone estimate that contains a
        # safety factor and minimum allocation floor.
        return max(
            1,
            int(
                self.affine._estimate_hash_triplet_capacity(
                    self.dem_wrapper.sims,
                    vf_candidate_capacity=vf_candidates,
                    ee_candidate_capacity=ee_candidates,
                    full_symmetric_input=True,
                    safety_factor=1.0,
                    minimum_capacity=1,
                )
            ),
        )

    def _configured_contact_raw_capacity(self):
        """Block-scatter bound implied by fixed mixed PT/EE capacities."""
        pt_capacity = int(self.simulation.max_point_triangle_pairs)
        ee_capacity = int(self.simulation.max_edge_edge_pairs)
        contribution_count = 2 if self.contact.activate_friction else 1
        return max(
            1,
            contribution_count * (169 * pt_capacity + 100 * ee_capacity),
        )

    @staticmethod
    def _coo_scalar_capacity(
        affine_controls,
        fem_nodes,
        affine_raw_blocks,
        fem_raw_blocks,
        contact_raw_blocks,
    ):
        """Return a safe scalar COO capacity for the three block sources.

        All three sources append dense 3x3 blocks, hence nine scalar COO slots
        per raw block.  Each source also owns an accumulated dense diagonal
        which is appended separately.
        """
        affine_controls = int(affine_controls)
        fem_nodes = int(fem_nodes)
        node_count = affine_controls + fem_nodes
        dof_count = 3 * node_count
        source_diagonals = 9 * (affine_controls + fem_nodes + node_count)
        source_off_diagonals = 9 * (int(affine_raw_blocks) + int(fem_raw_blocks) + int(contact_raw_blocks))
        # Constraint elimination may append one scalar identity per DOF.
        return source_diagonals + source_off_diagonals + dof_count

    def _fem_assembler(self):
        assembler = getattr(
            self.fem,
            "classical_assembler",
            getattr(self.fem, "cloth_assembler", None),
        )
        if assembler is None:
            raise RuntimeError("FEM--AffineBody IPC requires a Taichi FEM assembler")
        return assembler

    def _build_fem_source_matrix(self):
        assembler = self._fem_assembler()
        contact_capacity = self._configured_fem_contact_raw_capacity()
        capacity = max(
            1,
            int(assembler.stiffness_block_pair_count) + contact_capacity,
        )
        reduced_capacity = max(
            1,
            int(
                getattr(
                    assembler,
                    "stiffness_unique_block_pair_count",
                    capacity,
                )
            )
            + (contact_capacity + 1) // 2,
        )
        self.fem_source_reduced_capacity = reduced_capacity
        return BuildTriplet(
            dim=3,
            max_pairs_num=capacity,
            max_nonzeros=reduced_capacity,
            max_active_nodes=self.fem_nodes,
            symmetric=False,
            solver="BiCGSTAB",
            matrix_symmetric=False,
            device_reduction=True,
            raw_only=True,
        )

    def _fem_contact_assemblers(self):
        if self.fem_contact is None:
            return ()
        return tuple(getattr(self.fem_contact, "assemblers", (self.fem_contact,)))

    def _configured_fem_contact_raw_capacity(self):
        """Bound FEM IPC off-diagonal block scatters from declared pair caps."""
        capacity = 0
        for assembler in self._fem_contact_assemblers():
            contribution_count = 2 if assembler.activate_friction else 1
            capacity += (
                12 * contribution_count * (int(assembler.max_point_triangle_pairs) + int(assembler.max_edge_edge_pairs))
            )
        return capacity

    def _friction_controls(self):
        settings = []
        if self.contact.activate_friction:
            settings.append(self.contact_model)
        settings.extend(
            assembler.contact for assembler in self._fem_contact_assemblers() if assembler.activate_friction
        )
        if not settings:
            return False, 1, False, math.inf
        automatic = any(setting.friction_iterations == -1 for setting in settings)
        outer_count = (
            max(setting.friction_max_iterations for setting in settings)
            if automatic
            else max(setting.friction_iterations for setting in settings)
        )
        tolerance = min(setting.friction_tolerance for setting in settings)
        return True, outer_count, automatic, tolerance

    def _contact_diagnostics(self):
        result = dict(self.contact.diagnostics())
        if self.fem_contact is not None:
            result["fem_contact"] = self.fem_contact.device_diagnostics()
        return result

    def _ensure_matrices(self, pt_count, ee_count):
        pt_count = int(pt_count)
        ee_count = int(ee_count)
        if pt_count > int(self.simulation.max_point_triangle_pairs):
            raise RuntimeError(
                "FEM--AffineBody max_point_triangle_pairs is too small: "
                f"need {pt_count}, allocated "
                f"{self.simulation.max_point_triangle_pairs}"
            )
        if ee_count > int(self.simulation.max_edge_edge_pairs):
            raise RuntimeError(
                "FEM--AffineBody max_edge_edge_pairs is too small: "
                f"need {ee_count}, allocated "
                f"{self.simulation.max_edge_edge_pairs}"
            )
        required = self.contact.contact_block_capacity(pt_count, ee_count)
        if required <= self._contact_capacity:
            return
        configured_capacity = self._configured_contact_raw_capacity()
        if required > configured_capacity:
            raise RuntimeError(
                "FEM--AffineBody contact block capacity is too small: "
                f"need {required}, configured bound {configured_capacity}"
            )
        # Allocate from the declared contact-pair contract once.  This keeps
        # both HashTriplet and COO capacities deterministic as the active set
        # changes during Newton/CCD iterations.
        self._contact_capacity = configured_capacity
        self.contact_hash = BuildTriplet(
            dim=3,
            max_pairs_num=self._contact_capacity,
            max_nonzeros=self._contact_capacity,
            max_active_nodes=self.node_count,
            symmetric=False,
            solver="BiCGSTAB",
            matrix_symmetric=False,
            device_reduction=True,
            raw_only=True,
        )
        total_pairs = (
            int(self.affine_hash.non_diag.max_pairs_num)
            + int(self.fem_hash.non_diag.max_pairs_num)
            + self._contact_capacity
        )
        # The three sources contain a complete symmetric input.  The
        # monolithic PCG matrix canonicalizes it to one global block triangle,
        # so its reduced-pattern capacity is the sum of the per-source upper
        # bounds, not the full directed/raw count.
        total_nonzeros = (
            (self.affine_source_reduced_capacity + 1) // 2
            + (self.fem_source_reduced_capacity + 1) // 2
            + (self._contact_capacity + 1) // 2
        )
        if self.assemble_type == "HashTriplet":
            self.monolithic_hash = BuildTriplet(
                dim=3,
                max_pairs_num=max(1, total_pairs),
                max_nonzeros=max(1, total_nonzeros),
                max_active_nodes=self.node_count,
                symmetric=False,
                solver="PCG",
                matrix_symmetric=True,
                full_symmetric_input=True,
                device_reduction=True,
            )
        else:
            scalar_capacity = self._coo_scalar_capacity(
                self.affine_controls,
                self.fem_nodes,
                self.affine_hash.non_diag.max_pairs_num,
                self.fem_hash.non_diag.max_pairs_num,
                self._contact_capacity,
            )
            if self.monolithic_coo is not None:
                self.monolithic_coo.clear()
            self.monolithic_coo = CoordinateSparseMatrix(
                max(1, scalar_capacity),
                self.dof_count,
                preconditioned=True,
                symmetry=True,
            )

    @ti.kernel
    def _add_fem_dynamic_diagonal(self, factor: ti.f64):
        for node in range(self.fem_nodes):
            for component in ti.static(range(3)):
                self.fem_hash.diag[node][3 * component + component] += factor * self.fem.state.mass[node]

    def _assemble_fem_source(self, need_matrix, assembler=None):
        self.fem._prepare_contact_iteration_device(self.fem.state.position)
        if assembler is None:
            internal_force = self.fem._assemble_internal_device(need_stiffness=need_matrix)
            stiffness = self.fem._current_stiffness
        else:
            if self.fem_contact is not None:
                raise ValueError("exact coupled FEM adjoints do not yet support FEM self-contact")
            internal_force, stiffness = assembler.assemble_device(
                self.fem.state.position,
                need_stiffness=need_matrix,
            )
        self.fem.state.assemble_implicit_residual(
            internal_force,
            self.fem.damping,
            self.dt,
            self.fem.beta,
            self.fem.gamma,
            int(self.fem.quasi_static),
        )
        if need_matrix:
            if stiffness.values.size:
                raise RuntimeError("FEM--AffineBody cannot consume host FEM triplets")
            self.fem_hash.reset_system()
            if stiffness.base_assembler is not None:
                stiffness.base_assembler.scatter_stiffness_to_hash(self.fem_hash)
            for contribution in stiffness.device_contributions:
                contribution.scatter_stiffness_to_hash(self.fem_hash)
            factor = self.fem._dynamic_diagonal_factor()
            if factor:
                self._add_fem_dynamic_diagonal(factor)
        return internal_force

    @ti.kernel
    def _load_rhs(self, inverse_affine_scale: ti.f64):
        for dof in range(self.dof_count):
            value = 0.0
            if dof < ti.static(3 * self.affine_controls):
                control = dof // 3
                component = dof % 3
                value = -inverse_affine_scale * self.affine.grad[control][component]
            else:
                local = dof - ti.static(3 * self.affine_controls)
                node = local // 3
                component = local % 3
                value = -self.fem.state.residual[node][component]
            self.rhs[dof] = value
            self.correction[dof] = 0.0
            self.fixed[dof] = 0

    @ti.kernel
    def _load_fem_constraints(self):
        for node, component in ti.ndrange(self.fem_nodes, 3):
            local_dof = 3 * node + component
            dof = ti.static(3 * self.affine_controls) + local_dof
            if self.fem.state.constrained[local_dof] != 0:
                self.fixed[dof] = 1
                self.rhs[dof] = 0.0

    @ti.kernel
    def _load_affine_constraints(self):
        for control, component in ti.ndrange(self.affine_controls, 3):
            if self.affine_prescribed_body[control // 4] != 0:
                dof = 3 * control + component
                self.fixed[dof] = 1
                self.rhs[dof] = 0.0

    @ti.kernel
    def _copy_affine_source_rhs(self):
        for dof in range(3 * self.affine_controls):
            self.affine_source_rhs[dof] = self.rhs[dof]

    @ti.kernel
    def _measure_affine_contact_resultant(self):
        for body in range(self.affine.body_num):
            self.affine_contact_resultant[body] = ti.Vector.zero(ti.f64, 3)
        for control, component in ti.ndrange(self.affine_controls, 3):
            ti.atomic_add(
                self.affine_contact_resultant[control // 4][component],
                self.rhs[3 * control + component] - self.affine_source_rhs[3 * control + component],
            )

    @ti.kernel
    def _copy_physical_rhs(self):
        for dof in range(self.dof_count):
            self.physical_rhs[dof] = self.rhs[dof]

    @ti.kernel
    def _eliminate_hash_constraints(self):
        for block in range(self.node_count):
            for row, column in ti.static(ti.ndrange(3, 3)):
                row_dof = 3 * block + row
                column_dof = 3 * block + column
                index = 3 * row + column
                if self.fixed[row_dof] != 0 or self.fixed[column_dof] != 0:
                    self.monolithic_hash.diag[block][index] = 0.0
            for component in ti.static(range(3)):
                dof = 3 * block + component
                if self.fixed[dof] != 0:
                    self.monolithic_hash.diag[block][3 * component + component] = 1.0
        for entry in range(self.monolithic_hash.raw_non_diag_count[0]):
            if entry < self.monolithic_hash.non_diag.blockI.shape[0]:
                block_i = self.monolithic_hash.non_diag.blockI[entry]
                block_j = self.monolithic_hash.non_diag.blockJ[entry]
                if block_i >= 0 and block_j >= 0:
                    block = self.monolithic_hash.non_diag.blockH[entry]
                    for row, column in ti.static(ti.ndrange(3, 3)):
                        if self.fixed[3 * block_i + row] != 0 or self.fixed[3 * block_j + column] != 0:
                            block[3 * row + column] = 0.0
                    self.monolithic_hash.non_diag.blockH[entry] = block

    @ti.func
    def _append_coo_value(self, row, column, value):
        entry = ti.atomic_add(self.coo_count[None], 1)
        if entry < self.monolithic_coo.capacity:
            self.monolithic_coo.rows[entry] = row
            self.monolithic_coo.cols[entry] = column
            self.monolithic_coo.data[entry] = value
        else:
            self.coo_overflow[None] = 1

    @ti.kernel
    def _append_hash_to_coo(
        self,
        source: ti.template(),
        active_nodes: ti.i32,
        block_offset: ti.i32,
        scale: ti.f64,
    ):
        for block in range(active_nodes):
            for row, column in ti.static(ti.ndrange(3, 3)):
                value = scale * source.diag[block][3 * row + column]
                if value != 0.0:
                    self._append_coo_value(
                        3 * (block + block_offset) + row,
                        3 * (block + block_offset) + column,
                        value,
                    )
        for entry in range(source.raw_non_diag_count[0]):
            if entry < source.non_diag.blockI.shape[0]:
                block_i = source.non_diag.blockI[entry]
                block_j = source.non_diag.blockJ[entry]
                if block_i >= 0 and block_j >= 0:
                    for row, column in ti.static(ti.ndrange(3, 3)):
                        value = scale * source.non_diag.blockH[entry][3 * row + column]
                        if value != 0.0:
                            self._append_coo_value(
                                3 * (block_i + block_offset) + row,
                                3 * (block_j + block_offset) + column,
                                value,
                            )

    @ti.kernel
    def _eliminate_coo_constraints(self):
        for entry in range(self.coo_count[None]):
            if self.fixed[self.monolithic_coo.rows[entry]] != 0 or self.fixed[self.monolithic_coo.cols[entry]] != 0:
                self.monolithic_coo.data[entry] = 0.0
        for dof in range(self.dof_count):
            if self.fixed[dof] != 0:
                self._append_coo_value(dof, dof, 1.0)

    @ti.kernel
    def _build_coo_diagonal(self):
        for dof in range(self.dof_count):
            self.coo_diagonal[dof] = 0.0
        for entry in range(self.coo_count[None]):
            if self.monolithic_coo.rows[entry] == self.monolithic_coo.cols[entry]:
                ti.atomic_add(
                    self.coo_diagonal[self.monolithic_coo.rows[entry]],
                    self.monolithic_coo.data[entry],
                )
        for dof in range(self.dof_count):
            if ti.abs(self.coo_diagonal[dof]) < 1.0e-14:
                self.coo_diagonal[dof] = 1.0

    def assemble_system(
        self,
        need_matrix=True,
        *,
        project_spd=True,
        solver_shift=True,
        fem_assembler=None,
    ):
        inverse_scale = 1.0 / (self.dt * self.dt)
        if need_matrix:
            self.affine_hash.reset_system()
            self.affine.bind_hash_triplet(self.affine_hash, full_symmetric_input=True)
        affine_energy = self.affine.assemble_device(
            need_matrix=bool(need_matrix),
            matrix_mode=MATRIX_HASH_TRIPLET,
            project_spd=bool(project_spd),
            solver_shift=bool(solver_shift),
        )
        fem_internal_force = self._assemble_fem_source(bool(need_matrix), assembler=fem_assembler)
        pt_count, ee_count = self.contact.prepare(self.fem.state.position)
        self._ensure_matrices(pt_count, ee_count)
        self._load_rhs(inverse_scale)
        self._copy_affine_source_rhs()
        if need_matrix:
            self.contact_hash.reset_system()
        self.contact.assemble(
            pt_count,
            ee_count,
            self.contact_hash,
            self.rhs,
            bool(need_matrix),
            project_spd=bool(project_spd),
        )
        self._measure_affine_contact_resultant()
        self._copy_physical_rhs()

        matrix = None
        if need_matrix:
            if self.assemble_type == "HashTriplet":
                matrix = self.monolithic_hash
                matrix.reset_system()
                matrix.append_raw_from(
                    self.affine_hash,
                    active_nodes=self.affine_controls,
                    block_offset=0,
                    scale=inverse_scale,
                )
                matrix.append_raw_from(
                    self.fem_hash,
                    active_nodes=self.fem_nodes,
                    block_offset=self.affine_controls,
                )
                matrix.append_raw_from(
                    self.contact_hash,
                    active_nodes=self.node_count,
                    block_offset=0,
                )
                matrix.canonicalize_full_symmetric_input()
            else:
                matrix = self.monolithic_coo
                matrix.reset()
                self.coo_count[None] = 0
                self.coo_overflow[None] = 0
                self._append_hash_to_coo(
                    self.affine_hash,
                    self.affine_controls,
                    0,
                    inverse_scale,
                )
                self._append_hash_to_coo(
                    self.fem_hash,
                    self.fem_nodes,
                    self.affine_controls,
                    1.0,
                )
                self._append_hash_to_coo(self.contact_hash, self.node_count, 0, 1.0)
        self._load_affine_constraints()
        self._load_fem_constraints()
        if need_matrix:
            if self.assemble_type == "HashTriplet":
                self._eliminate_hash_constraints()
                matrix.finalize_taichi_assembly()
            else:
                self._eliminate_coo_constraints()
                if int(self.coo_overflow[None]) != 0:
                    raise RuntimeError("FEM--AffineBody COO capacity exceeded")
                matrix.linear_operator.update_active_dofs(self.dof_count)
                matrix.linear_operator.update_nnz(int(self.coo_count[None]))
                self._build_coo_diagonal()
        self.fem.state.reduce_dynamic_potential(
            self.fem.state.position,
            self.fem.damping,
            self.dt,
            self.fem.beta,
            self.fem.gamma,
            int(self.fem.quasi_static),
        )
        energy = (
            inverse_scale * affine_energy
            + self.fem._internal_energy_device()
            + float(self.fem.state.dynamic_potential[None])
            + float(self.contact.total_energy[None])
        )
        return {
            "matrix": matrix,
            "energy": energy,
            "fem_internal_force": fem_internal_force,
        }

    @ti.kernel
    def _load_host_solution(self, values: ti.types.ndarray(dtype=ti.f64, ndim=1)):
        for dof in range(self.dof_count):
            self.correction[dof] = values[dof]

    def _solve_linear_system(self, system):
        self.correction.fill(0.0)
        if self.linear_solver == "Scipy":
            if self.assemble_type == "HashTriplet":
                matrix = self.monolithic_hash.to_scipy(self.node_count).tocsr()
            else:
                matrix = self.monolithic_coo._to_scipy().tocsr()
            from scipy.sparse.linalg import spsolve

            solution = np.asarray(spsolve(matrix, self.rhs.to_numpy()), dtype=np.float64)
            if not np.all(np.isfinite(solution)):
                raise RuntimeError("FEM--AffineBody Scipy solve returned non-finite values")
            self._load_host_solution(solution)
            result = {
                "converged": True,
                "iterations": -1,
                "residual": 0.0,
                "backend": "scipy_spsolve",
            }
        elif self.assemble_type == "HashTriplet":
            result = self.monolithic_hash.solve_flat_system(
                self.rhs,
                self.correction,
                active_nodes=self.node_count,
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
                maxiter=self.linear_solver_max_iters,
                rel_tol=self.linear_solver_relative_tolerance,
            )
            solver = self.monolithic_coo.linear_solver
            result = {
                "converged": bool(converged),
                "iterations": int(solver.last_iterations),
                "residual": float(solver.last_residual),
                "backend": "taichi_coo_pcg",
            }
        self.last_linear_solve = result
        if not result["converged"]:
            raise NewtonConvergenceError(
                "FEM--AffineBody linear solve did not converge: "
                f"step={self.step_count + 1}, time={self.time + self.dt:.6e}, "
                f"residual={result['residual']:.6e}, "
                f"recursive={result.get('recursive_residual', math.nan):.6e}, "
                f"initial={result.get('initial_residual', math.nan):.6e}, "
                f"target={result.get('convergence_tolerance', math.nan):.6e}, "
                f"iterations={result['iterations']}"
            )

    @ti.kernel
    def _split_direction(self):
        for control, component in ti.ndrange(self.affine_controls, 3):
            self.affine.direction_y[control][component] = self.correction[3 * control + component]
        for node, component in ti.ndrange(self.fem_nodes, 3):
            dof = ti.static(3 * self.affine_controls) + 3 * node + component
            self.fem.state.direction[node][component] = self.correction[dof]

    @ti.kernel
    def _reduce_metrics(self):
        self.residual_squared[None] = 0.0
        self.directional_derivative[None] = 0.0
        self.correction_inf_norm[None] = 0.0
        for dof in range(self.dof_count):
            ti.atomic_add(self.residual_squared[None], self.rhs[dof] ** 2)
            ti.atomic_add(
                self.directional_derivative[None],
                -self.rhs[dof] * self.correction[dof],
            )
            ti.atomic_max(self.correction_inf_norm[None], ti.abs(self.correction[dof]))

    @ti.kernel
    def _reduce_physical_correction(self):
        """Measure a Newton update in actual surface/FEM coordinates.

        Affine control points are an auxiliary coordinate frame and can move
        much farther than the represented surface.  The ABD convergence
        measure is therefore the maximum physical surface displacement,
        combined here with the maximum FEM nodal displacement.
        """
        self.physical_correction_inf_norm[None] = 0.0
        for vertex in range(self.affine.vertex_num):
            body = self.affine.node2body[vertex]
            direction = ti.Vector.zero(ti.f64, 3)
            for local_control in ti.static(range(4)):
                direction += (
                    self.affine.basis[vertex, local_control] * self.affine.direction_y[body * 4 + local_control]
                )
            for component in ti.static(range(3)):
                ti.atomic_max(
                    self.physical_correction_inf_norm[None],
                    ti.abs(direction[component]),
                )
        for node, component in ti.ndrange(self.fem_nodes, 3):
            ti.atomic_max(
                self.physical_correction_inf_norm[None],
                ti.abs(self.fem.state.direction[node][component]),
            )

    def _updated_friction_residual(self):
        system = self.assemble_system(need_matrix=True)
        self._reduce_metrics()
        if math.sqrt(max(float(self.residual_squared[None]), 0.0)) <= self.absolute_tolerance:
            return 0.0
        self._solve_linear_system(system)
        self._split_direction()
        self._reduce_physical_correction()
        return float(self.physical_correction_inf_norm[None]) / self.dt

    def _total_energy(self):
        self.energy_rhs.fill(0.0)
        inverse_scale = 1.0 / (self.dt * self.dt)
        affine_energy = self.affine.assemble_device(need_matrix=False, matrix_mode=MATRIX_HASH_TRIPLET)
        self.fem._prepare_contact_iteration_device(self.fem.state.position)
        self.fem._assemble_internal_device_at(self.fem.state.position, need_stiffness=False)
        self.fem.state.reduce_dynamic_potential(
            self.fem.state.position,
            self.fem.damping,
            self.dt,
            self.fem.beta,
            self.fem.gamma,
            int(self.fem.quasi_static),
        )
        pt_count, ee_count = self.contact.prepare(self.fem.state.position)
        self.contact.assemble(
            pt_count,
            ee_count,
            self.contact_hash,
            self.energy_rhs,
            False,
        )
        return (
            inverse_scale * affine_energy
            + self.fem._internal_energy_device()
            + float(self.fem.state.dynamic_potential[None])
            + float(self.contact.total_energy[None])
        )

    def _line_search(self, base_energy):
        self._split_direction()
        self._reduce_metrics()
        slope = float(self.directional_derivative[None])
        if not math.isfinite(slope) or slope >= 0.0:
            raise NewtonConvergenceError("FEM--AffineBody Newton direction is not descending")
        affine_step = self.affine.init_step_size_device(
            ccd_type="ccd",
            eta=self.contact.ccd_eta,
            max_iteration=self.contact.ccd_max_iterations,
        )
        fem_step = self.fem._maximum_admissible_step_device()
        mixed_step = self.contact.maximum_step(self.fem.state.position, self.fem.state.direction)
        self.last_line_search_limits = {
            "affine": float(affine_step),
            "fem": float(fem_step),
            "mixed": float(mixed_step),
        }
        alpha = min(1.0, affine_step, fem_step, mixed_step)
        initial_alpha = alpha
        trials = []
        self.affine.device_backup_line_search_base()
        for backtrack in range(self.line_search_max_backtracks + 1):
            if alpha < self.line_search_minimum_step:
                break
            self.affine.device_set_line_search_trial(alpha)
            self.fem.state.set_trial_position(alpha)
            self.fem.state.accept_trial_position()
            minimum = self.fem._minimum_jacobian_ratio_device(self.fem.state.position)
            trial = math.inf
            if minimum > self.fem.minimum_jacobian:
                trial = self._total_energy()
            armijo = base_energy + self.line_search_c1 * alpha * slope
            trials.append((float(alpha), float(trial), float(armijo)))
            if math.isfinite(trial) and trial <= armijo:
                self.fem._after_nonlinear_update_device(alpha)
                self.contact.accept_update(self.fem.state.position)
                return alpha, backtrack, trial
            # Restore FEM before constructing the next trial.  Affine trials
            # are always formed from the operator's saved base field.
            self.fem.state.set_trial_position(-alpha)
            self.fem.state.accept_trial_position()
            alpha *= self.line_search_reduction
        self.affine.device_restore_line_search_base()
        raise NewtonConvergenceError(
            "FEM--AffineBody IPC Armijo line search failed: "
            f"base={base_energy:.17e}, slope={slope:.17e}, "
            f"initial_alpha={initial_alpha:.17e}, limits={self.last_line_search_limits}, "
            f"trials={trials[-8:]}"
        )

    @ti.kernel
    def _restore_fem_step(self):
        for node in range(self.fem_nodes):
            self.fem.state.position[node] = self.fem.state.old_position[node]
            self.fem.state.velocity[node] = self.fem.state.old_velocity[node]
            self.fem.state.acceleration[node] = self.fem.state.old_acceleration[node]

    @ti.kernel
    def _store_fem_equilibrium(self):
        offset = ti.static(3 * self.affine_controls)
        for node, component in ti.ndrange(self.fem_nodes, 3):
            local = 3 * node + component
            value = -self.physical_rhs[offset + local]
            self.fem.state.residual[node][component] = value
            self.fem.state.reaction[node][component] = value if self.fem.state.constrained[local] != 0 else 0.0

    def pre_calculate(self):
        return None

    def _set_timestep(self, timestep):
        value = float(timestep)
        self.dt = value
        self.simulation.set_timestep(value)
        self.dem_wrapper.sims.set_timestep(value)
        self.fem.dt = value

    def _failure_diagnostics(self, exception, attempt, timestep):
        try:
            contact = self._contact_diagnostics()
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
            "line_search_limits": self.last_line_search_limits,
            "linear_solver": self.last_linear_solve,
        }

    def diagnostics_snapshot(self):
        try:
            contact = self._contact_diagnostics()
        except Exception as diagnostic_error:
            contact = {"unavailable": str(diagnostic_error)}
        return {
            "schema_version": 1,
            "subsystem": "fem_affine_ipc",
            "time": float(self.time),
            "step": int(self.step_count),
            "timestep": float(self.dt),
            "contact": contact,
            "line_search_limits": self.last_line_search_limits,
            "friction": {
                "iterations": int(self.last_friction_iterations),
                "residual": float(self.last_friction_residual),
                "converged": bool(self.last_friction_converged),
            },
            "linear_solver": self.last_linear_solve,
            "last_failure": self.last_failure,
            "last_step": self.history[-1] if self.history else None,
            "affine_pressure_servos": list(self.affine_pressure_history),
        }

    def prescribe_affine_body_velocity(self, body_id, velocity):
        body_id = int(body_id)
        if not 0 <= body_id < self.affine.body_num:
            raise IndexError(f"AffineBody id {body_id} is outside [0, {self.affine.body_num})")
        velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(velocity)):
            raise ValueError("prescribed AffineBody velocity must be finite")
        self.affine_velocity_body_ids.add(body_id)
        self.affine_prescribed_body[body_id] = 1
        self.affine.device_set_body_translation_velocity(body_id, *velocity.tolist())

    @staticmethod
    def _pressure_servo_velocity(contact_force, inward_normal, target_force, velocity_gain, max_velocity):
        measured_force = max(0.0, -float(np.dot(contact_force, inward_normal)))
        relative_error = (float(target_force) - measured_force) / float(target_force)
        speed = float(np.clip(float(velocity_gain) * relative_error, -float(max_velocity), float(max_velocity)))
        return speed * np.asarray(inward_normal, dtype=np.float64), measured_force

    def add_affine_body_pressure_servo(
        self,
        body_id,
        inward_normal,
        area,
        target_pressure,
        velocity_gain=0.01,
        max_velocity=0.02,
    ):
        body_id = int(body_id)
        normal = np.asarray(inward_normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(normal))
        values = (float(area), float(target_pressure), float(velocity_gain), float(max_velocity), norm)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("pressure-servo area, pressure, gains, speed and normal norm must be finite and positive")
        normal /= norm
        self.affine_pressure_servos[body_id] = {
            "body_id": body_id,
            "inward_normal": normal,
            "area": float(area),
            "target_pressure": float(target_pressure),
            "target_force": float(area) * float(target_pressure),
            "velocity_gain": float(velocity_gain),
            "max_velocity": float(max_velocity),
            "measured_force": 0.0,
            "measured_pressure": 0.0,
        }
        self.prescribe_affine_body_velocity(body_id, velocity_gain * normal)

    def _apply_affine_velocity_boundaries(self):
        for body_id in self.affine_velocity_body_ids:
            self.affine.device_apply_body_translation_prediction(body_id)

    def _update_affine_pressure_servos(self):
        if not self.affine_pressure_servos:
            return
        contact_force = self.affine_contact_resultant.to_numpy()
        step_record = {"step": int(self.step_count + 1), "time": float(self.time + self.dt), "walls": []}
        for body_id, servo in self.affine_pressure_servos.items():
            velocity, measured_force = self._pressure_servo_velocity(
                contact_force[body_id],
                servo["inward_normal"],
                servo["target_force"],
                servo["velocity_gain"],
                servo["max_velocity"],
            )
            servo["measured_force"] = measured_force
            servo["measured_pressure"] = measured_force / servo["area"]
            self.affine.device_set_body_translation_velocity(body_id, *velocity.tolist())
            step_record["walls"].append(
                {
                    "body_id": int(body_id),
                    "target_pressure": servo["target_pressure"],
                    "measured_pressure": servo["measured_pressure"],
                    "normal_velocity": float(np.dot(velocity, servo["inward_normal"])),
                }
            )
        self.affine_pressure_history.append(step_record)

    def _add_energy_record(self, record):
        record["kinetic_energy"] = float(self.fem.state.kinetic_energy[None])

    def _solve_frozen_friction_newton(self, verbose=False):
        initial_norm = None
        converged = False
        records = []
        semi_progress = 0.0
        for iteration in range(self.max_iterations + 1):
            if getattr(self.contact, "is_semi", False) and iteration > 1 and semi_progress > 0.999:
                converged = True
                break
            with self.simulation.timer.section("System assembly"):
                system = self.assemble_system(need_matrix=True)
                self._reduce_metrics()
            residual = math.sqrt(max(float(self.residual_squared[None]), 0.0))
            if initial_norm is None:
                initial_norm = max(residual, 1.0)
            convergence_tolerance = self.absolute_tolerance + self.residual_tolerance * initial_norm
            records.append(
                {
                    "iteration": iteration,
                    "residual_norm": residual,
                    "convergence_tolerance": convergence_tolerance,
                    "convergence_reason": None,
                    **self.contact.diagnostics(),
                }
            )
            if residual <= convergence_tolerance and self.contact.contact_converged():
                records[-1]["convergence_reason"] = "force_residual"
                converged = True
                break
            if iteration == self.max_iterations:
                break
            with self.simulation.timer.section("Linear solve"):
                self._solve_linear_system(system)
            self._split_direction()
            self._reduce_physical_correction()
            correction_norm = float(self.physical_correction_inf_norm[None])
            correction_velocity = correction_norm / self.dt
            records[-1]["correction_inf_norm"] = correction_norm
            records[-1]["correction_velocity"] = correction_velocity
            records[-1]["correction_velocity_tolerance"] = self.correction_velocity_tolerance
            if self.contact.contact_converged() and (
                correction_velocity == 0.0
                or (iteration > 0 and correction_velocity <= self.correction_velocity_tolerance)
            ):
                records[-1]["convergence_reason"] = "physical_correction_velocity"
                converged = True
                if verbose:
                    print(
                        f"FEM--AffineBody Newton {iteration + 1}: "
                        f"residual={residual:.4e}, "
                        f"correction_velocity={correction_velocity:.4e} (converged)"
                    )
                break
            with self.simulation.timer.section("Line search"):
                alpha, backtracks, _ = self._line_search(system["energy"])
            records[-1]["line_search_step"] = alpha
            records[-1]["line_search_backtracks"] = backtracks
            records[-1]["line_search_limits"] = self.last_line_search_limits
            self.fem.apply_boundary_step(self.dt, 0)
            if getattr(self.contact, "is_semi", False):
                semi_progress += (1.0 - semi_progress) * alpha
            if verbose:
                limiter = min(self.last_line_search_limits, key=self.last_line_search_limits.get)
                print(
                    f"FEM--AffineBody Newton {iteration + 1}: "
                    f"residual={residual:.4e}, "
                    f"correction_velocity={correction_velocity:.4e}, "
                    f"alpha={alpha:.4e}, ccd_limiter={limiter}"
                )
        return converged, records

    def solve_lagged_friction_fixed_point(self, next_time, verbose=False):
        activate_friction, outer_count, automatic, friction_tolerance = self._friction_controls()
        outer_records = []
        friction_residuals = []
        converged = True
        self.last_friction_iterations = 0
        self.last_friction_residual = 0.0 if not activate_friction else math.inf
        self.last_friction_converged = not activate_friction
        for outer in range(outer_count):
            self.affine.device_backup_lagged_friction_for_adjoint()
            self.contact.backup_lagged_friction_for_adjoint_device()
            outer_converged, records = self._solve_frozen_friction_newton(verbose)
            outer_records.append(records)
            converged = converged and outer_converged
            if not outer_converged:
                if self.raise_on_nonconvergence:
                    tail = "; ".join(
                        f"r={entry['residual_norm']:.3e},"
                        f"cv={entry.get('correction_velocity', math.inf):.3e},"
                        f"a={entry.get('line_search_step', 0.0):.3e}"
                        for entry in records[-5:]
                    )
                    raise NewtonConvergenceError(
                        "FEM--AffineBody implicit solve did not converge at "
                        f"time {next_time:.6g}, friction iteration "
                        f"{outer + 1}/{outer_count}: "
                        f"residual={records[-1]['residual_norm']:.6e}, "
                        f"tolerance={records[-1]['convergence_tolerance']:.6e}, "
                        f"tail=[{tail}]"
                    )
                break
            if activate_friction:
                self.last_friction_iterations = outer + 1
                self.affine.initialize_contact_damping_device()
                self.contact.refresh_friction(self.fem.state.position)
                if self.fem_contact is not None:
                    self.fem_contact.refresh_friction_device(self.fem.state.position)
                self.last_friction_residual = self._updated_friction_residual()
                friction_residuals.append(self.last_friction_residual)
                if not math.isfinite(self.last_friction_residual):
                    raise NewtonConvergenceError("FEM--AffineBody lagged IPC friction residual is non-finite")
                if self.last_friction_residual <= friction_tolerance:
                    self.last_friction_converged = True
                    break
        if automatic and not self.last_friction_converged:
            tail = ", ".join(f"{value:.3e}" for value in friction_residuals[-8:])
            raise NewtonConvergenceError(
                "FEM--AffineBody lagged IPC friction fixed point did not converge "
                f"at time {next_time:.6g} within {outer_count} iterations "
                f"(residual={self.last_friction_residual:.6e}, "
                f"tolerance={friction_tolerance:.6e}, tail=[{tail}])"
            )
        return converged, outer_records

    def _step_once(self, verbose=False):
        with self.simulation.timer.section("Step setup"):
            self.fem.state.save_step_state()
            self.affine.device_backup_step_start()
        next_time = self.time + self.dt
        try:
            with self.simulation.timer.section("Prediction and contact setup"):
                if self.fem_contact is not None:
                    self.fem_contact.begin_step_device(self.fem.state.old_position, self.dt)
                self.fem.set_boundary_data_step(next_time, self.step_count + 1)
                self.fem.apply_boundary_step(self.dt, 0)
                self.fem.update_external_force_step(next_time, self.step_count + 1)
                self.fem.state.build_newmark_prediction(self.dt, self.fem.beta)
                self.affine.reset_semi_state()
                self.affine.device_begin_step(self.dt)
                self._apply_affine_velocity_boundaries()
                self.affine.initialize_contact_damping_device()
                self.contact.begin_step(
                    self.fem.state.position,
                    self.fem.state.old_position,
                    self.dt,
                )
            converged, outer_records = self.solve_lagged_friction_fixed_point(next_time, verbose)
            with self.simulation.timer.section("State update"):
                self.assemble_system(need_matrix=False)
                self._store_fem_equilibrium()
                self.fem.state.finalize_newmark(
                    self.dt,
                    self.fem.beta,
                    self.fem.gamma,
                    int(self.fem.quasi_static),
                )
                if self.record_history_step:
                    self.sample_energy_step()
                self.affine.device_accept_step(self.dt)
                self._update_affine_pressure_servos()
                self.affine.sync_output_state()
                self.advance_constitutive_state()
                self.minimum_jacobian = self.fem._minimum_jacobian_ratio_device(self.fem.state.position)
            self.time = next_time
            self.step_count += 1
            self.simulation.current_time = self.time
            self.simulation.current_step = self.step_count
            self.fem.time = self.time
            self.fem.step_count = self.step_count
            self.dem_wrapper.sims.current_time = self.time
            self.dem_wrapper.sims.current_step = self.step_count
            fem_record = {
                "step": self.step_count,
                "time": self.time,
                "minimum_jacobian": self.minimum_jacobian,
            }
            self.add_energy_record(fem_record)
            step_record = {
                "step": self.step_count,
                "time": self.time,
                "converged": converged,
                "newton": outer_records[-1],
                "friction_iterations": outer_records,
                "friction_iteration_count": self.last_friction_iterations,
                "friction_residual": self.last_friction_residual,
                "friction_converged": self.last_friction_converged,
                "contact": self._contact_diagnostics(),
                "assembly": self.assemble_type,
                "linear_solver": self.linear_solver,
            }
            self.last_step_record = step_record
            if self.record_history_step:
                self.step_schedule.append_history(self.fem.history, fem_record)
                self.step_schedule.append_history(self.history, step_record)
            return converged
        except BaseException as exception:
            try:
                self.affine.device_restore_step_start()
                self._restore_fem_step()
            except BaseException as rollback_error:
                raise exception from rollback_error
            raise

    def step(self, verbose=False, record_history=True):
        self.record_history_step = bool(record_history)
        original_timestep = self.dt
        attempt_timestep = original_timestep
        attempts = []
        for attempt in range(self.step_retry.maximum_retries + 1):
            self._set_timestep(attempt_timestep)
            try:
                result = self._step_once(verbose=verbose)
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
                        "FEM--AffineBody implicit step retry: "
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

        raise AssertionError("unreachable FEM--AffineBody retry state")

    def run(self, steps=None, verbose=True, postprocessing=()):
        target_time = float(self.simulation.time)
        target_step = None if steps is None else self.step_count + int(steps)
        overall = True

        def has_remaining_step():
            return self.time < target_time - 1.0e-14 if target_step is None else self.step_count < target_step

        def advance_one():
            nonlocal overall
            next_step = self.step_count + 1
            final_step = (
                next_step >= target_step if target_step is not None else self.time + self.dt >= target_time - 1.0e-14
            )
            record_history = self.step_schedule.history_due(next_step, final=final_step)
            overall = self.step(verbose=verbose, record_history=record_history) and overall
            runtime_checkpoint()
            with self.simulation.timer.section("Postprocess"):
                for callback in postprocessing:
                    callback(self)

        if self.compile_seconds is None and has_remaining_step():
            print("Compiling first ... ...")
            compile_start = time.perf_counter()
            next_step = self.step_count + 1
            final_step = (
                next_step >= target_step if target_step is not None else self.time + self.dt >= target_time - 1.0e-14
            )
            overall = (
                self.step(
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
        return {
            "converged": overall,
            "time": self.time,
            "step": self.step_count,
            "history": list(self.history),
        }


__all__ = ["FEMAffineIPCEngine"]
