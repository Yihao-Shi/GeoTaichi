"""Implicit engine responsibilities."""

import math
import time

import numpy as np
import taichi as ti

import src.igampm.config as config
from src.contact_detection.continuous_contact_detection.AdditiveCCD import point_nurbs_accd_increment
from src.physics_model.contact_model.ipc.NurbsContact import (
    get_distance_to_curve_moving_fixed_dim,
    get_distance_to_surface_moving_fixed_dim,
)
from src.mpm.engines.direct.Convergence import particle_correction_measure
from src.utils.FieldIO import field_to_numpy_prefix
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverRuntime import inexact_newton_relative_tolerance, normalize_callbacks
from src.utils.StepRetry import (
    is_recoverable_nonlinear_failure,
    nonlinear_failure_kind,
)


class ImplicitEngineMixin:
    @ti.kernel
    def _inspect_implicit_reference_state_device(self):
        """Reduce reference-state validity flags without a field download."""
        for index in ti.static(range(7)):
            self.implicit_state_status[index] = 0
        for node in self.iga.patch.volume:
            value = self.iga.patch.volume[node]
            if not (-ti.math.inf < value and value < ti.math.inf):
                self.implicit_state_status[0] = 1
            if value != 0.0:
                self.implicit_state_status[1] = 1
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
                    self.implicit_state_status[2] = 1
                if value != 0.0:
                    deformation_nonzero = 1
            if deformation_nonzero != 0:
                self.implicit_state_status[3] = 1
            determinant = self.mpm.F0[particle].determinant()
            if not (determinant > 0.0):
                self.implicit_state_status[6] = 1
        for node in self.mpm.grid:
            value = self.mpm.grid[node].m
            if not (-ti.math.inf < value and value < ti.math.inf):
                self.implicit_state_status[4] = 1
            if value > self.mpm.val_lim:
                self.implicit_state_status[5] = 1

    @ti.kernel
    def _snapshot_implicit_physical_state_device(self):
        for node in self.iga.patch.control_points:
            self.implicit_snapshot_iga_control_points[node] = self.iga.patch.control_points[node]
            self.implicit_snapshot_iga_velocity[node] = self.iga.patch.velocitys[node]
            self.implicit_snapshot_iga_acceleration[node] = self.iga.patch.accelerations[node]
        for particle in self.mpm.particle:
            self.implicit_snapshot_mpm_position[particle] = self.mpm.particle[particle].x
            self.implicit_snapshot_mpm_velocity[particle] = self.mpm.particle[particle].v
            self.implicit_snapshot_mpm_acceleration[particle] = self.mpm.particle[particle].a
            self.implicit_snapshot_mpm_deformation[particle] = self.mpm.F0[particle]
            if ti.static(self.mpm_has_plastic_history):
                self.implicit_snapshot_mpm_plastic_history[particle] = self.mpm.material.get_history_state(particle)
        for node in self.mpm.grid:
            self.implicit_snapshot_grid_mass[node] = self.mpm.grid[node].m
            self.implicit_snapshot_grid_velocity[node] = self.mpm.grid[node].v
            self.implicit_snapshot_grid_acceleration[node] = self.mpm.grid[node].a

    @ti.kernel
    def _restore_implicit_physical_state_device(self):
        for node in self.iga.patch.control_points:
            self.iga.patch.control_points[node] = self.implicit_snapshot_iga_control_points[node]
            self.iga.patch.velocitys[node] = self.implicit_snapshot_iga_velocity[node]
            self.iga.patch.accelerations[node] = self.implicit_snapshot_iga_acceleration[node]
        for particle in self.mpm.particle:
            self.mpm.particle[particle].x = self.implicit_snapshot_mpm_position[particle]
            self.mpm.particle[particle].v = self.implicit_snapshot_mpm_velocity[particle]
            self.mpm.particle[particle].a = self.implicit_snapshot_mpm_acceleration[particle]
            self.mpm.F0[particle] = self.implicit_snapshot_mpm_deformation[particle]
            if ti.static(self.mpm_has_plastic_history):
                self.mpm.material.set_history_state(
                    particle,
                    self.implicit_snapshot_mpm_plastic_history[particle],
                )
        for node in self.mpm.grid:
            self.mpm.grid[node].m = self.implicit_snapshot_grid_mass[node]
            self.mpm.grid[node].v = self.implicit_snapshot_grid_velocity[node]
            self.mpm.grid[node].a = self.implicit_snapshot_grid_acceleration[node]

    def _initialize_implicit_ipc_state(self):
        """Initialize reference data that must persist across coupled steps."""
        if self.implicit_initialized:
            return
        if self.monolithic_hash_matrix is None and self.monolithic_coo_matrix is None:
            raise RuntimeError("implicit IGA-MPM runtime requires a device COO or " "HashTriplet monolithic backend")

        self._inspect_implicit_reference_state_device()
        if int(self.implicit_state_status[0]) != 0:
            raise RuntimeError("IGA nodal volumes contain non-finite values")
        if int(self.implicit_state_status[1]) == 0:
            self.iga.patch.volume.fill(0.0)
            self.iga.precompute()
            self._inspect_implicit_reference_state_device()
            if int(self.implicit_state_status[0]) != 0 or int(self.implicit_state_status[1]) == 0:
                raise RuntimeError("IGA nodal-volume precomputation failed")

        particle_num = int(self.mpm.particleNum[0])
        if particle_num > 0:
            if int(self.implicit_state_status[2]) != 0:
                raise RuntimeError("MPM reference deformation gradients contain non-finite values")
            # A newly constructed direct implicit backend leaves F0 at zero;
            # never overwrite a valid (possibly already deformed) state.
            if int(self.implicit_state_status[3]) == 0:
                self.mpm.init_F0()
            elif int(self.implicit_state_status[6]) != 0:
                raise RuntimeError("MPM reference deformation gradients must have finite " "positive determinants")

        # TLMPM keeps its reference interpolation and nodal masses fixed.  Its
        # constructor computes those masses, but the compact active mapping and
        # inertia diagonal are normally initialized by initial_simulation().
        if self.total_lagrangian_mpm:
            if int(self.implicit_state_status[4]) != 0:
                raise RuntimeError("TLMPM nodal masses contain non-finite values")
            if particle_num > 0 and int(self.implicit_state_status[5]) == 0:
                self.mpm.compute_shapefn()
                self.mpm.mass_p2g()
                self._inspect_implicit_reference_state_device()
                if int(self.implicit_state_status[4]) != 0:
                    raise RuntimeError("TLMPM nodal-mass precomputation produced non-finite values")
            self.mpm.find_active_node()
            self.mpm.prefix_sum_executor.run(self.mpm.node2dof)
            self.mpm.active_dof = self.mpm.set_active_dof()
            self._validate_active_mpm_dofs()
            self.mpm.mass_vec.fill(0.0)
            self.compute_dynamic_mass_list()
        elif int(self.mpm.active_dof) > 0:
            # IGAMPM.build() performs the first UL active-map construction so
            # assembly-only callers also need the matching inertia diagonal.
            self._validate_active_mpm_dofs()
            self.mpm.mass_vec.fill(0.0)
            self.compute_dynamic_mass_list()

        self.implicit_initialized = True

    def begin_implicit_ipc_step(self):
        """Prepare one coupled solve as an all-or-nothing device transaction."""
        if self.implicit_step_in_progress:
            raise RuntimeError("an IGA-MPM implicit step is already in progress")
        self._snapshot_implicit_physical_state_device()
        self._save_device_monolithic_entry_displacements()
        try:
            return self._prepare_implicit_ipc_step()
        except BaseException:
            self._restore_implicit_physical_state_device()
            self._restore_device_monolithic_entry_displacements()
            self.implicit_step_in_progress = False
            raise

    def _prepare_implicit_ipc_step(self):
        self._initialize_implicit_ipc_state()

        if self.is_semi:
            self.semi_multiplier.fill(0.0)
            self.semi_constraint_violation[None] = 0.0

        self.iga.grid_disp.fill(0.0)
        self.iga.grid_disp_temp.fill(0.0)

        self.prepare_mpm_step()

        self.mpm.grid_disp.fill(0.0)
        self.mpm.grid_disp_temp.fill(0.0)
        self._implicit_step_iga_displacement = None
        self._implicit_step_mpm_displacement = None
        # Establish strict feasibility at the actual beginning-of-step state,
        # before any Newton iterate can mutate trial geometry.
        self.initialize_barrier(self.mpm.grid_disp, self.iga.grid_disp)
        self.implicit_step_in_progress = True
        return {
            "step": self.implicit_step_index,
            "active_mpm_dof": int(self.mpm.active_dof),
            "minimum_distance": self.minimum_contact_distance(),
        }

    def _prepare_total_lagrangian_mpm_step(self):
        # Total-Lagrangian shape functions, masses, and active mapping are
        # fixed; only nodal velocity/acceleration data is transferred.
        self.mpm.grid_reset()
        self.mpm.vel_acc_p2g()
        self.compute_mpm_traction()
        self.mpm.compute_nodal_vel_acc()

    def _prepare_updated_lagrangian_mpm_step(self):
        # Updated-Lagrangian particles change cells, so every part of the
        # active compact mapping must be rebuilt at every accepted step.
        self.mpm.mass_vec.fill(0.0)
        self.mpm.grid_reset()
        self.mpm.compute_shapefn()
        self.mpm.mass_vel_acc_p2g()
        self.mpm.find_active_node()
        self.mpm.prefix_sum_executor.run(self.mpm.node2dof)
        self.mpm.active_dof = self.mpm.set_active_dof()
        self._validate_active_mpm_dofs()
        self.compute_mpm_traction()
        self.mpm.compute_nodal_vel_acc()
        self.compute_dynamic_mass_list()

    def abort_implicit_ipc_step(self):
        """Discard a failed nonlinear solve without advancing physical state."""
        if not self.implicit_step_in_progress:
            return
        self._restore_implicit_physical_state_device()
        self._restore_device_monolithic_entry_displacements()
        self.implicit_step_in_progress = False

    def accept_implicit_ipc_step(self, return_increments=False):
        """Atomically advance both physical backends after a successful solve.

        Accepted increments remain in Taichi fields until the physical update
        consumes them.  Full NumPy snapshots are an explicit output request;
        the production substep path therefore performs no device-to-host array
        transfer here.
        """
        if not self.implicit_step_in_progress:
            raise RuntimeError("no IGA-MPM implicit step is in progress")
        self.initialize_barrier(self.mpm.grid_disp, self.iga.grid_disp)
        iga_increment = None
        mpm_increment = None
        if return_increments:
            # Explicit result/output boundary, never solver state.
            iga_increment = self.iga.grid_disp.to_numpy().copy()
            mpm_increment = self.mpm.grid_disp.to_numpy().copy()
        self._snapshot_implicit_physical_state_device()
        try:
            self.iga.iterative_dynamic_advance()
            self.mpm.update_nodal_acc(self.mpm.integration)
            self.mpm.advent_particles(self.mpm.coeffPIC)

            # The accepted increment is now represented in physical control
            # points and particles.  Canonicalize the next query to zero local
            # displacement so a later initialize_barrier() cannot apply it a
            # second time.
            self.iga.grid_disp.fill(0.0)
            self.iga.grid_disp_temp.fill(0.0)
            self.mpm.grid_disp.fill(0.0)
            self.mpm.grid_disp_temp.fill(0.0)
            self.initialize_barrier(self.mpm.grid_disp, self.iga.grid_disp)
        except BaseException:
            self._restore_implicit_physical_state_device()
            self._restore_device_monolithic_entry_displacements()
            self.implicit_step_in_progress = False
            raise

        self.implicit_step_in_progress = False
        self.implicit_step_index += 1
        accepted = {
            "step": self.implicit_step_index - 1,
            "minimum_distance": self.minimum_contact_distance(),
        }
        if return_increments:
            accepted["iga_increment"] = iga_increment
            accepted["mpm_increment"] = mpm_increment
        return accepted

    def _split_monolithic_correction(self, correction):
        correction = np.asarray(correction, dtype=np.float64).reshape(-1)
        iga_dof = int(self.iga.degree_of_freedom)
        active_mpm_dof = int(self.mpm.active_dof)
        expected = iga_dof + active_mpm_dof
        if correction.size != expected:
            raise ValueError(
                "coupled correction has size "
                f"{correction.size}, expected {expected} "
                "(IGA dofs plus active MPM dofs)"
            )
        if not np.all(np.isfinite(correction)):
            raise ValueError("coupled correction must contain only finite values")
        return correction

    def _load_external_correction(self, correction):
        """Upload one externally solved increment and prepare device views."""
        correction = self._split_monolithic_correction(correction)
        active_mpm_dof = int(self.mpm.active_dof)
        self._load_host_linear_solution(correction.size, correction)
        self._split_device_monolithic_correction(active_mpm_dof)
        return correction

    def _synchronize_trial_state_with_accepted(self):
        """Restore every trial-only field to the currently accepted iterate."""
        self._sync_device_trial_displacements()
        self.initialize_barrier(self.mpm.grid_disp, self.iga.grid_disp)

    def contact_aware_armijo(
        self,
        correction,
        directional_derivative,
        initial_step=1.0,
        include_friction=True,
        energy_function=None,
        c1=None,
        max_backtracks=None,
        verbose=False,
    ):
        """Upload an external correction and run the Taichi line search."""
        self._load_external_correction(correction)
        return self.contact_aware_armijo_device(
            directional_derivative,
            initial_step=initial_step,
            include_friction=include_friction,
            energy_function=energy_function,
            c1=c1,
            max_backtracks=max_backtracks,
            verbose=verbose,
        )

    def _device_monolithic_available(self, include_friction=False):
        """Whether the coupled exact block system can stay in Taichi."""
        if not hasattr(self.iga, "hash_matrix") or not hasattr(self.mpm, "hash_matrix"):
            return False
        assemble_type = getattr(self, "assemble_type", "HashTriplet")
        expected_solver = self.monolithic_solver_name
        expected_symmetric_storage = self.monolithic_matrix_symmetric
        if assemble_type == "HashTriplet":
            if self.monolithic_hash_matrix is None:
                return False
            if (
                self.monolithic_hash_matrix.solver != expected_solver
                or self.monolithic_hash_matrix.symmetric
                or bool(self.monolithic_hash_matrix.matrix_symmetric) != expected_symmetric_storage
                or bool(
                    getattr(
                        self.monolithic_hash_matrix,
                        "full_symmetric_input",
                        False,
                    )
                )
                != expected_symmetric_storage
            ):
                return False
        elif assemble_type == "COO":
            if self.monolithic_coo_matrix is None:
                return False
        else:
            return False
        if self.friction_mode not in ("lagged", "fully_implicit"):
            return False
        if self.friction_mode == "fully_implicit" and not self._fully_implicit_device_available():
            return False
        return True

    @ti.kernel
    def _prepare_device_monolithic_vectors(self, active_mpm_dof: ti.i32, include_friction: ti.template()):
        iga_dof = ti.static(self.iga.degree_of_freedom)
        active_total_dof = iga_dof + active_mpm_dof
        for dof in self.monolithic_rhs:
            value = 0.0
            if dof < iga_dof:
                value = self.iga.rhs[dof] + self.barrier_grad[dof]
                if ti.static(include_friction):
                    value += self.friction_grad[dof]
            elif dof < active_total_dof:
                local_dof = dof - iga_dof
                value = self.mpm.rhs[local_dof] + self.barrier_grad[dof]
                if ti.static(include_friction):
                    value += self.friction_grad[dof]
            self.monolithic_rhs[dof] = value
            self.monolithic_physical_rhs[dof] = value
            self.monolithic_correction[dof] = 0.0
            self.monolithic_fixed[dof] = 0
            self.monolithic_fixed_correction[dof] = 0.0

    @ti.func
    def _append_monolithic_coo_scalar(self, row, column, value):
        slot = ti.atomic_add(self.monolithic_coo_count[None], 1)
        if slot < self.monolithic_coo_matrix.data.shape[0]:
            self.monolithic_coo_matrix.rows[slot] = row
            self.monolithic_coo_matrix.cols[slot] = column
            self.monolithic_coo_matrix.data[slot] = value
        else:
            self.monolithic_coo_overflow[None] = 1

    @ti.kernel
    def _append_hash_source_to_monolithic_coo(
        self,
        source: ti.template(),
        active_nodes: ti.i32,
        block_offset: ti.i32,
    ):
        for block in range(active_nodes):
            for row_component, column_component in ti.static(ti.ndrange(config.DIM, config.DIM)):
                row = config.DIM * (block + block_offset) + row_component
                column = config.DIM * (block + block_offset) + column_component
                entry = row_component * config.DIM + column_component
                self._append_monolithic_coo_scalar(row, column, source.diag[block][entry])
        raw_count = source.raw_non_diag_count[0]
        for raw_index in range(raw_count):
            row_block = source.non_diag.blockI[raw_index]
            column_block = source.non_diag.blockJ[raw_index]
            if 0 <= row_block < active_nodes and 0 <= column_block < active_nodes:
                for row_component, column_component in ti.static(ti.ndrange(config.DIM, config.DIM)):
                    row = config.DIM * (row_block + block_offset) + row_component
                    column = config.DIM * (column_block + block_offset) + column_component
                    entry = row_component * config.DIM + column_component
                    self._append_monolithic_coo_scalar(
                        row,
                        column,
                        source.non_diag.blockH[raw_index][entry],
                    )

    @ti.kernel
    def _eliminate_device_monolithic_dirichlet_coo(self, active_dof: ti.i32):
        entry_count = self.monolithic_coo_count[None]
        for entry in range(entry_count):
            row = self.monolithic_coo_matrix.rows[entry]
            column = self.monolithic_coo_matrix.cols[entry]
            if 0 <= row < active_dof and 0 <= column < active_dof:
                value = self.monolithic_coo_matrix.data[entry]
                if self.monolithic_fixed[column] != 0:
                    ti.atomic_add(
                        self.monolithic_rhs[row],
                        -value * self.monolithic_fixed_correction[column],
                    )
                if self.monolithic_fixed[row] != 0 or self.monolithic_fixed[column] != 0:
                    self.monolithic_coo_matrix.data[entry] = 0.0
        for dof in range(active_dof):
            if self.monolithic_fixed[dof] != 0:
                self._append_monolithic_coo_scalar(dof, dof, 1.0)
                self.monolithic_rhs[dof] = self.monolithic_fixed_correction[dof]

    @ti.kernel
    def _build_device_monolithic_coo_diagonal(self, active_dof: ti.i32):
        for dof in self.monolithic_coo_diagonal:
            self.monolithic_coo_diagonal[dof] = 0.0
        entry_count = self.monolithic_coo_count[None]
        for entry in range(entry_count):
            row = self.monolithic_coo_matrix.rows[entry]
            column = self.monolithic_coo_matrix.cols[entry]
            if row == column and 0 <= row < active_dof:
                ti.atomic_add(
                    self.monolithic_coo_diagonal[row],
                    self.monolithic_coo_matrix.data[entry],
                )
        for dof in range(active_dof):
            if ti.abs(self.monolithic_coo_diagonal[dof]) < 1.0e-14:
                self.monolithic_coo_diagonal[dof] = 1.0

    @ti.kernel
    def _load_device_iga_dirichlet(self):
        for dof in range(self.iga.degree_of_freedom):
            if self.iga.dirichlet.node[dof] != 0:
                self.monolithic_fixed[dof] = 1
                self.monolithic_fixed_correction[dof] = self.iga.dirichlet.value[dof] - self.iga.grid_disp[dof]

    @ti.kernel
    def _load_device_mpm_dirichlet(self, active_mpm_nodes: ti.i32):
        iga_dof = ti.static(self.iga.degree_of_freedom)
        for block in range(active_mpm_nodes):
            grid_id = self.mpm.dof2node[block]
            for component in ti.static(range(config.DIM)):
                source_dof = config.DIM * grid_id + component
                if self.mpm.dirichlet.node[source_dof] != 0:
                    compact_dof = config.DIM * block + component
                    coupled_dof = iga_dof + compact_dof
                    self.monolithic_fixed[coupled_dof] = 1
                    self.monolithic_fixed_correction[coupled_dof] = (
                        self.mpm.dirichlet.value[source_dof] - self.mpm.grid_disp[compact_dof]
                    )

    @ti.kernel
    def _eliminate_device_monolithic_dirichlet(self, active_nodes: ti.i32):
        """Apply symmetric row/column elimination to unreduced block data."""
        for block in range(active_nodes):
            for row_component in ti.static(range(config.DIM)):
                row = config.DIM * block + row_component
                for column_component in ti.static(range(config.DIM)):
                    column = config.DIM * block + column_component
                    entry = row_component * config.DIM + column_component
                    value = self.monolithic_hash_matrix.diag[block][entry]
                    if self.monolithic_fixed[column] != 0:
                        ti.atomic_add(
                            self.monolithic_rhs[row],
                            -value * self.monolithic_fixed_correction[column],
                        )
                    if self.monolithic_fixed[row] != 0 or self.monolithic_fixed[column] != 0:
                        self.monolithic_hash_matrix.diag[block][entry] = 0.0

        raw_count = self.monolithic_hash_matrix.raw_non_diag_count[0]
        for raw_index in range(raw_count):
            row_block = self.monolithic_hash_matrix.non_diag.blockI[raw_index]
            column_block = self.monolithic_hash_matrix.non_diag.blockJ[raw_index]
            if 0 <= row_block < active_nodes and 0 <= column_block < active_nodes:
                for row_component in ti.static(range(config.DIM)):
                    row = config.DIM * row_block + row_component
                    for column_component in ti.static(range(config.DIM)):
                        column = config.DIM * column_block + column_component
                        entry = row_component * config.DIM + column_component
                        value = self.monolithic_hash_matrix.non_diag.blockH[raw_index][entry]
                        column_fixed = self.monolithic_fixed[column] != 0
                        row_fixed = self.monolithic_fixed[row] != 0
                        if column_fixed:
                            ti.atomic_add(
                                self.monolithic_rhs[row],
                                -value * self.monolithic_fixed_correction[column],
                            )
                        if ti.static(self.monolithic_hash_matrix.matrix_symmetric):
                            # The lagged matrix stores only this upper block;
                            # apply the transposed mirror when the row DOF is
                            # constrained so elimination and PCG see the same
                            # structural operator.
                            if row_fixed:
                                ti.atomic_add(
                                    self.monolithic_rhs[column],
                                    -value * self.monolithic_fixed_correction[row],
                                )
                        if row_fixed or column_fixed:
                            self.monolithic_hash_matrix.non_diag.blockH[raw_index][entry] = 0.0

    @ti.kernel
    def _finish_device_monolithic_dirichlet(self, active_dof: ti.i32):
        active_nodes = active_dof // config.DIM
        for block in range(active_nodes):
            # One thread owns the whole dense diagonal block.  Component-wise
            # vector-field writes from separate DOF threads can otherwise
            # overwrite another fixed component on device.
            diagonal = self.monolithic_hash_matrix.diag[block]
            for component in ti.static(range(config.DIM)):
                dof = config.DIM * block + component
                if self.monolithic_fixed[dof] != 0:
                    diagonal[component * config.DIM + component] = 1.0
                    self.monolithic_rhs[dof] = self.monolithic_fixed_correction[dof]
            self.monolithic_hash_matrix.diag[block] = diagonal

    @ti.kernel
    def _split_device_monolithic_correction(self, active_mpm_dof: ti.i32):
        iga_dof = ti.static(self.iga.degree_of_freedom)
        for dof in range(iga_dof + active_mpm_dof):
            if self.monolithic_fixed[dof] != 0:
                self.monolithic_correction[dof] = self.monolithic_fixed_correction[dof]
        for dof in self.iga.incre_resolution:
            self.iga.incre_resolution[dof] = self.monolithic_correction[dof]
        for dof in self.mpm.incre_resolution:
            value = 0.0
            if dof < active_mpm_dof:
                value = self.monolithic_correction[iga_dof + dof]
            self.mpm.incre_resolution[dof] = value

    @ti.kernel
    def _load_host_linear_solution(
        self,
        active_dof: ti.i32,
        solution: ti.types.ndarray(),
    ):
        """Upload only the explicitly selected host linear-solve result."""
        for dof in self.monolithic_correction:
            value = 0.0
            if dof < active_dof:
                value = solution[dof]
            self.monolithic_correction[dof] = value

    @ti.kernel
    def _monolithic_correction_inf_norm(self, active_dof: ti.i32) -> ti.f64:
        result = 0.0
        for dof in range(active_dof):
            ti.atomic_max(result, ti.abs(self.monolithic_correction[dof]))
        return result

    @ti.kernel
    def _device_monolithic_free_rhs_norm(self, active_dof: ti.i32) -> ti.f64:
        squared = 0.0
        for dof in range(active_dof):
            if self.monolithic_fixed[dof] == 0:
                ti.atomic_add(squared, self.monolithic_rhs[dof] * self.monolithic_rhs[dof])
        return ti.sqrt(ti.max(squared, 0.0))

    def _solve_monolithic_linear_system(self, system, linear_solve=None):
        """Solve one assembled device system.

        ``linear_solve`` is the sole allowed production numerical host
        boundary.  Assembly is finalized before conversion, and only the
        active sparse matrix, right-hand side, and solved increment cross it.
        """
        assemble_type = getattr(self, "assemble_type", "HashTriplet")
        if linear_solve is None:
            relative_tolerance = system.get(
                "linear_relative_tolerance", self.monolithic_linear_solver_relative_tolerance
            )
            if assemble_type == "COO":
                self.monolithic_correction.fill(0.0)
                converged = self.monolithic_coo_matrix.solve(
                    self.monolithic_rhs,
                    self.monolithic_correction,
                    self.monolithic_coo_diagonal,
                    tol=self.monolithic_linear_solver_tolerance,
                    rel_tol=relative_tolerance,
                    maxiter=self.monolithic_linear_solver_max_iters,
                )
                return {
                    "converged": bool(converged),
                    "residual": float(self.monolithic_coo_matrix.linear_solver.last_residual),
                    "iterations": int(self.monolithic_coo_matrix.linear_solver.last_iterations),
                    "solution_inf_norm": float(self._monolithic_correction_inf_norm(int(system["active_dof"]))),
                    "backend": f"taichi_coo_{self.monolithic_solver_name.lower()}",
                }
            return self.monolithic_hash_matrix.solve_flat_system(
                self.monolithic_rhs,
                self.monolithic_correction,
                active_nodes=system["active_nodes"],
                tol=self.monolithic_linear_solver_tolerance,
                rel_tol=relative_tolerance,
                maxiter=self.monolithic_linear_solver_max_iters,
                return_solution=False,
            )

        active_dof = int(system["active_dof"])
        if assemble_type == "COO":
            matrix = self.monolithic_coo_matrix._to_scipy().tocsr()
            matrix = matrix[:active_dof, :active_dof]
        else:
            matrix = self.monolithic_hash_matrix.to_scipy(int(system["active_nodes"])).tocsr()
        rhs = field_to_numpy_prefix(self.monolithic_rhs, active_dof)
        solution = np.asarray(linear_solve(matrix, rhs), dtype=np.float64).reshape(-1)
        if solution.size != active_dof:
            raise ValueError("selected host linear solver returned " f"{solution.size} values, expected {active_dof}")
        if not np.all(np.isfinite(solution)):
            raise RuntimeError("selected host linear solver returned a non-finite increment")
        self._load_host_linear_solution(active_dof, solution)
        return {
            "converged": True,
            "residual": 0.0,
            "iterations": -1,
            "solution_inf_norm": float(self._monolithic_correction_inf_norm(active_dof)),
            "backend": "explicit_host_linear_solve",
        }

    @ti.kernel
    def _set_device_trial_displacements(self, alpha: ti.f64):
        for dof in self.iga.grid_disp_temp:
            self.iga.grid_disp_temp[dof] = self.iga.grid_disp[dof] + alpha * self.iga.incre_resolution[dof]
        for dof in self.mpm.grid_disp_temp:
            self.mpm.grid_disp_temp[dof] = self.mpm.grid_disp[dof] + alpha * self.mpm.incre_resolution[dof]

    @ti.kernel
    def _accept_device_trial_displacements(self):
        for dof in self.iga.grid_disp:
            self.iga.grid_disp[dof] = self.iga.grid_disp_temp[dof]
        for dof in self.mpm.grid_disp:
            self.mpm.grid_disp[dof] = self.mpm.grid_disp_temp[dof]

    @ti.kernel
    def _sync_device_trial_displacements(self):
        for dof in self.iga.grid_disp:
            self.iga.grid_disp_temp[dof] = self.iga.grid_disp[dof]
        for dof in self.mpm.grid_disp:
            self.mpm.grid_disp_temp[dof] = self.mpm.grid_disp[dof]

    @ti.kernel
    def _save_device_monolithic_entry_displacements(self):
        for dof in self.iga.grid_disp:
            self.monolithic_entry_iga_displacement[dof] = self.iga.grid_disp[dof]
        for dof in self.mpm.grid_disp:
            self.monolithic_entry_mpm_displacement[dof] = self.mpm.grid_disp[dof]

    @ti.kernel
    def _restore_device_monolithic_entry_displacements(self):
        for dof in self.iga.grid_disp:
            value = self.monolithic_entry_iga_displacement[dof]
            self.iga.grid_disp[dof] = value
            self.iga.grid_disp_temp[dof] = value
        for dof in self.mpm.grid_disp:
            value = self.monolithic_entry_mpm_displacement[dof]
            self.mpm.grid_disp[dof] = value
            self.mpm.grid_disp_temp[dof] = value

    @ti.kernel
    def _set_device_current_from_trial_base(self, alpha: ti.f64):
        """Set current state to ``q_base + alpha p`` for residual probes."""
        for dof in self.iga.grid_disp:
            self.iga.grid_disp[dof] = self.iga.grid_disp_temp[dof] + alpha * self.iga.incre_resolution[dof]
        for dof in self.mpm.grid_disp:
            self.mpm.grid_disp[dof] = self.mpm.grid_disp_temp[dof] + alpha * self.mpm.incre_resolution[dof]

    @ti.kernel
    def _restore_device_current_from_trial_base(self):
        for dof in self.iga.grid_disp:
            self.iga.grid_disp[dof] = self.iga.grid_disp_temp[dof]
        for dof in self.mpm.grid_disp:
            self.mpm.grid_disp[dof] = self.mpm.grid_disp_temp[dof]

    @ti.kernel
    def _device_monolithic_correction_residual(self, active_mpm_dof: ti.i32, iga_dt: ti.f64, mpm_dt: ti.f64) -> ti.f64:
        result = 0.0
        iga_dof = ti.static(self.iga.degree_of_freedom)
        for dof in range(iga_dof):
            ti.atomic_max(
                result,
                ti.abs(self.monolithic_correction[dof]) / iga_dt,
            )
        # Measure the represented motion and strain, not almost-empty grid modes.
        for particle_id in range(self.mpm.particleNum[0]):
            measure = particle_correction_measure(
                self.mpm, self.monolithic_correction, particle_id, active_mpm_dof, iga_dof, config.DIM, config.DIM
            )
            ti.atomic_max(result, measure / mpm_dt)
        return result

    @ti.kernel
    def _device_monolithic_directional_derivative(self, active_dof: ti.i32) -> ti.f64:
        result = 0.0
        for dof in range(active_dof):
            result -= self.monolithic_physical_rhs[dof] * self.monolithic_correction[dof]
        return result

    @ti.kernel
    def _device_dirichlet_residual(self, active_dof: ti.i32) -> ti.f64:
        result = 0.0
        for dof in range(active_dof):
            if self.monolithic_fixed[dof] != 0:
                ti.atomic_max(
                    result,
                    ti.abs(self.monolithic_fixed_correction[dof]),
                )
        return result

    @ti.func
    def _contact_point_direction(self, particle_id):
        direction = ti.Vector.zero(ti.f64, config.DIM)
        for local_node in range(self.mpm.offset[particle_id]):
            grid_id = self.mpm.LnID[particle_id, local_node]
            compact_node = self.mpm.node2dof[grid_id] - 1
            if compact_node >= 0:
                shape = self.mpm.shape[particle_id, local_node]
                for component in ti.static(range(config.DIM)):
                    direction[component] += shape * self.mpm.incre_resolution[config.DIM * compact_node + component]
        return direction

    @ti.func
    def _point_nurbs_motion_bound(self, point_direction, start_control, end_control):
        bound = 0.0
        for local_control_id in range(start_control, end_control):
            bound = ti.max(
                bound,
                (point_direction - self.contact_surface.control_point_direction[local_control_id]).norm(),
            )
        return bound

    @ti.kernel
    def _prepare_point_nurbs_accd(self, max_step: ti.f64):
        self.contact_step_alpha[None] = max_step
        self.contact_query_status[None] = 0
        for sample in range(self.mpm.total_surface_num):
            direction = self._contact_point_direction(self.mpm.surface_id[sample])
            self.contact_point_direction[sample] = direction
            for component in ti.static(range(config.DIM)):
                if not ti.abs(direction[component]) < ti.math.inf:
                    ti.atomic_max(self.contact_query_status[None], 2)
        for local_control_id in range(self.contact_surface.total_ctrlpts):
            weight = self.contact_surface.weights[local_control_id]
            if not (weight > 0.0 and weight < ti.math.inf):
                ti.atomic_max(self.contact_query_status[None], 1)
            for component in ti.static(range(config.DIM)):
                if not ti.abs(self.contact_surface.control_point_direction[local_control_id][component]) < ti.math.inf:
                    ti.atomic_max(self.contact_query_status[None], 2)

    @ti.kernel
    def _screen_point_nurbs_accd(
        self,
        group_begin: ti.i32,
        group_end: ti.i32,
        max_step: ti.f64,
        safety: ti.f64,
        clearance: ti.f64,
        surface: ti.template(),
    ):
        self.contact_accd_active_count[None] = 0
        for sample_id in range(self.mpm.total_surface_num):
            point = self.mpm.p_temp[sample_id]
            point_direction = self.contact_point_direction[sample_id]
            end_point = point + max_step * point_direction
            point_lower, point_upper = ti.min(point, end_point), ti.max(point, end_point)
            node = 0
            while node < surface.surface_tree_count:
                left, _, escape, surface_id = surface.surface_tree_nodes[node]
                overlap = surface.swept_box_overlap(
                    point_lower, point_upper, surface.swept_tree_lower[node], surface.swept_tree_upper[node], clearance
                )
                if not overlap:
                    node = escape
                elif surface_id < 0:
                    node = left
                else:
                    node = escape
                    order = surface.surface_group_order[surface_id]
                    if group_begin <= order < group_end:
                        if ti.static(config.DIM == 3):
                            overlap = surface.swept_span_overlap(surface_id, point_lower, point_upper, clearance)
                        if overlap:
                            contact_id = sample_id * surface.num_surfaces + surface_id
                            contact = self.contacts[contact_id]
                            distance = contact.distance
                            if (
                                contact.surface_id == surface_id
                                and 0 <= contact.particle_id < self.mpm.particleNum[0]
                                and clearance < distance < ti.math.inf
                            ):
                                motion_bound = surface.relative_motion_upper_bound(surface_id, point_direction)
                                if max_step * motion_bound > safety * (distance - clearance):
                                    motion_bound = self._point_nurbs_motion_bound(
                                        point_direction,
                                        surface.prefix_num_ctrlpts_field[surface_id],
                                        surface.prefix_num_ctrlpts_field[surface_id + 1],
                                    )
                                if motion_bound > 0.0 and max_step * motion_bound > safety * (distance - clearance):
                                    slot = ti.atomic_add(self.contact_accd_active_count[None], 1)
                                    self.contact_accd_candidates[slot] = contact_id
                                    self.contact_accd_toc[contact_id] = 0.0
                                    self.contact_accd_distance[contact_id] = distance
                                    self.contact_accd_motion_bound[contact_id] = motion_bound
                                    self.contact_accd_active[contact_id] = 1
                            else:
                                ti.atomic_max(self.contact_query_status[None], 2)

    @ti.kernel
    def _advance_point_nurbs_accd(
        self,
        group_begin: ti.i32,
        group_end: ti.i32,
        candidate_count: ti.i32,
        max_step: ti.f64,
        safety: ti.f64,
        clearance: ti.f64,
        minimum_step: ti.f64,
        surface: ti.template(),
        basis: ti.template(),
    ):
        """One ACCD iteration; keep the moving NURBS query outside a nested ACCD loop.

        Taichi 1.7's CFG optimizer becomes prohibitively slow when that loop
        encloses the already nested closest-point search. Only the active count
        crosses to the host; pair state and all geometry remain on the device.
        """
        self.contact_accd_active_count[None] = 0
        for candidate in range(candidate_count):
            contact_id = self.contact_accd_candidates[candidate]
            surface_id = contact_id % surface.num_surfaces
            sample_id = contact_id // surface.num_surfaces
            if self.contact_accd_active[contact_id] != 0:
                pair_toc = self.contact_accd_toc[contact_id]
                remaining = max_step - pair_toc
                increment = ti.min(
                    remaining,
                    point_nurbs_accd_increment(
                        self.contact_accd_distance[contact_id],
                        self.contact_accd_motion_bound[contact_id],
                        safety,
                        clearance,
                    ),
                )
                active = 0
                if not (increment < minimum_step and remaining > minimum_step):
                    trial_toc = pair_toc + increment
                    trial_point = self.mpm.p_temp[sample_id] + trial_toc * self.contact_point_direction[sample_id]
                    start_u = surface.prefix_num_knot_u_field[surface_id]
                    num_u = surface.prefix_num_knot_u_field[surface_id + 1] - start_u
                    start_control = surface.prefix_num_ctrlpts_field[surface_id]
                    trial_distance = 0.0
                    if ti.static(config.DIM == 2):
                        _, trial_distance, _ = get_distance_to_curve_moving_fixed_dim(
                            start_u,
                            start_control,
                            num_u,
                            surface.knot_vector_u,
                            surface.control_points_hat,
                            surface.control_point_direction,
                            trial_toc,
                            surface.weights,
                            trial_point,
                            basis,
                        )
                    else:
                        start_v = surface.prefix_num_knot_v_field[surface_id]
                        num_v = surface.prefix_num_knot_v_field[surface_id + 1] - start_v
                        _, _, trial_distance, _ = get_distance_to_surface_moving_fixed_dim(
                            start_u,
                            start_v,
                            start_control,
                            num_u,
                            num_v,
                            surface.knot_vector_u,
                            surface.knot_vector_v,
                            surface.control_points_hat,
                            surface.control_point_direction,
                            trial_toc,
                            surface.weights,
                            trial_point,
                            basis,
                        )
                    target_excess = (1.0 - safety) * (self.contacts[contact_id].distance - clearance)
                    target_crossed = pair_toc > 0.0 and trial_distance - clearance < target_excess
                    if clearance < trial_distance < ti.math.inf and not target_crossed:
                        self.contact_accd_toc[contact_id] = trial_toc
                        self.contact_accd_distance[contact_id] = trial_distance
                        active = ti.cast(trial_toc < max_step, ti.i32)
                self.contact_accd_active[contact_id] = active
                ti.atomic_add(self.contact_accd_active_count[None], active)

    @ti.kernel
    def _finish_point_nurbs_accd(self, candidate_count: ti.i32):
        # Reduce only final TOCs: intermediate iterates are lower bounds that
        # would incorrectly keep the global step at the first accepted iterate.
        for candidate in range(candidate_count):
            contact_id = self.contact_accd_candidates[candidate]
            ti.atomic_min(self.contact_step_alpha[None], self.contact_accd_toc[contact_id])

    def assemble_monolithic_newton_system(
        self, include_friction=None, need_matrix=True, *, prepare_contacts=True, residual_prepared=False
    ):
        """Assemble the coupled residual/tangent in Taichi device fields."""
        if include_friction is None:
            include_friction = self.activate_fric
        include_friction = bool(include_friction)
        need_matrix = bool(need_matrix)
        if not self._device_monolithic_available(include_friction):
            raise RuntimeError(
                "Taichi monolithic assembly requires a device COO or "
                "HashTriplet backend and a supported friction law"
            )

        self._initialize_implicit_ipc_state()
        if prepare_contacts:
            self.initialize_barrier(self.mpm.grid_disp, self.iga.grid_disp)
        self.assemble_barrier_system(need_matrix=need_matrix)
        if include_friction:
            self.assemble_friction_system(need_matrix=need_matrix)

        # Body forces and tangents remain in each subsystem's native Taichi
        # fields. Body sources append raw blocks; contact sources are bucket
        # reduced before appending their unique blocks to the same destination.
        direct_iga = need_matrix and self.assemble_type == "HashTriplet"
        if direct_iga:
            self.monolithic_hash_matrix.reset_system()
        reuse_iga_residual = residual_prepared
        if not reuse_iga_residual:
            self.iga.rhs.fill(0.0)
        if need_matrix:
            self.iga.incre_resolution.fill(0.0)
            if not direct_iga:
                self.iga.hash_matrix.reset_system()
        self.iga.assemble_body_matrix(
            need_matrix=need_matrix,
            project_spd=self.project_lagged_hessians,
            need_force=not reuse_iga_residual,
            matrix=self.monolithic_hash_matrix if direct_iga else self.iga.hash_matrix,
            fixed_slots=self.iga_fixed_slots if direct_iga else None,
        )
        if self.iga.neumann.num > 0 and not reuse_iga_residual:
            self.iga.apply_neumann()

        active_mpm_dof = int(self.mpm.active_dof)
        if active_mpm_dof < 0 or active_mpm_dof % config.DIM != 0:
            raise RuntimeError(
                "active MPM degrees of freedom must be a non-negative " "multiple of the spatial dimension"
            )
        prepared_material = hasattr(self.mpm, "prepare_material_response")
        if need_matrix:
            self.mpm.incre_resolution.fill(0.0)
            self.mpm.hash_matrix.reset_system()
        if not residual_prepared:
            self._clear_mpm_body_rhs()
            self.mpm.assemble_inertia_force(
                active_mpm_dof, self.mpm.damping, self.mpm.gravity, self.mpm.integration, self.mpm.grid_disp
            )
            if prepared_material:
                self.mpm.prepare_material_response(self.mpm.grid_disp)
                self.mpm.assemble_material_force(active_mpm_dof, self.mpm.grid_disp, reuse_response=True)
            else:
                self.mpm.assemble_material_force(active_mpm_dof, self.mpm.grid_disp)
            self.apply_mpm_neumann()
        if need_matrix:
            material_options = {"reuse_response": True} if prepared_material else {}
            if self.nonassociated_newton:
                material_options["exact_plastic_tangent"] = True
            self.mpm.assemble_stiffness_matrix_hash(
                active_mpm_dof,
                self.mpm.grid_disp,
                project_spd=self.project_lagged_hessians,
                **material_options,
            )
            self.assemble_mpm_mass_matrix()

        iga_nodes = int(self.iga.degree_of_freedom) // config.DIM
        active_mpm_nodes = active_mpm_dof // config.DIM
        active_nodes = iga_nodes + active_mpm_nodes
        active_dof = config.DIM * active_nodes
        matrix = None
        if need_matrix:
            if self.assemble_type == "HashTriplet":
                matrix = self.monolithic_hash_matrix
                matrix.append_raw_from(
                    self.mpm.hash_matrix,
                    active_nodes=active_mpm_nodes,
                    block_offset=iga_nodes,
                )
                matrix.append_reduced_from(
                    self.barrier_hash_matrix,
                    active_nodes=active_nodes,
                    block_offset=0,
                )
                if include_friction:
                    matrix.append_reduced_from(
                        self.friction_hash_matrix,
                        active_nodes=active_nodes,
                        block_offset=0,
                    )
                if matrix.full_symmetric_input:
                    matrix.canonicalize_full_symmetric_input()
            else:
                matrix = self.monolithic_coo_matrix
                matrix.reset()
                self.monolithic_coo_count[None] = 0
                self.monolithic_coo_overflow[None] = 0
                self._append_hash_source_to_monolithic_coo(self.iga.hash_matrix, iga_nodes, 0)
                self._append_hash_source_to_monolithic_coo(self.mpm.hash_matrix, active_mpm_nodes, iga_nodes)
                self._append_hash_source_to_monolithic_coo(self.barrier_hash_matrix, active_nodes, 0)
                if include_friction:
                    self._append_hash_source_to_monolithic_coo(self.friction_hash_matrix, active_nodes, 0)

        self._prepare_device_monolithic_vectors(active_mpm_dof, include_friction)
        # The default boundary objects intentionally have no Taichi fields;
        # launch these loaders only for finalized, non-empty constraints.
        if self.iga.dirichlet.num > 0:
            self._load_device_iga_dirichlet()
        if self.mpm.dirichlet.num > 0:
            self._load_device_mpm_dirichlet(active_mpm_nodes)
        if need_matrix and (self.iga.dirichlet.num > 0 or self.mpm.dirichlet.num > 0):
            if self.assemble_type == "HashTriplet":
                self._eliminate_device_monolithic_dirichlet(active_nodes)
                matrix.eliminate_fixed_constraints(
                    self.monolithic_fixed, self.monolithic_fixed_correction, self.monolithic_rhs
                )
                self._finish_device_monolithic_dirichlet(active_dof)
            else:
                self._eliminate_device_monolithic_dirichlet_coo(active_dof)
        elif not need_matrix:
            self._finish_device_residual_constraints(active_dof)

        if need_matrix:
            if self.assemble_type == "HashTriplet":
                matrix.finalize_taichi_assembly()
            else:
                if int(self.monolithic_coo_overflow[None]) != 0:
                    raise RuntimeError(
                        "IGA-MPM COO Hessian capacity exceeded; increase the "
                        "assembly capacity by rebuilding the coupled engine"
                    )
                active_nnz = int(self.monolithic_coo_count[None])
                matrix.linear_operator.update_active_dofs(active_dof)
                matrix.linear_operator.update_nnz(active_nnz)
                self._build_device_monolithic_coo_diagonal(active_dof)
        return {
            "matrix": matrix,
            "rhs": self.monolithic_rhs,
            "unconstrained_rhs": self.monolithic_physical_rhs,
            "correction": self.monolithic_correction,
            "active_nodes": active_nodes,
            "active_dof": active_dof,
            "active_mpm_dof": active_mpm_dof,
            "need_matrix": need_matrix,
            "assemble_type": self.assemble_type,
            "backend": (
                f"taichi_device_{self.assemble_type.lower()}_nonassociated_bicgstab"
                if self.nonassociated_newton
                else (
                    "taichi_device_coo_exact_fi_bicgstab"
                    if self.assemble_type == "COO" and not self.monolithic_matrix_symmetric
                    else (
                        "taichi_device_coo_lagged_pcg"
                        if self.assemble_type == "COO"
                        else (
                            "taichi_device_hashtriplet_exact_fi_bicgstab"
                            if not self.monolithic_matrix_symmetric
                            else "taichi_device_hashtriplet_lagged_pcg"
                        )
                    )
                )
            ),
        }

    def _material_feasible_step_device(self, max_step=1.0):
        alpha = float(max_step)
        alpha = min(alpha, float(self.iga_material_ccd()))
        alpha = min(alpha, float(self.mpm_material_ccd()))
        return max(0.0, alpha)

    def _synchronize_device_trial_state_with_accepted(self):
        self._sync_device_trial_displacements()
        self.initialize_barrier(self.mpm.grid_disp, self.iga.grid_disp)

    def conservative_contact_step_device(self, max_step=1.0, safety=None, verify=True):
        """Run independent point--NURBS ACCD loops on the Taichi device."""
        return self._conservative_contact_step_device_impl(
            max_step,
            safety,
            verify,
            prepare_contacts=True,
        )

    def _conservative_contact_step_device_impl(
        self,
        max_step,
        safety,
        verify,
        *,
        prepare_contacts,
    ):
        """ACCD core with an explicit prepared-contact ownership boundary.

        The Armijo caller has just evaluated the current contact potential and
        therefore owns a valid base-state query.  Other callers request the
        preparation here so the public method remains self-contained.
        """
        max_step = float(max_step)
        safety = self.contact_ccd_safety if safety is None else float(safety)
        if not math.isfinite(max_step) or max_step < 0.0:
            raise ValueError("max_step must be finite and non-negative")
        if not 0.0 < safety < 1.0:
            raise ValueError("contact CCD safety must lie strictly between 0 and 1")
        if max_step == 0.0:
            self.last_contact_ccd_step = 0.0
            return 0.0

        if prepare_contacts:
            self.initialize_barrier(self.mpm.grid_disp, self.iga.grid_disp)
        self.last_contact_ccd_min_distance = self.minimum_contact_distance()
        clearance = 0.0 if self.is_semi else float(self.barrier.minimum_distance) + self.strict_feasibility_tolerance
        self.contact_surface.update_control_point_direction(self.iga.incre_resolution)
        self._prepare_point_nurbs_accd(max_step)
        query_status = int(self.contact_query_status[None])
        if query_status == 1:
            raise RuntimeError(
                "conservative point-NURBS contact stepping requires finite " "strictly positive NURBS weights"
            )
        if query_status != 0:
            raise RuntimeError("point-NURBS CCD requires finite motion directions")
        self.contact_surface.update_swept_bounds(max_step)

        for basis_group, basis in enumerate(self.contact_surface.accd_basis):
            group_begin = int(self.contact_surface.accd_basis_group_offsets[basis_group])
            group_end = int(self.contact_surface.accd_basis_group_offsets[basis_group + 1])
            self._screen_point_nurbs_accd(
                group_begin,
                group_end,
                max_step,
                safety,
                clearance,
                self.contact_surface,
            )
            candidate_count = int(self.contact_accd_active_count[None])
            for _ in range(self.contact_ccd_max_iterations):
                if self.contact_accd_active_count[None] == 0:
                    break
                self._advance_point_nurbs_accd(
                    group_begin,
                    group_end,
                    candidate_count,
                    max_step,
                    safety,
                    clearance,
                    self.contact_ccd_min_step,
                    self.contact_surface,
                    basis,
                )
            self._finish_point_nurbs_accd(candidate_count)

        query_status = int(self.contact_query_status[None])
        if query_status != 0:
            raise RuntimeError("point-NURBS ACCD found an invalid contact pair, distance, " "or initial clearance")
        alpha = float(self.contact_step_alpha[None])
        if not math.isfinite(alpha) or alpha < 0.0 or alpha > max_step:
            raise RuntimeError("point-NURBS ACCD produced an invalid step")

        if verify and alpha > 0.0:
            self._set_device_trial_displacements(alpha)
            try:
                self.initialize_barrier(
                    self.mpm.grid_disp_temp,
                    self.iga.grid_disp_temp,
                )
                if self.minimum_contact_distance() <= clearance:
                    raise RuntimeError("point-NURBS ACCD accepted a non-feasible trial state")
            finally:
                self._synchronize_device_trial_state_with_accepted()

        self.last_contact_ccd_step = float(alpha)
        return float(alpha)

    def contact_aware_armijo_device(
        self,
        directional_derivative,
        initial_step=1.0,
        include_friction=True,
        energy_function=None,
        c1=None,
        max_backtracks=None,
        verbose=False,
        *,
        prepare_contacts=True,
    ):
        """Armijo search that consumes the correction from Taichi fields."""
        directional_derivative = float(directional_derivative)
        if not math.isfinite(directional_derivative):
            raise ValueError("directional_derivative must be finite")
        if directional_derivative >= 0.0:
            raise RuntimeError("contact-aware Armijo requires a strict descent direction")
        c1 = self.armijo_c1 if c1 is None else float(c1)
        max_backtracks = self.armijo_max_backtracks if max_backtracks is None else int(max_backtracks)
        if not 0.0 < c1 < 1.0:
            raise ValueError("Armijo c1 must lie strictly between 0 and 1")
        if max_backtracks <= 0:
            raise ValueError("max_backtracks must be positive")

        accepted = False
        backtracks = 0
        alpha = 0.0
        accepted_energy = math.inf
        previous_energy = math.inf
        try:
            self._set_device_trial_displacements(0.0)
            if prepare_contacts:
                self.initialize_barrier(self.mpm.grid_disp_temp, self.iga.grid_disp_temp)
            if energy_function is None:
                previous_energy = self.coupled_potential_energy(
                    self.iga.grid_disp_temp,
                    self.mpm.grid_disp_temp,
                    include_friction=include_friction,
                )
            else:
                previous_energy = float(energy_function(self))
            if not math.isfinite(previous_energy):
                raise RuntimeError("current coupled potential energy is not finite")
            energy_tolerance = self.line_search_energy_atol + self.line_search_energy_rtol * max(
                1.0, abs(previous_energy)
            )

            alpha = self._conservative_contact_step_device_impl(
                max_step=initial_step,
                safety=self.contact_ccd_safety,
                # The base contact query was built immediately above.  ACCD
                # itself is conservative, so the expensive full-scene trial
                # verification remains an explicit diagnostic option rather
                # than part of every production Armijo iteration.
                verify=False,
                prepare_contacts=False,
            )
            accepted_energy = previous_energy
            for backtracks in range(max_backtracks):
                if alpha < self.contact_ccd_min_step:
                    break
                self._set_device_trial_displacements(alpha)
                try:
                    self.initialize_barrier(
                        self.mpm.grid_disp_temp,
                        self.iga.grid_disp_temp,
                    )
                    if energy_function is None:
                        trial_energy = self.coupled_potential_energy(
                            self.iga.grid_disp_temp,
                            self.mpm.grid_disp_temp,
                            include_friction=include_friction,
                        )
                    else:
                        trial_energy = float(energy_function(self))
                except (FloatingPointError, RuntimeError, ValueError):
                    trial_energy = math.inf

                bound = previous_energy + c1 * alpha * directional_derivative + energy_tolerance
                if math.isfinite(trial_energy) and trial_energy <= bound:
                    self._accept_device_trial_displacements()
                    accepted_energy = float(trial_energy)
                    accepted = True
                    break
                alpha *= 0.5
                if verbose:
                    print(
                        "IGA-MPM Taichi contact Armijo backtrack "
                        f"{backtracks + 1}: alpha={alpha:.6e}, "
                        f"energy={trial_energy:.12e}"
                    )
        finally:
            if accepted:
                # The successful trial already owns this accepted geometry's
                # contact table and span hulls; only trial vectors need sync.
                self._sync_device_trial_displacements()
            else:
                self._synchronize_device_trial_state_with_accepted()

        self.last_armijo_step = float(alpha if accepted else 0.0)
        self.last_armijo_backtracks = int(backtracks)
        return {
            "accepted": accepted,
            "step": self.last_armijo_step,
            "backtracks": self.last_armijo_backtracks,
            "previous_energy": float(previous_energy),
            "energy": float(accepted_energy),
            "minimum_distance": self.minimum_contact_distance(),
            "backend": "taichi_device",
        }

    def _newton_linear_tolerance(self, force_residual, previous_force):
        return inexact_newton_relative_tolerance(
            force_residual, previous_force, self.monolithic_linear_solver_relative_tolerance
        )

    def _solve_monolithic_newton_device(
        self,
        *,
        include_friction,
        max_iterations,
        tolerance,
        energy_function,
        verbose,
        linear_solve=None,
    ):
        self.last_monolithic_iterations = 0
        self.last_monolithic_residual = math.inf
        self.last_monolithic_force_residual = math.inf
        self.last_monolithic_converged = False
        last_system = None
        last_armijo = None
        initial_force_residual = None
        previous_force_residual = None
        inexact = bool(getattr(self, "monolithic_inexact_newton", False))
        force_tolerance = math.inf
        convergence_reason = None
        has_plastic_history = bool(getattr(self, "mpm_has_plastic_history", False))
        has_incremental_potential = has_plastic_history and bool(
            getattr(getattr(self.mpm, "material", None), "has_incremental_potential", False)
        )
        if self.nonassociated_newton and energy_function is not None:
            raise TypeError("nonassociated DP uses residual-norm Armijo, not an energy_function")
        semi_progress = 0.0
        contacts_prepared = False
        for iteration in range(max_iterations):
            if (
                self.is_semi
                and not self.nonassociated_newton
                and not inexact
                and iteration > 1
                and semi_progress > 0.999
            ):
                self.last_monolithic_converged = True
                convergence_reason = "semi_ipc_projection_progress"
                break
            last_system = self.assemble_monolithic_newton_system(
                include_friction=include_friction, need_matrix=False, prepare_contacts=not contacts_prepared
            )
            active_dof = int(last_system["active_dof"])
            force_residual = float(self._device_monolithic_free_rhs_norm(active_dof))
            dirichlet_residual = float(self._device_dirichlet_residual(active_dof))
            # Prescribed motion can increase potential through boundary work.
            # Once imposed data are reached, use the material's energy merit.
            use_residual_merit = has_plastic_history and (
                not has_incremental_potential or dirichlet_residual > self.monolithic_dirichlet_tolerance
            )
            self.last_monolithic_force_residual = force_residual
            self.last_monolithic_dirichlet_residual = dirichlet_residual
            if initial_force_residual is None:
                initial_force_residual = max(force_residual, 1.0)
                if getattr(self, "_monolithic_force_reference", None) is None:
                    self._monolithic_force_reference = initial_force_residual
                initial_force_residual = self._monolithic_force_reference
            force_tolerance = self.monolithic_force_atol + self.monolithic_force_rtol * initial_force_residual
            if (
                force_residual <= force_tolerance
                and (not self.nonassociated_newton or (force_residual == 0.0 and dirichlet_residual == 0.0))
                and dirichlet_residual <= self.monolithic_dirichlet_tolerance
                and self.semi_contact_converged()
            ):
                self.last_monolithic_residual = 0.0
                self.last_monolithic_converged = True
                convergence_reason = "force_residual"
                break
            last_system = self.assemble_monolithic_newton_system(
                include_friction=include_friction,
                need_matrix=True,
                prepare_contacts=False,
                residual_prepared=True,
            )
            if inexact:
                last_system["linear_relative_tolerance"] = self._newton_linear_tolerance(
                    force_residual, previous_force_residual
                )
            solve_result = self._solve_monolithic_linear_system(last_system, linear_solve=linear_solve)
            previous_force_residual = force_residual
            if not solve_result["converged"]:
                raise RuntimeError(
                    f"IGA-MPM Taichi {self.monolithic_solver_name} did not "
                    "converge: "
                    f"residual={solve_result['residual']:.6e}, "
                    f"initial={solve_result.get('initial_residual', math.nan):.6e}, "
                    f"target={solve_result.get('convergence_tolerance', math.nan):.6e}, "
                    f"iterations={solve_result['iterations']}"
                )
            self._split_device_monolithic_correction(last_system["active_mpm_dof"])
            residual = float(
                self._device_monolithic_correction_residual(
                    last_system["active_mpm_dof"],
                    float(self.iga.dt),
                    float(self.mpm.dt),
                )
            )
            self.last_monolithic_residual = residual
            last_system["linear_solve"] = solve_result
            if not math.isfinite(residual):
                raise RuntimeError(f"IGA-MPM Taichi {self.monolithic_solver_name} returned " "a non-finite correction")
            if (
                inexact
                and residual <= tolerance
                and force_residual <= force_tolerance
                and last_system["linear_relative_tolerance"] > self.monolithic_linear_solver_relative_tolerance
            ):
                # Verify a terminal correction with the configured Krylov accuracy.
                last_system["linear_relative_tolerance"] = self.monolithic_linear_solver_relative_tolerance
                solve_result = self._solve_monolithic_linear_system(last_system, linear_solve=linear_solve)
                if not solve_result["converged"]:
                    raise RuntimeError("IGA-MPM terminal Newton linear solve did not converge")
                self._split_device_monolithic_correction(last_system["active_mpm_dof"])
                residual = float(
                    self._device_monolithic_correction_residual(
                        last_system["active_mpm_dof"], float(self.iga.dt), float(self.mpm.dt)
                    )
                )
                self.last_monolithic_residual = residual
                last_system["linear_solve"] = solve_result
                if not math.isfinite(residual):
                    raise RuntimeError("IGA-MPM terminal Newton correction is non-finite")
            if (
                residual <= tolerance
                and (not self.nonassociated_newton or force_residual <= force_tolerance)
                and dirichlet_residual <= self.monolithic_dirichlet_tolerance
                and self.semi_contact_converged()
            ):
                self.last_monolithic_converged = True
                convergence_reason = "force_and_correction" if self.nonassociated_newton else "correction_velocity"
                break

            initial_step = self._material_feasible_step_device()
            if use_residual_merit:
                merit_slope = self._assemble_device_physical_tangent_product(
                    active_mpm_dof=last_system["active_mpm_dof"],
                    include_friction=include_friction,
                )
                last_armijo = self.fully_implicit_residual_armijo_device(
                    force_residual,
                    merit_slope,
                    initial_step=initial_step,
                    include_friction=include_friction,
                    verbose=verbose,
                    prepare_contacts=linear_solve is not None,
                )
            else:
                directional_derivative = float(
                    self._device_monolithic_directional_derivative(last_system["active_dof"])
                )
                last_armijo = self.contact_aware_armijo_device(
                    directional_derivative,
                    initial_step=initial_step,
                    include_friction=include_friction,
                    energy_function=energy_function,
                    verbose=verbose,
                    prepare_contacts=linear_solve is not None or energy_function is not None,
                )
            self.last_monolithic_iterations = iteration + 1
            if not last_armijo["accepted"]:
                break
            # Successful Armijo trials own the accepted geometry's query.
            # Semi-IPC changes activation through its multipliers; external
            # callbacks retain the self-contained preparation path.
            contacts_prepared = not self.is_semi and linear_solve is None and energy_function is None
            if self.is_semi:
                self._update_semi_multipliers(int(self.mpm.total_surface_num) * int(self.contact_surface.num_surfaces))
                semi_progress += (1.0 - semi_progress) * float(last_armijo["step"])
            if verbose:
                measure = f"residual={force_residual:.6e}" if use_residual_merit else f"correction={residual:.6e}"
                print(
                    "IGA-MPM Taichi monolithic Newton " f"{iteration + 1}: {measure}, alpha={last_armijo['step']:.6e}"
                )

        return {
            "iterations": self.last_monolithic_iterations,
            "residual": self.last_monolithic_residual,
            "force_residual": self.last_monolithic_force_residual,
            "force_tolerance": force_tolerance,
            "convergence_reason": convergence_reason,
            "converged": self.last_monolithic_converged,
            "system": last_system,
            "line_search": last_armijo,
            "backend": (
                "taichi_device_nonassociated_bicgstab" if self.nonassociated_newton else "taichi_device_lagged_pcg"
            ),
        }

    def solve_monolithic_newton(
        self,
        outer_iteration=0,
        include_friction=None,
        max_iterations=None,
        tolerance=None,
        linear_solve=None,
        energy_function=None,
        verbose=False,
    ):
        """Solve coupled equilibrium with lagged friction frozen."""
        if outer_iteration == 0:
            self._monolithic_force_reference = None
        if include_friction is None:
            include_friction = self.activate_fric
        if bool(include_friction) and self.friction_mode == "fully_implicit":
            raise RuntimeError(
                "solve_monolithic_newton() is the conservative/lagged-friction "
                "solver and cannot solve fully implicit friction; use "
                "solve_fully_implicit_friction_newton() instead"
            )
        max_iterations = self.monolithic_max_iterations if max_iterations is None else int(max_iterations)
        tolerance = self.monolithic_tolerance if tolerance is None else float(tolerance)
        if max_iterations <= 0:
            raise ValueError("monolithic Newton max_iterations must be positive")
        if tolerance < 0.0:
            raise ValueError("monolithic Newton tolerance must be non-negative")
        if self._device_monolithic_available(include_friction):
            return self._solve_monolithic_newton_device(
                include_friction=bool(include_friction),
                max_iterations=max_iterations,
                tolerance=tolerance,
                energy_function=energy_function,
                verbose=verbose,
                linear_solve=linear_solve,
            )
        raise RuntimeError(
            "monolithic IGA-MPM requires a COO or HashTriplet device assembly; "
            "a SciPy/custom solver changes only the finalized linear-system "
            "boundary and never enables host assembly"
        )

    @staticmethod
    def _require_converged_implicit_result(result, label):
        if not bool(result.get("converged", False)):
            raise RuntimeError(f"{label} did not converge " f"(residual={float(result.get('residual', math.inf)):.6e})")
        return result

    def _solve_configured_conservative_system(
        self,
        *,
        linear_solve,
        energy_function,
        newton_max_iterations,
        newton_tolerance,
        verbose,
    ):
        result = self.solve_monolithic_newton(
            include_friction=False,
            max_iterations=newton_max_iterations,
            tolerance=newton_tolerance,
            linear_solve=linear_solve,
            energy_function=energy_function,
            verbose=verbose,
        )
        return self._require_converged_implicit_result(result, "IGA-MPM monolithic Newton solve")

    def _solve_configured_lagged_friction_system(
        self,
        *,
        linear_solve,
        energy_function,
        newton_max_iterations,
        newton_tolerance,
        verbose,
    ):
        return self.solve_lagged_friction_fixed_point(
            None,
            None,
            grid_disp=self.mpm.grid_disp,
            probe_solve=None,
            include_friction=True,
            linear_solve=linear_solve,
            energy_function=energy_function,
            newton_max_iterations=newton_max_iterations,
            newton_tolerance=newton_tolerance,
            verbose=verbose,
        )

    def _solve_configured_fully_implicit_system(
        self,
        *,
        linear_solve,
        energy_function,
        newton_max_iterations,
        newton_tolerance,
        verbose,
    ):
        if energy_function is not None:
            raise TypeError("fully implicit friction uses residual-norm Armijo, not a " "conservative energy_function")
        result = self.solve_fully_implicit_friction_newton(
            include_friction=True,
            max_iterations=newton_max_iterations,
            tolerance=newton_tolerance,
            linear_solve=linear_solve,
            verbose=verbose,
        )
        return self._require_converged_implicit_result(result, "IGA-MPM fully implicit friction Newton solve")

    def _solve_lagged_material_fixed_point(self, solve_inner):
        if getattr(self.mpm, "has_lagged_material", False) is not True:
            return solve_inner()

        self.mpm.begin_lagged_material_state()
        result = None
        for _ in range(self.mpm.material_lagged_max_iterations):
            result = solve_inner()
            if self.mpm.refresh_lagged_material_state(self.mpm.grid_disp) <= self.mpm.material_lagged_tolerance:
                result = dict(result)
                result["material_lagged_iterations"] = self.mpm.last_material_lagged_iterations
                result["material_lagged_error"] = self.mpm.last_material_lagged_error
                return result
        raise RuntimeError(
            "IGA-MPM lagged material state did not converge: "
            f"error={self.mpm.last_material_lagged_error:.6e} after "
            f"{self.mpm.material_lagged_max_iterations} iterations"
        )

    def _implicit_ipc_substep_once(
        self,
        inner_solve=None,
        assemble_updated_system=None,
        grid_disp=None,
        probe_solve=None,
        include_friction=None,
        linear_solve=None,
        energy_function=None,
        newton_max_iterations=None,
        newton_tolerance=None,
        return_increments=False,
        verbose=False,
    ):
        """Execute one transactional prepare/solve/accept IPC time step."""
        if (inner_solve is None) != (assemble_updated_system is None):
            raise TypeError(
                "inner_solve and assemble_updated_system must either both be " "provided or both be omitted"
            )
        use_configured_driver = (
            include_friction is None and inner_solve is None and assemble_updated_system is None and probe_solve is None
        )
        if include_friction is None:
            include_friction = self.activate_fric
        include_friction = bool(include_friction and self.activate_fric)
        if grid_disp is not None and grid_disp is not self.mpm.grid_disp:
            raise ValueError("a solved IGA-MPM step must use the engine's accepted " "mpm.grid_disp field")

        self.begin_implicit_ipc_step()
        try:
            if use_configured_driver:
                result = self._solve_lagged_material_fixed_point(
                    lambda: self.solve_configured_implicit_system(
                        linear_solve=linear_solve,
                        energy_function=energy_function,
                        newton_max_iterations=newton_max_iterations,
                        newton_tolerance=newton_tolerance,
                        verbose=verbose,
                    )
                )
            elif include_friction and self.friction_mode == "fully_implicit":
                if inner_solve is not None or assemble_updated_system is not None:
                    raise TypeError(
                        "fully implicit IGA-MPM friction owns the coupled "
                        "residual/Jacobian solve and does not accept lagged "
                        "inner-solve callbacks"
                    )
                if energy_function is not None:
                    raise TypeError(
                        "fully implicit friction uses residual-norm Armijo, " "not a conservative energy_function"
                    )
                result = self._solve_lagged_material_fixed_point(
                    lambda: self._solve_configured_fully_implicit_system(
                        linear_solve=linear_solve,
                        energy_function=energy_function,
                        newton_max_iterations=newton_max_iterations,
                        newton_tolerance=newton_tolerance,
                        verbose=verbose,
                    )
                )
            elif inner_solve is not None or include_friction:
                result = self._solve_lagged_material_fixed_point(
                    lambda: self.solve_lagged_friction_fixed_point(
                        inner_solve,
                        assemble_updated_system,
                        grid_disp=self.mpm.grid_disp,
                        probe_solve=probe_solve,
                        include_friction=include_friction,
                        linear_solve=linear_solve,
                        energy_function=energy_function,
                        newton_max_iterations=newton_max_iterations,
                        newton_tolerance=newton_tolerance,
                        verbose=verbose,
                    )
                )
            else:
                result = self._solve_lagged_material_fixed_point(
                    lambda: self._solve_configured_conservative_system(
                        linear_solve=linear_solve,
                        energy_function=energy_function,
                        newton_max_iterations=newton_max_iterations,
                        newton_tolerance=newton_tolerance,
                        verbose=verbose,
                    )
                )

            accepted_state = self.accept_implicit_ipc_step(return_increments=return_increments)
        except BaseException:
            self.abort_implicit_ipc_step()
            raise

        result = dict(result)
        result.update(
            {
                "accepted": True,
                "step": accepted_state["step"],
                "minimum_distance": accepted_state["minimum_distance"],
            }
        )
        if return_increments:
            result["iga_increment"] = accepted_state["iga_increment"]
            result["mpm_increment"] = accepted_state["mpm_increment"]
        return result

    def _set_implicit_timestep(self, timestep):
        timestep = float(timestep)
        self.iga.dt = timestep
        self.mpm.dt = timestep

    def _implicit_failure_diagnostics(self, exception, attempt, timestep):
        return {
            "kind": nonlinear_failure_kind(exception),
            "exception": type(exception).__name__,
            "message": str(exception),
            "attempt": int(attempt),
            "timestep": float(timestep),
            "time": float(self.time),
            "step": int(self.implicit_step_index),
            "contact": {
                "model": self.barrier.model,
                "ccd_step": float(self.last_contact_ccd_step),
                "minimum_distance": float(self.last_contact_ccd_min_distance),
                "constraint_violation": float(self.semi_constraint_violation[None]) if self.is_semi else 0.0,
            },
        }

    @staticmethod
    def _add_implicit_energy_record(step_record, result):
        step_record["potential_energy"] = float(result["energy"])

    def diagnostics_snapshot(self):
        return {
            "schema_version": 1,
            "subsystem": "igampm_implicit_ipc",
            "time": float(self.time),
            "step": int(self.implicit_step_index),
            "timestep": float(self.iga.dt),
            "contact": {
                "model": self.barrier.model,
                "ccd_step": float(self.last_contact_ccd_step),
                "minimum_distance": float(self.last_contact_ccd_min_distance),
                "constraint_violation": float(self.semi_constraint_violation[None]) if self.is_semi else 0.0,
            },
            "linear_solver": {
                "name": self.monolithic_solver_name,
                "iterations": int(self.last_monolithic_iterations),
                "residual": float(self.last_monolithic_residual),
                "force_residual": float(self.last_monolithic_force_residual),
                "converged": bool(self.last_monolithic_converged),
            },
            "last_failure": self.last_failure,
            "last_step": self.last_step_record,
        }

    def implicit_ipc_substep(
        self,
        inner_solve=None,
        assemble_updated_system=None,
        grid_disp=None,
        probe_solve=None,
        include_friction=None,
        linear_solve=None,
        energy_function=None,
        newton_max_iterations=None,
        newton_tolerance=None,
        return_increments=False,
        verbose=False,
        record_history=True,
    ):
        """Run one bounded, transactional implicit IPC step."""

        step_retry = self.step_retry
        original_timestep = float(self.iga.dt)
        attempt_timestep = original_timestep
        attempts = []
        for attempt in range(step_retry.maximum_retries + 1):
            self._set_implicit_timestep(attempt_timestep)
            try:
                result = self._implicit_ipc_substep_once(
                    inner_solve=inner_solve,
                    assemble_updated_system=assemble_updated_system,
                    grid_disp=grid_disp,
                    probe_solve=probe_solve,
                    include_friction=include_friction,
                    linear_solve=linear_solve,
                    energy_function=energy_function,
                    newton_max_iterations=newton_max_iterations,
                    newton_tolerance=newton_tolerance,
                    return_increments=return_increments,
                    verbose=verbose,
                )
            except RuntimeError as exception:
                if not is_recoverable_nonlinear_failure(exception):
                    raise
                failure = self._implicit_failure_diagnostics(exception, attempt, attempt_timestep)
                attempts.append(failure)
                next_timestep = step_retry.next_timestep(attempt_timestep, attempt)
                if next_timestep is None:
                    self.last_failure = {
                        **failure,
                        "original_timestep": original_timestep,
                        "attempts": attempts,
                    }
                    # The physical transaction has already been restored by
                    # ``abort_implicit_ipc_step``.  Trial increments are not
                    # physical state and must remain canonical zero after an
                    # exhausted solve, just as after an accepted step.
                    self.iga.grid_disp.fill(0.0)
                    self.iga.grid_disp_temp.fill(0.0)
                    self.mpm.grid_disp.fill(0.0)
                    self.mpm.grid_disp_temp.fill(0.0)
                    self._set_implicit_timestep(original_timestep)
                    raise
                attempt_timestep = next_timestep
                continue

            retry_record = {
                "enabled": bool(step_retry.enabled),
                "original_timestep": original_timestep,
                "accepted_timestep": float(attempt_timestep),
                "retry_count": int(attempt),
                "attempts": attempts,
            }
            result["step_retry"] = retry_record
            self.time += float(attempt_timestep)
            step_record = {
                "step": int(self.implicit_step_index),
                "time": float(self.time),
                "step_retry": retry_record,
                "minimum_distance": result.get("minimum_distance"),
                "converged": bool(result.get("converged", True)),
            }
            self.add_implicit_energy_record(step_record, result)
            self.last_step_record = step_record
            if record_history:
                self.step_schedule.append_history(self.history, step_record)
            self.last_failure = None
            return result

        raise AssertionError("unreachable IGA-MPM retry state")

    def run_implicit_ipc_contact(
        self,
        steps=None,
        inner_solve=None,
        assemble_updated_system=None,
        grid_disp=None,
        probe_solve=None,
        include_friction=None,
        linear_solve=None,
        energy_function=None,
        newton_max_iterations=None,
        newton_tolerance=None,
        return_increments=False,
        verbose=False,
        postprocessing=(),
    ):
        """Run the requested number of complete coupled implicit time steps."""
        if steps is None:
            steps = max(
                1,
                min(
                    int(getattr(self.iga, "total_step", 1)),
                    int(getattr(self.mpm, "total_step", 1)),
                ),
            )
        try:
            step_count = int(steps)
            exact_integer = float(steps) == step_count
        except (TypeError, ValueError, OverflowError):
            step_count = -1
            exact_integer = False
        if not exact_integer or step_count < 0:
            raise ValueError("IGA-MPM implicit steps must be a non-negative integer")

        result = None
        callbacks = normalize_callbacks(postprocessing)
        for local_step in range(step_count):
            next_step = self.implicit_step_index + 1
            output_due = next_step % self.output_interval == 0
            record_history = self.step_schedule.history_due(
                next_step,
                output=output_due,
                final=local_step + 1 == step_count,
            )
            compiling = self.compile_seconds is None
            if compiling:
                print("Compiling first ... ...")
                compile_start = time.perf_counter()
            with self.timer.section("IGAMPM implicit step"):
                result = self.implicit_ipc_substep(
                    inner_solve=inner_solve,
                    assemble_updated_system=assemble_updated_system,
                    grid_disp=grid_disp,
                    probe_solve=probe_solve,
                    include_friction=include_friction,
                    linear_solve=linear_solve,
                    energy_function=energy_function,
                    newton_max_iterations=newton_max_iterations,
                    newton_tolerance=newton_tolerance,
                    return_increments=return_increments,
                    verbose=verbose,
                    record_history=record_history,
                )
            if compiling:
                ti.sync()
                self.compile_seconds = time.perf_counter() - compile_start
                print(f"Compiling time = {self.compile_seconds} \n")
                self.timer.profile1()
            with self.timer.section("Postprocess"):
                for callback in callbacks:
                    callback(self)
            runtime_checkpoint()
        if result is None:
            return {"accepted": False, "completed_steps": 0}
        result["completed_steps"] = step_count
        return result
