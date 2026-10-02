import taichi as ti
import numpy as np
import time
from taichi.lang.impl import current_cfg

import src.iga.config as config
from src.contact_detection.continuous_contact_detection import (
    deformation_gradient_ccd,
)
from src.iga.engines.IGASolver import IGASolver
from src.iga.engines.EngineUtils import (
    copy_group_field,
    jacobian2parent2parametric1d,
    jacobian2parent2parametric2d,
    linearize,
    matrix_cols,
    vectorize_id,
)
from src.linear_solver.BuildTriplet import BuildTriplet, solve_csr_system
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix
from src.physics_model.contact_model.ipc.ContactAssembly import psd_project_nd
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverRuntime import normalize_callbacks
from src.utils.StepRetry import (
    StepRetryPolicy,
    is_recoverable_nonlinear_failure,
    nonlinear_failure_kind,
)
from src.utils.linalg import no_operation


class IGAConvergenceError(RuntimeError):
    """A recoverable failure before an implicit IGA step is accepted."""


@ti.data_oriented
class ImplicitIGA(IGASolver):
    def __init__(self, primitives, dirichlet=None, neumann=None, **kwargs):
        super().__init__(primitives, dirichlet, neumann, **kwargs)
        self.integration = kwargs.get("newmark", [0.5, 0.25, 0.5])  # [alpha, beta, gamma]
        integration = np.asarray(self.integration, dtype=np.float64)
        if integration.size < 3 or not np.all(np.isfinite(integration[:3])) or np.any(integration[:3] <= 0.0):
            raise ValueError("newmark must contain finite positive [alpha, beta, gamma]")
        if not np.isfinite(float(self.dt)) or float(self.dt) <= 0.0:
            raise ValueError("implicit IGA time step must be finite and positive")
        self.max_iters = int(kwargs.get("max_iters", 100))
        self.tol = float(kwargs.get("residual", 1e-2))
        self.assemble_type = self._normalize_assemble_type(kwargs.get("assemble_type", kwargs.get("assembly", "Hash")))
        # The IPC projected-Newton route clamps each local elastic
        # tangent to PSD before scattering it.  Together with the positive
        # Newmark mass term and Dirichlet identities this makes PCG the device
        # production solver.  Exact (unprojected) tangents remain available
        # for fully implicit/nonsymmetric coupled solves through BiCGSTAB.
        default_linear_solver = "PCG"
        self.linear_solver = self._normalize_linear_solver(kwargs.get("linear_solver", default_linear_solver))
        self.project_hessian_to_psd = bool(
            kwargs.get(
                "project_hessian_to_psd",
                self.linear_solver == "PCG",
            )
        )
        if self.linear_solver == "PCG" and not self.project_hessian_to_psd:
            raise ValueError(
                "ImplicitIGA PCG requires project_hessian_to_psd=True; "
                "use BiCGSTAB for the exact, potentially indefinite tangent"
            )
        self.linear_solver_tolerance = float(kwargs.get("linear_solver_tolerance", 1.0e-10))
        if self.max_iters <= 0:
            raise ValueError("implicit IGA max_iters must be positive")
        if not np.isfinite(self.tol) or self.tol < 0.0:
            raise ValueError("implicit IGA residual/correction tolerance must be finite " "and non-negative")
        if not np.isfinite(self.linear_solver_tolerance) or self.linear_solver_tolerance < 0.0:
            raise ValueError("implicit IGA linear_solver_tolerance must be finite and " "non-negative")
        self.step_retry = StepRetryPolicy(
            enabled=kwargs.get("enable_step_retry", False),
            maximum_retries=kwargs.get("step_retry_max_retries", 2),
            reduction=kwargs.get("step_retry_reduction", 0.5),
            minimum_timestep=kwargs.get("step_retry_minimum_timestep", 0.0),
        )
        self.time = 0.0
        self.step_count = 0
        self.history = []
        self.last_step_record = None
        self.record_history_step = True
        self.last_failure = None
        self.device_solve = (self.assemble_type == "Hash" and self.linear_solver in ("PCG", "BiCGSTAB")) or (
            self.assemble_type == "COO" and self.linear_solver == "PCG"
        )
        self.solve_newton_iteration = (
            self._solve_device_newton_iteration if self.device_solve else self._solve_host_newton_iteration
        )

        self.stiffness_nnz = 0
        # Keep capacity arithmetic in int64/Python integers.  Backend
        # constructors issue a clear error before a Taichi int32 field can be
        # requested; silently wrapping this prefix used to hide an oversized
        # COO/Hash allocation.
        self.prefix_nnz = np.zeros(self.patch.primitive.num_primitives + 1, dtype=np.int64)
        self.influence_range = self.element.total_knot_range * config.DIM
        for i in range(self.patch.num_patch):
            nnz = self.patch.total_num_element[i + 1] * self.influence_range * self.influence_range
            self.prefix_nnz[i + 1] = self.prefix_nnz[i] + nnz
            self.stiffness_nnz += nnz
        self.total_nnz = int(self.stiffness_nnz + self.degree_of_freedom)
        # Hash assembly stores one dense block per
        # (patch, element, Gauss point, local-node pair).  Reserving those
        # slots explicitly makes the raw coordinate stream deterministic
        # across Newton iterations, which lets the device pattern cache reuse
        # its raw-slot mapping. ``stiffness_nnz`` counts scalar entries, hence
        # the division by the number of scalars in one dense block.
        raw_hash_pairs = max(
            1,
            self.stiffness_nnz // (config.DIM * config.DIM) * self.element.gauss_number,
        )
        # Gauss points repeat the same element/control-point block topology.
        # They increase raw deterministic scatter slots, but cannot increase
        # the number of distinct reduced block coordinates.
        reduced_hash_pairs = max(1, self.stiffness_nnz // (config.DIM * config.DIM))
        self.hash_matrix = BuildTriplet(
            dim=config.DIM,
            max_pairs_num=raw_hash_pairs,
            max_nonzeros=reduced_hash_pairs,
            max_active_nodes=max(1, int(self.degree_of_freedom / config.DIM)),
            symmetric=False,
            solver=("PCG" if self.linear_solver == "PCG" else "BiCGSTAB"),
            device_reduction=True,
        )
        # Keep the native body tangent complete: IGA-MPM fully implicit
        # friction reuses both physical row orientations when evaluating its
        # exact, generally nonsymmetric Jacobian product.  A standalone PCG
        # solve lazily copies that source into a canonical upper-triangle
        # matrix instead, making symmetry structural rather than relying on
        # two independently accumulated floating-point triangles.
        self.pcg_hash_matrix = None
        self._pcg_hash_prepared = False
        self.coo_matrix = None
        self.diag_A = None
        self.linear_solver_max_iters = int(kwargs.get("linear_solver_max_iters", max(100, 10 * self.degree_of_freedom)))
        if self.assemble_type == "COO":
            self.coo_matrix = CoordinateSparseMatrix(
                self.total_nnz,
                self.degree_of_freedom,
                preconditioned=True,
                symmetry=True,
                linear_solver=True,
            )
            self.diag_A = ti.field(ti.f64, shape=self.degree_of_freedom)

        self.energy = ti.field(ti.f64, shape=(), needs_grad=True)  # total energy
        self.rhs = ti.field(ti.f64, shape=self.degree_of_freedom)  # global residual force
        self.mass_vec = ti.field(ti.f64, shape=self.degree_of_freedom)  # global mass list
        self.incre_resolution = ti.field(ti.f64, shape=self.degree_of_freedom)  # global displacement increment
        self.grid_disp = ti.field(
            ti.f64, shape=self.degree_of_freedom, needs_grad=True
        )  # global displacement in a substep
        self.grid_disp_temp = ti.field(ti.f64, shape=self.degree_of_freedom)  # global displacement per iteration
        self.apply_dirichlet_step = no_operation
        if self.dirichlet.num > 0:
            self.apply_dirichlet_step = (
                self.apply_dirichlet_coo if self.assemble_type == "COO" else self.apply_dirichlet_hash
            )
        self.apply_neumann_step = self.apply_neumann if self.neumann.num > 0 else no_operation
        self.get_neumann_energy_step = self.get_neumann_energy if self.neumann.num > 0 else no_operation

    @staticmethod
    def _normalize_assemble_type(assemble_type):
        assemble_type = str(assemble_type).replace("_", "").replace("-", "").lower()
        if assemble_type in ("hash", "hashtriplet", "triplet", "buildtriplet"):
            return "Hash"
        if assemble_type in ("coo", "coordinatesparse", "coordinatesparsematrix"):
            return "COO"
        raise ValueError(f"Unsupported IGA assemble_type: {assemble_type}")

    @staticmethod
    def _normalize_linear_solver(linear_solver):
        linear_solver = str(linear_solver).replace("_", "").replace("-", "").lower()
        if linear_solver in ("scipy", "spsolve", "cpu"):
            return "Scipy"
        if linear_solver in ("pcg", "taichipcg", "matrixfreepcg"):
            return "PCG"
        if linear_solver in ("bicgstab", "taichibicgstab", "bicg"):
            return "BiCGSTAB"
        raise ValueError(f"Unsupported IGA linear_solver: {linear_solver}")

    @ti.func
    def _axisymmetric_reference_radius(self, N, rest_ctrl_coords):
        radius = 1.0
        if ti.static(self.is_axisymmetric):
            radius = self.element.interpolate_component(N, rest_ctrl_coords, 0) - ti.static(self.axis_offset)
            radius = self.require_positive_reference_radius(radius)
        return radius

    @ti.func
    def _physical_quadrature_weight(self, base_weight, reference_radius):
        weight = base_weight
        if ti.static(self.is_axisymmetric):
            weight *= 2.0 * ti.math.pi * reference_radius
        return weight

    @ti.func
    def _constitutive_deformation_gradient(
        self,
        N,
        shape_gradients,
        rest_ctrl_coords,
        current_ctrl_coords,
    ):
        deformation_gradient = ti.Matrix.identity(ti.f64, self.material_dimension)
        if ti.static(self.is_axisymmetric):
            deformation_gradient = self.element.compute_axisymmetric_deformation_gradient(
                N,
                shape_gradients,
                rest_ctrl_coords,
                current_ctrl_coords,
                ti.static(self.axis_offset),
            )
        else:
            deformation_gradient = self.element.compute_deformation_gradient_from_shape_gradients(
                shape_gradients, current_ctrl_coords
            )
        return deformation_gradient

    @ti.kernel
    def calculate_material_energy(
        self,
        total_num_element: ti.i32,
        prefix_total_num_ctrlpts: ti.i32,
        prefix_num_knot: ti.types.vector(config.DIM, ti.i32),
        prefix_num_element: ti.types.vector(config.DIM, ti.i32),
        num_knot: ti.types.vector(config.DIM, ti.i32),
        num_element: ti.types.vector(config.DIM, ti.i32),
        num_ctrlpts: ti.types.vector(config.DIM, ti.i32),
        grid_disp: ti.template(),
    ):
        for ele in range(total_num_element):
            eleid = vectorize_id(ele, num_element)
            elrange_u = self.patch.element_u[prefix_num_element[0] + eleid[0]]
            elrange_v = self.patch.element_v[prefix_num_element[1] + eleid[1]]
            elrange_w = ti.Vector.zero(ti.f64, 2)
            j2 = jacobian2parent2parametric2d(elrange_u, elrange_v)
            if ti.static(config.DIM == 3):
                elrange_w = self.patch.element_w[prefix_num_element[2] + eleid[2]]
                j2 *= jacobian2parent2parametric1d(elrange_w)

            rest_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            current_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            for nodeID in ti.grouped(ti.ndrange(*self.element.knot_range)):
                global_nodeID = nodeID + eleid
                local_offset = linearize(nodeID, self.element.knot_range)
                global_offset = linearize(global_nodeID, num_ctrlpts)
                rest_control_points = self.patch.rest_control_points[prefix_total_num_ctrlpts + global_offset]
                control_points = self.patch.control_points[prefix_total_num_ctrlpts + global_offset]
                for j in ti.static(range(config.DIM)):
                    rest_ctrl_coords[local_offset, j] = rest_control_points[j]
                    current_ctrl_coords[local_offset, j] = (
                        control_points[j] + grid_disp[config.DIM * (prefix_total_num_ctrlpts + global_offset) + j]
                    )

            for gauss_id in range(self.element.gauss_number):
                N, dNdnat = self.element.dshapefn(
                    gauss_id,
                    elrange_u,
                    elrange_v,
                    elrange_w,
                    prefix_num_knot,
                    prefix_total_num_ctrlpts,
                    num_knot,
                    self.patch,
                )
                jacobian = self.calculate_jacobian(dNdnat, rest_ctrl_coords)
                j1 = self.require_positive_reference_jacobian(jacobian)
                reference_radius = self._axisymmetric_reference_radius(N, rest_ctrl_coords)
                volume = self._physical_quadrature_weight(
                    j1 * j2 * self.element.gauss_weights[gauss_id],
                    reference_radius,
                )
                dnatdX = jacobian.inverse()
                shape_gradients = self.element.compute_shape_gradients(dNdnat, dnatdX)
                deformation_gradient = self._constitutive_deformation_gradient(
                    N,
                    shape_gradients,
                    rest_ctrl_coords,
                    current_ctrl_coords,
                )
                self.energy[None] += self.material.Psi(deformation_gradient) * volume

    @ti.kernel
    def calculate_inertia_energy(
        self,
        total_num_ctrlpts: ti.i32,
        prefix_total_num_ctrlpts: ti.i32,
        integration: ti.types.vector(3, ti.f64),
        gravity: ti.types.vector(config.DIM, ti.f64),
        grid_disp: ti.template(),
    ):
        dt = self.TIdt[None]
        param1 = 1.0 / (2.0 * dt * dt * integration[0] * integration[1])
        param2 = param1 * dt
        param3 = 0.5 / integration[1] - 1.0
        for index in range(total_num_ctrlpts):
            ctrlpt_id = prefix_total_num_ctrlpts + index
            disp = ti.Vector([grid_disp[config.DIM * ctrlpt_id + d] for d in ti.static(range(config.DIM))])
            previous_velocity = self.patch.velocitys[ctrlpt_id]
            previous_acceleration = self.patch.accelerations[ctrlpt_id]
            nodal_mass = self.patch.volume[ctrlpt_id] * self.material.density
            self.energy[None] += 0.5 * nodal_mass * disp.dot(
                param1 * disp - 2.0 * param2 * previous_velocity - 2.0 * param3 * previous_acceleration
            ) - nodal_mass * gravity.dot(disp)

    @ti.kernel
    def get_neumann_energy(self, grid_disp: ti.template()):
        for i in self.neumann.node:
            dof_id = self.neumann.node[i]
            external_force = self.neumann.value[i]
            self.energy[None] -= grid_disp[dof_id] * external_force

    @ti.func
    def compute_local_hessian(
        self,
        local_offset1,
        local_offset2,
        d2Psi_d2F,
        shape_gradients,
        N,
        reference_radius,
    ):
        result = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
        if ti.static(self.is_axisymmetric):
            result = self.element.compute_axisymmetric_local_hessian(
                local_offset1,
                local_offset2,
                d2Psi_d2F,
                N,
                shape_gradients,
                reference_radius,
            )
        else:
            result = self.element.compute_local_hessian_from_shape_gradients(
                local_offset1,
                local_offset2,
                d2Psi_d2F,
                shape_gradients,
            )
        return result

    @ti.func
    def compute_local_gradient(
        self,
        local_offset,
        dPsi_dF,
        shape_gradients,
        N,
        reference_radius,
    ):
        result = ti.Vector.zero(ti.f64, config.DIM)
        if ti.static(self.is_axisymmetric):
            result = self.element.compute_axisymmetric_local_gradient(
                local_offset,
                dPsi_dF,
                N,
                shape_gradients,
                reference_radius,
            )
        else:
            result = self.element.compute_local_gradient_from_shape_gradients(
                local_offset,
                dPsi_dF,
                shape_gradients,
            )
        return result

    @ti.func
    def set_hash_block_entry(self, raw_slot, block_i, block_j, block):
        """Write one deterministic raw block slot.

        Diagonal contributions still accumulate in the dedicated diagonal
        field.  Their reserved raw slots are marked invalid so both host and
        device reductions skip them without changing the slot numbering of
        the remaining stencil.
        """
        if raw_slot < self.hash_matrix.non_diag.blockI.shape[0]:
            if block_i >= 0 and block_j >= 0 and block_i != block_j:
                self.hash_matrix.non_diag.blockI[raw_slot] = block_i
                self.hash_matrix.non_diag.blockJ[raw_slot] = block_j
                for d1 in ti.static(range(config.DIM)):
                    for d2 in ti.static(range(config.DIM)):
                        self.hash_matrix.non_diag.blockH[raw_slot][d1 * config.DIM + d2] = block[d1, d2]
            else:
                self.hash_matrix.non_diag.blockI[raw_slot] = -1
                self.hash_matrix.non_diag.blockJ[raw_slot] = -1
                self.hash_matrix.non_diag.blockH[raw_slot] = ti.Vector.zero(float, config.DIM * config.DIM)
                if block_i >= 0 and block_i == block_j:
                    for d1 in ti.static(range(config.DIM)):
                        for d2 in ti.static(range(config.DIM)):
                            ti.atomic_add(
                                self.hash_matrix.diag[block_i][d1 * config.DIM + d2],
                                block[d1, d2],
                            )
        else:
            self.hash_matrix.overflow[0] = 1

    def assemble_stiffness_matrix(
        self,
        prefix_num_nnz: ti.i32,
        total_num_ctrlpts: ti.i32,
        total_num_element: ti.i32,
        prefix_total_num_ctrlpts: ti.i32,
        prefix_num_knot: ti.types.vector(config.DIM, ti.i32),
        prefix_num_element: ti.types.vector(config.DIM, ti.i32),
        num_knot: ti.types.vector(config.DIM, ti.i32),
        num_element: ti.types.vector(config.DIM, ti.i32),
        num_ctrlpts: ti.types.vector(config.DIM, ti.i32),
        integration: ti.types.vector(3, ti.f64),
        gravity: ti.types.vector(config.DIM, ti.f64),
        grid_disp: ti.template(),
        need_matrix=True,
        project_spd=False,
    ):
        if self.assemble_type == "COO":
            self.assemble_stiffness_matrix_coo(
                prefix_num_nnz,
                total_num_ctrlpts,
                total_num_element,
                prefix_total_num_ctrlpts,
                prefix_num_knot,
                prefix_num_element,
                num_knot,
                num_element,
                num_ctrlpts,
                integration,
                gravity,
                grid_disp,
                bool(need_matrix),
                bool(project_spd),
            )
        else:
            prefix_hash_pair = int(prefix_num_nnz) // (config.DIM * config.DIM) * self.element.gauss_number
            self.assemble_stiffness_matrix_hash(
                prefix_hash_pair,
                total_num_ctrlpts,
                total_num_element,
                prefix_total_num_ctrlpts,
                prefix_num_knot,
                prefix_num_element,
                num_knot,
                num_element,
                num_ctrlpts,
                integration,
                gravity,
                grid_disp,
                bool(need_matrix),
                bool(project_spd),
            )

    @ti.kernel
    def assemble_stiffness_matrix_coo(
        self,
        prefix_num_nnz: ti.i32,
        total_num_ctrlpts: ti.i32,
        total_num_element: ti.i32,
        prefix_total_num_ctrlpts: ti.i32,
        prefix_num_knot: ti.types.vector(config.DIM, ti.i32),
        prefix_num_element: ti.types.vector(config.DIM, ti.i32),
        num_knot: ti.types.vector(config.DIM, ti.i32),
        num_element: ti.types.vector(config.DIM, ti.i32),
        num_ctrlpts: ti.types.vector(config.DIM, ti.i32),
        integration: ti.types.vector(3, ti.f64),
        gravity: ti.types.vector(config.DIM, ti.f64),
        grid_disp: ti.template(),
        need_matrix: ti.template(),
        project_spd: ti.template(),
    ):
        dt = self.TIdt[None]
        param1 = 1.0 / (2.0 * dt * dt * integration[0] * integration[1])
        param2 = param1 * dt
        param3 = 0.5 / integration[1] - 1.0
        element_nnz = self.influence_range * self.influence_range
        for ele in range(total_num_element):
            eleid = vectorize_id(ele, num_element)
            elrange_u = self.patch.element_u[prefix_num_element[0] + eleid[0]]
            elrange_v = self.patch.element_v[prefix_num_element[1] + eleid[1]]
            elrange_w = ti.Vector.zero(ti.f64, 2)
            j2 = jacobian2parent2parametric2d(elrange_u, elrange_v)
            if ti.static(config.DIM == 3):
                elrange_w = self.patch.element_w[prefix_num_element[2] + eleid[2]]
                j2 *= jacobian2parent2parametric1d(elrange_w)

            rest_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            current_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            for nodeID in ti.grouped(ti.ndrange(*self.element.knot_range)):
                global_nodeID = nodeID + eleid
                local_offset = linearize(nodeID, self.element.knot_range)
                global_offset = linearize(global_nodeID, num_ctrlpts)
                ctrlpt_id = prefix_total_num_ctrlpts + global_offset
                rest_control_points = self.patch.rest_control_points[ctrlpt_id]
                control_points = self.patch.control_points[ctrlpt_id]
                for j in ti.static(range(config.DIM)):
                    rest_ctrl_coords[local_offset, j] = rest_control_points[j]
                    current_ctrl_coords[local_offset, j] = control_points[j] + grid_disp[config.DIM * ctrlpt_id + j]

            for gauss_id in range(self.element.gauss_number):
                N, dNdnat = self.element.dshapefn(
                    gauss_id,
                    elrange_u,
                    elrange_v,
                    elrange_w,
                    prefix_num_knot,
                    prefix_total_num_ctrlpts,
                    num_knot,
                    self.patch,
                )
                jacobian = self.calculate_jacobian(dNdnat, rest_ctrl_coords)
                j1 = self.require_positive_reference_jacobian(jacobian)
                reference_radius = self._axisymmetric_reference_radius(N, rest_ctrl_coords)
                volume = self._physical_quadrature_weight(
                    j1 * j2 * self.element.gauss_weights[gauss_id],
                    reference_radius,
                )
                dnatdX = jacobian.inverse()
                shape_gradients = self.element.compute_shape_gradients(dNdnat, dnatdX)
                deformation_gradient = self._constitutive_deformation_gradient(
                    N,
                    shape_gradients,
                    rest_ctrl_coords,
                    current_ctrl_coords,
                )
                dPsi_dF = self.material.dPsi_div_dF(deformation_gradient) * volume
                d2Psi_d2F = ti.Matrix.zero(
                    ti.f64,
                    self.material_dimension * self.material_dimension,
                    self.material_dimension * self.material_dimension,
                )
                if ti.static(need_matrix):
                    d2Psi_d2F = self.material.d2Psi_div_d2F(deformation_gradient) * volume
                    if ti.static(project_spd):
                        d2Psi_d2F = psd_project_nd(d2Psi_d2F)

                for nodeID1 in ti.grouped(ti.ndrange(*self.element.knot_range)):
                    global_nodeID1 = nodeID1 + eleid
                    local_offset1 = linearize(nodeID1, self.element.knot_range)
                    global_offset1 = linearize(global_nodeID1, num_ctrlpts)
                    block_i = prefix_total_num_ctrlpts + global_offset1
                    local_gradient = self.compute_local_gradient(
                        local_offset1,
                        dPsi_dF,
                        shape_gradients,
                        N,
                        reference_radius,
                    )
                    for d in ti.static(range(config.DIM)):
                        self.rhs[config.DIM * block_i + d] -= local_gradient[d]
                    if ti.static(need_matrix):
                        for nodeID2 in ti.grouped(ti.ndrange(*self.element.knot_range)):
                            global_nodeID2 = nodeID2 + eleid
                            local_offset2 = linearize(nodeID2, self.element.knot_range)
                            global_offset2 = linearize(global_nodeID2, num_ctrlpts)
                            block_j = prefix_total_num_ctrlpts + global_offset2
                            local_d2Psi_d2x = self.compute_local_hessian(
                                local_offset1,
                                local_offset2,
                                d2Psi_d2F,
                                shape_gradients,
                                N,
                                reference_radius,
                            )
                            for d1 in ti.static(range(config.DIM)):
                                local_row = local_offset1 * config.DIM + d1
                                row = config.DIM * block_i + d1
                                for d2 in ti.static(range(config.DIM)):
                                    local_col = local_offset2 * config.DIM + d2
                                    col = config.DIM * block_j + d2
                                    entry = (
                                        prefix_num_nnz
                                        + ele * element_nnz
                                        + local_row * self.influence_range
                                        + local_col
                                    )
                                    self.coo_matrix.rows[entry] = row
                                    self.coo_matrix.cols[entry] = col
                                    ti.atomic_add(self.coo_matrix.data[entry], local_d2Psi_d2x[d1, d2])

        for index in range(total_num_ctrlpts):
            ctrlpt_id = prefix_total_num_ctrlpts + index
            disp = ti.Vector([grid_disp[config.DIM * ctrlpt_id + d] for d in ti.static(range(config.DIM))])
            previous_velocity = self.patch.velocitys[ctrlpt_id]
            previous_acceleration = self.patch.accelerations[ctrlpt_id]
            nodal_mass = self.patch.volume[ctrlpt_id] * self.material.density
            grid_a = param1 * disp - param2 * previous_velocity - param3 * previous_acceleration
            for d in ti.static(range(config.DIM)):
                dof = config.DIM * ctrlpt_id + d
                self.rhs[dof] += nodal_mass * (gravity[d] - grid_a[d])
                if ti.static(need_matrix):
                    entry = self.stiffness_nnz + dof
                    self.coo_matrix.rows[entry] = dof
                    self.coo_matrix.cols[entry] = dof
                    self.coo_matrix.data[entry] += param1 * nodal_mass

    @ti.kernel
    def assemble_stiffness_matrix_hash(
        self,
        prefix_hash_pair: ti.i32,
        total_num_ctrlpts: ti.i32,
        total_num_element: ti.i32,
        prefix_total_num_ctrlpts: ti.i32,
        prefix_num_knot: ti.types.vector(config.DIM, ti.i32),
        prefix_num_element: ti.types.vector(config.DIM, ti.i32),
        num_knot: ti.types.vector(config.DIM, ti.i32),
        num_element: ti.types.vector(config.DIM, ti.i32),
        num_ctrlpts: ti.types.vector(config.DIM, ti.i32),
        integration: ti.types.vector(3, ti.f64),
        gravity: ti.types.vector(config.DIM, ti.f64),
        grid_disp: ti.template(),
        need_matrix: ti.template(),
        project_spd: ti.template(),
    ):
        dt = self.TIdt[None]
        param1 = 1.0 / (2.0 * dt * dt * integration[0] * integration[1])
        param2 = param1 * dt
        param3 = 0.5 / integration[1] - 1.0
        local_node_count = ti.static(self.element.total_knot_range)
        raw_pairs_per_element = ti.static(
            self.element.gauss_number * self.element.total_knot_range * self.element.total_knot_range
        )
        if ti.static(need_matrix):
            raw_end = prefix_hash_pair + total_num_element * raw_pairs_per_element
            ti.atomic_max(self.hash_matrix.raw_non_diag_count[0], raw_end)
            if raw_end > self.hash_matrix.non_diag.blockI.shape[0]:
                self.hash_matrix.overflow[0] = 1
        for ele in range(total_num_element):
            eleid = vectorize_id(ele, num_element)
            elrange_u = self.patch.element_u[prefix_num_element[0] + eleid[0]]
            elrange_v = self.patch.element_v[prefix_num_element[1] + eleid[1]]
            elrange_w = ti.Vector.zero(ti.f64, 2)
            j2 = jacobian2parent2parametric2d(elrange_u, elrange_v)
            if ti.static(config.DIM == 3):
                elrange_w = self.patch.element_w[prefix_num_element[2] + eleid[2]]
                j2 *= jacobian2parent2parametric1d(elrange_w)

            rest_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            current_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            for nodeID in ti.grouped(ti.ndrange(*self.element.knot_range)):
                global_nodeID = nodeID + eleid
                local_offset = linearize(nodeID, self.element.knot_range)
                global_offset = linearize(global_nodeID, num_ctrlpts)
                ctrlpt_id = prefix_total_num_ctrlpts + global_offset
                rest_control_points = self.patch.rest_control_points[ctrlpt_id]
                control_points = self.patch.control_points[ctrlpt_id]
                for j in ti.static(range(config.DIM)):
                    rest_ctrl_coords[local_offset, j] = rest_control_points[j]
                    current_ctrl_coords[local_offset, j] = control_points[j] + grid_disp[config.DIM * ctrlpt_id + j]

            for gauss_id in range(self.element.gauss_number):
                N, dNdnat = self.element.dshapefn(
                    gauss_id,
                    elrange_u,
                    elrange_v,
                    elrange_w,
                    prefix_num_knot,
                    prefix_total_num_ctrlpts,
                    num_knot,
                    self.patch,
                )
                jacobian = self.calculate_jacobian(dNdnat, rest_ctrl_coords)
                j1 = self.require_positive_reference_jacobian(jacobian)
                reference_radius = self._axisymmetric_reference_radius(N, rest_ctrl_coords)
                volume = self._physical_quadrature_weight(
                    j1 * j2 * self.element.gauss_weights[gauss_id],
                    reference_radius,
                )
                dnatdX = jacobian.inverse()
                shape_gradients = self.element.compute_shape_gradients(dNdnat, dnatdX)
                deformation_gradient = self._constitutive_deformation_gradient(
                    N,
                    shape_gradients,
                    rest_ctrl_coords,
                    current_ctrl_coords,
                )
                dPsi_dF = self.material.dPsi_div_dF(deformation_gradient) * volume
                d2Psi_d2F = ti.Matrix.zero(
                    ti.f64,
                    self.material_dimension * self.material_dimension,
                    self.material_dimension * self.material_dimension,
                )
                if ti.static(need_matrix):
                    d2Psi_d2F = self.material.d2Psi_div_d2F(deformation_gradient) * volume
                    if ti.static(project_spd):
                        d2Psi_d2F = psd_project_nd(d2Psi_d2F)

                for nodeID1 in ti.grouped(ti.ndrange(*self.element.knot_range)):
                    global_nodeID1 = nodeID1 + eleid
                    local_offset1 = linearize(nodeID1, self.element.knot_range)
                    global_offset1 = linearize(global_nodeID1, num_ctrlpts)
                    block_i = prefix_total_num_ctrlpts + global_offset1
                    local_gradient = self.compute_local_gradient(
                        local_offset1,
                        dPsi_dF,
                        shape_gradients,
                        N,
                        reference_radius,
                    )
                    for d in ti.static(range(config.DIM)):
                        self.rhs[config.DIM * block_i + d] -= local_gradient[d]
                    if ti.static(need_matrix):
                        for nodeID2 in ti.grouped(ti.ndrange(*self.element.knot_range)):
                            global_nodeID2 = nodeID2 + eleid
                            local_offset2 = linearize(nodeID2, self.element.knot_range)
                            global_offset2 = linearize(global_nodeID2, num_ctrlpts)
                            block_j = prefix_total_num_ctrlpts + global_offset2
                            local_d2Psi_d2x = self.compute_local_hessian(
                                local_offset1,
                                local_offset2,
                                d2Psi_d2F,
                                shape_gradients,
                                N,
                                reference_radius,
                            )
                            raw_slot = (
                                prefix_hash_pair
                                + ele * raw_pairs_per_element
                                + gauss_id * local_node_count * local_node_count
                                + local_offset1 * local_node_count
                                + local_offset2
                            )
                            self.set_hash_block_entry(
                                raw_slot,
                                block_i,
                                block_j,
                                local_d2Psi_d2x,
                            )

        for index in range(total_num_ctrlpts):
            ctrlpt_id = prefix_total_num_ctrlpts + index
            disp = ti.Vector([grid_disp[config.DIM * ctrlpt_id + d] for d in ti.static(range(config.DIM))])
            previous_velocity = self.patch.velocitys[ctrlpt_id]
            previous_acceleration = self.patch.accelerations[ctrlpt_id]
            nodal_mass = self.patch.volume[ctrlpt_id] * self.material.density
            grid_v = (
                0.5 * integration[2] / integration[0] / integration[1] / dt * disp
                - (0.5 * integration[2] / integration[0] / integration[1] - 1.0) * previous_velocity
                - 0.5 * dt * (integration[2] / integration[1] - 2.0) * previous_acceleration
            )
            grid_a = param1 * disp - param2 * previous_velocity - param3 * previous_acceleration
            for d in ti.static(range(config.DIM)):
                self.rhs[config.DIM * ctrlpt_id + d] += nodal_mass * (gravity[d] - grid_a[d])
                if ti.static(need_matrix):
                    self.hash_matrix.diag[ctrlpt_id][d * config.DIM + d] += param1 * nodal_mass

    def apply_dirichlet(self):
        if self.assemble_type == "COO":
            self.apply_dirichlet_coo()
        else:
            self.apply_dirichlet_hash()

    @ti.kernel
    def apply_dirichlet_coo(self):
        for k in range(self.total_nnz):
            row = self.coo_matrix.rows[k]
            col = self.coo_matrix.cols[k]
            value = self.coo_matrix.data[k]
            if 0 <= row < self.degree_of_freedom and 0 <= col < self.degree_of_freedom:
                if self.dirichlet.node[col] == 1 or self.dirichlet.node[row] == 1:
                    prescribed_correction = self.dirichlet.value[col] - self.grid_disp[col]
                    self.rhs[row] -= value * prescribed_correction
                    self.coo_matrix.data[k] = 0.0

        for i in range(self.degree_of_freedom):
            if self.dirichlet.node[i] == 1:
                self.rhs[i] = self.dirichlet.value[i] - self.grid_disp[i]
                entry = self.stiffness_nnz + i
                self.coo_matrix.rows[entry] = i
                self.coo_matrix.cols[entry] = i
                self.coo_matrix.data[entry] = 1.0

    def _ensure_pcg_hash_matrix(self):
        if self.pcg_hash_matrix is None:
            self.pcg_hash_matrix = BuildTriplet(
                dim=config.DIM,
                max_pairs_num=self.hash_matrix.non_diag.max_pairs_num,
                max_nonzeros=self.hash_matrix.max_nonzeros,
                max_active_nodes=self.hash_matrix.max_active_nodes,
                symmetric=False,
                solver="PCG",
                matrix_symmetric=True,
                full_symmetric_input=True,
                device_reduction=self.hash_matrix.non_diag.device_reduction,
            )
        return self.pcg_hash_matrix

    def _prepare_pcg_hash_matrix(self):
        matrix = self._ensure_pcg_hash_matrix()
        matrix.reset_system()
        matrix.append_raw_from(
            self.hash_matrix,
            active_nodes=self.degree_of_freedom // config.DIM,
        )
        # Dirichlet elimination below is mirror-aware, so canonicalization
        # happens first and its RHS correction uses exactly the operator PCG
        # will later see.
        matrix.canonicalize_full_symmetric_input()
        self._pcg_hash_prepared = True
        return matrix

    def apply_dirichlet_hash(self):
        matrix = self.hash_matrix
        if self.linear_solver == "PCG":
            matrix = self._prepare_pcg_hash_matrix()
        self._apply_dirichlet_hash_matrix(matrix)

    @ti.kernel
    def _apply_dirichlet_hash_matrix(self, matrix: ti.template()):
        active_nodes = self.degree_of_freedom // config.DIM
        for block in range(active_nodes):
            # Update a local dense block and store it once.  Component-wise
            # writes through a nested Vector-field subscript can be lost when
            # a component is first cleared and then reset to the Dirichlet
            # identity in the same Taichi kernel (observed for fully fixed
            # control points).  A single whole-vector write also makes the
            # intended row/column elimination ordering explicit.
            diagonal = matrix.diag[block]
            for d1 in ti.static(range(config.DIM)):
                row = config.DIM * block + d1
                for d2 in ti.static(range(config.DIM)):
                    col = config.DIM * block + d2
                    h_index = d1 * config.DIM + d2
                    value = diagonal[h_index]
                    if self.dirichlet.node[col] == 1 or self.dirichlet.node[row] == 1:
                        prescribed_correction = self.dirichlet.value[col] - self.grid_disp[col]
                        self.rhs[row] -= value * prescribed_correction
                        diagonal[h_index] = 0.0
                if self.dirichlet.node[row] == 1:
                    diagonal[d1 * config.DIM + d1] = 1.0
            matrix.diag[block] = diagonal

        raw_nnz = matrix.raw_non_diag_count[0]
        for k in range(raw_nnz):
            bi = matrix.non_diag.blockI[k]
            bj = matrix.non_diag.blockJ[k]
            if 0 <= bi < active_nodes and 0 <= bj < active_nodes:
                block_hessian = matrix.non_diag.blockH[k]
                for d1 in ti.static(range(config.DIM)):
                    row = config.DIM * bi + d1
                    for d2 in ti.static(range(config.DIM)):
                        col = config.DIM * bj + d2
                        h_index = d1 * config.DIM + d2
                        value = block_hessian[h_index]
                        column_fixed = self.dirichlet.node[col] == 1
                        row_fixed = self.dirichlet.node[row] == 1
                        if column_fixed:
                            prescribed_correction = self.dirichlet.value[col] - self.grid_disp[col]
                            self.rhs[row] -= value * prescribed_correction
                        if ti.static(matrix.matrix_symmetric):
                            # Only the upper block is stored. Its transposed
                            # mirror contributes to the other RHS orientation.
                            if row_fixed:
                                prescribed_correction = self.dirichlet.value[row] - self.grid_disp[row]
                                ti.atomic_add(
                                    self.rhs[col],
                                    -value * prescribed_correction,
                                )
                        if column_fixed or row_fixed:
                            block_hessian[h_index] = 0.0
                matrix.non_diag.blockH[k] = block_hessian

        for i in self.rhs:
            if self.dirichlet.node[i] == 1:
                self.rhs[i] = self.dirichlet.value[i] - self.grid_disp[i]

    def solve_hash_system(self, return_solution=True):
        active_nodes = self.degree_of_freedom // config.DIM
        matrix = self.hash_matrix
        if self.linear_solver == "PCG":
            if not self._pcg_hash_prepared:
                matrix = self._prepare_pcg_hash_matrix()
            else:
                matrix = self.pcg_hash_matrix
        matrix.finalize_taichi_assembly()
        if self.linear_solver in ("PCG", "BiCGSTAB"):
            result = matrix.solve_flat_system(
                self.rhs,
                self.incre_resolution,
                active_nodes=active_nodes,
                tol=self.linear_solver_tolerance,
                maxiter=self.linear_solver_max_iters,
                return_solution=return_solution,
            )
            if not result["converged"]:
                raise RuntimeError(
                    f"ImplicitIGA Taichi {self.linear_solver} did not "
                    "converge: "
                    f"residual={result['residual']:.6e}, "
                    f"iterations={result['iterations']}"
                )
            return result["x"] if return_solution else result
        csr_matrixes = matrix.to_scipy(active_nodes).tocsr()
        rhs = self.rhs.to_numpy()
        return solve_csr_system(rhs, csr_matrixes)

    @ti.kernel
    def build_coo_diagonal_preconditioner(self):
        for i in range(self.degree_of_freedom):
            self.diag_A[i] = 0.0
        for k in range(self.total_nnz):
            row = self.coo_matrix.rows[k]
            col = self.coo_matrix.cols[k]
            if row == col and 0 <= row < self.degree_of_freedom:
                ti.atomic_add(self.diag_A[row], self.coo_matrix.data[k])
        for i in range(self.degree_of_freedom):
            if ti.abs(self.diag_A[i]) < 1.0e-14:
                self.diag_A[i] = 1.0

    @ti.kernel
    def increment_inf_norm(self) -> ti.f64:
        value = 0.0
        for dof in range(self.degree_of_freedom):
            ti.atomic_max(value, ti.abs(self.incre_resolution[dof]))
        return value

    def solve_coo_system(self, return_solution=True):
        if self.linear_solver == "PCG":
            self.build_coo_diagonal_preconditioner()
            self.incre_resolution.fill(0.0)
            converged = self.coo_matrix.solve(
                self.rhs,
                self.incre_resolution,
                self.diag_A,
                tol=self.linear_solver_tolerance,
                maxiter=self.linear_solver_max_iters,
            )
            diagnostics = {
                "converged": bool(converged),
                "residual": float(self.coo_matrix.linear_solver.last_residual),
                "iterations": int(self.coo_matrix.linear_solver.last_iterations),
                "solution_inf_norm": float(self.increment_inf_norm()),
            }
            if not converged:
                solver = self.coo_matrix.linear_solver
                raise IGAConvergenceError(
                    "IGA COO-PCG failed to converge: "
                    f"residual={solver.last_residual:.6e}, "
                    f"iterations={solver.last_iterations}, "
                    f"reason={solver.last_breakdown_reason or 'unknown'}"
                )
            if return_solution:
                # Explicit result/output boundary only. Runtime Newton calls
                # this method with ``return_solution=False`` and keeps the
                # correction in ``incre_resolution``.
                return self.incre_resolution.to_numpy()
            return diagnostics

        csr_matrixes = self.coo_matrix._to_scipy().tocsr()
        return self.coo_matrix.spsolve(self.rhs, csr_matrixes)

    def solve_system(self, return_solution=True):
        if self.assemble_type == "COO":
            return self.solve_coo_system(return_solution=return_solution)
        return self.solve_hash_system(return_solution=return_solution)

    @ti.kernel
    def apply_neumann(self):
        for i in self.neumann.node:
            self.rhs[self.neumann.node[i]] += self.neumann.value[i]

    @ti.kernel
    def dynamic_advance(
        self, total_num_ctrlpts: ti.i32, prefix_total_num_ctrlpts: ti.i32, integration: ti.types.vector(3, ti.f64)
    ):
        dt = self.TIdt[None]
        param1 = 1.0 / (2.0 * dt * dt * integration[0] * integration[1])
        param2 = param1 * dt
        param3 = 0.5 / integration[1] - 1.0
        for index in range(total_num_ctrlpts):
            ctrlpt_id = prefix_total_num_ctrlpts + index
            displacement = ti.Vector([self.grid_disp[config.DIM * ctrlpt_id + d] for d in ti.static(range(config.DIM))])
            previous_velocity = self.patch.velocitys[ctrlpt_id]
            previous_acceleration = self.patch.accelerations[ctrlpt_id]
            self.patch.velocitys[ctrlpt_id] = (
                0.5 * integration[2] / integration[0] / integration[1] / dt * displacement
                - (0.5 * integration[2] / integration[0] / integration[1] - 1.0) * previous_velocity
                - 0.5 * dt * (integration[2] / integration[1] - 2.0) * previous_acceleration
            )
            self.patch.accelerations[ctrlpt_id] = (
                param1 * displacement - param2 * previous_velocity - param3 * previous_acceleration
            )
            self.patch.control_points[ctrlpt_id] += displacement

    @ti.kernel
    def material_ccd(
        self,
        total_num_element: ti.i32,
        prefix_total_num_ctrlpts: ti.i32,
        prefix_num_knot: ti.types.vector(config.DIM, ti.i32),
        prefix_num_element: ti.types.vector(config.DIM, ti.i32),
        num_knot: ti.types.vector(config.DIM, ti.i32),
        num_element: ti.types.vector(config.DIM, ti.i32),
        num_ctrlpts: ti.types.vector(config.DIM, ti.i32),
        slackness: ti.f64,
    ) -> ti.f64:
        alpha = 1.0
        for ele in range(total_num_element):
            eleid = vectorize_id(ele, num_element)
            elrange_u = self.patch.element_u[prefix_num_element[0] + eleid[0]]
            elrange_v = self.patch.element_v[prefix_num_element[1] + eleid[1]]
            elrange_w = ti.Vector.zero(ti.f64, 2)
            if ti.static(config.DIM == 3):
                elrange_w = self.patch.element_w[prefix_num_element[2] + eleid[2]]

            rest_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            current_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            current_incre_disps = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            for nodeID in ti.grouped(ti.ndrange(*self.element.knot_range)):
                global_nodeID = nodeID + eleid
                offset = linearize(nodeID, self.element.knot_range)
                global_offset = linearize(global_nodeID, num_ctrlpts)
                rest_control_points = self.patch.rest_control_points[prefix_total_num_ctrlpts + global_offset]
                control_points = self.patch.control_points[prefix_total_num_ctrlpts + global_offset]
                for j in ti.static(range(config.DIM)):
                    rest_ctrl_coords[offset, j] = rest_control_points[j]
                    current_ctrl_coords[offset, j] = (
                        control_points[j] + self.grid_disp[config.DIM * (prefix_total_num_ctrlpts + global_offset) + j]
                    )
                    current_incre_disps[offset, j] = self.incre_resolution[
                        config.DIM * (prefix_total_num_ctrlpts + global_offset) + j
                    ]

            alphaE = 1.0
            for gauss_id in range(self.element.gauss_number):
                N, dNdnat = self.element.dshapefn(
                    gauss_id,
                    elrange_u,
                    elrange_v,
                    elrange_w,
                    prefix_num_knot,
                    prefix_total_num_ctrlpts,
                    num_knot,
                    self.patch,
                )
                jacobian = self.calculate_jacobian(dNdnat, rest_ctrl_coords)
                self.require_positive_reference_jacobian(jacobian)
                dnatdX = jacobian.inverse()
                shape_gradients = self.element.compute_shape_gradients(dNdnat, dnatdX)
                deformation_gradient = self._constitutive_deformation_gradient(
                    N,
                    shape_gradients,
                    rest_ctrl_coords,
                    current_ctrl_coords,
                )
                deformation_gradient_incre = ti.Matrix.zero(ti.f64, self.material_dimension, self.material_dimension)
                if ti.static(self.is_axisymmetric):
                    reference_radius = self._axisymmetric_reference_radius(N, rest_ctrl_coords)
                    for local_offset in range(self.element.total_knot_range):
                        for spatial, material_axis in ti.static(ti.ndrange(2, 2)):
                            deformation_gradient_incre[spatial, material_axis] += (
                                shape_gradients[local_offset, material_axis]
                                * current_incre_disps[local_offset, spatial]
                            )
                    deformation_gradient_incre[2, 2] = self.element.interpolate_component(
                        N, current_incre_disps, 0
                    ) / ti.max(reference_radius, 1.0e-30)
                else:
                    deformation_gradient_incre = self.element.compute_deformation_gradient(
                        dNdnat, dnatdX, current_incre_disps
                    )
                solution = deformation_gradient_ccd(
                    deformation_gradient,
                    deformation_gradient_incre,
                    slackness,
                )
                if solution < alphaE:
                    alphaE = solution
            ti.atomic_min(alpha, alphaE)
        return alpha

    @ti.kernel
    def calc_g0(self) -> ti.f64:
        g = 0.0
        for i in self.incre_resolution:
            g += self.incre_resolution[i] * self.rhs[i]
        return g

    @ti.kernel
    def update_grid_disp(self, alpha: ti.f64):
        for i in self.grid_disp:
            self.grid_disp_temp[i] = self.grid_disp[i] + alpha * self.incre_resolution[i]

    def assemble_body_matrix(self, need_matrix=True, project_spd=None):
        if project_spd is None:
            project_spd = self.project_hessian_to_psd
        if need_matrix:
            self._pcg_hash_prepared = False
        for patch_id in range(self.patch.primitive.num_primitives):
            prefix_nnz = self.prefix_nnz[patch_id]
            prefix_num_knot = self.patch.prefix_num_knot[patch_id]
            prefix_num_element = self.patch.prefix_num_element[patch_id]
            prefix_total_num_ctrlpts = self.patch.prefix_total_num_ctrlpts[patch_id]
            total_num_ctrlpts = self.patch.total_num_ctrlpts[patch_id + 1]
            total_num_element = self.patch.total_num_element[patch_id + 1]
            num_knot = self.patch.num_knot[patch_id + 1]
            num_element = self.patch.num_element[patch_id + 1]
            num_ctrlpts = self.patch.num_ctrlpts[patch_id + 1]
            self.assemble_stiffness_matrix(
                prefix_nnz,
                total_num_ctrlpts,
                total_num_element,
                prefix_total_num_ctrlpts,
                prefix_num_knot,
                prefix_num_element,
                num_knot,
                num_element,
                num_ctrlpts,
                self.integration,
                self.gravity,
                self.grid_disp,
                need_matrix=need_matrix,
                project_spd=bool(project_spd),
            )

    def reset_linear_system(self):
        if self.assemble_type == "COO":
            self.coo_matrix.reset()
        else:
            self.hash_matrix.reset_system()
            self._pcg_hash_prepared = False

    def get_material_energy(self, grid_disp):
        for patch_id in range(self.patch.primitive.num_primitives):
            prefix_num_knot = self.patch.prefix_num_knot[patch_id]
            prefix_num_element = self.patch.prefix_num_element[patch_id]
            prefix_total_num_ctrlpts = self.patch.prefix_total_num_ctrlpts[patch_id]
            total_num_element = self.patch.total_num_element[patch_id + 1]
            num_knot = self.patch.num_knot[patch_id + 1]
            num_element = self.patch.num_element[patch_id + 1]
            num_ctrlpts = self.patch.num_ctrlpts[patch_id + 1]
            total_num_element = self.patch.total_num_element[patch_id + 1]
            self.calculate_material_energy(
                total_num_element,
                prefix_total_num_ctrlpts,
                prefix_num_knot,
                prefix_num_element,
                num_knot,
                num_element,
                num_ctrlpts,
                grid_disp,
            )

    def get_inertia_energy(self, grid_disp):
        for patch_id in range(self.patch.primitive.num_primitives):
            prefix_total_num_ctrlpts = self.patch.prefix_total_num_ctrlpts[patch_id]
            total_num_ctrlpts = self.patch.total_num_ctrlpts[patch_id + 1]
            self.calculate_inertia_energy(
                total_num_ctrlpts, prefix_total_num_ctrlpts, self.integration, self.gravity, grid_disp
            )

    def iterative_dynamic_advance(self):
        for patch_id in range(self.patch.primitive.num_primitives):
            prefix_total_num_ctrlpts = self.patch.prefix_total_num_ctrlpts[patch_id]
            total_num_ctrlpts = self.patch.total_num_ctrlpts[patch_id + 1]
            self.dynamic_advance(total_num_ctrlpts, prefix_total_num_ctrlpts, self.integration)

    def total_energy(self, grid_disp):
        self.energy[None] = 0.0
        self.get_material_energy(grid_disp)
        self.get_inertia_energy(grid_disp)
        self.get_neumann_energy_step(grid_disp)
        return self.energy[None]

    def ccd(self):
        slackness_m = 0.8
        alpha_material = 1.0
        for patch_id in range(self.patch.primitive.num_primitives):
            prefix_num_knot = self.patch.prefix_num_knot[patch_id]
            prefix_num_element = self.patch.prefix_num_element[patch_id]
            total_num_element = self.patch.total_num_element[patch_id + 1]
            prefix_total_num_ctrlpts = self.patch.prefix_total_num_ctrlpts[patch_id]
            num_knot = self.patch.num_knot[patch_id + 1]
            num_element = self.patch.num_element[patch_id + 1]
            num_ctrlpts = self.patch.num_ctrlpts[patch_id + 1]
            alpha_material = min(
                alpha_material,
                self.material_ccd(
                    total_num_element,
                    prefix_total_num_ctrlpts,
                    prefix_num_knot,
                    prefix_num_element,
                    num_knot,
                    num_element,
                    num_ctrlpts,
                    slackness_m,
                ),
            )
        return alpha_material

    def line_search(self, verbose=True):
        g0 = -self.calc_g0()
        if not np.isfinite(g0) or g0 > 0.0:
            raise IGAConvergenceError(f"implicit IGA Newton direction is not descending: g0={g0}")
        alpha = self.ccd()
        self.update_grid_disp(alpha)
        previous_energy = self.total_energy(self.grid_disp)
        accepted = False
        while alpha > 1e-6:
            current_energy = self.total_energy(self.grid_disp_temp)
            if current_energy <= previous_energy + 1e-4 * alpha * g0:
                accepted = True
                break
            alpha *= 0.5
            if verbose:
                print(f"current alpha: {alpha}, current energy: {current_energy}, previous energy: {previous_energy}")
            self.update_grid_disp(alpha)
        if not accepted:
            raise IGAConvergenceError("implicit IGA line search failed to find an admissible step")
        copy_group_field(self.grid_disp, self.grid_disp_temp)

    def _solve_device_newton_iteration(self):
        result = self.solve_system(return_solution=False)
        return result["solution_inf_norm"] / self.dt

    def _solve_host_newton_iteration(self):
        result = self.solve_system(return_solution=True)
        self.incre_resolution.from_numpy(result)
        return np.linalg.norm(result, np.inf) / self.dt

    def precompute(self):
        super().precompute()

    def initial_simulation(self):
        self.precompute()
        self.visualize_stress()
        self.visualize()

    def record(self, log=True):
        self.visualize_stress()
        self.visualize(log=log)

    def _substep_once(self, verbose=True):
        self.grid_disp.fill(0)
        iter_num = 0
        residual = 1.0
        while iter_num < self.max_iters:
            self.rhs.fill(0)
            self.incre_resolution.fill(0)
            self.reset_linear_system()

            self.assemble_body_matrix()
            self.apply_dirichlet_step()
            self.apply_neumann_step()

            try:
                residual = self.solve_newton_iteration()
            except RuntimeError as exception:
                if not is_recoverable_nonlinear_failure(exception):
                    raise
                raise IGAConvergenceError("implicit IGA linear solve did not converge") from exception
            if residual < self.tol:
                break

            self.line_search(verbose)
            iter_num += 1
        if residual >= self.tol:
            raise IGAConvergenceError(
                f"implicit IGA did not converge: residual={residual:.6e} " f"after {iter_num} iterations"
            )
        self.iterative_dynamic_advance()
        self.time += float(self.dt)
        self.step_count += 1
        record = {
            "step": int(self.step_count),
            "time": float(self.time),
            "converged": True,
            "iterations": int(iter_num),
            "residual": float(residual),
        }
        self.last_step_record = record
        if self.record_history_step:
            self.step_schedule.append_history(self.history, record)
        if verbose:
            print(f"Iteration {iter_num}, residual: {residual}")

    def _failure_diagnostics(self, exception, attempt, timestep):
        return {
            "kind": nonlinear_failure_kind(exception),
            "exception": type(exception).__name__,
            "message": str(exception),
            "attempt": int(attempt),
            "timestep": float(timestep),
            "time": float(self.time),
            "step": int(self.step_count),
        }

    def diagnostics_snapshot(self):
        return {
            "schema_version": 1,
            "subsystem": "iga_implicit",
            "time": float(self.time),
            "step": int(self.step_count),
            "timestep": float(self.dt),
            "last_failure": self.last_failure,
            "last_step": self.history[-1] if self.history else None,
        }

    def substep(self, verbose=True, record_history=True):
        self.record_history_step = bool(record_history)
        original_timestep = float(self.dt)
        attempt_timestep = original_timestep
        attempts = []
        for attempt in range(self.step_retry.maximum_retries + 1):
            self.dt = float(attempt_timestep)
            try:
                result = self._substep_once(verbose=verbose)
            except IGAConvergenceError as exception:
                failure = self._failure_diagnostics(exception, attempt, attempt_timestep)
                attempts.append(failure)
                next_timestep = self.step_retry.next_timestep(attempt_timestep, attempt)
                if next_timestep is None:
                    self.last_failure = {
                        **failure,
                        "original_timestep": original_timestep,
                        "attempts": attempts,
                    }
                    self.dt = original_timestep
                    raise
                attempt_timestep = next_timestep
                continue

            if self.last_step_record is not None:
                self.last_step_record["step_retry"] = {
                    "enabled": bool(self.step_retry.enabled),
                    "original_timestep": original_timestep,
                    "accepted_timestep": float(attempt_timestep),
                    "retry_count": int(attempt),
                    "attempts": attempts,
                }
            self.last_failure = None
            return result

        raise AssertionError("unreachable IGA retry state")

    def run(self, verbose=True, postprocessing=()):
        postprocessing = normalize_callbacks(postprocessing)
        with self.timer.section("IGA initialization"):
            self.precompute()
        with self.timer.section("Output"):
            self.record()
        self.timer.profile0()
        with self.timer.section("Postprocess"):
            for f in postprocessing:
                f()
        for i in range(self.total_step):
            for interval_index in range(self.output_interval):
                next_step = self.step_count + 1
                output_due = interval_index + 1 == self.output_interval
                final_step = i + 1 == self.total_step and output_due
                compiling = self.compile_seconds is None
                if compiling:
                    print("Compiling first ... ...")
                    compile_start = time.perf_counter()
                with self.timer.section("IGA implicit step"):
                    self.substep(
                        verbose,
                        record_history=self.step_schedule.history_due(
                            next_step,
                            output=output_due,
                            final=final_step,
                        ),
                    )
                if compiling:
                    ti.sync()
                    self.compile_seconds = time.perf_counter() - compile_start
                    print(f"Compiling time = {self.compile_seconds} \n")
                    self.timer.profile1()
                runtime_checkpoint()
            with self.timer.section("Output"):
                self.record()
            self.timer.profile0()
            with self.timer.section("Postprocess"):
                for f in postprocessing:
                    f()


__all__ = ["IGAConvergenceError", "ImplicitIGA"]
