import taichi as ti
import numpy as np

import src.mpm.config as config
from src.mpm.engines.direct.MPMSolver import MPMSolver
from src.mpm.utils import copy_field, copy_grad
from src.physics_model.consititutive_model.finite_strain.NeoHookean import NeoHookeanModel
from src.physics_model.consititutive_model.finite_strain.DruckerPrager import (
    FiniteStrainDruckerPragerModel,
)
from src.physics_model.consititutive_model.finite_strain.VonMises import (
    FiniteStrainVonMisesModel,
)
from src.physics_model.consititutive_model.finite_strain.ModifiedCamClay import (
    FiniteStrainModifiedCamClayModel,
)
from src.linear_solver.BuildTriplet import BuildTriplet, solve_csr_system
from src.utils.PrefixSum import PrefixSumExecutor
from src.utils.linalg import no_operation


@ti.data_oriented
class ImplicitMPM(MPMSolver):
    def __init__(self, bodies, dirichlet=None, neumann=None, **kwargs):
        material = kwargs.get("material", "neoHookean")
        material_key = str(material).replace("-", "").replace("_", "").replace(" ", "").lower()
        if material_key in ("neohookean", "neohookeanmodel", "neo"):
            self.material = NeoHookeanModel().initialize_from_kwargs(**kwargs)
        elif material_key in (
            "druckerprager",
            "druckerpragermodel",
            "finitedruckerprager",
            "finitedruckerpragermodel",
            "dp",
        ):
            self.material = FiniteStrainDruckerPragerModel().initialize_from_kwargs(**kwargs)
        elif material_key in (
            "vonmises",
            "vonmisesmodel",
            "finitevonmises",
            "finitevonmisesmodel",
            "j2",
            "j2plasticity",
        ):
            self.material = FiniteStrainVonMisesModel().initialize_from_kwargs(**kwargs)
        elif material_key in (
            "modifiedcamclay",
            "modifiedcamclaymodel",
            "finitemodifiedcamclay",
            "finitemodifiedcamclaymodel",
            "camclay",
            "mcc",
        ):
            self.material = FiniteStrainModifiedCamClayModel().initialize_from_kwargs(**kwargs)
        elif material_key == "linearelastic":
            raise ValueError(
                "Direct implicit finite-strain MPM no longer supports material='linearElastic'. "
                "Use material='neoHookean' here, or use the standard infinitesimal-strain MPM material manager for LinearElastic."
            )
        else:
            raise ValueError(f"Unsupported direct implicit finite-strain material: {material}")
        super().__init__(bodies, dirichlet, neumann, **kwargs)
        self.assemble_neumann_step = self.apply_neumann if self.neumann.num > 0 else no_operation
        self.apply_dirichlet_step = self.apply_dirichlet_hash if self.dirichlet.num > 0 else no_operation
        self.add_neumann_energy_step = self.get_neumann_energy if self.neumann.num > 0 else no_operation
        self.is_finite_strain_plastic = bool(getattr(self.material, "is_finite_strain_plastic", False))
        plane_strain_option = kwargs.get(
            "plane_strain",
            self.is_finite_strain_plastic and config.DIM == 2 and not self.is_axisymmetric,
        )
        if not isinstance(plane_strain_option, (bool, np.bool_)):
            raise TypeError("Direct MPM plane_strain must be a boolean")
        requested_plane_strain = bool(plane_strain_option)
        if requested_plane_strain and config.DIM != 2:
            raise ValueError("Direct MPM plane_strain requires dimension=2")
        self.is_plane_strain = bool(requested_plane_strain and not self.is_axisymmetric)
        if self.is_finite_strain_plastic and config.DIM == 2 and not self.is_axisymmetric and not self.is_plane_strain:
            raise ValueError("classical finite-strain DP/VonMises/MCC in 2D requires " "plane_strain=True")
        self.material_dimension = 3 if self.is_axisymmetric or self.is_plane_strain else config.DIM
        self.F0 = ti.Matrix.field(
            self.material_dimension,
            self.material_dimension,
            dtype=ti.f64,
            shape=self.n_particles,
        )
        self.vonMises = ti.field(ti.f64, shape=self.n_particles)
        if self.is_finite_strain_plastic:
            self.material.allocate_state(self.n_particles)

        self.active_dof = 0
        self.integration = kwargs.get("newmark", [0.5, 0.25, 0.5])  # [alpha, beta, gamma]
        self.max_iters = int(kwargs.get("max_iters", 100))
        self.tol = float(kwargs.get("residual", 1e-2))
        self.do_line_search = kwargs.get("line_search", True)
        self.do_ccd = kwargs.get("ccd", True)
        self.project_pd = bool(kwargs.get("project_pd", self.is_finite_strain_plastic)) or self.is_finite_strain_plastic

        self.degree_of_freedom = self.get_degree_of_freedom(**kwargs)
        self.linear_solver_tolerance = float(kwargs.get("linear_solver_tolerance", 1.0e-10))
        self.linear_solver_max_iters = int(
            kwargs.get(
                "linear_solver_max_iters",
                max(500, 5 * self.degree_of_freedom),
            )
        )
        if not np.isfinite(self.linear_solver_tolerance) or self.linear_solver_tolerance <= 0.0:
            raise ValueError("linear_solver_tolerance must be finite and positive")
        if self.max_iters <= 0:
            raise ValueError("Direct implicit MPM max_iters must be positive")
        if not np.isfinite(self.tol) or self.tol < 0.0:
            raise ValueError("Direct implicit MPM correction tolerance must be finite and " "non-negative")
        if self.linear_solver_max_iters <= 0:
            raise ValueError("linear_solver_max_iters must be positive")
        # One deterministic raw block slot is reserved for every possible
        # particle/local-node pair. ``influenced_node`` includes scalar DOFs,
        # whereas BuildTriplet stores dense node blocks, so the block-stencil
        # width is the maximum number of grid nodes influenced by a particle.
        self.stiffness_stencil_width = int(self.shape_func.max_node_per_particle)
        self.stiffness_stencil_stride = self.stiffness_stencil_width * self.stiffness_stencil_width
        self.stiffness_nnz = self.stiffness_stencil_stride * self.n_particles
        active_node_capacity = max(1, int(self.degree_of_freedom / config.DIM))
        support_width = int(self.shape_func.max_node_per_particle_one_axis)
        # Raw slots scale with particles, but the reduced grid matrix has a
        # much tighter topology bound: one node can couple only to nodes in
        # the union of two overlapping particle supports.  Keeping these
        # capacities separate prevents the persistent GPU hash table from
        # allocating two entries for every raw particle/local-pair slot.
        neighbor_block_capacity = (2 * support_width - 1) ** config.DIM
        stiffness_reduced_nnz = min(
            self.stiffness_nnz,
            active_node_capacity * max(0, neighbor_block_capacity - 1),
        )
        self.total_nnz = int(self.stiffness_nnz + self.degree_of_freedom)
        device_reduction = kwargs.get("device_reduction", True)
        if device_reduction is False:
            raise ValueError(
                "device_reduction=False is not a runtime backend: direct "
                "implicit MPM keeps sparse reduction and Krylov data in "
                "Taichi fields on every architecture"
            )
        self.hash_matrix = BuildTriplet(
            dim=config.DIM,
            max_pairs_num=max(1, self.stiffness_nnz),
            max_nonzeros=max(1, stiffness_reduced_nnz),
            max_active_nodes=active_node_capacity,
            symmetric=False,
            solver="BiCGSTAB",
            device_reduction=device_reduction,
        )
        self.prefix_sum_executor = PrefixSumExecutor(self.total_background_grid_num)
        self.node2dof = ti.field(int, shape=self.prefix_sum_executor.get_length())  # node2dof for active nodes
        self.dof2node = ti.field(int, shape=int(self.degree_of_freedom / config.DIM))

        self.energy = ti.field(ti.f64, shape=(), needs_grad=True)  # total energy
        self.rhs = ti.field(ti.f64, shape=self.degree_of_freedom)  # global residual force
        self.volume_force = ti.field(dtype=ti.f64, shape=self.degree_of_freedom)
        self.mass_vec = ti.field(ti.f64, shape=self.degree_of_freedom)  # global mass list
        self.incre_resolution = ti.field(ti.f64, shape=self.degree_of_freedom)  # global displacement increment
        self.grid_disp = ti.field(
            ti.f64, shape=self.degree_of_freedom, needs_grad=True
        )  # global displacement in a substep
        self.grid_disp_temp = ti.field(ti.f64, shape=self.degree_of_freedom)  # global displacement per iteration

    def get_degree_of_freedom(self, **kwargs):
        raise NotImplementedError

    @ti.kernel
    def init_F0(self):
        for i in self.F0:
            self.F0[i] = ti.Matrix.identity(ti.f64, self.material_dimension)

    @ti.kernel
    def traction_p2g(self):
        self.volume_force.fill(0)
        for i in range(self.tractionNum[0]):
            pid = self.traction[i].particleID
            traction = self.particle_traction_force(i)
            for j in range(self.offset[pid]):
                grid_id = self.LnID[pid, j]
                if self.grid[grid_id].m > self.val_lim:
                    dofs = config.DIM * (self.node2dof[grid_id] - 1)
                    shape_fn = self.shape[pid, j]
                    force = traction * shape_fn
                    for d in ti.static(range(config.DIM)):
                        self.volume_force[dofs + d] += force[d]

    @ti.kernel
    def matrix_reset(self):
        for i in self.incre_resolution:
            self.incre_resolution[i] = 0.0
            self.rhs[i] = 0.0

    @ti.kernel
    def compute_mass_list(self, integration: ti.types.vector(3, ti.f64)):
        dt = self.TIdt[None]
        coeff1 = 1.0 / (2.0 * dt * dt * integration[0] * integration[1])
        coeff2 = integration[2] * dt * coeff1
        for grid_id in self.grid:
            grid_mass = self.grid[grid_id].m
            if grid_mass > self.val_lim:
                dofs = config.DIM * (self.node2dof[grid_id] - 1)
                for d in ti.static(range(config.DIM)):
                    self.mass_vec[dofs + d] = coeff1 * grid_mass + coeff2 * self.damping * grid_mass

    @ti.kernel
    def compute_nodal_vel_acc(self):
        for i in self.grid:
            if self.grid[i].m > self.val_lim:
                self.grid[i].v /= self.grid[i].m
                self.grid[i].a /= self.grid[i].m

    @ti.kernel
    def update_nodal_acc(self, integration: ti.types.vector(3, float)):
        dt = self.TIdt[None]
        param1 = 1.0 / (2.0 * dt * dt * integration[0] * integration[1])
        param2 = param1 * dt
        param3 = 0.5 / integration[1] - 1.0
        for grid_id in self.grid:
            if self.grid[grid_id].m > self.val_lim:
                dofs = config.DIM * (self.node2dof[grid_id] - 1)
                disp = ti.Vector([self.grid_disp[dofs + d] for d in ti.static(range(config.DIM))])
                previous_velocity = self.grid[grid_id].v
                previous_acceleration = self.grid[grid_id].a
                self.grid[grid_id].v = (
                    0.5 * integration[2] / integration[0] / integration[1] / dt * disp
                    - (0.5 * integration[2] / integration[0] / integration[1] - 1.0) * previous_velocity
                    - 0.5 * dt * (integration[2] / integration[1] - 2.0) * previous_acceleration
                )
                self.grid[grid_id].a = param1 * disp - param2 * previous_velocity - param3 * previous_acceleration

    @ti.kernel
    def find_active_node(self):
        self.node2dof.fill(0)
        for i in self.grid:
            if self.grid[i].m > self.val_lim:
                self.node2dof[i] = 1

    def set_active_dof(self):
        """Build the compact node map and fail safely on undersized storage.

        ``_set_active_dof`` returns the true active-DOF count but never writes
        beyond ``dof2node``.  The scalar check therefore happens before any
        later assembly can consume a truncated map, without downloading a
        full grid-sized field from CUDA.
        """
        active_dof = int(self._set_active_dof())
        active_nodes = active_dof // config.DIM
        node_capacity = int(self.dof2node.shape[0])
        if active_nodes > node_capacity or active_dof > self.degree_of_freedom:
            raise RuntimeError(
                "Direct implicit MPM active-DOF capacity exceeded: "
                f"required {active_dof} DOFs ({active_nodes} nodes), "
                f"allocated {self.degree_of_freedom} DOFs "
                f"({node_capacity} nodes). Increase scale."
            )
        return active_dof

    @ti.kernel
    def _set_active_dof(self) -> int:
        for i in self.grid:
            if self.grid[i].m > self.val_lim:
                rowth = self.node2dof[i] - 1
                if 0 <= rowth < self.dof2node.shape[0]:
                    self.dof2node[rowth] = i
        return config.DIM * self.node2dof[self.node2dof.shape[0] - 1]

    @ti.func
    def get_displacement_incre(self, i, grid_disp):
        gradu = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
        for j in range(self.offset[i]):
            grid_id = self.LnID[i, j]
            dofs = config.DIM * (self.node2dof[grid_id] - 1)
            gradu += ti.Vector([grid_disp[dofs + d] for d in ti.static(range(config.DIM))]).outer_product(
                self.dshape[i, j]
            )
        return gradu

    @ti.func
    def get_axisymmetric_incremental_map(self, i, grid_disp):
        """Return the 3D no-swirl incremental map of a 2D meridian."""
        incremental = ti.Matrix.identity(ti.f64, 3)
        radial_displacement = 0.0
        for j in range(self.offset[i]):
            grid_id = self.LnID[i, j]
            dofs = 2 * (self.node2dof[grid_id] - 1)
            displacement = ti.Vector([grid_disp[dofs + d] for d in ti.static(range(2))])
            for spatial, material_axis in ti.static(ti.ndrange(2, 2)):
                incremental[spatial, material_axis] += displacement[spatial] * self.dshape[i, j][material_axis]
            radial_displacement += self.shape[i, j] * displacement[0]
        radius = self.particle[i].x[0] - ti.static(self.axis_offset)
        incremental[2, 2] += radial_displacement / ti.max(radius, 1.0e-30)
        return incremental

    @ti.func
    def get_plane_strain_incremental_map(self, particle_id, grid_disp):
        """Embed a 2D displacement gradient in the classical 3D plane strain map."""
        incremental = ti.Matrix.identity(ti.f64, 3)
        for local_id in range(self.offset[particle_id]):
            grid_id = self.LnID[particle_id, local_id]
            dofs = 2 * (self.node2dof[grid_id] - 1)
            displacement = ti.Vector([grid_disp[dofs + component] for component in ti.static(range(2))])
            for spatial, material_axis in ti.static(ti.ndrange(2, 2)):
                incremental[spatial, material_axis] += (
                    displacement[spatial] * self.dshape[particle_id, local_id][material_axis]
                )
        return incremental

    @ti.func
    def axisymmetric_dF_du(self, particle_id, local_id, component):
        """Derivative of ``F = f F_n`` with respect to one meridian DOF."""
        derivative_incremental = ti.Matrix.zero(ti.f64, 3, 3)
        for material_axis in ti.static(range(2)):
            derivative_incremental[component, material_axis] = self.dshape[particle_id, local_id][material_axis]
        if component == 0:
            radius = self.particle[particle_id].x[0] - ti.static(self.axis_offset)
            derivative_incremental[2, 2] = self.shape[particle_id, local_id] / ti.max(radius, 1.0e-30)
        return derivative_incremental @ self.F0[particle_id]

    @ti.func
    def plane_strain_dF_du(self, particle_id, local_id, component):
        """Derivative of the embedded plane-strain ``F = f F_n`` map."""
        derivative_incremental = ti.Matrix.zero(ti.f64, 3, 3)
        for material_axis in ti.static(range(2)):
            derivative_incremental[component, material_axis] = self.dshape[particle_id, local_id][material_axis]
        return derivative_incremental @ self.F0[particle_id]

    @ti.func
    def axisymmetric_local_stiffness(self, particle_id, local_i, local_j, tangent):
        block = ti.Matrix.zero(ti.f64, 2, 2)
        for component_i, component_j in ti.static(ti.ndrange(2, 2)):
            derivative_i = self.axisymmetric_dF_du(particle_id, local_i, component_i)
            derivative_j = self.axisymmetric_dF_du(particle_id, local_j, component_j)
            value = 0.0
            for column_i, row_i, column_j, row_j in ti.ndrange(3, 3, 3, 3):
                value += (
                    derivative_i[row_i, column_i]
                    * tangent[row_i + 3 * column_i, row_j + 3 * column_j]
                    * derivative_j[row_j, column_j]
                )
            block[component_i, component_j] = value
        return block

    @ti.func
    def plane_strain_local_stiffness(self, particle_id, local_i, local_j, tangent):
        block = ti.Matrix.zero(ti.f64, 2, 2)
        for component_i, component_j in ti.static(ti.ndrange(2, 2)):
            derivative_i = self.plane_strain_dF_du(particle_id, local_i, component_i)
            derivative_j = self.plane_strain_dF_du(particle_id, local_j, component_j)
            value = 0.0
            for column_i, row_i, column_j, row_j in ti.ndrange(3, 3, 3, 3):
                value += (
                    derivative_i[row_i, column_i]
                    * tangent[row_i + 3 * column_i, row_j + 3 * column_j]
                    * derivative_j[row_j, column_j]
                )
            block[component_i, component_j] = value
        return block

    @ti.kernel
    def assemble_inertia_force(
        self,
        active_dof: int,
        damping: ti.f64,
        gravity: ti.types.vector(config.DIM, float),
        integration: ti.types.vector(3, float),
        grid_disp: ti.template(),
    ):
        dt = self.TIdt[None]
        param1 = 1.0 / (2.0 * dt * dt * integration[0] * integration[1])
        param2 = param1 * dt
        param3 = 0.5 / integration[1] - 1.0
        for grid_id in self.grid:
            nodal_mass = self.grid[grid_id].m
            if nodal_mass > self.val_lim:
                dofs = config.DIM * (self.node2dof[grid_id] - 1)
                residual_force = nodal_mass * gravity + ti.Vector(
                    [self.volume_force[dofs + d] for d in ti.static(range(config.DIM))]
                )
                if ti.static(config.DYNAMIC):
                    disp = ti.Vector([grid_disp[dofs + d] for d in ti.static(range(config.DIM))])
                    previous_velocity = self.grid[grid_id].v
                    previous_acceleration = self.grid[grid_id].a
                    grid_v = (
                        0.5 * integration[2] / integration[0] / integration[1] / dt * disp
                        - (0.5 * integration[2] / integration[0] / integration[1] - 1.0) * previous_velocity
                        - 0.5 * dt * (integration[2] / integration[1] - 2.0) * previous_acceleration
                    )
                    grid_a = param1 * disp - param2 * previous_velocity - param3 * previous_acceleration
                    residual_force += -nodal_mass * (grid_a + damping * grid_v)
                for d in ti.static(range(config.DIM)):
                    self.rhs[dofs + d] = residual_force[d]

    @ti.kernel
    def get_material_energy(self, grid_disp: ti.template()):
        raise NotImplementedError

    @ti.kernel
    def get_inertia_energy(
        self,
        damping: ti.f64,
        integration: ti.types.vector(3, float),
        gravity: ti.types.vector(config.DIM, ti.f64),
        grid_disp: ti.template(),
    ):
        dt = self.TIdt[None]
        for grid_id in range(self.grid.shape[0]):
            if self.grid[grid_id].m > self.val_lim:
                dofs = config.DIM * (self.node2dof[grid_id] - 1)
                disp = ti.Vector([grid_disp[dofs + d] for d in ti.static(range(config.DIM))])
                nodal_mass = self.grid[grid_id].m
                total_energy = -(
                    nodal_mass * gravity
                    + ti.Vector([self.volume_force[dofs + d] for d in ti.static(range(config.DIM))])
                ).dot(disp)
                if ti.static(config.DYNAMIC):
                    param1 = 1.0 / (2.0 * dt * dt * integration[0] * integration[1])
                    param2 = param1 * dt
                    param3 = 0.5 / integration[1] - 1.0
                    previous_velocity = self.grid[grid_id].v
                    previous_acceleration = self.grid[grid_id].a
                    vel = (
                        0.5 * integration[2] / integration[0] / integration[1] / dt * disp
                        - (0.5 * integration[2] / integration[0] / integration[1] - 1.0) * previous_velocity
                        - 0.5 * dt * (integration[2] / integration[1] - 2.0) * previous_acceleration
                    )
                    inertia_energy = (
                        0.5
                        * nodal_mass
                        * disp.dot(
                            param1 * disp - 2.0 * param2 * previous_velocity - 2.0 * param3 * previous_acceleration
                        )
                    )
                    velocity_factor = 0.5 * integration[2] / integration[0] / integration[1] / dt
                    # Integrate the damping force c*m*v_new with respect to displacement.
                    damping_energy = damping * nodal_mass * disp.dot(vel - 0.5 * velocity_factor * disp)
                    total_energy += inertia_energy + damping_energy
                self.energy[None] += total_energy

    @ti.kernel
    def get_neumann_energy(self, grid_disp: ti.template()):
        for i in self.neumann.node:
            node_dof = self.neumann.node[i]
            node_id = int(node_dof // config.DIM)
            if self.grid[node_id].m > self.val_lim:
                dofs = config.DIM * (self.node2dof[node_id] - 1) + int(node_dof % config.DIM)
                external_force = self.neumann.value[i]
                self.energy[None] -= grid_disp[dofs] * external_force

    @ti.func
    def local_stiffness(self, dF_dx1, dF_dx2, d2Psi_dF2):
        H = ti.Matrix.zero(float, config.DIM, config.DIM)
        for i in range(config.DIM):
            for t in range(config.DIM):
                for j in range(config.DIM):
                    for n in range(config.DIM):
                        val1 = d2Psi_dF2[j * config.DIM + i, n * config.DIM + t]
                        val2 = dF_dx1[j]
                        val3 = dF_dx2[n]
                        H[i, t] += val1 * val2 * val3
        return H

    def assemble_mass_matrix(self, prefix: int):
        self.assemble_mass_matrix_hash()

    @ti.kernel
    def assemble_mass_matrix_hash(self):
        for grid_id in self.grid:
            if self.grid[grid_id].m > 0.0:
                block = self.node2dof[grid_id] - 1
                if block >= 0:
                    dofs = config.DIM * block
                    for d in ti.static(range(config.DIM)):
                        self.hash_matrix.diag[block][d * config.DIM + d] += self.mass_vec[dofs + d]

    @ti.func
    def add_hash_block_entry(self, block_i, block_j, block):
        if block_i >= 0 and block_j >= 0:
            if block_i == block_j:
                for d1 in ti.static(range(config.DIM)):
                    for d2 in ti.static(range(config.DIM)):
                        self.hash_matrix.diag[block_i][d1 * config.DIM + d2] += block[d1, d2]
            else:
                idx = ti.atomic_add(self.hash_matrix.raw_non_diag_count[0], 1)
                if idx < self.hash_matrix.non_diag.blockI.shape[0]:
                    self.hash_matrix.non_diag.blockI[idx] = block_i
                    self.hash_matrix.non_diag.blockJ[idx] = block_j
                    for d1 in ti.static(range(config.DIM)):
                        for d2 in ti.static(range(config.DIM)):
                            self.hash_matrix.non_diag.blockH[idx][d1 * config.DIM + d2] = block[d1, d2]
                else:
                    self.hash_matrix.overflow[0] = 1

    @ti.func
    def invalidate_particle_stiffness_slots(self, particle_id):
        """Mark every reserved local-pair slot as absent for this particle."""
        base = particle_id * self.stiffness_stencil_stride
        for local_slot in range(self.stiffness_stencil_stride):
            raw_index = base + local_slot
            if raw_index < self.hash_matrix.non_diag.blockI.shape[0]:
                self.hash_matrix.non_diag.blockI[raw_index] = -1
                self.hash_matrix.non_diag.blockJ[raw_index] = -1
            else:
                self.hash_matrix.overflow[0] = 1

    @ti.func
    def add_fixed_stiffness_block_entry(self, raw_index, block_i, block_j, block):
        """Write a body-stiffness block into its deterministic raw slot.

        Diagonal contributions stay in ``diag`` and leave the reserved raw
        slot invalid. Off-diagonal slots are written without an append atomic;
        ``raw_index`` is uniquely owned by one particle/local-pair tuple.
        """
        if raw_index < self.hash_matrix.non_diag.blockI.shape[0]:
            if block_i >= 0 and block_j >= 0:
                if block_i == block_j:
                    for d1 in ti.static(range(config.DIM)):
                        for d2 in ti.static(range(config.DIM)):
                            ti.atomic_add(
                                self.hash_matrix.diag[block_i][d1 * config.DIM + d2],
                                block[d1, d2],
                            )
                else:
                    self.hash_matrix.non_diag.blockI[raw_index] = block_i
                    self.hash_matrix.non_diag.blockJ[raw_index] = block_j
                    for d1 in ti.static(range(config.DIM)):
                        for d2 in ti.static(range(config.DIM)):
                            self.hash_matrix.non_diag.blockH[raw_index][d1 * config.DIM + d2] = block[d1, d2]
        else:
            self.hash_matrix.overflow[0] = 1

    def apply_dirichlet(self, prefix: int, active_dof: int):
        self.apply_dirichlet_hash(active_dof)

    @ti.kernel
    def apply_dirichlet_hash(self, active_dof: int):
        active_nodes = active_dof // config.DIM
        for block in range(active_nodes):
            base_grid = self.dof2node[block]
            for d1 in ti.static(range(config.DIM)):
                row = config.DIM * block + d1
                rdof_id = config.DIM * base_grid + d1
                for d2 in ti.static(range(config.DIM)):
                    col = config.DIM * block + d2
                    cdof_id = config.DIM * base_grid + d2
                    h_index = d1 * config.DIM + d2
                    value = self.hash_matrix.diag[block][h_index]
                    if self.dirichlet.node[cdof_id] == 1 or self.dirichlet.node[rdof_id] == 1:
                        fixed_correction = self.dirichlet.value[cdof_id] - self.grid_disp[col]
                        self.rhs[row] -= value * fixed_correction
                        self.hash_matrix.diag[block][h_index] = 0.0
                if self.dirichlet.node[rdof_id] == 1:
                    self.hash_matrix.diag[block][d1 * config.DIM + d1] = 1.0

        raw_nnz = self.hash_matrix.raw_non_diag_count[0]
        for k in range(raw_nnz):
            bi = self.hash_matrix.non_diag.blockI[k]
            bj = self.hash_matrix.non_diag.blockJ[k]
            if 0 <= bi < active_nodes and 0 <= bj < active_nodes:
                rgrid = self.dof2node[bi]
                cgrid = self.dof2node[bj]
                for d1 in ti.static(range(config.DIM)):
                    row = config.DIM * bi + d1
                    rdof_id = config.DIM * rgrid + d1
                    for d2 in ti.static(range(config.DIM)):
                        cdof_id = config.DIM * cgrid + d2
                        h_index = d1 * config.DIM + d2
                        value = self.hash_matrix.non_diag.blockH[k][h_index]
                        if self.dirichlet.node[cdof_id] == 1 or self.dirichlet.node[rdof_id] == 1:
                            col = config.DIM * bj + d2
                            fixed_correction = self.dirichlet.value[cdof_id] - self.grid_disp[col]
                            self.rhs[row] -= value * fixed_correction
                            self.hash_matrix.non_diag.blockH[k][h_index] = 0.0

        for i in range(active_dof):
            dof_id = config.DIM * self.dof2node[int(i // config.DIM)] + int(i % config.DIM)
            if self.dirichlet.node[dof_id] == 1:
                self.rhs[i] = self.dirichlet.value[dof_id] - self.grid_disp[i]

    def solve_hash_system(self, active_dof):
        active_dof = int(active_dof)
        active_nodes = active_dof // config.DIM
        self.hash_matrix.finalize_taichi_assembly()

        result = self.hash_matrix.solve_flat_system(
            self.rhs,
            self.incre_resolution,
            active_nodes=active_nodes,
            tol=self.linear_solver_tolerance,
            maxiter=self.linear_solver_max_iters,
            return_solution=False,
        )
        result["backend"] = "taichi_bicgstab"
        if not result["converged"]:
            raise RuntimeError(
                "Direct implicit MPM Taichi BiCGSTAB did not converge: "
                f"residual={result['residual']:.6e}, "
                f"iterations={result['iterations']}"
            )
        return result

    @ti.kernel
    def apply_neumann(self):
        for i in self.neumann.node:
            node_dof = self.neumann.node[i]
            node_id = int(node_dof // config.DIM)
            # Inactive nodes inherit the preceding active node's prefix sum.
            if self.grid[node_id].m > self.val_lim:
                dofs = config.DIM * (self.node2dof[node_id] - 1) + int(node_dof % config.DIM)
                self.rhs[dofs] += self.neumann.value[i]

    @ti.kernel
    def compute_particle_disp(self, grid_disp: ti.template()) -> ti.f64:
        max_disp = 0.0
        for i in range(self.particleNum[0]):
            disp = ti.Vector.zero(ti.f64, config.DIM)
            for j in range(self.offset[i]):
                grid_id = self.LnID[i, j]
                dofs = config.DIM * (self.node2dof[grid_id] - 1)
                disp += self.shape[i, j] * ti.Vector([grid_disp[dofs + d] for d in ti.static(range(config.DIM))])
            ti.atomic_max(max_disp, disp.norm())
        return max_disp

    @ti.kernel
    def calc_g0(self, active_dof: int) -> ti.f64:
        g = 0.0
        for i in range(active_dof):
            g += self.incre_resolution[i] * self.rhs[i]
        return g

    @ti.kernel
    def update_grid_disp(self, active_dof: int, alpha: float):
        for i in range(active_dof):
            self.grid_disp_temp[i] = self.grid_disp[i] + alpha * self.incre_resolution[i]

    @ti.kernel
    def material_ccd(self, slackness: ti.f64) -> ti.f64:
        raise NotImplementedError

    def total_energy(self, grid_disp):
        self.energy[None] = 0.0
        self.get_material_energy(grid_disp)
        self.get_inertia_energy(self.damping, self.integration, self.gravity, grid_disp)
        self.add_neumann_energy_step(grid_disp)
        return self.energy[None]

    def ccd(self, active_dof):
        if self.do_ccd:
            slackness_m = 0.8
            alpha_material = self.material_ccd(slackness_m)
            return alpha_material
        else:
            return 1.0

    def line_search(self, active_dof, verbose=False):
        g0 = -self.calc_g0(active_dof)
        assert g0 <= 0, f"Warning: Not a descent direction! g0: {g0}"
        alpha = self.ccd(active_dof)
        self.update_grid_disp(active_dof, alpha)
        if self.do_line_search:
            previous_energy = self.total_energy(self.grid_disp)
            while alpha > 1e-12:
                current_energy = self.total_energy(self.grid_disp_temp)
                if current_energy <= previous_energy + 1e-5 * alpha * g0:
                    break
                alpha *= 0.5
                if verbose:
                    print(
                        f"current alpha: {alpha}, current energy: {current_energy}, previous energy: {previous_energy}"
                    )
                self.update_grid_disp(active_dof, alpha)
        copy_field(active_dof, self.grid_disp, self.grid_disp_temp)
        return False if alpha < 1e-12 else True

    def auto_assemble_residual_force(self, active_dof, grid_disp, energy_gradient):
        with ti.ad.Tape(loss=self.energy):
            self.total_energy(grid_disp)
        copy_grad(active_dof, energy_gradient, grid_disp.grad)

    def initial_simulation(self):
        self.init_F0()
        super().initial_simulation()

    def calculate_von_mises(self):
        particle_num = self.particleNum.to_numpy()[0]

        @ti.kernel
        def visualize_stress():
            for i in range(self.particleNum[0]):
                deformation_gradient = self.F0[i]
                if ti.static(self.is_finite_strain_plastic):
                    self.vonMises[i] = self.material.total_von_mises_at(i, deformation_gradient)
                else:
                    self.vonMises[i] = self.material.VonMises(deformation_gradient)

        visualize_stress()
        return self.vonMises.to_numpy()[:particle_num]
