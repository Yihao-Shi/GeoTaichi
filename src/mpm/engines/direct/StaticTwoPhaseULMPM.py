import os
import math
import numpy as np
import taichi as ti

import src.mpm.config as config
from src.mpm.engines.direct.MPMSolver import MPMSolver
from src.mpm.boundaries.BoundaryCondition import DirichletBoundary, NeumannBoundary
from src.linear_solver.BuildTriplet import BuildTriplet
from src.utils.PrefixSum import PrefixSumExecutor
from src.utils.SolverConsole import print_save_file_info
from src.utils.linalg import no_operation
from third_party.pyevtk.hl import pointsToVTK


@ti.data_oriented
class StaticTwoPhaseULMPM(MPMSolver):
    def __init__(self, bodies, dirichlet=None, neumann=None, name="case", **kwargs):
        if config.DIM not in (2, 3):
            raise NotImplementedError("StaticTwoPhaseULMPM currently supports 2D and 3D only.")

        self.component = config.DIM + 1
        self.pressure_component = config.DIM
        self.young_modulus = kwargs.get("young_modulus")
        self.poisson_ratio = kwargs.get("poisson_ratio")
        self.density = kwargs.get("density")
        self.material_model = kwargs.get("material", "linearElastic")
        self.shape_function_name = kwargs.get("shape_function", "linear")
        if self.material_model not in ("linearElastic", "neoHookean", "druckerPrager"):
            raise ValueError("StaticTwoPhaseULMPM material must be 'linearElastic', 'neoHookean', or 'druckerPrager'.")
        self.fluid_density = kwargs.get("fluid_density", 1.0)
        self.phi0 = kwargs.get("porosity", kwargs.get("phi", 0.5))
        self.mobility = kwargs.get("mobility", kwargs.get("mobility_constant", 0.0))
        self.ppp_stabilization = kwargs.get("pressure_stabilization", True)
        self.formulation = kwargs.get("formulation", "static").lower()
        if self.formulation not in ("static", "dynamic"):
            raise ValueError("StaticTwoPhaseULMPM formulation must be 'static' or 'dynamic'.")
        self.dynamic_formulation = self.formulation == "dynamic"
        self.max_iters = kwargs.get("max_iters", 20)
        self.tol = kwargs.get("residual", 1e-8)
        self.rhs_tol_abs = kwargs.get("rhs_tolerance_abs", 1e-8)
        self.rhs_tol_rel = kwargs.get("rhs_tolerance_rel", 1e-6)
        self.line_search = kwargs.get("line_search", False)
        self.line_search_max_backtrack = kwargs.get("line_search_max_backtrack", 8)
        self.line_search_beta = kwargs.get("line_search_beta", 0.5)
        requested_solver = (
            str(kwargs.get("linear_solver", "BiCGSTAB")).strip().replace("_", "").replace("-", "").lower()
        )
        if requested_solver in ("bicg", "bicgstab", "taichi", "device"):
            self.linear_solver = "BiCGSTAB"
        elif requested_solver in ("scipy", "direct", "spsolve"):
            self.linear_solver = "Scipy"
        elif requested_solver in ("gmres", "minres"):
            self.linear_solver = requested_solver
        else:
            raise ValueError(
                "StaticTwoPhaseULMPM linear_solver must be BiCGSTAB or "
                "an explicitly selected Scipy/direct/gmres/minres solver"
            )
        self.solve_host_linear_system = no_operation
        self.solve_reduced_krylov_system = no_operation
        if self.linear_solver == "Scipy":
            self.solve_host_linear_system = self.solve_reduced_direct
        elif self.linear_solver == "gmres":
            self.solve_host_linear_system = self.solve_reduced_iterative
            self.solve_reduced_krylov_system = self._solve_reduced_gmres
        elif self.linear_solver == "minres":
            self.solve_host_linear_system = self.solve_reduced_iterative
            self.solve_reduced_krylov_system = self._solve_reduced_minres
        self.direct_regularization = kwargs.get("direct_regularization", 0.0)
        self.iterative_regularization = kwargs.get("iterative_regularization", self.direct_regularization)
        self.iterative_preconditioner = kwargs.get("iterative_preconditioner", "diag")
        self.iterative_rtol = kwargs.get("iterative_rtol", 1.0e-8)
        self.iterative_atol = kwargs.get("iterative_atol", 1.0e-10)
        self.iterative_maxiter = kwargs.get("iterative_maxiter", 500)
        self.ppd = kwargs.get("particles_per_dir", kwargs.get("ppd", 2))
        self.dp_friction_angle = kwargs.get("friction_angle", 35.0)
        self.dp_dilation_angle = kwargs.get("dilation_angle", self.dp_friction_angle)
        self.dp_cohesion = kwargs.get("cohesion", 0.0)
        self.dp_shape_factor = kwargs.get("shape_factor", 0.0)
        self.dp_local_tol = kwargs.get("dp_local_tolerance", 1.0e-10)
        self.dp_local_max_iters = kwargs.get("dp_local_max_iters", 25)
        newmark = kwargs.get("newmark", [0.5, 0.25, 0.5])
        if len(newmark) < 3:
            raise ValueError("newmark must provide at least three entries [alpha, beta, gamma].")
        self.newmark_beta = float(newmark[1])
        self.newmark_gamma = float(newmark[2])
        if self.newmark_beta <= 0.0:
            raise ValueError("Newmark beta must be positive.")

        self.lambda_ = (
            self.young_modulus * self.poisson_ratio / ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio))
        )
        self.mu_ = 0.5 * self.young_modulus / (1.0 + self.poisson_ratio)
        self.bulk_ = self.lambda_ + 2.0 * self.mu_ / 3.0
        self.solid_fraction0 = 1.0 - self.phi0
        self.material_id = (
            0 if self.material_model == "linearElastic" else (1 if self.material_model == "neoHookean" else 2)
        )

        material = type("LinearElasticMaterial", (), {"density": self.density})()
        self.material = material

        user_dirichlet = dirichlet
        user_neumann = neumann
        super().__init__(bodies, None, None, name=name, solver="Implicit", **kwargs)
        self.max_support = self.shape_func.max_node_per_particle

        total_dof = self.component * self.total_background_grid_num
        if user_dirichlet is not None:
            self.dirichlet = user_dirichlet
            self.dirichlet.finalize(total_dof)
        else:
            self.dirichlet = DirichletBoundary()
        if user_neumann is not None:
            self.neumann = user_neumann
            self.neumann.finalize()
        else:
            self.neumann = NeumannBoundary()
        self.assemble_neumann_step = self.apply_neumann if self.neumann.num > 0 else no_operation
        self.apply_dirichlet_step = self.apply_dirichlet_hash if self.dirichlet.num > 0 else no_operation

        self.active_dof = 0
        self.degree_of_freedom = int(kwargs.get("scale", 1.2) * self.component * self.total_background_grid_num)
        self.prefix_sum_executor = PrefixSumExecutor(self.total_background_grid_num)
        self.node2dof = ti.field(ti.i32, shape=self.prefix_sum_executor.get_length())
        self.dof2node = ti.field(ti.i32, shape=int(self.degree_of_freedom / self.component))
        self.supported_node = ti.field(ti.i32, shape=self.total_background_grid_num)
        self.triplets_per_particle = self.max_support * self.max_support * self.component * self.component
        self.stiffness_nnz = self.triplets_per_particle * self.n_particles
        self.block_triplets_per_particle = self.max_support * self.max_support
        self.stiffness_block_nnz = self.block_triplets_per_particle * self.n_particles
        active_node_capacity = max(1, int(self.degree_of_freedom / self.component))
        support_width = int(
            getattr(
                self.shape_func,
                "max_node_per_particle_one_axis",
                int(np.ceil(self.max_support ** (1.0 / config.DIM))),
            )
        )
        neighbor_blocks = (2 * support_width - 1) ** config.DIM
        reduced_block_nnz = min(
            self.stiffness_block_nnz,
            active_node_capacity * max(0, neighbor_blocks - 1),
        )
        self.hash_matrix = BuildTriplet(
            dim=self.component,
            max_pairs_num=max(1, self.stiffness_block_nnz),
            max_nonzeros=max(1, reduced_block_nnz),
            max_active_nodes=active_node_capacity,
            symmetric=False,
            solver="BiCGSTAB",
            device_reduction=True,
        )

        projection_nnz = max(1, self.max_support * self.max_support * self.n_particles)
        projection_reduced_nnz = min(
            projection_nnz,
            self.total_background_grid_num * max(0, neighbor_blocks - 1),
        )
        self.pressure_projection_matrix = BuildTriplet(
            dim=1,
            max_pairs_num=projection_nnz,
            max_nonzeros=max(1, projection_reduced_nnz),
            max_active_nodes=self.total_background_grid_num,
            symmetric=False,
            solver="BiCGSTAB",
            device_reduction=True,
        )

        self.rhs = ti.field(ti.f64, shape=self.degree_of_freedom)
        self.increment = ti.field(ti.f64, shape=self.degree_of_freedom)
        self.old_solution = ti.field(ti.f64, shape=self.degree_of_freedom)
        self.new_solution = ti.field(ti.f64, shape=self.degree_of_freedom)
        self.solution_backup = ti.field(ti.f64, shape=self.degree_of_freedom)
        self.linear_rhs = ti.field(ti.f64, shape=self.degree_of_freedom)

        self.shape_avg = ti.field(ti.f64, shape=(self.n_particles, self.max_support))
        self.dshape_ref = ti.Vector.field(config.DIM, ti.f64, shape=(self.n_particles, self.max_support))
        self.lp = ti.Vector.field(config.DIM, ti.f64, shape=self.n_particles)

        self.F_old = ti.Matrix.field(config.DIM, config.DIM, dtype=ti.f64, shape=self.n_particles)
        self.F_new = ti.Matrix.field(config.DIM, config.DIM, dtype=ti.f64, shape=self.n_particles)
        self.stress_old = ti.Matrix.field(config.DIM, config.DIM, dtype=ti.f64, shape=self.n_particles)
        self.stress_new = ti.Matrix.field(config.DIM, config.DIM, dtype=ti.f64, shape=self.n_particles)
        self.elastic_strain_old = ti.Matrix.field(config.DIM, config.DIM, dtype=ti.f64, shape=self.n_particles)
        self.elastic_strain_new = ti.Matrix.field(config.DIM, config.DIM, dtype=ti.f64, shape=self.n_particles)
        self.J_old = ti.field(ti.f64, shape=self.n_particles)
        self.J_new = ti.field(ti.f64, shape=self.n_particles)
        self.particle_pressure = ti.field(ti.f64, shape=self.n_particles)
        self.particle_pressure_new = ti.field(ti.f64, shape=self.n_particles)
        self.particle_pressure_rate = ti.field(ti.f64, shape=self.n_particles)
        self.particle_pressure_rate_new = ti.field(ti.f64, shape=self.n_particles)
        self.particle_pressure_accel = ti.field(ti.f64, shape=self.n_particles)
        self.particle_pressure_accel_new = ti.field(ti.f64, shape=self.n_particles)
        self.particle_disp = ti.Vector.field(config.DIM, ti.f64, shape=self.n_particles)
        self.particle_velocity_new = ti.Vector.field(config.DIM, ti.f64, shape=self.n_particles)
        self.particle_acceleration_new = ti.Vector.field(config.DIM, ti.f64, shape=self.n_particles)
        self.vonMises = ti.field(ti.f64, shape=self.n_particles)

        self.pressure_projection = ti.field(ti.f64, shape=self.total_background_grid_num)
        self.pressure_rate_projection = ti.field(ti.f64, shape=self.total_background_grid_num)
        self.pressure_accel_projection = ti.field(ti.f64, shape=self.total_background_grid_num)
        self.pressure_projection_rhs = ti.field(ti.f64, shape=self.total_background_grid_num)
        self.rhs_inf_field = ti.field(ti.f64, shape=())
        self.increment_inf_field = ti.field(ti.f64, shape=())
        self.state_is_valid = ti.field(ti.i32, shape=())
        self.last_iterations = 0
        self.last_delta_inf = math.inf
        self.last_rhs_inf = math.inf
        self.last_rhs_target = math.inf
        self.last_line_search_alpha = 1.0
        self.last_converged = False
        self.require_both_convergence_checks = kwargs.get("require_both_convergence_checks", False)
        self.solve_linear_increment_step = (
            self._solve_device_linear_increment
            if self.linear_solver == "BiCGSTAB"
            else self._solve_host_linear_increment
        )
        self.accept_newton_increment_step = (
            self._accept_line_search_increment if self.line_search else self._accept_full_newton_increment
        )
        self.newton_converged = (
            self._both_newton_checks_converged
            if self.require_both_convergence_checks
            else self._either_newton_check_converged
        )
        self.initialize_particle_domain()
        self.initialize_state()

    @ti.kernel
    def initialize_particle_domain(self):
        lp0 = self.dx / self.ppd / 2.0 - 1.0e-6
        for p in range(self.particleNum[0]):
            lp = ti.Vector.zero(ti.f64, config.DIM)
            for d in ti.static(range(config.DIM)):
                lp[d] = lp0
            self.lp[p] = lp

    @ti.kernel
    def initialize_state(self):
        for p in range(self.particleNum[0]):
            self.F_old[p] = ti.Matrix.identity(ti.f64, config.DIM)
            self.F_new[p] = ti.Matrix.identity(ti.f64, config.DIM)
            self.stress_old[p] = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            self.stress_new[p] = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            self.elastic_strain_old[p] = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            self.elastic_strain_new[p] = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            self.J_old[p] = 1.0
            self.J_new[p] = 1.0
            self.particle_pressure[p] = 0.0
            self.particle_pressure_new[p] = 0.0
            self.particle_pressure_rate[p] = 0.0
            self.particle_pressure_rate_new[p] = 0.0
            self.particle_pressure_accel[p] = 0.0
            self.particle_pressure_accel_new[p] = 0.0
            self.particle_disp[p] = ti.Vector.zero(ti.f64, config.DIM)
            self.particle_velocity_new[p] = self.particle[p].v
            self.particle_acceleration_new[p] = self.particle[p].a

    @ti.func
    def safe_sign(self, x):
        s = 0.0
        if x > 0.0:
            s = 1.0
        elif x < 0.0:
            s = -1.0
        return s

    @ti.func
    def gimp_1d_standard_avg(self, xp, dx, inv_dx, lp):
        left = xp - lp
        right = xp + lp
        left_base = ti.cast(left * inv_dx + 1.0e-8, ti.i32)
        right_base = ti.cast(right * inv_dx + 1.0e-8, ti.i32)

        w = ti.Vector.zero(ti.f64, 3)
        dw = ti.Vector.zero(ti.f64, 3)
        wa = ti.Vector.zero(ti.f64, 3)

        if right_base == left_base:
            base = ti.floor(xp * inv_dx + 1.0e-8)
            fx = xp * inv_dx - base
            w[0] = 1.0 - fx
            w[1] = fx
            dw[0] = -inv_dx
            dw[1] = inv_dx
            wa[0] = 0.5
            wa[1] = 0.5
        else:
            x0 = float(left_base) * dx
            x1 = float(left_base + 1) * dx
            x2 = float(left_base + 2) * dx

            a0 = dx + lp - ti.abs(xp - x0)
            a2 = dx + lp - ti.abs(xp - x2)
            s0 = self.safe_sign(xp - x0)
            s2 = self.safe_sign(xp - x2)

            w[0] = a0 * a0 / (4.0 * dx * lp)
            w[1] = 1.0 - ((xp - x1) * (xp - x1) + lp * lp) / (2.0 * dx * lp)
            w[2] = a2 * a2 / (4.0 * dx * lp)

            dw[0] = -a0 / (2.0 * dx * lp) * s0
            dw[1] = -(xp - x1) / (dx * lp)
            dw[2] = -a2 / (2.0 * dx * lp) * s2

            wa[0] = a0 / (4.0 * lp)
            wa[1] = 0.5
            wa[2] = a2 / (4.0 * lp)
        return left_base, w, dw, wa

    @ti.func
    def bspline_piece_antiderivative(self, xi, piece):
        val = 0.0
        if piece == 0:
            val = xi * xi * xi / 6.0 + 0.75 * xi * xi + 1.125 * xi
        elif piece == 1:
            val = 0.75 * xi - xi * xi * xi / 3.0
        else:
            val = xi * xi * xi / 6.0 - 0.75 * xi * xi + 1.125 * xi
        return val

    @ti.func
    def bspline_interval_integral(self, left, right, seg_left, seg_right, piece):
        a = ti.max(left, seg_left)
        b = ti.min(right, seg_right)
        val = 0.0
        if b > a:
            val = self.bspline_piece_antiderivative(b, piece) - self.bspline_piece_antiderivative(a, piece)
        return val

    @ti.func
    def bspline_1d_average(self, xp, xg, inv_dx, lp):
        half_span = lp * inv_dx
        val = self.shape_func.shapefn(xp, xg, inv_dx, 0)
        if half_span >= 1.0e-12:
            center = (xp - xg) * inv_dx
            left = center - half_span
            right = center + half_span
            integral = 0.0
            integral += self.bspline_interval_integral(left, right, -1.5, -0.5, 0)
            integral += self.bspline_interval_integral(left, right, -0.5, 0.5, 1)
            integral += self.bspline_interval_integral(left, right, 0.5, 1.5, 2)
            val = integral / (2.0 * half_span)
        return val

    @ti.kernel
    def compute_shapefn(self):
        self.offset.fill(0)
        for p in range(self.particleNum[0]):
            if ti.static(self.shape_function_name == "linear"):
                bid = self.particle[p].bodyID
                goffset = self.body[bid].goffset
                grid_num = self.body[bid].grid_num
                xmin = self.body[bid].xmin
                dx = self.body[bid].grid_size
                inv_dx = 1.0 / dx
                pos = self.particle[p].x
                base = ti.cast((pos - xmin) * inv_dx - self.shape_func.offset, ti.i32)

                support = 2**config.DIM
                for count in range(support):
                    offset = self.stencil_offset_from_flat(count, 2)
                    grid_id = base + offset
                    shapefn = ti.Vector.zero(ti.f64, config.DIM)
                    shapefn_grad = ti.Vector.zero(ti.f64, config.DIM)
                    for d in ti.static(range(config.DIM)):
                        xg = xmin[d] + grid_id[d] * dx
                        shapefn[d] = self.shape_func.shapefn(pos[d], xg, inv_dx, 0.0)
                        shapefn_grad[d] = self.shape_func.dshapefn(pos[d], xg, inv_dx, 0.0)

                    N = 1.0
                    dN = ti.Vector.zero(ti.f64, config.DIM)
                    for d in ti.static(range(config.DIM)):
                        N *= shapefn[d]
                    for d in ti.static(range(config.DIM)):
                        grad_comp = shapefn_grad[d]
                        for e in ti.static(range(config.DIM)):
                            if e != d:
                                grad_comp *= shapefn[e]
                        dN[d] = grad_comp

                    linear_grid_id = grid_id[0]
                    stride = grid_num[0]
                    for d in ti.static(range(1, config.DIM)):
                        linear_grid_id += grid_id[d] * stride
                        stride *= grid_num[d]
                    self.LnID[p, count] = linear_grid_id + goffset
                    self.shape[p, count] = N
                    self.shape_avg[p, count] = N
                    self.dshape[p, count] = dN
                    self.dshape_ref[p, count] = dN
                self.offset[p] = support
            elif ti.static(self.shape_function_name == "gimp"):
                xp = self.particle[p].x
                base = ti.Vector.zero(ti.i32, config.DIM)
                w = ti.Matrix.zero(ti.f64, config.DIM, 3)
                dw = ti.Matrix.zero(ti.f64, config.DIM, 3)
                wa = ti.Matrix.zero(ti.f64, config.DIM, 3)
                for d in ti.static(range(config.DIM)):
                    base_d, w_d, dw_d, wa_d = self.gimp_1d_standard_avg(xp[d], self.dx, self.inv_dx, self.lp[p][d])
                    base[d] = base_d
                    for i in ti.static(range(3)):
                        w[d, i] = w_d[i]
                        dw[d, i] = dw_d[i]
                        wa[d, i] = wa_d[i]
                count = 0
                flat_offset = 0
                while flat_offset < 3**config.DIM:
                    offset = self.stencil_offset_from_flat(flat_offset, 3)
                    N = 1.0
                    Navg = 1.0
                    dN = ti.Vector.zero(ti.f64, config.DIM)
                    for d in ti.static(range(config.DIM)):
                        N *= w[d, offset[d]]
                        Navg *= wa[d, offset[d]]
                    for d in ti.static(range(config.DIM)):
                        grad_comp = dw[d, offset[d]]
                        for e in ti.static(range(config.DIM)):
                            if e != d:
                                grad_comp *= w[e, offset[e]]
                        dN[d] = grad_comp
                    if N > self.val_lim or Navg > self.val_lim:
                        node = base + offset
                        grid_id = node[0]
                        stride = self.grid_num[0]
                        for d in ti.static(range(1, config.DIM)):
                            grid_id += node[d] * stride
                            stride *= self.grid_num[d]
                        self.LnID[p, count] = grid_id
                        self.shape[p, count] = N
                        self.shape_avg[p, count] = Navg
                        self.dshape[p, count] = dN
                        self.dshape_ref[p, count] = dN
                        count += 1
                    flat_offset += 1
                self.offset[p] = count
            else:
                bid = self.particle[p].bodyID
                goffset = self.body[bid].goffset
                grid_num = self.body[bid].grid_num
                xmin = self.body[bid].xmin
                dx = self.body[bid].grid_size
                inv_dx = 1.0 / dx
                pos = self.particle[p].x
                base = ti.cast((pos - xmin) * inv_dx - self.shape_func.offset, ti.i32)

                count = 0
                flat_offset = 0
                while flat_offset < 3**config.DIM:
                    offset = self.stencil_offset_from_flat(flat_offset, 3)
                    grid_id = base + offset
                    shapefn = ti.Vector.zero(ti.f64, config.DIM)
                    shapefn_grad = ti.Vector.zero(ti.f64, config.DIM)
                    for d in ti.static(range(config.DIM)):
                        xg = xmin[d] + grid_id[d] * dx
                        shapefn[d] = self.shape_func.shapefn(pos[d], xg, inv_dx, 0)
                        shapefn_grad[d] = self.shape_func.dshapefn(pos[d], xg, inv_dx, 0)

                    N = 1.0
                    dN = ti.Vector.zero(ti.f64, config.DIM)
                    for d in ti.static(range(config.DIM)):
                        N *= shapefn[d]
                    for d in ti.static(range(config.DIM)):
                        grad_comp = shapefn_grad[d]
                        for e in ti.static(range(config.DIM)):
                            if e != d:
                                grad_comp *= shapefn[e]
                        dN[d] = grad_comp

                    if N > self.val_lim:
                        linear_grid_id = grid_id[0]
                        stride = grid_num[0]
                        for d in ti.static(range(1, config.DIM)):
                            linear_grid_id += grid_id[d] * stride
                            stride *= grid_num[d]
                        Navg = 1.0
                        for d in ti.static(range(config.DIM)):
                            xg = xmin[d] + grid_id[d] * dx
                            Navg *= self.bspline_1d_average(pos[d], xg, inv_dx, self.lp[p][d])
                        self.LnID[p, count] = linear_grid_id + goffset
                        self.shape[p, count] = N
                        self.shape_avg[p, count] = Navg
                        self.dshape[p, count] = dN
                        self.dshape_ref[p, count] = dN
                        count += 1
                    flat_offset += 1
                self.offset[p] = count

    @ti.kernel
    def grid_reset(self):
        for i in self.grid:
            self.grid[i].m = 0.0

    @ti.kernel
    def mass_p2g(self):
        for p in range(self.particleNum[0]):
            p_mass = self.particle[p].m
            for a in range(self.offset[p]):
                node = self.LnID[p, a]
                self.grid[node].m += self.shape[p, a] * p_mass

    @ti.kernel
    def find_active_node(self):
        self.node2dof.fill(0)
        self.supported_node.fill(0)
        for p in range(self.particleNum[0]):
            for a in range(self.offset[p]):
                node = self.LnID[p, a]
                self.supported_node[node] = 1
        for node in self.grid:
            if self.supported_node[node] == 1:
                self.node2dof[node] = 1

    @ti.kernel
    def set_active_dof(self) -> ti.i32:
        for node in self.grid:
            if self.supported_node[node] == 1:
                row = self.node2dof[node] - 1
                self.dof2node[row] = node
        return self.component * self.node2dof[self.node2dof.shape[0] - 1]

    @ti.func
    def global_index(self, node, comp):
        return self.component * (self.node2dof[node] - 1) + comp

    @ti.func
    def column(self, A, col):
        return ti.Vector([A[i, col] for i in ti.static(range(config.DIM))])

    @ti.func
    def neo_hookean_stress(self, F, J):
        b = F @ F.transpose()
        I = ti.Matrix.identity(ti.f64, config.DIM)
        return self.mu_ / J * (b - I) + self.lambda_ / J * ti.log(J) * I

    @ti.func
    def deviatoric(self, A):
        return A - A.trace() / 3.0 * ti.Matrix.identity(ti.f64, config.DIM)

    @ti.func
    def dp_material_constants(self, angle_deg):
        angle = angle_deg * np.pi / 180.0
        cos_angle = ti.cos(angle)
        sin_angle = ti.sin(angle)
        A = 2.0 * ti.sqrt(6.0) * self.dp_cohesion * cos_angle / (3.0 - sin_angle)
        B = 2.0 * ti.sqrt(6.0) * sin_angle / (3.0 - sin_angle)
        return A, B

    @ti.func
    def dp_yield_value(self, p_mean, t_dev, Af, Bf):
        kf = self.dp_shape_factor * Af
        rf = ti.sqrt(t_dev * t_dev + kf * kf)
        return rf - Af + Bf * p_mean

    @ti.func
    def dp_local_update(self, eps_trial):
        I = ti.Matrix.identity(ti.f64, config.DIM)
        sigma_trial = self.lambda_ * eps_trial.trace() * I + 2.0 * self.mu_ * eps_trial
        p_trial = sigma_trial.trace() / 3.0
        s_trial = self.deviatoric(sigma_trial)
        t_trial = ti.sqrt(ti.max((s_trial * s_trial).sum(), 0.0))

        Af, Bf = self.dp_material_constants(self.dp_friction_angle)
        Ag, Bg = self.dp_material_constants(self.dp_dilation_angle)
        f_trial = self.dp_yield_value(p_trial, t_trial, Af, Bf)

        sigma = sigma_trial
        eps_elastic = eps_trial
        t_dev = t_trial
        delta_lambda = 0.0

        if f_trial > self.dp_local_tol and p_trial < 0.0:
            t_dev = t_trial
            converged = 0
            iteration = 0
            while iteration < self.dp_local_max_iters and iteration < 32 and converged == 0:
                kf = self.dp_shape_factor * Af
                kg = self.dp_shape_factor * Ag
                rf = ti.sqrt(ti.max(t_dev * t_dev + kf * kf, 1.0e-30))
                rg = ti.sqrt(ti.max(t_dev * t_dev + kg * kg, 1.0e-30))
                R1 = t_dev - t_trial + 2.0 * self.mu_ * delta_lambda * t_dev / rg
                R2 = rf - Af + Bf * (p_trial - self.bulk_ * Bg * delta_lambda)
                if ti.abs(R1) < self.dp_local_tol and ti.abs(R2) < self.dp_local_tol:
                    converged = 1
                else:
                    a11 = 1.0 + 2.0 * self.mu_ * delta_lambda * kg * kg / (rg * rg * rg)
                    a12 = 2.0 * self.mu_ * t_dev / rg
                    a21 = t_dev / rf
                    a22 = -Bf * self.bulk_ * Bg
                    det = a11 * a22 - a12 * a21
                    if ti.abs(det) < 1.0e-20:
                        det = 1.0e-20 if det >= 0.0 else -1.0e-20
                    dt = (-a22 * R1 + a12 * R2) / det
                    dl = (a21 * R1 - a11 * R2) / det
                    t_dev += dt
                    delta_lambda += dl
                    t_dev = ti.max(t_dev, 0.0)
                iteration += 1

            p_new = p_trial - self.bulk_ * Bg * delta_lambda
            alpha = 0.0
            if t_trial > 1.0e-20:
                alpha = t_dev / t_trial
            s_new = alpha * s_trial
            sigma = p_new * I + s_new

            kf = self.dp_shape_factor * Af
            kg = self.dp_shape_factor * Ag
            rg = ti.sqrt(ti.max(t_dev * t_dev + kg * kg, 1.0e-30))
            flow = Bg / 3.0 * I
            if rg > 1.0e-20:
                flow += s_new / rg
            eps_elastic = eps_trial - delta_lambda * flow
        return sigma, eps_elastic, sigma_trial, s_trial, p_trial, t_trial, t_dev, delta_lambda

    @ti.func
    def dp_consistent_tangent_apply(self, deps, s_trial, p_trial, t_trial, t_dev, delta_lambda):
        I = ti.Matrix.identity(ti.f64, config.DIM)
        Af, Bf = self.dp_material_constants(self.dp_friction_angle)
        Ag, Bg = self.dp_material_constants(self.dp_dilation_angle)
        kf = self.dp_shape_factor * Af
        kg = self.dp_shape_factor * Ag

        dp_trial = self.bulk_ * deps.trace()
        ds_trial = 2.0 * self.mu_ * self.deviatoric(deps)
        f_trial = self.dp_yield_value(p_trial, t_trial, Af, Bf)
        sigma = self.lambda_ * deps.trace() * I + 2.0 * self.mu_ * deps

        if f_trial > self.dp_local_tol and p_trial < 0.0:
            n_trial = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            dt_trial = 0.0
            alpha = 0.0
            if t_trial > 1.0e-20:
                n_trial = s_trial / t_trial
                dt_trial = (n_trial * ds_trial).sum()
                alpha = t_dev / t_trial

            rf = ti.sqrt(ti.max(t_dev * t_dev + kf * kf, 1.0e-30))
            rg = ti.sqrt(ti.max(t_dev * t_dev + kg * kg, 1.0e-30))
            a11 = 1.0 + 2.0 * self.mu_ * delta_lambda * kg * kg / (rg * rg * rg)
            a12 = 2.0 * self.mu_ * t_dev / rg
            a21 = t_dev / rf
            a22 = -Bf * self.bulk_ * Bg
            det = a11 * a22 - a12 * a21
            if ti.abs(det) < 1.0e-20:
                det = 1.0e-20 if det >= 0.0 else -1.0e-20

            dt_dev_coeff = a22 / det
            dt_vol_coeff = a12 * Bf / det
            dl_dev_coeff = -a21 / det
            dl_vol_coeff = -a11 * Bf / det

            dt = dt_dev_coeff * dt_trial + dt_vol_coeff * dp_trial
            dlam = dl_dev_coeff * dt_trial + dl_vol_coeff * dp_trial
            dp_new = dp_trial - self.bulk_ * Bg * dlam
            ds_new = alpha * ds_trial
            if t_trial > 1.0e-20:
                ds_new += (dt - alpha * dt_trial) * n_trial
            sigma = dp_new * I + ds_new
        return sigma

    @ti.func
    def constitutive_response(self, p, Fnew, Jnew, grad_u):
        deps = 0.5 * (grad_u + grad_u.transpose())
        sigma = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
        if ti.static(self.material_id == 0):
            sigma = (
                self.stress_old[p]
                + self.lambda_ * deps.trace() * ti.Matrix.identity(ti.f64, config.DIM)
                + 2.0 * self.mu_ * deps
            )
            self.elastic_strain_new[p] = self.elastic_strain_old[p]
        elif ti.static(self.material_id == 1):
            sigma = self.neo_hookean_stress(Fnew, Jnew)
            self.elastic_strain_new[p] = self.elastic_strain_old[p]
        else:
            eps_trial = self.elastic_strain_old[p] + deps
            sigma, eps_elastic, _, _, _, _, _, _ = self.dp_local_update(eps_trial)
            self.elastic_strain_new[p] = eps_elastic
        return sigma

    @ti.func
    def constitutive_tangent_apply(self, p, Fnew, Jnew, grad_u, G, comp, eta):
        deps = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
        for d in ti.static(range(config.DIM)):
            deps[comp, d] += 0.5 * G[d]
            deps[d, comp] += 0.5 * G[d]
        sigma = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
        if ti.static(self.material_id == 0):
            sigma = self.lambda_ * deps.trace() * ti.Matrix.identity(ti.f64, config.DIM) + 2.0 * self.mu_ * deps
        elif ti.static(self.material_id == 1):
            H = self.F_old[p].transpose() @ G
            dF = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            for d in ti.static(range(config.DIM)):
                dF[comp, d] = H[d]
            b = Fnew @ Fnew.transpose()
            db = dF @ Fnew.transpose() + Fnew @ dF.transpose()
            dJ = Jnew * eta
            sigma = self.mu_ / Jnew * db - self.mu_ / (Jnew * Jnew) * (b - ti.Matrix.identity(ti.f64, config.DIM)) * dJ
            sigma += self.lambda_ * (1.0 - ti.log(Jnew)) / (Jnew * Jnew) * dJ * ti.Matrix.identity(ti.f64, config.DIM)
        else:
            eps_trial = self.elastic_strain_old[p] + 0.5 * (grad_u + grad_u.transpose())
            _, _, _, s_trial, p_trial, t_trial, t_dev, delta_lambda = self.dp_local_update(eps_trial)
            sigma = self.dp_consistent_tangent_apply(deps, s_trial, p_trial, t_trial, t_dev, delta_lambda)
        return sigma

    @ti.func
    def von_mises_stress(self, stress):
        if ti.static(config.DIM == 2):
            sxx = stress[0, 0]
            syy = stress[1, 1]
            sxy = stress[0, 1]
            return ti.sqrt(ti.max(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))
        tr = stress.trace()
        s = stress - tr / 3.0 * ti.Matrix.identity(ti.f64, config.DIM)
        j2 = 0.5 * (s * s).sum()
        ti.ba
        return ti.sqrt(ti.max(3.0 * j2, 0.0))

    @ti.kernel
    def reset_step_solution(self):
        for i in range(self.active_dof):
            self.old_solution[i] = 0.0
            self.new_solution[i] = 0.0

    @ti.kernel
    def assemble_pressure_projection_matrix_and_rhs(self, particle_values: ti.template()):
        for node in range(self.total_background_grid_num):
            self.pressure_projection_rhs[node] = 0.0
        for p in range(self.particleNum[0]):
            mass = self.particle[p].m
            for a in range(self.offset[p]):
                node_a = self.LnID[p, a]
                shape_a = self.shape[p, a]
                ti.atomic_add(
                    self.pressure_projection_rhs[node_a],
                    mass * shape_a * particle_values[p],
                )
                for b in range(self.offset[p]):
                    node_b = self.LnID[p, b]
                    self.pressure_projection_matrix.add_scalar_entry(
                        node_a,
                        node_b,
                        mass * shape_a * self.shape[p, b],
                        0,
                        0,
                    )

    @ti.kernel
    def assemble_pressure_projection_rhs(self, particle_values: ti.template()):
        for node in range(self.total_background_grid_num):
            self.pressure_projection_rhs[node] = 0.0
        for p in range(self.particleNum[0]):
            mass = self.particle[p].m
            for a in range(self.offset[p]):
                node = self.LnID[p, a]
                ti.atomic_add(
                    self.pressure_projection_rhs[node],
                    mass * self.shape[p, a] * particle_values[p],
                )

    @ti.func
    def pressure_projection_node_is_constrained(self, node):
        constrained = self.supported_node[node] == 0
        if ti.static(self.dirichlet.num > 0):
            dof = self.component * node + self.pressure_component
            constrained = constrained or self.dirichlet.node[dof] == 1
        return constrained

    @ti.kernel
    def apply_pressure_projection_constraints(self):
        for entry in range(self.pressure_projection_matrix.raw_non_diag_count[0]):
            if entry < self.pressure_projection_matrix.non_diag.blockI.shape[0]:
                row = self.pressure_projection_matrix.non_diag.blockI[entry]
                column = self.pressure_projection_matrix.non_diag.blockJ[entry]
                if self.pressure_projection_node_is_constrained(row) or self.pressure_projection_node_is_constrained(
                    column
                ):
                    self.pressure_projection_matrix.non_diag.blockH[entry][0] = 0.0
        for node in range(self.total_background_grid_num):
            if self.pressure_projection_node_is_constrained(node):
                self.pressure_projection_matrix.diag[node][0] = 1.0
                self.pressure_projection_rhs[node] = 0.0

    @ti.kernel
    def apply_pressure_projection_rhs_constraints(self):
        for node in range(self.total_background_grid_num):
            if self.pressure_projection_node_is_constrained(node):
                self.pressure_projection_rhs[node] = 0.0

    def solve_pressure_projection(self, particle_values, nodal_values):
        self.assemble_pressure_projection_rhs(particle_values)
        self.apply_pressure_projection_rhs_constraints()
        result = self.pressure_projection_matrix.solve_flat_system(
            self.pressure_projection_rhs,
            nodal_values,
            active_nodes=self.total_background_grid_num,
            tol=self.iterative_rtol,
            maxiter=self.iterative_maxiter,
            return_solution=False,
        )
        if not result["converged"]:
            raise RuntimeError(
                "StaticTwoPhaseULMPM device pressure projection did not "
                f"converge: residual={result['residual']:.6e}, "
                f"iterations={result['iterations']}"
            )

    def assemble_pressure_projection(self):
        self.pressure_projection_matrix.reset_system()
        self.assemble_pressure_projection_matrix_and_rhs(self.particle_pressure)
        self.apply_pressure_projection_constraints()
        self.pressure_projection_matrix.finalize_taichi_assembly()
        result = self.pressure_projection_matrix.solve_flat_system(
            self.pressure_projection_rhs,
            self.pressure_projection,
            active_nodes=self.total_background_grid_num,
            tol=self.iterative_rtol,
            maxiter=self.iterative_maxiter,
            return_solution=False,
        )
        if not result["converged"]:
            raise RuntimeError(
                "StaticTwoPhaseULMPM device pressure projection did not "
                f"converge: residual={result['residual']:.6e}, "
                f"iterations={result['iterations']}"
            )
        if self.dynamic_formulation:
            self.solve_pressure_projection(
                self.particle_pressure_rate,
                self.pressure_rate_projection,
            )
            self.solve_pressure_projection(
                self.particle_pressure_accel,
                self.pressure_accel_projection,
            )
        else:
            self.pressure_rate_projection.fill(0.0)
            self.pressure_accel_projection.fill(0.0)

    @ti.kernel
    def apply_pressure_projection_to_solution(self):
        for node in self.grid:
            if self.supported_node[node] == 1:
                dof = self.global_index(node, self.pressure_component)
                self.old_solution[dof] = self.pressure_projection[node]
                self.new_solution[dof] = self.pressure_projection[node]

    @ti.kernel
    def zero_vector(self, active_dof: ti.i32, field: ti.template()):
        for i in range(active_dof):
            field[i] = 0.0

    @ti.kernel
    def prepare_device_linear_rhs(self):
        for dof in range(self.active_dof):
            self.linear_rhs[dof] = -self.rhs[dof]

    @ti.kernel
    def reduce_rhs_inf(self):
        self.rhs_inf_field[None] = 0.0
        for dof in range(self.active_dof):
            ti.atomic_max(self.rhs_inf_field[None], ti.abs(self.rhs[dof]))

    @ti.kernel
    def reduce_increment_inf(self):
        self.increment_inf_field[None] = 0.0
        for dof in range(self.active_dof):
            include = 1
            if ti.static(self.dirichlet.num > 0):
                node = self.dof2node[dof // self.component]
                component = dof % self.component
                full_dof = self.component * node + component
                include = ti.cast(self.dirichlet.node[full_dof] == 0, ti.i32)
            if include != 0:
                ti.atomic_max(
                    self.increment_inf_field[None],
                    ti.abs(self.increment[dof]),
                )

    @ti.kernel
    def validate_current_step_state(self):
        self.state_is_valid[None] = 1
        for p in range(self.particleNum[0]):
            valid = (
                self.J_new[p] > 1.0e-12
                and self.J_new[p] < ti.math.inf
                and -ti.math.inf < self.particle_pressure_new[p]
                and self.particle_pressure_new[p] < ti.math.inf
            )
            for row, column in ti.static(ti.ndrange(config.DIM, config.DIM)):
                value = self.stress_new[p][row, column]
                valid = valid and -ti.math.inf < value and value < ti.math.inf
            if not valid:
                self.state_is_valid[None] = 0

    def current_step_state_is_finite_and_positive(self):
        self.validate_current_step_state()
        return bool(self.state_is_valid[None])

    def assemble_current_system_arrays(self):
        self.zero_vector(self.active_dof, self.rhs)
        self.zero_vector(self.active_dof, self.increment)
        self.hash_matrix.reset_system()
        gravity = ti.Vector(self.gravity)
        self.assemble_system(self.dt, gravity)
        self.assemble_neumann_step()
        self.apply_dirichlet_step()
        K = self.current_hash_matrix()
        K = self.sanitize_dirichlet_matrix(K)
        r = self.rhs.to_numpy()[: self.active_dof].copy()
        return K[: self.active_dof, : self.active_dof], r

    def assemble_current_unconstrained_residual(self, include_neumann=True):
        self.zero_vector(self.active_dof, self.rhs)
        self.zero_vector(self.active_dof, self.increment)
        self.hash_matrix.reset_system()
        gravity = ti.Vector(self.gravity)
        self.assemble_system(self.dt, gravity)
        if include_neumann:
            self.assemble_neumann_step()
        return self.rhs.to_numpy()[: self.active_dof].copy()

    def extract_reaction_forces(self, full_dofs, include_neumann=True):
        residual = self.assemble_current_unconstrained_residual(include_neumann=include_neumann)
        dof2node = self.dof2node.to_numpy()
        reactions = np.zeros(len(full_dofs), dtype=np.float64)
        for i, full_dof in enumerate(full_dofs):
            node = int(full_dof // self.component)
            comp = int(full_dof % self.component)
            prefix = self.node2dof[node]
            if prefix <= 0:
                continue
            local_dof = int(prefix - 1) * self.component + comp
            if 0 <= local_dof < self.active_dof:
                active_node = int(dof2node[local_dof // self.component])
                if active_node == node:
                    reactions[i] = residual[local_dof]
        return reactions

    def solve_reduced_direct(self, K, rhs):
        from scipy.sparse import eye
        from scipy.sparse.linalg import spsolve

        active_slice = slice(0, self.active_dof)
        K_active = K[active_slice, active_slice].tocsr()
        rhs_active = rhs[active_slice]
        non_zero_rows = K_active.getnnz(axis=1) != 0
        non_zero_cols = K_active.getnnz(axis=0) != 0
        active_mask = np.logical_or(non_zero_rows, non_zero_cols)
        K_reduced = K_active[active_mask, :][:, active_mask]
        rhs_reduced = rhs_active[active_mask]
        delta_active = np.zeros(self.active_dof, dtype=np.float64)
        if K_reduced.shape[0] > 0:
            if self.direct_regularization > 0.0:
                K_reduced = K_reduced + self.direct_regularization * eye(K_reduced.shape[0], format="csr")
            delta_reduced = spsolve(K_reduced, rhs_reduced)
            delta_active[active_mask] = delta_reduced
        delta = np.zeros(self.degree_of_freedom, dtype=np.float64)
        delta[active_slice] = delta_active
        return delta

    def build_preconditioner(self, K, reduced_dofs=None):
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import LinearOperator, spilu, splu

        if self.iterative_preconditioner == "none":
            return None
        if self.iterative_preconditioner == "diag":
            diag = K.diagonal().copy()
            safe = np.abs(diag) > 1.0e-30
            inv_diag = np.ones_like(diag)
            inv_diag[safe] = 1.0 / diag[safe]
            return LinearOperator(K.shape, matvec=lambda x: inv_diag * x, dtype=np.float64)
        if self.iterative_preconditioner == "block_diag":
            if reduced_dofs is None:
                return None
            comp_ids = reduced_dofs % self.component
            disp_ids = np.where(comp_ids != self.pressure_component)[0]
            pres_ids = np.where(comp_ids == self.pressure_component)[0]
            if disp_ids.size == 0:
                return None
            Kuu = K[disp_ids, :][:, disp_ids].tocsc()
            try:
                kuu_solver = splu(Kuu)
            except Exception:
                try:
                    kuu_solver = spilu(Kuu)
                except Exception:
                    return None
            if pres_ids.size > 0:
                kpp_diag = K[pres_ids, :][:, pres_ids].diagonal().copy()
                safe = np.abs(kpp_diag) > 1.0e-30
                inv_kpp = np.ones_like(kpp_diag)
                inv_kpp[safe] = 1.0 / kpp_diag[safe]
            else:
                inv_kpp = None

            def apply_block(x):
                y = np.zeros_like(x)
                y[disp_ids] = kuu_solver.solve(x[disp_ids])
                if pres_ids.size > 0:
                    y[pres_ids] = inv_kpp * x[pres_ids]
                return y

            return LinearOperator(K.shape, matvec=apply_block, dtype=np.float64)
        if self.iterative_preconditioner == "schur_lu":
            if reduced_dofs is None:
                return None
            comp_ids = reduced_dofs % self.component
            disp_ids = np.where(comp_ids != self.pressure_component)[0]
            pres_ids = np.where(comp_ids == self.pressure_component)[0]
            if disp_ids.size == 0 or pres_ids.size == 0:
                return None
            Kuu = K[disp_ids, :][:, disp_ids].tocsc()
            Kup = K[disp_ids, :][:, pres_ids].tocsc()
            Kpu = K[pres_ids, :][:, disp_ids].tocsc()
            Kpp = K[pres_ids, :][:, pres_ids].tocsc()
            try:
                kuu_solver = splu(Kuu)
            except Exception:
                try:
                    kuu_solver = spilu(Kuu)
                except Exception:
                    return None
            try:
                kup_dense = Kup.toarray()
                kuu_inv_kup = kuu_solver.solve(kup_dense)
                schur_dense = Kpp.toarray() - (Kpu @ kuu_inv_kup)
                schur = csc_matrix(schur_dense)
                schur_solver = splu(schur)
            except Exception:
                schur_diag = schur.diagonal().copy() if "schur" in locals() else Kpp.diagonal().copy()
                safe = np.abs(schur_diag) > 1.0e-30
                inv_schur = np.ones_like(schur_diag)
                inv_schur[safe] = 1.0 / schur_diag[safe]
                schur_solver = None

            def apply_schur(x):
                y = np.zeros_like(x)
                bu = x[disp_ids]
                bp = x[pres_ids]
                u_tilde = kuu_solver.solve(bu)
                schur_rhs = bp - Kpu @ u_tilde
                if schur_solver is not None:
                    xp = schur_solver.solve(schur_rhs)
                else:
                    xp = inv_schur * schur_rhs
                xu = kuu_solver.solve(bu - Kup @ xp)
                y[disp_ids] = xu
                y[pres_ids] = xp
                return y

            return LinearOperator(K.shape, matvec=apply_schur, dtype=np.float64)
        if self.iterative_preconditioner == "ilu":
            try:
                ilu = spilu(K.tocsc())
                return LinearOperator(K.shape, matvec=ilu.solve, dtype=np.float64)
            except Exception:
                return None
        raise ValueError(f"Unknown iterative_preconditioner: {self.iterative_preconditioner}")

    def _solve_reduced_gmres(self, matrix, rhs, preconditioner):
        from scipy.sparse.linalg import gmres

        from src.linear_solver.ScipyKrylov import checked_scipy_krylov

        solution = checked_scipy_krylov(
            gmres,
            matrix,
            rhs,
            M=preconditioner,
            atol=self.iterative_atol,
            rtol=self.iterative_rtol,
            restart=min(200, matrix.shape[0]),
            maxiter=self.iterative_maxiter,
            solver_name="SciPy GMRES",
        )
        return solution, 0

    def _solve_reduced_minres(self, matrix, rhs, preconditioner):
        from scipy.sparse.linalg import minres

        from src.linear_solver.ScipyKrylov import checked_scipy_krylov

        solution = checked_scipy_krylov(
            minres,
            matrix,
            rhs,
            M=preconditioner,
            rtol=self.iterative_rtol,
            atol=None,
            maxiter=self.iterative_maxiter,
            solver_name="SciPy MINRES",
        )
        return solution, 0

    def solve_reduced_iterative(self, K, rhs):
        from scipy.sparse import eye

        active_slice = slice(0, self.active_dof)
        K_active = K[active_slice, active_slice].tocsr()
        rhs_active = rhs[active_slice]
        non_zero_rows = K_active.getnnz(axis=1) != 0
        non_zero_cols = K_active.getnnz(axis=0) != 0
        active_mask = np.logical_or(non_zero_rows, non_zero_cols)
        K_reduced = K_active[active_mask, :][:, active_mask]
        rhs_reduced = rhs_active[active_mask]
        reduced_dofs = np.flatnonzero(active_mask)
        delta_active = np.zeros(self.active_dof, dtype=np.float64)
        if K_reduced.shape[0] > 0:
            if self.iterative_regularization > 0.0:
                K_reduced = K_reduced + self.iterative_regularization * eye(K_reduced.shape[0], format="csr")
            M = self.build_preconditioner(K_reduced, reduced_dofs=reduced_dofs)
            delta_reduced, info = self.solve_reduced_krylov_system(K_reduced, rhs_reduced, M)
            if info != 0:
                raise RuntimeError(f"{self.linear_solver} failed with info={info}")
            delta_active[active_mask] = delta_reduced
        delta = np.zeros(self.degree_of_freedom, dtype=np.float64)
        delta[active_slice] = delta_active
        return delta

    def solve_linear_system(self, K, rhs):
        return self.solve_host_linear_system(K, rhs)

    def sanitize_dirichlet_matrix(self, K):
        if self.dirichlet.num == 0:
            return K
        K = K.tocsr()
        constrained = self.get_active_dirichlet_mask()
        if not np.any(constrained):
            return K
        K = K[: self.active_dof, : self.active_dof].tolil(copy=True)
        ids = np.where(constrained)[0]
        for idx in ids:
            K.rows[idx] = [int(idx)]
            K.data[idx] = [1.0]
        K = K.tocsc()
        for idx in ids:
            start = K.indptr[idx]
            end = K.indptr[idx + 1]
            rows = K.indices[start:end]
            data = K.data[start:end]
            for j in range(rows.shape[0]):
                data[j] = 1.0 if rows[j] == idx else 0.0
        return K.tocsr()

    def get_active_dirichlet_mask(self):
        constrained = np.zeros(self.active_dof, dtype=bool)
        if self.dirichlet.num == 0:
            return constrained
        dirichlet_nodes = self.dirichlet.node.to_numpy()
        dof2node = self.dof2node.to_numpy()
        for i in range(self.active_dof):
            node = int(dof2node[i // self.component])
            comp = int(i % self.component)
            full_dof = self.component * node + comp
            constrained[i] = dirichlet_nodes[full_dof] == 1
        return constrained

    @ti.kernel
    def assemble_system(self, dt: ti.f64, gravity: ti.types.vector(config.DIM, ti.f64)):
        for p in range(self.particleNum[0]):
            grad_u = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            p_new = 0.0
            p_old = 0.0
            p_avg = 0.0
            p_old_avg = 0.0
            p_rate_old = 0.0
            p_rate_old_avg = 0.0
            p_acc_old = 0.0
            p_acc_old_avg = 0.0
            delta_u = ti.Vector.zero(ti.f64, config.DIM)

            for a in range(self.offset[p]):
                node = self.LnID[p, a]
                u = ti.Vector.zero(ti.f64, config.DIM)
                for d in ti.static(range(config.DIM)):
                    u[d] = self.new_solution[self.global_index(node, d)]
                pn = self.new_solution[self.global_index(node, self.pressure_component)]
                po = self.old_solution[self.global_index(node, self.pressure_component)]
                Na = self.shape[p, a]
                Navg = self.shape_avg[p, a]
                Gref = self.dshape_ref[p, a]
                grad_u += u.outer_product(Gref)
                delta_u += Na * u
                p_new += Na * pn
                p_old += Na * po
                p_avg += Navg * pn
                p_old_avg += Navg * po
                if ti.static(self.dynamic_formulation):
                    p_rate_node = self.pressure_rate_projection[node]
                    p_acc_node = self.pressure_accel_projection[node]
                    p_rate_old += Na * p_rate_node
                    p_rate_old_avg += Navg * p_rate_node
                    p_acc_old += Na * p_acc_node
                    p_acc_old_avg += Navg * p_acc_node

            deltaF = ti.Matrix.identity(ti.f64, config.DIM) + grad_u
            deltaF_inv = deltaF.inverse()
            Fnew = deltaF @ self.F_old[p]
            Jold = self.J_old[p]
            Jnew = Fnew.determinant()
            vol = self.particle[p].vol0 * Jnew
            log_ratio = ti.log(ti.max(Jnew, 1.0e-12)) - ti.log(ti.max(Jold, 1.0e-12))
            porosity = 1.0 - self.solid_fraction0 / ti.max(Jnew, 1.0e-12)
            rho_mix = porosity * self.fluid_density + (1.0 - porosity) * self.density
            q = ti.Vector.zero(ti.f64, config.DIM)
            tau = 0.5 / self.mu_

            for a in range(self.offset[p]):
                gcur = deltaF_inv @ self.dshape_ref[p, a]
                node = self.LnID[p, a]
                pn = self.new_solution[self.global_index(node, self.pressure_component)]
                q += pn * gcur

            sigma_eff = self.constitutive_response(p, Fnew, Jnew, grad_u)
            sigma_total = sigma_eff - p_new * ti.Matrix.identity(ti.f64, config.DIM)
            a_new = ti.Vector.zero(ti.f64, config.DIM)
            v_new = ti.Vector.zero(ti.f64, config.DIM)
            p_rate_new = 0.0
            p_acc_new = 0.0
            ppp_gap = (p_new - p_avg) - (p_old - p_old_avg)
            p_rate_gap = ppp_gap
            if ti.static(self.dynamic_formulation):
                a_new = self.newmark_acceleration_from_disp(delta_u, self.particle[p].v, self.particle[p].a, dt)
                v_new = self.newmark_velocity_from_disp(delta_u, self.particle[p].v, self.particle[p].a, dt)
                p_rate_new = self.newmark_pressure_rate(p_new, p_old, p_rate_old, p_acc_old, dt)
                p_rate_new_avg = self.newmark_pressure_rate(p_avg, p_old_avg, p_rate_old_avg, p_acc_old_avg, dt)
                p_acc_new = self.newmark_pressure_accel(p_new, p_old, p_rate_old, p_acc_old, dt)
                p_rate_gap = (p_rate_new - p_rate_new_avg) - (p_rate_old - p_rate_old_avg)

            self.F_new[p] = Fnew
            self.J_new[p] = Jnew
            self.stress_new[p] = sigma_eff
            self.particle_pressure_new[p] = p_new
            self.particle_disp[p] = delta_u
            self.particle_velocity_new[p] = v_new
            self.particle_acceleration_new[p] = a_new
            self.particle_pressure_rate_new[p] = p_rate_new
            self.particle_pressure_accel_new[p] = p_acc_new

            for a in range(self.offset[p]):
                node_a = self.LnID[p, a]
                Na = self.shape[p, a]
                Navg_a = self.shape_avg[p, a]
                gI = deltaF_inv @ self.dshape_ref[p, a]
                row_p = self.global_index(node_a, self.pressure_component)

                base_force = (-sigma_total @ gI + Na * rho_mix * gravity) * vol
                if ti.static(self.dynamic_formulation):
                    base_force -= Na * self.particle[p].m * (a_new + self.damping * v_new)
                for d in ti.static(range(config.DIM)):
                    self.rhs[self.global_index(node_a, d)] += base_force[d]

                darcy = dt * self.mobility * gI.dot(q)
                mass_res = (Na * log_ratio + darcy) * vol
                if ti.static(self.ppp_stabilization):
                    if ti.static(self.dynamic_formulation):
                        mass_res += tau * (Na - Navg_a) * p_rate_gap * vol
                    else:
                        mass_res += tau * (Na - Navg_a) * ppp_gap * vol
                self.rhs[row_p] += mass_res

                for b in range(self.offset[p]):
                    node_b = self.LnID[p, b]
                    Nb = self.shape[p, b]
                    Navg_b = self.shape_avg[p, b]
                    GJ = self.dshape_ref[p, b]
                    gJ = deltaF_inv @ GJ
                    block_i = self.node2dof[node_a] - 1
                    block_j = self.node2dof[node_b] - 1
                    raw_block = self.begin_block_triplet(block_i, block_j)

                    Kpp = dt * self.mobility * gI.dot(gJ) * vol
                    if ti.static(self.ppp_stabilization):
                        if ti.static(self.dynamic_formulation):
                            Kpp += (
                                tau
                                * (self.newmark_gamma / (self.newmark_beta * dt))
                                * (Na - Navg_a)
                                * (Nb - Navg_b)
                                * vol
                            )
                        else:
                            Kpp += tau * (Na - Navg_a) * (Nb - Navg_b) * vol

                    Kup = vol * Nb * gI

                    for comp in ti.static(range(config.DIM)):
                        hcol = self.column(deltaF_inv, comp)
                        eta = GJ.dot(hcol)
                        dgI = -hcol * GJ.dot(gI)
                        dv = vol * eta
                        coeff_rho = (self.fluid_density - self.density) * self.solid_fraction0 / ti.max(Jnew, 1.0e-12)
                        drho = coeff_rho * eta
                        dsigma = self.constitutive_tangent_apply(p, Fnew, Jnew, grad_u, GJ, comp, eta)

                        mech = (-dsigma @ gI - sigma_total @ dgI + Na * drho * gravity) * vol + (
                            -sigma_total @ gI + Na * rho_mix * gravity
                        ) * dv
                        if ti.static(self.dynamic_formulation):
                            mech[comp] -= (
                                self.particle[p].m
                                * Na
                                * Nb
                                * (
                                    1.0 / (self.newmark_beta * dt * dt)
                                    + self.damping * self.newmark_gamma / (self.newmark_beta * dt)
                                )
                            )
                        for row_comp in ti.static(range(config.DIM)):
                            self.add_block_component(block_i, block_j, raw_block, row_comp, comp, mech[row_comp])

                        kpu = Na * (1.0 + log_ratio) * eta * vol
                        scalar_grad_term = (
                            (-GJ.dot(gI) * hcol.dot(q) - GJ.dot(q) * hcol.dot(gI) + gI.dot(q) * eta)
                            * dt
                            * self.mobility
                            * vol
                        )
                        kpu += scalar_grad_term
                        if ti.static(self.ppp_stabilization):
                            if ti.static(self.dynamic_formulation):
                                kpu += tau * (Na - Navg_a) * p_rate_gap * dv
                            else:
                                kpu += tau * (Na - Navg_a) * ppp_gap * dv
                        self.add_block_component(block_i, block_j, raw_block, self.pressure_component, comp, kpu)

                    for row_comp in ti.static(range(config.DIM)):
                        self.add_block_component(
                            block_i, block_j, raw_block, row_comp, self.pressure_component, Kup[row_comp]
                        )
                    self.add_block_component(
                        block_i, block_j, raw_block, self.pressure_component, self.pressure_component, Kpp
                    )

    @ti.func
    def begin_block_triplet(self, block_i, block_j):
        raw_block = -1
        if block_i >= 0 and block_j >= 0:
            if block_i != block_j:
                idx = ti.atomic_add(self.hash_matrix.raw_non_diag_count[0], 1)
                if idx < self.hash_matrix.non_diag.blockI.shape[0]:
                    self.hash_matrix.non_diag.blockI[idx] = block_i
                    self.hash_matrix.non_diag.blockJ[idx] = block_j
                    k = 0
                    while k < self.component * self.component:
                        self.hash_matrix.non_diag.blockH[idx][k] = 0.0
                        k += 1
                    raw_block = idx
                else:
                    self.hash_matrix.overflow[0] = 1
        return raw_block

    @ti.func
    def add_block_component(self, block_i, block_j, raw_block, row_comp, col_comp, value):
        if block_i >= 0 and block_j >= 0:
            h_index = row_comp * self.component + col_comp
            if block_i == block_j:
                ti.atomic_add(self.hash_matrix.diag[block_i][h_index], value)
            elif raw_block >= 0:
                ti.atomic_add(self.hash_matrix.non_diag.blockH[raw_block][h_index], value)

    @ti.kernel
    def apply_neumann(self):
        for i in self.neumann.node:
            full_dof = self.neumann.node[i]
            node = int(full_dof // self.component)
            comp = int(full_dof % self.component)
            if self.node2dof[node] > 0:
                self.rhs[self.global_index(node, comp)] += self.neumann.value[i]

    @ti.kernel
    def apply_dirichlet_solution(self):
        for i in range(self.active_dof):
            node = self.dof2node[int(i // self.component)]
            comp = int(i % self.component)
            full_dof = self.component * node + comp
            if self.dirichlet.node[full_dof] == 1:
                self.rhs[i] = self.new_solution[i] - self.dirichlet.value[full_dof]

    @ti.kernel
    def apply_dirichlet_hash_kernel(self):
        active_nodes = self.active_dof // self.component
        for block in range(active_nodes):
            row_node = self.dof2node[block]
            row_comp = 0
            while row_comp < self.component:
                row = self.component * block + row_comp
                row_full_dof = self.component * row_node + row_comp
                col_comp = 0
                while col_comp < self.component:
                    col_full_dof = self.component * row_node + col_comp
                    h_index = row_comp * self.component + col_comp
                    value = self.hash_matrix.diag[block][h_index]
                    if self.dirichlet.node[row_full_dof] == 1 or self.dirichlet.node[col_full_dof] == 1:
                        self.rhs[row] -= value * self.dirichlet.value[col_full_dof]
                        self.hash_matrix.diag[block][h_index] = 0.0
                    col_comp += 1
                if self.dirichlet.node[row_full_dof] == 1:
                    self.hash_matrix.diag[block][row_comp * self.component + row_comp] = 1.0
                row_comp += 1

        raw_nnz = self.hash_matrix.raw_non_diag_count[0]
        for k in range(raw_nnz):
            block_i = self.hash_matrix.non_diag.blockI[k]
            block_j = self.hash_matrix.non_diag.blockJ[k]
            if 0 <= block_i < active_nodes and 0 <= block_j < active_nodes:
                row_node = self.dof2node[block_i]
                col_node = self.dof2node[block_j]
                row_comp = 0
                while row_comp < self.component:
                    row = self.component * block_i + row_comp
                    row_full_dof = self.component * row_node + row_comp
                    col_comp = 0
                    while col_comp < self.component:
                        col_full_dof = self.component * col_node + col_comp
                        h_index = row_comp * self.component + col_comp
                        value = self.hash_matrix.non_diag.blockH[k][h_index]
                        if self.dirichlet.node[row_full_dof] == 1 or self.dirichlet.node[col_full_dof] == 1:
                            self.rhs[row] -= value * self.dirichlet.value[col_full_dof]
                            self.hash_matrix.non_diag.blockH[k][h_index] = 0.0
                        col_comp += 1
                    row_comp += 1

        for i in range(self.active_dof):
            node = self.dof2node[int(i // self.component)]
            comp = int(i % self.component)
            full_dof = self.component * node + comp
            if self.dirichlet.node[full_dof] == 1:
                self.rhs[i] = self.new_solution[i] - self.dirichlet.value[full_dof]

    @ti.func
    def newmark_acceleration_from_disp(self, disp, velocity_old, acceleration_old, dt):
        beta = self.newmark_beta
        return disp / (beta * dt * dt) - velocity_old / (beta * dt) - (0.5 / beta - 1.0) * acceleration_old

    @ti.func
    def newmark_velocity_from_disp(self, disp, velocity_old, acceleration_old, dt):
        beta = self.newmark_beta
        gamma = self.newmark_gamma
        return (
            gamma / (beta * dt) * disp
            - (gamma / beta - 1.0) * velocity_old
            - dt * (gamma / (2.0 * beta) - 1.0) * acceleration_old
        )

    @ti.func
    def newmark_pressure_rate(self, p_new, p_old, p_rate_old, p_acc_old, dt):
        beta = self.newmark_beta
        gamma = self.newmark_gamma
        return (
            gamma / (beta * dt) * (p_new - p_old)
            + (1.0 - gamma / beta) * p_rate_old
            - dt * (gamma / (2.0 * beta) - 1.0) * p_acc_old
        )

    @ti.func
    def newmark_pressure_accel(self, p_new, p_old, p_rate_old, p_acc_old, dt):
        beta = self.newmark_beta
        return (p_new - p_old) / (beta * dt * dt) - p_rate_old / (beta * dt) - (0.5 / beta - 1.0) * p_acc_old

    def apply_dirichlet(self):
        self.apply_dirichlet_hash()

    def apply_dirichlet_hash(self):
        if self.dirichlet.num == 0:
            return
        self.apply_dirichlet_hash_kernel()

    def current_hash_matrix(self):
        self.hash_matrix.finalize_taichi_assembly()
        return self.hash_matrix.to_scipy(self.active_dof // self.component).tocsr()

    def refresh_active_dofs(self):
        self.grid_reset()
        self.compute_shapefn()
        self.mass_p2g()
        self.find_active_node()
        self.prefix_sum_executor.run(self.node2dof)
        self.active_dof = self.set_active_dof()
        assert self.active_dof < self.degree_of_freedom, "Increase scale for StaticTwoPhaseULMPM."

    @ti.kernel
    def update_solution(self):
        for i in range(self.active_dof):
            self.new_solution[i] += self.increment[i]

    @ti.kernel
    def backup_current_solution(self):
        for i in range(self.active_dof):
            self.solution_backup[i] = self.new_solution[i]

    @ti.kernel
    def apply_increment_with_alpha(self, alpha: ti.f64):
        for i in range(self.active_dof):
            self.new_solution[i] = self.solution_backup[i] + alpha * self.increment[i]

    @ti.kernel
    def commit_step(self):
        for p in range(self.particleNum[0]):
            self.particle[p].x += self.particle_disp[p]
            self.F_old[p] = self.F_new[p]
            self.J_old[p] = self.J_new[p]
            self.stress_old[p] = self.stress_new[p]
            self.elastic_strain_old[p] = self.elastic_strain_new[p]
            self.particle_pressure[p] = self.particle_pressure_new[p]
            if ti.static(self.dynamic_formulation):
                self.particle[p].v = self.particle_velocity_new[p]
                self.particle[p].a = self.particle_acceleration_new[p]
                self.particle_pressure_rate[p] = self.particle_pressure_rate_new[p]
                self.particle_pressure_accel[p] = self.particle_pressure_accel_new[p]
            if ti.static(self.shape_function_name == "gimp"):
                bid = self.particle[p].bodyID
                xmin = self.body[bid].xmin
                xmax = self.body[bid].xmax
                for d in ti.static(range(config.DIM)):
                    col_norm_sq = 0.0
                    for i in ti.static(range(config.DIM)):
                        col_norm_sq += self.F_new[p][i, d] * self.F_new[p][i, d]
                    lp_updated = (self.dx / self.ppd / 2.0 - 1.0e-6) * ti.sqrt(ti.max(col_norm_sq, 0.0))
                    pos_d = self.particle[p].x[d]
                    if pos_d - lp_updated < xmin[d]:
                        lp_updated = pos_d - xmin[d] - 1.0e-6
                    if pos_d + lp_updated > xmax[d]:
                        lp_updated = xmax[d] - pos_d - 1.0e-6
                    self.lp[p][d] = ti.max(lp_updated, 1.0e-8)

    def _solve_device_linear_increment(self):
        self.hash_matrix.finalize_taichi_assembly()
        self.prepare_device_linear_rhs()
        result = self.hash_matrix.solve_flat_system(
            self.linear_rhs,
            self.increment,
            active_nodes=self.active_dof // self.component,
            tol=self.iterative_rtol,
            maxiter=self.iterative_maxiter,
            return_solution=False,
        )
        if not result["converged"]:
            raise RuntimeError(
                "StaticTwoPhaseULMPM device BiCGSTAB did not "
                f"converge: residual={result['residual']:.6e}, "
                f"iterations={result['iterations']}"
            )

    def _solve_host_linear_increment(self):
        # Explicit compatibility choice: only the linear solve crosses to
        # SciPy; all nonlinear state and residual assembly remain in fields.
        matrix = self.current_hash_matrix()
        matrix = self.sanitize_dirichlet_matrix(matrix)
        residual_snapshot = self.rhs.to_numpy()
        delta = self.solve_linear_system(matrix, -residual_snapshot)
        self.increment.from_numpy(delta)

    def _accept_full_newton_increment(self, _gravity, rhs_inf, step_inf):
        self.update_solution()
        return step_inf, rhs_inf, 1.0

    def _accept_line_search_increment(self, gravity, rhs_inf, step_inf):
        self.backup_current_solution()
        accepted_rhs = rhs_inf
        accepted_alpha = 1.0
        best_rhs = math.inf
        best_alpha = 1.0
        found_finite_trial = False
        accepted_on_break = False
        for ls_it in range(self.line_search_max_backtrack + 1):
            alpha = self.line_search_beta**ls_it
            self.apply_increment_with_alpha(alpha)
            self.zero_vector(self.active_dof, self.rhs)
            self.hash_matrix.reset_system()
            self.assemble_system(self.dt, gravity)
            self.assemble_neumann_step()
            self.apply_dirichlet_step()
            if not self.current_step_state_is_finite_and_positive():
                continue
            self.reduce_rhs_inf()
            trial_rhs_inf = float(self.rhs_inf_field[None])
            if not math.isfinite(trial_rhs_inf):
                continue
            found_finite_trial = True
            if trial_rhs_inf < best_rhs:
                best_rhs = trial_rhs_inf
                best_alpha = alpha
            if trial_rhs_inf < rhs_inf:
                accepted_rhs = trial_rhs_inf
                accepted_alpha = alpha
                accepted_on_break = True
                break
        else:
            if found_finite_trial:
                accepted_alpha = best_alpha
            else:
                accepted_alpha = 0.0
                accepted_rhs = rhs_inf
        if (not accepted_on_break) or accepted_alpha == 0.0:
            self.apply_increment_with_alpha(accepted_alpha)
            self.zero_vector(self.active_dof, self.rhs)
            self.hash_matrix.reset_system()
            self.assemble_system(self.dt, gravity)
            self.assemble_neumann_step()
            self.apply_dirichlet_step()
            if not self.current_step_state_is_finite_and_positive():
                accepted_alpha = 0.0
                self.apply_increment_with_alpha(accepted_alpha)
                self.zero_vector(self.active_dof, self.rhs)
                self.hash_matrix.reset_system()
                self.assemble_system(self.dt, gravity)
                self.assemble_neumann_step()
                self.apply_dirichlet_step()
                accepted_rhs = rhs_inf
            else:
                self.reduce_rhs_inf()
                accepted_rhs = float(self.rhs_inf_field[None])
        residual = step_inf if accepted_alpha == 0.0 else accepted_alpha * step_inf
        return residual, accepted_rhs, accepted_alpha

    def _both_newton_checks_converged(self, residual, rhs_inf, rhs_target):
        return residual < self.tol and rhs_inf < rhs_target

    def _either_newton_check_converged(self, residual, rhs_inf, rhs_target):
        return residual < self.tol or rhs_inf < rhs_target

    def solve_current_step(self, verbose=True):
        residual = math.inf
        rhs_inf = math.inf
        rhs_target = self.rhs_tol_abs
        accepted_alpha = 1.0
        gravity = ti.Vector(self.gravity)
        converged = False
        for it in range(self.max_iters):
            self.zero_vector(self.active_dof, self.rhs)
            self.zero_vector(self.active_dof, self.increment)
            self.hash_matrix.reset_system()
            self.assemble_system(self.dt, gravity)
            self.assemble_neumann_step()
            self.apply_dirichlet_step()
            self.reduce_rhs_inf()
            rhs_inf = float(self.rhs_inf_field[None])
            if it == 0:
                rhs_target = max(self.rhs_tol_abs, self.rhs_tol_rel * rhs_inf)
            self.solve_linear_increment_step()
            self.reduce_increment_inf()
            step_inf = float(self.increment_inf_field[None])
            residual, rhs_inf, accepted_alpha = self.accept_newton_increment_step(gravity, rhs_inf, step_inf)
            converged = self.newton_converged(residual, rhs_inf, rhs_target)
            if converged:
                converged = True
                break
        if verbose:
            print(
                "StaticTwoPhaseULMPM iterations: "
                f"{it + 1}, delta_inf: {residual:.3e}, rhs_inf: {rhs_inf:.3e}, rhs_target: {rhs_target:.3e}"
            )
        self.last_iterations = it + 1
        self.last_delta_inf = residual
        self.last_rhs_inf = rhs_inf
        self.last_rhs_target = rhs_target
        self.last_line_search_alpha = accepted_alpha
        self.last_converged = converged

    def substep(self, verbose=True):
        self.refresh_active_dofs()
        self.reset_step_solution()
        self.assemble_pressure_projection()
        self.apply_pressure_projection_to_solution()
        self.solve_current_step(verbose)
        self.commit_step()
        return self.last_converged

    def calculate_von_mises(self):
        particle_num = self.particleNum.to_numpy()[0]

        @ti.kernel
        def update():
            for p in range(self.particleNum[0]):
                self.vonMises[p] = self.von_mises_stress(self.stress_old[p])

        update()
        return self.vonMises.to_numpy()[:particle_num]

    def visualize(self, log=True):
        vtk_path = os.path.join(self.path, "vtks")
        if not os.path.exists(vtk_path):
            os.makedirs(vtk_path)
        particle_num = self.particleNum.to_numpy()[0]
        pos = np.ascontiguousarray(self.particle.x.to_numpy()[:particle_num])
        pressure = np.ascontiguousarray(self.particle_pressure.to_numpy()[:particle_num])
        stress = self.stress_old.to_numpy()[:particle_num]
        posz = np.zeros(particle_num)
        if config.DIM == 3:
            posz = np.ascontiguousarray(pos[:, 2])
        pointsToVTK(
            vtk_path + f"/GraphicMPMParticle{self.output_count:06d}",
            np.ascontiguousarray(pos[:, 0]),
            np.ascontiguousarray(pos[:, 1]),
            posz,
            data={
                "pressure": pressure,
                "stress_xx": np.ascontiguousarray(stress[:, 0, 0]),
                "stress_yy": np.ascontiguousarray(stress[:, 1, 1]),
                "stress_zz": np.ascontiguousarray(stress[:, 2, 2]) if config.DIM == 3 else np.zeros(particle_num),
                "von_mises": self.calculate_von_mises(),
            },
        )
        if log:
            print_save_file_info(
                "MPM",
                self.step_count,
                self.output_count,
                self.time,
                self.path,
            )
        self.output_count += 1

    def get_particle_state(self):
        particle_num = self.particleNum.to_numpy()[0]
        return {
            "position": self.particle.x.to_numpy()[:particle_num].copy(),
            "displacement": self.particle_disp.to_numpy()[:particle_num].copy(),
            "pressure": self.particle_pressure.to_numpy()[:particle_num].copy(),
            "pressure_rate": self.particle_pressure_rate.to_numpy()[:particle_num].copy(),
            "pressure_accel": self.particle_pressure_accel.to_numpy()[:particle_num].copy(),
            "velocity": self.particle.v.to_numpy()[:particle_num].copy(),
            "acceleration": self.particle.a.to_numpy()[:particle_num].copy(),
            "stress": self.stress_old.to_numpy()[:particle_num].copy(),
            "lp": self.lp.to_numpy()[:particle_num].copy(),
            "volume_ratio": self.J_old.to_numpy()[:particle_num].copy(),
        }

    def get_solver_state(self):
        particle_num = self.particleNum.to_numpy()[0]
        return {
            "position": self.particle.x.to_numpy()[:particle_num].copy(),
            "F_old": self.F_old.to_numpy()[:particle_num].copy(),
            "J_old": self.J_old.to_numpy()[:particle_num].copy(),
            "stress_old": self.stress_old.to_numpy()[:particle_num].copy(),
            "elastic_strain_old": self.elastic_strain_old.to_numpy()[:particle_num].copy(),
            "particle_pressure": self.particle_pressure.to_numpy()[:particle_num].copy(),
            "particle_pressure_rate": self.particle_pressure_rate.to_numpy()[:particle_num].copy(),
            "particle_pressure_accel": self.particle_pressure_accel.to_numpy()[:particle_num].copy(),
            "particle_velocity": self.particle.v.to_numpy()[:particle_num].copy(),
            "particle_acceleration": self.particle.a.to_numpy()[:particle_num].copy(),
            "lp": self.lp.to_numpy()[:particle_num].copy(),
            "output_count": self.output_count,
        }

    def set_solver_state(self, state):
        particle_num = self.particleNum.to_numpy()[0]
        pos = self.particle.x.to_numpy()
        pos[:particle_num] = state["position"]
        self.particle.x.from_numpy(pos)

        F_old = self.F_old.to_numpy()
        F_old[:particle_num] = state["F_old"]
        self.F_old.from_numpy(F_old)

        J_old = self.J_old.to_numpy()
        J_old[:particle_num] = state["J_old"]
        self.J_old.from_numpy(J_old)

        stress_old = self.stress_old.to_numpy()
        stress_old[:particle_num] = state["stress_old"]
        self.stress_old.from_numpy(stress_old)

        elastic_strain_old = self.elastic_strain_old.to_numpy()
        elastic_strain_old[:particle_num] = state["elastic_strain_old"]
        self.elastic_strain_old.from_numpy(elastic_strain_old)

        particle_pressure = self.particle_pressure.to_numpy()
        particle_pressure[:particle_num] = state["particle_pressure"]
        self.particle_pressure.from_numpy(particle_pressure)

        particle_pressure_rate = self.particle_pressure_rate.to_numpy()
        particle_pressure_rate[:particle_num] = state.get(
            "particle_pressure_rate", np.zeros_like(particle_pressure_rate[:particle_num])
        )
        self.particle_pressure_rate.from_numpy(particle_pressure_rate)

        particle_pressure_accel = self.particle_pressure_accel.to_numpy()
        particle_pressure_accel[:particle_num] = state.get(
            "particle_pressure_accel", np.zeros_like(particle_pressure_accel[:particle_num])
        )
        self.particle_pressure_accel.from_numpy(particle_pressure_accel)

        particle_velocity = self.particle.v.to_numpy()
        particle_velocity[:particle_num] = state.get(
            "particle_velocity", np.zeros_like(particle_velocity[:particle_num])
        )
        self.particle.v.from_numpy(particle_velocity)

        particle_acceleration = self.particle.a.to_numpy()
        particle_acceleration[:particle_num] = state.get(
            "particle_acceleration", np.zeros_like(particle_acceleration[:particle_num])
        )
        self.particle.a.from_numpy(particle_acceleration)

        lp = self.lp.to_numpy()
        lp[:particle_num] = state["lp"]
        self.lp.from_numpy(lp)

        self.output_count = int(np.asarray(state["output_count"]))
