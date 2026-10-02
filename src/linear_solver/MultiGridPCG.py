import taichi as ti
from math import ceil, isfinite


@ti.data_oriented
class MGPCGPoissonSolver:
    def __init__(self, dimension, gnum, n_mg_levels=4, pre_and_post_smoothing=2, bottom_smoothing=50, smoother="rbgs"):

        self.FLUID = 1
        self.SOLID = 2
        self.AIR = 0

        # grid parameters
        self.dim = dimension
        self.gnum = [max(1, int(gnum[_])) for _ in range(dimension)]
        self.n_mg_levels = self.get_valid_multigrid_levels(n_mg_levels)
        self.pre_and_post_smoothing = pre_and_post_smoothing
        self.bottom_smoothing = bottom_smoothing
        self.eps = 1e-12
        self.smoother = smoother
        self.jacobi_omega = 2.0 / 3.0
        self.last_iterations = 0
        self.initial_residual = 0.0
        self.final_residual = 0.0
        self.breakdown_reason = ""
        self.breakdown_value = 0.0

        # rhs of linear system
        self.b = ti.field(dtype=float, shape=self.gnum)
        self.r = [
            ti.field(dtype=float, shape=[ceil(self.gnum[_] / (2**l)) for _ in range(dimension)])
            for l in range(self.n_mg_levels)
        ]
        self.z = [
            ti.field(dtype=float, shape=[ceil(self.gnum[_] / (2**l)) for _ in range(dimension)])
            for l in range(self.n_mg_levels)
        ]
        self.z_tmp = [
            ti.field(dtype=float, shape=[ceil(self.gnum[_] / (2**l)) for _ in range(dimension)])
            for l in range(self.n_mg_levels)
        ]

        # grid type
        self.grid_type = [
            ti.field(dtype=int, shape=[ceil(self.gnum[_] / (2**l)) for _ in range(dimension)])
            for l in range(self.n_mg_levels)
        ]

        # lhs of linear system and its corresponding form in coarse grids
        self.Adiag = [
            ti.field(dtype=float, shape=[ceil(self.gnum[_] / (2**l)) for _ in range(dimension)])
            for l in range(self.n_mg_levels)
        ]
        self.A_full = [
            ti.field(dtype=float, shape=[ceil(self.gnum[_] / (2**l)) for _ in range(dimension)])
            for l in range(self.n_mg_levels)
        ]
        self.Ax = [
            ti.Vector.field(dimension, dtype=float, shape=[ceil(self.gnum[_] / (2**l)) for _ in range(dimension)])
            for l in range(self.n_mg_levels)
        ]

        self.x = ti.field(dtype=float, shape=self.gnum)  # solution
        self.p = ti.field(dtype=float, shape=self.gnum)  # conjugate gradient
        self.Ap = ti.field(dtype=float, shape=self.gnum)  # matrix-vector product
        self.sum = ti.field(dtype=float, shape=())  # storage for reductions
        self.alpha = ti.field(dtype=float, shape=())  # step size
        self.beta = ti.field(dtype=float, shape=())  # step size

    def get_valid_multigrid_levels(self, requested_levels):
        levels = 1
        shape = list(self.gnum)
        while levels < max(1, int(requested_levels)) and any(size > 1 for size in shape):
            shape = [ceil(size / 2) for size in shape]
            levels += 1
        return levels

    @ti.kernel
    def init_gridtype(self, grid0: ti.template(), grid: ti.template()):
        fine_shape = ti.Vector([grid0.shape[d] for d in ti.static(range(self.dim))])
        for I in ti.grouped(grid):
            I2 = I * 2
            tot_fluid = 0
            tot_air = 0
            # Keep coarse nodes GPU-parallel, but enumerate each node's
            # 2**dim fine-cell block serially at runtime.  The old static
            # ndrange plus static dimension loop emitted 24 copies in 3D.
            flat = 0
            while flat < 2**self.dim:
                code = flat
                offset = ti.Vector.zero(int, self.dim)
                d = 0
                while d < self.dim:
                    offset[d] = code % 2
                    code //= 2
                    d += 1
                fine_id = I2 + offset
                inside = True
                d = 0
                while d < self.dim:
                    inside = inside and fine_id[d] < fine_shape[d]
                    d += 1
                if inside:
                    attr = int(grid0[fine_id])
                    if attr == self.AIR:
                        tot_air += 1
                    elif attr == self.FLUID:
                        tot_fluid += 1
                flat += 1
            if tot_air > 0:
                grid[I] = self.AIR
            elif tot_fluid > 0:
                grid[I] = self.FLUID
            else:
                grid[I] = self.SOLID

    @ti.kernel
    def initialize(self):
        for I in ti.grouped(ti.ndrange(*[self.gnum[_] for _ in range(self.dim)])):
            self.r[0][I] = 0
            self.z[0][I] = 0
            self.Ap[I] = 0
            self.p[I] = 0
            self.x[I] = 0
            self.b[I] = 0

        for l in ti.static(range(self.n_mg_levels)):
            for I in ti.grouped(ti.ndrange(*[ti.ceil(self.gnum[_] / (2**l), int) for _ in range(self.dim)])):
                self.grid_type[l][I] = 0
                self.Adiag[l][I] = 0
                self.Ax[l][I] = ti.zero(self.Ax[l][I])

    def reinitialize(self, cell_type):
        self.initialize()
        self.grid_type[0].copy_from(cell_type)

    @ti.func
    def is_inside(self, I, field: ti.template()):
        inside = True
        for d in ti.static(range(self.dim)):
            inside = inside and I[d] >= 0 and I[d] < field.shape[d]
        return inside

    @ti.func
    def neighbor_sum(self, Ax, x, I):
        ret = ti.cast(0.0, float)
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            left = I - offset
            right = I + offset
            if self.is_inside(left, x):
                ret += Ax[left][i] * x[left]
            if self.is_inside(right, x):
                ret += Ax[I][i] * x[right]
        return ret

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase and self.grid_type[l][I] == self.FLUID and ti.abs(self.Adiag[l][I]) > self.eps:
                self.z[l][I] = (self.r[l][I] - self.neighbor_sum(self.Ax[l], self.z[l], I)) / self.Adiag[l][I]

    @ti.kernel
    def smooth_jacobi(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            if self.grid_type[l][I] == self.FLUID and ti.abs(self.Adiag[l][I]) > self.eps:
                correction = (self.r[l][I] - self.neighbor_sum(self.Ax[l], self.z[l], I)) / self.Adiag[l][I]
                self.z_tmp[l][I] = (1.0 - self.jacobi_omega) * self.z[l][I] + self.jacobi_omega * correction
            else:
                self.z_tmp[l][I] = 0.0
        for I in ti.grouped(self.r[l]):
            self.z[l][I] = self.z_tmp[l][I]

    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            if self.grid_type[l][I] == self.FLUID:
                Az = self.Adiag[l][I] * self.z[l][I]
                Az += self.neighbor_sum(self.Ax[l], self.z[l], I)
                res = self.r[l][I] - Az
                self.r[l + 1][I // 2] += res

    @ti.kernel
    def restrict_full_weighting(self, l: ti.template()):
        fine_shape = ti.Vector([self.r[l].shape[d] for d in ti.static(range(self.dim))])
        for I in ti.grouped(self.r[l + 1]):
            value = 0.0
            # The coarse-grid struct-for remains GPU-parallel.  Its 3**dim
            # local full-weighting stencil is a serial runtime loop, avoiding
            # 27 offset and 81 offset-dimension compile-time expansions in 3D.
            flat = 0
            while flat < 3**self.dim:
                code = flat
                offset = ti.Vector.zero(int, self.dim)
                weight = 1.0
                d = 0
                while d < self.dim:
                    offset[d] = code % 3 - 1
                    code //= 3
                    weight *= 0.5 if offset[d] == 0 else 0.25
                    d += 1
                fine = I * 2 + offset
                inside = True
                d = 0
                while d < self.dim:
                    inside = inside and fine[d] >= 0 and fine[d] < fine_shape[d]
                    d += 1
                if inside and self.grid_type[l][fine] == self.FLUID:
                    Az = self.Adiag[l][fine] * self.z[l][fine]
                    Az += self.neighbor_sum(self.Ax[l], self.z[l], fine)
                    value += weight * (self.r[l][fine] - Az)
                flat += 1
            self.r[l + 1][I] = value

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] += self.z[l + 1][I // 2]

    def v_cycle(self):
        self.z[0].fill(0.0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)

            self.r[l + 1].fill(0.0)
            self.z[l + 1].fill(0.0)
            self.restrict(l)

        # solve Az = r on the coarse grid
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 1)
            self.smooth(self.n_mg_levels - 1, 0)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def v_cycle_jacobi(self):
        self.z[0].fill(0.0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth_jacobi(l)

            self.r[l + 1].fill(0.0)
            self.z[l + 1].fill(0.0)
            self.restrict_full_weighting(l)

        for i in range(self.bottom_smoothing):
            self.smooth_jacobi(self.n_mg_levels - 1)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth_jacobi(l)

    def apply_v_cycle(self):
        if self.smoother == "jacobi":
            self.v_cycle_jacobi()
        else:
            self.v_cycle()

    def fail(self, reason, value):
        self.breakdown_reason = reason
        self.breakdown_value = float(value)
        self.x.fill(0.0)
        return False

    def solve(self, max_iters=-1, rel_tol=1e-12, abs_tol=1e-14, eps=1e-12):
        self.compute_true_residual()
        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.sum[None]
        self.initial_residual = initial_rTr
        self.final_residual = initial_rTr
        self.last_iterations = 0
        self.breakdown_reason = ""
        self.breakdown_value = 0.0

        if not isfinite(initial_rTr) or initial_rTr < 0.0:
            return self.fail("initial_rTr", initial_rTr)
        tol = max(abs_tol, initial_rTr * rel_tol)
        if initial_rTr < tol:
            return True
        self.beta[None] = 0.0
        self.apply_v_cycle()
        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]
        if not isfinite(old_zTr) or old_zTr <= 0.0:
            return self.fail("initial_zTr", old_zTr)

        # Conjugate gradients. Recompute b-Ax before accepting convergence;
        # the recursively updated residual can lose accuracy for stiff systems.
        iter = 0
        reliable_update_interval = 32
        while max_iters == -1 or iter < max_iters:
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            if not isfinite(pAp) or pAp <= 0.0:
                return self.fail("pAp", pAp)
            self.alpha[None] = old_zTr / pAp

            self.update_xr()
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]
            self.final_residual = rTr
            iter += 1
            self.last_iterations = iter
            if not isfinite(rTr) or rTr < 0.0:
                return self.fail("rTr", rTr)

            if rTr < tol or iter % reliable_update_interval == 0:
                self.compute_true_residual()
                self.reduce(self.r[0], self.r[0])
                verified_rTr = self.sum[None]
                self.final_residual = verified_rTr
                if not isfinite(verified_rTr) or verified_rTr < 0.0:
                    return self.fail("verified_rTr", verified_rTr)
                if verified_rTr < tol:
                    return True
                self.beta[None] = 0.0
                self.apply_v_cycle()
                self.update_p()
                self.reduce(self.z[0], self.r[0])
                old_zTr = self.sum[None]
                if not isfinite(old_zTr) or old_zTr <= 0.0:
                    return self.fail("verified_zTr", old_zTr)
                continue

            self.apply_v_cycle()
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            if not isfinite(new_zTr) or new_zTr < 0.0:
                return self.fail("new_zTr", new_zTr)
            if old_zTr <= 0.0:
                return self.fail("old_zTr", old_zTr)
            self.beta[None] = new_zTr / old_zTr

            self.update_p()
            old_zTr = new_zTr
        self.compute_true_residual()
        self.reduce(self.r[0], self.r[0])
        self.final_residual = self.sum[None]
        if isfinite(self.final_residual) and 0.0 <= self.final_residual < tol:
            return True
        self.breakdown_reason = "max_iterations"
        self.breakdown_value = float(self.final_residual)
        return False

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            if self.grid_type[0][I] == self.FLUID:
                self.sum[None] += p[I] * q[I]

    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            if self.grid_type[0][I] == self.FLUID:
                r = self.Adiag[0][I] * self.p[I]
                r += self.neighbor_sum(self.Ax[0], self.p, I)
                self.Ap[I] = r

    @ti.kernel
    def compute_true_residual(self):
        for I in ti.grouped(self.r[0]):
            self.r[0][I] = 0.0
            if self.grid_type[0][I] == self.FLUID:
                Ax = self.Adiag[0][I] * self.x[I]
                Ax += self.neighbor_sum(self.Ax[0], self.x, I)
                self.r[0][I] = self.b[I] - Ax

    @ti.kernel
    def update_xr(self):
        alpha = self.alpha[None]
        for I in ti.grouped(self.p):
            if self.grid_type[0][I] == self.FLUID:
                self.x[I] += alpha * self.p[I]
                self.r[0][I] -= alpha * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            if self.grid_type[0][I] == self.FLUID:
                self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]
