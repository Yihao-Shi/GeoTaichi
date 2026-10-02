import taichi as ti
from math import ceil, isfinite
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix


@ti.data_oriented
class MGPCGMixPoissonSolver_Axi:
    def __init__(
        self, dimension, gnum, dr, n_mg_levels=4, pre_and_post_smoothing=2, bottom_smoothing=50, axi_offset=0.0
    ):

        self.FLUID = 1
        self.SOLID = 2
        self.AIR = 0

        # grid parameters
        self.dim = dimension
        self.gnum = gnum
        self.dr = dr
        self.axisy_offset = axi_offset
        self.n_mg_levels = n_mg_levels
        self.pre_and_post_smoothing = pre_and_post_smoothing
        self.bottom_smoothing = bottom_smoothing

        # rhs of linear system
        self.b = ti.field(dtype=float, shape=gnum)
        self.r = [
            ti.field(dtype=float, shape=[ceil(gnum[_] / (2**l)) for _ in range(dimension)])
            for l in range(self.n_mg_levels)
        ]
        self.z = [
            ti.field(dtype=float, shape=[ceil(gnum[_] / (2**l)) for _ in range(dimension)])
            for l in range(self.n_mg_levels)
        ]

        # grid type
        self.grid_type = [
            ti.field(dtype=int, shape=[ceil(gnum[_] / (2**l)) for _ in range(dimension)])
            for l in range(self.n_mg_levels)
        ]

        # lhs of linear system and its corresponding form in coarse grids
        self.Adiag = [
            ti.field(dtype=float, shape=[ceil(gnum[_] / (2**l)) for _ in range(dimension)])
            for l in range(self.n_mg_levels)
        ]
        self.Ax = [
            ti.Vector.field(dimension, dtype=float, shape=[ceil(gnum[_] / (2**l)) for _ in range(dimension)])
            for l in range(self.n_mg_levels)
        ]
        self.Ax_neg = [
            ti.Vector.field(dimension, dtype=float, shape=[ceil(gnum[_] / (2**l)) for _ in range(dimension)])
            for l in range(self.n_mg_levels)
        ]

        self.x = ti.field(dtype=float, shape=gnum)  # solution
        self.p = ti.field(dtype=float, shape=gnum)  # conjugate gradient
        self.Ap = ti.field(dtype=float, shape=gnum)  # matrix-vector product
        self.sum = ti.field(dtype=float, shape=())  # storage for reductions
        self.alpha = ti.field(dtype=float, shape=())  # step size
        self.beta = ti.field(dtype=float, shape=())  # step size
        self.last_iterations = 0
        self.initial_residual = 0.0
        self.final_residual = 0.0
        self.breakdown_reason = ""
        self.breakdown_value = 0.0

        # direct solver matrix and vectors
        dofs = gnum[0] * gnum[1]
        # self.right_hand_vector = ti.field(dtype=float)
        # ti.root.dense(ti.i, int(dofs)).place(self.right_hand_vector)
        self.sparse_matrix = CoordinateSparseMatrix(int(dofs * 5), int(dofs))

    @ti.kernel
    def init_gridtype(self, grid0: ti.template(), grid: ti.template()):
        fine_shape = ti.Vector([grid0.shape[d] for d in ti.static(range(self.dim))])
        for I in ti.grouped(grid):
            I2 = I * 2
            tot_fluid = 0
            tot_air = 0
            for offset in ti.static(ti.grouped(ti.ndrange(*((0, 2),) * self.dim))):
                fine_id = I2 + offset
                inside = True
                for d in ti.static(range(self.dim)):
                    inside = inside and fine_id[d] < fine_shape[d]
                if inside:
                    attr = int(grid0[fine_id])
                    if attr == self.AIR:
                        tot_air += 1
                    elif attr == self.FLUID:
                        tot_fluid += 1
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
                self.Ax_neg[l][I] = ti.zero(self.Ax_neg[l][I])

    def reinitialize(self, cell_type):
        self.initialize()
        self.grid_type[0].copy_from(cell_type)

    @ti.func
    def neighbor_sum(self, Ax, Ax_neg, x, I):
        ret = ti.cast(0.0, float)
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            if I[i] > 0:
                ret += Ax_neg[I][i] * x[I - offset]
            if I[i] + 1 < x.shape[i]:
                ret += Ax[I][i] * x[I + offset]
        return ret

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase and self.grid_type[l][I] == self.FLUID:
                self.z[l][I] = (
                    self.r[l][I] - self.neighbor_sum(self.Ax[l], self.Ax_neg[l], self.z[l], I)
                ) / self.Adiag[l][I]

    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            if self.grid_type[l][I] == self.FLUID:
                Az = self.Adiag[l][I] * self.z[l][I]
                Az += self.neighbor_sum(self.Ax[l], self.Ax_neg[l], self.z[l], I)
                res = self.r[l][I] - Az
                self.r[l + 1][I // 2] += res

    @ti.kernel
    def restrict_(self, l: ti.template()):
        for I in ti.grouped(self.z[l + 1]):
            self.z[l + 1][I] = 0.0

        for I in ti.grouped(self.r[l]):
            if self.grid_type[l][I] == self.FLUID:
                Az = self.Adiag[l][I] * self.z[l][I]
                Az += self.neighbor_sum(self.Ax[l], self.Ax_neg[l], self.z[l], I)
                res = self.r[l][I] - Az
                coarse_idx = I // 2
                w = (I[0] - 1.0 + 0.5) * self.dr * (2**l)  # r_weight at level l
                ti.atomic_add(self.r[l + 1][coarse_idx], w * res)
                ti.atomic_add(self.z[l + 1][coarse_idx], w)

        for I in ti.grouped(self.r[l + 1]):
            if self.z[l + 1][I] > 1e-12:
                self.r[l + 1][I] /= self.z[l + 1][I]

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

    def fail(self, reason, value):
        self.breakdown_reason = reason
        self.breakdown_value = float(value)
        self.x.fill(0.0)
        return False

    def solve(self, max_iters=-1, rel_tol=1e-3, abs_tol=1e-5, eps=1e-12):
        self.compute_true_residual()
        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.sum[None]
        self.last_iterations = 0
        self.initial_residual = initial_rTr
        self.final_residual = initial_rTr
        self.breakdown_reason = ""
        self.breakdown_value = 0.0

        if not isfinite(initial_rTr) or initial_rTr < 0.0:
            return self.fail("initial_rTr", initial_rTr)
        tol = max(abs_tol, initial_rTr * rel_tol)
        if initial_rTr < tol:
            return True
        self.v_cycle()
        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]
        if not isfinite(old_zTr) or abs(old_zTr) <= eps:
            return self.fail("initial_zTr", old_zTr)

        # Conjugate gradients with reliable true-residual replacement.
        iteration = 0
        reliable_update_interval = 32
        while max_iters == -1 or iteration < max_iters:
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            if not isfinite(pAp) or abs(pAp) <= eps:
                return self.fail("pAp", pAp)
            self.alpha[None] = old_zTr / pAp

            self.update_xr()
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]
            iteration += 1
            self.last_iterations = iteration
            self.final_residual = rTr
            if not isfinite(rTr) or rTr < 0.0:
                return self.fail("rTr", rTr)

            if rTr < tol or iteration % reliable_update_interval == 0:
                self.compute_true_residual()
                self.reduce(self.r[0], self.r[0])
                verified_rTr = self.sum[None]
                self.final_residual = verified_rTr
                if not isfinite(verified_rTr) or verified_rTr < 0.0:
                    return self.fail("verified_rTr", verified_rTr)
                if verified_rTr < tol:
                    return True
                self.v_cycle()
                self.beta[None] = 0.0
                self.update_p()
                self.reduce(self.z[0], self.r[0])
                old_zTr = self.sum[None]
                if not isfinite(old_zTr) or abs(old_zTr) <= eps:
                    return self.fail("verified_zTr", old_zTr)
                continue

            self.v_cycle()
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            if not isfinite(new_zTr) or abs(new_zTr) <= eps:
                return self.fail("new_zTr", new_zTr)
            if abs(old_zTr) <= eps:
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
                r += self.neighbor_sum(self.Ax[0], self.Ax_neg[0], self.p, I)
                self.Ap[I] = r

    @ti.kernel
    def compute_true_residual(self):
        for I in ti.grouped(self.r[0]):
            self.r[0][I] = 0.0
            if self.grid_type[0][I] == self.FLUID:
                Ax = self.Adiag[0][I] * self.x[I]
                Ax += self.neighbor_sum(self.Ax[0], self.Ax_neg[0], self.x, I)
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

    @ti.kernel
    def update_z(self):
        # Jacobi预处理：z = D⁻¹·r，其中D是A的对角线
        for I in ti.grouped(self.z[0]):
            if self.grid_type[0][I] == self.FLUID:
                if ti.abs(self.Adiag[0][I]) >= 1e-10:
                    self.z[0][I] = self.r[0][I] / self.Adiag[0][I]
                else:
                    print("Warning: zero diagonal entry in preconditioner!")
                    self.z[0][I] = self.r[0][I]
