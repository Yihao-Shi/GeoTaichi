from math import isfinite, sqrt

import taichi as ti
from taichi.lang.impl import current_cfg

from src.linear_solver.LinearOperator import LinearOperator

from src.linear_solver.MatrixFreeKrylovKernel import (
    cg_reset as reset,
    cg_init as init,
    reduce_atomic,
    cg_update_x as update_x,
    cg_update_r as update_r,
    cg_update_p as update_p,
)


class MatrixFreeCG(object):
    def __init__(self, length) -> None:
        self.p = ti.field(dtype=float)
        self.r = ti.field(dtype=float)
        self.Ap = ti.field(dtype=float)
        self.Ax = ti.field(dtype=float)
        ti.root.dense(ti.i, int(length)).place(self.p, self.r, self.Ap, self.Ax)
        self.scalar_rest()

        if current_cfg().arch == ti.cuda:
            self.reduce = reduce_atomic
        elif current_cfg().arch == ti.cpu:
            self.reduce = reduce_atomic
        else:
            raise RuntimeError(f"{str(current_cfg().arch)} is not supported for preconditioned bi-conjuction gradient.")

    def scalar_rest(self):
        self.alpha = 0.0
        self.beta = 0.0
        self.last_initial_residual = 0.0
        self.last_residual = 0.0
        self.last_iterations = 0
        self.last_converged = False
        self.last_breakdown_reason = ""
        self.last_residual_restarts = 0

    def solve(self, A: LinearOperator, b, x, size, tol=1e-6, maxiter=5000, rel_tol=0.0):
        """Matrix-free conjugate-gradient solver.

        Use conjugate-gradient method to solve the linear system Ax = b, where A is implicitly
        represented as a LinearOperator.

        Args:
            A (LinearOperator): The coefficient matrix A of the linear system.
            b (Field): The right-hand side of the linear system.
            x (Field): The initial guess for the solution.
            size (int): The size of stiffness at current time
            maxiter (int): Maximum number of iterations.
            tol: Tolerance(absolute) for convergence.
        """
        tol = float(tol)
        rel_tol = float(rel_tol)
        maxiter = int(maxiter)
        if not isfinite(tol) or tol < 0.0:
            raise ValueError("CG absolute tolerance must be finite and non-negative")
        if not isfinite(rel_tol) or rel_tol < 0.0:
            raise ValueError("CG relative tolerance must be finite and non-negative")
        if maxiter < 0:
            raise ValueError("CG maxiter must be non-negative")

        reset(size, self.p, self.r, self.Ap, self.Ax)
        self.scalar_rest()
        A.matvec(x, self.Ax)
        init(size, b, self.p, self.r, self.Ap, self.Ax)
        initial_rTr = self.reduce(size, self.r, self.r)
        initial_residual = sqrt(max(initial_rTr, 0.0))
        convergence_tol = max(tol, rel_tol * initial_residual)
        self.last_initial_residual = initial_residual
        self.last_residual = initial_residual
        if not isfinite(initial_rTr) or initial_rTr < 0.0:
            self.last_residual = float("inf")
            self.last_breakdown_reason = "non_finite_initial_residual"
            return False
        if initial_residual <= convergence_tol:
            self.last_converged = True
            return True

        old_rTr = initial_rTr
        update_p(size, self.p, self.r, self.beta)
        reliable_update_interval = 32
        for iteration in range(maxiter):
            A.matvec(self.p, self.Ap)
            pAp = self.reduce(size, self.p, self.Ap)
            if not isfinite(pAp) or pAp <= 0.0:
                self.last_breakdown_reason = "non_positive_curvature"
                break
            self.alpha = old_rTr / pAp
            if not isfinite(self.alpha):
                self.last_breakdown_reason = "non_finite_alpha"
                break
            update_x(size, x, self.p, self.alpha)
            update_r(size, self.r, self.Ap, self.alpha)
            new_rTr = self.reduce(size, self.r, self.r)
            self.last_iterations = iteration + 1
            if not isfinite(new_rTr) or new_rTr < 0.0:
                self.last_residual = float("inf")
                self.last_breakdown_reason = "non_finite_residual"
                break
            self.last_residual = sqrt(new_rTr)
            if self.last_residual <= convergence_tol or (iteration + 1) % reliable_update_interval == 0:
                A.matvec(x, self.Ax)
                init(size, b, self.p, self.r, self.Ap, self.Ax)
                verified_rTr = self.reduce(size, self.r, self.r)
                if not isfinite(verified_rTr) or verified_rTr < 0.0:
                    self.last_residual = float("inf")
                    self.last_breakdown_reason = "non_finite_residual"
                    break
                self.last_residual = sqrt(verified_rTr)
                if self.last_residual <= convergence_tol:
                    self.last_converged = True
                    break
                old_rTr = verified_rTr
                update_p(size, self.p, self.r, 0.0)
                self.last_residual_restarts += 1
                continue
            if old_rTr <= 0.0:
                self.last_breakdown_reason = "non_positive_previous_residual"
                break
            self.beta = new_rTr / old_rTr
            if not isfinite(self.beta):
                self.last_breakdown_reason = "non_finite_beta"
                break
            update_p(size, self.p, self.r, self.beta)
            old_rTr = new_rTr

        if not self.last_converged:
            A.matvec(x, self.Ax)
            init(size, b, self.p, self.r, self.Ap, self.Ax)
            verified_rTr = self.reduce(size, self.r, self.r)
            if isfinite(verified_rTr) and verified_rTr >= 0.0:
                self.last_residual = sqrt(verified_rTr)
                self.last_converged = self.last_residual <= convergence_tol
            else:
                self.last_residual = float("inf")
                self.last_breakdown_reason = "non_finite_residual"
            if not self.last_converged and not self.last_breakdown_reason:
                self.last_breakdown_reason = "maximum_iterations"
        return self.last_converged
