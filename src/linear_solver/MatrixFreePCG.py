from math import isfinite, sqrt

import taichi as ti

from src.utils.constants import BLOCK_SZ
from src.linear_solver.LinearOperator import LinearOperator


from src.linear_solver.MatrixFreePCGKernel import (
    reset,
    init,
    reduce_shared,
    reduce_atomic,
    update_x,
    update_r,
    update_z,
    update_x_r_z,
    update_p,
)


class MatrixFreePCG(object):
    def __init__(self, length) -> None:
        self.p = ti.field(dtype=float)
        self.r = ti.field(dtype=float)
        self.z = ti.field(dtype=float)
        self.Ap = ti.field(dtype=float)
        self.Ax = ti.field(dtype=float)
        ti.root.dense(ti.i, int(length)).place(self.p, self.r, self.z, self.Ap, self.Ax)
        self.scalar_rest()
        self.reduce = reduce_atomic

    def scalar_rest(self):
        self.alpha = 0.0
        self.beta = 0.0
        self.last_initial_residual = 0.0
        self.last_residual = 0.0
        self.last_iterations = 0
        self.last_converged = False
        self.last_breakdown_reason = ""
        self.last_residual_restarts = 0

    def solve(self, A: LinearOperator, b, x, M, size, tol=1e-6, maxiter=5000, rel_tol=0.0):
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
            raise ValueError("PCG absolute tolerance must be finite and non-negative")
        if not isfinite(rel_tol) or rel_tol < 0.0:
            raise ValueError("PCG relative tolerance must be finite and non-negative")
        if maxiter < 0:
            raise ValueError("PCG maxiter must be non-negative")

        self.scalar_rest()
        A.matvec(x, self.Ax)
        init(size, M, b, self.Ax, self.r, self.z, self.p)
        initial_rTz = self.reduce(size, self.r, self.z)
        old_rTz = initial_rTz
        # ``r.T @ M^-1 @ r`` is the PCG recurrence scalar, not an
        # unscaled residual norm.  Comparing its square root with an absolute
        # tolerance makes convergence depend on the arbitrary scaling of the
        # preconditioner.  In particular, a stiff IPC barrier can make this
        # quantity tiny while ``||b - A x||`` is still large, returning the
        # untouched zero initial guess.  Use the true residual for every
        # stopping decision and keep rTz only in alpha/beta.
        initial_rTr = self.reduce(size, self.r, self.r)
        initial_residual = sqrt(max(initial_rTr, 0.0))
        convergence_tol = max(tol, rel_tol * initial_residual)
        self.last_initial_residual = initial_residual
        self.last_residual = initial_residual
        self.last_iterations = 0
        if initial_residual <= convergence_tol:
            self.last_converged = True
            return True
        if not isfinite(old_rTz) or old_rTz <= 0.0:
            self.last_breakdown_reason = "non_positive_preconditioned_residual"
            return False

        # Recursive CG residuals gradually lose the identity ``r = b - A x``
        # in finite precision.  This matters for IPC because a barrier Hessian
        # can be many orders of magnitude stiffer than inertia: the recurrence
        # may cancel to zero although the residual obtained from a fresh
        # matrix-vector product is still nonzero.  Periodically replace the
        # recurrence residual and always verify a prospective convergence.
        reliable_update_interval = 32
        for iteration in range(maxiter):
            A.matvec(self.p, self.Ap)
            pAp = self.reduce(size, self.p, self.Ap)
            if not isfinite(pAp) or pAp <= 0.0:
                self.last_breakdown_reason = "non_positive_curvature"
                break
            self.alpha = old_rTz / pAp
            if not isfinite(self.alpha):
                self.last_breakdown_reason = "non_finite_alpha"
                break
            update_x_r_z(size, x, self.p, self.r, self.Ap, self.z, M, self.alpha)
            self.last_iterations = iteration + 1
            residual_squared = self.reduce(size, self.r, self.r)
            if not isfinite(residual_squared) or residual_squared < 0.0:
                self.last_residual = float("inf")
                self.last_breakdown_reason = "non_finite_residual"
                break
            self.last_residual = sqrt(residual_squared)
            verify_residual = self.last_residual <= convergence_tol or (iteration + 1) % reliable_update_interval == 0
            if verify_residual:
                A.matvec(x, self.Ax)
                # ``init`` recomputes r=b-Ax, applies M^-1, and restarts the
                # Krylov direction with p=z.
                init(size, M, b, self.Ax, self.r, self.z, self.p)
                verified_squared = self.reduce(size, self.r, self.r)
                if not isfinite(verified_squared) or verified_squared < 0.0:
                    self.last_residual = float("inf")
                    self.last_breakdown_reason = "non_finite_residual"
                    break
                self.last_residual = sqrt(verified_squared)
                if self.last_residual <= convergence_tol:
                    self.last_converged = True
                    break
                old_rTz = self.reduce(size, self.r, self.z)
                if not isfinite(old_rTz) or old_rTz <= 0.0:
                    self.last_breakdown_reason = "non_positive_preconditioned_residual"
                    break
                self.last_residual_restarts += 1
                continue
            new_rTz = self.reduce(size, self.r, self.z)
            if not isfinite(new_rTz) or new_rTz <= 0.0:
                self.last_breakdown_reason = "non_positive_preconditioned_residual"
                break
            self.beta = new_rTz / old_rTz
            if not isfinite(self.beta):
                self.last_breakdown_reason = "non_finite_beta"
                break
            update_p(size, self.p, self.z, self.beta)
            old_rTz = new_rTz

        if not self.last_converged:
            # Report and decide from the actual residual, never from the
            # recursively updated vector left by the last Krylov iteration.
            A.matvec(x, self.Ax)
            init(size, M, b, self.Ax, self.r, self.z, self.p)
            verified_squared = self.reduce(size, self.r, self.r)
            if isfinite(verified_squared) and verified_squared >= 0.0:
                self.last_residual = sqrt(verified_squared)
                if self.last_residual <= convergence_tol:
                    self.last_converged = True
                    self.last_breakdown_reason = ""
            else:
                self.last_residual = float("inf")
                self.last_breakdown_reason = "non_finite_residual"
        if not self.last_converged and not self.last_breakdown_reason:
            self.last_breakdown_reason = "maximum_iterations"
        return self.last_converged
