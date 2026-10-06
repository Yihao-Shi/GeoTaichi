from math import isfinite, sqrt

import taichi as ti
from taichi.lang.impl import current_cfg

from src.linear_solver.LinearOperator import LinearOperator

from src.linear_solver.MatrixFreeKrylovKernel import (
    bicgstab_reset as reset,
    bicgstab_init as init,
    bicgstab_restart as restart,
    true_residual,
    reduce_shared,
    reduce_atomic,
    copy,
    update_preconditioned,
    bicgstab_update_p as update_p,
    bicgstab_update_s as update_s,
    bicgstab_update_h as update_h,
    bicgstab_update_x as update_x,
    bicgstab_update_r as update_r,
)


class MatrixFreePBICGSTAB(object):
    def __init__(self, length) -> None:
        self.p = ti.field(dtype=float)
        self.r = ti.field(dtype=float)
        self.r_tld = ti.field(dtype=float)
        self.y = ti.field(dtype=float)
        self.z = ti.field(dtype=float)
        self.s = ti.field(dtype=float)
        self.h = ti.field(dtype=float)
        self.Ay = ti.field(dtype=float)
        self.Ax = ti.field(dtype=float)
        self.Az = ti.field(dtype=float)
        ti.root.dense(ti.i, int(length)).place(
            self.p, self.r, self.r_tld, self.s, self.y, self.z, self.h, self.Ay, self.Ax, self.Az
        )
        self.scalar_rest()

        if current_cfg().arch == ti.cuda:
            self.reduce = reduce_shared
        else:
            self.reduce = reduce_atomic

    def scalar_rest(self):
        self.rho = 0.0
        self.rho_1 = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.omega = 0.0
        self.last_initial_residual = 0.0
        self.last_residual = 0.0
        self.last_iterations = 0
        self.last_converged = False
        self.last_breakdown_reason = ""
        self.last_residual_restarts = 0

    def solve(self, A: LinearOperator, b, x, M, size, tol=1e-6, maxiter=5000, rel_tol=0.0):
        """Matrix-free biconjugate-gradient stabilized solver (BiCGSTAB).

        Use BiCGSTAB method to solve the linear system Ax = b, where A is implicitly
        represented as a LinearOperator.

        Args:
            A (LinearOperator): The coefficient matrix A of the linear system.
            b (Field): The right-hand side of the linear system.
            x (Field): The initial guess for the solution.
            size (int): The size of stiffness at current time
            maxiter (int): Maximum number of iterations.
            atol: Tolerance(absolute) for convergence.
            quiet (bool): Switch to turn on/off iteration log.
        """
        tol = float(tol)
        rel_tol = float(rel_tol)
        maxiter = int(maxiter)
        if not isfinite(tol) or tol < 0.0:
            raise ValueError("BiCGSTAB absolute tolerance must be finite and non-negative")
        if not isfinite(rel_tol) or rel_tol < 0.0:
            raise ValueError("BiCGSTAB relative tolerance must be finite and non-negative")
        if maxiter < 0:
            raise ValueError("BiCGSTAB maxiter must be non-negative")

        reset(size, self.p, self.r, self.r_tld, self.s, self.h, self.Ay, self.Ax, self.Az)
        self.scalar_rest()
        A.matvec(x, self.Ax)
        init(size, b, self.p, self.r, self.r_tld, self.Ay, self.Ax, self.Az)
        self.rho = self.reduce(size, self.r, self.r_tld)
        self.rho_1 = self.rho
        initial_rTr = self.reduce(size, self.r, self.r)
        initial_residual = sqrt(max(initial_rTr, 0.0))
        convergence_tol = max(tol, rel_tol * initial_residual)
        self.last_initial_residual = initial_residual
        self.last_residual = initial_residual
        self.last_iterations = 0
        if initial_residual <= convergence_tol:
            self.last_converged = True
            return True

        reliable_update_interval = 32
        for i in range(maxiter):
            if not isfinite(self.rho_1) or self.rho_1 == 0.0:
                self.last_breakdown_reason = "zero_shadow_residual_product"
                break
            update_preconditioned(size, self.y, M, self.p)
            A.matvec(self.y, self.Ay)
            alpha_lower = self.reduce(size, self.r_tld, self.Ay)
            if not isfinite(alpha_lower) or alpha_lower == 0.0:
                self.last_breakdown_reason = "zero_alpha_denominator"
                break
            self.alpha = self.rho_1 / alpha_lower
            if not isfinite(self.alpha):
                self.last_breakdown_reason = "non_finite_alpha"
                break
            update_h(size, x, self.h, self.y, self.alpha)
            update_s(size, self.r, self.s, self.Ay, self.alpha)
            sTs = self.reduce(size, self.s, self.s)
            self.last_iterations = i + 1
            if not isfinite(sTs) or sTs < 0.0:
                self.last_residual = float("inf")
                self.last_breakdown_reason = "non_finite_residual"
                break
            s_norm = sqrt(sTs)
            if s_norm <= convergence_tol:
                copy(size, self.h, x)
                A.matvec(x, self.Ax)
                restart(size, b, self.p, self.r, self.Ay, self.Ax, self.Az)
                verified_rTr = self.reduce(size, self.r, self.r)
                if not isfinite(verified_rTr) or verified_rTr < 0.0:
                    self.last_residual = float("inf")
                    self.last_breakdown_reason = "non_finite_residual"
                    break
                self.last_residual = sqrt(verified_rTr)
                if self.last_residual <= convergence_tol:
                    self.last_converged = True
                    break
                self.rho_1 = self.reduce(size, self.r, self.r_tld)
                if not isfinite(self.rho_1) or self.rho_1 == 0.0:
                    self.last_breakdown_reason = "zero_shadow_residual_product"
                    break
                self.last_residual_restarts += 1
                continue

            update_preconditioned(size, self.z, M, self.s)
            A.matvec(self.z, self.Az)
            omega_upper = self.reduce(size, self.Az, self.s)
            omega_lower = self.reduce(size, self.Az, self.Az)
            if not isfinite(omega_lower) or omega_lower <= 0.0:
                self.last_breakdown_reason = "zero_omega_denominator"
                break
            self.omega = omega_upper / omega_lower
            if not isfinite(self.omega) or self.omega == 0.0:
                self.last_breakdown_reason = "zero_or_non_finite_omega"
                break
            update_x(size, x, self.h, self.z, self.omega)
            update_r(size, self.r, self.s, self.Az, self.omega)
            rTr = self.reduce(size, self.r, self.r)
            if not isfinite(rTr) or rTr < 0.0:
                self.last_residual = float("inf")
                self.last_breakdown_reason = "non_finite_residual"
                break
            self.last_residual = sqrt(rTr)
            if self.last_residual <= convergence_tol or (i + 1) % reliable_update_interval == 0:
                A.matvec(x, self.Ax)
                restart(size, b, self.p, self.r, self.Ay, self.Ax, self.Az)
                verified_rTr = self.reduce(size, self.r, self.r)
                if not isfinite(verified_rTr) or verified_rTr < 0.0:
                    self.last_residual = float("inf")
                    self.last_breakdown_reason = "non_finite_residual"
                    break
                self.last_residual = sqrt(verified_rTr)
                if self.last_residual <= convergence_tol:
                    self.last_converged = True
                    break
                self.rho_1 = self.reduce(size, self.r, self.r_tld)
                if not isfinite(self.rho_1) or self.rho_1 == 0.0:
                    self.last_breakdown_reason = "zero_shadow_residual_product"
                    break
                self.last_residual_restarts += 1
                continue
            self.rho = self.reduce(size, self.r, self.r_tld)
            if not isfinite(self.rho) or self.rho == 0.0:
                self.last_breakdown_reason = "zero_shadow_residual_product"
                break
            self.beta = (self.rho / self.rho_1) * (self.alpha / self.omega)
            if not isfinite(self.beta):
                self.last_breakdown_reason = "non_finite_beta"
                break
            update_p(size, self.p, self.r, self.Ay, self.beta, self.omega)
            self.rho_1 = self.rho

        if not self.last_converged:
            A.matvec(x, self.Ax)
            true_residual(size, b, self.r, self.Ax)
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
