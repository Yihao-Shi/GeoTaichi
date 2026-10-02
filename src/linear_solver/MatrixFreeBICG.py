from math import sqrt

import taichi as ti
from taichi.lang.impl import current_cfg

from src.utils.constants import BLOCK_SZ
from src.linear_solver.LinearOperator import LinearOperator


class MatrixFreeBICG(object):
    def __init__(self, length) -> None:
        self.p = ti.field(dtype=float)
        self.p_hat = ti.field(dtype=float)
        self.r = ti.field(dtype=float)
        self.r_tld = ti.field(dtype=float)
        self.s = ti.field(dtype=float)
        self.s_hat = ti.field(dtype=float)
        self.t = ti.field(dtype=float)
        self.Ap = ti.field(dtype=float)
        self.Ax = ti.field(dtype=float)
        self.Ashat = ti.field(dtype=float)
        ti.root.dense(ti.i, int(length)).place(self.p, self.p_hat, self.r, self.r_tld, self.s, self.s_hat, self.t, self.Ap, self.Ax, self.Ashat)
        self.scalar_reset()

        if current_cfg().arch == ti.cuda:
            self.reduce = reduce_shared
        elif current_cfg().arch == ti.cpu:
            self.reduce = reduce_atomic
        else:
            raise RuntimeError(f"{str(current_cfg().arch)} is not supported for preconditioned bi-conjuction gradient.")

    def scalar_reset(self):
        self.alpha = 1.
        self.beta = 0.
        self.omega = 1.
        self.rho = 1.
        self.rho_1 = 1.

    def solve(self, A: LinearOperator, b, x, size, tol=1e-6, maxiter=5000):
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
        succeeded = True
        reset(size, self.p, self.p_hat, self.r, self.r_tld, self.s, self.s_hat, self.t, self.Ap, self.Ax, self.Ashat)
        self.scalar_reset()
        A.matvec(x, self.Ax)
        init(size, b, self.p, self.r, self.r_tld, self.Ap, self.Ax, self.Ashat)
        initial_rTr = self.reduce(size, self.r, self.r)
        rTr = initial_rTr
        if sqrt(initial_rTr) >= tol:  # Do nothing if the initial residual is small enough
            for i in range(maxiter):
                self.rho = self.reduce(size, self.r, self.r_tld)
                if self.rho == 0.0:
                    succeeded = False
                    break
                if i == 0:
                    copy(size, self.r, self.p)
                else:
                    self.beta = (self.rho / self.rho_1) * (self.alpha / self.omega)
                    update_p(size, self.p, self.r, self.Ap, self.beta, self.omega)
                update_phat(size, self.p, self.p_hat)
                A.matvec(self.p, self.Ap)
                alpha_lower = self.reduce(size, self.r_tld, self.Ap)
                self.alpha = self.rho / alpha_lower
                update_s(size, self.r, self.s, self.Ap, self.alpha)
                update_shat(size, self.s, self.s_hat)
                A.matvec(self.s_hat, self.Ashat)
                copy(size, self.Ashat, self.t)
                omega_upper = self.reduce(size, self.t, self.s)
                omega_lower = self.reduce(size, self.t, self.t)
                self.omega = omega_upper / (omega_lower + 1e-16) if omega_lower == 0.0 else omega_upper / omega_lower
                update_x(size, x, self.p_hat, self.s_hat, self.alpha, self.omega)
                update_r(size, self.r, self.s, self.t, self.omega)
                rTr = self.reduce(size, self.r, self.r)
                if sqrt(rTr) < tol:
                    break
                self.rho_1 = self.rho
        if rTr >= tol:
            succeeded = False
        return succeeded


@ti.kernel
def reset(size: int, p: ti.template(), p_hat: ti.template(), r: ti.template(), r_tld: ti.template(), s: ti.template(), s_hat: ti.template(), t: ti.template(), Ap: ti.template(), 
            Ax: ti.template(), Ashat: ti.template()):
    for i in range(size):
        p[i] = 0.
        p_hat[i] = 0.
        r[i] = 0.
        r_tld[i] = 0.
        s[i] = 0.
        s_hat[i] = 0.
        t[i] = 0.
        Ap[i] = 0.
        Ax[i] = 0.
        Ashat[i] = 0.


@ti.kernel
def init(size: int, b: ti.template(), p: ti.template(), r: ti.template(), r_tld: ti.template(), Ap: ti.template(), Ax: ti.template(), Ashat: ti.template()):
    for i in range(size):
        r[i] = b[i] - Ax[i]
        r_tld[i] = b[i]
        p[i] = 0.0
        Ap[i] = 0.0
        Ashat[i] = 0.0


@ti.kernel
def reduce_shared(size: int, p: ti.template(), q: ti.template()) -> float:
    result = float(0.0)
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in range(size):
        thread_id = i % BLOCK_SZ
        pad_vector = ti.simt.block.SharedArray((64, ), ti.f64)

        pad_vector[thread_id] = p[i] * q[i]
        ti.simt.block.sync()

        j = int(0.5 * BLOCK_SZ)
        while j != 0:
            if thread_id < j:
                pad_vector[thread_id] += pad_vector[thread_id + j]
            ti.simt.block.sync()
            j >>= 1

        if thread_id == 0:
            result += pad_vector[thread_id]
    return result


@ti.kernel
def reduce_atomic(size: int, p: ti.template(), q: ti.template()) -> float:
    result = float(0.0)
    for i in range(size):
        result += p[i] * q[i]
    return result


@ti.kernel
def copy(size: int, orig: ti.template(), dest: ti.template()):
    for i in range(size):
        dest[i] = orig[i]


@ti.kernel
def update_p(size: int, p: ti.template(), r: ti.template(), Ap: ti.template(), beta: float, omega: float):
    for i in range(size):
        p[i] = r[i] + beta * (p[i] - omega * Ap[i])


@ti.kernel
def update_phat(size: int, p: ti.template(), p_hat: ti.template()):
    for i in range(size):
        p_hat[i] = p[i]


@ti.kernel
def update_s(size: int, r: ti.template(), s: ti.template(), Ap: ti.template(), alpha: float):
    for i in range(size):
        s[i] = r[i] - alpha * Ap[i]


@ti.kernel
def update_shat(size: int, s: ti.template(), s_hat: ti.template()):
    for i in range(size):
        s_hat[i] = s[i]


@ti.kernel
def update_x(size: int, x: ti.template(), p_hat: ti.template(), s_hat: ti.template(), alpha: float, omega: float):
    for i in range(size):
        x[i] += alpha * p_hat[i] + omega * s_hat[i]


@ti.kernel
def update_r(size: int, r: ti.template(), s: ti.template(), t: ti.template(), omega: float):
    for i in range(size):
        r[i] = s[i] - omega * t[i]
