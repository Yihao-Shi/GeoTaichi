from math import sqrt

import taichi as ti
from taichi.lang.impl import current_cfg

from src.utils.constants import BLOCK_SZ
from src.linear_solver.LinearOperator import LinearOperator


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
        ti.root.dense(ti.i, int(length)).place(self.p, self.r, self.r_tld, self.s, self.y, self.z, self.h, self.Ay, self.Ax, self.Az)
        self.scalar_rest()

        if current_cfg().arch == ti.cuda:
            self.reduce = reduce_shared
        elif current_cfg().arch == ti.cpu:
            self.reduce = reduce_atomic
        else:
            raise RuntimeError(f"{str(current_cfg().arch)} is not supported for preconditioned bi-conjuction gradient.")

    def scalar_rest(self):
        self.rho = 0.0
        self.rho_1 = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.omega = 0.0

    def solve(self, A: LinearOperator, b, x, M, size, tol=1e-6, maxiter=5000):
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
        reset(size, self.p, self.r, self.r_tld, self.s, self.h, self.Ay, self.Ax, self.Az)
        self.scalar_rest()
        A.matvec(x, self.Ax)
        init(size, b, self.p, self.r, self.r_tld, self.Ay, self.Ax, self.Az)
        copy(size, self.r, self.p)
        self.rho = self.reduce(size, self.r, self.r_tld)
        self.rho_1 = self.rho
        initial_rTr = self.reduce(size, self.r, self.r)
        rTr = initial_rTr
        if sqrt(initial_rTr) >= tol:  # Do nothing if the initial residual is small enough
            for i in range(maxiter):
                update_preconditioned(size, self.y, M, self.p)
                A.matvec(self.y, self.Ay)
                alpha_lower = self.reduce(size, self.r_tld, self.Ay)
                self.alpha = self.rho_1 / alpha_lower
                update_h(size, x, self.h, self.p, self.alpha)
                update_s(size, self.r, self.s, self.Ay, self.alpha)
                sTs = self.reduce(size, self.s, self.s)
                if sqrt(sTs) < tol:
                    copy(size, self.h, x)
                    break
                update_preconditioned(size, self.z, M, self.s)
                A.matvec(self.z, self.Az)
                omega_upper = self.reduce(size, self.Az, self.s)
                omega_lower = self.reduce(size, self.Az, self.Az)
                self.omega = omega_upper / (omega_lower + 1e-16) if omega_lower == 0.0 else omega_upper / omega_lower
                update_x(size, x, self.h, self.s, self.omega)
                update_r(size, self.r, self.s, self.Az, self.omega)
                rTr = self.reduce(size, self.r, self.r)
                if sqrt(rTr) < tol:
                    break
                self.rho = self.reduce(size, self.r, self.r_tld)
                self.beta = (self.rho / self.rho_1) * (self.alpha / self.omega)
                update_p(size, self.p, self.r, self.Ay, self.beta, self.omega)
                self.rho_1 = self.rho
        if rTr >= tol:
            succeeded = False
        return succeeded


@ti.kernel
def reset(size: int, p: ti.template(), r: ti.template(), r_tld: ti.template(), s: ti.template(), h: ti.template(), Ay: ti.template(), Ax: ti.template(), Az: ti.template()):
    for i in range(size):
        p[i] = 0.
        r[i] = 0.
        r_tld[i] = 0.
        s[i] = 0.
        h[i] = 0.
        Ay[i] = 0.
        Ax[i] = 0.
        Az[i] = 0.


@ti.kernel
def init(size: int, b: ti.template(), p: ti.template(), r: ti.template(), r_tld: ti.template(), Ay: ti.template(), Ax: ti.template(), Az: ti.template()):
    for i in range(size):
        r[i] = b[i] - Ax[i]
        r_tld[i] = b[i]
        p[i] = r[i]
        Ay[i] = 0.0
        Az[i] = 0.0


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
def update_preconditioned(size: int, arr: ti.template(), M: ti.template(), p: ti.template()):
    for i in range(size):
        arr[i] = p[i] / M[i]


@ti.kernel
def update_p(size: int, p: ti.template(), r: ti.template(), Ay: ti.template(), beta: float, omega: float):
    for i in range(size):
        p[i] = r[i] + beta * (p[i] - omega * Ay[i])


@ti.kernel
def update_s(size: int, r: ti.template(), s: ti.template(), Ay: ti.template(), alpha: float):
    for i in range(size):
        s[i] = r[i] - alpha * Ay[i]


@ti.kernel
def update_h(size: int, x: ti.template(), h: ti.template(), p: ti.template(), alpha: float):
    for i in range(size):
        h[i] = x[i] + alpha * p[i] 


@ti.kernel
def update_x(size: int, x: ti.template(), h: ti.template(), s: ti.template(), omega: float):
    for i in range(size):
        x[i] = h[i] + omega * s[i]


@ti.kernel
def update_r(size: int, r: ti.template(), s: ti.template(), t: ti.template(), omega: float):
    for i in range(size):
        r[i] = s[i] - omega * t[i]
