from math import sqrt

import taichi as ti
from taichi.lang.impl import current_cfg

from src.utils.constants import BLOCK_SZ
from src.linear_solver.LinearOperator import LinearOperator


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
        self.alpha = 0.
        self.beta = 0.

    def solve(self, A: LinearOperator, b, x, M, size, tol=1e-6, maxiter=5000):
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
        succeeded = True
        reset(size, self.p, self.Ap, self.Ax)
        self.scalar_rest()
        A.matvec(x, self.Ax)
        init(size, M, b, self.Ax, self.r, self.z)
        initial_rTz = self.reduce(size, self.r, self.z)
        old_rTz = initial_rTz
        new_rTz = initial_rTz
        update_p(size, self.p, self.z, self.beta)
        if sqrt(abs(initial_rTz)) >= tol:  # Do nothing if the initial residual is small enough
            # -- Main loop --
            for _ in range(maxiter):
                A.matvec(self.p, self.Ap)  # compute Ap = A x p
                pAp = self.reduce(size, self.p, self.Ap)
                self.alpha = old_rTz / pAp
                update_x(size, x, self.p, self.alpha)
                update_r(size, self.r, self.Ap, self.alpha)
                update_z(size, self.z, self.r, M)
                new_rTz = self.reduce(size, self.r, self.z)
                if sqrt(ti.abs(new_rTz)) < tol:
                    break
                self.beta = new_rTz / old_rTz 
                update_p(size, self.p, self.z, self.beta)
                old_rTz = new_rTz
        assert new_rTz < tol, f"Failed to convergence. Final residual is {sqrt(ti.abs(new_rTz))}"
        return succeeded
        

@ti.kernel
def reset(size: int, p: ti.template(), Ap: ti.template(), Ax: ti.template()):
    for i in range(size):
        p[i] = 0.
        Ap[i] = 0.
        Ax[i] = 0.


@ti.kernel
def init(size: int, M: ti.template(), b: ti.template(), Ax: ti.template(), r: ti.template(), z: ti.template()):
    for i in range(size):
        r[i] = b[i] - Ax[i]
        z[i] = r[i] / M[i]


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
def update_x(size: int, x: ti.template(), p: ti.template(), alpha: float):
    for i in range(size):
        x[i] += alpha * p[i]


@ti.kernel
def update_r(size: int, r: ti.template(), Ap: ti.template(), alpha: float):
    for i in range(size):
        r[i] -= alpha * Ap[i]


@ti.kernel
def update_z(size: int, z: ti.template(), r: ti.template(), M: ti.template()):
    for i in range(size):
        z[i] = r[i] / M[i]


@ti.kernel
def update_p(size: int, p: ti.template(), z: ti.template(), beta: float):
    for i in range(size):
        p[i] = z[i] + beta * p[i]
