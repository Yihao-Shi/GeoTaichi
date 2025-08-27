from math import sqrt

import taichi as ti
from taichi.lang.impl import current_cfg

from src.utils.constants import BLOCK_SZ
from src.linear_solver.LinearOperator import LinearOperator


class MatrixFreeCG(object):
    def __init__(self, length) -> None:
        self.p = ti.field(dtype=float)
        self.r = ti.field(dtype=float)
        self.Ap = ti.field(dtype=float)
        self.Ax = ti.field(dtype=float)
        ti.root.dense(ti.i, int(length)).place(self.p, self.r, self.Ap, self.Ax)
        self.scalar_rest()

        if current_cfg().arch == ti.cuda:
            self.reduce = reduce_shared
        elif current_cfg().arch == ti.cpu:
            self.reduce = reduce_atomic
        else:
            raise RuntimeError(f"{str(current_cfg().arch)} is not supported for preconditioned bi-conjuction gradient.")

    def scalar_rest(self):
        self.alpha = 0.
        self.beta = 0.

    def solve(self, A: LinearOperator, b, x, size, tol=1e-6, maxiter=5000):
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
        reset(size, self.p, self.r, self.Ap, self.Ax)
        self.scalar_rest()
        A.matvec(x, self.Ax)
        init(size, b, self.p, self.r, self.Ap, self.Ax)
        initial_rTr = self.reduce(size, self.r, self.r)
        old_rTr = initial_rTr
        new_rTr = initial_rTr
        update_p(size, self.p, self.r, self.beta)
        if sqrt(initial_rTr) >= tol:  # Do nothing if the initial residual is small enough
            # -- Main loop --
            for _ in range(maxiter):
                A.matvec(self.p, self.Ap)  # compute Ap = A x p
                pAp = self.reduce(size, self.p, self.Ap)
                self.alpha = old_rTr / pAp
                update_x(size, x, self.p, self.alpha)
                update_r(size, self.r, self.Ap, self.alpha)
                new_rTr = self.reduce(size, self.r, self.r)
                if sqrt(new_rTr) < tol:
                    break
                self.beta = new_rTr / old_rTr
                update_p(size, self.p, self.r, self.beta)
                old_rTr = new_rTr
        if new_rTr >= tol:
            succeeded = False
        return succeeded
        

@ti.kernel
def reset(size: int, p: ti.template(), r: ti.template(), Ap: ti.template(), Ax: ti.template()):
    for i in range(size):
        p[i] = 0.
        r[i] = 0.
        Ap[i] = 0.
        Ax[i] = 0.


@ti.kernel
def init(size: int, b: ti.template(), p: ti.template(), r: ti.template(), Ap: ti.template(), Ax: ti.template()):
    for i in range(size):
        r[i] = b[i] - Ax[i]
        p[i] = 0.0
        Ap[i] = 0.0


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
def update_p(size: int, p: ti.template(), r: ti.template(), beta: float):
    for i in range(size):
        p[i] = r[i] + beta * p[i]
