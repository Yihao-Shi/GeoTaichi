"""Taichi kernels for preconditioned matrix-free conjugate gradient."""

import taichi as ti

from src.utils.constants import BLOCK_SZ


@ti.kernel
def reset(size: int, p: ti.template(), Ap: ti.template(), Ax: ti.template()):
    for i in range(size):
        p[i] = 0.0
        Ap[i] = 0.0
        Ax[i] = 0.0


@ti.kernel
def init(
    size: int,
    M: ti.template(),
    b: ti.template(),
    Ax: ti.template(),
    r: ti.template(),
    z: ti.template(),
    p: ti.template(),
):
    for i in range(size):
        r[i] = b[i] - Ax[i]
        z[i] = r[i] / M[i]
        p[i] = z[i]


@ti.kernel
def reduce_shared(size: int, p: ti.template(), q: ti.template()) -> float:
    result = float(0.0)
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in range(size):
        thread_id = i % BLOCK_SZ
        pad_vector = ti.simt.block.SharedArray((64,), ti.f64)

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
def update_x_r_z(
    size: int,
    x: ti.template(),
    p: ti.template(),
    r: ti.template(),
    Ap: ti.template(),
    z: ti.template(),
    M: ti.template(),
    alpha: float,
):
    for i in range(size):
        x[i] += alpha * p[i]
        r[i] -= alpha * Ap[i]
        z[i] = r[i] / M[i]


@ti.kernel
def update_p(size: int, p: ti.template(), z: ti.template(), beta: float):
    for i in range(size):
        p[i] = z[i] + beta * p[i]
