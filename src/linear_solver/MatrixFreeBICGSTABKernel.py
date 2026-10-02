"""Taichi kernels for matrix-free BiCGSTAB."""

import taichi as ti

from src.utils.constants import BLOCK_SZ


@ti.kernel
def reset(
    size: int,
    p: ti.template(),
    r: ti.template(),
    r_tld: ti.template(),
    s: ti.template(),
    h: ti.template(),
    Ap: ti.template(),
    Ax: ti.template(),
    As: ti.template(),
):
    for i in range(size):
        p[i] = 0.0
        r[i] = 0.0
        r_tld[i] = 0.0
        s[i] = 0.0
        h[i] = 0.0
        Ap[i] = 0.0
        Ax[i] = 0.0
        As[i] = 0.0


@ti.kernel
def init(
    size: int,
    b: ti.template(),
    p: ti.template(),
    r: ti.template(),
    r_tld: ti.template(),
    Ap: ti.template(),
    Ax: ti.template(),
    As: ti.template(),
):
    for i in range(size):
        r[i] = b[i] - Ax[i]
        # BiCGSTAB's shadow residual belongs to the actual initial residual,
        # not to b.  These differ whenever the caller supplies a nonzero
        # initial guess.
        r_tld[i] = r[i]
        p[i] = r[i]
        Ap[i] = 0.0
        As[i] = 0.0


@ti.kernel
def restart(
    size: int,
    b: ti.template(),
    p: ti.template(),
    r: ti.template(),
    Ap: ti.template(),
    Ax: ti.template(),
    As: ti.template(),
):
    for i in range(size):
        r[i] = b[i] - Ax[i]
        p[i] = r[i]
        Ap[i] = 0.0
        As[i] = 0.0


@ti.kernel
def true_residual(size: int, b: ti.template(), r: ti.template(), Ax: ti.template()):
    for i in range(size):
        r[i] = b[i] - Ax[i]


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
def copy(size: int, orig: ti.template(), dest: ti.template()):
    for i in range(size):
        dest[i] = orig[i]


@ti.kernel
def update_p(size: int, p: ti.template(), r: ti.template(), Ap: ti.template(), beta: float, omega: float):
    for i in range(size):
        p[i] = r[i] + beta * (p[i] - omega * Ap[i])


@ti.kernel
def update_s(size: int, r: ti.template(), s: ti.template(), Ap: ti.template(), alpha: float):
    for i in range(size):
        s[i] = r[i] - alpha * Ap[i]


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
