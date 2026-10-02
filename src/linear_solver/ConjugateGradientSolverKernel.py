"""Taichi kernels for the legacy row-major conjugate-gradient solver."""

import taichi as ti


@ti.func
def A_get(A, ij, i, j):
    target_j = 0
    for j0 in range(ij[i][0]):
        if ij[i][j0 + 1] == j:
            target_j = j0
    return A[i][target_j]


@ti.kernel
def M_init(M: ti.template(), A: ti.template(), ij: ti.template()):
    for i in M:
        M[i] = 1.0 / A_get(A, ij, i, i)


@ti.kernel
def compute_Ad(A: ti.template(), ij: ti.template(), d: ti.template(), Ad: ti.template()):
    for i in A:
        Ad[i] = 0.0
        for j0 in range(ij[i][0]):
            Ad[i] += A[i][j0] * d[ij[i][j0 + 1]]


@ti.kernel
def r_d_init(b: ti.template(), M: ti.template(), r: ti.template(), d: ti.template()):
    for i in r:
        r[i] = b[i]
    for i in d:
        d[i] = M[i] * r[i]


@ti.kernel
def rmax(r: ti.template()) -> float:
    result = 0.0
    for i in r:
        ti.atomic_max(result, ti.abs(r[i]))
    return result


@ti.kernel
def compute_rMr(r: ti.template(), M: ti.template()) -> float:
    result = 0.0
    for i in r:
        result += r[i] * M[i] * r[i]
    return result


@ti.kernel
def update_x(x: ti.template(), d: ti.template(), alpha: float):
    for i in x:
        x[i] += alpha * d[i]


@ti.kernel
def update_r(r: ti.template(), Ad: ti.template(), alpha: float):
    for i in r:
        r[i] -= alpha * Ad[i]


@ti.kernel
def update_d(d: ti.template(), M: ti.template(), r: ti.template(), beta: float):
    for i in d:
        d[i] = M[i] * r[i] + beta * d[i]


@ti.kernel
def dot_product(y: ti.template(), z: ti.template()) -> float:
    result = 0.0
    for i in y:
        result += y[i] * z[i]
    return result
