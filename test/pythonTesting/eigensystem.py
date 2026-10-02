import taichi as ti
import numpy as np
from numba import jit, int32, float32
ti.init(arch=ti.gpu)

a = np.array([[1, 0, 0], [0, 5, 0], [0, 0, 1]])

@jit(cache=True, nopython=True)
def _kernel_jacobi_(matrix, evalues, evectors):
    error = 1
    b, z = np.zeros(3), np.zeros(3)

    for i in range(3):
        b[i] = matrix[i, i]
        evalues[i] = matrix[i, i]

    for iter in range(50):
        sm = 0.
        for i in range(2):
            for j in range(i + 1, 3):
                sm += np.abs(matrix[i, j])
        if sm == 0.: return 0
        
        tresh = 0.
        if iter < 4: tresh = 0.2 * sm / (3*3)

        for i in range(2):
            for j in range(i + 1, 3):
                g = 100. * np.abs(matrix[i, j])
                if iter > 4 and abs(evalues[i]) + g == np.abs(evalues[i]) and np.abs(evalues[j]) + g == abs(evalues[j]):
                    matrix[i, j] = 0.
                elif np.abs(matrix[i, j]) > tresh:
                    h = evalues[j] - evalues[i]
                    t = 0.
                    if np.abs(h) + g == np.abs(h):
                        t = matrix[i, j] / h
                    else:
                        theta = 0.5 * h / matrix[i, j]
                        t = 1. / (np.abs(theta) + np.sqrt(1. + theta * theta))
                        if theta < 0.: t = -t
                    c = 1. / np.sqrt(1. + t * t)
                    s = t * c
                    tau = s / (1. + c)
                    h = t * matrix[i, j]
                    z[i] -= h
                    z[j] += h
                    evalues[i] -= h
                    evalues[j] += h
                    matrix[i, j] = 0.
                    for k in range(i):
                        u = matrix[k, i]
                        v = matrix[k, j]
                        matrix[k, i] = u - s * (v + u * tau)
                        matrix[k, j] = v + s * (u - v * tau)
                    for k in range(i + 1, j):
                        u = matrix[i, k]
                        v = matrix[k, j]
                        matrix[i, k] = u - s * (v + u * tau)
                        matrix[k, j] = v + s * (u - v * tau)
                    for k in range(j + 1, 3):
                        u = matrix[i, k]
                        v = matrix[j, k]
                        matrix[i, k] = u - s * (v + u * tau)
                        matrix[j, k] = v + s * (u - v * tau)
                    for k in range(3):
                        u = evectors[k, i]
                        v = evectors[k, j]
                        evectors[k, i] = u - s * (v + u * tau)
                        evectors[k, j] = v + s * (u - v * tau)

        for i in range(3):
            b[i] += z[i]
            evalues[i] = b[i]
            z[i] = 0.
    return 1

def eigensystem(a):
    print(a)
    inertia, evector = np.zeros(3), np.eye(3)
    error = _kernel_jacobi_(a, inertia, evector)
    if error == 1:
        print("Insufficient Jacobi rotations for rigid body")
    print(a, inertia, evector)
    
eigensystem(a)

import taichi as ti
ti.init()

b = ti.Matrix([[1, 2, 1], [2, 5, 2], [1, 2, 1]])
@ti.kernel
def k(b: ti.types.matrix(3, 3, float)):
    x, y = ti.sym_eig(b)
    print(x, y)
k(b)
