import taichi as ti
import numpy as np

from src.utils.constants import ZEROVEC3f, Threshold, SQRT3, DBL_EPSILON
from src.utils.TypeDefination import vec3f, vec6f, mat2x2, mat3x3, vec2f


@ti.func
def equal(vec1, vec2):
    return (abs(vec1 - vec2) < Threshold).all()

@ti.func
def dot2(vector):
    return vector.dot(vector)

@ti.func
def ndot(vector1, vector2):
    return vector1[0] * vector2[0] - vector1[1] * vector2[1]

@ti.func
def wedgeProduct2D(firstFactor, secondFactor):
    return firstFactor[0] * secondFactor[1] - firstFactor[1] * secondFactor[0]

@ti.func
def cartesian_coosys_to_local_orthogonal(globals, ex_local, ey_local, ez_local):
    locals = ZEROVEC3f
    dot1 = ex_local.dot(ey_local)
    dot2 = ey_local.dot(ez_local)
    dot3 = ez_local.dot(ex_local)

    assert dot1 <= Threshold and dot2 <= Threshold and dot3 <= Threshold, \
        "Insufficient accuracy: using VectorFunction::cartesian_coosys_to_local_orthogonal() for non-orthogonal coo-sys"
    
    locals[0] = globals[0] * ex_local[0] + globals[1] * ex_local[1] + globals[2] * ex_local[2]
    locals[1] = globals[0] * ey_local[0] + globals[1] * ey_local[1] + globals[2] * ey_local[2]
    locals[2] = globals[0] * ez_local[0] + globals[1] * ez_local[1] + globals[2] * ez_local[2]
    return locals

@ti.func
def return_uniform3f():
    return vec3f([2*(ti.random()-0.5), 2*(ti.random()-0.5), 2*(ti.random()-0.5)]).normalized()

@ti.func
def TACIHI_NUMPY_SUB(TI_VEC, NP_VEC):
    if TI_VEC.n != np.size(NP_VEC):
        raise RuntimeError("Vector Dimension Error")
    return vec3f([TI_VEC[i] - NP_VEC[i] for i in range(TI_VEC.n)])

@ti.func
def summation(vector):
    result = 0.
    for n in ti.static(range(vector.n)):
        result += vector[n] * vector[n]
    return result

@ti.func
def Normalize(var):
    squareLen = 0.
    for d in ti.static(range(var.n)):
        squareLen += var[d] * var[d]
    sqrt_var = ti.sqrt(squareLen)
    if sqrt_var > Threshold:
        var /= sqrt_var
    return var

@ti.func
def MeanValue(var):
    var_val = 0.
    for d in ti.static(range(var.n)):
        var_val += var[d]
    return var_val / var.n

@ti.func
def voigt_tensor_dot(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2] + \
           2. * (vec1[3] * vec2[3] + vec1[4] * vec2[4] + vec1[5] * vec2[5])

@ti.func
def voigt_tensor_trace(vector):
    return vector[0] + vector[1] + vector[2]

@ti.func
def vsign(x):
    for i in ti.static(range(x.n)):
        if x[i] != 0:
            x[i] /= ti.abs(x[i])
    return x 

@ti.func
def Squared(vec):
    return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]

@ti.func
def SquaredLength(vec1, vec2):
    vec = vec1 - vec2
    return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]

@ti.func
def SIGN(vector):
    for i in ti.static(range(vector.n)):
        if vector[i] > Threshold:
            vector[i] = vector[i]
        elif vector[i] < -Threshold:
            vector[i] = -vector[i]
        else:
            vector[i] = 0
    return vector

@ti.func
def clamp(low_bound, high_bound, vec):
    for d in ti.static(range(vec.n)):
        if vec[d] < low_bound: vec[d] = low_bound
        if vec[d] > high_bound: vec[d] = high_bound
    return vec

@ti.func
def tensordot(Dmatrix, evector):
    return Dmatrix @ evector

@ti.func
def symtensordot(vector1, vector2):
    return vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2] \
           + 2 * (vector1[3] * vector2[3] + vector1[4] * vector2[4] + vector1[5] * vector2[5])

@ti.func
def voigt_form(matrix):
    return vec6f(matrix[0, 0], matrix[1, 1], matrix[2, 2],
                 0.5 * (matrix[0, 1] + matrix[1, 0]), 0.5 * (matrix[1, 2] + matrix[2, 1]), 0.5 * (matrix[0, 2] + matrix[2, 0]))

@ti.func
def equivalent_voigt(vector):
    return ti.sqrt(2./3. * symtensordot(vector, vector))

@ti.func
def principal_tensor(vector):
    matrix = mat3x3([[vector[0], vector[3], vector[5]], 
                   [vector[3], vector[1], vector[4]], 
                   [vector[5], vector[4], vector[2]]])
    m = matrix.trace()
    dd = matrix[0, 1] * matrix[0, 1]
    ee = matrix[1, 2] * matrix[1, 2]
    ff = matrix[0, 2] * matrix[0, 2]
    c1 = matrix[0, 0] * matrix[1, 1] + matrix[0, 0] * matrix[2, 2] + matrix[1, 1] * matrix[2, 2] - (dd + ee +
                                                                      ff)
    c0 = matrix[2, 2] * dd + matrix[0, 0] * ee + matrix[1, 1] * ff - matrix[0, 0] * matrix[1, 1] * matrix[
        2, 2] - 2.0 * matrix[0, 2] * matrix[0, 1] * matrix[1, 2]

    p = m * m - 3.0 * c1
    q = m * (p - 1.5 * c1) - 13.5 * c0
    sqrt_p = ti.sqrt(ti.abs(p))
    phi = 27.0 * (0.25 * c1 * c1 * (p - c1) + c0 * (q + 6.75 * c0))
    phi = (1.0 / 3.0) * ti.atan2(ti.sqrt(ti.abs(phi)), q)

    c = sqrt_p * ti.cos(phi)
    s = (1.0 / SQRT3) * sqrt_p * ti.sin(phi)
    eigenvalues = ZEROVEC3f
    eigenvalues[2] = (1.0 / 3.0) * (m - c)
    eigenvalues[1] = eigenvalues[2] + s
    eigenvalues[0] = eigenvalues[2] + c
    eigenvalues[2] = eigenvalues[2] - s
    return eigenvalues

@ti.func
def outer_product(vec1, vec2):
    return mat3x3([[vec1[0] * vec2[0], vec1[0] * vec2[1], vec1[0] * vec2[2]],
                   [vec1[1] * vec2[0], vec1[1] * vec2[1], vec1[1] * vec2[2]],
                   [vec1[2] * vec2[0], vec1[2] * vec2[1], vec1[2] * vec2[2]]])

@ti.func
def outer_product2D(vec1, vec2):
    return mat2x2([vec1[0] * vec2[0], vec1[1] * vec2[0]],
                  [vec1[0] * vec2[1], vec1[1] * vec2[1]])

@ti.func
def SquareLen(vec):
    squareLen = 0.
    for d in ti.static(range(vec.n)):
        squareLen += vec[d] * vec[d]
    return squareLen


@ti.func
def inner_multiply(vector):
    value = 1
    for i in ti.static(range(vector.n)):
        value *= vector[i]
    return value 

@ti.func
def truncation(vector):
    for i in ti.static(range(vector.n)):
        if ti.abs(vector[i]) < DBL_EPSILON:
            vector[i] = 0.
    return vector


@ti.func
def cross_matrix(vec):
    return mat3x3([0., -vec[2], vec[1]], [vec[2], 0., -vec[0]], [-vec[1], vec[0], 0.])

@ti.func
def spherical_angle(vec):
    return vec2f(ti.acos(vec[2]), ti.atan2(vec[1], vec[0]))

@ti.func
def normal_from_angles(alpha, beta):
    return vec3f(ti.sin(alpha) * ti.cos(beta), ti.sin(alpha) * ti.sin(beta), ti.cos(alpha))

@ti.func
def coord_global2local(scale, rotation_matrix, coord, offset):
    return rotation_matrix.transpose() @ (coord - offset) / scale

@ti.func
def coord_local2global(scale, rotation_matrix, coord, offset):
    return rotation_matrix @ coord * scale + offset

@ti.func
def global2local(vec, scale, rotation_matrix):
    return rotation_matrix.transpose() @ vec / scale

@ti.func
def local2global(vec, scale, rotation_matrix):
    return rotation_matrix @ vec * scale