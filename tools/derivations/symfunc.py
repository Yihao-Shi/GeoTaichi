from sympy import *


def make_scalar(name: str, real=True):
    return symbols(name, real=real)

def make_vector(name: str, n: int, real=True):
    vec = []
    for i in range(n):
        vec.append(symbols(name + '_' + str(i), real=real))
    return Matrix(vec)

def make_matrix(name: str, n: int, m: int, real=True):
    mat = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(symbols(name + '_' + str(i) + '_' + str(j), real=real))
        mat.append(row)
    return Matrix(mat)

def outer_prod(vec1: Matrix, vec2: Matrix):
    return vec1 * vec2.transpose()

def dot_prod(vec1: Matrix, vec2: Matrix):
    assert len(vec1) == len(vec2)
    return (vec1.transpose() * vec2)[0]

def matrix_vector_prod(mat: Matrix, vec: Matrix):
    assert mat.shape[1] == len(vec)
    return mat * vec

def matrix_matrix_prod(mat1: Matrix, mat2: Matrix):
    assert mat1.shape[1] == mat2.shape[0]  # Ensure dimensions are compatible
    return mat1 * mat2


def double_dot(A: Matrix, B: Matrix):
    assert A.shape == B.shape
    return sum(A[i, j] * B[i, j] for i in range(A.rows) for j in range(A.cols))
