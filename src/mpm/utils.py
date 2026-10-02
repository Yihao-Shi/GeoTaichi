import taichi as ti

vec2i = ti.types.vector(2, ti.i32)
vec2f = ti.types.vector(2, ti.f64)
vec3i = ti.types.vector(3, ti.i32)
vec3f = ti.types.vector(3, ti.f64)
mat3x3 = ti.types.matrix(3, 3, ti.f64)


@ti.func
def diag(vec):
    return ti.Matrix([[vec[i] if i == j else 0 for j in ti.static(range(vec.n))] for i in ti.static(range(vec.n))])


@ti.func
def PSD(hess):
    lam, V = sym_eig2x2(hess)  # Eigen decomposition on symmetric matrix
    for i in ti.static(range(0, hess.n)):
        lam[i] = max(0, lam[i])
    return V @ diag(lam) @ V.transpose()


@ti.func
def sym_eig2x2(A):
    # Robust analytic eigendecomposition for 2x2 symmetric matrices.
    a = A[0, 0]
    b = 0.5 * (A[0, 1] + A[1, 0])
    d = A[1, 1]
    tr = a + d
    gap = (a - d) * (a - d) + 4.0 * b * b
    gap = max(gap, 0.0)
    root = ti.sqrt(gap)
    lambda1 = 0.5 * (tr + root)
    lambda2 = 0.5 * (tr - root)
    eigenvalues = ti.Vector([lambda1, lambda2], dt=float)

    eps = 1e-14
    v1 = ti.Vector.zero(float, 2)
    if ti.abs(b) < eps:
        if a >= d:
            v1 = ti.Vector([1.0, 0.0], dt=float)
        else:
            v1 = ti.Vector([0.0, 1.0], dt=float)
    else:
        # Use the more stable branch to avoid normalizing a zero vector.
        if ti.abs(b) > ti.abs(lambda1 - a):
            v1 = ti.Vector([1.0, (lambda1 - a) / b], dt=float)
        else:
            v1 = ti.Vector([b / (lambda1 - d), 1.0], dt=float)
        v1 = v1 / ti.max(v1.norm(), eps)
    v2 = ti.Vector([-v1[1], v1[0]], dt=float)
    eigenvectors = ti.Matrix.cols([v1, v2])
    return eigenvalues, eigenvectors


@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V


@ti.func
def vectorize(index, countVecx, countVecy, countVecz):
    ig = (index % (countVecx * countVecy)) % countVecx
    jg = (index % (countVecx * countVecy)) // countVecx
    kg = index // (countVecx * countVecy)
    return ig, jg, kg


@ti.func
def clamp_matrix_small_vals(mat, tol=1e-8):
    for i in ti.static(range(mat.n)):
        for j in ti.static(range(mat.m)):
            if ti.abs(mat[i, j]) < tol:
                mat[i, j] = 0.0
    return mat


@ti.func
def matrix_cols(matrix, cols):
    return ti.Vector([matrix[i, cols] for i in ti.static(range(matrix.m))])


@ti.func
def linearize(index, vector):
    assert index.n == vector.n
    if ti.static(vector.n == 2):
        return int(index[0] + index[1] * vector[0])
    elif ti.static(vector.n == 3):
        return int(index[0] + index[1] * vector[0] + index[2] * vector[0] * vector[1])


@ti.func
def vectorize_id(index, countVec):
    if ti.static(countVec.n == 2):
        ig = index % countVec[0]
        jg = index // countVec[0]
        return ig, jg
    elif ti.static(countVec.n == 3):
        ig = (index % (countVec[0] * countVec[1])) % countVec[0]
        jg = (index % (countVec[0] * countVec[1])) // countVec[0]
        kg = index // (countVec[0] * countVec[1])
        return ig, jg, kg


@ti.func
def matrix2vigot(matrix):
    return ti.Vector(
        [
            matrix[0, 0],
            matrix[1, 1],
            matrix[2, 2],
            0.5 * (matrix[0, 1] + matrix[1, 0]),
            0.5 * (matrix[1, 2] + matrix[2, 1]),
            0.5 * (matrix[0, 2] + matrix[2, 0]),
        ]
    )


@ti.func
def vigot2matrix(tensor):
    return ti.Matrix(
        [[tensor[0], tensor[3], tensor[5]], [tensor[3], tensor[1], tensor[4]], [tensor[5], tensor[4], tensor[2]]]
    )


@ti.pyfunc
def RodriguesRotationMatrix(origin, target):
    cos_theta = origin.dot(target)
    norm_vec = origin.cross(target)
    norm_vec_invert = mat3x3(
        [[0.0, -norm_vec[2], norm_vec[1]], [norm_vec[2], 0.0, -norm_vec[0]], [-norm_vec[1], norm_vec[0], 0.0]]
    )
    RotationMartix = (
        ti.Matrix.identity(ti.f64, 3) + norm_vec_invert + (norm_vec_invert @ norm_vec_invert) / (1 + cos_theta)
    )
    return RotationMartix


@ti.kernel
def copy_group_field(a: ti.template(), b: ti.template()):
    for I in ti.grouped(a):
        a[I] = b[I]


@ti.kernel
def add_field(active_dof: int, a: ti.template(), b: ti.template(), c: ti.template()):
    for I in range(active_dof):
        a[I] = b[I] + c[I]


@ti.kernel
def subtract_field(active_dof: int, a: ti.template(), b: ti.template(), c: ti.template()):
    for I in range(active_dof):
        a[I] = b[I] - c[I]


@ti.kernel
def copy_field(active_dof: int, a: ti.template(), b: ti.template()):
    for I in range(active_dof):
        a[I] = b[I]


@ti.kernel
def copy_grad(active_dof: int, a: ti.template(), b: ti.template()):
    for I in range(active_dof):
        a[I] = -b[I]


@ti.kernel
def clear_field(active_dof: int, a: ti.template()):
    for i in range(active_dof):
        a[i] = 0
