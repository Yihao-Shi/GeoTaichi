import taichi as ti

from src.utils.constants import ZEROVEC3f, SQRT3, PI, DBL_EPSILON
from src.utils.ScalarFunction import clamp
from src.utils.TypeDefination import vec3f, mat3x3, mat4x4, mat2x2, mat3x4


@ti.func
def Diagonal(vec):
    return mat3x3([[vec[0], 0, 0], 
                   [0, vec[1], 0], 
                   [0, 0, vec[2]]])


@ti.func
def matrix_form(tensor):
    return mat3x3([[tensor[0], tensor[3], tensor[5]], 
                   [tensor[3], tensor[1], tensor[4]], 
                   [tensor[5], tensor[4], tensor[2]]])


@ti.func
def inverse_matrix_2x2(matrix):
    det = matrix[0, 0] * matrix[1, 1] - matrix[1, 0] * matrix[0, 1]
    return 1. / det * mat2x2([matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]])


@ti.func
def cwise_product(matrix1, matrix2):
    return mat3x3(matrix1[0, 0] * matrix2[0, 0], matrix1[0, 1] * matrix2[0, 1], matrix1[0, 2] * matrix2[0, 2],
                  matrix1[1, 0] * matrix2[1, 0], matrix1[1, 1] * matrix2[1, 1], matrix1[1, 2] * matrix2[1, 2],
                  matrix1[2, 0] * matrix2[2, 0], matrix1[2, 1] * matrix2[2, 1], matrix1[2, 2] * matrix2[2, 2])


@ti.func
def sum_cwise_product(matrix1, matrix2):
    return matrix1[0, 0] * matrix2[0, 0] + matrix1[0, 1] * matrix2[0, 1] + matrix1[0, 2] * matrix2[0, 2] + \
           matrix1[1, 0] * matrix2[1, 0] + matrix1[1, 1] * matrix2[1, 1] + matrix1[1, 2] * matrix2[1, 2] + \
           matrix1[2, 0] * matrix2[2, 0] + matrix1[2, 1] * matrix2[2, 1] + matrix1[2, 2] * matrix2[2, 2]


@ti.func
def trace(matrix):
    ti.static_assert(matrix.n == matrix.m)
    matrix_trace = 0.
    if ti.static(matrix.n == 2):
        matrix_trace = matrix[0, 0] + matrix[1, 1]
    elif ti.static(matrix.n == 3):
        matrix_trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    return matrix_trace


@ti.func
def principal_sym_tensor(matrix):
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
def hessenberg(matrix):
    assert matrix.n == matrix.m
    Qh = ti.Matrix.identity(float, matrix.n)
    H = matrix
    # 对列 0,1 做 Householder
    for col in ti.static(range(2)):
        v = ti.Vector.zero(float, matrix.n)
        for i in ti.static(range(col+1, matrix.n)):
            v[i] = matrix[i, col]
        alpha = ti.sqrt(v.dot(v))
        if v[col+1] > 0:
            alpha = -alpha
        v[col+1] -= alpha
        beta = 0.0
        for i in ti.static(range(3)):
            beta += v[i] * v[i]
        P = ti.Matrix.identity(float, matrix.n)
        if beta >= 1e-8:
            beta = 2.0 / beta
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    P[i, j] -= beta * v[i] * v[j]
        H = P @ H @ P
        Qh = Qh @ P
    return H, Qh


@ti.func
def QR(matrix):
    assert matrix.n == matrix.m
    Q = ti.Matrix.zero(float, matrix.n, matrix.m)
    R = ti.Matrix.zero(float, matrix.n, matrix.m)
    for j in ti.static(range(matrix.n)):
        v = ti.Vector([matrix[i, j] for i in range(matrix.n)])
        for i in ti.static(range(j)):
            qi = ti.Vector([Q[k, i] for k in range(matrix.n)])
            R[i, j] = qi.dot(v)
            for k in ti.static(range(matrix.n)):
                v[k] -= R[i, j] * qi[k]
        R[j, j] = ti.sqrt(v.dot(v))
        if R[j, j] > 1e-6:
            for k in ti.static(range(matrix.n)):
                Q[k, j] = v[k] / R[j, j]
    return Q, R


@ti.func
def sortEigenValueDescending(value):
    for i in ti.static(range(value.n - 1)):
        k = i
        for j in ti.static(range(i + 1, value.n)):
            if value[k] < value[j]:
                k = j
        if k != i:
            temp = value[i]
            value[i] = value[k]
            value[k] = temp
    return value


@ti.func
def sortEigenValueAscending(value):
    for i in ti.static(range(value.n - 1)):
        k = i
        for j in ti.static(range(i + 1, value.n)):
            if value[k] > value[j]:
                k = j
        if k != i:
            temp = value[i]
            value[i] = value[k]
            value[k] = temp
    return value


@ti.func
def eigenvalue(matrix):
    value = ti.Matrix.zero(float, matrix.n)
    temp = matrix

    for _ in range(50):
        Q, R = QR(temp)
        temp = R @ Q

    for i in ti.static(range(matrix.n)):
        value[i] = temp[i, i]

    value = sortEigenValueAscending(value)
    return value


@ti.func
def eigenvector(matrix, value):
    eigenVector = ti.Matrix.zero(float, matrix.n, matrix.m)
    temp = ti.Matrix.zero(float, matrix.n, matrix.m)

    for count in ti.static(range(matrix.m)):
        eValue = value[count]
        temp = matrix
        for i in ti.static(range(temp.n)):
            temp[i, i] -= eValue
        for i in ti.static(range(temp.n - 1)):
            coe = temp[i, i]
            for j in ti.static(range(i, temp.m)):
                temp[i, j] /= coe if ti.abs(coe) > DBL_EPSILON else 0.
            for m in ti.static(range(i+1, temp.n)):
                coe = temp[m, i]
                for n in ti.static(range(i, temp.m)):
                    temp[m, n] -= coe * temp[i, n]
        
        sum1 = eigenVector[eigenVector.n - 1, count] = 1.
        for m in ti.static(range(temp.n - 2, -1, -1)):
            sum2 = 0.
            for n in ti.static(range(m + 1, temp.m)):
                sum2 += temp[m, n] * eigenVector[n, count]
            sum2 = -sum2 / temp[m, m]
            sum1 += sum2 * sum2
            eigenVector[m, count] = sum2

        sum1 = ti.sqrt(sum1)
        for i in ti.static(range(eigenVector.n)):
            eigenVector[i, count] /= sum1

    return eigenVector


@ti.func
def get_eigenvalue(matrix):
    return eigenvalue(matrix)


@ti.func
def get_eigenvector(matrix):
    value = get_eigenvalue(matrix)
    eigenVector = eigenvector(matrix, value)
    return value, eigenVector


@ti.func
def eig(matrix):
    H, Qh = hessenberg(matrix)
    Q_tot = Qh
    for _ in range(50):
        Q, R = QR(H)
        H = R @ Q
        Q_tot = Q_tot @ Q
    eigenvalues = ti.Vector([H[i, i] for i in ti.static(range(matrix.n))])
    eigenvectors = Q_tot
    return eigenvalues, eigenvectors


@ti.func
def get_eigenvalue_3x3(matrix):
    value = ZEROVEC3f
    I1 = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    I2 = 0.5 * (I1 * I1 - matrix[0, 0] * matrix[0, 0] - matrix[1, 1] * matrix[1, 1] - matrix[2, 2] * matrix[2, 2])
    I3 = matrix[0, 0] * matrix[1, 1] * matrix[2, 2] + matrix[1, 0] * matrix[2, 1] * matrix[0, 2] + matrix[1, 2] * matrix[0, 1] * matrix[2, 0] - \
         matrix[0, 2] * matrix[1, 1] * matrix[2, 0] - matrix[0, 1] * matrix[1, 0] * matrix[2, 2] - matrix[0, 0] * matrix[1, 2] * matrix[2, 1]

    J2 = 1./3. * (I1 * I1 - 3. * I2)
    J3 = 1./27. * (2. * I1 * I1 * I1 - 9. * I1 * I2 + 27. * I3)
    lode = clamp(-1, 1, 1./3. * ti.acos(1.5 * ti.sqrt(3) * J3 / J2 ** 1.5))
    A = 2./3. * ti.sqrt(3. * J2)
    cos_theta_23 = ti.cos(lode + 2. * PI / 3.)
    cos_theta_43 = ti.cos(lode + 4. * PI / 3.)
    I_3 = 1./3. * I1
    value[0] = A * cos_theta_23 + I_3
    value[1] = A * cos_theta_43 + I_3
    value[2] = A * ti.cos(lode) + I_3
    return value


@ti.func
def principal_tensor(matrix):
    directors, values, _ = ti.svd(matrix)
    return vec3f([values[2, 2], values[1, 1], values[0, 0]]), \
           mat3x3([[directors[0, 2], directors[0, 1], directors[0, 0]],
                   [directors[1, 2], directors[1, 1], directors[1, 0]],
                   [directors[2, 2], directors[2, 1], directors[2, 0]]])


@ti.func
def getDeterminant(matrix):
    value = 0.0
    ndim = ti.static(matrix.n)
    mdim = ti.static(matrix.m)
    if ti.static(ndim == 1):
        value = matrix[0, 0]
    elif ti.static(ndim == 2):
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    else:
        sign = 1
        for i in ti.static(range(ndim)):
            sub_vector = ti.Matrix.zero(ndim - 1, mdim - 1)
            for m in ti.static(range(1, ndim)):
                z = 0
                for n in ti.static(range(ndim)):
                    if n != i:
                        sub_vector[m-1, z] = matrix[m, n]
                        z +=1
            value += sign * matrix[0, i] * getDeterminant(sub_vector)
            sign = -sign
    return value


@ti.func
def getCofactor(matrix):
    ndim = ti.static(matrix.n)
    mdim = ti.static(matrix.m)
    solution = ti.Matrix.zero(ndim, mdim)
    sub_vector = ti.Matrix.zero(ndim - 1, mdim - 1)

    for i in ti.static(range(ndim)):
        for j in ti.static(range(mdim)):
            p = 0
            for x in ti.static(range(ndim)):
                if x == i: continue
                q = 0
                for y in ti.static(range(ndim)):
                    if y == j: continue
                    sub_vector[p, q] = matrix[x, y]
                    q +=1
                p += 1
            solution[i, j] = ti.pow(-1, i + j) * getDeterminant(sub_vector)
    return solution


@ti.func
def getTranspose(matrix):
    ndim = ti.static(matrix.n)
    mdim = ti.static(matrix.m)
    solution = ti.Matrix.zero(ndim, mdim)
    for i in ti.static(range(ndim)):
        for j in ti.static(range(mdim)):
            solution[j, i] = matrix[i, j]


@ti.func
def getInverse(matrix):
    ndim = ti.static(matrix.n)
    mdim = ti.static(matrix.m)
    d = 1.0 / getDeterminant(matrix)
    solution = ti.Matrix.zero(ndim, mdim)
    for i in ti.static(range(ndim)):
        for j in ti.static(range(mdim)):
            solution[i, j] = matrix[i, j] * d
    return getTranspose(getCofactor(solution))


@ti.func
def E(mat, i, j, n):
    return mat[i % n, j % n]


@ti.func
def determinant1x1(mat):
    return mat[0, 0]


@ti.func
def determinant2x2(mat):
    return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]


@ti.func
def determinant3x3(mat):
    return (
            mat[0, 0] * (mat[1, 1] * mat[2, 2] - mat[2, 1] * mat[1, 2])
            - mat[1, 0] * (mat[0, 1] * mat[2, 2] - mat[2, 1] * mat[0, 2])
            + mat[2, 0] * (mat[0, 1] * mat[1, 2] - mat[1, 1] * mat[0, 2])
           )


@ti.func
def determinant4x4(mat):
    det = mat[0, 0] * 0  # keep type
    for i in ti.static(range(4)):
        det = det + (-1) ** i * \
        (mat[i, 0] * 
            (
                E(mat, i + 1, 1, 4) * (E(mat, i + 2, 2, 4) * E(mat, i + 3, 3, 4) - E(mat, i + 3, 2, 4) * E(mat, i + 2, 3, 4))
                - E(mat, i + 2, 1, 4) * (E(mat, i + 1, 2, 4) * E(mat, i + 3, 3, 4) - E(mat, i + 3, 2, 4) * E(mat, i + 1, 3, 4))
                + E(mat, i + 3, 1, 4) * (E(mat, i + 1, 2, 4) * E(mat, i + 2, 3, 4) - E(mat, i + 2, 2, 4) * E(mat, i + 1, 3, 4))
            )
        )
    return det


@ti.func
def determinant5x5(mat):
    det = mat[0, 0] * 0  # keep type
    for i in ti.static(range(5)):
        sub_vector = mat4x4([E(mat, i + 1, 1, 5), E(mat, i + 2, 1, 5), E(mat, i + 3, 1, 5), E(mat, i + 4, 1, 5)],
                            [E(mat, i + 1, 2, 5), E(mat, i + 2, 2, 5), E(mat, i + 3, 2, 5), E(mat, i + 4, 2, 5)],
                            [E(mat, i + 1, 3, 5), E(mat, i + 2, 3, 5), E(mat, i + 3, 3, 5), E(mat, i + 4, 3, 5)],
                            [E(mat, i + 1, 4, 5), E(mat, i + 2, 4, 5), E(mat, i + 3, 4, 5), E(mat, i + 4, 4, 5)])
        det = det + (-1) ** i * (mat[i, 0] * determinant4x4(sub_vector))
    return det


@ti.func
def get_determinant(mat):
    shape = ti.static(mat.n)
    if ti.static(shape == 1):
        return determinant1x1(mat)
    if ti.static(shape == 2):
        return determinant2x2(mat)
    if ti.static(shape == 3):
        return determinant3x3(mat)
    if ti.static(shape == 4):
        return determinant4x4(mat)
    if ti.static(shape == 5):
        return determinant5x5(mat)
    

@ti.func
def get_jacobian_inverse1(mat):
    return ti.Matrix([[1.0 / mat[0, 0]]])
    

@ti.func
def get_jacobian_inverse2(mat):
    shape = ti.static(mat.n)
    inv_determinant = 1.0 / determinant2x2(mat)
    return inv_determinant * mat2x2([mat[1, 1], -mat[0, 1]], [-mat[1, 0], mat[0, 0]])
    
    
@ti.func
def get_jacobian_inverse3(mat):
    inv_determinant = 1.0 / determinant3x3(mat)    
    return inv_determinant * ti.Matrix(
        [
            [
                E(mat, i + 1, j + 1, 3) * E(mat, i + 2, j + 2, 3)
                - E(mat, i + 2, j + 1, 3) * E(mat, i + 1, j + 2, 3)
                for i in ti.static(range(3))
            ]
            for j in ti.static(range(3))
        ]
    )
    

@ti.func
def get_jacobian_inverse4(mat):
    inv_determinant = 1.0 / determinant4x4(mat)
    return inv_determinant * ti.Matrix(
        [
            [
                (-1) ** (i + j)
                * (
                    (
                        E(mat, i + 1, j + 1, 4)
                        * (
                            E(mat, i + 2, j + 2, 4) * E(mat, i + 3, j + 3, 4)
                            - E(mat, i + 3, j + 2, 4) * E(mat, i + 2, j + 3, 4)
                        )
                        - E(mat, i + 2, j + 1, 4)
                        * (
                            E(mat, i + 1, j + 2, 4) * E(mat, i + 3, j + 3, 4)
                            - E(mat, i + 3, j + 2, 4) * E(mat, i + 1, j + 3, 4)
                        )
                        + E(mat, i + 3, j + 1, 4)
                        * (
                            E(mat, i + 1, j + 2, 4) * E(mat, i + 2, j + 3, 4)
                            - E(mat, i + 2, j + 2, 4) * E(mat, i + 1, j + 3, 4)
                        )
                    )
                )
                for i in ti.static(range(4))
            ]
            for j in ti.static(range(4))
        ]
    )
    

@ti.func
def get_jacobian_inverse5(mat):
    inv_determinant = 1.0 / determinant5x5(mat)
    return inv_determinant * ti.Matrix(
        [
            [
                (-1) ** (i + j)
                * (
                    (
                        E(mat, i + 1, j + 1, 5)
                        * ( 
                            E(mat, i + 2, j + 2, 5) * (E(mat, i + 3, j + 3, 5) * E(mat, i + 4, j + 4, 5) - E(mat, i + 4, j + 3, 5) * E(mat, i + 3, j + 4, 5))
                            - E(mat, i + 3, j + 2, 5) * (E(mat, i + 2, j + 3, 5) * E(mat, i + 4, j + 4, 5) - E(mat, i + 4, j + 3, 5) * E(mat, i + 2, j + 4, 5))
                            + E(mat, i + 4, j + 2, 5) * (E(mat, i + 2, j + 3, 5) * E(mat, i + 3, j + 4, 5) - E(mat, i + 3, j + 3, 5) * E(mat, i + 2, j + 4, 5))
                        )
                        - E(mat, i + 2, j + 1, 5)
                        * (
                            E(mat, i + 1, j + 2, 5) * (E(mat, i + 3, j + 3, 5) * E(mat, i + 4, j + 4, 5) - E(mat, i + 4, j + 3, 5) * E(mat, i + 3, j + 4, 5))
                            - E(mat, i + 3, j + 2, 5) * (E(mat, i + 1, j + 3, 5) * E(mat, i + 4, j + 4, 5) - E(mat, i + 4, j + 3, 5) * E(mat, i + 1, j + 4, 5))
                            + E(mat, i + 4, j + 2, 5) * (E(mat, i + 1, j + 3, 5) * E(mat, i + 3, j + 4, 5) - E(mat, i + 3, j + 3, 5) * E(mat, i + 1, j + 4, 5))
                        )
                        + E(mat, i + 3, j + 1, 5)
                        * (
                            E(mat, i + 1, j + 2, 5) * (E(mat, i + 2, j + 3, 5) * E(mat, i + 4, j + 4, 5) - E(mat, i + 4, j + 3, 5) * E(mat, i + 2, j + 4, 5))
                            - E(mat, i + 2, j + 2, 5) * (E(mat, i + 1, j + 3, 5) * E(mat, i + 4, j + 4, 5) - E(mat, i + 4, j + 3, 5) * E(mat, i + 1, j + 4, 5))
                            + E(mat, i + 4, j + 2, 5) * (E(mat, i + 1, j + 3, 5) * E(mat, i + 2, j + 4, 5) - E(mat, i + 2, j + 3, 5) * E(mat, i + 1, j + 4, 5))
                        )
                        + E(mat, i + 4, j + 1, 5)
                        * (
                            E(mat, i + 1, j + 2, 5) * (E(mat, i + 2, j + 3, 5) * E(mat, i + 3, j + 4, 5) - E(mat, i + 3, j + 3, 5) * E(mat, i + 2, j + 4, 5))
                            - E(mat, i + 2, j + 2, 5) * (E(mat, i + 1, j + 3, 5) * E(mat, i + 3, j + 4, 5) - E(mat, i + 3, j + 3, 5) * E(mat, i + 1, j + 4, 5))
                            + E(mat, i + 3, j + 2, 5) * (E(mat, i + 1, j + 3, 5) * E(mat, i + 2, j + 4, 5) - E(mat, i + 2, j + 3, 5) * E(mat, i + 1, j + 4, 5))
                        )
                    )
                )
                for i in ti.static(range(5))
            ]
            for j in ti.static(range(5))
        ]
    )


@ti.func
def get_jacobian_inverse(mat):
    shape = ti.static(mat.n)
    if ti.static(shape == 1):
        return get_jacobian_inverse1(mat)
    if ti.static(shape == 2):
        return get_jacobian_inverse2(mat)
    if ti.static(shape == 3):
        return get_jacobian_inverse3(mat)
    if ti.static(shape == 4):
        return get_jacobian_inverse4(mat)
    if ti.static(shape == 5):
        return get_jacobian_inverse5(mat)
    

@ti.func
def truncation(mat):
    for i in ti.static(range(mat.n)):
        for j in ti.static(range(mat.m)):
            if ti.abs(mat[i, j]) < DBL_EPSILON:
                mat[i, j] = 0.
    return mat


@ti.func
def LUinverse(mat):
    L, U = polar_decomposition(mat)
    Linv = ti.Matrix.zero(float, mat.n, mat.m)
    Uinv = ti.Matrix.zero(float, mat.n, mat.m)
    for j in ti.static(range(mat.n)):
        for i in ti.static(range(j, mat.m)):
            if i == j: Linv[i, j] = 1. / L[i, j]
            elif i < j: Linv[i, j] = 0.
            else:
                s = 0.
                for k in ti.static(range(j, i)):
                    s += L[i, k] * Linv[k, j]
                Linv[i, j] = -Linv[j, j] * s
    for j in ti.static(range(mat.n)):
        for i in ti.static(range(j, -1, -1)):
            if i == j: Uinv[i, j] = 1 / U[i, j]
            elif i > j: Uinv[i, j] = 0
            else:
                s = 0.0
                for k in ti.static(range(i + 1, j + 1)):
                    s += U[i, k] * Uinv[k, j]
                Uinv[i, j] = -1 / U[i, i] * s
    return Uinv @ Linv


@ti.func
def gauss_elimination(A, b):
    Ab = mat3x4([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            Ab[i, j] = A[i, j]
    for i in ti.static(range(3)):
        Ab[i, 3] = b[i]

    for i in ti.static(range(3)):
        max_row = i
        max_v = ti.abs(Ab[i, i])
        for j in ti.static(range(i + 1, 3)):
            if ti.abs(Ab[j, i]) > max_v:
                max_row = j
                max_v = ti.abs(Ab[j, i])
        assert max_v != 0.0, "Matrix is singular in linear solve."
        if i != max_row:
            if max_row == 1:
                for col in ti.static(range(4)):
                    Ab[i, col], Ab[1, col] = Ab[1, col], Ab[i, col]
            else:
                for col in ti.static(range(4)):
                    Ab[i, col], Ab[2, col] = Ab[2, col], Ab[i, col]
        assert Ab[i, i] != 0.0, "Matrix is singular in linear solve."
        for j in ti.static(range(i + 1, 3)):
            scale = Ab[j, i] / Ab[i, i]
            Ab[j, i] = 0.0
            for k in ti.static(range(i + 1, 4)):
                Ab[j, k] -= Ab[i, k] * scale
    # Back substitution
    x = vec3f(0, 0, 0)
    for i in ti.static(range(2, -1, -1)):
        x[i] = Ab[i, 3]
        for k in ti.static(range(i + 1, 3)):
            x[i] -= Ab[i, k] * x[k]
        x[i] = x[i] / Ab[i, i]
    return x


@ti.func
def polar_decomposition(mat):
    L = ti.Matrix.zero(float, mat.n, mat.m)
    U = ti.Matrix.zero(float, mat.n, mat.m)
    for i in ti.static(range(mat.n)):
        L[i, i] = 1.
    for j in ti.static(range(mat.n)):
        U[0, j] = mat[0, j]
    for i in ti.static(range(1, mat.n)):
        L[i, 0] = mat[i, 0] / U[0, 0]
    for i in ti.static(range(mat.n)):
        for j in ti.static(range(i, mat.n)):
            s = 0.
            for k in ti.static(range(i)):
                s += L[i, k] * U[k, j]
            U[i, j] = mat[i, j] - s
        for d in ti.static(range(i, mat.n)):
            s = 0.
            for k in ti.static(range(i)):
                s += L[d, k] * U[k, i]
            L[d, i] = (mat[d, i] - s) / U[i, i]
    return L, U


@ti.func
def ti_polar_decomposition(mat):
    dim = mat.n
    # SVD(A) = U S V*
    U, S, V = ti.svd(mat)
    # now do polar decomposition into M = R * T, where R is rotation and T is translation matrix
    R = U @ V.transpose()
    if R.determinant() < 0.0:
        # identify the smallest entry in S and flip its sign
        S[dim - 1, dim - 1] *= -1
        # recompute R using flipped stretch eigenvalues
        R = mat * V * S.inverse() * V.transpose()
    assert R.determinant() > 0.0
    return R


@ti.func
def ti_polar_decomposition_stable(mat):
    dim = mat.n
    # SVD(A) = U S V*
    U, S, V = ti.svd(mat)
    # now do polar decomposition into M = R * T, where R is rotation and T is translation matrix
    R = U @ V.transpose()
    # this is an improper rotation
    if R.determinant() < 0.0:
        # identify the smallest entry in S and flip its sign
        S[dim - 1, dim - 1] *= -1
        # recompute R using flipped stretch eigenvalues
        R = mat * V * S.inverse() * V.transpose()
    assert R.determinant() > 0.0

    # scale S to avoid small principal strains
    minval = 0.3                                   # 0.3^2 = 0.09, should suffice for most problems
    maxval = 2.0
    for i in ti.static(range(dim)):
        S[i, i] = clamp(minval, maxval, S(i, i))
    T = V * S * V.transpose()
    return R, T


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
def flatten_matrix(mat):
    # column first
    return ti.Vector([mat[i, j] for j in ti.static(range(mat.n)) for i in ti.static(range(mat.m))], float)


@ti.func
def unflatten_matrix(vec, matrix_like):
    mat = ti.Matrix.zero(float, matrix_like.m, matrix_like.n)
    for j in ti.static(range(matrix_like.n)):
        for i in ti.static(range(matrix_like.m)):
            mat[i, j] = vec[j * matrix_like.m + i]
    return mat


@ti.func
def contraction(mat1, mat2):
    sum = 0.
    for n in ti.static(range(mat1.n)):
        for m in ti.static(range(mat1.m)):
            sum += mat1[n, m] * mat2[n, m]
    return sum


@ti.func
def QRUnroll(matrix):
    assert matrix.n == matrix.m
    Q = ti.Matrix.zero(float, matrix.n, matrix.m)
    R = ti.Matrix.zero(float, matrix.n, matrix.m)
    for j in range(matrix.n):
        v = ti.Vector([matrix[i, j] for i in range(matrix.n)])
        for i in range(j):
            qi = ti.Vector([Q[k, i] for k in range(matrix.n)])
            R[i, j] = qi.dot(v)
            for k in range(matrix.n):
                v[k] -= R[i, j] * qi[k]
        R[j, j] = ti.sqrt(v.dot(v))
        if R[j, j] > DBL_EPSILON:
            for k in range(matrix.n):
                Q[k, j] = v[k] / R[j, j]
    return Q, R

@ti.func
def makePSD3x3(mat):
    S, V = ti.sym_eig(mat)
    lam = ti.Matrix.zero(float, S.n, S.n)
    for i in range(S.n):
        lam[i, i] = ti.max(0.0, S[i])
    return V @ lam @ V.transpose()

@ti.func
def shifted_qr(matrix):
    V = ti.Matrix.identity(float, matrix.n)
    for _ in range(100):
        mu = matrix[matrix.n-1, matrix.m-1]
        B = matrix - mu * ti.Matrix.identity(float, matrix.n)
        Q, R = QRUnroll(B)
        matrix = R @ Q + mu * ti.Matrix.identity(float, matrix.n)
        V = V @ Q
        real_off = 0.0
        for i, j in ti.ndrange(matrix.n, matrix.m):
            if i != j:
                real_off += ti.abs(matrix[i, j])
        if real_off < DBL_EPSILON:
            break
    return ti.Vector([matrix[i, i] for i in range(matrix.n)]), V


@ti.func
def makePSD(mat):
    lam, V = shifted_qr(mat)
    """ for i in range(lam.n):
        lam[i] = ti.max(0.0, lam[i]) """
    return mat#reconstruct_matrix(V, lam, V)


@ti.func
def reconstruct_matrix(left_matrix, vector, right_matrix):
    result = ti.Matrix.zero(float, left_matrix.n, right_matrix.m)
    for k in range(left_matrix.n):
        bk = vector[k]
        for i in range(right_matrix.m):
            aik = right_matrix[i, k]
            coef = aik * bk
            for j in range(left_matrix.m):
                result[i, j] += coef * left_matrix[j, k]
    return result

@ti.func
def global2local_mat3x3(hess, scale, rotation_matrix):
    return rotation_matrix.transpose() @ hess @ rotation_matrix / (scale * scale)

@ti.func
def local2global_mat3x3(hess, scale, rotation_matrix):
    return rotation_matrix @ hess @ rotation_matrix.transpose() * (scale * scale)