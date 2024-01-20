import taichi as ti

from src.utils.constants import ZEROVEC3f, SQRT3, PI, ZEROMAT3x3, DBL_EPSILON
from src.utils.ScalarFunction import clamp
from src.utils.TypeDefination import vec3f, mat3x3, mat4x4, mat2x2


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
    return matrix[0, 0] + matrix[1, 1] + matrix[2, 2]

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
def QR(matrix):
    Q, R = ZEROMAT3x3, ZEROMAT3x3
    col_A, col_Q = ZEROVEC3f, ZEROVEC3f
    for j in ti.static(range(matrix.m)):
        for i in ti.static(range(matrix.n)):
            col_A[i] = matrix[i, j]
            col_Q[i] = matrix[i, j]
        for k in range(j):
            R[k, j] = 0.
            for x in ti.static(range(matrix.m)):
                R[k, j] += col_A[x] * Q[x, k]
            for y in ti.static(range(matrix.m)):
                col_Q[y] -= R[k, j] * Q[y, k]
            
        temp = col_Q.norm()
        R[j, j] = temp
        for z in ti.static(range(matrix.m)):
            Q[z, j] = col_Q[z] / temp
    return Q, R


@ti.func
def sortEigenValueDescending(value):
    size = value.n

    for i in ti.static(range(size - 1)):
        k = i
        for j in range(i + 1, size):
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
        for j in range(i + 1, value.n):
            if value[k] > value[j]:
                k = j
        if k != i:
            temp = value[i]
            value[i] = value[k]
            value[k] = temp
    return value


@ti.func
def eigenvalue(matrix):
    value = ZEROVEC3f
    temp = matrix

    for _ in range(50):
        Q, R = QR(temp)
        temp = R @ Q

    for i in ti.static(range(3)):
        value[i] = temp[i, i]

    value = sortEigenValueAscending(value)
    return value


@ti.func
def eigenvector(matrix, value):
    eigenVector = ZEROMAT3x3
    temp = ZEROMAT3x3

    for count in ti.static(range(matrix.m)):
        eValue = value[count]
        temp = matrix
        for i in ti.static(range(temp.n)):
            temp[i, i] -= eValue
        for i in ti.static(range(temp.n - 1)):
            coe = temp[i, i]
            for j in range(i, temp.m):
                temp[i, j] /= coe
            for m in range(i+1, temp.n):
                coe = temp[m, i]
                for n in range(i, temp.m):
                    temp[m, n] -= coe * temp[i, n]
        
        sum1 = eigenVector[eigenVector.n - 1, count] = 1.
        for m in ti.static(range(temp.n - 2, -1, -1)):
            sum2 = 0.
            for n in range(m + 1, temp.m):
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
    value = eigenvalue(matrix)
    return value


@ti.func
def get_eigenvector(matrix):
    value = get_eigenvalue(matrix)
    eigenVector = eigenvector(matrix, value)
    return value, eigenVector


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
        for i in range(ndim):
            sub_vector = ti.Matrix.zero(ndim - 1, mdim - 1)
            for m in range(1, ndim):
                z = 0
                for n in range(ndim):
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

    for i in range(ndim):
        for j in range(mdim):
            p = 0
            for x in range(ndim):
                if x == i: continue
                q = 0
                for y in range(ndim):
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