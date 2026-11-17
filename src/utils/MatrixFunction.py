from src.utils.TypeDefinition import vec6f
from src.utils.constants import ZEROVEC6F
import taichi as ti

from src.utils.constants import ZEROMAT3X3, ZEROVEC3F, SQRT3, PI, DBL_EPSILON, THRESHOLD
from src.utils.ScalarFunction import clamp
from src.utils.TypeDefinition import vec3f, mat3x3, mat4x4, mat2x2, mat3x4


@ti.func
def diagonal(vec):
    return mat3x3([[vec[0], 0, 0], [0, vec[1], 0], [0, 0, vec[2]]])


@ti.func
def matrix_form(tensor):
    return mat3x3(
        [[tensor[0], tensor[3], tensor[5]], [tensor[3], tensor[1], tensor[4]], [tensor[5], tensor[4], tensor[2]]]
    )


@ti.func
def matrix_form_2d(tensor):
    return mat2x2([[tensor[0], tensor[3]], [tensor[3], tensor[1]]])


@ti.func
def matrix_form_3d_from_stress(stress):
    matrix = mat3x3([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    if ti.static(stress.n == 3):
        matrix = stress
    elif ti.static(stress.n == 2):
        matrix = mat3x3([[stress[0, 0], stress[0, 1], 0.0], [stress[1, 0], stress[1, 1], 0.0], [0.0, 0.0, 0.0]])
    else:
        matrix = matrix_form(stress)
    return matrix


@ti.func
def eig3_symmetric(stress):
    matrix = ZEROMAT3X3
    directors = ZEROMAT3X3
    xN1 = ZEROVEC3F
    xN2 = ZEROVEC3F
    xN3 = ZEROVEC3F

    matrix[0, 0] = stress[0]
    matrix[0, 1] = stress[3]
    matrix[0, 2] = stress[5]
    matrix[1, 0] = stress[3]
    matrix[1, 1] = stress[1]
    matrix[1, 2] = stress[4]
    matrix[2, 0] = stress[5]
    matrix[2, 1] = stress[4]
    matrix[2, 2] = stress[2]

    directors[0, 0] = 1.0
    directors[1, 0] = 0.0
    directors[2, 0] = 0.0
    directors[0, 1] = 0.0
    directors[1, 1] = 1.0
    directors[2, 1] = 0.0
    directors[0, 2] = 0.0
    directors[1, 2] = 0.0
    directors[2, 2] = 1.0

    abs_max_s = 0.0
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            abs_max_s = max(abs_max_s, abs(matrix[i, j]))
    tolerance = 1.0e-20 * abs_max_s
    iteration = 0
    max_iteration = 50

    while iteration < max_iteration and (abs(matrix[0, 1]) + abs(matrix[1, 2]) + abs(matrix[0, 2])) > tolerance:
        iteration += 1
        ip, iq = 0, 0
        for k in ti.static(range(3)):
            if k == 0:
                ip, iq = 0, 1
            elif k == 1:
                ip, iq = 1, 2
            else:
                ip, iq = 0, 2
            if abs(matrix[ip, iq]) > 1.0e-50:
                tau = (matrix[iq, iq] - matrix[ip, ip]) / (2.0 * matrix[ip, iq])
                sign_tau = 1.0 if tau >= 0.0 else -1.0
                tangent = sign_tau / (abs(tau) + ti.sqrt(1.0 + tau * tau))
                cosine = 1.0 / ti.sqrt(1.0 + tangent * tangent)
                sine = tangent * cosine

                a1p = cosine * matrix[0, ip] - sine * matrix[0, iq]
                a2p = cosine * matrix[1, ip] - sine * matrix[1, iq]
                a3p = cosine * matrix[2, ip] - sine * matrix[2, iq]
                matrix[0, iq] = sine * matrix[0, ip] + cosine * matrix[0, iq]
                matrix[1, iq] = sine * matrix[1, ip] + cosine * matrix[1, iq]
                matrix[2, iq] = sine * matrix[2, ip] + cosine * matrix[2, iq]
                matrix[0, ip] = a1p
                matrix[1, ip] = a2p
                matrix[2, ip] = a3p

                v1p = cosine * directors[0, ip] - sine * directors[0, iq]
                v2p = cosine * directors[1, ip] - sine * directors[1, iq]
                v3p = cosine * directors[2, ip] - sine * directors[2, iq]
                directors[0, iq] = sine * directors[0, ip] + cosine * directors[0, iq]
                directors[1, iq] = sine * directors[1, ip] + cosine * directors[1, iq]
                directors[2, iq] = sine * directors[2, ip] + cosine * directors[2, iq]
                directors[0, ip] = v1p
                directors[1, ip] = v2p
                directors[2, ip] = v3p

                ap1 = cosine * matrix[ip, 0] - sine * matrix[iq, 0]
                ap2 = cosine * matrix[ip, 1] - sine * matrix[iq, 1]
                ap3 = cosine * matrix[ip, 2] - sine * matrix[iq, 2]
                matrix[iq, 0] = sine * matrix[ip, 0] + cosine * matrix[iq, 0]
                matrix[iq, 1] = sine * matrix[ip, 1] + cosine * matrix[iq, 1]
                matrix[iq, 2] = sine * matrix[ip, 2] + cosine * matrix[iq, 2]
                matrix[ip, 0] = ap1
                matrix[ip, 1] = ap2
                matrix[ip, 2] = ap3

    s1 = matrix[0, 0]
    s2 = matrix[1, 1]
    s3 = matrix[2, 2]
    mean_stress = (s1 + s2 + s3) / 3.0
    deviatoric_norm = ti.sqrt(((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2) / 2.0)

    is1 = 0
    is2 = 1
    is3 = 2
    if s1 > s2:
        temp = s2
        s2 = s1
        s1 = temp
        itemp = is2
        is2 = is1
        is1 = itemp
    if s2 > s3:
        temp = s3
        s3 = s2
        s2 = temp
        itemp = is3
        is3 = is2
        is2 = itemp
    if s1 > s2:
        temp = s2
        s2 = s1
        s1 = temp
        itemp = is2
        is2 = is1
        is1 = itemp
    for i in ti.static(range(3)):
        xN1[i] = directors[i, is1]
        xN2[i] = directors[i, is2]
        xN3[i] = directors[i, is3]
    return xN1, xN2, xN3, s1, s2, s3, mean_stress, deviatoric_norm


@ti.func
def inverse_matrix_2x2(matrix):
    det = matrix[0, 0] * matrix[1, 1] - matrix[1, 0] * matrix[0, 1]
    return 1.0 / det * mat2x2([matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]])


@ti.func
def cwise_product(matrix1, matrix2):
    return mat3x3(
        matrix1[0, 0] * matrix2[0, 0],
        matrix1[0, 1] * matrix2[0, 1],
        matrix1[0, 2] * matrix2[0, 2],
        matrix1[1, 0] * matrix2[1, 0],
        matrix1[1, 1] * matrix2[1, 1],
        matrix1[1, 2] * matrix2[1, 2],
        matrix1[2, 0] * matrix2[2, 0],
        matrix1[2, 1] * matrix2[2, 1],
        matrix1[2, 2] * matrix2[2, 2],
    )


@ti.func
def sum_cwise_product(matrix1, matrix2):
    return (
        matrix1[0, 0] * matrix2[0, 0]
        + matrix1[0, 1] * matrix2[0, 1]
        + matrix1[0, 2] * matrix2[0, 2]
        + matrix1[1, 0] * matrix2[1, 0]
        + matrix1[1, 1] * matrix2[1, 1]
        + matrix1[1, 2] * matrix2[1, 2]
        + matrix1[2, 0] * matrix2[2, 0]
        + matrix1[2, 1] * matrix2[2, 1]
        + matrix1[2, 2] * matrix2[2, 2]
    )


@ti.func
def trace(matrix):
    ti.static_assert(matrix.n == matrix.m)
    matrix_trace = 0.0
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
    c1 = matrix[0, 0] * matrix[1, 1] + matrix[0, 0] * matrix[2, 2] + matrix[1, 1] * matrix[2, 2] - (dd + ee + ff)
    c0 = (
        matrix[2, 2] * dd
        + matrix[0, 0] * ee
        + matrix[1, 1] * ff
        - matrix[0, 0] * matrix[1, 1] * matrix[2, 2]
        - 2.0 * matrix[0, 2] * matrix[0, 1] * matrix[1, 2]
    )

    p = m * m - 3.0 * c1
    q = m * (p - 1.5 * c1) - 13.5 * c0
    sqrt_p = ti.sqrt(ti.abs(p))
    phi = 27.0 * (0.25 * c1 * c1 * (p - c1) + c0 * (q + 6.75 * c0))
    phi = (1.0 / 3.0) * ti.atan2(ti.sqrt(ti.abs(phi)), q)

    c = sqrt_p * ti.cos(phi)
    s = (1.0 / SQRT3) * sqrt_p * ti.sin(phi)
    eigenvalues = ZEROVEC3F
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
        for i in ti.static(range(col + 1, matrix.n)):
            v[i] = matrix[i, col]
        alpha = ti.sqrt(v.dot(v))
        if v[col + 1] > 0:
            alpha = -alpha
        v[col + 1] -= alpha
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
def qr(matrix):
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
def sort_eigen_value_descending(value):
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
def sort_eigen_value_ascending(value):
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
        Q, R = qr(temp)
        temp = R @ Q

    for i in ti.static(range(matrix.n)):
        value[i] = temp[i, i]

    value = sort_eigen_value_ascending(value)
    return value


@ti.func
def eigenvector(matrix, value):
    eigen_vector = ti.Matrix.zero(float, matrix.n, matrix.m)
    temp = ti.Matrix.zero(float, matrix.n, matrix.m)

    for count in ti.static(range(matrix.m)):
        e_value = value[count]
        temp = matrix
        for i in ti.static(range(temp.n)):
            temp[i, i] -= e_value
        for i in ti.static(range(temp.n - 1)):
            coe = temp[i, i]
            for j in ti.static(range(i, temp.m)):
                temp[i, j] /= coe if ti.abs(coe) > DBL_EPSILON else 0.0
            for m in ti.static(range(i + 1, temp.n)):
                coe = temp[m, i]
                for n in ti.static(range(i, temp.m)):
                    temp[m, n] -= coe * temp[i, n]

        sum1 = eigen_vector[eigen_vector.n - 1, count] = 1.0
        for m in ti.static(range(temp.n - 2, -1, -1)):
            sum2 = 0.0
            for n in ti.static(range(m + 1, temp.m)):
                sum2 += temp[m, n] * eigen_vector[n, count]
            sum2 = -sum2 / temp[m, m]
            sum1 += sum2 * sum2
            eigen_vector[m, count] = sum2

        sum1 = ti.sqrt(sum1)
        for i in ti.static(range(eigen_vector.n)):
            eigen_vector[i, count] /= sum1

    return eigen_vector


@ti.func
def get_eigenvector(matrix):
    value = eigenvalue(matrix)
    eigen_vector = eigenvector(matrix, value)
    return value, eigen_vector


@ti.func
def eig(matrix):
    H, Qh = hessenberg(matrix)
    Q_tot = Qh
    for _ in range(50):
        Q, R = qr(H)
        H = R @ Q
        Q_tot = Q_tot @ Q
    eigenvalues = ti.Vector([H[i, i] for i in ti.static(range(matrix.n))])
    eigenvectors = Q_tot
    return eigenvalues, eigenvectors


@ti.func
def get_eigenvalue_3x3(matrix):
    value = ZEROVEC3F
    mean = matrix.trace() / 3.0
    deviator = matrix - mean * ti.Matrix.identity(float, 3)
    squared_norm = 0.0
    for row, column in ti.static(ti.ndrange(3, 3)):
        squared_norm += deviator[row, column] * deviator[row, column]
    p_squared = squared_norm / 6.0

    # For an isotropic tensor all three roots coincide.  Handling that case
    # explicitly also avoids the undefined 0 / 0 in the trigonometric form.
    if p_squared <= DBL_EPSILON * ti.max(1.0, mean * mean):
        value = vec3f(mean, mean, mean)
    else:
        p = ti.sqrt(p_squared)
        normalized = deviator / p
        determinant = (
            normalized[0, 0] * (normalized[1, 1] * normalized[2, 2] - normalized[1, 2] * normalized[2, 1])
            - normalized[0, 1] * (normalized[1, 0] * normalized[2, 2] - normalized[1, 2] * normalized[2, 0])
            + normalized[0, 2] * (normalized[1, 0] * normalized[2, 1] - normalized[1, 1] * normalized[2, 0])
        )
        angle = ti.acos(clamp(-1.0, 1.0, 0.5 * determinant)) / 3.0
        value[2] = mean + 2.0 * p * ti.cos(angle)
        value[0] = mean + 2.0 * p * ti.cos(angle + 2.0 * PI / 3.0)
        value[1] = 3.0 * mean - value[0] - value[2]
    return value


@ti.func
def principal_tensor(matrix):
    directors, values, _ = ti.svd(matrix)
    return vec3f([values[2, 2], values[1, 1], values[0, 0]]), mat3x3(
        [
            [directors[0, 2], directors[0, 1], directors[0, 0]],
            [directors[1, 2], directors[1, 1], directors[1, 0]],
            [directors[2, 2], directors[2, 1], directors[2, 0]],
        ]
    )


@ti.func
def determinant_recursive(matrix):
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
                        sub_vector[m - 1, z] = matrix[m, n]
                        z += 1
            value += sign * matrix[0, i] * determinant_recursive(sub_vector)
            sign = -sign
    return value


@ti.func
def get_cofactor(matrix):
    ndim = ti.static(matrix.n)
    mdim = ti.static(matrix.m)
    solution = ti.Matrix.zero(ndim, mdim)
    sub_vector = ti.Matrix.zero(ndim - 1, mdim - 1)

    for i in ti.static(range(ndim)):
        for j in ti.static(range(mdim)):
            p = 0
            for x in ti.static(range(ndim)):
                if x == i:
                    continue
                q = 0
                for y in ti.static(range(ndim)):
                    if y == j:
                        continue
                    sub_vector[p, q] = matrix[x, y]
                    q += 1
                p += 1
            solution[i, j] = ti.pow(-1, i + j) * determinant_recursive(sub_vector)
    return solution


@ti.func
def get_transpose(matrix):
    ndim = ti.static(matrix.n)
    mdim = ti.static(matrix.m)
    solution = ti.Matrix.zero(ndim, mdim)
    for i in ti.static(range(ndim)):
        for j in ti.static(range(mdim)):
            solution[j, i] = matrix[i, j]


@ti.func
def get_inverse(matrix):
    ndim = ti.static(matrix.n)
    mdim = ti.static(matrix.m)
    d = 1.0 / determinant_recursive(matrix)
    solution = ti.Matrix.zero(ndim, mdim)
    for i in ti.static(range(ndim)):
        for j in ti.static(range(mdim)):
            solution[i, j] = matrix[i, j] * d
    return get_transpose(get_cofactor(solution))


@ti.func
def periodic_matrix_entry(mat, i, j, n):
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
    minor0 = ti.Matrix(
        [
            [mat[1, 1], mat[1, 2], mat[1, 3]],
            [mat[2, 1], mat[2, 2], mat[2, 3]],
            [mat[3, 1], mat[3, 2], mat[3, 3]],
        ]
    )
    minor1 = ti.Matrix(
        [
            [mat[1, 0], mat[1, 2], mat[1, 3]],
            [mat[2, 0], mat[2, 2], mat[2, 3]],
            [mat[3, 0], mat[3, 2], mat[3, 3]],
        ]
    )
    minor2 = ti.Matrix(
        [
            [mat[1, 0], mat[1, 1], mat[1, 3]],
            [mat[2, 0], mat[2, 1], mat[2, 3]],
            [mat[3, 0], mat[3, 1], mat[3, 3]],
        ]
    )
    minor3 = ti.Matrix(
        [
            [mat[1, 0], mat[1, 1], mat[1, 2]],
            [mat[2, 0], mat[2, 1], mat[2, 2]],
            [mat[3, 0], mat[3, 1], mat[3, 2]],
        ]
    )
    return (
        mat[0, 0] * determinant3x3(minor0)
        - mat[0, 1] * determinant3x3(minor1)
        + mat[0, 2] * determinant3x3(minor2)
        - mat[0, 3] * determinant3x3(minor3)
    )


@ti.func
def cofactor5x5(mat, excluded_row, excluded_column):
    # With four retained rows/columns, rotating the natural minor ordering
    # contributes (-1)^excluded_row and (-1)^excluded_column.  Their product
    # is exactly the cofactor sign, so this cyclic minor is the cofactor.
    minor = ti.Matrix(
        [
            [
                periodic_matrix_entry(mat, excluded_row + 1, excluded_column + 1, 5),
                periodic_matrix_entry(mat, excluded_row + 1, excluded_column + 2, 5),
                periodic_matrix_entry(mat, excluded_row + 1, excluded_column + 3, 5),
                periodic_matrix_entry(mat, excluded_row + 1, excluded_column + 4, 5),
            ],
            [
                periodic_matrix_entry(mat, excluded_row + 2, excluded_column + 1, 5),
                periodic_matrix_entry(mat, excluded_row + 2, excluded_column + 2, 5),
                periodic_matrix_entry(mat, excluded_row + 2, excluded_column + 3, 5),
                periodic_matrix_entry(mat, excluded_row + 2, excluded_column + 4, 5),
            ],
            [
                periodic_matrix_entry(mat, excluded_row + 3, excluded_column + 1, 5),
                periodic_matrix_entry(mat, excluded_row + 3, excluded_column + 2, 5),
                periodic_matrix_entry(mat, excluded_row + 3, excluded_column + 3, 5),
                periodic_matrix_entry(mat, excluded_row + 3, excluded_column + 4, 5),
            ],
            [
                periodic_matrix_entry(mat, excluded_row + 4, excluded_column + 1, 5),
                periodic_matrix_entry(mat, excluded_row + 4, excluded_column + 2, 5),
                periodic_matrix_entry(mat, excluded_row + 4, excluded_column + 3, 5),
                periodic_matrix_entry(mat, excluded_row + 4, excluded_column + 4, 5),
            ],
        ]
    )
    return determinant4x4(minor)


@ti.func
def determinant5x5(mat):
    det = mat[0, 0] * 0
    for column in ti.static(range(5)):
        det += mat[0, column] * cofactor5x5(mat, 0, column)
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
                periodic_matrix_entry(mat, i + 1, j + 1, 3) * periodic_matrix_entry(mat, i + 2, j + 2, 3)
                - periodic_matrix_entry(mat, i + 2, j + 1, 3) * periodic_matrix_entry(mat, i + 1, j + 2, 3)
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
                        periodic_matrix_entry(mat, i + 1, j + 1, 4)
                        * (
                            periodic_matrix_entry(mat, i + 2, j + 2, 4) * periodic_matrix_entry(mat, i + 3, j + 3, 4)
                            - periodic_matrix_entry(mat, i + 3, j + 2, 4) * periodic_matrix_entry(mat, i + 2, j + 3, 4)
                        )
                        - periodic_matrix_entry(mat, i + 2, j + 1, 4)
                        * (
                            periodic_matrix_entry(mat, i + 1, j + 2, 4) * periodic_matrix_entry(mat, i + 3, j + 3, 4)
                            - periodic_matrix_entry(mat, i + 3, j + 2, 4) * periodic_matrix_entry(mat, i + 1, j + 3, 4)
                        )
                        + periodic_matrix_entry(mat, i + 3, j + 1, 4)
                        * (
                            periodic_matrix_entry(mat, i + 1, j + 2, 4) * periodic_matrix_entry(mat, i + 2, j + 3, 4)
                            - periodic_matrix_entry(mat, i + 2, j + 2, 4) * periodic_matrix_entry(mat, i + 1, j + 3, 4)
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
    inverse_determinant = 1.0 / determinant5x5(mat)
    return inverse_determinant * ti.Matrix(
        [[cofactor5x5(mat, column, row) for column in ti.static(range(5))] for row in ti.static(range(5))]
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
                mat[i, j] = 0.0
    return mat


@ti.func
def l_uinverse(mat):
    L, U = polar_decomposition(mat)
    linv = ti.Matrix.zero(float, mat.n, mat.m)
    uinv = ti.Matrix.zero(float, mat.n, mat.m)
    for j in ti.static(range(mat.n)):
        for i in ti.static(range(j, mat.m)):
            if i == j:
                linv[i, j] = 1.0 / L[i, j]
            elif i < j:
                linv[i, j] = 0.0
            else:
                s = 0.0
                for k in ti.static(range(j, i)):
                    s += L[i, k] * linv[k, j]
                linv[i, j] = -linv[j, j] * s
    for j in ti.static(range(mat.n)):
        for i in ti.static(range(j, -1, -1)):
            if i == j:
                uinv[i, j] = 1 / U[i, j]
            elif i > j:
                uinv[i, j] = 0
            else:
                s = 0.0
                for k in ti.static(range(i + 1, j + 1)):
                    s += U[i, k] * uinv[k, j]
                uinv[i, j] = -1 / U[i, i] * s
    return uinv @ linv


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
        L[i, i] = 1.0
    for j in ti.static(range(mat.n)):
        U[0, j] = mat[0, j]
    for i in ti.static(range(1, mat.n)):
        L[i, 0] = mat[i, 0] / U[0, 0]
    for i in ti.static(range(mat.n)):
        for j in ti.static(range(i, mat.n)):
            s = 0.0
            for k in ti.static(range(i)):
                s += L[i, k] * U[k, j]
            U[i, j] = mat[i, j] - s
        for d in ti.static(range(i, mat.n)):
            s = 0.0
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
    minval = 0.3  # 0.3^2 = 0.09, should suffice for most problems
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
    sum = 0.0
    for n in ti.static(range(mat1.n)):
        for m in ti.static(range(mat1.m)):
            sum += mat1[n, m] * mat2[n, m]
    return sum


@ti.func
def qr_unroll(matrix):
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
def make_psd3x3(mat):
    S, V = ti.sym_eig(mat)
    lam = ti.Matrix.zero(float, S.n, S.n)
    for i in range(S.n):
        lam[i, i] = ti.max(0.0, S[i])
    return V @ lam @ V.transpose()


@ti.func
def shifted_qr(matrix):
    V = ti.Matrix.identity(float, matrix.n)
    for _ in range(100):
        mu = matrix[matrix.n - 1, matrix.m - 1]
        B = matrix - mu * ti.Matrix.identity(float, matrix.n)
        Q, R = qr_unroll(B)
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
def make_psd(mat):
    lam, V = shifted_qr(mat)
    """ for i in range(lam.n):
        lam[i] = ti.max(0.0, lam[i]) """
    return mat  # reconstruct_matrix(V, lam, V)


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


@ti.func
def affine_velocity_matrix(Bp, Wp, grid_size):
    scaled_Bp, scaled_Wp = Bp, Wp
    for i, j in ti.static(ti.ndrange(Bp.n, Bp.m)):
        scaled_Bp[i, j] /= grid_size[j]
        scaled_Wp[i, j] /= grid_size[i] * grid_size[j]
    result = Wp * 0.0
    # Test rank in grid units so mesh refinement cannot silently disable APIC.
    if ti.abs(scaled_Wp.determinant()) > THRESHOLD:
        result = scaled_Bp @ scaled_Wp.inverse()
        for i, j in ti.static(ti.ndrange(Bp.n, Bp.m)):
            result[i, j] /= grid_size[j]
    return truncation(result)


@ti.func
def _dense_block_matvec(h, x, dim: ti.template()):
    y = ti.Vector.zero(float, dim)
    for i in range(dim):
        for j in range(dim):
            y[i] += h[i * dim + j] * x[j]
    return y


@ti.func
def _dense_block_transpose_matvec(h, x, dim: ti.template()):
    y = ti.Vector.zero(float, dim)
    for i in range(dim):
        for j in range(dim):
            y[i] += h[j * dim + i] * x[j]
    return y


@ti.func
def _apply_dense_block_jacobi(h, r, dim: ti.template()):
    """Dense counterpart of the scale-aware block-Jacobi solve above."""
    z = r
    scale = 0.0
    finite_input = 1
    for row in range(dim):
        for column in range(dim):
            value = h[row * dim + column]
            if not (-ti.math.inf < value and value < ti.math.inf):
                finite_input = 0
            scale = ti.max(scale, ti.abs(value))

    condition_tolerance = 1.0e-12
    determinant_floor = 1.0e-30
    if finite_input != 0 and scale > 0.0 and scale < ti.math.inf:
        for row in ti.static(range(4)):
            if ti.static(row < dim):
                diagonal = h[row * dim + row]
                denominator = scale
                if ti.abs(diagonal) > condition_tolerance * scale:
                    denominator = diagonal
                value = r[row] / denominator
                if -ti.math.inf < value and value < ti.math.inf:
                    z[row] = value

    if ti.static(dim == 1):
        if finite_input != 0 and scale > 0.0 and scale < ti.math.inf:
            z[0] = r[0] / h[0]
    elif ti.static(dim == 2):
        if finite_input != 0 and scale > 0.0 and scale < ti.math.inf:
            a00, a01 = h[0] / scale, h[1] / scale
            a10, a11 = h[2] / scale, h[3] / scale
            det = a00 * a11 - a01 * a10
            if ti.abs(det) > determinant_floor:
                inverse00, inverse01 = a11 / det, -a01 / det
                inverse10, inverse11 = -a10 / det, a00 / det
                matrix_norm = ti.max(
                    ti.abs(a00) + ti.abs(a01),
                    ti.abs(a10) + ti.abs(a11),
                )
                inverse_norm = ti.max(
                    ti.abs(inverse00) + ti.abs(inverse01),
                    ti.abs(inverse10) + ti.abs(inverse11),
                )
                rcond = 0.0
                if matrix_norm > 0.0 and inverse_norm > 0.0:
                    rcond = 1.0 / (matrix_norm * inverse_norm)
                if rcond > condition_tolerance:
                    candidate0 = (inverse00 * r[0] + inverse01 * r[1]) / scale
                    candidate1 = (inverse10 * r[0] + inverse11 * r[1]) / scale
                    if (
                        -ti.math.inf < candidate0
                        and candidate0 < ti.math.inf
                        and -ti.math.inf < candidate1
                        and candidate1 < ti.math.inf
                    ):
                        z[0] = candidate0
                        z[1] = candidate1
    elif ti.static(dim == 3):
        if finite_input != 0 and scale > 0.0 and scale < ti.math.inf:
            a00, a01, a02 = h[0] / scale, h[1] / scale, h[2] / scale
            a10, a11, a12 = h[3] / scale, h[4] / scale, h[5] / scale
            a20, a21, a22 = h[6] / scale, h[7] / scale, h[8] / scale
            c00 = a11 * a22 - a12 * a21
            c01 = a02 * a21 - a01 * a22
            c02 = a01 * a12 - a02 * a11
            c10 = a12 * a20 - a10 * a22
            c11 = a00 * a22 - a02 * a20
            c12 = a02 * a10 - a00 * a12
            c20 = a10 * a21 - a11 * a20
            c21 = a01 * a20 - a00 * a21
            c22 = a00 * a11 - a01 * a10
            det = a00 * c00 + a01 * c10 + a02 * c20
            if ti.abs(det) > determinant_floor:
                inverse00, inverse01, inverse02 = c00 / det, c01 / det, c02 / det
                inverse10, inverse11, inverse12 = c10 / det, c11 / det, c12 / det
                inverse20, inverse21, inverse22 = c20 / det, c21 / det, c22 / det
                matrix_norm = ti.max(
                    ti.abs(a00) + ti.abs(a01) + ti.abs(a02),
                    ti.abs(a10) + ti.abs(a11) + ti.abs(a12),
                    ti.abs(a20) + ti.abs(a21) + ti.abs(a22),
                )
                inverse_norm = ti.max(
                    ti.abs(inverse00) + ti.abs(inverse01) + ti.abs(inverse02),
                    ti.abs(inverse10) + ti.abs(inverse11) + ti.abs(inverse12),
                    ti.abs(inverse20) + ti.abs(inverse21) + ti.abs(inverse22),
                )
                rcond = 0.0
                if matrix_norm > 0.0 and inverse_norm > 0.0:
                    rcond = 1.0 / (matrix_norm * inverse_norm)
                if rcond > condition_tolerance:
                    candidate0 = (inverse00 * r[0] + inverse01 * r[1] + inverse02 * r[2]) / scale
                    candidate1 = (inverse10 * r[0] + inverse11 * r[1] + inverse12 * r[2]) / scale
                    candidate2 = (inverse20 * r[0] + inverse21 * r[1] + inverse22 * r[2]) / scale
                    if (
                        -ti.math.inf < candidate0
                        and candidate0 < ti.math.inf
                        and -ti.math.inf < candidate1
                        and candidate1 < ti.math.inf
                        and -ti.math.inf < candidate2
                        and candidate2 < ti.math.inf
                    ):
                        z[0] = candidate0
                        z[1] = candidate1
                        z[2] = candidate2
    else:
        if finite_input != 0 and scale > 0.0 and scale < ti.math.inf:
            A = ti.Matrix.zero(float, 4, 4)
            for i in range(4):
                for j in range(4):
                    A[i, j] = h[i * 4 + j] / scale
            det = A.determinant()
            if ti.abs(det) > determinant_floor:
                inverse = A.inverse()
                matrix_norm = 0.0
                inverse_norm = 0.0
                finite_inverse = 1
                for row in range(4):
                    row_norm = 0.0
                    inverse_row_norm = 0.0
                    for column in range(4):
                        row_norm += ti.abs(A[row, column])
                        inverse_row_norm += ti.abs(inverse[row, column])
                        if not (-ti.math.inf < inverse[row, column] and inverse[row, column] < ti.math.inf):
                            finite_inverse = 0
                    matrix_norm = ti.max(matrix_norm, row_norm)
                    inverse_norm = ti.max(inverse_norm, inverse_row_norm)
                rcond = 0.0
                if matrix_norm > 0.0 and inverse_norm > 0.0:
                    rcond = 1.0 / (matrix_norm * inverse_norm)
                if finite_inverse != 0 and rcond > condition_tolerance:
                    candidate = (inverse @ r) / scale
                    finite_candidate = 1
                    for row in ti.static(range(4)):
                        if not (-ti.math.inf < candidate[row] and candidate[row] < ti.math.inf):
                            finite_candidate = 0
                    if finite_candidate != 0:
                        z = candidate
    return z


@ti.func
def diag_nd(vector):
    matrix = ti.Matrix.zero(ti.f64, vector.n, vector.n)
    for dimension in range(vector.n):
        matrix[dimension, dimension] = vector[dimension]
    return matrix


@ti.func
def symmetric_eigendecomposition_nd(matrix):
    """Device-side eigendecomposition of a fixed-size symmetric matrix.

    The input is symmetrized, matching IPC's local ``make_pd`` operation.
    The 2x2 case uses a scale-stable closed form.  Every larger fixed block
    uses cyclic Jacobi rotations and therefore never leaves the CUDA kernel.
    Taichi 1.7's ``ti.sym_eig`` is intentionally not used for 3x3 blocks: on
    rank-deficient matrices with a repeated positive eigenvalue (the exact
    spectrum of sliding Coulomb friction), it can manufacture an eigenvalue
    many orders of magnitude larger than the input.
    """
    symmetric = 0.5 * (matrix + matrix.transpose())
    eigenvalues = ti.Vector.zero(ti.f64, matrix.n)
    eigenvectors = ti.Matrix.identity(ti.f64, matrix.n)
    if ti.static(matrix.n == 2):
        a = symmetric[0, 0]
        b = symmetric[0, 1]
        d = symmetric[1, 1]
        midpoint = 0.5 * (a + d)
        half_difference = 0.5 * (a - d)

        # Normalize the two coordinates of the spectral radius before taking
        # their norm.  Besides avoiding overflow/underflow, this is homogeneous
        # in the matrix entries: scaling the whole matrix cannot change the
        # eigenvectors or trigger an absolute-epsilon branch.
        spectral_scale = ti.max(ti.abs(half_difference), ti.abs(b))
        radius = 0.0
        cosine_twice = 1.0
        sine_twice = 0.0
        if spectral_scale > 0.0:
            scaled_difference = half_difference / spectral_scale
            scaled_off_diagonal = b / spectral_scale
            scaled_radius = ti.sqrt(scaled_difference * scaled_difference + scaled_off_diagonal * scaled_off_diagonal)
            radius = spectral_scale * scaled_radius
            cosine_twice = scaled_difference / scaled_radius
            sine_twice = scaled_off_diagonal / scaled_radius

        lambda1 = midpoint + radius
        lambda2 = midpoint - radius
        cosine = ti.sqrt(ti.max(0.5 * (1.0 + cosine_twice), 0.0))
        sine = ti.sqrt(ti.max(0.5 * (1.0 - cosine_twice), 0.0))
        if sine_twice < 0.0:
            sine = -sine
        vector1 = ti.Vector([cosine, sine])
        vector2 = ti.Vector([-vector1[1], vector1[0]])
        eigenvalues = ti.Vector([lambda1, lambda2])
        eigenvectors = ti.Matrix.cols([vector1, vector2])
    else:
        diagonalized = symmetric
        # Twelve cyclic sweeps are normally sufficient for these small
        # self-adjoint blocks; twenty-four gives a conservative f64 margin
        # for clustered spectra without any CPU convergence check.  These
        # loops intentionally remain serial because every Jacobi pivot reads
        # the result of the previous pivot.  ``while`` prevents Taichi from
        # statically expanding blocks larger than 3x3 while preserving that
        # exact sweep/pivot order on the device.
        sweep = 0
        while sweep < 24:
            p = 0
            while p < matrix.n:
                q = p + 1
                while q < matrix.n:
                    app = diagonalized[p, p]
                    aqq = diagonalized[q, q]
                    apq = diagonalized[p, q]
                    pivot_scale = ti.max(ti.abs(app) + ti.abs(aqq), 1.0e-30)
                    if ti.abs(apq) > 1.0e-14 * pivot_scale:
                        tau = (aqq - app) / (2.0 * apq)
                        sign_tau = 1.0
                        if tau < 0.0:
                            sign_tau = -1.0
                        tangent = sign_tau / (ti.abs(tau) + ti.sqrt(1.0 + tau * tau))
                        cosine = 1.0 / ti.sqrt(1.0 + tangent * tangent)
                        sine = tangent * cosine

                        k = 0
                        while k < matrix.n:
                            if k != p and k != q:
                                akp = diagonalized[k, p]
                                akq = diagonalized[k, q]
                                rotated_kp = cosine * akp - sine * akq
                                rotated_kq = sine * akp + cosine * akq
                                diagonalized[k, p] = rotated_kp
                                diagonalized[p, k] = rotated_kp
                                diagonalized[k, q] = rotated_kq
                                diagonalized[q, k] = rotated_kq
                            k += 1

                        diagonalized[p, p] = app - tangent * apq
                        diagonalized[q, q] = aqq + tangent * apq
                        diagonalized[p, q] = 0.0
                        diagonalized[q, p] = 0.0

                        k = 0
                        while k < matrix.n:
                            vkp = eigenvectors[k, p]
                            vkq = eigenvectors[k, q]
                            eigenvectors[k, p] = cosine * vkp - sine * vkq
                            eigenvectors[k, q] = sine * vkp + cosine * vkq
                            k += 1
                    q += 1
                p += 1
            sweep += 1

        eigenmode = 0
        while eigenmode < matrix.n:
            eigenvalues[eigenmode] = diagonalized[eigenmode, eigenmode]
            eigenmode += 1
    return eigenvalues, eigenvectors


@ti.func
def psd_project_nd(matrix):
    """Spectrally clamp a fixed-size symmetric matrix on the device.

    The returned matrix is ``V max(D, 0) V.T`` for complete local blocks.
    """
    eigenvalues, eigenvectors = symmetric_eigendecomposition_nd(matrix)
    projected = ti.Matrix.zero(ti.f64, matrix.n, matrix.n)
    row = 0
    while row < matrix.n:
        column = 0
        while column < matrix.n:
            value = 0.0
            eigenmode = 0
            while eigenmode < matrix.n:
                eigenvalue = ti.max(eigenvalues[eigenmode], 0.0)
                value += eigenvalue * eigenvectors[row, eigenmode] * eigenvectors[column, eigenmode]
                eigenmode += 1
            projected[row, column] = value
            column += 1
        row += 1
    projected = 0.5 * (projected + projected.transpose())
    return projected


@ti.func
def psd_project_gershgorin_nd(matrix):
    """Cheap device PSD majorizer for large local contact blocks.

    The symmetric matrix is kept off diagonal and receives one global diagonal
    shift that dominates all Gershgorin radii.  The result is symmetric
    diagonally dominant with non-negative diagonal, hence PSD, without the
    cubic-size Jacobi eigendecomposition used by ``psd_project_nd``.  This is the same
    device-local projection contract needed by FEM/IGA-style block assembly,
    but keeps CUDA compilation bounded for 12x12 PT/EE contact blocks.
    """
    # Keep the dimensions compile-time constants.  A dynamic ``while`` over
    # ``matrix.n`` makes CUDA instantiate the loop-control path for every
    # caller; IGA's block assembly uses the same fixed-size device pattern.
    symmetric = 0.5 * (matrix + matrix.transpose())
    # Use one global diagonal shift instead of constructing a second matrix
    # element-by-element.  It is the same Gershgorin PSD majorizer, but keeps
    # the generated CUDA IR small for the 6x6 cloth material block.
    lower_bound = 1.0e30
    for row in ti.static(range(matrix.n)):
        radius = 0.0
        for column in ti.static(range(matrix.n)):
            if ti.static(column != row):
                radius += ti.abs(symmetric[row, column])
        lower_bound = ti.min(lower_bound, symmetric[row, row] - radius)
    shift = ti.max(-lower_bound, 0.0)
    return symmetric + shift * ti.Matrix.identity(ti.f64, matrix.n)


@ti.func
def outer_product_nd(vector1, vector2):
    """Dimension-generic outer product used by the NURBS derivatives."""
    matrix = ti.Matrix.zero(ti.f64, vector1.n, vector2.n)
    if ti.static(vector1.n * vector2.n <= 9):
        for i in ti.static(range(vector1.n)):
            for j in ti.static(range(vector2.n)):
                matrix[i, j] = vector1[i] * vector2[j]
    else:
        flat = 0
        while flat < vector1.n * vector2.n:
            i = flat // vector2.n
            j = flat - i * vector2.n
            matrix[i, j] = vector1[i] * vector2[j]
            flat += 1
    return matrix


@ti.func
def _stress_increment_voigt(stress):
    increment = ZEROVEC6F
    if ti.static(stress.n == 6):
        increment = stress
    elif ti.static(stress.n == 2):
        increment = vec6f(stress[0, 0], stress[1, 1], 0.0, 0.5 * (stress[0, 1] + stress[1, 0]), 0.0, 0.0)
    else:
        increment = vec6f(
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            0.5 * (stress[0, 1] + stress[1, 0]),
            0.5 * (stress[1, 2] + stress[2, 1]),
            0.5 * (stress[0, 2] + stress[2, 0]),
        )
    return increment


@ti.func
def _stress_increment_mat3(stress):
    increment = mat3x3([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    if ti.static(stress.n == 6):
        increment = mat3x3(
            [stress[0], stress[3], stress[5]], [stress[3], stress[1], stress[4]], [stress[5], stress[4], stress[2]]
        )
    elif ti.static(stress.n == 2):
        increment = mat3x3([stress[0, 0], stress[0, 1], 0.0], [stress[1, 0], stress[1, 1], 0.0], [0.0, 0.0, 0.0])
    else:
        increment = stress
    return increment


@ti.func
def _stress_increment_mat2(stress):
    increment = mat2x2([[0.0, 0.0], [0.0, 0.0]])
    if ti.static(stress.n == 6):
        increment = mat2x2([stress[0], stress[3]], [stress[3], stress[1]])
    elif ti.static(stress.n == 2):
        increment = stress
    else:
        increment = mat2x2([stress[0, 0], stress[0, 1]], [stress[1, 0], stress[1, 1]])
    return increment


@ti.func
def _stress_increment_pressure(stress):
    increment = _stress_increment_voigt(stress)
    return 1.0 / 3.0 * (increment[0] + increment[1] + increment[2])


@ti.func
def matrix_cols(matrix, cols):
    return ti.Vector([matrix[i, cols] for i in ti.static(range(matrix.n))])


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
