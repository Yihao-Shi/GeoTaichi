from collections import namedtuple
from src.utils.errors import SimulationStateError
from src.utils.Root import RootResults
from src.utils.Root import _ECONVERGED
from src.utils.Root import _ECONVERR
import warnings, operator
import numpy as np
import numpy, math, warnings
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP


def no_operation(*args, **kwargs):
    pass


def positive(name, value, allow_zero=False):
    """Validate and return a finite positive (optionally nonnegative) scalar."""
    value = float(value)
    valid = value >= 0.0 if allow_zero else value > 0.0
    if not math.isfinite(value) or not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


def bounding_box(points):
    if points.ndim == 1:
        return [(min(points)), (max(points))]
    else:
        assert points.ndim == 2
        if points.shape[1] == 2:
            x_coordinates, y_coordinates = zip(*points)
            return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]
        elif points.shape[2] == 3:
            x_coordinates, y_coordinates, z_coordinates = zip(*points)
            return [
                (min(x_coordinates), min(y_coordinates), min(z_coordinates)),
                (max(x_coordinates), max(y_coordinates), max(z_coordinates)),
            ]


def make_list(input):
    if isinstance(input, (int, float)):
        return [input]
    elif isinstance(input, (list, tuple, numpy.ndarray)):
        return list(input)


def align_size(x, align):
    return (x + (align - 1)) & ~(align - 1)


def ranges(nv, start=0):
    shifts = numpy.cumsum(nv)
    id_arr = numpy.ones(shifts[-1], dtype=numpy.int_)
    id_arr[shifts[:-1]] = -numpy.asarray(nv[:-1]) + 1
    id_arr[0] = start
    return id_arr.cumsum()


def flip2d(array2d, size_u=0, size_v=0):
    array2d = numpy.array(array2d)
    if size_u <= 0 or size_v <= 0:
        # Detect array shapes
        size_u = array2d.shape[1]
        size_v = array2d.shape[0]

    new_array2d = array2d.transpose()
    return new_array2d


def flip3d(array3d, size_u=0, size_v=0, size_w=0):
    array3d = numpy.array(array3d)
    if size_u <= 0 or size_v <= 0 or size_w <= 0:
        # Detect array shapes
        size_u = array3d.shape[2]
        size_v = array3d.shape[1]
        size_w = array3d.shape[0]

    new_array3d = array3d.transpose()
    return new_array3d


def flip2d_linear(array2d, size_u, size_v):
    array2d = numpy.array(array2d).reshape(size_u, size_v, 2)
    return flip2d(array2d, size_u, size_v).reshape(size_u * size_v, 2)


def flip3d_linear(array3d, size_u, size_v, size_w):
    array3d = numpy.array(array3d).reshape(size_u, size_v, size_w, 3)
    return flip3d(array3d, size_u, size_v, size_w).reshape(size_u * size_v * size_w, 3)


def right_round(num, keep_n):
    if isinstance(num, float):
        num = str(num)
    return Decimal(num).quantize((Decimal("0." + "0" * keep_n)), rounding=ROUND_HALF_UP)


def non_negative(x):
    if x <= 0:
        return 1
    else:
        return int(x)


def next_pow2(x):
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return x + 1


def round32(n):
    if n % 32 == 0:
        return n
    else:
        return ((n >> 5) + 1) << 5


def rotation_matrix_direction_2d(dir1, dir2):
    cos_theta = numpy.dot(dir1, dir2)
    sin_theta = math.pow(1 - cos_theta * cos_theta, 0.5)
    rotation_martix = numpy.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    return rotation_martix


def rotation_matrix_direction(dir1, dir2):
    cos_theta = numpy.dot(dir1, dir2)
    norm_vec = numpy.cross(dir1, dir2)
    norm_vec_invert = numpy.array(
        [[0.0, -norm_vec[2], norm_vec[1]], [norm_vec[2], 0.0, -norm_vec[0]], [-norm_vec[1], norm_vec[0], 0.0]]
    )
    rotation_martix = numpy.eye(3) + norm_vec_invert + (norm_vec_invert @ norm_vec_invert) / (1 + cos_theta)
    return rotation_martix


def transformation_matrix_direction(dir1, dir2):
    rotation_martix = rotation_matrix_direction(dir1, dir2)
    return numpy.array(
        [
            [rotation_martix[0, 0], rotation_martix[0, 1], rotation_martix[0, 2], 0],
            [rotation_martix[1, 0], rotation_martix[1, 1], rotation_martix[1, 2], 0],
            [rotation_martix[2, 0], rotation_martix[2, 1], rotation_martix[2, 2], 0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def transformation_matrix_coordinate_system(axis1, axis2):
    """
    Return a transformation matrix from axis1 to axis2
    axis1 [nxn]: the old coordinate system
    axis2 [nxn]: the new coordinate system
    """
    # reference: https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/dd277ec654440f4c2b5b07d6c286c3fd_MIT16_07F09_Lec26.pdf
    return numpy.array(
        [
            [
                numpy.dot(axis1[0, :], axis2[0, :]),
                numpy.dot(axis1[0, :], axis2[1, :]),
                numpy.dot(axis1[0, :], axis2[2, :]),
                0,
            ],
            [
                numpy.dot(axis1[1, :], axis2[0, :]),
                numpy.dot(axis1[1, :], axis2[1, :]),
                numpy.dot(axis1[1, :], axis2[2, :]),
                0,
            ],
            [
                numpy.dot(axis1[2, :], axis2[0, :]),
                numpy.dot(axis1[2, :], axis2[1, :]),
                numpy.dot(axis1[2, :], axis2[2, :]),
                0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rotation_matrix_coordinate_system(axis1, axis2):
    """
    Return a rotation matrix from axis1 to axis2
    axis1 [nxn]: the old coordinate system
    axis2 [nxn]: the new coordinate system
    """
    # reference: https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/dd277ec654440f4c2b5b07d6c286c3fd_MIT16_07F09_Lec26.pdf
    return numpy.array(
        [
            [
                numpy.dot(axis1[0, :], axis2[0, :]),
                numpy.dot(axis1[0, :], axis2[1, :]),
                numpy.dot(axis1[0, :], axis2[2, :]),
            ],
            [
                numpy.dot(axis1[1, :], axis2[0, :]),
                numpy.dot(axis1[1, :], axis2[1, :]),
                numpy.dot(axis1[1, :], axis2[2, :]),
            ],
            [
                numpy.dot(axis1[2, :], axis2[0, :]),
                numpy.dot(axis1[2, :], axis2[1, :]),
                numpy.dot(axis1[2, :], axis2[2, :]),
            ],
        ]
    )


def matrix_from_quanternion(q):
    q = q.reshape(-1, 4)
    qw, qx, qy, qz = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
    return numpy.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def heaviside_function(epsilon, phi):
    h = 0.0
    ieps = 1.0 / epsilon
    return numpy.select(
        [phi > epsilon, numpy.logical_and(phi > -epsilon, phi < epsilon)],
        [1.0, 0.5 * (1 + phi * ieps + numpy.sin(math.pi * phi * ieps) / math.pi)],
        default=0.0,
    )


def linearize(xInd, yInd, zInd, nGPx, nGPy):
    return xInd + yInd * nGPx + zInd * nGPx * nGPy


def vectorize(i, nGPx, nGPy):
    x_ind = (i % (nGPx * nGPy)) % nGPx
    y_ind = (i % (nGPx * nGPy)) // nGPx
    z_ind = i // (nGPx * nGPy)
    return x_ind, y_ind, z_ind


def reduced(var1, var2):
    return var1 * var2 / (var1 + var2)


def set_to_rotation(qs):
    qs = numpy.asarray(qs)
    qx = qs[:, 0]
    qy = qs[:, 1]
    qz = qs[:, 2]
    qw = qs[:, 3]

    R = numpy.empty((qs.shape[0], 3, 3), dtype=qs.dtype)
    R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


def sphere2_certesian(vector):
    return vector[0] * numpy.array(
        [numpy.sin(vector[1]) * numpy.cos(vector[2]), numpy.sin(vector[1]) * numpy.sin(vector[2]), numpy.cos(vector[1])]
    )


def certesian2_sphere(vector):
    vec = numpy.zeros(3)
    r = numpy.linalg.norm(vector)
    if r != 0.0:
        theta = numpy.arccos(vector[2] / r)
        vec_proj = numpy.array([vector[0], vector[1], 0])
        norm_proj = numpy.linalg.norm(vec_proj)

        phi = 0.0
        if norm_proj != 0.0:
            cos_val = vec_proj[0] / norm_proj
            phi = numpy.arccos(cos_val) if vec[1] > 0.0 else 2 * math.pi - numpy.arccos(cos_val)
        vec = numpy.array([r, theta, phi])
    return vec


def iter_read(input, function):
    if isinstance(input, dict):
        function(input)
    elif isinstance(input, list):
        for i in input:
            function(i)


def doolittle(matrix_a):
    """Doolittle's Method for LU-factorization.

    :param matrix_a: Input matrix (must be a square matrix)
    :type matrix_a: list, tuple
    :return: a tuple containing matrices (L,U)
    :rtype: tuple
    """
    # Initialize L and U matrices
    matrix_u = [[0.0 for _ in range(len(matrix_a))] for _ in range(len(matrix_a))]
    matrix_l = [[0.0 for _ in range(len(matrix_a))] for _ in range(len(matrix_a))]

    # Doolittle Method
    for i in range(0, len(matrix_a)):
        for k in range(i, len(matrix_a)):
            # Upper triangular (U) matrix
            matrix_u[i][k] = float(matrix_a[i][k] - sum([matrix_l[i][j] * matrix_u[j][k] for j in range(0, i)]))
            # Lower triangular (L) matrix
            if i == k:
                matrix_l[i][i] = 1.0
            else:
                matrix_l[k][i] = float(matrix_a[k][i] - sum([matrix_l[k][j] * matrix_u[j][i] for j in range(0, i)]))
                # Handle zero division error
                try:
                    matrix_l[k][i] /= float(matrix_u[i][i])
                except ZeroDivisionError:
                    matrix_l[k][i] = 0.0

    return matrix_l, matrix_u


def remove_connectivity_by_inactive_faces(connectivity, faceID, vertexCount, active):
    connectivity = numpy.asarray(connectivity)
    faceID = numpy.asarray(faceID, dtype=int)
    vertexCount = numpy.asarray(vertexCount, dtype=int)
    active = numpy.asarray(active, dtype=bool)
    prefix_vertex_id = numpy.zeros(len(vertexCount) + 1, dtype=int)
    prefix_vertex_id[1:] = numpy.cumsum(vertexCount)
    prefix_face_id = numpy.zeros(len(faceID) + 1, dtype=int)
    prefix_face_id[1:] = numpy.cumsum(faceID)
    keep_indices = numpy.where(active)[0]
    new_vertex_count = vertexCount[keep_indices]
    new_face_id = faceID[keep_indices]
    mapping = numpy.full(prefix_vertex_id[-1], -1, dtype=int)
    new_id = 0
    for i in range(len(vertexCount)):
        start = prefix_vertex_id[i]
        end = prefix_vertex_id[i + 1]
        if active[i]:
            mapping[start:end] = numpy.arange(new_id, new_id + (end - start))
            new_id += end - start
    keep_faces_list = [connectivity[prefix_face_id[i] : prefix_face_id[i + 1]] for i in keep_indices]
    new_connectivity = (
        numpy.vstack(keep_faces_list) if keep_faces_list else numpy.empty((0, 3), dtype=connectivity.dtype)
    )
    new_connectivity = mapping[new_connectivity]
    return new_connectivity, new_face_id, new_vertex_count


def square_norm(array):
    return sum([i * i for i in array])


def ndot(array1, array2):
    return array1[0] * array2[0] - array1[1] * array2[1]


def np_normalized(array):
    return array / numpy.linalg.norm(array)


def scalar_sum(vector, scalar):
    return [i + scalar for i in vector]


def scalar_multiply(vector, scalar):
    return [i * scalar for i in vector]


def inner_sum(vector):
    result = 0.0
    for i in vector:
        result += i
    return result


def vector_subtract(vector1, vector2, coeff=1.0):
    return [v1 - coeff * v2 for v1, v2 in zip(vector1, vector2)]


def inner_multiply(vector):
    result = 1.0
    for i in vector:
        result *= i
    return result


def vector_max(vector1, vector2):
    return [max(v1, v2) for v1, v2 in zip(vector1, vector2)]


def vector_inverse(vector):
    result = []
    for i in range(len(vector) - 1, -1, -1):
        result.append(vector[i])
    return result


def type_convert(vector, types=int):
    return [types(i) for i in vector]


def vector_cross(vector1, vector2):
    try:
        if vector1 is None or len(vector1) == 0 or vector2 is None or len(vector2) == 0:
            raise ValueError("Input vectors cannot be empty")
    except TypeError as e:
        print("An error occurred: {}".format(e.args[-1]))
        raise TypeError("Input must be a list or tuple")

    if not 1 < len(vector1) <= 3 or not 1 < len(vector2) <= 3:
        raise ValueError("The input vectors should contain 2 or 3 elements")

    # Convert 2-D to 3-D, if necessary
    if len(vector1) == 2:
        v1 = [float(v) for v in vector1] + [0.0]
    else:
        v1 = vector1

    if len(vector2) == 2:
        v2 = [float(v) for v in vector2] + [0.0]
    else:
        v2 = vector2

    # Compute cross product
    vector_out = [
        (v1[1] * v2[2]) - (v1[2] * v2[1]),
        (v1[2] * v2[0]) - (v1[0] * v2[2]),
        (v1[0] * v2[1]) - (v1[1] * v2[0]),
    ]

    # Return the cross product of the input vectors
    return vector_out


def vector_dot(vector1, vector2):
    try:
        if vector1 is None or len(vector1) == 0 or vector2 is None or len(vector2) == 0:
            raise ValueError("Input vectors cannot be empty")
    except TypeError as e:
        print("An error occurred: {}".format(e.args[-1]))
        raise TypeError("Input must be a list or tuple")

    # Compute dot product
    prod = 0.0
    for v1, v2 in zip(vector1, vector2):
        prod += v1 * v2

    # Return the dot product of the input vectors
    return prod


def vector_multiply(vector_in, scalar):
    scaled_vector = [v * scalar for v in vector_in]
    return scaled_vector


def vector_sum(vector1, vector2, coeff=1.0):
    summed_vector = [v1 + (coeff * v2) for v1, v2 in zip(vector1, vector2)]
    return summed_vector


def vector_normalize(vector_in, decimals=18):
    try:
        if vector_in is None or len(vector_in) == 0:
            raise ValueError("Input vector cannot be empty")
    except TypeError as e:
        print("An error occurred: {}".format(e.args[-1]))
        raise TypeError("Input must be a list or tuple")

    # Calculate magnitude of the vector
    magnitude = vector_magnitude(vector_in)

    # Normalize the vector
    if magnitude > 0:
        vector_out = []
        for vin in vector_in:
            vector_out.append(vin / magnitude)

        # Return the normalized vector and consider the number of significands
        return [float(("{:." + str(decimals) + "f}").format(vout)) for vout in vector_out]
    else:
        raise ValueError("The magnitude of the vector is zero")


def vector_generate(start_pt, end_pt, normalize=False):
    try:
        if start_pt is None or len(start_pt) == 0 or end_pt is None or len(end_pt) == 0:
            raise ValueError("Input points cannot be empty")
    except TypeError as e:
        print("An error occurred: {}".format(e.args[-1]))
        raise TypeError("Input must be a list or tuple")

    ret_vec = []
    for sp, ep in zip(start_pt, end_pt):
        ret_vec.append(ep - sp)

    if normalize:
        ret_vec = vector_normalize(ret_vec)
    return ret_vec


def vector_mean(*args):
    sz = len(args)
    mean_vector = [0.0 for _ in range(len(args[0]))]
    for input_vector in args:
        mean_vector = [a + b for a, b in zip(mean_vector, input_vector)]
    mean_vector = [a / sz for a in mean_vector]
    return mean_vector


def vector_magnitude(vector_in):
    sq_sum = 0.0
    for vin in vector_in:
        sq_sum += vin**2
    return math.sqrt(sq_sum)


def vector_angle_between(vector1, vector2, **kwargs):
    degrees = kwargs.get("degrees", True)
    magn1 = vector_magnitude(vector1)
    magn2 = vector_magnitude(vector2)
    acos_val = vector_dot(vector1, vector2) / (magn1 * magn2)
    angle_radians = math.acos(acos_val)
    if degrees:
        return math.degrees(angle_radians)
    else:
        return angle_radians


def vector_is_zero(vector_in, tol=10e-8):
    if not isinstance(vector_in, (list, tuple)):
        raise TypeError("Input vector must be a list or a tuple")

    res = [False for _ in range(len(vector_in))]
    for idx in range(len(vector_in)):
        if abs(vector_in[idx]) < tol:
            res[idx] = True
    return all(res)


def point_translate(point_in, vector_in):
    try:
        if point_in is None or len(point_in) == 0 or vector_in is None or len(vector_in) == 0:
            raise ValueError("Input arguments cannot be empty")
    except TypeError as e:
        print("An error occurred: {}".format(e.args[-1]))
        raise TypeError("Input must be a list or tuple")

    # Translate the point using the input vector
    point_out = [coord + comp for coord, comp in zip(point_in, vector_in)]

    return point_out


def point_distance(pt1, pt2):
    if len(pt1) != len(pt2):
        raise ValueError("The input points should have the same dimension")

    dist_vector = vector_generate(pt1, pt2, normalize=False)
    distance = vector_magnitude(dist_vector)
    return distance


def point_mid(pt1, pt2):
    if len(pt1) != len(pt2):
        raise ValueError("The input points should have the same dimension")

    dist_vector = vector_generate(pt1, pt2, normalize=False)
    half_dist_vector = vector_multiply(dist_vector, 0.5)
    return point_translate(pt1, half_dist_vector)


def matrix_identity(n):
    imat = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)]
    return imat


def matrix_pivot(m, sign=False):
    mp = deepcopy(m)
    n = len(mp)
    p = matrix_identity(n)  # permutation matrix
    num_rowswap = 0
    for j in range(0, n):
        row = j
        a_max = 0.0
        for i in range(j, n):
            a_abs = abs(mp[i][j])
            if a_abs > a_max:
                a_max = a_abs
                row = i
        if j != row:
            num_rowswap += 1
            for q in range(0, n):
                # Swap rows
                p[j][q], p[row][q] = p[row][q], p[j][q]
                mp[j][q], mp[row][q] = mp[row][q], mp[j][q]
    if sign:
        return mp, p, math.pow(-1, num_rowswap)
    return mp, p


def matrix_inverse(m):
    mp, p = matrix_pivot(m)
    m_inv = lu_solve(mp, p)
    return m_inv


def matrix_determinant(m):
    mp, p, sign = matrix_pivot(m, sign=True)
    m_l, m_u = lu_decomposition(mp)
    det = 1.0
    for i in range(len(m)):
        det *= m_l[i][i] * m_u[i][i]
    det *= sign
    return det


def matrix_transpose(m):
    num_cols = len(m)
    num_rows = len(m[0])
    m_t = []
    for i in range(num_rows):
        temp = []
        for j in range(num_cols):
            temp.append(m[j][i])
        m_t.append(temp)
    return m_t


def matrix_multiply(mat1, mat2):
    n = len(mat1)
    p1 = len(mat1[0])
    p2 = len(mat2)
    if p1 != p2:
        raise SimulationStateError("Column - row size mismatch")
    try:
        # Matrix - matrix multiplication
        m = len(mat2[0])
        mat3 = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for k in range(p2):
                    mat3[i][j] += float(mat1[i][k] * mat2[k][j])
    except TypeError:
        # Matrix - vector multiplication
        mat3 = [0.0 for _ in range(n)]
        for i in range(n):
            for k in range(p2):
                mat3[i] += float(mat1[i][k] * mat2[k])
    return mat3


def matrix_scalar(m, sc):
    mm = [[0.0 for _ in range(len(m[0]))] for _ in range(len(m))]
    for i in range(len(m)):
        for j in range(len(m[0])):
            mm[i][j] = float(m[i][j] * sc)
    return mm


def triangle_normal(tri):
    vec1 = vector_generate(tri.vertices[0].data, tri.vertices[1].data)
    vec2 = vector_generate(tri.vertices[1].data, tri.vertices[2].data)
    return vector_cross(vec1, vec2)


def triangle_center(tri, uv=False):
    if uv:
        data = [t.uv for t in tri]
        mid = [0.0, 0.0]
    else:
        data = tri.vertices
        mid = [0.0, 0.0, 0.0]
    for vert in data:
        mid = [m + v for m, v in zip(mid, vert)]
    mid = [float(m) / 3.0 for m in mid]
    return tuple(mid)


def lu_decomposition(matrix_a):
    # Check if the 2-dimensional input matrix is a square matrix
    q = len(matrix_a)
    for idx, m_a in enumerate(matrix_a):
        if len(m_a) != q:
            raise ValueError(
                "The input must be a square matrix. " + "Row " + str(idx + 1) + " has a size of " + str(len(m_a)) + "."
            )

    # Return L and U matrices
    return doolittle(matrix_a)


def forward_substitution(matrix_l, matrix_b):
    q = len(matrix_b)
    matrix_y = [0.0 for _ in range(q)]
    matrix_y[0] = float(matrix_b[0]) / float(matrix_l[0][0])
    for i in range(1, q):
        matrix_y[i] = float(matrix_b[i]) - sum([matrix_l[i][j] * matrix_y[j] for j in range(0, i)])
        matrix_y[i] /= float(matrix_l[i][i])
    return matrix_y


def backward_substitution(matrix_u, matrix_y):
    q = len(matrix_y)
    matrix_x = [0.0 for _ in range(q)]
    matrix_x[q - 1] = float(matrix_y[q - 1]) / float(matrix_u[q - 1][q - 1])
    for i in range(q - 2, -1, -1):
        matrix_x[i] = float(matrix_y[i]) - sum([matrix_u[i][j] * matrix_x[j] for j in range(i, q)])
        matrix_x[i] /= float(matrix_u[i][i])
    return matrix_x


def lu_solve(matrix_a, b):
    # Variable initialization
    dim = len(b[0])
    num_x = len(b)
    x = [[0.0 for _ in range(dim)] for _ in range(num_x)]

    # LU decomposition
    m_l, m_u = lu_decomposition(matrix_a)

    # Solve the system of linear equations
    for i in range(dim):
        bt = [b1[i] for b1 in b]
        y = forward_substitution(m_l, bt)
        xt = backward_substitution(m_u, y)
        for j in range(num_x):
            x[j][i] = xt[j]

    # Return the solution
    return x


def lu_factor(matrix_a, b):
    # Variable initialization
    dim = len(b[0])
    num_x = len(b)
    x = [[0.0 for _ in range(dim)] for _ in range(num_x)]

    # LUP decomposition
    mp, p = matrix_pivot(matrix_a)
    m_l, m_u = lu_decomposition(mp)

    # Solve the system of linear equations
    for i in range(dim):
        bt = [b1[i] for b1 in b]
        y = forward_substitution(m_l, bt)
        xt = backward_substitution(m_u, y)
        for j in range(num_x):
            x[j][i] = xt[j]

    # Return the solution
    return x


def linspace(start, stop, num, repeat_num, repeat, decimals=18):
    if (num - 2) % repeat > 0:
        raise ValueError("Input value /num/ and /repeat/ do not satisfy the equation")

    jump, per_jump = 0, 0
    if 0 < repeat_num * repeat < num - 2:
        jump = num - repeat_num * repeat - 2
        per_jump = int(jump // (repeat_num + 1))
        if jump % (repeat_num + 1) > 0:
            raise ValueError("Input value /repeat_num/ and /jump/ do not satisfy the equation")
    elif repeat_num * repeat > num - 2:
        raise ValueError("Input value /repeat_num/ and /repeat/ do not satisfy the equation")

    start = float(start)
    stop = float(stop)
    if abs(start - stop) <= 10e-8:
        return [start]
    num = int(num)
    is_jump = 0
    repeats = repeat
    if num > 1:
        div = int((num - 2 + repeat) // repeat) + math.ceil(jump / 2)
        delta = stop - start
        relist = [start]

        for x in range(1, div):
            if is_jump == per_jump:
                repeats = repeat
                is_jump = 0
            elif is_jump < per_jump:
                repeats = 1
                is_jump += 1

            for _ in range(repeats):
                relist += [
                    float(("{:." + str(decimals) + "f}").format((start + (float(x) * float(delta) / float(div)))))
                ]

        relist += [stop]
        return relist
    return [float(("{:." + str(decimals) + "f}").format(start))]


def interpolation(pt, Extr, knownVal):
    # Performs interpolation in a 2D space that is denoted x just for the purpose of the present function, with
    # pt the point where we want to know the value through interpolation
    # knownVal, known values at x0, x1 with eg knownVal[0] = value at x0
    # Extr = (x0,x1)
    x0 = Extr[:, 0]
    gx = Extr[:, 1] - x0
    f0 = knownVal[:, 0]
    f1 = knownVal[:, 1]
    return (pt[0] - x0) / gx * (f1 - f0) + f0


def bi_interpolate(pt, x_extr, y_extr, known_val):
    # Performs interpolation in a 2D space that is denoted (x,y) just for the purpose of the present function, with
    # pt the point where we want to know the value through interpolation
    # known_val, known values at (x0,y0), (x1,y0), (x0,y1), (x1,y1) with eg known_val[0][1] = value at (x0,y1)
    # x_extr = (x0,x1) and y_extr = (y0,y1)
    x0 = x_extr[:, 0]
    y0 = y_extr[:, 0]
    gx = x_extr[:, 1] - x0
    gy = y_extr[:, 1] - y0
    f00 = known_val[:, 0, 0]
    f01 = known_val[:, 0, 1]
    f10 = known_val[:, 1, 0]
    f11 = known_val[:, 1, 1]
    bracket = (pt[:, 1] - y0) / gy * (f11 - f10 - f01 + f00) + f10 - f00
    return (pt[:, 0] - x0) / gx * bracket + (pt[:, 1] - y0) / gy * (f01 - f00) + f00


def tri_interpolate(pt, xExtr, yExtr, zExtr, knownVal):
    # Performs interpolation in a 3D space that is denoted (x,y,z) just for the purpose of the present function, with
    # pt the point where we want to know the value through interpolation
    # knownVal, known values at (x0,y0), (x1,y0), (x0,y1), (x1,y1) with eg knownVal[0][1] = value at (x0,y1)
    # xExtr = (x0,x1) and yExtr = (y0,y1)
    x0 = xExtr[0]
    y0 = yExtr[0]
    z0 = zExtr[0]
    gx = xExtr[1] - x0
    gy = yExtr[1] - y0
    gz = zExtr[1] - z0
    f000 = knownVal[:, 0, 0, 0]
    f010 = knownVal[:, 0, 1, 0]
    f100 = knownVal[:, 1, 0, 0]
    f110 = knownVal[:, 1, 1, 0]
    f001 = knownVal[:, 0, 0, 1]
    f011 = knownVal[:, 0, 1, 1]
    f101 = knownVal[:, 1, 0, 1]
    f111 = knownVal[:, 1, 1, 1]
    f00 = (pt[2] - z0) / gz * (f001 - f000) + f000
    f01 = (pt[2] - z0) / gz * (f011 - f010) + f010
    f10 = (pt[2] - z0) / gz * (f101 - f100) + f100
    f11 = (pt[2] - z0) / gz * (f111 - f110) + f110
    bracket = (pt[:, 1] - y0) / gy * (f11 - f10 - f01 + f00) + f10 - f00
    return (pt[0] - x0) / gx * bracket + (pt[1] - y0) / gy * (f01 - f00) + f00


def generate_grid(x0, y0, z0, x1, y1, z1, resolution, order="z"):
    X = numpy.linspace(x0, x1, resolution + 1)
    Y = numpy.linspace(y0, y1, resolution + 1)
    Z = numpy.linspace(z0, z1, resolution + 1)
    P = cartesian_product(X, Y, Z, order=order)
    return X, Y, Z, P


def generate_grid_step(x0, y0, z0, x1, y1, z1, step, order="z"):
    X = numpy.arange(x0, x1 + step, step)
    Y = numpy.arange(y0, y1 + step, step)
    Z = numpy.arange(z0, z1 + step, step)
    P = cartesian_product(X, Y, Z, order=order)
    return X, Y, Z, P


def cartesian_product(*arrays, **order):
    la = len(arrays)
    order = order.get("order", "z")
    if order == "x":
        arrays = (arrays[2], arrays[1], arrays[0])
    elif order == "y":
        arrays = (arrays[0], arrays[2], arrays[1])
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[..., i] = a
    arr = arr.reshape(-1, la)
    if order == "x":
        arr[:, 0], arr[:, 2] = arr[:, 2].copy(), arr[:, 0].copy()
    elif order == "y":
        arr[:, 1], arr[:, 2] = arr[:, 2].copy(), arr[:, 1].copy()
    return arr


def binomial_coefficient(k, i):
    if i > k:
        return 0.0
    return math.factorial(k) / (math.factorial(k - i) * math.factorial(i))


def triangle_normal(v0, v1, v2):
    return np_normalized(numpy.cross(v1 - v0, v2 - v0))


def _results_select(full_output, r):
    """Select from a tuple of (root, funccalls, iterations, flag)"""
    x, funcalls, iterations, flag = r
    if full_output:
        results = RootResults(root=x, iterations=iterations, function_calls=funcalls, flag=flag)
        return x, results
    return x


def newton(
    func,
    x0,
    fprime=None,
    args=(),
    tol=1.48e-8,
    maxiter=50,
    fprime2=None,
    x1=None,
    rtol=0.0,
    full_output=False,
    disp=True,
):
    if tol <= 0:
        raise ValueError("tol too small (%g <= 0)" % tol)
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    if np.size(x0) > 1:
        return _array_newton(func, x0, fprime, args, tol, maxiter, fprime2, full_output)

    # Convert to float (don't use float(x0); this works also for complex x0)
    p0 = 1.0 * x0
    funcalls = 0
    if fprime is not None:
        # Newton-Raphson method
        for itr in range(maxiter):
            # first evaluate fval
            fval = func(p0, *args)
            funcalls += 1
            # If fval is 0, a root has been found, then terminate
            if fval == 0:
                return _results_select(full_output, (p0, funcalls, itr, _ECONVERGED))
            fder = fprime(p0, *args)
            funcalls += 1
            if fder == 0:
                msg = "Derivative was zero."
                if disp:
                    msg += " Failed to converge after %d iterations, value is %s." % (itr + 1, p0)
                    raise SimulationStateError(msg)
                warnings.warn(msg, RuntimeWarning)
                return _results_select(full_output, (p0, funcalls, itr + 1, _ECONVERR))
            newton_step = fval / fder
            if fprime2:
                fder2 = fprime2(p0, *args)
                funcalls += 1
                # Halley's method:
                #   newton_step /= (1.0 - 0.5 * newton_step * fder2 / fder)
                # Only do it if denominator stays close enough to 1
                # Rationale: If 1-adj < 0, then Halley sends x in the
                # opposite direction to Newton. Doesn't happen if x is close
                # enough to root.
                adj = newton_step * fder2 / fder / 2
                if np.abs(adj) < 1:
                    newton_step /= 1.0 - adj
            p = p0 - newton_step
            if np.isclose(p, p0, rtol=rtol, atol=tol):
                return _results_select(full_output, (p, funcalls, itr + 1, _ECONVERGED))
            p0 = p
    else:
        # Secant method
        if x1 is not None:
            if x1 == x0:
                raise ValueError("x1 and x0 must be different")
            p1 = x1
        else:
            eps = 1e-4
            p1 = x0 * (1 + eps)
            p1 += eps if p1 >= 0 else -eps
        q0 = func(p0, *args)
        funcalls += 1
        q1 = func(p1, *args)
        funcalls += 1
        if abs(q1) < abs(q0):
            p0, p1, q0, q1 = p1, p0, q1, q0
        for itr in range(maxiter):
            if q1 == q0:
                if p1 != p0:
                    msg = "Tolerance of %s reached." % (p1 - p0)
                    if disp:
                        msg += " Failed to converge after %d iterations, value is %s." % (itr + 1, p1)
                        raise SimulationStateError(msg)
                    warnings.warn(msg, RuntimeWarning)
                p = (p1 + p0) / 2.0
                return _results_select(full_output, (p, funcalls, itr + 1, _ECONVERGED))
            else:
                if abs(q1) > abs(q0):
                    p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
                else:
                    p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
            if np.isclose(p, p1, rtol=rtol, atol=tol):
                return _results_select(full_output, (p, funcalls, itr + 1, _ECONVERGED))
            p0, q0 = p1, q1
            p1 = p
            q1 = func(p1, *args)
            funcalls += 1

    if disp:
        msg = "Failed to converge after %d iterations, value is %s." % (itr + 1, p)
        raise SimulationStateError(msg)

    return _results_select(full_output, (p, funcalls, itr + 1, _ECONVERR))


def _array_newton(func, x0, fprime, args, tol, maxiter, fprime2, full_output):
    """
    A vectorized version of Newton, Halley, and secant methods for arrays.

    Do not use this method directly. This method is called from `newton`
    when ``np.size(x0) > 1`` is ``True``. For docstring, see `newton`.
    """
    # Explicitly copy `x0` as `p` will be modified inplace, but the
    # user's array should not be altered.
    p = np.array(x0, copy=True)

    failures = np.ones_like(p, dtype=bool)
    nz_der = np.ones_like(failures)
    if fprime is not None:
        # Newton-Raphson method
        for iteration in range(maxiter):
            # first evaluate fval
            fval = np.asarray(func(p, *args))
            # If all fval are 0, all roots have been found, then terminate
            if not fval.any():
                failures = fval.astype(bool)
                break
            fder = np.asarray(fprime(p, *args))
            nz_der = fder != 0
            # stop iterating if all derivatives are zero
            if not nz_der.any():
                break
            # Newton step
            dp = fval[nz_der] / fder[nz_der]
            if fprime2 is not None:
                fder2 = np.asarray(fprime2(p, *args))
                dp = dp / (1.0 - 0.5 * dp * fder2[nz_der] / fder[nz_der])
            # only update nonzero derivatives
            p = np.asarray(p, dtype=np.result_type(p, dp, np.float64))
            p[nz_der] -= dp
            failures[nz_der] = np.abs(dp) >= tol  # items not yet converged
            # stop iterating if there aren't any failures, not incl zero der
            if not failures[nz_der].any():
                break
    else:
        # Secant method
        dx = np.finfo(float).eps ** 0.33
        p1 = p * (1 + dx) + np.where(p >= 0, dx, -dx)
        q0 = np.asarray(func(p, *args))
        q1 = np.asarray(func(p1, *args))
        active = np.ones_like(p, dtype=bool)
        for iteration in range(maxiter):
            nz_der = q1 != q0
            # stop iterating if all derivatives are zero
            if not nz_der.any():
                p = (p1 + p) / 2.0
                break
            # Secant Step
            dp = (q1 * (p1 - p))[nz_der] / (q1 - q0)[nz_der]
            # only update nonzero derivatives
            p = np.asarray(p, dtype=np.result_type(p, p1, dp, np.float64))
            p[nz_der] = p1[nz_der] - dp
            active_zero_der = ~nz_der & active
            p[active_zero_der] = (p1 + p)[active_zero_der] / 2.0
            active &= nz_der  # don't assign zero derivatives again
            failures[nz_der] = np.abs(dp) >= tol  # not yet converged
            # stop iterating if there aren't any failures, not incl zero der
            if not failures[nz_der].any():
                break
            p1, p = p, p1
            q0 = q1
            q1 = np.asarray(func(p1, *args))

    zero_der = ~nz_der & failures  # don't include converged with zero-ders
    if zero_der.any():
        # Secant warnings
        if fprime is None:
            nonzero_dp = p1 != p
            # non-zero dp, but infinite newton step
            zero_der_nz_dp = zero_der & nonzero_dp
            if zero_der_nz_dp.any():
                rms = np.sqrt(sum((p1[zero_der_nz_dp] - p[zero_der_nz_dp]) ** 2))
                warnings.warn("RMS of {:g} reached".format(rms), RuntimeWarning)
        # Newton or Halley warnings
        else:
            all_or_some = "all" if zero_der.all() else "some"
            msg = "{:s} derivatives were zero".format(all_or_some)
            warnings.warn(msg, RuntimeWarning)
    elif failures.any():
        all_or_some = "all" if failures.all() else "some"
        msg = "{0:s} failed to converge after {1:d} iterations".format(all_or_some, maxiter)
        if failures.all():
            raise SimulationStateError(msg)
        warnings.warn(msg, RuntimeWarning)

    if full_output:
        result = namedtuple("result", ("root", "converged", "zero_der"))
        p = result(p, ~failures, zero_der)

    return p


def _quadratic_bspline(xp, xg, inv_dx, boundary_type):
    value = 0.0
    gradient = 0.0
    d = (xp - xg) * inv_dx
    if boundary_type == 0:
        if 0.5 <= d < 1.5:
            value = (0.5 * d - 1.5) * d + 1.125
            gradient = d - 1.5
        elif -0.5 <= d < 0.5:
            value = -d * d + 0.75
            gradient = -2.0 * d
        elif -1.5 <= d < 0.5:
            value = (0.5 * d + 1.5) * d + 1.125
            gradient = d + 1.5
    elif boundary_type == 1:
        if 0.0 <= d < 0.5:
            value = 1.0 - d
            gradient = -1.0
        elif 0.5 <= d < 1.5:
            value = (0.5 * d - 1.5) * d + 1.125
            gradient = d - 1.5
    elif boundary_type == 2:
        if -1.0 <= d < -0.5:
            value = 1.0 + d
            gradient = 1.0
        elif -0.5 <= d < 0.5:
            value = -d * d + 0.75
            gradient = -2.0 * d
        elif 0.5 <= d < 1.5:
            value = (0.5 * d - 1.5) * d + 1.125
            gradient = d - 1.5
    elif boundary_type == 3:
        if -1.5 <= d < -0.5:
            value = (0.5 * d + 1.5) * d + 1.125
            gradient = d + 1.5
        elif -0.5 <= d < 0.5:
            value = -d * d + 0.75
            gradient = -2.0 * d
        elif 0.5 <= d < 1.0:
            value = 1.0 - d
            gradient = -1.0
    elif boundary_type == 4:
        if -1.5 <= d < -0.5:
            value = (0.5 * d + 1.5) * d + 1.125
            gradient = d + 1.5
        elif -0.5 <= d <= 0.0:
            value = 1.0 + d
            gradient = 1.0
    return value, gradient * inv_dx


def _cubic_bspline(xp, xg, inv_dx, boundary_type):
    value = 0.0
    gradient = 0.0
    d = (xp - xg) * inv_dx
    if 1.0 <= d < 2.0:
        if boundary_type != 3:
            value = ((-d / 6.0 + 1.0) * d - 2.0) * d + 4.0 / 3.0
            gradient = (-0.5 * d + 2.0) * d - 2.0
    elif 0.0 <= d < 1.0:
        if boundary_type == 1:
            value = (d * d / 6.0 - 1.0) * d + 1.0
            gradient = 0.5 * d * d - 1.0
        elif boundary_type == 3:
            value = (d / 3.0 - 1.0) * d * d + 2.0 / 3.0
            gradient = d * (d - 2.0)
        else:
            value = (0.5 * d - 1.0) * d * d + 2.0 / 3.0
            gradient = (1.5 * d - 2.0) * d
    elif -1.0 <= d < 0.0:
        if boundary_type == 4:
            value = (-d * d / 6.0 + 1.0) * d + 1.0
            gradient = -0.5 * d * d + 1.0
        elif boundary_type == 2:
            value = (-d / 3.0 - 1.0) * d * d + 2.0 / 3.0
            gradient = (-d - 2.0) * d
        else:
            value = (-0.5 * d - 1.0) * d * d + 2.0 / 3.0
            gradient = (-1.5 * d - 2.0) * d
    elif -2.0 <= d < -1.0:
        value = ((d / 6.0 + 1.0) * d + 2.0) * d + 4.0 / 3.0
        gradient = (0.5 * d + 2.0) * d + 2.0
    return value, gradient * inv_dx


def _tetra_barycentric(point, vertices):
    jacobian = np.column_stack(
        (
            vertices[1] - vertices[0],
            vertices[2] - vertices[0],
            vertices[3] - vertices[0],
        )
    )
    inverse = np.linalg.inv(jacobian)
    tail = inverse @ (point - vertices[0])
    barycentric = np.asarray(
        (1.0 - np.sum(tail), tail[0], tail[1], tail[2]),
        dtype=np.float64,
    )
    gradients = np.empty((4, 3), dtype=np.float64)
    gradients[1:] = inverse
    gradients[0] = -np.sum(gradients[1:], axis=0)
    return barycentric, gradients


def _rotation_from_quaternion(q):
    qx, qy, qz, qw = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _euler_to_matrix(orientation):
    if orientation is None or isinstance(orientation, str):
        return np.eye(3)
    theta = np.radians(np.asarray(orientation, dtype=np.float64).reshape(3))
    cx, cy, cz = np.cos(theta)
    sx, sy, sz = np.sin(theta)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _bilinear_interpolate(a, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, a.shape[1] - 1)
    x1 = np.clip(x1, 0, a.shape[1] - 1)
    y0 = np.clip(y0, 0, a.shape[0] - 1)
    y1 = np.clip(y1, 0, a.shape[0] - 1)

    pa = a[y0, x0]
    pb = a[y1, x0]
    pc = a[y0, x1]
    pd = a[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * pa + wb * pb + wc * pc + wd * pd


def _compute_cotangent(first, second):
    sine = np.linalg.norm(np.cross(first, second))
    if sine <= 1.0e-14 * max(np.linalg.norm(first) * np.linalg.norm(second), 1.0):
        raise ValueError("cloth bending element contains a degenerate reference triangle")
    return float(np.dot(first, second) / sine)


def _signed_dihedral_angle(points):
    x0, x1, x2, x3 = points
    normal0 = np.cross(x1 - x0, x2 - x0)
    normal1 = np.cross(x2 - x3, x1 - x3)
    denominator = np.linalg.norm(normal0) * np.linalg.norm(normal1)
    if denominator <= 1.0e-28:
        raise ValueError("cloth dihedral bending contains a degenerate hinge")
    cosine = np.clip(np.dot(normal0, normal1) / denominator, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if np.dot(np.cross(normal1, normal0), x1 - x2) < 0.0:
        angle = -angle
    return angle


def normalized(arr):
    arr_norm = np.linalg.norm(arr)
    if arr_norm != 0:
        return arr / np.linalg.norm(arr)
    else:
        return arr


def evaluate_bounding_box(ctrlpts):
    dimension = len(ctrlpts[0])
    bbmin = [float("inf") for _ in range(0, dimension)]
    bbmax = [float("-inf") for _ in range(0, dimension)]
    for cpt in ctrlpts:
        for i, arr in enumerate(zip(cpt, bbmin)):
            if arr[0] < arr[1]:
                bbmin[i] = arr[0]
        for i, arr in enumerate(zip(cpt, bbmax)):
            if arr[0] > arr[1]:
                bbmax[i] = arr[0]
    return np.asarray(bbmin), np.asarray(bbmax)


def check_same_side(point1, point2, point):
    vec1 = point2 - point
    vec2 = point1 - point
    return np.dot(vec1, vec2) >= 0


def solve_using_cod(A, b, rcond=1e-12):
    Q, R = np.linalg.qr(A, mode="reduced")
    U, S, VT = np.linalg.svd(R, full_matrices=False)
    tol = rcond * S.max()
    S_inv = np.diag(np.where(S > tol, 1.0 / S, 0.0))
    R_inv = VT.T @ S_inv @ U.T
    rhs = Q.T @ b
    x = R_inv @ rhs
    return x


def project_point_to_plane(points):
    points = np.asarray(points)
    relative_positions = points - np.mean(points, axis=0)
    cov_matrix = np.cov(relative_positions, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues /= points.shape[0]
    x_coords = np.dot(relative_positions, eigenvectors[:, 0])
    y_coords = np.dot(relative_positions, eigenvectors[:, 1])
    return np.column_stack((x_coords, y_coords))


def point_to_edges(point, vertice1, vertice2, norm="L2"):
    pa = point - vertice1
    ba = vertice2 - vertice1
    h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
    if norm == "L1":
        return np.sum(np.abs(pa - ba * h))
    elif norm == "L2":
        return np.linalg.norm(pa - ba * h)
    else:
        raise ValueError("please choose L1 or L2 norm")


def point_to_point(point1, point2, axis=None, norm="L2"):
    if norm == "L1":
        return np.sum(np.abs(point1 - point2), axis=axis)
    elif norm == "L2":
        return np.linalg.norm(point1 - point2, axis=axis)
    else:
        raise ValueError("please choose L1 or L2 norm")


def get_determinant_3x2(matrix):
    U, sigma, V = np.linalg.svd(matrix)
    return sigma[0] * sigma[1]


def project_evd(matrix):
    eigenvalues, Q = np.linalg.eigh(matrix)
    eigenvalues_proj = np.maximum(eigenvalues, 0)
    A_proj = Q @ np.diag(eigenvalues_proj) @ Q.T
    return A_proj


def quadratic_root(a, b, c):
    root = np.array([np.inf, np.inf])
    if b * b - 4.0 * a * c > 0.0:
        root[0] = 0.5 * (-b + np.sqrt(b * b - 4.0 * a * c)) / a
        root[1] = 0.5 * (-b - np.sqrt(b * b - 4.0 * a * c)) / a
    return root


def quadrilateral_to_triangle_connectivity(connectivity):
    window_size = 3
    shape = connectivity.shape[:-1] + (connectivity.shape[-1] - window_size + 1, window_size)
    strides = connectivity.strides + (connectivity.strides[-1],)
    return np.lib.stride_tricks.as_strided(connectivity, shape=shape, strides=strides)


def insert_midpoints(points):
    points = np.asarray(points)
    midpoints = 0.5 * (points[:-1] + points[1:])
    new_points = np.empty((len(points) + len(midpoints), 2))
    new_points[0::2] = points
    new_points[1::2] = midpoints
    return new_points


def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + 0.5 * (x[-1] * y[0] - x[0] * y[-1])


def tsp_greedy(points, initial_index=0):
    n = len(points)
    if n == 0:
        return points
    visited = np.zeros(n, dtype=bool)
    path = [initial_index]
    visited[initial_index] = True
    current_index = initial_index
    for _ in range(n - 1):
        remaining_indices = np.where(~visited)[0]
        distances = np.linalg.norm(points[remaining_indices] - points[current_index], axis=1)
        next_index = remaining_indices[np.argmin(distances)]
        path.append(next_index)
        visited[next_index] = True
        current_index = next_index
    return np.array(path)


def sort_points_counterclockwise(points, initial_index=0, method=0):
    if method == 0:
        order = tsp_greedy(points, initial_index)
        sorted_points = points[order]
        area = polygon_area(sorted_points)
        if area < 0:
            # 保持起始点不变，反转其余点的顺序
            sorted_points = np.concatenate(([sorted_points[0]], sorted_points[1:][::-1]))
        return sorted_points
    elif method == 1:
        # 找到重心
        centroid = np.mean(points, axis=0)

        # 计算每个点相对于重心的角度
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

        # 按角度排序
        sorted_indices = np.argsort(angles)
        return points[sorted_indices]


def distance_aabb(bounding_box1, bounding_box2):
    lower = np.maximum(bounding_box1["box_min"], bounding_box2["box_min"])
    upper = np.minimum(bounding_box1["box_max"], bounding_box2["box_max"])
    delta = np.maximum(0.0, lower - upper)
    return np.linalg.norm(delta)


def pca_analysis(points):
    points = np.asarray(points)
    points_center = np.mean(points, axis=0)
    centered_points = points - points_center
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues /= points.shape[0]
    return points_center, eigenvectors, eigenvalues


def triangle_area(vertices):
    """Return the area of one 3D triangle."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape != (3, 3):
        raise ValueError("triangle_area expects an array with shape (3, 3)")
    return 0.5 * float(np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])))


def segment_length(vertices):
    """Return the length of one 2D or 3D boundary segment."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] != 2 or vertices.shape[1] not in (2, 3):
        raise ValueError("segment_length expects shape (2, 2) or (2, 3)")
    return float(np.linalg.norm(vertices[1] - vertices[0]))


def lumped_segment_vertex_measures(vertices, edges):
    """Lump half of every boundary-segment length to each endpoint."""
    vertices = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int32).reshape((-1, 2))
    measures = np.zeros(vertices.shape[0], dtype=np.float64)
    for edge in edges:
        measures[edge] += 0.5 * segment_length(vertices[edge])
    return measures


def lumped_boundary_vertex_measures(vertices, elements):
    """Dimension-independent FEM boundary vertex quadrature.

    Two-node elements are segments; three-node elements are surface triangles.
    """
    elements = np.asarray(elements, dtype=np.int32)
    if elements.ndim != 2 or elements.shape[1] not in (2, 3):
        raise ValueError("boundary elements must have shape (n, 2) or (n, 3)")
    if elements.shape[1] == 2:
        return lumped_segment_vertex_measures(vertices, elements)
    return lumped_vertex_measures(vertices, elements)


def lumped_vertex_measures(vertices, faces):
    """Barycentrically lump 3D triangle areas onto their three vertices."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("lumped_vertex_measures expects vertices with shape (n, 3)")
    faces = np.asarray(faces, dtype=np.int32).reshape((-1, 3))
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    measures = np.zeros(vertices.shape[0], dtype=np.float64)
    np.add.at(measures, faces.reshape(-1), np.repeat(areas / 3.0, 3))
    return measures


def surface_edges_and_measures(vertices, faces):
    """Build unique undirected edges and their incident lumped face measure.

    Edge ordering follows first appearance in ``faces`` so existing engine
    candidate identifiers remain stable across the refactor.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32).reshape((-1, 3))
    edge_ids = {}
    edge_measures = []
    for face in faces:
        face_measure = triangle_area(vertices[face])
        for local_a, local_b in ((0, 1), (1, 2), (2, 0)):
            a = int(face[local_a])
            b = int(face[local_b])
            key = (min(a, b), max(a, b))
            if key not in edge_ids:
                edge_ids[key] = len(edge_ids)
                edge_measures.append(0.0)
            edge_measures[edge_ids[key]] += face_measure / 3.0
    edges = np.asarray(list(edge_ids.keys()), dtype=np.int32).reshape((-1, 2))
    return edges, np.asarray(edge_measures, dtype=np.float64)


def surface_vertex_edge_measures(vertices, faces):
    """Return ``(vertex_measure, edges, edge_measure)`` for a triangle mesh."""
    return (
        lumped_vertex_measures(vertices, faces),
        *surface_edges_and_measures(vertices, faces),
    )
