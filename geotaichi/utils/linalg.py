import numpy
import math
from copy import deepcopy


def Sphere2Certesian(vector):
    return vector[0] * numpy.array([numpy.sin(vector[1]) * numpy.cos(vector[2]), numpy.sin(vector[1]) * numpy.sin(vector[2]), numpy.cos(vector[1])])


def Certesian2Sphere(vector):
    vec = numpy.zeros(3)
    r = numpy.linalg.norm(vector)
    if r != 0.:
        theta = numpy.arccos(vector[2] / r)
        vecProj = numpy.array([vector[0], vector[1], 0])
        normProj = numpy.linalg.norm(vecProj)

        phi = 0.
        if normProj != 0.:
            cosVal = vecProj[0] / normProj
            phi = numpy.arccos(cosVal) if vec[1] > 0. else 2 * math.pi - numpy.arccos(cosVal)
        vec = numpy.array([r, theta, phi])
    return vec


def iterRead(input, function):
    if isinstance(input, dict):
       function(input)
    elif isinstance(input, list):
        for i in input:
            function(i)


def doolittle(matrix_a):
    """ Doolittle's Method for LU-factorization.

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


def ndot(array1, array2):
    return array1[0] * array2[0] - array1[1] * array2[1]


def np_normalized(array):
    return array / numpy.linalg.norm(array)


def scalar_sum(vector, scalar):
    return [i + scalar for i in vector]


def scalar_multiply(vector, scalar):
    return [i * scalar for i in vector]


def inner_sum(vector):
    result = 0.
    for i in vector:
        result += i
    return result


def vector_subtract(vector1, vector2, coeff=1.0):
    return [v1 - coeff * v2 for v1, v2 in zip(vector1, vector2)]


def inner_multiply(vector):
    result = 1.
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
    except Exception:
        raise

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
    vector_out = [(v1[1] * v2[2]) - (v1[2] * v2[1]),
                  (v1[2] * v2[0]) - (v1[0] * v2[2]),
                  (v1[0] * v2[1]) - (v1[1] * v2[0])]

    # Return the cross product of the input vectors
    return vector_out


def vector_dot(vector1, vector2):
    try:
        if vector1 is None or len(vector1) == 0 or vector2 is None or len(vector2) == 0:
            raise ValueError("Input vectors cannot be empty")
    except TypeError as e:
        print("An error occurred: {}".format(e.args[-1]))
        raise TypeError("Input must be a list or tuple")
    except Exception:
        raise

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
    except Exception:
        raise

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
    except Exception:
        raise

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
        mean_vector = [a+b for a, b in zip(mean_vector, input_vector)]
    mean_vector = [a / sz for a in mean_vector]
    return mean_vector


def vector_magnitude(vector_in):
    sq_sum = 0.0
    for vin in vector_in:
        sq_sum += vin**2
    return math.sqrt(sq_sum)


def vector_angle_between(vector1, vector2, **kwargs):
    degrees = kwargs.get('degrees', True)
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
    except Exception:
        raise

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
        raise RuntimeError("Column - row size mismatch")
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
            raise ValueError("The input must be a square matrix. " +
                             "Row " + str(idx + 1) + " has a size of " + str(len(m_a)) + ".")

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
                relist += [float(("{:." + str(decimals) + "f}").format((start + (float(x) * float(delta) / float(div)))))]

        relist += [stop]
        return relist
    return [float(("{:." + str(decimals) + "f}").format(start))]


def Interpolation(pt, Extr, knownVal):
    # Performs interpolation in a 2D space that is denoted x just for the purpose of the present function, with
    # pt the point where we want to know the value through interpolation
    # knownVal, known values at x0, x1 with eg knownVal[0] = value at x0
    # Extr = (x0,x1) 
    x0 = Extr[0]
    gx = Extr[1] - x0
    f0 = knownVal[0]
    f1 = knownVal[1] 
    return (pt[0] - x0) / gx * (f1 + f0) + f0


def biInterpolate(pt, xExtr, yExtr, knownVal):
    # Performs interpolation in a 2D space that is denoted (x,y) just for the purpose of the present function, with
    # pt the point where we want to know the value through interpolation
    # knownVal, known values at (x0,y0), (x1,y0), (x0,y1), (x1,y1) with eg knownVal[0][1] = value at (x0,y1)
    # xExtr = (x0,x1) and yExtr = (y0,y1)
    x0 = xExtr[0]
    y0 = yExtr[0]
    gx = xExtr[1] - x0
    gy = yExtr[1] - y0
    f00 = knownVal[0][0] 
    f01 = knownVal[0][1] 
    f10 = knownVal[1][0] 
    f11 = knownVal[1][1]
    bracket = (pt[1] - y0) / gy * (f11 - f10 - f01 + f00) + f10 - f00
    return (pt[0] - x0) / gx * bracket + (pt[1] - y0) / gy * (f01 - f00) + f00


def triInterpolate(pt, xExtr, yExtr, zExtr, knownVal):
    # Performs interpolation in a 3D space that is denoted (x,y,z) just for the purpose of the present function, with
    # pt the point where we want to know the value through interpolation
    # knownVal, known values at (x0,y0), (x1,y0), (x0,y1), (x1,y1) with eg knownVal[0][1] = value at (x0,y1)
    # xExtr = (x0,x1) and yExtr = (y0,y1)
    x0 = xExtr[0]
    y0 = yExtr[0]
    gx = xExtr[1] - x0
    gy = yExtr[1] - y0
    f00 = knownVal[0][0] 
    f01 = knownVal[0][1] 
    f10 = knownVal[1][0] 
    f11 = knownVal[1][1]
    bracket = (pt[1] - y0) / gy * (f11 - f10 - f01 + f00) + f10 - f00
    return (pt[0] - x0) / gx * bracket + (pt[1] - y0) / gy * (f01 - f00) + f00

