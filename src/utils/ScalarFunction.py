import taichi as ti

from src.utils.constants import THRESHOLD


@ti.func
def equal_to(i, value):
    return ti.abs(i - value) < THRESHOLD


@ti.func
def linearize(index, vector):
    assert index.n == vector.n
    if ti.static(vector.n == 2):
        return int(index[0] + index[1] * vector[0])
    elif ti.static(vector.n == 3):
        return int(index[0] + index[1] * vector[0] + index[2] * vector[0] * vector[1])


@ti.func
def linearize2d(i, j, vector):
    return int(i + j * vector[0])


@ti.func
def linearize3d(i, j, k, vector):
    return int(i + j * vector[0] + k * vector[0] * vector[1])


@ti.func
def vectorize_id(index, count_vec):
    if ti.static(count_vec.n == 2):
        ig = index % count_vec[0]
        jg = index // count_vec[0]
        return ig, jg
    elif ti.static(count_vec.n == 3):
        ig = (index % (count_vec[0] * count_vec[1])) % count_vec[0]
        jg = (index % (count_vec[0] * count_vec[1])) // count_vec[0]
        kg = index // (count_vec[0] * count_vec[1])
        return ig, jg, kg


@ti.func
def isnan(scalar):
    return not (scalar >= 0 or scalar <= 0)


@ti.func
def isinf(scalar):
    return 2 * scalar == scalar and scalar != 0


@ti.func
def swap(a, b):
    return b, a


@ti.func
def copysign(x, y):
    return ti.abs(x) * sgn(y)


@ti.pyfunc
def sgn(x):
    return ti.cast((x >= 0.0), float) - ti.cast((x <= 0.0), float)


@ti.func
def maximum(i, j):
    m = j
    if i > j:
        m = i
    return m


@ti.func
def minimum(i, j):
    m = j
    if i < j:
        m = i
    return m


@ti.func
def effective_value(x, y):
    return x * y / (x + y)


@ti.func
def xor(a, b):
    return (a + b) & 1


@ti.pyfunc
def pairing_function(i, j):
    a = ti.min(i, j)
    b = ti.max(i, j)
    return int(0.5 * (a + b) * (a + b + 1) + b)


@ti.pyfunc
def de_pairing_function(pair_id):
    w = int(0.5 * (ti.sqrt(8 * pair_id + 1) - 1))
    t = 0.5 * w * (w + 1)
    j = ti.cast(pair_id - t, ti.i32)
    i = ti.cast(w - j, ti.i32)
    return i, j


@ti.pyfunc
def encode_pair(i, j):
    a = ti.cast(ti.min(i, j), ti.i64)
    b = ti.cast(ti.max(i, j), ti.i64)
    return (a << 32) | b


@ti.pyfunc
def decode_pair(pair_id):
    i = ti.cast(pair_id >> 32, ti.i32)
    j = ti.cast(pair_id & 0xFFFFFFFF, ti.i32)
    return i, j


@ti.func
def pairing_mapping(i, j, length):
    return int(i * length + j)


@ti.func
def clamp(min_val, max_val, val):
    return ti.min(ti.max(min_val, val), max_val)


@ti.func
def macauley(value):
    return value if value > 0.0 else 0.0


@ti.func
def macauley_index(value):
    return 1.0 if value > 0.0 else 0.0


@ti.func
def binary_search(begining, ending, key, KEY):
    loc = -1
    while begining <= ending:
        mid_point = int((begining + ending) / 2)
        if KEY[mid_point] == key:
            loc = mid_point
            break
        elif KEY[mid_point] > key:
            ending = mid_point - 1
        elif KEY[mid_point] < key:
            begining = mid_point + 1
    return loc


@ti.func
def bi_interpolate(pt, x_extr, y_extr, known_val):
    x0 = x_extr[0]
    y0 = y_extr[0]
    gx = x_extr[1] - x0
    gy = y_extr[1] - y0
    f00 = known_val[0, 0]
    f01 = known_val[0, 1]
    f10 = known_val[1, 0]
    f11 = known_val[1, 1]
    bracket = (pt[1] - y0) / gy * (f11 - f10 - f01 + f00) + f10 - f00
    return (pt[0] - x0) / gx * bracket + (pt[1] - y0) / gy * (f01 - f00) + f00


@ti.func
def _isfinite_scalar(value):
    return (value == value) and ti.abs(value) < 1.0e300


@ti.func
def get_smallest_positive_real_quad_root(a, b, c, tol=1e-12):
    t = -1.0
    if ti.abs(a) <= tol:
        if ti.abs(b) > tol:
            t = -c / b
    else:
        desc = b * b - 4.0 * a * c
        if desc > 0:
            t1 = (-b - ti.sqrt(desc)) / (2.0 * a)
            t2 = (-b + ti.sqrt(desc)) / (2.0 * a)
            if t1 > 0 and (t < 0 or t1 < t):
                t = t1
            if t2 > 0 and (t < 0 or t2 < t):
                t = t2
    return t


@ti.func
def complex_div_num(num_re, num_im, den_re, den_im):
    den_norm = den_re * den_re + den_im * den_im
    return ((num_re * den_re + num_im * den_im) / den_norm, (num_im * den_re - num_re * den_im) / den_norm)


@ti.func
def get_smallest_positive_real_cubic_root(a, b, c, d, tol=1e-12):
    t = -1.0
    if ti.abs(a) <= tol:
        t = get_smallest_positive_real_quad_root(b, c, d, tol)
    else:
        delta0 = b * b - 3.0 * a * c
        delta1 = 2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d

        discriminant = delta1 * delta1 - 4.0 * delta0 * delta0 * delta0

        # ---- compute C ----
        C = ti.pow((delta1 + ti.sqrt(discriminant)) / 2.0, 1.0 / 3.0)
        if ti.abs(C) == 0:
            C = ti.pow((delta1 - ti.sqrt(discriminant)) / 2.0, 1.0 / 3.0)

        # cube roots of unity
        u2_re = -0.5
        u2_im = 0.5 * ti.sqrt(3.0)
        u3_re = -0.5
        u3_im = -0.5 * ti.sqrt(3.0)

        # because C and delta0 are real
        t1 = (b + C + delta0 / C) / (-3.0 * a)

        # t2
        C2_re = u2_re * C
        C2_im = u2_im * C
        num2_re = delta0
        num2_im = 0.0
        den2_re = C2_re
        den2_im = C2_im
        frac2_re, frac2_im = complex_div_num(num2_re, num2_im, den2_re, den2_im)
        t2_re = (b + C2_re + frac2_re) / (-3.0 * a)
        t2_im = (C2_im + frac2_im) / (-3.0 * a)

        # t3
        C3_re = u3_re * C
        C3_im = u3_im * C
        num3_re = delta0
        num3_im = 0.0
        den3_re = C3_re
        den3_im = C3_im
        frac3_re, frac3_im = complex_div_num(num3_re, num3_im, den3_re, den3_im)
        t3_re = (b + C3_re + frac3_re) / (-3.0 * a)
        t3_im = (C3_im + frac3_im) / (-3.0 * a)

        # choose positive nearly-real roots
        if t1 > 0 and (t < 0 or t1 < t):
            t = t1

        if ti.abs(t2_im) < tol:
            if t2_re > 0 and (t < 0 or t2_re < t):
                t = t2_re

        if ti.abs(t3_im) < tol:
            if t3_re > 0 and (t < 0 or t3_re < t):
                t = t3_re
    return t
