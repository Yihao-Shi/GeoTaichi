import taichi as ti


# =================================== Hyperelastic constitutive model =================================== #
# refer to: Dynamic deformables: implementation and production particalities (now with code!). ACM SIGGRAPH 2022 Courses.
@ti.func
def get_i1(td):
    sums = 0.0
    for i in ti.static(range(td.n)):
        for j in ti.static(range(td.m)):
            sums += td[i, j] * td[i, j]
    return sums


@ti.func
def get_i2(td):
    I_1 = get_i1(td)
    right_cauchy_green = td.transpose() @ td
    trace_c_squared = 0.0
    for i in ti.static(range(right_cauchy_green.n)):
        for j in ti.static(range(right_cauchy_green.m)):
            trace_c_squared += right_cauchy_green[i, j] * right_cauchy_green[i, j]
    return 0.5 * (I_1**2 - trace_c_squared)


@ti.func
def get_i3(td):
    return td.determinant() ** 2


@ti.func
def get_i1dev(AJ, td):
    return AJ ** (-2.0 / 3.0) * get_i1(td)


@ti.func
def get_i2dev(AJ, td):
    return AJ ** (-4.0 / 3.0) * get_i2(td)


@ti.func
def get_i3dev(AJ, td=None):
    return AJ


@ti.func
def compute_d_i1d_f(td):
    return 2.0 * td


@ti.func
def compute_d2_i1_d_f2(td):
    # flatten to column first
    if ti.static(td.n == 2):
        return 2.0 * ti.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    elif ti.static(td.n == 3):
        return 2.0 * ti.Matrix(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
            float,
        )


@ti.func
def compute_d_i2d_f(td):
    I_1 = get_i1(td)
    return 2.0 * (td * I_1 - td @ td.transpose() @ td)


@ti.func
def compute_d2_i2d_f2(td):
    I_1 = get_i1(td)
    right_cauchy_green = td.transpose() @ td
    left_cauchy_green = td @ td.transpose()
    size = ti.static(td.n * td.m)
    hessian = ti.Matrix.zero(float, size, size)
    hessian_entry = 0
    while hessian_entry < size * size:
        row = hessian_entry // size
        column = hessian_entry - row * size
        a = row // td.n
        i = row - a * td.n
        b = column // td.n
        j = column - b * td.n
        value = 4.0 * td[i, a] * td[j, b]
        if i == j and a == b:
            value += 2.0 * I_1
        if i == j:
            value -= 2.0 * right_cauchy_green[b, a]
        value -= 2.0 * td[i, b] * td[j, a]
        if a == b:
            value -= 2.0 * left_cauchy_green[i, j]
        hessian[row, column] = value
        hessian_entry += 1
    return hessian


@ti.func
def compute_d_jd_f(td):
    if ti.static(td.n == 2):
        F00, F01, F10, F11 = td[0, 0], td[0, 1], td[1, 0], td[1, 1]
        return ti.Matrix([[F11, -F10], [-F01, F00]], float)
    elif ti.static(td.n == 3):
        F00, F01, F02, F10, F11, F12, F20, F21, F22 = (
            td[0, 0],
            td[0, 1],
            td[0, 2],
            td[1, 0],
            td[1, 1],
            td[1, 2],
            td[2, 0],
            td[2, 1],
            td[2, 2],
        )
        return ti.Matrix(
            [
                [F11 * F22 - F12 * F21, -F10 * F22 + F12 * F20, F10 * F21 - F11 * F20],
                [-F01 * F22 + F02 * F21, F00 * F22 - F02 * F20, -F00 * F21 + F01 * F20],
                [F01 * F12 - F02 * F11, -F00 * F12 + F02 * F10, F00 * F11 - F01 * F10],
            ],
            float,
        )


@ti.func
def compute_d2_j_d_f2(td):
    if ti.static(td.n == 2):
        return ti.Matrix(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], float
        )
    elif ti.static(td.n == 3):
        F00, F01, F02, F10, F11, F12, F20, F21, F22 = (
            td[0, 0],
            td[0, 1],
            td[0, 2],
            td[1, 0],
            td[1, 1],
            td[1, 2],
            td[2, 0],
            td[2, 1],
            td[2, 2],
        )
        return ti.Matrix(
            [
                [0, 0, 0, 0, F22, -F12, 0, -F21, F11],
                [0, 0, 0, -F22, 0, F02, F21, 0, -F01],
                [0, 0, 0, F12, -F02, 0, -F11, F01, 0],
                [0, -F22, F12, 0, 0, 0, 0, F20, -F10],
                [F22, 0, -F02, 0, 0, 0, -F20, 0, F00],
                [-F12, F02, 0, 0, 0, 0, F10, -F00, 0],
                [0, F21, -F11, 0, -F20, F10, 0, 0, 0],
                [-F21, 0, F01, F20, 0, -F00, 0, 0, 0],
                [F11, -F01, 0, -F10, F00, 0, 0, 0, 0],
            ],
            float,
        )


@ti.func
def get_pk1(dUdI1, dUdI2, dUdJ, td):
    return dUdI1 * compute_d_i1d_f(td) + dUdI2 * compute_d_i2d_f(td) + dUdJ * compute_d_jd_f(td)


@ti.func
def get_invariant_hessian(
    td,
    dUdI1,
    dUdI2,
    dUdJ,
    d2UdI1I1,
    d2UdI1I2,
    d2UdI1J,
    d2UdI2I2,
    d2UdI2J,
    d2UdJJ,
):
    """Exact ``dP/dF`` for an energy ``W(I1(F), I2(F), J(F))``.

    Rows and columns use GeoTaichi's column-major flattening convention,
    ``component(i, a) = a * dimension + i``.  The expression is the full
    second-order chain rule; no differencing or Hessian symmetrization is
    performed here.
    """
    gradient_i1 = compute_d_i1d_f(td)
    gradient_i2 = compute_d_i2d_f(td)
    gradient_j = compute_d_jd_f(td)
    invariant_i1 = get_i1(td)
    right_cauchy_green = td.transpose() @ td
    left_cauchy_green = td @ td.transpose()
    determinant = td.determinant()
    inverse_transpose = td.inverse().transpose()
    size = ti.static(td.n * td.m)
    tangent = ti.Matrix.zero(float, size, size)
    tangent_entry = 0
    while tangent_entry < size * size:
        row = tangent_entry // size
        column = tangent_entry - row * size
        a = row // td.n
        i = row - a * td.n
        b = column // td.n
        j = column - b * td.n
        i1_row = gradient_i1[i, a]
        i2_row = gradient_i2[i, a]
        j_row = gradient_j[i, a]
        i1_column = gradient_i1[j, b]
        i2_column = gradient_i2[j, b]
        j_column = gradient_j[j, b]
        hessian_i1 = 0.0
        if i == j and a == b:
            hessian_i1 = 2.0
        hessian_i2 = 4.0 * td[i, a] * td[j, b]
        if i == j and a == b:
            hessian_i2 += 2.0 * invariant_i1
        if i == j:
            hessian_i2 -= 2.0 * right_cauchy_green[b, a]
        hessian_i2 -= 2.0 * td[i, b] * td[j, a]
        if a == b:
            hessian_i2 -= 2.0 * left_cauchy_green[i, j]
        hessian_j = determinant * (
            inverse_transpose[i, a] * inverse_transpose[j, b] - inverse_transpose[i, b] * inverse_transpose[j, a]
        )
        tangent[row, column] = (
            dUdI1 * hessian_i1
            + dUdI2 * hessian_i2
            + dUdJ * hessian_j
            + d2UdI1I1 * i1_row * i1_column
            + d2UdI1I2 * (i1_row * i2_column + i2_row * i1_column)
            + d2UdI1J * (i1_row * j_column + j_row * i1_column)
            + d2UdI2I2 * i2_row * i2_column
            + d2UdI2J * (i2_row * j_column + j_row * i2_column)
            + d2UdJJ * j_row * j_column
        )
        tangent_entry += 1
    return tangent


@ti.func
def f3d(td):
    if ti.static(td.n == 3):
        return td
    else:
        return ti.Matrix([[td[0, 0], td[0, 1], 0], [td[1, 0], td[1, 1], 0], [0, 0, 1]])
