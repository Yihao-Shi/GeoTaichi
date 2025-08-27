import taichi as ti


# =================================== Hyperelastic constitutive model =================================== #
# refer to: Dynamic deformables: implementation and production particalities (now with code!). ACM SIGGRAPH 2022 Courses.
@ti.func
def getI1(td):
    sums = 0.
    for i in ti.static(range(td.n)):
        for j in ti.static(range(td.m)):
            sums += td[i, j] * td[i, j]
    return sums

@ti.func
def getI2(td):
    I_1 = getI1(td)
    sums = 0.
    for i in ti.static(range(td.n)):
        rowsum = 0.
        for j in ti.static(range(td.m)):
            rowsum += td[i, j] * td[i, j]
        sums += rowsum * rowsum
    return 0.5 * (I_1 ** 2 - sums)

@ti.func
def getI3(td):
    return td.determinant() ** 2

@ti.func
def getI1dev(AJ, td):
    return AJ ** (-2./3.) * getI1(td)

@ti.func
def getI2dev(AJ, td):
    return AJ ** (-4./3.) * getI2(td)

@ti.func
def getI3dev(AJ, td=None):
    return AJ

@ti.func
def compute_dI1dF(td):
    return 2. * td

@ti.func
def compute_d2I1_dF2(td):
    # flatten to column first
    if ti.static(td.n == 2):
        return 2. * ti.Matrix([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], float)
    elif ti.static(td.n == 3):
        return 2. * ti.Matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1]], float)

@ti.func
def compute_dI2dF(td):
    I_1 = getI1(td)
    return 2. * (td * I_1 - td @ td @ td.transpose())

@ti.func
def compute_d2I2dF2(td):
    I_1 = getI1(td)
    d2I1_dF2 = compute_d2I1_dF2(td)
    pass

@ti.func
def compute_dJdF(td):
    if ti.static(td.n == 2):
        F00, F01, F10, F11 = td[0, 0], td[0, 1], td[1, 0], td[1, 1]
        return ti.Matrix([[F11, -F10],
                          [-F01, F00]], float)
    elif ti.static(td.n == 3):
        F00, F01, F02, F10, F11, F12, F20, F21, F22 = td[0, 0], td[0, 1], td[0, 2], td[1, 0], td[1, 1], td[1, 2], td[2, 0], td[2, 1], td[2, 2]
        return ti.Matrix([[F11 * F22 - F12 * F21, -F10 * F22 + F12 * F20, F10 * F21 - F11 * F20],
                          [-F01 * F22 + F02 * F21, F00 * F22 - F02 * F20, -F00 * F21 + F01 * F20],
                          [F01 * F12 - F02 * F11, -F00 * F12 + F02 * F10, F00 * F11 - F01 * F10]], float)

@ti.func
def compute_d2J_dF2(td):
    if ti.static(td.n == 2):
        return ti.Matrix([[0., 0., 0., 1.],
                          [0., 0., -1., 0.],
                          [0., -1., 0., 0.],
                          [1., 0., 0., 0.]], float)
    elif ti.static(td.n == 3):
        F00, F01, F02, F10, F11, F12, F20, F21, F22 = td[0, 0], td[0, 1], td[0, 2], td[1, 0], td[1, 1], td[1, 2], td[2, 0], td[2, 1], td[2, 2]
        return ti.Matrix([[0, 0, 0, 0, F22, -F12, 0, -F21, F11],
                        [0, 0, 0, -F22, 0, F02, F21, 0, -F01],
                        [0, 0, 0, F12, -F02, 0, -F11, F01, 0],
                        [0, -F22, F12, 0, 0, 0, 0, F20, -F10],
                        [F22, 0, -F02, 0, 0, 0, -F20, 0, F00],
                        [-F12, F02, 0, 0, 0, 0, F10, -F00, 0],
                        [0, F21, -F11, 0, -F20, F10, 0, 0, 0],
                        [-F21, 0, F01, F20, 0, -F00, 0, 0, 0],
                        [F11, -F01, 0, -F10, F00, 0, 0, 0, 0]], float)

@ti.func
def getPK1(dUdI1, dUdI2, dUdJ, td):
    return dUdI1 * compute_dI1dF(td) + dUdI2 * compute_dI2dF(td) + dUdJ * compute_dJdF(td)