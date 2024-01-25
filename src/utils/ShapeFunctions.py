import taichi as ti

from src.utils.constants import ZEROVEC8f, ZEROMAT8x3, PI, SQRT_PI, Threshold, iPI
from src.utils.ScalarFunction import sign


# ========================================================= #
#                  GIMP shape function                      #
# ========================================================= #
@ti.func
def ShapeGIMP(xp, xg, idx, lp):
    nx = 0.
    dx = 1. / idx
    d = xp - xg
    a = dx + lp
    b = dx * lp
    if d < a:
        if d > (dx - lp):
            nx = (a - d) * (a - d) / (4. * b)
        elif d > lp:
            nx = 1. - d * idx
        elif d > -lp:
            nx = 1. - (d * d + lp * lp) / (2. * b)
        elif d > (-dx + lp):
            nx = 1. + d * idx
        elif d > -a:
            nx = (a + d) * (a + d) / (4. * b)
    return nx


@ti.func
def GShapeGIMP(xp, xg, idx, lp):
    dnx = 0.
    dx = 1. / idx
    d = xp - xg
    a = dx + lp
    b = dx * lp
    if d < a:
        if d > (dx - lp):
            dnx = (d - a) / (2. * b)
        elif d > lp:
            dnx = -1. * idx
        elif d > -lp:
            dnx = -d / b
        elif d > (-dx + lp):
            dnx = 1. * idx
        elif d > -a:
            dnx = (a + d) / (2. * b)
    return dnx


@ti.func
def ShapeGIMPCenter(xp, xg, idx, lp):
    nxc = 0.
    dx = 1. / idx
    d = xp - xg
    a = dx + lp
    b = dx * lp
    if d < a:
        if d > (dx - lp):
            nxc = (a - d) / (4. * lp)
        elif d > (-dx + lp):
            nxc = 1./2.
        elif d > -a:
            nxc = (a + d) / (4. * lp)
    return nxc


# ========================================================= #
#                  Linear shape function                    #
# ========================================================= #
@ti.func
def local_linear_shapefn(natural_position):
    shapefn = ZEROVEC8f
    shapefn[0] = 0.125 * (1 - natural_position[0]) * (1 - natural_position[1]) * (1 - natural_position[2])
    shapefn[1] = 0.125 * (1 + natural_position[0]) * (1 - natural_position[1]) * (1 - natural_position[2])
    shapefn[2] = 0.125 * (1 - natural_position[0]) * (1 + natural_position[1]) * (1 - natural_position[2])
    shapefn[3] = 0.125 * (1 + natural_position[0]) * (1 + natural_position[1]) * (1 - natural_position[2])
    shapefn[4] = 0.125 * (1 - natural_position[0]) * (1 - natural_position[1]) * (1 + natural_position[2])
    shapefn[5] = 0.125 * (1 + natural_position[0]) * (1 - natural_position[1]) * (1 + natural_position[2])
    shapefn[6] = 0.125 * (1 - natural_position[0]) * (1 + natural_position[1]) * (1 + natural_position[2])
    shapefn[7] = 0.125 * (1 + natural_position[0]) * (1 + natural_position[1]) * (1 + natural_position[2])
    return shapefn
    

@ti.func
def local_linear_dshapefn(natural_position):
    dshapefn = ZEROMAT8x3
    dshapefn[0, 0] = -0.125 * (1 - natural_position[1]) * (1 - natural_position[2])
    dshapefn[0, 1] = -0.125 * (1 - natural_position[0]) * (1 - natural_position[2])
    dshapefn[0, 2] = -0.125 * (1 - natural_position[0]) * (1 - natural_position[1])

    dshapefn[1, 0] = -0.125 * (1 - natural_position[1]) * (1 - natural_position[2])
    dshapefn[1, 1] = -0.125 * (1 + natural_position[0]) * (1 - natural_position[2])
    dshapefn[1, 2] = -0.125 * (1 + natural_position[0]) * (1 - natural_position[1])

    dshapefn[2, 0] = -0.125 * (1 + natural_position[1]) * (1 - natural_position[2])
    dshapefn[2, 1] = -0.125 * (1 - natural_position[0]) * (1 - natural_position[2])
    dshapefn[2, 2] = -0.125 * (1 - natural_position[0]) * (1 + natural_position[1])

    dshapefn[3, 0] = -0.125 * (1 + natural_position[1]) * (1 - natural_position[2])
    dshapefn[3, 1] = -0.125 * (1 + natural_position[0]) * (1 - natural_position[2])
    dshapefn[3, 2] = -0.125 * (1 + natural_position[0]) * (1 + natural_position[1])

    dshapefn[4, 0] = -0.125 * (1 - natural_position[1]) * (1 + natural_position[2])
    dshapefn[4, 1] = -0.125 * (1 - natural_position[0]) * (1 + natural_position[2])
    dshapefn[4, 2] = -0.125 * (1 - natural_position[0]) * (1 - natural_position[1])

    dshapefn[5, 0] = -0.125 * (1 + natural_position[1]) * (1 + natural_position[2])
    dshapefn[5, 1] = -0.125 * (1 + natural_position[0]) * (1 + natural_position[2])
    dshapefn[5, 2] = -0.125 * (1 + natural_position[0]) * (1 - natural_position[1])

    dshapefn[6, 0] = -0.125 * (1 + natural_position[1]) * (1 + natural_position[2])
    dshapefn[6, 1] = -0.125 * (1 - natural_position[0]) * (1 + natural_position[2])
    dshapefn[6, 2] = -0.125 * (1 - natural_position[0]) * (1 + natural_position[1])

    dshapefn[7, 0] = -0.125 * (1 + natural_position[1]) * (1 + natural_position[2])
    dshapefn[7, 1] = -0.125 * (1 + natural_position[0]) * (1 + natural_position[2])
    dshapefn[7, 2] = -0.125 * (1 + natural_position[0]) * (1 + natural_position[1])
    return dshapefn


@ti.func
def ShapeLinear(xp, xg, idx, lp):
    nx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 1.:
        nx = 1. - d
    return nx


@ti.func
def GShapeLinear(xp, xg, idx, lp):
    dnx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 1.:
        dnx = -sign(xp - xg) * idx
    return dnx


@ti.func
def ShapeLinearCenter(xp, xg, idx, lp):
    return 1./2.


# ========================================================= #
#           Quadratic B-spline shape function               #
# ========================================================= #
@ti.func
def ShapeBsplineQ(xp, xg, idx, lp):
    nx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 1.5:
        if d > 0.5:
            nx = 0.5 * (1.5 - d) * (1.5 - d)
        else:
            nx = 0.75 - d * d
    return nx


@ti.func
def GShapeBsplineQ(xp, xg, idx, lp):
    dnx = 0.
    d = ti.abs(xp - xg) * idx
    a = sign(xp - xg)
    if d < 1.5:
        if d > 0.5:
            dnx = (d - 1.5) * a * idx
        else:
            dnx = -2 * d * a  * idx
    return dnx

@ti.func
def ShapeBsplineQ1(xp, xg, idx, lp):
    nx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 1.5:
        if d > 0.5:
            nx = 0.5 * (1.5 - d) * (1.5 - d)
        else:
            nx = 0.75 - d * d
    return nx

@ti.func
def ShapeBsplineQ2(xp, xg, idx, lp):
    nx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 1.5:
        if d > 0.5:
            nx = 0.5 * (1.5 - d) * (1.5 - d)
        else:
            nx = 0.75 - d * d
    return nx

@ti.func
def ShapeBsplineQ3(xp, xg, idx, lp):
    nx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 1.5:
        if d > 0.5:
            nx = 0.5 * (1.5 - d) * (1.5 - d)
        else:
            nx = 0.75 - d * d
    return nx

@ti.func
def GShapeBsplineQ1(xp, xg, idx, lp):
    dnx = 0.
    d = ti.abs(xp - xg) * idx
    a = sign(xp - xg)
    if d < 1.5:
        if d > 0.5:
            dnx = (d - 1.5) * a * idx
        else:
            dnx = -2 * d * a  * idx
    return dnx

@ti.func
def GShapeBsplineQ2(xp, xg, idx, lp):
    dnx = 0.
    d = ti.abs(xp - xg) * idx
    a = sign(xp - xg)
    if d < 1.5:
        if d > 0.5:
            dnx = (d - 1.5) * a * idx
        else:
            dnx = -2 * d * a  * idx
    return dnx

@ti.func
def GShapeBsplineQ3(xp, xg, idx, lp):
    dnx = 0.
    d = ti.abs(xp - xg) * idx
    a = sign(xp - xg)
    if d < 1.5:
        if d > 0.5:
            dnx = (d - 1.5) * a * idx
        else:
            dnx = -2 * d * a  * idx
    return dnx

# ========================================================= #
#             Cubic B-spline shape function                 #
# ========================================================= #
@ti.func
def ShapeBsplineC(xp, xg, idx, lp):
    nx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 2:
        if d > 1:
            nx = (2 - d) ** 3 / 6
        else:
            nx = 0.5 * d ** 3 - d ** 2 + 2. / 3.
    return nx


@ti.func
def GShapeBsplineC(xp, xg, idx, lp):
    dnx = 0.
    d = ti.abs(xp - xg) * idx
    a = sign(xp - xg)
    if d < 2:
        if d > 1:
            dnx = -0.5 * (2 - d) ** 2 * a * idx
        else:
            dnx = (1.5 * d ** 2 - 2 * d) * a * idx
    return dnx

@ti.func
def ShapeBsplineC1(xp, xg, idx, lp):
    nx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 2:
        if d >= 1:
            nx = -1./6. * d * d * d + d * d - 2 * d + 4./3.
        elif d >= 0:
            nx = 1./6. * d * d * d - d + 1.
        elif d >= -1:
            nx = -1./6. * d * d * d + d + 1.
        elif d >= -2:
            nx = 1./6. * d * d * d + d * d + 2 * d + 4./3.
    return nx


@ti.func
def ShapeBsplineC2(xp, xg, idx, lp):
    nx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 2:
        if d > 1:
            nx = -1./6. * d * d * d + d * d - 2 * d + 4./3.
        elif d >= 0:
            nx = 0.5 * d * d * d - d * d + 2./3.
        elif d >= -1:
            nx = -1./3. * d * d * d - d * d + 2./3.
    return nx


@ti.func
def ShapeBsplineC3(xp, xg, idx, lp):
    nx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 2:
        if d >= 1:
            nx = -1./6. * d * d * d + d * d - 2 * d + 4./3.
        elif d >= 0:
            nx = 0.5 * d * d * d - d * d + 2./3.
        elif d >= -1:
            nx = -0.5 * d * d * d - d * d + 2./3.
        elif d >= -2:
            nx = 1./6. * d * d * d + d * d + 2 * d + 4./3.
    return nx


@ti.func
def ShapeBsplineC4(xp, xg, idx, lp):
    nx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 1:
        if d >= 0:
            nx = 1./3. * d * d * d - d * d + 2./3.
        elif d >= -1:
            nx = -0.5 * d * d * d - d * d + 2./3.
        elif d >= -2:
            nx = 1./6. * d * d * d + d * d + 2 * d + 4./3.
    return nx


@ti.func
def GShapeBsplineC1(xp, xg, idx, lp):
    dnx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 2:
        if d >= 1:
            dnx = -0.5 * d * d + 2 * d - 2.
        elif d >= 0:
            dnx = 0.5 * d * d - 1.
        elif d >= -1:
            dnx = -0.5 * d * d + 1.
        elif d >= -2:
            dnx = 0.5 * d * d + 2 * d + 2.
    return dnx


@ti.func
def GShapeBsplineC2(xp, xg, idx, lp):
    dnx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 2:
        if d > 1:
            dnx = -0.5 * d * d + 2 * d - 2.
        elif d >= 0:
            dnx = 1.5 * d * d - 2 * d 
        elif d >= -1:
            dnx = -1. * d * d - 2 * d 
    return dnx


@ti.func
def GShapeBsplineC3(xp, xg, idx, lp):
    dnx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 2:
        if d >= 1:
            dnx = -0.5 * d * d + 2 * d - 2 
        elif d >= 0:
            dnx = 1.5 * d * d - 2 * d 
        elif d >= -1:
            dnx = -1.5 * d * d - 2 * d 
        elif d >= -2:
            dnx = 0.5 * d * d + 2 * d + 2 
    return dnx


@ti.func
def GShapeBsplineC4(xp, xg, idx, lp):
    dnx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 1:
        if d >= 0:
            dnx = d * d - 2 * d 
        elif d >= -1:
            dnx = -1.5 * d * d - 2 * d 
        elif d >= -2:
            dnx = 0.5 * d * d + 2 * d + 2 
    return dnx


# ========================================================= #
#              Cubic spline radial function                 #
# ========================================================= #
@ti.func
def CubicSpline(smoothing_length, relative_distance):
    norm_distance = relative_distance.norm()
    multiplier = 3.0 / (2.0 * PI * smoothing_length * smoothing_length * smoothing_length)
    radius = norm_distance / smoothing_length
    basis_function = multiplier
    if radius >= 0.0 and radius < 1.0:
        basis_function *= (2.0 / 3.0 - radius * radius + 0.5 * radius * radius * radius)
    elif radius >= 1.0 and radius < 2.0:
        basis_function *= (1.0 / 6.0 * (2.0 - radius) * (2.0 - radius) * (2.0 - radius))
    else:
        basis_function = 0.0
    return basis_function

@ti.func
def GCubicSpline(smoothing_length, relative_distance):
    norm_distance = relative_distance.norm()
    multiplier = 3.0 / (2.0 * PI * smoothing_length * smoothing_length * smoothing_length)
    radius = norm_distance / smoothing_length
    dw_dr = multiplier
    if radius >= 0.0 and radius < 1.0:
        dw_dr *= (-2.0 * radius + 1.5 * radius * radius)
    elif radius >= 1.0 and radius < 2.0:
        dw_dr *= (-0.5 * (2.0 - radius) * (2.0 - radius))
    else:
        dw_dr = 0.0
    
    gradient = relative_distance
    if norm_distance > Threshold:
        gradient *= dw_dr  /(norm_distance * smoothing_length)
    else:
        gradient *= 0.
    return gradient


# ========================================================= #
#             Quintic spline radial function                #
# ========================================================= #
@ti.func
def QuinticSplineRadial(smoothing_length, relative_distance):
    norm_distance = relative_distance.norm()
    multiplier = 3.0 / (359.0 * PI * smoothing_length * smoothing_length * smoothing_length)
    radius = norm_distance / smoothing_length
    basis_function = multiplier
    if radius >= 0.0 and radius < 1.0:
        basis_function *= ((3.0 - radius) ** 5 - 6.0 * (2.0 - radius) ** 5 + 15.0 * (1.0 - radius) ** 5)
    elif radius >= 1.0 and radius < 2.0:
        basis_function *= ((3.0 - radius) ** 5 - 6.0 * (2.0 - radius) ** 5)
    elif radius >= 2.0 and radius < 3.0:
        basis_function *= (3.0 - radius) ** 5
    else:
        basis_function = 0.0
    return basis_function

@ti.func
def GQuinticSplineRadial(smoothing_length, relative_distance):
    norm_distance = relative_distance.norm()
    multiplier = 3.0 / (359.0 * PI * smoothing_length * smoothing_length * smoothing_length)
    radius = norm_distance / smoothing_length
    dw_dr = multiplier
    if radius >= 0.0 and radius < 1.0:
        dw_dr *= (-5.0 * (3.0 - radius) ** 4 + 30. * (2.0 - radius) ** 4 - 75. * (1.0 - radius) ** 4)
    elif radius >= 1.0 and radius < 2.0:
        dw_dr *= (-5.0 * (3.0 - radius) ** 4 + 30. * (2.0 - radius) ** 4)
    elif radius >= 2.0 and radius < 3.0:
        dw_dr *= -5.0 * (3.0 - radius) ** 4
    else:
        dw_dr = 0.0
    
    gradient = relative_distance
    if norm_distance > Threshold:
        gradient *= dw_dr  /(norm_distance * smoothing_length)
    else:
        gradient *= 0.
    return gradient


# ========================================================= #
#                      Guass Kernel                         #
# ========================================================= #
@ti.func
def Guassian(smoothing_length, relative_distance):
    norm_distance = relative_distance.norm()
    multiplier = 1.0 / ((SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length))
    radius = norm_distance / smoothing_length
    basis_function = multiplier
    if radius >= 0.0 and radius <= 3.0:
        basis_function *= ti.exp(-radius * radius)
    else:
        basis_function = 0.0
    return basis_function

@ti.func
def GGuassian(smoothing_length, relative_distance):
    norm_distance = relative_distance.norm()
    multiplier = 1.0 / ((SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length))
    radius = norm_distance / smoothing_length
    dw_dr = multiplier
    if radius >= 0.0 and radius <= 3.0:
        dw_dr *= -2. * radius * ti.exp(-radius * radius)
    else:
        dw_dr = 0.0

    gradient = relative_distance
    if norm_distance > Threshold:
        gradient *= dw_dr  /(norm_distance * smoothing_length)
    else:
        gradient *= 0.
    return gradient


# ========================================================= #
#                   Super Guass Kernel                      #
# ========================================================= #
@ti.func
def SuperGuassian(smoothing_length, relative_distance):
    norm_distance = relative_distance.norm()
    multiplier = 1.0 / ((SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length))
    radius = norm_distance / smoothing_length
    basis_function = multiplier
    if radius >= 0.0 and radius <= 3.0:
        basis_function *= ti.exp(-radius * radius) * (3./2. + 1. - radius * radius)
    else:
        basis_function = 0.0
    return basis_function

@ti.func
def GSuperGuassian(smoothing_length, relative_distance):
    norm_distance = relative_distance.norm()
    multiplier = 1.0 / ((SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length))
    radius = norm_distance / smoothing_length
    dw_dr = multiplier
    if radius >= 0.0 and radius <= 3.0:
        dw_dr *= radius * ti.exp(-radius * radius) * (-3 + 2. * radius * radius - 4.)
    else:
        dw_dr = 0.0
    
    gradient = relative_distance
    if norm_distance > Threshold:
        gradient *= dw_dr  /(norm_distance * smoothing_length)
    else:
        gradient *= 0.
    return gradient


# ========================================================= #
#                    Wieght Functions                       #
# ========================================================= #
@ti.func
def CubicSplineWeight(xp, xg, idx):
    weight = 0.
    radius = ti.abs(xp - xg) * idx
    if radius <= 0.5: 
        weight = 2./3. - 4 * radius * radius + 4 * radius * radius * radius
    elif radius <= 1:
        weight = 4./3. - 4. * radius + 4 * radius - radius - 4./3. * radius * radius * radius
    return weight

@ti.func
def QuarticSplineWeight(xp, xg, idx):
    weight = 0.
    radius = ti.abs(xp - xg) * idx
    if radius <= 1:
        weight = 1. - 6. * radius * radius + 8 * radius * radius * radius - 3. * radius * radius * radius * radius
    return weight

@ti.func
def CubicSplineWeight3D(xp, xg, idx):
    wx = CubicSplineWeight(xp[0], xg[0], idx[0])
    wy = CubicSplineWeight(xp[0], xg[0], idx[0]) 
    wz = CubicSplineWeight(xp[0], xg[0], idx[0])
    return wx * wy * wz

@ti.func
def QuarticSplineWeight3D(xp, xg, idx):
    wx = QuarticSplineWeight(xp[0], xg[0], idx[0])
    wy = QuarticSplineWeight(xp[0], xg[0], idx[0]) 
    wz = QuarticSplineWeight(xp[0], xg[0], idx[0])
    return wx * wy * wz


# ========================================================= #
#                  Heaviside Functions                      #
# ========================================================= #
@ti.func
def SmoothedHeavisideFunction(epsilon, phi):
    h = 0.
    ieps = 1. / epsilon
    if phi > -epsilon:
        if phi < epsilon:
            h = 0.5 * (1 + phi * ieps + ti.sin(PI * phi * ieps) * iPI)
        else:
            h = 1.
    return h
