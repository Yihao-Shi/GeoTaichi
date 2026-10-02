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
#                  AxisyGIMP shape function                 #
# ========================================================= #
# refer to Axisymmetric Generalized Interpolation Material Point Method for Fully Coupled Thermomechanical Evalution of Transient Responses. International Journal of Computational Methods
@ti.func
def ShapeAxisyGIMP(xp, xg, idx, lp):
    nx = 0.
    dx = 1. / idx
    d = xp - xg
    a = dx + lp
    b = dx * lp
    if xg == 0 and (0 <= xp <= lp):
        nx = 1. - 2. * (lp + xp) / (3. * dx)
    elif xg == dx and (0 <= xp <= lp):
        nx =2. * (lp + xp) / (3. * dx)
    else:
        if d < a:
            if d > (dx - lp):
                nx = (dx - 2. * lp + xg + 2. * xp) * (a - d) * (a - d)/ (12. * b * xp)
            elif d > lp:
                nx = 1. - (lp * lp + 3. * xp * xp - 3. * xg * xp)/(3. * dx * xp)
            elif d > -lp:
                nx = 1. - (2. * xp * xp * xp + xg * xg * xg - 3. * lp * lp * xg + 6. * lp * lp * xp - 3. * xp * xp * xg) / (6. * b * xp)
            elif d > (-dx + lp):
                nx = 1. + (lp * lp + 3. * xp * xp - 3. * xg * xp)/(3. * dx * xp)
            elif d > -a:
                nx = (2. * lp - dx + xg + 2. * xp) * (a + d) * (a + d)/ (12. * b * xp)
    return nx


@ti.func
def GShapeAxisyGIMP(xp, xg, idx, lp):
    dnx = 0.
    dx = 1. / idx
    d = xp - xg
    a = dx + lp
    b = dx * lp
    if xg ==0 and (0 <= xp <= lp):
        dnx = -1. * idx
    elif xg == dx and (0 <= xp <= lp):
        dnx = 1. * idx
    else:
        if d < a:
            if d > (dx - lp):
                dnx = ((lp - xp) * (lp - xp) - (dx + xg) * (dx + xg)) / (4. * b * xp)
            elif d > lp:
                dnx = -1. * idx
            elif d > -lp:
                dnx = (xg * xg - lp * lp- xp * xp) / (2. * b * xp)
            elif d > (-dx + lp):
                dnx = 1. * idx
            elif d > -a:
                dnx = ((lp + xp) * (lp + xp) - (dx - xg) * (dx - xg)) / (4. * b * xp)
    return dnx


@ti.func
def ShapeAixGIMPCenter(xp, xg, idx, lp):
    nxc = 0.
    dx = 1. / idx
    d = xp - xg
    a = dx + lp
    b = dx * lp
    if abs(d) < a:
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
#           Quadratic bernstein shape function              #
# ========================================================= #
@ti.func
def ShapeBernsteinQ(xp, xg, idx, btype):
    nx = 0.
    d = ti.abs(xp - xg) * idx
    if d < 1.0: 
        if btype == 1:                            # Inside node:
            if d >= 0.5: 
                nx = 0.
            else:
                nx = 0.5 - 2. * d * d
        else:                                     # Edge node:
            nx = (1 - d) * (1 - d)
    return nx


@ti.func
def GShapeBernsteinQ(xp, xg, idx, btype):
    dnx = 0.
    d_signed = (xp - xg) * idx
    d = ti.abs(d_signed) 
    if d < 1.0: 
        if btype == 1:                            # Inside node:
            if d >= 0.5: 
                dnx = -4 * d_signed
            else:
                dnx = 0.5 - 2. * d * d
        else:                                     # Edge node:
            if d_signed > 0.:
                dnx = -2 * (1 - d_signed)
            else:
                dnx = 2 * (1 + d_signed)
    return dnx * idx


# ========================================================= #
#           Quadratic B-spline shape function               #
# ========================================================= #
@ti.func
def ShapeBsplineQ(xp, xg, idx, btype):
    nx = 0.
    d = (xp - xg) * idx
    if btype == 0:
        if d >= 0.5 and d < 1.5:
            nx = (0.5 * d - 1.5) * d + 1.125
        elif d >= -0.5 and d < 0.5:
            nx = -d * d + 0.75
        elif d >= -1.5 and d < 0.5:
            nx = (0.5 * d + 1.5) * d + 1.125
    elif btype == 1:
        if d >= 0. and d < 0.5:
            nx = 1. - d
        elif d >= 0.5 and d < 1.5:
            nx = (0.5 * d - 1.5) * d + 1.125
    elif btype == 2:
        if d >= -1. and d < -0.5:
            nx = 1. + d
        elif d >= -0.5 and d < 0.5:
            nx = -d * d + 0.75
        elif d >= 0.5 and d < 1.5:
            nx = (0.5 * d - 1.5) * d + 1.125
    elif btype == 3:
        if d >= -1.5 and d < -0.5:
            nx = (0.5 * d + 1.5) * d + 1.125
        elif d >= -0.5 and d < 0.5:
            nx = -d * d + 0.75
        elif d >= 0.5 and d < 1.:
            nx = 1. - d
    elif btype == 4:
        if d >= -1.5 and d < -0.5:
            nx = (0.5 * d + 1.5) * d + 1.125
        elif d >= -0.5 and d <= 0.:
            nx = 1. + d
    return nx

@ti.func
def GShapeBsplineQ(xp, xg, idx, btype):
    dnx = 0.
    d = (xp - xg) * idx
    if btype == 0:
        if d >= 0.5 and d < 1.5:
            dnx = d - 1.5
        elif d >= -0.5 and d < 0.5:
            dnx = -2. * d 
        elif d >= -1.5 and d < 0.5:
            dnx = d + 1.5
    elif btype == 1:
        if d >= 0. and d < 0.5:
            dnx = -1.
        elif d >= 0.5 and d < 1.5:
            dnx = d - 1.5
    elif btype == 2:
        if d >= -1. and d < -0.5:
            dnx = 1. 
        elif d >= -0.5 and d < 0.5:
            dnx = -2. * d
        elif d >= 0.5 and d < 1.5:
            dnx = d - 1.5
    elif btype == 3:
        if d >= -1.5 and d < -0.5:
            dnx = d + 1.5
        elif d >= -0.5 and d < 0.5:
            dnx = -2. * d
        elif d >= 0.5 and d < 1.:
            dnx = -1.
    elif btype == 4:
        if d >= -1.5 and d < -0.5:
            dnx = d + 1.5
        elif d >= -0.5 and d <= 0.:
            dnx = 1. 
    return dnx * idx

# ========================================================= #
#             Cubic B-spline shape function                 #
# ========================================================= #
@ti.func
def ShapeBsplineC(xp, xg, idx, btype):
    nx = 0.
    d = (xp - xg) * idx
    if d >= 1 and d < 2:
        if btype != 3:
            nx = ((-1./6. * d + 1) * d - 2) * d + 4.0/3.0
    elif d >= 0 and d < 1:
        if btype == 1:
            nx = (1./6. * d * d - 1) * d + 1
        elif btype == 3:
            nx = (1./3. * d - 1) * d * d + 2./3.
        else:
            nx = (0.5 * d - 1) * d * d + 2./3.
    elif d >= -1 and d < 0:
        if btype == 4:
            nx = (-1./6. * d * d + 1) * d + 1
        elif btype == 2:
            nx = (-1./3. * d - 1) * d * d + 2./3.
        else:
            nx = (-0.5 * d - 1) * d * d + 2./3.
    elif d >= -2 and d < -1:
        nx = ((1./6. * d + 1) * d + 2) * d + 4./3.
    return nx

@ti.func
def GShapeBsplineC(xp, xg, idx, btype):
    dnx = 0.
    d = (xp - xg) * idx
    if d >= 1 and d < 2:
        if btype != 3:
            dnx = (-0.5 * d + 2) * d - 2.
    elif d >= 0 and d < 1:
        if btype == 1:
            dnx = 0.5 * d * d - 1.
        elif btype == 3:
            dnx = d * (d - 2.)
        else:
            dnx = (3.0 / 2.0 * d - 2) * d
    elif d >= -1 and d < 0:
        if btype == 4:
            dnx = -0.5 * d * d + 1.
        elif btype == 2:
            dnx = (-d - 2.) * d
        else:
            dnx = (-3./2. * d - 2.) * d
    elif d >= -2 and d < -1:
        dnx = (0.5 * d + 2.) * d + 2.
    return dnx * idx

@ti.func
def ShapeBsplineC_THB(xp, xg, dx, nlevel, ntype):
    nx = 0.
    r = xp - xg
    subdx = dx / (2.0 ** nlevel)
    r /= subdx
    if ntype == 3:
        if -2.00 <= r < -1.00:
            nx = r**3 / 6.0 + r**2 + 2.0*r + 4.0/3.0
        elif -1.00 <= r < 0.00:
            nx = -r**3 / 2.0 - r**2 + 2.0/3.0
        elif 0.00 <= r < 1.00:
            nx = 0.5*r**3 - r**2 + 2.0/3.0
        elif 1.00 <= r <= 2.00:
            nx = -r**3 / 6.0 + r**2 - 2.0*r + 4.0/3.0
        else:
            nx = 0.0
    elif ntype ==1 or ntype ==9:
        if 0.00 <= r < 1.00:
            nx = r**3 / 6.0 - r + 1.0
        elif 1.00 <= r <= 2.00:
            nx = -r**3 / 6.0 + r**2 - 2.0*r + 4.0/3.0
        elif -2.00 <= r < -1.00:
            nx = r**3 / 6.0 + r**2 + 2.0*r + 4.0/3.0
        elif -1.00 <= r < 0.00:
            nx = -r**3 / 6.0 + r + 1.0
    elif ntype == 2:
        if -1.00 <= r < 0.00:
            nx = -r**3 / 3.0 - r**2 + 2.0/3.0
        elif 0.00 <= r < 1.00:
            nx = 0.5*r**3 - r**2 + 2.0/3.0
        elif 1.00 <= r <= 2.00:
            nx = -r**3 / 6.0 + r**2 - 2.0*r + 4.0/3.0
    elif ntype == 8:
        if -2.00 <= r < -1.00:
            nx = r**3 / 6.0 + r**2 + 2.0*r + 4.0/3.0
        elif -1.00 <= r < 0.00:
            nx = -r**3 / 2.0 - r**2 + 2.0/3.0
        elif 0.00 <= r < 1.00:
            nx = r**3 / 3.0 - r**2 + 2.0/3.0
    elif ntype == 4:
        if -2.00 <= r < -1.00:
            nx = r**3 /6.0 + r**2 + 2.0*r + 4.0/3.0
        elif -1.00 <= r < 0.00:
            nx = -r**3/ 2.0 - r**2 + 2.0/3.0
        elif 0.00 <= r < 0.50:
            nx = r**3 / 3.0 - r**2 + 2.0/3.0
        elif 0.50 <= r < 1.00:
            nx = r**3 - 2.0*r**2 + 0.5*r + 7.0/12.0
        elif 1.00 <= r < 1.50:
            nx = -2.0/3.0*r**3 + 3.0*r**2 - 4.5*r + 9.0/4.0
    elif ntype == 7:
        if 1.00 <= r < 2.00:
            nx = -r**3 /6.0 + r**2 - 2.0*r + 4.0/3.0
        elif 0.00 <= r < 1.00:
            nx = r**3/ 2.0 - r**2 + 2.0/3.0
        elif -0.50 <= r < 0.00:
            nx = -r**3/3.0 - r**2 + 2.0/3.0
        elif -1.00 <= r < -0.50:
            nx = -r**3 - 2.0*r**2 - 0.5*r + 7.0/12.0
        elif -1.50 <= r < -1.00:
            nx = 2.0/3.0*r**3 + 3.0*r**2 + 4.5*r + 9.0/4.0
    elif ntype == 5:
        if -1.50 <= r < -0.5:
            nx = 1.0/6.0*r**3 + 3.0/4.0*r**2 + 9.0/8.0*r + 9.0/16.0
        elif -0.50 <= r < 0.00:
            nx = -3.0/2.0*r**3 - 7.0/4.0*r**2 - 1.0/8.0*r + 17.0/48.0
        elif 0.00 <= r < 0.50:
            nx = 11.0/6.0*r**3 - 7.0/4.0*r**2 - 1.0/8.0*r + 17.0/48.0
        elif 0.50 <= r < 1.00:
            nx = -2.0/3.0*r**3 + 2.0*r**2 - 2.0*r + 2.0/3.0
    elif ntype == 6:
        if 0.50 <= r < 1.50:
            nx = -1.0/6.0*r**3 + 3.0/4.0*r**2 - 9.0/8.0*r + 9.0/16.0
        elif 0.00 <= r < 0.50:
            nx = 3.0/2.0*r**3 - 7.0/4.0*r**2 + 1.0/8.0*r + 17.0/48.0
        elif -0.50 <= r < 0.00:
            nx = -11.0/6.0*r**3 - 7.0/4.0*r**2 + 1.0/8.0*r + 17.0/48.0
        elif -1.00 <= r < -0.50:
            nx = 2.0/3.0*r**3 + 2.0*r**2 + 2.0*r + 2.0/3.0
    return nx


@ti.func
def GShapeBsplineC_THB(xp, xg, dx, nlevel, ntype):
    dnx = 0.
    r = xp - xg
    subdx = dx / (2.0 ** nlevel)
    r /= subdx
    if ntype == 3:
        if -2.00 <= r < -1.00:
            dnx = r**2 /2.0 + 2.0*r + 2.0
        elif -1.00 <= r < 0.00:
            dnx = -1.5 * r**2 - 2.0*r
        elif 0.00 <= r < 1.00:
            dnx = 1.5 * r**2 - 2.0*r
        elif 1.00 <= r <= 2.00:
            dnx = -0.5*r**2 + 2.0*r - 2.0
        else:
            dnx = 0.0
    elif ntype == 1 or ntype == 9:
        if 0.00 <= r < 1.00:
            dnx = 0.5 * r**2 - 1.0
        elif 1.00 <= r <= 2.00:
            dnx = -0.5*r**2 + 2.0*r - 2.0
        elif -2.00 <= r < -1.00:
            dnx = r**2 /2.0 + 2.0*r + 2.0
        elif -1.00 <= r < 0.00:
            dnx = -0.5 * r**2 + 1.0
    elif ntype == 2:
        if -1.00 <= r < 0.00:
            dnx = -r**2 - 2.0*r
        elif 0.00 <= r < 1.00:
            dnx = 1.5 * r**2 - 2.0*r
        elif 1.00 <= r <= 2.00:
            dnx = -0.5*r**2 + 2.0*r - 2.0
    elif ntype == 8:
        if -2.00 <= r < -1.00:
            dnx = r**2 /2.0 + 2.0*r + 2.0
        elif -1.00 <= r < 0.00:
            dnx = -1.5 * r**2 - 2.0*r
        elif 0.00 <= r < 1.00:
            dnx = r**2 - 2.0*r
    elif ntype == 4:
        if -2.00 <= r < -1.00:
            dnx = r**2 /2.0 + 2.0*r + 2.0
        elif -1.00 <= r < 0.00:
            dnx = -1.5 * r**2 - 2.0*r
        elif 0.00 <= r < 0.50:
            dnx = r**2 - 2.0*r
        elif 0.50 <= r < 1.00:
            dnx = 3.0*r**2 - 4.0*r + 0.5
        elif 1.00 <= r < 1.50:
            dnx = -2.0*r**2 + 6.0*r - 4.5
    elif ntype == 7:
        if 1.00 <= r < 2.00:
            dnx = -r**2 /2.0 + 2.0*r - 2.0
        elif 0.00 <= r < 1.00:
            dnx = 1.5 * r**2 - 2.0*r
        elif -0.50 <= r < 0.00:
            dnx = -r**2 - 2.0*r
        elif -1.00 <= r < -0.50:
            dnx = -3.0*r**2 - 4.0*r - 0.5
        elif -1.50 <= r < -1.00:
            dnx = 2.0*r**2 + 6.0*r + 4.5
    elif ntype == 5:
        if -1.50 <= r < -0.5:
            dnx = 0.5*r**2 + 1.5*r + 9.0/8.0
        elif -0.50 <= r < 0.00:
            dnx = -4.5* r**2 - 3.5*r - 1.0/8.0
        elif 0.00 <= r < 0.50:
            dnx = 11.0/2.0* r**2 - 3.5*r - 1.0/8.0
        elif 0.50 <= r < 1.00:
            dnx = -2.0*r**2 + 4.0*r - 2.0
    elif ntype == 6:
        if 0.50 <= r < 1.50:
            dnx = -0.5*r**2 + 1.5*r - 9.0/8.0
        elif 0.00 <= r < 0.50:
            dnx = 4.5* r**2 - 3.5*r + 1.0/8.0
        elif -0.50 <= r < 0.00:
            dnx = -11.0/2.0* r**2 - 3.5*r + 1.0/8.0
        elif -1.00 <= r < -0.50:
            dnx = 2.0*r**2 + 4.0*r + 2.0
    return dnx / subdx


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
        gradient *= dw_dr / (norm_distance * smoothing_length)
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
def Guassian(smoothing_length, relative_distance, cut_off=3.):
    norm_distance = relative_distance.norm()
    multiplier = 1.0 / ((SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length))
    radius = norm_distance / smoothing_length
    basis_function = multiplier
    if radius >= 0.0 and radius <= cut_off:
        basis_function *= ti.exp(-radius * radius)
    else:
        basis_function = 0.0
    return basis_function

@ti.func
def GGuassian(smoothing_length, relative_distance, cut_off=3.):
    norm_distance = relative_distance.norm()
    multiplier = 1.0 / ((SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length))
    radius = norm_distance / smoothing_length
    dw_dr = multiplier
    if radius >= 0.0 and radius <= cut_off:
        dw_dr *= -2. * radius * ti.exp(-radius * radius)
    else:
        dw_dr = 0.0

    gradient = relative_distance
    if norm_distance > Threshold:
        gradient *= dw_dr / (norm_distance * smoothing_length)
    else:
        gradient *= 0.
    return gradient


# ========================================================= #
#                   Super Guass Kernel                      #
# ========================================================= #
@ti.func
def SuperGuassian(smoothing_length, relative_distance, cut_off=3.):
    norm_distance = relative_distance.norm()
    multiplier = 1.0 / ((SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length))
    radius = norm_distance / smoothing_length
    basis_function = multiplier
    if radius >= 0.0 and radius <= cut_off:
        basis_function *= ti.exp(-radius * radius) * (3./2. + 1. - radius * radius)
    else:
        basis_function = 0.0
    return basis_function

@ti.func
def GSuperGuassian(smoothing_length, relative_distance, cut_off=3.):
    norm_distance = relative_distance.norm()
    multiplier = 1.0 / ((SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length) * (SQRT_PI * smoothing_length))
    radius = norm_distance / smoothing_length
    dw_dr = multiplier
    if radius >= 0.0 and radius <= cut_off:
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
#                    Weight Functions                       #
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


# ========================================================= #
#               Distortion Ratio Functions                  #
# ========================================================= #
@ti.func
def DistortionRatioFuction(eta):
    value = 0.
    if 0 <= eta < 0.5:
        value = 0.5 * eta
    elif 0.5 <= eta < 1.:
        value = -6. * eta * eta * eta + 14. * eta * eta - 9. * eta + 2.
    elif eta >= 1.:
        value = eta
    return value


@ti.func
def CubicSmooth(relative_distance, support_domain):
    norm_distance = relative_distance.norm()
    relative_position = norm_distance / support_domain
    single_value = 1 - relative_position * relative_position
    return single_value * single_value * single_value if relative_position < 1. else 0.