import taichi as ti


@ti.func
def find_span(start_knot, num_ctrlpts, knot, knot_vector):
    low = 2
    high = num_ctrlpts
    span = ti.cast(0.5 * (low + high), int)
    if knot > knot_vector[num_ctrlpts + start_knot]:
        span = num_ctrlpts - 1
    elif knot < knot_vector[2 + start_knot]:
        span = 2
    else:
        if knot == knot_vector[num_ctrlpts + start_knot]:
            span = num_ctrlpts - 1
        else:
            while (knot < knot_vector[span + start_knot] or knot >= knot_vector[span + 1 + start_knot]) and high != low:
                if knot < knot_vector[span + start_knot]:
                    high = span
                else:
                    low = span
                span = ti.cast(0.5 * (low + high), int)
    return span


@ti.func
def BSplineLeftBoundary(knot_val):
    return (knot_val - 1.0) * (knot_val - 1.0)


@ti.func
def BSplineLeftInterior1(knot_val):
    return -1.5 * knot_val * knot_val + 2.0 * knot_val


@ti.func
def BSplineLeftInterior2(knot_val):
    return 0.5 * knot_val * knot_val - knot_val + 0.5


@ti.func
def BSplineInterior1(knot_val):
    return 0.5 * knot_val * knot_val


@ti.func
def BSplineInterior2(knot_val):
    return -knot_val * knot_val + knot_val + 0.5


@ti.func
def BSplineInterior3(knot_val):
    return 0.5 * (knot_val - 1.0) * (knot_val - 1.0)


@ti.func
def BSplineRightInterior1(knot_val):
    return 0.5 * knot_val * knot_val


@ti.func
def BSplineRightInterior2(knot_val):
    return -1.5 * knot_val * knot_val + knot_val + 0.5


@ti.func
def BSplineRightBoundary(knot_val):
    return knot_val * knot_val


@ti.func
def BSplineDerivativeLeftBoundary(knot_val):
    return 2 * (knot_val - 1.0)


@ti.func
def BSplineDerivativeLeftInterior1(knot_val):
    return -3.0 * knot_val + 2.0


@ti.func
def BSplineDerivativeLeftInterior2(knot_val):
    return knot_val - 1.0


@ti.func
def BSplineDerivativeInterior1(knot_val):
    return knot_val


@ti.func
def BSplineDerivativeInterior2(knot_val):
    return -2.0 * knot_val + 1.0


@ti.func
def BSplineDerivativeInterior3(knot_val):
    return knot_val - 1.0


@ti.func
def BSplineDerivativeRightInterior1(knot_val):
    return knot_val


@ti.func
def BSplineDerivativeRightInterior2(knot_val):
    return -3.0 * knot_val + 1.0


@ti.func
def BSplineDerivativeRightBoundary(knot_val):
    return 2.0 * knot_val


@ti.func
def BSplineSecondDerivativeLeftBoundary(knot_val):
    return 2


@ti.func
def BSplineSecondDerivativeLeftInterior1(knot_val):
    return -3.0


@ti.func
def BSplineSecondDerivativeLeftInterior2(knot_val):
    return 1.0


@ti.func
def BSplineSecondDerivativeInterior1(knot_val):
    return 1.0


@ti.func
def BSplineSecondDerivativeInterior2(knot_val):
    return -2.0


@ti.func
def BSplineSecondDerivativeInterior3(knot_val):
    return 1.0


@ti.func
def BSplineSecondDerivativeRightInterior1(knot_val):
    return 1.0


@ti.func
def BSplineSecondDerivativeRightInterior2(knot_val):
    return -3.0


@ti.func
def BSplineSecondDerivativeRightBoundary(knot_val):
    return 2.0


@ti.func
def choose_bspline_shape_function(num_ctrlpts, span, knot_val):
    shape_func = ti.Vector([0.0, 0.0, 0.0])
    if span == 2:
        shape_func = ti.Vector(
            [BSplineLeftBoundary(knot_val), BSplineLeftInterior1(knot_val), BSplineInterior1(knot_val)]
        )
    elif span == 3:
        shape_func = ti.Vector([BSplineLeftInterior2(knot_val), BSplineInterior2(knot_val), BSplineInterior1(knot_val)])
    elif span == num_ctrlpts - 2:
        shape_func = ti.Vector(
            [BSplineInterior3(knot_val), BSplineRightInterior2(knot_val), BSplineRightBoundary(knot_val)]
        )
    elif span == num_ctrlpts - 3:
        shape_func = ti.Vector(
            [BSplineInterior3(knot_val), BSplineInterior2(knot_val), BSplineRightInterior1(knot_val)]
        )
    else:
        shape_func = ti.Vector([BSplineInterior3(knot_val), BSplineInterior2(knot_val), BSplineInterior1(knot_val)])
    return shape_func


@ti.func
def choose_bspline_derivative_shape_function(num_ctrlpts, span, knot_val):
    dshape_func = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    if span == 2:
        dshape_func = ti.Vector(
            [
                [BSplineLeftBoundary(knot_val), BSplineLeftInterior1(knot_val), BSplineInterior1(knot_val)],
                [
                    BSplineDerivativeLeftBoundary(knot_val),
                    BSplineDerivativeLeftInterior1(knot_val),
                    BSplineDerivativeInterior1(knot_val),
                ],
            ]
        )
    elif span == 3:
        dshape_func = ti.Vector(
            [
                [BSplineLeftInterior2(knot_val), BSplineInterior2(knot_val), BSplineInterior1(knot_val)],
                [
                    BSplineDerivativeLeftInterior2(knot_val),
                    BSplineDerivativeInterior2(knot_val),
                    BSplineDerivativeInterior1(knot_val),
                ],
            ]
        )
    elif span == num_ctrlpts - 2:
        dshape_func = ti.Vector(
            [
                [BSplineInterior3(knot_val), BSplineRightInterior2(knot_val), BSplineRightBoundary(knot_val)],
                [
                    BSplineDerivativeInterior3(knot_val),
                    BSplineDerivativeRightInterior2(knot_val),
                    BSplineDerivativeRightBoundary(knot_val),
                ],
            ]
        )
    elif span == num_ctrlpts - 3:
        dshape_func = ti.Vector(
            [
                [BSplineInterior3(knot_val), BSplineInterior2(knot_val), BSplineRightInterior1(knot_val)],
                [
                    BSplineDerivativeInterior3(knot_val),
                    BSplineDerivativeInterior2(knot_val),
                    BSplineDerivativeRightInterior1(knot_val),
                ],
            ]
        )
    else:
        dshape_func = ti.Vector(
            [
                [BSplineInterior3(knot_val), BSplineInterior2(knot_val), BSplineInterior1(knot_val)],
                [
                    BSplineDerivativeInterior3(knot_val),
                    BSplineDerivativeInterior2(knot_val),
                    BSplineDerivativeInterior1(knot_val),
                ],
            ]
        )
    return dshape_func


@ti.func
def quadratic_nurbs_derivative(
    start_knot_u,
    start_knot_v,
    start_knot_w,
    start_ctrlpts,
    num_knot_u,
    num_knot_v,
    num_knot_w,
    xi,
    eta,
    zeta,
    knot_vector_u,
    knot_vector_v,
    knot_vector_w,
    weight,
):
    N = ti.Vector.zero(float, 3 * 3 * 3)
    dNdnat = ti.Matrix.zero(float, 3 * 3 * 3, 3)
    num_ctrlpts_u = num_knot_u - 1 - 2
    num_ctrlpts_v = num_knot_v - 1 - 2
    num_ctrlpts_w = num_knot_w - 1 - 2

    spanU = find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
    spanV = find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
    spanW = find_span(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
    dNurbsU = choose_bspline_derivative_shape_function(num_ctrlpts_u, spanU, xi)
    dNurbsV = choose_bspline_derivative_shape_function(num_ctrlpts_v, spanV, eta)
    dNurbsW = choose_bspline_derivative_shape_function(num_ctrlpts_w, spanW, zeta)

    w, dwdxi, dwdet, dwdze = 0.0, 0.0, 0.0, 0.0
    uind = spanU - 2
    for i in ti.static(range(2 + 1)):
        wind = spanW - 2 + i
        for j in ti.static(range(2 + 1)):
            vind = spanV - 2 + j
            for k in ti.static(range(2 + 1)):
                linear_id = start_ctrlpts + uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                wgt = weight[linear_id]
                w += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
                dwdxi += dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
                dwdet += dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * wgt
                dwdze += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * wgt

    kk = 0
    uind = spanU - 2
    for i in ti.static(range(3)):
        wind = spanW - 2 + i
        for j in ti.static(range(3)):
            vind = spanV - 2 + j
            for k in ti.static(range(3)):
                linear_id = start_ctrlpts + uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                fac = weight[linear_id] / (w * w)
                nmp = dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i]
                N[kk] = nmp * fac * w
                dNdnat[kk, 0] = (dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * w - nmp * dwdxi) * fac
                dNdnat[kk, 1] = (dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * w - nmp * dwdet) * fac
                dNdnat[kk, 2] = (dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * w - nmp * dwdze) * fac
                kk += 1
    return N, dNdnat
