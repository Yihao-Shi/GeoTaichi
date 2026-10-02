import taichi as ti

from src.utils.constants import WENO_EPS


# Domain cells (I{i}) reference:
#
#                |           |   u(i)    |           |
#                |  u(i-1)   |___________|           |
#                |___________|           |   u(i+1)  |
#                |           |           |___________|
#             ...|-----0-----|-----0-----|-----0-----|...
#                |    i-1    |     i     |    i+1    |
#                |-         +|-         +|-         +|
#              i-3/2       i-1/2       i+1/2       i+3/2
#
# ENO stencils (S{r}) reference:
#
#
#                               |___________S2__________|
#                               |                       |
#                       |___________S1__________|       |
#                       |                       |       |
#               |___________S0__________|       |       |
#             ..|---o---|---o---|---o---|---o---|---o---|...
#               | I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}|
#                                      -|
#                                     i+1/2
#
#
#               |___________S0__________|
#               |                       |
#               |       |___________S1__________|
#               |       |                       |
#               |       |       |___________S2__________|
#             ..|---o---|---o---|---o---|---o---|---o---|...
#               | I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}|
#                               |+
#                             i-1/2
#
# WENO stencil: S{i} = [ I{i-2},...,I{i+2} ]

# =================================================================================== #
#                                Linear inteploation                                  #
# =================================================================================== #
@ti.func
def linear_mid_uniform_grid(fi, fip):
    return 0.5 * (fi + fip)

@ti.func
def linear_uniform_grid(nodeic, nodeip, pos, fi, fip):
    return (nodeip - pos) * fi + (pos - nodeic) * fip

@ti.func
def linear(xs, xi, xip, fi, fip):
    hi = xi - xs
    return ((xip - xs) * fi + (xs - xi) * fip) / hi


# =================================================================================== #
#                                 WENO3 inteploation                                  #
# =================================================================================== #
# reference: G. Janet et al. A novel fourth-order WENO interpolation technique: A possible new tool designed for radiative transfer. Astronomy & Astrophysics. 2021 1-16.
@ti.func
def weno3_mid_uniform_grid(yim, yi, yip):
    # NOTE(cmo): Quadratics over substencils
    q1 = (-yim * 0.5 + yi * 1.5)
    q2 = (yi + yip) * 0.5

    yyim = yi - yim
    yyi = yip - yi
    yyyi = yip - yim

    # NOTE(cmo): Smoothness indicators
    beta1 = 0.25 * (ti.abs(yyyi) - ti.abs(3. * yyim - yyi)) * (ti.abs(yyyi) - ti.abs(3. * yyim - yyi))
    beta2 = 0.25 * (ti.abs(yyyi) - ti.abs(yyim - 3. * yyi)) * (ti.abs(yyyi) - ti.abs(yyim - 3. * yyi))

    # NOTE(cmo): Linear weights
    # NOTE(cmo): Non-linear weights
    alpha1 = 0.25 / (WENO_EPS + beta1) ** 1.5
    alpha2 = 0.75 / (WENO_EPS + beta2) ** 1.5

    omega1 = alpha1 / (alpha1 + alpha2)
    omega2 = alpha2 / (alpha1 + alpha2)
    return omega1 * q1 + omega2 * q2

@ti.func
def weno3_uniform_grid(nodeim, nodeic, nodeip, pos, yim, yi, yip):
    # NOTE(cmo): Quadratics over substencils
    q1 = (-yim * (pos - nodeic) + yi * (pos - nodeim))
    q2 = (-yi * (pos - nodeip) + yip * (pos - nodeic)) 

    yyim = yi - yim
    yyi = yip - yi
    yyyi = yip - yim

    # NOTE(cmo): Smoothness indicators
    beta1 = 0.25 * (ti.abs(yyyi) - ti.abs(3. * yyim - yyi)) * (ti.abs(yyyi) - ti.abs(3. * yyim - yyi))
    beta2 = 0.25 * (ti.abs(yyyi) - ti.abs(yyim - 3. * yyi)) * (ti.abs(yyyi) - ti.abs(yyim - 3. * yyi))

    # NOTE(cmo): Linear weights
    gamma1 = -0.5 * (pos - nodeip) 
    gamma2 = 0.5 * (pos - nodeim) 

    # NOTE(cmo): Non-linear weights
    alpha1 = gamma1 / (WENO_EPS + beta1) ** 1.5
    alpha2 = gamma2 / (WENO_EPS + beta2) ** 1.5

    omega1 = alpha1 / (alpha1 + alpha2)
    omega2 = alpha2 / (alpha1 + alpha2)
    return omega1 * q1 + omega2 * q2

@ti.func
def weno3(x, xim, xi, xip, yim, yi, yip):
    him = xi - xim
    hi = xip - xi

    # NOTE(cmo): Quadratics over substencils
    q1 = (-yim * (x - xi) + yi * (x - xim)) / him
    q2 = (-yi * (x - xip) + yip * (x - xi)) / hi

    yyim = yi - yim
    yyi = yip - yi
    yyyi = yip - yim

    # NOTE(cmo): Smoothness indicators
    beta1 = 0.25 * (ti.abs(yyyi) - ti.abs(3. * yyim - yyi)) * (ti.abs(yyyi) - ti.abs(3. * yyim - yyi))
    beta2 = 0.25 * (ti.abs(yyyi) - ti.abs(yyim - 3. * yyi)) * (ti.abs(yyyi) - ti.abs(yyim - 3. * yyi))

    # NOTE(cmo): Linear weights
    gamma1 = - (x - xip) / (xip - xim)
    gamma2 = (x - xim) / (xip - xim)

    # NOTE(cmo): Non-linear weights
    alpha1 = gamma1 / (WENO_EPS + beta1) ** 1.5
    alpha2 = gamma2 / (WENO_EPS + beta2) ** 1.5

    omega1 = alpha1 / (alpha1 + alpha2)
    omega2 = alpha2 / (alpha1 + alpha2)
    return omega1 * q1 + omega2 * q2


# =================================================================================== #
#                                 WENO4 inteploation                                  #
# =================================================================================== #
# reference: G. Janet et al. A novel fourth-order WENO interpolation technique: A possible new tool designed for radiative transfer. Astronomy & Astrophysics. 2021 1-16.
@ti.func
def weno4_mid_uniform_grid(yim, yi, yip, yipp):
    # NOTE(cmo): Quadratics over substencils
    q2 = -0.125 * yim + 0.75 * yi + 0.375 * yip
    q3 = 0.375 * yi + 0.75 * yip - 0.125 * yipp

    # NOTE(cmo): Smoothness indicators
    beta2 = 4. * (ti.abs(0.5 * yim - 1.5 * yi + 1.5 * yip + 0.5 * yipp) - ti.abs(1.5 * yim - 3.5 * yi + 1.5 * yip - 0.5 * yipp)) ** 2
    beta3 = 4. * (ti.abs(-0.5 * yim + 2.5 * yi - 3.5 * yip + 1.5 * yipp) - ti.abs(0.5 * yim - 1.5 * yi + 1.5 * yip + 0.5 * yipp)) ** 2

    # NOTE(cmo): Non-linear weights
    alpha2 = 0.5 / (WENO_EPS + beta2)
    alpha3 = 0.5 / (WENO_EPS + beta3)

    omega2 = alpha2 / (alpha2 + alpha3)
    omega3 = alpha3 / (alpha2 + alpha3)
    return omega2 * q2 + omega3 * q3

@ti.func
def weno4_uniform_grid(nodeim, nodeic, nodeip, nodeipp, pos, yim, yi, yip, yipp):
    # NOTE(cmo): Quadratics over substencils
    q2 = 0.5 * yim * ((pos - nodeic) * (pos - nodeip)) - yi * ((pos - nodeim) * (pos - nodeip)) + 0.5 * yip * ((pos - nodeim) * (pos - nodeic))
    q3 = 0.5 * yi * ((pos - nodeip) * (pos - nodeipp)) - yip * ((pos - nodeic) * (pos - nodeipp)) + 0.5 * yipp * ((pos - nodeic) * (pos - nodeip))

    # NOTE(cmo): Smoothness indicators
    beta2 = 4. * (ti.abs(0.5 * yim - 1.5 * yi + 1.5 * yip + 0.5 * yipp) - ti.abs(1.5 * yim - 3.5 * yi + 1.5 * yip - 0.5 * yipp)) ** 2
    beta3 = 4. * (ti.abs(-0.5 * yim + 2.5 * yi - 3.5 * yip + 1.5 * yipp) - ti.abs(0.5 * yim - 1.5 * yi + 1.5 * yip + 0.5 * yipp)) ** 2

    # NOTE(cmo): Linear weights
    gamma2 = -1./3. * (pos - nodeipp) 
    gamma3 = 1./3. * (pos - nodeim)

    # NOTE(cmo): Non-linear weights
    alpha2 = gamma2 / (WENO_EPS + beta2)
    alpha3 = gamma3 / (WENO_EPS + beta3)

    omega2 = alpha2 / (alpha2 + alpha3)
    omega3 = alpha3 / (alpha2 + alpha3)
    return omega2 * q2 + omega3 * q3

@ti.func
def weno4(x, xim, xi, xip, xipp, yim, yi, yip, yipp):
    him = xi - xim
    hi = xip - xi
    hip = xipp - xip

    # NOTE(cmo): Quadratics over substencils
    q2 = yim * ((x - xi) * (x - xip)) / (him * (him + hi)) - yi * ((x - xim) * (x - xip)) / (him * hi) + yip * ((x - xim) * (x - xi)) / ((him + hi) * hi)
    q3 = yi * ((x - xip) * (x - xipp)) / (hi * (hi + hip)) - yip * ((x - xi) * (x - xipp)) / (hi * hip) + yipp * ((x - xi) * (x - xip)) / ((hi + hip) * hip)
    H = him + hi + hip
    yyim = -((2*him + hi)*H + him*(him + hi)) / (him*(him + hi)*H) * yim + ((him + hi)*H) / (him*hi*(hi + hip)) * yi - (him*H) / ((him + hi)*hi*hip) * yip + (him*(him + hi)) / ((hi + hip)*hip*H) * yipp
    yyi = -(hi*(hi + hip)) / (him*(him + hi)*H) * yim + (hi*(hi + hip) - him*(2*hi + hip)) / (him*hi*(hi + hip)) * yi + (him*(hi + hip)) / ((him + hi)*hi*hip) * yip - (him*hi) / ((hi + hip)*hip*H) * yipp
    yyip = (hi*hip) / (him*(him + hi)*H) * yim - (hip*(him + hi)) / (him*hi*(hi + hip)) * yi + ((him + 2*hi)*hip - (him + hi)*hi) / ((him + hi)*hi*hip) * yip + ((him + hi)*hi) / ((hi + hip)*hip*H) * yipp
    yyipp = -((hi + hip)*hip) / (him*(him + hi)*H) * yim + (hip*H) / (him*hi*(hi + hip)) * yi - ((hi + hip) * H) / ((him + hi)*hi*hip) * yip + ((2*hip + hi)*H + hip*(hi + hip)) / ((hi + hip)*hip*H) * yipp

    # NOTE(cmo): Smoothness indicators
    beta2 = (hi + hip)**2 * (abs(yyip - yyi) / hi - abs(yyi - yyim) / him)**2
    beta3 = (him + hi)**2 * (abs(yyipp - yyip) / hip - abs(yyip - yyi) / hi)**2

    # NOTE(cmo): Linear weights
    gamma2 = -(x - xipp) / (xipp - xim)
    gamma3 = (x - xim) / (xipp - xim)

    # NOTE(cmo): Non-linear weights
    alpha2 = gamma2 / (WENO_EPS + beta2)
    alpha3 = gamma3 / (WENO_EPS + beta3)

    omega2 = alpha2 / (alpha2 + alpha3)
    omega3 = alpha3 / (alpha2 + alpha3)
    return omega2 * q2 + omega3 * q3


# =================================================================================== #
#                                 WENO5 Intepolation                                  #
# =================================================================================== #
# reference: Jiang G S, Shu C W. Efficient implementation of weighted ENO schemes. J Comput Phys, 1996, 126: 202


@ti.func
def nonlinear_weighted(f0, f1, f2, f3, f4):
    beta_1 = (1./3. * (4.*f0**2 - 19.*f0*f1 + 25.*f1**2 + 11.*f0*f2 - 31.*f1*f2 + 10.*f2**2))
    beta_2 = (1./3. * (4.*f1**2 - 13.*f1*f2 + 13.*f2**2 + 5.*f1*f3 - 13.*f2*f3 + 4.*f3**2))
    beta_3 = (1./3. * (10.*f2**2 - 31.*f2*f3 + 25.*f3**2 + 11.*f2*f4 - 19.*f3*f4 + 4.*f4**2))
    w_til_1 = 0.1 / ((WENO_EPS + beta_1) * (WENO_EPS + beta_1))
    w_til_2 = 0.6 / ((WENO_EPS + beta_2) * (WENO_EPS + beta_2))
    w_til_3 = 0.3 / ((WENO_EPS + beta_3) * (WENO_EPS + beta_3))
    w_til = w_til_1 + w_til_2 + w_til_3
    w1 = w_til_1/w_til
    w2 = w_til_2/w_til
    w3 = w_til_3/w_til
    sa = 2./6. * f0 - 7./6. * f1 + 11./6. * f2
    sb = -1./6. * f1 + 5./6. * f2 +  2./6. * f3
    sc = 1./3. * f2 + 5./6. * f3 -  1./6. * f4
    return w1*sa+w2*sb+w3*sc

@ti.func
def weno5_js(beta1, beta2, beta3):
    # calculates the non-linear weights for each stencil that are determined by their smoothness indicators
    w_til_1 = 0.1 / ((WENO_EPS + beta1) * (WENO_EPS + beta1))
    w_til_2 = 0.6 / ((WENO_EPS + beta2) * (WENO_EPS + beta2))
    w_til_3 = 0.3 / ((WENO_EPS + beta3) * (WENO_EPS + beta3))
    return w_til_1, w_til_2, w_til_3

# reference: Borges, Rafael, et al. "An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws." JCP 227.6 (2008): 3191-3211.
@ti.func
def weno5_z(beta1, beta2, beta3):
    # Constants
    tau5 = ti.abs(beta1 - beta3)

    # calculates the non-linear weights for each stencil that are determined by their smoothness indicators
    w_til_1 = 0.1 * (1 + tau5 / (WENO_EPS + beta1))
    w_til_2 = 0.6 * (1 + tau5 / (WENO_EPS + beta2))
    w_til_3 = 0.3 * (1 + tau5 / (WENO_EPS + beta3))
    return w_til_1, w_til_2, w_til_3

# reference: Henrick, Andrew K., Tariq D. Aslam, and Joseph M. Powers. "Mapped weighted essentially non-oscillatory schemes: achieving optimal order near critical points." JCP 207.2 (2005): 542-567.
@ti.func
def weno5_m(beta1, beta2, beta3):
    # Constants
    d1n = 0.1; d2n = 0.6; d3n = 0.3

    # calculates the non-linear weights for each stencil that are determined by their smoothness indicators
    w_til_1 = d1n / ((WENO_EPS + beta1) * (WENO_EPS + beta1))
    w_til_2 = d2n / ((WENO_EPS + beta2) * (WENO_EPS + beta2))
    w_til_3 = d3n / ((WENO_EPS + beta3) * (WENO_EPS + beta3))
    w_til = w_til_1 + w_til_2 + w_til_3

    # ENO stencils weigths
    w1n = w_til_1 / w_til
    w2n = w_til_2 / w_til
    w3n = w_til_3 / w_til

    # Mapping functions
    g1n = w1n * (d1n + d1n * d1n - 3 * d1n * w1n + w1n * w1n) / (d1n * d1n + w1n * (1 - 2 * d1n))
    g2n = w2n * (d2n + d2n * d2n - 3 * d2n * w2n + w2n * w2n) / (d2n * d2n + w2n * (1 - 2 * d2n))
    g3n = w3n * (d3n + d3n * d3n - 3 * d3n * w3n + w3n * w3n) / (d3n * d3n + w3n * (1 - 2 * d3n))
    
    return g1n, g2n, g3n

@ti.func
def weno5_mid_uniform_grid(f0, f1, f2, f3, f4):
    # calculates the smoothness indicators
    beta_1 = (1./3. * (4. * f0 * f0 - 19. * f0 * f1 + 25. * f1 * f1 + 11. * f0 * f2 - 31. * f1 * f2 + 10. * f2 * f2))
    beta_2 = (1./3. * (4. * f1 * f1 - 13. * f1 * f2 + 13. * f2 * f2 + 5. * f1 * f3 - 13. * f2 * f3 + 4. * f3 * f3))
    beta_3 = (1./3. * (10. * f2 * f2 - 31. * f2 * f3 + 25. * f3 * f3 + 11. * f2 * f4 - 19. * f3 * f4 + 4. * f4 * f4))

    # calculates the non-linear weights for each stencil that are determined by their smoothness indicators
    w_til_1, w_til_2, w_til_3 = weno5_js(beta_1, beta_2, beta_3)
    w_til = w_til_1 + w_til_2 + w_til_3

    # Modified weigths
    w1 = w_til_1 / w_til
    w2 = w_til_2 / w_til
    w3 = w_til_3 / w_til

    sa = 1./3. * f0 - 7./6. * f1 + 11./6. * f2
    sb = -1./6. * f1 + 5./6. * f2 +  1./3. * f3
    sc = 1./3. * f0 - 7./6. * f1 + 11./6. * f2
    return w1 * sa + w2 * sb + w3 * sc

# reference: Jiang G S, Peng. D. P. Weighted ENO Schemes for Hamilton-Jacobi Equations. SIAM J. Sci. Comput., 2000, 21(6): 2126-2143
@ti.func
def weno5HJcommon(a, b, c, d):
    numer = -a + 7. * b + 7. * c - d
    return numer / 12.

@ti.func
def weno5HJ(a, b, c, d):
    beta_1 = 13 * (a - b) * (a - b) + 3 * (a - 3 * b) * (a - 3 * b)
    beta_2 = 13 * (b - c) * (b - c) + 3 * (b + c) * (b + c)
    beta_3 = 13 * (c - d) * (c - d) + 3 * (3 * c - d) * (3 * c - d)

    w_til_1, w_til_2, w_til_3 = weno5_js(beta_1, beta_2, beta_3)
    w_til = w_til_1 + w_til_2 + w_til_3
    w1 = w_til_1 / w_til
    w3 = w_til_3 / w_til
    return 1./3. * w1 * (a - 2 * b + c) + 1./6. * (w3 - 0.5) * (b - 2 * c + d)


# =================================================================================== #
#                            WENO/ENO inteploation numpy                              #
# =================================================================================== #
import numpy as np
import numpy.polynomial.polynomial as poly
def weighted_essentially_non_oscillatory(eno_order: int, values: np.ndarray, spacing: float, boundary_condition):
    """Implements an upwind weighted essentially non-oscillatory (WENO) scheme for first derivative approximation.

    Args:
        eno_order: The order of the underlying essentially non-oscillatory (ENO) scheme; the resulting WENO scheme is
            `(2 * eno_order - 1)`th-order accurate.
        values: 1-dimensional array of function values assumed to be evaluated at a uniform grid in the domain.
        spacing: Grid spacing of the `values`.
        boundary_condition: A function used to pad `values` to implement a boundary condition (e.g., periodic).

    Returns:
        A tuple of arrays `(left_derivatives, right_derivatives)` each the same shape as `values` which contain,
        respectively, left and right approximations of the first derivative at the grid points of `values`.
    """
    if eno_order < 1:
        raise ValueError(f"`eno_order` must be at least 1; got {eno_order}.")

    values = boundary_condition(values, eno_order)
    diffs = (values[1:] - values[:-1]) / spacing

    if eno_order == 1:
        return (diffs[:-1], diffs[1:])

    substencil_approximations = tuple(
        _unrolled_correlate(diffs[i:len(diffs) - eno_order + i], c)
        for (i, c) in enumerate(_diff_coefficients(eno_order)))
    diffs2 = diffs[1:] - diffs[:-1]
    smoothness_indicators = [
        sum(
            _unrolled_correlate(diffs2[i + j:len(diffs2) - eno_order + i + 1], L[j:, j])**2
            for j in range(eno_order - 1))
        for (i, L) in enumerate(np.linalg.cholesky(_smoothness_indicator_quad_form(eno_order)))
    ]
    left_and_right_unnormalized_weights = [[
        c / (s[i:len(s) + i - 1] + WENO_EPS)**2 for (c, s) in zip(coefficients, smoothness_indicators)
    ] for (i, coefficients) in enumerate(_substencil_coefficients(eno_order))]
    return tuple(
        sum(w * a for (w, a) in zip(unnormalized_weights, substencil_approximations[i:eno_order + i])) /
        sum(unnormalized_weights) for (i, unnormalized_weights) in enumerate(left_and_right_unnormalized_weights))


def essentially_non_oscillatory(order: int, values: np.ndarray, spacing: float, boundary_condition):
    """Implements an upwind essentially non-oscillatory (ENO) scheme for first derivative approximation.

    Args:
        order: The desired order of accuracy for the ENO scheme.
        values: 1-dimensional array of function values assumed to be evaluated at a uniform grid in the domain.
        spacing: Grid spacing of the `values`.
        boundary_condition: A function used to pad `values` to implement a boundary condition (e.g., periodic).

    Returns:
        A tuple of arrays `(left_derivatives, right_derivatives)` each the same shape as `values` which contain,
        respectively, left and right approximations of the first derivative at the grid points of `values`.
    """
    if order < 1:
        raise ValueError(f"`order` must be at least 1; got {order}.")

    values = boundary_condition(values, order)
    diffs = (values[1:] - values[:-1]) / spacing

    if order == 1:
        return (diffs[:-1], diffs[1:])

    substencil_approximations = tuple(
        _unrolled_correlate(diffs[i:len(diffs) - order + i], c) for (i, c) in enumerate(_diff_coefficients(order)))

    undivided_differences = []
    for i in range(2, order):
        diffs = diffs[1:] - diffs[:-1]
        undivided_differences.append(diffs[order - i:i - order])

    abs_diffs = np.abs(diffs[1:] - diffs[:-1])
    stencil_indices = abs_diffs[1:] < abs_diffs[:-1]
    for diffs in reversed(undivided_differences):
        abs_diffs = np.abs(diffs)
        stencil_indices = np.where(abs_diffs[1:] < abs_diffs[:-1], stencil_indices[1:] + 1, stencil_indices[:-1])

    return (np.select([stencil_indices[:-1] == i for i in range(order - 1)], substencil_approximations[:-2],
                       substencil_approximations[-2]),
            np.select([stencil_indices[1:] == i for i in range(order - 1)], substencil_approximations[1:-1],
                       substencil_approximations[-1]))

def _unrolled_correlate(a: np.ndarray, v: np.ndarray):
    """An unrolled equivalent of `np.correlate`."""
    return sum(a[i:len(a) - len(v) + i + 1] * x for (i, x) in enumerate(v))


def _substencils(k: int):
    """Returns the `k + 1` subranges of length `k + 1` from the full stencil range `[-k, k + 1)`."""
    return np.arange(k + 1) + np.arange(k + 1)[:, np.newaxis] - k


def _spread_substencil_values(x: np.ndarray):
    """Offsets each successive row of a matrix `x` by one additional column."""
    return np.reshape(np.reshape(np.pad(x, ((0, 0), (0, x.shape[0]))), -1)[:-x.shape[0]], (x.shape[0], -1))


def _diff_coefficients(k: int = None, stencil: np.ndarray = None):
    """Returns first derivative approximation finite difference coefficients for function value first differences."""
    if k is None:
        if stencil is None:
            raise ValueError("One of `k` or `stencil` must be provided.")
        k = stencil.shape[-1] - 1
    else:
        if stencil is None:
            stencil = _substencils(k)
        elif k != stencil.shape[-1] - 1:
            raise ValueError("`k` must match `stencil.shape[-1] - 1` if both arguments are provided; got "
                             f"{(k, stencil.shape[-1] - 1)}.")
    system = np.diff(
        poly.polyvander(stencil, k), axis=-2
    )[..., 1:].swapaxes(-1, -2)
    # NumPy 2.x interprets a two-dimensional right-hand side as a stack of
    # matrices, not a broadcast vector.  Materialize one derivative target
    # vector per substencil so the batched solve is unambiguous.
    target = np.broadcast_to(
        np.eye(k)[0], system.shape[:-2] + (k,)
    )[..., np.newaxis]
    return np.linalg.solve(system, target)[..., 0]


def _substencil_coefficients(k: int):
    """Returns coefficients for combining substencil approximations to yield higher order left/right approximations."""
    left_coefficients = np.linalg.solve(_spread_substencil_values(_diff_coefficients(k))[:-1, :k].T, _diff_coefficients(stencil=np.arange(-k, k))[:k])
    return np.array([left_coefficients, left_coefficients[::-1]])


def _polyder_operator(k: int, d: int):
    """Returns a matrix `D` such that `D @ p == poly.polyder(p, d)` for polynomials `p` of degree `k`."""
    return np.concatenate([np.zeros((k + 1 - d, d)), np.diag(poly.polyder(np.ones(k + 1), d))], 1)


def _smoothness_indicator_quad_form(k: int):
    """Returns quadratic forms for computing substencil smoothness indicators as functions of second differences."""
    interp_poly_second_der = (poly.polyder(np.ones(k + 1), 2)[:, np.newaxis] *
                              np.linalg.inv(np.diff(poly.polyvander(_substencils(k)[1:], k), 2, axis=-2)[..., 2:]))

    quad_form = np.zeros((k, k - 1, k - 1))
    for m in range(k - 1):
        integrator_matrix = 1 / (np.arange(k - 1 - m) + np.arange(k - 1 - m)[:, np.newaxis] + 1)
        interp_poly_m_plus_2_der = _polyder_operator(k - 2, m) @ interp_poly_second_der
        quad_form += interp_poly_m_plus_2_der.swapaxes(-1, -2) @ integrator_matrix @ interp_poly_m_plus_2_der
    return quad_form
