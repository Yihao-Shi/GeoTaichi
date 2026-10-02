"""Edge--edge parallelism mollifier shared by IPC discretizations.

The implementation follows the IPC polynomial ``m(e)`` with
``e = ||(a1-a0) x (b1-b0)||^2``.  It intentionally owns the scalar value and
its complete 12-DOF derivatives so FEM, affine-body, and future surface
couplings use one convention.
"""

import taichi as ti


@ti.func
def edge_edge_mollifier_threshold(a0_rest, a1_rest, b0_rest, b1_rest):
    edge_a = a1_rest - a0_rest
    edge_b = b1_rest - b0_rest
    return 1.0e-3 * edge_a.dot(edge_a) * edge_b.dot(edge_b)


@ti.func
def edge_edge_cross_norm2(a0, a1, b0, b1):
    return (a1 - a0).cross(b1 - b0).norm_sqr()


@ti.func
def _edge_edge_cross_norm2_grad_hess(a0, a1, b0, b1):
    """Gradient/Hessian of the cross-product norm in endpoint coordinates."""
    edge_a = a1 - a0
    edge_b = b1 - b0
    aa = edge_a.dot(edge_a)
    ab = edge_a.dot(edge_b)
    bb = edge_b.dot(edge_b)
    identity = ti.Matrix.identity(float, 3)

    grad_a = 2.0 * (bb * edge_a - ab * edge_b)
    grad_b = 2.0 * (aa * edge_b - ab * edge_a)
    hess_aa = 2.0 * (bb * identity - edge_b.outer_product(edge_b))
    hess_bb = 2.0 * (aa * identity - edge_a.outer_product(edge_a))
    hess_ab = 2.0 * (2.0 * edge_a.outer_product(edge_b) - edge_b.outer_product(edge_a) - ab * identity)

    edge_gradient = ti.Vector.zero(float, 6)
    edge_hessian = ti.Matrix.zero(float, 6, 6)
    for i in ti.static(range(3)):
        edge_gradient[i] = grad_a[i]
        edge_gradient[3 + i] = grad_b[i]
        for j in ti.static(range(3)):
            edge_hessian[i, j] = hess_aa[i, j]
            edge_hessian[i, 3 + j] = hess_ab[i, j]
            edge_hessian[3 + i, j] = hess_ab[j, i]
            edge_hessian[3 + i, 3 + j] = hess_bb[i, j]

    # [edge_a, edge_b] = B [a0, a1, b0, b1].
    endpoint_sign = ti.Vector([-1.0, 1.0, -1.0, 1.0])
    endpoint_edge = ti.Vector([0, 0, 1, 1])
    gradient = ti.Vector.zero(float, 12)
    hessian = ti.Matrix.zero(float, 12, 12)
    for site_i in range(4):
        edge_i = endpoint_edge[site_i]
        sign_i = endpoint_sign[site_i]
        for component_i in range(3):
            local_i = 3 * site_i + component_i
            edge_dof_i = 3 * edge_i + component_i
            gradient[local_i] = sign_i * edge_gradient[edge_dof_i]
            for site_j in range(4):
                edge_j = endpoint_edge[site_j]
                sign_j = endpoint_sign[site_j]
                for component_j in range(3):
                    local_j = 3 * site_j + component_j
                    edge_dof_j = 3 * edge_j + component_j
                    hessian[local_i, local_j] = sign_i * sign_j * edge_hessian[edge_dof_i, edge_dof_j]
    return gradient, hessian


@ti.func
def _edge_edge_cross_norm2_grad(a0, a1, b0, b1):
    edge_a = a1 - a0
    edge_b = b1 - b0
    grad_a = 2.0 * (edge_b.dot(edge_b) * edge_a - edge_a.dot(edge_b) * edge_b)
    grad_b = 2.0 * (edge_a.dot(edge_a) * edge_b - edge_a.dot(edge_b) * edge_a)
    gradient = ti.Vector.zero(float, 12)
    for component in ti.static(range(3)):
        gradient[component] = -grad_a[component]
        gradient[3 + component] = grad_a[component]
        gradient[6 + component] = -grad_b[component]
        gradient[9 + component] = grad_b[component]
    return gradient


@ti.func
def edge_edge_mollifier(a0, a1, b0, b1, eps_x):
    cross_norm2 = edge_edge_cross_norm2(a0, a1, b0, b1)
    mollifier = 1.0
    # A zero threshold comes only from a degenerate rest edge.  Treating it as
    # un-mollified avoids 0/0 while preserving the finite IPC branch.
    if eps_x > 0.0 and cross_norm2 < eps_x:
        ratio = cross_norm2 / eps_x
        mollifier = ratio * (2.0 - ratio)
    return mollifier


@ti.func
def edge_edge_mollifier_grad_hess(a0, a1, b0, b1, eps_x):
    cross_norm2 = edge_edge_cross_norm2(a0, a1, b0, b1)
    gradient = ti.Vector.zero(float, 12)
    hessian = ti.Matrix.zero(float, 12, 12)
    if eps_x > 0.0 and cross_norm2 < eps_x:
        cross_gradient, cross_hessian = _edge_edge_cross_norm2_grad_hess(a0, a1, b0, b1)
        first = 2.0 * (1.0 - cross_norm2 / eps_x) / eps_x
        second = -2.0 / (eps_x * eps_x)
        gradient = first * cross_gradient
        hessian = first * cross_hessian + second * cross_gradient.outer_product(cross_gradient)
    return gradient, hessian


@ti.func
def edge_edge_mollifier_grad(a0, a1, b0, b1, eps_x):
    cross_norm2 = edge_edge_cross_norm2(a0, a1, b0, b1)
    gradient = ti.Vector.zero(float, 12)
    if eps_x > 0.0 and cross_norm2 < eps_x:
        gradient = 2.0 * (1.0 - cross_norm2 / eps_x) / eps_x * _edge_edge_cross_norm2_grad(a0, a1, b0, b1)
    return gradient


# -----------------------------------------------------------------------------
# Scalar and shape-derivative API.
#
# C++ overloads ``edge_edge_mollifier`` for a scalar cross norm and for four
# endpoints.  Python cannot overload by signature, so the scalar functions use
# an explicit ``_scalar`` suffix while endpoint functions retain the historical
# GeoTaichi names above.


@ti.func
def edge_edge_cross_squarednorm(a0, a1, b0, b1):
    return edge_edge_cross_norm2(a0, a1, b0, b1)


@ti.func
def edge_edge_cross_squarednorm_gradient(a0, a1, b0, b1):
    gradient, _ = _edge_edge_cross_norm2_grad_hess(a0, a1, b0, b1)
    return gradient


@ti.func
def edge_edge_cross_squarednorm_hessian(a0, a1, b0, b1):
    _, hessian = _edge_edge_cross_norm2_grad_hess(a0, a1, b0, b1)
    return hessian


@ti.func
def edge_edge_mollifier_scalar_terms(cross_squarednorm, eps_x):
    """Return ``m``, ``dm/dx``, and ``d2m/dx2`` for the IPC EE mollifier."""
    value = 1.0
    gradient = 0.0
    hessian = 0.0
    if eps_x > 0.0 and cross_squarednorm < eps_x:
        inverse_eps = 1.0 / eps_x
        ratio = cross_squarednorm * inverse_eps
        value = ratio * (2.0 - ratio)
        gradient = 2.0 * inverse_eps * (1.0 - ratio)
        hessian = -2.0 * inverse_eps * inverse_eps
    return value, gradient, hessian


@ti.func
def edge_edge_mollifier_scalar(cross_squarednorm, eps_x):
    value, _, _ = edge_edge_mollifier_scalar_terms(cross_squarednorm, eps_x)
    return value


@ti.func
def edge_edge_mollifier_scalar_gradient(cross_squarednorm, eps_x):
    _, gradient, _ = edge_edge_mollifier_scalar_terms(cross_squarednorm, eps_x)
    return gradient


@ti.func
def edge_edge_mollifier_scalar_hessian(cross_squarednorm, eps_x):
    _, _, hessian = edge_edge_mollifier_scalar_terms(cross_squarednorm, eps_x)
    return hessian


@ti.func
def edge_edge_mollifier_derivative_wrt_eps_x(cross_squarednorm, eps_x):
    """Derivative of the scalar mollifier with respect to its threshold."""
    derivative = 0.0
    if eps_x > 0.0 and cross_squarednorm < eps_x:
        derivative = 2.0 * cross_squarednorm * (-eps_x + cross_squarednorm) / (eps_x * eps_x * eps_x)
    return derivative


@ti.func
def edge_edge_mollifier_gradient_derivative_wrt_eps_x(cross_squarednorm, eps_x):
    """Derivative of ``dm/dx`` with respect to the threshold ``eps_x``."""
    derivative = 0.0
    if eps_x > 0.0 and cross_squarednorm < eps_x:
        derivative = 2.0 * (-eps_x + 2.0 * cross_squarednorm) / (eps_x * eps_x * eps_x)
    return derivative


@ti.func
def edge_edge_mollifier_gradient(a0, a1, b0, b1, eps_x):
    return edge_edge_mollifier_grad(a0, a1, b0, b1, eps_x)


@ti.func
def edge_edge_mollifier_hessian(a0, a1, b0, b1, eps_x):
    _, hessian = edge_edge_mollifier_grad_hess(a0, a1, b0, b1, eps_x)
    return hessian


@ti.func
def edge_edge_mollifier_terms(a0, a1, b0, b1, eps_x):
    value = edge_edge_mollifier(a0, a1, b0, b1, eps_x)
    gradient, hessian = edge_edge_mollifier_grad_hess(a0, a1, b0, b1, eps_x)
    return value, gradient, hessian


@ti.func
def edge_edge_mollifier_threshold_gradient(a0_rest, a1_rest, b0_rest, b1_rest):
    """Gradient of ``1e-3 ||ea||^2 ||eb||^2`` in endpoint order."""
    edge_a = a1_rest - a0_rest
    edge_b = b1_rest - b0_rest
    edge_a2 = edge_a.dot(edge_a)
    edge_b2 = edge_b.dot(edge_b)
    gradient = ti.Vector.zero(float, 12)
    gradient_a = 2.0e-3 * edge_b2 * edge_a
    gradient_b = 2.0e-3 * edge_a2 * edge_b
    for component in ti.static(range(3)):
        gradient[component] = -gradient_a[component]
        gradient[3 + component] = gradient_a[component]
        gradient[6 + component] = -gradient_b[component]
        gradient[9 + component] = gradient_b[component]
    return gradient


@ti.func
def edge_edge_mollifier_gradient_wrt_x(
    a0_rest,
    a1_rest,
    b0_rest,
    b1_rest,
    a0,
    a1,
    b0,
    b1,
):
    """Derivative through the rest-position threshold.

    This contains the contribution from ``eps_x(rest)``. When
    differentiating a deformed position ``x + u`` with
    fixed ``u``, add the ordinary endpoint gradient separately.
    """
    eps_x = edge_edge_mollifier_threshold(a0_rest, a1_rest, b0_rest, b1_rest)
    cross_squarednorm = edge_edge_cross_squarednorm(a0, a1, b0, b1)
    gradient = ti.Vector.zero(float, 12)
    if eps_x > 0.0 and cross_squarednorm < eps_x:
        gradient = edge_edge_mollifier_derivative_wrt_eps_x(
            cross_squarednorm, eps_x
        ) * edge_edge_mollifier_threshold_gradient(a0_rest, a1_rest, b0_rest, b1_rest)
    return gradient


@ti.func
def edge_edge_mollifier_gradient_jacobian_wrt_x(
    a0_rest,
    a1_rest,
    b0_rest,
    b1_rest,
    a0,
    a1,
    b0,
    b1,
):
    """Specialized Jacobian for the EE shape derivative.

    It evaluates
    ``m_s,eps grad(eps) grad(s)^T + m_s Hessian(s)`` and is stored with
    rest-position DOFs on rows and current-gradient components on columns,
    and is not the ordinary endpoint mollifier Hessian (whose additional
    scalar-curvature term is ``m_ss grad(s) grad(s)^T``).
    """
    eps_x = edge_edge_mollifier_threshold(a0_rest, a1_rest, b0_rest, b1_rest)
    cross_squarednorm = edge_edge_cross_squarednorm(a0, a1, b0, b1)
    jacobian = ti.Matrix.zero(float, 12, 12)
    if eps_x > 0.0 and cross_squarednorm < eps_x:
        threshold_gradient = edge_edge_mollifier_threshold_gradient(a0_rest, a1_rest, b0_rest, b1_rest)
        cross_gradient, cross_hessian = _edge_edge_cross_norm2_grad_hess(a0, a1, b0, b1)
        jacobian = (
            edge_edge_mollifier_gradient_derivative_wrt_eps_x(cross_squarednorm, eps_x)
            * threshold_gradient.outer_product(cross_gradient)
            + edge_edge_mollifier_scalar_gradient(cross_squarednorm, eps_x) * cross_hessian
        )
    return jacobian
