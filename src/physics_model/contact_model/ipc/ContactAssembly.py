"""Reusable IPC chain-rule and sparse-scatter kernels for contact stencils.

Geometry returns a local contact gradient/Hessian.  Every discretization then
provides a linear map ``x_a = sum_I B[a, I] q_I``.  The common operation is

    g_I = sum_a B[a, I]^T g_a,
    H_IJ = sum_ab B[a, I]^T H_ab B[b, J].

The fixed four-site helpers cover PP/PE/PT/EE.  NURBS and MPM use the same
low-level weighted block scatter with a variable number of basis functions.
"""

from src.utils.MatrixFunction import diag_nd, symmetric_eigendecomposition_nd, psd_project_nd, psd_project_gershgorin_nd

import taichi as ti


@ti.func
def weighted_relative_vector4(weights, value0, value1, value2, value3):
    return weights[0] * value0 + weights[1] * value1 + weights[2] * value2 + weights[3] * value3


@ti.func
def pullback_relative_gradient4(weights, relative_gradient):
    dimension = ti.static(relative_gradient.n)
    local_gradient = ti.Vector.zero(float, 4 * dimension)
    for site in range(4):
        for component in range(dimension):
            local_gradient[site * dimension + component] = weights[site] * relative_gradient[component]
    return local_gradient


@ti.func
def pullback_relative_hessian4(weights, relative_hessian):
    dimension = ti.static(relative_hessian.n)
    local_hessian = ti.Matrix.zero(float, 4 * dimension, 4 * dimension)
    for site_i in range(4):
        for site_j in range(4):
            scale = weights[site_i] * weights[site_j]
            for row in range(dimension):
                for column in range(dimension):
                    local_hessian[
                        site_i * dimension + row,
                        site_j * dimension + column,
                    ] = (
                        scale * relative_hessian[row, column]
                    )
    return local_hessian


@ti.func
def compose_distance2_hessian(distance_gradient, distance_hessian, first, second):
    """Hessian of ``phi(distance2(x))`` from scalar derivatives."""
    return first * distance_hessian + second * distance_gradient.outer_product(distance_gradient)


@ti.func
def compose_mollified_hessian(
    potential,
    potential_gradient,
    potential_hessian,
    mollifier,
    mollifier_gradient,
    mollifier_hessian,
):
    """Product rule for an EE mollified contact potential ``m * phi``."""
    gradient = mollifier * potential_gradient + potential * mollifier_gradient
    hessian = mollifier * potential_hessian + potential * mollifier_hessian
    hessian += mollifier_gradient.outer_product(potential_gradient)
    hessian += potential_gradient.outer_product(mollifier_gradient)
    return gradient, hessian


@ti.func
def psd_project_cloth_tangent_6x6(matrix):
    """Fixed-size Gershgorin PSD majorizer for the ClothARAP 6x6 tangent."""
    symmetric = 0.5 * (matrix + matrix.transpose())
    lower_bound = 1.0e30
    for row in range(6):
        radius = 0.0
        for column in ti.static(range(6)):
            if column != row:
                radius += ti.abs(symmetric[row, column])
        lower_bound = ti.min(lower_bound, symmetric[row, row] - radius)
    shift = ti.max(-lower_bound, 0.0)
    return symmetric + shift * ti.Matrix.identity(ti.f64, 6)


@ti.func
def scatter_flat_gradient(global_gradient: ti.template(), dof_offset, value):
    """Atomically add one vector to a flat scalar global vector."""
    if ti.static(value.n <= 3):
        for component in ti.static(range(value.n)):
            ti.atomic_add(global_gradient[dof_offset + component], value[component])
    else:
        component = 0
        while component < value.n:
            ti.atomic_add(global_gradient[dof_offset + component], value[component])
            component += 1


@ti.func
def scatter_flat_weighted_gradient(global_gradient: ti.template(), dof_offset, weight, value):
    if ti.static(value.n <= 3):
        for component in ti.static(range(value.n)):
            ti.atomic_add(
                global_gradient[dof_offset + component],
                weight * value[component],
            )
    else:
        component = 0
        while component < value.n:
            ti.atomic_add(
                global_gradient[dof_offset + component],
                weight * value[component],
            )
            component += 1


@ti.func
def scatter_vector_weighted_gradient(global_gradient: ti.template(), node, weight, value):
    if ti.static(value.n <= 3):
        for component in ti.static(range(value.n)):
            ti.atomic_add(global_gradient[node][component], weight * value[component])
    else:
        component = 0
        while component < value.n:
            ti.atomic_add(global_gradient[node][component], weight * value[component])
            component += 1


@ti.func
def scatter_hash_block(matrix: ti.template(), block_i, block_j, weight_i, weight_j, block):
    """Scatter a weighted dense block into ``BuildTriplet``-style sinks."""
    matrix.add_block_entry(block_i, block_j, weight_i * weight_j * block)


@ti.func
def scatter_hash_scalar(matrix: ti.template(), row, column, value, dimension: ti.template()):
    row_component = row - (row // dimension) * dimension
    column_component = column - (column // dimension) * dimension
    matrix.add_scalar_entry(row, column, value, row_component, column_component)
