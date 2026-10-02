"""Reusable IPC chain-rule and sparse-scatter kernels for contact stencils.

Geometry returns a local contact gradient/Hessian.  Every discretization then
provides a linear map ``x_a = sum_I B[a, I] q_I``.  The common operation is

    g_I = sum_a B[a, I]^T g_a,
    H_IJ = sum_ab B[a, I]^T H_ab B[b, J].

The fixed four-site helpers cover PP/PE/PT/EE.  NURBS and MPM use the same
low-level weighted block scatter with a variable number of basis functions.
"""

import numpy as np
import taichi as ti


def pullback_dense(geometry_gradient, geometry_hessian, interpolation):
    """NumPy oracle for ``B.T @ g`` and ``B.T @ H @ B``.

    This is also useful to FEM/IGA callers which assemble contact blocks on the
    host.  ``interpolation`` is the dense geometric-to-generalized Jacobian B.
    """
    gradient = np.asarray(geometry_gradient, dtype=np.float64)
    hessian = np.asarray(geometry_hessian, dtype=np.float64)
    interpolation = np.asarray(interpolation, dtype=np.float64)
    return interpolation.T @ gradient, interpolation.T @ hessian @ interpolation


def project_to_psd_dense(matrix, method="clamp", tolerance=0.0):
    """Project a complete local contact Hessian on the host.

    Projection must happen after all stencil blocks have been assembled; doing
    it independently to CC/CM/MM blocks is not equivalent. ``method``
    accepts ``NONE``, ``CLAMP``, and ``ABS``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("contact Hessian must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("contact Hessian must be finite")
    key = str(method).strip().replace("-", "_").lower()
    if key in ("none", "off", "false"):
        return 0.5 * (matrix + matrix.T)
    if key not in ("clamp", "abs", "absolute"):
        raise ValueError("PSD projection method must be 'none', 'clamp', or 'abs'")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("PSD projection tolerance must be finite and non-negative")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if key == "clamp":
        eigenvalues = np.maximum(eigenvalues, tolerance)
    else:
        eigenvalues = np.maximum(np.abs(eigenvalues), tolerance)
    projected = (eigenvectors * eigenvalues) @ eigenvectors.T
    return 0.5 * (projected + projected.T)


def dense_local_triplets(dofs, local_matrix, zero_tolerance=0.0):
    """Return COO arrays for one already-composed local contact matrix."""
    dofs = np.asarray(dofs, dtype=np.int64).reshape(-1)
    local_matrix = np.asarray(local_matrix, dtype=np.float64)
    if local_matrix.shape != (dofs.size, dofs.size):
        raise ValueError("local_matrix shape must match the number of dofs")
    if not np.isfinite(local_matrix).all():
        raise ValueError("local_matrix must be finite")
    tolerance = float(zero_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative")
    local_rows, local_cols = np.nonzero(np.abs(local_matrix) > tolerance)
    return (
        dofs[local_rows],
        dofs[local_cols],
        local_matrix[local_rows, local_cols],
    )


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
def diag_nd(vector):
    matrix = ti.Matrix.zero(ti.f64, vector.n, vector.n)
    for dimension in range(vector.n):
        matrix[dimension, dimension] = vector[dimension]
    return matrix


@ti.func
def symmetric_eigendecomposition_nd(matrix):
    """Device-side eigendecomposition of a fixed-size symmetric matrix.

    The input is symmetrized, matching IPC's local ``make_pd`` operation.
    The 2x2 case uses a scale-stable closed form.  Every larger fixed block
    uses cyclic Jacobi rotations and therefore never leaves the CUDA kernel.
    Taichi 1.7's ``ti.sym_eig`` is intentionally not used for 3x3 blocks: on
    rank-deficient matrices with a repeated positive eigenvalue (the exact
    spectrum of sliding Coulomb friction), it can manufacture an eigenvalue
    many orders of magnitude larger than the input.
    """
    symmetric = 0.5 * (matrix + matrix.transpose())
    eigenvalues = ti.Vector.zero(ti.f64, matrix.n)
    eigenvectors = ti.Matrix.identity(ti.f64, matrix.n)
    if ti.static(matrix.n == 2):
        a = symmetric[0, 0]
        b = symmetric[0, 1]
        d = symmetric[1, 1]
        midpoint = 0.5 * (a + d)
        half_difference = 0.5 * (a - d)

        # Normalize the two coordinates of the spectral radius before taking
        # their norm.  Besides avoiding overflow/underflow, this is homogeneous
        # in the matrix entries: scaling the whole matrix cannot change the
        # eigenvectors or trigger an absolute-epsilon branch.
        spectral_scale = ti.max(ti.abs(half_difference), ti.abs(b))
        radius = 0.0
        cosine_twice = 1.0
        sine_twice = 0.0
        if spectral_scale > 0.0:
            scaled_difference = half_difference / spectral_scale
            scaled_off_diagonal = b / spectral_scale
            scaled_radius = ti.sqrt(scaled_difference * scaled_difference + scaled_off_diagonal * scaled_off_diagonal)
            radius = spectral_scale * scaled_radius
            cosine_twice = scaled_difference / scaled_radius
            sine_twice = scaled_off_diagonal / scaled_radius

        lambda1 = midpoint + radius
        lambda2 = midpoint - radius
        cosine = ti.sqrt(ti.max(0.5 * (1.0 + cosine_twice), 0.0))
        sine = ti.sqrt(ti.max(0.5 * (1.0 - cosine_twice), 0.0))
        if sine_twice < 0.0:
            sine = -sine
        vector1 = ti.Vector([cosine, sine])
        vector2 = ti.Vector([-vector1[1], vector1[0]])
        eigenvalues = ti.Vector([lambda1, lambda2])
        eigenvectors = ti.Matrix.cols([vector1, vector2])
    else:
        diagonalized = symmetric
        # Twelve cyclic sweeps are normally sufficient for these small
        # self-adjoint blocks; twenty-four gives a conservative f64 margin
        # for clustered spectra without any CPU convergence check.  These
        # loops intentionally remain serial because every Jacobi pivot reads
        # the result of the previous pivot.  ``while`` prevents Taichi from
        # statically expanding blocks larger than 3x3 while preserving that
        # exact sweep/pivot order on the device.
        sweep = 0
        while sweep < 24:
            p = 0
            while p < matrix.n:
                q = p + 1
                while q < matrix.n:
                    app = diagonalized[p, p]
                    aqq = diagonalized[q, q]
                    apq = diagonalized[p, q]
                    pivot_scale = ti.max(ti.abs(app) + ti.abs(aqq), 1.0e-30)
                    if ti.abs(apq) > 1.0e-14 * pivot_scale:
                        tau = (aqq - app) / (2.0 * apq)
                        sign_tau = 1.0
                        if tau < 0.0:
                            sign_tau = -1.0
                        tangent = sign_tau / (ti.abs(tau) + ti.sqrt(1.0 + tau * tau))
                        cosine = 1.0 / ti.sqrt(1.0 + tangent * tangent)
                        sine = tangent * cosine

                        k = 0
                        while k < matrix.n:
                            if k != p and k != q:
                                akp = diagonalized[k, p]
                                akq = diagonalized[k, q]
                                rotated_kp = cosine * akp - sine * akq
                                rotated_kq = sine * akp + cosine * akq
                                diagonalized[k, p] = rotated_kp
                                diagonalized[p, k] = rotated_kp
                                diagonalized[k, q] = rotated_kq
                                diagonalized[q, k] = rotated_kq
                            k += 1

                        diagonalized[p, p] = app - tangent * apq
                        diagonalized[q, q] = aqq + tangent * apq
                        diagonalized[p, q] = 0.0
                        diagonalized[q, p] = 0.0

                        k = 0
                        while k < matrix.n:
                            vkp = eigenvectors[k, p]
                            vkq = eigenvectors[k, q]
                            eigenvectors[k, p] = cosine * vkp - sine * vkq
                            eigenvectors[k, q] = sine * vkp + cosine * vkq
                            k += 1
                    q += 1
                p += 1
            sweep += 1

        eigenmode = 0
        while eigenmode < matrix.n:
            eigenvalues[eigenmode] = diagonalized[eigenmode, eigenmode]
            eigenmode += 1
    return eigenvalues, eigenvectors


@ti.func
def psd_project_nd(matrix):
    """Spectrally clamp a fixed-size symmetric matrix on the device.

    The returned matrix is ``V max(D, 0) V.T`` for complete local blocks.
    """
    eigenvalues, eigenvectors = symmetric_eigendecomposition_nd(matrix)
    projected = ti.Matrix.zero(ti.f64, matrix.n, matrix.n)
    row = 0
    while row < matrix.n:
        column = 0
        while column < matrix.n:
            value = 0.0
            eigenmode = 0
            while eigenmode < matrix.n:
                eigenvalue = ti.max(eigenvalues[eigenmode], 0.0)
                value += eigenvalue * eigenvectors[row, eigenmode] * eigenvectors[column, eigenmode]
                eigenmode += 1
            projected[row, column] = value
            column += 1
        row += 1
    projected = 0.5 * (projected + projected.transpose())
    return projected


@ti.func
def psd_project_gershgorin_nd(matrix):
    """Cheap device PSD majorizer for large local contact blocks.

    The symmetric matrix is kept off diagonal and receives one global diagonal
    shift that dominates all Gershgorin radii.  The result is symmetric
    diagonally dominant with non-negative diagonal, hence PSD, without the
    cubic-size Jacobi eigendecomposition used by ``psd_project_nd``.  This is the same
    device-local projection contract needed by FEM/IGA-style block assembly,
    but keeps CUDA compilation bounded for 12x12 PT/EE contact blocks.
    """
    # Keep the dimensions compile-time constants.  A dynamic ``while`` over
    # ``matrix.n`` makes CUDA instantiate the loop-control path for every
    # caller; IGA's block assembly uses the same fixed-size device pattern.
    symmetric = 0.5 * (matrix + matrix.transpose())
    # Use one global diagonal shift instead of constructing a second matrix
    # element-by-element.  It is the same Gershgorin PSD majorizer, but keeps
    # the generated CUDA IR small for the 6x6 cloth material block.
    lower_bound = 1.0e30
    for row in ti.static(range(matrix.n)):
        radius = 0.0
        for column in ti.static(range(matrix.n)):
            if ti.static(column != row):
                radius += ti.abs(symmetric[row, column])
        lower_bound = ti.min(lower_bound, symmetric[row, row] - radius)
    shift = ti.max(-lower_bound, 0.0)
    return symmetric + shift * ti.Matrix.identity(ti.f64, matrix.n)


@ti.func
def psd_project_cloth_tangent_6x6(matrix):
    """Fixed-size Gershgorin PSD majorizer for the ClothARAP 6x6 tangent."""
    symmetric = 0.5 * (matrix + matrix.transpose())
    lower_bound = 1.0e30
    for row in ti.static(range(6)):
        radius = 0.0
        for column in ti.static(range(6)):
            if ti.static(column != row):
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
