"""Dense NumPy assembly oracles for tests only."""

import numpy as np


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
