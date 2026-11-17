"""Shared solver tolerances, scalar acceptance, and explicit host sparse solves."""

import math


def solve_csr_system(rhs, sparse_matrixes):
    import numpy as np
    import taichi as ti
    from scipy.sparse.linalg import spsolve

    if isinstance(rhs, ti.ScalarField):
        rhs = rhs.to_numpy()
    sparse_matrixes = sparse_matrixes.tocsr()
    sparse_matrixes.data[np.abs(sparse_matrixes.data) < 2.2204460492503131e-15] = 0
    sparse_matrixes.eliminate_zeros()
    non_zero_rows = sparse_matrixes.getnnz(axis=1) != 0
    non_zero_cols = sparse_matrixes.getnnz(axis=0) != 0
    csr_matrixes_reduced = sparse_matrixes[non_zero_rows, :][:, non_zero_cols]
    rhs_reduced = rhs[non_zero_rows]
    x_reduced = spsolve(csr_matrixes_reduced, rhs_reduced)
    x_full = np.zeros(sparse_matrixes.shape[1])
    x_full[non_zero_cols] = x_reduced
    return x_full


def accepts_monotone_energy(initial, trial, step, slope):
    """Allow reduction roundoff only when the predicted decrease is also unresolved."""
    if not all(math.isfinite(value) for value in (initial, trial, step, slope)) or step <= 0.0 or slope >= 0.0:
        return False
    # Match f64 IPC reduction accuracy without an absolute, unit-dependent energy floor.
    roundoff = 1.0e-12 * max(abs(initial), abs(trial))
    return trial <= initial or (-step * slope <= roundoff and trial - initial <= roundoff)


def inexact_newton_relative_tolerance(residual, previous_residual, minimum=1.0e-7):
    """Bound the adaptive Krylov forcing term without relaxing Newton convergence."""
    if previous_residual is None or previous_residual <= 0.0:
        return 1.0e-2
    return min(1.0e-2, max(minimum, 0.9 * (residual / previous_residual) ** 1.5))


def inexact_newton_absolute_tolerance(residual, forcing, absolute_tolerance):
    """Scale the linear error by physical force, before Dirichlet elimination."""
    if not all(math.isfinite(value) and value >= 0.0 for value in (residual, forcing, absolute_tolerance)):
        raise ValueError("Inexact Newton residual and tolerances must be finite and nonnegative")
    return max(absolute_tolerance, forcing * residual)
