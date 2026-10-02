import numpy as np


def checked_scipy_krylov(
    solver,
    matrix,
    rhs,
    *,
    rtol=1.0e-5,
    atol=0.0,
    solver_name=None,
    **kwargs,
):
    """Run a SciPy Krylov solver and reject reported or true-residual failure."""
    right_hand_side = np.asarray(rhs, dtype=np.float64).reshape(-1)
    options = dict(kwargs, rtol=rtol)
    if atol is not None:
        options["atol"] = atol
    solution, info = solver(matrix, right_hand_side, **options)
    solution = np.asarray(solution, dtype=np.float64).reshape(-1)
    matrix_product = np.asarray(matrix @ solution, dtype=np.float64).reshape(-1)
    residual = float(np.linalg.norm(right_hand_side - matrix_product))
    rhs_norm = float(np.linalg.norm(right_hand_side))
    target = max(0.0 if atol is None else float(atol), float(rtol) * rhs_norm)
    roundoff = 32.0 * np.finfo(np.float64).eps * max(rhs_norm, float(np.linalg.norm(matrix_product)), 1.0)
    name = solver_name or getattr(solver, "__name__", "SciPy Krylov")
    if info != 0 or not np.all(np.isfinite(solution)) or not np.isfinite(residual) or residual > target + roundoff:
        raise RuntimeError(
            f"{name} failed true-residual verification: info={info}, " f"residual={residual:.6e}, target={target:.6e}"
        )
    return solution
