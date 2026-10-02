"""Unit checks for coordinate-sparse hash reduction."""

import numpy as np

from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix


def _assert_same_csr(A, B, atol=1.0e-12):
    diff = (A - B).tocoo()
    max_abs = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    rel = max_abs / max(float(np.max(np.abs(A.data))) if A.nnz else 0.0, 1.0e-30)
    print(f"CoordinateSparseMatrix hash reduction: nnz_scipy={A.nnz}, nnz_hash={B.nnz}, max_abs={max_abs:.3e}, rel={rel:.3e}")
    assert A.shape == B.shape
    assert max_abs <= atol


def test_coordinate_sparse_hash_reduction_matches_scipy(taichi_runtime):
    rng = np.random.default_rng(4)

    dofs = 300_128
    unique_rows = rng.integers(0, dofs, size=24, dtype=np.int32)
    unique_cols = rng.integers(0, dofs, size=24, dtype=np.int32)
    unique_rows[:4] = np.array([0, 511, 262_144, 300_000], dtype=np.int32)
    unique_cols[:4] = np.array([300_001, 262_145, 511, 0], dtype=np.int32)

    pattern_id = rng.integers(0, unique_rows.shape[0], size=256, dtype=np.int32)
    rows = unique_rows[pattern_id]
    cols = unique_cols[pattern_id]
    data = rng.normal(size=pattern_id.shape[0])

    matrix = CoordinateSparseMatrix(rows.shape[0], dofs, linear_solver=False)
    matrix.rows.from_numpy(rows)
    matrix.cols.from_numpy(cols)
    matrix.data.from_numpy(data)

    scipy_ref = matrix._to_scipy().tocsr()
    hash_matrix = matrix._to_scipy_hash().tocsr()
    _assert_same_csr(scipy_ref, hash_matrix)
