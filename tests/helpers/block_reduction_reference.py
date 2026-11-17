"""Independent sparse block reduction oracles used by tests."""

import numpy as np


def reduce_triplets_reference(block_i, block_j, block_h):
    block_i = np.asarray(block_i, dtype=np.int32)
    block_j = np.asarray(block_j, dtype=np.int32)
    block_h = np.asarray(block_h, dtype=np.float64)
    valid = (block_i >= 0) & (block_j >= 0)
    block_i = block_i[valid]
    block_j = block_j[valid]
    block_h = block_h[valid]
    order = np.lexsort((block_j, block_i))
    block_i = block_i[order]
    block_j = block_j[order]
    block_h = block_h[order]

    out_i = []
    out_j = []
    out_h = []
    for idx in range(block_i.shape[0]):
        if idx == 0 or block_i[idx] != out_i[-1] or block_j[idx] != out_j[-1]:
            out_i.append(int(block_i[idx]))
            out_j.append(int(block_j[idx]))
            out_h.append(block_h[idx].copy())
        else:
            out_h[-1] += block_h[idx]
    reduced_h = np.asarray(out_h, dtype=np.float64) if out_h else np.zeros((0, block_h.shape[1]), dtype=np.float64)
    return (
        np.asarray(out_i, dtype=np.int32),
        np.asarray(out_j, dtype=np.int32),
        reduced_h,
    )


def reduce_triplets_scipy(block_i, block_j, block_h, shape=None):
    from scipy.sparse import coo_matrix

    block_i = np.asarray(block_i, dtype=np.int32)
    block_j = np.asarray(block_j, dtype=np.int32)
    block_h = np.asarray(block_h, dtype=np.float64)
    valid = (block_i >= 0) & (block_j >= 0)
    block_i = block_i[valid]
    block_j = block_j[valid]
    block_h = block_h[valid]
    if shape is None:
        n = int(max(block_i.max(initial=0), block_j.max(initial=0)) + 1)
        shape = (n, n)
    pattern = coo_matrix((np.ones(block_i.shape[0]), (block_i, block_j)), shape=shape).tocsr().tocoo()
    hessian_size = block_h.shape[1]
    reduced_h = np.zeros((pattern.nnz, hessian_size), dtype=np.float64)
    for comp in range(hessian_size):
        reduced = coo_matrix((block_h[:, comp], (block_i, block_j)), shape=shape).tocsr()
        reduced_h[:, comp] = np.asarray(reduced[pattern.row, pattern.col]).reshape(-1)
    return pattern.row.astype(np.int32), pattern.col.astype(np.int32), reduced_h
