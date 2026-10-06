"""Contact buckets must preserve changing blocks and mixed body assembly."""

import os

import numpy as np
import pytest
import taichi as ti

from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.BucketReduction import BucketReduction
from src.linear_solver.HashReduction import HashReduction


def test_contact_bucket_reduction_and_mixed_assembly():
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(
            arch=ti.cuda if os.environ.get("GEOTAICHI_BUCKET_CUDA") else ti.cpu,
            default_fp=ti.f64,
            debug=True,
            offline_cache=False,
        )
    rng = np.random.default_rng(53)
    coordinates = np.array([0, 1, 511, 512, 1025, 262144], dtype=np.int32)
    for dim in (2, 3):
        reduction = BucketReduction(4096, dim=dim, max_nnz=4096, hessian_size=dim * dim)
        for count in (0, 1, 31, 32, 33, 65, 257, 1025, 4096, 0):
            i, j = rng.choice(coordinates, (2, count)).astype(np.int32)
            h = rng.normal(size=(count, dim * dim))
            if count == 4096:
                i.fill(512)
                j.fill(1025)
            if count == 257:
                i[::7] = -1
            reduction.set_triplets_from_numpy(i, j, h)
            reduction.go(count)
            out_i, out_j, out_h = reduction.get_reduced_triplets_numpy()
            ref_i, ref_j, ref_h = HashReduction.reduce_triplets_reference(i, j, h)
            order = np.lexsort((out_j, out_i))
            np.testing.assert_array_equal(out_i[order], ref_i)
            np.testing.assert_array_equal(out_j[order], ref_j)
            np.testing.assert_allclose(out_h[order], ref_h, rtol=1e-10, atol=1e-11)
        reduction.set_triplets_from_numpy(np.full(33, -1, np.int32), np.zeros(33, np.int32), np.ones((33, dim * dim)))
        reduction.go(33)
        assert int(reduction.element_pair_num[0]) == 0

        options = dict(
            dim=dim, max_pairs_num=128, max_nonzeros=128, max_active_nodes=4, symmetric=False, device_reduction=True
        )
        body = BuildTriplet(**options, raw_only=True)
        contact = BuildTriplet(**options, reduction="bucket")
        baseline = BuildTriplet(**options, matrix_symmetric=True, full_symmetric_input=True)
        hybrid = BuildTriplet(**options, matrix_symmetric=True, full_symmetric_input=True)
        assert type(body.non_diag) is HashReduction
        assert type(hybrid.non_diag) is HashReduction
        assert type(contact.non_diag) is BucketReduction
        for topology in (0, 1, 2):
            body.reset_system()
            contact.reset_system()
            i = np.array([0, 0, 1, 1, -1], np.int32)
            j = np.array([1, 1, 0, 0, -1], np.int32)
            if topology == 1:
                j[:2], i[2:4] = 2, 2
            h = rng.normal(size=(5, dim * dim))
            for source in (body, contact):
                source.non_diag.set_triplets_from_numpy(i, j, h)
                source.raw_non_diag_count[0] = 0 if topology == 2 else len(i)
                source.diag.from_numpy(rng.normal(size=(4, dim * dim)))
            for destination, reduced in ((baseline, False), (hybrid, True)):
                destination.reset_system()
                destination.append_raw_from(body, active_nodes=3)
                append = destination.append_reduced_from if reduced else destination.append_raw_from
                append(contact, active_nodes=3, block_offset=1, scale=0.7)
                destination.finalize_taichi_assembly()
            np.testing.assert_allclose(hybrid.to_scipy(4).toarray(), baseline.to_scipy(4).toarray(), atol=1e-11)
            assert int(hybrid.raw_non_diag_count[0]) <= int(baseline.raw_non_diag_count[0])
        with pytest.raises(ValueError, match="storage must match"):
            hybrid.append_reduced_from(baseline)
        body.overflow[0] = 1
        with pytest.raises(RuntimeError, match="raw-only"):
            hybrid.append_reduced_from(body)
    reduction = BucketReduction(2, dim=2, max_nnz=1, hessian_size=4)
    reduction.set_triplets_from_numpy(np.array([0, 1], np.int32), np.array([1, 0], np.int32), np.ones((2, 4)))
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        reduction.go(2)
    with pytest.raises(ValueError, match="capacity"):
        reduction.go(3)
