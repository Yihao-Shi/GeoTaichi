"""Taichi kernels for coordinate sparse matrix operations."""

import taichi as ti

from src.utils.BitFunction import ballot, brev, clz
from src.utils.constants import BLOCK_SZ, WARP_SZ


@ti.kernel
def set_from_triplets(bsr_matrix: ti.template(), rows: ti.template(), cols: ti.template(), values: ti.template()):
    ti.loop_config(block_dim=BLOCK_SZ)
    for idx in range(rows.shape[0]):
        offset = ti.simt.block.SharedArray((1,), ti.i32)
        thread_id = idx % BLOCK_SZ
        block_id = idx // BLOCK_SZ
        lane_id = thread_id & 0x1F
        warp_id = thread_id // WARP_SZ

        row_ind = rows[idx]
        col_ind = cols[idx]
        rdata = values[idx]

        mask = ti.simt.warp.active_mask()
        prev_endID1 = ti.simt.warp.shfl_up_i32(mask, row_ind, 1)
        ti.simt.block.sync()

        bBoundary = (lane_id == 0) or (row_ind != prev_endID1)
        mark = ballot(bBoundary)
        mark = brev(mark)
        interval = ti.min(clz(mark << (lane_id + 1)), 31 - lane_id)
        ti.simt.block.sync()

        index = 1
        while index < min(WARP_SZ, 12):
            if interval >= index:
                pass
            index <<= 1

        if bBoundary == 1:
            pass


@ti.kernel
def compute_Ap(
    nnz: int, row: ti.template(), col: ti.template(), value: ti.template(), p: ti.template(), Ap: ti.template()
):
    for i in range(nnz):
        ti.atomic_add(Ap[row[i]], p[col[i]] * value[i])


@ti.kernel
def _copy_triplets_to_hash_reducer_impl(
    nnz: int,
    rows: ti.template(),
    cols: ti.template(),
    data: ti.template(),
    block_i: ti.template(),
    block_j: ti.template(),
    block_h: ti.template(),
):
    for i in range(nnz):
        block_i[i] = rows[i]
        block_j[i] = cols[i]
        block_h[i][0] = data[i]
