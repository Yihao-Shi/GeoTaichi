"""Device bucket sort/reduction for changing IPC contact block coordinates.

CUDA uses one warp per bucket; shared memory holds output offsets only.
Merge buffers stay in global memory. CPU/Metal use serial work within each
bucket while retaining the same device fields and prefix-sum/scatter stages.
"""

import numpy as np
import taichi as ti
from taichi.lang.impl import current_cfg

from src.linear_solver.HashReduction import HashReduction
from src.linear_solver.BlockSparsityPattern import BlockSparsityPatternCache
from src.utils.FieldIO import runtime_float_numpy_dtype
from src.utils.PrefixSum import PrefixSumExecutor
from src.utils.constants import WARP_SZ
from src.utils.BitFunction import ffs_u32
from src.utils.WarpReduce import warp_reduce_sum_f32, warp_reduce_sum_f64


@ti.data_oriented
class BucketReduction(HashReduction):
    def __init__(
        self,
        max_pairs_num,
        dim=3,
        max_nnz=0,
        hessian_size=6,
        pattern_cache=False,
        pattern_cache_extra_fraction=0.01,
        pattern_cache_max_age=25,
        device_reduction=None,
    ):
        max_pairs_num, max_nnz = int(max_pairs_num), int(max_nnz or max_pairs_num)
        if not 0 < max_pairs_num <= np.iinfo(np.int32).max or not 0 < max_nnz <= np.iinfo(np.int32).max:
            raise ValueError("bucket capacities must be positive int32 values")
        if device_reduction is False:
            raise ValueError("bucket reduction requires Taichi device fields")
        # Reuse upload/export/reference helpers without allocating the cached
        # hash table and raw-output mapping, which contact no longer uses.
        self.dim, self.hessian_size = dim, int(hessian_size)
        self.max_pairs_num = max_pairs_num
        self.device_reduction = True
        self.numpy_float_dtype = runtime_float_numpy_dtype()
        self.element_pair_num = ti.field(ti.i32, shape=1)
        self.blockI = ti.field(ti.i32, shape=max_pairs_num)
        self.blockJ = ti.field(ti.i32, shape=max_pairs_num)
        self.blockH = ti.Vector.field(self.hessian_size, float, shape=max_pairs_num)
        self.tripletI = ti.field(ti.i32, shape=max_nnz)
        self.tripletJ = ti.field(ti.i32, shape=max_nnz)
        self.tripletH = ti.Vector.field(self.hessian_size, float, shape=max_nnz)
        self.device_overflow = ti.field(ti.i32, shape=1)
        self.device_epoch = self.device_pattern_version = 0
        self.device_pattern_initialized = False
        self.device_pattern_enabled = False
        self.pattern_cache = BlockSparsityPatternCache(max_nnz, enabled=False)
        self.cell_shift = 9
        self.cell_length = 1 << self.cell_shift
        self.cell_mask = self.cell_length - 1
        self.cell_num = self.cell_length * self.cell_length
        self.block_size = 128
        self.use_f64_reduce = current_cfg().default_fp == ti.f64
        self.cuda = current_cfg().arch == ti.cuda
        self.pse = PrefixSumExecutor(self.cell_num + 1)
        self.cell_count = ti.field(ti.i32, shape=self.pse.get_length())
        self.bucket_cursor = ti.field(ti.i32, shape=self.cell_num)
        self.hash_table = ti.field(ti.i32, shape=max_pairs_num)
        self.sorted_hash_table = ti.field(ti.i32, shape=max_pairs_num)
        self.go = self.go_with_bucket

    def go_with_bucket(self, pairs_num):
        pairs_num = int(pairs_num)
        if not 0 <= pairs_num <= self.max_pairs_num:
            raise ValueError("raw pair count exceeds capacity")
        self.device_overflow[0] = 0
        self.element_pair_num[0] = 0
        self.device_epoch += 1
        if pairs_num:
            self.kernel_insert_hash_cell(pairs_num)
            self.pse.run(self.cell_count)
            self.kernel_scatter_hash_cell(pairs_num)
            if self.cuda:
                self.kernel_merge_hash_table()
                self.kernel_hash_reduction()
            else:
                self.kernel_sort_and_reduce_portable()
        if int(self.device_overflow[0]):
            raise RuntimeError("bucket reduction output capacity exceeded")

    def pattern_cache_statistics(self):
        nnz = int(self.element_pair_num[0])
        return dict(
            enabled=False,
            backend="taichi_bucket",
            pattern_nonzeros=nnz,
            current_nonzeros=nnz,
            pattern_version=self.device_epoch,
            pattern_rebuilds=self.device_epoch,
            pattern_hits=0,
            reductions=self.device_epoch,
            last_new_entries=nnz,
            last_mapping_misses=0,
        )

    @ti.func
    def cell_hash_value(self, x, y):
        xcell = (ti.cast(x, ti.i32) >> self.cell_shift) & self.cell_mask
        ycell = (ti.cast(y, ti.i32) >> self.cell_shift) & self.cell_mask
        return ycell | (xcell << self.cell_shift)

    @ti.kernel
    def kernel_sort_and_reduce_portable(self):
        for bucket in range(self.cell_num):
            begin, end = self.cell_count[bucket], self.cell_count[bucket + 1]
            width = 1
            while width < end - begin:
                left = begin
                while left < end:
                    middle = ti.min(left + width, end)
                    right = ti.min(left + 2 * width, end)
                    i, j, k = left, middle, left
                    while k < right:
                        take_left = j >= right
                        if i < middle and j < right:
                            a, b = self.hash_table[i], self.hash_table[j]
                            take_left = self.blockI[a] < self.blockI[b] or (
                                self.blockI[a] == self.blockI[b] and self.blockJ[a] <= self.blockJ[b]
                            )
                        if i < middle and take_left:
                            self.sorted_hash_table[k] = self.hash_table[i]
                            i += 1
                        else:
                            self.sorted_hash_table[k] = self.hash_table[j]
                            j += 1
                        k += 1
                    left += 2 * width
                for k in range(begin, end):
                    self.hash_table[k] = self.sorted_hash_table[k]
                width *= 2
            cursor = begin
            while cursor < end:
                raw = self.hash_table[cursor]
                row, col = self.blockI[raw], self.blockJ[raw]
                hess = ti.Vector.zero(float, self.hessian_size)
                same = True
                while cursor < end and same:
                    raw = self.hash_table[cursor]
                    if self.blockI[raw] == row and self.blockJ[raw] == col:
                        hess += self.blockH[raw]
                        cursor += 1
                    else:
                        same = False
                output = ti.atomic_add(self.element_pair_num[0], 1)
                if output < self.tripletI.shape[0]:
                    self.tripletI[output], self.tripletJ[output] = row, col
                    self.tripletH[output] = hess
                else:
                    self.device_overflow[0] = 1

    @ti.kernel
    def kernel_insert_hash_cell(self, pairs_num: int):
        self.cell_count.fill(0)
        for raw in range(pairs_num):
            if self.blockI[raw] >= 0 and self.blockJ[raw] >= 0:
                bucket = self.cell_hash_value(self.blockI[raw], self.blockJ[raw])
                ti.atomic_add(self.cell_count[bucket + 1], 1)

    @ti.kernel
    def kernel_scatter_hash_cell(self, pairs_num: int):
        self.bucket_cursor.fill(0)
        for raw in range(pairs_num):
            if self.blockI[raw] >= 0 and self.blockJ[raw] >= 0:
                bucket = self.cell_hash_value(self.blockI[raw], self.blockJ[raw])
                offset = ti.atomic_add(self.bucket_cursor[bucket], 1)
                self.hash_table[self.cell_count[bucket] + offset] = raw

    @ti.kernel
    def kernel_merge_hash_table(self):
        ti.loop_config(block_dim=self.block_size)
        for warp_thread in range(self.cell_num * WARP_SZ):
            nc = warp_thread // WARP_SZ
            thread_id = warp_thread % self.block_size
            lane_id = thread_id & (WARP_SZ - 1)
            begin = self.cell_count[nc]
            end = self.cell_count[nc + 1]
            num = end - begin
            width = 1
            while width < num:
                iternum = (int((num + width * 2 - 1) / (width * 2)) >> 5) + 1
                for witer in range(iternum):
                    left = (lane_id + witer * WARP_SZ) * width * 2
                    right = ti.min(left + width * 2, num)
                    middle = ti.min(left + width, num)
                    if left < num:
                        i = left
                        j = middle
                        k = left
                        while i < middle and j < right:
                            ij_left = self.hash_table[begin + i]
                            ij_right = self.hash_table[begin + j]
                            if self.blockI[ij_left] > self.blockI[ij_right]:
                                self.sorted_hash_table[begin + k] = self.hash_table[begin + j]
                                j += 1
                            elif (
                                self.blockI[ij_left] == self.blockI[ij_right]
                                and self.blockJ[ij_left] > self.blockJ[ij_right]
                            ):
                                self.sorted_hash_table[begin + k] = self.hash_table[begin + j]
                                j += 1
                            else:
                                self.sorted_hash_table[begin + k] = self.hash_table[begin + i]
                                i += 1
                            k += 1
                        while i < middle:
                            self.sorted_hash_table[begin + k] = self.hash_table[begin + i]
                            i += 1
                            k += 1
                        while j < right:
                            self.sorted_hash_table[begin + k] = self.hash_table[begin + j]
                            j += 1
                            k += 1
                ti.simt.warp.sync(ti.u32(0xFFFFFFFF))

                lane_iter = lane_id
                while lane_iter < num:
                    self.hash_table[begin + lane_iter] = self.sorted_hash_table[begin + lane_iter]
                    lane_iter += WARP_SZ
                ti.simt.warp.sync(ti.u32(0xFFFFFFFF))
                width *= 2

    @ti.kernel
    def kernel_hash_reduction(self):
        self.element_pair_num[0] = 0
        ti.loop_config(block_dim=self.block_size)
        for warp_thread in range(self.cell_num * WARP_SZ):
            nc = warp_thread // WARP_SZ
            thread_id = warp_thread % self.block_size
            lane_id = thread_id & (WARP_SZ - 1)
            begin = self.cell_count[nc]
            end = self.cell_count[nc + 1]
            num = end - begin
            if num <= 0:
                continue
            count = 0

            warp_start_index = ti.simt.block.SharedArray(self.block_size >> 5, dtype=ti.i32)

            lane_iter = lane_id + 1
            while lane_iter < num:
                blockI1 = self.blockI[self.hash_table[begin + lane_iter]]
                blockJ1 = self.blockJ[self.hash_table[begin + lane_iter]]
                blockI2 = self.blockI[self.hash_table[begin + lane_iter - 1]]
                blockJ2 = self.blockJ[self.hash_table[begin + lane_iter - 1]]
                if blockI1 != blockI2 or blockJ1 != blockJ2:
                    count += 1
                lane_iter += WARP_SZ
            ti.simt.warp.sync(ti.u32(0xFFFFFFFF))

            count += ti.simt.warp.shfl_xor_i32(ti.simt.warp.active_mask(), count, 16)
            count += ti.simt.warp.shfl_xor_i32(ti.simt.warp.active_mask(), count, 8)
            count += ti.simt.warp.shfl_xor_i32(ti.simt.warp.active_mask(), count, 4)
            count += ti.simt.warp.shfl_xor_i32(ti.simt.warp.active_mask(), count, 2)
            count += ti.simt.warp.shfl_xor_i32(ti.simt.warp.active_mask(), count, 1)

            if lane_id == 0:
                warp_start_index[thread_id >> 5] = ti.atomic_add(self.element_pair_num[0], count + 1)

            ti.simt.warp.sync(ti.u32(0xFFFFFFFF))
            st = 0
            ij_index = 0
            while st < num:
                base_i = self.blockI[self.hash_table[begin + st]]
                base_j = self.blockJ[self.hash_table[begin + st]]
                ij_num = 0
                witer_num = ((num - st) >> 5) + 1
                hess = ti.Vector.zero(float, self.hessian_size)
                for witer in range(witer_num):
                    lane_iter = lane_id + witer * WARP_SZ + st
                    a = 1
                    if lane_iter < num:
                        index = self.hash_table[begin + lane_iter]
                        end_i = self.blockI[index]
                        end_j = self.blockJ[index]
                        if end_i == base_i and end_j == base_j:
                            a = 0
                        hess += self.blockH[index] * (1.0 - a)
                    b = ti.simt.warp.ballot(a)
                    ed = ffs_u32(b) - 1

                    if ed == -1:
                        ij_num += WARP_SZ
                    else:
                        ij_num += ed
                        st += ij_num
                        break
                ti.simt.warp.sync(ti.u32(0xFFFFFFFF))

                reduced_hess = ti.Vector.zero(float, self.hessian_size)
                for m in ti.static(range(self.hessian_size)):
                    if ti.static(self.use_f64_reduce):
                        reduced_hess[m] = warp_reduce_sum_f64(ti.cast(hess[m], ti.f64))
                    else:
                        reduced_hess[m] = warp_reduce_sum_f32(hess[m])

                ti.simt.warp.sync(ti.u32(0xFFFFFFFF))

                if lane_id == 0:
                    output = warp_start_index[thread_id >> 5] + ij_index
                    if output < self.tripletI.shape[0]:
                        self.tripletI[output] = base_i
                        self.tripletJ[output] = base_j
                        self.tripletH[output] = reduced_hess
                    else:
                        self.device_overflow[0] = 1
                    ij_index += 1
