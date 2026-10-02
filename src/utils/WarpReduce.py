import taichi as ti

from src.utils.BitFunction import split_f64_precision, merge_f64_precision
from src.utils.BitFunction import brev, clz


@ti.func
def warp_shfl_up_i32(lane_id, val):
    # Intra-warp scan, manually unrolled
    offset_j = 1
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 2
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 4
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 8
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 16
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    return val


@ti.func
def warp_shfl_up_f32(lane_id, val):
    # Intra-warp scan, manually unrolled
    offset_j = 1
    n = ti.simt.warp.shfl_up_f32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 2
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 4
    n = ti.simt.warp.shfl_up_f32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 8
    n = ti.simt.warp.shfl_up_f32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 16
    n = ti.simt.warp.shfl_up_f32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    return val


@ti.func
def warp_shfl_down_i32(lane_id, val):
    # Intra-warp scan, manually unrolled
    offset_j = 1
    n = ti.simt.warp.shfl_down_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 2
    n = ti.simt.warp.shfl_down_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 4
    n = ti.simt.warp.shfl_down_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 8
    n = ti.simt.warp.shfl_down_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 16
    n = ti.simt.warp.shfl_down_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    return val


@ti.func
def warp_shfl_down_f32(lane_id, val):
    # Intra-warp scan, manually unrolled
    offset_j = 1
    n = ti.simt.warp.shfl_down_f32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 2
    n = ti.simt.warp.shfl_down_f32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 4
    n = ti.simt.warp.shfl_down_f32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 8
    n = ti.simt.warp.shfl_down_f32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 16
    n = ti.simt.warp.shfl_down_f32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    return val


@ti.kernel
def warp_reduce_i32(value: ti.template(), flag: ti.template(), output: ti.template()):
    for threadIdx in value:
        rdata = value[threadIdx]
        warp_id = threadIdx & 0x1f
        bBoundary = (int(flag[threadIdx]) == 1) or (warp_id == 0)

        mark = brev(ti.simt.warp.ballot(bBoundary))
        interval = min(ti.i32(clz(mark << (warp_id + 1))), 31 - warp_id)
        rdata = warp_shfl_down_i32(interval, rdata)
        if bBoundary:
            output[threadIdx] += rdata


@ti.kernel
def warp_reduce_f32(value: ti.template(), flag: ti.template(), output: ti.template()):
    for threadIdx in value:
        rdata = value[threadIdx]
        warp_id = threadIdx & 0x1f
        bBoundary = (int(flag[threadIdx]) == 1) or (warp_id == 0)

        mark = brev(ti.simt.warp.ballot(bBoundary))
        interval = min(ti.i32(clz(mark << (warp_id + 1))), 31 - warp_id)
        rdata = warp_shfl_down_f32(interval, rdata)
        if bBoundary:
            output[threadIdx] += rdata


@ti.kernel
def warp_reduce_f64(value: ti.template(), flag: ti.template(), output: ti.template()):
    for threadIdx in value:
        rdata = value[threadIdx]
        warp_id = threadIdx & 0x1f
        bBoundary = (int(flag[threadIdx]) == 1) or (warp_id == 0)

        mark = brev(ti.simt.warp.ballot(bBoundary))
        interval = min(ti.i32(clz(mark << (warp_id + 1))), 31 - warp_id)
        high_f32, low_f32 = split_f64_precision(rdata)
        high_f32 = warp_shfl_down_f32(interval, high_f32)
        low_f32 = warp_shfl_down_f32(interval, low_f32)
        if bBoundary:
            output[threadIdx] += merge_f64_precision(high_f32, low_f32)
