import taichi as ti

from src.utils.BitFunction import split_f64, merge_f64, brev, clz
from src.utils.constants import WARP_SZ

@ti.func
def warp_scan_up_i32(lane_id, val):
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
def warp_scan_up_f32(lane_id, val):
    # Intra-warp scan, manually unrolled
    offset_j = 1
    n = ti.simt.warp.shfl_up_f32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 2
    n = ti.simt.warp.shfl_up_f32(ti.simt.warp.active_mask(), val, offset_j)
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
def warp_scan_down_i32(lane_id, val):
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
def warp_scan_down_f32(lane_id, val):
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

@ti.func
def shfl_down_f64(x: ti.f64, offset: ti.i32):
    lo, hi = split_f64(x)
    lo_i32 = ti.bit_cast(lo, ti.i32)
    hi_i32 = ti.bit_cast(hi, ti.i32)
    lo2 = ti.simt.warp.shfl_down_i32(ti.simt.warp.active_mask(), lo_i32, offset)
    hi2 = ti.simt.warp.shfl_down_i32(ti.simt.warp.active_mask(), hi_i32, offset)
    return merge_f64(ti.bit_cast(lo2, ti.u32), ti.bit_cast(hi2, ti.u32))

@ti.func
def warp_scan_down_f64(lane_id, val):
    offset_j = 1
    n = shfl_down_f64(val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 2
    n = shfl_down_f64(val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 4
    n = shfl_down_f64(val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 8
    n = shfl_down_f64(val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 16
    n = shfl_down_f64(val, offset_j)
    if lane_id >= offset_j:
        val += n
    return val

@ti.func
def warp_reduce_sum_f32(val):
    val = ti.cast(val, ti.f32)
    val += ti.simt.warp.shfl_down_f32(ti.simt.warp.active_mask(), val, 16)
    val += ti.simt.warp.shfl_down_f32(ti.simt.warp.active_mask(), val, 8)
    val += ti.simt.warp.shfl_down_f32(ti.simt.warp.active_mask(), val, 4)
    val += ti.simt.warp.shfl_down_f32(ti.simt.warp.active_mask(), val, 2)
    val += ti.simt.warp.shfl_down_f32(ti.simt.warp.active_mask(), val, 1,)
    return val

@ti.func
def warp_reduce_sum_f64(val: ti.f64):
    val += shfl_down_f64(val, 16)
    val += shfl_down_f64(val, 8)
    val += shfl_down_f64(val, 4)
    val += shfl_down_f64(val, 2)
    val += shfl_down_f64(val, 1)
    return val

@ti.kernel
def warp_reduce_i32(value: ti.template(), flag: ti.template(), output: ti.template()):
    for threadIdx in value:
        rdata = value[threadIdx]
        warp_id = threadIdx & 0x1f
        bBoundary = (int(flag[threadIdx]) == 1) or (warp_id == 0)

        mark = brev(ti.simt.warp.ballot(bBoundary))
        interval = min(ti.i32(clz(mark << (warp_id + 1))), 31 - warp_id)
        rdata = warp_scan_down_i32(interval, rdata)
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
        rdata = warp_scan_down_f32(interval, rdata)
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
        rdata = warp_scan_down_f64(interval, rdata)
        if bBoundary:
            output[threadIdx] += rdata
