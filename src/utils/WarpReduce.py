import taichi as ti

from src.utils.constants import WARP_SZ


@ti.func
def warp_shfl_up_i32(val):
    global_tid = ti.simt.block.global_thread_idx()
    lane_id = global_tid % WARP_SZ
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
def warp_shfl_up_f32(val):
    global_tid = ti.simt.block.global_thread_idx()
    lane_id = global_tid % WARP_SZ
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