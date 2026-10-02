"""Taichi kernels for CSR sparse matrix operations."""

import taichi as ti

from src.utils.constants import WARP_SZ
from src.utils.WarpReduce import warp_scan_up_f32


@ti.kernel
def compute_Ap(
    total_row: int,
    values: ti.template(),
    indices: ti.template(),
    offsets: ti.template(),
    p: ti.template(),
    Ap: ti.template(),
):
    for i in range(total_row):
        sums = 0.0
        for index in range(offsets[i], offsets[i + 1]):
            sums += p[indices[index]] * values[index]
        Ap[i] = sums


@ti.kernel
def compute_Ap_warp_reduce(
    total_row: int,
    values: ti.template(),
    indices: ti.template(),
    offsets: ti.template(),
    p: ti.template(),
    Ap: ti.template(),
):
    for thread_id in range(WARP_SZ * total_row):
        warp_id = thread_id // WARP_SZ
        lane_id = thread_id % WARP_SZ

        sums = ti.cast(0.0, ti.f32)
        if warp_id < total_row:
            index = offsets[warp_id] + lane_id
            while index < offsets[warp_id + 1]:
                sums += p[indices[index]] * values[index]
                index += WARP_SZ

        sums = warp_scan_up_f32(lane_id, sums)
        ti.simt.block.sync()
        if lane_id == 0 and warp_id < total_row:
            Ap[warp_id] = sums


@ti.kernel
def compute_Ap_shared_reduce(
    total_row: int,
    values: ti.template(),
    indices: ti.template(),
    offsets: ti.template(),
    p: ti.template(),
    Ap: ti.template(),
):
    ti.loop_config(block_dim=WARP_SZ)
    for thread_id in range(WARP_SZ * total_row):
        warp_id = thread_id // WARP_SZ
        lane_id = thread_id % WARP_SZ
        sdata = ti.simt.block.SharedArray((WARP_SZ,), ti.f64)

        if warp_id < total_row:
            index = offsets[warp_id] + lane_id
            while index < offsets[warp_id + 1]:
                sdata[warp_id] += p[indices[index]] * values[index]
                index += WARP_SZ
        ti.simt.block.sync()

        temp = int(0.5 * WARP_SZ)
        while temp > 1:
            if lane_id < temp:
                sdata[lane_id] += sdata[lane_id + temp]
                temp = int(0.5 * temp)

        if lane_id == 0 and warp_id < total_row:
            Ap[warp_id] = sdata[0]
