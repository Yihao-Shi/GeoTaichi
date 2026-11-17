"""Taichi kernels for spatial bin sorting."""

import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.utils.ScalarFunction import linearize


@ti.kernel
def fill_object_bin(
    particle_num: int,
    igrid_size: ti.template(),
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    position: ti.template(),
    particle_count: ti.template(),
):
    particle_count.fill(0)
    ti.block_local(particle_count)
    for np in range(particle_num):
        grid_idx = ti.floor(position[np] * igrid_size, int)
        cell_id = linearize(grid_idx, cnum)
        ti.atomic_add(particle_count[cell_id + 1], 1)


@ti.kernel
def initialize_bin_cursor(
    cell_count: int,
    particle_count: ti.template(),
    bin_cursor: ti.template(),
):
    for cell_id in range(cell_count):
        bin_cursor[cell_id] = particle_count[cell_id]


@ti.kernel
def object_sorted(
    particle_num: int,
    igrid_size: ti.template(),
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    position: ti.template(),
    bin_cursor: ti.template(),
    particle_id: ti.template(),
):
    for np in range(particle_num):
        grid_idx = ti.floor(position[np] * igrid_size, int)
        cell_id = linearize(grid_idx, cnum)
        grain_location = ti.atomic_add(bin_cursor[cell_id], 1)
        particle_id[grain_location] = np


@ti.kernel
def fill_object_bin_condition(
    particle_num: int,
    igrid_size: ti.template(),
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    condition: ti.template(),
    position: ti.template(),
    particle_count: ti.template(),
):
    particle_count.fill(0)
    ti.block_local(particle_count)
    for np in range(particle_num):
        if int(condition[np]) == 1:
            continue
        grid_idx = ti.floor(position[np] * igrid_size, int)
        cell_id = linearize(grid_idx, cnum)
        ti.atomic_add(particle_count[cell_id + 1], 1)


@ti.kernel
def object_sorted_condition(
    particle_num: int,
    igrid_size: ti.template(),
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    condition: ti.template(),
    position: ti.template(),
    bin_cursor: ti.template(),
    particle_id: ti.template(),
):
    for np in range(particle_num):
        if int(condition[np]) == 1:
            continue
        grid_idx = ti.floor(position[np] * igrid_size, int)
        cell_id = linearize(grid_idx, cnum)
        grain_location = ti.atomic_add(bin_cursor[cell_id], 1)
        particle_id[grain_location] = np
