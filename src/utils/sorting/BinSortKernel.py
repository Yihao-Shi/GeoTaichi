"""Taichi kernels for spatial bin sorting."""

import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.utils.ScalarFunction import linearize


@ti.kernel
def fill_object_bin(
    particleNum: int,
    igrid_size: ti.template(),
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    position: ti.template(),
    particle_count: ti.template(),
):
    particle_count.fill(0)
    ti.block_local(particle_count)
    for np in range(particleNum):
        grid_idx = ti.floor(position[np] * igrid_size, int)
        cellID = linearize(grid_idx, cnum)
        ti.atomic_add(particle_count[cellID + 1], 1)


@ti.kernel
def initialize_bin_cursor(
    cell_count: int,
    particle_count: ti.template(),
    bin_cursor: ti.template(),
):
    for cellID in range(cell_count):
        bin_cursor[cellID] = particle_count[cellID]


@ti.kernel
def object_sorted(
    particleNum: int,
    igrid_size: ti.template(),
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    position: ti.template(),
    bin_cursor: ti.template(),
    particleID: ti.template(),
):
    for np in range(particleNum):
        grid_idx = ti.floor(position[np] * igrid_size, int)
        cellID = linearize(grid_idx, cnum)
        grain_location = ti.atomic_add(bin_cursor[cellID], 1)
        particleID[grain_location] = np


@ti.kernel
def fill_object_bin_condition(
    particleNum: int,
    igrid_size: ti.template(),
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    condition: ti.template(),
    position: ti.template(),
    particle_count: ti.template(),
):
    particle_count.fill(0)
    ti.block_local(particle_count)
    for np in range(particleNum):
        if int(condition[np]) == 1:
            continue
        grid_idx = ti.floor(position[np] * igrid_size, int)
        cellID = linearize(grid_idx, cnum)
        ti.atomic_add(particle_count[cellID + 1], 1)


@ti.kernel
def object_sorted_condition(
    particleNum: int,
    igrid_size: ti.template(),
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    condition: ti.template(),
    position: ti.template(),
    bin_cursor: ti.template(),
    particleID: ti.template(),
):
    for np in range(particleNum):
        if int(condition[np]) == 1:
            continue
        grid_idx = ti.floor(position[np] * igrid_size, int)
        cellID = linearize(grid_idx, cnum)
        grain_location = ti.atomic_add(bin_cursor[cellID], 1)
        particleID[grain_location] = np
