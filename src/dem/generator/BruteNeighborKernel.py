"""Taichi kernels and device helpers for brute-force DEM neighbors."""

import taichi as ti

from src.utils.constants import THRESHOLD
from src.utils.GeometryFunction import intersection_ob_bs
from src.utils.Quaternion import set_to_rotate, set_from_euler


@ti.kernel
def kernel_pre_neighbor_sphere(
    body_num: int,
    offset: ti.template(),
    particle: ti.template(),
    sphere: ti.template(),
    position: ti.template(),
    radius: ti.template(),
    check_in_region: ti.template(),
    start_point: ti.types.vector(3, float),
):
    for nb in range(body_num):
        index = sphere[nb].sphereIndex
        if int(particle.multisphereIndex[index]) < 0:
            pos = particle[index].x
            rad = particle[index].rad
            if check_in_region(pos, rad):
                pre_insert_particle(start_point, pos, rad, offset, position, radius)


@ti.kernel
def kernel_pre_neighbor_clump(
    body_num: int,
    offset: ti.template(),
    particle: ti.template(),
    clump: ti.template(),
    position_field: ti.template(),
    radius_field: ti.template(),
    check_in_region: ti.template(),
    start_point: ti.types.vector(3, float),
):
    for nb in range(body_num):
        start, end = clump[nb].startIndex, clump[nb].endIndex
        is_in_region = 1
        for npebble in range(start, end):
            position = particle[npebble].x
            rad = particle[npebble].rad
            if not check_in_region(position, rad):
                is_in_region = 0
                break
        if is_in_region:
            for npebble in range(start, end):
                position = particle.x[npebble]
                radius = particle.rad[npebble]
                pre_insert_particle(start_point, position, radius, offset, position_field, radius_field)


@ti.kernel
def kernel_pre_neighbor_bounding_sphere(
    rigid_num: int,
    offset: ti.template(),
    bounding_sphere: ti.template(),
    position: ti.template(),
    radius: ti.template(),
    check_in_region: ti.template(),
    start_point: ti.types.vector(3, float),
):
    for nb in range(rigid_num):
        pos = bounding_sphere[nb].x
        rad = bounding_sphere[nb].rad
        if check_in_region(pos, rad):
            pre_insert_rigid_particle(start_point, pos, rad, offset, position, radius)


@ti.func
def pre_insert_particle(pos0, pos, rad, offset, position, radius):
    particle_number = ti.atomic_add(offset[None], 1)
    position[particle_number] = pos - pos0
    radius[particle_number] = rad


@ti.func
def pre_insert_rigid_particle(pos0, pos, rad, offset, position, radius):
    particle_number = ti.atomic_add(offset[None], 1)
    position[particle_number] = pos - pos0
    radius[particle_number] = rad


@ti.func
def insert_particle(cell_num, cell_size, pos, rad, offset, position, radius, num_particle_in_cell, particle_neighbor):
    position[offset[None]] = pos
    radius[offset[None]] = rad
    offset[None] += 1


@ti.func
def insert_rigid_particle(
    cell_num, cell_size, pos, rad, ori, offset, position, radius, orient, num_particle_in_cell, particle_neighbor
):
    position[offset[None]] = pos
    radius[offset[None]] = rad
    orient[offset[None]] = ori
    offset[None] += 1


@ti.func
def overlap(cell_num, cell_size, pos, rad, offset, position, radius, num_particle_in_cell, particle_neighbor):
    isoverlap = 0
    for np in range(offset[None]):
        dist = (position[np] - pos).norm()
        delta = -dist + (radius[np] + rad)
        if delta > THRESHOLD:
            isoverlap = 1
            break
    return isoverlap


@ti.func
def overlap_rigid(
    cell_num,
    cell_size,
    pos,
    rad,
    offset,
    position,
    radius,
    orient,
    num_particle_in_cell,
    particle_neighbor,
    template_xbound,
    template_rbound,
    min_box,
    max_box,
    ori,
):
    isoverlap = 0
    scale_factor1 = rad / template_rbound
    rotation_matrix1 = set_to_rotate(set_from_euler(*ori))
    obb1 = (
        pos
        - scale_factor1 * rotation_matrix1 @ template_xbound
        + rotation_matrix1 @ (scale_factor1 * 0.5 * (min_box + max_box))
    )
    for np in range(offset[None]):
        slave_pos = position[np]
        slave_rad = radius[np]
        slave_ori = orient[np]
        scale_factor2 = slave_rad / template_rbound
        rotation_matrix2 = set_to_rotate(set_from_euler(*slave_ori))
        obb2 = (
            slave_pos
            - scale_factor2 * rotation_matrix2 @ template_xbound
            + rotation_matrix2 @ (scale_factor2 * 0.5 * (min_box + max_box))
        )
        if intersection_ob_bs(
            obb1,
            obb2,
            scale_factor1 * (max_box - min_box),
            scale_factor2 * (max_box - min_box),
            rotation_matrix1,
            rotation_matrix2,
        ):
            isoverlap = 1
            break
    return isoverlap
