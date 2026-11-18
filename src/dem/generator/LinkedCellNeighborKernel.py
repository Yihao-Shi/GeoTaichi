"""Taichi kernels and device helpers for linked-cell DEM neighbors."""

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
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    neighbor_position: ti.template(),
    neighbor_radius: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    check_in_region: ti.template(),
    start_point: ti.types.vector(3, float),
):
    for nb in range(body_num):
        index = sphere[nb].sphereIndex
        position = particle[index].x
        radius = particle[index].rad
        if check_in_region(position, radius):
            pre_insert_particle(
                cell_num,
                cell_size,
                start_point,
                position,
                radius,
                offset,
                neighbor_position,
                neighbor_radius,
                num_particle_in_cell,
                particle_neighbor,
            )


@ti.kernel
def kernel_pre_neighbor_clump(
    body_num: int,
    offset: ti.template(),
    particle: ti.template(),
    clump: ti.template(),
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    neighbor_position: ti.template(),
    neighbor_radius: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    check_in_region: ti.template(),
    start_point: ti.types.vector(3, float),
):
    for nb in range(body_num):
        start, end = clump[nb].startIndex, clump[nb].endIndex
        is_in_region = 1
        for npebble in range(start, end):
            position = particle[npebble].x
            radius = particle[npebble].rad
            if not check_in_region(position, radius):
                is_in_region = 0
                break
        if is_in_region:
            for npebble in range(start, end):
                position = particle[npebble].x
                radius = particle[npebble].rad
                pre_insert_particle(
                    cell_num,
                    cell_size,
                    start_point,
                    position,
                    radius,
                    offset,
                    neighbor_position,
                    neighbor_radius,
                    num_particle_in_cell,
                    particle_neighbor,
                )


@ti.kernel
def kernel_pre_neighbor_bounding_sphere(
    body_num: int,
    offset: ti.template(),
    bounding_sphere: ti.template(),
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    neighbor_position: ti.template(),
    neighbor_radius: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    check_in_region: ti.template(),
    start_point: ti.types.vector(3, float),
):
    for nb in range(body_num):
        position = bounding_sphere[nb].x
        radius = bounding_sphere[nb].rad
        if check_in_region(position, radius):
            pre_insert_rigid_particle(
                cell_num,
                cell_size,
                start_point,
                position,
                radius,
                offset,
                neighbor_position,
                neighbor_radius,
                num_particle_in_cell,
                particle_neighbor,
            )


@ti.func
def get_cell_index(cell_size, pos):
    index = ti.floor(pos / cell_size, int)
    return index


@ti.func
def get_cell_id(idx, idy, idz, cell_num):
    return int(idx + idy * cell_num[0] + idz * cell_num[0] * cell_num[1])


@ti.func
def pre_insert_particle(
    cell_num, cell_size, pos0, pos, rad, offset, position, radius, num_particle_in_cell, particle_neighbor
):
    cell_index = get_cell_index(cell_size, pos - pos0)
    cell_id = get_cell_id(cell_index[0], cell_index[1], cell_index[2], cell_num)
    particle_number = ti.atomic_add(offset[None], 1)
    position[particle_number] = pos - pos0
    radius[particle_number] = rad
    particle_num_in_cell = ti.atomic_add(num_particle_in_cell[cell_id], 1)
    particle_neighbor[cell_id, particle_num_in_cell] = particle_number


@ti.func
def pre_insert_rigid_particle(
    cell_num, cell_size, pos0, pos, rad, offset, position, radius, num_particle_in_cell, particle_neighbor
):
    cell_index = get_cell_index(cell_size, pos - pos0)
    cell_id = get_cell_id(cell_index[0], cell_index[1], cell_index[2], cell_num)
    particle_number = ti.atomic_add(offset[None], 1)
    position[particle_number] = pos - pos0
    radius[particle_number] = rad
    particle_num_in_cell = ti.atomic_add(num_particle_in_cell[cell_id], 1)
    particle_neighbor[cell_id, particle_num_in_cell] = particle_number


@ti.func
def insert_particle(cell_num, cell_size, pos, rad, offset, position, radius, num_particle_in_cell, particle_neighbor):
    cell_index = get_cell_index(cell_size, pos)
    cell_id = get_cell_id(cell_index[0], cell_index[1], cell_index[2], cell_num)
    particle_id = ti.atomic_add(offset[None], 1)
    particle_in_cell = num_particle_in_cell[cell_id]

    position[particle_id] = pos
    radius[particle_id] = rad
    particle_neighbor[cell_id, particle_in_cell] = particle_id
    ti.atomic_add(num_particle_in_cell[cell_id], 1)


@ti.func
def insert_rigid_particle(
    cell_num, cell_size, pos, rad, ori, offset, position, radius, orient, num_particle_in_cell, particle_neighbor
):
    cell_index = get_cell_index(cell_size, pos)
    cell_id = get_cell_id(cell_index[0], cell_index[1], cell_index[2], cell_num)
    particle_id = ti.atomic_add(offset[None], 1)
    particle_in_cell = num_particle_in_cell[cell_id]

    position[particle_id] = pos
    radius[particle_id] = rad
    orient[particle_id] = ori
    particle_neighbor[cell_id, particle_in_cell] = particle_id
    ti.atomic_add(num_particle_in_cell[cell_id], 1)


@ti.func
def overlap(cell_num, cell_size, pos, rad, offset, position, radius, num_particle_in_cell, particle_neighbor):
    isoverlap = 0
    cell_index = get_cell_index(cell_size, pos)
    x_begin = ti.cast(ti.max(cell_index[0] - 1, 0), int)
    x_end = ti.cast(ti.min(cell_index[0] + 2, cell_num[0]), int)
    y_begin = ti.cast(ti.max(cell_index[1] - 1, 0), int)
    y_end = ti.cast(ti.min(cell_index[1] + 2, cell_num[1]), int)
    z_begin = ti.cast(ti.max(cell_index[2] - 1, 0), int)
    z_end = ti.cast(ti.min(cell_index[2] + 2, cell_num[2]), int)
    for i, j, k in ti.ndrange((x_begin, x_end), (y_begin, y_end), (z_begin, z_end)):
        cell_id = get_cell_id(i, j, k, cell_num)
        for particle_number in range(num_particle_in_cell[cell_id]):
            pid = particle_neighbor[cell_id, particle_number]
            slave_pos = position[pid]
            slave_rad = radius[pid]
            dist_vec = slave_pos - pos
            dist = dist_vec.norm()
            delta = -dist + slave_rad + rad
            if delta > THRESHOLD:
                isoverlap = 1
                break
        if isoverlap == 1:
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
    cell_index = get_cell_index(cell_size, pos)
    x_begin = ti.cast(ti.max(cell_index[0] - 1, 0), int)
    x_end = ti.cast(ti.min(cell_index[0] + 2, cell_num[0]), int)
    y_begin = ti.cast(ti.max(cell_index[1] - 1, 0), int)
    y_end = ti.cast(ti.min(cell_index[1] + 2, cell_num[1]), int)
    z_begin = ti.cast(ti.max(cell_index[2] - 1, 0), int)
    z_end = ti.cast(ti.min(cell_index[2] + 2, cell_num[2]), int)
    scale_factor1 = rad / template_rbound
    rotation_matrix1 = set_to_rotate(set_from_euler(*ori))
    obb1 = (
        pos
        - scale_factor1 * rotation_matrix1 @ template_xbound
        + rotation_matrix1 @ (scale_factor1 * 0.5 * (min_box + max_box))
    )
    for i, j, k in ti.ndrange((x_begin, x_end), (y_begin, y_end), (z_begin, z_end)):
        cell_id = get_cell_id(i, j, k, cell_num)
        for particle_number in range(num_particle_in_cell[cell_id]):
            pid = particle_neighbor[cell_id, particle_number]
            slave_pos = position[pid]
            slave_rad = radius[pid]
            slave_ori = orient[pid]
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
        if isoverlap == 1:
            break
    return isoverlap
