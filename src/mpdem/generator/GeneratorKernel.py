import taichi as ti

from src.utils.TypeDefination import vec3i, vec3f
from src.utils.Quaternion import SetToRotate


@ti.kernel
def copy_values(psize: ti.types.ndarray(), calLength: ti.template(), particle: ti.template()):
    for i in range(psize.shape[0]):
        psize[i, :] = calLength[int(particle[i].bodyID)]


@ti.func
def get_cell_index(cell_size, pos):
    index = ti.floor(pos / cell_size, int)
    return index


@ti.func
def get_cell_id(idx, idy, idz, cell_num):
    return int(idx + idy * cell_num[0] + idz * cell_num[0] * cell_num[1])


@ti.func
def get_mpm_particle_radius(np, particle: ti.template(), psize: ti.types.ndarray()):
    radius2 = 0.0
    if ti.static(hasattr(particle, "rad")):
        radius2 = particle[np].rad * particle[np].rad
    else:
        for d in ti.static(range(3)):
            radius2 += psize[np, d] * psize[np, d]
    return ti.sqrt(radius2)


@ti.kernel
def insert_to_cell(
    dem_start_number: int,
    dem_end_number: int,
    cell_size: float,
    cell_num: ti.types.vector(3, int),
    particle: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
):
    for np in range(dem_start_number, dem_end_number):
        cell_index = get_cell_index(cell_size, particle[np].x)
        cell_id = get_cell_id(cell_index[0], cell_index[1], cell_index[2], cell_num)
        particle_num_in_cell = ti.atomic_add(num_particle_in_cell[cell_id], 1)
        particle_neighbor[cell_id, particle_num_in_cell] = np


@ti.kernel
def insert_wall_to_cell(
    dem_start_number: int,
    dem_end_number: int,
    cell_size: float,
    cell_num: ti.types.vector(3, int),
    wall: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
):
    for np in range(dem_start_number, dem_end_number):
        cell_index = get_cell_index(cell_size, wall[np]._get_center())
        cell_id = get_cell_id(cell_index[0], cell_index[1], cell_index[2], cell_num)
        particle_num_in_cell = ti.atomic_add(num_particle_in_cell[cell_id], 1)
        particle_neighbor[cell_id, particle_num_in_cell] = np


@ti.kernel
def find_overlap(
    mpm_start_number: int,
    mpm_end_number: int,
    cell_size: float,
    cell_num: ti.types.vector(3, int),
    mpm_particle: ti.template(),
    mpm_psize: ti.types.ndarray(),
    dem_particle: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
):
    for np in range(mpm_start_number, mpm_end_number):
        mpm_particle[np].active = ti.u8(0)

    for np in range(mpm_start_number, mpm_end_number):
        position = mpm_particle[np].x
        radius = get_mpm_particle_radius(np, mpm_particle, mpm_psize)
        cell_index = get_cell_index(cell_size, position)

        is_overlap = 0
        x_begin = ti.cast(ti.max(cell_index[0] - 1, 0), int)
        x_end = ti.cast(ti.min(cell_index[0] + 2, cell_num[0]), int)
        y_begin = ti.cast(ti.max(cell_index[1] - 1, 0), int)
        y_end = ti.cast(ti.min(cell_index[1] + 2, cell_num[1]), int)
        z_begin = ti.cast(ti.max(cell_index[2] - 1, 0), int)
        z_end = ti.cast(ti.min(cell_index[2] + 2, cell_num[2]), int)
        for i, j, k in ti.ndrange((x_begin, x_end), (y_begin, y_end), (z_begin, z_end)):
            cell_id = get_cell_id(i, j, k, cell_num)
            for ndp in range(num_particle_in_cell[cell_id]):
                pos = dem_particle[particle_neighbor[cell_id, ndp]].x
                rad = dem_particle[particle_neighbor[cell_id, ndp]].rad
                if (position - pos).norm() - rad - radius < 0.0:
                    is_overlap = 1
                    break
            if is_overlap == 1:
                break

        if is_overlap == 0:
            mpm_particle[np].active = ti.u8(1)


@ti.kernel
def find_lsoverlap(
    mpm_start_number: int,
    mpm_end_number: int,
    cell_size: float,
    cell_num: ti.types.vector(3, int),
    mpm_particle: ti.template(),
    dem_particle: ti.template(),
    dem_rigid: ti.template(),
    dem_box: ti.template(),
    dem_grid: ti.template(),
    mpm_psize: ti.types.ndarray(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
):
    for np in range(mpm_start_number, mpm_end_number):
        mpm_particle[np].active = ti.u8(0)

    for np in range(mpm_start_number, mpm_end_number):
        position = mpm_particle[np].x
        radius = get_mpm_particle_radius(np, mpm_particle, mpm_psize)
        cell_index = get_cell_index(cell_size, position)

        is_overlap = 0
        x_begin = ti.cast(ti.max(cell_index[0] - 1, 0), int)
        x_end = ti.cast(ti.min(cell_index[0] + 2, cell_num[0]), int)
        y_begin = ti.cast(ti.max(cell_index[1] - 1, 0), int)
        y_end = ti.cast(ti.min(cell_index[1] + 2, cell_num[1]), int)
        z_begin = ti.cast(ti.max(cell_index[2] - 1, 0), int)
        z_end = ti.cast(ti.min(cell_index[2] + 2, cell_num[2]), int)
        for i, j, k in ti.ndrange((x_begin, x_end), (y_begin, y_end), (z_begin, z_end)):
            cell_id = get_cell_id(i, j, k, cell_num)
            for ndp in range(num_particle_in_cell[cell_id]):
                pID = particle_neighbor[cell_id, ndp]
                pos = dem_particle[pID].x
                rad = dem_particle[pID].rad
                mass_center = dem_rigid[pID].mass_center
                if (position - pos).norm() - rad - radius < 0.0:
                    rotation_matrix = SetToRotate(dem_rigid[pID].q).transpose()
                    local_pos = rotation_matrix @ (position - mass_center)
                    if not dem_box[pID]._in_box(local_pos):
                        continue
                    if dem_box[pID].distance(local_pos, dem_grid) < radius:
                        is_overlap = 1
                        break
            if is_overlap == 1:
                break

        if is_overlap == 0:
            mpm_particle[np].active = ti.u8(1)


@ti.kernel
def adaptive_boundary_radius(
    mpm_start_number: int,
    mpm_end_number: int,
    cell_size: float,
    cell_num: ti.types.vector(3, int),
    mpm_particle: ti.template(),
    mpm_psize: ti.types.ndarray(),
    wall: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
):
    for np in range(mpm_start_number, mpm_end_number):
        if mpm_particle[np].active == 0:
            continue
        position = mpm_particle[np].x
        cell_index = get_cell_index(cell_size, position)

        min_distance = 1e10
        x_begin = ti.cast(ti.max(cell_index[0] - 1, 0), int)
        x_end = ti.cast(ti.min(cell_index[0] + 2, cell_num[0]), int)
        y_begin = ti.cast(ti.max(cell_index[1] - 1, 0), int)
        y_end = ti.cast(ti.min(cell_index[1] + 2, cell_num[1]), int)
        z_begin = ti.cast(ti.max(cell_index[2] - 1, 0), int)
        z_end = ti.cast(ti.min(cell_index[2] + 2, cell_num[2]), int)
        for i, j, k in ti.ndrange((x_begin, x_end), (y_begin, y_end), (z_begin, z_end)):
            cell_id = get_cell_id(i, j, k, cell_num)
            for ndp in range(num_particle_in_cell[cell_id]):
                wID = particle_neighbor[cell_id, ndp]
                distance = wall[wID]._get_norm_distance(position)
                if distance < min_distance:
                    min_distance = distance

        radius = get_mpm_particle_radius(np, mpm_particle, mpm_psize)
        if min_distance < 0.5 * radius:
            mpm_particle[np].active = ti.u8(0)
        elif min_distance < radius:
            if ti.static(hasattr(mpm_particle, "rad")):
                mpm_particle[np].rad = min_distance
