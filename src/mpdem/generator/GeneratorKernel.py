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


@ti.kernel
def insert_to_cell(dem_start_number: int, dem_end_number: int, cell_size: float, cell_num: ti.types.vector(3, int), particle: ti.template(), num_particle_in_cell: ti.template(), particle_neighbor: ti.template()):
    for np in range(dem_start_number, dem_end_number):
        cell_index = get_cell_index(cell_size, particle[np].x)
        cell_id = get_cell_id(cell_index[0], cell_index[1], cell_index[2], cell_num)
        particle_num_in_cell = ti.atomic_add(num_particle_in_cell[cell_id], 1)
        particle_neighbor[cell_id, particle_num_in_cell] = np


@ti.kernel
def find_overlap(mpm_start_number: int, mpm_end_number: int, cell_size: float, cell_num: ti.types.vector(3, int), mpm_particle: ti.template(), dem_particle: ti.template(), num_particle_in_cell: ti.template(), particle_neighbor: ti.template()):
    for np in range(mpm_start_number, mpm_end_number):
        mpm_particle[np].active = ti.u8(0)
        
    for np in range(mpm_start_number, mpm_end_number):
        position = mpm_particle[np].x
        radius = mpm_particle[np].rad
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
                if (position - pos).norm() - rad - radius < 0.:
                    is_overlap = 1
                    break
            if is_overlap == 1:
                break

        if is_overlap == 0:
            mpm_particle[np].active = ti.u8(1)


@ti.kernel
def find_lsoverlap(mpm_start_number: int, mpm_end_number: int, cell_size: float, cell_num: ti.types.vector(3, int), mpm_particle: ti.template(), dem_particle: ti.template(), dem_rigid: ti.template(), dem_box: ti.template(),
                   dem_grid: ti.template(), num_particle_in_cell: ti.template(), particle_neighbor: ti.template()):
    for np in range(mpm_start_number, mpm_end_number):
        mpm_particle[np].active = ti.u8(0)
        
    for np in range(mpm_start_number, mpm_end_number):
        position = mpm_particle[np].x
        radius = mpm_particle[np].rad
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
                if (position - pos).norm() - rad - radius < 0.:
                    rotation_matrix = SetToRotate(dem_rigid[pID].q).transpose()
                    local_pos = rotation_matrix @ (position - pos)
                    if not dem_box[pID]._in_box(local_pos): continue
                    if dem_box[pID].distance(local_pos, dem_grid) < radius:
                        is_overlap = 1
                        break
            if is_overlap == 1:
                break

        if is_overlap == 0:
            mpm_particle[np].active = ti.u8(1)