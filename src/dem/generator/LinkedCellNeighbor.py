import taichi as ti
import math

from src.utils.constants import Threshold
from src.utils.TypeDefination import vec3i


class LinkedCell(object):
    def __init__(self) -> None:
        self.cell_num = vec3i(0, 0, 0)
        self.cell_size = 0.
        self.snode_tree = None
        self.position = None
        self.radius = None
        self.num_particle_in_cell = None
        self.particle_neighbor = None
        self.destroy = False
        
    def clear(self):
        self.destroy = True
        self.snode_tree.destroy()
        del self.cell_num, self.cell_size, self.num_particle_in_cell, self.particle_neighbor, self.position, self.radius

    def neighbor_init(self, min_rad, max_rad, region_size, expected_particle_number):
        self.cell_size = 2 * max_rad
        ratio = math.ceil(max_rad / min_rad)
        particle_per_cell = max(ratio * ratio * ratio, 4)
        self.cell_num = vec3i(ti.ceil(region_size[0] / self.cell_size), ti.ceil(region_size[1] / self.cell_size), ti.ceil(region_size[2] / self.cell_size))
        total_cell = int(self.cell_num[0] * self.cell_num[1] * self.cell_num[2])
        
        field_bulider = ti.FieldsBuilder()
        self.num_particle_in_cell = ti.field(int)
        self.particle_neighbor = ti.field(int)
        self.position = ti.Vector.field(3, float)
        self.radius = ti.field(float)
        field_bulider.dense(ti.i, total_cell).place(self.num_particle_in_cell)
        field_bulider.dense(ti.ij, (total_cell, particle_per_cell)).place(self.particle_neighbor)
        field_bulider.dense(ti.i, expected_particle_number).place(self.position, self.radius)
        self.snode_tree = field_bulider.finalize()
 
        self.pre_neighbor_sphere = pre_neighbor_sphere
        self.pre_neighbor_clump = pre_neighbor_clump
        self.pre_neighbor_bounding_sphere = pre_neighbor_bounding_sphere
        self.pre_insert_particle = pre_insert_particle
        self.insert_particle = insert_particle
        self.overlap = overlap

@ti.kernel
def pre_neighbor_sphere(bodyNum: int, offset: ti.template(), particle: ti.template(), sphere: ti.template(), check_in_region: ti.template(), start_point: ti.types.vector(3, float)):
    for nb in range(bodyNum):
        index = sphere[nb].sphereIndex
        position = particle[index].x
        radius = particle[index].rad
        if check_in_region(position, radius):
            pre_insert_particle(start_point, position, radius, offset)

@ti.kernel
def pre_neighbor_clump(bodyNum: int, offset: ti.template(), particle: ti.template(), clump: ti.template(), check_in_region: ti.template(), start_point: ti.types.vector(3, float)):
    for nb in range(bodyNum):
        start, end = clump[nb].startIndex, clump[nb].endIndex
        is_in_region = 1
        for npebble in range(start, end):
            position = particle.x[npebble]
            radius = particle[npebble].rad
            if not check_in_region(position, radius):
                is_in_region = 0
                break
        if is_in_region:
            for npebble in range(start, end):
                position = particle[npebble].x
                radius = particle[npebble].rad
                pre_insert_particle(start_point, position, radius, offset)

@ti.kernel
def pre_neighbor_bounding_sphere(bodyNum: int, offset: ti.template(), bounding_sphere: ti.template(), check_in_region: ti.template(), start_point: ti.types.vector(3, float)):
    for nb in range(bodyNum):
        position = bounding_sphere[nb].x
        radius = bounding_sphere[nb].rad
        if check_in_region(position, radius):
            pre_insert_particle(start_point, position, radius, offset)

@ti.func
def get_cell_index(cell_size, pos):
    index = ti.floor(pos / cell_size, int)
    return index

@ti.func
def get_cell_id(idx, idy, idz, cell_num):
    return int(idx + idy * cell_num[0] + idz * cell_num[0] * cell_num[1])

@ti.func
def pre_insert_particle(cell_num, cell_size, pos0, pos, rad, offset, position, radius, num_particle_in_cell, particle_neighbor):
    cell_index = get_cell_index(cell_size, pos - pos0)
    cell_id = get_cell_id(cell_index[0], cell_index[1], cell_index[2], cell_num)
    particle_number = ti.atomic_add(offset[None], 1)
    position[particle_number] = pos
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
            if delta > Threshold:
                isoverlap = 1
                break
        if isoverlap == 1:
            break
    return isoverlap