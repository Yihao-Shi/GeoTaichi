import taichi as ti

from src.utils.constants import Threshold
from src.utils.TypeDefination import vec3i

class BruteSearch(object):
    def __init__(self) -> None:
        self.snode_tree = None
        self.position = None
        self.radius = None
        self.num_particle_in_cell = None
        self.particle_neighbor = None
        self.destroy = False

    def clear(self):
        self.destroy = True
        self.snode_tree.destroy()
        del self.position, self.radius

    def neighbor_init(self, neighbor_size):
        self.cell_size = 0.
        self.cell_num = vec3i(0, 0, 0)
        field_bulider = ti.FieldsBuilder()
        self.position = ti.Vector.field(3, float)
        self.radius = ti.field(float)
        field_bulider.dense(ti.i, neighbor_size).place(self.position, self.radius)
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
        if int(particle.multisphereIndex[index]) < 0:
            pos = particle[index].x
            rad = particle[index].rad
            if check_in_region(pos, rad):
                pre_insert_particle(start_point, pos, rad, offset)


@ti.kernel
def pre_neighbor_clump(bodyNum: int, offset: ti.template(), particle: ti.template(), clump: ti.template(), check_in_region: ti.template(), start_point: ti.types.vector(3, float)):
    for nb in range(bodyNum):
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
                pre_insert_particle(start_point, position, radius, offset)

@ti.kernel
def pre_neighbor_bounding_sphere(rigidNum: int, offset: ti.template(), bounding_sphere: ti.template(), check_in_region: ti.template(), start_point: ti.types.vector(3, float)):
    for nb in range(rigidNum):
        pos = bounding_sphere[nb].x
        rad = bounding_sphere[nb].rad
        if check_in_region(pos, rad):
            pre_insert_particle(start_point, pos, rad, offset)

@ti.func
def pre_insert_particle(cell_num, cell_size, pos0, pos, rad, offset, position, radius, num_particle_in_cell, particle_neighbor):
    particle_number = ti.atomic_add(offset[None], 1)
    position[particle_number] = pos
    radius[particle_number] = rad


@ti.func
def insert_particle(cell_num, cell_size, pos, rad, offset, position, radius, num_particle_in_cell, particle_neighbor):
    position[offset[None]] = pos
    radius[offset[None]] = rad
    offset[None] += 1


@ti.func
def overlap(cell_num, cell_size, pos, rad, offset, position, radius, num_particle_in_cell, particle_neighbor):
    isoverlap = 0
    for np in range(offset[None]):
        dist = (position[np] - pos).norm()
        delta = -dist + (radius[np] + rad)
        if delta > Threshold:
            isoverlap = 1
            break
    return isoverlap
