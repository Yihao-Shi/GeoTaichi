import taichi as ti
import math

from src.utils.TypeDefination import vec3i

from src.dem.generator.LinkedCellNeighborKernel import (
    kernel_pre_neighbor_sphere,
    kernel_pre_neighbor_clump,
    kernel_pre_neighbor_bounding_sphere,
    get_cell_index,
    get_cell_id,
    pre_insert_particle,
    pre_insert_rigid_particle,
    insert_particle,
    insert_rigid_particle,
    overlap,
    overlap_rigid,
)


class LinkedCell(object):
    def __init__(self, rigid=False) -> None:
        self.cell_num = vec3i(0, 0, 0)
        self.cell_size = 0.0
        self.snode_tree = None
        self.position = None
        self.radius = None
        self.num_particle_in_cell = None
        self.particle_neighbor = None
        self.destroy = False
        self.rigid = rigid

    def clear(self):
        self.destroy = True
        self.snode_tree.destroy()
        del self.cell_num, self.cell_size, self.num_particle_in_cell, self.particle_neighbor, self.position, self.radius

    def neighbor_init(self, min_rad, max_rad, region_size, expected_particle_number):
        self.cell_size = 2 * max_rad
        ratio = math.ceil(max_rad / min_rad)
        particle_per_cell = max(ratio * ratio * ratio, 4)
        self.cell_num = vec3i(
            ti.ceil(region_size[0] / self.cell_size),
            ti.ceil(region_size[1] / self.cell_size),
            ti.ceil(region_size[2] / self.cell_size),
        )
        total_cell = int(self.cell_num[0] * self.cell_num[1] * self.cell_num[2])

        field_bulider = ti.FieldsBuilder()
        self.num_particle_in_cell = ti.field(int)
        self.particle_neighbor = ti.field(int)
        self.position = ti.Vector.field(3, float)
        self.radius = ti.field(float)
        field_bulider.dense(ti.i, total_cell).place(self.num_particle_in_cell)
        field_bulider.dense(ti.ij, (total_cell, particle_per_cell)).place(self.particle_neighbor)
        if self.rigid:
            self.orient = ti.Vector.field(3, float)
            field_bulider.dense(ti.i, expected_particle_number).place(self.position, self.radius, self.orient)
        else:
            field_bulider.dense(ti.i, expected_particle_number).place(self.position, self.radius)
        self.snode_tree = field_bulider.finalize()

        self.pre_insert_particle = pre_insert_particle
        if self.rigid:
            self.insert_particle = insert_rigid_particle
            self.overlap = overlap_rigid
        else:
            self.insert_particle = insert_particle
            self.overlap = overlap

    def pre_neighbor_sphere(self, bodyNum, offset, particle, sphere, check_in_region, start_point):
        kernel_pre_neighbor_sphere(
            bodyNum,
            offset,
            particle,
            sphere,
            self.cell_num,
            self.cell_size,
            self.position,
            self.radius,
            self.num_particle_in_cell,
            self.particle_neighbor,
            check_in_region,
            start_point,
        )

    def pre_neighbor_clump(self, bodyNum, offset, particle, clump, check_in_region, start_point):
        kernel_pre_neighbor_clump(
            bodyNum,
            offset,
            particle,
            clump,
            self.cell_num,
            self.cell_size,
            self.position,
            self.radius,
            self.num_particle_in_cell,
            self.particle_neighbor,
            check_in_region,
            start_point,
        )

    def pre_neighbor_bounding_sphere(self, bodyNum, offset, bounding_sphere, check_in_region, start_point):
        kernel_pre_neighbor_bounding_sphere(
            bodyNum,
            offset,
            bounding_sphere,
            self.cell_num,
            self.cell_size,
            self.position,
            self.radius,
            self.num_particle_in_cell,
            self.particle_neighbor,
            check_in_region,
            start_point,
        )
