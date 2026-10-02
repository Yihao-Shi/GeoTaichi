import taichi as ti

from src.utils.TypeDefination import vec3i

from src.dem.generator.BrustNeighborKernel import (
    kernel_pre_neighbor_sphere,
    kernel_pre_neighbor_clump,
    kernel_pre_neighbor_bounding_sphere,
    pre_insert_particle,
    pre_insert_rigid_particle,
    insert_particle,
    insert_rigid_particle,
    overlap,
    overlap_rigid,
)


class BruteSearch(object):
    def __init__(self, rigid=False) -> None:
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
        del self.position, self.radius

    def neighbor_init(self, neighbor_size):
        self.cell_size = 0.0
        self.cell_num = vec3i(0, 0, 0)
        field_bulider = ti.FieldsBuilder()
        self.position = ti.Vector.field(3, float)
        self.radius = ti.field(float)
        if self.rigid:
            self.orient = ti.Vector.field(3, float)
            field_bulider.dense(ti.i, neighbor_size).place(self.position, self.radius, self.orient)
        else:
            field_bulider.dense(ti.i, neighbor_size).place(self.position, self.radius)
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
            bodyNum, offset, particle, sphere, self.position, self.radius, check_in_region, start_point
        )

    def pre_neighbor_clump(self, bodyNum, offset, particle, clump, check_in_region, start_point):
        kernel_pre_neighbor_clump(
            bodyNum, offset, particle, clump, self.position, self.radius, check_in_region, start_point
        )

    def pre_neighbor_bounding_sphere(self, bodyNum, offset, bounding_sphere, check_in_region, start_point):
        kernel_pre_neighbor_bounding_sphere(
            bodyNum, offset, bounding_sphere, self.position, self.radius, check_in_region, start_point
        )
