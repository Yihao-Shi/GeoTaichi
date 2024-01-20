import taichi as ti
import math, os
import numpy as np

from src.dem.GenerateManager import GenerateManager as DEMGenerateManager
from src.dem.SceneManager import myScene as DEMScene
from src.mpdem.Simulation import Simulation
from src.mpm.GenerateManager import GenerateManager as MPMGenerateManager
from src.mpm.SceneManager import myScene as MPMScene
from src.utils.TypeDefination import vec3i


class MixtureGenerator(object):
    demGenerator: DEMGenerateManager
    mpmGenerator: MPMGenerateManager

    def __init__(self, mpm_gene, den_gene) -> None:
        self.demGenerator = den_gene
        self.mpmGenerator = mpm_gene
        self.check_overlap = False

    def set_essentials(self, check_overlap):
        self.check_overlap = check_overlap

    def read_files(self, dem_particle, mpm_body, dscene, mscene, dsims, msims):
        self.demGenerator.read_body_file(dem_particle, dsims, dscene)
        self.mpmGenerator.read_body_file(mpm_body, msims, mscene)
        
    def add_mixture(self, sims, dscene: DEMScene, mscene: MPMScene, dem_particle, mpm_body, dsims, msims):
        dem_start_number = int(dscene.particleNum[0])
        self.demGenerator.add_body(dem_particle, dsims, dscene)
        dem_end_number = int(dscene.particleNum[0])

        mpm_start_number = int(mscene.particleNum[0])
        self.mpmGenerator.add_body(mpm_body, msims, mscene)
        mpm_end_number = int(mscene.particleNum[0])

        if self.check_overlap:
            self.delete_overlap(sims, dscene, mscene, dem_start_number, dem_end_number, mpm_start_number, mpm_end_number)

    def regenerate(self, sims, mscene: MPMScene, dscene: DEMScene):
        dem_start_number = int(dscene.particleNum[0])
        dem_new_body = self.demGenerator.regenerate(dscene)
        dem_end_number = int(dscene.particleNum[0])

        mpm_start_number = int(mscene.particleNum[0])
        mpm_new_body = self.mpmGenerator.regenerate(mscene)
        mpm_end_number = int(mscene.particleNum[0])

        new_body = mpm_new_body or dem_new_body
        if new_body and self.check_overlap:
            self.delete_overlap(sims, dscene, mscene, dem_start_number, dem_end_number, mpm_start_number, mpm_end_number)
        return new_body

    def delete_overlap(self, sims: Simulation, dscene: DEMScene, mscene: MPMScene, dem_start_number, dem_end_number, mpm_start_number, mpm_end_number):
        max_radius = max(dscene.find_particle_max_radius(), mscene.find_particle_max_radius())
        min_radius = max(dscene.find_particle_min_radius(), mscene.find_particle_min_radius())
        cell_size = 2 * max_radius
        ratio = math.ceil(max_radius / min_radius)
        particle_per_cell = max(ratio * ratio * ratio, 4)
        cell_num = vec3i(math.ceil(sims.domain[0] / cell_size), math.ceil(sims.domain[1] / cell_size), math.ceil(sims.domain[2] / cell_size))
        total_cell = int(cell_num[0] * cell_num[1] * cell_num[2])

        field_bulider = ti.FieldsBuilder()
        num_particle_in_cell = ti.field(int)
        particle_neighbor = ti.field(int)
        field_bulider.dense(ti.i, total_cell).place(num_particle_in_cell)
        field_bulider.dense(ti.ij, (total_cell, particle_per_cell)).place(particle_neighbor)
        snode_tree = field_bulider.finalize()
        insert_to_cell(dem_start_number, dem_end_number, cell_size, cell_num, dscene.particle, num_particle_in_cell, particle_neighbor)
        find_overlap(mpm_start_number, mpm_end_number, cell_size, cell_num, mscene.particle, dscene.particle, num_particle_in_cell, particle_neighbor)
        snode_tree.destroy()
        mscene.check_overlap_coupling()


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


        