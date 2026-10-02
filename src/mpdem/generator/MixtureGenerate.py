import taichi as ti
import math

from src.dem.GenerateManager import GenerateManager as DEMGenerateManager
from src.dem.Simulation import Simulation as DEMSimulation
from src.dem.SceneManager import myScene as DEMScene
from src.mpdem.Simulation import Simulation
from src.mpdem.generator.GeneratorKernel import *
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
        
    def add_mixture(self, sims, dscene: DEMScene, mscene: MPMScene, dem_particle, mpm_body, dsims: DEMSimulation, msims):
        dem_start_number, dem_end_number = 0, int(dscene.particleNum[0])
        if dem_particle is not None:
            dem_start_number = int(dscene.particleNum[0])
            self.demGenerator.add_body(dem_particle, dsims, dscene)
            dem_end_number = int(dscene.particleNum[0])

        mpm_start_number, mpm_end_number = 0, int(mscene.particleNum[0])
        if mpm_body is not None:
            mpm_start_number = int(mscene.particleNum[0])
            self.mpmGenerator.add_body(mpm_body, msims, mscene)
            mpm_end_number = int(mscene.particleNum[0])

        if self.check_overlap:
            if dsims.scheme == "DEM":
                self.delete_overlap(sims, dscene, mscene, dem_start_number, dem_end_number, mpm_start_number, mpm_end_number)
            elif dsims.scheme == "LSDEM":
                self.delete_lsoverlap(sims, dscene, mscene, dem_start_number, dem_end_number, mpm_start_number, mpm_end_number)

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
        max_radius = max(dscene.find_bounding_sphere_max_radius(None), mscene.find_particle_max_radius())
        min_radius = min(dscene.find_bounding_sphere_min_radius(None), mscene.find_particle_min_radius())
        cell_size = 2 * max_radius
        ratio = math.ceil(max_radius / min_radius)
        particle_per_cell = ratio * ratio * ratio
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

    def delete_lsoverlap(self, sims: Simulation, dscene: DEMScene, mscene: MPMScene, dem_start_number, dem_end_number, mpm_start_number, mpm_end_number):
        max_radius = max(dscene.find_bounding_sphere_max_radius(None), mscene.find_particle_max_radius())
        min_radius = max(dscene.find_bounding_sphere_min_radius(None), mscene.find_particle_min_radius())
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
        find_lsoverlap(mpm_start_number, mpm_end_number, cell_size, cell_num, mscene.particle, dscene.particle, dscene.rigid, dscene.box, dscene.rigid_grid, num_particle_in_cell, particle_neighbor)
        snode_tree.destroy()
        mscene.check_overlap_coupling()

    


