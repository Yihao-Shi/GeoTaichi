import taichi as ti
from functools import reduce

from src.utils.PrefixSum import PrefixSumExecutor
from src.utils.ScalarFunction import linearize
import src.utils.GlobalVariable as GlobalVariable


class BinSort(object):
    def __init__(self, cell_size, cell_num, max_object_num):
        self.cell_size = cell_size
        self.icell_size = 1. / cell_size
        self.cell_num = cell_num
        self.cellSum = reduce((lambda x, y: int(max(1, x) * max(1, y))), list(cell_num))

        self.cell_pse = PrefixSumExecutor(self.cellSum + 1)
        self.bin_count = ti.field(int, shape=self.cell_pse.get_length())
        self.prefix_index = ti.field(int, shape=max_object_num)
        self.object_list = ti.field(int, shape=max_object_num)

    def run(self, current_object_num: int, position: ti.template()):
        fill_object_bin(current_object_num, self.icell_size, self.cell_num, position, self.bin_count, self.prefix_index)
        self.cell_pse.run(self.bin_count)
        object_sorted(current_object_num, self.icell_size, self.cell_num, position, self.bin_count, self.prefix_index, self.object_list)

    def run_with_condition(self, current_object_num: int, position: ti.template(), condition: ti.template()):
        fill_object_bin_condition(current_object_num, self.icell_size, self.cell_num, condition, position, self.bin_count, self.prefix_index)
        self.cell_pse.run(self.bin_count)
        object_sorted_condition(current_object_num, self.icell_size, self.cell_num, condition, position, self.bin_count, self.prefix_index, self.object_list)


@ti.kernel
def fill_object_bin(particleNum: int, igrid_size: float, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), position: ti.template(), particle_count: ti.template(), particle_current: ti.template()):
    particle_count.fill(0)
    ti.block_local(particle_count)
    for np in range(particleNum):  
        grid_idx = ti.floor(position[np] * igrid_size , int)
        cellID = linearize(grid_idx, cnum)
        particle_current[np] = ti.atomic_add(particle_count[cellID + 1], 1)
    
@ti.kernel
def object_sorted(particleNum: int, igrid_size: float, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), position: ti.template(), particle_count: ti.template(), particle_current: ti.template(), particleID: ti.template()):
    for np in range(particleNum):
        grid_idx = ti.floor(position[np] * igrid_size, int)
        cellID = linearize(grid_idx, cnum)
        grain_location = particle_count[cellID] + particle_current[np]
        particleID[grain_location] = np

@ti.kernel
def fill_object_bin_condition(particleNum: int, igrid_size: float, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), condition: ti.template(), position: ti.template(), particle_count: ti.template(), particle_current: ti.template()):
    particle_count.fill(0)
    ti.block_local(particle_count)
    for np in range(particleNum):  
        if int(condition[np]) == 1: continue
        grid_idx = ti.floor(position[np] * igrid_size , int)
        cellID = linearize(grid_idx[0], cnum)
        particle_current[np] = ti.atomic_add(particle_count[cellID + 1], 1)
    
@ti.kernel
def object_sorted_condition(particleNum: int, igrid_size: float, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), condition: ti.template(), position: ti.template(), particle_count: ti.template(), particle_current: ti.template(), particleID: ti.template()):
    for np in range(particleNum):
        if int(condition[np]) == 1: continue
        grid_idx = ti.floor(position[np] * igrid_size, int)
        cellID = linearize(grid_idx, cnum)
        grain_location = particle_count[cellID] + particle_current[np]
        particleID[grain_location] = np