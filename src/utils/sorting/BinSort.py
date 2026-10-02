import taichi as ti
from functools import reduce

from src.utils.PrefixSum import PrefixSumExecutor
from src.utils.ScalarFunction import linearize
import src.utils.GlobalVariable as GlobalVariable


from src.utils.sorting.BinSortKernel import (
    fill_object_bin,
    initialize_bin_cursor,
    object_sorted,
    fill_object_bin_condition,
    object_sorted_condition,
)


class BinSort(object):
    def __init__(self, cell_size, cell_num, max_object_num):
        self.cell_size = cell_size
        self.icell_size = 1.0 / cell_size
        self.cell_num = cell_num
        self.cellSum = reduce((lambda x, y: int(max(1, x) * max(1, y))), list(cell_num))

        self.cell_pse = PrefixSumExecutor(self.cellSum + 1)
        self.bin_count = ti.field(int, shape=self.cell_pse.get_length())
        self.bin_cursor = ti.field(int, shape=self.cell_pse.get_length())
        self.object_list = ti.field(int, shape=max_object_num)

    def run(self, current_object_num: int, position: ti.template()):
        fill_object_bin(
            current_object_num,
            self.icell_size,
            self.cell_num,
            position,
            self.bin_count,
        )
        self.cell_pse.run(self.bin_count)
        initialize_bin_cursor(self.cellSum, self.bin_count, self.bin_cursor)
        object_sorted(
            current_object_num,
            self.icell_size,
            self.cell_num,
            position,
            self.bin_cursor,
            self.object_list,
        )

    def run_with_condition(self, current_object_num: int, position: ti.template(), condition: ti.template()):
        fill_object_bin_condition(
            current_object_num,
            self.icell_size,
            self.cell_num,
            condition,
            position,
            self.bin_count,
        )
        self.cell_pse.run(self.bin_count)
        initialize_bin_cursor(self.cellSum, self.bin_count, self.bin_cursor)
        object_sorted_condition(
            current_object_num,
            self.icell_size,
            self.cell_num,
            condition,
            position,
            self.bin_cursor,
            self.object_list,
        )
