import taichi as ti

from src.contact_detection.bounding_volume_hierarchy.AABB import AABB
from src.utils.BitFunction import morton3d32
from src.utils.sorting.RadixSort import RadixSort


@ti.data_oriented
class SpatialDivision:
    def __init__(self, primitive_count):
        self.primitive_count = primitive_count
        self.sort = RadixSort(primitive_count, dtype=int, val_col=1)
        self.primitive_pot = self.sort.pse.get_length()
        self.morton_code = ti.field(dtype=int, shape=(primitive_count, 3))
        self.leaf_count = ti.field(dtype=int, shape=1)
        self.aabb = AABB(primitive_count)

    @ti.kernel
    def build_morton_3d(self, primitive_count: int, domain: ti.types.vector(3, float)):
        for i in range(primitive_count):
            centre_p = self.aabb.get_center(i)
            norm_p = centre_p / domain
            self.sort.data_in[i, 0] = morton3d32(norm_p[0], norm_p[1], norm_p[2])
            self.sort.data_in[i, 1] = i
            self.morton_code[i, 0] = 0
            self.morton_code[i, 1] = 1
            self.morton_code[i, 2] = 0

    @ti.pyfunc
    def merge(self, max_prim):
        if max_prim > 1:
            self.merge_num_condition(max_prim)
        self.merge_equal_condition(max_prim)

        #scan
        # input     : 0 0 0 1 0 1 1 0 0 1
        # scan      : 0 0 0 1 1 2 3 3 3 4
        # scatter   ：3 2 1 3 1
        self.hillis_scan_for_merge_host()
        self.scatter_for_merge()

        #construct
        # motorn    : 3 4 5 6 7
        # scan      ：0 3 5 6 9  start index of prim
        # count     ：3 2 1 3 1  num of same prim
        self.hillis_scan_for_merge_host()
        self.merge_construct()

    @ti.kernel
    def merge_num_condition(self, max_prim:int):
        for i in self.sort.data_out:
            if i%max_prim == 0:
                self.sort.data_out[i, 0] = 1
                self.sort.data_out[i, 1] = 1
            else:
                self.sort.data_out[i, 0] = 0
                self.sort.data_out[i, 1] = 0

    @ti.kernel
    def merge_equal_condition(self, max_prim:int):
        for i in self.sort.data_out:
            if i == 0:
                self.sort.data_out[i, 0] = 0
                self.sort.data_out[i, 1] = 0
            else:
                if(self.sort.data_in[i, 0] == self.sort.data_in[i-1, 0]):
                    self.sort.data_out[i, 0] = 0
                    self.sort.data_out[i, 1] = 0
                elif max_prim == 1:
                    self.sort.data_out[i, 0] = 1
                    self.sort.data_out[i, 1] = 1

    @ti.pyfunc
    def hillis_scan_for_merge_host(self):
        mod = 1
        while mod < self.primitive_pot:
            self.hillis_scan_reduce_for_merge(mod)
            mod *= 2

    @ti.kernel
    def hillis_scan_reduce_for_merge(self, mod: int):
        for i in self.sort.data_out:
            if i + mod < self.primitive_count:
                self.sort.data_out[i + mod, 1] += self.sort.data_out[i, 0]

        for i in self.sort.data_out:
            self.sort.data_out[i, 0] = self.sort.data_out[i, 1]
    
    @ti.kernel
    def scatter_for_merge(self):
        for i in self.sort.data_out:
            pos = self.sort.data_out[i, 0]
            self.morton_code[pos, 1]  = 0
            self.morton_code[pos, 2]  += 1
            
        for i in self.sort.data_out:
            self.sort.data_out[i, 0] = self.morton_code[i, 2] 
            self.sort.data_out[i, 1] = self.morton_code[i, 2] 
            if self.morton_code[i, 2] > 0:
                self.leaf_count[0] += 1
   
    @ti.kernel
    def merge_construct(self):
        for i in self.sort.data_out:
            self.morton_code[i, 1] = self.sort.data_out[i, 0] - self.morton_code[i, 2]
            self.morton_code[i, 0] = self.sort.data_in[self.morton_code[i, 1], 0]