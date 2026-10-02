import taichi as ti
from functools import reduce

from src.levelset.LevelSet import LevelSet
from src.utils.ScalarFunction import sgn
from src.utils.PriorityQueue import PriorityQueue


# J. Sethian. A fast marching level set method for monotonically ad- vancing fronts. Proc. Natl. Acad. Sci., 93:1591–1595, 1996.
@ti.data_oriented
class FastMarchingLevelSet(LevelSet):
    def __init__(self, dimension, grid_size, resolution, ghost_cell=0, local_band=None, visualize=False, place=True):
        super().__init__(dimension, grid_size, resolution, ghost_cell, local_band, visualize, place)

        self.priority_queue = PriorityQueue(dimension, resolution)
        self.surface_grid = ti.Vector.field(dimension, dtype=int, shape=reduce(lambda x, y: x * y, resolution))
        self.total_sg = ti.field(dtype=int, shape=())

    @ti.func
    def sg_to_pq(self):
        self.priority_queue.clear()
        cnt = 0
        while cnt < self.total_sg[None]:
            I = self.surface_grid[cnt]
            self.priority_queue.push(self.distance_field_temp[I], I)
            cnt += 1

    @ti.kernel
    def init_queue(self):
        self.total_sg[None] = 0
        for I in ti.grouped(self.valid):
            if self.valid[I] != -1:
                offset = ti.atomic_add(self.total_sg[None], 1)
                self.surface_grid[offset] = I
        self.sg_to_pq()

    @ti.kernel
    def propagate(self):
        while not self.priority_queue.empty():
            I0 = self.priority_queue.top()
            self.priority_queue.pop()

            for k in ti.static(range(self.dimension)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(self.dimension, k) * s
                    I = I0 + offset
                    if I[k] >= self.begins[k] and I[k] < self.ends[k] and self.valid[I] == -1:
                        d = self.update_from_neighbor(I)
                        if d < ti.abs(self.distance_field_temp[I]):
                            self.distance_field_temp[I] = d * sgn(self.distance_field[I0])
                        self.valid[I] = 0
                        self.priority_queue.push(ti.abs(self.distance_field_temp[I]), I)

    def redistance(self):
        self.valid.fill(-1)
        self.target_surface()
        self.init_queue()
        self.propagate()
        self.distance_field.copy_from(self.distance_field_temp)
