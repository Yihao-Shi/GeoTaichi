import taichi as ti

from src.levelset.LevelSet import LevelSet
from src.utils.ScalarFunction import sgn


# H. Zhao. A fast sweeping method for Eikonal equations. Math. Comp., 74:603–627, 2005.
@ti.data_oriented
class FastSweepingLevelSet(LevelSet):
    def __init__(self, dimension, grid_size, resolution, iteration_num=2, ghost_cell=0, local_band=None, visualize=False, place=True):
        super().__init__(dimension, grid_size, resolution, ghost_cell, local_band, visualize, place)
        self.repeat_times = iteration_num

    @ti.func
    def propagate_update(self, I):
        if self.valid[I] == -1:
            d = self.update_from_neighbor(I)
            if ti.abs(d) < ti.abs(self.distance_field_temp[I]): 
                self.distance_field_temp[I] = d * sgn(self.distance_field[I])

    @ti.kernel
    def propagate(self):
        if ti.static(self.dimension == 2):
            for _ in ti.static(range(self.repeat_times)):
                for i in range(self.begins[0], self.ends[0]):
                    for j in range(self.begins[1], self.ends[1]):
                        self.propagate_update([i, j])

                for i in range(self.begins[0], self.ends[0]):
                    for j in range(self.ends[1] - self.begins[1]):
                        self.propagate_update([i, self.ends[1] - 1 - j])
            
                for j in range(self.begins[1], self.ends[1]):
                    for i in range(self.begins[0], self.ends[0]):
                        self.propagate_update([i, j])

                for j in range(self.begins[1], self.ends[1]):
                    for i in range(self.ends[0] - self.begins[0]):
                        self.propagate_update([self.ends[0] - 1 - i, j])

        elif ti.static(self.dimension == 3):
            for _ in ti.static(range(self.repeat_times)):
                for i, j in ti.ndrange((self.begins[0], self.ends[0]), (self.begins[1], self.ends[1])):
                    for k in range(self.ends[2] - self.begins[2]):
                        self.propagate_update([i, j, k])

                for i, j in ti.ndrange((self.begins[0], self.ends[0]), (self.begins[1], self.ends[1])):
                    for k in range(self.ends[2] - self.begins[2]):
                        self.propagate_update([i, j, self.ends[2] - 1 - k])

                for i, k in ti.ndrange((self.begins[0], self.ends[0]), (self.begins[2], self.ends[2])):
                    for j in range(self.ends[1] - self.begins[1]):
                        self.propagate_update([i, j, k])

                for i, k in ti.ndrange((self.begins[0], self.ends[0]), (self.begins[2], self.ends[2])):
                    for j in range(self.ends[1] - self.begins[1]):
                        self.propagate_update([i, self.ends[1] - 1 - j, k])

                for j, k in ti.ndrange((self.begins[1], self.ends[1]), (self.begins[2], self.ends[2])):
                    for i in range(self.ends[0] - self.begins[0]):
                        self.propagate_update([i, j, k])

                for j, k in ti.ndrange((self.begins[1], self.ends[1]), (self.begins[2], self.ends[2])):
                    for i in range(self.ends[0] - self.begins[0]):
                        self.propagate_update([self.ends[0] - 1 - i, j, k])

    def redistance(self):
        self.valid.fill(-1)
        self.target_surface()
        self.propagate()
        self.distance_field.copy_from(self.distance_field_temp)
