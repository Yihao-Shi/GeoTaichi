import taichi as ti

from src.levelset.FastSweepingLevelSet import FastSweepingLevelSet
from src.utils.ShapeFunctions import CubicSmooth
from src.utils.TypeDefination import vec2i, vec3i


@ti.data_oriented
class GeometryMethod(FastSweepingLevelSet):
    def __init__(self, dimension, grid_size, resolution, iteration_num=2, local_band=None, visualize=False):
        super().__init__(dimension, grid_size, resolution, iteration_num, ghost_cell=0, local_band=local_band, visualize=visualize)

    @ti.kernel
    def target_minus(self):
        for I in ti.grouped(self.distance_field):
            self.distance_field[I] -= (0.99 * self.grid_size) # the particle radius r (typically just a little less than the grid cell size grid_size)

        for I in ti.grouped(self.distance_field):
            sign_change = False
            for k in ti.static(range(self.dimension)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(self.dimension, k) * s
                    I1 = I + offset
                    if I1[k] >= self.begins[k] and I1[k] < self.ends[k]:
                        if self.distance_field[I] * self.distance_field[I1] < 0.:
                            sign_change = True

            if sign_change and self.distance_field[I] <= 0:
                self.valid[I] = 0
                self.distance_field_temp[I] = self.distance_field[I]
            elif self.distance_field[I] <= 0:
                self.distance_field_temp[I] = ti.cast(-1, float)
            else:
                self.distance_field_temp[I] = self.distance_field[I]
                #self.valid[I] = 0

    @ti.kernel
    def markers_propagate(self, markers: ti.template(), particle_count: ti.template(), particle_current: ti.template(), particleID: ti.template()):
        for cell_id in ti.grouped(self.distance_field):
            coords = (cell_id + 0.5) * self.grid_size
            smoothing_length = 1.33 * self.grid_size

            x_begin = ti.max(cell_id[0] - 1, self.begins[0])
            x_end = ti.min(cell_id[0] + 2, self.ends[0])
            y_begin = ti.max(cell_id[1] - 1, self.begins[1])
            y_end = ti.min(cell_id[1] + 2, self.ends[1])
            z_begin = ti.max(cell_id[2] - 1, self.begins[2])
            z_end = ti.min(cell_id[2] + 2, self.ends[2])
            
            weights, distance = 0., 0.
            positions = ti.Vector.zero(float, self.dimension)
            if ti.static(self.dimension == 2):
                for neigh_i in range(x_begin, x_end):
                    for neigh_j in range(y_begin, y_end):
                        cellID = cell_id + vec2i(neigh_i, neigh_j)
                        for hash_index in range(particle_count[cellID] - particle_current[cellID], particle_count[cellID]):
                            neighborID = particleID[hash_index]
                            rel_coord = markers[neighborID].x - coords
                            kernel_value = CubicSmooth(rel_coord, smoothing_length)
                            weights += kernel_value
                            distance += kernel_value * self.distance_field[cell_id]
                            positions += kernel_value * markers[neighborID].x
            elif ti.static(self.dimension == 3):
                for neigh_i in range(x_begin, x_end):
                    for neigh_j in range(y_begin, y_end):
                        for neigh_k in range(z_begin, z_end):
                            cellID = cell_id + vec3i(neigh_i, neigh_j, neigh_k)
                            for hash_index in range(particle_count[cellID] - particle_current[cellID], particle_count[cellID]):
                                neighborID = particleID[hash_index]
                                rel_coord = markers[neighborID].x - coords
                                kernel_value = CubicSmooth(rel_coord, smoothing_length)
                                weights += kernel_value
                                distance += kernel_value * self.distance_field[cell_id]
                                positions += kernel_value * markers[neighborID].x
            self.distance_field[cell_id] = distance - (positions - coords).norm()
        
    def run(self, particle, particle_count, particle_current, particleID):
        # reference: Adams, B., Pauly, M., Keiser, R., Guibas, L. 2007. Adaptively Sampled Particle Fluids. ACM Trans. Graph. 26, 3, Article 48 (July 2007), 7 pages
        self.distance_field.fill(1e20)
        self.markers_propagate(particle, particle_count, particle_current, particleID)
        self.valid.fill(-1)
        self.target_surface()
        self.propagate()