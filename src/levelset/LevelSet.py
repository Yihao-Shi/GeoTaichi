import taichi as ti

from src.utils.constants import MThreshold
from src.utils.ScalarFunction import sgn
from src.utils.TypeDefination import vec2i, vec3i
from src.levelset.WENO import *
from src.levelset.BoundaryConditions import *


@ti.data_oriented
class LevelSet:
    def __init__(self, dimension, grid_size, resolution, ghost_cell=0, local_band=None, visualize=False, place=True):
        self.ghost_cell = ghost_cell
        self.dimension = dimension
        self.begins = [-self.ghost_cell for _ in range(dimension)]
        self.ends = [resolution[_] + self.ghost_cell for _ in range(dimension)]
        self.grid_size = grid_size
        self.inv_dx = 1. / grid_size
        self.local_band = local_band

        if visualize:
            self.phi_smooth = ti.field(float, shape=[resolution[i] + 1 for i in range(dimension)])

        total_cell = [resolution[_] + 2 * self.ghost_cell for _ in range(dimension)]
        self.valid = ti.field(dtype=int) 
        self.distance_field = ti.field(dtype=float)
        self.distance_field_temp = ti.field(dtype=float)
        if place:
            index = ti.i 
            if dimension == 3:
                index = ti.ijk
            elif dimension == 2: index = ti.ij
            offset = [-self.ghost_cell] * dimension
            ti.root.dense(index, total_cell).place(self.valid, self.distance_field, self.distance_field_temp, offset=offset)

    @ti.kernel
    def target_surface(self):
        for I in ti.grouped(self.distance_field):
            sign_change = False
            est = ti.cast(MThreshold, float)
            for k in ti.static(range(self.dimension)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(self.dimension, k) * s
                    I1 = I + offset
                    if I1[k] >= self.begins[k] and I1[k] < self.ends[k]:
                        if self.distance_field[I] * self.distance_field[I1] < 0.:
                            theta = self.distance_field[I] / (self.distance_field[I] - self.distance_field[I1])
                            est0 = sgn(self.distance_field[I]) * theta * self.grid_size
                            est = est0 if ti.abs(est0) < ti.abs(est) else est
                            sign_change = True
            # TODO: LOCAL BAND
            if sign_change:
                self.valid[I] = 0
            self.distance_field_temp[I] = est

    @ti.func
    def update_from_neighbor(self, I):
        # reference: Improved Fast Iterative Algorithm for Eikonal Equation for GPU Computing
        # solve the Eikonal equation
        nb = ti.Vector.zero(float, self.dimension)
        for k in ti.static(range(self.dimension)):
            offset = ti.Vector.unit(self.dimension, k)
            if I[k] == self.begins[k] or (I[k] < self.ends[k] - 1 and ti.abs(self.distance_field_temp[I + offset]) < ti.abs(self.distance_field_temp[I - offset])): 
                nb[k] = ti.abs(self.distance_field_temp[I + offset])
            else: 
                nb[k] = ti.abs(self.distance_field_temp[I - offset])

        d = 0.
        if ti.static(self.dimension == 2):
            if abs(nb[0] - nb[1]) > ti.sqrt(2) * self.grid_size:
                d = ti.min(nb[0], nb[1]) + self.grid_size
            else:
                d = 0.5 * (nb[0] + nb[1] + ti.sqrt(2 * self.grid_size * self.grid_size - (nb[1] - nb[0]) * (nb[1] - nb[0])))
        elif ti.static(self.dimension == 3):
            # sort
            for i in ti.static(range(self.dimension - 1)):
                for j in ti.static(range(self.dimension - 1 - i)):
                    if nb[j] > nb[j + 1]: 
                        nb[j], nb[j + 1] = nb[j + 1], nb[j]

            if ti.abs(nb[0] - nb[2]) < self.grid_size:
                sumnb = nb[0] + nb[1] + nb[2]
                d = 1./3. * (sumnb + ti.sqrt(ti.max(0, sumnb * sumnb - 3 * (nb[0] * nb[0] + nb[1] * nb[1] + nb[2] * nb[2] - self.grid_size * self.grid_size))))
            elif ti.abs(nb[0] - nb[1]) < self.grid_size:
                d = 0.5 * (nb[0] + nb[1] + ti.sqrt(2 * self.grid_size * self.grid_size - (nb[1] - nb[0]) * (nb[1] - nb[0])))
            else:
                d = ti.min(nb[0], nb[1]) + self.grid_size
        return d

    @ti.kernel
    def smoothing(self, phi : ti.template(), phi_temp : ti.template()):
        for I in ti.grouped(phi_temp):
            phi_avg = ti.cast(0, float)
            total_cell = ti.cast(0, int)
            for k in ti.static(range(self.dimension)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(self.dimension, k) * s
                    I1 = I + offset
                    if I1[k] >= self.begins[k] and I1[k] < self.ends[k]:
                        phi_avg += phi_temp[I1]
                        total_cell += 1
            phi_avg /= total_cell
            phi[I] = phi_avg if phi_avg < phi_temp[I] else phi_temp[I]

    @ti.kernel
    def postphi(self):
        for I in ti.grouped(self.phi_smooth):
            phi_avg = ti.cast(0, float)
            total_cell = ti.cast(0, int)
            if ti.static(self.dimension == 2):
                for i, j in ti.static(ti.ndrange(2, 2)):
                    I1 = I - vec2i(i, j)
                    if all(self.begins <= I1 < self.ends):
                        phi_avg += self.distance_field[I1]
                        total_cell += 1
                phi_avg /= total_cell
                self.phi_smooth[I] = phi_avg
            elif ti.static(self.dimension == 3):
                for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
                    I1 = I - vec3i(i, j, k)
                    if all(self.begins <= I1 < self.ends):
                        phi_avg += self.distance_field[I1]
                        total_cell += 1
                phi_avg /= total_cell
                self.phi_smooth[I] = phi_avg
    
    def smooth(self):
        self.smoothing(self.distance_field, self.distance_field_temp)
        self.smoothing(self.distance_field_temp, self.distance_field)
        self.smoothing(self.distance_field, self.distance_field_temp)

    def visualize(self):
        self.phi_smooth.fill(0)
        self.postphi()

    def redistance(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError
