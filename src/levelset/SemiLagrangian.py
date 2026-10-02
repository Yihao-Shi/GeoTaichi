import taichi as ti

from src.levelset.HighOrderInterpolation import *
from src.levelset.FastSweepingLevelSet import FastSweepingLevelSet


@ti.data_oriented
class SemiLagrangian(FastSweepingLevelSet):
    def __init__(self, dimension, grid_size, resolution, ghost_cell=0, order=2, runge_kutta=2, iteration_num=2, local_band=None, grid_type="SemiStaggered", visualize=False) -> None:
        super().__init__(dimension, grid_size, resolution, iteration_num, ghost_cell, local_band, visualize)
        self.order = order
        self.runge_kutta = runge_kutta
        self.local_band = local_band
        if grid_type == "Staggered":
            self.velocity_interpolation = self.stagger_velocity_interpolation
        elif grid_type == "SemiStaggered":
            self.velocity_interpolation = self.interpolation

    @ti.func
    def backtrace_rk1(self, I, dst, src, v, dt):
        pos = (I + 0.5) * self.grid_size
        p0 = pos - dt * self.velocity_interpolation(v, pos * self.inv_dx)
        dst[I] = self.interpolation(src, p0 * self.inv_dx - 0.5)
        print(weno31d(src, p0 * self.inv_dx - 0.5), weno41d(src, p0 * self.inv_dx - 0.5))

    @ti.func
    def backtrace_rk2(self, I, dst, src, v, dt):
        pos = (I + 0.5) * self.grid_size
        midvel1 = self.velocity_interpolation(v, pos * self.inv_dx)
        midpos1 = pos - midvel1 * 0.5 * dt
        midvel2 = self.velocity_interpolation(v, midpos1 * self.inv_dx)
        p0 = pos - dt * midvel2
        dst[I] = self.interpolation(src, p0 * self.inv_dx - 0.5)

    @ti.func
    def backtrace_rk3(self, I, dst, src, v, dt):
        pos = (I + 0.5) * self.grid_size
        midvel1 = self.velocity_interpolation(v, pos * self.inv_dx)
        midpos1 = pos - midvel1 * 0.5 * dt
        midvel2 = self.velocity_interpolation(v, midpos1 * self.inv_dx)
        midpos2 = pos - midvel2 * 0.75 * dt
        midvel3 = self.velocity_interpolation(v, midpos2 * self.inv_dx)
        p0 = pos - dt * (2./9. * midvel1 + 1./3. * midvel2 + 4./9. * midvel3)
        dst[I] = self.interpolation(src, p0 * self.inv_dx - 0.5)

    @ti.func
    def backtrace_rk4(self, I, dst, src, v, dt):
        pos = (I + 0.5) * self.grid_size
        midvel1 = self.velocity_interpolation(v, pos * self.inv_dx)
        midpos1 = pos - midvel1 * 0.5 * dt
        midvel2 = self.velocity_interpolation(v, midpos1 * self.inv_dx)
        midpos2 = pos - midvel2 * 0.5 * dt
        midvel3 = self.velocity_interpolation(v, midpos2 * self.inv_dx)
        midpos3 = pos - midvel3 * dt
        midvel4 = self.velocity_interpolation(v, midpos3 * self.inv_dx)
        p0 = pos - 1./6. * dt * (midvel1 + 2. * midvel2 + 2. * midvel3 + midvel4)
        dst[I] = self.interpolation(src, p0 * self.inv_dx - 0.5)

    @ti.func
    def advect(self, I, dst, src, v, dt):
        if ti.static(self.runge_kutta == 1):
            self.backtrace_rk1(I, dst, src, v, dt)
        elif ti.static(self.runge_kutta == 2):
            self.backtrace_rk2(I, dst, src, v, dt)
        elif ti.static(self.runge_kutta == 3):
            self.backtrace_rk3(I, dst, src, v, dt)
        elif ti.static(self.runge_kutta == 4):
            self.backtrace_rk4(I, dst, src, v, dt)

    @ti.func
    def stagger_velocity_interpolation(self, mac, pos):
        v = ti.Vector.zero(float, self.dimension)
        for k in ti.static(range(self.dimension)):
            v[k] = self.interpolation(mac.velocity[k], pos - 0.5 * (1 - ti.Vector.unit(self.dimension, k)))
        return v
    
    @ti.func
    def interpolation(self, data, posIndex):
        # static unfold for efficiency
        if ti.static(len(data.shape) == 1):
            if ti.static(self.order == 2):
                return linear1d(data, posIndex)
            elif ti.static(self.order == 3):
                return weno31d(data, posIndex)
            elif ti.static(self.order == 4):
                return weno41d(data, posIndex)
        if ti.static(len(data.shape) == 2):
            if ti.static(self.order == 2):
                return bilinear(data, posIndex)
            elif ti.static(self.order == 3):
                return biweno3(data, posIndex)
            elif ti.static(self.order == 4):
                return biweno4(data, posIndex)
        elif ti.static(len(data.shape) == 3):
            if ti.static(self.order == 2):
                return trilinear(data, posIndex)
            elif ti.static(self.order == 3):
                return triweno3(data, posIndex)
            elif ti.static(self.order == 4):
                return triweno4(data, posIndex)

    @ti.kernel
    def advect_quantity(self, dt: ti.template(), grid_v: ti.template()):
        for I in ti.grouped(self.distance_field):
            self.advect(I, self.distance_field_temp, self.distance_field, grid_v, dt)

    def run(self, grid_v, dt):
        self.advect_quantity(dt, grid_v)
        self.distance_field.copy_from(self.distance_field_temp)
        #self.redistance()


@ti.data_oriented
class SemiLagrangianMultiGrid(SemiLagrangian):
    def __init__(self, dimension, grid_size, resolution, ghost_cell=0, order=2, runge_kutta=2, local_band=5, grid_type="Structured", visualize=False) -> None:
        super().__init__(dimension, grid_size, resolution, ghost_cell, visualize)
        self.order = order
        self.runge_kutta = runge_kutta
        if grid_type == "Staggered":
            self.velocity_interpolation = self.stagger_velocity_interpolation
        elif grid_type == "Structured":
            self.velocity_interpolation = self.interpolation

    @ti.func
    def backtrace_rk1(self, I, dst, src, v, dt):
        pos = (I + 0.5) * self.grid_size
        p0 = pos - dt * self.velocity_interpolation(v, pos * self.inv_dx)
        dst[I] = self.interpolation(src, p0 * self.inv_dx - 0.5)

    @ti.func
    def backtrace_rk2(self, I, dst, src, v, dt):
        pos = (I + 0.5) * self.grid_size
        midvel1 = self.velocity_interpolation(v, pos * self.inv_dx)
        midpos1 = pos - midvel1 * 0.5 * dt
        midvel2 = self.velocity_interpolation(v, midpos1 * self.inv_dx)
        p0 = pos - dt * midvel2
        dst[I] = self.interpolation(src, p0 * self.inv_dx - 0.5)

    @ti.func
    def backtrace_rk3(self, I, dst, src, v, dt):
        pos = (I + 0.5) * self.grid_size
        midvel1 = self.velocity_interpolation(v, pos * self.inv_dx)
        midpos1 = pos - midvel1 * 0.5 * dt
        midvel2 = self.velocity_interpolation(v, midpos1 * self.inv_dx)
        midpos2 = pos - midvel2 * 0.75 * dt
        midvel3 = self.velocity_interpolation(v, midpos2 * self.inv_dx)
        p0 = pos - dt * (2./9. * midvel1 + 1./3. * midvel2 + 4./9. * midvel3)
        dst[I] = self.interpolation(src, p0 * self.inv_dx - 0.5)

    @ti.func
    def backtrace_rk4(self, I, dst, src, v, dt):
        pos = (I + 0.5) * self.grid_size
        midvel1 = self.velocity_interpolation(v, pos * self.inv_dx)
        midpos1 = pos - midvel1 * 0.5 * dt
        midvel2 = self.velocity_interpolation(v, midpos1 * self.inv_dx)
        midpos2 = pos - midvel2 * 0.5 * dt
        midvel3 = self.velocity_interpolation(v, midpos2 * self.inv_dx)
        midpos3 = pos - midvel3 * dt
        midvel4 = self.velocity_interpolation(v, midpos3 * self.inv_dx)
        p0 = pos - 1./6. * dt * (midvel1 + 2. * midvel2 + 2. * midvel3 + midvel4)
        dst[I] = self.interpolation(src, p0 * self.inv_dx - 0.5)

    @ti.func
    def bilinear(self, data, posIndex):
        tot = data.shape
        i, ip = get_index_linear(posIndex[0], tot[0])
        j, jp = get_index_linear(posIndex[1], tot[1])
        weno1 = linear_uniform_grid(i, ip, posIndex[0], data[i, j], data[ip, j])
        weno2 = linear_uniform_grid(i, ip, posIndex[0], data[i, jp], data[ip, jp])
        return linear_uniform_grid(j, jp, posIndex[1], weno1, weno2)
    
    @ti.func
    def trilinear(self, data, posIndex):
        tot = data.shape
        i, ip = get_index_linear(posIndex[0], tot[0])
        j, jp = get_index_linear(posIndex[1], tot[1])
        k, kp = get_index_linear(posIndex[2], tot[2])
        weno11 = linear_uniform_grid(i, ip, posIndex[0], data[i, j, k], data[ip, j, k])
        weno12 = linear_uniform_grid(i, ip, posIndex[0], data[i, jp, k], data[ip, jp, k])
        weno21 = linear_uniform_grid(i, ip, posIndex[0], data[i, j, kp], data[ip, j, kp])
        weno22 = linear_uniform_grid(i, ip, posIndex[0], data[i, jp, kp], data[ip, jp, kp])
        weno1 = linear_uniform_grid(j, jp, posIndex[1], weno11, weno12)
        weno2 = linear_uniform_grid(j, jp, posIndex[1], weno21, weno22)
        return linear_uniform_grid(k, kp, posIndex[2], weno1, weno2)
    
    @ti.func
    def biweno3(self, data, posIndex):
        tot = data.shape
        im, i, ip = get_index_weno3(posIndex[0], tot[0])
        jm, j, jp = get_index_weno3(posIndex[1], tot[1])
        weno1 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jm], data[i, jm], data[ip, jm])
        weno2 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, j], data[i, j], data[ip, j])
        weno3 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jp], data[i, jp], data[ip, jp])
        return weno3_uniform_grid(jm, j, jp, posIndex[1], weno1, weno2, weno3)
    
    @ti.func
    def triweno3(self, data, posIndex):
        tot = data.shape
        im, i, ip = get_index_weno3(posIndex[0], tot[0])
        jm, j, jp = get_index_weno3(posIndex[1], tot[1])
        km, k, kp = get_index_weno3(posIndex[2], tot[2])
        weno11 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jm, km], data[i, jm, km], data[ip, jm, km])
        weno12 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, j, km], data[i, j, km], data[ip, j, km])
        weno13 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jp, km], data[i, jp, km], data[ip, jp, km])
        weno21 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jm, k], data[i, jm, k], data[ip, jm, k])
        weno22 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, j, k], data[i, j, k], data[ip, j, k])
        weno23 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jp, k], data[i, jp, k], data[ip, jp, k])
        weno31 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jm, kp], data[i, jm, kp], data[ip, jm, kp])
        weno32 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, j, kp], data[i, j, kp], data[ip, j, kp])
        weno33 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jp, kp], data[i, jp, kp], data[ip, jp, kp])
        weno1 = weno3_uniform_grid(jm, j, jp, posIndex[1], weno11, weno12, weno13)
        weno2 = weno3_uniform_grid(jm, j, jp, posIndex[1], weno21, weno22, weno23)
        weno3 = weno3_uniform_grid(jm, j, jp, posIndex[1], weno31, weno32, weno33)
        return weno4_uniform_grid(km, k, kp, posIndex[2], weno1, weno2, weno3)
    
    @ti.func
    def biweno4(self, data, posIndex):
        tot = data.shape
        im, i, ip, ipp = get_index_weno4(posIndex[0], tot[0])
        jm, j, jp, jpp = get_index_weno4(posIndex[1], tot[1])
        weno1 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jm], data[i, jm], data[ip, jm], data[ipp, jm])
        weno2 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, j], data[i, j], data[ip, j], data[ipp, j])
        weno3 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jp], data[i, jp], data[ip, jp], data[ipp, jp])
        weno4 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jpp], data[i, jpp], data[ip, jpp], data[ipp, jpp])
        return weno4_uniform_grid(jm, j, jp, jpp, posIndex[1], weno1, weno2, weno3, weno4)
    
    @ti.func
    def triweno4(self, data, posIndex):
        tot = data.shape
        im, i, ip, ipp = get_index_weno4(posIndex[0], tot[0])
        jm, j, jp, jpp = get_index_weno4(posIndex[1], tot[1])
        km, k, kp, kpp = get_index_weno4(posIndex[2], tot[2])
        weno11 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jm, km], data[i, jm, km], data[ip, jm, km], data[ipp, jm, km])
        weno12 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, j, km], data[i, j, km], data[ip, j, km], data[ipp, j, km])
        weno13 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jp, km], data[i, jp, km], data[ip, jp, km], data[ipp, jp, km])
        weno14 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jpp, km], data[i, jpp, km], data[ip, jpp, km], data[ipp, jpp, km])
        weno21 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jm, k], data[i, jm, k], data[ip, jm, k], data[ipp, jm, k])
        weno22 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, j, k], data[i, j, k], data[ip, j, k], data[ipp, j, k])
        weno23 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jp, k], data[i, jp, k], data[ip, jp, k], data[ipp, jp, k])
        weno24 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jpp, k], data[i, jpp, k], data[ip, jpp, k], data[ipp, jpp, k])
        weno31 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jm, kp], data[i, jm, kp], data[ip, jm, kp], data[ipp, jm, kp])
        weno32 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, j, kp], data[i, j, kp], data[ip, j, kp], data[ipp, j, kp])
        weno33 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jp, kp], data[i, jp, kp], data[ip, jp, kp], data[ipp, jp, kp])
        weno34 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jpp, kp], data[i, jpp, kp], data[ip, jpp, kp], data[ipp, jpp, kp])
        weno41 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jm, kpp], data[i, jm, kpp], data[ip, jm, kpp], data[ipp, jm, kpp])
        weno42 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, j, kpp], data[i, j, kpp], data[ip, j, kpp], data[ipp, j, kpp])
        weno43 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jp, kpp], data[i, jp, kpp], data[ip, jp, kpp], data[ipp, jp, kpp])
        weno44 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jpp, kpp], data[i, jpp, kpp], data[ip, jpp, kpp], data[ipp, jpp, kpp])
        weno1 = weno4_uniform_grid(jm, j, jp, jpp, posIndex[1], weno11, weno12, weno13, weno14)
        weno2 = weno4_uniform_grid(jm, j, jp, jpp, posIndex[1], weno21, weno22, weno23, weno24)
        weno3 = weno4_uniform_grid(jm, j, jp, jpp, posIndex[1], weno31, weno32, weno33, weno34)
        weno4 = weno4_uniform_grid(jm, j, jp, jpp, posIndex[1], weno41, weno42, weno43, weno44)
        return weno4_uniform_grid(km, k, kp, kpp, posIndex[2], weno1, weno2, weno3, weno4)

    @ti.func
    def advect(self, I, dst, src, v, dt):
        if ti.static(self.runge_kutta == 1):
            self.backtrace_rk1(I, dst, src, v, dt)
        elif ti.static(self.runge_kutta == 2):
            self.backtrace_rk2(I, dst, src, v, dt)
        elif ti.static(self.runge_kutta == 3):
            self.backtrace_rk3(I, dst, src, v, dt)
        elif ti.static(self.runge_kutta == 4):
            self.backtrace_rk4(I, dst, src, v, dt)

    @ti.func
    def stagger_velocity_interpolation(self, mac, pos):
        v = ti.Vector.zero(float, self.dimension)
        for k in ti.static(range(self.dimension)):
            v[k] = self.interpolation(mac.velocity[k], pos - 0.5 * (1 - ti.Vector.unit(self.dimension, k)))
        return v
    
    @ti.func
    def interpolation(self, data, posIndex):
        # static unfold for efficiency
        if ti.static(len(data.shape) == 1):
            if ti.static(self.order == 2):
                return linear1d(data, posIndex)
            elif ti.static(self.order == 3):
                return weno31d(data, posIndex)
            elif ti.static(self.order == 4):
                return weno41d(data, posIndex)
        if ti.static(len(data.shape) == 2):
            if ti.static(self.order == 2):
                return bilinear(data, posIndex)
            elif ti.static(self.order == 3):
                return biweno3(data, posIndex)
            elif ti.static(self.order == 4):
                return biweno4(data, posIndex)
        elif ti.static(len(data.shape) == 3):
            if ti.static(self.order == 2):
                return trilinear(data, posIndex)
            elif ti.static(self.order == 3):
                return triweno3(data, posIndex)
            elif ti.static(self.order == 4):
                return triweno4(data, posIndex)

    @ti.kernel
    def advect_quantity(self, dt: ti.template(), grid: ti.template()):
        for I in ti.grouped(self.distance_field):
            if ti.static(self.local_band != None):
                if ti.abs(self.distance_field[I]) < self.local_band:
                    self.advect(I, self.distance_field_temp, self.distance_field, grid, dt[None])
            elif ti.static(self.local_band == None):
                self.advect(I, self.distance_field_temp, self.distance_field, grid, dt[None])

    def run(self, dt, grid):
        self.advect_quantity(dt, grid)
        self.distance_field.copy_from(self.distance_field_temp)
        #self.redistance()