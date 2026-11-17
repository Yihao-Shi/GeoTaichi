import taichi as ti
from src.levelset.HighOrderInterpolation import *
from src.levelset.FastSweepingLevelSet import FastSweepingLevelSet


@ti.data_oriented
class SemiLagrangian(FastSweepingLevelSet):
    def __init__(
        self,
        dimension,
        grid_size,
        resolution,
        ghost_cell=0,
        order=2,
        runge_kutta=2,
        iteration_num=2,
        local_band=None,
        grid_type="SemiStaggered",
        visualize=False,
    ) -> None:
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
        p0 = pos - dt * (2.0 / 9.0 * midvel1 + 1.0 / 3.0 * midvel2 + 4.0 / 9.0 * midvel3)
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
        p0 = pos - 1.0 / 6.0 * dt * (midvel1 + 2.0 * midvel2 + 2.0 * midvel3 + midvel4)
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
    def interpolation(self, data, pos_index):
        # static unfold for efficiency
        if ti.static(len(data.shape) == 1):
            if ti.static(self.order == 2):
                return linear1d(data, pos_index)
            elif ti.static(self.order == 3):
                return weno31d(data, pos_index)
            elif ti.static(self.order == 4):
                return weno41d(data, pos_index)
        if ti.static(len(data.shape) == 2):
            if ti.static(self.order == 2):
                return bilinear(data, pos_index)
            elif ti.static(self.order == 3):
                return biweno3(data, pos_index)
            elif ti.static(self.order == 4):
                return biweno4(data, pos_index)
        elif ti.static(len(data.shape) == 3):
            if ti.static(self.order == 2):
                return trilinear(data, pos_index)
            elif ti.static(self.order == 3):
                return triweno3(data, pos_index)
            elif ti.static(self.order == 4):
                return triweno4(data, pos_index)

    @ti.kernel
    def advect_quantity(self, dt: ti.template(), grid_v: ti.template()):
        for I in ti.grouped(self.distance_field):
            self.advect(I, self.distance_field_temp, self.distance_field, grid_v, dt)

    def run(self, grid_v, dt):
        self.advect_quantity(dt, grid_v)
        self.distance_field.copy_from(self.distance_field_temp)
