import taichi as ti

from src.levelset.WENOFlux import *
from src.levelset.HighOrderInterpolation import *
from src.levelset.FastSweepingLevelSet import FastSweepingLevelSet
from src.utils.VectorFunction import Squared


# reference: GUANG-SHAN JIANG AND DANPING PENG. WEIGHTED ENO SCHEMES FOR HAMILTON–JACOBI EQUATIONS. SIAM J. SCI. COMPUT.
@ti.data_oriented
class FiniteDifference(FastSweepingLevelSet):
    def __init__(self, dimension, grid_size, resolution, order=4, runge_kutta=3, iteration_num=2, local_band=None, grid_type="SemiStaggered", visualize=False) -> None:
        pad = 3                         # int((order + 1) // 2)
        self.range_internal = [[0, resolution[d]] for d in range(dimension)]
        index = ti.i 
        if dimension == 3:
            index = ti.ijk
        elif dimension == 2: index = ti.ij
        voxel_size = [resolution[i] + 2 * pad for i in range(dimension)]
        offset = [-pad] * dimension
        super().__init__(dimension, grid_size, resolution, iteration_num, pad, local_band, visualize, place=False)

        if runge_kutta == 1:
            ti.root.dense(index, voxel_size).place(self.valid, self.distance_field, self.distance_field_temp, offset=offset)
            self.time_integration = self.TVDRK1
        else:
            self.phi_stage1 = ti.field(float)
            ti.root.dense(index, voxel_size).place(self.valid, self.distance_field, self.distance_field_temp, self.phi_stage1, offset=offset)
            if runge_kutta == 2:
                self.time_integration = self.TVDRK2
            elif runge_kutta == 3:
                self.time_integration = self.TVDRK3
        if grid_type == "Staggered":
            self.velocity_interpolation = self.stagger_velocity_interpolation
        elif grid_type == "SemiStaggered":
            self.velocity_interpolation = self.interpolation
        self.order = order

    @ti.kernel
    def next_step_rk1(self, y: ti.template(), dy: ti.template()):
        for i in ti.grouped(y):
            y[i] += dy[i] 

    @ti.kernel
    def next_step_rk2(self, y: ti.template(), y1: ti.template(), dy: ti.template()):
        for i in ti.grouped(y):
            y[i] += 0.5 * (y[i] + y1[i] + dy[i]) 

    @ti.kernel
    def next_step_rk31(self, y: ti.template(), y1: ti.template(), dy: ti.template()):
        for i in ti.grouped(y):
            y[i] = 0.75 * y1[i] + 0.25 * (y[i] + dy[i])

    @ti.kernel
    def next_step_rk32(self, y: ti.template(), y2: ti.template(), dy: ti.template()):
        for i in ti.grouped(y):
            y[i] = 1./3. * y[i] + 2./3. * (y2[i] + dy[i])

    @ti.kernel
    def rhs_reinitialization(self, dt: float, phi: ti.template(), reinit_rhs: ti.template()):
        for I in ti.grouped(ti.ndrange(*self.range_internal)):
            sdf_x = ti.Vector.zero(float, self.dimension)
            for d in ti.static(range(self.dimension)):
                sdf_x[d] = reinitialization_phi_x(d, I, self.inv_dx, phi)
            phi_cur = phi[I]
            sgn_phi = 0.
            sdf_x_norm = Squared(sdf_x)
            if ti.abs(phi_cur) > Threshold:
                sgn_phi = phi_cur / ti.sqrt(phi_cur * phi_cur + sdf_x_norm * self.grid_size * self.grid_size)
            reinit_rhs[I] = sgn_phi * (1. - ti.sqrt(sdf_x_norm)) * dt

    @ti.kernel
    def compuate_advection(self, dt: float, grid_v: ti.template(), phi: ti.template(), dphi: ti.template()):
        dphi.fill(0)
        for I in ti.grouped(ti.ndrange(*self.range_internal)):
            if ti.static(self.local_band != None):
                if ti.abs(self.distance_field[I]) < self.local_band:
                    vel = self.velocity_interpolation(grid_v, I + 0.5)
                    for d in ti.static(range(self.dimension)):
                        ComputeWENO5HJUpwind(d, I, self.inv_dx, dt, vel[d], phi, dphi)
            elif ti.static(self.local_band == None):
                vel = self.velocity_interpolation(grid_v, I)
                for d in ti.static(range(self.dimension)):
                    ComputeWENO5HJUpwind(d, I, self.inv_dx, dt, vel, phi, dphi)

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
        elif ti.static(len(data.shape) == 2):
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

    def TVDRK1(self, dt, grid_v):
        # first order total variation diminishing Runge Kutta
        self.compuate_advection(dt, grid_v, self.distance_field, self.distance_field_temp)
        self.next_step_rk1(self.distance_field, self.distance_field_temp)

    def TVDRK2(self, dt, grid_v):
        # second order total variation diminishing Runge Kutta
        self.compuate_advection(dt, grid_v, self.distance_field, self.distance_field_temp)
        self.next_step_rk1(self.distance_field, self.distance_field_temp)
        self.compuate_advection(dt, grid_v, self.phi_stage1, self.distance_field_temp)
        self.next_step_rk2(self.distance_field, self.phi_stage1, self.distance_field_temp)

    def TVDRK3(self, dt, grid_v):
        # third order total variation diminishing Runge Kutta
        self.compuate_advection(dt, grid_v, self.distance_field, self.distance_field_temp)
        self.next_step_rk1(self.phi_stage1, self.distance_field_temp)
        self.compuate_advection(dt, grid_v, self.phi_stage1, self.distance_field_temp)
        self.next_step_rk31(self.phi_stage1, self.distance_field, self.distance_field_temp)
        self.compuate_advection(dt, grid_v, self.distance_field, self.distance_field_temp)
        self.next_step_rk32(self.distance_field, self.phi_stage1, self.distance_field_temp)

    def run(self, grid_velocity, dt):
        self.time_integration(dt, grid_velocity)
        #self.redistance()