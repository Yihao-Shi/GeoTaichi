import taichi as ti
import numpy as np
import time


# apply force after grid normalization (or explosion)
# pressure in air cells should be 0 (or volume shrink quickly)
# velocity in air cells may not be 0

# FIX: adjusted the boundary response in handle_boundary() and advect_particles(), is seems to be more energetic now

# TODO: solve vorticity enhancement explosion problem


ti.init(arch=ti.gpu, default_fp=ti.f32)


res = 512
dt = 2e-2 #2e-3 #2e-2
substep = 1


rho = 1000
jacobi_iters = 500
jacobi_damped_para = 1



m_g = 128
n_grid = m_g*m_g
n_particle = 64*64*4

length = 10.0
dx = length/m_g
inv_dx = 1/dx

# solid boundary
boundary_width = 8

eps = 1e-5


use_multigrid = True

# show multi-grid types
# -1: disable debug
multigrid_debug_level = 1

multigrid_debug_descend_ratio = False


# MAC grid
velocities_u = ti.field(dtype=ti.f32, shape=(m_g+1, m_g))
velocities_v = ti.field(dtype=ti.f32, shape=(m_g, m_g+1))

weights_u = ti.field(dtype=ti.f32, shape=(m_g+1, m_g))
weights_v = ti.field(dtype=ti.f32, shape=(m_g, m_g+1))


pressures = ti.field(dtype=ti.f32, shape=(m_g, m_g))
new_pressures = ti.field(dtype=ti.f32, shape=(m_g, m_g))
divergences = ti.field(dtype=ti.f32, shape=(m_g, m_g))
vorticities = ti.field(dtype=ti.f32, shape=(m_g, m_g))
pressures_factor = ti.field(dtype=ti.f32, shape=(m_g, m_g))


FLUID = 0
AIR = 1
SOLID = 2

types = ti.field(dtype=ti.i32, shape=(m_g, m_g))

particle_velocity = ti.Vector.field(2, dtype=ti.f32, shape=n_particle)
particle_position = ti.Vector.field(2, dtype=ti.f32, shape=n_particle)



@ti.data_oriented
class MultiGridSolver:

    def grid_shape(self, l):
            return (self.n//(1<<l), self.n//(1<<l))
    
    def __init__(self, n, level = 4):
        self.n = n
        self.level = level


        self.r = [ti.field(dtype=ti.f32, shape=self.grid_shape(i)) for i in range(level)]
        self.x = [ti.field(dtype=ti.f32, shape=self.grid_shape(i)) for i in range(level)]
        self.diag = [ti.field(dtype=ti.f32, shape=self.grid_shape(i)) for i in range(level)]
        self.type = [ti.field(dtype=ti.i32, shape=self.grid_shape(i)) for i in range(level)]

        self.smooth_num = [12//(i+1) for i in range(level)]

        self.smooth_num[0] = 10

        self.smooth_num[self.level-1] = 100

    def clear(self):
        for l in range(self.level):
            self.x[l].fill(0)
            self.r[l].fill(0)
            self.diag[l].fill(0)
            self.type[l].fill(0)


    # have to change tabs to spaces when using @ti.kernel in @ti.data_oriented classes, otherwise taichi compiler will report a error
    @ti.kernel
    def init(self):
        for i, j in self.x[0]:
            self.x[0][i, j] = pressures[i, j]
            self.r[0][i, j] = divergences[i, j] * rho / dt * (dx*dx)
            self.diag[0][i, j] = pressures_factor[i, j]
            self.type[0][i, j] = types[i, j]


        l = 0
        for l in ti.static(range(1, self.level)):

            for i, j in self.type[l]:
                i2 = i * 2
                j2 = j * 2
                if self.type[l-1][i2, j2] == AIR or self.type[l-1][i2+1, j2] == AIR or self.type[l-1][i2, j2+1] == AIR or self.type[l-1][i2+1, j2+1] == AIR:
                    self.type[l][i, j] = AIR
                else:
                    if self.type[l-1][i2, j2] == FLUID or self.type[l-1][i2+1, j2] == FLUID or self.type[l-1][i2, j2+1] == FLUID or self.type[l-1][i2+1, j2+1] == FLUID:
                        self.type[l][i, j] = FLUID
                    else:
                        self.type[l][i, j] = SOLID


        for l in ti.static(range(self.level)):
            for i, j in self.diag[l]:
                if self.type[l][i, j] == FLUID:
   
                    k = 4
                    if self.type[l][i-1, j] == SOLID:
                        k -= 1
                    if self.type[l][i+1, j] == SOLID:
                        k -= 1
                    if self.type[l][i, j-1] == SOLID:
                        k -= 1
                    if self.type[l][i, j+1] == SOLID:
                        k -= 1

                    self.diag[l][i, j] = k


    @ti.kernel
    def residual(self, l:ti.template()) -> ti.f32:
        res = 0.0
        for i, j in self.x[l]:
            if self.type[l][i, j] == FLUID:
                x_c = self.x[l][i, j]
                x_l = self.x[l][i-1, j]
                x_r = self.x[l][i+1, j]
                x_d = self.x[l][i, j-1]
                x_u = self.x[l][i, j+1]
                b_hat = x_l + x_r + x_d + x_u - x_c * self.diag[l][i, j]
                b = self.r[l][i, j]
                res += abs(b - b_hat)
        return res


    @ti.kernel
    def smooth(self, l:ti.template(), phase:ti.i32):

        w = 1.43

        for i, j in self.x[l]:
            if (i + j) & 1 == phase and self.type[l][i, j] == FLUID:

                x_c = self.x[l][i, j]
                x_l = self.x[l][i-1, j]
                x_r = self.x[l][i+1, j]
                x_d = self.x[l][i, j-1]
                x_u = self.x[l][i, j+1]
                self.x[l][i, j] = (1-w) * self.x[l][i, j] + w * (x_l + x_r + x_d + x_u - self.r[l][i, j]) / self.diag[l][i, j]


    # @ti.kernel
    # def smooth_boundary(self, l:ti.template(), phase:ti.i32):
    #     for i, j in self.x[l]:
    #         if (i + j) & 1 == phase and self.type[l][i, j] == FLUID and (self.type[l][i-1, j] != FLUID or self.type[l][i+1, j] != FLUID or self.type[l][i, j-1] != FLUID or self.type[l][i, j+1] != FLUID):

    #             x_c = self.x[l][i, j]
    #             x_l = self.x[l][i-1, j]
    #             x_r = self.x[l][i+1, j]
    #             x_d = self.x[l][i, j-1]
    #             x_u = self.x[l][i, j+1]
    #             self.x[l][i, j] = (x_l + x_r + x_d + x_u - self.r[l][i, j]) / self.diag[l][i, j]

    @ti.kernel
    def restrict(self, l:ti.template()):
        for i, j in self.x[l]:
            if self.type[l][i, j] == FLUID:
                x_c = self.x[l][i, j]
                x_l = self.x[l][i-1, j]
                x_r = self.x[l][i+1, j]
                x_d = self.x[l][i, j-1]
                x_u = self.x[l][i, j+1]
                b_hat = x_l + x_r + x_d + x_u - x_c * self.diag[l][i, j]
                b = self.r[l][i, j]
                residual = b - b_hat
                self.r[l+1][i//2, j//2] += 0.25 * residual * 4


    @ti.kernel
    def prolongate(self, l:ti.template()):
        for i, j in self.x[l]:
            if self.type[l][i, j] == FLUID:
                self.x[l][i, j] += self.x[l+1][i//2, j//2]


    def solve(self, iter_num):
        print("-----------")
        las = 0.0
        for k in range(iter_num):
            if multigrid_debug_descend_ratio:
                print("iter %3d" % k)

            before = [0.0]*self.level
            after = [0.0]*self.level
            before_pro = [0.0]*self.level
            after_pro = [0.0]*self.level


            for l in range(self.level-1):
                if multigrid_debug_descend_ratio:
                    before[l] = abs(self.residual(l))
                
                # for i in range(3):
                #     self.smooth_boundary(l, 0)
                #     self.smooth_boundary(l, 1)

                for i in range(self.smooth_num[l]):
                    self.smooth(l, 0)
                    self.smooth(l, 1)

                # for i in range(3):
                #     self.smooth_boundary(l, 0)
                #     self.smooth_boundary(l, 1)

                self.r[l+1].fill(0)
                self.x[l+1].fill(0)
                self.restrict(l)

            if multigrid_debug_descend_ratio:
                before[self.level-1] = abs(self.residual(self.level-1))

            for i in range(self.smooth_num[self.level-1]):

                self.smooth(self.level-1, 0)
                self.smooth(self.level-1, 1)

            if multigrid_debug_descend_ratio:
                for l in range(self.level):
                    after[l] = abs(self.residual(l))


            for l in reversed(range(self.level-1)):
                # before = abs(self.residual(l))
                self.prolongate(l)
                before_pro[l] = abs(self.residual(l))

                # for i in range(3):
                #     self.smooth_boundary(l, 0)
                #     self.smooth_boundary(l, 1)

                for i in range(self.smooth_num[l]):
                    self.smooth(l, 0)
                    self.smooth(l, 1)

                # for i in range(3):
                #     self.smooth_boundary(l, 0)
                #     self.smooth_boundary(l, 1)


                after_pro[l] = abs(self.residual(l))
                # after = abs(self.residual(l))
                # if multigrid_debug_descend_ratio:
                #     print("iter %3d" % k, "level", l, "before %.6f"%before, "after %.6f"%after)
            
            if multigrid_debug_descend_ratio:
                for l in range(self.level):
                    print("level", l, "[before smooth %07.6f"%before[l], "] [smooth %07.6f"%after[l], "] [prolongate %07.6f"%before_pro[l], "] [smooth %07.6f"%after_pro[l], "]")

            cur = abs(self.residual(0))
            if multigrid_debug_descend_ratio and k > 0:
                # print("bottom level:", self.residual(self.level-1))
                # for l in range(self.level):
                #     print("level ", l, abs(self.residual(l)))
                print("iter %3d:" % k, "[residual: %.6f]" % cur, "[descent ratio: %.6f]" % (las/max(cur, 1e-5)))
            las = cur
            if cur < 1e-5:
                break
        
        pressures.copy_from(self.x[0])

mg_solver = MultiGridSolver(m_g, 1)






@ti.kernel
def init_grid():
    for i, j in types:
        if i < boundary_width or i >= m_g-boundary_width or j < boundary_width or j >= m_g-boundary_width:
            types[i, j] = SOLID


# should not generate particles in solid cells
@ti.kernel
def init_particle():
    for i in particle_position:
        ip = (i % (2 * 64)) * 0.5 * dx + boundary_width * dx + 0.25 * dx
        jp = (i // (2 * 64)) * 0.5 * dx + boundary_width * dx + 0.25 * dx
        particle_position[i] = ti.Vector([ip, jp])
        particle_velocity[i] = ti.Vector([0.0, 0.0])


@ti.func
def is_valid(i, j):
    return i >= 0 and i <= m_g-1 and j >= 0 and j <= m_g-1

@ti.func
def is_solid(i, j):
    return  (is_valid(i, j) and types[i, j] == SOLID) #or (not is_valid(i, j)) 

@ti.func
def is_air(i, j):
    return is_valid(i, j) and types[i, j] == AIR

@ti.func
def is_fluid(i, j):
    return is_valid(i, j) and types[i, j] == FLUID


@ti.kernel
def handle_boundary():

    for i, j in velocities_u:
        if is_solid(i-1, j) and velocities_u[i, j] < 0:
            velocities_u[i, j] = 0.0
        if is_solid(i, j) and velocities_u[i, j] > 0:
            velocities_u[i, j] = 0.0  
    
    for i, j in velocities_v:
        if is_solid(i, j-1) and velocities_v[i, j] < 0:
            velocities_v[i, j] = 0.0
        if is_solid(i, j) and velocities_v[i, j] > 0:
            velocities_v[i, j] = 0.0



@ti.kernel
def init_step():

    for i, j in types:
        if not is_solid(i, j):
            types[i, j] = AIR


    for k in particle_velocity:
        grid = (particle_position[k] * inv_dx).cast(int)
        if not is_solid(grid.x, grid.y):
            types[grid] = FLUID

    for k in ti.grouped(velocities_u):
        velocities_u[k] = 0.0
        weights_u[k] = 0.0
    
    for k in ti.grouped(velocities_v):
        velocities_v[k] = 0.0
        weights_v[k] = 0.0

    for k in ti.grouped(pressures):

        divergences[k] = 0
        pressures_factor[k] = 1

        if is_air(k.x, k.y):
            pressures[k] = 0.0
            new_pressures[k] = 0.0


@ti.func
def scatter(grid_v, grid_m, xp, vp, stagger):
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # B-spline

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * vp
            grid_m[base + offset] += weight


@ti.kernel
def particle_to_grid():

    for k in particle_position:

        p = particle_position[k]

        stagger_u = ti.Vector([0.0, 0.5])
        stagger_v = ti.Vector([0.5, 0.0])

        scatter(velocities_u, weights_u, p, particle_velocity[k].x, stagger_u)
        scatter(velocities_v, weights_v, p, particle_velocity[k].y, stagger_v)


@ti.kernel
def grid_normalization():

    for k in ti.grouped(velocities_u):
        weight = weights_u[k]
        if weight > 0:
            velocities_u[k] = velocities_u[k] / weight

    for k in ti.grouped(velocities_v):
        weight = weights_v[k]
        if weight > 0:
            velocities_v[k] = velocities_v[k] / weight


@ti.kernel
def apply_gravity():
    for i, j in velocities_v:
        velocities_v[i, j] += -9.8 * dt


@ti.kernel
def solve_divergence():

    for i, j in divergences:
        if is_fluid(i, j):

            # velocities in solid cells are enforced to be 0
            v_l = velocities_u[i, j]
            v_r = velocities_u[i+1, j]
            v_d = velocities_v[i, j]
            v_u = velocities_v[i, j+1]

            div = v_r - v_l + v_u - v_d


            divergences[i, j] = div / (dx)

            k = 4
            if is_solid(i-1, j):
                k -= 1
            if is_solid(i+1, j):
                k -= 1
            if is_solid(i, j-1):
                k -= 1
            if is_solid(i, j+1):
                k -= 1

            pressures_factor[i, j] = k

@ti.kernel
def pressure_jacobi(p:ti.template(), new_p:ti.template()):

    w = jacobi_damped_para

    for i, j in p:
        if is_fluid(i, j):

            p_l = p[i-1, j]
            p_r = p[i+1, j]
            p_d = p[i, j-1]
            p_u = p[i, j+1]

            new_p[i, j] = (1 - w) * p[i, j] + w * ( p_l + p_r + p_d + p_u - divergences[i, j] * rho / dt * (dx*dx) ) / pressures_factor[i, j]


@ti.kernel
def projection():

    for i, j in ti.ndrange(m_g, m_g):
        if is_fluid(i-1, j) or is_fluid(i, j):
            if is_solid(i-1, j) or is_solid(i, j):
                # velocities_u[i, j] = 0.0
                pass
            else:
                velocities_u[i, j] -= (pressures[i, j] - pressures[i-1, j]) / dx / rho * dt

        if is_fluid(i, j-1) or is_fluid(i, j):
            if is_solid(i, j-1) or is_solid(i, j):
                # velocities_v[i, j] = 0.0
                pass
            else:
                velocities_v[i, j] -= (pressures[i, j] - pressures[i, j-1]) / dx / rho * dt

@ti.func
def gather(grid_v, xp, stagger):
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline

    vel = 0.0

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1]
            vel += weight * grid_v[base + offset]

    return vel


@ti.kernel
def grid_to_particle():

    stagger_u = ti.Vector([0.0, 0.5])
    stagger_v = ti.Vector([0.5, 0.0])
    
    for k in ti.grouped(particle_position):
    
        p = particle_position[k]

        new_u = gather(velocities_u, p, stagger_u)
        new_v = gather(velocities_v, p, stagger_v)

        particle_velocity[k] = ti.Vector([new_u, new_v])



# two methods [clamping velcocity] or [using normal] seem get a similar result
@ti.kernel
def advect_particles():

    for k in ti.grouped(particle_position):

        pos = particle_position[k]
        vel = particle_velocity[k]
        
        pos += vel * dt


        if pos.x < dx * boundary_width and vel.x < 0:
            pos.x = dx * boundary_width
            vel.x = 0

        if pos.x >= length - dx * boundary_width and vel.x > 0:
            pos.x = length - dx * boundary_width - eps
            vel.x = 0

        if pos.y < dx * boundary_width and vel.y < 0:
            pos.y = dx * boundary_width
            vel.y = 0

        if pos.y >= length - dx * boundary_width and vel.y > 0:
            pos.y = length - dx * boundary_width - eps
            vel.y = 0



        particle_position[k] = pos
        particle_velocity[k] = vel


@ti.kernel
def calculate_residual() -> ti.f32:

    res = 0.0

    for i, j in pressures:
        if is_fluid(i, j):

            p_c = pressures[i, j]
            p_l = pressures[i-1, j]
            p_r = pressures[i+1, j]
            p_d = pressures[i, j-1]
            p_u = pressures[i, j+1]

            b_hat = p_l + p_r + p_d + p_u - p_c * pressures_factor[i, j]
            b =  divergences[i, j] * rho / dt * (dx*dx)
            res += b-b_hat

    return res

    # print("residual:", res)


def step():

    init_step()

    particle_to_grid()
    grid_normalization()

    apply_gravity()
    handle_boundary()

    solve_divergence()

    if use_multigrid:
        mg_solver.clear()
        mg_solver.init()
        mg_solver.solve(30)
    else:
        for i in range(jacobi_iters):
            global pressures, new_pressures
            pressure_jacobi(pressures, new_pressures)
            pressures, new_pressures = new_pressures, pressures
            print("jacobi iter %3d: " % i, calculate_residual())


    print("residual:", calculate_residual())


    projection()

    grid_to_particle()
    advect_particles()







init_grid()
init_particle()


gui = ti.GUI("PIC", (res, res))


result_dir = "./result"

for frame in range(299):

    gui.clear(0xFFFFFF)

    debug_frame = 14
    multigrid_debug_level = min(frame//70, mg_solver.level-1)

    for i in range(substep):
        step()

    # break
    if use_multigrid and multigrid_debug_level != -1:
        l = min(max(multigrid_debug_level, 0), mg_solver.level-1)
        mx_n = m_g//(1<<l)
        print(frame, multigrid_debug_level, mx_n, mg_solver.type[l].shape)
        for i in range(mx_n):
            for j in range(mx_n):
                color = 0
                if mg_solver.type[l][i, j] == FLUID:
                    color = 0x99FF99
                elif mg_solver.type[l][i, j] == AIR:
                    color = 0xFFFFFF
                elif mg_solver.type[l][i, j] == SOLID:
                    color = 0xFF9999
                # gui.circle([(i+0.5)/mx_n, (j+0.5)/mx_n], radius = 2, color = color)
                gui.triangle([(i)/mx_n, (j)/mx_n], [(i+1)/mx_n, (j+1)/mx_n], [(i+1)/mx_n, (j)/mx_n], color = color)
                gui.triangle([(i)/mx_n, (j)/mx_n], [(i+1)/mx_n, (j+1)/mx_n], [(i)/mx_n, (j+1)/mx_n], color = color)



    gui.circles(particle_position.to_numpy() / length, radius=0.8, color=0x3399FF)

    gui.text(content=f'Level {multigrid_debug_level}', pos=(0.05, 0.97), color=0x0)

    gui.show()
