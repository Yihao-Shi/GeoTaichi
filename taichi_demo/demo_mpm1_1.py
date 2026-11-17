import taichi as ti
ti.init(arch=ti.cpu)

n_particles, n_grid = 64000, 128
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4
p_vol, p_rho = dx ** 2 / 4, 10
p_mass = p_vol * p_rho
E, nu = 5000, 0.2  # Young's modulus and Poisson's ratio
mu, la = 0.3 * E / (2 * (1 + nu)), 0.3 * E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

@ti.dataclass
class particle:
    x: ti.types.vector(2, float)
    v: ti.types.vector(2, float)
    Gradv: ti.types.matrix(2, 2, float)
    F: ti.types.matrix(2, 2, float)

material = ti.field(dtype=int, shape=n_particles)  # material id

@ti.dataclass
class grid:
    grid_mv: ti.types.vector(2, float)
    grid_v: ti.types.vector(2, float)
    grid_m: ti.f32

    @ti.func
    def Operation(self, i, j):
        if self.grid_m > 0:  # No need for epsilon here
            self.grid_v = (1 / self.grid_m) * self.grid_mv  # Momentum to velocity
            self.grid_v += dt * gravity[None] * 30  # gravity
            if i < 3 and self.grid_v[0] < 0:
                self.grid_v[0] = 0  # Boundary conditions
            if i > n_grid - 3 and self.grid_v[0] > 0: self.grid_v[0] = 0
            if j < 3 and self.grid_v[1] < 0: self.grid_v[1] = 0
            if j > n_grid - 3 and self.grid_v[1] > 0: self.grid_v[1] = 0

gravity = ti.Vector.field(2, dtype=float, shape=())

particles = particle.field(shape=n_particles)
grids: grid = grid.field(shape=(n_grid, n_grid))

@ti.kernel
def substep():
    for i, j in grids:
        grids[i, j].grid_mv = [0, 0]
        grids[i, j].grid_v = [0, 0]
        grids[i, j].grid_m = 0
    for p in particles:  # Particle state update and scatter to grid (P2G)
        base = (particles[p].x * inv_dx - 0.5).cast(int)
        fx = particles[p].x * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        particles[p].F = (ti.Matrix.identity(float, 2) + dt * particles[p].Gradv) @ particles[p].F  # deformation gradient update
        U, sig, V = ti.svd(particles[p].F)
        J = sig[0, 0] * sig[1, 1]
        stress = 2 * mu * (particles[p].F - U @ V.transpose()) @ particles[p].F.transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * particles[p].Gradv
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grids[base + offset].grid_mv += weight * (p_mass * particles[p].v + affine @ dpos)
            grids[base + offset].grid_m += weight * p_mass
    for i, j in grids:
        if grids[i, j].grid_m > 0:  # No need for epsilon here
            grids[i, j].grid_v = (1 / grids[i, j].grid_m) * grids[i, j].grid_mv  # Momentum to velocity
            grids[i, j].grid_v += dt * gravity[None] * 30  # gravity
            if i < 3 and grids[i, j].grid_v[0] < 0:
                grids[i, j].grid_v[0] = 0  # Boundary conditions
            if i > n_grid - 3 and grids[i, j].grid_v[0] > 0: grids[i, j].grid_v[0] = 0
            if j < 3 and grids[i, j].grid_v[1] < 0: grids[i, j].grid_v[1] = 0
            if j > n_grid - 3 and grids[i, j].grid_v[1] > 0: grids[i, j].grid_v[1] = 0
        #grids[i, j].Operation(i, j)
    for p in particles:  # grid to particle (G2P)
        base = (particles[p].x * inv_dx - 0.5).cast(int)
        fx = particles[p].x * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grids[base + ti.Vector([i, j])].grid_v
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        particles[p].v, particles[p].Gradv = new_v, new_C
        particles[p].x += dt * particles[p].v  # advection


@ti.kernel
def reset():
    group_size = n_particles // 2
    for i in range(n_particles):
        particles[i].x = [ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size), ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)]
        material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
        particles[i].v = [0, 0]
        particles[i].F = ti.Matrix([[1, 0], [0, 1]])
        particles[i].Gradv = ti.Matrix.zero(float, 2, 2)

gui = ti.GUI("Taichi MLS-MPM-128", res=512, background_color=0x112F41)
reset()
gravity[None] = [0, -1]
for frame in range(20000):
    for s in range(int(2e-3 // dt)):
        substep()
    gui.circles(particles.x.to_numpy(), radius=1.5, color=0x068587)
    gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk
