import taichi as ti
ti.init(arch=ti.gpu)

n_particles, n_grid = 6400, 128
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4
p_vol, p_rho = dx ** 2 / 4, 10
p_mass = p_vol * p_rho
E, nu = 5000, 0.2  # Young's modulus and Poisson's ratio
mu, la = 0.3 * E / (2 * (1 + nu)), 0.3 * E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
Gradv = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
#x = ti.Vector.field(2, float)
#v = ti.Vector.field(2, float)
#Gradv = ti.Matrix.field(2, 2, float)
#F = ti.Matrix.field(2, 2, float)
#ti.root.dense(ti.i, n_particles).place(x, F, v, Gradv)
material = ti.field(dtype=int, shape=n_particles)  # material id
grid_mv = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
#grid_m = ti.field(ti.f32)
#grid_mv = ti.Vector.field(2, float)
#grid_v = ti.Vector.field(2, float)
#ti.root.dense(ti.ij, (n_grid//8, n_grid//8)).dense(ti.ij, (8, 8)).place(grid_m, grid_mv, grid_v)
gravity = ti.Vector.field(2, dtype=float, shape=())


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_mv[i, j] = [0, 0]
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        x = x[p]
        base = (x * inv_dx - 0.5).cast(int)
        fx = x * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F[p] = (ti.Matrix.identity(float, 2) + dt * Gradv[p]) @ F[p]  # deformation gradient update
        U, sig, V = ti.svd(F[p])
        J = sig[0, 0] * sig[1, 1]
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * Gradv[p]
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_mv[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_mv[i, j]  # Momentum to velocity
            grid_v[i, j] += dt * gravity[None] * 30  # gravity
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], Gradv[p] = new_v, new_C
        x[p] += dt * v[p]  # advection


@ti.kernel
def reset():
    group_size = n_particles // 2
    for i in range(n_particles):
        x[i] = [ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size), ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)]
        material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Gradv[i] = ti.Matrix.zero(float, 2, 2)

print(ti.log(10))
gui = ti.GUI("Taichi MLS-MPM-128", res=512, background_color=0x112F41)
reset()
gravity[None] = [0, -1]
for frame in range(20000):
    for s in range(int(2e-3 // dt)):
        substep()
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk
