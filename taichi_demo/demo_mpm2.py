import taichi as ti
import numpy as np
import time
ti.init(arch=ti.cpu)

quality = 1
n_particles = 20000 * quality ** 3
n_s_particles = ti.field(dtype = int, shape = ())
n_w_particles = ti.field(dtype = int, shape = ())
n_grid = 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 2e-4 / quality
gravity = ti.Vector([0, 0, -9.8])
d = 3

# sand particle properties
x_s = ti.Vector.field(3, dtype = float, shape = n_particles) # position
v_s = ti.Vector.field(3, dtype = float, shape = n_particles) # velocity
C_s = ti.Matrix.field(3, 3, dtype = float, shape = n_particles) # affine velocity matrix
F_s = ti.Matrix.field(3, 3, dtype = float, shape = n_particles) # deformation gradient
phi_s = ti.field(dtype = float, shape = n_particles) # cohesion and saturation
c_C0 = ti.field(dtype = float, shape = n_particles) # initial cohesion (as maximum)
vc_s = ti.field(dtype = float, shape = n_particles) # tracks changes in the log of the volume gained during extension
alpha_s = ti.field(dtype = float, shape = n_particles) # yield surface size
q_s = ti.field(dtype = float, shape = n_particles) # harding state

# sand grid properties
grid_sv = ti.Vector.field(3, dtype = float) # grid node momentum/velocity
grid_sm = ti.field(dtype = float) # grid node mass
grid_sf = ti.Vector.field(3, dtype = float) # forces in the sand

# water particle properties
x_w = ti.Vector.field(3, dtype = float, shape = n_particles) # position
v_w = ti.Vector.field(3, dtype = float, shape = n_particles) # velocity
C_w = ti.Matrix.field(3, 3, dtype = float, shape = n_particles) # affine velocity matrix
J_w = ti.field(dtype = float, shape = n_particles) # ratio of volume increase

# water grid properties
grid_wv = ti.Vector.field(3, dtype = float) # grid node momentum/velocity
grid_wm = ti.field(dtype = float) # grid node mass
grid_wf = ti.Vector.field(3, dtype = float) #  forces in the water

block_size = 16
block0 = ti.root.pointer(ti.ijk, n_grid // block_size)
block1 = block0.dense(ti.ijk, block_size)
block1.place(grid_sv, grid_sm, grid_sf, grid_wv, grid_wm, grid_wf)

# constant values
p_vol, s_rho, w_rho = (dx * 0.5) ** 3, 400, 400
s_mass, w_mass = p_vol * s_rho, p_vol * w_rho

w_k, w_gamma = 50, 3 # bulk modulus of water and gamma is a term that more stiffy penalizes large deviations from incompressibility

n, k_hat = 0.4, 0.2 # sand porosity and permeability

E_s, nu_s = 3.537e5, 0.3 # sand's Young's modulus and Poisson's ratio
mu_s, lambda_s = E_s / (2 * (1 + nu_s)), E_s * nu_s / ((1 + nu_s) * (1 - 2 * nu_s)) # sand's Lame parameters

mu_b = 0.75 # coefficient of friction

a, b, c0, sC = -3.0, 0, 1e-2, 0.15
# The scalar function h_s is chosen so that the multiplier function is twice continuously differentiable
@ti.func
def h_s(z):
    ret = 0.0
    if z < 0: ret = 1
    if z > 1: ret = 0
    ret = 1 - 10 * (z ** 3) + 15 * (z ** 4) - 6 * (z ** 5)
    return ret

# multiplier
sqrt2 = ti.sqrt(2)
@ti.func
def h(e):
    u = e.trace() / sqrt2
    v = ti.abs(ti.Vector([e[0, 0] - u / sqrt2, e[1, 1] - u / sqrt2]).norm())
    fe = c0 * (v ** 4) / (1 + v ** 3)

    ret = 0.0
    if u + fe < a + sC: ret = 1
    if u + fe > b + sC: ret = 0
    ret = h_s((u + fe - a - sC) / (b - a))
    return ret

state = ti.field(dtype = int, shape = n_particles)
pi = 3.14159265358979
@ti.func
def project(e0, p):
    e = e0 + vc_s[p] / d * ti.Matrix.identity(float, 3) # volume correction treatment
    e += (c_C0[p] * (1.0 - phi_s[p])) / (d * alpha_s[p]) * ti.Matrix.identity(float, 3) # effects of cohesion
    ehat = e - e.trace() / d * ti.Matrix.identity(float, 3)
    Fnorm = ti.sqrt(ehat[0, 0] ** 2 + ehat[1, 1] ** 2 + ehat[2, 2] ** 2) # Frobenius norm
    yp = Fnorm + (d * lambda_s + 2 * mu_s) / (2 * mu_s) * e.trace() * alpha_s[p] # delta gamma
    new_e = ti.Matrix.zero(float, 3, 3)
    delta_q = 0.0
    if Fnorm <= 0 or e.trace() > 0: # Case II:
        new_e = ti.Matrix.zero(float, 3, 3)
        delta_q = ti.sqrt(e[0, 0] ** 2 + e[1, 1] ** 2 + e[2, 2] ** 2)
        state[p] = 0
    elif yp <= 0: # Case I:
        new_e = e0 # return initial matrix without volume correction and cohesive effect
        delta_q = 0
        state[p] = 1
    else: # Case III:
        new_e = e - yp / Fnorm * ehat
        delta_q = yp
        state[p] = 2

    return new_e, delta_q

h0, h1, h2, h3 = 35, 9, 0.2, 10
@ti.func
def hardening(dq, p): # The amount of hardening depends on the amount of correction that occurred due to plasticity
    q_s[p] += dq
    phi = h0 + (h1 * q_s[p] - h3) * ti.exp(-h2 * q_s[p])
    phi = phi / 180 * pi # details in Table. 3: Friction angle phi_F and hardening parameters h0, h1, and h3 are listed in degrees for convenience
    sin_phi = ti.sin(phi)
    alpha_s[p] = ti.sqrt(2 / 3) * (2 * sin_phi) / (3 - sin_phi)

@ti.kernel
def substep():
    # set zero initial state for both water/sand grid
    for i, j, k in grid_sm:
        grid_sv[i, j, k], grid_wv[i, j, k] = [0, 0, 0], [0, 0, 0]
        grid_sm[i, j, k], grid_wm[i, j, k] = 0, 0
        grid_sf[i, j, k], grid_wf[i, j, k] = [0, 0, 0], [0, 0, 0]

    # P2G (sand's part)
    for p in range(n_s_particles[None]):
        base = (x_s[p] * inv_dx - 0.5).cast(int)
        if base[0] < 0 or base[1] < 0 or base[2] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2 or base[2] >= n_grid - 2:
            continue
        fx = x_s[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        U, sig, V = ti.svd(F_s[p])
        inv_sig = sig.inverse()
        e = ti.Matrix([[ti.log(sig[0, 0]), 0, 0], [0, ti.log(sig[1, 1]), 0], [0, 0, ti.log(sig[2, 2])]])
        stress = U @ (2 * mu_s * inv_sig @ e + lambda_s * e.trace() * inv_sig) @ V.transpose() # formula (25)
        stress = (-p_vol * 4 * inv_dx * inv_dx) * stress @ F_s[p].transpose()
        # stress *= h(e)
        # print(h(e))
        affine = s_mass * C_s[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_sv[base + offset] += weight * (s_mass * v_s[p] + affine @ dpos)
            grid_sm[base + offset] += weight * s_mass
            grid_sf[base + offset] += weight * stress @ dpos

    # P2G (water's part):
    for p in range(n_w_particles[None]):
        base = (x_w[p] * inv_dx - 0.5).cast(int)
        fx = x_w[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = w_k * (1 - 1 / (J_w[p] ** w_gamma))
        stress = (-p_vol * 4 * inv_dx * inv_dx) * stress * J_w[p]
        # stress = -4 * 400 * p_vol * (J_w[p] - 1) / dx ** 2 (special case when gamma equals to 1)
        affine = w_mass * C_w[p]
        # affine = ti.Matrix([[stress, 0], [0, stress]]) + w_mass * C_w[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_wv[base + offset] += weight * (w_mass * v_w[p] + affine @ dpos)
            grid_wm[base + offset] += weight * w_mass
            grid_wf[base + offset] += weight * stress * dpos

    # Update Grids Momentum
    for i, j, k in grid_sm:
        if grid_sm[i, j, k] > 0:
            grid_sv[i, j, k] = (1 / grid_sm[i, j, k]) * grid_sv[i, j, k] # Momentum to velocity
        if grid_wm[i, j, k] > 0:
            grid_wv[i, j, k] = (1 / grid_wm[i, j, k]) * grid_wv[i, j, k]

        # Momentum exchange
        cE = (n ** 3 * w_rho * gravity[2]) / k_hat #  drag coefficient
        if grid_sm[i, j, k] > 0 and grid_wm[i, j, k] > 0:
            sm, wm = grid_sm[i, j, k], grid_wm[i, j, k]
            sv, wv = grid_sv[i, j, k], grid_wv[i, j, k]
            d = cE * sm * wm
            M = ti.Matrix([[sm, 0], [0, wm]])
            D = ti.Matrix([[-d, d], [d, -d]])
            V = ti.Matrix.rows([sv, wv])
            G = ti.Matrix.rows([gravity, gravity])
            F = ti.Matrix.rows([grid_sf[i, j, k], grid_wf[i, j, k]])

            A = M + dt * D
            B = M @ V + dt * (M @ G + F)
            X = A.inverse() @ B
            grid_sv[i, j, k], grid_wv[i, j, k] = ti.Vector([X[0, 0], X[0, 1], X[0, 2]]), ti.Vector([X[1, 0], X[1, 1], X[1, 2]])

        elif grid_sm[i, j, k] > 0:
            grid_sv[i, j, k] += dt * (gravity + grid_sf[i, j, k] / grid_sm[i, j, k]) # Update explicit force
        elif grid_wm[i, j, k] > 0:
            grid_wv[i, j, k] += dt * (gravity + grid_wf[i, j, k] / grid_wm[i, j, k])

        if grid_sm[i, j, k] > 0:
            if i < 3 and grid_sv[i, j, k][0] < 0:          grid_sv[i, j, k][0] = 0 # Boundary conditions
            if i > n_grid - 3 and grid_sv[i, j, k][0] > 0: grid_sv[i, j, k][0] = 0
            if j < 3 and grid_sv[i, j, k][1] < 0:          grid_sv[i, j, k][1] = 0
            if j > n_grid - 3 and grid_sv[i, j, k][1] > 0: grid_sv[i, j, k][1] = 0
            if k < 3 and grid_sv[i, j, k][2] < 0:          grid_sv[i, j, k][2] = 0
            if k > n_grid - 3 and grid_sv[i, j, k][2] > 0: grid_sv[i, j, k][2] = 0

        if grid_wm[i, j, k] > 0:
            if i < 3 and grid_wv[i, j, k][0] < 0:          grid_wv[i, j, k][0] = 0 # Boundary conditionw
            if i > n_grid - 3 and grid_wv[i, j, k][0] > 0: grid_wv[i, j, k][0] = 0
            if j < 3 and grid_wv[i, j, k][1] < 0:          grid_wv[i, j, k][1] = 0
            if j > n_grid - 3 and grid_wv[i, j, k][1] > 0: grid_wv[i, j, k][1] = 0
            if k < 3 and grid_wv[i, j, k][2] < 0:          grid_wv[i, j, k][2] = 0
            if k > n_grid - 3 and grid_wv[i, j, k][2] > 0: grid_wv[i, j, k][2] = 0

    # G2P (water's part)
    for p in range(n_w_particles[None]):
        base = (x_w[p] * inv_dx - 0.5).cast(int)
        fx = x_w[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            dpos = ti.Vector([i, j, k]).cast(float) - fx
            g_v = grid_wv[base + ti.Vector([i, j, k])]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        J_w[p] = (1 + dt * new_C.trace()) * J_w[p]
        v_w[p], C_w[p] = new_v, new_C
        x_w[p] += dt * v_w[p]

    # G2P (sand's part)
    for p in range(n_s_particles[None]):
        base = (x_s[p] * inv_dx - 0.5).cast(int)
        if base[0] < 0 or base[1] < 0 or base[2] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2 or base[2] >= n_grid - 2:
            continue
        fx = x_s[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        phi_s[p] = 0.0 # Saturation
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)): # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j, k]).cast(float) - fx
            g_v = grid_sv[base + ti.Vector([i, j, k])]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            if grid_sm[base + ti.Vector([i, j, k])] > 0 and grid_wm[base + ti.Vector([i, j, k])] > 0:
                phi_s[p] += weight # formula (24)

        F_s[p] = (ti.Matrix.identity(float, 3) + dt * new_C) @ F_s[p]
        v_s[p], C_s[p] = new_v, new_C
        x_s[p] += dt * v_s[p]

        U, sig, V = ti.svd(F_s[p])
        e = ti.Matrix([[ti.log(sig[0, 0]), 0, 0], [0, ti.log(sig[1, 1]), 0], [0, 0, ti.log(sig[2, 2])]])
        new_e, dq = project(e, p)
        hardening(dq, p)
        new_F = U @ ti.Matrix([[ti.exp(new_e[0, 0]), 0, 0], [0, ti.exp(new_e[1, 1]), 0], [0, 0, ti.exp(new_e[2, 2])]]) @ V.transpose()
        vc_s[p] += -ti.log(new_F.determinant()) + ti.log(F_s[p].determinant()) # formula (26)
        F_s[p] = new_F

@ti.kernel
def initialize():
    n_s_particles[None] = 10000 * quality ** 3
    for i in range(n_s_particles[None]):
        x_s[i] = [ti.random() * 0.25 + 0.4, ti.random() * 0.25 + 0.4, ti.random() * 0.4 + 0.01]
        v_s[i] = ti.Matrix([0, 0, 0])
        F_s[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        c_C0[i] = -0.01
        alpha_s[i] = 0.267765

    n_w_particles[None] = 0

@ti.kernel
def update_jet():
    if n_w_particles[None] < 20000 - 50:
        for i in range(n_w_particles[None], n_w_particles[None] + 50):
            x_w[i] = [ti.random() * 0.03 + 0.92, 0.485 + ti.random() * 0.03, ti.random() * 0.03 + 0.5]
            v_w[i] = ti.Matrix([-1.5, 0, 0])
            J_w[i] = 1

        n_w_particles[None] += 50

@ti.func
def color_lerp(r1, g1, b1, r2, g2, b2, t):
    return int((r1 * (1 - t) + r2 * t) * 0x100) * 0x10000 + int((g1 * (1 - t) + g2 * t) * 0x100) * 0x100 + int((b1 * (1 - t) + b2 * t) * 0x100)

color_s = ti.field(dtype = int, shape = n_particles)
color_w = ti.field(dtype = int, shape = n_particles)
@ti.kernel
def update_color():
    for i in range(n_s_particles[None]):
        color_s[i] = color_lerp(0.521, 0.368, 0.259, 0.318, 0.223, 0.157, phi_s[i])
    for i in range(n_w_particles[None]):
        color_w[i] = color_lerp(0.2, 0.231, 0.792, 0.867, 0.886, 0.886, v_w[i].norm() / 3.0)

pos_s = ti.Vector.field(2, dtype = float, shape = n_particles)
pos_w = ti.Vector.field(2, dtype = float, shape = n_particles)
@ti.kernel
def update_pos():
    for i in range(n_s_particles[None]):
        pos_s[i] = ti.Vector([x_s[i][0], x_s[i][2]])
        # pos_s[i] = ti.Vector([x_s[i][0], x_s[i][1]])
    for i in range(n_w_particles[None]):
        pos_w[i] = ti.Vector([x_w[i][0], x_w[i][2]])
        # pos_w[i] = ti.Vector([x_w[i][0], x_w[i][1]])
initialize()

project_view = False
frame = 0
gui = ti.GUI("2D Dam", res = 512, background_color = 0xFFFFFF)
while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == gui.SPACE: project_view = not project_view
        elif e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
    update_jet()
    for s in range(5):
        substep()
    update_pos()
    if project_view:
        gui.circles(pos_w.to_numpy(), radius = 1.5, color = 0x068587)
        colors = np.array([0xFF0000, 0x00FF00, 0x0000FF], dtype = np.uint32)
        gui.circles(pos_s.to_numpy(), radius = 1.5, color = colors[state.to_numpy()])
    else:
        update_color()
        gui.circles(pos_w.to_numpy(), radius = 1.5, color = color_w.to_numpy())
        gui.circles(pos_s.to_numpy(), radius = 1.5, color = color_s.to_numpy())
    # gui.show(f'{frame:06d}.png')
    gui.show()
    frame += 1