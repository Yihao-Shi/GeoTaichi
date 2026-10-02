import taichi as ti
import math
import os

ti.init(arch=ti.gpu)
vec = ti.math.vec2

SAVE_FRAMES = False

window_size = 512 # 1024  # Number of pixels of the window
n = 8192  # Number of grains

density = 100.0
stiffness = 8e3
restitution_coef = 0.001
gravity = -9.81
dt = 0.0001  # Larger dt might lead to unstable results.
substeps = 60*2


@ti.dataclass
class Grain:
    p: vec  # Position
    m: ti.f32  # Mass
    r: ti.f32  # Radius
    v: vec  # Velocity
    a: vec  # Acceleration
    f: vec  # Force

gf = Grain.field(shape=(n, ))
gf2 = Grain.field(shape=(n, ))

grid_n = 128
grid_size = 1.0 / grid_n  # Simulation domain of size [0, 1]
print(f"Grid size: {grid_n}x{grid_n}")

grain_r_min = 0.002
grain_r_max = 0.003

assert grain_r_max * 2 < grid_size

@ti.kernel
def init():
    for i in gf:
        # Spread grains in a restricted area.
        l = i * grid_size
        padding = 0.1
        region_width = 1.0 - padding * 2
        pos = vec(l % region_width + padding + grid_size * ti.random() * 0.2,
                  l // region_width * grid_size + 0.1)
        gf[i].p = pos
        gf[i].r = ti.random() * (grain_r_max - grain_r_min) + grain_r_min
        gf[i].m = density * math.pi * gf[i].r**2


@ti.kernel
def update():
    for i in gf:
        a = gf[i].f / gf[i].m
        gf[i].v += (gf[i].a + a) * dt / 2.0
        gf[i].p += gf[i].v * dt + 0.5 * a * dt**2
        gf[i].a = a


@ti.kernel
def apply_bc():
    bounce_coef = 0.3  # Velocity damping
    for i in gf:
        x = gf[i].p[0]
        y = gf[i].p[1]

        if y - gf[i].r < 0:
            gf[i].p[1] = gf[i].r
            gf[i].v[1] *= -bounce_coef

        elif y + gf[i].r > 1.0:
            gf[i].p[1] = 1.0 - gf[i].r
            gf[i].v[1] *= -bounce_coef

        if x - gf[i].r < 0:
            gf[i].p[0] = gf[i].r
            gf[i].v[0] *= -bounce_coef

        elif x + gf[i].r > 1.0:
            gf[i].p[0] = 1.0 - gf[i].r
            gf[i].v[0] *= -bounce_coef


@ti.func
def resolve(i, j):
    rel_pos = gf[j].p - gf[i].p
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
    delta = -dist + gf[i].r + gf[j].r  # delta = d - 2 * r
    if delta > 0:  # in contact
        normal = rel_pos / dist
        f1 = normal * delta * stiffness
        # Damping force
        M = (gf[i].m * gf[j].m) / (gf[i].m + gf[j].m)
        K = stiffness
        C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)
                  ) * ti.sqrt(K * M)
        V = (gf[j].v - gf[i].v) * normal
        f2 = C * V * normal
        gf[i].f += f2 - f1
        gf[j].f -= f2 - f1


# list_head = ti.field(dtype=ti.i32, shape=(grid_n , grid_n))
# list_cur = ti.field(dtype=ti.i32, shape=(grid_n , grid_n))
# list_tail = ti.field(dtype=ti.i32, shape=(grid_n , grid_n))
# prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="prefix_sum")
# grain_count = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="grain_count")

# list_st = ti.Vector.field(n=4,dtype=ti.i32,shape=(grid_n, grid_n))
# list_st = ti.field(ti.i32)
list_st = ti.Vector.field(n=4, dtype=ti.i32)
grid_sp = 8
ti.root.dense(ti.ij, (grid_n // grid_sp, grid_n // grid_sp)).dense(ti.ij, (grid_sp, grid_sp)).place(list_st)

column_sum = ti.field(dtype=ti.i32, shape=grid_n, name="column_sum")
particle_id = ti.field(dtype=ti.i32, shape=n, name="particle_id")


# def contact(gf: ti.template()):
@ti.kernel
def contact():
    '''
    Handle the collision between grains.
    '''
    for i in gf:
        gf[i].f = vec(0., gravity * gf[i].m)  # Apply gravity.

    list_st.fill(0)

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        list_st[grid_idx].w += 1

    for i in range(grid_n):
        sum = 0
        for j in range(grid_n):
            sum += list_st[i, j].w
        column_sum[i] = sum

    list_st[0, 0].z = 0

    ti.loop_config(serialize=True)
    for i in range(1, grid_n):
        list_st[i, 0].z = list_st[i - 1, 0].z + column_sum[i - 1]

    for i in range(grid_n):
        for j in range(grid_n):
            if j == 0:
                list_st[i, j].z += list_st[i, j].w
            else:
                list_st[i, j].z = list_st[i, j - 1].z + list_st[i, j].w

            list_st[i,j].x = list_st[i, j].z - list_st[i, j].w
            list_st[i,j].y = list_st[i,j].x

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        grain_location = ti.atomic_add(list_st[grid_idx].y, 1)
        gf2[grain_location] = gf[i]

    for i in gf:
        gf[i] = gf2[i]

    # Brute-force collision detection
    '''
    for i in range(n):
        for j in range(i + 1, n):
            resolve(i, j)
    '''

    # Fast collision detection
    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                for j in range(list_st[neigh_i , neigh_j].x,
                                   list_st[neigh_i , neigh_j].z):
                    if i < j:
                        resolve(i, j)


init()
gui = ti.GUI('tiDEM', (window_size, window_size))
step = 0

if SAVE_FRAMES:
    os.makedirs('output', exist_ok=True)

while gui.running:
    for s in range(substeps):
        update()
        apply_bc()
        contact()
    pos = gf.p.to_numpy()
    r = gf.r.to_numpy() * window_size
    gui.circles(pos, radius=r)
    if SAVE_FRAMES:
        gui.show(f'output/{step:06d}.png')
    else:
        gui.show()
    step += 1
