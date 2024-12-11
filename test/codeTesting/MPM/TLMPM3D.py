import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64)
from math import pi

from src.utils.ShapeFunctions import ShapeLinear, GShapeLinear
from src.utils.VectorFunction import vsign

vec3f = ti.types.vector(3, float)
mat3x3 = ti.types.matrix(3, 3, float)

val_lim = 1e-15
base = 5
xrange, yrange, zrange = 5., 5., 5.
damp = 0.2
npic = 2
px1, py1, pz1 = 0.5, 1.0, 3.5
lx1, ly1, lz1 = 4.2, 0.5, 0.5
dx, inv_dx = 0.1, 10
dt = 1e-4
p_rho = 2500
gravity = vec3f(0., 0., -9.8)
alpha = 0.01
inf_nodes = 8

pi = 3.1415926536
e, nu = 1e7, 0.3
mu = 0.5 * e / (1 + nu)
la = e * nu / (1 + nu) / (1 - 2. * nu)

ex1, ey1, ez1 = lx1 * inv_dx, ly1 * inv_dx, lz1 * inv_dx
pb1 = int(ex1 * ey1 * ez1 * npic * npic * npic)
p_vol = dx * dx * dx / npic / npic / npic 
p_mass = p_vol * p_rho
p_rad = dx / npic / 2.
n_particles = pb1
pdx = dx / npic
grid_x, grid_y, grid_z = int(xrange * inv_dx + 1), int(yrange * inv_dx + 1), int(zrange * inv_dx + 1)
space = px1 * inv_dx

LnID = ti.field(int)
shape = ti.field(float)
dshape = ti.Vector.field(3, float)
ti.root.dense(ti.ij, (n_particles, inf_nodes)).place(LnID, shape, dshape)
offset = ti.field(int, shape=n_particles)

@ti.dataclass
class Particle:
    bID: int
    x: vec3f
    v: vec3f
    vol0: float
    m: float
    lp: vec3f
    stress: mat3x3
    td: mat3x3

@ti.dataclass
class Grid:
    m: float
    p: vec3f
    f: vec3f

p = Particle.field(shape=n_particles)
g = Grid.field(shape=grid_x * grid_y * grid_z)

@ti.kernel
def particle_init():
    for i in range(pb1):
        a = (i % ((ex1 * npic) * (ey1 * npic))) % (ex1 * npic)
        b = (i % ((ex1 * npic) * (ey1 * npic))) // (ex1 * npic)
        c = i // ((ex1 * npic) * (ey1 * npic))
        p[i].bID = 0
        p[i].x = vec3f(px1 + dx / npic / 2. + a * pdx, py1 + dx / npic / 2. + b * pdx, pz1 + dx / npic / 2. + c * pdx)
        p[i].lp = vec3f(dx / npic / 2., dx / npic / 2., dx / npic / 2.)
        p[i].vol0 = p_vol
        p[i].m = p_mass
        p[i].td = mat3x3([1, 0, 0], [0, 1, 0], [0, 0, 1])

@ti.kernel
def shape_init():
    for i in range(n_particles):
        pos = p[i].x
        x = ti.floor(pos[0] / dx, int) #- 1
        y = ti.floor(pos[1] / dx, int) #- 1
        z = ti.floor(pos[2] / dx, int) #- 1

        for a, b, c in ti.ndrange(2, 2, 2):
            grid_idx = x + a
            grid_idy = y + b
            grid_idz = z + c
            if grid_idx < 0 or grid_idx >= grid_x: continue
            if grid_idy < 0 or grid_idy >= grid_y: continue
            if grid_idz < 0 or grid_idz >= grid_z: continue

            sx = ShapeLinear(pos[0], grid_idx * dx, inv_dx, p[i].lp[0])
            sy = ShapeLinear(pos[1], grid_idy * dx, inv_dx, p[i].lp[1])
            sz = ShapeLinear(pos[2], grid_idz * dx, inv_dx, p[i].lp[2])
            gsx = GShapeLinear(pos[0], grid_idx * dx, inv_dx, p[i].lp[0])
            gsy = GShapeLinear(pos[1], grid_idy * dx, inv_dx, p[i].lp[1])
            gsz = GShapeLinear(pos[2], grid_idz * dx, inv_dx, p[i].lp[2])
            s = sx * sy * sz
            gs = vec3f(gsx * sy * sz, gsy * sx * sz, gsz * sx * sy)

            if s <= val_lim: continue

            linear_grid_id = int(grid_idx + grid_idy * grid_x + grid_idz * grid_x * grid_y)

            count = ti.atomic_add(offset[i], 1)
            LnID[i, count] = linear_grid_id
            shape[i, count] = s
            dshape[i, count] = gs

@ti.kernel
def gmass_init():
    for i in range(n_particles):
        for j in range(offset[i]):
            g[LnID[i, j]].m += shape[i, j] * p[i].m

@ti.kernel
def solve():
    for i in g:
        g[i].p = vec3f(0, 0, 0)
        g[i].f = vec3f(0, 0, 0)

    for i in range(n_particles):
        for j in range(offset[i]):
            g[LnID[i, j]].p += shape[i, j] * p[i].m * p[i].v
        
    for i in range(n_particles):
        for j in range(offset[i]):
            g[LnID[i, j]].f += shape[i, j] * p[i].m * gravity - p[i].vol0 * (p[i].stress) @ dshape[i, j] 

    for i in g:
        if g[i].m > val_lim:
            if g[i].p.dot(g[i].f) > 0.:
                g[i].f -= damp * g[i].f.norm() * vsign(g[i].p)
            g[i].p += g[i].f * dt

    for i in g:
        if g[i].m > val_lim:
            idx = (i % (grid_x * grid_y)) % grid_x

            if idx <= space: 
                g[i].p = vec3f(0, 0, 0)
                g[i].f = vec3f(0, 0, 0)
    
    for i in range(n_particles):
        acc = vec3f(0, 0, 0)
        vel = vec3f(0, 0, 0)
        for j in range(offset[i]):
            acc += shape[i, j] * g[LnID[i, j]].f / g[LnID[i, j]].m
            vel += shape[i, j] * g[LnID[i, j]].p / g[LnID[i, j]].m
        
        p[i].v += acc * dt
        p[i].x += vel * dt

    for i in g:
        g[i].p = vec3f(0, 0, 0)

    for i in range(n_particles):
        for j in range(offset[i]):
            g[LnID[i, j]].p += shape[i, j] * p[i].m * p[i].v

    for i in g:
        if g[i].m > val_lim:
            idx = (i % (grid_x * grid_y)) % grid_x

            if idx <= space: 
                g[i].p = vec3f(0, 0, 0)
                g[i].f = vec3f(0, 0, 0)

    for i in range(n_particles):
        td_rate = mat3x3([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for j in range(offset[i]):
            td_rate += (g[LnID[i, j]].p / g[LnID[i, j]].m).outer_product(dshape[i, j])
        
        p[i].td += td_rate * dt
        p[i].stress = mu * (p[i].td - p[i].td.inverse().transpose()) + la * ti.log(p[i].td.determinant()) * p[i].td.inverse().transpose()
         
@ti.kernel
def copy():
    for i in range(n_particles):
        visp[i] = ti.cast(vec3f([p[i].x[0], p[i].x[1], p[i].x[2]]), ti.f32)

window = ti.ui.Window('Window Title', (892, 892), show_window = True, vsync=False)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(2.5, -5, 3)  # x, y, z
camera.lookat(2.5, 5, 2.9)
camera.up(0, 1, 0)
camera.fov(70)
scene.set_camera(camera)
visp = ti.Vector.field(3, ti.f32, n_particles)
particle_init()
shape_init()
gmass_init()
while window.running:
    solve()
    copy()
    camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.LMB)
    scene.set_camera(camera)

    scene.point_light((6., 0.2, 6.), color=(1.0, 1.0, 1.0))
    scene.particles(visp, radius=p_rad, color=(1, 1, 1))

    canvas.set_background_color((0, 0, 0))
    canvas.scene(scene)
    window.show()
