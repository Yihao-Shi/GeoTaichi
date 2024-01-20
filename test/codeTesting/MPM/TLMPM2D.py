import sys
sys.path.append("/home/eleven/work/GeoTaichi")

import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64)
from math import pi

from src.utils.ShapeFunctions import ShapeLinear, GShapeLinear
from src.utils.VectorFunction import vsign

vec2f = ti.types.vector(2, float)
vec3f = ti.types.vector(3, float)
mat2x2 = ti.types.matrix(2, 2, float)

val_lim = 1e-15
base = 5
xrange, yrange = 5., 5.
damp = 0.02
npic = 2
px1, py1 = 0.5, 3.5
lx1, ly1 = 4.2, 0.5
dx, inv_dx = 0.1, 10
dt = 1e-4
p_rho = 2500
gravity = vec2f(0., -9.8)
space = px1 * inv_dx
alpha = 0.01

pi = 3.1415926536
e, nu = 2e7, 0.3
mu = 0.5 * e / (1 + nu)
la = e * nu / (1 + nu) / (1 - 2. * nu)

ex1, ey1 = lx1 * inv_dx, ly1 * inv_dx
pb1 = int(ex1 * ey1 * npic * npic)
p_vol = dx * dx / npic / npic 
p_mass = p_vol * p_rho
p_rad = dx / npic / 2.
n_particles = pb1
pdx = dx / npic
grid_x, grid_y = int(xrange * inv_dx + 1), int(yrange * inv_dx + 1)
cell_x, cell_y = int(xrange * inv_dx), int(yrange * inv_dx)
dofs = 2*int(2 * (ex1 + 1) * (ey1 + 1))
inf_nodes = 4

LnID = ti.field(int)
shape = ti.field(float)
dshape = ti.Vector.field(2, float)
ti.root.dense(ti.ij, (n_particles, inf_nodes)).place(LnID, shape, dshape)
offset = ti.field(int, shape=n_particles)

@ti.dataclass
class Particle:
    bID: int
    x: vec2f
    v: vec2f
    vol0: float
    m: float
    lp: vec2f
    stress: mat2x2
    td: mat2x2

@ti.dataclass
class Grid:
    m: float
    p: vec2f
    f: vec2f

p = Particle.field(shape=n_particles)
g = Grid.field(shape=grid_x * grid_y)

@ti.kernel
def particle_init():
    for i in range(pb1):
        a = i % (ex1 * npic)
        b = i // (ex1 * npic)
        p[i].bID = 0
        p[i].x = vec2f(px1 + dx / npic / 2. + a * pdx, py1 + dx / npic / 2. + b * pdx)
        p[i].lp = vec2f(dx / npic / 2., dx / npic / 2.)
        p[i].vol0 = p_vol
        p[i].m = p_mass
        p[i].td = mat2x2([1, 0], [0, 1])

@ti.kernel
def shape_init():
    for i in range(n_particles):
        pos = p[i].x
        x = ti.floor(pos[0] / dx, int) #- 1
        y = ti.floor(pos[1] / dx, int) #- 1

        for a, b in ti.ndrange(2, 2):
            grid_idx = x + a
            grid_idy = y + b
            if grid_idx < 0 or grid_idx >= grid_x: continue
            if grid_idy < 0 or grid_idy >= grid_y: continue

            sx = ShapeLinear(pos[0], grid_idx * dx, inv_dx, p[i].lp[0])
            sy = ShapeLinear(pos[1], grid_idy * dx, inv_dx, p[i].lp[1])
            gsx = GShapeLinear(pos[0], grid_idx * dx, inv_dx, p[i].lp[0])
            gsy = GShapeLinear(pos[1], grid_idy * dx, inv_dx, p[i].lp[1])
            s = sx * sy
            gs = vec2f(gsx * sy, gsy * sx)

            if s <= val_lim: continue

            linear_grid_id = int(grid_idx + grid_idy * grid_x)

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
        g[i].p = vec2f(0, 0)
        g[i].f = vec2f(0, 0)

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
            idx = i % grid_x
            idy = i // grid_x

            if idx <= space: 
                g[i].p = vec2f(0, 0)
                g[i].f = vec2f(0, 0)
    
    for i in range(n_particles):
        acc = vec2f(0, 0)
        vel = vec2f(0, 0)
        for j in range(offset[i]):
            acc += shape[i, j] * g[LnID[i, j]].f / g[LnID[i, j]].m
            vel += shape[i, j] * g[LnID[i, j]].p / g[LnID[i, j]].m
        
        p[i].v += acc * dt
        p[i].x += vel * dt

    for i in g:
        g[i].p = vec2f(0, 0)

    for i in range(n_particles):
        for j in range(offset[i]):
            g[LnID[i, j]].p += shape[i, j] * p[i].m * p[i].v

    for i in g:
        if g[i].m > val_lim:
            idx = i % grid_x
            idy = i // grid_x

            if idx <= space: 
                g[i].p = vec2f(0, 0)
                g[i].f = vec2f(0, 0)

    for i in range(n_particles):
        td_rate = mat2x2([[0, 0], [0, 0]])
        for j in range(offset[i]):
            td_rate += (g[LnID[i, j]].p / g[LnID[i, j]].m).outer_product(dshape[i, j])
        
        p[i].td += td_rate * dt
        p[i].stress = mu * (p[i].td - p[i].td.inverse().transpose()) + la * ti.log(p[i].td.determinant()) * p[i].td.inverse().transpose()
         
@ti.kernel
def copy():
    for i in range(n_particles):
        visp[i] = ti.cast(vec3f([p[i].x[0], p[i].x[1], 0]) / base, ti.f32)

window = ti.ui.Window('Window Title', (892, 892))
visp = ti.Vector.field(3, ti.f32, n_particles)
particle_init()
shape_init()
gmass_init()
solve()
while window.running:
    copy()
    canvas = window.get_canvas()
    canvas.set_background_color((0, 0, 0))
    canvas.circles(visp, p_rad/base, (1, 1, 1))
    solve()
    window.show()
