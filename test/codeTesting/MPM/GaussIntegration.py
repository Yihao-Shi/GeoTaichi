import sys
sys.path.append("/home/eleven/work/GeoTaichi")

import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)

from src.utils.ShapeFunctions import ShapeLinear, GShapeLinear
from src.utils.VectorFunction import vsign
from src.utils.GaussPoint import GaussPointInRectangle

vec2f = ti.types.vector(2, float)
vec2i = ti.types.vector(2, int)
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
ngp = 2

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
inf_nodes = 4
cell_vol = dx * dx

LnID = ti.field(int)
shape = ti.field(float)
dshape = ti.Vector.field(2, float)
ti.root.dense(ti.ij, (n_particles, inf_nodes)).place(LnID, shape, dshape)
offset = ti.field(int, shape=n_particles)

@ti.dataclass
class Particle:
    bID: int
    x: vec2f
    disp: vec2f
    v: vec2f
    vol0: float
    m: float
    vol: float
    lp: vec2f
    stress: mat2x2
    td: mat2x2

@ti.dataclass
class Grid:
    m: float
    p: vec2f
    f: vec2f

@ti.dataclass
class Cell:
    active: ti.u8
    volume: float


@ti.dataclass
class GuassCell:
    stress: mat2x2
    vol: float


p = Particle.field(shape=n_particles)
g = Grid.field(shape=(grid_x * grid_y))
c = Cell.field(shape=(cell_x * cell_y))
gc = GuassCell.field(shape=(cell_x * cell_y * ngp * ngp))
gp = GaussPointInRectangle(gauss_point=ngp, dimemsion=2)

LnID = ti.field(int)
shape = ti.field(float)
dshape = ti.Vector.field(2, float)
ti.root.dense(ti.ij, (n_particles, inf_nodes)).place(LnID, shape, dshape)
offset = ti.field(int, shape=n_particles)

@ti.kernel
def init():
    for i in range(pb1):
        a = i % (ex1 * npic)
        b = i // (ex1 * npic)
        p[i].bID = 0
        p[i].x = vec2f(px1 + dx / npic / 2. + a * pdx, py1 + dx / npic / 2. + b * pdx)
        p[i].lp = vec2f(dx / npic / 2., dx / npic / 2.)
        p[i].vol0 = p_vol
        p[i].m = p_mass
        p[i].td = mat2x2([1, 0], [0, 1])
    
    for i in c:
        c[i].active = ti.u8(1)

@ti.kernel
def shape_init():
    offset.fill(0)
    for i in range(n_particles):
        pos = p[i].x
        x = ti.floor(pos[0] * inv_dx, int) #- 1
        y = ti.floor(pos[1] * inv_dx, int) #- 1

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
def solve():
    for i in g:
        g[i].p = vec2f(0, 0)
        g[i].f = vec2f(0, 0)
        g[i].m = 0

    for i in range(n_particles):
        for j in range(offset[i]):
            g[LnID[i, j]].m += shape[i, j] * p[i].m
            g[LnID[i, j]].p += shape[i, j] * p[i].m * p[i].v

    for i in g:
        if g[i].m > val_lim:
            idx = i % grid_x

            if idx <= space: 
                g[i].p = vec2f(0, 0)
                g[i].f = vec2f(0, 0)

    for i in range(n_particles):
        gradv = mat2x2([[0, 0], [0, 0]])
        for j in range(offset[i]):
            gradv += (g[LnID[i, j]].p / g[LnID[i, j]].m).outer_product(dshape[i, j])
        
        de = 0.5 * (gradv + gradv.transpose()) * dt
        p[i].td = (mat2x2([1, 0], [0, 1]) + gradv * dt) @ p[i].td
        p[i].vol = p[i].vol0 * p[i].td.determinant()
        p[i].stress += 2. * mu * de + la * de.trace() * mat2x2([1, 0], [0, 1])

    for i in c:
        if c[i].volume > val_lim:
            c[i].volume = 0.

    for i in gc:
        if gc[i].vol > 0.:
            gc[i].vol = 0.
            gc[i].stress = mat2x2([[0, 0], [0, 0]])

    for np in range(n_particles):
        cellID = ti.floor(p[np].x * inv_dx, int)
        linear_cellID = int(cellID[0] + cellID[1] * cell_x)
        c[linear_cellID].volume += p[np].vol

    for nc in range(c.shape[0]):
        if int(c[nc].active) == 1:
            if c[nc].volume / cell_vol > 0.9:
                c[nc].active = ti.u8(1)
            else:
                c[nc].active = ti.u8(0)

    gauss_number = ngp * ngp 
    for np in range(n_particles):
        element_id = ti.floor(p[np].x * inv_dx, int)
        linear_element_id = int(element_id[0] + element_id[1] * cell_x)
        if int(c[linear_element_id].active) == 1:
            volume = p[np].vol
            sub_element_id = ti.floor((p[np].x - element_id * dx) * inv_dx * ngp, int)
            sub_linear_element_id = sub_element_id[0] + sub_element_id[1] * ngp 
            gc[linear_element_id * gauss_number + sub_linear_element_id].stress += p[np].stress * volume
            gc[linear_element_id * gauss_number + sub_linear_element_id].vol += volume

    for nc in range(gc.shape[0]):
        if int(c[nc // gauss_number].active) == 1 and gc[nc].vol > val_lim:
            gc[nc].stress /= gc[nc].vol

    for nc in range(c.shape[0]):
        if int(c[nc].active) == 1:
            pressure = 0.
            for ngp in range(gauss_number):
                stress = gc[nc * gauss_number + ngp].stress
                pressure += (stress[0, 0] + stress[1, 1]) / 2.
            pressure /= gauss_number

            for ngp in range(gauss_number):
                stress = gc[nc * gauss_number + ngp].stress
                p = (stress[0, 0] + stress[1, 1]) / 2.
                ave_stress = stress - (p - pressure) * mat2x2([1, 0], [0, 1])
                gc[nc * gauss_number + ngp].stress = ave_stress

    for nc in range(c.shape[0]):
        if int(c[nc].active) == 1:
            ic = nc % cell_x
            jc = nc // cell_x
            base = vec2i(ic, jc)
            weight = dx * dx / gauss_number
            for ngp in range(gauss_number):
                gpoint = 0.5 * dx * (gp.gpcoords[ngp] + 1) + base * dx
                fInt = -weight * gc[nc * gauss_number + ngp].stress
                for i, j in ti.static(ti.ndrange(2, 2)):
                    nx, ny = base[0] + i, base[1] + j
                    nodeID = nx + ny * grid_x
                    sx = ShapeLinear(gpoint[0], nx * dx, inv_dx, 0)
                    sy = ShapeLinear(gpoint[1], ny * dx, inv_dx, 0)
                    gsx = GShapeLinear(gpoint[0], nx * dx, inv_dx, 0)
                    gsy = GShapeLinear(gpoint[1], ny * dx, inv_dx, 0)
                    dshape_fn = vec2f(gsx * sy, gsy * sx)
                    internal_force = dshape_fn @ fInt
                    g[nodeID].f += internal_force

    for i in range(n_particles):
        element_id = ti.floor(p[i].x * inv_dx, int)
        linear_element_id = element_id[0] + element_id[1] * cell_x
        if int(c[linear_element_id].active) == 0:
            fInt = -p[i].vol * p[i].stress
            for j in range(offset[i]):
                internal_force = dshape[i, j] @ fInt
                g[LnID[i, j]].f += internal_force
        
    for i in range(n_particles):
        for j in range(offset[i]):
            g[LnID[i, j]].f += shape[i, j] * p[i].m * gravity

    for i in g:
        if g[i].m > val_lim:
            if g[i].p.dot(g[i].f) > 0.:
                g[i].f -= damp * g[i].f.norm() * vsign(g[i].p)
            g[i].p += g[i].f * dt

    for i in g:
        if g[i].m > val_lim:
            idx = i % grid_x

            if idx <= space: 
                g[i].p = vec2f(0, 0)
                g[i].f = vec2f(0, 0)

    for i in range(n_particles):
        acc, vel = vec2f(0, 0), vec2f(0, 0)
        for j in range(offset[i]):
            acc += shape[i, j] * g[LnID[i, j]].f / g[LnID[i, j]].m
            vel += shape[i, j] * g[LnID[i, j]].p / g[LnID[i, j]].m
        
        p[i].v += acc * dt
        p[i].x += vel * dt
         
@ti.kernel
def copy():
    for i in range(n_particles):
        visp[i] = ti.cast(vec3f([p[i].x[0], p[i].x[1], 0]) / base, ti.f32)

window = ti.ui.Window('Window Title', (892, 892))
visp = ti.Vector.field(3, ti.f32, n_particles)
init()
gp.create_gauss_point()
while window.running:
    copy()
    canvas = window.get_canvas()
    canvas.set_background_color((0, 0, 0))
    canvas.circles(visp, p_rad/base, (1, 1, 1))
    shape_init()
    solve()
    window.show()

@ti.kernel
def find_max_disp() -> float:
    max_disp = 0.
    for i in range(n_particles):
        ti.atomic_max(max_disp, p[i].disp.norm())
    return max_disp
print(find_max_disp())