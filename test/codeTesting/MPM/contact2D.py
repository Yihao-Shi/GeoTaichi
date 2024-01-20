import sys
sys.path.append("/home/eleven/work/GeoTaichi")

import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64)
from math import pi

from src.utils.ShapeFunctions import ShapeGIMP, GShapeGIMP
from src.utils.VectorFunction import vsign
from src.mpm.materials.infinitesimal_strain.LinearElastic import LinearElastic
from src.mpm.materials.infinitesimal_strain.DruckerPrager import DruckerPrager

vec2f = ti.types.vector(2, float)
vec3f = ti.types.vector(3, float)
mat2x2 = ti.types.matrix(2, 2, float)

val_lim = 1e-12
base = 12.
damp = 0.0
npic = 2
space = 1
px1, py1, px2, py2 = 0.1, 0.1, 8.1, 0.1
lx1, ly1, lx2, ly2 = 8., 4., 1., 5.
n_particles, grid_x, grid_y = 14800, 120, 120
dx, inv_dx = 0.1, 10.
dt = 1e-4
p_rho = 2500
gravity = vec2f(0., -9.8)

pi = 3.1415926536
e, nu = 2e7, 0.3
fricfa = 0.577
mu = 0.5 * e / (1 + nu)
la = e * nu / (1 + nu) / (1 - 2. * nu)
k = la + 2./3. * mu

phi = 30 * pi / 180.
psi = 0.
c = 0.
bot = ti.sqrt(9. + 12. * ti.tan(phi) ** 2)
A_dp = 3. * ti.tan(phi) / bot
B_dp = 3. / bot
Ad_dp = 3. * ti.tan(psi / ti.sqrt(9. + 12. * ti.tan(psi) ** 2))

gx1, gy1, gx2, gy2 = lx1 * inv_dx, ly1 * inv_dx, lx2 * inv_dx, ly2 * inv_dx
pb1, pb2 = int(gx1 * gy1 * npic * npic), int(gx2 * gy2 * npic * npic)
p_vol = dx * dx / npic / npic
p_mass = p_vol * p_rho
p_rad = dx / npic / 2.
pdx = dx / npic

assert n_particles >= pb1 + pb2

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
    norm: vec2f

p = Particle.field(shape=n_particles)
g = Grid.field(shape=(grid_x * grid_y, 2))

@ti.kernel
def init():
    for i in range(pb1):
        a = i % (gx1 * npic)
        b = i // (gx1 * npic)
        p[i].bID = 0
        p[i].x = vec2f(px1 + dx / npic / 2. + a * pdx, py1 + dx / npic / 2. + b * pdx)
        syy = -p_rho * gravity[1] * (py1 + dx / npic / 2. + b * pdx - 4.)
        sxx = syy * nu * (1. - nu)
        p[i].stress[0, 0] = sxx
        p[i].stress[1, 1] = syy
        p[i].lp = vec2f(dx / npic / 2., dx / npic / 2.)
        p[i].vol = p_vol
        p[i].vol0 = p_vol
        p[i].m = p_mass
        p[i].td = mat2x2([1, 0], [0, 1])

    for j in range(pb1, pb1+pb2):
        i = j - pb1
        a = i % (gx2 * npic)
        b = i // (gx2 * npic)
        p[j].bID = 1
        p[j].x = vec2f(px2 + dx / npic / 2. + a * pdx, py2 + dx / npic / 2. + b * pdx)
        syy = -p_rho * gravity[1] * (py2 + dx / npic / 2. + b * pdx - 5.)
        sxx = syy * nu * (1. - nu)
        p[j].stress[0, 0] = sxx
        p[j].stress[1, 1] = syy
        p[j].lp = vec2f(dx / npic / 2., dx / npic / 2.)
        p[j].vol = p_vol
        p[j].vol0 = p_vol
        p[j].m = p_mass
        p[j].td = mat2x2([1, 0], [0, 1])

@ti.kernel
def solve():
    for i, j in g:
        g[i, j].p = vec2f(0, 0)
        g[i, j].f = vec2f(0, 0)
        g[i, j].m = 0
        g[i, j].norm = vec2f(0, 0)

    for i in range(n_particles):
        pos = p[i].x
        x = ti.floor(pos[0] / dx, int) - 1
        y = ti.floor(pos[1] / dx, int) - 1
        
        for a, b in ti.ndrange(4, 4):
            grid_idx = x + a
            grid_idy = y + b
            if grid_idx < 0 or grid_idx >= grid_x: continue
            if grid_idy < 0 or grid_idy >= grid_y: continue

            sx = ShapeGIMP(pos[0], grid_idx * dx, inv_dx, p[i].lp[0])
            sy = ShapeGIMP(pos[1], grid_idy * dx, inv_dx, p[i].lp[1])
            s = sx * sy

            if s <= val_lim: continue

            linear_grid_id = int(grid_idx + grid_idy * grid_x)
            g[linear_grid_id, p[i].bID].m += s * p[i].m
            g[linear_grid_id, p[i].bID].p += s * p[i].m * p[i].v

    for i, j in g:
        if g[i, j].m > val_lim:
            idx = i % grid_x
            idy = i // grid_x

            if idx <= space: 
                g[i, j].p[0] = 0.
                g[i, j].f[0] = 0.
            elif idx >= grid_x - space - 1:
                g[i, j].p[0] = 0.
                g[i, j].f[0] = 0.

            if idy <= space: 
                g[i, j].p = vec2f(0, 0)
                g[i, j].f = vec2f(0, 0)
            elif idy >= grid_x - space - 1:
                g[i, j].p = vec2f(0, 0)
                g[i, j].f = vec2f(0, 0)

    for i in range(n_particles):
        pos = p[i].x
        x = ti.floor(pos[0] / dx, int) - 1
        y = ti.floor(pos[1] / dx, int) - 1
        
        gradv = mat2x2([[0, 0], [0, 0]])
        for a, b in ti.ndrange(4, 4):
            grid_idx = x + a
            grid_idy = y + b
            if grid_idx < 0 or grid_idx >= grid_x: continue
            if grid_idy < 0 or grid_idy >= grid_y: continue

            sx = ShapeGIMP(pos[0], grid_idx * dx, inv_dx, p[i].lp[0])
            sy = ShapeGIMP(pos[1], grid_idy * dx, inv_dx, p[i].lp[1])
            gsx = GShapeGIMP(pos[0], grid_idx * dx, inv_dx, p[i].lp[0])
            gsy = GShapeGIMP(pos[1], grid_idy * dx, inv_dx, p[i].lp[1])
            s = sx * sy
            gs = vec2f(gsx * sy, gsy * sx)

            if s <= val_lim: continue
            
            linear_grid_id = int(grid_idx + grid_idy * grid_x)
            gradv += (g[linear_grid_id, p[i].bID].p / g[linear_grid_id, p[i].bID].m).outer_product(gs)
        
        de = 0.5 * (gradv + gradv.transpose()) * dt
        p[i].td = (mat2x2([1, 0], [0, 1]) + gradv * dt) @ p[i].td
        p[i].vol = p[i].vol0 * p[i].td.determinant()
        if p[i].bID==0:
            p[i].stress += 2. * mu * de + la * de.trace() * mat2x2([1, 0], [0, 1])
            ms = p[i].stress.trace() / 2.
            ss = p[i].stress - ms * mat2x2([1, 0], [0, 1])
            j2sqr = ti.sqrt(0.5 * (ss * ss).sum())
            f = j2sqr + A_dp * ms - B_dp * c
            if f > 0.:
                if A_dp * (ms - j2sqr / mu * k * Ad_dp) - B_dp * c < 0.:
                    dl = (j2sqr + A_dp * ms - B_dp * c) / (mu + A_dp * k * Ad_dp)
                    p[i].stress -= dl * (mu / j2sqr * ss + k * Ad_dp * mat2x2([1, 0], [0, 1]))
                else:
                    p[i].stress = B_dp * c / A_dp * mat2x2([1, 0], [0, 1])
        elif p[i].bID==1:
            p[i].stress += 2. * mu * de + la * de.trace() * mat2x2([1, 0], [0, 1])

        
    for i in range(n_particles):
        pos = p[i].x
        x = ti.floor(pos[0] / dx, int) - 1
        y = ti.floor(pos[1] / dx, int) - 1
        
        for a, b in ti.ndrange(4, 4):    
            grid_idx = x + a
            grid_idy = y + b
            if grid_idx < 0 or grid_idx >= grid_x: continue
            if grid_idy < 0 or grid_idy >= grid_y: continue

            sx = ShapeGIMP(pos[0], grid_idx * dx, inv_dx, p[i].lp[0])
            sy = ShapeGIMP(pos[1], grid_idy * dx, inv_dx, p[i].lp[1])
            gsx = GShapeGIMP(pos[0], grid_idx * dx, inv_dx, p[i].lp[0])
            gsy = GShapeGIMP(pos[1], grid_idy * dx, inv_dx, p[i].lp[1])
            s = sx * sy
            gs = vec2f(gsx * sy, gsy * sx)

            if s <= val_lim: continue
            
            linear_grid_id = int(grid_idx + grid_idy * grid_x)
            g[linear_grid_id, p[i].bID].f += s * p[i].m * gravity - p[i].vol * gs @ p[i].stress
            g[linear_grid_id, p[i].bID].norm += gs * p[i].vol

    for i, j in g:
        if g[i, j].m > val_lim:
            if g[i, j].p.dot(g[i, j].f) > 0.:
                g[i, j].f -= damp * g[i, j].f.norm() * vsign(g[i, j].p)
            g[i, j].p += g[i, j].f * dt

    for i, j in g:
        if g[i, j].m > val_lim:
            idx = i % grid_x
            idy = i // grid_x

            if idx <= space: 
                g[i, j].p[0] = 0.
                g[i, j].f[0] = 0.
            elif idx >= grid_x - space - 1:
                g[i, j].p[0] = 0.
                g[i, j].f[0] = 0.

            if idy <= space: 
                g[i, j].p = vec2f(0, 0)
                g[i, j].f = vec2f(0, 0)
            elif idy >= grid_x - space - 1:
                g[i, j].p = vec2f(0, 0)
                g[i, j].f = vec2f(0, 0)
    
    for i in range(g.shape[0]):
        if g[i, 0].m > val_lim and g[i, 1].m > val_lim:
            n = (g[i, 0].norm - g[i, 1].norm).normalized()
            crit = (g[i, 0].p * g[i, 1].m - g[i, 1].p * g[i, 0].m).dot(n)
            
            if crit > val_lim:
                tt = (g[i, 0].m + g[i, 1].m) * dt
                nomforce = crit / tt
                cforce = vec2f(0, 0)
                if fricfa > val_lim:
                    cforce = (g[i, 0].p * g[i, 1].m - g[i, 1].p * g[i, 0].m) / tt
                    fstick = cforce - nomforce * n
                    val_fstick = fstick.norm()
                    val_fslip = fricfa * ti.abs(nomforce)
                    if val_fslip < val_fstick:
                        cforce = nomforce * n + val_fslip * (fstick / val_fstick)
                else:
                    cforce = nomforce * n

                idx = i % grid_x
                idy = i // grid_x

                if space < idx < grid_x - space - 1: 
                    g[i, 0].p[0] -= cforce[0] * dt
                    g[i, 0].f[0] -= cforce[0]
                    g[i, 1].p[0] += cforce[0] * dt
                    g[i, 1].f[0] += cforce[0]

                if space < idy < grid_y - space - 1: 
                    g[i, 0].p[1] -= cforce[1] * dt
                    g[i, 0].f[1] -= cforce[1]
                    g[i, 1].p[1] += cforce[1] * dt
                    g[i, 1].f[1] += cforce[1]
    

    for i in range(n_particles):
        pos = p[i].x
        x = ti.floor(pos[0] / dx, int) - 1
        y = ti.floor(pos[1] / dx, int) - 1
        
        acc = vec2f(0, 0)
        vel = vec2f(0, 0)
        for a, b in ti.ndrange(4, 4):
            grid_idx = x + a
            grid_idy = y + b
            if grid_idx < 0 or grid_idx >= grid_x: continue
            if grid_idy < 0 or grid_idy >= grid_y: continue

            sx = ShapeGIMP(pos[0], grid_idx * dx, inv_dx, p[i].lp[0])
            sy = ShapeGIMP(pos[1], grid_idy * dx, inv_dx, p[i].lp[1])
            s = sx * sy

            if s <= val_lim: continue

            linear_grid_id = int(grid_idx + grid_idy * grid_x)
            acc += s * g[linear_grid_id, p[i].bID].f / g[linear_grid_id, p[i].bID].m
            vel += s * g[linear_grid_id, p[i].bID].p / g[linear_grid_id, p[i].bID].m
        
        p[i].v += acc * dt
        p[i].x += vel * dt
        p[i].disp += vel * dt
         
@ti.kernel
def copy():
    for i in range(n_particles):
        visp[i] = ti.cast(vec3f([p[i].x[0], p[i].x[1], 0]) / base, ti.f32)

window = ti.ui.Window('Window Title', (892, 892))
visp = ti.Vector.field(3, ti.f32, n_particles)
init()
solve()
while window.running:
    copy()
    canvas = window.get_canvas()
    canvas.set_background_color((0, 0, 0))
    canvas.circles(visp, p_rad/base, (1, 1, 1))
    solve()
    window.show()

@ti.kernel
def find_max_disp() -> float:
    max_disp = 0.
    for i in range(n_particles):
        ti.atomic_max(max_disp, p[i].disp.norm())
    return max_disp
print(find_max_disp())