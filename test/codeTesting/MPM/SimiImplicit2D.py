import sys
sys.path.append("/home/eleven/work/GeoTaichi")

import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, kernel_profiler=False, debug=False)

from src.utils.ShapeFunctions import ShapeLinear, GShapeLinear
from src.utils.MatrixSolver.MatrixFreePCG import MatrixFreePCG
from src.utils.MatrixSolver.LinearOperator import LinearOperator

vec2f = ti.types.vector(2, float)
vec2i = ti.types.vector(2, int)
vec3f = ti.types.vector(3, float)
mat2x2 = ti.types.matrix(2, 2, float)

MAXVAL = 1e15
val_lim = 1e-13                                                    # zero limitation
base = 5                                                           # calculation domain
xrange, yrange = 5., 5.                           
damp = 0.0                                                         # damping factor
npic = 2                                                           # number of particles per cell
px1, py1 = 1., 2.                                                  # start point of the beam
lx1, ly1 = 3., 0.5                                                  # size of the beam
dx, inv_dx = 0.1, 10.                                                 # grid size
dt = 1e-2                                                          # time step
p_rho = 2500                                                       # density
space = px1 * inv_dx                                               # left Dirichlet boundary
gravity = vec2f(0., -9.8)                                          # gravity
inf_nodes = 4

pi = 3.1415926536
e, nu = 2e7, 0.3
mu = 0.5 * e / (1 + nu)
la = e * nu / (1 + nu) / (1 - 2. * nu)
a1 = e * (1 - nu) / ((1 + nu) * (1. - 2. * nu))
a2 = a1 * nu / (1. - nu)
shear = 0.5 * e / (1. + nu)

ex1, ey1 = lx1 * inv_dx, ly1 * inv_dx
pb1 = int(ex1 * ey1 * npic * npic)
p_vol = dx * dx / npic / npic 
p_mass = p_vol * p_rho
p_rad = dx / npic / 2.
n_particles = pb1
pdx = dx / npic
grid_x, grid_y = int(xrange * inv_dx + 1), int(yrange * inv_dx + 1)
cell_x, cell_y = int(xrange * inv_dx), int(yrange * inv_dx)
dofs = int(2 * 2 * (ex1 + 1) * (ey1 + 1))

# newmark integration parameters
gamma=0.5
beta=0.25
iter_max=1000

@ti.dataclass
class Particle:
    bID: int
    x: vec2f
    disp: vec2f
    traction: vec2f
    v: vec2f
    a: vec2f
    m: float
    vol: float
    lp: vec2f
    stress: mat2x2

@ti.dataclass
class Grid:
    m: float
    v: vec2f
    extf: vec2f
    intf: vec2f
    a: vec2f
    disp: vec2f
    dof: vec2i
    du: vec2f

@ti.dataclass
class DispConstranits:
    value: float
    dof: int
 
active_id = ti.field(int, shape=())                                         # assign index for activated nodes                   
x = ti.field(dtype=float, shape=int(dofs))                                  # x
b = ti.field(dtype=float, shape=int(dofs))                                  # right vector
M = ti.field(dtype=float, shape=int(dofs))                                  # use for pcg

p = Particle.field(shape=n_particles)                                       # particles
g = Grid.field(shape=grid_x * grid_y)                                       # grids
disp_constraint = DispConstranits.field(shape=2 * grid_x * grid_y)            # Dirichlet boundary (have not been used)

LnID = ti.field(int)
shape = ti.field(float)
dshape = ti.Vector.field(2, float)
ti.root.dense(ti.ij, (n_particles, inf_nodes)).place(LnID, shape, dshape)
offset = ti.field(int, shape=n_particles)


@ti.kernel
def particle_init():
    for i in range(pb1):
        a = i % (ex1 * npic)
        b = i // (ex1 * npic)
        p[i].bID = 0
        p[i].x = vec2f(px1 + dx / npic / 2. + a * pdx, py1 + dx / npic / 2. + b * pdx)
        p[i].lp = vec2f(dx / npic / 2., dx / npic / 2.)
        p[i].vol = p_vol
        p[i].m = p_mass

@ti.kernel
def grid_init():
    for i in g:
        g[i].v = vec2f(0, 0)
        g[i].extf = vec2f(0, 0)
        g[i].intf = vec2f(0, 0)
        g[i].m = 0
        g[i].a = vec2f(0, 0)
        g[i].dof = vec2i(-1, -1)
        g[i].du = vec2f(0, 0)

@ti.kernel
def shape_init():
    offset.fill(0)
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

'''@ti.kernel
def nodal_conn_init():
    for i in range(nodal_conn.shape[0]):
        nodal_conn[i, 0] = i + i // cell_x
        nodal_conn[i, 1] = i + i // cell_x + 1
        nodal_conn[i, 2] = i + i // cell_x + grid_x + 1
        nodal_conn[i, 3] = i + i // cell_x + grid_x 

@ti.kernel
def mesh_conn_init():
    for i in range(mesh_conn.shape[0]):
        mesh_conn[i, 0] = i - i // grid_x - grid_x
        mesh_conn[i, 1] = i - i // grid_x - grid_x + 1
        mesh_conn[i, 2] = i - i // grid_x - 1
        mesh_conn[i, 3] = i - i // grid_x
        if i // grid_x == 0:
            mesh_conn[i, 0] = -1
            mesh_conn[i, 1] = -1
        if i % grid_x == 0:
            mesh_conn[i, 0] = -1
            mesh_conn[i, 2] = -1
        if i // grid_x == grid_y - 1:
            mesh_conn[i, 2] = -1
            mesh_conn[i, 3] = -1
        if i % grid_x == grid_x - 1:
            mesh_conn[i, 1] = -1
            mesh_conn[i, 3] = -1'''

@ti.kernel
def assign_displacement_constraints():
    for i in g:
        idx = i % grid_x

        if idx <= space:
            disp_constraint[2 * i].value = 0.
            disp_constraint[2 * i].dof = 1
            disp_constraint[2 * i + 1].value = 0.
            disp_constraint[2 * i + 1].dof = 1

@ti.kernel
def grid_reset():
    for i in g:
        if g[i].m > val_lim:
            g[i].v = vec2f(0, 0)
            g[i].extf = vec2f(0, 0)
            g[i].m = 0.
            g[i].a = vec2f(0, 0)
            g[i].disp = vec2f(0, 0)
            g[i].dof = vec2i(-1, -1)

@ti.kernel
def mass_vel_acc_g2p():
    for i in range(n_particles):
        for j in range(offset[i]):
            g[LnID[i, j]].m += shape[i, j] * p[i].m
            g[LnID[i, j]].v += shape[i, j] * p[i].m * p[i].v
            g[LnID[i, j]].a += shape[i, j] * p[i].m * p[i].a

@ti.kernel
def ext_force_g2p():
    for i in range(n_particles):
        for j in range(offset[i]):
            g[LnID[i, j]].extf += shape[i, j] * (p[i].traction * p[i].vol ** (2./3.) + p[i].m * gravity)

@ti.kernel
def compute_nodal_vel_acc():
    for i in g:
        if g[i].m > val_lim:
            g[i].v /= g[i].m
            g[i].a /= g[i].m

@ti.kernel
def update_nodal_acc():
    for i in g:
        if g[i].m > val_lim:
            acc = 1. / beta / dt / dt * g[i].disp - 1. / beta / dt * g[i].v - (0.5 / beta - 1.) * g[i].a
            g[i].a = acc

@ti.kernel
def find_active_node():
    active_id[None] = 0
    for i in g:
        if g[i].m > val_lim:
            rowth = ti.atomic_add(active_id[None], 2)
            g[i].dof[0] = rowth
            g[i].dof[1] = rowth + 1

@ti.kernel
def compute_stress_strain():
    for i in range(n_particles):
        gradu = mat2x2([[0, 0], [0, 0]])
        for j in range(offset[i]):
            gradu += g[LnID[i, j]].du.outer_product(dshape[i, j])
        de = 0.5 * (gradu + gradu.transpose())
        p[i].vol *= (mat2x2([1, 0], [0, 1]) + gradu).determinant()
        p[i].stress += 2. * mu * de + la * de.trace() * mat2x2([1, 0], [0, 1])

@ti.kernel
def int_force_p2g():
    for i in range(n_particles):
        for j in range(offset[i]):
            g[LnID[i, j]].intf += - p[i].vol * dshape[i, j] @ p[i].stress

@ti.kernel
def assemble_residual_force():
    for grid_id in g:
        if g[grid_id].m > val_lim:
            b[g[grid_id].dof[0]] = g[grid_id].extf[0] + g[grid_id].intf[0] - g[grid_id].m * (1. / beta / dt / dt * g[grid_id].disp[0] - 1. / beta / dt * g[grid_id].v[0] - (0.5 / beta - 1.) * g[grid_id].a[0])
            b[g[grid_id].dof[1]] = g[grid_id].extf[1] + g[grid_id].intf[1] - g[grid_id].m * (1. / beta / dt / dt * g[grid_id].disp[1] - 1. / beta / dt * g[grid_id].v[1] - (0.5 / beta - 1.) * g[grid_id].a[1])
            
            if disp_constraint[2 * grid_id].dof == 1:
                b[g[grid_id].dof[0]] = MAXVAL * disp_constraint[2 * grid_id].value * M[g[grid_id].dof[0]]
            if disp_constraint[2 * grid_id + 1].dof == 1:
                b[g[grid_id].dof[1]] = MAXVAL * disp_constraint[2 * grid_id + 1].value * M[g[grid_id].dof[1]]

@ti.kernel
def inv_diag_A():
    M.fill(0)
    for i in range(n_particles):
        for j in range(offset[i]):
            for k in range(offset[i]):
                if LnID[i, j] == LnID[i, k]:
                    a = MAXVAL * disp_constraint[2 * LnID[i, k]].dof + 1
                    b = MAXVAL * disp_constraint[2 * LnID[i, k] + 1].dof + 1
                    M[g[LnID[i, j]].dof[0]] += a * (dshape[i, j][0] * dshape[i, k][0] * a1 + dshape[i, j][1] * dshape[i, k][1] * shear) * p[i].vol
                    M[g[LnID[i, j]].dof[1]] += b * (dshape[i, j][1] * dshape[i, k][1] * a1 + dshape[i, j][0] * dshape[i, k][0] * shear) * p[i].vol
            
    for grid_id in g:
        if g[grid_id].m > val_lim:  
            a = MAXVAL * disp_constraint[2 * grid_id].dof + 1
            b = MAXVAL * disp_constraint[2 * grid_id + 1].dof + 1         
            M[g[grid_id].dof[0]] += a * (g[grid_id].m / dt / dt / beta)
            M[g[grid_id].dof[1]] += b * (g[grid_id].m / dt / dt / beta)

@ti.kernel
def cg(v: ti.template(), mv: ti.template()):
    mv.fill(0)
    for i in range(n_particles):
        for j in range(offset[i]):
            for k in range(offset[i]):
                a, b = 1., 1.
                if LnID[i, j] == LnID[i, k]:
                    a = MAXVAL * disp_constraint[2 * LnID[i, k]].dof + 1
                    b = MAXVAL * disp_constraint[2 * LnID[i, k] + 1].dof + 1
                mv[g[LnID[i, j]].dof[0]] += (a * (dshape[i, j][0] * dshape[i, k][0] * a1 + dshape[i, j][1] * dshape[i, k][1] * shear) * v[g[LnID[i, k]].dof[0]] + (dshape[i, j][0] * dshape[i, k][1] * a2 + dshape[i, j][1] * dshape[i, k][0] * shear) * v[g[LnID[i, k]].dof[1]]) * p[i].vol
                mv[g[LnID[i, j]].dof[1]] += ((dshape[i, j][1] * dshape[i, k][0] * a2 + dshape[i, j][0] * dshape[i, k][1] * shear) * v[g[LnID[i, k]].dof[0]] + b * (dshape[i, j][1] * dshape[i, k][1] * a1 + dshape[i, j][0] * dshape[i, k][0] * shear) * v[g[LnID[i, k]].dof[1]]) * p[i].vol

    for grid_id in g:
        if g[grid_id].m > val_lim:      
            a = MAXVAL * disp_constraint[2 * grid_id].dof + 1
            b = MAXVAL * disp_constraint[2 * grid_id + 1].dof + 1 
            mv[g[grid_id].dof[0]] += a * (g[grid_id].m / dt / dt / beta) * v[g[grid_id].dof[0]]
            mv[g[grid_id].dof[1]] += b * (g[grid_id].m / dt / dt / beta) * v[g[grid_id].dof[1]] 


@ti.kernel
def update_nodal_disp():
    for grid_id in g:
        if g[grid_id].m > val_lim:
            g[grid_id].disp += vec2f(x[g[grid_id].dof[0]], x[g[grid_id].dof[1]])
            g[grid_id].du = vec2f(x[g[grid_id].dof[0]], x[g[grid_id].dof[1]])

@ti.kernel
def compute_disp_error() -> float:
    delta_u = 0.
    u = 0.
    for grid_id in g:
        if g[grid_id].m > val_lim:
            delta_u += g[grid_id].du[0] ** 2 + g[grid_id].du[1] ** 2
            u += g[grid_id].disp[0] ** 2 + g[grid_id].disp[1] ** 2
    return ti.sqrt(delta_u / u) 

@ti.kernel
def compute_residual_error() -> float:
    rfs = 0.
    for i in range(active_id[None]):
        rfs += b[2 * i] * b[2 * i] + b[2 * i + 1] * b[2 * i + 1]
    return ti.sqrt(rfs)
      
@ti.kernel
def advent_particles():
    for i in range(n_particles):
        acc = vec2f(0, 0)
        disp = vec2f(0, 0)
        for j in range(offset[i]):
            acc += shape[i, j] * g[LnID[i, j]].a
            disp += shape[i, j] * g[LnID[i, j]].disp
        p[i].v += 0.5 * (acc + p[i].a) * dt
        p[i].a = acc
        p[i].x += disp


@ti.kernel
def matrix_reset():
    for i in x:
        x[i] = 0.
        b[i] = 0.

@ti.kernel
def nodal_force_reset():
    for i in g:
        if g[i].m > val_lim:
            g[i].intf = vec2f(0, 0)

iter_max = 100
def solve(A, matrix_free):
    grid_reset()
    shape_init()
    mass_vel_acc_g2p()
    ext_force_g2p()
    find_active_node()
    compute_nodal_vel_acc()

    iter_num = 0
    convergence = False
    while not convergence and iter_num < iter_max:
        matrix_reset()
        nodal_force_reset()
        int_force_p2g()

        assemble_residual_force()
        inv_diag_A()
        matrix_free.solve(A, b, x, M, active_id[None], maxiter=10 * active_id[None], tol=1e-15)
        update_nodal_disp()
        compute_stress_strain()
        convergence = compute_disp_error() < 1e-3 #or compute_residual_error() < 1e-7
        iter_num += 1
    update_nodal_acc()
    advent_particles()
    
@ti.kernel
def copy():
    for i in range(n_particles):
        visp[i] = ti.cast(vec3f([p[i].x[0], p[i].x[1], 0]) / base, ti.f32)

window = ti.ui.Window('Window Title', (892, 892))
visp = ti.Vector.field(3, ti.f32, n_particles)

def log_info(disp=True, vector=False, particle=False):
    if disp:
        for i in range(g.shape[0]):
            if g[i].m > val_lim:
                print(i, g[i].disp)
    if vector:
        for i in range(b.shape[0]):
            print(b[i], x[i])
    if particle:
        for i in range(p.shape[0]):
            print(i, p[i].x, p[i].v, p[i].a)

def main():
    t=0.
    particle_init()
    grid_init()
    A = LinearOperator(cg)
    matrix_free = MatrixFreePCG(int(dofs))
    assign_displacement_constraints()
    '''solve(A, matrix_free)
    inv_diag_A()
    print(M)
    print(b)
    print(x)'''

    while window.running:
        copy()
        canvas = window.get_canvas()
        canvas.set_background_color((0, 0, 0))
        canvas.circles(visp, p_rad/base, (1, 1, 1))
        solve(A, matrix_free)
        window.show()
        t+=dt
        #print(t)
main()

log_info(particle=True)
ti.profiler.print_kernel_profiler_info()