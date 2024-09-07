import sys
sys.path.append("/home/eleven/work/GeoTaichi")

import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, kernel_profiler=True, debug=False)

from src.utils.ShapeFunctions import ShapeLinear, GShapeLinear
from src.utils.MatrixSolver.MatrixFreePCG import MatrixFreePCG
from src.utils.MatrixSolver.LinearOperator import LinearOperator
from src.utils.PrefixSum import PrefixSumExecutor

vec3i = ti.types.vector(3, int)
vec3f = ti.types.vector(3, float)
mat3x3 = ti.types.matrix(3, 3, float)

MAXVAL = 1e15
val_lim = 1e-15
base = 5
xrange, yrange, zrange = 5., 5., 5.
damp = 0.2
npic = 2
px1, py1, pz1 = 0.5, 1.0, 3.5
lx1, ly1, lz1 = 3., 0.5, 0.5
dx, inv_dx = 0.05, 20
dt = 1e-2
p_rho = 2500
gravity = vec3f(0., 0., -9.8)
alpha = 0.01
inf_nodes = 8

pi = 3.1415926536
e, nu = 2e7, 0.3
mu = 0.5 * e / (1 + nu)
la = e * nu / (1 + nu) / (1 - 2. * nu)
a1 = e * (1 - nu) / ((1 + nu) * (1. - 2. * nu))
a2 = a1 * nu / (1. - nu)
shear = 0.5 * e / (1. + nu)

ex1, ey1, ez1 = lx1 * inv_dx, ly1 * inv_dx, lz1 * inv_dx
pb1 = int(ex1 * ey1 * ez1 * npic * npic * npic)
p_vol = dx * dx * dx / npic / npic / npic 
p_mass = p_vol * p_rho
p_rad = dx / npic / 2.
n_particles = pb1
pdx = dx / npic
grid_x, grid_y, grid_z = int(xrange * inv_dx + 1), int(yrange * inv_dx + 1), int(zrange * inv_dx + 1)
cell_x, cell_y, cell_z = int(xrange * inv_dx), int(yrange * inv_dx), int(zrange * inv_dx)
dofs = int(3 * 3 * (ex1 + 1) * (ey1 + 1) * (ez1 + 1))
ifnode = 24
neighbor_num = 8
space1 = px1 * inv_dx
space2 = (px1 + lx1) * inv_dx

# newmark integration parameters
gamma=0.5
beta=0.25
iter_max=1000

@ti.dataclass
class Particle:
    bID: int
    x: vec3f
    disp: vec3f
    traction: vec3f
    v: vec3f
    a: vec3f
    m: float
    vol: float
    lp: vec3f
    stress: mat3x3
    vol0: float
    stress0: mat3x3

@ti.dataclass
class Grid:
    m: float
    v: vec3f
    extf: vec3f
    intf: vec3f
    a: vec3f
    disp: vec3f
    dof: vec3i
    cf: vec3f

@ti.dataclass
class DispConstranits:
    value: float
    dof: int
 
active_id = ti.field(int, shape=())                                         # assign index for activated nodes    
active_node = ti.field(int, shape=int(dofs))    
x = ti.field(dtype=float, shape=int(dofs))                                  # x
b = ti.field(dtype=float, shape=int(dofs))                                  # right vector
M = ti.field(dtype=float, shape=int(dofs))                                  # use for pcg
gm = ti.field(dtype=float, shape=int(dofs)) 

p = Particle.field(shape=n_particles)                                       # particles
g = Grid.field(shape=grid_x * grid_y * grid_z)                              # grids
disp_constraint = DispConstranits.field(shape=3 * grid_x * grid_y * grid_z) # Dirichlet boundary (have not been used)
disp_list = ti.field(int, shape=())
flag = ti.field(int, shape=grid_x * grid_y * grid_z)
element_flag = ti.field(int, shape=grid_x * grid_y * grid_z)
pse = PrefixSumExecutor(grid_x * grid_y * grid_z)

LnID = ti.field(int)
shape = ti.field(float)
dshape = ti.Vector.field(3, float)
ti.root.dense(ti.ij, (n_particles, inf_nodes)).place(LnID, shape, dshape)
offset = ti.field(int, shape=n_particles)
local_stiffness = ti.field(float, shape=int(ifnode * ifnode * ex1 * ey1 * ez1 * 3 * 3))


@ti.kernel
def particle_init():
    ti.loop_config(serialize=True)
    for i in range(pb1):
        a = (i % ((ex1 * npic) * (ey1 * npic))) % (ex1 * npic)
        b = (i % ((ex1 * npic) * (ey1 * npic))) // (ex1 * npic)
        c = i // ((ex1 * npic) * (ey1 * npic))
        p[i].bID = 0
        p[i].x = vec3f(px1 + dx / npic / 2. + a * pdx, py1 + dx / npic / 2. + b * pdx, pz1 + dx / npic / 2. + c * pdx)
        p[i].lp = vec3f(dx / npic / 2., dx / npic / 2., dx / npic / 2.)
        p[i].vol = p_vol
        p[i].vol0 = p_vol
        p[i].m = p_mass

@ti.kernel
def grid_init():
    for i in g:
        g[i].v = vec3f(0, 0, 0)
        g[i].extf = vec3f(0, 0, 0)
        g[i].intf = vec3f(0, 0, 0)
        g[i].m = 0.
        g[i].a = vec3f(0, 0, 0)
        g[i].dof = vec3i(-1, -1, -1)
        g[i].cf = vec3f(0, 0, 0)

@ti.kernel
def shape_init():
    offset.fill(0)
    element_flag.fill(0)
    for i in range(n_particles):
        pos = p[i].x
        x = ti.floor(pos[0] / dx, int) #- 1
        y = ti.floor(pos[1] / dx, int) #- 1
        z = ti.floor(pos[2] / dx, int) #- 1

        for c, b, a in ti.ndrange(2, 2, 2):
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
        element_flag[int(x + y * grid_x + z * grid_x * grid_y)] = 1

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

        if idx == space1:
            offset = ti.atomic_add(disp_list[None], 3)
            disp_constraint[offset].value = 0.0
            disp_constraint[offset].dof = 3 * i
            disp_constraint[offset + 1].value = 0.
            disp_constraint[offset + 1].dof = 3 * i + 1
            disp_constraint[offset + 2].value = 0.
            disp_constraint[offset + 2].dof = 3 * i + 2

    '''for i in g:
        idx = i % grid_x

        if idx == space2:
            offset = ti.atomic_add(disp_list[None], 3)
            disp_constraint[offset].value = 0.
            disp_constraint[offset].dof = 3 * i
            disp_constraint[offset + 1].value = 0.
            disp_constraint[offset + 1].dof = 3 * i + 1
            disp_constraint[offset + 2].value = 0.
            disp_constraint[offset + 2].dof = 3 * i + 2'''

@ti.kernel
def grid_reset():
    for i in g:
        if g[i].m > val_lim:
            g[i].v = vec3f(0, 0, 0)
            g[i].extf = vec3f(0, 0, 0)
            g[i].m = 0.
            g[i].a = vec3f(0, 0, 0)
            g[i].disp = vec3f(0, 0, 0)
            g[i].dof = vec3i(-1, -1, -1)

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
def compute_mass_list():
    #gm.fill(0)
    for grid_id in g:
        if g[grid_id].m > val_lim:
            gm[g[grid_id].dof[0]] = g[grid_id].m / dt / dt / beta
            gm[g[grid_id].dof[1]] = g[grid_id].m / dt / dt / beta
            gm[g[grid_id].dof[2]] = g[grid_id].m / dt / dt / beta

@ti.kernel
def get_penalty():
    for i in range(disp_list[None]):
        nodeID = disp_constraint[i].dof // 3
        dofID = disp_constraint[i].dof % 3
        if g[nodeID].dof[dofID] >= 0:
            gm[g[nodeID].dof[dofID]] += MAXVAL

@ti.kernel
def update_nodal_acc():
    for i in g:
        if g[i].m > val_lim:
            g[i].a = 1. / beta / dt / dt * g[i].disp - 1. / beta / dt * g[i].v - (0.5 / beta - 1.) * g[i].a

@ti.kernel
def find_active_node():
    flag.fill(0)
    for i in g:
        if g[i].m > val_lim:
            flag[i] = 1


@ti.kernel
def set_active_dof() -> int:
    for i in g:
        if g[i].m > val_lim:
            rowth = flag[i] - 1
            g[i].dof[0] = 3 * rowth
            g[i].dof[1] = 3 * rowth + 1
            g[i].dof[2] = 3 * rowth + 2

            if element_flag[i] - element_flag[i-1]==1:
                active_node[element_flag[i] - 1] = i
    active_id[None] = element_flag[element_flag.shape[0] - 1]
    return 3 * flag[flag.shape[0] - 1]


@ti.kernel
def compute_stress_strain():
    for i in range(n_particles):
        gradu = mat3x3([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for j in range(offset[i]):
            gradu += g[LnID[i, j]].disp.outer_product(dshape[i, j])
        de = 0.5 * (gradu + gradu.transpose())
        p[i].vol = p[i].vol0 * (mat3x3([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) + gradu).determinant()
        p[i].stress = p[i].stress0 + 2. * mu * de + la * de.trace() * mat3x3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

@ti.kernel
def update_stress_strain():
    for i in range(n_particles):
        p[i].vol0 = p[i].vol
        p[i].stress0 = p[i].stress

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
            b[g[grid_id].dof[2]] = g[grid_id].extf[2] + g[grid_id].intf[2] - g[grid_id].m * (1. / beta / dt / dt * g[grid_id].disp[2] - 1. / beta / dt * g[grid_id].v[2] - (0.5 / beta - 1.) * g[grid_id].a[2])
            
    for i in range(disp_list[None]):
        nodeID = disp_constraint[i].dof // 3
        dofID = disp_constraint[i].dof % 3
        if g[nodeID].dof[dofID] >= 0:
            b[g[nodeID].dof[dofID]] += MAXVAL * (disp_constraint[i].value - g[nodeID].disp[dofID])

@ti.kernel
def inv_diag_A():
    for i in range(ifnode * active_id[None]):
        nodeID = active_node[int(i // ifnode)]
        dofID = i % ifnode
        local_dof1 = dofID % 3
        local_node1 = dofID // 3
        nodeID1 = int(nodeID +get_node_id(local_node1%4%2, local_node1%4//2, local_node1//4))
        base_offset = flag[nodeID-1]
        offset = base_offset * ifnode * ifnode + dofID * ifnode

        M[g[nodeID1].dof[local_dof1]] += local_stiffness[offset + dofID] 

    for i in M:   
        M[i] += gm[i]


@ti.kernel
def assemble_local_stiffness():
    local_stiffness.fill(0)
    for i in range(n_particles):
        pvol = p[i].vol
        base_node = LnID[i, 0]
        base_offset = flag[base_node-1]
        for j in range(offset[i]):
            ijdshape = dshape[i, j]
            for k in range(offset[i]):
                ikdshape = dshape[i, k]
                a, b, c = 1., 1., 1.
                local_stiffness[base_offset * ifnode * ifnode + 3 * j + (3 * k) * ifnode] += a * (ijdshape[0] * ikdshape[0] * a1 + (ijdshape[1] * ikdshape[1] + ijdshape[2] * ikdshape[2]) * shear) * pvol
                local_stiffness[base_offset * ifnode * ifnode + 3 * j + (3 * k + 1) * ifnode] += (ijdshape[0] * ikdshape[1] * a2 + ijdshape[1] * ikdshape[0] * shear) * pvol
                local_stiffness[base_offset * ifnode * ifnode + 3 * j + (3 * k + 2) * ifnode] += (ijdshape[0] * ikdshape[2] * a2 + ijdshape[2] * ikdshape[0] * shear) * pvol
                local_stiffness[base_offset * ifnode * ifnode + (3 * j + 1) + (3 * k) * ifnode] += (ijdshape[1] * ikdshape[0] * a2 + ijdshape[0] * ikdshape[1] * shear) * pvol
                local_stiffness[base_offset * ifnode * ifnode + (3 * j + 1) + (3 * k + 1) * ifnode] += b * (ijdshape[1] * ikdshape[1] * a1 + (ijdshape[0] * ikdshape[0] + ijdshape[2] * ikdshape[2]) * shear) * pvol
                local_stiffness[base_offset * ifnode * ifnode + (3 * j + 1) + (3 * k + 2) * ifnode] += (ijdshape[1] * ikdshape[2] * a2 + ijdshape[2] * ikdshape[1] * shear) * pvol
                local_stiffness[base_offset * ifnode * ifnode + (3 * j + 2) + (3 * k) * ifnode] += (ijdshape[2] * ikdshape[0] * a2 + ijdshape[0] * ikdshape[2] * shear) * pvol
                local_stiffness[base_offset * ifnode * ifnode + (3 * j + 2) + (3 * k + 1) * ifnode] += (ijdshape[2] * ikdshape[1] * a2 + ijdshape[1] * ikdshape[2] * shear) * pvol
                local_stiffness[base_offset * ifnode * ifnode + (3 * j + 2) + (3 * k + 2) * ifnode] += c * (ijdshape[2] * ikdshape[2] * a1 + (ijdshape[0] * ikdshape[0] + ijdshape[1] * ikdshape[1]) * shear) * pvol

@ti.func
def get_node_id(a, b, c):
    return a + b * (grid_x) + c * (grid_x) * (grid_y)


@ti.kernel
def cg(v: ti.template(), mv: ti.template()):
    mv.fill(0)
    for i in range(ifnode * active_id[None]):
        nodeID = active_node[int(i // ifnode)]
        
        dofID = i % ifnode
        local_dof1 = dofID % 3
        local_node1 = dofID // 3
        nodeID1 = int(nodeID +get_node_id(local_node1%4%2, local_node1%4//2, local_node1//4))
        base_offset = flag[nodeID-1]
        offset = base_offset * ifnode * ifnode + dofID
        
        sum = 0.
        for j in range(ifnode):
            local_node2 = j // 3
            local_dof2 = j % 3
            nodeID2 = int(nodeID + get_node_id(local_node2%4%2, local_node2%4//2, local_node2//4))
            sum += local_stiffness[offset + j * ifnode] * v[g[nodeID2].dof[local_dof2]]
        
        mv[g[nodeID1].dof[local_dof1]] += sum
        
    for i in mv:   
        mv[i] += gm[i] * v[i]

@ti.kernel
def update_nodal_disp():
    for grid_id in g:
        if g[grid_id].m > val_lim:
            g[grid_id].disp += vec3f(x[g[grid_id].dof[0]], x[g[grid_id].dof[1]], x[g[grid_id].dof[2]])

@ti.kernel
def find_contact_force():
    for i in range(disp_list[None]):
        nodeID = disp_constraint[i].dof // 3
        dofID = disp_constraint[i].dof % 3
        if g[nodeID].dof[dofID] >= 0:
            g[nodeID].cf[dofID] = g[nodeID].disp[dofID] * MAXVAL

@ti.kernel
def compute_disp_error() -> float:
    delta_u = 0.
    u = 0.
    for grid_id in g:
        if g[grid_id].m > val_lim:
            delta_u += x[g[grid_id].dof[0]] ** 2 + x[g[grid_id].dof[1]] ** 2 + x[g[grid_id].dof[2]] ** 2
            u += g[grid_id].disp[0] ** 2 + g[grid_id].disp[1] ** 2 + g[grid_id].disp[2] ** 2
    return ti.sqrt(delta_u / u) 

@ti.kernel
def compute_residual_error() -> float:
    rfs = 0.
    for i in range(active_id[None]):
        rfs += b[3 * i] * b[3 * i] + b[3 * i + 1] * b[3 * i + 1] + b[3 * i + 2] * b[3 * i + 2]
    return ti.sqrt(rfs)
      
@ti.kernel
def advent_particles():
    for i in range(n_particles):
        acc = vec3f(0, 0, 0)
        disp = vec3f(0, 0, 0)
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
        M[i] = 0.
        #b[i] = 0.

@ti.kernel
def nodal_force_reset():
    for i in g:
        if g[i].m > val_lim:
            g[i].intf = vec3f(0, 0, 0)

iter_max = 100
def solve(A, matrix_free):
    grid_reset()
    shape_init()
    mass_vel_acc_g2p()
    ext_force_g2p()
    find_active_node()
    pse.run(flag, grid_x * grid_y * grid_z)
    pse.run(element_flag, grid_x * grid_y * grid_z)
    total_nodes = set_active_dof()
    compute_nodal_vel_acc()
    compute_mass_list()
    get_penalty()

    iter_num = 0
    convergence = False
    while not convergence and iter_num < iter_max:
        matrix_reset()
        nodal_force_reset()
        int_force_p2g()

        assemble_local_stiffness()
        assemble_residual_force()
        inv_diag_A()
        matrix_free.solve(A, b, x, M, total_nodes, maxiter=10 * total_nodes, tol=1e-10)
        convergence = compute_disp_error() < 1e-4 #or compute_residual_error() < 1e-7
        update_nodal_disp()
        compute_stress_strain()
        iter_num += 1
    update_stress_strain()
    update_nodal_acc()
    advent_particles()
    find_contact_force()
    
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
import time as time
def main():
    t=0.
    particle_init()
    grid_init()
    A = LinearOperator(cg)
    matrix_free = MatrixFreePCG(int(dofs))
    assign_displacement_constraints()

    while t<0.5:
        start = time.time()
        solve(A, matrix_free)
        copy()
        camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.LMB)
        scene.set_camera(camera)

        scene.point_light((6., 0.2, 6.), color=(1.0, 1.0, 1.0))
        scene.particles(visp, radius=p_rad, color=(1, 1, 1))

        canvas.set_background_color((0, 0, 0))
        canvas.scene(scene)
        window.show()
        
        end = time.time()
        t+=dt
        print(t, end-start)
main()

#log_info(particle=True)
ti.profiler.print_kernel_profiler_info()