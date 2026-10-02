import taichi as ti
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import string

real = ti.f32
ti.init(arch=ti.cpu, kernel_profiler=True, debug=True)# CPU Parallel switch
#ti.init(arch=ti.gpu, kernel_profiler=True)# gpu Parallel switch， 
                                           # if there is a NVIDIA gpu in the computer or clusters, 
                                           # CUDA is chosen automatically, or OpenGL and mental will be chosen.
#ti.init(arch=ti.cuda, use_unified_memory=False, device_memory_fraction=0.8) # GPU-GUDA Parallel switch


# control parmeters
ntime = 300
itime = 0 #
itime_first = 0 # 
itime_last = 100 # 
list_line_number = 0
snap_n = 2

rtime_current = 0.0 #
rtime_first = 0.0 # 
rtime_last = 0.0 #
dt = 0.01 # 
cmap_name = 'magma_r' # python colormap
# mesh parameters
n = ti.Vector([32,32,16])# number of cells in three directions, don't include the ghost cells 0 and n+1
start = ti.Vector([0.0,0.0,0.0])# coordinate-start-points in three directions
finish = ti.Vector([1.0,1.0,1.0])
delta = ti.Vector([(finish[0]-start[0]) / n[0], (finish[1]-start[1]) / n[1], (finish[2]-start[2]) / n[2]]) #mesh delta in three directions
idelta = 1. / delta
# mesh edge coordinates
x = ti.field(real, shape=n[0]+3)
y = ti.field(real, shape=n[1]+3)
z = ti.field(real, shape=n[2]+3)
# mesh center coordinates
xc = ti.field(real, shape=n[0]+2)
yc = ti.field(real, shape=n[1]+2)
zc = ti.field(real, shape=n[2]+2)
# mesh intervals
dx = ti.field(real, shape=n[0]+2)
dy = ti.field(real, shape=n[1]+2)
dz = ti.field(real, shape=n[2]+2)
# reciprocal mesh intervals
rdx = ti.field(real, shape=n[0]+2)
rdy = ti.field(real, shape=n[1]+2)
rdz = ti.field(real, shape=n[2]+2)
# mesh central intervals
dxbar = ti.field(real, shape=n[0]+1)
dybar = ti.field(real, shape=n[1]+1)
dzbar = ti.field(real, shape=n[2]+1)
# reciprocal mesh central intervals
rdxbar = ti.field(real, shape=n[0]+1)
rdybar = ti.field(real, shape=n[1]+1)
rdzbar = ti.field(real, shape=n[2]+1)

@ti.dataclass
class Particle:
    m: real
    vol: real
    x: ti.types.vector(3, real)
    v: ti.types.vector(3, real)
    C: ti.types.matrix(3, 3, real)
particle_num = n[0]*n[1]*n[2]*8
particle = Particle.field(shape=n[0]*n[1]*n[2]*8)

# six velocity boundary conditons flags - xm, xp, zm, zp, ym,yp
# xm - western   boundary  
# xp - eastern   boundary     
# zm - southern  boundary       
# zp - northern  boundary     
# ym - bottom    boundary  
# yp - top       boundary  

# 1 - no   slip  boundary
# 2 - free slip  boundary
# 0 - periodic   boundary
xm = 1 # 
xp = 1 # 
zm = 2 # 
zp = 2 # 
ym = 1 # 
yp = 1 # 



# fluid properties
nu = 0.01 # viscosity of the fluid
ro = 1000 # density of the fluid
trap_big_umax = 12.0 # characteristic velocity, for channel flow
trap_big_ran = 1.0 # magnitude of random seeding, for channel flow
trap_big_h = 1.0 # scale height, for channel flow
sph_h = 0.0 # Height of layer occupied by solid, for channel flow
fbody_x = 0.0 # body force in x-direction
fbody_y = 0.0 # body force in y-direction
fbody_z = 0.0 # body force in z-direction

# pressure field
p = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))
px = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))
py = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))
pz = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))
# source term
s = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))
# velocity field
vel = ti.Vector.field(3, real, shape=(n[0]+2, n[1]+2, n[2]+2))
# historical velocity field
hvel = ti.Vector.field(3, real, shape=(n[0]+2, n[1]+2, n[2]+2))
# average cell center field for output data
au = ti.field(real, shape=(n[0] * n[1] * n[2]))
av = ti.field(real, shape=(n[0] * n[1] * n[2]))
aw = ti.field(real, shape=(n[0] * n[1] * n[2]))
ax = ti.field(real, shape=(n[0] * n[1] * n[2]))
ay = ti.field(real, shape=(n[0] * n[1] * n[2]))
az = ti.field(real, shape=(n[0] * n[1] * n[2]))
ap = ti.field(real, shape=(n[0] * n[1] * n[2]))
line = ti.field(ti.i32, shape=()) # number of line of tecplot output data
# pressure gradient fiedld
pg = ti.Vector.field(3, real, shape=(n[0]+2, n[1]+2, n[2]+2))
# wall distance
wd = ti.Vector.field(3, real, shape=(n[0]+2, n[1]+2, n[2]+2))
# stress fields
txx = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))
txy = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))
txz = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))
tyx = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))
tyy = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))
tyz = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))
tzx = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))
tzy = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))
tzz = ti.field(real, shape=(n[0]+2, n[1]+2, n[2]+2))

M = ti.Vector.field(7, real, shape=(n[0]+2, n[1]+2, n[2]+2)) # matrix coefficient of PCG


# setup sparse data arrays of preconditioned conjugate gradient (PCG)
n_mg_levels = 2
pre_and_post_smoothing = 2
bottom_smoothing = 50

use_multigrid = True # switch of multigrid preconditiner
max_iters = 1000 # maximum number of PPE iterations 

nx = n[0] 
ny = n[1] 
nz = n[2] 
nx_ext = nx // 2  # number of ext cells set so that that total grid size is still power of 2
nx_tot = 2 * nx
ny_ext = ny // 2  # number of ext cells set so that that total grid size is still power of 2
ny_tot = 2 * ny
nz_ext = nz // 2  # number of ext cells set so that that total grid size is still power of 2
nz_tot = 2 * nz

r = [ti.field(dtype=real) for _ in range(n_mg_levels)]  # residual
zk = [ti.field(dtype=real) for _ in range(n_mg_levels)]  # M^-1 r
xk = ti.field(dtype=real)  # solution
pk = ti.field(dtype=real)  # conjugate gradient
Apk = ti.field(dtype=real)  # matrix-vector product
alpha = ti.field(dtype=real)  # step size
beta = ti.field(dtype=real)  # step size
sum = ti.field(dtype=real)  # storage for reductions

grid = ti.root.pointer(ti.ijk, [nx_tot // 4, ny_tot // 4, nz_tot // 4]).dense(ti.ijk, 4).place(xk, pk, Apk)

for l in range(n_mg_levels):
    grid = ti.root.dense(ti.ijk,
                           [nx_tot // (4 * 2**l), ny_tot // (4 * 2**l), nz_tot // (4 * 2**l)]).dense(ti.ijk,
                                                                               4).place(r[l], zk[l])

ti.root.place(alpha, beta, sum)

#-----------------------------------------------------------------------------#
# Defination of the functions of fluid calculation
#-----------------------------------------------------------------------------#
@ti.kernel
def init_material_points():
    for index in particle:
        ip = (index % (4 * n[0] * n[1])) % (2 * n[0])
        jp = (index % (4 * n[0] * n[1])) // (2 * n[0])
        kp = index // (4 * n[0] * n[1])
        particle[index] = [ip, jp, kp]

@ti.func
def init_u(x,y,z):
    
    temp = 0.0
    if y > sph_h:
        yx = y - sph_h
        temp = trap_big_umax*(yx/trap_big_h)*(2.0-(yx/trap_big_h))+trap_big_ran*2.0*(ti.random(float)-0.5)
    return temp

@ti.func
def init_v(x,y,z):
    # return 2.0*(ts.randgen.rand()-0.5)/100000
    return 0

@ti.func
def init_w(x,y,z):
    # return 2.0*(ts.randgen.rand()-0.5)/100000
    return 0

@ti.func
def init_p(x,y,z):
    return 0    


@ti.kernel
def generate_cell_coordinate():
    #print(22222222222222)
    # cell edge coordinates in three directions
    for i in x:
        x[i] = start[0] + (i-1) * delta[0]
    for j in y:
        y[j] = start[1] + (j-1) * delta[1]
    for k in z:
        z[k] = start[2] + (k-1) * delta[2]  
    # cell center coordinates in three directions
    for i in xc:
        xc[i] = 0.5 * (x[i] + x[i+1])     
    for i in yc:
        yc[i] = 0.5 * (y[i] + y[i+1])       
    for i in zc:
        zc[i] = 0.5 * (z[i] + z[i+1])  
    # cell intervals and its reciprocal in three directions
    for i in dx:
        dx[i] = x[i+1] - x[i]
        rdx[i] = 1.0 / dx[i]
    for i in dy:
        dy[i] = y[i+1] - y[i]
        rdy[i] = 1.0 / dy[i]
    for i in dz:
        dz[i] = z[i+1] - z[i] 
        rdz[i] = 1.0 / dz[i]   
    # cell central intervals and its reciprocal in three directions
    for i in dxbar:
        dxbar[i] = xc[i+1] - xc[i]
        rdxbar[i] = 1.0 / dxbar[i]
    for i in dybar:
        dybar[i] = yc[i+1] - yc[i]
        rdybar[i] = 1.0 / dybar[i]
    for i in dzbar:
        dzbar[i] = zc[i+1] - zc[i] 
        rdzbar[i] = 1.0 / dzbar[i]         

# set the initial velocity and pressure
@ti.kernel
def initialize_fluid():
    nx= n[0] + 1
    ny= n[1] + 1
    nz= n[2] + 1
    for i, j, k in ti.ndrange((1, nx), (1, ny), (1,nz)):
        vel[i,j,k][0] = init_u(x[i], yc[j], zc[k])
        vel[i,j,k][1] = init_v(xc[i], y[j], zc[k])
        vel[i,j,k][2] = init_w(xc[i], yc[j], z[k])
        p[i,j,k] = init_p(xc[i], yc[j], zc[k])

@ti.func
def ShapeBsplineQ(xp, xg, idx, btype):
    nx = 0.
    d = (xp - xg) * idx
    if btype == 0:
        if d >= 0.5 and d < 1.5:
            nx = (0.5 * d - 1.5) * d + 1.125
        elif d >= -0.5 and d < 0.5:
            nx = -d * d + 0.75
        elif d >= -1.5 and d < 0.5:
            nx = (0.5 * d + 1.5) * d + 1.125
    elif btype == 1:
        if d >= 0. and d < 0.5:
            nx = 1. - d
        elif d >= 0.5 and d < 1.5:
            nx = (0.5 * d - 1.5) * d + 1.125
    elif btype == 2:
        if d >= -1. and d < -0.5:
            nx = 1. + d
        elif d >= -0.5 and d < 0.5:
            nx = -d * d + 0.75
        elif d >= 0.5 and d < 1.5:
            nx = (0.5 * d - 1.5) * d + 1.125
    elif btype == 3:
        if d >= -1.5 and d < -0.5:
            nx = (0.5 * d + 1.5) * d + 1.125
        elif d >= -0.5 and d < 0.5:
            nx = -d * d + 0.75
        elif d >= 0.5 and d < 1.:
            nx = 1. - d
    elif btype == 4:
        if d >= -1.5 and d < -0.5:
            nx = (0.5 * d + 1.5) * d + 1.125
        elif d >= -0.5 and d <= 0.:
            nx = 1. + d
    return nx

@ti.kernel
def p2g():
    for i in particle:
        p_mass = particle[i].m
        position = particle[i].x
        velocity = particle[i].v
        velocity_gradient = particle[i].velocity_gradient
        for d in ti.static(range(3)):
            stagger = 0#0.5 * (1 - ti.Vector.unit(3, d))
            for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * 3)))):
                base = ti.floor(position * idelta - 0.5 - stagger).cast(int)
                grid_id = base + offset
                
                grid_pos = (grid_id + stagger) * delta
                shape_fn = ti.Vector.zero(float, 3)
                for d in ti.static(range(3)):
                    btypes = 0#int(boundary_type[d])
                    shape_fn[d] = ShapeBsplineQ(position[d], grid_pos[d], idelta[d], btypes)
                weight = 1.0
                for d in ti.static(range(3)):
                    weight *= shape_fn[d]
                nmass = weight * p_mass
                momentum = nmass * velocity[d]
                dpos = position - grid_pos
                momentum -= nmass * ti.Vector([velocity_gradient[d, k] for k in ti.static(range(3))]).dot(dpos)
                #node.m[d][grid_id] += nmass
                #node.velocity[d][grid_id] += momentum

@ti.kernel
def boundary_u():
    nx= n[0] + 1
    ny= n[1] + 1
    nz= n[2] + 1

    # western boundary
    for j, k in ti.ndrange((1, ny), (1,nz)):
        if xm == 0:
            vel[0,j,k][0] = vel[nx-1,j,k][0]
        if xm == 1:
            vel[1,j,k][0] = 0.0

    # eastern boundary   
    for j, k in ti.ndrange((1, ny), (1,nz)):
        if xp == 0:
            vel[nx,j,k][0] = vel[1,j,k][0]   
        if xp == 1:
            vel[nx,j,k][0] = 0.0

    # southern boundary   
    for i, j in ti.ndrange((1, nx), (1,ny)):
        if zm == 0:
            vel[i,j,0][0] = vel[i,j,nz-1][0]   
        if zm == 2:
            vel[i,j,0][0] = vel[i,j,1][0] 


    # northern boundary   
    for i, j in ti.ndrange((1, nx), (1,ny)):
        if zp == 0:
            vel[i,j,nz][0] = vel[i,j,1][0]  
        if zp == 2:
            vel[i,j,nz][0] = vel[i,j,nz-1][0] 


    # bottom boundary   
    for i, k in ti.ndrange((1, nx), (1,nz)):
        if ym == 1:
            vel[i,0,k][0] = -vel[i,1,k][0]

    # top boundary   
    for i, k in ti.ndrange((1, nx), (1,nz)):
        if yp == 1:
            vel[i,ny,k][0] = 2*1 - vel[i,ny-1,k][0]  
            # print(vel[i,ny,k][0])


@ti.kernel
def boundary_v():
    nx= n[0] + 1
    ny= n[1] + 1
    nz= n[2] + 1
    # western boundary
    for j, k in ti.ndrange((1, ny), (1,nz)):
        if xm == 0:
            vel[0,j,k][1] = vel[nx-1,j,k][1]
        if xm == 1:
            vel[0,j,k][1] = -vel[1,j,k][1]

    # eastern boundary   
    for j, k in ti.ndrange((1, ny), (1,nz)):
        if xp == 0:
            vel[nx,j,k][1] = vel[1,j,k][1]  
        if xp == 1:
            vel[nx,j,k][1] = -vel[nx-1,j,k][1]  

    # southern boundary   
    for i, j in ti.ndrange((1, nx), (1,ny)):
        if zm == 0:
            vel[i,j,0][1] = vel[i,j,nz-1][1]  
        if zm == 2:
            vel[i,j,0][1] = vel[i,j,1][1]  

    # northern boundary   
    for i, j in ti.ndrange((1, nx), (1,ny)):
        if zp == 0:
            vel[i,j,nz][1] = vel[i,j,1][1]  
        if zp == 2:
            vel[i,j,nz][1] = vel[i,j,nz-1][1] 


    # bottom boundary   
    for i, k in ti.ndrange((1, nx), (1,nz)):
        if ym == 1:
            vel[i,1,k][1] = 0.0

    # top boundary   
    for i, k in ti.ndrange((1, nx), (1,nz)):
        if yp == 1:
            vel[i,ny,k][1] = 0.0


@ti.kernel
def boundary_w():
    nx= n[0] + 1
    ny= n[1] + 1
    nz= n[2] + 1
    # western boundary
    for j, k in ti.ndrange((1, ny), (1, nz)):
        if xm == 0:
            vel[0,j,k][2] = vel[nx-1,j,k][2]
        if xm == 1:
            vel[0,j,k][2] = -vel[1,j,k][2]

    # eastern boundary   
    for j, k in ti.ndrange((1, ny), (1,nz)):
        if xp == 0:
            vel[nx,j,k][2] = vel[1,j,k][2]   
        if xp == 1:
            vel[nx,j,k][2] = -vel[nx-1,j,k][2]

    # southern boundary   
    for i, j in ti.ndrange((1, nx), (1,ny)):
        if zm == 0:
            vel[i,j,0][2] = vel[i,j,nz-1][2]  
        if zm == 2:
            vel[i,j,1][2] = 0.0 


    # northern boundary   
    for i, j in ti.ndrange((1, nx), (1,ny)):
        if zp == 0:
            vel[i,j,nz][2] = vel[i,j,1][2]  
        if zp == 2:
            vel[i,j,nz][2] = 0.0 


    # bottom boundary   
    for i, k in ti.ndrange((1, nx), (1,nz)):
        if ym == 1:
            vel[i,0,k][2] = -vel[i,1,k][2]

    # top boundary   
    for i, k in ti.ndrange((1, nx), (1,nz)):
        if yp == 1:  
            vel[i,ny,k][0] = 2*1 - vel[i,ny-1,k][0]  


@ti.kernel
def hist():
    nx= n[0] + 1
    ny= n[1] + 1
    nz= n[2] + 1

    # Tau_xx
    for i, j, k in ti.ndrange((0, nx), (1,ny), (1,nz)):
        uc = 0.5 * (vel[i+1,j,k][0]+vel[i,j,k][0])
        u_x = rdx[i] * (vel[i+1,j,k][0]-vel[i,j,k][0])
        txx[i,j,k] = nu * (2.0 * u_x) - uc * uc

    # Tau_yy
    for i, j, k in ti.ndrange((1, nx), (0,ny), (1,nz)):
        vc = 0.5 * (vel[i,j+1,k][1]+vel[i,j,k][1])
        v_y = rdy[j] * (vel[i,j+1,k][1]-vel[i,j,k][1])
        tyy[i,j,k] = nu * (2.0 * v_y) - vc * vc    

    # Tau_zz
    for i, j, k in ti.ndrange((1, nx), (1,ny), (0,nz)):
        wc = 0.5 * (vel[i,j,k+1][2]+vel[i,j,k][2])
        w_z = rdz[k] * (vel[i,j,k+1][2]-vel[i,j,k][2])
        tzz[i,j,k] = nu * (2.0 * w_z) - wc * wc       

    # Tau_xy, Tau_yx
    for i, j, k in ti.ndrange((1, nx+1), (1,ny+1), (1,nz)): 
        cxp = 0.5 * dx[i-1] * rdxbar[i-1]
        cxm = 0.5 * dx[i] * rdxbar[i-1]
        cyp = 0.5 * dy[j-1] * rdybar[j-1]
        cym = 0.5 * dy[j] * rdybar[j-1]
        uay = (cyp * vel[i,j,k][0] + cym * vel[i,j-1,k][0])
        vax = (cxp * vel[i,j,k][1] + cxm * vel[i-1,j,k][1])
        u_y = (vel[i,j,k][0] - vel[i,j-1,k][0]) * rdybar[j-1]
        v_x = (vel[i,j,k][1] - vel[i-1,j,k][1]) * rdxbar[i-1]
        txy[i,j,k] = nu * (u_y + v_x) - uay * vax
        tyx[i,j,k] = nu * (v_x + u_y) - uay * vax

    # Tau_xz, Tau_zx
    for i, j, k in ti.ndrange((1, nx+1), (1,ny), (1,nz+1)): 
        cxp = 0.5 * dx[i-1] * rdxbar[i-1]
        cxm = 0.5 * dx[i] * rdxbar[i-1]
        czp = 0.5 * dz[k-1] * rdzbar[k-1]
        czm = 0.5 * dz[k] * rdzbar[k-1]
        uaz = (czp * vel[i,j,k][0] + czm * vel[i,j,k-1][0])
        wax = (cxp * vel[i,j,k][2] + cxm * vel[i-1,j,k][2])
        u_z = (vel[i,j,k][0] - vel[i,j,k-1][0]) * rdzbar[k-1]
        w_x = (vel[i,j,k][2] - vel[i-1,j,k][2]) * rdxbar[i-1]
        txz[i,j,k] = nu * (u_z + w_x) - uaz * wax
        tzx[i,j,k] = nu * (w_x + u_z) - wax * uaz
    
    # Tau_yz, Tau_zy
    for i, j, k in ti.ndrange((1, nx), (1,ny+1), (1,nz+1)): 
        cyp = 0.5 * dy[j-1] * rdybar[j-1]
        cym = 0.5 * dy[j] * rdybar[j-1]
        czp = 0.5 * dz[k-1] * rdzbar[k-1]
        czm = 0.5 * dz[k] * rdzbar[k-1]
        vaz = (czp * vel[i,j,k][1] + czm * vel[i,j,k-1][1])
        way = (cyp * vel[i,j,k][2] + cym * vel[i,j-1,k][2])
        v_z = (vel[i,j,k][1] - vel[i,j,k-1][1]) * rdzbar[k-1]
        w_y = (vel[i,j,k][2] - vel[i,j-1,k][2]) * rdybar[j-1]
        tyz[i,j,k] = nu * (v_z + w_y) - vaz * way
        tzy[i,j,k] = nu * (w_y + v_z) - way * vaz

    # difference the stress-momentum tensor to get the accelerations, then
    # update the velocity fields and historic terms
    # Adams-Bashforth
    c1 = 1.5 * dt
    c2 = -0.5 *dt     
    for i, j, k in ti.ndrange((2, nx), (1,ny), (1,nz)): 
        udot = rdxbar[i-1] * (txx[i,j,k] - txx[i-1,j,k])\
             + rdy[j] * (tyx[i,j+1,k] - tyx[i,j,k])\
             + rdz[k] * (tzx[i,j,k+1] - tzx[i,j,k])\
             + fbody_x    
        vel[i,j,k][0] = vel[i,j,k][0] + c1 * udot +c2 * hvel[i,j,k][0]
        hvel[i,j,k][0] = udot


    for i, j, k in ti.ndrange((1, nx), (2,ny), (1,nz)): 
        vdot = rdx[i] * (txy[i+1,j,k] - txy[i,j,k])\
             + rdybar[j-1] * (tyy[i,j,k] - tyy[i,j-1,k])\
             + rdz[k] * (tzy[i,j,k+1] - tzy[i,j,k])\
             + fbody_y
        vel[i,j,k][1] = vel[i,j,k][1] + c1 * vdot +c2 * hvel[i,j,k][1]
        hvel[i,j,k][1] = vdot

    for i, j, k in ti.ndrange((1, nx), (1,ny), (2,nz)): 
        wdot = rdx[i] * (txz[i+1,j,k] - txz[i,j,k])\
             + rdy[j] * (tyz[i,j+1,k] - tyz[i,j,k])\
             + rdzbar[k-1] * (tzz[i,j,k] - tzz[i,j,k-1])\
             + fbody_z    
        vel[i,j,k][2] = vel[i,j,k][2] + c1 * wdot +c2 * hvel[i,j,k][2]
        hvel[i,j,k][2] = wdot


# calculate the right-hand term of pressure Poisson Equation
@ti.kernel
def source():
    nx= n[0] + 1
    ny= n[1] + 1
    nz= n[2] + 1
    sc = 2.0 / (3.0 * dt) # Adams Bashforth
    for i, j, k in ti.ndrange((1, nx), (1,ny), (1,nz)): 
        s[i,j,k] = (vel[i+1,j,k][0]-vel[i,j,k][0]) * rdx[i]\
                 + (vel[i,j+1,k][1]-vel[i,j,k][1]) * rdy[j]\
                 + (vel[i,j,k+1][2]-vel[i,j,k][2]) * rdz[k]
        s[i,j,k] = sc * s[i,j,k]  


# compute the pressure gradient
@ti.kernel
def pressure_gradient():
    nx= n[0] + 1
    ny= n[1] + 1
    nz= n[2] + 1
    for i,j, k in ti.ndrange((0, nx+1), (0,ny+1), (0,nz+1)): 
        if i != 0:
            px[i,j,k] = (p[i,j,k]-p[i-1,j,k]) * rdxbar[i-1]
        if j != 0:
            py[i,j,k] = (p[i,j,k]-p[i,j-1,k]) * rdybar[j-1]
        if k != 0:
            pz[i,j,k] = (p[i,j,k]-p[i,j,k-1]) * rdzbar[k-1]


# newu: correct the velocity field for continuity
@ti.kernel
def newu():
    nx= n[0] + 1
    ny= n[1] + 1
    nz= n[2] + 1 
    c1 = 1.5 * dt

    for i, j, k in ti.ndrange((2, nx), (1, ny), (1, nz)):
        vel[i,j,k][0] -= c1 * px[i,j,k]
        hvel[i,j,k][0] -= px[i,j,k]

    for i, j, k in ti.ndrange((1, nx), (2, ny), (1, nz)):
        vel[i,j,k][1] -= c1 * py[i,j,k]
        hvel[i,j,k][1] -= py[i,j,k]

    for i, j, k in ti.ndrange((1, nx), (1, ny), (2, nz)):
        vel[i,j,k][2] -= c1 * pz[i,j,k]
        hvel[i,j,k][2] -= pz[i,j,k]


#-----------------------------------------------------------------------------#
# Defination of the functions of preconditioned conjugate gradient (PCG)
#-----------------------------------------------------------------------------#
# initialize the matrix coefficient according to grid spacing, for the Newman boundary condition
@ti.kernel
def initialize_M():
    nx = n[0] + 1
    ny = n[1] + 1
    nz = n[2] + 1

    for i, j, k in ti.ndrange((1, nx), (1,ny), (1,nz)): 
        M[i,j,k][1] = rdxbar[i-1] * rdx[i]
        M[i,j,k][2] = rdxbar[i] * rdx[i]
        M[i,j,k][3] = rdybar[j-1] * rdy[j]
        M[i,j,k][4] = rdybar[j] * rdy[j]
        M[i,j,k][5] = rdzbar[k-1] * rdz[k]
        M[i,j,k][6] = rdzbar[k] * rdz[k]

# modify the matrix coefficient in west boundary
    for j, k in ti.ndrange((1,ny), (1,nz)): 
        M[1,j,k][1] = 0.0

# modify the matrix coefficient in east boundary
    for j, k in ti.ndrange((1,ny), (1,nz)): 
         M[nx-1,j,k][2] = 0.0

# modify the matrix coefficient in bottom boundary
    for i, k in ti.ndrange((1,nx), (1,nz)): 
        M[i,1,k][3] = 0.0

# modify the matrix coefficient in top boundary
    for i, k in ti.ndrange((1,nx), (1,nz)): 
        M[i,ny-1,k][4] = 0.0

# modify the matrix coefficient in south boundary
    for i, j in ti.ndrange((1,nx), (1,ny)): 
        M[i,j,1][5] = 0.0

# modify the matrix coefficient in north boundary
    for i, j in ti.ndrange((1,nx), (1,ny)): 
        M[i,j,nz-1][6] = 0.0

    for i, j, k in ti.ndrange((1, nx), (1,ny), (1,nz)): 
        M[i,j,k][0] = -1 * (M[i,j,k][1] + M[i,j,k][2]\
                           + M[i,j,k][3] + M[i,j,k][4]\
                           + M[i,j,k][5] + M[i,j,k][6])


@ti.kernel
def initialize_pcg():
    for i, j, k in ti.ndrange((nx_ext, nx_tot - nx_ext), (ny_ext, ny_tot - ny_ext), (nz_ext, nz_tot - nz_ext)):
        zk[0][i, j, k] = 0.0
        Apk[i, j, k] = 0.0
        pk[i, j, k] = 0.0
        xk[i, j, k] = 0.0 # for ppreconditioner, xk is 0 first

    alpha[None] = 0.0
    beta[None] = 0.0


# Ax0 = s    r0 = s − Ax0
@ti.kernel
def initialize_r():
    for i, j, k in ti.ndrange((nx_ext, nx_tot - nx_ext), (ny_ext, ny_tot - ny_ext), (nz_ext, nz_tot - nz_ext)):
        tempi = i - nx_ext + 1
        tempj = j - ny_ext + 1
        tempk = k - nz_ext + 1
        r[0][i, j, k] = s[tempi, tempj, tempk] - M[tempi,tempj,tempk][0] * xk[i, j, k]\
                                               - M[tempi,tempj,tempk][1] * xk[i-1, j, k]\
                                               - M[tempi,tempj,tempk][2] * xk[i+1, j, k]\
                                               - M[tempi,tempj,tempk][3] * xk[i, j-1, k]\
                                               - M[tempi,tempj,tempk][4] * xk[i, j+1, k]\
                                               - M[tempi,tempj,tempk][5] * xk[i, j, k-1]\
                                               - M[tempi,tempj,tempk][6] * xk[i, j, k+1]


@ti.kernel
def reduce(p: ti.template(), q: ti.template()):
    for I in ti.grouped(p):
        sum[None] += p[I] * q[I] 

@ti.kernel
def restrict(l: ti.template()):
    # temp = 1/ ((2**l)*(2**l))
    for i, j, k in r[l]:
        tempi = i * (2 ** l) - nx_ext + 1
        tempj = j * (2 ** l) - ny_ext + 1
        tempk = k * (2 ** l) - nz_ext + 1
        res = r[l][i, j, k] - M[tempi,tempj,tempk][0] * zk[l][i, j, k]\
                            - M[tempi,tempj,tempk][1] * zk[l][i-1, j, k]\
                            - M[tempi,tempj,tempk][2] * zk[l][i+1, j, k]\
                            - M[tempi,tempj,tempk][3] * zk[l][i, j-1, k]\
                            - M[tempi,tempj,tempk][4] * zk[l][i, j+1, k]\
                            - M[tempi,tempj,tempk][5] * zk[l][i, j, k-1]\
                            - M[tempi,tempj,tempk][6] * zk[l][i, j, k+1]
        r[l + 1][i // 2, j // 2, k // 2] += res * 0.5       

@ti.kernel
def smooth(l: ti.template(), phase: ti.template()):
    # phase = red/black Gauss-Seidel phase
    for i, j, k in r[l]:
        if (i + j + k) & 1 == phase:
            tempi = i * (2 ** l) - nx_ext + 1
            tempj = j * (2 ** l) - ny_ext + 1
            tempk = k * (2 ** l) - nz_ext + 1
            zk[l][i, j, k] = 1/(M[tempi,tempj,tempk][0])\
                                          * (r[l][i, j, k] 
                                          - M[tempi,tempj,tempk][1] * zk[l][i-1, j, k]\
                                          - M[tempi,tempj,tempk][2] * zk[l][i+1, j, k]\
                                          - M[tempi,tempj,tempk][3] * zk[l][i, j-1, k]\
                                          - M[tempi,tempj,tempk][4] * zk[l][i, j+1, k]\
                                          - M[tempi,tempj,tempk][5] * zk[l][i, j, k-1]\
                                          - M[tempi,tempj,tempk][6] * zk[l][i, j, k+1]) 

@ti.kernel
def prolongate(l: ti.template()):
    for I in ti.grouped(zk[l]):
        zk[l][I] = zk[l + 1][I // 2]

def apply_preconditioner():
    zk[0].fill(0)
    for l in range(n_mg_levels - 1):
        for _ in range(pre_and_post_smoothing << l):
            smooth(l, 0)
            smooth(l, 1)
        zk[l + 1].fill(0)
        r[l + 1].fill(0)
        restrict(l)

    for _ in range(bottom_smoothing):
        smooth(n_mg_levels - 1, 0)
        smooth(n_mg_levels - 1, 1)

    for l in reversed(range(n_mg_levels - 1)):
        prolongate(l)
        for _ in range(pre_and_post_smoothing << l):
            smooth(l, 1)
            smooth(l, 0)

@ti.kernel
def update_pk():
    for I in ti.grouped(pk):
        pk[I] = zk[0][I] + beta[None] * pk[I]

@ti.kernel
def compute_Apk():
    for i, j, k in Apk:
        tempi = i - nx_ext + 1
        tempj = j - ny_ext + 1
        tempk = k - nz_ext + 1
        Apk[i,j,k] = M[tempi,tempj,tempk][0] * pk[i, j, k]\
                   + M[tempi,tempj,tempk][1] * pk[i-1, j, k]\
                   + M[tempi,tempj,tempk][2] * pk[i+1, j, k]\
                   + M[tempi,tempj,tempk][3] * pk[i, j-1, k]\
                   + M[tempi,tempj,tempk][4] * pk[i, j+1, k]\
                   + M[tempi,tempj,tempk][5] * pk[i, j, k-1]\
                   + M[tempi,tempj,tempk][6] * pk[i, j, k+1]

@ti.kernel
def update_xk():
    for I in ti.grouped(pk):
        xk[I] += alpha[None] * pk[I]

@ti.kernel
def update_r():
    for I in ti.grouped(pk):
        r[0][I] -= alpha[None] * Apk[I]

@ti.kernel
def get_Poisson_solution():
    for i, j, k in xk:
        tempi = i - nx_ext + 1
        tempj = j - ny_ext + 1
        tempk = k - nz_ext + 1
        p[tempi, tempj, tempk] = xk[i,j,k]      


#-----------------------------------------------------------------------------#
# Defination of the functions of output
#-----------------------------------------------------------------------------#
# prepare for the Tecplot format
@ti.kernel
def calculate_cellcenter_velocity():
    nx= n[0] + 1
    ny= n[1] + 1
    nz= n[2] + 1
    for k, j, i in ti.ndrange((1, nz), (1,ny), (1,nx)):
        line[None] = (k-1)*(ny-1)*(nx-1)+(j-1)*(nx-1) +i -1
        ax[line[None]] = xc[i]
        ay[line[None]] = yc[j]
        az[line[None]] = zc[k]
        ap[line[None]] = p[i,j,k]
        au[line[None]] = 0.5*(vel[i,j,k][0] + vel[i+1,j,k][0])
        av[line[None]] = 0.5*(vel[i,j,k][1] + vel[i,j+1,k][1])
        aw[line[None]] = 0.5*(vel[i,j,k][2] + vel[i,j,k+1][2]) 

# output Tecplot format
def output_data_Tec():
    tu = au.to_numpy()
    tv = av.to_numpy()
    tw = aw.to_numpy()
    tx = ax.to_numpy()
    ty = ay.to_numpy()
    tz = az.to_numpy()
    tp = ap.to_numpy()   
    #xx=dx.to_numpy() 
    filename="/Users/shiyh/work/GeoTaichi/taichi_demo/OutputData/" + "data" + str(itime) + ".dat"
    head="VARIABLES=\"X\",\"Y\",\"Z\",\"U\",\"V\",\"W\",\"P\"\n" + \
         "ZONE SOLUTIONTIME=%d,I=%d, J=%d, K=%d F=POINT" %  (itime,n[0], n[1], n[2])
    np.savetxt(filename, np.column_stack((tx,ty,tz,tu,tv,tw,tp)), fmt='%.6f %.6f %.6f %.6f %.6f %.6f %.6f', header=head, comments='')
    # np.savetxt(filename, np.column_stack((tx,ty,tz,tu,tv,tw,tp)), fmt='%.6f %.6f %.6f %.6f %.6f %.6f %.6f', comments='')
    # np.savetxt('dx.txt',xx)


#-----------------------------------------------------------------------------#
# Star of the solver
#-----------------------------------------------------------------------------#
generate_cell_coordinate() # generte the grid
# initialize_fluid() # initialize fluid condition
pressure_gradient()
initialize_M() # initialize matrix coefficients for Neumann boundary conditions


#start time step cycle
itime = itime_first
rtime_current = rtime_first
while itime<=ntime:
    rtime_current += dt
    # --- start of set boundary condition ---
    boundary_u()
    boundary_v()
    boundary_w()
    # --- end of set boundary condition ---
    hist() # calculate fluid shear forces
    source() # calculate the source term for Possion equation


    # ---start of PCG for Possion equation---
    initialize_pcg() # initialize z, xk, pk, Apk
    initialize_r() # initialize r, r=s-Axk

    sum[None] = 0.0
    reduce(r[0], r[0])
    initial_rTr = sum[None]
    # print(initial_rTr)

    if use_multigrid:
        apply_preconditioner()
    else:
        zk[0].copy_from(r[0])

    update_pk()

    sum[None] = 0.0
    reduce(zk[0], r[0])
    old_zTr = sum[None]
    # print(old_zTr)
    # start the iteration cycle
    for i in range(10000): 
        # alpha = rTr / pkT*Apk
        compute_Apk()
        sum[None] = 0.0
        reduce(pk, Apk)
        pAp = sum[None]
        alpha[None] = old_zTr / pAp

        update_xk() # x = x + alpha p
        
        update_r() # r = r - alpha Ap

         # check for convergence
        sum[None] = 0.0
        reduce(r[0], r[0])
        rTr = sum[None]
        # print(rTr)
        if rTr < initial_rTr * 1.0e-12:
            break

        # z = M^-1 r
        if use_multigrid:
            apply_preconditioner()
        else:
            zk[0].copy_from(r[0])

        # beta = new_rTr / old_rTr
        sum[None] = 0.0
        reduce(zk[0], r[0])
        new_zTr = sum[None]
        beta[None] = new_zTr / old_zTr

        # p = z + beta p
        update_pk()
        old_zTr = new_zTr

        # print(' ')
        # print(i)
        # print(rTr)
    # ---end of PCG for Possion equation---
    get_Poisson_solution()

    pressure_gradient()
    newu()
    # output data
    if itime % snap_n == 0:
        calculate_cellcenter_velocity()
        output_data_Tec()
    # output_plot_data()
    print(itime)
    itime+=1
    
