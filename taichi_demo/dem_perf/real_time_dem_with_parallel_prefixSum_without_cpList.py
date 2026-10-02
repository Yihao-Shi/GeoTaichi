import taichi as ti
import math
import os

ti.init(arch=ti.gpu)
vec = ti.math.vec2

SAVE_FRAMES = False

window_size = 1024  # Number of pixels of the window
n = 80000  # Number of grains

density = 100.0
stiffness = 1e4
restitution_coef = 0.5
gravity = -9.81
dt = 0.0001  # Larger dt might lead to unstable results.
substeps = 60


@ti.dataclass
class Grain:
    p: vec  # Position
    m: ti.f32  # Mass
    r: ti.f32  # Radius
    v: vec  # Velocity
    a: vec  # Acceleration
    f: vec  # Force


gf = Grain.field(shape=(n, ))

grain_r_min = 0.001
grain_r_max = 0.0015

grid_size = 2*grain_r_max  # Simulation domain of size [0, 1]
grid_n = int(1.0/grain_r_max)



@ti.kernel
def init():
    for n in gf:
        # Spread grains in a restricted area.
        l = n * grid_size
        padding = 0.1
        region_width = 1.0 - padding * 2
        ppl = region_width // (2*grain_r_max)
        i = n % ppl
        j = n // ppl
        index=0
        if i%2==0: index=1
        pos = padding+vec([(2*i+1)*grain_r_max, (2*j+1)*grain_r_max])+index * grain_r_min
        gf[n].p = pos
        gf[n].r = ti.random() * (grain_r_max - grain_r_min) + grain_r_min
        gf[n].m = density * math.pi * gf[n].r**2


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


@ti.kernel
def resolve():
    for n in range(contact.shape[0]):
        
        if contact[n].end2==-1: continue
        i = contact[n].end1
        j = contact[n].end2
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


list_cur = ti.field(dtype=ti.i32)
grain_count = ti.field(dtype=ti.i32)
ti.root.dense(ti.i,grid_n*grid_n).place(list_cur,grain_count)
contact = ti.Struct.field({"end1": int, "end2": int}, shape=18*n)
particle_id = ti.field(dtype=ti.i32, shape=n, name="particle_id")


@ti.kernel
def calculate_position():
    '''
    Handle the collision between grains.
    '''
    for i in gf:
        gf[i].f = vec(0., gravity * gf[i].m)  # Apply gravity.

    grain_count.fill(0)

    for i in range(n):
        grid_idx = ti.floor(gf[i].p / grid_size, int)
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        grain_count[linear_idx] += 1

@ti.kernel
def search():            
    list_cur.fill(0)
    for i in range(n):
        grid_idx = ti.floor(gf[i].p / grid_size, int)
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        
        grain_location = grain_count[linear_idx] -ti.atomic_add(list_cur[linear_idx], 1)-1
        particle_id[grain_location] = i
        
prefix_particle=ti.field(int, shape=n)                        
@ti.kernel
def update_contact():
    contact.end2.fill(-1)
    prefix_particle.fill(0)

    for i in range(n):
        grid_idx = ti.floor(gf[i].p / grid_size, int)
        x_begin = ti.max(grid_idx[0] - 1, 0)
        x_end = ti.min(grid_idx[0] + 2, grid_n)

        y_begin = ti.max(grid_idx[1] - 1, 0)
        y_end = ti.min(grid_idx[1] + 2, grid_n)
        offset=18*i
        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                neigh_linear_idx = neigh_i * grid_n + neigh_j
                for p_idx in range(grain_count[neigh_linear_idx]-list_cur[neigh_linear_idx],
                                   grain_count[neigh_linear_idx]):
                    j = particle_id[p_idx]
                    if i < j:
                        contact[offset].end1=i
                        contact[offset].end2=j
                        ti.atomic_add(offset,1)
        prefix_particle[i]=offset
                        

init()
gui = ti.GUI('Taichi DEM', (window_size, window_size))
step = 0

if SAVE_FRAMES:
    os.makedirs('output', exist_ok=True)
cell_pse=ti.algorithms.PrefixSumExecutor(int(grid_n*grid_n))
import time 

start=time.time()
while step<5e4:
    #for s in range(substeps):
    update()
    apply_bc()
    calculate_position()
    cell_pse.run(grain_count)
    search()
    update_contact()
    resolve()
    step+=1
    '''pos = gf.p.to_numpy()
    r = gf.r.to_numpy() * window_size
    gui.circles(pos, radius=r)
    if SAVE_FRAMES:
        gui.show(f'output/{step:06d}.png')
    else:
        gui.show()'''
ti.sync()
end=time.time()
print(end-start)
