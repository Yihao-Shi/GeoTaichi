import taichi as ti
ti.init()

N = 200000
pos1 = ti.field(ti.f32)
vel1 = ti.field(ti.f32)
# AoS placement
ti.root.dense(ti.i, N).place(pos1, vel1)
dt = 1.
k=10000

pos2 = ti.field(ti.f32)
vel2 = ti.field(ti.f32)
# SoA placement
ti.root.dense(ti.i, N).place(pos2)
ti.root.dense(ti.i, N).place(vel2)

@ti.dataclass
class Particle:
    pos: ti.f32
    vel: ti.f32
particle = Particle.field(shape=N)

@ti.data_oriented
class Particles:
    def __init__(self):
        self.particle = Particle.field(shape=N)
        #self.pos = ti.field(ti.f32)
        #self.vel = ti.field(ti.f32)
        #ti.root.dense(ti.i, N).place(self.pos, self.vel)
particles = Particles()    

@ti.kernel
def aos(pos: ti.template(), vel: ti.template(), k: int, dt: float):
    for i in range(N):
        pos[i] += vel[i] * dt
        vel[i] += -k * pos[i] * dt
    
@ti.kernel
def soa(pos: ti.template(), vel: ti.template(), k: int, dt: float):
    for i in range(N):
        pos[i] += vel[i] * dt
        vel[i] += -k * pos[i] * dt
        
@ti.kernel
def dclass(particle: ti.template(), k: int, dt: float):
    for i in range(N):
        particle.pos[i] += particle.vel[i] * dt
        particle.vel[i] += -k * particle.pos[i] * dt
        
@ti.kernel
def dorient(particle: ti.template(), k: int, dt: float):
    for i in range(N):
        particle.pos[i] += particle.vel[i] * dt
        particle.vel[i] += -k * particle.pos[i] * dt
    
aos(pos1, vel1, k, dt)
soa(pos2, vel2, k, dt)
dclass(particle, k, dt)
dorient(particles.particle, k, dt)

import time
start=time.time()
aos(pos1, vel1, k, dt)
end=time.time()
print(end-start)

start=time.time()
soa(pos2, vel2, k, dt)
end=time.time()
print(end-start)

start=time.time()
dclass(particle, k, dt)
end=time.time()
print(end-start)

start=time.time()
dorient(particles.particle, k, dt)
end=time.time()
print(end-start)



