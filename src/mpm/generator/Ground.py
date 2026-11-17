import taichi as ti
import numpy as np

import src.mpm.config as config


@ti.data_oriented
class Ground:
    def __init__(self):
        self.np_pos = []
        self.np_norm = []
        self.np_vel = []
        self.num = 0

    def append(self, pos, norm, vel=None):
        if vel is None:
            vel = [0] * config.DIM
        self.np_pos.append(pos)
        self.np_norm.append(norm)
        self.np_vel.append(vel)
        self.num += 1

    def finalize(self):
        # Taichi does not allow zero-sized fields.  Keep one unused slot when
        # contact is purely particle/body based and no ground plane was added.
        storage_num = max(1, self.num)
        self.pos = ti.Vector.field(config.DIM, ti.f64, shape=storage_num)
        self.vel = ti.Vector.field(config.DIM, ti.f64, shape=storage_num)
        self.norm = ti.Vector.field(config.DIM, ti.f64, shape=storage_num)
        self.add_ground(np.asarray(self.np_pos), np.asarray(self.np_vel), np.asarray(self.np_norm))

    def add_ground(self, pos: ti.types.ndarray(), vel: ti.types.ndarray(), norm: ti.types.ndarray()):
        for i in range(self.num):
            self.pos[i] = ti.Vector([pos[i, d] for d in ti.static(range(config.DIM))])
            self.vel[i] = ti.Vector([vel[i, d] for d in ti.static(range(config.DIM))])
            self.norm[i] = ti.Vector([norm[i, d] for d in ti.static(range(config.DIM))])

    @ti.kernel
    def move(self, dt: ti.f64):
        for ground_id in range(self.num):
            self.pos[ground_id] += self.vel[ground_id] * dt

    @ti.func
    def distance(self, ground_id, pos):
        return self.norm[ground_id].dot(pos - self.pos[ground_id])

    @ti.func
    def tangent_operator(self, ground_id):
        normal = self.norm[ground_id]
        return ti.Matrix.identity(ti.f64, config.DIM) - normal.outer_product(normal)
