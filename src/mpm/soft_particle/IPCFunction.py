import taichi as ti
import numpy as np

import src.mpm.config as config
from src.mpm.generator.Ground import Ground


@ti.data_oriented
class PointPointDerivative:
    @ti.func
    def distance(self, pos1, pos2):
        return (pos1 - pos2).norm()

    @ti.func
    def Ddistance_div_Dpoint(self, pos1, pos2):
        r = pos1 - pos2
        norm_r = ti.max(r.norm(), 1e-12)
        derivate = r / norm_r
        return derivate, -derivate

    @ti.func
    def D2distance_div_Dpoint2(self, pos1, pos2):
        r = pos1 - pos2
        norm_r = ti.max(r.norm(), 1e-12)
        one_div_norm_r = 1.0 / norm_r
        temp = one_div_norm_r * (
            ti.Matrix.identity(float, config.DIM) - one_div_norm_r * one_div_norm_r * r.outer_product(r)
        )
        return temp


@ti.data_oriented
class PointGroundDerivative:
    def __init__(self, ground: Ground):
        self.ground = ground

    @ti.func
    def distance(self, ground_id, pos):
        return self.ground.distance(ground_id, pos)

    @ti.func
    def Ddistance_div_Dpoint(self, ground_id):
        return self.ground.norm[ground_id]

    @ti.func
    def D2distance_div_Dpoint2(self):
        return ti.Matrix.zero(float, config.DIM, config.DIM)


@ti.data_oriented
class PointLevelSetDerivative:
    def __init__(self, level_set):
        self.level_set = level_set

    @ti.func
    def distance(self, object_id, pos):
        return self.level_set.distance(object_id, pos)

    @ti.func
    def Ddistance_div_Dpoint(self, object_id, pos):
        return self.level_set.gradient(object_id, pos)

    @ti.func
    def D2distance_div_Dpoint2(self):
        return ti.Matrix.zero(float, config.DIM, config.DIM)
