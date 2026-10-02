import taichi as ti

from src.utils.constants import LThreshold
from src.utils.linalg import no_operation


@ti.data_oriented
class DomainBoundary:
    def __init__(self, domain) -> None:
        self.domain = domain
        self.xboundary = None
        self.yboundary = None
        self.zboundary = None
        self.need_run = 0

    def find_boundary_function(self, boundary_condition):
        if boundary_condition == 0:
            return self.reflect_boundary
        elif boundary_condition == 1:
            return self.destroy_boundary
        elif boundary_condition == 2:
            return self.period_boundary
        elif boundary_condition == -1:
            return self.none_boundary
        else:
            raise RuntimeError("Unknown boundary")
        
    def set_boundary_condition(self, boundary):
        if boundary[0] != -1 or boundary[1] != -1 or boundary[2] != -1:
            self.xboundary = self.find_boundary_function(boundary[0])
            self.yboundary = self.find_boundary_function(boundary[1])
            self.zboundary = self.find_boundary_function(boundary[2])
            self.need_run = 1

    @ti.kernel
    def apply_boundary_conditions(self, particleNum: int, particle: ti.template()) -> int:
        not_in_xdomain, not_in_ydomain, not_in_zdomain = 0, 0, 0
        for np in range(particleNum):
            if int(particle[np].active) == 1:
                not_in_xdomain |= self.xboundary(np, 0, particle)
                not_in_ydomain |= self.yboundary(np, 1, particle)
                not_in_zdomain |= self.zboundary(np, 2, particle)
        return not_in_xdomain | not_in_ydomain | not_in_zdomain

    @ti.func
    def none_boundary(self, np, axis, particle):
        return 0

    @ti.func
    def destroy_boundary(self, np, axis: ti.template(), particle):
        in_domain = 1
        position = particle[np].x
        if in_domain == 1 and position[axis] < 0.: 
            in_domain = 0
            particle[np].active = ti.u8(0)
        elif in_domain == 1 and position[axis] > self.domain[axis]: 
            in_domain = 0
            particle[np].active = ti.u8(0)
        return not in_domain

    @ti.func
    def reflect_boundary(self, np, axis: ti.template(), particle):
        in_domain = 1
        position = particle[np].x
        if in_domain == 1 and position[axis] < 0.: 
            in_domain = 0
            particle[np].x[axis] = LThreshold * self.domain[axis]
            particle[np].v[axis] = -particle[np].v[axis]
        elif in_domain == 1 and position[axis] > self.domain[axis]: 
            in_domain = 0
            particle[np].x[axis] = (1. - LThreshold) * self.domain[axis]
            particle[np].v[axis] = -particle[np].v[axis]
        return 0

    @ti.func
    def period_boundary(self, np, axis: ti.template(), particle):
        in_domain = 1
        position = particle[np].x
        if in_domain == 1 and position[axis] < 0.: 
            in_domain = 0
            particle[np].x[axis] += self.domain[axis]
        elif in_domain == 1 and position[axis] > self.domain[axis]: 
            in_domain = 0
            particle[np].x[axis] -= self.domain[axis]
        return 0