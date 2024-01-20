import taichi as ti

from src.utils.constants import Threshold
from src.utils.TypeDefination import vec2f, vec2u8, vec3f, vec3u8
from src.utils.VectorFunction import Zero2OneVector, clamp


@ti.dataclass
class AbsorbingConstraint2D:
    node: int
    level: ti.u8
    norm: vec2f
    delta: float
    h_min: float
    alpha: float
    beta: float


@ti.dataclass
class AbsorbingConstraint:
    node: int
    level: ti.u8
    norm: vec3f
    delta: float
    h_min: float
    alpha: float
    beta: float


@ti.dataclass
class FrictionConstraint2D:
    node: int
    level: ti.u8
    mu: float
    norm: vec2f

    @ti.func
    def set_boundary_condition(self, node, level, mu, norm):
        self.node = node
        self.level = ti.u8(level)
        self.mu = mu
        self.norm = norm

    @ti.func
    def clear_boundary_condition(self):
        self.level = ti.u8(255)


@ti.dataclass
class FrictionConstraint:
    node: int
    level: ti.u8
    mu: float
    norm: vec3f

    @ti.func
    def set_boundary_condition(self, node, level, mu, norm):
        self.node = node
        self.level = ti.u8(level)
        self.mu = mu
        self.norm = norm

    @ti.func
    def clear_boundary_condition(self):
        self.level = ti.u8(255)


@ti.dataclass
class TractionConstraint2D:
    node: int
    level: ti.u8
    traction: vec2f

    @ti.func
    def set_boundary_condition(self, node, level, traction):
        self.node = node
        self.level = ti.u8(level)
        self.traction += traction

    @ti.func
    def clear_boundary_condition(self):
        self.level = ti.u8(255)


@ti.dataclass
class TractionConstraint:
    node: int
    level: ti.u8
    traction: vec3f

    @ti.func
    def set_boundary_condition(self, node, level, traction):
        self.node = node
        self.level = ti.u8(level)
        self.traction += traction

    @ti.func
    def clear_boundary_condition(self):
        self.level = ti.u8(255)


@ti.dataclass
class VelocityConstraint2D:
    node: int
    level: ti.u8
    fix_v: vec2u8
    unfix_v: vec2u8
    velocity: vec2f

    @ti.func
    def set_boundary_condition(self, node, level, direction, velocity):
        self.node = node
        self.level = ti.u8(level)
        self.fix_v += ti.cast(direction, ti.u8)
        self.fix_v = ti.cast(clamp(0, 1, ti.cast(self.fix_v, int)), ti.u8)
        self.unfix_v = ti.cast(Zero2OneVector(self.fix_v), ti.u8)
        self.velocity += velocity

    @ti.func
    def clear_boundary_condition(self):
        self.level = ti.u8(255)


@ti.dataclass
class VelocityConstraint:
    node: int
    level: ti.u8
    fix_v: vec3u8
    unfix_v: vec3u8
    velocity: vec3f

    @ti.func
    def set_boundary_condition(self, node, level, direction, velocity):
        self.node = node
        self.level = ti.u8(level)
        self.fix_v += ti.cast(direction, ti.u8)
        self.fix_v = ti.cast(clamp(0, 1, ti.cast(self.fix_v, int)), ti.u8)
        self.unfix_v = ti.cast(Zero2OneVector(self.fix_v), ti.u8)
        self.velocity += velocity

    @ti.func
    def clear_boundary_condition(self):
        self.level = ti.u8(255)


@ti.dataclass
class ReflectionConstraint2D:
    node: int
    level: ti.u8
    norm1: vec2f
    norm2: vec2f
    norm3: vec2f

    @ti.func
    def set_boundary_condition(self, node, level, direction):
        self.node = node
        self.level = ti.u8(level)
        
        if all(ti.abs(self.norm1) < Threshold):
            self.norm1 = direction.normalized()
        else:
            if all(ti.abs(self.norm2) < Threshold):
                self.norm2 = direction.normalized()
            else:
                if all(ti.abs(self.norm3) < Threshold):
                    self.norm3 = direction.normalized()

    @ti.func
    def clear_boundary_condition(self):
        self.level = ti.u8(255)


@ti.dataclass
class ReflectionConstraint:
    node: int
    level: ti.u8
    norm1: vec3f
    norm2: vec3f
    norm3: vec3f

    @ti.func
    def set_boundary_condition(self, node, level, direction):
        self.node = node
        self.level = ti.u8(level)
        
        if all(ti.abs(self.norm1) < Threshold):
            self.norm1 = direction.normalized()
        else:
            if all(ti.abs(self.norm2) < Threshold):
                self.norm2 = direction.normalized()
            else:
                if all(ti.abs(self.norm3) < Threshold):
                    self.norm3 = direction.normalized()

    @ti.func
    def clear_boundary_condition(self):
        self.level = ti.u8(255)

@ti.dataclass
class DisplacementConstraint:
    level: ti.u8
    value: float
    node: int
    dof: ti.u8

    @ti.func
    def set_boundary_condition(self, node, level, dof, value):
        self.value = value
        self.level = ti.u8(level)
        self.node = node
        self.dof = ti.u8(dof)

    @ti.func
    def clear_boundary_condition(self):
        self.level = ti.u8(255)