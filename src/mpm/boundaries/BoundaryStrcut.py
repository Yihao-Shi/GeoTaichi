import taichi as ti

from src.utils.constants import DELTA, DELTA2D
from src.utils.TypeDefination import vec2f, vec3f, mat2x2, mat3x3


@ti.dataclass
class AbsorbingConstraint:
    node: int
    level: ti.u8
    dirs: ti.u8
    delta: float
    h_min: float
    alpha: float
    beta: float

    @ti.func
    def clear_boundary_condition(self):
        self.node = -1
        self.level = ti.u8(255)


@ti.dataclass
class FrictionConstraint:
    node: int
    level: ti.u8
    dirs: ti.u8
    signs: ti.i8
    mu: float

    @ti.func
    def set_boundary_condition(self, node, level, mu, dirs, signs):
        self.node = node
        self.level = ti.u8(level)
        self.mu = mu
        self.dirs = ti.u8(dirs)
        self.signs = ti.u8(signs)

    @ti.func
    def clear_boundary_condition(self):
        self.node = -1
        self.level = ti.u8(255)


@ti.dataclass
class TractionConstraint:
    node: int
    level: ti.u8
    dirs: ti.u8
    traction: float

    @ti.func
    def set_boundary_condition(self, node, level, traction):
        self.node = node
        self.level = ti.u8(level)
        self.traction += traction

    @ti.func
    def clear_boundary_condition(self):
        self.node = -1
        self.level = ti.u8(255)


@ti.dataclass
class VelocityConstraint:
    node: int
    level: ti.u8
    dirs: ti.u8
    velocity: float

    @ti.func
    def set_boundary_condition(self, node, level, direction, velocity):
        self.node = node
        self.level = ti.u8(level)
        self.dirs = ti.cast(direction, ti.u8)
        self.velocity = velocity

    @ti.func
    def clear_boundary_condition(self):
        self.node = -1
        self.level = ti.u8(255)


@ti.dataclass
class MacVelocityConstraint:
    node: int
    velocity: float
    signs: ti.u8

    @ti.func
    def set_boundary_condition(self, node, signs, velocity):
        self.node = node
        self.signs = ti.cast(signs, ti.u8)
        self.velocity = velocity

    @ti.func
    def clear_boundary_condition(self):
        self.node = -1


@ti.dataclass
class ReflectionConstraint:
    node: int
    level: ti.u8
    dirs: ti.u8
    signs: ti.i8

    @ti.func
    def set_boundary_condition(self, node, level, direction, signs):
        self.node = node
        self.level = ti.u8(level)
        self.dirs = ti.u8(direction)
        self.signs = ti.i8(signs)

    @ti.func
    def clear_boundary_condition(self):
        self.node = -1
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
        self.node = -1
        self.level = ti.u8(255)


@ti.dataclass
class ParticleLoadNanson:
    pid: int
    traction: vec3f
    psize: vec3f
    td: mat3x3

    @ti.func
    def set_boundary_condition(self, pid, traction, psize):
        self.pid = pid
        self.traction += traction
        self.psize = psize
        self.td = mat3x3([1., 0., 0.], [0., 1., 0.], [0., 0., 1.])

    @ti.func
    def clear_boundary_condition(self):
        self.pid = -1
        self.traction = vec3f(0., 0., 0.)

    @ti.func
    def _compute_traction_force(self):
        # Nanson's formula: \mathbf{n}da = J\dot\mathbf{F}^{-T}\mathbf{N}dA
        psize = self.psize
        return 4. * self.traction @ self.td.inverse().transpose() * self.td.determinant() * vec3f(psize[1] * psize[2], psize[0] * psize[2], psize[0] * psize[1])
    
    @ti.func
    def _calc_psize_cp(self, dt, velocity_gradient):
        deformation_gradient_rate = DELTA + dt[None] * velocity_gradient
        self.td = deformation_gradient_rate @ self.td


@ti.dataclass
class ParticleLoadNanson2D:
    pid: int
    traction: vec2f
    psize: vec2f
    td: mat2x2

    @ti.func
    def set_boundary_condition(self, pid, traction, psize):
        self.pid = pid
        self.traction += traction
        self.psize = psize
        self.td = mat2x2([1., 0.], [0., 1.])

    @ti.func
    def clear_boundary_condition(self):
        self.pid = -1
        self.traction = vec2f(0., 0.)

    @ti.func
    def _compute_traction_force(self):
        # Nanson's formula: \mathbf{n}da = J\dot\mathbf{F}^{-T}\mathbf{N}dA
        psize = self.psize
        return 2. * self.traction @ self.td.inverse().transpose() * self.td.determinant() * vec2f(psize[1], psize[0])
    
    @ti.func
    def _calc_psize_cp(self, dt, velocity_gradient):
        deformation_gradient_rate = DELTA2D + dt[None] * velocity_gradient
        self.td = deformation_gradient_rate @ self.td


@ti.dataclass
class ParticleLoad:
    pid: int
    traction: vec3f
    psize: vec3f

    @ti.func
    def set_boundary_condition(self, pid, traction, psize):
        self.pid = pid
        self.traction += traction
        self.psize = psize

    @ti.func
    def clear_boundary_condition(self):
        self.pid = -1
        self.traction = vec3f(0., 0., 0.)

    @ti.func
    def _compute_traction_force(self):
        psize = self.psize
        return 4. * self.traction * vec3f(psize[1] * psize[2], psize[0] * psize[2], psize[0] * psize[1])
    
    @ti.func
    def _calc_psize_cp(self, dt, velocity_gradient):
        deformation_gradient_rate = DELTA + dt[None] * velocity_gradient
        self.psize[0] *= deformation_gradient_rate[0, 0] 
        self.psize[1] *= deformation_gradient_rate[1, 1]
        self.psize[2] *= deformation_gradient_rate[2, 2] 

    @ti.func
    def _calc_psize_r(self, dt, velocity_gradient):
        # reference: iGIMP: An implicit generalised interpolation material point method for large deformations
        deformation_gradient_rate = DELTA + dt[None] * velocity_gradient
        self.psize[0] *= ti.sqrt(deformation_gradient_rate[0, 0] ** 2 + deformation_gradient_rate[1, 0] ** 2 + deformation_gradient_rate[2, 0] ** 2)
        self.psize[1] *= ti.sqrt(deformation_gradient_rate[0, 1] ** 2 + deformation_gradient_rate[1, 1] ** 2 + deformation_gradient_rate[2, 1] ** 2)
        self.psize[2] *= ti.sqrt(deformation_gradient_rate[0, 2] ** 2 + deformation_gradient_rate[1, 2] ** 2 + deformation_gradient_rate[2, 2] ** 2)

    
@ti.dataclass
class ParticleLoad2D:
    pid: int
    traction: vec2f
    psize: vec2f

    @ti.func
    def set_boundary_condition(self, pid, traction, psize):
        self.pid = pid
        self.traction += traction
        self.psize = psize

    @ti.func
    def clear_boundary_condition(self):
        self.pid = -1
        self.traction = vec2f(0., 0.)

    @ti.func
    def _compute_traction_force(self):
        psize = self.psize
        return 2. * self.traction * vec2f(psize[1], psize[0])

    @ti.func
    def _calc_psize_cp(self, dt, velocity_gradient):
        deformation_gradient_rate = DELTA2D + dt[None] * velocity_gradient
        self.psize[0] *= deformation_gradient_rate[0, 0] 
        self.psize[1] *= deformation_gradient_rate[1, 1]

    @ti.func
    def _calc_psize_r(self, dt, velocity_gradient):
        deformation_gradient_rate = DELTA2D + dt[None] * velocity_gradient
        self.psize[0] *= ti.sqrt(deformation_gradient_rate[0, 0] ** 2 + deformation_gradient_rate[1, 0] ** 2)
        self.psize[1] *= ti.sqrt(deformation_gradient_rate[0, 1] ** 2 + deformation_gradient_rate[1, 1] ** 2)


@ti.dataclass
class ParticleLoadTwoPhase2D:
    pid: int
    tractions: vec2f
    tractionf: vec2f
    psize: vec2f

    @ti.func
    def set_boundary_condition(self, pid, traction, tractionf, psize):
        self.pid = pid
        self.tractions += traction
        self.tractionf += tractionf
        self.psize = psize

    @ti.func
    def clear_boundary_condition(self):
        self.pid = -1
        self.tractions = vec2f(0., 0.)
        self.tractionf = vec2f(0., 0.)

    @ti.func
    def _compute_traction_force(self):
        psize = self.psize
        return 2. * (self.tractions - self.tractionf) * vec2f(psize[1], psize[0]), 2. * self.tractionf * vec2f(psize[1], psize[0])

    @ti.func
    def _calc_psize_cp(self, dt, velocity_gradient):
        deformation_gradient_rate = DELTA2D + dt[None] * velocity_gradient
        self.psize[0] *= deformation_gradient_rate[0, 0] 
        self.psize[1] *= deformation_gradient_rate[1, 1]

    @ti.func
    def _calc_psize_r(self, dt, velocity_gradient):
        deformation_gradient_rate = DELTA2D + dt[None] * velocity_gradient
        self.psize[0] *= ti.sqrt(deformation_gradient_rate[0, 0] ** 2 + deformation_gradient_rate[1, 0] ** 2)
        self.psize[1] *= ti.sqrt(deformation_gradient_rate[0, 1] ** 2 + deformation_gradient_rate[1, 1] ** 2)


@ti.dataclass
class ParticleLoad2DAxisy:
    pid: int
    traction: vec2f
    psize: vec2f

    @ti.func
    def set_boundary_condition(self, pid, traction, psize):
        self.pid = pid
        self.traction += traction
        self.psize = psize

    @ti.func
    def clear_boundary_condition(self):
        self.pid = -1
        self.traction = vec2f(0., 0.)

    @ti.func
    def _compute_traction_force(self):
        psize = self.psize
        return 2. * self.traction * vec2f(psize[1], psize[0])

    @ti.func
    def _calc_psize_cp(self, dt, velocity_gradient):
        deformation_gradient_rate = DELTA + dt[None] * velocity_gradient
        self.psize[0] *= deformation_gradient_rate[0, 0] 
        self.psize[1] *= deformation_gradient_rate[1, 1]

    @ti.func
    def _calc_psize_r(self, dt, velocity_gradient):
        pass
        '''deformation_gradient_rate = DELTA + dt[None] * velocity_gradient
        self.psize[0] *= ti.sqrt(deformation_gradient_rate[0, 0] ** 2 + deformation_gradient_rate[1, 0] ** 2)
        self.psize[1] *= ti.sqrt(deformation_gradient_rate[0, 1] ** 2 + deformation_gradient_rate[1, 1] ** 2)'''

class VirtualLoad:
    def __init__(self, dim) -> None:
        self.virtual_stress = None
        self.virtual_force = None
        self.auxiliary_cell = None
        self.auxiliary_node = None

    def input(self, virtual_stress, virtual_force):
        self.virtual_stress = virtual_stress
        self.virtual_force = virtual_force

    def set_lists(self, cellSum, gridSum, grid_level):
        self.auxiliary_cell = ti.field(ti.types.quant.int(bits=2, signed=False))
        bitpack = ti.BitpackedFields(max_num_bits=32)
        bitpack.place(self.auxiliary_cell)
        ti.root.dense(ti.ij, (cellSum, grid_level)).place(bitpack)

        self.auxiliary_node = ti.field(ti.types.quant.int(bits=1, signed=False))
        bitpack = ti.BitpackedFields(max_num_bits=32)
        bitpack.place(self.auxiliary_node)
        ti.root.dense(ti.ij, (gridSum, grid_level)).place(bitpack)

    def stress_check(self, dim):
        return True
    
    def force_check(self, dim):
        return True