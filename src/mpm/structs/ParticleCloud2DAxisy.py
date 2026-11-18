from src.utils.MatrixFunction import _stress_increment_mat3
from src.utils.MatrixFunction import _stress_increment_voigt

import taichi as ti
from src.utils.TypeDefinition import real, vec2f, vec2u8, vec6f, mat3x3
from src.utils.BitFunction import zero2_one_vector
import src.utils.GlobalVariable as GlobalVariable


@ti.dataclass
class ParticleCloud2DAxisy:  # memory usage: 108B
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: real
    vol: real
    x: vec2f
    v: vec2f
    stress: vec6f
    velocity_gradient: mat3x3
    fix_v: vec2u8

    @ti.func
    def _restart(
        self,
        particle_id,
        body_id,
        material_id,
        active,
        mass,
        position,
        velocity,
        volume,
        stress,
        velocity_gradient,
        fix_v,
    ):
        self.particleID = int(particle_id)
        self.bodyID = ti.u8(body_id)
        self.materialID = ti.u8(material_id)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.stress = float(stress)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particle_id, body_id, material_id, density, particle_volume, position, init_v, fix_v):
        self.particleID = particle_id
        self.active = ti.u8(1)
        self.bodyID = ti.u8(body_id)
        self.materialID = ti.u8(material_id)
        self.vol = float(position[0] * particle_volume)
        self.m = float(position[0] * particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        if ti.static(self.stress.n == 6):
            self.stress += _stress_increment_voigt(gamma)
        elif ti.static(self.stress.n == 3):
            self.stress += _stress_increment_mat3(gamma)

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * vec2f(gravity[0], gravity[1])

    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress

    @ti.func
    def _update_particle_state(self, dt, alpha, v_pic, input_v_flip):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = zero2_one_vector(flag1)
        self.v = (alpha * v_pic + (1 - alpha) * (input_v_flip * dt[None] + v0)) * flag2 + v0 * flag1
        self.x += v_pic * dt[None] * flag2 + v0 * dt[None] * flag1
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE

    @ti.func
    def _update_stress(self, stress):
        if ti.static(self.stress.n == 6):
            self.stress += _stress_increment_voigt(stress)
        elif ti.static(self.stress.n == 3):
            self.stress += _stress_increment_mat3(stress)

    @ti.func
    def _get_mean_stress(self):
        pressure = 0.0
        if ti.static(self.stress.n == 6):
            pressure = 1.0 / 3.0 * (self.stress[0] + self.stress[1] + self.stress[2])
        elif ti.static(self.stress.n == 3):
            pressure = 1.0 / 3.0 * (self.stress[0, 0] + self.stress[1, 1] + self.stress[2, 2])
        return pressure

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]

    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)
