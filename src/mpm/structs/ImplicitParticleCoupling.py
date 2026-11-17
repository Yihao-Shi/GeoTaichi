from src.utils.MatrixFunction import _stress_increment_mat3
from src.utils.MatrixFunction import _stress_increment_pressure
from src.utils.MatrixFunction import _stress_increment_voigt

import taichi as ti
from src.utils.constants import ZEROVEC3F
from src.utils.TypeDefinition import real, vec3f, vec3u8, vec6f, mat3x3
from src.utils.BitFunction import zero2_one_vector
import src.utils.GlobalVariable as GlobalVariable


@ti.dataclass
class ImplicitParticleCoupling:

    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    coupling: ti.u8
    m: real
    vol: real
    rad: real
    x: vec3f
    a: vec3f
    verletDisp: vec3f
    v: vec3f
    external_force: vec3f
    stress: vec6f
    pressure: real
    velocity_gradient: mat3x3
    fix_v: vec3u8

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
        self.pressure = self._get_mean_stress()
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particle_id, body_id, material_id, density, particle_volume, position, init_v, fix_v):
        self.particleID = particle_id
        self.active = ti.u8(1)
        self.coupling = ti.u8(1)
        self.bodyID = ti.u8(body_id)
        self.materialID = ti.u8(material_id)
        self.vol = float(particle_volume)
        self.rad = 0.5 * float((particle_volume) ** (1.0 / 3.0))
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.pressure = 0.0
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        if ti.static(self.stress.n == 6):
            self.stress += _stress_increment_voigt(gamma)
        elif ti.static(self.stress.n == 3):
            self.stress += _stress_increment_mat3(gamma)
        self.pressure += _stress_increment_pressure(gamma)

    @ti.func
    def _update_stress(self, stress):
        if ti.static(self.stress.n == 6):
            self.stress += _stress_increment_voigt(stress)
        elif ti.static(self.stress.n == 3):
            self.stress += _stress_increment_mat3(stress)
        self.pressure += _stress_increment_pressure(stress)

    @ti.func
    def _get_mean_stress(self):
        pressure = 0.0
        if ti.static(self.stress.n == 6):
            pressure = 1.0 / 3.0 * (self.stress[0] + self.stress[1] + self.stress[2])
        elif ti.static(self.stress.n == 3):
            pressure = 1.0 / 3.0 * (self.stress[0, 0] + self.stress[1, 1] + self.stress[2, 2])
        return pressure

    @ti.func
    def _reset_contact_force(self):
        self.external_force = ZEROVEC3F

    @ti.func
    def _reset_mass_density(self):
        self.mass_density = 0.0

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * gravity + self.external_force

    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress

    @ti.func
    def _update_particle_state(self, dt, alpha, v_pic, acc, disp):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = zero2_one_vector(flag1)
        v_flip = 0.5 * (acc + self.a) * dt[None]
        self.a = acc
        self.v = (alpha * v_pic + (1 - alpha) * (v_flip + v0)) * flag2 + v0 * flag1
        deltax = disp * flag2 + v0 * dt[None] * flag1
        self.x += deltax
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE
        if ti.static(GlobalVariable.MPMZPBC):
            self.x[2] -= ti.floor(self.x[2] / GlobalVariable.MPMZSIZE) * GlobalVariable.MPMZSIZE
        if int(self.coupling) == 1:
            self.verletDisp += deltax

    @ti.func
    def _update_incompressible_particle_state(self, dt, alpha, v_pic, v_flip, transport_velocity):
        v0 = self.v
        fixed = self.fix_v.cast(float)
        free = 1.0 - fixed
        self.v = (alpha * v_pic + (1 - alpha) * (v_flip + v0)) * free + v0 * fixed
        deltax = (transport_velocity * free + v0 * fixed) * dt[None]
        self.x += deltax
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE
        if ti.static(GlobalVariable.MPMZPBC):
            self.x[2] -= ti.floor(self.x[2] / GlobalVariable.MPMZSIZE) * GlobalVariable.MPMZSIZE
        if int(self.coupling) == 1:
            self.verletDisp += deltax

    @ti.func
    def _update_rigid_body(self, dt):
        deltax = self.v * dt[None]
        self.x += deltax
        if int(self.coupling) == 1:
            self.verletDisp += deltax

    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)

    @ti.func
    def _update_contact_interaction(self, cforce, ctorque):
        self.external_force += cforce

    @ti.func
    def _get_position(self):
        return self.x

    @ti.func
    def _get_contact_radius(self, gapn):
        return self.rad + gapn

    @ti.func
    def _get_radius(self):
        return self.rad

    @ti.func
    def _get_mass(self):
        return self.m

    @ti.func
    def _get_velocity(self):
        return self.v

    @ti.func
    def _get_angular_velocity(self):
        return ZEROVEC3F
