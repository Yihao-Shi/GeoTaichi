import taichi as ti
from src.utils.TypeDefinition import vec2f, vec2u8, mat2x2
import src.utils.GlobalVariable as GlobalVariable
from src.utils.TypeDefinition import vec3f, vec3u8, mat3x3


@ti.dataclass
class StaggeredParticle:
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    x: vec3f
    v: vec3f
    pressure: float
    velocity_gradient: mat3x3
    fix_v: vec3u8

    @ti.func
    def _restart(
        self, body_id, material_id, active, mass, position, velocity, volume, pressure, velocity_gradient, fix_v
    ):
        self.bodyID = ti.u8(body_id)
        self.materialID = ti.u8(material_id)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.pressure = float(pressure)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particle_id, body_id, material_id, density, particle_volume, position, init_v, fix_v):
        self.particleID = particle_id
        self.active = ti.u8(1)
        self.bodyID = ti.u8(body_id)
        self.materialID = ti.u8(material_id)
        self.vol = float(particle_volume)
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        self.pressure += 1.0 / 3.0 * float(gamma[0, 0] + gamma[1, 1] + gamma[2, 2])

    @ti.func
    def _compute_internal_force(self):
        pass

    @ti.func
    def _update_particle_state(self, dt, alpha, v_pic, input_v_flip):
        v0 = self.v
        self.v = alpha * v_pic + (1 - alpha) * (input_v_flip + v0)
        self.x += v_pic * dt[None]
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[2] -= ti.floor(self.x[2] / GlobalVariable.MPMZSIZE) * GlobalVariable.MPMZSIZE

    @ti.func
    def _update_stress(self, stress):
        self.pressure += 1.0 / 3.0 * (stress[0] + stress[1] + stress[2])

    @ti.func
    def _get_mean_stress(self):
        return self.pressure

    @ti.func
    def _update_rigid_body(self, dt):
        pass


@ti.dataclass
class StaggeredParticle2D:
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    x: vec2f
    v: vec2f
    pressure: float
    velocity_gradient: mat2x2
    fix_v: vec2u8
    unfix_v: vec2u8

    @ti.func
    def _restart(
        self, body_id, material_id, active, mass, position, velocity, volume, pressure, velocity_gradient, fix_v
    ):
        self.bodyID = ti.u8(body_id)
        self.materialID = ti.u8(material_id)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.pressure = float(pressure)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particle_id, body_id, material_id, density, particle_volume, position, init_v, fix_v):
        self.particleID = particle_id
        self.active = ti.u8(1)
        self.bodyID = ti.u8(body_id)
        self.materialID = ti.u8(material_id)
        self.vol = float(particle_volume)
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        self.pressure += 1.0 / 3.0 * float(gamma[0, 0] + gamma[1, 1] + gamma[2, 2])

    @ti.func
    def _compute_internal_force(self):
        pass

    @ti.func
    def _update_particle_state(self, dt, alpha, v_pic, input_v_flip):
        v0 = self.v
        self.v = alpha * v_pic + (1 - alpha) * (input_v_flip + v0)
        self.x += v_pic * dt[None]
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE

    @ti.func
    def _update_stress(self, stress):
        self.pressure += 1.0 / 3.0 * (stress[0] + stress[1] + stress[2])

    @ti.func
    def _get_mean_stress(self):
        return self.pressure

    @ti.func
    def _update_rigid_body(self, dt):
        pass
