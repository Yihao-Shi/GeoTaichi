import taichi as ti

from src.utils.TypeDefination import vec2f, vec3f, vec2u8, vec3u8
from src.utils.BitFunction import Zero2OneVector


@ti.dataclass
class StaggeredPartilce:
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    x: vec3f
    v: vec3f
    pressure: float
    xvelocity_gradient: vec3f
    yvelocity_gradient: vec3f
    zvelocity_gradient: vec3f
    fix_v: vec3u8
    unfix_v: vec3u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, pressure, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.pressure = float(pressure)
        self.xvelocity_gradient = vec3f(velocity_gradient[0, 0], velocity_gradient[1, 0], velocity_gradient[2, 0])
        self.yvelocity_gradient = vec3f(velocity_gradient[0, 1], velocity_gradient[1, 1], velocity_gradient[2, 1])
        self.zvelocity_gradient = vec3f(velocity_gradient[0, 2], velocity_gradient[1, 2], velocity_gradient[2, 2])
        self.fix_v = ti.cast(fix_v, ti.u8)
        self.unfix_v = ti.cast(Zero2OneVector(fix_v), ti.u8)

    @ti.func
    def _set_essential(self, bodyID, materialID, density, particle_volume, position, init_v, fix_v):
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v
        self.unfix_v = Zero2OneVector(fix_v)

    @ti.func
    def _add_gravity_field(self, k0, gamma):
        self.pressure -= float(gamma)

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * gravity 
    
    @ti.func
    def _update_stress(self, stress):
        self.pressure += 1./3. * (stress[0] + stress[1] + stress[2])
    
    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, vFLIP):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = int(self.unfix_v)
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP * dt[None] + v0)) * flag2 + v0 * flag1
        self.x += vPIC * dt[None] * flag2 + v0 * dt[None] * flag1

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]


@ti.dataclass
class StaggeredPartilce2D:
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    x: vec2f
    v: vec2f
    pressure: float
    xvelocity_gradient: vec2f
    yvelocity_gradient: vec2f
    fix_v: vec2u8
    unfix_v: vec2u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, pressure, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.pressure = float(pressure)
        self.xvelocity_gradient = vec2f(velocity_gradient[0, 0], velocity_gradient[1, 0])
        self.yvelocity_gradient = vec2f(velocity_gradient[0, 1], velocity_gradient[1, 1])
        self.fix_v = ti.cast(fix_v, ti.u8)
        self.unfix_v = ti.cast(Zero2OneVector(fix_v), ti.u8)

    @ti.func
    def _set_essential(self, bodyID, materialID, density, particle_volume, position, init_v, fix_v):
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v
        self.unfix_v = Zero2OneVector(fix_v)

    @ti.func
    def _add_gravity_field(self, k0, gamma):
        self.pressure -= float(gamma)

    @ti.func
    def _update_stress(self, stress):
        self.pressure += 1./3. * (stress[0] + stress[1] + stress[2])

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * gravity 
    
    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, vFLIP):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = int(self.unfix_v)
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP * dt[None] + v0)) * flag2 + v0 * flag1
        self.x += vPIC * dt[None] * flag2 + v0 * dt[None] * flag1

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]
    
