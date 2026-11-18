import taichi as ti
from src.utils.constants import ZEROVEC6F
from src.utils.TypeDefinition import real, vec2f, vec2u8, vec6f, mat3x3
from src.utils.BitFunction import zero2_one_vector


@ti.dataclass
class ParticleCloudTwoPhase2DAxisy:
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: real
    ms: real  # soild phase
    mf: real  # fluid phase
    vol: real
    area: real
    porosity: real
    x: vec2f
    v: vec2f
    vs: vec2f
    vf: vec2f
    stress: vec6f
    pressure: real
    permeability: real
    velocity_gradient: mat3x3
    velocity_gradientf: mat3x3
    fix_v: vec2u8
    # free surface
    free_surface: ti.u8
    rad: real
    mass_density: real
    normal: vec2f
    verletDisp: vec2f

    @ti.func
    def _restart(
        self,
        body_id,
        material_id,
        active,
        mass,
        mass_s,
        mass_f,
        position,
        velocity,
        velocity_s,
        velocity_f,
        volume,
        porosity,
        stress,
        pressure,
        permeability,
        velocity_gradient,
        fix_v,
        free_surface,
        normal,
    ):
        self.bodyID = ti.u8(body_id)
        self.materialID = ti.u8(material_id)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.ms = float(mass_s)
        self.mf = float(mass_f)
        self.x = float(position)
        self.v = float(velocity)
        self.vs = float(velocity_s)
        self.vf = float(velocity_f)
        self.vol = float(volume)
        self.porosity = float(porosity)
        self.stress = float(stress)
        self.pressure = float(pressure)
        self.permeability = float(permeability)
        self.velocity_gradient = float(velocity_gradient)
        self.velocity_gradientf = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)
        self.free_surface = ti.u8(free_surface)
        self.normal = float(normal)

    @ti.func
    def _set_essential(
        self,
        body_id,
        material_id,
        densitys,
        densityf,
        porosity,
        particle_volume,
        position,
        init_v,
        fix_v,
        permeability,
        x_offset,
    ):
        self.active = ti.u8(1)
        self.bodyID = ti.u8(body_id)
        self.materialID = ti.u8(material_id)
        self.porosity = float(porosity)
        self.area = float(particle_volume)
        self.vol = float((position[0] - x_offset) * particle_volume)
        self.ms = float((position[0] - x_offset) * particle_volume * densitys * (1.0 - porosity))
        self.mf = float((position[0] - x_offset) * particle_volume * densityf * porosity)
        self.m = self.ms + self.mf
        self.x = float(position)
        self.v = init_v
        self.vs = init_v
        self.vf = init_v
        self.permeability = float(permeability)
        self.fix_v = fix_v
        self.rad = 0.5 * float((particle_volume) ** (1.0 / 2.0))

    @ti.func
    def _set_essential_rigid(self, body_id, material_id, density, particle_volume, position, init_v, fix_v, x_offset):
        self.active = ti.u8(1)
        self.bodyID = ti.u8(body_id)
        self.materialID = ti.u8(material_id)
        self.vol = float((position[0] - x_offset) * particle_volume)
        self.m = float((position[0] - x_offset) * particle_volume * density)
        self.ms = float((position[0] - x_offset) * particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.vs = init_v
        self.fix_v = fix_v

    @ti.func
    def _reset_mass_density(self):
        self.mass_density = 0.0

    @ti.func
    def _add_gravity_field(self, k0, gamma):
        self.stress[1] += float(gamma)
        self.stress[0] += float(k0 * gamma)
        self.stress[2] += float(k0 * gamma)

    @ti.func
    def _update_stress(self, stress):
        self.stress += stress

    @ti.func  # total, fluid
    def _compute_external_force(self, gravity):
        return self.m * vec2f(gravity[0], gravity[1]), self.mf * vec2f(gravity[0], gravity[1])

    @ti.func
    def _compute_drag_force(self):
        return -self.porosity * self.porosity * 9.8 * 1000.0 * self.vol * (self.vf - self.vs) / self.permeability

    @ti.func
    def _compute_drag_force_semi(self):
        return self.porosity * self.porosity * 9.8 * 1000.0 * self.vol / self.permeability

    @ti.func  # total internal force, fluid internal force
    def _compute_internal_force(self):
        stressf = ZEROVEC6F
        stressf[0] = self.pressure
        stressf[1] = self.pressure
        stressf[2] = self.pressure
        return -self.vol * (self.stress - stressf), self.vol * self.porosity * stressf

    @ti.func  # total internal force, fluid internal force
    def _compute_internal_force_semi(self, beta):
        stressf = ZEROVEC6F
        stressf[0] = self.pressure * beta
        stressf[1] = self.pressure * beta
        stressf[2] = self.pressure * beta
        return -self.vol * (self.stress - stressf), self.vol * self.porosity * stressf

    @ti.func
    def _update_particle_state(self, dt, alpha, v_pic, input_v_flip, v_pi_cs, v_fli_ps, v_pi_cf, v_fli_pf):
        v0s = self.vs
        v0f = self.vf
        flag1 = int(self.fix_v)
        flag2 = zero2_one_vector(flag1)
        self.vs = (alpha * v_pi_cs + (1 - alpha) * (v_fli_ps + v0s)) * flag2 + v0s * flag1
        self.vf = (alpha * v_pi_cf + (1 - alpha) * (v_fli_pf + v0f)) * flag2 + v0f * flag1
        self.v = self.vs
        self.x += v_pi_cs * dt[None]

    @ti.func
    def _update_particle_pressure(self, beta, dp):
        p0 = self.pressure
        self.pressure = p0 * beta + dp
        self.pressure = ti.max(self.pressure, -15e3)  # prevent too large negative pressure

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]

    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)
