import taichi as ti
from src.utils.constants import ZEROVEC3F, ZEROVEC6F
from src.utils.TypeDefinition import real, vec3f, vec3u8, vec6f, mat3x3
from src.utils.BitFunction import zero2_one_vector
import src.utils.GlobalVariable as GlobalVariable


@ti.dataclass
class ParticleCloudTwoPhase:
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    coupling: ti.u8
    phase: ti.u8
    m: real
    ms: real
    mf: real
    vol: real
    porosity: real
    x: vec3f
    v: vec3f
    vs: vec3f
    vf: vec3f
    stress: vec6f
    pressure: real
    permeability: real
    solid_velocity_gradient: mat3x3
    fluid_velocity_gradient: mat3x3
    fix_v: vec3u8
    free_surface: ti.u8
    rad: real
    mass_density: real
    normal: vec3f
    verletDisp: vec3f
    external_force: vec3f

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
        self.phase = ti.u8(0)
        self.coupling = ti.u8(0)
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
        self.solid_velocity_gradient = float(velocity_gradient)
        self.fluid_velocity_gradient = float(velocity_gradient)
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
        self.phase = ti.u8(0)
        self.coupling = ti.u8(0)
        self.bodyID = ti.u8(body_id)
        self.materialID = ti.u8(material_id)
        self.vol = float(particle_volume)
        self.porosity = float(porosity)
        self.ms = float(particle_volume * densitys * (1.0 - porosity))
        self.mf = float(particle_volume * densityf * porosity)
        self.m = self.ms + self.mf
        self.x = float(position)
        self.v = init_v
        self.vs = init_v
        self.vf = init_v
        self.permeability = float(permeability)
        self.fix_v = fix_v
        self.rad = 0.5 * float((particle_volume) ** (1.0 / 3.0))

    @ti.func
    def _set_essential_double_point(
        self,
        body_id,
        material_id,
        phase,
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
        self.phase = ti.u8(phase)
        # In the double-point formulation only the solid material points use
        # ordinary DEM contact.  Fluid points are coupled to LSDEM by IBM.
        self.coupling = ti.u8(phase == 1)
        self.bodyID = ti.u8(body_id)
        self.materialID = ti.u8(material_id)
        self.vol = float(particle_volume)
        self.porosity = float(porosity)
        self.ms = 0.0
        self.mf = 0.0
        if phase == 1:
            self.ms = float(particle_volume * densitys * (1.0 - porosity))
        elif phase == 2:
            self.mf = float(particle_volume * densityf * porosity)
        self.m = self.ms + self.mf
        self.x = float(position)
        self.v = init_v
        self.vs = init_v
        self.vf = init_v
        self.permeability = float(permeability)
        self.fix_v = fix_v
        self.rad = 0.5 * float((particle_volume) ** (1.0 / 3.0))

    @ti.func
    def _set_essential_rigid(self, body_id, material_id, density, particle_volume, position, init_v, fix_v, x_offset):
        self.active = ti.u8(1)
        self.phase = ti.u8(1)
        self.bodyID = ti.u8(body_id)
        self.materialID = ti.u8(material_id)
        self.vol = float(particle_volume)
        self.m = float(particle_volume * density)
        self.ms = float(particle_volume * density)
        self.mf = 0.0
        self.x = float(position)
        self.v = init_v
        self.vs = init_v
        self.vf = init_v
        self.fix_v = fix_v

    @ti.func
    def _reset_mass_density(self):
        self.mass_density = 0.0

    @ti.func
    def _reset_contact_force(self):
        self.external_force = ZEROVEC3F

    @ti.func
    def _update_contact_interaction(self, cforce, ctorque):
        self.external_force += cforce

    @ti.func
    def _get_position(self):
        return self.x

    @ti.func
    def _get_contact_radius(self, contact_position):
        return self.rad

    @ti.func
    def _get_radius(self):
        return self.rad

    @ti.func
    def _get_mass(self):
        return self.ms

    @ti.func
    def _get_velocity(self):
        return self.vs

    @ti.func
    def _get_angular_velocity(self):
        return ZEROVEC3F

    @ti.func
    def _add_gravity_field(self, k0, gamma):
        self.stress[1] += float(gamma)
        self.stress[0] += float(k0 * gamma)
        self.stress[2] += float(k0 * gamma)

    @ti.func
    def _update_stress(self, stress):
        self.stress += stress

    @ti.func
    def _compute_external_force(self, gravity):
        g = vec3f(gravity[0], gravity[1], gravity[2])
        return self.m * g, self.mf * g

    @ti.func
    def _compute_drag_force(self):
        return -self.porosity * self.porosity * 9.8 * 1000.0 * self.vol * (self.vf - self.vs) / self.permeability

    @ti.func
    def _compute_drag_force_semi(self):
        return self.porosity * self.porosity * 9.8 * 1000.0 * self.vol / self.permeability

    @ti.func
    def _compute_internal_force(self):
        stressf = ZEROVEC6F
        stressf[0] = self.pressure
        stressf[1] = self.pressure
        stressf[2] = self.pressure
        return -self.vol * (self.stress - stressf), self.vol * self.porosity * stressf

    @ti.func
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
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE
        if ti.static(GlobalVariable.MPMZPBC):
            self.x[2] -= ti.floor(self.x[2] / GlobalVariable.MPMZSIZE) * GlobalVariable.MPMZSIZE

    @ti.func
    def _update_particle_state_u_p(self, dt, alpha, v_pic, v_flip):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = zero2_one_vector(flag1)
        self.v = (alpha * v_pic + (1 - alpha) * (v_flip + v0)) * flag2 + v0 * flag1
        self.x += v_pic * dt[None] * flag2 + v0 * dt[None] * flag1

    @ti.func
    def _update_particle_pressure(self, beta, dp):
        p0 = self.pressure
        self.pressure = p0 * beta + dp

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]

    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.solid_velocity_gradient @ (self.x - xg)
