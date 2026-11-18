import taichi as ti
from src.utils.constants import PI, ZEROVEC3F
from src.utils.TypeDefinition import vec3f, vec3u8, vec4f
from src.utils.BitFunction import zero2_one_vector


@ti.dataclass
class ImplicitSurfaceParticle:
    groupID: ti.u8
    materialID: ti.u8
    templateID: ti.u8
    scale: float
    m: float
    equi_r: float
    mass_center: vec3f
    a: vec3f
    v: vec3f
    w: vec3f
    angmoment: vec3f
    q: vec4f
    inv_I: vec3f
    contact_force: vec3f
    contact_torque: vec3f
    is_fix: vec3u8

    @ti.func
    def _restart(
        self,
        group_id,
        material_id,
        template_id,
        scale,
        mass,
        equiv_rad,
        mass_center,
        a,
        angmoment,
        v,
        w,
        q,
        inv_I,
        is_fix,
    ):
        self.groupID = ti.u8(group_id)
        self.materialID = ti.u8(material_id)
        self.templateID = int(template_id)
        self.scale = float(scale)
        self.m = float(mass)
        self.equi_r = float(equiv_rad)
        self.mass_center = float(mass_center)
        self.a = float(a)
        self.v = float(v)
        self.w = float(w)
        self.angmoment = float(angmoment)
        self.q = float(q)
        self.inv_I = float(inv_I)
        self.is_fix = ti.cast(is_fix, ti.u8)

    @ti.func
    def _add_body_attribute(self, scale_factor, centor_of_mass, mass, equiv_rad, inv_inertia, q):
        self.scale = float(scale_factor)
        self.mass_center = float(centor_of_mass)
        self.m = float(mass)
        self.equi_r = float(equiv_rad)
        self.inv_I = float(inv_inertia)
        self.q = float(q)

    @ti.func
    def _add_body_properties(self, material_id, group_id, template_id, density):
        self.materialID = ti.u8(material_id)
        self.groupID = ti.u8(group_id)
        self.templateID = ti.u8(template_id)
        self.m *= density
        self.inv_I /= density

    @ti.func
    def _add_body_kinematic(self, init_v, init_w, is_fix):
        self.v = float(init_v)
        self.w = float(init_w)
        self.angmoment = self.w / self.inv_I
        self.is_fix = ti.cast(zero2_one_vector(is_fix), ti.u8)

    @ti.func
    def _scale(self, factor):
        self.m *= float(factor * factor * factor)
        self.inv_I /= float(factor * factor * factor * factor * factor)
        self.scale *= factor

    @ti.func
    def _velocity_update(self, dcurr_v):
        self.v += dcurr_v

    @ti.func
    def _angular_velocity_update(self, dcurr_w):
        self.w += dcurr_w

    @ti.func
    def calm(self):
        self.v = ZEROVEC3F
        self.w = ZEROVEC3F

    @ti.func
    def _update_contact_interaction(self, cforce, ctorque):
        self.contact_force += cforce
        self.contact_torque += ctorque

    @ti.func
    def _add_particle_kinematics(self, x, v, w):
        self.mass_center = x
        self.v = v
        self.w = w

    @ti.func
    def _get_material(self):
        return self.materialID

    @ti.func
    def _get_group(self):
        return self.groupID

    @ti.func
    def _get_template(self):
        return self.templateID

    @ti.func
    def _get_scale(self):
        return self.scale

    @ti.func
    def _get_mass(self):
        return self.m

    @ti.func
    def _get_position(self):
        return self.mass_center

    @ti.func
    def _get_velocity(self):
        return self.v

    @ti.func
    def _get_angular_velocity(self):
        return self.w

    @ti.func
    def _get_volume(self):
        return 4.0 / 3.0 * PI * self.equi_r * self.equi_r * self.equi_r

    @ti.func
    def _get_contact_radius(self, contact_point):
        return (contact_point - self.mass_center).norm()

    @ti.func
    def _get_radius(self):
        return self.equi_r

    @ti.func
    def _get_margin(self):
        return 0.05 * self.equi_r
