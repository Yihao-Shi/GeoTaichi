import taichi as ti
from src.utils.constants import PI, ZEROVEC3F
from src.utils.TypeDefinition import vec3f


@ti.dataclass
class ParticleFamily:  # device memory: 84B
    active: ti.u8
    multisphereIndex: int
    materialID: ti.u8
    groupID: ti.u8
    rad: float
    m: float
    x: vec3f
    v: vec3f
    w: vec3f
    contact_force: vec3f
    contact_torque: vec3f
    verletDisp: vec3f

    @ti.func
    def _restart(self, active, multisphere_index, group_id, material_id, m, rad, x, v, w):
        self.active = ti.u8(active)
        self.multisphereIndex = int(multisphere_index)
        self.groupID = ti.u8(group_id)
        self.materialID = ti.u8(material_id)
        self.m = float(m)
        self.rad = float(rad)
        self.x = float(x)
        self.v = float(v)
        self.w = float(w)

    @ti.func
    def _scale(self, factor):
        self.m *= factor * factor * factor
        self.rad *= factor

    @ti.func
    def _pebble_scale(self, factor, centor_of_mass):
        self.m *= factor * factor * factor
        self.rad *= factor
        self.x = self.x * factor + (1 - factor) * centor_of_mass

    @ti.func
    def _move(self, disp, true_disp):
        self.x += disp
        self.verletDisp += true_disp

    @ti.func
    def _renew_verlet(self):
        self.verletDisp = ZEROVEC3F

    @ti.func
    def _velocity_update(self, dcurr_v):
        self.v += dcurr_v

    @ti.func
    def _pebble_velocity_update(self, dcurr_v):
        self.v += dcurr_v

    @ti.func
    def _angular_velocity_update(self, dcurr_w):
        self.w += dcurr_w

    @ti.func
    def _pebble_angular_velocity_update(self, dcurr_w):
        self.w += dcurr_w

    @ti.func
    def calm(self):
        self.v = ZEROVEC3F
        self.w = ZEROVEC3F

    @ti.func
    def _add_index(self, multisphere_index):
        self.multisphereIndex = int(multisphere_index)

    @ti.func
    def _update_contact_interaction(self, cforce, ctorque):
        self.contact_force += cforce
        self.contact_torque += ctorque

    @ti.func
    def _add_particle_proporities(self, material_id, group_id, radius, mass):
        self.active = ti.u8(1)
        self.materialID = ti.u8(material_id)
        self.groupID = ti.u8(group_id)
        self.rad = float(radius)
        self.m = float(mass)

    @ti.func
    def _add_particle_kinematics(self, x, v, w):
        self.x = x
        self.v = v
        self.w = w

    @ti.func
    def _get_multisphere_index1(self):
        return self.multisphereIndex

    @ti.func
    def _get_multisphere_index2(self):
        return self.multisphereIndex

    @ti.func
    def _get_material(self):
        return self.materialID

    @ti.func
    def _get_group(self):
        return self.groupID

    @ti.func
    def _get_volume(self):
        return 4.0 / 3.0 * PI * self.rad * self.rad * self.rad

    @ti.func
    def _get_radius(self):
        return self.rad

    @ti.func
    def _get_mass(self):
        return self.m

    @ti.func
    def _get_position(self):
        return self.x

    @ti.func
    def _get_velocity(self):
        return self.v

    @ti.func
    def _get_angular_velocity(self):
        return self.w

    @ti.func
    def _get_verlet_displacement(self):
        return self.verletDisp

    @ti.func
    def _get_contact_radius(self, contact_point):
        return (contact_point - self.x).norm()
