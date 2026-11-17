from src.dem.structs.ParticleFamily import ParticleFamily

import taichi as ti
from src.utils.constants import PI
from src.utils.TypeDefinition import vec3f, vec4f


@ti.dataclass
class ClumpFamily:  # device memory: 84B
    grainIndex: int
    startIndex: int
    endIndex: int
    m: float
    equi_r: float
    a: vec3f
    angmoment: vec3f
    mass_center: vec3f
    v: vec3f
    w: vec3f
    q: vec4f
    inv_I: vec3f

    @ti.func
    def _restart(self, input_start_index, input_end_index, m, equi_r, mass_center, v, w, a, angmoment, q, inv_I):
        self.startIndex = input_start_index
        self.endIndex = input_end_index
        self.m = m
        self.equi_r = equi_r
        self.mass_center = mass_center
        self.v = v
        self.w = w
        self.q = q
        self.a = a
        self.angmoment = angmoment
        self.inv_I = inv_I

    @ti.func
    def _scale(self, factor):
        self.m *= factor * factor * factor
        self.equi_r *= factor
        self.inv_I /= factor * factor * factor * factor * factor

    @ti.func
    def _move(self, disp):
        self.mass_center += disp

    @ti.func
    def _velocity_update(self, dcurr_v):
        self.v += dcurr_v

    @ti.func
    def _angular_velocity_update(self, dcurr_w):
        self.w += dcurr_w

    @ti.func
    def _add_index(self, grain_index, start_index, end_index):
        self.grainIndex = grain_index
        self.startIndex = int(start_index)
        self.endIndex = int(end_index)

    @ti.func
    def _add_clump_kinematic(self, centor_of_mass, init_v, init_w, inv_i):
        self.mass_center = float(centor_of_mass)
        self.v = float(init_v)
        self.w = float(init_w)
        self.angmoment = self.w / inv_i

    @ti.func
    def _add_clump_attribute(self, mass, equi_r, inv_inertia, q):
        self.m = float(mass)
        self.equi_r = float(equi_r)
        self.inv_I = float(inv_inertia)
        self.q = float(q)

    @ti.func
    def _get_start_index(self):
        return self.startIndex

    @ti.func
    def _get_end_index(self):
        return self.endIndex

    @ti.func
    def _get_mass(self):
        return self.m

    @ti.func
    def _get_volume(self):
        return 4 / 3.0 * PI * self.equi_r * self.equi_r * self.equi_r

    @ti.func
    def _get_equivalent_radius(self):
        return self.equi_r

    @ti.func
    def _get_center_of_mass(self):
        return self.mass_center

    @ti.func
    def _get_velocity(self):
        return self.v

    @ti.func
    def _get_angular_velocity(self):
        return self.w

    @ti.func
    def _get_quanternion(self):
        return self.q

    @ti.func
    def _get_inverse_inertia(self):
        return self.inv_I

    def deactivate(self, particle: ParticleFamily):
        for npebble in range(self.startIndex, self.endIndex + 1):
            particle[npebble] = ti.u8(0)
