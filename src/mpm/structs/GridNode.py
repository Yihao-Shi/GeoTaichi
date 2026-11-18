import taichi as ti
from src.utils.constants import ZEROVEC2F
from src.utils.TypeDefinition import vec2f, real
from src.utils.constants import ZEROVEC3F
from src.utils.TypeDefinition import vec3f, real
from src.utils.TypeDefinition import real


@ti.dataclass
class ExtraNode:
    vol: real

    @ti.func
    def _grid_reset(self):
        self.vol = 0.0

    @ti.func
    def _update_nodal_volume(self, volume):
        self.vol += volume


@ti.dataclass
class IncompressibleNodes2D:
    m: real
    force: vec2f
    momentum: vec2f
    vbar: vec2f

    @ti.func
    def _grid_reset(self):
        self.m = 0.0
        self.momentum = ZEROVEC2F
        self.vbar = ZEROVEC2F
        self.force = ZEROVEC2F

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.momentum += momentum

    @ti.func
    def _compute_nodal_velocity(self, gravity, dt):
        self.vbar = self.momentum
        acceleration = self.force / self.m + vec2f(gravity[0], gravity[1])
        self.momentum += acceleration * dt[None]

    @ti.func
    def _compute_nodal_acceleration(self, dt):
        self.force = self.momentum - self.vbar

    @ti.func
    def velocity_constraint(self, dirs, prescribed_velocity):
        self.momentum[dirs] = prescribed_velocity
        self.force[dirs] = 0.0

    @ti.func
    def rigid_body_velocity_constraint(self, dirs):
        pass

    @ti.func
    def reflection_constraint(self, dirs, signs):
        pre_velocity = self.momentum[dirs]
        if pre_velocity * signs > 0:
            self.momentum[dirs] = 0.0
            self.force[dirs] = 0.0

    @ti.func
    def rigid_body_reflection_constraint(self, dirs, signs):
        pass


@ti.dataclass
class IncompressibleNodes3D:
    m: real
    force: vec3f
    momentum: vec3f
    vbar: vec3f

    @ti.func
    def _grid_reset(self):
        self.m = 0.0
        self.vbar = ZEROVEC3F
        self.momentum = ZEROVEC3F
        self.force = ZEROVEC3F

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.momentum += momentum

    @ti.func
    def _compute_nodal_velocity(self, gravity, dt):
        self.vbar = self.momentum
        acceleration = self.force / self.m + gravity
        self.momentum += acceleration * dt[None]

    @ti.func
    def _compute_nodal_acceleration(self, dt):
        self.force = self.momentum - self.force

    @ti.func
    def velocity_constraint(self, dirs, prescribed_velocity):
        self.momentum[dirs] = prescribed_velocity
        self.force[dirs] = 0.0

    @ti.func
    def rigid_body_velocity_constraint(self, dirs):
        pass

    @ti.func
    def reflection_constraint(self, dirs, signs):
        pre_velocity = self.momentum[dirs]
        if pre_velocity * signs > 0:
            self.momentum[dirs] = 0.0
            self.force[dirs] = 0.0

    @ti.func
    def rigid_body_reflection_constraint(self, dirs, signs):
        pass


@ti.dataclass
class ImplicitNodes:
    m: real
    inertia: vec3f
    ext_force: vec3f
    int_force: vec3f
    momentum: vec3f
    displacement: vec3f

    @ti.func
    def _grid_reset(self):
        self.m = 0.0
        self.momentum = ZEROVEC3F
        self.ext_force = ZEROVEC3F
        self.inertia = ZEROVEC3F
        self.displacement = ZEROVEC3F

    @ti.func
    def _reset_internal_force(self):
        self.int_force = ZEROVEC3F

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.momentum += momentum

    @ti.func
    def _update_nodal_acceleration(self, inertia):
        self.inertia += inertia

    @ti.func
    def _compute_nodal_velocity(self):
        self.momentum /= self.m

    @ti.func
    def _compute_nodal_acceleration(self):
        self.inertia /= self.m

    @ti.func
    def _update_external_force(self, external_force):
        self.ext_force += external_force

    @ti.func
    def _update_internal_force(self, internal_force):
        self.int_force += internal_force

    @ti.func
    def _compute_nodal_kinematic(self, damp, dt):
        pass

    @ti.func
    def _update_nodal_kinematic_newmark(self, beta, gamma, dt):
        previous_velocity = self.momentum
        previous_acceleration = self.inertia
        self.momentum = (
            gamma / beta / dt[None] * self.displacement
            - (gamma / beta - 1.0) * previous_velocity
            - 0.5 * dt[None] * (gamma / beta - 2.0) * previous_acceleration
        )
        self.inertia = (
            1.0 / beta / dt[None] / dt[None] * self.displacement
            - 1.0 / beta / dt[None] * previous_velocity
            - (0.5 / beta - 1.0) * previous_acceleration
        )

    @ti.func
    def _update_nodal_disp(self, disp):
        self.displacement += disp


@ti.dataclass
class ImplicitNodes2D:
    m: real
    inertia: vec2f
    ext_force: vec2f
    int_force: vec2f
    momentum: vec2f
    displacement: vec2f

    @ti.func
    def _grid_reset(self):
        self.m = 0.0
        self.momentum = ZEROVEC2F
        self.ext_force = ZEROVEC2F
        self.inertia = ZEROVEC2F
        self.displacement = ZEROVEC2F

    @ti.func
    def _reset_internal_force(self):
        self.int_force = ZEROVEC2F

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.momentum += momentum

    @ti.func
    def _update_nodal_acceleration(self, inertia):
        self.inertia += inertia

    @ti.func
    def _compute_nodal_velocity(self):
        self.momentum /= self.m

    @ti.func
    def _compute_nodal_acceleration(self):
        self.inertia /= self.m

    @ti.func
    def _update_external_force(self, external_force):
        self.ext_force += external_force

    @ti.func
    def _update_internal_force(self, internal_force):
        self.int_force += internal_force

    @ti.func
    def _compute_nodal_kinematic(self, damp, dt):
        pass

    @ti.func
    def _update_nodal_kinematic_newmark(self, beta, gamma, dt):
        previous_velocity = self.momentum
        previous_acceleration = self.inertia
        self.momentum = (
            gamma / beta / dt[None] * self.displacement
            - (gamma / beta - 1.0) * previous_velocity
            - 0.5 * dt[None] * (gamma / beta - 2.0) * previous_acceleration
        )
        self.inertia = (
            1.0 / beta / dt[None] / dt[None] * self.displacement
            - 1.0 / beta / dt[None] * previous_velocity
            - (0.5 / beta - 1.0) * previous_acceleration
        )

    @ti.func
    def _update_nodal_disp(self, disp):
        self.displacement += disp
