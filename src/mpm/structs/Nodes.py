from src.mpm.structs.Utilities import project_coulomb_wall

import taichi as ti
from src.utils.constants import ZEROVEC3F
from src.utils.ScalarFunction import sgn
from src.utils.TypeDefinition import vec3f, real


@ti.dataclass
class Nodes:
    m: real
    force: vec3f
    momentum: vec3f

    @ti.func
    def _tlgrid_reset(self):
        self.momentum = ZEROVEC3F
        self.force = ZEROVEC3F

    @ti.func
    def _grid_reset(self):
        self.m = 0.0
        self.momentum = ZEROVEC3F
        self.force = ZEROVEC3F

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.momentum += momentum

    @ti.func
    def _compute_nodal_velocity(self):
        self.momentum /= self.m

    @ti.func
    def _update_nodal_force(self, force):
        self.force += force

    @ti.func
    def _update_external_force(self, external_force):
        self.force += external_force

    @ti.func
    def _update_internal_force(self, internal_force):
        self.force += internal_force

    @ti.func
    def _compute_nodal_kinematic(self, damp, dt):
        unbalanced_force = self.force
        velocity = self.momentum
        for d in ti.static(range(3)):
            if velocity[d] * unbalanced_force[d] > 0.0:
                unbalanced_force[d] -= damp * ti.abs(unbalanced_force[d]) * sgn(velocity[d])
        acceleration = unbalanced_force / self.m
        self.momentum += acceleration * dt[None]
        self.force = acceleration

    @ti.func
    def _update_nodal_kinematic(self):
        self.force /= self.m
        self.momentum /= self.m

    @ti.func
    def contact_velocity_constraint(self, dirs):
        pass

    @ti.func
    def contact_reflection_constraint(self, dirs, signs):
        pass

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

    @ti.func
    def friction_constraint(self, mu, dirs_n, signs, dt):
        self.momentum, self.force = project_coulomb_wall(self.momentum, self.force, mu, dirs_n, signs, dt[None])

    @ti.func
    def rigid_friction_constraint(self, dirs, signs):
        pass
