import taichi as ti
from src.utils.constants import ZEROVEC3F
from src.utils.ScalarFunction import sgn
from src.utils.TypeDefinition import vec3f, real


@ti.dataclass
class NodeTwoPhase:
    dof: int
    m: real
    ms: real
    mf: real
    drag: vec3f
    force: vec3f
    forces: vec3f
    forcef: vec3f
    momentum: vec3f
    momentums: vec3f
    momentumf: vec3f
    pressure: real
    dpressure: real
    dragval: real
    porosity: real
    weight: real
    contact_force_s: vec3f
    contact_force_f: vec3f
    extra_stabilize: vec3f
    grad_domain: vec3f

    @ti.func
    def _tlgrid_reset(self):
        self.momentum = ZEROVEC3F
        self.momentums = ZEROVEC3F
        self.momentumf = ZEROVEC3F
        self.drag = ZEROVEC3F
        self.force = ZEROVEC3F
        self.forces = ZEROVEC3F
        self.forcef = ZEROVEC3F
        self.contact_force_s = ZEROVEC3F
        self.contact_force_f = ZEROVEC3F
        self.grad_domain = ZEROVEC3F

    @ti.func
    def _grid_reset(self):
        self.m = 0.0
        self.ms = 0.0
        self.mf = 0.0
        self.drag = ZEROVEC3F
        self.momentum = ZEROVEC3F
        self.momentums = ZEROVEC3F
        self.momentumf = ZEROVEC3F
        self.force = ZEROVEC3F
        self.forces = ZEROVEC3F
        self.forcef = ZEROVEC3F
        self.dof = -1
        self.pressure = 0.0
        self.dpressure = 0.0
        self.dragval = 0.0
        self.porosity = 0.0
        self.weight = 0.0
        self.contact_force_s = ZEROVEC3F
        self.contact_force_f = ZEROVEC3F
        self.grad_domain = ZEROVEC3F

    @ti.func
    def _set_dofs(self, rowth):
        self.dof = int(rowth)

    @ti.func
    def _update_nodal_mass(self, m, ms, mf):
        self.m += m
        self.ms += ms
        self.mf += mf

    @ti.func
    def _update_nodal_momentum(self, momentum, momentums, momentumf):
        self.momentum += momentum
        self.momentums += momentums
        self.momentumf += momentumf

    @ti.func
    def _update_nodal_pressure(self, pressure):
        self.pressure += pressure

    @ti.func
    def _compute_nodal_velocity(self, cutoff):
        if self.m > cutoff:
            self.momentum /= self.m
        if self.ms > cutoff:
            self.momentums /= self.ms
            self.pressure /= self.ms
        if self.mf > cutoff:
            self.momentumf /= self.mf

    @ti.func
    def _update_nodal_force(self, force, forcef):
        self.force += force
        self.forcef += forcef

    @ti.func
    def _update_nodal_dragforce(self, force):
        self.drag += force

    @ti.func
    def _update_nodal_dragval(self, value):
        self.dragval += value

    @ti.func
    def _update_external_force(self, external_force, external_forces, external_forcef):
        self.force += external_force
        self.forces += external_forces
        self.forcef += external_forcef

    @ti.func
    def _update_internal_force(self, internal_force, internal_forces, internal_forcef):
        self.force += internal_force
        self.forces += internal_forces
        self.forcef += internal_forcef

    @ti.func
    def _compute_nodal_kinematic(self, damp, dt):
        unbalanced_force = self.force / self.m
        self.force = unbalanced_force
        self.momentum += unbalanced_force * dt[None]

    @ti.func
    def _compute_nodal_kinematic_fluid(self, damp, dt):
        unbalanced_force = self.forcef / self.mf
        velocity = self.momentumf
        for d in ti.static(range(3)):
            if velocity[d] * unbalanced_force[d] > 0.0:
                unbalanced_force[d] -= damp * ti.abs(unbalanced_force[d]) * sgn(velocity[d])
        self.momentumf += unbalanced_force * dt[None]
        self.forcef = unbalanced_force

    @ti.func
    def _compute_nodal_kinematic_solid(self, damp, dt):
        unbalanced_force = (self.force - self.forcef * self.mf) / self.ms
        forces = unbalanced_force
        self.momentums += forces * dt[None]
        self.forces = forces

    @ti.func
    def _compute_nodal_kinematic_semi(self, damp, dt, forces, forcef):
        self.forces = forces
        self.forcef = forcef
        self.momentums += forces * dt[None]
        self.momentumf += forcef * dt[None]

    @ti.func
    def _correct_nodal_kinematic_semi(self, dt, forces, forcef):
        self.momentums += forces * dt[None]
        self.momentumf += forcef * dt[None]

    @ti.func
    def _correct_nodal_kinematic_u_p(self, dt, force):
        self.momentum += force * dt[None]

    @ti.func
    def _update_nodal_kinematic(self):
        self.force /= self.m
        self.forces /= self.ms
        self.forcef /= self.mf
        self.momentum /= self.m
        self.momentums /= self.ms
        self.momentumf /= self.mf

    @ti.func
    def _update_nodal_dpressure(self, dpressure):
        self.dpressure = dpressure

    @ti.func
    def velocity_constraint(self, dirs, prescribed_velocity):
        self.momentum[dirs] = prescribed_velocity
        self.momentums[dirs] = prescribed_velocity
        self.momentumf[dirs] = prescribed_velocity
        self.force[dirs] = 0.0
        self.forces[dirs] = 0.0
        self.forcef[dirs] = 0.0
        self.extra_stabilize[dirs] = 0.0

    @ti.func
    def force_constraint(self, dirs):
        self.force[dirs] = 0.0
        self.forces[dirs] = 0.0
        self.forcef[dirs] = 0.0

    @ti.func
    def rigid_body_velocity_constraint(self, dirs):
        self.grad_domain[dirs] = 0.0

    @ti.func
    def reflection_constraint(self, dirs, signs):
        pre_velocity = self.momentum[dirs]
        if pre_velocity * signs > 0:
            self.momentum[dirs] = 0.0
            self.force[dirs] = 0.0

    @ti.func
    def rigid_body_reflection_constraint(self, dirs, signs):
        if self.grad_domain[dirs] * signs > 0.0:
            self.grad_domain[dirs] = 0.0
