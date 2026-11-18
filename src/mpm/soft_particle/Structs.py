import taichi as ti
from src.utils.constants import ZEROVEC3F
from src.utils.TypeDefinition import real, vec3f
from src.utils.VectorFunction import vsign
from src.utils.TypeDefinition import mat3x3, real, vec3f


@ti.dataclass
class TemplateSoftNode:
    parameter: real

    @ti.func
    def _set_coefficient(self, coeff):
        self.parameter = coeff


@ti.dataclass
class SoftSurfacePoint:
    pointID: int
    contact_force: vec3f

    @ti.func
    def _reset(self):
        self.contact_force = ZEROVEC3F

    @ti.func
    def _update_contact_interaction(self, cforce, ctorque):
        self.contact_force += cforce


@ti.dataclass
class SoftPoint:
    x: vec3f
    v: vec3f

    @ti.func
    def _set_surface_node(self, x):
        self.x = float(x)

    @ti.func
    def _scale(self, scale, centor_of_mass):
        self.x = float(scale * (self.x - centor_of_mass) + centor_of_mass)


class VerticeSoftNode:
    def __init__(self, surface_point, material_point) -> None:
        self.template_point = TemplateSoftNode.field(shape=material_point)
        self.surface_point = SoftPoint(shape=material_point)
        self.soft_point = SoftSurfacePoint(shape=surface_point)


@ti.dataclass
class DeformableGrid:
    m: real
    f: vec3f
    v: vec3f
    contact_force: vec3f

    @ti.func
    def _grid_reset(self):
        pending_contact_force = self.contact_force
        self.m = 0.0
        self.v = ZEROVEC3F
        self.f = pending_contact_force
        self.contact_force = ZEROVEC3F

    @ti.func
    def _tlmpm_step_reset(self):
        """Reset forces while retaining cross-step mass and velocity."""
        pending_contact_force = self.contact_force
        self.f = pending_contact_force
        self.contact_force = ZEROVEC3F

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.v += momentum

    @ti.func
    def _compute_nodal_velocity(self):
        self.v /= self.m

    @ti.func
    def _update_nodal_force(self, force):
        self.f += force

    @ti.func
    def _update_external_force(self, external_force):
        self.f += external_force

    @ti.func
    def _update_internal_force(self, internal_force):
        self.f += internal_force

    @ti.func
    def _compute_nodal_kinematic(self, damp, dt):
        unbalanced_force = self.f
        velocity = self.v
        if velocity.dot(unbalanced_force) > 0.0:
            unbalanced_force -= damp * unbalanced_force.norm() * vsign(velocity)
        acceleration = unbalanced_force / self.m
        self.v += acceleration * dt[None]
        self.f = acceleration

    @ti.func
    def _update_nodal_kinematic(self):
        self.f /= self.m
        self.v /= self.m


@ti.dataclass
class SoftMaterialPoint:
    active: int
    bodyID: int
    materialID: int
    groupID: int
    x: vec3f
    x0: vec3f
    v: vec3f
    contact_force: vec3f
    external_force: vec3f
    m: real
    vol0: real
    surface_weight: real
    F: mat3x3
    stress: mat3x3
    strain_energy: real

    @ti.func
    def _add_point(self, body_id, material_id, group_id, x, x0, v, mass, volume):
        self.active = 1
        self.bodyID = int(body_id)
        self.materialID = int(material_id)
        self.groupID = int(group_id)
        self.x = float(x)
        self.x0 = float(x0)
        self.v = float(v)
        self.contact_force = ZEROVEC3F
        self.external_force = ZEROVEC3F
        self.m = float(mass)
        self.vol0 = float(volume)
        self.surface_weight = 0.0
        self.F = ti.Matrix.identity(float, 3)
        self.stress = ti.Matrix.zero(float, 3, 3)
        self.strain_energy = 0.0

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
    def _set_surface_weight(self, weight):
        self.surface_weight = float(weight)

    @ti.func
    def _reset_contact(self):
        self.contact_force = ZEROVEC3F

    @ti.func
    def _set_external_force(self, force):
        self.external_force = float(force)

    @ti.func
    def _update_contact_interaction(self, cforce, ctorque):
        self.contact_force += cforce
