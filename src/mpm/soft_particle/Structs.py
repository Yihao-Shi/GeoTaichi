import taichi as ti

from src.utils.constants import ZEROVEC3f
from src.utils.TypeDefination import mat3x3, vec3f, vec3i
from src.utils.VectorFunction import vsign


@ti.dataclass
class DeformableGrid:
    m: float
    f: vec3f
    v: vec3f
    contact_force: vec3f

    @ti.func
    def _grid_reset(self):
        pending_contact_force = self.contact_force
        self.m = 0.0
        self.v = ZEROVEC3f
        self.f = pending_contact_force
        self.contact_force = ZEROVEC3f

    @ti.func
    def _tlmpm_step_reset(self):
        """Reset forces while retaining cross-step mass and velocity."""
        pending_contact_force = self.contact_force
        self.f = pending_contact_force
        self.contact_force = ZEROVEC3f

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
class TemplateSoftNode:
    parameter: float

    @ti.func
    def _set_coefficient(self, coeff):
        self.parameter = coeff


@ti.dataclass
class SoftSurfacePoint:
    pointID: int
    contact_force: vec3f

    @ti.func
    def _reset(self):
        self.contact_force = ZEROVEC3f

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
class SoftBody:
    bodyID: int
    groupID: int
    materialID: int
    startIndex: int
    endIndex: int
    startNode: int
    endNode: int
    localNode: int
    surfacePointStart: int
    surfacePointEnd: int
    gridStart: int
    mpmGridStart: int
    gridNum: int
    mpmGridNum: int
    mpmGridOffset: vec3i
    mpmGridSize: vec3i
    templatePointStart: int
    templateSurfaceStart: int
    templateSdfStart: int
    gridType: ti.u8
    scale: float
    gridSpace: float
    referenceRotation: mat3x3
    m: float
    mass_center0: vec3f
    mass_center: vec3f
    previous_center: vec3f
    previous_rotation: mat3x3
    surface_inertia: mat3x3
    v: vec3f
    contact_force: vec3f
    external_load_factor: float
    damp_energy: float

    @ti.func
    def _restart(self, bodyID, startIndex, endIndex, groupID, materialID):
        self.bodyID = int(bodyID)
        self.startIndex = int(startIndex)
        self.endIndex = int(endIndex)
        self.groupID = int(groupID)
        self.materialID = int(materialID)

    @ti.func
    def _add_body_attribute(self, mass, mass_center):
        self.m = float(mass)
        self.mass_center0 = float(mass_center)
        self.mass_center = float(mass_center)
        self.previous_center = float(mass_center)
        self.v = ZEROVEC3f
        self.contact_force = ZEROVEC3f
        self.external_load_factor = 1.0
        self.damp_energy = 0.0

    @ti.func
    def _set_external_load_factor(self, factor):
        self.external_load_factor = float(factor)

    @ti.func
    def _add_body_properties(self, materialID, groupID):
        self.materialID = int(materialID)
        self.groupID = int(groupID)

    @ti.func
    def _add_material_point_index(self, start_index, end_index):
        self.startIndex = int(start_index)
        self.endIndex = int(end_index)

    @ti.func
    def _add_grid_index(self, grid_start, mpm_grid_start, grid_num, mpm_grid_num, mpm_grid_offset, mpm_grid_size):
        self.gridStart = int(grid_start)
        self.mpmGridStart = int(mpm_grid_start)
        self.gridNum = int(grid_num)
        self.mpmGridNum = int(mpm_grid_num)
        self.mpmGridOffset = int(mpm_grid_offset)
        self.mpmGridSize = int(mpm_grid_size)

    @ti.func
    def _add_surface_index(self, start_index, end_index, local_index):
        self.startNode = int(start_index)
        self.endNode = int(end_index)
        self.localNode = int(local_index)

    @ti.func
    def _add_surface_point_index(self, start_index, end_index):
        self.surfacePointStart = int(start_index)
        self.surfacePointEnd = int(end_index)

    @ti.func
    def _add_template_support(
        self,
        point_start,
        surface_start,
        sdf_start,
        grid_type,
        scale,
        grid_space,
        reference_rotation,
    ):
        self.templatePointStart = int(point_start)
        self.templateSurfaceStart = int(surface_start)
        self.templateSdfStart = int(sdf_start)
        self.gridType = ti.u8(grid_type)
        self.scale = float(scale)
        self.gridSpace = float(grid_space)
        self.referenceRotation = reference_rotation

    @ti.func
    def _get_vertice_number(self):
        return int(self.endNode - self.startNode)

    @ti.func
    def _start_node(self):
        return self.localNode

    @ti.func
    def _end_node(self):
        return int(self.localNode + self.endNode - self.startNode)

    @ti.func
    def local_node_to_global(self, node):
        return int(node - self.localNode + self.startNode)

    @ti.func
    def global_node_to_local(self, node):
        return int(node - self.startNode + self.localNode)

    @ti.func
    def _get_material(self):
        return self.materialID

    @ti.func
    def _get_group(self):
        return self.groupID


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
    m: float
    vol0: float
    surface_weight: float
    F: mat3x3
    stress: mat3x3
    strain_energy: float

    @ti.func
    def _add_point(self, bodyID, materialID, groupID, x, x0, v, mass, volume):
        self.active = 1
        self.bodyID = int(bodyID)
        self.materialID = int(materialID)
        self.groupID = int(groupID)
        self.x = float(x)
        self.x0 = float(x0)
        self.v = float(v)
        self.contact_force = ZEROVEC3f
        self.external_force = ZEROVEC3f
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
        self.contact_force = ZEROVEC3f

    @ti.func
    def _set_external_force(self, force):
        self.external_force = float(force)

    @ti.func
    def _update_contact_interaction(self, cforce, ctorque):
        self.contact_force += cforce
