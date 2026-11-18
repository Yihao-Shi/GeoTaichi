from src.dem.structs.ParticleFamily import ParticleFamily
import taichi as ti
from src.utils.TypeDefinition import vec3f, vec3u8, vec4f
from src.utils.BitFunction import zero2_one_vector
from src.utils.constants import ZEROVEC3F
from src.utils.TypeDefinition import vec3f
import numpy as np
from src.utils.TypeDefinition import vec3f, vec3i, vec2i
from src.utils.VectorFunction import squared


@ti.dataclass
class LevelSetGrid:
    distance_field: float
    distance_field_temp: float
    distance_field0: float

    @ti.func
    def _set_grid(self, sdf):
        self.distance_field = float(sdf)
        self.distance_field_temp = float(sdf)
        self.distance_field0 = float(sdf)

    @ti.func
    def _get_sdf(self):
        return self.distance_field

    @ti.func
    def _scale(self, scale):
        self.distance_field *= float(scale)
        self.distance_field_temp *= float(scale)
        self.distance_field0 *= float(scale)

    @ti.func
    def distance(self, scale):
        return self.distance_field * scale


@ti.dataclass
class HierarchicalBody:
    level: ti.u8
    max_potential_particle_pairs: int
    max_potential_wall_pairs: int

    @ti.func
    def _set(self, potential_particle_ratio, body_coordination_number, wall_coordination_number):
        self.max_potential_particle_pairs = ti.ceil(potential_particle_ratio * body_coordination_number)
        self.max_potential_wall_pairs = wall_coordination_number

    @ti.func
    def _set_level(self, level):
        self.level = ti.u8(level)

    @ti.func
    def potential_particle_num(self):
        return self.max_potential_particle_pairs

    @ti.func
    def potential_wall_num(self):
        return self.max_potential_wall_pairs


@ti.dataclass
class HierarchicalCell:
    pnum: int
    rad_min: float
    rad_max: float
    grid_size: float
    igrid_size: float
    factor: float
    cell_index: int
    cnum: vec3i
    wall_per_cell: int
    wall_cells: int

    @ti.func
    def _set(self, grid_size, factor, cnum, wall_per_cell):
        self.grid_size = float(grid_size)
        self.igrid_size = 1.0 / float(grid_size)
        self.factor = float(factor)
        self.cnum = cnum
        self.wall_per_cell = wall_per_cell

    @ti.func
    def _set_wall_cells(self, wall_cells):
        self.wall_cells = wall_cells

    @ti.func
    def _set_cell_index(self, csum):
        self.cell_index = float(csum)

    @ti.func
    def _cell_sum(self):
        cnum = self.cnum
        return cnum[0] * cnum[1] * cnum[2]

    @ti.func
    def _calculate(self):
        pass


@ti.dataclass
class EnergyFamily:
    kinetic: float
    potential: float


@ti.dataclass
class LSPotentialContact:
    end1: int
    end2: int

    @ti.func
    def _set(self, end1, end2):
        self.end1 = end1
        self.end2 = end2


@ti.dataclass
class ContactTable:
    endID1: int
    endID2: int
    cnforce: vec3f
    csforce: vec3f
    oldTangOverlap: vec3f
    normalOverlap: float
    normalOverlapActive: ti.u8

    @ti.func
    def _set_id(self, end_id1, end_id2):
        self.endID1 = end_id1
        self.endID2 = end_id2

    @ti.func
    def _set_contact(self, cnforce, csforce, overlap):
        self.cnforce = cnforce
        self.csforce = csforce
        self.oldTangOverlap = overlap

    @ti.func
    def _no_contact(self):
        self.cnforce = ZEROVEC3F
        self.csforce = ZEROVEC3F
        self.oldTangOverlap = ZEROVEC3F
        self.normalOverlap = 0.0
        self.normalOverlapActive = ti.u8(0)

    @ti.func
    def _is_active(self):
        return squared(self.cnforce) > 0.0


@ti.dataclass
class ISContactTable:
    endID1: int
    endID2: int
    cnforce: vec3f
    csforce: vec3f
    oldTangOverlap: vec3f
    contactSA: vec3f

    @ti.func
    def _set_id(self, end_id1, end_id2):
        self.endID1 = end_id1
        self.endID2 = end_id2

    @ti.func
    def _set_contact(self, cnforce, csforce, overlap, contact_sa):
        self.cnforce = cnforce
        self.csforce = csforce
        self.oldTangOverlap = overlap
        self.contactSA = contact_sa

    @ti.func
    def _no_contact(self, contact_sa):
        self.cnforce = ZEROVEC3F
        self.csforce = ZEROVEC3F
        self.oldTangOverlap = ZEROVEC3F
        self.contactSA = contact_sa

    @ti.func
    def _is_active(self):
        return squared(self.cnforce) > 0.0


@ti.dataclass
class VerletContactTable:
    endID1: int
    endID2: int
    verletDisp: vec3f

    @ti.func
    def _renew_verlet(self):
        self.verletDisp = ZEROVEC3F


@ti.dataclass
class CoupledContactTable:
    endID1: int
    endID2: int
    oldTangOverlap: vec3f
    normalOverlap: float
    normalOverlapActive: ti.u8

    @ti.func
    def _set_id(self, end_id1, end_id2):
        self.endID1 = end_id1
        self.endID2 = end_id2

    @ti.func
    def _set_contact(self, normal_force, tangential_force, overlap):
        self.oldTangOverlap = overlap

    @ti.func
    def _no_contact(self):
        self.oldTangOverlap = ZEROVEC3F
        self.normalOverlap = 0.0
        self.normalOverlapActive = ti.u8(0)

    @ti.func
    def _is_active(self):
        return squared(self.oldTangOverlap) > 0.0


@ti.dataclass
class CoupledRollingContactTable:
    endID1: int
    endID2: int
    oldTangOverlap: vec3f
    oldRollAngle: vec3f
    oldTwistAngle: vec3f

    @ti.func
    def _set_id(self, end_id1, end_id2):
        self.endID1 = end_id1
        self.endID2 = end_id2

    @ti.func
    def _set_contact(self, normal_force, tangential_force, tangential_overlap, rolling_overlap, twisting_overlap):
        self.oldTangOverlap = tangential_overlap
        self.oldRollAngle = rolling_overlap
        self.oldTwistAngle = twisting_overlap

    @ti.func
    def _no_contact(self):
        self.oldTangOverlap = ZEROVEC3F
        self.oldRollAngle = ZEROVEC3F
        self.oldTwistAngle = ZEROVEC3F

    @ti.func
    def _is_active(self):
        return squared(self.oldTangOverlap) > 0.0


@ti.dataclass
class DigitalContactTable:
    oldTangOverlap: vec3f

    @ti.func
    def _set_contact(self, var1, var2, tangential_overlap):
        self.oldTangOverlap = tangential_overlap

    @ti.func
    def _no_contact(self):
        self.oldTangOverlap = ZEROVEC3F

    @ti.func
    def _is_active(self):
        return squared(self.oldTangOverlap) > 0.0


@ti.dataclass
class DigitalRollingContactTable:
    oldTangOverlap: vec3f
    oldRollAngle: vec3f
    oldTwistAngle: vec3f

    @ti.func
    def _set_contact(self, tangential_overlap, rolling_overlap, twisting_overlap):
        self.oldTangOverlap = tangential_overlap
        self.oldRollAngle = rolling_overlap
        self.oldTwistAngle = twisting_overlap

    @ti.func
    def _no_contact(self):
        self.oldTangOverlap = ZEROVEC3F
        self.oldRollAngle = ZEROVEC3F
        self.oldTwistAngle = ZEROVEC3F

    @ti.func
    def _is_active(self):
        return squared(self.oldTangOverlap) > 0.0


@ti.dataclass
class RollingContactTable:
    endID1: int
    endID2: int
    cnforce: vec3f
    csforce: vec3f
    oldTangOverlap: vec3f
    oldRollAngle: vec3f
    oldTwistAngle: vec3f

    @ti.func
    def _set_id(self, end_id1, end_id2):
        self.endID1 = end_id1
        self.endID2 = end_id2

    @ti.func
    def _set_contact(self, cnforce, csforce, tangential_overlap, rolling_overlap, twisting_overlap):
        self.cnforce = cnforce
        self.csforce = csforce
        self.oldTangOverlap = tangential_overlap
        self.oldRollAngle = rolling_overlap
        self.oldTwistAngle = twisting_overlap

    @ti.func
    def _no_contact(self):
        self.cnforce = ZEROVEC3F
        self.csforce = ZEROVEC3F
        self.oldTangOverlap = ZEROVEC3F
        self.oldRollAngle = ZEROVEC3F
        self.oldTwistAngle = ZEROVEC3F

    @ti.func
    def _is_active(self):
        return squared(self.cnforce) > 0.0


@ti.dataclass
class RollingISContactTable:
    endID1: int
    endID2: int
    cnforce: vec3f
    csforce: vec3f
    oldTangOverlap: vec3f
    oldRollAngle: vec3f
    oldTwistAngle: vec3f
    contactSA: vec3f

    @ti.func
    def _set_id(self, end_id1, end_id2):
        self.endID1 = end_id1
        self.endID2 = end_id2

    @ti.func
    def _set_contact(self, cnforce, csforce, tangential_overlap, rolling_overlap, twisting_overlap, contact_sa):
        self.cnforce = cnforce
        self.csforce = csforce
        self.oldTangOverlap = tangential_overlap
        self.oldRollAngle = rolling_overlap
        self.oldTwistAngle = twisting_overlap
        self.contactSA = contact_sa

    @ti.func
    def _no_contact(self, contact_sa):
        self.cnforce = ZEROVEC3F
        self.csforce = ZEROVEC3F
        self.oldTangOverlap = ZEROVEC3F
        self.oldRollAngle = ZEROVEC3F
        self.oldTwistAngle = ZEROVEC3F
        self.contactSA = contact_sa

    @ti.func
    def _is_active(self):
        return squared(self.cnforce) > 0.0


@ti.dataclass
class HistoryISContactTable:
    DstID: int
    oldTangOverlap: vec3f
    contactSA: vec3f

    @ti.func
    def _copy(self, end_id, overlap, contact_sa):
        self.DstID = end_id
        self.oldTangOverlap = overlap
        self.contactSA = contact_sa


@ti.dataclass
class HistoryContactTable:
    DstID: int
    oldTangOverlap: vec3f
    normalOverlap: float
    normalOverlapActive: ti.u8

    @ti.func
    def _copy(self, end_id, overlap):
        self.DstID = end_id
        self.oldTangOverlap = overlap
        self.normalOverlap = 0.0
        self.normalOverlapActive = ti.u8(0)


@ti.dataclass
class HistoryRollingISContactTable:
    DstID: int
    oldTangOverlap: vec3f
    oldRollAngle: vec3f
    oldTwistAngle: vec3f
    contactSA: vec3f

    @ti.func
    def _copy(self, end_id, tangential_overlap, rolling_overlap, twisting_overlap, contact_sa):
        self.DstID = end_id
        self.oldTangOverlap = tangential_overlap
        self.oldRollAngle = rolling_overlap
        self.oldTwistAngle = twisting_overlap
        self.contactSA = contact_sa


@ti.dataclass
class HistoryRollingContactTable:
    DstID: int
    oldTangOverlap: vec3f
    oldRollAngle: vec3f
    oldTwistAngle: vec3f

    @ti.func
    def _copy(self, end_id, tangential_overlap, rolling_overlap, twisting_overlap):
        self.DstID = end_id
        self.oldTangOverlap = tangential_overlap
        self.oldRollAngle = rolling_overlap
        self.oldTwistAngle = twisting_overlap


class DigitalElevationModel(object):
    def __init__(self) -> None:
        self.materialID = 0
        self.digital_size = 0.0
        self.idigital_size = 0.0
        self.digital_dim = [0, 0]
        self.height_dim = [0, 0]
        self.no_data = -9999.0
        self.height = None

    def set_digital_elevation(self, materialID, cell_size, cnum, grid_number=None, no_data=-9999.0, height=None):
        self.materialID = materialID
        self.digital_size = cell_size
        self.idigital_size = 1.0 / cell_size
        self.digital_dim = vec2i(cnum)
        if grid_number is None:
            grid_number = [int(cnum[0]) + 1, int(cnum[1]) + 1]
        self.height_dim = vec2i(grid_number)
        self.no_data = no_data
        if height is not None:
            height = np.ascontiguousarray(height, dtype=np.float32)
            self.height = ti.field(float, shape=int(height.shape[0]))
            self.height.from_numpy(height)


@ti.dataclass
class SphereFamily:  # device memory: 48B
    grainIndex: int
    sphereIndex: int
    inv_I: float
    a: vec3f
    angmoment: vec3f
    q: vec4f
    fix_v: vec3u8
    fix_w: vec3u8

    @ti.func
    def _restart(self, grain_index, sphere_index, inv_I, q, a, angmoment, fix_v, fix_w):
        self.grainIndex = grain_index
        self.sphereIndex = int(sphere_index)
        self.inv_I = float(inv_I)
        self.q = float(q)
        self.a = a
        self.angmoment = angmoment
        self.fix_v = ti.cast(fix_v, ti.u8)
        self.fix_w = ti.cast(fix_w, ti.u8)

    @ti.func
    def _scale(self, factor):
        self.inv_I /= factor * factor * factor * factor * factor

    @ti.func
    def _add_index(self, grain_index, sphere_index):
        self.grainIndex = grain_index
        self.sphereIndex = int(sphere_index)

    @ti.func
    def _add_sphere_attribute(self, init_w, inv_inertia, q, fix_v, fix_w):
        self.inv_I = float(inv_inertia)
        self.q = float(q)
        self.fix_v = ti.cast(zero2_one_vector(fix_v), ti.u8)
        self.fix_w = ti.cast(zero2_one_vector(fix_w), ti.u8)
        self.angmoment = init_w / self.inv_I * int(self.fix_w)

    @ti.func
    def _get_sphere_index(self):
        return self.sphereIndex

    @ti.func
    def _get_inverse_inertia(self):
        return self.inv_I

    @ti.func
    def _get_quaternion(self):
        return self.q

    @ti.func
    def _get_is_velocity_fixed(self):
        return self.fix_v

    @ti.func
    def _get_is_angular_velocity_fixed(self):
        return self.fix_w

    def deactivate(self, particle: ParticleFamily):
        particle[self.sphereIndex] = ti.u8(0)


@ti.dataclass
class VerticeNode:
    x: vec3f
    x0: vec3f
    v: vec3f
    contact_force: vec3f
    external_force: vec3f
    parameter: float

    @ti.func
    def _restart(self, parameter, x):
        self.parameter = float(parameter)
        self.x = float(x)
        self.x0 = float(x)
        self.v = ZEROVEC3F
        self.contact_force = ZEROVEC3F
        self.external_force = ZEROVEC3F

    @ti.func
    def _set_surface_node(self, x):
        self.x = float(x)
        self.x0 = float(x)
        self.v = ZEROVEC3F
        self.contact_force = ZEROVEC3F
        self.external_force = ZEROVEC3F

    @ti.func
    def _set_coefficient(self, coeff):
        self.parameter = coeff

    @ti.func
    def _scale(self, scale, centor_of_mass):
        self.x = float(scale * (self.x - centor_of_mass) + centor_of_mass)
        self.x0 = float(scale * (self.x0 - centor_of_mass) + centor_of_mass)

    @ti.func
    def _reset_contact(self):
        self.contact_force = ZEROVEC3F

    @ti.func
    def _set_external_force(self, force):
        self.external_force = force

    @ti.func
    def _update_contact_interaction(self, cforce, ctorque):
        self.contact_force += cforce

    @ti.func
    def _update_kinematics(self, x, velocity):
        self.x = float(x)
        self.v = float(velocity)


@ti.dataclass
class BoundingSphere:
    active: ti.u8
    rad: float
    x: vec3f
    verletDisp: vec3f

    @ti.func
    def _restart(self, active, bounding_center, bounding_radius):
        self.active = ti.u8(active)
        self.x = float(bounding_center)
        self.rad = float(bounding_radius)

    @ti.func
    def _add_bounding_sphere(self, bounding_center, bounding_radius):
        self.active = ti.u8(1)
        self.x = float(bounding_center)
        self.rad = float(bounding_radius)

    @ti.func
    def _move(self, disp, true_disp):
        self.x += disp
        self.verletDisp += true_disp

    @ti.func
    def _scale(self, scale, centor_of_mass):
        self.rad *= float(scale)
        self.x = float(scale * (self.x - centor_of_mass) + centor_of_mass)

    @ti.func
    def _translate(self, offset):
        self.x += offset

    @ti.func
    def _renew_verlet(self):
        self.verletDisp = ZEROVEC3F

    @ti.func
    def _get_multisphere_index1(self):
        return -1

    @ti.func
    def _get_multisphere_index2(self):
        return 1

    @ti.func
    def _get_position(self):
        return self.x

    @ti.func
    def _get_radius(self):
        return self.rad

    @ti.func
    def _get_verlet_displacement(self):
        return self.verletDisp
