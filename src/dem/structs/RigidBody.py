import taichi as ti
from src.utils.constants import PI, ZEROVEC3F
from src.utils.TypeDefinition import vec3f, vec3u8, vec4f
from src.utils.BitFunction import zero2_one_vector


@ti.dataclass
class RigidBody:
    groupID: ti.u8
    materialID: ti.u8
    startNode: int
    endNode: int
    localNode: int
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
    is_soft: ti.u8
    softID: int

    @ti.func
    def _restart(
        self,
        group_id,
        material_id,
        start_node,
        end_node,
        local_node,
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
        self.startNode = int(start_node)
        self.endNode = int(end_node)
        self.localNode = int(local_node)
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
        self.is_soft = ti.u8(0)
        self.softID = -1

    @ti.func
    def _add_body_attribute(self, centor_of_mass, mass, equiv_rad, inv_inertia, q):
        self.mass_center = float(centor_of_mass)
        self.m = float(mass)
        self.equi_r = float(equiv_rad)
        self.inv_I = float(inv_inertia)
        self.q = float(q)

    @ti.func
    def _add_body_properties(self, material_id, group_id, density):
        self.materialID = ti.u8(material_id)
        self.groupID = ti.u8(group_id)
        self.m *= density
        self.inv_I /= density

    @ti.func
    def _add_surface_index(self, start_index, end_index, local_index):
        self.startNode = int(start_index)
        self.endNode = int(end_index)
        self.localNode = int(local_index)

    @ti.func
    def _add_body_kinematic(self, init_v, init_w, is_fix):
        self.v = float(init_v)
        self.w = float(init_w)
        self.angmoment = self.w / self.inv_I
        self.is_fix = ti.cast(zero2_one_vector(is_fix), ti.u8)
        self.is_soft = ti.u8(0)
        self.softID = -1

    @ti.func
    def _mark_soft_body(self, soft_id):
        self.is_soft = ti.u8(1)
        self.softID = int(soft_id)

    @ti.func
    def _scale(self, factor):
        self.m *= float(factor * factor * factor)
        self.inv_I /= float(factor * factor * factor * factor * factor)

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
