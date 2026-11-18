import taichi as ti
from src.utils.constants import PI, ZEROVEC3F
from src.utils.GeometryFunction import sphere_triangle_intersection_area, distance_from_point_to_triangle
from src.utils.TypeDefinition import vec3f, vec3d


@ti.dataclass
class FacetFamily:
    wallID: int
    active: ti.u8
    materialID: ti.u8
    contact_stiffness: float
    vertice1: vec3f
    vertice2: vec3f
    vertice3: vec3f
    reference1: vec3f
    reference2: vec3f
    reference3: vec3f
    translation: vec3d
    verletDisp: vec3f
    norm: vec3f
    v: vec3f
    contact_force: vec3f
    bound_beg: vec3f
    bound_end: vec3f

    @ti.pyfunc
    def add_material_id(self, mat_id):
        self.materialID = mat_id

    @ti.pyfunc
    def add_wall_geometry(self, input_wall_id, vertice1, vertice2, vertice3, norm, init_v):
        self.wallID = int(input_wall_id)
        self.active = 1
        self.vertice1 = vec3f([float(vertice) for vertice in vertice1])
        self.vertice2 = vec3f([float(vertice) for vertice in vertice2])
        self.vertice3 = vec3f([float(vertice) for vertice in vertice3])
        self.reference1 = self.vertice1
        self.reference2 = self.vertice2
        self.reference3 = self.vertice3
        self.translation = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
        self.norm = norm
        self.v = init_v

    @ti.pyfunc
    def add_wall_geometry_from_vertices(self, wall_id, vertice1, vertice2, vertice3, vel):
        self.active = 1
        self.wallID = int(wall_id)
        self.vertice1 = vec3f([float(vertice) for vertice in vertice1])
        self.vertice2 = vec3f([float(vertice) for vertice in vertice2])
        self.vertice3 = vec3f([float(vertice) for vertice in vertice3])
        self.reference1 = self.vertice1
        self.reference2 = self.vertice2
        self.reference3 = self.vertice3
        self.translation = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
        self.norm = ((vertice2 - vertice1).cross(vertice3 - vertice1)).normalized()
        self.v = vel

    def deactivate(self):
        self.active = 0

    @ti.func
    def _restart(self, active, wall_id, material_id, point1, point2, point3, norm, velocity):
        self.active = ti.u8(active)
        self.wallID = int(wall_id)
        self.materialID = ti.u8(material_id)
        self.vertice1 = float(point1)
        self.vertice2 = float(point2)
        self.vertice3 = float(point3)
        self.reference1 = float(point1)
        self.reference2 = float(point2)
        self.reference3 = float(point3)
        self.translation = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
        self.norm = float(norm)
        self.v = float(velocity)

    @ti.func
    def _contact_force_reset(self):
        self.contact_force = ZEROVEC3F

    @ti.func
    def _contact_stiffness_reset(self):
        self.contact_stiffness = 0.0

    @ti.func
    def _reset(self):
        self.contact_force = ZEROVEC3F
        self.contact_stiffness = 0.0

    @ti.func
    def _get_square(self):
        a = self.vertice2 - self.vertice1
        b = self.vertice3 - self.vertice1
        return 0.5 * ti.sqrt(
            (a[1] * b[2] - b[1] * a[2]) * (a[1] * b[2] - b[1] * a[2])
            + (a[0] * b[2] - b[0] * a[2]) * (a[0] * b[2] - b[0] * a[2])
            + (a[0] * b[1] - b[0] * a[1]) * (a[0] * b[1] - b[0] * a[1])
        )

    @ti.func
    def _move(self, disp):
        self.translation += ti.cast(disp, ti.f64)
        translation = ti.cast(self.translation, ti.f32)
        self.vertice1 = self.reference1 + translation
        self.vertice2 = self.reference2 + translation
        self.vertice3 = self.reference3 + translation
        self.verletDisp += disp

    @ti.func
    def _renew_verlet(self):
        self.verletDisp = ZEROVEC3F

    @ti.func
    def _get_status(self):
        return self.active

    @ti.func
    def _get_material_id(self):
        return self.materialID

    @ti.func
    def _get_vertice1(self):
        return self.vertice1

    @ti.func
    def _get_vertice2(self):
        return self.vertice2

    @ti.func
    def _get_vertice3(self):
        return self.vertice3

    @ti.func
    def _get_norm(self, pos=0):
        return self.norm

    @ti.func
    def _get_bounding_box(self):
        return self.bound_beg, self.bound_end

    @ti.func
    def _get_center(self):
        return (self.vertice1 + self.vertice2 + self.vertice3) / 3.0

    @ti.func
    def _get_velocity(self):
        return self.v

    @ti.func
    def _update_contact_stiffness(self, stiffess):
        self.contact_stiffness += stiffess

    @ti.func
    def _update_contact_interaction(self, cforce):
        self.contact_force += cforce

    @ti.func
    def _point_projection(self, point):
        center = self._get_center()
        norm = self._get_norm()
        distance = (point - center).dot(norm)
        return point - distance * norm

    @ti.func
    def _point_projection_by_distance(self, point, distance):
        return point - distance * self.norm

    @ti.func
    def _point_to_wall_distance(self, point):
        return distance_from_point_to_triangle(point, self.vertice1, self.vertice2, self.vertice3, -self.norm)

    @ti.func
    def _get_norm_distance(self, point):
        return (point - self._get_center()).dot(self.norm)

    @ti.func
    def _is_positive_direction(self, point):
        return (point - self.vertice1).dot(self.norm) > 0.0

    @ti.func
    def _is_in_plane(self, point):
        p1 = self.vertice1
        p2 = self.vertice2
        p3 = self.vertice3
        u = (p1 - point).cross(p2 - point)
        v = (p2 - point).cross(p3 - point)
        w = (p3 - point).cross(p1 - point)
        return u.dot(v) >= 0.0 and u.dot(w) >= 0.0

    @ti.func
    def _get_contact_type(self, projection_point):
        contact_type = 0
        u = (self.vertice1 - projection_point).cross(self.vertice2 - projection_point)
        v = (self.vertice2 - projection_point).cross(self.vertice3 - projection_point)
        w = (self.vertice3 - projection_point).cross(self.vertice1 - projection_point)
        if u.dot(v) > 0.0 and u.dot(w) > 0.0 and v.dot(w) > 0.0:
            contact_type = 1
        elif u.dot(v) == 0.0 and u.dot(w) == 0.0 and v.dot(w) == 0.0:
            contact_type = 3
        elif u.dot(v) > 0.0 or u.dot(w) > 0.0 or v.dot(w) > 0.0:
            contact_type = 2
        return contact_type

    @ti.func
    def _bounding_box(self):
        xmin, ymin, zmin = self._wall_boundary_min()
        xmax, ymax, zmax = self._wall_boundary_max()
        self.bound_beg = vec3f([xmin, ymin, zmin])
        self.bound_end = vec3f([xmax, ymax, zmax])

    @ti.func
    def _wall_boundary_min(self):
        return (
            ti.min(self.vertice1[0], self.vertice2[0], self.vertice3[0]),
            ti.min(self.vertice1[1], self.vertice2[1], self.vertice3[1]),
            ti.min(self.vertice1[2], self.vertice2[2], self.vertice3[2]),
        )

    @ti.func
    def _wall_boundary_max(self):
        return (
            ti.max(self.vertice1[0], self.vertice2[0], self.vertice3[0]),
            ti.max(self.vertice1[1], self.vertice2[1], self.vertice3[1]),
            ti.max(self.vertice1[2], self.vertice2[2], self.vertice3[2]),
        )

    @ti.func
    def _is_sphere_intersect(self, position, contact_radius):
        distance = self._point_to_wall_distance(
            position
        )  # self._get_norm_distance(position) or self._point_to_wall_distance(position) ???
        # in_plane = self._is_in_plane(self._point_projection_by_distance(position, distance))
        return distance <= contact_radius  # and in_plane

    @ti.func
    def process_circle_shape(self, point, radius, distance):
        fraction = 0.0
        if 0.0 < distance < radius:
            r = ti.sqrt(radius * radius - distance * distance)
            area0 = PI * r * r
            position = self._point_projection_by_distance(point, distance)
            area = sphere_triangle_intersection_area(
                position, r, self.vertice1, self.vertice2, self.vertice3, self.norm
            )
            fraction = area / area0
        return fraction

    @ti.func
    def process_implicit_surface_shape(self):
        return 1.0
