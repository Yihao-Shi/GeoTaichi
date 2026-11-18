import taichi as ti
from src.utils.constants import ZEROVEC3F
from src.utils.TypeDefinition import vec3f


@ti.dataclass
class PlaneFamily:
    active: ti.u8
    wallID: int
    materialID: ti.u8
    point: vec3f
    norm: vec3f

    @ti.pyfunc
    def add_material_id(self, mat_id):
        self.materialID = mat_id

    @ti.pyfunc
    def add_wall_geometry(self, input_wall_id, point, norm):
        self.wallID = input_wall_id
        self.active = 1
        self.point = vec3f([float(center) for center in point])
        self.norm = vec3f([float(normal) for normal in norm])

    def deactivate(self):
        self.active = 0

    @ti.func
    def _restart(self, active, wall_id, material_id, point, norm):
        self.active = ti.u8(active)
        self.wallID = int(wall_id)
        self.materialID = ti.u8(material_id)
        self.point = float(point)
        self.norm = float(norm)

    @ti.func
    def _get_status(self):
        return self.active

    @ti.func
    def _get_material_id(self):
        return self.materialID

    @ti.func
    def _get_center(self):
        return self.point

    @ti.func
    def _get_norm(self, pos=0):
        return self.norm

    @ti.func
    def _get_velocity(self):
        return ZEROVEC3F

    @ti.func
    def _update_contact_stiffness(self, stiffess):
        pass

    @ti.func
    def _update_contact_interaction(self, cforce):
        pass

    @ti.func
    def _move(self, disp):
        pass

    @ti.func
    def _renew_verlet(self):
        pass

    @ti.func
    def _point_projection(self, point):
        center = self._get_center()
        norm = self._get_norm()
        distance = (point - center).dot(norm)
        return point - distance * norm

    @ti.func
    def _point_to_wall_distance(self, point):
        return (point - self._get_center()).dot(self.norm)

    @ti.func
    def _get_norm_distance(self, point):
        return (point - self._get_center()).dot(self.norm)

    @ti.func
    def _is_in_plane(self, point):
        return 1

    @ti.func
    def _is_sphere_intersect(self, position, contact_radius):
        distance = self._point_to_wall_distance(position)
        return distance <= contact_radius

    @ti.func
    def process_circle_shape(self, point, distance, criteria):
        return 1.0

    @ti.func
    def process_implicit_surface_shape(self):
        return 1.0
