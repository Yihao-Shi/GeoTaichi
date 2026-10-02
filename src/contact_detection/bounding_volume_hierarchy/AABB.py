import taichi as ti

from src.utils.linalg import make_list
from src.utils.Quaternion import SetToRotate
from src.utils.TypeDefination import vec3f, vec4f


AABB_MIN_EXTENT = 1.0e-6


@ti.data_oriented
class AABB(object):
    """
    AABB (Axis-Aligned Bounding Box) class for managing collections of bounding boxes in batches.

    This class defines an axis-aligned bounding box (AABB) structure and provides a Taichi dataclass
    for efficient computation and intersection testing on the GPU. Each AABB is represented by its
    minimum and maximum 3D coordinates. The class supports batch processing of multiple AABBs.

    Attributes:
        n_aabbs (list/int): Number of AABBs per batch.
        ti_aabb (taichi.dataclass): Taichi dataclass representing an individual AABB with min and max vectors.
        aabbs (taichi.field): Taichi field storing all AABBs in the specified batches.

    Args:
        n_batches (int): Number of batches to allocate.
        n_aabbs (int): Number of AABBs per batch.

    Example:
        aabb_manager = AABB(n_batches=4, n_aabbs=128)
    """

    def __init__(self, n_aabbs, dimension=3, oriented=False):
        self.first_run = True
        self.dimension = dimension
        self.initilize(n_aabbs)

        if self.first_run:
            self.initial_fields(oriented)

    def initilize(self, n_aabbs):
        self.batch_size = [x for x in make_list(n_aabbs) if x != 0]
        self.n_aabbs = sum(self.batch_size)
        self.n_batches = len(self.batch_size)
        if self.n_batches > 128:
            raise RuntimeError("")

        self.prefix_batch_size = self.batch_size.copy()
        self.prefix_batch_size.insert(0, 0)
        for i in range(1, len(self.prefix_batch_size)):
            self.prefix_batch_size[i] = self.prefix_batch_size[i] + self.prefix_batch_size[i - 1]

    def reset(self, current_n_aabbs=None):
        if current_n_aabbs is None:
            self.current_batch_size = self.batch_size.copy()
        else:
            self.current_batch_size = [int(x) for x in make_list(current_n_aabbs)]
        if len(self.current_batch_size) != self.n_batches:
            raise RuntimeError("Active AABB batch count should match allocated batch count")
        for current_size, max_size in zip(self.current_batch_size, self.batch_size):
            if current_size <= 0:
                raise RuntimeError("Each active AABB batch should contain at least one object")
            if current_size > max_size:
                raise RuntimeError("Active AABB batch size exceeds allocated batch size")
        self.current_n_aabbs = sum(self.current_batch_size)
        self.current_n_batches = len(self.current_batch_size)
        if self.current_n_batches > 128:
            raise RuntimeError("")

        self.prefix_current_batch_size = self.current_batch_size.copy()
        self.prefix_current_batch_size.insert(0, 0)
        for i in range(1, len(self.prefix_current_batch_size)):
            self.prefix_current_batch_size[i] = self.prefix_current_batch_size[i] + self.prefix_current_batch_size[i - 1]

    def initial_fields(self, oriented=False):
        @ti.dataclass
        class ti_aabb:
            min: vec3f
            max: vec3f

            @ti.func
            def intersects(self, other, expand1, expand2) -> bool:
                """
                Check if this AABB intersects with another AABB.
                """
                return (
                    (self.min[0] - expand1) <= (other.max[0] + expand2)
                    and (self.max[0] + expand1) >= (other.min[0] - expand2)
                    and (self.min[1] - expand1) <= (other.max[1] + expand2)
                    and (self.max[1] + expand1) >= (other.min[1] - expand2)
                    and (self.min[2] - expand1) <= (other.max[2] + expand2)
                    and (self.max[2] + expand1) >= (other.min[2] - expand2)
                )
            
            @ti.func
            def inside(self, point, search_rad) -> bool:
                """
                Check if the point is inside this AABB.
                """
                return ((self.min[0] - search_rad) <= point[0] <= (self.max[0] + search_rad)
                        and (self.min[1] - search_rad) <= point[1] <= (self.max[1] + search_rad)
                        and (self.min[2] - search_rad) <= point[2] <= (self.max[2] + search_rad)
                )

        self.ti_aabb = ti_aabb
            
        if oriented:
            ti_aabb.members.update({"q": vec4f})  # Quaternion for orientation

        self.aabbs = ti_aabb.field(
            shape=self.n_aabbs,
            needs_grad=False,
            layout=ti.Layout.SOA,
        )
        self.first_run = False

    @ti.kernel
    def set_sphere_aabbs(self, particleNum: int, prefix_batch_size: int, verlet_distance: float, particle: ti.template()):
        for np in range(particleNum):
            position = particle[np].x
            radius = particle[np].rad
            self.update_sphere_aabb(np, position, radius, verlet_distance, batch=prefix_batch_size)

    @ti.kernel
    def set_levelset_body_aabbs(self, particleNum: int, prefix_batch_size: int, verlet_distance: float, rigid: ti.template(), box: ti.template()):
        for np in range(particleNum):
            self.update_levelset_body_aabb(
                np, rigid[np], box[np], verlet_distance,
                batch=prefix_batch_size)

    @ti.kernel
    def set_lsmpm_body_aabbs(self, particleNum: int,
                              prefix_batch_size: int,
                              verlet_distance: float,
                              particle: ti.template(),
                              rigid: ti.template(), box: ti.template()):
        for np in range(particleNum):
            if int(rigid[np].is_soft) == 1:
                self.update_lsmpm_soft_body_aabb(
                    np, rigid[np], box[np], verlet_distance,
                    batch=prefix_batch_size)
            else:
                self.update_levelset_body_aabb(
                    np, rigid[np], box[np], verlet_distance,
                    batch=prefix_batch_size)

    @ti.kernel
    def set_triangle_aabbs(self, triNum: int, prefix_batch_size: int, verlet_distance: float, triangle: ti.template()):
        for np in range(triNum):
            vertice1, vertice2, vertice3 = triangle[np].vertice1, triangle[np].vertice2, triangle[np].vertice3
            self.update_triangle_aabb(np, vertice1, vertice2, vertice3, verlet_distance, batch=prefix_batch_size)

    @ti.func
    def get_aabb(self, index, batch=0):
        return self.aabbs.min[batch + index], self.aabbs.max[batch + index]

    @ti.func
    def get_center(self, index, batch=0):
        return 0.5 * (self.aabbs.min[batch + index] + self.aabbs.max[batch + index])

    @ti.func
    def update_sphere_aabb(self, index, position, radius, verlet_distance, batch=0):
        box_min, box_max = self._update_sphere_aabb(position, radius, verlet_distance)
        self.aabbs.min[batch + index] = box_min
        self.aabbs.max[batch + index] = box_max

    @ti.func
    def update_levelset_body_aabb(self, index, rigid, box,
                                  verlet_distance, batch=0):
        rotate_matrix = SetToRotate(rigid.q)
        center = rigid.mass_center + rotate_matrix @ box._get_shape_center()
        half_extent = 0.5 * box._get_shape_dim()
        world_half = ti.Vector.zero(float, self.dimension)
        for i in ti.static(range(self.dimension)):
            world_half[i] = (
                ti.abs(rotate_matrix[i, 0]) * half_extent[0]
                + ti.abs(rotate_matrix[i, 1]) * half_extent[1]
                + ti.abs(rotate_matrix[i, 2]) * half_extent[2]
            )
        self.aabbs.min[batch + index] = (
            center - world_half - verlet_distance
        )
        self.aabbs.max[batch + index] = (
            center + world_half + verlet_distance
        )

    @ti.func
    def update_lsmpm_soft_body_aabb(self, index, rigid, box,
                                    verlet_distance, batch=0):
        rotate_matrix = SetToRotate(rigid.q)
        center = rigid.mass_center + rotate_matrix @ box._get_shape_center()
        half_extent = 0.5 * ti.max(box._get_shape_dim(), vec3f(0.0, 0.0, 0.0))
        # |R| e is the exact axis-aligned half extent of a rotated OBB. It is
        # equivalent to rotating all eight corners and taking componentwise
        # extrema; rotating only shape_min and shape_max would not be safe.
        world_half = ti.Vector.zero(float, self.dimension)
        for i in ti.static(range(self.dimension)):
            world_half[i] = (
                ti.abs(rotate_matrix[i, 0]) * half_extent[0]
                + ti.abs(rotate_matrix[i, 1]) * half_extent[1]
                + ti.abs(rotate_matrix[i, 2]) * half_extent[2]
            )
        padding = verlet_distance + box.grid_space
        self.aabbs.min[batch + index] = center - world_half - padding
        self.aabbs.max[batch + index] = center + world_half + padding

    @ti.func
    def update_triangle_aabb(self, index, vertice1, vertice2, vertice3, verlet_distance, batch=0):
        box_min, box_max = self._update_triangle_aabb(vertice1, vertice2, vertice3, verlet_distance)
        self.aabbs.min[batch + index] = box_min
        self.aabbs.max[batch + index] = box_max

    @ti.func
    def update_patch_aabb(self, index, point_cloud, verlet_distance, batch=0):
        box_min, box_max = self._update_patch_aabb(point_cloud, verlet_distance)
        self.aabbs.min[batch + index] = box_min
        self.aabbs.max[batch + index] = box_max

    @ti.func
    def _enlarge_degenerate_aabb(self, box_min, box_max):
        for d in ti.static(range(self.dimension)):
            extent = box_max[d] - box_min[d]
            min_extent = AABB_MIN_EXTENT * ti.max(ti.abs(box_min[d]), ti.abs(box_max[d]), 1.0)
            if extent < min_extent:
                padding = 0.5 * (min_extent - extent)
                box_min[d] -= padding
                box_max[d] += padding
        return box_min, box_max

    @ti.func
    def _update_sphere_aabb(self, position, radius, verlet_distance):
        box_min = position - radius * ti.Vector.one(float, self.dimension)
        box_max = position + radius * ti.Vector.one(float, self.dimension)
        return box_min - verlet_distance, box_max + verlet_distance

    @ti.func
    def _update_triangle_aabb(self, vertice1, vertice2, vertice3, verlet_distance):
        box_min = ti.Vector.zero(float, self.dimension)
        box_max = ti.Vector.zero(float, self.dimension)
        for d in ti.static(range(self.dimension)):
            box_min[d] = ti.min(vertice1[d], vertice2[d], vertice3[d])
            box_max[d] = ti.max(vertice1[d], vertice2[d], vertice3[d])
        box_min -= verlet_distance
        box_max += verlet_distance
        return self._enlarge_degenerate_aabb(box_min, box_max)

    @ti.func
    def _update_patch_aabb(self, point_cloud, verlet_distance):
        box_min = ti.Vector.zero(float, self.dimension)
        box_max = ti.Vector.zero(float, self.dimension)
        for d in ti.static(range(self.dimension)):
            box_min[d] = point_cloud[0, d]
            box_max[d] = point_cloud[0, d]
        for i in ti.static(range(1, point_cloud.n)):
            for d in ti.static(range(self.dimension)):
                box_min[d] = ti.min(box_min[d], point_cloud[i, d])
                box_max[d] = ti.max(box_max[d], point_cloud[i, d])
        box_min -= verlet_distance
        box_max += verlet_distance
        return self._enlarge_degenerate_aabb(box_min, box_max)

    @ti.func
    def _update_patch_obb(self, point_cloud):
        center = ti.Vector.zero(float, self.dimension)
        for i in ti.static(range(point_cloud.n)):
            global_coord = ti.Vector([point_cloud[i, d] for d in ti.static(range(self.dimension))])
            center += global_coord
        center /= point_cloud.n

        consistent_matrix = ti.Matrix.zero(float, self.dimension, self.dimension)
        for i in ti.static(range(point_cloud.n)):
            global_coord = ti.Vector([point_cloud[i, d] for d in ti.static(range(self.dimension))])
            center_coord = global_coord - center
            consistent_matrix += center_coord.outer_product(center_coord)
        consistent_matrix /= point_cloud.n

        _, eigen_vector = ti.sym_eig(consistent_matrix)
        local_coords = ti.Matrix.zero(float, point_cloud.n, self.dimension)
        for i in ti.static(range(point_cloud.n)):
            global_coord = ti.Vector([point_cloud[i, d] for d in ti.static(range(self.dimension))])
            local_coord = eigen_vector.transpose() @ (global_coord - center)
            for d in ti.static(range(self.dimension)):
                local_coords[i, d] = local_coord[d]

        box_min, box_max = self._update_patch_aabb(local_coords, 0.)
        return box_min, box_max, eigen_vector
