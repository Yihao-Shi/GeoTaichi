import taichi as ti

from src.utils.linalg import make_list
from src.utils.TypeDefination import vec3f, vec4f


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
        self.resize(n_aabbs)

        if self.first_run:
            self.initial_fields(oriented)

    def resize(self, n_aabbs):
        self.batch_size = make_list(n_aabbs)
        self.n_aabbs = sum(self.batch_size)
        self.n_batches = len(self.batch_size)
        if self.n_batches > 128:
            raise RuntimeError("")

        self.prefix_batch_size = self.batch_size.copy()
        self.prefix_batch_size.insert(0, 0)
        for i in range(1, len(self.prefix_batch_size)):
            self.prefix_batch_size[i] = self.prefix_batch_size[i] + self.prefix_batch_size[i - 1]

    def initial_fields(self, oriented=False):
        @ti.dataclass
        class ti_aabb:
            min: vec3f
            max: vec3f

            @ti.func
            def intersects(self, other) -> bool:
                """
                Check if this AABB intersects with another AABB.
                """
                return (
                    self.min[0] <= other.max[0]
                    and self.max[0] >= other.min[0]
                    and self.min[1] <= other.max[1]
                    and self.max[1] >= other.min[1]
                    and self.min[2] <= other.max[2]
                    and self.max[2] >= other.min[2]
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

    def reset(self, active_aabb):
        self.resize(active_aabb)

    @ti.kernel
    def set_sphere_aabbs(self, particleNum: int, prefix_batch_size: int, verlet_distance: float, particle: ti.template()):
        for np in range(particleNum):
            position = particle[np].x
            radius = particle[np].rad
            self.update_sphere_aabb(np, position, radius, verlet_distance, batch=prefix_batch_size)

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
        return box_min - verlet_distance, box_max + verlet_distance

    @ti.func
    def _update_patch_aabb(self, point_cloud, verlet_distance):
        box_min = ti.Vector.zero(float, self.dimension)
        box_max = ti.Vector.zero(float, self.dimension)
        for i in ti.static(range(point_cloud.n)):
            for d in ti.static(range(self.dimension)):
                box_min[d] = ti.min(box_min[d], point_cloud[i, d])
                box_max[d] = ti.max(box_max[d], point_cloud[i, d])
        return box_min - verlet_distance, box_max + verlet_distance

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

        box_min, box_max = self._update_patch_aabb(local_coords)
        return box_min, box_max, eigen_vector
