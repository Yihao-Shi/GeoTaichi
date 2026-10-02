import taichi as ti
import numpy as np

from src.nurbs.core.NurbsGeometry import NurbsBasisFunction2d


@ti.data_oriented
class Surface:
    def __init__(self, surface):
        self.surface = surface
        self.num_surfaces = 0
        self.num_knot_u = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.num_knot_v = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.num_ctrlpts = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.num_ctrlpts_u = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.num_ctrlpts_v = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.num_elements = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.num_element_u = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.num_element_v = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.prefix_num_knot_u = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.prefix_num_knot_v = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.prefix_num_ctrlpts = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.prefix_num_ctrlpts_u = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.prefix_num_ctrlpts_v = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.prefix_num_elements = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.prefix_num_element_u = np.zeros(surface.num_surfaces + 1, dtype=np.int32)
        self.prefix_num_element_v = np.zeros(surface.num_surfaces + 1, dtype=np.int32)

        self.control_points = ti.Vector.field(3, ti.f64, shape=surface.num_ctrlpts)
        self.control_points_hat = ti.Vector.field(3, ti.f64, shape=surface.num_ctrlpts)
        self.control_points_id = ti.field(int, shape=surface.num_ctrlpts)
        self.weights = ti.field(ti.f64, shape=surface.num_ctrlpts)
        self.knot_vector_u = ti.field(ti.f64, shape=surface.num_knot_u)
        self.knot_vector_v = ti.field(ti.f64, shape=surface.num_knot_v)
        self.area = ti.field(ti.f64, shape=surface.num_elements)
        self.bounding_box = ti.Vector.field(3, ti.f64, shape=(surface.num_knot_v, 2))
        self.basis = []

    def add_surfaces(self):
        for primitive in self.surface.surfaces:
            self.num_knot_u[self.num_surfaces + 1] = primitive.num_knot_u
            self.num_knot_v[self.num_surfaces + 1] = primitive.num_knot_v
            self.num_ctrlpts[self.num_surfaces + 1] = primitive.num_ctrlpts_u * primitive.num_ctrlpts_v
            self.num_ctrlpts_u[self.num_surfaces + 1] = primitive.num_ctrlpts_u
            self.num_ctrlpts_v[self.num_surfaces + 1] = primitive.num_ctrlpts_v
            self.num_elements[self.num_surfaces + 1] = primitive.num_element_u * primitive.num_element_v
            self.num_element_u[self.num_surfaces + 1] = primitive.num_element_u
            self.num_element_v[self.num_surfaces + 1] = primitive.num_element_v

            begin_ctrlpts, begin_knot_u, begin_knot_v = 0, 0, 0
            for i in range(self.num_surfaces + 1):
                begin_ctrlpts += int(self.num_ctrlpts_u[i]) * int(self.num_ctrlpts_v[i])
                begin_knot_u += int(self.num_knot_u[i])
                begin_knot_v += int(self.num_knot_v[i])

            self.fill_ctrlpts_weights(
                begin_ctrlpts, primitive.control_points, primitive.weights, primitive.parent_indices, primitive.area
            )
            self.fill_knot_vector(begin_knot_u, primitive.knot_vector_u, self.knot_vector_u)
            self.fill_knot_vector(begin_knot_v, primitive.knot_vector_v, self.knot_vector_v)

            self.basis.append(NurbsBasisFunction2d(primitive.degree_u, primitive.degree_v))
            self.num_surfaces += 1

    def finalize(self):
        self.prefix_num_knot_u = np.cumsum(self.num_knot_u)
        self.prefix_num_knot_v = np.cumsum(self.num_knot_v)
        self.prefix_num_ctrlpts = np.cumsum(self.num_ctrlpts)
        self.prefix_num_ctrlpts_u = np.cumsum(self.num_ctrlpts_u)
        self.prefix_num_ctrlpts_v = np.cumsum(self.num_ctrlpts_v)
        self.prefix_num_elements = np.cumsum(self.num_elements)
        self.prefix_num_element_u = np.cumsum(self.num_element_u)
        self.prefix_num_element_v = np.cumsum(self.num_element_v)

    @ti.kernel
    def fill_ctrlpts_weights(
        self,
        begin_index: int,
        control_points: ti.types.ndarray(),
        weights: ti.types.ndarray(),
        ctrlpt_ids: ti.types.ndarray(),
        area: ti.types.ndarray(),
    ):
        for i in range(control_points.shape[0]):
            self.control_points[begin_index + i] = ti.Vector(
                [control_points[i, 0], control_points[i, 1], control_points[i, 2]]
            )
            self.control_points_hat[begin_index + i] = ti.Vector(
                [control_points[i, 0], control_points[i, 1], control_points[i, 2]]
            )
            self.control_points_id[begin_index + i] = ctrlpt_ids[i]
            self.weights[begin_index + i] = weights[i]
        for i in range(area.shape[0]):
            self.area[begin_index + i] = area[i]

    @ti.kernel
    def fill_knot_vector(self, begin_index: int, knot_vector_src: ti.types.ndarray(), knot_vector_dst: ti.template()):
        for i in range(knot_vector_src.shape[0]):
            knot_vector_dst[begin_index + i] = knot_vector_src[i]
