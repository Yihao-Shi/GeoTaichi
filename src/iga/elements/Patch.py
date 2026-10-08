import taichi as ti
import numpy as np
from math import prod
from third_party.pyevtk.hl import unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkHexahedron, VtkQuad

import src.iga.config as config
from src.iga.generator.Primitives import Primitives


def _sampled_grid_connectivity(resolution):
    """Build VTK cells for a sampled tensor-product grid.

    pyevtk serializes multidimensional point arrays in Fortran order, so the
    first parametric direction is the fastest-varying point index here too.
    """
    if len(resolution) == 2:
        nu, nv = resolution
        cell_count = max(nu - 1, 0) * max(nv - 1, 0)
        connectivity = np.empty((cell_count, 4), dtype=np.int64)
        cursor = 0
        for j in range(nv - 1):
            for i in range(nu - 1):
                n0 = i + nu * j
                connectivity[cursor] = (n0, n0 + 1, n0 + nu + 1, n0 + nu)
                cursor += 1
        vtk_cell_type = VtkQuad.tid
    elif len(resolution) == 3:
        nu, nv, nw = resolution
        cell_count = max(nu - 1, 0) * max(nv - 1, 0) * max(nw - 1, 0)
        connectivity = np.empty((cell_count, 8), dtype=np.int64)
        cursor = 0
        for k in range(nw - 1):
            for j in range(nv - 1):
                for i in range(nu - 1):
                    n0 = i + nu * (j + nv * k)
                    n1 = n0 + 1
                    n3 = n0 + nu
                    n2 = n3 + 1
                    layer_stride = nu * nv
                    connectivity[cursor] = (
                        n0,
                        n1,
                        n2,
                        n3,
                        n0 + layer_stride,
                        n1 + layer_stride,
                        n2 + layer_stride,
                        n3 + layer_stride,
                    )
                    cursor += 1
        vtk_cell_type = VtkHexahedron.tid
    else:
        raise ValueError("IGA visualization supports only 2D and 3D sampled grids")

    nodes_per_cell = connectivity.shape[1]
    offsets = np.arange(
        nodes_per_cell,
        (cell_count + 1) * nodes_per_cell,
        nodes_per_cell,
        dtype=np.int64,
    )
    cell_types = np.full(cell_count, vtk_cell_type, dtype=np.uint8)
    return (
        np.ascontiguousarray(connectivity.reshape(-1)),
        offsets,
        cell_types,
    )


def _sampled_grid_vtu_data(sampled_points, displacement, stress):
    resolution = sampled_points.shape[:-1]
    dimension = sampled_points.shape[-1]
    if dimension not in (2, 3) or displacement.shape != sampled_points.shape:
        raise ValueError("IGA VTU point and displacement arrays must describe a 2D or 3D grid")
    if stress.shape != resolution:
        raise ValueError("IGA VTU stress array must match the sampled-grid resolution")

    points = [np.ascontiguousarray(sampled_points[..., d].reshape(-1, order="F")) for d in range(dimension)]
    displacement_components = [
        np.ascontiguousarray(displacement[..., d].reshape(-1, order="F")) for d in range(dimension)
    ]
    if dimension == 2:
        points.append(np.zeros(sampled_points[..., 0].size, dtype=sampled_points.dtype))
        displacement_components.append(np.zeros(displacement[..., 0].size, dtype=displacement.dtype))

    connectivity, offsets, cell_types = _sampled_grid_connectivity(resolution)
    point_data = {
        "displacement": tuple(displacement_components),
        "stress": np.ascontiguousarray(stress.reshape(-1, order="F")),
    }
    return points, connectivity, offsets, cell_types, point_data


@ti.data_oriented
class Patch:
    def __init__(self, primitive: Primitives):
        self.current_print = 0
        self.num_patch = 0
        self.total_nonzeros = 0
        self.primitive = primitive

        self.num_knot = np.zeros((primitive.num_primitives + 1, config.DIM), dtype=np.int32)
        self.num_ctrlpts = np.zeros((primitive.num_primitives + 1, config.DIM), dtype=np.int32)
        self.num_element = np.zeros((primitive.num_primitives + 1, config.DIM), dtype=np.int32)

        self.prefix_num_knot = np.zeros((primitive.num_primitives + 1, config.DIM), dtype=np.int32)
        self.prefix_num_element = np.zeros((primitive.num_primitives + 1, config.DIM), dtype=np.int32)
        self.prefix_num_ctrlpts = np.zeros((primitive.num_primitives + 1, config.DIM), dtype=np.int32)

        self.total_num_ctrlpts = np.zeros(primitive.num_primitives + 1, dtype=np.int32)
        self.total_num_element = np.zeros(primitive.num_primitives + 1, dtype=np.int32)
        self.prefix_total_num_ctrlpts = np.zeros(primitive.num_primitives + 1, dtype=np.int32)
        self.prefix_total_num_element = np.zeros(primitive.num_primitives + 1, dtype=np.int32)

        self.nonzero_counting = [[] for _ in range(primitive.num_primitives)]
        self.column_index = [[] for _ in range(primitive.num_primitives)]

        self.volume = ti.field(ti.f64, shape=primitive.num_ctrlpts)
        self.stress = ti.field(ti.f64, shape=primitive.num_ctrlpts)
        self.control_points = ti.Vector.field(config.DIM, ti.f64, shape=primitive.num_ctrlpts)
        self.velocitys = ti.Vector.field(config.DIM, ti.f64, shape=primitive.num_ctrlpts)
        self.accelerations = ti.Vector.field(config.DIM, ti.f64, shape=primitive.num_ctrlpts)
        self.initial_control_points = ti.Vector.field(config.DIM, ti.f64, shape=primitive.num_ctrlpts)
        self.rest_control_points = ti.Vector.field(config.DIM, ti.f64, shape=primitive.num_ctrlpts)
        self.weights = ti.field(ti.f64, shape=primitive.num_ctrlpts)

        self.element_u = ti.Vector.field(2, ti.f64, shape=primitive.num_element[0])
        self.element_v = ti.Vector.field(2, ti.f64, shape=primitive.num_element[1])
        if config.DIM == 3:
            self.element_w = ti.Vector.field(2, ti.f64, shape=primitive.num_element[2])
        self.knot_vector_u = ti.field(ti.f64, shape=primitive.num_knot[0])
        self.knot_vector_v = ti.field(ti.f64, shape=primitive.num_knot[1])
        if config.DIM == 3:
            self.knot_vector_w = ti.field(ti.f64, shape=primitive.num_knot[2])

    def add_patches(self, gravity, rest_shape=None):
        named_rest_shapes = rest_shape if isinstance(rest_shape, dict) else None
        if named_rest_shapes is not None:
            unknown = set(named_rest_shapes) - set(self.primitive.body)
            if unknown:
                raise KeyError(f"IGA rest_shape contains unknown primitives: {sorted(unknown)}")
            flat_rest_shape = None
        elif rest_shape is not None:
            flat_rest_shape = np.asarray(rest_shape, dtype=np.float64)
            expected_shape = (self.primitive.num_ctrlpts, config.DIM)
            if flat_rest_shape.shape != expected_shape:
                raise ValueError(f"IGA rest_shape must have shape {expected_shape}")
            if not np.all(np.isfinite(flat_rest_shape)):
                raise ValueError("IGA rest_shape must be finite")
            flat_rest_shape = np.ascontiguousarray(flat_rest_shape)
        else:
            flat_rest_shape = None

        for name, meta in self.primitive.body.items():
            primitive = meta["primitive"]
            init_v = meta["init_v"]
            assert primitive.dimension == config.DIM

            self.num_element[self.num_patch + 1] = primitive.num_element_list
            self.num_ctrlpts[self.num_patch + 1] = primitive.num_ctrlpts_list
            self.num_knot[self.num_patch + 1] = primitive.num_knot_list
            self.total_num_element[self.num_patch + 1] = primitive.num_element
            self.total_num_ctrlpts[self.num_patch + 1] = primitive.num_ctrlpts

            begin_ctrlpts = np.sum(self.total_num_ctrlpts[: self.num_patch + 1])
            begin_knot = [np.sum(self.num_knot[: self.num_patch + 1, d]) for d in range(primitive.dimension)]
            begin_element = [np.sum(self.num_element[: self.num_patch + 1, d]) for d in range(primitive.dimension)]

            def get_influence_node(index, degree, num_ctrlpts):
                return [i for i in range(index - degree, index + degree + 1) if 0 <= i < num_ctrlpts]

            def linearize_index(indices, num_ctrlpts_list):
                idx = 0
                for d in range(len(indices)):
                    stride = np.prod(num_ctrlpts_list[:d]) if d > 0 else 1
                    idx += indices[d] * stride
                return idx

            for indices in np.ndindex(*[primitive.num_ctrlpts_list[d] for d in range(primitive.dimension)]):
                neighbor_lists = [
                    get_influence_node(indices[d], primitive.degree[d], primitive.num_ctrlpts_list[d])
                    for d in range(primitive.dimension)
                ]
                counts = [len(nl) for nl in neighbor_lists]

                total_neighbors = np.prod(counts)
                self.nonzero_counting[self.num_patch].append(total_neighbors * primitive.dimension)

                meshgrid = np.array(np.meshgrid(*neighbor_lists, indexing="ij")).T.reshape(-1, primitive.dimension)
                for col_indices in meshgrid:
                    for d in range(primitive.dimension):
                        lin_idx = linearize_index(col_indices, primitive.num_ctrlpts_list[: primitive.dimension])
                        self.column_index[self.num_patch].append(lin_idx * primitive.dimension + d)

            primitive_rest_shape = meta.get("rest_shape")
            if named_rest_shapes is not None and name in named_rest_shapes:
                primitive_rest_shape = named_rest_shapes[name]
            elif flat_rest_shape is not None:
                primitive_rest_shape = flat_rest_shape[begin_ctrlpts : begin_ctrlpts + primitive.num_ctrlpts]
            if primitive_rest_shape is None:
                primitive_rest_shape = primitive.control_points
            primitive_rest_shape = np.asarray(primitive_rest_shape, dtype=np.float64)
            expected_shape = (primitive.num_ctrlpts, config.DIM)
            if primitive_rest_shape.shape != expected_shape:
                raise ValueError(f"{name}: rest_shape must have shape {expected_shape}")
            if not np.all(np.isfinite(primitive_rest_shape)):
                raise ValueError(f"{name}: rest_shape must be finite")
            self.fill_ctrlpts_weights(
                begin_ctrlpts,
                primitive.control_points,
                np.ascontiguousarray(primitive_rest_shape),
                primitive.weights,
            )
            self.fill_initial_condition(begin_ctrlpts, primitive.num_ctrlpts, init_v, gravity)
            self.fill_knot_vector(begin_knot[0], primitive.knot_vector_u, self.knot_vector_u)
            self.fill_element(begin_element[0], primitive.element_u, self.element_u)
            self.fill_knot_vector(begin_knot[1], primitive.knot_vector_v, self.knot_vector_v)
            self.fill_element(begin_element[1], primitive.element_v, self.element_v)
            if primitive.dimension == 3:
                self.fill_knot_vector(begin_knot[2], primitive.knot_vector_w, self.knot_vector_w)
                self.fill_element(begin_element[2], primitive.element_w, self.element_w)
            self.num_patch += 1

    def finalize(self):
        for d in range(config.DIM):
            self.prefix_num_knot[:, d] = np.cumsum(self.num_knot[:, d])
            self.prefix_num_element[:, d] = np.cumsum(self.num_element[:, d])
            self.prefix_num_ctrlpts[:, d] = np.cumsum(self.num_ctrlpts[:, d])

        self.prefix_total_num_element = np.cumsum(self.total_num_element)
        self.prefix_total_num_ctrlpts = np.cumsum(self.total_num_ctrlpts)

        for patch_index, patch_nonzero in enumerate(self.nonzero_counting):
            for node_index, nonzero in enumerate(patch_nonzero):
                self.total_nonzeros += nonzero
                self.nonzero_counting[patch_index][node_index] = self.total_nonzeros

    def visualize(self, path, res=None):
        import os

        if not os.path.exists(path):
            os.makedirs(path)

        from src.nurbs.NurbsBasis import NurbsBasisInterpolations2d, NurbsBasisInterpolations3d

        current_ctrlpts = self.control_points.to_numpy()
        current_stress = self.stress.to_numpy()

        for index, (name, meta) in enumerate(self.primitive.body.items()):
            primitive = meta["primitive"]
            assert primitive.dimension == config.DIM
            resolution = [
                int(self.num_ctrlpts[index + 1, d] - primitive.degree[d] + 1) for d in range(primitive.dimension)
            ]

            if isinstance(res, (int, float)):
                resolution = [int(res)] * config.DIM
            elif isinstance(res, (list, tuple, np.ndarray)):
                resolution = [int(res[d]) for d in range(config.DIM)]

            samp_pts = np.zeros((*resolution, config.DIM))
            disp = np.zeros((*resolution, config.DIM))
            von_mises = np.zeros(resolution)

            xi = [np.linspace(0.0, 1.0, resolution[d]) for d in range(config.DIM)]
            for idx in np.ndindex(*resolution):
                coords = [xi[d][idx[d]] for d in range(config.DIM)]
                if primitive.dimension == 2:
                    samp_pts[idx] = NurbsBasisInterpolations2d(
                        *coords,
                        *primitive.degree,
                        primitive.knot_vector_u,
                        primitive.knot_vector_v,
                        current_ctrlpts,
                        primitive.weights,
                    )
                    disp[idx] = NurbsBasisInterpolations2d(
                        *coords,
                        *primitive.degree,
                        primitive.knot_vector_u,
                        primitive.knot_vector_v,
                        current_ctrlpts - primitive.control_points,
                        primitive.weights,
                    )
                    von_mises[idx] = NurbsBasisInterpolations2d(
                        *coords,
                        *primitive.degree,
                        primitive.knot_vector_u,
                        primitive.knot_vector_v,
                        current_stress,
                        primitive.weights,
                    )
                else:
                    samp_pts[idx] = NurbsBasisInterpolations3d(
                        *coords,
                        *primitive.degree,
                        primitive.knot_vector_u,
                        primitive.knot_vector_v,
                        primitive.knot_vector_w,
                        current_ctrlpts,
                        primitive.weights,
                    )
                    disp[idx] = NurbsBasisInterpolations3d(
                        *coords,
                        *primitive.degree,
                        primitive.knot_vector_u,
                        primitive.knot_vector_v,
                        primitive.knot_vector_w,
                        current_ctrlpts - primitive.control_points,
                        primitive.weights,
                    )
                    von_mises[idx] = NurbsBasisInterpolations3d(
                        *coords,
                        *primitive.degree,
                        primitive.knot_vector_u,
                        primitive.knot_vector_v,
                        primitive.knot_vector_w,
                        current_stress,
                        primitive.weights,
                    )

            # Emit one representation per frame; the sampled VTU keeps the
            # same tensor-product cells, positions, displacement and stress.
            vtu_points, connectivity, offsets, cell_types, vtu_point_data = _sampled_grid_vtu_data(
                samp_pts, disp, von_mises
            )
            unstructuredGridToVTK(
                os.path.join(path, f"NurbsVolume{name}{self.current_print:06d}"),
                *vtu_points,
                connectivity=connectivity,
                offsets=offsets,
                cell_types=cell_types,
                pointData=vtu_point_data,
            )
        self.current_print += 1

    @ti.kernel
    def fill_ctrlpts_weights(
        self,
        begin_index: ti.i32,
        control_points: ti.types.ndarray(),
        rest_control_points: ti.types.ndarray(),
        weights: ti.types.ndarray(),
    ):
        for i in range(control_points.shape[0]):
            self.initial_control_points[begin_index + i] = ti.Vector(
                [control_points[i, d] for d in ti.static(range(config.DIM))]
            )
            self.control_points[begin_index + i] = ti.Vector(
                [control_points[i, d] for d in ti.static(range(config.DIM))]
            )
            self.rest_control_points[begin_index + i] = ti.Vector(
                [rest_control_points[i, d] for d in ti.static(range(config.DIM))]
            )
            self.weights[begin_index + i] = weights[i]

    @ti.kernel
    def fill_initial_condition(
        self,
        begin_index: ti.i32,
        total_num_ctrlpts: ti.i32,
        init_v: ti.types.vector(config.DIM, ti.f64),
        init_a: ti.types.vector(config.DIM, ti.f64),
    ):
        for i in range(total_num_ctrlpts):
            self.velocitys[begin_index + i] = init_v
            self.accelerations[begin_index + i] = init_a

    @ti.kernel
    def fill_knot_vector(
        self, begin_index: ti.i32, knot_vector_src: ti.types.ndarray(), knot_vector_dst: ti.template()
    ):
        for i in range(knot_vector_src.shape[0]):
            knot_vector_dst[begin_index + i] = knot_vector_src[i]

    @ti.kernel
    def fill_element(self, begin_index: ti.i32, element_src: ti.types.ndarray(), element_dst: ti.template()):
        for i in range(element_src.shape[0]):
            element_dst[begin_index + i] = ti.Vector([element_src[i, 0], element_src[i, 1]])
