from dataclasses import dataclass

import numpy as np


HEXAHEDRON = 0
TETRAHEDRON = 1


@dataclass(frozen=True)
class SoftTemplateSupport:
    grid_type: int
    point_node: np.ndarray
    point_shape: np.ndarray
    point_dshape: np.ndarray
    point_count: np.ndarray
    surface_node: np.ndarray
    surface_shape: np.ndarray
    surface_count: np.ndarray
    sdf_node: np.ndarray
    sdf_shape: np.ndarray
    sdf_count: np.ndarray
    grid_origin: np.ndarray
    grid_shape: np.ndarray
    grid_space: float
    grid_base_space: float = 0.0

    @property
    def material_point_number(self):
        return int(self.point_count.shape[0])

    @property
    def surface_node_number(self):
        return int(self.surface_count.shape[0])

    @property
    def levelset_node_number(self):
        return int(self.sdf_count.shape[0])


def normalize_soft_grid_type(grid_type):
    key = str(grid_type).replace("_", "").replace("-", "").replace(" ", "").lower()
    aliases = {
        "hex": ("Hexahedron", HEXAHEDRON),
        "hexahedron": ("Hexahedron", HEXAHEDRON),
        "hexahedral": ("Hexahedron", HEXAHEDRON),
        "cartesian": ("Hexahedron", HEXAHEDRON),
        "tet": ("Tetrahedron", TETRAHEDRON),
        "tetrahedron": ("Tetrahedron", TETRAHEDRON),
        "tetrahedral": ("Tetrahedron", TETRAHEDRON),
        "simplex": ("Tetrahedron", TETRAHEDRON),
        # The LSMPM soft-particle implementation is currently three-dimensional.
        "triangle": ("Tetrahedron", TETRAHEDRON),
        "triangular": ("Tetrahedron", TETRAHEDRON),
    }
    if key not in aliases:
        raise RuntimeError("LSMPM soft_grid_type must be one of " "['Hexahedron', 'Tetrahedron']")
    return aliases[key]


def _boundary_type(grid_id, grid_num):
    boundary_type = min(2, grid_id) - min(grid_num - 1 - grid_id, 2)
    if boundary_type < 0:
        boundary_type += 3
    elif boundary_type > 0:
        boundary_type += 2
    return boundary_type


def _quadratic_bspline(xp, xg, inv_dx, boundary_type):
    value = 0.0
    gradient = 0.0
    d = (xp - xg) * inv_dx
    if boundary_type == 0:
        if 0.5 <= d < 1.5:
            value = (0.5 * d - 1.5) * d + 1.125
            gradient = d - 1.5
        elif -0.5 <= d < 0.5:
            value = -d * d + 0.75
            gradient = -2.0 * d
        elif -1.5 <= d < 0.5:
            value = (0.5 * d + 1.5) * d + 1.125
            gradient = d + 1.5
    elif boundary_type == 1:
        if 0.0 <= d < 0.5:
            value = 1.0 - d
            gradient = -1.0
        elif 0.5 <= d < 1.5:
            value = (0.5 * d - 1.5) * d + 1.125
            gradient = d - 1.5
    elif boundary_type == 2:
        if -1.0 <= d < -0.5:
            value = 1.0 + d
            gradient = 1.0
        elif -0.5 <= d < 0.5:
            value = -d * d + 0.75
            gradient = -2.0 * d
        elif 0.5 <= d < 1.5:
            value = (0.5 * d - 1.5) * d + 1.125
            gradient = d - 1.5
    elif boundary_type == 3:
        if -1.5 <= d < -0.5:
            value = (0.5 * d + 1.5) * d + 1.125
            gradient = d + 1.5
        elif -0.5 <= d < 0.5:
            value = -d * d + 0.75
            gradient = -2.0 * d
        elif 0.5 <= d < 1.0:
            value = 1.0 - d
            gradient = -1.0
    elif boundary_type == 4:
        if -1.5 <= d < -0.5:
            value = (0.5 * d + 1.5) * d + 1.125
            gradient = d + 1.5
        elif -0.5 <= d <= 0.0:
            value = 1.0 + d
            gradient = 1.0
    return value, gradient * inv_dx


def _cubic_bspline(xp, xg, inv_dx, boundary_type):
    value = 0.0
    gradient = 0.0
    d = (xp - xg) * inv_dx
    if 1.0 <= d < 2.0:
        if boundary_type != 3:
            value = ((-d / 6.0 + 1.0) * d - 2.0) * d + 4.0 / 3.0
            gradient = (-0.5 * d + 2.0) * d - 2.0
    elif 0.0 <= d < 1.0:
        if boundary_type == 1:
            value = (d * d / 6.0 - 1.0) * d + 1.0
            gradient = 0.5 * d * d - 1.0
        elif boundary_type == 3:
            value = (d / 3.0 - 1.0) * d * d + 2.0 / 3.0
            gradient = d * (d - 2.0)
        else:
            value = (0.5 * d - 1.0) * d * d + 2.0 / 3.0
            gradient = (1.5 * d - 2.0) * d
    elif -1.0 <= d < 0.0:
        if boundary_type == 4:
            value = (-d * d / 6.0 + 1.0) * d + 1.0
            gradient = -0.5 * d * d + 1.0
        elif boundary_type == 2:
            value = (-d / 3.0 - 1.0) * d * d + 2.0 / 3.0
            gradient = (-d - 2.0) * d
        else:
            value = (-0.5 * d - 1.0) * d * d + 2.0 / 3.0
            gradient = (-1.5 * d - 2.0) * d
    elif -2.0 <= d < -1.0:
        value = ((d / 6.0 + 1.0) * d + 2.0) * d + 4.0 / 3.0
        gradient = (0.5 * d + 2.0) * d + 2.0
    return value, gradient * inv_dx


def _compact_node_id(ijk, topology):
    local = np.asarray(ijk, dtype=np.int64) - topology.compact_origin
    if np.any(local < 0) or np.any(local >= topology.compact_shape):
        return -1
    return int(
        local[0]
        + local[1] * topology.compact_shape[0]
        + local[2] * topology.compact_shape[0] * topology.compact_shape[1]
    )


def _linear_support(point, xmin, dx, gnum, topology):
    base = np.floor((point - xmin) / dx).astype(np.int64)
    base = np.minimum(gnum - 2, np.maximum(0, base))
    origin = xmin + base * dx
    fraction = (point - origin) / dx
    entries = []
    for i in range(2):
        wx = 1.0 - fraction[0] if i == 0 else fraction[0]
        dwx = -1.0 / dx if i == 0 else 1.0 / dx
        for j in range(2):
            wy = 1.0 - fraction[1] if j == 0 else fraction[1]
            dwy = -1.0 / dx if j == 0 else 1.0 / dx
            for k in range(2):
                wz = 1.0 - fraction[2] if k == 0 else fraction[2]
                dwz = -1.0 / dx if k == 0 else 1.0 / dx
                node = _compact_node_id(base + (i, j, k), topology)
                if node >= 0:
                    entries.append(
                        (
                            node,
                            wx * wy * wz,
                            np.array(
                                (
                                    dwx * wy * wz,
                                    wx * dwy * wz,
                                    wx * wy * dwz,
                                ),
                                dtype=np.float64,
                            ),
                        )
                    )
    return entries


def _point_inside_grid(point, xmin, dx, gnum, tolerance=1.0e-10):
    upper = xmin + (gnum - 1) * dx
    scale = max(float(dx), 1.0)
    eps = tolerance * scale
    return bool(np.all(point >= xmin - eps) and np.all(point <= upper + eps))


def _linear_trace_support(point, xmin, dx, gnum, topology):
    if not _point_inside_grid(point, xmin, dx, gnum):
        return []
    return _linear_support(point, xmin, dx, gnum, topology)


def _bspline_support(point, xmin, dx, gnum, topology, shape_function_type):
    inv_dx = 1.0 / dx
    particle_size = 0.5 * dx if shape_function_type == 1 else dx
    influenced_node = 3 if shape_function_type == 1 else 4
    base = np.floor((point - xmin - particle_size) * inv_dx).astype(np.int64)
    supports = []
    for axis in range(3):
        axis_support = []
        for offset in range(influenced_node):
            node = int(base[axis] + offset)
            if node < 0 or node >= int(gnum[axis]):
                continue
            boundary_type = _boundary_type(node, int(gnum[axis]))
            node_position = xmin[axis] + node * dx
            if shape_function_type == 1:
                value, gradient = _quadratic_bspline(point[axis], node_position, inv_dx, boundary_type)
            else:
                value, gradient = _cubic_bspline(point[axis], node_position, inv_dx, boundary_type)
            axis_support.append((node, value, gradient))
        supports.append(axis_support)

    entries = []
    for i, sx, dsx in supports[0]:
        for j, sy, dsy in supports[1]:
            for k, sz, dsz in supports[2]:
                value = sx * sy * sz
                if value <= 1.0e-12:
                    continue
                node = _compact_node_id((i, j, k), topology)
                if node >= 0:
                    entries.append(
                        (
                            node,
                            value,
                            np.array(
                                (dsx * sy * sz, sx * dsy * sz, sx * sy * dsz),
                                dtype=np.float64,
                            ),
                        )
                    )
    return entries


def _pack_support(entries, max_nodes, include_gradient):
    item_count = len(entries)
    node = np.full((item_count, max_nodes), -1, dtype=np.int32)
    shape = np.zeros((item_count, max_nodes), dtype=np.float64)
    dshape = np.zeros((item_count, max_nodes, 3), dtype=np.float64) if include_gradient else None
    count = np.zeros(item_count, dtype=np.int32)
    for item, item_entries in enumerate(entries):
        if len(item_entries) > max_nodes:
            raise RuntimeError(
                f"Soft template support requires {len(item_entries)} nodes, " f"but only {max_nodes} were allocated"
            )
        count[item] = len(item_entries)
        for local, entry in enumerate(item_entries):
            node[item, local] = entry[0]
            shape[item, local] = entry[1]
            if include_gradient:
                dshape[item, local] = entry[2]
    return node, shape, dshape, count


def _pack_support_stream(
    points,
    item_count,
    support_builder,
    max_nodes,
    include_gradient,
):
    """Pack support rows without retaining Python objects for every point.

    Fine LSMPM templates can contain hundreds of thousands of material
    points.  Keeping each point's support as a list of tuples (including one
    small NumPy gradient array per node) can consume tens of gigabytes before
    the compact arrays are allocated.  Build one row at a time so peak host
    memory is governed by the returned arrays instead.
    """
    item_count = int(item_count)
    node = np.full((item_count, max_nodes), -1, dtype=np.int32)
    shape = np.zeros((item_count, max_nodes), dtype=np.float64)
    dshape = np.zeros((item_count, max_nodes, 3), dtype=np.float64) if include_gradient else None
    count = np.zeros(item_count, dtype=np.int32)
    seen = 0
    for item, point in enumerate(points):
        if item >= item_count:
            raise RuntimeError(
                "Soft template support iterator contains more points than " f"the declared count ({item_count})"
            )
        item_entries = support_builder(point)
        if len(item_entries) > max_nodes:
            raise RuntimeError(
                f"Soft template support requires {len(item_entries)} nodes, " f"but only {max_nodes} were allocated"
            )
        count[item] = len(item_entries)
        for local, entry in enumerate(item_entries):
            node[item, local] = entry[0]
            shape[item, local] = entry[1]
            if include_gradient:
                dshape[item, local] = entry[2]
        seen = item + 1
    if seen != item_count:
        raise RuntimeError(
            "Soft template support iterator contains fewer points than " f"the declared count ({seen} != {item_count})"
        )
    return node, shape, dshape, count


def build_hexahedral_template_support(
    template,
    material_points,
    topology,
    shape_function_type,
    mechanical_grid,
):
    xmin = np.asarray(mechanical_grid.minBox(), dtype=np.float64)
    dx = float(mechanical_grid.spacing)
    gnum = np.asarray(mechanical_grid.gnum, dtype=np.int64)
    material_points = np.asarray(material_points, dtype=np.float64)
    if shape_function_type == 0:

        def point_support_builder(point):
            return _linear_support(point, xmin, dx, gnum, topology)

    else:

        def point_support_builder(point):
            return _bspline_support(
                point,
                xmin,
                dx,
                gnum,
                topology,
                shape_function_type,
            )

    max_point_nodes = (2, 3, 4)[shape_function_type] ** 3
    point_node, point_shape, point_dshape, point_count = _pack_support_stream(
        material_points,
        material_points.shape[0],
        point_support_builder,
        max_point_nodes,
        True,
    )

    surface_points = np.asarray(template.objects.mesh.vertices, dtype=np.float64)

    def trace_support_builder(point):
        return _linear_trace_support(point, xmin, dx, gnum, topology)

    surface_node, surface_shape, _, surface_count = _pack_support_stream(
        surface_points,
        surface_points.shape[0],
        trace_support_builder,
        8,
        False,
    )

    sdf_grid = template.objects.grid
    sdf_xmin = np.asarray(sdf_grid.minBox(), dtype=np.float64)
    sdf_dx = float(sdf_grid.grid_space)
    sdf_gnum = np.asarray(sdf_grid.gnum, dtype=np.int64)
    sdf_grid_sum = int(sdf_grid.gridSum)

    def sdf_points():
        for logical in range(sdf_grid_sum):
            i = logical % sdf_gnum[0]
            j = (logical % (sdf_gnum[0] * sdf_gnum[1])) // sdf_gnum[0]
            k = logical // (sdf_gnum[0] * sdf_gnum[1])
            yield sdf_xmin + np.array((i, j, k), dtype=np.float64) * sdf_dx

    sdf_node, sdf_shape, _, sdf_count = _pack_support_stream(
        sdf_points(),
        sdf_grid_sum,
        trace_support_builder,
        8,
        False,
    )
    return SoftTemplateSupport(
        grid_type=HEXAHEDRON,
        point_node=np.ascontiguousarray(point_node),
        point_shape=np.ascontiguousarray(point_shape),
        point_dshape=np.ascontiguousarray(point_dshape),
        point_count=np.ascontiguousarray(point_count),
        surface_node=np.ascontiguousarray(surface_node),
        surface_shape=np.ascontiguousarray(surface_shape),
        surface_count=np.ascontiguousarray(surface_count),
        sdf_node=np.ascontiguousarray(sdf_node),
        sdf_shape=np.ascontiguousarray(sdf_shape),
        sdf_count=np.ascontiguousarray(sdf_count),
        grid_origin=np.ascontiguousarray(xmin),
        grid_shape=np.ascontiguousarray(gnum, dtype=np.int32),
        grid_space=dx,
    )


_CUBE_VERTEX_OFFSETS = np.asarray(
    (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ),
    dtype=np.int64,
)

# Kuhn triangulation about the 000--111 body diagonal.  The six tetrahedra
# have equal positive volume and tile the hexahedral cell without gaps.
_CELL_TETRAHEDRA = np.asarray(
    (
        (0, 1, 3, 7),
        (0, 1, 5, 7),
        (0, 2, 3, 7),
        (0, 2, 6, 7),
        (0, 4, 5, 7),
        (0, 4, 6, 7),
    ),
    dtype=np.int64,
)


def _logical_node_id(ijk, gnum):
    return int(ijk[0] + ijk[1] * gnum[0] + ijk[2] * gnum[0] * gnum[1])


def _logical_node_ijk(node, gnum):
    return np.asarray(
        (
            node % gnum[0],
            (node % (gnum[0] * gnum[1])) // gnum[0],
            node // (gnum[0] * gnum[1]),
        ),
        dtype=np.int64,
    )


def _tetra_barycentric(point, vertices):
    jacobian = np.column_stack(
        (
            vertices[1] - vertices[0],
            vertices[2] - vertices[0],
            vertices[3] - vertices[0],
        )
    )
    inverse = np.linalg.inv(jacobian)
    tail = inverse @ (point - vertices[0])
    barycentric = np.asarray(
        (1.0 - np.sum(tail), tail[0], tail[1], tail[2]),
        dtype=np.float64,
    )
    gradients = np.empty((4, 3), dtype=np.float64)
    gradients[1:] = inverse
    gradients[0] = -np.sum(gradients[1:], axis=0)
    return barycentric, gradients


def _rectilinear_tetra_support(points, axes, gnum):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    base = np.empty_like(points, dtype=np.int64)
    fraction = np.empty_like(points)
    valid = np.ones(points.shape[0], dtype=bool)
    scale = max(
        1.0,
        *(float(np.max(np.abs(axis))) for axis in axes),
    )
    tolerance = 1.0e-10 * scale
    for d, axis in enumerate(axes):
        valid &= (points[:, d] >= axis[0] - tolerance) & (points[:, d] <= axis[-1] + tolerance)
        base[:, d] = np.searchsorted(axis, points[:, d], side="right") - 1
        base[:, d] = np.clip(base[:, d], 0, axis.size - 2)
        lower = axis[base[:, d]]
        width = axis[base[:, d] + 1] - lower
        fraction[:, d] = np.clip((points[:, d] - lower) / width, 0.0, 1.0)

    order = np.argsort(-fraction, axis=1, kind="stable")
    identity = np.eye(3, dtype=np.int64)
    first = identity[order[:, 0]]
    second = first + identity[order[:, 1]]
    offsets = np.stack(
        (
            np.zeros_like(first),
            first,
            second,
            np.ones_like(first),
        ),
        axis=1,
    )
    node_ijk = base[:, None, :] + offsets
    logical_nodes = (node_ijk[:, :, 0] + node_ijk[:, :, 1] * gnum[0] + node_ijk[:, :, 2] * gnum[0] * gnum[1]).astype(
        np.int64
    )

    sorted_fraction = np.take_along_axis(fraction, order, axis=1)
    barycentric = np.column_stack(
        (
            1.0 - sorted_fraction[:, 0],
            sorted_fraction[:, 0] - sorted_fraction[:, 1],
            sorted_fraction[:, 1] - sorted_fraction[:, 2],
            sorted_fraction[:, 2],
        )
    )
    return logical_nodes, barycentric, valid


def _compact_node_ids(logical_nodes, gnum, topology):
    logical_nodes = np.asarray(logical_nodes, dtype=np.int64)
    i = logical_nodes % gnum[0]
    j = (logical_nodes % (gnum[0] * gnum[1])) // gnum[0]
    k = logical_nodes // (gnum[0] * gnum[1])
    local_i = i - int(topology.compact_origin[0])
    local_j = j - int(topology.compact_origin[1])
    local_k = k - int(topology.compact_origin[2])
    valid = (
        (local_i >= 0)
        & (local_i < int(topology.compact_shape[0]))
        & (local_j >= 0)
        & (local_j < int(topology.compact_shape[1]))
        & (local_k >= 0)
        & (local_k < int(topology.compact_shape[2]))
    )
    compact = (
        local_i
        + local_j * int(topology.compact_shape[0])
        + local_k * int(topology.compact_shape[0]) * int(topology.compact_shape[1])
    )
    return compact.astype(np.int32), valid


def _rectilinear_trace_arrays(points, axes, gnum, topology):
    logical, weights, inside = _rectilinear_tetra_support(points, axes, gnum)
    compact, retained = _compact_node_ids(logical, gnum, topology)
    supported = inside & np.all(retained, axis=1)
    count = np.where(supported, 4, 0).astype(np.int32)
    node = np.full((logical.shape[0], 8), -1, dtype=np.int32)
    shape = np.zeros((logical.shape[0], 8), dtype=np.float64)
    node[supported, :4] = compact[supported]
    shape[supported, :4] = np.maximum(weights[supported], 0.0)
    return node, shape, count


def build_tetrahedral_template_support(
    template,
    mechanical_grid,
    storage,
    padding_cells,
    *,
    levelset_extent_cells,
    verlet_padding_cells,
    levelset_verlet_padding_cells=0,
    reference_volume=None,
):
    from src.mpm.soft_particle.GridTopology import SoftGridTopology

    sdf_grid = template.objects.grid
    sdf_gnum = np.asarray(sdf_grid.gnum, dtype=np.int64)
    axes = tuple(np.asarray(axis, dtype=np.float64) for axis in mechanical_grid.coordinate_axes)
    xmin = np.asarray([axis[0] for axis in axes], dtype=np.float64)
    gnum = np.asarray(mechanical_grid.gnum, dtype=np.int64)

    point_chunks = []
    logical_chunks = []
    gradient_chunks = []
    volume_chunks = []
    cell_i, cell_j = np.meshgrid(
        np.arange(gnum[0] - 1, dtype=np.int64),
        np.arange(gnum[1] - 1, dtype=np.int64),
        indexing="ij",
    )
    cell_i = cell_i.reshape(-1)
    cell_j = cell_j.reshape(-1)
    width_x = np.diff(axes[0])[cell_i]
    width_y = np.diff(axes[1])[cell_j]
    lower_x = axes[0][cell_i]
    lower_y = axes[1][cell_j]
    for cell_k in range(int(gnum[2] - 1)):
        width_z = float(axes[2][cell_k + 1] - axes[2][cell_k])
        lower_z = float(axes[2][cell_k])
        widths = np.column_stack(
            (
                width_x,
                width_y,
                np.full(cell_i.size, width_z, dtype=np.float64),
            )
        )
        lower = np.column_stack(
            (
                lower_x,
                lower_y,
                np.full(cell_i.size, lower_z, dtype=np.float64),
            )
        )
        base_logical = cell_i + cell_j * gnum[0] + cell_k * gnum[0] * gnum[1]
        for tetrahedron in _CELL_TETRAHEDRA:
            offsets = _CUBE_VERTEX_OFFSETS[tetrahedron]
            center_fraction = np.mean(offsets, axis=0)
            centers = lower + widths * center_fraction
            inside = np.asarray(template.objects(centers)).reshape(-1) <= 0.0
            if not np.any(inside):
                continue
            point_chunks.append(centers[inside])
            logical_offset = offsets[:, 0] + offsets[:, 1] * gnum[0] + offsets[:, 2] * gnum[0] * gnum[1]
            logical_chunks.append(base_logical[inside, None] + logical_offset[None, :])
            _, reference_gradient = _tetra_barycentric(center_fraction, offsets.astype(np.float64))
            gradient_chunks.append(reference_gradient[None, :, :] / widths[inside, None, :])
            volume_chunks.append(np.prod(widths[inside], axis=1) / 6.0)
    if not point_chunks:
        raise RuntimeError("Tetrahedral soft-grid preprocessing found no interior Gauss point")
    material_points = np.ascontiguousarray(np.concatenate(point_chunks, axis=0), dtype=np.float64)
    point_logical = np.ascontiguousarray(np.concatenate(logical_chunks, axis=0), dtype=np.int64)
    point_dshape = np.ascontiguousarray(np.concatenate(gradient_chunks, axis=0), dtype=np.float64)
    point_volume = np.ascontiguousarray(np.concatenate(volume_chunks), dtype=np.float64)
    if reference_volume is None:
        reference_volume = float(template.objects.volume)
    reference_volume = float(reference_volume)
    if not np.isfinite(reference_volume) or reference_volume <= 0.0:
        raise ValueError("Soft-particle ReferenceVolume must be positive")
    point_volume *= reference_volume / float(np.sum(point_volume))

    surface_points = np.asarray(template.objects.mesh.vertices, dtype=np.float64)
    surface_logical, _, surface_inside = _rectilinear_tetra_support(surface_points, axes, gnum)
    if not np.all(surface_inside):
        raise RuntimeError("The tetrahedral mechanical grid does not contain all surface nodes")
    active_ids = np.unique(np.concatenate((point_logical.reshape(-1), surface_logical.reshape(-1))))
    active_ijk = np.asarray(
        [_logical_node_ijk(node, gnum) for node in active_ids],
        dtype=np.int64,
    )
    requested_dense = str(storage).strip().lower() == "dense"
    if requested_dense:
        compact_origin = np.zeros(3, dtype=np.int32)
        compact_upper = gnum.astype(np.int32)
    else:
        compact_origin = np.maximum(0, active_ijk.min(axis=0) - int(padding_cells)).astype(np.int32)
        compact_upper = np.minimum(gnum, active_ijk.max(axis=0) + 1 + int(padding_cells)).astype(np.int32)
    compact_shape = compact_upper - compact_origin
    topology = SoftGridTopology(
        compact_origin=np.ascontiguousarray(compact_origin),
        compact_shape=np.ascontiguousarray(compact_shape),
        logical_count=int(np.prod(gnum, dtype=np.int64)),
        compact_count=int(np.prod(compact_shape, dtype=np.int64)),
        support_count=int(active_ids.size),
        padding_cells=int(padding_cells),
        storage="Dense" if requested_dense else "Compact",
        levelset_extent_cells=int(levelset_extent_cells),
        verlet_padding_cells=int(verlet_padding_cells),
        levelset_verlet_padding_cells=int(levelset_verlet_padding_cells),
    )

    point_node, retained = _compact_node_ids(point_logical, gnum, topology)
    if not np.all(retained):
        raise RuntimeError("A tetrahedral material-point support node was removed from the " "compact grid")
    point_shape = np.full(point_node.shape, 0.25, dtype=np.float64)
    point_count = np.full(point_node.shape[0], 4, dtype=np.int32)

    surface_node, surface_shape, surface_count = _rectilinear_trace_arrays(surface_points, axes, gnum, topology)
    if not np.all(surface_count == 4):
        raise RuntimeError("A tetrahedral surface trace lies outside the compact grid")

    sdf_xmin = np.asarray(sdf_grid.minBox(), dtype=np.float64)
    sdf_dx = float(sdf_grid.grid_space)
    sdf_node = np.full((int(sdf_grid.gridSum), 8), -1, dtype=np.int32)
    sdf_shape = np.zeros((int(sdf_grid.gridSum), 8), dtype=np.float64)
    sdf_count = np.zeros(int(sdf_grid.gridSum), dtype=np.int32)
    chunk_size = 250_000
    for start in range(0, int(sdf_grid.gridSum), chunk_size):
        stop = min(start + chunk_size, int(sdf_grid.gridSum))
        logical = np.arange(start, stop, dtype=np.int64)
        i = logical % sdf_gnum[0]
        j = (logical % (sdf_gnum[0] * sdf_gnum[1])) // sdf_gnum[0]
        k = logical // (sdf_gnum[0] * sdf_gnum[1])
        points = sdf_xmin + np.column_stack((i, j, k)) * sdf_dx
        node, shape, count = _rectilinear_trace_arrays(points, axes, gnum, topology)
        sdf_node[start:stop] = node
        sdf_shape[start:stop] = shape
        sdf_count[start:stop] = count
    minimum_spacing = float(mechanical_grid.minimum_spacing)
    support = SoftTemplateSupport(
        grid_type=TETRAHEDRON,
        point_node=np.ascontiguousarray(point_node),
        point_shape=np.ascontiguousarray(point_shape),
        point_dshape=np.ascontiguousarray(point_dshape),
        point_count=np.ascontiguousarray(point_count),
        surface_node=np.ascontiguousarray(surface_node),
        surface_shape=np.ascontiguousarray(surface_shape),
        surface_count=np.ascontiguousarray(surface_count),
        sdf_node=np.ascontiguousarray(sdf_node),
        sdf_shape=np.ascontiguousarray(sdf_shape),
        sdf_count=np.ascontiguousarray(sdf_count),
        grid_origin=np.ascontiguousarray(xmin),
        grid_shape=np.ascontiguousarray(gnum, dtype=np.int32),
        grid_space=minimum_spacing,
        grid_base_space=float(mechanical_grid.spacing),
    )
    return material_points, point_volume, topology, support
