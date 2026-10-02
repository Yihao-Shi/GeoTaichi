from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SoftGridTopology:
    compact_origin: np.ndarray
    compact_shape: np.ndarray
    logical_count: int
    compact_count: int
    support_count: int
    padding_cells: int
    storage: str
    levelset_extent_cells: int = 0
    verlet_padding_cells: int = 0
    levelset_verlet_padding_cells: int = 0


@dataclass(frozen=True)
class SoftMechanicalGrid:
    origin: np.ndarray
    shape: np.ndarray
    spacing: float
    axes: tuple = None

    @property
    def gnum(self):
        return self.shape

    @property
    def gridSum(self):
        return int(np.prod(self.shape, dtype=np.int64))

    def minBox(self):
        if self.axes is None:
            return self.origin
        return np.asarray([axis[0] for axis in self.axes], dtype=np.float64)

    def maxBox(self):
        if self.axes is None:
            return self.origin + (self.shape - 1) * self.spacing
        return np.asarray([axis[-1] for axis in self.axes], dtype=np.float64)

    @property
    def coordinate_axes(self):
        if self.axes is not None:
            return self.axes
        return tuple(self.origin[d] + np.arange(int(self.shape[d]), dtype=np.float64) * self.spacing for d in range(3))

    @property
    def minimum_spacing(self):
        if self.axes is None:
            return float(self.spacing)
        return min(float(np.min(np.diff(axis))) for axis in self.axes)

    @property
    def is_uniform(self):
        return self.axes is None


def normalize_soft_grid_refinement(refinement, coarse_spacing):
    if refinement is None:
        return None
    if not isinstance(refinement, dict):
        raise TypeError("MechanicalGridRefinement must be a dictionary")

    def get(*keys):
        for key in keys:
            if key in refinement:
                return refinement[key]
        return None

    region_min = get("RegionMin", "region_min", "Min", "min")
    region_max = get("RegionMax", "region_max", "Max", "max")
    fine_spacing = get("FineSpacing", "fine_spacing", "Spacing", "spacing")
    if region_min is None or region_max is None or fine_spacing is None:
        raise ValueError("MechanicalGridRefinement requires RegionMin, RegionMax, and " "FineSpacing")
    region_min = np.asarray(region_min, dtype=np.float64)
    region_max = np.asarray(region_max, dtype=np.float64)
    if region_min.shape != (3,) or region_max.shape != (3,):
        raise ValueError("Mechanical-grid refinement bounds must be 3-vectors")
    if not np.isfinite(region_min).all() or not np.isfinite(region_max).all():
        raise ValueError("Mechanical-grid refinement bounds must be finite")
    if np.any(region_max <= region_min):
        raise ValueError("Mechanical-grid refinement RegionMax must exceed RegionMin")

    coarse_spacing = float(coarse_spacing)
    fine_spacing = float(fine_spacing)
    if not np.isfinite(fine_spacing) or fine_spacing <= 0.0:
        raise ValueError("Mechanical-grid FineSpacing must be positive")
    ratio = coarse_spacing / fine_spacing
    subdivision = int(round(ratio))
    if subdivision < 2 or not np.isclose(ratio, subdivision, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(
            "Mechanical-grid refinement requires coarse/fine spacing to be " "an integer ratio of at least two"
        )
    return {
        "region_min": np.ascontiguousarray(region_min),
        "region_max": np.ascontiguousarray(region_max),
        "fine_spacing": fine_spacing,
        "subdivision": subdivision,
    }


def _refine_axis(base_axis, region_min, region_max, subdivision):
    base_axis = np.asarray(base_axis, dtype=np.float64)
    cell_lower = base_axis[:-1]
    cell_upper = base_axis[1:]
    selected = (cell_upper > region_min) & (cell_lower < region_max)
    selected_ids = np.flatnonzero(selected)
    if selected_ids.size == 0:
        raise ValueError("Mechanical-grid refinement region does not intersect the grid")
    fractions = np.arange(1, int(subdivision), dtype=np.float64) / subdivision
    inserted = (
        cell_lower[selected_ids, None]
        + (cell_upper[selected_ids] - cell_lower[selected_ids])[:, None] * fractions[None, :]
    )
    axis = np.sort(np.concatenate((base_axis, inserted.reshape(-1))))
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(axis))))
    keep = np.concatenate(([True], np.diff(axis) > tolerance))
    return np.ascontiguousarray(axis[keep], dtype=np.float64)


def build_soft_mechanical_grid(
    template,
    spacing,
    shape_function_type,
    verlet_padding_cells=0,
    refinement=None,
):
    """Build a reference mechanical grid without using the SDF topology."""
    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Soft mechanical-grid spacing must be positive")

    vertices = np.asarray(template.objects.mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 1:
        raise RuntimeError("Soft template must provide three-dimensional surface vertices")
    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    center = 0.5 * (lower + upper)
    half_size = 0.5 * (upper - lower)

    stencil_halo = (1, 2, 2)[int(shape_function_type)]
    halo_cells = stencil_halo + max(int(verlet_padding_cells), 0)
    half_cells = np.ceil(half_size / spacing).astype(np.int64) + halo_cells
    half_cells = np.maximum(half_cells, 1)
    origin = center - half_cells * spacing
    shape = 2 * half_cells + 1
    normalized_refinement = normalize_soft_grid_refinement(refinement, spacing)
    axes = None
    if normalized_refinement is not None:
        base_axes = tuple(origin[d] + np.arange(int(shape[d]), dtype=np.float64) * spacing for d in range(3))
        axes = tuple(
            _refine_axis(
                base_axes[d],
                normalized_refinement["region_min"][d],
                normalized_refinement["region_max"][d],
                normalized_refinement["subdivision"],
            )
            for d in range(3)
        )
        shape = np.asarray([axis.size for axis in axes], dtype=np.int32)
    return SoftMechanicalGrid(
        origin=np.ascontiguousarray(origin, dtype=np.float64),
        shape=np.ascontiguousarray(shape, dtype=np.int32),
        spacing=spacing,
        axes=axes,
    )


def verlet_padding_cell_count(verlet_distance, grid_space):
    if grid_space <= 0.0:
        raise ValueError("Soft-grid spacing must be positive")
    ratio = max(float(verlet_distance), 0.0) / float(grid_space)
    roundoff = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(ratio))
    return int(np.ceil(max(ratio - roundoff, 0.0)))


def _linear_support(point, xmin, dx, gnum):
    base = np.floor((point - xmin) / dx).astype(np.int64)
    base = np.minimum(gnum - 2, np.maximum(0, base))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                yield int(base[0] + i + (base[1] + j) * gnum[0] + (base[2] + k) * gnum[0] * gnum[1])


def _boundary_type(grid_id, grid_num):
    boundary_type = min(2, grid_id) - min(grid_num - 1 - grid_id, 2)
    if boundary_type < 0:
        boundary_type += 3
    elif boundary_type > 0:
        boundary_type += 2
    return boundary_type


def _quadratic_bspline_weight(xp, xg, inv_dx, boundary_type):
    d = (xp - xg) * inv_dx
    if boundary_type == 0:
        if 0.5 <= d < 1.5:
            return (0.5 * d - 1.5) * d + 1.125
        if -0.5 <= d < 0.5:
            return -d * d + 0.75
        if -1.5 <= d < 0.5:
            return (0.5 * d + 1.5) * d + 1.125
    elif boundary_type == 1:
        if 0.0 <= d < 0.5:
            return 1.0 - d
        if 0.5 <= d < 1.5:
            return (0.5 * d - 1.5) * d + 1.125
    elif boundary_type == 2:
        if -1.0 <= d < -0.5:
            return 1.0 + d
        if -0.5 <= d < 0.5:
            return -d * d + 0.75
        if 0.5 <= d < 1.5:
            return (0.5 * d - 1.5) * d + 1.125
    elif boundary_type == 3:
        if -1.5 <= d < -0.5:
            return (0.5 * d + 1.5) * d + 1.125
        if -0.5 <= d < 0.5:
            return -d * d + 0.75
        if 0.5 <= d < 1.0:
            return 1.0 - d
    elif boundary_type == 4:
        if -1.5 <= d < -0.5:
            return (0.5 * d + 1.5) * d + 1.125
        if -0.5 <= d <= 0.0:
            return 1.0 + d
    return 0.0


def _cubic_bspline_weight(xp, xg, inv_dx, boundary_type):
    d = (xp - xg) * inv_dx
    if 1.0 <= d < 2.0:
        if boundary_type != 3:
            return ((-d / 6.0 + 1.0) * d - 2.0) * d + 4.0 / 3.0
    elif 0.0 <= d < 1.0:
        if boundary_type == 1:
            return (d * d / 6.0 - 1.0) * d + 1.0
        if boundary_type == 3:
            return (d / 3.0 - 1.0) * d * d + 2.0 / 3.0
        return (0.5 * d - 1.0) * d * d + 2.0 / 3.0
    elif -1.0 <= d < 0.0:
        if boundary_type == 4:
            return (-d * d / 6.0 + 1.0) * d + 1.0
        if boundary_type == 2:
            return (-d / 3.0 - 1.0) * d * d + 2.0 / 3.0
        return (-0.5 * d - 1.0) * d * d + 2.0 / 3.0
    elif -2.0 <= d < -1.0:
        return ((d / 6.0 + 1.0) * d + 2.0) * d + 4.0 / 3.0
    return 0.0


def _bspline_support(point, xmin, dx, gnum, shape_function_type):
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
            xg = xmin[axis] + node * dx
            if shape_function_type == 1:
                weight = _quadratic_bspline_weight(point[axis], xg, inv_dx, boundary_type)
            else:
                weight = _cubic_bspline_weight(point[axis], xg, inv_dx, boundary_type)
            axis_support.append((node, weight))
        supports.append(axis_support)

    for i, wx in supports[0]:
        for j, wy in supports[1]:
            for k, wz in supports[2]:
                # The lower threshold makes preprocessing conservative with
                # respect to the 1e-12 cutoff used by the Taichi insertion.
                if wx * wy * wz > 1.0e-14:
                    yield int(i + j * gnum[0] + k * gnum[0] * gnum[1])


def build_soft_grid_topology(
    template,
    material_points,
    shape_function_type,
    storage="Compact",
    padding_cells=0,
    *,
    mechanical_grid=None,
    levelset_extent_cells=0,
    verlet_padding_cells=0,
    levelset_verlet_padding_cells=0,
):
    grid = mechanical_grid
    if grid is None:
        raise ValueError("An independent SoftMechanicalGrid is required for LSMPM topology")
    gnum = np.asarray(grid.gnum, dtype=np.int64)
    logical_count = int(np.prod(gnum, dtype=np.int64))
    if logical_count != int(grid.gridSum):
        raise RuntimeError("Soft-grid dimensions do not match gridSum")

    storage_key = str(storage).strip().lower()
    xmin = np.asarray(grid.minBox(), dtype=np.float64)
    dx = float(grid.spacing)
    active = set()
    for point in np.asarray(material_points, dtype=np.float64):
        if shape_function_type == 0:
            active.update(_linear_support(point, xmin, dx, gnum))
        else:
            active.update(_bspline_support(point, xmin, dx, gnum, shape_function_type))
    for point in np.asarray(template.objects.mesh.vertices, dtype=np.float64):
        active.update(_linear_support(point, xmin, dx, gnum))
    if not active:
        raise RuntimeError("Soft-grid preprocessing found no supported nodes")

    active_ids = np.asarray(sorted(active), dtype=np.int64)
    active_ijk = np.column_stack(
        (
            active_ids % gnum[0],
            (active_ids % (gnum[0] * gnum[1])) // gnum[0],
            active_ids // (gnum[0] * gnum[1]),
        )
    )
    padding_cells = max(int(padding_cells), 0)
    if storage_key == "dense":
        compact_origin = np.zeros(3, dtype=np.int32)
        compact_upper = gnum.astype(np.int32)
    else:
        compact_origin = np.maximum(0, active_ijk.min(axis=0) - padding_cells).astype(np.int32)
        compact_upper = np.minimum(gnum, active_ijk.max(axis=0) + 1 + padding_cells).astype(np.int32)
    compact_shape = compact_upper - compact_origin
    return SoftGridTopology(
        compact_origin=np.ascontiguousarray(compact_origin),
        compact_shape=np.ascontiguousarray(compact_shape),
        logical_count=logical_count,
        compact_count=int(np.prod(compact_shape, dtype=np.int64)),
        support_count=int(active_ids.size),
        padding_cells=padding_cells,
        storage="Dense" if storage_key == "dense" else "Compact",
        levelset_extent_cells=max(int(levelset_extent_cells), 0),
        verlet_padding_cells=max(int(verlet_padding_cells), 0),
        levelset_verlet_padding_cells=max(int(levelset_verlet_padding_cells), 0),
    )


def select_soft_grid_topology(
    template,
    material_points,
    shape_function_type,
    storage="Compact",
    padding_cells=0,
    *,
    mechanical_grid=None,
    levelset_extent_cells=0,
    verlet_padding_cells=0,
    levelset_verlet_padding_cells=0,
):
    requested = str(storage).strip().lower()
    return build_soft_grid_topology(
        template,
        material_points,
        shape_function_type,
        "Dense" if requested == "dense" else "Compact",
        padding_cells,
        mechanical_grid=mechanical_grid,
        levelset_extent_cells=levelset_extent_cells,
        verlet_padding_cells=verlet_padding_cells,
        levelset_verlet_padding_cells=levelset_verlet_padding_cells,
    )
