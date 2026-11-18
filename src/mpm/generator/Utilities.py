from __future__ import annotations
from src.utils.errors import ConfigurationError, SimulationStateError
from src.mpm.soft_particle.Utilities import build_hexahedral_template_support
from src.mpm.soft_particle.Utilities import build_soft_mechanical_grid
from src.mpm.soft_particle.Utilities import build_tetrahedral_template_support
from src.mpm.soft_particle.Utilities import normalize_soft_grid_refinement
from src.mpm.soft_particle.Utilities import normalize_soft_grid_type
from src.mpm.soft_particle.Utilities import select_soft_grid_topology
from src.mpm.soft_particle.Utilities import verlet_padding_cell_count
import numpy as np
import hashlib
import os
from src.mpm.generator.InsertionKernel import *
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.utils.FieldIO import runtime_float_numpy_dtype


def sample_soft_material_points(
    template_ptr: GeneralShapeTemplate,
    points_per_cell,
    center_quadrature=False,
    material_points=None,
    point_volume=None,
    mechanical_grid=None,
):
    if material_points is not None:
        points = np.asarray(material_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
            raise ValueError("MaterialPointCoordinates must have shape (point_count, 3)")
        if not np.isfinite(points).all():
            raise ValueError("MaterialPointCoordinates must be finite")
        if point_volume is None:
            point_volume = float(template_ptr.objects.volume) / points.shape[0]
        point_volume = np.asarray(point_volume, dtype=np.float64)
        if point_volume.ndim == 0:
            point_volume = np.full(points.shape[0], float(point_volume), dtype=np.float64)
        if point_volume.shape != (points.shape[0],):
            raise ValueError("MaterialPointVolume must be a scalar or have one value per " "material point")
        if not np.isfinite(point_volume).all() or np.any(point_volume <= 0.0):
            raise ValueError("MaterialPointVolume must be positive and finite")
        return (
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray(point_volume, dtype=np.float64),
        )

    if mechanical_grid is None:

        mechanical_grid = build_soft_mechanical_grid(
            template_ptr,
            0.15 * float(template_ptr.objects.eqradius),
            shape_function_type=0,
        )
    ppc = max(int(points_per_cell), 1)
    spacing = mechanical_grid.spacing / ppc
    lower = np.asarray(mechanical_grid.min_box(), dtype=np.float64)
    upper = np.asarray(mechanical_grid.max_box(), dtype=np.float64)
    axes = [np.arange(lower[d] + 0.5 * spacing, upper[d], spacing) for d in range(3)]
    if axes[0].size == 0 or axes[1].size == 0 or axes[2].size == 0:
        points = np.asarray(template_ptr.objects.center, dtype=np.float64).reshape(1, 3)
        return (
            np.ascontiguousarray(points),
            np.asarray([float(template_ptr.objects.volume)], dtype=np.float64),
        )

    xv, yv, zv = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    points = np.column_stack((xv.reshape(-1), yv.reshape(-1), zv.reshape(-1)))
    # Trimesh proximity queries can allocate candidate-triangle workspaces
    # proportional to the number of query points.  A fine three-dimensional
    # LSMPM template may have close to a million voxel-center candidates, so
    # one monolithic signed-distance call can exceed a worker's host-memory
    # limit before any Taichi field is allocated.  Chunking is mathematically
    # identical because the signed distance is evaluated independently at
    # every point.
    distance_batch_size = max(
        int(os.environ.get("GT_SOFT_TEMPLATE_DISTANCE_BATCH_SIZE", "16384")),
        1,
    )
    inside = np.empty(points.shape[0], dtype=bool)
    for start in range(0, points.shape[0], distance_batch_size):
        stop = min(start + distance_batch_size, points.shape[0])
        inside[start:stop] = np.asarray(template_ptr.objects(points[start:stop])).reshape(-1) <= 0.0
    points = points[inside]
    if points.shape[0] == 0:
        points = np.asarray(template_ptr.objects.center, dtype=np.float64).reshape(1, 3)
    if center_quadrature:
        target_center = np.asarray(template_ptr.objects.center, dtype=np.float64)
        points += target_center - np.mean(points, axis=0)
    # Preserve the template's integrated volume independently of voxel-center
    # sampling resolution; every body then carries the requested bulk density.
    point_volume = np.full(
        points.shape[0],
        float(template_ptr.objects.volume) / points.shape[0],
        dtype=np.float64,
    )
    return (
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(point_volume),
    )


def preprocess_soft_grid_template(
    template_ptr,
    points_per_cell,
    center_quadrature,
    shape_function_type,
    storage,
    verlet_distance_multiplier,
    material_points=None,
    point_volume=None,
    grid_type="Hexahedron",
    mechanical_grid_spacing=None,
    mechanical_grid_refinement=None,
    reference_volume=None,
):
    from src.mpm.soft_particle.TemplateSupport import HEXAHEDRON, TETRAHEDRON

    normalized_grid_type, grid_type_id = normalize_soft_grid_type(grid_type)
    mechanical_shape_function_type = 0 if grid_type_id == TETRAHEDRON else int(shape_function_type)
    if mechanical_grid_spacing is None:
        mechanical_grid_spacing = 0.15 * float(template_ptr.objects.eqradius)
    mechanical_grid_spacing = float(mechanical_grid_spacing)
    if not np.isfinite(mechanical_grid_spacing) or mechanical_grid_spacing <= 0.0:
        raise ValueError("MechanicalGridSpacing must be positive and finite")
    if grid_type_id == TETRAHEDRON and material_points is not None:
        raise ConfigurationError(
            "Tetrahedral LSMPM places one material point at each linear "
            "tetrahedron Gauss point; MaterialPointCoordinates is only "
            "supported by the hexahedral grid"
        )
    normalized_refinement = normalize_soft_grid_refinement(mechanical_grid_refinement, mechanical_grid_spacing)
    if grid_type_id == HEXAHEDRON and normalized_refinement is not None:
        raise ConfigurationError("MechanicalGridRefinement is only supported by the linear " "tetrahedral soft grid")
    cache = getattr(template_ptr, "soft_grid_preprocess_cache", None)
    if cache is None:
        cache = {}
        template_ptr.soft_grid_preprocess_cache = cache
    custom_digest = None
    if material_points is not None:
        custom_points = np.ascontiguousarray(material_points, dtype=np.float64)
        custom_digest = hashlib.sha256(custom_points.view(np.uint8)).hexdigest()
    point_volume_digest = None
    if point_volume is not None:
        point_volume_values = np.ascontiguousarray(np.asarray(point_volume, dtype=np.float64))
        point_volume_digest = hashlib.sha256(point_volume_values.view(np.uint8)).hexdigest()
    refinement_key = None
    if normalized_refinement is not None:
        refinement_key = (
            tuple(normalized_refinement["region_min"]),
            tuple(normalized_refinement["region_max"]),
            float(normalized_refinement["fine_spacing"]),
            int(normalized_refinement["subdivision"]),
        )
    if reference_volume is not None:
        reference_volume = float(reference_volume)
        if not np.isfinite(reference_volume) or reference_volume <= 0.0:
            raise ValueError("ReferenceVolume must be positive and finite")
    quadrature_per_cell = 1 if grid_type_id == TETRAHEDRON else max(int(points_per_cell), 1)
    cache_key = (
        quadrature_per_cell,
        bool(center_quadrature),
        mechanical_shape_function_type,
        str(storage).strip().lower(),
        float(verlet_distance_multiplier),
        int(template_ptr.objects.grid.extent),
        float(template_ptr.objects.grid.grid_space),
        tuple(np.asarray(template_ptr.objects.grid.gnum, dtype=np.int64)),
        mechanical_grid_spacing,
        custom_digest,
        point_volume_digest,
        normalized_grid_type,
        refinement_key,
        reference_volume,
    )
    if cache_key in cache:
        return cache[cache_key]
    verlet_distance = max(float(verlet_distance_multiplier), 0.0) * float(template_ptr.boundings.r_bound)
    levelset_verlet_padding_cells = verlet_padding_cell_count(
        verlet_distance,
        float(template_ptr.objects.grid.grid_space),
    )
    levelset_extent_cells = int(template_ptr.objects.grid.extent)
    if levelset_verlet_padding_cells > levelset_extent_cells:
        raise SimulationStateError(
            "Level-set template extent is smaller than the LSDEM Verlet "
            f"requirement: extent={levelset_extent_cells}, "
            f"required={levelset_verlet_padding_cells}"
        )
    minimum_mechanical_spacing = (
        normalized_refinement["fine_spacing"] if normalized_refinement is not None else mechanical_grid_spacing
    )
    mechanical_verlet_padding_cells = verlet_padding_cell_count(
        verlet_distance,
        minimum_mechanical_spacing,
    )
    mechanical_grid_halo_cells = verlet_padding_cell_count(
        verlet_distance,
        mechanical_grid_spacing,
    )
    mechanical_grid = build_soft_mechanical_grid(
        template_ptr,
        mechanical_grid_spacing,
        mechanical_shape_function_type,
        verlet_padding_cells=mechanical_grid_halo_cells,
        refinement=normalized_refinement,
    )
    padding_cells = mechanical_verlet_padding_cells
    if grid_type_id == HEXAHEDRON:
        material_points, point_volume = sample_soft_material_points(
            template_ptr,
            quadrature_per_cell,
            center_quadrature,
            material_points=material_points,
            point_volume=point_volume,
            mechanical_grid=mechanical_grid,
        )
        topology = select_soft_grid_topology(
            template_ptr,
            material_points,
            mechanical_shape_function_type,
            storage,
            padding_cells,
            mechanical_grid=mechanical_grid,
            levelset_extent_cells=levelset_extent_cells,
            verlet_padding_cells=mechanical_verlet_padding_cells,
            levelset_verlet_padding_cells=levelset_verlet_padding_cells,
        )
        support = build_hexahedral_template_support(
            template_ptr,
            material_points,
            topology,
            mechanical_shape_function_type,
            mechanical_grid,
        )
    else:
        (
            material_points,
            point_volume,
            topology,
            support,
        ) = build_tetrahedral_template_support(
            template_ptr,
            mechanical_grid,
            storage,
            padding_cells,
            levelset_extent_cells=levelset_extent_cells,
            verlet_padding_cells=mechanical_verlet_padding_cells,
            levelset_verlet_padding_cells=levelset_verlet_padding_cells,
            reference_volume=reference_volume,
        )
    result = (material_points, point_volume, topology, support)
    cache[cache_key] = result
    return result


def soft_template_kernel_arrays(template_ptr, material_points, point_volumes):
    dtype = runtime_float_numpy_dtype()
    return (
        np.ascontiguousarray(template_ptr.objects.mesh.vertices, dtype=dtype),
        np.ascontiguousarray(template_ptr.parameter, dtype=dtype),
        np.ascontiguousarray(template_ptr.objects.grid.distance_field, dtype=dtype),
        np.ascontiguousarray(material_points, dtype=dtype),
        np.ascontiguousarray(point_volumes, dtype=dtype),
    )


def soft_template_shape_bounds(surface_nodes, material_points, point_volumes, scale_factor):
    """Return conservative template-space bounds for one soft body."""
    dtype = np.result_type(
        np.asarray(surface_nodes).dtype,
        np.asarray(material_points).dtype,
        np.asarray(point_volumes).dtype,
    )
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        dtype = np.dtype(np.float64)
    surface = np.asarray(surface_nodes, dtype=dtype)
    points = np.asarray(material_points, dtype=dtype)
    volumes = np.asarray(point_volumes, dtype=dtype)
    scale = float(scale_factor)
    if surface.ndim != 2 or surface.shape[1] != 3 or surface.shape[0] == 0:
        raise ValueError("soft template requires at least one surface node")
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("soft template requires at least one material point")
    if volumes.shape != (points.shape[0],) or np.any(volumes <= 0.0):
        raise ValueError("soft material-point volumes must be positive")

    scaled_surface = scale * surface
    scaled_points = scale * points
    point_padding = 0.5 * np.cbrt(volumes * scale**3)
    shape_min = np.minimum(
        np.min(scaled_surface, axis=0),
        np.min(scaled_points - point_padding[:, None], axis=0),
    )
    shape_max = np.maximum(
        np.max(scaled_surface, axis=0),
        np.max(scaled_points + point_padding[:, None], axis=0),
    )
    shape_radius = max(
        float(np.max(np.linalg.norm(scaled_surface, axis=1))),
        float(np.max(np.linalg.norm(scaled_points, axis=1) + np.sqrt(3.0) * point_padding)),
    )
    return (
        np.ascontiguousarray(shape_min, dtype=dtype),
        np.ascontiguousarray(shape_max, dtype=dtype),
        shape_radius,
    )
