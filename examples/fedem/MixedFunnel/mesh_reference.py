"""Quality-controlled reference meshes used by this example family."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def tetra_quality(points: np.ndarray, cells: np.ndarray) -> dict[str, float]:
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int32)
    tetra = points[cells]
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_square_sum = sum(np.sum((tetra[:, first] - tetra[:, second]) ** 2, axis=1) for first, second in edge_pairs)
    determinants = np.linalg.det(
        np.stack(
            (
                tetra[:, 1] - tetra[:, 0],
                tetra[:, 2] - tetra[:, 0],
                tetra[:, 3] - tetra[:, 0],
            ),
            axis=2,
        )
    )
    volumes = np.abs(determinants) / 6.0
    mean_ratio = 12.0 * np.power(3.0 * volumes, 2.0 / 3.0) / np.maximum(edge_square_sum, 1.0e-30)
    return {
        "minimum_tetra_mean_ratio": float(np.min(mean_ratio)),
        "mean_tetra_mean_ratio": float(np.mean(mean_ratio)),
        "minimum_tetra_volume": float(np.min(volumes)),
        "maximum_tetra_volume": float(np.max(volumes)),
    }


def _volume_centroid(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    tetra = points[cells]
    determinants = np.linalg.det(
        np.stack(
            (
                tetra[:, 1] - tetra[:, 0],
                tetra[:, 2] - tetra[:, 0],
                tetra[:, 3] - tetra[:, 0],
            ),
            axis=2,
        )
    )
    volumes = np.abs(determinants) / 6.0
    return np.sum(volumes[:, None] * np.mean(tetra, axis=1), axis=0) / np.sum(volumes)


def normalize_reference(mesh):
    """Center a tetra mesh at its volume centroid and set bounding radius to one."""

    from src.fem.generator import FEMMesh

    points = np.asarray(mesh.points, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    points = points - _volume_centroid(points, cells)[None, :]
    radius = float(np.max(np.linalg.norm(points, axis=1)))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("reference FEM particle has an invalid bounding radius")
    normalized = FEMMesh(
        points / radius,
        cells.copy(),
        "TET4",
        name=f"{mesh.name}_unit_reference",
    )
    quality = tetra_quality(normalized.points, normalized.cells)
    quality["connected_body_count"] = int(normalized.body_ids.size)
    return normalized, quality


def load_sphere_reference(path: Path):
    from src.fem.generator import FEMGenerateManager

    mesh = FEMGenerateManager().read(Path(path), cell_type="TET4")
    normalized, quality = normalize_reference(mesh)
    if quality["connected_body_count"] != 1:
        raise ValueError(
            f"sphere reference {Path(path)} contains "
            f"{quality['connected_body_count']} disconnected volume meshes; "
            "one visual particle must contain exactly one FEM body"
        )
    return normalized, quality


def tetrahedralize_irregular_reference(
    path: Path,
    *,
    maximum_size_fraction: float = 0.10,
):
    """Tetrahedralize a closed irregular STL and normalize it to unit radius."""

    import gmsh
    from src.fem.generator import FEMMesh

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not 0.02 <= maximum_size_fraction <= 0.30:
        raise ValueError("maximum_size_fraction must lie in [0.02, 0.30]")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("llm_assist_irregular_fem_particle")
        gmsh.merge(str(path))
        gmsh.model.mesh.classifySurfaces(
            math.radians(40.0),
            boundary=True,
            forReparametrization=True,
            curveAngle=math.pi,
        )
        gmsh.model.mesh.createGeometry()
        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            raise RuntimeError(f"Gmsh found no closed surfaces in {path}")
        surface_loop = gmsh.model.geo.addSurfaceLoop([tag for _, tag in surfaces])
        gmsh.model.geo.addVolume([surface_loop])
        gmsh.model.geo.synchronize()

        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        for dim, tag in surfaces:
            bounds = gmsh.model.getBoundingBox(dim, tag)
            lower = np.minimum(lower, bounds[:3])
            upper = np.maximum(upper, bounds[3:])
        characteristic = float(np.max(upper - lower))
        maximum_size = maximum_size_fraction * characteristic
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.45 * maximum_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", maximum_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
        tag_to_local = {int(tag): index for index, tag in enumerate(np.asarray(node_tags))}
        blocks = []
        element_types, _, element_nodes = gmsh.model.mesh.getElements(dim=3)
        for element_type, flattened in zip(element_types, element_nodes):
            properties = gmsh.model.mesh.getElementProperties(element_type)
            node_count = int(properties[3])
            primary_count = int(properties[5])
            if primary_count != 4:
                continue
            tags = np.asarray(flattened, dtype=np.int64).reshape(-1, node_count)[:, :4]
            blocks.append(
                np.asarray(
                    [[tag_to_local[int(tag)] for tag in row] for row in tags],
                    dtype=np.int32,
                )
            )
        if not blocks:
            raise RuntimeError("Gmsh generated no first-order tetrahedra")
        cells = np.vstack(blocks)
        used = np.unique(cells)
        inverse = np.full(points.shape[0], -1, dtype=np.int32)
        inverse[used] = np.arange(used.size, dtype=np.int32)
        mesh = FEMMesh(
            points[used],
            inverse[cells],
            "TET4",
            name="irregular_sand_particle",
        )
        return normalize_reference(mesh)
    finally:
        gmsh.finalize()


def tetrahedralize_primitive_reference(
    shape: str,
    *,
    minimum_element_count: int = 1000,
    minimum_mean_ratio: float = 0.05,
):
    """Generate a quality-controlled unit sphere or cube TET4 reference.

    The mesher is deliberately count-gated instead of relying on a nominal
    spacing: Gmsh versions may realize different counts for the same target
    size.  Successive sizes are attempted until both the element-count and
    tetrahedral-quality gates pass.
    """

    import gmsh
    from src.fem.generator import FEMMesh

    normalized_shape = str(shape).strip().lower()
    if normalized_shape not in ("sphere", "cube"):
        raise ValueError("primitive FEM reference must be 'sphere' or 'cube'")
    minimum_element_count = int(minimum_element_count)
    if minimum_element_count < 1:
        raise ValueError("minimum_element_count must be positive")

    attempts = []
    for maximum_size in (0.18, 0.15, 0.12, 0.10, 0.08):
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(f"llm_assist_{normalized_shape}_fem_reference")
            if normalized_shape == "sphere":
                gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
            else:
                gmsh.model.occ.addBox(-0.5, -0.5, -0.5, 1.0, 1.0, 1.0)
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.55 * maximum_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", maximum_size)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.optimize("Netgen")

            node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
            points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
            tag_to_local = {int(tag): index for index, tag in enumerate(np.asarray(node_tags))}
            blocks = []
            element_types, _, element_nodes = gmsh.model.mesh.getElements(dim=3)
            for element_type, flattened in zip(element_types, element_nodes):
                properties = gmsh.model.mesh.getElementProperties(element_type)
                node_count = int(properties[3])
                primary_count = int(properties[5])
                if primary_count != 4:
                    continue
                tags = np.asarray(flattened, dtype=np.int64).reshape(-1, node_count)[:, :4]
                blocks.append(
                    np.asarray(
                        [[tag_to_local[int(tag)] for tag in row] for row in tags],
                        dtype=np.int32,
                    )
                )
            if not blocks:
                raise RuntimeError("Gmsh generated no first-order tetrahedra")
            cells = np.vstack(blocks)
            used = np.unique(cells)
            inverse = np.full(points.shape[0], -1, dtype=np.int32)
            inverse[used] = np.arange(used.size, dtype=np.int32)
            cells = inverse[cells]
            points = points[used]
            tetra = points[cells]
            determinants = np.linalg.det(
                np.stack(
                    (
                        tetra[:, 1] - tetra[:, 0],
                        tetra[:, 2] - tetra[:, 0],
                        tetra[:, 3] - tetra[:, 0],
                    ),
                    axis=2,
                )
            )
            negative = determinants < 0.0
            cells[negative, 1], cells[negative, 2] = (
                cells[negative, 2].copy(),
                cells[negative, 1].copy(),
            )
            candidate, quality = normalize_reference(
                FEMMesh(
                    points,
                    cells,
                    "TET4",
                    name=f"gmsh_{normalized_shape}_reference",
                )
            )
            quality.update(
                node_count=int(candidate.number_of_nodes),
                dof_count=int(3 * candidate.number_of_nodes),
                element_count=int(candidate.number_of_cells),
                requested_minimum_element_count=minimum_element_count,
                requested_minimum_mean_ratio=float(minimum_mean_ratio),
                maximum_size_fraction=float(maximum_size),
            )
            quality["passed"] = bool(
                quality["element_count"] >= minimum_element_count
                and quality["minimum_tetra_volume"] > 0.0
                and quality["minimum_tetra_mean_ratio"] >= float(minimum_mean_ratio)
            )
            attempts.append(dict(quality))
            if quality["passed"]:
                quality["attempts"] = attempts
                return candidate, quality
        finally:
            gmsh.finalize()
    raise RuntimeError(
        f"unable to generate a {normalized_shape} FEM reference with at least "
        f"{minimum_element_count} TET4 elements and mean ratio >= "
        f"{minimum_mean_ratio}; attempts={attempts}"
    )


def euler_rotation(angles_degrees) -> np.ndarray:
    x, y, z = np.deg2rad(np.asarray(angles_degrees, dtype=np.float64))
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, math.cos(x), -math.sin(x)], [0.0, math.sin(x), math.cos(x)]])
    ry = np.asarray([[math.cos(y), 0.0, math.sin(y)], [0.0, 1.0, 0.0], [-math.sin(y), 0.0, math.cos(y)]])
    rz = np.asarray([[math.cos(z), -math.sin(z), 0.0], [math.sin(z), math.cos(z), 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def place_reference(reference, center, radius: float, angles_degrees, *, name: str):
    from src.fem.generator import FEMMesh

    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("particle radius must be positive")
    rotation = euler_rotation(angles_degrees)
    points = (
        radius * (np.asarray(reference.points, dtype=np.float64) @ rotation.T)
        + np.asarray(center, dtype=np.float64)[None, :]
    )
    return FEMMesh(points, reference.cells.copy(), "TET4", name=name)
