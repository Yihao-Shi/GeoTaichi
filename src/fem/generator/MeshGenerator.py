"""Built-in and file-based mesh generation for the FEM engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.fem.generator.Mesh import FEMMesh, normalize_cell_type


def _positive_divisions(values, size, label):
    divisions = np.asarray(values, dtype=np.int32).reshape(-1)
    if divisions.size != size or np.any(divisions <= 0):
        raise ValueError(f"{label} must contain {size} positive integers")
    return divisions


def _origin(origin):
    origin = np.asarray(origin, dtype=np.float64).reshape(-1)
    if origin.size == 2:
        origin = np.append(origin, 0.0)
    if origin.size != 3:
        raise ValueError("origin must contain two or three coordinates")
    return origin


def _compact(points, cells, node_sets=None):
    """Remove nodes not referenced by the selected cell formulation."""
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int32)
    used = np.unique(cells)
    if used.size == points.shape[0] and np.array_equal(used, np.arange(points.shape[0])):
        return points, cells, dict(node_sets or {})
    inverse = np.full(points.shape[0], -1, dtype=np.int32)
    inverse[used] = np.arange(used.size, dtype=np.int32)
    compact_sets = {}
    for set_name, indices in dict(node_sets or {}).items():
        mapped = inverse[np.asarray(indices, dtype=np.int32).reshape(-1)]
        compact_sets[set_name] = np.unique(mapped[mapped >= 0])
    return points[used], inverse[cells], compact_sets


class FEMGenerateManager:
    """Create structured primitive meshes or import meshio-supported files."""

    def create_box(
        self,
        size=(1.0, 1.0, 1.0),
        divisions=(1, 1, 1),
        element_type="TET4",
        origin=(0.0, 0.0, 0.0),
        name="box",
    ) -> FEMMesh:
        size = np.asarray(size, dtype=np.float64).reshape(-1)
        if size.size != 3 or np.any(size <= 0.0):
            raise ValueError("box size must contain three positive lengths")
        nx, ny, nz = _positive_divisions(divisions, 3, "box divisions")
        origin = _origin(origin)
        x = np.linspace(origin[0], origin[0] + size[0], nx + 1)
        y = np.linspace(origin[1], origin[1] + size[1], ny + 1)
        z = np.linspace(origin[2], origin[2] + size[2], nz + 1)
        points = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
        node_id = np.arange(points.shape[0], dtype=np.int32).reshape(nx + 1, ny + 1, nz + 1)
        hexahedra = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    hexahedra.append(
                        [
                            node_id[i, j, k],
                            node_id[i + 1, j, k],
                            node_id[i + 1, j + 1, k],
                            node_id[i, j + 1, k],
                            node_id[i, j, k + 1],
                            node_id[i + 1, j, k + 1],
                            node_id[i + 1, j + 1, k + 1],
                            node_id[i, j + 1, k + 1],
                        ]
                    )
        hexahedra = np.asarray(hexahedra, dtype=np.int32)
        cell_type = normalize_cell_type(element_type)
        if cell_type == "hexahedron":
            cells = hexahedra
        elif cell_type == "tetra":
            # Six conforming tetrahedra share the 0--6 body diagonal.  The
            # same cube split is used everywhere, so neighboring faces match.
            split = np.asarray(
                (
                    (0, 1, 2, 6),
                    (0, 2, 3, 6),
                    (0, 3, 7, 6),
                    (0, 7, 4, 6),
                    (0, 4, 5, 6),
                    (0, 5, 1, 6),
                ),
                dtype=np.int32,
            )
            cells = hexahedra[:, split].reshape(-1, 4)
        else:
            raise ValueError("box volume meshes require element_type='TET4' or 'HEX8'")
        return FEMMesh(points, cells, cell_type, name=name)

    def create_rectangle(
        self,
        size=(1.0, 1.0),
        divisions=(1, 1),
        origin=(0.0, 0.0, 0.0),
        plane="xy",
        name="rectangle",
    ) -> FEMMesh:
        size = np.asarray(size, dtype=np.float64).reshape(-1)
        if size.size != 2 or np.any(size <= 0.0):
            raise ValueError("rectangle size must contain two positive lengths")
        nx, ny = _positive_divisions(divisions, 2, "rectangle divisions")
        origin = _origin(origin)
        u = np.linspace(0.0, size[0], nx + 1)
        v = np.linspace(0.0, size[1], ny + 1)
        uv = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1).reshape(-1, 2)
        plane = str(plane).lower()
        points = np.repeat(origin[None, :], uv.shape[0], axis=0)
        plane_axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        if plane not in plane_axes:
            raise ValueError("rectangle plane must be 'xy', 'xz', or 'yz'")
        points[:, plane_axes[plane][0]] += uv[:, 0]
        points[:, plane_axes[plane][1]] += uv[:, 1]
        node_id = np.arange(points.shape[0], dtype=np.int32).reshape(nx + 1, ny + 1)
        triangles = []
        for i in range(nx):
            for j in range(ny):
                n00 = node_id[i, j]
                n10 = node_id[i + 1, j]
                n11 = node_id[i + 1, j + 1]
                n01 = node_id[i, j + 1]
                if (i + j) % 2 == 0:
                    triangles.extend(((n00, n10, n11), (n00, n11, n01)))
                else:
                    triangles.extend(((n00, n10, n01), (n10, n11, n01)))
        return FEMMesh(points, np.asarray(triangles, dtype=np.int32), "triangle", name=name)

    def create_circle(
        self,
        radius=1.0,
        radial_divisions=4,
        circumferential_divisions=32,
        center=(0.0, 0.0, 0.0),
        plane="xy",
        name="circle",
    ) -> FEMMesh:
        radius = float(radius)
        radial_divisions = int(radial_divisions)
        circumferential_divisions = int(circumferential_divisions)
        if radius <= 0.0 or radial_divisions <= 0 or circumferential_divisions < 3:
            raise ValueError("circle radius/divisions must be positive and use at least three angular sectors")
        center = _origin(center)
        local_points = [[0.0, 0.0]]
        angles = 2.0 * np.pi * np.arange(circumferential_divisions) / circumferential_divisions
        for radial_id in range(1, radial_divisions + 1):
            current_radius = radius * radial_id / radial_divisions
            local_points.extend(np.column_stack((current_radius * np.cos(angles), current_radius * np.sin(angles))))
        local_points = np.asarray(local_points, dtype=np.float64)
        plane = str(plane).lower()
        plane_axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        if plane not in plane_axes:
            raise ValueError("circle plane must be 'xy', 'xz', or 'yz'")
        points = np.repeat(center[None, :], local_points.shape[0], axis=0)
        points[:, plane_axes[plane][0]] += local_points[:, 0]
        points[:, plane_axes[plane][1]] += local_points[:, 1]

        def ring_node(ring, angular):
            return 1 + (ring - 1) * circumferential_divisions + angular % circumferential_divisions

        triangles = []
        for angular in range(circumferential_divisions):
            triangles.append((0, ring_node(1, angular), ring_node(1, angular + 1)))
        for ring in range(1, radial_divisions):
            for angular in range(circumferential_divisions):
                a = ring_node(ring, angular)
                b = ring_node(ring + 1, angular)
                c = ring_node(ring + 1, angular + 1)
                d = ring_node(ring, angular + 1)
                triangles.extend(((a, b, c), (a, c, d)))
        mesh = FEMMesh(points, np.asarray(triangles, dtype=np.int32), "triangle", name=name)
        radial_distance = np.linalg.norm(local_points, axis=1)
        tolerance = 1.0e-10 * max(radius, 1.0)
        mesh.node_sets["rim"] = np.flatnonzero(np.isclose(radial_distance, radius, atol=tolerance, rtol=0.0)).astype(
            np.int32
        )
        mesh.node_sets["center"] = np.asarray([0], dtype=np.int32)
        return mesh

    def create_cylinder(
        self,
        radius=1.0,
        height=1.0,
        radial_divisions=3,
        circumferential_divisions=24,
        height_divisions=4,
        origin=(0.0, 0.0, 0.0),
        name="cylinder",
    ) -> FEMMesh:
        """Create a polygonal cylindrical TET4 volume using SciPy Delaunay."""
        radius = float(radius)
        height = float(height)
        height_divisions = int(height_divisions)
        if height <= 0.0 or height_divisions <= 0:
            raise ValueError("cylinder height and height_divisions must be positive")
        disk = self.create_circle(
            radius,
            radial_divisions,
            circumferential_divisions,
            center=(0.0, 0.0, 0.0),
        )
        origin = _origin(origin)
        layers = []
        for z in np.linspace(0.0, height, height_divisions + 1):
            layer = disk.points.copy()
            layer[:, 2] = z
            layers.append(layer + origin)
        points = np.vstack(layers)
        try:
            from scipy.spatial import Delaunay
        except ImportError as exc:
            raise ImportError("create_cylinder requires scipy, which is a GeoTaichi dependency") from exc
        cells = Delaunay(points, qhull_options="QJ Qt").simplices.astype(np.int32)
        vertices = points[cells]
        determinants = np.linalg.det(
            np.stack(
                (
                    vertices[:, 1] - vertices[:, 0],
                    vertices[:, 2] - vertices[:, 0],
                    vertices[:, 3] - vertices[:, 0],
                ),
                axis=2,
            )
        )
        tolerance = 1.0e-12 * max(radius, height, 1.0) ** 3
        cells = cells[np.abs(determinants) > tolerance]
        mesh = FEMMesh(points, cells, "tetra", name=name)
        relative = mesh.points - origin
        scale = max(radius, height, 1.0)
        atol = 1.0e-9 * scale
        mesh.node_sets["bottom"] = np.flatnonzero(np.isclose(relative[:, 2], 0.0, atol=atol)).astype(np.int32)
        mesh.node_sets["top"] = np.flatnonzero(np.isclose(relative[:, 2], height, atol=atol)).astype(np.int32)
        radial = np.linalg.norm(relative[:, :2], axis=1)
        mesh.node_sets["side"] = np.flatnonzero(np.isclose(radial, radius, atol=atol)).astype(np.int32)
        return mesh

    def read(self, filename, cell_type=None, name=None) -> FEMMesh:
        """Read OBJ, Gmsh, Abaqus, VTK and other meshio-supported files."""
        filename = Path(filename)
        if filename.suffix.lower() == ".obj":
            requested = normalize_cell_type(cell_type or "TRI3")
            if requested != "triangle":
                raise ValueError("OBJ input is a surface mesh and requires cell_type='TRI3'")
            points = []
            texture_points = []
            polygon_corners = []
            for raw_line in filename.read_text(encoding="utf8", errors="ignore").splitlines():
                fields = raw_line.strip().split()
                if not fields or fields[0].startswith("#"):
                    continue
                if fields[0] == "v" and len(fields) >= 4:
                    points.append([float(fields[1]), float(fields[2]), float(fields[3])])
                elif fields[0] == "vt" and len(fields) >= 3:
                    texture_points.append([float(fields[1]), float(fields[2]), 0.0])
                elif fields[0] == "f" and len(fields) >= 4:
                    face = []
                    for token in fields[1:]:
                        indices = token.split("/")
                        vertex = int(indices[0])
                        vertex = vertex - 1 if vertex > 0 else len(points) + vertex
                        texture = None
                        if len(indices) > 1 and indices[1]:
                            texture = int(indices[1])
                            texture = texture - 1 if texture > 0 else len(texture_points) + texture
                        face.append((vertex, texture))
                    for local_id in range(1, len(face) - 1):
                        polygon_corners.append((face[0], face[local_id], face[local_id + 1]))
            if not points or not polygon_corners:
                raise ValueError(f"OBJ file {filename} contains no triangularizable faces")

            # OBJ texture vertices have an independent index stream. Keep the
            # geometric and material connectivity separate so a UV seam does
            # not disconnect a physical cloth node.
            has_texture = [corner[1] is not None for face in polygon_corners for corner in face]
            material_points = None
            material_cells = None
            if has_texture and all(has_texture):
                triangles = [[corner[0] for corner in face] for face in polygon_corners]
                material_cells = [[corner[1] for corner in face] for face in polygon_corners]
                if min(min(face) for face in triangles) < 0 or max(max(face) for face in triangles) >= len(points):
                    raise ValueError(f"OBJ file {filename} contains an invalid vertex index")
                if min(min(face) for face in material_cells) < 0 or max(max(face) for face in material_cells) >= len(
                    texture_points
                ):
                    raise ValueError(f"OBJ file {filename} contains an invalid texture index")
                points, triangles, _ = _compact(points, triangles)
                material_points, material_cells, _ = _compact(texture_points, material_cells)
            elif any(has_texture):
                raise ValueError(f"OBJ file {filename} mixes textured and untextured face corners")
            else:
                triangles = [[corner[0] for corner in face] for face in polygon_corners]
                points, triangles, _ = _compact(points, triangles)
            return FEMMesh(
                points,
                triangles,
                "triangle",
                name=name or filename.stem,
                material_points=material_points,
                material_cells=material_cells,
            )
        try:
            import meshio
        except ImportError as exc:
            raise ImportError(
                f"Reading {filename.suffix or 'this mesh format'} requires the GeoTaichi meshio dependency"
            ) from exc
        imported = meshio.read(filename)
        requested = normalize_cell_type(cell_type) if cell_type is not None else None
        candidates = []
        conversions = {
            "tetra": ("tetra", 4),
            "tetra10": ("tetra", 4),
            "hexahedron": ("hexahedron", 8),
            "hexahedron20": ("hexahedron", 8),
            "hexahedron27": ("hexahedron", 8),
            "triangle": ("triangle", 3),
            "triangle6": ("triangle", 3),
            "quad": ("quad", 4),
            "quad8": ("quad", 4),
            "quad9": ("quad", 4),
        }
        for block in imported.cells:
            if block.type in conversions:
                target, corners = conversions[block.type]
                candidates.append((target, np.asarray(block.data[:, :corners], dtype=np.int32)))
        if requested is None:
            available = {target for target, _ in candidates}
            requested = next((kind for kind in ("tetra", "hexahedron", "triangle") if kind in available), None)
        selected = [cells for target, cells in candidates if target == requested]
        if not selected and requested == "triangle":
            quads = [cells for target, cells in candidates if target == "quad"]
            for cells in quads:
                selected.append(np.vstack((cells[:, (0, 1, 2)], cells[:, (0, 2, 3)])).astype(np.int32))
        if not selected:
            available = sorted({target for target, _ in candidates})
            raise ValueError(f"No supported {requested or ''} cells found in {filename}; available blocks: {available}")
        cells = np.vstack(selected)
        node_sets = {
            str(set_name): np.asarray(indices, dtype=np.int32)
            for set_name, indices in dict(imported.point_sets or {}).items()
            if indices is not None
        }
        # Meshio exposes Gmsh physical surfaces and Abaqus element sets as
        # cell sets.  Promote their incident nodes so the same name can be
        # passed directly to DirichletBoundary/NeumannBoundary.
        for set_name, type_sets in getattr(imported, "cell_sets_dict", {}).items():
            incident = []
            for imported_type, indices in type_sets.items():
                if imported_type not in imported.cells_dict:
                    continue
                block = np.asarray(imported.cells_dict[imported_type])
                indices = np.asarray(indices, dtype=np.int32)
                if indices.size:
                    incident.append(block[indices].reshape(-1))
            if incident:
                node_sets.setdefault(str(set_name), np.unique(np.concatenate(incident)))
        material_points = None
        for key in ("material_points", "texture_coordinates", "TextureCoordinates", "uv"):
            if key in imported.point_data:
                material_points = np.asarray(imported.point_data[key], dtype=np.float64)
                break
        original_points = np.asarray(imported.points)
        used_points = np.unique(cells)
        points, cells, node_sets = _compact(original_points, cells, node_sets)
        if material_points is not None:
            if material_points.shape[0] == original_points.shape[0] and points.shape[0] != original_points.shape[0]:
                material_points = material_points[used_points]
        return FEMMesh(
            points,
            cells,
            requested,
            node_sets=node_sets,
            name=name or filename.stem,
            material_points=material_points,
        )

    load = read


__all__ = ["FEMGenerateManager"]
