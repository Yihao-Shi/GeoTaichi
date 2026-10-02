"""Finite-element mesh container and geometric selection utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np


_CELL_ALIASES = {
    "tet4": "tetra",
    "tetra4": "tetra",
    "tetrahedron": "tetra",
    "tetra": "tetra",
    "hex8": "hexahedron",
    "hexa8": "hexahedron",
    "hexahedron": "hexahedron",
    "hex": "hexahedron",
    "tri3": "triangle",
    "triangle3": "triangle",
    "triangle": "triangle",
    "membrane3": "triangle",
}


def normalize_cell_type(cell_type: str) -> str:
    key = str(cell_type).strip().replace("_", "").replace("-", "").lower()
    if key not in _CELL_ALIASES:
        supported = "TET4, HEX8, and TRI3"
        raise ValueError(f"Unsupported FEM cell type {cell_type!r}; expected {supported}")
    return _CELL_ALIASES[key]


def _as_points(points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("mesh points must have shape (number_of_nodes, 2 or 3)")
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(points.shape[0])))
    if not np.all(np.isfinite(points)):
        raise ValueError("mesh points must be finite")
    return np.ascontiguousarray(points)


@dataclass
class FEMMesh:
    """A single compatible FEM cell block.

    GeoTaichi's first FEM backend deliberately keeps one formulation per mesh:
    volume meshes use TET4 or HEX8, while surface meshes use TRI3 membranes.
    Multiple bodies can be represented by concatenating compatible blocks.
    """

    points: np.ndarray
    cells: np.ndarray
    cell_type: str
    node_sets: Dict[str, np.ndarray] = field(default_factory=dict)
    cell_sets: Dict[str, np.ndarray] = field(default_factory=dict)
    name: str = "fem_mesh"
    material_points: Optional[np.ndarray] = None
    material_cells: Optional[np.ndarray] = None
    rest_shape: Optional[np.ndarray] = None
    cell_body_ids: Optional[np.ndarray] = None

    def __post_init__(self):
        self.points = _as_points(self.points)
        if self.rest_shape is None:
            self.rest_shape = self.points.copy()
        else:
            self.rest_shape = _as_points(self.rest_shape)
            if self.rest_shape.shape != self.points.shape:
                raise ValueError("rest_shape must contain one coordinate per mesh node")
        self.cell_type = normalize_cell_type(self.cell_type)
        expected_nodes = {"tetra": 4, "hexahedron": 8, "triangle": 3}[self.cell_type]
        self.cells = np.asarray(self.cells, dtype=np.int32)
        if self.cells.ndim != 2 or self.cells.shape[1] != expected_nodes:
            raise ValueError(f"{self.cell_type} connectivity must have shape (number_of_cells, {expected_nodes})")
        if self.cells.size == 0:
            raise ValueError("FEM mesh must contain at least one cell")
        if np.min(self.cells) < 0 or np.max(self.cells) >= self.points.shape[0]:
            raise ValueError("cell connectivity contains an invalid node index")
        self.cells = np.ascontiguousarray(self.cells)
        self._material_coordinates_follow_rest_shape = self.material_points is None
        if self.material_points is None:
            self.material_points = self.rest_shape.copy()
        else:
            self.material_points = _as_points(self.material_points)
        if self.material_cells is None:
            self.material_cells = self.cells.copy()
        else:
            self.material_cells = np.ascontiguousarray(self.material_cells, dtype=np.int32)
            if self.material_cells.shape != self.cells.shape:
                raise ValueError("material_cells must contain one material-coordinate index per cell corner")
        if np.min(self.material_cells) < 0 or np.max(self.material_cells) >= self.material_points.shape[0]:
            raise ValueError("material_cells contains an invalid material-coordinate index")
        self.node_sets = self._normalize_sets(self.node_sets, self.number_of_nodes, "node")
        self.cell_sets = self._normalize_sets(self.cell_sets, self.number_of_cells, "cell")
        self._orient_tetrahedra()
        self._initialize_body_ids()
        self._add_geometric_node_sets()

    @staticmethod
    def _normalize_sets(sets, upper_bound: int, label: str):
        normalized = {}
        for name, indices in dict(sets or {}).items():
            values = np.unique(np.asarray(indices, dtype=np.int32).reshape(-1))
            if values.size and (values[0] < 0 or values[-1] >= upper_bound):
                raise ValueError(f"{label} set {name!r} contains an invalid index")
            normalized[str(name)] = values
        return normalized

    @property
    def number_of_nodes(self) -> int:
        return int(self.points.shape[0])

    @property
    def number_of_cells(self) -> int:
        return int(self.cells.shape[0])

    @property
    def is_volume(self) -> bool:
        return self.cell_type in ("tetra", "hexahedron")

    @property
    def is_membrane(self) -> bool:
        return self.cell_type == "triangle"

    @property
    def bounds(self):
        return np.min(self.points, axis=0), np.max(self.points, axis=0)

    def copy(self) -> "FEMMesh":
        copied = FEMMesh(
            self.points.copy(),
            self.cells.copy(),
            self.cell_type,
            {name: values.copy() for name, values in self.node_sets.items()},
            {name: values.copy() for name, values in self.cell_sets.items()},
            self.name,
            self.material_points.copy(),
            self.material_cells.copy(),
            self.rest_shape.copy(),
            self.cell_body_ids.copy(),
        )
        copied._material_coordinates_follow_rest_shape = self._material_coordinates_follow_rest_shape
        return copied

    @staticmethod
    def concatenate(meshes, name="fem_soft_particles"):
        """Combine compatible disconnected meshes into independent FEM bodies."""
        meshes = tuple(meshes)
        if not meshes:
            raise ValueError("FEMMesh.concatenate requires at least one mesh")
        cell_type = meshes[0].cell_type
        if any(mesh.cell_type != cell_type for mesh in meshes):
            raise ValueError("concatenated FEM meshes must use the same cell type")
        points = []
        rest_shape = []
        material_points = []
        cells = []
        material_cells = []
        body_ids = []
        node_sets = {}
        cell_sets = {}
        node_offset = 0
        material_offset = 0
        cell_offset = 0
        body_offset = 0
        for mesh_id, mesh in enumerate(meshes):
            points.append(mesh.points)
            rest_shape.append(mesh.rest_shape)
            material_points.append(mesh.material_points)
            cells.append(mesh.cells + node_offset)
            material_cells.append(mesh.material_cells + material_offset)
            local_body = np.asarray(mesh.cell_body_ids, dtype=np.int32)
            body_ids.append(local_body + body_offset)
            for set_name, indices in mesh.node_sets.items():
                node_sets[f"body{mesh_id}:{set_name}"] = indices + node_offset
            for set_name, indices in mesh.cell_sets.items():
                cell_sets[f"body{mesh_id}:{set_name}"] = indices + cell_offset
            node_offset += mesh.number_of_nodes
            material_offset += mesh.material_points.shape[0]
            cell_offset += mesh.number_of_cells
            body_offset += int(np.max(local_body)) + 1
        return FEMMesh(
            np.ascontiguousarray(np.concatenate(points, axis=0)),
            np.ascontiguousarray(np.concatenate(cells, axis=0)),
            cell_type,
            node_sets=node_sets,
            cell_sets=cell_sets,
            name=str(name),
            material_points=np.ascontiguousarray(np.concatenate(material_points, axis=0)),
            material_cells=np.ascontiguousarray(np.concatenate(material_cells, axis=0)),
            rest_shape=np.ascontiguousarray(np.concatenate(rest_shape, axis=0)),
            cell_body_ids=np.ascontiguousarray(np.concatenate(body_ids, axis=0)),
        )

    @property
    def rest_points(self):
        return self.rest_shape

    def set_rest_shape(self, rest_shape, *, update_material_coordinates=False):
        rest_shape = _as_points(rest_shape)
        if rest_shape.shape != self.points.shape:
            raise ValueError("rest_shape must contain one coordinate per mesh node")
        self.rest_shape = rest_shape
        if update_material_coordinates or self._material_coordinates_follow_rest_shape:
            self.material_points = rest_shape.copy()
            self.material_cells = self.cells.copy()
            self._material_coordinates_follow_rest_shape = True
        self._orient_tetrahedra()
        return self

    def _initialize_body_ids(self):
        if self.cell_body_ids is None:
            parent = np.arange(self.number_of_cells, dtype=np.int32)
            first_owner = {}

            def find(index):
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = int(parent[index])
                return index

            def union(first, second):
                first_root, second_root = find(first), find(second)
                if first_root != second_root:
                    parent[second_root] = first_root

            for cell_id, cell in enumerate(self.cells):
                for node in cell:
                    node = int(node)
                    if node in first_owner:
                        union(cell_id, first_owner[node])
                    else:
                        first_owner[node] = cell_id
            roots = [find(cell_id) for cell_id in range(self.number_of_cells)]
            labels = {root: body_id for body_id, root in enumerate(sorted(set(roots)))}
            self.cell_body_ids = np.asarray([labels[root] for root in roots], dtype=np.int32)
        else:
            self.cell_body_ids = np.asarray(self.cell_body_ids, dtype=np.int32).reshape(-1)
            if self.cell_body_ids.shape != (self.number_of_cells,):
                raise ValueError("cell_body_ids must contain one body ID per FEM cell")
            if np.any(self.cell_body_ids < 0):
                raise ValueError("FEM body IDs must be non-negative")
            self.cell_body_ids = np.ascontiguousarray(self.cell_body_ids)

        self.node_body_ids = np.full(self.number_of_nodes, -1, dtype=np.int32)
        for cell_id, cell in enumerate(self.cells):
            body_id = int(self.cell_body_ids[cell_id])
            for node in cell:
                node = int(node)
                previous = int(self.node_body_ids[node])
                if previous >= 0 and previous != body_id:
                    raise ValueError("cells with different FEM body IDs cannot share nodes")
                self.node_body_ids[node] = body_id
        if np.any(self.node_body_ids < 0):
            raise ValueError("FEM mesh contains a node that belongs to no body")

    @property
    def body_ids(self):
        return np.unique(self.cell_body_ids)

    def _orient_tetrahedra(self):
        if self.cell_type != "tetra":
            return
        vertices = self.rest_shape[self.cells]
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
        scale = max(float(np.ptp(self.rest_shape, axis=0).max()), 1.0)
        if np.any(np.abs(determinants) <= 1.0e-13 * scale**3):
            raise ValueError("tetrahedral mesh contains a degenerate element")
        negative = determinants < 0.0
        if np.any(negative):
            old_one = self.cells[negative, 1].copy()
            self.cells[negative, 1] = self.cells[negative, 2]
            self.cells[negative, 2] = old_one
            old_material_one = self.material_cells[negative, 1].copy()
            self.material_cells[negative, 1] = self.material_cells[negative, 2]
            self.material_cells[negative, 2] = old_material_one

    def _add_geometric_node_sets(self):
        lower, upper = self.bounds
        scale = max(float(np.max(upper - lower)), 1.0)
        tolerance = 1.0e-10 * scale
        axes = "xyz"
        for axis, name in enumerate(axes):
            self.node_sets.setdefault(
                f"{name}min",
                np.flatnonzero(np.isclose(self.points[:, axis], lower[axis], atol=tolerance, rtol=0.0)),
            )
            self.node_sets.setdefault(
                f"{name}max",
                np.flatnonzero(np.isclose(self.points[:, axis], upper[axis], atol=tolerance, rtol=0.0)),
            )
        boundary_nodes = np.unique(self.boundary_facets()[0].reshape(-1))
        self.node_sets.setdefault("boundary", boundary_nodes.astype(np.int32))
        self.node_sets.setdefault("all", np.arange(self.number_of_nodes, dtype=np.int32))

    def select_nodes(
        self,
        node_set: Optional[str] = None,
        selector: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        axis: Optional[object] = None,
        value: Optional[float] = None,
        bounds: Optional[Iterable[Iterable[float]]] = None,
        tolerance: Optional[float] = None,
    ) -> np.ndarray:
        """Select nodes by a named set, predicate, coordinate plane, or box."""
        choices = sum(item is not None for item in (node_set, selector, axis, bounds))
        if choices != 1:
            raise ValueError("select_nodes expects exactly one selection method")
        if node_set is not None:
            if node_set not in self.node_sets:
                raise KeyError(f"Unknown node set {node_set!r}; available sets: {sorted(self.node_sets)}")
            return self.node_sets[node_set].copy()
        if selector is not None:
            selected = np.asarray(selector(self.points))
            if selected.dtype == bool:
                if selected.shape != (self.number_of_nodes,):
                    raise ValueError("node selector must return one boolean per mesh node")
                return np.flatnonzero(selected).astype(np.int32)
            return np.unique(selected.astype(np.int32).reshape(-1))
        lower, upper = self.bounds
        scale = max(float(np.max(upper - lower)), 1.0)
        tolerance = 1.0e-9 * scale if tolerance is None else float(tolerance)
        if axis is not None:
            axis_id = {"x": 0, "y": 1, "z": 2}.get(str(axis).lower(), axis)
            if axis_id not in (0, 1, 2) or value is None:
                raise ValueError("axis selection needs axis='x'/'y'/'z' and a coordinate value")
            return np.flatnonzero(
                np.isclose(self.points[:, int(axis_id)], float(value), atol=tolerance, rtol=0.0)
            ).astype(np.int32)
        box = np.asarray(bounds, dtype=np.float64)
        if box.shape != (2, 3):
            raise ValueError("selection bounds must be [[xmin, ymin, zmin], [xmax, ymax, zmax]]")
        mask = np.all(self.points >= box[0] - tolerance, axis=1) & np.all(self.points <= box[1] + tolerance, axis=1)
        return np.flatnonzero(mask).astype(np.int32)

    def boundary_facets(self):
        """Return boundary connectivity and the owning cell for each facet."""
        if self.cell_type == "triangle":
            local_facets = ((0, 1), (1, 2), (2, 0))
        elif self.cell_type == "tetra":
            local_facets = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
        else:
            local_facets = (
                (0, 3, 2, 1),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
            )
        owners = {}
        for cell_id, cell in enumerate(self.cells):
            for local in local_facets:
                facet = tuple(int(cell[index]) for index in local)
                key = tuple(sorted(facet))
                if key in owners:
                    owners[key] = None
                else:
                    owners[key] = (facet, cell_id)
        boundary = [item for item in owners.values() if item is not None]
        facets = np.asarray([item[0] for item in boundary], dtype=np.int32)
        cells = np.asarray([item[1] for item in boundary], dtype=np.int32)
        return facets, cells

    def write(self, filename, point_data=None, cell_data=None):
        """Write the mesh and optional FEM fields through meshio."""
        import meshio

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized_cell_data = None
        if cell_data:
            normalized_cell_data = {name: [np.asarray(values)] for name, values in cell_data.items()}
        meshio.write(
            path,
            meshio.Mesh(
                self.points,
                [(self.cell_type, self.cells)],
                point_data=dict(point_data or {}),
                cell_data=normalized_cell_data,
            ),
        )


__all__ = ["FEMMesh", "normalize_cell_type"]
