"""Surface extraction and broad-phase candidates for FEM contact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactSurface:
    faces: np.ndarray
    edges: np.ndarray
    vertices: np.ndarray
    node_area: np.ndarray
    edge_area: np.ndarray
    face_body: np.ndarray
    edge_body: np.ndarray
    vertex_body: np.ndarray


def _triangulated_boundary(mesh):
    if mesh.cell_type == "triangle":
        return (
            np.ascontiguousarray(mesh.cells, dtype=np.int32),
            np.ascontiguousarray(mesh.cell_body_ids, dtype=np.int32),
        )
    facets, owners = mesh.boundary_facets()
    body = mesh.cell_body_ids[owners]
    if facets.shape[1] == 3:
        return (
            np.ascontiguousarray(facets, dtype=np.int32),
            np.ascontiguousarray(body, dtype=np.int32),
        )
    triangles = np.empty((2 * facets.shape[0], 3), dtype=np.int32)
    triangles[0::2] = facets[:, (0, 1, 2)]
    triangles[1::2] = facets[:, (0, 2, 3)]
    return triangles, np.repeat(body, 2).astype(np.int32)


def build_contact_surface(mesh, body_ids=None):
    """Build the triangulated surface and lumped primitive area measures."""
    faces, face_body = _triangulated_boundary(mesh)
    if body_ids is not None:
        selected = np.isin(face_body, np.asarray(tuple(body_ids), dtype=np.int32))
        faces = np.ascontiguousarray(faces[selected])
        face_body = np.ascontiguousarray(face_body[selected])
        if faces.shape[0] == 0:
            raise ValueError("FEM contact body selection contains no surface faces")
    edge_map = {}
    edges = []
    edge_body = []
    for face, body_id in zip(faces, face_body):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(first), int(second))))
            if key not in edge_map:
                edge_map[key] = len(edges)
                edges.append(key)
                edge_body.append(int(body_id))
            elif edge_body[edge_map[key]] != int(body_id):
                raise ValueError("a FEM contact edge cannot belong to different bodies")
    edges = np.asarray(edges, dtype=np.int32)
    edge_body = np.asarray(edge_body, dtype=np.int32)
    node_area = np.zeros(mesh.number_of_nodes, dtype=np.float64)
    edge_area = np.zeros(edges.shape[0], dtype=np.float64)
    for face in faces:
        points = mesh.rest_shape[face]
        area = 0.5 * float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))
        if area <= 0.0:
            raise ValueError("FEM contact surface contains a degenerate triangle")
        node_area[face] += area / 3.0
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_area[edge_map[tuple(sorted((int(first), int(second))))]] += area / 3.0
    vertices = np.unique(faces).astype(np.int32)
    vertex_body = np.ascontiguousarray(mesh.node_body_ids[vertices], dtype=np.int32)
    return ContactSurface(
        faces,
        edges,
        vertices,
        node_area,
        edge_area,
        np.ascontiguousarray(face_body, dtype=np.int32),
        edge_body,
        vertex_body,
    )


def _boxes(points, primitives, end_points=None):
    start = points[primitives]
    lower = np.min(start, axis=1)
    upper = np.max(start, axis=1)
    if end_points is not None:
        end = end_points[primitives]
        lower = np.minimum(lower, np.min(end, axis=1))
        upper = np.maximum(upper, np.max(end, axis=1))
    return lower, upper


def _reference_broad_phase_candidates(surface, positions, radius, end_positions=None):
    """Quadratic host oracle retained only for broad-phase verification."""
    positions = np.asarray(positions, dtype=np.float64)
    end_positions = None if end_positions is None else np.asarray(end_positions, dtype=np.float64)
    radius = float(radius)
    faces = surface.faces
    edges = surface.edges
    face_lower, face_upper = _boxes(positions, faces, end_positions)
    edge_lower, edge_upper = _boxes(positions, edges, end_positions)

    point_triangle = []
    for vertex in surface.vertices:
        lower = positions[vertex].copy()
        upper = lower.copy()
        if end_positions is not None:
            lower = np.minimum(lower, end_positions[vertex])
            upper = np.maximum(upper, end_positions[vertex])
        overlap = np.all(face_upper + radius >= lower, axis=1) & np.all(face_lower - radius <= upper, axis=1)
        incident = np.any(faces == vertex, axis=1)
        for face_id in np.flatnonzero(overlap & ~incident):
            point_triangle.append((int(vertex), *map(int, faces[face_id])))

    edge_edge = []
    for first in range(edges.shape[0]):
        overlap = np.all(edge_upper[first + 1 :] + radius >= edge_lower[first], axis=1)
        overlap &= np.all(edge_lower[first + 1 :] - radius <= edge_upper[first], axis=1)
        second_ids = np.flatnonzero(overlap) + first + 1
        for second in second_ids:
            if np.intersect1d(edges[first], edges[second], assume_unique=True).size == 0:
                edge_edge.append((*map(int, edges[first]), *map(int, edges[second])))

    pt = np.asarray(point_triangle, dtype=np.int32).reshape(-1, 4)
    ee = np.asarray(edge_edge, dtype=np.int32).reshape(-1, 4)
    return pt, ee


def broad_phase_candidates(surface, positions, radius, end_positions=None, broad_phase="LinkedCell"):
    """Build PT/EE candidates with a selected dynamic Taichi backend.

    This convenience adapter accepts and returns NumPy arrays at its public
    API boundary.  The actual current/swept AABB insertion, count, prefix
    sums, candidate counting and fill are device kernels.  Production FEM
    keeps one selected broad-phase instance and does not perform these
    input/output copies.
    """
    import taichi as ti

    key = str(broad_phase).strip().replace("_", "").replace("-", "").replace(" ", "").lower()
    if key in ("linkedcell", "cell", "dynamiclinkedcell"):
        from src.fem.contact.LinkedCellBroadPhase import (
            DynamicLinkedCellBroadPhase as BroadPhase,
        )
    elif key in ("bvh", "boundingvolumehierarchy"):
        from src.fem.contact.BVHBroadPhase import (
            DynamicBVHBroadPhase as BroadPhase,
        )
    else:
        raise ValueError("broad_phase must be 'LinkedCell' or 'BVH'")

    if ti.lang.impl.get_runtime().prog is None:
        raise RuntimeError("broad_phase_candidates requires an initialized Taichi runtime")
    real_type = ti.lang.impl.current_cfg().default_fp
    numpy_type = np.float64 if real_type == ti.f64 else np.float32
    positions = np.ascontiguousarray(positions, dtype=numpy_type)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("FEM contact positions must have shape (n, 3)")
    position_field = ti.Vector.field(3, dtype=real_type, shape=positions.shape[0])
    position_field.from_numpy(positions)
    end_field = None
    if end_positions is not None:
        end_positions = np.ascontiguousarray(end_positions, dtype=numpy_type)
        if end_positions.shape != positions.shape:
            raise ValueError("swept FEM contact endpoints must match positions")
        end_field = ti.Vector.field(
            3,
            dtype=real_type,
            shape=end_positions.shape[0],
        )
        end_field.from_numpy(end_positions)
    broad_phase = BroadPhase(
        surface.faces,
        surface.edges,
        surface.vertices,
        surface.node_area,
        surface.edge_area,
        positions,
    )
    point_triangle_count, edge_edge_count = broad_phase.rebuild(
        position_field,
        float(radius),
        end_positions=end_field,
    )
    return (
        broad_phase.point_triangle.to_numpy()[:point_triangle_count].copy(),
        broad_phase.edge_edge.to_numpy()[:edge_edge_count].copy(),
    )


__all__ = ["ContactSurface", "broad_phase_candidates", "build_contact_surface"]
