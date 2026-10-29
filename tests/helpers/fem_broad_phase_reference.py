"""Quadratic NumPy broad-phase oracle for tests only."""

import numpy as np


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
