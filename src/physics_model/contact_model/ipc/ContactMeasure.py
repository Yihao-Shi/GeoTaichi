"""Geometry-only quadrature measures for IPC contact stencils.

The routines in this module intentionally do not include IPC time-step,
barrier-width, or material coefficients.  A contact record stores a geometric
measure and the constitutive model applies its own scaling exactly once.
"""

from __future__ import annotations

import numpy as np
import taichi as ti


def triangle_area(vertices):
    """Return the area of one 3D triangle."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape != (3, 3):
        raise ValueError("triangle_area expects an array with shape (3, 3)")
    return 0.5 * float(
        np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
    )


def segment_length(vertices):
    """Return the length of one 2D or 3D boundary segment."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] != 2 or vertices.shape[1] not in (2, 3):
        raise ValueError("segment_length expects shape (2, 2) or (2, 3)")
    return float(np.linalg.norm(vertices[1] - vertices[0]))


def lumped_segment_vertex_measures(vertices, edges):
    """Lump half of every boundary-segment length to each endpoint."""
    vertices = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int32).reshape((-1, 2))
    measures = np.zeros(vertices.shape[0], dtype=np.float64)
    for edge in edges:
        measures[edge] += 0.5 * segment_length(vertices[edge])
    return measures


def lumped_boundary_vertex_measures(vertices, elements):
    """Dimension-independent FEM boundary vertex quadrature.

    Two-node elements are segments; three-node elements are surface triangles.
    """
    elements = np.asarray(elements, dtype=np.int32)
    if elements.ndim != 2 or elements.shape[1] not in (2, 3):
        raise ValueError("boundary elements must have shape (n, 2) or (n, 3)")
    if elements.shape[1] == 2:
        return lumped_segment_vertex_measures(vertices, elements)
    return lumped_vertex_measures(vertices, elements)


def lumped_vertex_measures(vertices, faces):
    """Barycentrically lump triangle areas onto their three vertices."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32).reshape((-1, 3))
    measures = np.zeros(vertices.shape[0], dtype=np.float64)
    for face in faces:
        measures[face] += triangle_area(vertices[face]) / 3.0
    return measures


def surface_edges_and_measures(vertices, faces):
    """Build unique undirected edges and their incident lumped face measure.

    Edge ordering follows first appearance in ``faces`` so existing engine
    candidate identifiers remain stable across the refactor.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32).reshape((-1, 3))
    edge_ids = {}
    edge_measures = []
    for face in faces:
        face_measure = triangle_area(vertices[face])
        for local_a, local_b in ((0, 1), (1, 2), (2, 0)):
            a = int(face[local_a])
            b = int(face[local_b])
            key = (min(a, b), max(a, b))
            if key not in edge_ids:
                edge_ids[key] = len(edge_ids)
                edge_measures.append(0.0)
            edge_measures[edge_ids[key]] += face_measure / 3.0
    edges = np.asarray(list(edge_ids.keys()), dtype=np.int32).reshape((-1, 2))
    return edges, np.asarray(edge_measures, dtype=np.float64)


def surface_vertex_edge_measures(vertices, faces):
    """Return ``(vertex_measure, edges, edge_measure)`` for a triangle mesh."""
    return (
        lumped_vertex_measures(vertices, faces),
        *surface_edges_and_measures(vertices, faces),
    )


def reference_point_measure(volume, dimension):
    """Fallback boundary measure inferred from a material-point volume.

    Explicit boundary quadrature or user-provided measures are preferable.
    This helper is kept explicit so callers cannot accidentally confuse a
    volume with a surface measure.
    """
    volume = float(volume)
    dimension = int(dimension)
    if not np.isfinite(volume) or volume < 0.0:
        raise ValueError("reference point volume must be finite and non-negative")
    if dimension not in (2, 3):
        raise ValueError("reference_point_measure supports dimension 2 or 3")
    return volume ** ((dimension - 1.0) / dimension)


@ti.func
def symmetric_contact_measure(measure_a, measure_b):
    """Symmetric quadrature weight for a pair of sampled primitives."""
    return 0.5 * (measure_a + measure_b)


@ti.func
def point_contact_measure(measure):
    """Clamp round-off-only negative point measures without hiding bad input."""
    return ti.max(measure, 0.0)


@ti.func
def curve_quadrature_measure(tangent, parameter_weight):
    """Physical IGA curve measure ``||dx/du|| du``."""
    return ti.max(parameter_weight, 0.0) * tangent.norm()


@ti.func
def surface_quadrature_measure(tangent_u, tangent_v, parameter_weight):
    """Physical IGA surface measure ``||dx/du x dx/dv|| du dv``."""
    return ti.max(parameter_weight, 0.0) * tangent_u.cross(tangent_v).norm()
