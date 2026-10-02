"""Reference-space operators for triangular cloth finite elements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClothElement:
    connectivity: np.ndarray
    inverse_reference_jacobian: np.ndarray
    inverse_reference_metric: np.ndarray
    integration_weight: np.ndarray
    characteristic_length: np.ndarray
    bending_connectivity: np.ndarray
    bending_stiffness: np.ndarray
    bending_edge_length: np.ndarray
    bending_height: np.ndarray
    bending_rest_angle: np.ndarray

    def compute_deformation_gradient(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        triangles = positions[self.connectivity]
        current_jacobian = np.stack(
            (triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=2,
        )
        return np.einsum("eij,ejk->eik", current_jacobian, self.inverse_reference_jacobian)

    def compute_metric_tensor(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        triangles = positions[self.connectivity]
        current_jacobian = np.stack(
            (triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=2,
        )
        current_metric = np.einsum("eki,ekj->eij", current_jacobian, current_jacobian)
        return np.einsum("eij,ejk->eik", current_metric, self.inverse_reference_metric)


def _compute_cotangent(first, second):
    sine = np.linalg.norm(np.cross(first, second))
    if sine <= 1.0e-14 * max(np.linalg.norm(first) * np.linalg.norm(second), 1.0):
        raise ValueError("cloth bending element contains a degenerate reference triangle")
    return float(np.dot(first, second) / sine)


def _signed_dihedral_angle(points):
    x0, x1, x2, x3 = points
    normal0 = np.cross(x1 - x0, x2 - x0)
    normal1 = np.cross(x2 - x3, x1 - x3)
    denominator = np.linalg.norm(normal0) * np.linalg.norm(normal1)
    if denominator <= 1.0e-28:
        raise ValueError("cloth dihedral bending contains a degenerate hinge")
    cosine = np.clip(np.dot(normal0, normal1) / denominator, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if np.dot(np.cross(normal1, normal0), x1 - x2) < 0.0:
        angle = -angle
    return angle


def _build_bending_operator(mesh, bending_modulus):
    if bending_modulus <= 0.0:
        return (
            np.empty((0, 4), dtype=np.int32),
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    half_edges = {}
    connectivity = []
    stiffness = []
    edge_length = []
    height = []
    rest_angle = []
    reference = mesh.rest_shape
    for triangle in mesh.cells:
        for local_id in range(3):
            first = int(triangle[local_id])
            second = int(triangle[(local_id + 1) % 3])
            opposite = int(triangle[(local_id + 2) % 3])
            reverse = (second, first)
            if reverse not in half_edges:
                half_edges[(first, second)] = opposite
                continue
            other = half_edges.pop(reverse)
            bending_nodes = np.asarray((opposite, first, second, other), dtype=np.int32)
            x0, x1, x2, x3 = reference[bending_nodes]
            edge0 = x2 - x1
            edge1 = x0 - x1
            edge2 = x3 - x1
            edge3 = x0 - x2
            edge4 = x3 - x2
            area0 = 0.5 * np.linalg.norm(np.cross(edge0, edge1))
            area1 = 0.5 * np.linalg.norm(np.cross(edge0, edge2))
            if area0 <= 0.0 or area1 <= 0.0:
                raise ValueError("cloth bending element contains a zero-area reference triangle")
            c01 = _compute_cotangent(edge0, edge1)
            c02 = _compute_cotangent(edge0, edge2)
            c03 = _compute_cotangent(-edge0, edge3)
            c04 = _compute_cotangent(-edge0, edge4)
            coefficients = np.asarray(
                (-c01 - c03, c03 + c04, c01 + c02, -c02 - c04),
                dtype=np.float64,
            )
            factor = bending_modulus * 3.0 / (2.0 * (area0 + area1))
            connectivity.append(bending_nodes)
            stiffness.append(factor * np.outer(coefficients, coefficients))
            hinge_length = np.linalg.norm(edge0)
            edge_length.append(hinge_length)
            height.append((2.0 * area0 + 2.0 * area1) / (hinge_length * 6.0))
            rest_angle.append(_signed_dihedral_angle((x0, x1, x2, x3)))
    if not connectivity:
        return (
            np.empty((0, 4), dtype=np.int32),
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.ascontiguousarray(connectivity),
        np.ascontiguousarray(stiffness),
        np.ascontiguousarray(edge_length),
        np.ascontiguousarray(height),
        np.ascontiguousarray(rest_angle),
    )


def create_cloth_element(mesh, material):
    """Construct membrane integration data and both bending operators."""
    if mesh.cell_type != "triangle":
        raise ValueError("cloth FEM requires a TRI3 surface mesh")
    count = mesh.number_of_cells
    inverse_jacobian = np.empty((count, 2, 2), dtype=np.float64)
    inverse_metric = np.empty((count, 2, 2), dtype=np.float64)
    weights = np.empty(count, dtype=np.float64)
    lengths = np.empty(count, dtype=np.float64)
    for element_id, connectivity in enumerate(mesh.cells):
        material_triangle = mesh.material_points[mesh.material_cells[element_id]]
        edge01 = material_triangle[1] - material_triangle[0]
        edge02 = material_triangle[2] - material_triangle[0]
        length01 = np.linalg.norm(edge01)
        double_area = np.linalg.norm(np.cross(edge01, edge02))
        scale = max(length01, np.linalg.norm(edge02), 1.0)
        if length01 <= 1.0e-13 * scale or double_area <= 1.0e-13 * scale**2:
            raise ValueError(f"cloth triangle {element_id} has degenerate material coordinates")
        metric = np.asarray(
            (
                (np.dot(edge01, edge01), np.dot(edge01, edge02)),
                (np.dot(edge01, edge02), np.dot(edge02, edge02)),
            )
        )
        edge_matrix = np.asarray(
            (
                (length01, np.dot(edge01, edge02) / length01),
                (0.0, double_area / length01),
            )
        )
        inverse_jacobian[element_id] = np.linalg.inv(edge_matrix)
        inverse_metric[element_id] = np.linalg.inv(metric)
        weights[element_id] = 0.5 * double_area * material.thickness
        lengths[element_id] = max(length01, np.linalg.norm(edge02), np.linalg.norm(edge02 - edge01))
    (
        bending_connectivity,
        bending_stiffness,
        bending_edge_length,
        bending_height,
        bending_rest_angle,
    ) = _build_bending_operator(mesh, material.quadratic_bending_modulus)
    return ClothElement(
        np.ascontiguousarray(mesh.cells, dtype=np.int32),
        np.ascontiguousarray(inverse_jacobian),
        np.ascontiguousarray(inverse_metric),
        np.ascontiguousarray(weights),
        np.ascontiguousarray(lengths),
        bending_connectivity,
        bending_stiffness,
        bending_edge_length,
        bending_height,
        bending_rest_angle,
    )


__all__ = ["ClothElement", "create_cloth_element"]
