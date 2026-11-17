from __future__ import annotations
from src.fem.elements.Element import ElementData
from src.fem.elements.Element import _HEX_SIGNS
import numpy as np
from src.fem.generator.Mesh import FEMMesh


def _tetrahedron_data(mesh: FEMMesh) -> ElementData:
    natural_gradients = np.asarray(
        ((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    gradients = np.empty((mesh.number_of_cells, 1, 4, 3), dtype=np.float64)
    weights = np.empty((mesh.number_of_cells, 1), dtype=np.float64)
    for element_id, connectivity in enumerate(mesh.cells):
        reference = mesh.rest_shape[connectivity]
        jacobian = reference.T @ natural_gradients
        determinant = float(np.linalg.det(jacobian))
        if determinant <= 0.0:
            raise ValueError(f"TET4 element {element_id} has a non-positive reference Jacobian")
        gradients[element_id, 0] = natural_gradients @ np.linalg.inv(jacobian)
        weights[element_id, 0] = determinant / 6.0
    return ElementData(
        mesh.cells,
        np.asarray(((0.25, 0.25, 0.25, 0.25),), dtype=np.float64),
        gradients,
        weights,
        3,
        "TET4",
    )


def _hex_shape(natural_coordinates):
    shifted = 1.0 + _HEX_SIGNS * np.asarray(natural_coordinates, dtype=np.float64)
    values = 0.125 * np.prod(shifted, axis=1)
    gradients = np.empty((8, 3), dtype=np.float64)
    for direction in range(3):
        other = [index for index in range(3) if index != direction]
        gradients[:, direction] = 0.125 * _HEX_SIGNS[:, direction] * shifted[:, other[0]] * shifted[:, other[1]]
    return values, gradients


def _hexahedron_data(mesh: FEMMesh) -> ElementData:
    location = 1.0 / np.sqrt(3.0)
    quadrature = np.asarray(
        [
            (xi, eta, zeta)
            for xi in (-location, location)
            for eta in (-location, location)
            for zeta in (-location, location)
        ],
        dtype=np.float64,
    )
    shape_values = np.empty((8, 8), dtype=np.float64)
    natural_gradients = np.empty((8, 8, 3), dtype=np.float64)
    for quadrature_id, coordinates in enumerate(quadrature):
        shape_values[quadrature_id], natural_gradients[quadrature_id] = _hex_shape(coordinates)
    gradients = np.empty((mesh.number_of_cells, 8, 8, 3), dtype=np.float64)
    weights = np.empty((mesh.number_of_cells, 8), dtype=np.float64)
    for element_id, connectivity in enumerate(mesh.cells):
        reference = mesh.rest_shape[connectivity]
        for quadrature_id in range(8):
            jacobian = reference.T @ natural_gradients[quadrature_id]
            determinant = float(np.linalg.det(jacobian))
            if determinant <= 0.0:
                raise ValueError(
                    f"HEX8 element {element_id} has a non-positive reference Jacobian at Gauss point {quadrature_id}"
                )
            gradients[element_id, quadrature_id] = natural_gradients[quadrature_id] @ np.linalg.inv(jacobian)
            weights[element_id, quadrature_id] = determinant
    return ElementData(mesh.cells, shape_values, gradients, weights, 3, "HEX8")


def _triangle_membrane_data(
    mesh: FEMMesh,
    thickness: float,
    formulation: str,
    *,
    axis_offset=0.0,
) -> ElementData:
    natural_gradients = np.asarray(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    gradients = np.empty((mesh.number_of_cells, 1, 3, 2), dtype=np.float64)
    weights = np.empty((mesh.number_of_cells, 1), dtype=np.float64)
    reference_frames = np.empty((mesh.number_of_cells, 3, 3), dtype=np.float64)
    reference_radii = (
        np.empty((mesh.number_of_cells, 1), dtype=np.float64) if formulation == "axisymmetric_solid" else None
    )
    for element_id, connectivity in enumerate(mesh.cells):
        # Classical TRI3 builds its reference Jacobian from physical reference
        # positions in a local orthonormal
        # frame. Only the separately selected cloth formulation uses material
        # coordinates, which may carry an independent OBJ texture topology.
        if formulation == "surface_cloth":
            reference = mesh.material_points[mesh.material_cells[element_id]]
        else:
            reference = mesh.rest_shape[connectivity]
        if formulation == "axisymmetric_solid":
            meridian_scale = max(
                float(np.ptp(reference[:, 0])),
                float(np.ptp(reference[:, 1])),
                1.0,
            )
            if float(np.ptp(reference[:, 2])) > 1.0e-12 * meridian_scale:
                raise ValueError(
                    "axisymmetric TRI3 coordinates must lie in an r-z plane "
                    "stored in the first two coordinate components"
                )
        edge_01 = reference[1] - reference[0]
        edge_02 = reference[2] - reference[0]
        length_01 = np.linalg.norm(edge_01)
        normal = np.cross(edge_01, edge_02)
        double_area = np.linalg.norm(normal)
        scale = max(np.linalg.norm(edge_01), np.linalg.norm(edge_02), 1.0)
        if length_01 <= 1.0e-13 * scale or double_area <= 1.0e-13 * scale**2:
            raise ValueError(f"TRI3 membrane element {element_id} is degenerate")
        basis_1 = edge_01 / length_01
        basis_3 = normal / double_area
        basis_2 = np.cross(basis_3, basis_1)
        reference_frames[element_id] = np.stack((basis_1, basis_2, basis_3))
        if formulation == "axisymmetric_solid":
            # Axisymmetric kinematics use the fixed global (r, z) material
            # axes.  A per-triangle co-rotated frame would incorrectly turn
            # the identity reference map into a rotation on differently
            # oriented triangles.
            reference_jacobian = reference[:, :2].T @ natural_gradients
            if np.linalg.det(reference_jacobian) <= 0.0:
                raise ValueError("axisymmetric TRI3 connectivity must be positively " "oriented in the r-z plane")
            gradients[element_id, 0] = natural_gradients @ np.linalg.inv(reference_jacobian)
            reference_radius = float(np.mean(reference[:, 0]) - axis_offset)
            if reference_radius <= 0.0:
                raise ValueError("axisymmetric TRI3 quadrature requires radius > axis_offset")
            reference_radii[element_id, 0] = reference_radius
            weights[element_id, 0] = 0.5 * double_area * reference_radius
        else:
            local_02 = np.asarray((np.dot(edge_02, basis_1), np.dot(edge_02, basis_2)))
            edge_matrix = np.asarray(((length_01, local_02[0]), (0.0, local_02[1])))
            gradients[element_id, 0] = natural_gradients @ np.linalg.inv(edge_matrix)
            weights[element_id, 0] = 0.5 * double_area * thickness
    return ElementData(
        mesh.cells,
        np.asarray(((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),), dtype=np.float64),
        gradients,
        weights,
        2,
        (
            "TRI3 axisymmetric solid"
            if formulation == "axisymmetric_solid"
            else ("TRI3 classical membrane" if formulation == "classical_membrane" else "TRI3 surface cloth")
        ),
        formulation,
        reference_frames,
        reference_radii,
        float(axis_offset),
    )


def create_element(
    mesh: FEMMesh,
    thickness=1.0,
    formulation="classical",
    *,
    axisymmetric=False,
    axis_offset=0.0,
) -> ElementData:
    thickness = float(thickness)
    if thickness <= 0.0:
        raise ValueError("membrane thickness must be positive")
    if mesh.cell_type == "tetra":
        return _tetrahedron_data(mesh)
    if mesh.cell_type == "hexahedron":
        return _hexahedron_data(mesh)
    if mesh.cell_type == "triangle":
        if axisymmetric:
            return _triangle_membrane_data(
                mesh,
                thickness,
                "axisymmetric_solid",
                axis_offset=axis_offset,
            )
        normalized = str(formulation).strip().replace("-", "_").lower()
        if normalized in ("classical", "standard", "classical_membrane"):
            normalized = "classical_membrane"
        elif normalized in ("cloth", "surface", "surface_cloth"):
            normalized = "surface_cloth"
        else:
            raise ValueError("TRI3 formulation must be 'classical' or 'surface_cloth'")
        return _triangle_membrane_data(mesh, thickness, normalized)
    raise ValueError(f"Unsupported FEM mesh type {mesh.cell_type!r}")
