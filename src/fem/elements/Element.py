"""Total-Lagrangian finite elements used by the FEM solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.fem.generator.Mesh import FEMMesh


@dataclass(frozen=True)
class ElementData:
    connectivity: np.ndarray
    shape_values: np.ndarray
    shape_gradients: np.ndarray
    reference_weights: np.ndarray
    material_dimension: int
    name: str
    formulation: str = "volume"
    reference_frames: np.ndarray | None = None
    reference_radii: np.ndarray | None = None
    axis_offset: float = 0.0

    @property
    def number_of_cells(self):
        return int(self.connectivity.shape[0])

    @property
    def nodes_per_cell(self):
        return int(self.connectivity.shape[1])

    @property
    def quadrature_count(self):
        return int(self.shape_values.shape[0])

    @property
    def is_membrane(self):
        return self.material_dimension == 2 and not self.is_axisymmetric

    @property
    def is_axisymmetric(self):
        return self.formulation == "axisymmetric_solid"

    @property
    def constitutive_dimension(self):
        return 3 if self.is_axisymmetric else self.material_dimension

    @property
    def is_classical_membrane(self):
        return self.formulation == "classical_membrane"

    @staticmethod
    def _triangle_frame(coordinates):
        edge_01 = coordinates[1] - coordinates[0]
        edge_02 = coordinates[2] - coordinates[0]
        length_01 = np.linalg.norm(edge_01)
        normal = np.cross(edge_01, edge_02)
        double_area = np.linalg.norm(normal)
        scale = max(np.linalg.norm(edge_01), np.linalg.norm(edge_02), 1.0)
        if length_01 <= 1.0e-13 * scale or double_area <= 1.0e-13 * scale**2:
            raise ValueError("TRI3 membrane element is degenerate")
        basis_1 = edge_01 / length_01
        basis_3 = normal / double_area
        basis_2 = np.cross(basis_3, basis_1)
        return np.stack((basis_1, basis_2, basis_3))

    def current_frames(self, positions):
        if not self.is_classical_membrane:
            raise ValueError("current local frames only belong to the classical TRI3 membrane")
        positions = np.asarray(positions, dtype=np.float64)
        return np.asarray([self._triangle_frame(positions[cell]) for cell in self.connectivity])

    def embedded_deformation_gradients(self, positions):
        """Return the 3x2 surface map used only for geometry/CCD checks.

        The classical constitutive update does not use this map: it uses the
        co-rotated 2x2 Jacobian returned by :meth:`deformation_gradients`.
        """
        positions = np.asarray(positions, dtype=np.float64)
        element_positions = positions[self.connectivity]
        return np.einsum("eai,eqaj->eqij", element_positions, self.shape_gradients)

    def deformation_gradients(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        element_positions = positions[self.connectivity]
        if self.is_axisymmetric:
            deformation_gradient = np.zeros(
                (
                    self.number_of_cells,
                    self.quadrature_count,
                    3,
                    3,
                ),
                dtype=np.float64,
            )
            deformation_gradient[:, :, :2, :2] = np.einsum(
                "eai,eqaj->eqij",
                element_positions[:, :, :2],
                self.shape_gradients,
            )
            current_radius = np.einsum(
                "qa,ea->eq",
                self.shape_values,
                element_positions[:, :, 0] - self.axis_offset,
            )
            deformation_gradient[:, :, 2, 2] = current_radius / self.reference_radii
            return deformation_gradient
        if self.is_classical_membrane:
            frames = self.current_frames(positions)
            local_positions = np.einsum("eij,eaj->eai", frames[:, :2], element_positions)
            return np.einsum("eai,eqaj->eqij", local_positions, self.shape_gradients)
        return np.einsum("eai,eqaj->eqij", element_positions, self.shape_gradients)

    def minimum_jacobian_ratio(self, positions):
        deformation_gradient = (
            self.embedded_deformation_gradients(positions)
            if self.is_membrane
            else self.deformation_gradients(positions)
        )
        if self.is_membrane:
            metric = np.einsum("eqiJ,eqiK->eqJK", deformation_gradient, deformation_gradient)
            determinant = np.linalg.det(metric)
            return float(np.sqrt(np.maximum(np.min(determinant), 0.0)))
        return float(np.min(np.linalg.det(deformation_gradient)))

    @staticmethod
    def _first_positive_root(coefficients):
        coefficients = np.asarray(coefficients, dtype=np.float64)
        scale = max(float(np.max(np.abs(coefficients))), 1.0)
        while coefficients.size > 1 and abs(coefficients[-1]) <= 1.0e-14 * scale:
            coefficients = coefficients[:-1]
        if coefficients.size <= 1:
            return np.inf
        roots = np.polynomial.polynomial.polyroots(coefficients)
        real_roots = [
            float(root.real)
            for root in roots
            if abs(root.imag) <= 1.0e-8 * (1.0 + abs(root.real)) and root.real > 1.0e-12
        ]
        return min(real_roots, default=np.inf)

    def maximum_admissible_step(self, positions, direction, minimum_jacobian=1.0e-8, safety=0.9):
        """Limit a line-search segment before its first collapse/inversion.

        For a volume element ``det(F + alpha*dF)`` is cubic.  For TRI3,
        ``||(f0 + alpha*df0) x (f1 + alpha*df1)||^2`` is quartic.  Solving
        those small polynomials prevents a trial step from jumping over a
        zero-Jacobian state and ending in an apparently positive metric.
        """
        if self.is_axisymmetric:
            deformation_gradient = self.deformation_gradients(positions)
            element_direction = np.asarray(direction, dtype=np.float64)[self.connectivity]
            gradient_increment = np.zeros_like(deformation_gradient)
            gradient_increment[:, :, :2, :2] = np.einsum(
                "eai,eqaj->eqij",
                element_direction[:, :, :2],
                self.shape_gradients,
            )
            gradient_increment[:, :, 2, 2] = (
                np.einsum(
                    "qa,ea->eq",
                    self.shape_values,
                    element_direction[:, :, 0],
                )
                / self.reference_radii
            )
        elif self.is_membrane:
            deformation_gradient = self.embedded_deformation_gradients(positions)
            gradient_increment = self.embedded_deformation_gradients(direction)
        else:
            deformation_gradient = self.deformation_gradients(positions)
            gradient_increment = self.deformation_gradients(direction)
        maximum = np.inf
        threshold = float(minimum_jacobian)
        for element_id in range(self.number_of_cells):
            for quadrature_id in range(self.quadrature_count):
                current = deformation_gradient[element_id, quadrature_id]
                increment = gradient_increment[element_id, quadrature_id]
                if self.is_membrane:
                    c = np.cross(current[:, 0], current[:, 1])
                    b = np.cross(increment[:, 0], current[:, 1]) + np.cross(current[:, 0], increment[:, 1])
                    a = np.cross(increment[:, 0], increment[:, 1])
                    coefficients = (
                        np.dot(c, c) - threshold**2,
                        2.0 * np.dot(c, b),
                        np.dot(b, b) + 2.0 * np.dot(c, a),
                        2.0 * np.dot(b, a),
                        np.dot(a, a),
                    )
                else:
                    f0, f1, f2 = current.T
                    d0, d1, d2 = increment.T
                    coefficients = (
                        np.linalg.det(current) - threshold,
                        np.linalg.det(np.column_stack((d0, f1, f2)))
                        + np.linalg.det(np.column_stack((f0, d1, f2)))
                        + np.linalg.det(np.column_stack((f0, f1, d2))),
                        np.linalg.det(np.column_stack((d0, d1, f2)))
                        + np.linalg.det(np.column_stack((d0, f1, d2)))
                        + np.linalg.det(np.column_stack((f0, d1, d2))),
                        np.linalg.det(increment),
                    )
                maximum = min(maximum, self._first_positive_root(coefficients))
        if np.isfinite(maximum):
            return max(0.0, float(safety) * maximum)
        return 1.0


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


_HEX_SIGNS = np.asarray(
    (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ),
    dtype=np.float64,
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
            weights[element_id, 0] = 0.5 * double_area * 2.0 * np.pi * reference_radius
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


__all__ = ["ElementData", "create_element"]
