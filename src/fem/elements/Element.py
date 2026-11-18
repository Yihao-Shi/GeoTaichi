"""Total-Lagrangian finite elements used by the FEM solvers."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


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
