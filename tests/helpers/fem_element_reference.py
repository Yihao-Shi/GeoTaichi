"""Independent NumPy element geometry checks; never used by a solver."""

import numpy as np


def embedded_deformation_gradients(self, positions):
    """Return the 3x2 surface map used only for geometry/CCD checks.

    The classical constitutive update does not use this map: it uses the
    co-rotated 2x2 Jacobian returned by :meth:`deformation_gradients`.
    """
    positions = np.asarray(positions, dtype=np.float64)
    element_positions = positions[self.connectivity]
    return np.einsum("eai,eqaj->eqij", element_positions, self.shape_gradients)


def minimum_jacobian_ratio(self, positions):
    deformation_gradient = (
        embedded_deformation_gradients(self, positions) if self.is_membrane else self.deformation_gradients(positions)
    )
    if self.is_membrane:
        metric = np.einsum("eqiJ,eqiK->eqJK", deformation_gradient, deformation_gradient)
        determinant = np.linalg.det(metric)
        return float(np.sqrt(np.maximum(np.min(determinant), 0.0)))
    return float(np.min(np.linalg.det(deformation_gradient)))


def _first_positive_root(coefficients):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    scale = max(float(np.max(np.abs(coefficients))), 1.0)
    while coefficients.size > 1 and abs(coefficients[-1]) <= 1.0e-14 * scale:
        coefficients = coefficients[:-1]
    if coefficients.size <= 1:
        return np.inf
    roots = np.polynomial.polynomial.polyroots(coefficients)
    real_roots = [
        float(root.real) for root in roots if abs(root.imag) <= 1.0e-8 * (1.0 + abs(root.real)) and root.real > 1.0e-12
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
        deformation_gradient = embedded_deformation_gradients(self, positions)
        gradient_increment = embedded_deformation_gradients(self, direction)
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
            maximum = min(maximum, _first_positive_root(coefficients))
    if np.isfinite(maximum):
        return max(0.0, float(safety) * maximum)
    return 1.0


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
