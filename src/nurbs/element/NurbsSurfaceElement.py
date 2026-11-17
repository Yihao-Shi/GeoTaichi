"""Host-side IGA quadrature and orientation checks for planar spline fitting."""

from itertools import product

import numpy as np
from scipy.sparse import lil_matrix

from src.nurbs.element.Element import Element


class NurbsSurfaceElement(Element):
    def __init__(self, surface, energy=None):
        if surface.dimension != 2:
            raise ValueError("Planar fitting requires a two-dimensional spline surface")
        points = surface.control_points
        if not np.isfinite(points).all():
            raise ValueError("Surface control points must be finite")
        weights = getattr(surface, "weights", None)
        if weights is not None and (not np.isfinite(weights).all() or np.any(weights <= 0.0)):
            raise ValueError("NURBS weights must be finite and positive")

        self.gauss_number = (surface.degree_u + 1) * (surface.degree_v + 1)
        self._control_connectivity = surface.connectivity[:, [0, 1, 3, 2]]
        self._gradients, self._reference_area = [], []
        self._check_gradients, self._check_connectivity = [], []
        connectivity = []
        for span_u, span_v in product(surface.element_u, surface.element_v):
            midpoint = (np.mean(span_u), np.mean(span_v))
            conn = surface.ctrlpt_id(*midpoint)
            connectivity.append(conn)
            gauss_u, weight_u = np.polynomial.legendre.leggauss(surface.degree_u + 1)
            gauss_v, weight_v = np.polynomial.legendre.leggauss(surface.degree_v + 1)
            lower = np.array([span_u[0], span_v[0]])
            size = np.array([np.diff(span_u)[0], np.diff(span_v)[0]])
            for i, j in product(range(len(gauss_u)), range(len(gauss_v))):
                parameter = lower + 0.5 * (np.array([gauss_u[i], gauss_v[j]]) + 1.0) * size
                _, du, dv = surface.coefficient_derivate(*parameter)
                self._gradients.append(np.column_stack((du, dv)))
                self._reference_area.append(0.25 * np.prod(size) * weight_u[i] * weight_v[j])
            # shortcut: positivity is sampled; use Bernstein bounds when a global certificate is required.
            for u, v in product(
                np.linspace(0.0, 1.0, 2 * surface.degree_u + 1), np.linspace(0.0, 1.0, 2 * surface.degree_v + 1)
            ):
                parameter = lower + np.array([u, v]) * size
                parameter = np.clip(parameter, np.nextafter(lower, lower + size), np.nextafter(lower + size, lower))
                _, du, dv = surface.coefficient_derivate(*parameter)
                self._check_gradients.append(np.column_stack((du, dv)))
                self._check_connectivity.append(conn)

        self._gradients = np.asarray(self._gradients)
        self._reference_area = np.asarray(self._reference_area)
        self._check_gradients = np.asarray(self._check_gradients)
        self._check_connectivity = np.asarray(self._check_connectivity)
        super().__init__(points, points, np.asarray(connectivity), energy)
        self._sample_connectivity = np.repeat(self.connectivity, self.gauss_number, axis=0)
        self._check_gradients = np.concatenate((self._check_gradients, self._gradients))
        self._check_connectivity = np.concatenate((self._check_connectivity, self._sample_connectivity))
        self.validate(points)

    def compute_deformation_gradient(self, vertices):
        return np.einsum("qai,qaj->qij", np.asarray(vertices)[self._sample_connectivity], self._gradients)

    def compute_ddeformation_dx(self):
        return np.asarray([np.kron(gradient.T, np.eye(2)) for gradient in self._gradients])

    def jacobian_determinants(self, vertices):
        jacobian = np.einsum("qai,qaj->qij", np.asarray(vertices)[self._check_connectivity], self._check_gradients)
        return np.linalg.det(jacobian)

    def validate(self, vertices):
        control_determinants = np.linalg.det(self._control_jacobians(vertices))
        if not np.isfinite(control_determinants).all() or np.any(control_determinants <= 0.0):
            raise ValueError("Control net has a folded or degenerate quadrilateral")
        determinants = self.jacobian_determinants(vertices)
        if not np.isfinite(determinants).all() or np.any(determinants <= 0.0):
            raise ValueError("Spline has an inverted or degenerate IGA Jacobian at a knot-span sample")

    def _control_jacobians(self, vertices):
        cells = np.asarray(vertices)[self._control_connectivity]
        edges = np.roll(cells, -1, axis=1) - cells
        return np.stack((edges, np.roll(edges, -1, axis=1)), axis=-1).reshape(-1, 2, 2)

    def compute_step_size(self, delta, deformation_gradient, tol=1e-7):
        if not np.isfinite(delta).all():
            raise ValueError("Fitting direction must be finite")
        current = np.einsum("qai,qaj->qij", self.vertice[self._check_connectivity], self._check_gradients)
        increment = np.einsum("qai,qaj->qij", np.asarray(delta)[self._check_connectivity], self._check_gradients)
        current = np.concatenate((current, self._control_jacobians(self.vertice)))
        increment = np.concatenate((increment, self._control_jacobians(delta)))
        step = 1.0
        for jacobian, change in zip(current, increment):
            a = np.linalg.det(change)
            b = (
                jacobian[0, 0] * change[1, 1]
                + change[0, 0] * jacobian[1, 1]
                - jacobian[0, 1] * change[1, 0]
                - change[0, 1] * jacobian[1, 0]
            )
            c = np.linalg.det(jacobian)
            coefficients = np.array([a, b, c])
            if not np.isfinite(coefficients).all():
                raise ValueError("Fitting Jacobian increment overflowed")
            coefficients /= np.max(np.abs(coefficients))
            # Terms below roundoff on [0, 1] cannot constrain a unit step.
            while len(coefficients) > 1 and abs(coefficients[0]) <= np.finfo(float).eps:
                coefficients = coefficients[1:]
            for root in np.roots(coefficients):
                if abs(root.imag) < tol and 0.0 < root.real <= 1.0:
                    step = min(step, 0.99 * root.real)
        return step

    def surface_energy(self, deformation_gradient):
        return sum(self.energy.psi(value) * weight for value, weight in zip(deformation_gradient, self._reference_area))

    def surface_energy_gradient(self, deformation_gradient, dF_dx):
        gradient = np.zeros(self.vertice_num * self.dimension)
        for conn, value, derivative, weight in zip(
            self._sample_connectivity, deformation_gradient, dF_dx, self._reference_area
        ):
            dofs = (2 * conn[:, None] + np.arange(2)).ravel()
            gradient[dofs] += weight * self.energy.d_psi_div_dx(value, derivative)
        return gradient

    def surface_energy_hessian(self, deformation_gradient, dF_dx):
        hessian = lil_matrix((self.vertice_num * self.dimension, self.vertice_num * self.dimension))
        for conn, value, derivative, weight in zip(
            self._sample_connectivity, deformation_gradient, dF_dx, self._reference_area
        ):
            dofs = (2 * conn[:, None] + np.arange(2)).ravel()
            hessian[np.ix_(dofs, dofs)] += weight * self.energy.d2_psi_div_dx2(value, derivative)
        return hessian
