import numpy as np

from src.nurbs.Energy import SurfaceEnergy

class Element:
    _energy: SurfaceEnergy
    def __init__(self, current_configuration, reference_configuration, connectivity, energy=None):
        self._face_num = 0
        self._vertice_num = 0
        self._dimension = 2
        self._reference_dimension = 2
        self._vertice = None
        self._connectivity = None
        self._energy = None
        self.initialize(current_configuration, reference_configuration, connectivity, energy)
        if self.dimension == 3:
            self.find_zero_volume_roots = self.find_zero_volume_roots_3x2
        elif self.dimension == 2:
            self.find_zero_volume_roots = self.find_zero_volume_roots_2x2

    @property
    def dimension(self):
        return self._dimension
    
    @dimension.setter
    def dimension(self, dimension):
        self._dimension = dimension

    @property
    def reference_dimension(self):
        return self._reference_dimension
    
    @reference_dimension.setter
    def reference_dimension(self, reference_dimension):
        self._reference_dimension = reference_dimension

    @property
    def face_num(self):
        return self._face_num
    
    @face_num.setter
    def face_num(self, face_num):
        self._face_num = face_num

    @property
    def vertice_num(self):
        return self._vertice_num
    
    @vertice_num.setter
    def vertice_num(self, vertice_num):
        self._vertice_num = vertice_num

    @property
    def vertice(self):
        return self._vertice

    @vertice.setter
    def vertice(self, vertice):
        vertice = np.asarray(vertice)
        self._vertice = vertice.copy()
        self._dimension = self._vertice.shape[1]

    @property
    def reference_vertice(self):
        return self._reference_vertice

    @reference_vertice.setter
    def reference_vertice(self, reference_vertice):
        reference_vertice = np.asarray(reference_vertice)
        self._reference_vertice = reference_vertice.copy()

    @property
    def connectivity(self):
        return self._connectivity

    @connectivity.setter
    def connectivity(self, connectivity):
        connectivity = np.asarray(connectivity)
        if connectivity.ndim == 1:
            self._face_num = 1
        else:
            self._face_num = connectivity.shape[0]
        self._connectivity = connectivity.copy().reshape(self.face_num, -1)

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        self._energy = energy

    def initialize(self, current_configuration, reference_configuration, connectivity, energy=None):
        self.vertice = current_configuration
        self.reference_vertice = reference_configuration
        self.connectivity = connectivity
        self.energy = energy
        self.vertice_num = self.vertice.shape[0]
        if self.vertice.shape[0] != self.reference_vertice.shape[0]:
            raise ValueError("Current shape and rest shape should have the same dimension!")
        
    def find_zero_volume_roots_2x2(self, a, b, c, tol=1e-7):
        coeffs = [a, b, c]
        roots = np.roots(coeffs)
        alphas = []
        for r in roots:
            if np.isclose(r.imag, 0, atol=tol):
                alphas.append(r.real)
        positive_alphas = [r for r in alphas if r > tol]
        positive_alphas.append(1.0)
        return min(positive_alphas)

    def find_zero_volume_roots_3x2(self, a, b, c, tol=1e-7):
        if np.linalg.norm(np.cross(c, b)) > tol * (np.linalg.norm(c) + np.linalg.norm(b)) or np.linalg.norm(np.cross(c, a)) > tol * (np.linalg.norm(c) + np.linalg.norm(a)):
            return 1.
        if np.allclose(c, 0, atol=tol):
            return 0.0
        absC2 = np.abs(a)
        if np.max(absC2) > tol:
            idx = np.argmax(absC2)
            c2 = a[idx]
            c1 = b[idx]
            c0 = c[idx]
        else:
            absC1 = np.abs(b)
            if np.max(absC1) > tol:
                idx = np.argmax(absC1)
                c2 = 0.0
                c1 = b[idx]
                c0 = c[idx]
            else:
                raise ValueError("Degenerate case: both C2 and C1 are nearly zero.")
        alphas = []
        if np.abs(c2) > tol:
            coeffs = [c2, c1, c0]
            roots = np.roots(coeffs)
            for r in roots:
                if np.isclose(r.imag, 0, atol=tol):
                    alphas.append(r.real)
        else:
            alpha_lin = -c0 / c1
            alphas.append(alpha_lin)
        positive_alphas = [r for r in alphas if r > tol]
        positive_alphas.append(1.0)
        return min(positive_alphas)

    def compute_deformation_gradient(self, vertices):
        raise NotImplementedError

    def compute_ddeformation_dx(self):
        raise NotImplementedError

    def surface_energy(self, deformation_gradient):
        raise NotImplementedError

    def surface_energy_gradient(self, deformation_gradient, dF_dx):
        raise NotImplementedError

    def surface_energy_hessian(self, deformation_gradient, dF_dx):
        raise NotImplementedError
    
    def compute_step_size(self, delta, deformation_gradient, tol=1e-7):
        raise NotImplementedError
    
    def visualize(self):
        raise NotImplementedError