import numpy as np
from scipy.sparse import lil_matrix

from src.nurbs.element.Element import Element

class LinearTriangleElement(Element):
    def __init__(self, current_configuration, reference_configuration, connectivity, energy=None):
        self.Bm = None
        self._reference_vertice = None
        self.reference_dimension= 2
        super().__init__(current_configuration, reference_configuration, connectivity, energy)

    def initialize(self, current_configuration, reference_configuration, connectivity, energy=None):
        super().initialize(current_configuration, reference_configuration, connectivity, energy)
        self.Bm = np.zeros((self.face_num, 2, 2))
        for element_id in range(self.face_num):
            face_vertice = self._reference_vertice[self.connectivity[element_id]]
            l01 = face_vertice[1] - face_vertice[0]
            l02 = face_vertice[2] - face_vertice[0]
            l01_norm = np.linalg.norm(l01)
            l01_normalized = l01 / l01_norm
            self.Bm[element_id] = np.linalg.inv(np.array([[l01_norm, np.dot(l02, l01_normalized)],
                                                          [0., np.linalg.norm(np.cross(l02, l01_normalized))]]))

    def compute_area(self, element_id, vertice):
        face_vertex = vertice[self.connectivity[element_id]]
        v0 = face_vertex[1] - face_vertex[0]
        v1 = face_vertex[2] - face_vertex[0]
        return 0.5 * np.linalg.norm(np.cross(v0, v1))
    
    def compute_step_size(self, delta, deformation_gradient, tol=1e-7):
        alpha_max = np.zeros(self.face_num)
        deformation_gradient_increment = self.compute_deformation_gradient(delta)
        for nele in range(self.face_num):
            a = np.cross(deformation_gradient_increment[nele, :, 0], deformation_gradient_increment[nele, :, 1])
            b = np.cross(deformation_gradient[nele, :, 0], deformation_gradient_increment[nele, :, 1]) + np.cross(deformation_gradient_increment[nele, :, 0], deformation_gradient[nele, :, 1]) 
            c = np.cross(deformation_gradient[nele, :, 0], deformation_gradient[nele, :, 1]) 
            alpha_max[nele] = self.find_zero_volume_roots(a, b, c, tol)
        return np.min(alpha_max)
    
    def compute_deformation_gradient(self, vertices):
        deformation_gradient = np.zeros((self.face_num, 3, 2))
        for element_id in range(self.face_num):
            face_vertex = vertices[self.connectivity[element_id]]
            deformation_gradient[element_id] =  np.array([[face_vertex[1][0] - face_vertex[0][0], face_vertex[2][0] - face_vertex[0][0]],
                                                          [face_vertex[1][1] - face_vertex[0][1], face_vertex[2][1] - face_vertex[0][1]],
                                                          [face_vertex[1][2] - face_vertex[0][2], face_vertex[2][2] - face_vertex[0][2]]]) @ self.Bm[element_id]
        return deformation_gradient
    
    def compute_ddeformation_dx(self):
        dF_dx = np.zeros((self.face_num, 6, 9))
        for element_id in range(self.face_num):
            Bm = self.Bm[element_id]
            dF_dx[element_id, 0, 0] = -Bm[0, 0] - Bm[1, 0]
            dF_dx[element_id, 0, 3] = Bm[0, 0]
            dF_dx[element_id, 0, 6] = Bm[1, 0]

            dF_dx[element_id, 1, 1] = -Bm[0, 0] - Bm[1, 0]
            dF_dx[element_id, 1, 4] = Bm[0, 0]
            dF_dx[element_id, 1, 7] = Bm[1, 0]

            dF_dx[element_id, 2, 2] = -Bm[0, 0] - Bm[1, 0]
            dF_dx[element_id, 2, 5] = Bm[00, 0]
            dF_dx[element_id, 2, 8] = Bm[1, 0]

            dF_dx[element_id, 3, 0] = -Bm[0, 1] - Bm[1, 1]
            dF_dx[element_id, 3, 3] = Bm[0, 1]
            dF_dx[element_id, 3, 6] = Bm[1, 1]

            dF_dx[element_id, 4, 1] = -Bm[0, 1] - Bm[1, 1]
            dF_dx[element_id, 4, 4] = Bm[0, 1]
            dF_dx[element_id, 4, 7] = Bm[1, 1]

            dF_dx[element_id, 5, 2] = -Bm[0, 1] - Bm[1, 1]
            dF_dx[element_id, 5, 5] = Bm[0, 1]
            dF_dx[element_id, 5, 8] = Bm[1, 1]
        return dF_dx
    
    def compute_quadrature_energy(self, vertices):
        quadrature_energy = np.zeros(self.face_num)
        for element_id in range(self.face_num):
            face_vertex = vertices[self.connectivity[element_id]]
            deformation_gradient =  np.array([[face_vertex[1][0] - face_vertex[0][0], face_vertex[2][0] - face_vertex[0][0]],
                                              [face_vertex[1][1] - face_vertex[0][1], face_vertex[2][1] - face_vertex[0][1]],
                                              [face_vertex[1][2] - face_vertex[0][2], face_vertex[2][2] - face_vertex[0][2]]]) @ self.Bm[element_id]
            volume = self.compute_area(element_id, self._reference_vertice)
            quadrature_energy[element_id] = self.energy.Psi(deformation_gradient) * volume
        return quadrature_energy
    
    def surface_energy(self, deformation_gradient):
        total_energy = 0.
        for element_id in range(self.face_num):
            td = deformation_gradient[element_id]
            volume = self.compute_area(element_id, self._reference_vertice)
            total_energy += self.energy.Psi(td) * volume
        return total_energy

    def surface_energy_gradient(self, deformation_gradient, dF_dx):
        grad = np.zeros(self.vertice_num * 3)
        for element_id in range(self.face_num):
            conn = self.connectivity[element_id]
            td = deformation_gradient[element_id]
            dfdx = dF_dx[element_id]
            dPsi_dx = self.energy.dPsi_div_dx(td, dfdx)
            volume = self.compute_area(element_id, self._reference_vertice)
            for i in range(3):
                grad[3 * conn[i]:3 * conn[i] + 3] += dPsi_dx[3 * i:3 * i + 3] * volume
        return grad

    def project_evd(self, matrix):
        eigenvalues, Q = np.linalg.eigh(matrix)
        eigenvalues_proj = np.maximum(eigenvalues, 0)
        A_proj = Q @ np.diag(eigenvalues_proj) @ Q.T
        return A_proj

    def surface_energy_hessian(self, deformation_gradient, dF_dx):
        hess = lil_matrix((self.vertice_num * 3, self.vertice_num * 3))
        for element_id in range(self.face_num):
            conn = self.connectivity[element_id]
            td = deformation_gradient[element_id]
            dfdx = dF_dx[element_id]
            d2Psi_dx2 = self.energy.d2Psi_div_dx2(td, dfdx)
            volume = self.compute_area(element_id, self._reference_vertice)
            for i in range(3):
                for j in range(3):
                    hess[3 * conn[i]:3 * conn[i] + 3, 3 * conn[j]:3 * conn[j] + 3] += d2Psi_dx2[3 * i:3 * i + 3, 3 * j:3 * j + 3] * volume
        return hess
    
    def visualize(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(self.vertice[:, 0], self.vertice[:, 1], self.vertice[:, 2], triangles=self.connectivity, edgecolor='k', alpha=0.8)
        ax.scatter(self.vertice[:, 0], self.vertice[:, 1], self.vertice[:, 2], color='red', s=15, marker='^', label="Vertice")
        ax.plot_trisurf(self.reference_vertice[:, 0], self.reference_vertice[:, 1], self.reference_vertice[:, 2], triangles=self.connectivity, edgecolor='k', alpha=0.8)
        ax.scatter(self.reference_vertice[:, 0], self.reference_vertice[:, 1], self.reference_vertice[:, 2], color='black', s=15, marker='o', label="ReferenceVertice")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
        
