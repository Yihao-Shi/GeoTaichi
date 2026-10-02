import numpy as np
from scipy.sparse import lil_matrix

from src.nurbs.element.Element import Element
from src.nurbs.Utilities import get_determinant_3x2
from src.mesh.GaussPoint import GaussPointInRectangle

class LinearQuadrilateralElement(Element):
    def __init__(self, current_configuration, reference_configuration, connectivity, energy=None, gauss_point_number=1):
        self._reference_area = None
        self.reference_dimension= 2
        gauss_point = GaussPointInRectangle(gauss_point=gauss_point_number, dimemsion=2)
        gauss_point.create_gauss_point(taichi_field=False)
        self._gauss_points = gauss_point.gpcoords.copy()
        self._gauss_weights = gauss_point.weight.copy()
        self._gauss_number = gauss_point_number ** 2
        super().__init__(current_configuration, reference_configuration, connectivity, energy)

    @property
    def gauss_points(self):
        return self._gauss_points

    @property
    def gauss_weights(self):
        return self._gauss_weights

    @property
    def gauss_number(self):
        return self._gauss_number

    @property
    def reference_area(self):
        return self._reference_area

    @reference_area.setter
    def reference_area(self, reference_area):
        self._reference_area = np.asarray(reference_area)

    def initialize(self, current_configuration, reference_configuration, connectivity, energy=None):
        super().initialize(current_configuration, reference_configuration, connectivity, energy)
        self.compute_area(np.asarray(reference_configuration))

    def standard_shape(self):
        # Keep the reference-node order identical to ``shape_function``:
        # bottom-left, bottom-right, top-right, top-left.  Swapping the last
        # two nodes folds the reference square and gives opposite-signed
        # quadrature areas in its upper half.
        return np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])

    def quadrilateral_area(vertices):
        x0 = vertices[:-1, :-1, 0]
        y0 = vertices[:-1, :-1, 1]
        x1 = vertices[1:, :-1, 0]
        y1 = vertices[1:, :-1, 1]
        x2 = vertices[1:, 1:, 0]
        y2 = vertices[1:, 1:, 1]
        x3 = vertices[:-1, 1:, 0]
        y3 = vertices[:-1, 1:, 1]
        return 0.5 * np.abs(x0 * y1 + x1 * y2 + x2 * y3 + x3 * y0 - (y0 * x1 + y1 * x2 + y2 * x3 + y3 * x0))

    def compute_area(self, reference_configuration):
        self._reference_area = np.zeros(self.face_num * self.gauss_number)
        for nele in range(self.face_num):
            local_nodes = reference_configuration[self.connectivity[nele]]
            rest_shape = self.standard_shape()
            dXdnat = self.compute_jacobian(rest_shape)
            dxdnat = self.compute_jacobian(local_nodes)
            for ngp in range(self.gauss_number):
                dnatdX = np.linalg.inv(dXdnat[ngp])
                deformation_gradient = dxdnat[ngp] @ dnatdX
                global_gauss_id = nele * self.gauss_number + ngp
                self._reference_area[global_gauss_id] = self.gauss_weights[ngp] * np.linalg.det(deformation_gradient)

    def shape_function(self, natural_coords):
        return np.array([0.25 * (1. - natural_coords[0]) * (1. - natural_coords[1]),
                         0.25 * (1. + natural_coords[0]) * (1. - natural_coords[1]),
                         0.25 * (1. + natural_coords[0]) * (1. + natural_coords[1]),
                         0.25 * (1. - natural_coords[0]) * (1. + natural_coords[1])])

    def dshape_dnat(self, natural_coords):
        return np.array([[-(1. - natural_coords[1]) * 0.25, -(1. - natural_coords[0]) * 0.25],
                         [ (1. - natural_coords[1]) * 0.25, -(1. + natural_coords[0]) * 0.25],
                         [ (1. + natural_coords[1]) * 0.25,  (1. + natural_coords[0]) * 0.25],
                         [-(1. + natural_coords[1]) * 0.25,  (1. - natural_coords[0]) * 0.25]])
    
    def d2shape_dnat2(self, natural_coords):
        return np.array([[0., 0.25, 0.],
                         [0., 0.25, 0.],
                         [0., 0.25, 0.],
                         [0., 0.25, 0.]])

    def get_mesh(self, elements: np.ndarray):
        mesh = set()
        face2ele = {} 
        for iele, ele in enumerate(elements):
            faces = [(ele[0], ele[1], ele[2]), (ele[0], ele[2], ele[3])]
            faces = list(map(lambda face: tuple(sorted(face)), faces))
            for face in faces:
                mesh.add(face)
                if face in face2ele:
                    face2ele[face].add(iele)
                else:
                    face2ele[face] = {iele}
        surfaces = set()
        for face in face2ele:
            if len(face2ele[face]) == 1:
                surfaces.add(face)
        mesh = np.array(list(mesh)); surfaces = np.array(list(surfaces))
        return mesh, face2ele, surfaces

    def extrapolate(self, internal_vals, nodal_vals):
        tmp = np.sqrt(3)
        natural_coordss = np.array([[-tmp, -tmp], [tmp, -tmp], [tmp, tmp], [-tmp, tmp]])
        for ele in nodal_vals:
            vec = np.array([internal_vals[ele, i] for i in range(self.gauss_points.shape[0])])
            for node in range(nodal_vals[ele].n):
                nodal_vals[ele][node] = (self.shape_function(natural_coordss[node, :]) * vec).sum()

    def compute_step_size(self, delta, deformation_gradient, tol=1e-7):
        alpha_max = np.zeros((self.face_num * self.gauss_number))
        deformation_gradient_increment = self.compute_deformation_gradient(delta)
        for nele in range(self.face_num):
            for ngp in range(self.gauss_number):
                global_gauss_id = nele * self.gauss_number + ngp
                a = np.cross(deformation_gradient_increment[global_gauss_id, :, 0], deformation_gradient_increment[global_gauss_id, :, 1])
                b = np.cross(deformation_gradient[global_gauss_id, :, 0], deformation_gradient_increment[global_gauss_id, :, 1]) + np.cross(deformation_gradient_increment[global_gauss_id, :, 0], deformation_gradient[global_gauss_id, :, 1]) 
                c = np.cross(deformation_gradient[global_gauss_id, :, 0], deformation_gradient[global_gauss_id, :, 1]) 
                alpha_max[global_gauss_id] = self.find_zero_volume_roots(a, b, c, tol)
        return np.min(alpha_max)

    def compute_jacobian(self, local_nodes):
        dnodednat = np.zeros((self.gauss_number, local_nodes.shape[1], 2))
        for ngp in range(self.gauss_number):
            gauss_point = self.gauss_points[ngp]
            dshape = self.dshape_dnat(gauss_point)
            dnodednat[ngp] = local_nodes.transpose() @ dshape
        return dnodednat

    def compute_deformation_gradient(self, vertices):
        deformation_gradient = np.zeros((self.face_num * self.gauss_number, self.dimension, self.reference_dimension))
        for nele in range(self.face_num):
            local_nodes = vertices[self.connectivity[nele]]
            rest_shape = self.reference_vertice[self.connectivity[nele]]
            dXdnat = self.compute_jacobian(rest_shape)
            dxdnat = self.compute_jacobian(local_nodes)
            for ngp in range(self.gauss_number):
                dnatdX = np.linalg.inv(dXdnat[ngp])
                deformation_gradient[nele * self.gauss_number + ngp] = dxdnat[ngp] @ dnatdX
        return deformation_gradient

    def compute_ddeformation_dx(self):
        dF_dx = np.zeros((self.face_num * self.gauss_number, self.dimension * self.reference_dimension, self.dimension * 4))
        for nele in range(self.face_num):
            rest_shape = self.reference_vertice[self.connectivity[nele]]
            dXdnat = self.compute_jacobian(rest_shape)
            for ngp in range(self.gauss_number):
                gauss_point = self.gauss_points[ngp]
                dshape_dnat = self.dshape_dnat(gauss_point)
                dnatdX = np.linalg.inv(dXdnat[ngp]) 
                if self.dimension == 3:
                    dF_dx[nele * self.gauss_number + ngp, 0, 0] = dshape_dnat[0, 0] * dnatdX[0, 0] + dshape_dnat[0, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 3, 0] = dshape_dnat[0, 0] * dnatdX[0, 1] + dshape_dnat[0, 1] * dnatdX[1, 1]
                    dF_dx[nele * self.gauss_number + ngp, 1, 1] = dshape_dnat[0, 0] * dnatdX[0, 0] + dshape_dnat[0, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 4, 1] = dshape_dnat[0, 0] * dnatdX[0, 1] + dshape_dnat[0, 1] * dnatdX[1, 1]
                    dF_dx[nele * self.gauss_number + ngp, 2, 2] = dshape_dnat[0, 0] * dnatdX[0, 0] + dshape_dnat[0, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 5, 2] = dshape_dnat[0, 0] * dnatdX[0, 1] + dshape_dnat[0, 1] * dnatdX[1, 1]

                    dF_dx[nele * self.gauss_number + ngp, 0, 3] = dshape_dnat[1, 0] * dnatdX[0, 0] + dshape_dnat[1, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 3, 3] = dshape_dnat[1, 0] * dnatdX[0, 1] + dshape_dnat[1, 1] * dnatdX[1, 1]
                    dF_dx[nele * self.gauss_number + ngp, 1, 4] = dshape_dnat[1, 0] * dnatdX[0, 0] + dshape_dnat[1, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 4, 4] = dshape_dnat[1, 0] * dnatdX[0, 1] + dshape_dnat[1, 1] * dnatdX[1, 1]
                    dF_dx[nele * self.gauss_number + ngp, 2, 5] = dshape_dnat[1, 0] * dnatdX[0, 0] + dshape_dnat[1, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 5, 5] = dshape_dnat[1, 0] * dnatdX[0, 1] + dshape_dnat[1, 1] * dnatdX[1, 1]

                    dF_dx[nele * self.gauss_number + ngp, 0, 6] = dshape_dnat[2, 0] * dnatdX[0, 0] + dshape_dnat[2, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 3, 6] = dshape_dnat[2, 0] * dnatdX[0, 1] + dshape_dnat[2, 1] * dnatdX[1, 1]
                    dF_dx[nele * self.gauss_number + ngp, 1, 7] = dshape_dnat[2, 0] * dnatdX[0, 0] + dshape_dnat[2, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 4, 7] = dshape_dnat[2, 0] * dnatdX[0, 1] + dshape_dnat[2, 1] * dnatdX[1, 1]
                    dF_dx[nele * self.gauss_number + ngp, 2, 8] = dshape_dnat[2, 0] * dnatdX[0, 0] + dshape_dnat[2, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 5, 8] = dshape_dnat[2, 0] * dnatdX[0, 1] + dshape_dnat[2, 1] * dnatdX[1, 1]

                    dF_dx[nele * self.gauss_number + ngp, 0, 9] = dshape_dnat[3, 0] * dnatdX[0, 0] + dshape_dnat[3, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 3, 9] = dshape_dnat[3, 0] * dnatdX[0, 1] + dshape_dnat[3, 1] * dnatdX[1, 1]
                    dF_dx[nele * self.gauss_number + ngp, 1, 10] = dshape_dnat[3, 0] * dnatdX[0, 0] + dshape_dnat[3, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 4, 10] = dshape_dnat[3, 0] * dnatdX[0, 1] + dshape_dnat[3, 1] * dnatdX[1, 1]
                    dF_dx[nele * self.gauss_number + ngp, 2, 11] = dshape_dnat[3, 0] * dnatdX[0, 0] + dshape_dnat[3, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 5, 11] = dshape_dnat[3, 0] * dnatdX[0, 1] + dshape_dnat[3, 1] * dnatdX[1, 1]
                elif self.dimension == 2:
                    dF_dx[nele * self.gauss_number + ngp, 0, 0] = dshape_dnat[0, 0] * dnatdX[0, 0] + dshape_dnat[0, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 2, 0] = dshape_dnat[0, 0] * dnatdX[0, 1] + dshape_dnat[0, 1] * dnatdX[1, 1]
                    dF_dx[nele * self.gauss_number + ngp, 1, 1] = dshape_dnat[0, 0] * dnatdX[0, 0] + dshape_dnat[0, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 3, 1] = dshape_dnat[0, 0] * dnatdX[0, 1] + dshape_dnat[0, 1] * dnatdX[1, 1]

                    dF_dx[nele * self.gauss_number + ngp, 0, 2] = dshape_dnat[1, 0] * dnatdX[0, 0] + dshape_dnat[1, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 2, 2] = dshape_dnat[1, 0] * dnatdX[0, 1] + dshape_dnat[1, 1] * dnatdX[1, 1]
                    dF_dx[nele * self.gauss_number + ngp, 1, 3] = dshape_dnat[1, 0] * dnatdX[0, 0] + dshape_dnat[1, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 3, 3] = dshape_dnat[1, 0] * dnatdX[0, 1] + dshape_dnat[1, 1] * dnatdX[1, 1]

                    dF_dx[nele * self.gauss_number + ngp, 0, 4] = dshape_dnat[2, 0] * dnatdX[0, 0] + dshape_dnat[2, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 2, 4] = dshape_dnat[2, 0] * dnatdX[0, 1] + dshape_dnat[2, 1] * dnatdX[1, 1]
                    dF_dx[nele * self.gauss_number + ngp, 1, 5] = dshape_dnat[2, 0] * dnatdX[0, 0] + dshape_dnat[2, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 3, 5] = dshape_dnat[2, 0] * dnatdX[0, 1] + dshape_dnat[2, 1] * dnatdX[1, 1]

                    dF_dx[nele * self.gauss_number + ngp, 0, 6] = dshape_dnat[3, 0] * dnatdX[0, 0] + dshape_dnat[3, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 2, 6] = dshape_dnat[3, 0] * dnatdX[0, 1] + dshape_dnat[3, 1] * dnatdX[1, 1]
                    dF_dx[nele * self.gauss_number + ngp, 1, 7] = dshape_dnat[3, 0] * dnatdX[0, 0] + dshape_dnat[3, 1] * dnatdX[1, 0]
                    dF_dx[nele * self.gauss_number + ngp, 3, 7] = dshape_dnat[3, 0] * dnatdX[0, 1] + dshape_dnat[3, 1] * dnatdX[1, 1]
        return dF_dx
    
    def compute_quadrature_energy(self):
        quadrature_energy = np.zeros(self.face_num * self.gauss_number)
        for element_id in range(self.face_num):
            local_nodes = self.vertice[self.connectivity[element_id]]
            rest_shape = self.reference_vertice[self.connectivity[element_id]]
            dXdnat = self.compute_jacobian(rest_shape)
            dxdnat = self.compute_jacobian(local_nodes)
            for ngp in range(self.gauss_number):
                dnatdX = np.linalg.inv(dXdnat[ngp])
                deformation_gradient = dxdnat[ngp] @ dnatdX
                volume = self.reference_area[element_id * self.gauss_number + ngp]
                quadrature_energy[element_id * self.gauss_number + ngp] = self.energy.Psi(deformation_gradient) * volume
        return quadrature_energy

    def surface_energy(self, deformation_gradient):
        total_energy = 0.
        for element_id in range(self.face_num):
            for ngp in range(self.gauss_number):
                td = deformation_gradient[element_id * self.gauss_number + ngp]
                volume = self.reference_area[element_id * self.gauss_number + ngp]
                total_energy += self.energy.Psi(td) * volume
        return total_energy

    def surface_energy_gradient(self, deformation_gradient, dF_dx):
        grad = np.zeros(self.vertice_num * self.dimension)
        for element_id in range(self.face_num):
            conn = self.connectivity[element_id]
            for ngp in range(self.gauss_number):
                td = deformation_gradient[element_id * self.gauss_number + ngp]
                dPsi_dx = self.energy.dPsi_div_dx(td, dF_dx[element_id * self.gauss_number + ngp])
                volume = self.reference_area[element_id * self.gauss_number + ngp]
                for i in range(4):
                    grad[self.dimension * conn[i]:self.dimension * conn[i] + self.dimension] += dPsi_dx[self.dimension * i:self.dimension * i + self.dimension] * volume
        return grad

    def surface_energy_hessian(self, deformation_gradient, dF_dx):
        hess = lil_matrix((self.vertice_num * self.dimension, self.vertice_num * self.dimension))
        for element_id in range(self.face_num):
            conn = self.connectivity[element_id]
            for ngp in range(self.gauss_number):
                td = deformation_gradient[element_id * self.gauss_number + ngp]
                d2Psi_dx2 = self.energy.d2Psi_div_dx2(td, dF_dx[element_id * self.gauss_number + ngp])
                volume = self.reference_area[element_id * self.gauss_number + ngp]
                for i in range(4):
                    for j in range(4):
                        hess[self.dimension * conn[i]:self.dimension * conn[i] + self.dimension, self.dimension * conn[j]:self.dimension * conn[j] + self.dimension] += d2Psi_dx2[self.dimension * i:self.dimension * i + self.dimension, self.dimension * j:self.dimension * j + self.dimension] * volume
        return hess 
    
    def visualize(self, size):
        import matplotlib.pyplot as plt
        if self.dimension == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            structured_grid = self.vertice.reshape((size[1], size[0], -1))
            ax.plot_wireframe(structured_grid[:, :, 0], structured_grid[:, :, 1], structured_grid[:, :, 2], color='red')
            ax.scatter(structured_grid[:, :, 0], structured_grid[:, :, 1], structured_grid[:, :, 2], color='red', s=15, marker='^', label="Vertice")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        elif self.dimension == 2:
            structured_grid = self.vertice.reshape((size[1], size[0], -1))
            plt.figure(figsize=(6, 6))
            for i in range(size[0]):
                plt.plot(structured_grid[i, :, 0], structured_grid[i, :, 1], 'k-', linewidth=0.8)
            for j in range(size[1]):
                plt.plot(structured_grid[:, j, 0], structured_grid[:, j, 1], 'k-', linewidth=0.8)
        plt.show()
