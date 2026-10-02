import taichi as ti
import numpy as np
from math import prod

import src.iga.config as config
from src.mesh.GaussPoint import GaussPointInRectangle
from src.nurbs.core.NurbsGeometry import NurbsBasisFunction2d, NurbsBasisFunction3d


@ti.data_oriented
class Element:
    def __init__(self, degree):
        self.knot_range = np.array([p + 1 for p in degree])
        self.total_knot_range = prod(self.knot_range)
        gauss_point = GaussPointInRectangle(gauss_point=self.knot_range, dimemsion=config.DIM)
        gauss_point.create_gauss_point(taichi_field=False)
        self.gauss_number = gauss_point.get_gauss_point_number()
        self.num_gauss_point = ti.Vector(gauss_point.ngp)
        self.gauss_points = ti.Vector.field(config.DIM, ti.f64, self.gauss_number)
        self.gauss_weights = ti.field(ti.f64, self.gauss_number)
        self.gauss_points.from_numpy(gauss_point.gpcoords)
        self.gauss_weights.from_numpy(gauss_point.weight)
        if config.DIM == 3:
            self.patch_basis = NurbsBasisFunction3d(*degree)
        else:
            self.patch_basis = NurbsBasisFunction2d(*degree)
        self.knot_range = ti.Vector(self.knot_range)

    @ti.func
    def compute_local_hessian(self, local_offset1, local_offset2, d2Psi_d2F, dFdx):
        local_hessian = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
        for d1 in range(config.DIM):
            for d2 in range(config.DIM):
                for i in range(config.DIM * config.DIM):
                    for j in range(config.DIM * config.DIM):
                        local_hessian[d1, d2] += (
                            dFdx[config.DIM * local_offset1 + d1, i]
                            * d2Psi_d2F[i, j]
                            * dFdx[config.DIM * local_offset2 + d2, j]
                        )
        return local_hessian

    @ti.func
    def compute_local_gradient(self, local_offset, dPsi_dF, dFdx):
        local_gradient = ti.Vector.zero(ti.f64, config.DIM)
        for d in range(config.DIM):
            for i in range(config.DIM * config.DIM):
                local_gradient[d] += dFdx[config.DIM * local_offset + d, i] * dPsi_dF[i]
        return local_gradient

    @ti.func
    def compute_shape_gradient(self, local_offset, dNdnat, dnatdX):
        """Return grad_X N_I without constructing the full dF/dx map."""
        shape_gradient = ti.Vector.zero(ti.f64, config.DIM)
        for material_axis in ti.static(range(config.DIM)):
            parametric_axis = 0
            while parametric_axis < config.DIM:
                shape_gradient[material_axis] += (
                    dNdnat[local_offset, parametric_axis] * dnatdX[parametric_axis, material_axis]
                )
                parametric_axis += 1
        return shape_gradient

    @ti.func
    def compute_shape_gradients(self, dNdnat, dnatdX):
        """Cache only support x DIM shape gradients for implicit assembly."""
        shape_gradients = ti.Matrix.zero(ti.f64, self.total_knot_range, config.DIM)
        for local_offset in range(self.total_knot_range):
            shape_gradient = self.compute_shape_gradient(local_offset, dNdnat, dnatdX)
            for material_axis in ti.static(range(config.DIM)):
                shape_gradients[local_offset, material_axis] = shape_gradient[material_axis]
        return shape_gradients

    @ti.func
    def compute_local_gradient_from_shape_gradients(self, local_offset, dPsi_dF, shape_gradients):
        """Apply (dF/dx_I)^T P using dF_ab/dx_{I,d}=delta_ad gradN_I[b]."""
        local_gradient = ti.Vector.zero(ti.f64, config.DIM)
        for displacement_component in ti.static(range(config.DIM)):
            for material_axis in ti.static(range(config.DIM)):
                deformation_component = displacement_component + config.DIM * material_axis
                local_gradient[displacement_component] += (
                    shape_gradients[local_offset, material_axis] * dPsi_dF[deformation_component]
                )
        return local_gradient

    @ti.func
    def compute_local_hessian_from_shape_gradients(
        self,
        local_offset1,
        local_offset2,
        d2Psi_d2F,
        shape_gradients,
    ):
        """Apply (dF/dx_I)^T C (dF/dx_J) without a support*DIM by DIM^2 map."""
        local_hessian = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
        for row_component in ti.static(range(config.DIM)):
            for column_component in ti.static(range(config.DIM)):
                row_material_axis = 0
                while row_material_axis < config.DIM:
                    row_deformation_component = row_component + config.DIM * row_material_axis
                    column_material_axis = 0
                    while column_material_axis < config.DIM:
                        column_deformation_component = column_component + config.DIM * column_material_axis
                        local_hessian[row_component, column_component] += (
                            shape_gradients[local_offset1, row_material_axis]
                            * d2Psi_d2F[
                                row_deformation_component,
                                column_deformation_component,
                            ]
                            * shape_gradients[local_offset2, column_material_axis]
                        )
                        column_material_axis += 1
                    row_material_axis += 1
        return local_hessian

    @ti.func
    def compute_deformation_gradient_from_shape_gradients(self, shape_gradients, current_ctrl_coords):
        F = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
        for j in range(self.total_knot_range):
            for a in ti.static(range(config.DIM)):
                for b in ti.static(range(config.DIM)):
                    F[a, b] += shape_gradients[j, b] * current_ctrl_coords[j, a]
        return F

    @ti.func
    def compute_deformation_gradient(self, dNdnat, dnatdX, current_ctrl_coords):
        F = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
        for local_offset in range(self.total_knot_range):
            shape_gradient = self.compute_shape_gradient(local_offset, dNdnat, dnatdX)
            for spatial_axis in ti.static(range(config.DIM)):
                for material_axis in ti.static(range(config.DIM)):
                    F[spatial_axis, material_axis] += (
                        shape_gradient[material_axis] * current_ctrl_coords[local_offset, spatial_axis]
                    )
        return F

    @ti.func
    def interpolate_component(self, N, coordinates, component):
        value = 0.0
        for local_offset in range(self.total_knot_range):
            value += N[local_offset] * coordinates[local_offset, component]
        return value

    @ti.func
    def compute_axisymmetric_deformation_gradient(
        self,
        N,
        shape_gradients,
        rest_ctrl_coords,
        current_ctrl_coords,
        axis_offset,
    ):
        """3D no-swirl deformation gradient from a 2D meridian patch."""
        F = ti.Matrix.zero(ti.f64, 3, 3)
        for local_offset in range(self.total_knot_range):
            for spatial, material_axis in ti.static(ti.ndrange(2, 2)):
                F[spatial, material_axis] += (
                    shape_gradients[local_offset, material_axis] * current_ctrl_coords[local_offset, spatial]
                )
        reference_radius = self.interpolate_component(N, rest_ctrl_coords, 0) - axis_offset
        current_radius = self.interpolate_component(N, current_ctrl_coords, 0) - axis_offset
        F[2, 2] = current_radius / ti.max(reference_radius, 1.0e-30)
        return F

    @ti.func
    def axisymmetric_local_derivative(self, local_offset, component, N, shape_gradients, reference_radius):
        derivative = ti.Matrix.zero(ti.f64, 3, 3)
        for material_axis in ti.static(range(2)):
            derivative[component, material_axis] = shape_gradients[local_offset, material_axis]
        if component == 0:
            derivative[2, 2] = N[local_offset] / ti.max(reference_radius, 1.0e-30)
        return derivative

    @ti.func
    def compute_axisymmetric_local_gradient(
        self,
        local_offset,
        dPsi_dF,
        N,
        shape_gradients,
        reference_radius,
    ):
        gradient = ti.Vector.zero(ti.f64, 2)
        for component in ti.static(range(2)):
            derivative = self.axisymmetric_local_derivative(
                local_offset,
                component,
                N,
                shape_gradients,
                reference_radius,
            )
            for column, row in ti.static(ti.ndrange(3, 3)):
                gradient[component] += derivative[row, column] * dPsi_dF[row + 3 * column]
        return gradient

    @ti.func
    def compute_axisymmetric_local_hessian(
        self,
        local_offset_i,
        local_offset_j,
        tangent,
        N,
        shape_gradients,
        reference_radius,
    ):
        hessian = ti.Matrix.zero(ti.f64, 2, 2)
        for component_i, component_j in ti.static(ti.ndrange(2, 2)):
            derivative_i = self.axisymmetric_local_derivative(
                local_offset_i,
                component_i,
                N,
                shape_gradients,
                reference_radius,
            )
            derivative_j = self.axisymmetric_local_derivative(
                local_offset_j,
                component_j,
                N,
                shape_gradients,
                reference_radius,
            )
            value = 0.0
            for column_i, row_i, column_j, row_j in ti.ndrange(3, 3, 3, 3):
                value += (
                    derivative_i[row_i, column_i]
                    * tangent[row_i + 3 * column_i, row_j + 3 * column_j]
                    * derivative_j[row_j, column_j]
                )
            hessian[component_i, component_j] = value
        return hessian

    @ti.func
    def compute_dF_div_dx(self, dNdnat, dnatdX):
        dFdx = ti.Matrix.zero(ti.f64, self.total_knot_range * config.DIM, config.DIM * config.DIM)
        for linear_id in range(self.total_knot_range):
            shape_gradient = self.compute_shape_gradient(linear_id, dNdnat, dnatdX)
            for a in ti.static(range(config.DIM)):
                for b in ti.static(range(config.DIM)):
                    dFdx[
                        config.DIM * linear_id + a,
                        a + config.DIM * b,
                    ] = shape_gradient[b]
        return dFdx

    @ti.func
    def shapefn(
        self, gauss_id, element_u, element_v, element_w, prefix_num_knot, prefix_total_num_ctrlpts, num_knot, patches
    ):
        pts = self.gauss_points[gauss_id]
        if ti.static(config.DIM == 2):
            knots = self.parent2parametric2d(element_u, element_v, pts)
            N = self.patch_basis.NurbsBasis2d(
                prefix_num_knot[0],
                prefix_num_knot[1],
                prefix_total_num_ctrlpts,
                num_knot[0],
                num_knot[1],
                knots[0],
                knots[1],
                patches.knot_vector_u,
                patches.knot_vector_v,
                patches.weights,
            )
            return N
        else:
            knots = self.parent2parametric3d(element_u, element_v, element_w, pts)
            N = self.patch_basis.NurbsBasis3d(
                prefix_num_knot[0],
                prefix_num_knot[1],
                prefix_num_knot[2],
                prefix_total_num_ctrlpts,
                num_knot[0],
                num_knot[1],
                num_knot[2],
                knots[0],
                knots[1],
                knots[2],
                patches.knot_vector_u,
                patches.knot_vector_v,
                patches.knot_vector_w,
                patches.weights,
            )
            return N

    @ti.func
    def dshapefn(
        self, gauss_id, element_u, element_v, element_w, prefix_num_knot, prefix_total_num_ctrlpts, num_knot, patches
    ):
        pts = self.gauss_points[gauss_id]
        if ti.static(config.DIM == 2):
            knots = self.parent2parametric2d(element_u, element_v, pts)
            N, dNdnat = self.patch_basis.NurbsBasisDers2d(
                prefix_num_knot[0],
                prefix_num_knot[1],
                prefix_total_num_ctrlpts,
                num_knot[0],
                num_knot[1],
                knots[0],
                knots[1],
                patches.knot_vector_u,
                patches.knot_vector_v,
                patches.weights,
            )
            return N, dNdnat
        else:
            knots = self.parent2parametric3d(element_u, element_v, element_w, pts)
            N, dNdnat = self.patch_basis.NurbsBasisDers3d(
                prefix_num_knot[0],
                prefix_num_knot[1],
                prefix_num_knot[2],
                prefix_total_num_ctrlpts,
                num_knot[0],
                num_knot[1],
                num_knot[2],
                knots[0],
                knots[1],
                knots[2],
                patches.knot_vector_u,
                patches.knot_vector_v,
                patches.knot_vector_w,
                patches.weights,
            )
            return N, dNdnat

    @ti.func
    def ddshapefn(
        self, gauss_id, element_u, element_v, element_w, prefix_num_knot, prefix_total_num_ctrlpts, num_knot, patches
    ):
        pts = self.gauss_points[gauss_id]
        if ti.static(config.DIM == 2):
            knots = self.parent2parametric2d(element_u, element_v, pts)
            N, dNdnat, d2Ndnat = self.patch_basis.NurbsBasis2ndDers2d(
                prefix_num_knot[0],
                prefix_num_knot[1],
                prefix_total_num_ctrlpts,
                num_knot[0],
                num_knot[1],
                knots[0],
                knots[1],
                patches.knot_vector_u,
                patches.knot_vector_v,
                patches.weights,
            )
            return N, dNdnat, d2Ndnat
        else:
            knots = self.parent2parametric3d(element_u, element_v, element_w, pts)
            N, dNdnat, d2Ndnat = self.patch_basis.NurbsBasis2ndDers3d(
                prefix_num_knot[0],
                prefix_num_knot[1],
                prefix_num_knot[2],
                prefix_total_num_ctrlpts,
                num_knot[0],
                num_knot[1],
                num_knot[2],
                knots[0],
                knots[1],
                knots[2],
                patches.knot_vector_u,
                patches.knot_vector_v,
                patches.knot_vector_w,
                patches.weights,
            )
            return N, dNdnat, d2Ndnat

    @ti.func
    def parent2parametric1d(self, eleknot, point):
        return 0.5 * ((eleknot[1] - eleknot[0]) * point + eleknot[1] + eleknot[0])

    @ti.func
    def parent2parametric2d(self, eleknot_u, eleknot_v, point):
        return ti.Vector([self.parent2parametric1d(eleknot_u, point[0]), self.parent2parametric1d(eleknot_v, point[1])])

    @ti.func
    def parent2parametric3d(self, eleknot_u, eleknot_v, eleknot_w, point):
        return ti.Vector(
            [
                self.parent2parametric1d(eleknot_u, point[0]),
                self.parent2parametric1d(eleknot_v, point[1]),
                self.parent2parametric1d(eleknot_w, point[2]),
            ]
        )
