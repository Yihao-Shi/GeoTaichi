"""Geometry utilities shared by IPC NURBS contact implementations.

This module owns the geometry-only pieces historically implemented under
``src.igampm.contact``: CPU curve closest-point queries, Taichi curve-distance
queries, and point--NURBS distance derivatives.
It deliberately does not own contact search, quadrature measures, constitutive
barrier/friction laws, or solver-specific matrix scattering.

The Taichi functions infer their spatial dimension from vector or matrix
arguments wherever possible, so the same implementation can be used by IGA,
MPM, and future FEM contact code in both two and three dimensions.
The moving-query variants evaluate linear virtual control-point trajectories
without writing shared geometry, which lets each ACCD contact pair own its
complete device-side iteration.
"""

from src.utils.MatrixFunction import outer_product_nd


import taichi as ti
from src.utils.TypeDefinition import mat2x2
from src.utils.constants import DBL_EPSILON


@ti.data_oriented
class PointNurbsDerivative:
    """First and second derivatives of squared point--NURBS distance.

    ``dimension`` is only needed by the two legacy velocity-Jacobian methods,
    whose scalar arguments do not carry a dimension.  All distance methods
    infer it from their vector inputs.  Omitting it preserves historical IGA
    behavior by reading the configured IGA--MPM dimension at construction.
    """

    def __init__(self, dimension=None):
        if dimension is None:
            import src.igampm.config as igampm_config

            dimension = igampm_config.DIM
        dimension = int(dimension)
        if dimension not in (2, 3):
            raise ValueError("PointNurbsDerivative dimension must be 2 or 3")
        self.dimension = dimension

    @ti.func
    def ddistance2_div_dpoint(self, pointer):
        return -2.0 * pointer

    @ti.func
    def ddistance2_div_dctrlpt(self, pointer, nshape):
        return 2.0 * nshape * pointer

    @ti.func
    def dknot_div_dpoint_curve(self, pointer, tangent, curvature):
        coefficient = tangent.dot(tangent) + curvature.dot(pointer)
        result = ti.Vector.zero(ti.f64, pointer.n)
        if ti.abs(coefficient) > 1.0e-30:
            result = tangent / coefficient
        return result

    @ti.func
    def dknot_div_dpoint_surface(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
    ):
        coefficient_matrix = mat2x2(
            [0.0, 0.0],
            [0.0, 0.0],
        )
        coefficient_uu = tangent_u.dot(tangent_u) + curvature_uu.dot(pointer)
        coefficient_uv = tangent_u.dot(tangent_v) + curvature_uv.dot(pointer)
        coefficient_vv = tangent_v.dot(tangent_v) + curvature_vv.dot(pointer)
        determinant = coefficient_uu * coefficient_vv - coefficient_uv * coefficient_uv
        if ti.abs(determinant) > 1.0e-30:
            coefficient_matrix = mat2x2(
                [coefficient_vv / determinant, -coefficient_uv / determinant],
                [-coefficient_uv / determinant, coefficient_uu / determinant],
            )
        return (
            coefficient_matrix[0, 0] * tangent_u + coefficient_matrix[0, 1] * tangent_v,
            coefficient_matrix[1, 0] * tangent_u + coefficient_matrix[1, 1] * tangent_v,
        )

    @ti.func
    def d2distance2_div_d2point_curve(self, pointer, tangent, curvature):
        identity = ti.Matrix.identity(ti.f64, pointer.n)
        parameter_gradient = self.dknot_div_dpoint_curve(pointer, tangent, curvature)
        return -2.0 * (outer_product_nd(tangent, parameter_gradient) - identity)

    @ti.func
    def d2distance2_div_d2point_curve_active(self, pointer, tangent, curvature, free_parameter):
        result = 2.0 * ti.Matrix.identity(ti.f64, pointer.n)
        if free_parameter != 0:
            result = self.d2distance2_div_d2point_curve(pointer, tangent, curvature)
        return result

    @ti.func
    def d2distance2_div_d2point_surface(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
    ):
        identity = ti.Matrix.identity(ti.f64, pointer.n)
        du_dpoint, dv_dpoint = self.dknot_div_dpoint_surface(
            pointer,
            tangent_u,
            tangent_v,
            curvature_uu,
            curvature_vv,
            curvature_uv,
        )
        return -2.0 * (outer_product_nd(tangent_u, du_dpoint) + outer_product_nd(tangent_v, dv_dpoint) - identity)

    @ti.func
    def d2distance2_div_d2point_surface_active(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
        free_u,
        free_v,
    ):
        result = 2.0 * ti.Matrix.identity(ti.f64, pointer.n)
        if free_u != 0 and free_v != 0:
            result = self.d2distance2_div_d2point_surface(
                pointer,
                tangent_u,
                tangent_v,
                curvature_uu,
                curvature_vv,
                curvature_uv,
            )
        elif free_u != 0:
            result = self.d2distance2_div_d2point_curve(pointer, tangent_u, curvature_uu)
        elif free_v != 0:
            result = self.d2distance2_div_d2point_curve(pointer, tangent_v, curvature_vv)
        return result

    @ti.func
    def d2distance2_div_d2ctrlpt_curve(
        self,
        pointer,
        tangent,
        curvature,
        nshape1,
        derivative1,
        nshape2,
        derivative2,
    ):
        identity = ti.Matrix.identity(ti.f64, pointer.n)
        coefficient = tangent.dot(tangent) + curvature.dot(pointer)
        mixed1 = derivative1 * pointer + nshape1 * tangent
        mixed2 = derivative2 * pointer + nshape2 * tangent
        result = 2.0 * nshape1 * nshape2 * identity
        if ti.abs(coefficient) > 1.0e-30:
            result -= 2.0 * outer_product_nd(mixed1, mixed2) / coefficient
        return result

    @ti.func
    def d2distance2_div_d2ctrlpt_curve_active(
        self,
        pointer,
        tangent,
        curvature,
        nshape1,
        derivative1,
        nshape2,
        derivative2,
        free_parameter,
    ):
        result = 2.0 * nshape1 * nshape2 * ti.Matrix.identity(ti.f64, pointer.n)
        if free_parameter != 0:
            result = self.d2distance2_div_d2ctrlpt_curve(
                pointer,
                tangent,
                curvature,
                nshape1,
                derivative1,
                nshape2,
                derivative2,
            )
        return result

    @ti.func
    def d2distance2_div_d2ctrlpt_surface(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
        nshape1,
        derivative_u1,
        derivative_v1,
        nshape2,
        derivative_u2,
        derivative_v2,
    ):
        identity = ti.Matrix.identity(ti.f64, pointer.n)
        coefficient_matrix = mat2x2([0.0, 0.0], [0.0, 0.0])
        coefficient_uu = tangent_u.dot(tangent_u) + curvature_uu.dot(pointer)
        coefficient_uv = tangent_u.dot(tangent_v) + curvature_uv.dot(pointer)
        coefficient_vv = tangent_v.dot(tangent_v) + curvature_vv.dot(pointer)
        determinant = coefficient_uu * coefficient_vv - coefficient_uv * coefficient_uv
        if ti.abs(determinant) > 1.0e-30:
            coefficient_matrix = mat2x2(
                [coefficient_vv / determinant, -coefficient_uv / determinant],
                [-coefficient_uv / determinant, coefficient_uu / determinant],
            )
        mixed_u1 = derivative_u1 * pointer + nshape1 * tangent_u
        mixed_v1 = derivative_v1 * pointer + nshape1 * tangent_v
        mixed_u2 = derivative_u2 * pointer + nshape2 * tangent_u
        mixed_v2 = derivative_v2 * pointer + nshape2 * tangent_v
        mixed1 = ti.Matrix.rows([mixed_u1, mixed_v1])
        mixed2 = ti.Matrix.rows([mixed_u2, mixed_v2])
        return 2.0 * (nshape1 * nshape2 * identity - mixed1.transpose() @ coefficient_matrix @ mixed2)

    @ti.func
    def d2distance2_div_d2ctrlpt_surface_active(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
        nshape1,
        derivative_u1,
        derivative_v1,
        nshape2,
        derivative_u2,
        derivative_v2,
        free_u,
        free_v,
    ):
        result = 2.0 * nshape1 * nshape2 * ti.Matrix.identity(ti.f64, pointer.n)
        if free_u != 0 and free_v != 0:
            result = self.d2distance2_div_d2ctrlpt_surface(
                pointer,
                tangent_u,
                tangent_v,
                curvature_uu,
                curvature_vv,
                curvature_uv,
                nshape1,
                derivative_u1,
                derivative_v1,
                nshape2,
                derivative_u2,
                derivative_v2,
            )
        elif free_u != 0:
            result = self.d2distance2_div_d2ctrlpt_curve(
                pointer,
                tangent_u,
                curvature_uu,
                nshape1,
                derivative_u1,
                nshape2,
                derivative_u2,
            )
        elif free_v != 0:
            result = self.d2distance2_div_d2ctrlpt_curve(
                pointer,
                tangent_v,
                curvature_vv,
                nshape1,
                derivative_v1,
                nshape2,
                derivative_v2,
            )
        return result

    @ti.func
    def d2distance2_div_dctrlpt_dpoint_curve(self, pointer, tangent, curvature, nshape, derivative):
        identity = ti.Matrix.identity(ti.f64, pointer.n)
        parameter_gradient = self.dknot_div_dpoint_curve(pointer, tangent, curvature)
        return 2.0 * (
            (outer_product_nd(tangent, parameter_gradient) - identity) * nshape
            + outer_product_nd(pointer * derivative, parameter_gradient)
        )

    @ti.func
    def d2distance2_div_dctrlpt_dpoint_curve_active(
        self, pointer, tangent, curvature, nshape, derivative, free_parameter
    ):
        result = -2.0 * nshape * ti.Matrix.identity(ti.f64, pointer.n)
        if free_parameter != 0:
            result = self.d2distance2_div_dctrlpt_dpoint_curve(pointer, tangent, curvature, nshape, derivative)
        return result

    @ti.func
    def d2distance2_div_dctrlpt_dpoint_surface(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
        nshape,
        derivative_u,
        derivative_v,
    ):
        identity = ti.Matrix.identity(ti.f64, pointer.n)
        du_dpoint, dv_dpoint = self.dknot_div_dpoint_surface(
            pointer,
            tangent_u,
            tangent_v,
            curvature_uu,
            curvature_vv,
            curvature_uv,
        )
        return 2.0 * (
            (outer_product_nd(tangent_u, du_dpoint) + outer_product_nd(tangent_v, dv_dpoint) - identity) * nshape
            + outer_product_nd(pointer * derivative_u, du_dpoint)
            + outer_product_nd(pointer * derivative_v, dv_dpoint)
        )

    @ti.func
    def d2distance2_div_dctrlpt_dpoint_surface_active(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
        nshape,
        derivative_u,
        derivative_v,
        free_u,
        free_v,
    ):
        result = -2.0 * nshape * ti.Matrix.identity(ti.f64, pointer.n)
        if free_u != 0 and free_v != 0:
            result = self.d2distance2_div_dctrlpt_dpoint_surface(
                pointer,
                tangent_u,
                tangent_v,
                curvature_uu,
                curvature_vv,
                curvature_uv,
                nshape,
                derivative_u,
                derivative_v,
            )
        elif free_u != 0:
            result = self.d2distance2_div_dctrlpt_dpoint_curve(
                pointer,
                tangent_u,
                curvature_uu,
                nshape,
                derivative_u,
            )
        elif free_v != 0:
            result = self.d2distance2_div_dctrlpt_dpoint_curve(
                pointer,
                tangent_v,
                curvature_vv,
                nshape,
                derivative_v,
            )
        return result

    @ti.func
    def dvelocity_div_dpoint(self):
        return ti.Matrix.identity(ti.f64, self.dimension)

    @ti.func
    def dvelocity_div_dctrlpt(self, nshape):
        return -nshape * ti.Matrix.identity(ti.f64, self.dimension)
