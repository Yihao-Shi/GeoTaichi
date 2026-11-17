from src.nurbs.core.NurbsBasisFunction import NurbsBasisFunction

import taichi as ti


@ti.data_oriented
class NurbsBasisFunction1d:
    def __init__(self, degree, dimension=1):
        self.basis_u = NurbsBasisFunction(degree)
        self.dimension = dimension

    @ti.func
    def find_element(self, start_knot, num_knot, xi, knot_vector):
        num_ctrlpts_u = num_knot - self.basis_u.degree - 1
        span_u = self.basis_u.find_span(start_knot, num_ctrlpts_u, xi, knot_vector)
        return span_u - self.basis_u.degree

    @ti.func
    def nurbs_basis1d(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, weight):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1))
        num_ctrlpts_u = num_knot - self.basis_u.degree - 1

        span_u = self.basis_u.find_span(start_knot, num_ctrlpts_u, xi, knot_vector)
        nurbs_u = self.basis_u.basis_funcs(start_knot, span_u, xi, knot_vector)

        w, kk = 0.0, 0
        uind = span_u - self.basis_u.degree
        k = 0
        while k < self.basis_u.degree + 1:
            linear_id = uind + k + start_ctrlpt
            nmp = nurbs_u[k]
            w += nmp * weight[linear_id]
            N[kk] = nmp * weight[linear_id]
            kk += 1
            k += 1
        return N / w

    @ti.func
    def nurbs_basis_ders1d(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, weight):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1))
        dNdnat = ti.Vector.zero(float, (self.basis_u.degree + 1))
        num_ctrlpts = num_knot - 1 - self.basis_u.degree

        span_u = self.basis_u.find_span(start_knot, num_ctrlpts, xi, knot_vector)
        dNurbs = self.basis_u.first_ders_basis_funcs(start_knot, span_u, xi, knot_vector)

        w, dwdxi = 0.0, 0.0
        uind = span_u - self.basis_u.degree
        k = 0
        while k < self.basis_u.degree + 1:
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            w += dNurbs[0, k] * wgt
            dwdxi += dNurbs[1, k] * wgt
            k += 1

        kk = 0
        uind = span_u - self.basis_u.degree
        k = 0
        while k < self.basis_u.degree + 1:
            linear_id = uind + k + start_ctrlpt
            fac = weight[linear_id] / (w * w)
            nmp = dNurbs[0, k]
            N[kk] = nmp * fac * w
            dNdnat[kk] = (dNurbs[1, k] * w - nmp * dwdxi) * fac
            kk += 1
            k += 1
        return N, dNdnat

    @ti.func
    def nurbs_basis2nd_ders1d(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, weight):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1))
        dNdnat = ti.Vector.zero(float, (self.basis_u.degree + 1))
        d2Ndnat = ti.Vector.zero(float, (self.basis_u.degree + 1))
        num_ctrlpts = num_knot - 1 - self.basis_u.degree

        span = self.basis_u.find_span(start_knot, num_ctrlpts, xi, knot_vector)
        nurbs = self.basis_u.second_ders_basis_funcs(start_knot, span, xi, knot_vector)

        w = 0.0
        dwdxi, d2wdxi = 0.0, 0.0
        uind = span - self.basis_u.degree
        k = 0
        while k < self.basis_u.degree + 1:
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            w += nurbs[0, k] * wgt
            dwdxi += nurbs[1, k] * wgt
            d2wdxi += nurbs[2, k] * wgt
            k += 1

        kk = 0
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = span - self.basis_u.degree
        k = 0
        while k < self.basis_u.degree + 1:
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = nurbs[0, k]

            N[kk] = nmp * fac * w
            dNdnat[kk] = (nurbs[1, k] * w - nmp * dwdxi) * fac
            d2Ndnat[kk] = (
                nurbs[2, k] * inv_w
                - 2 * nurbs[1, k] * dwdxi * inv_w2
                - nurbs[0, k] * d2wdxi * inv_w2
                + 2 * nurbs[0, k] * dwdxi * dwdxi * inv_w3
            ) * wgt
            kk += 1
            k += 1
        return N, dNdnat, d2Ndnat

    @ti.func
    def nurbs_basis_interpolations1d(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, point, weight):
        num_ctrlpts = num_knot - 1 - self.basis_u.degree

        span = self.basis_u.find_span(start_knot, num_ctrlpts, xi, knot_vector)
        nurbs = self.basis_u.basis_funcs(start_knot, span, xi, knot_vector)

        position, w = ti.Vector.zero(float, self.dimension), 0.0
        uind = span - self.basis_u.degree
        k = 0
        while k < self.basis_u.degree + 1:
            linear_id = uind + k + start_ctrlpt
            position += nurbs[k] * weight[linear_id] * point[linear_id]
            w += nurbs[k] * weight[linear_id]
            k += 1
        return position / w

    @ti.func
    def nurbs_basis_interpolations_ders1d(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, point, weight):
        num_ctrlpts = num_knot - 1 - self.basis_u.degree

        span_u = self.basis_u.find_span(start_knot, num_ctrlpts, xi, knot_vector)
        dNurbs = self.basis_u.first_ders_basis_funcs(start_knot, span_u, xi, knot_vector)

        w, dwdxi = 0.0, 0.0
        uind = span_u - self.basis_u.degree
        k = 0
        while k < self.basis_u.degree + 1:
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            w += dNurbs[0, k] * wgt
            dwdxi += dNurbs[1, k] * wgt
            k += 1

        position, dirs_u = ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension)
        uind = span_u - self.basis_u.degree
        k = 0
        while k < self.basis_u.degree + 1:
            linear_id = uind + k + start_ctrlpt
            fac = weight[linear_id] / (w * w)
            nmp = dNurbs[0, k]
            position += nmp * fac * w * point[linear_id]
            dirs_u += (dNurbs[1, k] * w - nmp * dwdxi) * fac * point[linear_id]
            k += 1
        return position, dirs_u

    @ti.func
    def nurbs_basis_interpolations2nd_ders1d(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, point, weight):
        num_ctrlpts = num_knot - 1 - self.basis_u.degree

        span = self.basis_u.find_span(start_knot, num_ctrlpts, xi, knot_vector)
        dNurbs = self.basis_u.second_ders_basis_funcs(start_knot, span, xi, knot_vector)

        w = 0.0
        dwdxi, d2wdxi = 0.0, 0.0
        uind = span - self.basis_u.degree
        k = 0
        while k < self.basis_u.degree + 1:
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            w += dNurbs[0, k] * wgt
            dwdxi += dNurbs[1, k] * wgt
            d2wdxi += dNurbs[2, k] * wgt
            k += 1

        position, dirs_u, ddirs_uu = (
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
        )
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = span - self.basis_u.degree
        k = 0
        while k < self.basis_u.degree + 1:
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = dNurbs[0, k]

            position += nmp * fac * w * point[linear_id]
            dirs_u += (dNurbs[1, k] * w - nmp * dwdxi) * fac * point[linear_id]
            ddirs_uu += (
                (
                    dNurbs[2, k] * inv_w
                    - 2 * dNurbs[1, k] * dwdxi * inv_w2
                    - dNurbs[0, k] * d2wdxi * inv_w2
                    + 2 * dNurbs[0, k] * dwdxi * dwdxi * inv_w3
                )
                * wgt
                * point[linear_id]
            )
            k += 1
        return position, dirs_u, ddirs_uu

    @ti.func
    def nurbs_basis_moving_interpolations2nd_ders1d(
        self,
        start_knot,
        start_ctrlpt,
        num_knot,
        xi,
        knot_vector,
        point,
        direction,
        alpha,
        weight,
    ):
        """Interpolate a linearly moving curve from one rational basis pass."""
        num_ctrlpts = num_knot - 1 - self.basis_u.degree
        span = self.basis_u.find_span(start_knot, num_ctrlpts, xi, knot_vector)
        shape, gradient, hessian = self.nurbs_basis2nd_ders1d(
            start_knot,
            start_ctrlpt,
            num_knot,
            xi,
            knot_vector,
            weight,
        )

        position = ti.Vector.zero(float, self.dimension)
        dirs_u = ti.Vector.zero(float, self.dimension)
        ddirs_uu = ti.Vector.zero(float, self.dimension)
        uind = span - self.basis_u.degree
        k = 0
        while k < self.basis_u.degree + 1:
            linear_id = start_ctrlpt + uind + k
            moving_point = point[linear_id] + alpha * direction[linear_id]
            position += shape[k] * moving_point
            dirs_u += gradient[k] * moving_point
            ddirs_uu += hessian[k] * moving_point
            k += 1
        return position, dirs_u, ddirs_uu

    @ti.func
    def nurbs_basis_hessian(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, point, weight):
        num_ctrlpts = num_knot - 1 - self.basis_u.degree

        span = self.basis_u.find_span(start_knot, num_ctrlpts, xi, knot_vector)
        dNurbs = self.basis_u.second_ders_basis_funcs(start_knot, span, xi, knot_vector)

        w = 0.0
        dwdxi, d2wdxi = 0.0, 0.0
        uind = span - self.basis_u.degree
        k = 0
        while k < self.basis_u.degree + 1:
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            w += dNurbs[0, k] * wgt
            dwdxi += dNurbs[1, k] * wgt
            d2wdxi += dNurbs[2, k] * wgt
            k += 1

        position, dirs_u, ddirs_uu = (
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
        )
        N = ti.Vector.zero(float, (self.basis_u.degree + 1))
        dNdnat = ti.Vector.zero(float, (self.basis_u.degree + 1))
        d2Ndnat = ti.Vector.zero(float, (self.basis_u.degree + 1))
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        kk = 0
        uind = span - self.basis_u.degree
        k = 0
        while k < self.basis_u.degree + 1:
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = dNurbs[0, k]

            nshape = nmp * fac * w
            derivative = (dNurbs[1, k] * w - nmp * dwdxi) * fac
            dderivative = (
                dNurbs[2, k] * inv_w
                - 2 * dNurbs[1, k] * dwdxi * inv_w2
                - dNurbs[0, k] * d2wdxi * inv_w2
                + 2 * dNurbs[0, k] * dwdxi * dwdxi * inv_w3
            ) * wgt

            position += nshape * point[linear_id]
            dirs_u += derivative * point[linear_id]
            ddirs_uu += dderivative * point[linear_id]

            N[kk] = nshape
            dNdnat[kk] = derivative
            d2Ndnat[kk] = dderivative
            kk += 1
            k += 1
        return span, N, dNdnat, dirs_u, ddirs_uu
