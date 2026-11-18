from src.nurbs.core.NurbsBasisFunction import NurbsBasisFunction

import taichi as ti


@ti.data_oriented
class NurbsBasisFunction2d:
    def __init__(self, degree_u, degree_v, dimension=2):
        self.basis_u = NurbsBasisFunction(degree_u)
        self.basis_v = NurbsBasisFunction(degree_v)
        self.dimension = dimension

    @ti.func
    def find_element(self, start_knot_u, start_knot_v, num_knot_u, num_knot_v, xi, eta, knot_vector_u, knot_vector_v):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        return span_u - self.basis_u.degree, span_v - self.basis_v.degree

    @ti.func
    def nurbs_basis2d(
        self,
        start_knot_u,
        start_knot_v,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        xi,
        eta,
        knot_vector_u,
        knot_vector_v,
        weight,
    ):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        nurbs_u = self.basis_u.basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        nurbs_v = self.basis_v.basis_funcs(start_knot_v, span_v, eta, knot_vector_v)

        # Keep degree-dependent local-vector shapes templated, but iterate tensor
        # supports at runtime.  Static nested loops grow as
        # ``(degree_u + 1) * (degree_v + 1)`` and make high-order kernels
        # prohibitively expensive to compile.
        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_size = support_u * support_v
        w, kk = 0.0, 0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            nmp = nurbs_u[k] * nurbs_v[j]
            w += nmp * weight[linear_id]
            N[kk] = nmp * weight[linear_id]
            kk += 1
            flat += 1
        return N / w

    @ti.func
    def nurbs_basis_ders2d(
        self,
        start_knot_u,
        start_knot_v,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        xi,
        eta,
        knot_vector_u,
        knot_vector_v,
        weight,
    ):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        dNdnat = ti.Matrix.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1), 2)
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        dNurbsU = self.basis_u.first_ders_basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        dNurbsV = self.basis_v.first_ders_basis_funcs(start_knot_v, span_v, eta, knot_vector_v)

        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_size = support_u * support_v
        w, dwdxi, dwdet = 0.0, 0.0, 0.0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            wgt = weight[linear_id]
            w += dNurbsU[0, k] * dNurbsV[0, j] * wgt
            dwdxi += dNurbsU[1, k] * dNurbsV[0, j] * wgt
            dwdet += dNurbsU[0, k] * dNurbsV[1, j] * wgt
            flat += 1

        kk = 0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            fac = weight[linear_id] / (w * w)
            nmp = dNurbsU[0, k] * dNurbsV[0, j]
            N[kk] = nmp * fac * w
            dNdnat[kk, 0] = (dNurbsU[1, k] * dNurbsV[0, j] * w - nmp * dwdxi) * fac
            dNdnat[kk, 1] = (dNurbsU[0, k] * dNurbsV[1, j] * w - nmp * dwdet) * fac
            kk += 1
            flat += 1
        return N, dNdnat

    @ti.func
    def nurbs_basis2nd_ders2d(
        self,
        start_knot_u,
        start_knot_v,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        xi,
        eta,
        knot_vector_u,
        knot_vector_v,
        weight,
    ):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        dNdnat = ti.Matrix.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1), 2)
        d2Ndnat = ti.Matrix.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1), 3)
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        nurbs_u = self.basis_u.second_ders_basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        nurbs_v = self.basis_v.second_ders_basis_funcs(start_knot_v, span_v, eta, knot_vector_v)

        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_size = support_u * support_v
        w = 0.0
        dwdxi, dwdet = 0.0, 0.0
        d2wdxi, d2wdet, d2wdxe = 0.0, 0.0, 0.0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            wgt = weight[linear_id]
            w += nurbs_u[0, k] * nurbs_v[0, j] * wgt
            dwdxi += nurbs_u[1, k] * nurbs_v[0, j] * wgt
            dwdet += nurbs_u[0, k] * nurbs_v[1, j] * wgt
            d2wdxi += nurbs_u[2, k] * nurbs_v[0, j] * wgt
            d2wdet += nurbs_u[0, k] * nurbs_v[2, j] * wgt
            d2wdxe += nurbs_u[1, k] * nurbs_v[1, j] * wgt
            flat += 1

        kk = 0
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = nurbs_u[0, k] * nurbs_v[0, j]

            N[kk] = nmp * fac * w
            dNdnat[kk, 0] = (nurbs_u[1, k] * nurbs_v[0, j] * w - nmp * dwdxi) * fac
            dNdnat[kk, 1] = (nurbs_u[0, k] * nurbs_v[1, j] * w - nmp * dwdet) * fac
            d2Ndnat[kk, 0] = (
                nurbs_u[2, k] * nurbs_v[0, j] * inv_w
                - 2 * nurbs_u[1, k] * nurbs_v[0, j] * dwdxi * inv_w2
                - nurbs_u[0, k] * nurbs_v[0, j] * d2wdxi * inv_w2
                + 2 * nurbs_u[0, k] * nurbs_v[0, j] * dwdxi * dwdxi * inv_w3
            ) * wgt
            d2Ndnat[kk, 1] = (
                nurbs_u[0, k] * nurbs_v[2, j] * inv_w
                - 2 * nurbs_u[0, k] * nurbs_v[1, j] * dwdet * inv_w2
                - nurbs_u[0, k] * nurbs_v[0, j] * d2wdet * inv_w2
                + 2 * nurbs_u[0, k] * nurbs_v[0, j] * dwdet * dwdet * inv_w3
            ) * wgt
            d2Ndnat[kk, 2] = (
                nurbs_u[1, k] * nurbs_v[1, j] * inv_w
                - nurbs_u[1, k] * nurbs_v[0, j] * dwdet * inv_w2
                - nurbs_u[0, k] * nurbs_v[1, j] * dwdxi * inv_w2
                - nurbs_u[0, k] * nurbs_v[0, j] * d2wdxe * inv_w2
                + 2 * nurbs_u[0, k] * nurbs_v[0, j] * dwdxi * dwdet * inv_w3
            ) * wgt
            kk += 1
            flat += 1
        return N, dNdnat, d2Ndnat

    @ti.func
    def nurbs_basis_local_interpolations2d(
        self, start_knot_u, start_knot_v, span_u, span_v, xi, eta, knot_vector_u, knot_vector_v, point, weight
    ):
        nurbs_u = self.basis_u.basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        nurbs_v = self.basis_v.basis_funcs(start_knot_v, span_v, eta, knot_vector_v)

        position, wght = ti.Vector.zero(float, self.dimension), 0.0
        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_size = support_u * support_v
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            linear_id = k + j * support_u
            nmp = nurbs_u[k] * nurbs_v[j]
            position += (
                nmp * ti.Vector([point[linear_id, d] for d in ti.static(range(self.dimension))]) * weight[linear_id]
            )
            wght += nmp * weight[linear_id]
            flat += 1
        return position / wght

    @ti.func
    def nurbs_basis_interpolations2d(
        self,
        start_knot_u,
        start_knot_v,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        xi,
        eta,
        knot_vector_u,
        knot_vector_v,
        point,
        weight,
    ):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        nurbs_u = self.basis_u.basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        nurbs_v = self.basis_v.basis_funcs(start_knot_v, span_v, eta, knot_vector_v)

        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_size = support_u * support_v
        position, wght = ti.Vector.zero(float, self.dimension), 0.0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            nmp = nurbs_u[k] * nurbs_v[j]
            position += nmp * point[linear_id] * weight[linear_id]
            wght += nmp * weight[linear_id]
            flat += 1
        return position / wght

    @ti.func
    def nurbs_basis_interpolations_ders2d(
        self,
        start_knot_u,
        start_knot_v,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        xi,
        eta,
        knot_vector_u,
        knot_vector_v,
        point,
        weight,
    ):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        dNurbsU = self.basis_u.first_ders_basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        dNurbsV = self.basis_v.first_ders_basis_funcs(start_knot_v, span_v, eta, knot_vector_v)

        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_size = support_u * support_v
        w, dwdxi, dwdet = 0.0, 0.0, 0.0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            wgt = weight[linear_id]
            w += dNurbsU[0, k] * dNurbsV[0, j] * wgt
            dwdxi += dNurbsU[1, k] * dNurbsV[0, j] * wgt
            dwdet += dNurbsU[0, k] * dNurbsV[1, j] * wgt
            flat += 1

        position, dirs_u, dirs_v = (
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
        )
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            fac = weight[linear_id] / (w * w)
            nmp = dNurbsU[0, k] * dNurbsV[0, j]
            position += nmp * fac * point[linear_id] * w
            dirs_u += (dNurbsU[1, k] * dNurbsV[0, j] * w - nmp * dwdxi) * fac * point[linear_id]
            dirs_v += (dNurbsU[0, k] * dNurbsV[1, j] * w - nmp * dwdet) * fac * point[linear_id]
            flat += 1
        return position, dirs_u, dirs_v

    @ti.func
    def nurbs_basis_interpolations2nd_ders2d(
        self,
        start_knot_u,
        start_knot_v,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        xi,
        eta,
        knot_vector_u,
        knot_vector_v,
        point,
        weight,
    ):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        nurbs_u = self.basis_u.second_ders_basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        nurbs_v = self.basis_v.second_ders_basis_funcs(start_knot_v, span_v, eta, knot_vector_v)

        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_size = support_u * support_v
        w = 0.0
        dwdxi, dwdet = 0.0, 0.0
        d2wdxi, d2wdet, d2wdxe = 0.0, 0.0, 0.0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            wgt = weight[linear_id]
            w += nurbs_u[0, k] * nurbs_v[0, j] * wgt
            dwdxi += nurbs_u[1, k] * nurbs_v[0, j] * wgt
            dwdet += nurbs_u[0, k] * nurbs_v[1, j] * wgt
            d2wdxi += nurbs_u[2, k] * nurbs_v[0, j] * wgt
            d2wdet += nurbs_u[0, k] * nurbs_v[2, j] * wgt
            d2wdxe += nurbs_u[1, k] * nurbs_v[1, j] * wgt
            flat += 1

        position, dirs_u, dirs_v = (
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
        )
        ddirs_uu, ddirs_vv, ddirs_uv = (
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
        )
        # dirsU -> dNdxi, dirsV -> dNdeta
        # ddirsUU -> d2Ndxi2, ddirsVV -> d2Ndeta2, ddirsUV -> d2Ndxideta
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = nurbs_u[0, k] * nurbs_v[0, j]

            position += nmp * fac * w * point[linear_id]
            dirs_u += (nurbs_u[1, k] * nurbs_v[0, j] * w - nmp * dwdxi) * fac * point[linear_id]
            dirs_v += (nurbs_u[0, k] * nurbs_v[1, j] * w - nmp * dwdet) * fac * point[linear_id]
            ddirs_uu += (
                (
                    nurbs_u[2, k] * nurbs_v[0, j] * inv_w
                    - 2 * nurbs_u[1, k] * nurbs_v[0, j] * dwdxi * inv_w2
                    - nurbs_u[0, k] * nurbs_v[0, j] * d2wdxi * inv_w2
                    + 2 * nurbs_u[0, k] * nurbs_v[0, j] * dwdxi * dwdxi * inv_w3
                )
                * wgt
                * point[linear_id]
            )
            ddirs_vv += (
                (
                    nurbs_u[0, k] * nurbs_v[2, j] * inv_w
                    - 2 * nurbs_u[0, k] * nurbs_v[1, j] * dwdet * inv_w2
                    - nurbs_u[0, k] * nurbs_v[0, j] * d2wdet * inv_w2
                    + 2 * nurbs_u[0, k] * nurbs_v[0, j] * dwdet * dwdet * inv_w3
                )
                * wgt
                * point[linear_id]
            )
            ddirs_uv += (
                (
                    nurbs_u[1, k] * nurbs_v[1, j] * inv_w
                    - nurbs_u[1, k] * nurbs_v[0, j] * dwdet * inv_w2
                    - nurbs_u[0, k] * nurbs_v[1, j] * dwdxi * inv_w2
                    - nurbs_u[0, k] * nurbs_v[0, j] * d2wdxe * inv_w2
                    + 2 * nurbs_u[0, k] * nurbs_v[0, j] * dwdxi * dwdet * inv_w3
                )
                * wgt
                * point[linear_id]
            )
            flat += 1
        return position, dirs_u, dirs_v, ddirs_uu, ddirs_vv, ddirs_uv

    @ti.func
    def nurbs_basis_moving_interpolations2nd_ders2d(
        self,
        start_knot_u,
        start_knot_v,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        xi,
        eta,
        knot_vector_u,
        knot_vector_v,
        point,
        direction,
        alpha,
        weight,
    ):
        """Interpolate homogeneous derivatives, then apply the rational quotient.

        Accumulating six vectors directly avoids materializing support-sized
        shape/gradient/Hessian matrices inside every closest-point iteration.
        """
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree
        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        ders_u = self.basis_u.second_ders_basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        ders_v = self.basis_v.second_ders_basis_funcs(start_knot_v, span_v, eta, knot_vector_v)
        w, wu, wv, wuu, wvv, wuv = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        p = ti.Vector.zero(float, self.dimension)
        pu = ti.Vector.zero(float, self.dimension)
        pv = ti.Vector.zero(float, self.dimension)
        puu = ti.Vector.zero(float, self.dimension)
        pvv = ti.Vector.zero(float, self.dimension)
        puv = ti.Vector.zero(float, self.dimension)
        support_u = self.basis_u.degree + 1
        flat = 0
        while flat < support_u * (self.basis_v.degree + 1):
            j = flat // support_u
            k = flat - j * support_u
            control = (
                start_ctrlpt + span_u - self.basis_u.degree + k + (span_v - self.basis_v.degree + j) * num_ctrlpts_u
            )
            weight_value = weight[control]
            n = ders_u[0, k] * ders_v[0, j] * weight_value
            nu = ders_u[1, k] * ders_v[0, j] * weight_value
            nv = ders_u[0, k] * ders_v[1, j] * weight_value
            nuu = ders_u[2, k] * ders_v[0, j] * weight_value
            nvv = ders_u[0, k] * ders_v[2, j] * weight_value
            nuv = ders_u[1, k] * ders_v[1, j] * weight_value
            moving_point = point[control] + alpha * direction[control]
            w += n
            wu += nu
            wv += nv
            wuu += nuu
            wvv += nvv
            wuv += nuv
            p += n * moving_point
            pu += nu * moving_point
            pv += nv * moving_point
            puu += nuu * moving_point
            pvv += nvv * moving_point
            puv += nuv * moving_point
            flat += 1
        p /= w
        pu = (pu - wu * p) / w
        pv = (pv - wv * p) / w
        puu = (puu - 2.0 * wu * pu - wuu * p) / w
        pvv = (pvv - 2.0 * wv * pv - wvv * p) / w
        puv = (puv - wu * pv - wv * pu - wuv * p) / w
        return p, pu, pv, puu, pvv, puv

    @ti.func
    def nurbs_span(self, start_knot_u, start_knot_v, num_knot_u, num_knot_v, xi, eta, knot_vector_u, knot_vector_v):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        return span_u, span_v

    @ti.func
    def nurbs_basis_shape(
        self,
        start_knot_u,
        start_knot_v,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        xi,
        eta,
        knot_vector_u,
        knot_vector_v,
        weight,
    ):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        nurbs_u = self.basis_u.second_ders_basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        nurbs_v = self.basis_v.second_ders_basis_funcs(start_knot_v, span_v, eta, knot_vector_v)

        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_size = support_u * support_v
        w = 0.0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            wgt = weight[linear_id]
            w += nurbs_u[0, k] * nurbs_v[0, j] * wgt
            flat += 1

        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        kk = 0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = nurbs_u[0, k] * nurbs_v[0, j]

            nshape = nmp * fac * w
            N[kk] = nshape
            kk += 1
            flat += 1
        return span_u, span_v, N

    @ti.func
    def nurbs_basis_hessian(
        self,
        start_knot_u,
        start_knot_v,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        xi,
        eta,
        knot_vector_u,
        knot_vector_v,
        point,
        weight,
    ):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        nurbs_u = self.basis_u.second_ders_basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        nurbs_v = self.basis_v.second_ders_basis_funcs(start_knot_v, span_v, eta, knot_vector_v)

        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_size = support_u * support_v
        w = 0.0
        dwdxi, dwdet = 0.0, 0.0
        d2wdxi, d2wdet, d2wdxe = 0.0, 0.0, 0.0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            wgt = weight[linear_id]
            w += nurbs_u[0, k] * nurbs_v[0, j] * wgt
            dwdxi += nurbs_u[1, k] * nurbs_v[0, j] * wgt
            dwdet += nurbs_u[0, k] * nurbs_v[1, j] * wgt
            d2wdxi += nurbs_u[2, k] * nurbs_v[0, j] * wgt
            d2wdet += nurbs_u[0, k] * nurbs_v[2, j] * wgt
            d2wdxe += nurbs_u[1, k] * nurbs_v[1, j] * wgt
            flat += 1

        position, dirs_u, dirs_v = (
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
        )
        ddirs_uu, ddirs_vv, ddirs_uv = (
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
        )
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        dNdnatU = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        dNdnatV = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        d2NdnatUU = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        d2NdnatVV = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        d2NdnatUV = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        kk = 0
        # dirsU -> dNdxi, dirsV -> dNdeta
        # ddirsUU -> d2Ndxi2, ddirsVV -> d2Ndeta2, ddirsUV -> d2Ndxideta
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            j = flat // support_u
            k = flat - j * support_u
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = nurbs_u[0, k] * nurbs_v[0, j]

            nshape = nmp * fac * w
            derivate_u = (nurbs_u[1, k] * nurbs_v[0, j] * w - nmp * dwdxi) * fac
            derivate_v = (nurbs_u[0, k] * nurbs_v[1, j] * w - nmp * dwdet) * fac
            dderivate_uu = (
                nurbs_u[2, k] * nurbs_v[0, j] * inv_w
                - 2 * nurbs_u[1, k] * nurbs_v[0, j] * dwdxi * inv_w2
                - nurbs_u[0, k] * nurbs_v[0, j] * d2wdxi * inv_w2
                + 2 * nurbs_u[0, k] * nurbs_v[0, j] * dwdxi * dwdxi * inv_w3
            ) * wgt
            dderivate_vv = (
                nurbs_u[0, k] * nurbs_v[2, j] * inv_w
                - 2 * nurbs_u[0, k] * nurbs_v[1, j] * dwdet * inv_w2
                - nurbs_u[0, k] * nurbs_v[0, j] * d2wdet * inv_w2
                + 2 * nurbs_u[0, k] * nurbs_v[0, j] * dwdet * dwdet * inv_w3
            ) * wgt
            dderivate_uv = (
                nurbs_u[1, k] * nurbs_v[1, j] * inv_w
                - nurbs_u[1, k] * nurbs_v[0, j] * dwdet * inv_w2
                - nurbs_u[0, k] * nurbs_v[1, j] * dwdxi * inv_w2
                - nurbs_u[0, k] * nurbs_v[0, j] * d2wdxe * inv_w2
                + 2 * nurbs_u[0, k] * nurbs_v[0, j] * dwdxi * dwdet * inv_w3
            ) * wgt

            position += nshape * point[linear_id]
            dirs_u += derivate_u * point[linear_id]
            dirs_v += derivate_v * point[linear_id]
            ddirs_uu += dderivate_uu * point[linear_id]
            ddirs_vv += dderivate_vv * point[linear_id]
            ddirs_uv += dderivate_uv * point[linear_id]

            N[kk] = nshape
            # ``derivate_*`` already contains the rational ``w_i / w^2``
            # factor.  Multiplying by ``fac`` a second time corrupts
            # non-uniform-weight surface derivatives and all contact
            # closest-point Hessians built from them.
            dNdnatU[kk] = derivate_u
            dNdnatV[kk] = derivate_v
            d2NdnatUU[kk] = dderivate_uu
            d2NdnatVV[kk] = dderivate_vv
            d2NdnatUV[kk] = dderivate_uv
            kk += 1
            flat += 1
        return span_u, span_v, N, dNdnatU, dNdnatV, dirs_u, dirs_v, ddirs_uu, ddirs_vv, ddirs_uv
