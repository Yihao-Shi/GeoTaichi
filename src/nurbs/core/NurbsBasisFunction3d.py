from src.nurbs.core.NurbsBasisFunction import NurbsBasisFunction

import taichi as ti


@ti.data_oriented
class NurbsBasisFunction3d:
    def __init__(self, degree_u, degree_v, degree_w):
        self.dimension = 3
        self.basis_u = NurbsBasisFunction(degree_u)
        self.basis_v = NurbsBasisFunction(degree_v)
        self.basis_w = NurbsBasisFunction(degree_w)

    @ti.func
    def find_element(
        self,
        start_knot_u,
        start_knot_v,
        start_knot_w,
        num_knot_u,
        num_knot_v,
        num_knot_w,
        xi,
        eta,
        zeta,
        knot_vector_u,
        knot_vector_v,
        knot_vector_w,
    ):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        span_w = self.basis_w.find_span(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        return span_u - self.basis_u.degree, span_v - self.basis_v.degree, span_w - self.basis_w.degree

    @ti.func
    def nurbs_basis3d(
        self,
        start_knot_u,
        start_knot_v,
        start_knot_w,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        num_knot_w,
        xi,
        eta,
        zeta,
        knot_vector_u,
        knot_vector_v,
        knot_vector_w,
        weight,
    ):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1) * (self.basis_w.degree + 1))
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        span_w = self.basis_w.find_span(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        nurbs_u = self.basis_u.basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        nurbs_v = self.basis_v.basis_funcs(start_knot_v, span_v, eta, knot_vector_v)
        nurbs_w = self.basis_w.basis_funcs(start_knot_w, span_w, zeta, knot_vector_w)

        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_w = self.basis_w.degree + 1
        support_uv = support_u * support_v
        support_size = support_uv * support_w
        w, kk = 0.0, 0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            i = flat // support_uv
            rem = flat - i * support_uv
            j = rem // support_u
            k = rem - j * support_u
            wind = span_w - self.basis_w.degree + i
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
            nmp = nurbs_u[k] * nurbs_v[j] * nurbs_w[i]
            w += nmp * weight[linear_id]
            N[kk] = nmp * weight[linear_id]
            kk += 1
            flat += 1
        return N / w

    @ti.func
    def nurbs_basis_ders3d(
        self,
        start_knot_u,
        start_knot_v,
        start_knot_w,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        num_knot_w,
        xi,
        eta,
        zeta,
        knot_vector_u,
        knot_vector_v,
        knot_vector_w,
        weight,
    ):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1) * (self.basis_w.degree + 1))
        dNdnat = ti.Matrix.zero(
            float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1) * (self.basis_w.degree + 1), 3
        )
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        span_w = self.basis_w.find_span(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        dNurbsU = self.basis_u.first_ders_basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        dNurbsV = self.basis_v.first_ders_basis_funcs(start_knot_v, span_v, eta, knot_vector_v)
        dNurbsW = self.basis_w.first_ders_basis_funcs(start_knot_w, span_w, zeta, knot_vector_w)

        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_w = self.basis_w.degree + 1
        support_uv = support_u * support_v
        support_size = support_uv * support_w
        w, dwdxi, dwdet, dwdze = 0.0, 0.0, 0.0, 0.0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            i = flat // support_uv
            rem = flat - i * support_uv
            j = rem // support_u
            k = rem - j * support_u
            wind = span_w - self.basis_w.degree + i
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
            wgt = weight[linear_id]
            w += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
            dwdxi += dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
            dwdet += dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * wgt
            dwdze += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * wgt
            flat += 1

        kk = 0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            i = flat // support_uv
            rem = flat - i * support_uv
            j = rem // support_u
            k = rem - j * support_u
            wind = span_w - self.basis_w.degree + i
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
            fac = weight[linear_id] / (w * w)
            nmp = dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i]
            N[kk] = nmp * fac * w
            dNdnat[kk, 0] = (dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * w - nmp * dwdxi) * fac
            dNdnat[kk, 1] = (dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * w - nmp * dwdet) * fac
            dNdnat[kk, 2] = (dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * w - nmp * dwdze) * fac
            kk += 1
            flat += 1
        return N, dNdnat

    @ti.func
    def nurbs_basis2nd_ders3d(
        self,
        start_knot_u,
        start_knot_v,
        start_knot_w,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        num_knot_w,
        xi,
        eta,
        zeta,
        knot_vector_u,
        knot_vector_v,
        knot_vector_w,
        weight,
    ):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1) * (self.basis_w.degree + 1))
        dNdnat = ti.Matrix.zero(
            float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1) * (self.basis_w.degree + 1), 3
        )
        d2Ndnat = ti.Matrix.zero(
            float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1) * (self.basis_w.degree + 1), 6
        )
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        span_w = self.basis_w.find_span(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        nurbs_u = self.basis_u.second_ders_basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        nurbs_v = self.basis_v.second_ders_basis_funcs(start_knot_v, span_v, eta, knot_vector_v)
        nurbs_w = self.basis_w.second_ders_basis_funcs(start_knot_w, span_w, zeta, knot_vector_w)

        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_w = self.basis_w.degree + 1
        support_uv = support_u * support_v
        support_size = support_uv * support_w
        w = 0.0
        dwdxi, dwdet, dwdze = 0.0, 0.0, 0.0
        d2wdxi, d2wdet, d2wdze = 0.0, 0.0, 0.0
        d2wdxe, d2wdez, d2wdxz = 0.0, 0.0, 0.0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            i = flat // support_uv
            rem = flat - i * support_uv
            j = rem // support_u
            k = rem - j * support_u
            wind = span_w - self.basis_w.degree + i
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
            wgt = weight[linear_id]

            w += nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * wgt
            dwdxi += nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[0, i] * wgt
            dwdet += nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[0, i] * wgt
            dwdze += nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[1, i] * wgt
            d2wdxi += nurbs_u[2, k] * nurbs_v[0, j] * nurbs_w[0, i] * wgt
            d2wdet += nurbs_u[0, k] * nurbs_v[2, j] * nurbs_w[0, i] * wgt
            d2wdze += nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[2, i] * wgt
            d2wdxe += nurbs_u[1, k] * nurbs_v[1, j] * nurbs_w[0, i] * wgt
            d2wdez += nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[1, i] * wgt
            d2wdxz += nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[1, i] * wgt
            flat += 1

        kk = 0
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            i = flat // support_uv
            rem = flat - i * support_uv
            j = rem // support_u
            k = rem - j * support_u
            wind = span_w - self.basis_w.degree + i
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i]

            N[kk] = nmp * fac * w
            dNdnat[kk, 0] = (nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[0, i] * w - nmp * dwdxi) * fac
            dNdnat[kk, 1] = (nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[0, i] * w - nmp * dwdet) * fac
            dNdnat[kk, 2] = (nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[1, i] * w - nmp * dwdze) * fac
            d2Ndnat[kk, 0] = (
                nurbs_u[2, k] * nurbs_v[0, j] * nurbs_w[0, i] * inv_w
                - 2 * nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdxi * inv_w2
                - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * d2wdxi * inv_w2
                + 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdxi * dwdxi * inv_w3
            ) * wgt
            d2Ndnat[kk, 1] = (
                nurbs_u[0, k] * nurbs_v[2, j] * nurbs_w[0, i] * inv_w
                - 2 * nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[0, i] * dwdet * inv_w2
                - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * d2wdet * inv_w2
                + 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdet * dwdet * inv_w3
            ) * wgt
            d2Ndnat[kk, 2] = (
                nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[2, i] * inv_w
                - 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[1, i] * dwdze * inv_w2
                - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * d2wdze * inv_w2
                + 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdze * dwdze * inv_w3
            ) * wgt
            d2Ndnat[kk, 3] = (
                nurbs_u[1, k] * nurbs_v[1, j] * nurbs_w[0, i] * inv_w
                - nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdet * inv_w2
                - nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[0, i] * dwdxi * inv_w2
                - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * d2wdxe * inv_w2
                + 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdxi * dwdet * inv_w3
            ) * wgt
            d2Ndnat[kk, 4] = (
                nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[1, i] * inv_w
                - nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[0, i] * dwdze * inv_w2
                - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[1, i] * dwdet * inv_w2
                - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * d2wdez * inv_w2
                + 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdet * dwdze * inv_w3
            ) * wgt
            d2Ndnat[kk, 5] = (
                nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[1, i] * inv_w
                - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[1, i] * dwdxi * inv_w2
                - nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdze * inv_w2
                - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * d2wdxz * inv_w2
                + 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdxi * dwdze * inv_w3
            ) * wgt
            kk += 1
            flat += 1
        return N, dNdnat, d2Ndnat

    @ti.func
    def nurbs_basis_interpolations3d(
        self,
        start_knot_u,
        start_knot_v,
        start_knot_w,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        num_knot_w,
        xi,
        eta,
        zeta,
        knot_vector_u,
        knot_vector_v,
        knot_vector_w,
        point,
        weight,
    ):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        span_w = self.basis_w.find_span(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        nurbs_u = self.basis_u.basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        nurbs_v = self.basis_v.basis_funcs(start_knot_v, span_v, eta, knot_vector_v)
        nurbs_w = self.basis_w.basis_funcs(start_knot_w, span_w, zeta, knot_vector_w)

        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_w = self.basis_w.degree + 1
        support_uv = support_u * support_v
        support_size = support_uv * support_w
        position, wght = ti.Vector.zero(float, self.dimension), 0.0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            i = flat // support_uv
            rem = flat - i * support_uv
            j = rem // support_u
            k = rem - j * support_u
            wind = span_w - self.basis_w.degree + i
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
            nmp = nurbs_u[k] * nurbs_v[j] * nurbs_w[i]
            position += nmp * point[linear_id] * weight[linear_id]
            wght += nmp * weight[linear_id]
            flat += 1
        return position / wght

    @ti.func
    def nurbs_basis_interpolations_ders3d(
        self,
        start_knot_u,
        start_knot_v,
        start_knot_w,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        num_knot_w,
        xi,
        eta,
        zeta,
        knot_vector_u,
        knot_vector_v,
        knot_vector_w,
        point,
        weight,
    ):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        span_w = self.basis_w.find_span(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        dNurbsU = self.basis_u.first_ders_basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        dNurbsV = self.basis_v.first_ders_basis_funcs(start_knot_v, span_v, eta, knot_vector_v)
        dNurbsW = self.basis_w.first_ders_basis_funcs(start_knot_w, span_w, zeta, knot_vector_w)

        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_w = self.basis_w.degree + 1
        support_uv = support_u * support_v
        support_size = support_uv * support_w
        w, dwdxi, dwdet, dwdze = 0.0, 0.0, 0.0, 0.0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            i = flat // support_uv
            rem = flat - i * support_uv
            j = rem // support_u
            k = rem - j * support_u
            wind = span_w - self.basis_w.degree + i
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
            wgt = weight[linear_id]
            w += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
            dwdxi += dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
            dwdet += dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * wgt
            dwdze += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * wgt
            flat += 1

        position, dirs_u, dirs_v, dirs_w = (
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
        )
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            i = flat // support_uv
            rem = flat - i * support_uv
            j = rem // support_u
            k = rem - j * support_u
            wind = span_w - self.basis_w.degree + i
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
            fac = weight[linear_id] / (w * w)
            nmp = dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i]
            position += nmp * fac * w * point[linear_id]
            dirs_u += (dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * w - nmp * dwdxi) * fac * point[linear_id]
            dirs_v += (dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * w - nmp * dwdet) * fac * point[linear_id]
            dirs_w += (dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * w - nmp * dwdze) * fac * point[linear_id]
            flat += 1
        return position, dirs_u, dirs_v, dirs_w

    @ti.func
    def nurbs_basis_interpolations2nd_ders3d(
        self,
        start_knot_u,
        start_knot_v,
        start_knot_w,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        num_knot_w,
        xi,
        eta,
        zeta,
        knot_vector_u,
        knot_vector_v,
        knot_vector_w,
        point,
        weight,
    ):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree

        span_u = self.basis_u.find_span(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        span_v = self.basis_v.find_span(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        span_w = self.basis_w.find_span(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        nurbs_u = self.basis_u.second_ders_basis_funcs(start_knot_u, span_u, xi, knot_vector_u)
        nurbs_v = self.basis_v.second_ders_basis_funcs(start_knot_v, span_v, eta, knot_vector_v)
        nurbs_w = self.basis_w.second_ders_basis_funcs(start_knot_w, span_w, zeta, knot_vector_w)

        support_u = self.basis_u.degree + 1
        support_v = self.basis_v.degree + 1
        support_w = self.basis_w.degree + 1
        support_uv = support_u * support_v
        support_size = support_uv * support_w
        w = 0.0
        dwdxi, dwdet, dwdze = 0.0, 0.0, 0.0
        d2wdxi, d2wdet, d2wdze = 0.0, 0.0, 0.0
        d2wdxe, d2wdez, d2wdxz = 0.0, 0.0, 0.0
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            i = flat // support_uv
            rem = flat - i * support_uv
            j = rem // support_u
            k = rem - j * support_u
            wind = span_w - self.basis_w.degree + i
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
            wgt = weight[linear_id]

            w += nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * wgt
            dwdxi += nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[0, i] * wgt
            dwdet += nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[0, i] * wgt
            dwdze += nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[1, i] * wgt
            d2wdxi += nurbs_u[2, k] * nurbs_v[0, j] * nurbs_w[0, i] * wgt
            d2wdet += nurbs_u[0, k] * nurbs_v[2, j] * nurbs_w[0, i] * wgt
            d2wdze += nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[2, i] * wgt
            d2wdxe += nurbs_u[1, k] * nurbs_v[1, j] * nurbs_w[0, i] * wgt
            d2wdez += nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[1, i] * wgt
            d2wdxz += nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[1, i] * wgt
            flat += 1

        position, dirs_u, dirs_v, dirs_w = (
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
        )
        ddirs_uu, ddirs_vv, ddirs_ww = (
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
        )
        ddirs_uv, ddirs_vw, ddirs_uw = (
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
            ti.Vector.zero(float, self.dimension),
        )
        # dirsU -> dNdxi, dirsV -> dNdeta, dirs[2] -> dNdzeta
        # ddirs[0] -> d2Ndxi2, ddirs[1] -> d2Ndeta2, ddirs[2] -> d2Ndzeta, ddirs[3] -> d2Ndxideta, ddirs[4] -> d2Ndetadzeta, ddirs[5] -> d2Ndxidzeta
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = span_u - self.basis_u.degree
        flat = 0
        while flat < support_size:
            i = flat // support_uv
            rem = flat - i * support_uv
            j = rem // support_u
            k = rem - j * support_u
            wind = span_w - self.basis_w.degree + i
            vind = span_v - self.basis_v.degree + j
            linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i]

            position += nmp * fac * w * point[linear_id]
            dirs_u += (nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[0, i] * w - nmp * dwdxi) * fac * point[linear_id]
            dirs_v += (nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[0, i] * w - nmp * dwdet) * fac * point[linear_id]
            dirs_w += (nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[1, i] * w - nmp * dwdze) * fac * point[linear_id]
            ddirs_uu += (
                (
                    nurbs_u[2, k] * nurbs_v[0, j] * nurbs_w[0, i] * inv_w
                    - 2 * nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdxi * inv_w2
                    - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * d2wdxi * inv_w2
                    + 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdxi * dwdxi * inv_w3
                )
                * wgt
                * point[linear_id]
            )
            ddirs_vv += (
                (
                    nurbs_u[0, k] * nurbs_v[2, j] * nurbs_w[0, i] * inv_w
                    - 2 * nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[0, i] * dwdet * inv_w2
                    - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * d2wdet * inv_w2
                    + 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdet * dwdet * inv_w3
                )
                * wgt
                * point[linear_id]
            )
            ddirs_ww += (
                (
                    nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[2, i] * inv_w
                    - 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[1, i] * dwdze * inv_w2
                    - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * d2wdze * inv_w2
                    + 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdze * dwdze * inv_w3
                )
                * wgt
                * point[linear_id]
            )
            ddirs_uv += (
                (
                    nurbs_u[1, k] * nurbs_v[1, j] * nurbs_w[0, i] * inv_w
                    - nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdet * inv_w2
                    - nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[0, i] * dwdxi * inv_w2
                    - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * d2wdxe * inv_w2
                    + 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdxi * dwdet * inv_w3
                )
                * wgt
                * point[linear_id]
            )
            ddirs_vw += (
                (
                    nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[1, i] * inv_w
                    - nurbs_u[0, k] * nurbs_v[1, j] * nurbs_w[0, i] * dwdze * inv_w2
                    - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[1, i] * dwdet * inv_w2
                    - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * d2wdez * inv_w2
                    + 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdet * dwdze * inv_w3
                )
                * wgt
                * point[linear_id]
            )
            ddirs_uw += (
                (
                    nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[1, i] * inv_w
                    - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[1, i] * dwdxi * inv_w2
                    - nurbs_u[1, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdze * inv_w2
                    - nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * d2wdxz * inv_w2
                    + 2 * nurbs_u[0, k] * nurbs_v[0, j] * nurbs_w[0, i] * dwdxi * dwdze * inv_w3
                )
                * wgt
                * point[linear_id]
            )
            flat += 1
        return position, dirs_u, dirs_v, dirs_w, ddirs_uu, ddirs_vv, ddirs_ww, ddirs_uv, ddirs_vw, ddirs_uw
