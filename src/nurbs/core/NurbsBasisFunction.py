import taichi as ti


@ti.data_oriented
class NurbsBasisFunction:
    def __init__(self, degree):
        self.degree = degree

    @ti.func
    def find_span_linear(self, start_knot, num_ctrlpts, knot, knot_vector):
        span = self.degree + 1
        while span < num_ctrlpts and knot_vector[span + start_knot] <= knot:
            span += 1
        return span - 1

    @ti.func
    def find_span(self, start_knot, num_ctrlpts, knot, knot_vector):
        low = self.degree
        high = num_ctrlpts
        span = ti.cast(0.5 * (low + high), int)
        if knot > knot_vector[num_ctrlpts + start_knot]:
            span = num_ctrlpts - 1
        elif knot < knot_vector[self.degree + start_knot]:
            span = self.degree
        else:
            if knot == knot_vector[num_ctrlpts + start_knot]:
                span = num_ctrlpts - 1
            else:
                while (
                    knot < knot_vector[span + start_knot] or knot >= knot_vector[span + 1 + start_knot]
                ) and high != low:
                    if knot < knot_vector[span + start_knot]:
                        high = span
                    else:
                        low = span
                    span = ti.cast(0.5 * (low + high), int)
        return span

    @ti.func
    def basis_funcs(self, start_knot, span, knot, knot_vector):
        nurbs = ti.Vector.zero(float, self.degree + 1)
        left = ti.Vector.zero(float, self.degree + 1)
        right = ti.Vector.zero(float, self.degree + 1)
        nurbs[0] = 1.0

        i = 1
        while i < self.degree + 1:
            left[i] = knot - knot_vector[span + 1 - i + start_knot]
            right[i] = knot_vector[span + i + start_knot] - knot
            saved = 0.0

            for j in range(i):
                temp = nurbs[j] / (right[j + 1] + left[i - j])
                nurbs[j] = saved + right[j + 1] * temp
                saved = left[i - j] * temp
            nurbs[i] = saved
            i += 1
        return nurbs

    @ti.func
    def first_ders_basis_funcs(self, start_knot, span, knot, knot_vector):
        der_nurbs = ti.Matrix.zero(float, 2, self.degree + 1)
        left = ti.Vector.zero(float, self.degree + 1)
        right = ti.Vector.zero(float, self.degree + 1)
        ndu = ti.Matrix.zero(float, self.degree + 1, self.degree + 1)
        a = ti.Matrix.zero(float, 2, self.degree + 1)

        ndu[0, 0] = 1.0
        i = 1
        while i < self.degree + 1:
            left[i] = knot - knot_vector[span + 1 - i + start_knot]
            right[i] = knot_vector[span + i + start_knot] - knot
            saved = 0.0
            for j in range(i):
                ndu[i, j] = right[j + 1] + left[i - j]
                temp = ndu[j, i - 1] / ndu[i, j]

                ndu[j, i] = saved + right[j + 1] * temp
                saved = left[i - j] * temp
            ndu[i, i] = saved
            i += 1

        i = 0
        while i < self.degree + 1:
            der_nurbs[0, i] = ndu[i, self.degree]
            i += 1

        r = 0
        while r < self.degree + 1:
            s1 = 0
            s2 = 1
            a[0, 0] = 1.0
            k = 1

            # the first derviative
            d = 0.0
            rk = r - k
            pk = self.degree - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else self.degree - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]

            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            der_nurbs[k, r] = d
            s1, s2 = s2, s1
            r += 1

        r = self.degree
        j = 0
        while j < self.degree + 1:
            der_nurbs[1, j] *= r
            j += 1
        r *= self.degree - 1
        return der_nurbs

    @ti.func
    def second_ders_basis_funcs(self, start_knot, span, knot, knot_vector):
        der_nurbs = ti.Matrix.zero(float, 3, self.degree + 1)
        left = ti.Vector.zero(float, self.degree + 1)
        right = ti.Vector.zero(float, self.degree + 1)
        ndu = ti.Matrix.zero(float, self.degree + 1, self.degree + 1)
        a = ti.Matrix.zero(float, 2, self.degree + 1)

        ndu[0, 0] = 1.0
        i = 1
        while i < self.degree + 1:
            left[i] = knot - knot_vector[span + 1 - i + start_knot]
            right[i] = knot_vector[span + i + start_knot] - knot
            saved = 0.0
            for j in range(i):
                ndu[i, j] = right[j + 1] + left[i - j]
                temp = ndu[j, i - 1] / ndu[i, j]

                ndu[j, i] = saved + right[j + 1] * temp
                saved = left[i - j] * temp
            ndu[i, i] = saved
            i += 1

        i = 0
        while i < self.degree + 1:
            der_nurbs[0, i] = ndu[i, self.degree]
            i += 1

        r = 0
        while r < self.degree + 1:
            s1 = 0
            s2 = 1
            a[0, 0] = 1.0

            for k in ti.static(range(1, 3)):
                d = 0.0
                rk = r - k
                pk = self.degree - k
                if r >= k:
                    a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                    d = a[s2, 0] * ndu[rk, pk]
                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if r - 1 <= pk else self.degree - r
                for j in range(j1, j2 + 1):
                    a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                    d += a[s2, j] * ndu[rk + j, pk]

                if r <= pk:
                    a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                    d += a[s2, k] * ndu[r, pk]
                der_nurbs[k, r] = d
                s1, s2 = s2, s1
            r += 1

        r = self.degree
        for k in ti.static(range(1, 3)):
            j = 0
            while j < self.degree + 1:
                der_nurbs[k, j] *= r
                j += 1
            r *= self.degree - k
        return der_nurbs
