import numpy as np


def find_span_linear(num_ctrlpts, knot, degree, knot_vector):
    span = degree + 1  # Knot span index starts from zero
    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1
    return span - 1


def find_span(num_ctrlpts, knot, degree, knot_vector):
    low = degree
    high = num_ctrlpts
    span = int(0.5 * (low + high))
    if knot > knot_vector[num_ctrlpts]:
        return num_ctrlpts - 1
    elif knot < knot_vector[degree]:
        return degree
    
    if knot == knot_vector[num_ctrlpts]: 
        span = num_ctrlpts - 1
    else:
        while (knot < knot_vector[span] or knot >= knot_vector[span + 1]):
            if knot < knot_vector[span]: 
                high = span
            else:                
                low = span
            span = int(0.5 * (low + high))
    return span


def find_element(knot_vector):
    return np.unique(np.asarray(knot_vector).copy())


def find_multiplicity(knot_vector):
    _, multiplicity = np.unique(np.asarray(knot_vector).copy(), return_counts=True)
    return multiplicity

def basis_functions(span, knot, degree, knot_vector):
    Nurbs = np.zeros(degree + 1)
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)
    Nurbs[0] = 1.0

    for i in range(1, degree + 1):
        left[i] = knot - knot_vector[span + 1 - i]
        right[i] = knot_vector[span + i] - knot
        saved = 0.

        for j in range(i):
            temp = Nurbs[j] / (right[j + 1] + left[i - j])
            Nurbs[j] = saved + right[j + 1] * temp
            saved = left[i - j] * temp
        Nurbs[i] = saved
    return Nurbs


def basis_function_ders(span, knot, degree, knot_vector, order=1):
    derNurbs = np.zeros((order + 1, degree + 1))
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)
    ndu = np.zeros((degree + 1, degree + 1))
    a = np.zeros((2, degree + 1))

    ndu[0, 0] = 1.
    for i in range(1, degree + 1):
        left[i] = knot - knot_vector[span + 1 - i]
        right[i] = knot_vector[span + i] - knot
        saved = 0.
        for j in range(i):
            ndu[i, j] = right[j + 1] + left[i - j]
            temp = ndu[j, i - 1] / ndu[i, j]

            ndu[j, i] = saved + right[j + 1] * temp
            saved = left[i - j] * temp
        ndu[i, i] = saved

    for i in range(degree + 1):
        derNurbs[0, i] = ndu[i, degree]

    for r in range(degree + 1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.

        for k in range(1, order + 1):
            d = 0
            rk = r - k
            pk = degree - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else degree - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            derNurbs[k, r] = d
            s1, s2 = s2, s1
    
    r = degree
    for k in range(1, order + 1):
        for j in range(degree + 1):
            derNurbs[k, j] *= r
        r *= (degree - k)
    return derNurbs
    

def one_basis_function(knot_span, knot, degree, knot_vector):
    Nurbs = np.zeros(degree + 1)
    if (knot_span == 0 and knot == knot_vector[0]) or (knot_span == knot_vector.shape[0] - degree - 2 and knot == knot_vector[knot_vector.shape[0] - 1]):
        return 1.0
    if knot < knot_vector[knot_span] or knot >= knot_vector[knot_span + degree + 1]:
        return 0.0
    else:
        for i in range(degree + 1):
            if knot >= knot_vector[knot_span + i] and knot < knot_vector[knot_span + i + 1]:
                Nurbs[i] = 1.0
            else:
                Nurbs[i] = 0.0
        for k in range(1, degree + 1):
            saved = 0.0
            if Nurbs[0] != 0.0: 
                saved = (knot - knot_vector[knot_span]) * Nurbs[0] / (knot_vector[knot_span + k] - knot_vector[knot_span])
            for j in range(degree - k + 1):
                Uleft = knot_vector[knot_span + j + 1]
                Uright = knot_vector[knot_span + j + k + 1]
                if Nurbs[j + 1] == 0.0:
                    Nurbs[j] = saved
                    saved = 0.0
                else:
                    temp = Nurbs[j + 1] / (Uright - Uleft)
                    Nurbs[j] = saved + (Uright - knot) * temp
                    saved = (knot - Uleft) * temp
    return Nurbs[0]


def one_basis_function_ders_1st(knot_span, knot, degree, knot_vector, order=1):
    N = np.zeros((degree + 1, degree + 1))
    ND = np.zeros(degree + 1)
    
    basis, ders = 0., np.zeros(order + 1)
    if knot < knot_vector[knot_span] or knot >= knot_vector[knot_span + degree + 1]:
        return basis, ders

    for i in range(degree + 1):
        if knot_vector[knot_span + i] <= knot < knot_vector[knot_span + i + 1]:
            N[i, 0] = 1.0
    
    for k in range(1, degree + 1):
        if N[0, k - 1] == 0.:
            saved = 0.0
        else:
            saved = (knot - knot_vector[knot_span]) * N[0, k - 1] / (knot_vector[knot_span + k] - knot_vector[knot_span])
        for j in range(degree - k + 1):
            Uleft = knot_vector[knot_span + j + 1]
            Uright = knot_vector[knot_span + j + k + 1]
            if N[j + 1, k - 1] == 0.0:
                N[j, k] = saved
                saved = 0.0
            else:
                temp = N[j + 1, k - 1] / (Uright - Uleft)
                N[j, k] = saved + (Uright - knot) * temp
                saved = (knot - Uleft) * temp

    basis = N[0, degree]
    for k in range(1, order + 1):
        for j in range(0, k + 1):
            ND[j] = N[j, degree - k]
        for jj in range(1, k + 1):
            saved = 0.0
            if ND[0] != 0.0: 
                saved = ND[0] / (knot_vector[knot_span + degree - k + jj] - knot_vector[knot_span])
            for j in range(k - jj + 1):
                Uleft = knot_vector[knot_span + j + 1]
                Uright = knot_vector[knot_span + j + degree + jj]
                if ND[j + 1] == 0.0:
                    ND[j] = (degree - k + jj) * saved
                    saved = 0.0
                else:
                    temp = ND[j + 1] / (Uright - Uleft)
                    ND[j] = (degree - k + jj) * (saved - temp)
                    saved = temp
            ders[k] = ND[0]
    return basis, ders