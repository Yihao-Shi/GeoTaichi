import numpy as np

from src.nurbs.Nurbs import find_span, find_multiplicity, find_element
from src.nurbs.NurbsBasis import (BsplineBasisInterpolations1d, BsplineBasisInterpolations2d, BsplineBasisInterpolations2ndDers1d, BsplineBasisInterpolations2ndDers2d,
                                  NurbsBasisInterpolations1d, NurbsBasisInterpolations2d, NurbsBasisInterpolations2ndDers1d, NurbsBasisInterpolations2ndDers2d)
from src.nurbs.Utilities import check_closed_boundary, check_open_boundary, TOL


def knot_insertion(degree, knot_vector, ctrlpts, addition_knot=list()):
    if not isinstance(addition_knot, (list, tuple, np.ndarray)):
        raise RuntimeError(f"The insertions {addition_knot} must be a list, tuple or a ndarray")
    for idx, val in enumerate(addition_knot):
        if val < 0:
            raise RuntimeError(f'Number of insertions {val} in {addition_knot} must be a positive integer value')
    addition_knot = np.asarray(addition_knot)
    knot_vector = np.asarray(knot_vector)
        
    for val, counts in zip(*np.unique(addition_knot, return_counts=True)):
        multiplicity = np.sum(np.isclose(knot_vector, val, rtol=1e-14, atol=1e-14))

        if counts > degree - multiplicity + 1:
            raise RuntimeError("Knot " + str(val) + " cannot be inserted " + str(counts) + " times in knot vector " + str(knot_vector) + " with degree " + str(degree) + " and multiplicity " + str(multiplicity))

        num_ctrlpts = len(ctrlpts) - 1
        span = find_span(num_ctrlpts, val, degree, knot_vector)
        new_knot_vector, new_ctrlpts = knot_insertion_update(degree, val, knot_vector, ctrlpts, span, counts, multiplicity)

        # Update curve
        ctrlpts = new_ctrlpts.copy()
        knot_vector = new_knot_vector.copy()
    return knot_vector, ctrlpts


def knot_insertion_update(degree, knot, knot_vector, ctrlpts, span, counts, multiplicity):
    np_ctrl = len(ctrlpts)
    nq_ctrl = np_ctrl + counts

    knot_vector_size = len(knot_vector)
    knot_vector_updated = np.zeros(knot_vector_size + counts)
    knot_vector_updated[:span + 1] = knot_vector[:span + 1]
    knot_vector_updated[span + 1:span + 1 + counts] = knot
    knot_vector_updated[span + 1 + counts:] = knot_vector[span + 1:]

    ctrlpts_new = np.zeros((nq_ctrl, ctrlpts.shape[1]))
    ctrlpts_new[:span - degree + 1] = ctrlpts[:span - degree + 1]
    ctrlpts_new[span + counts - multiplicity:] = ctrlpts[span - multiplicity:]
    temp = ctrlpts[span - degree:span + 1]

    for j in range(1, counts + 1):
        L = span - degree + j
        K = degree - j - multiplicity + 1
        alpha = (knot - knot_vector[L:L + K]) / (knot_vector[span + 1:span + 1 + K] - knot_vector[L:L + K])
        temp[:K] = alpha[:, np.newaxis] * temp[1:K + 1] + (1.0 - alpha[:, np.newaxis]) * temp[:K]
        ctrlpts_new[L] = temp[0].copy()
        ctrlpts_new[span + counts - j - multiplicity] = temp[degree - j - multiplicity].copy()

    L = span - degree + counts
    ctrlpts_new[L + 1:span - multiplicity] = np.copy(temp[L + 1 - L:span - multiplicity - L])
    return knot_vector_updated, ctrlpts_new

def knot_refinement(degree, knot_vector, ctrlpts, density=1, addition_knot=list()):
    if not isinstance(density, int):
        raise ValueError(f"Density {density} value must be an integer")

    if density < 1:
        raise ValueError(f"Density {density} value cannot be less than 1")
    
    if not isinstance(addition_knot, (list, tuple, np.ndarray)):
            raise RuntimeError(f"The insertions {addition_knot} must be a list, tuple or a ndarray")
    for idx, val in enumerate(addition_knot):
        if val < 0:
            raise RuntimeError(f'Number of insertions {val} in {addition_knot} must be a positive integer value')
    
    knot_list = list()
    if len(addition_knot) > 0:
        knot_list += list(addition_knot)
    knot_list = sorted(set(knot_list))
    
    for _ in range(density):
        sorted_knot_vector = sorted(set(list(knot_vector)))
        multiplicity = find_multiplicity(sorted_knot_vector + knot_list)
        element = find_element(sorted_knot_vector + knot_list)
        num_ele = multiplicity.shape[0] - 1
        for j in range(num_ele):
            knot_list += [element[j] + 0.5 * (element[j + 1] - element[j])]
    knot_list = np.sort(np.asarray(knot_list))

    degree_w = len(knot_list) - 1
    num_ctrlpts = len(ctrlpts) - 1
    num_knot = num_ctrlpts + degree + 1
    a = find_span(num_ctrlpts, knot_list[0], degree, knot_vector)
    b = find_span(num_ctrlpts, knot_list[degree_w], degree, knot_vector) + 1

    new_knot, new_ctrlpts = np.zeros(num_knot + degree_w + 2), np.zeros((num_ctrlpts + degree_w + 2, ctrlpts.shape[1]))
    new_ctrlpts[0: a - degree + 1] = ctrlpts[0: a - degree + 1]
    new_ctrlpts[b + degree_w: num_ctrlpts + degree_w + 2] = ctrlpts[b - 1: num_ctrlpts + 1]
    new_knot[0: a + 1] = knot_vector[0: a + 1]
    new_knot[b + degree + degree_w + 1: num_knot + degree_w + 2] = knot_vector[b+degree: num_knot + 1]

    i = b + degree - 1
    s = b + degree + degree_w
    for j in range(degree_w, -1, -1):
        while knot_list[j] <= knot_vector[i] and i > a:
            new_ctrlpts[s-degree-1] = ctrlpts[i-degree-1]
            new_knot[s] = knot_vector[i]
            s -= 1
            i -= 1
        new_ctrlpts[s-degree-1] = new_ctrlpts[s-degree]
        for l in range(1, degree+1):
            ind = s - degree + l
            alfa = new_knot[s + l] - knot_list[j]
            if abs(alfa) < TOL:
                new_ctrlpts[ind - 1] = new_ctrlpts[ind]
            else:
                alfa /= (new_knot[s + l] - knot_vector[i - degree + l])
                new_ctrlpts[ind - 1] = alfa * new_ctrlpts[ind - 1] + (1. - alfa) * new_ctrlpts[ind]
        new_knot[s] = knot_list[j]
        s -= 1
    return new_knot, new_ctrlpts


def initial_guess_curve(point, num_ctrlpts, degree, knot_vector, ctrlpts, weight, func):
    # Set array of values of parameter u to evaluate
    num_u_spans = 2 * num_ctrlpts
    eval_u = 1 / (num_u_spans - 1) * np.array(list(range(0, num_u_spans)))
    u0 = eval_u[0]

    # Set minimum value as a high number to start with
    min_val = 1e15

    # Loop through list of evaluation knots
    for u in range(0, num_u_spans):
        r = point - func(eval_u[u], degree, knot_vector, ctrlpts, weight)
        normr = np.linalg.norm(r, 2)
        if normr < min_val:
            min_val = normr
            u0 = eval_u[u]
    return u0


def initial_guess_surf(point, num_ctrlpts_u, num_ctrlpts_v, degree_u, degree_v, knot_vector_u, knot_vector_v, ctrlpts, weight, func):
    # Set array of values of (u, v) to evaluate
    num_u_spans = 2 * num_ctrlpts_u
    num_v_spans = 2 * num_ctrlpts_v
    eval_u = 1 / (num_u_spans - 1) * np.array(list(range(0, num_u_spans)))
    eval_v = 1 / (num_v_spans - 1) * np.array(list(range(0, num_v_spans)))
    u0 = eval_u[0]
    v0 = eval_v[0]

    # Set minimum value
    min_val = 1e15

    # Evaluate surface. Careful, this assumes the surface is open in both directions.
    for u in range(0, num_u_spans):
        for v in range(0, num_v_spans):
            r = point - func(eval_u[u], eval_v[v], degree_u, degree_v, knot_vector_u, knot_vector_v, ctrlpts, weight)
            normr = np.linalg.norm(r, 2)
            if normr < min_val:
                min_val = normr
                u0 = eval_u[u]
                v0 = eval_v[v]
    return u0, v0


def curve_inversion(point, degree, knot_vector, ctrlpts, weight=None, get_knot=True, get_distance=True, closed=False):
    func, dfunc = NurbsBasisInterpolations1d, NurbsBasisInterpolations2ndDers1d
    if weight is None:
        func, dfunc = BsplineBasisInterpolations1d, BsplineBasisInterpolations2ndDers1d
    ctrlpts = np.array(ctrlpts)
    if ctrlpts.ndim == 1:
        ctrlpts = ctrlpts.reshape((-1, 1))
    nPtsX = knot_vector.shape[0] - degree - 1

    u = initial_guess_curve(point, nPtsX, degree, knot_vector, ctrlpts, weight, func)
    iter, du, distance = 0, 0., 0.
    while iter < 50:
        u = np.clip(u, knot_vector[degree], knot_vector[nPtsX])
        position, dirs, ddirs = dfunc(u, degree, knot_vector, ctrlpts, weight)
        residual1 = position - point
        if np.linalg.norm(residual1) < TOL:
            if get_distance: distance = 0.
            break
        
        residual2 = np.dot(residual1, dirs[0]) / (np.linalg.norm(residual1) * np.linalg.norm(dirs[0]))
        if abs(residual2) < TOL:
            if get_distance: distance = np.linalg.norm(residual1)
            break
        
        f = np.dot(residual1, dirs[0])
        det = np.dot(dirs[0], dirs[0]) + np.dot(residual1, ddirs[0])
        du = f / det

        if closed is False:
            new_u = check_open_boundary(u - du, knot_vector)
            du = u - new_u
            u = new_u.copy()

        residual4 = du * np.linalg.norm((dirs[0]))
        if residual4 < TOL: 
            if get_distance: distance = np.linalg.norm(residual1)
            break

        if closed:
            u = check_closed_boundary(u - du, knot_vector)
        iter += 1

        if iter == 50:
            raise RuntimeError("Failed to find the closest point on spline curve!")
        
    if get_distance and get_knot: 
        return u, distance
    elif get_knot: 
        return u
    elif get_distance:
        return distance


def surface_inversion(point, degree_u, degree_v, knot_vector_u, knot_vector_v, ctrlpts, weight=None, get_knot=True, get_distance=True, closed=False):
    func, dfunc = NurbsBasisInterpolations2d, NurbsBasisInterpolations2ndDers2d
    if weight is None:
        func, dfunc = BsplineBasisInterpolations2d, BsplineBasisInterpolations2ndDers2d

    ctrlpts = np.array(ctrlpts)
    if ctrlpts.ndim == 1:
        ctrlpts = ctrlpts.reshape((-1, 1))
    nPtsX = knot_vector_u.shape[0] - degree_u - 1
    nPtsY = knot_vector_v.shape[0] - degree_v - 1

    u, v = initial_guess_surf(point, nPtsX, nPtsY, degree_u, degree_v, knot_vector_u, knot_vector_v, ctrlpts, weight, func)
    iter, du, dv, distance = 0, 0., 0., 0.
    while iter < 100:
        u = np.clip(u, knot_vector_u[0], knot_vector_u[-1])
        v = np.clip(v, knot_vector_v[0], knot_vector_v[-1])
        position, dirs, ddirs = dfunc(u, v, degree_u, degree_v, knot_vector_u, knot_vector_v, ctrlpts, weight)
        residual1 = position - point
        if np.linalg.norm(residual1) < TOL:
            if get_distance: distance = 0.
            break
        
        residual2 = np.dot(residual1, dirs[0]) / (np.linalg.norm(residual1) * np.linalg.norm(dirs[0]))
        residual3 = np.dot(residual1, dirs[1]) / (np.linalg.norm(residual1) * np.linalg.norm(dirs[1]))
        if abs(residual2) < TOL and abs(residual3) < TOL:
            if get_distance: distance = np.linalg.norm(residual1)
            break
        
        f = np.dot(residual1, dirs[0])
        g = np.dot(residual1, dirs[1])
        a = np.dot(dirs[0], dirs[0]) + np.dot(residual1, ddirs[0])
        b = np.dot(dirs[0], dirs[1]) + np.dot(residual1, ddirs[2])
        d = np.dot(dirs[1], dirs[1]) + np.dot(residual1, ddirs[1])
        jacobian = np.array([[a, b], [b, d]])
        kappa = -np.array([[f], [g]])
        delta = np.matmul(np.linalg.inv(jacobian), kappa)
        du, dv = delta[0, 0], delta[1, 0]

        if closed is False:
            new_u = check_open_boundary(u + du, knot_vector_u)
            new_v = check_open_boundary(v + dv, knot_vector_v)
            du = new_u - u
            dv = new_v - v
            u = new_u.copy()
            v = new_v.copy()

        residual4 = np.linalg.norm(du * dirs[0] + dv * dirs[1])
        if residual4 < TOL: 
            if get_distance: distance = np.linalg.norm(residual1)
            break

        if closed:
            u = check_closed_boundary(u + du, knot_vector_u)
            v = check_closed_boundary(v + dv, knot_vector_v)
        iter += 1

        if iter == 50:
            raise RuntimeError(f"Failed to find the closest point on spline surface! Residual error is {np.linalg.norm(residual1)}")
    
    if get_distance and get_knot: 
        return u, v, distance
    elif get_knot: 
        return u, v
    elif get_distance:
        return distance