import taichi as ti


HEAPSIZE = 64
@ti.func
def heap_insert(heap, inserted_queue, current_size):
    pos = 0
    l = 0
    r = current_size
    while l < r:
        mid = (l + r) // 2
        if heap[mid][2] < inserted_queue[2]:
            l = mid + 1
        else:
            r = mid
    pos = l

    n = current_size
    while n > pos:
        heap[n][0] = heap[n - 1][0]
        heap[n][1] = heap[n - 1][1]
        heap[n][2] = heap[n - 1][2]
        n -= 1

    heap[pos][0] = inserted_queue[0]
    heap[pos][1] = inserted_queue[1]
    heap[pos][2] = inserted_queue[2]
    current_size += 1
    assert current_size <= HEAPSIZE
    return heap, current_size

@ti.func
def point2surface_distance_interval_splitting(start_knot_u, start_knot_v, start_ctrlpt, num_knot_u, num_knot_v, knot_vector_u, knot_vector_v, ctrlpts, weight, point, basis):
    pass

@ti.func
def point2surface_distance_augment_lagrangian(start_knot_u, start_knot_v, start_ctrlpt, num_knot_u, num_knot_v, knot_vector_u, knot_vector_v, ctrlpts, weight, point, basis):
    num_ctrlpts_u = num_knot_u - basis.basis_u.degree - 1
    num_ctrlpts_v = num_knot_v - basis.basis_v.degree - 1

    u, v, minDist = 0., 0., 1e15
    for j in range(num_ctrlpts_v):
        for i in range(num_ctrlpts_u):
            linear_id = i + j * num_ctrlpts_u
            dist2 = Squared(ctrlpts[linear_id + start_ctrlpt] - point)
            if dist2 < minDist:
                u = 0.5 * (knot_vector_u[start_knot_u + i] + knot_vector_u[start_knot_u + i + 1 + basis.basis_u.degree])
                v = 0.5 * (knot_vector_v[start_knot_v + j] + knot_vector_v[start_knot_v + j + 1 + basis.basis_v.degree])
                minDist = dist2
    
    iter, du, dv, distance, residual1 = 0, 0., 0., 0., ti.Vector.zero(float, point.n)
    while iter < 50:
        position, dirsU, dirsV, ddirsUU, ddirsVV, ddirsUV = basis.NurbsBasisInterpolations2ndDers2d(start_knot_u, start_knot_v, start_ctrlpt, num_knot_u, num_knot_v, u, v, knot_vector_u, knot_vector_v, ctrlpts, weight)
        residual1 = position - point
        if Squared(residual1) < TOL:
            distance = 0.
            break
        
        residual2 = residual1.dot(dirsU) / (residual1.norm() * dirsU.norm())
        residual3 = residual1.dot(dirsV) / (residual1.norm() * dirsV.norm())
        if abs(residual2) < TOL and abs(residual3) < TOL:
            distance = residual1.norm()
            break
        
        f = residual1.dot(dirsU)
        g = residual1.dot(dirsV)
        a = Squared(dirsU) + residual1.dot(ddirsUU)
        b = dirsU.dot(dirsV) + residual1.dot(ddirsUV)
        d = Squared(dirsV) + residual1.dot(ddirsVV)
        inv_jac = inverse_matrix_2x2(mat2x2([a, b], [b, d]))
        delta = inv_jac @ -vec2f(f, g)
        du, dv = delta[0], delta[1]

        u = clamp(knot_vector_u[start_knot_u + basis.basis_u.degree], knot_vector_u[start_knot_u + num_ctrlpts_u], u + du)
        v = clamp(knot_vector_v[start_knot_v + basis.basis_v.degree], knot_vector_v[start_knot_v + num_ctrlpts_v], v + dv)

        r4 = du * du * Squared(dirsU) + dv * dv * Squared(dirsV)
        if r4 < TOL: 
            distance = residual1.norm()
            break
        iter += 1
    if iter == 50:
        distance = residual1.norm()
    return u, v, distance, residual1