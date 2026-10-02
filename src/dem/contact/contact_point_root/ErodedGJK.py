import taichi as ti

from src.utils.constants import DBL_MAX, DBL_EPSILON, Threshold
from src.utils.VectorFunction import Squared, equal, coord_local2global, global2local
from src.utils.TypeDefination import vec3f


@ti.func
def valid(s, m_all_bits, m_det):
    is_touch = 1
    for i in ti.static(range(4)):
        bit = 1 << i
        if is_touch == 1 and m_all_bits & bit != 0x0:
            if s & bit != 0x0:
                if m_det[s, i] <= 0.:
                    is_touch = 0
            elif m_det[s | bit, i] > 0.:
                is_touch = 0
    return is_touch

@ti.func
def compute_vector(s, m_det, m_y, m_ylen2):
    v = vec3f(0, 0, 0)
    m_maxlen2 = 0.
    sum = 0.
    for i in ti.static(range(4)):
        bit = 1 << i
        if s & bit != 0x0:
            sum += m_det[s, i]
            if m_maxlen2 < m_ylen2[i]:
                m_maxlen2 = m_ylen2[i]
            v += vec3f(m_y[i, 0], m_y[i, 1], m_y[i, 2]) * m_det[s, i]
    return v / sum, m_maxlen2

@ti.func
def in_simplex(w, m_all_bits, m_y):
    is_in_simplex = 0
    for i in ti.static(range(4)):
        bit = 1 << i
        if is_in_simplex == 0 and m_all_bits & bit != 0x0 and equal(w, vec3f(m_y[i, 0], m_y[i, 1], m_y[i, 2])):
            is_in_simplex = 1
    return is_in_simplex

@ti.func
def compute_contact_points(m_bits, m_det, m_p, m_q):
    sum = 0.
    p1, p2 = vec3f(0., 0., 0.), vec3f(0., 0., 0.)
    for i in ti.static(range(4)):
        bit = 1 << i
        if m_bits & bit != 0x0:
            sum += m_det[m_bits, i]
            p1 += vec3f(m_p[i, 0], m_p[i, 1], m_p[i, 2]) * m_det[m_bits, i]
            p2 += vec3f(m_q[i, 0], m_q[i, 1], m_q[i, 2]) * m_det[m_bits, i]
    s = 1. / sum
    p1 *= s
    p2 *= s
    return p1, p2


@ti.func
def refine_common_normal_support_pair(
    initial_unitSA,
    mass_center1,
    mass_center2,
    rotate_matrix1,
    rotate_matrix2,
    params1,
    params2,
    surface1,
    surface2,
):
    """Refine a deep-contact support pair toward a shared surface normal."""
    # ``initial_unitSA`` points from body 2 toward body 1; support directions
    # use the opposite convention.  Minimise the directional overlap
    # h(n) = n.(support_1(n) - support_2(-n)) on the unit sphere.  By the
    # envelope theorem its tangent gradient is d - (d.n)n.  At a stationary
    # point, p - q is parallel to the shared support normal, matching the
    # common-normal condition used by implicit-surface overlap methods.
    support_axis = -initial_unitSA
    pa, pb = vec3f(0., 0., 0.), vec3f(0., 0., 0.)
    for _ in range(80):
        local_normal1 = global2local(support_axis, 1., rotate_matrix1)
        local_normal2 = global2local(-support_axis, 1., rotate_matrix2)
        pa = coord_local2global(
            1., rotate_matrix1, surface1.support(local_normal1, params1), mass_center1
        )
        pb = coord_local2global(
            1., rotate_matrix2, surface2.support(local_normal2, params2), mass_center2
        )
        distance_vector = pa - pb
        if Squared(distance_vector) > Threshold * Threshold:
            directional_overlap = distance_vector.dot(support_axis)
            tangent_gradient = distance_vector - directional_overlap * support_axis
            characteristic_length = ti.max(
                (pa - mass_center1).norm() + (pb - mass_center2).norm(), Threshold
            )
            next_axis = support_axis - 0.3 * tangent_gradient / characteristic_length
            if Squared(next_axis) > Threshold * Threshold:
                support_axis = next_axis.normalized(Threshold)

    local_normal1 = global2local(support_axis, 1., rotate_matrix1)
    local_normal2 = global2local(-support_axis, 1., rotate_matrix2)
    pa = coord_local2global(
        1., rotate_matrix1, surface1.support(local_normal1, params1), mass_center1
    )
    pb = coord_local2global(
        1., rotate_matrix2, surface2.support(local_normal2, params2), mass_center2
    )
    return pa, pb, -support_axis


@ti.func
def GJKiteration(margin1, margin2, radius1, radius2, mass_center1, mass_center2, normal, scale1, scale2, rotate_matrix1, rotate_matrix2, surface1, surface2):
    if Squared(normal) < Threshold:
        normal = mass_center1 - mass_center2
    if Squared(normal) < Threshold:
        normal = vec3f(1., 0., 0.)

    # gjk attributes
    dist2 = DBL_MAX; m_maxlen2 = 0.
    m_bits = 0x0
    m_all_bits = 0x0
    m_p, m_q, m_y = ti.Matrix.zero(float, 4, 3), ti.Matrix.zero(float, 4, 3), ti.Matrix.zero(float, 4, 3)
    m_ylen2 = ti.Vector.zero(float, 4)
    m_edge1 = ti.Matrix.zero(float, 4, 4)
    m_edge2 = ti.Matrix.zero(float, 4, 4)
    m_edge3 = ti.Matrix.zero(float, 4, 4)
    m_det = ti.Matrix.zero(float, 16, 4)

    # contact information
    contactSA = normal
    unitSA = contactSA.normalized(Threshold)
    initial_unitSA = unitSA
    is_touch = 0
    pa, pb = vec3f(0., 0., 0.), vec3f(0., 0., 0.)
    params1, params2 = surface1.physical_parameters(scale1), surface2.physical_parameters(scale2)

    # main loop
    iteration = 0
    while m_bits != 0xf and dist2 > DBL_EPSILON * m_maxlen2 and iteration < 64:
        iteration += 1
        local_normal1 = global2local(-unitSA, 1., rotate_matrix1)
        local_normal2 = global2local(unitSA, 1., rotate_matrix2)
        p = coord_local2global(1., rotate_matrix1, surface1.support(local_normal1, params1), mass_center1)
        q = coord_local2global(1., rotate_matrix2, surface2.support(local_normal2, params2), mass_center2)
        w = p - q

        delta = contactSA.dot(w)
        if delta > 0.:
            is_touch = 0
            break

        if ti.abs(delta / (contactSA.norm() * w.norm())) > 0.95 and Squared(w) < (margin1 + margin2) * (margin1 + margin2):
            pa, pb,  = p, q
            is_touch = 1
            break

        p += unitSA * margin1
        q -= unitSA * margin2
        w = p - q
        delta = contactSA.dot(w)
        if in_simplex(w, m_all_bits, m_y) or dist2 - delta <= dist2 * 1e-6:
            pa, pb = compute_contact_points(m_bits, m_det, m_p, m_q)
            pa -= unitSA * margin1
            pb += unitSA * margin2
            is_touch = 1
            break

        # add vertex
        m_last = 0
        m_last_bit = 0x1
        while (m_bits & m_last_bit) != 0x0:
            m_last += 1
            m_last_bit <<= 1
        for d in ti.static(range(3)):
            m_y[m_last, d] = w[d]
        m_ylen2[m_last] = Squared(w)
        m_all_bits = m_bits | m_last_bit

        # update_cache
        for i in ti.static(range(4)):
            bit = 1 << i
            if m_bits & bit != 0x0:
                m_edge1[i, m_last] = m_y[i, 0] - m_y[m_last, 0]
                m_edge2[i, m_last] = m_y[i, 1] - m_y[m_last, 1]
                m_edge3[i, m_last] = m_y[i, 2] - m_y[m_last, 2]
                m_edge1[m_last, i] = -m_edge1[i, m_last]
                m_edge2[m_last, i] = -m_edge2[i, m_last]
                m_edge3[m_last, i] = -m_edge3[i, m_last]
        
        # compute det
        m_det[m_last_bit, m_last] = 1.0
        m_y_last = vec3f(m_y[m_last, 0], m_y[m_last, 1], m_y[m_last, 2])
        if m_bits != 0x0:
            for i in ti.static(range(4)):
                si = 1 << i
                if m_bits & si != 0x0:
                    s2 = si | m_last_bit
                    m_yi = vec3f(m_y[i, 0], m_y[i, 1], m_y[i, 2])
                    m_det[s2, i] = vec3f(m_edge1[m_last, i], m_edge2[m_last, i], m_edge3[m_last, i]).dot(m_y_last)
                    m_det[s2, m_last] = vec3f(m_edge1[i, m_last], m_edge2[i, m_last], m_edge3[i, m_last]).dot(m_yi)

                    for j in ti.static(range(i)):
                        sj = 1 << j
                        if m_bits & sj != 0x0:
                            s3 = sj | s2
                            m_yj = vec3f(m_y[j, 0], m_y[j, 1], m_y[j, 2])
                            m_edge_ij = vec3f(m_edge1[i, j], m_edge2[i, j], m_edge3[i, j])
                            m_edge_ji = vec3f(m_edge1[j, i], m_edge2[j, i], m_edge3[j, i])
                            m_edge_j_mlast = vec3f(m_edge1[j, m_last], m_edge2[j, m_last], m_edge3[j, m_last])
                            m_det[s3, j] = m_det[s2, i] * m_edge_ij.dot(m_yi) + m_det[s2, m_last] * m_edge_ij.dot(m_y_last)
                            m_det[s3, i] = m_det[sj | m_last_bit, j] * m_edge_ji.dot(m_yj) + m_det[sj | m_last_bit, m_last] * m_edge_ji.dot(m_y_last)
                            m_det[s3, m_last] = m_det[sj | si, j] * m_edge_j_mlast.dot(m_yj) + m_det[sj | si, i] * m_edge_j_mlast.dot(m_yi)

            if m_all_bits == 0xf:
                m_edge_10 = vec3f(m_edge1[1, 0], m_edge2[1, 0], m_edge3[1, 0])
                m_edge_01 = vec3f(m_edge1[0, 1], m_edge2[0, 1], m_edge3[0, 1])
                m_edge_02 = vec3f(m_edge1[0, 2], m_edge2[0, 2], m_edge3[0, 2])
                m_edge_03 = vec3f(m_edge1[0, 3], m_edge2[0, 3], m_edge3[0, 3])
                my0 = vec3f(m_y[0, 0], m_y[0, 1], m_y[0, 2])
                my1 = vec3f(m_y[1, 0], m_y[1, 1], m_y[1, 2])
                my2 = vec3f(m_y[2, 0], m_y[2, 1], m_y[2, 2])
                my3 = vec3f(m_y[3, 0], m_y[3, 1], m_y[3, 2])

                m_det[0xf, 0] = m_det[0xe, 1] * m_edge_10.dot(my1) + m_det[0xe, 2] * m_edge_10.dot(my2) + m_det[0xe, 3] * m_edge_10.dot(my3)
                m_det[0xf, 1] = m_det[0xd, 0] * m_edge_01.dot(my0) + m_det[0xd, 2] * m_edge_01.dot(my2) + m_det[0xd, 3] * m_edge_01.dot(my3)
                m_det[0xf, 2] = m_det[0xb, 0] * m_edge_02.dot(my0) + m_det[0xb, 1] * m_edge_02.dot(my1) + m_det[0xb, 3] * m_edge_02.dot(my3)
                m_det[0xf, 3] = m_det[0x7, 0] * m_edge_03.dot(my0) + m_det[0x7, 1] * m_edge_03.dot(my1) + m_det[0x7, 2] * m_edge_03.dot(my2)

        for d in ti.static(range(3)):
            m_p[m_last, d] = p[d]
            m_q[m_last, d] = q[d]
        
        # is affinely dependent
        sum = 0.
        for i in ti.static(range(4)):
            bit = 1 << i
            if m_all_bits & bit != 0x0:
                sum += m_det[m_all_bits, i]
        if sum <= 0.:
            pa, pb = compute_contact_points(m_bits, m_det, m_p, m_q)
            pa -= unitSA * margin1
            pb += unitSA * margin2
            is_touch = 1
            break

        # is not closest
        is_closest = 0
        for ss in range(m_bits):
            s = m_bits - ss
            if (s & m_bits) == s and valid(s | m_last_bit, m_all_bits, m_det):
                m_bits = s | m_last_bit
                contactSA, m_maxlen2 = compute_vector(m_bits, m_det, m_y, m_ylen2)
                if Squared(contactSA) > Threshold * Threshold:
                    unitSA = contactSA.normalized(Threshold)
                is_closest = 1
                break

        if valid(m_last_bit, m_all_bits, m_det):
            m_bits = m_last_bit
            m_maxlen2 = m_ylen2[m_last]
            contactSA = vec3f(m_y[m_last, 0], m_y[m_last, 1], m_y[m_last, 2])
            if Squared(contactSA) > Threshold * Threshold:
                unitSA = contactSA.normalized(Threshold)
            is_closest = 1

        if not is_closest: 
            pa, pb = compute_contact_points(m_bits, m_det, m_p, m_q)
            pa -= unitSA * margin1
            pb += unitSA * margin2
            is_touch = 1
            break

        dist2 = Squared(contactSA)

    # Eroded GJK has no EPA phase. The old implementation consequently
    # returned ``no contact`` when the eroded cores touched (zero distance),
    # or formed a full simplex.  At that point intersection has already been
    # established by the GJK loop. Recover the un-eroded witnesses with a
    # common-normal refinement so a deep valid contact is not silently dropped.
    if is_touch == 0 and m_bits != 0x0 and (
        m_bits == 0xf
        or dist2 <= DBL_EPSILON * ti.max(m_maxlen2, 1.)
    ):
        pa, pb, contactSA = refine_common_normal_support_pair(
            initial_unitSA,
            mass_center1,
            mass_center2,
            rotate_matrix1,
            rotate_matrix2,
            params1,
            params2,
            surface1,
            surface2,
        )
        is_touch = 1
    elif is_touch == 1 and Squared(pa - pb) >= (margin1 + margin2) * (margin1 + margin2):
        # The ordinary eroded-GJK witness reconstruction is accurate for the
        # intended shallow-contact band.  For overlap beyond that band, apply
        # the common-normal refinement instead of un-eroding a simplex with a
        # single approximate direction.
        pa, pb, contactSA = refine_common_normal_support_pair(
            initial_unitSA,
            mass_center1,
            mass_center2,
            rotate_matrix1,
            rotate_matrix2,
            params1,
            params2,
            surface1,
            surface2,
        )
    return is_touch, pa, pb, contactSA
