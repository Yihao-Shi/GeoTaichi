import taichi as ti
import numpy as np
import trimesh

from src.utils.constants import PI, THRESHOLD
from src.utils.Quaternion import rodrigues_rotation_matrix
from src.utils.ScalarFunction import clamp, sgn
from src.utils.TypeDefinition import vec3f
from src.utils.VectorFunction import dot2


@ti.func
def sphere_triangle_intersection_area(position, radius, vertice1, vertice2, vertice3, norm):
    area1 = process_segment(position, radius, vertice1, vertice2, norm)
    area2 = process_segment(position, radius, vertice2, vertice3, norm)
    area3 = process_segment(position, radius, vertice3, vertice1, norm)
    return area1 + area2 + area3


@ti.func
def process_segment(position, radius, initial_vertex, final_vertex, norm):
    segment_displacement = final_vertex - initial_vertex
    segment_length = segment_displacement.norm()
    center_to_initial_displacement = initial_vertex - position
    left_x = center_to_initial_displacement.dot(segment_displacement) / segment_length
    right_x = left_x + segment_length
    wedge_product = segment_displacement.cross(center_to_initial_displacement)
    y = wedge_product.dot(norm) / segment_length
    return process_segment_standard_geometry(radius, left_x, right_x, y)


@ti.func
def process_segment_standard_geometry(radius, left_x, right_x, y):
    area = 0.0
    if y * y > radius * radius:
        area = process_non_intersecting_region(radius, left_x, right_x, y)
    else:
        intersection_x = ti.sqrt(radius * radius - y * y)
        if left_x < -intersection_x:
            left_region_right_endpoint = ti.min(-intersection_x, right_x)
            area += process_non_intersecting_region(radius, left_x, left_region_right_endpoint, y)
        if intersection_x < right_x:
            right_region_left_endpoint = ti.max(intersection_x, left_x)
            area += process_non_intersecting_region(radius, right_region_left_endpoint, right_x, y)
        middle_region_left_endpoint = ti.max(-intersection_x, left_x)
        middle_region_right_endpoint = ti.min(intersection_x, right_x)
        middle_region_length = ti.max(middle_region_right_endpoint - middle_region_left_endpoint, 0)
        area -= 0.5 * middle_region_length * y
    return area


@ti.func
def process_non_intersecting_region(radius, left_x, right_x, y):
    initial_theta = ti.atan2(y, left_x)
    final_theta = ti.atan2(y, right_x)
    delta_theta = final_theta - initial_theta
    if delta_theta < -PI:
        delta_theta += 2 * PI
    elif delta_theta > PI:
        delta_theta -= 2 * PI
    return 0.5 * radius * radius * delta_theta


@ti.func
def sphere_triangle_intersection_area_and_centroid(position, radius, vertice1, vertice2, vertice3, norm):
    area1, centroid1 = process_segment_and_centroid(position, radius, vertice1, vertice2, norm)
    area2, centroid2 = process_segment_and_centroid(position, radius, vertice2, vertice3, norm)
    area3, centroid3 = process_segment_and_centroid(position, radius, vertice3, vertice1, norm)
    area = area1 + area2 + area3
    weighted_centroid = centroid1 + centroid2 + centroid3
    weighted_centroid = weighted_centroid / area if area > 0.0 else vec3f(0.0, 0.0, 0.0)
    return area, weighted_centroid


@ti.func
def process_segment_and_centroid(position, radius, initial_vertex, final_vertex, norm):
    segment_displacement = final_vertex - initial_vertex
    segment_length = segment_displacement.norm()
    center_to_initial_displacement = initial_vertex - position
    left_x = center_to_initial_displacement.dot(segment_displacement) / segment_length
    right_x = left_x + segment_length
    wedge_product = segment_displacement.cross(center_to_initial_displacement)
    y = wedge_product.dot(norm) / segment_length
    area, centroid = 0.0, vec3f(0.0, 0.0, 0.0)
    if y * y > radius * radius:
        darea, center = process_non_intersecting_centroid(radius, left_x, right_x, y)
        AB = center_to_initial_displacement
        AC = center_to_initial_displacement + segment_displacement
        ABnorm = AB.norm()
        ACnorm = AC.norm()
        centroid += darea * (
            position + (ACnorm / (ABnorm + ACnorm) * AB + ABnorm / (ACnorm + ABnorm) * AC).normalized() * center
        )
        area += darea
    else:
        intersection_x = ti.sqrt(radius * radius - y * y)
        if left_x < -intersection_x:
            left_region_right_endpoint = ti.min(-intersection_x, right_x)
            darea, center = process_non_intersecting_centroid(radius, left_x, left_region_right_endpoint, y)
            AB = center_to_initial_displacement
            AC = center_to_initial_displacement + segment_displacement / segment_length * (
                left_region_right_endpoint - left_x
            )
            ABnorm = AB.norm()
            ACnorm = AC.norm()
            centroid += darea * (
                position + (ACnorm / (ABnorm + ACnorm) * AB + ABnorm / (ACnorm + ABnorm) * AC).normalized() * center
            )
            area += darea
        if intersection_x < right_x:
            right_region_left_endpoint = ti.max(intersection_x, left_x)
            darea, center = process_non_intersecting_centroid(radius, right_region_left_endpoint, right_x, y)
            AB = (
                final_vertex - position - segment_displacement / segment_length * (right_x - right_region_left_endpoint)
            )
            AC = final_vertex - position
            ABnorm = AB.norm()
            ACnorm = AC.norm()
            centroid += darea * (
                position + (ACnorm / (ABnorm + ACnorm) * AB + ABnorm / (ACnorm + ABnorm) * AC).normalized() * center
            )
            area += darea
        middle_region_left_endpoint = ti.max(-intersection_x, left_x)
        middle_region_right_endpoint = ti.min(intersection_x, right_x)
        middle_region_length = ti.max(middle_region_right_endpoint - middle_region_left_endpoint, 0)
        darea = -0.5 * middle_region_length * y
        centroid += (
            darea
            * 1.0
            / 3.0
            * (
                position
                + initial_vertex
                + segment_displacement / segment_length * (middle_region_left_endpoint - left_x)
                + final_vertex
                - segment_displacement / segment_length * (right_x - middle_region_right_endpoint)
            )
        )
        area += darea
    return area, centroid


@ti.func
def process_non_intersecting_centroid(radius, left_x, right_x, y):
    initial_theta = ti.atan2(y, left_x)
    final_theta = ti.atan2(y, right_x)
    delta_theta = final_theta - initial_theta
    if delta_theta < -PI:
        delta_theta += 2 * PI
    elif delta_theta > PI:
        delta_theta -= 2 * PI
    alpha = 0.5 * delta_theta
    darea = 0.5 * radius * radius * delta_theta
    inv_alpha = 1.0 / alpha if alpha > 0.0 else 0.0
    center = 2.0 / 3.0 * radius * ti.sin(alpha) * inv_alpha
    return darea, center


@ti.func
def is_line_intersect_triangle(line_point1, line_point2, _plane_point1, _plane_point2, _plane_point3):
    is_intersection = 0

    u = _plane_point3 - _plane_point1
    v = _plane_point2 - _plane_point1
    n = u * v
    dir = line_point2 - line_point1
    w0 = line_point1 - _plane_point1
    a = -n.dot(w0)
    b = n.dot(dir)

    if ti.abs(b) > 0.0:
        r = a / b
        if r > 0.0:
            I = line_point1 + dir * r

            uu = u.dot(u)
            uv = u.dot(v)
            vv = v.dot(v)
            w = I - _plane_point1
            wu = w.dot(u)
            wv = w.dot(v)
            D = uv * uv - uu * vv

            s = (uv * wv - vv * wu) / D
            if s < 0.0 or s > 1.0:
                is_intersection = 1
            t = (uv * wu - uu * wv) / D
            if t >= 0.0 and (s + t) <= 1.0:
                is_intersection = 1
    return is_intersection


@ti.func
def intersect_ray_sphere(_P, _D, _sphere_center, _d_radius, _p_length, _p_intersection):
    is_intersection = 0
    m = _P - _sphere_center
    b = m.dot(_D)
    dC = m.dot(m) - _d_radius * _d_radius
    if dC <= 0 or b <= 0:
        dDiscr = b * b - dC
        if dDiscr >= 0:
            _p_length = -b
            if _p_length < 0:
                _p_length = 0
            _p_intersection = _P + _D * _p_length
            is_intersection = 1
    return is_intersection, _p_intersection


# refers to Arvo, A Simple Method for Box-Sphere Intersection Testing, Graphics Gems, pp. 247-250, 1993
@ti.func
def is_circle_intersect_rectangle(circle_center, radius, recenter, hlength):
    v = ti.abs(circle_center - recenter)
    u = ti.max(v - hlength, 0.0)
    return u.dot(u) < radius * radius


@ti.func
def point_projection_to_rectangle(circle_center, recenter, norm, length, width, height):
    p = circle_center - recenter
    rotation_matrix = rodrigues_rotation_matrix(vec3f(0, 0, 1), norm)
    localp = rotation_matrix.transpose() @ p

    dist1 = localp.dot(vec3f(1, 0, 0))
    dist2 = localp.dot(vec3f(0, 1, 0))
    dist3 = localp.dot(vec3f(0, 0, 1))
    dist1 = clamp(-length, length, dist1)
    dist2 = clamp(-width, width, dist2)
    dist3 = clamp(-height, height, dist3)
    q = dist1 * vec3f(1, 0, 0) + dist2 * vec3f(0, 1, 0) + dist3 * vec3f(0, 0, 1)
    return rotation_matrix @ q + recenter


@ti.func
def spheres_intersection_volume(center1, center2, rad1, rad2):
    dR = ti.max(rad1, rad2)
    dr = ti.min(rad1, rad2)
    dDistance = (center1 - center2).norm()

    volume = 0.0
    if dDistance + dr < dR:
        volume = 4.0 / 3.0 * PI * dr * dr * dr
    else:
        db = (dr * dr - dR * dR + dDistance * dDistance) / (2.0 * dDistance)
        if db >= 0.0:
            dh1 = dr - db
            dh2 = dR - dDistance + db
            volume = PI / 3.0 * dh1 * dh1 * (3 * dr - dh1) + PI / 3.0 * dh2 * dh2 * (3.0 * dR - dh2)
        else:
            dh1 = dr - ti.abs(db)
            dh2 = dR - dDistance - ti.abs(db)
            volume = (
                4.0 / 3.0 * PI * dr * dr * dr
                - PI / 3.0 * dh1 * dh1 * (3 * dr - dh1)
                + PI / 3.0 * dh2 * dh2 * (3 * dR - dh2)
            )
    return volume


@ti.func
def distance_from_point_to_segment(point, vertice1, vertice2):
    return ((point - vertice1).cross(point - vertice2)).norm() / (vertice2 - vertice1).norm()


@ti.func
def line_line_intersection(p1L1, p2L1, p1L2, p2L2):
    dir_vec_l1 = p2L1 - p1L1
    dir_vec_l2 = p2L2 - p1L2
    denom_x = dir_vec_l1[1] * dir_vec_l2[0] - dir_vec_l2[1] * dir_vec_l1[0]
    denom_y = dir_vec_l1[0] * dir_vec_l2[1] - dir_vec_l2[0] * dir_vec_l1[1]
    denom_z = dir_vec_l1[1] * dir_vec_l2[2] - dir_vec_l2[1] * dir_vec_l1[2]
    x = (
        p1L1[0] * dir_vec_l1[1] * dir_vec_l2[0]
        - p1L2[0] * dir_vec_l2[1] * dir_vec_l1[0]
        - p1L1[1] * dir_vec_l1[0] * dir_vec_l2[0]
        + p1L2[1] * dir_vec_l2[0] * dir_vec_l1[0]
    )
    y = (
        p1L1[1] * dir_vec_l1[0] * dir_vec_l2[1]
        - p1L2[1] * dir_vec_l2[0] * dir_vec_l1[1]
        - p1L1[0] * dir_vec_l1[1] * dir_vec_l2[1]
        + p1L2[0] * dir_vec_l2[1] * dir_vec_l1[1]
    )
    z = (
        p1L1[2] * dir_vec_l1[1] * dir_vec_l2[2]
        - p1L2[2] * dir_vec_l2[1] * dir_vec_l1[2]
        - p1L1[1] * dir_vec_l1[2] * dir_vec_l2[2]
        + p1L2[1] * dir_vec_l2[2] * dir_vec_l1[2]
    )

    if denom_x == 0:
        if x != 0:
            x = 0 / 0
    else:
        x /= denom_x
    if denom_y == 0:
        if y != 0:
            y = 0 / 0
    else:
        y /= denom_y
    if denom_z == 0:
        if z != 0:
            z = 0 / 0
    else:
        z /= denom_z
    return vec3f(x, y, z)


# Refers to https://iquilezles.org/articles/distfunctions/
@ti.func
def distance_from_point_to_triangle(point, vertice1, vertice2, vertice3, normal):
    ba = vertice2 - vertice1
    cb = vertice3 - vertice2
    ac = vertice1 - vertice3
    pa = point - vertice1
    pb = point - vertice2
    pc = point - vertice3
    norm = ba.cross(ac)

    A = sgn((ba.cross(norm).dot(pa))) + sgn((cb.cross(norm).dot(pb))) + sgn((ac.cross(norm).dot(pc)))
    dist = (
        ti.sqrt(
            ti.min(
                dot2(ba * clamp(0.0, 1.0, ba.dot(pa) / dot2(ba)) - pa),
                dot2(cb * clamp(0.0, 1.0, cb.dot(pb) / dot2(cb)) - pb),
                dot2(ac * clamp(0.0, 1.0, ac.dot(pc) / dot2(ac)) - pc),
            )
        )
        if A < 2.0
        else ti.sqrt(norm.dot(pa) * norm.dot(pa) / dot2(norm))
    )
    return -sgn(pa.dot(normal)) * dist


@ti.func
def distance_from_point_to_rectangle(point, vertice1, vertice2, vertice3, vertice4, norm):
    ba = vertice2 - vertice1
    cb = vertice3 - vertice2
    dc = vertice4 - vertice3
    ad = vertice1 - vertice4
    pa = point - vertice1
    pb = point - vertice2
    pc = point - vertice3
    pd = point - vertice4
    # norm = ba.cross(ad)

    A = (
        sgn((ba.cross(norm).dot(pa)))
        + sgn((cb.cross(norm).dot(pb)))
        + sgn((dc.cross(norm).dot(pc)))
        + sgn((ad.cross(norm).dot(pd)))
    )
    dist = (
        ti.sqrt(
            ti.min(
                dot2(ba * clamp(0.0, 1.0, ba.dot(pa) / dot2(ba)) - pa),
                dot2(cb * clamp(0.0, 1.0, cb.dot(pb) / dot2(cb)) - pb),
                dot2(dc * clamp(0.0, 1.0, dc.dot(pc) / dot2(dc)) - pc),
                dot2(ad * clamp(0.0, 1.0, ad.dot(pd) / dot2(ad)) - pd),
            )
        )
        if A < 3.0
        else ti.sqrt(norm.dot(pa) * norm.dot(pa) / dot2(norm))
    )
    return sgn(pa.dot(-norm)) * dist


# Refers to Philip J. Schneider, and David H. Eberly. Geometric Tools for Computer Graphics. Morgan Kaufmann, 2002
@ti.func
def distance_from_point_to_triangle2(point, vertice1, vertice2, vertice3):
    v_edge0 = vertice2 - vertice1
    v_edge1 = vertice3 - vertice1
    vD = vertice1 - point

    a = v_edge0.dot(v_edge0)
    b = v_edge0.dot(v_edge1)
    c = v_edge1.dot(v_edge1)
    d = v_edge0.dot(vD)
    e = v_edge1.dot(vD)
    f = dot2(vD)

    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    if s + t < det:
        if s < 0.0:
            if t < 0.0:
                if d < 0.0:
                    s = clamp(0.0, 1.0, -d / a)
                    t = 0.0
                else:
                    s = 0.0
                    t = clamp(0.0, 1.0, -e / c)
            else:
                s = 0.0
                t = clamp(0.0, 1.0, -e / c)
        elif t < 0.0:
            s = clamp(0.0, 1.0, -d / a)
            t = 0.0
        else:
            inv_det = 1.0 / det
            s *= inv_det
            t *= inv_det
    else:
        if s < 0.0:
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                s = clamp(0.0, 1.0, numer / denom)
                t = 1 - s
            else:
                t = clamp(0.0, 1.0, -e / c)
                s = 0
        elif t < 0.0:
            if a + d > b + e:
                numer = c + e - b - d
                denom = a - 2.0 * b + c
                s = clamp(0.0, 1.0, numer / denom)
                t = 1 - s
            else:
                s = clamp(0.0, 1.0, -e / c)
                t = 0
        else:
            numer = c + e - b - d
            denom = a - 2.0 * b + c
            s = clamp(0.0, 1.0, numer / denom)
            t = 1.0 - s
    return ti.sqrt(a * s * s + 2 * b * s * t + c * t * t + 2 * d * s + 2 * e * t + f)


@ti.func
def point_projection_distance(position, center, norm):
    return (position - center).dot(norm)


@ti.func
def projection_point_in_plane(position, norm, distance):
    return position - norm * distance


@ti.func
def is_triangle_anti_clockwise(vertice1, vertice2, vertice3, norm):
    return (vertice2 - vertice1).cross(vertice3 - vertice1).dot(norm) > 0.0


@ti.func
def is_point_in_triangle(projection_point, vertice1, vertice2, vertice3):
    edge21 = vertice2 - vertice1
    edge31 = vertice3 - vertice1
    W = projection_point - vertice1

    d00 = edge21.dot(edge21)
    d01 = edge21.dot(edge31)
    d11 = edge31.dot(edge31)
    d20 = W.dot(edge21)
    d21 = W.dot(edge31)
    inv_denom = 1.0 / (d00 * d11 - d01 * d01)
    gamma = (d11 * d20 - d01 * d21) * inv_denom
    betta = (d00 * d21 - d01 * d20) * inv_denom
    alpha = 1.0 - gamma - betta
    return (gamma > 0 and gamma < 1) and (alpha > 0 and alpha < 1) and (betta > 0 and betta < 1)


# GJK contact algorithm
@ti.func
def is_point_in_triangle2(projection_point, vertice1, vertice2, vertice3):
    u = (vertice1 - projection_point).cross(vertice2 - projection_point)
    v = (vertice2 - projection_point).cross(vertice3 - projection_point)
    w = (vertice3 - projection_point).cross(vertice1 - projection_point)
    return u.dot(v) >= 0.0 and u.dot(w) >= 0.0


@ti.func
def is_point_in_tetra(point, vertice1, vertice2, vertice3, vertice4):
    pass


@ti.func
def ray_triangle_interaction(origin, direction, vertice1, vertice2, vertice3):
    t, u, v = 0.0, 0.0, 0.0
    # Reference: T. Möller, B. Trumbore (1997) Fast, minimum storage ray-triangle intersection. J. Graph. Tools, 2, 21-28.
    edge1 = vertice2 - vertice1
    edge2 = vertice3 - vertice1
    pvec = direction.cross(edge2)

    det = edge1.dot(pvec)
    if det > THRESHOLD:
        tvec = origin - vertice1
        u = tvec.dot(pvec)
        if 0.0 <= u <= det:
            qvec = tvec.cross(edge1)
            v = direction.dot(qvec)
            if 0.0 <= v <= det:
                t = edge2.dot(qvec)
                inv_det = 1.0 / det
                t *= inv_det
                u *= inv_det
                v *= inv_det
            else:
                (
                    u,
                    v,
                ) = (
                    0.0,
                    0.0,
                )
        else:
            u = 0.0
    return t, u, v


@ti.func
def is_sphere_intersect_triangle(bound_beg, bound_end, vertice1, vertice2, vertice3, norm, position, radius):
    status = 0
    point = vec3f(0.0, 0.0, 0.0)

    if (
        position[0] <= bound_beg[0] - radius
        and position[1] <= bound_beg[1] - radius
        and position[2] <= bound_beg[2] - radius
        and position[0] >= bound_end[0] + radius
        and position[1] >= bound_end[1] + radius
        and position[2] >= bound_end[2] + radius
    ):
        center = (vertice1 + vertice2 + vertice3) / 3.0
        ppd = point_projection_distance(position, center, norm)
        if ti.abs(ppd) < radius:
            A = projection_point_in_plane(position, norm, ppd)
            if is_point_in_triangle(A, vertice1, vertice2, vertice3):
                status = 1
                point = A
            else:
                edge21 = vertice2 - vertice1
                edge32 = vertice3 - vertice2
                edge13 = vertice1 - vertice3
                lc1 = clamp(0.0, 1.0, (A - vertice1).dot(edge21) / dot2(edge21))
                lc2 = clamp(0.0, 1.0, (A - vertice2).dot(edge32) / dot2(edge32))
                lc3 = clamp(0.0, 1.0, (A - vertice3).dot(edge13) / dot2(edge13))
                c1_is_vertice = lc1 == 0 or lc1 == 1
                c2_is_vertice = lc2 == 0 or lc2 == 1
                c3_is_vertice = lc3 == 0 or lc3 == 1
                C1 = vertice1 + lc1 * edge21
                C2 = vertice2 + lc2 * edge32
                C3 = vertice3 + lc3 * edge13
                sqr_length1 = dot2(C1 - position)
                sqr_length2 = dot2(C2 - position)
                sqr_length3 = dot2(C3 - position)
                if ti.min({sqr_length1, sqr_length2, sqr_length3}) < radius * radius:
                    if sqr_length1 <= sqr_length2 and sqr_length1 <= sqr_length3:
                        status = 3 if c1_is_vertice else 2
                        point = C1
                    elif sqr_length2 <= sqr_length3:
                        status = 3 if c2_is_vertice else 2
                        point = C2
                    else:
                        status = 3 if c3_is_vertice else 2
                        point = C3
    return status, point


@ti.func
def spheres_contact_point(center1, center2, radius1, radius2):
    return center1 + (center2 - center1) * radius1 / (radius1 + radius2)


@ti.func
def get_bounding_box(vertice1, vertice2, vertice3):
    coord_beg = vec3f(0, 0, 0)
    coord_end = vec3f(0, 0, 0)

    coord_beg = ti.min(coord_beg, vertice1)
    coord_beg = ti.min(coord_beg, vertice2)
    coord_beg = ti.min(coord_beg, vertice3)

    coord_end = ti.max(coord_end, vertice1)
    coord_end = ti.max(coord_end, vertice2)
    coord_end = ti.max(coord_end, vertice3)
    return coord_beg, coord_end


@ti.func
def intersection_ob_bs(center1, center2, extent1, extent2, rotate_matrix1, rotate_matrix2):
    cmatrix = rotate_matrix1.transpose() @ rotate_matrix2
    amatrix = ti.abs(cmatrix)
    dvector = center2 - center1

    a0, a1, a2 = 0.5 * extent1
    b0, b1, b2 = 0.5 * extent2
    A0 = vec3f(rotate_matrix1[0, 0], rotate_matrix1[1, 0], rotate_matrix1[2, 0])
    A1 = vec3f(rotate_matrix1[0, 1], rotate_matrix1[1, 1], rotate_matrix1[2, 1])
    A2 = vec3f(rotate_matrix1[0, 2], rotate_matrix1[1, 2], rotate_matrix1[2, 2])
    B0 = vec3f(rotate_matrix2[0, 0], rotate_matrix2[1, 0], rotate_matrix2[2, 0])
    B1 = vec3f(rotate_matrix2[0, 1], rotate_matrix2[1, 1], rotate_matrix2[2, 1])
    B2 = vec3f(rotate_matrix2[0, 2], rotate_matrix2[1, 2], rotate_matrix2[2, 2])

    separated = False
    if not separated and ti.abs(A0.dot(dvector)) > a0 + (b0 * amatrix[0, 0] + b1 * amatrix[0, 1] + b2 * amatrix[0, 2]):
        separated = True
    if not separated and ti.abs(A1.dot(dvector)) > a1 + (b0 * amatrix[1, 0] + b1 * amatrix[1, 1] + b2 * amatrix[1, 2]):
        separated = True
    if not separated and ti.abs(A2.dot(dvector)) > a2 + (b0 * amatrix[2, 0] + b1 * amatrix[2, 1] + b2 * amatrix[2, 2]):
        separated = True

    if not separated and ti.abs(B0.dot(dvector)) > b0 + (a0 * amatrix[0, 0] + a1 * amatrix[1, 0] + a2 * amatrix[2, 0]):
        separated = True
    if not separated and ti.abs(B1.dot(dvector)) > b1 + (a0 * amatrix[0, 1] + a1 * amatrix[1, 1] + a2 * amatrix[2, 1]):
        separated = True
    if not separated and ti.abs(B2.dot(dvector)) > b2 + (a0 * amatrix[0, 2] + a1 * amatrix[1, 2] + a2 * amatrix[2, 2]):
        separated = True

    if not separated and ti.abs(cmatrix[1, 0] * A2.dot(dvector) - cmatrix[2, 0] * A1.dot(dvector)) > (
        a1 * amatrix[2, 0] + a2 * amatrix[1, 0]
    ) + (b1 * amatrix[0, 2] + b2 * amatrix[0, 1]):
        separated = True
    if not separated and ti.abs(cmatrix[1, 1] * A2.dot(dvector) - cmatrix[2, 1] * A1.dot(dvector)) > (
        a1 * amatrix[2, 1] + a2 * amatrix[1, 1]
    ) + (b0 * amatrix[0, 2] + b2 * amatrix[0, 0]):
        separated = True
    if not separated and ti.abs(cmatrix[1, 2] * A2.dot(dvector) - cmatrix[2, 2] * A1.dot(dvector)) > (
        a1 * amatrix[2, 2] + a2 * amatrix[1, 2]
    ) + (b0 * amatrix[0, 1] + b1 * amatrix[0, 0]):
        separated = True

    if not separated and ti.abs(cmatrix[2, 0] * A0.dot(dvector) - cmatrix[0, 0] * A2.dot(dvector)) > (
        a0 * amatrix[2, 0] + a2 * amatrix[0, 0]
    ) + (b1 * amatrix[1, 2] + b2 * amatrix[1, 1]):
        separated = True
    if not separated and ti.abs(cmatrix[2, 1] * A0.dot(dvector) - cmatrix[0, 1] * A2.dot(dvector)) > (
        a0 * amatrix[2, 1] + a2 * amatrix[0, 1]
    ) + (b0 * amatrix[1, 2] + b2 * amatrix[1, 0]):
        separated = True
    if not separated and ti.abs(cmatrix[2, 2] * A0.dot(dvector) - cmatrix[0, 2] * A2.dot(dvector)) > (
        a0 * amatrix[2, 2] + a2 * amatrix[0, 2]
    ) + (b0 * amatrix[1, 1] + b1 * amatrix[1, 0]):
        separated = True

    if not separated and ti.abs(cmatrix[0, 0] * A1.dot(dvector) - cmatrix[1, 0] * A0.dot(dvector)) > (
        a0 * amatrix[1, 0] + a1 * amatrix[0, 0]
    ) + (b1 * amatrix[2, 2] + b2 * amatrix[2, 1]):
        separated = True
    if not separated and ti.abs(cmatrix[0, 1] * A1.dot(dvector) - cmatrix[1, 1] * A0.dot(dvector)) > (
        a0 * amatrix[1, 1] + a1 * amatrix[0, 1]
    ) + (b0 * amatrix[2, 2] + b2 * amatrix[2, 0]):
        separated = True
    if not separated and ti.abs(cmatrix[0, 2] * A1.dot(dvector) - cmatrix[1, 2] * A0.dot(dvector)) > (
        a0 * amatrix[1, 2] + a1 * amatrix[0, 2]
    ) + (b0 * amatrix[2, 1] + b1 * amatrix[2, 0]):
        separated = True
    return not separated


@ti.func
def intersection_triangle_ob_bs(mass_center, extent, rotate_matrix, vertice1, vertice2, vertice3, norm):
    wall_center = 1.0 / 3.0 * (vertice1 + vertice2 + vertice3)
    vertice1 = rotate_matrix @ (vertice1 - wall_center) + wall_center
    vertice2 = rotate_matrix @ (vertice2 - wall_center) + wall_center
    vertice3 = rotate_matrix @ (vertice3 - wall_center) + wall_center
    edge1, edge2, edge3 = vertice2 - vertice1, vertice3 - vertice1, vertice3 - vertice2
    norm = rotate_matrix @ norm
    dvector = vertice1 - mass_center

    a0, a1, a2 = 0.5 * extent
    A0, A1, A2 = rotate_matrix @ vec3f(1, 0, 0), rotate_matrix @ vec3f(0, 1, 0), rotate_matrix @ vec3f(0, 0, 1)
    not_intersect = True
    if not_intersect:
        R = a0 * ti.abs(norm.dot(A0)) + a1 * ti.abs(norm.dot(A1)) + a2 * ti.abs(norm.dot(A2))
        if -R < norm.dot(dvector) < R:
            not_intersect = False
    if not_intersect:
        p0 = A0.dot(dvector)
        if -a0 < ti.max(p0, p0 + A0.dot(edge1), p0 + A0.dot(edge2)) < a0:
            not_intersect = False
    if not_intersect:
        p0 = A1.dot(dvector)
        if -a1 < ti.max(p0, p0 + A1.dot(edge1), p0 + A1.dot(edge2)) < a1:
            not_intersect = False
    if not_intersect:
        p0 = A2.dot(dvector)
        if -a2 < ti.max(p0, p0 + A2.dot(edge1), p0 + A2.dot(edge2)) < a2:
            not_intersect = False
    if not_intersect:
        p0 = A0.cross(edge1).dot(dvector)
        R = a1 * ti.abs(A2.dot(edge1)) + a2 * ti.abs(A1.dot(edge1))
        if -R < ti.max(p0, p0 + A0.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A0.cross(edge2).dot(dvector)
        R = a1 * ti.abs(A2.dot(edge2)) + a2 * ti.abs(A1.dot(edge2))
        if -R < ti.max(p0, p0 - A0.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A0.cross(edge3).dot(dvector)
        R = a1 * ti.abs(A2.dot(edge3)) + a2 * ti.abs(A1.dot(edge3))
        if -R < ti.max(p0, p0 + A0.dot(norm), p0 - A0.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A1.cross(edge1).dot(dvector)
        R = a0 * ti.abs(A2.dot(edge1)) + a2 * ti.abs(A0.dot(edge1))
        if -R < ti.max(p0, p0 + A1.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A1.cross(edge2).dot(dvector)
        R = a0 * ti.abs(A2.dot(edge2)) + a2 * ti.abs(A0.dot(edge2))
        if -R < ti.max(p0, p0 - A1.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A1.cross(edge3).dot(dvector)
        R = a0 * ti.abs(A2.dot(edge3)) + a2 * ti.abs(A0.dot(edge3))
        if -R < ti.max(p0, p0 + A1.dot(norm), p0 - A1.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A2.cross(edge1).dot(dvector)
        R = a0 * ti.abs(A1.dot(edge1)) + a1 * ti.abs(A0.dot(edge1))
        if -R < ti.max(p0, p0 + A2.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A2.cross(edge2).dot(dvector)
        R = a0 * ti.abs(A1.dot(edge2)) + a1 * ti.abs(A0.dot(edge2))
        if -R < ti.max(p0, p0 - A2.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A2.cross(edge3).dot(dvector)
        R = a0 * ti.abs(A1.dot(edge3)) + a1 * ti.abs(A0.dot(edge3))
        if -R < ti.max(p0, p0 + A2.dot(norm), p0 - A2.dot(norm)) < R:
            not_intersect = False
    return not not_intersect


@ti.pyfunc
def dist3d_segment_to_segment(A0, A1, B0, B1):
    u = A1 - A0
    v = B1 - B0
    w = A0 - B0
    a = u.norm_sqr()
    b = u.dot(v)
    c = v.norm_sqr()
    d = u.dot(w)
    e = v.dot(w)
    D = a * c - b * b
    sc, sN, sD = D, D, D
    tc, tN, tD = D, D, D
    if D < 1e-7:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if -d + b < 0.0:
            sN = 0.0
        elif -d + b > a:
            sN = sD
        else:
            sN = -d + b
            sD = a

    if ti.abs(sN) < 1e-7:
        sc = 0.0
    else:
        sc = sN / sD
    if ti.abs(tN) < 1e-7:
        tc = 0.0
    else:
        tc = tN / tD
    dP = -w - (sc * u) + (tc * v)  # Qc - Pc
    return dP, sc, tc


@ti.pyfunc
def dist3d_point_triangle(P, V0, V1, V2):
    cord0 = 0.0
    cord1 = 0.0
    cord2 = 0.0
    v = V2 - V0
    u = V1 - V0
    n_vec = u.cross(v)
    s_p = (n_vec.dot(P - V0)) / (n_vec.dot(n_vec))
    P0 = P - s_p * n_vec  # P project to plane
    w = P0 - V0
    n_cross_v = n_vec.cross(v)
    n_cross_u = n_vec.cross(u)
    s = w.dot(n_cross_v) / (u.dot(n_cross_v))
    t = w.dot(n_cross_u) / (v.dot(n_cross_u))
    if s >= 0.0 and t >= 0.0:
        if s + t <= 1.0:
            cord0 = 1.0 - s - t
            cord1 = s
            cord2 = t
        else:
            q = V2 - V1
            k = (P - V1).dot(q) / (q.dot(q))
            if k > 1.0:
                cord2 = 1.0
            elif k < 0.0:
                cord1 = 1.0
            else:
                cord1 = 1.0 - k
                cord2 = k
    elif s >= 0.0 and t < 0.0:
        k = w.dot(u) / (u.dot(u))
        if k > 1.0:
            cord1 = 1.0
        elif k < 0.0:
            cord0 = 1.0
        else:
            cord0 = 1.0 - k
            cord1 = k
    elif s < 0.0 and t >= 0.0:
        k = w.dot(v) / (v.dot(v))
        if k > 1.0:
            cord2 = 1.0
        elif k < 0.0:
            cord0 = 1.0
        else:
            cord0 = 1.0 - k
            cord2 = k
    else:  # s < 0 and t < 0
        cord0 = 1.0
    return cord0, cord1, cord2


@ti.pyfunc
def point_to_segment_distance2d(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1

    distance = 0.0
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        distance = ti.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    else:
        c2 = vx * vx + vy * vy
        if c2 <= c1:
            distance = ti.sqrt((px - x2) ** 2 + (py - y2) ** 2)
        else:
            b = c1 / c2
            bx, by = x1 + b * vx, y1 + b * vy
            distance = ti.sqrt((px - bx) ** 2 + (py - by) ** 2)
    return distance


@ti.pyfunc
def point_in_polygon(x, y, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i, 0], polygon[i, 1]
        x2, y2 = polygon[(i + 1) % polygon.n, 0], polygon[(i + 1) % polygon.n, 1]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


@ti.pyfunc
def circle_inside_geometry(polygon, cx, cy, r):
    inside = True
    if not point_in_polygon(cx, cy, polygon):
        inside = False
    else:
        for i in range(len(polygon)):
            x1, y1 = polygon[i, 0], polygon[i, 1]
            x2, y2 = polygon[(i + 1) % polygon.n, 0], polygon[(i + 1) % polygon.n, 1]
            dist = point_to_segment_distance2d(cx, cy, x1, y1, x2, y2)
            if dist < r:
                inside = False
                break
    return inside


def distance_along_direction_to_surface(
    mesh: trimesh.Trimesh, points: np.ndarray, direction: np.ndarray, max_distance=np.inf
):
    """
    points: (N,3)
    direction: (3,) 所有点共用同一射线方向
    返回:
      distances: (N,) 数组: 每个点的最近交点距离; 若无交点 -> np.nan
    """
    points = np.asarray(points, dtype=float)
    direction = np.asarray(direction, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 必须是 (N,3)")

    # 归一化方向
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("direction 不能为零向量")
    direction = direction / norm

    N = len(points)
    origins = points
    directions = np.repeat(direction.reshape(1, 3), N, axis=0)

    # 求交
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )

    # 先全设为 nan
    distances = np.full(N, np.nan, dtype=float)

    if len(locations) == 0:
        return distances  # 所有点都没撞到

    # 对每条射线，可能有多个交点，需要取最近的
    # index_ray: 对应 origins 的下标
    # locations: 对应每个交点的坐标
    deltas = locations - origins[index_ray]
    dists = np.linalg.norm(deltas, axis=1)

    # 把每个 ray 的最小距离写到结果里
    for i, ray_idx in enumerate(index_ray):
        di = dists[i]
        if di > max_distance:
            continue
        if np.isnan(distances[ray_idx]) or di < distances[ray_idx]:
            distances[ray_idx] = di

    return distances
