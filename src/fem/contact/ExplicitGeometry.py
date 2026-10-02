"""Shared closest-feature geometry for explicit FEM proximity and contact."""

import taichi as ti

from src.utils.constants import Threshold


@ti.func
def closest_point_triangle(point, first, second, third):
    ab = second - first
    ac = third - first
    ap = point - first
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    barycentric = ti.Vector([1.0, 0.0, 0.0])
    if d1 <= 0.0 and d2 <= 0.0:
        pass
    else:
        bp = point - second
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)
        if d3 >= 0.0 and d4 <= d3:
            barycentric = ti.Vector([0.0, 1.0, 0.0])
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                ratio = d1 / ti.max(d1 - d3, Threshold)
                barycentric = ti.Vector([1.0 - ratio, ratio, 0.0])
            else:
                cp = point - third
                d5 = ab.dot(cp)
                d6 = ac.dot(cp)
                if d6 >= 0.0 and d5 <= d6:
                    barycentric = ti.Vector([0.0, 0.0, 1.0])
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        ratio = d2 / ti.max(d2 - d6, Threshold)
                        barycentric = ti.Vector([1.0 - ratio, 0.0, ratio])
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
                            ratio = (d4 - d3) / ti.max(d4 - d3 + d5 - d6, Threshold)
                            barycentric = ti.Vector([0.0, 1.0 - ratio, ratio])
                        else:
                            inverse = 1.0 / ti.max(va + vb + vc, Threshold)
                            barycentric[1] = vb * inverse
                            barycentric[2] = vc * inverse
                            barycentric[0] = 1.0 - barycentric[1] - barycentric[2]
    closest = barycentric[0] * first + barycentric[1] * second + barycentric[2] * third
    return closest, barycentric


@ti.func
def signed_closest_feature_geometry(
    offset,
    barycentric,
    stencil,
    face_normal,
    pseudonormals,
):
    pseudo = (
        barycentric[0] * pseudonormals[stencil[1]]
        + barycentric[1] * pseudonormals[stencil[2]]
        + barycentric[2] * pseudonormals[stencil[3]]
    )
    if pseudo.norm_sqr() <= Threshold * Threshold:
        pseudo = face_normal
    pseudo = pseudo.normalized(Threshold)
    signed_projection = offset.dot(pseudo)
    side = ti.select(signed_projection >= 0.0, 1.0, -1.0)
    distance = offset.norm()
    normal = pseudo
    signed_gap = signed_projection
    if distance > Threshold:
        normal = side * offset / distance
        signed_gap = side * distance
    return normal, signed_gap


@ti.func
def signed_point_triangle_gap(
    point,
    first,
    second,
    third,
    stencil,
    pseudonormals,
):
    """Return the oriented closest-feature gap used by explicit PT search."""

    signed_gap = 1.0e30
    closest, barycentric = closest_point_triangle(point, first, second, third)
    face_cross = (second - first).cross(third - first)
    face_length = face_cross.norm()
    if face_length > Threshold:
        _, signed_gap = signed_closest_feature_geometry(
            point - closest,
            barycentric,
            stencil,
            face_cross / face_length,
            pseudonormals,
        )
    return signed_gap


__all__ = [
    "closest_point_triangle",
    "signed_closest_feature_geometry",
    "signed_point_triangle_gap",
]
