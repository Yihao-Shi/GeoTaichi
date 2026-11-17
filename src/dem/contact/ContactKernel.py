import taichi as ti

from src.utils.constants import ZEROVEC3F, THRESHOLD, PI
from src.utils.Quaternion import set_to_rotate
from src.utils.ScalarFunction import pairing_mapping, effective_value
from src.utils.TypeDefinition import vec3f
from src.utils.VectorFunction import coord_local2global, global2local, local2global, squared
from src.utils import GlobalVariable


@ti.func
def lsmpm_surface_quadrature_coefficient(body_id, local_node, rigid, box, vertice):
    reference_area = box[body_id].reference_surface_area * box[body_id].scale * box[body_id].scale
    if reference_area <= 0.0:
        reference_area = 4.0 * PI * rigid[body_id].equi_r * rigid[body_id].equi_r
    return ti.max(reference_area * vertice[local_node].parameter, 0.0)


@ti.kernel
def kernel_get_min_ratio(material_id: int, rigid_num: int, material: ti.template(), rigid: ti.template()) -> float:
    ratio = 1.0
    for i in range(rigid_num):
        if rigid[i].materialID == material_id:
            temp = material[material_id].ncut / rigid[i].equi_r
            ti.atomic_min(ratio, temp)
            assert 0.0 < temp <= 1.0
    return ratio


@ti.kernel
def kernel_find_max_stiffness(
    max_material_num: int,
    surface_num: int,
    rigid: ti.template(),
    surface: ti.template(),
    vertice: ti.template(),
    surface_props: ti.template(),
) -> float:
    max_stiff = 0.0
    for i in range(surface_num):
        end1 = surface[i]
        local_node = rigid[end1].global_node_to_local(i)
        material_id1 = int(rigid[end1].materialID)
        parameter = vertice[local_node].parameter
        for material_id2 in range(max_material_num):
            property_id = pairing_mapping(material_id1, material_id2, max_material_num)
            if ti.static(GlobalVariable.ADAPTIVESTIFF):
                if surface_props[property_id].emod > 0.0 and surface_props[property_id].kratio > 0.0:
                    radius = rigid[end1].equi_r
                    kn = PI * 0.5 * radius * surface_props[property_id].emod
                    ti.atomic_max(max_stiff, parameter * max(kn, kn / surface_props[property_id].kratio))
            else:
                ti.atomic_max(max_stiff, parameter * max(surface_props[property_id].kn, surface_props[property_id].ks))
    return max_stiff


@ti.kernel
def kernel_find_max_stiffness_lsmpm(
    max_material_num: int,
    surface_num: int,
    rigid: ti.template(),
    surface: ti.template(),
    vertice: ti.template(),
    box: ti.template(),
    surface_props: ti.template(),
) -> float:
    max_stiff = 0.0
    for i in range(surface_num):
        end1 = surface[i]
        local_node = rigid[end1].global_node_to_local(i)
        material_id1 = int(rigid[end1].materialID)
        coefficient = lsmpm_surface_quadrature_coefficient(end1, local_node, rigid, box, vertice)
        for material_id2 in range(max_material_num):
            property_id = pairing_mapping(material_id1, material_id2, max_material_num)
            if ti.static(GlobalVariable.ADAPTIVESTIFF):
                if surface_props[property_id].emod > 0.0 and surface_props[property_id].kratio > 0.0:
                    parameter = rigid[end1].equi_r
                    kn = PI * parameter * surface_props[property_id].emod * coefficient
                    ti.atomic_max(max_stiff, max(kn, kn / surface_props[property_id].kratio))
            else:
                ti.atomic_max(
                    max_stiff, coefficient * max(surface_props[property_id].kn, surface_props[property_id].ks)
                )
    return max_stiff


@ti.kernel
def kernel_find_max_penalty_stiffness_lsmpm(
    max_material_num: int,
    surface_num: int,
    penetration_bound: float,
    gradient_bound: float,
    rigid: ti.template(),
    surface: ti.template(),
    vertice: ti.template(),
    box: ti.template(),
    surface_props: ti.template(),
) -> float:
    max_stiff = 0.0
    for i in range(surface_num):
        end1 = surface[i]
        local_node = rigid[end1].global_node_to_local(i)
        material_id1 = int(rigid[end1].materialID)
        coefficient = lsmpm_surface_quadrature_coefficient(end1, local_node, rigid, box, vertice) * gradient_bound
        for material_id2 in range(max_material_num):
            property_id = pairing_mapping(material_id1, material_id2, max_material_num)
            theta = surface_props[property_id].theta
            normal_stiffness = surface_props[property_id].kn * coefficient
            if theta > 2.0:
                normal_stiffness *= (theta - 1.0) * ti.pow(ti.max(penetration_bound, THRESHOLD), theta - 2.0)
            tangential_stiffness = surface_props[property_id].ks * coefficient
            ti.atomic_max(
                max_stiff,
                ti.max(normal_stiffness, tangential_stiffness),
            )
    return max_stiff


@ti.kernel
def kernel_find_max_hertz_stiffness_lsmpm(
    max_material_num: int,
    surface_num: int,
    penetration_bound: float,
    rigid: ti.template(),
    surface: ti.template(),
    vertice: ti.template(),
    box: ti.template(),
    surface_props: ti.template(),
) -> float:
    max_stiff = 0.0
    for i in range(surface_num):
        end1 = surface[i]
        local_node = rigid[end1].global_node_to_local(i)
        material_id1 = int(rigid[end1].materialID)
        coefficient = lsmpm_surface_quadrature_coefficient(end1, local_node, rigid, box, vertice)
        contact_scale = ti.sqrt(ti.max(rigid[end1].equi_r * penetration_bound, THRESHOLD))
        for material_id2 in range(max_material_num):
            property_id = pairing_mapping(material_id1, material_id2, max_material_num)
            normal_stiffness = 2.0 * surface_props[property_id].YoungModulus * contact_scale * coefficient
            tangential_stiffness = 8.0 * surface_props[property_id].ShearModulus * contact_scale * coefficient
            ti.atomic_max(
                max_stiff,
                ti.max(normal_stiffness, tangential_stiffness),
            )
    return max_stiff


@ti.kernel
def kernel_find_max_barrier_stiffness_lsmpm(
    max_material_num: int,
    surface_num: int,
    gradient_bound: float,
    rigid: ti.template(),
    surface: ti.template(),
    vertice: ti.template(),
    box: ti.template(),
    surface_props: ti.template(),
) -> float:
    max_stiff = 0.0
    for i in range(surface_num):
        end1 = surface[i]
        local_node = rigid[end1].global_node_to_local(i)
        material_id1 = int(rigid[end1].materialID)
        coefficient = lsmpm_surface_quadrature_coefficient(end1, local_node, rigid, box, vertice) * gradient_bound
        radius = ti.max(rigid[end1].equi_r, THRESHOLD)
        for material_id2 in range(max_material_num):
            property_id = pairing_mapping(material_id1, material_id2, max_material_num)
            if surface_props[property_id].kappa > 0.0:
                ratio = ti.min(
                    1.0 - THRESHOLD,
                    ti.max(surface_props[property_id].ncut / radius, THRESHOLD),
                )
                stiffness = (
                    -surface_props[property_id].kappa
                    * coefficient
                    * (2.0 * ti.log(ratio) + ((ratio - 1.0) * (3.0 * ratio + 1.0)) / (ratio * ratio))
                )
                ti.atomic_max(max_stiff, stiffness)
    return max_stiff


@ti.func
def find_history(end1, end2, hist_cplist, hist_object_object):
    tang_overlap_old = ZEROVEC3F
    for offset in range(hist_object_object[end1], hist_object_object[end1 + 1]):
        if end2 == hist_cplist[offset].DstID:
            tang_overlap_old = hist_cplist[offset].oldTangOverlap
            break
    return tang_overlap_old


@ti.func
def find_history_with_normal_overlap(end1, end2, hist_cplist, hist_object_object):
    tang_overlap = ZEROVEC3F
    normal_overlap = 0.0
    normal_active = ti.u8(0)
    for offset in range(hist_object_object[end1], hist_object_object[end1 + 1]):
        if end2 == hist_cplist[offset].DstID:
            tang_overlap = hist_cplist[offset].oldTangOverlap
            normal_overlap = hist_cplist[offset].normalOverlap
            normal_active = hist_cplist[offset].normalOverlapActive
            break
    return tang_overlap, normal_overlap, normal_active


@ti.func
def find_i_shistory(end1, end2, hist_cplist, hist_object_object):
    tang_overlap_old = ZEROVEC3F
    contact_sa = ZEROVEC3F
    for offset in range(hist_object_object[end1], hist_object_object[end1 + 1]):
        if end2 == hist_cplist[offset].DstID:
            tang_overlap_old = hist_cplist[offset].oldTangOverlap
            contact_sa = hist_cplist[offset].contactSA
            break
    return tang_overlap_old, contact_sa


@ti.func
def find_addition_history(end1, end2, hist_cplist, hist_object_object):
    tang_overlap_old, roll_angle_old, twist_angle_old = ZEROVEC3F, ZEROVEC3F, ZEROVEC3F
    for offset in range(hist_object_object[end1], hist_object_object[end1 + 1]):
        if end2 == hist_cplist[offset].DstID:
            tang_overlap_old = hist_cplist[offset].oldTangOverlap
            roll_angle_old = hist_cplist[offset].oldRollAngle
            twist_angle_old = hist_cplist[offset].oldTwistAngle
            break
    return tang_overlap_old, roll_angle_old, twist_angle_old


@ti.func
def find_addition_i_shistory(end1, end2, hist_cplist, hist_object_object):
    tang_overlap_old, roll_angle_old, twist_angle_old, contact_sa = ZEROVEC3F, ZEROVEC3F, ZEROVEC3F, ZEROVEC3F
    for offset in range(hist_object_object[end1], hist_object_object[end1 + 1]):
        if end2 == hist_cplist[offset].DstID:
            tang_overlap_old = hist_cplist[offset].oldTangOverlap
            roll_angle_old = hist_cplist[offset].oldRollAngle
            twist_angle_old = hist_cplist[offset].oldTwistAngle
            contact_sa = hist_cplist[offset].contactSA
            break
    return tang_overlap_old, roll_angle_old, twist_angle_old, contact_sa


# ========================================================= #
#                   Bit Table Resolve                       #
# ========================================================= #
@ti.kernel
def bit_table_reset(contact_active: ti.template()):
    ti.loop_config(bit_vectorize=True)
    for i in contact_active:
        contact_active[i] = 0


@ti.func
def set_bit_table(end1, end2, max_row_num, contact_active):
    hash_number = pairing_mapping(end1, end2, max_row_num)
    contact_active[hash_number] = 1


@ti.func
def clear_bit_table(end1, end2, max_row_num, contact_active):
    hash_number = pairing_mapping(end1, end2, max_row_num)
    contact_active[hash_number] = 0


@ti.func
def get_bit_table(end1, end2, max_row_num, contact_active):
    hash_number = pairing_mapping(end1, end2, max_row_num)
    return int(contact_active[hash_number])


@ti.kernel
def update_contact_bit_table(
    potential_particle_num: int,
    max_particle_num: int,
    input_particle_num: int,
    particle_particle: ti.template(),
    hist_particle_particle: ti.template(),
    potential_list_particle_particle: ti.template(),
    cplist: ti.template(),
    hist_cplist: ti.template(),
    contact_active: ti.template(),
    inherit_overlap: ti.template(),
):
    for i in range(input_particle_num * potential_particle_num):
        end1 = i // potential_particle_num
        offset = i % potential_particle_num
        particle_num = particle_particle[end1 + 1] - particle_particle[end1]
        if offset < particle_num:
            nc = particle_particle[end1] + offset
            end2 = potential_list_particle_particle[i]

            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2

            exist = get_bit_table(end1, end2, max_particle_num, contact_active)
            if exist:
                for j in range(hist_particle_particle[end1], hist_particle_particle[end1 + 1]):
                    if end1 == hist_cplist[j].endID1:
                        inherit_overlap(nc, j, cplist, hist_cplist)
                        break
            else:
                clear_bit_table(end1, end2, max_particle_num, contact_active)


@ti.kernel
def update_contact_wall_bit_table(
    potential_wall_num: int,
    max_particle_num: int,
    input_particle_num: int,
    particle_wall: ti.template(),
    hist_particle_wall: ti.template(),
    potential_list_particle_wall: ti.template(),
    cplist: ti.template(),
    hist_cplist: ti.template(),
    contact_active: ti.template(),
    inherit_overlap: ti.template(),
):
    for i in range(input_particle_num * potential_wall_num):
        end1 = i // potential_wall_num
        offset = i % potential_wall_num
        particle_num = particle_wall[end1 + 1] - particle_wall[end1]
        if offset < particle_num:
            nc = particle_wall[end1] + offset
            end2 = potential_list_particle_wall[i]

            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2

            exist = get_bit_table(end1, end2, max_particle_num, contact_active)
            if exist:
                for j in range(hist_particle_wall[end1], hist_particle_wall[end1 + 1]):
                    if end1 == hist_cplist[j].endID1:
                        inherit_overlap(nc, j, cplist, hist_cplist)
                        break
            else:
                clear_bit_table(end1, end2, max_particle_num, contact_active)


# ========================================================= #
#              Particle Contact Matrix Resolve              #
# ========================================================= #
@ti.func
def implicit_surface_wall_contact_geometry(mass_center, scale, rotate_matrix, surface, wall):
    """Return the closest implicit-surface point and its signed wall gap."""
    normal = wall._get_norm(mass_center).normalized(THRESHOLD)
    local_normal = global2local(-normal, 1.0, rotate_matrix)
    physical_parameters = surface.physical_parameters(scale)
    support_point = coord_local2global(
        1.0, rotate_matrix, surface.support(local_normal, physical_parameters), mass_center
    )
    gapn = wall._get_norm_distance(support_point)
    projection_point = support_point - gapn * normal
    return normal, support_point, projection_point, gapn


@ti.func
def point_in_finite_triangle(point, vertice1, vertice2, vertice3, normal):
    """Tolerance-aware point-in-triangle test on the triangle plane."""
    edge1 = vertice2 - vertice1
    edge2 = vertice3 - vertice2
    edge3 = vertice1 - vertice3
    side1 = normal.dot(edge1.cross(point - vertice1))
    side2 = normal.dot(edge2.cross(point - vertice2))
    side3 = normal.dot(edge3.cross(point - vertice3))
    area_scale = ti.max((vertice2 - vertice1).cross(vertice3 - vertice1).norm(), THRESHOLD)
    tolerance = 1.0e-8 * area_scale
    return (side1 >= -tolerance and side2 >= -tolerance and side3 >= -tolerance) or (
        side1 <= tolerance and side2 <= tolerance and side3 <= tolerance
    )


@ti.func
def implicit_surface_segment_minimum(local_point1, local_point2, params, surface):
    """Minimise the particle potential on a finite wall edge."""
    golden = 0.6180339887498949
    lower, upper = 0.0, 1.0
    left = upper - golden * (upper - lower)
    right = lower + golden * (upper - lower)
    direction = local_point2 - local_point1
    value_left = surface.fx(*(local_point1 + left * direction), params)
    value_right = surface.fx(*(local_point1 + right * direction), params)
    for _ in range(40):
        if value_left <= value_right:
            upper = right
            right = left
            value_right = value_left
            left = upper - golden * (upper - lower)
            value_left = surface.fx(*(local_point1 + left * direction), params)
        else:
            lower = left
            left = right
            value_left = value_right
            right = lower + golden * (upper - lower)
            value_right = surface.fx(*(local_point1 + right * direction), params)

    parameter = 0.5 * (lower + upper)
    value = surface.fx(*(local_point1 + parameter * direction), params)
    value_at_start = surface.fx(*local_point1, params)
    value_at_end = surface.fx(*local_point2, params)
    if value_at_start <= value:
        parameter, value = 0.0, value_at_start
    if value_at_end <= value:
        parameter, value = 1.0, value_at_end
    return parameter, value


@ti.func
def implicit_surface_finite_wall_feature(mass_center, scale, rotate_matrix, surface, wall):
    """Return the closest triangle feature in potential space."""
    face_normal = wall._get_norm(mass_center).normalized(THRESHOLD)

    # Patch walls may represent an offset shell.  Project the reference
    # vertices onto the active side so face, edge and corner witnesses all use
    # the same physical surface as ``_get_norm_distance``.
    vertice1 = wall._get_vertice1()
    vertice2 = wall._get_vertice2()
    vertice3 = wall._get_vertice3()
    vertice1 -= wall._get_norm_distance(vertice1) * face_normal
    vertice2 -= wall._get_norm_distance(vertice2) * face_normal
    vertice3 -= wall._get_norm_distance(vertice3) * face_normal

    params = surface.physical_parameters(scale)
    local_normal = global2local(face_normal, 1.0, rotate_matrix)
    local_vertice1 = global2local(vertice1 - mass_center, 1.0, rotate_matrix)
    plane_offset = local_normal.dot(local_vertice1)
    local_plane_minimum = surface.plane_minimum(local_normal, plane_offset, params)
    plane_minimum = coord_local2global(1.0, rotate_matrix, local_plane_minimum, mass_center)

    feature_type = 1
    feature_point = plane_minimum
    edge_tangent = vec3f(0.0, 0.0, 0.0)
    if not point_in_finite_triangle(plane_minimum, vertice1, vertice2, vertice3, face_normal):
        feature_type = 0
        best_value = 1.0e30
        vertices = ti.Matrix.rows([vertice1, vertice2, vertice3])
        for edge in ti.static(range(3)):
            point1 = vertices[edge, :]
            point2 = vertices[(edge + 1) % 3, :]
            local_point1 = global2local(point1 - mass_center, 1.0, rotate_matrix)
            local_point2 = global2local(point2 - mass_center, 1.0, rotate_matrix)
            parameter, value = implicit_surface_segment_minimum(local_point1, local_point2, params, surface)
            if value < best_value:
                best_value = value
                feature_type = 2
                feature_point = point1 + parameter * (point2 - point1)
                edge_tangent = (point2 - point1).normalized(THRESHOLD)
                if parameter <= 1.0e-6:
                    feature_type = 3
                    feature_point = point1
                    edge_tangent = vec3f(0.0, 0.0, 0.0)
                elif parameter >= 1.0 - 1.0e-6:
                    feature_type = 3
                    feature_point = point2
                    edge_tangent = vec3f(0.0, 0.0, 0.0)

    return face_normal, feature_point, edge_tangent, feature_type


@ti.func
def implicit_surface_feature_line_contact(
    mass_center,
    scale,
    rotate_matrix,
    surface,
    face_normal,
    feature_point,
    edge_tangent,
    feature_type,
):
    """Recover particle and wall witnesses for an edge or corner feature."""
    params = surface.physical_parameters(scale)
    local_feature = global2local(feature_point - mass_center, 1.0, rotate_matrix)
    value_at_feature = surface.fx(*local_feature, params)
    local_gradient = surface.gradient(*local_feature, params)
    outward = local2global(local_gradient, 1.0, rotate_matrix)
    if feature_type == 2:
        outward -= outward.dot(edge_tangent) * edge_tangent
    if outward.norm_sqr() <= THRESHOLD * THRESHOLD:
        outward = feature_point - mass_center
        if feature_type == 2:
            outward -= outward.dot(edge_tangent) * edge_tangent
    if outward.norm_sqr() <= THRESHOLD * THRESHOLD:
        outward = -face_normal
    outward = outward.normalized(THRESHOLD)
    normal = -outward

    characteristic_length = THRESHOLD
    for d in ti.static(range(6)):
        characteristic_length = ti.max(characteristic_length, params[d])

    support_point = feature_point
    gapn = 1.0e30
    if value_at_feature < -1.0e-10:
        # The wall feature lies inside the particle.  Walk in the particle's
        # outward direction and bisect the first implicit-surface crossing.
        lower, upper = 0.0, characteristic_length
        upper_value = surface.fx(
            *(local_feature + upper * global2local(outward, 1.0, rotate_matrix)),
            params,
        )
        for _ in ti.static(range(8)):
            if upper_value < 0.0:
                upper *= 2.0
                upper_value = surface.fx(
                    *(local_feature + upper * global2local(outward, 1.0, rotate_matrix)),
                    params,
                )
        for _ in range(48):
            middle = 0.5 * (lower + upper)
            middle_value = surface.fx(
                *(local_feature + middle * global2local(outward, 1.0, rotate_matrix)),
                params,
            )
            if middle_value < 0.0:
                lower = middle
            else:
                upper = middle
        depth = 0.5 * (lower + upper)
        support_point = feature_point + depth * outward
        gapn = -depth
    elif value_at_feature <= 1.0e-10:
        gapn = 0.0
    else:
        # Retain a finite positive gap for cohesive cut-offs.  Search from the
        # wall feature towards the particle and bisect the entry crossing.
        local_inward = global2local(normal, 1.0, rotate_matrix)
        step = characteristic_length / 32.0
        lower, upper = 0.0, 0.0
        found = 0
        for sample in range(128):
            if found == 0:
                distance = (sample + 1) * step
                value = surface.fx(*(local_feature + distance * local_inward), params)
                if value <= 0.0:
                    lower = sample * step
                    upper = distance
                    found = 1
        if found == 1:
            for _ in range(48):
                middle = 0.5 * (lower + upper)
                middle_value = surface.fx(*(local_feature + middle * local_inward), params)
                if middle_value > 0.0:
                    lower = middle
                else:
                    upper = middle
            gapn = 0.5 * (lower + upper)
            support_point = feature_point + gapn * normal

    return normal, support_point, feature_point, gapn


@ti.func
def implicit_surface_finite_wall_contact_geometry(mass_center, scale, rotate_matrix, surface, wall):
    face_normal, feature_point, edge_tangent, feature_type = implicit_surface_finite_wall_feature(
        mass_center, scale, rotate_matrix, surface, wall
    )
    normal, support_point, projection_point, gapn = implicit_surface_wall_contact_geometry(
        mass_center, scale, rotate_matrix, surface, wall
    )
    if feature_type > 1:
        normal, support_point, projection_point, gapn = implicit_surface_feature_line_contact(
            mass_center,
            scale,
            rotate_matrix,
            surface,
            face_normal,
            feature_point,
            edge_tangent,
            feature_type,
        )
    return normal, support_point, projection_point, gapn, feature_type


@ti.kernel
def update_contact_table(
    potential_object_num: int,
    particle_num: int,
    object_object: ti.template(),
    potential_list_object_object: ti.template(),
    cplist: ti.template(),
):
    assert (
        object_object[particle_num] <= cplist.shape[0]
    ), f"Keyword:: /compaction_ratio/ is too small, at least {object_object[particle_num]} divided by {potential_list_object_object.shape[0]}"
    for i in range(particle_num * potential_object_num):
        end1 = i // potential_object_num
        offset = i % potential_object_num
        object_num = object_object[end1 + 1] - object_object[end1]
        if offset < object_num:
            nc = object_object[end1] + offset
            end2 = potential_list_object_object[i]

            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2


@ti.kernel
def update_wall_contact_table(
    potential_object_num: int,
    particle_num: int,
    rigid: ti.template(),
    surface: ti.template(),
    wall: ti.template(),
    object_object: ti.template(),
    potential_list_object_object: ti.template(),
    cplist: ti.template(),
    contact_type: ti.template(),
):
    assert (
        object_object[particle_num] <= cplist.shape[0]
    ), f"Keyword:: /compaction_ratio/ ics too small, at least {object_object[particle_num]} divided by {potential_list_object_object.shape[0]}"

    for i in range(particle_num * potential_object_num):
        end1 = i // potential_object_num
        offset = i % potential_object_num
        object_num = object_object[end1 + 1] - object_object[end1]
        if offset < object_num:
            nc = object_object[end1] + offset
            end2 = potential_list_object_object[i]

            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2
    ti.sync()

    total_contact_num = object_object[particle_num]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        mass_center, template_id, scale = (
            rigid[end1]._get_position(),
            rigid[end1]._get_template(),
            rigid[end1]._get_scale(),
        )
        rotate_matrix = set_to_rotate(rigid[end1].q)
        _, feature_point, _, feature_type = implicit_surface_finite_wall_feature(
            mass_center, scale, rotate_matrix, surface[template_id], wall[end2]
        )
        contact_type[nc] = ti.u8(feature_type)
    ti.sync()

    for particle_id in range(particle_num):
        to_beg = object_object[particle_id]
        to_end = object_object[particle_id + 1]

        for cwlist_i in range(to_beg, to_end):
            contact_type1 = int(contact_type[cwlist_i])
            if contact_type1 > 0:
                wall_id1 = cplist[cwlist_i].endID2
                mass_center = rigid[particle_id]._get_position()
                template_id = rigid[particle_id]._get_template()
                scale = rigid[particle_id]._get_scale()
                rotate_matrix = set_to_rotate(rigid[particle_id].q)
                _, feature_point1, _, _ = implicit_surface_finite_wall_feature(
                    mass_center,
                    scale,
                    rotate_matrix,
                    surface[template_id],
                    wall[wall_id1],
                )
                for cwlist_j in range(cwlist_i + 1, to_end):
                    contact_type2 = int(contact_type[cwlist_j])
                    wall_id2 = cplist[cwlist_j].endID2
                    if contact_type2 > 0:
                        _, feature_point2, _, _ = implicit_surface_finite_wall_feature(
                            mass_center,
                            scale,
                            rotate_matrix,
                            surface[template_id],
                            wall[wall_id2],
                        )
                        tolerance = 1.0e-7 * ti.max(rigid[particle_id]._get_radius(), 1.0e-6)
                        if squared(feature_point1 - feature_point2) <= tolerance * tolerance:
                            # Shared triangle edges/corners represent one
                            # physical feature. Prefer face over edge over
                            # corner, and otherwise keep the first contact.
                            if contact_type2 < contact_type1:
                                contact_type1 = 0
                            else:
                                contact_type2 = 0
                    contact_type[cwlist_j] = ti.u8(contact_type2)
            contact_type[cwlist_i] = ti.u8(contact_type1)


@ti.kernel
def update_point_particle_flag(
    particle_num: int,
    particle: ti.template(),
    object_object: ti.template(),
    cplist: ti.template(),
    contact_type: ti.template(),
):
    contact_type.fill(1)
    for particle_id in range(particle_num):
        to_beg = object_object[particle_id]
        to_end = object_object[particle_id + 1]

        for cwlist_i in range(to_beg, to_end):
            if int(contact_type[cwlist_i]) == 1:
                particle_id1 = cplist[cwlist_i].endID2
                norm1 = particle[particle_id1].normal
                for cwlist_j in range(cwlist_i + 1, to_end):
                    particle_id2 = cplist[cwlist_j].endID2
                    norm2 = particle[particle_id2].normal
                    if squared(norm1 - norm2) < THRESHOLD:
                        contact_type[cwlist_j] = 0


@ti.kernel
def update_l_scontact_table(
    potential_object_num: int,
    surface_num: int,
    object_object: ti.template(),
    potential_list_object_object: ti.template(),
    cplist: ti.template(),
):
    assert (
        object_object[surface_num] <= cplist.shape[0]
    ), f"Keyword:: /compaction_ratio/ is too small, at least {object_object[surface_num]} divided by {potential_list_object_object.shape[0]}"
    for i in range(surface_num * potential_object_num):
        node_id = i // potential_object_num
        offset = i % potential_object_num
        particle_num = object_object[node_id + 1] - object_object[node_id]
        if offset < particle_num:
            nc = object_object[node_id] + offset

            cplist[nc].endID1 = node_id
            cplist[nc].endID2 = potential_list_object_object[i]


@ti.kernel
def update_contact_table_hierarchical(
    input_particle_num: int,
    particle_particle: ti.template(),
    potential_list_particle_particle: ti.template(),
    cplist: ti.template(),
    body: ti.template(),
):
    assert (
        particle_particle[input_particle_num] <= cplist.shape[0]
    ), f"Keyword:: /compaction_ratio[0]/ is too small, at least {particle_particle[input_particle_num]} divided by {potential_list_particle_particle.shape[0]}"
    for end1 in range(input_particle_num):
        particle_num = particle_particle[end1 + 1] - particle_particle[end1]
        potential_particle_num = body[end1].potential_particle_num()
        for offset in range(particle_num):
            i = potential_particle_num + offset
            end2 = potential_list_particle_particle[i]

            nc = particle_particle[end1] + offset
            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2


@ti.kernel
def update_wall_contact_table_hierarchical(
    input_particle_num: int,
    particle_wall: ti.template(),
    potential_list_particle_wall: ti.template(),
    cplist: ti.template(),
    body: ti.template(),
):
    assert (
        particle_wall[input_particle_num] <= cplist.shape[0]
    ), f"Keyword:: /compaction_ratio[1]/ is too small, at least {particle_wall[input_particle_num]} divided by {potential_list_particle_wall.shape[0]}"
    for end1 in range(input_particle_num):
        particle_num = particle_wall[end1 + 1] - particle_wall[end1]
        potential_wall_num = body[end1].potential_wall_num()

        for offset in range(particle_num):
            i = potential_wall_num + offset
            end2 = potential_list_particle_wall[i]

            nc = particle_wall[end1] + offset
            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2


@ti.kernel
def kernel_inherit_contact_history(
    particle_num: int,
    cplist: ti.template(),
    hist_cplist: ti.template(),
    object_object: ti.template(),
    hist_object_object: ti.template(),
):
    total_contact_num = object_object[particle_num]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        tang_overlap, normal_overlap, normal_active = find_history_with_normal_overlap(
            end1, end2, hist_cplist, hist_object_object
        )
        cplist[nc].oldTangOverlap = tang_overlap
        cplist[nc].normalOverlap = normal_overlap
        cplist[nc].normalOverlapActive = normal_active


@ti.kernel
def kernel_inherit_rolling_history(
    particle_num: int,
    cplist: ti.template(),
    hist_cplist: ti.template(),
    object_object: ti.template(),
    hist_object_object: ti.template(),
):
    total_contact_num = object_object[particle_num]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        tang_overlap_old, roll_angle_old, twist_angle_old = find_addition_history(
            end1, end2, hist_cplist, hist_object_object
        )
        cplist[nc].oldTangOverlap = tang_overlap_old
        cplist[nc].oldRollAngle = roll_angle_old
        cplist[nc].oldTwistAngle = twist_angle_old


@ti.kernel
def kernel_inherit_i_scontact_history(
    particle_num: int,
    cplist: ti.template(),
    hist_cplist: ti.template(),
    object_object: ti.template(),
    hist_object_object: ti.template(),
):
    total_contact_num = object_object[particle_num]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        tang_overlap_old, contact_sa = find_i_shistory(end1, end2, hist_cplist, hist_object_object)
        cplist[nc].oldTangOverlap = tang_overlap_old
        cplist[nc].contactSA = contact_sa


@ti.kernel
def kernel_inherit_i_srolling_history(
    particle_num: int,
    cplist: ti.template(),
    hist_cplist: ti.template(),
    object_object: ti.template(),
    hist_object_object: ti.template(),
):
    total_contact_num = object_object[particle_num]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        tang_overlap_old, roll_angle_old, twist_angle_old, contact_sa = find_addition_i_shistory(
            end1, end2, hist_cplist, hist_object_object
        )
        cplist[nc].oldTangOverlap = tang_overlap_old
        cplist[nc].oldRollAngle = roll_angle_old
        cplist[nc].oldTwistAngle = twist_angle_old
        cplist[nc].contactSA = contact_sa


@ti.kernel
def copy_contact_table(
    object_object: ti.template(), particle_num: int, cplist: ti.template(), hist_cplist: ti.template()
):
    total_contact_num = object_object[particle_num]
    for nc in range(total_contact_num):
        hist_cplist[nc].DstID = cplist[nc].endID2
        hist_cplist[nc].oldTangOverlap = cplist[nc].oldTangOverlap
        hist_cplist[nc].normalOverlap = cplist[nc].normalOverlap
        hist_cplist[nc].normalOverlapActive = cplist[nc].normalOverlapActive


@ti.kernel
def copy_lsmpm_contact_table(
    object_object: ti.template(),
    contact_node_num: int,
    cplist: ti.template(),
    hist_cplist: ti.template(),
):
    total_contact_num = object_object[contact_node_num]
    for nc in range(total_contact_num):
        hist_cplist[nc].DstID = cplist[nc].endID2
        hist_cplist[nc].oldTangOverlap = cplist[nc].oldTangOverlap
        hist_cplist[nc].normalOverlap = cplist[nc].normalOverlap
        hist_cplist[nc].normalOverlapActive = cplist[nc].normalOverlapActive


@ti.kernel
def kernel_inherit_lsmpm_contact_history(
    contact_node_num: int,
    cplist: ti.template(),
    hist_cplist: ti.template(),
    object_object: ti.template(),
    hist_object_object: ti.template(),
):
    total_contact_num = object_object[contact_node_num]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        tang_overlap, normal_overlap, normal_active = find_history_with_normal_overlap(
            end1, end2, hist_cplist, hist_object_object
        )
        cplist[nc].oldTangOverlap = tang_overlap
        cplist[nc].normalOverlap = normal_overlap
        cplist[nc].normalOverlapActive = normal_active


@ti.kernel
def copy_addition_contact_table(
    object_object: ti.template(), particle_num: int, cplist: ti.template(), hist_cplist: ti.template()
):
    total_contact_num = object_object[particle_num]
    for nc in range(total_contact_num):
        hist_cplist[nc].DstID = cplist[nc].endID2
        hist_cplist[nc].oldTangOverlap = cplist[nc].oldTangOverlap
        hist_cplist[nc].oldRollAngle = cplist[nc].oldRollAngle
        hist_cplist[nc].oldTwistAngle = cplist[nc].oldTwistAngle


@ti.kernel
def kernel_update_active_collisions(
    particle_num: int, particle: ti.template(), cplist: ti.template(), object_object: ti.template()
):
    total_contact_num = object_object[particle_num]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        pos1, pos2 = particle[end1].x, particle[end2].x
        rad1, rad2 = particle[end1].rad, particle[end2].rad
        gapn = (pos1 - pos2).norm() - (rad1 + rad2)
        cplist[nc].avtice = ti.u8(gapn < 0.0)


@ti.kernel
def kernel_particle_particle_force_assemble(
    particle_num: int,
    dt: ti.template(),
    max_material_num: int,
    surface_props: ti.template(),
    particle1: ti.template(),
    particle2: ti.template(),
    cplist: ti.template(),
    particle_particle: ti.template(),
    contact_model: ti.template(),
):
    total_contact_num = particle_particle[particle_num]
    # ti.block_local(dt)
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        mat_id1, mat_id2 = particle1[end1].materialID, particle2[end2].materialID
        pos1, pos2 = particle1[end1]._get_position(), particle2[end2]._get_position()
        rad1, rad2 = particle1[end1]._get_radius(), particle2[end2]._get_radius()
        if ti.static(GlobalVariable.DEMXPBC):
            if ti.abs(pos2[0] - pos1[0]) > 0.5 * GlobalVariable.DEMXSIZE:
                pos2[0] += (
                    ti.cast(pos2[0] < 0.5 * GlobalVariable.DEMXSIZE, float)
                    - ti.cast(pos2[0] > 0.5 * GlobalVariable.DEMXSIZE, float)
                ) * GlobalVariable.DEMXSIZE
        if ti.static(GlobalVariable.DEMYPBC):
            if ti.abs(pos2[1] - pos1[1]) > 0.5 * GlobalVariable.DEMYSIZE:
                pos2[1] += (
                    ti.cast(pos2[1] < 0.5 * GlobalVariable.DEMYSIZE, float)
                    - ti.cast(pos2[1] > 0.5 * GlobalVariable.DEMYSIZE, float)
                ) * GlobalVariable.DEMYSIZE
        if ti.static(GlobalVariable.DEMZPBC):
            if ti.abs(pos2[2] - pos1[2]) > 0.5 * GlobalVariable.DEMZSIZE:
                pos2[2] += (
                    ti.cast(pos2[2] < 0.5 * GlobalVariable.DEMZSIZE, float)
                    - ti.cast(pos2[2] > 0.5 * GlobalVariable.DEMZSIZE, float)
                ) * GlobalVariable.DEMZSIZE
        gapn = (pos1 - pos2).norm() - (rad1 + rad2)
        material_id = pairing_mapping(mat_id1, mat_id2, max_material_num)

        if gapn < surface_props[material_id].ncut:
            norm = (pos1 - pos2).normalized(THRESHOLD)
            contact_model(
                material_id,
                nc,
                end1,
                end2,
                gapn,
                norm,
                pos1,
                pos2,
                rad1,
                rad2,
                particle1,
                particle2,
                surface_props,
                cplist,
                dt,
            )
        else:
            cplist[nc]._no_contact()


@ti.kernel
def kernel_enhanced_coupling_particle_particle_force_assemble(
    particle_num: int,
    dt: ti.template(),
    max_material_num: int,
    surface_props: ti.template(),
    particle1: ti.template(),
    particle2: ti.template(),
    cplist: ti.template(),
    particle_particle: ti.template(),
    contact_flag: ti.template(),
    contact_model: ti.template(),
):
    total_contact_num = particle_particle[particle_num]
    # ti.block_local(dt)
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        mat_id1, mat_id2 = particle1[end1].materialID, particle2[end2].materialID
        pos1, pos2 = particle1[end1]._get_position(), particle2[end2]._get_position()
        rad1, rad2 = particle1[end1]._get_radius(), particle2[end2]._get_radius()
        norm = particle2[end2].normal
        crit = (pos1 - pos2).norm() - (rad1 + ti.sqrt(2) * rad2)
        material_id = pairing_mapping(mat_id1, mat_id2, max_material_num)

        if crit < surface_props[material_id].ncut:
            gapn = (pos1 - pos2).dot(norm) - (rad1 + rad2)
            contact_model(
                material_id,
                nc,
                end1,
                end2,
                gapn,
                norm,
                pos1,
                pos2,
                rad1,
                rad2,
                particle1,
                particle2,
                surface_props,
                cplist,
                dt,
            )
        else:
            cplist[nc]._no_contact()


@ti.kernel
def kernel_l_sparticle_l_sparticle_force_assemble(
    surface_num: int,
    dt: ti.template(),
    max_material_num: int,
    surface_props: ti.template(),
    rigid: ti.template(),
    grid: ti.template(),
    vertice: ti.template(),
    surface: ti.template(),
    box: ti.template(),
    cplist: ti.template(),
    particle_particle: ti.template(),
    contact_model: ti.template(),
):
    total_contact_num = particle_particle[surface_num]
    for nc in range(total_contact_num):
        global_node, end2 = cplist[nc].endID1, cplist[nc].endID2
        end1 = surface[global_node]
        local_node = rigid[end1].global_node_to_local(global_node)
        mass_center1, mass_center2 = rigid[end1]._get_position(), rigid[end2]._get_position()

        if ti.static(GlobalVariable.DEMXPBC):
            if ti.abs(mass_center2[0] - mass_center1[0]) > 0.5 * GlobalVariable.DEMXSIZE:
                mass_center2[0] += (
                    ti.cast(mass_center2[0] < 0.5 * GlobalVariable.DEMXSIZE, float)
                    - ti.cast(mass_center2[0] > 0.5 * GlobalVariable.DEMXSIZE, float)
                ) * GlobalVariable.DEMXSIZE
        if ti.static(GlobalVariable.DEMYPBC):
            if ti.abs(mass_center2[1] - mass_center1[1]) > 0.5 * GlobalVariable.DEMYSIZE:
                mass_center2[1] += (
                    ti.cast(mass_center2[1] < 0.5 * GlobalVariable.DEMYSIZE, float)
                    - ti.cast(mass_center2[1] > 0.5 * GlobalVariable.DEMYSIZE, float)
                ) * GlobalVariable.DEMYSIZE
        if ti.static(GlobalVariable.DEMZPBC):
            if ti.abs(mass_center2[2] - mass_center1[2]) > 0.5 * GlobalVariable.DEMZSIZE:
                mass_center2[2] += (
                    ti.cast(mass_center2[2] < 0.5 * GlobalVariable.DEMZSIZE, float)
                    - ti.cast(mass_center2[2] > 0.5 * GlobalVariable.DEMZSIZE, float)
                ) * GlobalVariable.DEMZSIZE

        rotate_matrix1, rotate_matrix2 = set_to_rotate(rigid[end1].q), set_to_rotate(rigid[end2].q)
        mat_id1, mat_id2 = rigid[end1].materialID, rigid[end2].materialID
        material_id = pairing_mapping(mat_id1, mat_id2, max_material_num)

        surface_node = mass_center1 + rotate_matrix1 @ (box[end1].scale * vertice[local_node].x)
        surface_node_in_box = rotate_matrix2.transpose() @ (surface_node - mass_center2)
        if not box[end2]._in_box(surface_node_in_box):
            cplist[nc]._no_contact()
            continue
        gapn = box[end2].distance(surface_node_in_box, grid)
        parameter = vertice[local_node].parameter
        if gapn < surface_props[material_id].ncut:
            contact_model(
                material_id,
                nc,
                parameter,
                end1,
                end2,
                0.0,
                gapn,
                mass_center1,
                mass_center2,
                surface_node,
                surface_node_in_box,
                rotate_matrix2,
                rigid,
                grid,
                rigid,
                box,
                surface_props,
                cplist,
                dt,
            )
            if int(rigid[end1].is_soft) == 1 and cplist[nc]._is_active():
                vertice[local_node]._update_contact_interaction(cplist[nc].cnforce + cplist[nc].csforce, ZEROVEC3F)
        else:
            cplist[nc]._no_contact()


@ti.kernel
def kernel_i_sparticle_i_sparticle_force_assemble(
    particle_num: int,
    dt: ti.template(),
    max_material_num: int,
    surface_props: ti.template(),
    particle: ti.template(),
    rigid: ti.template(),
    surface: ti.template(),
    cplist: ti.template(),
    particle_particle: ti.template(),
    contact_model: ti.template(),
    iterative_model: ti.template(),
):
    total_contact_num = particle_particle[particle_num]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        mass_center1, mass_center2 = rigid[end1]._get_position(), rigid[end2]._get_position()

        if ti.static(GlobalVariable.DEMXPBC):
            if ti.abs(mass_center2[0] - mass_center1[0]) > 0.5 * GlobalVariable.DEMXSIZE:
                mass_center2[0] += (
                    ti.cast(mass_center2[0] < 0.5 * GlobalVariable.DEMXSIZE, float)
                    - ti.cast(mass_center2[0] > 0.5 * GlobalVariable.DEMXSIZE, float)
                ) * GlobalVariable.DEMXSIZE
        if ti.static(GlobalVariable.DEMYPBC):
            if ti.abs(mass_center2[1] - mass_center1[1]) > 0.5 * GlobalVariable.DEMYSIZE:
                mass_center2[1] += (
                    ti.cast(mass_center2[1] < 0.5 * GlobalVariable.DEMYSIZE, float)
                    - ti.cast(mass_center2[1] > 0.5 * GlobalVariable.DEMYSIZE, float)
                ) * GlobalVariable.DEMYSIZE
        if ti.static(GlobalVariable.DEMZPBC):
            if ti.abs(mass_center2[2] - mass_center1[2]) > 0.5 * GlobalVariable.DEMZSIZE:
                mass_center2[2] += (
                    ti.cast(mass_center2[2] < 0.5 * GlobalVariable.DEMZSIZE, float)
                    - ti.cast(mass_center2[2] > 0.5 * GlobalVariable.DEMZSIZE, float)
                ) * GlobalVariable.DEMZSIZE

        radius1, radius2 = particle[end1].rad, particle[end2].rad
        mat_id1, mat_id2 = rigid[end1].materialID, rigid[end2].materialID
        scale1, scale2 = rigid[end1].scale, rigid[end2].scale
        material_id = pairing_mapping(mat_id1, mat_id2, max_material_num)
        rotate_matrix1, rotate_matrix2 = set_to_rotate(rigid[end1].q), set_to_rotate(rigid[end2].q)
        margin1, margin2 = rigid[end1]._get_margin(), rigid[end2]._get_margin()
        template_id1, template_id2 = rigid[end1].templateID, rigid[end2].templateID

        extra_hist = cplist[nc].contactSA
        is_touch, pa, pb, extra_hist = iterative_model(
            margin1,
            margin2,
            radius1,
            radius2,
            mass_center1,
            mass_center2,
            extra_hist,
            scale1,
            scale2,
            rotate_matrix1,
            rotate_matrix2,
            surface[template_id1],
            surface[template_id2],
        )

        if is_touch:
            contact_model(
                material_id,
                nc,
                end1,
                end2,
                mass_center1,
                mass_center2,
                pa,
                pb,
                extra_hist,
                rigid,
                rigid,
                surface_props,
                cplist,
                dt,
            )
        else:
            cplist[nc]._no_contact(extra_hist)


@ti.kernel
def kernel_particle_i_sparticle_force_assemble(
    particle_num: int,
    dt: ti.template(),
    max_material_num: int,
    surface_props: ti.template(),
    particle: ti.template(),
    rigid: ti.template(),
    mpm_surface: ti.template(),
    dem_surface: ti.template(),
    cplist: ti.template(),
    particle_particle: ti.template(),
    contact_model: ti.template(),
    iterative_model: ti.template(),
):
    total_contact_num = particle_particle[particle_num]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        mass_center1, mass_center2 = particle[end1]._get_position(), rigid[end2]._get_position()
        mat_id1, mat_id2 = particle[end1].materialID, rigid[end2].materialID
        material_id = pairing_mapping(mat_id1, mat_id2, max_material_num)
        position1, position2 = particle[end1]._get_position(), rigid[end2]._get_position()
        rotate_matrix1, rotate_matrix2 = set_to_rotate(rigid[end1].q), set_to_rotate(rigid[end2].q)
        margin1, margin2 = particle[end1]._get_margin(), rigid[end2]._get_margin()
        template_id1, template_id2 = int(particle[end1].bodyID), rigid[end2].templateID

        extra_hist = cplist[nc].contactSA
        is_touch, pa, pb, extra_hist = iterative_model(
            margin1,
            margin2,
            position1,
            position2,
            extra_hist,
            rotate_matrix1,
            rotate_matrix2,
            mpm_surface[template_id1],
            dem_surface[template_id2],
        )

        if is_touch:
            contact_model(
                material_id,
                nc,
                end1,
                end2,
                mass_center1,
                mass_center2,
                pa,
                pb,
                extra_hist,
                rigid,
                rigid,
                surface_props,
                cplist,
                dt,
            )
        else:
            cplist[nc]._no_contact(extra_hist)


@ti.kernel
def kernel_particle_l_sparticle_force_assemble(
    particle_num: int,
    dt: ti.template(),
    max_material_num: int,
    surface_props: ti.template(),
    particle: ti.template(),
    rigid: ti.template(),
    grid: ti.template(),
    box: ti.template(),
    cplist: ti.template(),
    particle_particle: ti.template(),
    contact_model: ti.template(),
):
    total_contact_num = particle_particle[particle_num]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        mass_center1, mass_center2 = particle[end1]._get_position(), rigid[end2]._get_position()
        rad1, rotate_matrix2 = particle[end1]._get_radius(), set_to_rotate(rigid[end2].q)
        mat_id1, mat_id2 = particle[end1].materialID, rigid[end2].materialID
        material_id = pairing_mapping(mat_id1, mat_id2, max_material_num)

        surface_node_in_box = rotate_matrix2.transpose() @ (mass_center1 - mass_center2)
        if not box[end2]._in_box(surface_node_in_box):
            cplist[nc]._no_contact()
            continue
        gapn = box[end2].distance(surface_node_in_box, grid) - rad1
        if gapn < surface_props[material_id].ncut:
            contact_model(
                material_id,
                nc,
                1.0,
                end1,
                end2,
                rad1,
                gapn,
                mass_center1,
                mass_center2,
                mass_center1,
                surface_node_in_box,
                rotate_matrix2,
                particle,
                grid,
                rigid,
                box,
                surface_props,
                cplist,
                dt,
            )
        else:
            cplist[nc]._no_contact()


@ti.kernel
def kernel_particle_wall_force_assemble(
    particle_num: int,
    dt: ti.template(),
    max_material_num: int,
    surface_props: ti.template(),
    particle: ti.template(),
    wall: ti.template(),
    cplist: ti.template(),
    particle_wall: ti.template(),
    contact_model: ti.template(),
):
    total_contact_num = particle_wall[particle_num]
    # ti.block_local(dt)
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        pos1, particle_rad = particle[end1]._get_position(), particle[end1]._get_radius()
        distance = wall[end2]._get_norm_distance(pos1)
        gapn = distance - particle_rad

        mat_id1, mat_id2 = particle[end1].materialID, wall[end2].materialID
        material_id = pairing_mapping(mat_id1, mat_id2, max_material_num)
        fraction = ti.abs(wall[end2].process_circle_shape(pos1, particle_rad, distance))

        if int(wall[end2].active) == 1 and gapn < surface_props[material_id].ncut and fraction > THRESHOLD:
            contact_model(
                material_id,
                nc,
                end1,
                end2,
                gapn,
                fraction,
                pos1,
                particle_rad,
                particle,
                wall,
                surface_props,
                cplist,
                dt,
            )
        else:
            cplist[nc]._no_contact()


@ti.kernel
def kernel_i_sparticle_wall_force_assemble(
    particle_num: int,
    wall_type: ti.template(),
    dt: ti.template(),
    max_material_num: int,
    surface_props: ti.template(),
    rigid: ti.template(),
    surface: ti.template(),
    wall: ti.template(),
    cplist: ti.template(),
    particle_wall: ti.template(),
    contact_type: ti.template(),
    contact_model: ti.template(),
):
    total_contact_num = particle_wall[particle_num]
    # ti.block_local(dt)
    for nc in range(total_contact_num):
        if int(contact_type[nc]) > 0:
            end1, end2 = cplist[nc].endID1, cplist[nc].endID2
            if int(wall[end2].active) == 0:
                cplist[nc]._no_contact()
                continue
            mass_center, template_id, scale = (
                rigid[end1]._get_position(),
                rigid[end1]._get_template(),
                rigid[end1]._get_scale(),
            )
            rotate_matrix = set_to_rotate(rigid[end1].q)

            mat_id1, mat_id2 = rigid[end1].materialID, wall[end2].materialID
            material_id = pairing_mapping(mat_id1, mat_id2, max_material_num)
            fraction = ti.abs(wall[end2].process_implicit_surface_shape())

            normal = vec3f(0.0, 0.0, 0.0)
            support_point = vec3f(0.0, 0.0, 0.0)
            projection_point = vec3f(0.0, 0.0, 0.0)
            gapn = 1.0e30
            if ti.static(wall_type == 0):
                normal, support_point, projection_point, gapn = implicit_surface_wall_contact_geometry(
                    mass_center,
                    scale,
                    rotate_matrix,
                    surface[template_id],
                    wall[end2],
                )
            else:
                normal, support_point, projection_point, gapn, _ = implicit_surface_finite_wall_contact_geometry(
                    mass_center,
                    scale,
                    rotate_matrix,
                    surface[template_id],
                    wall[end2],
                )
            if gapn < surface_props[material_id].ncut and fraction > THRESHOLD:
                contact_model(
                    material_id,
                    nc,
                    end1,
                    end2,
                    fraction,
                    gapn,
                    normal,
                    mass_center,
                    support_point,
                    projection_point,
                    rigid,
                    wall,
                    surface_props,
                    cplist,
                    dt,
                )
            else:
                cplist[nc]._no_contact()
        else:
            cplist[nc]._no_contact()


@ti.kernel
def kernel_l_sparticle_wall_force_assemble(
    surface_num: int,
    dt: ti.template(),
    max_material_num: int,
    surface_props: ti.template(),
    rigid: ti.template(),
    vertice: ti.template(),
    surface: ti.template(),
    box: ti.template(),
    wall: ti.template(),
    cplist: ti.template(),
    particle_wall: ti.template(),
    contact_model: ti.template(),
):
    # ti.block_local(dt)
    total_contact_num = particle_wall[surface_num]
    for nc in range(total_contact_num):
        global_node, end2 = cplist[nc].endID1, cplist[nc].endID2
        end1 = surface[global_node]
        local_node = rigid[end1].global_node_to_local(global_node)
        mass_center1 = rigid[end1]._get_position()
        rotate_matrix1 = set_to_rotate(rigid[end1].q)
        mat_id1, mat_id2 = rigid[end1].materialID, wall[end2].materialID
        material_id = pairing_mapping(mat_id1, mat_id2, max_material_num)

        surface_node = mass_center1 + rotate_matrix1 @ (box[end1].scale * vertice[local_node].x)
        gapn = wall[end2]._get_norm_distance(surface_node)
        projected_point = wall[end2]._point_projection(surface_node)
        parameter = vertice[local_node].parameter
        if (
            int(wall[end2].active) == 1
            and wall[end2]._is_in_plane(projected_point)
            and gapn < surface_props[material_id].ncut
        ):
            contact_model(
                material_id,
                nc,
                parameter,
                end1,
                end2,
                gapn,
                mass_center1,
                surface_node,
                rigid,
                wall,
                surface_props,
                cplist,
                dt,
            )
            if int(rigid[end1].is_soft) == 1 and cplist[nc]._is_active():
                vertice[local_node]._update_contact_interaction(cplist[nc].cnforce + cplist[nc].csforce, ZEROVEC3F)
        else:
            cplist[nc]._no_contact()


@ti.kernel
def kernel_accumulate_linear_lsparticle_wall_elastic_energy(
    surface_num: int,
    target_wall_id: int,
    max_material_num: int,
    surface_props: ti.template(),
    rigid: ti.template(),
    vertices: ti.template(),
    surface: ti.template(),
    boxes: ti.template(),
    walls: ti.template(),
    contacts: ti.template(),
    particle_wall: ti.template(),
    elastic_energy: ti.template(),
):
    """Measure the penalty energy removed with one LSDEM facet wall."""

    elastic_energy[None] = 0.0
    total_contact_num = particle_wall[surface_num]
    for contact in range(total_contact_num):
        global_node = contacts[contact].endID1
        wall_index = contacts[contact].endID2
        body = surface[global_node]
        if int(walls[wall_index].active) == 1 and int(walls[wall_index].wallID) == target_wall_id:
            local_node = rigid[body].global_node_to_local(global_node)
            center = rigid[body]._get_position()
            rotation = set_to_rotate(rigid[body].q)
            point = center + rotation @ (boxes[body].scale * vertices[local_node].x)
            gap = walls[wall_index]._get_norm_distance(point)
            projection = walls[wall_index]._point_projection(point)
            if gap < 0.0 and walls[wall_index]._is_in_plane(projection):
                material = pairing_mapping(
                    rigid[body].materialID,
                    walls[wall_index].materialID,
                    max_material_num,
                )
                coefficient = vertices[local_node].parameter
                contact_radius = rigid[body]._get_contact_radius(point)
                normal_stiffness, tangential_stiffness = surface_props[material]._get_stiffness(
                    coefficient, contact_radius
                )
                overlap = contacts[contact].oldTangOverlap
                ti.atomic_add(
                    elastic_energy[None],
                    0.5 * normal_stiffness * gap * gap + 0.5 * tangential_stiffness * overlap.norm_sqr(),
                )


@ti.kernel
def kernel_compact_contact_table(total_num: int, compact_table: ti.template(), active_contact: ti.template()):
    for i in range(1, total_num + 1):
        if active_contact[i] - active_contact[i - 1] == 1:
            compact_table[active_contact[i - 1]] = i - 1


@ti.kernel
def kernel_calculate_contact_force(
    particle_num: int,
    dt: ti.template(),
    max_material_num: int,
    surface_props: ti.template(),
    particle: ti.template(),
    cplist: ti.template(),
    particle_particle: ti.template(),
    compact_table: ti.template(),
    active_contact: ti.template(),
):
    for i in range(active_contact[particle_particle[particle_num]]):
        nc = compact_table[i]
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        mat_id1, mat_id2 = particle[end1].materialID, particle[end2].materialID
        pos1, pos2 = particle[end1].x, particle[end2].x
        rad1, rad2 = particle[end1].rad, particle[end2].rad
        gapn = (pos1 - pos2).norm() - (rad1 + rad2)
        material_id = pairing_mapping(mat_id1, mat_id2, max_material_num)

        if gapn < 0.0:
            norm = (pos1 - pos2).normalized(THRESHOLD)
            cpos = pos2 + (rad2 + 0.5 * gapn) * norm
            surface_props[material_id]._particle_particle_force_assemble(
                nc, end1, end2, gapn, norm, cpos, dt, particle, cplist
            )
        else:
            cplist[nc]._no_contact()


@ti.kernel
def kernel_rebulid_coupling_history_contact_list(
    cplist: ti.template(),
    hist_object_object: ti.template(),
    object_object: ti.types.ndarray(),
    dst: ti.types.ndarray(),
    old_tang_overlap: ti.types.ndarray(),
):
    for i in range(object_object.shape[0]):
        hist_object_object[i] = object_object[i]

    for cp in range(object_object[object_object.shape[0] - 1]):
        cplist[cp].endID2 = dst[cp]
        cplist[cp].oldTangOverlap = vec3f(old_tang_overlap[cp, 0], old_tang_overlap[cp, 1], old_tang_overlap[cp, 2])


@ti.kernel
def kernel_rebulid_history_contact_list(
    cplist: ti.template(),
    hist_object_object: ti.template(),
    object_object: ti.types.ndarray(),
    dst: ti.types.ndarray(),
    normal_force: ti.types.ndarray(),
    tangential_force: ti.types.ndarray(),
    old_tang_overlap: ti.types.ndarray(),
):
    for i in range(object_object.shape[0]):
        hist_object_object[i] = object_object[i]

    for cp in range(object_object[object_object.shape[0] - 1]):
        cplist[cp].endID2 = dst[cp]
        cplist[cp].cnforce = vec3f(normal_force[cp, 0], normal_force[cp, 1], normal_force[cp, 2])
        cplist[cp].csforce = vec3f(tangential_force[cp, 0], tangential_force[cp, 1], tangential_force[cp, 2])
        cplist[cp].oldTangOverlap = vec3f(old_tang_overlap[cp, 0], old_tang_overlap[cp, 1], old_tang_overlap[cp, 2])


@ti.kernel
def kernel_rebulid_addition_history_contact_list(
    cplist: ti.template(),
    hist_object_object: ti.template(),
    object_object: ti.types.ndarray(),
    dst: ti.types.ndarray(),
    normal_force: ti.types.ndarray(),
    tangential_force: ti.types.ndarray(),
    old_tang_overlap: ti.types.ndarray(),
    old_roll_angle: ti.types.ndarray(),
    old_twist_angle: ti.types.ndarray(),
):
    for i in range(object_object.shape[0]):
        hist_object_object[i] = object_object[i]

    for cp in range(object_object[object_object.shape[0] - 1]):
        cplist[cp].endID2 = dst[cp]
        cplist[cp].cnforce = vec3f(normal_force[cp, 0], normal_force[cp, 1], normal_force[cp, 2])
        cplist[cp].csforce = vec3f(tangential_force[cp, 0], tangential_force[cp, 1], tangential_force[cp, 2])
        cplist[cp].oldTangOverlap = vec3f(old_tang_overlap[cp, 0], old_tang_overlap[cp, 1], old_tang_overlap[cp, 2])
        cplist[cp].oldRollAngle = vec3f(old_roll_angle[cp, 0], old_roll_angle[cp, 1], old_roll_angle[cp, 2])
        cplist[cp].oldTwistAngle = vec3f(old_twist_angle[cp, 0], old_twist_angle[cp, 1], old_twist_angle[cp, 2])


@ti.func
def stiffness_parameter(rad1, rad2):
    if ti.static(GlobalVariable.ADAPTIVESTIFF):
        return min(rad1, rad2) ** 2 / (rad1 + rad2)
    else:
        return 0.0


@ti.func
def particle_contact_model_type1(
    material_id, nc, end1, end2, gapn, norm, pos1, pos2, rad1, rad2, particle1, particle2, surface_props, cplist, dt
):
    mass1, mass2 = particle1[end1]._get_mass(), particle2[end2]._get_mass()
    vel1, vel2 = particle1[end1]._get_velocity(), particle2[end2]._get_velocity()
    w1, w2 = particle1[end1]._get_angular_velocity(), particle2[end2]._get_angular_velocity()
    tang_overlap_old = cplist[nc].oldTangOverlap

    m_eff = effective_value(mass1, mass2)
    rad_eff = effective_value(rad1 + 0.5 * gapn, rad2 + 0.5 * gapn)
    cpos = pos2 + (rad2 - 0.5 * gapn) * norm
    v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))
    w_rel = w1 - w2

    param = stiffness_parameter(rad1, rad2)

    normal_force, tangential_force, momentum, tang_over_temp = surface_props[material_id]._force_assemble(
        m_eff, rad_eff, gapn, 1.0, param, norm, v_rel, w_rel, tang_overlap_old, dt
    )

    ftotal = normal_force + tangential_force
    momentum1 = tangential_force.cross(pos1 - cpos) + momentum
    momentum2 = tangential_force.cross(cpos - pos2) - momentum

    cplist[nc]._set_contact(normal_force, tangential_force, tang_over_temp)
    particle1[end1]._update_contact_interaction(ftotal, momentum1)
    particle2[end2]._update_contact_interaction(-ftotal, momentum2)


@ti.func
def particle_contact_model_type2(
    material_id, nc, end1, end2, gapn, norm, pos1, pos2, rad1, rad2, particle1, particle2, surface_props, cplist, dt
):
    mass1, mass2 = particle1[end1]._get_mass(), particle2[end2]._get_mass()
    vel1, vel2 = particle1[end1]._get_velocity(), particle2[end2]._get_velocity()
    w1, w2 = particle1[end1]._get_angular_velocity(), particle2[end2]._get_angular_velocity()
    m_eff = effective_value(mass1, mass2)
    rad_eff = effective_value(rad1, rad2)

    cpos = pos2 + (rad2 - 0.5 * gapn) * norm
    v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))
    w_rel = w1 - w2
    wr_rel = norm.cross(w1) - norm.cross(w2)

    tang_overlap_old = cplist[nc].oldTangOverlap
    tang_rolling_old = cplist[nc].oldRollAngle
    tang_twisting_old = cplist[nc].oldTwistAngle

    param = stiffness_parameter(rad1, rad2)

    normal_force, tangential_force, momentum, tang_over_temp, tang_rolling_temp, tang_twisting_temp = surface_props[
        material_id
    ]._force_assemble(
        m_eff,
        rad_eff,
        gapn,
        1.0,
        param,
        norm,
        v_rel,
        w_rel,
        wr_rel,
        tang_overlap_old,
        tang_rolling_old,
        tang_twisting_old,
        dt,
    )
    ftotal = normal_force + tangential_force
    resultant_momentum1 = tangential_force.cross(pos1 - cpos) + momentum
    resultant_momentum2 = tangential_force.cross(cpos - pos2) - momentum

    cplist[nc]._set_contact(normal_force, tangential_force, tang_over_temp, tang_rolling_temp, tang_twisting_temp)
    particle1[end1]._update_contact_interaction(ftotal, resultant_momentum1)
    particle2[end2]._update_contact_interaction(-ftotal, resultant_momentum2)


@ti.func
def i_sparticle_contact_model_type1(
    material_id,
    nc,
    end1,
    end2,
    mass_center1,
    mass_center2,
    pa,
    pb,
    extra_hist,
    rigid1,
    rigid2,
    surface_props,
    cplist,
    dt,
):
    mass1, mass2 = rigid1[end1]._get_mass(), rigid2[end2]._get_mass()
    rad1, rad2 = rigid1[end1]._get_radius(), rigid2[end2]._get_radius()
    vel1, vel2 = rigid1[end1]._get_velocity(), rigid2[end2]._get_velocity()
    w1, w2 = rigid1[end1]._get_angular_velocity(), rigid2[end2]._get_angular_velocity()
    tang_overlap_old = cplist[nc].oldTangOverlap

    m_eff = effective_value(mass1, mass2)
    rad_eff = effective_value(rad1, rad2)
    cpos = 0.5 * (pa + pb)
    gapn = -(pa - pb).norm()
    contact_normal = -(pa - pb).normalized(THRESHOLD)
    v_rel = vel1 + w1.cross(cpos - mass_center1) - (vel2 + w2.cross(cpos - mass_center2))
    w_rel = w1 - w2

    param = stiffness_parameter(rad1, rad2)

    normal_force, tangential_force, momentum, tang_over_temp = surface_props[material_id]._force_assemble(
        m_eff, rad_eff, gapn, 1.0, param, contact_normal, v_rel, w_rel, tang_overlap_old, dt
    )

    ftotal = normal_force + tangential_force
    momentum1 = ftotal.cross(mass_center1 - cpos) + momentum
    momentum2 = ftotal.cross(cpos - mass_center2) - momentum

    cplist[nc]._set_contact(normal_force, tangential_force, tang_over_temp, extra_hist)
    rigid1[end1]._update_contact_interaction(ftotal, momentum1)
    rigid2[end2]._update_contact_interaction(-ftotal, momentum2)


@ti.func
def i_sparticle_contact_model_type2(
    material_id,
    nc,
    end1,
    end2,
    mass_center1,
    mass_center2,
    pa,
    pb,
    extra_hist,
    rigid1,
    rigid2,
    surface_props,
    cplist,
    dt,
):
    mass1, mass2 = rigid1[end1]._get_mass(), rigid2[end2]._get_mass()
    rad1, rad2 = rigid1[end1]._get_radius(), rigid2[end2]._get_radius()
    vel1, vel2 = rigid1[end1]._get_velocity(), rigid2[end2]._get_velocity()
    w1, w2 = rigid1[end1]._get_angular_velocity(), rigid2[end2]._get_angular_velocity()

    m_eff = effective_value(mass1, mass2)
    rad_eff = effective_value(rad1, rad2)
    gapn = -(pa - pb).norm()
    contact_normal = -(pa - pb).normalized(THRESHOLD)
    cpos = 0.5 * (pa + pb)
    v_rel = vel1 + w1.cross(cpos - mass_center1) - (vel2 + w2.cross(cpos - mass_center2))
    w_rel = w1 - w2
    wr_rel = contact_normal.cross(w1) - contact_normal.cross(w2)

    tang_overlap_old = cplist[nc].oldTangOverlap
    tang_rolling_old = cplist[nc].oldRollAngle
    tang_twisting_old = cplist[nc].oldTwistAngle

    param = stiffness_parameter(rad1, rad2)

    normal_force, tangential_force, momentum, tang_over_temp, tang_rolling_temp, tang_twisting_temp = surface_props[
        material_id
    ]._force_assemble(
        m_eff,
        rad_eff,
        gapn,
        1.0,
        param,
        contact_normal,
        v_rel,
        w_rel,
        wr_rel,
        tang_overlap_old,
        tang_rolling_old,
        tang_twisting_old,
        dt,
    )
    ftotal = normal_force + tangential_force
    resultant_momentum1 = ftotal.cross(mass_center1 - cpos) + momentum
    resultant_momentum2 = ftotal.cross(cpos - mass_center2) - momentum

    cplist[nc]._set_contact(
        normal_force, tangential_force, tang_over_temp, tang_rolling_temp, tang_twisting_temp, extra_hist
    )
    rigid1[end1]._update_contact_interaction(ftotal, resultant_momentum1)
    rigid2[end2]._update_contact_interaction(-ftotal, resultant_momentum2)


@ti.func
def l_sparticle_contact_model_type0(
    material_id,
    nc,
    coeff,
    end1,
    end2,
    contact_radius,
    min_dist,
    mass_center1,
    mass_center2,
    global_intruding_node,
    local_intruding_node,
    rotate_matrix2,
    object,
    grid,
    rigid,
    box,
    surface_props,
    cplist,
    dt,
):
    vel1, vel2 = object[end1]._get_velocity(), rigid[end2]._get_velocity()
    w1, w2 = object[end1]._get_angular_velocity(), rigid[end2]._get_angular_velocity()

    dgdx = rotate_matrix2 @ box[end2].calculate_gradient(local_intruding_node, grid)
    norm = dgdx.normalized(THRESHOLD)
    cpos = global_intruding_node - 0.5 * norm * (min_dist + contact_radius)  # is right ??
    v_rel = vel1 + w1.cross(cpos - mass_center1) - (vel2 + w2.cross(cpos - mass_center2))
    tang_overlap_old = cplist[nc].oldTangOverlap
    mass1, mass2 = object[end1]._get_mass(), rigid[end2]._get_mass()
    rad1, rad2 = object[end1]._get_contact_radius(cpos), rigid[end2]._get_contact_radius(cpos)
    m_eff = effective_value(mass1, mass2)
    rad_eff = effective_value(rad1, rad2)

    normal_force, tangential_force, tang_over_temp = surface_props[material_id]._force_assemble(
        m_eff, rad_eff, min_dist, coeff, dgdx, v_rel, tang_overlap_old, dt
    )

    cplist[nc]._set_contact(normal_force, tangential_force, tang_over_temp)
    cplist[nc].normalOverlap = min_dist
    cplist[nc].normalOverlapActive = ti.u8(1)
    ftotal = normal_force + tangential_force
    momentum1 = ftotal.cross(mass_center1 - cpos)
    momentum2 = ftotal.cross(cpos - mass_center2)
    object[end1]._update_contact_interaction(ftotal, momentum1)
    rigid[end2]._update_contact_interaction(-ftotal, momentum2)


@ti.func
def l_sparticle_contact_model_type1(
    material_id,
    nc,
    coeff,
    end1,
    end2,
    contact_radius,
    min_dist,
    mass_center1,
    mass_center2,
    global_intruding_node,
    local_intruding_node,
    rotate_matrix2,
    object,
    grid,
    rigid,
    box,
    surface_props,
    cplist,
    dt,
):
    vel1, vel2 = object[end1]._get_velocity(), rigid[end2]._get_velocity()
    w1, w2 = object[end1]._get_angular_velocity(), rigid[end2]._get_angular_velocity()

    norm = rotate_matrix2 @ box[end2].calculate_normal(local_intruding_node, grid)
    cpos = global_intruding_node - 0.5 * norm * (min_dist + contact_radius)  # is right ??
    v_rel = vel1 + w1.cross(cpos - mass_center1) - (vel2 + w2.cross(cpos - mass_center2))
    w_rel = w1 - w2
    tang_overlap_old = cplist[nc].oldTangOverlap
    mass1, mass2 = object[end1]._get_mass(), rigid[end2]._get_mass()
    rad1, rad2 = object[end1]._get_contact_radius(cpos), rigid[end2]._get_contact_radius(cpos)
    m_eff = effective_value(mass1, mass2)
    rad_eff = effective_value(rad1, rad2)

    param = stiffness_parameter(rad1, rad2)

    normal_force, tangential_force, momentum, tang_over_temp = surface_props[material_id]._force_assemble(
        m_eff, rad_eff, min_dist, coeff, param, norm, v_rel, w_rel, tang_overlap_old, dt
    )

    cplist[nc]._set_contact(normal_force, tangential_force, tang_over_temp)
    cplist[nc].normalOverlap = min_dist
    cplist[nc].normalOverlapActive = ti.u8(1)
    ftotal = normal_force + tangential_force
    momentum1 = ftotal.cross(mass_center1 - cpos) + momentum
    momentum2 = ftotal.cross(cpos - mass_center2) - momentum
    object[end1]._update_contact_interaction(ftotal, momentum1)
    rigid[end2]._update_contact_interaction(-ftotal, momentum2)


@ti.func
def fluid_particle_contact_model(
    material_id, nc, end1, end2, gapn, norm, pos1, pos2, rad1, rad2, particle1, particle2, surface_props, cplist, dt
):
    mass1, mass2 = particle1[end1]._get_mass(), particle2[end2]._get_mass()
    vel1, vel2 = particle1[end1]._get_velocity(), particle2[end2]._get_velocity()
    w1, w2 = particle1[end1]._get_angular_velocity(), particle2[end2]._get_angular_velocity()
    m_eff = effective_value(mass1, mass2)

    cpos = pos2 + (rad2 - 0.5 * gapn) * norm
    v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))

    normal_force, tangential_force = surface_props[material_id]._fluid_force_assemble(m_eff, gapn, 1.0, norm, v_rel, dt)
    ftotal = normal_force + tangential_force
    resultant_momentum1 = tangential_force.cross(pos1 - cpos)
    resultant_momentum2 = tangential_force.cross(cpos - pos2)

    cplist[nc]._set_contact(normal_force, tangential_force, ZEROVEC3F)
    particle1[end1]._update_contact_interaction(ftotal, resultant_momentum1)
    particle2[end2]._update_contact_interaction(-ftotal, resultant_momentum2)


@ti.func
def fluid_l_sparticle_contact_model(
    material_id,
    nc,
    coeff,
    end1,
    end2,
    contact_radius,
    min_dist,
    mass_center1,
    mass_center2,
    global_intruding_node,
    local_intruding_node,
    rotate_matrix2,
    object,
    grid,
    rigid,
    box,
    surface_props,
    cplist,
    dt,
):
    vel1, vel2 = object[end1]._get_velocity(), rigid[end2]._get_velocity()
    w1, w2 = object[end1]._get_angular_velocity(), rigid[end2]._get_angular_velocity()

    norm = rotate_matrix2 @ box[end2].calculate_normal(local_intruding_node, grid)
    cpos = global_intruding_node - 0.5 * norm * (min_dist + contact_radius)  # is right ??
    mass1, mass2 = object[end1]._get_mass(), rigid[end2]._get_mass()
    m_eff = effective_value(mass1, mass2)
    v_rel = vel1 + w1.cross(cpos - mass_center1) - (vel2 + w2.cross(cpos - mass_center2))

    normal_force, tangential_force = surface_props[material_id]._fluid_force_assemble(
        m_eff, min_dist, coeff, norm, v_rel, dt
    )
    cplist[nc]._set_contact(normal_force, tangential_force, ZEROVEC3F)
    ftotal = normal_force + tangential_force
    momentum1 = ftotal.cross(mass_center1 - cpos)
    momentum2 = ftotal.cross(cpos - mass_center2)
    object[end1]._update_contact_interaction(ftotal, momentum1)
    rigid[end2]._update_contact_interaction(-ftotal, momentum2)


@ti.func
def wall_contact_model_type1(
    material_id, nc, end1, end2, gapn, fraction, pos1, particle_rad, particle, wall, surface_props, cplist, dt
):
    vel1, vel2 = particle[end1]._get_velocity(), wall[end2]._get_velocity()
    w1 = particle[end1]._get_angular_velocity()
    m_eff, rad_eff = particle[end1]._get_mass(), particle_rad + 0.5 * gapn
    tang_overlap_old = cplist[nc].oldTangOverlap

    norm = wall[end2]._get_norm(pos1)
    cpos = wall[end2]._point_projection(pos1) - 0.5 * gapn * norm
    v_rel = vel1 + w1.cross(cpos - pos1) - vel2
    w_rel = w1

    param = particle_rad
    normal_force, tangential_force, momentum, tang_over_temp = surface_props[material_id]._force_assemble(
        m_eff, rad_eff, gapn, 1.0, param, norm, v_rel, w_rel, tang_overlap_old, dt
    )

    ftotal = fraction * (normal_force + tangential_force)
    resultant_momentum = ftotal.cross(pos1 - cpos) + momentum
    cplist[nc]._set_contact(fraction * normal_force, fraction * tangential_force, tang_over_temp)
    particle[end1]._update_contact_interaction(fraction * ftotal, fraction * resultant_momentum)
    if ti.static(GlobalVariable.ENABLESHELL):
        wall[end2]._update_contact_interaction(-fraction * ftotal, -fraction * resultant_momentum)


@ti.func
def wall_contact_model_type2(
    material_id, nc, end1, end2, gapn, fraction, pos1, particle_rad, particle, wall, surface_props, cplist, dt
):
    vel1, vel2 = particle[end1]._get_velocity(), wall[end2]._get_velocity()
    w1 = particle[end1]._get_angular_velocity()
    m_eff = particle[end1]._get_mass()
    rad_eff = particle_rad

    norm = wall[end2]._get_norm(pos1)
    cpos = wall[end2]._point_projection(pos1) - 0.5 * gapn * norm
    v_rel = vel1 + w1.cross(cpos - pos1) - vel2
    w_rel = w1
    wr_rel = norm.cross(w1)

    tang_overlap_old = cplist[nc].oldTangOverlap
    tang_rolling_old = cplist[nc].oldRollAngle
    tang_twisting_old = cplist[nc].oldTwistAngle

    param = particle_rad

    normal_force, tangential_force, momentum, tang_over_temp, tang_rolling_temp, tang_twisting_temp = surface_props[
        material_id
    ]._force_assemble(
        m_eff,
        rad_eff,
        gapn,
        1.0,
        param,
        norm,
        v_rel,
        w_rel,
        wr_rel,
        tang_overlap_old,
        tang_rolling_old,
        tang_twisting_old,
        dt,
    )
    ftotal = normal_force + tangential_force
    resultant_momentum = ftotal.cross(pos1 - cpos) + momentum

    cplist[nc]._set_contact(
        fraction * normal_force, fraction * tangential_force, tang_over_temp, tang_rolling_temp, tang_twisting_temp
    )
    particle[end1]._update_contact_interaction(fraction * ftotal, fraction * resultant_momentum)
    if ti.static(GlobalVariable.ENABLESHELL):
        wall[end2]._update_contact_interaction(-fraction * ftotal, -fraction * resultant_momentum)


@ti.func
def i_sparticle_wall_contact_model_type1(
    material_id,
    nc,
    end1,
    end2,
    fraction,
    gapn,
    norm,
    mass_center,
    support_point,
    projection_point,
    rigid,
    wall,
    surface_props,
    cplist,
    dt,
):
    vel1, vel2 = rigid[end1]._get_velocity(), wall[end2]._get_velocity()
    w1 = rigid[end1]._get_angular_velocity()
    m_eff, rad_eff = rigid[end1]._get_mass(), rigid[end1]._get_radius()
    tang_overlap_old = cplist[nc].oldTangOverlap

    cpos = 0.5 * (support_point + projection_point)
    v_rel = vel1 + w1.cross(cpos - mass_center) - vel2
    w_rel = w1

    param = rad_eff

    normal_force, tangential_force, momentum, tang_over_temp = surface_props[material_id]._force_assemble(
        m_eff, rad_eff, gapn, 1.0, param, norm, v_rel, w_rel, tang_overlap_old, dt
    )

    ftotal = normal_force + tangential_force
    resultant_momentum = ftotal.cross(mass_center - cpos) + momentum
    cplist[nc]._set_contact(fraction * normal_force, fraction * tangential_force, tang_over_temp)
    rigid[end1]._update_contact_interaction(fraction * ftotal, fraction * resultant_momentum)
    if ti.static(GlobalVariable.ENABLESHELL):
        wall[end2]._update_contact_interaction(-fraction * ftotal, -fraction * resultant_momentum)


@ti.func
def i_sparticle_wall_contact_model_type2(
    material_id,
    nc,
    end1,
    end2,
    fraction,
    gapn,
    norm,
    mass_center,
    support_point,
    projection_point,
    rigid,
    wall,
    surface_props,
    cplist,
    dt,
):
    vel1, vel2 = rigid[end1]._get_velocity(), wall[end2]._get_velocity()
    w1 = rigid[end1]._get_angular_velocity()
    m_eff, rad_eff = rigid[end1]._get_mass(), rigid[end1]._get_radius()
    tang_overlap_old = cplist[nc].oldTangOverlap
    tang_rolling_old = cplist[nc].oldRollAngle
    tang_twisting_old = cplist[nc].oldTwistAngle

    cpos = 0.5 * (support_point + projection_point)
    v_rel = vel1 + w1.cross(cpos - mass_center) - vel2
    w_rel = w1
    wr_rel = norm.cross(w1)

    param = rad_eff

    normal_force, tangential_force, momentum, tang_over_temp, tang_rolling_temp, tang_twisting_temp = surface_props[
        material_id
    ]._force_assemble(
        m_eff,
        rad_eff,
        gapn,
        1.0,
        param,
        norm,
        v_rel,
        w_rel,
        wr_rel,
        tang_overlap_old,
        tang_rolling_old,
        tang_twisting_old,
        dt,
    )
    ftotal = normal_force + tangential_force
    resultant_momentum = ftotal.cross(mass_center - cpos) + momentum

    cplist[nc]._set_contact(
        fraction * normal_force, fraction * tangential_force, tang_over_temp, tang_rolling_temp, tang_twisting_temp
    )
    rigid[end1]._update_contact_interaction(fraction * ftotal, fraction * resultant_momentum)
    if ti.static(GlobalVariable.ENABLESHELL):
        wall[end2]._update_contact_interaction(-fraction * ftotal, -fraction * resultant_momentum)


@ti.func
def fluid_wall_contact_model(
    material_id, nc, end1, end2, gapn, fraction, pos1, particle_rad, particle, wall, surface_props, cplist, dt
):
    vel1, vel2 = particle[end1]._get_velocity(), wall[end2]._get_velocity()
    w1 = particle[end1]._get_angular_velocity()
    m_eff, rad_eff = particle[end1]._get_mass(), particle_rad + 0.5 * gapn

    norm = wall[end2]._get_norm(pos1)
    cpos = wall[end2]._point_projection(pos1) - 0.5 * gapn * norm
    v_rel = vel1 + w1.cross(cpos - pos1) - vel2

    normal_force, tangential_force = surface_props[material_id]._fluid_force_assemble(m_eff, gapn, 1.0, norm, v_rel, dt)
    ftotal = fraction * (normal_force + tangential_force)
    resultant_momentum = ftotal.cross(pos1 - cpos)

    cplist[nc]._set_contact(fraction * normal_force, fraction * tangential_force, ZEROVEC3F)
    particle[end1]._update_contact_interaction(ftotal, resultant_momentum)
    if ti.static(GlobalVariable.ENABLESHELL):
        wall[end2]._update_contact_interaction(-ftotal, -resultant_momentum)


@ti.func
def l_sparticle_wall_contact_model_type0(
    material_id, nc, coeff, end1, end2, min_dist, mass_center1, intruding_node, rigid, wall, surface_props, cplist, dt
):
    """LSDEM facet-wall adapter for energy/barrier potential properties."""

    vel1, vel2 = rigid[end1]._get_velocity(), wall[end2]._get_velocity()
    w1 = rigid[end1]._get_angular_velocity()
    dgdx = wall[end2]._get_norm(mass_center1)
    # The signed gap is evaluated at this LSDEM surface node.  Apply the
    # interaction and evaluate its relative velocity at that same point so
    # force/torque power is conjugate to the gap rate for a rotating body.
    cpos = intruding_node
    v_rel = vel1 + w1.cross(cpos - mass_center1) - vel2
    tang_overlap_old = cplist[nc].oldTangOverlap
    m_eff = rigid[end1]._get_mass()
    rad_eff = rigid[end1]._get_contact_radius(cpos)

    normal_force, tangential_force, tang_over_temp = surface_props[material_id]._force_assemble(
        m_eff,
        rad_eff,
        min_dist,
        coeff,
        dgdx,
        v_rel,
        tang_overlap_old,
        dt,
    )

    ftotal = normal_force + tangential_force
    resultant_momentum = ftotal.cross(mass_center1 - cpos)
    cplist[nc]._set_contact(normal_force, tangential_force, tang_over_temp)
    cplist[nc].normalOverlap = min_dist
    cplist[nc].normalOverlapActive = ti.u8(1)
    rigid[end1]._update_contact_interaction(ftotal, resultant_momentum)
    if ti.static(GlobalVariable.ENABLESHELL):
        wall[end2]._update_contact_interaction(-ftotal, -resultant_momentum)


@ti.func
def l_sparticle_wall_contact_model(
    material_id, nc, coeff, end1, end2, min_dist, mass_center1, intruding_node, rigid, wall, surface_props, cplist, dt
):
    vel1, vel2 = rigid[end1]._get_velocity(), wall[end2]._get_velocity()
    w1 = rigid[end1]._get_angular_velocity()
    norm = wall[end2]._get_norm(mass_center1)
    # Use the same surface point for signed gap, velocity, force, and torque.
    cpos = intruding_node
    v_rel = vel1 + w1.cross(cpos - mass_center1) - vel2
    w_rel = w1
    tang_overlap_old = cplist[nc].oldTangOverlap
    m_eff, rad_eff = rigid[end1]._get_mass(), rigid[end1]._get_contact_radius(cpos)

    param = rad_eff

    normal_force, tangential_force, momentum, tang_over_temp = surface_props[material_id]._force_assemble(
        m_eff, rad_eff, min_dist, coeff, param, norm, v_rel, w_rel, tang_overlap_old, dt
    )

    ftotal = normal_force + tangential_force
    resultant_momentum = ftotal.cross(mass_center1 - cpos) + momentum
    cplist[nc]._set_contact(normal_force, tangential_force, tang_over_temp)
    cplist[nc].normalOverlap = min_dist
    cplist[nc].normalOverlapActive = ti.u8(1)
    rigid[end1]._update_contact_interaction(ftotal, resultant_momentum)
    if ti.static(GlobalVariable.ENABLESHELL):
        wall[end2]._update_contact_interaction(-ftotal, -resultant_momentum)
