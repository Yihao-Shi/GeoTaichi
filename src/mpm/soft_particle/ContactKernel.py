import taichi as ti

from src.dem.contact.ContactKernel import (
    lsmpm_surface_quadrature_coefficient,
    stiffness_parameter,
)
from src.utils.constants import ZEROVEC3f, Threshold
from src.utils.Quaternion import SetToRotate
from src.utils.ScalarFunction import PairingMapping, EffectiveValue, linearize3D
from src.utils import GlobalVariable


@ti.func
def soft_soft_work_conjugate_penetration_state_(
    stored_penetration,
    stored_active,
    gap_rate,
    dt,
):
    """Advance the normal state without injecting energy at activation.

    An inactive query may already see a finite negative transported-SDF gap.
    That geometry detects closure but is not material-trace work, so a new or
    reactivated spring starts from zero stored penetration.  Its first energy
    increment is generated only by the closing trace displacement in this
    step.  Existing active histories retain their accumulated state.
    """
    penetration = 0.0
    assemble_contact = True
    if int(stored_active) == 1:
        penetration = ti.max(stored_penetration, 0.0)
    elif gap_rate > 0.0:
        # A lagging transported SDF may remain negative during separation.
        assemble_contact = False
    next_penetration = ti.max(penetration - gap_rate * dt, 0.0)
    return penetration, next_penetration, assemble_contact


@ti.func
def soft_surface_node_mass_(bodyID, rigid):
    node_count = ti.max(rigid[bodyID]._get_vertice_number(), 1)
    return rigid[bodyID].m / node_count


@ti.kernel
def reset_soft_contact_trace_(
    soft_num: int,
    max_grid_num: int,
    soft: ti.template(),
    trace_velocity: ti.template(),
    trace_weight: ti.template(),
    trace_force: ti.template(),
    uncovered: ti.template(),
):
    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            node = soft[sb].gridStart + local_grid
            trace_velocity[node] = ZEROVEC3f
            trace_weight[node] = 0.0
            trace_force[node] = ZEROVEC3f


@ti.kernel
def project_current_soft_surface_to_contact_trace_(
    surface_num: int,
    surface: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    soft: ti.template(),
    vertice: ti.template(),
    trace_velocity: ti.template(),
    trace_weight: ti.template(),
):
    for ns in range(surface_num):
        body_id = surface[ns]
        if int(rigid[body_id].is_soft) == 1:
            soft_id = rigid[body_id].softID
            local_node = rigid[body_id].global_node_to_local(ns)
            position = box[body_id].scale * vertice[local_node].x
            area_weight = ti.max(vertice[local_node].parameter, 0.0)
            inv_dx = 1.0 / box[body_id].grid_space
            base = ti.Vector([0, 0, 0])
            for d in ti.static(range(3)):
                base[d] = ti.min(
                    box[body_id].gnum[d] - 2,
                    ti.max(
                        0,
                        int(ti.floor((position[d] - box[body_id].xmin[d]) * inv_dx)),
                    ),
                )
            origin = box[body_id].xmin + ti.cast(base, float) * box[body_id].grid_space
            fraction = (position - origin) * inv_dx
            for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
                weight = (
                    ((1.0 - fraction[0]) if i == 0 else fraction[0])
                    * ((1.0 - fraction[1]) if j == 0 else fraction[1])
                    * ((1.0 - fraction[2]) if k == 0 else fraction[2])
                    * area_weight
                )
                local_sdf_node = linearize3D(
                    base[0] + i,
                    base[1] + j,
                    base[2] + k,
                    box[body_id].gnum,
                )
                sdf_node = soft[soft_id].gridStart + local_sdf_node
                ti.atomic_add(trace_weight[sdf_node], weight)
                ti.atomic_add(
                    trace_velocity[sdf_node],
                    weight * vertice[local_node].v,
                )


@ti.kernel
def normalize_soft_contact_trace_(
    soft_num: int,
    max_grid_num: int,
    soft: ti.template(),
    trace_velocity: ti.template(),
    trace_weight: ti.template(),
):
    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            node = soft[sb].gridStart + local_grid
            if trace_weight[node] > Threshold:
                trace_velocity[node] /= trace_weight[node]
            else:
                trace_velocity[node] = ZEROVEC3f


@ti.func
def soft_grid_trace_velocity_(
    bodyID,
    local_position,
    rigid,
    box,
    soft,
    trace_velocity,
    trace_weight,
    uncovered,
):
    soft_id = rigid[bodyID].softID
    inv_dx = 1.0 / box[bodyID].grid_space
    base = ti.Vector([0, 0, 0])
    for d in ti.static(range(3)):
        base[d] = ti.min(
            box[bodyID].gnum[d] - 2,
            ti.max(
                0,
                int((local_position[d] - box[bodyID].xmin[d]) * inv_dx),
            ),
        )
    origin = box[bodyID].xmin + ti.cast(base, float) * box[bodyID].grid_space
    fraction = (local_position - origin) * inv_dx
    velocity = ZEROVEC3f
    active_weight = 0.0
    for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
        weight = (
            ((1.0 - fraction[0]) if i == 0 else fraction[0])
            * ((1.0 - fraction[1]) if j == 0 else fraction[1])
            * ((1.0 - fraction[2]) if k == 0 else fraction[2])
        )
        local_sdf_node = linearize3D(
            base[0] + i,
            base[1] + j,
            base[2] + k,
            box[bodyID].gnum,
        )
        sdf_node = soft[soft_id].gridStart + local_sdf_node
        nodal_weight = weight * trace_weight[sdf_node]
        active_weight += nodal_weight
        velocity += nodal_weight * trace_velocity[sdf_node]
    if active_weight > Threshold:
        velocity /= active_weight
    else:
        ti.atomic_add(uncovered[None], 1)
        velocity = rigid[bodyID].v + rigid[bodyID].w.cross(
            rigid[bodyID].mass_center + SetToRotate(rigid[bodyID].q) @ local_position - rigid[bodyID].mass_center
        )
    return velocity


@ti.func
def transfer_force_to_soft_grid_(
    bodyID,
    local_position,
    force,
    rigid,
    box,
    soft,
    contact_weight,
    trace_force,
):
    soft_id = rigid[bodyID].softID
    ti.atomic_add(soft[soft_id].contact_force, force)
    inv_dx = 1.0 / box[bodyID].grid_space
    base = ti.Vector([0, 0, 0])
    for d in ti.static(range(3)):
        base[d] = ti.min(
            box[bodyID].gnum[d] - 2,
            ti.max(
                0,
                int((local_position[d] - box[bodyID].xmin[d]) * inv_dx),
            ),
        )
    origin = box[bodyID].xmin + ti.cast(base, float) * box[bodyID].grid_space
    fraction = (local_position - origin) * inv_dx
    active_weight = 0.0
    for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
        interpolation_weight = (
            ((1.0 - fraction[0]) if i == 0 else fraction[0])
            * ((1.0 - fraction[1]) if j == 0 else fraction[1])
            * ((1.0 - fraction[2]) if k == 0 else fraction[2])
        )
        local_sdf_node = linearize3D(
            base[0] + i,
            base[1] + j,
            base[2] + k,
            box[bodyID].gnum,
        )
        sdf_node = soft[soft_id].gridStart + local_sdf_node
        active_weight += interpolation_weight * contact_weight[sdf_node]
    if active_weight > Threshold:
        for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
            interpolation_weight = (
                ((1.0 - fraction[0]) if i == 0 else fraction[0])
                * ((1.0 - fraction[1]) if j == 0 else fraction[1])
                * ((1.0 - fraction[2]) if k == 0 else fraction[2])
            )
            local_sdf_node = linearize3D(
                base[0] + i,
                base[1] + j,
                base[2] + k,
                box[bodyID].gnum,
            )
            sdf_node = soft[soft_id].gridStart + local_sdf_node
            ti.atomic_add(
                trace_force[sdf_node],
                interpolation_weight / active_weight * force,
            )


@ti.kernel
def gather_soft_contact_trace_force_(
    surface_num: int,
    surface: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    soft: ti.template(),
    vertice: ti.template(),
    trace_force: ti.template(),
):
    for ns in range(surface_num):
        body_id = surface[ns]
        if int(rigid[body_id].is_soft) == 1:
            soft_id = rigid[body_id].softID
            local_node = rigid[body_id].global_node_to_local(ns)
            position = box[body_id].scale * vertice[local_node].x
            area_weight = ti.max(vertice[local_node].parameter, 0.0)
            inv_dx = 1.0 / box[body_id].grid_space
            base = ti.Vector([0, 0, 0])
            for d in ti.static(range(3)):
                base[d] = ti.min(
                    box[body_id].gnum[d] - 2,
                    ti.max(
                        0,
                        int(ti.floor((position[d] - box[body_id].xmin[d]) * inv_dx)),
                    ),
                )
            origin = box[body_id].xmin + ti.cast(base, float) * box[body_id].grid_space
            fraction = (position - origin) * inv_dx
            force = ZEROVEC3f
            for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
                weight = (
                    ((1.0 - fraction[0]) if i == 0 else fraction[0])
                    * ((1.0 - fraction[1]) if j == 0 else fraction[1])
                    * ((1.0 - fraction[2]) if k == 0 else fraction[2])
                    * area_weight
                )
                local_sdf_node = linearize3D(
                    base[0] + i,
                    base[1] + j,
                    base[2] + k,
                    box[body_id].gnum,
                )
                sdf_node = soft[soft_id].gridStart + local_sdf_node
                force += weight * trace_force[sdf_node]
            if force.dot(force) > Threshold * Threshold:
                vertice[local_node]._update_contact_interaction(force, ZEROVEC3f)


@ti.kernel
def kernel_LSMPM_LSparticle_LSparticle_force_assemble_(
    contactNodeNum: int,
    dt: ti.template(),
    max_material_num: int,
    surfaceProps: ti.template(),
    rigid: ti.template(),
    grid: ti.template(),
    vertice: ti.template(),
    box: ti.template(),
    soft: ti.template(),
    trace_velocity: ti.template(),
    trace_weight: ti.template(),
    trace_force: ti.template(),
    trace_uncovered: ti.template(),
    ls_contact_body: ti.template(),
    ls_contact_kind: ti.template(),
    ls_contact_ref: ti.template(),
    cplist: ti.template(),
    particle_particle: ti.template(),
    model_type: ti.template(),
):
    total_contact_num = particle_particle[contactNodeNum]
    for nc in range(total_contact_num):
        contact_node, end2 = cplist[nc].endID1, cplist[nc].endID2
        end1 = ls_contact_body[contact_node]
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

        rotate_matrix1, rotate_matrix2 = SetToRotate(rigid[end1].q), SetToRotate(rigid[end2].q)
        matID1, matID2 = rigid[end1].materialID, rigid[end2].materialID
        materialID = PairingMapping(matID1, matID2, max_material_num)

        intruding_node = ZEROVEC3f
        intruding_velocity = ZEROVEC3f
        coeff = 1.0
        contact_radius = 0.0
        master_is_soft = int(ls_contact_kind[contact_node]) == 1
        global_node = ls_contact_ref[contact_node]
        master_local_node = rigid[end1].global_node_to_local(global_node)

        # For a mixed pair, integrate only soft trace nodes against the rigid
        # SDF. The reverse query differentiates an advected soft SDF and is not
        # a work-conjugate duplicate of the deformable-trace contribution.
        if not master_is_soft and int(rigid[end2].is_soft) == 1:
            cplist[nc]._no_contact()
            continue

        intruding_node = mass_center1 + rotate_matrix1 @ (box[end1].scale * vertice[master_local_node].x)
        coeff = lsmpm_surface_quadrature_coefficient(end1, master_local_node, rigid, box, vertice)
        if master_is_soft:
            intruding_velocity = vertice[master_local_node].v
        else:
            intruding_velocity = rigid[end1].v + rigid[end1].w.cross(intruding_node - mass_center1)

        # Soft--soft pairs are searched in both directions. Each pass carries
        # half of the area quadrature to avoid duplicate integration.
        if master_is_soft and int(rigid[end2].is_soft) == 1:
            coeff *= 0.5

        local_intruding_node = rotate_matrix2.transpose() @ (intruding_node - mass_center2)
        if not box[end2]._in_box(local_intruding_node):
            cplist[nc]._no_contact()
            continue
        gapn = box[end2].distance(local_intruding_node, grid) - contact_radius
        retained_work_conjugate_contact = False
        if ti.static(GlobalVariable.LSMPM_SOFT_SOFT_WORK_CONJUGATE and model_type == 0):
            if master_is_soft and int(rigid[end2].is_soft) == 1:
                retained_work_conjugate_contact = int(cplist[nc].normalOverlapActive) == 1
        if gapn < surfaceProps[materialID].ncut or retained_work_conjugate_contact:
            dgdx = ZEROVEC3f
            norm = ZEROVEC3f
            if ti.static(model_type == 0):
                dgdx = rotate_matrix2 @ box[end2].calculate_gradient(local_intruding_node, grid)
                norm = dgdx.normalized(Threshold)
            else:
                norm = rotate_matrix2 @ box[end2].calculate_normal(local_intruding_node, grid)
            cpos = intruding_node - 0.5 * norm * (gapn + contact_radius)
            slave_velocity = rigid[end2].v + rigid[end2].w.cross(cpos - mass_center2)
            slave_projection_local = local_intruding_node - gapn * (rotate_matrix2.transpose() @ norm)
            if int(rigid[end2].is_soft) == 1:
                slave_velocity = soft_grid_trace_velocity_(
                    end2,
                    slave_projection_local,
                    rigid,
                    box,
                    soft,
                    trace_velocity,
                    trace_weight,
                    trace_uncovered,
                )

            v_rel = intruding_velocity - slave_velocity
            w_rel = ZEROVEC3f
            if not master_is_soft:
                w_rel += rigid[end1].w
            if int(rigid[end2].is_soft) == 0:
                w_rel -= rigid[end2].w
            tangOverlapOld = cplist[nc].oldTangOverlap
            mass1 = rigid[end1].m
            rad1 = rigid[end1]._get_contact_radius(cpos)
            if master_is_soft:
                mass1 = soft_surface_node_mass_(end1, rigid)
                rad1 = rigid[end1].equi_r
            mass2 = rigid[end2].m
            rad2 = rigid[end2]._get_contact_radius(cpos)
            if int(rigid[end2].is_soft) == 1:
                mass2 = soft_surface_node_mass_(end2, rigid)
                rad2 = rigid[end2].equi_r
            m_eff = EffectiveValue(mass1, mass2)
            rad_eff = EffectiveValue(rad1, rad2)
            param = stiffness_parameter(rad1, rad2)
            assemble_contact = True
            penetration = 0.0
            next_penetration = 0.0
            use_work_conjugate_contact = False
            if ti.static(GlobalVariable.LSMPM_SOFT_SOFT_WORK_CONJUGATE and model_type == 0):
                if master_is_soft and int(rigid[end2].is_soft) == 1:
                    use_work_conjugate_contact = True
                    penetration, next_penetration, assemble_contact = soft_soft_work_conjugate_penetration_state_(
                        cplist[nc].normalOverlap,
                        cplist[nc].normalOverlapActive,
                        v_rel.dot(dgdx),
                        dt[None],
                    )

            if assemble_contact:
                normal_force = ZEROVEC3f
                tangential_force = ZEROVEC3f
                momentum = ZEROVEC3f
                tangOverTemp = ZEROVEC3f
                if ti.static(model_type == 0):
                    if ti.static(GlobalVariable.LSMPM_SOFT_SOFT_WORK_CONJUGATE):
                        if use_work_conjugate_contact:
                            normal_force, tangential_force, tangOverTemp = surfaceProps[
                                materialID
                            ]._force_assemble_work_conjugate(
                                m_eff, rad_eff, penetration, next_penetration, coeff, dgdx, v_rel, tangOverlapOld, dt
                            )
                        else:
                            normal_force, tangential_force, tangOverTemp = surfaceProps[materialID]._force_assemble(
                                m_eff, rad_eff, gapn, coeff, dgdx, v_rel, tangOverlapOld, dt
                            )
                    else:
                        normal_force, tangential_force, tangOverTemp = surfaceProps[materialID]._force_assemble(
                            m_eff, rad_eff, gapn, coeff, dgdx, v_rel, tangOverlapOld, dt
                        )
                else:
                    normal_force, tangential_force, momentum, tangOverTemp = surfaceProps[materialID]._force_assemble(
                        m_eff, rad_eff, gapn, coeff, param, norm, v_rel, w_rel, tangOverlapOld, dt
                    )

                cplist[nc]._set_contact(normal_force, tangential_force, tangOverTemp)
                if use_work_conjugate_contact:
                    cplist[nc].normalOverlap = next_penetration
                    cplist[nc].normalOverlapActive = ti.u8(next_penetration > Threshold)
                Ftotal = normal_force + tangential_force
                if master_is_soft:
                    vertice[master_local_node]._update_contact_interaction(Ftotal, ZEROVEC3f)
                else:
                    momentum1 = Ftotal.cross(mass_center1 - cpos) + momentum
                    rigid[end1]._update_contact_interaction(Ftotal, momentum1)
                if int(rigid[end2].is_soft) == 1:
                    transfer_force_to_soft_grid_(
                        end2,
                        slave_projection_local,
                        -Ftotal,
                        rigid,
                        box,
                        soft,
                        trace_weight,
                        trace_force,
                    )
                else:
                    momentum2 = Ftotal.cross(cpos - mass_center2) - momentum
                    rigid[end2]._update_contact_interaction(-Ftotal, momentum2)
            else:
                cplist[nc]._no_contact()
        else:
            cplist[nc]._no_contact()


@ti.func
def assemble_lsmpm_wall_contact_force_(
    surface_properties: ti.template(),
    material_id,
    model_type: ti.template(),
    mass,
    radius,
    gap,
    coefficient,
    parameter,
    normal,
    relative_velocity,
    relative_angular_velocity,
    old_tangential_overlap,
    dt,
):
    normal_force = ZEROVEC3f
    tangential_force = ZEROVEC3f
    momentum = ZEROVEC3f
    tangential_overlap = ZEROVEC3f
    if ti.static(model_type == 0):
        normal_force, tangential_force, tangential_overlap = surface_properties[material_id]._force_assemble(
            mass, radius, gap, coefficient, normal, relative_velocity, old_tangential_overlap, dt
        )
    else:
        normal_force, tangential_force, momentum, tangential_overlap = surface_properties[material_id]._force_assemble(
            mass,
            radius,
            gap,
            coefficient,
            parameter,
            normal,
            relative_velocity,
            relative_angular_velocity,
            old_tangential_overlap,
            dt,
        )
    return normal_force, tangential_force, momentum, tangential_overlap


@ti.kernel
def kernel_LSMPM_LSparticle_wall_force_assemble_(
    contactNodeNum: int,
    dt: ti.template(),
    max_material_num: int,
    surfaceProps: ti.template(),
    rigid: ti.template(),
    vertice: ti.template(),
    box: ti.template(),
    wall: ti.template(),
    soft: ti.template(),
    soft_point: ti.template(),
    soft_surface_point_id: ti.template(),
    ls_contact_body: ti.template(),
    ls_contact_kind: ti.template(),
    ls_contact_ref: ti.template(),
    cplist: ti.template(),
    particle_wall: ti.template(),
    model_type: ti.template(),
):
    total_contact_num = particle_wall[contactNodeNum]
    for nc in range(total_contact_num):
        contact_node, end2 = cplist[nc].endID1, cplist[nc].endID2
        end1 = ls_contact_body[contact_node]
        mass_center1 = rigid[end1]._get_position()
        rotate_matrix1 = SetToRotate(rigid[end1].q)
        matID1, matID2 = rigid[end1].materialID, wall[end2].materialID
        materialID = PairingMapping(matID1, matID2, max_material_num)

        intruding_node = ZEROVEC3f
        intruding_velocity = ZEROVEC3f
        coeff = 1.0
        contact_radius = 0.0
        master_is_soft = int(ls_contact_kind[contact_node]) == 1
        global_node = ls_contact_ref[contact_node]
        master_local_node = rigid[end1].global_node_to_local(global_node)
        intruding_node = mass_center1 + rotate_matrix1 @ (box[end1].scale * vertice[master_local_node].x)
        coeff = lsmpm_surface_quadrature_coefficient(end1, master_local_node, rigid, box, vertice)
        if master_is_soft:
            intruding_velocity = vertice[master_local_node].v
        else:
            intruding_velocity = rigid[end1].v + rigid[end1].w.cross(intruding_node - mass_center1)

        gapn = wall[end2]._get_norm_distance(intruding_node) - contact_radius
        if gapn < surfaceProps[materialID].ncut:
            norm = wall[end2]._get_norm(mass_center1)
            cpos = intruding_node + 0.5 * norm * (gapn + contact_radius)
            v_rel = intruding_velocity - wall[end2]._get_velocity()
            w_rel = ZEROVEC3f if master_is_soft else rigid[end1].w
            tangOverlapOld = cplist[nc].oldTangOverlap
            mass1 = rigid[end1].m
            rad_eff = rigid[end1]._get_contact_radius(cpos)
            if master_is_soft:
                mass1 = soft_surface_node_mass_(end1, rigid)
                rad_eff = rigid[end1].equi_r
            param = rad_eff
            normal_force, tangential_force, momentum, tangOverTemp = assemble_lsmpm_wall_contact_force_(
                surfaceProps,
                materialID,
                model_type,
                mass1,
                rad_eff,
                gapn,
                coeff,
                param,
                norm,
                v_rel,
                w_rel,
                tangOverlapOld,
                dt,
            )

            Ftotal = normal_force + tangential_force
            cplist[nc]._set_contact(normal_force, tangential_force, tangOverTemp)
            if master_is_soft:
                vertice[master_local_node]._update_contact_interaction(Ftotal, ZEROVEC3f)
            else:
                resultant_momentum = Ftotal.cross(mass_center1 - cpos) + momentum
                rigid[end1]._update_contact_interaction(Ftotal, resultant_momentum)
                if ti.static(GlobalVariable.ENABLESHELL):
                    wall[end2]._update_contact_interaction(-Ftotal, -resultant_momentum)
        else:
            cplist[nc]._no_contact()
