import taichi as ti

from src.utils.GeometryFunction import intersectionOBBs
from src.utils.Quaternion import SetToRotate
from src.utils.ScalarFunction import sgn
from src.utils.TypeDefination import vec3f
import src.utils.GlobalVariable as GlobalVariable


@ti.kernel
def rebuild_lsmpm_contact_nodes_(
    particleNum: int,
    rigid: ti.template(),
    soft: ti.template(),
    soft_surface_point_id: ti.template(),
    ls_contact_body: ti.template(),
    ls_contact_kind: ti.template(),
    ls_contact_ref: ti.template(),
    ls_contact_body_start: ti.template(),
    ls_contact_body_end: ti.template(),
    ls_contact_count: ti.template(),
):
    count = 0
    ti.loop_config(serialize=True)
    for body in range(particleNum):
        ls_contact_body_start[body] = count
        for node in range(rigid[body]._start_node(), rigid[body]._end_node()):
            global_node = rigid[body].local_node_to_global(node)
            ls_contact_body[count] = body
            ls_contact_kind[count] = ti.u8(rigid[body].is_soft)
            ls_contact_ref[count] = global_node
            count += 1
        ls_contact_body_end[body] = count
    ls_contact_count[None] = count


@ti.func
def append_lsmpm_directed_lsparticle_contacts_(
    master,
    slave,
    potential_point_num,
    verlet_distance,
    potential_list_point_particle,
    point_particle,
    rigid,
    box,
    vertice,
    grid,
    soft_point,
    ls_contact_kind,
    ls_contact_ref,
    ls_contact_body_start,
    ls_contact_body_end,
    rotate_matrix_master,
    rotate_matrix_slave,
    mass_center_master,
    mass_center_slave,
):
    for contact_node in range(ls_contact_body_start[master], ls_contact_body_end[master]):
        global_node = ls_contact_ref[contact_node]
        local_node = rigid[master].global_node_to_local(global_node)
        position = mass_center_master + rotate_matrix_master @ (box[master].scale * vertice[local_node].x)
        surface_node = rotate_matrix_slave.transpose() @ (position - mass_center_slave)
        if not box[slave]._in_box(surface_node):
            continue
        if box[slave].distance(surface_node, grid) < verlet_distance:
            sques = ti.atomic_add(point_particle[contact_node + 1], 1)
            potential_list_point_particle[sques + contact_node * potential_point_num] = slave
            assert (
                sques < potential_point_num
            ), f"Keyword:: /point_coordination_numbers[0]/ is too small, LSMPM contact node {contact_node} has {sques+1} potential contact number"


@ti.kernel
def board_search_lsmpm_lsparticle_lsparticle_linked_cell_(
    particleNum: int,
    potential_point_num: int,
    verlet_distance: float,
    pplist: ti.template(),
    potential_list_point_particle: ti.template(),
    particle_particle: ti.template(),
    point_particle: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    vertice: ti.template(),
    grid: ti.template(),
    soft_point: ti.template(),
    ls_contact_kind: ti.template(),
    ls_contact_ref: ti.template(),
    ls_contact_body_start: ti.template(),
    ls_contact_body_end: ti.template(),
    ls_contact_count: ti.template(),
):
    point_particle.fill(0)
    total_contact = particle_particle[particleNum]
    for nc in range(total_contact):
        master, slave = pplist[nc].endID1, pplist[nc].endID2
        rotate_matrix1, rotate_matrix2 = SetToRotate(rigid[master].q), SetToRotate(rigid[slave].q)
        mass_center1, mass_center2 = rigid[master]._get_position(), rigid[slave]._get_position()
        aabb1, aabb2 = box[master]._get_shape_center(), box[slave]._get_shape_center()

        if ti.static(GlobalVariable.DEMXPBC):
            if ti.abs(mass_center2[0] - mass_center1[0]) > 0.5 * GlobalVariable.DEMXSIZE:
                mass_center2[0] -= sgn(mass_center2[0] - mass_center1[0]) * GlobalVariable.DEMXSIZE
        if ti.static(GlobalVariable.DEMYPBC):
            if ti.abs(mass_center2[1] - mass_center1[1]) > 0.5 * GlobalVariable.DEMYSIZE:
                mass_center2[1] -= sgn(mass_center2[1] - mass_center1[1]) * GlobalVariable.DEMYSIZE
        if ti.static(GlobalVariable.DEMZPBC):
            if ti.abs(mass_center2[2] - mass_center1[2]) > 0.5 * GlobalVariable.DEMZSIZE:
                mass_center2[2] -= sgn(mass_center2[2] - mass_center1[2]) * GlobalVariable.DEMZSIZE

        obb1 = mass_center1 + rotate_matrix1 @ aabb1
        obb2 = mass_center2 + rotate_matrix2 @ aabb2
        extent1 = box[master]._get_shape_dim() + 2.0 * verlet_distance * vec3f(1.0, 1.0, 1.0)
        extent2 = box[slave]._get_shape_dim() + 2.0 * verlet_distance * vec3f(1.0, 1.0, 1.0)
        if intersectionOBBs(obb1, obb2, extent1, extent2, rotate_matrix1, rotate_matrix2):
            master_soft = int(rigid[master].is_soft)
            slave_soft = int(rigid[slave].is_soft)
            # RR uses the broad-phase direction, RS always uses the soft trace
            # against the rigid SDF, and SS retains both half-weighted passes.
            if master_soft == 1 or slave_soft == 0:
                append_lsmpm_directed_lsparticle_contacts_(
                    master,
                    slave,
                    potential_point_num,
                    verlet_distance,
                    potential_list_point_particle,
                    point_particle,
                    rigid,
                    box,
                    vertice,
                    grid,
                    soft_point,
                    ls_contact_kind,
                    ls_contact_ref,
                    ls_contact_body_start,
                    ls_contact_body_end,
                    rotate_matrix1,
                    rotate_matrix2,
                    mass_center1,
                    mass_center2,
                )
            if slave_soft == 1:
                append_lsmpm_directed_lsparticle_contacts_(
                    slave,
                    master,
                    potential_point_num,
                    verlet_distance,
                    potential_list_point_particle,
                    point_particle,
                    rigid,
                    box,
                    vertice,
                    grid,
                    soft_point,
                    ls_contact_kind,
                    ls_contact_ref,
                    ls_contact_body_start,
                    ls_contact_body_end,
                    rotate_matrix2,
                    rotate_matrix1,
                    mass_center2,
                    mass_center1,
                )


@ti.kernel
def board_search_lsmpm_lsparticle_wall_linked_cell_(
    particleNum: int,
    potential_point_num: int,
    verlet_distance: float,
    pwlist: ti.template(),
    potential_list_point_wall: ti.template(),
    particle_wall: ti.template(),
    point_wall: ti.template(),
    wall: ti.template(),
    rigid: ti.template(),
    vertice: ti.template(),
    box: ti.template(),
    soft_point: ti.template(),
    ls_contact_kind: ti.template(),
    ls_contact_ref: ti.template(),
    ls_contact_body_start: ti.template(),
    ls_contact_body_end: ti.template(),
):
    point_wall.fill(0)
    total_contact = particle_wall[particleNum]
    for nc in range(total_contact):
        master, wall_id = pwlist[nc].endID1, pwlist[nc].endID2
        mass_center1 = rigid[master]._get_position()
        rotate_matrix1 = SetToRotate(rigid[master].q)
        scale = box[master].scale
        for contact_node in range(ls_contact_body_start[master], ls_contact_body_end[master]):
            global_node = ls_contact_ref[contact_node]
            local_node = rigid[master].global_node_to_local(global_node)
            surface_node = mass_center1 + rotate_matrix1 @ (scale * vertice[local_node].x)
            projected_point = wall[wall_id]._point_projection(surface_node)
            if (
                wall[wall_id]._is_in_plane(projected_point)
                and wall[wall_id]._is_sphere_intersect(surface_node, verlet_distance) == 1
            ):
                sques = ti.atomic_add(point_wall[contact_node + 1], 1)
                potential_list_point_wall[sques + contact_node * potential_point_num] = wall_id
                assert (
                    sques < potential_point_num
                ), f"Keyword:: /point_coordination_number[1]/ is too small, LSMPM contact node {contact_node} has {sques+1} potential wall contact number"
