"""Taichi kernels for the migrated FEM--DEM surface laws."""

import taichi as ti

from src.utils.GeometryFunction import SphereTriangleIntersectionArea
from src.utils.Quaternion import SetToRotate
from src.utils.VectorFunction import Normalize
from src.utils.constants import PI, Threshold


@ti.func
def _triangle_area(first, second, third):
    return 0.5 * (second - first).cross(third - first).norm()


@ti.func
def _contact_area_fraction(point, radius, distance, normal, first, second, third):
    fraction = 0.0
    if 0.0 < distance < radius:
        section_radius = ti.sqrt(radius * radius - distance * distance)
        section_area = PI * section_radius * section_radius
        projection = point - distance * normal
        area = SphereTriangleIntersectionArea(projection, section_radius, first, second, third, normal)
        fraction = ti.abs(area / section_area)
    return fraction


@ti.func
def _apply_action_reaction(
    particle_id,
    face,
    total_force,
    contact_position,
    projection,
    first,
    second,
    third,
    particles,
    faces,
    nodal_force,
):
    # Preserve the source FEM--DEM torque convention, F x arm.
    torque = total_force.cross(particles[particle_id].x - contact_position)
    particles[particle_id]._update_contact_interaction(total_force, torque)
    total_area = _triangle_area(first, second, third)
    if total_area > 1.0e-30:
        equivalent_force = total_force / total_area
        ids = faces[face]
        weights = ti.Vector(
            [
                _triangle_area(projection, second, third),
                _triangle_area(projection, first, third),
                _triangle_area(projection, first, second),
            ]
        )
        for local in ti.static(range(3)):
            for component in ti.static(range(3)):
                ti.atomic_add(
                    nodal_force[ids[local]][component],
                    -weights[local] * equivalent_force[component],
                )


@ti.kernel
def reset_contact_force(contact_count: ti.i32, contacts: ti.template()):
    for contact in range(contact_count):
        contacts[contact].active = 0
        contacts[contact].normal_force = ti.Vector.zero(float, 3)
        contacts[contact].tangential_force = ti.Vector.zero(float, 3)


@ti.kernel
def reset_levelset_contact_force(contact_count: ti.i32, contacts: ti.template()):
    for contact in range(contact_count):
        contacts[contact].active = 0
        contacts[contact].normal_force = ti.Vector.zero(float, 3)
        contacts[contact].tangential_force = ti.Vector.zero(float, 3)
        contacts[contact].normal_gap = 0.0


@ti.kernel
def reset_compact_facet_wall_contact_force(
    candidate_count: ti.i32,
    contacts: ti.template(),
):
    """Reset instantaneous FEM--wall output without erasing friction history."""

    for candidate in range(candidate_count):
        contacts[candidate].active = 0
        contacts[candidate].normal_force = ti.Vector.zero(float, 3)
        contacts[candidate].tangential_force = ti.Vector.zero(float, 3)


@ti.kernel
def clear_facet_wall_history(
    capacity: ti.i32,
    state: ti.template(),
    overflow: ti.template(),
):
    overflow[None] = 0
    for slot in range(capacity):
        state[slot] = 2


@ti.kernel
def count_facet_wall_history(
    candidate_count: ti.i32,
    contacts: ti.template(),
    entry_count: ti.template(),
):
    entry_count[None] = 0
    for candidate in range(candidate_count):
        if contacts[candidate].normal_gap < 0.0 or contacts[candidate].old_tangential_overlap.norm_sqr() > 0.0:
            ti.atomic_add(entry_count[None], 1)


@ti.kernel
def save_facet_wall_history(
    candidate_count: ti.i32,
    candidate_pairs: ti.template(),
    contacts: ti.template(),
    capacity: ti.i32,
    state: ti.template(),
    keys: ti.template(),
    gaps: ti.template(),
    overlaps: ti.template(),
    overflow: ti.template(),
):
    mask = ti.i64(capacity - 1)
    for candidate in range(candidate_count):
        gap = contacts[candidate].normal_gap
        overlap = contacts[candidate].old_tangential_overlap
        if gap < 0.0 or overlap.norm_sqr() > 0.0:
            key = ti.cast(candidate_pairs[candidate], ti.i64)
            slot = int((key * ti.i64(1140071481932319845)) & mask)
            inserted = 0
            probe = 0
            while probe < capacity and inserted == 0:
                target = (slot + probe) & (capacity - 1)
                target_state = state[target]
                if target_state == 0:
                    if keys[target] == key:
                        gaps[target] = gap
                        overlaps[target] = overlap
                        inserted = 1
                    else:
                        probe += 1
                elif target_state == 2:
                    previous = ti.atomic_min(state[target], 1)
                    if previous == 2:
                        keys[target] = key
                        gaps[target] = gap
                        overlaps[target] = overlap
                        state[target] = 0
                        inserted = 1
                    elif previous == 0:
                        state[target] = 0
                    else:
                        probe += 1
                else:
                    probe += 1
            if inserted == 0:
                overflow[None] = 1


@ti.func
def _facet_wall_history_value(
    pair,
    capacity,
    state,
    keys,
    gaps,
    overlaps,
):
    gap = 0.0
    overlap = ti.Vector.zero(float, 3)
    key = ti.cast(pair, ti.i64)
    mask = ti.i64(capacity - 1)
    slot = int((key * ti.i64(1140071481932319845)) & mask)
    probe = 0
    finished = 0
    while probe < capacity and finished == 0:
        target = (slot + probe) & (capacity - 1)
        target_state = state[target]
        if target_state == 0:
            if keys[target] == key:
                gap = gaps[target]
                overlap = overlaps[target]
                finished = 1
            else:
                probe += 1
        elif target_state == 2:
            finished = 1
        else:
            probe += 1
    return gap, overlap


@ti.kernel
def build_compact_facet_wall_candidates(
    surface_vertex_count: ti.i32,
    wall_count: ti.i32,
    skin: float,
    wall: ti.template(),
    patch_nodes: ti.template(),
    surface_vertices: ti.template(),
    candidate_count: ti.template(),
    candidate_capacity: ti.i32,
    candidate_pairs: ti.template(),
    contacts: ti.template(),
    history_capacity: ti.i32,
    history_state: ti.template(),
    history_keys: ti.template(),
    history_gaps: ti.template(),
    history_overlaps: ti.template(),
):
    """Build a compact surface-node--facet list padded by one Verlet skin."""

    candidate_count[None] = 0
    for local_pair in range(surface_vertex_count * wall_count):
        local_node = local_pair // wall_count
        wall_index = local_pair - local_node * wall_count
        node = surface_vertices[local_node]
        pair = node * wall_count + wall_index
        old_gap, old_overlap = _facet_wall_history_value(
            pair,
            history_capacity,
            history_state,
            history_keys,
            history_gaps,
            history_overlaps,
        )
        if wall[wall_index].active != 0:
            point = patch_nodes[node]
            distance = wall[wall_index]._point_to_wall_distance(point)
            gapn = wall[wall_index]._get_norm_distance(point)
            projection = point - gapn * wall[wall_index].norm
            penetrated = gapn < 0.0 and wall[wall_index]._is_in_plane(projection) != 0
            retained = old_gap < 0.0 and penetrated
            if ti.abs(distance) <= skin or retained:
                target = ti.atomic_add(candidate_count[None], 1)
                if target < candidate_capacity:
                    candidate_pairs[target] = pair
                    contacts[target].active = 0
                    contacts[target].normal_gap = old_gap
                    contacts[target].normal_force = ti.Vector.zero(float, 3)
                    contacts[target].tangential_force = ti.Vector.zero(float, 3)
                    contacts[target].old_tangential_overlap = old_overlap


@ti.kernel
def commit_facet_wall_search_state(
    surface_vertex_count: ti.i32,
    patch_nodes: ti.template(),
    surface_vertices: ti.template(),
    search_nodes: ti.template(),
):
    for local in range(surface_vertex_count):
        node = surface_vertices[local]
        search_nodes[local] = patch_nodes[node]


@ti.kernel
def measure_facet_wall_rebuild_requirement(
    surface_vertex_count: ti.i32,
    threshold: float,
    patch_nodes: ti.template(),
    surface_vertices: ti.template(),
    search_nodes: ti.template(),
    rebuild_required: ti.template(),
):
    """Measure FEM surface sweep relative to fixed facet walls."""

    rebuild_required[None] = 0
    for local in range(surface_vertex_count):
        node = surface_vertices[local]
        sweep = (patch_nodes[node] - search_nodes[local]).norm()
        if sweep > threshold:
            ti.atomic_max(rebuild_required[None], 1)


@ti.kernel
def commit_moving_facet_wall_search_state(
    surface_vertex_count: ti.i32,
    wall_count: ti.i32,
    patch_nodes: ti.template(),
    surface_vertices: ti.template(),
    wall: ti.template(),
    search_nodes: ti.template(),
    search_wall_centers: ti.template(),
):
    """Store both sides of the FEM--moving-facet Verlet reference state."""

    for local in range(surface_vertex_count):
        node = surface_vertices[local]
        search_nodes[local] = patch_nodes[node]
    for facet in range(wall_count):
        search_wall_centers[facet] = wall[facet]._get_center()


@ti.kernel
def measure_moving_facet_wall_rebuild_requirement(
    surface_vertex_count: ti.i32,
    wall_count: ti.i32,
    threshold: float,
    patch_nodes: ti.template(),
    surface_vertices: ti.template(),
    wall: ti.template(),
    search_nodes: ti.template(),
    search_wall_centers: ti.template(),
    maximum_node_sweep: ti.template(),
    maximum_wall_sweep: ti.template(),
    rebuild_required: ti.template(),
):
    """Test the relative FEM--wall sweep since the last candidate build."""

    maximum_node_sweep[None] = 0.0
    maximum_wall_sweep[None] = 0.0
    rebuild_required[None] = 0
    for local in range(surface_vertex_count):
        node = surface_vertices[local]
        sweep = (patch_nodes[node] - search_nodes[local]).norm()
        ti.atomic_max(maximum_node_sweep[None], sweep)
    for facet in range(wall_count):
        sweep = (wall[facet]._get_center() - search_wall_centers[facet]).norm()
        ti.atomic_max(maximum_wall_sweep[None], sweep)
    if maximum_node_sweep[None] + maximum_wall_sweep[None] > threshold:
        rebuild_required[None] = 1


@ti.kernel
def resolve_linear_facet_wall_contact(
    candidate_count: ti.i32,
    wall_count: ti.i32,
    candidate_pairs: ti.template(),
    dt: float,
    properties: ti.template(),
    wall: ti.template(),
    patch_nodes: ti.template(),
    node_body: ti.template(),
    node_area: ti.template(),
    fem_velocity: ti.template(),
    fem_mass: ti.template(),
    fem_external_force: ti.template(),
    contacts: ti.template(),
    elastic_energy: ti.template(),
    friction_dissipation: ti.template(),
    damping_dissipation: ti.template(),
    track_energy: ti.template(),
):
    """Resolve explicit FEM surface-node contact against moving DEM facets."""

    for candidate in range(candidate_count):
        pair = candidate_pairs[candidate]
        node = pair // wall_count
        wall_index = pair - node * wall_count
        old = contacts[candidate].old_tangential_overlap
        if wall[wall_index].active != 0 and node_body[node] >= 0:
            point = patch_nodes[node]
            normal = wall[wall_index].norm
            gapn = wall[wall_index]._get_norm_distance(point)
            projection = point - gapn * normal
            area_weight = node_area[node]
            if gapn < 0.0 and area_weight > Threshold and wall[wall_index]._is_in_plane(projection) != 0:
                material = ti.cast(wall[wall_index].materialID, ti.i32)
                prop = properties[material, node_body[node]]
                assert prop.active != 0, "FEM--DEM wall contact property is missing"
                relative_velocity = fem_velocity[node] - wall[wall_index].v
                normal_velocity = relative_velocity.dot(normal)
                tangential_velocity = relative_velocity - normal_velocity * normal
                normal_stiffness = area_weight * prop.kn
                tangential_stiffness = area_weight * prop.ks
                effective_mass = ti.max(fem_mass[node], Threshold)
                normal_elastic_scalar = -normal_stiffness * gapn
                trial_normal_damping_scalar = (
                    -2.0 * prop.normal_damping * ti.sqrt(effective_mass * normal_stiffness) * normal_velocity
                )
                normal_scalar = ti.max(
                    normal_elastic_scalar + trial_normal_damping_scalar,
                    0.0,
                )
                # The unilateral contact applies no tensile force.  Use the
                # damping component of the force that was actually applied,
                # rather than the uncapped dashpot trial, in the work ledger.
                normal_damping_scalar = normal_scalar - normal_elastic_scalar
                normal_force = normal_scalar * normal
                rotated = old - old.dot(normal) * normal
                trial = tangential_velocity * dt + old.norm() * Normalize(rotated)
                current = trial
                elastic_tangential_trial = -tangential_stiffness * current
                tangential_force = elastic_tangential_trial
                limit = prop.friction * normal_scalar
                friction_increment = 0.0
                tangential_damping_power = 0.0
                # Match the validated DEM linear law: the Coulomb return map
                # acts on the elastic spring trial. Tangential damping is
                # added only while that spring remains in the sticking set.
                # Comparing the spring-plus-dashpot total instead would load
                # the spring instantaneously when a dashpot alone reaches the
                # Coulomb limit and creates non-conjugate stored energy.
                if elastic_tangential_trial.norm() > limit:
                    if elastic_tangential_trial.norm() > Threshold:
                        tangential_force = limit * elastic_tangential_trial.normalized()
                    current = -tangential_force / ti.max(tangential_stiffness, Threshold)
                    plastic_slip = trial - current
                    if ti.static(track_energy):
                        friction_increment = ti.max(-tangential_force.dot(plastic_slip), 0.0)
                else:
                    tangential_damping_force = (
                        -2.0
                        * prop.tangential_damping
                        * ti.sqrt(effective_mass * tangential_stiffness)
                        * tangential_velocity
                    )
                    tangential_force += tangential_damping_force
                    if ti.static(track_energy):
                        tangential_damping_power = ti.max(
                            0.0,
                            -tangential_damping_force.dot(tangential_velocity),
                        )
                total_force = normal_force + tangential_force
                contacts[candidate].active = 1
                contacts[candidate].normal_gap = gapn
                contacts[candidate].normal_force = normal_force
                contacts[candidate].tangential_force = tangential_force
                contacts[candidate].old_tangential_overlap = current
                for component in ti.static(range(3)):
                    ti.atomic_add(
                        fem_external_force[node][component],
                        total_force[component],
                    )
                wall[wall_index]._update_contact_interaction(-total_force)
                if ti.static(track_energy):
                    ti.atomic_add(
                        elastic_energy[None],
                        0.5 * normal_stiffness * gapn * gapn + 0.5 * tangential_stiffness * current.norm_sqr(),
                    )
                    ti.atomic_add(
                        damping_dissipation[None],
                        (
                            ti.max(
                                0.0,
                                -normal_damping_scalar * normal_velocity,
                            )
                            + tangential_damping_power
                        )
                        * dt,
                    )
                    ti.atomic_add(
                        friction_dissipation[None],
                        friction_increment,
                    )
            else:
                contacts[candidate].old_tangential_overlap = ti.Vector.zero(float, 3)
                contacts[candidate].normal_gap = 0.0
        else:
            contacts[candidate].old_tangential_overlap = ti.Vector.zero(float, 3)
            contacts[candidate].normal_gap = 0.0


@ti.kernel
def resolve_barrier_facet_wall_contact(
    candidate_count: ti.i32,
    wall_count: ti.i32,
    candidate_pairs: ti.template(),
    dt: float,
    properties: ti.template(),
    wall: ti.template(),
    patch_nodes: ti.template(),
    node_body: ti.template(),
    node_area: ti.template(),
    fem_velocity: ti.template(),
    fem_mass: ti.template(),
    fem_external_force: ti.template(),
    contacts: ti.template(),
    elastic_energy: ti.template(),
    friction_dissipation: ti.template(),
    damping_dissipation: ti.template(),
    track_energy: ti.template(),
):
    """Resolve the DEM barrier law between FEM nodes and DEM facets."""

    for candidate in range(candidate_count):
        pair = candidate_pairs[candidate]
        node = pair // wall_count
        wall_index = pair - node * wall_count
        old = contacts[candidate].old_tangential_overlap
        if wall[wall_index].active != 0 and node_body[node] >= 0:
            point = patch_nodes[node]
            normal = wall[wall_index].norm
            gapn = wall[wall_index]._get_norm_distance(point)
            contacts[candidate].normal_gap = gapn
            projection = point - gapn * normal
            area_weight = node_area[node]
            material = ti.cast(wall[wall_index].materialID, ti.i32)
            prop = properties[material, node_body[node]]
            assert prop.active != 0, "FEM--DEM wall barrier property is missing"
            if gapn < prop.normal_cutoff and area_weight > Threshold and wall[wall_index]._is_in_plane(projection) != 0:
                relative_velocity = fem_velocity[node] - wall[wall_index].v
                normal_velocity = relative_velocity.dot(normal)
                tangential_velocity = relative_velocity - normal_velocity * normal
                (
                    normal_force,
                    tangential_force,
                    current,
                    stored,
                    friction_increment,
                    damping_increment,
                ) = _barrier_contact_force(
                    prop,
                    gapn,
                    area_weight,
                    ti.max(fem_mass[node], Threshold),
                    normal,
                    normal_velocity,
                    tangential_velocity,
                    old,
                    dt,
                    track_energy,
                )
                total_force = normal_force + tangential_force
                contacts[candidate].active = 1
                contacts[candidate].normal_force = normal_force
                contacts[candidate].tangential_force = tangential_force
                contacts[candidate].old_tangential_overlap = current
                for component in ti.static(range(3)):
                    ti.atomic_add(
                        fem_external_force[node][component],
                        total_force[component],
                    )
                wall[wall_index]._update_contact_interaction(-total_force)
                if ti.static(track_energy):
                    ti.atomic_add(elastic_energy[None], stored)
                    ti.atomic_add(friction_dissipation[None], friction_increment)
                    ti.atomic_add(damping_dissipation[None], damping_increment)
            else:
                contacts[candidate].old_tangential_overlap = ti.Vector.zero(float, 3)
        else:
            contacts[candidate].old_tangential_overlap = ti.Vector.zero(float, 3)
            contacts[candidate].normal_gap = 0.0


@ti.kernel
def accumulate_linear_facet_wall_elastic_energy(
    candidate_count: ti.i32,
    wall_count: ti.i32,
    target_wall_id: ti.i32,
    candidate_pairs: ti.template(),
    properties: ti.template(),
    wall: ti.template(),
    patch_nodes: ti.template(),
    node_body: ti.template(),
    node_area: ti.template(),
    contacts: ti.template(),
    elastic_energy: ti.template(),
):
    """Measure the FEM penalty energy removed with one facet-wall group."""

    elastic_energy[None] = 0.0
    for candidate in range(candidate_count):
        pair = candidate_pairs[candidate]
        node = pair // wall_count
        wall_index = pair - node * wall_count
        if (
            int(wall[wall_index].active) == 1
            and int(wall[wall_index].wallID) == target_wall_id
            and node_body[node] >= 0
        ):
            point = patch_nodes[node]
            gap = wall[wall_index]._get_norm_distance(point)
            projection = point - gap * wall[wall_index].norm
            area = node_area[node]
            if gap < 0.0 and area > Threshold and wall[wall_index]._is_in_plane(projection) != 0:
                material = ti.cast(wall[wall_index].materialID, ti.i32)
                prop = properties[material, node_body[node]]
                if prop.active != 0:
                    overlap = contacts[candidate].old_tangential_overlap
                    ti.atomic_add(
                        elastic_energy[None],
                        0.5 * area * prop.kn * gap * gap + 0.5 * area * prop.ks * overlap.norm_sqr(),
                    )


@ti.func
def _node_rigid_effective_mass(node, nodal_mass, rigid_mass):
    inverse_mass = 1.0 / ti.max(rigid_mass, Threshold)
    inverse_mass += 1.0 / ti.max(nodal_mass[node], Threshold)
    return 1.0 / ti.max(inverse_mass, Threshold)


@ti.func
def _apply_levelset_action_reaction(
    node,
    point,
    total_force,
    rigid_id,
    rigid,
    nodal_force,
):
    for component in ti.static(range(3)):
        ti.atomic_add(nodal_force[node][component], total_force[component])
    reaction = -total_force
    center = rigid[rigid_id].mass_center
    torque = (point - center).cross(reaction)
    rigid[rigid_id]._update_contact_interaction(reaction, torque)


@ti.func
def _barrier_contact_force(
    prop,
    gapn,
    coefficient,
    effective_mass,
    normal,
    normal_velocity,
    tangential_velocity,
    old_tangential_overlap,
    dt,
    track_energy: ti.template(),
):
    """Evaluate the existing DEM barrier potential for one FEM quadrature node."""

    eta = gapn + prop.normal_cutoff
    d_cap = 2.0 * prop.normal_cutoff
    assert eta > 0.0, "explicit FEDEM step crossed the barrier gap"
    logarithm = ti.log(eta / d_cap)
    normal_stiffness = coefficient * prop.kappa * (-2.0 * logarithm - (eta - d_cap) * (3.0 * eta + d_cap) / (eta * eta))
    normal_elastic = prop.kappa * coefficient * (eta - d_cap) * (2.0 * logarithm - d_cap / eta + 1.0)
    normal_damping = (
        -2.0 * prop.normal_damping * ti.sqrt(ti.max(effective_mass * normal_stiffness, 0.0)) * normal_velocity
    )
    normal_scalar = ti.max(normal_elastic + normal_damping, 0.0)
    normal_force = normal_scalar * normal

    rotated = old_tangential_overlap - old_tangential_overlap.dot(normal) * normal
    trial = tangential_velocity * dt + old_tangential_overlap.norm() * Normalize(rotated)
    direction = Normalize(trial)
    cutoff = (
        prop.friction
        * normal_scalar
        / ti.max(
            prop.stiffness_ratio * prop.kappa * coefficient,
            Threshold,
        )
    )
    current = trial
    tangential_force = -prop.friction * normal_scalar * direction
    friction_increment = 0.0
    tangential_damping_power = 0.0
    if trial.norm() < cutoff:
        magnitude_ratio = 0.0
        if cutoff > Threshold:
            magnitude_ratio = (
                trial.norm() * (2.0 * cutoff - trial.norm()) / ti.max(cutoff * cutoff, Threshold * Threshold)
            )
        tangential_force *= magnitude_ratio
        equivalent_stiffness = 0.0
        if trial.norm() > Threshold:
            equivalent_stiffness = tangential_force.norm() / trial.norm()
        tangential_damping_force = (
            -2.0
            * prop.tangential_damping
            * ti.sqrt(ti.max(effective_mass * equivalent_stiffness, 0.0))
            * tangential_velocity
        )
        tangential_force += tangential_damping_force
        if ti.static(track_energy):
            tangential_damping_power = ti.max(-tangential_damping_force.dot(tangential_velocity), 0.0)
    else:
        current = cutoff * direction
        if ti.static(track_energy):
            friction_increment = ti.max(-tangential_force.dot(trial - current), 0.0)
    stored = 0.0
    damping_increment = 0.0
    if ti.static(track_energy):
        stored = -prop.kappa * coefficient * (eta - d_cap) * (eta - d_cap) * logarithm + ti.max(
            -current.dot(tangential_force), 0.0
        )
        damping_increment = (ti.max(-normal_damping * normal_velocity, 0.0) + tangential_damping_power) * dt
    return (
        normal_force,
        tangential_force,
        current,
        stored,
        friction_increment,
        damping_increment,
    )


@ti.kernel
def resolve_barrier_levelset_contact(
    contact_count: ti.i32,
    dt: float,
    properties: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    grid: ti.template(),
    patch_nodes: ti.template(),
    node_body: ti.template(),
    node_area: ti.template(),
    fem_velocity: ti.template(),
    fem_mass: ti.template(),
    fem_external_force: ti.template(),
    contacts: ti.template(),
    elastic_energy: ti.template(),
    friction_dissipation: ti.template(),
    damping_dissipation: ti.template(),
    track_energy: ti.template(),
):
    for contact in range(contact_count):
        rigid_id = contacts[contact].rigid_id
        node = contacts[contact].node_id
        point = patch_nodes[node]
        center = rigid[rigid_id].mass_center
        rotation = SetToRotate(rigid[rigid_id].q)
        material = ti.cast(rigid[rigid_id].materialID, ti.i32)
        prop = properties[material, node_body[node]]
        assert prop.active != 0, "FEM--LSDEM barrier property is missing"
        old = contacts[contact].old_tangential_overlap
        local_point = rotation.transpose() @ (point - center)
        if box[rigid_id]._in_box(local_point):
            gapn = box[rigid_id].distance(local_point, grid)
            contacts[contact].normal_gap = gapn
            area_weight = node_area[node]
            if gapn < prop.normal_cutoff and area_weight > Threshold:
                gradient = rotation @ box[rigid_id].calculate_gradient(local_point, grid)
                if gradient.norm_sqr() > Threshold * Threshold:
                    gradient_norm = gradient.norm()
                    normal = gradient / gradient_norm
                    rigid_velocity = rigid[rigid_id].v + rigid[rigid_id].w.cross(point - center)
                    relative_velocity = fem_velocity[node] - rigid_velocity
                    normal_velocity = relative_velocity.dot(normal)
                    tangential_velocity = relative_velocity - normal_velocity * normal
                    effective_mass = _node_rigid_effective_mass(node, fem_mass, rigid[rigid_id].m)
                    (
                        normal_force,
                        tangential_force,
                        current,
                        stored,
                        friction_increment,
                        damping_increment,
                    ) = _barrier_contact_force(
                        prop,
                        gapn,
                        area_weight * gradient_norm,
                        effective_mass,
                        normal,
                        normal_velocity,
                        tangential_velocity,
                        old,
                        dt,
                        track_energy,
                    )
                    contacts[contact].active = 1
                    contacts[contact].normal_force = normal_force
                    contacts[contact].tangential_force = tangential_force
                    contacts[contact].old_tangential_overlap = current
                    _apply_levelset_action_reaction(
                        node,
                        point,
                        normal_force + tangential_force,
                        rigid_id,
                        rigid,
                        fem_external_force,
                    )
                    if ti.static(track_energy):
                        ti.atomic_add(elastic_energy[None], stored)
                        ti.atomic_add(friction_dissipation[None], friction_increment)
                        ti.atomic_add(damping_dissipation[None], damping_increment)
            else:
                contacts[contact].old_tangential_overlap = ti.Vector.zero(float, 3)
        else:
            contacts[contact].old_tangential_overlap = ti.Vector.zero(float, 3)


@ti.kernel
def _resolve_linear_levelset_contact_with_energy(
    contact_count: ti.i32,
    dt: float,
    properties: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    grid: ti.template(),
    patch_nodes: ti.template(),
    node_body: ti.template(),
    node_area: ti.template(),
    fem_velocity: ti.template(),
    fem_mass: ti.template(),
    fem_external_force: ti.template(),
    contacts: ti.template(),
    elastic_energy: ti.template(),
    friction_dissipation: ti.template(),
    damping_dissipation: ti.template(),
    track_energy: ti.template(),
):
    for contact in range(contact_count):
        rigid_id = contacts[contact].rigid_id
        node = contacts[contact].node_id
        point = patch_nodes[node]
        area_weight = node_area[node]
        center = rigid[rigid_id].mass_center
        rotation = SetToRotate(rigid[rigid_id].q)
        material = ti.cast(rigid[rigid_id].materialID, ti.i32)
        prop = properties[material, node_body[node]]
        assert prop.active != 0, "FEM--LSDEM contact property is missing"
        old = contacts[contact].old_tangential_overlap
        local_point = rotation.transpose() @ (point - center)
        if box[rigid_id]._in_box(local_point):
            gapn = box[rigid_id].distance(local_point, grid)
            contacts[contact].normal_gap = gapn
            if area_weight > Threshold:
                gradient = rotation @ box[rigid_id].calculate_gradient(local_point, grid)
                if gradient.norm_sqr() > Threshold * Threshold:
                    gradient_norm = gradient.norm()
                    normal = gradient.normalized(Threshold)
                    # Node-to-level-set contact applies the equal and opposite
                    # forces at the FEM surface node.  Evaluating the rigid
                    # velocity and torque at that same spatial point preserves
                    # both discrete power and angular momentum despite the
                    # small numerical penetration.
                    rigid_velocity = rigid[rigid_id].v + rigid[rigid_id].w.cross(point - center)
                    relative_velocity = fem_velocity[node] - rigid_velocity
                    normal_velocity = relative_velocity.dot(normal)
                    tangential_velocity = relative_velocity - normal_velocity * normal
                    effective_mass = _node_rigid_effective_mass(node, fem_mass, rigid[rigid_id].m)
                    normal_stiffness = area_weight * prop.kn
                    tangential_stiffness = area_weight * prop.ks
                    if gapn < 0.0:
                        normal_scalar = (
                            -normal_stiffness * gapn
                            - 2.0 * prop.normal_damping * ti.sqrt(effective_mass * normal_stiffness) * normal_velocity
                        )
                        # The trilinear level-set interpolant is the discrete
                        # gap used by the stored contact energy. Its raw
                        # gradient is needed for the force to be the
                        # derivative of that energy.
                        normal_scalar *= gradient_norm
                        normal_scalar = ti.max(normal_scalar, 0.0)
                        normal_force = normal_scalar * normal
                        rotated = old - old.dot(normal) * normal
                        trial = tangential_velocity * dt + old.norm() * Normalize(rotated)
                        current = trial
                        elastic_tangential_trial = -tangential_stiffness * current
                        tangential_force = elastic_tangential_trial
                        limit = prop.friction * normal_scalar
                        friction_increment = 0.0
                        if elastic_tangential_trial.norm() > limit:
                            if elastic_tangential_trial.norm() > Threshold:
                                tangential_force = limit * elastic_tangential_trial.normalized()
                            current = -tangential_force / ti.max(tangential_stiffness, Threshold)
                            plastic_slip = trial - current
                            if ti.static(track_energy):
                                friction_increment = ti.max(
                                    -tangential_force.dot(plastic_slip),
                                    0.0,
                                )
                        else:
                            tangential_force -= (
                                2.0
                                * prop.tangential_damping
                                * ti.sqrt(effective_mass * tangential_stiffness)
                                * tangential_velocity
                            )
                        contacts[contact].active = 1
                        contacts[contact].normal_force = normal_force
                        contacts[contact].tangential_force = tangential_force
                        contacts[contact].old_tangential_overlap = current
                        _apply_levelset_action_reaction(
                            node,
                            point,
                            normal_force + tangential_force,
                            rigid_id,
                            rigid,
                            fem_external_force,
                        )
                        if ti.static(track_energy):
                            stored, _, damping = _levelset_energy_terms(
                                0,
                                prop,
                                gapn,
                                area_weight,
                                normal,
                                gradient_norm,
                                relative_velocity,
                                current,
                                normal_force,
                                tangential_force,
                                dt,
                            )
                            ti.atomic_add(elastic_energy[None], stored)
                            ti.atomic_add(
                                friction_dissipation[None],
                                friction_increment,
                            )
                            ti.atomic_add(damping_dissipation[None], damping)
                    else:
                        contacts[contact].old_tangential_overlap = ti.Vector.zero(float, 3)
            else:
                contacts[contact].old_tangential_overlap = ti.Vector.zero(float, 3)
        else:
            contacts[contact].old_tangential_overlap = ti.Vector.zero(float, 3)


def resolve_linear_levelset_contact(*args):
    """Resolve linear LSDEM contact with optional energy-ledger outputs.

    The 14-argument form is retained for callers that only request the
    historical friction field. The coupled engine uses the 17-argument form,
    including its static energy-ledger switch, in the same device traversal.
    """
    if len(args) == 17:
        return _resolve_linear_levelset_contact_with_energy(*args)
    if len(args) == 14:
        friction = args[-1]
        elastic = ti.field(dtype=friction.dtype, shape=())
        damping = ti.field(dtype=friction.dtype, shape=())
        elastic.fill(0.0)
        damping.fill(0.0)
        return _resolve_linear_levelset_contact_with_energy(*args[:-1], elastic, friction, damping, True)
    raise TypeError("resolve_linear_levelset_contact expects 14 or 17 arguments")


@ti.kernel
def resolve_hertz_mindlin_levelset_contact(
    contact_count: ti.i32,
    dt: float,
    properties: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    grid: ti.template(),
    patch_nodes: ti.template(),
    node_body: ti.template(),
    node_area: ti.template(),
    fem_velocity: ti.template(),
    fem_mass: ti.template(),
    fem_external_force: ti.template(),
    contacts: ti.template(),
    elastic_energy: ti.template(),
    friction_dissipation: ti.template(),
    damping_dissipation: ti.template(),
    track_energy: ti.template(),
):
    for contact in range(contact_count):
        rigid_id = contacts[contact].rigid_id
        node = contacts[contact].node_id
        point = patch_nodes[node]
        area_weight = node_area[node]
        center = rigid[rigid_id].mass_center
        rotation = SetToRotate(rigid[rigid_id].q)
        material = ti.cast(rigid[rigid_id].materialID, ti.i32)
        prop = properties[material, node_body[node]]
        assert prop.active != 0, "FEM--LSDEM contact property is missing"
        old = contacts[contact].old_tangential_overlap
        local_point = rotation.transpose() @ (point - center)
        if box[rigid_id]._in_box(local_point):
            gapn = box[rigid_id].distance(local_point, grid)
            if gapn < 0.0 and area_weight > Threshold:
                gradient = rotation @ box[rigid_id].calculate_gradient(local_point, grid)
                if gradient.norm_sqr() > Threshold * Threshold:
                    gradient_norm = gradient.norm()
                    normal = gradient.normalized(Threshold)
                    rigid_velocity = rigid[rigid_id].v + rigid[rigid_id].w.cross(point - center)
                    relative_velocity = fem_velocity[node] - rigid_velocity
                    normal_velocity = relative_velocity.dot(normal)
                    tangential_velocity = relative_velocity - normal_velocity * normal
                    effective_mass = _node_rigid_effective_mass(node, fem_mass, rigid[rigid_id].m)
                    radius = ti.sqrt(area_weight / PI)
                    contact_radius = ti.sqrt(ti.max(-gapn * radius, Threshold))
                    normal_stiffness = area_weight * 2.0 * prop.effective_young * contact_radius
                    tangential_stiffness = area_weight * 8.0 * prop.effective_shear * contact_radius
                    normal_scalar = -(
                        2.0 / 3.0
                    ) * normal_stiffness * gapn - 1.8257 * prop.damping * normal_velocity * ti.sqrt(
                        normal_stiffness * effective_mass
                    )
                    normal_scalar *= gradient_norm
                    normal_scalar = ti.max(normal_scalar, 0.0)
                    normal_force = normal_scalar * normal
                    rotated = old - old.dot(normal) * normal
                    current = tangential_velocity * dt + old.norm() * Normalize(rotated)
                    elastic_tangential_trial = -tangential_stiffness * current
                    tangential_force = elastic_tangential_trial
                    limit = prop.friction * normal_scalar
                    if elastic_tangential_trial.norm() > limit:
                        if elastic_tangential_trial.norm() > Threshold:
                            tangential_force = limit * elastic_tangential_trial.normalized()
                        current = -tangential_force / ti.max(tangential_stiffness, Threshold)
                    else:
                        tangential_force -= (
                            1.8257 * prop.damping * tangential_velocity * ti.sqrt(tangential_stiffness * effective_mass)
                        )
                    contacts[contact].active = 1
                    contacts[contact].normal_force = normal_force
                    contacts[contact].tangential_force = tangential_force
                    contacts[contact].old_tangential_overlap = current
                    _apply_levelset_action_reaction(
                        node,
                        point,
                        normal_force + tangential_force,
                        rigid_id,
                        rigid,
                        fem_external_force,
                    )
                    if ti.static(track_energy):
                        stored, friction, damping = _levelset_energy_terms(
                            1,
                            prop,
                            gapn,
                            area_weight,
                            normal,
                            gradient_norm,
                            relative_velocity,
                            current,
                            normal_force,
                            tangential_force,
                            dt,
                        )
                        ti.atomic_add(elastic_energy[None], stored)
                        ti.atomic_add(friction_dissipation[None], friction)
                        ti.atomic_add(damping_dissipation[None], damping)
            else:
                contacts[contact].old_tangential_overlap = ti.Vector.zero(float, 3)
        else:
            contacts[contact].old_tangential_overlap = ti.Vector.zero(float, 3)


@ti.kernel
def resolve_linear_contact(
    contact_count: ti.i32,
    dt: float,
    properties: ti.template(),
    particles: ti.template(),
    patch_nodes: ti.template(),
    faces: ti.template(),
    face_body: ti.template(),
    normals: ti.template(),
    fem_velocity: ti.template(),
    fem_external_force: ti.template(),
    contacts: ti.template(),
):
    for contact in range(contact_count):
        particle_id = contacts[contact].particle_id
        face = contacts[contact].face_id
        ids = faces[face]
        first = patch_nodes[ids[0]]
        second = patch_nodes[ids[1]]
        third = patch_nodes[ids[2]]
        normal = normals[face]
        particle_position = particles[particle_id].x
        face_center = (first + second + third) / 3.0
        radius = particles[particle_id].rad
        distance = (particle_position - face_center).dot(normal)
        normal_gap = distance - radius
        fraction = _contact_area_fraction(particle_position, radius, distance, normal, first, second, third)
        if normal_gap < 0.0 and fraction > 1.0e-15:
            dem_material = ti.cast(particles[particle_id].materialID, ti.i32)
            body = face_body[face]
            prop = properties[dem_material, body]
            assert prop.active != 0, "FEDEM contact property is missing"
            projection = particle_position - distance * normal
            contact_position = projection + 0.5 * normal_gap * normal
            face_velocity = (fem_velocity[ids[0]] + fem_velocity[ids[1]] + fem_velocity[ids[2]]) / 3.0
            relative_velocity = (
                particles[particle_id].v
                + particles[particle_id].w.cross(contact_position - particle_position)
                - face_velocity
            )
            normal_velocity = relative_velocity.dot(normal)
            tangential_velocity = relative_velocity - normal_velocity * normal
            normal_elastic = -prop.kn * normal_gap
            normal_damping = -2.0 * prop.normal_damping * ti.sqrt(particles[particle_id].m * prop.kn) * normal_velocity
            normal_force = (normal_elastic + normal_damping) * normal

            old_overlap = contacts[contact].old_tangential_overlap
            rotated = old_overlap - old_overlap.dot(normal) * normal
            overlap = tangential_velocity * dt + old_overlap.norm() * Normalize(rotated)
            trial_force = -prop.ks * overlap
            tangential_damping = (
                -2.0 * prop.tangential_damping * ti.sqrt(particles[particle_id].m * prop.ks) * tangential_velocity
            )
            friction_limit = prop.friction * ti.abs(normal_elastic + normal_damping)
            tangential_force = ti.Vector.zero(float, 3)
            if trial_force.norm() > friction_limit:
                if trial_force.norm() > 1.0e-30:
                    tangential_force = friction_limit * trial_force.normalized()
                overlap = -tangential_force / prop.ks
            else:
                tangential_force = trial_force + tangential_damping
            total_force = fraction * (normal_force + tangential_force)
            contacts[contact].active = 1
            contacts[contact].old_tangential_overlap = overlap
            contacts[contact].normal_force = fraction * normal_force
            contacts[contact].tangential_force = fraction * tangential_force
            _apply_action_reaction(
                particle_id,
                face,
                total_force,
                contact_position,
                projection,
                first,
                second,
                third,
                particles,
                faces,
                fem_external_force,
            )
        else:
            contacts[contact].old_tangential_overlap = ti.Vector.zero(float, 3)


@ti.kernel
def resolve_hertz_mindlin_contact(
    contact_count: ti.i32,
    dt: float,
    properties: ti.template(),
    particles: ti.template(),
    patch_nodes: ti.template(),
    faces: ti.template(),
    face_body: ti.template(),
    normals: ti.template(),
    fem_velocity: ti.template(),
    fem_external_force: ti.template(),
    contacts: ti.template(),
):
    for contact in range(contact_count):
        particle_id = contacts[contact].particle_id
        face = contacts[contact].face_id
        ids = faces[face]
        first = patch_nodes[ids[0]]
        second = patch_nodes[ids[1]]
        third = patch_nodes[ids[2]]
        normal = normals[face]
        particle_position = particles[particle_id].x
        face_center = (first + second + third) / 3.0
        radius = particles[particle_id].rad
        distance = (particle_position - face_center).dot(normal)
        normal_gap = distance - radius
        fraction = _contact_area_fraction(particle_position, radius, distance, normal, first, second, third)
        if normal_gap < 0.0 and fraction > 1.0e-15:
            dem_material = ti.cast(particles[particle_id].materialID, ti.i32)
            body = face_body[face]
            prop = properties[dem_material, body]
            assert prop.active != 0, "FEDEM contact property is missing"
            contact_radius = ti.sqrt(-normal_gap * radius)
            kn = 2.0 * prop.effective_young * contact_radius
            ks = 8.0 * prop.effective_shear * contact_radius
            projection = particle_position - distance * normal
            contact_position = projection + 0.5 * normal_gap * normal
            face_velocity = (fem_velocity[ids[0]] + fem_velocity[ids[1]] + fem_velocity[ids[2]]) / 3.0
            relative_velocity = (
                particles[particle_id].v
                + particles[particle_id].w.cross(contact_position - particle_position)
                - face_velocity
            )
            normal_velocity = relative_velocity.dot(normal)
            tangential_velocity = relative_velocity - normal_velocity * normal
            normal_elastic = -(2.0 / 3.0) * kn * normal_gap
            normal_damping = -1.8257 * prop.damping * normal_velocity * ti.sqrt(kn * particles[particle_id].m)
            normal_force = (normal_elastic + normal_damping) * normal

            old_overlap = contacts[contact].old_tangential_overlap
            rotated = old_overlap - old_overlap.dot(normal) * normal
            overlap = tangential_velocity * dt + old_overlap.norm() * Normalize(rotated)
            trial_force = -ks * overlap
            tangential_damping = -1.8257 * prop.damping * tangential_velocity * ti.sqrt(ks * particles[particle_id].m)
            friction_limit = prop.friction * ti.abs(normal_elastic + normal_damping)
            tangential_force = ti.Vector.zero(float, 3)
            if trial_force.norm() > friction_limit:
                if trial_force.norm() > 1.0e-30:
                    tangential_force = friction_limit * trial_force.normalized()
                overlap = -tangential_force / ks
            else:
                tangential_force = trial_force + tangential_damping
            total_force = fraction * (normal_force + tangential_force)
            contacts[contact].active = 1
            contacts[contact].old_tangential_overlap = overlap
            contacts[contact].normal_force = fraction * normal_force
            contacts[contact].tangential_force = fraction * tangential_force
            _apply_action_reaction(
                particle_id,
                face,
                total_force,
                contact_position,
                projection,
                first,
                second,
                third,
                particles,
                faces,
                fem_external_force,
            )
        else:
            contacts[contact].old_tangential_overlap = ti.Vector.zero(float, 3)


@ti.func
def _levelset_energy_terms(
    model_type: ti.template(),
    prop,
    gapn,
    area_weight,
    normal,
    gradient_norm,
    relative_velocity,
    tangential_overlap,
    normal_force,
    tangential_force,
    dt,
):
    normal_stiffness = 0.0
    tangential_stiffness = 0.0
    normal_elastic_scalar = 0.0
    normal_elastic_energy = 0.0
    if ti.static(model_type == 0):
        normal_stiffness = area_weight * prop.kn
        tangential_stiffness = area_weight * prop.ks
        normal_elastic_scalar = -normal_stiffness * gapn * gradient_norm
        normal_elastic_energy = 0.5 * normal_stiffness * gapn * gapn
    else:
        radius = ti.sqrt(area_weight / PI)
        contact_radius = ti.sqrt(ti.max(-gapn * radius, Threshold))
        normal_stiffness = area_weight * 2.0 * prop.effective_young * contact_radius
        tangential_stiffness = area_weight * 8.0 * prop.effective_shear * contact_radius
        normal_elastic_scalar = -(2.0 / 3.0) * normal_stiffness * gapn * gradient_norm
        normal_elastic_energy = 0.4 * normal_elastic_scalar * (-gapn) / gradient_norm

    normal_velocity = relative_velocity.dot(normal)
    tangential_velocity = relative_velocity - normal_velocity * normal
    stored = normal_elastic_energy + 0.5 * tangential_stiffness * tangential_overlap.norm_sqr()
    normal_damping_force = normal_force - normal_elastic_scalar * normal
    damping = ti.max(
        -normal_damping_force.dot(normal_velocity * normal) * dt,
        0.0,
    )
    friction = 0.0
    friction_limit = prop.friction * normal_force.norm()
    elastic_tangential_magnitude = tangential_stiffness * tangential_overlap.norm()
    sliding = friction_limit > Threshold and elastic_tangential_magnitude >= friction_limit - 1.0e-8 * ti.max(
        friction_limit, Threshold
    )
    if sliding:
        friction = ti.max(
            -tangential_force.dot(tangential_velocity) * dt,
            0.0,
        )
    else:
        tangential_damping_force = tangential_force + tangential_stiffness * tangential_overlap
        damping += ti.max(
            -tangential_damping_force.dot(tangential_velocity) * dt,
            0.0,
        )
    return stored, friction, damping


@ti.kernel
def accumulate_linear_levelset_contact_energy(
    contact_count: ti.i32,
    dt: float,
    properties: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    grid: ti.template(),
    patch_nodes: ti.template(),
    node_body: ti.template(),
    node_area: ti.template(),
    fem_velocity: ti.template(),
    fem_mass: ti.template(),
    fem_external_force: ti.template(),
    contacts: ti.template(),
    elastic_energy: ti.template(),
    friction_dissipation: ti.template(),
    damping_dissipation: ti.template(),
):
    for contact in range(contact_count):
        if contacts[contact].active != 0:
            rigid_id = contacts[contact].rigid_id
            node = contacts[contact].node_id
            point = patch_nodes[node]
            center = rigid[rigid_id].mass_center
            rotation = SetToRotate(rigid[rigid_id].q)
            local_point = rotation.transpose() @ (point - center)
            gapn = box[rigid_id].distance(local_point, grid)
            gradient = rotation @ box[rigid_id].calculate_gradient(local_point, grid)
            if gradient.norm_sqr() > Threshold * Threshold:
                gradient_norm = gradient.norm()
                normal = gradient.normalized(Threshold)
                rigid_velocity = rigid[rigid_id].v + rigid[rigid_id].w.cross(point - center)
                relative_velocity = fem_velocity[node] - rigid_velocity
                material = ti.cast(rigid[rigid_id].materialID, ti.i32)
                prop = properties[material, node_body[node]]
                stored, _, damping = _levelset_energy_terms(
                    0,
                    prop,
                    gapn,
                    node_area[node],
                    normal,
                    gradient_norm,
                    relative_velocity,
                    contacts[contact].old_tangential_overlap,
                    contacts[contact].normal_force,
                    contacts[contact].tangential_force,
                    dt,
                )
                ti.atomic_add(elastic_energy[None], stored)
                ti.atomic_add(damping_dissipation[None], damping)


@ti.kernel
def accumulate_hertz_mindlin_levelset_contact_energy(
    contact_count: ti.i32,
    dt: float,
    properties: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    grid: ti.template(),
    patch_nodes: ti.template(),
    node_body: ti.template(),
    node_area: ti.template(),
    fem_velocity: ti.template(),
    fem_mass: ti.template(),
    fem_external_force: ti.template(),
    contacts: ti.template(),
    elastic_energy: ti.template(),
    friction_dissipation: ti.template(),
    damping_dissipation: ti.template(),
):
    for contact in range(contact_count):
        if contacts[contact].active != 0:
            rigid_id = contacts[contact].rigid_id
            node = contacts[contact].node_id
            point = patch_nodes[node]
            center = rigid[rigid_id].mass_center
            rotation = SetToRotate(rigid[rigid_id].q)
            local_point = rotation.transpose() @ (point - center)
            gapn = box[rigid_id].distance(local_point, grid)
            gradient = rotation @ box[rigid_id].calculate_gradient(local_point, grid)
            if gradient.norm_sqr() > Threshold * Threshold:
                gradient_norm = gradient.norm()
                normal = gradient.normalized(Threshold)
                rigid_velocity = rigid[rigid_id].v + rigid[rigid_id].w.cross(point - center)
                relative_velocity = fem_velocity[node] - rigid_velocity
                material = ti.cast(rigid[rigid_id].materialID, ti.i32)
                prop = properties[material, node_body[node]]
                stored, friction, damping = _levelset_energy_terms(
                    1,
                    prop,
                    gapn,
                    node_area[node],
                    normal,
                    gradient_norm,
                    relative_velocity,
                    contacts[contact].old_tangential_overlap,
                    contacts[contact].normal_force,
                    contacts[contact].tangential_force,
                    dt,
                )
                ti.atomic_add(elastic_energy[None], stored)
                ti.atomic_add(friction_dissipation[None], friction)
                ti.atomic_add(damping_dissipation[None], damping)


__all__ = [
    "accumulate_hertz_mindlin_levelset_contact_energy",
    "accumulate_linear_levelset_contact_energy",
    "build_compact_facet_wall_candidates",
    "clear_facet_wall_history",
    "commit_facet_wall_search_state",
    "commit_moving_facet_wall_search_state",
    "count_facet_wall_history",
    "measure_facet_wall_rebuild_requirement",
    "measure_moving_facet_wall_rebuild_requirement",
    "reset_contact_force",
    "reset_compact_facet_wall_contact_force",
    "reset_levelset_contact_force",
    "save_facet_wall_history",
    "resolve_linear_contact",
    "resolve_linear_facet_wall_contact",
    "resolve_barrier_facet_wall_contact",
    "resolve_linear_levelset_contact",
    "resolve_hertz_mindlin_contact",
    "resolve_hertz_mindlin_levelset_contact",
    "resolve_barrier_levelset_contact",
]
