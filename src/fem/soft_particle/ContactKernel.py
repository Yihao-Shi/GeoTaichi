"""Device PT/EE DEM-style contact for FEM soft particles."""

import taichi as ti

from src.fem.contact.ExplicitGeometry import (
    closest_point_triangle as _closest_point_triangle,
    signed_closest_feature_geometry as _signed_closest_feature_geometry,
)
from src.utils.VectorFunction import Normalize
from src.utils.constants import PI, Threshold


@ti.kernel
def update_surface_pseudonormals(
    face_count: ti.i32,
    faces: ti.template(),
    positions: ti.template(),
    pseudonormals: ti.template(),
):
    """Build current angle-weighted vertex pseudonormals for gap signs."""

    for node in pseudonormals:
        pseudonormals[node] = ti.Vector.zero(float, 3)
    for face_id in range(face_count):
        face = faces[face_id]
        first = positions[face[0]]
        second = positions[face[1]]
        third = positions[face[2]]
        face_cross = (second - first).cross(third - first)
        face_length = face_cross.norm()
        if face_length > Threshold:
            normal = face_cross / face_length
            for local in ti.static(range(3)):
                center = positions[face[local]]
                before = positions[face[(local + 1) % 3]] - center
                after = positions[face[(local + 2) % 3]] - center
                cosine = before.dot(after) / ti.max(before.norm() * after.norm(), Threshold)
                angle = ti.acos(ti.max(-1.0, ti.min(1.0, cosine)))
                for component in ti.static(range(3)):
                    ti.atomic_add(
                        pseudonormals[face[local]][component],
                        angle * normal[component],
                    )


@ti.func
def _closest_segment_parameters(first0, first1, second0, second1):
    first_direction = first1 - first0
    second_direction = second1 - second0
    offset = first0 - second0
    aa = first_direction.dot(first_direction)
    ab = first_direction.dot(second_direction)
    bb = second_direction.dot(second_direction)
    ao = first_direction.dot(offset)
    bo = second_direction.dot(offset)
    first_ratio = 0.0
    second_ratio = 0.0
    if aa <= Threshold and bb <= Threshold:
        pass
    elif aa <= Threshold:
        second_ratio = ti.max(0.0, ti.min(1.0, bo / ti.max(bb, Threshold)))
    elif bb <= Threshold:
        first_ratio = ti.max(0.0, ti.min(1.0, -ao / ti.max(aa, Threshold)))
    else:
        denominator = aa * bb - ab * ab
        if denominator > Threshold:
            first_ratio = ti.max(0.0, ti.min(1.0, (ab * bo - bb * ao) / denominator))
        projected = ab * first_ratio + bo
        if projected < 0.0:
            second_ratio = 0.0
            first_ratio = ti.max(0.0, ti.min(1.0, -ao / aa))
        elif projected > bb:
            second_ratio = 1.0
            first_ratio = ti.max(0.0, ti.min(1.0, (ab - ao) / aa))
        else:
            second_ratio = projected / bb
    return first_ratio, second_ratio


@ti.func
def _effective_mass(stencil, weights, mass):
    inverse_mass = 0.0
    for local in ti.static(range(4)):
        inverse_mass += weights[local] * weights[local] / ti.max(mass[stencil[local]], Threshold)
    return 1.0 / ti.max(inverse_mass, Threshold)


@ti.func
def _contact_force(
    model_type: ti.template(),
    prop,
    overlap,
    measure,
    effective_mass,
    normal,
    normal_velocity,
    tangential_velocity,
    old_tangential_overlap,
    dt,
):
    normal_stiffness = measure * prop.kn
    tangential_stiffness = measure * prop.ks
    normal_scalar = 0.0
    if ti.static(model_type == 0):
        normal_scalar = (
            normal_stiffness * overlap
            - 2.0 * prop.normal_damping * ti.sqrt(ti.max(effective_mass * normal_stiffness, 0.0)) * normal_velocity
        )
    elif ti.static(model_type == 1):
        effective_radius = ti.sqrt(ti.max(measure / PI, Threshold))
        contact_radius = ti.sqrt(ti.max(overlap * effective_radius, Threshold))
        normal_stiffness = measure * 2.0 * prop.effective_young * contact_radius
        tangential_stiffness = measure * 8.0 * prop.effective_shear * contact_radius
        normal_scalar = (
            2.0 / 3.0
        ) * normal_stiffness * overlap - 1.8257 * prop.restitution_damping * normal_velocity * ti.sqrt(
            ti.max(normal_stiffness * effective_mass, 0.0)
        )
    else:
        gapn = -overlap
        eta = gapn + prop.barrier_cutoff
        d_cap = 2.0 * prop.barrier_cutoff
        assert eta > 0.0, "explicit FEM contact crossed the barrier gap"
        logarithm = ti.log(eta / d_cap)
        normal_stiffness = (
            measure * prop.barrier_kappa * (-2.0 * logarithm - (eta - d_cap) * (3.0 * eta + d_cap) / (eta * eta))
        )
        normal_scalar = (
            measure * prop.barrier_kappa * (eta - d_cap) * (2.0 * logarithm - d_cap / eta + 1.0)
            - 2.0 * prop.normal_damping * ti.sqrt(ti.max(effective_mass * normal_stiffness, 0.0)) * normal_velocity
        )
    normal_scalar = ti.max(normal_scalar, 0.0)
    normal_force = normal_scalar * normal
    rotated = old_tangential_overlap - old_tangential_overlap.dot(normal) * normal
    trial_overlap = tangential_velocity * dt + old_tangential_overlap.norm() * Normalize(rotated)
    current_overlap = trial_overlap
    elastic_tangential_trial = -tangential_stiffness * current_overlap
    tangential_force = elastic_tangential_trial
    friction_limit = prop.friction * normal_scalar
    friction_increment = 0.0
    if ti.static(model_type == 2):
        cutoff = friction_limit / ti.max(
            prop.barrier_stiffness_ratio * prop.barrier_kappa * measure,
            Threshold,
        )
        direction = Normalize(trial_overlap)
        tangential_force = -friction_limit * direction
        if trial_overlap.norm() < cutoff:
            magnitude_ratio = (
                trial_overlap.norm()
                * (2.0 * cutoff - trial_overlap.norm())
                / ti.max(cutoff * cutoff, Threshold * Threshold)
            )
            tangential_force *= magnitude_ratio
            equivalent_stiffness = 0.0
            if trial_overlap.norm() > Threshold:
                equivalent_stiffness = tangential_force.norm() / trial_overlap.norm()
            tangential_force -= (
                2.0
                * prop.tangential_damping
                * ti.sqrt(ti.max(effective_mass * equivalent_stiffness, 0.0))
                * tangential_velocity
            )
        else:
            current_overlap = cutoff * direction
            friction_increment = ti.max(
                -tangential_force.dot(trial_overlap - current_overlap),
                0.0,
            )
    elif elastic_tangential_trial.norm() > friction_limit:
        if elastic_tangential_trial.norm() > Threshold:
            tangential_force = friction_limit * elastic_tangential_trial.normalized()
        current_overlap = -tangential_force / ti.max(tangential_stiffness, Threshold)
        plastic_slip = trial_overlap - current_overlap
        friction_increment = ti.max(-tangential_force.dot(plastic_slip), 0.0)
    elif ti.static(model_type == 0):
        tangential_force -= (
            2.0
            * prop.tangential_damping
            * ti.sqrt(ti.max(effective_mass * tangential_stiffness, 0.0))
            * tangential_velocity
        )
    else:
        tangential_force -= (
            1.8257
            * prop.restitution_damping
            * tangential_velocity
            * ti.sqrt(ti.max(tangential_stiffness * effective_mass, 0.0))
        )
    return (
        normal_force,
        tangential_force,
        current_overlap,
        friction_increment,
    )


@ti.func
def _contact_energy_terms(
    model_type: ti.template(),
    prop,
    overlap,
    measure,
    normal,
    relative_velocity,
    current_tangential_overlap,
    normal_force,
    tangential_force,
    dt,
):
    """Stored contact energy and viscous work from resolved contact data."""
    normal_velocity = relative_velocity.dot(normal)
    tangential_velocity = relative_velocity - normal_velocity * normal
    normal_stiffness = measure * prop.kn
    tangential_stiffness = measure * prop.ks
    normal_elastic_scalar = normal_stiffness * overlap
    normal_elastic_energy = 0.5 * normal_stiffness * overlap * overlap
    if ti.static(model_type == 1):
        effective_radius = ti.sqrt(ti.max(measure / PI, Threshold))
        contact_radius = ti.sqrt(ti.max(overlap * effective_radius, Threshold))
        normal_stiffness = measure * 2.0 * prop.effective_young * contact_radius
        tangential_stiffness = measure * 8.0 * prop.effective_shear * contact_radius
        normal_elastic_scalar = (2.0 / 3.0) * normal_stiffness * overlap
        normal_elastic_energy = 0.4 * normal_elastic_scalar * overlap
    elif ti.static(model_type == 2):
        gapn = -overlap
        eta = gapn + prop.barrier_cutoff
        d_cap = 2.0 * prop.barrier_cutoff
        logarithm = ti.log(eta / d_cap)
        normal_elastic_scalar = measure * prop.barrier_kappa * (eta - d_cap) * (2.0 * logarithm - d_cap / eta + 1.0)
        normal_elastic_energy = -measure * prop.barrier_kappa * (eta - d_cap) * (eta - d_cap) * logarithm
        tangential_stiffness = 0.0

    elastic = normal_elastic_energy + 0.5 * tangential_stiffness * current_tangential_overlap.norm_sqr()
    if ti.static(model_type == 2):
        elastic += ti.max(-current_tangential_overlap.dot(tangential_force), 0.0)
    normal_damping_force = normal_force - normal_elastic_scalar * normal
    damping = ti.max(
        -normal_damping_force.dot(normal_velocity * normal) * dt,
        0.0,
    )
    if ti.static(model_type == 2):
        friction_limit = prop.friction * normal_force.norm()
        tangential_cutoff = friction_limit / ti.max(
            prop.barrier_stiffness_ratio * prop.barrier_kappa * measure,
            Threshold,
        )
        elastic_tangential_force = ti.Vector.zero(float, 3)
        if current_tangential_overlap.norm() > Threshold:
            magnitude_ratio = 1.0
            if current_tangential_overlap.norm() < tangential_cutoff:
                magnitude_ratio = (
                    current_tangential_overlap.norm()
                    * (2.0 * tangential_cutoff - current_tangential_overlap.norm())
                    / ti.max(
                        tangential_cutoff * tangential_cutoff,
                        Threshold * Threshold,
                    )
                )
            elastic_tangential_force = -friction_limit * magnitude_ratio * current_tangential_overlap.normalized()
        elastic = normal_elastic_energy + ti.max(
            -current_tangential_overlap.dot(elastic_tangential_force),
            0.0,
        )
        damping += ti.max(
            -(tangential_force - elastic_tangential_force).dot(tangential_velocity) * dt,
            0.0,
        )
        return elastic, damping
    friction_limit = prop.friction * normal_force.norm()
    elastic_tangential_magnitude = tangential_stiffness * current_tangential_overlap.norm()
    if ti.static(model_type == 2):
        elastic_tangential_magnitude = tangential_force.norm()
    sliding = friction_limit > Threshold and elastic_tangential_magnitude >= friction_limit - 1.0e-8 * ti.max(
        friction_limit, Threshold
    )
    # Coulomb work is accumulated exactly once by _contact_force from the
    # returned plastic slip.  Only viscous tangential work remains here.
    if not sliding:
        tangential_damping_force = tangential_force + tangential_stiffness * current_tangential_overlap
        damping += ti.max(
            -tangential_damping_force.dot(tangential_velocity) * dt,
            0.0,
        )
    return elastic, damping


@ti.kernel
def update_surface_node_area(
    face_count: ti.i32,
    faces: ti.template(),
    positions: ti.template(),
    node_area: ti.template(),
):
    for node in node_area:
        node_area[node] = 0.0
    for face_id in range(face_count):
        face = faces[face_id]
        area = 0.5 * (positions[face[1]] - positions[face[0]]).cross(positions[face[2]] - positions[face[0]]).norm()
        for local in ti.static(range(3)):
            ti.atomic_add(node_area[face[local]], area / 3.0)


@ti.kernel
def build_point_triangle_ranges(
    node_count: ti.i32,
    contact_count: ti.i32,
    stencils: ti.template(),
    starts: ti.template(),
    ends: ti.template(),
):
    """Build contiguous candidate ranges for each slave surface node."""

    for node in range(node_count):
        starts[node] = contact_count
        ends[node] = -1
    for contact in range(contact_count):
        node = stencils[contact][0]
        ti.atomic_min(starts[node], contact)
        ti.atomic_max(ends[node], contact)


@ti.kernel
def measure_active_point_triangle_penetration(
    contact_count: ti.i32,
    stencils: ti.template(),
    positions: ti.template(),
    pseudonormals: ti.template(),
    active: ti.template(),
    maximum_penetration: ti.template(),
):
    """Measure inward depth before replacing the current Verlet list."""

    maximum_penetration[None] = 0.0
    for contact in range(contact_count):
        if active[contact] != 0:
            stencil = stencils[contact]
            point = positions[stencil[0]]
            first = positions[stencil[1]]
            second = positions[stencil[2]]
            third = positions[stencil[3]]
            closest, barycentric = _closest_point_triangle(point, first, second, third)
            face_cross = (second - first).cross(third - first)
            face_length = face_cross.norm()
            if face_length > Threshold:
                offset = point - closest
                _, signed_gap = _signed_closest_feature_geometry(
                    offset,
                    barycentric,
                    stencil,
                    face_cross / face_length,
                    pseudonormals,
                )
                if signed_gap < 0.0:
                    ti.atomic_max(maximum_penetration[None], offset.norm())


@ti.func
def _is_nearest_point_triangle_for_body(
    contact,
    stencil,
    closest,
    stencils,
    node_body,
    positions,
    starts,
    ends,
):
    """Choose one continuous closest surface feature per node/body pair."""

    selected = True
    point = stencil[0]
    target_body = node_body[stencil[1]]
    distance2 = (positions[point] - closest).norm_sqr()
    for other in range(starts[point], ends[point] + 1):
        other_stencil = stencils[other]
        if other_stencil[0] == point and node_body[other_stencil[1]] == target_body:
            other_closest, _ = _closest_point_triangle(
                positions[point],
                positions[other_stencil[1]],
                positions[other_stencil[2]],
                positions[other_stencil[3]],
            )
            other_distance2 = (positions[point] - other_closest).norm_sqr()
            tolerance = 1.0e-14 * ti.max(1.0, distance2)
            if other_distance2 < distance2 - tolerance or (
                ti.abs(other_distance2 - distance2) <= tolerance and other < contact
            ):
                selected = False
    return selected


@ti.kernel
def resolve_point_triangle_contact(
    model_type: ti.template(),
    track_energy: ti.template(),
    contact_count: ti.i32,
    dt: float,
    advance_history: ti.template(),
    stencils: ti.template(),
    node_body: ti.template(),
    properties: ti.template(),
    positions: ti.template(),
    velocity: ti.template(),
    mass: ti.template(),
    node_area: ti.template(),
    pseudonormals: ti.template(),
    point_triangle_start: ti.template(),
    point_triangle_end: ti.template(),
    external_force: ti.template(),
    active: ti.template(),
    normal_force_out: ti.template(),
    tangential_force_out: ti.template(),
    tangential_overlap: ti.template(),
    elastic_energy: ti.template(),
    friction_dissipation: ti.template(),
    damping_dissipation: ti.template(),
):
    for contact in range(contact_count):
        stencil = stencils[contact]
        point = positions[stencil[0]]
        first = positions[stencil[1]]
        second = positions[stencil[2]]
        third = positions[stencil[3]]
        first_body = node_body[stencil[0]]
        second_body = node_body[stencil[1]]
        prop = properties[first_body, second_body]
        active[contact] = 0
        normal_force_out[contact] = ti.Vector.zero(float, 3)
        tangential_force_out[contact] = ti.Vector.zero(float, 3)
        closest, barycentric = _closest_point_triangle(point, first, second, third)
        selected = _is_nearest_point_triangle_for_body(
            contact,
            stencil,
            closest,
            stencils,
            node_body,
            positions,
            point_triangle_start,
            point_triangle_end,
        )
        face_cross = (second - first).cross(third - first)
        face_length = face_cross.norm()
        if prop.active != 0 and face_length > Threshold and selected:
            face_normal = face_cross / face_length
            offset = point - closest
            distance = offset.norm()
            normal, signed_gap = _signed_closest_feature_geometry(
                offset,
                barycentric,
                stencil,
                face_normal,
                pseudonormals,
            )
            overlap = prop.thickness - signed_gap
            contact_active = overlap > 0.0
            if ti.static(model_type == 2):
                contact_active = overlap > -prop.barrier_cutoff
            if contact_active:
                weights = ti.Vector([1.0, -barycentric[0], -barycentric[1], -barycentric[2]])
                relative_velocity = ti.Vector.zero(float, 3)
                for local in ti.static(range(4)):
                    relative_velocity += weights[local] * velocity[stencil[local]]
                normal_velocity = relative_velocity.dot(normal)
                tangential_velocity = relative_velocity - normal_velocity * normal
                effective_mass = _effective_mass(stencil, weights, mass)
                measure = 0.5 * node_area[stencil[0]]
                (
                    normal_force,
                    tangential_force,
                    current_overlap,
                    friction_increment,
                ) = _contact_force(
                    model_type,
                    prop,
                    overlap,
                    measure,
                    effective_mass,
                    normal,
                    normal_velocity,
                    tangential_velocity,
                    tangential_overlap[contact],
                    dt,
                )
                total_force = normal_force + tangential_force
                for local, component in ti.static(ti.ndrange(4, 3)):
                    ti.atomic_add(
                        external_force[stencil[local]][component],
                        weights[local] * total_force[component],
                    )
                active[contact] = 1
                normal_force_out[contact] = normal_force
                tangential_force_out[contact] = tangential_force
                if ti.static(track_energy):
                    diagnostic_overlap = tangential_overlap[contact]
                    if ti.static(advance_history != 0):
                        diagnostic_overlap = current_overlap
                    stored, damping = _contact_energy_terms(
                        model_type,
                        prop,
                        overlap,
                        measure,
                        normal,
                        relative_velocity,
                        diagnostic_overlap,
                        normal_force,
                        tangential_force,
                        dt,
                    )
                    ti.atomic_add(elastic_energy[None], stored)
                    if ti.static(advance_history != 0):
                        ti.atomic_add(friction_dissipation[None], friction_increment)
                        ti.atomic_add(damping_dissipation[None], damping)
                if ti.static(advance_history != 0):
                    tangential_overlap[contact] = current_overlap
            elif ti.static(advance_history != 0):
                tangential_overlap[contact] = ti.Vector.zero(float, 3)


@ti.kernel
def resolve_edge_edge_contact(
    model_type: ti.template(),
    track_energy: ti.template(),
    contact_count: ti.i32,
    dt: float,
    advance_history: ti.template(),
    stencils: ti.template(),
    node_body: ti.template(),
    properties: ti.template(),
    measures: ti.template(),
    positions: ti.template(),
    velocity: ti.template(),
    mass: ti.template(),
    external_force: ti.template(),
    active: ti.template(),
    normal_force_out: ti.template(),
    tangential_force_out: ti.template(),
    tangential_overlap: ti.template(),
    elastic_energy: ti.template(),
    friction_dissipation: ti.template(),
    damping_dissipation: ti.template(),
):
    for contact in range(contact_count):
        stencil = stencils[contact]
        first_body = node_body[stencil[0]]
        second_body = node_body[stencil[2]]
        prop = properties[first_body, second_body]
        active[contact] = 0
        normal_force_out[contact] = ti.Vector.zero(float, 3)
        tangential_force_out[contact] = ti.Vector.zero(float, 3)
        first_ratio, second_ratio = _closest_segment_parameters(
            positions[stencil[0]],
            positions[stencil[1]],
            positions[stencil[2]],
            positions[stencil[3]],
        )
        first_point = (1.0 - first_ratio) * positions[stencil[0]] + first_ratio * positions[stencil[1]]
        second_point = (1.0 - second_ratio) * positions[stencil[2]] + second_ratio * positions[stencil[3]]
        delta = first_point - second_point
        distance = delta.norm()
        overlap = prop.thickness - distance
        contact_active = overlap > 0.0
        if ti.static(model_type == 2):
            contact_active = overlap > -prop.barrier_cutoff
        if prop.active != 0 and contact_active:
            normal = ti.Vector.zero(float, 3)
            if distance > Threshold:
                normal = delta / distance
            else:
                normal = (
                    (positions[stencil[1]] - positions[stencil[0]])
                    .cross(positions[stencil[3]] - positions[stencil[2]])
                    .normalized(Threshold)
                )
            if normal.norm_sqr() > Threshold * Threshold:
                weights = ti.Vector([1.0 - first_ratio, first_ratio, -(1.0 - second_ratio), -second_ratio])
                relative_velocity = ti.Vector.zero(float, 3)
                for local in ti.static(range(4)):
                    relative_velocity += weights[local] * velocity[stencil[local]]
                normal_velocity = relative_velocity.dot(normal)
                tangential_velocity = relative_velocity - normal_velocity * normal
                effective_mass = _effective_mass(stencil, weights, mass)
                (
                    normal_force,
                    tangential_force,
                    current_overlap,
                    friction_increment,
                ) = _contact_force(
                    model_type,
                    prop,
                    overlap,
                    measures[contact],
                    effective_mass,
                    normal,
                    normal_velocity,
                    tangential_velocity,
                    tangential_overlap[contact],
                    dt,
                )
                total_force = normal_force + tangential_force
                for local, component in ti.static(ti.ndrange(4, 3)):
                    ti.atomic_add(
                        external_force[stencil[local]][component],
                        weights[local] * total_force[component],
                    )
                active[contact] = 1
                normal_force_out[contact] = normal_force
                tangential_force_out[contact] = tangential_force
                if ti.static(track_energy):
                    diagnostic_overlap = tangential_overlap[contact]
                    if ti.static(advance_history != 0):
                        diagnostic_overlap = current_overlap
                    stored, damping = _contact_energy_terms(
                        model_type,
                        prop,
                        overlap,
                        measures[contact],
                        normal,
                        relative_velocity,
                        diagnostic_overlap,
                        normal_force,
                        tangential_force,
                        dt,
                    )
                    ti.atomic_add(elastic_energy[None], stored)
                    if ti.static(advance_history != 0):
                        ti.atomic_add(friction_dissipation[None], friction_increment)
                        ti.atomic_add(damping_dissipation[None], damping)
                if ti.static(advance_history != 0):
                    tangential_overlap[contact] = current_overlap
            elif ti.static(advance_history != 0):
                tangential_overlap[contact] = ti.Vector.zero(float, 3)
        elif ti.static(advance_history != 0):
            tangential_overlap[contact] = ti.Vector.zero(float, 3)


@ti.kernel
def accumulate_point_triangle_contact_energy(
    model_type: ti.template(),
    contact_count: ti.i32,
    dt: float,
    advance_history: ti.template(),
    stencils: ti.template(),
    node_body: ti.template(),
    properties: ti.template(),
    positions: ti.template(),
    velocity: ti.template(),
    node_area: ti.template(),
    pseudonormals: ti.template(),
    point_triangle_start: ti.template(),
    point_triangle_end: ti.template(),
    active: ti.template(),
    normal_force: ti.template(),
    tangential_force: ti.template(),
    tangential_overlap: ti.template(),
    elastic_energy: ti.template(),
    friction_dissipation: ti.template(),
    damping_dissipation: ti.template(),
):
    for contact in range(contact_count):
        if active[contact] != 0:
            stencil = stencils[contact]
            point = positions[stencil[0]]
            first = positions[stencil[1]]
            second = positions[stencil[2]]
            third = positions[stencil[3]]
            closest, barycentric = _closest_point_triangle(point, first, second, third)
            selected = _is_nearest_point_triangle_for_body(
                contact,
                stencil,
                closest,
                stencils,
                node_body,
                positions,
                point_triangle_start,
                point_triangle_end,
            )
            face_cross = (second - first).cross(third - first)
            face_length = face_cross.norm()
            if face_length > Threshold and selected:
                face_normal = face_cross / face_length
                offset = point - closest
                distance = offset.norm()
                normal, signed_gap = _signed_closest_feature_geometry(
                    offset,
                    barycentric,
                    stencil,
                    face_normal,
                    pseudonormals,
                )
                overlap = properties[node_body[stencil[0]], node_body[stencil[1]]].thickness - signed_gap
                weights = ti.Vector([1.0, -barycentric[0], -barycentric[1], -barycentric[2]])
                relative_velocity = ti.Vector.zero(float, 3)
                for local in ti.static(range(4)):
                    relative_velocity += weights[local] * velocity[stencil[local]]
                prop = properties[node_body[stencil[0]], node_body[stencil[1]]]
                stored, friction, damping = _contact_energy_terms(
                    model_type,
                    prop,
                    overlap,
                    0.5 * node_area[stencil[0]],
                    normal,
                    relative_velocity,
                    tangential_overlap[contact],
                    normal_force[contact],
                    tangential_force[contact],
                    dt,
                )
                ti.atomic_add(elastic_energy[None], stored)
                if ti.static(advance_history != 0):
                    ti.atomic_add(friction_dissipation[None], friction)
                    ti.atomic_add(damping_dissipation[None], damping)


@ti.kernel
def accumulate_edge_edge_contact_energy(
    model_type: ti.template(),
    contact_count: ti.i32,
    dt: float,
    advance_history: ti.template(),
    stencils: ti.template(),
    node_body: ti.template(),
    properties: ti.template(),
    measures: ti.template(),
    positions: ti.template(),
    velocity: ti.template(),
    active: ti.template(),
    normal_force: ti.template(),
    tangential_force: ti.template(),
    tangential_overlap: ti.template(),
    elastic_energy: ti.template(),
    friction_dissipation: ti.template(),
    damping_dissipation: ti.template(),
):
    for contact in range(contact_count):
        if active[contact] != 0:
            stencil = stencils[contact]
            first_ratio, second_ratio = _closest_segment_parameters(
                positions[stencil[0]],
                positions[stencil[1]],
                positions[stencil[2]],
                positions[stencil[3]],
            )
            first_point = (1.0 - first_ratio) * positions[stencil[0]] + first_ratio * positions[stencil[1]]
            second_point = (1.0 - second_ratio) * positions[stencil[2]] + second_ratio * positions[stencil[3]]
            delta = first_point - second_point
            distance = delta.norm()
            normal = ti.Vector.zero(float, 3)
            if distance > Threshold:
                normal = delta / distance
            else:
                normal = (
                    (positions[stencil[1]] - positions[stencil[0]])
                    .cross(positions[stencil[3]] - positions[stencil[2]])
                    .normalized(Threshold)
                )
            if normal.norm_sqr() > Threshold * Threshold:
                prop = properties[node_body[stencil[0]], node_body[stencil[2]]]
                overlap = prop.thickness - distance
                weights = ti.Vector(
                    [
                        1.0 - first_ratio,
                        first_ratio,
                        -(1.0 - second_ratio),
                        -second_ratio,
                    ]
                )
                relative_velocity = ti.Vector.zero(float, 3)
                for local in ti.static(range(4)):
                    relative_velocity += weights[local] * velocity[stencil[local]]
                stored, friction, damping = _contact_energy_terms(
                    model_type,
                    prop,
                    overlap,
                    measures[contact],
                    normal,
                    relative_velocity,
                    tangential_overlap[contact],
                    normal_force[contact],
                    tangential_force[contact],
                    dt,
                )
                ti.atomic_add(elastic_energy[None], stored)
                if ti.static(advance_history != 0):
                    ti.atomic_add(friction_dissipation[None], friction)
                    ti.atomic_add(damping_dissipation[None], damping)


__all__ = [
    "build_point_triangle_ranges",
    "measure_active_point_triangle_penetration",
    "accumulate_edge_edge_contact_energy",
    "accumulate_point_triangle_contact_energy",
    "resolve_edge_edge_contact",
    "resolve_point_triangle_contact",
    "update_surface_pseudonormals",
    "update_surface_node_area",
]
