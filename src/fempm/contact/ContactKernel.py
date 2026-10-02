"""Taichi DEM-contact kernels for FEM--MPM force exchange."""

import taichi as ti

from src.utils.GeometryFunction import SphereTriangleIntersectionArea
from src.utils.VectorFunction import Normalize
from src.utils.constants import PI


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
    projection,
    first,
    second,
    third,
    particles,
    faces,
    nodal_force,
):
    # Several face candidates may touch one material point, so both sides use
    # atomics. ParticleCoupling.external_force is consumed by MPM P2G.
    for component in ti.static(range(3)):
        ti.atomic_add(
            particles[particle_id].external_force[component],
            total_force[component],
        )
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
            mpm_material = ti.cast(particles[particle_id].materialID, ti.i32)
            body = face_body[face]
            prop = properties[mpm_material, body]
            assert prop.active != 0, "FEMPM contact property is missing"
            projection = particle_position - distance * normal
            face_velocity = (fem_velocity[ids[0]] + fem_velocity[ids[1]] + fem_velocity[ids[2]]) / 3.0
            relative_velocity = particles[particle_id].v - face_velocity
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
            mpm_material = ti.cast(particles[particle_id].materialID, ti.i32)
            body = face_body[face]
            prop = properties[mpm_material, body]
            assert prop.active != 0, "FEMPM contact property is missing"
            contact_radius = ti.sqrt(-normal_gap * radius)
            kn = 2.0 * prop.effective_young * contact_radius
            ks = 8.0 * prop.effective_shear * contact_radius
            projection = particle_position - distance * normal
            face_velocity = (fem_velocity[ids[0]] + fem_velocity[ids[1]] + fem_velocity[ids[2]]) / 3.0
            relative_velocity = particles[particle_id].v - face_velocity
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


__all__ = [
    "reset_contact_force",
    "resolve_linear_contact",
    "resolve_hertz_mindlin_contact",
]
