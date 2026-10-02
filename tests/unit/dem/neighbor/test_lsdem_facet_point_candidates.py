from __future__ import annotations

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.dem.contact.ContactKernel import (
    LSparticle_wall_contact_model,
    kernel_accumulate_linear_lsparticle_wall_elastic_energy,
    kernel_LSparticle_wall_force_assemble_,
)
from src.dem.neighbor.neighbor_kernel import (
    accumulate_pwrelative_displacement_,
    board_search_lsparticle_wall_linked_cell_,
    flag_relative_displacement_,
)
from src.dem.structs.BaseStruct import (
    BoundingBox,
    ContactTable,
    FacetFamily,
    ParticleFamily,
    RigidBody,
    VerletContactTable,
    VerticeNode,
)
from src.physics_model.contact_model.LinearModel import LinearSurfaceProperty


pytestmark = [pytest.mark.unit, pytest.mark.dem, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
        log_level=ti.ERROR,
    )
    yield
    ti.reset()


def test_surface_node_retains_adjacent_facets_and_resolves_only_finite_face():
    """A node crossing a triangulated quad keeps both candidates safely."""
    rigid = RigidBody.field(shape=1)
    vertice = VerticeNode.field(shape=1)
    box = BoundingBox.field(shape=1)
    wall = FacetFamily.field(shape=2)
    broad_pairs = ContactTable.field(shape=2)
    body_wall_prefix = ti.field(dtype=ti.i32, shape=2)
    point_wall_count = ti.field(dtype=ti.i32, shape=2)
    potential_point_wall = ti.field(dtype=ti.i32, shape=2)
    surface_body = ti.field(dtype=ti.i32, shape=1)
    contacts = ContactTable.field(shape=2)
    contact_prefix = ti.field(dtype=ti.i32, shape=2)
    properties = LinearSurfaceProperty.field(shape=1)
    dt = ti.field(dtype=ti.f64, shape=())
    removed_energy = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def initialize():
        rigid[0].mass_center = ti.Vector([0.0, 0.0, 0.0])
        rigid[0].q = ti.Vector([0.0, 0.0, 0.0, 1.0])
        rigid[0].startNode = 0
        rigid[0].endNode = 1
        rigid[0].localNode = 0
        rigid[0].m = 1.0
        rigid[0].equi_r = 0.1
        rigid[0].materialID = ti.u8(0)
        rigid[0].is_soft = ti.u8(0)
        vertice[0]._set_surface_node(ti.Vector([0.51, 0.51, -0.01]))
        vertice[0]._set_coefficient(1.0)
        box[0]._set_bounding_box(
            ti.Vector([-1.0, -1.0, -1.0]),
            ti.Vector([1.0, 1.0, 1.0]),
        )
        box[0]._add_grid(0, 0.1, ti.Vector([2, 2, 2]), 1.0, 0)

        wall[0].active = ti.u8(1)
        wall[0].wallID = 7
        wall[0].materialID = ti.u8(0)
        wall[0].vertice1 = ti.Vector([0.0, 0.0, 0.0])
        wall[0].vertice2 = ti.Vector([1.0, 0.0, 0.0])
        wall[0].vertice3 = ti.Vector([0.0, 1.0, 0.0])
        wall[0].norm = ti.Vector([0.0, 0.0, 1.0])
        wall[1].active = ti.u8(1)
        wall[1].wallID = 8
        wall[1].materialID = ti.u8(0)
        wall[1].vertice1 = ti.Vector([1.0, 1.0, 0.0])
        wall[1].vertice2 = ti.Vector([0.0, 1.0, 0.0])
        wall[1].vertice3 = ti.Vector([1.0, 0.0, 0.0])
        wall[1].norm = ti.Vector([0.0, 0.0, 1.0])

        for wall_id in ti.static(range(2)):
            broad_pairs[wall_id]._set_id(0, wall_id)
            contacts[wall_id]._set_id(0, wall_id)
        body_wall_prefix[0] = 0
        body_wall_prefix[1] = 2
        contact_prefix[0] = 0
        contact_prefix[1] = 2
        surface_body[0] = 0
        dt[None] = 1.0e-4

    initialize()
    properties[0].add_surface_property(
        1.0e3,
        5.0e2,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    board_search_lsparticle_wall_linked_cell_(
        1,
        2,
        5.0e-2,
        broad_pairs,
        potential_point_wall,
        body_wall_prefix,
        point_wall_count,
        wall,
        rigid,
        vertice,
        box,
    )

    assert int(point_wall_count[1]) == 2
    assert set(potential_point_wall.to_numpy().tolist()) == {0, 1}

    @ti.kernel
    def deactivate_wall_zero():
        wall[0].active = ti.u8(0)

    @ti.kernel
    def activate_wall_zero():
        wall[0].active = ti.u8(1)

    @ti.kernel
    def deactivate_wall_one():
        wall[1].active = ti.u8(0)

    @ti.kernel
    def activate_wall_one():
        wall[1].active = ti.u8(1)

    # Inactive facets are excluded even if they remain in a stale body-level
    # BVH/Verlet table.
    deactivate_wall_zero()
    board_search_lsparticle_wall_linked_cell_(
        1,
        2,
        5.0e-2,
        broad_pairs,
        potential_point_wall,
        body_wall_prefix,
        point_wall_count,
        wall,
        rigid,
        vertice,
        box,
    )
    assert int(point_wall_count[1]) == 1
    assert int(potential_point_wall[0]) == 1
    activate_wall_zero()

    kernel_LSparticle_wall_force_assemble_(
        1,
        dt,
        1,
        properties,
        rigid,
        vertice,
        surface_body,
        box,
        wall,
        contacts,
        contact_prefix,
        LSparticle_wall_contact_model,
    )

    forces = contacts.cnforce.to_numpy()
    np.testing.assert_allclose(forces[0], np.zeros(3), atol=0.0)
    assert forces[1, 2] > 0.0
    assert contacts.normalOverlap.to_numpy()[1] == pytest.approx(-0.01)
    assert int(contacts.normalOverlapActive.to_numpy()[1]) == 1

    kernel_accumulate_linear_lsparticle_wall_elastic_energy(
        1,
        8,
        1,
        properties,
        rigid,
        vertice,
        surface_body,
        box,
        wall,
        contacts,
        contact_prefix,
        removed_energy,
    )
    assert removed_energy[None] == pytest.approx(0.5 * 1.0e3 * 0.01**2)

    kernel_accumulate_linear_lsparticle_wall_elastic_energy(
        1,
        9,
        1,
        properties,
        rigid,
        vertice,
        surface_body,
        box,
        wall,
        contacts,
        contact_prefix,
        removed_energy,
    )
    assert removed_energy[None] == pytest.approx(0.0)

    # Force assembly is defensive as well: deactivation clears a contact
    # immediately, before any subsequent neighbor rebuild.
    deactivate_wall_one()
    kernel_LSparticle_wall_force_assemble_(
        1,
        dt,
        1,
        properties,
        rigid,
        vertice,
        surface_body,
        box,
        wall,
        contacts,
        contact_prefix,
        LSparticle_wall_contact_model,
    )
    np.testing.assert_allclose(contacts.cnforce.to_numpy()[1], np.zeros(3))
    activate_wall_one()

    @ti.kernel
    def set_deep_surface_node():
        vertice[0]._set_surface_node(ti.Vector([0.9, 0.9, -0.2]))

    @ti.kernel
    def set_far_surface_node():
        vertice[0]._set_surface_node(ti.Vector([1.5, 1.5, -0.2]))

    # A deeply penetrated point remains paired with the active finite face,
    # but the facet on the other side of the diagonal is not retained.
    set_deep_surface_node()
    board_search_lsparticle_wall_linked_cell_(
        1,
        2,
        5.0e-2,
        broad_pairs,
        potential_point_wall,
        body_wall_prefix,
        point_wall_count,
        wall,
        rigid,
        vertice,
        box,
    )
    assert int(point_wall_count[1]) == 1
    assert int(potential_point_wall[0]) == 1

    # A point far behind both finite triangles must not be retained merely
    # because its signed distance is negative.
    set_far_surface_node()
    board_search_lsparticle_wall_linked_cell_(
        1,
        2,
        5.0e-2,
        broad_pairs,
        potential_point_wall,
        body_wall_prefix,
        point_wall_count,
        wall,
        rigid,
        vertice,
        box,
    )
    assert int(point_wall_count[1]) == 0


def test_lsdem_point_wall_sweep_accumulates_on_device_before_flag_read():
    particle = ParticleFamily.field(shape=1)
    rigid = RigidBody.field(shape=1)
    wall = FacetFamily.field(shape=1)
    particle_wall = ti.field(dtype=ti.i32, shape=2)
    broad_pairs = VerletContactTable.field(shape=1)
    dt = ti.field(dtype=ti.f64, shape=())
    rebuild_required = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def initialize():
        particle[0].x = ti.Vector([0.0, 0.0, 1.0])
        particle[0].rad = 1.0
        rigid[0].v = ti.Vector([0.0, 0.0, -1.0])
        rigid[0].w = ti.Vector([0.0, 1.0, 0.0])
        wall[0].norm = ti.Vector([0.0, 0.0, 1.0])
        wall[0].v = ti.Vector.zero(ti.f64, 3)
        particle_wall[0] = 0
        particle_wall[1] = 1
        broad_pairs[0].endID1 = 0
        broad_pairs[0].endID2 = 0
        broad_pairs[0].verletDisp = ti.Vector.zero(ti.f64, 3)
        dt[None] = 0.1
        rebuild_required[None] = 0

    initialize()
    for _ in range(3):
        accumulate_pwrelative_displacement_(
            1,
            dt,
            particle,
            wall,
            rigid,
            particle_wall,
            broad_pairs,
        )

    np.testing.assert_allclose(
        broad_pairs.verletDisp.to_numpy()[0],
        np.array([-0.3, 0.0, -0.3]),
    )
    flag_relative_displacement_(
        0.17,
        1,
        particle_wall,
        broad_pairs,
        rebuild_required,
    )
    assert rebuild_required[None] == 1
