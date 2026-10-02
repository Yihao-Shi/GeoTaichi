from types import SimpleNamespace
import os

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.dem.structs.BaseStruct import (
    BoundingBox,
    FacetFamily,
    LevelSetGrid,
    RigidBody,
)
from src.fedem.Patch import FEMSurfacePatch
from src.fedem.contact.ContactKernel import (
    _barrier_contact_force,
    accumulate_linear_facet_wall_elastic_energy,
    build_compact_facet_wall_candidates,
    commit_facet_wall_search_state,
    commit_moving_facet_wall_search_state,
    measure_facet_wall_rebuild_requirement,
    measure_moving_facet_wall_rebuild_requirement,
    resolve_linear_facet_wall_contact,
    resolve_linear_levelset_contact,
)
from src.fedem.contact.Barrier import BarrierSurfaceProperty
from src.fedem.contact.Linear import LinearSurfaceProperty
from src.fedem.neighbor.LevelSet import FEMLevelSetBroadPhase
from src.fedem.structs import FEMFacetWallContact, FEMLevelSetContact
from src.utils.ScalarFunction import linearize3D


pytestmark = [
    pytest.mark.unit,
    pytest.mark.fedem,
    pytest.mark.contact,
    pytest.mark.cpu,
    pytest.mark.serial,
]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    arch = ti.cuda if os.environ.get("GEOTAICHI_TEST_ARCH") == "cuda" else ti.cpu
    ti.init(
        arch=arch,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
    )
    yield
    ti.reset()


@pytest.mark.parametrize("search", ["LinkedCell", "BVH"])
def test_levelset_broadphase_maps_rigid_node_ids_and_keeps_history(search):
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    position = ti.Vector.field(3, dtype=ti.f64, shape=3)
    position.from_numpy(nodes)
    patch = FEMSurfacePatch(
        3,
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array([0], dtype=np.int32),
    )
    patch.update(position)

    rigid = RigidBody.field(shape=2)
    box = BoundingBox.field(shape=2)

    @ti.kernel
    def initialize_rigid():
        rigid[0].mass_center = ti.Vector([0.0, 0.0, 0.05])
        rigid[0].q = ti.Vector([0.0, 0.0, 0.0, 1.0])
        box[0]._set_bounding_box(
            ti.Vector([-0.1, -0.1, -0.1]),
            ti.Vector([0.1, 0.1, 0.1]),
        )
        rigid[1].mass_center = ti.Vector([1.5, 1.5, 1.5])
        rigid[1].q = ti.Vector([0.0, 0.0, 0.0, 1.0])
        box[1]._set_bounding_box(
            ti.Vector([-0.1, -0.1, -0.1]),
            ti.Vector([0.1, 0.1, 0.1]),
        )

    initialize_rigid()
    scene = SimpleNamespace(
        particleNum=np.array([2], dtype=np.int32),
        rigid=rigid,
        box=box,
        # Deliberately much larger than body 0: candidates must still use
        # each current rigid AABB rather than one global query radius.
        find_bounding_sphere_max_radius=lambda _: 10.0,
    )
    simulation = SimpleNamespace(
        search=search,
        max_contact_pairs=8,
        max_levelset_cell_pairs=64,
        domain=np.array([2.0, 2.0, 2.0]),
        verlet_distance=0.2,
    )
    broad_phase = FEMLevelSetBroadPhase(simulation, patch, SimpleNamespace(), scene)

    assert broad_phase.rebuild(position, rigid, box) == 1
    assert int(broad_phase.contacts.rigid_id.to_numpy()[0]) == 0
    assert int(broad_phase.contacts.node_id.to_numpy()[0]) == 0

    expected = np.arange(3, dtype=np.float64)
    broad_phase.contacts.old_tangential_overlap.from_numpy(np.pad(expected[None], ((0, 7), (0, 0))))
    assert broad_phase.rebuild(position, rigid, box) == 1
    np.testing.assert_allclose(broad_phase.contacts.old_tangential_overlap.to_numpy()[0], expected)


def test_fedem_barrier_matches_dem_normal_potential_and_force():
    prop = BarrierSurfaceProperty.field(shape=1)
    force = ti.Vector.field(3, dtype=ti.f64, shape=())
    energy = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate():
        prop[0].active = 1
        prop[0].kappa = 3.0
        prop[0].normal_cutoff = 0.2
        prop[0].stiffness_ratio = 1.0
        prop[0].friction = 0.0
        prop[0].normal_damping = 0.0
        prop[0].tangential_damping = 0.0
        normal_force, _, _, stored, _, _ = _barrier_contact_force(
            prop[0],
            0.0,
            2.0,
            1.0,
            ti.Vector([0.0, 0.0, 1.0]),
            0.0,
            ti.Vector.zero(ti.f64, 3),
            ti.Vector.zero(ti.f64, 3),
            1.0e-3,
            True,
        )
        force[None] = normal_force
        energy[None] = stored

    evaluate()
    eta = 0.2
    d_cap = 0.4
    coefficient = 2.0
    kappa = 3.0
    expected_force = kappa * coefficient * (eta - d_cap) * (2.0 * np.log(eta / d_cap) - d_cap / eta + 1.0)
    expected_energy = -kappa * coefficient * (eta - d_cap) ** 2 * np.log(eta / d_cap)
    np.testing.assert_allclose(force.to_numpy(), [0.0, 0.0, expected_force], rtol=2.0e-7)
    assert float(energy[None]) == pytest.approx(expected_energy, rel=2.0e-7)


@pytest.mark.parametrize("search", ["LinkedCell", "BVH"])
def test_levelset_rebuild_uses_fem_translation_and_rotation_verlet_motion(search):
    nodes = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]])
    position = ti.Vector.field(3, dtype=ti.f64, shape=3)
    position.from_numpy(nodes)
    patch = FEMSurfacePatch(
        3,
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array([0], dtype=np.int32),
    )
    patch.update(position)

    rigid = RigidBody.field(shape=1)
    box = BoundingBox.field(shape=1)

    @ti.kernel
    def initialize_rigid():
        rigid[0].mass_center = ti.Vector([0.0, 0.0, 0.0])
        rigid[0].q = ti.Vector([0.0, 0.0, 0.0, 1.0])
        rigid[0].equi_r = 1.0
        box[0]._set_bounding_box(
            ti.Vector([-0.1, -0.1, -0.1]),
            ti.Vector([0.1, 0.1, 0.1]),
        )

    @ti.kernel
    def set_rigid_pose(x: float, half_angle: float):
        rigid[0].mass_center = ti.Vector([x, 0.0, 0.0])
        rigid[0].q = ti.Vector([0.0, 0.0, ti.sin(half_angle), ti.cos(half_angle)])

    initialize_rigid()
    scene = SimpleNamespace(
        particleNum=np.array([1], dtype=np.int32),
        rigid=rigid,
        box=box,
        find_bounding_sphere_max_radius=lambda _: 1.0,
    )
    simulation = SimpleNamespace(
        search=search,
        max_contact_pairs=8,
        max_levelset_cell_pairs=64,
        domain=np.array([2.0, 2.0, 2.0]),
        verlet_distance=0.2,
    )
    broad_phase = FEMLevelSetBroadPhase(simulation, patch, SimpleNamespace(), scene)
    broad_phase.rebuild(position, rigid, box)
    assert broad_phase.requires_rebuild(rigid) is False

    position.from_numpy(nodes + np.array([0.09, 0.0, 0.0]))
    patch.update(position, update_normals=False)
    assert broad_phase.requires_rebuild(rigid) is False
    position.from_numpy(nodes + np.array([0.11, 0.0, 0.0]))
    patch.update(position, update_normals=False)
    assert broad_phase.requires_rebuild(rigid) is True

    position.from_numpy(nodes)
    patch.update(position, update_normals=False)
    broad_phase.rebuild(position, rigid, box)
    set_rigid_pose(0.09, 0.0)
    assert broad_phase.requires_rebuild(rigid) is False
    set_rigid_pose(0.11, 0.0)
    assert broad_phase.requires_rebuild(rigid) is True

    set_rigid_pose(0.0, 0.0)
    broad_phase.rebuild(position, rigid, box)
    position.from_numpy(nodes + np.array([0.25, 0.0, 0.0]))
    patch.update(position, update_normals=False)
    set_rigid_pose(0.25, 0.0)
    assert broad_phase.requires_rebuild(rigid) is False

    position.from_numpy(nodes)
    patch.update(position, update_normals=False)
    set_rigid_pose(0.0, 0.0)
    broad_phase.rebuild(position, rigid, box)
    set_rigid_pose(0.0, 0.04)
    assert broad_phase.requires_rebuild(rigid) is False
    set_rigid_pose(0.0, 0.06)
    assert broad_phase.requires_rebuild(rigid) is True


def test_fixed_facet_wall_candidates_are_compact_and_verlet_bounded():
    nodes = np.array(
        [[0.2, 0.2, 0.01], [2.0, 2.0, 0.01], [0.2, 0.2, 0.5]],
        dtype=np.float64,
    )
    positions = ti.Vector.field(3, dtype=ti.f64, shape=3)
    positions.from_numpy(nodes)
    surface_vertices = ti.field(dtype=ti.i32, shape=3)
    surface_vertices.from_numpy(np.arange(3, dtype=np.int32))
    wall = FacetFamily.field(shape=1)
    candidate_count = ti.field(dtype=ti.i32, shape=())
    candidate_pairs = ti.field(dtype=ti.i32, shape=3)
    contacts = FEMFacetWallContact.field(shape=3)
    history_state = ti.field(dtype=ti.i32, shape=4)
    history_keys = ti.field(dtype=ti.i64, shape=4)
    history_gaps = ti.field(dtype=ti.f64, shape=4)
    history_overlaps = ti.Vector.field(3, dtype=ti.f64, shape=4)
    search_nodes = ti.Vector.field(3, dtype=ti.f64, shape=3)
    rebuild_required = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def initialize_wall():
        wall[0].active = 1
        wall[0].vertice1 = ti.Vector([0.0, 0.0, 0.0])
        wall[0].vertice2 = ti.Vector([1.0, 0.0, 0.0])
        wall[0].vertice3 = ti.Vector([0.0, 1.0, 0.0])
        wall[0].norm = ti.Vector([0.0, 0.0, 1.0])

    @ti.kernel
    def translate_nodes(value: float):
        for node in range(3):
            positions[node][0] += value

    @ti.kernel
    def move_second_node(value: float):
        positions[1][0] += value

    initialize_wall()
    history_state.fill(2)
    build_compact_facet_wall_candidates(
        3,
        1,
        0.05,
        wall,
        positions,
        surface_vertices,
        candidate_count,
        3,
        candidate_pairs,
        contacts,
        4,
        history_state,
        history_keys,
        history_gaps,
        history_overlaps,
    )
    assert candidate_count[None] == 1
    assert candidate_pairs[0] == 0

    commit_facet_wall_search_state(
        3,
        positions,
        surface_vertices,
        search_nodes,
    )
    translate_nodes(0.02)
    measure_facet_wall_rebuild_requirement(
        3,
        0.025,
        positions,
        surface_vertices,
        search_nodes,
        rebuild_required,
    )
    assert rebuild_required[None] == 0

    move_second_node(0.03)
    measure_facet_wall_rebuild_requirement(
        3,
        0.025,
        positions,
        surface_vertices,
        search_nodes,
        rebuild_required,
    )
    assert rebuild_required[None] == 1


def test_moving_facet_wall_rebuild_uses_two_sided_verlet_sweep_bound():
    positions = ti.Vector.field(3, dtype=ti.f64, shape=1)
    surface_vertices = ti.field(dtype=ti.i32, shape=1)
    wall = FacetFamily.field(shape=1)
    search_nodes = ti.Vector.field(3, dtype=ti.f64, shape=1)
    search_wall_centers = ti.Vector.field(3, dtype=ti.f64, shape=1)
    maximum_node_sweep = ti.field(dtype=ti.f64, shape=())
    maximum_wall_sweep = ti.field(dtype=ti.f64, shape=())
    rebuild_required = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def initialize():
        positions[0] = ti.Vector([0.25, 0.25, 0.01])
        surface_vertices[0] = 0
        wall[0].active = 1
        wall[0].vertice1 = ti.Vector([0.0, 0.0, 0.0])
        wall[0].vertice2 = ti.Vector([1.0, 0.0, 0.0])
        wall[0].vertice3 = ti.Vector([0.0, 1.0, 0.0])
        wall[0].norm = ti.Vector([0.0, 0.0, 1.0])

    @ti.kernel
    def move_node(distance: float):
        positions[0][2] += distance

    @ti.kernel
    def move_wall(distance: float):
        wall[0].vertice1[2] += distance
        wall[0].vertice2[2] += distance
        wall[0].vertice3[2] += distance

    initialize()
    commit_moving_facet_wall_search_state(
        1,
        1,
        positions,
        surface_vertices,
        wall,
        search_nodes,
        search_wall_centers,
    )
    move_node(0.02)
    move_wall(-0.02)
    measure_moving_facet_wall_rebuild_requirement(
        1,
        1,
        0.05,
        positions,
        surface_vertices,
        wall,
        search_nodes,
        search_wall_centers,
        maximum_node_sweep,
        maximum_wall_sweep,
        rebuild_required,
    )
    assert maximum_node_sweep[None] == pytest.approx(0.02)
    assert maximum_wall_sweep[None] == pytest.approx(0.02)
    assert rebuild_required[None] == 0

    move_wall(-0.02)
    measure_moving_facet_wall_rebuild_requirement(
        1,
        1,
        0.05,
        positions,
        surface_vertices,
        wall,
        search_nodes,
        search_wall_centers,
        maximum_node_sweep,
        maximum_wall_sweep,
        rebuild_required,
    )
    assert rebuild_required[None] == 1


def test_fixed_facet_wall_requires_verlet_entry_and_retains_active_penetration():
    positions = ti.Vector.field(3, dtype=ti.f64, shape=1)
    surface_vertices = ti.field(dtype=ti.i32, shape=1)
    wall = FacetFamily.field(shape=1)
    candidate_count = ti.field(dtype=ti.i32, shape=())
    candidate_pairs = ti.field(dtype=ti.i32, shape=1)
    contacts = FEMFacetWallContact.field(shape=1)
    history_state = ti.field(dtype=ti.i32, shape=4)
    history_keys = ti.field(dtype=ti.i64, shape=4)
    history_gaps = ti.field(dtype=ti.f64, shape=4)
    history_overlaps = ti.Vector.field(3, dtype=ti.f64, shape=4)

    @ti.kernel
    def initialize():
        positions[0] = ti.Vector([0.25, 0.25, -0.2])
        surface_vertices[0] = 0
        wall[0].active = 1
        wall[0].vertice1 = ti.Vector([0.0, 0.0, 0.0])
        wall[0].vertice2 = ti.Vector([1.0, 0.0, 0.0])
        wall[0].vertice3 = ti.Vector([0.0, 1.0, 0.0])
        wall[0].norm = ti.Vector([0.0, 0.0, 1.0])
        history_state[0] = 0
        history_keys[0] = 0

    @ti.kernel
    def set_previous_gap(value: float):
        history_gaps[0] = value

    @ti.kernel
    def move_to_positive_side():
        positions[0] = ti.Vector([0.25, 0.25, 0.2])

    initialize()
    build_compact_facet_wall_candidates(
        1,
        1,
        0.05,
        wall,
        positions,
        surface_vertices,
        candidate_count,
        1,
        candidate_pairs,
        contacts,
        4,
        history_state,
        history_keys,
        history_gaps,
        history_overlaps,
    )
    assert candidate_count[None] == 0

    set_previous_gap(-0.01)
    build_compact_facet_wall_candidates(
        1,
        1,
        0.05,
        wall,
        positions,
        surface_vertices,
        candidate_count,
        1,
        candidate_pairs,
        contacts,
        4,
        history_state,
        history_keys,
        history_gaps,
        history_overlaps,
    )
    assert candidate_count[None] == 1
    assert candidate_pairs[0] == 0

    move_to_positive_side()
    build_compact_facet_wall_candidates(
        1,
        1,
        0.05,
        wall,
        positions,
        surface_vertices,
        candidate_count,
        1,
        candidate_pairs,
        contacts,
        4,
        history_state,
        history_keys,
        history_gaps,
        history_overlaps,
    )
    assert candidate_count[None] == 0


def test_total_lagrangian_patch_keeps_reference_contact_weights():
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    positions = ti.Vector.field(3, dtype=ti.f64, shape=3)
    positions.from_numpy(reference)
    patch = FEMSurfacePatch(
        3,
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array([0], dtype=np.int32),
    )
    patch.update(positions)
    reference_weights = patch.node_area.to_numpy()

    deformed = reference.copy()
    deformed[1, 0] = 2.0
    positions.from_numpy(deformed)
    patch.update(positions, update_area=False)

    np.testing.assert_allclose(patch.node_area.to_numpy(), reference_weights)
    patch.update(positions, update_area=True)
    np.testing.assert_allclose(patch.node_area.to_numpy(), 2.0 * reference_weights)


def test_facet_wall_unilateral_cutoff_uses_applied_dashpot_work():
    positions = ti.Vector.field(3, dtype=ti.f64, shape=1)
    node_body = ti.field(dtype=ti.i32, shape=1)
    node_area = ti.field(dtype=ti.f64, shape=1)
    velocity = ti.Vector.field(3, dtype=ti.f64, shape=1)
    mass = ti.field(dtype=ti.f64, shape=1)
    force = ti.Vector.field(3, dtype=ti.f64, shape=1)
    wall = FacetFamily.field(shape=1)
    properties = LinearSurfaceProperty.field(shape=(1, 1))
    contacts = FEMFacetWallContact.field(shape=1)
    candidate_pairs = ti.field(dtype=ti.i32, shape=1)
    elastic_energy = ti.field(dtype=ti.f64, shape=())
    friction_dissipation = ti.field(dtype=ti.f64, shape=())
    damping_dissipation = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def initialize():
        positions[0] = ti.Vector([0.25, 0.25, -0.01])
        node_body[0] = 0
        node_area[0] = 1.0
        velocity[0] = ti.Vector([0.0, 0.0, 1.0])
        mass[0] = 1.0
        force[0] = ti.Vector.zero(ti.f64, 3)
        wall[0].active = 1
        wall[0].materialID = 0
        wall[0].vertice1 = ti.Vector([0.0, 0.0, 0.0])
        wall[0].vertice2 = ti.Vector([1.0, 0.0, 0.0])
        wall[0].vertice3 = ti.Vector([0.0, 1.0, 0.0])
        wall[0].norm = ti.Vector([0.0, 0.0, 1.0])
        wall[0].v = ti.Vector.zero(ti.f64, 3)
        candidate_pairs[0] = 0

    initialize()
    properties[0, 0] = {
        "active": 1,
        "kn": 100.0,
        "ks": 50.0,
        "friction": 0.0,
        "normal_damping": 1.0,
        "tangential_damping": 0.0,
    }
    resolve_linear_facet_wall_contact(
        1,
        1,
        candidate_pairs,
        0.1,
        properties,
        wall,
        positions,
        node_body,
        node_area,
        velocity,
        mass,
        force,
        contacts,
        elastic_energy,
        friction_dissipation,
        damping_dissipation,
        True,
    )

    # The 1 N compressed spring is cancelled by the capped dashpot, so the
    # applied contact force is zero.  Its 0.1 J unloading work remains in the
    # damping ledger instead of using the much larger raw dashpot trial.
    np.testing.assert_allclose(force.to_numpy()[0], np.zeros(3), atol=0.0)
    assert elastic_energy[None] == pytest.approx(0.5 * 100.0 * 0.01**2)
    assert damping_dissipation[None] == pytest.approx(0.1)

    elastic_energy[None] = 3.0
    friction_dissipation[None] = 4.0
    damping_dissipation[None] = 5.0
    force.fill(0.0)
    resolve_linear_facet_wall_contact(
        1,
        1,
        candidate_pairs,
        0.1,
        properties,
        wall,
        positions,
        node_body,
        node_area,
        velocity,
        mass,
        force,
        contacts,
        elastic_energy,
        friction_dissipation,
        damping_dissipation,
        False,
    )
    assert elastic_energy[None] == pytest.approx(3.0)
    assert friction_dissipation[None] == pytest.approx(4.0)
    assert damping_dissipation[None] == pytest.approx(5.0)


def test_facet_wall_does_not_count_coulomb_clipped_dashpot_trial():
    positions = ti.Vector.field(3, dtype=ti.f64, shape=1)
    node_body = ti.field(dtype=ti.i32, shape=1)
    node_area = ti.field(dtype=ti.f64, shape=1)
    velocity = ti.Vector.field(3, dtype=ti.f64, shape=1)
    mass = ti.field(dtype=ti.f64, shape=1)
    force = ti.Vector.field(3, dtype=ti.f64, shape=1)
    wall = FacetFamily.field(shape=1)
    properties = LinearSurfaceProperty.field(shape=(1, 1))
    contacts = FEMFacetWallContact.field(shape=1)
    candidate_pairs = ti.field(dtype=ti.i32, shape=1)
    elastic_energy = ti.field(dtype=ti.f64, shape=())
    friction_dissipation = ti.field(dtype=ti.f64, shape=())
    damping_dissipation = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def initialize():
        positions[0] = ti.Vector([0.25, 0.25, -0.01])
        node_body[0] = 0
        node_area[0] = 1.0
        velocity[0] = ti.Vector([1.0, 0.0, -0.1])
        mass[0] = 1.0
        force[0] = ti.Vector.zero(ti.f64, 3)
        wall[0].active = 1
        wall[0].materialID = 0
        wall[0].vertice1 = ti.Vector([0.0, 0.0, 0.0])
        wall[0].vertice2 = ti.Vector([1.0, 0.0, 0.0])
        wall[0].vertice3 = ti.Vector([0.0, 1.0, 0.0])
        wall[0].norm = ti.Vector([0.0, 0.0, 1.0])
        wall[0].v = ti.Vector.zero(ti.f64, 3)
        candidate_pairs[0] = 0

    initialize()
    properties[0, 0] = {
        "active": 1,
        "kn": 100.0,
        "ks": 50.0,
        "friction": 0.0,
        "normal_damping": 0.0,
        "tangential_damping": 1.0,
    }
    resolve_linear_facet_wall_contact(
        1,
        1,
        candidate_pairs,
        0.1,
        properties,
        wall,
        positions,
        node_body,
        node_area,
        velocity,
        mass,
        force,
        contacts,
        elastic_energy,
        friction_dissipation,
        damping_dissipation,
        True,
    )

    # With mu=0 the Coulomb cap removes the entire tangential trial force.
    # Its dashpot therefore performs no actual work and contributes no loss.
    np.testing.assert_allclose(force.to_numpy()[0, :2], np.zeros(2), atol=0.0)
    assert damping_dissipation[None] == pytest.approx(0.0)


def test_facet_wall_uses_dem_elastic_trial_for_stick_slip_branch():
    positions = ti.Vector.field(3, dtype=ti.f64, shape=1)
    node_body = ti.field(dtype=ti.i32, shape=1)
    node_area = ti.field(dtype=ti.f64, shape=1)
    velocity = ti.Vector.field(3, dtype=ti.f64, shape=1)
    mass = ti.field(dtype=ti.f64, shape=1)
    force = ti.Vector.field(3, dtype=ti.f64, shape=1)
    wall = FacetFamily.field(shape=1)
    properties = LinearSurfaceProperty.field(shape=(1, 1))
    contacts = FEMFacetWallContact.field(shape=1)
    candidate_pairs = ti.field(dtype=ti.i32, shape=1)
    elastic_energy = ti.field(dtype=ti.f64, shape=())
    friction_dissipation = ti.field(dtype=ti.f64, shape=())
    damping_dissipation = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def initialize():
        positions[0] = ti.Vector([0.25, 0.25, -0.01])
        node_body[0] = 0
        node_area[0] = 1.0
        velocity[0] = ti.Vector([1.0, 0.0, 0.0])
        mass[0] = 1.0
        force[0] = ti.Vector.zero(ti.f64, 3)
        wall[0].active = 1
        wall[0].materialID = 0
        wall[0].vertice1 = ti.Vector([0.0, 0.0, 0.0])
        wall[0].vertice2 = ti.Vector([1.0, 0.0, 0.0])
        wall[0].vertice3 = ti.Vector([0.0, 1.0, 0.0])
        wall[0].norm = ti.Vector([0.0, 0.0, 1.0])
        wall[0].v = ti.Vector.zero(ti.f64, 3)
        candidate_pairs[0] = 0

    initialize()
    properties[0, 0] = {
        "active": 1,
        "kn": 100.0,
        "ks": 100.0,
        "friction": 0.5,
        "normal_damping": 0.0,
        "tangential_damping": 1.0,
    }
    dt = 1.0e-4
    resolve_linear_facet_wall_contact(
        1,
        1,
        candidate_pairs,
        dt,
        properties,
        wall,
        positions,
        node_body,
        node_area,
        velocity,
        mass,
        force,
        contacts,
        elastic_energy,
        friction_dissipation,
        damping_dissipation,
        True,
    )

    # This is the branch used by DEM LinearSurfaceProperty._tangential_force:
    # the 0.01 N elastic trial is inside the 0.5 N Coulomb limit, so the
    # overlap remains elastic and the 20 N dashpot is added as sticking work.
    # A spring-plus-dashpot branch test would incorrectly return-map here.
    np.testing.assert_allclose(
        contacts.old_tangential_overlap.to_numpy()[0],
        [dt, 0.0, 0.0],
        atol=1.0e-15,
    )
    assert contacts.tangential_force.to_numpy()[0, 0] == pytest.approx(-20.01)
    assert friction_dissipation[None] == pytest.approx(0.0)
    assert damping_dissipation[None] == pytest.approx(20.0 * dt)


def test_facet_wall_sliding_return_map_matches_dem_work_split():
    positions = ti.Vector.field(3, dtype=ti.f64, shape=1)
    node_body = ti.field(dtype=ti.i32, shape=1)
    node_area = ti.field(dtype=ti.f64, shape=1)
    velocity = ti.Vector.field(3, dtype=ti.f64, shape=1)
    mass = ti.field(dtype=ti.f64, shape=1)
    force = ti.Vector.field(3, dtype=ti.f64, shape=1)
    wall = FacetFamily.field(shape=1)
    properties = LinearSurfaceProperty.field(shape=(1, 1))
    contacts = FEMFacetWallContact.field(shape=1)
    candidate_pairs = ti.field(dtype=ti.i32, shape=1)
    elastic_energy = ti.field(dtype=ti.f64, shape=())
    friction_dissipation = ti.field(dtype=ti.f64, shape=())
    damping_dissipation = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def initialize():
        positions[0] = ti.Vector([0.25, 0.25, -0.01])
        node_body[0] = 0
        node_area[0] = 1.0
        velocity[0] = ti.Vector([1.0, 0.0, 0.0])
        mass[0] = 1.0
        force[0] = ti.Vector.zero(ti.f64, 3)
        wall[0].active = 1
        wall[0].materialID = 0
        wall[0].vertice1 = ti.Vector([0.0, 0.0, 0.0])
        wall[0].vertice2 = ti.Vector([1.0, 0.0, 0.0])
        wall[0].vertice3 = ti.Vector([0.0, 1.0, 0.0])
        wall[0].norm = ti.Vector([0.0, 0.0, 1.0])
        wall[0].v = ti.Vector.zero(ti.f64, 3)
        candidate_pairs[0] = 0

    initialize()
    properties[0, 0] = {
        "active": 1,
        "kn": 100.0,
        "ks": 100.0,
        "friction": 0.5,
        "normal_damping": 0.0,
        "tangential_damping": 1.0,
    }
    resolve_linear_facet_wall_contact(
        1,
        1,
        candidate_pairs,
        0.1,
        properties,
        wall,
        positions,
        node_body,
        node_area,
        velocity,
        mass,
        force,
        contacts,
        elastic_energy,
        friction_dissipation,
        damping_dissipation,
        True,
    )

    # DEM first caps the elastic trial at mu Fn, return-maps the spring, and
    # assigns only the discarded overlap to irreversible frictional work.
    # The tangential dashpot is inactive on this sliding branch.
    np.testing.assert_allclose(
        contacts.old_tangential_overlap.to_numpy()[0],
        [0.005, 0.0, 0.0],
        atol=1.0e-15,
    )
    assert contacts.tangential_force.to_numpy()[0, 0] == pytest.approx(-0.5)
    assert friction_dissipation[None] == pytest.approx(0.5 * 0.095)
    assert damping_dissipation[None] == pytest.approx(0.0)
    assert elastic_energy[None] == pytest.approx(0.5 * 100.0 * (0.01**2 + 0.005**2))


def test_facet_wall_removal_energy_is_group_specific_penalty_energy():
    positions = ti.Vector.field(3, dtype=ti.f64, shape=1)
    node_body = ti.field(dtype=ti.i32, shape=1)
    node_area = ti.field(dtype=ti.f64, shape=1)
    wall = FacetFamily.field(shape=1)
    properties = LinearSurfaceProperty.field(shape=(1, 1))
    contacts = FEMFacetWallContact.field(shape=1)
    candidate_pairs = ti.field(dtype=ti.i32, shape=1)
    removed_energy = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def initialize():
        positions[0] = ti.Vector([0.25, 0.25, -0.02])
        node_body[0] = 0
        node_area[0] = 0.4
        wall[0].active = 1
        wall[0].wallID = 7
        wall[0].materialID = 0
        wall[0].vertice1 = ti.Vector([0.0, 0.0, 0.0])
        wall[0].vertice2 = ti.Vector([1.0, 0.0, 0.0])
        wall[0].vertice3 = ti.Vector([0.0, 1.0, 0.0])
        wall[0].norm = ti.Vector([0.0, 0.0, 1.0])
        candidate_pairs[0] = 0
        contacts[0].old_tangential_overlap = ti.Vector([0.01, 0.0, 0.0])

    initialize()
    properties[0, 0] = {
        "active": 1,
        "kn": 100.0,
        "ks": 50.0,
        "friction": 0.0,
        "normal_damping": 0.0,
        "tangential_damping": 0.0,
    }
    arguments = (
        1,
        1,
        7,
        candidate_pairs,
        properties,
        wall,
        positions,
        node_body,
        node_area,
        contacts,
        removed_energy,
    )
    accumulate_linear_facet_wall_elastic_energy(*arguments)
    expected = 0.5 * 0.4 * (100.0 * 0.02**2 + 50.0 * 0.01**2)
    assert removed_energy[None] == pytest.approx(expected)

    accumulate_linear_facet_wall_elastic_energy(*arguments[:2], 8, *arguments[3:])
    assert removed_energy[None] == pytest.approx(0.0)


def test_levelset_surface_node_contact_is_work_conjugate_and_preserves_action_reaction():
    nodes = np.array([[0.2, 0.2, 0.05], [0.8, 0.2, 0.05], [0.2, 0.8, 0.05]])
    patch = FEMSurfacePatch(
        3,
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array([0], dtype=np.int32),
    )
    position = ti.Vector.field(3, dtype=ti.f64, shape=3)
    velocity = ti.Vector.field(3, dtype=ti.f64, shape=3)
    mass = ti.field(dtype=ti.f64, shape=3)
    force = ti.Vector.field(3, dtype=ti.f64, shape=3)
    position.from_numpy(nodes)
    velocity.fill(0.0)
    mass.fill(1.0)
    force.fill(0.0)
    patch.update(position)

    rigid = RigidBody.field(shape=1)
    box = BoundingBox.field(shape=1)
    grid = LevelSetGrid.field(shape=8)
    properties = LinearSurfaceProperty.field(shape=(1, 1))
    contacts = FEMLevelSetContact.field(shape=1)
    friction_dissipation = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def initialize():
        rigid[0].mass_center = ti.Vector([0.0, 0.0, 0.0])
        rigid[0].m = 1.0
        rigid[0].materialID = 0
        rigid[0].q = ti.Vector([0.0, 0.0, 0.0, 1.0])
        rigid[0].v = ti.Vector.zero(ti.f64, 3)
        rigid[0].w = ti.Vector.zero(ti.f64, 3)
        rigid[0].contact_force = ti.Vector.zero(ti.f64, 3)
        rigid[0].contact_torque = ti.Vector.zero(ti.f64, 3)
        box[0]._set_bounding_box(
            ti.Vector([0.0, 0.0, 0.0]),
            ti.Vector([1.0, 1.0, 1.0]),
        )
        box[0]._add_grid(0, 1.0, ti.Vector([2, 2, 2]), 1.0, 0)
        for i, j, k in ti.ndrange(2, 2, 2):
            node = linearize3D(i, j, k, ti.Vector([2, 2, 2]))
            # phi(z) = 2 z - 0.2 deliberately has a non-unit gradient.
            # The elastic force must equal -d(0.5 * kn * area * phi^2)/dz.
            grid[node].distance_field = 2.0 * ti.cast(k, ti.f64) - 0.2
        contacts[0].rigid_id = 0
        contacts[0].node_id = 0
        contacts[0].old_tangential_overlap = ti.Vector([0.01, 0.0, 0.0])
        friction_dissipation[None] = 0.0

    initialize()
    properties[0, 0] = {
        "active": 1,
        "kn": 1000.0,
        "ks": 500.0,
        "friction": 10.0,
        "normal_damping": 0.0,
        "tangential_damping": 0.0,
    }
    resolve_linear_levelset_contact(
        1,
        1.0e-3,
        properties,
        rigid,
        box,
        grid,
        patch.nodes,
        patch.node_body,
        patch.node_area,
        velocity,
        mass,
        force,
        contacts,
        friction_dissipation,
    )

    fem_force = force.to_numpy().sum(axis=0)
    rigid_force = rigid.contact_force.to_numpy()[0]
    expected = (0.5 * 0.6 * 0.6 / 3.0) * 1000.0 * 0.1 * 2.0
    assert int(contacts.active.to_numpy()[0]) == 1
    assert contacts.normal_gap.to_numpy()[0] == pytest.approx(-0.1)
    assert fem_force[2] == pytest.approx(expected)
    assert fem_force[0] == pytest.approx(-0.3)
    np.testing.assert_allclose(fem_force + rigid_force, 0.0, atol=1.0e-13)
    fem_moment = np.cross(nodes[0], fem_force)
    rigid_moment = rigid.contact_torque.to_numpy()[0]
    np.testing.assert_allclose(
        fem_moment + rigid_moment,
        np.zeros(3),
        atol=1.0e-13,
    )
