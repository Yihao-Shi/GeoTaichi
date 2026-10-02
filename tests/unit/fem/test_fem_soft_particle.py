import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem.engines.FEMState import FEMState
from src.fem import FEM
from src.fem.generator import FEMMesh
from src.fem.soft_particle import FEMSoftParticleContactModel
from src.fem.soft_particle.ContactKernel import _contact_force
from src.fem.soft_particle.ContactManager import FEMSoftParticleContactManager
from src.fem.soft_particle.ContactModel import FEMSoftParticleProperty


pytestmark = [
    pytest.mark.unit,
    pytest.mark.fem,
    pytest.mark.contact,
    pytest.mark.cpu,
    pytest.mark.serial,
]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
    )
    yield
    ti.reset()


def _two_facing_tetrahedra():
    lower = FEMMesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        ),
        np.array([[0, 1, 2, 3]], dtype=np.int32),
        "TET4",
    )
    upper = FEMMesh(
        np.array(
            [
                [0.0, 0.0, 0.05],
                [1.0, 0.0, 0.05],
                [0.0, 1.0, 0.05],
                [0.0, 0.0, 1.05],
            ]
        ),
        np.array([[0, 1, 2, 3]], dtype=np.int32),
        "TET4",
    )
    return FEMMesh.concatenate((lower, upper))


def _two_separated_offset_cubes():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cells = np.array(
        [
            [0, 1, 3, 4],
            [1, 2, 3, 6],
            [1, 3, 4, 6],
            [1, 4, 5, 6],
            [3, 4, 6, 7],
        ],
        dtype=np.int32,
    )
    first = FEMMesh(points, cells, "TET4")
    second = FEMMesh(points + np.array([1.35, 0.43, 0.0]), cells, "TET4")
    return FEMMesh.concatenate((first, second))


def test_soft_particle_pair_capacities_use_surface_triangle_coordination():
    model = FEMSoftParticleContactModel(
        "Linear",
        raw_point_triangle_coordination_number=2.5,
        raw_edge_edge_coordination_number=8.0,
        point_triangle_coordination_number=1.25,
        edge_edge_coordination_number=4.5,
    )

    assert model.pair_capacities(10) == {
        "raw_point_triangle": 25,
        "raw_edge_edge": 80,
        "verlet_point_triangle": 13,
        "verlet_edge_edge": 45,
    }


def test_soft_particle_absolute_capacity_overrides_coordination():
    model = FEMSoftParticleContactModel(
        "Linear",
        max_point_triangle_pairs=7,
        raw_point_triangle_coordination_number=20.0,
    )

    assert model.pair_capacities(10)["raw_point_triangle"] == 7


def test_soft_contact_return_map_records_unloading_plastic_slip():
    prop = FEMSoftParticleProperty.field(shape=1)
    force = ti.Vector.field(3, dtype=ti.f64, shape=())
    overlap = ti.Vector.field(3, dtype=ti.f64, shape=())
    friction = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate():
        prop[0].active = 1
        prop[0].friction = 0.3
        prop[0].kn = 100.0
        prop[0].ks = 100.0
        prop[0].normal_damping = 0.0
        prop[0].tangential_damping = 0.0
        _, tangential_force, current, friction_increment = _contact_force(
            0,
            prop[0],
            0.01,
            1.0,
            1.0,
            ti.Vector([0.0, 0.0, 1.0]),
            0.0,
            ti.Vector([0.0, 0.0, 0.0]),
            ti.Vector([0.01, 0.0, 0.0]),
            0.1,
        )
        force[None] = tangential_force
        overlap[None] = current
        friction[None] = friction_increment

    evaluate()
    assert force[None][0] == pytest.approx(-0.3)
    assert overlap[None][0] == pytest.approx(0.003)
    assert friction[None] == pytest.approx(0.3 * 0.007)


def test_soft_particle_barrier_matches_dem_normal_force():
    prop = FEMSoftParticleProperty.field(shape=1)
    force = ti.Vector.field(3, dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate():
        prop[0].active = 1
        prop[0].friction = 0.0
        prop[0].normal_damping = 0.0
        prop[0].tangential_damping = 0.0
        prop[0].barrier_kappa = 3.0
        prop[0].barrier_cutoff = 0.2
        prop[0].barrier_stiffness_ratio = 1.0
        normal_force, _, _, _ = _contact_force(
            2,
            prop[0],
            0.0,
            2.0,
            1.0,
            ti.Vector([0.0, 0.0, 1.0]),
            0.0,
            ti.Vector.zero(ti.f64, 3),
            ti.Vector.zero(ti.f64, 3),
            1.0e-3,
        )
        force[None] = normal_force

    evaluate()
    eta = 0.2
    d_cap = 0.4
    expected = 3.0 * 2.0 * (eta - d_cap) * (2.0 * np.log(eta / d_cap) - d_cap / eta + 1.0)
    np.testing.assert_allclose(force.to_numpy(), [0.0, 0.0, expected], rtol=2.0e-7)


def test_soft_particle_barrier_resolve_is_finite_and_balanced():
    mesh = _two_facing_tetrahedra()
    state = FEMState(
        mesh.points,
        np.zeros_like(mesh.points),
        np.ones(mesh.number_of_nodes),
    )
    model = FEMSoftParticleContactModel(
        "Barrier",
        search="BVH",
        verlet_distance=0.05,
        ContactThickness=0.05,
        Stiffness=1.0e4,
        NormalCutOff=0.05,
        StiffnessRatio=1.0,
        Friction=0.0,
    )
    manager = FEMSoftParticleContactManager(mesh, state, model)
    state.external_force.fill(0.0)
    manager.resolve(1.0e-4)

    diagnostics = manager.diagnostics()
    assert diagnostics["point_triangle_active"] > 0
    force = state.external_force.to_numpy()
    assert np.isfinite(force).all()
    np.testing.assert_allclose(np.sum(force, axis=0), np.zeros(3), atol=1.0e-10)
    energy = manager.energy_diagnostics()
    assert energy["elastic_energy"] >= 0.0


def test_fem_state_direction_inf_norm_uses_free_physical_dofs():
    positions = np.zeros((2, 3), dtype=np.float64)
    state = FEMState(positions, positions.copy(), np.ones(2))
    state.direction.from_numpy(np.array([[1.0e-4, -3.0e-4, 2.0e-4], [9.0, 4.0e-4, 0.0]]))
    constrained = np.zeros(6, dtype=np.int32)
    constrained[3] = 1
    state.constrained.from_numpy(constrained)

    assert state.direction_inf_norm() == pytest.approx(4.0e-4)


@pytest.mark.parametrize("search", ["LinkedCell", "BVH"])
def test_soft_particle_pt_ee_contact_is_device_resident_and_balanced(search):
    mesh = _two_facing_tetrahedra()
    state = FEMState(
        mesh.points,
        np.zeros_like(mesh.points),
        np.ones(mesh.number_of_nodes),
    )
    model = FEMSoftParticleContactModel(
        "Linear",
        search=search,
        verlet_distance=0.05,
        ContactThickness=0.10,
        NormalStiffness=1.0e4,
        TangentialStiffness=5.0e3,
        Friction=0.3,
    )
    manager = FEMSoftParticleContactManager(mesh, state, model)
    state.external_force.fill(0.0)
    manager.resolve(1.0e-4)

    diagnostics = manager.diagnostics()
    assert diagnostics["point_triangle_candidates"] > 0
    assert diagnostics["edge_edge_candidates"] > 0
    assert diagnostics["point_triangle_active"] > 0
    assert diagnostics["edge_edge_active"] > 0
    np.testing.assert_allclose(
        np.sum(state.external_force.to_numpy(), axis=0),
        np.zeros(3),
        atol=1.0e-10,
    )
    contact_force = state.external_force.to_numpy()
    contact_torque = np.sum(np.cross(state.position.to_numpy(), contact_force), axis=0)
    np.testing.assert_allclose(contact_torque, np.zeros(3), atol=1.0e-10)


@pytest.mark.parametrize("search", ["LinkedCell", "BVH"])
def test_soft_particle_contact_force_matches_stored_energy_gradient(search):
    mesh = _two_facing_tetrahedra()
    state = FEMState(
        mesh.points,
        np.zeros_like(mesh.points),
        np.ones(mesh.number_of_nodes),
    )
    model = FEMSoftParticleContactModel(
        "Linear",
        search=search,
        verlet_distance=0.05,
        ContactThickness=0.10,
        NormalStiffness=1.0e4,
        TangentialStiffness=5.0e3,
        Friction=0.0,
        NormalViscousDamping=0.0,
        TangentialViscousDamping=0.0,
    )
    manager = FEMSoftParticleContactManager(mesh, state, model)
    reference_area = manager.node_area.to_numpy().copy()

    def evaluate(positions):
        state.position.from_numpy(np.ascontiguousarray(positions))
        state.external_force.fill(0.0)
        manager.resolve(
            1.0e-4,
            advance_history=False,
            check_rebuild=False,
        )
        return (
            float(manager.elastic_energy[None]),
            state.external_force.to_numpy().copy(),
        )

    energy, force = evaluate(mesh.points)
    # A relative body translation keeps the deliberately coincident tetra
    # edges in the same feature class, so the finite difference measures the
    # smooth contact potential instead of crossing a nonsmooth PT/EE boundary.
    direction = np.zeros_like(mesh.points)
    direction[mesh.node_body_ids == 0, 2] = 1.0
    direction[mesh.node_body_ids == 1, 2] = -1.0
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-7
    plus = evaluate(mesh.points + epsilon * direction)[0]
    minus = evaluate(mesh.points - epsilon * direction)[0]
    finite_difference = (plus - minus) / (2.0 * epsilon)
    analytic = -float(np.sum(force * direction))

    assert energy > 0.0
    assert finite_difference == pytest.approx(analytic, rel=2.0e-6, abs=2.0e-8)
    np.testing.assert_allclose(manager.node_area.to_numpy(), reference_area)


@pytest.mark.parametrize("search", ["LinkedCell", "BVH"])
def test_offset_closed_surfaces_do_not_contact_before_geometric_touch(search):
    mesh = _two_separated_offset_cubes()
    state = FEMState(
        mesh.points,
        np.zeros_like(mesh.points),
        np.ones(mesh.number_of_nodes),
    )
    model = FEMSoftParticleContactModel(
        "Linear",
        search=search,
        verlet_distance=0.50,
        ContactThickness=0.0,
        NormalStiffness=1.0e4,
        TangentialStiffness=5.0e3,
        Friction=0.0,
    )
    manager = FEMSoftParticleContactManager(mesh, state, model)
    state.external_force.fill(0.0)
    manager.resolve(1.0e-4, check_rebuild=False)

    diagnostics = manager.diagnostics()
    assert diagnostics["point_triangle_candidates"] > 0
    assert diagnostics["point_triangle_active"] == 0
    assert diagnostics["edge_edge_active"] == 0
    np.testing.assert_allclose(state.external_force.to_numpy(), 0.0)

    penetrated = state.position.to_numpy()
    penetrated[mesh.node_body_ids == 1, 0] -= 0.36
    state.position.from_numpy(penetrated)
    manager.rebuild(state.position)
    state.external_force.fill(0.0)
    manager.resolve(1.0e-4, check_rebuild=False)
    diagnostics = manager.diagnostics()
    assert diagnostics["point_triangle_active"] > 0
    np.testing.assert_allclose(
        np.sum(state.external_force.to_numpy(), axis=0),
        np.zeros(3),
        atol=1.0e-10,
    )


def test_bvh_verlet_list_retains_adjacent_point_triangle_features():
    """A node must be able to change its nearest face before the next rebuild."""

    mesh = _two_separated_offset_cubes()
    state = FEMState(
        mesh.points,
        np.zeros_like(mesh.points),
        np.ones(mesh.number_of_nodes),
    )
    model = FEMSoftParticleContactModel(
        "Linear",
        search="BVH",
        verlet_distance=0.50,
        ContactThickness=0.0,
        NormalStiffness=1.0e4,
        TangentialStiffness=5.0e3,
        Friction=0.0,
    )
    manager = FEMSoftParticleContactManager(mesh, state, model)

    stencils = manager.culling.point_triangle.to_numpy()[: manager.pt_count]
    _, candidate_counts = np.unique(stencils[:, 0], return_counts=True)

    # Keeping only the face that is nearest at rebuild time invalidates the
    # Verlet list when the nearest feature changes along an edge or corner.
    assert np.max(candidate_counts) > 1


def test_soft_particle_history_survives_verlet_rebuild():
    mesh = _two_facing_tetrahedra()
    velocity = np.zeros_like(mesh.points)
    velocity[:4, 0] = 0.25
    state = FEMState(mesh.points, velocity, np.ones(mesh.number_of_nodes))
    model = FEMSoftParticleContactModel(
        "Linear",
        search="LinkedCell",
        verlet_distance=0.02,
        ContactThickness=0.10,
        NormalStiffness=1.0e4,
        TangentialStiffness=5.0e3,
        Friction=0.8,
    )
    manager = FEMSoftParticleContactManager(mesh, state, model)
    state.external_force.fill(0.0)
    manager.resolve(1.0e-3)
    first_history = manager.pt_tangential_overlap.to_numpy()[: manager.pt_count]
    assert np.max(np.linalg.norm(first_history, axis=1)) > 0.0

    shifted = state.position.to_numpy()
    shifted[:4, 0] += 0.02
    state.position.from_numpy(shifted)
    assert manager.needs_rebuild(state.position)
    manager.rebuild(state.position)
    rebuilt_history = manager.pt_tangential_overlap.to_numpy()[: manager.pt_count]
    assert np.max(np.linalg.norm(rebuilt_history, axis=1)) > 0.0


def test_soft_particle_rebuild_retains_penetrating_stencil_beyond_one_side_skin():
    mesh = _two_facing_tetrahedra()
    state = FEMState(
        mesh.points,
        np.zeros_like(mesh.points),
        np.ones(mesh.number_of_nodes),
    )
    model = FEMSoftParticleContactModel(
        "Linear",
        search="BVH",
        verlet_distance=0.02,
        ContactThickness=0.10,
        NormalStiffness=1.0e4,
        TangentialStiffness=5.0e3,
        Friction=0.0,
    )
    manager = FEMSoftParticleContactManager(mesh, state, model)
    state.external_force.fill(0.0)
    manager.resolve(1.0e-4, check_rebuild=False)

    penetrated = state.position.to_numpy()
    penetrated[mesh.node_body_ids == 1, 2] -= 0.25
    state.position.from_numpy(penetrated)

    manager.rebuild(state.position)
    state.external_force.fill(0.0)
    manager.resolve(1.0e-4, check_rebuild=False)

    diagnostics = manager.diagnostics()
    assert diagnostics["point_triangle_active"] > 0
    assert diagnostics["maximum_point_triangle_penetration"] > manager.verlet_distance


def test_soft_particle_penetration_quality_limit_is_independent_of_skin():
    mesh = _two_facing_tetrahedra()
    state = FEMState(
        mesh.points,
        np.zeros_like(mesh.points),
        np.ones(mesh.number_of_nodes),
    )
    model = FEMSoftParticleContactModel(
        "Linear",
        search="BVH",
        verlet_distance=0.02,
        maximum_penetration_fraction=0.1,
        ContactThickness=0.10,
        NormalStiffness=1.0e4,
        TangentialStiffness=5.0e3,
        Friction=0.0,
    )
    manager = FEMSoftParticleContactManager(mesh, state, model)
    state.external_force.fill(0.0)
    manager.resolve(1.0e-4, check_rebuild=False)

    penetrated = state.position.to_numpy()
    penetrated[mesh.node_body_ids == 1, 2] -= 0.25
    state.position.from_numpy(penetrated)

    with pytest.raises(RuntimeError, match="mesh-scale quality limit"):
        manager.rebuild(state.position)


def test_soft_particle_verlet_measure_ignores_common_translation():
    mesh = _two_facing_tetrahedra()
    state = FEMState(
        mesh.points,
        np.zeros_like(mesh.points),
        np.ones(mesh.number_of_nodes),
    )
    model = FEMSoftParticleContactModel(
        "Linear",
        search="BVH",
        verlet_distance=0.02,
        ContactThickness=0.10,
        NormalStiffness=1.0e4,
        TangentialStiffness=5.0e3,
        Friction=0.3,
    )
    manager = FEMSoftParticleContactManager(mesh, state, model)

    translated = state.position.to_numpy() + np.array([0.4, -0.2, 0.7])
    state.position.from_numpy(translated)
    assert not manager.needs_rebuild(state.position)

    translated[4:, 0] += 0.021
    state.position.from_numpy(translated)
    assert manager.needs_rebuild(state.position)


def test_soft_particle_default_property_table_is_symmetric():
    mesh = _two_facing_tetrahedra()
    state = FEMState(
        mesh.points,
        np.zeros_like(mesh.points),
        np.ones(mesh.number_of_nodes),
    )
    model = FEMSoftParticleContactModel(
        "Linear",
        search="BVH",
        verlet_distance=0.02,
        ContactThickness=0.10,
        NormalStiffness=1.0e4,
        TangentialStiffness=5.0e3,
        Friction=0.3,
    )
    manager = FEMSoftParticleContactManager(mesh, state, model)
    properties = manager.properties.to_numpy()

    assert properties["active"][0, 0] == 0
    assert properties["active"][1, 1] == 0
    assert properties["active"][0, 1] == 1
    assert properties["active"][1, 0] == 1
    assert properties["kn"][0, 1] == pytest.approx(1.0e4)
    assert properties["friction"][1, 0] == pytest.approx(0.3)


def test_fem_facade_runs_one_soft_particle_step():
    merged = _two_facing_tetrahedra()
    lower = FEMMesh(merged.points[:4], merged.cells[:1], "TET4")
    upper = FEMMesh(merged.points[4:], merged.cells[1:] - 4, "TET4")
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit")
    fem.add_soft_particle(lower)
    fem.add_soft_particle(upper)
    fem.add_material(
        "StVK",
        density=1000.0,
        young_modulus=1.0e4,
        poisson_ratio=0.3,
    )
    fem.add_soft_particle_contact(
        "Linear",
        search="BVH",
        verlet_distance=0.05,
        ContactThickness=0.10,
        NormalStiffness=1.0e3,
        TangentialStiffness=5.0e2,
        Friction=0.0,
    )
    fem.set_solver(dt=1.0e-6, steps=1)

    result = fem.run(verbose=False)

    assert result.time == pytest.approx(1.0e-6)
    assert fem.engine.soft_particle_contact.pt_count > 0
    assert fem.engine.soft_particle_contact.ee_count > 0
    assert not fem.engine.soft_particle_contact.culling.enable_ccd
    assert fem.engine.soft_particle_contact.culling.ccd_pt_capacity == 1
    assert fem.engine.soft_particle_contact.culling.ccd_ee_capacity == 1
    assert not fem.engine.classical_assembler.allocate_hessian
    assert not hasattr(fem.engine.classical_assembler, "cell_hessian")
    with pytest.raises(RuntimeError, match="Hessian storage is disabled"):
        fem.engine._assemble_internal_device(need_stiffness=True)
