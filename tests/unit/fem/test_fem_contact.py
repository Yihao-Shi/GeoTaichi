import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem import FEM
from src.fem.contact import (
    FEMCollisionCulling,
    FEMContact,
    broad_phase_candidates,
    build_contact_surface,
)
from src.fem.contact.ContactAssembler import FEMContactAssembler
from src.fem.contact.ContactAssembler import FEMMultiContactAssembler
from src.fem.contact.LinkedCellBroadPhase import DynamicLinkedCellBroadPhase
from src.fem.contact.BVHBroadPhase import DynamicBVHBroadPhase
from src.fem.contact.ContactTopology import _reference_broad_phase_candidates
from src.fem.engines.SparseMatrix import FEMSparseMatrix
from src.fem.generator import FEMMesh
from src.physics_model.consititutive_model.finite_strain import ClothARAP

pytestmark = [pytest.mark.unit, pytest.mark.fem, pytest.mark.contact, pytest.mark.cpu, pytest.mark.serial]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
    yield
    ti.reset()


def two_triangle_mesh(separation=0.05):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.1, separation],
            [1.1, 0.1, separation],
            [0.1, 1.1, separation],
        ]
    )
    return FEMMesh(points, np.array([[0, 1, 2], [3, 4, 5]]), "TRI3")


def test_ipc_body_pairs_have_independent_parameters_and_candidates():
    base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    points = np.vstack((base, base + (0.0, 0.0, 0.05), base + (0.0, 0.0, 0.10)))
    cells = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
    mesh = FEMMesh(points, cells, "TRI3")
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    fem.add_mesh(mesh)
    fem.add_material(
        "StVK",
        density=1000.0,
        young_modulus=1.0e4,
        poisson_ratio=0.3,
    )
    fem.add_contact("IPC", dhat=0.1, self_contact=False)
    fem.add_contact_property(0, 1, dhat=0.08, dmin=0.0, kappa=2.0e3)
    fem.add_contact_property(1, 2, dhat=0.06, dmin=0.0, kappa=4.0e3)
    fem.set_solver(dt=1.0e-3, steps=1, linear_solver="BiCGSTAB")

    engine = fem.build()
    assembler = engine.contact_assembler
    assert isinstance(assembler, FEMMultiContactAssembler)
    assert [entry.contact.body_pair for entry in assembler.assemblers] == [
        (0, 1),
        (1, 2),
    ]
    assert [entry.contact.kappa for entry in assembler.assemblers] == [
        2.0e3,
        4.0e3,
    ]

    assembler.prepare_iteration_device(engine.state.position)
    aggregate = assembler.device_diagnostics()
    diagnostics = aggregate["body_pairs"]
    assert all(entry["pt_candidates"] == 6 for entry in diagnostics)
    assert all(entry["ee_candidates"] == 9 for entry in diagnostics)
    assert aggregate["contact_penalty"] == max(entry["contact_penalty"] for entry in diagnostics)

    engine._assemble_internal_device(need_stiffness=False)
    pair_energy = [float(entry.total_energy[None]) for entry in assembler.assemblers]
    assert all(value > 0.0 for value in pair_energy)
    assert pair_energy[0] != pytest.approx(pair_energy[1])


def test_surface_measures_and_nonincident_broad_phase():
    mesh = two_triangle_mesh()
    surface = build_contact_surface(mesh)
    pt, ee = broad_phase_candidates(surface, mesh.points, radius=0.1)

    assert np.sum(surface.node_area) == pytest.approx(1.0)
    assert np.sum(surface.edge_area) == pytest.approx(1.0)
    assert pt.shape[1] == ee.shape[1] == 4
    assert all(vertex not in triangle for vertex, *triangle in pt)
    assert all(len(set(stencil)) == 4 for stencil in ee)


def test_dynamic_linked_cell_matches_reference_and_rebuilds_swept_aabbs():
    mesh = two_triangle_mesh()
    surface = build_contact_surface(mesh)
    positions = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    end_positions = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    positions.from_numpy(mesh.points)
    end = mesh.points.copy()
    end[3:, 2] -= 0.04
    end_positions.from_numpy(end)
    broad_phase = DynamicLinkedCellBroadPhase(
        surface.faces,
        surface.edges,
        surface.vertices,
        surface.node_area,
        surface.edge_area,
        mesh.points,
    )

    pt_count, ee_count = broad_phase.rebuild(positions, radius=0.1, end_positions=end_positions)
    reference_pt, reference_ee = _reference_broad_phase_candidates(surface, mesh.points, radius=0.1, end_positions=end)
    device_pt = broad_phase.point_triangle.to_numpy()[:pt_count]
    device_ee = broad_phase.edge_edge.to_numpy()[:ee_count]
    primitives = broad_phase.point_triangle_primitive.to_numpy()[:pt_count]

    assert {tuple(value) for value in device_pt} == {tuple(value) for value in reference_pt}
    assert {tuple(value) for value in device_ee} == {tuple(value) for value in reference_ee}
    assert broad_phase.face_member_capacity >= int(broad_phase.face_prefix[broad_phase.cell_count])
    for stencil, (vertex_id, face_id) in zip(device_pt, primitives):
        assert stencil[0] == surface.vertices[vertex_id]
        np.testing.assert_array_equal(stencil[1:], surface.faces[face_id])


def test_dynamic_bvh_matches_reference_and_rebuilds_swept_aabbs():
    mesh = two_triangle_mesh()
    surface = build_contact_surface(mesh)
    positions = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    end_positions = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    positions.from_numpy(mesh.points)
    end = mesh.points.copy()
    end[3:, 2] -= 0.04
    end_positions.from_numpy(end)
    broad_phase = DynamicBVHBroadPhase(
        surface.faces,
        surface.edges,
        surface.vertices,
        surface.node_area,
        surface.edge_area,
        mesh.points,
    )

    pt_count, ee_count = broad_phase.rebuild(positions, radius=0.1, end_positions=end_positions)
    reference_pt, reference_ee = _reference_broad_phase_candidates(surface, mesh.points, radius=0.1, end_positions=end)
    device_pt = broad_phase.point_triangle.to_numpy()[:pt_count]
    primitives = broad_phase.point_triangle_primitive.to_numpy()[:pt_count]

    assert {tuple(value) for value in device_pt} == {tuple(value) for value in reference_pt}
    assert {tuple(value) for value in broad_phase.edge_edge.to_numpy()[:ee_count]} == {
        tuple(value) for value in reference_ee
    }
    for stencil, (vertex_id, face_id) in zip(device_pt, primitives):
        assert stencil[0] == surface.vertices[vertex_id]
        np.testing.assert_array_equal(stencil[1:], surface.faces[face_id])


@pytest.mark.parametrize(
    "broad_phase_type",
    [DynamicLinkedCellBroadPhase, DynamicBVHBroadPhase],
)
def test_collision_culling_can_retain_only_cross_system_stencils(
    broad_phase_type,
):
    mesh = two_triangle_mesh()
    surface = build_contact_surface(mesh)
    positions = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    positions.from_numpy(mesh.points)

    same_system = broad_phase_type(
        surface.faces,
        surface.edges,
        surface.vertices,
        surface.node_area,
        surface.edge_area,
        mesh.points,
    )
    culled = FEMCollisionCulling(
        same_system,
        mesh.number_of_nodes,
        node_system_ids=np.zeros(mesh.number_of_nodes, dtype=np.int32),
        cross_system_only=True,
    )
    assert culled.rebuild_proximity(positions, 0.1) == (0, 0)

    mixed_system = broad_phase_type(
        surface.faces,
        surface.edges,
        surface.vertices,
        surface.node_area,
        surface.edge_area,
        mesh.points,
    )
    culled = FEMCollisionCulling(
        mixed_system,
        mesh.number_of_nodes,
        node_system_ids=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        cross_system_only=True,
    )
    pt_count, ee_count = culled.rebuild_proximity(positions, 0.1)
    assert pt_count + ee_count > 0


@pytest.mark.parametrize("broad_phase_type", [DynamicLinkedCellBroadPhase, DynamicBVHBroadPhase])
def test_edge_edge_broad_phase_uses_pairwise_search_radius(
    broad_phase_type,
):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.15, 0.0],
            [1.0, 0.15, 0.0],
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 10.0],
        ]
    )
    positions = ti.Vector.field(3, dtype=ti.f64, shape=points.shape[0])
    positions.from_numpy(points)
    broad_phase = broad_phase_type(
        np.array([[4, 5, 6]], dtype=np.int32),
        np.array([[0, 1], [2, 3]], dtype=np.int32),
        np.arange(points.shape[0], dtype=np.int32),
        np.ones(points.shape[0]),
        np.ones(2),
        points,
    )

    _, separated_count = broad_phase.rebuild(positions, radius=0.1)
    _, nearby_count = broad_phase.rebuild(positions, radius=0.16)

    assert separated_count == 0
    assert nearby_count == 1


@pytest.mark.parametrize("broad_phase_type", [DynamicLinkedCellBroadPhase, DynamicBVHBroadPhase])
def test_collision_culling_rejects_aabb_false_positives(broad_phase_type):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.8, 0.0],
            [1.8, 0.8, 0.0],
            [0.8, 1.8, 0.0],
        ]
    )
    mesh = FEMMesh(
        points,
        np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        "TRI3",
    )
    surface = build_contact_surface(mesh)
    positions = ti.Vector.field(3, dtype=ti.f64, shape=points.shape[0])
    positions.from_numpy(points)
    broad_phase = broad_phase_type(
        surface.faces,
        surface.edges,
        surface.vertices,
        surface.node_area,
        surface.edge_area,
        points,
    )
    culling = FEMCollisionCulling(broad_phase, points.shape[0])

    pt_count, ee_count = culling.rebuild_proximity(positions, 0.1)
    diagnostics = culling.diagnostics()

    assert diagnostics["raw_pt_candidates"] > 0
    assert diagnostics["raw_ee_candidates"] > 0
    assert pt_count == 0
    assert ee_count == 0


@pytest.mark.parametrize("broad_phase_type", [DynamicLinkedCellBroadPhase, DynamicBVHBroadPhase])
def test_collision_culling_compacts_ccd_limited_stencils(broad_phase_type):
    mesh = two_triangle_mesh(separation=0.2)
    surface = build_contact_surface(mesh)
    positions = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    end_positions = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    positions.from_numpy(mesh.points)
    broad_phase = broad_phase_type(
        surface.faces,
        surface.edges,
        surface.vertices,
        surface.node_area,
        surface.edge_area,
        mesh.points,
    )
    culling = FEMCollisionCulling(broad_phase, mesh.number_of_nodes)

    no_impact = mesh.points.copy()
    no_impact[3:, 2] -= 0.15
    end_positions.from_numpy(no_impact)
    pt_count, ee_count, alpha = culling.compute_ccd(
        positions,
        end_positions,
        0.11,
        eta=0.1,
        thickness=0.01,
        max_iterations=100,
    )
    no_impact_diagnostics = culling.diagnostics()

    assert no_impact_diagnostics["raw_pt_candidates"] + no_impact_diagnostics["raw_ee_candidates"] > 0
    assert pt_count + ee_count == 0
    assert alpha == pytest.approx(1.0)

    impact = mesh.points.copy()
    impact[3:, 2] -= 0.25
    end_positions.from_numpy(impact)
    pt_count, ee_count, alpha = culling.compute_ccd(
        positions,
        end_positions,
        0.11,
        eta=0.1,
        thickness=0.01,
        max_iterations=100,
    )

    assert pt_count + ee_count > 0
    assert 0.0 < alpha < 1.0


def test_collision_culling_excludes_garment_stitch_neighborhoods():
    mesh = two_triangle_mesh()
    surface = build_contact_surface(mesh)
    positions = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    positions.from_numpy(mesh.points)
    broad_phase = DynamicLinkedCellBroadPhase(
        surface.faces,
        surface.edges,
        surface.vertices,
        surface.node_area,
        surface.edge_area,
        mesh.points,
    )
    culling = FEMCollisionCulling(broad_phase, mesh.number_of_nodes)
    original_pt_count, original_ee_count = culling.rebuild_proximity(positions, 0.1)
    culling.set_stitch_exclusions([[3, 0, 1]])
    pt_count, ee_count = culling.rebuild_proximity(positions, 0.1)
    point_triangle = culling.point_triangle.to_numpy()[:pt_count]
    edge_edge = culling.edge_edge.to_numpy()[:ee_count]

    assert pt_count + ee_count < original_pt_count + original_ee_count
    assert all(
        not ((stencil[0] == 3 and ({0, 1} & set(stencil[1:]))) or (stencil[0] in (0, 1) and 3 in stencil[1:]))
        for stencil in point_triangle
    )
    assert all(
        not any(
            first in (3,) and second in (0, 1) or second in (3,) and first in (0, 1)
            for first in stencil[:2]
            for second in stencil[2:]
        )
        for stencil in edge_edge
    )


def test_contact_configuration_selects_bvh():
    mesh = two_triangle_mesh()
    assembler = FEMContactAssembler(
        mesh,
        ClothARAP(1.0e3, thickness=0.01),
        FEMContact("IPC", broad_phase="BVH", dhat=0.1, dmin=0.01),
    )

    assert isinstance(assembler.broad_phase, DynamicBVHBroadPhase)
    assert assembler.collision_culling.broad_phase is assembler.broad_phase


def test_contact_coordination_preallocates_fixed_device_buffers():
    mesh = two_triangle_mesh()
    assembler = FEMContactAssembler(
        mesh,
        ClothARAP(1.0e3, thickness=0.01),
        FEMContact(
            "IPC",
            dhat=0.1,
            dmin=0.01,
            point_triangle_coordination_number=4.0,
            edge_edge_coordination_number=5.0,
        ),
    )

    assert assembler.max_point_triangle_pairs == 8
    assert assembler.max_edge_edge_pairs == 10
    assert assembler.device_pt_capacity == 8
    assert assembler.device_ee_capacity == 10
    assert not hasattr(assembler, "device_pt_hessian")
    assert not hasattr(assembler, "device_ee_hessian")
    assert not hasattr(assembler, "friction_pt_hessian")
    assert not hasattr(assembler, "friction_ee_hessian")
    assert not hasattr(assembler, "plane_hessian")

    assembler.prepare_iteration_device(assembler.device_positions)

    with pytest.raises(RuntimeError, match="point_triangle_coordination_number"):
        assembler._require_device_contact_capacity(9, 0)
    with pytest.raises(RuntimeError, match="edge_edge_coordination_number"):
        assembler._require_device_friction_capacity(0, 11)


def test_contact_explicit_pair_capacities_override_coordination_estimates():
    assembler = FEMContactAssembler(
        two_triangle_mesh(),
        ClothARAP(1.0e3, thickness=0.01),
        FEMContact(
            "IPC",
            dhat=0.1,
            dmin=0.01,
            point_triangle_coordination_number=1.0,
            edge_edge_coordination_number=1.0,
            max_point_triangle_pairs=11,
            max_edge_edge_pairs=13,
        ),
    )

    assert assembler.max_point_triangle_pairs == 11
    assert assembler.max_edge_edge_pairs == 13


def test_cloth_engine_registers_stitches_with_contact_culling():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    fem.add_mesh(two_triangle_mesh())
    fem.add_material("ClothARAP", stretch_stiffness=1.0e3, thickness=0.01)
    fem.add_stitch([[3, 0, 1]], stiffness=1.0e3, ratios=[0.1])
    fem.add_contact("IPC", broad_phase="LinkedCell", dhat=0.1, dmin=0.01)
    fem.set_solver(linear_solver="BiCGSTAB", project_pd=False, step=0)

    engine = fem.build()
    culling = engine.contact_assembler.collision_culling
    prefix = culling.stitch_prefix.to_numpy()
    neighbors = culling.stitch_neighbors.to_numpy()

    assert set(neighbors[prefix[3] : prefix[4]]) == {0, 1}


def test_ipc_energy_gradient_tangent_and_ccd():
    mesh = two_triangle_mesh()
    material = ClothARAP(1.0e3, thickness=0.01)
    contact = FEMContact("IPC", dhat=0.1, dmin=0.01, kappa=2.0e3, project_pd=False)
    assembler = FEMContactAssembler(mesh, material, contact)
    positions = mesh.points.copy()
    assembler.prepare_iteration(positions)
    energy, force, stiffness = assembler.assemble(positions, need_stiffness=True)
    direction = np.random.default_rng(18).normal(size=positions.shape)
    direction -= np.mean(direction, axis=0)
    epsilon = 1.0e-7
    plus, plus_force, _ = assembler.assemble(positions + epsilon * direction)
    minus, minus_force, _ = assembler.assemble(positions - epsilon * direction)
    finite_gradient = (plus - minus) / (2.0 * epsilon)
    finite_tangent_product = (plus_force - minus_force) / (2.0 * epsilon)

    assert np.isfinite(energy) and energy > 0.0
    assert np.sum(force * direction) == pytest.approx(finite_gradient, rel=2.0e-5, abs=2.0e-5)
    np.testing.assert_allclose(stiffness.toarray(), stiffness.toarray().T, atol=1.0e-7)
    np.testing.assert_allclose(
        (stiffness @ direction.reshape(-1)).reshape(positions.shape),
        finite_tangent_product,
        rtol=2.0e-4,
        atol=2.0e-4,
    )

    closing = np.zeros_like(positions)
    closing[3:, 2] = -0.08
    alpha = assembler.maximum_admissible_step(positions, closing)
    assert 0.0 < alpha < 1.0


def test_ipc_self_contact_device_force_matches_output_adapter():
    mesh = two_triangle_mesh()
    material = ClothARAP(1.0e3, thickness=0.01)
    contact = FEMContact("IPC", dhat=0.1, dmin=0.01, kappa=2.0e3, project_pd=False)
    assembler = FEMContactAssembler(mesh, material, contact)
    positions = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    force = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    positions.from_numpy(mesh.points)
    force.fill(0.0)
    assembler.prepare_iteration_device(positions)
    assembler.assemble_device(positions, force, None, need_stiffness=False)
    device_energy = float(assembler.total_energy[None])
    device_force = force.to_numpy()

    assembler.prepare_iteration(mesh.points)
    output_energy, output_force, _ = assembler.assemble(mesh.points)
    assert device_energy == pytest.approx(output_energy, rel=2.0e-12)
    np.testing.assert_allclose(device_force, output_force, rtol=2.0e-11, atol=2.0e-11)


def test_nonbarrier_al_projects_multipliers_and_preserves_oriented_gap():
    mesh = two_triangle_mesh(separation=0.005)
    material = ClothARAP(1.0e3, thickness=0.01)
    contact = FEMContact(
        "AugmentedLagrangian",
        dhat=0.05,
        dmin=0.01,
        penalty=1.0e4,
        penalty_growth=2.0,
        penalty_update_interval=1,
        sufficient_reduction=0.5,
        project_pd=True,
    )
    assembler = FEMContactAssembler(mesh, material, contact)
    positions = mesh.points.copy()
    assembler.prepare_iteration(positions)
    energy, force, stiffness = assembler.assemble(positions, need_stiffness=True)

    assert np.isfinite(energy)
    assert assembler.constraint_violation(positions) > 0.0
    assert stiffness.shape == (18, 18)
    assert np.linalg.eigvalsh(stiffness.toarray()).min() >= -1.0e-8

    assembler.accept_update(positions, step=0.0)
    assert assembler._al_state
    old_penalty = assembler.penalty
    crossed = positions.copy()
    crossed[3:, 2] = -0.005
    assembler.prepare_iteration(crossed)
    assert assembler.constraint_violation(crossed) >= 0.014
    assembler.prepare_iteration(positions)
    assembler.accept_update(positions, step=0.0)
    assert assembler.penalty > old_penalty
    assert float(assembler.penalty_field[None]) == pytest.approx(assembler.penalty)
    assert all(value >= 0.0 for value in assembler._al_state.values())


def test_nonbarrier_al_self_contact_runs_in_device_fields():
    mesh = two_triangle_mesh(separation=0.005)
    material = ClothARAP(1.0e3, thickness=0.01)
    contact = FEMContact(
        "AugmentedLagrangian",
        dhat=0.05,
        dmin=0.01,
        penalty=1.0e4,
        project_pd=True,
    )
    assembler = FEMContactAssembler(mesh, material, contact)
    position = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    force = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    position.from_numpy(mesh.points)
    assembler.prepare_iteration_device(position)
    force.fill(0.0)
    matrix = FEMSparseMatrix(
        3 * mesh.number_of_nodes,
        assemble_type="HashTriplet",
        linear_solver="PCG",
    )
    assembler.assemble_device(position, force, matrix, need_stiffness=True)

    assert float(assembler.total_energy[None]) > 0.0
    assert np.isfinite(force.to_numpy()).all()
    assert int(assembler.al_pt_hash_count[None]) > 0
    old_penalty = assembler.penalty
    assembler.accept_update_device(position)
    assert np.max(assembler.al_pt_hash_multiplier.to_numpy()) > 0.0
    assert float(assembler.constraint_violation_field[None]) > 0.0
    assert assembler.penalty == old_penalty


def test_contact_configuration_accepts_al_friction():
    barrier = FEMContact("BarrierIPC")
    assert barrier.model == "IPC"
    assert barrier.point_triangle_coordination_number == pytest.approx(32.0)
    assert barrier.edge_edge_coordination_number == pytest.approx(128.0)
    semi = FEMContact("SemiIPC")
    assert semi.model == "AugmentedLagrangian"
    assert semi.penalty_growth == pytest.approx(2.0)
    assert semi.penalty_update_interval == 50
    assert FEMContact("non barrier augment lagrangian").model == "AugmentedLagrangian"
    aliased = FEMContact.create({"Model": "AL", "InitialPenalty": 12.0, "ContactDistance": 0.02})
    assert aliased.penalty == pytest.approx(12.0)
    assert aliased.dmin == pytest.approx(0.02)
    contact = FEMContact("AugmentedLagrangian", friction_coefficient=0.2)
    assert contact.friction_coefficient == pytest.approx(0.2)


def test_augmented_lagrangian_reuses_lagged_coulomb_friction():
    mesh = two_triangle_mesh(separation=0.005)
    assembler = FEMContactAssembler(
        mesh,
        ClothARAP(1.0e3, thickness=0.01),
        FEMContact(
            "AugmentedLagrangian",
            dhat=0.05,
            dmin=0.01,
            penalty=1.0e4,
            friction_coefficient=0.3,
            epsv=1.0e-3,
            project_pd=True,
        ),
    )
    position = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    force = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    position.from_numpy(mesh.points)
    assembler.begin_step_device(position, 0.1)
    moved = mesh.points.copy()
    moved[3:, 0] += 0.01
    position.from_numpy(moved)
    assembler.prepare_iteration_device(position)
    force.fill(0.0)
    assembler.assemble_device(position, force, None, need_stiffness=False)

    device_force = force.to_numpy()
    assert np.max(assembler.friction_pt_normal_force.to_numpy()) > 0.0
    assert np.sum(device_force[3:, 0]) > 0.0
    np.testing.assert_allclose(np.sum(device_force, axis=0), 0.0, atol=1.0e-8)


def test_lagged_friction_opposes_tangential_motion():
    mesh = two_triangle_mesh()
    material = ClothARAP(1.0e3, thickness=0.01)
    contact = FEMContact(
        "IPC",
        dhat=0.1,
        dmin=0.01,
        kappa=2.0e3,
        friction_coefficient=0.3,
        epsv=1.0e-3,
    )
    assembler = FEMContactAssembler(mesh, material, contact)
    assembler.begin_step(mesh.points, dt=0.1)
    assert float(assembler.friction_dt_field[None]) == pytest.approx(0.1)
    positions = mesh.points.copy()
    positions[3:, 0] += 0.01
    assembler.prepare_iteration(positions)
    energy, force, stiffness = assembler.assemble(positions, need_stiffness=True)
    direction = np.zeros_like(positions)
    direction[3:, 0] = 1.0
    epsilon = 1.0e-7
    plus = assembler.assemble(positions + epsilon * direction)[0]
    minus = assembler.assemble(positions - epsilon * direction)[0]

    assert energy > 0.0
    assert np.sum(force[3:, 0]) > 0.0
    assert np.sum(force * direction) == pytest.approx((plus - minus) / (2.0 * epsilon), rel=2.0e-5)
    np.testing.assert_allclose(np.sum(force, axis=0), 0.0, atol=1.0e-7)
    assert np.linalg.eigvalsh(stiffness.toarray()).min() >= -1.0e-6


def test_lagged_friction_device_path_matches_output_adapter():
    mesh = two_triangle_mesh()
    material = ClothARAP(1.0e3, thickness=0.01)
    contact = FEMContact(
        "IPC",
        dhat=0.1,
        dmin=0.01,
        kappa=2.0e3,
        friction_coefficient=0.3,
        epsv=1.0e-3,
        project_pd=True,
    )
    assembler = FEMContactAssembler(mesh, material, contact)
    position = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    force = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    position.from_numpy(mesh.points)
    assembler.begin_step_device(position, 0.1)
    moved = mesh.points.copy()
    moved[3:, 0] += 0.01
    position.from_numpy(moved)
    assembler.prepare_iteration_device(position)
    force.fill(0.0)
    matrix = FEMSparseMatrix(
        3 * mesh.number_of_nodes,
        assemble_type="COO",
        linear_solver="Scipy",
    )
    assembler.assemble_device(position, force, matrix, need_stiffness=True)
    device_energy = float(assembler.total_energy[None])
    device_force = force.to_numpy()
    device_stiffness = matrix.toarray()

    reference = FEMContactAssembler(mesh, material, contact)
    reference.begin_step(mesh.points, 0.1)
    reference.prepare_iteration(moved)
    energy, output_force, output_stiffness = reference.assemble(moved, need_stiffness=True)
    assert device_energy == pytest.approx(energy, rel=2.0e-10)
    np.testing.assert_allclose(device_force, output_force, rtol=2.0e-9, atol=2.0e-9)
    np.testing.assert_allclose(
        device_stiffness,
        output_stiffness.toarray(),
        rtol=2.0e-8,
        atol=2.0e-8,
    )


def test_lagged_friction_device_refresh_preserves_beginning_of_step_hat():
    mesh = two_triangle_mesh()
    assembler = FEMContactAssembler(
        mesh,
        ClothARAP(1.0e3, thickness=0.01),
        FEMContact(
            "IPC",
            dhat=0.1,
            dmin=0.01,
            kappa=2.0e3,
            friction_coefficient=0.3,
        ),
    )
    position = ti.Vector.field(3, dtype=ti.f64, shape=mesh.number_of_nodes)
    position.from_numpy(mesh.points)
    assembler.begin_step_device(position, 0.1)
    step_hat = assembler.friction_hat_position.to_numpy().copy()

    moved = mesh.points.copy()
    moved[3:, 0] += 0.01
    position.from_numpy(moved)
    assembler.refresh_friction_device(position)

    np.testing.assert_array_equal(assembler.friction_hat_position.to_numpy(), step_hat)


def test_ipc_plane_energy_and_ccd_limit_are_one_sided():
    mesh = two_triangle_mesh()
    mesh.points[:, 2] += 0.04
    material = ClothARAP(1.0e3, thickness=0.01)
    contact = FEMContact(
        "IPC",
        self_contact=False,
        planes=[{"point": (0.0, 0.0, 0.0), "normal": (0.0, 0.0, 1.0)}],
        dhat=0.05,
        dmin=0.01,
        kappa=10.0,
    )
    assembler = FEMContactAssembler(mesh, material, contact)
    assembler.prepare_iteration(mesh.points)
    energy, force, stiffness = assembler.assemble(mesh.points, need_stiffness=True)
    direction = np.zeros_like(mesh.points)
    direction[:, 2] = -0.1
    alpha = assembler.maximum_admissible_step(mesh.points, direction)

    assert energy > 0.0
    assert np.all(force[:, 2] <= 0.0)
    assert alpha == pytest.approx(0.27)
    assert np.linalg.eigvalsh(stiffness.toarray()).min() >= -1.0e-8


def test_fem_facade_dispatches_contact_into_implicit_cloth_engine():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    fem.add_mesh(two_triangle_mesh())
    fem.add_material("ClothARAP", stretch_stiffness=1.0e3, thickness=0.01)
    fem.add_contact("IPC", dhat=0.1, dmin=0.01, kappa=2.0e3)
    fem.set_solver(quasi_static=False, dt=0.01, step=0)

    engine = fem.build()
    engine._prepare_contact_iteration(engine.positions)
    energy, force, stiffness, _ = engine.assemble_internal(need_stiffness=True)

    assert engine.contact_assembler is not None
    assert engine.contact_assembler.is_ipc
    assert energy > 0.0
    assert np.linalg.norm(force) > 0.0
    assert stiffness.shape == (engine.degree_of_freedom, engine.degree_of_freedom)


def test_same_contact_backend_assembles_on_volume_boundary_surface():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    fem.add_mesh(
        geometry="box",
        size=(1.0, 1.0, 0.1),
        divisions=(1, 1, 1),
        element_type="TET4",
        origin=(0.0, 0.0, 0.03),
    )
    fem.add_material("StVK", young_modulus=1.0e4, poisson_ratio=0.3, density=1.0)
    fem.add_contact(
        "IPC",
        self_contact=False,
        planes=[((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))],
        dhat=0.05,
        dmin=0.01,
        kappa=10.0,
    )
    fem.set_solver(quasi_static=False, dt=0.01, step=0)

    engine = fem.build()
    engine._prepare_contact_iteration(engine.positions)
    energy, force, stiffness, _ = engine.assemble_internal(need_stiffness=True)

    assert type(engine).__name__ == "ClassicalImplicitFEM"
    assert engine.backend == "taichi"
    assert engine.contact_assembler.surface.faces.shape[1] == 3
    assert energy > 0.0
    assert np.min(force[:, 2]) < 0.0
    assert stiffness.shape == (engine.degree_of_freedom, engine.degree_of_freedom)
