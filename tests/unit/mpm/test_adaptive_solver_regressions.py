"""Adaptive transfers and assembly preserve volume, pressure and constrained connectivity."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.mpm.elements.AdaptiveNodeMap import AdaptiveNodeMap, build_hanging_constraint_table
from src.mpm.elements.AdaptiveQuadrilateralKernel import (
    split_refined_particles_2d,
    split_refined_particles_with_traction_2d,
)
from src.mpm.elements.AdaptiveHexahedronKernel import (
    cache_refined_particle_parents,
    copy_refined_particle_children,
    split_refined_particles,
)
from src.mpm.engines.EngineKernel import kernel_update_displacement_gradient_2D, kernel_update_displacement_gradient
from src.mpm.engines.NewtonIteration import MomentumConservation
from src.mpm.engines.TLExplicitEngine import TLExplicitEngine
from src.mpm.engines.ULExplicitEngine import ULExplicitEngine
from src.mpm.structs.Particle import ImplicitParticle2D, ImplicitParticle, ParticleCloud2D

DIMENSIONS = [
    pytest.param(2, marks=pytest.mark.isolated_dimension(2)),
    pytest.param(3, marks=pytest.mark.isolated_dimension(3)),
]


def test_native_coo_uses_configured_relative_residual(taichi_runtime):
    from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix

    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, -2.0])
    iterator = MomentumConservation()
    iterator.operator = SimpleNamespace(active_dofs=2)
    iterator.sparse_matrix = CoordinateSparseMatrix(4, 2)
    rows, cols = np.indices(matrix.shape)
    iterator.sparse_matrix.rows.from_numpy(rows.ravel().astype(np.int32))
    iterator.sparse_matrix.cols.from_numpy(cols.ravel().astype(np.int32))
    iterator.sparse_matrix.data.from_numpy(matrix.ravel())
    iterator.right_hand_vector = ti.field(ti.f64, shape=2)
    iterator.unknow_vector = ti.field(ti.f64, shape=2)
    iterator.diag_A = ti.field(ti.f64, shape=2)
    iterator.right_hand_vector.from_numpy(rhs)
    iterator.diag_A.from_numpy(np.diag(matrix))
    sims = SimpleNamespace(
        residual_tolerance=1e-12,
        linear_solver_relative_tolerance=0.0,
        timer=SimpleNamespace(begin=lambda *_: None, end=lambda *_: None),
    )
    for relative, iterations in [(0.0, 2), (0.3, 1)]:
        sims.linear_solver_relative_tolerance = relative
        iterator.unknow_vector.fill(0)
        iterator.solve_coo(sims, None)
        assert iterator.sparse_matrix.linear_solver.last_iterations == iterations
        residual = np.linalg.norm(rhs - matrix @ iterator.unknow_vector.to_numpy())
        assert residual <= max(sims.residual_tolerance, relative * np.linalg.norm(rhs))


def _split_scene(dimension, implicit=True, traction=False, contact=False):
    count = 2**dimension
    particle_type = (ImplicitParticle2D if dimension == 2 else ImplicitParticle) if implicit else ParticleCloud2D
    particle = (
        ti.Struct.field(dict(particle_type.members, contact_traction=ti.types.vector(2, float)), shape=count)
        if contact
        else particle_type.field(shape=count)
    )
    if contact:
        particle[0].contact_traction = [12.0, -4.0]
    particle[0].active = 1
    particle[0].materialID = 1
    particle[0].m = 2.0
    particle[0].vol = 1.0
    if implicit:
        particle[0].vol0 = 1.0
    particle[0].x = [0.5] * dimension
    particle[0].stress = [12.0, 12.0, 12.0, 0.0, 0.0, 0.0]
    state = ti.Struct.field(dict(history=ti.f64), shape=count)
    state[0].history = 7.0
    size = ti.Vector.field(dimension, float, shape=count)
    length = ti.Vector.field(dimension, float, shape=count)
    size[0] = [0.1] * dimension
    level = ti.field(ti.u8, shape=count)
    refined = ti.field(ti.u8, shape=count)
    cells = ti.field(ti.u8, shape=1)
    cells[0] = 1
    new_count = ti.field(int, shape=())
    new_count[None] = 1
    particle_num = np.array([1], dtype=np.int32)
    ones = ti.Vector([1.0] * dimension)
    cnum = ti.Vector([1] * dimension)
    if dimension == 2:
        if traction:
            tr = ti.Struct.field(dict(pid=ti.i32, psize=ti.types.vector(2, float)), shape=count)
            head = ti.field(int, shape=count)
            head.fill(-1)
            head[0] = 0
            next_id = ti.field(int, shape=count)
            next_id.fill(-1)
            tr_count = ti.field(int, shape=())
            tr_count[None] = 1

        def split(*args):
            if traction:
                split_refined_particles_with_traction_2d(
                    ones,
                    cnum,
                    1,
                    1,
                    False,
                    cells,
                    particle,
                    state,
                    size,
                    length,
                    level,
                    refined,
                    tr,
                    head,
                    next_id,
                    new_count,
                    tr_count,
                )
                assert tr_count[None] == count
            else:
                split_refined_particles_2d(
                    ones,
                    cnum,
                    1,
                    False,
                    cells,
                    particle,
                    state,
                    size,
                    length,
                    level,
                    refined,
                    new_count,
                )
            particle_num[0] = int(new_count[None])

    else:
        parents = ti.field(int, shape=1)
        vectors = [ti.Vector.field(3, float, shape=1) for _ in range(4)]
        gradient = ti.Matrix.field(3, 3, float, shape=1)
        mass = ti.field(float, shape=1)
        volume = ti.field(float, shape=1)
        child_level = ti.field(ti.u8, shape=1)
        overflow = ti.field(int, shape=())

        def split(*args):
            cache_refined_particle_parents(
                1,
                parents,
                particle,
                size,
                length,
                vectors[0],
                vectors[1],
                gradient,
                vectors[2],
                vectors[3],
                mass,
                volume,
                child_level,
                refined,
            )
            copy_refined_particle_children(1, 1, parents, particle, state, count, overflow)
            assert overflow[None] == 0
            split_refined_particles(
                1,
                1,
                parents,
                particle,
                vectors[0],
                vectors[1],
                gradient,
                vectors[2],
                vectors[3],
                mass,
                volume,
                child_level,
                size,
                length,
                level,
                refined,
            )
            particle_num[0] = count

    node = ti.Struct.field(dict(m=float, pressure=float, displacement=ti.types.vector(dimension, float)), shape=(1, 1))
    node[0, 0].m = 2.0
    node_size = ti.field(ti.u8, shape=count)
    node_size[0] = 1
    lnid = ti.field(int, shape=count)
    shape = ti.field(float, shape=count)
    shape[0] = 1.0
    dshape = ti.Vector.field(dimension, float, shape=count)
    scene = SimpleNamespace(
        particle=particle,
        particleNum=particle_num,
        node=node,
        mass_cut_off=0.0,
        is_rigid=ti.field(ti.u8, shape=1),
        element=SimpleNamespace(
            grid_nodes=1, gnum=cnum, LnID=lnid, shape_fn=shape, dshape_fn=dshape, node_size=node_size
        ),
    )
    return scene, state, split


@ti.data_oriented
class _VolumeMaterial:
    @ti.func
    def update_particle_volume_2D(self, np, gradient, state: ti.template(), dt: ti.template()):
        return (ti.Matrix.identity(float, 2) + gradient * dt[None]).determinant()

    @ti.func
    def update_particle_volume(self, np, gradient, state: ti.template(), dt: ti.template()):
        return (ti.Matrix.identity(float, 3) + gradient * dt[None]).determinant()


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_implicit_split_preserves_reference_volume_at_next_newton_update(taichi_runtime, dimension):
    scene, state, split = _split_scene(dimension)
    split()
    count = int(scene.particleNum[0])
    np.testing.assert_allclose(scene.particle.vol.to_numpy(), 1.0 / count)
    np.testing.assert_allclose(scene.particle.vol0.to_numpy(), 1.0 / count)
    np.testing.assert_allclose(scene.particle.m.to_numpy().sum(), 2.0)
    np.testing.assert_allclose(state.history.to_numpy(), 7.0)
    ids = ti.field(int, shape=count)
    ids.from_numpy(np.arange(count, dtype=np.int32))
    dt = ti.field(float, shape=())
    dt[None] = 0.01
    update = kernel_update_displacement_gradient_2D if dimension == 2 else kernel_update_displacement_gradient
    update(
        1,
        0,
        count,
        dt,
        scene.node,
        scene.particle,
        ids,
        _VolumeMaterial(),
        state,
        scene.element.LnID,
        scene.element.dshape_fn,
        scene.element.node_size,
    )
    np.testing.assert_allclose(scene.particle.vol.to_numpy().sum(), 1.0)


@pytest.mark.isolated_dimension(2)
def test_split_with_traction_preserves_distinct_reference_volume(taichi_runtime):
    scene, _, split = _split_scene(2, traction=True)
    scene.particle[0].vol0 = 0.8
    split()
    np.testing.assert_allclose(scene.particle.vol.to_numpy(), 0.25)
    np.testing.assert_allclose(scene.particle.vol0.to_numpy(), 0.2)


@pytest.mark.isolated_dimension(2)
@pytest.mark.parametrize("engine_type", [ULExplicitEngine, TLExplicitEngine])
@pytest.mark.parametrize("mapping", ["usl_updating", "usf_updating", "musl_updating"])
def test_explicit_refinement_preserves_uniform_pressure(taichi_runtime, engine_type, mapping):
    scene, _, split = _split_scene(2, implicit=False)
    engine = object.__new__(engine_type)
    for callback in (
        "calculate_interpolation",
        "update_adaptive_interpolation",
        "compute_nodal_kinematic",
        "compute_grid_velcity",
        "compute_forces",
        "apply_particle_traction_constraints",
        "apply_traction_constraints",
        "apply_absorbing_constraints",
        "compute_grid_kinematic",
        "pre_contact_calculate",
        "apply_kinematic_constraints",
        "apply_adaptive_constraints",
        "apply_adaptive_velocity_constraints",
        "apply_dirichlet_constraints",
        "compute_contact_force_",
        "compute_particle_kinematic",
        "compute_velocity_gradient",
        "compute_stress_strains",
        "compute_affine_velocity_gradient",
        "postmapping_grid_velocity",
    ):
        setattr(engine, callback, lambda *a: None)
    engine.pressure_smoothing_ = engine.pressure_smoothing
    engine.update_grid_refinement_profiled = split
    sims = SimpleNamespace(current_step=0, timer=SimpleNamespace(begin=lambda *a: None, end=lambda *a: None))
    getattr(engine, mapping)(sims, scene)
    assert scene.particleNum[0] == 4
    np.testing.assert_allclose(scene.particle.stress.to_numpy()[:, :3], 12.0)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("max_level", [1, 2])
def test_hanging_constraints_resolve_only_to_independent_nodes(dimension, max_level):
    cnum = np.array([2] * dimension)
    levels = np.array([0, 1, 1, 2] if dimension == 2 else [0, 0, 0, 1, 0, 1, 1, 2], dtype=np.uint8)
    levels = np.minimum(levels, max_level)
    nodes = AdaptiveNodeMap(cnum, 0, 1.0, False, max_level)
    nodes.ensure_refined_cells(levels, "Linear")
    slaves, masters, weights, _ = build_hanging_constraint_table(
        levels,
        cnum,
        nodes.fine_gnum,
        nodes.logical_to_compact,
        nodes.compact_to_logical,
        nodes.next_compact_id,
        2,
        max_level,
    )
    assert len(slaves) > 0
    assert not set(masters[weights > 0]) & set(slaves)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    coords = np.array(np.unravel_index(nodes.compact_to_logical[: nodes.next_compact_id], nodes.fine_gnum, order="F")).T
    for slave, ids, fractions in zip(slaves, masters, weights):
        used = fractions > 0
        np.testing.assert_allclose(fractions[used] @ coords[ids[used]], coords[slave])


@ti.func
def _test_stiffness(a, b, material, stress):
    block = ti.Vector.zero(float, GlobalVariable.DIMENSION**2)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        block[d * GlobalVariable.DIMENSION + d] = a.dot(b)
    return block


@pytest.mark.parametrize("dimension", DIMENSIONS)
@pytest.mark.parametrize("sparse", [False, True])
def test_adaptive_coo_matches_dense_oracle_after_support_growth(taichi_runtime, dimension, sparse):
    count = 2**dimension
    capacity = count + 3
    particle = ti.Struct.field(dict(active=ti.i32, bodyID=ti.i32, vol=float, stress=ti.types.vector(6, float)), shape=2)
    particle[0].active, particle[0].vol = 1, 0.7
    particle[1].active, particle[1].vol = 1, 0.3
    node_size = ti.field(int, shape=2)
    lnid = ti.field(int, shape=2 * capacity)
    gradients = ti.Vector.field(dimension, float, shape=2 * capacity)
    flag = ti.field(int, shape=2 * capacity)
    flag.from_numpy(np.arange(2 * capacity, dtype=np.int32) * dimension)
    element = SimpleNamespace(
        adaptive=True,
        gridSum=2 * capacity,
        grid_nodes=capacity,
        influenced_dofs=capacity * dimension,
        influenced_node=2,
        gnum=ti.Vector([5] * dimension),
        node_size=node_size,
        LnID=lnid,
        dshape_fn=gradients,
        flag=flag,
    )
    scene = SimpleNamespace(
        element=element,
        sparse_grid=None,
        particle=particle,
        particleNum=np.array([2]),
        material=SimpleNamespace(stiffness_matrix=ti.field(float, shape=2)),
    )
    sims = SimpleNamespace(
        dimension=dimension,
        sparse_grid=sparse,
        dof_multiplier=1,
        max_particle_num=2,
        assemble_type="COO",
        calculate_reaction_force=False,
        hash_triplet_matrix_symmetric=False,
        ndisplacement=2,
        timer=SimpleNamespace(begin=lambda *a: None, end=lambda *a: None),
    )
    dofs = dimension * element.gridSum
    iterator = MomentumConservation()
    iterator.set_matrix_vector(dofs, sims, scene)
    iterator.operator = SimpleNamespace(active_dofs=dofs)
    iterator.assemble_stiffness_matrix = _test_stiffness
    iterator.mass_matrix.fill(2.0)
    previous_capacity = iterator.sparse_matrix.capacity
    for support in (count, capacity):
        node_size.fill(support)
        ids = np.zeros(2 * capacity, dtype=np.int32)
        # Scrambled compact IDs deliberately have no Cartesian neighborhood meaning.
        ids[:support] = np.arange(support) * 2
        ids[capacity : capacity + support] = np.arange(support)[::-1] * 2
        lnid.from_numpy(ids)
        grad = np.arange(2 * capacity * dimension, dtype=np.float32).reshape(-1, dimension) / 13.0
        gradients.from_numpy(grad)
        iterator.assemble_coo_matrix(sims, scene)
        if ti.lang.impl.current_cfg().arch != ti.metal:
            assert iterator.sparse_matrix.capacity > previous_capacity
        previous_capacity = iterator.sparse_matrix.capacity
        expected = 2.0 * np.eye(dofs)
        for p, volume in enumerate((0.7, 0.3)):
            offsets = p * capacity + np.arange(support)
            local = np.kron(grad[offsets] @ grad[offsets].T, np.eye(dimension)) * volume
            indices = (ids[offsets, None] * dimension + np.arange(dimension)).ravel()
            expected[np.ix_(indices, indices)] += local
        actual = iterator.sparse_matrix._to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        assert iterator.coo_base_nnz == dofs + 2 * (support * dimension) ** 2
        assert iterator.sparse_matrix.capacity >= iterator.coo_base_nnz + sims.ndisplacement
        # Reassembly must not accumulate the previous Newton iteration's entries.
        iterator.assemble_coo_matrix(sims, scene)
        np.testing.assert_allclose(iterator.sparse_matrix._to_numpy(), expected, rtol=2e-6, atol=2e-6)


def test_adaptive_coo_reports_triplet_count_overflow_before_assembly(taichi_runtime):
    from src.mpm.engines.AssembleMatrixKernel import kernel_adaptive_coo_offsets

    particle = ti.Struct.field(dict(active=ti.i32), shape=1)
    particle[0].active = 1
    size = ti.field(ti.u8, shape=1)
    size[0] = 1
    offsets = ti.field(int, shape=1)
    counts = kernel_adaptive_coo_offsets(np.iinfo(np.int32).max - 1, 1, particle, size, offsets)
    assert int(counts[1]) == 1


@pytest.mark.isolated_dimension(2)
@pytest.mark.parametrize("traction", [False, True])
def test_split_conserves_dem_contact_history(taichi_runtime, traction):
    scene, _, split = _split_scene(2, implicit=False, traction=traction, contact=True)
    split()
    np.testing.assert_allclose(scene.particle.contact_traction.to_numpy(), [[3.0, -1.0]] * 4)
