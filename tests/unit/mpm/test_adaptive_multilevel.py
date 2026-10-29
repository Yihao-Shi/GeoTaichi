"""Mixed 0/1/2 grids must reproduce affine fields and retain split scheduling."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.mpm.Simulation import Simulation
from src.mpm.elements.AdaptiveQuadrilateralElement import AdaptiveQuadrilateralElement
from src.mpm.elements.AdaptiveHexahedronElement import AdaptiveHexahedronElement
from src.mpm.elements.AdaptiveNodeMap import (
    AdaptiveNodeMap,
    build_bridging_node_weights,
    build_hanging_constraint_table,
)

DIMS = [
    pytest.param(2, marks=pytest.mark.isolated_dimension(2)),
    pytest.param(3, marks=pytest.mark.isolated_dimension(3)),
]


def make_element(dimension, shape="Linear", mode="ShapeFunction", **options):
    sims = Simulation()
    sims.dimension = dimension
    sims.domain = ti.Vector([0.8] * dimension, dt=ti.f64)
    sims.max_particle_num = 128
    sims.solver_type = "Explicit"
    sims.material_type = "Solid"
    sims.shape_function = shape
    sims.ptraction_method = "Stable"
    sims.mapping = "USL"
    sims.velocity_projection_scheme = None
    config = dict(
        MaxLevel=2,
        RefineRatio=1.0,
        RefineInterval=100,
        ParticleSplitInterval=2,
        RefineThreshold=1.0,
        RefineCriterion="EquivalentStress",
        BufferCells=0,
        HangingConstraintMode=mode,
        HangingShapeCapacityFactor=3 if dimension == 3 and shape == "CubicBSpline" else 2**dimension,
    )
    config.update(options)
    cls = AdaptiveQuadrilateralElement if dimension == 2 else AdaptiveHexahedronElement
    element = cls("Q4N2D" if dimension == 2 else "R8N3D", 1, 1, config)
    element.create_nodes(sims, ti.Vector([0.1] * dimension, dt=ti.f64))
    element.set_characteristic_length(sims)
    element.element_initialize(sims)
    return sims, element


def balance_levels_on_device(element, dimension, levels):
    if dimension == 2:
        from src.mpm.elements.AdaptiveQuadrilateralKernel import balance_refined_cells_2d as balance
        from src.mpm.elements.AdaptiveQuadrilateralKernel import merge_refined_cells_2d as merge
    else:
        from src.mpm.elements.AdaptiveHexahedronKernel import balance_refined_cells as balance
        from src.mpm.elements.AdaptiveHexahedronKernel import merge_refined_cells as merge
    element.refined_cell.from_numpy(np.asarray(levels, dtype=np.uint8))
    source, target = element.refined_cell, element.refined_cell_buffer
    for _ in range(element.max_level - 1):
        balance(element.coarse_cnum, element.max_level, source, target)
        source, target = target, source
    if source is not element.refined_cell:
        merge(element.refined_cell, source)
    return element.refined_cell.to_numpy()


@pytest.mark.parametrize("dimension", DIMS)
@pytest.mark.parametrize("shape", ["Linear", "GIMP", "QuadBSpline", "CubicBSpline"])
def test_mixed_levels_reproduce_affine_fields(taichi_runtime, dimension, shape):
    sims, element = make_element(dimension, shape)
    cells = np.zeros(tuple(element.coarse_cnum), dtype=np.uint8)
    cells[2:6] = 1
    cells[3:5] = 2
    levels = cells.flatten(order="F")
    element.node_map.ensure_refined_cells(levels, shape)
    element.refined_cell.from_numpy(levels)
    element._update_hanging_constraint_table(levels)
    points = np.array([[x] + [0.347] * (dimension - 1) for x in (0.195, 0.205, 0.295, 0.305, 0.495, 0.505)])
    particle = ti.Struct.field(dict(x=ti.types.vector(dimension, float)), shape=128)
    positions = np.zeros((128, dimension))
    positions[: len(points)] = points
    particle.x.from_numpy(positions)
    element.calculate_characteristic_length(sims, len(points), particle, np.full((128, dimension), 0.002))
    element.calculate(np.array([len(points)], dtype=np.int32), particle)
    ids = element.LnID.to_numpy().reshape(128, -1)
    weights = element.shape_fn.to_numpy().reshape(128, -1)
    gradients = element.dshape_fn.to_numpy().reshape(128, -1, dimension)
    counts = element.node_size.to_numpy()
    logical = element.compact_to_logical.to_numpy()
    assert element.hanging_node_count_host > 0
    for p, point in enumerate(points):
        count = int(counts[p])
        assert 0 < count <= element.grid_nodes
        selected = ids[p, :count]
        coords = np.array(np.unravel_index(logical[selected], tuple(element.gnum), order="F")).T * np.array(
            element.fine_grid_size
        )
        np.testing.assert_allclose(weights[p, :count].sum(), 1.0, atol=1e-8)
        np.testing.assert_allclose(gradients[p, :count].sum(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(weights[p, :count] @ coords, point, atol=1e-8)
        np.testing.assert_allclose(coords.T @ gradients[p, :count], np.eye(dimension), atol=1e-8)
        assert np.all(element.hanging_lookup.to_numpy()[selected] < 0)


@pytest.mark.parametrize("dimension", DIMS)
def test_split_interval_rechecks_particles_after_previous_empty_pass(taichi_runtime, dimension):
    _, element = make_element(dimension, FollowRefinedParticles=False)
    element.last_reported_refined_cells = 1
    element.last_observed_particle_num = 1
    element.remaining_unrefined_particles = 0
    # A particle can move into an already refined cell without changing grid or count.
    assert element.should_update_refinement(SimpleNamespace(current_step=2), np.array([1]))


@pytest.mark.parametrize("dimension", DIMS)
@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("step", [20, 100])
def test_following_split_particles_does_not_dilate_a_new_buffer(taichi_runtime, dimension, level, step, monkeypatch):
    sims, element = make_element(dimension, BufferCells=1)
    particle = ti.Struct.field(
        dict(
            active=int,
            materialID=int,
            particleID=int,
            m=float,
            vol=float,
            x=ti.types.vector(dimension, float),
            v=ti.types.vector(dimension, float),
            velocity_gradient=ti.types.matrix(dimension, dimension, float),
        ),
        shape=128,
    )
    for p, x in enumerate([0.35, 0.35 + 0.1 * level]):
        particle[p].active = particle[p].materialID = 1
        particle[p].x = [x] + [0.35] * (dimension - 1)
        particle[p].m, particle[p].vol = 1.0, 0.001
        element.particle_size[p] = [0.005] * dimension
    element.particle_refined[0] = level
    state = ti.Struct.field(dict(history=float), shape=128)
    scene = SimpleNamespace(
        particle=particle,
        particleNum=np.array([2], dtype=np.int32),
        material=SimpleNamespace(
            stateVars=state,
            matProps=SimpleNamespace(size=lambda: 1),
            update_material_mapping_device=lambda *args: None,
        ),
        boundary=SimpleNamespace(particle_traction=None),
    )
    levels = np.zeros(tuple(element.coarse_cnum), dtype=np.uint8)
    levels[(2,) + (3,) * (dimension - 1)] = level
    balance_levels_on_device(element, dimension, levels.flatten(order="F"))
    element.last_reported_refined_cells = int(np.count_nonzero(element.refined_cell.to_numpy()))
    element.last_observed_particle_num = 2
    # The first particle crossed into cell 3; cell 3+level is outside its required 2:1 transition.
    for current_step in [step, step + 20]:
        sims.current_step = current_step
        with monkeypatch.context() as guard:
            forbid_bulk_transfers(guard, element)
            element.update_refinement(sims, scene)
        cells = element.refined_cell.to_numpy().reshape(tuple(element.coarse_cnum), order="F")
        assert cells[(3,) + (3,) * (dimension - 1)] == level
        assert cells[(3 + level,) + (3,) * (dimension - 1)] == 0
        assert scene.particleNum[0] == 2
    np.testing.assert_allclose(particle.m.to_numpy()[:2].sum(), 2.0)


@pytest.mark.isolated_dimension(2)
@pytest.mark.parametrize("criterion", ["PlasticDevStrain", "EquivalentPlasticStrain"])
def test_mohr_coulomb_plastic_indicator_reads_epdstrain_not_stress(taichi_runtime, criterion):
    from src.mpm.elements.AdaptiveQuadrilateralKernel import mark_refined_cells_epdstrain_2d

    _, element = make_element(2, RefineCriterion=criterion, RefineThreshold=0.01)
    material = type("MohrCoulombModel", (), {"is_soft": False})()
    marker = element._select_refinement_marker(material)
    assert marker is mark_refined_cells_epdstrain_2d
    particle = ti.Struct.field(
        dict(active=int, materialID=int, x=ti.types.vector(2, float), stress=ti.types.vector(6, float)),
        shape=2,
    )
    state = ti.Struct.field(dict(epdstrain=float), shape=2)
    for i in range(2):
        particle[i].active = particle[i].materialID = 1
        particle[i].x = [0.05 + 0.6 * i] * 2
    particle[0].stress = [1e6, 0, 0, 0, 0, 0]
    particle[1].stress = [0, 0, 0, 0, 0, 0]
    state[0].epdstrain, state[1].epdstrain = 0.005, 0.02
    marker(
        0.01,
        2,
        element.coarse_igrid_size,
        element.coarse_cnum,
        2,
        particle,
        state,
        element.refined_cell,
        element.refinement_seed,
        1,
    )
    seeds = element.refinement_seed.to_numpy()
    assert seeds[0] == 0
    assert seeds[6 + 6 * int(element.coarse_cnum[0])] == 1
    assert np.count_nonzero(seeds) == 1


@pytest.mark.parametrize("dimension", DIMS)
def test_splitting_cannot_spread_refinement_without_particle_motion(taichi_runtime, dimension, monkeypatch):
    sims, element = make_element(dimension, MaxLevel=1, BufferCells=0)
    particle = ti.Struct.field(
        dict(
            active=int,
            materialID=int,
            particleID=int,
            m=float,
            vol=float,
            x=ti.types.vector(dimension, float),
            v=ti.types.vector(dimension, float),
            velocity_gradient=ti.types.matrix(dimension, dimension, float),
        ),
        shape=128,
    )
    positions = np.array([[(p + 0.5) * 0.025 + 0.0075] + [0.35] * (dimension - 1) for p in range(16)])
    for p, position in enumerate(positions):
        particle[p].active = particle[p].materialID = 1
        particle[p].particleID = p
        particle[p].x = position
        particle[p].m, particle[p].vol = 1.0, 0.025**dimension
        element.particle_size[p] = [0.0125] * dimension
    state = ti.Struct.field(dict(history=float), shape=128)
    scene = SimpleNamespace(
        particle=particle,
        particleNum=np.array([16], dtype=np.int32),
        material=SimpleNamespace(
            stateVars=state,
            matProps=SimpleNamespace(size=lambda: 1),
            update_material_mapping_device=lambda *args: None,
        ),
        boundary=SimpleNamespace(particle_traction=None),
    )
    initial_grid = np.zeros(tuple(element.coarse_cnum), dtype=np.uint8)
    initial_grid[(0,) + (3,) * (dimension - 1)] = 1
    element.refined_cell.from_numpy(initial_grid.flatten(order="F"))
    element.last_reported_refined_cells = 1
    element.last_observed_particle_num = 16
    # No advection or physical indicator: topology may not grow just because children are created.
    for step in [2, 20, 40, 60, 80, 100]:
        sims.current_step = step
        with monkeypatch.context() as guard:
            forbid_bulk_transfers(guard, element)
            element.update_refinement(sims, scene)
        count = int(scene.particleNum[0])
        cells = element.refined_cell.to_numpy().reshape(tuple(element.coarse_cnum), order="F")
        np.testing.assert_array_equal(cells, initial_grid)
        child_cells = np.floor(particle.x.to_numpy()[:count] / 0.1).astype(int)
        assert np.all(cells[tuple(child_cells.T)] >= element.particle_refined.to_numpy()[:count])
    assert count == 16 + 3 * (2**dimension - 1)
    assert element.particle_refined[3] == 0
    if dimension == 2:
        from src.mpm.elements.AdaptiveQuadrilateralKernel import count_particle_traction_constraints_to_split_2d

        head, next_id = ti.field(int, shape=128), ti.field(int, shape=1)
        head.fill(-1)
        head[3], next_id[0] = 0, -1
        assert (
            count_particle_traction_constraints_to_split_2d(
                element.coarse_igrid_size,
                element.coarse_cnum,
                count,
                particle,
                head,
                next_id,
                element.refined_cell,
                element.particle_refined,
                element.particle_size,
            )
            == 0
        )
    # Deferred edge parents split normally once the neighboring cell is refined.
    initial_grid[(1,) + (3,) * (dimension - 1)] = 1
    element.refined_cell.from_numpy(initial_grid.flatten(order="F"))
    sims.current_step = 120
    with monkeypatch.context() as guard:
        forbid_bulk_transfers(guard, element)
        element.update_refinement(sims, scene)
    count = int(scene.particleNum[0])
    assert element.particle_refined[3] == 1
    assert count == 16 + 7 * (2**dimension - 1)
    np.testing.assert_allclose(particle.m.to_numpy()[:count].sum(), 16.0)
    np.testing.assert_allclose(particle.vol.to_numpy()[:count].sum(), 16 * 0.025**dimension)
    np.testing.assert_allclose(particle.m.to_numpy()[:count] @ particle.x.to_numpy()[:count], positions.sum(axis=0))


@pytest.mark.parametrize("dimension", DIMS)
def test_material_indicator_keeps_its_configured_buffer(taichi_runtime, dimension, monkeypatch):
    sims, element = make_element(dimension, BufferCells=1, RefineParticles=False)
    model = type("LinearElasticModel", (), {})()

    class MaterialTable:
        def size(self):
            return 2

        def __getitem__(self, material_id):
            return model

    particle = ti.Struct.field(
        dict(active=int, materialID=int, x=ti.types.vector(dimension, float), stress=ti.types.vector(6, float)),
        shape=1,
    )
    particle[0].active = particle[0].materialID = 1
    particle[0].x, particle[0].stress = [0.35] * dimension, [2, 0, 0, 0, 0, 0]
    scene = SimpleNamespace(
        particle=particle,
        particleNum=np.array([1], dtype=np.int32),
        material=SimpleNamespace(matProps=MaterialTable(), stateVars=ti.field(float, shape=1)),
    )
    sims.current_step = 100
    with monkeypatch.context() as guard:
        forbid_bulk_transfers(guard, element)
        element.update_refinement(sims, scene)
    levels = element.refined_cell.to_numpy().reshape(tuple(element.coarse_cnum), order="F")
    assert np.count_nonzero(levels) == 3**dimension
    np.testing.assert_array_equal(levels[(slice(2, 5),) * dimension], 1)


@pytest.mark.parametrize("dimension", [2, 3])
def test_multilevel_limits_and_bridging_rejected_before_allocation(dimension):
    cls = AdaptiveQuadrilateralElement if dimension == 2 else AdaptiveHexahedronElement
    for config, match in [({"MaxLevel": 3}, "MaxLevel"), ({"MaxLevel": 2, "BridgingDomain": True}, "MaxLevel=1")]:
        with pytest.raises(ValueError, match=match):
            cls("Q4N2D" if dimension == 2 else "R8N3D", 1, 1, config)


@pytest.mark.parametrize("dimension", DIMS)
def test_all_refinement_indicators_filter_material_and_stop_at_maximum(taichi_runtime, dimension):
    from src.mpm.elements import AdaptiveQuadrilateralKernel as k2
    from src.mpm.elements import AdaptiveHexahedronKernel as k3

    kernels = k2 if dimension == 2 else k3
    suffix = "_2d" if dimension == 2 else ""
    current = ti.field(ti.u8, shape=2**dimension)
    seeds = ti.field(ti.u8, shape=2**dimension)
    state = ti.Struct.field(dict(epdstrain=float, epstrain=float, strain=ti.types.vector(6, float)), shape=2)
    for i in range(2):
        state[i].epdstrain = state[i].epstrain = 10.0
        state[i].strain = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for criterion in ("epdstrain", "epstrain", "strain", "deviatoric_strain", "stress", "matrix_stress", "softening"):
        stress_type = (
            ti.types.matrix(dimension, dimension, float) if criterion == "matrix_stress" else ti.types.vector(6, float)
        )
        particles = ti.Struct.field(
            dict(active=int, materialID=int, x=ti.types.vector(dimension, float), stress=stress_type), shape=2
        )
        for i in range(2):
            particles[i].active = 1
            particles[i].materialID = i + 1
            particles[i].x = [0.25 + 0.5 * i] * dimension
            stress = np.zeros((dimension, dimension) if criterion == "matrix_stress" else (6,))
            stress.flat[0] = 10.0
            particles[i].stress = stress
        marker = getattr(kernels, "mark_refined_cells_" + criterion + suffix)
        for level in (0, 1, 2):
            current.fill(level)
            seeds.fill(0)
            marker(
                1.0, 2, ti.Vector([2.0] * dimension), ti.Vector([2] * dimension), 2, particles, state, current, seeds, 1
            )
            expected = np.zeros(2**dimension, dtype=np.uint8)
            if level < 2:
                expected[0] = level + 1
            np.testing.assert_array_equal(seeds.to_numpy(), expected)


@pytest.mark.parametrize("dimension", DIMS)
def test_boundary_ids_stable_through_multilevel_growth(taichi_runtime, dimension):
    _, element = make_element(dimension)
    start = np.array([0.2] * dimension)
    end = np.array([0.4] * dimension)
    end[-1] = start[-1]
    ids = element.get_boundary_nodes(start, end)
    logical = element.node_map.compact_to_logical[ids].copy()
    levels = np.ones(element.cellSum, dtype=np.uint8)
    levels[0] = 2
    levels = balance_levels_on_device(element, dimension, levels)
    element.node_map.ensure_refined_cells(levels, "Linear")
    np.testing.assert_array_equal(element.node_map.compact_to_logical[ids], logical)
    coords = np.array(np.unravel_index(logical, tuple(element.gnum), order="F")).T * np.array(element.fine_grid_size)
    assert np.all(coords >= start - 1e-7) and np.all(coords <= end + 1e-7)
    np.testing.assert_allclose(coords[:, -1], start[-1], atol=1e-7)


@pytest.mark.parametrize("dimension", DIMS)
def test_refined_cell_balance_is_2_to_1_and_idempotent(taichi_runtime, dimension):
    _, element = make_element(dimension)
    shape = tuple(element.coarse_cnum)
    levels = np.zeros(shape, dtype=np.uint8)
    center = tuple(size // 2 for size in shape)
    levels[center] = element.max_level

    balanced = balance_levels_on_device(element, dimension, levels.flatten(order="F")).reshape(shape, order="F")

    assert np.all(balanced >= levels)
    assert np.all(balanced <= element.max_level)
    for axis in range(dimension):
        lower = [slice(None)] * dimension
        upper = [slice(None)] * dimension
        lower[axis] = slice(None, -1)
        upper[axis] = slice(1, None)
        assert np.max(np.abs(balanced[tuple(lower)].astype(int) - balanced[tuple(upper)].astype(int))) <= 1
    np.testing.assert_array_equal(
        balance_levels_on_device(element, dimension, balanced.flatten(order="F")), balanced.flatten(order="F")
    )


def forbid_bulk_transfers(monkeypatch, element):
    def fail(*args, **kwargs):
        pytest.fail("Adaptive topology attempted a bulk host/device transfer")

    for field_type in {type(element.refined_cell), type(element.compact_to_logical), type(element.calLength)}:
        monkeypatch.setattr(field_type, "to_numpy", fail)
        monkeypatch.setattr(field_type, "from_numpy", fail)


@pytest.mark.parametrize("dimension", DIMS)
@pytest.mark.parametrize("max_level", [1, 2])
def test_device_topology_matches_reference_without_bulk_transfers(taichi_runtime, dimension, max_level, monkeypatch):
    _, element = make_element(dimension, shape="GIMP", MaxLevel=max_level)
    shape = tuple(element.coarse_cnum)
    levels = np.zeros(shape, dtype=np.uint8)
    levels[2:5] = 1
    previous_ids = element.compact_to_logical.to_numpy()[: element.node_map.next_compact_id]
    reference = AdaptiveNodeMap(element.coarse_cnum, element.ghost_cell, 1.0, max_level=max_level)
    for growth in range(2):
        if growth:
            levels[1:6] = 1
            levels[3:4] = max_level
        flat = levels.flatten(order="F")
        element.refined_cell.from_numpy(flat)
        with monkeypatch.context() as guard:
            forbid_bulk_transfers(guard, element)
            element.node_map.ensure_refined_cells_device(element.refined_cell, "GIMP")
            element._update_hanging_constraint_table(element.refined_cell)
        active = element.node_map.next_compact_id
        logical_ids = element.compact_to_logical.to_numpy()[:active]
        np.testing.assert_array_equal(logical_ids[: previous_ids.size], previous_ids)
        previous_ids = logical_ids.copy()

        reference.ensure_refined_cells(flat, "GIMP")
        np.testing.assert_array_equal(logical_ids, reference.compact_to_logical[: reference.next_compact_id])
        mapping = element.logical_to_compact.to_numpy()
        expected = build_hanging_constraint_table(
            flat, element.coarse_cnum, element.gnum, mapping, logical_ids, active, max_level=max_level
        )
        count = element.hanging_node_count_host
        slaves = element.hanging_node_id.to_numpy()[:count]
        masters = element.hanging_master_id.to_numpy()[:count]
        weights = element.hanging_master_weight.to_numpy()[:count]
        expected_rows = {
            int(slave): {int(m): w for m, w in zip(ids, row) if w > 0} for slave, ids, row in zip(*expected[:3])
        }
        assert set(slaves) == set(expected_rows)
        for slave, ids, row in zip(slaves, masters, weights):
            actual = {int(m): w for m, w in zip(ids, row) if w > 0}
            assert actual.keys() == expected_rows[int(slave)].keys()
            for master, weight in actual.items():
                assert weight == pytest.approx(expected_rows[int(slave)][master])
        touched = element.hanging_touched_node_id.to_numpy()[: element.hanging_touched_node_count_host]
        # The old builder also listed coarse identity rows; only actual constraints need updates.
        required_touched = np.unique(np.concatenate((slaves, masters[weights > 0])))
        np.testing.assert_array_equal(np.sort(touched), required_touched)
    with monkeypatch.context() as guard:
        forbid_bulk_transfers(guard, element)
        element.node_map.ensure_refined_cells_device(element.refined_cell, "GIMP")
    assert int(element.node_map.device.request_count[None]) == 0
    assert element.node_map.next_compact_id == active


@pytest.mark.parametrize("dimension", DIMS)
def test_device_bridging_weights_match_reference_without_bulk_transfers(taichi_runtime, dimension, monkeypatch):
    _, element = make_element(dimension, mode="Penalty", MaxLevel=1, BridgingDomain=True, BridgingCells=3)
    levels = np.zeros(tuple(element.coarse_cnum), dtype=np.uint8)
    levels[2:5] = 1
    flat = levels.flatten(order="F")
    expected = build_bridging_node_weights(flat, element.coarse_cnum, 3)
    element.refined_cell.from_numpy(flat)
    with monkeypatch.context() as guard:
        forbid_bulk_transfers(guard, element)
        element.node_map.ensure_refined_cells_device(element.refined_cell, "GIMP")
        element.node_map.device.build_bridge_weights(element.refined_cell, 3, element.bridge_coarse_weight)
    np.testing.assert_allclose(element.bridge_coarse_weight.to_numpy(), expected, atol=1e-14)


@pytest.mark.parametrize("dimension", DIMS)
def test_device_node_allocation_rejects_capacity_overflow(taichi_runtime, dimension):
    _, element = make_element(dimension, MaxLevel=2, RefineRatio=0.01)
    element.refined_cell.fill(2)
    initial_count = element.node_map.next_compact_id
    with pytest.raises(RuntimeError, match="dense node capacity.*exceeded"):
        element.node_map.ensure_refined_cells_device(element.refined_cell, "GIMP")
    assert element.node_map.next_compact_id == initial_count


@pytest.mark.parametrize("dimension", DIMS)
def test_device_hanging_topology_rejects_missing_masters(taichi_runtime, dimension):
    _, element = make_element(dimension, MaxLevel=1)
    levels = np.zeros(tuple(element.coarse_cnum), dtype=np.uint8)
    levels[2:5] = 1
    element.refined_cell.from_numpy(levels.flatten(order="F"))
    element.node_map.ensure_refined_cells_device(element.refined_cell, "Linear")
    element.logical_to_compact.fill(-1)
    with pytest.raises(RuntimeError, match="missing master node"):
        element._update_hanging_constraint_table(element.refined_cell)


@pytest.mark.isolated_dimension(3)
@pytest.mark.parametrize("refine_particles", [False, True])
def test_3d_split_workspace_uses_parent_capacity_bound(taichi_runtime, refine_particles):
    sims, element = make_element(3, RefineParticles=refine_particles)
    expected = sims.max_particle_num // 8 if refine_particles else 1
    assert element.split_parent_capacity == expected
    assert element.split_parent_position.shape[0] == expected


@pytest.mark.parametrize(
    "material_ids", [[0, 2, 1, 1, 2, 1, 2, 1], [0, 2, 2, 2, 2, 2, 2, 2], [index % 3 for index in range(600)]]
)
def test_split_material_mapping_stays_on_device(taichi_runtime, monkeypatch, material_ids):
    from src.mpm.MaterialManager import MaterialHandle

    sims = Simulation()
    sims.max_particle_num, sims.max_material_num = max(12, len(material_ids)), 3
    materials = MaterialHandle(sims)
    particle = ti.Struct.field(dict(materialID=int), shape=sims.max_particle_num)
    for index, material_id in enumerate(material_ids):
        particle[index].materialID = material_id
    materials.update_material_mapping(particle, 4)

    def fail(*args, **kwargs):
        pytest.fail("Particle material regrouping attempted a bulk transfer")

    with monkeypatch.context() as guard:
        guard.setattr(type(materials.materialID), "to_numpy", fail)
        guard.setattr(type(materials.materialID), "from_numpy", fail)
        materials.update_material_mapping_device(particle, len(material_ids))

    particle_ids = materials.materialID.to_numpy()[: len(material_ids)]
    np.testing.assert_array_equal(particle_ids, np.argsort(material_ids, kind="stable"))
    np.testing.assert_array_equal(np.sort(particle_ids), np.arange(len(material_ids)))
    grouped = np.asarray(material_ids)[particle_ids]
    np.testing.assert_array_equal(grouped, np.sort(material_ids))
    np.testing.assert_array_equal(materials.mapping, np.cumsum(np.bincount(material_ids)))
    np.testing.assert_array_equal(materials.materialID_numpy, material_ids)


@pytest.mark.parametrize("dimension", DIMS)
def test_device_penalty_counts_preserve_momentum(taichi_runtime, dimension, monkeypatch):
    _, element = make_element(dimension, mode="Penalty", MaxLevel=1, Penalty=0.2)
    levels = np.zeros(tuple(element.coarse_cnum), dtype=np.uint8)
    levels[2:5] = 1
    element.refined_cell.from_numpy(levels.flatten(order="F"))
    element.node_map.ensure_refined_cells_device(element.refined_cell, "Linear")
    element._update_hanging_constraint_table(element.refined_cell)
    node = ti.Struct.field(
        dict(m=float, momentum=ti.types.vector(dimension, float), force=ti.types.vector(dimension, float)),
        shape=(element.gridSum, 1),
    )
    node.m.fill(1.0)
    slave = int(element.hanging_node_id[0])
    node[slave, 0].momentum = [1.0] + [0.0] * (dimension - 1)
    before = node.momentum.to_numpy()
    dt = ti.field(float, shape=())
    dt[None] = 1e-4
    with monkeypatch.context() as guard:
        forbid_bulk_transfers(guard, element)
        element.apply_hanging_constraints(1e-12, dt, node, None)
    after = node.momentum.to_numpy()
    tolerance = 1e-7 if ti.lang.impl.current_cfg().default_fp == ti.f32 else 1e-13
    np.testing.assert_allclose(after.sum(axis=(0, 1)), before.sum(axis=(0, 1)), atol=tolerance)
    assert np.linalg.norm(after) < np.linalg.norm(before)


@pytest.mark.parametrize(
    "dimension,interior_only,limited_capacity",
    [
        pytest.param(2, False, False, marks=pytest.mark.isolated_dimension(2)),
        pytest.param(3, False, False, marks=pytest.mark.isolated_dimension(3)),
        pytest.param(3, True, False, marks=pytest.mark.isolated_dimension(3)),
        pytest.param(2, False, True, marks=pytest.mark.isolated_dimension(2)),
        pytest.param(3, False, True, marks=pytest.mark.isolated_dimension(3)),
    ],
)
def test_existing_level_two_grid_splits_new_arrivals_on_successive_intervals(
    taichi_runtime, dimension, interior_only, limited_capacity
):
    sims, element = make_element(
        dimension, FollowRefinedParticles=False, ParticleSplitBatch=2, SplitInteriorOnly=interior_only
    )
    particle = ti.Struct.field(
        dict(
            active=int,
            materialID=int,
            particleID=int,
            m=float,
            vol=float,
            x=ti.types.vector(dimension, float),
            v=ti.types.vector(dimension, float),
            velocity_gradient=ti.types.matrix(dimension, dimension, float),
        ),
        shape=128,
    )
    particle[0].active = particle[0].materialID = 1
    particle[0].m, particle[0].vol = 2.0, 0.001
    particle[0].x = [0.05] * dimension
    particle[0].v = [0.2] * dimension
    element.particle_size[0] = [0.01] * dimension
    state = ti.Struct.field(dict(history=float), shape=128)
    state[0].history = 7.0
    scene = SimpleNamespace(
        particle=particle,
        particleNum=np.array([1], dtype=np.int32),
        material=SimpleNamespace(stateVars=state, update_material_mapping_device=lambda *args: None),
        boundary=SimpleNamespace(particle_traction=None),
    )
    element.refined_cell[0] = 2
    element.last_reported_refined_cells = 1
    element.last_observed_particle_num = 1
    element.remaining_unrefined_particles = 0
    if limited_capacity:
        sims.max_particle_num = 2
        sims.current_step = 2
        before = particle.m.to_numpy().copy()
        with pytest.raises((AssertionError, RuntimeError), match="max_particle_number|particle capacity"):
            element.update_refinement(sims, scene)
        assert scene.particleNum[0] == 1
        np.testing.assert_array_equal(particle.m.to_numpy(), before)
        return
    if interior_only:
        particle[0].vol = 0.00001
        sims.current_step = 2
        element.update_refinement(sims, scene)
        assert scene.particleNum[0] == 1
        particle[0].vol = 0.001
    for step, level in [(4, 1), (6, 2)] if interior_only else [(2, 1), (4, 2)]:
        sims.current_step = step
        element.update_refinement(sims, scene)
        count = int(scene.particleNum[0])
        assert count == 2 ** (dimension * level)
        np.testing.assert_array_equal(element.particle_refined.to_numpy()[:count], level)
        np.testing.assert_allclose(particle.m.to_numpy()[:count].sum(), 2.0)
        np.testing.assert_allclose(particle.vol.to_numpy()[:count].sum(), 0.001)
        np.testing.assert_allclose(state.history.to_numpy()[:count], 7.0)
        np.testing.assert_allclose(particle.v.to_numpy()[:count], 0.2)


@pytest.mark.isolated_dimension(3)
def test_u8_support_capacity_is_still_enforced(taichi_runtime):
    with pytest.raises(ValueError, match="u8 node_size limit"):
        make_element(3, "CubicBSpline", HangingShapeCapacityFactor=8)


@pytest.mark.isolated_dimension(2)
@pytest.mark.parametrize("axisymmetric", [False, True])
@pytest.mark.parametrize("nanson", [False, True])
def test_successive_split_preserves_particle_traction_force(taichi_runtime, axisymmetric, nanson):
    from src.mpm.elements.AdaptiveQuadrilateralKernel import (
        build_particle_traction_linked_list_2d,
        split_refined_particles_with_traction_2d,
    )
    from src.mpm.boundaries.BoundaryStrcut import ParticleLoad2D, ParticleLoadNanson2D

    particle = ti.Struct.field(
        dict(
            active=int,
            materialID=int,
            particleID=int,
            m=float,
            vol=float,
            x=ti.types.vector(2, float),
            v=ti.types.vector(2, float),
            velocity_gradient=ti.types.matrix(2, 2, float),
        ),
        shape=16,
    )
    particle[0].active = particle[0].materialID = 1
    particle[0].x = [0.5, 0.5]
    particle[0].m, particle[0].vol = 2.0, 0.04
    sizes = ti.Vector.field(2, float, shape=16)
    sizes[0] = [0.1, 0.1]
    length = ti.Vector.field(2, float, shape=16)
    state = ti.Struct.field(dict(history=float), shape=16)
    cells = ti.field(ti.u8, shape=1)
    cells[0] = 2
    level, refined = [ti.field(ti.u8, shape=16) for _ in range(2)]
    count, traction_count = [ti.field(int, shape=()) for _ in range(2)]
    count[None] = traction_count[None] = 1
    loads = (ParticleLoadNanson2D if nanson else ParticleLoad2D).field(shape=16)
    loads[0].pid = 0
    loads[0].traction = [10.0, 20.0]
    loads[0].psize = [0.08, 0.12]  # Load area can already be deformed independently of particle size.
    if nanson:
        loads[0].td = [[1.1, 0.2], [0.0, 0.9]]
    head, next_id = [ti.field(int, shape=16) for _ in range(2)]

    def total_force():
        n = int(traction_count[None])
        traction = loads.traction.to_numpy()[:n]
        area = loads.psize.to_numpy()[:n, ::-1]
        if nanson:
            td = loads.td.to_numpy()[:n]
            traction = (
                np.einsum("ni,nij->nj", traction, np.linalg.inv(td).transpose(0, 2, 1)) * np.linalg.det(td)[:, None]
            )
        return np.sum(2.0 * traction * area, axis=0)

    initial_force = total_force()
    for target_count in (4, 16):
        old_count = int(count[None])
        build_particle_traction_linked_list_2d(old_count, old_count, loads, head, next_id)
        split_refined_particles_with_traction_2d(
            ti.Vector([1.0, 1.0]),
            ti.Vector([1, 1]),
            old_count,
            old_count,
            axisymmetric,
            cells,
            particle,
            state,
            sizes,
            length,
            level,
            refined,
            loads,
            head,
            next_id,
            count,
            traction_count,
        )
        assert count[None] == traction_count[None] == target_count
        np.testing.assert_allclose(total_force(), initial_force, rtol=1e-7)
        assert len(np.unique(loads.pid.to_numpy()[:target_count])) == target_count


@pytest.mark.parametrize("dimension", DIMS)
@pytest.mark.parametrize(
    "option,value,match",
    [
        ("mls", True, "moving_least_square=False"),
        ("ptraction_method", "Virtual", "Stable or Nanson"),
        ("mode", "Lightweight", "mode='Normal'"),
        ("material_type", "TwoPhaseSingleLayer", "material_type='Solid'"),
    ],
)
def test_incompatible_adaptive_paths_fail_before_stepping(taichi_runtime, dimension, option, value, match):
    sims, element = make_element(dimension)
    setattr(sims, option, value)
    with pytest.raises(RuntimeError, match=match):
        element.validate_solver(sims)
