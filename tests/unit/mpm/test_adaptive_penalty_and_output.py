import numpy as np
import taichi as ti

from src.mpm.elements.AdaptiveGridOutput import build_adaptive_leaf_mesh
from src.mpm.elements.AdaptiveHexahedronKernel import (
    apply_hanging_penalty_impulse,
    apply_hanging_penalty_impulse_list,
    assemble_bridging_penalty_impulse,
    assemble_hanging_penalty_impulse_list,
    mark_refined_cells_epdstrain,
    mark_refined_cells_softening,
)
from src.mpm.elements.AdaptiveQuadrilateralKernel import (
    apply_hanging_penalty_impulse_2d,
    apply_hanging_penalty_impulse_list_2d,
    assemble_bridging_penalty_impulse_2d,
    assemble_hanging_penalty_impulse_list_2d,
    mark_refined_cells_epdstrain_2d,
    mark_refined_cells_softening_2d,
)
from src.utils.TypeDefination import vec2f, vec2i, vec3f, vec3i


@ti.dataclass
class Node2:
    m: float
    momentum: vec2f
    force: vec2f


@ti.dataclass
class Particle2:
    vol: float
    active: ti.u8
    materialID: int
    x: vec2f


@ti.dataclass
class State:
    epdstrain: float


@ti.dataclass
class Node3:
    m: float
    momentum: vec3f
    force: vec3f


@ti.dataclass
class Particle3:
    vol: float
    active: ti.u8
    materialID: int
    x: vec3f


@ti.kernel
def init2(node: ti.template(), particle: ti.template(), lnid: ti.template(),
          shape: ti.template(), alpha: ti.template(), coarse_size: ti.template(),
          body_id: ti.template(), node_size: ti.template(), dt: ti.template()):
    for i in range(8):
        node[i, 0].m = 1.
        node[i, 0].momentum = vec2f(1. if i < 2 else 0., 0.)
        node[i, 0].force = vec2f(0., 0.)
        lnid[i] = i
        shape[i] = 0.5 if i < 4 else 0.
    particle[0].vol = 0.01
    alpha[0] = 0.5
    coarse_size[0] = ti.cast(2, ti.u8)
    body_id[0] = ti.cast(0, ti.u8)
    node_size[0] = ti.cast(4, ti.u8)
    dt[None] = 1e-3


@ti.kernel
def init_hanging2(hanging_node_id: ti.template(),
                  hanging_master_id: ti.template(),
                  hanging_master_weight: ti.template(),
                  touched: ti.template()):
    hanging_node_id[0] = 0
    hanging_master_id[0] = ti.Vector([1, 2, 3, 0])
    hanging_master_weight[0] = ti.Vector([0.5, 0.5, 0., 0.])
    touched[0] = 0
    touched[1] = 1
    touched[2] = 2


@ti.kernel
def init_marker2(particle: ti.template(), state: ti.template()):
    particle[0].active = ti.u8(1)
    particle[0].materialID = 1
    particle[0].x = vec2f(0.25, 0.25)
    state[0].epdstrain = 1.0


@ti.kernel
def init3(node: ti.template(), particle: ti.template(), lnid: ti.template(),
          shape: ti.template(), alpha: ti.template(), coarse_size: ti.template(),
          body_id: ti.template(), node_size: ti.template()):
    for i in range(16):
        node[i, 0].m = 1.
        node[i, 0].momentum = vec3f(1. if i < 4 else 0., 0., 0.)
        node[i, 0].force = vec3f(0., 0., 0.)
        lnid[i] = i
        shape[i] = 0.25 if i < 8 else 0.
    particle[0].vol = 0.001
    alpha[0] = 0.5
    coarse_size[0] = ti.cast(4, ti.u8)
    body_id[0] = ti.cast(0, ti.u8)
    node_size[0] = ti.cast(8, ti.u8)


@ti.kernel
def init_hanging3(hanging_node_id: ti.template(),
                  hanging_master_id: ti.template(),
                  hanging_master_weight: ti.template(),
                  touched: ti.template()):
    hanging_node_id[0] = 0
    hanging_master_id[0] = ti.Vector([1, 2, 3, 4, 0, 0, 0, 0])
    hanging_master_weight[0] = ti.Vector([0.25, 0.25, 0.25, 0.25, 0., 0., 0., 0.])
    for i in range(5):
        touched[i] = i


@ti.kernel
def init_marker3(particle: ti.template(), state: ti.template()):
    particle[0].active = ti.u8(1)
    particle[0].materialID = 1
    particle[0].x = vec3f(0.25, 0.25, 0.25)
    state[0].epdstrain = 1.0


def test_adaptive_penalty_and_refinement_2d(taichi_runtime):
    node = Node2.field(shape=(8, 1))
    particle = Particle2.field(shape=1)
    penalty = ti.Vector.field(2, float, shape=(8, 1))
    bridge_alpha = ti.field(float, shape=1)
    bridge_coarse_size = ti.field(ti.u8, shape=1)
    bridge_body_id = ti.field(ti.u8, shape=1)
    node_size = ti.field(ti.u8, shape=1)
    lnid = ti.field(int, shape=8)
    shape = ti.field(float, shape=8)
    dt = ti.field(float, shape=())

    init2(node, particle, lnid, shape, bridge_alpha, bridge_coarse_size,
          bridge_body_id, node_size, dt)
    assemble_bridging_penalty_impulse_2d(
        1e-12, 1., 1, 10., 1e5, dt, 8, 1, particle, bridge_alpha,
        bridge_coarse_size, bridge_body_id, node_size, lnid, shape, node, penalty
    )
    apply_hanging_penalty_impulse_2d(1e-12, dt, node, penalty)

    hanging_node_id = ti.field(int, shape=1)
    hanging_master_id = ti.Vector.field(4, int, shape=1)
    hanging_master_weight = ti.Vector.field(4, float, shape=1)
    touched = ti.field(int, shape=3)
    init_hanging2(hanging_node_id, hanging_master_id, hanging_master_weight, touched)
    assemble_hanging_penalty_impulse_list_2d(
        1e-12, 1., 1, 10., 1e5, 0.01, dt, 1, hanging_node_id,
        hanging_master_id, hanging_master_weight, node, penalty
    )
    apply_hanging_penalty_impulse_list_2d(1e-12, dt, 3, touched, node, penalty)
    penalty_values = penalty.to_numpy()
    momentum_values = node.momentum.to_numpy()
    assert np.isfinite(penalty_values).all()
    assert np.linalg.norm(penalty_values) > 0.0
    assert np.isfinite(momentum_values).all()

    state = State.field(shape=1)
    current_refined = ti.field(ti.u8, shape=4)
    refined = ti.field(ti.u8, shape=4)
    init_marker2(particle, state)
    mark_refined_cells_epdstrain_2d(
        0.5, 2, vec2f(2., 2.), vec2i(2, 2), 1, particle, state,
        current_refined, refined,
    )
    assert refined.to_numpy()[0] == 1
    current_refined.from_numpy(refined.to_numpy())
    refined.fill(0)
    mark_refined_cells_softening_2d(
        0.5, 2, vec2f(2., 2.), vec2i(2, 2), 1, particle, state,
        current_refined, refined,
    )
    assert refined.to_numpy()[0] == 2


def test_adaptive_penalty_and_refinement_3d(taichi_runtime):
    node = Node3.field(shape=(16, 1))
    particle = Particle3.field(shape=1)
    penalty = ti.Vector.field(3, float, shape=(16, 1))
    bridge_alpha = ti.field(float, shape=1)
    bridge_coarse_size = ti.field(ti.u8, shape=1)
    bridge_body_id = ti.field(ti.u8, shape=1)
    node_size = ti.field(ti.u8, shape=1)
    lnid = ti.field(int, shape=16)
    shape = ti.field(float, shape=16)
    dt = ti.field(float, shape=())
    dt[None] = 1e-3

    init3(node, particle, lnid, shape, bridge_alpha, bridge_coarse_size,
          bridge_body_id, node_size)
    assemble_bridging_penalty_impulse(
        1e-12, 1., 1, 10., 1e5, dt, 16, 1, particle, bridge_alpha,
        bridge_coarse_size, bridge_body_id, node_size, lnid, shape, node, penalty
    )
    apply_hanging_penalty_impulse(1e-12, dt, node, penalty)

    hanging_node_id = ti.field(int, shape=1)
    hanging_master_id = ti.Vector.field(8, int, shape=1)
    hanging_master_weight = ti.Vector.field(8, float, shape=1)
    touched = ti.field(int, shape=5)
    init_hanging3(hanging_node_id, hanging_master_id, hanging_master_weight, touched)
    assemble_hanging_penalty_impulse_list(
        1e-12, 1., 1, 10., 1e5, 0.001, dt, 1, hanging_node_id,
        hanging_master_id, hanging_master_weight, node, penalty
    )
    apply_hanging_penalty_impulse_list(1e-12, dt, 5, touched, node, penalty)
    penalty_values = penalty.to_numpy()
    momentum_values = node.momentum.to_numpy()
    assert np.isfinite(penalty_values).all()
    assert np.linalg.norm(penalty_values) > 0.0
    assert np.isfinite(momentum_values).all()

    state = State.field(shape=1)
    current_refined = ti.field(ti.u8, shape=8)
    refined = ti.field(ti.u8, shape=8)
    init_marker3(particle, state)
    mark_refined_cells_epdstrain(
        0.5, 2, vec3f(2., 2., 2.), vec3i(2, 2, 2), 1, particle, state,
        current_refined, refined,
    )
    assert refined.to_numpy()[0] == 1
    current_refined.from_numpy(refined.to_numpy())
    refined.fill(0)
    mark_refined_cells_softening(
        0.5, 2, vec3f(2., 2., 2.), vec3i(2, 2, 2), 1, particle, state,
        current_refined, refined,
    )
    assert refined.to_numpy()[0] == 2


def test_adaptive_leaf_mesh_matches_refinement_counts():
    mesh2 = build_adaptive_leaf_mesh(
        refined_cell=np.asarray([0, 1, 2, 0], dtype=np.uint8),
        coarse_cnum=np.asarray([2, 2], dtype=np.int32),
        fine_grid_size=np.asarray([0.25, 0.25], dtype=np.float64),
        max_level=2,
    )
    assert mesh2["connectivity"].shape[0] == 1 + 4 + 16 + 1
    assert np.bincount(mesh2["grid_level"], minlength=3).tolist()[:3] == [2, 4, 16]
    assert np.allclose(mesh2["coords"].max(axis=0), [2.0, 2.0])

    mesh3 = build_adaptive_leaf_mesh(
        refined_cell=np.asarray([0, 1, 2, 0, 1, 0, 2, 0], dtype=np.uint8),
        coarse_cnum=np.asarray([2, 2, 2], dtype=np.int32),
        fine_grid_size=np.asarray([0.25, 0.25, 0.25], dtype=np.float64),
        max_level=2,
    )
    assert mesh3["connectivity"].shape[0] == 4 + 2 * 8 + 2 * 64
    assert np.bincount(mesh3["grid_level"], minlength=3).tolist()[:3] == [4, 16, 128]
    assert np.allclose(mesh3["coords"].max(axis=0), [2.0, 2.0, 2.0])
