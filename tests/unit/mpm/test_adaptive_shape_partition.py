import numpy as np
import pytest
import taichi as ti

from src.mpm.elements.AdaptiveHexahedronKernel import (
    adaptive_global_update,
    load_hanging_constraint_table,
)
from src.mpm.elements.AdaptiveNodeMap import (
    AdaptiveNodeMap,
    build_hanging_constraint_table,
)
from src.utils.ShapeFunctions import GShapeLinear, ShapeLinear
from src.utils.TypeDefination import vec3f, vec3i


@ti.dataclass
class Particle:
    x: vec3f


@ti.kernel
def init_particles(particle: ti.template(), positions: ti.types.ndarray()):
    for np in range(particle.shape[0]):
        particle[np].x = vec3f(positions[np, 0], positions[np, 1], positions[np, 2])


@ti.kernel
def reduce_sums(
    particle_num: int,
    total_nodes: int,
    node_size: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    out: ti.types.ndarray(),
):
    for np in range(particle_num):
        shape_sum = 0.0
        grad_sum = vec3f(0.0, 0.0, 0.0)
        for local_id in range(int(node_size[np])):
            active_id = np * total_nodes + local_id
            shape_sum += shape_fn[active_id]
            grad_sum += dshape_fn[active_id]
        out[np, 0] = shape_sum
        out[np, 1] = grad_sum[0]
        out[np, 2] = grad_sum[1]
        out[np, 3] = grad_sum[2]
        out[np, 4] = float(node_size[np])


def run_level(max_level):
    coarse_cnum = np.array([7, 3, 3], dtype=np.int32)
    positions = np.ascontiguousarray(
        np.array(
            [
                [0.30, 0.20, 0.15],
                [1.10, 0.60, 0.55],
                [2.30, 0.80, 0.95],
                [0.75, 0.35, 0.25],
                [0.225, 0.125, 0.075],
                [0.275, 0.175, 0.125],
                [0.325, 0.225, 0.175],
                [0.375, 0.275, 0.225],
            ],
            dtype=np.float64,
        )
    )
    finest_ratio = 2**max_level
    fine_gnum = coarse_cnum * finest_ratio + 1
    refined = np.full(int(np.prod(coarse_cnum)), max_level, dtype=np.uint8)

    node_map = AdaptiveNodeMap(coarse_cnum, 0, 1.0, False, max_level)
    node_map.ensure_refined_cells(refined, "Linear")

    logical_to_compact = ti.field(int, shape=node_map.full_logical_node_count)
    compact_to_logical = ti.field(int, shape=node_map.capacity)
    logical_to_compact.from_numpy(node_map.logical_to_compact)
    compact_to_logical.from_numpy(node_map.compact_to_logical)

    slave_ids, master_ids, master_weights, touched_ids = build_hanging_constraint_table(
        refined,
        coarse_cnum,
        fine_gnum,
        node_map.logical_to_compact,
        node_map.compact_to_logical,
        node_map.next_compact_id,
        2,
        max_level,
    )
    hanging_lookup = ti.field(int, shape=node_map.capacity)
    hanging_lookup.fill(-1)
    hanging_node_id = ti.field(int, shape=node_map.capacity)
    hanging_master_id = ti.Vector.field(8, int, shape=node_map.capacity)
    hanging_master_weight = ti.Vector.field(8, float, shape=node_map.capacity)
    touched = ti.field(int, shape=node_map.capacity)
    if slave_ids.size:
        load_hanging_constraint_table(
            int(slave_ids.size),
            int(touched_ids.size),
            slave_ids,
            master_ids,
            master_weights,
            touched_ids,
            hanging_node_id,
            hanging_lookup,
            hanging_master_id,
            hanging_master_weight,
            touched,
        )

    particle_num = positions.shape[0]
    total_nodes = 64
    particle = Particle.field(shape=particle_num)
    init_particles(particle, positions)
    cal_length = ti.Vector.field(3, float, shape=particle_num)
    cal_length.fill(vec3f(0.0, 0.0, 0.0))
    refined_cell = ti.field(ti.u8, shape=refined.size)
    refined_cell.from_numpy(refined)
    particle_level = ti.field(ti.u8, shape=particle_num)
    support_overflow = ti.field(int, shape=())
    node_size = ti.field(ti.u8, shape=particle_num)
    lnid = ti.field(int, shape=particle_num * total_nodes)
    shape_fn = ti.field(float, shape=particle_num * total_nodes)
    dshape_fn = ti.Vector.field(3, float, shape=particle_num * total_nodes)

    adaptive_global_update(
        total_nodes,
        2,
        0,
        1,
        2,
        max_level,
        finest_ratio,
        vec3f(0.4, 0.4, 0.4),
        vec3f(2.5, 2.5, 2.5),
        vec3i(coarse_cnum),
        vec3f(0.4 / finest_ratio, 0.4 / finest_ratio, 0.4 / finest_ratio),
        vec3f(2.5 * finest_ratio, 2.5 * finest_ratio, 2.5 * finest_ratio),
        vec3i(fine_gnum),
        particle_num,
        particle,
        cal_length,
        refined_cell,
        particle_level,
        logical_to_compact,
        hanging_lookup,
        hanging_master_id,
        hanging_master_weight,
        support_overflow,
        node_size,
        lnid,
        shape_fn,
        dshape_fn,
        ShapeLinear,
        GShapeLinear,
    )
    out = np.zeros((particle_num, 5), dtype=np.float64)
    reduce_sums(particle_num, total_nodes, node_size, shape_fn, dshape_fn, out)
    return {
        "max_level": max_level,
        "capacity": int(node_map.capacity),
        "active_nodes": int(node_map.next_compact_id),
        "hanging_nodes": int(slave_ids.size),
        "overflow": int(support_overflow[None]),
        "shape_grad_node_size": out,
    }


@pytest.mark.parametrize("max_level", (1, 2))
def test_adaptive_shape_partition_of_unity(taichi_runtime, max_level):
    result = run_level(max_level)
    values = result["shape_grad_node_size"]
    split_values = values[4:]

    assert result["capacity"] >= result["active_nodes"] > 0
    assert result["hanging_nodes"] >= 0
    assert result["overflow"] == 0
    assert np.all(split_values[:, 4] > 0)
    assert np.allclose(split_values[:, 0], 1.0, rtol=0.0, atol=1.0e-6)
    assert np.allclose(split_values[:, 1:4], 0.0, rtol=0.0, atol=1.0e-6)
