import numpy as np


def build_adaptive_leaf_mesh(refined_cell, coarse_cnum, fine_grid_size, max_level=None):
    coarse_cnum = np.asarray(coarse_cnum, dtype=np.int32)
    fine_grid_size = np.asarray(fine_grid_size, dtype=np.float64)
    refined_cell = np.asarray(refined_cell, dtype=np.uint8).reshape(-1)
    if max_level is None:
        max_level = max(1, int(np.max(refined_cell)) if refined_cell.size else 1)
    max_level = int(max_level)
    if max_level < 1:
        raise ValueError("Adaptive grid output requires max_level >= 1")
    dimension = coarse_cnum.size
    if dimension not in (2, 3):
        raise ValueError("Adaptive grid output only supports two or three dimensions")
    finest_ratio = 1 << max_level

    logical_nodes = []
    node_map = {}
    connectivity = []
    grid_level = []
    coarse_cell_id = []

    def compact_node_id(logical_index):
        logical_index = tuple(int(value) for value in logical_index)
        node_id = node_map.get(logical_index)
        if node_id is None:
            node_id = len(logical_nodes)
            node_map[logical_index] = node_id
            logical_nodes.append(logical_index)
        return node_id

    def append_cell(logical_corners, level, cell_id):
        connectivity.append([compact_node_id(index) for index in logical_corners])
        grid_level.append(level)
        coarse_cell_id.append(cell_id)

    if dimension == 2:
        coarse_nx, coarse_ny = coarse_cnum
        for j in range(coarse_ny):
            for i in range(coarse_nx):
                cell_id = i + j * coarse_nx
                level = min(int(refined_cell[cell_id]), max_level)
                divisions = 1 << level
                stride = finest_ratio // divisions
                origin = np.array([finest_ratio * i, finest_ratio * j], dtype=np.int32)
                for sub_j in range(divisions):
                    for sub_i in range(divisions):
                        x0, y0 = origin + stride * np.array([sub_i, sub_j], dtype=np.int32)
                        append_cell(
                            (
                                (x0, y0),
                                (x0 + stride, y0),
                                (x0 + stride, y0 + stride),
                                (x0, y0 + stride),
                            ),
                            level,
                            cell_id,
                        )
    else:
        coarse_nx, coarse_ny, coarse_nz = coarse_cnum
        for k in range(coarse_nz):
            for j in range(coarse_ny):
                for i in range(coarse_nx):
                    cell_id = i + j * coarse_nx + k * coarse_nx * coarse_ny
                    level = min(int(refined_cell[cell_id]), max_level)
                    divisions = 1 << level
                    stride = finest_ratio // divisions
                    origin = np.array(
                        [finest_ratio * i, finest_ratio * j, finest_ratio * k],
                        dtype=np.int32,
                    )
                    for sub_k in range(divisions):
                        for sub_j in range(divisions):
                            for sub_i in range(divisions):
                                x0, y0, z0 = origin + stride * np.array(
                                    [sub_i, sub_j, sub_k],
                                    dtype=np.int32,
                                )
                                append_cell(
                                    (
                                        (x0, y0, z0),
                                        (x0 + stride, y0, z0),
                                        (x0 + stride, y0 + stride, z0),
                                        (x0, y0 + stride, z0),
                                        (x0, y0, z0 + stride),
                                        (x0 + stride, y0, z0 + stride),
                                        (x0 + stride, y0 + stride, z0 + stride),
                                        (x0, y0 + stride, z0 + stride),
                                    ),
                                    level,
                                    cell_id,
                                )

    logical_nodes = np.asarray(logical_nodes, dtype=np.int32)
    fine_gnum = finest_ratio * coarse_cnum + 1
    logical_node_id = logical_nodes[:, 0] + logical_nodes[:, 1] * fine_gnum[0]
    if dimension == 3:
        logical_node_id += logical_nodes[:, 2] * fine_gnum[0] * fine_gnum[1]

    return {
        "coords": np.ascontiguousarray(logical_nodes * fine_grid_size),
        "logical_node_id": np.ascontiguousarray(logical_node_id, dtype=np.int64),
        "connectivity": np.ascontiguousarray(connectivity, dtype=np.int64),
        "grid_level": np.ascontiguousarray(grid_level, dtype=np.uint8),
        "coarse_cell_id": np.ascontiguousarray(coarse_cell_id, dtype=np.int64),
    }
