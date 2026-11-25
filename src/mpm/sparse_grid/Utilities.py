import taichi as ti
import src.utils.GlobalVariable as GlobalVariable

from src.utils.TypeDefinition import vec2i, vec3i


@ti.func
def compact_node_grid_coord(
    compact_node: int,
    block_count: ti.template(),
    block_size: ti.template(),
    block_volume: ti.template(),
    active_block_ids: ti.template(),
):
    coord = ti.Vector.zero(int, GlobalVariable.DIMENSION)
    if compact_node >= 0:
        compact_block = compact_node // block_volume
        local = compact_node - compact_block * block_volume
        physical_block = active_block_ids[compact_block]
        if physical_block >= 0:
            if ti.static(GlobalVariable.DIMENSION == 2):
                bx = physical_block % block_count[0]
                by = physical_block // block_count[0]
                lx = local % block_size
                ly = local // block_size
                coord = vec2i([bx * block_size + lx, by * block_size + ly])
            else:
                plane = block_count[0] * block_count[1]
                bz = physical_block // plane
                block_local = physical_block - bz * plane
                by = block_local // block_count[0]
                bx = block_local - by * block_count[0]
                lx = local % block_size
                ly = (local // block_size) % block_size
                lz = local // (block_size * block_size)
                coord = vec3i([bx * block_size + lx, by * block_size + ly, bz * block_size + lz])
    return coord


@ti.func
def compact_node_id(
    physical_node: int,
    gnum: ti.template(),
    block_count: ti.template(),
    block_size: ti.template(),
    block_volume: ti.template(),
    block_map: ti.template(),
):
    compact = -1
    if physical_node >= 0:
        if ti.static(GlobalVariable.DIMENSION == 2):
            ix = physical_node % gnum[0]
            iy = physical_node // gnum[0]
            bx = ix // block_size
            by = iy // block_size
            block_id = bx + by * block_count[0]
            compact_block = block_map[block_id]
            if compact_block >= 0:
                lx = ix - bx * block_size
                ly = iy - by * block_size
                compact = compact_block * block_volume + lx + ly * block_size
        else:
            layer = gnum[0] * gnum[1]
            iz = physical_node // layer
            local = physical_node - iz * layer
            iy = local // gnum[0]
            ix = local - iy * gnum[0]
            bx = ix // block_size
            by = iy // block_size
            bz = iz // block_size
            block_id = bx + by * block_count[0] + bz * block_count[0] * block_count[1]
            compact_block = block_map[block_id]
            if compact_block >= 0:
                lx = ix - bx * block_size
                ly = iy - by * block_size
                lz = iz - bz * block_size
                compact = compact_block * block_volume + lx + ly * block_size + lz * block_size * block_size
    return compact


@ti.func
def storage_node_id(physical_node: int, gnum, sparse_grid: ti.template()):
    node_id = physical_node
    if ti.static(sparse_grid != None):
        node_id = compact_node_id(
            physical_node,
            gnum,
            sparse_grid.block_count,
            sparse_grid.block_size,
            sparse_grid.block_volume,
            sparse_grid.block_map,
        )
    return node_id
