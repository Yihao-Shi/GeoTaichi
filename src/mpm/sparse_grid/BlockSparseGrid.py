import math

import numpy as np
import taichi as ti
from taichi.lang.impl import current_cfg

from src.utils.PrefixSum import PrefixSumExecutor
from src.utils.TypeDefination import vec2i, vec3i
import src.utils.GlobalVariable as GlobalVariable


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


@ti.data_oriented
class BlockSparseGrid:
    def __init__(self, sims, element, grid_level) -> None:
        if current_cfg().arch not in (ti.cpu, ti.cuda):
            raise RuntimeError(
                "BlockScan sparse_grid requires the Taichi CPU or CUDA backend; "
                f"current backend is {current_cfg().arch}."
            )
        self.dimension = int(sims.dimension)
        self.grid_level = int(grid_level)
        self.block_size = int(sims.sparse_grid_block_size)
        self.block_volume = int(self.block_size**self.dimension)
        self.gnum_np = np.array([int(element.gnum[d]) for d in range(self.dimension)], dtype=np.int32)
        self.block_count_np = np.ceil(self.gnum_np / self.block_size).astype(np.int32)
        self.total_blocks = int(np.prod(self.block_count_np))
        self.dense_node_slots = int(element.gridSum)

        self.max_active_blocks = self._estimate_max_active_blocks(sims, element)
        self.node_capacity = int(self.max_active_blocks * self.block_volume)
        self.capacity_ratio = self.node_capacity / max(1, self.dense_node_slots)

        self.prefix_sum = PrefixSumExecutor(self.total_blocks)
        self.metadata_bytes = self._estimate_metadata_bytes()
        self.block_mask = ti.field(ti.i32, shape=self.prefix_sum.get_length())
        self.block_map = ti.field(ti.i32, shape=self.total_blocks)
        self.active_block_ids = ti.field(ti.i32, shape=self.max_active_blocks)
        self.active_block_count = ti.field(ti.i32, shape=())
        self.active_node_slots = ti.field(ti.i32, shape=())
        self.overflow = ti.field(ti.i32, shape=())

        if self.dimension == 2:
            self.block_count = vec2i([int(self.block_count_np[0]), int(self.block_count_np[1])])
        else:
            self.block_count = vec3i(
                [
                    int(self.block_count_np[0]),
                    int(self.block_count_np[1]),
                    int(self.block_count_np[2]),
                ]
            )
        self.reset_metadata()

    def _estimate_max_active_blocks(self, sims, element):
        if sims.sparse_grid_max_blocks > 0:
            requested = int(sims.sparse_grid_max_blocks)
            return max(1, min(self.total_blocks, requested))

        influence_nodes = max(1, int(round(element.grid_nodes ** (1.0 / self.dimension))))
        per_dim_block_bound = int(math.ceil((influence_nodes + self.block_size - 1) / self.block_size))
        support_block_bound = max(1, per_dim_block_bound**self.dimension)
        min_active_cells = int(math.ceil(sims.max_particle_num / max(1, self.block_volume)))
        estimated = int(math.ceil(min_active_cells * support_block_bound * float(sims.sparse_grid_capacity_factor)))
        return max(1, min(self.total_blocks, estimated))

    def reset_metadata(self):
        self.block_mask.fill(0)
        self.block_map.fill(-1)
        self.active_block_ids.fill(-1)
        self.active_block_count[None] = 0
        self.active_node_slots[None] = 0
        self.overflow[None] = 0

    def rebuild(self, particle_num, total_nodes, LnID, node_size, gnum):
        self.reset_metadata()
        if self.dimension == 2:
            self._mark_active_blocks_2d(int(particle_num), int(total_nodes), LnID, node_size, gnum)
        else:
            self._mark_active_blocks_3d(int(particle_num), int(total_nodes), LnID, node_size, gnum)
        self.prefix_sum.run(self.block_mask)
        if self.dimension == 2:
            self._build_block_map_2d(gnum)
        else:
            self._build_block_map_3d(gnum)
        self._finalize_counts()
        if int(self.overflow[None]) != 0:
            active = int(self.block_mask[self.total_blocks - 1])
            raise RuntimeError(
                "BlockSparseGrid capacity overflow: "
                f"active blocks={active}, capacity={self.max_active_blocks}. "
                "Increase sparse_grid/MaxActiveBlocks or sparse_grid/CapacityFactor."
            )
        if self.dimension == 2:
            self._remap_lnid_2d(int(particle_num), int(total_nodes), LnID, node_size, gnum)
        else:
            self._remap_lnid_3d(int(particle_num), int(total_nodes), LnID, node_size, gnum)

    def get_active_blocks(self):
        return int(self.active_block_count[None])

    def get_active_node_slots(self):
        return int(self.active_node_slots[None])

    def _estimate_metadata_bytes(self):
        prefix_workspace = 0
        if len(getattr(self.prefix_sum, "npad", [])) > 0:
            prefix_workspace = int(self.prefix_sum.npad[0])
        return 4 * (self.prefix_sum.get_length() + self.total_blocks + self.max_active_blocks + prefix_workspace + 3)

    def describe(self, node_slot_bytes=0):
        dense_slots = self.dense_node_slots * self.grid_level
        sparse_slots = self.node_capacity * self.grid_level
        active_slots = self.get_active_node_slots() * self.grid_level
        info = {
            "backend": "BlockScan",
            "block_size": self.block_size,
            "block_volume": self.block_volume,
            "block_count": self.block_count_np.tolist(),
            "total_blocks": self.total_blocks,
            "max_active_blocks": self.max_active_blocks,
            "node_capacity": self.node_capacity,
            "dense_node_slots": self.dense_node_slots,
            "allocated_node_slots": sparse_slots,
            "active_node_slots": active_slots,
            "capacity_ratio": self.capacity_ratio,
            "allocated_node_slot_ratio": sparse_slots / max(1, dense_slots),
            "active_node_slot_ratio": active_slots / max(1, dense_slots),
            "metadata_bytes": self.metadata_bytes,
        }
        if node_slot_bytes > 0:
            dense_bytes = dense_slots * node_slot_bytes
            allocated_bytes = sparse_slots * node_slot_bytes + self.metadata_bytes
            active_bytes = active_slots * node_slot_bytes + self.metadata_bytes
            info.update(
                {
                    "node_slot_bytes": node_slot_bytes,
                    "dense_node_bytes": dense_bytes,
                    "allocated_sparse_bytes": allocated_bytes,
                    "active_sparse_bytes": active_bytes,
                    "allocated_sparse_byte_ratio": allocated_bytes / max(1, dense_bytes),
                    "active_sparse_byte_ratio": active_bytes / max(1, dense_bytes),
                }
            )
        return info

    @ti.kernel
    def _mark_active_blocks_2d(
        self,
        particle_num: int,
        total_nodes: int,
        LnID: ti.template(),
        node_size: ti.template(),
        gnum: ti.types.vector(2, int),
    ):
        for np in range(particle_num):
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                node_id = LnID[ln]
                ix = node_id % gnum[0]
                iy = node_id // gnum[0]
                bx = ix // ti.static(self.block_size)
                by = iy // ti.static(self.block_size)
                block_id = bx + by * ti.static(int(self.block_count_np[0]))
                self.block_mask[block_id] = 1

    @ti.kernel
    def _mark_active_blocks_3d(
        self,
        particle_num: int,
        total_nodes: int,
        LnID: ti.template(),
        node_size: ti.template(),
        gnum: ti.types.vector(3, int),
    ):
        for np in range(particle_num):
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                node_id = LnID[ln]
                layer = gnum[0] * gnum[1]
                iz = node_id // layer
                local = node_id - iz * layer
                iy = local // gnum[0]
                ix = local - iy * gnum[0]
                bx = ix // ti.static(self.block_size)
                by = iy // ti.static(self.block_size)
                bz = iz // ti.static(self.block_size)
                block_id = (
                    bx
                    + by * ti.static(int(self.block_count_np[0]))
                    + bz * ti.static(int(self.block_count_np[0] * self.block_count_np[1]))
                )
                self.block_mask[block_id] = 1

    @ti.kernel
    def _build_block_map_2d(self, gnum: ti.types.vector(2, int)):
        for block_id in range(self.total_blocks):
            prefix = self.block_mask[block_id]
            prev = ti.cast(0, ti.i32)
            if block_id > 0:
                prev = self.block_mask[block_id - 1]
            if prefix > prev:
                compact_block = prefix - 1
                if compact_block < ti.static(self.max_active_blocks):
                    self.block_map[block_id] = compact_block
                    self.active_block_ids[compact_block] = block_id
                else:
                    self.overflow[None] = 1

    @ti.kernel
    def _build_block_map_3d(self, gnum: ti.types.vector(3, int)):
        for block_id in range(self.total_blocks):
            prefix = self.block_mask[block_id]
            prev = ti.cast(0, ti.i32)
            if block_id > 0:
                prev = self.block_mask[block_id - 1]
            if prefix > prev:
                compact_block = prefix - 1
                if compact_block < ti.static(self.max_active_blocks):
                    self.block_map[block_id] = compact_block
                    self.active_block_ids[compact_block] = block_id
                else:
                    self.overflow[None] = 1

    @ti.kernel
    def _finalize_counts(self):
        active_blocks = self.block_mask[ti.static(self.total_blocks - 1)]
        max_active_blocks = ti.cast(ti.static(self.max_active_blocks), ti.i32)
        if active_blocks > max_active_blocks:
            self.active_block_count[None] = max_active_blocks
            self.overflow[None] = 1
        else:
            self.active_block_count[None] = active_blocks
        self.active_node_slots[None] = self.active_block_count[None] * ti.static(self.block_volume)

    @ti.kernel
    def _remap_lnid_2d(
        self,
        particle_num: int,
        total_nodes: int,
        LnID: ti.template(),
        node_size: ti.template(),
        gnum: ti.types.vector(2, int),
    ):
        for np in range(particle_num):
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                LnID[ln] = compact_node_id(
                    LnID[ln],
                    gnum,
                    self.block_count,
                    ti.static(self.block_size),
                    ti.static(self.block_volume),
                    self.block_map,
                )

    @ti.kernel
    def _remap_lnid_3d(
        self,
        particle_num: int,
        total_nodes: int,
        LnID: ti.template(),
        node_size: ti.template(),
        gnum: ti.types.vector(3, int),
    ):
        for np in range(particle_num):
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                LnID[ln] = compact_node_id(
                    LnID[ln],
                    gnum,
                    self.block_count,
                    ti.static(self.block_size),
                    ti.static(self.block_volume),
                    self.block_map,
                )
