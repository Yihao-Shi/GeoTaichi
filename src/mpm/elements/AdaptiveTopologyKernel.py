"""Device-resident node allocation and interface topology shared by 2D/3D MPM."""

from math import prod

import taichi as ti


@ti.func
def adaptive_list_count(count: ti.template()):
    value = 0
    if ti.static(hasattr(count, "snode")):
        value = count[None]
    else:
        value = count
    return value


@ti.data_oriented
class AdaptiveTopology:
    def __init__(self, coarse_cnum, max_level, capacity, mapping, compact, initial_count, bridging=False):
        self.dimension = len(coarse_cnum)
        self.corners = 1 << self.dimension
        self.coarse_cnum = ti.Vector([int(value) for value in coarse_cnum])
        self.coarse_gnum = self.coarse_cnum + 1
        self.max_level = max_level
        self.finest_ratio = 1 << max_level
        self.fine_gnum = self.finest_ratio * self.coarse_cnum + 1
        self.capacity = capacity
        self.mapping = mapping
        self.compact = compact
        self.node_count = ti.field(int, shape=())
        self.node_count[None] = initial_count
        self.request_count = ti.field(int, shape=())
        self.scan_block_size = 256
        self.scan_blocks = (self.mapping.shape[0] + self.scan_block_size - 1) // self.scan_block_size
        self.block_counts = ti.field(int, shape=self.scan_blocks)
        self.block_starts = ti.field(int, shape=self.scan_blocks)
        self.cell_levels = ti.field(ti.u8, shape=prod(int(value) for value in coarse_cnum))
        if bridging:
            coarse_nodes = prod(int(value) for value in self.coarse_gnum)
            self.bridge_refined = ti.field(int, shape=coarse_nodes)
            self.bridge_distance = ti.field(int, shape=coarse_nodes)
            self.bridge_buffer = ti.field(int, shape=coarse_nodes)
        else:
            self.error = ti.field(int, shape=())
            self.constrained = ti.field(int, shape=capacity)
            self.touched = ti.field(int, shape=capacity)
            self.raw_ids = ti.Vector.field(self.corners, int, shape=capacity)
            self.raw_weights = ti.Vector.field(self.corners, float, shape=capacity)
            self.expanded_ids = ti.Vector.field(self.corners, int, shape=capacity) if max_level > 1 else self.raw_ids
            self.expanded_weights = (
                ti.Vector.field(self.corners, float, shape=capacity) if max_level > 1 else self.raw_weights
            )

    @ti.func
    def index(self, linear_id, dimensions):
        coordinates = ti.Vector.zero(int, self.dimension)
        remainder = linear_id
        for axis in ti.static(range(self.dimension)):
            coordinates[axis] = remainder % dimensions[axis]
            remainder //= dimensions[axis]
        return coordinates

    @ti.func
    def linearize(self, coordinates, dimensions):
        linear_id, stride = 0, 1
        for axis in ti.static(range(self.dimension)):
            linear_id += coordinates[axis] * stride
            stride *= dimensions[axis]
        return linear_id

    @ti.func
    def corner(self, corner_id):
        offset = ti.Vector.zero(int, self.dimension)
        for axis in ti.static(range(self.dimension)):
            offset[axis] = (corner_id >> (self.dimension - axis - 1)) & 1
        return offset

    @ti.kernel
    def mark_nodes(self, refined_cell: ti.template(), halo: int):
        self.request_count[None] = 0
        for cell_id in range(refined_cell.shape[0]):
            previous = int(self.cell_levels[cell_id])
            current = ti.min(int(refined_cell[cell_id]), self.max_level)
            if current > previous:
                cell = self.index(cell_id, self.coarse_cnum)
                for level in range(previous + 1, current + 1):
                    divisions = 1 << level
                    stride = self.finest_ratio // divisions
                    lower = ti.max(divisions * cell - halo, 0)
                    upper = ti.min(divisions * cell + divisions + halo + 1, self.coarse_cnum * divisions + 1)
                    extent = upper - lower
                    node_count = 1
                    for axis in ti.static(range(self.dimension)):
                        node_count *= extent[axis]
                    for offset_id in range(node_count):
                        node = stride * (lower + self.index(offset_id, extent))
                        logical_id = self.linearize(node, self.fine_gnum)
                        if self.mapping[logical_id] < 0:
                            # -1 is unused, -2 is pending; reuse the map instead of a full-grid flag field.
                            previous_id = ti.atomic_min(self.mapping[logical_id], -2)
                            if previous_id == -1:
                                ti.atomic_add(self.request_count[None], 1)
                self.cell_levels[cell_id] = ti.cast(current, ti.u8)

    @ti.kernel
    def count_request_blocks(self):
        for block in range(self.scan_blocks):
            count = 0
            for logical_id in range(
                block * self.scan_block_size, ti.min((block + 1) * self.scan_block_size, self.mapping.shape[0])
            ):
                count += ti.cast(self.mapping[logical_id] == -2, int)
            self.block_counts[block] = count

    @ti.kernel
    def allocate_nodes(self) -> int:
        start = self.node_count[None]
        required = start + self.request_count[None]
        offset = start
        ti.loop_config(serialize=True)
        for block in range(self.scan_blocks):
            self.block_starts[block] = offset
            offset += self.block_counts[block]
        # Logical ordering preserves locality in all following particle/grid transfers.
        for block in range(self.scan_blocks):
            cursor = self.block_starts[block]
            for logical_id in range(
                block * self.scan_block_size, ti.min((block + 1) * self.scan_block_size, self.mapping.shape[0])
            ):
                if required <= self.capacity and self.mapping[logical_id] == -2:
                    self.mapping[logical_id] = cursor
                    self.compact[cursor] = logical_id
                    cursor += 1
        if required <= self.capacity:
            self.node_count[None] = required
        return required

    def ensure_refined_cells(self, refined_cell, shape_function):
        halo = int(shape_function in ("GIMP", "QuadBSpline", "CubicBSpline"))
        self.mark_nodes(refined_cell, halo)
        self.count_request_blocks()
        required = self.allocate_nodes()
        if required > self.capacity:
            raise RuntimeError(
                f"AdaptiveGrid dense node capacity {self.capacity} exceeded: {required} nodes are required. "
                "Increase AdaptiveGrid/RefineRatio."
            )

    @ti.kernel
    def build_raw_constraints(self, refined_cell: ti.template(), mapping: ti.template(), lookup: ti.template()):
        self.error[None] = 0
        for node_id in range(self.node_count[None]):
            node = self.index(self.compact[node_id], self.fine_gnum)
            base_cell = node // self.finest_ratio
            master_level = self.max_level
            for corner_id in ti.static(range(self.corners)):
                adjacent = base_cell
                offset = self.corner(corner_id)
                for axis in ti.static(range(self.dimension)):
                    if node[axis] % self.finest_ratio == 0:
                        adjacent[axis] += offset[axis] - 1
                if all(adjacent >= 0) and all(adjacent < self.coarse_cnum):
                    master_level = ti.min(master_level, int(refined_cell[self.linearize(adjacent, self.coarse_cnum)]))
            stride = self.finest_ratio // (1 << master_level)
            ids = ti.Vector([-1] * self.corners)
            weights = ti.Vector.zero(float, self.corners)
            constrained = not all(node % stride == 0)
            if constrained:
                master_cell = ti.min(node // stride, self.coarse_cnum * (1 << master_level) - 1)
                coordinate = (node - stride * master_cell).cast(float) / stride
                for corner_id in ti.static(range(self.corners)):
                    offset = self.corner(corner_id)
                    weight = 1.0
                    for axis in ti.static(range(self.dimension)):
                        weight *= coordinate[axis] if offset[axis] else 1.0 - coordinate[axis]
                    if weight > 0.0:
                        logical_id = self.linearize(stride * (master_cell + offset), self.fine_gnum)
                        master = mapping[logical_id]
                        if master >= 0:
                            ids[corner_id] = master
                            weights[corner_id] = weight
                        else:
                            ti.atomic_or(self.error[None], 1)
            else:
                ids[0] = node_id
                weights[0] = 1.0
            self.raw_ids[node_id] = ids
            self.raw_weights[node_id] = weights
            self.constrained[node_id] = ti.cast(constrained, int)
            self.touched[node_id] = 0
            lookup[node_id] = -1

    @ti.kernel
    def expand_constraints(
        self,
        source_ids: ti.template(),
        source_weights: ti.template(),
        target_ids: ti.template(),
        target_weights: ti.template(),
    ):
        for node_id in range(self.node_count[None]):
            ids = source_ids[node_id]
            weights = source_weights[node_id]
            if self.constrained[node_id] > 0:
                ids = ti.Vector([-1] * self.corners)
                weights = ti.Vector.zero(float, self.corners)
                for slot in range(self.corners):
                    parent = source_ids[node_id][slot]
                    parent_weight = source_weights[node_id][slot]
                    if parent_weight > 0.0:
                        for child_slot in range(self.corners):
                            master = source_ids[parent][child_slot]
                            weight = parent_weight * source_weights[parent][child_slot]
                            if weight > 0.0:
                                target = -1
                                for candidate in range(self.corners):
                                    if ids[candidate] == master:
                                        target = candidate
                                if target < 0:
                                    for candidate in range(self.corners):
                                        if target < 0 and ids[candidate] < 0:
                                            target = candidate
                                if target >= 0:
                                    ids[target] = master
                                    weights[target] += weight
                                else:
                                    ti.atomic_or(self.error[None], 2)
            target_ids[node_id] = ids
            target_weights[node_id] = weights

    @ti.kernel
    def pack_constraints(
        self,
        source_ids: ti.template(),
        source_weights: ti.template(),
        count: ti.template(),
        node_ids: ti.template(),
        lookup: ti.template(),
        master_ids: ti.template(),
        master_weights: ti.template(),
    ):
        count[None] = 0
        for node_id in range(self.node_count[None]):
            if self.constrained[node_id] > 0:
                row = ti.atomic_add(count[None], 1)
                node_ids[row] = node_id
                lookup[node_id] = row
                master_ids[row] = source_ids[node_id]
                master_weights[row] = source_weights[node_id]
                ti.atomic_or(self.touched[node_id], 1)
                for slot in range(self.corners):
                    if source_weights[node_id][slot] > 0.0:
                        master = source_ids[node_id][slot]
                        if self.constrained[master] > 0:
                            ti.atomic_or(self.error[None], 4)
                        ti.atomic_or(self.touched[master], 1)

    @ti.kernel
    def pack_touched(self, count: ti.template(), node_ids: ti.template()) -> int:
        count[None] = 0
        for node_id in range(self.node_count[None]):
            if self.touched[node_id] > 0:
                row = ti.atomic_add(count[None], 1)
                node_ids[row] = node_id
        return self.error[None]

    def build_constraints(
        self, refined_cell, mapping, count, node_ids, lookup, master_ids, master_weights, touched_count, touched_ids
    ):
        self.build_raw_constraints(refined_cell, mapping, lookup)
        source_ids, target_ids = self.raw_ids, self.expanded_ids
        source_weights, target_weights = self.raw_weights, self.expanded_weights
        # Each dependency moves to a strictly coarser level.
        for _ in range(self.max_level - 1):
            self.expand_constraints(source_ids, source_weights, target_ids, target_weights)
            source_ids, target_ids = target_ids, source_ids
            source_weights, target_weights = target_weights, source_weights
        self.pack_constraints(source_ids, source_weights, count, node_ids, lookup, master_ids, master_weights)
        error = self.pack_touched(touched_count, touched_ids)
        if error:
            raise RuntimeError(
                f"AdaptiveGrid hanging topology failed (error mask {error}): "
                "1=missing master node, 2=master capacity exceeded, 4=unresolved constraint chain"
            )

    @ti.kernel
    def mark_bridge_interface(self, refined_cell: ti.template(), unreachable: int):
        for node_id in range(self.bridge_distance.shape[0]):
            node = self.index(node_id, self.coarse_gnum)
            has_refined, has_coarse = False, False
            for corner_id in ti.static(range(self.corners)):
                cell = node + self.corner(corner_id) - 1
                if all(cell >= 0) and all(cell < self.coarse_cnum):
                    level = refined_cell[self.linearize(cell, self.coarse_cnum)]
                    has_refined = has_refined or level > 0
                    has_coarse = has_coarse or level == 0
            self.bridge_refined[node_id] = ti.cast(has_refined, int)
            self.bridge_distance[node_id] = 0 if has_refined and has_coarse else unreachable

    @ti.kernel
    def propagate_bridge_distance(self, source: ti.template(), target: ti.template()):
        for node_id in range(source.shape[0]):
            distance = source[node_id]
            if self.bridge_refined[node_id] > 0:
                node = self.index(node_id, self.coarse_gnum)
                for axis in ti.static(range(self.dimension)):
                    for direction in ti.static((-1, 1)):
                        neighbor = node
                        neighbor[axis] += direction
                        if 0 <= neighbor[axis] < self.coarse_gnum[axis]:
                            distance = ti.min(distance, source[self.linearize(neighbor, self.coarse_gnum)] + 1)
            target[node_id] = distance

    @ti.kernel
    def write_bridge_weights(self, distance: ti.template(), bridge_cells: int, weights: ti.template()):
        for node_id in range(weights.shape[0]):
            weight = 0.0
            if distance[node_id] <= bridge_cells:
                weight = 1.0 - distance[node_id] / float(bridge_cells)
            weights[node_id] = weight

    def build_bridge_weights(self, refined_cell, bridge_cells, weights):
        self.mark_bridge_interface(refined_cell, bridge_cells + 1)
        source, target = self.bridge_distance, self.bridge_buffer
        for _ in range(bridge_cells):
            self.propagate_bridge_distance(source, target)
            source, target = target, source
        self.write_bridge_weights(source, bridge_cells, weights)
