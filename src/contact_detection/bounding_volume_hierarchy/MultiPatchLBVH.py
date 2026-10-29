import taichi as ti
import numpy as np
import operator

from src.contact_detection.bounding_volume_hierarchy.AABB import AABB
from src.utils.BitFunction import morton3d32, morton3d60
from src.utils.TypeDefination import vec3f


def validate_bvh_rebuild_interval(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("bvh_rebuild_interval must be a positive integer")
    try:
        interval = operator.index(value)
    except TypeError as exc:
        raise ValueError("bvh_rebuild_interval must be a positive integer") from exc
    if interval <= 0:
        raise ValueError("bvh_rebuild_interval must be a positive integer")
    return interval


@ti.data_oriented
class LBVH(object):
    """
    Linear BVH is a simple BVH that is used to accelerate collision detection. It supports parallel building and
    querying of the BVH tree. Only supports axis-aligned bounding boxes (AABBs).

    Attributes
    -----
        aabbs : ti.field
        The input AABBs to be organized in the BVH, shape (n_batches, n_aabbs).
        n_aabbs : int
            Number of AABBs per batch.
        n_batches : int
            Number of batches.
        max_n_query_results : int
            Maximum number of query results allowed.
        max_stack_depth : int
            Maximum stack depth for BVH traversal.
        aabb_centers : ti.field
            Centers of the AABBs, shape (n_batches, n_aabbs).
        aabb_min : ti.field
            Minimum coordinates of AABB centers per batch, shape (n_batches).
        aabb_max : ti.field
            Maximum coordinates of AABB centers per batch, shape (n_batches).
        scale : ti.field
            Scaling factors for normalizing AABB centers, shape (n_batches).
        morton_codes : ti.field
            Morton codes for each AABB, shape (n_batches, n_aabbs).
        hist : ti.field
            Histogram for radix sort, shape (n_batches, 256).
        prefix_sum : ti.field
            Prefix sum for histogram, shape (n_batches, 256).
        offset : ti.field
            Offset for radix sort, shape (n_batches, n_aabbs).
        tmp_morton_codes : ti.field
            Temporary storage for radix sort, shape (n_batches, n_aabbs).
        Node : ti.dataclass
            Node structure for the BVH tree, containing left, right, parent indices and bounding box.
        nodes : ti.field
            BVH nodes, shape (n_batches, n_aabbs * 2 - 1).
        internal_node_depth : ti.field
            Internal-node depths for synchronized bottom-up bounds construction.
        query_result : ti.field
            Query results as a vector of (batch id, self id, query id), shape (max_n_query_results).
        query_result_count : ti.field
            Counter for the number of query results.

    Notes
    ------
        For algorithmic details, see:
        https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf
    """

    def __init__(self, n_aabbs=0, aabb: AABB = None, extended_morton=False):
        if aabb is None:
            aabb = AABB(n_aabbs)
        self.aabb = aabb
        self.extended_morton = bool(extended_morton)

        # Maximum stack depth for traversal
        self.max_stack_depth = 64
        self.aabb_min = ti.field(self.aabb.vector_type, shape=(self.aabb.n_batches))
        self.aabb_max = ti.field(self.aabb.vector_type, shape=(self.aabb.n_batches))
        self.scale = ti.field(self.aabb.vector_type, shape=(self.aabb.n_batches))
        self.morton_codes = ti.field(ti.u64, shape=(self.aabb.n_aabbs))
        self.morton_object_bits = ti.field(ti.i32, shape=(self.aabb.n_batches))

        # Histogram for radix sort
        max_bits = 64
        radix = 8
        self.num_buckets = 1 << radix
        self.num_iters = int(max_bits / radix)
        self.hist = ti.field(ti.u32, shape=self.aabb.n_batches * self.num_buckets)
        # Prefix sum for histogram
        self.prefix_sum = ti.field(ti.u32, shape=self.aabb.n_batches * self.num_buckets)
        # Offset for radix sort
        self.offset = ti.field(ti.u32, shape=(self.aabb.n_aabbs))
        self.sort_chunk_size = 256
        chunk_counts = [(size + self.sort_chunk_size - 1) // self.sort_chunk_size for size in self.aabb.batch_size]
        max_chunks = sum(chunk_counts)
        self.sort_chunk_prefix = ti.field(ti.i32, shape=self.aabb.n_batches + 1)
        self.sort_chunk_batch = ti.field(ti.i32, shape=max_chunks)
        self.sort_chunk_hist = ti.field(ti.u32, shape=(max_chunks, self.num_buckets))
        # Temporary storage for radix sort
        self.tmp_morton_codes = ti.field(ti.u64, shape=(self.aabb.n_aabbs))
        self.primitive_ids = ti.field(ti.i32, shape=(self.aabb.n_aabbs))
        self.tmp_primitive_ids = ti.field(ti.i32, shape=(self.aabb.n_aabbs))

        @ti.dataclass
        class Node:
            """
            Node structure for the BVH tree.

            Attributes:
                left (int): Index of the left child node.
                right (int): Index of the right child node.
                parent (int): Index of the parent node.
                bound (ti_aabb): Bounding box of the node, represented as an AABB.
            """

            left: ti.i32
            right: ti.i32
            parent: ti.i32
            bound: aabb.ti_aabb

        self.Node = Node

        # Nodes of the BVH, first n_aabbs - 1 are internal nodes, last n_aabbs are leaf nodes
        self.nodes = self.Node.field(shape=(sum([batch_size * 2 - 1 for batch_size in self.aabb.batch_size])))
        self.internal_node_depth = ti.field(
            ti.i32, shape=max(sum(batch_size - 1 for batch_size in self.aabb.batch_size), 1)
        )
        # Auxiliary list
        self.prefix_batch_size = ti.field(int, shape=self.aabb.n_batches + 1)
        self.leaf2batch = ti.field(ti.i32, shape=self.aabb.n_aabbs)
        self.internal2batch = ti.field(ti.i32, shape=max(self.aabb.n_aabbs - self.aabb.n_batches, 1))

    def initialize(self, domain=None, active_aabbs=None):
        self._last_rebuild_step = None
        self.aabb.reset(active_aabbs)
        object_bits = []
        chunk_prefix = [0]
        for batch_size in self.aabb.current_batch_size:
            bits = 0
            while (1 << bits) < max(int(batch_size), 1):
                bits += 1
            object_bits.append(bits)
            chunk_prefix.append(chunk_prefix[-1] + (batch_size + self.sort_chunk_size - 1) // self.sort_chunk_size)
        self.sort_chunk_prefix.from_numpy(np.asarray(chunk_prefix, dtype=np.int32))
        self.morton_object_bits.from_numpy(np.asarray(object_bits, dtype=np.int32))
        self.prefix_batch_size.from_numpy(
            np.pad(
                np.asarray(self.aabb.prefix_current_batch_size).astype(np.int32),
                (0, self.prefix_batch_size.shape[0] - len(self.aabb.prefix_current_batch_size)),
                mode="constant",
                constant_values=self.aabb.prefix_current_batch_size[-1],
            )
        )
        self._initialize_batch_maps()
        if domain is None:
            self.adaptive_simulation_domain(self.aabb.n_batches, self.aabb.current_n_aabbs)
        else:
            self.initial_simulation_domain(self.aabb.n_batches, domain)

    @ti.kernel
    def _initialize_batch_maps(self):
        for batch in range(self.aabb.n_batches):
            begin, end = self.prefix_batch_size[batch], self.prefix_batch_size[batch + 1]
            for leaf in range(begin, end):
                self.leaf2batch[leaf] = batch
            for internal in range(begin - batch, end - batch - 1):
                self.internal2batch[internal] = batch
            for chunk in range(self.sort_chunk_prefix[batch], self.sort_chunk_prefix[batch + 1]):
                self.sort_chunk_batch[chunk] = batch

    @ti.func
    def _escape_index(self, root, node, stop):
        cursor, escape = node - root, stop
        parent = self.nodes[node].parent
        while parent >= 0:
            if self.nodes[root + parent].left == cursor:
                escape = root + self.nodes[root + parent].right
                break
            cursor = parent
            parent = self.nodes[root + cursor].parent
        return escape

    @ti.kernel
    def export_escape_links(self, escape: ti.template()):
        """Export threaded traversal links once for a fixed device topology."""
        for leaf in range(self.aabb.current_n_aabbs):
            batch = self.leaf2batch[leaf]
            begin, end = self.prefix_batch_size[batch], self.prefix_batch_size[batch + 1]
            root, stop = 2 * begin - batch, 2 * end - batch - 1
            node = root + end - begin - 1 + leaf - begin
            escape[node] = self._escape_index(root, node, stop)
        for internal in range(self.aabb.current_n_aabbs - self.aabb.n_batches):
            batch = self.internal2batch[internal]
            begin, end = self.prefix_batch_size[batch], self.prefix_batch_size[batch + 1]
            root, stop = 2 * begin - batch, 2 * end - batch - 1
            node = root + internal - begin + batch
            escape[node] = self._escape_index(root, node, stop)

    @ti.kernel
    def export_stackless(self, nodes: ti.template(), prefix: ti.template()):
        """Export child/escape/primitive indices without a host topology copy."""
        for batch in range(self.aabb.n_batches):
            begin, end = self.prefix_batch_size[batch], self.prefix_batch_size[batch + 1]
            root, stop = 2 * begin - batch, 2 * end - batch - 1
            prefix[batch] = root
            for node in range(root, stop):
                left, right = self.nodes[node].left, self.nodes[node].right
                primitive = -1
                if left < 0:
                    primitive = begin + self.primitive_index(begin + node - root - (end - begin - 1))
                escape = self._escape_index(root, node, stop)
                nodes[node] = ti.Vector(
                    [
                        ti.select(left >= 0, root + left, -1),
                        ti.select(right >= 0, root + right, -1),
                        escape,
                        primitive,
                    ]
                )
        prefix[self.aabb.n_batches] = 2 * self.aabb.current_n_aabbs - self.aabb.n_batches

    @ti.kernel
    def initial_simulation_domain(self, n_batches: int, domain: ti.types.vector(3, float)):
        for i_b in ti.ndrange(n_batches):
            self.aabb_min[i_b] = [0, 0, 0]
            self.aabb_max[i_b] = domain

        for i_b in ti.ndrange(n_batches):
            scale = self.aabb_max[i_b] - self.aabb_min[i_b]
            for i in ti.static(range(3)):
                self.scale[i_b][i] = ti.select(scale[i] > 1e-7, 1.0 / scale[i], 1)

    @ti.kernel
    def adaptive_simulation_domain(self, n_batches: int, currnet_n_aabb: int):
        for batch_id in ti.ndrange(n_batches):
            prefix_batch_num = self.prefix_batch_size[batch_id]
            self.aabb_min[batch_id] = self.aabb.get_center(0, prefix_batch_num)
            self.aabb_max[batch_id] = self.aabb.get_center(0, prefix_batch_num)

        for n_aabb in range(currnet_n_aabb):
            batch_id = int(self.leaf2batch[n_aabb])
            ti.atomic_min(self.aabb_min[batch_id], self.aabb.aabbs[n_aabb].min)
            ti.atomic_max(self.aabb_max[batch_id], self.aabb.aabbs[n_aabb].max)

        for batch_id in ti.ndrange(n_batches):
            scale = self.aabb_max[batch_id] - self.aabb_min[batch_id]
            for i in ti.static(range(3)):
                self.scale[batch_id][i] = ti.select(scale[i] > 1e-7, 1.0 / scale[i], 1)

    def update(self, current_step, rebuild_interval, *, n_batches=None):
        """Refresh an explicit search; missed query steps do not skip a due rebuild."""
        step = int(current_step)
        if self._last_rebuild_step is None:
            self.build()
            self._last_rebuild_step = step
            return
        batches = self.aabb.n_batches if n_batches is None else n_batches
        count = self.aabb.prefix_current_batch_size[batches]
        if step < self._last_rebuild_step or step - self._last_rebuild_step >= rebuild_interval:
            self.build_prefix(batches, count)
            self._last_rebuild_step = step
        else:
            self.refit_prefix(batches, count)

    def build(self, currnet_n_aabb=None):
        """
        Build the BVH from the axis-aligned bounding boxes (AABBs).
        """
        if currnet_n_aabb is None:
            currnet_n_aabb = self.aabb.current_n_aabbs
        self.build_prefix(self.aabb.n_batches, currnet_n_aabb)

    def build_prefix(self, n_batches, currnet_n_aabb):
        self.adaptive_simulation_domain(n_batches, currnet_n_aabb)
        self.compute_morton_codes(currnet_n_aabb)
        self.radix_sort_morton_codes(n_batches, currnet_n_aabb)
        self.build_radix_tree(n_batches, currnet_n_aabb)
        self.compute_bounds(n_batches, currnet_n_aabb)

    def refit(self, currnet_n_aabb=None):
        if currnet_n_aabb is None:
            currnet_n_aabb = self.aabb.current_n_aabbs
        self.refit_prefix(self.aabb.n_batches, currnet_n_aabb)

    def refit_prefix(self, n_batches, currnet_n_aabb):
        self.compute_bounds(n_batches, currnet_n_aabb)

    @ti.kernel
    def compute_morton_codes(self, currnet_n_aabb: int):
        """
        Compute the Morton codes for each AABB.

        The first 32 bits is the Morton code for the x, y, z coordinates, and the last 32 bits is the index of the AABB
        in the original array. The x, y, z coordinates are scaled to a 10-bit integer range [0, 1024) and interleaved to
        form the Morton code.
        """
        for n_aabb in range(currnet_n_aabb):
            batch_id = int(self.leaf2batch[n_aabb])
            prefix_batch_num = self.prefix_batch_size[batch_id]
            center = self.aabb.get_center(n_aabb) - self.aabb_min[batch_id]
            scaled_center = center * self.scale[batch_id]
            local_id = n_aabb - prefix_batch_num
            if ti.static(self.extended_morton):
                morton_code = morton3d60(*scaled_center)
                object_bits = self.morton_object_bits[batch_id]
                spatial_code = (morton_code >> ti.u64(object_bits)) << ti.u64(object_bits)
                tag = ti.u64(batch_id & 0x3) << ti.u64(61)
                self.morton_codes[n_aabb] = spatial_code | ti.u64(local_id) | tag
                self.primitive_ids[n_aabb] = local_id
            else:
                morton_code = morton3d32(*scaled_center)
                self.morton_codes[n_aabb] = (ti.u64(morton_code) << 32) | ti.u64(local_id)
                self.primitive_ids[n_aabb] = local_id

    @ti.kernel
    def radix_sort_morton_codes(self, n_batches: int, currnet_n_aabb: int):
        """
        Stable batched radix sort with parallel, bounded-size chunk histograms.
        """
        for i in ti.static(range(self.num_iters)):
            for chunk, bucket in ti.ndrange(self.sort_chunk_prefix[n_batches], self.num_buckets):
                self.sort_chunk_hist[chunk, bucket] = 0

            # Each thread preserves input order within one chunk; chunk prefixes
            # preserve order between chunks without serializing the whole batch.
            for chunk in range(self.sort_chunk_prefix[n_batches]):
                batch_id = self.sort_chunk_batch[chunk]
                begin = (
                    self.prefix_batch_size[batch_id] + (chunk - self.sort_chunk_prefix[batch_id]) * self.sort_chunk_size
                )
                end = ti.min(begin + self.sort_chunk_size, self.prefix_batch_size[batch_id + 1], currnet_n_aabb)
                for n_aabb in range(begin, end):
                    code = ti.i32((self.morton_codes[n_aabb] >> (i * 8)) & 0xFF)
                    self.offset[n_aabb] = self.sort_chunk_hist[chunk, code]
                    self.sort_chunk_hist[chunk, code] += 1

            for batch_id, bucket in ti.ndrange(n_batches, self.num_buckets):
                total = ti.u32(0)
                for chunk in range(self.sort_chunk_prefix[batch_id], self.sort_chunk_prefix[batch_id + 1]):
                    count = self.sort_chunk_hist[chunk, bucket]
                    self.sort_chunk_hist[chunk, bucket] = total
                    total += count
                self.hist[batch_id * self.num_buckets + bucket] = total

            # Compute prefix sum
            for batch_id in range(n_batches):
                self.prefix_sum[batch_id * self.num_buckets + 0] = 0
                for j in range(1, self.num_buckets):  # sequential prefix sum
                    self.prefix_sum[batch_id * self.num_buckets + j] = (
                        self.prefix_sum[batch_id * self.num_buckets + j - 1]
                        + self.hist[batch_id * self.num_buckets + j - 1]
                    )

            # Reorder morton codes
            for n_aabb in range(currnet_n_aabb):
                batch_id = int(self.leaf2batch[n_aabb])
                prefix_batch_num = self.prefix_batch_size[batch_id]
                code = (self.morton_codes[n_aabb] >> (i * 8)) & 0xFF
                chunk = self.sort_chunk_prefix[batch_id] + (n_aabb - prefix_batch_num) // self.sort_chunk_size
                idx = ti.i32(
                    self.offset[n_aabb]
                    + self.sort_chunk_hist[chunk, ti.i32(code)]
                    + self.prefix_sum[batch_id * self.num_buckets + ti.i32(code)]
                )
                self.tmp_morton_codes[prefix_batch_num + idx] = self.morton_codes[n_aabb]
                self.tmp_primitive_ids[prefix_batch_num + idx] = self.primitive_ids[n_aabb]

            # Swap the temporary and original morton codes
            for n_aabb in range(currnet_n_aabb):
                self.morton_codes[n_aabb] = self.tmp_morton_codes[n_aabb]
                self.primitive_ids[n_aabb] = self.tmp_primitive_ids[n_aabb]

    @ti.kernel
    def build_radix_tree(self, n_batches: int, currnet_n_aabb: int):
        """
        Build the radix tree from the sorted morton codes.

        The tree is built in parallel for every internal node.
        """
        # Initialize the first node
        for batch_id in range(n_batches):
            prefix_batch_num = self.prefix_batch_size[batch_id]
            self.nodes[2 * prefix_batch_num - batch_id].parent = -1

        # Initialize the leaf nodes
        for n_aabb in range(currnet_n_aabb):
            batch_id = int(self.leaf2batch[n_aabb])
            prefix_batch_num = self.prefix_batch_size[batch_id]
            batch_size = self.prefix_batch_size[batch_id + 1] - prefix_batch_num
            i_aabb = n_aabb - prefix_batch_num
            i_leaf = 2 * prefix_batch_num - batch_id + i_aabb + batch_size - 1
            self.nodes[i_leaf].left = -1
            self.nodes[i_leaf].right = -1

        # Parallel build for every internal node
        for n_internal in range(currnet_n_aabb - n_batches):
            batch_id = int(self.internal2batch[n_internal])
            prefix_batch_num = self.prefix_batch_size[batch_id]
            batch_size = self.prefix_batch_size[batch_id + 1] - prefix_batch_num
            prefix_i_node = 2 * prefix_batch_num - batch_id
            i_internal = n_internal - prefix_batch_num + batch_id
            d = ti.select(
                self.delta(i_internal, i_internal + 1, prefix_batch_num, batch_size)
                > self.delta(i_internal, i_internal - 1, prefix_batch_num, batch_size),
                1,
                -1,
            )

            delta_min = self.delta(i_internal, i_internal - d, prefix_batch_num, batch_size)
            l_max = ti.i32(2)
            while self.delta(i_internal, i_internal + l_max * d, prefix_batch_num, batch_size) > delta_min:
                l_max *= 2
            l = ti.i32(0)

            t = l_max // 2
            while t > 0:
                if self.delta(i_internal, i_internal + (l + t) * d, prefix_batch_num, batch_size) > delta_min:
                    l += t
                t //= 2
            j = i_internal + l * d
            delta_node = self.delta(i_internal, j, prefix_batch_num, batch_size)
            s = ti.i32(0)
            t = (l + 1) // 2
            while t > 0:
                if self.delta(i_internal, i_internal + (s + t) * d, prefix_batch_num, batch_size) > delta_node:
                    s += t
                t = ti.select(t > 1, (t + 1) // 2, 0)

            gamma = i_internal + ti.i32(s) * d + ti.min(d, 0)
            left = ti.select(ti.min(i_internal, j) == gamma, gamma + batch_size - 1, gamma)
            right = ti.select(ti.max(i_internal, j) == gamma + 1, gamma + batch_size, gamma + 1)
            self.nodes[prefix_i_node + i_internal].left = ti.i32(left)
            self.nodes[prefix_i_node + i_internal].right = ti.i32(right)
            self.nodes[prefix_i_node + ti.i32(left)].parent = i_internal
            self.nodes[prefix_i_node + ti.i32(right)].parent = i_internal

    @ti.func
    def delta(self, i, j, prefix_batch_num, currnet_n_aabb):
        """
        Compute the longest common prefix (LCP) of the morton codes of two AABBs.
        """
        result = -1
        if j >= 0 and j < currnet_n_aabb:
            result = 64
            x = self.morton_codes[prefix_batch_num + ti.i32(i)] ^ self.morton_codes[prefix_batch_num + ti.i32(j)]
            for b in range(64):
                if x & (ti.u64(1) << (63 - b)):
                    result = b
                    break
            if ti.static(self.extended_morton):
                if x == ti.u64(0):
                    tie = ti.u64(
                        self.primitive_ids[prefix_batch_num + ti.i32(i)]
                        ^ self.primitive_ids[prefix_batch_num + ti.i32(j)]
                    )
                    result = 128
                    for b in range(64):
                        if tie & (ti.u64(1) << (63 - b)):
                            result = 64 + b
                            break
        return result

    @ti.func
    def primitive_index(self, morton_index):
        result = ti.i32(self.morton_codes[morton_index] & ti.u64(0xFFFFFFFF))
        if ti.static(self.extended_morton):
            result = self.primitive_ids[morton_index]
        return result

    def compute_bounds(self, n_batches: int, currnet_n_aabb: int):
        maximum_depth = self.initialize_leaf_bounds(n_batches, currnet_n_aabb)
        self.maximum_depth = int(maximum_depth)
        if maximum_depth + 2 > self.max_stack_depth:
            raise RuntimeError("LBVH depth exceeds traversal capacity or contains a parent cycle")
        # Kernel boundaries publish every child's bounds before its parent's layer reads them.
        for depth in range(maximum_depth, -1, -1):
            self.compute_bounds_at_depth(n_batches, currnet_n_aabb, depth)

    @ti.kernel
    def initialize_leaf_bounds(self, n_batches: int, currnet_n_aabb: int) -> ti.i32:
        for n_aabb in range(currnet_n_aabb):
            batch_id = int(self.leaf2batch[n_aabb])
            prefix_batch_num = self.prefix_batch_size[batch_id]
            i_aabb = n_aabb - prefix_batch_num
            batch_size = self.prefix_batch_size[batch_id + 1] - prefix_batch_num
            prefix_i_node = 2 * prefix_batch_num - batch_id
            i_leaf = prefix_i_node + i_aabb + batch_size - 1
            idx = self.primitive_index(n_aabb)
            self.nodes[i_leaf].bound.min = self.aabb.aabbs[prefix_batch_num + idx].min
            self.nodes[i_leaf].bound.max = self.aabb.aabbs[prefix_batch_num + idx].max

        maximum_depth = 0
        for n_internal in range(currnet_n_aabb - n_batches):
            batch_id = int(self.internal2batch[n_internal])
            prefix_batch_num = self.prefix_batch_size[batch_id]
            prefix_i_node = 2 * prefix_batch_num - batch_id
            local_index = n_internal - prefix_batch_num + batch_id
            parent = self.nodes[prefix_i_node + local_index].parent
            depth = 0
            while parent != -1 and depth < ti.static(self.max_stack_depth):
                depth += 1
                parent = self.nodes[prefix_i_node + parent].parent
            self.internal_node_depth[n_internal] = depth
            ti.atomic_max(maximum_depth, depth)
        return maximum_depth

    @ti.kernel
    def compute_bounds_at_depth(self, n_batches: int, currnet_n_aabb: int, depth: ti.i32):
        for n_internal in range(currnet_n_aabb - n_batches):
            if self.internal_node_depth[n_internal] == depth:
                batch_id = int(self.internal2batch[n_internal])
                prefix_batch_num = self.prefix_batch_size[batch_id]
                prefix_i_node = 2 * prefix_batch_num - batch_id
                node_index = prefix_i_node + n_internal - prefix_batch_num + batch_id
                left = self.nodes[prefix_i_node + self.nodes[node_index].left].bound
                right = self.nodes[prefix_i_node + self.nodes[node_index].right].bound
                self.nodes[node_index].bound.min = ti.min(left.min, right.min)
                self.nodes[node_index].bound.max = ti.max(left.max, right.max)

    @ti.kernel
    def query(
        self,
        primitiveNum: int,
        potential_primitive_num: int,
        batch_id: int,
        aabbs: ti.template(),
        potential_list_object_object: ti.template(),
        object_object: ti.template(),
    ):
        """
        Query the BVH for intersections with the given AABBs.

        The results are stored in the query_result field.
        """
        prefix_batch_num = self.prefix_batch_size[batch_id]
        batch_num = self.prefix_batch_size[batch_id + 1] - prefix_batch_num
        prefix_i_node = 2 * prefix_batch_num - batch_id
        for master in range(primitiveNum):
            query_stack = ti.Vector.zero(ti.i32, 64)
            stack_depth = 1

            sques = master * potential_primitive_num
            while stack_depth > 0:
                stack_depth -= 1
                node_idx = query_stack[stack_depth]
                node = self.nodes[prefix_i_node + node_idx]
                # Check if the AABB intersects with the node's bounding box
                if aabbs[master].intersects(node.bound, 0.0, 0.0):
                    # If it's a leaf node, add the AABB index to the query results
                    if node.left == -1 and node.right == -1:
                        slave = self.primitive_index(prefix_batch_num + node_idx - (batch_num - 1))
                        potential_list_object_object[sques] = slave
                        sques += 1
                    else:
                        # Push children onto the stack
                        if node.right != -1:
                            query_stack[stack_depth] = node.right
                            stack_depth += 1
                        if node.left != -1:
                            query_stack[stack_depth] = node.left
                            stack_depth += 1
            neighbors = sques - master * potential_primitive_num
            assert (
                neighbors <= potential_primitive_num
            ), f"Keyword:: /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
            object_object[master + 1] = neighbors

    @ti.kernel
    def self_query(
        self,
        primitiveNum: int,
        potential_primitive_num: int,
        master_batch: int,
        slave_batch: int,
        aabbs: ti.template(),
        potential_list_object_object: ti.template(),
        object_object: ti.template(),
    ):
        """
        Query the BVH for intersections with the given AABBs.

        The results are stored in the query_result field.
        """
        prefix_master_batch_num = self.prefix_batch_size[master_batch]
        prefix_slave_batch_num = self.prefix_batch_size[slave_batch]
        slave_batch_num = self.prefix_batch_size[slave_batch + 1] - prefix_slave_batch_num
        prefix_slave_i_node = 2 * prefix_slave_batch_num - slave_batch
        for master in range(primitiveNum):
            query_stack = ti.Vector.zero(ti.i32, 64)
            stack_depth = 1

            sques = master * potential_primitive_num
            while stack_depth > 0:
                stack_depth -= 1
                node_idx = query_stack[stack_depth]
                node = self.nodes[prefix_slave_i_node + node_idx]
                # Check if the AABB intersects with the node's bounding box
                if aabbs[prefix_master_batch_num + master].intersects(node.bound, 0.0, 0.0):
                    # If it's a leaf node, add the AABB index to the query results
                    if node.left == -1 and node.right == -1:
                        slave = self.primitive_index(prefix_slave_batch_num + node_idx - (slave_batch_num - 1))
                        potential_list_object_object[sques] = slave
                        sques += 1
                    else:
                        # Push children onto the stack
                        if node.right != -1:
                            query_stack[stack_depth] = node.right
                            stack_depth += 1
                        if node.left != -1:
                            query_stack[stack_depth] = node.left
                            stack_depth += 1
            neighbors = sques - master * potential_primitive_num
            assert (
                neighbors <= potential_primitive_num
            ), f"Keyword:: /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
            object_object[master + 1] = neighbors
