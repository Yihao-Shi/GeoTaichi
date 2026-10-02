import taichi as ti
import numpy as np

from src.contact_detection.bounding_volume_hierarchy.AABB import AABB
from src.utils.BitFunction import morton3d32
from src.utils.TypeDefination import vec3f



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
        internal_node_visited : ti.field
            Flags indicating if an internal node has been visited during traversal, shape (n_batches, n_aabbs - 1).
        query_result : ti.field
            Query results as a vector of (batch id, self id, query id), shape (max_n_query_results).
        query_result_count : ti.field
            Counter for the number of query results.

    Notes
    ------
        For algorithmic details, see:
        https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf
    """

    def __init__(self, n_aabbs=0, aabb: AABB=None):
        if aabb is None:
            aabb = AABB(n_aabbs)
        self.aabb = aabb

        # Maximum stack depth for traversal
        self.max_stack_depth = 64
        self.aabb_min = ti.field(vec3f, shape=(self.aabb.n_batches))
        self.aabb_max = ti.field(vec3f, shape=(self.aabb.n_batches))
        self.scale = ti.field(vec3f, shape=(self.aabb.n_batches))
        self.morton_codes = ti.field(ti.u64, shape=(self.aabb.n_aabbs))

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
        # Temporary storage for radix sort
        self.tmp_morton_codes = ti.field(ti.u64, shape=(self.aabb.n_aabbs))

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
        # Whether an internal node has been visited during traversal
        self.internal_node_visited = ti.field(ti.u8, shape=(sum([batch_size - 1 for batch_size in self.aabb.batch_size])))
        # Auxiliary list
        self.prefix_batch_size = ti.field(int, shape=self.aabb.n_batches + 1)
        self.leaf2batch = ti.field(ti.u8, shape=self.aabb.n_aabbs)
        self.internal2batch = ti.field(ti.u8, shape=self.aabb.n_aabbs - self.aabb.n_batches)

    def initialize(self, domain=None, active_aabbs=None):
        leaf2batch = np.repeat(np.arange(self.aabb.n_batches, dtype=np.uint8), np.asarray(self.aabb.batch_size))
        internal2batch = np.repeat(np.arange(self.aabb.n_batches, dtype=np.uint8), np.asarray(self.aabb.batch_size) - 1)
        self.leaf2batch.from_numpy(leaf2batch)
        self.internal2batch.from_numpy(internal2batch)
        self.prefix_batch_size.from_numpy(np.asarray(self.aabb.prefix_batch_size).astype(np.int32))
        if active_aabbs is not None:
            self.aabb.reset(active_aabbs)
        if domain is None:
            self.adaptive_simulation_domain(self.aabb.n_batches, self.aabb.n_aabbs)
        else:
            self.initial_simulation_domain(self.aabb.n_batches, domain)

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

    def build(self):
        """
        Build the BVH from the axis-aligned bounding boxes (AABBs).
        """
        self.adaptive_simulation_domain(self.aabb.n_batches, self.aabb.n_aabbs)
        self.compute_morton_codes(self.aabb.n_aabbs)
        self.radix_sort_morton_codes(self.aabb.n_batches, self.aabb.n_aabbs)
        self.build_radix_tree(self.aabb.n_batches, self.aabb.n_aabbs)
        self.compute_bounds(self.aabb.n_batches, self.aabb.n_aabbs)

    def refit(self):
        self.compute_bounds(self.aabb.n_batches, self.aabb.n_aabbs)

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
            morton_code = morton3d32(*scaled_center)
            self.morton_codes[n_aabb] = (ti.u64(morton_code) << 32) | ti.u64(n_aabb - prefix_batch_num)

    @ti.kernel
    def radix_sort_morton_codes(self, n_batches: int, currnet_n_aabb: int):
        """
        Radix sort the morton codes, using 8 bits at a time.
        """
        for i in ti.static(range(self.num_iters)):
            # Clear histogram
            for j in range(n_batches * self.num_buckets):
                self.hist[j] = 0

            # Fill histogram
            # TODO: Why this section cannot be parallized?
            ti.loop_config(serialize=True)
            for n_aabb in range(currnet_n_aabb):
                batch_id = int(self.leaf2batch[n_aabb])
                code = (self.morton_codes[n_aabb] >> (i * 8)) & 0xFF
                self.offset[n_aabb] = ti.atomic_add(self.hist[batch_id * self.num_buckets + ti.i32(code)], 1)

            # Compute prefix sum
            for batch_id in range(n_batches):
                self.prefix_sum[batch_id * self.num_buckets + 0] = 0
                for j in range(1, self.num_buckets):  # sequential prefix sum
                    self.prefix_sum[batch_id * self.num_buckets + j] = self.prefix_sum[batch_id * self.num_buckets + j - 1] + self.hist[batch_id * self.num_buckets + j - 1]

            # Reorder morton codes
            for n_aabb in range(currnet_n_aabb):
                batch_id = int(self.leaf2batch[n_aabb])
                prefix_batch_num = self.prefix_batch_size[batch_id]
                code = (self.morton_codes[n_aabb] >> (i * 8)) & 0xFF
                idx = ti.i32(self.offset[n_aabb] + self.prefix_sum[batch_id * self.num_buckets + ti.i32(code)])
                self.tmp_morton_codes[prefix_batch_num + idx] = self.morton_codes[n_aabb]

            # Swap the temporary and original morton codes
            for n_aabb in range(currnet_n_aabb):
                self.morton_codes[n_aabb] = self.tmp_morton_codes[n_aabb]

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
                self.delta(i_internal, i_internal + 1, prefix_batch_num, batch_size) > self.delta(i_internal, i_internal - 1, prefix_batch_num, batch_size),
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
        return result

    @ti.kernel
    def compute_bounds(self, n_batches: int, currnet_n_aabb: int):
        """
        Compute the bounds of the BVH nodes.

        Starts from the leaf nodes and works upwards.
        """
        for n_internal in range(currnet_n_aabb - n_batches):
            batch_id = int(self.internal2batch[n_internal])
            prefix_batch_num = self.prefix_batch_size[batch_id]
            self.internal_node_visited[n_internal] = ti.u8(0)

        for n_aabb in range(currnet_n_aabb):
            batch_id = int(self.leaf2batch[n_aabb])
            prefix_batch_num = self.prefix_batch_size[batch_id]
            i_aabb = n_aabb - prefix_batch_num
            batch_size = self.prefix_batch_size[batch_id + 1] - prefix_batch_num
            prefix_i_node = 2 * prefix_batch_num - batch_id
            i_leaf = prefix_i_node + i_aabb + batch_size - 1
            idx = ti.i32(self.morton_codes[n_aabb])
            self.nodes[i_leaf].bound.min = self.aabb.aabbs[prefix_batch_num + idx].min
            self.nodes[i_leaf].bound.max = self.aabb.aabbs[prefix_batch_num + idx].max

            cur_idx = self.nodes[i_leaf].parent
            while cur_idx != -1:
                cur_i_leaf = prefix_i_node + cur_idx
                visited = ti.u1(ti.atomic_or(self.internal_node_visited[prefix_batch_num - batch_id + cur_idx], ti.u8(1)))
                if not visited:
                    break
                left_bound = self.nodes[prefix_i_node + self.nodes[cur_i_leaf].left].bound
                right_bound = self.nodes[prefix_i_node + self.nodes[cur_i_leaf].right].bound
                self.nodes[cur_i_leaf].bound.min = ti.min(left_bound.min, right_bound.min)
                self.nodes[cur_i_leaf].bound.max = ti.max(left_bound.max, right_bound.max)
                cur_idx = self.nodes[cur_i_leaf].parent

    @ti.kernel
    def query(self, primitiveNum: int, potential_primitive_num: int, batch_id: int, aabbs: ti.template(), potential_list_object_object: ti.template(), object_object: ti.template()):
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
                if aabbs[master].intersects(node.bound):
                    # If it's a leaf node, add the AABB index to the query results
                    if node.left == -1 and node.right == -1:
                        code = self.morton_codes[prefix_batch_num + node_idx - (batch_num - 1)]
                        slave = ti.i32(code & ti.u64(0xFFFFFFFF))
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
            assert neighbors <= potential_primitive_num, f"Keyword:: /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
            object_object[master + 1] = neighbors

    @ti.kernel
    def self_query(self, primitiveNum: int, potential_primitive_num: int, master_batch: int, slave_batch: int, aabbs: ti.template(), potential_list_object_object: ti.template(), object_object: ti.template()):
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
                if aabbs[prefix_master_batch_num + master].intersects(node.bound):
                    # If it's a leaf node, add the AABB index to the query results
                    if node.left == -1 and node.right == -1:
                        code = self.morton_codes[prefix_slave_batch_num + node_idx - (slave_batch_num - 1)]
                        slave = ti.i32(code & ti.u64(0xFFFFFFFF))
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
            assert neighbors <= potential_primitive_num, f"Keyword:: /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
            object_object[master + 1] = neighbors
