import taichi as ti

from src.contact_detection.bounding_volume_hierarchy.BVH import Bvh
from src.contact_detection.bounding_volume_hierarchy.SpatialDivision import SpatialDivision
from src.utils.BitFunction import clz
from src.utils.constants import DBL_MAX, MAX_PRIM, IS_LEAF


@ti.data_oriented
class LBvh(Bvh):
    def __init__(self, primitive_count, domain):
        super().__init__(self, primitive_count, domain)
        self.spatial = SpatialDivision(primitive_count)
        self.bvh_done = ti.field(dtype=int, shape=1)

    def build(self, current_primitive_count, primitive):
        self.spatial.build_morton_3d(current_primitive_count, self.domain, primitive)
        self.spatial.sort.run(current_primitive_count)
        self.spatial.merge(MAX_PRIM)
        self.leaf_count = self.spatial.leaf_count.to_numpy()[0]
        self.node_count = self.leaf_count * 2 - 1
        self.build_bvh()
        done_prev = 0
        done_num  = 0
        while done_num < self.leaf_count - 1:
            self.gen_aabb()
            done_num  = self.bvh_done.to_numpy()
            if done_num == done_prev:
                break
            done_prev = done_num
        assert done_num == self.leaf_count - 1, f"aabb gen error!!!!!!!!!!!!!!!!!!!{done_num}"

    def refit(self, current_primitive_count):
        self.refit_leaf()
        done_prev = 0
        done_num  = 0
        while done_num < self.leaf_count - 1:
            self.gen_aabb()
            done_num  = self.bvh_done.to_numpy()
            if done_num == done_prev:
                break
            done_prev = done_num
        assert done_num == self.leaf_count - 1, f"aabb gen error!!!!!!!!!!!!!!!!!!!{done_num}"

    def query(self):
        pass

    ############algorithm##############
    @ti.func
    def common_upper_bits(self, lhs, rhs):
        x = lhs ^ rhs
        return clz(x)

    @ti.func
    def determine_range(self, idx):
        l_r_range = ti.cast(ti.Vector([0, self.leaf_count - 1]), int)
        if idx != 0:
            self_code = self.spatial.morton_code[idx, 0]
            l = idx - 1
            r = idx + 1
            l_code = self.spatial.morton_code[l, 0]
            r_code = self.spatial.morton_code[r, 0]

            assert l_code == self_code or r_code == self_code, "fatal error!!"

            L_delta = self.common_upper_bits(self_code, l_code)
            R_delta = self.common_upper_bits(self_code, r_code)

            d = -1
            if R_delta > L_delta:
                d = 1
            delta_min = min(L_delta, R_delta)
            l_max = 2
            delta = -1
            i_tmp = idx + d * l_max

            if ((0 <= i_tmp) & (i_tmp < self.spatial.leaf_count[0])):
                delta = self.common_upper_bits(self_code, self.spatial.morton_code[i_tmp, 0])

            while delta > delta_min:
                l_max <<= 1
                i_tmp = idx + d * l_max
                delta = -1
                if ( (0 <= i_tmp) & (i_tmp < self.spatial.leaf_count[0])):
                    delta = self.common_upper_bits(self_code, self.spatial.morton_code[i_tmp, 0])

            l = 0
            t = l_max >> 1

            while(t > 0):
                i_tmp = idx + (l + t) * d
                delta = -1
                if ( (0 <= i_tmp) & (i_tmp < self.spatial.leaf_count[0])):
                    delta = self.common_upper_bits(self_code, self.spatial.morton_code[i_tmp, 0])
                if(delta > delta_min):
                    l += t
                t >>= 1

            l_r_range[0] = idx
            l_r_range[1] = idx + l * d
            if(d < 0):
                tmp = l_r_range[0]
                l_r_range[0] = l_r_range[1]
                l_r_range[1] = tmp 
        return l_r_range
        
    @ti.func
    def find_split(self, first, last):
        first_code = self.spatial.morton_code[first, 0]
        last_code  = self.spatial.morton_code[last, 0]
        delta_node = self.common_upper_bits(first_code, last_code)
        split = first
        stride = last - first
        while 1:
            stride = (stride + 1) >> 1
            middle = split + stride
            if (middle < last):
                delta = self.common_upper_bits(first_code, self.spatial.morton_code[middle, 0])
                if (delta > delta_node):
                    split = middle
            if stride <= 1:
                break
        return split

    @ti.kernel
    def build_bvh(self):
        for i in self.bvh_node:
            self.init_bvh_node( i)
            self.bvh_done[0] = 0

        for i in self.bvh_node:
            if i >= self.spatial.leaf_count[0]-1:
                self.set_node_type(i, IS_LEAF)
                leaf_index = self.spatial.morton_code[i - self.spatial.leaf_count[0] + 1, 1]
                leaf_count = self.spatial.morton_code[i - self.spatial.leaf_count[0] + 1, 2]

                self.set_node_leaf_index(i, leaf_index)
                self.set_node_leaf_count(i, leaf_count)
                min_v3 =ti.Vector([DBL_MAX, DBL_MAX, DBL_MAX])
                max_v3 =ti.Vector([-DBL_MAX, -DBL_MAX, -DBL_MAX])

                count = 0
                while count < leaf_count:
                    prim_index = self.spatial.sort.data_out[leaf_index + count, 1]
                    min_tmp, max_tmp = self.spatial.aabb.get_aabb(prim_index)
                    for l in  ti.static(range(3)):
                        min_v3[l] = ti.min(min_v3[l], min_tmp[l])
                        max_v3[l] = ti.max(max_v3[l], max_tmp[l])
                    count +=1
                self.set_node_min_max(i, min_v3,max_v3)
            else:
                self.set_node_type( i, 1-IS_LEAF)
                l_r_range   = self.determine_range(i)
                spilt       = self.find_split(l_r_range[0], l_r_range[1])
                left_node   = spilt
                right_node  = spilt + 1

                assert l_r_range[0] != l_r_range[1], f"{l_r_range}, {spilt}, {left_node}, {right_node}, wrong"

                if min(l_r_range[0], l_r_range[1]) == spilt :
                    left_node  += self.spatial.leaf_count[0] - 1
            
                if max(l_r_range[0], l_r_range[1]) == spilt + 1:
                    right_node  += self.spatial.leaf_count[0] - 1

                self.set_node_left(i, left_node)
                self.set_node_right(i, right_node)
                self.set_node_parent(left_node, i)
                self.set_node_parent(right_node, i)

    @ti.kernel
    def refit_leaf(self):
        for i in range(self.spatial.leaf_count[0] - 1, 2 * self.spatial.leaf_count[0] - 1):
            self.set_node_type(i, IS_LEAF)
            leaf_index = self.spatial.morton_code[i - self.spatial.leaf_count[0] + 1, 1]
            leaf_count = self.spatial.morton_code[i - self.spatial.leaf_count[0] + 1, 2]

            min_v3 = ti.Vector([DBL_MAX, DBL_MAX, DBL_MAX])
            max_v3 = ti.Vector([-DBL_MAX, -DBL_MAX, -DBL_MAX])
            for count in range(leaf_count):
                prim_index = self.spatial.sort.data_out[leaf_index + count, 1]
                min_tmp, max_tmp = self.spatial.aabb.get_aabb(prim_index)
                for l in ti.static(range(3)):
                    min_v3[l] = min(min_v3[l], min_tmp[l])
                    max_v3[l] = max(max_v3[l], max_tmp[l])
            self.set_node_min_max(i, min_v3, max_v3)

    @ti.kernel
    def gen_aabb(self):
        for j in range(self.spatial.leaf_count[0] - 1): 
            i = self.spatial.leaf_count[0] - 2 - j
            if (self.get_node_has_box(i) == 0):
                left_node, right_node = self.get_node_child(i) 
                
                is_left_rdy  = self.get_node_has_box(left_node)
                is_right_rdy = self.get_node_has_box(right_node)

                if (is_left_rdy and is_right_rdy) > 0:
                    l_min,l_max = self.get_node_min_max(left_node)  
                    r_min,r_max = self.get_node_min_max(right_node)  
                    self.set_node_min_max(i, min(l_min, r_min),max(l_max, r_max))
                    self.bvh_done[0] += 1

    @ti.func
    def get_node_prim_index(self, leaf_index, count):
        return self.spatial.sort.data_out[count + leaf_index, 1]
    
    @ti.kernel
    def query_all_aabbs(self, query_min_list: ti.template(), query_max_list: ti.template(), result: ti.template(), result_counts: ti.template()):
        for q in query_min_list:
            query_min = query_min_list[q]
            query_max = query_max_list[q]

            stack = ti.Vector([0 for _ in range(64)])  # local stack
            stack_size = 1

            res_offset = 0

            while stack_size > 0 and res_offset < result.shape[1]:
                stack_size -= 1
                node = stack[stack_size]

                node_min = self.get_node_min(node)
                node_max = self.get_node_max(node)

                overlaps = True
                for i in ti.static(range(3)):
                    if node_max[i] < query_min[i] or node_min[i] > query_max[i]:
                        overlaps = False

                if not overlaps:
                    continue

                if self.get_node_type(node) == Bvh.IS_LEAF:
                    leaf_index = self.get_node_leaf_index(node)
                    leaf_count = self.get_node_leaf_count(node)
                    for j in range(leaf_count):
                        prim_index = self.radix_sort.morton_code_s[leaf_index + j][1]
                        prim_min, prim_max = self.prim.aabb(prim_index)

                        prim_overlaps = True
                        for k in ti.static(range(3)):
                            if prim_max[k] < query_min[k] or prim_min[k] > query_max[k]:
                                prim_overlaps = False

                        if prim_overlaps and res_offset < result.shape[1]:
                            result[q, res_offset] = prim_index
                            res_offset += 1
                else:
                    left = self.get_node_left(node)
                    right = self.get_node_right(node)
                    if stack_size + 2 < 64:
                        stack[stack_size] = left
                        stack_size += 1
                        stack[stack_size] = right
                        stack_size += 1

            result_counts[q] = res_offset