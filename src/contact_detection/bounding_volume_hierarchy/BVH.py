import taichi as ti

from src.utils.constants import IS_LEAF, MAX_PRIM, DBL_MAX

#            0                  | 1         | 2             | 3             | 4             |5          |6      |9      | 
#            32bit              | 32bit     | 32bit         | 32bit         | 32bit         |32bit      |96bit  |96bit 
#self.bvh_node  : is_leaf axis  |left_node    right_node      parent_node     leaf_index     leaf_count  min_v3  max_v3   12
#            1bit   2bit     


@ti.data_oriented
class Bvh:
    def __init__(self, primitive_count, domain):
        self.domain = domain
        self.primitive_count = primitive_count
        if MAX_PRIM > 1:
            self.leaf_count = int(self.primitive_count / MAX_PRIM + 1)
        else:
            self.leaf_count = self.primitive_count
        self.node_count = self.leaf_count * 2 - 1
        self.bvh_node = ti.Vector.field(12, dtype=float)
        ti.root.dense(ti.i, self.node_count).place(self.bvh_node )

    ########################host function#####################################
    def print_node_info(self):
        bvh_node = self.bvh_node.to_numpy()
        fo = open("nodelist.txt", "w")
        for index in range(self.node_count):
            is_leaf = int(bvh_node[index, 0]) & 0x0001
            left    = int(bvh_node[index, 1])
            right   = int(bvh_node[index, 2])

            parent  = int(bvh_node[index, 3])
            prim_index = int(bvh_node[index, 4])
            prim_count = int(bvh_node[index, 5])

            min_point = [bvh_node[index, 6], bvh_node[index, 7],  bvh_node[index, 8]]
            max_point = [bvh_node[index, 9], bvh_node[index, 10], bvh_node[index, 11]]
            chech_pass = 1
            leaf_node_count = 0

            if is_leaf == IS_LEAF:
                leaf_node_count += 1
            else:
                for i in range(3):
                    if (min_point[i] != min(bvh_node[left, 6+i], bvh_node[right, 6+i])) & (max_point[i] != max(bvh_node[left, 9+i], bvh_node[right, 9+i])):
                        chech_pass = 0
                        break
                    
            if chech_pass == 1:
                print("node:%d l:%d r:%d p:%d leaf:%d %d  min:%.2f %.2f %.2f max:%.2f %.2f %.2f"%(index, left, right, parent, prim_index,prim_count, min_point[0],min_point[1],min_point[2],\
                    max_point[0],max_point[1],max_point[2]), file = fo)
            else:
                print("xxxx:%d l:%d r:%d p:%d leaf:%d %d  min:%.2f %.2f %.2f max:%.2f %.2f %.2f"%(index, left, right, parent, prim_index,prim_count,  min_point[0],min_point[1],min_point[2],\
                    max_point[0],max_point[1],max_point[2]), file = fo)
        fo.close()

    ############node manipulate ##############
    @ti.func
    def init_bvh_node(self, index):
        self.bvh_node[index][0]  = -1.0
        self.bvh_node[index][1]  = -1.0
        self.bvh_node[index][2]  = -1.0
        self.bvh_node[index][3]  = -1.0
        self.bvh_node[index][4]  = -1.0
        self.bvh_node[index][5]  = -1.0       
        self.bvh_node[index][6]  = DBL_MAX
        self.bvh_node[index][7]  = DBL_MAX
        self.bvh_node[index][8]  = DBL_MAX
        self.bvh_node[index][9]  = -DBL_MAX
        self.bvh_node[index][10] = -DBL_MAX
        self.bvh_node[index][11] = -DBL_MAX

    @ti.func
    def set_node_type(self, index, type):
        self.bvh_node[index][0] = float(int(self.bvh_node[index][0]) & (0xfffe | type))
    
    @ti.func
    def set_node_axis(self, index, axis):
        axis = axis<<1
        self.bvh_node[index][0] =float(int(self.bvh_node[index][0]) & (0xfff9 | type))
    
    @ti.func
    def set_node_left(self,  index, left):
        self.bvh_node[index][1]  = float(left)

    @ti.func
    def set_node_right(self,  index, right):
        self.bvh_node[index][2]  = float(right)

    @ti.func
    def set_node_parent(self, index, parent):
        self.bvh_node[index][3]  = float(parent)

    @ti.func
    def set_node_leaf_index(self, index, leaf):
        self.bvh_node[index][4]  = float(leaf)

    @ti.func
    def set_node_leaf_count(self, index, count):
        self.bvh_node[index][5]  = float(count)

    @ti.func
    def set_node_min_max(self,  index, minv,maxv):
        self.bvh_node[index][6]  = minv[0]
        self.bvh_node[index][7]  = minv[1]
        self.bvh_node[index][8]  = minv[2]
        self.bvh_node[index][9]  = maxv[0]
        self.bvh_node[index][10] = maxv[1]
        self.bvh_node[index][11] = maxv[2]

    @ti.func
    def get_node_has_box(self,  index):
        return (self.bvh_node[index][6] <= self.bvh_node[index][9]) and (self.bvh_node[index][7] <= self.bvh_node[index][10]) and (self.bvh_node[index][8] <= self.bvh_node[index][11])

    @ti.func
    def get_node_child(self, index):
        return int(self.bvh_node[index][1]), int(self.bvh_node[index][2])
    
    @ti.func
    def get_node_parent(self, index):
        return int(self.bvh_node[index][3])
    
    @ti.func
    def get_node_leaf_index(self,  index):
        return int(self.bvh_node[index][4])
    
    @ti.func
    def get_node_leaf_count(self,  index):
        return int(self.bvh_node[index][5])
    
    @ti.func
    def get_node_min(self, index):
        return ti.Vector([self.bvh_node[index][6], self.bvh_node[index][7], self.bvh_node[index][8]])
    
    @ti.func
    def get_node_max(self, index):
        return ti.Vector([self.bvh_node[index][9], self.bvh_node[index][10], self.bvh_node[index][11]])
    
    @ti.func
    def get_node_min_max(self, index):
        return self.get_node_min(index), self.get_node_max(index)

    @ti.func
    def get_node_prim_index(self, leaf_index, count):
        raise NotImplementedError()

    