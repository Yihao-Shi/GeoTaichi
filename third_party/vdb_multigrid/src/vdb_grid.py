## @package VdbGrid
# Documentations for an implementation of multi-resolution sparse structure with snode
import taichi as ti
from src.utils import *
from src.vdb_math import *


@ti.data_oriented
class VdbTransform:
    def __init__(self, voxel_extent, voxel_dim, origin):
        self.voxel_extent = voxel_extent
        self.voxel_dim = voxel_dim
        self.inv_voxel_dim = list(1.0 / np.array(voxel_dim))
        self.origin = origin
        self.coord_max = voxel_dim * voxel_extent + origin

    @ti.func
    def voxel_to_coord_packed(self, ijk: ti.template()):
        return ijk * self.voxel_dim + self.origin

    @ti.func
    def voxel_to_coord(self, i, j, k):
        return self.voxel_to_coord_packed(ti.Vector([i, j, k]))

    @ti.func
    def coord_to_voxel_packed(self, xyz: ti.template()):
        return ti.round((xyz - self.origin) * self.inv_voxel_dim, ti.i32)

    @ti.func
    def coord_to_voxel(self, x, y, z):
        return self.coord_to_voxel_packed(ti.Vector([x, y, z]))

    @ti.func
    def is_contain_packed(self, xyz: ti.template()) -> bool:
        res = self.origin <= xyz
        res |= xyz < self.coord_max
        return res[0] and res[1] and res[2]

    @ti.func
    def is_contain(self, x, y, z):
        return self.is_contain_packed(ti.Vector([x, y, z]))



## Represents the configuration of three dimensions a level in a vdb grid
class VdbLevelConfig:
    def __init__(self, log2x: int, log2y: int, log2z: int, dtype=ti.f32, child_node=None):
        vdb_assert(log2x > 0 and log2y > 0 and log2z > 0, "VdbLevelConfig needs all log2x, log2y, log2z to be positive.")
        self.dtype = dtype
        self.log2x = log2x
        self.log2y = log2y
        self.log2z = log2z
        self.slog2x = log2x
        self.slog2y = log2y
        self.slog2z = log2z

        if ti.static(child_node is not None):
            self.slog2x += child_node.slog2x
            self.slog2y += child_node.slog2y
            self.slog2z += child_node.slog2z

## Represents the operation id that is supported in a vdb grid
class VdbOpId:
    set_op = 0
    add_op = 1
    sub_op = 2
    mul_op = 3
    div_op = 4
    min_op = 5
    max_op = 6




## Internal implementation of the vdb grid 
@ti.data_oriented
class VdbDataWrapper:
    ## Maximum levels of vdb grid supported
    max_vdb_levels = 5

    ## @param num_vdb_levels a positive integer that represents the number of levels of vdb grid to construct
    #                        it needs to be less than VdbDataWrapper.max_vdb_levels
    #  @param config a taichi vector fields of size num_vdb_levels + 1 that denotes configuration of each level of the vdb grid
    #  @param sconfig a taichi vector fields of size num_vdb_levels + 1 that denotes bottom up configuration of each level of the vdb grid
    #  @param voxel_dim a taichi vector that denotes the dimensions of the smallest voxel
    def __init__(self, num_vdb_levels, config: ti.template(), sconfig: ti.template(), voxel_dim, dtype=ti.f32,
                 background_value=0.0):
        # Temporary lists for ease of construction
        pointer_list = [ ti.root ]
        bitmasked_list = []
        value_list = []

        self.voxel_dim = voxel_dim
        self.inv_voxel_dim = list(1.0 / np.array(voxel_dim))
        self.config = config
        self.sconfig = sconfig
        self.num_vdb_levels = ti.static(num_vdb_levels)
        self.background_value = background_value
        self.dtype = dtype

        # Compile time determination of fields
        if ti.static(self.num_vdb_levels > 0):
            self.value0 = ti.field(dtype)
            value_list.append(self.value0)
            self.leaf_value = self.value0
        if ti.static(self.num_vdb_levels > 1):
            self.value1 = ti.field(dtype)
            value_list.append(self.value1)
            self.leaf_value = self.value1
        if ti.static(self.num_vdb_levels > 2):
            self.value2 = ti.field(dtype)
            value_list.append(self.value2)
            self.leaf_value = self.value2
        if ti.static(self.num_vdb_levels > 3):
            self.value3 = ti.field(dtype)
            value_list.append(self.value3)
            self.leaf_value = self.value3
        if ti.static(self.num_vdb_levels > 4):
            self.value4 = ti.field(dtype)
            value_list.append(self.value4)
            self.leaf_value = self.value4


        for i in ti.static(range(self.num_vdb_levels)):
            bitmasked_list.append(pointer_list[-1].bitmasked(ti.ijk, 1 << config[i]))
            bitmasked_list[-1].place(value_list[i])
            if ti.static(i == 0):
                self.level0_bitmasked = bitmasked_list[0]
            if i + 1 < self.num_vdb_levels:
                pointer_list.append(pointer_list[-1].pointer(ti.ijk, 1 << config[i]))

        if ti.static(self.num_vdb_levels > 1):
            self.child0 = pointer_list[1]
        if ti.static(self.num_vdb_levels > 2):
            self.child1 = pointer_list[2]
        if ti.static(self.num_vdb_levels > 3):
            self.child2 = pointer_list[3]
        if ti.static(self.num_vdb_levels > 4):
            self.child3 = pointer_list[4]

    ## @brief: Deactivate all internal snodes. Clear all the data.
    def clear(self):
        self.level0_bitmasked.deactivate_all()
        if ti.static(self.num_vdb_levels > 1):
            self.child0.deactivate_all()

    
    ## @param i, j, k The coordinates for the set operation
    #  @param value The value for the set operation
    #  @param f The field for the set operation
    @ti.func
    def set_op(self, i, j, k, value, f: ti.template()):
        f[i, j, k] = value

    ## @param i, j, k The coordinates for the add operation
    #  @param value The value for the add operation
    #  @param f The field for the add operation
    @ti.func
    def add_op(self, i, j, k, value, f: ti.template()):
        f[i, j, k] += value

    ## @param i, j, k The coordinates for the sub operation
    #  @param value The value for the sub operation
    #  @param f The field for the sub operation
    @ti.func
    def sub_op(self, i, j, k, value, f: ti.template()):
        f[i, j, k] -= value

    ## @param i, j, k The coordinates for the mul operation
    #  @param value The value for the mul operation
    #  @param f The field for the mul operation
    @ti.func
    def mul_op(self, i, j, k, value, f: ti.template()):
        f[i, j, k] *= value

    ## @param i, j, k The coordinates for the div operation
    #  @param value The value for the div operation
    #  @param f The field for the div operation
    @ti.func
    def div_op(self, i, j, k, value, f: ti.template()):
        f[i, j, k] /= value


    ## @param i, j, k The coordinates for the atomic min operation
    #  @param value The value for the atomic min operation
    #  @param f The field for the atomic min operation
    @ti.func
    def min_op(self, i, j, k, value, f: ti.template()):
        ti.atomic_min(f[i, j, k], value)

    ## @param i, j, k The coordinates for the atomic max operation
    #  @param value The value for the atomic max operation
    #  @param f The field for the atomic max operation
    @ti.func
    def max_op(self, i, j, k, value, f: ti.template()):
        ti.atomic_max(f[i, j, k], value)


    ## @param i, j, k The coordinates for operation to apply
    #  @param value The value for the operation to apply
    #  @param f The field for the operation to apply
    #  @param op The operation function
    #  @param fixed_level The compile-time fixed level that correponds to the field \a f 
    #  @param level The compile-time level that indicates level to apply the operation
    @ti.func
    def apply_op(self, i, j, k, value, op: ti.template(), f: ti.template(), fixed_level: ti.template(), level: ti.template()):
        if ti.static(level == fixed_level):
            op(i, j, k, value, f)


    ## @param level The compile-time level to apply specific operation \a op
    #  @param i, j, k The local voxel coordinates to apply operation 
    #  @param value The value to apply operations
    #  
    #  @brief Implementation of modifying a value at (i, j, k) with given operations with multi-level grids
    @ti.func
    def modify_value(self, level: ti.template(), i, j, k, value: ti.template(), op: ti.template()):
        assert level >= 0, "Level needs to be non-negative."
        assert level < self.num_vdb_levels, "Level exceeds maximum vdb level {}".format(self.num_vdb_levels)

        if ti.static(self.num_vdb_levels > 0):
            self.apply_op(i, j, k, value, op, self.value0, 0, level)
        if ti.static(self.num_vdb_levels > 1):
            self.apply_op(i, j, k, value, op, self.value1, 1, level)
        if ti.static(self.num_vdb_levels > 2):
            self.apply_op(i, j, k, value, op, self.value2, 2, level)
        if ti.static(self.num_vdb_levels > 3):
            self.apply_op(i, j, k, value, op, self.value3, 3, level)
        if ti.static(self.num_vdb_levels > 4):
            self.apply_op(i, j, k, value, op, self.value4, 4, level)


    ## @param level The compile-time level to set 
    #  @param i, j, k The local voxel coordinates for the set operation
    #  @param value The value for the set operation
    @ti.func
    def set_value_local(self, level: ti.template(), i, j, k, value):
        self.modify_value(level, i, j, k, value, self.set_op)

    ## @param level The compile-time level to add 
    #  @param i, j, k The local voxel coordinates for the add operation
    #  @param value The value for the add operation
    @ti.func
    def add_value_local(self, level: ti.template(), i, j, k, value):
        self.modify_value(level, i, j, k, value, self.add_op)

    ## @param level The compile-time level to subtract 
    #  @param i, j, k The local voxel coordinates for the sub operation
    #  @param value The value for the sub operation
    @ti.func
    def sub_value_local(self, level: ti.template(), i, j, k, value):
        self.modify_value(level, i, j, k, value, self.sub_op)

    ## @param level The compile-time level to multiply 
    #  @param i, j, k The local voxel coordinates for the mul operation
    #  @param value The value for the mul operation
    @ti.func
    def mul_value_local(self, level: ti.template(), i, j, k, value):
        self.modify_value(level, i, j, k, value, self.mul_op)

    ## @param level The compile-time level to divide 
    #  @param i, j, k The local voxel coordinates for the div operation
    #  @param value The value for the div operation
    @ti.func
    def div_value_local(self, level: ti.template(), i, j, k, value):
        self.modify_value(level, i, j, k, value, self.div_op)

    ## @param level The compile-time level to perform atomic min
    #  @param i, j, k The local voxel coordinates for the min operation
    #  @param value The value for the min operation
    @ti.func
    def min_value_local(self, level: ti.template(), i, j, k, value):
        self.modify_value(level, i, j, k, value, self.min_op)

    ## @param level The compile-time level to perform atomic max
    #  @param i, j, k The local voxel coordinates for the max operation
    #  @param value The value for the max operation
    @ti.func
    def max_value_local(self, level: ti.template(), i, j, k, value):
        self.modify_value(level, i, j, k, value, self.max_op)

    ## @param level The compile-time level to read value from  
    #  @param i, j, k The local voxel coordinates to read value from
    #  @return The value read from \a level at (i, j, k)
    @ti.func
    def read_value_local(self, level: ti.template(), i, j, k):
        assert level < self.num_vdb_levels, "Level exceeds maximum vdb level {}".format(self.num_vdb_levels)

        res = self.background_value
        if ti.static(level + 1 == self.num_vdb_levels):
            res = self.leaf_value[i, j, k]
        else:
            if ti.static(level == 0 and self.num_vdb_levels > 0):
                res = self.value0[i, j, k]
            elif ti.static(level == 1 and self.num_vdb_levels > 1):
                res = self.value1[i, j, k]
            elif ti.static(level == 2 and self.num_vdb_levels > 2):
                res = self.value2[i, j, k]
            elif ti.static(level == 3 and self.num_vdb_levels > 3):
                res = self.value3[i, j, k]
            elif ti.static(level == 4 and self.num_vdb_levels > 4):
                res = self.value4[i, j, k]

        return ti.cast(res, self.dtype)

    ## @param level The compile-time level to read value from  
    #  @param i, j, k The world voxel coordinates to read value from
    #  @return The value read from \a level at (i, j, k)
    @ti.func
    def read_value_world(self, level: ti.template(), i, j, k):
        assert level < self.num_vdb_levels, "Level exceeds maximum vdb level {}".format(self.num_vdb_levels)

        res = self.background_value
        if ti.static(level + 1 == self.num_vdb_levels):
            res = self.leaf_value[i, j, k]
        else:
            i, j, k = self.rescale_index_from_world_voxel(level, i, j, k)
            res = self.read_value_local(level, i, j, k)

        return ti.cast(res, self.dtype)
    


    ## @param level   The target child snode to scale to
    #  @param i, j, k The coordinates of world voxel to rescale
    @ti.func
    def rescale_index_from_world_voxel(self, level: ti.template(), i, j, k):
        assert level + 1 <= self.num_vdb_levels, "Should never rescale index beyond the leaf level"
        index = ti.Vector.zero(ti.i32, n=3)

        if ti.static(level + 1 == self.num_vdb_levels):
            index = ti.Vector([i, j, k])
        elif ti.static(level == 0 and self.num_vdb_levels > 1):
            index = ti.rescale_index(self.leaf_value, self.child0, ti.Vector([i, j, k]))
        elif ti.static(level == 1 and self.num_vdb_levels > 2):
            index = ti.rescale_index(self.leaf_value, self.child1, ti.Vector([i, j, k]))
        elif ti.static(level == 2 and self.num_vdb_levels > 3):
            index = ti.rescale_index(self.leaf_value, self.child2, ti.Vector([i, j, k]))
        elif ti.static(level == 3 and self.num_vdb_levels > 4):
            index = ti.rescale_index(self.leaf_value, self.child3, ti.Vector([i, j, k]))
        else:
            assert False

        return index


    ## @param level The level of sparse grid to check
    #  @param i, j, k The local voxel coordinate 
    #@detail: Return if a child is active
    @ti.func
    def is_child_active_local(self, level: ti.template(), i, j, k):
        res = False

        if ti.static(level == 0 and self.num_vdb_levels > 1):
            res = ti.is_active(self.child0, [i, j, k])
        elif ti.static(level == 1 and self.num_vdb_levels > 2):
            res = ti.is_active(self.child1, [i, j, k])
        elif ti.static(level == 2 and self.num_vdb_levels > 3):
            res = ti.is_active(self.child2, [i, j, k])
        elif ti.static(level == 3 and self.num_vdb_levels > 4):
            res = ti.is_active(self.child3, [i, j, k])

        return res


    ## @param level denotes the level of sparse grid to check
    #  @param i, j, k denotes the coordinate of world voxel
    #  @return Active status of a child in \a level and world voxel coodinates (i, j, k) 
    @ti.func
    def is_child_active_world(self, level: ti.template(), i, j, k) -> bool:
        i, j, k = self.rescale_index_from_world_voxel(level, i, j, k)
        return self.is_child_active_local(i, j, k)


    ## @param vertices denotes The vertex field that will be field with 1 cube of vertices
    #  @param vertex_offset The offset of in the vertex field to start filling the vertices
    #  @param i, j, k The index of the cube in corresponding cube_dim
    #  @param cube_dim The dimensions of specified cube
    @ti.func
    def fill_vdb_bbox_cube_vertices(self, vertices:ti.template(), vertex_offset, indices: ti.template(), index_offset, i, j, k, cube_dim: ti.template()):
        position_offset = ti.Vector([i, j, k]) * cube_dim
        dx = ti.Vector([1, 0, 0])
        dy = ti.Vector([0, 1, 0])
        dz = ti.Vector([0, 0, 1])

        vertices[vertex_offset] = position_offset
        vertices[vertex_offset + 1] = position_offset + dx * cube_dim
        vertices[vertex_offset + 2] = position_offset + (dx + dy) * cube_dim
        vertices[vertex_offset + 3] = position_offset + dy * cube_dim
        vertices[vertex_offset + 4] = position_offset + dz * cube_dim
        vertices[vertex_offset + 5] = position_offset + (dx + dz) * cube_dim
        vertices[vertex_offset + 6] = position_offset + (dx + dy + dz) * cube_dim
        vertices[vertex_offset + 7] = position_offset + (dy + dz) * cube_dim

        indices[index_offset + 0] = vertex_offset + 0
        indices[index_offset + 1] = vertex_offset + 1

        indices[index_offset + 2] = vertex_offset + 1
        indices[index_offset + 3] = vertex_offset + 2

        indices[index_offset + 4] = vertex_offset + 2
        indices[index_offset + 5] = vertex_offset + 3

        indices[index_offset + 6] = vertex_offset + 3
        indices[index_offset + 7] = vertex_offset + 0

        indices[index_offset + 8] = vertex_offset + 0
        indices[index_offset + 9] = vertex_offset + 4

        indices[index_offset + 10] = vertex_offset + 4
        indices[index_offset + 11] = vertex_offset + 5

        indices[index_offset + 12] = vertex_offset + 5
        indices[index_offset + 13] = vertex_offset + 6

        indices[index_offset + 14] = vertex_offset + 6
        indices[index_offset + 15] = vertex_offset + 2

        indices[index_offset + 16] = vertex_offset + 7
        indices[index_offset + 17] = vertex_offset + 4

        indices[index_offset + 18] = vertex_offset + 6
        indices[index_offset + 19] = vertex_offset + 7

        indices[index_offset + 20] = vertex_offset + 1
        indices[index_offset + 21] = vertex_offset + 5

        indices[index_offset + 22] = vertex_offset + 3
        indices[index_offset + 23] = vertex_offset + 7



    ## @param vertices The vertex field of visualized bounding box
    #  @param indices The index field of visualized bounding box
    #  @param per_vertex_color The per_vertex_color field for the visualization in ggui
    @ti.kernel
    def generate_vdb_bbox_vertices_impl(self, vertices: ti.template(), indices: ti.template(), per_vertex_color: ti.template()):
        vertex_count = 0
        index_count = 0
        if ti.static(self.num_vdb_levels > 1):
            for i, j, k in self.child0:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim * (1 << self.sconfig[1]))
                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.16, 0.5, 0.73)

        if ti.static(self.num_vdb_levels > 0):
            for i, j, k in self.value0:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim * (1 << self.sconfig[1]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.16, 0.5, 0.73)


        if ti.static(self.num_vdb_levels > 2):

            for i, j, k in self.child1:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim * (1 << self.sconfig[2]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.906, 0.298, 0.235)

        if ti.static(self.num_vdb_levels > 1):
            for i, j, k in self.value1:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim * (1 << self.sconfig[2]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.906, 0.298, 0.235)

        if ti.static(self.num_vdb_levels > 3):
            for i, j, k in self.child2:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim * (1 << self.sconfig[3]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.153, 0.682, 0.376)

        if ti.static(self.num_vdb_levels > 2):
            for i, j, k in self.value2:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k,
                                                 self.voxel_dim * (1 << self.sconfig[3]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.153, 0.682, 0.376)

        if ti.static(self.num_vdb_levels > 4):
            for i, j, k in self.child3:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim * (1 << self.sconfig[4]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.557, 0.267, 0.678)

        if ti.static(self.num_vdb_levels > 3):
            for i, j, k in self.value3:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k,
                                                 self.voxel_dim * (1 << self.sconfig[4]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.557, 0.267, 0.678)

        if ti.static(self.num_vdb_levels > 4):
            for i, j, k in self.value4:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim)

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.902, 0.494, 0.133)


    ## @param level The level in the vdb to prune 
    #  @param snode The snode for the child pointer at \a level 
    #  @param f The value field in the vdb grid to read value from
    #  @param tolerance The compile-time value that is allowed for different values to be considered the same for pruning
    @ti.func
    def prune_level_tolerance(self, level: ti.template(), snode: ti.template(), f: ti.template(), tolerance: ti.template()):
        assert level + 1 < self.num_vdb_levels, "Must not prune level at or beyond the leaf level."
        config = self.config[level + 1]
        counter = 0
        for i, j, k in snode:
            is_equal = True
            ni, nj, nk = ti.rescale_index(snode, f, [i, j, k])
            value = f[ni, nj, nk]
            for x, y, z in ti.ndrange(1 << config[0], 1 << config[1], 1 << config[2]):
                nx = ni + x
                ny = nj + y
                nz = nk + z
                if ti.static(level + 1 == self.num_vdb_levels) or not self.is_child_active_local(level + 1, nx, ny, nz):
                    if ti.static(tolerance != 0):
                        is_equal &= approx_equal(f[nx, ny, nz], value, tolerance)
                    else:
                        is_equal &= (f[nx, ny, nz] == value)
                else:
                    is_equal &= False

                if not is_equal:
                    break

            if is_equal:
                ti.deactivate(snode, [i, j, k])
                counter += 1
                if value != self.background_value:
                    self.set_value_local(level, i, j, k, value)

    ## @param tolerance The compile-time value that is allowed for different values to be considered the same for pruning
    @ti.kernel
    def prune(self, tolerance:ti.template()):
        # Prune the tree from bottom up
        if ti.static(self.num_vdb_levels > 4):
            self.prune_level_tolerance(3, self.child3, self.value4, tolerance)
        if ti.static(self.num_vdb_levels > 3):
            self.prune_level_tolerance(2, self.child2, self.value3, tolerance)
        if ti.static(self.num_vdb_levels > 2):
            self.prune_level_tolerance(1, self.child1, self.value2, tolerance)
        if ti.static(self.num_vdb_levels > 1):
            self.prune_level_tolerance(0, self.child0, self.value1, tolerance)

@ti.data_oriented
class VdbGrid:

    def __init__(self, voxel_dim, level_configs=None, dtype=ti.f32, background_value=0.0, origin=ti.Vector([0.0, 0.0, 0.0])):


        if level_configs is None:
            level_configs = [5, 5, 5, 5, 3]
        else:
            vdb_assert(isinstance(level_configs, type([])), "Tree levels should be an array type.")
            for level_dim in level_configs:
                vdb_assert(level_dim > 0, "The vdb level dimension must be greater than 0.")

        self.num_vdb_levels = ti.static(len(level_configs))
        self.leaf_level = self.num_vdb_levels - 1

        # Build configuration of each level
        config_list = []
        for i in range(self.num_vdb_levels - 1, -1, -1):
            if i + 1 == self.num_vdb_levels:
                config_list.append(VdbLevelConfig(level_configs[i], level_configs[i], level_configs[i], dtype))

            else:
                config_list.append(VdbLevelConfig(level_configs[i], level_configs[i], level_configs[i], dtype, config_list[-1]))

        config_list.reverse()

        # Store configurations in fields
        self.config = ti.Vector.field(n=3, dtype=ti.i32, shape=self.num_vdb_levels + 1)
        self.sconfig = ti.Vector.field(n=3, dtype=ti.i32, shape=self.num_vdb_levels + 1)
        self.ssize = ti.field(ti.f32, shape=self.num_vdb_levels + 1)


        for i in ti.static(range(self.num_vdb_levels)):
            self.config[i] = ti.Vector([config_list[i].log2x, config_list[i].log2y, config_list[i].log2z])
            self.sconfig[i] = ti.Vector([config_list[i].slog2x, config_list[i].slog2y, config_list[i].slog2z])
            sconfig = self.sconfig[i]
            self.ssize[i] = 1 << (sconfig[0] + sconfig[1] + sconfig[2])

        voxel_extent = ti.Vector([1 << self.sconfig[0][0], 1 << self.sconfig[0][1], 1 << self.sconfig[0][2]])
        self.transform = VdbTransform(voxel_extent, voxel_dim, origin)

        self.data_wrapper = VdbDataWrapper(self.num_vdb_levels, self.config, self.sconfig, self.transform.voxel_dim, dtype,
                                           background_value)

    

    ##@param: i, j, k denotes the coordinates in the voxel space
    #        value denotes the target value to set
    # @detail: change the value of voxel to specificed
    @ti.func
    def modify_value_world(self, i, j, k, value, op: ti.template()): 
        if ti.static(op == VdbOpId.set_op):
            self.data_wrapper.set_value_local(self.leaf_level, i, j, k, value)
        elif ti.static(op == VdbOpId.add_op):
            self.data_wrapper.add_value_local(self.leaf_level, i, j, k, value)
        elif ti.static(op == VdbOpId.sub_op):
            self.data_wrapper.sub_value_local(self.leaf_level, i, j, k, value)
        elif ti.static(op == VdbOpId.mul_op):
            self.data_wrapper.mul_value_local(self.leaf_level, i, j, k, value)
        elif ti.static(op == VdbOpId.div_op):
            self.data_wrapper.div_value_local(self.leaf_level, i, j, k, value)
        elif ti.static(op == VdbOpId.min_op):
            self.data_wrapper.min_value_local(self.leaf_level, i, j, k, value)
        elif ti.static(op == VdbOpId.max_op):
            self.data_wrapper.max_value_local(self.leaf_level, i, j, k, value)
        else:
            print("Unrecognized operation to modify vdb value")
            pass

    @ti.func
    def set_value_world(self, i, j, k, value):
        if i >= 0 and j >= 0 and k >= 0:
            self.modify_value_world(i, j, k, value, VdbOpId.set_op)

    @ti.func
    def add_value_world(self, i, j, k, value):
        if i >= 0 and j >= 0 and k >= 0:
            self.modify_value_world(i, j, k, value, VdbOpId.add_op)

    @ti.func
    def sub_value_world(self, i, j, k, value):
        if i >= 0 and j >= 0 and k >= 0:
            self.modify_value_world(i, j, k, value, VdbOpId.sub_op)

    @ti.func
    def mul_value_world(self, i, j, k, value):
        if i >= 0 and j >= 0 and k >= 0:
            self.modify_value_world(i, j, k, value, VdbOpId.mul_op)

    @ti.func
    def div_value_world(self, i, j, k, value):
        if i >= 0 and j >= 0 and k >= 0:
            self.modify_value_world(i, j, k, value, VdbOpId.div_op)

    @ti.func
    def min_value_world(self, i, j, k, value):
        if i >= 0 and j >= 0 and k >= 0:
            self.modify_value_world(i, j, k, value, VdbOpId.min_op)

    @ti.func
    def max_value_world(self, i, j, k, value):
        if i >= 0 and j >= 0 and k >= 0:
            self.modify_value_world(i, j, k, value, VdbOpId.max_op)

    @ti.func
    def read_value_impl(self, level: ti.template(), i, j, k):
        res = self.data_wrapper.background_value
        if ti.static(level + 1 == self.num_vdb_levels):
            res = self.data_wrapper.read_value_world(self.leaf_level, i, j, k)
        else:
            if not self.data_wrapper.is_child_active_world(level, i, j, k):
                res = self.data_wrapper.read_value_world(level, i, j, k)
            else:
                res = self.read_value_impl(level + 1, i, j, k)

        return ti.cast(res, self.data_wrapper.dtype)

    @ti.func
    def read_value_world(self, i, j, k):
        res = self.data_wrapper.background_value
        if i >= 0 and j >= 0 and k >= 0:
            res = self.read_value_impl(self.leaf_level, i, j, k)
        return ti.cast(res, self.data_wrapper.dtype)

    @ti.func
    def set_value_packed(self, xyz: ti.template(), value):
        if self.transform.is_contain_packed(xyz):
            i, j, k = self.transform.coord_to_voxel_packed(xyz)
            self.set_value_world(i, j, k, value)
    @ti.func
    def set_value(self, x, y, z, value):
        self.set_value_packed(ti.Vector([x, y, z]), value)

    @ti.func
    def read_value_packed(self, xyz: ti.template()):
        res = self.data_wrapper.background_value
        if self.transform.is_contain_packed(xyz):
            i, j, k = self.transform.coord_to_voxel_packed(xyz)
            res = self.read_value_world(i, j, k)
        return ti.cast(res, self.data_wrapper.dtype)

    @ti.func
    def read_value(self, x, y, z):
        return self.read_value_packed(ti.Vector([x, y, z]))

    @ti.func
    def add_value_packed(self, xyz: ti.template(), value):
        if self.transform.is_contain_packed(xyz):
            i, j, k = self.transform.coord_to_voxel_packed(xyz)
            self.add_value_world(i, j, k, value)

    @ti.func
    def add_value(self, x, y, z, value):
        self.add_value_packed(ti.Vector([x, y, z]), value)


    def prune(self, tolerance: ti.template()):
        self.data_wrapper.prune(tolerance)

    def clear(self):
        self.data_wrapper.clear()






            
        

