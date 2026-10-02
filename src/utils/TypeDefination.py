import taichi as ti


#===================================== #
#           Type Definition            #
#===================================== #
vec2f = ti.types.vector(2, float)
vec3f = ti.types.vector(3, float)
vec4f = ti.types.vector(4, float)
vec5f = ti.types.vector(5, float)
vec6f = ti.types.vector(6, float)
vec8f = ti.types.vector(8, float)
vec9f = ti.types.vector(9, float)
vec12f = ti.types.vector(12, float)
vec2i = ti.types.vector(2, int)
vec3i = ti.types.vector(3, int)
vec5i = ti.types.vector(5, int)
vec8i = ti.types.vector(8, int)
vec6i = ti.types.vector(6, int)
vec26i = ti.types.vector(26, int)
vec2u8 = ti.types.vector(2, ti.u8)
vec3u8 = ti.types.vector(3, ti.u8)

mat2x2 = ti.types.matrix(2, 2, float)
mat2x5 = ti.types.matrix(2, 5, float)
mat3x2 = ti.types.matrix(3, 2, float)
mat3x3 = ti.types.matrix(3, 3, float)
mat3x4 = ti.types.matrix(3, 4, float)
mat3x5 = ti.types.matrix(3, 5, float)
mat3x9 = ti.types.matrix(3, 9, float)
mat4x4 = ti.types.matrix(4, 4, float)
mat5x5 = ti.types.matrix(5, 5, float)
mat6x3 = ti.types.matrix(6, 3, float)
mat6x6 = ti.types.matrix(6, 6, float)
mat8x3 = ti.types.matrix(8, 3, float)
mat9x9 = ti.types.matrix(9, 9, float)
u1 = ti.types.quant.int(1, False)