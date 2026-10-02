import os

import taichi as ti


_REAL_DTYPE_ENV = "GEOTAICHI_REAL_DTYPE"
_real_dtype_name = os.environ.get(_REAL_DTYPE_ENV, "float32").strip().lower()
if _real_dtype_name in ("float32", "f32"):
    real = ti.f32
    REAL_DTYPE_NAME = "float32"
elif _real_dtype_name in ("float64", "f64"):
    real = ti.f64
    REAL_DTYPE_NAME = "float64"
else:
    raise RuntimeError(
        f"{_REAL_DTYPE_ENV} must be one of float32/f32/float64/f64, "
        f"got {_real_dtype_name!r}"
    )


#===================================== #
#           Type Definition            #
#===================================== #
vec2f = ti.types.vector(2, real)
vec3f = ti.types.vector(3, real)
vec4f = ti.types.vector(4, real)
vec5f = ti.types.vector(5, real)
vec6f = ti.types.vector(6, real)
vec8f = ti.types.vector(8, real)
vec9f = ti.types.vector(9, real)
vec12f = ti.types.vector(12, real)
vec3d = ti.types.vector(3, ti.f64)
vec2i = ti.types.vector(2, int)
vec3i = ti.types.vector(3, int)
vec4i = ti.types.vector(4, int)
vec5i = ti.types.vector(5, int)
vec8i = ti.types.vector(8, int)
vec6i = ti.types.vector(6, int)
vec26i = ti.types.vector(26, int)
vec2u8 = ti.types.vector(2, ti.u8)
vec3u8 = ti.types.vector(3, ti.u8)

mat2x2 = ti.types.matrix(2, 2, real)
mat2x5 = ti.types.matrix(2, 5, real)
mat3x2 = ti.types.matrix(3, 2, real)
mat3x3 = ti.types.matrix(3, 3, real)
mat3x4 = ti.types.matrix(3, 4, real)
mat3x5 = ti.types.matrix(3, 5, real)
mat3x9 = ti.types.matrix(3, 9, real)
mat4x4 = ti.types.matrix(4, 4, real)
mat5x5 = ti.types.matrix(5, 5, real)
mat6x3 = ti.types.matrix(6, 3, real)
mat6x6 = ti.types.matrix(6, 6, real)
mat8x3 = ti.types.matrix(8, 3, real)
mat9x9 = ti.types.matrix(9, 9, real)
u1 = ti.types.quant.int(1, False)
