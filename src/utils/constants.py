from math import pi, sqrt
import numpy as np

from src.utils.TypeDefinition import (
    vec2i,
    vec2f,
    vec3f,
    vec3i,
    mat3x3,
    vec4f,
    vec5f,
    vec6f,
    vec8f,
    mat6x3,
    mat4x4,
    mat2x2,
    mat2x5,
    mat3x5,
    mat5x5,
    mat8x3,
    mat3x9,
)

BLOCK_SZ = 128
WARP_SZ = 32

IL_THRESHOLD = 1e6
L_THRESHOLD = 1e-6
THRESHOLD = 1e-14
M_THRESHOLD = 1e15
PENALTY = 1e25
DBL_DIG = 15
DBL_EPSILON = 2.2204460492503131e-16
DBL_MAX = 1.7976931348623158e308
DBL_MIN = 2.2250738585072014e-308
DBL_MAX_10_EXP = 308
DBL_MAX_EXP = 1024
DBL_MIN_10_EXP = -307
DBL_MIN_EXP = -1021
FLT_DIG = 6
FLT_EPSILON = 1.192092896e-7
FLT_MAX = 3.402823466e38
FLT_MIN = 1.175494351e-38
FLT_MAX_10_EXP = 38
FLT_MAX_EXP = 128
FLT_MIN_10_EXP = -37
FLT_MIN_EXP = -125


PI = pi
INV_PI = 1.0 / pi
SQRT_PI = sqrt(pi)
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)
SQRT3_INV = sqrt(1.0 / 3.0)
NP_IDENTITY = np.array([1.0, 1.0, 1.0])

IDENTITY = vec3f([1.0, 1.0, 1.0])
EYE = vec6f([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
EYE2D = vec3f([1.0, 1.0, 0.0])
DELTA = mat3x3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
DELTA2D = mat2x2([[1.0, 0.0], [0.0, 1.0]])

ZEROVEC2I = vec2i([0, 0])
ZEROVEC2F = vec2f([0.0, 0.0])
ZEROVEC3F = vec3f([0.0, 0.0, 0.0])
ZEROVEC4F = vec4f([0.0, 0.0, 0.0, 0.0])
ZEROVEC3I = vec3i([0, 0, 0])
ZEROVEC5F = vec5f([0.0, 0.0, 0.0, 0.0, 0.0])
ZEROVEC6F = vec6f([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ZEROVEC8F = vec8f([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

ZEROMAT2X2 = mat2x2([[0.0, 0.0], [0.0, 0.0]])
ZEROMAT2X5 = mat2x5([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
ZEROMAT3X3 = mat3x3([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
ZEROMAT3X5 = mat3x5([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
ZEROMAT3X9 = mat3x9([0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0])
ZEROMAT4X4 = mat4x4([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
ZEROMAT5X5 = mat5x5(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
ZEROMAT6X3 = mat6x3(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
)
ZEROMAT8X3 = mat8x3(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
)

# Consititutive models
FTOL = 1.0e-6  # yield function tolerance
STOL = 1.0e-3  # stress tolerance
LTOL = 1.0e-6  # detecting tolerance
MAXITS = 5  # maximum iteration number
NSUB = 10
MIN_DT = 1e-4
EPS = 1.0e-16  # machine error
ITS = 3
ITERATION_STEP = 100
SUBSTEP = 100
FTOLERANCE = 1e-5
GTOLERANCE = 1e-5

# WENO intepolation
WENO_EPS = 1e-6

# Bounding volume hierarchy
IS_LEAF = 1
PRIMITIVE_NONE = 0
PRIMITIVE_TRI = 1
PRIMITIVE_SHAPE = 2
SHAPE_NONE = 0
SHAPE_SPHERE = 1
SHAPE_QUAD = 2
SHAPE_SPOT = 3
SHAPE_LASER = 4
MAX_PRIM = 6
