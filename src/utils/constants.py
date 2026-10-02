from math import pi, sqrt
import numpy as np

from src.utils.TypeDefination import (vec2i, vec2f, vec3f, vec3i, mat3x3, vec4f, vec5f, vec6f, vec8f,
                                      mat6x3, mat4x4, mat2x2, mat2x5, mat3x5, mat5x5, mat6x6, mat8x3, mat3x9, mat9x9)

BLOCK_SZ = 128
WARP_SZ = 32

ILThreshold = 1e6
LThreshold = 1e-6
Threshold = 1e-14
MThreshold = 1e15
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
iPI = 1. / pi
SQRT_PI = sqrt(pi)
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)
SQRT3_INV = sqrt(1./3.)
NP_IDENTITY = np.array([1., 1., 1.])

IDENTITY = vec3f([1., 1., 1.])
EYE = vec6f([1., 1., 1., 0., 0., 0.])
EYE2D = vec3f([1., 1., 0.])
DELTA = mat3x3([[1., 0., 0.], [0., 1., 0.],[0., 0., 1.]])
DELTA2D = mat2x2([[1., 0.], [0., 1.]])

ZEROVEC2i = vec2i([0, 0])
ZEROVEC2f = vec2f([0., 0.])
ZEROVEC3f = vec3f([0., 0., 0.])
ZEROVEC4f = vec4f([0., 0., 0., 0.])
ZEROVEC3i = vec3i([0, 0, 0])
ZEROVEC5f = vec5f([0., 0., 0., 0., 0.])
ZEROVEC6f = vec6f([0., 0., 0., 0., 0., 0.])
ZEROVEC8f = vec8f([0., 0., 0., 0., 0., 0., 0., 0.])

ZEROMAT2x2 = mat2x2([[0., 0.], 
                     [0., 0.]])
ZEROMAT2x5 = mat2x5([[0., 0., 0., 0., 0.], 
                     [0., 0., 0., 0., 0.]])
ZEROMAT3x3 = mat3x3([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
ZEROMAT3x5 = mat3x5([[0., 0., 0., 0., 0.], 
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.]])
ZEROMAT3x9 = mat3x9([0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0])
ZEROMAT4x4 = mat4x4([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
ZEROMAT5x5 = mat5x5([[0., 0., 0., 0., 0.], 
                     [0., 0., 0., 0., 0.], 
                     [0., 0., 0., 0., 0.], 
                     [0., 0., 0., 0., 0.], 
                     [0., 0., 0., 0., 0.]])
ZEROMAT6x6 = mat6x6([[0., 0., 0., 0., 0., 0.], 
                     [0., 0., 0., 0., 0., 0.], 
                     [0., 0., 0., 0., 0., 0.], 
                     [0., 0., 0., 0., 0., 0.], 
                     [0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.]])
ZEROMAT6x3 = mat6x3([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], 
                     [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
ZEROMAT8x3 = mat8x3([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], 
                     [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
ZEROMAT9x9 = mat9x9([0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0])

# Consititutive models
FTOL = 1.e-6        # yield function tolerance
STOL = 1.e-3        # stress tolerance
LTOL = 1.e-6        # detecting tolerance
MAXITS = 3         # maximum iteration number
NSUB = 10
dTmin = 1e-4
EPS = 1.e-16        # machine error
ITS = 3 
itrstep = 100
substep = 100
Ftolerance = 1e-5
Gtolerance = 1e-5

# WENO intepolation
WENO_EPS = 1e-6

# Bounding volume hierarchy
IS_LEAF         = 1
PRIMITIVE_NONE  = 0
PRIMITIVE_TRI   = 1
PRIMITIVE_SHAPE = 2
SHPAE_NONE      = 0
SHPAE_SPHERE    = 1
SHPAE_QUAD      = 2
SHPAE_SPOT      = 3
SHPAE_LASER     = 4
MAX_PRIM        = 6