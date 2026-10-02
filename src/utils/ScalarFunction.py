import taichi as ti

from src.utils.constants import Threshold



@ti.func
def equal_to(i, value):
    return ti.abs(i - value) < Threshold

@ti.func
def linearize(index, vector):
    assert index.n == vector.n
    if ti.static(vector.n == 2):
        return int(index[0] + index[1] * vector[0])
    elif ti.static(vector.n == 3):
        return int(index[0] + index[1] * vector[0] + index[2] * vector[0] * vector[1])
    
@ti.func
def linearize2D(i, j, vector):
    return int(i + j * vector[0])
    
@ti.func
def linearize3D(i, j, k, vector):
    return int(i + j * vector[0] + k * vector[0] * vector[1])

@ti.func
def vectorize_id(index, countVec):
    if ti.static(countVec.n == 2):
        ig = index % countVec[0]
        jg = index // countVec[0]
        return ig, jg
    elif ti.static(countVec.n == 3):
        ig = (index % (countVec[0] * countVec[1])) % countVec[0]
        jg = (index % (countVec[0] * countVec[1])) // countVec[0]
        kg = index // (countVec[0] * countVec[1])
        return ig, jg, kg 

@ti.func
def isnan(scalar):
    return not (scalar >= 0 or scalar <= 0)

@ti.func
def isinf(scalar):
    return 2 * scalar == scalar and scalar != 0

@ti.func
def swap(a, b):
    return b, a

@ti.func
def copysign(x, y):
    return ti.abs(x) * sgn(y)

@ti.pyfunc
def sgn(x):
    return ti.cast((x >= 0.0), float) - ti.cast((x <= 0.0), float)

@ti.func
def Max(i, j):
    m = j
    if i > j:
        m = i
    return m

@ti.func
def Min(i, j):
    m = j
    if i < j:
        m = i
    return m

@ti.func
def EffectiveValue(x, y):
    return x * y / (x + y)


@ti.func
def xor(a, b):
    return (a + b) & 1


@ti.pyfunc
def PairingFunction(i, j):
    a = ti.min(i, j)
    b = ti.max(i, j)
    return int(0.5 * (a + b) * (a + b + 1) + b)


@ti.pyfunc
def DePairingFunction(pair_id):
    w = int(0.5 * (ti.sqrt(8 * pair_id + 1) - 1))
    t = 0.5 * w * (w + 1)
    j = ti.cast(pair_id - t, ti.i32)
    i = ti.cast(w - j, ti.i32)
    return i, j


@ti.pyfunc
def encode_pair(i, j):
    a = ti.cast(ti.min(i, j), ti.i64)
    b = ti.cast(ti.max(i, j), ti.i64)
    return (a << 32) | b


@ti.pyfunc
def decode_pair(pair_id):
    i = ti.cast(pair_id >> 32, ti.i32)
    j = ti.cast(pair_id & 0xffffffff, ti.i32)
    return i, j


@ti.func
def PairingMapping(i, j, length):
    return int(i * length + j)


@ti.func
def clamp(min_val, max_val, val):
    return ti.min(ti.max(min_val, val), max_val)


@ti.func
def macauley(value):
    return value if value > 0. else 0.


@ti.func
def macauley_index(value):
    return 1. if value > 0. else 0.


@ti.func
def BinarySearch(begining, ending, key, KEY):
    loc = -1
    while begining <= ending:
        mid_point = int((begining + ending) / 2)
        if KEY[mid_point] == key:
            loc = mid_point
            break
        elif KEY[mid_point] > key:
            ending = mid_point - 1
        elif KEY[mid_point] < key:
            begining = mid_point + 1
    return loc


@ti.func
def biInterpolate(pt, xExtr, yExtr, knownVal):
    x0 = xExtr[0]
    y0 = yExtr[0]
    gx = xExtr[1] - x0
    gy = yExtr[1] - y0
    f00 = knownVal[0, 0] 
    f01 = knownVal[0, 1] 
    f10 = knownVal[1, 0] 
    f11 = knownVal[1, 1]
    bracket = (pt[1] - y0) / gy * (f11 - f10 - f01 + f00) + f10 - f00
    return (pt[0] - x0) / gx * bracket + (pt[1] - y0) / gy * (f01 - f00) + f00
