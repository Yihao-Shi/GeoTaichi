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
def sign(x):
    if x >= 0:
        x = 1
    else: x = -1
    return x

@ti.func
def copysign(x, y):
    return ti.abs(x) * sgn(y)

@ti.func
def sgn(x):
    return ti.select(x >= 0., 1, 0) - ti.select(x <= 0., 1, 0)

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


@ti.func
def PairingFunction(i, j):
    return int(0.5 * (i + j) * (i + j + 1) + j)


@ti.func
def PairingMapping(i, j, length):
    return int(i * length + j)


@ti.func
def clamp(min_val, max_val, val):
    return ti.min(ti.max(min_val, val), max_val)


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
