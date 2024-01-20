import taichi as ti
from decimal import Decimal, ROUND_HALF_UP


def right_round(num, keep_n):
    if isinstance(num, float):
        num = str(num)
    return Decimal(num).quantize((Decimal('0.' + '0'*keep_n)),rounding=ROUND_HALF_UP)

@ti.func
def linearize(i, j, k, vector):
    return int(i + j * vector[0] + k * vector[0] * vector[1])

@ti.func
def isnan(scalar):
    return not scalar >= 0. and not scalar < 0.

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
def sgn(x):
    if x != 0:
        x /= ti.abs(x)
    return x


@ti.func
def Zero2One(x):
    k = 1
    if x == 1:
        k = 0
    return k


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


def NonNegative(x):
    if x <= 0:
        return 1
    else:
        return int(x)

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

def next_pow2(x):
    x -= 1
    x |= (x >> 1)
    x |= (x >> 2)
    x |= (x >> 4)
    x |= (x >> 8)
    x |= (x >> 16)
    return x + 1

def round32(n):
    if(n % 32 == 0): return n
    else: return ((n >> 5) + 1) << 5

@ti.func
def morton3d32(x, y, z):
    md = 0
    x &= 0x3ff
    x = (x | x << 16) & 0x30000ff
    x = (x | x << 8) & 0x300f00f
    x = (x | x << 4) & 0x30c30c3
    x = (x | x << 2) & 0x9249249
    y &= 0x3ff
    y = (y | y << 16) & 0x30000ff
    y = (y | y << 8) & 0x300f00f
    y = (y | y << 4) & 0x30c30c3
    y = (y | y << 2) & 0x9249249
    z &= 0x3ff
    z = (z | z << 16) & 0x30000ff
    z = (z | z << 8) & 0x300f00f
    z = (z | z << 4) & 0x30c30c3
    z = (z | z << 2) & 0x9249249
    md |= x | y << 1 | z << 2
    return md


@ti.func
def demorton3d32(md):
    x = md &        0x09249249
    y = (md >> 1) & 0x09249249
    z = (md >> 2) & 0x09249249
    
    x = ((x >> 2) | x) & 0x030C30C3
    x = ((x >> 4) | x) & 0x0300F00F
    x = ((x >> 8) | x) & 0x030000FF
    x = ((x >>16) | x) & 0x000003FF
    
    y = ((y >> 2) | y) & 0x030C30C3
    y = ((y >> 4) | y) & 0x0300F00F
    y = ((y >> 8) | y) & 0x030000FF
    y = ((y >>16) | y) & 0x000003FF
    
    z = ((z >> 2) | z) & 0x030C30C3
    z = ((z >> 4) | z) & 0x0300F00F
    z = ((z >> 8) | z) & 0x030000FF
    z = ((z >>16) | z) & 0x000003FF
    return x, y, z


@ti.func
def vectorize_id(index, countVec):
    ig = (index % (countVec[0] * countVec[1])) % countVec[0]
    jg = (index % (countVec[0] * countVec[1])) // countVec[0]
    kg = index // (countVec[0] * countVec[1])
    return ig, jg, kg 