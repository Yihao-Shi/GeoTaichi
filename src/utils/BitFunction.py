import taichi as ti



@ti.func
def vector_bit_or(vec1, vec2):
    for i in ti.static(vec1.n):
        vec2[i] = vec1[i] ^ vec2[i]
    return vec2


@ti.func
def Zero2One(x):
    return int(x) ^ 1


@ti.func
def Zero2OneVector(x):
    for i in ti.static(range(x.n)):
        x[i] = int(x[i]) ^ 1
    return x


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
def bit_reverse_u32(x):
    x = ti.u32(x)
    reversed_bits = ti.u32(0)
    for i in ti.static(range(32)):
        if x & (1 << i) != 0.:
            reversed_bits |= 1 << (31 - i)
    return reversed_bits


@ti.func
def invert_bit_i32(x):
    return ~x & ((1 << 32) - 1)


@ti.func
def counting_leading_zeros_u32(x):
    x = ti.u32(x)
    count = 0
    for i in range(32):
        if x & (1 << (31 - i)) == 0:
            count += 1
        else: break
    return count


@ti.func
def ballot_32(cond):
    mask = ti.u32(0)
    for i in range(32):
        if cond[i] != 0:
            mask |= 1 << i
    return mask


@ti.func
def ballot_64(cond):
    mask = ti.u32(0)
    for i in range(64):
        if cond[i] != 0:
            mask |= 1 << i
    return mask
