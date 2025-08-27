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
def expandBits(v):
    """
    Expands a 10-bit integer into 30 bits by inserting 2 zeros before each bit.
    """
    v = (v * ti.u32(0x00010001)) & ti.u32(0xFF0000FF)
    v = (v * ti.u32(0x00000101)) & ti.u32(0x0F00F00F)
    v = (v * ti.u32(0x00000011)) & ti.u32(0xC30C30C3)
    v = (v * ti.u32(0x00000005)) & ti.u32(0x49249249)
    return v


@ti.func
def morton3d32(x, y, z):
    x = ti.floor(min(max(x * 1023.0, 0.0), 1023.0))
    y = ti.floor(min(max(y * 1023.0, 0.0), 1023.0))
    z = ti.floor(min(max(z * 1023.0, 0.0), 1023.0))
    morton_code_x = expandBits(ti.u32(x))
    morton_code_y = expandBits(ti.u32(y))
    morton_code_z = expandBits(ti.u32(z))
    morton_code = (morton_code_x << 2) | (morton_code_y << 1) | (morton_code_z)
    return morton_code


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
def brev(v):
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1)
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2)
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4)
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8)
    v = (v >> 16) | (v << 16)
    return v


@ti.func
def brevll(x):
    x = ((x >> 1)  & 0x5555555555555555) | ((x & 0x5555555555555555) << 1)
    x = ((x >> 2)  & 0x3333333333333333) | ((x & 0x3333333333333333) << 2)
    x = ((x >> 4)  & 0x0F0F0F0F0F0F0F0F) | ((x & 0x0F0F0F0F0F0F0F0F) << 4)
    x = ((x >> 8)  & 0x00FF00FF00FF00FF) | ((x & 0x00FF00FF00FF00FF) << 8)
    x = ((x >> 16) & 0x0000FFFF0000FFFF) | ((x & 0x0000FFFF0000FFFF) << 16)
    x = (x >> 32) | (x << 32)
    return x


@ti.func
def clz(x):
    result = 32
    for b in range(32):
        if x & (ti.u32(1) << (31 - b)):
            result = b
            break
    return result


@ti.func
def clzll(x: ti.u64) -> ti.i32:
    result = 64
    for b in range(64):
        if x & (ti.u64(1) << (63 - b)):
            result = b
            break
    return result


@ti.func
def invert_bit_i32(x):
    return ~x & ((1 << 32) - 1)


@ti.func
def ballot(x):
    return ti.simt.warp.ballot(x)


@ti.func
def merge_i64(first_i32, second_i32):
    high = ti.cast(first_i32, ti.i64) << 32
    low = ti.cast(ti.cast(second_i32, ti.i64) & ti.i64(0xFFFFFFFF), ti.i64)
    return high | low


@ti.func
def split_i64(value_i64):
    a = ti.cast(value_i64 >> 32, ti.i32)
    b = ti.cast(value_i64 & ti.i64(0xFFFFFFFF), ti.i32)
    return ti.Vector([a, b])


@ti.func
def split_f64(value):
    bits = ti.bit_cast(value, ti.i64)
    high_bits = (bits >> 32) & ti.u64(0xFFFFFFFF)
    low_bits = bits & ti.u64(0xFFFFFFFF)
    high_f32 = ti.bit_cast(ti.i32(high_bits), ti.f32)  
    low_f32 = ti.bit_cast(ti.i32(low_bits), ti.f32)
    return high_f32, low_f32   


@ti.func
def merge_f64(high_f32, low_f32):
    high_bits = ti.bit_cast(high_f32, ti.i32)      
    low_bits = ti.bit_cast(low_f32, ti.i32)        
    combined_bits = (ti.i64(high_bits) << 32) | ti.i64(low_bits)  
    return ti.bit_cast(combined_bits, ti.f64)     


@ti.func
def split_f64_precision(value):
    f32_high = ti.cast(value, ti.f32)
    f32_low = value - ti.cast(f32_high, ti.f64)
    return ti.cast(f32_high, ti.f32), ti.cast(f32_low, ti.f32)


@ti.func
def merge_f64_precision(high_f32, low_f32):
    return ti.cast(high_f32 + low_f32, ti.f32)


@ti.func
def atomic_cas(input: int, compare: int, values: int):
    original = input
    if ti.atomic_add(input, 0) == compare: 
        input = values 
    return original
