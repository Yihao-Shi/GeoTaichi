import taichi as ti
ti.init()

contact_pairs = 10000000

@ti.dataclass
class ContactPairPP:
    endID1: int
    endID2: int
    isActive: int
    gapn: float
    norm: vec3f
    cnforce: vec3f
    csforce: vec3f
    oldTangOverlap: vec3f
    
    
contact = ContactPairPP(shape=contact_pairs)

@ti.func
def get_bit(bs, idx)
    return (bs[idx >> 5 + 1] & (ti.i32(1) << (idx & 31))) != 0

@ti.func
def set_bit!(bs, idx)
    CUDA.atomic_or!(pointer(bs, idx >> 5 + 1), UInt32(1) << (idx & 31))

@ti.func
def clear_bit!(bs, idx)
    CUDA.atomic_and!(pointer(bs, idx >> 5 + 1), ~(UInt32(1) << (idx & 31)))
