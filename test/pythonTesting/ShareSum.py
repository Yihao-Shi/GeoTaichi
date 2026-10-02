import taichi as ti
ti.init(arch=ti.cuda, debug=True, default_fp=ti.f64)

@ti.kernel
def reduce1(size: int, p: ti.template(), q: ti.template()) -> float:
    result = float(0.0)
    BLOCK_SZ = 64
    ti.loop_config(block_dim=64)
    for i in range(size):
        thread_id = i % BLOCK_SZ
        pad_vector1 = ti.simt.block.SharedArray((64, ), ti.f64)
        pad_vector2 = ti.simt.block.SharedArray((64, ), ti.f64)

        pad_vector1[thread_id] = p[i]
        pad_vector2[thread_id] = q[i]
        ti.simt.block.sync()

        temp = 0.    
        if thread_id == BLOCK_SZ - 1 or i == size - 1:
            for k in range(thread_id + 1):
                temp += pad_vector1[k] * pad_vector2[k]
        ti.simt.block.sync()    
        result += temp
    return result

@ti.kernel
def reduce2(size: int, p: ti.template(), q: ti.template()) -> float:
    result = 0.
    for i in range(size):
        result += p[i] * q[i]
    return result

shape=105500
a = ti.field(float, shape=shape)
b = ti.field(float, shape=shape)
a.fill(2)
b.fill(2)

print(reduce1(shape, a, b))
print(reduce2(shape, a, b))

import time
s=time.time()
print(reduce1(shape, a, b))
e=time.time()
print(e-s)

s=time.time()
print(reduce2(shape, a, b))
e=time.time()
print(e-s)