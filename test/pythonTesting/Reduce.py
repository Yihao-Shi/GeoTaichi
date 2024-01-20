import taichi as ti
ti.init(arch=ti.gpu)

WARP_SZ = 32
BLOCK_SZ = 256

a = ti.field(float, 256)

@ti.kernel
def fill(var: ti.template()):
    for i in var:
        var[i] = 1
fill(a)

@ti.kernel
def reduce_add0(var: ti.template()) -> float:
    sum = 0.
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in var:
        pad_shared = ti.simt.block.SharedArray((BLOCK_SZ, ), ti.f32)

        thread_id = i % BLOCK_SZ
        block_id = int(i // BLOCK_SZ)
        lane_id = thread_id % WARP_SZ
        warp_id = thread_id // WARP_SZ
        
        pad_shared[thread_id] = var[i]
        ti.simt.block.sync()

        s = 1
        while s < BLOCK_SZ:
            if thread_id % (2 * s) == 0:
                pad_shared[thread_id] += pad_shared[thread_id + s]
            ti.simt.block.sync()
            s *= 2

        if thread_id == 0:
            sum += pad_shared[thread_id]
    return sum

@ti.kernel
def reduce_add1(var: ti.template()) -> float:
    sum = 0.
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in var:
        pad_shared = ti.simt.block.SharedArray((BLOCK_SZ, ), ti.f32)

        thread_id = i % BLOCK_SZ
        block_id = int(i // BLOCK_SZ)
        lane_id = thread_id % WARP_SZ
        warp_id = thread_id // WARP_SZ
        
        pad_shared[thread_id] = var[i]
        ti.simt.block.sync()

        s = 1
        while s < BLOCK_SZ:
            index = 2 * s * thread_id
            if index < BLOCK_SZ:
                pad_shared[index] += pad_shared[index + s]
            ti.simt.block.sync()
            s *= 2

        if thread_id == 0:
            sum += pad_shared[thread_id]
    return sum

@ti.kernel
def reduce_add2(var: ti.template()) -> float:
    sum = 0.
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in var:
        pad_shared = ti.simt.block.SharedArray((BLOCK_SZ, ), ti.f32)

        thread_id = i % BLOCK_SZ
        block_id = int(i // BLOCK_SZ)
        lane_id = thread_id % WARP_SZ
        warp_id = thread_id // WARP_SZ
        
        pad_shared[thread_id] = var[i]
        ti.simt.block.sync()

        s = int(0.5 * BLOCK_SZ)
        while s > 0:
            if thread_id < s:
                pad_shared[thread_id] += pad_shared[thread_id + s]
            ti.simt.block.sync()
            s >>=1

        if thread_id == 0:
            sum += pad_shared[thread_id]
    return sum

@ti.kernel
def reduce_add3(var: ti.template()) -> float:
    sum = 0.
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in var:
        pad_shared = ti.simt.block.SharedArray((BLOCK_SZ, ), ti.f32)

        thread_id = i % BLOCK_SZ
        block_id = int(i // BLOCK_SZ)
        lane_id = thread_id % WARP_SZ
        warp_id = thread_id // WARP_SZ
        
        pad_shared[thread_id] = var[i]
        ti.simt.block.sync()

        s = int(0.5 * BLOCK_SZ)
        while s > 0:
            if thread_id < s:
                pad_shared[thread_id] += pad_shared[thread_id + s]
            ti.simt.block.sync()
            s >>=1

        if thread_id == 0:
            sum += pad_shared[thread_id]
    return sum

@ti.kernel
def atomic_add(var: ti.template()) -> float:
    sum = 0.
    for i in var:
        sum += var[i]
    return sum

import time 

start = time.time()
sum = reduce_add0(a)
end = time.time()
print(end-start, sum)

start = time.time()
sum = reduce_add1(a)
end = time.time()
print(end-start, sum)

start = time.time()
sum = reduce_add2(a)
end = time.time()
print(end-start, sum)

start = time.time()
sum = atomic_add(a)
end = time.time()
print(end-start, sum)