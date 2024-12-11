import taichi as ti
from copy import deepcopy

from src.utils.constants import WARP_SZ, BLOCK_SZ


class ReduceSum(object):
    def __init__(self, max_size) -> None:
        self.num_per_block = max_size // deepcopy(BLOCK_SZ)
        self.output = ti.field(float, shape=max_size)

    def runCPU(self, size, input):
        return atomic_add(size, input)

    def runGPU(self, size, input):
        remaining = size
        while remaining > 1:
            reduce_add(remaining, input, self.output)
            remaining = (remaining - 1) // (2 * BLOCK_SZ) + 1
            if remaining > 1:
                copy(remaining, input, self.output)
        return self.output[0]

@ti.kernel
def atomic_add(size: int, input: ti.template()) -> float:
    sum = 0.
    for i in range(size):
        sum += input[i]
    return sum

@ti.kernel
def copy(size: int, input: ti.template(), output: ti.template()):
    for i in range(size):
        input[i] = output[i]

@ti.kernel
def reduce_add(size: int, input: ti.template(), output: ti.template()):
    block_size = int((0.5 * size - 1) // BLOCK_SZ + 1)
    block_dim = min(BLOCK_SZ, ti.ceil(0.5 * size, int))
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in range(int(block_size * block_dim)):
        thread_id = i % BLOCK_SZ
        block_id = i // BLOCK_SZ
        pad_shared = ti.simt.block.SharedArray((BLOCK_SZ, ), ti.f64)
        index = block_id * (2 * BLOCK_SZ) + thread_id
        pad_shared[thread_id] = input[index] + input[index + block_dim]
        ti.simt.block.sync()

        if block_dim >= 512:
            if thread_id < 256:
                pad_shared[thread_id] += pad_shared[thread_id+256]
            ti.simt.block.sync()
        if block_dim >= 256:
            if thread_id < 128:
                pad_shared[thread_id] += pad_shared[thread_id+128]
            ti.simt.block.sync()
        if block_dim >= 128:
            if thread_id < 64:
                pad_shared[thread_id] += pad_shared[thread_id+64]
            ti.simt.block.sync()

        if thread_id < 32: 
            if block_dim >= 64: pad_shared[thread_id] += pad_shared[thread_id + 32]
            if block_dim >= 32: pad_shared[thread_id] += pad_shared[thread_id + 16]
            if block_dim >= 16: pad_shared[thread_id] += pad_shared[thread_id + 8]
            if block_dim >= 8: pad_shared[thread_id] += pad_shared[thread_id + 4]
            if block_dim >= 4: pad_shared[thread_id] += pad_shared[thread_id + 2]
            if block_dim >= 2: pad_shared[thread_id] += pad_shared[thread_id + 1]
        if thread_id == 0:
            output[block_id] = pad_shared[thread_id]