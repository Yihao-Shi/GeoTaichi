import sys
sys.path.append('/home/eleven/work/GeoTaichi')

import taichi as ti
ti.init(arch=ti.gpu, debug=False, default_fp=ti.f64)

from taichi.lang.impl import current_cfg
from src.utils.ScalarFunction import next_pow2

class PrefixSumExecutor:
    """Parallel Prefix Sum (Scan) Helper

    Use this helper to perform an inclusive in-place's parallel prefix sum.

    References:
        https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
        https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu
    """
    def __init__(self, length):
        self.sorting_length = length
        BLOCK_SZ = 64
        ele_num = length
        self.ele_nums = [ele_num]
        start_pos = 0
        self.ele_nums_pos = [start_pos]

        if length > 3:
            if current_cfg().arch == ti.cuda:
                while ele_num > 1:
                    ele_num = int((ele_num + BLOCK_SZ - 1) / BLOCK_SZ)
                    self.ele_nums.append(ele_num)
                    start_pos += BLOCK_SZ * ele_num
                    self.ele_nums_pos.append(start_pos)

                self.result = ti.field(float, shape=len(self.ele_nums))
                self.large_arr = ti.field(int, shape=start_pos)
                self.run = self.runGPU
            elif current_cfg().arch == ti.cpu:
                self.large_arr = ti.field(int, shape=next_pow2(length))
                self.run = self.runCPU
            else:
                raise RuntimeError(f"{str(current_cfg().arch)} is not supported for prefix sum.")
        else:
            self.run = self.no_operation

    def no_operation(self, input_arr, length):
        pass

    def runGPU(self, input_arr, length):
        # length = input_arr.shape[0]
        ele_nums = self.ele_nums
        ele_nums_pos = self.ele_nums_pos

        blit_from_field_to_field(self.large_arr, input_arr, 0, length)

        # Kogge-Stone construction
        for i in range(len(ele_nums) - 1):
            scan_add_inclusive(i, self.large_arr, ele_nums_pos[i], ele_nums_pos[i + 1], self.result)
        
        #for i in range(len(ele_nums) - 3, -1, -1):
        #    uniform_add(self.result, ele_nums_pos[i], ele_nums_pos[i + 1])


    def runCPU(self, input_arr, length):
        npad = self.large_arr.shape[0]
        blit_from_field_to_field(self.large_arr, input_arr, 0, length)

        offset = 1
        d = npad >> 1
        while(d > 0):
            down_sweep(d, npad, offset, self.large_arr)
            offset <<= 1
            d >>= 1
        
        self.large_arr[npad - 1] = 0
        d = 1
        while(d < npad):
            offset >>= 1
            up_sweep(d, npad, offset, self.large_arr)
            d <<= 1
        blit_from_field_to_field(input_arr, self.large_arr, 1, length)

@ti.func
def warp_shfl_up_i32(val):
    global_tid = ti.simt.block.global_thread_idx()
    WARP_SZ = 32
    lane_id = global_tid % WARP_SZ
    # Intra-warp scan, manually unrolled
    offset_j = 1
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 2
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 4
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 8
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 16
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    return val


@ti.kernel
def scan_add_inclusive(loc: int, arr_in: ti.template(), in_beg: int, in_end: int, result: ti.template()):
    WARP_SZ = 32
    BLOCK_SZ = 64
    ti.loop_config(block_dim=64)
    for i in range(in_beg, in_end):
        val = arr_in[i]

        thread_id = i % BLOCK_SZ
        lane_id = thread_id % WARP_SZ
        warp_id = thread_id // WARP_SZ

        pad_shared = ti.simt.block.SharedArray((65, ), ti.f64)

        val = warp_shfl_up_i32(val)
        ti.simt.block.sync()

        # Put warp scan results to smem
        # TODO replace smem with real smem when available
        if thread_id % WARP_SZ == WARP_SZ - 1:
            pad_shared[warp_id] = val
        ti.simt.block.sync()
        
        # Update data with warp sums
        if warp_id > 0 and thread_id == BLOCK_SZ - 1:
            result[0] += val + pad_shared[warp_id - 1]


@ti.kernel
def uniform_add(result: ti.template(), in_beg: int, in_end: int):
    BLOCK_SZ = 64
    ti.loop_config(block_dim=64)
    for i in range(1, int((in_end - in_beg) // BLOCK_SZ)):
        result[i] += result[i + 1]


@ti.kernel
def blit_from_field_to_field(dst: ti.template(), src: ti.template(), offset: int, size: int):
    for i in range(offset, offset + size):
        dst[i - offset] = src[i]
    for i in range(offset + size, dst.shape[0]):
        dst[i] = 0

    
@ti.kernel
def serial(output:ti.template(), input:ti.template()):
    n = input.shape[0]
    output[0] = 0
    ti.loop_config(serialize=True)
    for i in range(1, n): 
        output[i] = output[i - 1] + input[i - 1]


@ti.kernel
def down_sweep(d: int, n: int, offset: int, output: ti.template()):
    for i in range(n):
        if i < d:
            ai = offset * (2 * i + 1) - 1
            bi = offset * (2 * i + 2) - 1
            output[bi] += output[ai]


@ti.kernel
def up_sweep(d: int, n: int, offset: int, output: ti.template()):
    for i in range(n):
        if i < d:
            ai = offset * (2 * i + 1) - 1
            bi = offset * (2 * i + 2) - 1
            tmp = output[ai]
            output[ai] = output[bi]
            output[bi] += tmp



a = ti.field(int, shape=1028000)

@ti.kernel
def k(a: ti.template()):
    for i in range(0,1028000):
        a[i] = i
    #a[125] = 19625
k(a)        
pse = PrefixSumExecutor(1028000)

import time
x = time.time()
pse.run(a, 1028000)
y = time.time()
print(y - x, pse.result[0], pse.result[1])

@ti.kernel
def run()->float:
    a = 0.
    for i in range(1028000): a+=i
    return a

x = time.time()
run()
y = time.time()
print(y - x)    

'''@ti.kernel
def reduce_sum(input: ti.template()):
    ti.loop_config(block_dim=64)
    for i in range(input.shape[0]):
        gtid = ti.simt.block.global_thread_idx()
        shm = ti.simt.block.SharedArray(64, ti.f64)

        ti.simt.block.sync()


@ti.kernel
def reduce_max(input: ti.template()):
    ti.loop_config(block_dim=64)
    for i in range(input.shape[0]):
        gtid = ti.simt.block.global_thread_idx()
        shm = ti.simt.block.SharedArray((64, ), float)

        ti.simt.block.sync()


@ti.kernel
def reduce_min(input: ti.template()):
    ti.loop_config(block_dim=64)
    for i in range(input.shape[0]):
        gtid = ti.simt.block.global_thread_idx()
        shm = ti.simt.block.SharedArray((64, ), float)

        ti.simt.block.sync()

input = ti.field(int, 3)
reduce_sum(input)'''