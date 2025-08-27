import taichi as ti
from taichi.lang.impl import current_cfg

from src.utils.linalg import next_pow2, no_operation, make_list
from src.utils.WarpReduce import warp_shfl_up_i32
from src.utils.BitFunction import merge_i64, split_i64


WARP_SZ = 32
BLOCK_SZ = 256
@ti.data_oriented
class PrefixSumExecutor:
    """Parallel Prefix Sum (Scan) Helper

    Use this helper to perform an inclusive in-place's parallel prefix sum.

    References:
        https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
        https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu
    """
    def __init__(self, length, dtype=ti.i32, level=1):
        self.array_length = []
        self.level = level
        self.large_arr = None
        self.npad = []
        self.dtype = dtype
        self.initialize(length)

    def initialize(self, length):
        length = make_list(length)
        if current_cfg().arch == ti.cuda:
            self.run = self.runGPU
            self.ele_nums = []
            self.ele_nums_pos = []
            self.array_length = []
            for l in range(self.level):
                if length[l] > 2:
                    ele_num = length[l]
                    start_pos = 0
                    ele_nums = [ele_num]
                    ele_nums_pos = [start_pos]

                    while ele_num > 1:
                        ele_num = int((ele_num + BLOCK_SZ - 1) / BLOCK_SZ)
                        ele_nums.append(ele_num)
                        start_pos += BLOCK_SZ * ele_num
                        ele_nums_pos.append(start_pos)

                    self.ele_nums.append(ele_nums)
                    self.ele_nums_pos.append(ele_nums_pos)
                    self.array_length.append(start_pos)
                else:
                    self.array_length.append(length[l])
                    self.run = no_operation
        elif current_cfg().arch == ti.cpu:
            self.run = self.runCPU
            self.npad = []
            for l in range(self.level):
                if length[l] > 2:
                    self.large_arr = ti.field(self.dtype, shape=next_pow2(length[l] + 1))
                    self.npad.append(next_pow2(length[l] + 1))
                    self.array_length.append(length[l])
                else:
                    self.array_length.append(length[l])
                    self.run = no_operation
        else:
            raise RuntimeError(f"{str(current_cfg().arch)} is not supported for prefix sum.")

    def get_length(self, level=0):
        return self.array_length[level]

    def runGPU(self, input_arr, nlevel=0):
        ele_nums = self.ele_nums[nlevel]
        ele_nums_pos = self.ele_nums_pos[nlevel]

        # Kogge-Stone construction
        for i in range(len(ele_nums) - 1):
            if i == len(ele_nums) - 2:
                self.scan_add_inclusive(input_arr, ele_nums_pos[i], ele_nums_pos[i + 1], 1)
            else:
                self.scan_add_inclusive(input_arr, ele_nums_pos[i], ele_nums_pos[i + 1], 0)
        
        for i in range(len(ele_nums) - 3, -1, -1):
            self.uniform_add(input_arr, ele_nums_pos[i], ele_nums_pos[i + 1])

    @ti.kernel
    def scan_add_inclusive(self, arr_in: ti.template(), in_beg: int, in_end: int, single_block: int):
        ti.loop_config(block_dim=BLOCK_SZ)
        for i in range(in_beg, in_end):
            val = arr_in[i]
            val1, val2 = split_i64(val)

            thread_id = i % BLOCK_SZ
            block_id = int((i - in_beg) // BLOCK_SZ)
            lane_id = thread_id & 0x1f
            warp_id = thread_id // WARP_SZ

            pad_shared = ti.simt.block.SharedArray((65, ), self.dtype)
            if ti.static(self.dtype == ti.i64):
                val1, val2 = split_i64(val)
                val1 = warp_shfl_up_i32(lane_id, val1)
                ti.simt.block.sync()
                val2 = warp_shfl_up_i32(lane_id, val2)
                ti.simt.block.sync() 
                val = merge_i64(val1, val2)
            else:
                val = warp_shfl_up_i32(lane_id, val)
            ti.simt.block.sync()

            # Put warp scan results to smem
            # TODO replace smem with real smem when available
            if thread_id % WARP_SZ == WARP_SZ - 1:
                pad_shared[warp_id] = val
            ti.simt.block.sync()

            # Inter-warp scan, use the first thread in the first warp
            if warp_id == 0 and lane_id == 0:
                for k in range(1, int(BLOCK_SZ / WARP_SZ)):
                    pad_shared[k] += pad_shared[k - 1]
            ti.simt.block.sync()

            # Update data with warp sums
            warp_sum = ti.cast(0, self.dtype)
            if warp_id > 0:
                warp_sum = pad_shared[warp_id - 1]
            val += warp_sum
            arr_in[i] = val

            # Update partial sums except the final block
            if single_block == 0 and (thread_id == BLOCK_SZ - 1):
                arr_in[in_end + block_id] = val


    @ti.kernel
    def uniform_add(self, arr_in: ti.template(), in_beg: int, in_end: int):
        ti.loop_config(block_dim=BLOCK_SZ)
        for i in range(in_beg + BLOCK_SZ, in_end):
            block_id = int((i - in_beg) // BLOCK_SZ)
            arr_in[i] += arr_in[in_end + block_id - 1]


    @ti.kernel
    def blit_from_field_to_field(self, dst: ti.template(), src: ti.template(), offset: int, size: int):
        for i in range(offset, offset + size):
            dst[i - offset] = src[i]
        for i in range(offset + size, dst.shape[0]):
            dst[i] = 0

    def runCPU(self, input_arr, nlevel=0):
        length = input_arr.shape[0]
        npad = self.npad[nlevel]
        self.blit_from_field_to_field(self.large_arr, input_arr, 0, length)
        offset = 1
        d = npad >> 1
        while(d > 0):
            self.down_sweep(d, npad, offset, self.large_arr)
            offset <<= 1
            d >>= 1
        
        self.large_arr[npad - 1] = 0
        d = 1
        while(d < npad):
            offset >>= 1
            self.up_sweep(d, npad, offset, self.large_arr)
            d <<= 1
        self.blit_from_field_to_field(input_arr, self.large_arr, 1, length)

    @ti.kernel
    def down_sweep(self, d: int, n: int, offset: int, output: ti.template()):
        for i in range(n):
            if i < d:
                ai = offset * (2 * i + 1) - 1
                bi = offset * (2 * i + 2) - 1
                output[bi] += output[ai]


    @ti.kernel
    def up_sweep(self, d: int, n: int, offset: int, output: ti.template()):
        for i in range(n):
            if i < d:
                ai = offset * (2 * i + 1) - 1
                bi = offset * (2 * i + 2) - 1
                tmp = output[ai]
                output[ai] = output[bi]
                output[bi] += tmp


@ti.kernel
def serial(input: ti.template()):
    n = input.shape[0]
    ti.loop_config(serialize=True)
    for i in range(1, n): 
        input[i] = input[i] + input[i - 1]
    
  