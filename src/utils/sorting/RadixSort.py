import taichi as ti
from taichi.lang.impl import current_cfg

from src.utils.PrefixSum import PrefixSumExecutor


# Reference: Ha, L., Krüger, J., & Silva, C. T. (2009, December). Fast Four‐Way Parallel Radix Sorting on GPUs. In Computer Graphics Forum (Vol. 28, No. 8, pp. 2368-2378). Oxford, UK: Blackwell Publishing Ltd.
BLOCK_SZ = 256
class RadixSort(object):
    def __init__(self, input_len) -> None:
        self.data_in = ti.field(dtype=ti.i32, shape=input_len)
        self.data_out = ti.field(dtype=ti.i32, shape=input_len)

        if current_cfg().arch == ti.cuda:
            self.grid_sz = int(input_len / BLOCK_SZ) if input_len % BLOCK_SZ == 0 else int(input_len / BLOCK_SZ) + 1
            self.pse = PrefixSumExecutor(4 * self.grid_sz)
            self.prefix_sums = ti.field(int, shape=input_len)
            self.block_sums = ti.field(int, shape=self.pse.get_length())
            self.run = self.radix_sort_gpu
        elif current_cfg().arch == ti.cpu:
            pass
        else:
            raise RuntimeError(f"{str(current_cfg().arch)} is not supported for radix sort.")

    def radix_sort_gpu(self, input_len):
        for shift_width in range(0, 31, 2):
            self.block_sums.fill(0)
            radix_sort_local(shift_width, input_len, self.grid_sz, self.data_in, self.data_out, self.block_sums, self.prefix_sums)
            self.pse.run(self.block_sums)
            glbl_shuffle(shift_width, input_len, self.grid_sz, self.data_out, self.data_in, self.block_sums, self.prefix_sums)


@ti.kernel
def radix_sort_local(shift_width: int, size: int, grid_sz: int, data_in: ti.template(), data_out: ti.template(), block_sums: ti.template(), prefix_sums: ti.template()):
    ti.loop_config(block_dim=BLOCK_SZ)
    for cpy_idx in range(size):
        s_data = ti.simt.block.SharedArray((BLOCK_SZ, ), ti.i32)
        mask_out = ti.simt.block.SharedArray((BLOCK_SZ + 1, ), ti.i32)
        merged_scan_mask_out = ti.simt.block.SharedArray((BLOCK_SZ, ), ti.i32)
        mask_out_sums = ti.simt.block.SharedArray((4, ), ti.i32)
        scan_mask_out_sums = ti.simt.block.SharedArray((4, ), ti.i32)

        thid = int(cpy_idx % BLOCK_SZ)
        blid = int(cpy_idx // BLOCK_SZ)
        t_data = data_in[cpy_idx]
        t_2bit_extract = (t_data >> shift_width) & 3

        for i in ti.static(range(4)):
            mask_out[thid] = 0
            if thid == 0:
                mask_out[ti.min(size, BLOCK_SZ)] = 0
            ti.simt.block.sync()

            # generate bitmask
            val_equals_i = t_2bit_extract == i
            mask_out[thid] = val_equals_i
            ti.simt.block.sync()

            # local prefix sum
            sum = 0
            for d in ti.static(range(int(ti.log(BLOCK_SZ) / ti.log(2)))):
                partner = thid - (1 << d)
                if partner >= 0:
                    sum = mask_out[thid] + mask_out[partner]
                else:
                    sum = mask_out[thid]
                ti.simt.block.sync()
                mask_out[thid] = sum
                ti.simt.block.sync()

            cpy_val = mask_out[thid]
            ti.simt.block.sync()
            mask_out[thid + 1] = cpy_val
            ti.simt.block.sync()

            if thid == 0:
                mask_out[0] = 0
                total_sum = mask_out[ti.min(size, BLOCK_SZ)]
                mask_out_sums[i] = total_sum
                block_sums[i * grid_sz + blid] = total_sum
            ti.simt.block.sync()

            if val_equals_i:
                merged_scan_mask_out[thid] = mask_out[thid]
            ti.simt.block.sync()
            
        if thid == 0:
            run_sum = 0
            for i in ti.static(range(4)):
                scan_mask_out_sums[i] = run_sum
                run_sum += mask_out_sums[i]
        ti.simt.block.sync()

        t_prefix_sum = merged_scan_mask_out[thid]
        new_pos = t_prefix_sum + scan_mask_out_sums[t_2bit_extract]
        ti.simt.block.sync()

        s_data[new_pos] = t_data
        merged_scan_mask_out[new_pos] = t_prefix_sum
        ti.simt.block.sync()
        
        prefix_sums[cpy_idx] = merged_scan_mask_out[thid]
        data_out[cpy_idx] = s_data[thid]
        ti.simt.block.sync()
        
@ti.kernel
def glbl_shuffle(shift_width: int, size: int, grid_sz: int, data_in: ti.template(), data_out: ti.template(), block_sums: ti.template(), prefix_sums: ti.template()):
    ti.loop_config(block_dim=BLOCK_SZ)
    for thread_id in range(size):
        t_data = data_in[thread_id]
        t_2bit_extract = (t_data >> shift_width) & 3
        t_prefix_sum = prefix_sums[thread_id]
        index = t_2bit_extract * grid_sz + thread_id // BLOCK_SZ
        offset = block_sums[index - 1] if index > 0 else 0
        data_glbl_pos = offset + t_prefix_sum
        ti.simt.block.sync()
        data_out[data_glbl_pos] = t_data
