import taichi as ti
from taichi.lang.impl import current_cfg

from src.utils.PrefixSum import PrefixSumExecutor


BLOCK_SZ = 64
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
        for shift_width in range(0, 32, 2):
            radix_sort_local(shift_width, input_len, self.grid_sz, self.data_in, self.data_out, self.block_sums, self.prefix_sums)
            self.pse.run(self.block_sums)
            glbl_shuffle(shift_width, input_len, self.grid_sz, self.data_in, self.data_out, self.block_sums, self.prefix_sums)


@ti.kernel
def radix_sort_local(shift_width: int, size: int, grid_sz: int, data_in: ti.template(), data_out: ti.template(), block_sums: ti.template(), prefix_sums: ti.template()):
    ti.loop_config(block_dim=BLOCK_SZ)
    for thread_id in range(size):
        s_data = ti.simt.block.SharedArray((BLOCK_SZ, ), ti.i32)
        mask_out = ti.simt.block.SharedArray((BLOCK_SZ + 1, ), ti.i32)
        merged_scan_mask_out = ti.simt.block.SharedArray((BLOCK_SZ, ), ti.i32)
        mask_out_sums = ti.simt.block.SharedArray((BLOCK_SZ, ), ti.i32)
        scan_mask_out_sums = ti.simt.block.SharedArray((4, ), ti.i32)

        thid = int(thread_id % BLOCK_SZ)
        if thread_id < size:
            s_data[thid] = data_in[thread_id]
        else:
            s_data[thid] = 0
        ti.simt.block.sync()

        t_data = s_data[thid]
        t_2bit_extract = (t_data >> shift_width) & 3

        for i in ti.static(range(4)):
            mask_out[thid] = 0
            if thid == 0:
                mask_out[BLOCK_SZ] = 0
            ti.simt.block.sync()

            val_equals_i = False
            if thread_id < size:
                val_equals_i = t_2bit_extract == i
                mask_out[thid] = val_equals_i
            ti.simt.block.sync()

            partner = 0
            sum = 0
            max_steps = int(ti.log(BLOCK_SZ * 0.5))
            for d in range(max_steps):
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
                total_sum = mask_out[BLOCK_SZ - 1]
                mask_out_sums[i] = total_sum
                block_sums[i * grid_sz + thread_id // BLOCK_SZ] = total_sum
            ti.simt.block.sync()

            if val_equals_i and (thread_id < size):
                merged_scan_mask_out[thid] = mask_out[thid]
            ti.simt.block.sync()
        
        if thid == 0:
            run_sum = 0
            for i in ti.static(range(4)):
                scan_mask_out_sums[i] = run_sum
                run_sum += mask_out_sums[i]
        ti.simt.block.sync()

        if thread_id < size:
            t_prefix_sum = merged_scan_mask_out[thid]
            new_pos = t_prefix_sum + scan_mask_out_sums[t_2bit_extract]
            ti.simt.block.sync()
            s_data[new_pos] = t_data
            merged_scan_mask_out[new_pos] = t_prefix_sum
            
            ti.simt.block.sync()
            prefix_sums[thread_id] = merged_scan_mask_out[thid]
            data_out[thread_id] = s_data[thid]
        
@ti.kernel
def glbl_shuffle(shift_width: int, size: int, grid_sz: int, data_in: ti.template(), data_out: ti.template(), block_sums: ti.template(), prefix_sums: ti.template()):
    ti.loop_config(block_dim=BLOCK_SZ)
    for thread_id in range(size):
        if thread_id < size:
            t_data = data_in[thread_id]
            t_2bit_extract = (t_data >> shift_width) & 3
            t_prefix_sum = prefix_sums[thread_id]
            data_glbl_pos = block_sums[t_2bit_extract * grid_sz + thread_id // BLOCK_SZ] + t_prefix_sum
            ti.simt.block.sync()
            data_out[data_glbl_pos] = t_data

if __name__ == "__main__":
    ti.init(arch=ti.gpu)

    sort = RadixSort(128)

    @ti.kernel
    def initialize():
        for i in range(128):
            sort.data_in[i] = ti.random(ti.i32) & 0xFF  

    initialize()
    print("Before sorting:")
    print(sort.data_in)

    sort.run(128)

    print("After sorting:")
    print(sort.data_in)