import taichi as ti
from taichi.lang.impl import current_cfg

from src.utils.PrefixSum import PrefixSumExecutor


# Reference: Ha, L., Krüger, J., & Silva, C. T. (2009, December). Fast Four‐Way Parallel Radix Sorting on GPUs. In Computer Graphics Forum (Vol. 28, No. 8, pp. 2368-2378). Oxford, UK: Blackwell Publishing Ltd.
BLOCK_SZ = 64
@ti.data_oriented
class RadixSort(object):
    def __init__(self, input_len, dtype, val_col=0, mode='general') -> None:
        self.dtype = dtype
        self.data_in = ti.field(dtype=dtype, shape=(input_len, val_col+1))
        self.data_out = ti.field(dtype=dtype, shape=(input_len, val_col+1))
        self.sort_with_value = False
        self.value_in = None
        self.value_out = None
        self.val_col = val_col+1
        if val_col > 0:
            self.sort_with_value = True
        self.max_bits = 64 if dtype == ti.i64 else 32
        
        if mode=='general':
            radix = 8
            self.num_buckets = 1 << radix
            self.num_iters = int(self.max_bits / radix)
            self.hist = ti.field(ti.u32, shape=256)
            self.prefix_sum = ti.field(ti.u32, shape=256)
            self.offset = ti.field(ti.u32, shape=input_len)
            self.run = self.radix_sort_general
        else:
            if current_cfg().arch == ti.cuda:
                self.grid_sz = int(input_len / BLOCK_SZ) if input_len % BLOCK_SZ == 0 else int(input_len / BLOCK_SZ) + 1
                self.pse = PrefixSumExecutor(4 * self.grid_sz, dtype=ti.i32)
                self.prefix_sums = ti.field(dtype, shape=input_len)
                self.block_sums = ti.field(dtype, shape=self.pse.get_length())
                self.run = self.radix_sort_gpu
            elif current_cfg().arch == ti.cpu:
                self.pse = PrefixSumExecutor(input_len + 1)
                self.radix_offset0 = ti.field(dtype, shape=self.pse.get_length())
                self.radix_offset1 = ti.field(dtype, shape=self.pse.get_length())
                self.run = self.radix_sort_cpu
            else:
                raise RuntimeError(f"{str(current_cfg().arch)} is not supported for radix sort.")
            
    def radix_sort_general(self, input_len):
        self.general_radix_sort(input_len, self.hist, self.prefix_sum, self.offset, self.data_out, self.data_in)
    
    @ti.kernel
    def general_radix_sort(self, input_len: int, hist: ti.template(), prefix_sum: ti.template(), offset: ti.template(), data_out: ti.template(), data_in: ti.template()):
        """
        Radix sort the morton codes, using 8 bits at a time.
        """
        for i in ti.static(range(self.num_iters)):
            # Clear histogram
            for j in range(self.num_buckets):
                hist[j] = 0

            # Fill histogram
            ti.loop_config(serialize=True)
            for i_a in range(input_len):
                code = (data_in[i_a, 0] >> (i * 8)) & 0xFF
                offset[i_a] = ti.atomic_add(hist[ti.i32(code)], 1)

            # Compute prefix sum
            prefix_sum[0] = 0
            for j in range(1, self.num_buckets):  # sequential prefix sum
                prefix_sum[j] = prefix_sum[j - 1] + hist[j - 1]

            # Reorder morton codes
            for i_a in range(input_len):
                code = (data_in[i_a, 0] >> (i * 8)) & 0xFF
                idx = ti.i32(offset[i_a] + prefix_sum[ti.i32(code)])
                for j in ti.static(range(self.val_col)):
                    data_out[idx, j] = data_in[i_a, j]

    def radix_sort_cpu(self, input_len):
        for move in range(30):
            mask = 0x00000001 << move
            self.radix_sort_predicate(input_len, mask, move, self.radix_offset0, self.radix_offset1, self.data_in)
            self.pse.run(self.radix_offset0)
            self.pse.run(self.radix_offset1)
            self.radix_sort_fill(input_len, mask, move, self.radix_offset0, self.radix_offset1, self.data_in, self.data_out)
            self.data_in, self.data_out = self.data_out, self.data_in

    @ti.kernel
    def radix_sort_predicate(self, input_len: int, mask: int, move: int, radix_offset0: ti.template(), radix_offset1: ti.template(), data_in: ti.template()):
        for i in range(input_len):
            radix_offset1[i + 1] = (data_in[i, 0] & mask) >> move
            radix_offset0[i + 1] = 1 - radix_offset1[i + 1]
        for i in range(input_len + 1, radix_offset0.shape[0]):
            radix_offset0[i] = 0
            radix_offset1[i] = 0

    @ti.kernel
    def radix_sort_fill(self, input_len: int, mask: int, move: int, radix_offset0: ti.template(), radix_offset1: ti.template(), data_in: ti.template(), data_out: ti.template()):
        radix_count_zero = radix_offset0[radix_offset0.shape[0] - 1]
        for i in range(input_len):
            condition = (data_in[i, 0] & mask) >> move
            offset = 0
            if condition == 1:
                offset = radix_offset1[i] + radix_count_zero
            else:
                offset = radix_offset0[i]
            for j in ti.static(range(self.val_col)):
                data_out[offset, j] = data_in[i, j]
        
    def radix_sort_gpu(self, input_len):
        for shift_width in range(0, self.max_bits-1, 2):
            self.block_sums.fill(0)
            self.radix_sort_local(shift_width, input_len, self.grid_sz, self.data_in, self.data_out, self.block_sums, self.prefix_sums)
            self.pse.run(self.block_sums)
            self.glbl_shuffle(shift_width, input_len, self.grid_sz, self.data_out, self.data_in, self.block_sums, self.prefix_sums)
    
    @ti.kernel
    def radix_sort_local(self, shift_width: int, size: int, grid_sz: int, data_in: ti.template(), data_out: ti.template(), block_sums: ti.template(), prefix_sums: ti.template()):
        ti.loop_config(block_dim=BLOCK_SZ)
        for cpy_idx in range(size):
            s_data = ti.simt.block.SharedArray((BLOCK_SZ, ), self.dtype)
            s_value = ti.simt.block.SharedArray((BLOCK_SZ, self.val_col), self.dtype)
            mask_out = ti.simt.block.SharedArray((BLOCK_SZ + 1, ), ti.i32)
            merged_scan_mask_out = ti.simt.block.SharedArray((BLOCK_SZ, ), ti.i32)
            mask_out_sums = ti.simt.block.SharedArray((4, ), ti.i32)
            scan_mask_out_sums = ti.simt.block.SharedArray((4, ), ti.i32)

            thid = int(cpy_idx % BLOCK_SZ)
            blid = int(cpy_idx // BLOCK_SZ)
            t_data = data_in[cpy_idx, 0]
            t_value = ti.Vector.zero(self.dtype, self.val_col)
            if ti.static(self.sort_with_value):
                for i in ti.static(range(1, self.val_col)):
                    t_value[i-1] = data_in[cpy_idx, i]
            t_2bit_extract = ti.cast((t_data >> shift_width) & 3, ti.i32)

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
                sum = ti.cast(0, self.dtype)
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
                run_sum = ti.cast(0, self.dtype)
                for i in ti.static(range(4)):
                    scan_mask_out_sums[i] = run_sum
                    run_sum += mask_out_sums[i]
            ti.simt.block.sync()

            t_prefix_sum = merged_scan_mask_out[thid]
            new_pos = t_prefix_sum + scan_mask_out_sums[t_2bit_extract]
            ti.simt.block.sync()

            s_data[new_pos] = t_data
            if ti.static(self.sort_with_value):
                for i in ti.static(range(1, self.val_col)):
                    s_value[new_pos, i-1] = t_value[i-1]
            merged_scan_mask_out[new_pos] = t_prefix_sum
            ti.simt.block.sync()
            
            prefix_sums[cpy_idx] = merged_scan_mask_out[thid]
            data_out[cpy_idx, 0] = s_data[thid]
            if ti.static(self.sort_with_value):
                for i in ti.static(range(1, self.val_col)):
                    data_out[cpy_idx, i] = s_value[thid, i-1]
            ti.simt.block.sync()
            
    @ti.kernel
    def glbl_shuffle(self, shift_width: int, size: int, grid_sz: int, data_in: ti.template(), data_out: ti.template(), block_sums: ti.template(), prefix_sums: ti.template()):
        ti.loop_config(block_dim=BLOCK_SZ)
        for thread_id in range(size):
            t_data = data_in[thread_id, 0]
            t_2bit_extract = ti.cast((t_data >> shift_width) & 3, ti.i32)
            t_prefix_sum = prefix_sums[thread_id]
            index = t_2bit_extract * grid_sz + thread_id // BLOCK_SZ
            offset = block_sums[index - 1] if index > 0 else 0
            data_glbl_pos = offset + t_prefix_sum
            t_value = ti.Vector.zero(self.dtype, self.val_col)
            if ti.static(self.sort_with_value):
                for i in ti.static(range(1, self.val_col)):
                    t_value[i-1] = data_in[thread_id, i]
            ti.simt.block.sync()
            data_out[data_glbl_pos, 0] = t_data
            if ti.static(self.sort_with_value):
                for i in ti.static(range(1, self.val_col)):
                    data_out[data_glbl_pos, i] = t_value[i-1]
