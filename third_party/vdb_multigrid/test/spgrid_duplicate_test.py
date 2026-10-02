import taichi as ti
ti.init(arch=ti.cuda, device_memory_GB=4, offline_cache=False, debug=False, kernel_profiler=True)

sg = ti.field(ti.f32)
sg0 = ti.root.pointer(ti.ijk, (16, 16, 16))
sg1 = sg0.pointer(ti.ijk, (16, 16, 16))
sg2 = sg1.dense(ti.ijk, (4, 4, 4))
sg2.place(sg)

@ti.kernel
def test_sp_read_write():
    fill_dim = ti.Vector([1000, 1000, 500])
    query_dim = ti.Vector([1000, 1000, 1000])
    ti.loop_config(block_dim=512)
    for i, j, k in ti.ndrange(fill_dim[0], fill_dim[1], fill_dim[2]):
        sg[i, j, k] = i * j * k


    ti.loop_config(block_dim=512)
    for i, j, k in ti.ndrange(fill_dim[0], fill_dim[1], fill_dim[2]):
        sg[i, j, k] = i * j * k

    ti.loop_config(block_dim=512)
    for i, j, k in ti.ndrange(fill_dim[0], fill_dim[1], fill_dim[2]):
        sg[i, j, k] = i * j * k

    ti.loop_config(block_dim=512)
    for i, j, k in ti.ndrange(fill_dim[0], fill_dim[1], fill_dim[2]):
        sg[i, j, k] = i * j * k

    ti.loop_config(block_dim=512)
    for i, j, k in ti.ndrange(query_dim[0], query_dim[1], query_dim[2]):
        value = sg[i, j, k]
        expected = i * j * k if k < fill_dim[2] else 0
        assert value == expected, "Value differs at ({}, {}, {}). Expected: {}, But Got: {}".format(i, j, k, i * j * k, value)


if __name__ == "__main__":
    test_sp_read_write()
    ti.profiler.print_kernel_profiler_info()

    