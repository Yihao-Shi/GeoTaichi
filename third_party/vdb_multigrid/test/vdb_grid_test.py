import os.path
import sys
sys.path.append("/home/eleven/work/GeoTaichi/third_party/vdb_multigrid")
from src.vdb_grid import *

ti.init(arch=ti.cuda, device_memory_GB=6, offline_cache=False, debug=False, kernel_profiler=True)



sparse_grid_default_levels = [4, 4, 2]
vdb_grid = VdbGrid(sparse_grid_default_levels)

fill_dim = ti.Vector([1000, 1000, 500])
query_dim = ti.Vector([1000, 1000, 1000])


@ti.kernel
def test_basic_write():
    ti.loop_config(block_dim=512)
    for i, j, k in ti.ndrange(fill_dim[0], fill_dim[1], fill_dim[2]):
        vdb_grid.set_value(i, j, k, i * j * k)

    ti.loop_config(block_dim=512)
    for i, j, k in ti.ndrange(fill_dim[0], fill_dim[1], fill_dim[2]):
        vdb_grid.set_value(i, j, k, i * j * k)

    ti.loop_config(block_dim=512)
    for i, j, k in ti.ndrange(fill_dim[0], fill_dim[1], fill_dim[2]):
        vdb_grid.set_value(i, j, k, i * j * k)

    ti.loop_config(block_dim=512)
    for i, j, k in ti.ndrange(fill_dim[0], fill_dim[1], fill_dim[2]):
        vdb_grid.set_value(i, j, k, i * j * k)

@ti.kernel
def test_basic_read():
    ti.loop_config(block_dim=512)
    for i, j, k in ti.ndrange(query_dim[0], query_dim[1], query_dim[2]):
        value = vdb_grid.read_value(i, j, k)
        expected = i * j * k if k < fill_dim[2] else 0
        assert value == expected, "Value differs at ({}, {}, {}). Expected: {}, But Got: {}".format(i, j, k, i * j * k, value)






if __name__ == "__main__":
    # unittest.main()
    test_basic_write()
    test_basic_read()
    ti.profiler.print_kernel_profiler_info()
