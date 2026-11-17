import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")
from src.vdb_grid import *
from src.vdb_viewer import *
import random
import open3d as o3d
import numpy as np

ti.init(arch=ti.cuda, device_memory_GB=4, offline_cache=False, debug=False, kernel_profiler=False)

sparse_grid_default_levels = [4, 4, 2]
bounding_box = [ti.Vector([0.0, 0.0, 0.0]),
                ti.Vector([8.0, 8.0, 8.0])
                ]
voxel_bounding_box = [
    ti.Vector([-1, -1, -1]),
    ti.Vector([1024, 1024, 1024])
]
vdb_grid = VdbGrid(bounding_box, sparse_grid_default_levels)


@ti.kernel
def test_basic_write(offset: ti.template()):
    for i, j, k in ti.ndrange(2, 2, 2):
        if i + offset[0] < 1024 and j + offset[1] < 1024 and offset[2] + k < 1024:
            if i + offset[0] >= 0 and j + offset[1] >= 0 and k + offset[2] >= 0:
                vdb_grid.set_value_world(offset[0] + i, offset[1] + j, offset[2] + k, 1)


def has_equal(a: ti.template(), b: ti.template()):
    return a[0] == b[0] or a[1] == b[1] or a[2] == b[2]


@ti.kernel
def print_active(snode: ti.template()):
    for i, j, k in snode:
        print(f"({i}, {j}, {k}) is active")


@ti.func
def transform(n, pos: ti.template()):
    voxel_coord = pos * n
    return ti.cast(voxel_coord, ti.i32)

@ti.kernel
def set_point_cloud(f: ti.template(), n: ti.template()):
    ti.loop_config(block_dim=512)
    for i in range(n):
        voxel_coord = transform(1024, f[i])
        vdb_grid.set_value_world(voxel_coord[0], voxel_coord[1], voxel_coord[2], 1)


@ti.kernel
def print_pc():
    for i, j, k in ti.ndrange(2, 2, 2):
        print(vdb_grid.read_value(i, j, k))

def read_point_cloud(file_path: str):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    f = ti.Vector.field(n=3, dtype=ti.f32, shape=points.shape[0])
    f.from_numpy(points)
    set_point_cloud(f, points.shape[0])
    print_pc()




if __name__ == "__main__":
    # unittest.main()

    vdb_viewer = VdbViewer(vdb_grid, bounding_box, num_max_vertices=10000, num_max_indices=30000)
    counter = 0
    offset = bounding_box[0]
    speed = 1
    direction = ti.Vector([1, 1, 1])
    prev_offset = offset
    while True:
        counter = counter + 1

        if counter % 300 == 0:
            direction = ti.Vector([random.randint(-2, 2), random.randint(-2, 2), random.randint(-2, 2)])
            ti.deactivate_all_snodes()
            counter = 0

        generate_new_frame = False
        if counter % 10 == 0:
            prev_offset = offset
            offset += direction * speed
            offset = ti.min(offset, voxel_bounding_box[1])
            offset = ti.max(offset, voxel_bounding_box[0])
            while prev_offset[0] == offset[0] and prev_offset[1] == offset[1] and prev_offset[2] == offset[2] or \
                    has_equal(offset, voxel_bounding_box[1]) or has_equal(offset, voxel_bounding_box[0]):
                direction = ti.Vector([random.randint(-2, 2), random.randint(-2, 2), random.randint(-2, 2)])
                offset += direction * speed
                offset = ti.min(offset, voxel_bounding_box[1])
                offset = ti.max(offset, voxel_bounding_box[0])
            # ti.deactivate_all_snodes()
            test_basic_write(ti.Vector([int(offset[0]), int(offset[1]), int(offset[2])]))
            vdb_grid.prune(1e-7)
            generate_new_frame = True

        vdb_viewer.run_viewer_frame(vdb_grid, generate_new_frame)

