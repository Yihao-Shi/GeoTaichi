import os.path
import sys
sys.path.append("/home/eleven/work/GeoTaichi/third_party/vdb_multigrid")
import taichi as ti

ti.init(arch=ti.cpu, device_memory_GB=13, offline_cache=False, debug=False, kernel_profiler=True)

from src.vdb_grid import *
from src.tools.particle_to_sdf import *
from src.vdb_viewer import *
from src.tools.volume_to_mesh import *

particle_radius = 0.008
voxel_dim = ti.Vector([particle_radius, particle_radius, particle_radius]) * 2
sdf_voxel_dim = voxel_dim * 1

max_num_particles = 10000000
point_cloud = ti.Vector.field(3, ti.f32, max_num_particles)

shape_cube = 0
shape_sphere = 1
@ti.kernel
def make_shape(point_cloud : ti.template(), shape_id: ti.template()) -> ti.i32:
    counter = 0
    if ti.static(shape_id == shape_sphere):
        base_coord = ti.Vector([2, 2, 2])
        center = ti.Vector([3, 3, 3])
        for i, j, k in ti.ndrange(200, 200, 200):
            pos = base_coord + ti.Vector([i + 0.5, j + 0.5, k + 0.5]) * particle_radius * 2
            if (pos - center).norm() < 1:
                index = ti.atomic_add(counter, 1)
                point_cloud[index] = pos
    elif ti.static(shape_id == shape_cube):
        base_coord = ti.Vector([0.25, 0.25, 0.25])
        for i, j, k in ti.ndrange(100, 100, 100):
            index = ti.atomic_add(counter, 1)
            pos = base_coord + ti.Vector([i, j, k]) * particle_radius * 2
            point_cloud[index] = pos

    return counter


vdb_default_levels = [5, 4, 4]
vdb_grid = VdbGrid(voxel_dim, vdb_default_levels)

num_vertices = ti.field(dtype=ti.i32, shape=())
num_indices = ti.field(dtype=ti.i32, shape=())

vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=20000000)
normal_buffer = ti.Vector.field(n=4, dtype=ti.f32, shape=3000000)
indices = ti.field(dtype=ti.i32, shape=400000000)


use_dual_contouring = True
show_mesh = False
profile_epoch = 1
export_mesh = True

@ti.kernel
def mark_sdf(sdf:ti.template()) -> ti.i32:
    counter = 0
    for i, j, k in sdf:
        value = sdf[i, j, k]
        if value != 0.0:
            id = ti.atomic_add(counter, 1)
            point_cloud[id] = ti.Vector([i, j, k]) * voxel_dim

    return counter


@ti.kernel
def print_sdf(sdf: ti.template()):
    for i, j, k in sdf:
        value = sdf[i, j, k]
        if value < 0:
            print(value)


sdf_tool = ParticleToSdf(sdf_voxel_dim, voxel_dim, vdb_default_levels, max_num_particles)
@ti.kernel
def test_kernel():
    print(sdf_tool.vdb.transform.coord_to_voxel_packed(ti.Vector([0.5, 1.2, 1.04])))
    print(sdf_tool.vdb.transform.coord_to_voxel_packed(ti.Vector([0.5, 1.2, 1.05])))

def read_particles():
    import_particle_count = 0
    with open("/home/appledorem/serialized_frame_500.txt") as f:
        for line in f.readlines():
            line = line[:-1]
            pos = line.split(",")
            pos_vec = ti.Vector([float(pos[0]), float(pos[1]), float(pos[2])])
            if pos_vec[0] >= 0.0 and pos_vec[1] >= 0.0 and pos_vec[2] >= 0.0:
                point_cloud[import_particle_count] = pos_vec
                import_particle_count += 1
                if import_particle_count % 100000 == 0:
                    print(f"Read {import_particle_count} particles")
    print("Finished Reading")
    return import_particle_count


@ti.kernel
def fill_sphere_sdf(sdf: ti.template()):
    center = ti.Vector([1, 1, 1])
    for i, j, k in ti.ndrange((50, 151), (50, 151), (50, 151)):
        value = (ti.Vector([i, j, k]) * voxel_dim - center).norm()
        if value < 0.6:
            sdf[i, j, k] = value - 0.5


if __name__ == "__main__":

    # num_particles = make_shape(point_cloud, shape_sphere)
    num_particles = read_particles()
    for i in range(profile_epoch):
        sdf_tool.clear()
        print(f"{num_particles} of particles in total.")
        sdf_tool.particle_to_sdf_anisotropic(point_cloud, num_particles, particle_radius, smoothing_radius=4 * voxel_dim[0])
        # fill_sphere_sdf(sdf_tool.sdf.data_wrapper.leaf_value)
        num_indices[None] = 0
        num_vertices[None] = 0
        vdb_grid.clear()
        if use_dual_contouring:
            VolumeToMesh.dual_contouring(sdf_tool.sdf, vdb_grid, -0.1, num_vertices, vertices, num_indices, indices)
        else:
            VolumeToMesh.marching_cube(sdf_tool.sdf, vdb_grid, 0.0, num_vertices, vertices, num_indices, indices, normal_buffer)
        print(f"Generated {num_vertices[None]} vertices and {num_indices[None]} indices")

    if export_mesh:
        writer = ti.tools.PLYWriter(num_vertices=num_vertices[None], num_faces=num_indices[None]//3, face_type="tri")
        arr_vertices = vertices.to_numpy()[:num_vertices[None]]
        arr_indices = indices.to_numpy()[:num_indices[None]]
        arr_normals = normal_buffer.to_numpy()[:num_vertices[None]]
        writer.add_vertex_pos(arr_vertices[:, 0], arr_vertices[:, 1], arr_vertices[:, 2])
        # writer.add_vertex_normal(arr_normals[:, 0], arr_normals[:, 1], arr_normals[:, 2])
        writer.add_faces(arr_indices)
        writer.export("mesh.ply")

    # print_sdf(sdf_tool.sdf.data_wrapper.leaf_value)
    num_mark = mark_sdf(sdf_tool.sdf.data_wrapper.leaf_value)
    if show_mesh:
        window = ti.ui.Window("Mesh Viewer", (2560, 1440))
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()
        camera.position(0, 0.5, -0.5)

        while True:
            vdb_grid.clear()
            camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.LMB)
            scene.set_camera(camera)
            scene.ambient_light((0.8, 0.8, 0.8))
            scene.point_light(pos=(1.5, 1.5, 1.5), color=(1, 1, 1))
            scene.point_light(pos=(3.5, 3, 3.5), color=(0.2, 0.2, 0.2))
            scene.point_light(pos=(0.5, 3, 0.5), color=(0.2, 0.2, 0.2))
            # scene.particles(centers=point_cloud, index_count=num_mark, radius=particle_radius)
            # scene.particles(centers=point_cloud, index_count=num_particles, radius=particle_radius)
            # scene.mesh(vertices=vertices, indices=indices, vertex_count=num_vertices[None], index_count=num_indices[None])
            canvas.scene(scene)
            window.show()

    ti.profiler.print_kernel_profiler_info()
