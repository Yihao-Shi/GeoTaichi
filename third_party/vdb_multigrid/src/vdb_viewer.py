import taichi as ti
from src.vdb_grid import *


@ti.data_oriented
class VdbViewer:
    edge_per_cube = 12

    def __init__(self, vdb: VdbGrid, bounding_box, max_vertex_usage=0.001, max_index_usage=0.0001,
                 num_max_vertices=None, num_max_indices=None):

        num_cubes = 0
        # Calculate maximum number of vertices to visualize vdb

        tpdconfig = ti.Vector([0, 0, 0])
        for i in ti.static(range(vdb.num_vdb_levels)):
            tpdconfig += vdb.config[i]
            num_cubes += 1 << (tpdconfig[0] + tpdconfig[1] + tpdconfig[2])

        self.num_vdb_levels = vdb.num_vdb_levels
        self.max_num_vertices = ti.static(8 * num_cubes)
        self.max_num_indices = ti.static(VdbViewer.edge_per_cube * 2 * num_cubes)
        vertex_size = int(self.max_num_vertices * max_vertex_usage) if num_max_vertices is None else num_max_vertices
        vertex_size += 1 if vertex_size & 1 else 0
        self.max_num_vertices = vertex_size
        self.vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_vertices)
        self.vertex_color = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_vertices)
        index_size = int(self.max_num_indices * max_index_usage) if num_max_indices is None else num_max_indices
        index_size += 1 if index_size & 1 else 0
        self.max_num_indices = index_size
        self.indices = ti.field(dtype=ti.i32, shape=self.max_num_indices)
        self.bounding_box = bounding_box
        self.viewer_voxel_dim = (self.bounding_box[1] - self.bounding_box[0]) / (1 << tpdconfig[0])

        self.window = ti.ui.Window("Vdb Viewer", (1280, 720))
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0, 0.5, -0.5)


    @ti.kernel
    def clear_frame_data(self):
        for i in range(self.max_num_vertices):
            self.vertices[i] = ti.Vector.zero(n=3, dt=ti.f32)
        for i in range(self.max_num_indices):
            self.indices[i] = 0

    def run_viewer_frame(self, vdb: VdbGrid, generate_new_frame=True):
        if generate_new_frame:
            self.clear_frame_data()
            vdb.data_wrapper.generate_vdb_bbox_vertices_impl(self.vertices, self.indices, self.vertex_color)

        self.scene.lines(vertices=self.vertices,indices=self.indices, width=1.0, per_vertex_color=self.vertex_color)







            










