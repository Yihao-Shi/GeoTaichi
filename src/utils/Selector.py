import taichi as ti
import trimesh

ti.init()

mesh = trimesh.load("model/bunny.obj")
particles = ti.Vector.field(3, dtype=ti.f32, shape=mesh.vertices.shape[0])
particles.from_numpy(mesh.vertices)


@ti.data_oriented
class Selector:
    """
    A particle selector. Use your left mouse botton to draw a rectangle on the screen and select the particles in the rectangle.
    Example::
        # First, instantiate a selector
        selector = Selector(camera, window, particle_pos)
        while window.running:
            #select the particles in the rectangle when dragging the mouse
            selector.select() 
            #clear the selection when press "c"
            if window.is_pressed("c"):
                selector.clear()
            
            #print the selected particle ids when press "i"
            if window.is_pressed("i"):
                print(selector.get_ids())
            
            #draw the selected particles in red
            scene.particles(particle_pos, radius=0.01, per_vertex_color=selector.per_vertex_color)
    """
    def __init__(self, camera, window, particle_pos) -> None:
        """
        Args:
            camera (ti.ui.Camera): the camera used in the scene
            window (ti.ui.Window): the window used in the scene
            particle_pos (3d ti.Vector.field): the position of all the particles which could be selected
        """
        self.camera = camera
        self.window = window
        self.aspect = window.get_window_shape()[0]/window.get_window_shape()[1]
        self.start = (-1e5,-1e5)
        self.end   = (1e5,1e5)
        self.canvas = window.get_canvas()
        self.per_vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=particles.shape[0])
        self.per_vertex_color.fill([0.1229,0.2254,0.7207])
        self.screen_pos = ti.Vector.field(2, shape=particles.shape[0], dtype=float)
        self.is_in_rect = ti.field(dtype=ti.i32, shape=particles.shape[0])
        self.rect_verts = ti.Vector.field(2, dtype=ti.f32, shape=8)
        self.particle_pos = particle_pos
        self.num_selected = ti.field(dtype=int, shape=())
        self.selected_ids = ti.field(shape=particles.shape[0], dtype=int)
        self.selected_ids.fill(-1)

    def select_particles(self, start, end):
        world_pos = self.particle_pos
        leftbottom = [min(start[0], end[0]), min(start[1], end[1])]
        righttop   = [max(start[0], end[0]), max(start[1], end[1])]
        view_ti = ti.math.mat4(self.camera.get_view_matrix())
        proj_ti = ti.math.mat4(self.camera.get_projection_matrix(self.aspect))

        @ti.kernel
        def world_to_screen_kernel(world_pos:ti.template()):
            for i in range(world_pos.shape[0]):
                pos_homo = ti.math.vec4([world_pos[i][0], world_pos[i][1], world_pos[i][2], 1.0])
                ndc = pos_homo @ view_ti @ proj_ti #CAUTION: right multiply
                ndc /= ndc[3]

                self.screen_pos[i][0] = ndc[0]
                self.screen_pos[i][1] = ndc[1]
                #from [-1,1] scale to [0,1]
                self.screen_pos[i][0] = (self.screen_pos[i][0] + 1) /2
                self.screen_pos[i][1] = (self.screen_pos[i][1] + 1) /2
            
        @ti.kernel
        def judge_point_in_rect_kernel():
            for i in range(self.screen_pos.shape[0]):
                if  self.screen_pos[i][0] > leftbottom[0] and\
                    self.screen_pos[i][0] < righttop[0] and\
                    self.screen_pos[i][1] > leftbottom[1] and\
                    self.screen_pos[i][1] < righttop[1]:
                    self.is_in_rect[i] = True
                    self.per_vertex_color[i] = [1,0,0]
        
        world_to_screen_kernel(world_pos)
        judge_point_in_rect_kernel()
    
    def select(self):
        if self.window.is_pressed(ti.ui.LMB):
            self.clear()
            self.start = self.window.get_cursor_pos()
            if self.window.get_event(ti.ui.RELEASE):
                self.end = self.window.get_cursor_pos()
            self.rect(self.start[0], self.start[1], self.end[0], self.end[1])
            self.canvas.lines(vertices=self.rect_verts, color=(1,0,0), width=0.005)

            self.select_particles(self.start, self.end)

    def clear(self):
        self.per_vertex_color.fill([0.1229,0.2254,0.7207])
        self.is_in_rect.fill(0)

    def rect(self, x_min, y_min, x_max, y_max):
        self.rect_verts[0] = [x_min, y_min]
        self.rect_verts[1] = [x_max, y_min]
        self.rect_verts[2] = [x_min, y_max]
        self.rect_verts[3] = [x_max, y_max]
        self.rect_verts[4] = [x_min, y_min]
        self.rect_verts[5] = [x_min, y_max]
        self.rect_verts[6] = [x_max, y_min]
        self.rect_verts[7] = [x_max, y_max]

    def get_ids(self):
        @ti.kernel
        def get_ids_kernel():
            self.num_selected[None] = 0
            for i in range(self.is_in_rect.shape[0]):
                if self.is_in_rect[i]:
                    self.selected_ids[self.num_selected[None]] = i
                    self.num_selected[None] += 1
        get_ids_kernel()
        ids_np = self.selected_ids.to_numpy()
        ids_np = ids_np[:self.num_selected[None]]
        return ids_np
    
    def get_num_selected(self):
        return self.num_selected[None]


def visualize(particle_pos):
    window = ti.ui.Window("visualizer", (1080, 720), vsync=True)
    camera = ti.ui.Camera()
    camera.position(0,0,0)
    camera.lookat(0,0,-1)
    camera.fov(45) 
    canvas = window.get_canvas()
    canvas.set_background_color((1,1,1))
    scene = ti.ui.Scene()
    
    selector = Selector(camera, window, particle_pos)

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        selector.select()
        if window.is_pressed("c"):
            selector.clear()
        if window.is_pressed("i"):
            print(selector.get_ids())
        scene.particles(particle_pos, radius=0.01, per_vertex_color=selector.per_vertex_color)

        canvas.scene(scene)
        window.show()


visualize(particles)