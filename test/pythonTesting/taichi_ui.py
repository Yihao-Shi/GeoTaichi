import taichi as ti

ti.init(arch=ti.cuda)

N = 10

particles_pos = ti.Vector.field(3, dtype=ti.f32, shape = N)
points_pos = ti.Vector.field(3, dtype=ti.f32, shape = N)
points_indices = ti.Vector.field(1, dtype=ti.i32, shape = N)

@ti.kernel
def init_points_pos(points : ti.template()):
    for i in range(points.shape[0]):
        points[i] = [i for j in range(3)]
        # points[i] = [ti.sin(i * 1.0), i * 0.2, ti.cos(i * 1.0)]

@ti.kernel
def init_points_indices(points_indices : ti.template()):
    for i in range(N):
        points_indices[i][0] = i // 2 + i % 2

init_points_pos(particles_pos)
init_points_pos(points_pos)
#init_points_indices(points_indices)

window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(5, -5, 5)
camera.up(1.0, 1.0, 1.0)
camera.lookat(5.0, 10.0, 5.0)
camera.fov(120)

while window.running:
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((1, 1, 1))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

    scene.particles(particles_pos, color = (0.68, 0.26, 0.19), radius = 0.1)
    # Here you will get visible part from the 3rd point with (N - 4) points.
    scene.lines(points_pos, color = (0.28, 0.68, 0.99), width = 5.0, vertex_count = N - 4, vertex_offset = 2)
    # Using indices to indicate which vertex to use
    # scene.lines(points_pos, color = (0.28, 0.68, 0.99), width = 5.0, indices = points_indices)
    # Case 1, vertex_count will be changed to N - 2 when drawing.
    # scene.lines(points_pos, color = (0.28, 0.68, 0.99), width = 5.0, vertex_count = N - 1, vertex_offset = 0)
    # Case 2, vertex_count will be changed to N - 2 when drawing.
    # scene.lines(points_pos, color = (0.28, 0.68, 0.99), width = 5.0, vertex_count = N, vertex_offset = 2)
    canvas.scene(scene)
    window.show()