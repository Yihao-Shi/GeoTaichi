import taichi as ti
ti.init(arch=ti.gpu)

rad_max = 0.005
rad_min = 0.005
size = 1.

domain = ti.Vector([size, size, size])
particle_num = 100
particle = ti.Vector.field(3, dtype=ti.f32, shape=particle_num)
radius = ti.field(ti.f32, shape=particle_num)

@ti.kernel
def fills():
    for i in range(particle_num):
        particle[i] = rad_max + (size - 2. * rad_max) * ti.Vector([ti.random(), ti.random(), ti.random()])
        radius[i] = rad_min + (rad_max - rad_min) * ti.random()

fills()

window = ti.ui.Window("Test", (1024, 1024, 1024), show_window=True)
camera = ti.ui.Camera()
camera.position(0.5, -1, 0.5)
camera.up(0, 1, 0)
camera.lookat(0, 1, 0)
camera.fov(70)
scene = ti.ui.Scene()
scene.set_camera(camera)
canvas = window.get_canvas()

while window.running:
    camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.LMB)
    scene.set_camera(camera)
    scene.point_light((0.5, 0.5, 1.), color=(1., 1., 1.))
    scene.particles(particle, rad_max, color=(0.5, 0.5, 0.5))
    canvas.set_background_color((0, 0, 0))
    canvas.scene(scene)

    window.show()

