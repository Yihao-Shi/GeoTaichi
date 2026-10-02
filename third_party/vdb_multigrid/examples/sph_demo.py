from src.algorithms.simulations.sph import *
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, device_memory_GB=10, offline_cache=True, debug=False)

particle_count = 500000
particle_pos = ti.field(dtype=ti.math.vec3, shape=particle_count)
particle_radius = 0.01
particle_mass = 1000 * ti.pow(particle_radius, 3) * 4 / (3 * np.pi)

@ti.kernel
def init_particles():
    base_coord = ti.Vector([0.25, 0.25, 0.25])
    counter = 0
    for i, j, k in ti.ndrange(100, 50, 100):
        index = ti.atomic_add(counter, 1)
        pos = base_coord + ti.Vector([i, j, k]) * particle_radius * 2
        particle_pos[index] = pos



if __name__ == "__main__":
    bounding_box = [
        ti.Vector([0, 0, 0]),
        ti.Vector([1, 1, 1])
    ]
    viscosity = 0.01
    sph = SPHFluid(10, 1000, viscosity, particle_radius * 2, 1000000, particle_mass, bounding_box=bounding_box)

    init_particles()
    sph.add_particles(particle_count, particle_pos, None)

    window = ti.ui.Window("SPH Viewer", (1600, 720))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0.5, -0.5)
    camera.lookat(0.25, 0.25, 0.25)
    gui = window.get_gui()

    while True:
        camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.LMB)
        with gui.sub_window("Debug Panel",x=0,y=0,width=0.1,height=0.1):
            is_next_substep = gui.button("Substep")
            if is_next_substep:
                sph.step(0.01)
                print("Finished step")

        scene.particles(centers=sph.particle_pos, radius=particle_radius, color=(0.203, 0.596, 0.859))

        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(1.5, 1.5, 1.5), color=(1, 1, 1))
        scene.point_light(pos=(3.5, 3, 3.5), color=(0.2, 0.2, 0.2))
        scene.point_light(pos=(0.5, 3, 0.5), color=(0.2, 0.2, 0.2))
        canvas.scene(scene)
        window.show()

