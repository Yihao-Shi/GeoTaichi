import time

import taichi as ti

from src.dem.ContactManager import ContactManager
from src.dem.engines.ExplicitEngine import ExplicitEngine
from src.dem.GenerateManager import GenerateManager
from src.dem.Recorder import WriteFile
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.constants import Threshold
from src.utils.TypeDefination import vec3f


class Solver(object):
    sims: Simulation
    generator: GenerateManager
    contact: ContactManager
    engine: ExplicitEngine
    recorder: WriteFile

    def __init__(self, sims, generator, contact, engine, recorder):
        self.sims = sims
        self.generator = generator
        self.contact = contact
        self.engine = engine
        self.recorder = recorder

        self.last_save_time = 0.
        self.last_print_time = 0.
        self.calm_interval = 0
        self.last_calm = 0
        self.postprocess = []
    
    def set_callback_function(self, functions):
        if not functions is None:
            if isinstance(functions, list):
                for f in functions:
                    self.postprocess.append(ti.kernel(f))
            elif isinstance(functions, dict):
                for f in functions.values():
                    self.postprocess.append(ti.kernel(f))
            elif isinstance(functions, type(lambda: None)):
                self.postprocess.append(ti.kernel(functions))
    
    def set_particle_calm(self, scene, calm_interval):
        if calm_interval:
            self.calm_interval = calm_interval
            self.postprocess.append(lambda: self.engine.calm(self.sims.current_step, self.calm_interval, scene))

    def save_file(self, scene):
        print('# Step =', self.sims.current_step, '   ', 'Save Number =', self.sims.current_print, '   ', 'Simulation time =', self.sims.current_time)
        self.recorder.output(self.sims, scene)
        self.sims.timer.profile0()
        print('\n')

    def compile(self, scene):
        print("Compiling first ... ...")
        start_time = time.time()
        self.core(scene)
        end_time = time.time()
        print(f'Compiling time = {end_time - start_time} \n')

    def Solver(self, scene: myScene):
        print("#", " Start Simulation ".center(67,"="), "#")
        
        self.engine.pre_calculation(self.sims, scene, self.contact.neighbor)
        if self.sims.current_time < Threshold:
            self.save_file(scene)
            self.sims.current_print += 1
            self.last_save_time = -0.8 * self.sims.delta

        self.compile(scene)
        start_time = time.time()
        while self.sims.current_time <= self.sims.time:
            self.core(scene)

            new_body = self.generator.regenerate(scene)
            if self.sims.current_time - self.last_save_time + 0.1 * self.sims.delta > self.sims.save_interval or new_body:
                self.save_file(scene)
                self.last_save_time = 1. * self.sims.current_time
                self.sims.current_print += 1
                if new_body:
                    self.engine.update_verlet_table(self.sims, scene, self.contact.neighbor)
                    self.sims.set_max_bounding_sphere_radius(scene.find_bounding_sphere_max_radius(self.sims))

            self.sims.current_time += self.sims.delta
            self.sims.current_step += 1
        end_time = time.time()

        if abs(self.sims.current_time - self.last_save_time) > 0.99 * self.sims.save_interval:
            self.save_file(scene)
            self.last_save_time = 1. * self.sims.current_time
            self.sims.current_print += 1

        print('Physical time = ', end_time - start_time)
        print("#", " End Simulation ".center(67,"="), "#", '\n')

    def Visualize(self, scene: myScene):
        print("#", " Start Simulation ".center(67,"="), "#")

        window = ti.ui.Window('GeoTaichi', self.sims.window_size, show_window = True, vsync=False)
        camera = ti.ui.Camera()
        camera.position(*self.sims.look_from)
        camera.up(*self.sims.camera_up)
        camera.lookat(*self.sims.look_at)
        camera.fov(self.sims.view_angle)
        ui_scene = ti.ui.Scene()
        ui_scene.set_camera(camera)
        canvas = window.get_canvas()

        # Draw the lines for domain
        x_max, y_max, z_max = self.sims.domain[0], self.sims.domain[1], self.sims.domain[2]
        box_anchors = ti.Vector.field(3, dtype=ti.f32, shape=8)
        box_anchors[0] = vec3f([0.0, 0.0, 0.0])
        box_anchors[1] = vec3f([0.0, y_max, 0.0])
        box_anchors[2] = vec3f([x_max, 0.0, 0.0])
        box_anchors[3] = vec3f([x_max, y_max, 0.0])

        box_anchors[4] = vec3f([0.0, 0.0, z_max])
        box_anchors[5] = vec3f([0.0, y_max, z_max])
        box_anchors[6] = vec3f([x_max, 0.0, z_max])
        box_anchors[7] = vec3f([x_max, y_max, z_max])

        box_lines_indices = ti.field(int, shape=(2 * 12))

        for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
            box_lines_indices[i] = val

        self.engine.pre_calculation(self.sims, scene, self.contact.neighbor)
        if self.sims.current_time < Threshold:
            self.save_file(scene)
            self.sims.current_print += 1
            self.last_save_time = -0.8 * self.sims.delta

        self.compile(scene)
        start_time = time.time()
        while window.running:
            self.core(scene)
            
            new_body = self.generator.regenerate(scene)
            if self.sims.current_time - self.last_save_time + 0.1 * self.sims.delta > self.sims.save_interval or new_body:
                self.save_file(scene)
                self.last_save_time = 1. * self.sims.current_time
                self.sims.current_print += 1
                if new_body:
                    self.engine.update_verlet_table(self.sims, scene, self.contact.neighbor)
                    self.sims.set_max_bounding_sphere_radius(scene.find_bounding_sphere_max_radius(self.sims))

            if self.sims.current_time - self.last_print_time + 0.1 * self.sims.delta > self.sims.visualize_interval or new_body:
                camera.track_user_inputs(window, movement_speed=self.sims.move_velocity, hold_key=ti.ui.LMB)
                ui_scene.set_camera(camera)

                ui_scene.point_light(self.sims.point_light, color=(1.0, 1.0, 1.0))
                ui_scene.particles(scene.particle.x, per_vertex_radius=scene.particle.rad, color=self.sims.particle_color)

                ui_scene.lines(box_anchors, indices=box_lines_indices, color = (0.99, 0.68, 0.28), width = 1.0)
                canvas.set_background_color(self.sims.background_color)
                canvas.scene(ui_scene)
            
                window.show()
                self.last_print_time = 1. * self.sims.current_time

            self.sims.current_time += self.sims.delta
            self.sims.current_step += 1

        end_time = time.time()

        if abs(self.sims.current_time - self.last_save_time) > self.sims.save_interval:
            self.save_file(scene)
            self.last_save_time = 1. * self.sims.current_time
            self.sims.current_print += 1

        print('Physical time = ', end_time - start_time)
        print("#", " End Simulation ".center(67,"="), "#", '\n')

    def core(self, scene):
        self.sims.timer.begin('Reset')
        self.engine.reset_wall_message(scene)
        self.engine.reset_particle_message(scene)
        self.sims.timer.end('Reset')
        self.engine.update_neighbor_lists(self.sims, scene, self.contact.neighbor)
        self.engine.integration(self.sims, scene, self.contact.neighbor)
        for functions in self.postprocess:
            functions()
    

    
