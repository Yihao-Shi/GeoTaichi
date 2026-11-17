from src.utils.errors import ConfigurationError, SimulationStateError

import numpy as np
import taichi as ti
import os
from src.mpm.generator.Generator import Generator
from src.mpm.elements.ElementBase import ElementBase
from src.mpm.elements.HexahedronKernel import transform_local_to_global
from src.mpm.generator.InsertionKernel import *
from src.mpm.SceneManager import SceneManager
from src.mpm.Simulation import Simulation
from src.mesh.GaussPointInTriangle import GaussPointInTriangle
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefinition import vec3f, vec2u8, vec3u8, vec2f
from third_party.pyevtk.hl import pointsToVTK
from src.utils.TypeDefinition import vec3f


class BodyGenerator(Generator):
    sims: Simulation

    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.write_file = False
        self.write_path = "."
        self.check_history = False
        self.snode_tree: ti.SNode = None
        self.PHASE = {
            "Solid": 1,
            "Fluid": 2,
            "solid": 1,
            "fluid": 2,
            1: 1,
            2: 2,
        }

        self.particle = None
        self.insert_particle_num = None

    def no_print(self):
        self.log = False

    def deactivate(self):
        self.active = False

    def set_system_strcuture(self, body_dict):
        period = DictIO.get_alternative(body_dict, "Period", [self.sims.current_time, self.sims.current_time, 1e6])
        self.start_time = period[0]
        self.end_time = period[1]
        self.insert_interval = period[2]
        self.write_file = DictIO.get_alternative(body_dict, "WriteFile", self.write_file)
        self.write_path = DictIO.get_alternative(body_dict, "WritePath", self.write_path)
        self.visualize = DictIO.get_alternative(body_dict, "Visualize", False)
        self.myTemplate = DictIO.get_essential(body_dict, "Template")
        self.check_history = DictIO.get_alternative(body_dict, "CheckHistory", False)

    def set_region(self, region):
        self.myRegion = region

    def get_region_ptr(self, name):
        if not self.myRegion is None:
            return self.myRegion[name]
        else:
            raise SimulationStateError("Region class should be activated first!")

    def finalize(self):
        self.snode_tree.destroy()
        del self.myRegion, self.active  # , self.field_builder, self.snode_tree
        del self.start_time, self.end_time, self.insert_interval, self.visualize, self.write_file, self.myTemplate

    def begin(self, scene: SceneManager):
        if self.sims.current_time < self.next_generate_time:
            return 0
        if self.sims.current_time < self.start_time or self.sims.current_time > self.end_time:
            return 0

        print("#", "Start adding material points ......")
        self.add_body(scene)

        if not scene.particle is None:
            scene.material.update_material_mapping(scene.particle, int(scene.particleNum[0]))

        if self.visualize:
            if not self.write_file:
                self.scene_visualization(scene)
            elif self.write_file:
                self.generator_visualization()

        if (
            self.sims.current_time + self.insert_interval > self.end_time
            or self.insert_interval > self.sims.time
            or self.end_time == 0
            or self.start_time > self.end_time
        ):
            self.deactivate()
        else:
            self.next_generate_time = self.sims.current_time + self.insert_interval
        return 1

    def regenerate(self, scene: SceneManager):
        if self.sims.current_time < self.next_generate_time:
            return 0
        if self.sims.current_time < self.start_time or self.sims.current_time > self.end_time:
            return 0

        print("#", "Start adding material points ......")
        self.add_points_to_scene(scene)

        if not scene.particle is None:
            scene.material.update_material_mapping(scene.particle, int(scene.particleNum[0]))

        if self.sims.current_time + self.insert_interval > self.end_time:
            self.deactivate()
        else:
            self.next_generate_time = self.sims.current_time + self.insert_interval
        return 1

    def scene_visualization(self, scene: SceneManager):
        start_particle = int(scene.particleNum[0]) - self.insert_particle_num[None]
        end_particle = int(scene.particleNum[0])
        data = scene.material.get_state_vars_dict(start_index=start_particle, end_index=end_particle)
        position = self.particle.to_numpy()[0 : self.insert_particle_num[None]]
        posx, posy, posz = (
            np.ascontiguousarray(position[:, 0]),
            np.ascontiguousarray(position[:, 1]),
            np.ascontiguousarray(position[:, 2]),
        )
        pointsToVTK(f"MPMPackings", posx, posy, posz, data=data)

    def generator_visualization(self):
        position = self.particle.to_numpy()[0 : self.insert_particle_num[None]]
        posx, posy, posz = (
            np.ascontiguousarray(position[:, 0]),
            np.ascontiguousarray(position[:, 1]),
            np.ascontiguousarray(position[:, 2]),
        )
        pointsToVTK(f"MPMPackings", posx, posy, posz, data={})

    def allocate_material_point_memory(self, expected_total_particle_number):
        field_bulider = ti.FieldsBuilder()
        self.particle = ti.Vector.field(self.sims.dimension, float)
        field_bulider.dense(ti.i, expected_total_particle_number).place(self.particle)
        self.snode_tree = field_bulider.finalize()
        self.insert_particle_num = ti.field(int, shape=())

    def check_body_id(self, scene: SceneManager, bodyID):
        if bodyID > scene.grid_level - 1:
            raise ConfigurationError(f"Keyword:: /bodyID/ must be smaller than {scene.grid_level}")

    def add_body(self, scene: SceneManager):
        expected_total_particle_number = 0
        if type(self.myTemplate) is dict:
            expected_total_particle_number += self.sum_up_expected_particle_number(self.myTemplate, scene.element)
        elif type(self.myTemplate) is list:
            for template in self.myTemplate:
                expected_total_particle_number += self.sum_up_expected_particle_number(template, scene.element)
        self.allocate_material_point_memory(expected_total_particle_number)

        if type(self.myTemplate) is dict:
            self.generate_material_points(scene, self.myTemplate)
        elif type(self.myTemplate) is list:
            for template in self.myTemplate:
                self.generate_material_points(scene, template)

    def sum_up_expected_particle_number(self, template, element: ElementBase):
        name = DictIO.get_essential(template, "RegionName")
        n_particles_per_cell = DictIO.get_alternative(template, "nParticlesPerCell", 2)
        region: RegionFunction = self.get_region_ptr(name)

        initial_particle_volume = element.calc_volume() / element.calc_total_particle(n_particles_per_cell)
        region.estimate_expected_particle_num_by_volume(initial_particle_volume)
        return region.expected_particle_number

    def rotate_body(self, region: RegionFunction, start_particle_num, end_particle_num):
        kernel_position_rotate(region.rotate, region.rotate_center, self.particle, start_particle_num, end_particle_num)

    def generate_material_points(self, scene: SceneManager, template):
        name = DictIO.get_essential(template, "RegionName")
        n_particles_per_cell = DictIO.get_alternative(template, "nParticlesPerCell", 2)
        region: RegionFunction = self.get_region_ptr(name)
        particle_volume = scene.element.calc_volume() / scene.element.calc_total_particle(n_particles_per_cell)
        psize = scene.element.calc_particle_size(n_particles_per_cell)

        if self.check_history:
            scene.delete_particles_in_region(region.function)
            initial_particle = self.insert_particle_num[None]
            kernel_delete_particle_slots_in_region(self.insert_particle_num[None], self.particle, region.function)
            finial_particle = self.insert_particle_num[None]
            print(f"Total {-finial_particle + initial_particle} particles has been deleted from local slot", "\n")

        start_particle_num = self.insert_particle_num[None]
        self.generate(scene, region, n_particles_per_cell)
        end_particle_num = self.insert_particle_num[None]
        particle_count = end_particle_num - start_particle_num

        if self.write_file:
            self.write_text(start_particle_num, end_particle_num, particle_volume, psize)
            scene.element.mesh.write(self.write_path + "/Element.txt")
        elif not self.write_file:
            material = scene.get_material_ptr()
            particles = scene.get_particle_ptr()
            particle_num = int(scene.particleNum[0])

            body_id = DictIO.get_essential(template, "BodyID")
            self.check_body_id(scene, body_id)
            rigid_body = DictIO.get_alternative(template, "RigidBody", False)
            if rigid_body:
                material_id = 0
                density = DictIO.get_alternative(template, "Density", 2650)
                scene.is_rigid[body_id] = 1
                material.matProps[material_id].density = density
            else:
                material_id = DictIO.get_essential(template, "MaterialID")
                if self.sims.random_field:
                    scene.material.matProps[material_id].read_random_field(
                        particle_num, particle_num + particle_count, scene.material.stateVars
                    )
                    density = np.ascontiguousarray(material.stateVars.density.to_numpy())
                else:
                    density = material.matProps[material_id].density

                phase = 0
                if self.sims.material_type == "TwoPhaseSingleLayer" or self.sims.material_type == "TwoPhaseDoubleLayer":
                    if not self.sims.random_field:
                        density = material.matProps[material_id].solid_density
                    densityf = material.matProps[material_id].fluid_density
                    porosity = material.matProps[material_id].porosity
                    permeability = material.matProps[material_id].permeability
                    if self.sims.material_type == "TwoPhaseDoubleLayer":
                        phase = DictIO.get_essential(self.PHASE, DictIO.get_essential(template, "Phase"))
                if material_id <= 0:
                    raise SimulationStateError(f"Material ID {material_id} should be larger than 0")

            if isinstance(density, (np.ndarray, list, tuple)):
                density = np.asarray(density)
            elif isinstance(density, (int, float)):
                density = np.repeat(density, particle_count)

            particle_stress = DictIO.get_alternative(template, "ParticleStress", {"InternalStress": [0, 0, 0, 0, 0, 0]})
            init_v = DictIO.get_alternative(
                template, "InitialVelocity", [0.0, 0.0, 0.0] if self.sims.dimension == 3 else [0.0, 0.0]
            )
            fix_v_str = DictIO.get_alternative(
                template, "FixVelocity", ["Free", "Free", "Free"] if self.sims.dimension == 3 else ["Free", "Free"]
            )
            if self.sims.dimension == 3:
                fix_v = vec3u8([DictIO.get_essential(self.FIX, is_fix) for is_fix in fix_v_str])
                if isinstance(init_v, (list, tuple)):
                    init_v = vec3f(init_v)
            elif self.sims.dimension == 2:
                fix_v = vec2u8([DictIO.get_essential(self.FIX, is_fix) for is_fix in fix_v_str])
                if isinstance(init_v, (list, tuple)):
                    init_v = vec2f(init_v)
            scene.check_particle_num(self.sims, particle_count)
            if self.sims.dimension == 3:
                if self.sims.material_type == "Solid" or self.sims.material_type == "Fluid":
                    kernel_add_body(
                        particles,
                        particle_num,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        body_id,
                        material_id,
                        density,
                        init_v,
                        fix_v,
                    )
                elif self.sims.material_type == "TwoPhaseSingleLayer" and rigid_body:
                    kernel_add_body_twophase_rigid(
                        particles,
                        particle_num,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        body_id,
                        material_id,
                        density,
                        init_v,
                        fix_v,
                    )
                elif self.sims.material_type == "TwoPhaseSingleLayer":
                    kernel_add_body_twophase(
                        particles,
                        particle_num,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        body_id,
                        material_id,
                        density,
                        densityf,
                        porosity,
                        permeability,
                        init_v,
                        fix_v,
                    )
                elif self.sims.material_type == "TwoPhaseDoubleLayer":
                    kernel_add_body_twophase_double_point(
                        particles,
                        particle_num,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        body_id,
                        material_id,
                        phase,
                        density,
                        densityf,
                        porosity,
                        permeability,
                        init_v,
                        fix_v,
                    )
            elif self.sims.dimension == 2:
                if self.sims.material_type == "Solid" or self.sims.material_type == "Fluid":
                    kernel_add_body_2d(
                        particles,
                        particle_num,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        body_id,
                        material_id,
                        density,
                        init_v,
                        fix_v,
                    )
                elif self.sims.material_type == "TwoPhaseSingleLayer" and rigid_body:
                    kernel_add_body_2d_rigid(
                        particles,
                        particle_num,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        body_id,
                        material_id,
                        density,
                        init_v,
                        fix_v,
                        self.sims.axis_offset,
                    )
                elif self.sims.material_type == "TwoPhaseSingleLayer":
                    kernel_add_body_twophase2d(
                        particles,
                        particle_num,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        body_id,
                        material_id,
                        density,
                        densityf,
                        porosity,
                        permeability,
                        init_v,
                        fix_v,
                        self.sims.axis_offset,
                    )
                elif self.sims.material_type == "TwoPhaseDoubleLayer":
                    kernel_add_body_twophase_double_point2d(
                        particles,
                        particle_num,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        body_id,
                        material_id,
                        phase,
                        density,
                        densityf,
                        porosity,
                        permeability,
                        init_v,
                        fix_v,
                        self.sims.axis_offset,
                    )
            self.set_particle_stress(scene, particle_num, particle_count, particle_stress)
            scene.push_psize(np.repeat([psize], particle_count, axis=0))
            traction = DictIO.get_alternative(template, "Traction", {})
            self.set_traction(particle_count, traction, scene, region)
            scene.material.state_vars_initialize(
                material_id, particle_num, particle_num + particle_count, scene.particle
            )
            self.print_particle_info(
                body_id, material_id, init_v, fix_v_str, particle_count, particle_volume, n_particles_per_cell
            )
            scene.particleNum[0] += particle_count

    def add_points_to_scene(self, scene: SceneManager):
        if type(self.myTemplate) is dict:
            self.add_point_to_scene(scene, self.myTemplate)
        elif type(self.myTemplate) is list:
            for template in self.myTemplate:
                self.add_point_to_scene(scene, template)

    def add_point_to_scene(self, scene: SceneManager, template):
        material = scene.get_material_ptr()
        particles = scene.get_particle_ptr()
        particle_num = int(scene.particleNum[0])

        body_id = DictIO.get_essential(template, "BodyID")
        name = DictIO.get_essential(template, "RegionName")
        n_particles_per_cell = DictIO.get_alternative(template, "nParticlesPerCell", 2)
        region: RegionFunction = self.get_region_ptr(name)
        psize = scene.element.calc_particle_size(n_particles_per_cell)
        particle_volume = scene.element.calc_volume() / scene.element.calc_total_particle(n_particles_per_cell)
        self.check_body_id(scene, body_id)
        rigid_body = DictIO.get_alternative(template, "RigidBody", False)
        if rigid_body:
            material_id = 0
            density = DictIO.get_alternative(template, "Density", 2650)
            scene.is_rigid[body_id] = 1
            material.matProps[material_id].density = density
        else:
            material_id = DictIO.get_essential(template, "MaterialID")
            if self.sims.random_field:
                scene.material.matProps[material_id].read_random_field(
                    particle_num, particle_num + self.insert_particle_num[None], scene.material.stateVars
                )
                density = np.ascontiguousarray(material.stateVars.density.to_numpy())
            else:
                density = material.matProps[material_id].density

            phase = 0
            if self.sims.material_type == "TwoPhaseSingleLayer" or self.sims.material_type == "TwoPhaseDoubleLayer":
                if not self.sims.random_field:
                    density = material.matProps[material_id].solid_density
                densityf = material.matProps[material_id].fluid_density
                porosity = material.matProps[material_id].porosity
                permeability = material.matProps[material_id].permeability
                if self.sims.material_type == "TwoPhaseDoubleLayer":
                    phase = DictIO.get_essential(self.PHASE, DictIO.get_essential(template, "Phase"))
            if material_id <= 0:
                raise SimulationStateError(f"Material ID {material_id} should be larger than 0")

        if isinstance(density, (np.ndarray, list, tuple)):
            density = np.asarray(density)
        elif isinstance(density, (int, float)):
            density = np.repeat(density, self.insert_particle_num[None])

        particle_stress = DictIO.get_alternative(template, "ParticleStress", {"InternalStress": [0, 0, 0, 0, 0, 0]})
        init_v = DictIO.get_alternative(
            template, "InitialVelocity", [0.0, 0.0, 0.0] if self.sims.dimension == 3 else [0.0, 0.0]
        )
        fix_v_str = DictIO.get_alternative(
            template, "FixVelocity", ["Free", "Free", "Free"] if self.sims.dimension == 3 else ["Free", "Free"]
        )
        if self.sims.dimension == 3:
            fix_v = vec3u8([DictIO.get_essential(self.FIX, is_fix) for is_fix in fix_v_str])
            if isinstance(init_v, (list, tuple)):
                init_v = vec3f(init_v)
        elif self.sims.dimension == 2:
            fix_v = vec2u8([DictIO.get_essential(self.FIX, is_fix) for is_fix in fix_v_str])
            if isinstance(init_v, (list, tuple)):
                init_v = vec2f(init_v)
        scene.check_particle_num(self.sims, self.insert_particle_num[None])

        if self.sims.dimension == 3:
            if self.sims.material_type == "Solid" or self.sims.material_type == "Fluid":
                kernel_add_body(
                    particles,
                    particle_num,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    body_id,
                    material_id,
                    density,
                    init_v,
                    fix_v,
                )
            elif self.sims.material_type == "TwoPhaseSingleLayer" and rigid_body:
                kernel_add_body_twophase_rigid(
                    particles,
                    particle_num,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    body_id,
                    material_id,
                    density,
                    init_v,
                    fix_v,
                )
            elif self.sims.material_type == "TwoPhaseSingleLayer":
                kernel_add_body_twophase(
                    particles,
                    particle_num,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    body_id,
                    material_id,
                    density,
                    densityf,
                    porosity,
                    permeability,
                    init_v,
                    fix_v,
                )
            elif self.sims.material_type == "TwoPhaseDoubleLayer":
                kernel_add_body_twophase_double_point(
                    particles,
                    particle_num,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    body_id,
                    material_id,
                    phase,
                    density,
                    densityf,
                    porosity,
                    permeability,
                    init_v,
                    fix_v,
                )
        elif self.sims.dimension == 2:
            if self.sims.material_type == "Solid" or self.sims.material_type == "Fluid":
                kernel_add_body_2d(
                    particles,
                    particle_num,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    body_id,
                    material_id,
                    density,
                    init_v,
                    fix_v,
                )
            elif self.sims.material_type == "TwoPhaseSingleLayer" and rigid_body:
                kernel_add_body_2d_rigid(
                    particles,
                    particle_num,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    body_id,
                    material_id,
                    density,
                    init_v,
                    fix_v,
                    self.sims.axis_offset,
                )
            elif self.sims.material_type == "TwoPhaseSingleLayer":
                kernel_add_body_twophase2d(
                    particles,
                    particle_num,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    body_id,
                    material_id,
                    density,
                    densityf,
                    porosity,
                    permeability,
                    init_v,
                    fix_v,
                    self.sims.axis_offset,
                )
            elif self.sims.material_type == "TwoPhaseDoubleLayer":
                kernel_add_body_twophase_double_point2d(
                    particles,
                    particle_num,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    body_id,
                    material_id,
                    phase,
                    density,
                    densityf,
                    porosity,
                    permeability,
                    init_v,
                    fix_v,
                    self.sims.axis_offset,
                )

        self.set_particle_stress(scene, particle_num, self.insert_particle_num[None], particle_stress)
        scene.push_psize(np.repeat([psize], self.insert_particle_num[None], axis=0))
        traction = DictIO.get_alternative(template, "Traction", {})
        self.set_traction(self.insert_particle_num[None], traction, scene, region)
        scene.material.state_vars_initialize(
            material_id, particle_num, particle_num + self.insert_particle_num[None], scene.particle
        )
        self.print_particle_info(
            body_id,
            material_id,
            init_v,
            fix_v_str,
            self.insert_particle_num[None],
            particle_volume,
            n_particles_per_cell,
        )
        scene.particleNum[0] += self.insert_particle_num[None]

    def generate(self, scene: SceneManager, region: RegionFunction, nParticlesPerCell):
        if scene.is_rectangle_cell():
            if self.sims.dimension == 3:
                kernel_place_particles(
                    scene.element.grid_size,
                    scene.element.igrid_size,
                    region.start_point,
                    region.region_size,
                    region.expected_particle_number,
                    nParticlesPerCell,
                    self.particle,
                    self.insert_particle_num,
                    region.function,
                )
            elif self.sims.dimension == 2:
                kernel_place_particles_2d(
                    scene.element.grid_size,
                    scene.element.igrid_size,
                    region.start_point,
                    region.region_size,
                    region.expected_particle_number,
                    nParticlesPerCell,
                    self.particle,
                    self.insert_particle_num,
                    region.function,
                )
        elif scene.is_triangle_cell():
            if scene.element.cell_active is None:
                fb = ti.FieldsBuilder()
                snode_tree = scene.element.set_up_cell_active_flag(fb)

            point = GaussPointInTriangle(order=nParticlesPerCell)
            point.create_gauss_point()

            scene.element.reset_cell_status()
            kernel_activate_cell(
                region.start_point,
                region.region_size,
                scene.element.mesh.nodal_coords,
                scene.element.node_connectivity,
                scene.element.cell_active,
                region.function,
            )
            kernel_fill_particle_in_cell(
                point.gpcoords,
                scene.element.cell_active,
                scene.element.mesh.nodal_coords,
                scene.element.node_connectivity,
                scene.particle,
                self.insert_particle_num,
                transform_local_to_global,
            )
            snode_tree.destroy()
        else:
            raise SimulationStateError("Wrong element type!")

    def write_text(self, to_start, to_end, particle_vol, particle_size):
        print("#", "Writing particle(s) into 'Particle' ......")
        print(f"Inserted Sphere Number: {to_end - to_start}")
        particle = self.particle.to_numpy()[to_start:to_end]
        volume = np.repeat(particle_vol, to_end - to_start)
        psize = np.repeat([particle_size], to_end - to_start, axis=0)
        if not os.path.exists(self.write_path + "/Particle.txt"):
            np.savetxt(
                self.write_path + "/Particle.txt",
                np.column_stack((particle, volume, psize)),
                header="     PositionX            PositionY                PositionZ            Volume            SizeX            SizeY            SizeZ",
                delimiter=" ",
            )
        else:
            with open(self.write_path + "/Particle.txt", "ab") as file:
                np.savetxt(file, np.column_stack((particle, volume, psize)), delimiter=" ")
