import numpy as np
import taichi as ti
import os, types

from src.mpm.elements.ElementBase import ElementBase
from src.mpm.elements.HexahedronKernel import transform_local_to_global
from src.mpm.generator.InsertionKernel import *
from src.consititutive_model.ConstitutiveModelBase import ConstitutiveBase
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.utils.GaussPoint import GaussPointInRectangle, GaussPointInTriangle
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefination import vec3f, vec6f, vec2u8, vec3u8, vec2f
from third_party.pyevtk.hl import pointsToVTK


class BodyGenerator(object):
    sims: Simulation
    
    def __init__(self, sims) -> None:
        self.sims = sims
        self.active = True
        self.myRegion = None
        self.myTemplate = None
        
        self.start_time = 0.
        self.end_time = 0.
        self.next_generate_time = 0.
        self.insert_interval = 1e10
        self.write_file = False
        self.check_history = False
        self.snode_tree: ti.SNode = None

        self.particle = None
        self.insert_particle_num = None

        self.FIX = {
                    "Free": 0,
                    "Fix": 1
                   }
  
    def no_print(self):
        self.log = False

    def deactivate(self):
        self.active = False

    def set_system_strcuture(self, body_dict):
        period = DictIO.GetAlternative(body_dict, "Period", [0, 0, 1e6])
        self.start_time = period[0]
        self.end_time = period[1]
        self.insert_interval = period[2]
        self.write_file = DictIO.GetAlternative(body_dict, "WriteFile", False)
        self.visualize = DictIO.GetAlternative(body_dict, "Visualize", False)
        self.myTemplate = DictIO.GetEssential(body_dict, "Template")
        self.check_history = DictIO.GetAlternative(body_dict, "CheckHistory", False)

    def set_region(self, region):
        self.myRegion = region

    def get_region_ptr(self, name):
        if not self.myRegion is None:
            return self.myRegion[name]
        else:
            raise RuntimeError("Region class should be activated first!")

    def print_particle_info(self, nParticlesPerCell, bodyID, materialID, init_v, fix_v, particle_volume, particle_count):
        if particle_count == 0:
            raise RuntimeError("Zero Particles are inserted into region!")
        print("Body ID = ", bodyID)
        print("Material ID = ", materialID)
        print("Particle Number: ", particle_count)
        print("The Number of Particle per Cell: ", nParticlesPerCell)
        print("Initial Velocity = ", init_v)
        print("Fixed Velocity = ", fix_v)
        print("Particle Volume = ", particle_volume)
        print('\n')

    def finalize(self):
        self.snode_tree.destroy()
        del self.myRegion, self.active #, self.field_builder, self.snode_tree
        del self.start_time, self.end_time, self.insert_interval, self.visualize, self.write_file, self.myTemplate

    def begin(self, scene: myScene):
        if self.sims.current_time < self.next_generate_time: return 0
        if self.sims.current_time < self.start_time or self.sims.current_time > self.end_time: return 0
        
        print('#', "Start adding material points ......")
        self.add_body(scene)

        if not scene.particle is None:
            start_particle = int(scene.particleNum[0]) - self.insert_particle_num[None]
            end_particle = int(scene.particleNum[0])
            if not (self.sims.solver_type == "Implicit" and self.sims.material_type == "Fluid"):
                scene.material.state_vars_initialize(start_particle, end_particle, scene.particle)

        if self.visualize:
            if not self.write_file:
                self.scene_visualization(scene)
            elif self.write_file:
                self.generator_visualization()

        if self.sims.current_time + self.insert_interval > self.end_time or self.insert_interval > self.sims.time or self.end_time > self.sims.time or \
            self.end_time == 0 or self.start_time > self.end_time:
            self.deactivate()
        else:
            self.next_generate_time = self.sims.current_time + self.insert_interval
        return 1
    
    def regenerate(self, scene: myScene):
        if self.sims.current_time < self.next_generate_time: return 0
        if self.sims.current_time < self.start_time or self.sims.current_time > self.end_time: return 0
        
        print('#', "Start adding material points ......")
        self.add_points_to_scene(scene)

        if not scene.particle is None:
            start_particle = int(scene.particleNum[0]) - self.insert_particle_num[None]
            end_particle = int(scene.particleNum[0])
            scene.material.state_vars_initialize(start_particle, end_particle, scene.particle)

        if self.sims.current_time + self.insert_interval > self.end_time:
            self.deactivate()
        else:
            self.next_generate_time = self.sims.current_time + self.insert_interval
        return 1

    def scene_visualization(self, scene: myScene):
        start_particle = int(scene.particleNum[0]) - self.insert_particle_num[None]
        end_particle = int(scene.particleNum[0])
        data = scene.material.get_state_vars_dict(start_index=start_particle, end_index=end_particle)
        position = self.particle.to_numpy()[0:self.insert_particle_num[None]]
        posx, posy, posz = np.ascontiguousarray(position[:, 0]), np.ascontiguousarray(position[:, 1]), np.ascontiguousarray(position[:, 2])
        pointsToVTK(f'MPMPackings', posx, posy, posz, data=data)

    def generator_visualization(self):
        position = self.particle.to_numpy()[0:self.insert_particle_num[None]]
        posx, posy, posz = np.ascontiguousarray(position[:, 0]), np.ascontiguousarray(position[:, 1]), np.ascontiguousarray(position[:, 2])
        pointsToVTK(f'MPMPackings', posx, posy, posz, data={})

    def allocate_material_point_memory(self, expected_total_particle_number):
        field_bulider = ti.FieldsBuilder()
        self.particle = ti.Vector.field(self.sims.dimension, float)
        field_bulider.dense(ti.i, expected_total_particle_number).place(self.particle)
        self.snode_tree = field_bulider.finalize()
        self.insert_particle_num = ti.field(int, shape=())

    def check_bodyID(self, scene: myScene, bodyID):
        if bodyID > scene.grid_level - 1:
            raise RuntimeError(f"Keyword:: /bodyID/ must be smaller than {scene.grid_level}")

    def add_body(self, scene: myScene):
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
        name = DictIO.GetEssential(template, "RegionName")
        nParticlesPerCell = DictIO.GetAlternative(template, "nParticlesPerCell", 2)
        region: RegionFunction = self.get_region_ptr(name)

        initial_particle_volume = element.calc_volume() / element.calc_total_particle(nParticlesPerCell)
        region.estimate_expected_particle_num_by_volume(initial_particle_volume)
        return region.expected_particle_number
    
    def rotate_body(self, region: RegionFunction, start_particle_num, end_particle_num):
        kernel_position_rotate_(region.zdirection, region.rotate_center, self.particle, start_particle_num, end_particle_num)

    def generate_material_points(self, scene: myScene, template):
        name = DictIO.GetEssential(template, "RegionName")
        nParticlesPerCell = DictIO.GetAlternative(template, "nParticlesPerCell", 2)
        region: RegionFunction = self.get_region_ptr(name)
        particle_volume = scene.element.calc_volume() / scene.element.calc_total_particle(nParticlesPerCell)
        psize = scene.element.calc_particle_size(nParticlesPerCell)

        start_particle_num = self.insert_particle_num[None]
        self.Generate(scene, region, nParticlesPerCell)
        end_particle_num = self.insert_particle_num[None]
        particle_count = end_particle_num - start_particle_num

        if self.write_file:
            self.write_text(start_particle_num, end_particle_num, particle_volume, psize, scene.element.get_nodal_coords(), scene.element.get_node_connectivity())
        elif not self.write_file:
            material = scene.get_material_ptr()
            particles = scene.get_particle_ptr()
            particleNum = int(scene.particleNum[0])

            bodyID = DictIO.GetEssential(template, "BodyID")
            self.check_bodyID(scene, bodyID)
            rigid_body = DictIO.GetAlternative(template, "RigidBody", False)
            if rigid_body:
                materialID = 0
                density = DictIO.GetAlternative(template, "Density", 2650)
                scene.is_rigid[bodyID] = 1
                material.matProps[materialID].density = density
            else:
                materialID = DictIO.GetEssential(template, "MaterialID")
                density = material.matProps[materialID].density

                density, densityf, porosity, permeability = 0., 0., 0., 0.
                if self.sims.material_type == "TwoPhaseSingleLayer":
                    density = material.matProps[materialID].solid_density
                    densityf = material.matProps[materialID].fluid_density
                    porosity = material.matProps[materialID].porosity
                    permeability = material.matProps[materialID].permeability
                else:
                    density = material.matProps[materialID].density

                if materialID <= 0:
                    raise RuntimeError(f"Material ID {materialID} should be larger than 0")

            particle_stress = DictIO.GetAlternative(template, "ParticleStress", {"GravityField": False, "InternalStress": vec6f([0, 0, 0, 0, 0, 0])})
            init_v = DictIO.GetAlternative(template, "InitialVelocity", [0., 0., 0.] if self.sims.dimension == 3 else [0., 0.])
            fix_v_str = DictIO.GetAlternative(template, "FixVelocity", ["Free", "Free", "Free"] if self.sims.dimension == 3 else ["Free", "Free"])
            if self.sims.dimension == 3:
                fix_v = vec3u8([DictIO.GetEssential(self.FIX, is_fix) for is_fix in fix_v_str])
                if isinstance(init_v, (list, tuple)):
                    init_v = vec3f(init_v)
            elif self.sims.dimension == 2:
                fix_v = vec2u8([DictIO.GetEssential(self.FIX, is_fix) for is_fix in fix_v_str])
                if isinstance(init_v, (list, tuple)):
                    init_v = vec2f(init_v)
            scene.check_particle_num(self.sims, particle_count)
            if self.sims.dimension == 3:
                kernel_add_body_(particles, particleNum, start_particle_num, end_particle_num, self.particle, particle_volume, bodyID, materialID, density, init_v, fix_v)
            elif self.sims.dimension == 2:
                if self.sims.material_type == 'Solid' or self.sims.material_type == 'Fluid':
                    kernel_add_body_2D(particles, particleNum, start_particle_num, end_particle_num, self.particle, particle_volume, bodyID, materialID, density, init_v, fix_v)
                elif self.sims.material_type == 'TwoPhaseSingleLayer':
                    kernel_add_body_twophase2D(particles, particleNum, start_particle_num, end_particle_num, self.particle, particle_volume, bodyID, materialID, density, densityf, porosity, permeability, init_v, fix_v)
            self.set_particle_stress(scene, materialID, particleNum, particle_count, particle_stress, psize)
            scene.push_psize(np.repeat([psize], particle_count, axis=0))
            print(" Body(s) Information ".center(71, '-'))
            traction = DictIO.GetAlternative(template, "Traction", {})
            self.set_traction(scene, region, particle_count, traction)
            self.print_particle_info(nParticlesPerCell, bodyID, materialID, init_v, fix_v_str, particle_volume, particle_count)
            scene.particleNum[0] += particle_count

    def add_points_to_scene(self, scene: myScene):
        if type(self.myTemplate) is dict:
            self.add_point_to_scene(scene, self.myTemplate)
        elif type(self.myTemplate) is list:
            for template in self.myTemplate:
                self.add_point_to_scene(scene, template)

    def add_point_to_scene(self, scene: myScene, template):
        material = scene.get_material_ptr()
        particles = scene.get_particle_ptr()
        particleNum = int(scene.particleNum[0])

        bodyID = DictIO.GetEssential(template, "BodyID")
        name = DictIO.GetEssential(template, "RegionName")
        nParticlesPerCell = DictIO.GetAlternative(template, "nParticlesPerCell", 2)
        region: RegionFunction = self.get_region_ptr(name)
        psize = scene.element.calc_particle_size(nParticlesPerCell)
        particle_volume = scene.element.calc_volume() / scene.element.calc_total_particle(nParticlesPerCell)
        self.check_bodyID(scene, bodyID)
        rigid_body = DictIO.GetAlternative(template, "RigidBody", False)
        if rigid_body:
            materialID = 0
            density = DictIO.GetAlternative(template, "Density", 2650)
            scene.is_rigid[bodyID] = 1
            material.matProps[materialID].density = density
        else:
            materialID = DictIO.GetEssential(template, "MaterialID")
            density = material.matProps[materialID].density
            if materialID <= 0:
                raise RuntimeError(f"Material ID {materialID} should be larger than 0")

        particle_stress = DictIO.GetAlternative(template, "ParticleStress", {"GravityField": False, "InternalStress": vec6f([0, 0, 0, 0, 0, 0])})
        init_v = DictIO.GetAlternative(template, "InitialVelocity", [0., 0., 0.] if self.sims.dimension == 3 else [0., 0.])
        fix_v_str = DictIO.GetAlternative(template, "FixVelocity", ["Free", "Free", "Free"] if self.sims.dimension == 3 else ["Free", "Free"])
        if self.sims.dimension == 3:
            fix_v = vec3u8([DictIO.GetEssential(self.FIX, is_fix) for is_fix in fix_v_str])
            if isinstance(init_v, (list, tuple)):
                init_v = vec3f(init_v)
        elif self.sims.dimension == 2:
            fix_v = vec2u8([DictIO.GetEssential(self.FIX, is_fix) for is_fix in fix_v_str])
            if isinstance(init_v, (list, tuple)):
                init_v = vec2f(init_v)
        scene.check_particle_num(self.sims, self.insert_particle_num[None])
        if self.sims.dimension == 3:
            kernel_add_body_(particles, particleNum, 0, self.insert_particle_num[None], self.particle, psize, particle_volume, bodyID, materialID, density, init_v, fix_v)
        elif self.sims.dimension == 2:
            kernel_add_body_2D(particles, particleNum, 0, self.insert_particle_num[None], self.particle, psize, particle_volume, bodyID, materialID, density, init_v, fix_v)
        self.set_particle_stress(scene, materialID, particleNum, self.insert_particle_num[None], particle_stress, psize)
        scene.push_psize(np.repeat([psize], self.insert_particle_num[None], axis=0))
        print(" Body(s) Information ".center(71, '-'))
        traction = DictIO.GetAlternative(template, "Traction", {})
        self.set_traction(scene, region, self.insert_particle_num[None], traction)
        self.print_particle_info(nParticlesPerCell, bodyID, materialID, init_v, fix_v_str, particle_volume, self.insert_particle_num[None])
        scene.particleNum[0] += self.insert_particle_num[None]
        
    def Generate(self, scene: myScene, region: RegionFunction, nParticlesPerCell):
        if scene.is_rectangle_cell():
            if self.sims.dimension == 3:
                kernel_place_particles_(scene.element.grid_size, scene.element.igrid_size, region.start_point, region.region_size, region.expected_particle_number, nParticlesPerCell,
                                        self.particle, self.insert_particle_num, region.function)
            elif self.sims.dimension == 2:
                kernel_place_particles_2D(scene.element.grid_size, scene.element.igrid_size, region.start_point, region.region_size, region.expected_particle_number, nParticlesPerCell,
                                          self.particle, self.insert_particle_num, region.function)
        elif scene.is_triangle_cell():
            if scene.element.cell_active is None:
                fb = ti.FieldsBuilder()
                snode_tree = scene.element.set_up_cell_active_flag(fb)
            
            point = GaussPointInTriangle(order=nParticlesPerCell)
            point.create_gauss_point()
            
            scene.element.reset_cell_status()
            kernel_activate_cell_(region.start_point, region.region_size, scene.element.nodal_coords, scene.element.node_connectivity, scene.element.cell_active, region.function)
            kernel_fill_particle_in_cell_(point.gpcoords, scene.element.cell_active, scene.element.nodal_coords, scene.element.node_connectivity, scene.particle, self.insert_particle_num, transform_local_to_global)
            snode_tree.destroy()
        else:
            raise RuntimeError("Wrong element type!")
        
    def set_particle_stress(self, scene: myScene, materialID, particleNum, particle_count, particle_stress, psize):
        if type(particle_stress) is str:
            stress_file = DictIO.GetAlternative(particle_stress, "File", "ParticleStress.txt")
            stress_cloud = np.loadtxt(stress_file, unpack=True, comments='#').transpose()
            if stress_cloud.shape[0] != particle_count:
                raise ValueError("The length of File:: /ParticleStress/ is error!")
            if stress_cloud.shape[1] != 6:
                raise ValueError("The stress tensor should be transform to viogt format")
            kernel_apply_stress_from_file(particleNum, particleNum + particle_count, stress_cloud, scene.particle)
        elif type(particle_stress) is dict:
            gravityField = DictIO.GetAlternative(particle_stress, "GravityField", False)
            initialStress = DictIO.GetAlternative(particle_stress, "InternalStress", vec6f([0, 0, 0, 0, 0, 0]))
            if isinstance(initialStress, (int, float)):
                initialStress = vec6f([float(initialStress), float(initialStress), float(initialStress), 0., 0., 0.])
            elif isinstance(initialStress, (tuple, list)):
                initialStress = vec6f(initialStress)
            if self.sims.material_type == "TwoPhaseSingleLayer":
                porePressure = DictIO.GetAlternative(particle_stress, "PorePressure", 0.)
                self.set_internal_stress(materialID, scene.material, scene.particle, particle_count, int(scene.particleNum[0]), psize, gravityField, initialStress, porePressure)
            else:
                self.set_internal_stress(materialID, scene.material, scene.particle, particle_count, int(scene.particleNum[0]), psize, gravityField, initialStress)

    def set_internal_stress(self, materialID, material: ConstitutiveBase, particle, particle_num, init_particle_num, psize, gravityField, initialStress, porePressure=0):
        if gravityField and materialID >= 0:
            if np.linalg.norm(np.array(self.sims.gravity)) < 1e-12:
                raise ValueError("Gravity must be activated")
            
            gravity = np.array(self.sims.gravity)
            k0 = material.get_lateral_coefficient(materialID)
            directions = (gravity / np.linalg.norm(gravity))[0:self.sims.dimension]
            point_cloud = np.ascontiguousarray(self.particle.to_numpy()[init_particle_num:init_particle_num + particle_num])
            distances = np.full(particle_num, -1.)

            if isinstance(gravityField, types.FunctionType):
                distances = gravityField(point_cloud)
            elif isinstance(gravityField, (np.ndarray, list, tuple)):
                distances = np.asarray(gravityField)
            else:
                projections = point_cloud @ -directions
                t_max = np.max(projections)
                distances = t_max - projections

                denominator = (directions[0] / psize[0]) ** 2 + (directions[1] / psize[1]) ** 2
                if self.sims.dimension == 3:
                    denominator += (directions[2] / psize[2]) ** 2
                distances += np.sqrt(1. / denominator)
            
            density = material.matProps[materialID].density
            kernel_apply_gravity_field_(density, init_particle_num, init_particle_num + particle_num, k0, self.sims.gravity, distances, particle)
        
        if initialStress.n != 6:
            raise ValueError(f"The dimension of initial stress: {initialStress.n} is inconsistent with the dimension of stress vigot tensor in 3D: 6")
        kernel_apply_vigot_stress_(init_particle_num, init_particle_num + particle_num, initialStress, particle)
        if self.sims.material_type == "TwoPhaseSingleLayer":
            kernel_apply_pore_pressure_(init_particle_num, init_particle_num + particle_num, porePressure, particle)

    def set_traction(self, scene: myScene, region: RegionFunction, particle_num, tractions):
        scene.boundary.get_essentials(scene.is_rigid, scene.psize, self.myRegion)
        if tractions:
            if type(tractions) is dict:
                scene.boundary.set_particle_traction(self.sims, tractions, particle_num, int(scene.particleNum[0]), scene.particle, scene.psize, region)
            elif type(tractions) is list:
                for traction in tractions:
                    scene.boundary.set_particle_traction(self.sims, traction, particle_num, int(scene.particleNum[0]), scene.particle, scene.psize, region)

    def write_text(self, to_start, to_end, particle_vol, particle_size, nodal_coords, node_connectivity):
        print('#', "Writing particle(s) into 'Particle' ......")
        print(f"Inserted Sphere Number: {to_end - to_start}")
        particle = self.particle.to_numpy()[to_start:to_end]
        volume = np.repeat(particle_vol, to_end - to_start)
        psize = np.repeat([particle_size], to_end - to_start, axis=0)
        if not os.path.exists("Particle.txt"):
            np.savetxt('Particle.txt', np.column_stack((particle, volume, psize)), header="     PositionX            PositionY                PositionZ            Volume            SizeX            SizeY            SizeZ", delimiter=" ")
        else: 
            with open('Particle.txt', 'ab') as file:
                np.savetxt(file, np.column_stack((particle, volume, psize)), delimiter=" ")
        
        if not os.path.exists("Node.txt"):
            print('#', "Writing node(s) into 'Node' ......")
            node_id = np.arange(0, nodal_coords.shape[0], 1)
            np.savetxt('Node.txt', np.column_stack((node_id, nodal_coords)), header="     NodeID            PositionX            PositionY            PositionZ", delimiter=" ")
        
        if not node_connectivity is None and not os.path.exists("Cell.txt"):
            print('#', "Writing cell(s) into 'Cell' ......")
            cell_id = np.arange(0, node_connectivity.shape[0], 1)
            header = "     CellID"
            for node in range(node_connectivity.shape[0]):
                header += "            NodeID" + str(node)
            np.savetxt('Cell.txt', np.column_stack((cell_id, node_connectivity)), header=header, delimiter=" ")
        
    
