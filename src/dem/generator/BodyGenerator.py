import math
import os

import numpy as np
import taichi as ti

from src.dem.generator.BrustNeighbor import BruteSearch
from src.dem.generator.InsertionKernel import *
from src.dem.generator.LinkedCellNeighbor import LinkedCell
from src.dem.generator.ClumpTemplate import ClumpTemplate
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefination import vec3f, vec3i
from third_party.pyevtk.hl import pointsToVTK


class ParticleCreator(object):
    def __init__(self) -> None:
        self.myTemplate = None
        self.FIX = {
                    "Free": 0,
                    "Fix": 1
                   }

    def create(self, sims, scene, body_dict):
        btype = DictIO.GetEssential(body_dict, "BodyType")
        template = DictIO.GetEssential(body_dict, "Template")
        if btype == "Sphere":
            self.create_sphere(sims, scene, template)
        elif btype == "Clump":
            self.create_multisphere(sims, scene, template)
        else:
            lists = ["Sphere", "Clump", "RigidBody", "RigidSDF"]
            raise RuntimeError(f"Invalid Keyword:: /BodyType/: {btype}. Only the following {lists} are valid")

    def create_sphere(self, sims, scene: myScene, template):
        if type(template) is dict:
            self.create_template_sphere(sims, scene, template)
        elif type(template) is list:
            for temp in template:
                self.create_template_sphere(sims, scene, temp)

    def create_template_sphere(self, sims, scene: myScene, template):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        material = scene.get_material_ptr()
        bodyNum = int(scene.sphereNum[0])
        particleNum = int(scene.particleNum[0])

        radius = DictIO.GetEssential(template, "Radius")
        position = DictIO.GetEssential(template, "BodyPoint")
        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_v_str = DictIO.GetAlternative(template, "FixVelocity", ["Free", "Free", "Free"])
        fix_w_str = DictIO.GetAlternative(template, "FixAngularVelocity", ["Free", "Free", "Free"])
        scene.check_particle_num(sims, particle_number=1)
        scene.check_sphere_number(sims, body_number=1)
        fix_v = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_v_str])
        fix_w = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_w_str])
        kernel_create_sphere_(particle, sphere, material, bodyNum, particleNum, position, radius, groupID, matID, init_v, init_w, fix_v, fix_w)
        print(" Sphere Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, position, init_v, init_w, fix_v=fix_v_str, fix_w=fix_w_str)
        scene.sphereNum[0] += 1
        scene.particleNum[0] += 1

    def set_clump_template(self, template_ptr):
        self.myTemplate = template_ptr

    def get_template_ptr_by_name(self, name):
        clump_ptr = DictIO.GetOptional(self.myTemplate, name)
        if not clump_ptr:
            raise KeyError(f"Template name: {name} is not set before")
        return clump_ptr

    def create_multisphere(self, sims, scene: myScene, template):
        if type(template) is dict:
            self.create_template_multisphere(sims, scene, template)
        elif type(template) is list:
            for temp in template:
                self.create_template_multisphere(sims, scene, temp)

    def create_template_multisphere(self, sims, scene: myScene, template):
        particle = scene.get_particle_ptr()
        clump = scene.get_clump_ptr()
        material = scene.get_material_ptr()
        bodyNum = int(scene.clumpNum[0])
        particleNum = int(scene.particleNum[0])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
        com_pos = DictIO.GetEssential(template, "BodyPoint")
        equiv_rad = DictIO.GetEssential(template, "Radius")
        particle_orientation=DictIO.GetAlternative(template, "BodyOrientation", None)
        orientation_parameter = DictIO.GetAlternative(template, "OrientationParameter", None)
        set_orientations = set_orientation()
        set_orientations.set_orientation(particle_orientation, orientation_parameter)

        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        scene.check_particle_num(sims, particle_number=template_ptr.nspheres)
        scene.check_clump_number(sims, body_number=1)
        kernel_create_multisphere_(particle, clump, material, bodyNum, particleNum, template_ptr.nspheres, template_ptr.r_equiv, template_ptr.inertia,
                                   template_ptr.x_pebble, template_ptr.rad_pebble, com_pos, equiv_rad, set_orientations.get_orientation, groupID, matID, init_v, init_w)
        print(" Clump Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, com_pos, init_v, init_w, name=name)
        scene.clumpNum[0] += 1
        scene.particleNum[0] += template_ptr.nspheres

    def print_particle_info(self, groupID, matID, com_pos, init_v, init_w, fix_v=None, fix_w=None, name=None):
        if name:
            print("Template name: ", name)
        print("Group ID = ", groupID)
        print("Material ID = ", matID)
        print("Center of Mass", com_pos)
        print("Initial Velocity = ", init_v)
        print("Initial Angular Velocity = ", init_w)
        if fix_v: print("Fixed Velocity = ", fix_v)
        if fix_w: print("Fixed Angular Velocity = ", fix_w)
        print('\n')



class ParticleGenerator(object):
    region: RegionFunction
    sims: Simulation
    
    def __init__(self, sims) -> None:
        self.sims = sims
        self.region = None
        self.active = True
        self.myTemplate = None
        self.neighbor = None
        self.snode_tree: ti.SNode = None
        self.save_path = ''
        
        self.start_time = 0.
        self.end_time = 0.
        self.next_generate_time = 0.
        self.insert_interval = 1e10
        self.tries_number = 0
        self.porosity = 0.345
        self.is_poission = False
        self.type = None
        self.btype = None
        self.name = None
        self.check_hist = False
        self.write_file = False
        self.visualize = False
        self.template_dict = None

        self.FIX = {
                    "Free": 0,
                    "Fix": 1
                   }
  
    def no_print(self):
        self.log = False

    def deactivate(self):
        self.active = False

    def set_system_strcuture(self, body_dict):
        self.type = DictIO.GetEssential(body_dict, "GenerateType")
        self.btype = DictIO.GetEssential(body_dict, "BodyType")
        self.name = DictIO.GetEssential(body_dict, "RegionName")
        self.template_dict = DictIO.GetEssential(body_dict, "Template")
        period = DictIO.GetAlternative(body_dict, "Period", [0, 0, 1e6])
        self.start_time = period[0]
        self.end_time = period[1]
        self.insert_interval = period[2]
        self.check_hist = DictIO.GetAlternative(body_dict, "CheckHistory", True)
        self.write_file = DictIO.GetAlternative(body_dict, "WriteFile", False)
        self.visualize = DictIO.GetAlternative(body_dict, "Visualize", False)
        self.save_path = DictIO.GetAlternative(body_dict, "SavePath", '')
        self.tries_number = DictIO.GetAlternative(body_dict, "TryNumber", 100)
        self.is_poission = DictIO.GetAlternative(body_dict, "PoissionSampling", False)
        self.porosity = DictIO.GetAlternative(body_dict, "Porosity", 0.345)

    def set_clump_template(self, template_ptr):
        self.myTemplate = template_ptr

    def set_region(self, region):
        self.region = DictIO.GetEssential(region, self.name)

    def print_particle_info(self, groupID, matID, init_v, init_w, fix_v=None, fix_w=None, body_num=0, insert_volume=0, name=None):
        if body_num == 0:
            raise RuntimeError("Zero Particles are inserted into region!")
        if name:
            print("Template name: ", name)
        print("Group ID = ", groupID)
        print("Material ID = ", matID)
        print("Body Number: ", body_num)
        if insert_volume != 0:
            print("Insert Volume: ", insert_volume)
        print("Initial Velocity = ", init_v)
        print("Initial Angular Velocity = ", init_w)
        if fix_v: print("Fixed Velocity = ", fix_v)
        if fix_w: print("Fixed Angular Velocity = ", fix_w)
        print('\n')

    def finalize(self):
        self.snode_tree.destroy()
        del self.region, self.active, self.snode_tree
        del self.start_time, self.end_time, self.insert_interval, self.type, self.name, self.check_hist, self.template_dict, self.write_file
    
    def reset(self):
        if self.type == 'Generate' and self.neighbor.destroy is True:
            self.neighbor.clear()
        self.region.dem_reset()

    def begin(self, scene: myScene):
        if self.sims.current_time < self.next_generate_time: return 0
        if self.sims.current_time < self.start_time or self.sims.current_time > self.end_time: return 0
        if self.btype == "Sphere":
            print('#', "Start adding sphere(s) ......")
            if self.type == "Generate":
                self.generate_spheres(scene)
            elif self.type == "Distribute":
                self.distribute_spheres(scene)
            else:
                lists = ["Generate", "Distribute"]
                raise RuntimeError(f"Invalid Keyword:: /GenerateType/: {self.type}. Only the following {lists} are valid")
        elif self.btype == "Clump":
            print('#', "Start adding Clump(s) ......")
            if self.type == "Generate":
                self.generate_multispheres(scene)
            elif self.type == "Distribute":
                self.distribute_multispheres(scene)
            else:
                lists = ["Generate", "Distribute"]
                raise RuntimeError(f"Invalid Keyword:: /GenerateType/: {self.type}. Only the following {lists} are valid")
        else:
            lists = ["Sphere", "Clump"]
            raise RuntimeError(f"Invalid Keyword:: /BodyType/: {self.btype}. Only the following {lists} are valid")
        
        if self.visualize and not scene.particle is None and not self.write_file:
            self.scene_visualization(scene)
        elif self.visualize and self.write_file:
            self.generator_visualization()

        self.reset()

        if self.sims.current_time + self.next_generate_time > self.end_time or self.insert_interval > self.sims.time or self.end_time > self.sims.time or \
            self.end_time == 0 or self.start_time > self.end_time:
            self.deactivate()
        else:
            self.next_generate_time = self.sims.current_time + self.insert_interval
        return 1
    
    def regenerate(self, scene: myScene):
        if self.sims.current_time < self.next_generate_time: return 0
        if self.sims.current_time < self.start_time or self.sims.current_time > self.end_time: return 0
        if self.btype == "Sphere":
            print('#', "Start adding sphere(s) ......")
            self.add_spheres_to_scene(scene)
        elif self.btype == "Clump":
            print('#', "Start adding Clump(s) ......")
            self.add_clumps_to_scene(scene)
        self.reset()

        if self.sims.current_time + self.next_generate_time > self.end_time or self.insert_interval > self.sims.time or self.end_time > self.sims.time or \
            self.end_time == 0 or self.start_time > self.end_time:
            self.deactivate()
        else:
            self.next_generate_time = self.sims.current_time + self.insert_interval
        return 1
    
    def scene_visualization(self, scene: myScene):
        position = scene.particle.x.to_numpy()[0:int(scene.particleNum[0])]
        posx, posy, posz = np.ascontiguousarray(position[:, 0]), \
                        np.ascontiguousarray(position[:, 1]), \
                        np.ascontiguousarray(position[:, 2])
        bodyID = np.ascontiguousarray(scene.particle.multisphereIndex.to_numpy()[0:int(scene.particleNum[0])])
        groupID = np.ascontiguousarray(scene.particle.groupID.to_numpy()[0:int(scene.particleNum[0])])
        rad = np.ascontiguousarray(scene.particle.rad.to_numpy()[0:int(scene.particleNum[0])])
        pointsToVTK(self.save_path+'DEMPackings', posx, posy, posz, data={'bodyID': bodyID, 'group': groupID, "rad": rad})

    def generator_visualization(self):
        if self.btype == "Sphere":
            position = self.sphere_coords.to_numpy()[0:self.insert_particle_in_neighbor[None]]
            posx, posy, posz = np.ascontiguousarray(position[:, 0]), \
                            np.ascontiguousarray(position[:, 1]), \
                            np.ascontiguousarray(position[:, 2])
            rad = np.ascontiguousarray(self.sphere_radii.to_numpy()[0:self.insert_particle_in_neighbor[None]])
            pointsToVTK(self.save_path+'DEMPackings', posx, posy, posz, data={"rad": rad})
        elif self.btype == "Clump":
            position = self.pebble_coords.to_numpy()[0:self.insert_particle_in_neighbor[None]]
            posx, posy, posz = np.ascontiguousarray(position[:, 0]), \
                            np.ascontiguousarray(position[:, 1]), \
                            np.ascontiguousarray(position[:, 2])
            rad = np.ascontiguousarray(self.pebble_radii.to_numpy()[0:self.insert_particle_in_neighbor[None]])
            pointsToVTK(self.save_path+'DEMPackings', posx, posy, posz, data={"rad": rad})

    # ========================================================= #
    #                        SPHERES                            #
    # ========================================================= #
    def allocate_sphere_memory(self, sphere_num, generate=False, distribute=False):
        field_builder = ti.FieldsBuilder()
        self.sphere_coords = ti.Vector.field(3, float)
        self.sphere_radii = ti.field(float)
        field_builder.dense(ti.i, sphere_num).place(self.sphere_coords, self.sphere_radii)
        self.snode_tree = field_builder.finalize()
        if generate or distribute:
            self.insert_body_num = ti.field(int, shape=())
            self.insert_particle_in_neighbor = ti.field(int, shape=())

    def generate_spheres(self, scene: myScene):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        clump = scene.get_clump_ptr()
        sphereNum = int(scene.sphereNum[0])
        clumpNum = int(scene.clumpNum[0])
        
        insert_num, min_radius, max_radius = 0, [], []
        if type(self.template_dict) is dict:
            insert_num += DictIO.GetEssential(self.template_dict, "BodyNumber")
            min_radius.append(DictIO.GetEssential(self.template_dict, "MinRadius", "Radius"))
            max_radius.append(DictIO.GetEssential(self.template_dict, "MaxRadius", "Radius"))
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                insert_num += DictIO.GetEssential(temp, "BodyNumber")
                min_radius.append(DictIO.GetEssential(temp, "MinRadius", "Radius"))
                max_radius.append(DictIO.GetEssential(temp, "MaxRadius", "Radius"))
        particle_in_region = self.hist_check_by_number(particle, sphere, clump, sphereNum, clumpNum, insert_num, nsphere=1)

        if particle_in_region < 1000:
            if self.neighbor is None: self.neighbor = BruteSearch()
            self.neighbor.neighbor_init(particle_in_region)
        elif particle_in_region >= 1000:
            if self.neighbor is None: self.neighbor = LinkedCell()
            self.neighbor.neighbor_init(min(min_radius), max(max_radius), self.region.region_size, particle_in_region)
        self.allocate_sphere_memory(insert_num, generate=True)

        if self.check_hist and self.region.inserted_particle_num > 0:
            self.neighbor.pre_neighbor_sphere(sphereNum, self.insert_particle_in_neighbor, particle, sphere, self.neighbor.position, self.neighbor.radius, self.region.function, self.region.start_point)
            self.neighbor.pre_neighbor_clump(clumpNum, self.insert_particle_in_neighbor, particle, clump, self.neighbor.position, self.neighbor.radius, self.region.function, self.region.start_point)

        if type(self.template_dict) is dict:
            self.generate_template_spheres(scene, self.template_dict)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.generate_template_spheres(scene, temp)
    
    def hist_check_by_number(self, particle, sphere, clump, sphereNum, clumpNum, insert_num, nsphere):
        if insert_num > 0.:
            if self.check_hist and self.region.inserted_particle_num == 0:
                reval = vec2i([0, 0])
                if sphere: reval += kernel_update_particle_number_by_sphere_(sphereNum, sphere, particle, self.region.function)
                if clump: reval += kernel_update_pebble_number_by_clump_(clumpNum, clump, particle, self.region.function)
                self.region.add_inserted_body(reval[0])
                self.region.add_inserted_particle(reval[1])
        else:
            raise RuntimeError(f"BodyNumber should be set in the dictionary: {self.name}!")
        return insert_num * nsphere + self.region.inserted_particle_num
    
    def hist_check_by_volume(self, particle, sphere, clump, sphereNum, clumpNum):
        if self.check_hist and self.region.inserted_particle_num == 0 and (sphere or clump):
            reval = vec2f([0, 0])
            if sphere: reval += kernel_update_particle_volume_by_sphere_(sphereNum, sphere, particle, self.region.function)
            if clump: reval += kernel_update_particle_volume_by_clump_(clumpNum, clump, particle, self.region.function)
            self.region.add_inserted_particle_volume(reval[0])
            self.region.add_inserted_particle(reval[1])

    def generate_template_spheres(self, scene: myScene, template): 
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        material = scene.get_material_ptr()
        sphereNum = int(scene.sphereNum[0])
        particleNum = int(scene.particleNum[0])

        actual_body = DictIO.GetEssential(template, "BodyNumber")
        min_radius = DictIO.GetEssential(template, "MinRadius", "Radius")
        max_radius = DictIO.GetEssential(template, "MaxRadius", "Radius")   
        start_body_num = self.insert_body_num[None]
        self.GenerateSphere(min_radius, max_radius, actual_body, start_body_num)
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num
        
        self.rotate_sphere_packing(start_body_num, end_body_num)
        if self.write_file:
            self.write_sphere_text(start_body_num, end_body_num)
        else:
            groupID = DictIO.GetEssential(template, "GroupID")
            matID = DictIO.GetEssential(template, "MaterialID")
            init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
            init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
            fix_v_str = DictIO.GetAlternative(template, "FixVelocity", ["Free", "Free", "Free"])
            fix_w_str = DictIO.GetAlternative(template, "FixAngularVelocity", ["Free", "Free", "Free"])
            scene.check_particle_num(self.sims, particle_number=self.insert_body_num[None])
            scene.check_sphere_number(self.sims, body_number=self.insert_body_num[None])
            fix_v = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_v_str])
            fix_w = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_w_str])
            kernel_add_sphere_packing(particle, sphere, material, sphereNum, particleNum, start_body_num, end_body_num, self.sphere_coords, self.sphere_radii, 
                                      groupID, matID, init_v, init_w, fix_v, fix_w)
            print(" Sphere(s) Information ".center(71, '-'))
            self.print_particle_info(groupID, matID, init_v, init_w, fix_v=fix_v_str, fix_w=fix_w_str, body_num=body_count)
        scene.sphereNum[0] += body_count
        scene.particleNum[0] += body_count

    def add_spheres_to_scene(self, scene: myScene):
        if type(self.template_dict) is dict:
            self.add_sphere_to_scene(scene, self.template_dict)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.add_sphere_to_scene(scene, temp)

    def add_sphere_to_scene(self, scene: myScene, template):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        material = scene.get_material_ptr()
        sphereNum = int(scene.sphereNum[0])
        particleNum = int(scene.particleNum[0])

        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_v_str = DictIO.GetAlternative(template, "FixVelocity", ["Free", "Free", "Free"])
        fix_w_str = DictIO.GetAlternative(template, "FixAngularVelocity", ["Free", "Free", "Free"])
        scene.check_particle_num(self.sims, particle_number=self.insert_body_num[None])
        scene.check_sphere_number(self.sims, body_number=self.insert_body_num[None])
        fix_v = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_v_str])
        fix_w = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_w_str])
        kernel_add_sphere_packing(particle, sphere, material, sphereNum, particleNum, 0, self.insert_body_num[None], self.sphere_coords, self.sphere_radii, 
                                  groupID, matID, init_v, init_w, fix_v, fix_w)
        print(" Sphere(s) Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, init_v, init_w, fix_v=fix_v_str, fix_w=fix_w_str, body_num=self.insert_body_num[None])
        scene.sphereNum[0] += self.insert_body_num[None]
        scene.particleNum[0] += self.insert_body_num[None]

    def GenerateSphere(self, min_rad, max_rad, actual_body, start_body_num):
        if self.is_poission:
            if self.insert_particle_in_neighbor[None] - start_body_num == 0:
                position = self.region.start_point + 0.5 * self.region.region_size
                radius = 0.5 * (max_rad + min_rad)
            else:
                location = self.insert_particle_in_neighbor[None] - 1
                position = self.neighbor.position[location]
                radius = self.neighbor.radius[location]

            kernel_insert_first_sphere_(self.region.start_point, position, radius, self.insert_body_num, self.insert_particle_in_neighbor, self.sphere_coords, self.sphere_radii, self.neighbor.cell_num, 
                                        self.neighbor.cell_size, self.neighbor.position, self.neighbor.radius, self.neighbor.num_particle_in_cell, self.neighbor.particle_neighbor, self.neighbor.insert_particle)
            kernel_sphere_possion_sampling_(min_rad, max_rad, self.tries_number, actual_body + start_body_num, self.region.start_point, self.insert_body_num, self.insert_particle_in_neighbor, self.sphere_coords, 
                                            self.sphere_radii, self.neighbor.cell_num, self.neighbor.cell_size, self.neighbor.position, self.neighbor.radius, self.neighbor.num_particle_in_cell, self.neighbor.particle_neighbor, 
                                            self.region.function, self.neighbor.overlap, self.neighbor.insert_particle)
        elif not self.is_poission:
            kernel_sphere_generate_without_overlap_(min_rad, max_rad, self.tries_number, actual_body + start_body_num, self.region.start_point, self.region.region_size, self.insert_body_num, self.insert_particle_in_neighbor, 
                                                    self.sphere_coords, self.sphere_radii, self.neighbor.cell_num, self.neighbor.cell_size, self.neighbor.position, self.neighbor.radius, self.neighbor.num_particle_in_cell, 
                                                    self.neighbor.particle_neighbor, self.region.function, self.neighbor.overlap, self.neighbor.insert_particle)

    def distribute_spheres(self, scene: myScene):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        clump = scene.get_clump_ptr()
        sphereNum = int(scene.sphereNum[0])
        clumpNum = int(scene.clumpNum[0])

        if self.porosity >= 0.345:
            self.region.calculate_expected_particle_volume(self.porosity)
        else:
            raise RuntimeError(f"Porosity should be set in the dictionary: {self.name}!")

        total_fraction, insert_particle = 0., 0
        if type(self.template_dict) is dict:
            fraction = DictIO.GetAlternative(self.template_dict, "Fraction", 1.0)
            min_rad = DictIO.GetEssential(self.template_dict, "MinRadius", "Radius")
            template_vol = self.region.estimate_body_volume(fraction)
            insert_particle += math.ceil(template_vol / (4./3. * PI * min_rad * min_rad * min_rad))
            total_fraction += fraction
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                fraction = DictIO.GetAlternative(temp, "Fraction", 1.0)
                min_rad = DictIO.GetEssential(temp, "MinRadius", "Radius")
                template_vol = self.region.estimate_body_volume(fraction)
                insert_particle += math.ceil(template_vol / (4./3. * PI * min_rad * min_rad * min_rad))
                total_fraction += fraction
        if total_fraction < 0. or total_fraction > 1.: 
            raise ValueError("Fraction value error")
        self.hist_check_by_volume(particle, sphere, clump, sphereNum, clumpNum)
        self.region.estimate_body_volume(total_fraction)
        self.allocate_sphere_memory(insert_particle, distribute=True)

        if type(self.template_dict) is dict:
            self.distribute_template_spheres(scene, self.template_dict)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.distribute_template_spheres(scene, temp)
        
    def distribute_template_spheres(self, scene: myScene, template):       
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        material = scene.get_material_ptr()
        sphereNum = int(scene.sphereNum[0])
        particleNum = int(scene.particleNum[0])
         
        fraction = DictIO.GetAlternative(template, "Fraction", 1.0)
        min_radius = DictIO.GetEssential(template, "MinRadius", "Radius")
        max_radius = DictIO.GetEssential(template, "MaxRadius", "Radius")   
        actual_volume = fraction * self.region.expected_particle_volume
        start_body_num =  self.insert_body_num[None]
        insert_volume = self.DistributeSphere(min_radius, max_radius, actual_volume)
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num
        
        self.rotate_sphere_packing(start_body_num, end_body_num)
        if self.write_file:
            self.write_sphere_text(start_body_num, end_body_num)
        else:
            groupID = DictIO.GetEssential(template, "GroupID")
            matID = DictIO.GetEssential(template, "MaterialID")
            init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
            init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
            fix_v_str = DictIO.GetAlternative(template, "FixVelocity", ["Free", "Free", "Free"])
            fix_w_str = DictIO.GetAlternative(template, "FixAngularVelocity", ["Free", "Free", "Free"])
            scene.check_particle_num(self.sims, particle_number=self.insert_body_num[None])
            scene.check_sphere_number(self.sims, body_number=self.insert_body_num[None])
            fix_v = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_v_str])
            fix_w = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_w_str])
            kernel_add_sphere_packing(particle, sphere, material, sphereNum, particleNum, start_body_num, end_body_num, self.sphere_coords, self.sphere_radii, 
                                      groupID, matID, init_v, init_w, fix_v, fix_w)
            print(" Sphere(s) Information ".center(71, '-'))
            self.print_particle_info(groupID, matID, init_v, init_w, fix_v=fix_v_str, fix_w=fix_w_str, body_num=body_count, insert_volume=insert_volume)

        scene.sphereNum[0] += body_count
        scene.particleNum[0] += body_count

    def rotate_sphere_packing(self, start_body_num, end_body_num):
        kernel_position_rotate_(self.region.zdirection, self.region.rotate_center, self.sphere_coords, start_body_num, end_body_num)

    def DistributeSphere(self, min_rad, max_rad, actual_volume):
        return kernel_distribute_sphere_(min_rad, max_rad, actual_volume, self.insert_body_num, self.insert_particle_in_neighbor, self.sphere_coords, self.sphere_radii, 
                                         self.region.start_point, self.region.region_size, self.region.function)
    
    def write_sphere_text(self, to_start, to_end):
        print('#', "Writing sphere(s) into 'SpherePacking' ......")
        print(f"Inserted Sphere Number: {to_end - to_start}", '\n')
        position = self.sphere_coords.to_numpy()[to_start:to_end]
        radius = self.sphere_radii.to_numpy()[to_start:to_end]
        if not os.path.exists("SpherePacking.txt"):
            np.savetxt('SpherePacking.txt', np.column_stack((position, radius)), header="     PositionX            PositionY                PositionZ            Radius", delimiter=" ")
        else: 
            with open('SpherePacking.txt', 'ab') as file:
                np.savetxt(file, np.column_stack((position, radius)), delimiter=" ")

    # ========================================================= #
    #                         CLUMPS                            #
    # ========================================================= #
    def allocate_clump_memory(self, clump_num, template_ptr: ClumpTemplate, generate=False, distribute=False):
        field_builder = ti.FieldsBuilder()
        self.clump_coords = ti.Vector.field(3, float)
        self.clump_radii = ti.field(float)
        self.clump_orients = ti.Vector.field(3, float)
        self.pebble_coords = ti.Vector.field(3, float)
        self.pebble_radii = ti.field(float)
        field_builder.dense(ti.i, clump_num).place(self.clump_coords, self.clump_radii, self.clump_orients)
        field_builder.dense(ti.i, clump_num * template_ptr.nspheres).place(self.pebble_coords, self.pebble_radii)
        self.snode_tree = field_builder.finalize()
        if generate or distribute:
            self.insert_body_num = ti.field(int, shape=())
            self.insert_particle_in_neighbor = ti.field(int, shape=())

    def get_template_ptr_by_name(self, name):
        clump_ptr = DictIO.GetOptional(self.myTemplate, name)
        if not clump_ptr:
            raise KeyError(f"Template name: {name} is not set before")
        return clump_ptr

    def generate_multispheres(self, scene: myScene):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        clump = scene.get_clump_ptr()
        sphereNum = int(scene.sphereNum[0])
        clumpNum = int(scene.clumpNum[0])

        insert_num, min_radius, max_radius = 0, [], []
        if type(self.template_dict) is dict:
            name = DictIO.GetEssential(self.template_dict, "Name")
            template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
            insert_num += DictIO.GetEssential(self.template_dict, "BodyNumber")
            scale_factor_min = DictIO.GetEssential(self.template_dict, "MinRadius", "Radius") / template_ptr.r_equiv
            scale_factor_max = DictIO.GetEssential(self.template_dict, "MaxRadius", "Radius") / template_ptr.r_equiv
            min_radius.append(scale_factor_min * template_ptr.pebble_radius_min)
            max_radius.append(scale_factor_max * template_ptr.pebble_radius_max)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                name = DictIO.GetEssential(temp, "Name")
                template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
                insert_num += DictIO.GetEssential(temp, "BodyNumber")
                scale_factor_min = DictIO.GetEssential(temp, "MinRadius", "Radius") / template_ptr.r_equiv
                scale_factor_max = DictIO.GetEssential(temp, "MaxRadius", "Radius") / template_ptr.r_equiv
                min_radius.append(scale_factor_min * template_ptr.pebble_radius_min)
                max_radius.append(scale_factor_max * template_ptr.pebble_radius_max)
        particle_in_region = self.hist_check_by_number(particle, sphere, clump, sphereNum, clumpNum, insert_num, nsphere=template_ptr.nspheres)

        if particle_in_region < 1000:
            if self.neighbor is None: self.neighbor = BruteSearch()
            self.neighbor.neighbor_init(particle_in_region)
        elif particle_in_region >= 1000:
            if self.neighbor is None: self.neighbor = LinkedCell()
            self.neighbor.neighbor_init(min(min_radius), max(max_radius), self.region.region_size, particle_in_region)
        self.allocate_clump_memory(insert_num, template_ptr, generate=True)

        if self.check_hist and self.region.inserted_particle_num > 0:
            self.neighbor.pre_neighbor_sphere(sphereNum, self.insert_particle_in_neighbor, particle, sphere, self.neighbor.position, self.neighbor.radius, self.region.function, self.region.start_point)
            self.neighbor.pre_neighbor_clump(clumpNum, self.insert_particle_in_neighbor, particle, clump, self.neighbor.position, self.neighbor.radius, self.region.function, self.region.start_point)

        if type(self.template_dict) is dict:
            self.generate_template_multispheres(scene, self.template_dict)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.generate_template_multispheres(scene, temp)
    
    def generate_template_multispheres(self, scene: myScene, template): 
        particle = scene.get_particle_ptr()
        clump = scene.get_clump_ptr()
        material = scene.get_material_ptr()
        clumpNum = int(scene.clumpNum[0])
        particleNum = int(scene.particleNum[0])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)

        actual_body = DictIO.GetEssential(template, "BodyNumber")
        min_radius = DictIO.GetEssential(template, "MinRadius", "Radius")
        max_radius = DictIO.GetEssential(template, "MaxRadius", "Radius")  
        particle_orientation=DictIO.GetAlternative(template, "BodyOrientation", None)
        orientation_parameter = DictIO.GetAlternative(template, "OrientationParameter", None)
        
        set_orientations = set_orientation()
        set_orientations.set_orientation(particle_orientation, orientation_parameter) 
        start_body_num, start_pebble_num =  self.insert_body_num[None], self.insert_particle_in_neighbor[None]
        self.GenerateMultiSphere(min_radius, max_radius, actual_body + start_body_num, start_pebble_num, template_ptr, set_orientations)
        end_body_num, end_pebble_num = self.insert_body_num[None], self.insert_particle_in_neighbor[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_pebble_num
        
        self.rotate_clump_packing(start_body_num, end_body_num, start_pebble_num, end_pebble_num, template_ptr.nspheres)
        if self.write_file:
            self.write_clump_text(template_ptr, start_body_num, end_body_num, particleNum)
        else:
            groupID = DictIO.GetEssential(template, "GroupID")
            matID = DictIO.GetEssential(template, "MaterialID")
            init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
            init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
            scene.check_particle_num(self.sims, particle_number=self.insert_particle_in_neighbor[None] - self.region.inserted_particle_num)
            scene.check_clump_number(self.sims, body_number=self.insert_body_num[None])
            kernel_add_multisphere_packing(particle, clump, material, clumpNum, particleNum, start_body_num, end_body_num, start_pebble_num, template_ptr, 
                                           self.clump_coords, self.clump_radii, self.clump_orients, self.pebble_coords, self.pebble_radii, groupID, matID, init_v, init_w)
            print(" Clump(s) Information ".center(71, '-'))
            self.print_particle_info(groupID, matID, init_v, init_w, body_num=body_count)

        scene.clumpNum[0] += body_count
        scene.particleNum[0] += body_count * template_ptr.nspheres

    def add_clumps_to_scene(self, scene: myScene):
        if type(self.template_dict) is dict:
            self.add_clump_to_scene(scene, self.template_dict)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.add_clump_to_scene(scene, temp)

    def add_clump_to_scene(self, scene: myScene, template):
        particle = scene.get_particle_ptr()
        clump = scene.get_clump_ptr()
        material = scene.get_material_ptr()
        clumpNum = int(scene.clumpNum[0])
        particleNum = int(scene.particleNum[0])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)

        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        scene.check_particle_num(self.sims, particle_number=self.insert_particle_in_neighbor[None] - self.region.inserted_particle_num)
        scene.check_clump_number(self.sims, body_number=self.insert_body_num[None])
        kernel_add_multisphere_packing(particle, clump, material, clumpNum, particleNum, 0, self.insert_body_num[None], 0, template_ptr, 
                                        self.clump_coords, self.clump_radii, self.clump_orients, self.pebble_coords, self.pebble_radii, groupID, matID, init_v, init_w)
        print(" Clump(s) Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, init_v, init_w, body_num=self.insert_body_num[None])
        scene.clumpNum[0] += self.insert_body_num[None]
        scene.particleNum[0] += self.insert_body_num[None] * template_ptr.nspheres

    def GenerateMultiSphere(self, min_rad, max_rad, actual_body, start_pebble_num, template_ptr: ClumpTemplate, set_orientations):
        if self.is_poission:
            if self.insert_particle_in_neighbor[None] - start_pebble_num == 0:
                position = self.region.start_point + 0.5 * self.region.region_size
                radius = 0.5 * (max_rad + min_rad)
            else:
                location = self.insert_particle_in_neighbor[None] - 1
                position = self.neighbor.position[location]
                radius = self.neighbor.radius[location]

            kernel_insert_first_multisphere_(self.region.start_point, template_ptr.nspheres, template_ptr.r_equiv, template_ptr.x_pebble, template_ptr.rad_pebble, position, radius, self.insert_body_num, self.insert_particle_in_neighbor, 
                                             self.clump_coords, self.clump_radii, self.clump_orients, set_orientations.get_orientation, self.pebble_coords, self.pebble_radii, self.neighbor.cell_num, self.neighbor.cell_size, self.neighbor.position, 
                                             self.neighbor.radius, self.neighbor.num_particle_in_cell, self.neighbor.particle_neighbor, self.neighbor.insert_particle)
            kernel_multisphere_possion_sampling_(template_ptr, min_rad, max_rad, self.tries_number, actual_body, self.region.start_point, self.insert_body_num, self.insert_particle_in_neighbor, self.clump_coords, self.clump_radii, self.clump_orients, 
                                                 set_orientations.get_orientation, self.pebble_coords, self.pebble_radii, self.neighbor.cell_num, self.neighbor.cell_size, self.neighbor.position, self.neighbor.radius, self.neighbor.num_particle_in_cell, 
                                                 self.neighbor.particle_neighbor, self.region.function, self.neighbor.overlap, self.neighbor.insert_particle)
        elif not self.is_poission:
            kernel_multisphere_generate_without_overlap_(template_ptr.nspheres, template_ptr.r_equiv, template_ptr.x_pebble, template_ptr.rad_pebble, min_rad, max_rad, self.tries_number, actual_body, self.insert_body_num, self.insert_particle_in_neighbor, 
                                                         self.clump_coords, self.clump_radii, self.clump_orients, set_orientations.get_orientation, self.pebble_coords, self.pebble_radii, self.neighbor.cell_num, self.neighbor.cell_size, self.neighbor.position, 
                                                         self.neighbor.radius, self.neighbor.num_particle_in_cell, self.neighbor.particle_neighbor, self.region.function, self.region.start_point, self.region.region_size, self.neighbor.overlap, self.neighbor.insert_particle)

    def distribute_multispheres(self, scene: myScene):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        clump = scene.get_clump_ptr()
        sphereNum = int(scene.sphereNum[0])
        clumpNum = int(scene.clumpNum[0])

        if self.porosity >= 0.345:
            self.region.calculate_expected_particle_volume(self.porosity)
        else:
            raise RuntimeError(f"Porosity should be set in the dictionary: {self.name}!")

        total_fraction, insert_particle = 0., 0
        if type(self.template_dict) is dict:
            name = DictIO.GetEssential(self.template_dict, "Name")
            template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
            fraction = DictIO.GetAlternative(self.template_dict, "Fraction", 1.0)
            min_rad = DictIO.GetEssential(self.template_dict, "MinRadius", "Radius")
            template_vol = self.region.estimate_body_volume(fraction)
            insert_particle += math.ceil(template_vol / (4./3. * PI * min_rad * min_rad * min_rad)) * template_ptr.nspheres
            total_fraction += fraction
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                name = DictIO.GetEssential(temp, "Name")
                template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
                fraction = DictIO.GetAlternative(temp, "Fraction", 1.0)
                min_rad = DictIO.GetEssential(temp, "MinRadius", "Radius")
                template_vol = self.region.estimate_body_volume(fraction)
                insert_particle += math.ceil(template_vol / (4./3. * PI * min_rad * min_rad * min_rad)) * template_ptr.nspheres
                total_fraction += fraction
        if total_fraction < 0. or total_fraction > 1.: 
            raise ValueError("Fraction value error")
        self.hist_check_by_volume(particle, sphere, clump, sphereNum, clumpNum, total_fraction)
        self.region.estimate_body_volume(total_fraction)
        self.allocate_clump_memory(insert_particle, distribute=True)

        if type(self.template_dict) is dict:
            self.distribute_template_spheres(scene, self.template_dict)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.distribute_template_spheres(scene, temp)
        
    def distribute_template_multispheres(self, scene: myScene, template): 
        particle = scene.get_particle_ptr()
        clump = scene.get_clump_ptr()
        material = scene.get_material_ptr()
        clumpNum = int(scene.clumpNum[0])
        particleNum = int(scene.particleNum[0])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)

        fraction = DictIO.GetAlternative(template, "Fraction", 1.0)
        min_radius = DictIO.GetEssential(template, "MinRadius", "Radius")
        max_radius = DictIO.GetEssential(template, "MaxRadius", "Radius")   
        particle_orientation=DictIO.GetAlternative(template, "BodyOrientation", None)
        orientation_parameter = DictIO.GetAlternative(template, "OrientationParameter", None)
        
        set_orientations = set_orientation()
        set_orientations.set_orientation(particle_orientation, orientation_parameter) 
        actual_volume = fraction * self.region.expected_particle_volume
        start_body_num, start_pebble_num =  self.insert_body_num[None], self.insert_particle_in_neighbor[None]
        insert_volume = self.DistributeMultiSphere(min_radius, max_radius, actual_volume, template_ptr, set_orientations)
        end_body_num, end_pebble_num = self.insert_body_num[None], self.insert_particle_in_neighbor[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_pebble_num
        
        self.rotate_clump_packing(start_body_num, end_body_num, start_pebble_num, end_pebble_num, template_ptr.nspheres)
        if self.write_file:
            self.write_clump_text(template_ptr, start_body_num, end_body_num, particleNum)
        else:
            groupID = DictIO.GetEssential(template, "GroupID")
            matID = DictIO.GetEssential(template, "MaterialID")
            init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
            init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
            scene.check_particle_num(self.sims, particle_number=self.insert_particle_in_neighbor[None] - self.region.inserted_particle_num)
            scene.check_clump_number(self.sims, body_number=self.insert_body_num[None])
            kernel_add_multisphere_packing(particle, clump, material, clumpNum, particleNum, start_body_num, end_body_num, start_pebble_num, template_ptr,  
                                           self.clump_coords, self.clump_radii, self.clump_orients, self.pebble_coords, self.pebble_radii, groupID, matID, init_v, init_w)
            print(" Clump(s) Information ".center(71, '-'))
            self.print_particle_info(groupID, matID, init_v, init_w, body_num=body_count, insert_volume=insert_volume)
        scene.clumpNum[0] += body_count
        scene.particleNum[0] += body_count * template_ptr.nspheres

    def DistributeMultiSphere(self, min_rad, max_rad, actual_volume, template_ptr: ClumpTemplate, set_orientations):
        return kernel_distribute_multisphere_(template_ptr.nspheres, template_ptr.r_equiv, template_ptr.volume_expect, template_ptr.x_pebble, template_ptr.rad_pebble, min_rad, max_rad, actual_volume, self.insert_body_num, 
                                              self.insert_particle_in_neighbor, self.clump_coords, self.clump_radii, self.clump_orients, set_orientations.get_orientation, self.pebble_coords, self.pebble_radii, 
                                              self.region.start_point, self.region.region_size, self.region.function)

    def rotate_clump_packing(self, start_body_num, end_body_num, start_pebble_num, end_pebble_num, nsphere):
        kernel_position_rotate_(self.region.zdirection, self.region.rotate_center, self.clump_coords, start_body_num, end_body_num)
        kernel_position_rotate_(self.region.zdirection, self.region.rotate_center, self.pebble_coords, start_pebble_num, end_pebble_num)

    def write_clump_text(self, template_ptr: ClumpTemplate, start_body_num, end_body_num, particleNum):
        print('#', "Writing clump(s) into 'ClumpPacking' ......")
        body_num = end_body_num - start_body_num
        print(f"Inserted Clump Number: {body_num}")
        print(f"Inserted Particle Number: {body_num * template_ptr.nspheres}", '\n')
        
        position = self.clump_coords.to_numpy()[start_body_num: end_body_num]
        radius = self.clump_radii.to_numpy()[start_body_num: end_body_num]
        orientation = self.clump_orients.to_numpy()[start_body_num: end_body_num]
        inertia_vol = np.outer((radius / template_ptr.r_equiv) ** 5, template_ptr.inertia)
        pebbleIndex1 = np.arange(particleNum, particleNum + body_num * template_ptr.nspheres, template_ptr.nspheres)
        pebbleIndex2 = np.arange(particleNum + template_ptr.nspheres, particleNum + (body_num + 1) * template_ptr.nspheres, template_ptr.nspheres) - 1
        
        particle_pos = self.pebble_coords.to_numpy()[particleNum: particleNum + body_num * template_ptr.nspheres]
        particle_rad = self.pebble_radii.to_numpy()[particleNum: particleNum + body_num * template_ptr.nspheres]
        multisphereIndex = np.repeat(np.arange(start_body_num, end_body_num, 1), template_ptr.nspheres)

        if not os.path.exists("ClumpPacking.txt"):
            np.savetxt('ClumpPacking.txt', np.column_stack((position, radius, orientation, inertia_vol, pebbleIndex1, pebbleIndex2)), header="     PositionX            PositionY                PositionZ                Radius                OrientX                OrientY                OrientZ                moi_volX                moi_volY                moi_volZ                startIndex                endIndex", delimiter=" ")
        else:
            with open('ClumpPacking.txt', 'ab') as file:
                np.savetxt(file, np.column_stack((position, radius, orientation, inertia_vol, pebbleIndex1, pebbleIndex2)), delimiter=" ")

        if not os.path.exists("PebblePacking.txt"):
            np.savetxt('PebblePacking.txt', np.column_stack((particle_pos, particle_rad, multisphereIndex)), header="     PositionX            PositionY                PositionZ                Radius                multisphereIndex", delimiter=" ")
        else:
            with open('PebblePacking.txt', 'ab') as file:
                np.savetxt(file, np.column_stack((particle_pos, particle_rad, multisphereIndex)), delimiter=" ")

@ti.data_oriented
class set_orientation:
    def __init__(self) -> None:
        self.fix_orient = ti.Vector.field(3, float, shape=())

    @ti.kernel
    def fix_orientation(self, orient: ti.types.vector(3, float)):
        self.fix_orient[None] = orient

    def set_orientation(self, particle_orientation, orientation_parameter):
        if particle_orientation: 
            if particle_orientation == 'constant':
                if orientation_parameter:
                    self.fix_orientation(orientation_parameter.normalized())
                else:
                    self.fix_orientation(vec3f([0, 0, 1]))
                self.get_orientation = self.get_fixed_orientation
            elif particle_orientation == 'uniform':
                self.get_orientation = self.get_uniform_orientation
            elif particle_orientation == 'gaussian': 
                self.get_orientation = self.get_uniform_orientation
            else:
                raise ValueError("Orientation distribution error!")
        else:
            self.fix_orientation(vec3f([0, 0, 1]))
            self.get_orientation = self.get_fixed_orientation

    @ti.func
    def get_fixed_orientation(self):
        return vec3f(self.fix_orient[None][0], self.fix_orient[None][1], self.fix_orient[None][2])

    @ti.func
    def get_uniform_orientation(self):
        return vec3f([2*(ti.random()-0.5), 2*(ti.random()-0.5), 2*(ti.random()-0.5)]).normalized()
