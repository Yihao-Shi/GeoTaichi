import math
import os

import numpy as np
import taichi as ti

from src.dem.generator.BrustNeighbor import BruteSearch
from src.dem.generator.InsertionKernel import *
from src.dem.generator.LinkedCellNeighbor import LinkedCell
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.dem.generator.ClumpTemplate import ClumpTemplate
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.sorting.ParallelSort import parallel_sort_with_value
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefination import vec3f, vec3i
from src.utils.Orientation import set_orientation
from src.mesh.TetraMesh import TetraMesh
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
        elif btype == "RigidBody":
            self.create_rigid_body(sims, scene, template)
        elif btype == "SoftBody":
            self.create_soft_body(sims, scene, template)
        else:
            lists = ["Sphere", "Clump", "RigidBody"]
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

    def set_template(self, template_ptr):
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
        equiv_rad = DictIO.GetAlternative(template, "Radius", None)
        scale_factor = DictIO.GetAlternative(template, "ScaleFactor", None)
        orientation = DictIO.GetAlternative(template, "BodyOrientation", None)
        set_orientations = set_orientation(orientation)

        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
      
        if equiv_rad is None and isinstance(scale_factor, (float, int)):
            equiv_rad = float(scale_factor) * template_ptr.r_equiv
        elif scale_factor is None and isinstance(equiv_rad, (float, int)):
            scale_factor = float(equiv_rad) / template_ptr.r_equiv
        else:
            raise RuntimeError("Keyword conflict!, You should set either Keyword:: /Radius/ or Keyword:: /ScaleFactor/.")
        
        scene.check_particle_num(sims, particle_number=template_ptr.nspheres)
        scene.check_clump_number(sims, body_number=1)
        kernel_create_multisphere_(particle, clump, material, bodyNum, particleNum, template_ptr.nspheres, scale_factor, template_ptr.inertia, template_ptr.x_pebble, 
                                   template_ptr.rad_pebble, com_pos, equiv_rad, set_orientations.get_orientation, groupID, matID, init_v, init_w)
        print(" Clump Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, com_pos, init_v, init_w, name=name)
        scene.clumpNum[0] += 1
        scene.particleNum[0] += template_ptr.nspheres

    def create_rigid_body(self, sims, scene: myScene, template):
        if type(template) is dict:
            self.create_template_rigid_body(sims, scene, template)
        elif type(template) is list:
            for temp in template:
                self.create_template_rigid_body(sims, scene, temp)

    def create_template_rigid_body(self, sims: Simulation, scene: myScene, template):
        bounding_sphere = scene.get_bounding_sphere()
        bounding_box = scene.get_bounding_box()
        master = scene.get_surface()
        rigid_body = scene.get_rigid_ptr()
        material = scene.get_material_ptr()
        particleNum = int(scene.particleNum[0])
        surfaceNum = int(scene.surfaceNum[0])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)
        index = DictIO.GetEssential(scene.prefixID, name)
        com_pos = DictIO.GetEssential(template, "BodyPoint")
        equiv_rad = DictIO.GetAlternative(template, "Radius", None)
        bounding_rad = DictIO.GetAlternative(template, "BoundingRadius", None)
        scale_factor = DictIO.GetAlternative(template, "ScaleFactor", None) 
        orientation = DictIO.GetAlternative(template, "BodyOrientation", None)
        set_orientations = set_orientation(orientation)

        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.GetAlternative(template, "FixMotion", ["Free", "Free", "Free"])
        is_fix = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_str])

        if isinstance(scale_factor, (float, int)):
            equiv_rad = float(scale_factor) * template_ptr.objects.eqradius
        elif isinstance(equiv_rad, (float, int)):
            scale_factor = float(equiv_rad) / template_ptr.objects.eqradius
        elif isinstance(bounding_rad, (float, int)):
            equiv_rad = template_ptr.objects.eqradius / template_ptr.boundings.r_bound * bounding_rad
            scale_factor = float(equiv_rad) / template_ptr.objects.eqradius
        else:
            raise RuntimeError("Keyword conflict!, You should set either Keyword:: /Radius/ or Keyword:: /ScaleFactor/.")

        if sims.scheme == "PolySuperEllipsoid" or sims.scheme == "PolySuperQuadrics":
            if sims.iterative_model == "LagrangianMultiplier":
                if sims.scheme == "PolySuperEllipsoid":
                    assert 0.25 <= template_ptr.objects.physical_parameter["epsilon_e"] <= 1, f"The optimal value range of parameter /epsilon_e/ is 0.25 to 1"
                    assert 0.25 <= template_ptr.objects.physical_parameter["epsilon_n"] <= 1, f"The optimal value range of parameter /epsilon_n/ is 0.25 to 1"
                elif sims.scheme == "PolySuperQuadrics":
                    assert 0.25 <= template_ptr.objects.physical_parameter["epsilon_x"] <= 1, f"The optimal value range of parameter /epsilon_x/ is 0.25 to 1"
                    assert 0.25 <= template_ptr.objects.physical_parameter["epsilon_y"] <= 1, f"The optimal value range of parameter /epsilon_y/ is 0.25 to 1"
                    assert 0.25 <= template_ptr.objects.physical_parameter["epsilon_z"] <= 1, f"The optimal value range of parameter /epsilon_z/ is 0.25 to 1"
            template_id = scene.prefixID[name]
            scene.check_rigid_body_number(sims, rigid_body_number=1)
            kernel_create_implicit_surface_rigid_body_(rigid_body, bounding_sphere, material, particleNum, template_id, template_ptr.boundings.r_bound, vec3f(template_ptr.boundings.x_bound), 
                                                scale_factor, vec3f(template_ptr.objects.inertia), com_pos, equiv_rad, set_orientations.get_orientation, groupID, matID, init_v, init_w, is_fix)
            print(" Implicit surface body Information ".center(71, '-'))
        else:
            gridNum = scene.gridID[index]
            verticeNum = scene.verticeID[index]
            scene.check_rigid_body_number(sims, rigid_body_number=1)
            kernel_create_level_set_rigid_body_(rigid_body, bounding_box, bounding_sphere, master, material, particleNum, gridNum, verticeNum, surfaceNum, vec3f(template_ptr.objects.grid.minBox()), vec3f(template_ptr.objects.grid.maxBox()), 
                                                template_ptr.boundings.r_bound, vec3f(template_ptr.boundings.x_bound), template_ptr.surface_node_number, template_ptr.objects.grid.grid_space, vec3i(template_ptr.objects.grid.gnum), 
                                                template_ptr.objects.grid.extent, scale_factor, vec3f(template_ptr.objects.inertia), com_pos, equiv_rad, set_orientations.get_orientation, groupID, matID, init_v, init_w, is_fix)
            
            print(" Level set body Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, com_pos, init_v, init_w, fix_v=is_fix, fix_w=is_fix, name=name)
        
        faces = scene.add_connectivity(1, template_ptr.surface_node_number, template_ptr.objects)
        scene.particleNum[0] += 1
        scene.rigidNum[0] += 1
        scene.surfaceNum[0] += template_ptr.surface_node_number

    def create_soft_body(self, sims, scene, template):
        if type(template) is dict:
            self.create_template_soft_body(sims, scene, template)
        elif type(template) is list:
            for temp in template:
                self.create_template_soft_body(sims, scene, temp)

    def create_template_soft_body(self, sims, scene: myScene, template):
        bounding_sphere = scene.get_bounding_sphere()
        bounding_box = scene.get_bounding_box()
        master = scene.get_surface()
        rigid_body = scene.get_rigid_ptr()
        material = scene.get_material_ptr()
        particleNum = int(scene.particleNum[0])
        surfaceNum = int(scene.surfaceNum[0])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: GeneralShapeTemplate = DictIO.GetEssential(self.dgenerator.myTemplate, name)
        com_pos = DictIO.GetEssential(template, "BodyPoint")
        equiv_rad = DictIO.GetAlternative(template, "Radius", None)
        bounding_rad = DictIO.GetAlternative(template, "BoundingRadius", None)
        scale_factor = DictIO.GetAlternative(template, "ScaleFactor", None) 
        orientation = DictIO.GetAlternative(template, "BodyOrientation", None)
        total_points = DictIO.GetAlternative(template, "TotalParticles", 1000)
        set_orientations = set_orientation(orientation)

        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.GetAlternative(template, "FixMotion", ["Free", "Free", "Free"])
        is_fix = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_str])

        if isinstance(scale_factor, (float, int)):
            equiv_rad = float(scale_factor) * template_ptr.objects.eqradius
        elif isinstance(equiv_rad, (float, int)):
            scale_factor = float(equiv_rad) / template_ptr.objects.eqradius
        elif isinstance(bounding_rad, (float, int)):
            equiv_rad = template_ptr.objects.eqradius / template_ptr.boundings.r_bound * bounding_rad
            scale_factor = float(equiv_rad) / template_ptr.objects.eqradius
        else:
            raise RuntimeError("Keyword conflict!, You should set either Keyword:: /Radius/ or Keyword:: /ScaleFactor/.")

        '''tetra = TetraMesh()
        scene.check_soft_body_number(sims, body_number=1)
        kernel_create_level_set_soft_body_(rigid_body, bounding_box, bounding_sphere, master, material, particleNum, gridNum, verticeNum, surfaceNum, vec3f(template_ptr.objects.grid.minBox()), vec3f(template_ptr.objects.grid.maxBox()), 
                                           template_ptr.boundings.r_bound, vec3f(template_ptr.boundings.x_bound), template_ptr.surface_node_number, template_ptr.objects.grid.grid_space, vec3i(template_ptr.objects.grid.gnum), 
                                           template_ptr.objects.grid.extent, scale_factor, vec3f(template_ptr.objects.inertia), com_pos, equiv_rad, set_orientations.get_orientation, groupID, matID, init_v, init_w, is_fix)
        
        print(" Level set body Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, com_pos, init_v, init_w, fix_v=is_fix, fix_w=is_fix, name=name)
        faces = np.array(template_ptr.objects.mesh.faces, dtype=np.int32)
        scene.connectivity = np.append(scene.connectivity, faces + scene.surfaceNum[0]).reshape(-1, 3)
        scene.particleNum[0] += 1
        scene.softNum[0] += 1
        scene.surfaceNum[0] += template_ptr.surface_node_number'''

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

        self.faces = np.array([], dtype=np.int32)

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
        period = DictIO.GetAlternative(body_dict, "Period", [self.sims.current_time, self.sims.current_time, 1e6])
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

    def set_template(self, template_ptr):
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
            print("Actual Void Fraction: ", 1 - insert_volume / self.region.cal_volume())
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
            elif self.type == "Lattice":
                self.lattice_spheres(scene)
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
        elif self.btype == "RigidBody":
            if self.type == "Generate":
                self.generate_LSbodys(scene)
            elif self.type == "Lattice":
                self.lattice_LSbodys(scene)
            else:
                lists = ["Generate", "Distribute"]
                raise RuntimeError(f"Invalid Keyword:: /GenerateType/: {self.type}. Only the following {lists} are valid")
        else:
            lists = ["Sphere", "Clump", "RigidBody", "SoftBody"]
            raise RuntimeError(f"Invalid Keyword:: /BodyType/: {self.btype}. Only the following {lists} are valid")
        
        if self.visualize and not scene.particle is None and not self.write_file:
            self.scene_visualization(scene)
        elif self.visualize and self.write_file:
            self.generator_visualization()

        self.reset()
        
        if self.sims.current_time + self.insert_interval > self.end_time or self.insert_interval > self.sims.time or \
            self.end_time == 0 or self.start_time > self.end_time:
            self.deactivate()
            if self.end_time > self.sims.time: self.end_time = 1. * self.sims.time
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
        elif self.btype == "RigidBody":
            print('#', "Start adding Level-set(s) ......")
            self.add_levelsets_to_scene(scene)
        self.reset()

        if self.sims.current_time + self.insert_interval > self.end_time:
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
    def allocate_sphere_memory(self, sphere_num, generate=False, distribute=False, levelset=False, lattice=False):
        field_builder = ti.FieldsBuilder()
        self.sphere_coords = ti.Vector.field(3, float)
        self.sphere_radii = ti.field(float)
        if levelset:
            self.orients = ti.Vector.field(3, float)
            if lattice:
                self.valid = ti.field(int)
                field_builder.dense(ti.i, sphere_num).place(self.sphere_coords, self.sphere_radii, self.orients, self.valid)
            else:
                field_builder.dense(ti.i, sphere_num).place(self.sphere_coords, self.sphere_radii, self.orients)
        else:
            if lattice:
                self.valid = ti.field(int)
                field_builder.dense(ti.i, sphere_num).place(self.sphere_coords, self.sphere_radii, self.valid)
            else:
                field_builder.dense(ti.i, sphere_num).place(self.sphere_coords, self.sphere_radii)
        self.snode_tree = field_builder.finalize()
        if lattice: self.remains = sphere_num
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
        self.hist_check_by_number(particle, sphere, clump, sphereNum, clumpNum, insert_num)
        particle_in_region = insert_num + self.region.inserted_particle_num

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
    
    def hist_check_by_number(self, particle, sphere, clump, sphereNum, clumpNum, insert_num):
        if insert_num > 0.:
            if self.check_hist and self.region.inserted_particle_num == 0:
                reval = vec2i([0, 0])
                if sphere: reval += kernel_update_particle_number_by_sphere_(sphereNum, sphere, particle, self.region.function)
                if clump: reval += kernel_update_pebble_number_by_clump_(clumpNum, clump, particle, self.region.function)
                self.region.add_inserted_body(reval[0])
                self.region.add_inserted_particle(reval[1])
        else:
            raise RuntimeError(f"BodyNumber should be set in the dictionary: {self.name}!")
    
    def hist_check_by_volume(self, particle, sphere, clump, sphereNum, clumpNum):
        if self.check_hist and self.region.inserted_particle_num == 0 and (sphere or clump):
            reval = vec2f([0, 0])
            if sphere: reval += kernel_update_particle_volume_by_sphere_(sphereNum, sphere, particle, self.region.function)
            if clump: reval += kernel_update_particle_volume_by_clump_(clumpNum, clump, particle, self.region.function)
            self.region.add_inserted_particle_volume(reval[0])
            self.region.add_inserted_particle(reval[1])

    def generate_template_spheres(self, scene: myScene, template): 
        actual_body = DictIO.GetEssential(template, "BodyNumber")
        min_radius = DictIO.GetEssential(template, "MinRadius", "Radius")
        max_radius = DictIO.GetEssential(template, "MaxRadius", "Radius")   

        if min_radius > max_radius:
            raise RuntimeError("Keyword:: /MinRadius/ must not be larger than /MaxRadius/")

        start_body_num = self.insert_body_num[None]
        self.GenerateSphere(min_radius, max_radius, actual_body, start_body_num)
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num
        
        # self.rotate_sphere_packing(start_body_num, end_body_num)
        if self.write_file:
            self.write_sphere_text(start_body_num, end_body_num)
        else:
            self.insert_sphere(scene, template, start_body_num, end_body_num, body_count)

    def lattice_spheres(self, scene: myScene):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        clump = scene.get_clump_ptr()
        sphereNum = int(scene.sphereNum[0])
        clumpNum = int(scene.clumpNum[0])

        min_rad, max_rad, total_fraction = 0., 0., 0.
        if type(self.template_dict) is dict:
            fraction = DictIO.GetAlternative(self.template_dict, "Fraction", 1.0)
            min_rad = DictIO.GetEssential(self.template_dict, "MinRadius", "Radius") 
            max_rad = DictIO.GetEssential(self.template_dict, "MaxRadius", "Radius") 
            total_fraction += fraction
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                fraction = DictIO.GetAlternative(temp, "Fraction", 1.0)
                min_rad = min(min_rad, DictIO.GetEssential(temp, "MinRadius", "Radius"))
                max_rad = max(max_rad, DictIO.GetEssential(temp, "MaxRadius", "Radius"))
                total_fraction += fraction
        if total_fraction < 0. or total_fraction > 1.: 
            raise ValueError("Fraction value error")
        
        insert_particle = np.floor(0.5 * np.array(self.region.region_size) / max_rad).astype(np.int32)
        insertNum = int(insert_particle[0] * insert_particle[1] * insert_particle[2])
        self.hist_check_by_number(particle, sphere, clump, sphereNum, clumpNum, insertNum)
        particle_in_region = insertNum + self.region.inserted_particle_num

        if particle_in_region < 1000:
            if self.neighbor is None: self.neighbor = BruteSearch()
            self.neighbor.neighbor_init(particle_in_region)
        elif particle_in_region >= 1000:
            if self.neighbor is None: self.neighbor = LinkedCell()
            self.neighbor.neighbor_init(min_rad, max_rad, self.region.region_size, particle_in_region)
        self.allocate_sphere_memory(insertNum, generate=True, levelset=True, lattice=True)

        if self.check_hist and self.region.inserted_particle_num > 0:
            self.neighbor.pre_neighbor_sphere(sphereNum, self.insert_particle_in_neighbor, particle, sphere, self.neighbor.position, self.neighbor.radius, self.region.function, self.region.start_point)
            self.neighbor.pre_neighbor_clump(clumpNum, self.insert_particle_in_neighbor, particle, clump, self.neighbor.position, self.neighbor.radius, self.region.function, self.region.start_point)

        if type(self.template_dict) is dict:
            self.lattice_template_sphere(scene, self.template_dict, insert_particle)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.lattice_template_sphere(scene, temp, insert_particle)

    def lattice_template_sphere(self, scene: myScene, template, insert_particle):
        fraction = DictIO.GetAlternative(template, "Fraction", 1.0)
        min_radius = DictIO.GetEssential(template, "MinRadius", "Radius") 
        max_radius = DictIO.GetEssential(template, "MaxRadius", "Radius") 

        actual_body = int(fraction * int(insert_particle[0] * insert_particle[1] * insert_particle[2]))
        start_body_num =  self.insert_body_num[None]
        self.LatticeSphere(min_radius, max_radius, actual_body, start_body_num, insert_particle)
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num

        if self.write_file:
            self.write_sphere_text(start_body_num, end_body_num)
        else:
            self.insert_sphere(scene, template, start_body_num, end_body_num, body_count)

    def add_spheres_to_scene(self, scene: myScene):
        if type(self.template_dict) is dict:
            self.insert_sphere(scene, self.template_dict, 0, self.insert_body_num[None], self.insert_body_num[None])
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.insert_sphere(scene, temp, 0, self.insert_body_num[None], self.insert_body_num[None])

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
            kernel_sphere_poisson_sampling_(min_rad, max_rad, self.tries_number, actual_body + start_body_num, self.region.start_point, self.insert_body_num, self.insert_particle_in_neighbor, self.sphere_coords, 
                                            self.sphere_radii, self.neighbor.cell_num, self.neighbor.cell_size, self.neighbor.position, self.neighbor.radius, self.neighbor.num_particle_in_cell, self.neighbor.particle_neighbor, 
                                            self.region.function, self.neighbor.overlap, self.neighbor.insert_particle)
        elif not self.is_poission:
            kernel_sphere_generate_without_overlap_(min_rad, max_rad, self.tries_number, actual_body + start_body_num, self.region.start_point, self.region.region_size, self.insert_body_num, self.insert_particle_in_neighbor, 
                                                    self.sphere_coords, self.sphere_radii, self.neighbor.cell_num, self.neighbor.cell_size, self.neighbor.position, self.neighbor.radius, self.neighbor.num_particle_in_cell, 
                                                    self.neighbor.particle_neighbor, self.region.function, self.neighbor.overlap, self.neighbor.insert_particle)

    def LatticeSphere(self, min_rad, max_rad, actual_body, start_body_num, insert_particle):
        if self.insert_body_num[None] == 0:
            fill_valid(self.valid)
        kernel_sphere_generate_lattice_(min_rad, max_rad, actual_body + start_body_num, vec3i(insert_particle), self.region.start_point, self.valid, self.insert_body_num, self.insert_particle_in_neighbor, 
                                        self.sphere_coords, self.sphere_radii, self.neighbor.cell_num, self.neighbor.cell_size, self.neighbor.position, self.neighbor.radius, self.neighbor.num_particle_in_cell, 
                                        self.neighbor.particle_neighbor, self.region.function, self.neighbor.overlap, self.neighbor.insert_particle)
        self.remains = update_valid(self.remains, self.valid)

    def distribute_spheres(self, scene: myScene):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        clump = scene.get_clump_ptr()
        sphereNum = int(scene.sphereNum[0])
        clumpNum = int(scene.clumpNum[0])

        if self.porosity >= 0.2594:
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
        
        # self.rotate_sphere_packing(start_body_num, end_body_num)
        if self.write_file:
            self.write_sphere_text(start_body_num, end_body_num)
        else:
            self.insert_sphere(scene, template, start_body_num, end_body_num, body_count, insert_volume)

    def insert_sphere(self, scene: myScene, template, start_body_num, end_body_num, body_count, insert_volume=0):
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
        scene.check_particle_num(self.sims, particle_number=body_count)
        scene.check_sphere_number(self.sims, body_number=body_count)
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
    def allocate_clump_memory(self, clump_num, insert_particle, generate=False, distribute=False):
        field_builder = ti.FieldsBuilder()
        self.clump_coords = ti.Vector.field(3, float)
        self.clump_radii = ti.field(float)
        self.clump_orients = ti.Vector.field(3, float)
        self.pebble_coords = ti.Vector.field(3, float)
        self.pebble_radii = ti.field(float)
        field_builder.dense(ti.i, clump_num).place(self.clump_coords, self.clump_radii, self.clump_orients)
        field_builder.dense(ti.i, insert_particle).place(self.pebble_coords, self.pebble_radii)
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

        insert_num, insert_particle, min_radius, max_radius = 0, 0, [], []
        if type(self.template_dict) is dict:
            name = DictIO.GetEssential(self.template_dict, "Name")
            template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
            body_num = DictIO.GetEssential(self.template_dict, "BodyNumber")
            scale_factor_min = DictIO.GetEssential(self.template_dict, "MinRadius", "Radius") / template_ptr.r_equiv
            scale_factor_max = DictIO.GetEssential(self.template_dict, "MaxRadius", "Radius") / template_ptr.r_equiv
            min_radius.append(scale_factor_min * template_ptr.pebble_radius_min)
            max_radius.append(scale_factor_max * template_ptr.pebble_radius_max)
            insert_num += body_num
            insert_particle += body_num * template_ptr.nspheres
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                name = DictIO.GetEssential(temp, "Name")
                template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
                body_num = DictIO.GetEssential(temp, "BodyNumber")
                scale_factor_min = DictIO.GetEssential(temp, "MinRadius", "Radius") / template_ptr.r_equiv
                scale_factor_max = DictIO.GetEssential(temp, "MaxRadius", "Radius") / template_ptr.r_equiv
                min_radius.append(scale_factor_min * template_ptr.pebble_radius_min)
                max_radius.append(scale_factor_max * template_ptr.pebble_radius_max)
                insert_num += body_num
                insert_particle += body_num * template_ptr.nspheres
        self.hist_check_by_number(particle, sphere, clump, sphereNum, clumpNum, insert_num)
        particle_in_region = insert_particle + self.region.inserted_particle_num

        if particle_in_region < 1000:
            if self.neighbor is None: self.neighbor = BruteSearch()
            self.neighbor.neighbor_init(particle_in_region)
        elif particle_in_region >= 1000:
            if self.neighbor is None: self.neighbor = LinkedCell()
            self.neighbor.neighbor_init(min(min_radius), max(max_radius), self.region.region_size, particle_in_region)
        self.allocate_clump_memory(insert_num, insert_particle, generate=True)

        if self.check_hist and self.region.inserted_particle_num > 0:
            self.neighbor.pre_neighbor_sphere(sphereNum, self.insert_particle_in_neighbor, particle, sphere, self.neighbor.position, self.neighbor.radius, self.region.function, self.region.start_point)
            self.neighbor.pre_neighbor_clump(clumpNum, self.insert_particle_in_neighbor, particle, clump, self.neighbor.position, self.neighbor.radius, self.region.function, self.region.start_point)

        if type(self.template_dict) is dict:
            self.generate_template_multispheres(scene, self.template_dict)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.generate_template_multispheres(scene, temp)
    
    def generate_template_multispheres(self, scene: myScene, template): 
        particleNum = int(scene.particleNum[0])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)

        actual_body = DictIO.GetEssential(template, "BodyNumber")
        min_radius = DictIO.GetEssential(template, "MinRadius", "Radius")
        max_radius = DictIO.GetEssential(template, "MaxRadius", "Radius")
        orientation = DictIO.GetAlternative(template, "BodyOrientation", None)
        set_orientations = set_orientation(orientation)  

        if min_radius > max_radius:
            raise RuntimeError("Keyword:: /MinRadius/ must not be larger than /MaxRadius/")
        
        start_body_num, start_pebble_num =  self.insert_body_num[None], self.insert_particle_in_neighbor[None]
        self.GenerateMultiSphere(min_radius, max_radius, actual_body + start_body_num, start_pebble_num, template_ptr, set_orientations)
        end_body_num, end_pebble_num = self.insert_body_num[None], self.insert_particle_in_neighbor[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_pebble_num
        
        # self.rotate_clump_packing(start_body_num, end_body_num, start_pebble_num, end_pebble_num, template_ptr.nspheres)
        if self.write_file:
            self.write_clump_text(template_ptr, start_body_num, end_body_num, particleNum)
        else:
            self.insert_multispheres(scene, template, start_body_num, start_pebble_num, end_body_num, end_pebble_num, body_count)

    def add_clumps_to_scene(self, scene: myScene):
        if type(self.template_dict) is dict:
            self.insert_multispheres(scene, self.template_dict, 0, 0, self.insert_body_num[None], self.insert_particle_in_neighbor[None], self.insert_body_num[None])
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.insert_multispheres(scene, temp, 0, 0, self.insert_body_num[None], self.insert_particle_in_neighbor[None], self.insert_body_num[None])

    def GenerateMultiSphere(self, min_rad, max_rad, actual_body, start_pebble_num, template_ptr: ClumpTemplate, set_orientations: set_orientation):
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
            kernel_multisphere_poisson_sampling_(template_ptr, min_rad, max_rad, self.tries_number, actual_body, self.region.start_point, self.insert_body_num, self.insert_particle_in_neighbor, self.clump_coords, self.clump_radii, self.clump_orients, 
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

        if self.porosity >= 0.01:
            self.region.calculate_expected_particle_volume(self.porosity)
        else:
            raise RuntimeError(f"Porosity should be set in the dictionary: {self.name}!")

        total_fraction, insert_num, insert_particle = 0., 0, 0
        if type(self.template_dict) is dict:
            name = DictIO.GetEssential(self.template_dict, "Name")
            template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
            fraction = DictIO.GetAlternative(self.template_dict, "Fraction", 1.0)
            min_rad = DictIO.GetEssential(self.template_dict, "MinRadius", "Radius")
            template_vol = self.region.estimate_body_volume(fraction)
            insert_num += math.ceil(template_vol / (4./3. * PI * min_rad * min_rad * min_rad))
            insert_particle += math.ceil(template_vol / (4./3. * PI * min_rad * min_rad * min_rad)) * template_ptr.nspheres
            total_fraction += fraction
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                name = DictIO.GetEssential(temp, "Name")
                template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
                fraction = DictIO.GetAlternative(temp, "Fraction", 1.0)
                min_rad = DictIO.GetEssential(temp, "MinRadius", "Radius")
                template_vol = self.region.estimate_body_volume(fraction)
                insert_num += math.ceil(template_vol / (4./3. * PI * min_rad * min_rad * min_rad))
                insert_particle += math.ceil(template_vol / (4./3. * PI * min_rad * min_rad * min_rad)) * template_ptr.nspheres
                total_fraction += fraction
        if total_fraction < 0. or total_fraction > 1.: 
            raise ValueError("Fraction value error")
        
        self.hist_check_by_volume(particle, sphere, clump, sphereNum, clumpNum)
        self.region.estimate_body_volume(total_fraction)
        self.allocate_clump_memory(insert_num, insert_particle, distribute=True)

        if type(self.template_dict) is dict:
            self.distribute_template_multispheres(scene, self.template_dict)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.distribute_template_multispheres(scene, temp)
        
    def distribute_template_multispheres(self, scene: myScene, template): 
        particleNum = int(scene.particleNum[0])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)

        fraction = DictIO.GetAlternative(template, "Fraction", 1.0)
        min_radius = DictIO.GetEssential(template, "MinRadius", "Radius")
        max_radius = DictIO.GetEssential(template, "MaxRadius", "Radius")   
        orientation = DictIO.GetAlternative(template, "BodyOrientation", None)
        set_orientations = set_orientation(orientation)

        actual_volume = fraction * self.region.expected_particle_volume
        start_body_num, start_pebble_num =  self.insert_body_num[None], self.insert_particle_in_neighbor[None]
        insert_volume = self.DistributeMultiSphere(min_radius, max_radius, actual_volume, template_ptr, set_orientations)
        end_body_num, end_pebble_num = self.insert_body_num[None], self.insert_particle_in_neighbor[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_pebble_num
        
        # self.rotate_clump_packing(start_body_num, end_body_num, start_pebble_num, end_pebble_num, template_ptr.nspheres)
        if self.write_file:
            self.write_clump_text(template_ptr, start_body_num, end_body_num, particleNum)
        else:
            self.insert_multispheres(scene, template, start_body_num, start_pebble_num, end_body_num, end_pebble_num, body_count, insert_volume)

    def insert_multispheres(self, scene: myScene, template, start_body_num, start_pebble_num, end_body_num, end_pebble_num, body_count, insert_volume=0):
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
        scene.check_particle_num(self.sims, particle_number=end_pebble_num - start_pebble_num)
        scene.check_clump_number(self.sims, body_number=body_count)
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

    # ========================================================= #
    #                        LEVELSET                           #
    # ========================================================= #
    def hist_check_levelset_number(self, bounding_sphere, rigidNum, insert_num):
        if insert_num > 0.:
            if self.check_hist and self.region.inserted_particle_num == 0:
                reval = vec2i([0, 0])
                if bounding_sphere: reval += kernel_update_particle_number_by_levelset_(rigidNum, bounding_sphere, self.region.function)
                self.region.add_inserted_body(reval[0])
                self.region.add_inserted_particle(reval[1])
        else:
            raise RuntimeError(f"BodyNumber should be set in the dictionary: {self.name}!")
        
    def hist_check_levelset_volume(self, bounding_sphere, rigid, rigidNum):
        if self.check_hist and self.region.inserted_particle_num == 0 and bounding_sphere:
            reval = vec2f([0, 0])
            if bounding_sphere: reval += kernel_update_particle_volume_by_levelset_(rigidNum, bounding_sphere, rigid, self.region.function)
            self.region.add_inserted_particle_volume(reval[0])
            self.region.add_inserted_particle(reval[1])

    def GetOrient(self, start_body_num, end_body_num, orient_function):
        kernel_get_orient(start_body_num, end_body_num, orient_function, self.orients)

    def get_bounding_radius(self, template_dict):
        rad_min, rad_max = 0., 0.
        if "Radius" in template_dict or ("MinRadius" in template_dict and "MaxRadius" in template_dict):
            if self.myTemplate is None:
                raise RuntimeError("The template must be set first")
            name = DictIO.GetEssential(template_dict, "Name")
            template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)
            weight = template_ptr.boundings.r_bound / template_ptr.objects.eqradius
            rad_min = DictIO.GetEssential(template_dict, "MinRadius", "Radius") * weight
            rad_max = DictIO.GetEssential(template_dict, "MaxRadius", "Radius") * weight
        elif "BoundingRadius" in template_dict or ("MinBoundingRadius" in template_dict and "MaxBoundingRadius" in template_dict):
            rad_min = DictIO.GetEssential(template_dict, "MinBoundingRadius", "BoundingRadius") 
            rad_max = DictIO.GetEssential(template_dict, "MaxBoundingRadius", "BoundingRadius") 
        else:
            raise RuntimeError("Failed reading bounding radius")
        return rad_min, rad_max
        
    def generate_LSbodys(self, scene: myScene):
        bounding_sphere = scene.get_bounding_sphere()
        particleNum = int(scene.particleNum[0])

        insert_num, min_radius, max_radius = 0, [], []
        if type(self.template_dict) is dict:
            body_num = DictIO.GetEssential(self.template_dict, "BodyNumber")
            rad_min, rad_max = self.get_bounding_radius(self.template_dict)
            min_radius.append(rad_min)
            max_radius.append(rad_max)
            insert_num += body_num
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                body_num = DictIO.GetEssential(temp, "BodyNumber")
                rad_min, rad_max = self.get_bounding_radius(temp)
                min_radius.append(rad_min)
                max_radius.append(rad_max)
                insert_num += body_num
        self.hist_check_levelset_number(bounding_sphere, particleNum, insert_num)
        particle_in_region = insert_num + self.region.inserted_particle_num

        if particle_in_region < 1000:
            if self.neighbor is None: self.neighbor = BruteSearch()
            self.neighbor.neighbor_init(particle_in_region)
        elif particle_in_region >= 1000:
            if self.neighbor is None: self.neighbor = LinkedCell()
            self.neighbor.neighbor_init(min(min_radius), max(max_radius), self.region.region_size, particle_in_region)
        self.allocate_sphere_memory(insert_num, generate=True, levelset=True)

        if self.check_hist and self.region.inserted_particle_num > 0:
            self.neighbor.pre_neighbor_bounding_sphere(particleNum, self.insert_particle_in_neighbor, bounding_sphere, self.neighbor.position, self.neighbor.radius, self.region.function, self.region.start_point)

        if type(self.template_dict) is dict:
            self.generate_template_rigid_body(scene, self.template_dict)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.generate_template_rigid_body(scene, temp)

    def generate_template_rigid_body(self, scene: myScene, template): 
        actual_body = DictIO.GetEssential(template, "BodyNumber")
        min_radius, max_radius = self.get_bounding_radius(template)
        orientation = DictIO.GetAlternative(template, "BodyOrientation", None)
        set_orientations = set_orientation(orientation)

        if min_radius > max_radius:
            raise RuntimeError("Keyword:: /MinRadius/ must not be larger than /MaxRadius/")

        start_body_num = self.insert_body_num[None]
        self.GenerateSphere(min_radius, max_radius, actual_body, start_body_num)
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num
        parallel_sort_with_value(self.sphere_radii, self.sphere_coords, start_body_num, body_count)

        self.GetOrient(start_body_num, end_body_num, set_orientations.get_orientation)
        if self.write_file:
            self.write_body_text(start_body_num, end_body_num)
        else:
            if self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                self.insert_rigid_implicit_surface(scene, template, start_body_num, end_body_num, body_count)
            else:
                self.insert_rigid_levelset(scene, template, start_body_num, end_body_num, body_count)

    def lattice_LSbodys(self, scene: myScene):
        bounding_sphere = scene.get_bounding_sphere()
        particleNum = int(scene.particleNum[0])

        min_rad, max_rad, total_fraction = 0., 0., 0.
        if type(self.template_dict) is dict:
            fraction = DictIO.GetAlternative(self.template_dict, "Fraction", 1.0)
            min_rad, max_rad = self.get_bounding_radius(self.template_dict)
            total_fraction += fraction
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                fraction = DictIO.GetAlternative(temp, "Fraction", 1.0)
                min_radius, max_radius = self.get_bounding_radius(temp)
                min_rad = min(min_rad, min_radius)
                max_rad = max(max_rad, max_radius)
                total_fraction += fraction
        if total_fraction < 0. or total_fraction > 1.: 
            raise ValueError("Fraction value error")
        
        insert_particle = np.floor(0.5 * np.array(self.region.region_size) / max_rad).astype(np.int32)
        insertNum = int(insert_particle[0] * insert_particle[1] * insert_particle[2])
        self.hist_check_levelset_number(bounding_sphere, particleNum, insertNum)
        particle_in_region = insertNum + self.region.inserted_particle_num

        if particle_in_region < 1000:
            if self.neighbor is None: self.neighbor = BruteSearch()
            self.neighbor.neighbor_init(particle_in_region)
        elif particle_in_region >= 1000:
            if self.neighbor is None: self.neighbor = LinkedCell()
            self.neighbor.neighbor_init(min_rad, max_rad, self.region.region_size, particle_in_region)
        self.allocate_sphere_memory(insertNum, generate=True, levelset=True, lattice=True)

        if self.check_hist and self.region.inserted_particle_num > 0:
            self.neighbor.pre_neighbor_bounding_sphere(particleNum, self.insert_particle_in_neighbor, bounding_sphere, self.neighbor.position, self.neighbor.radius, self.region.function, self.region.start_point)

        if type(self.template_dict) is dict:
            self.lattice_template_rigid_body(scene, self.template_dict, insert_particle)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.lattice_template_rigid_body(scene, temp, insert_particle)

    def lattice_template_rigid_body(self, scene: myScene, template, insert_particle):
        fraction = DictIO.GetAlternative(template, "Fraction", 1.0)
        min_radius, max_radius = self.get_bounding_radius(self.template_dict)
        orientation = DictIO.GetAlternative(template, "BodyOrientation", None)
        set_orientations = set_orientation(orientation)

        actual_body = int(fraction * int(insert_particle[0] * insert_particle[1] * insert_particle[2]))
        start_body_num =  self.insert_body_num[None]
        self.LatticeSphere(min_radius, max_radius, actual_body, start_body_num, insert_particle)
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num
        parallel_sort_with_value(self.sphere_radii, self.sphere_coords, start_body_num, body_count)

        self.GetOrient(start_body_num, end_body_num, set_orientations.get_orientation)
        if self.write_file:
            self.write_body_text(start_body_num, end_body_num)
        else:
            if self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                self.insert_rigid_implicit_surface(scene, template, start_body_num, end_body_num, body_count)
            else:
                self.insert_rigid_levelset(scene, template, start_body_num, end_body_num, body_count)

    def add_levelsets_to_scene(self, scene: myScene):
        if type(self.template_dict) is dict:
            if self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                self.insert_rigid_implicit_surface(scene, self.template_dict, 0, self.insert_body_num[None], self.insert_body_num[None])
            else:
                self.insert_rigid_levelset(scene, self.template_dict, 0, self.insert_body_num[None], self.insert_body_num[None])
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                if self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                    self.insert_rigid_implicit_surface(scene, temp, 0, self.insert_body_num[None], self.insert_body_num[None])
                else:
                    self.insert_rigid_levelset(scene, temp, 0, self.insert_body_num[None], self.insert_body_num[None])

    def insert_rigid_levelset(self, scene: myScene, template, start_body_num, end_body_num, body_count):
        bounding_sphere = scene.get_bounding_sphere()
        bounding_box = scene.get_bounding_box()
        surface = scene.get_surface()
        rigid_body = scene.get_rigid_ptr()
        material = scene.get_material_ptr()
        particleNum = int(scene.particleNum[0])
        surfaceNum = int(scene.surfaceNum[0])
            
        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.GetAlternative(template, "FixMotion", ["Free", "Free", "Free"])
        is_fix = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_str])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)
        index = DictIO.GetEssential(scene.prefixID, name)

        gridNum = scene.gridID[index]
        verticeNum = scene.verticeID[index]
        scene.check_rigid_body_number(self.sims, rigid_body_number=body_count)
        kernel_add_levelset_packing(rigid_body, bounding_box, bounding_sphere, surface, material, particleNum, gridNum, surfaceNum, verticeNum, vec3f(template_ptr.objects.grid.minBox()), vec3f(template_ptr.objects.grid.maxBox()), template_ptr.boundings.r_bound, 
                                    vec3f(template_ptr.boundings.x_bound), template_ptr.surface_node_number, template_ptr.objects.grid.grid_space, vec3i(template_ptr.objects.grid.gnum), template_ptr.objects.grid.extent,
                                    vec3f(template_ptr.objects.inertia), template_ptr.objects.eqradius, groupID, matID, init_v, init_w, is_fix, start_body_num, end_body_num, self.sphere_coords, self.sphere_radii, self.orients)
        print(" Level-set body Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, init_v, init_w, fix_v=is_fix, fix_w=is_fix, body_num=body_count)

        faces = scene.add_connectivity(body_count, template_ptr.surface_node_number, template_ptr.objects)
        scene.particleNum[0] += body_count
        scene.rigidNum[0] += body_count
        scene.surfaceNum[0] += template_ptr.surface_node_number * body_count
        self.faces = np.append(self.faces, faces).reshape(-1, 3)

    def insert_rigid_implicit_surface(self, scene: myScene, template, start_body_num, end_body_num, body_count):
        bounding_sphere = scene.get_bounding_sphere()
        rigid_body = scene.get_rigid_ptr()
        material = scene.get_material_ptr()
        particleNum = int(scene.particleNum[0])
            
        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.GetAlternative(template, "FixMotion", ["Free", "Free", "Free"])
        is_fix = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_str])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)
        template_id = scene.prefixID[name]

        if self.sims.iterative_model == "LagrangianMultiplier":
            if self.sims.scheme == "PolySuperEllipsoid":
                assert 0.25 <= template_ptr.objects.physical_parameter["epsilon_e"] <= 1, f"The optimal value range of parameter /epsilon_e/ is 0.25 to 1"
                assert 0.25 <= template_ptr.objects.physical_parameter["epsilon_n"] <= 1, f"The optimal value range of parameter /epsilon_n/ is 0.25 to 1"
            elif self.sims.scheme == "PolySuperQuadrics":
                assert 0.25 <= template_ptr.objects.physical_parameter["epsilon_x"] <= 1, f"The optimal value range of parameter /epsilon_x/ is 0.25 to 1"
                assert 0.25 <= template_ptr.objects.physical_parameter["epsilon_y"] <= 1, f"The optimal value range of parameter /epsilon_y/ is 0.25 to 1"
                assert 0.25 <= template_ptr.objects.physical_parameter["epsilon_z"] <= 1, f"The optimal value range of parameter /epsilon_z/ is 0.25 to 1"

        scene.check_rigid_body_number(self.sims, rigid_body_number=body_count)
        kernel_add_implicit_surface_packing(rigid_body, bounding_sphere, material, particleNum, template_id, template_ptr.boundings.r_bound, vec3f(template_ptr.boundings.x_bound), 
                                            vec3f(template_ptr.objects.inertia), template_ptr.objects.eqradius, groupID, matID, init_v, init_w, is_fix, start_body_num, end_body_num, self.sphere_coords, self.sphere_radii, self.orients)
        print(" Implicit surface body Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, init_v, init_w, fix_v=is_fix, fix_w=is_fix, body_num=body_count)

        faces = scene.add_connectivity(body_count, template_ptr.surface_node_number, template_ptr.objects)
        scene.particleNum[0] += body_count
        scene.rigidNum[0] += body_count
        scene.surfaceNum[0] += template_ptr.surface_node_number * body_count
        self.faces = np.append(self.faces, faces).reshape(-1, 3)

    def write_body_text(self, to_start, to_end):
        print('#', "Writing sphere(s) into 'BoundingSphere' ......")
        print(f"Inserted Bounding Number: {to_end - to_start}", '\n')
        position = self.sphere_coords.to_numpy()[to_start:to_end]
        radius = self.sphere_radii.to_numpy()[to_start:to_end]
        orients = self.orients.to_numpy()[to_start:to_end]
        if not os.path.exists("BoundingSphere.txt"):
            np.savetxt('BoundingSphere.txt', np.column_stack((position, radius, orients)), header="     PositionX            PositionY                PositionZ            Radius            DirX            DirY            DirZ", delimiter=" ")
        else: 
            with open('BoundingSphere.txt', 'ab') as file:
                np.savetxt(file, np.column_stack((position, radius, orients)), delimiter=" ")

