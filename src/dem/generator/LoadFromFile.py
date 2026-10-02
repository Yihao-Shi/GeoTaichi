import numpy as np
import trimesh
import os

from src.dem.generator.InsertionKernel import *
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec3f, vec3i


class ParticleReader(object):
    sims: Simulation

    def __init__(self, sims) -> None:
        self.sims = sims
        self.active = True

        self.start_time = 0.
        self.end_time = 0.
        self.next_generate_time = 0.
        self.insert_interval = 1e10
        self.file_type = None
        self.check_hist = False
        self.write_file = False
        self.template_dict = None
        self.myTemplate = None

        self.faces = np.array([], dtype=np.int32)

        self.FIX = {
                    "Free": 0,
                    "Fix": 1
                   }

    def set_system_strcuture(self, body_dict):
        self.file_type = DictIO.GetEssential(body_dict, "FileType")
        self.template_dict = DictIO.GetEssential(body_dict, "Template")
        period = DictIO.GetAlternative(body_dict, "Period", [self.sims.current_time, self.sims.current_time, 1e6])
        self.start_time = period[0]
        self.end_time = period[1]
        self.insert_interval = period[2]

    def set_template(self, template_ptr):
        self.myTemplate = template_ptr

    def get_template_ptr_by_name(self, name):
        clump_ptr = DictIO.GetOptional(self.myTemplate, name)
        if not clump_ptr:
            raise KeyError(f"Template name: {name} is not set before")
        return clump_ptr

    def deactivate(self):
        self.active = False

    def finalize(self):
        del self.sims, self.active, self.start_time, self.end_time, self.next_generate_time, self.insert_interval
        del self.check_hist, self.write_file, self.template_dict

    def begin(self, scene: myScene):
        if self.sims.current_time < self.next_generate_time: return 0
        if self.sims.current_time < self.start_time or self.sims.current_time > self.end_time: return 0
        
        if self.file_type == "TXT":
            if type(self.template_dict) is dict:
                self.load_txt_file(scene, self.template_dict)
            elif type(self.template_dict) is list:
                for temp in self.template_dict:
                    self.load_txt_file(scene, temp)
        elif self.file_type == "NPZ":
            if type(self.template_dict) is dict:
                self.load_npz_file(scene, self.template_dict)
            elif type(self.template_dict) is list:
                for temp in self.template_dict:
                    self.load_npz_file(scene, temp)
        elif self.file_type == "OBJ":
            if type(self.template_dict) is dict:
                self.load_obj_file(scene, self.template_dict)
            elif type(self.template_dict) is list:
                for temp in self.template_dict:
                    self.load_obj_file(scene, temp)
        else:
            raise RuntimeError("Invalid file type. Only the following file types are aviliable: ['TXT', 'NPZ', 'OBJ']")

        if self.sims.current_time + self.insert_interval > self.end_time or self.insert_interval > self.sims.time or \
            self.end_time == 0 or self.start_time > self.end_time:
            self.deactivate()
        else:
            self.next_generate_time = self.sims.current_time + self.insert_interval
        return 1
    
    def regenerate(self, scene: myScene):
        if self.sims.current_time < self.next_generate_time: return 0
        if self.sims.current_time < self.start_time or self.sims.current_time > self.end_time: return 0
        if self.file_type == "TXT":
            if type(self.template_dict) is dict:
                self.load_txt_file(scene, self.template_dict)
            elif type(self.template_dict) is list:
                for temp in self.template_dict:
                    self.load_txt_file(scene, temp)
        elif self.file_type == "NPZ":
            if type(self.template_dict) is dict:
                self.load_npz_file(scene, self.template_dict)
            elif type(self.template_dict) is list:
                for temp in self.template_dict:
                    self.load_npz_file(scene, temp)
        elif self.file_type == "OBJ":
            if type(self.template_dict) is dict:
                self.load_obj_file(scene, self.template_dict)
            elif type(self.template_dict) is list:
                for temp in self.template_dict:
                    self.load_obj_file(scene, temp)
        else:
            raise RuntimeError("Invalid file type. Only the following file types are aviliable: ['TXT', 'NPZ', 'OBJ']")

        if self.sims.current_time + self.insert_interval > self.end_time:
            self.deactivate()
        else:
            self.next_generate_time = self.sims.current_time + self.insert_interval
        return 1

    def print_particle_info(self, groupID, matID, init_v, init_w, fix_v=None, fix_w=None, body_num=0, particle_volume=0, name=None):
        if body_num == 0:
            raise RuntimeError("Zero Particles are inserted into region!")
        if name:
            print("Template name: ", name)
        print("Group ID = ", groupID)
        print("Material ID = ", matID)
        print("Body Number: ", body_num)
        if particle_volume != 0:
            print("Particle Volume: ", particle_volume)
        print("Initial Velocity = ", init_v)
        print("Initial Angular Velocity = ", init_w)
        if fix_v: print("Fixed Velocity = ", fix_v)
        if fix_w: print("Fixed Angular Velocity = ", fix_w)
        print('\n')

    # ========================================================= #
    #                          Txt File                         #
    # ========================================================= #
    def load_txt_file(self, scene, template):
        btype = DictIO.GetEssential(template, "BodyType")
        if btype == "Sphere":
            print('#', "Start adding sphere(s) ......")
            self.load_txt_spheres(scene, template)
        elif btype == "Clump":
            print('#', "Start adding Clump(s) ......")
            self.load_txt_multispheres(scene, template)
        elif btype == "RigidBody":
            print('#', "Start adding Level-set(s) ......")
            if self.sims.scheme == "LSDEM":
                self.load_txt_levelsets(scene, template)
            else:
                self.load_txt_implicit_surface(scene, template)
        else:
            raise ValueError("Particle type error")

    def load_txt_spheres(self, scene: myScene, template):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        material = scene.get_material_ptr()
        sphereNum = int(scene.sphereNum[0])
        particleNum = int(scene.particleNum[0])

        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        offset = DictIO.GetAlternative(template, "Translation", [0, 0, 0])
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_v_str = DictIO.GetAlternative(template, "FixVelocity", ["Free", "Free", "Free"])
        fix_w_str = DictIO.GetAlternative(template, "FixAngularVelocity", ["Free", "Free", "Free"])
        file = DictIO.GetAlternative(template, "File", "SpherePacking.txt")
        particle_num = DictIO.GetAlternative(template, "ParticleNumber", -1)
        fix_v = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_v_str])
        fix_w = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_w_str])

        if not os.path.exists(file):
            raise EOFError("Invaild particle path")
        particle_cloud = np.loadtxt(file, unpack=True, comments='#').transpose() 
        particle_num = particle_cloud.shape[0] if particle_num == -1 else min(particle_num, particle_cloud.shape[0])
        particle_cloud = particle_cloud[0:particle_num]
        body_num = particle_cloud.shape[0]

        coords = np.ascontiguousarray(particle_cloud[:, [0, 1, 2]]) + np.array(offset)
        radii = np.ascontiguousarray(particle_cloud[:, 3])
        
        scene.check_particle_num(self.sims, particle_number=particle_num)
        scene.check_sphere_number(self.sims, body_number=body_num)
        kernel_add_sphere_files(particle, sphere, material, sphereNum, particleNum, body_num, coords, radii, groupID, matID, init_v, init_w, fix_v, fix_w)
        print(" Sphere Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, init_v, init_w, fix_v=fix_v_str, fix_w=fix_w_str, body_num=body_num)

        scene.sphereNum[0] += body_num
        scene.particleNum[0] += particle_num 

    def load_txt_multispheres(self, scene: myScene, template):
        particle = scene.get_particle_ptr()
        clump = scene.get_clump_ptr()
        material = scene.get_material_ptr()
        clumpNum = int(scene.clumpNum[0])
        particleNum = int(scene.particleNum[0])

        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        offset = DictIO.GetAlternative(template, "Translation", [0, 0, 0])
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        clump_file = DictIO.GetAlternative(template, "ClumpFile", "ClumpPacking.txt")
        pebble_file = DictIO.GetAlternative(template, "PebbleFile", "PebblePacking.txt")
        particle_num = DictIO.GetAlternative(template, "ParticleNumber", -1)
        body_num = DictIO.GetAlternative(template, "BodyNumber", -1)
        
        if not os.path.exists(clump_file):
            raise EOFError("Invaild clump file path")
        if not os.path.exists(pebble_file):
            raise EOFError("Invaild pebble file path")
        
        clump_cloud = np.loadtxt(clump_file, unpack=True, comments='#').transpose()
        pebble_cloud = np.loadtxt(pebble_file, unpack=True, comments='#').transpose() 
        particle_num = pebble_cloud.shape[0] if particle_num == -1 else min(particle_num, pebble_cloud.shape[0])
        body_num = clump_cloud.shape[0] if body_num == -1 else min(body_num, clump_cloud.shape[0])
        clump_cloud = clump_cloud[0:body_num]
        pebble_cloud = pebble_cloud[0:particle_num]
        body_num = clump_cloud.shape[0]
        particle_num = pebble_cloud.shape[0]

        clump_coords = np.ascontiguousarray(clump_cloud[:, [0, 1, 2]]) + np.array(offset)
        clump_radii = np.ascontiguousarray(clump_cloud[:, 3])
        clump_orients = np.ascontiguousarray(clump_cloud[:, [4, 5, 6]])
        clump_inertia_vol = np.ascontiguousarray(clump_cloud[:, [7, 8, 9]])
        startIndex = np.ascontiguousarray(clump_cloud[:, 10])
        endIndex = np.ascontiguousarray(clump_cloud[:, 11])
        pebble_coords = np.ascontiguousarray(pebble_cloud[:, [0, 1, 2]]) + np.array(offset)
        pebble_radii = np.ascontiguousarray(pebble_cloud[:, 3])
        multisphereIndics = np.ascontiguousarray(pebble_cloud[:, 4])
        
        scene.check_particle_num(self.sims, particle_number=particle_num)
        scene.check_clump_number(self.sims, body_number=body_num)
        kernel_add_multisphere_files(particle, clump, material, clumpNum, particleNum, body_num, particle_num, 
                                     clump_coords, clump_radii, clump_orients, clump_inertia_vol, startIndex, endIndex, pebble_coords, pebble_radii, multisphereIndics,
                                     groupID, matID, init_v, init_w)
        print(" Clump Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, init_v, init_w, body_num=body_num)

        scene.clumpNum[0] += body_num
        scene.particleNum[0] += particle_num

    def load_txt_levelsets(self, scene: myScene, template):
        bounding_sphere = scene.get_bounding_sphere()
        bounding_box = scene.get_bounding_box()
        rigid_body = scene.get_rigid_ptr()
        material = scene.get_material_ptr()
        surface = scene.get_surface()
        particleNum = int(scene.particleNum[0])
        surfaceNum = int(scene.surfaceNum[0])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)
        index = DictIO.GetEssential(scene.prefixID, name)
        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        offset = DictIO.GetAlternative(template, "Translation", [0, 0, 0])
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.GetAlternative(template, "FixMotion", ["Free", "Free", "Free"])
        is_fix = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_str])
        file = DictIO.GetAlternative(template, "File", "BoundingSphere.txt")
        body_num = DictIO.GetAlternative(template, "ParticleNumber", -1)

        if not os.path.exists(file):
            raise EOFError("Invaild particle path")
        particle_cloud = np.loadtxt(file, unpack=True, comments='#').transpose()
        body_num = particle_cloud.shape[0] if body_num == -1 else min(body_num, particle_cloud.shape[0])
        particle_cloud = particle_cloud[0:body_num]
        body_num = particle_cloud.shape[0]

        coords = np.ascontiguousarray(particle_cloud[:, [0, 1, 2]]) + np.array(offset)
        radii = np.ascontiguousarray(particle_cloud[:, 3])
        orients = np.ascontiguousarray(particle_cloud[:, [4, 5, 6]])
        
        gridNum = scene.gridID[index]
        verticeNum = scene.verticeID[index]
        scene.check_rigid_body_number(self.sims, rigid_body_number=body_num)
        kernel_add_levelset_files(rigid_body, bounding_box, bounding_sphere, surface, particleNum, gridNum, surfaceNum, template_ptr.boundings.r_bound, vec3f(template_ptr.boundings.x_bound), 
                                  vec3f(template_ptr.objects.grid.minBox()), vec3f(template_ptr.objects.grid.maxBox()), template_ptr.surface_node_number, vec3f(template_ptr.objects.inertia), 
                                  template_ptr.objects.eqradius, template_ptr.objects.grid.grid_space, vec3i(template_ptr.objects.grid.gnum), template_ptr.objects.grid.extent, body_num, coords, radii, orients)
        kernel_add_rigid_body(rigid_body, material, particleNum, surfaceNum, verticeNum, body_num, template_ptr.surface_node_number, groupID, matID, init_v, init_w, is_fix)
        print(" Level-set body Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, init_v, init_w, fix_v=is_fix, fix_w=is_fix, body_num=body_num)

        faces = scene.add_connectivity(body_num, template_ptr.surface_node_number, template_ptr.objects)
        scene.particleNum[0] += body_num
        scene.rigidNum[0] += body_num
        scene.surfaceNum[0] += template_ptr.surface_node_number * body_num
        self.faces = np.append(self.faces, faces).reshape(-1, 3)

    def load_txt_implicit_surface(self, scene: myScene, template):
        bounding_sphere = scene.get_bounding_sphere()
        rigid_body = scene.get_rigid_ptr()
        material = scene.get_material_ptr()
        particleNum = int(scene.particleNum[0])
            
        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        offset = DictIO.GetAlternative(template, "Translation", [0, 0, 0])
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.GetAlternative(template, "FixMotion", ["Free", "Free", "Free"])
        is_fix = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_str])
        file = DictIO.GetAlternative(template, "File", "BoundingSphere.txt")
        body_num = DictIO.GetAlternative(template, "ParticleNumber", -1)

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

        if not os.path.exists(file):
            raise EOFError("Invaild particle path")
        particle_cloud = np.loadtxt(file, unpack=True, comments='#').transpose()
        body_num = particle_cloud.shape[0] if body_num == -1 else min(body_num, particle_cloud.shape[0])
        particle_cloud = particle_cloud[0:body_num]
        body_num = particle_cloud.shape[0]

        coords = np.ascontiguousarray(particle_cloud[:, [0, 1, 2]]) + np.array(offset)
        radii = np.ascontiguousarray(particle_cloud[:, 3])
        orients = np.ascontiguousarray(particle_cloud[:, [4, 5, 6]])
        
        scene.check_rigid_body_number(self.sims, rigid_body_number=body_num)
        kernel_add_implicit_surface_files(rigid_body, bounding_sphere, material, particleNum, template_id, template_ptr.boundings.r_bound, vec3f(template_ptr.boundings.x_bound), vec3f(template_ptr.objects.inertia), 
                                            template_ptr.objects.eqradius, body_num, coords, radii, orients, groupID, matID, init_v, init_w, is_fix)
        print(" Implicit surface body Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, init_v, init_w, fix_v=is_fix, fix_w=is_fix, body_num=body_num)

        faces = scene.add_connectivity(body_num, template_ptr.surface_node_number, template_ptr.objects)
        scene.particleNum[0] += body_num
        scene.rigidNum[0] += body_num
        scene.surfaceNum[0] += template_ptr.surface_node_number * body_num
        self.faces = np.append(self.faces, faces).reshape(-1, 3)

    # ========================================================= #
    #                          npz File                         #
    # ========================================================= #
    def load_npz_file(self, scene, template):
        print('#', "Start adding body(s) ......")
        if self.sims.scheme == "DEM":
            particle_file_name = DictIO.GetAlternative(template, "ParticleFile", None)
            sphere_file_name = DictIO.GetAlternative(template, "SphereFile", None)
            clump_file_name = DictIO.GetAlternative(template, "ClumpFile", None)

            if not particle_file_name is None:
                self.restart_particles(scene, particle_file_name)

            if not sphere_file_name is None:
                if particle_file_name is None:
                    raise EOFError("Invalid path to read particle information")
                self.restart_spheres(scene, sphere_file_name)
            if not clump_file_name is None:
                if particle_file_name is None:
                    raise EOFError("Invalid path to read particle information")
                self.restart_clumps(scene, clump_file_name)
        elif self.sims.scheme == "LSDEM":
            rigid_file_name = DictIO.GetAlternative(template, "RigidFile", None)
            grid_file_name = DictIO.GetAlternative(template, "GridFile", None)
            bounding_sphere_file_name = DictIO.GetAlternative(template, "BoundingSphereFile", None)
            bounding_box_file_name = DictIO.GetAlternative(template, "BoundingBoxFile", None)
            surface_file_name = DictIO.GetAlternative(template, "SurfaceFile", None)
            if not rigid_file_name is None:
                self.restart_rigid_particles(scene, rigid_file_name)
            if not grid_file_name is None:
                self.restart_levelset_grids(scene, grid_file_name)
            if not bounding_sphere_file_name is None:
                self.restart_bounding_volumes(scene, bounding_sphere_file_name, bounding_box_file_name)
            if not surface_file_name is None:
                self.restart_surfaces(scene, surface_file_name)
        elif self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
            rigid_file_name = DictIO.GetAlternative(template, "RigidFile", None)
            bounding_sphere_file_name = DictIO.GetAlternative(template, "BoundingSphereFile", None)
            surface_file_name = DictIO.GetAlternative(template, "SurfaceFile", None)
            if not rigid_file_name is None:
                self.restart_rigid_particles(scene, rigid_file_name)
            if not bounding_sphere_file_name is None:
                self.restart_bounding_volumes(scene, bounding_sphere_file_name, None)
            if not surface_file_name is None:
                self.restart_surfaces(scene, surface_file_name)
        print('\n')
        
    def restart_rigid_particles(self, scene: myScene, rigid_file_name):
        if not os.path.exists(rigid_file_name):
            raise EOFError("Invaild rigid particle path")
        
        particle_info = np.load(rigid_file_name, allow_pickle=True) 
        rigid_number = int(DictIO.GetEssential(particle_info, "body_num"))
        if self.sims.is_continue:
            self.sims.current_time = DictIO.GetEssential(particle_info, "t_current")
            self.sims.CurrentTime[None] = DictIO.GetEssential(particle_info, "t_current")

        scene.check_rigid_body_number(self.sims, rigid_body_number=rigid_number)
        if self.sims.scheme == "LSDEM":
            kernel_rebulid_levelset_body(int(scene.rigidNum[0]), rigid_number, scene.rigid, 
                                    DictIO.GetEssential(particle_info, "groupID"), 
                                    DictIO.GetEssential(particle_info, "materialID"),
                                    DictIO.GetEssential(particle_info, "startNode"),
                                    DictIO.GetEssential(particle_info, "endNode"),
                                    DictIO.GetEssential(particle_info, "localNode"),
                                    DictIO.GetEssential(particle_info, "mass"),
                                    DictIO.GetEssential(particle_info, "equivalentRadius"),
                                    DictIO.GetEssential(particle_info, "mass_center"),
                                    DictIO.GetEssential(particle_info, "acceleration"),
                                    DictIO.GetEssential(particle_info, "angular_moment"),
                                    DictIO.GetEssential(particle_info, "velocity"),
                                    DictIO.GetEssential(particle_info, "omega"),
                                    DictIO.GetEssential(particle_info, "quanternion"),
                                    DictIO.GetEssential(particle_info, "inverse_inertia"),
                                    DictIO.GetEssential(particle_info, "is_fix"))
        elif self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
            kernel_rebulid_implicit_surface_body(int(scene.rigidNum[0]), rigid_number, scene.rigid, 
                                                 DictIO.GetEssential(particle_info, "groupID"), 
                                                 DictIO.GetEssential(particle_info, "materialID"),
                                                 DictIO.GetEssential(particle_info, "templateID"),
                                                 DictIO.GetEssential(particle_info, "scale"),
                                                 DictIO.GetEssential(particle_info, "mass"),
                                                 DictIO.GetEssential(particle_info, "equivalentRadius"),
                                                 DictIO.GetEssential(particle_info, "mass_center"),
                                                 DictIO.GetEssential(particle_info, "acceleration"),
                                                 DictIO.GetEssential(particle_info, "angular_moment"),
                                                 DictIO.GetEssential(particle_info, "velocity"),
                                                 DictIO.GetEssential(particle_info, "omega"),
                                                 DictIO.GetEssential(particle_info, "quanternion"),
                                                 DictIO.GetEssential(particle_info, "inverse_inertia"),
                                                 DictIO.GetAlternative(particle_info, "is_fix", np.ones((rigid_number, 3))))
        scene.rigidNum[0] += rigid_number
        print("Inserted rigid body Number: ", rigid_number)
        
    def restart_levelset_grids(self, scene: myScene, grid_file_name):
        if not os.path.exists(grid_file_name):
            raise EOFError("Invaild level-set grid path")
        
        grid_info = np.load(grid_file_name, allow_pickle=True) 
        grid_number = int(DictIO.GetEssential(grid_info, "total_grid_num"))
        if self.sims.is_continue:
            self.sims.current_time = DictIO.GetEssential(grid_info, "t_current")
            self.sims.CurrentTime[None] = DictIO.GetEssential(grid_info, "t_current")

        scene.check_grid_number(self.sims, grid_number=grid_number)
        scene.gridID = DictIO.GetEssential(grid_info, "grid_num")
        kernel_rebulid_levelset_grid(int(scene.gridNum[0]), grid_number, scene.rigid_grid, 
                                DictIO.GetEssential(grid_info, "distance_field"))
        scene.gridNum[0] += grid_number
        print("Inserted grid Number: ", grid_number)
        
    def restart_bounding_volumes(self, scene: myScene, bounding_sphere_file_name, bounding_box_file_name):
        if not os.path.exists(bounding_sphere_file_name):
            raise EOFError("Invaild bounding sphere path")
        
        bounding_sphere_info = np.load(bounding_sphere_file_name, allow_pickle=True) 
        particle_number = int(DictIO.GetEssential(bounding_sphere_info, "body_num"))
        if self.sims.is_continue:
            self.sims.current_time = DictIO.GetEssential(bounding_sphere_info, "t_current")
            self.sims.CurrentTime[None] = DictIO.GetEssential(bounding_sphere_info, "t_current")
        
        scene.check_particle_num(self.sims, particle_number=particle_number)

        kernel_rebulid_bounding_sphere(int(scene.particleNum[0]), particle_number, scene.particle, 
                                       DictIO.GetEssential(bounding_sphere_info, "active"),
                                       DictIO.GetEssential(bounding_sphere_info, "radius"),
                                       DictIO.GetEssential(bounding_sphere_info, "center"))
        
        if bounding_box_file_name is not None:
            if not os.path.exists(bounding_box_file_name):
                raise EOFError("Invaild bounding box path")
            
            bounding_sphere_info = np.load(bounding_box_file_name, allow_pickle=True) 
            kernel_rebulid_bounding_box(int(scene.particleNum[0]), particle_number, scene.box, 
                                       DictIO.GetEssential(bounding_sphere_info, "min_box"),
                                       DictIO.GetEssential(bounding_sphere_info, "max_box"),
                                       DictIO.GetEssential(bounding_sphere_info, "startGrid"),
                                       DictIO.GetEssential(bounding_sphere_info, "grid_num"),
                                       DictIO.GetEssential(bounding_sphere_info, "grid_space"),
                                       DictIO.GetEssential(bounding_sphere_info, "scale"),
                                       DictIO.GetEssential(bounding_sphere_info, "extent"))
        scene.particleNum[0] += particle_number
        print("Inserted bounding volume Number: ", particle_number)
        
    def restart_surfaces(self, scene: myScene, surface_file_name):
        if not os.path.exists(surface_file_name):
            raise EOFError("Invaild surface path")
        
        surface_info = np.load(surface_file_name, allow_pickle=True) 
        scene.connectivity = DictIO.GetEssential(surface_info, "connectivity")
        scene.face_index = DictIO.GetEssential(surface_info, "face_index")
        scene.vertice_index = DictIO.GetEssential(surface_info, "vertice_index")

        if self.sims.scheme == "LSDEM":
            scene.verticeID = DictIO.GetEssential(surface_info, "surface_num")
            surface_number = DictIO.GetEssential(surface_info, "total_surface_num")
            if self.sims.is_continue:
                self.sims.current_time = DictIO.GetEssential(surface_info, "t_current")
                self.sims.CurrentTime[None] = DictIO.GetEssential(surface_info, "t_current")

            scene.check_surface_node_number(self.sims, surface_node_number=int(scene.verticeID[-1]))
            master = DictIO.GetEssential(surface_info, "master")
            scene.surface.from_numpy(np.pad(master, (0, scene.surface.shape[0] - master.shape[0]), mode='constant', constant_values=0))
            kernel_rebulid_surface_node(0, int(scene.verticeID[-1]), scene.vertice,
                                        DictIO.GetEssential(surface_info, "vertices"),
                                        DictIO.GetEssential(surface_info, "parameters"))
            scene.surfaceNum[0] += surface_number
            print("Inserted surface node Number: ", surface_number)
        elif self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
            scene.surfaceNum[0] += sum(scene.vertice_index)
        
    def restart_particles(self, scene: myScene, particle_file_name):
        if not os.path.exists(particle_file_name):
            raise EOFError("Invaild particle path")
        
        particle_info = np.load(particle_file_name, allow_pickle=True) 
        particle_number = int(DictIO.GetEssential(particle_info, "body_num"))
        if self.sims.is_continue:
            self.sims.current_time = DictIO.GetEssential(particle_info, "t_current")
            self.sims.CurrentTime[None] = DictIO.GetEssential(particle_info, "t_current")

        scene.check_particle_num(self.sims, particle_number=particle_number)
        kernel_rebulid_particle(int(scene.particleNum[0]), particle_number, scene.particle, 
                                DictIO.GetAlternative(particle_info, "active", np.zeros(particle_number) + 1), 
                                DictIO.GetEssential(particle_info, "Index"),
                                DictIO.GetEssential(particle_info, "groupID"),
                                DictIO.GetEssential(particle_info, "materialID"),
                                DictIO.GetEssential(particle_info, "mass"),
                                DictIO.GetEssential(particle_info, "radius"),
                                DictIO.GetEssential(particle_info, "position"),
                                DictIO.GetEssential(particle_info, "velocity"),
                                DictIO.GetEssential(particle_info, "omega"))
        scene.particleNum[0] += particle_number
        print("Inserted particle Number: ", particle_number)
        
    def restart_particles(self, scene: myScene, particle_file_name):
        if not os.path.exists(particle_file_name):
            raise EOFError("Invaild particle path")
        
        particle_info = np.load(particle_file_name, allow_pickle=True) 
        particle_number = int(DictIO.GetEssential(particle_info, "body_num"))
        if self.sims.is_continue:
            self.sims.current_time = DictIO.GetEssential(particle_info, "t_current")
            self.sims.CurrentTime[None] = DictIO.GetEssential(particle_info, "t_current")

        scene.check_particle_num(self.sims, particle_number=particle_number)
        kernel_rebulid_particle(int(scene.particleNum[0]), particle_number, scene.particle, 
                                DictIO.GetAlternative(particle_info, "active", np.zeros(particle_number) + 1), 
                                DictIO.GetEssential(particle_info, "Index"),
                                DictIO.GetEssential(particle_info, "groupID"),
                                DictIO.GetEssential(particle_info, "materialID"),
                                DictIO.GetEssential(particle_info, "mass"),
                                DictIO.GetEssential(particle_info, "radius"),
                                DictIO.GetEssential(particle_info, "position"),
                                DictIO.GetEssential(particle_info, "velocity"),
                                DictIO.GetEssential(particle_info, "omega"))
        scene.particleNum[0] += particle_number
        print("Inserted particle Number: ", particle_number)

    def restart_spheres(self, scene: myScene, sphere_file_name):
        if not os.path.exists(sphere_file_name):
            raise EOFError("Invaild sphere path")
        
        sphere_info = np.load(sphere_file_name, allow_pickle=True) 
        sphere_number = int(DictIO.GetEssential(sphere_info, "body_num"))

        scene.check_sphere_number(self.sims, body_number=sphere_number)
        kernel_rebuild_sphere(int(scene.sphereNum[0]), sphere_number, scene.sphere, 
                              DictIO.GetEssential(sphere_info, "sphereIndex"),
                              DictIO.GetEssential(sphere_info, "inverseInertia"),
                              DictIO.GetEssential(sphere_info, "quanternion"),
                              DictIO.GetEssential(sphere_info, "acceleration"),
                              DictIO.GetEssential(sphere_info, "angular_moment"),
                              DictIO.GetEssential(sphere_info, "fix_v"),
                              DictIO.GetEssential(sphere_info, "fix_w"))
        scene.sphereNum[0] += sphere_number
        print("Inserted sphere Number: ", sphere_number)

    def restart_clumps(self, scene: myScene, clump_file_name):
        if not os.path.exists(clump_file_name):
            raise EOFError("Invaild clump path")
        
        clump_info = np.load(clump_file_name, allow_pickle=True) 
        clump_number = int(DictIO.GetEssential(clump_info, "body_num"))

        scene.check_clump_number(self.sims, body_number=clump_number)
        kernel_rebuild_clump(int(scene.clumpNum[0]), clump_number, scene.clump, 
                             DictIO.GetEssential(clump_info, "startIndex"),
                             DictIO.GetEssential(clump_info, "endIndex"),
                             DictIO.GetEssential(clump_info, "mass"),
                             DictIO.GetEssential(clump_info, "equivalentRadius"),
                             DictIO.GetEssential(clump_info, "centerOfMass"),
                             DictIO.GetEssential(clump_info, "velocity"),
                             DictIO.GetEssential(clump_info, "omega"),
                             DictIO.GetEssential(clump_info, "acceleration"),
                             DictIO.GetEssential(clump_info, "angular_moment"),
                             DictIO.GetEssential(clump_info, "quanternion"),
                             DictIO.GetEssential(clump_info, "inverse_inertia"))
        scene.clumpNum[0] += clump_number
        print("Inserted clump Number: ", clump_number)

    # ========================================================= #
    #                          OBJ File                         #
    # ========================================================= #
    def load_obj_file(self, scene: myScene, template):
        print('#', "Start voxelizing body(s) ......")
        body_file = DictIO.GetEssential(template, "GeometryFile")
        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))

        scale_factor = DictIO.GetAlternative(template, "ScaleFactor", 1.)
        offset = DictIO.GetAlternative(template, "Translation", vec3f([0, 0, 0]))
        orientation = DictIO.GetAlternative(template, "Orientation", vec3f([0, 0, 0]))

        discretize_domain = DictIO.GetEssential(template, "DiscretizeDomain")
        max_rad = DictIO.GetEssential(template, "MaxRadius", "Radius")
        min_rad = DictIO.GetEssential(template, "MinRadius", "Radius")

        rotation = ThetaToRotationMatrix(orientation)
        mesh = trimesh.load(body_file)
        mesh.apply_scale(scale_factor)
        rot_matrix = trimesh.transformations.rotation_matrix(rotation, orientation, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)

        mesh_backup = mesh.copy()
        mesh_backup.vertices += offset
        voxelized_mesh = mesh.voxelized(pitch=discretize_domain).fill()
        voxelized_points_np = voxelized_mesh.points + offset

        scene.check_particle_num(self.sims, particle_number=len(voxelized_points_np))
        scene.check_sphere_number(self.sims, body_number=len(voxelized_points_np))
        generate_sphere_from_file(min_rad, max_rad, groupID, matID, init_v, init_w)

