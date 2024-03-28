import taichi as ti
import numpy as np
import trimesh, os

from src.mpm.generator.InsertionKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefination import vec3f, vec6f



class BodyReader(object):
    sims: Simulation

    def __init__(self, sims) -> None:
        self.sims = sims
        self.active = True
        self.myRegion = None
        self.myTemplate = None
        self.file_type = None

        self.start_time = 0.
        self.end_time = 0.
        self.next_generate_time = 0.
        self.insert_interval = 1e10

        self.FIX = {
                    "Free": 0,
                    "Fix": 1
                   }

    def no_print(self):
        self.log = False

    def deactivate(self):
        self.active = False

    def set_region(self, region):
        self.myRegion = region

    def get_region_ptr(self, name):
        if not self.myRegion is None:
            return self.myRegion[name]
        else:
            raise RuntimeError("Region class should be activated first!")

    def set_system_strcuture(self, body_dict):
        if self.sims.dimension == 2:
            raise RuntimeError("Load File class does not support 2-Dimensional condition")
        self.file_type = DictIO.GetEssential(body_dict, "FileType")
        self.myTemplate = DictIO.GetEssential(body_dict, "Template")
        period = DictIO.GetAlternative(body_dict, "Period", [0, 0, 1e6])
        self.start_time = period[0]
        self.end_time = period[1]
        self.insert_interval = period[2]

    def finalize(self):
        del self.active #, self.field_builder, self.snode_tree
        del self.start_time, self.end_time, self.insert_interval, self.myTemplate

    def begin(self, scene: myScene):
        if self.sims.current_time < self.next_generate_time: return 0
        if self.sims.current_time < self.start_time or self.sims.current_time > self.end_time: return 0
        
        
        if self.file_type == "TXT":
            if type(self.myTemplate) is dict:
                self.add_txt_body(scene, self.myTemplate)
            elif type(self.myTemplate) is list:
                for template in self.myTemplate:
                    self.add_txt_body(scene, template)
        elif self.file_type == "NPZ":
            if type(self.myTemplate) is dict:
                self.add_npz_body(scene, self.myTemplate)
            elif type(self.myTemplate) is list:
                for template in self.myTemplate:
                    self.add_npz_body(scene, template)
        elif self.file_type == "OBJ":
            if type(self.myTemplate) is dict:
                self.add_obj_body(scene, self.myTemplate)
            elif type(self.myTemplate) is list:
                for template in self.myTemplate:
                    self.add_obj_body(scene, template)
        else:
            raise RuntimeError("Invalid file type. Only the following file types are aviliable: ['TXT', 'NPZ', 'OBJ']")

        if self.sims.current_time + self.next_generate_time > self.end_time or self.insert_interval > self.sims.time or self.end_time > self.sims.time or\
            self.end_time == 0 or self.start_time > self.end_time:
            self.deactivate()
        else:
            self.next_generate_time = self.sims.current_time + self.insert_interval
        return 1

    def print_particle_info(self, nParticlesPerCell, bodyID, materialID, init_v, fix_v, particle_volume,particle_count):
        if particle_count == 0:
            raise RuntimeError("Zero Particles are inserted into region!")
        print("Body ID = ", bodyID)
        print("Material ID = ", materialID)
        print("Add Particle Number: ", particle_count)
        print("The Number of Particle per Cell: ", nParticlesPerCell)
        print("Initial Velocity = ", init_v)
        print("Fixed Velocity = ", fix_v)
        print("Particle Volume = ", particle_volume)
        print('\n')

    def check_bodyID(self, scene: myScene, bodyID):
        if bodyID > scene.node.shape[1] - 1:
            raise RuntimeError(f"Keyword:: /bodyID/ must be smaller than {scene.node.shape[1] - 1}")

    def rotate_body(self, orientation, coords, particle_num):
        bounding_center = kernel_calc_mass_of_center_(coords)
        kernel_position_rotate_for_array_(orientation, bounding_center, coords, 0,particle_num)

    def set_element_calLength(self, scene: myScene, bodyID, psize):
        if self.sims.shape_function == "Linear":
            scene.element.calLength[bodyID] = [0, 0, 0]
        elif self.sims.shape_function == "QuadBSpline" or self.sims.shape_function == "CubicBSpline":
            scene.element.calLength[bodyID] = 0.5 * scene.element.grid_size
        elif self.sims.shape_function == "GIMP":
            scene.element.calLength[bodyID] = psize

    def add_txt_body(self, scene: myScene, template):
        particle_file = DictIO.GetAlternative(template, "ParticleFile", "Particle.txt") 
        print('#', f"Start adding material points from {particle_file}......")
        if not os.path.exists(particle_file):
            raise EOFError("Invaild path")
        
        bodyID = DictIO.GetEssential(template, "BodyID")
        self.check_bodyID(scene, bodyID)
        rigid_body = DictIO.GetAlternative(template, "RigidBody", False)
        if rigid_body:
            materialID = 0
            density = DictIO.GetAlternative(template, "Density", 2650)
            scene.is_rigid[bodyID] = 1
        else:
            materialID = DictIO.GetEssential(template, "MaterialID")
            density = scene.material.matProps[materialID].density

            if materialID <= 0:
                raise RuntimeError(f"Material ID {materialID} should be larger than 0")
            
        particle_stress = DictIO.GetAlternative(template, "ParticleStress", {"GravityField": False, "InternalStress": vec6f([0, 0, 0, 0, 0, 0])})
        traction = DictIO.GetAlternative(template, "Traction", {})
        orientation = DictIO.GetAlternative(template, "Orientation", vec3f([0, 0, 1]))
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        fix_v_str = DictIO.GetAlternative(template, "FixVelocity", ["Free", "Free", "Free"])
        fix_v = vec3u8([DictIO.GetEssential(self.FIX, is_fix) for is_fix in fix_v_str])

        #fix_v = DictIO.GetEssential(self.FIX, fix_v_str) #fix
        fix_v = vec3u8([DictIO.GetEssential(self.FIX, is_fix) for is_fix in fix_v_str])
        init_particle_num = int(scene.particleNum[0])
        particle_cloud = np.loadtxt(particle_file, unpack=True, comments='#').transpose()
        coords = np.ascontiguousarray(particle_cloud[:, [0, 1, 2]])
        volume = np.ascontiguousarray(particle_cloud[:, 3])
        psize = np.ascontiguousarray(particle_cloud[:, [4, 5, 6]])
        particle_num = coords.shape[0]
        self.set_element_calLength(scene, bodyID, psize[0,:])

        self.rotate_body(orientation, coords, particle_num)
        scene.check_particle_num(self.sims, particle_number=particle_num)
        kernel_read_particle_file_(scene.particle, int(scene.particleNum[0]), particle_num, coords, psize, volume, bodyID, materialID, density, init_v, fix_v)
        
        self.set_particle_stress(scene, materialID, init_particle_num, particle_num, particle_stress)
        name = DictIO.GetEssential(template, "RegionName")
        region: RegionFunction = self.get_region_ptr(name)
        self.set_traction(
            tractions=traction,
            region=region ,
            particle=scene.particle, 
            particle_num=particle_num, 
            init_particle_num=int(scene.particleNum[0]))
        scene.particleNum[0] += particle_num
        self.print_particle_info(nParticlesPerCell='Custom', bodyID = bodyID, materialID = materialID, init_v=init_v, 
                                 fix_v=fix_v_str, particle_volume=volume[0], particle_count = particle_num)
    def add_npz_body(self, scene: myScene, template): #fix maybe exist bug
        particle_file = DictIO.GetEssential(template, "File")
        print('#', f"Start adding material points from {particle_file}......")
        if not os.path.exists(particle_file):
            raise EOFError("Invaild path")
        
        if DictIO.GetAlternative(template, "Restart", False):
            particle_info = np.load(particle_file, allow_pickle=True) 
            if self.sims.is_continue:
                self.sims.current_time = DictIO.GetEssential(particle_info, 't_current')
                self.sims.CurrentTime[None] = DictIO.GetEssential(particle_info, 't_current')
            particle_number = int(DictIO.GetEssential(particle_info, "body_num"))

            if particle_number > self.sims.max_particle_num:
                raise RuntimeError("/max_particle_number/ should be enlarged")
            if self.sims.coupling or self.sims.neighbor_detection:
                kernel_rebulid_particle_coupling(particle_number, scene.particle, scene.is_rigid,
                                                 DictIO.GetEssential(particle_info, "bodyID"), 
                                                 DictIO.GetEssential(particle_info, "materialID"), 
                                                 DictIO.GetEssential(particle_info, "active"), 
                                                 DictIO.GetEssential(particle_info, "free_surface"), 
                                                 DictIO.GetEssential(particle_info, "normal"), 
                                                 DictIO.GetEssential(particle_info, "mass"), 
                                                 DictIO.GetEssential(particle_info, "position"), 
                                                 DictIO.GetEssential(particle_info, "velocity"), 
                                                 DictIO.GetEssential(particle_info, "volume"), 
                                                 DictIO.GetEssential(particle_info, "traction"), 
                                                 DictIO.GetEssential(particle_info, "strain"), 
                                                 DictIO.GetEssential(particle_info, "stress"), 
                                                 DictIO.GetEssential(particle_info, "psize"), 
                                                 DictIO.GetEssential(particle_info, "velocity_gradient"), 
                                                 DictIO.GetEssential(particle_info, "fix_v"))
            else:
                kernel_rebulid_particle(particle_number, scene.particle, scene.is_rigid,
                                        DictIO.GetEssential(particle_info, "bodyID"), 
                                        DictIO.GetEssential(particle_info, "materialID"), 
                                        DictIO.GetEssential(particle_info, "active"), 
                                        DictIO.GetEssential(particle_info, "mass"),
                                        DictIO.GetEssential(particle_info, "position"), 
                                        DictIO.GetEssential(particle_info, "velocity"), 
                                        DictIO.GetEssential(particle_info, "volume"), 
                                        DictIO.GetEssential(particle_info, "traction"), 
                                        DictIO.GetEssential(particle_info, "strain"), 
                                        DictIO.GetEssential(particle_info, "stress"), 
                                        DictIO.GetEssential(particle_info, "psize"), 
                                        DictIO.GetEssential(particle_info, "velocity_gradient"), 
                                        DictIO.GetEssential(particle_info, "fix_v"))
                
            stateVars = DictIO.GetEssential(particle_info, "state_vars")
            scene.material.reload_state_variables(stateVars)
            scene.particleNum[0] = particle_number
        else:
            bodyID = DictIO.GetEssential(template, "BodyID")
            self.check_bodyID(scene, bodyID)
            rigid_body = DictIO.GetAlternative(template, "RigidBody", False)
            if rigid_body:
                materialID = 0
                density = DictIO.GetAlternative(template, "Density", 2650)
                scene.is_rigid[bodyID] = 1
            else:
                materialID = DictIO.GetEssential(template, "MaterialID")
                density = scene.material.matProps[materialID].density

                if materialID <= 0:
                    raise RuntimeError(f"Material ID {materialID} should be larger than 0")
                
            particle_stress = DictIO.GetAlternative(template, "ParticleStress", {"GravityField": False, "InternalStress": vec6f([0, 0, 0, 0, 0, 0])})
            traction = DictIO.GetAlternative(template, "Traction", {})
            orientation = DictIO.GetAlternative(template, "Orientation", vec3f([0, 0, 1]))
            init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
            fix_v_str = DictIO.GetAlternative(template, "FixVelocity", ["Free", "Free", "Free"])
            fix_v = vec3u8([DictIO.GetEssential(self.FIX, is_fix) for is_fix in fix_v_str])
            init_particle_num = int(scene.particleNum[0])

            particle_cloud = np.load(particle_file, allow_pickle=True) 
            coords = DictIO.GetEssential(particle_cloud, "position")
            psize = DictIO.GetEssential(particle_cloud, "psize")
            volume = DictIO.GetEssential(particle_cloud, "volume")
            particle_num = coords.shape[0]

            if self.sims.shape_function == "Linear":
                scene.element.calLength[bodyID] = [0, 0, 0]
            elif self.sims.shape_function == "QuadBSpline" or self.sims.shape_function == "CubicBSpline":
                scene.element.calLength[bodyID] = 0.5 * scene.element.grid_size
            elif self.sims.shape_function == "GIMP":
                scene.element.calLength[bodyID] = psize

            self.rotate_body(orientation, coords, particle_num)
            scene.check_particle_num(self.sims, particle_number=particle_num)
            kernel_read_particle_file_(scene.particle, int(scene.particleNum[0]), particle_num, coords, psize, volume, bodyID, materialID, density, init_v, fix_v)
            
            self.set_particle_stress(scene, materialID, init_particle_num, particle_num, particle_stress)
            self.set_traction(volume, traction, scene.particle, particle_num, int(scene.particleNum[0]))
            scene.particleNum[0] += particle_num

    def add_obj_body(self, scene: myScene, template):
        particle_file = DictIO.GetAlternative(template, "File", "Particle.obj")
        print('#', f"Start adding material points from {particle_file}......")
        if not os.path.exists(particle_file):
            raise EOFError("Invaild path")
        
        bodyID = DictIO.GetEssential(template, "BodyID")
        self.check_bodyID(scene, bodyID)
        rigid_body = DictIO.GetAlternative(template, "RigidBody", False)
        if rigid_body:
            materialID = 0
            density = DictIO.GetAlternative(template, "Density", 2650)
            scene.is_rigid[bodyID] = 1
        else:
            materialID = DictIO.GetEssential(template, "MaterialID")
            density = scene.material.matProps[materialID].density

            if materialID <= 0:
                raise RuntimeError(f"Material ID {materialID} should be larger than 0")
            
        particle_stress = DictIO.GetAlternative(template, "ParticleStress", {"GravityField": False, "InternalStress": vec6f([0, 0, 0, 0, 0, 0])})
        traction = DictIO.GetAlternative(template, "Traction", {})
        orientation = DictIO.GetAlternative(template, "Orientation", vec3f([0, 0, 1]))
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        fix_v_str = DictIO.GetAlternative(template, "FixVelocity", ["Free", "Free", "Free"])
        fix_v = vec3u8([DictIO.GetEssential(self.FIX, is_fix) for is_fix in fix_v_str])

        particle_cloud = self.load_obj_file(particle_file)
        scale_factor = DictIO.GetAlternative(particle_file, "ScaleFactor", default=1.0)
        offset = DictIO.GetAlternative(particle_file, "Offset", default=0.0)
        orientation = DictIO.GetAlternative(particle_file, "Orientation", default=vec3f([0., 0., 1.]))
        
        mesh = trimesh.load(particle_file)
        mesh.apply_scale(scale_factor)
        mesh_backup = mesh.copy()
        mesh_backup.vertices += offset
        com = mesh_backup.vertices.mean(axis=0)
        offset += self.GetCenterOfCell(com) - com
        voxelized_mesh = mesh.voxelized(pitch=self.dx).fill()
        voxelized_points_np = voxelized_mesh.points + offset

        # TODO: activate cell

        '''if self.sims.shape_function == "Linear":
            scene.element.calLength[bodyID] = [0, 0, 0]
        elif self.sims.shape_function == "QuadBSpline" or self.sims.shape_function == "CubicBSpline":
            scene.element.calLength[bodyID] = 0.5 * scene.element.grid_size
        elif self.sims.shape_function == "GIMP":
            scene.element.calLength[bodyID] = psize

        self.rotate_body(orientation, coords, particle_num)
        scene.check_particle_num(self.sims, particle_number=particle_num)
        kernel_read_particle_file_(scene.particle, scene.particleNum, particle_num, coords, psize, volume, bodyID, materialID, density, init_v, fix_v)
        
        self.set_particle_stress(scene, materialID, init_particle_num, particle_num, particle_stress)
        self.set_traction(traction, scene.particle, particle_num, scene.particleNum[0])
        scene.particleNum[0] += particle_num'''

    def set_particle_stress(self, scene: myScene, materialID, init_particle_num, particle_num, particle_stress):
        if type(particle_stress) is str:
            stress_file = DictIO.GetAlternative(particle_stress, "File", "ParticleStress.txt")
            stress_cloud = np.loadtxt(stress_file, unpack=True, comments='#').transpose()
            if stress_cloud.shape[0] != particle_num:
                raise ValueError("The length of File:: /ParticleStress/ is error!")
            if stress_cloud.shape[1] != 6:
                raise ValueError("The stress tensor should be transform to viogt format")
            kernel_apply_stress_(init_particle_num, init_particle_num + particle_num, initialStress, scene.particle)
        elif type(particle_stress) is dict:
            gravityField = DictIO.GetAlternative(particle_stress, "GravityField", False)
            initialStress = DictIO.GetAlternative(particle_stress, "InternalStress", vec6f([0, 0, 0, 0, 0, 0]))
            self.set_internal_stress(
                scene=scene, 
                materialID=materialID, 
                particle_num=particle_num, 
                gravityField=gravityField, 
                initialStress=initialStress, 
                )

    def set_internal_stress(self, 
                            scene: myScene, 
                            materialID, 
                            particle_num, 
                            gravityField, 
                            initialStress):
        if gravityField and materialID >= 0:
            k0 = scene.material.get_lateral_coefficient(materialID)
            top_position = scene.find_min_z_position()
            print("Warning: The outline of particles should be aligned to Z axis when set /GravityField/ active!")
            if not all(np.abs(np.array(self.sims.gravity) - np.array([0., 0., -9.8])) < 1e-12):
                raise ValueError("Gravity must be set as [0, 0, -9.8] when gravity activated")
            density = scene.material.matProps[materialID].density
            kernel_apply_gravity_field_(
                density, 
                int(scene.particleNum[0]), 
                int(scene.particleNum[0]) + particle_num, 
                k0, 
                top_position, 
                self.sims.gravity, 
                scene.particle)

        if initialStress.n != 6:
            raise ValueError(f"The dimension of initial stress: {initialStress.n} is inconsistent with the dimension of stress vigot tensor in 3D: 6")
        kernel_apply_vigot_stress_(
            int(scene.particleNum[0]), 
            int(scene.particleNum[0]) + particle_num, 
            initialStress, 
            scene.particle)

    def set_traction(self, tractions, region, particle, particle_num, init_particle_num):
        if tractions:
            if type(tractions) is dict:
                self.set_particle_traction(tractions, region, particle, particle_num, init_particle_num)
            elif type(tractions) is list:
                for traction in tractions:
                    self.set_particle_traction(traction, region, particle, particle_num, init_particle_num)

    def set_particle_traction(self, traction, region: RegionFunction, particle, particle_num, init_particle_num):
        traction_force = DictIO.GetEssential(traction, "Pressure") 
        if isinstance(traction_force, float):
            traction_force *= DictIO.GetEssential(traction, "OuterNormal")
        region_function = region.function
        region_name = DictIO.GetAlternative(traction, "RegionName", None)
        if region_name:
            traction_region: RegionFunction = self.get_region_ptr(region_name)
            region_function = traction_region.function
        region_function = DictIO.GetAlternative(traction, "RegionFunction", region_function)
        kernel_set_particle_traction_(init_particle_num, init_particle_num + particle_num, region_function, traction_force, particle)
