import numpy as np
import os

from src.mpm.generator.InsertionKernel import *
from src.mpm.Contact import DEMContact
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefination import vec2u8, vec3f, vec6f
import trimesh as tm


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
        
        print('#', "Start adding material points ......")
        start_particle = int(scene.particleNum[0])
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
        
        if not scene.particle is None:
            end_particle = int(scene.particleNum[0])
            scene.material.state_vars_initialize(start_particle, end_particle, scene.particle)

        if self.sims.current_time + self.next_generate_time > self.end_time or self.insert_interval > self.sims.time or self.end_time > self.sims.time or\
            self.end_time == 0 or self.start_time > self.end_time:
            self.deactivate()
        else:
            self.next_generate_time = self.sims.current_time + self.insert_interval
        return 1

    def print_particle_info(self, nParticlesPerCell, bodyID, materialID, init_v, fix_v, particle_count):
        print(" Body(s) Information ".center(71, '-'))
        if particle_count == 0:
            raise RuntimeError("Zero Particles are inserted into region!")
        print("Body ID = ", bodyID)
        print("Material ID = ", materialID)
        print("Particle Number: ", particle_count)
        if not nParticlesPerCell is None:
            print("The Number of Particle per Cell: ", nParticlesPerCell)
        print("Initial Velocity = ", init_v)
        print("Fixed Velocity = ", fix_v)
        print('\n')

    def check_bodyID(self, scene: myScene, bodyID):
        if bodyID > scene.node.shape[1] - 1:
            raise RuntimeError(f"Keyword:: /bodyID/ must be smaller than {scene.node.shape[1] - 1}")

    def rotate_body(self, orientation, coords, init_particle_num, particle_num):
        if self.sims.dimension == 3:
            bounding_center = kernel_calc_mass_of_center_(coords)
            kernel_position_rotate_(orientation, bounding_center, coords, init_particle_num, particle_num)
        elif self.sims.dimension == 2:
            bounding_center = kernel_calc_mass_of_center_2D(coords)
            kernel_position_rotate_2D(orientation, bounding_center, coords, init_particle_num, particle_num)

    def set_element_calLength(self, scene: myScene, bodyID, psize):
        if self.sims.shape_function == "Linear":
            scene.element.calLength[bodyID] = [0, 0, 0]
        elif self.sims.shape_function == "QuadBSpline":
            scene.element.calLength[bodyID] = 0.5 * scene.element.grid_size
        elif self.sims.shape_function == "CubicBSpline":
            scene.element.calLength[bodyID] = scene.element.grid_size
        elif self.sims.shape_function == "GIMP":
            scene.element.calLength[bodyID] = psize

    def particle_cloud_filter(self, particle_cloud, region: RegionFunction):
        if region.region_type == 'Rectangle':
            start_points = region.local_start_point
            size_points = region.local_region_size
            bounding_box = [start_points[0], start_points[0] + size_points[0], start_points[1], start_points[1] + size_points[1]]
            particle_cloud = particle_cloud[np.logical_and(particle_cloud[:, 0] >= bounding_box[0], particle_cloud[:, 0] <= bounding_box[1])]
            particle_cloud = particle_cloud[np.logical_and(particle_cloud[:, 1] >= bounding_box[2], particle_cloud[:, 1] <= bounding_box[3])]
            if self.sims.dimension == 3:
                bounding_box += [start_points[2], start_points[2] + size_points[2]]
                particle_cloud = particle_cloud[np.logical_and(particle_cloud[:, 2] >= bounding_box[4], particle_cloud[:, 2] <= bounding_box[5])]
        return particle_cloud
        
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
            
        particle_number = DictIO.GetAlternative(template, "ParticleNumber", -1)
        particle_stress = DictIO.GetAlternative(template, "ParticleStress", {"GravityField": False, "InternalStress": vec6f([0, 0, 0, 0, 0, 0])})
        orientation = DictIO.GetAlternative(template, "Orientation", vec3f([0, 0, 1]))
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

        init_particle_num = int(scene.particleNum[0])
        particle_cloud = np.loadtxt(particle_file, unpack=True, comments='#').transpose()
        coords, volume = None, None
        if self.sims.dimension == 3:
            coords = np.ascontiguousarray(particle_cloud[0:particle_number, [0, 1, 2]])
            volume = np.ascontiguousarray(particle_cloud[0:particle_number, 3])
            psize = np.ascontiguousarray(particle_cloud[0:particle_number, [4, 5, 6]])
        elif self.sims.dimension == 2:
            coords = np.ascontiguousarray(particle_cloud[0:particle_number, [0, 1]])
            volume = np.ascontiguousarray(particle_cloud[0:particle_number, 2])
            psize = np.ascontiguousarray(particle_cloud[0:particle_number, [3, 4]])
        particle_num = coords.shape[0]

        self.rotate_body(orientation, coords, init_particle_num, particle_num)
        scene.check_particle_num(self.sims, particle_number=particle_num)
        if self.sims.dimension == 3:
            self.set_element_calLength(scene, bodyID, vec3f(psize[0, 0],psize[0, 1], psize[0, 2]))
            kernel_read_particle_file_(scene.particle, int(scene.particleNum[0]), particle_num, coords, volume, bodyID, materialID, density, init_v, fix_v)
        elif self.sims.dimension == 2:
            self.set_element_calLength(scene, bodyID, vec2f(psize[0, 0],psize[0, 1]))
            kernel_read_particle_file_2D(scene.particle, int(scene.particleNum[0]), particle_num, coords, volume, bodyID, materialID, density, init_v, fix_v)

        self.set_particle_stress(scene, materialID, init_particle_num, particle_num, particle_stress)
        scene.push_psize(particle_num, psize)
        traction = DictIO.GetAlternative(template, "Traction", {})
        self.set_traction(scene, particle_num, traction)
        scene.particleNum[0] += particle_num
        self.print_particle_info(None, bodyID, materialID, init_v, fix_v, particle_num)
  
    def add_npz_body(self, scene: myScene, template):
        particle_file = DictIO.GetEssential(template, "File")
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
                                                 DictIO.GetEssential(particle_info, "stress"), 
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
                                        DictIO.GetEssential(particle_info, "stress"), 
                                        DictIO.GetEssential(particle_info, "velocity_gradient"), 
                                        DictIO.GetEssential(particle_info, "fix_v"))
            psize = DictIO.GetEssential(particle_info, "psize")
            scene.push_psize(1, psize)
            traction = DictIO.GetAlternative(template, "Traction", {})
            self.set_traction(scene, particle_number, traction)
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

            self.rotate_body(orientation, coords, init_particle_num, particle_num)
            scene.check_particle_num(self.sims, particle_number=particle_num)
            kernel_read_particle_file_(scene.particle, int(scene.particleNum[0]), particle_num, coords, volume, bodyID, materialID, density, init_v, fix_v)
            self.set_particle_stress(scene, materialID, init_particle_num, particle_num, particle_stress)
            scene.push_psize(1, psize)
            traction = DictIO.GetAlternative(template, "Traction", {})
            self.set_traction(scene, particle_num, traction)
            scene.particleNum[0] += particle_num

    def add_obj_body(self, scene: myScene, template):
        if self.sims.dimension == 2:
            raise RuntimeError("2D conditions do not support voxelization technique")
        particle_file = DictIO.GetAlternative(template, "ParticleFile", "Particle.obj")
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
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        fix_v_str = DictIO.GetAlternative(template, "FixVelocity", ["Free", "Free", "Free"])
        fix_v = vec3u8([DictIO.GetEssential(self.FIX, is_fix) for is_fix in fix_v_str])
        init_particle_num = int(scene.particleNum[0])

        scale_factor = DictIO.GetAlternative(template, "ScaleFactor", default=1.0)
        offset = np.array(DictIO.GetAlternative(template, "Offset", [0., 0., 0.]))
        orientation = DictIO.GetAlternative(template, "Orientation", vec3f([0, 0, 1]))
        nParticlesPerCell = DictIO.GetAlternative(template, "nParticlesPerCell", 2)
        
        mesh = tm.load(particle_file)
        com = mesh.vertices.mean(axis=0)
        mesh.apply_translation(offset - com)
        mesh.apply_scale(scale_factor)
        cell_size = scene.element.grid_size
        diameter = min(cell_size[0], cell_size[1], cell_size[2]) / nParticlesPerCell
        voxelized_mesh = mesh.voxelized(pitch=diameter).fill()
        voxelized_points_np = voxelized_mesh.points.copy()
        particle_num = voxelized_points_np.shape[0]

        psize = np.repeat(0.5 * diameter, particle_num * 3).reshape((particle_num, 3))
        volume = np.repeat(4./3. * np.pi * (0.5 * diameter) ** 3, particle_num)
        self.rotate_body(orientation, voxelized_points_np, init_particle_num, particle_num)
        scene.check_particle_num(self.sims, particle_number=particle_num)
        kernel_read_particle_file_(scene.particle, int(scene.particleNum[0]), particle_num, voxelized_points_np, volume, bodyID, materialID, density, init_v, fix_v)
        
        self.set_particle_stress(scene, materialID, init_particle_num, particle_num, particle_stress)
        scene.push_psize(1, psize)
        traction = DictIO.GetAlternative(template, "Traction", {})
        self.set_traction(scene, particle_num, traction)
        scene.particleNum[0] += particle_num

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
            if self.sims.material_type == "TwoPhaseSingleLayer":
                porePressure = DictIO.GetAlternative(particle_stress, "PorePressure", 0.)
                self.set_internal_stress(scene, materialID, particle_num, gravityField, initialStress, porePressure)
            else:
                self.set_internal_stress(scene, materialID, particle_num, gravityField, initialStress)

    def set_internal_stress(self, scene: myScene, materialID, particle_num, gravityField, initialStress, porePressure=0):
        if gravityField and materialID >= 0:
            k0 = scene.material.get_lateral_coefficient(materialID)
            if self.sims.dimension == 3:
                top_position = scene.find_min_z_position()
                print("Warning: The outline of particles should be aligned to Z axis when set /GravityField/ active!")
                if not all(np.abs(np.array(self.sims.gravity) - np.array([0., 0., -9.8])) < 1e-12):
                    raise ValueError("Gravity must be set as [0, 0, -9.8] when gravity activated")
                density = scene.material.matProps[materialID].density
                kernel_apply_gravity_field_(density, int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_num, k0, top_position, self.sims.gravity, scene.particle)
            elif self.sims.dimension == 2:
                top_position = scene.find_min_y_position()
                print("Warning: The outline of particles should be aligned to Y axis when set /GravityField/ active!")
                if not all(np.abs(np.array(self.sims.gravity) - np.array([0., -9.8, 0.])) < 1e-12):
                    raise ValueError("Gravity must be set as [0, -9.8] when gravity activated")
                density = scene.material.matProps[materialID].density
                kernel_apply_gravity_field_2D(density, int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_num, k0, top_position, self.sims.gravity, scene.particle)

        if initialStress.n != 6:
            raise ValueError(f"The dimension of initial stress: {initialStress.n} is inconsistent with the dimension of stress vigot tensor in 3D: 6")
        kernel_apply_vigot_stress_(int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_num, initialStress, scene.particle)
        if self.sims.material_type == "TwoPhaseSingleLayer":
            kernel_apply_pore_pressure_(int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_num, porePressure, scene.particle)

    def set_traction(self, scene: myScene, particle_num, tractions):
        scene.boundary.get_essentials(scene.is_rigid, scene.psize, self.myRegion)
        if tractions:
            if type(tractions) is dict:
                scene.boundary.set_particle_traction(self.sims, tractions, particle_num, int(scene.particleNum[0]), scene.particle, scene.psize)
            elif type(tractions) is list:
                for traction in tractions:
                    scene.boundary.set_particle_traction(self.sims, traction, particle_num, int(scene.particleNum[0]), scene.particle, scene.psize)

    def set_polygons(self, contact: DEMContact, body_dict):
        vertices = DictIO.GetEssential(body_dict, "Vertices")
        if isinstance(vertices, str):
            print('#', f"Start adding material points from {vertices}......")
            if not os.path.exists(vertices):
                raise EOFError("Invaild path")
            vertice = np.loadtxt(vertices)[:, 0:self.sims.dimension]
        elif isinstance(vertice, (list, tuple, np.ndarray)):
            vertice = np.array(vertice)
        
        velocity = DictIO.GetAlternative(body_dict, "InitialVelocity", [0., 0.])
        contact.generate_vertice_field(self.sims.dimension, vertice.shape[0])
        contact.polygon_vertices.from_numpy(vertice)
        contact.velocity = ti.Vector(np.array(velocity))
            
