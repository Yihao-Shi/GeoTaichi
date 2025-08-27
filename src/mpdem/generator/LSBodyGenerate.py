import numpy as np
import taichi as ti

from src.dem.GenerateManager import GenerateManager as DEMGenerator
from src.dem.SceneManager import myScene as DEMScene
from src.dem.Simulation import Simulation as DEMSimulation
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.mpm.SceneManager import myScene as MPMScene
from src.mpm.Simulation import Simulation as MPMSimulation
from src.utils.ObjectIO import DictIO
from src.utils.sorting.ParallelSort import parallel_sort_with_value
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefination import vec3f, vec3i
from src.mesh.GaussPoint import GaussPointInTriangle
from src.utils.Orientation import set_orientation
from third_party.pyevtk.hl import pointsToVTK


class ParticleCreator(object):
    mscene: MPMScene
    msims: MPMSimulation
    dscene: DEMScene
    dsims: DEMSimulation
    dgenerator: DEMGenerator

    def __init__(self, mscene, msims, dscene, dsims, dgenerator) -> None:
        self.mscene = mscene
        self.msims = msims
        self.dscene = dscene
        self.dsims = dsims
        self.dgenerator = dgenerator
        self.myTemplate = None
        self.FIX = {
                    "Free": 0,
                    "Fix": 1
                   }

    def create(self, body_dict):
        template = DictIO.GetAlternative(body_dict, "Template", body_dict)
        self.create_soft_body(template)

    def create_soft_body(self, template):
        if type(template) is dict:
            self.create_template_soft_body(template)
        elif type(template) is list:
            for temp in template:
                self.create_template_soft_body(temp)

    def create_template_soft_body(self, template):
        bounding_sphere = self.dscene.get_bounding_sphere()
        bounding_box = self.dscene.get_bounding_box()
        master = self.dscene.get_surface()
        rigid_body = self.dscene.get_rigid_ptr()
        material = self.dscene.get_material_ptr()
        particleNum = int(self.dscene.particleNum[0])
        surfaceNum = int(self.dscene.surfaceNum[0])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: GeneralShapeTemplate = DictIO.GetEssential(self.dgenerator.myTemplate, name)
        com_pos = DictIO.GetEssential(template, "BodyPoint")
        equiv_rad = DictIO.GetAlternative(template, "Radius", None)
        bounding_rad = DictIO.GetAlternative(template, "BoundingRadius", None)
        scale_factor = DictIO.GetAlternative(template, "ScaleFactor", None) 
        orientation = DictIO.GetAlternative(template, "BodyOrientation", None)
        npic = DictIO.GetAlternative(template, "nParticlesPerCell", 1)
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

        gauss = GaussPointInTriangle(npic)
        self.dscene.check_soft_body_number(self.dsims, body_number=1)
        self.mscene.check_particle_num(self.msims, particle_number=npic * template_ptr.objects.tetramesh.cell.shape[0])
        kernel_create_level_set_body_(rigid_body, bounding_box, bounding_sphere, master, material, particleNum, gridNum, verticeNum, surfaceNum, vec3f(template_ptr.objects.grid.minBox()), vec3f(template_ptr.objects.grid.maxBox()), 
                                      template_ptr.boundings.r_bound, vec3f(template_ptr.boundings.x_bound), template_ptr.surface_node_number, template_ptr.objects.grid.grid_space, vec3i(template_ptr.objects.grid.gnum), 
                                      template_ptr.objects.grid.extent, scale_factor, vec3f(template_ptr.objects.inertia), com_pos, equiv_rad, set_orientations.get_orientation, groupID, matID, init_v, init_w, is_fix)
        
        print(" Level set body Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, com_pos, init_v, init_w, fix_v=is_fix, fix_w=is_fix, name=name)
        faces = np.array(template_ptr.objects.mesh.faces, dtype=np.int32)
        self.dscene.connectivity = np.append(self.dscene.connectivity, faces + self.dscene.surfaceNum[0]).reshape(-1, 3)
        self.dscene.particleNum[0] += 1
        self.dscene.softNum[0] += 1
        self.dscene.surfaceNum[0] += template_ptr.surface_node_number

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
        if self.type == "Generate":
            self.generate_LSbodys(scene)
        elif self.type == "Lattice":
            self.lattice_LSbodys(scene)
        else:
            lists = ["Generate", "Distribute"]
            raise RuntimeError(f"Invalid Keyword:: /GenerateType/: {self.type}. Only the following {lists} are valid")
     
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
        print('#', "Start adding Level-set(s) ......")
        self.add_levelsets_to_scene(scene)
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
    #                        LEVELSET                           #
    # ========================================================= #
    def insert_soft_levelset(self, scene: myScene, template, start_body_num, end_body_num, body_count):
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
        npic = DictIO.GetAlternative(template, "nParticlesPerCell", 1)
        is_fix = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_str])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)

        gauss = GaussPointInTriangle(npic)
        scene.check_soft_body_number(self.sims, body_number=body_count)
        kernel_add_levelset_packing(rigid_body, bounding_box, bounding_sphere, surface, material, particleNum, gridNum, surfaceNum, verticeNum, vec3f(template_ptr.objects.grid.minBox()), vec3f(template_ptr.objects.grid.maxBox()), template_ptr.boundings.r_bound, 
                                    vec3f(template_ptr.boundings.x_bound), template_ptr.surface_node_number, template_ptr.objects.grid.grid_space, vec3i(template_ptr.objects.grid.gnum), template_ptr.objects.grid.extent,
                                    vec3f(template_ptr.objects.inertia), template_ptr.objects.eqradius, groupID, matID, init_v, init_w, is_fix, start_body_num, end_body_num, self.sphere_coords, self.sphere_radii, self.orients)
        print(" Level-set body Information ".center(71, '-'))
        self.print_particle_info(groupID, matID, init_v, init_w, fix_v=is_fix, fix_w=is_fix, body_num=body_count)


        faces = scene.add_connectivity(body_count, template_ptr.surface_node_number, template_ptr.objects)
        scene.particleNum[0] += body_count
        scene.softNum[0] += body_count
        scene.surfaceNum[0] += template_ptr.surface_node_number * body_count
        self.faces = np.append(self.faces, faces).reshape(-1, 3)


