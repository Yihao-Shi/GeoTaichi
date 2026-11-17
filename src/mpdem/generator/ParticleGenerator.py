from src.utils.errors import SimulationStateError

import numpy as np
import taichi as ti
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefinition import vec3f, vec3i
from src.mesh.GaussPointInTriangle import GaussPointInTriangle
from third_party.pyevtk.hl import pointsToVTK


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
        self.save_path = ""

        self.start_time = 0.0
        self.end_time = 0.0
        self.next_generate_time = 0.0
        self.insert_interval = 1e10
        self.tries_number = 0
        self.porosity = 0.345
        self.is_poisson = False
        self.type = None
        self.btype = None
        self.name = None
        self.check_hist = False
        self.write_file = False
        self.visualize = False
        self.template_dict = None

        self.faces = np.array([], dtype=np.int32)

        self.FIX = {"Free": 0, "Fix": 1}

    def no_print(self):
        self.log = False

    def deactivate(self):
        self.active = False

    def set_system_strcuture(self, body_dict):
        self.type = DictIO.get_essential(body_dict, "GenerateType")
        self.name = DictIO.get_essential(body_dict, "RegionName")
        self.template_dict = DictIO.get_essential(body_dict, "Template")
        period = DictIO.get_alternative(body_dict, "Period", [self.sims.current_time, self.sims.current_time, 1e6])
        self.start_time = period[0]
        self.end_time = period[1]
        self.insert_interval = period[2]
        self.check_hist = DictIO.get_alternative(body_dict, "CheckHistory", True)
        self.write_file = DictIO.get_alternative(body_dict, "WriteFile", False)
        self.visualize = DictIO.get_alternative(body_dict, "Visualize", False)
        self.save_path = DictIO.get_alternative(body_dict, "SavePath", "")
        self.tries_number = DictIO.get_alternative(body_dict, "TryNumber", 100)
        self.is_poisson = DictIO.get_alternative(body_dict, "PoissonSampling", False)

    def set_template(self, template_ptr):
        self.myTemplate = template_ptr

    def set_region(self, region):
        self.region = DictIO.get_essential(region, self.name)

    def print_particle_info(
        self, groupID, matID, init_v, init_w, fix_v=None, fix_w=None, body_num=0, insert_volume=0, name=None
    ):
        if body_num == 0:
            raise SimulationStateError("Zero Particles are inserted into region!")
        if name:
            print("Template name: ", name)
        print("Group ID = ", groupID)
        print("Material ID = ", matID)
        print("Body Number: ", body_num)
        if insert_volume != 0:
            print("Actual Void Fraction: ", 1 - insert_volume / self.region.cal_volume())
        print("Initial Velocity = ", init_v)
        print("Initial Angular Velocity = ", init_w)
        if fix_v:
            print("Fixed Velocity = ", fix_v)
        if fix_w:
            print("Fixed Angular Velocity = ", fix_w)
        print("\n")

    def finalize(self):
        self.snode_tree.destroy()
        del self.region, self.active, self.snode_tree
        del (
            self.start_time,
            self.end_time,
            self.insert_interval,
            self.type,
            self.name,
            self.check_hist,
            self.template_dict,
            self.write_file,
        )

    def reset(self):
        if self.type == "Generate" and self.neighbor.destroy is True:
            self.neighbor.clear()
        self.region.dem_reset()

    def begin(self, scene: SceneManager):
        if self.sims.current_time < self.next_generate_time:
            return 0
        if self.sims.current_time < self.start_time or self.sims.current_time > self.end_time:
            return 0
        if self.type == "Generate":
            self.generate_l_sbodys(scene)
        elif self.type == "Lattice":
            self.lattice_l_sbodys(scene)
        else:
            lists = ["Generate", "Distribute"]
            raise SimulationStateError(
                f"Invalid Keyword:: /GenerateType/: {self.type}. Only the following {lists} are valid"
            )

        if self.visualize and not scene.particle is None and not self.write_file:
            self.scene_visualization(scene)
        elif self.visualize and self.write_file:
            self.generator_visualization()

        self.reset()

        if (
            self.sims.current_time + self.next_generate_time > self.end_time
            or self.insert_interval > self.sims.time
            or self.end_time > self.sims.time
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
        print("#", "Start adding Level-set(s) ......")
        self.add_levelsets_to_scene(scene)
        self.reset()

        if (
            self.sims.current_time + self.next_generate_time > self.end_time
            or self.insert_interval > self.sims.time
            or self.end_time > self.sims.time
            or self.end_time == 0
            or self.start_time > self.end_time
        ):
            self.deactivate()
        else:
            self.next_generate_time = self.sims.current_time + self.insert_interval
        return 1

    def scene_visualization(self, scene: SceneManager):
        position = scene.particle.x.to_numpy()[0 : int(scene.particleNum[0])]
        posx, posy, posz = (
            np.ascontiguousarray(position[:, 0]),
            np.ascontiguousarray(position[:, 1]),
            np.ascontiguousarray(position[:, 2]),
        )
        body_id = np.ascontiguousarray(scene.particle.multisphereIndex.to_numpy()[0 : int(scene.particleNum[0])])
        group_id = np.ascontiguousarray(scene.particle.groupID.to_numpy()[0 : int(scene.particleNum[0])])
        rad = np.ascontiguousarray(scene.particle.rad.to_numpy()[0 : int(scene.particleNum[0])])
        pointsToVTK(
            self.save_path + "DEMPackings", posx, posy, posz, data={"bodyID": body_id, "group": group_id, "rad": rad}
        )

    def generator_visualization(self):
        if self.btype == "Sphere":
            position = self.sphere_coords.to_numpy()[0 : self.insert_particle_in_neighbor[None]]
            posx, posy, posz = (
                np.ascontiguousarray(position[:, 0]),
                np.ascontiguousarray(position[:, 1]),
                np.ascontiguousarray(position[:, 2]),
            )
            rad = np.ascontiguousarray(self.sphere_radii.to_numpy()[0 : self.insert_particle_in_neighbor[None]])
            pointsToVTK(self.save_path + "DEMPackings", posx, posy, posz, data={"rad": rad})
        elif self.btype == "Clump":
            position = self.pebble_coords.to_numpy()[0 : self.insert_particle_in_neighbor[None]]
            posx, posy, posz = (
                np.ascontiguousarray(position[:, 0]),
                np.ascontiguousarray(position[:, 1]),
                np.ascontiguousarray(position[:, 2]),
            )
            rad = np.ascontiguousarray(self.pebble_radii.to_numpy()[0 : self.insert_particle_in_neighbor[None]])
            pointsToVTK(self.save_path + "DEMPackings", posx, posy, posz, data={"rad": rad})

    # ========================================================= #
    #                        LEVELSET                           #
    # ========================================================= #
    def insert_soft_levelset(self, scene: SceneManager, template, start_body_num, end_body_num, body_count):
        bounding_sphere = scene.get_bounding_sphere()
        bounding_box = scene.get_bounding_box()
        surface = scene.get_surface()
        rigid_body = scene.get_rigid_ptr()
        material = scene.get_material_ptr()
        particle_num = int(scene.particleNum[0])
        surface_num = int(scene.surfaceNum[0])

        group_id = DictIO.get_essential(template, "GroupID")
        mat_id = DictIO.get_essential(template, "MaterialID")
        init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.get_alternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.get_alternative(template, "FixMotion", ["Free", "Free", "Free"])
        npic = DictIO.get_alternative(template, "nParticlesPerCell", 1)
        is_fix = vec3i([DictIO.get_essential(self.FIX, i) for i in fix_str])

        name = DictIO.get_essential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)

        gauss = GaussPointInTriangle(npic)
        scene.check_soft_body_number(self.sims, body_number=body_count)
        kernel_add_levelset_packing(
            rigid_body,
            bounding_box,
            bounding_sphere,
            surface,
            material,
            particle_num,
            gridNum,
            surface_num,
            verticeNum,
            vec3f(template_ptr.objects.grid.min_box()),
            vec3f(template_ptr.objects.grid.max_box()),
            template_ptr.boundings.r_bound,
            vec3f(template_ptr.boundings.x_bound),
            template_ptr.surface_node_number,
            template_ptr.surface_area,
            template_ptr.objects.grid.grid_space,
            vec3i(template_ptr.objects.grid.gnum),
            template_ptr.objects.grid.extent,
            vec3f(template_ptr.objects.inertia),
            template_ptr.objects.eqradius,
            group_id,
            mat_id,
            init_v,
            init_w,
            is_fix,
            start_body_num,
            end_body_num,
            self.sphere_coords,
            self.sphere_radii,
            self.orients,
            False,
        )
        print(" Level-set body Information ".center(71, "-"))
        self.print_particle_info(group_id, mat_id, init_v, init_w, fix_v=is_fix, fix_w=is_fix, body_num=body_count)

        faces = scene.add_connectivity(body_count, template_ptr.surface_node_number, template_ptr.objects)
        scene.particleNum[0] += body_count
        scene.softNum[0] += body_count
        scene.surfaceNum[0] += template_ptr.surface_node_number * body_count
        self.faces = np.append(self.faces, faces).reshape(-1, 3)
