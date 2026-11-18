from src.dem.generator.AffineBodyTemplate import AffineBodyTemplate
from src.utils.errors import ConfigurationError, GeometryError, SimulationStateError

import copy
import math
import os
import numpy as np
import taichi as ti
from src.dem.generator.BruteNeighbor import BruteSearch
from src.dem.generator.InsertionKernel import *
from src.dem.generator.LinkedCellNeighbor import LinkedCell
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.dem.generator.ClumpTemplate import ClumpTemplate
from src.dem.SceneManager import SceneManager
from src.dem.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.sorting.ParallelSort import parallel_sort_with_two_values
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefinition import vec3f, vec3i
from src.utils.Orientation import Orientation
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
        self.write_path = "."
        self.visualize = False
        self.template_dict = None
        self.radius_dist = []
        self.template_num = 0
        self.coordinates_are_mass_centers = False
        self.diffipc_options = {}
        self.diffipc_enabled = True
        self.last_diffipc_result = None
        self.diffipc_initializer = None

        self.faces = np.array([], dtype=np.int32)

        self.FIX = {"Free": 0, "Fix": 1}

    def no_print(self):
        self.log = False

    def deactivate(self):
        self.active = False

    def set_system_strcuture(self, body_dict):
        self.type = DictIO.get_essential(body_dict, "GenerateType")
        self.btype = DictIO.get_essential(body_dict, "BodyType")
        self.name = DictIO.get_essential(body_dict, "RegionName")
        self.template_dict = DictIO.get_essential(body_dict, "Template")
        period = DictIO.get_alternative(body_dict, "Period", [self.sims.current_time, self.sims.current_time, 1e6])
        self.start_time = period[0]
        self.end_time = period[1]
        self.insert_interval = period[2]
        self.check_hist = DictIO.get_alternative(body_dict, "CheckHistory", True)
        self.write_file = DictIO.get_alternative(body_dict, "WriteFile", False)
        self.write_path = DictIO.get_alternative(body_dict, "WritePath", self.write_path)
        self.visualize = DictIO.get_alternative(body_dict, "Visualize", False)
        self.save_path = DictIO.get_alternative(body_dict, "SavePath", "")
        self.tries_number = DictIO.get_alternative(body_dict, "TryNumber", 100)
        self.is_poisson = DictIO.get_alternative(body_dict, "PoissonSampling", False)
        self.porosity = DictIO.get_alternative(body_dict, "Porosity", 0.345)
        diffipc = DictIO.get_alternative(
            body_dict,
            "DiffIPC",
            DictIO.get_alternative(body_dict, "DifferentiableIPC", {}),
        )
        if isinstance(diffipc, bool):
            self.diffipc_enabled = diffipc
            self.diffipc_options = {}
        elif isinstance(diffipc, dict):
            self.diffipc_enabled = bool(DictIO.get_alternative(diffipc, "Enabled", True))
            self.diffipc_options = diffipc
        else:
            raise TypeError("DiffIPC must be a boolean or dictionary")

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

    def _soft_particle_backend(self, scene):
        backend = getattr(scene, "soft_particle_backend", None)
        if backend is None:
            raise SimulationStateError(
                "SoftBody generation is an MPM soft-particle mode. Use MPDEM/MPM soft_particle instead of DEM generator."
            )
        return backend

    def generate_soft_bodys(self, scene):
        return self._soft_particle_backend(scene).generate_soft_bodys(self, scene)

    def lattice_soft_bodys(self, scene):
        return self._soft_particle_backend(scene).lattice_soft_bodys(self, scene)

    def add_soft_levelsets_to_scene(self, scene):
        return self._soft_particle_backend(scene).add_soft_levelsets_to_scene(self, scene)

    def begin(self, scene: SceneManager):
        if self.sims.current_time < self.next_generate_time:
            return 0
        if self.sims.current_time < self.start_time or self.sims.current_time > self.end_time:
            return 0
        if self.btype == "Sphere":
            print("#", "Start adding sphere(s) ......")
            if self.type == "Generate":
                self.generate_spheres(scene)
            elif self.type == "Distribute":
                self.distribute_spheres(scene)
            elif self.type == "Lattice":
                self.lattice_spheres(scene)
            else:
                lists = ["Generate", "Distribute"]
                raise SimulationStateError(
                    f"Invalid Keyword:: /GenerateType/: {self.type}. Only the following {lists} are valid"
                )
        elif self.btype == "Clump":
            print("#", "Start adding Clump(s) ......")
            if self.type == "Generate":
                self.generate_multispheres(scene)
            elif self.type == "Distribute":
                self.distribute_multispheres(scene)
            else:
                lists = ["Generate", "Distribute"]
                raise SimulationStateError(
                    f"Invalid Keyword:: /GenerateType/: {self.type}. Only the following {lists} are valid"
                )
        elif self.btype == "RigidBody":
            if self.type == "Generate":
                self.generate_l_sbodys(scene)
            elif self.type == "Lattice":
                self.lattice_l_sbodys(scene)
            else:
                lists = ["Generate", "Distribute"]
                raise SimulationStateError(
                    f"Invalid Keyword:: /GenerateType/: {self.type}. Only the following {lists} are valid"
                )
        elif self.btype == "AffineBody":
            if self.type == "Generate":
                self.generate_affine_rigid_bodies(scene)
            elif self.type == "Distribute":
                self.distribute_affine_rigid_bodies(scene)
            else:
                lists = ["Generate", "Distribute"]
                raise SimulationStateError(
                    f"Invalid Keyword:: /GenerateType/: {self.type}. Only the following {lists} are valid"
                )
        elif self.btype == "SoftBody":
            if self.type == "Generate":
                self.generate_soft_bodys(scene)
            elif self.type == "Lattice":
                self.lattice_soft_bodys(scene)
            else:
                lists = ["Generate", "Lattice"]
                raise SimulationStateError(
                    f"Invalid Keyword:: /GenerateType/: {self.type}. Only the following {lists} are valid"
                )
        else:
            lists = ["Sphere", "Clump", "RigidBody", "AffineBody", "SoftBody"]
            raise SimulationStateError(
                f"Invalid Keyword:: /BodyType/: {self.btype}. Only the following {lists} are valid"
            )

        if self.visualize and not scene.particle is None and not self.write_file:
            self.scene_visualization(scene)
        elif self.visualize and self.write_file:
            self.generator_visualization()

        self.reset()

        if (
            self.sims.current_time + self.insert_interval > self.end_time
            or self.insert_interval > self.sims.time
            or self.end_time == 0
            or self.start_time > self.end_time
        ):
            self.deactivate()
            if self.end_time > self.sims.time:
                self.end_time = 1.0 * self.sims.time
        else:
            self.next_generate_time = self.sims.current_time + self.insert_interval
        return 1

    def regenerate(self, scene: SceneManager):
        if self.sims.current_time < self.next_generate_time:
            return 0
        if self.sims.current_time < self.start_time or self.sims.current_time > self.end_time:
            return 0
        if self.btype == "Sphere":
            print("#", "Start adding sphere(s) ......")
            self.add_spheres_to_scene(scene)
        elif self.btype == "Clump":
            print("#", "Start adding Clump(s) ......")
            self.add_clumps_to_scene(scene)
        elif self.btype == "RigidBody":
            print("#", "Start adding Level-set(s) ......")
            self.add_levelsets_to_scene(scene)
        elif self.btype == "SoftBody":
            print("#", "Start adding level-set MPM soft body(s) ......")
            self.add_soft_levelsets_to_scene(scene)
        self.reset()

        if self.sims.current_time + self.insert_interval > self.end_time:
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
                field_builder.dense(ti.i, sphere_num).place(
                    self.sphere_coords, self.sphere_radii, self.orients, self.valid
                )
            else:
                field_builder.dense(ti.i, sphere_num).place(self.sphere_coords, self.sphere_radii, self.orients)
        else:
            if lattice:
                self.valid = ti.field(int)
                field_builder.dense(ti.i, sphere_num).place(self.sphere_coords, self.sphere_radii, self.valid)
            else:
                field_builder.dense(ti.i, sphere_num).place(self.sphere_coords, self.sphere_radii)
        self.snode_tree = field_builder.finalize()
        if lattice:
            self.remains = sphere_num
        if generate or distribute:
            self.insert_body_num = ti.field(int, shape=())
            self.insert_particle_in_neighbor = ti.field(int, shape=())

    def get_radius_distribution(self, temp_dict, body_number):
        radius_dist = DictIO.get_alternative(temp_dict, "RadiusDistribution", None)
        if radius_dist is None:
            min_radius = DictIO.get_essential(temp_dict, "MinRadius", "Radius")
            max_radius = DictIO.get_essential(temp_dict, "MaxRadius", "Radius")
            if min_radius > max_radius:
                raise SimulationStateError(
                    f"Keyword:: /MinRadius/ must not be larger than /MaxRadius/ in Template {self.template_num}!"
                )
            radius_dist = np.random.uniform(min_radius, max_radius, body_number)
        else:
            radius_dist = np.asarray(radius_dist)
            if len(radius_dist) < body_number:
                raise ConfigurationError(
                    f"The length of /RadiusDistribution/: {len(radius_dist)} must be equal to /BodyNumber/: {body_number} in Template {self.template_num}!"
                )
        return radius_dist

    def generate_spheres(self, scene: SceneManager):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        clump = scene.get_clump_ptr()
        sphere_num = int(scene.sphereNum[0])
        clump_num = int(scene.clumpNum[0])

        insert_num = 0
        if type(self.template_dict) is dict:
            insert_num += DictIO.get_essential(self.template_dict, "BodyNumber")
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                insert_num += DictIO.get_essential(temp, "BodyNumber")
        self.hist_check_by_number(particle, sphere, clump, sphere_num, clump_num, insert_num)
        particle_in_region = insert_num + self.region.inserted_particle_num

        if type(self.template_dict) is dict:
            body_number = DictIO.get_essential(self.template_dict, "BodyNumber")
            self.radius_dist.append(self.get_radius_distribution(self.template_dict, body_number))
            self.template_num = 1
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                body_number = DictIO.get_essential(temp, "BodyNumber")
                self.radius_dist.append(self.get_radius_distribution(temp, body_number))
                self.template_num += 1
        min_radius, max_radius = [], []
        for i in range(self.template_num):
            min_radius.append(np.min(self.radius_dist[i]))
            max_radius.append(np.max(self.radius_dist[i]))

        if particle_in_region < 1000:
            if self.neighbor is None:
                self.neighbor = BruteSearch()
            self.neighbor.neighbor_init(particle_in_region)
        elif particle_in_region >= 1000:
            if self.neighbor is None:
                self.neighbor = LinkedCell()
            self.neighbor.neighbor_init(min(min_radius), max(max_radius), self.region.region_size, particle_in_region)
        self.allocate_sphere_memory(insert_num, generate=True)

        if self.check_hist and self.region.inserted_particle_num > 0:
            self.neighbor.pre_neighbor_sphere(
                sphere_num,
                self.insert_particle_in_neighbor,
                particle,
                sphere,
                self.region.function,
                self.region.start_point,
            )
            self.neighbor.pre_neighbor_clump(
                clump_num,
                self.insert_particle_in_neighbor,
                particle,
                clump,
                self.region.function,
                self.region.start_point,
            )

        if type(self.template_dict) is dict:
            self.generate_template_spheres(scene, self.template_dict, self.radius_dist[0])
        elif type(self.template_dict) is list:
            for temp, radius_dist in zip(self.template_dict, self.radius_dist):
                self.generate_template_spheres(scene, temp, radius_dist)

    def hist_check_by_number(self, particle, sphere, clump, sphereNum, clumpNum, insert_num):
        if insert_num > 0.0:
            if self.check_hist and self.region.inserted_particle_num == 0:
                reval = vec2i([0, 0])
                if sphere:
                    reval += kernel_update_particle_number_by_sphere(sphereNum, sphere, particle, self.region.function)
                if clump:
                    reval += kernel_update_pebble_number_by_clump(clumpNum, clump, particle, self.region.function)
                self.region.add_inserted_body(reval[0])
                self.region.add_inserted_particle(reval[1])
        else:
            raise SimulationStateError(f"BodyNumber should be set in the dictionary: {self.name}!")

    def hist_check_by_volume(self, particle, sphere, clump, sphereNum, clumpNum):
        if self.check_hist and self.region.inserted_particle_num == 0 and (sphere or clump):
            reval = vec2f([0, 0])
            if sphere:
                reval += kernel_update_particle_volume_by_sphere(sphereNum, sphere, particle, self.region.function)
            if clump:
                reval += kernel_update_particle_volume_by_clump(clumpNum, clump, particle, self.region.function)
            self.region.add_inserted_particle_volume(reval[0])
            self.region.add_inserted_particle(reval[1])

    def generate_template_spheres(self, scene: SceneManager, template, radius_dist):
        actual_body = DictIO.get_essential(template, "BodyNumber")

        start_body_num = self.insert_body_num[None]
        self.generate_sphere(actual_body, start_body_num, radius_dist)
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num

        # self.rotate_sphere_packing(start_body_num, end_body_num)
        if self.write_file:
            self.write_sphere_text(start_body_num, end_body_num)
        else:
            self.insert_sphere(scene, template, start_body_num, end_body_num, body_count)

    def lattice_spheres(self, scene: SceneManager):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        clump = scene.get_clump_ptr()
        sphere_num = int(scene.sphereNum[0])
        clump_num = int(scene.clumpNum[0])

        total_fraction = 0.0
        if type(self.template_dict) is dict:
            fraction = DictIO.get_alternative(self.template_dict, "Fraction", 1.0)
            total_fraction += fraction
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                fraction = DictIO.get_alternative(temp, "Fraction", 1.0)
                total_fraction += fraction
        if total_fraction < 0.0 or total_fraction > 1.0:
            raise ValueError("Fraction value error")

        if type(self.template_dict) is dict:
            self.radius_dist.append(self.get_radius_distribution(self.template_dict, insert_num))
            self.template_num = 1
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.radius_dist.append(self.get_radius_distribution(temp, insert_num))
                self.template_num += 1

        min_radius, max_radius = [], []
        for i in range(self.template_num):
            min_radius.append(np.min(self.radius_dist[i]))
            max_radius.append(np.max(self.radius_dist[i]))
        min_rad, max_rad = min(min_radius), max(max_radius)

        insert_particle = np.floor(0.5 * np.array(self.region.region_size) / max_rad).astype(np.int32)
        insert_num = int(insert_particle[0] * insert_particle[1] * insert_particle[2])
        self.hist_check_by_number(particle, sphere, clump, sphere_num, clump_num, insert_num)
        particle_in_region = insert_num + self.region.inserted_particle_num

        if particle_in_region < 1000:
            if self.neighbor is None:
                self.neighbor = BruteSearch()
            self.neighbor.neighbor_init(particle_in_region)
        elif particle_in_region >= 1000:
            if self.neighbor is None:
                self.neighbor = LinkedCell()
            self.neighbor.neighbor_init(min_rad, max_rad, self.region.region_size, particle_in_region)
        self.allocate_sphere_memory(insert_num, generate=True, levelset=True, lattice=True)

        if self.check_hist and self.region.inserted_particle_num > 0:
            self.neighbor.pre_neighbor_sphere(
                sphere_num,
                self.insert_particle_in_neighbor,
                particle,
                sphere,
                self.region.function,
                self.region.start_point,
            )
            self.neighbor.pre_neighbor_clump(
                clump_num,
                self.insert_particle_in_neighbor,
                particle,
                clump,
                self.region.function,
                self.region.start_point,
            )

        if type(self.template_dict) is dict:
            self.lattice_template_sphere(scene, self.template_dict, insert_particle, self.radius_dist[0])
        elif type(self.template_dict) is list:
            for temp, radius_dist in zip(self.template_dict, self.radius_dist):
                self.lattice_template_sphere(scene, temp, insert_particle, radius_dist)

    def lattice_template_sphere(self, scene: SceneManager, template, insert_particle, radius_dist):
        min_radii, max_radii = [], []
        for i in range(self.template_num):
            min_radii.append(np.min(radius_dist))
            max_radii.append(np.max(radius_dist))
        min_radius, max_radius = min(min_radii), max(max_radii)

        fraction = DictIO.get_alternative(template, "Fraction", 1.0)
        actual_body = int(fraction * int(insert_particle[0] * insert_particle[1] * insert_particle[2]))
        start_body_num = self.insert_body_num[None]
        self.lattice_sphere(min_radius, max_radius, actual_body, start_body_num, insert_particle, radius_dist)
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num

        if self.write_file:
            self.write_sphere_text(start_body_num, end_body_num)
        else:
            self.insert_sphere(scene, template, start_body_num, end_body_num, body_count)

    def add_spheres_to_scene(self, scene: SceneManager):
        if type(self.template_dict) is dict:
            self.insert_sphere(scene, self.template_dict, 0, self.insert_body_num[None], self.insert_body_num[None])
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.insert_sphere(scene, temp, 0, self.insert_body_num[None], self.insert_body_num[None])

    def generate_sphere(self, actual_body, start_body_num, radius_dist):
        min_rad, max_rad = np.min(radius_dist), np.max(radius_dist)
        if self.is_poisson:
            if self.insert_particle_in_neighbor[None] - start_body_num == 0:
                position = self.region.start_point + 0.5 * self.region.region_size
                radius = 0.5 * (max_rad + min_rad)
            else:
                location = self.insert_particle_in_neighbor[None] - 1
                position = self.neighbor.position[location]
                radius = self.neighbor.radius[location]

            kernel_insert_first_sphere(
                self.region.start_point,
                position,
                radius,
                self.insert_body_num,
                self.insert_particle_in_neighbor,
                self.sphere_coords,
                self.sphere_radii,
                self.neighbor.cell_num,
                self.neighbor.cell_size,
                self.neighbor.position,
                self.neighbor.radius,
                self.neighbor.num_particle_in_cell,
                self.neighbor.particle_neighbor,
                self.neighbor.insert_particle,
            )
            kernel_sphere_poisson_sampling(
                radius_dist,
                self.tries_number,
                actual_body + start_body_num,
                self.region.start_point,
                self.insert_body_num,
                self.insert_particle_in_neighbor,
                self.sphere_coords,
                self.sphere_radii,
                self.neighbor.cell_num,
                self.neighbor.cell_size,
                self.neighbor.position,
                self.neighbor.radius,
                self.neighbor.num_particle_in_cell,
                self.neighbor.particle_neighbor,
                self.region.function,
                self.neighbor.overlap,
                self.neighbor.insert_particle,
            )
        elif not self.is_poisson:
            kernel_sphere_generate_without_overlap(
                radius_dist,
                self.tries_number,
                actual_body + start_body_num,
                self.region.start_point,
                self.region.region_size,
                self.insert_body_num,
                self.insert_particle_in_neighbor,
                self.sphere_coords,
                self.sphere_radii,
                self.neighbor.cell_num,
                self.neighbor.cell_size,
                self.neighbor.position,
                self.neighbor.radius,
                self.neighbor.num_particle_in_cell,
                self.neighbor.particle_neighbor,
                self.region.function,
                self.neighbor.overlap,
                self.neighbor.insert_particle,
            )

    def lattice_sphere(self, actual_body, start_body_num, insert_particle, radius_dist):
        max_rad = np.max(radius_dist)
        if self.insert_body_num[None] == 0:
            fill_valid(self.valid)
        kernel_sphere_generate_lattice(
            max_rad,
            radius_dist,
            actual_body + start_body_num,
            self.remains,
            vec3i(insert_particle),
            self.region.start_point,
            self.valid,
            self.insert_body_num,
            self.insert_particle_in_neighbor,
            self.sphere_coords,
            self.sphere_radii,
            self.neighbor.cell_num,
            self.neighbor.cell_size,
            self.neighbor.position,
            self.neighbor.radius,
            self.neighbor.num_particle_in_cell,
            self.neighbor.particle_neighbor,
            self.region.function,
            self.neighbor.overlap,
            self.neighbor.insert_particle,
        )
        self.remains = update_valid(self.remains, self.valid)

    def lattice_level_set(
        self,
        actual_body,
        start_body_num,
        insert_particle,
        radius_dist,
        template_ptr: GeneralShapeTemplate,
        set_orientations: Orientation,
    ):
        max_rad = np.max(radius_dist)
        minbox = (
            template_ptr.objects.grid.min_box()
            + template_ptr.objects.grid.extent * template_ptr.objects.grid.grid_space
        )
        maxbox = (
            template_ptr.objects.grid.max_box()
            - template_ptr.objects.grid.extent * template_ptr.objects.grid.grid_space
        )
        if self.insert_body_num[None] == 0:
            fill_valid(self.valid)
        kernel_rigid_generate_lattice(
            max_rad,
            radius_dist,
            actual_body + start_body_num,
            self.remains,
            vec3i(insert_particle),
            self.region.start_point,
            self.valid,
            self.insert_body_num,
            self.insert_particle_in_neighbor,
            self.sphere_coords,
            self.sphere_radii,
            self.orients,
            vec3f(template_ptr.boundings.x_bound),
            template_ptr.boundings.r_bound,
            vec3f(minbox),
            vec3f(maxbox),
            self.neighbor.cell_num,
            self.neighbor.cell_size,
            self.neighbor.position,
            self.neighbor.radius,
            self.neighbor.orient,
            self.neighbor.num_particle_in_cell,
            self.neighbor.particle_neighbor,
            self.region.function,
            self.neighbor.overlap,
            self.neighbor.insert_particle,
            set_orientations.get_orientation,
        )
        self.remains = update_valid(self.remains, self.valid)

    def distribute_spheres(self, scene: SceneManager):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        clump = scene.get_clump_ptr()
        sphere_num = int(scene.sphereNum[0])
        clump_num = int(scene.clumpNum[0])

        if self.porosity >= 0.2594:
            self.region.calculate_expected_particle_volume(self.porosity)
        else:
            raise SimulationStateError(f"Porosity should be set in the dictionary: {self.name}!")

        total_fraction, insert_particle = 0.0, 0
        if type(self.template_dict) is dict:
            fraction = DictIO.get_alternative(self.template_dict, "Fraction", 1.0)
            min_rad = 0.0
            if "RadiusDistribution" in self.template_dict:
                radius_dist = np.asarray(DictIO.get_essential(self.template_dict, "RadiusDistribution"))
                min_rad = np.min(radius_dist)
            else:
                min_rad = DictIO.get_essential(self.template_dict, "MinRadius", "Radius")
            template_vol = self.region.estimate_body_volume(fraction)
            insert_particle += math.ceil(template_vol / (4.0 / 3.0 * PI * min_rad * min_rad * min_rad))
            total_fraction += fraction
            self.radius_dist.append(self.get_radius_distribution(self.template_dict, insert_particle))
            self.template_num = 1
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                fraction = DictIO.get_alternative(temp, "Fraction", 1.0)
                min_rad = 0.0
                if "RadiusDistribution" in temp:
                    radius_dist = np.asarray(DictIO.get_essential(temp, "RadiusDistribution"))
                    min_rad = np.min(radius_dist)
                else:
                    min_rad = DictIO.get_essential(temp, "MinRadius", "Radius")
                template_vol = self.region.estimate_body_volume(fraction)
                insert_particle += math.ceil(template_vol / (4.0 / 3.0 * PI * min_rad * min_rad * min_rad))
                total_fraction += fraction
                self.radius_dist.append(self.get_radius_distribution(temp, insert_particle))
                self.template_num += 1
        if total_fraction < 0.0 or total_fraction > 1.0:
            raise ValueError("Fraction value error")
        self.hist_check_by_volume(particle, sphere, clump, sphere_num, clump_num)
        self.region.estimate_body_volume(total_fraction)
        self.allocate_sphere_memory(insert_particle, distribute=True)

        if type(self.template_dict) is dict:
            self.distribute_template_spheres(scene, self.template_dict, self.radius_dist[0])
        elif type(self.template_dict) is list:
            for temp, radius_dist in zip(self.template_dict, self.radius_dist):
                self.distribute_template_spheres(scene, temp, radius_dist)

    def distribute_template_spheres(self, scene: SceneManager, template, radius_dist):
        fraction = DictIO.get_alternative(template, "Fraction", 1.0)
        actual_volume = fraction * self.region.expected_particle_volume
        start_body_num = self.insert_body_num[None]
        insert_volume = self.distribute_sphere(actual_volume, radius_dist)
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num

        # self.rotate_sphere_packing(start_body_num, end_body_num)
        if self.write_file:
            self.write_sphere_text(start_body_num, end_body_num)
        else:
            self.insert_sphere(scene, template, start_body_num, end_body_num, body_count, insert_volume)

    def insert_sphere(self, scene: SceneManager, template, start_body_num, end_body_num, body_count, insert_volume=0):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        material = scene.get_material_ptr()
        sphere_num = int(scene.sphereNum[0])
        particle_num = int(scene.particleNum[0])

        group_id = DictIO.get_essential(template, "GroupID")
        mat_id = DictIO.get_essential(template, "MaterialID")
        init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.get_alternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_v_str = DictIO.get_alternative(template, "FixVelocity", ["Free", "Free", "Free"])
        fix_w_str = DictIO.get_alternative(template, "FixAngularVelocity", ["Free", "Free", "Free"])
        scene.check_particle_num(self.sims, particle_number=body_count)
        scene.check_sphere_number(self.sims, body_number=body_count)
        fix_v = vec3i([DictIO.get_essential(self.FIX, i) for i in fix_v_str])
        fix_w = vec3i([DictIO.get_essential(self.FIX, i) for i in fix_w_str])
        kernel_add_sphere_packing(
            particle,
            sphere,
            material,
            sphere_num,
            particle_num,
            start_body_num,
            end_body_num,
            self.sphere_coords,
            self.sphere_radii,
            group_id,
            mat_id,
            init_v,
            init_w,
            fix_v,
            fix_w,
        )
        print(" Sphere(s) Information ".center(71, "-"))
        self.print_particle_info(
            group_id,
            mat_id,
            init_v,
            init_w,
            fix_v=fix_v_str,
            fix_w=fix_w_str,
            body_num=body_count,
            insert_volume=insert_volume,
        )

        scene.sphereNum[0] += body_count
        scene.particleNum[0] += body_count

    def rotate_sphere_packing(self, start_body_num, end_body_num):
        kernel_position_rotate(
            self.region.zdirection, self.region.rotate_center, self.sphere_coords, start_body_num, end_body_num
        )

    def distribute_sphere(self, actual_volume, radius_dist):
        return kernel_distribute_sphere(
            radius_dist,
            actual_volume,
            self.insert_body_num,
            self.insert_particle_in_neighbor,
            self.sphere_coords,
            self.sphere_radii,
            self.region.start_point,
            self.region.region_size,
            self.region.function,
        )

    def write_sphere_text(self, to_start, to_end):
        print("#", "Writing sphere(s) into 'SpherePacking' ......")
        print(f"Inserted Sphere Number: {to_end - to_start}", "\n")
        position = self.sphere_coords.to_numpy()[to_start:to_end]
        radius = self.sphere_radii.to_numpy()[to_start:to_end]
        if not os.path.exists(self.write_path + "/SpherePacking.txt"):
            np.savetxt(
                self.write_path + "/SpherePacking.txt",
                np.column_stack((position, radius)),
                header="     PositionX            PositionY                PositionZ            Radius",
                delimiter=" ",
            )
        else:
            with open(self.write_path + "/SpherePacking.txt", "ab") as file:
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
        clump_ptr = DictIO.get_optional(self.myTemplate, name)
        if not clump_ptr:
            raise KeyError(f"Template name: {name} is not set before")
        return clump_ptr

    def generate_multispheres(self, scene: SceneManager):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        clump = scene.get_clump_ptr()
        sphere_num = int(scene.sphereNum[0])
        clump_num = int(scene.clumpNum[0])

        insert_num, insert_particle, min_radius, max_radius = 0, 0, [], []
        if type(self.template_dict) is dict:
            name = DictIO.get_essential(self.template_dict, "Name")
            template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
            body_num = DictIO.get_essential(self.template_dict, "BodyNumber")
            scale_factor_min = DictIO.get_essential(self.template_dict, "MinRadius", "Radius") / template_ptr.r_equiv
            scale_factor_max = DictIO.get_essential(self.template_dict, "MaxRadius", "Radius") / template_ptr.r_equiv
            min_radius.append(scale_factor_min * template_ptr.pebble_radius_min)
            max_radius.append(scale_factor_max * template_ptr.pebble_radius_max)
            insert_num += body_num
            insert_particle += body_num * template_ptr.nspheres
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                name = DictIO.get_essential(temp, "Name")
                template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
                body_num = DictIO.get_essential(temp, "BodyNumber")
                scale_factor_min = DictIO.get_essential(temp, "MinRadius", "Radius") / template_ptr.r_equiv
                scale_factor_max = DictIO.get_essential(temp, "MaxRadius", "Radius") / template_ptr.r_equiv
                min_radius.append(scale_factor_min * template_ptr.pebble_radius_min)
                max_radius.append(scale_factor_max * template_ptr.pebble_radius_max)
                insert_num += body_num
                insert_particle += body_num * template_ptr.nspheres
        self.hist_check_by_number(particle, sphere, clump, sphere_num, clump_num, insert_num)
        particle_in_region = insert_particle + self.region.inserted_particle_num

        if particle_in_region < 1000:
            if self.neighbor is None:
                self.neighbor = BruteSearch()
            self.neighbor.neighbor_init(particle_in_region)
        elif particle_in_region >= 1000:
            if self.neighbor is None:
                self.neighbor = LinkedCell()
            self.neighbor.neighbor_init(min(min_radius), max(max_radius), self.region.region_size, particle_in_region)
        self.allocate_clump_memory(insert_num, insert_particle, generate=True)

        if self.check_hist and self.region.inserted_particle_num > 0:
            self.neighbor.pre_neighbor_sphere(
                sphere_num,
                self.insert_particle_in_neighbor,
                particle,
                sphere,
                self.region.function,
                self.region.start_point,
            )
            self.neighbor.pre_neighbor_clump(
                clump_num,
                self.insert_particle_in_neighbor,
                particle,
                clump,
                self.region.function,
                self.region.start_point,
            )

        if type(self.template_dict) is dict:
            self.generate_template_multispheres(scene, self.template_dict)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.generate_template_multispheres(scene, temp)

    def generate_template_multispheres(self, scene: SceneManager, template):
        particle_num = int(scene.particleNum[0])

        name = DictIO.get_essential(template, "Name")
        template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)

        actual_body = DictIO.get_essential(template, "BodyNumber")
        min_radius = DictIO.get_essential(template, "MinRadius", "Radius")
        max_radius = DictIO.get_essential(template, "MaxRadius", "Radius")
        orientation = DictIO.get_alternative(template, "BodyOrientation", None)
        set_orientations = Orientation(orientation)

        if min_radius > max_radius:
            raise SimulationStateError("Keyword:: /MinRadius/ must not be larger than /MaxRadius/")

        start_body_num, start_pebble_num = self.insert_body_num[None], self.insert_particle_in_neighbor[None]
        self.generate_multi_sphere(
            min_radius, max_radius, actual_body + start_body_num, start_pebble_num, template_ptr, set_orientations
        )
        end_body_num, end_pebble_num = self.insert_body_num[None], self.insert_particle_in_neighbor[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_pebble_num

        # self.rotate_clump_packing(start_body_num, end_body_num, start_pebble_num, end_pebble_num, template_ptr.nspheres)
        if self.write_file:
            self.write_clump_text(template_ptr, start_body_num, end_body_num, particle_num)
        else:
            self.insert_multispheres(
                scene, template, start_body_num, start_pebble_num, end_body_num, end_pebble_num, body_count
            )

    def add_clumps_to_scene(self, scene: SceneManager):
        if type(self.template_dict) is dict:
            self.insert_multispheres(
                scene,
                self.template_dict,
                0,
                0,
                self.insert_body_num[None],
                self.insert_particle_in_neighbor[None],
                self.insert_body_num[None],
            )
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.insert_multispheres(
                    scene,
                    temp,
                    0,
                    0,
                    self.insert_body_num[None],
                    self.insert_particle_in_neighbor[None],
                    self.insert_body_num[None],
                )

    def generate_multi_sphere(
        self,
        min_rad,
        max_rad,
        actual_body,
        start_pebble_num,
        template_ptr: ClumpTemplate,
        set_orientations: Orientation,
    ):
        if self.is_poisson:
            if self.insert_particle_in_neighbor[None] - start_pebble_num == 0:
                position = self.region.start_point + 0.5 * self.region.region_size
                radius = 0.5 * (max_rad + min_rad)
            else:
                location = self.insert_particle_in_neighbor[None] - 1
                position = self.neighbor.position[location]
                radius = self.neighbor.radius[location]

            kernel_insert_first_multisphere(
                self.region.start_point,
                template_ptr.nspheres,
                template_ptr.r_equiv,
                template_ptr.x_pebble,
                template_ptr.rad_pebble,
                position,
                radius,
                self.insert_body_num,
                self.insert_particle_in_neighbor,
                self.clump_coords,
                self.clump_radii,
                self.clump_orients,
                set_orientations.get_orientation,
                self.pebble_coords,
                self.pebble_radii,
                self.neighbor.cell_num,
                self.neighbor.cell_size,
                self.neighbor.position,
                self.neighbor.radius,
                self.neighbor.num_particle_in_cell,
                self.neighbor.particle_neighbor,
                self.neighbor.insert_particle,
            )
            kernel_multisphere_poisson_sampling(
                template_ptr,
                min_rad,
                max_rad,
                self.tries_number,
                actual_body,
                self.region.start_point,
                self.insert_body_num,
                self.insert_particle_in_neighbor,
                self.clump_coords,
                self.clump_radii,
                self.clump_orients,
                set_orientations.get_orientation,
                self.pebble_coords,
                self.pebble_radii,
                self.neighbor.cell_num,
                self.neighbor.cell_size,
                self.neighbor.position,
                self.neighbor.radius,
                self.neighbor.num_particle_in_cell,
                self.neighbor.particle_neighbor,
                self.region.function,
                self.neighbor.overlap,
                self.neighbor.insert_particle,
            )
        elif not self.is_poisson:
            kernel_multisphere_generate_without_overlap(
                template_ptr.nspheres,
                template_ptr.r_equiv,
                template_ptr.x_pebble,
                template_ptr.rad_pebble,
                min_rad,
                max_rad,
                self.tries_number,
                actual_body,
                self.insert_body_num,
                self.insert_particle_in_neighbor,
                self.clump_coords,
                self.clump_radii,
                self.clump_orients,
                set_orientations.get_orientation,
                self.pebble_coords,
                self.pebble_radii,
                self.neighbor.cell_num,
                self.neighbor.cell_size,
                self.neighbor.position,
                self.neighbor.radius,
                self.neighbor.num_particle_in_cell,
                self.neighbor.particle_neighbor,
                self.region.function,
                self.region.start_point,
                self.region.region_size,
                self.neighbor.overlap,
                self.neighbor.insert_particle,
            )

    def distribute_multispheres(self, scene: SceneManager):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        clump = scene.get_clump_ptr()
        sphere_num = int(scene.sphereNum[0])
        clump_num = int(scene.clumpNum[0])

        if self.porosity >= 0.01:
            self.region.calculate_expected_particle_volume(self.porosity)
        else:
            raise SimulationStateError(f"Porosity should be set in the dictionary: {self.name}!")

        total_fraction, insert_num, insert_particle = 0.0, 0, 0
        if type(self.template_dict) is dict:
            name = DictIO.get_essential(self.template_dict, "Name")
            template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
            fraction = DictIO.get_alternative(self.template_dict, "Fraction", 1.0)
            min_rad = DictIO.get_essential(self.template_dict, "MinRadius", "Radius")
            template_vol = self.region.estimate_body_volume(fraction)
            insert_num += math.ceil(template_vol / (4.0 / 3.0 * PI * min_rad * min_rad * min_rad))
            insert_particle += (
                math.ceil(template_vol / (4.0 / 3.0 * PI * min_rad * min_rad * min_rad)) * template_ptr.nspheres
            )
            total_fraction += fraction
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                name = DictIO.get_essential(temp, "Name")
                template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
                fraction = DictIO.get_alternative(temp, "Fraction", 1.0)
                min_rad = DictIO.get_essential(temp, "MinRadius", "Radius")
                template_vol = self.region.estimate_body_volume(fraction)
                insert_num += math.ceil(template_vol / (4.0 / 3.0 * PI * min_rad * min_rad * min_rad))
                insert_particle += (
                    math.ceil(template_vol / (4.0 / 3.0 * PI * min_rad * min_rad * min_rad)) * template_ptr.nspheres
                )
                total_fraction += fraction
        if total_fraction < 0.0 or total_fraction > 1.0:
            raise ValueError("Fraction value error")

        self.hist_check_by_volume(particle, sphere, clump, sphere_num, clump_num)
        self.region.estimate_body_volume(total_fraction)
        self.allocate_clump_memory(insert_num, insert_particle, distribute=True)

        if type(self.template_dict) is dict:
            self.distribute_template_multispheres(scene, self.template_dict)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.distribute_template_multispheres(scene, temp)

    def distribute_template_multispheres(self, scene: SceneManager, template):
        particle_num = int(scene.particleNum[0])

        name = DictIO.get_essential(template, "Name")
        template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)

        fraction = DictIO.get_alternative(template, "Fraction", 1.0)
        min_radius = DictIO.get_essential(template, "MinRadius", "Radius")
        max_radius = DictIO.get_essential(template, "MaxRadius", "Radius")
        orientation = DictIO.get_alternative(template, "BodyOrientation", None)
        set_orientations = Orientation(orientation)

        actual_volume = fraction * self.region.expected_particle_volume
        start_body_num, start_pebble_num = self.insert_body_num[None], self.insert_particle_in_neighbor[None]
        insert_volume = self.distribute_multi_sphere(
            min_radius, max_radius, actual_volume, template_ptr, set_orientations
        )
        end_body_num, end_pebble_num = self.insert_body_num[None], self.insert_particle_in_neighbor[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_pebble_num

        # self.rotate_clump_packing(start_body_num, end_body_num, start_pebble_num, end_pebble_num, template_ptr.nspheres)
        if self.write_file:
            self.write_clump_text(template_ptr, start_body_num, end_body_num, particle_num)
        else:
            self.insert_multispheres(
                scene,
                template,
                start_body_num,
                start_pebble_num,
                end_body_num,
                end_pebble_num,
                body_count,
                insert_volume,
            )

    def insert_multispheres(
        self,
        scene: SceneManager,
        template,
        start_body_num,
        start_pebble_num,
        end_body_num,
        end_pebble_num,
        body_count,
        insert_volume=0,
    ):
        particle = scene.get_particle_ptr()
        clump = scene.get_clump_ptr()
        material = scene.get_material_ptr()
        clump_num = int(scene.clumpNum[0])
        particle_num = int(scene.particleNum[0])

        name = DictIO.get_essential(template, "Name")
        template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)

        group_id = DictIO.get_essential(template, "GroupID")
        mat_id = DictIO.get_essential(template, "MaterialID")
        init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.get_alternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        scene.check_particle_num(self.sims, particle_number=end_pebble_num - start_pebble_num)
        scene.check_clump_number(self.sims, body_number=body_count)
        kernel_add_multisphere_packing(
            particle,
            clump,
            material,
            clump_num,
            particle_num,
            start_body_num,
            end_body_num,
            start_pebble_num,
            template_ptr,
            self.clump_coords,
            self.clump_radii,
            self.clump_orients,
            self.pebble_coords,
            self.pebble_radii,
            group_id,
            mat_id,
            init_v,
            init_w,
        )
        print(" Clump(s) Information ".center(71, "-"))
        self.print_particle_info(group_id, mat_id, init_v, init_w, body_num=body_count, insert_volume=insert_volume)
        scene.clumpNum[0] += body_count
        scene.particleNum[0] += body_count * template_ptr.nspheres

    def distribute_multi_sphere(self, min_rad, max_rad, actual_volume, template_ptr: ClumpTemplate, set_orientations):
        return kernel_distribute_multisphere(
            template_ptr.nspheres,
            template_ptr.r_equiv,
            template_ptr.volume_expect,
            template_ptr.x_pebble,
            template_ptr.rad_pebble,
            min_rad,
            max_rad,
            actual_volume,
            self.insert_body_num,
            self.insert_particle_in_neighbor,
            self.clump_coords,
            self.clump_radii,
            self.clump_orients,
            set_orientations.get_orientation,
            self.pebble_coords,
            self.pebble_radii,
            self.region.start_point,
            self.region.region_size,
            self.region.function,
        )

    def rotate_clump_packing(self, start_body_num, end_body_num, start_pebble_num, end_pebble_num, nsphere):
        kernel_position_rotate(
            self.region.zdirection, self.region.rotate_center, self.clump_coords, start_body_num, end_body_num
        )
        kernel_position_rotate(
            self.region.zdirection, self.region.rotate_center, self.pebble_coords, start_pebble_num, end_pebble_num
        )

    def write_clump_text(self, template_ptr: ClumpTemplate, start_body_num, end_body_num, particleNum):
        print("#", "Writing clump(s) into 'ClumpPacking' ......")
        body_num = end_body_num - start_body_num
        print(f"Inserted Clump Number: {body_num}")
        print(f"Inserted Particle Number: {body_num * template_ptr.nspheres}", "\n")

        position = self.clump_coords.to_numpy()[start_body_num:end_body_num]
        radius = self.clump_radii.to_numpy()[start_body_num:end_body_num]
        orientation = self.clump_orients.to_numpy()[start_body_num:end_body_num]
        inertia_vol = np.outer((radius / template_ptr.r_equiv) ** 5, template_ptr.inertia)
        pebble_index1 = np.arange(particleNum, particleNum + body_num * template_ptr.nspheres, template_ptr.nspheres)
        pebble_index2 = (
            np.arange(
                particleNum + template_ptr.nspheres,
                particleNum + (body_num + 1) * template_ptr.nspheres,
                template_ptr.nspheres,
            )
            - 1
        )

        particle_pos = self.pebble_coords.to_numpy()[particleNum : particleNum + body_num * template_ptr.nspheres]
        particle_rad = self.pebble_radii.to_numpy()[particleNum : particleNum + body_num * template_ptr.nspheres]
        multisphere_index = np.repeat(np.arange(start_body_num, end_body_num, 1), template_ptr.nspheres)

        if not os.path.exists(self.write_path + "/ClumpPacking.txt"):
            np.savetxt(
                self.write_path + "/ClumpPacking.txt",
                np.column_stack((position, radius, orientation, inertia_vol, pebble_index1, pebble_index2)),
                header="     PositionX            PositionY                PositionZ                Radius                OrientX                OrientY                OrientZ                moi_volX                moi_volY                moi_volZ                startIndex                endIndex",
                delimiter=" ",
            )
        else:
            with open(self.write_path + "/ClumpPacking.txt", "ab") as file:
                np.savetxt(
                    file,
                    np.column_stack((position, radius, orientation, inertia_vol, pebble_index1, pebble_index2)),
                    delimiter=" ",
                )

        if not os.path.exists(self.write_path + "/PebblePacking.txt"):
            np.savetxt(
                self.write_path + "/PebblePacking.txt",
                np.column_stack((particle_pos, particle_rad, multisphere_index)),
                header="     PositionX            PositionY                PositionZ                Radius                multisphereIndex",
                delimiter=" ",
            )
        else:
            with open(self.write_path + "/PebblePacking.txt", "ab") as file:
                np.savetxt(file, np.column_stack((particle_pos, particle_rad, multisphere_index)), delimiter=" ")

    # ========================================================= #
    #                        LEVELSET                           #
    # ========================================================= #
    def hist_check_levelset_number(self, bounding_sphere, rigidNum, insert_num):
        if insert_num > 0.0:
            if self.check_hist and self.region.inserted_particle_num == 0:
                reval = vec2i([0, 0])
                if bounding_sphere:
                    reval += kernel_update_particle_number_by_levelset(rigidNum, bounding_sphere, self.region.function)
                self.region.add_inserted_body(reval[0])
                self.region.add_inserted_particle(reval[1])
        else:
            raise SimulationStateError(f"BodyNumber should be set in the dictionary: {self.name}!")

    def hist_check_levelset_volume(self, bounding_sphere, rigid, rigidNum):
        if self.check_hist and self.region.inserted_particle_num == 0 and bounding_sphere:
            reval = vec2f([0, 0])
            if bounding_sphere:
                reval += kernel_update_particle_volume_by_levelset(
                    rigidNum, bounding_sphere, rigid, self.region.function
                )
            self.region.add_inserted_particle_volume(reval[0])
            self.region.add_inserted_particle(reval[1])

    def get_orient(self, start_body_num, end_body_num, orient_function):
        kernel_get_orient(start_body_num, end_body_num, orient_function, self.orients)

    def get_bounding_radius(self, template_dict, body_number):
        radius_dist = DictIO.get_alternative(template_dict, "RadiusDistribution", None)
        if radius_dist is None:
            rad_min, rad_max = 0.0, 0.0
            if "Radius" in template_dict or ("MinRadius" in template_dict and "MaxRadius" in template_dict):
                if self.myTemplate is None:
                    raise ConfigurationError("The template must be set first")
                name = DictIO.get_essential(template_dict, "Name")
                template_ptr = self.get_template_ptr_by_name(name)
                if isinstance(template_ptr, AffineBodyTemplate):
                    weight = template_ptr.bounding_radius / max(template_ptr.eqradius, 1.0e-30)
                else:
                    weight = template_ptr.boundings.r_bound / template_ptr.objects.eqradius
                rad_min = DictIO.get_essential(template_dict, "MinRadius", "Radius") * weight
                rad_max = DictIO.get_essential(template_dict, "MaxRadius", "Radius") * weight
            elif "BoundingRadius" in template_dict or (
                "MinBoundingRadius" in template_dict and "MaxBoundingRadius" in template_dict
            ):
                rad_min = DictIO.get_essential(template_dict, "MinBoundingRadius", "BoundingRadius")
                rad_max = DictIO.get_essential(template_dict, "MaxBoundingRadius", "BoundingRadius")
            else:
                raise SimulationStateError("Failed reading bounding radius")
            if rad_min > rad_max:
                raise SimulationStateError(
                    f"Keyword:: /MinRadius/ must not be larger than /MaxRadius/ in Template {self.template_num}!"
                )
            radius_dist = np.random.uniform(rad_min, rad_max, body_number)
        else:
            radius_dist = np.asarray(radius_dist)
            if len(radius_dist) < body_number:
                raise ConfigurationError(
                    f"The length of /RadiusDistribution/: {len(radius_dist)} must be equal to /BodyNumber/: {body_number} in Template {self.template_num}!"
                )
        return radius_dist

    def get_bounding_radius_range(self, template_dict):
        radius_dist = DictIO.get_alternative(template_dict, "RadiusDistribution", None)
        if radius_dist is not None:
            radius_dist = np.asarray(radius_dist)
            return float(np.min(radius_dist)), float(np.max(radius_dist))

        if "Radius" in template_dict or ("MinRadius" in template_dict and "MaxRadius" in template_dict):
            if self.myTemplate is None:
                raise ConfigurationError("The template must be set first")
            name = DictIO.get_essential(template_dict, "Name")
            template_ptr = self.get_template_ptr_by_name(name)
            if isinstance(template_ptr, AffineBodyTemplate):
                weight = template_ptr.bounding_radius / max(template_ptr.eqradius, 1.0e-30)
            else:
                weight = template_ptr.boundings.r_bound / template_ptr.objects.eqradius
            rad_min = DictIO.get_essential(template_dict, "MinRadius", "Radius") * weight
            rad_max = DictIO.get_essential(template_dict, "MaxRadius", "Radius") * weight
        elif "BoundingRadius" in template_dict or (
            "MinBoundingRadius" in template_dict and "MaxBoundingRadius" in template_dict
        ):
            rad_min = DictIO.get_essential(template_dict, "MinBoundingRadius", "BoundingRadius")
            rad_max = DictIO.get_essential(template_dict, "MaxBoundingRadius", "BoundingRadius")
        else:
            raise SimulationStateError("Failed reading bounding radius")

        if rad_min > rad_max:
            raise SimulationStateError(
                f"Keyword:: /MinRadius/ must not be larger than /MaxRadius/ in Template {self.template_num}!"
            )
        return float(rad_min), float(rad_max)

    @staticmethod
    def _affine_rotation_matrix_radians(orientation):
        theta = np.asarray(orientation, dtype=np.float64).reshape(3)
        cx, cy, cz = np.cos(theta)
        sx, sy, sz = np.sin(theta)
        rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
            dtype=np.float64,
        )
        ry = np.array(
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
            dtype=np.float64,
        )
        rz = np.array(
            [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return rz @ ry @ rx

    def _diffipc_option(self, name, default):
        return DictIO.get_alternative(self.diffipc_options, name, default)

    def _resolve_distributed_affine_overlaps(self, scene, records):
        if not self.diffipc_enabled:
            raise SimulationStateError(
                "AffineBody Distribute intentionally creates overlaps; "
                "DiffIPC cannot be disabled for this generation mode"
            )
        from src.dem.engines.AffineBodyOperator import TaichiAffineBodyOperator
        from src.dem.engines.AffineBodyState import AffineBodyState
        from src.dem.engines.TaichiAffineDiffIPCProjector import TaichiAffineDiffIPCProjector
        from src.dem.engines.TaichiAffineMeshDiffIPCProjector import TaichiAffineMeshDiffIPCProjector

        state = AffineBodyState.from_scene(scene, self.sims)
        representations = {str(body.get("contact_representation", "TriangleMesh")) for body in state.bodies}
        if len(representations) != 1:
            raise SimulationStateError(
                "AffineBody Distribute cannot mix TriangleMesh and LevelSet " "templates in one DiffIPC projection"
            )
        representation = next(iter(representations))
        if representation not in ("TriangleMesh", "LevelSet"):
            raise SimulationStateError("AffineBody DiffIPC requires TriangleMesh or LevelSet contact")
        if representation == "LevelSet":
            for body in state.bodies:
                if body.get("levelset") is None:
                    raise SimulationStateError(
                        "LevelSet DiffIPC distribution requires a native " "signed-distance field"
                    )
        projection_sims = copy.copy(self.sims)
        projection_sims.affine_friction_mode = "lagged"
        projection_sims.affine_contact_damping_stiffness = 0.0
        operator = TaichiAffineBodyOperator(
            state,
            projection_sims,
            scene,
            initialization_only=True,
        )
        region_lower = np.asarray(self.region.start_point, dtype=np.float64)
        region_upper = region_lower + np.asarray(self.region.region_size, dtype=np.float64)
        bounds = np.zeros((state.body_num, 3, 2), dtype=np.float64)
        for body_id, (body, spec) in enumerate(zip(state.bodies, scene.affine_bodies)):
            center = np.asarray(spec["body_point"], dtype=np.float64)
            radius = float(body["scale"]) * float(body["template"].bounding_radius)
            bounds[body_id, :, 0] = region_lower + radius - center
            bounds[body_id, :, 1] = region_upper - radius - center
        solve_options = dict(
            dhat=float(self._diffipc_option("Dhat", self.sims.affine_dhat)),
            kappa=float(
                self._diffipc_option(
                    "BarrierStiffness",
                    self.sims.affine_barrier_stiffness,
                )
            ),
            anchor_stiffness=float(self._diffipc_option("AnchorStiffness", 1.0)),
            continuation_ratio=float(self._diffipc_option("ContinuationRatio", 0.1)),
            stiffness_growth=float(self._diffipc_option("StiffnessGrowth", 10.0)),
            maximum_stages=int(self._diffipc_option("MaximumStages", 10)),
            maximum_iterations=int(self._diffipc_option("MaximumIterations", 100)),
            gradient_tolerance=float(self._diffipc_option("GradientTolerance", 1.0e-8)),
            gap_tolerance=float(self._diffipc_option("GapTolerance", 1.0e-7)),
            linear_tolerance=float(self._diffipc_option("LinearTolerance", 1.0e-6)),
            linear_maximum_iterations=int(self._diffipc_option("LinearMaximumIterations", 200)),
            line_search_maximum_iterations=int(self._diffipc_option("LineSearchMaximumIterations", 40)),
            hessian_shift=float(self._diffipc_option("HessianShift", 1.0e-9)),
        )
        if representation == "TriangleMesh":
            for body in state.bodies:
                mesh = body["template"].objects.mesh
                if not bool(getattr(mesh, "is_watertight", False)):
                    raise SimulationStateError(
                        "TriangleMesh untangling requires watertight particle "
                        "templates so strict containment can be detected"
                    )
                if not bool(getattr(mesh, "is_winding_consistent", False)):
                    raise SimulationStateError(
                        "TriangleMesh untangling requires consistently wound " "particle templates"
                    )
            projector = TaichiAffineMeshDiffIPCProjector(operator, state.y, bounds)
            untangle_result = projector.untangle(
                gaussian_epsilon=float(self._diffipc_option("GaussianEpsilon", solve_options["dhat"])),
                separation_weight=float(self._diffipc_option("SeparationWeight", 10.0)),
                gaussian_weight=float(self._diffipc_option("GaussianWeight", 1.0)),
                gaussian_cutoff=float(self._diffipc_option("GaussianCutoff", 4.0 * solve_options["dhat"])),
                minkowski_weight=float(self._diffipc_option("MinkowskiWeight", 1.0)),
                inclusion_weight=float(self._diffipc_option("InclusionWeight", 1.0)),
                anchor_stiffness=float(self._diffipc_option("AnchorStiffness", 1.0)),
                stiffness_growth=float(
                    self._diffipc_option(
                        "SeparationStiffnessGrowth",
                        self._diffipc_option("UntangleStiffnessGrowth", 10.0),
                    )
                ),
                maximum_stages=int(self._diffipc_option("UntangleMaximumStages", 6)),
                maximum_iterations=int(self._diffipc_option("UntangleMaximumIterations", 250)),
                gradient_tolerance=float(self._diffipc_option("UntangleGradientTolerance", 1.0e-9)),
                line_search_maximum_iterations=int(self._diffipc_option("LineSearchMaximumIterations", 40)),
                step_fraction=float(self._diffipc_option("UntangleStepFraction", 0.25)),
                intersection_tolerance=float(self._diffipc_option("IntersectionTolerance", 1.0e-10)),
            )
            if not untangle_result["success"]:
                raise GeometryError(
                    "Affine-body untangling could not remove all initial "
                    "overlaps; remaining="
                    f"{untangle_result['overlap_count']} "
                    "(triangle intersections="
                    f"{untangle_result['triangle_intersections']}, inside "
                    f"vertices={untangle_result['inside_vertices']}, "
                    f"iterations={untangle_result['iterations']}, "
                    f"stages={untangle_result['continuation_stages']})"
                )
            result = projector.solve(
                ccd_type=str(self._diffipc_option("CCDType", "accd")),
                ccd_eta=float(self._diffipc_option("CCDEta", 0.1)),
                accd_tolerance=float(self._diffipc_option("ACCDTolerance", 1.0e-9)),
                ccd_maximum_iterations=int(self._diffipc_option("CCDMaximumIterations", 50)),
                **solve_options,
            )
        else:
            projector = TaichiAffineDiffIPCProjector(operator, state.y, bounds)
            result = projector.solve(**solve_options)
        self.diffipc_initializer = projector
        self.last_diffipc_result = result
        scene.last_affine_diffipc_projector = projector
        scene.last_affine_diffipc_result = result
        if not result["success"]:
            raise SimulationStateError(
                "DiffIPC could not construct a nonpenetrating affine-body "
                f"packing at porosity {self.porosity}; minimum gap="
                f"{result['minimum_gap']:.6e}"
            )
        projector.apply_to_generated_positions(self.sphere_coords)
        for body_id, spec in enumerate(scene.affine_bodies):
            translation = projector.translation[body_id]
            for component in range(3):
                spec["body_point"][component] += float(translation[component])
        return result

    def distribute_affine_rigid_bodies(self, scene: SceneManager):
        """Porosity-driven random placement followed by DiffIPC projection."""
        if not (0.0 < float(self.porosity) < 1.0):
            raise ValueError("AffineBody Distribute Porosity must lie in (0, 1)")
        if len(scene.affine_bodies) != 0:
            raise SimulationStateError(
                "AffineBody Distribute currently requires an empty affine "
                "scene so every generated body participates in DiffIPC"
            )
        self.region.calculate_expected_particle_volume(self.porosity)
        templates = [self.template_dict] if isinstance(self.template_dict, dict) else list(self.template_dict)
        total_fraction = sum(float(DictIO.get_alternative(template, "Fraction", 1.0)) for template in templates)
        if total_fraction <= 0.0 or total_fraction > 1.0 + 1.0e-12:
            raise ValueError("AffineBody template fractions must sum to (0, 1]")

        specifications = []
        total_capacity = 0
        for template in templates:
            fraction = float(DictIO.get_alternative(template, "Fraction", 1.0))
            name = DictIO.get_essential(template, "Name")
            template_ptr = self.get_template_ptr_by_name(name)
            if not isinstance(template_ptr, AffineBodyTemplate):
                raise SimulationStateError("AffineBody Distribute needs AffineBody templates")
            minimum_radius, unused_maximum = self.get_bounding_radius_range(template)
            minimum_scale = minimum_radius / max(template_ptr.bounding_radius, 1.0e-30)
            minimum_volume = template_ptr.volume * minimum_scale**3
            if minimum_volume <= 0.0:
                raise ValueError("AffineBody template volume must be positive")
            target_volume = fraction * self.region.expected_particle_volume
            capacity = max(int(math.ceil(target_volume / minimum_volume)) + 1, 1)
            radius_distribution = self.get_bounding_radius(template, capacity)
            specifications.append(
                {
                    "template": template,
                    "template_ptr": template_ptr,
                    "target_volume": target_volume,
                    "capacity": capacity,
                    "radius_distribution": radius_distribution,
                }
            )
            total_capacity += capacity

        scene.check_rigid_body_number(self.sims, rigid_body_number=total_capacity)
        self.allocate_sphere_memory(total_capacity, distribute=True, levelset=True)
        records = []
        for specification in specifications:
            template = specification["template"]
            template_ptr = specification["template_ptr"]
            orientation = DictIO.get_alternative(template, "BodyOrientation", None)
            orientation_distribution = Orientation(orientation)
            start = int(self.insert_body_num[None])
            inserted_volume = kernel_distribute_affine_body(
                specification["radius_distribution"],
                specification["target_volume"],
                template_ptr.volume,
                template_ptr.bounding_radius,
                self.insert_body_num,
                self.insert_particle_in_neighbor,
                self.sphere_coords,
                self.sphere_radii,
                self.orients,
                self.region.start_point,
                self.region.region_size,
                self.region.function,
                orientation_distribution.get_orientation,
            )
            end = int(self.insert_body_num[None])
            if inserted_volume + 1.0e-14 < specification["target_volume"]:
                raise SimulationStateError(
                    "AffineBody radius distribution was exhausted before " "the requested solid volume was reached"
                )
            records.append(
                {
                    **specification,
                    "start": start,
                    "end": end,
                    "inserted_volume": float(inserted_volume),
                }
            )

        for record in records:
            body_count = record["end"] - record["start"]
            self.insert_affine_rigid_body(
                scene,
                record["template"],
                record["template_ptr"],
                record["start"],
                record["end"],
                body_count,
            )
        self._resolve_distributed_affine_overlaps(scene, records)
        self.region.inserted_body_num = int(self.insert_body_num[None])
        self.region.inserted_particle_num = int(self.insert_body_num[None])

    def generate_affine_rigid_bodies(self, scene: SceneManager):
        insert_num = 0
        if type(self.template_dict) is dict:
            insert_num += DictIO.get_essential(self.template_dict, "BodyNumber")
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                insert_num += DictIO.get_essential(temp, "BodyNumber")

        if type(self.template_dict) is dict:
            self.radius_dist.append(self.get_bounding_radius(self.template_dict, insert_num))
            self.template_num = 1
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                body_number = DictIO.get_essential(temp, "BodyNumber")
                self.radius_dist.append(self.get_bounding_radius(temp, body_number))
                self.template_num += 1

        min_radius, max_radius = [], []
        for i in range(self.template_num):
            min_radius.append(np.min(self.radius_dist[i]))
            max_radius.append(np.max(self.radius_dist[i]))
        min_rad, max_rad = min(min_radius), max(max_radius)

        mixed_lsmpm_ipc = self.sims.scheme == "LSMPM" and getattr(self.sims, "lsmpm_soft_rigid_contact", "DEM") == "IPC"
        if not mixed_lsmpm_ipc:
            scene.check_rigid_body_number(self.sims, rigid_body_number=insert_num)
        particle_in_region = insert_num + self.region.inserted_particle_num
        if particle_in_region < 1000:
            if self.neighbor is None:
                self.neighbor = BruteSearch(rigid=True)
            self.neighbor.neighbor_init(particle_in_region)
        else:
            if self.neighbor is None:
                self.neighbor = LinkedCell(rigid=True)
            self.neighbor.neighbor_init(min_rad, max_rad, self.region.region_size, particle_in_region)
        self.allocate_sphere_memory(insert_num, generate=True, levelset=True)

        if type(self.template_dict) is dict:
            self.generate_template_affine_rigid_body(scene, self.template_dict, self.radius_dist[0])
        elif type(self.template_dict) is list:
            for temp, radius_dist in zip(self.template_dict, self.radius_dist):
                self.generate_template_affine_rigid_body(scene, temp, radius_dist)

    def generate_template_affine_rigid_body(self, scene: SceneManager, template, radius_dist):
        actual_body = DictIO.get_essential(template, "BodyNumber")
        orientation = DictIO.get_alternative(template, "BodyOrientation", None)
        set_orientations = Orientation(orientation)
        name = DictIO.get_essential(template, "Name")
        template_ptr: AffineBodyTemplate = self.get_template_ptr_by_name(name)
        if not isinstance(template_ptr, AffineBodyTemplate):
            raise SimulationStateError("AffineBody generate expects an AffineBody template.")

        start_body_num = self.insert_body_num[None]
        self.generate_affine_rigid_body(actual_body, start_body_num, radius_dist, template_ptr, set_orientations)
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num

        if self.write_file:
            self.write_body_text(start_body_num, end_body_num)
        else:
            self.insert_affine_rigid_body(scene, template, template_ptr, start_body_num, end_body_num, body_count)

    def generate_affine_rigid_body(
        self,
        actual_body,
        start_body_num,
        radius_dist,
        template_ptr: AffineBodyTemplate,
        set_orientations: Orientation,
    ):
        min_rad, max_rad = np.min(radius_dist), np.max(radius_dist)
        if self.is_poisson:
            if self.insert_particle_in_neighbor[None] - start_body_num == 0:
                position = self.region.start_point + 0.5 * self.region.region_size
                radius = 0.5 * (max_rad + min_rad)
            else:
                location = self.insert_particle_in_neighbor[None] - 1
                position = self.neighbor.position[location]
                radius = self.neighbor.radius[location]

            kernel_insert_first_rigid(
                self.region.start_point,
                position,
                radius,
                self.insert_body_num,
                self.insert_particle_in_neighbor,
                self.sphere_coords,
                self.sphere_radii,
                self.orients,
                self.neighbor.cell_num,
                self.neighbor.cell_size,
                self.neighbor.position,
                self.neighbor.radius,
                self.neighbor.orient,
                self.neighbor.num_particle_in_cell,
                self.neighbor.particle_neighbor,
                self.neighbor.insert_particle,
                set_orientations.get_orientation,
            )
            kernel_rigid_poisson_sampling(
                radius_dist,
                self.tries_number,
                actual_body + start_body_num,
                self.region.start_point,
                self.insert_body_num,
                self.insert_particle_in_neighbor,
                self.sphere_coords,
                self.sphere_radii,
                self.orients,
                vec3f(template_ptr.x_bound),
                template_ptr.bounding_radius,
                vec3f(template_ptr.min_box),
                vec3f(template_ptr.max_box),
                self.neighbor.cell_num,
                self.neighbor.cell_size,
                self.neighbor.position,
                self.neighbor.radius,
                self.neighbor.orient,
                self.neighbor.num_particle_in_cell,
                self.neighbor.particle_neighbor,
                self.region.function,
                self.neighbor.overlap,
                self.neighbor.insert_particle,
                set_orientations.get_orientation,
            )
        elif not self.is_poisson:
            kernel_rigid_generate_without_overlap(
                radius_dist,
                self.tries_number,
                actual_body + start_body_num,
                self.region.start_point,
                self.region.region_size,
                self.insert_body_num,
                self.insert_particle_in_neighbor,
                self.sphere_coords,
                self.sphere_radii,
                self.orients,
                vec3f(template_ptr.x_bound),
                template_ptr.bounding_radius,
                vec3f(template_ptr.min_box),
                vec3f(template_ptr.max_box),
                self.neighbor.cell_num,
                self.neighbor.cell_size,
                self.neighbor.position,
                self.neighbor.radius,
                self.neighbor.orient,
                self.neighbor.num_particle_in_cell,
                self.neighbor.particle_neighbor,
                self.region.function,
                self.neighbor.overlap,
                self.neighbor.insert_particle,
                set_orientations.get_orientation,
            )

    def insert_affine_rigid_body(
        self, scene: SceneManager, template, template_ptr: AffineBodyTemplate, start_body_num, end_body_num, body_count
    ):
        group_id = DictIO.get_essential(template, "GroupID")
        mat_id = DictIO.get_essential(template, "MaterialID")
        init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.get_alternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        young = DictIO.get_alternative(template, "YoungModulus", None)
        friction = DictIO.get_alternative(template, "Friction", 0.0)
        local_damping = DictIO.get_alternative(template, "LocalDamping", None)
        fdamp = DictIO.get_alternative(
            template, "ForceLocalDamping", DictIO.get_alternative(template, "force_local_damping", local_damping)
        )
        tdamp = DictIO.get_alternative(
            template, "TorqueLocalDamping", DictIO.get_alternative(template, "torque_local_damping", local_damping)
        )
        name = DictIO.get_essential(template, "Name")
        positions = self.sphere_coords.to_numpy()[start_body_num:end_body_num]
        bounding_radii = self.sphere_radii.to_numpy()[start_body_num:end_body_num]
        orientations = np.degrees(self.orients.to_numpy()[start_body_num:end_body_num])
        for body_id in range(body_count):
            scale_factor = float(bounding_radii[body_id]) / max(template_ptr.bounding_radius, 1.0e-30)
            scene.add_affine_body(
                self.sims,
                name,
                positions[body_id],
                scale_factor,
                group_id,
                mat_id,
                init_v,
                init_w,
                orientations[body_id],
                young_modulus=young,
                friction=friction,
                force_local_damping=fdamp,
                torque_local_damping=tdamp,
            )
        print(" Affine body Information ".center(71, "-"))
        self.print_particle_info(group_id, mat_id, init_v, init_w, body_num=body_count)

    def generate_l_sbodys(self, scene: SceneManager):
        if self.sims.scheme == "AffineBody":
            self.generate_affine_rigid_bodies(scene)
            return
        bounding_sphere = scene.get_bounding_sphere()
        particle_num = int(scene.particleNum[0])

        insert_num = 0
        if type(self.template_dict) is dict:
            insert_num += DictIO.get_essential(self.template_dict, "BodyNumber")
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                insert_num += DictIO.get_essential(temp, "BodyNumber")
        self.hist_check_levelset_number(bounding_sphere, particle_num, insert_num)
        particle_in_region = insert_num + self.region.inserted_particle_num

        if type(self.template_dict) is dict:
            body_number = DictIO.get_essential(self.template_dict, "BodyNumber")
            self.radius_dist.append(self.get_bounding_radius(self.template_dict, body_number))
            self.template_num = 1
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                body_number = DictIO.get_essential(temp, "BodyNumber")
                self.radius_dist.append(self.get_bounding_radius(temp, body_number))
                self.template_num += 1
        min_radius, max_radius = [], []
        for i in range(self.template_num):
            min_radius.append(np.min(self.radius_dist[i]))
            max_radius.append(np.max(self.radius_dist[i]))

        if particle_in_region < 1000:
            if self.neighbor is None:
                self.neighbor = BruteSearch(rigid=True)
            self.neighbor.neighbor_init(particle_in_region)
        elif particle_in_region >= 1000:
            if self.neighbor is None:
                self.neighbor = LinkedCell(rigid=True)
            self.neighbor.neighbor_init(min(min_radius), max(max_radius), self.region.region_size, particle_in_region)
        self.allocate_sphere_memory(insert_num, generate=True, levelset=True)

        if self.check_hist and self.region.inserted_particle_num > 0:
            self.neighbor.pre_neighbor_bounding_sphere(
                particle_num,
                self.insert_particle_in_neighbor,
                bounding_sphere,
                self.region.function,
                self.region.start_point,
            )

        if type(self.template_dict) is dict:
            self.generate_template_rigid_body(scene, self.template_dict, self.radius_dist[0])
        elif type(self.template_dict) is list:
            for temp, radius_dist in zip(self.template_dict, self.radius_dist):
                self.generate_template_rigid_body(scene, temp, radius_dist)

    def generate_template_rigid_body(self, scene: SceneManager, template, radius_dist):
        actual_body = DictIO.get_essential(template, "BodyNumber")
        orientation = DictIO.get_alternative(template, "BodyOrientation", None)
        set_orientations = Orientation(orientation)

        name = DictIO.get_essential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)

        start_body_num = self.insert_body_num[None]
        self.generate_level_set(actual_body, start_body_num, radius_dist, template_ptr, set_orientations)
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num
        # parallel_sort_with_value(self.sphere_radii, self.sphere_coords, start_body_num, body_count)

        if self.write_file:
            self.write_body_text(start_body_num, end_body_num)
        else:
            if self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                self.insert_rigid_implicit_surface(scene, template, start_body_num, end_body_num, body_count)
            else:
                self.insert_rigid_levelset(scene, template, start_body_num, end_body_num, body_count)

    def generate_level_set(
        self,
        actual_body,
        start_body_num,
        radius_dist,
        template_ptr: GeneralShapeTemplate,
        set_orientations: Orientation,
    ):
        min_rad, max_rad = np.min(radius_dist), np.max(radius_dist)
        minbox = (
            template_ptr.objects.grid.min_box()
            + template_ptr.objects.grid.extent * template_ptr.objects.grid.grid_space
        )
        maxbox = (
            template_ptr.objects.grid.max_box()
            - template_ptr.objects.grid.extent * template_ptr.objects.grid.grid_space
        )
        if self.is_poisson:
            if self.insert_particle_in_neighbor[None] - start_body_num == 0:
                position = self.region.start_point + 0.5 * self.region.region_size
                radius = 0.5 * (max_rad + min_rad)
            else:
                location = self.insert_particle_in_neighbor[None] - 1
                position = self.neighbor.position[location]
                radius = self.neighbor.radius[location]

            kernel_insert_first_rigid(
                self.region.start_point,
                position,
                radius,
                self.insert_body_num,
                self.insert_particle_in_neighbor,
                self.sphere_coords,
                self.sphere_radii,
                self.orients,
                self.neighbor.cell_num,
                self.neighbor.cell_size,
                self.neighbor.position,
                self.neighbor.radius,
                self.neighbor.orient,
                self.neighbor.num_particle_in_cell,
                self.neighbor.particle_neighbor,
                self.neighbor.insert_particle,
                set_orientations.get_orientation,
            )
            kernel_rigid_poisson_sampling(
                radius_dist,
                self.tries_number,
                actual_body + start_body_num,
                self.region.start_point,
                self.insert_body_num,
                self.insert_particle_in_neighbor,
                self.sphere_coords,
                self.sphere_radii,
                self.orients,
                vec3f(template_ptr.boundings.x_bound),
                template_ptr.boundings.r_bound,
                vec3f(minbox),
                vec3f(maxbox),
                self.neighbor.cell_num,
                self.neighbor.cell_size,
                self.neighbor.position,
                self.neighbor.radius,
                self.neighbor.orient,
                self.neighbor.num_particle_in_cell,
                self.neighbor.particle_neighbor,
                self.region.function,
                self.neighbor.overlap,
                self.neighbor.insert_particle,
                set_orientations.get_orientation,
            )
        elif not self.is_poisson:
            kernel_rigid_generate_without_overlap(
                radius_dist,
                self.tries_number,
                actual_body + start_body_num,
                self.region.start_point,
                self.region.region_size,
                self.insert_body_num,
                self.insert_particle_in_neighbor,
                self.sphere_coords,
                self.sphere_radii,
                self.orients,
                vec3f(template_ptr.boundings.x_bound),
                template_ptr.boundings.r_bound,
                vec3f(minbox),
                vec3f(maxbox),
                self.neighbor.cell_num,
                self.neighbor.cell_size,
                self.neighbor.position,
                self.neighbor.radius,
                self.neighbor.orient,
                self.neighbor.num_particle_in_cell,
                self.neighbor.particle_neighbor,
                self.region.function,
                self.neighbor.overlap,
                self.neighbor.insert_particle,
                set_orientations.get_orientation,
            )

    def lattice_l_sbodys(self, scene: SceneManager):
        bounding_sphere = scene.get_bounding_sphere()
        particle_num = int(scene.particleNum[0])

        total_fraction = 0.0
        max_rad = 0.0
        if type(self.template_dict) is dict:
            fraction = DictIO.get_alternative(self.template_dict, "Fraction", 1.0)
            _, template_max = self.get_bounding_radius_range(self.template_dict)
            total_fraction += fraction
            max_rad = max(max_rad, template_max)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                fraction = DictIO.get_alternative(temp, "Fraction", 1.0)
                _, template_max = self.get_bounding_radius_range(temp)
                total_fraction += fraction
                max_rad = max(max_rad, template_max)
        if total_fraction < 0.0 or total_fraction > 1.0:
            raise ValueError("Fraction value error")
        if max_rad <= 0.0:
            raise SimulationStateError("The maximum bounding radius should be positive")

        insert_particle = np.floor(0.5 * np.array(self.region.region_size) / max_rad).astype(np.int32)
        if np.any(insert_particle <= 0):
            raise SimulationStateError(
                "The lattice spacing is too large for the selected region and level-set body radius"
            )
        insert_num = int(insert_particle[0] * insert_particle[1] * insert_particle[2])

        if type(self.template_dict) is dict:
            self.radius_dist.append(self.get_bounding_radius(self.template_dict, insert_num))
            self.template_num = 1
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.radius_dist.append(self.get_bounding_radius(temp, insert_num))
                self.template_num += 1

        min_radius, max_radius = [], []
        for i in range(self.template_num):
            min_radius.append(np.min(self.radius_dist[i]))
            max_radius.append(np.max(self.radius_dist[i]))
        min_rad, max_rad = min(min_radius), max(max_radius)

        self.hist_check_levelset_number(bounding_sphere, particle_num, insert_num)
        particle_in_region = insert_num + self.region.inserted_particle_num

        if particle_in_region < 1000:
            if self.neighbor is None:
                self.neighbor = BruteSearch(rigid=True)
            self.neighbor.neighbor_init(particle_in_region)
        elif particle_in_region >= 1000:
            if self.neighbor is None:
                self.neighbor = LinkedCell(rigid=True)
            self.neighbor.neighbor_init(min_rad, max_rad, self.region.region_size, particle_in_region)
        self.allocate_sphere_memory(insert_num, generate=True, levelset=True, lattice=True)

        if self.check_hist and self.region.inserted_particle_num > 0:
            self.neighbor.pre_neighbor_bounding_sphere(
                particle_num,
                self.insert_particle_in_neighbor,
                bounding_sphere,
                self.region.function,
                self.region.start_point,
            )

        if type(self.template_dict) is dict:
            self.lattice_template_rigid_body(scene, self.template_dict, insert_particle, self.radius_dist[0])
        elif type(self.template_dict) is list:
            for temp, radius_dist in zip(self.template_dict, self.radius_dist):
                self.lattice_template_rigid_body(scene, temp, insert_particle, radius_dist)

    def lattice_template_rigid_body(self, scene: SceneManager, template, insert_particle, radius_dist):
        fraction = DictIO.get_alternative(template, "Fraction", 1.0)
        orientation = DictIO.get_alternative(template, "BodyOrientation", None)
        set_orientations = Orientation(orientation)
        name = DictIO.get_essential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)

        actual_body = int(fraction * int(insert_particle[0] * insert_particle[1] * insert_particle[2]))
        start_body_num = self.insert_body_num[None]
        self.lattice_level_set(
            actual_body, start_body_num, insert_particle, radius_dist, template_ptr, set_orientations
        )
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num
        parallel_sort_with_two_values(self.sphere_radii, self.sphere_coords, self.orients, start_body_num, body_count)

        if self.write_file:
            self.write_body_text(start_body_num, end_body_num)
        else:
            if self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                self.insert_rigid_implicit_surface(scene, template, start_body_num, end_body_num, body_count)
            else:
                self.insert_rigid_levelset(scene, template, start_body_num, end_body_num, body_count)

    def add_levelsets_to_scene(self, scene: SceneManager):
        if type(self.template_dict) is dict:
            if self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                self.insert_rigid_implicit_surface(
                    scene, self.template_dict, 0, self.insert_body_num[None], self.insert_body_num[None]
                )
            else:
                self.insert_rigid_levelset(
                    scene, self.template_dict, 0, self.insert_body_num[None], self.insert_body_num[None]
                )
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                if self.sims.scheme == "PolySuperEllipsoid" or self.sims.scheme == "PolySuperQuadrics":
                    self.insert_rigid_implicit_surface(
                        scene, temp, 0, self.insert_body_num[None], self.insert_body_num[None]
                    )
                else:
                    self.insert_rigid_levelset(scene, temp, 0, self.insert_body_num[None], self.insert_body_num[None])

    def insert_rigid_levelset(self, scene: SceneManager, template, start_body_num, end_body_num, body_count):
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
        is_fix = vec3i([DictIO.get_essential(self.FIX, i) for i in fix_str])

        name = DictIO.get_essential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)
        index = DictIO.get_essential(scene.prefixID, name)

        grid_num = scene.gridID[index]
        vertice_num = scene.verticeID[index]
        scene.check_rigid_body_number(self.sims, rigid_body_number=body_count)
        kernel_add_levelset_packing(
            rigid_body,
            bounding_box,
            bounding_sphere,
            surface,
            material,
            particle_num,
            grid_num,
            surface_num,
            vertice_num,
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
            self.coordinates_are_mass_centers,
        )
        print(" Level-set body Information ".center(71, "-"))
        self.print_particle_info(group_id, mat_id, init_v, init_w, fix_v=is_fix, fix_w=is_fix, body_num=body_count)

        faces = scene.add_connectivity(body_count, template_ptr.surface_node_number, template_ptr.objects)
        scene.particleNum[0] += body_count
        scene.rigidNum[0] += body_count
        scene.surfaceNum[0] += template_ptr.surface_node_number * body_count
        self.faces = np.append(self.faces, faces).reshape(-1, 3)

    def insert_rigid_implicit_surface(self, scene: SceneManager, template, start_body_num, end_body_num, body_count):
        bounding_sphere = scene.get_bounding_sphere()
        rigid_body = scene.get_rigid_ptr()
        material = scene.get_material_ptr()
        particle_num = int(scene.particleNum[0])

        group_id = DictIO.get_essential(template, "GroupID")
        mat_id = DictIO.get_essential(template, "MaterialID")
        init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.get_alternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.get_alternative(template, "FixMotion", ["Free", "Free", "Free"])
        is_fix = vec3i([DictIO.get_essential(self.FIX, i) for i in fix_str])

        name = DictIO.get_essential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)
        template_id = scene.prefixID[name]

        if self.sims.iterative_model == "LagrangianMultiplier":
            if self.sims.scheme == "PolySuperEllipsoid":
                assert (
                    0.25 <= template_ptr.objects.physical_parameter["epsilon_e"] <= 1
                ), f"The optimal value range of parameter /epsilon_e/ is 0.25 to 1"
                assert (
                    0.25 <= template_ptr.objects.physical_parameter["epsilon_n"] <= 1
                ), f"The optimal value range of parameter /epsilon_n/ is 0.25 to 1"
            elif self.sims.scheme == "PolySuperQuadrics":
                assert (
                    0.25 <= template_ptr.objects.physical_parameter["epsilon_x"] <= 1
                ), f"The optimal value range of parameter /epsilon_x/ is 0.25 to 1"
                assert (
                    0.25 <= template_ptr.objects.physical_parameter["epsilon_y"] <= 1
                ), f"The optimal value range of parameter /epsilon_y/ is 0.25 to 1"
                assert (
                    0.25 <= template_ptr.objects.physical_parameter["epsilon_z"] <= 1
                ), f"The optimal value range of parameter /epsilon_z/ is 0.25 to 1"

        scene.check_rigid_body_number(self.sims, rigid_body_number=body_count)
        kernel_add_implicit_surface_packing(
            rigid_body,
            bounding_sphere,
            material,
            particle_num,
            template_id,
            template_ptr.boundings.r_bound,
            vec3f(template_ptr.boundings.x_bound),
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
        )
        print(" Implicit surface body Information ".center(71, "-"))
        self.print_particle_info(group_id, mat_id, init_v, init_w, fix_v=is_fix, fix_w=is_fix, body_num=body_count)

        faces = scene.add_connectivity(body_count, template_ptr.surface_node_number, template_ptr.objects)
        scene.particleNum[0] += body_count
        scene.rigidNum[0] += body_count
        scene.surfaceNum[0] += template_ptr.surface_node_number * body_count
        self.faces = np.append(self.faces, faces).reshape(-1, 3)

    def write_body_text(self, to_start, to_end):
        print("#", "Writing sphere(s) into 'BoundingSphere' ......")
        print(f"Inserted Bounding Number: {to_end - to_start}", "\n")
        position = self.sphere_coords.to_numpy()[to_start:to_end]
        radius = self.sphere_radii.to_numpy()[to_start:to_end]
        orients = self.orients.to_numpy()[to_start:to_end]
        if not os.path.exists(self.write_path + "/BoundingSphere.txt"):
            np.savetxt(
                self.write_path + "/BoundingSphere.txt",
                np.column_stack((position, radius, orients)),
                header="     PositionX            PositionY                PositionZ            Radius            DirX            DirY            DirZ",
                delimiter=" ",
            )
        else:
            with open(self.write_path + "/BoundingSphere.txt", "ab") as file:
                np.savetxt(file, np.column_stack((position, radius, orients)), delimiter=" ")
