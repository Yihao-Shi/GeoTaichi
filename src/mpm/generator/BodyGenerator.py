import numpy as np
import hashlib
import taichi as ti
import os

from src.mpm.generator.Generator import Generator
from src.mpm.elements.ElementBase import ElementBase
from src.mpm.elements.HexahedronKernel import transform_local_to_global
from src.mpm.generator.InsertionKernel import *
from src.mpm.soft_particle.SceneFields import register_soft_template_support
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mesh.GaussPoint import GaussPointInRectangle, GaussPointInTriangle
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefination import vec3f, vec6f, vec2u8, vec3u8, vec2f
from third_party.pyevtk.hl import pointsToVTK


def compact_soft_grid_nodes_in_region(region, support, topology):
    """Resolve a template-space box to compact mechanical-grid node IDs."""
    if not isinstance(region, dict):
        raise TypeError("MechanicalGridBoundary must be a dictionary")
    if support.grid_base_space not in (0.0, support.grid_space):
        raise RuntimeError("MechanicalGridBoundary currently requires a uniform grid")
    lower = np.asarray(DictIO.GetEssential(region, "StartPoint"), dtype=np.float64)
    upper = np.asarray(DictIO.GetEssential(region, "EndPoint"), dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("MechanicalGridBoundary StartPoint and EndPoint must be 3-vectors")
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    velocity = np.asarray(
        DictIO.GetAlternative(region, "Velocity", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if velocity.shape != (3,) or not np.allclose(velocity, 0.0):
        raise ValueError("LSMPM MechanicalGridBoundary currently supports fixed zero " "velocity only")
    shape = np.asarray(topology.compact_shape, dtype=np.int64)
    offset = np.asarray(topology.compact_origin, dtype=np.int64)
    axes = tuple(
        float(support.grid_origin[d]) + (offset[d] + np.arange(shape[d], dtype=np.float64)) * float(support.grid_space)
        for d in range(3)
    )
    selected_axes = tuple(
        np.flatnonzero((axis >= lower[d] - 1.0e-12) & (axis <= upper[d] + 1.0e-12)) for d, axis in enumerate(axes)
    )
    if any(index.size == 0 for index in selected_axes):
        raise RuntimeError("MechanicalGridBoundary selects no compact mechanical-grid node")
    ii, jj, kk = np.meshgrid(*selected_axes, indexing="ij")
    local = ii + jj * shape[0] + kk * shape[0] * shape[1]
    return np.ascontiguousarray(np.unique(local), dtype=np.int32)


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
        period = DictIO.GetAlternative(body_dict, "Period", [self.sims.current_time, self.sims.current_time, 1e6])
        self.start_time = period[0]
        self.end_time = period[1]
        self.insert_interval = period[2]
        self.write_file = DictIO.GetAlternative(body_dict, "WriteFile", self.write_file)
        self.write_path = DictIO.GetAlternative(body_dict, "WritePath", self.write_path)
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

    def finalize(self):
        self.snode_tree.destroy()
        del self.myRegion, self.active  # , self.field_builder, self.snode_tree
        del self.start_time, self.end_time, self.insert_interval, self.visualize, self.write_file, self.myTemplate

    def begin(self, scene: myScene):
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

    def regenerate(self, scene: myScene):
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

    def scene_visualization(self, scene: myScene):
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
        kernel_position_rotate_(
            region.rotate, region.rotate_center, self.particle, start_particle_num, end_particle_num
        )

    def generate_material_points(self, scene: myScene, template):
        name = DictIO.GetEssential(template, "RegionName")
        nParticlesPerCell = DictIO.GetAlternative(template, "nParticlesPerCell", 2)
        region: RegionFunction = self.get_region_ptr(name)
        particle_volume = scene.element.calc_volume() / scene.element.calc_total_particle(nParticlesPerCell)
        psize = scene.element.calc_particle_size(nParticlesPerCell)

        if self.check_history:
            scene.delete_particles_in_region(region.function)
            initial_particle = self.insert_particle_num[None]
            kernel_delete_particle_slots_in_region(self.insert_particle_num[None], self.particle, region.function)
            finial_particle = self.insert_particle_num[None]
            print(f"Total {-finial_particle + initial_particle} particles has been deleted from local slot", "\n")

        start_particle_num = self.insert_particle_num[None]
        self.Generate(scene, region, nParticlesPerCell)
        end_particle_num = self.insert_particle_num[None]
        particle_count = end_particle_num - start_particle_num

        if self.write_file:
            self.write_text(start_particle_num, end_particle_num, particle_volume, psize)
            scene.element.mesh.write(self.write_path + "/Element.txt")
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
                if self.sims.random_field:
                    scene.material.matProps[materialID].read_random_field(
                        particleNum, particleNum + particle_count, scene.material.stateVars
                    )
                    density = np.ascontiguousarray(material.stateVars.density.to_numpy())
                else:
                    density = material.matProps[materialID].density

                phase = 0
                if self.sims.material_type == "TwoPhaseSingleLayer" or self.sims.material_type == "TwoPhaseDoubleLayer":
                    if not self.sims.random_field:
                        density = material.matProps[materialID].solid_density
                    densityf = material.matProps[materialID].fluid_density
                    porosity = material.matProps[materialID].porosity
                    permeability = material.matProps[materialID].permeability
                    if self.sims.material_type == "TwoPhaseDoubleLayer":
                        phase = DictIO.GetEssential(self.PHASE, DictIO.GetEssential(template, "Phase"))
                if materialID <= 0:
                    raise RuntimeError(f"Material ID {materialID} should be larger than 0")

            if isinstance(density, (np.ndarray, list, tuple)):
                density = np.asarray(density)
            elif isinstance(density, (int, float)):
                density = np.repeat(density, particle_count)

            particle_stress = DictIO.GetAlternative(template, "ParticleStress", {"InternalStress": [0, 0, 0, 0, 0, 0]})
            init_v = DictIO.GetAlternative(
                template, "InitialVelocity", [0.0, 0.0, 0.0] if self.sims.dimension == 3 else [0.0, 0.0]
            )
            fix_v_str = DictIO.GetAlternative(
                template, "FixVelocity", ["Free", "Free", "Free"] if self.sims.dimension == 3 else ["Free", "Free"]
            )
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
                if self.sims.material_type == "Solid" or self.sims.material_type == "Fluid":
                    kernel_add_body_(
                        particles,
                        particleNum,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        bodyID,
                        materialID,
                        density,
                        init_v,
                        fix_v,
                    )
                elif self.sims.material_type == "TwoPhaseSingleLayer" and rigid_body:
                    kernel_add_body_twophase_rigid(
                        particles,
                        particleNum,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        bodyID,
                        materialID,
                        density,
                        init_v,
                        fix_v,
                    )
                elif self.sims.material_type == "TwoPhaseSingleLayer":
                    kernel_add_body_twophase(
                        particles,
                        particleNum,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        bodyID,
                        materialID,
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
                        particleNum,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        bodyID,
                        materialID,
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
                    kernel_add_body_2D(
                        particles,
                        particleNum,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        bodyID,
                        materialID,
                        density,
                        init_v,
                        fix_v,
                    )
                elif self.sims.material_type == "TwoPhaseSingleLayer" and rigid_body:
                    kernel_add_body_2D_rigid(
                        particles,
                        particleNum,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        bodyID,
                        materialID,
                        density,
                        init_v,
                        fix_v,
                        self.sims.axis_offset,
                    )
                elif self.sims.material_type == "TwoPhaseSingleLayer":
                    kernel_add_body_twophase2D(
                        particles,
                        particleNum,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        bodyID,
                        materialID,
                        density,
                        densityf,
                        porosity,
                        permeability,
                        init_v,
                        fix_v,
                        self.sims.axis_offset,
                    )
                elif self.sims.material_type == "TwoPhaseDoubleLayer":
                    kernel_add_body_twophase_double_point2D(
                        particles,
                        particleNum,
                        start_particle_num,
                        end_particle_num,
                        self.particle,
                        particle_volume,
                        bodyID,
                        materialID,
                        phase,
                        density,
                        densityf,
                        porosity,
                        permeability,
                        init_v,
                        fix_v,
                        self.sims.axis_offset,
                    )
            self.set_particle_stress(scene, particleNum, particle_count, particle_stress)
            scene.push_psize(np.repeat([psize], particle_count, axis=0))
            traction = DictIO.GetAlternative(template, "Traction", {})
            self.set_traction(particle_count, traction, scene, region)
            scene.material.state_vars_initialize(materialID, particleNum, particleNum + particle_count, scene.particle)
            self.print_particle_info(
                bodyID, materialID, init_v, fix_v_str, particle_count, particle_volume, nParticlesPerCell
            )
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
            if self.sims.random_field:
                scene.material.matProps[materialID].read_random_field(
                    particleNum, particleNum + self.insert_particle_num[None], scene.material.stateVars
                )
                density = np.ascontiguousarray(material.stateVars.density.to_numpy())
            else:
                density = material.matProps[materialID].density

            phase = 0
            if self.sims.material_type == "TwoPhaseSingleLayer" or self.sims.material_type == "TwoPhaseDoubleLayer":
                if not self.sims.random_field:
                    density = material.matProps[materialID].solid_density
                densityf = material.matProps[materialID].fluid_density
                porosity = material.matProps[materialID].porosity
                permeability = material.matProps[materialID].permeability
                if self.sims.material_type == "TwoPhaseDoubleLayer":
                    phase = DictIO.GetEssential(self.PHASE, DictIO.GetEssential(template, "Phase"))
            if materialID <= 0:
                raise RuntimeError(f"Material ID {materialID} should be larger than 0")

        if isinstance(density, (np.ndarray, list, tuple)):
            density = np.asarray(density)
        elif isinstance(density, (int, float)):
            density = np.repeat(density, self.insert_particle_num[None])

        particle_stress = DictIO.GetAlternative(template, "ParticleStress", {"InternalStress": [0, 0, 0, 0, 0, 0]})
        init_v = DictIO.GetAlternative(
            template, "InitialVelocity", [0.0, 0.0, 0.0] if self.sims.dimension == 3 else [0.0, 0.0]
        )
        fix_v_str = DictIO.GetAlternative(
            template, "FixVelocity", ["Free", "Free", "Free"] if self.sims.dimension == 3 else ["Free", "Free"]
        )
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
            if self.sims.material_type == "Solid" or self.sims.material_type == "Fluid":
                kernel_add_body_(
                    particles,
                    particleNum,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    bodyID,
                    materialID,
                    density,
                    init_v,
                    fix_v,
                )
            elif self.sims.material_type == "TwoPhaseSingleLayer" and rigid_body:
                kernel_add_body_twophase_rigid(
                    particles,
                    particleNum,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    bodyID,
                    materialID,
                    density,
                    init_v,
                    fix_v,
                )
            elif self.sims.material_type == "TwoPhaseSingleLayer":
                kernel_add_body_twophase(
                    particles,
                    particleNum,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    bodyID,
                    materialID,
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
                    particleNum,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    bodyID,
                    materialID,
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
                kernel_add_body_2D(
                    particles,
                    particleNum,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    bodyID,
                    materialID,
                    density,
                    init_v,
                    fix_v,
                )
            elif self.sims.material_type == "TwoPhaseSingleLayer" and rigid_body:
                kernel_add_body_2D_rigid(
                    particles,
                    particleNum,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    bodyID,
                    materialID,
                    density,
                    init_v,
                    fix_v,
                    self.sims.axis_offset,
                )
            elif self.sims.material_type == "TwoPhaseSingleLayer":
                kernel_add_body_twophase2D(
                    particles,
                    particleNum,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    bodyID,
                    materialID,
                    density,
                    densityf,
                    porosity,
                    permeability,
                    init_v,
                    fix_v,
                    self.sims.axis_offset,
                )
            elif self.sims.material_type == "TwoPhaseDoubleLayer":
                kernel_add_body_twophase_double_point2D(
                    particles,
                    particleNum,
                    0,
                    self.insert_particle_num[None],
                    self.particle,
                    particle_volume,
                    bodyID,
                    materialID,
                    phase,
                    density,
                    densityf,
                    porosity,
                    permeability,
                    init_v,
                    fix_v,
                    self.sims.axis_offset,
                )

        self.set_particle_stress(scene, particleNum, self.insert_particle_num[None], particle_stress)
        scene.push_psize(np.repeat([psize], self.insert_particle_num[None], axis=0))
        traction = DictIO.GetAlternative(template, "Traction", {})
        self.set_traction(self.insert_particle_num[None], traction, scene, region)
        scene.material.state_vars_initialize(
            materialID, particleNum, particleNum + self.insert_particle_num[None], scene.particle
        )
        self.print_particle_info(
            bodyID, materialID, init_v, fix_v_str, self.insert_particle_num[None], particle_volume, nParticlesPerCell
        )
        scene.particleNum[0] += self.insert_particle_num[None]

    def Generate(self, scene: myScene, region: RegionFunction, nParticlesPerCell):
        if scene.is_rectangle_cell():
            if self.sims.dimension == 3:
                kernel_place_particles_(
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
                kernel_place_particles_2D(
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
            kernel_activate_cell_(
                region.start_point,
                region.region_size,
                scene.element.mesh.nodal_coords,
                scene.element.node_connectivity,
                scene.element.cell_active,
                region.function,
            )
            kernel_fill_particle_in_cell_(
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
            raise RuntimeError("Wrong element type!")

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


# Soft-particle level-set body generation
import numpy as np

from src.dem.generator.BrustNeighbor import BruteSearch
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.dem.generator.LinkedCellNeighbor import LinkedCell
from src.mpm.generator.InsertionKernel import (
    kernel_initialize_level_set_soft_body_,
    kernel_initialize_level_set_soft_body_grids_,
    kernel_initialize_level_set_soft_body_points_,
    kernel_initialize_level_set_soft_body_surface_,
    kernel_initialize_packed_level_set_soft_body_,
    kernel_initialize_packed_level_set_soft_body_grids_,
    kernel_initialize_packed_level_set_soft_body_points_,
    kernel_initialize_packed_level_set_soft_body_surface_,
    kernel_prepare_soft_grid_topology_,
)
from src.mpm.soft_particle.GridTopology import (
    select_soft_grid_topology,
    verlet_padding_cell_count,
)
from src.utils.ObjectIO import DictIO
from src.utils.Orientation import set_orientation
from src.utils.FieldIO import runtime_float_numpy_dtype
from src.utils.TypeDefination import vec3f, vec3i
from src.utils.sorting.ParallelSort import parallel_sort_with_two_values


def sample_soft_material_points(
    template_ptr: GeneralShapeTemplate,
    points_per_cell,
    center_quadrature=False,
    material_points=None,
    point_volume=None,
    mechanical_grid=None,
):
    if material_points is not None:
        points = np.asarray(material_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
            raise ValueError("MaterialPointCoordinates must have shape (point_count, 3)")
        if not np.isfinite(points).all():
            raise ValueError("MaterialPointCoordinates must be finite")
        if point_volume is None:
            point_volume = float(template_ptr.objects.volume) / points.shape[0]
        point_volume = np.asarray(point_volume, dtype=np.float64)
        if point_volume.ndim == 0:
            point_volume = np.full(points.shape[0], float(point_volume), dtype=np.float64)
        if point_volume.shape != (points.shape[0],):
            raise ValueError("MaterialPointVolume must be a scalar or have one value per " "material point")
        if not np.isfinite(point_volume).all() or np.any(point_volume <= 0.0):
            raise ValueError("MaterialPointVolume must be positive and finite")
        return (
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray(point_volume, dtype=np.float64),
        )

    if mechanical_grid is None:
        from src.mpm.soft_particle.GridTopology import (
            build_soft_mechanical_grid,
        )

        mechanical_grid = build_soft_mechanical_grid(
            template_ptr,
            0.15 * float(template_ptr.objects.eqradius),
            shape_function_type=0,
        )
    ppc = max(int(points_per_cell), 1)
    spacing = mechanical_grid.spacing / ppc
    lower = np.asarray(mechanical_grid.minBox(), dtype=np.float64)
    upper = np.asarray(mechanical_grid.maxBox(), dtype=np.float64)
    axes = [np.arange(lower[d] + 0.5 * spacing, upper[d], spacing) for d in range(3)]
    if axes[0].size == 0 or axes[1].size == 0 or axes[2].size == 0:
        points = np.asarray(template_ptr.objects.center, dtype=np.float64).reshape(1, 3)
        return (
            np.ascontiguousarray(points),
            np.asarray([float(template_ptr.objects.volume)], dtype=np.float64),
        )

    xv, yv, zv = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    points = np.column_stack((xv.reshape(-1), yv.reshape(-1), zv.reshape(-1)))
    # Trimesh proximity queries can allocate candidate-triangle workspaces
    # proportional to the number of query points.  A fine three-dimensional
    # LSMPM template may have close to a million voxel-center candidates, so
    # one monolithic signed-distance call can exceed a worker's host-memory
    # limit before any Taichi field is allocated.  Chunking is mathematically
    # identical because the signed distance is evaluated independently at
    # every point.
    distance_batch_size = max(
        int(os.environ.get("GT_SOFT_TEMPLATE_DISTANCE_BATCH_SIZE", "16384")),
        1,
    )
    inside = np.empty(points.shape[0], dtype=bool)
    for start in range(0, points.shape[0], distance_batch_size):
        stop = min(start + distance_batch_size, points.shape[0])
        inside[start:stop] = np.asarray(template_ptr.objects(points[start:stop])).reshape(-1) <= 0.0
    points = points[inside]
    if points.shape[0] == 0:
        points = np.asarray(template_ptr.objects.center, dtype=np.float64).reshape(1, 3)
    if center_quadrature:
        target_center = np.asarray(template_ptr.objects.center, dtype=np.float64)
        points += target_center - np.mean(points, axis=0)
    # Preserve the template's integrated volume independently of voxel-center
    # sampling resolution; every body then carries the requested bulk density.
    point_volume = np.full(
        points.shape[0],
        float(template_ptr.objects.volume) / points.shape[0],
        dtype=np.float64,
    )
    return (
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(point_volume),
    )


def preprocess_soft_grid_template(
    template_ptr,
    points_per_cell,
    center_quadrature,
    shape_function_type,
    storage,
    verlet_distance_multiplier,
    material_points=None,
    point_volume=None,
    grid_type="Hexahedron",
    mechanical_grid_spacing=None,
    mechanical_grid_refinement=None,
    reference_volume=None,
):
    from src.mpm.soft_particle.GridTopology import (
        build_soft_mechanical_grid,
        normalize_soft_grid_refinement,
    )
    from src.mpm.soft_particle.TemplateSupport import (
        HEXAHEDRON,
        TETRAHEDRON,
        build_hexahedral_template_support,
        build_tetrahedral_template_support,
        normalize_soft_grid_type,
    )

    normalized_grid_type, grid_type_id = normalize_soft_grid_type(grid_type)
    mechanical_shape_function_type = 0 if grid_type_id == TETRAHEDRON else int(shape_function_type)
    if mechanical_grid_spacing is None:
        mechanical_grid_spacing = 0.15 * float(template_ptr.objects.eqradius)
    mechanical_grid_spacing = float(mechanical_grid_spacing)
    if not np.isfinite(mechanical_grid_spacing) or mechanical_grid_spacing <= 0.0:
        raise ValueError("MechanicalGridSpacing must be positive and finite")
    if grid_type_id == TETRAHEDRON and material_points is not None:
        raise RuntimeError(
            "Tetrahedral LSMPM places one material point at each linear "
            "tetrahedron Gauss point; MaterialPointCoordinates is only "
            "supported by the hexahedral grid"
        )
    normalized_refinement = normalize_soft_grid_refinement(mechanical_grid_refinement, mechanical_grid_spacing)
    if grid_type_id == HEXAHEDRON and normalized_refinement is not None:
        raise RuntimeError("MechanicalGridRefinement is only supported by the linear " "tetrahedral soft grid")
    cache = getattr(template_ptr, "soft_grid_preprocess_cache", None)
    if cache is None:
        cache = {}
        template_ptr.soft_grid_preprocess_cache = cache
    custom_digest = None
    if material_points is not None:
        custom_points = np.ascontiguousarray(material_points, dtype=np.float64)
        custom_digest = hashlib.sha256(custom_points.view(np.uint8)).hexdigest()
    point_volume_digest = None
    if point_volume is not None:
        point_volume_values = np.ascontiguousarray(np.asarray(point_volume, dtype=np.float64))
        point_volume_digest = hashlib.sha256(point_volume_values.view(np.uint8)).hexdigest()
    refinement_key = None
    if normalized_refinement is not None:
        refinement_key = (
            tuple(normalized_refinement["region_min"]),
            tuple(normalized_refinement["region_max"]),
            float(normalized_refinement["fine_spacing"]),
            int(normalized_refinement["subdivision"]),
        )
    if reference_volume is not None:
        reference_volume = float(reference_volume)
        if not np.isfinite(reference_volume) or reference_volume <= 0.0:
            raise ValueError("ReferenceVolume must be positive and finite")
    quadrature_per_cell = 1 if grid_type_id == TETRAHEDRON else max(int(points_per_cell), 1)
    cache_key = (
        quadrature_per_cell,
        bool(center_quadrature),
        mechanical_shape_function_type,
        str(storage).strip().lower(),
        float(verlet_distance_multiplier),
        int(template_ptr.objects.grid.extent),
        float(template_ptr.objects.grid.grid_space),
        tuple(np.asarray(template_ptr.objects.grid.gnum, dtype=np.int64)),
        mechanical_grid_spacing,
        custom_digest,
        point_volume_digest,
        normalized_grid_type,
        refinement_key,
        reference_volume,
    )
    if cache_key in cache:
        return cache[cache_key]
    verlet_distance = max(float(verlet_distance_multiplier), 0.0) * float(template_ptr.boundings.r_bound)
    levelset_verlet_padding_cells = verlet_padding_cell_count(
        verlet_distance,
        float(template_ptr.objects.grid.grid_space),
    )
    levelset_extent_cells = int(template_ptr.objects.grid.extent)
    if levelset_verlet_padding_cells > levelset_extent_cells:
        raise RuntimeError(
            "Level-set template extent is smaller than the LSDEM Verlet "
            f"requirement: extent={levelset_extent_cells}, "
            f"required={levelset_verlet_padding_cells}"
        )
    minimum_mechanical_spacing = (
        normalized_refinement["fine_spacing"] if normalized_refinement is not None else mechanical_grid_spacing
    )
    mechanical_verlet_padding_cells = verlet_padding_cell_count(
        verlet_distance,
        minimum_mechanical_spacing,
    )
    mechanical_grid_halo_cells = verlet_padding_cell_count(
        verlet_distance,
        mechanical_grid_spacing,
    )
    mechanical_grid = build_soft_mechanical_grid(
        template_ptr,
        mechanical_grid_spacing,
        mechanical_shape_function_type,
        verlet_padding_cells=mechanical_grid_halo_cells,
        refinement=normalized_refinement,
    )
    padding_cells = mechanical_verlet_padding_cells
    if grid_type_id == HEXAHEDRON:
        material_points, point_volume = sample_soft_material_points(
            template_ptr,
            quadrature_per_cell,
            center_quadrature,
            material_points=material_points,
            point_volume=point_volume,
            mechanical_grid=mechanical_grid,
        )
        topology = select_soft_grid_topology(
            template_ptr,
            material_points,
            mechanical_shape_function_type,
            storage,
            padding_cells,
            mechanical_grid=mechanical_grid,
            levelset_extent_cells=levelset_extent_cells,
            verlet_padding_cells=mechanical_verlet_padding_cells,
            levelset_verlet_padding_cells=levelset_verlet_padding_cells,
        )
        support = build_hexahedral_template_support(
            template_ptr,
            material_points,
            topology,
            mechanical_shape_function_type,
            mechanical_grid,
        )
    else:
        (
            material_points,
            point_volume,
            topology,
            support,
        ) = build_tetrahedral_template_support(
            template_ptr,
            mechanical_grid,
            storage,
            padding_cells,
            levelset_extent_cells=levelset_extent_cells,
            verlet_padding_cells=mechanical_verlet_padding_cells,
            levelset_verlet_padding_cells=levelset_verlet_padding_cells,
            reference_volume=reference_volume,
        )
    result = (material_points, point_volume, topology, support)
    cache[cache_key] = result
    return result


def soft_template_kernel_arrays(template_ptr, material_points, point_volumes):
    dtype = runtime_float_numpy_dtype()
    return (
        np.ascontiguousarray(template_ptr.objects.mesh.vertices, dtype=dtype),
        np.ascontiguousarray(template_ptr.parameter, dtype=dtype),
        np.ascontiguousarray(template_ptr.objects.grid.distance_field, dtype=dtype),
        np.ascontiguousarray(material_points, dtype=dtype),
        np.ascontiguousarray(point_volumes, dtype=dtype),
    )


def soft_template_shape_bounds(surface_nodes, material_points, point_volumes, scale_factor):
    """Return conservative template-space bounds for one soft body."""
    dtype = np.result_type(
        np.asarray(surface_nodes).dtype,
        np.asarray(material_points).dtype,
        np.asarray(point_volumes).dtype,
    )
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        dtype = np.dtype(np.float64)
    surface = np.asarray(surface_nodes, dtype=dtype)
    points = np.asarray(material_points, dtype=dtype)
    volumes = np.asarray(point_volumes, dtype=dtype)
    scale = float(scale_factor)
    if surface.ndim != 2 or surface.shape[1] != 3 or surface.shape[0] == 0:
        raise ValueError("soft template requires at least one surface node")
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("soft template requires at least one material point")
    if volumes.shape != (points.shape[0],) or np.any(volumes <= 0.0):
        raise ValueError("soft material-point volumes must be positive")

    scaled_surface = scale * surface
    scaled_points = scale * points
    point_padding = 0.5 * np.cbrt(volumes * scale**3)
    shape_min = np.minimum(
        np.min(scaled_surface, axis=0),
        np.min(scaled_points - point_padding[:, None], axis=0),
    )
    shape_max = np.maximum(
        np.max(scaled_surface, axis=0),
        np.max(scaled_points + point_padding[:, None], axis=0),
    )
    shape_radius = max(
        float(np.max(np.linalg.norm(scaled_surface, axis=1))),
        float(np.max(np.linalg.norm(scaled_points, axis=1) + np.sqrt(3.0) * point_padding)),
    )
    return (
        np.ascontiguousarray(shape_min, dtype=dtype),
        np.ascontiguousarray(shape_max, dtype=dtype),
        shape_radius,
    )


class SoftParticleCreatorMixin(object):
    def create_soft_body(self, sims, scene, template):
        if type(template) is dict:
            self.create_template_soft_body(sims, scene, template)
        elif type(template) is list:
            for temp in template:
                self.create_template_soft_body(sims, scene, temp)

    def sample_soft_material_points(
        self,
        template_ptr: GeneralShapeTemplate,
        points_per_cell,
        center_quadrature=False,
        material_points=None,
        point_volume=None,
    ):
        return sample_soft_material_points(
            template_ptr,
            points_per_cell,
            center_quadrature,
            material_points=material_points,
            point_volume=point_volume,
        )

    def create_template_soft_body(self, sims, scene, template):
        if sims.scheme != "LSMPM":
            raise RuntimeError("SoftBody with level-set MPM is only supported when scheme is /LSMPM/")
        bounding_sphere = scene.get_bounding_sphere()
        bounding_box = scene.get_bounding_box()
        master = scene.get_surface()
        rigid_body = scene.get_rigid_ptr()
        soft_body = scene.get_soft_ptr()
        soft_point = scene.get_soft_point_ptr()
        material = scene.get_material_ptr()
        particleNum = int(scene.particleNum[0])
        softNum = int(scene.softNum[0])
        pointStart = int(scene.softPointNum[0])
        mpmGridStart = int(scene.softGridNum[0])
        surfaceNum = int(scene.surfaceNum[0])

        name = DictIO.GetEssential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)
        com_pos = DictIO.GetEssential(template, "BodyPoint")
        equiv_rad = DictIO.GetAlternative(template, "Radius", None)
        bounding_rad = DictIO.GetAlternative(template, "BoundingRadius", None)
        scale_factor = DictIO.GetAlternative(template, "ScaleFactor", None)
        orientation = DictIO.GetAlternative(template, "BodyOrientation", None)
        points_per_cell = DictIO.GetAlternative(
            template, "MaterialPointsPerCell", DictIO.GetAlternative(template, "ParticlePerCell", 1)
        )
        center_quadrature = DictIO.GetAlternative(template, "CenterMaterialPointQuadrature", False)
        custom_material_points = DictIO.GetAlternative(template, "MaterialPointCoordinates", None)
        custom_point_volume = DictIO.GetAlternative(template, "MaterialPointVolume", None)
        mechanical_grid_spacing = DictIO.GetAlternative(
            template,
            "MechanicalGridSpacing",
            getattr(
                template_ptr,
                "soft_mechanical_grid_spacing",
                sims.soft_mechanical_grid_spacing_ratio * float(template_ptr.objects.eqradius),
            ),
        )
        mechanical_grid_refinement = DictIO.GetAlternative(
            template,
            "MechanicalGridRefinement",
            getattr(template_ptr, "soft_mechanical_grid_refinement", None),
        )
        reference_volume = DictIO.GetAlternative(
            template,
            "ReferenceVolume",
            getattr(template_ptr, "soft_reference_volume", None),
        )
        mechanical_grid_boundary = DictIO.GetAlternative(template, "MechanicalGridBoundary", None)
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
            raise RuntimeError(
                "Keyword conflict!, You should set either Keyword:: /Radius/ or Keyword:: /ScaleFactor/."
            )

        material_points, point_volume, topology, support = preprocess_soft_grid_template(
            template_ptr,
            points_per_cell,
            center_quadrature,
            sims.soft_shape_function_type,
            sims.soft_grid_storage,
            sims.verlet_distance_multiplier[1],
            custom_material_points,
            custom_point_volume,
            sims.soft_grid_type,
            mechanical_grid_spacing,
            mechanical_grid_refinement,
            reference_volume,
        )
        (
            surface_nodes,
            surface_parameters,
            distance_fields,
            kernel_points,
            kernel_point_volumes,
        ) = soft_template_kernel_arrays(template_ptr, material_points, point_volume)
        pointCount = int(material_points.shape[0])
        (
            templatePointStart,
            templateSurfaceStart,
            templateSdfStart,
            gridType,
        ) = register_soft_template_support(scene, support)
        gridStart = int(scene.gridNum[0])
        gridSum = int(template_ptr.objects.grid.gridSum)
        compactGridSum = topology.compact_count
        verticeNum = int(scene.verticeID[-1] + scene.surfaceNum[0])
        shape_min, shape_max, shape_radius = soft_template_shape_bounds(
            surface_nodes,
            kernel_points,
            kernel_point_volumes,
            scale_factor,
        )

        scene.check_particle_num(sims, particle_number=1)
        scene.check_soft_body_number(sims, soft_body_number=1)
        scene.check_material_point_number(sims, pointCount)
        scene.check_surface_node_number(sims, template_ptr.surface_node_number)
        if gridStart + gridSum > scene.rigid_grid.shape[0]:
            raise ValueError("The level-set grid storage should be enlarged to: ", gridStart + gridSum)
        if mpmGridStart + compactGridSum > scene.soft_grid_capacity:
            raise ValueError(
                "The compact soft MPM grid storage should be enlarged to: ",
                mpmGridStart + compactGridSum,
            )

        kernel_prepare_soft_grid_topology_(
            soft_body,
            softNum,
            1,
            gridStart,
            mpmGridStart,
            gridSum,
            int(topology.compact_origin[0]),
            int(topology.compact_origin[1]),
            int(topology.compact_origin[2]),
            int(topology.compact_shape[0]),
            int(topology.compact_shape[1]),
            int(topology.compact_shape[2]),
        )

        kernel_initialize_level_set_soft_body_(
            soft_body,
            rigid_body,
            bounding_box,
            bounding_sphere,
            material,
            particleNum,
            softNum,
            pointStart,
            pointCount,
            gridStart,
            mpmGridStart,
            verticeNum,
            surfaceNum,
            vec3f(template_ptr.objects.grid.minBox()),
            vec3f(template_ptr.objects.grid.maxBox()),
            template_ptr.surface_node_number,
            template_ptr.surface_area,
            gridSum,
            template_ptr.objects.grid.grid_space,
            vec3i(template_ptr.objects.grid.gnum),
            template_ptr.objects.grid.extent,
            vec3f(shape_min),
            vec3f(shape_max),
            shape_radius,
            float(np.sum(point_volume)),
            scale_factor,
            vec3f(template_ptr.objects.inertia),
            com_pos,
            equiv_rad,
            set_orientations.get_orientation,
            groupID,
            matID,
            init_v,
            init_w,
            is_fix,
            templatePointStart,
            templateSurfaceStart,
            templateSdfStart,
            gridType,
            support.grid_space,
        )
        kernel_initialize_level_set_soft_body_grids_(
            soft_body,
            scene.rigid_grid,
            scene.soft_grid,
            scene.soft_grid_owner,
            scene.soft_grid_local,
            softNum,
            gridStart,
            mpmGridStart,
            gridSum,
            distance_fields,
        )
        kernel_initialize_level_set_soft_body_surface_(
            rigid_body,
            master,
            scene.vertice,
            particleNum,
            verticeNum,
            surfaceNum,
            template_ptr.surface_node_number,
            scale_factor,
            surface_nodes,
            surface_parameters,
            init_v,
            init_w,
        )
        kernel_initialize_level_set_soft_body_points_(
            soft_body,
            soft_point,
            rigid_body,
            bounding_box,
            material,
            scene.rigid_grid,
            scene.soft_surface_point_id,
            particleNum,
            softNum,
            pointStart,
            pointCount,
            scale_factor,
            kernel_points,
            kernel_point_volumes,
            groupID,
            matID,
            init_v,
            init_w,
        )

        if mechanical_grid_boundary is not None:
            boundaries = (
                mechanical_grid_boundary if isinstance(mechanical_grid_boundary, list) else [mechanical_grid_boundary]
            )
            fixed_local_nodes = np.unique(
                np.concatenate(
                    [compact_soft_grid_nodes_in_region(boundary, support, topology) for boundary in boundaries]
                )
            ).astype(np.int32, copy=False)
            constraint_start = int(scene.softVelocityConstraintNum[0])
            constraint_end = constraint_start + fixed_local_nodes.size
            capacity = int(scene.soft_velocity_constraint.shape[0])
            if constraint_end > sims.max_soft_velocity_constraint_num:
                raise ValueError(
                    "The soft velocity-constraint storage should be enlarged "
                    f"to {constraint_end} (allocated {capacity})"
                )
            upload_soft_velocity_constraints_(
                constraint_start,
                mpmGridStart,
                np.ascontiguousarray(fixed_local_nodes),
                scene.soft_velocity_constraint,
            )
            scene.softVelocityConstraintNum[0] = constraint_end

        print(" Level-set MPM soft body Information ".center(71, "-"))
        self.print_particle_info(groupID, matID, com_pos, init_v, init_w, fix_v=is_fix, fix_w=is_fix, name=name)
        scene.add_connectivity(1, template_ptr.surface_node_number, template_ptr.objects)
        scene.particleNum[0] += 1
        scene.rigidNum[0] += 1
        scene.softNum[0] += 1
        scene.softPointNum[0] += pointCount
        scene.softGridNum[0] += compactGridSum
        scene.softMechanicalLogicalGridNum[0] += topology.logical_count
        scene.softLogicalGridNum[0] += gridSum
        scene.softMaxLogicalGridNum[0] = max(int(scene.softMaxLogicalGridNum[0]), gridSum)
        scene.surfaceNum[0] += template_ptr.surface_node_number
        scene.gridNum[0] += gridSum
        scene.gridID.append(int(scene.gridNum[0]))

        from src.mpm.soft_particle.ReferenceMap import initialize_soft_reference_sdf_

        initialize_soft_reference_sdf_(
            int(scene.softNum[0]),
            scene.soft,
            scene.rigid_grid,
            scene.soft_levelset_initial_sdf,
            scene.soft_levelset_initial_sdf_initialized,
        )


class SoftParticleGeneratorMixin(object):
    def sample_soft_material_points(
        self,
        template_ptr: GeneralShapeTemplate,
        points_per_cell,
        center_quadrature=False,
        material_points=None,
        point_volume=None,
    ):
        return sample_soft_material_points(
            template_ptr,
            points_per_cell,
            center_quadrature,
            material_points=material_points,
            point_volume=point_volume,
        )

    def generate_soft_bodys(self, scene):
        if self.sims.scheme != "LSMPM":
            raise RuntimeError("SoftBody generation is only supported when scheme is /LSMPM/")

        bounding_sphere = scene.get_bounding_sphere()
        particleNum = int(scene.particleNum[0])

        insert_num = 0
        if type(self.template_dict) is dict:
            insert_num += DictIO.GetEssential(self.template_dict, "BodyNumber")
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                insert_num += DictIO.GetEssential(temp, "BodyNumber")
        self.hist_check_levelset_number(bounding_sphere, particleNum, insert_num)
        particle_in_region = insert_num + self.region.inserted_particle_num

        if type(self.template_dict) is dict:
            body_number = DictIO.GetEssential(self.template_dict, "BodyNumber")
            self.radius_dist.append(self.get_bounding_radius(self.template_dict, body_number))
            self.template_num = 1
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                body_number = DictIO.GetEssential(temp, "BodyNumber")
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
                particleNum,
                self.insert_particle_in_neighbor,
                bounding_sphere,
                self.region.function,
                self.region.start_point,
            )

        if type(self.template_dict) is dict:
            self.generate_template_soft_body(scene, self.template_dict, self.radius_dist[0])
        elif type(self.template_dict) is list:
            for temp, radius_dist in zip(self.template_dict, self.radius_dist):
                self.generate_template_soft_body(scene, temp, radius_dist)

    def generate_template_soft_body(self, scene, template, radius_dist):
        actual_body = DictIO.GetEssential(template, "BodyNumber")
        orientation = DictIO.GetAlternative(template, "BodyOrientation", None)
        set_orientations = set_orientation(orientation)

        name = DictIO.GetEssential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)

        start_body_num = self.insert_body_num[None]
        self.GenerateLevelSet(actual_body, start_body_num, radius_dist, template_ptr, set_orientations)
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num

        if self.write_file:
            self.write_body_text(start_body_num, end_body_num)
        else:
            self.insert_soft_levelset(scene, template, start_body_num, end_body_num, body_count)

    def lattice_soft_bodys(self, scene):
        if self.sims.scheme != "LSMPM":
            raise RuntimeError("SoftBody lattice generation is only supported when scheme is /LSMPM/")

        bounding_sphere = scene.get_bounding_sphere()
        particleNum = int(scene.particleNum[0])

        total_fraction = 0.0
        max_rad = 0.0
        if type(self.template_dict) is dict:
            fraction = DictIO.GetAlternative(self.template_dict, "Fraction", 1.0)
            _, template_max = self.get_bounding_radius_range(self.template_dict)
            total_fraction += fraction
            max_rad = max(max_rad, template_max)
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                fraction = DictIO.GetAlternative(temp, "Fraction", 1.0)
                _, template_max = self.get_bounding_radius_range(temp)
                total_fraction += fraction
                max_rad = max(max_rad, template_max)
        if total_fraction < 0.0 or total_fraction > 1.0:
            raise ValueError("Fraction value error")
        if max_rad <= 0.0:
            raise RuntimeError("The maximum bounding radius should be positive")

        insert_particle = np.floor(0.5 * np.array(self.region.region_size) / max_rad).astype(np.int32)
        if np.any(insert_particle <= 0):
            raise RuntimeError("The lattice spacing is too large for the selected region and soft body radius")
        insertNum = int(insert_particle[0] * insert_particle[1] * insert_particle[2])

        if type(self.template_dict) is dict:
            self.radius_dist.append(self.get_bounding_radius(self.template_dict, insertNum))
            self.template_num = 1
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.radius_dist.append(self.get_bounding_radius(temp, insertNum))
                self.template_num += 1

        min_radius, max_radius = [], []
        for i in range(self.template_num):
            min_radius.append(np.min(self.radius_dist[i]))
            max_radius.append(np.max(self.radius_dist[i]))
        min_rad, max_rad = min(min_radius), max(max_radius)

        self.hist_check_levelset_number(bounding_sphere, particleNum, insertNum)
        particle_in_region = insertNum + self.region.inserted_particle_num

        if particle_in_region < 1000:
            if self.neighbor is None:
                self.neighbor = BruteSearch(rigid=True)
            self.neighbor.neighbor_init(particle_in_region)
        elif particle_in_region >= 1000:
            if self.neighbor is None:
                self.neighbor = LinkedCell(rigid=True)
            self.neighbor.neighbor_init(min_rad, max_rad, self.region.region_size, particle_in_region)
        self.allocate_sphere_memory(insertNum, generate=True, levelset=True, lattice=True)

        if self.check_hist and self.region.inserted_particle_num > 0:
            self.neighbor.pre_neighbor_bounding_sphere(
                particleNum,
                self.insert_particle_in_neighbor,
                bounding_sphere,
                self.region.function,
                self.region.start_point,
            )

        if type(self.template_dict) is dict:
            self.lattice_template_soft_body(scene, self.template_dict, insert_particle, self.radius_dist[0])
        elif type(self.template_dict) is list:
            for temp, radius_dist in zip(self.template_dict, self.radius_dist):
                self.lattice_template_soft_body(scene, temp, insert_particle, radius_dist)

    def lattice_template_soft_body(self, scene, template, insert_particle, radius_dist):
        fraction = DictIO.GetAlternative(template, "Fraction", 1.0)
        orientation = DictIO.GetAlternative(template, "BodyOrientation", None)
        set_orientations = set_orientation(orientation)
        name = DictIO.GetEssential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)

        actual_body = int(fraction * int(insert_particle[0] * insert_particle[1] * insert_particle[2]))
        start_body_num = self.insert_body_num[None]
        self.LatticeLevelSet(actual_body, start_body_num, insert_particle, radius_dist, template_ptr, set_orientations)
        end_body_num = self.insert_body_num[None]
        body_count = end_body_num - start_body_num
        self.region.inserted_body_num = end_body_num
        self.region.inserted_particle_num = end_body_num
        parallel_sort_with_two_values(self.sphere_radii, self.sphere_coords, self.orients, start_body_num, body_count)

        if self.write_file:
            self.write_body_text(start_body_num, end_body_num)
        else:
            self.insert_soft_levelset(scene, template, start_body_num, end_body_num, body_count)

    def add_soft_levelsets_to_scene(self, scene):
        if type(self.template_dict) is dict:
            self.insert_soft_levelset(
                scene, self.template_dict, 0, self.insert_body_num[None], self.insert_body_num[None]
            )
        elif type(self.template_dict) is list:
            for temp in self.template_dict:
                self.insert_soft_levelset(scene, temp, 0, self.insert_body_num[None], self.insert_body_num[None])

    def insert_soft_levelset(self, scene, template, start_body_num, end_body_num, body_count):
        if self.sims.scheme != "LSMPM":
            raise RuntimeError("SoftBody packing insertion is only supported when scheme is /LSMPM/")
        bounding_sphere = scene.get_bounding_sphere()
        bounding_box = scene.get_bounding_box()
        master = scene.get_surface()
        rigid_body = scene.get_rigid_ptr()
        soft_body = scene.get_soft_ptr()
        soft_point = scene.get_soft_point_ptr()
        material = scene.get_material_ptr()
        particleNum = int(scene.particleNum[0])
        softNum = int(scene.softNum[0])
        pointStart = int(scene.softPointNum[0])
        gridStart = int(scene.gridNum[0])
        mpmGridStart = int(scene.softGridNum[0])
        surfaceNum = int(scene.surfaceNum[0])

        groupID = DictIO.GetEssential(template, "GroupID")
        matID = DictIO.GetEssential(template, "MaterialID")
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.GetAlternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.GetAlternative(template, "FixMotion", ["Free", "Free", "Free"])
        is_fix = vec3i([DictIO.GetEssential(self.FIX, i) for i in fix_str])
        points_per_cell = DictIO.GetAlternative(
            template, "MaterialPointsPerCell", DictIO.GetAlternative(template, "ParticlePerCell", 1)
        )
        center_quadrature = DictIO.GetAlternative(template, "CenterMaterialPointQuadrature", False)
        custom_material_points = DictIO.GetAlternative(template, "MaterialPointCoordinates", None)
        custom_point_volume = DictIO.GetAlternative(template, "MaterialPointVolume", None)

        name = DictIO.GetEssential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)
        mechanical_grid_spacing = DictIO.GetAlternative(
            template,
            "MechanicalGridSpacing",
            getattr(
                template_ptr,
                "soft_mechanical_grid_spacing",
                self.sims.soft_mechanical_grid_spacing_ratio * float(template_ptr.objects.eqradius),
            ),
        )
        mechanical_grid_refinement = DictIO.GetAlternative(
            template,
            "MechanicalGridRefinement",
            getattr(template_ptr, "soft_mechanical_grid_refinement", None),
        )
        reference_volume = DictIO.GetAlternative(
            template,
            "ReferenceVolume",
            getattr(template_ptr, "soft_reference_volume", None),
        )
        material_points, point_volume, topology, support = preprocess_soft_grid_template(
            template_ptr,
            points_per_cell,
            center_quadrature,
            self.sims.soft_shape_function_type,
            self.sims.soft_grid_storage,
            self.sims.verlet_distance_multiplier[1],
            custom_material_points,
            custom_point_volume,
            self.sims.soft_grid_type,
            mechanical_grid_spacing,
            mechanical_grid_refinement,
            reference_volume,
        )
        (
            surface_nodes,
            surface_parameters,
            distance_fields,
            kernel_points,
            kernel_point_volumes,
        ) = soft_template_kernel_arrays(template_ptr, material_points, point_volume)
        pointCount = int(material_points.shape[0])
        (
            templatePointStart,
            templateSurfaceStart,
            templateSdfStart,
            gridType,
        ) = register_soft_template_support(scene, support)
        gridSum = int(template_ptr.objects.grid.gridSum)
        compactGridSum = topology.compact_count
        surfaceSum = int(template_ptr.surface_node_number)
        verticeNum = int(scene.verticeID[-1] + scene.surfaceNum[0])

        scene.check_particle_num(self.sims, particle_number=body_count)
        scene.check_soft_body_number(self.sims, soft_body_number=body_count)
        scene.check_material_point_number(self.sims, pointCount * body_count)
        scene.check_surface_node_number(self.sims, surfaceSum)
        if gridStart + gridSum * body_count > scene.rigid_grid.shape[0]:
            raise ValueError("The level-set grid storage should be enlarged to: ", gridStart + gridSum * body_count)
        if mpmGridStart + compactGridSum * body_count > scene.soft_grid_capacity:
            raise ValueError(
                "The compact soft MPM grid storage should be enlarged to: ",
                mpmGridStart + compactGridSum * body_count,
            )

        kernel_prepare_soft_grid_topology_(
            soft_body,
            softNum,
            body_count,
            gridStart,
            mpmGridStart,
            gridSum,
            int(topology.compact_origin[0]),
            int(topology.compact_origin[1]),
            int(topology.compact_origin[2]),
            int(topology.compact_shape[0]),
            int(topology.compact_shape[1]),
            int(topology.compact_shape[2]),
        )

        base_shape_min, base_shape_max, base_shape_radius = soft_template_shape_bounds(
            surface_nodes,
            kernel_points,
            kernel_point_volumes,
            1.0,
        )
        body_count = end_body_num - start_body_num
        kernel_initialize_packed_level_set_soft_body_(
            soft_body,
            rigid_body,
            bounding_box,
            bounding_sphere,
            material,
            particleNum,
            softNum,
            pointStart,
            pointCount,
            gridStart,
            mpmGridStart,
            verticeNum,
            surfaceNum,
            vec3f(template_ptr.objects.grid.minBox()),
            vec3f(template_ptr.objects.grid.maxBox()),
            template_ptr.boundings.r_bound,
            vec3f(template_ptr.boundings.x_bound),
            surfaceSum,
            template_ptr.surface_area,
            gridSum,
            template_ptr.objects.grid.grid_space,
            vec3i(template_ptr.objects.grid.gnum),
            template_ptr.objects.grid.extent,
            vec3f(base_shape_min),
            vec3f(base_shape_max),
            base_shape_radius,
            float(np.sum(point_volume)),
            vec3f(template_ptr.objects.inertia),
            template_ptr.objects.eqradius,
            groupID,
            matID,
            init_v,
            init_w,
            is_fix,
            start_body_num,
            end_body_num,
            self.sphere_coords,
            self.sphere_radii,
            self.orients,
            templatePointStart,
            templateSurfaceStart,
            templateSdfStart,
            gridType,
            support.grid_space,
            self.coordinates_are_mass_centers,
        )
        kernel_initialize_packed_level_set_soft_body_grids_(
            soft_body,
            scene.rigid_grid,
            scene.soft_grid,
            scene.soft_grid_owner,
            scene.soft_grid_local,
            softNum,
            body_count,
            gridStart,
            mpmGridStart,
            gridSum,
            compactGridSum,
            distance_fields,
        )
        kernel_initialize_packed_level_set_soft_body_surface_(
            soft_body,
            rigid_body,
            master,
            scene.vertice,
            particleNum,
            softNum,
            body_count,
            verticeNum,
            surfaceNum,
            surfaceSum,
            surface_nodes,
            surface_parameters,
            init_v,
            init_w,
        )
        kernel_initialize_packed_level_set_soft_body_points_(
            soft_body,
            soft_point,
            rigid_body,
            bounding_box,
            material,
            scene.rigid_grid,
            scene.soft_surface_point_id,
            particleNum,
            softNum,
            body_count,
            pointStart,
            pointCount,
            kernel_points,
            kernel_point_volumes,
            groupID,
            matID,
            init_v,
            init_w,
        )
        print(" Level-set MPM soft body Information ".center(71, "-"))
        self.print_particle_info(groupID, matID, init_v, init_w, fix_v=is_fix, fix_w=is_fix, body_num=body_count)

        faces = scene.add_connectivity(body_count, surfaceSum, template_ptr.objects)
        scene.particleNum[0] += body_count
        scene.rigidNum[0] += body_count
        scene.softNum[0] += body_count
        scene.softPointNum[0] += pointCount * body_count
        scene.softGridNum[0] += compactGridSum * body_count
        scene.softMechanicalLogicalGridNum[0] += topology.logical_count * body_count
        scene.softLogicalGridNum[0] += gridSum * body_count
        scene.softMaxLogicalGridNum[0] = max(int(scene.softMaxLogicalGridNum[0]), gridSum)
        scene.surfaceNum[0] += surfaceSum * body_count
        scene.gridNum[0] += gridSum * body_count
        scene.gridID.append(int(scene.gridNum[0]))
        from src.mpm.soft_particle.ReferenceMap import initialize_soft_reference_sdf_

        initialize_soft_reference_sdf_(
            int(scene.softNum[0]),
            scene.soft,
            scene.rigid_grid,
            scene.soft_levelset_initial_sdf,
            scene.soft_levelset_initial_sdf_initialized,
        )
        self.faces = np.append(self.faces, faces).reshape(-1, 3)
