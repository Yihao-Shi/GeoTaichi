from src.utils.errors import ConfigurationError, SimulationStateError
from src.mpm.generator.Utilities import preprocess_soft_grid_template
from src.mpm.generator.Utilities import sample_soft_material_points
from src.mpm.generator.Utilities import soft_template_kernel_arrays
from src.mpm.generator.Utilities import soft_template_shape_bounds

import numpy as np
from src.mpm.generator.InsertionKernel import *
from src.mpm.soft_particle.SceneFields import register_soft_template_support
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefinition import vec3f
from src.dem.generator.BruteNeighbor import BruteSearch
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.dem.generator.LinkedCellNeighbor import LinkedCell
from src.mpm.generator.InsertionKernel import (
    kernel_initialize_packed_level_set_soft_body,
    kernel_initialize_packed_level_set_soft_body_grids,
    kernel_initialize_packed_level_set_soft_body_points,
    kernel_initialize_packed_level_set_soft_body_surface,
    kernel_prepare_soft_grid_topology,
)
from src.utils.Orientation import Orientation
from src.utils.TypeDefinition import vec3f, vec3i
from src.utils.sorting.ParallelSort import parallel_sort_with_two_values


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
            raise ConfigurationError("SoftBody generation is only supported when scheme is /LSMPM/")

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
            self.generate_template_soft_body(scene, self.template_dict, self.radius_dist[0])
        elif type(self.template_dict) is list:
            for temp, radius_dist in zip(self.template_dict, self.radius_dist):
                self.generate_template_soft_body(scene, temp, radius_dist)

    def generate_template_soft_body(self, scene, template, radius_dist):
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

        if self.write_file:
            self.write_body_text(start_body_num, end_body_num)
        else:
            self.insert_soft_levelset(scene, template, start_body_num, end_body_num, body_count)

    def lattice_soft_bodys(self, scene):
        if self.sims.scheme != "LSMPM":
            raise ConfigurationError("SoftBody lattice generation is only supported when scheme is /LSMPM/")

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
            raise SimulationStateError("The lattice spacing is too large for the selected region and soft body radius")
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
            self.lattice_template_soft_body(scene, self.template_dict, insert_particle, self.radius_dist[0])
        elif type(self.template_dict) is list:
            for temp, radius_dist in zip(self.template_dict, self.radius_dist):
                self.lattice_template_soft_body(scene, temp, insert_particle, radius_dist)

    def lattice_template_soft_body(self, scene, template, insert_particle, radius_dist):
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
            raise ConfigurationError("SoftBody packing insertion is only supported when scheme is /LSMPM/")
        bounding_sphere = scene.get_bounding_sphere()
        bounding_box = scene.get_bounding_box()
        master = scene.get_surface()
        rigid_body = scene.get_rigid_ptr()
        soft_body = scene.get_soft_ptr()
        soft_point = scene.get_soft_point_ptr()
        material = scene.get_material_ptr()
        particle_num = int(scene.particleNum[0])
        soft_num = int(scene.softNum[0])
        point_start = int(scene.softPointNum[0])
        grid_start = int(scene.gridNum[0])
        mpm_grid_start = int(scene.softGridNum[0])
        surface_num = int(scene.surfaceNum[0])

        group_id = DictIO.get_essential(template, "GroupID")
        mat_id = DictIO.get_essential(template, "MaterialID")
        init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.get_alternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.get_alternative(template, "FixMotion", ["Free", "Free", "Free"])
        is_fix = vec3i([DictIO.get_essential(self.FIX, i) for i in fix_str])
        points_per_cell = DictIO.get_alternative(
            template, "MaterialPointsPerCell", DictIO.get_alternative(template, "ParticlePerCell", 1)
        )
        center_quadrature = DictIO.get_alternative(template, "CenterMaterialPointQuadrature", False)
        custom_material_points = DictIO.get_alternative(template, "MaterialPointCoordinates", None)
        custom_point_volume = DictIO.get_alternative(template, "MaterialPointVolume", None)

        name = DictIO.get_essential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)
        mechanical_grid_spacing = DictIO.get_alternative(
            template,
            "MechanicalGridSpacing",
            getattr(
                template_ptr,
                "soft_mechanical_grid_spacing",
                self.sims.soft_mechanical_grid_spacing_ratio * float(template_ptr.objects.eqradius),
            ),
        )
        mechanical_grid_refinement = DictIO.get_alternative(
            template,
            "MechanicalGridRefinement",
            getattr(template_ptr, "soft_mechanical_grid_refinement", None),
        )
        reference_volume = DictIO.get_alternative(
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
        point_count = int(material_points.shape[0])
        (
            template_point_start,
            template_surface_start,
            template_sdf_start,
            grid_type,
        ) = register_soft_template_support(scene, support)
        grid_sum = int(template_ptr.objects.grid.gridSum)
        compact_grid_sum = topology.compact_count
        surface_sum = int(template_ptr.surface_node_number)
        vertice_num = int(scene.verticeID[-1] + scene.surfaceNum[0])

        scene.check_particle_num(self.sims, particle_number=body_count)
        scene.check_soft_body_number(self.sims, soft_body_number=body_count)
        scene.check_material_point_number(self.sims, point_count * body_count)
        scene.check_surface_node_number(self.sims, surface_sum)
        if grid_start + grid_sum * body_count > scene.rigid_grid.shape[0]:
            raise ValueError("The level-set grid storage should be enlarged to: ", grid_start + grid_sum * body_count)
        if mpm_grid_start + compact_grid_sum * body_count > scene.soft_grid_capacity:
            raise ValueError(
                "The compact soft MPM grid storage should be enlarged to: ",
                mpm_grid_start + compact_grid_sum * body_count,
            )

        kernel_prepare_soft_grid_topology(
            soft_body,
            soft_num,
            body_count,
            grid_start,
            mpm_grid_start,
            grid_sum,
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
        kernel_initialize_packed_level_set_soft_body(
            soft_body,
            rigid_body,
            bounding_box,
            bounding_sphere,
            material,
            particle_num,
            soft_num,
            point_start,
            point_count,
            grid_start,
            mpm_grid_start,
            vertice_num,
            surface_num,
            vec3f(template_ptr.objects.grid.min_box()),
            vec3f(template_ptr.objects.grid.max_box()),
            template_ptr.boundings.r_bound,
            vec3f(template_ptr.boundings.x_bound),
            surface_sum,
            template_ptr.surface_area,
            grid_sum,
            template_ptr.objects.grid.grid_space,
            vec3i(template_ptr.objects.grid.gnum),
            template_ptr.objects.grid.extent,
            vec3f(base_shape_min),
            vec3f(base_shape_max),
            base_shape_radius,
            float(np.sum(point_volume)),
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
            template_point_start,
            template_surface_start,
            template_sdf_start,
            grid_type,
            support.grid_space,
            self.coordinates_are_mass_centers,
        )
        kernel_initialize_packed_level_set_soft_body_grids(
            soft_body,
            scene.rigid_grid,
            scene.soft_grid,
            scene.soft_grid_owner,
            scene.soft_grid_local,
            soft_num,
            body_count,
            grid_start,
            mpm_grid_start,
            grid_sum,
            compact_grid_sum,
            distance_fields,
        )
        kernel_initialize_packed_level_set_soft_body_surface(
            soft_body,
            rigid_body,
            master,
            scene.vertice,
            particle_num,
            soft_num,
            body_count,
            vertice_num,
            surface_num,
            surface_sum,
            surface_nodes,
            surface_parameters,
            init_v,
            init_w,
        )
        kernel_initialize_packed_level_set_soft_body_points(
            soft_body,
            soft_point,
            rigid_body,
            bounding_box,
            material,
            scene.rigid_grid,
            scene.soft_surface_point_id,
            particle_num,
            soft_num,
            body_count,
            point_start,
            point_count,
            kernel_points,
            kernel_point_volumes,
            group_id,
            mat_id,
            init_v,
            init_w,
        )
        print(" Level-set MPM soft body Information ".center(71, "-"))
        self.print_particle_info(group_id, mat_id, init_v, init_w, fix_v=is_fix, fix_w=is_fix, body_num=body_count)

        faces = scene.add_connectivity(body_count, surface_sum, template_ptr.objects)
        scene.particleNum[0] += body_count
        scene.rigidNum[0] += body_count
        scene.softNum[0] += body_count
        scene.softPointNum[0] += point_count * body_count
        scene.softGridNum[0] += compact_grid_sum * body_count
        scene.softMechanicalLogicalGridNum[0] += topology.logical_count * body_count
        scene.softLogicalGridNum[0] += grid_sum * body_count
        scene.softMaxLogicalGridNum[0] = max(int(scene.softMaxLogicalGridNum[0]), grid_sum)
        scene.surfaceNum[0] += surface_sum * body_count
        scene.gridNum[0] += grid_sum * body_count
        scene.gridID.append(int(scene.gridNum[0]))
        from src.mpm.soft_particle.ReferenceMap import initialize_soft_reference_sdf

        initialize_soft_reference_sdf(
            int(scene.softNum[0]),
            scene.soft,
            scene.rigid_grid,
            scene.soft_levelset_initial_sdf,
            scene.soft_levelset_initial_sdf_initialized,
        )
        self.faces = np.append(self.faces, faces).reshape(-1, 3)
