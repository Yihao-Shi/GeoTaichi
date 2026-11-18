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
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.mpm.generator.InsertionKernel import (
    kernel_initialize_level_set_soft_body,
    kernel_initialize_level_set_soft_body_grids,
    kernel_initialize_level_set_soft_body_points,
    kernel_initialize_level_set_soft_body_surface,
    kernel_prepare_soft_grid_topology,
)
from src.utils.Orientation import Orientation
from src.utils.TypeDefinition import vec3f, vec3i


class SoftParticleCreatorMixin(object):

    @staticmethod
    def compact_soft_grid_nodes_in_region(region, support, topology):
        """Resolve a template-space box to compact mechanical-grid node IDs."""
        if not isinstance(region, dict):
            raise TypeError("MechanicalGridBoundary must be a dictionary")
        if support.grid_base_space not in (0.0, support.grid_space):
            raise SimulationStateError("MechanicalGridBoundary currently requires a uniform grid")
        lower = np.asarray(DictIO.get_essential(region, "StartPoint"), dtype=np.float64)
        upper = np.asarray(DictIO.get_essential(region, "EndPoint"), dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("MechanicalGridBoundary StartPoint and EndPoint must be 3-vectors")
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        velocity = np.asarray(
            DictIO.get_alternative(region, "Velocity", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        if velocity.shape != (3,) or not np.allclose(velocity, 0.0):
            raise ValueError("LSMPM MechanicalGridBoundary currently supports fixed zero " "velocity only")
        shape = np.asarray(topology.compact_shape, dtype=np.int64)
        offset = np.asarray(topology.compact_origin, dtype=np.int64)
        axes = tuple(
            float(support.grid_origin[d])
            + (offset[d] + np.arange(shape[d], dtype=np.float64)) * float(support.grid_space)
            for d in range(3)
        )
        selected_axes = tuple(
            np.flatnonzero((axis >= lower[d] - 1.0e-12) & (axis <= upper[d] + 1.0e-12)) for d, axis in enumerate(axes)
        )
        if any(index.size == 0 for index in selected_axes):
            raise SimulationStateError("MechanicalGridBoundary selects no compact mechanical-grid node")
        ii, jj, kk = np.meshgrid(*selected_axes, indexing="ij")
        local = ii + jj * shape[0] + kk * shape[0] * shape[1]
        return np.ascontiguousarray(np.unique(local), dtype=np.int32)

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
            raise ConfigurationError("SoftBody with level-set MPM is only supported when scheme is /LSMPM/")
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
        mpm_grid_start = int(scene.softGridNum[0])
        surface_num = int(scene.surfaceNum[0])

        name = DictIO.get_essential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)
        com_pos = DictIO.get_essential(template, "BodyPoint")
        equiv_rad = DictIO.get_alternative(template, "Radius", None)
        bounding_rad = DictIO.get_alternative(template, "BoundingRadius", None)
        scale_factor = DictIO.get_alternative(template, "ScaleFactor", None)
        orientation = DictIO.get_alternative(template, "BodyOrientation", None)
        points_per_cell = DictIO.get_alternative(
            template, "MaterialPointsPerCell", DictIO.get_alternative(template, "ParticlePerCell", 1)
        )
        center_quadrature = DictIO.get_alternative(template, "CenterMaterialPointQuadrature", False)
        custom_material_points = DictIO.get_alternative(template, "MaterialPointCoordinates", None)
        custom_point_volume = DictIO.get_alternative(template, "MaterialPointVolume", None)
        mechanical_grid_spacing = DictIO.get_alternative(
            template,
            "MechanicalGridSpacing",
            getattr(
                template_ptr,
                "soft_mechanical_grid_spacing",
                sims.soft_mechanical_grid_spacing_ratio * float(template_ptr.objects.eqradius),
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
        mechanical_grid_boundary = DictIO.get_alternative(template, "MechanicalGridBoundary", None)
        set_orientations = Orientation(orientation)

        group_id = DictIO.get_essential(template, "GroupID")
        mat_id = DictIO.get_essential(template, "MaterialID")
        init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.get_alternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.get_alternative(template, "FixMotion", ["Free", "Free", "Free"])
        is_fix = vec3i([DictIO.get_essential(self.FIX, i) for i in fix_str])

        if isinstance(scale_factor, (float, int)):
            equiv_rad = float(scale_factor) * template_ptr.objects.eqradius
        elif isinstance(equiv_rad, (float, int)):
            scale_factor = float(equiv_rad) / template_ptr.objects.eqradius
        elif isinstance(bounding_rad, (float, int)):
            equiv_rad = template_ptr.objects.eqradius / template_ptr.boundings.r_bound * bounding_rad
            scale_factor = float(equiv_rad) / template_ptr.objects.eqradius
        else:
            raise SimulationStateError(
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
        point_count = int(material_points.shape[0])
        (
            template_point_start,
            template_surface_start,
            template_sdf_start,
            grid_type,
        ) = register_soft_template_support(scene, support)
        grid_start = int(scene.gridNum[0])
        grid_sum = int(template_ptr.objects.grid.gridSum)
        compact_grid_sum = topology.compact_count
        vertice_num = int(scene.verticeID[-1] + scene.surfaceNum[0])
        shape_min, shape_max, shape_radius = soft_template_shape_bounds(
            surface_nodes,
            kernel_points,
            kernel_point_volumes,
            scale_factor,
        )

        scene.check_particle_num(sims, particle_number=1)
        scene.check_soft_body_number(sims, soft_body_number=1)
        scene.check_material_point_number(sims, point_count)
        scene.check_surface_node_number(sims, template_ptr.surface_node_number)
        if grid_start + grid_sum > scene.rigid_grid.shape[0]:
            raise ValueError("The level-set grid storage should be enlarged to: ", grid_start + grid_sum)
        if mpm_grid_start + compact_grid_sum > scene.soft_grid_capacity:
            raise ValueError(
                "The compact soft MPM grid storage should be enlarged to: ",
                mpm_grid_start + compact_grid_sum,
            )

        kernel_prepare_soft_grid_topology(
            soft_body,
            soft_num,
            1,
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

        kernel_initialize_level_set_soft_body(
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
            template_ptr.surface_node_number,
            template_ptr.surface_area,
            grid_sum,
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
            group_id,
            mat_id,
            init_v,
            init_w,
            is_fix,
            template_point_start,
            template_surface_start,
            template_sdf_start,
            grid_type,
            support.grid_space,
        )
        kernel_initialize_level_set_soft_body_grids(
            soft_body,
            scene.rigid_grid,
            scene.soft_grid,
            scene.soft_grid_owner,
            scene.soft_grid_local,
            soft_num,
            grid_start,
            mpm_grid_start,
            grid_sum,
            distance_fields,
        )
        kernel_initialize_level_set_soft_body_surface(
            rigid_body,
            master,
            scene.vertice,
            particle_num,
            vertice_num,
            surface_num,
            template_ptr.surface_node_number,
            scale_factor,
            surface_nodes,
            surface_parameters,
            init_v,
            init_w,
        )
        kernel_initialize_level_set_soft_body_points(
            soft_body,
            soft_point,
            rigid_body,
            bounding_box,
            material,
            scene.rigid_grid,
            scene.soft_surface_point_id,
            particle_num,
            soft_num,
            point_start,
            point_count,
            scale_factor,
            kernel_points,
            kernel_point_volumes,
            group_id,
            mat_id,
            init_v,
            init_w,
        )

        if mechanical_grid_boundary is not None:
            boundaries = (
                mechanical_grid_boundary if isinstance(mechanical_grid_boundary, list) else [mechanical_grid_boundary]
            )
            fixed_local_nodes = np.unique(
                np.concatenate(
                    [
                        SoftParticleCreatorMixin.compact_soft_grid_nodes_in_region(boundary, support, topology)
                        for boundary in boundaries
                    ]
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
            upload_soft_velocity_constraints(
                constraint_start,
                mpm_grid_start,
                np.ascontiguousarray(fixed_local_nodes),
                scene.soft_velocity_constraint,
            )
            scene.softVelocityConstraintNum[0] = constraint_end

        print(" Level-set MPM soft body Information ".center(71, "-"))
        self.print_particle_info(group_id, mat_id, com_pos, init_v, init_w, fix_v=is_fix, fix_w=is_fix, name=name)
        scene.add_connectivity(1, template_ptr.surface_node_number, template_ptr.objects)
        scene.particleNum[0] += 1
        scene.rigidNum[0] += 1
        scene.softNum[0] += 1
        scene.softPointNum[0] += point_count
        scene.softGridNum[0] += compact_grid_sum
        scene.softMechanicalLogicalGridNum[0] += topology.logical_count
        scene.softLogicalGridNum[0] += grid_sum
        scene.softMaxLogicalGridNum[0] = max(int(scene.softMaxLogicalGridNum[0]), grid_sum)
        scene.surfaceNum[0] += template_ptr.surface_node_number
        scene.gridNum[0] += grid_sum
        scene.gridID.append(int(scene.gridNum[0]))

        from src.mpm.soft_particle.ReferenceMap import initialize_soft_reference_sdf

        initialize_soft_reference_sdf(
            int(scene.softNum[0]),
            scene.soft,
            scene.rigid_grid,
            scene.soft_levelset_initial_sdf,
            scene.soft_levelset_initial_sdf_initialized,
        )
