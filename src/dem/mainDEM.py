import numpy as np
import taichi as ti

from src.dem.SceneManager import myScene
from src.dem.GenerateManager import GenerateManager
from src.dem.ContactManager import ContactManager
from src.dem.engines.ExplicitEngine import ExplicitEngine
from src.dem.engines.AffineBodyEngine import AffineBodyEngine
from src.dem.DEMBase import Solver, AffineBodySolver
from src.dem.PostPlot import write_dem_vtk_file, write_lsdem_vtk_file
from src.dem.Recorder import WriteFile
from src.dem.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.SolverDiagnostics import SolverDiagnosticsMixin
from src.utils.SolverConsole import print_solver_section, runtime_architecture
from src.utils.StepRetry import StepRetryPolicy
from src.utils.TypeDefination import vec3f


class DEM(SolverDiagnosticsMixin):
    def __init__(self, title="A High Performance Multiscale and Multiphysics Simulator", log=True):
        if log:
            print("# =================================================================== #")
            print("#", "".center(67), "#")
            print("#", "Welcome to GeoTaichi -- Discrete Element Method Engine !".center(67), "#")
            print("#", "".center(67), "#")
            print("#", title.center(67), "#")
            print("#", "".center(67), "#")
            print("# =================================================================== #", "\n")
        self.sims = Simulation()
        self.scene = myScene()
        self.generator = GenerateManager()
        self.contactor = None
        self.enginer = None
        self.recorder = None
        self.solver = None
        self.first_run = True

    def set_configuration(self, log=True, **kwargs):
        if np.linalg.norm(np.array(self.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10:
            self.sims.set_domain(DictIO.GetEssential(kwargs, "domain"))
        self.sims.set_boundary(DictIO.GetAlternative(kwargs, "boundary", [None, None, None]))
        self.sims.set_gravity(DictIO.GetAlternative(kwargs, "gravity", vec3f([0.0, 0.0, -9.8])))
        self.sims.set_engine(DictIO.GetAlternative(kwargs, "engine", "SymplecticEuler"))
        self.sims.set_search(DictIO.GetAlternative(kwargs, "search", self.sims.search))
        self.sims.set_search_direction(DictIO.GetAlternative(kwargs, "search_direction", "Up"))
        self.sims.set_digital_elevation_contact_mode(
            DictIO.GetAlternative(kwargs, "digital_elevation_contact", self.sims.digital_elevation_contact_mode)
        )
        scheme = DictIO.GetAlternative(kwargs, "scheme", "DEM")
        if DictIO.GetAlternative(kwargs, "affine_body", False):
            scheme = "AffineBody"
        self.sims.set_dem_scheme(scheme)
        if scheme == "LSMPM" or "soft_shape_function" in kwargs or "lsmpm_shape_function" in kwargs:
            soft_shape_function = DictIO.GetAlternative(
                kwargs,
                "soft_shape_function",
                DictIO.GetAlternative(
                    kwargs,
                    "lsmpm_shape_function",
                    DictIO.GetAlternative(kwargs, "shape_function", self.sims.soft_shape_function),
                ),
            )
            self.sims.set_soft_shape_function(soft_shape_function)
            self.sims.set_soft_grid_storage(
                DictIO.GetAlternative(kwargs, "soft_grid_storage", self.sims.soft_grid_storage)
            )
            self.sims.set_soft_grid_type(DictIO.GetAlternative(kwargs, "soft_grid_type", self.sims.soft_grid_type))
            self.sims.set_soft_mechanical_grid_spacing_ratio(
                DictIO.GetAlternative(
                    kwargs,
                    "soft_mechanical_grid_spacing_ratio",
                    self.sims.soft_mechanical_grid_spacing_ratio,
                )
            )
            self.sims.set_soft_pic_fraction(
                DictIO.GetAlternative(kwargs, "soft_pic_fraction", self.sims.soft_pic_fraction)
            )
            soft_rigid_contact = DictIO.GetAlternative(
                kwargs, "soft_rigid_contact", DictIO.GetAlternative(kwargs, "lsmpm_soft_rigid_contact", None)
            )
            if soft_rigid_contact is not None:
                self.sims.set_lsmpm_soft_rigid_contact(soft_rigid_contact)
            soft_reinit = DictIO.GetAlternative(
                kwargs,
                "soft_levelset_reinitialization",
                DictIO.GetAlternative(kwargs, "lsmpm_levelset_reinitialization", None),
            )
            if isinstance(soft_reinit, dict):
                self.sims.set_soft_levelset_reinitialization(**soft_reinit)
            elif soft_reinit is not None:
                self.sims.set_soft_levelset_reinitialization(enabled=soft_reinit)
            self.sims.set_soft_levelset_reinitialization(
                advection_scheme=DictIO.GetAlternative(kwargs, "soft_levelset_advection_scheme", None),
                advection_cfl=DictIO.GetAlternative(kwargs, "soft_levelset_advection_cfl", None),
                advection_interval=DictIO.GetAlternative(kwargs, "soft_levelset_advection_interval", None),
                grad_threshold=DictIO.GetAlternative(kwargs, "soft_levelset_reinit_grad_threshold", None),
                check_interval=DictIO.GetAlternative(kwargs, "soft_levelset_reinit_check_interval", None),
                monitor_band=DictIO.GetAlternative(kwargs, "soft_levelset_reinit_monitor_band", None),
                reinit_band=DictIO.GetAlternative(kwargs, "soft_levelset_reinit_band", None),
                cfl=DictIO.GetAlternative(kwargs, "soft_levelset_reinit_cfl", None),
                iterations=DictIO.GetAlternative(kwargs, "soft_levelset_reinit_iterations", None),
                domain_check=DictIO.GetAlternative(kwargs, "soft_levelset_domain_check", None),
                domain_tolerance_cells=DictIO.GetAlternative(kwargs, "soft_levelset_domain_tolerance_cells", None),
            )
        affine_parameters = DictIO.GetAlternative(kwargs, "affine_body_parameters", None)
        if affine_parameters is not None:
            self.sims.set_affine_body_parameters(**affine_parameters)
        affine_assemble_type = DictIO.GetAlternative(kwargs, "affine_assemble_type", None)
        if affine_assemble_type is not None:
            self.sims.set_affine_body_parameters(assemble_type=affine_assemble_type)
        self.sims.set_track_energy(DictIO.GetAlternative(kwargs, "track_energy", self.sims.energy_tracking))
        self.sims.set_visualize(DictIO.GetAlternative(kwargs, "visualize", self.sims.visualize))
        self.sims.set_enable_shell(DictIO.GetAlternative(kwargs, "enable_shell", self.sims.enable_shell))
        self.sims.set_iterative_model(DictIO.GetAlternative(kwargs, "iterative_model", "LagrangianMultiplier"))
        self.sims.validate_configuration(require_memory=False)
        if log:
            self.print_basic_simulation_info()
            print("\n")

    def set_solver(self, solver, log=True):
        retry_policy = StepRetryPolicy(
            enabled=DictIO.GetAlternative(solver, "enable_step_retry", False),
            maximum_retries=DictIO.GetAlternative(solver, "step_retry_max_retries", 2),
            reduction=DictIO.GetAlternative(solver, "step_retry_reduction", 0.5),
            minimum_timestep=DictIO.GetAlternative(solver, "step_retry_minimum_timestep", 0.0),
        )
        retry_supported = self.sims.scheme == "AffineBody" or (
            self.sims.scheme == "LSMPM" and self.sims.lsmpm_soft_rigid_contact == "IPC"
        )
        if retry_policy.enabled and not retry_supported:
            raise ValueError("DEM step retry is available only for implicit AffineBody " "or LSMPM Soft-Affine IPC")
        self.sims.enable_step_retry = retry_policy.enabled
        self.sims.step_retry_max_retries = retry_policy.maximum_retries
        self.sims.step_retry_reduction = retry_policy.reduction
        self.sims.step_retry_minimum_timestep = retry_policy.minimum_timestep
        self.sims.set_timestep(DictIO.GetEssential(solver, "Timestep"))
        self.sims.set_simulation_time(DictIO.GetEssential(solver, "SimulationTime"))
        self.sims.set_CFL(DictIO.GetAlternative(solver, "CFL", 0.5))
        self.sims.set_adaptive_timestep(DictIO.GetAlternative(solver, "AdaptiveStep", 0))
        self.sims.set_save_interval(DictIO.GetEssential(solver, "SaveInterval"))
        self.sims.set_save_path(DictIO.GetAlternative(solver, "SavePath", "OutputData"))
        if log:
            self.print_solver_info()
            print("\n")

    def set_affine_body_parameters(self, **kwargs):
        self.sims.set_affine_body_parameters(**kwargs)

    def _infer_soft_template_support_capacity(self):
        point_num = 0
        surface_num = 0
        sdf_num = 0
        seen = set()
        for template in self.generator.myTemplate.values():
            cache = getattr(template, "soft_grid_preprocess_cache", {})
            for cached in cache.values():
                if len(cached) < 4:
                    continue
                support = cached[3]
                support_id = id(support)
                if support_id in seen:
                    continue
                seen.add(support_id)
                point_num += support.material_point_number
                surface_num += support.surface_node_number
                sdf_num += support.levelset_node_number
        return point_num, surface_num, sdf_num

    def memory_allocate(self, memory, log=True):
        self.sims.set_material_num(DictIO.GetEssential(memory, "max_material_number"))
        if self.sims.scheme == "DEM":
            self.sims.set_particle_num(DictIO.GetEssential(memory, "max_particle_number"))
            self.sims.set_sphere_num(DictIO.GetAlternative(memory, "max_sphere_number", 0))
            self.sims.set_clump_num(DictIO.GetAlternative(memory, "max_clump_number", 0))
        elif self.sims.scheme == "AffineBody":
            if "max_affine_body_number" in memory:
                affine_body_number = DictIO.GetEssential(memory, "max_affine_body_number")
            else:
                affine_body_number = DictIO.GetEssential(memory, "max_rigid_body_number")
            self.sims.set_rigid_body_num(affine_body_number)
            self.sims.set_rigid_template_num(DictIO.GetAlternative(memory, "max_rigid_template_number", 1))
            self.sims.set_surface_node_num(DictIO.GetAlternative(memory, "surface_node_number", 0))
        else:
            self.sims.set_rigid_body_num(DictIO.GetEssential(memory, "max_rigid_body_number"))
            self.sims.set_rigid_template_num(DictIO.GetAlternative(memory, "max_rigid_template_number", 1))
            if self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                self.sims.set_level_grid_num(DictIO.GetEssential(memory, "levelset_grid_number"))
                self.sims.set_surface_node_num(DictIO.GetEssential(memory, "surface_node_number"))
                self.sims.set_point_coordination_number(
                    DictIO.GetAlternative(memory, "point_coordination_number", [4, 2])
                )
                if self.sims.scheme == "LSMPM":
                    self.sims.set_soft_body_num(DictIO.GetEssential(memory, "max_soft_body_number"))
                    self.sims.set_material_point_num(DictIO.GetEssential(memory, "max_material_point_number"))
                    self.sims.set_soft_grid_num(
                        DictIO.GetAlternative(
                            memory,
                            "soft_grid_number",
                            DictIO.GetEssential(memory, "levelset_grid_number"),
                        )
                    )
                    self.sims.set_soft_velocity_constraint_num(
                        DictIO.GetAlternative(memory, "max_soft_velocity_constraint", 0)
                    )
                    inferred = self._infer_soft_template_support_capacity()
                    self.sims.set_soft_template_support_num(
                        DictIO.GetAlternative(
                            memory,
                            "soft_template_material_point_number",
                            inferred[0] or self.sims.max_material_point_num,
                        ),
                        DictIO.GetAlternative(
                            memory,
                            "soft_template_surface_node_number",
                            inferred[1] or self.sims.max_surface_node_num * max(self.sims.max_rigid_template_num, 1),
                        ),
                        DictIO.GetAlternative(
                            memory,
                            "soft_template_levelset_node_number",
                            inferred[2] or self.sims.max_level_grid_num * max(self.sims.max_rigid_template_num, 1),
                        ),
                    )
        self.sims.set_patch_num(DictIO.GetAlternative(memory, "max_patch_number", 0))
        self.sims.set_facet_num(DictIO.GetAlternative(memory, "max_facet_number", 0))
        self.sims.set_servo_wall_num(DictIO.GetAlternative(memory, "max_servo_wall_number", 0))
        self.sims.set_plane_num(DictIO.GetAlternative(memory, "max_plane_number", 0))
        self.sims.set_digital_elevation_facet_num(
            DictIO.GetAlternative(memory, "max_digital_elevation_facet_number", 0)
        )
        self.sims.set_compaction_ratio(DictIO.GetAlternative(memory, "compaction_ratio", [0.15, 0.05]))
        self.sims.set_hierarchical_level(DictIO.GetAlternative(memory, "hierarchical_level", 1))
        self.sims.set_rebuild_interval(DictIO.GetAlternative(memory, "bvh_rebuild_interval", 1000))
        if self.sims.search == "HierarchicalLinkedCell":
            self.sims.set_hierarchical_size(DictIO.GetEssential(memory, "hierarchical_size"))
        self.sims.define_work_load()

        self.sims.set_body_coordination_number(DictIO.GetAlternative(memory, "body_coordination_number", 16))
        affine_ipc = self.sims.scheme == "AffineBody" or (
            self.sims.scheme == "LSMPM" and self.sims.lsmpm_soft_rigid_contact == "IPC"
        )
        if affine_ipc:
            # Backwards-compatible defaults preserve the former primitive
            # buffer scale.  Production IPC scenes should set these two
            # topology-level capacities explicitly.
            body_coordination = self.sims.body_coordination_number
            if isinstance(body_coordination, (list, tuple, np.ndarray)):
                body_coordination = max(body_coordination)
            default_pt = 4 * self.sims.max_surface_node_num * int(body_coordination)
            default_ee = 12 * self.sims.max_surface_node_num * int(body_coordination)
            self.sims.set_affine_primitive_pair_capacity(
                DictIO.GetAlternative(memory, "max_point_triangle_pairs", default_pt),
                DictIO.GetAlternative(memory, "max_edge_edge_pairs", default_ee),
            )
            self.sims.set_affine_contact_block_capacity(
                DictIO.GetAlternative(memory, "affine_contact_block_capacity", 256)
            )
        self.sims.set_wall_coordination_number(
            DictIO.GetAlternative(memory, "wall_coordination_number", self.sims.max_wall_num)
        )
        self.sims.set_verlet_distance_multiplier(DictIO.GetAlternative(memory, "verlet_distance_multiplier", 0.0))
        self.sims.set_wall_per_cell(DictIO.GetAlternative(memory, "wall_per_cell", 4))
        self.sims.validate_configuration(require_memory=True)

        self.scene.activate_basic_class(self.sims)
        if self.sims.scheme in ("LSDEM", "LSMPM") and len(self.scene.prefixID) > 0:
            self.scene.add_rigid_template_grid_field(self.sims)
        if log:
            self.print_simulation_info()
            self.print_memory_info()
            self.print_neighbor_search_info()
            print("\n")

    def print_basic_simulation_info(self):
        print_solver_section(
            self._console_solver_name(),
            "Basic Configuration",
            [
                ("Simulation Type", runtime_architecture()),
                ("Simulation Domain", self.sims.domain),
                ("Boundary Condition", self.sims.boundary),
                ("Gravity", self.sims.gravity),
                ("DEM Scheme", self.sims.scheme),
            ],
        )

    def _console_solver_name(self):
        scheme = str(self.sims.scheme or "DEM")
        return "DEM" if scheme == "DEM" else f"DEM {scheme}"

    def print_simulation_info(self):
        entries = [
            ("Engine Type", self.sims.engine),
            ("Neighbor Search Type", self.sims.search),
            ("DEM Scheme", self.sims.scheme),
        ]
        if self.sims.scheme == "LSMPM":
            entries.extend(
                [
                    ("LSMPM Soft Shape Function", self.sims.soft_shape_function),
                    ("LSMPM Soft Grid Type", self.sims.soft_grid_type),
                    ("LSMPM Mechanical Grid h/R", self.sims.soft_mechanical_grid_spacing_ratio),
                    ("LSMPM Soft Grid Storage", self.sims.soft_grid_storage),
                    ("LSMPM Soft Level Set Reinit", self.sims.soft_levelset_reinitialization),
                ]
            )
        if self.sims.energy_tracking:
            entries.append(("Energy Tracking", "ON"))
        print_solver_section(self._console_solver_name(), "Engine Information", entries)

    def print_solver_info(self):
        print_solver_section(
            self._console_solver_name(),
            "Solver Information",
            [
                ("Engine Type", self.sims.engine),
                ("DEM Scheme", self.sims.scheme),
                ("Neighbor Search", self.sims.search),
                ("Initial Simulation Time", self.sims.current_time),
                ("Final Simulation Time", self.sims.current_time + self.sims.time),
                ("Time Step", self.sims.dt[None]),
                ("Adaptive Time Step", self.sims.adaptive_timestep),
                ("CFL", self.sims.CFL),
                ("Save Interval", self.sims.save_interval),
                ("Save Path", self.sims.path),
            ],
        )

    def print_memory_info(self):
        entries = [
            ("Maximum Materials", self.sims.max_material_num),
            ("Maximum Particles", self.sims.max_particle_num),
            ("Maximum Spheres", self.sims.max_sphere_num),
            ("Maximum Clumps", self.sims.max_clump_num),
            ("Maximum Rigid Bodies", self.sims.max_rigid_body_num),
            ("Maximum Walls", self.sims.max_wall_num),
            ("Maximum Servo Walls", self.sims.max_servo_wall_num),
            ("Body Coordination Number", self.sims.body_coordination_number),
            ("Wall Coordination Number", self.sims.wall_coordination_number),
            ("Particle Contact Capacity", self.sims.particle_contact_list_length),
            ("Wall Contact Capacity", self.sims.wall_contact_list_length),
        ]
        if self.sims.scheme in ("LSDEM", "LSMPM"):
            entries.extend(
                (
                    ("Maximum Level-set Grid Nodes", self.sims.max_level_grid_num),
                    ("Maximum Surface Nodes", self.sims.max_surface_node_num),
                )
            )
        if self.sims.scheme == "AffineBody":
            entries.extend(
                (
                    (
                        "Maximum Point-Triangle Pairs",
                        self.sims.max_point_triangle_pairs,
                    ),
                    (
                        "Maximum Edge-Edge Pairs",
                        self.sims.max_edge_edge_pairs,
                    ),
                    (
                        "Affine Contact Block Capacity",
                        self.sims.affine_contact_block_capacity,
                    ),
                )
            )
        if self.sims.scheme == "LSMPM":
            entries.extend(
                (
                    ("Maximum Soft Bodies", self.sims.max_soft_body_num),
                    ("Maximum Soft Material Points", self.sims.max_material_point_num),
                    ("Maximum Soft Grid Nodes", self.sims.max_soft_grid_num),
                )
            )
        print_solver_section(self._console_solver_name(), "Memory Information", entries)

    def print_neighbor_search_info(self):
        neighbor = None
        if self.contactor is not None:
            neighbor = self.contactor.neighbor
        print_solver_section(
            self._console_solver_name(),
            "Neighbor Search Information",
            [
                ("Search Method", self.sims.search),
                ("Search Direction", self.sims.search_direction),
                ("Runtime Search Object", type(neighbor).__name__ if neighbor is not None else None),
                ("Verlet Distance Multiplier", self.sims.verlet_distance_multiplier),
                ("Verlet Distance", self.sims.verlet_distance),
                ("Potential Particles per Body", self.sims.potential_particle_num),
                ("Potential Particle Pair Capacity", self.sims.max_potential_particle_pairs),
                ("Potential Wall Pair Capacity", self.sims.max_potential_wall_pairs),
                ("Particle Contact Capacity", self.sims.particle_contact_list_length),
                ("Wall Contact Capacity", self.sims.wall_contact_list_length),
            ],
        )

    def add_region(self, region):
        if type(region) is dict:
            self.generator.add_my_region(self.sims.dimension, self.sims.domain, region)
        elif type(region) is list:
            for region_dict in region:
                self.generator.add_my_region(self.sims.dimension, self.sims.domain, region_dict)

    def add_attribute(self, materialID, attribute):
        self.scene.add_attribute(self.sims, materialID, attribute)

    def add_template(self, template):
        types = self.sims.scheme
        if type(template) is dict:
            types = DictIO.GetAlternative(
                template, "TemplateType", DictIO.GetAlternative(template, "template_type", types)
            )
        self.generator.add_my_template(self.scene, template, types)
        if self.sims.max_particle_num > 0:
            if types == "LSDEM" or types == "LSMPM":
                self.scene.add_rigid_template_grid_field(self.sims)
            elif types == "PolySuperEllipsoid" or types == "PolySuperQuadrics":
                self.scene.add_rigid_implicit_surface_parameter(self.sims)

    def preprocess_soft_grid_template(
        self,
        name,
        points_per_cell=1,
        center_quadrature=False,
        verlet_distance_multiplier=None,
        material_points=None,
        point_volume=None,
        mechanical_grid_spacing=None,
        mechanical_grid_refinement=None,
        reference_volume=None,
    ):
        if self.sims.scheme != "LSMPM":
            raise RuntimeError("Soft-grid preprocessing requires scheme='LSMPM'")
        if name not in self.generator.myTemplate:
            raise KeyError(f"Template name: {name} is not set before")
        if verlet_distance_multiplier is None:
            verlet_distance_multiplier = self.sims.verlet_distance_multiplier[1]
        elif isinstance(verlet_distance_multiplier, (list, tuple)):
            verlet_distance_multiplier = verlet_distance_multiplier[1]
        if mechanical_grid_spacing is None:
            template_ptr = self.generator.myTemplate[name]
            mechanical_grid_spacing = self.sims.soft_mechanical_grid_spacing_ratio * float(
                template_ptr.objects.eqradius
            )
        else:
            template_ptr = self.generator.myTemplate[name]
        template_ptr.soft_mechanical_grid_spacing = float(mechanical_grid_spacing)
        template_ptr.soft_mechanical_grid_refinement = mechanical_grid_refinement
        template_ptr.soft_reference_volume = reference_volume

        from src.mpm.generator.BodyGenerator import (
            preprocess_soft_grid_template,
        )

        material_points, point_volume, topology, support = preprocess_soft_grid_template(
            self.generator.myTemplate[name],
            points_per_cell,
            center_quadrature,
            self.sims.soft_shape_function_type,
            self.sims.soft_grid_storage,
            verlet_distance_multiplier,
            material_points,
            point_volume,
            self.sims.soft_grid_type,
            mechanical_grid_spacing,
            mechanical_grid_refinement,
            reference_volume,
        )
        point_volume = np.asarray(point_volume, dtype=np.float64)
        return {
            "storage": topology.storage,
            "logical_grid_number": int(self.generator.myTemplate[name].objects.grid.gridSum),
            "levelset_grid_number": int(self.generator.myTemplate[name].objects.grid.gridSum),
            "mechanical_logical_grid_number": topology.logical_count,
            "soft_grid_number": topology.compact_count,
            "supported_grid_number": topology.support_count,
            "compact_origin": topology.compact_origin.copy(),
            "compact_shape": topology.compact_shape.copy(),
            "verlet_padding_cells": topology.verlet_padding_cells,
            "levelset_verlet_padding_cells": (topology.levelset_verlet_padding_cells),
            "levelset_extent_cells": topology.levelset_extent_cells,
            "levelset_deformation_padding_cells": (
                topology.levelset_extent_cells - topology.levelset_verlet_padding_cells
            ),
            "soft_grid_padding_cells": topology.padding_cells,
            "material_point_number": int(material_points.shape[0]),
            "point_volume": float(np.mean(point_volume)),
            "point_volume_min": float(np.min(point_volume)),
            "point_volume_max": float(np.max(point_volume)),
            "point_volume_sum": float(np.sum(point_volume)),
            "variable_point_volume": bool(
                not np.allclose(
                    point_volume,
                    point_volume[0],
                    rtol=1.0e-12,
                    atol=0.0,
                )
            ),
            "soft_grid_type": self.sims.soft_grid_type,
            "mechanical_grid_origin": support.grid_origin.copy(),
            "mechanical_grid_shape": support.grid_shape.copy(),
            "mechanical_grid_spacing": float(support.grid_space),
            "mechanical_grid_base_spacing": float(support.grid_base_space or support.grid_space),
            "mechanical_grid_refined": bool(
                support.grid_base_space and not np.isclose(support.grid_base_space, support.grid_space)
            ),
            "template_support_bytes": int(
                support.point_node.nbytes
                + support.point_shape.nbytes
                + support.point_dshape.nbytes
                + support.point_count.nbytes
                + support.surface_node.nbytes
                + support.surface_shape.nbytes
                + support.surface_count.nbytes
                + support.sdf_node.nbytes
                + support.sdf_shape.nbytes
                + support.sdf_count.nbytes
            ),
        }

    def create_body(self, body):
        self.generator.create_body(body, self.sims, self.scene)

    def create_body_batch(self, body):
        return self.generator.create_body_batch(body, self.sims, self.scene)

    def add_body(self, body):
        self.generator.add_body(body, self.sims, self.scene)

    def add_body_from_file(self, body):
        self.generator.read_body_file(body, self.sims, self.scene)

    def add_joint(self, joint):
        if self.sims.scheme != "AffineBody" and not self.is_lsmpm_soft_affine_ipc():
            raise RuntimeError("AffineBody joints require the AffineBody scheme or LSMPM-Affine IPC coupling")
        if self.enginer is not None and getattr(self.enginer, "operator", None) is not None:
            raise RuntimeError("AffineBody joints must be added before the first simulation run")
        self.scene.add_affine_joint(joint)

    def set_joint_target_angle(self, jointID, target_angle):
        """Set an affine revolute-joint motor target in degrees."""
        self.scene.set_affine_joint_target_angle(jointID, target_angle)
        if self.enginer is not None and getattr(self.enginer, "operator", None) is not None:
            self.enginer.operator.set_joint_target_angle(jointID, target_angle)

    def solve_affine_adjoint(self, loss_gradient):
        if self.enginer is None or not isinstance(self.enginer, AffineBodyEngine):
            raise RuntimeError("solve_affine_adjoint requires an initialized AffineBody simulation")
        return self.enginer.solve_adjoint(loss_gradient)

    def differentiate_affine_step(self, loss_gradient):
        if self.enginer is None or not isinstance(self.enginer, AffineBodyEngine):
            raise RuntimeError("differentiate_affine_step requires an initialized AffineBody simulation")
        return self.enginer.differentiate_step_parameters(loss_gradient)

    def differentiable_affine(self, steps):
        """Create a device-resident fixed-step AffineBody trajectory tape."""
        if self.enginer is None:
            self.add_essentials()
        if not isinstance(self.enginer, AffineBodyEngine):
            raise RuntimeError("differentiable_affine requires an AffineBody simulation")
        if self.enginer.operator is None:
            self.enginer.initialize(self.sims, self.scene)
        from src.dem.engines.DifferentiableABD import DifferentiableABD

        return DifferentiableABD(self.enginer, steps)

    def differentiable_soft_affine(self, steps):
        """Create a soft MPM-ABD trajectory tape; plastic soft materials are rejected."""
        if self.enginer is None:
            self.add_essentials()
        from src.mpdem.engines.SoftAffineIPCEngine import SoftAffineIPCEngine

        if not isinstance(self.enginer, SoftAffineIPCEngine):
            raise RuntimeError("differentiable_soft_affine requires LSMPM-Affine IPC coupling")
        if self.enginer.operator is None:
            self.enginer.initialize(self.sims, self.scene)
        from src.mpdem.engines.DifferentiableSoftAffine import (
            DifferentiableSoftAffine,
        )

        return DifferentiableSoftAffine(self.enginer, self.sims, self.scene, steps)

    def add_wall(self, body):
        self.generator.add_wall(body, self.sims, self.scene)

    def add_wall_from_file(self, body):
        self.generator.read_wall_file(body, self.sims, self.scene)

    def choose_neighbor(self):
        if self.sims.scheme == "AffineBody":
            return
        if self.contactor is None:
            self.contactor = ContactManager()
            self.contactor.choose_neighbor(self.sims, self.scene)

    def choose_contact_model(self, particle_particle_contact_model=None, particle_wall_contact_model=None):
        if self.sims.scheme == "AffineBody":
            return
        self.choose_neighbor()
        if self.sims.max_material_num == 0:
            raise RuntimeError("memory_allocate should be launched first!")
        self.sims.set_particle_particle_contact_model(particle_particle_contact_model)
        self.sims.set_particle_wall_contact_model(particle_wall_contact_model)
        self.contactor.particle_particle_initialize(self.sims)
        self.contactor.particle_wall_initialize(self.sims)

    def add_property(self, materialID1, materialID2, property, dType="all"):
        if self.sims.scheme == "AffineBody" or (
            self.sims.scheme == "LSMPM" and self.sims.lsmpm_soft_rigid_contact == "IPC"
        ):
            self.scene.add_affine_contact_property(materialID1, materialID2, property, dType)
            return
        if self.contactor is None:
            raise RuntimeError("Please choose contact model /DEM.choose_contact_model/ first")
        self.contactor.add_contact_property(self.sims, materialID1, materialID2, property, dType)

    def inherit_property(self, materialID, property):
        pass

    def load_history_contact(self):
        file_number = DictIO.GetAlternative(self.sims.history_contact_path, "file_number", 0)
        ppcontact = DictIO.GetAlternative(self.sims.history_contact_path, "ppcontact", None)
        pwcontact = DictIO.GetAlternative(self.sims.history_contact_path, "pwcontact", None)

        if not ppcontact is None:
            self.contactor.physpp.restart(self.contactor.neighbor, file_number, ppcontact, True)
        if not pwcontact is None:
            self.contactor.physpw.restart(self.contactor.neighbor, file_number, pwcontact, False)

    def select_save_data(
        self,
        particle=True,
        sphere=False,
        clump=False,
        surface=True,
        grid=False,
        bounding=False,
        wall=False,
        particle_particle_contact=False,
        particle_wall_contact=False,
    ):
        self.sims.set_save_data(
            particle, sphere, clump, surface, grid, bounding, wall, particle_particle_contact, particle_wall_contact
        )
        self.scene.activate_surface_node_visualization(self.sims)

    def read_restart(
        self,
        file_number,
        file_path,
        particle=True,
        sphere=True,
        clump=False,
        wall=True,
        servo=False,
        ppcontact=True,
        pwcontact=True,
        is_continue=True,
    ):
        self.sims.set_is_continue(is_continue)
        if self.sims.is_continue:
            self.sims.current_print = file_number
        particle_path = None
        clump_path = None
        sphere_path = None
        wall_path = None
        servo_path = None
        ppcontact_path = None
        pwcontact_path = None

        if particle:
            if self.sims.scheme == "DEM":
                particle_path = file_path + f"/particles/DEMParticle{file_number:06d}.npz"
                if sphere:
                    sphere_path = file_path + f"/particles/DEMSphere{file_number:06d}.npz"
                if clump:
                    clump_path = file_path + f"/particles/DEMClump{file_number:06d}.npz"
                if sphere is False and clump is False:
                    raise RuntimeError("sphere or clump file is not exist")
            elif self.sims.scheme == "LSDEM":
                rigid_path = file_path + f"/particles/LSDEMRigid{file_number:06d}.npz"
                surface_path = file_path + f"/particles/LSDEMSurface{file_number:06d}.npz"
                boundingsphere_path = file_path + f"/particles/BoundingSphere{file_number:06d}.npz"
                boundingbox_path = file_path + f"/particles/BoundingBox{file_number:06d}.npz"
        if wall:
            wall_path = file_path + f"/walls/DEMWall{file_number:06d}.npz"
            if servo:
                servo_path = file_path + f"/walls/DEMServo{file_number:06d}.npz"
        if ppcontact:
            ppcontact_path = file_path + "/contacts"
        if pwcontact:
            pwcontact_path = file_path + "/contacts"

        if particle:
            if self.sims.scheme == "DEM":
                self.add_body_from_file(
                    body={
                        "FileType": "NPZ",
                        "Template": {
                            "Restart": True,
                            "ParticleFile": particle_path,
                            "SphereFile": sphere_path,
                            "ClumpFile": clump_path,
                        },
                    }
                )
            elif self.sims.scheme == "LSDEM":
                self.add_body_from_file(
                    body={
                        "FileType": "NPZ",
                        "Template": {
                            "Restart": True,
                            "RigidFile": rigid_path,
                            "SurfaceFile": surface_path,
                            "BoundingSphereFile": boundingsphere_path,
                            "BoundingBoxFile": boundingbox_path,
                        },
                    }
                )
        if wall:
            self.add_wall_from_file(body={"FileType": "NPZ", "WallFile": wall_path, "ServoFile": servo_path})
        self.sims.history_contact_path.update(
            file_number=file_number, ppcontact=ppcontact_path, pwcontact=pwcontact_path
        )

    def modify_parameters(self, **kwargs):
        if len(kwargs) > 0:
            self.sims.set_simulation_time(DictIO.GetEssential(kwargs, "SimulationTime"))
            if "Timestep" in kwargs:
                self.sims.set_timestep(DictIO.GetEssential(kwargs, "Timestep"))
            if "CFL" in kwargs:
                self.sims.set_CFL(DictIO.GetEssential(kwargs, "CFL"))
            if "AdaptiveTimestep" in kwargs:
                self.sims.set_adaptive_timestep(DictIO.GetEssential(kwargs, "AdaptiveStep"))
            if "SaveInterval" in kwargs:
                self.sims.set_save_interval(DictIO.GetEssential(kwargs, "SaveInterval"))
            if "SavePath" in kwargs:
                self.sims.set_save_path(DictIO.GetEssential(kwargs, "SavePath"))
            if "gravity" in kwargs:
                self.sims.set_gravity(DictIO.GetEssential(kwargs, "gravity"))

    def add_engine(self, callback):
        if self.enginer is None:
            if self.sims.scheme == "AffineBody":
                self.enginer = AffineBodyEngine(self.scene, self.contactor)
            elif self.is_lsmpm_soft_affine_ipc():
                from src.mpdem.engines.SoftAffineIPCEngine import SoftAffineIPCEngine

                self.enginer = SoftAffineIPCEngine()
            else:
                self.enginer = ExplicitEngine(self.scene, self.contactor)
        self.enginer.choose_engine(self.sims, self.scene)
        if self.sims.scheme == "LSMPM" and len(self.scene.affine_bodies) > 0:
            self.sims.freeze_lsmpm_soft_rigid_contact()
        if self.sims.scheme != "AffineBody":
            self.enginer.set_servo_mechanism(self.sims, callback)

    def add_recorder(self):
        if self.recorder is None:
            if self.sims.scheme == "AffineBody":
                self.recorder = WriteFile(self.sims, None, None, None, self.enginer)
            elif self.is_lsmpm_soft_affine_ipc():
                self.recorder = WriteFile(self.sims, None, None, None, self.enginer)
            else:
                self.recorder = WriteFile(
                    self.sims, self.contactor.physpp, self.contactor.physpw, self.contactor.neighbor
                )

    def add_solver(self, **kwargs):
        if self.solver is None:
            if self.sims.scheme == "AffineBody":
                self.solver = AffineBodySolver(self.sims, self.generator, self.contactor, self.enginer, self.recorder)
            elif self.is_lsmpm_soft_affine_ipc():
                from src.mpdem.engines.SoftAffineIPCBase import SoftAffineIPCSolver

                self.solver = SoftAffineIPCSolver(
                    self.sims, self.generator, self.contactor, self.enginer, self.recorder
                )
            else:
                self.solver = Solver(self.sims, self.generator, self.contactor, self.enginer, self.recorder)
        if DictIO.GetAlternative(kwargs, "reset_function", True):
            self.solver.postprocess = []
            self.solver.clear_preintegration_callback_functions()
        self.solver.set_preintegration_callback_function(DictIO.GetAlternative(kwargs, "preintegration_function", None))
        self.solver.set_callback_function(DictIO.GetAlternative(kwargs, "function", None))
        self.solver.set_particle_calm(self.scene, DictIO.GetAlternative(kwargs, "calm", None))

    def add_postfunctions(self, **functions):
        self.solver.set_callback_function(functions)

    def is_lsmpm_soft_affine_ipc(self):
        return (
            self.sims.scheme == "LSMPM"
            and self.sims.lsmpm_soft_rigid_contact == "IPC"
            and len(self.scene.affine_bodies) > 0
        )

    def add_essentials(self, **kwargs):
        if self.sims.scheme == "AffineBody":
            self.sims.validate_configuration(require_memory=True)
            self.add_engine(DictIO.GetAlternative(kwargs, "callback", None))
            self.add_recorder()
            self.add_solver(**kwargs)
            self.scene.set_boundary_condition(self.sims)
            return
        if self.is_lsmpm_soft_affine_ipc():
            self.sims.validate_configuration(require_memory=True)
            self.add_engine(DictIO.GetAlternative(kwargs, "callback", None))
            self.add_recorder()
            self.add_solver(**kwargs)
            self.scene.set_boundary_condition(self.sims)
            if self.sims.is_continue:
                self.solver.last_save_time = 1.0 * self.sims.current_time
                self.sims.current_print += 1
                self.sims.set_is_continue(False)
            return
        self.sims.validate_configuration(require_memory=True)
        if self.contactor is None:
            self.choose_contact_model()
        if self.sims.max_particle_num >= 0:
            if self.contactor.have_initialise is False:
                self.contactor.initialize(self.sims, self.scene, **kwargs)
        else:
            if self.sims.coupling is False:
                raise RuntimeError("Particle should be added first")
        if self.first_run:
            self.load_history_contact()
        self.add_engine(DictIO.GetAlternative(kwargs, "callback", None))
        self.add_recorder()
        if self.sims.coupling == False:
            self.add_solver(**kwargs)
        if (self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM") and self.sims.max_particle_num > 0:
            self.check_verlet_distance_multiplier()
            self.scene.check_radius()
        self.scene.set_boundary_condition(self.sims)
        if self.sims.is_continue:
            if self.sims.coupling is False:
                self.solver.last_save_time = 1.0 * self.sims.current_time
                self.sims.current_print += 1
            self.sims.set_is_continue(False)

    def check_verlet_distance_multiplier(self):
        equivalent_rad = self.scene.find_particle_min_radius(self.sims)
        self.sims.set_point_verlet_distance(equivalent_rad)
        self.sims.check_multiplier(
            max(self.contactor.physpp.find_max_penetration(), self.contactor.physpw.find_max_penetration())
        )
        self.sims.check_grid_extent(*self.scene.find_expect_extent(self.sims, self.sims.point_verlet_distance))

    def set_static_wall(self, static_wall=True):
        self.sims.set_static_wall(static_wall)

    def servo_switch(self, status="On"):
        self.sims.update_servo_status(status)
        if self.sims.servo_status == "Off" and self.sims.wall_type == 1:
            self.scene.wall.v.fill(0)

    def set_window(self, window):
        self.sims.set_window_parameters(window)

    def run(self, visualize=False, **kwargs):
        self.add_essentials(**kwargs)
        if self.sims.scheme == "AffineBody":
            self.solver.Solver(self.scene)
            self.first_run = False
            return
        self.check_critical_timestep()
        if visualize is False:
            self.solver.Solver(self.scene)
        else:
            self.solver.Visualize(self.scene)
        self.first_run = False

    def check_critical_timestep(self):
        print("#", " Check Timestep ... ...".ljust(67))
        critical_timestep = self.get_critical_timestep()
        if self.sims.CFL * critical_timestep < self.sims.dt[None]:
            self.sims.update_critical_timestep(critical_timestep)
        else:
            print("The prescribed time step is sufficiently small\n")

    def get_critical_timestep(self):
        return min(
            self.contactor.physpp.calcu_critical_timesteps(self.scene),
            self.contactor.physpw.calcu_critical_timesteps(self.scene),
        )

    def update_material_properties(self, materialID, property_name, value, override=True):
        self.scene.update_material_properties(override, materialID, property_name, value)

    def update_particle_properties(
        self, property_name, value, override=True, bodyID=None, region_name=None, function=None
    ):
        if not bodyID is None:
            self.scene.update_particle_properties(override, property_name, value, bodyID)
        elif not region_name is None:
            region: RegionFunction = self.generator.get_region_ptr(region_name)
            self.scene.update_particle_properties_in_region(self.sims, override, property_name, value, region.function)
        elif not function is None:
            self.scene.update_particle_properties_in_region(
                self.sims, override, property_name, value, ti.pyfunc(function)
            )
        if not self.first_run:
            self.contactor.neighbor.pre_neighbor(self.scene)

    def update_wall_status(self, wallID, property_name, value, override=True):
        self.scene.update_wall_properties(self.sims, override, property_name, value, wallID)
        if property_name == "Status":
            self._refresh_wall_contact_topology()

    def _refresh_wall_contact_topology(self):
        if not self.first_run and self.contactor is not None and self.enginer is not None:
            # A runtime topology change cannot wait for particle motion to
            # exhaust the Verlet skin.  Reinitialize the broad phase, both
            # LSDEM levels when present, and their contact-history tables
            # before the next force assembly.
            self.enginer.pre_calculation(self.sims, self.scene, self.contactor.neighbor)

    def update_contact_properties(self, materialID1, materialID2, property_name, value, overide=True):
        self.contactor.update_contact_property(self.sims, materialID1, materialID2, property_name, value, overide)

    def delete_particles(self, bodyID=None, region_name=None, function=None):
        if not bodyID is None:
            self.scene.delete_particles(bodyID)
        elif not region_name is None:
            region: RegionFunction = self.generator.get_region_ptr(region_name)
            self.scene.delete_particles_in_region(self.sims, region.function)
        elif not function is None:
            self.scene.delete_particles_in_region(self.sims, ti.pyfunc(function))

    def delete_walls(self, wallID):
        self.scene.delete_walls(self.sims, wallID)
        self._refresh_wall_contact_topology()

    def save_data(self):
        if self.solver is None:
            self.add_essentials()
        self.solver.save_file(self.scene)

    def postprocessing(self, start_file=0, end_file=-1, read_path=None, write_path=None, scheme="DEM", **kwargs):
        if read_path is None:
            read_path = self.sims.path
            if write_path is None:
                write_path = self.sims.path + "/vtks"
            elif not write_path is None:
                write_path = read_path + "/vtks"

        if not read_path is None and write_path is None:
            write_path = read_path + "/vtks"

        if not write_path.endswith("vtks"):
            write_path = write_path + "/vtks"

        scheme = self.sims.scheme if scheme is None else scheme
        self.sims.set_dem_scheme(scheme)

        if self.sims.scheme == "DEM":
            write_dem_vtk_file(self.sims, start_file, end_file, read_path, write_path, kwargs)
        elif self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
            write_lsdem_vtk_file(self.sims, start_file, end_file, read_path, write_path, kwargs)
