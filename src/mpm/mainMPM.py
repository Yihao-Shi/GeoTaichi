import numpy as np
import taichi as ti

from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.mpm.engines.ULExplicitEngine import ULExplicitEngine
from src.mpm.engines.ULExplicitTwoPhaseEngine import ULExplicitTwoPhaseEngine
from src.mpm.engines.ULSemiImplicitTwoPhaseEngine import ULSemiImplicitTwoPhaseEngine
from src.mpm.engines.ULSemiImplicitTwoPhaseDoubleLayerEngine import ULSemiImplicitTwoPhaseDoubleLayerEngine
from src.mpm.engines.ULSemiImplicitTwoPhaseEngine_u_p import ULSemiImplicitTwoPhaseEngine_u_p
from src.mpm.engines.TLExplicitEngine import TLExplicitEngine
from src.mpm.engines.ULImplicitEngine import ImplicitEngine
from src.mpm.engines.IncompressibleEngine import IncompressibleEngine
from src.mpm.GenerateManager import GenerateManager
from src.mpm.MPMBase import Solver
from src.mpm.PostPlot import write_vtk_file
from src.mpm.Recorder import WriteFile
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.SolverDiagnostics import SolverDiagnosticsMixin
from src.utils.SolverConsole import (
    print_simulation_start,
    print_solver_section,
    runtime_architecture,
)
from src.utils.StepRetry import StepRetryPolicy
import src.utils.GlobalVariable as GlobalVariable


class MPM(SolverDiagnosticsMixin):
    def __init__(self, title="A High Performance Multiscale and Multiphysics Simulator", log=True):
        if log:
            print("# =================================================================== #")
            print("#", "".center(67), "#")
            print("#", "Welcome to GeoTaichi -- Material Point Method Engine !".center(67), "#")
            print("#", "".center(67), "#")
            print("#", title.center(67), "#")
            print("#", "".center(67), "#")
            print("# =================================================================== #", "\n")
        self.sims = Simulation()
        self.scene = myScene()
        self.generator = GenerateManager()
        self.enginer = None
        self.neighbor = None
        self.recorder = None
        self.solver = None
        self.first_run = True
        self.direct_bodies = None
        self.direct_ground = None
        self.direct_dirichlet = None
        self.direct_neumann = None
        self.direct_material = {}
        self.direct_solver = {}
        self.direct_element = {}

    def set_configuration(self, log=True, **kwargs):
        dimension = DictIO.GetOptional(kwargs, "dimension")
        if dimension is not None:
            GlobalVariable.DIMENSION = int(dimension)
        self.sims.set_dimension()
        self.sims.set_mpm_backend(
            DictIO.GetAlternative(
                kwargs,
                "mpm_backend",
                DictIO.GetAlternative(
                    kwargs, "solver_backend", DictIO.GetAlternative(kwargs, "backend", self.sims.mpm_backend)
                ),
            )
        )
        self.sims.set_ipc_contact(
            DictIO.GetAlternative(kwargs, "ipc", DictIO.GetAlternative(kwargs, "ipc_contact", self.sims.ipc_contact))
        )
        self.sims.set_direct_static_twophase(
            DictIO.GetAlternative(kwargs, "static_twophase", self.sims.direct_static_twophase)
        )
        self.sims.set_is_2DAxisy(
            bool(
                DictIO.GetAlternative(
                    kwargs,
                    "axisymmetric",
                    DictIO.GetAlternative(
                        kwargs,
                        "is_axisymmetric",
                        DictIO.GetAlternative(kwargs, "is_2DAxisy", False),
                    ),
                )
            )
        )
        self.sims.axis_offset = float(DictIO.GetAlternative(kwargs, "axis_offset", 0.0))
        if not np.isfinite(self.sims.axis_offset):
            raise ValueError("MPM axis_offset must be finite")
        self.sims.set_mode(DictIO.GetAlternative(kwargs, "mode", "Normal"))
        soft_particle = DictIO.GetOptional(kwargs, "soft_particle")
        if soft_particle is None:
            soft_particle = DictIO.GetOptional(kwargs, "soft_particle_mode")
        if soft_particle is not None:
            self.sims.set_soft_particle_mode(
                soft_particle,
                DictIO.GetAlternative(kwargs, "soft_particle_backend", "LevelSetTLMPM"),
            )
        elif self.sims.mode == "SoftParticle":
            self.sims.set_soft_particle_mode(True)
        if np.linalg.norm(np.array(self.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10:
            domain = DictIO.GetOptional(kwargs, "domain")
            if domain is not None:
                self.sims.set_domain(domain)
            elif not self.sims.is_direct_backend():
                self.sims.set_domain(DictIO.GetEssential(kwargs, "domain"))
        self.sims.set_boundary(DictIO.GetAlternative(kwargs, "boundary", [None, None, None]))
        self.sims.set_gravity(
            DictIO.GetAlternative(kwargs, "gravity", [0.0, 0.0, -9.8] if self.sims.dimension == 3 else [0.0, -9.8])
        )
        self.sims.set_background_damping(DictIO.GetAlternative(kwargs, "background_damping", 0.0))
        self.sims.set_alpha(DictIO.GetAlternative(kwargs, "alphaPIC", 0.0))
        self.sims.set_mapping_scheme(DictIO.GetAlternative(kwargs, "mapping", "MUSL"))
        self.sims.set_stabilize_technique(DictIO.GetAlternative(kwargs, "stabilize", None))
        self.sims.set_gauss_integration(DictIO.GetAlternative(kwargs, "gauss_number", 0))
        self.sims.set_boundary_direction(DictIO.GetAlternative(kwargs, "boundary_direction_detection", False))
        self.sims.set_free_surface_detection(DictIO.GetAlternative(kwargs, "free_surface_detection", False))
        self.sims.set_fluid_level_set(DictIO.GetAlternative(kwargs, "fluid_level_set", False))
        self.sims.set_fluid_domain_volume_fraction(
            DictIO.GetAlternative(kwargs, "fluid_domain_volume_fraction", self.sims.fluid_domain_volume_fraction)
        )
        self.sims.set_velocity_projection_scheme(DictIO.GetAlternative(kwargs, "velocity_projection", "PIC/FLIP"))
        self.sims.set_moving_least_square(DictIO.GetAlternative(kwargs, "moving_least_square", False))
        self.sims.set_shape_function(DictIO.GetAlternative(kwargs, "shape_function", "Linear"))
        self.sims.set_solver_type(DictIO.GetAlternative(kwargs, "solver_type", "Explicit"))
        self.sims.set_shape_smoothing(DictIO.GetAlternative(kwargs, "shape_smooth", 0.0))
        self.sims.set_pressure_smoothing(DictIO.GetAlternative(kwargs, "pressure_smoothing", False))
        self.sims.set_strain_smoothing(DictIO.GetAlternative(kwargs, "strain_smoothing", False))
        self.sims.set_configuration(DictIO.GetAlternative(kwargs, "configuration", "ULMPM"))
        self.sims.set_material_type(DictIO.GetAlternative(kwargs, "material_type", "Solid"))
        self.sims.set_visualize(DictIO.GetAlternative(kwargs, "visualize", self.sims.visualize))
        self.sims.set_sparse_grid(DictIO.GetAlternative(kwargs, "sparse_grid", None))
        self.sims.set_particle_shifting(DictIO.GetAlternative(kwargs, "particle_shifting", False))
        self.sims.set_particle_shifting_scale(DictIO.GetAlternative(kwargs, "particle_shifting_scale", 1.0))
        self.sims.set_density_projection(
            DictIO.GetAlternative(kwargs, "density_projection", False),
            DictIO.GetAlternative(kwargs, "density_projection_tolerance", 0.01),
            DictIO.GetAlternative(kwargs, "density_projection_error_clamp", 0.01),
            DictIO.GetAlternative(kwargs, "density_projection_max_shift_ratio", 0.05),
            DictIO.GetAlternative(kwargs, "density_projection_interior_only", True),
        )
        self.sims.set_delayed_fluid_advection(DictIO.GetAlternative(kwargs, "delayed_fluid_advection", True))
        self.sims.set_solid_sdf_cut_cell(
            DictIO.GetAlternative(kwargs, "solid_sdf_cut_cell", False),
            DictIO.GetAlternative(kwargs, "solid_cut_cell_min_fraction", 0.01),
            DictIO.GetAlternative(kwargs, "fluid_wall_no_slip", False),
        )
        self.sims.set_stress_integration(DictIO.GetAlternative(kwargs, "stress_integration", "ReturnMapping"))
        self.sims.set_discretization(DictIO.GetAlternative(kwargs, "discretization", "FEM"))
        self.sims.set_THB(DictIO.GetAlternative(kwargs, "set_THB", False))
        self.sims.set_particle_traction_method(DictIO.GetAlternative(kwargs, "particle_traction_method", "Stable"))
        self.sims.set_particle_traction_update_area(
            DictIO.GetAlternative(kwargs, "particle_traction_update_area", True)
        )
        self.sims.set_AOSOA(DictIO.GetAlternative(kwargs, "AOSOA", False))
        self.sims.set_random_field(DictIO.GetAlternative(kwargs, "random_field", False))
        self.sims.set_drift_correct(DictIO.GetAlternative(kwargs, "drift_correct", True))
        self.sims.set_track_energy(DictIO.GetAlternative(kwargs, "track_energy", self.sims.energy_tracking))
        self.sims.validate_configuration(require_solver_parameters=False)
        if log:
            self.print_basic_simulation_info()
            print("\n")

    def set_implicit_solver_parameters(self, **implicit_parameters):
        if self.sims.solver_type != "Implicit":
            raise RuntimeError("KeyError:: /solver_type/ should be set as Implicit")

        linear_solver = DictIO.GetAlternative(implicit_parameters, "linear_solver", "PCG")
        self.sims.set_linear_solver_relative_tolerance(
            DictIO.GetAlternative(
                implicit_parameters,
                "linear_solver_relative_tolerance",
                self.sims.linear_solver_relative_tolerance,
            )
        )
        if self.sims.material_type == "Solid":
            self.sims.set_calculate_reaction_force(
                DictIO.GetAlternative(implicit_parameters, "calculate_reaction_force", False)
            )
            self.sims.set_integration_scheme(
                DictIO.GetAlternative(implicit_parameters, "integration_scheme", "Newmark")
            )
            self.sims.set_displacement_tolerance(
                DictIO.GetAlternative(implicit_parameters, "displacement_tolerance", 1e-4)
            )
            self.sims.set_residual_tolerance(DictIO.GetAlternative(implicit_parameters, "residual_tolerance", 1e-10))
            self.sims.set_symmetrize_matrix_free_tangent(
                DictIO.GetAlternative(implicit_parameters, "symmetrize_matrix_free_tangent", False)
            )
            self.sims.set_use_elastic_matrix_free_tangent(
                DictIO.GetAlternative(implicit_parameters, "use_elastic_matrix_free_tangent", False)
            )
            self.sims.set_quasi_static(DictIO.GetAlternative(implicit_parameters, "quasi_static", False))
            self.sims.set_newmark_parameter(
                DictIO.GetAlternative(implicit_parameters, "newmark_parameter", [0.5, 0.25])
            )
            self.sims.set_max_iteration(DictIO.GetAlternative(implicit_parameters, "max_iteration_number", 50))
            self.sims.set_assemble_type(DictIO.GetAlternative(implicit_parameters, "assemble_type", "MatrixFree"))
            self.sims.set_hash_triplet_matrix_symmetric(
                DictIO.GetAlternative(implicit_parameters, "matrix_symmetric", None)
            )
            # self.sims.set_rayleigh_damping(DictIO.GetAlternative(implicit_parameters, "rayleigh_damping", [0.2, 0.7]))
        else:
            self.sims.set_residual_tolerance(
                DictIO.GetAlternative(implicit_parameters, "residual_tolerance", self.sims.residual_tolerance)
            )
            self.sims.set_max_iteration(
                DictIO.GetAlternative(implicit_parameters, "max_iteration_number", self.sims.iter_max)
            )
        self.sims.set_linear_solver(linear_solver)
        if self.sims.linear_solver == "MGPCG":
            self.sims.set_multigrid_paramter(
                DictIO.GetAlternative(implicit_parameters, "multilevel", 4),
                DictIO.GetAlternative(implicit_parameters, "pre_and_post_smoothing", 2),
                DictIO.GetAlternative(implicit_parameters, "bottom_smoothing", 10),
            )
        self.sims.validate_configuration()

    def set_semi_implicit_solver_parameters(self, semi_implicit_parameters):
        if self.sims.solver_type not in ("SemiImplicit", "SemiImplicit_u_p"):
            raise RuntimeError("KeyError:: /solver_type/ should be set as SemiImplicit or SemiImplicit_u_p")
        self.sims.set_residual_tolerance(DictIO.GetAlternative(semi_implicit_parameters, "residual_tolerance", 1e-4))
        self.sims.set_max_iteration(DictIO.GetAlternative(semi_implicit_parameters, "max_iteration_number", 500))
        self.sims.set_linear_solver(DictIO.GetAlternative(semi_implicit_parameters, "linear_solver", "PCG"))
        self.sims.set_assemble_type(DictIO.GetAlternative(semi_implicit_parameters, "assemble_type", "MatrixFree"))
        self.sims.set_pressure_solver(DictIO.GetAlternative(semi_implicit_parameters, "pressure_solver", None))
        self.sims.set_pressure_stabilize_technique(
            DictIO.GetAlternative(semi_implicit_parameters, "pressure_stabilize", self.sims.pressure_stabilize)
        )
        default_pressure_beta = (
            0.0
            if self.sims.use_mgpcg_pressure_solver()
            and getattr(self.sims, "material_type", None) == "TwoPhaseSingleLayer"
            else 1.0
        )
        pressure_beta = DictIO.GetAlternative(
            semi_implicit_parameters,
            "pressure_beta",
            DictIO.GetAlternative(semi_implicit_parameters, "pressure_bate", default_pressure_beta),
        )
        self.sims.set_pressure_parameter(pressure_beta)
        if self.sims.use_mgpcg_pressure_solver():
            self.sims.set_multigrid_paramter(
                DictIO.GetAlternative(semi_implicit_parameters, "multilevel", 2),
                DictIO.GetAlternative(semi_implicit_parameters, "pre_and_post_smoothing", 2),
                DictIO.GetAlternative(semi_implicit_parameters, "bottom_smoothing", 10),
            )
        self.sims.validate_configuration()

    def set_fbar_parameters(self, **kwargs):
        self.sims.set_fbar_fraction(DictIO.GetAlternative(kwargs, "fbar_fraction", 0.99))
        self.sims.set_jacobian_clamp(DictIO.GetAlternative(kwargs, "jacobian_clamp", [0.1, 10]))

    def set_solver(self, solver, log=True):
        retry_policy = StepRetryPolicy(
            enabled=DictIO.GetAlternative(solver, "enable_step_retry", False),
            maximum_retries=DictIO.GetAlternative(solver, "step_retry_max_retries", 2),
            reduction=DictIO.GetAlternative(solver, "step_retry_reduction", 0.5),
            minimum_timestep=DictIO.GetAlternative(solver, "step_retry_minimum_timestep", 0.0),
        )
        if retry_policy.enabled and not self.sims.is_direct_backend():
            raise ValueError(
                "MPM step retry currently requires mpm_backend='Direct'; "
                "the native implicit backend does not yet expose a complete "
                "accepted-state rollback transaction"
            )
        if retry_policy.enabled and self.sims.solver_type != "Implicit":
            raise ValueError("MPM step retry is available only for implicit solvers")
        solver = dict(solver)
        solver.update(
            enable_step_retry=retry_policy.enabled,
            step_retry_max_retries=retry_policy.maximum_retries,
            step_retry_reduction=retry_policy.reduction,
            step_retry_minimum_timestep=retry_policy.minimum_timestep,
        )
        if self.sims.is_direct_backend():
            self._set_direct_solver(solver)
            if log:
                self.print_solver_info()
                print("\n")
            return
        self.sims.set_timestep(DictIO.GetEssential(solver, "Timestep"))
        self.sims.set_simulation_time(DictIO.GetEssential(solver, "SimulationTime"))
        self.sims.set_CFL(DictIO.GetAlternative(solver, "CFL", 0.5))
        self.sims.set_adaptive_timestep(DictIO.GetAlternative(solver, "AdaptiveStep", 0))
        self.sims.set_save_interval(DictIO.GetAlternative(solver, "SaveInterval", self.sims.time / 20.0))
        self.sims.set_save_path(DictIO.GetAlternative(solver, "SavePath", "OutputData"))
        if log:
            self.print_solver_info()
            print("\n")

    def memory_allocate(self, memory, log=True):
        if self.sims.is_direct_backend():
            self.sims.max_material_num = int(
                DictIO.GetAlternative(memory, "max_material_number", self.sims.max_material_num)
            )
            self.sims.max_particle_num = int(
                DictIO.GetAlternative(memory, "max_particle_number", self.sims.max_particle_num)
            )
            if log:
                self.print_simulation_info()
                self.print_memory_info()
                self.print_neighbor_search_info()
                print("\n")
            return
        self.sims.set_material_num(DictIO.GetAlternative(memory, "max_material_number", 0))
        self.sims.set_particle_num(DictIO.GetAlternative(memory, "max_particle_number", 0))
        self.sims.set_constraint_num(DictIO.GetAlternative(memory, "max_constraint_number", {}))
        self.sims.set_verlet_distance_multiplier(DictIO.GetAlternative(memory, "verlet_distance_multiplier", 0.0))
        if self.sims.solver_type == "Implicit":
            self.sims.set_dof_multiplier(DictIO.GetAlternative(memory, "dof_multiplier", 2))
        if log:
            self.print_simulation_info()
            self.print_memory_info()
            self.print_neighbor_search_info()
            print("\n")

    def print_basic_simulation_info(self):
        print_solver_section(
            "MPM",
            "Basic Configuration",
            [
                ("Simulation Type", runtime_architecture()),
                ("Simulation Domain", self.sims.domain),
                ("Boundary Condition", self.sims.boundary),
                ("Gravity", self.sims.gravity),
                ("MPM Backend", self.sims.mpm_backend),
                ("Solver Type", self.sims.solver_type),
            ],
        )

    def print_simulation_info(self):
        entries = [
            ("Background Damping", self.sims.background_damping),
            ("Alpha PIC", self.sims.alphaPIC),
            ("Stabilization Technique", self.sims.stabilize),
        ]
        if self.sims.gauss_number > 0:
            entries.append(("Gauss Number", self.sims.gauss_number))
        entries.extend(
            [
                ("Boundary Direction Detection", self.sims.boundary_direction_detection),
                ("Free Surface Detection", self.sims.free_surface_detection),
                ("Fluid Level Set", self.sims.fluid_level_set),
                ("Sparse Grid", self.sims.sparse_grid),
                ("MPM Backend", self.sims.mpm_backend),
            ]
        )
        if self.sims.is_direct_backend():
            entries.append(("IPC Contact", self.sims.ipc_contact))
        if self.sims.sparse_grid:
            entries.extend(
                [
                    ("Sparse Grid Backend", self.sims.sparse_grid_backend),
                    ("Sparse Block Size", self.sims.sparse_grid_block_size),
                ]
            )
        entries.extend(
            [
                ("Mapping Scheme", self.sims.mapping),
                ("Shape Function", self.sims.shape_function),
                ("Velocity Projection", self.sims.velocity_projection_scheme),
            ]
        )
        if self.sims.soft_particle:
            entries.append(("Soft Particle Mode", self.sims.soft_particle_backend))
        if self.sims.solver_type in ("Implicit", "SemiImplicit", "SemiImplicit_u_p"):
            entries.append(("Assembly Type", self.sims.assemble_type))
        if self.sims.solver_type in ("SemiImplicit", "SemiImplicit_u_p"):
            entries.append(("Pressure Solver", self.sims.pressure_solver))
        print_solver_section("MPM", "Engine Information", entries)

    def print_solver_info(self):
        entries = [
            ("Solver Type", self.sims.solver_type),
            ("MPM Backend", self.sims.mpm_backend),
        ]
        if self.sims.solver_type in ("Implicit", "SemiImplicit", "SemiImplicit_u_p"):
            entries.extend(
                [
                    ("Assembly Type", self.sims.assemble_type),
                    ("Linear Solver", self.sims.linear_solver),
                ]
            )
        entries.extend(
            [
                ("Initial Simulation Time", self.sims.current_time),
                ("Final Simulation Time", self.sims.current_time + self.sims.time),
                ("Time Step", self.sims.dt[None]),
                ("Adaptive Time Step", self.sims.adaptive_timestep),
                ("CFL", self.sims.CFL),
                ("Save Interval", self.sims.save_interval),
                ("Save Path", self.sims.path),
            ]
        )
        print_solver_section("MPM", "Solver Information", entries)

    def print_memory_info(self):
        entries = [
            ("Maximum Materials", self.sims.max_material_num),
            ("Maximum Bodies", self.sims.max_body_num),
            ("Maximum Particles", self.sims.max_particle_num),
            ("Maximum Coupling Particles", self.sims.max_coupling_particle_num),
            ("Velocity Constraints", self.sims.nvelocity),
            ("Reflection Constraints", self.sims.nreflection),
            ("Friction Constraints", self.sims.nfriction),
            ("Absorbing Constraints", self.sims.nabsorbing),
            ("Traction Constraints", self.sims.ntraction),
            ("Particle Traction Constraints", self.sims.nptraction),
            ("Displacement Constraints", self.sims.ndisplacement),
        ]
        if self.sims.soft_particle:
            entries.extend(
                (
                    ("Maximum Soft Bodies", self.sims.max_soft_body_num),
                    ("Maximum Soft Material Points", self.sims.max_material_point_num),
                    ("Maximum Soft Grid Nodes", self.sims.max_soft_grid_num),
                )
            )
        print_solver_section("MPM", "Memory Information", entries)

    def print_neighbor_search_info(self):
        enabled = bool(self.sims.neighbor_detection)
        if not enabled and not self.sims.ipc_contact and not self.sims.soft_particle:
            return
        usages = []
        if enabled:
            usages.append("Contact neighbor detection")
        if self.sims.ipc_contact:
            usages.append("IPC candidate search")
        if self.sims.soft_particle:
            usages.append("Soft-particle contact search")
        entries = [("Search Usage", ", ".join(usages))]
        if enabled:
            entries.extend(
                [
                    ("Neighbor Detection Enabled", True),
                    (
                        "Runtime Search Object",
                        type(self.neighbor).__name__ if self.neighbor is not None else None,
                    ),
                    (
                        "Verlet Distance Multiplier",
                        self.sims.verlet_distance_multiplier,
                    ),
                    ("Verlet Distance", self.sims.verlet_distance),
                ]
            )
        if self.sims.ipc_contact:
            entries.append(("IPC Contact", True))
        if self.sims.soft_particle:
            entries.append(("Soft-particle Contact", True))
        print_solver_section(
            "MPM",
            "Neighbor Search Information",
            entries,
        )

    def add_contact(self, contact_type, **contact_phys):
        contact_key = str(contact_type).strip().replace("_", "").replace("-", "").replace(" ", "").lower()
        if contact_key in ("ipc", "ipccontact", "barrier", "barriercontact", "barrieripc", "semi", "semiipc"):
            if bool(contact_phys.pop("self_contact", False)):
                raise NotImplementedError("ordinary Direct MPM self-contact IPC is intentionally unsupported")
            contact_phys.setdefault(
                "ipc_model",
                "SemiIPC" if contact_key in ("semi", "semiipc") else "BarrierIPC",
            )
            self.sims.set_ipc_contact(True, contact_phys)
            self.direct_solver.update(contact_phys)
            return
        self.sims.set_contact_detection(contact_type)
        self.scene.activate_contact(self.sims, contact_phys)

    def add_material(self, model=None, material=None, **kwargs):
        if self.sims.is_direct_backend():
            self._add_direct_material(model, material, **kwargs)
            return
        self.scene.activate_material(self.sims, model, material)

    def add_element(self, element):
        if self.sims.is_direct_backend():
            self._add_direct_element(element)
            return
        self.scene.activate_element(self.sims, element)
        self.scene.activate_particle(self.sims)

    def add_region(self, region):
        if type(region) is dict:
            self.generator.add_my_region(self.sims.dimension, self.sims.domain, region)
        elif type(region) is list:
            for region_dict in region:
                self.generator.add_my_region(self.sims.dimension, self.sims.domain, region_dict)

    def add_body(self, body):
        if self.sims.is_direct_backend():
            self._add_direct_body(body)
            return
        self.scene.check_materials(self.sims)
        self.generator.add_body(body, self.sims, self.scene)

    def add_ground(self, ground):
        if not self.sims.is_direct_backend():
            raise RuntimeError("MPM.add_ground is only available for mpm_backend='Direct'")
        self.direct_ground = ground

    def create_body(self):
        if not self.sims.is_direct_backend():
            raise RuntimeError("MPM.create_body is only available for mpm_backend='Direct'")
        from src.mpm.generator.Body import Body

        return Body()

    def create_ground(self):
        if not self.sims.is_direct_backend():
            raise RuntimeError("MPM.create_ground is only available for mpm_backend='Direct'")
        from src.mpm.generator.Ground import Ground

        return Ground()

    def add_body_from_file(self, body):
        self.scene.check_materials(self.sims)
        self.generator.read_body_file(body, self.sims, self.scene)

    def add_polygons(self, body):
        self.generator.add_polygons(body, self.sims, self.scene)

    def read_restart(self, file_number, file_path, is_continue=True):
        self.sims.set_is_continue(is_continue)
        if self.sims.is_continue:
            self.sims.current_print = file_number
        self.add_body_from_file(
            body={
                "FileType": "NPZ",
                "Template": {"Restart": True, "File": file_path + f"/particles/MPMParticle{file_number:06d}.npz"},
            }
        )

    def add_boundary_condition(self, boundary=None, dirichlet=None, neumann=None):
        if self.sims.is_direct_backend():
            if dirichlet is not None:
                self.direct_dirichlet = dirichlet
            if neumann is not None:
                self.direct_neumann = neumann
            return
        self.scene.boundary.get_essentials(self.scene.is_rigid, self.scene.psize, self.generator.myRegion)
        if type(boundary) is list or type(boundary) is dict:
            self.scene.boundary.iterate_boundary_constraint(self.sims, self.scene.element, boundary, 0)
        elif type(boundary) is str:
            if boundary is None:
                boundary = "OutputData/boundary_conditions.txt"
            self.scene.boundary.read_boundary_constraint(self.sims, boundary)

    def add_particle_traction(self, traction):
        self.scene.boundary.iterate_particle_boundary_conditions(
            self.sims, traction, self.scene.particleNum[0], self.scene.particle, self.scene.psize
        )

    def add_virtual_stress_field(self, field, region_name=None, function=None):
        if not region_name is None:
            region: RegionFunction = self.generator.get_region_ptr(region_name)
            function = region.function
        sparse_node_capacity = None
        if self.scene.sparse_grid is not None:
            sparse_node_capacity = self.scene.sparse_grid.node_capacity
        self.scene.boundary.set_virtual_stress_field(
            self.sims, self.scene.element, field, function, sparse_node_capacity
        )

    def clean_boundary_condition(self, boundary):
        if type(boundary) is list or type(boundary) is dict:
            self.scene.boundary.iterate_boundary_constraint(self.sims, self.scene.element, boundary, 1)

    def write_boundary_condition(self, output_path="OutputData"):
        self.scene.boundary.write_boundary_constraint(output_path)

    def select_save_data(self, particle=True, grid=False, object=True):
        if self.scene.contact is None or self.scene.contact.polygon_vertices is None:
            object = False
        self.sims.set_save_data(particle, grid, object)

    def choose_coupling_region(self, region_name=None, function=None):
        if not region_name is None:
            region: RegionFunction = self.generator.get_region_ptr(region_name)
            self.scene.choose_coupling_region(self.sims, region.function)
        elif not function is None:
            self.scene.choose_coupling_region(self.sims, ti.pyfunc(function))
        self.scene.filter_particles(self.sims)

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
            if "background_damping" in kwargs:
                self.sims.set_background_damping(DictIO.GetEssential(kwargs, "background_damping"))
            if "alphaPIC" in kwargs:
                self.sims.set_alpha(DictIO.GetEssential(kwargs, "alphaPIC"))

    def _set_direct_solver(self, solver):
        self.direct_solver.update(solver)
        plane_strain = DictIO.GetOptional(solver, "plane_strain")
        if plane_strain is not None:
            if not isinstance(plane_strain, (bool, np.bool_)):
                raise TypeError("Direct MPM plane_strain must be a boolean")
            self.direct_solver["plane_strain"] = bool(plane_strain)
        domain = DictIO.GetOptional(solver, "domain")
        if domain is not None:
            self.sims.set_domain(domain)
        gravity = DictIO.GetOptional(solver, "gravity")
        if gravity is not None:
            self.sims.set_gravity(gravity)
        timestep = DictIO.GetOptional(solver, "Timestep")
        if timestep is None:
            timestep = DictIO.GetOptional(solver, "dt")
        if timestep is not None:
            self.sims.set_timestep(timestep)
        simulation_time = DictIO.GetOptional(solver, "SimulationTime")
        step = DictIO.GetOptional(solver, "step")
        if simulation_time is not None:
            self.sims.set_simulation_time(simulation_time)
        elif step is not None and timestep is not None:
            self.sims.set_simulation_time(float(step) * float(timestep))
        output_interval = DictIO.GetOptional(solver, "OutputInterval")
        if output_interval is None:
            output_interval = DictIO.GetOptional(solver, "interval")
        if output_interval is not None:
            self.direct_solver["interval"] = max(1, int(output_interval))
        save_interval = DictIO.GetOptional(solver, "SaveInterval")
        if save_interval is not None:
            self.sims.set_save_interval(save_interval)
            if output_interval is None and timestep is not None:
                self.direct_solver["interval"] = max(1, int(np.ceil(float(save_interval) / float(timestep))))
        save_path = DictIO.GetOptional(solver, "SavePath")
        if save_path is None:
            save_path = DictIO.GetOptional(solver, "path")
        if save_path is not None:
            self.sims.set_save_path(save_path)

    def _add_direct_material(self, model=None, material=None, **kwargs):
        parameters = {}
        if isinstance(material, dict):
            parameters.update(material)
        elif isinstance(material, str):
            parameters["material"] = self._direct_material_name(material)
        elif material is not None:
            parameters["material_parameters"] = material
        parameters.update(kwargs)
        if model is not None:
            parameters["material"] = self._direct_material_name(model)
        elif "model" in parameters:
            parameters["material"] = self._direct_material_name(parameters.pop("model"))
        elif "MaterialModel" in parameters:
            parameters["material"] = self._direct_material_name(parameters.pop("MaterialModel"))
        elif "ConstitutiveModel" in parameters:
            parameters["material"] = self._direct_material_name(parameters.pop("ConstitutiveModel"))

        aliases = {
            "Density": "density",
            "YoungModulus": "young_modulus",
            "ElasticModulus": "young_modulus",
            "PoissonRatio": "poisson_ratio",
        }
        for old, new in aliases.items():
            value = DictIO.GetOptional(parameters, old)
            if value is not None and new not in parameters:
                parameters[new] = value
        self.direct_material.update(parameters)

    def _direct_material_name(self, model):
        key = str(model).replace("-", "").replace("_", "").replace(" ", "").lower()
        aliases = {
            "linearelastic": "LinearElastic",
            "neo": "NeoHookean",
            "neohookean": "NeoHookean",
            "neohookeanmodel": "NeoHookean",
            "dp": "DruckerPrager",
            "druckerprager": "DruckerPrager",
            "druckerpragermodel": "DruckerPrager",
            "finitedruckerprager": "DruckerPrager",
            "statedependentdruckerprager": "StateDependentDruckerPrager",
            "statedependentdruckerpragermodel": "StateDependentDruckerPrager",
            "vonmises": "VonMises",
            "vonmisesmodel": "VonMises",
            "finitevonmises": "VonMises",
            "j2": "VonMises",
            "j2plasticity": "VonMises",
            "mcc": "ModifiedCamClay",
            "camclay": "ModifiedCamClay",
            "modifiedcamclay": "ModifiedCamClay",
            "modifiedcamclaymodel": "ModifiedCamClay",
            "finitemodifiedcamclay": "ModifiedCamClay",
            "finitemodifiedcamclaymodel": "ModifiedCamClay",
        }
        return aliases.get(key, str(model))

    def _normalize_direct_material_for_solver(self, material):
        key = str(material).replace("-", "").replace("_", "").replace(" ", "").lower()
        if self.sims.solver_type == "Implicit":
            aliases = {
                "linearelastic": "linearElastic",
                "neohookean": "neoHookean",
                "neohookeanmodel": "neoHookean",
                "dp": "druckerPrager",
                "druckerprager": "druckerPrager",
                "druckerpragermodel": "druckerPrager",
                "finitedruckerprager": "druckerPrager",
                "statedependentdruckerprager": "StateDependentDruckerPrager",
                "statedependentdruckerpragermodel": "StateDependentDruckerPrager",
                "vonmises": "vonMises",
                "vonmisesmodel": "vonMises",
                "finitevonmises": "vonMises",
                "j2": "vonMises",
                "j2plasticity": "vonMises",
                "mcc": "modifiedCamClay",
                "camclay": "modifiedCamClay",
                "modifiedcamclay": "modifiedCamClay",
                "modifiedcamclaymodel": "modifiedCamClay",
                "finitemodifiedcamclay": "modifiedCamClay",
                "finitemodifiedcamclaymodel": "modifiedCamClay",
            }
        else:
            aliases = {
                "linearelastic": "LinearElastic",
                "neohookean": "NeoHookean",
                "neohookeanmodel": "NeoHookean",
            }
        return aliases.get(key, material)

    def _add_direct_element(self, element):
        self.direct_element.update(element)
        dx = DictIO.GetOptional(element, "ElementSize")
        if dx is None:
            dx = DictIO.GetOptional(element, "GridSize")
        if dx is None:
            dx = DictIO.GetOptional(element, "dx")
        if isinstance(dx, (list, tuple, np.ndarray)):
            dx = dx[0]
        if dx is not None:
            self.direct_solver["dx"] = float(dx)
        shape_function = DictIO.GetOptional(element, "ShapeFunction")
        if shape_function is not None:
            self.direct_solver["shape_function"] = self._direct_shape_function(shape_function)

    def _add_direct_body(self, body):
        if hasattr(body, "bodies") and hasattr(body, "body_counter"):
            body.body_counter = max(int(body.body_counter), len(body.bodies))
            self.direct_bodies = body
            return
        raise RuntimeError("mpm_backend='Direct' expects a Body created by mpm.create_body().")

    def _direct_shape_function(self, shape_function):
        key = str(shape_function).replace("-", "").replace("_", "").replace(" ", "").lower()
        aliases = {
            "linear": "linear",
            "smoothlinear": "linear",
            "gimp": "gimp",
            "quadbspline": "bspline",
            "quadraticbspline": "bspline",
            "bspline": "bspline",
        }
        return aliases.get(key, str(shape_function).lower())

    def _direct_domain(self):
        if "domain" in self.direct_solver:
            domain = self.direct_solver["domain"]
        else:
            domain = self.sims.domain
        domain = np.asarray(domain, dtype=float).reshape(-1)
        if domain.size < self.sims.dimension or np.linalg.norm(domain[: self.sims.dimension]) < 1e-14:
            raise RuntimeError("mpm_backend='Direct' requires domain in set_configuration(...) or set_solver(...).")
        return domain[: self.sims.dimension].tolist()

    def _direct_gravity(self):
        gravity = np.asarray(self.sims.gravity, dtype=float).reshape(-1)
        if gravity.size < self.sims.dimension:
            gravity = np.pad(gravity, (0, self.sims.dimension - gravity.size), mode="constant")
        return gravity[: self.sims.dimension].tolist()

    def _build_direct_engine(self):
        if self.direct_bodies is None:
            raise RuntimeError("mpm_backend='Direct' requires add_body(mpm.create_body()).")
        if self.sims.ipc_contact and len(self.direct_bodies.bodies) > 1:
            raise NotImplementedError(
                "ordinary Direct MPM self/multibody IPC contact is intentionally unsupported; "
                "use ground or MPM--FEM/IGA/ABD cross-contact"
            )

        import src.mpm.config as direct_config

        direct_config.set_dimension(self.sims.dimension)
        kwargs = dict(self.direct_solver)
        kwargs.update(self.direct_material)
        if "material" in kwargs:
            kwargs["material"] = self._normalize_direct_material_for_solver(kwargs["material"])
        kwargs.setdefault("domain", self._direct_domain())
        kwargs.setdefault("gravity", self._direct_gravity())
        kwargs.setdefault("dt", float(self.sims.dt[None]) if self.sims.dt[None] > 0.0 else 1.0e-2)
        if "step" not in kwargs:
            if self.sims.time > 0.0 and kwargs["dt"] > 0.0:
                kwargs["step"] = max(1, int(np.ceil(self.sims.time / kwargs["dt"])))
            else:
                kwargs["step"] = 1
        kwargs.setdefault("interval", max(1, int(self.sims.save_interval)))
        kwargs.setdefault("path", self.sims.path or "OutputData")
        kwargs.setdefault("shape_function", self._direct_shape_function(self.sims.shape_function))
        kwargs.setdefault("visualize", self.sims.visualize)
        kwargs.setdefault("damping", self.sims.background_damping)
        kwargs.setdefault("alphaPIC", self.sims.alphaPIC)
        kwargs.setdefault("velocity_projection", self.sims.velocity_projection_scheme == "Affine")
        kwargs.setdefault("track_energy", self.sims.energy_tracking)
        kwargs.setdefault("residual", self.sims.residual_tolerance)
        kwargs.setdefault("max_iters", self.sims.iter_max)
        kwargs.setdefault("newmark", [0.5, self.sims.newmark_beta, self.sims.newmark_gamma])
        kwargs.setdefault("is_2DAxisy", self.sims.is_2DAxisy)
        kwargs.setdefault("axis_offset", self.sims.axis_offset)

        if self.sims.direct_static_twophase:
            from src.mpm.engines.direct.StaticTwoPhaseULMPM import (
                StaticTwoPhaseULMPM,
            )

            engine_cls = StaticTwoPhaseULMPM
        elif self.sims.ipc_contact:
            from src.mpm.generator.Ground import Ground

            if self.sims.configuration == "TLMPM":
                from src.mpm.soft_particle.IPCTLMPM import IPCTLMPM

                engine_cls = IPCTLMPM
            else:
                from src.mpm.soft_particle.IPCULMPM import IPCULMPM

                engine_cls = IPCULMPM
            if self.direct_ground is None:
                self.direct_ground = Ground()
        elif self.sims.solver_type == "Explicit":
            if self.sims.configuration == "TLMPM":
                from src.mpm.engines.direct.ExplicitTLMPM import ExplicitTLMPM

                engine_cls = ExplicitTLMPM
            else:
                from src.mpm.engines.direct.ExplicitULMPM import ExplicitULMPM

                engine_cls = ExplicitULMPM
        elif self.sims.solver_type == "Implicit":
            if self.sims.configuration == "TLMPM":
                from src.mpm.engines.direct.ImplicitTLMPM import ImplicitTLMPM

                engine_cls = ImplicitTLMPM
            else:
                from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

                engine_cls = ImplicitULMPM
        else:
            raise RuntimeError("mpm_backend='Direct' supports Explicit or Implicit solver_type.")

        if self.sims.ipc_contact:
            return engine_cls(
                self.direct_bodies, self.direct_ground, self.direct_dirichlet, self.direct_neumann, **kwargs
            )
        return engine_cls(self.direct_bodies, self.direct_dirichlet, self.direct_neumann, **kwargs)

    def add_spatial_grid(self):
        if self.sims.coupling == "Lagrangian" or self.sims.neighbor_detection:
            if self.neighbor is None:
                self.neighbor = SpatialHashGrid(self.sims)
            self.neighbor.neighbor_initialze(self.scene)

    def add_engine(self):
        self.sims.validate_configuration()
        if self.sims.is_direct_backend():
            if self.enginer is None:
                self.enginer = self._build_direct_engine()
                self.enginer.timer = self.sims.timer
            return
        if self.sims.soft_particle:
            raise RuntimeError(
                "MPM soft_particle mode has its TLMPM kernels under "
                "src.mpm.soft_particle, but the level-set body/soft-point "
                "scene ownership is not migrated from DEM yet. Use DEM "
                "scheme='LSMPM' for the current runnable path, or complete "
                "the DEMPM level-set soft-particle bridge before calling "
                "MPM.run()."
            )
        if self.enginer is None:
            if self.sims.configuration == "ULMPM":
                if self.sims.solver_type == "Explicit":
                    if self.sims.material_type == "TwoPhaseSingleLayer":
                        self.enginer = ULExplicitTwoPhaseEngine(self.sims)
                    else:
                        self.enginer = ULExplicitEngine(self.sims)
                elif self.sims.solver_type == "Implicit":
                    if self.sims.material_type == "Solid":
                        self.enginer = ImplicitEngine(self.sims)
                    elif self.sims.material_type == "Fluid":
                        self.enginer = IncompressibleEngine(self.sims)
                elif self.sims.solver_type == "SemiImplicit_u_p":
                    if self.sims.material_type == "TwoPhaseSingleLayer":
                        self.enginer = ULSemiImplicitTwoPhaseEngine_u_p(self.sims)
                elif self.sims.solver_type == "SemiImplicit":
                    if self.sims.material_type == "TwoPhaseSingleLayer":
                        self.enginer = ULSemiImplicitTwoPhaseEngine(self.sims)
                    elif self.sims.material_type == "TwoPhaseDoubleLayer":
                        self.enginer = ULSemiImplicitTwoPhaseDoubleLayerEngine(self.sims)
                    else:
                        raise RuntimeError(
                            "Keyword:: /material_type/ should be set as $TwoPhaseSingleLayer$ or $TwoPhaseDoubleLayer$"
                        )
            elif self.sims.configuration == "TLMPM":
                if self.sims.solver_type == "Explicit":
                    self.enginer = TLExplicitEngine(self.sims)
                else:
                    raise RuntimeError("Total lagrangian material point method only have explicit version currently")
        self.enginer.choose_engine(self.sims)
        self.enginer.choose_boundary_constraints(self.sims, self.scene)
        self.enginer.valid_contact(self.sims, self.scene)

    def add_recorder(self):
        if self.recorder is None:
            self.recorder = WriteFile(self.sims)

    def add_solver(self, **kwargs):
        if self.solver is None:
            self.solver = Solver(self.sims, self.generator, self.enginer, self.recorder)
        if DictIO.GetAlternative(kwargs, "reset_function", True):
            self.solver.postprocess = []
        self.solver.set_callback_function(DictIO.GetAlternative(kwargs, "function", None))

    def add_postfunctions(self, **functions):
        self.solver.set_callback_function(functions)

    def differentiable(self, steps=None):
        """Create a fixed-step tape for Direct ULMPM BarrierIPC."""
        if not self.sims.is_direct_backend():
            raise RuntimeError("MPM.differentiable requires mpm_backend='Direct'")
        self.add_essentials()
        from src.mpm.soft_particle.IPCULMPM import IPCULMPM

        if not isinstance(self.enginer, IPCULMPM):
            raise RuntimeError("MPM.differentiable requires implicit ULMPM with BarrierIPC contact")
        if self.first_run:
            self.enginer.initial_simulation()
            self.first_run = False
        from src.mpm.soft_particle.DifferentiableMPM import DifferentiableMPM

        return DifferentiableMPM(self.enginer, steps=steps)

    def set_window(self, window):
        self.sims.set_window_parameters(window)

    def add_essentials(self, **kwargs):
        if self.sims.is_direct_backend():
            self.add_engine()
            return
        self.scene.set_initial_gravity_field(self.sims, DictIO.GetAlternative(kwargs, "gravity_field", False))
        self.scene.boundary.copy_dict_to_field(self.sims)
        self.add_spatial_grid()
        self.add_engine()
        fluid_level_set_function = DictIO.GetAlternative(kwargs, "fluid_level_set_function", None)
        if fluid_level_set_function is not None and hasattr(self.enginer, "update_external_fluid_level_set"):
            self.enginer.update_external_fluid_level_set = fluid_level_set_function
        cut_cell_function = DictIO.GetAlternative(
            kwargs, "cut_cell_function", DictIO.GetAlternative(kwargs, "solid_sdf_function", None)
        )
        if cut_cell_function is not None and hasattr(self.enginer, "update_external_cut_cell_boundary"):
            if not self.sims.solid_sdf_cut_cell:
                raise RuntimeError(
                    "cut_cell_function requires solid_sdf_cut_cell=True; "
                    "use ibm_field_function for an immersed SDF rigid body"
                )
            self.enginer.update_external_cut_cell_boundary = cut_cell_function
        mac_boundary_function = DictIO.GetAlternative(kwargs, "mac_boundary_function", None)
        if mac_boundary_function is not None and hasattr(self.enginer, "update_external_mac_boundary"):
            self.enginer.update_external_mac_boundary = mac_boundary_function
        ibm_field_function = DictIO.GetAlternative(
            kwargs,
            "ibm_field_function",
            DictIO.GetAlternative(kwargs, "ibm_source_function", DictIO.GetAlternative(kwargs, "ibm_function", None)),
        )
        if ibm_field_function is not None and hasattr(self.enginer, "update_external_ibm_fields"):
            self.enginer.update_external_ibm_fields = ibm_field_function
        self.add_recorder()
        if self.sims.coupling is False:
            self.add_solver(**kwargs)
        self.scene.calc_mass_cutoff(self.sims)
        if self.first_run:
            self.scene.boundary.set_boundary(self.sims)
        if self.sims.mode == "Normal":
            self.scene.boundary.set_boundary_types(self.sims, self.scene.element)
        if self.sims.is_continue:
            if self.sims.coupling is False:
                self.solver.last_save_time = 1.0 * self.sims.current_time
                self.sims.current_print += 1
            self.sims.set_is_continue(False)

    def run(self, visualize=False, **kwargs):
        self.add_essentials(**kwargs)
        if self.sims.is_direct_backend():
            print_simulation_start("MPM")
            self.enginer.run(
                verbose=DictIO.GetAlternative(kwargs, "verbose", True),
                postprocessing=DictIO.GetAlternative(kwargs, "postprocessing", []),
            )
            self.first_run = False
            return
        self.check_critical_timestep()
        if visualize is False:
            self.solver.Solver(self.scene, self.neighbor)
        else:
            self.sims.set_visualize_interval(
                DictIO.GetAlternative(
                    kwargs,
                    "visualize_interval",
                    self.sims.visualize_interval,
                )
            )
            self.sims.set_window_size(DictIO.GetAlternative(kwargs, "WindowSize", self.sims.window_size))
            self.solver.Visualize(self.scene, self.neighbor)
        self.first_run = False

    def check_critical_timestep(self):
        if self.sims.solver_type == "Explicit":
            print("#", " Check Timestep ... ...".ljust(67))
            critical_timestep = self.scene.get_critical_timestep()
            if self.sims.CFL * critical_timestep < self.sims.dt[None]:
                self.sims.update_critical_timestep(self.sims.CFL * critical_timestep)
            else:
                print("The prescribed time step is sufficiently small\n")

    def update_particle_properties(
        self, property_name, value, override=True, bodyID=None, region_name=None, function=None
    ):
        if not bodyID is None:
            self.scene.update_particle_properties(self.sims, override, property_name, value, bodyID)
        elif not region_name is None:
            region: RegionFunction = self.generator.get_region_ptr(region_name)
            self.scene.update_particle_properties_in_region(self.sims, override, property_name, value, region.function)
        elif not function is None:
            self.scene.update_particle_properties_in_region(
                self.sims, override, property_name, value, ti.pyfunc(function)
            )

    def delete_particles(self, bodyID=None, region_name=None, function=None):
        if not bodyID is None:
            self.scene.delete_particles(bodyID)
        elif not region_name is None:
            region: RegionFunction = self.generator.get_region_ptr(region_name)
            self.scene.delete_particles_in_region(region.function)
        elif not function is None:
            self.scene.delete_particles_in_region(ti.pyfunc(function))

    def save_data(self):
        if self.solver is None:
            self.add_essentials()
        self.solver.save_file(self.scene)

    def postprocessing(self, start_file=0, end_file=-1, read_path=None, write_path=None, **kwargs):
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

        write_vtk_file(self.sims, start_file, end_file, read_path, write_path, kwargs)
