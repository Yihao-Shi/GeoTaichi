"""Public FEM--DEM/LSDEM/AffineBody coupling facade."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.fem.contact.ContactTopology import build_contact_surface
from src.fem.mainFEM import FEM
from src.fedem.ContactManager import ContactManager
from src.fedem.Checkpoint import load_checkpoint, save_checkpoint
from src.fedem.Engine import Engine
from src.fedem.FEDEMBase import Solver
from src.fedem.Patch import FEMSurfacePatch
from src.fedem.Recorder import WriteFile
from src.fedem.Simulation import FEDEMSimulation
from src.fedem.contact import AffineIPCModel
from src.utils.ObjectIO import DictIO
from src.utils.SolverConsole import (
    print_save_file_info,
    print_simulation_start,
    print_solver_section,
    runtime_architecture,
)
from src.utils.SolverDiagnostics import SolverDiagnosticsMixin
from src.utils.StepRetry import StepRetryPolicy


class FEDEM(SolverDiagnosticsMixin):
    """Two-way contact between DEM bodies and a deforming FEM surface."""

    def __init__(self, dem, fem: FEM, title="FEM & DEM Coupling Engine", log=True):
        self.log = bool(log)
        if log:
            print("# =================================================================== #")
            print("#", "GeoTaichi -- FEM & DEM Coupling Engine".center(67), "#")
            print("#", str(title).center(67), "#")
            print("# =================================================================== #", "\n")
        self.dem = dem
        self.fem = fem
        self.dem.sims.set_dem_coupling(True)
        self.sims = FEDEMSimulation()
        self.sims.current_time = float(self.dem.sims.current_time)
        self.sims.current_step = int(self.dem.sims.current_step)
        self.sims.current_print = int(self.dem.sims.current_print)
        self.contactor = ContactManager(self.sims)
        self.surface_faces = None
        self.surface_body = None
        self.surface_modifier = None
        self.patch = None
        self.enginer = None
        self.solver = None
        self.recorder = None
        self._memory = None
        self.solver_parameters = {}
        self.postprocess = []
        self._last_implicit_save_time = None
        self._affine_velocity_boundaries = []
        self._affine_pressure_servos = []
        # Work extracted when a loaded penalty boundary is removed.  This is
        # a physical topology-operation ledger, not a correction to the
        # contact forces or time integrator.
        self.removed_wall_contact_energy = 0.0

    def prescribe_affine_body_velocity(self, body_id, velocity):
        """Lock an AffineBody platen to a prescribed translational velocity."""
        specification = (int(body_id), tuple(float(value) for value in velocity))
        self._affine_velocity_boundaries.append(specification)
        if self.enginer is not None:
            self.enginer.prescribe_affine_body_velocity(*specification)

    def add_affine_body_pressure_servo(self, body_id, inward_normal, area, target_pressure, **kwargs):
        """Control an implicit AffineBody platen toward a target contact pressure."""
        specification = {
            "body_id": int(body_id),
            "inward_normal": tuple(float(value) for value in inward_normal),
            "area": float(area),
            "target_pressure": float(target_pressure),
            **kwargs,
        }
        self._affine_pressure_servos.append(specification)
        if self.enginer is not None:
            self.enginer.add_affine_body_pressure_servo(**specification)

    def set_configuration(self, log=True, **kwargs):
        domain = DictIO.GetAlternative(kwargs, "domain", None)
        if domain is None:
            domain = np.asarray(self.dem.sims.get_simulation_domain(), dtype=float)
        self.sims.set_domain(domain)
        search = str(DictIO.GetAlternative(kwargs, "search", "LinkedCell"))
        normalized = search.replace("_", "").replace("-", "").lower()
        if normalized == "linkedcell":
            self.sims.search = "LinkedCell"
        elif normalized in ("bvh", "lbvh"):
            self.sims.search = "BVH"
        else:
            self.sims.search = search
        contact_work_mode = DictIO.GetAlternative(
            kwargs,
            "contact_work_mode",
            DictIO.GetAlternative(kwargs, "ContactWorkMode", "Explicit"),
        )
        self.sims.set_contact_work_mode(contact_work_mode)
        gravity = DictIO.GetAlternative(kwargs, "gravity", None)
        if gravity is not None:
            self.dem.sims.set_gravity(gravity)
            self.fem.solver_kwargs["gravity"] = gravity
            if self.fem.engine is not None:
                fem_gravity = np.asarray(gravity, dtype=float).reshape(-1)
                if fem_gravity.size == 2:
                    fem_gravity = np.append(fem_gravity, 0.0)
                if fem_gravity.size != 3:
                    raise ValueError("FEDEM gravity must contain two or three components")
                self.fem.engine.gravity = fem_gravity
        if log:
            self.print_basic_simulation_info()
            print()

    def print_basic_simulation_info(self):
        print_solver_section(
            "FEDEM",
            "Basic Configuration",
            [
                ("Simulation Type", runtime_architecture()),
                ("Simulation Domain", self.sims.domain),
                ("Neighbor Search Type", self.sims.search),
                ("DEM Scheme", self.dem.sims.scheme),
                ("FEM Solver Type", self.fem.sims.solver_type),
                ("Contact Work Mode", self.sims.contact_work_mode),
                ("Gravity", self.dem.sims.gravity),
            ],
        )

    def set_solver(self, solver=None, log=True, **kwargs):
        parameters = {} if solver is None else dict(solver)
        parameters.update(kwargs)
        retry_policy = StepRetryPolicy(
            enabled=DictIO.GetAlternative(parameters, "enable_step_retry", False),
            maximum_retries=DictIO.GetAlternative(parameters, "step_retry_max_retries", 2),
            reduction=DictIO.GetAlternative(parameters, "step_retry_reduction", 0.5),
            minimum_timestep=DictIO.GetAlternative(parameters, "step_retry_minimum_timestep", 0.0),
        )
        parameters.update(
            enable_step_retry=retry_policy.enabled,
            step_retry_max_retries=retry_policy.maximum_retries,
            step_retry_reduction=retry_policy.reduction,
            step_retry_minimum_timestep=retry_policy.minimum_timestep,
        )
        self.solver_parameters.update(parameters)
        timestep = DictIO.GetAlternative(parameters, "Timestep", DictIO.GetAlternative(parameters, "dt", None))
        simulation_time = DictIO.GetAlternative(
            parameters,
            "SimulationTime",
            DictIO.GetAlternative(parameters, "simulation_time", None),
        )
        save_interval = DictIO.GetAlternative(
            parameters,
            "SaveInterval",
            DictIO.GetAlternative(parameters, "output_interval", 1.0e6),
        )
        if timestep is None or simulation_time is None:
            raise KeyError("FEDEM.set_solver requires Timestep and SimulationTime")
        self.sims.set_timestep(timestep)
        self.sims.set_simulation_time(simulation_time)
        self.sims.cfl = float(DictIO.GetAlternative(parameters, "CFL", 0.5))
        self.sims.set_save_interval(save_interval)
        self.sims.set_save_path(
            DictIO.GetAlternative(
                parameters,
                "SavePath",
                DictIO.GetAlternative(parameters, "path", "OutputData"),
            )
        )
        output_steps = max(1, int(math.ceil(save_interval / float(timestep))))
        self.sims.set_runtime_options(parameters, output_steps)
        dem_solver = {
            "Timestep": timestep,
            "SimulationTime": simulation_time,
            "SaveInterval": save_interval,
            "SavePath": self.sims.path,
            "CFL": self.sims.cfl,
        }
        self.dem.set_solver(dem_solver, log=False)
        fem_parameters = dict(parameters)
        fem_parameters.update(
            {
                "dt": timestep,
                "simulation_time": simulation_time,
                "output_interval": output_steps,
                "path": self.sims.path,
                "cfl": self.sims.cfl,
            }
        )
        self.fem.set_solver(fem_parameters, log=False)
        if log:
            self.print_solver_info()
            print()

    def print_solver_info(self):
        print_solver_section(
            "FEDEM",
            "Solver Information",
            [
                ("Simulation Time", self.sims.time),
                ("Time Step", self.sims.delta),
                ("Save Interval", self.sims.save_interval),
                ("Save Path", self.sims.path),
                ("Assembly Type", self.solver_parameters.get("assemble_type")),
                ("Linear Solver", self.solver_parameters.get("linear_solver")),
            ],
        )

    def _default_surface(self):
        if self.fem.scene.mesh is None:
            raise RuntimeError("add the FEM mesh before selecting its coupled surface")
        surface = build_contact_surface(self.fem.scene.mesh)
        return surface.faces, surface.face_body

    def add_surface(self, facet_sets=None, modifier=None, body_ids=None):
        if facet_sets is None:
            faces, face_body = self._default_surface()
            if body_ids is not None:
                selected = np.isin(face_body, np.asarray(body_ids, dtype=np.int32))
                faces, face_body = faces[selected], face_body[selected]
        else:
            if isinstance(facet_sets, dict):
                values = [np.asarray(value, dtype=np.int32).reshape(-1, 3) for value in facet_sets.values()]
                faces = np.concatenate(values, axis=0) if values else np.empty((0, 3), dtype=np.int32)
            else:
                faces = np.asarray(facet_sets, dtype=np.int32).reshape(-1, 3)
            node_body = self.fem.scene.mesh.node_body_ids
            face_body = node_body[faces[:, 0]]
            if np.any(node_body[faces] != face_body[:, None]):
                raise ValueError("each FEDEM surface facet must belong to one FEM body")
            if body_ids is not None:
                selected = np.isin(face_body, np.asarray(body_ids, dtype=np.int32))
                faces, face_body = faces[selected], face_body[selected]
        if faces.shape[0] == 0:
            raise ValueError("FEDEM coupled surface contains no triangles")
        self.surface_faces = np.ascontiguousarray(faces, dtype=np.int32)
        self.surface_body = np.ascontiguousarray(face_body, dtype=np.int32)
        self.surface_modifier = modifier
        self.patch = None
        return self.surface_faces

    add_patch = add_surface

    def memory_allocate(self, memory, dem_memory=None, log=True):
        if dem_memory is not None:
            self.dem.memory_allocate(dem_memory)
        if self.surface_faces is None:
            self.add_surface()
        body_capacity = int(np.max(self.surface_body)) + 1
        self.sims.configure_memory(
            dict(memory),
            self.dem.sims,
            len(self.surface_faces),
            body_capacity,
        )
        self.sims.validate()
        self._memory = dict(memory)
        if log:
            self.print_memory_info()
            self.print_neighbor_search_info()
            print()

    def print_memory_info(self):
        print_solver_section(
            "FEDEM",
            "Memory Information",
            [
                ("Maximum DEM Particles", self.sims.max_particle_num),
                ("Maximum Surface Facets", self.sims.max_surface_facet_num),
                ("Maximum DEM Materials", self.sims.max_dem_material_num),
                ("Maximum FEM Bodies", self.sims.max_fem_body_num),
                ("Contact Coordination Number", self.sims.contact_coordination_number),
                ("Contact Pair Capacity", self.sims.max_contact_pairs),
                (
                    "Point-Triangle Coordination Number",
                    self.sims.point_triangle_coordination_number,
                ),
                (
                    "Edge-Edge Coordination Number",
                    self.sims.edge_edge_coordination_number,
                ),
                ("Point-Triangle Pair Capacity", self.sims.max_point_triangle_pairs),
                ("Edge-Edge Pair Capacity", self.sims.max_edge_edge_pairs),
                ("Facet-cell Pair Capacity", self.sims.max_facet_cell_pairs),
                ("Level-set Cell Pair Capacity", self.sims.max_levelset_cell_pairs),
                ("Compaction Ratio", self.sims.compaction_ratio),
            ],
        )

    def print_neighbor_search_info(self):
        neighbor = None if self.contactor is None else self.contactor.neighbor
        active_pairs = None
        if neighbor is not None:
            active_pairs = getattr(neighbor, "contact_count", None)
        if active_pairs is None and self.enginer is not None:
            active_pairs = getattr(self.enginer, "last_contact_count", None)
        print_solver_section(
            "FEDEM",
            "Neighbor Search Information",
            [
                ("Search Method", self.sims.search),
                (
                    "Runtime Search Object",
                    type(neighbor).__name__ if neighbor is not None else None,
                ),
                ("Verlet Distance Multiplier", self.sims.verlet_distance_multiplier),
                ("Verlet Distance", self.sims.verlet_distance),
                ("Contact Coordination Number", self.sims.contact_coordination_number),
                ("Contact Pair Capacity", self.sims.max_contact_pairs),
                (
                    "Point-Triangle Coordination Number",
                    self.sims.point_triangle_coordination_number,
                ),
                (
                    "Edge-Edge Coordination Number",
                    self.sims.edge_edge_coordination_number,
                ),
                ("Point-Triangle Pair Capacity", self.sims.max_point_triangle_pairs),
                ("Edge-Edge Pair Capacity", self.sims.max_edge_edge_pairs),
                ("Facet-cell Pair Capacity", self.sims.max_facet_cell_pairs),
                ("Level-set Cell Pair Capacity", self.sims.max_levelset_cell_pairs),
                ("Current Active Pairs", active_pairs),
            ],
        )

    def choose_contact_model(self, contact_model="Linear", **kwargs):
        if self._memory is None:
            raise RuntimeError("call FEDEM.memory_allocate before choosing contact")
        return self.contactor.choose_contact_model(contact_model, **kwargs)

    def add_property(self, DEMmaterial, FEMbody, property, dType="all"):
        del dType
        self.contactor.add_property(DEMmaterial, FEMbody, property)

    def add_ipc_property(self, AffineBody, FEMbody, property=None, **kwargs):
        """Set IPC parameters for one affine-body/FEM-body pair."""
        parameters = dict(property or {})
        parameters.update(kwargs)
        self.add_property(AffineBody, FEMbody, parameters)

    def select_save_data(self, contact=True, checkpoint=False):
        self.sims.save_contact = bool(contact)
        self.sims.save_checkpoint = bool(checkpoint)

    def save_checkpoint(self, file_path=None):
        """Save one exact explicit FEM--DEM continuation point."""
        if file_path is None:
            if self.sims.path is None:
                raise RuntimeError("set a SavePath or pass an explicit checkpoint path")
            file_path = Path(self.sims.path) / "checkpoints" / f"FEDEMCheckpoint{self.sims.current_print:06d}.npz"
        return save_checkpoint(self, file_path)

    def read_restart(self, file_path, file_number=None):
        """Restore an exact coupled NPZ after rebuilding the same model."""
        if file_number is not None:
            file_path = Path(file_path) / "checkpoints" / f"FEDEMCheckpoint{int(file_number):06d}.npz"
        if self.enginer is None:
            raise RuntimeError("call FEDEM.add_essentials before read_restart")
        self.enginer.pre_calculate()
        return load_checkpoint(self, file_path)

    load_checkpoint = read_restart

    def wall_contact_elastic_energy(self, wall_id):
        """Return elastic penalty energy carried by one facet-wall group."""

        if self.enginer is None or not self.contactor.initialized:
            raise RuntimeError("initialize FEDEM before measuring wall contact energy")
        wall_id = int(wall_id)
        wall = self.dem.scene.wall
        energy = self.contactor.wall_elastic_energy(wall_id, wall)
        dem_wall_model = getattr(self.dem.contactor, "physpw", None)
        measure_dem = getattr(dem_wall_model, "lsparticle_wall_elastic_energy", None)
        if measure_dem is None:
            raise RuntimeError("DEM wall contact model does not expose elastic-energy accounting")
        energy += measure_dem(
            self.dem.scene,
            self.dem.contactor.neighbor,
            wall_id,
        )
        return float(energy)

    def deactivate_wall(self, wall_id, account_contact_energy=True):
        """Deactivate a wall group and immediately rebuild affected contacts."""

        wall_id = int(wall_id)
        wall_ids = self.dem.scene.wall.wallID.to_numpy()[: int(self.dem.scene.wallNum[0])]
        active = self.dem.scene.wall.active.to_numpy()[: int(self.dem.scene.wallNum[0])]
        selected = wall_ids == wall_id
        if not np.any(selected):
            raise KeyError(f"DEM facet-wall group {wall_id} does not exist")
        if not np.any(active[selected]):
            return 0.0
        removed_energy = self.wall_contact_elastic_energy(wall_id) if account_contact_energy else 0.0
        self.dem.update_wall_status(wall_id, "Status", "Off")
        # DEM.update_wall_status refreshes the DEM broad phase and histories.
        # Rebuild the separate FEM--facet list at the same topology boundary.
        self.contactor.rebuild_wall_candidates(self.dem.scene.wall)
        self.removed_wall_contact_energy += removed_energy
        return float(removed_energy)

    def _split_dem_functions(self, kwargs):
        result = {}
        for key, value in kwargs.items():
            if str(key).startswith("dem_"):
                result[str(key)[4:]] = value
        if "callback" in kwargs and "callback" not in result:
            result["callback"] = kwargs["callback"]
        return result

    def add_essentials(self, **kwargs):
        if self.sims.delta <= 0.0:
            raise RuntimeError("configure the FEDEM solver before run")
        if self._memory is None:
            self.memory_allocate({})
        if self.contactor.model is None:
            raise RuntimeError("choose a FEDEM contact model before run")
        if isinstance(self.contactor.model, AffineIPCModel):
            if self.dem.sims.scheme != "AffineBody":
                raise RuntimeError("FEDEM IPC requires DEM scheme='AffineBody'")
            if self.fem.sims.solver_type != "Implicit":
                raise RuntimeError("FEM--AffineBody IPC requires implicit FEM integration")
            if bool(getattr(self.fem.scene.material, "is_fem_elastoplastic", False)):
                raise RuntimeError("FEM--AffineBody IPC currently supports elastic FEM " "constitutive models only")
            if self.fem.engine is None:
                self.fem.build()
            fem_contact = getattr(self.fem.engine, "contact_assembler", None)
            if fem_contact is not None and not fem_contact.is_ipc:
                raise RuntimeError(
                    "FEM--AffineBody IPC can combine only FEM IPC contact; "
                    "augmented-Lagrangian FEM contact has a different nonlinear update"
                )
            self.fem.engine.dt = self.sims.delta
            self.fem.engine.time = self.sims.current_time
            self.fem.engine.step_count = self.sims.current_step
            self.dem.add_essentials(**self._split_dem_functions(kwargs))
            if self.enginer is None:
                from src.fedem.AffineIPCEngine import FEMAffineIPCEngine

                parameters = dict(self.solver_parameters)
                parameters.update(kwargs)
                self.enginer = FEMAffineIPCEngine(
                    self.sims,
                    self.fem,
                    self.dem,
                    self.contactor.model,
                    fem_faces=self.surface_faces,
                    **parameters,
                )
                for body_id, velocity in self._affine_velocity_boundaries:
                    self.enginer.prescribe_affine_body_velocity(body_id, velocity)
                for specification in self._affine_pressure_servos:
                    self.enginer.add_affine_body_pressure_servo(**specification)
                self.recorder = WriteFile(
                    self.sims,
                    self.fem,
                    self.dem,
                    self.contactor,
                    engine=self.enginer,
                )
                self.recorder.checkpoint_owner = self
            return
        if self.solver_parameters.get("enable_step_retry", False):
            raise RuntimeError("FEDEM step retry is available only with AffineBody IPC contact")
        if self.fem.sims.solver_type != "Explicit":
            raise RuntimeError("explicit FEDEM contact requires explicit FEM integration")
        if self.fem.scene.contact is not None:
            raise RuntimeError("FEM internal IPC/AL is not part of explicit FEDEM")
        if self.dem.sims.scheme not in ("DEM", "LSDEM"):
            raise RuntimeError("explicit FEDEM contact requires DEM scheme='DEM' or 'LSDEM'")
        if self.fem.engine is None:
            self.fem.build()
        self.fem.engine.dt = self.sims.delta
        self.fem.engine.time = self.sims.current_time
        self.fem.engine.step_count = self.sims.current_step
        self.dem.add_essentials(**self._split_dem_functions(kwargs))
        if self.enginer is None:
            self.patch = FEMSurfacePatch(
                self.fem.scene.mesh.number_of_nodes,
                self.surface_faces,
                self.surface_body,
                self.surface_modifier,
            )
            self.enginer = Engine(self.sims, self.fem, self.dem, self.contactor, self.patch)
            self.recorder = WriteFile(self.sims, self.fem, self.dem, self.contactor)
            self.recorder.checkpoint_owner = self
            self.solver = Solver(self.sims, self.fem, self.dem, self.enginer, self.recorder)
        elif DictIO.GetAlternative(kwargs, "reset_function", True):
            self.solver.postprocess.clear()
        self.solver.set_callback_function(DictIO.GetAlternative(kwargs, "function", None))

    def add_postfunctions(self, **functions):
        if isinstance(self.contactor.model, AffineIPCModel):
            self.postprocess.extend(functions.values())
            return
        if self.solver is None:
            raise RuntimeError("call FEDEM.add_essentials before adding callbacks")
        self.solver.set_callback_function(functions)

    def differentiable(self, steps):
        """Create a fixed-step device trajectory adjoint for FEM--AffineBody IPC."""
        if self.enginer is None:
            self.add_essentials()
        from src.fedem.DifferentiableFEMAffine import DifferentiableFEMAffine

        return DifferentiableFEMAffine(self.enginer, steps)

    def modify_parameters(self, **kwargs):
        simulation_time = DictIO.GetAlternative(kwargs, "SimulationTime", None)
        if simulation_time is not None:
            self.sims.set_simulation_time(simulation_time)
            self.dem.sims.set_simulation_time(simulation_time)
            if self.fem.engine is not None:
                self.fem.engine.total_step = int(math.ceil(float(simulation_time) / self.sims.delta))
        timestep = DictIO.GetAlternative(kwargs, "Timestep", None)
        if timestep is not None:
            self.sims.set_timestep(timestep)
            self.dem.sims.set_timestep(timestep)
            if self.fem.engine is not None:
                self.fem.engine.dt = float(timestep)
            if isinstance(self.contactor.model, AffineIPCModel) and self.enginer is not None:
                self.enginer.dt = float(timestep)
        if self.fem.engine is not None and (simulation_time is not None or timestep is not None):
            total_step = int(math.ceil(self.sims.time / self.sims.delta))
            self.fem.engine.reconfigure_device_boundary_timeline(self.sims.delta, total_step)
        save_interval = DictIO.GetAlternative(kwargs, "SaveInterval", None)
        if save_interval is not None:
            self.sims.set_save_interval(save_interval)
            self.dem.sims.set_save_interval(save_interval)
        gravity = DictIO.GetAlternative(kwargs, "gravity", None)
        if gravity is not None:
            self.dem.sims.set_gravity(gravity)
            if self.fem.engine is not None:
                self.fem.engine.gravity = np.asarray(gravity, dtype=float)

    def check_critical_timestep(self):
        if isinstance(self.contactor.model, AffineIPCModel):
            return self.sims.delta
        dem_dt = float(self.dem.get_critical_timestep())
        # Child explicit FEM stores its own CFL-scaled estimate.  The coupled
        # gate applies the shared CFL once to all three raw limits.
        fem_dt = float(self.fem.engine.stable_time_step(1.0))
        soft_contact = getattr(self.fem.engine, "soft_particle_contact", None)
        if soft_contact is not None:
            fem_dt = min(fem_dt, float(soft_contact.critical_timestep()))
        contact_dt = float(self.contactor.critical_timestep(self.dem.scene, self.dem.sims))
        stable = self.sims.cfl * min(dem_dt, fem_dt, contact_dt)
        if stable < self.sims.delta:
            print("The FEDEM time step is corrected as:", stable, "\n")
            self.sims.set_timestep(stable)
            self.dem.sims.set_timestep(stable)
            self.fem.engine.dt = stable
            total_step = int(math.ceil(self.sims.time / stable))
            self.fem.engine.reconfigure_device_boundary_timeline(stable, total_step)

    def run(self, **kwargs):
        self.add_essentials(**kwargs)
        if isinstance(self.contactor.model, AffineIPCModel):
            print_simulation_start("FEDEM")
            if self.sims.current_print == 0 and self.sims.current_time <= 1.0e-14:
                self._save_implicit_frame()
            elif self._last_implicit_save_time is None:
                self._last_implicit_save_time = self.sims.current_time
            callbacks = tuple(DictIO.GetAlternative(kwargs, "postprocessing", self.postprocess) or ())

            def postprocess_and_save(engine):
                for callback in callbacks:
                    callback(engine)
                if (
                    self.sims.current_time - self._last_implicit_save_time
                    >= self.sims.save_interval - 0.1 * self.sims.delta
                ):
                    self._save_implicit_frame()

            result = self.enginer.run(
                steps=DictIO.GetAlternative(kwargs, "steps", None),
                verbose=DictIO.GetAlternative(kwargs, "verbose", True),
                postprocessing=(postprocess_and_save,),
            )
            if self.sims.current_time - self._last_implicit_save_time > 0.9 * self.sims.save_interval:
                self._save_implicit_frame()
            self.dem.first_run = False
            return result
        if not self.contactor.initialized:
            self.enginer.pre_calculate()
        else:
            self.patch.update(
                self.fem.engine.position_field,
                update_normals=not self.contactor.level_set,
                update_area=False,
            )
        self.check_critical_timestep()
        self.solver.CouplingSolver(precalculated=True)
        self.dem.first_run = False

    solve = run

    def _save_implicit_frame(self):
        print_save_file_info(
            "FEDEM",
            self.sims.current_step,
            self.sims.current_print,
            self.sims.current_time,
            self.sims.path,
        )
        with self.sims.timer.section("Output"):
            self.recorder.output()
        self.sims.timer.profile0()
        self.sims.current_print += 1
        self.dem.sims.current_print += 1
        self._last_implicit_save_time = self.sims.current_time

    def save_data(self):
        if isinstance(self.contactor.model, AffineIPCModel):
            self.add_essentials()
            self._save_implicit_frame()
            return
        if self.solver is None:
            self.add_essentials()
            self.enginer.pre_calculate()
        self.solver.save_file()


__all__ = ["FEDEM"]
