"""Public explicit and implicit IPC FEM--MPM coupling facade."""

from __future__ import annotations

import math

import numpy as np

from src.fem.contact.ContactTopology import build_contact_surface
from src.fem.mainFEM import FEM
from src.fempm.ContactManager import ContactManager
from src.fempm.Engine import Engine
from src.fempm.FEMPMBase import Solver
from src.fempm.Patch import FEMPMSurfacePatch
from src.fempm.Recorder import WriteFile
from src.fempm.Simulation import FEMPMSimulation
from src.fempm.contact import IPCModel
from src.mpm.mainMPM import MPM
from src.utils.ObjectIO import DictIO
from src.utils.SolverConsole import (
    print_save_file_info,
    print_simulation_start,
    print_solver_section,
    runtime_architecture,
)
from src.utils.SolverDiagnostics import SolverDiagnosticsMixin
from src.utils.StepRetry import StepRetryPolicy


class FEMPM(SolverDiagnosticsMixin):
    """Two-way FEM--MPM contact with explicit DEM laws or implicit IPC."""

    def __init__(
        self,
        fem: FEM,
        mpm: MPM,
        title="FEM & MPM Coupling Engine",
        log=True,
    ):
        self.log = bool(log)
        if not mpm.sims.is_direct_backend():
            if mpm.scene.particle is not None and mpm.sims.coupling != "Lagrangian":
                raise RuntimeError(
                    "construct FEMPM before MPM particle fields, or configure " "MPM coupling='Lagrangian' first"
                )
            mpm.sims.set_mpm_coupling("Lagrangian")
        if log:
            print("# =================================================================== #")
            print("#", "GeoTaichi -- FEM & MPM Coupling Engine".center(67), "#")
            print("#", str(title).center(67), "#")
            print("# =================================================================== #", "\n")
        self.fem = fem
        self.mpm = mpm
        self.sims = FEMPMSimulation()
        self.sims.current_time = float(self.mpm.sims.current_time)
        self.sims.current_step = int(self.mpm.sims.current_step)
        self.sims.current_print = int(self.mpm.sims.current_print)
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

    def set_configuration(self, log=True, **kwargs):
        dimension = int(self.mpm.sims.dimension)
        fem_dimension = int(self.fem.sims.dimension)
        if fem_dimension != dimension:
            raise ValueError("FEMPM child solvers must use the same dimension")
        mpm_axisymmetric = bool(self.mpm.sims.is_2DAxisy)
        fem_axisymmetric = bool(self.fem.sims.is_axisymmetric)
        if mpm_axisymmetric != fem_axisymmetric:
            raise ValueError("FEMPM child solvers must use the same planar or " "axisymmetric mode")
        axis_offset = float(self.mpm.sims.axis_offset)
        requested_axisymmetric = bool(DictIO.GetAlternative(kwargs, "axisymmetric", mpm_axisymmetric))
        if requested_axisymmetric != mpm_axisymmetric:
            raise ValueError("FEMPM coupling and child solvers must use the same planar " "or axisymmetric mode")
        requested_axis_offset = float(DictIO.GetAlternative(kwargs, "axis_offset", axis_offset))
        if not math.isfinite(requested_axis_offset):
            raise ValueError("FEMPM axis_offset must be finite")
        if mpm_axisymmetric and not np.isclose(axis_offset, self.fem.sims.axis_offset):
            raise ValueError("axisymmetric FEMPM children must share axis_offset")
        if mpm_axisymmetric and not np.isclose(axis_offset, requested_axis_offset):
            raise ValueError("axisymmetric FEMPM coupling and children must share axis_offset")
        domain = DictIO.GetAlternative(kwargs, "domain", None)
        if domain is None:
            domain = np.asarray(self.mpm.sims.get_simulation_domain(), dtype=float)
        self.sims.set_domain(
            domain,
            dimension=dimension,
            axisymmetric=mpm_axisymmetric,
            axis_offset=requested_axis_offset,
        )
        child_domain = np.asarray(self.mpm.sims.get_simulation_domain(), dtype=float).reshape(-1)
        if child_domain.size < dimension or np.linalg.norm(child_domain[:dimension]) <= 1.0e-14:
            self.mpm.sims.set_domain(domain)
        elif not np.allclose(child_domain[:dimension], self.sims.domain[:dimension]):
            raise ValueError("FEMPM and MPM domains must match")

        search = str(DictIO.GetAlternative(kwargs, "search", "LinkedCell"))
        normalized = search.replace("_", "").replace("-", "").lower()
        if normalized == "linkedcell":
            self.sims.search = "LinkedCell"
        elif normalized in ("bvh", "lbvh"):
            self.sims.search = "BVH"
        else:
            self.sims.search = search
        gravity = DictIO.GetAlternative(kwargs, "gravity", None)
        if gravity is not None:
            self.mpm.sims.set_gravity(gravity)
            self.fem.solver_kwargs["gravity"] = gravity
            if self.fem.engine is not None:
                fem_gravity = np.asarray(gravity, dtype=float).reshape(-1)
                if fem_gravity.size not in (2, 3):
                    raise ValueError("FEMPM gravity must contain two or three components")
                if fem_gravity.size == 2:
                    fem_gravity = np.append(fem_gravity, 0.0)
                self.fem.engine.gravity = fem_gravity
        if log:
            self.print_basic_simulation_info()
            print()

    def print_basic_simulation_info(self):
        print_solver_section(
            "FEMPM",
            "Basic Configuration",
            [
                ("Simulation Type", runtime_architecture()),
                ("Dimension", self.sims.dimension),
                ("Simulation Domain", self.sims.domain),
                ("Neighbor Search Type", self.sims.search),
                ("MPM Backend", self.mpm.sims.mpm_backend),
                ("MPM Solver Type", self.mpm.sims.solver_type),
                ("FEM Solver Type", self.fem.sims.solver_type),
                ("Axisymmetric", self.sims.is_axisymmetric),
                ("Axis Offset", self.sims.axis_offset),
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
            raise KeyError("FEMPM.set_solver requires Timestep and SimulationTime")
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
        child_parameters = dict(parameters)
        child_parameters.update(
            {
                "Timestep": timestep,
                "SimulationTime": simulation_time,
                "SaveInterval": save_interval,
                "SavePath": self.sims.path,
                "CFL": self.sims.cfl,
            }
        )
        self.mpm.set_solver(child_parameters, log=False)
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
            "FEMPM",
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
        if self.sims.dimension == 2:
            edges, owners = self.fem.scene.mesh.boundary_facets()
            return (
                np.ascontiguousarray(edges, dtype=np.int32),
                np.ascontiguousarray(self.fem.scene.mesh.cell_body_ids[owners], dtype=np.int32),
            )
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
                values = [
                    np.asarray(value, dtype=np.int32).reshape(-1, 2 if self.sims.dimension == 2 else 3)
                    for value in facet_sets.values()
                ]
                faces = np.concatenate(values, axis=0) if values else np.empty((0, 3), dtype=np.int32)
            else:
                faces = np.asarray(facet_sets, dtype=np.int32).reshape(-1, 2 if self.sims.dimension == 2 else 3)
            node_body = self.fem.scene.mesh.node_body_ids
            face_body = node_body[faces[:, 0]]
            if np.any(node_body[faces] != face_body[:, None]):
                raise ValueError("each FEMPM surface facet must belong to one FEM body")
            if body_ids is not None:
                selected = np.isin(face_body, np.asarray(body_ids, dtype=np.int32))
                faces, face_body = faces[selected], face_body[selected]
        if faces.shape[0] == 0:
            raise ValueError("FEMPM coupled surface contains no boundary primitives")
        self.surface_faces = np.ascontiguousarray(faces, dtype=np.int32)
        self.surface_body = np.ascontiguousarray(face_body, dtype=np.int32)
        self.surface_modifier = modifier
        self.patch = None
        return self.surface_faces

    add_patch = add_surface

    def memory_allocate(self, memory, mpm_memory=None, log=True):
        if mpm_memory is not None:
            self.mpm.memory_allocate(mpm_memory)
        if self.surface_faces is None:
            self.add_surface()
        memory = dict(memory)
        if self.mpm.sims.is_direct_backend():
            direct_bodies = self.mpm.direct_bodies
            if direct_bodies is None:
                raise RuntimeError("add Direct MPM bodies before FEMPM.memory_allocate")
            memory.setdefault("max_particle_number", int(direct_bodies.particle_counter))
            memory.setdefault("max_mpm_material_number", 1)
        body_capacity = int(np.max(self.surface_body)) + 1
        self.sims.configure_memory(
            memory,
            self.mpm.sims,
            len(self.surface_faces),
            body_capacity,
        )
        self.sims.validate()
        self._memory = memory
        if log:
            self.print_memory_info()
            self.print_neighbor_search_info()
            print()

    def print_memory_info(self):
        print_solver_section(
            "FEMPM",
            "Memory Information",
            [
                ("Maximum MPM Particles", self.sims.max_particle_num),
                ("Maximum Surface Facets", self.sims.max_surface_facet_num),
                ("Maximum MPM Materials", self.sims.max_mpm_material_num),
                ("Maximum FEM Bodies", self.sims.max_fem_body_num),
                ("Contact Coordination Number", self.sims.contact_coordination_number),
                ("Contact Pair Capacity", self.sims.max_contact_pairs),
                (
                    "Point-Triangle Pair Capacity",
                    self.sims.max_point_triangle_pairs,
                ),
                ("Point-Edge Pair Capacity", self.sims.max_point_edge_pairs),
                ("Facet-cell Pair Capacity", self.sims.max_facet_cell_pairs),
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
            "FEMPM",
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
                    "Point-Triangle Pair Capacity",
                    self.sims.max_point_triangle_pairs,
                ),
                ("Point-Edge Pair Capacity", self.sims.max_point_edge_pairs),
                ("Facet-cell Pair Capacity", self.sims.max_facet_cell_pairs),
                ("Current Active Pairs", active_pairs),
            ],
        )

    def choose_contact_model(self, contact_model="Linear", **kwargs):
        if self._memory is None:
            raise RuntimeError("call FEMPM.memory_allocate before choosing contact")
        return self.contactor.choose_contact_model(contact_model, **kwargs)

    def add_property(self, MPMmaterial, FEMbody, property, dType="all"):
        del dType
        self.contactor.add_property(MPMmaterial, FEMbody, property)

    def add_ipc_property(self, MPMbody, FEMbody, property=None, **kwargs):
        """Set one Direct-MPM-body/FEM-body IPC parameter set."""
        parameters = dict(property or {})
        parameters.update(kwargs)
        self.add_property(MPMbody, FEMbody, parameters)

    def select_save_data(self, contact=True):
        self.sims.save_contact = bool(contact)

    def add_essentials(self, **kwargs):
        if self.sims.delta <= 0.0:
            raise RuntimeError("configure the FEMPM solver before run")
        if self._memory is None:
            self.memory_allocate({})
        if self.contactor.model is None:
            raise RuntimeError("choose a FEMPM contact model before run")
        if self.mpm.sims.dimension not in (2, 3):
            raise RuntimeError("FEMPM IPC requires dimension=2 or 3")
        if isinstance(self.contactor.model, IPCModel):
            if self.mpm.sims.soft_particle and bool(getattr(self.fem.scene.material, "is_cloth", False)):
                raise NotImplementedError("soft-particle MPM--cloth IPC coupling is intentionally unsupported")
            if not self.mpm.sims.is_direct_backend():
                raise RuntimeError(
                    "FEMPM IPC requires mpm_backend='Direct' so MPM active "
                    "grid DOFs can enter the monolithic Newton system"
                )
            if self.mpm.sims.solver_type != "Implicit":
                raise RuntimeError("FEMPM IPC requires implicit MPM integration")
            if self.mpm.sims.ipc_contact:
                raise RuntimeError("disable standalone MPM IPC; FEMPM owns the coupled IPC system")
            if self.fem.sims.solver_type != "Implicit":
                raise RuntimeError("FEMPM IPC requires implicit FEM integration")
            if bool(getattr(self.fem.scene.material, "is_fem_elastoplastic", False)):
                raise RuntimeError("FEMPM IPC currently supports elastic FEM constitutive " "models only")
            if self.fem.scene.contact is not None:
                raise RuntimeError("standalone FEM IPC/AL cannot be combined with FEMPM IPC yet")
            if self.fem.engine is None:
                self.fem.build()
            self.fem.engine.dt = self.sims.delta
            self.fem.engine.time = self.sims.current_time
            self.fem.engine.step_count = self.sims.current_step
            self.mpm.add_essentials(**kwargs)
            from src.mpm.engines.direct.ImplicitMPM import ImplicitMPM

            if not isinstance(self.mpm.enginer, ImplicitMPM):
                raise RuntimeError("FEMPM IPC requires the Direct finite-strain implicit " "MPM engine")
            if self.enginer is None:
                from src.fempm.ImplicitEngine import FEMPMImplicitEngine

                parameters = dict(self.solver_parameters)
                parameters.update(kwargs)
                self.enginer = FEMPMImplicitEngine(
                    self.sims,
                    self.fem,
                    self.mpm,
                    self.contactor.model,
                    self.surface_faces,
                    self.surface_body,
                    **parameters,
                )
                self.recorder = WriteFile(self.sims, self.fem, self.mpm, self.contactor)
            return
        if self.solver_parameters.get("enable_step_retry", False):
            raise RuntimeError("FEMPM step retry is available only with implicit IPC contact")
        if self.sims.dimension == 2:
            raise RuntimeError(
                "two-dimensional and axisymmetric FEMPM coupling currently " "require the implicit IPC contact model"
            )
        if self.mpm.sims.solver_type != "Explicit":
            raise RuntimeError("FEMPM supports explicit MPM integration only")
        if self.mpm.sims.is_direct_backend():
            raise RuntimeError("FEMPM requires the particle/grid MPM backend")
        if self.fem.sims.solver_type != "Explicit":
            raise RuntimeError("FEMPM supports explicit FEM integration only")
        if self.fem.scene.contact is not None:
            raise RuntimeError("FEM internal IPC/AL is not part of explicit FEMPM")
        self.mpm.scene.update_coupling_points_number(self.mpm.sims)
        if int(self.mpm.scene.couplingNum[0]) <= 0:
            raise RuntimeError("FEMPM requires at least one coupled MPM point")
        if int(self.mpm.scene.couplingNum[0]) > self.sims.max_particle_num:
            raise RuntimeError("FEMPM max_particle_number is smaller than the coupled MPM " "point count")
        if self.fem.engine is None:
            self.fem.build()
        self.fem.engine.dt = self.sims.delta
        self.fem.engine.time = self.sims.current_time
        self.fem.engine.step_count = self.sims.current_step
        self.mpm.add_essentials(**kwargs)
        if self.enginer is None:
            self.patch = FEMPMSurfacePatch(
                self.fem.scene.mesh.number_of_nodes,
                self.surface_faces,
                self.surface_body,
                self.surface_modifier,
            )
            self.enginer = Engine(self.sims, self.fem, self.mpm, self.contactor, self.patch)
            self.recorder = WriteFile(self.sims, self.fem, self.mpm, self.contactor)
            self.solver = Solver(self.sims, self.fem, self.mpm, self.enginer, self.recorder)
        elif DictIO.GetAlternative(kwargs, "reset_function", True):
            self.solver.postprocess.clear()
        self.solver.set_callback_function(DictIO.GetAlternative(kwargs, "function", None))

    def add_postfunctions(self, **functions):
        if isinstance(self.contactor.model, IPCModel):
            self.postprocess.extend(functions.values())
            return
        if self.solver is None:
            raise RuntimeError("call FEMPM.add_essentials before callbacks")
        self.solver.set_callback_function(functions)

    def modify_parameters(self, **kwargs):
        simulation_time = DictIO.GetAlternative(kwargs, "SimulationTime", None)
        if simulation_time is not None:
            self.sims.set_simulation_time(simulation_time)
            self.mpm.sims.set_simulation_time(simulation_time)
            if self.fem.engine is not None:
                self.fem.engine.total_step = int(math.ceil(float(simulation_time) / self.sims.delta))
        timestep = DictIO.GetAlternative(kwargs, "Timestep", None)
        if timestep is not None:
            self.sims.set_timestep(timestep)
            self.mpm.sims.set_timestep(timestep)
            if self.fem.engine is not None:
                self.fem.engine.dt = float(timestep)
            if isinstance(self.contactor.model, IPCModel) and self.enginer is not None:
                self.enginer.dt = float(timestep)
                self.mpm.enginer.dt = float(timestep)
        if self.fem.engine is not None and (simulation_time is not None or timestep is not None):
            total_step = int(math.ceil(self.sims.time / self.sims.delta))
            self.fem.engine.reconfigure_device_boundary_timeline(self.sims.delta, total_step)
        save_interval = DictIO.GetAlternative(kwargs, "SaveInterval", None)
        if save_interval is not None:
            self.sims.set_save_interval(save_interval)
            self.mpm.sims.set_save_interval(save_interval)
        gravity = DictIO.GetAlternative(kwargs, "gravity", None)
        if gravity is not None:
            self.mpm.sims.set_gravity(gravity)
            if self.fem.engine is not None:
                self.fem.engine.gravity = np.asarray(gravity, dtype=float)

    def check_critical_timestep(self):
        if isinstance(self.contactor.model, IPCModel):
            return self.sims.delta
        mpm_dt = float(self.mpm.scene.get_critical_timestep())
        fem_dt = float(self.fem.engine.stable_time_step(1.0))
        contact_dt = float(self.contactor.critical_timestep(self.mpm.scene))
        stable = self.sims.cfl * min(mpm_dt, fem_dt, contact_dt)
        if stable < self.sims.delta:
            print("The FEMPM time step is corrected as:", stable, "\n")
            self.sims.set_timestep(stable)
            self.mpm.sims.set_timestep(stable)
            self.fem.engine.dt = stable
            total_step = int(math.ceil(self.sims.time / stable))
            self.fem.engine.reconfigure_device_boundary_timeline(stable, total_step)

    def run(self, **kwargs):
        self.add_essentials(**kwargs)
        if isinstance(self.contactor.model, IPCModel):
            print_simulation_start("FEMPM")
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
            self.mpm.first_run = False
            return result
        if not self.contactor.initialized:
            self.enginer.pre_calculate()
        else:
            self.patch.update(self.fem.engine.position_field)
        self.check_critical_timestep()
        self.solver.CouplingSolver(precalculated=True)
        self.mpm.first_run = False

    solve = run

    def _save_implicit_frame(self):
        print_save_file_info(
            "FEMPM",
            self.sims.current_step,
            self.sims.current_print,
            self.sims.current_time,
            self.sims.path,
        )
        with self.sims.timer.section("Output"):
            self.recorder.output()
        self.sims.timer.profile0()
        self.sims.current_print += 1
        self.mpm.sims.current_print += 1
        self._last_implicit_save_time = self.sims.current_time

    def save_data(self):
        if isinstance(self.contactor.model, IPCModel):
            self.add_essentials()
            self._save_implicit_frame()
            return
        if self.solver is None:
            self.add_essentials()
            self.enginer.pre_calculate()
        self.solver.save_file()


__all__ = ["FEMPM"]
