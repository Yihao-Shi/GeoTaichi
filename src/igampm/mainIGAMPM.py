from src.igampm.ContactManager import ContactManager
from src.igampm.Simulation import Simulation
import src.igampm.config as config
from src.utils.SolverDiagnostics import SolverDiagnosticsMixin
from src.utils.SolverConsole import (
    print_save_file_info,
    print_simulation_start,
    print_solver_section,
    runtime_architecture,
)
from src.utils.StepRetry import StepRetryPolicy


class IGAMPM(SolverDiagnosticsMixin):
    def __init__(self, iga=None, mpm=None, title="IGA-MPM Coupling Engine", log=True, **kwargs):
        self.log = bool(log)
        if log:
            print("# =================================================================== #")
            print("#", "".center(67), "#")
            print("#", "Welcome to GeoTaichi -- IGA & MPM Coupling Engine !".center(67), "#")
            print("#", "".center(67), "#")
            print("#", title.center(67), "#")
            print("#", "".center(67), "#")
            print("# =================================================================== #", "\n")

        if iga is None:
            from src.iga.mainIGA import IGA

            iga = IGA(title="", log=False)
        if mpm is None:
            from src.mpm.mainMPM import MPM

            mpm = MPM(title="", log=False)
        self.iga = iga
        self.mpm = mpm
        self.sims = Simulation()
        self.contact_kwargs = dict(kwargs)
        self.contactor = ContactManager(**kwargs)
        if self.contactor.contact_model != "IPC" and not self.mpm.sims.is_direct_backend():
            if self.mpm.scene.particle is not None and self.mpm.sims.coupling != "Lagrangian":
                raise RuntimeError(
                    "construct explicit IGA-MPM before allocating MPM "
                    "particles, or configure coupling='Lagrangian' first"
                )
            self.mpm.sims.set_mpm_coupling("Lagrangian")
        self.engine = None
        self.iga_engine = None
        self.mpm_engine = None
        self._last_implicit_recorded_step = None

    def _contact_model_kwargs(self):
        return {key: value for key, value in self.contact_kwargs.items() if key != "contact_model"}

    def set_configuration(
        self,
        dimension=None,
        coupling_scheme="IGAMPM",
        contact_model="IPC",
        activate_friction=False,
        axisymmetric=False,
        axis_offset=0.0,
        log=True,
    ):
        if self.engine is not None and dimension is not None and int(dimension) != int(self.sims.dimension):
            raise RuntimeError(
                "IGA-MPM dimension cannot be changed after build(); "
                "the coupled fields and Taichi kernels use the build-time dimension"
            )
        if self.engine is not None:
            raise RuntimeError(
                "IGA-MPM configuration cannot be changed after build(); " "construct a new coupling engine instead"
            )
        self.sims.set_configuration(
            dimension=dimension,
            coupling_scheme=coupling_scheme,
            contact_model=contact_model,
            activate_friction=activate_friction,
            axisymmetric=axisymmetric,
            axis_offset=axis_offset,
        )
        self.choose_contact_model(
            contact_model=contact_model,
            activate_friction=activate_friction,
        )
        if log:
            self.print_basic_simulation_info()
            print()
        return self

    def print_basic_simulation_info(self):
        print_solver_section(
            "IGAMPM",
            "Basic Configuration",
            [
                ("Simulation Type", runtime_architecture()),
                ("Dimension", self.sims.dimension),
                ("Coupling Scheme", self.sims.coupling_scheme),
                ("Contact Model", self.contactor.contact_model),
                ("Friction Enabled", self.contactor.activate_friction),
                ("Axisymmetric", self.sims.is_axisymmetric),
                ("Axis Offset", self.sims.axis_offset),
            ],
        )

    def choose_contact_model(self, contact_model="IPC", **kwargs):
        if self.engine is not None:
            if bool(getattr(self.engine, "implicit_step_in_progress", False)):
                raise RuntimeError("IGA-MPM contact parameters cannot be changed during an " "implicit IPC time step")
            raise RuntimeError(
                "IGA-MPM contact_model cannot be changed after build(); "
                "friction mode and all contact parameters are frozen"
            )
        updated_contact_kwargs = dict(self.contact_kwargs)
        updated_contact_kwargs.update(kwargs)
        candidate_kwargs = {key: value for key, value in updated_contact_kwargs.items() if key != "contact_model"}
        candidate = ContactManager(contact_model=contact_model, **candidate_kwargs)
        if candidate.contact_model != "IPC":
            sims = getattr(self.mpm, "sims", None)
            scene = getattr(self.mpm, "scene", None)
            if sims is not None and not sims.is_direct_backend():
                if getattr(scene, "particle", None) is not None and sims.coupling != "Lagrangian":
                    raise RuntimeError(
                        "select explicit IGA-MPM contact before allocating MPM "
                        "particles, or configure coupling='Lagrangian' first"
                    )
                sims.set_mpm_coupling("Lagrangian")
        self.contact_kwargs = updated_contact_kwargs
        self.contactor = candidate
        self.sims.contact_model = self.contactor.contact_model
        return self

    def add_property(self, MPMmaterial, IGAbody, property, dType="all"):
        """Assign one MPM-material/IGA-body explicit DEM contact law."""
        del dType
        self.contactor.add_property(MPMmaterial, IGAbody, property)
        return self

    def _freeze_built_configuration(self):
        self.contactor.freeze_configuration()
        self.sims.freeze_configuration()

    def add_essentials(self):
        config.set_dimension(self.sims.dimension)
        return self

    def set_solver(self, iga=None, mpm=None, coupling=None, log=True, **kwargs):
        if iga is not None:
            if not hasattr(self.iga, "set_solver"):
                raise RuntimeError("The current IGA object does not support set_solver().")
            iga_parameters = dict(iga)
            iga_parameters.setdefault("log", False)
            self.iga.set_solver(**iga_parameters)
        if mpm is not None:
            if not hasattr(self.mpm, "set_solver"):
                raise RuntimeError("The current MPM object does not support set_solver().")
            self.mpm.set_solver(mpm, log=False)
        has_coupling_parameters = coupling is not None or bool(kwargs)
        coupling_parameters = {} if coupling is None else dict(coupling)
        coupling_parameters.update(kwargs)
        if not has_coupling_parameters:
            if log:
                self.print_solver_info()
                print()
            return self
        retry_policy = StepRetryPolicy(
            enabled=coupling_parameters.get("enable_step_retry", False),
            maximum_retries=coupling_parameters.get("step_retry_max_retries", 2),
            reduction=coupling_parameters.get("step_retry_reduction", 0.5),
            minimum_timestep=coupling_parameters.get("step_retry_minimum_timestep", 0.0),
        )
        coupling_parameters.update(
            enable_step_retry=retry_policy.enabled,
            step_retry_max_retries=retry_policy.maximum_retries,
            step_retry_reduction=retry_policy.reduction,
            step_retry_minimum_timestep=retry_policy.minimum_timestep,
        )
        if retry_policy.enabled and self.contactor.contact_model != "IPC":
            raise ValueError("IGA-MPM step retry is available only for implicit IPC contact")
        if coupling_parameters:
            if self.engine is not None:
                raise RuntimeError("IGA-MPM solver settings cannot change after build()")
            self.contact_kwargs.update(coupling_parameters)
        if log:
            self.print_solver_info()
            print()
        return self

    def print_solver_info(self):
        iga_solver = getattr(self.iga, "solver_kwargs", {})
        mpm_sims = getattr(self.mpm, "sims", None)
        try:
            mpm_timestep = mpm_sims.dt[None]
        except (AttributeError, RuntimeError, TypeError):
            mpm_timestep = None
        print_solver_section(
            "IGAMPM",
            "Solver Information",
            [
                ("IGA Time Step", iga_solver.get("dt")),
                ("MPM Time Step", mpm_timestep),
                ("Save Path", self._save_path()),
                (
                    "Assembly Type",
                    self.contact_kwargs.get("assemble_type", self.contact_kwargs.get("assembly")),
                ),
                ("Linear Solver", self.contact_kwargs.get("linear_solver")),
                ("Friction Mode", self.contactor.friction_mode),
            ],
        )

    def _save_path(self):
        iga_path = getattr(self.iga, "solver_kwargs", {}).get("path")
        mpm_sims = getattr(self.mpm, "sims", None)
        mpm_path = None if mpm_sims is None else getattr(mpm_sims, "path", None)
        if iga_path == mpm_path or mpm_path is None:
            return iga_path
        if iga_path is None:
            return mpm_path
        return f"IGA={iga_path}, MPM={mpm_path}"

    def print_memory_info(self):
        engine = self.engine
        iga_engine = self.iga_engine
        mpm_engine = self.mpm_engine
        print_solver_section(
            "IGAMPM",
            "Memory Information",
            [
                (
                    "IGA Degrees of Freedom",
                    None if iga_engine is None else getattr(iga_engine, "degree_of_freedom", None),
                ),
                (
                    "MPM Degrees of Freedom",
                    None if mpm_engine is None else getattr(mpm_engine, "degree_of_freedom", None),
                ),
                (
                    "Maximum MPM Particles",
                    getattr(
                        mpm_engine,
                        "n_particles",
                        getattr(getattr(self.mpm, "sims", None), "max_particle_num", None),
                    ),
                ),
                (
                    "MPM Contact Candidates",
                    None if engine is None else getattr(engine.mpm, "total_surface_num", None),
                ),
                (
                    "Contact Pair Capacity",
                    None if engine is None else getattr(engine, "contact_capacity", None),
                ),
                (
                    "Barrier Matrix Capacity",
                    None if engine is None else getattr(engine, "barrier_nnz_capacity", None),
                ),
                (
                    "Friction Matrix Capacity",
                    None if engine is None else getattr(engine, "friction_nnz_capacity", None),
                ),
            ],
        )

    def print_neighbor_search_info(self):
        engine = self.engine
        explicit = self.contactor.contact_model != "IPC"
        print_solver_section(
            "IGAMPM",
            "Neighbor Search Information",
            [
                (
                    "Search Method",
                    "Device NURBS surface search" if explicit else "Device IPC candidate assembly",
                ),
                ("Contact Model", self.contactor.contact_model),
                (
                    "Contact Pair Capacity",
                    None if engine is None else getattr(engine, "contact_capacity", None),
                ),
                (
                    "Current Active Pairs",
                    None if engine is None else getattr(engine, "last_contact_count", None),
                ),
            ],
        )

    def _build_iga_engine(self):
        if hasattr(self.iga, "degree_of_freedom") and hasattr(self.iga, "patch"):
            return self.iga
        if hasattr(self.iga, "build"):
            dimension = getattr(self.iga, "dimension", None)
            if dimension is not None:
                config.set_dimension(dimension)
            return self.iga.build()
        raise RuntimeError("IGAMPM requires an IGA wrapper with build() or a built IGA engine.")

    def _build_mpm_engine(self):
        if hasattr(self.mpm, "build_surface_node") and hasattr(self.mpm, "particle"):
            return self.mpm
        if hasattr(self.mpm, "add_engine"):
            dimension = getattr(getattr(self.mpm, "sims", None), "dimension", None)
            if dimension is not None:
                config.set_dimension(dimension)
            self.mpm.add_engine()
            if getattr(self.mpm, "enginer", None) is None:
                raise RuntimeError("MPM wrapper did not build an engine.")
            return self.mpm.enginer
        raise RuntimeError("IGAMPM requires an MPM wrapper with add_engine() or a built MPM engine.")

    def _prepare_mpm_engine(self, mpm):
        if int(mpm.active_dof) > 0:
            return
        # ``ImplicitEngineMixin._initialize_implicit_ipc_state()`` initializes
        # an all-zero F0 field and preserves every valid user-provided state.
        # Do not overwrite a predeformed/rest-shape configuration while this
        # facade builds the first updated-Lagrangian active map.
        mpm.mass_vec.fill(0.0)
        mpm.grid_reset()
        mpm.compute_shapefn()
        mpm.mass_vel_acc_p2g()
        mpm.find_active_node()
        mpm.prefix_sum_executor.run(mpm.node2dof)
        mpm.active_dof = mpm.set_active_dof()
        mpm.compute_nodal_vel_acc()

    def _is_implicit_engine(self, engine):
        from src.iga.engines.ImplicitIGA import ImplicitIGA
        from src.mpm.engines.direct.ImplicitMPM import ImplicitMPM

        return isinstance(engine, (ImplicitIGA, ImplicitMPM))

    def _validate_implicit_ipc_solvers(self):
        iga_implicit = self._is_implicit_engine(self.iga_engine)
        mpm_implicit = self._is_implicit_engine(self.mpm_engine)
        if iga_implicit and mpm_implicit:
            return
        raise RuntimeError("IGA-MPM IPC requires both IGA and MPM solvers to be implicit")

    def _build_explicit_engine(self):
        if bool(self.contact_kwargs.get("enable_step_retry", False)):
            raise ValueError("IGA-MPM step retry is available only for implicit IPC contact")
        if self.sims.dimension != 3 or self.sims.is_axisymmetric:
            raise RuntimeError(
                "explicit DEM-law IGA-MPM contact currently supports 3D "
                "Cartesian simulations; use IPC for 2D/axisymmetric coupling"
            )
        if self._is_implicit_engine(self.iga_engine):
            raise RuntimeError("explicit IGA-MPM requires an Explicit IGA solver")
        if self.mpm.sims.is_direct_backend():
            raise RuntimeError("explicit IGA-MPM requires the native particle/grid MPM backend")
        if self.mpm.sims.solver_type != "Explicit":
            raise RuntimeError("explicit IGA-MPM requires an Explicit MPM solver")
        if self.mpm.sims.coupling != "Lagrangian":
            raise RuntimeError("explicit IGA-MPM requires MPM coupling='Lagrangian'")
        if not self.contactor.phys.pending_properties:
            raise RuntimeError("add at least one explicit IGA-MPM contact property before build()")
        self.mpm.scene.update_coupling_points_number(self.mpm.sims)
        if int(self.mpm.scene.couplingNum[0]) <= 0:
            raise RuntimeError("explicit IGA-MPM requires at least one coupled MPM point")
        self.mpm.add_essentials(**self.contact_kwargs)
        self.mpm_engine = self.mpm.enginer
        mpm_dt = float(self.mpm.sims.dt[None])
        iga_dt = float(self.iga_engine.dt)
        if mpm_dt <= 0.0 or iga_dt <= 0.0:
            raise RuntimeError("explicit IGA-MPM requires positive child timesteps")
        tolerance = 1.0e-12 * max(1.0, abs(mpm_dt), abs(iga_dt))
        if abs(mpm_dt - iga_dt) > tolerance:
            raise RuntimeError("explicit IGA-MPM child solvers must use the same timestep")
        from src.igampm.engines.ExplicitEngine import ExplicitEngine

        return ExplicitEngine(self.iga, self.mpm, self.contactor, **self.contact_kwargs)

    def build(self):
        if self.engine is not None:
            self._freeze_built_configuration()
            return self.engine
        self.add_essentials()
        self.iga_engine = self._build_iga_engine()
        if self.contactor.contact_model != "IPC":
            self.engine = self._build_explicit_engine()
            self.engine.timer = self.sims.timer
            self._freeze_built_configuration()
            if self.log:
                self.print_memory_info()
                self.print_neighbor_search_info()
                print()
            return self.engine
        self.mpm_engine = self._build_mpm_engine()
        self._validate_implicit_ipc_solvers()
        iga_axisymmetric = bool(getattr(self.iga_engine, "is_axisymmetric", False))
        mpm_axisymmetric = bool(getattr(self.mpm_engine, "is_axisymmetric", False))
        if iga_axisymmetric != mpm_axisymmetric:
            raise RuntimeError(
                "IGA-MPM axisymmetric IPC requires both child solvers to use " "the same axisymmetric mode"
            )
        axis_configuration_explicit = (
            bool(self.sims.axis_configuration_explicit) or "axisymmetric" in self.contact_kwargs
        )
        requested_axisymmetric = (
            bool(self.sims.is_axisymmetric)
            if self.sims.axis_configuration_explicit
            else bool(self.contact_kwargs.get("axisymmetric", False))
        )
        if axis_configuration_explicit and requested_axisymmetric != iga_axisymmetric:
            raise RuntimeError(
                "IGA-MPM coupling and both child solvers must use the same " "planar or axisymmetric mode"
            )
        if iga_axisymmetric:
            iga_axis = float(self.iga_engine.axis_offset)
            mpm_axis = float(self.mpm_engine.axis_offset)
            if abs(iga_axis - mpm_axis) > 1.0e-12 * max(1.0, abs(iga_axis), abs(mpm_axis)):
                raise RuntimeError("IGA-MPM axisymmetric child solvers must share axis_offset")
            requested_axis_offset = (
                float(self.sims.axis_offset)
                if self.sims.axis_configuration_explicit
                else float(self.contact_kwargs.get("axis_offset", iga_axis))
            )
            axis_offset_explicit = bool(self.sims.axis_configuration_explicit) or "axis_offset" in self.contact_kwargs
            if axis_offset_explicit and abs(requested_axis_offset - iga_axis) > 1.0e-12 * max(
                1.0, abs(requested_axis_offset), abs(iga_axis)
            ):
                raise RuntimeError("IGA-MPM coupling and child solvers must share axis_offset")
            self.sims.is_axisymmetric = True
            self.sims.axis_offset = iga_axis
        self._prepare_mpm_engine(self.mpm_engine)
        from src.igampm.engines import Engine

        self.engine = Engine(self.iga_engine, self.mpm_engine, self.contactor, **self.contact_kwargs)
        self.engine.timer = self.sims.timer
        self.engine.compile_seconds = None
        self._freeze_built_configuration()
        if self.log:
            self.print_memory_info()
            self.print_neighbor_search_info()
            print()
        return self.engine

    def run(
        self,
        grid_disp=None,
        include_friction=False,
        steps=None,
        verbose=True,
        inner_solve=None,
        assemble_updated_system=None,
        probe_solve=None,
        linear_solve=None,
        energy_function=None,
        newton_max_iterations=None,
        newton_tolerance=None,
        solve=True,
        record=True,
        postprocessing=(),
    ):
        engine = self.build()
        print_simulation_start("IGAMPM")
        if self.contactor.contact_model != "IPC":
            unsupported = any(
                value is not None
                for value in (
                    grid_disp,
                    inner_solve,
                    assemble_updated_system,
                    probe_solve,
                    linear_solve,
                    energy_function,
                    newton_max_iterations,
                    newton_tolerance,
                )
            )
            if unsupported or include_friction or not solve:
                raise ValueError(
                    "implicit assembly/Newton options are unavailable for "
                    "explicit Linear/HertzMindlin IGA-MPM contact"
                )
            result = engine.run(
                steps=steps,
                verbose=verbose,
                record=record,
                postprocessing=postprocessing,
            )
            self.mpm.first_run = False
            return result
        if inner_solve is not None or assemble_updated_system is not None:
            if inner_solve is None or assemble_updated_system is None:
                raise ValueError("inner_solve and assemble_updated_system must be provided together")
        elif not solve:
            # Assembly-only mode returns the selected Taichi COO or HashTriplet
            # system consumed by the built-in nonlinear solver. Converting it to
            # SciPy is permitted only at an explicitly selected host linear-solve
            # boundary.
            return engine.assemble_monolithic_newton_system(
                include_friction=include_friction or engine.activate_fric,
                need_matrix=True,
            )

        callbacks = tuple(postprocessing or ())
        if record:
            engine._initialize_implicit_ipc_state()
            if engine.implicit_step_index == 0 and self._last_implicit_recorded_step is None:
                self._record_implicit_frame(engine)
            output_interval = max(1, int(getattr(self.iga_engine, "output_interval", 1)))

            def postprocess_and_record(coupled_engine):
                for callback in callbacks:
                    callback(coupled_engine)
                if coupled_engine.implicit_step_index % output_interval == 0:
                    self._record_implicit_frame(coupled_engine)

            engine_callbacks = (postprocess_and_record,)
        else:
            engine_callbacks = callbacks

        solve_kwargs = {
            "steps": steps,
            "grid_disp": grid_disp,
            "probe_solve": probe_solve,
            "include_friction": include_friction or engine.activate_fric,
            "linear_solve": linear_solve,
            "energy_function": energy_function,
            "newton_max_iterations": newton_max_iterations,
            "newton_tolerance": newton_tolerance,
            "verbose": verbose,
            "postprocessing": engine_callbacks,
        }
        if inner_solve is not None:
            solve_kwargs.update(
                inner_solve=inner_solve,
                assemble_updated_system=assemble_updated_system,
            )
        result = engine.run_implicit_ipc_contact(**solve_kwargs)

        if record and self._last_implicit_recorded_step != engine.implicit_step_index:
            self._record_implicit_frame(engine)
        self.mpm.first_run = False
        return result

    def _record_implicit_frame(self, engine):
        step = int(engine.implicit_step_index)
        simulation_time = float(engine.time)
        for child in (self.iga_engine, self.mpm_engine):
            child.step_count = step
            child.time = simulation_time
        mpm_sims = self.mpm.sims
        mpm_sims.current_step = step
        mpm_sims.current_time = simulation_time
        print_save_file_info(
            "IGAMPM",
            step,
            getattr(self.iga_engine, "output_count", 0),
            simulation_time,
            self._save_path(),
        )
        with self.sims.timer.section("Output"):
            self.iga_engine.record(log=False)
            self.mpm_engine.record(log=False)
        self.sims.timer.profile0()
        mpm_sims.current_print += 1
        self._last_implicit_recorded_step = step

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if self.engine is None:
            self.build()
        return getattr(self.engine, name)


__all__ = ["IGAMPM"]
