from types import SimpleNamespace

import numpy as np

from src.dem.mainDEM import DEM
from src.mpm.mainMPM import MPM

from src.mpdem.ContactManager import ContactManager
from src.mpdem.DEMPMBase import Solver
from src.mpdem.Engine import Engine
from src.mpdem.GenerateManager import GenerateManager
from src.mpdem.Recorder import WriteFile
from src.mpdem.Simulation import Simulation
from src.mpm.soft_particle.DEMPMBridge import install_dem_soft_particle_backend
from src.mpm.Recorder import monitor_soft_material_point
from src.utils.ObjectIO import DictIO
from src.utils.SolverDiagnostics import SolverDiagnosticsMixin
from src.utils.SolverConsole import print_solver_section, runtime_architecture
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverRuntime import normalize_callbacks
from src.utils.StepRetry import StepRetryPolicy


class _SoftParticleMPMRecorderProxy:
    def __init__(self, dem_recorder, mpm_recorder, mpm_sims):
        self.dem_recorder = dem_recorder
        self.mpm_recorder = mpm_recorder
        self.mpm_sims = mpm_sims

    def __getattr__(self, name):
        return getattr(self.dem_recorder, name)

    def _save_soft_points(self, dem_sims):
        monitor_type = set(getattr(dem_sims, "monitor_type", []))
        monitor_type.update(getattr(self.mpm_sims, "monitor_type", []))
        return bool(monitor_type.intersection({"particle", "soft", "material_point"}))

    def output(self, dem_sims, dem_scene):
        self.dem_recorder.output(dem_sims, dem_scene)
        if int(dem_scene.softPointNum[0]) == 0 or not self._save_soft_points(dem_sims):
            return

        original_time = self.mpm_sims.current_time
        original_print = self.mpm_sims.current_print
        self.mpm_sims.current_time = dem_sims.current_time
        self.mpm_sims.current_print = dem_sims.current_print
        monitor_soft_material_point(self.mpm_recorder, self.mpm_sims, dem_scene)
        self.mpm_sims.current_time = original_time
        self.mpm_sims.current_print = original_print


class DEMPM(SolverDiagnosticsMixin):
    def __init__(
        self, dem: DEM, mpm: MPM, title="A High Performance Multiscale and Multiphysics Simulator on GPU", log=True
    ):
        if log:
            print("# =================================================================== #")
            print("#", "".center(67), "#")
            print("#", "Welcome to GeoTaichi -- DEM & MPM Coupling Engine !".center(67), "#")
            print("#", "".center(67), "#")
            print("#", title.center(67), "#")
            print("#", "".center(67), "#")
            print("# =================================================================== #", "\n")
        self.dem = dem
        self.mpm = mpm
        install_dem_soft_particle_backend(self.dem)
        self.sims = Simulation()
        self.generator = GenerateManager(self.mpm.generator, self.dem.generator)
        self.contactor = None
        self.enginer = None
        self.solver = None
        self.recorder = None
        self.direct_affine_ipc_model = None
        self.direct_affine_contact = None
        self._direct_affine_output_started = False
        self.incompressible_affine_coupler = None
        self._incompressible_affine_output_started = False

    def set_configuration(self, log=True, **kwargs):
        if (
            np.linalg.norm(np.array(self.mpm.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10
            and np.linalg.norm(np.array(self.dem.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10
        ):
            domain = DictIO.GetEssential(kwargs, "domain")
            self.sims.set_domain(domain)
            self.dem.sims.set_domain(domain)
            self.mpm.sims.set_domain(domain)
        elif (
            np.linalg.norm(np.array(self.mpm.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10
            and np.linalg.norm(np.array(self.dem.sims.get_simulation_domain()) - np.zeros(3)) > 1e-10
        ):
            domain = self.dem.sims.get_simulation_domain()
            self.sims.set_domain(domain)
            self.mpm.sims.set_domain(domain)
        elif (
            np.linalg.norm(np.array(self.mpm.sims.get_simulation_domain()) - np.zeros(3)) > 1e-10
            and np.linalg.norm(np.array(self.dem.sims.get_simulation_domain()) - np.zeros(3)) > 1e-10
        ):
            domain = self.mpm.sims.get_simulation_domain()
            self.sims.set_domain(domain)
            self.dem.sims.set_domain(domain)
        elif (
            np.linalg.norm(np.array(self.mpm.sims.get_simulation_domain()) - np.zeros(3)) > 1e-10
            and np.linalg.norm(np.array(self.dem.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10
        ):
            if not all(self.mpm.sims.get_simulation_domain() == self.dem.sims.get_simulation_domain()):
                raise RuntimeError(
                    f"DEM simulation domain {self.dem.sims.get_simulation_domain()} is not in line with MPM simulation domain {self.mpm.sims.get_simulation_domain()}"
                )
            else:
                self.sims.set_domain(self.mpm.sims.get_simulation_domain())

        self.sims.set_coupling_scheme(DictIO.GetAlternative(kwargs, "coupling_scheme", "MPDEM"))
        self.sims.set_cfdem_resolution(DictIO.GetAlternative(kwargs, "cfdem_resolution", "Auto"))
        self.sims.set_particle_interaction(DictIO.GetAlternative(kwargs, "particle_interaction", True))
        self.sims.set_wall_interaction(DictIO.GetAlternative(kwargs, "wall_interaction", False))
        self.sims.set_digital_elevation_contact_mode(
            DictIO.GetAlternative(kwargs, "digital_elevation_contact", "heightfield")
        )
        self.dem.sims.set_digital_elevation_contact_mode(self.sims.digital_elevation_contact_mode)
        self.mpm.sims.set_gravity(DictIO.GetAlternative(kwargs, "gravity", [0.0, 0.0, -9.8]))
        self.dem.sims.set_gravity(DictIO.GetAlternative(kwargs, "gravity", [0.0, 0.0, -9.8]))
        self.mpm.sims.set_visualize(DictIO.GetAlternative(kwargs, "visualize", True))
        self.dem.sims.set_visualize(DictIO.GetAlternative(kwargs, "visualize", True))
        self.mpm.sims.set_track_energy(DictIO.GetAlternative(kwargs, "track_energy", False))
        self.dem.sims.set_track_energy(DictIO.GetAlternative(kwargs, "track_energy", False))
        self.dem.sims.set_enable_shell(DictIO.GetAlternative(kwargs, "enable_shell", False))
        self.sims.set_CFD_coupling_domain(DictIO.GetAlternative(kwargs, "CFD_coupling_domain", [3, 6]))
        self.sims.set_enhanced_coupling(DictIO.GetAlternative(kwargs, "enhanced_coupling", False))
        self.dem.sims.set_search(DictIO.GetAlternative(kwargs, "search", "LinkedCell"))
        if self.sims.enhanced_coupling:
            self.mpm.sims.set_norm_adaptivity(True)
        self.sims.validate_configuration()

        if self.mpm.sims.coupling is False and not self.is_direct_mpm_affine_ipc():
            raise RuntimeError(f"KeyWord::: /coupling/ should be activated in MPM")

        if self.dem.sims.coupling is False:
            raise RuntimeError(f"KeyWord::: /coupling/ should be activated in DEM")

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
        soft_affine_ipc = self.dem.sims.scheme == "LSMPM" and self.dem.sims.lsmpm_soft_rigid_contact == "IPC"
        direct_affine_ipc = self.is_direct_mpm_affine_ipc()
        if retry_policy.enabled and not (soft_affine_ipc or direct_affine_ipc):
            raise ValueError(
                "DEMPM/CFDEM step retry is available only for monolithic " "Soft-Affine or Direct-MPM--AffineBody IPC"
            )
        self.sims.enable_step_retry = retry_policy.enabled
        self.sims.step_retry_max_retries = retry_policy.maximum_retries
        self.sims.step_retry_reduction = retry_policy.reduction
        self.sims.step_retry_minimum_timestep = retry_policy.minimum_timestep
        timestep = DictIO.GetEssential(solver, "Timestep")
        self.sims.set_timestep(timestep)
        self.sims.set_dem_timestep(DictIO.GetAlternative(solver, "DEMTimestep", timestep))
        self.sims.set_simulation_time(DictIO.GetEssential(solver, "SimulationTime"))
        self.sims.set_CFL(DictIO.GetAlternative(solver, "CFL", 0.5))
        self.sims.set_adaptive_timestep(DictIO.GetAlternative(solver, "AdaptiveStep", False))
        self.sims.set_save_interval(DictIO.GetEssential(solver, "SaveInterval"))
        self.sims.set_save_path(DictIO.GetAlternative(solver, "SavePath", "OutputData"))
        mpm_solver = dict(solver)
        if soft_affine_ipc:
            mpm_solver["enable_step_retry"] = False
        self.mpm.set_solver(mpm_solver, log=False)
        dem_solver = dict(solver)
        dem_solver["Timestep"] = self.sims.dem_timestep
        self.dem.set_solver(dem_solver, log=False)
        if log:
            self.print_solver_info()
            print("\n")

    def memory_allocate(self, memory, dem_memory=None, mpm_memory=None, log=True):
        if dem_memory is not None:
            self.dem.memory_allocate(dem_memory)
        if mpm_memory is not None:
            self.mpm.memory_allocate(mpm_memory)

        if self.dem.sims.max_material_num == 0 or self.mpm.sims.max_material_num == 0:
            raise RuntimeError("Should allocate DEM and MPM memory first!")
        self.sims.set_material_num(max(self.dem.sims.max_material_num, self.mpm.sims.max_material_num))
        self.sims.set_body_coordination_number(DictIO.GetAlternative(memory, "body_coordination_number", 64))
        self.sims.set_wall_coordination_number(DictIO.GetAlternative(memory, "wall_coordination_number", 6))
        self.sims.set_compaction_ratio(DictIO.GetAlternative(memory, "compaction_ratio", [0.4, 0.3]))
        self.sims.set_particle_contact_list_capacity(DictIO.GetAlternative(memory, "max_particle_contact_pairs", 0))
        self.sims.set_wall_contact_list_capacity(DictIO.GetAlternative(memory, "max_wall_contact_pairs", 0))
        if self.is_direct_mpm_affine_ipc():
            particles = int(getattr(self.mpm.direct_bodies, "particle_counter", 0))
            default_pairs = max(1, particles * int(self.sims.body_coordination_number))
            self.sims.max_point_triangle_pairs = int(
                DictIO.GetAlternative(memory, "max_point_triangle_pairs", default_pairs)
            )
            self.sims.max_point_edge_pairs = int(DictIO.GetAlternative(memory, "max_point_edge_pairs", 1))
            self.sims.contact_search = str(self.dem.sims.search)
            if self.sims.max_point_triangle_pairs <= 0 or self.sims.max_point_edge_pairs <= 0:
                raise ValueError("Direct MPM--AffineBody IPC pair capacities must be positive")
        self.sims.validate_coupling_configuration(self.mpm.sims, self.dem.sims)
        if log:
            self.print_memory_info()
            self.print_neighbor_search_info()
            print("\n")

    def print_basic_simulation_info(self):
        solver_name = self.sims.coupling_scheme
        print_solver_section(
            solver_name,
            "Basic Configuration",
            [
                ("Simulation Type", runtime_architecture()),
                ("Simulation Domain", self.sims.domain),
                ("Coupling Scheme", self.sims.coupling_scheme),
                ("Particle Interaction", self.sims.particle_interaction),
                ("Wall Interaction", self.sims.wall_interaction),
            ],
        )

    def print_solver_info(self):
        print_solver_section(
            self.sims.coupling_scheme,
            "Solver Information",
            [
                ("Coupling Scheme", self.sims.coupling_scheme),
                ("Initial Simulation Time", self.sims.current_time),
                ("Final Simulation Time", self.sims.current_time + self.sims.time),
                ("Time Step", self.sims.dt[None]),
                ("DEM Time Step", self.sims.dem_timestep),
                ("Adaptive Time Step", self.sims.adaptive_timestep),
                ("CFL", self.sims.CFL),
                ("Save Interval", self.sims.save_interval),
                ("Save Path", self.sims.path),
            ],
        )

    def print_memory_info(self):
        print_solver_section(
            self.sims.coupling_scheme,
            "Memory Information",
            [
                ("Maximum Coupled Materials", self.sims.max_material_num),
                ("Maximum DEM Particles", self.dem.sims.max_particle_num),
                ("Maximum MPM Particles", self.mpm.sims.max_particle_num),
                ("Maximum Coupling Particles", self.mpm.sims.max_coupling_particle_num),
                ("Body Coordination Number", self.sims.body_coordination_number),
                ("Wall Coordination Number", self.sims.wall_coordination_number),
                ("Particle Contact Capacity", self.sims.particle_contact_list_capacity),
                ("Wall Contact Capacity", self.sims.wall_contact_list_capacity),
                ("Particle Contact Compaction", self.sims.compaction_ratio[0]),
                ("Wall Contact Compaction", self.sims.compaction_ratio[1]),
            ],
        )

    def print_neighbor_search_info(self):
        neighbor = None
        if self.contactor is not None:
            neighbor = self.contactor.neighbor
        print_solver_section(
            self.sims.coupling_scheme,
            "Neighbor Search Information",
            [
                ("DEM Search Method", self.dem.sims.search),
                ("Runtime Coupling Search", type(neighbor).__name__ if neighbor is not None else None),
                ("Particle Interaction", self.sims.particle_interaction),
                ("Wall Interaction", self.sims.wall_interaction),
                ("Body Coordination Number", self.sims.body_coordination_number),
                ("Wall Coordination Number", self.sims.wall_coordination_number),
                ("Potential Particle Pair Capacity", self.sims.max_potential_particle_pairs),
                ("Potential Wall Pair Capacity", self.sims.max_potential_wall_pairs),
                ("Particle Contact Capacity", self.sims.particle_contact_list_capacity),
                ("Wall Contact Capacity", self.sims.wall_contact_list_capacity),
            ],
        )

    def add_body(
        self, mpm_body=None, dem_particle=None, write_file=False, check_overlap=False, adaptive_boundary_radius=False
    ):
        self.generator.add_mixture(
            check_overlap,
            adaptive_boundary_radius,
            dem_particle,
            mpm_body,
            self.sims,
            self.dem.scene,
            self.mpm.scene,
            self.dem.sims,
            self.mpm.sims,
        )
        if write_file:
            self.dem.add_recorder()
            self.dem.recorder.save_particle(self.dem.sims, self.dem.scene)
            self.dem.recorder.save_sphere(self.dem.sims, self.dem.scene)
            self.dem.recorder.save_clump(self.dem.sims, self.dem.scene)
            self.mpm.add_recorder()
            self.mpm.recorder.save_particle(self.mpm.sims, self.mpm.scene)

    def choose_contact_model(self, particle_particle_contact_model, particle_wall_contact_model=None, **kwargs):
        if self.is_direct_mpm_affine_ipc():
            if particle_wall_contact_model is not None:
                raise ValueError("Direct MPM--AffineBody IPC does not use a separate wall contact model")
            from src.fempm.contact.IPC import IPCModel

            self.direct_affine_ipc_model = IPCModel(
                SimpleNamespace(search=str(getattr(self.sims, "contact_search", self.dem.sims.search))),
                ipc_model=particle_particle_contact_model,
                **kwargs,
            )
            return self.direct_affine_ipc_model
        if self.is_incompressible_mpm_affine_body():
            if particle_particle_contact_model is not None or particle_wall_contact_model is not None or kwargs:
                raise ValueError("Incompressible MPM--AffineBody IBM coupling does not use a cross-contact model")
            return None
        if self.dem.contactor is None:
            self.dem.choose_contact_model()
        if self.mpm.neighbor is None:
            self.mpm.add_spatial_grid()

        if self.contactor is None:
            self.contactor = ContactManager()
            if self.sims.coupling_scheme != "CFDEM":
                self.contactor.choose_neighbor(
                    self.sims, self.mpm.sims, self.dem.sims, self.mpm.neighbor, self.dem.contactor.neighbor
                )

        if self.sims.coupling_scheme != "CFDEM":
            if self.sims.max_material_num == 0:
                raise RuntimeError("memory_allocate should be launched first!")
            self.sims.set_particle_particle_contact_model(particle_particle_contact_model)
            self.sims.set_particle_wall_contact_model(particle_wall_contact_model)
            self.contactor.particle_particle_initialize(self.sims, self.mpm.sims, self.dem.sims)
            self.contactor.particle_wall_initialize(self.sims, self.mpm.sims, self.dem.sims)

    def add_property(self, DEMmaterial, MPMmaterial, property, dType="all"):
        if self.is_direct_mpm_affine_ipc():
            if self.direct_affine_ipc_model is None:
                raise RuntimeError("choose Direct MPM--AffineBody IPC before adding pair properties")
            del dType
            return self.direct_affine_ipc_model.add_property(MPMmaterial, DEMmaterial, property)
        if self.sims.coupling_scheme != "CFDEM":
            self.contactor.add_contact_property(self.sims, MPMmaterial, DEMmaterial, property, dType)

    def add_ipc_property(self, MPMbody, AffineBody, property=None, **kwargs):
        parameters = dict(property or {})
        parameters.update(kwargs)
        return self.add_property(AffineBody, MPMbody, parameters)

    def modify_parameters(self, **kwargs):
        if len(kwargs) > 0:
            simulation_time = DictIO.GetEssential(kwargs, "SimulationTime")
            self.sims.set_simulation_time(simulation_time)
            self.mpm.sims.set_simulation_time(simulation_time)
            self.dem.sims.set_simulation_time(simulation_time)
            if "Timestep" in kwargs:
                self.sims.set_timestep(DictIO.GetEssential(kwargs, "Timestep"))
                self.mpm.sims.set_timestep(DictIO.GetEssential(kwargs, "Timestep"))
                self.dem.sims.set_timestep(DictIO.GetEssential(kwargs, "Timestep"))
            if "CFL" in kwargs:
                self.sims.set_CFL(DictIO.GetEssential(kwargs, "CFL"))
                self.mpm.sims.set_CFL(DictIO.GetEssential(kwargs, "CFL"))
                self.dem.sims.set_CFL(DictIO.GetEssential(kwargs, "CFL"))
            if "AdaptiveTimestep" in kwargs:
                self.sims.set_adaptive_timestep(DictIO.GetEssential(kwargs, "AdaptiveStep"))
                self.mpm.sims.set_adaptive_timestep(DictIO.GetEssential(kwargs, "AdaptiveStep"))
                self.dem.sims.set_adaptive_timestep(DictIO.GetEssential(kwargs, "AdaptiveStep"))
            if "SaveInterval" in kwargs:
                self.sims.set_save_interval(DictIO.GetEssential(kwargs, "SaveInterval"))
                self.mpm.sims.set_save_interval(DictIO.GetEssential(kwargs, "SaveInterval"))
                self.dem.sims.set_save_interval(DictIO.GetEssential(kwargs, "SaveInterval"))
            if "SavePath" in kwargs:
                self.sims.set_save_path(DictIO.GetEssential(kwargs, "SavePath"))
                self.mpm.sims.set_save_path(DictIO.GetEssential(kwargs, "SavePath"))
                self.dem.sims.set_save_path(DictIO.GetEssential(kwargs, "SavePath"))

            if "gravity" in kwargs:
                self.mpm.sims.set_gravity(DictIO.GetEssential(kwargs, "gravity"))
                self.dem.sims.set_gravity(DictIO.GetEssential(kwargs, "gravity"))
            if "background_damping" in kwargs:
                self.mpm.sims.set_background_damping(DictIO.GetEssential(kwargs, "background_damping"))
            if "alphaPIC" in kwargs:
                self.mpm.sims.set_alpha(DictIO.GetEssential(kwargs, "alphaPIC"))
            if "coupling_scheme" in kwargs:
                self.sims.set_coupling_scheme(DictIO.GetAlternative(kwargs, "coupling_scheme", "DEM-MPM"))

    def sync_settings(self):
        if self.dem.sims.max_particle_num > 0:
            if self.dem.sims.is_continue != self.mpm.sims.is_continue:
                raise RuntimeError(
                    f"The continue flag in MPM {self.mpm.sims.is_continue} and DEM {self.dem.sims.is_continue} is different"
                )
        self.dem.sims.set_is_continue(self.mpm.sims.is_continue)
        self.sims.set_is_continue(self.mpm.sims.is_continue)
        if self.sims.is_continue:
            if self.dem.sims.max_particle_num > 0:
                if self.dem.sims.current_print != self.mpm.sims.current_print:
                    raise RuntimeError(
                        f"The print in MPM {self.mpm.sims.current_print} and DEM {self.dem.sims.current_print} is different"
                    )
                if self.dem.sims.current_time != self.mpm.sims.current_time:
                    raise RuntimeError(
                        f"The time in MPM {self.mpm.sims.current_time} and DEM {self.dem.sims.current_time} is different"
                    )
            self.dem.sims.current_print = 1 * self.mpm.sims.current_print
            self.sims.current_print = 1 * self.mpm.sims.current_print
            self.dem.sims.current_time = 1.0 * self.mpm.sims.current_time
            self.sims.current_time = 1.0 * self.mpm.sims.current_time
            self.sims.CurrentTime[None] = self.mpm.sims.current_time

    def read_restart(self, file_number, file_path, ppcontact=False, pwcontact=False):
        ppcontact_path = None
        pwcontact_path = None
        self.sync_settings()
        if ppcontact:
            ppcontact_path = file_path + "/DEMPMcontacts"
        if pwcontact:
            pwcontact_path = file_path + "/DEMPMcontacts"
        self.sims.history_contact_path.update(
            file_number=file_number, ppcontact=ppcontact_path, pwcontact=pwcontact_path
        )

    def load_history_contact(self):
        file_number = DictIO.GetAlternative(self.sims.history_contact_path, "file_number", 0)
        ppcontact = DictIO.GetAlternative(self.sims.history_contact_path, "ppcontact", None)
        pwcontact = DictIO.GetAlternative(self.sims.history_contact_path, "pwcontact", None)

        if not ppcontact is None:
            self.contactor.physpp.restart(self.contactor.neighbor, file_number, ppcontact, True)
        if not pwcontact is None:
            self.contactor.physpw.restart(self.contactor.neighbor, file_number, pwcontact, False)

    def select_save_data(self, particle_particle_contact=False, particle_wall_contact=False):
        self.sims.set_save_data(particle_particle_contact, particle_wall_contact)

    def add_essentials(self, **kwargs: dict):
        if self.is_direct_mpm_affine_ipc():
            self._add_direct_mpm_affine_ipc_essentials(**kwargs)
            return
        if self.is_incompressible_mpm_affine_body():
            self._add_incompressible_mpm_affine_essentials(**kwargs)
            return
        self.sync_settings()
        self.sims.validate_coupling_configuration(self.mpm.sims, self.dem.sims)

        def split_function(dicts, name):
            split_dict = {}
            for keys, values in dicts.items():
                if name in keys:
                    split_dict.update({keys.replace(name, "", 1): values})
            return split_dict

        self.mpm.scene.update_coupling_points_number(self.mpm.sims)
        if self.dem.scene.particleNum[0] > 0 or self.mpm.scene.particleNum[0] > 0:
            if self.sims.coupling_scheme != "CFDEM" and (self.sims.particle_interaction or self.sims.wall_interaction):
                if self.contactor.have_initialise is False:
                    self.contactor.initialize(self.sims, self.mpm.sims, self.dem.sims, self.mpm.scene, self.dem.scene)
        else:
            raise RuntimeError("DEM/MPM particle should be added first")
        self.load_history_contact()

        mpm_function = split_function(kwargs, "mpm_")
        dem_function = split_function(kwargs, "dem_")
        dem_function.update(
            {"max_bounding_radius": self.sims.max_bounding_rad, "min_bounding_radius": self.sims.min_bounding_rad}
        )
        self.mpm.add_essentials(**mpm_function)
        self.dem.add_essentials(**dem_function)

        if self.contactor is None:
            self.choose_contact_model(None, None)

        self.recorder = WriteFile(
            self.sims,
            self.mpm.sims,
            self.dem.sims,
            self.dem.recorder,
            self.mpm.recorder,
            self.contactor.physpp,
            self.contactor.physpw,
            self.contactor.neighbor,
        )
        if self.enginer is None:
            self.enginer = Engine(
                self.sims,
                self.mpm.sims,
                self.dem.sims,
                self.mpm.scene,
                self.dem.scene,
                self.mpm.enginer,
                self.dem.enginer,
                self.contactor.neighbor,
                self.mpm.neighbor,
                self.dem.contactor.neighbor,
                self.contactor.physpp,
                self.contactor.physpw,
            )
        self.enginer.choose_engine(DictIO.GetAlternative(kwargs, "drag_model", {}))
        self.enginer.set_servo_mechanism()

        if self.solver is None:
            self.solver = Solver(
                self.sims,
                self.mpm.sims,
                self.dem.sims,
                self.mpm.recorder,
                self.dem.recorder,
                self.generator,
                self.enginer,
                self.recorder,
            )
        if DictIO.GetAlternative(kwargs, "reset_function", True):
            self.solver.postprocess = []
        self.solver.set_callback_function(DictIO.GetAlternative(kwargs, "function", None))
        self.solver.set_particle_calm(self.dem.scene, DictIO.GetAlternative(kwargs, "calm", None))

        if self.sims.is_continue:
            self.sims.current_print += 1
            self.mpm.sims.current_print += 1
            self.dem.sims.current_print += 1
            self.sims.set_is_continue(False)
            self.mpm.sims.set_is_continue(False)
            self.dem.sims.set_is_continue(False)
            self.solver.last_save_time = 1.0 * self.mpm.sims.current_time

    def add_postfunctions(self, **functions):
        self.solver.set_callback_function(functions)

    def update_contact_properties(self, materialID1, materialID2, property_name, value, overide=True):
        self.contactor.update_contact_property(self.sims, materialID1, materialID2, property_name, value, overide)

    def is_dem_lsm_pm_soft_rigid_mode(self):
        return self.dem.sims.scheme == "LSMPM" and self.mpm.sims.max_material_num == 0

    def is_direct_mpm_affine_ipc(self):
        return (
            self.dem.sims.scheme == "AffineBody"
            and self.mpm.sims.is_direct_backend()
            and self.mpm.sims.solver_type == "Implicit"
        )

    def is_incompressible_mpm_affine_body(self):
        return (
            self.dem.sims.scheme == "AffineBody"
            and self.mpm.sims.solver_type == "Implicit"
            and self.mpm.sims.material_type == "Fluid"
            and self.mpm.sims.discretization == "FDM"
            and not self.mpm.sims.is_direct_backend()
        )

    def _add_incompressible_mpm_affine_essentials(self, **kwargs):
        self.sync_settings()
        self.sims.validate_coupling_configuration(self.mpm.sims, self.dem.sims)
        mpm_function = {key[4:]: value for key, value in kwargs.items() if key.startswith("mpm_")}
        dem_function = {key[4:]: value for key, value in kwargs.items() if key.startswith("dem_")}
        self.mpm.add_essentials(**mpm_function)
        self.dem.add_essentials(**dem_function)
        if self.incompressible_affine_coupler is not None:
            return

        affine_engine = self.dem.enginer
        affine_engine.initialize(self.dem.sims, self.dem.scene)
        from src.mpdem.fluid_dynamics.IncompressibleCoupling import IncompressibleAffineBodyCoupling

        self.incompressible_affine_coupler = IncompressibleAffineBodyCoupling(
            self.mpm.sims,
            self.mpm.scene,
            self.mpm.enginer,
            affine_engine,
        )
        self.incompressible_affine_coupler.attach()
        self.mpm.add_solver(**mpm_function)

    def run_incompressible_mpm_affine_body(self, **kwargs):
        self._add_incompressible_mpm_affine_essentials(**kwargs)
        coupler = self.incompressible_affine_coupler
        affine_engine = self.dem.enginer
        callbacks = normalize_callbacks(DictIO.GetAlternative(kwargs, "postprocessing", ()))

        if not self._incompressible_affine_output_started:
            self.mpm.enginer.pre_calculation(self.mpm.sims, self.mpm.scene, self.mpm.neighbor)
            self.mpm.solver.save_file(self.mpm.scene)
            affine_engine._prepare_output(self.dem.sims)
            affine_engine._save(self.dem.sims, self.dem.scene)
            self._incompressible_affine_output_started = True

        last_save_time = float(self.sims.current_time)
        target_time = float(self.sims.current_time + self.sims.time)
        nominal_timestep = float(self.sims.delta)
        while self.sims.current_time < target_time - 1.0e-14:
            step_dt = min(nominal_timestep, target_time - self.sims.current_time)
            for child in (self.sims, self.mpm.sims, self.dem.sims):
                child.set_timestep(step_dt)

            self.mpm.solver.core(self.mpm.scene, self.mpm.neighbor)
            coupler.accumulate_pressure_force(self.mpm.sims, self.mpm.scene)
            affine_engine.step(self.dem.sims, self.dem.scene)

            self.sims.current_time += step_dt
            self.sims.current_step += 1
            self.sims.CurrentTime[None] = self.sims.current_time
            for child in (self.mpm.sims, self.dem.sims):
                child.current_time = self.sims.current_time
                child.current_step = self.sims.current_step
            runtime_checkpoint()
            for callback in callbacks:
                callback(coupler)

            if self.sims.current_time - last_save_time >= self.sims.save_interval - 0.1 * step_dt:
                self.mpm.solver.save_file(self.mpm.scene)
                affine_engine._save(self.dem.sims, self.dem.scene)
                last_save_time = self.sims.current_time

        if self.sims.current_time - last_save_time > 0.9 * max(nominal_timestep, 1.0e-14):
            self.mpm.solver.save_file(self.mpm.scene)
            affine_engine._save(self.dem.sims, self.dem.scene)
        for child in (self.sims, self.mpm.sims, self.dem.sims):
            child.set_timestep(nominal_timestep)
        self.mpm.first_run = False
        self.dem.first_run = False
        return {"converged": True, "time": self.sims.current_time, "step": self.sims.current_step}

    def _add_direct_mpm_affine_ipc_essentials(self, **kwargs):
        if not self.mpm.sims.ipc_contact:
            raise RuntimeError("Direct MPM--AffineBody coupling requires MPM IPC contact")
        if self.direct_affine_ipc_model is None:
            raise RuntimeError("choose BarrierIPC or SemiIPC on MPDEM before run")
        self.sync_settings()
        self.sims.validate_coupling_configuration(self.mpm.sims, self.dem.sims)
        self.mpm.add_essentials(**{key[4:]: value for key, value in kwargs.items() if key.startswith("mpm_")})
        self.dem.add_essentials(**{key[4:]: value for key, value in kwargs.items() if key.startswith("dem_")})
        if self.enginer is not None:
            return

        direct_mpm = self.mpm.enginer
        if not hasattr(direct_mpm, "ipc") or not hasattr(direct_mpm, "prepare_step_device"):
            raise RuntimeError("Direct MPM--AffineBody IPC requires the ULMPM IPC engine")
        affine_engine = self.dem.enginer
        affine_engine.initialize(self.dem.sims, self.dem.scene)
        affine_engine.last_device_nonlinear_path = True
        from src.physics_model.contact_model.ipc.IPC import normalize_ipc_model

        mpm_model = normalize_ipc_model(direct_mpm.ipc.barrier.model)
        affine_model = normalize_ipc_model(affine_engine.operator.contact_model)
        mixed_model = normalize_ipc_model(self.direct_affine_ipc_model.contact.model)
        if len({mpm_model, affine_model, mixed_model}) != 1:
            raise ValueError(
                "Direct MPM, AffineBody, and mixed contact must all select "
                f"the same IPC model, got {mpm_model}, {affine_model}, {mixed_model}"
            )
        if mixed_model == "SemiIPC" and self.sims.enable_step_retry:
            raise ValueError("Direct MPM--AffineBody SemiIPC does not support step retry")

        from src.mpdem.engines.DirectAffineIPCAssembler import DirectAffineIPCAssembler
        from src.mpdem.engines.DirectAffineIPCSystem import DirectAffineIPCSystem

        self.direct_affine_contact = DirectAffineIPCAssembler(
            affine_engine.operator,
            direct_mpm.mpm,
            self.direct_affine_ipc_model,
            self.sims,
        )
        self.enginer = DirectAffineIPCSystem(
            affine_engine.operator,
            direct_mpm.ipc,
            self.direct_affine_contact,
        )
        direct_mpm.mpm.time = float(self.sims.current_time)
        direct_mpm.mpm.step_count = int(self.sims.current_step)

    def run_direct_mpm_affine_ipc(self, **kwargs):
        self._add_direct_mpm_affine_ipc_essentials(**kwargs)
        system = self.enginer
        direct_mpm = self.mpm.enginer
        mpm_solver = direct_mpm.mpm
        affine_engine = self.dem.enginer
        callbacks = normalize_callbacks(DictIO.GetAlternative(kwargs, "postprocessing", ()))

        if not self._direct_affine_output_started:
            direct_mpm.initial_simulation()
            affine_engine._prepare_output(self.dem.sims)
            affine_engine._save(self.dem.sims, self.dem.scene)
            self._direct_affine_output_started = True
        last_save_time = float(self.sims.current_time)
        target_time = float(self.sims.current_time + self.sims.time)
        nominal_timestep = float(self.sims.delta)
        while self.sims.current_time < target_time - 1.0e-14:
            mpm_solver.dt = min(nominal_timestep, target_time - self.sims.current_time)

            def coupled_substep(verbose):
                direct_mpm.prepare_step_device()
                result = system.solve_lagged_equilibrium_device(
                    self.dem.sims,
                    record_adjoint=bool(DictIO.GetAlternative(kwargs, "record_adjoint", False)),
                )
                system.commit_step_device()
                return result

            next_step = mpm_solver.step_count + 1
            final_step = self.sims.current_time + mpm_solver.dt >= target_time - 1.0e-14
            record_history = mpm_solver.step_schedule.history_due(next_step, final=final_step)
            step_result = mpm_solver.run_substep(
                coupled_substep,
                verbose=bool(DictIO.GetAlternative(kwargs, "verbose", True)),
                record_history=record_history,
            )
            contact = system.mixed.diagnostics()
            system.last_contact_count = int(contact["active_contacts"])
            system.last_candidate_count = int(contact["candidate_contacts"])
            self.sims.current_time = float(mpm_solver.time)
            self.sims.current_step = int(mpm_solver.step_count)
            self.sims.CurrentTime[None] = self.sims.current_time
            for child in (self.mpm.sims, self.dem.sims):
                child.current_time = self.sims.current_time
                child.current_step = self.sims.current_step
            system.last_step_record = {
                **step_result,
                "contact": contact,
                "step": self.sims.current_step,
                "time": self.sims.current_time,
            }
            runtime_checkpoint()
            for callback in callbacks:
                callback(system)
            if self.sims.current_time - last_save_time >= self.sims.save_interval - 0.1 * mpm_solver.dt:
                direct_mpm.mpm.record()
                affine_engine._save(self.dem.sims, self.dem.scene)
                last_save_time = self.sims.current_time

        if self.sims.current_time - last_save_time > 0.9 * max(nominal_timestep, 1.0e-14):
            direct_mpm.mpm.record()
            affine_engine._save(self.dem.sims, self.dem.scene)
        mpm_solver.dt = nominal_timestep
        self.mpm.first_run = False
        self.dem.first_run = False
        return {
            "converged": True,
            "time": self.sims.current_time,
            "step": self.sims.current_step,
            "last_step": system.last_step_record,
        }

    def run_dem_lsm_pm_soft_rigid(self, **kwargs):
        visualize = DictIO.GetAlternative(kwargs, "visualize", False)
        strict_timestep = bool(DictIO.GetAlternative(kwargs, "strict_timestep", False))
        run_kwargs = dict(kwargs)
        run_kwargs.pop("visualize", None)
        run_kwargs.pop("strict_timestep", None)

        original_coupling = self.dem.sims.coupling
        self.dem.sims.set_dem_coupling(False)
        try:
            self.dem.add_essentials(**run_kwargs)
            self.mpm.add_recorder()
            if not isinstance(self.dem.solver.recorder, _SoftParticleMPMRecorderProxy):
                self.dem.solver.recorder = _SoftParticleMPMRecorderProxy(
                    self.dem.solver.recorder, self.mpm.recorder, self.mpm.sims
                )
            if self.dem.sims.scheme == "AffineBody":
                self.dem.solver.Solver(self.dem.scene)
                self.dem.first_run = False
                return
            if hasattr(self.dem, "is_lsmpm_soft_affine_ipc") and self.dem.is_lsmpm_soft_affine_ipc():
                self.dem.solver.Solver(self.dem.scene)
                self.dem.first_run = False
                return
            requested_timestep = float(self.dem.sims.dt[None])
            self.dem.check_critical_timestep()
            actual_timestep = float(self.dem.sims.dt[None])
            tolerance = max(1.0e-15, 1.0e-12 * abs(requested_timestep))
            if strict_timestep and not np.isclose(
                actual_timestep,
                requested_timestep,
                rtol=0.0,
                atol=tolerance,
            ):
                raise RuntimeError(
                    "Strict LSMPM timestep gate rejected an automatic correction: "
                    f"requested={requested_timestep:.17g}, "
                    f"actual={actual_timestep:.17g}"
                )
            if visualize is False:
                self.dem.solver.Solver(self.dem.scene)
            else:
                self.dem.solver.Visualize(self.dem.scene)
            self.dem.first_run = False
        finally:
            self.dem.sims.set_dem_coupling(original_coupling)

    def run(self, **kwargs):
        if self.is_dem_lsm_pm_soft_rigid_mode():
            self.run_dem_lsm_pm_soft_rigid(**kwargs)
            return
        if self.is_direct_mpm_affine_ipc():
            return self.run_direct_mpm_affine_ipc(**kwargs)
        if self.is_incompressible_mpm_affine_body():
            return self.run_incompressible_mpm_affine_body(**kwargs)
        self.add_essentials(**kwargs)
        self.check_critical_timestep()
        self.solver.CouplingSolver(self.mpm.scene, self.dem.scene)

    def check_critical_timestep(self):
        print("#", " Check Timestep ... ...".ljust(67))
        dem_critical_timestep = self.dem.get_critical_timestep()
        mpm_critical_timestep = (
            self.mpm.scene.get_critical_timestep() if self.mpm.sims.solver_type == "Explicit" else np.inf
        )
        dempm_critical_timestep = self.get_critical_timestep() if self.sims.coupling_scheme != "CFDEM" else np.inf
        if self.sims.dem_timestep < self.sims.delta:
            stable_dem_timestep = self.sims.CFL * dem_critical_timestep
            if stable_dem_timestep < self.sims.dem_timestep:
                print("The DEM substep is corrected as:", stable_dem_timestep, "\n")
                self.sims.set_dem_timestep(stable_dem_timestep)
            outer_critical_timestep = min(mpm_critical_timestep, dempm_critical_timestep)
            if self.sims.CFL * outer_critical_timestep < self.sims.dt[None]:
                outer_timestep = self.sims.CFL * outer_critical_timestep
                print("The coupled time step is corrected as:", outer_timestep, "\n")
                self.sims.set_timestep(outer_timestep)
                self.mpm.sims.set_timestep(outer_timestep)
            else:
                print("The prescribed coupled time step is sufficiently small\n")
            self.dem.sims.set_timestep(self.sims.dem_timestep)
            self.sims.init_delta = self.sims.delta
            self.mpm.sims.init_delta = self.sims.delta
            self.dem.sims.init_delta = self.sims.dem_timestep
        else:
            critical_timestep = min(dem_critical_timestep, mpm_critical_timestep, dempm_critical_timestep)
            self.sims.update_critical_timestep(self.mpm.sims, self.dem.sims, critical_timestep)

    def get_critical_timestep(self):
        return self.contactor.physpp.calcu_critical_timesteps(
            self.mpm.scene, self.dem.sims, self.dem.scene, self.sims.max_material_num
        )

    def save_data(self):
        if self.solver is None:
            self.add_essentials()
        self.solver.save_file(self.mpm.scene, self.dem.scene)

    def postprocessing(self, start_file=0, end_file=-1, read_path=None, write_path=None, scheme=None, **kwargs):
        self.dem.postprocessing(start_file, end_file, read_path, write_path, scheme, **kwargs)
        self.mpm.postprocessing(start_file, end_file, read_path, write_path, **kwargs)
