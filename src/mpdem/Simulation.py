import taichi as ti
import math
import warnings

from src.dem.Simulation import Simulation as DEMSimulation
from src.mpm.Simulation import Simulation as MPMSimulation
from src.utils.constants import Threshold
from src.utils.TypeDefination import vec3f
from src.utils.TimeTicker import Timer


class Simulation(object):
    def __init__(self) -> None:
        self.domain = vec3f(0.0, 0.0, 0.0)
        self.coupling_scheme = "MPDEM"
        self.cfdem_resolution = "Auto"
        self.particle_interaction = True
        self.wall_interaction = False
        self.enhanced_coupling = False
        self.is_continue = False
        self.history_contact_path = {}
        self.infludence_domain = 0
        self.dependent_domain = 0
        self.timer = Timer()
        self.monitor_type = []

        self.dt = ti.field(float, shape=())
        self.delta = 0.0
        self.dem_timestep = 0.0
        self.init_delta = 0.0
        self.current_time = 0.0
        self.current_step = 0
        self.current_print = 0
        self.CurrentTime = ti.field(float, shape=())

        self.time = 0.0
        self.CFL = 0.2
        self.adaptive_timestep = 0
        self.enable_step_retry = False
        self.step_retry_max_retries = 2
        self.step_retry_reduction = 0.5
        self.step_retry_minimum_timestep = 0.0
        self.save_interval = 1e6
        self.visualize_interval = 0.0
        self.window_size = 1024
        self.path = None

        self.min_bounding_rad = 0.0
        self.max_bounding_rad = 0.0

        self.max_material_num = 0.0
        self.body_coordination_number = 15
        self.wall_coordination_number = 10
        self.potential_particle_num = 0.0
        self.max_potential_particle_pairs = 0.0
        self.max_potential_wall_pairs = 0.0
        self.compaction_ratio = [0.4, 0.3]

        self.particle_contact_list_length = 0
        self.wall_contact_list_length = 0
        self.particle_contact_list_capacity = 0
        self.wall_contact_list_capacity = 0

        self.particle_particle_contact_model = None
        self.particle_wall_contact_model = None
        self.digital_elevation_contact_mode = "heightfield"

    def set_domain(self, domain):
        self.domain = domain

    def set_coupling_scheme(self, coupling_scheme):
        valid_scheme = ["MPDEM", "DEMPM", "MPM", "DEM", "CFDEM"]
        if not coupling_scheme in valid_scheme:
            raise RuntimeError(
                f"KeyWord:: /CouplingScheme/ {coupling_scheme} is invalid. Only the followings are valid: {valid_scheme}"
            )
        self.coupling_scheme = coupling_scheme

    def set_cfdem_resolution(self, resolution):
        aliases = {
            "auto": "Auto",
            "semiresolved": "SemiResolved",
            "fullyresolved": "FullyResolved",
        }
        key = str(resolution).replace("-", "").replace("_", "").replace(" ", "").lower()
        if key not in aliases:
            raise RuntimeError("cfdem_resolution must be 'Auto', 'SemiResolved', or 'FullyResolved'")
        self.cfdem_resolution = aliases[key]

    def set_particle_interaction(self, particle_interaction):
        self.particle_interaction = particle_interaction

    def set_wall_interaction(self, wall_interaction):
        self.wall_interaction = wall_interaction

    def set_CFD_coupling_domain(self, coupling_domain):
        self.infludence_domain = int(coupling_domain[0])
        self.dependent_domain = int(coupling_domain[1])

    def set_enhanced_coupling(self, enhanced_coupling):
        if not isinstance(enhanced_coupling, bool):
            raise ValueError("KeyWord:: /EnhancedCoupling/ should be a boolean value")
        self.enhanced_coupling = enhanced_coupling

    def set_digital_elevation_contact_mode(self, contact_mode):
        if contact_mode is not None and str(contact_mode).lower() != "heightfield":
            warnings.warn("DEMPM DigitalElevation contact is forced to heightfield; the requested mode is ignored")
        self.digital_elevation_contact_mode = "heightfield"

    def use_digital_elevation_heightfield(self, dsims: DEMSimulation):
        return self.wall_interaction and dsims.wall_type == 3 and self.digital_elevation_contact_mode == "heightfield"

    def set_timestep(self, timestep):
        self.dt[None] = timestep
        self.delta = timestep

    def set_dem_timestep(self, timestep):
        timestep = float(timestep)
        if not math.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("KeyWord:: /DEMTimestep/ should be finite and positive")
        if self.delta > 0.0 and timestep > self.delta:
            raise ValueError("KeyWord:: /DEMTimestep/ cannot exceed /Timestep/")
        self.dem_timestep = timestep

    def set_simulation_time(self, time):
        self.time = time

    def set_CFL(self, CFL):
        self.CFL = CFL

    def set_adaptive_timestep(self, adaptive_timestep):
        self.adaptive_timestep = int(adaptive_timestep)

    def set_save_interval(self, save_interval):
        self.save_interval = save_interval

    def set_visualize_interval(self, visualize_interval):
        self.visualize_interval = visualize_interval

    def set_window_size(self, window_size):
        self.window_size = window_size

    def set_save_path(self, path):
        self.path = path

    def set_is_continue(self, is_continue):
        self.is_continue = is_continue

    def set_material_num(self, max_material_num):
        if max_material_num <= 0:
            raise ValueError("KeyWord:: /max_material_num/ should be larger than 0")
        self.max_material_num = max_material_num

    def set_body_coordination_number(self, body_coordination_number):
        self.body_coordination_number = body_coordination_number

    def set_wall_coordination_number(self, wall_coordination_number):
        self.wall_coordination_number = wall_coordination_number

    def set_compaction_ratio(self, compaction_ratio):
        if isinstance(compaction_ratio, (float, int)):
            compaction_ratio = [compaction_ratio, compaction_ratio]
        self.compaction_ratio = compaction_ratio

    def set_particle_contact_list_capacity(self, capacity):
        if capacity is None:
            capacity = 0
        capacity = int(capacity)
        if capacity < 0:
            raise ValueError("KeyWord:: /max_particle_contact_pairs/ should be non-negative")
        self.particle_contact_list_capacity = capacity

    def set_wall_contact_list_capacity(self, capacity):
        if capacity is None:
            capacity = 0
        capacity = int(capacity)
        if capacity < 0:
            raise ValueError("KeyWord:: /max_wall_contact_pairs/ should be non-negative")
        self.wall_contact_list_capacity = capacity

    def set_particle_particle_contact_model(self, model):
        if model is None and self.particle_interaction:
            raise ValueError("DEMPM:: Particle-Particle contact model have not been assigned")
        if self.particle_interaction is False:
            model = None
        self.particle_particle_contact_model = model

    def set_particle_wall_contact_model(self, model):
        if model is None and self.wall_interaction:
            raise ValueError("DEMPM:: Particle-Wall contact model have not been assigned")
        if self.wall_interaction is False:
            model = None
        self.particle_wall_contact_model = model

    def update_critical_timestep(self, msims: MPMSimulation, dsims: DEMSimulation, critical_timestep):
        dt = self.CFL * critical_timestep
        if dt < self.dt[None]:
            print("The time step is corrected as:", dt, "\n")
            self.set_timestep(dt)
            msims.set_timestep(dt)
            dsims.set_timestep(dt)
        else:
            dt = self.dt[None]
            print("The prescribed time step is sufficiently small\n")
        self.init_delta = dt
        msims.init_delta = dt
        dsims.init_delta = dt

    def set_potential_list_size(self, msims: MPMSimulation, dsims: DEMSimulation, dem_rad_max, mpm_rad_max):
        self.potential_particle_num = 0
        self.max_potential_particle_pairs = 0
        self.particle_contact_list_length = 0
        particle_radius_sum = dem_rad_max + mpm_rad_max
        if self.particle_interaction and particle_radius_sum > Threshold:
            potential_particle_ratio = (
                (particle_radius_sum + msims.verlet_distance + dsims.verlet_distance) / particle_radius_sum
            ) ** 3
            self.potential_particle_num = int(potential_particle_ratio * self.body_coordination_number)
            self.max_potential_particle_pairs = int(self.potential_particle_num * msims.max_coupling_particle_num)
            self.particle_contact_list_length = int(
                math.ceil(self.compaction_ratio[0] * self.max_potential_particle_pairs)
            )
            if self.particle_contact_list_capacity > 0:
                if self.particle_contact_list_capacity > self.max_potential_particle_pairs:
                    warnings.warn(
                        "KeyWord:: /max_particle_contact_pairs/ is larger than the "
                        "allocated potential particle-particle pairs and may waste GPU memory"
                    )
                self.particle_contact_list_length = self.particle_contact_list_capacity
        if dsims.max_wall_num > 0 and self.wall_interaction:
            if self.use_digital_elevation_heightfield(dsims):
                self.max_potential_wall_pairs = 0
                self.wall_contact_list_length = 0
                return
            self.max_potential_wall_pairs = int(self.wall_coordination_number * msims.max_coupling_particle_num)
            self.wall_contact_list_length = int(math.ceil(self.compaction_ratio[1] * self.max_potential_wall_pairs))
            if self.wall_contact_list_capacity > 0:
                if self.wall_contact_list_capacity > self.max_potential_wall_pairs:
                    warnings.warn(
                        "KeyWord:: /max_wall_contact_pairs/ is larger than the "
                        "allocated potential particle-wall pairs and may waste GPU memory"
                    )
                self.wall_contact_list_length = self.wall_contact_list_capacity

    def set_bounding_sphere(self, rad_min, rad_max):
        self.min_bounding_rad = rad_min
        self.max_bounding_rad = rad_max

    def set_save_data(self, ppcontact, pwcontact):
        if ppcontact:
            self.monitor_type.append("ppcontact")
        if pwcontact:
            self.monitor_type.append("pwcontact")

    def validate_configuration(self):
        if self.enhanced_coupling and self.coupling_scheme == "CFDEM":
            raise RuntimeError("enhanced_coupling is not used by CFDEM coupling")

    def _is_incompressible_fluid_fdm(self, msims: MPMSimulation):
        return msims.solver_type == "Implicit" and msims.material_type == "Fluid" and msims.discretization == "FDM"

    def _is_two_phase_double_layer_lsdem(self, msims: MPMSimulation, dsims: DEMSimulation):
        return (
            msims.solver_type == "SemiImplicit"
            and msims.material_type == "TwoPhaseDoubleLayer"
            and msims.dimension == 3
            and dsims.scheme == "LSDEM"
        )

    def validate_coupling_configuration(self, msims: MPMSimulation, dsims: DEMSimulation):
        self.validate_configuration()
        direct_affine_ipc = (
            dsims.scheme == "AffineBody"
            and msims.is_direct_backend()
            and msims.solver_type == "Implicit"
            and msims.ipc_contact
        )
        if direct_affine_ipc:
            if msims.dimension != 3:
                raise RuntimeError("Direct MPM--AffineBody IPC currently supports 3D only")
            return
        incompressible_affine = dsims.scheme == "AffineBody" and self._is_incompressible_fluid_fdm(msims)
        if incompressible_affine:
            if self.coupling_scheme not in ("MPDEM", "DEMPM"):
                raise RuntimeError(
                    "Incompressible MPM--AffineBody coupling requires coupling_scheme='MPDEM' or 'DEMPM'"
                )
            if msims.dimension != 3:
                raise RuntimeError("Incompressible MPM--AffineBody coupling currently supports 3D only")
            if self.dem_timestep != self.delta:
                raise RuntimeError("Incompressible MPM--AffineBody coupling does not support DEM subcycling")
            if self.enable_step_retry or self.adaptive_timestep:
                raise RuntimeError("Incompressible MPM--AffineBody coupling currently requires a fixed timestep")
            return
        if self.dem_timestep < self.delta and not (
            self.coupling_scheme == "CFDEM"
            and dsims.scheme in ("DEM", "LSDEM")
            and self._is_incompressible_fluid_fdm(msims)
        ):
            raise RuntimeError("DEMTimestep subcycling requires incompressible CFDEM with DEM scheme='DEM' or 'LSDEM'")
        if self.coupling_scheme in ("MPDEM", "DEMPM"):
            if self._is_two_phase_double_layer_lsdem(msims, dsims):
                if self.dem_timestep != self.delta:
                    raise RuntimeError("TwoPhaseDoubleLayer LSDEM coupling currently requires DEMTimestep == Timestep")
                return
            if getattr(msims, "soft_particle", False):
                if dsims.scheme != "LSDEM":
                    raise RuntimeError(
                        "DEMPM soft_particle coupling requires DEM scheme='LSDEM' " "for level-set rigid bodies."
                    )
                raise RuntimeError(
                    "DEMPM level-set DEM + MPM soft_particle coupling is staged "
                    "structurally, but not runnable until the soft body scene, "
                    "generator, contact-node fields, and recorder are migrated "
                    "from src.dem into src.mpm.soft_particle. The current runnable "
                    "path remains DEM scheme='LSMPM'."
                )
            if dsims.scheme not in ("DEM", "LSDEM"):
                raise RuntimeError(f"{self.coupling_scheme} currently supports DEM scheme='DEM' " "or scheme='LSDEM'")
            if msims.solver_type == "Implicit":
                if not self._is_incompressible_fluid_fdm(msims):
                    raise RuntimeError(
                        f"{self.coupling_scheme} with Implicit MPM currently "
                        "supports only incompressible Fluid FDM coupling"
                    )
                if msims.dimension != 3:
                    raise RuntimeError("Implicit incompressible DEM-MPM coupling currently supports 3D only")
            elif msims.solver_type in ("SemiImplicit", "SemiImplicit_u_p"):
                raise RuntimeError(f"{self.coupling_scheme} does not support {msims.solver_type} MPM coupling")
        elif self.coupling_scheme == "CFDEM":
            if msims.sparse_grid:
                raise RuntimeError("CFDEM does not support MPM sparse_grid yet")
            if dsims.scheme not in ("DEM", "LSDEM"):
                raise RuntimeError("CFDEM currently supports DEM scheme='DEM' or scheme='LSDEM'")
            if not self._is_incompressible_fluid_fdm(msims):
                raise RuntimeError(
                    "CFDEM requires incompressible semi-implicit MPM "
                    "(material_type='Fluid', solver_type='Implicit', discretization='FDM')"
                )
            if msims.dimension != 3:
                raise RuntimeError("Incompressible CFDEM currently supports 3D only")
            if self.cfdem_resolution == "FullyResolved" and dsims.scheme != "LSDEM":
                raise RuntimeError(
                    "FullyResolved incompressible CFDEM requires DEM scheme='LSDEM'; "
                    "pure DEM Sphere/Clump bodies are not fully resolved and must be represented as LSDEM SDF bodies"
                )
            if self.cfdem_resolution == "SemiResolved" and dsims.scheme != "DEM":
                raise RuntimeError("SemiResolved incompressible CFDEM requires DEM scheme='DEM'")
        elif self.coupling_scheme == "DEM":
            if dsims.scheme not in ("DEM", "LSDEM"):
                raise RuntimeError("coupling_scheme='DEM' supports DEM scheme='DEM' or scheme='LSDEM'")
