import taichi as ti
import math

from src.dem.Simulation import Simulation as DEMSimulation
from src.mpm.Simulation import Simulation as MPMSimulation
from src.utils.TypeDefination import vec3f
from src.utils.TimeTicker import Timer


class Simulation(object):
    def __init__(self) -> None:
        self.domain = vec3f(0., 0., 0.)
        self.coupling_scheme = "MPDEM"
        self.particle_interaction = True
        self.wall_interaction = False
        self.enhanced_coupling = False
        self.is_continue = True
        self.history_contact_path = {}
        self.infludence_domain = 0
        self.dependent_domain = 0
        self.timer = Timer()

        self.dt = ti.field(float, shape=())
        self.delta = 0.
        self.current_time = 0.
        self.current_step = 0
        self.current_print = 0
        self.CurrentTime = ti.field(float, shape=())

        self.time = 0.
        self.CFL = 0.2
        self.isadaptive = False
        self.save_interval = 1e6
        self.visualize_interval = 0.
        self.window_size = 1024
        self.path = None

        self.min_bounding_rad = 0.
        self.max_bounding_rad = 0.

        self.max_material_num = 0.
        self.body_coordination_number = 15
        self.wall_coordination_number = 10
        self.potential_particle_num = 0.
        self.max_potential_particle_pairs = 0.
        self.max_potential_wall_pairs = 0.
        self.compaction_ratio = [0.4, 0.3]

        self.particle_contact_list_length = 0
        self.wall_contact_list_length = 0

        self.particle_particle_contact_model = None
        self.particle_wall_contact_model = None

    def set_domain(self, domain):
        self.domain = domain

    def set_coupling_scheme(self, coupling_scheme):
        valid_scheme = ["MPDEM", "DEMPM", "MPM", "DEM", "CFDEM"]
        if not coupling_scheme in valid_scheme:
            raise RuntimeError(f"KeyWord:: /CouplingScheme/ {coupling_scheme} is invalid. Only the followings are valid: {valid_scheme}")
        self.coupling_scheme = coupling_scheme

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

    def set_timestep(self, timestep):
        self.dt[None] = timestep
        self.delta = timestep

    def set_simulation_time(self, time):
        self.time = time

    def set_CFL(self, CFL):
        self.CFL = CFL

    def set_adaptive_timestep(self, isadaptive):
        self.isadaptive = isadaptive

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

    def set_particle_particle_contact_model(self, model):
        if model is None and self.particle_interaction:
            raise ValueError("DEMPM:: Particle-Particle contact model have not been assigned")
        if self.particle_interaction is False: model = None
        self.particle_particle_contact_model = model

    def set_particle_wall_contact_model(self, model):
        if model is None and self.wall_interaction:
            raise ValueError("DEMPM:: Particle-Wall contact model have not been assigned")
        if self.wall_interaction is False: model = None
        self.particle_wall_contact_model = model

    def update_critical_timestep(self, msims: MPMSimulation, dsims: DEMSimulation, dt):
        print("The time step is corrected as:", dt, '\n')
        self.dt[None] = dt
        msims.dt[None] = dt
        dsims.dt[None] = dt
        self.delta = dt
        msims.delta = dt
        dsims.delta = dt

    def set_potential_list_size(self, msims: MPMSimulation, dsims: DEMSimulation, dem_rad_max, mpm_rad_max):
        potential_particle_ratio = ((dem_rad_max + mpm_rad_max + msims.verlet_distance + dsims.verlet_distance) / (dem_rad_max + mpm_rad_max)) ** 3
        self.potential_particle_num = int(potential_particle_ratio * self.body_coordination_number)
        self.max_potential_particle_pairs = int(self.potential_particle_num * msims.max_coupling_particle_num)
        self.particle_contact_list_length = int(math.ceil(self.compaction_ratio[0] * self.max_potential_particle_pairs))
        if dsims.max_wall_num > 0 and self.wall_interaction:
            self.max_potential_wall_pairs = int(self.wall_coordination_number * msims.max_coupling_particle_num)    
            if dsims.wall_type == 3:
                self.wall_contact_list_length = int(msims.max_coupling_particle_num)
            else:
                self.wall_contact_list_length = int(math.ceil(self.compaction_ratio[1] * self.max_potential_wall_pairs))

    def set_bounding_sphere(self, rad_min, rad_max):
        self.min_bounding_rad = rad_min
        self.max_bounding_rad = rad_max