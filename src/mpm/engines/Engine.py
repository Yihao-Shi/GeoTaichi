from src.mpm.BaseKernel import *
from src.mpm.engines.EngineKernel import *
from src.mpm.SceneManager import myScene


class Engine(object):
    def __init__(self, sims) -> None:
        self.compute = None
        self.compute_stress_strains = None
        self.bulid_neighbor_list = None
        self.apply_traction_constraints = None
        self.apply_absorbing_constraints = None
        self.apply_velocity_constraints = None
        self.apply_friction_constraints = None
        self.apply_reflection_constraints = None

        self.pre_contact_calculate = None
        self.compute_contact_force_ = None
        self.apply_contact_velocity_constraints = None
        self.apply_contact_reflection_constraints = None

        self.stress_smoothing_ = None
        self.compute_nodal_kinematic = None
        self.compute_internal_forces = None
        self.execute_board_serach = None
        self.calculate_interpolation = None
        self.manage_function(sims)

        self.limit = 0.
        
    def choose_engine(self, sims):
        raise NotImplementedError

    def choose_boundary_constraints(self, sims, scene):
        raise NotImplementedError

    def manage_function(self, sims):
        raise NotImplementedError

    def no_operation(self, sims, scene):
        pass

    def no_operation_neighbor(self, sims, scene, neighbor):
        pass

    def reset_particle_message(self, scene: myScene):
        contact_force_reset(scene.particleNum[0], scene.particle)

    def reset_grid_message(self, scene: myScene):
        grid_reset(scene.mass_cut_off, scene.node)

    def compute_nodal_kinematics(self, sims, scene):
        raise NotImplementedError

    def compute_grid_velcity(self, sims, scene):
        raise NotImplementedError

    def compute_stress_strain(self, sims, scene):
        raise NotImplementedError
        
    def compute_external_force(self, sims, scene):
        raise NotImplementedError

    def compute_internal_force(self, sims, scene):
        raise NotImplementedError
    
    def compute_internal_force_bbar(self, sims, scene):
        raise NotImplementedError

    def compute_grid_kinematic(self, sims, scene):
        raise NotImplementedError

    def apply_kinematic_constraints(self, sims, scene):
        raise NotImplementedError

    def apply_dirichlet_constraints(self, sims, scene):
        raise NotImplementedError

    def traction_constraints(self, sims, scene):
        raise NotImplementedError
    
    def absorbing_constraints(self, sims, scene):
        raise NotImplementedError

    def velocity_constraints(self, sims, scene):
        raise NotImplementedError
        
    def compute_particle_kinematics(self, sims, scene):
        raise NotImplementedError

    def postmapping_grid_velocity(self, sims, scene):
        raise NotImplementedError

    def update_verlet_table(self, sims, scene, neighbor):
        raise NotImplementedError

    def is_need_update_verlet_table(self, scene: myScene):
        return validate_particle_displacement_(self.limit, scene.particleNum[0], scene.particle)

    def pre_calculation(self, sims, scene: myScene, neighbor):
        raise NotImplementedError

    def usl_updating(self, sims, scene):
        raise NotImplementedError

    def usf_updating(self, sims, scene):
        raise NotImplementedError

    def musl_updating(self, sims, scene):
        raise NotImplementedError
