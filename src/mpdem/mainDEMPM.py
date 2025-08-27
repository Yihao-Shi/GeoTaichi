import numpy as np
from taichi.lang.impl import current_cfg

from src.dem.mainDEM import DEM
from src.mpm.mainMPM import MPM

from src.mpdem.ContactManager import ContactManager
from src.mpdem.DEMPMBase import Solver
from src.mpdem.Engine import Engine
from src.mpdem.GenerateManager import GenerateManager
from src.mpdem.Recorder import WriteFile
from src.mpdem.Simulation import Simulation
from src.utils.ObjectIO import DictIO


class DEMPM(object):
    def __init__(self, dem: DEM, mpm: MPM, title='A High Performance Multiscale and Multiphysics Simulator on GPU', log=True):
        if log:
            print('# =================================================================== #')
            print('#', "".center(67), '#')
            print('#', "Welcome to GeoTaichi -- DEM & MPM Coupling Engine !".center(67), '#')
            print('#', "".center(67), '#')
            print('#', title.center(67), '#')
            print('#', "".center(67), '#')
            print('# =================================================================== #', '\n')
        self.dem = dem
        self.mpm = mpm
        self.sims = Simulation()
        self.generator = GenerateManager(self.mpm.generator, self.dem.generator)
        self.contactor = None
        self.enginer = None
        self.solver = None
        self.recorder = None

    def set_configuration(self, log=True, **kwargs):
        if np.linalg.norm(np.array(self.mpm.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10 and \
            np.linalg.norm(np.array(self.dem.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10:
            domain = DictIO.GetEssential(kwargs, "domain")
            self.sims.set_domain(domain)
            self.dem.sims.set_domain(domain)
            self.mpm.sims.set_domain(domain)
        elif np.linalg.norm(np.array(self.mpm.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10 and \
            np.linalg.norm(np.array(self.dem.sims.get_simulation_domain()) - np.zeros(3)) > 1e-10:
            domain = self.dem.sims.get_simulation_domain()
            self.sims.set_domain(domain)
            self.mpm.sims.set_domain(domain)
        elif np.linalg.norm(np.array(self.mpm.sims.get_simulation_domain()) - np.zeros(3)) > 1e-10 and \
            np.linalg.norm(np.array(self.dem.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10:
            domain = self.mpm.sims.get_simulation_domain()
            self.sims.set_domain(domain)
            self.dem.sims.set_domain(domain)
        elif np.linalg.norm(np.array(self.mpm.sims.get_simulation_domain()) - np.zeros(3)) > 1e-10 and \
            np.linalg.norm(np.array(self.dem.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10:
            if not all(self.mpm.sims.get_simulation_domain() == self.dem.sims.get_simulation_domain()):
                raise RuntimeError(f"DEM simulation domain {self.dem.sims.get_simulation_domain()} is not in line with MPM simulation domain {self.mpm.sims.get_simulation_domain()}")
            else:
                self.sims.set_domain(self.mpm.sims.get_simulation_domain())
        
        self.sims.set_coupling_scheme(DictIO.GetAlternative(kwargs, "coupling_scheme", "MPDEM"))
        self.sims.set_particle_interaction(DictIO.GetAlternative(kwargs, "particle_interaction", True))
        self.sims.set_wall_interaction(DictIO.GetAlternative(kwargs, "wall_interaction", False))
        self.mpm.sims.set_gravity(DictIO.GetAlternative(kwargs, "gravity", [0., 0., -9.8]))
        self.dem.sims.set_gravity(DictIO.GetAlternative(kwargs, "gravity", [0., 0., -9.8]))
        self.sims.set_CFD_coupling_domain(DictIO.GetAlternative(kwargs, "CFD_coupling_domain", [3, 6]))
        self.sims.set_enhanced_coupling(DictIO.GetAlternative(kwargs, "enhanced_coupling", False))
        if self.sims.enhanced_coupling:
            self.mpm.sims.set_norm_adaptivity(True)
        
        if self.mpm.sims.coupling is False:
            raise RuntimeError(f"KeyWord::: /coupling/ should be activated in MPM")
        
        if self.dem.sims.coupling is False:
            raise RuntimeError(f"KeyWord::: /coupling/ should be activated in DEM")
        
        if log: 
            self.print_basic_simulation_info()
            print('\n')

    def set_solver(self, solver, log=True):
        self.sims.set_timestep(DictIO.GetEssential(solver, "Timestep"))
        self.sims.set_simulation_time(DictIO.GetEssential(solver, "SimulationTime"))
        self.sims.set_CFL(DictIO.GetAlternative(solver, "CFL", 0.5))
        self.sims.set_adaptive_timestep(DictIO.GetAlternative(solver, "AdaptiveTimestep", False))
        self.sims.set_save_interval(DictIO.GetEssential(solver, "SaveInterval"))
        self.sims.set_save_path(DictIO.GetAlternative(solver, "SavePath", 'OutputData'))
        self.mpm.set_solver(solver, log=False)
        self.dem.set_solver(solver, log=False)
        if log: 
            self.print_solver_info()
            print('\n')

    def memory_allocate(self, memory, dem_memory=None, mpm_memory=None):
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

    def print_basic_simulation_info(self):
        print(" DEMPM Basic Configuration ".center(71,"-"))
        print(("Simulation Type: " + str(current_cfg().arch)).ljust(67))
        print(("Simulation Domain: " + str(self.sims.domain)).ljust(67))

    def print_solver_info(self):
        print(" DEMPM Solver Information ".center(71,"-"))
        print(("Initial Simulation Time: " + str(self.sims.current_time)).ljust(67))
        print(("Finial Simulation Time: " + str(self.sims.current_time + self.sims.time)).ljust(67))
        print(("Time Step: " + str(self.sims.dt[None])).ljust(67))
        print(("Save Interval: " + str(self.sims.save_interval)).ljust(67))
        print(("Save Path: " + str(self.sims.path)).ljust(67))

    def add_body(self, mpm_body=None, dem_particle=None, write_file=False, check_overlap=False):
        if self.sims.coupling_scheme != "LSMPM":
            self.generator.add_mixture(check_overlap, dem_particle, mpm_body, self.sims, self.dem.scene, self.mpm.scene, self.dem.sims, self.mpm.sims)
        else:
            raise RuntimeError("Please use add_LSbody to generate deformable particles")
        if write_file:
            self.dem.add_recorder()
            self.dem.recorder.save_particle(self.dem.sims, self.dem.scene)
            self.dem.recorder.save_sphere(self.dem.sims, self.dem.scene)
            self.dem.recorder.save_clump(self.dem.sims, self.dem.scene)
            self.mpm.add_recorder()
            self.mpm.recorder.save_particle(self.mpm.sims, self.mpm.scene)

    def create_LSbody(self, body):
        self.generator.create_LSbody(body)
    
    def add_LSbody(self, body):
        self.generator.add_LSbody(body)

    def read_file(self, mpm_body, dem_particle):
        self.generator.read_files(dem_particle, mpm_body, self.dem.scene, self.mpm.scene, self.dem.sims, self.mpm.sims)

    def choose_contact_model(self, particle_particle_contact_model, particle_wall_contact_model=None):
        if self.dem.contactor is None:
            self.dem.choose_contact_model()
        if self.mpm.neighbor is None:
            self.mpm.add_spatial_grid()

        if self.contactor is None:
            self.contactor = ContactManager()
            if self.sims.coupling_scheme != "CFDEM":
                self.contactor.choose_neighbor(self.sims, self.mpm.sims, self.dem.sims, self.mpm.neighbor, self.dem.contactor.neighbor)
        
        if self.sims.coupling_scheme != "CFDEM":
            if self.sims.max_material_num == 0:
                raise RuntimeError("memory_allocate should be launched first!")
            self.sims.set_particle_particle_contact_model(particle_particle_contact_model)
            self.sims.set_particle_wall_contact_model(particle_wall_contact_model)
            self.contactor.particle_particle_initialize(self.sims, self.mpm.sims.material_type, self.dem.sims.scheme)
            self.contactor.particle_wall_initialize(self.sims, self.mpm.sims.material_type)

    def add_property(self, DEMmaterial, MPMmaterial, property, dType="all"):
        if self.sims.coupling_scheme != "CFDEM":
            self.contactor.add_contact_property(self.sims, MPMmaterial, DEMmaterial, property, dType)

    def modify_parameters(self, **kwargs):
        if len(kwargs) > 0:
            self.sims.set_simulation_time(DictIO.GetEssential(kwargs, "SimulationTime"))
            if "Timestep" in kwargs: 
                self.sims.set_timestep(DictIO.GetEssential(kwargs, "Timestep"))
                self.mpm.sims.set_timestep(DictIO.GetEssential(kwargs, "Timestep"))
                self.dem.sims.set_timestep(DictIO.GetEssential(kwargs, "Timestep"))
            if "CFL" in kwargs: 
                self.sims.set_CFL(DictIO.GetEssential(kwargs, "CFL"))
                self.mpm.sims.set_CFL(DictIO.GetEssential(kwargs, "CFL"))
                self.dem.sims.set_CFL(DictIO.GetEssential(kwargs, "CFL"))
            if "AdaptiveTimestep" in kwargs: 
                self.sims.set_adaptive_timestep(DictIO.GetEssential(kwargs, "AdaptiveTimestep"))
                self.mpm.sims.set_adaptive_timestep(DictIO.GetEssential(kwargs, "AdaptiveTimestep"))
                self.dem.sims.set_adaptive_timestep(DictIO.GetEssential(kwargs, "AdaptiveTimestep"))
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

    def read_restart(self, file_number, file_path, ppcontact=False, pwcontact=False):
        ppcontact_path = None
        pwcontact_path = None
        if self.dem.sims.is_continue != self.mpm.sims.is_continue:
            raise RuntimeError(f"The continue flag in MPM {self.mpm.sims.is_continue} and DEM {self.dem.sims.is_continue} is different")
        else:
            self.sims.is_continue = self.dem.sims.is_continue
        if self.sims.is_continue:    
            if self.dem.sims.current_print != self.mpm.sims.current_print:
                raise RuntimeError(f"The print in MPM {self.mpm.sims.current_print} and DEM {self.dem.sims.current_print} is different")
            else:
                self.sims.current_print = file_number
            if self.dem.sims.current_time != self.mpm.sims.current_time:
                raise RuntimeError(f"The time in MPM {self.mpm.sims.current_time} and DEM {self.dem.sims.current_time} is different")
            else:
                self.sims.current_time = self.mpm.sims.current_time
                self.sims.CurrentTime[None] = self.mpm.sims.current_time
        if ppcontact:
            ppcontact_path = file_path+"/contacts"
        if pwcontact:
            pwcontact_path = file_path+"/contacts"
        self.sims.history_contact_path.update(file_number=file_number, ppcontact=ppcontact_path, pwcontact=pwcontact_path)

    def load_history_contact(self):
        file_number = DictIO.GetAlternative(self.sims.history_contact_path, "file_number", 0)
        ppcontact = DictIO.GetAlternative(self.sims.history_contact_path, "ppcontact", None)
        pwcontact = DictIO.GetAlternative(self.sims.history_contact_path, "pwcontact", None)

        if not ppcontact is None:
            self.contactor.physpp.restart(self.contactor.neighbor, file_number, ppcontact, True)
        if not pwcontact is None:
            self.contactor.physpw.restart(self.contactor.neighbor, file_number, pwcontact, False)

    def add_essentials(self, kwargs: dict):
        def split_function(dicts, name):
            split_dict = {}
            for keys, values in dicts.items():
                if name in keys:
                    split_dict.update({keys.replace(name, "", 1): values})
            return split_dict
        self.mpm.scene.update_coupling_points_number(self.mpm.sims)
        if self.dem.scene.particleNum[0] > 0 or self.mpm.scene.particleNum[0] > 0:
            if self.sims.coupling_scheme != "CFDEM":
                if self.contactor.have_initialise is False:
                    self.contactor.initialize(self.sims, self.mpm.scene, self.dem.scene)
        else:
            raise RuntimeError("DEM/MPM particle should be added first")
        self.load_history_contact() 
        self.mpm.add_essentials(split_function(kwargs, "mpm_"))
        dem_function = split_function(kwargs, "dem_")
        dem_function.update({"max_bounding_radius": self.sims.max_bounding_rad, "min_bounding_radius": self.sims.min_bounding_rad})
        self.dem.add_essentials(dem_function)

        if self.contactor is None:
            self.choose_contact_model(None, None)

        self.recorder = WriteFile(self.sims, self.mpm.sims, self.dem.sims, self.dem.recorder, self.mpm.recorder, self.contactor.physpp, self.contactor.physpw, self.contactor.neighbor)
        if self.enginer is None:
            self.enginer = Engine(self.sims, self.mpm.sims, self.dem.sims, self.mpm.scene, self.dem.scene, self.mpm.enginer, self.dem.enginer,
                                  self.contactor.neighbor, self.mpm.neighbor, self.dem.contactor.neighbor, self.contactor.physpp, self.contactor.physpw)
        self.enginer.manage_function(DictIO.GetAlternative(kwargs, 'drag_model', {}))

        if self.solver is None:
            self.solver = Solver(self.sims, self.mpm.sims, self.dem.sims, self.mpm.recorder, self.dem.recorder, self.generator, self.enginer, self.recorder)
        self.solver.set_callback_function(DictIO.GetAlternative(kwargs, "function", None))
        self.solver.set_particle_calm(self.dem.scene, DictIO.GetAlternative(kwargs, "calm", None))

    def add_postfunctions(self, **functions):
        self.solver.set_callback_function(functions)

    def update_contact_properties(self, materialID1, materialID2, property_name, value, overide=True):
        self.contactor.update_contact_property(self.sims, materialID1, materialID2, property_name, value, overide)

    def run(self, **kwargs):
        self.add_essentials(kwargs)
        self.check_critical_timestep()
        self.solver.CouplingSolver(self.mpm.scene, self.dem.scene)

    def check_critical_timestep(self):
        print("#", " Check Timestep ... ...".ljust(67))
        dem_critical_timestep = self.dem.get_critical_timestep()
        mpm_critical_timestep = self.mpm.scene.get_critical_timestep()
        dempm_critical_timestep = self.get_critical_timestep() if self.sims.coupling_scheme != "CFDEM" else mpm_critical_timestep
        critical_timestep = min(dem_critical_timestep, mpm_critical_timestep, dempm_critical_timestep)
        if self.sims.CFL * critical_timestep < self.sims.dt[None]:
            self.sims.update_critical_timestep(self.mpm.sims, self.dem.sims, self.sims.CFL * critical_timestep)
        else:
            print("The prescribed time step is sufficiently small\n")

    def get_critical_timestep(self):
        return self.contactor.physpp.calcu_critical_timesteps(self.mpm.scene, self.dem.sims, self.dem.scene, self.sims.max_material_num)
    
