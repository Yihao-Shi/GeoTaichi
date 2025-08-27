import numpy as np
import taichi as ti
from taichi.lang.impl import current_cfg

from src.dem.SceneManager import myScene
from src.dem.GenerateManager import GenerateManager
from src.dem.ContactManager import ContactManager
from src.dem.engines.ExplicitEngine import ExplicitEngine
from src.dem.DEMBase import Solver
from src.dem.PostPlot import write_dem_vtk_file, write_lsdem_vtk_file
from src.dem.Recorder import WriteFile
from src.dem.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefination import vec3f


class DEM(object):
    def __init__(self, title='A High Performance Multiscale and Multiphysics Simulator', log=True):
        if log:
            print('# =================================================================== #')
            print('#', "".center(67), '#')
            print('#', "Welcome to GeoTaichi -- Discrete Element Method Engine !".center(67), '#')
            print('#', "".center(67), '#')
            print('#', title.center(67), '#')
            print('#', "".center(67), '#')
            print('# =================================================================== #', '\n')
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
        self.sims.set_boundary(DictIO.GetAlternative(kwargs, "boundary" ,[None, None, None]))
        self.sims.set_gravity(DictIO.GetAlternative(kwargs, "gravity", vec3f([0.,0.,-9.8])))
        self.sims.set_engine(DictIO.GetAlternative(kwargs, "engine", "SymplecticEuler"))
        self.sims.set_search(DictIO.GetAlternative(kwargs, "search", "LinkedCell"))
        self.sims.set_search_direction(DictIO.GetAlternative(kwargs, "search_direction", "Up"))
        self.sims.set_dem_scheme(DictIO.GetAlternative(kwargs, "scheme", "DEM"))
        self.sims.set_track_energy(DictIO.GetAlternative(kwargs, "track_energy", False))
        self.sims.set_visualize(DictIO.GetAlternative(kwargs, "visualize", True))
        self.sims.set_iterative_model(DictIO.GetAlternative(kwargs, "iterative_model", "LagrangianMultiplier"))
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
        if log: 
            self.print_solver_info()
            print('\n')

    def memory_allocate(self, memory, log=True):
        self.sims.set_material_num(DictIO.GetEssential(memory, "max_material_number"))
        if self.sims.scheme == "DEM":
            self.sims.set_particle_num(DictIO.GetEssential(memory, "max_particle_number"))
            self.sims.set_sphere_num(DictIO.GetAlternative(memory, "max_sphere_number", 0))
            self.sims.set_clump_num(DictIO.GetAlternative(memory, "max_clump_number", 0))
        else:
            self.sims.set_rigid_body_num(DictIO.GetEssential(memory, "max_rigid_body_number"))
            self.sims.set_rigid_template_num(DictIO.GetAlternative(memory, "max_rigid_template_number", 1))
            if self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                self.sims.set_level_grid_num(DictIO.GetEssential(memory, "levelset_grid_number"))
                self.sims.set_surface_node_num(DictIO.GetEssential(memory, "surface_node_number"))
                self.sims.set_point_coordination_number(DictIO.GetAlternative(memory, "point_coordination_number", [4, 2]))
                if self.sims.scheme == "LSMPM":
                    self.sims.set_soft_body_num(DictIO.GetEssential(memory, "max_soft_body_number"))
                    self.sims.set_material_point_num(DictIO.GetEssential(memory, "max_material_point_number"))
        self.sims.set_patch_num(DictIO.GetAlternative(memory, "max_patch_number", 0))
        self.sims.set_facet_num(DictIO.GetAlternative(memory, "max_facet_number", 0))
        self.sims.set_servo_wall_num(DictIO.GetAlternative(memory, "max_servo_wall_number", 0))
        self.sims.set_plane_num(DictIO.GetAlternative(memory, "max_plane_number", 0))
        self.sims.set_digital_elevation_facet_num(DictIO.GetAlternative(memory, "max_digital_elevation_facet_number", 0))
        self.sims.set_compaction_ratio(DictIO.GetAlternative(memory, "compaction_ratio", [0.15, 0.05]))
        self.sims.set_hierarchical_level(DictIO.GetAlternative(memory, "hierarchical_level", 1))
        self.sims.set_rebuild_interval(DictIO.GetAlternative(memory, "bvh_rebuild_interval", 1000))
        if self.sims.search == "HierarchicalLinkedCell":
            self.sims.set_hierarchical_size(DictIO.GetEssential(memory, "hierarchical_size"))
        self.sims.define_work_load()

        self.sims.set_body_coordination_number(DictIO.GetAlternative(memory, "body_coordination_number", 16))
        self.sims.set_wall_coordination_number(DictIO.GetAlternative(memory, "wall_coordination_number", self.sims.max_wall_num))
        self.sims.set_verlet_distance_multiplier(DictIO.GetAlternative(memory, "verlet_distance_multiplier", 0))
        self.sims.set_wall_per_cell(DictIO.GetAlternative(memory, "wall_per_cell", 4))
        
        self.scene.activate_basic_class(self.sims)
        if log: 
            self.print_simulation_info()
            print('\n')

    def print_basic_simulation_info(self):
        print(" DEM Basic Configuration ".center(71,"-"))
        print(("Simulation Type: " + str(current_cfg().arch)).ljust(67))
        print(("Simulation Domain: " + str(self.sims.domain)).ljust(67))
        print(("Boundary Condition: " + str(self.sims.boundary)).ljust(67))
        print(("Gravity: " + str(self.sims.gravity)).ljust(67))
    
    def print_simulation_info(self):
        print(" DEM Engine Information ".center(71,"-"))
        print(("Engine Type: " + str(self.sims.engine)).ljust(67))
        print(("Neighbor Search Type: " + str(self.sims.search)).ljust(67))
        print(("DEM Scheme: " + str(self.sims.scheme)).ljust(67))
        if self.sims.energy_tracking:
            print("Energy tracking: ON")

    def print_solver_info(self):
        print(" DEM Solver Information ".center(71,"-"))
        print(("Initial Simulation Time: " + str(self.sims.current_time)).ljust(67))
        print(("Finial Simulation Time: " + str(self.sims.current_time + self.sims.time)).ljust(67))
        print(("Time Step: " + str(self.sims.dt[None])).ljust(67))
        print(("Save Interval: " + str(self.sims.save_interval)).ljust(67))
        print(("Save Path: " + str(self.sims.path)).ljust(67))

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
        self.generator.add_my_template(self.scene, template, types)
        if self.sims.max_particle_num > 0:
            if (types == "LSDEM" or types == "LSMPM"):
                self.scene.add_rigid_template_grid_field(self.sims)
            elif (types == "PolySuperEllipsoid" or types == "PolySuperQuadrics"):
                self.scene.add_rigid_implicit_surface_parameter(self.sims)
        
    def create_body(self, body):
        self.generator.create_body(body, self.sims, self.scene)

    def add_body(self, body):
        self.generator.add_body(body, self.sims, self.scene)

    def add_body_from_file(self, body):
        self.generator.read_body_file(body, self.sims, self.scene)

    def add_wall(self, body):
        self.generator.add_wall(body, self.sims, self.scene)

    def add_wall_from_file(self, body):
        self.generator.read_wall_file(body, self.sims, self.scene)

    def choose_neighbor(self):
        if self.contactor is None:
            self.contactor = ContactManager()
            self.contactor.choose_neighbor(self.sims, self.scene)

    def choose_contact_model(self, particle_particle_contact_model=None, particle_wall_contact_model=None):
        self.choose_neighbor()
        if self.sims.max_material_num == 0:
            raise RuntimeError("memory_allocate should be launched first!")
        self.sims.set_particle_particle_contact_model(particle_particle_contact_model)
        self.sims.set_particle_wall_contact_model(particle_wall_contact_model)
        self.contactor.particle_particle_initialize(self.sims)
        self.contactor.particle_wall_initialize(self.sims)

    def add_property(self, materialID1, materialID2, property, dType="all"):
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

    def select_save_data(self, particle=True, sphere=False, clump=False, surface=True, grid=False, bounding=False, wall=False, 
                         particle_particle_contact=False, particle_wall_contact=False):
        self.sims.set_save_data(particle, sphere, clump, surface, grid, bounding, wall, particle_particle_contact, particle_wall_contact)
        self.scene.activate_surface_node_visualization(self.sims)

    def read_restart(self, file_number, file_path, particle=True, sphere=True, clump=False, wall=True, servo=False, ppcontact=True, pwcontact=True, is_continue=True):
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
            particle_path = file_path+f"/particles/DEMParticle{file_number:06d}.npz"
            if sphere:
                sphere_path = file_path+f"/particles/DEMSphere{file_number:06d}.npz"
            if clump:
                clump_path = file_path+f"/particles/DEMClump{file_number:06d}.npz"
            if sphere is False and clump is False:
                raise RuntimeError("sphere or clump file is not exist")
        if wall:
            wall_path = file_path+f"/walls/DEMWall{file_number:06d}.npz"
            if servo:
                servo_path = file_path+f"/walls/DEMServo{file_number:06d}.npz"
        if ppcontact:
            ppcontact_path = file_path+"/contacts"
        if pwcontact:
            pwcontact_path = file_path+"/contacts"
        
        if particle:
            self.add_body_from_file(body={"FileType": "NPZ", "Template": {"Restart": True, "ParticleFile": particle_path, "SphereFile": sphere_path, "ClumpFile": clump_path}})
        if wall:
            self.add_wall_from_file(body={"FileType": "NPZ", "WallFile": wall_path, "ServoFile": servo_path})
        self.sims.history_contact_path.update(file_number=file_number, ppcontact=ppcontact_path, pwcontact=pwcontact_path)

    def modify_parameters(self, **kwargs):
        if len(kwargs) > 0:
            self.sims.set_simulation_time(DictIO.GetEssential(kwargs, "SimulationTime"))
            if "Timestep" in kwargs: self.sims.set_timestep(DictIO.GetEssential(kwargs, "Timestep"))
            if "CFL" in kwargs: self.sims.set_CFL(DictIO.GetEssential(kwargs, "CFL"))
            if "AdaptiveTimestep" in kwargs: self.sims.set_adaptive_timestep(DictIO.GetEssential(kwargs, "AdaptiveTimestep"))
            if "SaveInterval" in kwargs: self.sims.set_save_interval(DictIO.GetEssential(kwargs, "SaveInterval"))
            if "SavePath" in kwargs: self.sims.set_save_path(DictIO.GetEssential(kwargs, "SavePath"))
            if "gravity" in kwargs: self.sims.set_gravity(DictIO.GetEssential(kwargs, "gravity"))

    def add_engine(self, callback):
        if self.enginer is None:
            self.enginer = ExplicitEngine(self.scene, self.contactor)
        self.enginer.choose_engine(self.sims, self.scene)
        self.enginer.set_servo_mechanism(self.sims, callback)

    def add_recorder(self):
        if self.recorder is None:
            self.recorder = WriteFile(self.sims, self.contactor.physpp, self.contactor.physpw, self.contactor.neighbor)

    def add_solver(self, kwargs):
        if self.solver is None:
            self.solver = Solver(self.sims, self.generator, self.contactor, self.enginer, self.recorder)
        self.solver.set_callback_function(DictIO.GetAlternative(kwargs, "function", None))
        self.solver.set_particle_calm(self.scene, DictIO.GetAlternative(kwargs, "calm", None))

    def add_postfunctions(self, **functions):
        self.solver.set_callback_function(functions)

    def add_essentials(self, kwargs):
        if self.contactor is None:
            self.choose_contact_model()
        if self.sims.max_particle_num >= 0:
            if self.contactor.have_initialise is False:
                self.contactor.initialize(self.sims, self.scene, kwargs)
        else:
            if self.sims.coupling is False:
                raise RuntimeError("Particle should be added first")
        self.load_history_contact()
        self.add_engine(DictIO.GetAlternative(kwargs, "callback", None))
        self.add_recorder()
        if self.sims.coupling == False:
            self.add_solver(kwargs)
        if self.sims.scheme == "LSDEM":
            self.check_verlet_distance_multiplier()
            self.scene.check_radius()
        self.scene.set_boundary_condition(self.sims)

    def check_verlet_distance_multiplier(self):
        equivalent_rad = self.scene.find_particle_min_radius(scheme="LSDEM")
        self.sims.set_point_verlet_distance(equivalent_rad)
        self.sims.check_multiplier(max(self.contactor.physpp.find_max_penetration(), self.contactor.physpw.find_max_penetration()))
        self.sims.check_grid_extent(*self.scene.find_expect_extent(self.sims, self.sims.point_verlet_distance))

    def static_wall(self, static_wall=True):
        self.sims.set_static_wall(static_wall)

    def servo_switch(self, status="On"):
        self.sims.update_servo_status(status)

    def set_window(self, window):
        self.sims.set_window_parameters(window)

    def run(self, visualize=False, **kwargs):
        self.add_essentials(kwargs)
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
            self.sims.update_critical_timestep(self.sims.CFL * critical_timestep)
        else:
            print("The prescribed time step is sufficiently small\n")

    def get_critical_timestep(self):
        return self.contactor.physpp.calcu_critical_timesteps(self.scene)
    
    def update_material_properties(self, materialID, property_name, value, override=True):
        self.scene.update_material_properties(override, materialID, property_name, value)
    
    def update_particle_properties(self, property_name, value, override=True, bodyID=None, region_name=None, function=None):
        if not bodyID is None:
            self.scene.update_particle_properties(override, property_name, value, bodyID)
        elif not region_name is None:
            region: RegionFunction = self.generator.get_region_ptr(region_name)
            self.scene.update_particle_properties_in_region(self.sims, override, property_name, value, region.function)
        elif not function is None:
            self.scene.update_particle_properties_in_region(self.sims, override, property_name, value, ti.pyfunc(function))
        if not self.first_run:
            self.contactor.neighbor.pre_neighbor(self.scene)

    def update_wall_status(self, wallID, property_name, value, override=True):
        self.scene.update_wall_properties(self.sims, override, property_name, value, wallID)
        if not self.first_run:
            self.contactor.neighbor.update_verlet_table(self.scene)

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

    def postprocessing(self, start_file=0, end_file=-1, read_path=None, write_path=None, scheme=None, **kwargs):
        if read_path is None:
            read_path = self.sims.path
            if write_path is None:
                write_path = self.sims.path+"/vtks"
            elif not write_path is None:
                write_path = read_path + "/vtks"
        
        if not read_path is None and write_path is None:
            write_path = read_path + "/vtks"

        if not write_path.endswith('vtks'): write_path = write_path + '/vtks'

        scheme = self.sims.scheme if scheme is None else scheme
        self.sims.set_dem_scheme(scheme)

        if self.sims.scheme == "DEM":
            write_dem_vtk_file(self.sims, start_file, end_file, read_path, write_path, kwargs)
        elif self.sims.scheme == "LSDEM":
            write_lsdem_vtk_file(self.sims, start_file, end_file, read_path, write_path, kwargs)
