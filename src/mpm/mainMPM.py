import numpy as np
from taichi.lang.impl import current_cfg

from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.mpm.engines.ULExplicitEngine import ULExplicitEngine
from src.mpm.engines.ULExplicitTwoPhaseEngine import ULExplicitTwoPhaseEngine
from src.mpm.engines.TLExplicitEngine import TLExplicitEngine
# from src.mpm.engines.ULImplicitEngine import ImplicitEngine
# from src.mpm.engines.IncompressibleEngine import IncompressibleEngine
from src.mpm.MaterialManager import ConstitutiveModel
from src.mpm.GenerateManager import GenerateManager
from src.mpm.MPMBase import Solver
from src.mpm.PostPlot import write_vtk_file
from src.mpm.Recorder import WriteFile
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction


class MPM(object):
    def __init__(self, title='A High Performance Multiscale and Multiphysics Simulator', log=True):  
        if log:
            print('# =================================================================== #')
            print('#', "".center(67), '#')
            print('#', "Welcome to GeoTaichi -- Material Point Method Engine !".center(67), '#')
            print('#', "".center(67), '#')
            print('#', title.center(67), '#')
            print('#', "".center(67), '#')
            print('# =================================================================== #', '\n')
        self.sims = Simulation()
        self.scene = myScene()
        self.generator = GenerateManager()
        self.material_handle = ConstitutiveModel()
        self.enginer = None
        self.neighbor = None
        self.recorder = None
        self.solver = None
        self.first_run = True

    def set_configuration(self, log=True, **kwargs):
        self.sims.set_dimension(DictIO.GetAlternative(kwargs, "dimension", "3-Dimension"))
        self.sims.set_is_2DAxisy(DictIO.GetAlternative(kwargs, "is_2DAxisy", False))
        if np.linalg.norm(np.array(self.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10:
            self.sims.set_domain(DictIO.GetEssential(kwargs, "domain"))
        self.sims.set_boundary(DictIO.GetAlternative(kwargs, "boundary" ,[None, None, None]))
        self.sims.set_gravity(DictIO.GetAlternative(kwargs, "gravity", [0.,0.,-9.8] if self.sims.dimension == 3 else [0., -9.8]))
        self.sims.set_background_damping(DictIO.GetAlternative(kwargs, "background_damping", 0.))
        self.sims.set_alpha(DictIO.GetAlternative(kwargs, "alphaPIC", 0.))
        self.sims.set_mapping_scheme(DictIO.GetAlternative(kwargs, "mapping", "MUSL"))
        self.sims.set_stabilize_technique(DictIO.GetAlternative(kwargs, "stabilize", None))
        self.sims.set_gauss_integration(DictIO.GetAlternative(kwargs, "gauss_number", 0))
        self.sims.set_boundary_direction(DictIO.GetAlternative(kwargs, "boundary_direction_detection", False))
        self.sims.set_free_surface_detection(DictIO.GetAlternative(kwargs, "free_surface_detection", False))
        self.sims.set_velocity_projection_scheme(DictIO.GetAlternative(kwargs, "velocity_projection", "PIC/FLIP"))
        self.sims.set_moving_least_square(DictIO.GetAlternative(kwargs, "moving_least_square", False))
        self.sims.set_shape_function(DictIO.GetAlternative(kwargs, "shape_function", "Linear"))
        self.sims.set_solver_type(DictIO.GetAlternative(kwargs, "solver_type", "Explicit"))
        self.sims.set_shape_smoothing(DictIO.GetAlternative(kwargs, "shape_smooth", 0.))
        self.sims.set_pressure_smoothing(DictIO.GetAlternative(kwargs, "pressure_smoothing", False))
        self.sims.set_strain_smoothing(DictIO.GetAlternative(kwargs, "strain_smoothing", False))
        self.sims.set_configuration(DictIO.GetAlternative(kwargs, "configuration", "ULMPM"))
        self.sims.set_material_type(DictIO.GetAlternative(kwargs, "material_type", "Solid"))
        self.sims.set_visualize(DictIO.GetAlternative(kwargs, "visualize", True))
        self.sims.set_sparse_grid(DictIO.GetAlternative(kwargs, "sparse_grid", None))
        self.sims.set_particle_shifting(DictIO.GetAlternative(kwargs, "particle_shifting", False))
        if log: 
            self.print_basic_simulation_info()
            print('\n')

    def set_implicit_solver_parameters(self, implicit_parameters={}):    
        if self.sims.solver_type != "Implicit":
            raise RuntimeError("KeyError:: /solver_type/ should be set as Implicit")
        
        if self.sims.material_type == "Solid":
            self.sims.set_calculate_reaction_force(DictIO.GetAlternative(implicit_parameters, "calculate_reaction_force", False))
            self.sims.set_integration_scheme(DictIO.GetAlternative(implicit_parameters, "integration_scheme", "Newmark"))
            self.sims.set_displacement_tolerance(DictIO.GetAlternative(implicit_parameters, "displacement_tolerance", 1e-4))
            self.sims.set_residual_tolerance(DictIO.GetAlternative(implicit_parameters, "residual_tolerance", 1e-10))
            self.sims.set_quasi_static(DictIO.GetAlternative(implicit_parameters, "quasi_static", False))
            self.sims.set_newmark_parameter(DictIO.GetAlternative(implicit_parameters, "newmark_parameter", [0.5, 0.25]))
            self.sims.set_max_iteration(DictIO.GetAlternative(implicit_parameters, "max_iteration_number", 50))
            self.sims.set_assemble_type(DictIO.GetAlternative(implicit_parameters, "assemble_type", "LocalStiffness"))
        self.sims.set_linear_solver(DictIO.GetAlternative(implicit_parameters, "linear_solver", "PCG"))
        if self.sims.linear_solver == "MGPCG":
            self.sims.set_multigrid_paramter(DictIO.GetAlternative(implicit_parameters, "multilevel", 4),
                                             DictIO.GetAlternative(implicit_parameters, "pre_and_post_smoothing", 2),
                                             DictIO.GetAlternative(implicit_parameters, "bottom_smoothing", 10))

    def set_solver(self, solver, log=True):
        self.sims.set_timestep(DictIO.GetEssential(solver, "Timestep"))
        self.sims.set_simulation_time(DictIO.GetEssential(solver, "SimulationTime"))
        self.sims.set_CFL(DictIO.GetAlternative(solver, "CFL", 0.5))
        self.sims.set_adaptive_timestep(DictIO.GetAlternative(solver, "AdaptiveTimestep", False))
        self.sims.set_save_interval(DictIO.GetAlternative(solver, "SaveInterval", self.sims.time / 20.))
        self.sims.set_save_path(DictIO.GetAlternative(solver, "SavePath", 'OutputData'))
        if log: 
            self.print_solver_info()
            print('\n')

    def memory_allocate(self, memory, log=True):    
        self.sims.set_material_num(DictIO.GetAlternative(memory, "max_material_number", 0))
        self.sims.set_particle_num(DictIO.GetAlternative(memory, "max_particle_number", 0))
        self.sims.set_constraint_num(DictIO.GetAlternative(memory, "max_constraint_number", {}))
        self.sims.set_verlet_distance_multiplier(DictIO.GetAlternative(memory, "verlet_distance_multiplier", 0.))
        if self.sims.solver_type == "Implicit":
            self.sims.set_dof_multiplier(DictIO.GetAlternative(memory, "dof_multiplier", 2))
        if log: 
            self.print_simulation_info()
            print('\n')

    def print_basic_simulation_info(self):
        print(" MPM Basic Configuration ".center(71,"-"))
        print(("Simulation Type: " + str(current_cfg().arch)).ljust(67))
        print(("Simulation Domain: " + str(self.sims.domain)).ljust(67))
        print(("Boundary Condition: " + str(self.sims.boundary)).ljust(67))
        print(("Gravity: " + str(self.sims.gravity)).ljust(67))

    def print_simulation_info(self):
        print(" MPM Engine Information ".center(71,"-"))
        print(("Background Damping: " + str(self.sims.background_damping)).ljust(67))
        print(("alpha Value: " + str(self.sims.alphaPIC)).ljust(67))
        print(("Stabilization Technique: " + str(self.sims.stabilize)).ljust(67))
        if self.sims.gauss_number > 0:
            print(("Guass Number: " + str(self.sims.gauss_number)).ljust(67))
        print(("Boundary Direction Detection: " + str(self.sims.boundary_direction_detection)).ljust(67))
        print(("Free Surface Detection: " + str(self.sims.free_surface_detection)).ljust(67))
        print(("Mapping Scheme: " + str(self.sims.mapping)).ljust(67))
        print(("Shape Function: " + str(self.sims.shape_function)).ljust(67))
        print(("Velocity Projection: " + str(self.sims.velocity_projection_scheme)).ljust(67))

    def print_solver_info(self):
        print(" MPM Solver Information ".center(71,"-"))
        print(("Initial Simulation Time: " + str(self.sims.current_time)).ljust(67))
        print(("Finial Simulation Time: " + str(self.sims.current_time + self.sims.time)).ljust(67))
        print(("Time Step: " + str(self.sims.dt[None])).ljust(67))
        print(("Save Interval: " + str(self.sims.save_interval)).ljust(67))
        print(("Save Path: " + str(self.sims.path)).ljust(67))

    def add_material(self, model, material):
        self.material_handle.save_material(model, material)

    def add_element(self, element):
        self.scene.activate_boundary(self.sims)
        self.scene.activate_element(self.sims, self.material_handle, element)
        self.scene.activate_particle(self.sims)

    def add_region(self, region):
        if type(region) is dict:
            self.generator.add_my_region(self.sims.dimension, self.sims.domain, region)
        elif type(region) is list:
            for region_dict in region:
                self.generator.add_my_region(self.sims.dimension, self.sims.domain, region_dict)

    def add_body(self, body):
        self.scene.check_materials(self.sims)
        self.generator.add_body(body, self.sims, self.scene)

    def add_body_from_file(self, body):
        self.scene.check_materials(self.sims)
        self.generator.read_body_file(body, self.sims, self.scene)

    def add_polygons(self, body):
        self.generator.add_polygons(body, self.sims, self.scene)

    def read_restart(self, file_number, file_path, is_continue=True):
        self.sims.set_is_continue(is_continue)
        if self.sims.is_continue:
            self.sims.current_print = file_number 
        self.add_body_from_file(body={"FileType":                         "NPZ",
                                      "Template":{
                                                        "Restart":        True,
                                                        "File":           file_path+f"/particles/MPMParticle{file_number:06d}.npz"
                                                 }
                                     }
                                )

    def add_boundary_condition(self, boundary=None):
        self.scene.boundary.get_essentials(self.scene.is_rigid, self.scene.psize, self.generator.myRegion)
        if type(boundary) is list or type(boundary) is dict:
            self.scene.boundary.iterate_boundary_constraint(self.sims, self.scene.element, boundary, 0)
        elif type(boundary) is str:
            if boundary is None:
                boundary = 'OutputData/boundary_conditions.txt'
            self.scene.boundary.read_boundary_constraint(self.sims, boundary)

    def clean_boundary_condition(self, boundary):
        if type(boundary) is list or type(boundary) is dict:
            self.scene.boundary.iterate_boundary_constraint(self.sims, self.scene.element, boundary, 1)

    def write_boundary_condition(self, output_path='OutputData'):
        self.scene.boundary.write_boundary_constraint(output_path)

    def select_save_data(self, particle=True, grid=False, object=True):
        if self.scene.contact_parameter is None or self.scene.contact_parameter.polygon_vertices is None:
            object = False
        self.sims.set_save_data(particle, grid, object)

    def choose_coupling_region(self, region_name=None, function=None):
        if not region_name is None:
            region: RegionFunction = self.generator.get_region_ptr(region_name)
            self.scene.choose_coupling_region(self.sims, region.function)
        elif not function is None:
            self.scene.choose_coupling_region(self.sims, function)
        self.scene.filter_particles()

    def modify_parameters(self, **kwargs):
        if len(kwargs) > 0:
            self.sims.set_simulation_time(DictIO.GetEssential(kwargs, "SimulationTime"))
            if "Timestep" in kwargs: 
                self.sims.set_timestep(DictIO.GetEssential(kwargs, "Timestep"))
            if "CFL" in kwargs: self.sims.set_CFL(DictIO.GetEssential(kwargs, "CFL"))
            if "AdaptiveTimestep" in kwargs: self.sims.set_adaptive_timestep(DictIO.GetEssential(kwargs, "AdaptiveTimestep"))
            if "SaveInterval" in kwargs: self.sims.set_save_interval(DictIO.GetEssential(kwargs, "SaveInterval"))
            if "SavePath" in kwargs: self.sims.set_save_path(DictIO.GetEssential(kwargs, "SavePath"))
            
            if "gravity" in kwargs: self.sims.set_gravity(DictIO.GetEssential(kwargs, "gravity"))
            if "background_damping" in kwargs: self.sims.set_background_damping(DictIO.GetEssential(kwargs, "background_damping"))
            if "alphaPIC" in kwargs: self.sims.set_alpha(DictIO.GetEssential(kwargs, "alphaPIC"))
    
    def add_spatial_grid(self):
        if self.sims.coupling == "Lagrangian" or self.sims.neighbor_detection:
            if self.neighbor is None:
                self.neighbor = SpatialHashGrid(self.sims)
            self.neighbor.neighbor_initialze(self.scene)

    def add_engine(self):
        if self.enginer is None:
            if self.sims.configuration == "ULMPM":
                if self.sims.solver_type == "Explicit":
                    if self.sims.material_type == "TwoPhaseSingleLayer":
                        self.enginer = ULExplicitTwoPhaseEngine(self.sims)
                    else:
                        self.enginer = ULExplicitEngine(self.sims)
                #elif self.sims.solver_type == "Implicit":
                #    if self.sims.material_type == "Solid":
                #        self.enginer = ImplicitEngine(self.sims)
                #    elif self.sims.material_type == "Fluid":
                #        self.enginer = IncompressibleEngine(self.sims)
                elif self.sims.solver_type == "SimiImplicit":
                    if self.sims.material_type == "TwoPhaseDoubleLayer":
                        self.enginer = None
                    else:
                        raise RuntimeError("Keyword:: /material_type/ should be set as $TwoPhase$")
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

    def add_solver(self, kwargs):
        if self.solver is None:
            self.solver = Solver(self.sims, self.generator, self.enginer, self.recorder)
        self.solver.set_callback_function(DictIO.GetAlternative(kwargs, "function", None))

    def add_postfunctions(self, **functions):
        self.solver.set_callback_function(functions)

    def set_window(self, window):
        self.sims.set_window_parameters(window)

    def add_essentials(self, kwargs):
        self.add_spatial_grid()
        self.add_engine()
        self.add_recorder()
        if self.sims.coupling is False:
            self.add_solver(kwargs)
        self.scene.calc_mass_cutoff(self.sims)
        if self.first_run:
            self.scene.boundary.set_boundary(self.sims)
        self.scene.boundary.set_boundary_types(self.sims, self.scene.element)

    def run(self, visualize=False, **kwargs):
        self.add_essentials(kwargs)
        self.check_critical_timestep()
        if visualize is False:
            self.solver.Solver(self.scene, self.neighbor)
        else:
            self.sims.set_visualize_interval(DictIO.GetEssential(kwargs, "visualize_interval"))
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

    def update_particle_properties(self, property_name, value, override=True, bodyID=None, region_name=None, function=None):
        if not bodyID is None:
            self.scene.update_particle_properties(self.sims, override, property_name, value, bodyID)
        elif not region_name is None:
            region: RegionFunction = self.generator.get_region_ptr(region_name)
            self.scene.update_particle_properties_in_region(self.sims, override, property_name, value, region.function)
        elif not function is None:
            self.scene.update_particle_properties_in_region(self.sims, override, property_name, value, function)

    def delete_particles(self, bodyID=None, region_name=None, function=None):
        if not bodyID is None:
            self.scene.delete_particles(bodyID)
        elif not region_name is None:
            region: RegionFunction = self.generator.get_region_ptr(region_name)
            self.scene.delete_particles_in_region(region.function)
        elif not function is None:
            self.scene.delete_particles_in_region(function)

    def postprocessing(self, start_file=0, end_file=-1, read_path=None, write_path=None, **kwargs):
        if read_path is None:
            read_path = self.sims.path
            if write_path is None:
                write_path = self.sims.path + "/vtks"
            elif not write_path is None:
                write_path = read_path + "/vtks"
        
        if not read_path is None and write_path is None:
            write_path = read_path + "/vtks"

        if not write_path.endswith('vtks'): write_path = write_path + '/vtks'

        write_vtk_file(self.sims, start_file, end_file, read_path, write_path, kwargs)
