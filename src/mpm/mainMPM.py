import numpy as np
from taichi.lang.impl import current_cfg

from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.mpm.engines.ULExplicitEngine import ULExplicitEngine
from src.mpm.GenerateManager import GenerateManager
from src.mpm.MPMBase import Solver
from src.mpm.PostPlot import write_vtk_file
from src.mpm.Recorder import WriteFile
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefination import vec3f


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
        self.enginer = None
        self.neighbor = None
        self.recorder = None
        self.solver = None

    def set_configuration(self, log=True, **kwargs):
        if np.linalg.norm(np.array(self.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10:
            self.sims.set_domain(DictIO.GetEssential(kwargs, "domain"))
        self.sims.set_dimension(DictIO.GetAlternative(kwargs, "dimension", "3-Dimension"))
        self.sims.set_boundary(DictIO.GetAlternative(kwargs, "boundary" ,["Destroy", "Destroy", "Destroy"]))
        self.sims.set_gravity(DictIO.GetAlternative(kwargs, "gravity", vec3f([0.,0.,-9.8])))
        self.sims.set_background_damping(DictIO.GetAlternative(kwargs, "background_damping", 0.))
        self.sims.set_alpha(DictIO.GetAlternative(kwargs, "alphaPIC", 0.))
        self.sims.set_stabilize_technique(DictIO.GetAlternative(kwargs, "stabilize", None))
        self.sims.set_gauss_integration(DictIO.GetAlternative(kwargs, "gauss_number", 0))
        self.sims.set_moving_least_square_order(DictIO.GetAlternative(kwargs, "mls_order", 0))
        self.sims.set_boundary_direction(DictIO.GetAlternative(kwargs, "boundary_direction_detection", False))
        self.sims.set_free_surface_detection(DictIO.GetAlternative(kwargs, "free_surface_detection", False))
        self.sims.set_mapping_scheme(DictIO.GetAlternative(kwargs, "mapping", "MUSL"))
        self.sims.set_shape_function(DictIO.GetAlternative(kwargs, "shape_function", "Linear"))
        self.sims.set_solver_type(DictIO.GetAlternative(kwargs, "solver_type", "Explicit"))
        self.sims.set_stress_smoothing(DictIO.GetAlternative(kwargs, "stress_smoothing", False))
        self.sims.set_strain_smoothing(DictIO.GetAlternative(kwargs, "strain_smoothing", False))
        self.sims.set_configuration(DictIO.GetAlternative(kwargs, "configuration", "ULMPM")) #fix only use ULMPM
        self.sims.set_material_type(DictIO.GetAlternative(kwargs, "material_type", "Solid"))
        if log: 
            self.print_basic_simulation_info()
            print('\n')

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
        self.scene.activate_basic_class(self.sims)
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
        if self.sims.mls_order > 0:
            print(("The order of moving least square: " + str(self.sims.order)).ljust(67))
        print(("Boundary Direction Detection: " + str(self.sims.boundary_direction_detection)).ljust(67))
        print(("Free Surface Detection: " + str(self.sims.free_surface_detection)).ljust(67))
        print(("Mapping Scheme: " + str(self.sims.mapping)).ljust(67))
        print(("Shape Function: " + str(self.sims.shape_function)).ljust(67))

    def print_solver_info(self):
        print(" MPM Solver Information ".center(71,"-"))
        print(("Initial Simulation Time: " + str(self.sims.current_time)).ljust(67))
        print(("Finial Simulation Time: " + str(self.sims.current_time + self.sims.time)).ljust(67))
        print(("Time Step: " + str(self.sims.dt[None])).ljust(67))
        print(("Save Interval: " + str(self.sims.save_interval)).ljust(67))
        print(("Save Path: " + str(self.sims.path)).ljust(67))

    def add_material(self, model, material):
        self.scene.activate_material(self.sims, model, material)

    def add_element(self, element):
        self.scene.activate_element(self.sims, element)

    def add_region(self, region):
        if type(region) is dict:
            self.generator.add_my_region(self.sims.domain, region)
        elif type(region) is list:
            for region_dict in region:
                self.generator.add_my_region(self.sims.domain, region_dict)

    def add_body(self, body):
        self.scene.check_materials(self.sims)
        self.generator.add_body(body, self.sims, self.scene)

    def add_body_from_file(self, body):
        """生成颗粒从文件
        Args:
            Body[dict]: 颗粒的参数
                FileType[str]: 文件类型 options:[txt, npz,obj] txt注释为#,[0,1,2]为x,y,z，[3]为体积，[4,5,6]为psize
                Template[dict/list]: 文件的模板
                    ParticleFile[str]: 颗粒文件的路径
                    BodyID[int]: 颗粒的ID
                    RigidBody[bool][option]:是否是刚体
                        Density[float][option]: 密度 仅在RigidBody为True时有效，default: 2650
                        MaterialID[uint]: 材料ID
                    ParticleStress[dict][option]: 颗粒的应力 
                        GravityField[bool]: 是否有重力场,default: False
                        InitialStress[vec6f]: 初始应力,default: [0., 0., 0., 0., 0., 0.]
                    Traction[dict][option]: 颗粒的牵引力,default:{}
                    Orientation[vec3f][option]: 颗粒的方向,default: [0., 0.,1]
                    InitialVelocity[vec3f][option]: 颗粒的初始速度,default: [0., 0., 0.]
                    FixVelocity[list][option]: 颗粒的速度约束,default: ["Free","Free","Free"]
        """
        self.scene.check_materials(self.sims)
        self.generator.read_body_file(body, self.sims, self.scene)

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
        """
        add boundary condition
        Args:
            sims[Simulation]: Simulation dataclass
            boundary[dict]: Boundary dict
                BoundaryType[str]: Boundary type option:[VelocityConstraint, ReflectionConstraint, FrictionConstraint, AbsorbingConstraint, TractionConstraint, DisplacementConstraint]
                NLevel[str/int][option]:  option:[All, 0, 1, 2, ...]
                StartPoint[vec3f]: Start point of boundary ,useage see below
                EndPoint[vec3f]: End point of boundary,useage see below
                when Boundary type = VelocityConstraint args include:
                    VelocityX[float/None][option]: Prescribed velocity along X axis
                    VelocityY[float/None][option]: Prescribed velocity along Y axis
                    VelocityZ[float/None][option]: Prescribed velocity along Z axis
                    Velocity[list][option]: Prescribed velocity
                when Boundary type = ReflectionConstraint args include:
                    Norm[vec3f]: Outer normal vector
                when Boundary type = FrictionConstraint args include:
                    Friction[float]: Friction angle
                    Norm[vec3f]: Outer normal vector
                when Boundary type = TractionConstraint args include:
                    ExternalForce[vec3f]: External force
                when Boundary type = DisplacementConstraint args include:
                    DisplacementX[float/None][option]: Prescribed displacement along X axis
                    DisplacementY[float/None][option]: Prescribed displacement along Y axis
                    DisplacementZ[float/None][option]: Prescribed displacement along Z axis
                    Displacement[list][option]: Prescribed displacement
            
            StartPoint = [x1,y1,z1],EndPoint = [x2,y2,z2] means Boundary particles constrained in the range x in [x1,x2],y in [y1,y2],z in [z1,z2].
            
        """
        if type(boundary) is list or type(boundary) is dict:
            self.scene.iterate_boundary_constraint(self.sims, boundary, 0)
        elif type(boundary) is str:
            if boundary is None:
                boundary = 'OutputData/boundary_conditions.txt'
            self.scene.read_boundary_constraint(self.sims, boundary)

    def clean_boundary_condition(self, boundary):
        if type(boundary) is list or type(boundary) is dict:
            self.scene.iterate_boundary_constraint(self.sims, boundary, 1)

    def write_boundary_condition(self, output_path='OutputData'):
        self.scene.write_boundary_constraint(output_path)

    def select_save_data(self, particle=True, grid=False):
        self.sims.set_save_data(particle, grid)

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
        if self.sims.coupling or self.sims.neighbor_detection:
            if self.neighbor is None:
                self.neighbor = SpatialHashGrid(self.sims)
            self.neighbor.neighbor_initialze(self.scene)

    def add_engine(self): #fix only use ULExplicitEngine
        if self.enginer is None:
            if self.sims.configuration == "ULMPM":
                if self.sims.solver_type == "Explicit":
                    self.enginer = ULExplicitEngine(self.sims)
                elif self.sims.solver_type == "SimiImplicit":
                    if self.sims.material_type == "TwoPhase":
                        self.enginer = None
                    else:
                        raise RuntimeError("Keyword:: /material_type/ should be set as $TwoPhase$")
        self.enginer.choose_engine(self.sims)
        self.enginer.choose_boundary_constraints(self.sims, self.scene)

    def add_recorder(self):
        if self.recorder is None:
            self.recorder = WriteFile(self.sims)

    def add_solver(self, kwargs):
        if self.solver is None:
            self.solver = Solver(self.sims, self.generator, self.enginer, self.recorder)
        self.solver.set_callback_function(kwargs)

    def set_window(self, window):
        self.sims.set_window_parameters(window)

    def add_essentials(self, kwargs):
        self.add_spatial_grid()
        self.add_engine()
        self.add_recorder()
        self.add_solver(kwargs)
        self.scene.calc_mass_cutoff(self.sims)
        self.scene.set_boundary(self.sims)

    def run(self, visualize=False, **kwargs):
        self.add_essentials(kwargs)
        self.check_critical_timestep()
        if visualize is False:
            self.solver.Solver(self.scene, self.neighbor) 
        else:
            self.sims.set_visualize_interval(DictIO.GetEssential(kwargs, "visualize_interval"))
            self.sims.set_window_size(DictIO.GetAlternative(kwargs, "WindowSize", self.sims.window_size))
            self.solver.Visualize(self.scene, self.neighbor)

    def check_critical_timestep(self):
        """Check the critical timestep for the simulation"""
        if self.sims.solver_type == "Explicit":
            print("#", " Check Timestep ... ...".ljust(67))
            critical_timestep = self.scene.get_critical_timestep()
            if self.sims.CFL * critical_timestep < self.sims.dt[None]:
                self.sims.update_critical_timestep(self.sims.CFL * critical_timestep)
            else:
                print("The prescribed time step is sufficiently small\n")

    def update_particle_properties(self, property_name, value, override=False, bodyID=None, region_name=None, function=None):
        if not bodyID is None:
            self.scene.update_particle_properties(override, property_name, value, bodyID)
        elif not region_name is None:
            region: RegionFunction = self.generator.get_region_ptr(region_name)
            self.scene.update_particle_properties_in_region(override, property_name, value, region.function)
        elif not function is None:
            self.scene.update_particle_properties_in_region(override, property_name, value, function)

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
            read_path = "OutputData"
            if write_path is None:
                write_path = "OutputData/vtks"
            elif not write_path is None:
                write_path = read_path + "/vtks"
        
        if not read_path is None and write_path is None:
            write_path = read_path + "/vtks"

        write_vtk_file(self.sims, start_file, end_file, read_path, write_path, kwargs)
