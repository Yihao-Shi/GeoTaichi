import numpy as np
from taichi.lang.impl import current_cfg

from src.dem.SceneManager import myScene
from src.dem.GenerateManager import GenerateManager
from src.dem.ContactManager import ContactManager
from src.dem.engines.ExplicitEngine import ExplicitEngine
from src.dem.DEMBase import Solver
from src.dem.PostPlot import write_vtk_file
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
        self.history_contact_path=dict()

    def set_configuration(self, log=True, **kwargs):
        """
        配置DEM基本参数
        参数:
        domain[list] :计算域大小
        boundary[list][option] :边界条件类型，可选项有["Destroy", "Reflect", "Period"] #注意“period”功能尚未开发
        gravity[list][option] :重力加速度
        engine[str][option] :求解器类型，可选项有["SymplecticEuler", "VelocityVerlet", "PredictCorrector"]
        search[str][option] :邻居搜索算法，可选项有["LinkedCell", "Brust"]
        
        Configure the basic parameters of DEM
        Parameters:
        domain[list] : Size of the computational domain
        boundary[list][option] : Type of boundary conditions, options include ["Destroy", "Reflect", "Period"] #Note that the "period" function has not yet been developed
        gravity[list][option] : Gravity acceleration
        engine[str][option] : Solver type, options include ["SymplecticEuler", "VelocityVerlet", "PredictCorrector"]
        search[str][option] : Neighbor search algorithm, options include ["LinkedCell", "Brust"]
        """
        if np.linalg.norm(np.array(self.sims.get_simulation_domain()) - np.zeros(3)) < 1e-10:
            self.sims.set_domain(DictIO.GetEssential(kwargs, "domain"))
        self.sims.set_boundary(DictIO.GetAlternative(kwargs, "boundary" ,["Destroy", "Destroy", "Destroy"]))
        self.sims.set_gravity(DictIO.GetAlternative(kwargs, "gravity", vec3f([0.,0.,-9.8])))
        self.sims.set_engine(DictIO.GetAlternative(kwargs, "engine", "SymplecticEuler"))
        self.sims.set_search(DictIO.GetAlternative(kwargs, "search", "LinkedCell"))
        if log: 
            self.print_basic_simulation_info()
            print('\n')
    
    def set_solver(self, solver, log=True):
        """
        求解器参数
        solver[dict]:
            Timestep[float] :时间步长
            SimulationTime[float] :模拟总时间
            CFL[float][option] :CFL数值
            AdaptiveTimestep[bool][option] :是否自适应时间步长
            SaveInterval[float] :保存间隔
            SavePath[str][option] :保存路径
        Set solver parameters.
            Args:
                solver (dict): Dictionary containing solver parameters.
                    - Timestep (float): Time step size.
                    - SimulationTime (float): Total simulation time.
                    - CFL (float, optional): CFL value. Default is 0.5.
                    - AdaptiveTimestep (bool, optional): Whether to use adaptive time step. Default is False.
                    - SaveInterval (float): Save interval.
                    - SavePath (str, optional): Save path. Default is 'OutputData'.
                log (bool, optional): Whether to print solver information. Default is True.
        """
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
        """
        内存分配参数
        memory [dict]:
            max_material_number[int]                : 最大材料数
            max_particle_number[float]              : 最大粒子数
            max_sphere_number[int][option]          : 最大球数
            max_clump_number[int][option]           : 最大多球数
            max_levelset_grid_number[int][option]   : 最大水平集网格数
            max_rigid_body_number[int][option]: 最大刚体数
            max_surface_node_number[int][option]: 最大表面节点数
            max_patch_number[int][option]: 最大三角面片个数
            max_facet_number[int][option]: 最大三角墙个数
            max_servo_wall_number[int][option]: 最大伺服墙数
            max_plane_number[int][option]: 最大平面数
            compaction_ratio[float][option]: 接触列表的压缩率
            body_coordination_number[int][option]: 粒子最大配位数
            wall_coordination_number[int][option]: 颗粒最大墙邻居数
            verlet_distance_multiplier[float][option]: Verlet距离倍数
            wall_per_cell[int][option]: 每个哈希网格单元的最大墙数
                Allocate memory parameters.

        Args:
            memory (dict): A dictionary containing the memory allocation parameters.
                max_material_number (int): The maximum number of materials.
                max_particle_number (float): The maximum number of particles.
                max_sphere_number (int, optional): The maximum number of spheres.
                max_clump_number (int, optional): The maximum number of clumps.
                max_levelset_grid_number (int, optional): The maximum number of level set grids.
                max_rigid_body_number (int, optional): The maximum number of rigid bodies.
                max_surface_node_number (int, optional): The maximum number of surface nodes.
                max_patch_number (int, optional): The maximum number of patch triangles.
                max_facet_number (int, optional): The maximum number of facet triangles.
                max_servo_wall_number (int, optional): The maximum number of servo walls.
                max_plane_number (int, optional): The maximum number of planes.
                compaction_ratio (float, optional): The compression ratio of the contact list.
                body_coordination_number (int, optional): The maximum coordination number of particles.
                wall_coordination_number (int, optional): The maximum coordination number of walls.
                verlet_distance_multiplier (float, optional): The Verlet distance multiplier.
                wall_per_cell (int, optional): The maximum number of walls per hash grid cell.

            log (bool, optional): Whether to print simulation information. Defaults to True.
        """
        self.sims.set_material_num(DictIO.GetEssential(memory, "max_material_number"))
        self.sims.set_particle_num(DictIO.GetEssential(memory, "max_particle_number"))
        self.sims.set_sphere_num(DictIO.GetAlternative(memory, "max_sphere_number", 0))
        self.sims.set_clump_num(DictIO.GetAlternative(memory, "max_clump_number", 0))
        self.sims.set_level_grid_num(DictIO.GetAlternative(memory, "max_levelset_grid_number", 0))
        self.sims.set_rigid_body_num(DictIO.GetAlternative(memory, "max_rigid_body_number", 0))
        self.sims.set_surface_node_num(DictIO.GetAlternative(memory, "max_surface_node_number", 0))
        self.sims.set_patch_num(DictIO.GetAlternative(memory, "max_patch_number", 0))
        self.sims.set_facet_num(DictIO.GetAlternative(memory, "max_facet_number", 0))
        self.sims.set_servo_wall_num(DictIO.GetAlternative(memory, "max_servo_wall_number", 0))
        self.sims.set_plane_num(DictIO.GetAlternative(memory, "max_plane_number", 0))
        self.sims.set_compaction_ratio(DictIO.GetAlternative(memory, "compaction_ratio", 0.5))
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

    def print_solver_info(self):
        print(" DEM Solver Information ".center(71,"-"))
        print(("Initial Simulation Time: " + str(self.sims.current_time)).ljust(67))
        print(("Finial Simulation Time: " + str(self.sims.current_time + self.sims.time)).ljust(67))
        print(("Time Step: " + str(self.sims.dt[None])).ljust(67))
        print(("Save Interval: " + str(self.sims.save_interval)).ljust(67))
        print(("Save Path: " + str(self.sims.path)).ljust(67))

    def add_region(self, region):
        if type(region) is dict:
            self.generator.add_my_region(self.sims.domain, region)
        elif type(region) is list:
            for region_dict in region:
                self.generator.add_my_region(self.sims.domain, region_dict)

    def add_attribute(self, materialID, attribute):
        self.scene.add_attribute(self.sims, materialID, attribute)

    def add_template(self, template, types="Clump"):
        self.generator.add_my_template(template, types)

    def create_body(self, body):
        self.generator.create_body(body, self.sims, self.scene)

    def add_body(self, body):
        self.generator.add_body(body, self.sims, self.scene)

    def add_body_from_file(self, body):
        """
        颗粒文件读取模板
        Args:
            Body[dict/list]: 颗粒文件读取的参数
                FileType[str]: 文件读取的类型, 可选项: TXT, NPZ,OBJ
                BodyType[str]: 颗粒的类型, 可选项: Clump, Sphere,仅当FileType为TXT时有效
                Period[list][option]:设定颗粒生成时间间隔,输入格式为[起始时间,结束时间],默认值为[0,0]
                Template[dict/list]: 颗粒的模板
                    ParticleFile[str][option]: 颗粒文件的路径,仅颗粒类型为Sphere时有效
                    ClumpFile[str][option]: 颗粒文件的路径,仅颗粒类型为Clump时有效
                    PebbleFile[str][option]: 颗粒文件的路径,仅颗粒类型为Pebble时有效
                    GeometryFile[str][option]: 设定需要体素的文件读取路径,当文件类型为“OBJ”时生效。
                    GroupID[int][option]: 颗粒的组ID
                    MaterialID[int][option]: 颗粒的材料ID
                    InitialVelocity[list][option]: 颗粒的初始速度
                    InitialAngularVelocity[list][option]: 颗粒的初始角速度
                    FixVelocity[list][option]: 颗粒的平动自由约束
                    FixAngularVelocity[list][option]: 颗粒的旋动自由约束
                    ScaleFactor[float][option]: 设定颗粒的缩放因子，默认值为1,仅当文件类型为“OBJ”时生效
                    Translation[list][option]: 设定颗粒的平移量，默认值为[0,0,0],仅当文件类型为“OBJ”时生效
                    Orientation[list][option]: 设定颗粒的旋转角度，默认值为[0,0,0],仅当文件类型为“OBJ”时生效
        """
        self.generator.read_body_file(body, self.sims, self.scene)

    def add_wall(self, body):
        self.generator.add_wall(body, self.sims, self.scene)

    def add_wall_from_file(self, body):
        self.generator.read_wall_file(body, self.sims, self.scene)

    def choose_contact_model(self, particle_particle_contact_model=None, particle_wall_contact_model=None):
        if self.contactor is None:
            self.contactor = ContactManager()
            self.contactor.choose_neighbor(self.sims, self.scene)
        
        if self.sims.max_material_num == 0:
            raise RuntimeError("memory_allocate should be launched first!")
        self.sims.set_particle_particle_contact_model(particle_particle_contact_model)
        self.sims.set_particle_wall_contact_model(particle_wall_contact_model)
        self.contactor.particle_particle_initialize(self.sims)
        self.contactor.particle_wall_initialize(self.sims)

    def add_property(self, materialID1, materialID2, property, dType="all"):
        self.contactor.add_contact_property(self.sims, materialID1, materialID2, property, dType)

    def inherit_property(self, materialID, property):
        pass

    def load_history_contact(self):
        file_number = DictIO.GetAlternative(self.history_contact_path, "file_number", 0)
        ppcontact = DictIO.GetAlternative(self.history_contact_path, "ppcontact", None)
        pwcontact = DictIO.GetAlternative(self.history_contact_path, "pwcontact", None)

        if not ppcontact is None:
            self.contactor.physpp.restart(self.contactor.neighbor, file_number, ppcontact, True)
        if not pwcontact is None:
            self.contactor.physpw.restart(self.contactor.neighbor, file_number, pwcontact, False)

    def select_save_data(self, particle=True, sphere=False, clump=False, wall=False, particle_particle_contact=False, particle_wall_contact=False):
        self.sims.set_save_data(particle, sphere, clump, wall, particle_particle_contact, particle_wall_contact)

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

        self.add_body_from_file(body={"FileType": "NPZ", "Template": {"Restart": True, "ParticleFile": particle_path, "SphereFile": sphere_path, "ClumpFile": clump_path}})
        self.add_wall_from_file(body={"FileType": "NPZ", "WallFile": wall_path, "ServoFile": servo_path})
        self.history_contact_path.update(file_number=file_number, ppcontact=ppcontact_path, pwcontact=pwcontact_path)

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
            self.enginer = ExplicitEngine(self.contactor)
        self.enginer.choose_engine(self.sims, self.scene)
        self.enginer.set_servo_mechanism(self.sims, callback)

    def add_recorder(self):
        if self.recorder is None:
            self.recorder = WriteFile(self.sims, self.contactor.physpp, self.contactor.physpw, self.contactor.neighbor)

    def add_solver(self, kwargs):
        if self.solver is None:
            self.solver = Solver(self.sims, self.generator, self.contactor, self.enginer, self.recorder)
        self.solver.set_callback_function(kwargs)

    def add_essentials(self, kwargs):
        if self.scene.particleNum[0] > 0:
            if self.contactor.have_initialise is False:
                self.contactor.initialize(self.sims, self.scene)
        else:
            raise RuntimeError("Particle should be added first")
        self.load_history_contact()
        self.add_engine(DictIO.GetAlternative(kwargs, "callback", None))
        self.add_recorder()
        self.add_solver(kwargs)

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

    def check_critical_timestep(self):
        print("#", " Check Timestep ... ...".ljust(67))
        critical_timestep = self.get_critical_timestep()
        if self.sims.CFL * critical_timestep < self.sims.dt[None]:
            self.sims.update_critical_timestep(self.sims.CFL * critical_timestep)
        else:
            print("The prescribed time step is sufficiently small\n")

    def get_critical_timestep(self):
        return self.contactor.physpp.calcu_critical_timesteps(self.scene, self.sims.max_material_num)
    
    def update_particle_properties(self, particle_type, property_name, value, override=False, bodyID=None, region_name=None, function=None):
        if not bodyID is None:
            self.scene.update_particle_properties(override, particle_type, property_name, value, bodyID)
        elif not region_name is None:
            region: RegionFunction = self.generator.get_region_ptr(region_name)
            self.scene.update_particle_properties_in_region(override, particle_type, property_name, value, region.function)
        elif not function is None:
            self.scene.update_particle_properties_in_region(override, particle_type, property_name, value, function)
        self.contactor.neighbor.pre_neighbor(self.scene)

    def update_wall_status(self, wallID, property_name, value, override=False):
        self.scene.update_wall_properties(self.sims, override, property_name, value, wallID)
        self.contactor.neighbor.pre_neighbor(self.scene)

    def update_contact_properties(self, materialID1, materialID2, property_name, value, overide=False):
        self.contactor.update_contact_property(self.sims, materialID1, materialID2, property_name, value, overide)

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
