from itertools import product

import numpy as np
import taichi as ti

from src.mpm.elements.HexahedronKernel import *
from src.mpm.structs import HexahedronGuassCell, HexahedronCell, ParticleCPDI, IncompressibleCell
from src.mpm.elements.ElementBase import ElementBase
from src.mpm.Simulation import Simulation
from src.mesh.GaussPoint import GaussPointInRectangle
from src.mesh.HexMesh import HexahedronMesh
from src.utils.PrefixSum import PrefixSumExecutor
from src.utils.linalg import round32
from src.utils.ShapeFunctions import *
from src.utils.TypeDefination import u1, vec3f, vec3i
import src.utils.GlobalVariable as GlobalVariable


Threshold = 1e-12
class HexahedronElement8Nodes(ElementBase):
    def __init__(self, element_type, grid_level, ghost_cell) -> None:
        super().__init__(element_type, grid_level, ghost_cell)
        self.mesh = None
        self.grid_size = vec3f(2., 2., 2.)
        self.igrid_size = vec3f(0.5, 0.5, 0.5)
        self.start_local_coord = vec3f(-1, -1, -1)
        self.gnum = vec3i(0, 0, 0)
        self.cnum = vec3i(0, 0, 0)
        self.ifnum = vec3i(0, 0, 0)
        self.gridSum = 0
        self.cellSum = 0
        self.grid_nodes = 8
        self.influenced_node = 2
        self.grid_nodes_lower_order = 8
        self.influenced_node_lower_order = 2
        self.influenced_dofs = 24
        self.cell_volume = 0.
        self.cell_active = None
        self.gauss_point = None
        self.cell = None
        self.gauss_cell = None
        self.LnID = None
        self.shape_fn = None
        self.b_matrix = None
        self.node_size = None
        self.calculate = None
        self.calLength = None
        self.calLength_lower_order = None

        self.shape_function = None
        self.grad_shape_function = None
        self.shape_function_center = None
        self.lower_shape_function = None
        self.lower_grad_shape_function = None
        self.inertia_tensor = vec3f(0., 0., 0.)

        self.local_node = np.array([[0, 1], [4, 3]])

    def create_nodes(self, sims: Simulation, grid_size):
        if isinstance(grid_size, (list, tuple)):
            grid_size = vec3f(grid_size)
        domain = np.array(sims.domain)
        grid_size = np.array(grid_size)
        cnum = np.floor((domain + Threshold) / grid_size) 
        for d in range(3):
            if cnum[d] == 0:
                cnum[d] = 1      
        if sims.linear_solver == "MGPCG":   
            multiplier = 2 * sims.multilevel
            cnum = np.array(multiplier * np.ceil(cnum / multiplier), dtype=np.int32) 
        
        self.grid_size = vec3f(domain / cnum)
        self.igrid_size = 1. / self.grid_size
        self.cell_volume = self.calc_volume()
        self.cnum = vec3i(cnum) + 2 * self.ghost_cell
        self.gnum = self.cnum + 1
        self.gridSum = int(self.gnum[0] * self.gnum[1] * self.gnum[2])
        self.cellSum = int(self.cnum[0] * self.cnum[1] * self.cnum[2])
        self.mesh = HexahedronMesh(*self.cnum, *self.grid_size, self.ghost_cell)

    def set_characteristic_length(self, sims: Simulation):
        if sims.shape_function == "CPDI1" or sims.shape_function == "CPDI2":
            self.calLength = ParticleCPDI.field(shape=sims.max_particle_num)
        else:
            self.calLength = ti.Vector.field(3, float, shape=sims.max_body_num)
            if sims.stabilize == "F-Bar Method":
                self.calLength_lower_order = ti.Vector.field(3, float, shape=sims.max_body_num)

    def element_initialize(self, sims: Simulation, local_coordiates=False):
        self.choose_shape_function(sims)
        if sims.mls:
            self.compute_inertia_tensor(sims.shape_function)

        is_bbar = False
        if sims.stabilize == "B-Bar Method":
            is_bbar = True
            if sims.shape_function == "QuadBSpline" or sims.shape_function == "CubicBSpline" or sims.shape_function == "CPDI1" or sims.shape_function == "CPDI2":
                raise RuntimeError("B bar method can only used at [Linear, GIMP] Shape Function")
            
        if sims.max_particle_num > 0:
            if sims.gauss_number > 0:
                self.gauss_point = GaussPointInRectangle(gauss_point=sims.gauss_number)
                self.gauss_point.create_gauss_point()
            if sims.mode == "Normal":
                self.set_essential_field(is_bbar, sims.max_particle_num, sims.shape_function, sims.mls)
        
        if sims.solver_type == "Implicit":
            if sims.discretization == "FEM":
                self.pse = PrefixSumExecutor(self.gridSum)
            elif sims.discretization == "FDM":
                self.pse = PrefixSumExecutor(self.cellSum)
            self.flag = ti.field(int, shape=self.pse.get_length())

    def compute_inertia_tensor(self, shape_function_type):
        if shape_function_type == "QuadBSpline":
            self.inertia_tensor[0] = 4. / (self.grid_size[1] * self.grid_size[2])
            self.inertia_tensor[1] = 4. / (self.grid_size[0] * self.grid_size[2])
            self.inertia_tensor[2] = 4. / (self.grid_size[0] * self.grid_size[1])
        elif shape_function_type == "CubicBSpline": 
            self.inertia_tensor[0] = 3. / (self.grid_size[1] * self.grid_size[2])
            self.inertia_tensor[1] = 3. / (self.grid_size[0] * self.grid_size[2])
            self.inertia_tensor[2] = 3. / (self.grid_size[0] * self.grid_size[1])

    def choose_shape_function(self, sims: Simulation):
        if sims.shape_function == "Linear" or sims.shape_function == "SmoothLinear":
            self.influenced_node = 2
            self.grid_nodes = 8
            self.ifnum = vec3i(2, 2, 2)

            self.shape_function = ShapeLinear
            self.grad_shape_function = GShapeLinear
            self.shape_function_center = ShapeLinearCenter

        elif sims.shape_function == "GIMP":
            self.influenced_node = 3
            self.grid_nodes = 27
            self.ifnum = vec3i(3, 3, 3)
            
            self.shape_function = ShapeGIMP
            self.grad_shape_function = GShapeGIMP
            self.shape_function_center = ShapeGIMPCenter

        elif sims.shape_function == "CPDI1":
            self.influenced_node = 3
            self.grid_nodes = 27
            self.ifnum = vec3i(3, 3, 3)
            
            self.shape_function = ShapeLinear
            self.grad_shape_function = GShapeLinear
            self.shape_function_center = None

        elif sims.shape_function == "QuadBSpline":
            self.influenced_node = 3
            self.grid_nodes = 27
            self.ifnum = vec3i(3, 3, 3)
            
            self.shape_function = ShapeBsplineQ
            self.grad_shape_function = GShapeBsplineQ
            if sims.stabilize == "F-Bar Method":
                self.influenced_node_lower_order = 2
                self.grid_nodes_lower_order = 8
                self.lower_shape_function = ShapeLinear
                self.lower_grad_shape_function = GShapeLinear

        elif sims.shape_function == "CubicBSpline":
            self.influenced_node = 4
            self.grid_nodes = 64
            self.ifnum = vec3i(4, 4, 4)
            
            self.shape_function = ShapeBsplineC
            self.grad_shape_function = GShapeBsplineC
            if sims.stabilize == "F-Bar Method":
                self.influenced_node_lower_order = 3
                self.grid_nodes_lower_order = 27
                self.lower_shape_function = ShapeBsplineQ
                self.lower_grad_shape_function = GShapeBsplineQ

        else:
            raise KeyError(f"The shape function type {sims.shape_function} is not exist!")
        
        GlobalVariable.INFLUENCENODE = self.influenced_node
        self.influenced_dofs = 3 * self.grid_nodes

    def get_nonzero_grids_per_row(self):
        return (2 * self.influenced_node - 1) ** 3 

    def set_essential_field_local_coordinate(self, is_bbar, max_particle_num):
        self.start_local_coord = vec3f([-1, -1, -1])
        self.calculate = self.calc_local_shape_fn

    def set_essential_field(self, is_bbar, max_particle_num, shape_function, mls):
        if is_bbar:
            self.calculate = self.calc_shape_fn_b_bar
            self.LnID = ti.field(int)
            self.shape_fn = ti.field(float)
            self.dshape_fn = ti.Vector.field(3, float)
            self.dshape_fnc = ti.Vector.field(3, float)
            ti.root.dense(ti.i, max_particle_num * self.grid_nodes).place(self.LnID, self.shape_fn, self.dshape_fn, self.dshape_fnc)
            self.LnID.fill(0)
            self.shape_fn.fill(0)
            self.dshape_fn.fill(0)
            self.dshape_fnc.fill(0)
        else:
            if shape_function == "QuadBSpline" or shape_function == "CubicBSpline":
                if mls:
                    self.calculate = self.calc_shape_fn_spline_without
                else:
                    self.calculate = self.calc_shape_fn_spline
            elif shape_function == "CPDI1" or shape_function == "CPDI2":
                self.calculate = self.calc_shape_fn_cpdi
            elif shape_function == "SmoothLinear":
                self.calculate = self.calc_shape_fn_smooth
            else:
                self.calculate = self.calc_shape_fn

            self.LnID = ti.field(int)
            self.shape_fn = ti.field(float)
            if mls:
                ti.root.dense(ti.i, max_particle_num * self.grid_nodes).place(self.LnID, self.shape_fn)
                self.LnID.fill(0)
                self.shape_fn.fill(0)
            else:
                self.dshape_fn = ti.Vector.field(3, float)
                ti.root.dense(ti.i, max_particle_num * self.grid_nodes).place(self.LnID, self.shape_fn, self.dshape_fn)
                self.LnID.fill(0)
                self.shape_fn.fill(0)
                self.dshape_fn.fill(0)
        self.node_size = ti.field(ti.u8, shape=max_particle_num)

    def calculate_characteristic_length(self, sims: Simulation, particleNum, particle, psize):
        if sims.shape_function == "Linear":
            self.calLength.fill(vec3f(0., 0., 0.))
        if sims.shape_function == "SmoothLinear":
            set_particle_characteristic_length_gimp(particleNum, sims.shape_smooth, self.calLength, particle, psize)
        elif sims.shape_function == "QuadBSpline":
            self.calLength.fill(0.5 * self.grid_size)
            if sims.stabilize == "F-Bar Method":
                self.calLength_lower_order.fill(vec3f(0., 0., 0.))
        elif sims.shape_function == "CubicBSpline":
            self.calLength.fill(self.grid_size)
            if sims.stabilize == "F-Bar Method":
                self.calLength_lower_order.fill(0.5 * self.grid_size)
        elif sims.shape_function == "GIMP":
            set_particle_characteristic_length_gimp(particleNum, 1., self.calLength, particle, psize)
        elif sims.shape_function == "CPDI1" or sims.shape_function == "CPDI2":
            set_particle_characteristic_length_cpdi(particleNum, self.calLength, particle, psize)

    def activate_gauss_cell(self, sims: Simulation):
        if not self.gauss_cell is None:
            print("Warning: Previous cells will be override!")
        self.cell = HexahedronCell.field(shape=(self.get_total_cell_number(), self.grid_level))
        self.gauss_cell = HexahedronGuassCell.field(shape=(self.get_total_cell_number() * sims.gauss_number ** 3, self.grid_level))
        activate_cell(self.cell)

    def activate_euler_cell(self, sims: Simulation):
        if self.element_type == "Staggered":
            if not self.cell is None:
                print("Warning: Previous Euler cells will be override!")
            self.cell = IncompressibleCell(self.cnum, self.cellSum, self.ghost_cell)

    def set_up_cell_active_flag(self, fb: ti.FieldsBuilder):
        self.cell_active = ti.field(u1)
        fb.dense(ti.i, round32(self.get_total_cell_number())//32).quant_array(ti.i, dimensions=32, max_num_bits=32).place(self.cell_active)
        return fb.finalize()
    
    def reset_cell_status(self):
        kernel_reset_cell_status(self.cell_active)

    def update_particle_in_cell(self, particleNum, particle):
        kernel_find_located_cell(self.igrid_size, self.gnum, particleNum[0], particle)
        
    def print_message(self):
        print("Grid Size = ", self.grid_size)
        print("The number of nodes = ", self.gnum)

    def get_boundary_nodes(self, start_point, end_point):
        start_bound = np.ceil([point - Threshold for point in start_point] * self.igrid_size)
        end_bound = np.floor([point + Threshold for point in end_point] * self.igrid_size) + 1
        end_bound = np.maximum(end_bound, start_bound + 1)

        xnode = np.arange(start_bound[0], end_bound[0], 1)
        ynode = np.arange(start_bound[1], end_bound[1], 1)
        znode = np.arange(start_bound[2], end_bound[2], 1)
        
        total_nodes = np.array(list(product(xnode, ynode, znode)))
        return np.array([n[0] + n[1] * self.gnum[0] + n[2] * self.gnum[0] * self.gnum[1] for n in total_nodes], dtype=np.int32)
    
    def get_cell_number(self):
        return self.gnum - 1
    
    def get_element_number(self, sims: Simulation):
        if sims.shape_function == "Linear":
            return self.cnum[0] * self.cnum[1] * self.cnum[2]
        elif sims.shape_function == "QuadBSpline":
            return (self.cnum[0] - 1) * (self.cnum[1] - 1) * (self.cnum[2] - 1)
        elif sims.shape_function == "CubicBSpline":
            return (self.cnum[0] - 2) * (self.cnum[1] - 2) * (self.cnum[2] - 2)
    
    def get_total_cell_number(self):
        cnum = self.get_cell_number()
        return int(cnum[0] * cnum[1] * cnum[2])
    
    def calc_volume(self):
        return self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

    def calc_total_particle(self, npic):
        return npic * npic * npic
    
    def calc_particle_size(self, npic):
        return 0.5 * self.grid_size / npic
    
    def calc_critical_timestep(self, velocity):
        return ti.min(self.grid_size[0], self.grid_size[1], self.grid_size[2]) / velocity if velocity > 0. else 10000

    def calc_local_shape_fn(self, particleNum, particle):
        update(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, self.start_local_coord, particleNum[0],
               particle, self.calLength, self.LnID, self.node_size, self.shape_fn, self.dshape_fn, self.shape_function, self.grad_shape_function)
        
    def calc_local_shape_fn_b_bar(self, particleNum, particle):
        updatebbar(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, self.start_local_coord, particleNum[0],
                   particle, self.calLength, self.LnID, self.node_size, self.shape_fn, self.dshape_fn, self.dshape_fnc, self.shape_function, self.grad_shape_function, self.shape_function_center)
        
    def calc_shape_fn(self, particleNum, particle):
        global_update(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, 
                      self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.shape_function, self.grad_shape_function)
        
    def calc_shape_fn_smooth(self, particleNum, particle):
        global_update_smooth(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, 
                      self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.shape_function, self.grad_shape_function)
        
    def calc_shape_fn_cpdi(self, particleNum, particle):
        global_update_spline(self.grid_nodes, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, 
                             self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.shape_function, self.grad_shape_function, self.boundary_type)
        
    def calc_shape_fn_spline_lower_order(self, particleNum, particle):
        global_update_spline(self.grid_nodes_lower_order, self.influenced_node_lower_order, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength_lower_order, 
                             self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.lower_shape_function, self.lower_grad_shape_function, self.boundary_type)

    def calc_shape_fn_spline(self, particleNum, particle):
        global_update_spline(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, 
                             self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.shape_function, self.grad_shape_function, self.boundary_type)
        
    def calc_shape_fn_spline_without(self, particleNum, particle):
        global_update_spline_fn(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, 
                                self.node_size, self.LnID, self.shape_fn, self.shape_function, self.boundary_type)

    def calc_shape_fn_b_bar(self, particleNum, particle):
        global_updatebbar(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, 
                          self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.dshape_fnc, self.shape_function, self.grad_shape_function, self.shape_function_center)

    def find_located_cell(self, particleNum, particle):
        kernel_find_located_cell(self.igrid_size, self.gnum, particleNum[0], particle)

    def set_boundary_type(self, sims: Simulation, grid_level):
        if sims.shape_function == "QuadBSpline" or sims.shape_function == "CubicBSpline":
            kernel_set_boundary_type(self.gridSum, grid_level, self.gnum, self.boundary_flag, self.boundary_type)