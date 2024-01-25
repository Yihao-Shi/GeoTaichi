from itertools import product

import numpy as np
import taichi as ti

from src.mpm.elements.QuadrilateralKernel import *
from src.mpm.BaseStruct import HexahedronGuassCell, HexahedronCell
from src.mpm.elements.ElementBase import ElementBase
from src.mpm.Simulation import Simulation
from src.utils.constants import Threshold
from src.utils.GaussPoint import GaussPointInRectangle
from src.utils.PrefixSum import PrefixSumExecutor
from src.utils.ScalarFunction import round32
from src.utils.ShapeFunctions import *
from src.utils.TypeDefination import u1, vec2f, vec2i


class QuadrilateralElement4Nodes(ElementBase):
    def __init__(self) -> None:
        super().__init__()
        self.contact_position_offset = vec2f(0., 0.)
        self.grid_size = vec2f(2., 2.)
        self.igrid_size = vec2f(0.5, 0.5)
        self.start_local_coord = vec2f(-1, -1)
        self.gnum = vec2i(0, 0)
        self.cnum = vec2i(0, 0)
        self.ifnum = vec2i(0, 0)
        self.gridSum = 0
        self.grid_nodes = 4
        self.influenced_node = 2
        self.influenced_dofs = 8
        self.cell_volume = 0.
        self.cell_active = None
        self.gauss_point = None
        self.cell = None
        self.gauss_cell = None
        self.node_connectivity = None
        self.LnID = None
        self.shape_fn = None
        self.b_matrix = None
        self.node_size = None
        self.nodal_coords = None
        self.calculate = None
        self.calLength = None

        self.shape_function = None
        self.grad_shape_function = None
        self.shape_function_center = None

    def create_nodes(self, sims: Simulation, grid_size):
        self.grid_size = grid_size
        self.igrid_size = 1. / self.grid_size
        self.cell_volume = self.calc_volume()
        self.gnum = vec2i([int(sims.domain[i] * self.igrid_size[i]) + 1 for i in range(2)])                                 
        for d in ti.static(range(2)):
            if self.gnum[d] == 0:
                self.gnum[d] = 1
        self.cnum = self.gnum - 1
        self.gridSum = int(self.gnum[0] * self.gnum[1])
        self.nodal_coords = ti.Vector.field(2, float, shape=self.gridSum)
        set_node_position_(self.nodal_coords, self.gnum, self.grid_size)

    def set_characteristic_length(self, sims: Simulation):
        self.calLength = ti.Vector.field(2, float, shape=sims.max_body_num)

    def element_initialize(self, sims: Simulation, local_coordiates=False):
        self.choose_shape_function(sims.shape_function)
        is_bbar = False
        if sims.stabilize == "B-Bar Method":
            is_bbar = True

            if sims.shape_function == "QuadBSpline" or sims.shape_function == "CubicBSpline":
                raise RuntimeError("B bar method can only used at [Linear, GIMP] Shape Function")
            
        if sims.max_particle_num > 0:
            if sims.gauss_number > 0:
                self.gauss_point = GaussPointInRectangle(gauss_point=sims.gauss_number, dimemsion=2)
                self.gauss_point.create_gauss_point()
            self.set_essential_field(is_bbar, sims.max_particle_num)
        
        self.node_connectivity = ti.Vector.field(self.grid_nodes, int, shape=self.get_total_cell_number())
        find_nodes_per_element_(0, self.get_total_cell_number(), self.gnum, self.node_connectivity, set_connectivity)
                
        if sims.solver_type == "Implicit":
            self.pse = PrefixSumExecutor(self.gridSum)
            self.flag = ti.field(int, shape=self.gridSum)

    def choose_shape_function(self, shape_function_type):
        if shape_function_type == "Linear":
            self.influenced_node = 2
            self.grid_nodes = 4
            self.ifnum = vec2i(2, 2)

            self.shape_function = ShapeLinear
            self.grad_shape_function = GShapeLinear
            self.shape_function_center = ShapeLinearCenter

        elif shape_function_type == "GIMP":
            self.influenced_node = 3
            self.grid_nodes = 9
            self.ifnum = vec2i(3, 3)
            
            self.shape_function = ShapeGIMP
            self.grad_shape_function = GShapeGIMP
            self.shape_function_center = ShapeGIMPCenter

        elif shape_function_type == "QuadBSpline":
            self.influenced_node = 3
            self.grid_nodes = 9
            self.ifnum = vec2i(3, 3)
            
            self.shape_function = ShapeBsplineQ
            self.grad_shape_function = GShapeBsplineQ

        elif shape_function_type == "CubicBSpline":
            self.influenced_node = 4
            self.grid_nodes = 16
            self.ifnum = vec2i(4, 4)
            
            self.shape_function = ShapeBsplineC
            self.grad_shape_function = GShapeBsplineC

        else:
            raise KeyError(f"The shape function type {shape_function_type} is not exist!")
        
        self.influenced_dofs = 2 * self.grid_nodes

    def set_essential_field_local_coordinate(self, is_bbar, max_particle_num):
        self.start_local_coord = vec2f([-1, -1])
        self.calculate = self.calc_local_shape_fn

    def set_essential_field(self, is_bbar, max_particle_num):
        if is_bbar:
            self.calculate = self.calc_shape_fn_b_bar

            self.LnID = ti.field(int)
            self.shape_fn = ti.field(float)
            self.dshape_fn = ti.Vector.field(3, float)
            self.dshape_fnc = ti.Vector.field(3, float)
            ti.root.dense(ti.i, max_particle_num * self.grid_nodes).place(self.LnID, self.shape_fn, self.dshape_fn, self.dshape_fnc)
        else:
            self.calculate = self.calc_shape_fn

            self.LnID = ti.field(int)
            self.shape_fn = ti.field(float)
            self.dshape_fn = ti.Vector.field(3, float)
            ti.root.dense(ti.i, max_particle_num * self.grid_nodes).place(self.LnID, self.shape_fn, self.dshape_fn)
        self.node_size = ti.field(ti.u8, shape=max_particle_num)

    def set_contact_position_offset(self, cutoff):
        self.contact_position_offset = cutoff * self.grid_size

    def activate_gauss_cell(self, sims: Simulation, grid_level):
        if not self.gauss_cell is None:
            print("Warning: Previous cells will be override!")
        self.cell = HexahedronCell.field(shape=(self.get_total_cell_number(), grid_level))
        self.gauss_cell = HexahedronGuassCell.field(shape=(self.get_total_cell_number() * sims.gauss_number ** 2, grid_level))
        activate_cell(self.cell)

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
        start_bound = np.ceil([point - Threshold for point in start_point] / self.grid_size)
        end_bound = np.floor([point + Threshold for point in end_point] / self.grid_size) + 1

        xnode = np.arange(start_bound[0], end_bound[0], 1)
        ynode = np.arange(start_bound[1], end_bound[1], 1)
        
        total_nodes = np.array(list(product(xnode, ynode)))
        return np.array([n[0] + n[1] * self.gnum[0] for n in total_nodes], dtype=np.int32)
    
    def get_cell_number(self):
        return self.gnum - 1
    
    def get_total_cell_number(self):
        cnum = self.get_cell_number()
        return int(cnum[0] * cnum[1])
    
    def calc_volume(self):
        return self.grid_size[0] * self.grid_size[1]

    def calc_total_particle(self, npic):
        return npic * npic
    
    def calc_particle_size(self, npic):
        return 0.5 * self.grid_size / npic
    
    def calc_critical_timestep(self, velocity):
        return ti.min(self.grid_size[0], self.grid_size[1]) / velocity
    
    def initial_estimate_active_dofs(self, cutoff, node):
        return estimate_active_dofs(cutoff, node)
    
    def find_active_nodes(self, cutoff, node, active_node):
        find_active_node(self.gridSum, cutoff, node, self.flag)
        self.pse.run(self.flag, self.gridSum)
        return set_active_dofs(self.gridSum, cutoff, node, self.flag, active_node)

    def calc_local_shape_fn(self, particleNum, particle):
        update(self.grid_nodes, self.influenced_node, self.igrid_size, self.gnum, self.start_local_coord, particleNum[0],
                                particle, self.calLength, self.nodal_coords, self.LnID, self.node_size, self.shape_fn, self.dshape_fn, self.shape_function, self.grad_shape_function)
        
    def calc_local_shape_fn_b_bar(self, particleNum, particle):
        updatebbar(self.grid_nodes, self.influenced_node, self.igrid_size, self.gnum, self.start_local_coord, particleNum[0],
                                    particle, self.calLength, self.nodal_coords, self.LnID, self.node_size, self.shape_fn, self.dshape_fn, self.dshape_fnc, self.shape_function, self.grad_shape_function, self.shape_function_center)
        
    def calc_shape_fn(self, particleNum, particle):
        global_update(self.grid_nodes, self.influenced_node, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, self.nodal_coords, 
                                       self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.shape_function, self.grad_shape_function)

    def calc_shape_fn_b_bar(self, particleNum, particle):
        global_updatebbar(self.grid_nodes, self.influenced_node, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, self.nodal_coords, 
                                           self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.dshape_fnc, self.shape_function, self.grad_shape_function, self.shape_function_center)

    def find_located_cell(self, particleNum, particle):
        kernel_find_located_cell(self.igrid_size, self.gnum, particleNum[0], particle)