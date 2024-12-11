from itertools import product

import numpy as np
import taichi as ti

from src.mpm.elements.QuadrilateralKernel import *
from src.mpm.structs import HexahedronGuassCell, HexahedronCell, ParticleCPDI, IncompressibleCell
from src.mpm.elements.ElementBase import ElementBase
from src.mpm.Simulation import Simulation
from src.utils.GaussPoint import GaussPointInRectangle
from src.utils.PrefixSum import PrefixSumExecutor
from src.utils.linalg import round32, flip2d_linear
from src.utils.ShapeFunctions import *
from src.utils.TypeDefination import u1, vec2f, vec2i


Threshold = 1e-12
class QuadrilateralElement4Nodes(ElementBase):
    def __init__(self, element_type) -> None:
        super().__init__(element_type)
        self.grid_size = vec2f(2., 2.)
        self.igrid_size = vec2f(0.5, 0.5)
        self.start_local_coord = vec2f(-1, -1)
        self.gnum = vec2i(0, 0)
        self.cnum = vec2i(0, 0)
        self.ifnum = vec2i(0, 0)
        self.gridSum = 0
        self.cellSum = 0
        self.grid_nodes = 4
        self.influenced_node = 2
        self.grid_nodes_lower_order = 4
        self.influenced_node_lower_order = 2
        self.influenced_dofs = 8
        self.cell_volume = 0.
        self.cell_active = None
        self.gauss_point = None
        self.cell = None
        self.gauss_cell = None
        self.node_connectivity = None
        self.LnID = None
        self.shape_fn = None
        self.shape_fnc = None
        self.dshape_fn = None
        self.dshape_fnc = None
        self.b_matrix = None
        self.node_size = None
        self.nodal_coords = None
        self.calculate = None
        self.calLength = None
        self.calLength_lower_order = None
        self.inertia_tensor = vec2f(0., 0.)

        self.shape_function = None
        self.shape_function_r = None
        self.shape_function_z = None
        self.grad_shape_function = None
        self.grad_shape_function_r = None
        self.grad_shape_function_z = None
        self.lower_shape_function = None
        self.lower_grad_shape_function = None
        self.shape_function_center = None

    def create_nodes(self, sims: Simulation, grid_size):
        if isinstance(grid_size, (list, tuple)):
            grid_size = ti.Vector(grid_size)
        domain = np.array(sims.domain)
        grid_size = np.array(grid_size)
        cnum = np.floor((domain + Threshold) / grid_size)          
        for d in range(2):
            if cnum[d] == 0:
                cnum[d] = 1
        if self.element_type == "Staggered":   
            multiplier = 2 * sims.multilevel
            cnum = np.array(multiplier * np.ceil(cnum / multiplier), dtype=np.int32) 

        self.grid_size = vec2f(domain / cnum)
        self.igrid_size = 1. / self.grid_size
        self.cell_volume = self.calc_volume()
        self.cnum = vec2i(cnum)
        self.gnum = self.cnum + 1
        self.gridSum = int(self.gnum[0] * self.gnum[1])
        self.cellSum = int(self.cnum[0] * self.cnum[1])
        self.set_nodal_coords()

    def set_characteristic_length(self, sims: Simulation):
        if sims.shape_function == "CPDI1" or sims.shape_function == "CPDI2":
            self.calLength = ParticleCPDI.field(shape=sims.max_particle_num)
        else:
            self.calLength = ti.Vector.field(2, float, shape=sims.max_body_num)
            if sims.stabilize == "F-Bar Method":
                self.calLength_lower_order = ti.Vector.field(2, float, shape=sims.max_body_num)

    def set_nodal_coords(self):
        X = np.linspace(0., self.cnum[0] * self.grid_size[0], int(self.gnum[0]))
        Y = np.linspace(0., self.cnum[1] * self.grid_size[1], int(self.gnum[1]))
        self.nodal_coords = flip2d_linear(np.array(list(product(X, Y))), size_u=X.shape[0], size_v=Y.shape[0])

    def element_initialize(self, sims: Simulation, local_coordiates=False):
        self.choose_shape_function(sims)
        if sims.mls:
            self.compute_inertia_tensor(sims.shape_function)

        is_bbar = False
        if sims.stabilize == "B-Bar Method":
            is_bbar = True

            if sims.shape_function == "QuadBSpline" or sims.shape_function == "CubicBSpline":
                raise RuntimeError("B bar method can only used at [Linear, GIMP] Shape Function")
            
        if sims.max_particle_num > 0:
            if sims.gauss_number > 0:
                self.gauss_point = GaussPointInRectangle(gauss_point=sims.gauss_number, dimemsion=2)
                self.gauss_point.create_gauss_point()
            self.set_essential_field(is_bbar, sims)
        
        self.node_connectivity = ti.Vector.field(self.grid_nodes, int, shape=self.get_total_cell_number())
        find_nodes_per_element_(0, self.get_total_cell_number(), self.gnum, self.node_connectivity, set_connectivity)
                
        if sims.solver_type == "Implicit":
            self.pse = PrefixSumExecutor(self.gridSum)
            self.flag = ti.field(int, shape=self.pse.get_length())

    def compute_inertia_tensor(self, shape_function_type):
        raise RuntimeError("MLS has not been verified")
        if shape_function_type == "QuadBSpline":
            pass
        elif shape_function_type == "CubicBSpline": 
            pass

    def choose_shape_function(self, sims: Simulation):
        if sims.shape_function == "Linear":
            self.influenced_node = 2
            self.grid_nodes = 4
            self.ifnum = vec2i(2, 2)

            self.shape_function = ShapeLinear
            self.grad_shape_function = GShapeLinear
            self.shape_function_center = ShapeLinearCenter

        elif sims.shape_function == "GIMP":
            if not sims.is_2DAxisy:
                self.influenced_node = 3
                self.grid_nodes = 9
                self.ifnum = vec2i(3, 3)
                
                self.shape_function = ShapeGIMP
                self.grad_shape_function = GShapeGIMP
                self.shape_function_center = ShapeGIMPCenter
            elif sims.is_2DAxisy:
                self.influenced_node = 3
                self.grid_nodes = 9
                self.ifnum = vec2i(3, 3)

                self.shape_function_r = ShapeAxisyGIMP
                self.shape_function_z = ShapeGIMP
                self.grad_shape_function_r = GShapeAxisyGIMP
                self.grad_shape_function_z = GShapeGIMP
                self.shape_function_center_r = ShapeAixGIMPCenter
                self.shape_function_center_z = ShapeGIMPCenter

        elif sims.shape_function == "QuadBSpline":
            self.influenced_node = 3
            self.grid_nodes = 9
            self.ifnum = vec2i(3, 3)
            
            self.shape_function = ShapeBsplineQ
            self.grad_shape_function = GShapeBsplineQ
            if sims.stabilize == "F-Bar Method":
                self.influenced_node_lower_order = 2
                self.grid_nodes_lower_order = 4
                self.lower_shape_function = ShapeLinear
                self.lower_grad_shape_function = GShapeLinear

        elif sims.shape_function == "CubicBSpline":
            self.influenced_node = 4
            self.grid_nodes = 16
            self.ifnum = vec2i(4, 4)
            
            self.shape_function = ShapeBsplineC
            self.grad_shape_function = GShapeBsplineC
            if sims.stabilize == "F-Bar Method":
                self.influenced_node_lower_order = 3
                self.grid_nodes_lower_order = 9
                self.lower_shape_function = ShapeBsplineQ
                self.lower_grad_shape_function = GShapeBsplineQ
        else:
            raise KeyError(f"The shape function type {sims.shape_function} is not exist!")
        
        self.influenced_dofs = 2 * self.grid_nodes

    def get_nonzero_grids_per_row(self):
        return (2 * self.influenced_node - 1) ** 2 

    def set_essential_field_local_coordinate(self, is_bbar, max_particle_num):
        self.start_local_coord = vec2f([-1, -1])
        self.calculate = self.calc_local_shape_fn

    def set_essential_field(self, is_bbar, sims: Simulation):
        if is_bbar:
            self.LnID = ti.field(int)
            self.shape_fn = ti.field(float)
            self.dshape_fn = ti.Vector.field(2, float)
            self.dshape_fnc = ti.Vector.field(2, float)

            if not sims.is_2DAxisy:
                self.calculate = self.calc_shape_fn_b_bar
                ti.root.dense(ti.i, sims.max_particle_num * self.grid_nodes).place(self.LnID, self.shape_fn, self.dshape_fn, self.dshape_fnc)
            elif sims.is_2DAxisy:
                self.shape_fnc = ti.field(float)
                self.calculate = self.calc_shape_fn_b_bar_axisy
                ti.root.dense(ti.i, sims.max_particle_num * self.grid_nodes).place(self.LnID, self.shape_fn, self.shape_fnc, self.dshape_fn, self.dshape_fnc)
            
        else:
            if not sims.is_2DAxisy:
                self.calculate = self.calc_shape_fn
                if sims.shape_function == "QuadBSpline" or sims.shape_function == "CubicBSpline":
                    self.calculate = self.calc_shape_fn_spline
            elif sims.is_2DAxisy:
                self.calculate = self.calc_shape_fn_axisy
                if sims.shape_function == "QuadBSpline" or sims.shape_function == "CubicBSpline":
                    self.calculate = self.calc_shape_fn_spline

            self.LnID = ti.field(int)
            self.shape_fn = ti.field(float)
            self.dshape_fn = ti.Vector.field(2, float)
            ti.root.dense(ti.i, sims.max_particle_num * self.grid_nodes).place(self.LnID, self.shape_fn, self.dshape_fn)
        self.node_size = ti.field(ti.u8, shape=sims.max_particle_num)

    def calculate_characteristic_length(self, sims: Simulation, particleNum, particle, psize):
        if sims.shape_function == "Linear":
            self.calLength.fill(vec2f(0., 0.))
        if sims.shape_function == "SmoothLinear":
            set_particle_characteristic_length_gimp(particleNum, sims.shape_smooth, self.calLength, particle, psize)
        elif sims.shape_function == "QuadBSpline":
            self.calLength.fill(0.5 * self.grid_size)
            if sims.stabilize == "F-Bar Method":
                self.calLength_lower_order.fill(vec2f(0., 0.))
        elif sims.shape_function == "CubicBSpline":
            self.calLength.fill(self.grid_size)
            if sims.stabilize == "F-Bar Method":
                self.calLength_lower_order.fill(0.5 * self.grid_size)
        elif sims.shape_function == "GIMP":
            set_particle_characteristic_length_gimp(particleNum, 1., self.calLength, particle, psize)
        elif sims.shape_function == "CPDI1" or sims.shape_function == "CPDI2":
            set_particle_characteristic_length_cpdi(particleNum, self.calLength, particle, psize)

    def get_nodal_coords(self):
        return self.nodal_coords

    def activate_gauss_cell(self, sims: Simulation, grid_level):
        if not self.gauss_cell is None:
            print("Warning: Previous cells will be override!")
        self.cell = HexahedronCell.field(shape=(self.get_total_cell_number(), grid_level))
        self.gauss_cell = HexahedronGuassCell.field(shape=(self.get_total_cell_number() * sims.gauss_number ** 2, grid_level))
        activate_cell(self.cell)

    def activate_euler_cell(self):
        if self.element_type == "Staggered":
            if not self.cell is None:
                print("Warning: Previous Euler cells will be override!")
            self.cell = IncompressibleCell()

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
        end_bound = np.maximum(end_bound, start_bound + 1)

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
        return ti.min(self.grid_size[0], self.grid_size[1]) / velocity if velocity > 0. else 10000
    
    def initial_estimate_active_dofs(self, cutoff, node):
        return estimate_active_dofs(cutoff, node)
    
    def find_active_nodes(self, cutoff, node):
        find_active_node(self.gridSum, cutoff, node, self.flag)
        self.pse.run(self.flag)
        return set_active_dofs(self.gridSum, cutoff, node, self.flag)

    def calc_local_shape_fn(self, particleNum, particle):
        update(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, self.start_local_coord, particleNum[0],
               particle, self.calLength, self.LnID, self.node_size, self.shape_fn, self.dshape_fn, self.shape_function, self.grad_shape_function)
        
    def calc_local_shape_fn_b_bar(self, particleNum, particle):
        updatebbar(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, self.start_local_coord, particleNum[0],
                   particle, self.calLength, self.LnID, self.node_size, self.shape_fn, self.dshape_fn, self.dshape_fnc, self.shape_function, self.grad_shape_function, self.shape_function_center)
        
    def calc_shape_fn(self, particleNum, particle):
        global_update(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, 
                      self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.shape_function, self.grad_shape_function)
        
    def calc_shape_fn_axisy(self, particleNum, particle):
        global_update_2DAxisy(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, 
                              self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.shape_function_r, self.shape_function_z, self.grad_shape_function_r, self.grad_shape_function_z)

    def calc_shape_fn_b_bar(self, particleNum, particle):
        global_updatebbar(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, 
                          self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.dshape_fnc, self.shape_function, self.grad_shape_function, self.shape_function_center)
        
    def calc_shape_fn_b_bar_axisy(self, particleNum, particle):
        global_updatebbar_axisy(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, 
                          self.node_size, self.LnID, self.shape_fn, self.shape_fnc, self.dshape_fn, self.dshape_fnc, self.shape_function_r, self.shape_function_z, self.grad_shape_function_r, self.grad_shape_function_z, self.shape_function_center_r, self.shape_function_center_z)

    def calc_shape_fn_spline(self, particleNum, particle):
        global_update_spline(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, 
                             self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.shape_function, self.grad_shape_function, self.boundary_type)
        
    def calc_shape_fn_spline_lower_order(self, particleNum, particle):
        global_update_spline(self.grid_nodes_lower_order, self.influenced_node_lower_order, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength_lower_order, 
                             self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.lower_shape_function, self.lower_grad_shape_function, self.boundary_type)

    def calc_shape_fn_spline_without(self, particleNum, particle):
        global_update_spline_fn(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, 
                                self.node_size, self.LnID, self.shape_fn, self.shape_function, self.boundary_type)
    
    def find_located_cell(self, particleNum, particle):
        kernel_find_located_cell(self.igrid_size, self.gnum, particleNum[0], particle)

    def set_boundary_type(self, sims: Simulation, grid_level):
        if sims.shape_function == "QuadBSpline" or sims.shape_function == "CubicBSpline":
            kernel_set_boundary_type(self.gridSum, grid_level, self.gnum, self.boundary_flag, self.boundary_type)
