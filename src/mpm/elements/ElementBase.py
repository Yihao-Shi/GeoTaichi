import taichi as ti

from src.mpm.Simulation import Simulation
from src.utils.TypeDefination import u1
from src.utils.linalg import round32


class ElementBase(object):
    def __init__(self, element_type, grid_level, ghost_cell) -> None:
        self.mesh = None
        self.multi_grids = None
        self.resolution = 0.1
        self.gridSum = 0
        self.cellSum = 0
        self.gnum = [0, 0, 0]
        self.cnum = [0, 0, 0]
        self.ghost_cell = ghost_cell

        self.boundary_type = None
        self.boundary_flag = None
        self.node_connectivity = None
        self.ti_nodal_coords = None
        self.element_type = element_type
        self.grid_level = grid_level

        if element_type != "Staggered":
            self.ghost_cell = 0

        self.pse = None
        self.flag = None

    def calculate_basis_function(self, sims: Simulation, grid_level):
        if sims.shape_function == "QuadBSpline" or sims.shape_function == "CubicBSpline":
            if sims.dimension == 2:
                self.boundary_type = ti.Vector.field(2, ti.u8, shape=(self.gridSum, grid_level))
            elif sims.dimension == 3:
                self.boundary_type = ti.Vector.field(3, ti.u8, shape=(self.gridSum, grid_level))
            self.boundary_flag = ti.field(u1)
            ti.root.dense(ti.i, round32(self.gridSum * grid_level * sims.dimension)//32).quant_array(ti.i, dimensions=32, max_num_bits=32).place(self.boundary_flag)
        else:
            self.boundary_type = ti.field(ti.u8, shape=(1, 1))

    def create_nodes(self, *args):
        raise NotImplementedError

    def calc_volume(self):
        raise NotImplementedError
    
    def get_element_number(self):
        raise NotImplementedError

    def calc_total_particle(self, *args):
        raise NotImplementedError

    def calc_particle_size(self, *args):
        raise NotImplementedError
    
    def get_total_cell_number(self):
        raise NotImplementedError
    
    def activate_cell(self, *args):
        raise NotImplementedError
    
    def set_up_cell_active_flag(self, *args):
        raise NotImplementedError
    
    def update_particle_in_cell(self, *args):
        raise NotImplementedError
    
    def activate_gauss_cell(self, *args):
        raise NotImplementedError
    
    def activate_euler_cell(self, *args):
        raise NotImplementedError
    
    def set_boundary_type(self, *args):
        raise NotImplementedError

    def get_boundary_nodes(self, *args):
        raise NotImplementedError
    
    def calculate_characteristic_length(self, *args):
        raise NotImplementedError
    
    def calc_shape_fn_spline_lower_order(self, *args):
        raise NotImplementedError