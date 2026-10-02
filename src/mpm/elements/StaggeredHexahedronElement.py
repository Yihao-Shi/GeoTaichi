from itertools import product
import taichi as ti
import numpy as np

from src.mpm.elements.HexahedronKernel import set_particle_characteristic_length_gimp
from src.mpm.elements.ElementBase import ElementBase
from src.mpm.Simulation import Simulation
from src.utils.linalg import flip3d_linear
from src.utils.ShapeFunctions import *
from src.utils.TypeDefination import vec3f, vec3i


Threshold = 1e-12
@ti.data_oriented
class StaggeredHexahedronElement(ElementBase):
    def __init__(self) -> None:
        super().__init__()
        self.grid_size = vec3f(2., 2., 2.)
        self.igrid_size = vec3f(0.5, 0.5, 0.5)
        self.gnum = vec3i(0, 0, 0)
        self.cnum = vec3i(0, 0, 0)
        self.ifnum = vec3i(0, 0, 0)
        self.gridSum = 0
        self.cellSum = 0
        self.grid_nodes = 8
        self.influenced_node = 2
        self.influenced_dofs = 24
        self.cell_volume = 0.
        self.cell_active = None
        self.cell = None
        self.LnID = None
        self.shape_fn = None
        self.node_size = None
        self.calculate = None
        self.calLength = None
        self.shape_function = None
        self.grad_shape_function = None

    def create_nodes(self, sims: Simulation, grid_size):
        if isinstance(grid_size, (list, tuple)):
            grid_size = ti.Vector(grid_size)
        
        domain = np.array(sims.domain)
        grid_size = np.array(grid_size)
        gnum = np.floor((domain + Threshold) / grid_size) + 1
        for d in range(3):
            if gnum[d] == 0:
                gnum[d] = 1
        gnum = np.array(2 * (gnum // 2), dtype=np.int32)

        self.grid_size = vec3f(domain / gnum)
        self.igrid_size = 1. / self.grid_size
        self.cell_volume = self.calc_volume()
        self.gnum = vec3i(gnum)
        self.cnum = self.gnum - 1
        self.gridSum = int(self.gnum[0] * self.gnum[1] * self.gnum[2])
        self.cellSum = int(self.cnum[0] * self.cnum[1] * self.cnum[2])
        self.set_nodal_coords()

    def set_characteristic_length(self, sims: Simulation):
        self.calLength = ti.Vector.field(3, float, shape=sims.max_body_num)

    def set_nodal_coords(self):
        X = np.linspace(0., self.cnum[0] * self.grid_size[0], int(self.gnum[0]))
        Y = np.linspace(0., self.cnum[1] * self.grid_size[1], int(self.gnum[1]))
        Z = np.linspace(0., self.cnum[2] * self.grid_size[2], int(self.gnum[2]))
        self.nodal_coords = flip3d_linear(np.array(list(product(X, Y, Z))), size_u=X.shape[0], size_v=Y.shape[0], size_w=Z.shape[0])
    
    def element_initialize(self, sims: Simulation):
        self.choose_shape_function(sims)
        if sims.max_particle_num > 0:
            self.set_essential_field(sims.max_particle_num, sims.shape_function, sims.mls)

    def calc_volume(self):
        volume = 1.
        for i in range(np.array(self.grid_size).shape[0]):
            volume *= self.grid_size[i]
        return volume
    
    def calc_total_particle(self, npic):
        return npic * npic * npic
    
    def calc_particle_size(self, npic):
        return 0.5 * self.grid_size / npic
    
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

        elif sims.shape_function == "QuadBSpline":
            self.influenced_node = 3
            self.grid_nodes = 27
            self.ifnum = vec3i(3, 3, 3)
            
            self.shape_function = ShapeBsplineQ
            self.grad_shape_function = GShapeBsplineQ
        elif sims.shape_function == "CubicBSpline":
            self.influenced_node = 4
            self.grid_nodes = 64
            self.ifnum = vec3i(4, 4, 4)
            
            self.shape_function = ShapeBsplineC
            self.grad_shape_function = GShapeBsplineC
        else:
            raise KeyError(f"The shape function type {sims.shape_function} is not exist!")
        
        self.influenced_dofs = 3 * self.grid_nodes

    def set_essential_field(self, max_particle_num, shape_function, mls):
        if shape_function == "QuadBSpline" or shape_function == "CubicBSpline":
            self.calculate = self.calc_shape_fn_spline
        else:
            self.calculate = self.calc_shape_fn

        self.LnID = ti.field(int)
        self.shape_fn = ti.field(float)
        if mls:
            ti.root.dense(ti.i, max_particle_num * self.grid_nodes).place(self.LnID, self.shape_fn)
        else:
            self.dshape_fn = ti.Vector.field(3, float)
            ti.root.dense(ti.i, max_particle_num * self.grid_nodes).place(self.LnID, self.shape_fn, self.dshape_fn)
        self.node_size = ti.field(ti.u8, shape=max_particle_num)

    def calculate_characteristic_length(self, sims: Simulation, particleNum, particle, psize):
        if sims.shape_function == "Linear":
            self.calLength.fill(vec3f(0., 0., 0.))
        elif sims.shape_function == "QuadBSpline":
            self.calLength.fill(0.5 * self.grid_size)
        elif sims.shape_function == "CubicBSpline":
            self.calLength.fill(self.grid_size)
        elif sims.shape_function == "GIMP":
            set_particle_characteristic_length_gimp(particleNum, 1., self.calLength, particle, psize)

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
    
    def calc_shape_fn(self, particleNum, particle):
        self.global_update_stagger(self.grid_nodes, self.influenced_node, self.grid_size, self.igrid_size, self.gnum, particleNum[0], particle, self.calLength, 
                              self.node_size, self.LnID, self.shape_fn, self.dshape_fn, self.shape_function, self.grad_shape_function)
        
    def calc_shape_fn_spline(self, particleNum, particle):
        pass

    @ti.kernel
    def global_update_stagger(self, total_nodes: int, influenced_node: int, element_size: ti.types.vector(3, float), ielement_size: ti.types.vector(3, float), gnum: ti.types.vector(3, int), particleNum: int, particle: ti.template(), 
                              calLength: ti.template(), node_size: ti.template(), LnID: ti.template(), shape_fn: ti.template(), dshape_fn: ti.template(), shape_function: ti.template(), grad_shape_function: ti.template()):
        for np in range(particleNum):
            position, psize = particle[np].x, calLength[int(particle[np].bodyID)]
            base_bound = self.calc_base_cell(ielement_size, psize, position)
            activeID = np * total_nodes
            for k in range(base_bound[2], base_bound[2] + influenced_node):
                if k < 0 or k >= gnum[2]: continue
                for j in range(base_bound[1], base_bound[1] + influenced_node):
                    if j < 0 or j >= gnum[1]: continue
                    for i in range(base_bound[0], base_bound[0] + influenced_node):
                        if i < 0 or i >= gnum[0]: continue
                        nodeID = int(i + j * gnum[0] + k * gnum[0] * gnum[1])
                        node_coords = vec3i(i, j, k) * element_size
                        shapen0, shapen1, shapen2 = self.shapefn(particle[np].x, node_coords, ielement_size, psize, shape_function)
                        shapeval = shapen0 * shapen1 * shapen2
                        if shapeval > Threshold:
                            dshapen0, dshapen1, dshapen2 = self.grad_shapefn(particle[np].x, node_coords, ielement_size, psize, grad_shape_function)
                            grad_shapeval = vec3f([dshapen0 * shapen1 * shapen2, shapen0 * dshapen1 * shapen2, shapen0 * shapen1 * dshapen2])
                            LnID[activeID] = nodeID
                            shape_fn[activeID]=shapeval
                            dshape_fn[activeID]=grad_shapeval
                            activeID += 1
            node_size[np] = ti.u8(activeID - np * total_nodes)

    @ti.func
    def calc_base_cell(self, ielement_size, particle_size, position):
        return ti.floor((position - particle_size) * ielement_size, int) 
    
    @ti.func
    def shapefn(natural_particle_position, natural_coords, ielement_size, natural_particle_size, shape_function: ti.template()):
        shapen0 = shape_function(natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0])
        shapen1 = shape_function(natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1])
        shapen2 = shape_function(natural_particle_position[2], natural_coords[2], ielement_size[2], natural_particle_size[2])
        return shapen0, shapen1, shapen2

    @ti.func
    def grad_shapefn(natural_particle_position, natural_coords, ielement_size, natural_particle_size, grad_shape_function: ti.template()):
        dshapen0 = grad_shape_function(natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0])
        dshapen1 = grad_shape_function(natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1])
        dshapen2 = grad_shape_function(natural_particle_position[2], natural_coords[2], ielement_size[2], natural_particle_size[2])
        return dshapen0, dshapen1, dshapen2

    @ti.func
    def cal_min_grid_size(self):
        return ti.min(self.grid_size[0], self.grid_size[1], self.grid_size[2])