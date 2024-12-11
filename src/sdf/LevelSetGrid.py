import copy, math
import numpy as np

from src.utils.ObjectIO import DictIO
from src.utils.linalg import cartesian_product, linearize, biInterpolate

EPSILON = 2.2204460492503131e-16

class LocalGrid(object):
    def __init__(self) -> None:
        self.start_point = np.zeros(3)
        self.region_size = np.zeros(3)

        self.grid_resolution = [80, 80, 80]
        self.gnum = np.zeros(3, dtype=np.int32)
        self.gridSum = 0
        self.grid_space = 0.
        self.extent = 0

        self.distance_field = None
        self.node_coord = None

    def clear(self):
        self.gnum = np.zeros(3, dtype=np.int32)
        self.gridSum = 0
        self.distance_field = None
        self.node_coord = None

    def minBox(self):
        return self.start_point
    
    def maxBox(self):
        return self.start_point + self.region_size

    def read_grid(self, **template_dict):
        self.grid_space = DictIO.GetEssential(template_dict, "space")
        self.extent = DictIO.GetAlternative(template_dict, "extent", 0)
    
    def set_distance_field(self, start_point, region_size, node_cood, distance_field, center_mass):
        self.start_point = start_point - center_mass
        self.region_size = region_size - center_mass
        self.node_coord = node_cood - center_mass
        self.distance_field = distance_field

        if not self.distance_field is None:
            if isinstance(self.distance_field, list):
                self.distance_field = np.array(self.distance_field)

            if not isinstance(self.distance_field, np.ndarray):
                raise RuntimeError("Keyword:: /DistanceField/ should be np.ndarray or list")
            self.check_field_shape()
            self.check_coord_shape()
        else:
            raise RuntimeError("Keyword conflict:: either of /DistanceField/ or /Object/ should not None")
                
    def check_field_shape(self):
        if self.distance_field.ndim == 3:
            self.distance_field = self.distance_field.flatten()
        if self.distance_field.ndim != 1:
            raise RuntimeError(f"The shape of Keyword:: The dimension of /DistanceField/ is {self.distance_field.ndim}. Invalid!")
        self.gridSum = self.distance_field.shape[0]
                
    def check_coord_shape(self):    
        if isinstance(self.node_coord, list):
            self.node_coord = np.array(self.node_coord)

        if self.node_coord.ndim == 4:
            dims = self.node_coord.shape[3]
            nsize = int(self.node_coord.shape[0] * self.node_coord.shape[1] * self.node_coord.shape[2])
            self.node_coord = self.node_coord.reshape(nsize, dims)

        nsize = self.node_coord.shape[0]
        if self.node_coord.shape[1] != 3:
            raise RuntimeError(f"The shape of Keyword:: The dimension of /NodeCoordination/ is {self.node_coord.shape[1]}. Invalid!")
        if self.gridSum != self.node_coord.shape[0]:
            raise RuntimeError(f"The shape of Keyword:: /DistanceField/ is {self.distance_field.shape[0]} is inconsistent with /NodeCoordination/ {self.node_coord.shape[0]}. Invalid!")

    def set_grid(self, start_point, region_size):
        if np.linalg.norm(region_size) == 0.: raise RuntimeError("KeyWord:: /RegionSize/ has not been set")

        mid_point = start_point + 0.5 * region_size
        half_size = 0.5 * region_size
        igrid_space = 1. / self.grid_space
        nInt = np.array([int(math.ceil(half_size[i] * igrid_space) + self.extent) for i in range(3)])  
           
        self.start_point = mid_point - nInt * self.grid_space
        self.region_size = 2. * nInt * self.grid_space
        self.gnum = np.array([int(math.ceil(round(self.region_size[i] * igrid_space, 10) + 1)) for i in range(3)], dtype=np.int32)  
        self.gridSum = int(self.gnum[0] * self.gnum[1] * self.gnum[2])

        if self.distance_field is not None:
            if self.gridSum != self.distance_field.shape[0]:
                raise RuntimeError(f"The shape of Keyword:: /DistanceField/ is {self.distance_field.shape[0]} is inconsistent with /NodeCoordination/ {self.gridSum}. Invalid!")

    def get_cell_volume(self):
        return self.grid_space ** 3
    
    def build_node_coords(self):
        nx = self.gnum[0]
        ny = self.gnum[1]
        nz = self.gnum[2]

        start_position = copy.deepcopy(self.start_point)
        end_position = copy.deepcopy(self.start_point + self.region_size)

        depsilonx = 1e-3 * self.region_size[0] / nx
        depsilony = 1e-3 * self.region_size[0] / ny
        depsilonz = 1e-3 * self.region_size[0] / nz

        gx = np.arange(start_position[0], end_position[0] + depsilonx, self.grid_space)
        gy = np.arange(start_position[1], end_position[1] + depsilony, self.grid_space)
        gz = np.arange(start_position[2], end_position[2] + depsilonz, self.grid_space)
        self.node_coord = cartesian_product(gx, gy, gz, order='x')
                
    def generate_sdf(self, distance_field):
        self.distance_field = distance_field
        if int(self.distance_field.size) != self.gridSum:
                raise RuntimeError("There is a size-inconsistency between the current level set grid and shape.distField for this body! The level set grid has changed \
                                    since the creation of this body, this is not supported.")
        
    def closet_corner(self, point):
        retIndices = np.zeros(point.shape, dtype=np.int32)
        temp = np.zeros(point.shape[0])
        for index in range(3):
            retIndices[:, index] = np.clip(np.floor((point[:, index] - self.start_point[index]) / self.grid_space), temp, temp + self.gnum[index] - 2)
        return retIndices
    
    def distance(self, p):
        indices = self.closet_corner(p)
        xInd, yInd, zInd = indices[:, 0].reshape(-1), indices[:, 1].reshape(-1), indices[:, 2].reshape(-1)
        nGPx, nGPy, nGPz = self.gnum

        Ind000 = linearize(xInd, yInd, zInd, nGPx, nGPy)
        Ind001 = linearize(xInd, yInd, zInd + 1, nGPx, nGPy) 
        Ind010 = linearize(xInd, yInd + 1, zInd, nGPx, nGPy) 
        Ind011 = linearize(xInd, yInd + 1, zInd + 1, nGPx, nGPy) 
        Ind100 = linearize(xInd + 1, yInd, zInd, nGPx, nGPy) 
        Ind101 = linearize(xInd + 1, yInd, zInd + 1, nGPx, nGPy) 
        Ind110 = linearize(xInd + 1, yInd + 1, zInd, nGPx, nGPy) 
        Ind111 = linearize(xInd + 1, yInd + 1, zInd + 1, nGPx, nGPy) 

        temp1 = self.node_coord[Ind000]
        temp2 = self.node_coord[Ind010]
        temp3 = self.node_coord[Ind001]
        yzCoord = np.array([p[:, 1], p[:, 2]]).T
        yExtr = np.array([temp1[:, 1], temp2[:, 1]]).T
        zExtr = np.array([temp1[:, 2], temp3[:, 2]]).T

        knownValx0, knownValx1 = np.zeros((p.shape[0], 2, 2)), np.zeros((p.shape[0], 2, 2))
        knownValx0[:, 0, 0] = self.distance_field[Ind000]
        knownValx0[:, 0, 1] = self.distance_field[Ind001]
        knownValx0[:, 1, 0] = self.distance_field[Ind010]
        knownValx0[:, 1, 1] = self.distance_field[Ind011]

        knownValx1[:, 0, 0] = self.distance_field[Ind100]
        knownValx1[:, 0, 1] = self.distance_field[Ind101]
        knownValx1[:, 1, 0] = self.distance_field[Ind110]
        knownValx1[:, 1, 1] = self.distance_field[Ind111]

        f0yz = biInterpolate(yzCoord, yExtr, zExtr, knownValx0)
        f1yz = biInterpolate(yzCoord, yExtr, zExtr, knownValx1)
        return (p[:, 0] - temp1[:, 0]) / self.grid_space * (f1yz - f0yz) + f0yz

    def move(self, mass_of_center):
        self.node_coord -= mass_of_center