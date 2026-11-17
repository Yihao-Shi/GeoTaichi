from src.utils.errors import ConfigurationError, SimulationStateError
import copy, math
import numpy as np

from src.utils.ObjectIO import DictIO
from src.utils.linalg import cartesian_product, linearize, bi_interpolate

EPSILON = 2.2204460492503131e-16


class LocalGrid(object):
    def __init__(self) -> None:
        self.start_point = np.zeros(3)
        self.region_size = np.zeros(3)

        self.grid_resolution = [80, 80, 80]
        self.gnum = np.zeros(3, dtype=np.int32)
        self.gridSum = 0
        self.grid_space = 0.0
        self.extent = 0

        self.distance_field = None
        self.node_coord = None

    def clear(self):
        self.gnum = np.zeros(3, dtype=np.int32)
        self.gridSum = 0
        self.distance_field = None
        self.node_coord = None

    def min_box(self):
        return self.start_point

    def max_box(self):
        return self.start_point + self.region_size

    def box_size(self):
        return self.region_size

    def read_grid(self, **template_dict):
        self.grid_space = DictIO.get_essential(template_dict, "space")
        self.extent = DictIO.get_alternative(template_dict, "extent", 0)

    def set_distance_field(self, start_point, region_size, node_cood, distance_field, center_mass):
        self.start_point = start_point - center_mass
        self.region_size = region_size - center_mass
        self.node_coord = node_cood - center_mass
        self.distance_field = distance_field

        if not self.distance_field is None:
            if isinstance(self.distance_field, list):
                self.distance_field = np.array(self.distance_field)

            if not isinstance(self.distance_field, np.ndarray):
                raise SimulationStateError("Keyword:: /DistanceField/ should be np.ndarray or list")
            self.check_field_shape()
            self.check_coord_shape()
        else:
            raise SimulationStateError("Keyword conflict:: either of /DistanceField/ or /Object/ should not None")

    def check_field_shape(self):
        if self.distance_field.ndim == 3:
            self.distance_field = self.distance_field.flatten()
        if self.distance_field.ndim != 1:
            raise SimulationStateError(
                f"The shape of Keyword:: The dimension of /DistanceField/ is {self.distance_field.ndim}. Invalid!"
            )
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
            raise SimulationStateError(
                f"The shape of Keyword:: The dimension of /NodeCoordination/ is {self.node_coord.shape[1]}. Invalid!"
            )
        if self.gridSum != self.node_coord.shape[0]:
            raise SimulationStateError(
                f"The shape of Keyword:: /DistanceField/ is {self.distance_field.shape[0]} is inconsistent with /NodeCoordination/ {self.node_coord.shape[0]}. Invalid!"
            )

    def set_grid(self, start_point, region_size):
        if np.linalg.norm(region_size) == 0.0:
            raise SimulationStateError("KeyWord:: /RegionSize/ has not been set")

        mid_point = start_point + 0.5 * region_size
        half_size = 0.5 * region_size
        igrid_space = 1.0 / self.grid_space
        n_int = np.array([int(math.ceil(half_size[i] * igrid_space) + self.extent) for i in range(3)])

        self.start_point = mid_point - n_int * self.grid_space
        self.region_size = 2.0 * n_int * self.grid_space
        self.gnum = np.array(
            [int(math.ceil(round(self.region_size[i] * igrid_space, 10) + 1)) for i in range(3)], dtype=np.int32
        )
        self.gridSum = int(self.gnum[0] * self.gnum[1] * self.gnum[2])

        if self.distance_field is not None:
            if self.gridSum != self.distance_field.shape[0]:
                raise SimulationStateError(
                    f"The shape of Keyword:: /DistanceField/ is {self.distance_field.shape[0]} is inconsistent with /NodeCoordination/ {self.gridSum}. Invalid!"
                )

    def get_cell_volume(self):
        return self.grid_space**3

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
        self.node_coord = cartesian_product(gx, gy, gz, order="x")

    def generate_sdf(self, distance_field):
        self.distance_field = distance_field
        if int(self.distance_field.size) != self.gridSum:
            raise ConfigurationError(
                "There is a size-inconsistency between the current level set grid and shape.distField for this body! The level set grid has changed \
                                    since the creation of this body, this is not supported."
            )

    def closet_corner(self, point):
        ret_indices = np.zeros(point.shape, dtype=np.int32)
        temp = np.zeros(point.shape[0])
        for index in range(3):
            ret_indices[:, index] = np.clip(
                np.floor((point[:, index] - self.start_point[index]) / self.grid_space),
                temp,
                temp + self.gnum[index] - 2,
            )
        return ret_indices

    def distance(self, p):
        indices = self.closet_corner(p)
        x_ind, y_ind, z_ind = indices[:, 0].reshape(-1), indices[:, 1].reshape(-1), indices[:, 2].reshape(-1)
        n_g_px, n_g_py, n_g_pz = self.gnum

        ind000 = linearize(x_ind, y_ind, z_ind, n_g_px, n_g_py)
        ind001 = linearize(x_ind, y_ind, z_ind + 1, n_g_px, n_g_py)
        ind010 = linearize(x_ind, y_ind + 1, z_ind, n_g_px, n_g_py)
        ind011 = linearize(x_ind, y_ind + 1, z_ind + 1, n_g_px, n_g_py)
        ind100 = linearize(x_ind + 1, y_ind, z_ind, n_g_px, n_g_py)
        ind101 = linearize(x_ind + 1, y_ind, z_ind + 1, n_g_px, n_g_py)
        ind110 = linearize(x_ind + 1, y_ind + 1, z_ind, n_g_px, n_g_py)
        ind111 = linearize(x_ind + 1, y_ind + 1, z_ind + 1, n_g_px, n_g_py)

        temp1 = self.node_coord[ind000]
        temp2 = self.node_coord[ind010]
        temp3 = self.node_coord[ind001]
        yz_coord = np.array([p[:, 1], p[:, 2]]).T
        y_extr = np.array([temp1[:, 1], temp2[:, 1]]).T
        z_extr = np.array([temp1[:, 2], temp3[:, 2]]).T

        known_valx0, known_valx1 = np.zeros((p.shape[0], 2, 2)), np.zeros((p.shape[0], 2, 2))
        known_valx0[:, 0, 0] = self.distance_field[ind000]
        known_valx0[:, 0, 1] = self.distance_field[ind001]
        known_valx0[:, 1, 0] = self.distance_field[ind010]
        known_valx0[:, 1, 1] = self.distance_field[ind011]

        known_valx1[:, 0, 0] = self.distance_field[ind100]
        known_valx1[:, 0, 1] = self.distance_field[ind101]
        known_valx1[:, 1, 0] = self.distance_field[ind110]
        known_valx1[:, 1, 1] = self.distance_field[ind111]

        f0yz = bi_interpolate(yz_coord, y_extr, z_extr, known_valx0)
        f1yz = bi_interpolate(yz_coord, y_extr, z_extr, known_valx1)
        return (p[:, 0] - temp1[:, 0]) / self.grid_space * (f1yz - f0yz) + f0yz

    def move(self, mass_of_center):
        self.node_coord -= mass_of_center
