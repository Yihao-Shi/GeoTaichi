## @package VolumeToMesh converts sampled sdf data to mesh
#
import taichi as ti
import numpy as np
from src.vdb_grid import *
from src.vdb_math import *


@ti.data_oriented
class VolumeToMesh:
    ## @detail Grids are arranged so that:
    # Grid Index: [0, 0, 0]      <-------> The grid at lower-left corner
    # Grid Vert Index: [0, 0, 0] <-------> The vertex coordinate at the lower-left corner
    # Each grid has vertex id from 0 - 7 indicating the 8 vertices, and the ids are arrange as:
    #
    #
    #       4 ---------- 5
    #       / |        /|
    #      /  |       / |
    #     7----------6  |
    #     |   |      |  |
    #     |   0------|--1
    #     |  /       | /
    #     | /        |/
    #     3----------2
    # 
    # 3 -> 2 indicates the positive x direction
    # 3 -> 0 indicates the positive y direction
    # 3 -> 7 indicates the positive z direction

    class Internal:
        tri_table_data = \
            np.array(
                [
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
                    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
                    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
                    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
                    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
                    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
                    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
                    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
                    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
                    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
                    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
                    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
                    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
                    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
                    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
                    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
                    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
                    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
                    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
                    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
                    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
                    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
                    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
                    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
                    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
                    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
                    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
                    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
                    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
                    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
                    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
                    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
                    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
                    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
                    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
                    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
                    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
                    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
                    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
                    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
                    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
                    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
                    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
                    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
                    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
                    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
                    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
                    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
                    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
                    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
                    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
                    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
                    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
                    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
                    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
                    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
                    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
                    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
                    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
                    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
                    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
                    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
                    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
                    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
                    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
                    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
                    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
                    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
                    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
                    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
                    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
                    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
                    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
                    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
                    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
                    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
                    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
                    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
                    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
                    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
                    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
                    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
                    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
                    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
                    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
                    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
                    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
                    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
                    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
                    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
                    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
                    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
                    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
                    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
                    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
                    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
                    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
                    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
                    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
                    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
                    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
                    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
                    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
                    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
                    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
                    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
                    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
                    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
                    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
                    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
                    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
                    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
                    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
                    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
                    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
                    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
                    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
                    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
                    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
                    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
                    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
                    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
                    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
                    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
                    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
                    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
                    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
                    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
                    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
                    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
                    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
                    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
                    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
                    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
                    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
                    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
                    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
                    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
                    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
                    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
                    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
                    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
                    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
                    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
                    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
                    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
                    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
                    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
                    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
                    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
                    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
                    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
                    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
                    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
                    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
                    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
                    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
                    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
                    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
                    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
                    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
                    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
                    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
                    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
                    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
                    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
                    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
                    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
                    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
                    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
                    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
                    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
                    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
                    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
                    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
                    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
                    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
                    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
                    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
                    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
                    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
                    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
                    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
                    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                ]
            )
        edge_table_data = \
            np.array(
                [
                    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
                    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
                    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
                    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
                    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
                    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
                    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
                    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
                    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
                    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
                    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
                    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
                    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
                    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
                    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
                    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
                    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
                    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
                    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
                    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
                    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
                    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
                    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
                    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
                    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
                    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
                    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
                    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
                    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
                    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
                    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
                    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0]
            )

    triangle_table = ti.Vector.field(n=16, dtype=ti.i32, shape=256)
    edge_table = ti.field(ti.i32, shape=256)

    triangle_table.from_numpy(Internal.tri_table_data)
    edge_table.from_numpy(Internal.edge_table_data)

    vertex_offset = ti.Vector.field(n=3, dtype=ti.i32, shape=8)
    vertex_offset[3] = ti.Vector([0, 0, 0])
    vertex_offset[2] = ti.Vector([1, 0, 0])
    vertex_offset[0] = ti.Vector([0, 1, 0])
    vertex_offset[7] = ti.Vector([0, 0, 1])
    vertex_offset[1] = ti.Vector([1, 1, 0])
    vertex_offset[6] = ti.Vector([1, 0, 1])
    vertex_offset[4] = ti.Vector([0, 1, 1])
    vertex_offset[5] = ti.Vector([1, 1, 1])

    @staticmethod
    def marching_cube(sdf: VdbGrid, aux: VdbGrid, isovalue: ti.f32, num_vertices: ti.template(),
                      vertices: ti.template(), num_indices: ti.template(), indices: ti.template(), normals: ti.template() = None):
        # Step 1: Mark all cells to process
        @ti.kernel
        def mark_cell():
            for i, j, k in sdf.data_wrapper.leaf_value:
                for dx, dy, dz in ti.static(ti.ndrange((-1, 1), (-1, 1), (-1, 1))):
                    aux.add_value_world(i + dx, j + dy, k + dz, 1)

        mark_cell()

        @ti.func
        def get_interpolate_indices(edge_index: ti.template(), is_compile_time: ti.template() = 1):
            res = ti.Vector.zero(dt=ti.i32, n=2)

            if ti.static(edge_index == 0):
                res = ti.Vector([0, 1])
            elif ti.static(edge_index == 1):
                res = ti.Vector([1, 2])
            elif ti.static(edge_index == 2):
                res = ti.Vector([2, 3])
            elif ti.static(edge_index == 3):
                res = ti.Vector([3, 0])
            elif ti.static(edge_index == 4):
                res = ti.Vector([4, 5])
            elif ti.static(edge_index == 5):
                res = ti.Vector([5, 6])
            elif ti.static(edge_index == 6):
                res = ti.Vector([6, 7])
            elif ti.static(edge_index == 7):
                res = ti.Vector([7, 4])
            elif ti.static(edge_index == 8):
                res = ti.Vector([0, 4])
            elif ti.static(edge_index == 9):
                res = ti.Vector([1, 5])
            elif ti.static(edge_index == 10):
                res = ti.Vector([2, 6])
            elif ti.static(edge_index == 11):
                res = ti.Vector([3, 7])

            return res

        @ti.func
        def vertex_interpolate(isolevel, p1, p2, val1, val2, eps=0.00001):
            offset = 0.0
            delta = val2 - val1

            if ti.abs(delta) < eps:
                offset = 0.5
            else:
                offset = (isolevel - val1) / delta
            return p1 + offset * (p2 - p1)

        # Step 2: Process all cells
        @ti.kernel
        def marching_cube_impl():
            for i, j, k in aux.data_wrapper.leaf_value:
                cube_index = 0
                for w in ti.static(range(8)):
                    offset = VolumeToMesh.vertex_offset[w]
                    if sdf.read_value_world(i + offset[0], j + offset[1], k + offset[2]) < isovalue:
                        cube_index |= ti.static(1 << w)

                if VolumeToMesh.edge_table[cube_index] != 0:
                    triangle_vertex_indices = ti.Vector.zero(dt=ti.i32, n=12)
                    cube_voxel_coord = ti.Vector([i, j, k])
                    for w in ti.static(range(12)):
                        edge_byte = ti.static(1 << w)
                        if VolumeToMesh.edge_table[cube_index] & edge_byte:
                            vertex_index = ti.atomic_add(num_vertices[None], 1)
                            interpolate_index0, interpolate_index1 = get_interpolate_indices(w)
                            vertex0 = cube_voxel_coord + VolumeToMesh.vertex_offset[interpolate_index0]
                            vertex1 = cube_voxel_coord + VolumeToMesh.vertex_offset[interpolate_index1]

                            vertices[vertex_index] = vertex_interpolate(isovalue,
                                                                        sdf.transform.voxel_to_coord_packed(vertex0),
                                                                        sdf.transform.voxel_to_coord_packed(vertex1),
                                                                        sdf.read_value_world(vertex0[0], vertex0[1],
                                                                                             vertex0[2]),
                                                                        sdf.read_value_world(vertex1[0], vertex1[1],
                                                                                             vertex1[2]),
                                                                        )
                            triangle_vertex_indices[w] = vertex_index

                    cube_triangle_data = VolumeToMesh.triangle_table[cube_index]
                    for p in ti.static(range(5)):
                        w = ti.static(p * 3)
                        if cube_triangle_data[w] != -1:
                            indices_index = ti.atomic_add(num_indices[None], 3)
                            indices[indices_index + 2] = triangle_vertex_indices[cube_triangle_data[w]]
                            indices[indices_index + 1] = triangle_vertex_indices[cube_triangle_data[w + 1]]
                            indices[indices_index + 0] = triangle_vertex_indices[cube_triangle_data[w + 2]]

        marching_cube_impl()

        # Step 3: Process normals
        if normals is not None:


            @ti.func
            def compute_face_normal(vert0: ti.template(), vert1: ti.template(), vert2: ti.template()):
                p = vert2 - vert0
                q = vert2 - vert1
                return ti.math.normalize(q.cross(p))
            @ti.kernel
            def process_normal():
                for i in range(num_indices[None] // 3):
                    ind0 = indices[i * 3]
                    ind1 = indices[i * 3 + 1]
                    ind2 = indices[i * 3 + 2]

                    vert0 = vertices[ind0]
                    vert1 = vertices[ind1]
                    vert2 = vertices[ind2]
                    face_normal = compute_face_normal(vert0, vert1, vert2)
                    normal_buffer = ti.Vector([face_normal[0], face_normal[1], face_normal[2], 1])
                    normals[ind0] += normal_buffer
                    normals[ind1] += normal_buffer
                    normals[ind2] += normal_buffer

                for i in range(num_vertices[None]):
                    normals[i] /= normals[i][3]

            process_normal()


    @staticmethod
    def dual_contouring(sdf: VdbGrid, aux: VdbGrid, isovalue: ti.f32, num_vertices: ti.template(),
                      vertices: ti.template(), num_indices: ti.template(), indices: ti.template(),
                        normals: ti.template() = None):

        # Step 1: Mark all cells to process
        @ti.kernel
        def mark_cell():
            for i, j, k in sdf.data_wrapper.leaf_value:
                for dx, dy, dz in ti.static(ti.ndrange((-1, 1), (-1, 1), (-1, 1))):
                    aux.add_value_world(i + dx, j + dy, k + dz, 1)

        mark_cell()

        # Step 2: Compute points in each cell

        @ti.func
        def get_interpolate_indices(edge_index: ti.template(), is_compile_time: ti.template() = 1):
            res = ti.Vector.zero(dt=ti.i32, n=2)
            if ti.static(is_compile_time == 1):
                if ti.static(edge_index == 0):
                    res = ti.Vector([0, 1])
                elif ti.static(edge_index == 1):
                    res = ti.Vector([1, 2])
                elif ti.static(edge_index == 2):
                    res = ti.Vector([2, 3])
                elif ti.static(edge_index == 3):
                    res = ti.Vector([3, 0])
                elif ti.static(edge_index == 4):
                    res = ti.Vector([4, 5])
                elif ti.static(edge_index == 5):
                    res = ti.Vector([5, 6])
                elif ti.static(edge_index == 6):
                    res = ti.Vector([6, 7])
                elif ti.static(edge_index == 7):
                    res = ti.Vector([7, 4])
                elif ti.static(edge_index == 8):
                    res = ti.Vector([0, 4])
                elif ti.static(edge_index == 9):
                    res = ti.Vector([1, 5])
                elif ti.static(edge_index == 10):
                    res = ti.Vector([2, 6])
                elif ti.static(edge_index == 11):
                    res = ti.Vector([3, 7])
            else:
                if edge_index == 0:
                    res = ti.Vector([0, 1])
                elif edge_index == 1:
                    res = ti.Vector([1, 2])
                elif edge_index == 2:
                    res = ti.Vector([2, 3])
                elif edge_index == 3:
                    res = ti.Vector([3, 0])
                elif edge_index == 4:
                    res = ti.Vector([4, 5])
                elif edge_index == 5:
                    res = ti.Vector([5, 6])
                elif edge_index == 6:
                    res = ti.Vector([6, 7])
                elif edge_index == 7:
                    res = ti.Vector([7, 4])
                elif edge_index == 8:
                    res = ti.Vector([0, 4])
                elif edge_index == 9:
                    res = ti.Vector([1, 5])
                elif edge_index == 10:
                    res = ti.Vector([2, 6])
                elif edge_index == 11:
                    res = ti.Vector([3, 7])

            return res

        @ti.func
        def vertex_interpolate(isovalue, p1, p2, val1, val2, eps=0.00001):
            mu = (isovalue - val1) / (val2 - val1)
            return p1 + mu * (p2 - p1)
        @ti.func
        def trilinear_interpolate(x_d, y_d, z_d, c000, c001, c010, c100, c011, c101, c110, c111):
            c00 = c000 * (1 - x_d) + c100 * x_d
            c01 = c001 * (1 - x_d) + c101 * x_d
            c10 = c010 * (1 - x_d) + c110 * x_d
            c11 = c011 * (1 - x_d) + c111 * x_d

            c0 = c00 * (1 - y_d) + c10 * y_d
            c1 = c01 * (1 - y_d) + c11 * y_d

            return c0 * (1 - z_d) + c1 * z_d

        @ti.func
        def compute_vertex_normal(x, y, z):
            normal = ti.Vector.zero(dt=ti.f32, n=3)
            if x == 0:
                normal[0] = -sdf.read_value_world(x + 1, y, z) * sdf.transform.voxel_dim[0]
            else:
                normal[0] = (sdf.read_value_world(x - 1, y, z) - sdf.read_value_world(x + 1, y, z)) * 0.5 * sdf.transform.voxel_dim[0]

            if y == 0:
                normal[1] = -sdf.read_value_world(x, y + 1, z) * sdf.transform.voxel_dim[1]
            else:
                normal[1] = (sdf.read_value_world(x, y - 1, z) - sdf.read_value_world(x, y + 1, z)) * 0.5 * sdf.transform.voxel_dim[1]

            if z == 0:
                normal[2] = -sdf.read_value_world(x, y, z + 1) * sdf.transform.voxel_dim[2]
            else:
                normal[2] = (sdf.read_value_world(x, y, z - 1) - sdf.read_value_world(x, y, z + 1)) * 0.5 * sdf.transform.voxel_dim[2]

            return ti.math.normalize(normal)

        @ti.func
        def compute_normal_at(pos: ti.template()):

            i, j, k = sdf.transform.coord_to_voxel_packed(pos)
            x_d, y_d, z_d = (pos - sdf.transform.voxel_to_coord(i, j, k)) / sdf.transform.voxel_dim

            c000 = compute_vertex_normal(i, j, k)
            c100 = compute_vertex_normal(i + 1, j, k)
            c010 = compute_vertex_normal(i, j + 1, k)
            c001 = compute_vertex_normal(i, j, k + 1)
            c110 = compute_vertex_normal(i + 1, j + 1, k)
            c011 = compute_vertex_normal(i, j + 1, k + 1)
            c101 = compute_vertex_normal(i + 1, j, k + 1)
            c111 = compute_vertex_normal(i + 1, j + 1, k + 1)

            return ti.math.normalize(trilinear_interpolate(x_d, y_d, z_d, c000, c001, c010, c100, c011, c101, c110,
                                                               c111))

        @ti.kernel
        def dual_contouring_impl():
            for i, j, k in aux.data_wrapper.leaf_value:
                cube_index = 0
                for w in ti.static(range(8)):
                    offset = VolumeToMesh.vertex_offset[w]
                    if sdf.read_value_world(i + offset[0], j + offset[1], k + offset[2]) < isovalue:
                        cube_index |= ti.static(1 << w)

                if VolumeToMesh.edge_table[cube_index] != 0:
                    A = ti.Matrix.zero(ti.f32, 4, 4)
                    cube_voxel_coord = ti.Vector([i, j, k])
                    count = 0
                    mean_point = ti.Vector.zero(ti.f32, n=3)
                    # Calculate mean point
                    for w in ti.static(range(12)):
                        edge_byte = ti.static(1 << w)
                        if VolumeToMesh.edge_table[cube_index] & edge_byte:
                            interpolate_index0, interpolate_index1 = get_interpolate_indices(w)
                            vertex0 = cube_voxel_coord + VolumeToMesh.vertex_offset[interpolate_index0]
                            vertex1 = cube_voxel_coord + VolumeToMesh.vertex_offset[interpolate_index1]
                            mean_point += vertex_interpolate(isovalue, sdf.transform.voxel_to_coord_packed(vertex0),
                                                                        sdf.transform.voxel_to_coord_packed(vertex1),
                                                                        sdf.read_value_world(vertex0[0], vertex0[1],
                                                                                             vertex0[2]),
                                                                        sdf.read_value_world(vertex1[0], vertex1[1],
                                                                                             vertex1[2]) )
                            count += 1

                    mean_point /= count
                    count = 0
                    # Fill A matrix
                    # for w in range(12):
                    #     edge_byte = 1 << w
                    #     if VolumeToMesh.edge_table[cube_index] & edge_byte:
                    #         interpolate_index0, interpolate_index1 = get_interpolate_indices(w, 0)
                    #         vertex0 = cube_voxel_coord + VolumeToMesh.vertex_offset[interpolate_index0]
                    #         vertex1 = cube_voxel_coord + VolumeToMesh.vertex_offset[interpolate_index1]
                    #
                    #         intersection = vertex_interpolate(isovalue, sdf.transform.voxel_to_coord_packed(vertex0),
                    #                                          sdf.transform.voxel_to_coord_packed(vertex1),
                    #                                          sdf.read_value_world(vertex0[0], vertex0[1],
                    #                                                               vertex0[2]),
                    #                                          sdf.read_value_world(vertex1[0], vertex1[1],
                    #                                                               vertex1[2]))
                    #         normal = compute_normal_at(intersection)
                    #         # intersection -= mean_point
                    #         A[count, 0] = normal[0]
                    #         A[count, 1] = normal[1]
                    #         A[count, 2] = normal[2]
                    #         A[count, 3] = normal.dot(intersection)
                    #         count += 1
                    #
                    # # # QR decomposition
                    # Q, Ahat = householder_qr_decomposition(A)
                    #
                    # local_coord = solve_qef(Ahat)
                    vertex_coord = mean_point #+ local_coord

                    vertex_voxel_coord = sdf.transform.coord_to_voxel_packed(vertex_coord)
                    if vertex_voxel_coord[0] != i or vertex_voxel_coord[1] != j or vertex_voxel_coord[2] != k:
                        vertex_coord = mean_point

                    vertex_id = ti.atomic_add(num_vertices[None], 1)
                    vertices[vertex_id] = vertex_coord
                    aux.set_value_world(i, j, k, vertex_id + 1)
                else:
                    aux.set_value_world(i, j, k, 0)

        dual_contouring_impl()
        aux.prune(0)

        # Step 3: Generate dual contouring polygon
        @ti.kernel
        def dual_contouring_polygen():
            for i, j, k in aux.data_wrapper.leaf_value:
                cube_index = 0
                for w in ti.static(range(8)):
                    offset = VolumeToMesh.vertex_offset[w]
                    if sdf.read_value_world(i + offset[0], j + offset[1], k + offset[2]) < isovalue:
                        cube_index |= ti.static(1 << w)

                for w in ti.static(range(12)):
                    if ti.static(w == 2) or ti.static(w == 3) or ti.static(w == 11):
                        edge_byte = ti.static(1 << w)
                        if VolumeToMesh.edge_table[cube_index] & edge_byte:
                            ind0, ind1 = get_interpolate_indices(w)
                            cube_voxel_coord0 = ti.Vector([i, j, k])
                            cube_voxel_coord1 = cube_voxel_coord0
                            cube_voxel_coord2 = cube_voxel_coord0
                            cube_voxel_coord3 = cube_voxel_coord0

                            if ti.static(w == 2):
                                cube_voxel_coord1 += ti.Vector([0, 0, -1])
                                cube_voxel_coord2 += ti.Vector([0, -1, -1])
                                cube_voxel_coord3 += ti.Vector([0, -1, 0])
                            elif ti.static(w == 3):
                                cube_voxel_coord1 += ti.Vector([0, 0, -1])
                                cube_voxel_coord2 += ti.Vector([-1, 0, -1])
                                cube_voxel_coord3 += ti.Vector([-1, 0, 0])
                            else:
                                cube_voxel_coord1 += ti.Vector([0, -1, 0])
                                cube_voxel_coord2 += ti.Vector([-1, -1, 0])
                                cube_voxel_coord3 += ti.Vector([-1, 0, 0])

                            vi0 = int(aux.read_value_world(cube_voxel_coord0[0], cube_voxel_coord0[1], cube_voxel_coord0[2]))
                            vi1 = int(aux.read_value_world(cube_voxel_coord1[0], cube_voxel_coord1[1], cube_voxel_coord1[2]))
                            vi2 = int(aux.read_value_world(cube_voxel_coord2[0], cube_voxel_coord2[1], cube_voxel_coord2[2]))
                            vi3 = int(aux.read_value_world(cube_voxel_coord3[0], cube_voxel_coord3[1], cube_voxel_coord3[2]))

                            is_z_direction = ti.static(w == 11)
                            is_not_vertex_0 = (cube_index & (1 << ind0)) != 0
                            is_revert_face = True if is_not_vertex_0 != is_z_direction else False

                            if vi0 != 0 and vi1 != 0 and vi2 != 0 and vi3 != 0:
                                vi0, vi1, vi2, vi3 = vi0 - 1, vi1 - 1, vi2 - 1, vi3 - 1

                                index = ti.atomic_add(num_indices[None], 3)
                                if is_revert_face:
                                    indices[index] = vi0
                                    indices[index + 1] = vi1
                                    indices[index + 2] = vi2
                                else:
                                    indices[index] = vi2
                                    indices[index + 1] = vi1
                                    indices[index + 2] = vi0

                                index = ti.atomic_add(num_indices[None], 3)
                                if is_revert_face:
                                    indices[index] = vi0
                                    indices[index + 1] = vi2
                                    indices[index + 2] = vi3
                                else:
                                    indices[index] = vi3
                                    indices[index + 1] = vi2
                                    indices[index + 2] = vi0

        dual_contouring_polygen()

        # Step 3: Process normals
        if normals is not None:
            @ti.func
            def compute_face_normal(vert0: ti.template(), vert1: ti.template(), vert2: ti.template()):
                p = vert2 - vert0
                q = vert2 - vert1
                return ti.math.normalize(q.cross(p))

            @ti.kernel
            def process_normal():
                for i in range(num_indices[None] // 3):
                    ind0 = indices[i * 3]
                    ind1 = indices[i * 3 + 1]
                    ind2 = indices[i * 3 + 2]

                    vert0 = vertices[ind0]
                    vert1 = vertices[ind1]
                    vert2 = vertices[ind2]
                    face_normal = compute_face_normal(vert0, vert1, vert2)
                    normal_buffer = ti.Vector([face_normal[0], face_normal[1], face_normal[2], 1])
                    normals[ind0] += normal_buffer
                    normals[ind1] += normal_buffer
                    normals[ind2] += normal_buffer

                for i in range(num_vertices[None]):
                    normals[i] /= normals[i][3]

            process_normal()




