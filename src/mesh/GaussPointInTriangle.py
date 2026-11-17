from src.utils.errors import SimulationStateError

import numpy as np


class GaussPointInTriangle:
    def __init__(self, gauss_point=1, dimemsion=3):
        self.dim = int(dimemsion)
        self.set_gauss_point_num(gauss_point)

    def set_gauss_point_num(self, gauss_point):
        self.ngp = gauss_point
        if self.dim == 2:
            if not gauss_point in [1, 3, 7, 13]:
                raise SimulationStateError("The number of gauss points should 1, 3, 7 or 13 under 2D")
        elif self.dim == 3:
            if not gauss_point in [1, 4, 5]:
                raise SimulationStateError("The number of gauss points should 1, 4 or 5 under 3D")

    def get_gauss_point_number(self):
        return self.ngp

    def create_gauss_point(self):
        if self.dim == 2:
            self.assemble2d()
        elif self.dim == 3:
            self.assemble3d()

    def assemble2d(self):
        if self.ngp == 1:
            self.gpcoords = np.zeros((1, self.dim))
            self.weight = np.zeros(1)
            self.gpcoords[0, :] = [0.3333333333333, 0.3333333333333]
            self.weight[0] = 1

        elif self.ngp == 3:
            self.gpcoords = np.zeros((3, self.dim))
            self.weight = np.zeros(3)
            self.gpcoords[0, :] = [0.1666666666667, 0.1666666666667]
            self.gpcoords[1, :] = [0.6666666666667, 0.1666666666667]
            self.gpcoords[2, :] = [0.1666666666667, 0.6666666666667]

            self.weight[0] = 0.3333333333333
            self.weight[1] = 0.3333333333333
            self.weight[2] = 0.3333333333333

        elif self.ngp == 7:
            self.gpcoords = np.zeros((7, self.dim))
            self.weight = np.zeros(7)
            self.gpcoords[0, :] = [0.1012865073235, 0.1012865073235]
            self.gpcoords[1, :] = [0.7974269853531, 0.1012865073235]
            self.gpcoords[2, :] = [0.1012865073235, 0.7974269853531]
            self.gpcoords[3, :] = [0.4701420641051, 0.0597158717898]
            self.gpcoords[4, :] = [0.4701420641051, 0.4701420641051]
            self.gpcoords[5, :] = [0.0597158717898, 0.4701420641051]
            self.gpcoords[6, :] = [0.3333333333333, 0.3333333333333]

            self.weight[0] = 0.1259391805448
            self.weight[1] = 0.1259391805448
            self.weight[2] = 0.1259391805448
            self.weight[3] = 0.1323941527885
            self.weight[4] = 0.1323941527885
            self.weight[5] = 0.1323941527885
            self.weight[6] = 0.2250000000000

        elif self.ngp == 13:
            self.gpcoords = np.zeros((13, self.dim))
            self.weight = np.zeros(13)
            self.gpcoords[0, :] = [0.0651301029022, 0.0651301029022]
            self.gpcoords[1, :] = [0.8697397941956, 0.0651301029022]
            self.gpcoords[2, :] = [0.0651301029022, 0.8697397941956]
            self.gpcoords[3, :] = [0.3128654960049, 0.0486903154253]
            self.gpcoords[4, :] = [0.6384441885698, 0.3128654960049]
            self.gpcoords[5, :] = [0.0486903154253, 0.6384441885698]
            self.gpcoords[6, :] = [0.6384441885698, 0.0486903154253]
            self.gpcoords[7, :] = [0.3128654960049, 0.6384441885698]
            self.gpcoords[8, :] = [0.0486903154253, 0.3128654960049]
            self.gpcoords[9, :] = [0.2603459660790, 0.2603459660790]
            self.gpcoords[10, :] = [0.4793080678419, 0.2603459660790]
            self.gpcoords[11, :] = [0.2603459660790, 0.4793080678419]
            self.gpcoords[12, :] = [0.3333333333333, 0.3333333333333]

            self.weight[0] = 0.0533472356088
            self.weight[1] = 0.0533472356088
            self.weight[2] = 0.0533472356088
            self.weight[3] = 0.0771137608903
            self.weight[4] = 0.0771137608903
            self.weight[5] = 0.0771137608903
            self.weight[6] = 0.0771137608903
            self.weight[7] = 0.0771137608903
            self.weight[8] = 0.0771137608903
            self.weight[9] = 0.1756152574332
            self.weight[10] = 0.1756152574332
            self.weight[11] = 0.1756152574332
            self.weight[12] = -0.1495700444677

    def assemble3d(self):
        if self.ngp == 1:
            self.gpcoords = np.zeros((1, self.dim))
            self.weight = np.zeros(1)
            self.gpcoords[0, :] = [0.25, 0.25, 0.25]
            self.weight[0] = 1

        elif self.ngp == 4:
            self.gpcoords = np.zeros((4, self.dim))
            self.weight = np.zeros(4)
            self.gpcoords[0, :] = [0.58541020, 0.13819660, 0.13819660]
            self.gpcoords[1, :] = [0.13819660, 0.58541020, 0.13819660]
            self.gpcoords[2, :] = [0.13819660, 0.13819660, 0.58541020]
            self.gpcoords[3, :] = [0.13819660, 0.13819660, 0.13819660]

            self.weight[0] = 0.25
            self.weight[1] = 0.25
            self.weight[2] = 0.25
            self.weight[3] = 0.25

        elif self.ngp == 5:
            self.gpcoords = np.zeros((5, self.dim))
            self.weight = np.zeros(5)
            self.gpcoords[0, :] = [0.25, 0.25, 0.25]
            self.gpcoords[1, :] = [1.0 / 2.0, 1.0 / 6.0, 1.0 / 6.0]
            self.gpcoords[2, :] = [1.0 / 6.0, 1.0 / 2.0, 1.0 / 6.0]
            self.gpcoords[3, :] = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0]
            self.gpcoords[4, :] = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]

            self.weight[0] = -4.0 / 5.0
            self.weight[1] = 9.0 / 20.0
            self.weight[2] = 9.0 / 20.0
            self.weight[3] = 9.0 / 20.0
            self.weight[4] = 9.0 / 20.0

    def update(self):
        self.create_gauss_point()

    def get_ith_weight(self, i):
        return self.weight[i]

    def get_ith_coord(self, i):
        return tuple(self.gpcoords[i])
