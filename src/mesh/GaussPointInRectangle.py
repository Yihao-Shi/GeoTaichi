from src.utils.errors import SimulationStateError

import math
import numpy as np
import taichi as ti
from src.utils.linalg import inner_multiply


class GaussPointInRectangle:
    def __init__(self, gauss_point=1, dimemsion=3):
        self.ngp = []
        self.dim = int(dimemsion)
        self.determine_gauss_point(gauss_point)

    def determine_gauss_point(self, gauss_point):
        if isinstance(gauss_point, int):
            for _ in range(self.dim):
                self.ngp.append(gauss_point)
        elif isinstance(gauss_point, (list, tuple, np.ndarray)):
            if len(list(gauss_point)) != self.dim:
                raise SimulationStateError(f"The dimension of gauss point should be {self.dim}")
            self.ngp = list(gauss_point)
        else:
            raise ValueError("Input parameter /gauss_point/ should be tuple, list or np.ndarray")

    def set_gauss_point_num(self, gauss_point):
        self.ngp = gauss_point

    def get_gauss_point_number(self):
        return int(inner_multiply(self.ngp))

    def create_gauss_point_1d(self, ngp):
        gpcoords = np.zeros((ngp, 2))
        if ngp == 1:
            gpcoords[0, 0] = 2.0
            gpcoords[0, 1] = 0.0

        elif ngp == 2:
            gpcoords[0, 0] = 1.0
            gpcoords[0, 1] = -math.sqrt(1.0 / 3.0)

            gpcoords[1, 0] = 1.0
            gpcoords[1, 1] = math.sqrt(1.0 / 3.0)

        elif ngp == 3:
            gpcoords[0, 0] = 5.0 / 9.0
            gpcoords[0, 1] = -math.sqrt(3.0 / 5.0)

            gpcoords[1, 0] = 8.0 / 9.0
            gpcoords[1, 1] = 0.0

            gpcoords[2, 0] = 5.0 / 9.0
            gpcoords[2, 1] = math.sqrt(3.0 / 5.0)

        elif ngp == 4:
            t = math.sqrt(4.8)
            w = 1.0 / 3.0 / t

            gpcoords[0, 1] = -math.sqrt((3.0 + t) / 7.0)
            gpcoords[1, 1] = -math.sqrt((3.0 - t) / 7.0)
            gpcoords[2, 1] = math.sqrt((3.0 - t) / 7.0)
            gpcoords[3, 1] = math.sqrt((3.0 + t) / 7.0)

            gpcoords[0, 0] = 0.5 - w
            gpcoords[1, 0] = 0.5 + w
            gpcoords[2, 0] = 0.5 + w
            gpcoords[3, 0] = 0.5 - w

        elif ngp == 5:
            t = math.sqrt(1120.0)
            gpcoords[0, 1] = -math.sqrt((70.0 + t) / 126.0)
            gpcoords[1, 1] = -math.sqrt((70.0 - t) / 126.0)
            gpcoords[2, 1] = 0.0
            gpcoords[3, 1] = math.sqrt((70.0 - t) / 126.0)
            gpcoords[4, 1] = math.sqrt((70.0 + t) / 126.0)

            gpcoords[0, 0] = (21.0 * t + 117.6) / (t * (70.0 + t))
            gpcoords[1, 0] = (21.0 * t - 117.6) / (t * (70.0 - t))
            gpcoords[2, 0] = 2.0 * (1.0 - gpcoords[0, 0] - gpcoords[1, 0])
            gpcoords[3, 0] = (21.0 * t - 117.6) / (t * (70.0 - t))
            gpcoords[4, 0] = (21.0 * t + 117.6) / (t * (70.0 + t))

        elif ngp == 6:
            gpcoords[0, 1] = -0.932469514203152
            gpcoords[1, 1] = -0.661209386466265
            gpcoords[2, 1] = -0.238619186003152
            gpcoords[3, 1] = 0.238619186003152
            gpcoords[4, 1] = 0.661209386466265
            gpcoords[5, 1] = 0.932469514203152

            gpcoords[0, 0] = 0.171324492379170
            gpcoords[1, 0] = 0.360761573048139
            gpcoords[2, 0] = 0.467913934572691
            gpcoords[3, 0] = 0.467913934572691
            gpcoords[4, 0] = 0.360761573048139
            gpcoords[5, 0] = 0.171324492379170

        elif ngp == 7:
            gpcoords[0, 1] = -0.949107912342759
            gpcoords[1, 1] = -0.741531185599394
            gpcoords[2, 1] = -0.405845151377397
            gpcoords[3, 1] = 0.000000000000000
            gpcoords[4, 1] = 0.405845151377397
            gpcoords[5, 1] = 0.741531185599394
            gpcoords[6, 1] = 0.949107912342759

            gpcoords[0, 0] = 0.129484966168870
            gpcoords[1, 0] = 0.279705391489277
            gpcoords[2, 0] = 0.381830050505119
            gpcoords[3, 0] = 0.417959183673469
            gpcoords[4, 0] = 0.381830050505119
            gpcoords[5, 0] = 0.279705391489277
            gpcoords[6, 0] = 0.129484966168870

        elif ngp == 8:
            gpcoords[0, 1] = -0.960289856497536
            gpcoords[1, 1] = -0.796666477413627
            gpcoords[2, 1] = -0.525532409916329
            gpcoords[3, 1] = -0.183434642495650
            gpcoords[4, 1] = 0.183434642495650
            gpcoords[5, 1] = 0.525532409916329
            gpcoords[6, 1] = 0.796666477413627
            gpcoords[7, 1] = 0.960289856497536

            gpcoords[0, 0] = 0.101228536290376
            gpcoords[1, 0] = 0.222381034453374
            gpcoords[2, 0] = 0.313706645877887
            gpcoords[3, 0] = 0.362683783378362
            gpcoords[4, 0] = 0.362683783378362
            gpcoords[5, 0] = 0.313706645877887
            gpcoords[6, 0] = 0.222381034453374
            gpcoords[7, 0] = 0.101228536290376

        elif ngp == 9:
            gpcoords[0, 1] = -0.968160239507626
            gpcoords[1, 1] = -0.836031107326636
            gpcoords[2, 1] = -0.613371432700590
            gpcoords[3, 1] = -0.324253423403809
            gpcoords[4, 1] = 0.000000000000000
            gpcoords[5, 1] = 0.324253423403809
            gpcoords[6, 1] = 0.613371432700590
            gpcoords[7, 1] = 0.836031107326636
            gpcoords[8, 1] = 0.968160239507626

            gpcoords[0, 0] = 0.081274388361574
            gpcoords[1, 0] = 0.180648160694857
            gpcoords[2, 0] = 0.260610696402935
            gpcoords[3, 0] = 0.312347077040003
            gpcoords[4, 0] = 0.330239355001260
            gpcoords[5, 0] = 0.312347077040003
            gpcoords[6, 0] = 0.260610696402935
            gpcoords[7, 0] = 0.180648160694857
            gpcoords[8, 0] = 0.081274388361574

        elif ngp == 10:
            gpcoords[0, 1] = -0.973906528517172
            gpcoords[1, 1] = -0.865063366688985
            gpcoords[2, 1] = -0.679409568299024
            gpcoords[3, 1] = -0.433395394129247
            gpcoords[4, 1] = -0.148874338981631
            gpcoords[5, 1] = 0.148874338981631
            gpcoords[6, 1] = 0.433395394129247
            gpcoords[7, 1] = 0.679409568299024
            gpcoords[8, 1] = 0.865063366688985
            gpcoords[9, 1] = 0.973906528517172

            gpcoords[0, 0] = 0.066671344308688
            gpcoords[1, 0] = 0.149451349150581
            gpcoords[2, 0] = 0.219086362515982
            gpcoords[3, 0] = 0.269266719309996
            gpcoords[4, 0] = 0.295524224714753
            gpcoords[5, 0] = 0.295524224714753
            gpcoords[6, 0] = 0.269266719309996
            gpcoords[7, 0] = 0.219086362515982
            gpcoords[8, 0] = 0.149451349150581
            gpcoords[9, 0] = 0.066671344308688
        else:
            raise ValueError("Unsuported gauss point number")

        return gpcoords

    def create_gauss_point(self, taichi_field=True):
        gauss_point_num = int(inner_multiply(self.ngp))

        if taichi_field:
            self.gpcoords = ti.Vector.field(self.dim, float)
            self.weight = ti.field(float)
            ti.root.dense(ti.i, gauss_point_num).place(self.gpcoords, self.weight)
        else:
            self.gpcoords = np.zeros((gauss_point_num, self.dim))
            self.weight = np.zeros(gauss_point_num)

        if self.dim == 1:
            self.assemble1d()
        elif self.dim == 2:
            self.assemble2d()
        elif self.dim == 3:
            self.assemble3d()

    def assemble1d(self):
        gpnum = 0
        gpcoords1d = self.create_gauss_point_1d(self.ngp[0])
        for i in range(self.ngp[0]):
            self.weight[gpnum] = gpcoords1d[i, 0]
            self.gpcoords[gpnum][0] = gpcoords1d[i, 1]
            gpnum += 1

    def assemble2d(self):
        gpnum = 0
        gpcoords1dx = self.create_gauss_point_1d(self.ngp[0])
        gpcoords1dy = self.create_gauss_point_1d(self.ngp[1])
        for j, i in ti.ndrange(self.ngp[1], self.ngp[0]):
            self.weight[gpnum] = gpcoords1dx[i, 0] * gpcoords1dy[j, 0]
            self.gpcoords[gpnum][0] = gpcoords1dx[i, 1]
            self.gpcoords[gpnum][1] = gpcoords1dy[j, 1]
            gpnum += 1

    def assemble3d(self):
        gpnum = 0
        gpcoords1dx = self.create_gauss_point_1d(self.ngp[0])
        gpcoords1dy = self.create_gauss_point_1d(self.ngp[1])
        gpcoords1dz = self.create_gauss_point_1d(self.ngp[2])
        for k, j, i in ti.ndrange(self.ngp[2], self.ngp[1], self.ngp[0]):
            self.weight[gpnum] = gpcoords1dx[i, 0] * gpcoords1dy[j, 0] * gpcoords1dz[k, 0]
            self.gpcoords[gpnum][0] = gpcoords1dx[i, 1]
            self.gpcoords[gpnum][1] = gpcoords1dy[j, 1]
            self.gpcoords[gpnum][2] = gpcoords1dz[k, 1]
            gpnum += 1

    def update(self):
        self.create_gauss_point()

    def get_ith_weight(self, i):
        return self.weight[i]

    def get_ith_coord(self, i):
        return tuple(self.gpcoords[i])
