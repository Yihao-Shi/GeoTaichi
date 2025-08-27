import math
import numpy as np
import taichi as ti

from src.utils.linalg import inner_multiply


class GaussPointInTriangle:
    def __init__(self, gauss_point=1, dimemsion=3):
        self.dim = int(dimemsion)
        self.set_gauss_point_num(gauss_point)

    def set_gauss_point_num(self, gauss_point):
        self.ngp = gauss_point 
        if self.dim == 2:
            if not gauss_point in [1, 3, 7, 13]:
                raise RuntimeError("The number of gauss points should 1, 3, 7 or 13 under 2D")
        elif self.dim == 3:
            if not gauss_point in [1, 4, 5]:
                raise RuntimeError("The number of gauss points should 1, 4 or 5 under 3D")

    def get_gauss_point_number(self):
        return self.ngp
    
    def create_gauss_point(self):
        if self.dim == 2: self.assemble2d()
        elif self.dim == 3: self.assemble3d()

    def assemble2d(self):
        if self.ngp == 1:
            self.gpcoords = np.zeros((1, self.dim))
            self.weight = np.zeros(1)
            self.gpcoords[0, :] = [ 0.3333333333333, 0.3333333333333 ]
            self.weight[0] = 1
        
        elif self.ngp == 3:
            self.gpcoords = np.zeros((3, self.dim))
            self.weight = np.zeros(3)
            self.gpcoords[0, :] = [ 0.1666666666667, 0.1666666666667 ]
            self.gpcoords[1, :] = [ 0.6666666666667, 0.1666666666667 ]
            self.gpcoords[2, :] = [ 0.1666666666667, 0.6666666666667 ]
            
            self.weight[0] = 0.3333333333333
            self.weight[1] = 0.3333333333333
            self.weight[2] = 0.3333333333333

        elif self.ngp == 7:
            self.gpcoords = np.zeros((7, self.dim))
            self.weight = np.zeros(7)
            self.gpcoords[0, :] = [ 0.1012865073235, 0.1012865073235 ]
            self.gpcoords[1, :] = [ 0.7974269853531, 0.1012865073235 ]
            self.gpcoords[2, :] = [ 0.1012865073235, 0.7974269853531 ]
            self.gpcoords[3, :] = [ 0.4701420641051, 0.0597158717898 ]
            self.gpcoords[4, :] = [ 0.4701420641051, 0.4701420641051 ]
            self.gpcoords[5, :] = [ 0.0597158717898, 0.4701420641051 ]
            self.gpcoords[6, :] = [ 0.3333333333333, 0.3333333333333 ]
            
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
            self.gpcoords[0 , :] = [ 0.0651301029022, 0.0651301029022 ]
            self.gpcoords[1 , :] = [ 0.8697397941956, 0.0651301029022 ]
            self.gpcoords[2 , :] = [ 0.0651301029022, 0.8697397941956 ]
            self.gpcoords[3 , :] = [ 0.3128654960049, 0.0486903154253 ]
            self.gpcoords[4 , :] = [ 0.6384441885698, 0.3128654960049 ]
            self.gpcoords[5 , :] = [ 0.0486903154253, 0.6384441885698 ]
            self.gpcoords[6 , :] = [ 0.6384441885698, 0.0486903154253 ]
            self.gpcoords[7 , :] = [ 0.3128654960049, 0.6384441885698 ]
            self.gpcoords[8 , :] = [ 0.0486903154253, 0.3128654960049 ]
            self.gpcoords[9, :] = [ 0.2603459660790, 0.2603459660790 ]
            self.gpcoords[10, :] = [ 0.4793080678419, 0.2603459660790 ]
            self.gpcoords[11, :] = [ 0.2603459660790, 0.4793080678419 ]
            self.gpcoords[12, :] = [ 0.3333333333333, 0.3333333333333 ]
            
            self.weight[0 ] = 0.0533472356088
            self.weight[1 ] = 0.0533472356088
            self.weight[2 ] = 0.0533472356088
            self.weight[3 ] = 0.0771137608903
            self.weight[4 ] = 0.0771137608903
            self.weight[5 ] = 0.0771137608903
            self.weight[6 ] = 0.0771137608903
            self.weight[7 ] = 0.0771137608903
            self.weight[8 ] = 0.0771137608903
            self.weight[9] = 0.1756152576332
            self.weight[10] = 0.1756152576332
            self.weight[11] = 0.1756152576332
            self.weight[12] =-0.1495700444677


    def assemble3d(self):
        if self.ngp == 1:
            self.gpcoords = np.zeros((1, self.dim))
            self.weight = np.zeros(1)
            self.gpcoords[0, :] = [ 0.25, 0.25, 0.25 ]
            self.weight[0] = 1
        
        elif self.ngp == 4:
            self.gpcoords = np.zeros((4, self.dim))
            self.weight = np.zeros(4)
            self.gpcoords[0, :] = [ 0.58541020,  0.13819660,  0.13819660 ]
            self.gpcoords[1, :] = [ 0.13819660,  0.58541020,  0.13819660 ]
            self.gpcoords[2, :] = [ 0.13819660,  0.13819660,  0.58541020 ]
            self.gpcoords[3, :] = [ 0.13819660,  0.13819660,  0.13819660 ]
            
            self.weight[0] = 0.25
            self.weight[1] = 0.25
            self.weight[2] = 0.25
            self.weight[3] = 0.25

        elif self.ngp == 5:
            self.gpcoords = np.zeros((4, self.dim))
            self.weight = np.zeros(4)
            self.gpcoords[0, :] = [ 0.25,  0.25,  0.25 ]
            self.gpcoords[1, :] = [ 1./2.,   1./6.,   1./6. ]
            self.gpcoords[2, :] = [ 1./6.,   1./2.,   1./6. ]
            self.gpcoords[3, :] = [ 1./6.,   1./6.,   1./2. ]
            self.gpcoords[4, :] = [ 1./6.,   1./6.,   1./6. ]
            
            self.weight[0] = -4./5.
            self.weight[1] = 9./20.
            self.weight[2] = 9./20.
            self.weight[3] = 9./20.
            self.weight[4] = 9./20.

    def update(self):
        self.create_gauss_point()

    def get_ith_weight(self, i):
        return self.gpcoords[i, 0]

    def get_ith_coord(self, i):
        return self.gpcoords[i, 1], self.gpcoords[i, 2], self.gpcoords[i, 3]


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
                raise RuntimeError(f"The dimension of gauss point should be {self.dim}")
            self.ngp = list(gauss_point)
        else:
            raise ValueError("Input parameter /gauss_point/ should be tuple, list or np.ndarray")

    def set_gauss_point_num(self, gauss_point):
        self.ngp = gauss_point 

    def get_gauss_point_number(self):
        return int(inner_multiply(self.ngp))

    def create_gauss_point_1d(self, ngp):
        gpcoords=np.zeros((ngp, 2))
        if ngp == 1:
            gpcoords[0, 0] = 2.
            gpcoords[0, 1] = 0.

        elif ngp == 2:
            gpcoords[0, 0] = 1.
            gpcoords[0, 1] = -math.sqrt(1./3.)

            gpcoords[1, 0] = 1.
            gpcoords[1, 1] = math.sqrt(1./3.)

        elif ngp == 3:
            gpcoords[0, 0] = 5./9.
            gpcoords[0, 1] = -math.sqrt(3./5.)

            gpcoords[1, 0] = 8./9.
            gpcoords[1, 1] = 0.

            gpcoords[2, 0] = 5./9.
            gpcoords[2, 1] = math.sqrt(3./5.)

        elif ngp == 4:
            t = math.sqrt(4.8)
            w = 1./3. / t

            gpcoords[0, 1] = -math.sqrt((3. + t) / 7.)
            gpcoords[1, 1] = -math.sqrt((3. - t) / 7.)
            gpcoords[2, 1] = math.sqrt((3. - t) / 7.)
            gpcoords[3, 1] = math.sqrt((3. + t) / 7.)

            gpcoords[0, 0]=0.5 - w
            gpcoords[1, 0]=0.5 + w
            gpcoords[2, 0]=0.5 + w
            gpcoords[3, 0]=0.5 - w

        elif ngp == 5:
            t = math.sqrt(1120.)
            gpcoords[0, 1] = -math.sqrt((70. + t) / 126.)
            gpcoords[1, 1] = -math.sqrt((70. - t) / 126.)
            gpcoords[2, 1] = 0.
            gpcoords[3, 1] = math.sqrt((70. - t) / 126.)
            gpcoords[4, 1] = math.sqrt((70. + t) / 126.)
     
            gpcoords[0, 0] = (21. * t + 117.6) / (t * (70. + t))
            gpcoords[1, 0] = (21. * t - 117.6) / (t * (70. - t))
            gpcoords[2, 0] = 2. * (1. - gpcoords[0, 0] - gpcoords[1, 0])
            gpcoords[3, 0] = (21. * t - 117.6) / (t * (70. - t))
            gpcoords[4, 0] = (21. * t + 117.6) / (t * (70. + t))
        
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
            self.weight =  np.zeros(gauss_point_num)

        if self.dim == 1: self.assemble1d()
        elif self.dim == 2: self.assemble2d()
        elif self.dim == 3: self.assemble3d()

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
        return self.gpcoords[i, 1], self.gpcoords[i, 2], self.gpcoords[i, 3]
