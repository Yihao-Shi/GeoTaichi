import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from math import pi

import nurbs_interpolation


def linspace(start, stop, num, decimals=18):
    start = float(start)
    stop = float(stop)
    if abs(start - stop) <= 10e-8:
        return [start]
    num = int(num)
    if num > 1:
        div = num - 1
        delta = stop - start
        return [float(("{:." + str(decimals) + "f}").format((start + (float(x) * float(delta) / float(div)))))
                for x in range(num)]
    return [float(("{:." + str(decimals) + "f}").format(start))]

points = np.array([[-2, 0, 4], [-2, 1, 4], [-2, 2, 4], [-2, 3, 4], [-2, 4, 4], [-2, 5, 4], [-2, 6, 4], [-2, 7, 4], 
                   [0, 0, 4], [0, 1, 4], [0, 2, 4], [0, 3, 4], [0, 4, 4], [0, 5, 4], [0, 6, 4], [0, 7, 4],
                   [2, 0, 4], [2, 1, 6], [2, 2, 4], [2, 3, 2], [2, 4, 0], [2, 5, 2], [2, 6, 4], [2, 7, 4],
                   [4, 0, 4], [4, 1, 4], [4, 2, 4], [4, 3, 4], [4, 4, 4], [4, 5, 4], [4, 6, 4], [4, 7, 4],
                   [6, 0, 4], [6, 1, 4], [6, 2, 4], [6, 3, 4], [6, 4, 4], [6, 5, 4], [6, 6, 4], [6, 7, 4]])

p = 2
q = 2
numCtrlPts_u = 8
numCtrlPts_v = 5
assert numCtrlPts_u*numCtrlPts_v==points.shape[0]

num_segments_u = numCtrlPts_u - (p + 1)
num_segments_v = numCtrlPts_v - (q + 1)

'''knot_u = [0.0 for _ in range(0, p)]
knot_u += linspace(0.0, 1.0, num_segments_u + 2)
knot_u += [1.0 for _ in range(0, p)]'''
knot_u = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6])

'''knot_v = [0.0 for _ in range(0, q)]
knot_v += linspace(0.0, 1.0, num_segments_v + 2)
knot_v += [1.0 for _ in range(0, q)]'''
knot_v = np.array([0, 0, 0, 1, 2, 3, 3, 3])

xi = np.arange(0., np.max(knot_u)+0.01, 0.1)
eta = np.arange(0., np.max(knot_v)+0.01, 0.1)
numPts_u = xi.shape[0]
numPts_v = eta.shape[0]

weights = np.array([1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1])

interpolatedPoints = np.zeros((xi.shape[0]*eta.shape[0], 3))
a, b = 0, 0
for c in range(xi.shape[0]*eta.shape[0]):
    i = c % xi.shape[0]
    j = c // xi.shape[0]
    result1 = nurbs_interpolation.nurbs_interpolation2d(xi[i], eta[j], p, q, knot_u, knot_v, points[:,0], weights)
    result2 = nurbs_interpolation.nurbs_interpolation2d(xi[i], eta[j], p, q, knot_u, knot_v, points[:,1], weights)
    result3 = nurbs_interpolation.nurbs_interpolation2d(xi[i], eta[j], p, q, knot_u, knot_v, points[:,2], weights)
    interpolatedPoints[c, :] = [result1[0], result2[0], result3[0]]

from src.iga.generator.NurbsPlot import NurbsPlot
plot = NurbsPlot(xi.shape[0]*eta.shape[0], dims=2, evaplts=interpolatedPoints)
plot.initialize()
plot.PlotSurface(eta.shape[0], xi.shape[0])

interpolatedPointsU = np.ascontiguousarray(interpolatedPoints[:, 0].reshape((1, numPts_v, numPts_u)))
interpolatedPointsV = np.ascontiguousarray(interpolatedPoints[:, 1].reshape((1, numPts_v, numPts_u)))
interpolatedPointsW = np.ascontiguousarray(interpolatedPoints[:, 2].reshape((1, numPts_v, numPts_u)))

from third_party.pyevtk.hl import gridToVTK
#gridToVTK(f"NurbsVolume", interpolatedPointsU, interpolatedPointsV, interpolatedPointsW, cellData={}, pointData={})