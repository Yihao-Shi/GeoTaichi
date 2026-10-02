import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, pi

import nurbs_basis_ders, nurbs_interpolation

ti.init(default_fp=ti.f64, debug = True)

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

points = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0], [0, 2, 0], [0, 1, 0]])
p = 2
numCtrlPts = points.shape[0]

num_repeat = p
num_segments = numCtrlPts - (p + 1)

'''
knot = [0.0 for _ in range(0, num_repeat)]
knot += linspace(0.0, 1.0, num_segments + 2)
knot += [1.0 for _ in range(0, num_repeat)]
'''
knot=[0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]
knot = np.array(knot)

xi = np.arange(0., np.max(knot)+0.01, 0.01)
numPts = xi.shape[0]
m = knot.shape[0] - 1

n = m - p
weights = np.ones(n)
weights[1] = 0.5 * sqrt(2)
weights[3] = 0.5 * sqrt(2)
weights[5] = 0.5 * sqrt(2)
weights[7] = 0.5 * sqrt(2)

interpolatedPoints = np.zeros((numPts, 2))
BsplineVals = np.zeros((numPts, n+1))
NURBSderivs = np.zeros((numPts, n+1))

for c in range(numPts):
    for i in range(n):
        result = nurbs_basis_ders.nurbs_basis(i, p, xi[c], knot, weights)
        BsplineVals[c, i+1] = result[0]
        NURBSderivs[c, i+1] = result[1]

plt.plot(xi, BsplineVals)
plt.show()
plt.close()

plt.plot(xi, NURBSderivs)
plt.show()
plt.close()

a, b = 0, 0
for c in range(numPts):
    result1 = nurbs_interpolation.nurbs_interpolation1d(xi[c], p, knot, points[:,0], weights)
    result2 = nurbs_interpolation.nurbs_interpolation1d(xi[c], p, knot, points[:,1], weights)
    interpolatedPoints[c, :] = [result1[0], result2[0]]

a = np.arange(0, 361, 10)
x = np.cos(a/180.*pi)+1
y = np.sin(a/180.*pi)+1

plt.figure(figsize=(6,6))
plt.plot(interpolatedPoints[:,0], interpolatedPoints[:,1])

plt.scatter(x, y)
plt.scatter(points[:,0], points[:,1], marker="*", color="red")
plt.xlim([-1, 3])
plt.ylim([-1, 3])
plt.show()
plt.close()

