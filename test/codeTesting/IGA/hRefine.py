import sys
sys.path.append('/home/eleven/work/GeoTaichi')

import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt

import nurbs_interpolation
import hrefine_nurbs

p = 2
points = np.array([[0, 1, 0], [0., 0., 0.], [1, 0, 0]])
knot = np.array([0, 0, 0, 1, 1, 1])
m = knot.shape[0] - 1
n = m - p
weights = np.ones(n)
weights[1] = 0.5*sqrt(2)

xi = np.arange(0., np.max(knot)+0.01, 0.01)
numPts = xi.shape[0]
interpolatedPoints = np.zeros((numPts, 2))

a, b = 0, 0
for c in range(numPts):
    result1 = nurbs_interpolation.nurbs_interpolation1d(xi[c], p, knot, np.ascontiguousarray(points[:,0]), weights)
    result2 = nurbs_interpolation.nurbs_interpolation1d(xi[c], p, knot, np.ascontiguousarray(points[:,1]), weights)
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

refine = 2
number = np.unique(knot).shape[0]
for i in range(refine):
    number = 2 * number - 1
number -= np.unique(knot).shape[0]
results = hrefine_nurbs.hrefine_nurbs1d(refine, p, points[:,0].shape[0], np.ascontiguousarray(points[:,0]), np.ascontiguousarray(points[:,1]), np.ascontiguousarray(points[:,2]), np.ascontiguousarray(weights), np.ascontiguousarray(knot))
new_ctrlpt = results[0:3*(points.shape[0] + number)].reshape((points.shape[0] + number, 3))
new_weight = results[3*(points.shape[0] + number):4*(points.shape[0] + number)]
new_knot = results[4*(points.shape[0] + number):]
print(points.shape[0] + number)
print(new_ctrlpt, new_knot)

interpolatedPoints = np.zeros((numPts, 2))
a, b = 0, 0
for c in range(numPts):
    result1 = nurbs_interpolation.nurbs_interpolation1d(xi[c], p, new_knot, np.ascontiguousarray(new_ctrlpt[:,0]), new_weight)
    result2 = nurbs_interpolation.nurbs_interpolation1d(xi[c], p, new_knot, np.ascontiguousarray(new_ctrlpt[:,1]), new_weight)
    interpolatedPoints[c, :] = [result1[0], result2[0]]

a = np.arange(0, 361, 10)
x = np.cos(a/180.*pi)+1
y = np.sin(a/180.*pi)+1

plt.figure(figsize=(6,6))
plt.plot(interpolatedPoints[:,0], interpolatedPoints[:,1])

plt.scatter(x, y)
plt.scatter(new_ctrlpt[:,0], new_ctrlpt[:,1], marker="*", color="red")
plt.xlim([-1, 3])
plt.ylim([-1, 3])
plt.show()
plt.close()