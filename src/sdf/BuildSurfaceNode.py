import numpy as np
import math, copy, warnings

from src.sdf.LevelSetGrid import LocalGrid
from src.utils.Root import newton
from src.utils.linalg import Sphere2Certesian, linearize, vectorize

DBL_EPSILON = 2.2204460492503131e-16

class RayTracing(object):
    def __init__(self) -> None:
        self.node_path = None
        self.normals = None
        self.surface_node_number = 0
        self.nangle = 0.
        self.phiMult = 0.
        self.lengthChar = 0.
        self.nodesTol = 50.
        self.minRad = 1e15
        self.maxRad = 0.
        self.surface_node = []
        self.grid = None

    def determine_node_path(self, ray_path):
        if ray_path == "Rectangular":
            self.node_path = self.rectangular_partition_method
        elif ray_path == "Spiral":
            self.node_path = self.spiral_point
        else:
            valid = ["Rectangular", "Spiral"]
            raise RuntimeError(f"Keyword:: /RayPath/ is invalid. Only the following {valid} are supported")
        
    def add_essentials(self, surface_node_number, grid: LocalGrid):
        self.surface_node_number = surface_node_number
        self.nangle = int(math.sqrt(surface_node_number))
        self.phiMult = math.pi * (3. - math.sqrt(5))
        self.lengthChar = np.cbrt(grid.get_cell_volume())
        self.grid = grid

    def run(self, mass_center, simple_shape, objects):
        if simple_shape == 3:
            objects.generate_sdf(estimate=True)
            for i in range(self.surface_node_number):
                theta, phi = self.node_path(i)
                self.ray_trace(mass_center, Sphere2Certesian(np.array([1, theta, phi])))
        elif simple_shape == 2:
            for i in range(self.surface_node_number):
                theta, phi = self.node_path(i)
                self.calculate_radial_distance(objects, theta, phi)
        else:
            raise RuntimeError("This SDF primitive does not support ray tracing!")

    def rectangular_partition_method(self, node):
        theta, phi = 0., 0.
        if (node - 2) / self.nangle >= self.nangle: 
            raise RuntimeError(f"Problems may come soon, please define nSurfNodes as a squared integer + 2. Otherwise you will get phi = {2 * math.pi * (node - 2) / self.nangle / self.nangle}")
        theta = ((node - 2) % self.nangle + 1) * math.pi / (self.nangle + 1.)
        phi = (node - 2) / self.nangle * 2. * math.pi / self.nangle
        return theta, phi

    def spiral_point(self, node):
        # refers to Rakhmanov, E.A., Saff, E.B., Zhou, Y.M., 1994. Minimal discrete energy on the sphere. Math. Res. Lett. 1, 647–662
        theta = np.arccos(-1. + (2. * node + 1.) / self.surface_node_number)
        phi   = node * self.phiMult
        return theta, phi

    def calculate_radial_distance(self, objects, theta, phi):
        radial_distance = objects.radial(theta, phi)
        self.surface_node.append(Sphere2Certesian(np.array([radial_distance, theta, phi])))

    def ray_trace(self, mass_center, ray):
        nGPx, nGPy, nGPz = self.grid.gnum[0], self.grid.gnum[1], self.grid.gnum[2]
        indices = self.grid.closet_corner(mass_center.reshape(-1, 3)).reshape(3)
        pointP = copy.deepcopy(mass_center)
        move = np.zeros(3, dtype=np.int32)
        kVal = np.zeros(3)
        signs, touched, touchedInCell = False, False, False
        minOtherAxes, gpDist = -1., 0.
        while True:
            linearInd = linearize(indices[0], indices[1], indices[2], nGPx, nGPy)
            point0 = self.grid.node_coord[linearInd]
            diffSign = False
            for gp in range(8):
                vectorInd = vectorize(gp, 2, 2) + indices
                Ind = linearize(vectorInd[0], vectorInd[1], vectorInd[2], nGPx, nGPy)
                gpDist = self.grid.distance_field[Ind]
                if gp == 0:
                    signs = gpDist > 0.
                if gp > 0. and (gpDist > 0.) != signs:
                    diffSign = diffSign or True
                if gpDist == 0:
                    diffSign = diffSign or True

            if diffSign:
                touchedInCell = self.ray_trace_in_cell(ray, pointP, point0, indices)
                touched = touched or touchedInCell

            if indices[0] == nGPx - 2 or indices[1] == nGPy - 2 or indices[2] == nGPz - 2 \
                or indices[0] == 0 or indices[1] == 0 or indices[2] == 0:
                if not touched: warnings.warn(f"Ray {ray} did not create a boundary node")
                break

            for axis in range(3):
                if ray[axis] > 0.:
                    kVal[axis] = (point0[axis] + self.grid.grid_space - pointP[axis]) / ray[axis]
                elif ray[axis] < 0.:
                    kVal[axis] = (point0[axis] - pointP[axis]) / ray[axis]
                else:
                    kVal[axis] = math.inf

            move = np.zeros(3, dtype=np.int32)
            for axis in range(3):
                minOtherAxes = min(kVal[(axis + 1) % 3], kVal[(axis + 2) % 3])
                if kVal[axis] - minOtherAxes < DBL_EPSILON * self.grid.grid_space:
                    move[axis] = int(np.sign(ray[axis]))
                    for j in range(1, 3):
                        if abs(kVal[axis] - kVal[(axis + j) % 3]) < DBL_EPSILON * self.grid.grid_space:
                            move[(axis + j) % 3] = int(np.sign(ray[(axis + j) % 3]))

            pointP += ray * np.min(kVal)
            indices += move
            if np.linalg.norm(move) == 0.:
                raise RuntimeError("We're stuck in the same cell !!!")

    def ray_trace_in_cell(self, ray, pointP, point0, indices):
        normNode = -1
        trialNode = np.zeros(3)
        touched = False
        space = self.grid.grid_space

        xP, yP, zP = pointP[0], pointP[1], pointP[2]
        ux, uy, uz = ray[0], ray[1], ray[2]
        x0, y0, z0 = point0[0], point0[1], point0[2]
        nGPx, nGPy = self.grid.gnum[0], self.grid.gnum[1]

        f000 = self.grid.distance_field[linearize(indices[0], indices[1], indices[2], nGPx, nGPy)]
        f111 = self.grid.distance_field[linearize(indices[0]+1, indices[1]+1, indices[2]+1, nGPx, nGPy)]
        f100 = self.grid.distance_field[linearize(indices[0]+1, indices[1], indices[2], nGPx, nGPy)]
        f010 = self.grid.distance_field[linearize(indices[0], indices[1]+1, indices[2], nGPx, nGPy)]
        f001 = self.grid.distance_field[linearize(indices[0], indices[1], indices[2]+1, nGPx, nGPy)]
        f101 = self.grid.distance_field[linearize(indices[0]+1, indices[1], indices[2]+1, nGPx, nGPy)]
        f011 = self.grid.distance_field[linearize(indices[0], indices[1]+1, indices[2]+1, nGPx, nGPy)]
        f110 = self.grid.distance_field[linearize(indices[0]+1, indices[1]+1, indices[2], nGPx, nGPy)]

        A = f111 + f100 + f010 + f001 - f101 - f110 - f011 - f000
        B = f110 - f100 - f010 + f000
        C = f011 - f010 - f001 + f000
        D = f101 - f100 - f001 + f000
        E = f100 - f000
        F = f010 - f000
        G = f001 - f000
        Ag3 = A / pow(space, 3)
        Bg2 = B / pow(space, 2)
        Cg2 = C / pow(space, 2)
        Dg2 = D / pow(space, 2)

        coeffs = np.array([float(self.grid.distance(pointP.reshape(-1, 3))) / space,
		                   Ag3 * (uz * (xP - x0) * (yP - y0) + uy * (xP - x0) * (zP - z0) + ux * (yP - y0) * (zP - z0)) + Bg2 * (ux * (yP - y0) + uy * (xP - x0)) \
		                   + Cg2 * (uy * (zP - z0) + uz * (yP - y0)) + Dg2 * (ux * (zP - z0) + uz * (xP - x0)) + E / space * ux + F / space * uy + G / space * uz,
		                  (Ag3 * ((xP - x0) * uy * uz + (yP - y0) * ux * uz + (zP - z0) * ux * uy) + Bg2 * ux * uy + Cg2 * uy * uz + Dg2 * ux * uz) * space,
		                  (Ag3 * ux * uy * uz) * pow(space, 2)])
        
        def rootFunc(k, coeff):
            return coeff[0] + coeff[1] * k + coeff[2] * k * k + coeff[3] * k * k * k
        
        def DerRootFunc(k, coeff):
            return coeff[1] + 2 * coeff[2] * k + 3 * coeff[3] * k * k
        
        def DDerRootFunc(k, coeff):
            return 2 * coeff[2] + 6 * coeff[3] * k 
        
        root = newton(func=rootFunc, x0=-np.sqrt(3), args=(coeffs,), fprime=DerRootFunc, fprime2=DDerRootFunc, tol=1e-12)
        if root >= DBL_EPSILON: trialNode = pointP + space * root * ray
        elif abs(root) <= DBL_EPSILON:
            trialNode = pointP
        else:
            return False
        
        if not self.isInBox(trialNode):
            return False
        
        if abs(float(self.grid.distance(trialNode.reshape(-1, 3)))) / self.lengthChar < self.nodesTol * DBL_EPSILON \
           and (len(self.surface_node) == 0 or (np.linalg.norm(trialNode - np.array(self.surface_node[-1])) / self.lengthChar > self.nodesTol * DBL_EPSILON)):
            self.surface_node.append(list(trialNode))
            touched = True
            normNode = np.linalg.norm(trialNode)
            if normNode > self.minRad: self.minRad = normNode
            elif normNode > self.maxRad: self.maxRad = normNode
        return touched
    
    def isInBox(self, point):
        bmin = self.grid.start_point
        bmax = self.grid.start_point + self.grid.region_size
        return point[0] > bmin[0] and point[0] < bmax[0] and point[1] > bmin[1] and point[1] < bmax[1] and point[2] > bmin[2] and point[2] < bmax[2]


