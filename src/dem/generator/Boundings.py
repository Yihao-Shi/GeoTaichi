import numpy as np
import math, warnings

EPSILON = 1e-10


class Sphere(object):
    def __init__(self, radius=0., center=np.zeros(3)) -> None:
        self._set(radius, center)

    def _set(self, radius, center):
        self.center = center
        self.radius = radius

    def __repr__(self):
        return '< Radius =' + str(self.radius) + ', Center =' + str(self.center) + ' >'
    
    def __add__(self, sphere):
        self.center += 0.5 * (sphere.center - self.center)
        self.radius = 0.5 * (self.radius + sphere.radius + np.linalg.norm(sphere.center - self.center))

    def _set_radius(self, radius):
        self.radius = radius

    def _set_center(self, center):
        self.center = center

    def _get_radius(self):
        return self.radius
    
    def _get_center(self):
        return self.center
    
    def _is_in_sphere(self, point):
        dist = point - self.center
        distance = dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]
        return (distance - self.radius * self.radius) < EPSILON


class Boundings(object):
    def __init__(self) -> None:
        self.r_bound = 0.
        self.x_bound = np.zeros(3)

        self.minBox = np.zeros(3) + math.inf
        self.maxBox = np.zeros(3) - math.inf
        self.fail = False

    def create_sphere1(self, point1):
        return 0., point1

    def create_sphere2(self, point1, point2):
        center = 0.5 * (point1 + point2)
        radius = 0.5 * np.linalg.norm(point2 - point1)
        return radius, center
    
    def create_sphere3(self, point1, point2, point3):
        # refer to https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates_from_cross-_and_dot-products
        vec12 = point1 - point2
        vec23 = point2 - point3
        vec31 = point3 - point1
        vec21 = point2 - point1
        vec32 = point3 - point2
        vec13 = point1 - point3
        
        d12 = np.linalg.norm(vec12)
        d23 = np.linalg.norm(vec23)
        d31 = np.linalg.norm(vec31)

        normalLength = np.linalg.norm(np.cross(vec12, vec23))
        normalLengthSqr = normalLength * normalLength
        r_bound = (d12 * d23 * d31) / (2. * normalLength)
        if normalLength == 0.:
            warnings.warn(f"Failed to create a sphere within 3 points, point1: {point1}, point2: {point2}, point3: {point3}")
            self.fail = True
        
        a = (d23 * d23) * np.dot(vec12, vec13) / (2. * normalLengthSqr)
        b = (d31 * d31) * np.dot(vec21, vec23) / (2. * normalLengthSqr)
        c = (d12 * d12) * np.dot(vec31, vec32) / (2. * normalLengthSqr)
        x_bound = a * point1 + b * point2 + c * point3
        return r_bound, x_bound
    
    def create_sphere4(self, point1, point2, point3, point4):
        # refer to http://www.abecedarical.com/zenosamples/zs_sphere4pts.html
        v1 = point2 - point1
        v2 = point3 - point1
        v3 = point4 - point1
        volume = 2. * np.dot(v1, np.cross(v2, v3))
        if volume == 0.:
            warnings.warn(f"Failed to create a sphere within 4 points, point1: {point1}, point2: {point2}, point3: {point3}, point4: {point4}")
            self.fail = True
        
        L1 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]
        L2 = v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]
        L3 = v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2]

        px = (point1[0] + (L1 * (v2[1] * v3[2] - v3[1] * v2[2]) - L2 * (v1[1] * v3[2] - v3[1] * v1[2]) + L3 * (v1[1] * v2[2] - v2[1] * v1[2])) / volume)
        py = (point1[1] + (-L1 * (v2[0] * v3[2] - v3[0] * v2[2]) + L2 * (v1[0] * v3[2] - v3[0] * v1[2]) - L3 * (v1[0] * v2[2] - v2[0] * v1[2])) / volume)
        pz = (point1[2] + (L1 * (v2[0] * v3[1] - v3[0] * v2[1]) - L2 * (v1[0] * v3[1] - v3[0] * v1[1]) + L3 * (v1[0] * v2[1] - v2[0] * v1[1])) / volume)
        x_bound = np.array([px, py, pz])
        r_bound = np.linalg.norm(x_bound - point1)
        return r_bound, x_bound

    def update_sphere0(self):
        r_bound = 0.
        x_bound = np.zeros(3) 
        return r_bound, x_bound
    
    def update_sphere0(self, boundary_point: list, point):
        return Sphere(), boundary_point

    def update_sphere1(self, boundary_point: list, point):
        boundary_point.append(point)
        minsphere = Sphere(*self.create_sphere2(boundary_point[0], point))

        return minsphere, boundary_point

    def update_sphere2(self, boundary_point: list, point):
        sphere = []
        index, minRad = -1, 1e15
        point0 = boundary_point[0]
        point1 = boundary_point[1]

        sphere.append(Sphere(*self.create_sphere2(point0, point)))
        if sphere[0]._is_in_sphere(point1):
            minRad = sphere[0]._get_radius()
            index = 0
        
        sphere.append(Sphere(*self.create_sphere2(point1, point)))
        if sphere[1]._get_radius() < minRad and sphere[1]._is_in_sphere(point0):
            minRad = sphere[1]._get_radius()
            index = 1
        
        minsphere = Sphere()
        if index != -1:
            minsphere = sphere[index]
            boundary_point[1 - index] = point
        else:
            minsphere = Sphere(*self.create_sphere3(point0, point1, point))
            boundary_point.append(point)
        
        return minsphere, boundary_point

    def update_sphere3(self, boundary_point: list, point):
        sphere = []
        index, minRad = -1, 1e15
        point0 = boundary_point[0]
        point1 = boundary_point[1]
        point2 = boundary_point[2]
        point3 = point

        sphere.append(Sphere(*self.create_sphere2(point0, point3)))
        if sphere[0]._is_in_sphere(point1) and sphere[0]._is_in_sphere(point2):
            minRad = sphere[0]._get_radius()
            index = 0

        sphere.append(Sphere(*self.create_sphere2(point1, point3)))
        if sphere[1]._get_radius() < minRad and sphere[1]._is_in_sphere(point0) and sphere[1]._is_in_sphere(point2):
            minRad = sphere[1]._get_radius()
            index = 1

        sphere.append(Sphere(*self.create_sphere2(point2, point3)))
        if sphere[2]._get_radius() < minRad and sphere[2]._is_in_sphere(point0) and sphere[2]._is_in_sphere(point1):
            minRad = sphere[2]._get_radius()
            index = 2

        sphere.append(Sphere(*self.create_sphere3(point0, point1, point3)))
        if sphere[3]._get_radius() < minRad and sphere[3]._is_in_sphere(point2):
            minRad = sphere[3]._get_radius()
            index = 3

        sphere.append(Sphere(*self.create_sphere3(point0, point2, point3)))
        if sphere[4]._get_radius() < minRad and sphere[4]._is_in_sphere(point1):
            minRad = sphere[4]._get_radius()
            index = 4

        sphere.append(Sphere(*self.create_sphere3(point1, point2, point3)))
        if sphere[5]._get_radius() < minRad and sphere[5]._is_in_sphere(point0):
            minRad = sphere[5]._get_radius()
            index = 5

        minsphere = Sphere()
        if index == -1:
            minsphere = Sphere(*self.create_sphere4(point0, point1, point2, point))
            boundary_point.append(point)
        else:
            minsphere = sphere[index]

            if index == 0:
                temp = boundary_point[0]
                boundary_point = []
                boundary_point.append(temp)
                boundary_point.append(point)
            elif index == 1:
                temp = boundary_point[1]
                boundary_point = []
                boundary_point.append(point)
                boundary_point.append(temp)
            elif index == 2:
                temp = boundary_point[2]
                boundary_point = []
                boundary_point.append(temp)
                boundary_point.append(point)
            elif index == 3:
                boundary_point[2] = point
            elif index == 4:
                boundary_point[1] = point
            elif index == 5:
                boundary_point[0] = point

        return minsphere, boundary_point

    def update_sphere4(self, boundary_point: list, point):
        sphere = []
        index, minRad = -1, 1e15
        point4 = point

        for i in range(4):
            point_in_sphere = False
            sphere.append(Sphere(*self.create_sphere2(boundary_point[i], point4)))
            if sphere[len(sphere) - 1]._get_radius() < minRad:
                point_in_sphere = True
                for j in range(1, 4):
                    if not sphere[len(sphere) - 1]._is_in_sphere(boundary_point[(i + j) % 4]):
                        point_in_sphere = False
                        break

            if point_in_sphere:
                minRad = sphere[len(sphere) - 1]._get_radius()
                index = len(sphere) - 1

        for i in range(4):
            for k in range(i + 1, 4):
                point_in_sphere = False
                sphere.append(Sphere(*self.create_sphere3(boundary_point[i], boundary_point[k], point4)))
                if sphere[len(sphere) - 1]._get_radius() < minRad:
                    point_in_sphere = True
                    for j in range(1, 4):
                        if not sphere[len(sphere) - 1]._is_in_sphere(boundary_point[(i + j) % 4]):
                            point_in_sphere = False
                            break

                if point_in_sphere:
                    minRad = sphere[len(sphere) - 1]._get_radius()
                    index = len(sphere) - 1

        for i in range(4):
            point_in_sphere = False
            sphere.append(Sphere(*self.create_sphere4(boundary_point[i], boundary_point[(i + 1) % 4], boundary_point[(i + 2) % 4], point4)))
            if sphere[len(sphere) - 1]._get_radius() < minRad:
                point_in_sphere = True
                for j in range(1, 4):
                    if not sphere[len(sphere) - 1]._is_in_sphere(boundary_point[(i + j) % 4]):
                        point_in_sphere = False
                        break

            if point_in_sphere:
                minRad = sphere[len(sphere) - 1]._get_radius()
                index = len(sphere) - 1

        minsphere = Sphere()
        if index == -1:
            minsphere = Sphere()
        else:
            minsphere = sphere[index]

            if index == 0:
                temp = boundary_point[0]
                boundary_point = []
                boundary_point.append(temp)
                boundary_point.append(point)
            elif index == 1:
                temp = boundary_point[1]
                boundary_point = []
                boundary_point.append(point)
                boundary_point.append(temp)
            elif index == 2:
                temp = boundary_point[2]
                boundary_point = []
                boundary_point.append(temp)
                boundary_point.append(point)
            elif index == 3:
                temp = boundary_point[3]
                boundary_point = []
                boundary_point.append(temp)
                boundary_point.append(point)
            elif index == 4:
                temp1 = boundary_point[0]
                temp2 = boundary_point[1]
                boundary_point = []
                boundary_point.append(temp1)
                boundary_point.append(temp2)
                boundary_point.append(point)
            elif index == 5:
                temp1 = boundary_point[0]
                temp2 = boundary_point[2]
                boundary_point = []
                boundary_point.append(temp1)
                boundary_point.append(temp2)
                boundary_point.append(point)
            elif index == 6:
                temp1 = boundary_point[0]
                temp2 = boundary_point[3]
                boundary_point = []
                boundary_point.append(temp1)
                boundary_point.append(temp2)
                boundary_point.append(point)
            elif index == 7:
                temp1 = boundary_point[1]
                temp2 = boundary_point[2]
                boundary_point = []
                boundary_point.append(temp1)
                boundary_point.append(temp2)
                boundary_point.append(point)
            elif index == 8:
                temp1 = boundary_point[1]
                temp2 = boundary_point[3]
                boundary_point = []
                boundary_point.append(temp1)
                boundary_point.append(temp2)
                boundary_point.append(point)
            elif index == 9:
                temp1 = boundary_point[2]
                temp2 = boundary_point[3]
                boundary_point = []
                boundary_point.append(temp1)
                boundary_point.append(temp2)
                boundary_point.append(point)
            elif index == 10:
                boundary_point[3] = point
            elif index == 11:
                temp1 = boundary_point[1]
                temp2 = boundary_point[2]
                temp3 = boundary_point[3]
                boundary_point = []
                boundary_point.append(temp1)
                boundary_point.append(temp2)
                boundary_point.append(temp3)
                boundary_point.append(point)
            elif index == 12:
                temp1 = boundary_point[2]
                temp2 = boundary_point[3]
                temp3 = boundary_point[0]
                boundary_point = []
                boundary_point.append(temp1)
                boundary_point.append(temp2)
                boundary_point.append(temp3)
                boundary_point.append(point)
            elif index == 13:
                temp1 = boundary_point[3]
                temp2 = boundary_point[0]
                temp3 = boundary_point[1]
                boundary_point = []
                boundary_point.append(temp1)
                boundary_point.append(temp2)
                boundary_point.append(temp3)
                boundary_point.append(point)

        return minsphere, boundary_point

    def update_sphere(self, boundary_point, point):
        numPoint = len(boundary_point)
        if numPoint == 0:
            return self.update_sphere0(boundary_point, point)
        elif numPoint == 1:
            return self.update_sphere1(boundary_point, point)
        elif numPoint == 2:
            return self.update_sphere2(boundary_point, point)
        elif numPoint == 3:
            return self.update_sphere3(boundary_point, point)
        elif numPoint == 4:
            return self.update_sphere4(boundary_point, point)

    def is_out(self, point, minsphere: Sphere):
        if np.linalg.norm(point - minsphere._get_center()) - minsphere._get_radius() > 1e-10:
            return True
        return False
    
    def support_set_contain(self, boundary_point, point):
        result = False

        for i in range(len(boundary_point)):
            test_point = boundary_point[i]
            dist = point - test_point
            distance = dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]

            if distance < EPSILON:
                result = True

        return result
    
    def switch(self, p1, p2):
        temp = p1
        p1 = p2
        p2 = temp
        return p1, p2
    
    def create_enclosing_sphere(self, surface_node):
        # refer to Bernd Gartner. Fast and robust smallest enclosing balls. Algorithms-ESA 99. Springer Berlin Heidelberg, pp. 325-338, 1999
        #          https://people.inf.ethz.ch/gaertner/subdir/software/miniball.html
        boundary_point = []
        minsphere = Sphere(*self.create_sphere1(surface_node[0]))

        index = 1
        boundary_point.append(surface_node[0])
        while index < surface_node.shape[0]:
            point = surface_node[index]
            if not self.support_set_contain(boundary_point, point):
                if not minsphere._is_in_sphere(point):
                    newsphere, boundary_point = self.update_sphere(boundary_point, point)
                    if newsphere._get_radius() > minsphere._get_radius():
                        minsphere = newsphere
                        index = 0
                        continue
            index += 1
        return minsphere._get_radius(), minsphere._get_center()
    
    def ritter_method(self, surface_node):
        # refer to Jack Ritter. An efficient Bounding sphere. doi: 10.1016/B978-0-08-050753-8.50063-2.
        xmin = np.where(surface_node[:, 0] == np.min(surface_node[:, 0]))[0][0]
        xmax = np.where(surface_node[:, 0] == np.max(surface_node[:, 0]))[0][0]
        ymin = np.where(surface_node[:, 1] == np.min(surface_node[:, 1]))[0][0]
        ymax = np.where(surface_node[:, 1] == np.max(surface_node[:, 1]))[0][0]
        zmin = np.where(surface_node[:, 2] == np.min(surface_node[:, 2]))[0][0]
        zmax = np.where(surface_node[:, 2] == np.max(surface_node[:, 2]))[0][0]

        xvector = surface_node[xmax] - surface_node[xmin]
        yvector = surface_node[ymax] - surface_node[ymin]
        zvector = surface_node[zmax] - surface_node[zmin]
        xlength = np.dot(xvector, xvector)
        ylength = np.dot(yvector, yvector)
        zlength = np.dot(zvector, zvector)

        length = xlength
        x_bound = 0.5 * (surface_node[xmax] + surface_node[xmin])
        if zlength > xlength and zlength > ylength:
            length = zlength
            x_bound = 0.5 * (surface_node[zmax] + surface_node[zmin])
        elif ylength > xlength and ylength > zlength:
            length = ylength
            x_bound = 0.5 * (surface_node[ymax] + surface_node[ymin])
        r_bound = 0.5 * np.sqrt(length)

        for i in range(surface_node.shape[0]):
            dvec = surface_node[i] - x_bound
            dist2 = np.dot(dvec, dvec)
            if dist2 > r_bound * r_bound:
                dist = np.sqrt(dist2)
                new_rad = 0.5 * (dist + r_bound)
                old_to_new = dist - new_rad
                r_bound = new_rad
                x_bound = (new_rad * x_bound + old_to_new * surface_node[i]) / dist
        return r_bound, x_bound
    
    def create_bounding_box(self, surface_node):
        self.minBox = np.min(surface_node, axis=0)
        self.maxBox = np.max(surface_node, axis=0)

    def create_boundings(self, surface_node, x_bound=None, r_bound=None):
        if np.isnan(surface_node).any() or np.isinf(surface_node).any():
            raise RuntimeError("Point coordinates contain NAN or Inf entries. Remove them and try again")
        
        # np.random.shuffle(surface_node)
        self.create_bounding_box(surface_node)
        r_bound1, x_bound1 = self.create_enclosing_sphere(surface_node)
        r_bound2, x_bound2 = self.ritter_method(surface_node)
        if self.fail:
            if x_bound is not None and r_bound is not None:
                x_bound1 = x_bound
                r_bound1 = r_bound
            else:
                x_bound1 = x_bound2
                r_bound1 = r_bound2
        dist1 = np.linalg.norm(surface_node - x_bound1, axis=1) - r_bound1
        dist2 = np.linalg.norm(surface_node - x_bound2, axis=1) - r_bound2
        maxlen1 = np.mean(np.abs(dist1))
        maxlen2 = np.mean(np.abs(dist2))
        if maxlen1 < maxlen2 and (dist1 < 1e-10).all():
            self.x_bound = x_bound1.copy()
            self.r_bound = r_bound1
        elif maxlen2 < maxlen1 and (dist2 < 1e-10).all():
            self.x_bound = x_bound2.copy()
            self.r_bound = r_bound2
        else:
            raise RuntimeError("Failed to generate a bounding sphere")
        
    def set_boundings(self, x_bound, r_bound, center_of_mass, extent):
        self.x_bound = x_bound.copy()
        self.r_bound = r_bound
        self.minBox = center_of_mass - 0.5 * extent
        self.maxBox = center_of_mass + 0.5 * extent
            
    def move(self, center_of_mass):
        self.x_bound -= center_of_mass
        self.minBox -= center_of_mass
        self.maxBox -= center_of_mass

