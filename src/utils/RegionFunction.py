import math
import numpy as np
import taichi as ti

from src.utils.constants import PI
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec2f, vec3f
from src.utils.Quaternion import ThetaToRotationMatrix, ThetaToRotationMatrix2D


@ti.data_oriented
class RegionFunction(object):
    def __init__(self, dims, types="MPM"):
        self.dims = dims
        self.set_default(types)

    def set_default(self, types):
        self.region_type = "Rectangle"
        if types == "DEM":
            self.expected_particle_volume = 0
            self.inserted_body_num = 0
            self.inserted_particle_volume = 0.
            self.fill_number = False
            self.fill_volume = False
            self.all_in = 1
            self.inserted_particle_num = 0
        elif types == "MPM":
            self.expected_particle_number = 0
            self.inserted_particle_num = 0
        elif types == "IGA":
            pass

        self.name = ""
        self.rotate2D = 0.
        self.rotate = vec3f([0, 0, 0])
        self.start_point = vec3f([0, 0, 0])
        self.region_size = vec3f([0, 0, 0])
        self.local_start_point = vec3f([0, 0, 0])
        self.local_region_size = vec3f([0, 0, 0])
        self.rotate_center = vec3f([0, 0, 0])
        self.cal_volume = None
        self.function = None

    def finalize(self):
        del self.name, self.rotate, self.start_point, self.region_size, self.local_start_point, self.local_region_size, self.cal_volume, self.function
        
    def dem_finalize(self):
        self.finalize()
        del self.expected_particle_volume, self.inserted_particle_num, self.inserted_body_num, self.inserted_particle_volume
        del self.all_in, self.fill_number, self.fill_volume
        
    def dem_reset(self):
        self.inserted_body_num = 0
        self.inserted_particle_num = 0
        self.inserted_body_num = 0
        self.inserted_particle_volume = 0.

    def mpm_finalize(self):
        self.finalize()
        del self.expected_particle_number, self.inserted_particle_num

    def iga_finalize(self):
        self.finalize()

    def mpm_reset(self):
        self.expected_particle_number = 0

    def check_condition(self, domain):
        if self.dims == 3:
            if self.start_point[0] < 0.: return False
            elif self.start_point[1] < 0.: return False
            elif self.start_point[2] < 0.: return False
            elif self.start_point[0] + self.region_size[0] > (1 + 1e-7) * domain[0]: return False
            elif self.start_point[1] + self.region_size[1] > (1 + 1e-7) * domain[1]: return False
            elif self.start_point[2] + self.region_size[2] > (1 + 1e-7) * domain[2]: return False
            else: return True
        elif self.dims == 2:
            if self.start_point[0] < 0.: return False
            elif self.start_point[1] < 0.: return False
            elif self.start_point[0] + self.region_size[0] > (1 + 1e-7) * domain[0]: return False
            elif self.start_point[1] + self.region_size[1] > (1 + 1e-7) * domain[1]: return False
            else: return True

    def check_in_domain(self, domain):
        valid = self.check_condition(domain)
        if not valid:
            raise ValueError(f"The Region is out of the simulation domain: [0, 0, 0] ----> {domain}")

    def set_region(self, region_dict, override=False, printf=True):
        if override: self.set_default(self.region_type)
        function_hashmap = {
                                "Rectangle":          self.RegionRetangle,
                                "TrianglarPrism":     self.RegionTrianglarPrism,
                                "Spheroid":           self.RegionSpheroid,
                                "Cylinder":           self.RegionCylinder,
                                "QuarterCylinder":    self.RegionQuarterCylinder,
                                "Rectangle2D":        self.RegionRetangle2D,
                                "Triangle2D":         self.RegionTriangle2D,
                                "Cone2D":             self.RegionCone2D,
                                "Polygon2D":          self.RegionPolygon2D
                           }

        volume_hashmap = {
                              "Rectangle":          self.RegionRetangleVolume,
                              "TrianglarPrism":     self.RegionTrianglarPrismVolume,
                              "Spheroid":           self.RegionSpheroidVolume,
                              "Cylinder":           self.RegionCylinderVolume,
                              "QuarterCylinder":    self.RegionQuarterCylinderVolume,
                              "Rectangle2D":        self.RegionRetangle2DVolume,
                              "Triangle2D":         self.RegionTriangle2DVolume,
                              "Cone2D":             self.RegionCone2DVolume,
                              "Polygon2D":          self.RegionPolygon2DVolume
                         }

        self.region_type = DictIO.GetEssential(region_dict, "Type")
        if not (self.region_type in function_hashmap or self.region_type == "UserDefined"):
            raise KeyError("Region Type error")
        self.name = DictIO.GetEssential(region_dict, "Name")
        self.rotate = DictIO.GetAlternative(region_dict, "rotate", [0., 0., 0.])
        self.rotate2D = DictIO.GetAlternative(region_dict, "rotate2D", 0.)
        self.all_in = 1 if DictIO.GetAlternative(region_dict, "all_in", True) else 0
        self.rotate = np.asarray(self.rotate)
        if not isinstance(self.rotate2D, (int, float)):
            raise RuntimeError("Keyword:: /rotate2D/ should be a float type!")
        
        if self.region_type == "UserDefined":
            self.cal_volume = ti.pyfunc(DictIO.GetEssential(region_dict, "RegionVolume"))
            self.function = ti.pyfunc(DictIO.GetEssential(region_dict, "RegionFunction"))
        else:
            self.cal_volume = DictIO.GetEssential(volume_hashmap, self.region_type)
            self.function = DictIO.GetEssential(function_hashmap, self.region_type)
            if "2D" in self.region_type and self.dims == 3:
                raise RuntimeError("Error: The 3D model is constructed by 2D region!")
            if "3D" in self.region_type and self.dims == 2:
                raise RuntimeError("Error: The 2D model is constructed by 3D region!")
            
        if "Polygon" in self.region_type:
            self.polygon = np.array(DictIO.GetEssential(region_dict, "Vertices"))
        else:
            self.local_start_point = DictIO.GetEssential(region_dict, "BoundingBoxPoint")
            self.local_region_size = DictIO.GetEssential(region_dict, "BoundingBoxSize")
        if not isinstance(self.local_start_point, ti.lang.matrix.Vector):
            if self.dims == 3:
                self.local_start_point = vec3f(self.local_start_point)
            elif self.dims == 2:
                self.local_start_point = vec2f(self.local_start_point)
        if not isinstance(self.local_region_size, ti.lang.matrix.Vector):
            if self.dims == 3:
                self.local_region_size = vec3f(self.local_region_size)
            elif self.dims == 2:
                self.local_region_size = vec2f(self.local_region_size)
        
        self.rotate_center = DictIO.GetAlternative(region_dict, "RotateCenter", self.local_start_point + 0.5 * self.local_region_size)
        del function_hashmap, volume_hashmap
        self.calcuate_actual_bounding_box()

        if printf: self.print_info()

    def print_info(self):
        print(" Region Information ".center(71,"-"))
        print("Region Name:", self.name)
        print("Region Type:", self.region_type)
        print("Bounding Box:", self.start_point, self.region_size)
        if self.dims == 3:
            print("Rotate angle:", self.rotate, '\n')
        elif self.dims == 2:
            print("Rotate angle:", self.rotate2D, '\n')

    def calcuate_actual_bounding_box(self):
        if self.dims == 3:
            vertice1 = vec3f(self.local_start_point[0],                             self.local_start_point[1],                             self.local_start_point[2])
            vertice2 = vec3f(self.local_start_point[0] + self.local_region_size[0], self.local_start_point[1],                             self.local_start_point[2])
            vertice3 = vec3f(self.local_start_point[0],                             self.local_start_point[1] + self.local_region_size[1], self.local_start_point[2])
            vertice4 = vec3f(self.local_start_point[0] + self.local_region_size[0], self.local_start_point[1] + self.local_region_size[1], self.local_start_point[2])
            vertice5 = vec3f(self.local_start_point[0],                             self.local_start_point[1],                             self.local_start_point[2] + self.local_region_size[2])
            vertice6 = vec3f(self.local_start_point[0] + self.local_region_size[0], self.local_start_point[1],                             self.local_start_point[2] + self.local_region_size[2])
            vertice7 = vec3f(self.local_start_point[0],                             self.local_start_point[1] + self.local_region_size[1], self.local_start_point[2] + self.local_region_size[2])
            vertice8 = vec3f(self.local_start_point[0] + self.local_region_size[0], self.local_start_point[1] + self.local_region_size[1], self.local_start_point[2] + self.local_region_size[2])

            rotation_matrix = ThetaToRotationMatrix(self.rotate).transpose()
            vertice1 = rotation_matrix @ (vertice1 - self.rotate_center) + self.rotate_center
            vertice2 = rotation_matrix @ (vertice2 - self.rotate_center) + self.rotate_center
            vertice3 = rotation_matrix @ (vertice3 - self.rotate_center) + self.rotate_center
            vertice4 = rotation_matrix @ (vertice4 - self.rotate_center) + self.rotate_center
            vertice5 = rotation_matrix @ (vertice5 - self.rotate_center) + self.rotate_center
            vertice6 = rotation_matrix @ (vertice6 - self.rotate_center) + self.rotate_center
            vertice7 = rotation_matrix @ (vertice7 - self.rotate_center) + self.rotate_center
            vertice8 = rotation_matrix @ (vertice8 - self.rotate_center) + self.rotate_center

            self.start_point = vec3f(min(vertice1[0], vertice2[0], vertice3[0], vertice4[0], vertice5[0], vertice6[0], vertice7[0], vertice8[0]),
                                     min(vertice1[1], vertice2[1], vertice3[1], vertice4[1], vertice5[1], vertice6[1], vertice7[1], vertice8[1]),
                                     min(vertice1[2], vertice2[2], vertice3[2], vertice4[2], vertice5[2], vertice6[2], vertice7[2], vertice8[2]))
            end_point = vec3f(max(vertice1[0], vertice2[0], vertice3[0], vertice4[0], vertice5[0], vertice6[0], vertice7[0], vertice8[0]),
                              max(vertice1[1], vertice2[1], vertice3[1], vertice4[1], vertice5[1], vertice6[1], vertice7[1], vertice8[1]),
                              max(vertice1[2], vertice2[2], vertice3[2], vertice4[2], vertice5[2], vertice6[2], vertice7[2], vertice8[2]))
            self.region_size = end_point - self.start_point
        elif self.dims == 2:
            vertice1 = vec2f(self.local_start_point[0], self.local_start_point[1])
            vertice2 = vec2f(self.local_start_point[0] + self.local_region_size[0], self.local_start_point[1])
            vertice3 = vec2f(self.local_start_point[0], self.local_start_point[1] + self.local_region_size[1])
            vertice4 = vec2f(self.local_start_point[0] + self.local_region_size[0],
                             self.local_start_point[1] + self.local_region_size[1])

            rotation_matrix = ThetaToRotationMatrix2D(self.rotate2D).transpose()
            vertice1 = rotation_matrix @ (vertice1 - self.rotate_center) + self.rotate_center
            vertice2 = rotation_matrix @ (vertice2 - self.rotate_center) + self.rotate_center
            vertice3 = rotation_matrix @ (vertice3 - self.rotate_center) + self.rotate_center
            vertice4 = rotation_matrix @ (vertice4 - self.rotate_center) + self.rotate_center

            self.start_point = vec2f(min(vertice1[0], vertice2[0], vertice3[0], vertice4[0]),
                                     min(vertice1[1], vertice2[1], vertice3[1], vertice4[1]))
            end_point = vec2f(max(vertice1[0], vertice2[0], vertice3[0], vertice4[0]),
                                   max(vertice1[1], vertice2[1], vertice3[1], vertice4[1]))
            self.region_size = end_point - self.start_point

    def calculate_expected_particle_volume(self, porosity):
        region_vol = self.cal_volume()
        self.expected_particle_volume = (1. - porosity) * region_vol

    def estimate_expected_particle_num_by_volume(self, initial_particle_volume):
        region_vol = self.region_size[0] * self.region_size[1]
        if self.dims == 3:
            region_vol *= self.region_size[2]
        self.expected_particle_number = math.ceil(region_vol / initial_particle_volume)

    def estimate_body_volume(self, fraction):
        actual_volume = fraction * self.expected_particle_volume 
        if self.expected_particle_volume - actual_volume - self.inserted_particle_volume < 0.:
            raise ValueError(f"The particle volume in the Region {self.name} is larger than the precribed value")
        return actual_volume

    def add_inserted_body(self, reval):
        self.inserted_body_num += int(reval)

    def add_inserted_particle(self, reval):
        self.inserted_particle_num += int(reval)

    def add_inserted_particle_volume(self, reval):
        self.inserted_particle_volume += reval

    # ====================================== Region define ===================================== #
    @ti.pyfunc
    def bounding_center(self):
        return self.rotate_center
    
    @ti.pyfunc
    def RegionRetangleVolume(self):
        return self.local_region_size[0] * self.local_region_size[1] * self.local_region_size[2]
    
    @ti.pyfunc
    def RegionTrianglarPrismVolume(self):
        return 0.5 * self.local_region_size[0] * self.local_region_size[1] * self.local_region_size[2]
    
    @ti.pyfunc
    def RegionSpheroidVolume(self):
        return 4./3. * PI * (0.5*self.local_region_size[0]) * (0.5*self.local_region_size[1]) * (0.5*self.local_region_size[2])
    
    @ti.pyfunc
    def RegionCylinderVolume(self):
        return PI * (0.5*self.local_region_size[0]) * (0.5*self.local_region_size[1]) * self.local_region_size[2]
    
    @ti.pyfunc
    def RegionRetangle2DVolume(self):
        return self.local_region_size[0] * self.local_region_size[1]
    
    @ti.pyfunc
    def RegionTriangle2DVolume(self):
        return 0.5 * self.local_region_size[0] * self.local_region_size[1]

    @ti.pyfunc
    def RegionCone2DVolume(self):
        return self.local_region_size[0] * self.local_region_size[1] - 0.8660254038 * self.local_region_size[0] * self.local_region_size[0]

    @ti.pyfunc
    def RegionQuarterCylinderVolume(self):
        return 0.25 * PI * (self.local_region_size[0]) * ( self.local_region_size[1]) * self.local_region_size[2]
    
    @ti.pyfunc
    def RegionPolygon2DVolume(self):
        area = 0
        for j in range(int(0.5 * self.polygon.shape[0])):
            i = 2 * j
            area += self.polygon[i + 1, 0] * (self.polygon[i + 2, 1]-self.polygon[i, 1]) + self.polygon[i + 1, 1] * (self.polygon[i, 0] - self.polygon[i + 2, 0])
        return 0.5 * area


    # ================================================================= #
    #   Default Orientation    ^        _________________________ ep    #
    #                         / \      /                        /|      #
    #                          |      /________________________/ |      #
    #                          |      |                       |  |      #
    #                          |      |                       |  |      #
    #                          |      |                       | /       #
    #                          |   rp:|_______________________|/        #
    # ================================================================= #
    @ti.pyfunc
    def RegionRetangle(self, new_position, new_radius=0.):
        rotation_matrix = ThetaToRotationMatrix(vec3f(self.rotate))
        local_position = rotation_matrix.transpose() @ (new_position - self.rotate_center) + self.rotate_center
        xpos = local_position[0]
        ypos = local_position[1]
        zpos = local_position[2]
        x0 = self.local_start_point[0]
        x1 = self.local_start_point[1]
        x2 = self.local_start_point[2]
        l0 = self.local_region_size[0]
        l1 = self.local_region_size[1]
        l2 = self.local_region_size[2]
        return x0 + self.all_in * new_radius <= xpos <= x0 + l0 - self.all_in * new_radius and \
                x1 + self.all_in * new_radius <= ypos <= x1 + l1 - self.all_in * new_radius and \
                x2 + self.all_in * new_radius <= zpos <= x2 + l2 - self.all_in * new_radius 


    # ============================================= #
    #   Default Orientation    ^                    #
    #                         / \      / \    ep    #
    #                          |      /   \         #
    #                          |      |\   \        #
    #                          |      | \   \       #
    #                          |      |  \  /       #
    #                          |   rp:|___\/        #
    # ============================================= #
    @ti.pyfunc
    def RegionTrianglarPrism(self, new_position, new_radius=0.):
        rotation_matrix = ThetaToRotationMatrix(vec3f(self.rotate))
        local_position = rotation_matrix.transpose() @ (new_position - self.rotate_center) + self.rotate_center
        xpos = local_position[0]
        ypos = local_position[1]
        zpos = local_position[2]
        x0 = self.local_start_point[0]
        x1 = self.local_start_point[1]
        x2 = self.local_start_point[2]
        l0 = self.local_region_size[0]
        l1 = self.local_region_size[1]
        l2 = self.local_region_size[2]
        return x0 + self.all_in * new_radius <= xpos <= x0 + l0 - self.all_in * new_radius and \
                x1 + self.all_in * new_radius <= ypos <= x1 + l1 - self.all_in * new_radius and \
                l2 * xpos + l0 * zpos - (x0 * l2 + x2 * l0 + l0 * l2) < 0.

    @ti.pyfunc
    def RegionSpheroid(self, new_position, new_radius=0.):
        rotation_matrix = ThetaToRotationMatrix(vec3f(self.rotate))
        local_position = rotation_matrix.transpose() @ (new_position - self.rotate_center) + self.rotate_center
        xpos = local_position[0]
        ypos = local_position[1]
        zpos = local_position[2]
        x0 = self.local_start_point[0]
        x1 = self.local_start_point[1]
        x2 = self.local_start_point[2]
        l0 = self.local_region_size[0]
        l1 = self.local_region_size[1]
        l2 = self.local_region_size[2]
        return ((xpos - (x0 + 0.5 * l0)) / (0.5 * l0 - 2 * self.all_in * new_radius)) ** 2 \
                + ((ypos - (x1 + 0.5 * l1)) / (0.5 * l1 - 2 * self.all_in * new_radius)) ** 2 \
                + ((zpos - (x2 + 0.5 * l2)) / (0.5 * l2 - 2 * self.all_in * new_radius)) ** 2 - 1. < 0.

    @ti.pyfunc
    def RegionCylinder(self, new_position, new_radius=0.):
        rotation_matrix = ThetaToRotationMatrix(vec3f(self.rotate))
        local_position = rotation_matrix.transpose() @ (new_position - self.rotate_center) + self.rotate_center
        xpos = local_position[0]
        ypos = local_position[1]
        zpos = local_position[2]
        x0 = self.local_start_point[0]
        x1 = self.local_start_point[1]
        x2 = self.local_start_point[2]
        l0 = self.local_region_size[0]
        l1 = self.local_region_size[1]
        l2 = self.local_region_size[2]

        return ((xpos - (x0 + 0.5 * l0)) / (0.5 * l0 - 2 * self.all_in * new_radius)) ** 2 + \
               ((ypos - (x1 + 0.5 * l1)) / (0.5 * l1 - 2 * self.all_in * new_radius)) ** 2 < 1. and \
               x2 + self.all_in * new_radius <= zpos <= x2 + l2 - self.all_in * new_radius 

    @ti.pyfunc
    def RegionQuarterCylinder(self, new_position, new_radius=0.):
        rotation_matrix = ThetaToRotationMatrix(vec3f(self.rotate))
        local_position = rotation_matrix.transpose() @ (new_position - self.rotate_center) + self.rotate_center
        xpos = local_position[0]
        ypos = local_position[1]
        zpos = local_position[2]
        x0 = self.local_start_point[0]
        x1 = self.local_start_point[1]
        x2 = self.local_start_point[2]
        l0 = self.local_region_size[0]
        l1 = self.local_region_size[1]
        l2 = self.local_region_size[2]

        return ((xpos - x0) / (l0 - 2 * self.all_in * new_radius)) ** 2 + \
            ((ypos - x1) / (l1 - 2 * self.all_in * new_radius)) ** 2 < 1. and \
            x2 + self.all_in * new_radius <= zpos <= x2 + l2 - self.all_in * new_radius and \
            xpos >= x0 and ypos >=x1
    
    @ti.pyfunc
    def RegionRetangle2D(self, new_position, new_radius=0.):
        rotation_matrix = ThetaToRotationMatrix2D(self.rotate2D)
        local_position = rotation_matrix.transpose() @ (new_position - self.rotate_center) + self.rotate_center
        xpos = local_position[0]
        ypos = local_position[1]
        x0 = self.local_start_point[0]
        x1 = self.local_start_point[1]
        l0 = self.local_region_size[0]
        l1 = self.local_region_size[1]
        return x0 + self.all_in * new_radius <= xpos <= x0 + l0 - self.all_in * new_radius and \
            x1 + self.all_in * new_radius <= ypos <= x1 + l1 - self.all_in * new_radius
    
    @ti.pyfunc
    def RegionTriangle2D(self, new_position, new_radius=0.):
        rotation_matrix = ThetaToRotationMatrix2D(self.rotate2D)
        local_position = rotation_matrix.transpose() @ (new_position - self.rotate_center) + self.rotate_center
        xpos = local_position[0]
        ypos = local_position[1]
        x0 = self.local_start_point[0]
        x1 = self.local_start_point[1]
        l0 = self.local_region_size[0]
        l1 = self.local_region_size[1]
        return x0 + self.all_in * new_radius <= xpos <= x0 + l0 - self.all_in * new_radius and \
            x1 + self.all_in * new_radius <= ypos <= x1 + l1 - self.all_in * new_radius + (x0 - xpos) * (l1 - self.all_in * new_radius) / (l0 - self.all_in * new_radius)

    @ti.pyfunc
    def RegionCone2D(self, new_position, new_radius=0.):
        rotation_matrix = ThetaToRotationMatrix2D(self.rotate2D)
        local_position = rotation_matrix.transpose() @ (new_position - self.rotate_center) + self.rotate_center
        xpos = local_position[0]
        ypos = local_position[1]
        x0 = self.local_start_point[0]
        x1 = self.local_start_point[1]
        l0 = self.local_region_size[0]
        l1 = self.local_region_size[1]
        return x0 + self.all_in * new_radius <= xpos <= x0 + l0 - self.all_in * new_radius and \
            x1 + 1.732050808 * (xpos - x0) <= ypos <= x1 + l1 - self.all_in * new_radius
    
    @ti.pyfunc
    def RegionPolygon2D(self, new_position, new_radius=0.):
        rotation_matrix = ThetaToRotationMatrix2D(self.rotate)
        local_position = rotation_matrix.transpose() @ (new_position - self.rotate_center) + self.rotate_center
        xpos = local_position[0]
        ypos = local_position[1]

        count = 0
        for i in range(self.polygon.shape[0]):
            p1 = vec2f(self.polygon[i, 0], self.polygon[i, 1])
            index = (i + 1) % self.polygon.shape[0]
            p2 = vec2f(self.polygon[index, 0], self.polygon[index, 1])
            if ((p1[1] > ypos) != (p2[1] > ypos)) and (xpos < (p2[0] - p1[0]) * (ypos - p1[1]) / (p2[1] - p1[1]) + p1[0]):
                count += 1
        return count % 2 == 1
    
    @ti.pyfunc
    def SpecifiedRotate(self, position):
        rotation_matrix = ThetaToRotationMatrix(self.rotate)
        return rotation_matrix.transpose() @ (position - self.rotate_center) + self.rotate_center 
    