from src.utils.errors import SimulationStateError

import math
import numpy as np
import taichi as ti
from src.utils.constants import PI
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefinition import vec2f, vec3f
from src.utils.Quaternion import theta_to_rotation_matrix, theta_to_rotation_matrix2d


@ti.data_oriented
class RegionFunction(object):

    @staticmethod
    def return_rand():
        return ti.random(float)

    def __init__(self, dims, types="MPM"):
        self.dims = dims
        self.set_default(types)

    def set_default(self, types):
        self.region_type = "Rectangle"
        if types == "DEM":
            self.expected_particle_volume = 0
            self.inserted_body_num = 0
            self.inserted_particle_volume = 0.0
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
        self.rotate2D = 0.0
        self.rotate = vec3f([0, 0, 0])
        self.start_point = vec3f([0, 0, 0])
        self.region_size = vec3f([0, 0, 0])
        self.local_start_point = vec3f([0, 0, 0])
        self.local_region_size = vec3f([0, 0, 0])
        self.rotate_center = vec3f([0, 0, 0])
        self.cal_volume = None
        self.function = None

    def finalize(self):
        del (
            self.name,
            self.rotate,
            self.start_point,
            self.region_size,
            self.local_start_point,
            self.local_region_size,
            self.cal_volume,
            self.function,
        )

    def dem_finalize(self):
        self.finalize()
        del (
            self.expected_particle_volume,
            self.inserted_particle_num,
            self.inserted_body_num,
            self.inserted_particle_volume,
        )
        del self.all_in, self.fill_number, self.fill_volume

    def dem_reset(self):
        self.inserted_body_num = 0
        self.inserted_particle_num = 0
        self.inserted_body_num = 0
        self.inserted_particle_volume = 0.0

    def mpm_finalize(self):
        self.finalize()
        del self.expected_particle_number, self.inserted_particle_num

    def iga_finalize(self):
        self.finalize()

    def mpm_reset(self):
        self.expected_particle_number = 0

    def check_condition(self, domain):
        if self.dims == 3:
            if self.start_point[0] < 0.0:
                return False
            elif self.start_point[1] < 0.0:
                return False
            elif self.start_point[2] < 0.0:
                return False
            elif self.start_point[0] + self.region_size[0] > (1 + 1e-7) * domain[0]:
                return False
            elif self.start_point[1] + self.region_size[1] > (1 + 1e-7) * domain[1]:
                return False
            elif self.start_point[2] + self.region_size[2] > (1 + 1e-7) * domain[2]:
                return False
            else:
                return True
        elif self.dims == 2:
            if self.start_point[0] < 0.0:
                return False
            elif self.start_point[1] < 0.0:
                return False
            elif self.start_point[0] + self.region_size[0] > (1 + 1e-7) * domain[0]:
                return False
            elif self.start_point[1] + self.region_size[1] > (1 + 1e-7) * domain[1]:
                return False
            else:
                return True

    def check_in_domain(self, domain):
        valid = self.check_condition(domain)
        if not valid:
            raise ValueError(f"The Region is out of the simulation domain: [0, 0, 0] ----> {domain}")

    def set_region(self, region_dict, override=False, printf=True):
        if override:
            self.set_default(self.region_type)
        function_hashmap = {
            "Rectangle": self.region_retangle,
            "TrianglarPrism": self.region_trianglar_prism,
            "Spheroid": self.region_spheroid,
            "Cylinder": self.region_cylinder,
            "QuarterCylinder": self.region_quarter_cylinder,
            "Rectangle2D": self.region_retangle2d,
            "Triangle2D": self.region_triangle2d,
            "Cone2D": self.region_cone2d,
            "Polygon2D": self.region_polygon2d,
        }

        volume_hashmap = {
            "Rectangle": self.region_retangle_volume,
            "TrianglarPrism": self.region_trianglar_prism_volume,
            "Spheroid": self.region_spheroid_volume,
            "Cylinder": self.region_cylinder_volume,
            "QuarterCylinder": self.region_quarter_cylinder_volume,
            "Rectangle2D": self.region_retangle2_d_volume,
            "Triangle2D": self.region_triangle2_d_volume,
            "Cone2D": self.region_cone2_d_volume,
            "Polygon2D": self.region_polygon2_d_volume,
        }

        self.region_type = DictIO.get_essential(region_dict, "Type")
        if not (self.region_type in function_hashmap or self.region_type == "UserDefined"):
            raise KeyError("Region Type error")
        self.name = DictIO.get_essential(region_dict, "Name")
        self.rotate = DictIO.get_alternative(region_dict, "rotate", [0.0, 0.0, 0.0])
        self.rotate2D = DictIO.get_alternative(region_dict, "rotate2D", 0.0)
        self.all_in = 1 if DictIO.get_alternative(region_dict, "all_in", True) else 0
        self.rotate = np.asarray(self.rotate)
        if not isinstance(self.rotate2D, (int, float)):
            raise SimulationStateError("Keyword:: /rotate2D/ should be a float type!")

        if self.region_type == "UserDefined":
            self.cal_volume = ti.pyfunc(DictIO.get_alternative(region_dict, "RegionVolume", RegionFunction.return_rand))
            self.function = ti.pyfunc(DictIO.get_essential(region_dict, "RegionFunction"))
        else:
            self.cal_volume = DictIO.get_essential(volume_hashmap, self.region_type)
            self.function = DictIO.get_essential(function_hashmap, self.region_type)
            if "2D" in self.region_type and self.dims == 3:
                raise SimulationStateError("Error: The 3D model is constructed by 2D region!")
            if "3D" in self.region_type and self.dims == 2:
                raise SimulationStateError("Error: The 2D model is constructed by 3D region!")

        if "Polygon" in self.region_type:
            polygon = np.asarray(DictIO.get_essential(region_dict, "Vertices"))
            self.polygon = ti.Vector.field(self.dims, dtype=ti.f64, shape=polygon.shape[0])
            self.polygon.from_numpy(polygon)

        self.local_start_point = DictIO.get_essential(region_dict, "BoundingBoxPoint")
        self.local_region_size = DictIO.get_essential(region_dict, "BoundingBoxSize")
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

        self.rotate_center = DictIO.get_alternative(
            region_dict, "RotateCenter", self.local_start_point + 0.5 * self.local_region_size
        )
        del function_hashmap, volume_hashmap
        self.calcuate_actual_bounding_box()

        if printf:
            self.print_info()

    def print_info(self):
        print(" Region Information ".center(71, "-"))
        print("Region Name:", self.name)
        print("Region Type:", self.region_type)
        print("Bounding Box:", self.start_point, self.start_point + self.region_size)
        if self.dims == 3:
            print("Rotate angle:", self.rotate, "\n")
        elif self.dims == 2:
            print("Rotate angle:", self.rotate2D, "\n")

    def calcuate_actual_bounding_box(self):
        if self.dims == 3:
            vertice1 = vec3f(self.local_start_point[0], self.local_start_point[1], self.local_start_point[2])
            vertice2 = vec3f(
                self.local_start_point[0] + self.local_region_size[0],
                self.local_start_point[1],
                self.local_start_point[2],
            )
            vertice3 = vec3f(
                self.local_start_point[0],
                self.local_start_point[1] + self.local_region_size[1],
                self.local_start_point[2],
            )
            vertice4 = vec3f(
                self.local_start_point[0] + self.local_region_size[0],
                self.local_start_point[1] + self.local_region_size[1],
                self.local_start_point[2],
            )
            vertice5 = vec3f(
                self.local_start_point[0],
                self.local_start_point[1],
                self.local_start_point[2] + self.local_region_size[2],
            )
            vertice6 = vec3f(
                self.local_start_point[0] + self.local_region_size[0],
                self.local_start_point[1],
                self.local_start_point[2] + self.local_region_size[2],
            )
            vertice7 = vec3f(
                self.local_start_point[0],
                self.local_start_point[1] + self.local_region_size[1],
                self.local_start_point[2] + self.local_region_size[2],
            )
            vertice8 = vec3f(
                self.local_start_point[0] + self.local_region_size[0],
                self.local_start_point[1] + self.local_region_size[1],
                self.local_start_point[2] + self.local_region_size[2],
            )

            rotation_matrix = theta_to_rotation_matrix(self.rotate)
            vertice1 = rotation_matrix @ (vertice1 - self.rotate_center) + self.rotate_center
            vertice2 = rotation_matrix @ (vertice2 - self.rotate_center) + self.rotate_center
            vertice3 = rotation_matrix @ (vertice3 - self.rotate_center) + self.rotate_center
            vertice4 = rotation_matrix @ (vertice4 - self.rotate_center) + self.rotate_center
            vertice5 = rotation_matrix @ (vertice5 - self.rotate_center) + self.rotate_center
            vertice6 = rotation_matrix @ (vertice6 - self.rotate_center) + self.rotate_center
            vertice7 = rotation_matrix @ (vertice7 - self.rotate_center) + self.rotate_center
            vertice8 = rotation_matrix @ (vertice8 - self.rotate_center) + self.rotate_center

            self.start_point = vec3f(
                min(
                    vertice1[0],
                    vertice2[0],
                    vertice3[0],
                    vertice4[0],
                    vertice5[0],
                    vertice6[0],
                    vertice7[0],
                    vertice8[0],
                ),
                min(
                    vertice1[1],
                    vertice2[1],
                    vertice3[1],
                    vertice4[1],
                    vertice5[1],
                    vertice6[1],
                    vertice7[1],
                    vertice8[1],
                ),
                min(
                    vertice1[2],
                    vertice2[2],
                    vertice3[2],
                    vertice4[2],
                    vertice5[2],
                    vertice6[2],
                    vertice7[2],
                    vertice8[2],
                ),
            )
            end_point = vec3f(
                max(
                    vertice1[0],
                    vertice2[0],
                    vertice3[0],
                    vertice4[0],
                    vertice5[0],
                    vertice6[0],
                    vertice7[0],
                    vertice8[0],
                ),
                max(
                    vertice1[1],
                    vertice2[1],
                    vertice3[1],
                    vertice4[1],
                    vertice5[1],
                    vertice6[1],
                    vertice7[1],
                    vertice8[1],
                ),
                max(
                    vertice1[2],
                    vertice2[2],
                    vertice3[2],
                    vertice4[2],
                    vertice5[2],
                    vertice6[2],
                    vertice7[2],
                    vertice8[2],
                ),
            )
            self.region_size = end_point - self.start_point
        elif self.dims == 2:
            vertice1 = vec2f(self.local_start_point[0], self.local_start_point[1])
            vertice2 = vec2f(self.local_start_point[0] + self.local_region_size[0], self.local_start_point[1])
            vertice3 = vec2f(self.local_start_point[0], self.local_start_point[1] + self.local_region_size[1])
            vertice4 = vec2f(
                self.local_start_point[0] + self.local_region_size[0],
                self.local_start_point[1] + self.local_region_size[1],
            )

            rotation_matrix = theta_to_rotation_matrix2d(self.rotate2D)
            vertice1 = rotation_matrix @ (vertice1 - self.rotate_center) + self.rotate_center
            vertice2 = rotation_matrix @ (vertice2 - self.rotate_center) + self.rotate_center
            vertice3 = rotation_matrix @ (vertice3 - self.rotate_center) + self.rotate_center
            vertice4 = rotation_matrix @ (vertice4 - self.rotate_center) + self.rotate_center

            self.start_point = vec2f(
                min(vertice1[0], vertice2[0], vertice3[0], vertice4[0]),
                min(vertice1[1], vertice2[1], vertice3[1], vertice4[1]),
            )
            end_point = vec2f(
                max(vertice1[0], vertice2[0], vertice3[0], vertice4[0]),
                max(vertice1[1], vertice2[1], vertice3[1], vertice4[1]),
            )
            self.region_size = end_point - self.start_point

    def calculate_expected_particle_volume(self, porosity):
        region_vol = self.cal_volume()
        self.expected_particle_volume = (1.0 - porosity) * region_vol

    def estimate_expected_particle_num_by_volume(self, initial_particle_volume):
        region_vol = self.region_size[0] * self.region_size[1]
        if self.dims == 3:
            region_vol *= self.region_size[2]
        self.expected_particle_number = math.ceil(region_vol / initial_particle_volume)

    def estimate_body_volume(self, fraction):
        actual_volume = fraction * self.expected_particle_volume
        if self.expected_particle_volume - actual_volume - self.inserted_particle_volume < 0.0:
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
    def region_retangle_volume(self):
        return self.local_region_size[0] * self.local_region_size[1] * self.local_region_size[2]

    @ti.pyfunc
    def region_trianglar_prism_volume(self):
        return 0.5 * self.local_region_size[0] * self.local_region_size[1] * self.local_region_size[2]

    @ti.pyfunc
    def region_spheroid_volume(self):
        return (
            4.0
            / 3.0
            * PI
            * (0.5 * self.local_region_size[0])
            * (0.5 * self.local_region_size[1])
            * (0.5 * self.local_region_size[2])
        )

    @ti.pyfunc
    def region_cylinder_volume(self):
        return PI * (0.5 * self.local_region_size[0]) * (0.5 * self.local_region_size[1]) * self.local_region_size[2]

    @ti.pyfunc
    def region_retangle2_d_volume(self):
        return self.local_region_size[0] * self.local_region_size[1]

    @ti.pyfunc
    def region_triangle2_d_volume(self):
        return 0.5 * self.local_region_size[0] * self.local_region_size[1]

    @ti.pyfunc
    def region_cone2_d_volume(self):
        return (
            self.local_region_size[0] * self.local_region_size[1]
            - 0.8660254038 * self.local_region_size[0] * self.local_region_size[0]
        )

    @ti.pyfunc
    def region_quarter_cylinder_volume(self):
        return 0.25 * PI * (self.local_region_size[0]) * (self.local_region_size[1]) * self.local_region_size[2]

    @ti.pyfunc
    def region_polygon2_d_volume(self):
        n = int(self.polygon.shape[0])
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.polygon[i, 0] * self.polygon[j, 1] - self.polygon[j, 0] * self.polygon[i, 1]
        return 0.5 * abs(area)

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
    def region_retangle(self, new_position, new_radius=0.0):
        rotation_matrix = theta_to_rotation_matrix(vec3f(self.rotate))
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
        return (
            x0 + self.all_in * new_radius <= xpos <= x0 + l0 - self.all_in * new_radius
            and x1 + self.all_in * new_radius <= ypos <= x1 + l1 - self.all_in * new_radius
            and x2 + self.all_in * new_radius <= zpos <= x2 + l2 - self.all_in * new_radius
        )

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
    def region_trianglar_prism(self, new_position, new_radius=0.0):
        rotation_matrix = theta_to_rotation_matrix(vec3f(self.rotate))
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
        return (
            x0 + self.all_in * new_radius <= xpos <= x0 + l0 - self.all_in * new_radius
            and x1 + self.all_in * new_radius <= ypos <= x1 + l1 - self.all_in * new_radius
            and l2 * xpos + l0 * zpos - (x0 * l2 + x2 * l0 + l0 * l2) < 0.0
        )

    @ti.pyfunc
    def region_spheroid(self, new_position, new_radius=0.0):
        rotation_matrix = theta_to_rotation_matrix(vec3f(self.rotate))
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
        return ((xpos - (x0 + 0.5 * l0)) / (0.5 * l0 - 2 * self.all_in * new_radius)) ** 2 + (
            (ypos - (x1 + 0.5 * l1)) / (0.5 * l1 - 2 * self.all_in * new_radius)
        ) ** 2 + ((zpos - (x2 + 0.5 * l2)) / (0.5 * l2 - 2 * self.all_in * new_radius)) ** 2 - 1.0 < 0.0

    @ti.pyfunc
    def region_cylinder(self, new_position, new_radius=0.0):
        rotation_matrix = theta_to_rotation_matrix(vec3f(self.rotate))
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

        return ((xpos - (x0 + 0.5 * l0)) / (0.5 * l0 - 2 * self.all_in * new_radius)) ** 2 + (
            (ypos - (x1 + 0.5 * l1)) / (0.5 * l1 - 2 * self.all_in * new_radius)
        ) ** 2 < 1.0 and x2 + self.all_in * new_radius <= zpos <= x2 + l2 - self.all_in * new_radius

    @ti.pyfunc
    def region_quarter_cylinder(self, new_position, new_radius=0.0):
        rotation_matrix = theta_to_rotation_matrix(vec3f(self.rotate))
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

        return (
            ((xpos - x0) / (l0 - 2 * self.all_in * new_radius)) ** 2
            + ((ypos - x1) / (l1 - 2 * self.all_in * new_radius)) ** 2
            < 1.0
            and x2 + self.all_in * new_radius <= zpos <= x2 + l2 - self.all_in * new_radius
            and xpos >= x0
            and ypos >= x1
        )

    @ti.pyfunc
    def region_retangle2d(self, new_position, new_radius=0.0):
        rotation_matrix = theta_to_rotation_matrix2d(self.rotate2D)
        local_position = rotation_matrix.transpose() @ (new_position - self.rotate_center) + self.rotate_center
        xpos = local_position[0]
        ypos = local_position[1]
        x0 = self.local_start_point[0]
        x1 = self.local_start_point[1]
        l0 = self.local_region_size[0]
        l1 = self.local_region_size[1]
        return (
            x0 + self.all_in * new_radius <= xpos <= x0 + l0 - self.all_in * new_radius
            and x1 + self.all_in * new_radius <= ypos <= x1 + l1 - self.all_in * new_radius
        )

    @ti.pyfunc
    def region_triangle2d(self, new_position, new_radius=0.0):
        rotation_matrix = theta_to_rotation_matrix2d(self.rotate2D)
        local_position = rotation_matrix.transpose() @ (new_position - self.rotate_center) + self.rotate_center
        xpos = local_position[0]
        ypos = local_position[1]
        x0 = self.local_start_point[0]
        x1 = self.local_start_point[1]
        l0 = self.local_region_size[0]
        l1 = self.local_region_size[1]
        return (
            x0 + self.all_in * new_radius <= xpos <= x0 + l0 - self.all_in * new_radius
            and x1 + self.all_in * new_radius
            <= ypos
            <= x1
            + l1
            - self.all_in * new_radius
            + (x0 - xpos) * (l1 - self.all_in * new_radius) / (l0 - self.all_in * new_radius)
        )

    @ti.pyfunc
    def region_cone2d(self, new_position, new_radius=0.0):
        rotation_matrix = theta_to_rotation_matrix2d(self.rotate2D)
        local_position = rotation_matrix.transpose() @ (new_position - self.rotate_center) + self.rotate_center
        xpos = local_position[0]
        ypos = local_position[1]
        x0 = self.local_start_point[0]
        x1 = self.local_start_point[1]
        l0 = self.local_region_size[0]
        l1 = self.local_region_size[1]
        return (
            x0 + self.all_in * new_radius <= xpos <= x0 + l0 - self.all_in * new_radius
            and x1 + 1.732050808 * (xpos - x0) <= ypos <= x1 + l1 - self.all_in * new_radius
        )

    @ti.pyfunc
    def region_polygon2d(self, new_position, new_radius=0.0):
        rotation_matrix = theta_to_rotation_matrix2d(self.rotate2D)
        local_position = rotation_matrix.transpose() @ (new_position - self.rotate_center) + self.rotate_center
        xpos = local_position[0]
        ypos = local_position[1]

        count = 0
        for i in range(self.polygon.shape[0]):
            p1 = self.polygon[i]
            index = (i + 1) % self.polygon.shape[0]
            p2 = self.polygon[index]
            if ((p1[1] > ypos) != (p2[1] > ypos)) and (
                xpos < (p2[0] - p1[0]) * (ypos - p1[1]) / (p2[1] - p1[1]) + p1[0]
            ):
                count += 1
        return count % 2 == 1

    @ti.pyfunc
    def specified_rotate(self, position):
        rotation_matrix = theta_to_rotation_matrix(self.rotate)
        return rotation_matrix.transpose() @ (position - self.rotate_center) + self.rotate_center
