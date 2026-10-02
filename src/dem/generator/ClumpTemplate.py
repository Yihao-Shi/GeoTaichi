import taichi as ti
import numpy as np

from src.utils.constants import PI, Threshold, ZEROVEC3f
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec3f
from third_party.pyevtk.hl import pointsToVTK

from src.dem.generator.ClumpTemplateKernel import (
    _kernel_get_max_radius_,
    _kernel_get_min_radius_,
    _kernel_bounding_box_left_limit_,
    _kernel_bounding_box_right_limit_,
    _kernel_bounding_sphere_1,
    _kernel_bounding_sphere_2,
    _kernel_bounding_radius_,
    _kernel_check_bounding_sphere_,
    _kernel_center_of_mass_1,
    _kernel_center_of_mass_2,
    _kernel_inertia_moment_1,
    _kernel_inertia_moment_2,
    _kernel_jacobi_,
    _kernel_pebble_cartesian_coosys_to_local_,
)


class ClumpTemplate(object):
    def __init__(self):
        self.name = None
        self.save_path = ""
        self.resolution = 80.0
        self.ntry = 2000000
        self.volume_expect = 0.0
        self.r_equiv = 0.0
        self.r_bound = 1e15
        self.pebble_radius_min = 0.0
        self.pebble_radius_max = 0.0
        self.x_bound = vec3f([0, 0, 0])
        self.xmin = vec3f([0, 0, 0])
        self.xmax = vec3f([0, 0, 0])
        self.inertia = vec3f([0, 0, 0])
        self.ex_space = vec3f([0, 0, 0])
        self.ey_space = vec3f([0, 0, 0])
        self.ez_space = vec3f([0, 0, 0])

    def clump_template(self, clump_dict, calculate=True, title=True):
        print("#", "Start calculating properties of clump template ...".ljust(67))
        self.nspheres = DictIO.GetEssential(clump_dict, "NSphere")
        pebble_dict = DictIO.GetEssential(clump_dict, "Pebble")
        self.save_path = DictIO.GetAlternative(clump_dict, "SavePath", "")
        if self.nspheres != len(pebble_dict):
            raise ValueError("NSphere is not equal to the length of pebble dict")
        if self.nspheres < 2 or self.nspheres > 127:
            raise ValueError("This version only support clumps with 2~127 pebbles")
        self.clump_template_initialize(pebble_dict)

        if calculate:
            self.name = DictIO.GetEssential(clump_dict, "Name")
            self.ntry = int(DictIO.GetAlternative(clump_dict, "TryNumber", self.ntry))
            self.resolution = DictIO.GetAlternative(clump_dict, "Resolution", self.resolution)
            clump_property = DictIO.GetAlternative(clump_dict, "ClumpProperty", "Grid")

            self.clump_property_initialize(clump_property)
            self.template_visualization()
            self.print_info(title)

    def clump_template_initialize(self, pebble_dict):
        self.x_pebble = np.zeros((self.nspheres, 3))
        self.rad_pebble = np.zeros(self.nspheres)

        counting = 0
        for pebble in range(self.nspheres):
            self.x_pebble[counting] = DictIO.GetEssential(pebble_dict[pebble], "Position")
            self.rad_pebble[counting] = DictIO.GetEssential(pebble_dict[pebble], "Radius")
            if self.rad_pebble[counting] < 0:
                raise ValueError("Radius must be larger than zero")
            counting += 1

    def clump_property_initialize(self, clump_property):
        self.bounding_box()
        self.get_radius_range()
        if type(clump_property) is str:
            if clump_property == "MonteCarlo":
                self.bounding_sphere_1()
                self.center_of_mass_1()
                moi_vol = self.inertia_moment_1()
            elif clump_property == "Grid":
                grid_size = self.pebble_radius_min / self.resolution
                self.bounding_sphere_2()
                self.center_of_mass_2(grid_size)
                moi_vol = self.inertia_moment_2(grid_size)
        elif type(clump_property) is dict:
            self.bounding_sphere_2()
            self.volume_expect = DictIO.GetEssential(clump_property, "Volume")
            self.cal_equivalent_radius()
            moi_vol = self.recorrect_inertia_moment(DictIO.GetEssential(clump_property, "InertiaMoment"))
        self.eigensystem(moi_vol)

    def clear(self):
        del self.xmin, self.xmax, self.ex_space, self.ey_space, self.ez_space, self.ntry

    def bounding_box(self):
        self.xmin = _kernel_bounding_box_left_limit_(self.nspheres, self.x_pebble, self.rad_pebble)
        self.xmax = _kernel_bounding_box_right_limit_(self.nspheres, self.x_pebble, self.rad_pebble)

    def bounding_sphere_1(self):
        field_builder_local = ti.FieldsBuilder()
        visit = ti.field(float)
        field_builder_local.dense(ti.i, self.nspheres).place(visit)
        visit_snode_tree = field_builder_local.finalize()
        return_val = _kernel_bounding_sphere_1(self.nspheres, visit, self.x_pebble, self.rad_pebble)
        visit_snode_tree.destroy()

        self.x_bound[0] = return_val[0]
        self.x_bound[1] = return_val[1]
        self.x_bound[2] = return_val[2]
        self.r_bound = return_val[3]
        _kernel_check_bounding_sphere_(self.nspheres, self.x_bound, self.r_bound, self.x_pebble)

    def bounding_sphere_2(self):
        return_val = _kernel_bounding_sphere_2(self.nspheres, self.x_pebble, self.rad_pebble)
        self.x_bound[0] = return_val[0]
        self.x_bound[1] = return_val[1]
        self.x_bound[2] = return_val[2]
        self.r_bound = _kernel_bounding_radius_(self.nspheres, return_val, self.x_pebble, self.rad_pebble)
        _kernel_check_bounding_sphere_(self.nspheres, self.x_bound, self.r_bound, self.x_pebble)

    def recorrect_boungings(self, xcm):
        self.x_bound[0] -= xcm[0]
        self.x_bound[1] -= xcm[1]
        self.x_bound[2] -= xcm[2]
        self.xmin[0] -= xcm[0]
        self.xmin[1] -= xcm[1]
        self.xmin[2] -= xcm[2]
        self.xmax[0] -= xcm[0]
        self.xmax[1] -= xcm[1]
        self.xmax[2] -= xcm[2]

    def cal_equivalent_radius(self):
        self.r_equiv = (3.0 / (4.0 * PI) * self.volume_expect) ** (1.0 / 3.0)

    def center_of_mass_1(self):
        xcm = ti.field(float)
        field_bulider_com = ti.FieldsBuilder()
        field_bulider_com.dense(ti.i, 3).place(xcm)
        com_snode_tree = field_bulider_com.finalize()

        nsuccess = _kernel_center_of_mass_1(
            self.ntry, self.nspheres, xcm, self.xmin, self.xmax, self.x_pebble, self.rad_pebble
        )
        self.volume_expect = (
            nsuccess
            / self.ntry
            * (self.xmax[0] - self.xmin[0])
            * (self.xmax[1] - self.xmin[1])
            * (self.xmax[2] - self.xmin[2])
        )
        self.cal_equivalent_radius()
        self.recorrect_boungings(xcm)
        com_snode_tree.destroy()

    def center_of_mass_2(self, grid_size):
        return_val = _kernel_center_of_mass_2(
            grid_size, self.nspheres, self.xmin, self.xmax, self.x_pebble, self.rad_pebble
        )
        self.volume_expect = return_val[3]
        self.cal_equivalent_radius()
        self.recorrect_boungings(vec3f(return_val[0], return_val[1], return_val[2]))

    def recorrect_inertia_moment(self, moi_vol):
        if abs(moi_vol[0, 1] - moi_vol[1, 0]) > Threshold:
            raise RuntimeError(
                "Fix particletemplate/multisphere:Error when calculating inertia_ tensor : Not enough accuracy. Boost ntry."
            )
        if abs(moi_vol[0, 2] - moi_vol[2, 0]) > Threshold:
            raise RuntimeError(
                "Fix particletemplate/multisphere:Error when calculating inertia_ tensor : Not enough accuracy. Boost ntry."
            )
        if abs(moi_vol[2, 1] - moi_vol[1, 2]) > Threshold:
            raise RuntimeError(
                "Fix particletemplate/multisphere:Error when calculating inertia_ tensor : Not enough accuracy. Boost ntry."
            )

        moi_vol[0, 1] = (moi_vol[0, 1] + moi_vol[1, 0]) / 2.0
        moi_vol[1, 0] = moi_vol[0, 1]
        moi_vol[0, 2] = (moi_vol[0, 2] + moi_vol[2, 0]) / 2.0
        moi_vol[2, 0] = moi_vol[0, 2]
        moi_vol[2, 1] = (moi_vol[2, 1] + moi_vol[1, 2]) / 2.0
        moi_vol[1, 2] = moi_vol[2, 1]
        return moi_vol

    def inertia_moment_1(self):
        xcm = ZEROVEC3f
        moi_vol = _kernel_inertia_moment_1(
            self.ntry, self.nspheres, xcm, self.xmin, self.xmax, self.x_pebble, self.rad_pebble
        )
        moi_vol *= (
            1.0
            / self.ntry
            * (self.xmax[0] - self.xmin[0])
            * (self.xmax[1] - self.xmin[1])
            * (self.xmax[2] - self.xmin[2])
        )
        return self.recorrect_inertia_moment(moi_vol)

    def inertia_moment_2(self, grid_size):
        xcm = ZEROVEC3f
        moi_vol = _kernel_inertia_moment_2(
            grid_size, self.nspheres, xcm, self.xmin, self.xmax, self.x_pebble, self.rad_pebble
        )
        return self.recorrect_inertia_moment(moi_vol)

    def eigensystem(self, moi_vol):
        evector = ti.field(float)
        # field_bulider_vector = ti.FieldsBuilder()
        ti.root.dense(ti.ij, (3, 3)).place(evector)
        # vector_snode_tree = field_bulider_vector.finalize()

        self.inertia = _kernel_jacobi_(moi_vol, evector)

        self.ex_space[0] = evector[0, 0]
        self.ex_space[1] = evector[1, 0]
        self.ex_space[2] = evector[2, 0]
        self.ey_space[0] = evector[0, 1]
        self.ey_space[1] = evector[1, 1]
        self.ey_space[2] = evector[2, 1]
        self.ez_space[0] = evector[0, 2]
        self.ez_space[1] = evector[1, 2]
        self.ez_space[2] = evector[2, 2]
        # vector_snode_tree.destroy()

        scale = ti.max(self.inertia[0], self.inertia[1], self.inertia[2])
        if self.inertia[0] < scale * Threshold:
            self.inertia[0] = 0.0
        if self.inertia[1] < scale * Threshold:
            self.inertia[1] = 0.0
        if self.inertia[2] < scale * Threshold:
            self.inertia[2] = 0.0

        ez = self.ex_space.cross(self.ey_space)
        result = ez.dot(self.ez_space)
        if result < 0.0:
            self.ez_space = -self.ez_space

    def calc_displace_xcm_x_body(self):
        dot1 = self.ex_space.dot(self.ey_space)
        dot2 = self.ey_space.dot(self.ez_space)
        dot3 = self.ez_space.dot(self.ex_space)
        flag = dot1 > Threshold or dot2 > Threshold or dot3 > Threshold
        if flag:
            raise RuntimeError("Insufficient accuracy: Using _kernel_pebble_cartesian_coosys_to_local_")

        bound_copy = vec3f([0, 0, 0])
        for d in ti.static(range(3)):
            bound_copy[d] = self.x_bound[d]

        _kernel_pebble_cartesian_coosys_to_local_(
            self.nspheres, self.x_pebble, self.ex_space, self.ey_space, self.ez_space
        )
        self.x_bound[0] = (
            bound_copy[0] * self.ex_space[0] + bound_copy[1] * self.ex_space[1] + bound_copy[2] * self.ex_space[2]
        )
        self.x_bound[1] = (
            bound_copy[0] * self.ey_space[0] + bound_copy[1] * self.ey_space[1] + bound_copy[2] * self.ey_space[2]
        )
        self.x_bound[2] = (
            bound_copy[0] * self.ez_space[0] + bound_copy[1] * self.ez_space[1] + bound_copy[2] * self.ez_space[2]
        )

    def get_radius_range(self):
        self.pebble_radius_max = _kernel_get_max_radius_(self.nspheres, self.rad_pebble)
        self.pebble_radius_min = _kernel_get_min_radius_(self.nspheres, self.rad_pebble)

    def template_visualization(self):
        posx, posy, posz = (
            np.ascontiguousarray(self.x_pebble[:, 0]),
            np.ascontiguousarray(self.x_pebble[:, 1]),
            np.ascontiguousarray(self.x_pebble[:, 2]),
        )
        pointsToVTK(
            self.save_path + f"{self.name}", posx, posy, posz, data={"rad": np.ascontiguousarray(self.rad_pebble)}
        )

    def print_info(self, title):
        if title:
            print(" Clump Template Information ".center(71, "-"))
        print("Template name: ", self.name)
        print("The number of pebble: ", self.nspheres)
        print("Volume = ", self.volume_expect)
        print("Equivalent radius = ", self.r_equiv)
        print("Center of mass = ", ZEROVEC3f)
        print("Center of bounding sphere = ", self.x_bound)
        print("Radius of bounding sphere = ", self.r_bound)
        print("Inertia tensor = ", self.inertia)
        print("Eigenvector towards X axis = ", self.ex_space)
        print("Eigenvector towards Y axis = ", self.ey_space)
        print("Eigenvector towards Z axis = ", self.ez_space, "\n")


# ========================================================= #
#                        KERNELS                            #
# ========================================================= #
