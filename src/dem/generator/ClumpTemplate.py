import taichi as ti
import numpy as np

from src.utils.constants import PI, Threshold, ZEROMAT3x3, ZEROVEC3f
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec4f, vec3f
from third_party.pyevtk.hl import pointsToVTK


class ClumpTemplate(object):
    def __init__(self):
        self.name = None
        self.save_path = ''
        self.resolution = 80.
        self.ntry = 2000000
        self.volume_expect = 0.
        self.r_equiv = 0.
        self.r_bound = 1e15
        self.pebble_radius_min = 0.
        self.pebble_radius_max = 0.
        self.x_bound = vec3f([0, 0, 0])
        self.xmin = vec3f([0, 0, 0]) 
        self.xmax = vec3f([0, 0, 0])
        self.inertia = vec3f([0, 0, 0])
        self.ex_space = vec3f([0, 0, 0])
        self.ey_space = vec3f([0, 0, 0])
        self.ez_space = vec3f([0, 0, 0])

    def clump_template(self, clump_dict, calculate=True, title=True):
        print('#', "Start calculating properties of clump template ...".ljust(67))
        self.nspheres = DictIO.GetEssential(clump_dict, "NSphere")
        pebble_dict = DictIO.GetEssential(clump_dict, "Pebble")
        self.save_path = DictIO.GetAlternative(clump_dict, "SavePath", '')
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
        self.r_equiv = (3./(4.*PI) * self.volume_expect) ** (1./3.)

    def center_of_mass_1(self):
        xcm = ti.field(float)
        field_bulider_com = ti.FieldsBuilder()
        field_bulider_com.dense(ti.i, 3).place(xcm)
        com_snode_tree = field_bulider_com.finalize()

        nsuccess = _kernel_center_of_mass_1(self.ntry, self.nspheres, xcm, self.xmin, self.xmax, self.x_pebble, self.rad_pebble)
        self.volume_expect = nsuccess / self.ntry * (self.xmax[0] - self.xmin[0]) * (self.xmax[1] - self.xmin[1]) * (self.xmax[2] - self.xmin[2])
        self.cal_equivalent_radius()
        self.recorrect_boungings(xcm)
        com_snode_tree.destroy()

    def center_of_mass_2(self, grid_size):
        return_val = _kernel_center_of_mass_2(grid_size, self.nspheres, self.xmin, self.xmax, self.x_pebble, self.rad_pebble)
        self.volume_expect = return_val[3]
        self.cal_equivalent_radius()
        self.recorrect_boungings(vec3f(return_val[0], return_val[1], return_val[2]))

    def recorrect_inertia_moment(self, moi_vol):
        if abs(moi_vol[0, 1] - moi_vol[1, 0]) > Threshold: 
            raise RuntimeError("Fix particletemplate/multisphere:Error when calculating inertia_ tensor : Not enough accuracy. Boost ntry.")
        if abs(moi_vol[0, 2] - moi_vol[2, 0]) > Threshold: 
            raise RuntimeError("Fix particletemplate/multisphere:Error when calculating inertia_ tensor : Not enough accuracy. Boost ntry.")
        if abs(moi_vol[2, 1] - moi_vol[1, 2]) > Threshold: 
            raise RuntimeError("Fix particletemplate/multisphere:Error when calculating inertia_ tensor : Not enough accuracy. Boost ntry.")

        moi_vol[0, 1] = (moi_vol[0, 1] + moi_vol[1, 0]) / 2.
        moi_vol[1, 0] = moi_vol[0, 1]
        moi_vol[0, 2] = (moi_vol[0, 2] + moi_vol[2, 0]) / 2.
        moi_vol[2, 0] = moi_vol[0, 2]
        moi_vol[2, 1] = (moi_vol[2, 1] + moi_vol[1, 2]) / 2.
        moi_vol[1, 2] = moi_vol[2, 1]
        return moi_vol

    def inertia_moment_1(self):
        xcm = ZEROVEC3f
        moi_vol = _kernel_inertia_moment_1(self.ntry, self.nspheres, xcm, self.xmin, self.xmax, self.x_pebble, self.rad_pebble)
        moi_vol *= 1./self.ntry * (self.xmax[0] - self.xmin[0]) * (self.xmax[1] - self.xmin[1]) * (self.xmax[2] - self.xmin[2])
        return self.recorrect_inertia_moment(moi_vol)
    
    def inertia_moment_2(self, grid_size):
        xcm = ZEROVEC3f
        moi_vol = _kernel_inertia_moment_2(grid_size, self.nspheres, xcm, self.xmin, self.xmax, self.x_pebble, self.rad_pebble)
        return self.recorrect_inertia_moment(moi_vol)

    def eigensystem(self, moi_vol):
        evector = ti.field(float)
        #field_bulider_vector = ti.FieldsBuilder()
        ti.root.dense(ti.ij, (3, 3)).place(evector)
        #vector_snode_tree = field_bulider_vector.finalize()
        
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
        #vector_snode_tree.destroy()
        
        scale = ti.max(self.inertia[0],self.inertia[1], self.inertia[2])
        if self.inertia[0] < scale * Threshold: self.inertia[0] = 0.
        if self.inertia[1] < scale * Threshold: self.inertia[1] = 0.
        if self.inertia[2] < scale * Threshold: self.inertia[2] = 0.
        
        ez = self.ex_space.cross(self.ey_space)
        result = ez.dot(self.ez_space)
        if result < 0.:
            self.ez_space = -self.ez_space
    
    def calc_displace_xcm_x_body(self):
        dot1 = self.ex_space.dot(self.ey_space)
        dot2 = self.ey_space.dot(self.ez_space)
        dot3 = self.ez_space.dot(self.ex_space)
        flag = dot1 > Threshold or dot2 > Threshold or dot3 > Threshold
        if flag: raise RuntimeError("Insufficient accuracy: Using _kernel_pebble_cartesian_coosys_to_local_")

        bound_copy = vec3f([0, 0, 0])
        for d in ti.static(range(3)):
            bound_copy[d] = self.x_bound[d]

        _kernel_pebble_cartesian_coosys_to_local_(self.nspheres, self.x_pebble, self.ex_space, self.ey_space, self.ez_space)
        self.x_bound[0] = bound_copy[0] * self.ex_space[0] + bound_copy[1] * self.ex_space[1] + bound_copy[2] * self.ex_space[2]
        self.x_bound[1] = bound_copy[0] * self.ey_space[0] + bound_copy[1] * self.ey_space[1] + bound_copy[2] * self.ey_space[2]
        self.x_bound[2] = bound_copy[0] * self.ez_space[0] + bound_copy[1] * self.ez_space[1] + bound_copy[2] * self.ez_space[2]

    def get_radius_range(self):
        self.pebble_radius_max = _kernel_get_max_radius_(self.nspheres, self.rad_pebble)
        self.pebble_radius_min = _kernel_get_min_radius_(self.nspheres, self.rad_pebble)

    def template_visualization(self):
        posx, posy, posz = np.ascontiguousarray(self.x_pebble[:, 0]), \
                           np.ascontiguousarray(self.x_pebble[:, 1]), \
                           np.ascontiguousarray(self.x_pebble[:, 2])
        pointsToVTK(self.save_path+f'{self.name}', posx, posy, posz, data={"rad": np.ascontiguousarray(self.rad_pebble)})
    
    def print_info(self, title):
        if title:
            print(" Clump Template Information ".center(71,"-"))
        print("Template name: ",  self.name)
        print("The number of pebble: ",  self.nspheres)
        print("Volume = ",  self.volume_expect)
        print("Equivalent radius = ",  self.r_equiv)
        print("Center of mass = ",  ZEROVEC3f)
        print("Center of bounding sphere = ",  self.x_bound)
        print("Radius of bounding sphere = ",  self.r_bound)
        print("Inertia tensor = ",  self.inertia)
        print("Eigenvector towards X axis = ",  self.ex_space)
        print("Eigenvector towards Y axis = ",  self.ey_space)
        print("Eigenvector towards Z axis = ",  self.ez_space, '\n')

# ========================================================= #
#                        KERNELS                            #
# ========================================================= #
@ti.kernel
def _kernel_get_max_radius_(nspheres: int, rad_pebble: ti.types.ndarray()) -> float:
    max_radius = 0.
    ti.loop_config(serialize=True)
    for pebble in range(nspheres):
        if max_radius < rad_pebble[pebble]:
            max_radius = rad_pebble[pebble]
    return max_radius
    

@ti.kernel
def _kernel_get_min_radius_(nspheres: int, rad_pebble: ti.types.ndarray()) -> float:
    min_radius = 1e15
    ti.loop_config(serialize=True)
    for pebble in range(nspheres):
        if min_radius > rad_pebble[pebble]:
            min_radius = rad_pebble[pebble]
    return min_radius

@ti.kernel
def _kernel_bounding_box_left_limit_(nspheres: int, x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray()) -> ti.types.vector(3, float):
    xmin = ZEROVEC3f + 1e15
    ti.loop_config(serialize=True)
    for pebble in range(nspheres):
        if x_pebble[pebble, 0] - rad_pebble[pebble] < xmin[0]:
            xmin[0] = x_pebble[pebble, 0] - rad_pebble[pebble]
        if x_pebble[pebble, 1] - rad_pebble[pebble] < xmin[1]:
            xmin[1] = x_pebble[pebble, 1] - rad_pebble[pebble]
        if x_pebble[pebble, 2] - rad_pebble[pebble] < xmin[2]:
            xmin[2] = x_pebble[pebble, 2] - rad_pebble[pebble]
    return xmin


@ti.kernel
def _kernel_bounding_box_right_limit_(nspheres: int, x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray()) -> ti.types.vector(3, float):
    xmax = ZEROVEC3f - 1e15
    ti.loop_config(serialize=True)
    for pebble in range(nspheres):
        if x_pebble[pebble, 0] + rad_pebble[pebble] > xmax[0]:
            xmax[0] = x_pebble[pebble, 0] + rad_pebble[pebble]
        if x_pebble[pebble, 1] + rad_pebble[pebble] > xmax[1]:
            xmax[1] = x_pebble[pebble, 1] + rad_pebble[pebble]
        if x_pebble[pebble, 2] + rad_pebble[pebble] > xmax[2]:
            xmax[2] = x_pebble[pebble, 2] + rad_pebble[pebble]
    return xmax


@ti.kernel
def _kernel_bounding_sphere_1(nspheres: int, visit: ti.template(), x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray()) -> ti.types.vector(4, float):
    x_bound, r_bound = ZEROVEC3f, 1e15
    ti.loop_config(serialize=True)
    for _ in range(200):
        for pebble in range(nspheres):
            visit[pebble] = 0

        isphere, nvisit = -1, 0
        while isphere < 0 or visit[isphere] == 1 or isphere >= nspheres:
            isphere = int(ti.random() * nspheres)

        nvisit += 1
        visit[isphere] = 1

        x_bound_temp = x_pebble[isphere]
        rbound_temp = rad_pebble[isphere]

        while nvisit < nspheres:
            while isphere < 0 or visit[isphere] == 1 or isphere >= nspheres:
                isphere = int(ti.random() * nspheres)

            nvisit += 1
            visit[isphere] = 1

            d = x_pebble[isphere] - x_bound_temp
            dist = d.norm()

            if dist + rad_pebble[isphere] > rbound_temp:
                fact = (dist + rad_pebble[isphere] - rbound_temp) / (2. * dist)
                d *= fact
                x_bound_temp += d
                rbound_temp += d.norm()

        if rbound_temp < r_bound:
            r_bound = rbound_temp
            x_bound = x_bound_temp
    return vec4f([x_bound, r_bound])


@ti.kernel
def _kernel_bounding_sphere_2(nspheres: int, x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray()) -> ti.types.vector(3, float):
    volume = 0.
    pvol = vec3f(0, 0, 0)
    for pebble in range(nspheres):
        radius = rad_pebble[pebble]
        position = vec3f(x_pebble[pebble, 0], x_pebble[pebble, 1], x_pebble[pebble, 2])
        volume += 4./3. * PI * radius * radius * radius
        pvol += 4./3. * PI * radius * radius * radius * position
    return pvol / volume


@ti.kernel
def _kernel_bounding_radius_(nspheres: int, bounding_center: ti.types.vector(3, float), x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray()) -> float:
    bounding_radius = 0.
    for pebble in range(nspheres):
        ti.atomic_max(bounding_radius, rad_pebble[pebble] + (vec3f(x_pebble[pebble, 0], x_pebble[pebble, 1], x_pebble[pebble, 2]) - bounding_center).norm())
    return bounding_radius


@ti.kernel
def _kernel_check_bounding_sphere_(nspheres: int, x_bound: ti.types.vector(3, float), r_bound: float, x_pebble: ti.types.ndarray()):
    ti.loop_config(serialize=True)
    for pebble in range(nspheres):
        temp = x_bound - vec3f(x_pebble[pebble, 0], x_pebble[pebble, 1], x_pebble[pebble, 2])
        if temp.norm() > r_bound:
            print(f"Bounding sphere calculation for template failed, pebble{pebble} is out of range. Try number should increase.")
    

@ti.kernel
def _kernel_center_of_mass_1(ntry: int, nspheres: int, xcm: ti.template(), xmin: ti.types.vector(3, float), xmax: ti.types.vector(3, float), 
                             x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray()) -> float:
    nsuccess, x_try = 0, ZEROVEC3f
    ti.loop_config(serialize=True)
    for _ in range(ntry):
        x_try[0] = xmin[0] + (xmax[0] - xmin[0]) * ti.random()
        x_try[1] = xmin[1] + (xmax[1] - xmin[1]) * ti.random()
        x_try[2] = xmin[2] + (xmax[2] - xmin[2]) * ti.random()

        alreadyChecked = False
        for pebble in range(nspheres):
            dist_j_sqr = (x_try[0] - x_pebble[pebble, 0]) * (x_try[0] - x_pebble[pebble, 0]) \
                        + (x_try[1] - x_pebble[pebble, 1]) * (x_try[1] - x_pebble[pebble, 1]) \
                        + (x_try[2] - x_pebble[pebble, 2]) * (x_try[2] - x_pebble[pebble, 2])
            if alreadyChecked: break
            if dist_j_sqr < rad_pebble[pebble] * rad_pebble[pebble]:
                xcm[0] = (xcm[0] * nsuccess + x_try[0]) / (nsuccess + 1)
                xcm[1] = (xcm[1] * nsuccess + x_try[1]) / (nsuccess + 1)
                xcm[2] = (xcm[2] * nsuccess + x_try[2]) / (nsuccess + 1)
                nsuccess += 1
                alreadyChecked = True

    # transform into a system with center of mass=0/0/0
    for pebble in range(nspheres):
        for d in ti.static(range(3)):
            x_pebble[pebble, d] -= xcm[d]
    return nsuccess


@ti.kernel
def _kernel_center_of_mass_2(grid_size: float, nspheres: int, xmin: ti.types.vector(3, float), xmax: ti.types.vector(3, float), 
                             x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray()) -> ti.types.vector(4, float):
    gnum = ti.ceil((xmax - xmin) / grid_size, int)
    gridSum = ti.cast(gnum[0] * gnum[1] * gnum[2], int)
    volume = 0.
    volume_center = vec3f(0, 0, 0)
    for ng in range(gridSum):
        i = (ng % (gnum[0] * gnum[1])) % gnum[0]
        j = (ng % (gnum[0] * gnum[1])) // gnum[0]
        k = ng // (gnum[0] * gnum[1])
        grid_center = (vec3f(i, j, k) + 0.5) * grid_size + xmin

        is_overlap = 0
        for npebble in range(nspheres):
            if (grid_center - vec3f(x_pebble[npebble, 0], x_pebble[npebble, 1], x_pebble[npebble, 2])).norm() < rad_pebble[npebble]:
                is_overlap = 1
                break

        if is_overlap == 1:
            volume += grid_size * grid_size * grid_size
            volume_center += grid_size * grid_size * grid_size * grid_center
    volume_center /= volume

    for pebble in range(nspheres):
        for d in ti.static(range(3)):
            x_pebble[pebble, d] -= volume_center[d]
    return vec4f(volume_center[0], volume_center[1], volume_center[2], volume)


@ti.kernel
def _kernel_inertia_moment_1(ntry: int, nspheres: int, xcm: ti.types.vector(3, float), xmin: ti.types.vector(3, float), xmax: ti.types.vector(3, float), 
                            x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray()) -> ti.types.matrix(3, 3, float):
    x_try, moi_vol = ZEROVEC3f, ZEROMAT3x3
    ti.loop_config(serialize=True)
    for _ in range(ntry):
        x_try[0] = xmin[0] + (xmax[0] - xmin[0]) * ti.random()
        x_try[1] = xmin[1] + (xmax[1] - xmin[1]) * ti.random()
        x_try[2] = xmin[2] + (xmax[2] - xmin[2]) * ti.random()
        
        alreadyChecked = False
        for pebble in range(nspheres):
            if alreadyChecked: break
            dist_j_sqr = (x_try[0] - x_pebble[pebble, 0]) * (x_try[0] - x_pebble[pebble, 0]) \
                        + (x_try[1] - x_pebble[pebble, 1]) * (x_try[1] - x_pebble[pebble, 1]) \
                        + (x_try[2] - x_pebble[pebble, 2]) * (x_try[2] - x_pebble[pebble, 2])
            
            if dist_j_sqr < rad_pebble[pebble] * rad_pebble[pebble]:
                moi_vol[0, 0] +=  (x_try[1] - xcm[1]) * (x_try[1] - xcm[1]) + (x_try[2] - xcm[2]) * (x_try[2] - xcm[2])
                moi_vol[0, 1] -=  (x_try[0] - xcm[0]) * (x_try[1] - xcm[1])
                moi_vol[0, 2] -=  (x_try[0] - xcm[0]) * (x_try[2] - xcm[2])
                moi_vol[1, 0] -=  (x_try[1] - xcm[1]) * (x_try[0] - xcm[0])
                moi_vol[1, 1] +=  (x_try[0] - xcm[0]) * (x_try[0] - xcm[0]) + (x_try[2] - xcm[2]) * (x_try[2] - xcm[2])
                moi_vol[1, 2] -=  (x_try[1] - xcm[1]) * (x_try[2] - xcm[2])
                moi_vol[2, 0] -=  (x_try[2] - xcm[2]) * (x_try[0] - xcm[0])
                moi_vol[2, 1] -=  (x_try[2] - xcm[2]) * (x_try[1] - xcm[1])
                moi_vol[2, 2] +=  (x_try[0] - xcm[0]) * (x_try[0] - xcm[0]) + (x_try[1] - xcm[1]) * (x_try[1] - xcm[1])
                alreadyChecked = True
    return moi_vol


@ti.kernel
def _kernel_inertia_moment_2(grid_size: float, nspheres: int, xcm: ti.types.vector(3, float), xmin: ti.types.vector(3, float), xmax: ti.types.vector(3, float), 
                            x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray()) -> ti.types.matrix(3, 3, float):
    gnum = ti.ceil((xmax - xmin) / grid_size, int)
    gridSum = ti.cast(gnum[0] * gnum[1] * gnum[2], int)
    moi_vol = ZEROMAT3x3
    for ng in range(gridSum):
        i = (ng % (gnum[0] * gnum[1])) % gnum[0]
        j = (ng % (gnum[0] * gnum[1])) // gnum[0]
        k = ng // (gnum[0] * gnum[1])
        grid_center = (vec3f(i, j, k) + 0.5) * grid_size + xmin

        is_overlap = 0
        for npebble in range(nspheres):
            if (grid_center - vec3f(x_pebble[npebble, 0], x_pebble[npebble, 1], x_pebble[npebble, 2])).norm() < rad_pebble[npebble]:
                is_overlap = 1
                break

        if is_overlap == 1:
            moi_vol[0, 0] += grid_center[1] * grid_center[1] + grid_center[2] * grid_center[2] 
            moi_vol[1, 1] += grid_center[0] * grid_center[0] + grid_center[2] * grid_center[2] 
            moi_vol[2, 2] += grid_center[1] * grid_center[1] + grid_center[0] * grid_center[0] 
            moi_vol[0, 1] -= grid_center[0] * grid_center[1] 
            moi_vol[1, 0] -= grid_center[0] * grid_center[1] 
            moi_vol[0, 2] -= grid_center[0] * grid_center[2] 
            moi_vol[2, 0] -= grid_center[0] * grid_center[2] 
            moi_vol[1, 2] -= grid_center[1] * grid_center[2] 
            moi_vol[2, 1] -= grid_center[1] * grid_center[2] 
    return grid_size * grid_size * grid_size * moi_vol


@ti.kernel
def _kernel_jacobi_(moi: ti.types.matrix(3, 3, float), evectors: ti.template()) -> ti.types.vector(3, float):
    error = 1
    evalues, b, z = ZEROVEC3f, ZEROVEC3f, ZEROVEC3f
    matrix = ZEROMAT3x3

    for i, j in ti.static(ti.ndrange(3, 3)):
        matrix[i, j] = moi[i, j]
        moi[i, j] = 32

    for i in ti.static(range(3)):
        b[i] = matrix[i, i]
        evalues[i] = matrix[i, i]

    ti.loop_config(serialize=True)
    for iter in range(50):
        sm = 0.
        for i in range(2):
            for j in range(i + 1, 3):
                sm += ti.abs(matrix[i, j])
        if sm == 0.: 
            error = 0
            break
        
        tresh = 0.
        if iter < 4: tresh = 0.2 * sm / (3*3)

        for i in range(2):
            for j in range(i + 1, 3):
                g = 100. * ti.abs(matrix[i, j])
                if iter > 4 and abs(evalues[i]) + g == ti.abs(evalues[i]) and ti.abs(evalues[j]) + g == ti.abs(evalues[j]):
                    matrix[i, j] = 0.
                elif ti.abs(matrix[i, j]) > tresh:
                    h = evalues[j] - evalues[i]
                    t = 0.
                    if ti.abs(h) + g == ti.abs(h):
                        t = matrix[i, j] / h
                    else:
                        theta = 0.5 * h / matrix[i, j]
                        t = 1. / (ti.abs(theta) + ti.sqrt(1. + theta * theta))
                        if theta < 0.: t = -t
                    c = 1. / ti.sqrt(1. + t * t)
                    s = t * c
                    tau = s / (1. + c)
                    h = t * matrix[i, j]
                    z[i] -= h
                    z[j] += h
                    evalues[i] -= h
                    evalues[j] += h
                    matrix[i, j] = 0.
                    for k in range(i):
                        u = matrix[k, i]
                        v = matrix[k, j]
                        matrix[k, i] = u - s * (v + u * tau)
                        matrix[k, j] = v + s * (u - v * tau)
                    for k in range(i + 1, j):
                        u = matrix[i, k]
                        v = matrix[k, j]
                        matrix[i, k] = u - s * (v + u * tau)
                        matrix[k, j] = v + s * (u - v * tau)
                    for k in range(j + 1, 3):
                        u = matrix[i, k]
                        v = matrix[j, k]
                        matrix[i, k] = u - s * (v + u * tau)
                        matrix[j, k] = v + s * (u - v * tau)
                    for k in range(3):
                        u = evectors[k, i]
                        v = evectors[k, j]
                        evectors[k, i] = u - s * (v + u * tau)
                        evectors[k, j] = v + s * (u - v * tau)

        for i in range(3):
            b[i] += z[i]
            evalues[i] = b[i]
            z[i] = 0.

    if error == 1:
        print("Insufficient Jacobi rotations for rigid body")
    return evalues


@ti.kernel
def _kernel_pebble_cartesian_coosys_to_local_(nspheres: int, x_pebble: ti.types.ndarray(), 
                                              ex_space: ti.types.vector(3, float), ey_space: ti.types.vector(3, float), ez_space: ti.types.vector(3, float)):
    ti.loop_config(serialize=True)
    for pebble in range(nspheres):
        x_copy = ZEROVEC3f
        for d in ti.static(range(3)):
            x_copy[d] = x_pebble[pebble, d]

        x_pebble[pebble, 0] = x_copy[0] * ex_space[0] + x_copy[1] * ex_space[1] + x_copy[2] * ex_space[2]
        x_pebble[pebble, 1] = x_copy[0] * ey_space[0] + x_copy[1] * ey_space[1] + x_copy[2] * ey_space[2]
        x_pebble[pebble, 2] = x_copy[0] * ez_space[0] + x_copy[1] * ez_space[1] + x_copy[2] * ez_space[2]