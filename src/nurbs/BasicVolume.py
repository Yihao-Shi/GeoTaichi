import numpy as np
import warnings, random

from src.nurbs.NurbsPrimitives import NurbsVolume
from src.utils.linalg import linspace


class PrimitiveVolume(NurbsVolume):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shape = None
        self.clamped = True

    def check_ctrlpt_num(self, degree, num_ctrlpts):
        if num_ctrlpts <= degree:
            raise RuntimeError(f"The minimum number of control points is {degree + 1}")

    def set_parameters(self, body_dict):
        raise NotImplementedError
    
    def generate_ctrlpts(self, numctrlpts):
        raise NotImplementedError
    
    def generate_weights(self, num_ctrlpts, ctrlpts=None):
        raise NotImplementedError
    
    def generate_knot_u(self, degree, num_ctrlpts):
        raise NotImplementedError

    def generate_knot_v(self, degree, num_ctrlpts):
        raise NotImplementedError

    def generate_knot_w(self, degree, num_ctrlpts):
        raise NotImplementedError
    
    def generate_knot(self, degree, num_ctrlpts, repeat_num, repeat=1):
        if degree == 0 or num_ctrlpts == 0:
            raise ValueError("Input values should be different than zero.")

        num_repeat = degree
        num_segments = num_ctrlpts - (degree + 1)

        if not self.clamped:
            num_repeat = 0
            num_segments = degree + num_ctrlpts - 1

        knot_vector = [0.0 for _ in range(0, num_repeat)]
        knot_vector += linspace(0.0, 1.0, num_segments + 2, repeat_num, repeat)
        knot_vector += [1.0 for _ in range(0, num_repeat)]
        return np.array(knot_vector)
    
    def generate_bumps(self, num_bumps, bump_height=5., base_extent=2, base_adjust=0, max_trials=25):
        if not self.control_points:
            raise RuntimeError("Object must be generated before calling this function")

        if not isinstance(num_bumps, int):
            num_bumps = int(num_bumps)
            warnings.warn("Number of bumps must be an integer value. Automatically rounding to %d" % num_bumps, UserWarning)

        if isinstance(bump_height, (list, tuple)):
            if len(bump_height) != num_bumps:
                raise ValueError("Number of bump heights must be equal to number of bumps")
            else:
                bump_height_is_array = True
        else:
            bump_height_is_array = False
            bump_height = [float(bump_height)]

        if base_extent < 1:
            raise ValueError("Base size must be bigger than 1 grid point")

        if (2 * base_extent) + base_adjust > self.num_ctrlpts_u or (2 * base_extent) + base_adjust > self.num_ctrlpts_v:
            raise ValueError("The area of the base must be less than the area of the grid")

        bump_list = []
        max_trials = int(max_trials)
        for _ in range(0, num_bumps):
            trials = 0
            while trials < max_trials:
                u = random.randint(base_extent, (self.num_ctrlpts_u - 1) - base_extent)
                v = random.randint(base_extent, (self.num_ctrlpts_v - 1) - base_extent)
                temp = [u, v]
                if self.check_bump(bump_list, temp, base_extent, base_adjust):
                    bump_list.append(temp)
                    trials = max_trials + 1 
                    break
                else:
                    trials = trials + 1
            if trials == max_trials:
                raise RuntimeError("Cannot generate %d bumps with a base extent of %d on this grid. "
                                    "You need to generate a grid larger than %dx%d." % (num_bumps, base_extent, self.num_ctrlpts_u, self.num_ctrlpts_v))

        idx = 0
        for u, v in bump_list:
            h_increment = bump_height[idx] / base_extent
            height = h_increment
            for j in range(base_extent - 1, -1, -1):
                self.create_bump(u, v, j, height)
                height += h_increment
            if bump_height_is_array:
                idx += 1

    def check_bump(self, uv_list, to_be_checked_uv, base_extent, padding):
        if not uv_list:
            return True

        for uv in uv_list:
            u = to_be_checked_uv[0]
            v = to_be_checked_uv[1]
            check_list = []
            for ur in range(-(base_extent + 1 + padding), base_extent + 2 + padding):
                for vr in range(-(base_extent + 1 + padding), base_extent + 2 + padding):
                    check_list.append([u + ur, v + vr])
            for check in check_list:
                if abs(uv[0] - check[0]) < self._delta and abs(uv[1] - check[1]) < self._delta:
                    return False
        return True

    def create_bump(self, u, v, jump, height):
        start_u = u - jump
        stop_u = u + jump + 1
        start_v = v - jump
        stop_v = v + jump + 1

        for j in range(start_v, stop_v):
            for i in range(start_u, stop_u):
                self.control_points[-1, j, i, 2] = height


class Cube(PrimitiveVolume):
    def __init__(self) -> None:
        super().__init__()
        self.shape = "Rectangle"
        self.start_point = None
        self.size = None

    def set_parameters(self, **body_dict):
        start_point = body_dict.get("start_point", [0., 0., 0.])
        size = body_dict.get("size")
        self.clamped = body_dict.get("clamped", True)

        if isinstance(start_point, (list, np.ndarray, tuple)):
            self.start_point = list(start_point)
        else:
            raise TypeError("The /StartPoint/ type should be a list, numpy.array or tuple")
        
        if isinstance(size, (list, np.ndarray, tuple)):
            self.size = list(size)
        else:
            raise TypeError("The /Size/ type should be a list, numpy.array or tuple")
    
    def generate_knot_u(self, degree, num_ctrlpts):
        self.check_ctrlpt_num(degree, num_ctrlpts)
        self.degree_u = degree
        self.knot_vector_u = self.generate_knot(degree, num_ctrlpts, repeat_num=0, repeat=1)

    def generate_knot_v(self, degree, num_ctrlpts):
        self.check_ctrlpt_num(degree, num_ctrlpts)
        self.degree_v = degree
        self.knot_vector_v = self.generate_knot(degree, num_ctrlpts, repeat_num=0, repeat=1)

    def generate_knot_w(self, degree, num_ctrlpts):
        self.check_ctrlpt_num(degree, num_ctrlpts)
        self.degree_w = degree
        self.knot_vector_w = self.generate_knot(degree, num_ctrlpts, repeat_num=0, repeat=1)
    
    def generate_ctrlpts(self):
        ctrlpts = []
        spacing_x = self.size[0] / (self.num_ctrlpts_u - 1)
        spacing_y = self.size[1] / (self.num_ctrlpts_v - 1)
        spacing_z = self.size[2] / (self.num_ctrlpts_w - 1)

        current_z = self.start_point[2]
        for _ in range(0, self.num_ctrlpts_w):
            facet = []
            current_y = self.start_point[1]
            for _ in range(0, self.num_ctrlpts_v):
                row = []
                current_x = self.start_point[0]
                for _ in range(0, self.num_ctrlpts_u):
                    row.append([current_x, current_y, current_z])
                    current_x = current_x + spacing_x
                facet.append(row)
                current_y = current_y + spacing_y
            ctrlpts.append(facet)
            current_z = current_z + spacing_z
        self.control_points = np.array(ctrlpts).reshape(-1, 3)

    def generate_weights(self):
        self.weights = np.array([[[1 for _ in range(self.num_ctrlpts_u)] for _ in range(self.num_ctrlpts_v)] for _ in range(self.num_ctrlpts_w)]).reshape(-1)
    

class Tube(PrimitiveVolume):
    def __init__(self) -> None:
        super().__init__()
        self.shape = "Tube"
        self.center = None
        self.length = None
        self.radius = None
        self.thickness = None

    def set_parameters(self, **body_dict):
        self.length = body_dict.get("length")
        self.radius = body_dict.get("radius")
        self.thickness = body_dict.get("thickness")
        self.clamped = body_dict.get("clamped", True)
        center = body_dict.get("base_center", [self.radius + self.thickness, self.radius + self.thickness, 0.])

        if isinstance(center, (list, np.ndarray, tuple)):
            self.center = list(center)
        else:
            raise TypeError("The /BaseCenter/ type should be a list, numpy.array or tuple")
        
    def check_parameters(self, degree, numctrlpts):
        if degree[0] != 2:
            degree[0] = 2
            warnings.warn(f"The degree along circunferential is automatically set as 2.")
        if numctrlpts[0] != 9:
            numctrlpts[0] = 9
            warnings.warn(f"The number of control points along circunferential is automatically set as 9.")
        return degree, numctrlpts

    def generate_knot_u(self, degree, num_ctrlpts):
        self.check_ctrlpt_num(degree, num_ctrlpts)
        self.degree_u = degree
        self.knot_vector_u = self.generate_knot(degree, num_ctrlpts, repeat_num=3, repeat=2)

    def generate_knot_v(self, degree, num_ctrlpts):
        self.check_ctrlpt_num(degree, num_ctrlpts)
        self.degree_v = degree
        self.knot_vector_v = self.generate_knot(degree, num_ctrlpts, repeat_num=0, repeat=1)

    def generate_knot_w(self, degree, num_ctrlpts):
        self.check_ctrlpt_num(degree, num_ctrlpts)
        self.degree_w = degree
        self.knot_vector_w = self.generate_knot(degree, num_ctrlpts, repeat_num=0, repeat=1)

    def generate_ctrlpts(self):
        if (self.num_ctrlpts_u - 1) % 8 > 0:
            self.num_ctrlpts_u = int(((self.num_ctrlpts_u - 1) // 8 + 1) * 8)
            warnings.warn(f"The number of control points along circumferential direction is automatically set as {self.num_ctrlpts_u}")

        ctrlpts = []
        numctrlpts_u8 = int((self.num_ctrlpts_u - 1) // 8)
        spacing_y = self.thickness / (self.num_ctrlpts_v - 1)
        spacing_z = self.length / (self.num_ctrlpts_w - 1)

        current_z = self.center[2]
        for _ in range(0, self.num_ctrlpts_w):
            facet = []
            current_y = self.radius
            for _ in range(0, self.num_ctrlpts_v):
                spacing_x = current_y / numctrlpts_u8
                row = []

                current_x = 0.
                for _ in range(0, numctrlpts_u8):
                    row.append([current_y, current_x, current_z])
                    current_x = current_x + spacing_x
                
                current_x = current_y
                for _ in range(0, 2 * numctrlpts_u8):
                    row.append([current_x, current_y, current_z])
                    current_x = current_x - spacing_x
                
                current_x = current_y
                for _ in range(0, 2 * numctrlpts_u8):
                    row.append([-current_y, current_x, current_z])
                    current_x = current_x - spacing_x
                
                current_x = -current_y
                for _ in range(0, 2 * numctrlpts_u8):
                    row.append([current_x, -current_y, current_z])
                    current_x = current_x + spacing_x

                current_x = -current_y
                for _ in range(0, numctrlpts_u8 + 1):
                    row.append([current_y, current_x, current_z])
                    current_x = current_x + spacing_x

                facet.append(row)
                current_y = current_y + spacing_y
            ctrlpts.append(facet)
            current_z = current_z + spacing_z
        self.control_points = np.array(ctrlpts).reshape(-1, 3)

    def generate_weights(self):
        weights = []
        numctrlpts_u8 = int((self.num_ctrlpts_u - 1) // 8)

        for _ in range(0, self.num_ctrlpts_w):
            facet = []
            current_y = self.radius
            for _ in range(0, self.num_ctrlpts_v):
                row = []
                spacing_x = current_y / numctrlpts_u8

                current_x = 0.
                for _ in range(0, numctrlpts_u8):
                    row.append(np.cos(np.arctan2(current_x, current_y)))
                    current_x = current_x + spacing_x
                
                current_x = current_y
                for _ in range(0, 2 * numctrlpts_u8):
                    row.append(np.cos(np.arctan2(current_x, current_y)))
                    current_x = current_x - spacing_x
                
                current_x = current_y
                for _ in range(0, 2 * numctrlpts_u8):
                    row.append(np.cos(np.arctan2(current_x, current_y)))
                    current_x = current_x - spacing_x
                
                current_x = -current_y
                for _ in range(0, 2 * numctrlpts_u8):
                    row.append(np.cos(np.arctan2(current_x, current_y)))
                    current_x = current_x + spacing_x

                current_x = -current_y
                for _ in range(0, numctrlpts_u8 + 1):
                    row.append(np.cos(np.arctan2(current_x, current_y)))
                    current_x = current_x + spacing_x

                facet.append(row)
            weights.append(facet)
        self.weights = np.array(weights).reshape(-1)


class Cylinder(PrimitiveVolume):
    def __init__(self) -> None:
        super().__init__()
        self.shape = "Cylinder"
        self.center = None
        self.radius = None
        self.length = None

    def set_parameters(self, **body_dict):
        self.radius = body_dict.get("radius", 1.0)
        self.length = body_dict.get("length", 1.0)
        self.center = np.array(body_dict.get("base_center", [0., 0., 0.]))
        self.clamped = body_dict.get("clamped", True)

    def generate_knot_u(self, degree, num_ctrlpts):
        self.check_ctrlpt_num(degree, num_ctrlpts)
        self.degree_u = degree
        # u: around circumferential, set degree=2 and 9 ctrlpts for circle
        self.knot_vector_u = self.generate_knot(degree, num_ctrlpts, repeat_num=3, repeat=2)

    def generate_knot_v(self, degree, num_ctrlpts):
        self.check_ctrlpt_num(degree, num_ctrlpts)
        self.degree_v = degree
        # v: along radius
        self.knot_vector_v = self.generate_knot(degree, num_ctrlpts, repeat_num=0, repeat=1)

    def generate_knot_w(self, degree, num_ctrlpts):
        self.check_ctrlpt_num(degree, num_ctrlpts)
        self.degree_w = degree
        # w: along height
        self.knot_vector_w = self.generate_knot(degree, num_ctrlpts, repeat_num=0, repeat=1)

    def generate_ctrlpts(self):
        ctrlpts = []
        n_u = self.num_ctrlpts_u
        n_v = self.num_ctrlpts_v
        n_w = self.num_ctrlpts_w

        for k in range(n_w):
            z = self.center[2] + self.length * k / (n_w - 1)
            for j in range(n_v):
                r = self.radius * j / (n_v - 1)
                for i in range(n_u):
                    theta = 2 * np.pi * i / (n_u - 1)
                    x = self.center[0] + r * np.cos(theta)
                    y = self.center[1] + r * np.sin(theta)
                    ctrlpts.append([x, y, z])
        self.control_points = np.array(ctrlpts)

    def generate_weights(self):
        n_u = self.num_ctrlpts_u
        n_v = self.num_ctrlpts_v
        n_w = self.num_ctrlpts_w
        sqrt2_2 = np.sqrt(2) / 2
        weights = []

        for k in range(n_w):
            for j in range(n_v):
                for i in range(n_u):
                    # 圆周方向权重
                    if i % 2 == 1:
                        w = sqrt2_2
                    else:
                        w = 1.0
                    weights.append(w)
        self.weights = np.array(weights)