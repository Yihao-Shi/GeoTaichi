from src.utils.errors import ConfigurationError, SimulationStateError

import numpy as np
import warnings, random
from src.nurbs.NurbsVolume import NurbsVolume
from src.utils.linalg import linspace


class PrimitiveVolume(NurbsVolume):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shape = None
        self.clamped = True

    def check_ctrlpt_num(self, degree, num_ctrlpts):
        if num_ctrlpts <= degree:
            raise SimulationStateError(f"The minimum number of control points is {degree + 1}")

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

    def generate_bumps(self, num_bumps, bump_height=5.0, base_extent=2, base_adjust=0, max_trials=25):
        if not self.control_points:
            raise ConfigurationError("Object must be generated before calling this function")

        if not isinstance(num_bumps, int):
            num_bumps = int(num_bumps)
            warnings.warn(
                "Number of bumps must be an integer value. Automatically rounding to %d" % num_bumps, UserWarning
            )

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
                raise SimulationStateError(
                    "Cannot generate %d bumps with a base extent of %d on this grid. "
                    "You need to generate a grid larger than %dx%d."
                    % (num_bumps, base_extent, self.num_ctrlpts_u, self.num_ctrlpts_v)
                )

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
