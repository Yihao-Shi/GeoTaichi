from src.utils.errors import ConfigurationError, SimulationStateError
import numpy as np
import warnings, random
from src.nurbs.NurbsSurface import NurbsSurface
from src.utils.linalg import linspace


class PrimitiveSurface(NurbsSurface):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shape = None
        self.clamped = True
        self._delta = 1e-6

    def check_ctrlpt_num(self, degree, num_ctrlpts):
        if num_ctrlpts <= degree:
            raise SimulationStateError(f"The minimum number of control points is {degree + 1}")

    def set_parameters(self, body_dict):
        raise NotImplementedError

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

    def generate_bumps(self, num_bumps, bump_height=0.1, base_extent=2, base_adjust=0, max_trials=25):
        """
        在二维控制点平面上生成 bumps
        """
        if self.control_points is None:
            raise ConfigurationError("Control points must be generated before calling this function")

        if not isinstance(num_bumps, int):
            num_bumps = int(num_bumps)
            warnings.warn(f"Number of bumps rounded to {num_bumps}", UserWarning)

        if base_extent < 1:
            raise ValueError("Base size must be >= 1")

        num_ctrlpts_v, num_ctrlpts_u, _ = self.control_points.shape

        if (2 * base_extent) + base_adjust > num_ctrlpts_u or (2 * base_extent) + base_adjust > num_ctrlpts_v:
            raise ValueError("Base extent too large for current grid")

        bump_list = []
        for _ in range(num_bumps):
            trials = 0
            while trials < max_trials:
                u = random.randint(base_extent, num_ctrlpts_u - 1 - base_extent)
                v = random.randint(base_extent, num_ctrlpts_v - 1 - base_extent)
                if self.check_bump(bump_list, [u, v], base_extent, base_adjust):
                    bump_list.append([u, v])
                    break
                trials += 1
            if trials == max_trials:
                raise SimulationStateError(f"Cannot place {num_bumps} bumps on the grid")

        # 填充 bump 高度
        for u, v in bump_list:
            h_increment = bump_height / base_extent
            height = h_increment
            for j in range(base_extent - 1, -1, -1):
                self.create_bump(u, v, j, height)
                height += h_increment

    def check_bump(self, uv_list, to_be_checked_uv, base_extent, padding):
        if not uv_list:
            return True
        u, v = to_be_checked_uv
        for uv in uv_list:
            for ur in range(-(base_extent + 1 + padding), base_extent + 2 + padding):
                for vr in range(-(base_extent + 1 + padding), base_extent + 2 + padding):
                    if abs(uv[0] - (u + ur)) < self._delta and abs(uv[1] - (v + vr)) < self._delta:
                        return False
        return True

    def create_bump(self, u, v, jump, height):
        """
        bump 在二维平面上修改 y 值（或第二个维度）
        """
        start_u = max(u - jump, 0)
        stop_u = min(u + jump + 1, self.control_points.shape[1])
        start_v = max(v - jump, 0)
        stop_v = min(v + jump + 1, self.control_points.shape[0])

        for j in range(start_v, stop_v):
            for i in range(start_u, stop_u):
                self.control_points[j, i, 1] += height  # 第二维作为 bump 高度
