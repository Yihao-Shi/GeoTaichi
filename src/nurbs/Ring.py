from src.nurbs.PrimitiveSurface import PrimitiveSurface

import numpy as np


class Ring(PrimitiveSurface):
    def __init__(self):
        super().__init__()
        self.shape = "Ring"
        self.center = None
        self.inner_radius = None
        self.outer_radius = None

    def set_parameters(self, **body_dict):
        self.center = np.array(body_dict.get("center", [0.0, 0.0]))
        self.inner_radius = body_dict.get("inner_radius", 0.5)
        self.outer_radius = body_dict.get("outer_radius", 1.0)
        self.clamped = body_dict.get("clamped", True)

    def generate_knot_u(self, degree, num_ctrlpts):
        self.check_ctrlpt_num(degree, num_ctrlpts)
        self.degree_u = degree
        self.knot_vector_u = self.generate_knot(degree, num_ctrlpts, repeat_num=0, repeat=1)

    def generate_knot_v(self, degree, num_ctrlpts):
        self.check_ctrlpt_num(degree, num_ctrlpts)
        self.degree_v = degree
        self.knot_vector_v = self.generate_knot(degree, num_ctrlpts, repeat_num=0, repeat=1)

    def generate_ctrlpts(self):
        n_u = self.num_ctrlpts_u  # 圆周方向
        n_v = self.num_ctrlpts_v  # 径向方向
        ctrlpts = []

        for j in range(n_v):
            r = self.inner_radius + (self.outer_radius - self.inner_radius) * j / max(n_v - 1, 1)
            for i in range(n_u):
                theta = 2 * np.pi * i / n_u  # 角度均匀分布
                x = self.center[0] + r * np.cos(theta)
                y = self.center[1] + r * np.sin(theta)
                ctrlpts.append([x, y])
        self.control_points = np.array(ctrlpts).reshape(-1, 2)

    def generate_weights(self):
        n_u = self.num_ctrlpts_u
        n_v = self.num_ctrlpts_v
        sqrt2_2 = np.sqrt(2) / 2
        weights = []

        for j in range(n_v):
            row = []
            for i in range(n_u):
                # 保持圆周方向光滑，角度接近 45° 的控制点使用 sqrt(2)/2，其余使用 1
                angle_fraction = i / n_u
                if np.isclose(angle_fraction % 0.25, 0.125, atol=1e-8):
                    w = sqrt2_2
                else:
                    w = 1.0
                row.append(w)
            weights.append(row)
        self.weights = np.array(weights).reshape(-1)
