from src.nurbs.PrimitiveVolume import PrimitiveVolume
import numpy as np


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
        self.center = np.array(body_dict.get("base_center", [0.0, 0.0, 0.0]))
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
