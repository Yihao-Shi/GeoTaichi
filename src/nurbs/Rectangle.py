from src.nurbs.PrimitiveSurface import PrimitiveSurface

import numpy as np


class Rectangle(PrimitiveSurface):
    def __init__(self) -> None:
        super().__init__()
        self.shape = "Rectangle2D"
        self.start_point = None
        self.size = None

    def set_parameters(self, **body_dict):
        self.start_point = body_dict.get("start_point", [0.0, 0.0])
        self.size = body_dict.get("size", [1.0, 1.0])
        self.clamped = body_dict.get("clamped", True)

        if not isinstance(self.start_point, (list, tuple, np.ndarray)):
            raise TypeError("start_point must be list, tuple or np.ndarray")
        if not isinstance(self.size, (list, tuple, np.ndarray)):
            raise TypeError("size must be list, tuple or np.ndarray")

    def generate_knot_u(self, degree, num_ctrlpts):
        self.check_ctrlpt_num(degree, num_ctrlpts)
        self.degree_u = degree
        self.knot_vector_u = self.generate_knot(degree, num_ctrlpts, repeat_num=0, repeat=1)

    def generate_knot_v(self, degree, num_ctrlpts):
        self.check_ctrlpt_num(degree, num_ctrlpts)
        self.degree_v = degree
        self.knot_vector_v = self.generate_knot(degree, num_ctrlpts, repeat_num=0, repeat=1)

    def generate_ctrlpts(self):
        ctrlpts = []
        spacing_x = self.size[0] / (self.num_ctrlpts_u - 1)
        spacing_y = self.size[1] / (self.num_ctrlpts_v - 1)

        for j in range(self.num_ctrlpts_v):
            for i in range(self.num_ctrlpts_u):
                x = self.start_point[0] + i * spacing_x
                y = self.start_point[1] + j * spacing_y
                ctrlpts.append([x, y])
        self.control_points = np.array(ctrlpts)

    def generate_weights(self):
        self.weights = np.ones((self.num_ctrlpts_v * self.num_ctrlpts_u))
