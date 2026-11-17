from src.nurbs.PrimitiveVolume import PrimitiveVolume
import numpy as np


class Cube(PrimitiveVolume):
    def __init__(self) -> None:
        super().__init__()
        self.shape = "Rectangle"
        self.start_point = None
        self.size = None

    def set_parameters(self, **body_dict):
        start_point = body_dict.get("start_point", [0.0, 0.0, 0.0])
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
        self.weights = np.array(
            [
                [[1 for _ in range(self.num_ctrlpts_u)] for _ in range(self.num_ctrlpts_v)]
                for _ in range(self.num_ctrlpts_w)
            ]
        ).reshape(-1)
