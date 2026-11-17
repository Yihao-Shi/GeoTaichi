from src.nurbs.PrimitiveVolume import PrimitiveVolume

import numpy as np
import warnings


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
        center = body_dict.get("base_center", [self.radius + self.thickness, self.radius + self.thickness, 0.0])

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
            warnings.warn(
                f"The number of control points along circumferential direction is automatically set as {self.num_ctrlpts_u}"
            )

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

                current_x = 0.0
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

                current_x = 0.0
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
