import numpy as np

import src.iga.config as config


class Primitives:
    def __init__(self):
        self.body = {}
        self.body_counter = 0
        self.num_element = [0, 0, 0]
        self.num_knot = [0, 0, 0]
        self.num_ctrlpts = 0
        self.num_primitives = 0

    def _get_name(self, name):
        if name is None:
            name = f"body_{self.body_counter}"
            self.body_counter += 1
        return name

    def _pack_meta(self, primitive, init_v, name, rest_shape=None):
        if len(init_v) != config.DIM:
            raise ValueError(f"{name}: dimension mismatch! primitives are {config.DIM}D, " f"but init_v={init_v}")

        if rest_shape is not None:
            rest_shape = np.asarray(rest_shape, dtype=np.float64)
            expected_shape = (primitive.num_ctrlpts, config.DIM)
            if rest_shape.shape != expected_shape:
                raise ValueError(f"{name}: rest_shape must have shape {expected_shape}")
            if not np.all(np.isfinite(rest_shape)):
                raise ValueError(f"{name}: rest_shape must be finite")
            rest_shape = np.ascontiguousarray(rest_shape)

        self.body[name] = {
            "primitive": primitive,
            "init_v": np.array(init_v),
            "rest_shape": rest_shape,
        }
        self.num_ctrlpts += primitive.num_ctrlpts
        self.num_primitives += 1

    def append(self, primitive, name=None, init_v=None, rest_shape=None):
        if init_v is None:
            init_v = [0] * config.DIM
        name = self._get_name(name)
        self._pack_meta(primitive, init_v, name, rest_shape)

    def finialize(self):
        for name, meta in self.body.items():
            primitive = meta["primitive"]
            self.num_element[0] += primitive.num_element_u
            self.num_element[1] += primitive.num_element_v
            self.num_knot[0] += primitive.num_knot_u
            self.num_knot[1] += primitive.num_knot_v
            if primitive.dimension == 3:
                self.num_element[2] += primitive.num_element_w
                self.num_knot[2] += primitive.num_knot_w


class Surface:
    def __init__(self):
        self.surfaces = []
        self.num_surfaces = 0
        self.num_ctrlpts = 0
        self.num_knot_u = 0
        self.num_knot_v = 0
        self.num_elements = 0

    def append(self, primitive):
        self.surfaces.append(primitive)

    def finialize(self):
        for primitive in self.surfaces:
            self.num_ctrlpts += primitive.num_ctrlpts
            self.num_knot_u += primitive.num_knot_u
            self.num_knot_v += primitive.num_knot_v
            self.num_elements += primitive.num_element_u * primitive.num_element_v
            self.num_surfaces += 1
