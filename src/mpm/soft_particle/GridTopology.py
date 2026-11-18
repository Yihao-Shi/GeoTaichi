from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SoftGridTopology:
    compact_origin: np.ndarray
    compact_shape: np.ndarray
    logical_count: int
    compact_count: int
    support_count: int
    padding_cells: int
    storage: str
    levelset_extent_cells: int = 0
    verlet_padding_cells: int = 0
    levelset_verlet_padding_cells: int = 0


@dataclass(frozen=True)
class SoftMechanicalGrid:
    origin: np.ndarray
    shape: np.ndarray
    spacing: float
    axes: tuple = None

    @property
    def gnum(self):
        return self.shape

    @property
    def gridSum(self):
        return int(np.prod(self.shape, dtype=np.int64))

    def min_box(self):
        if self.axes is None:
            return self.origin
        return np.asarray([axis[0] for axis in self.axes], dtype=np.float64)

    def max_box(self):
        if self.axes is None:
            return self.origin + (self.shape - 1) * self.spacing
        return np.asarray([axis[-1] for axis in self.axes], dtype=np.float64)

    @property
    def coordinate_axes(self):
        if self.axes is not None:
            return self.axes
        return tuple(self.origin[d] + np.arange(int(self.shape[d]), dtype=np.float64) * self.spacing for d in range(3))

    @property
    def minimum_spacing(self):
        if self.axes is None:
            return float(self.spacing)
        return min(float(np.min(np.diff(axis))) for axis in self.axes)

    @property
    def is_uniform(self):
        return self.axes is None
