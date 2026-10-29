"""Validated affine level-set input storage and initialization result schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_float_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(array)


def _as_positive_scalar(value, name):
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


@dataclass(frozen=True)
class TrilinearLevelSet:
    """A scalar level-set field on an x-fast regular Cartesian grid.

    ``template_to_grid`` maps an affine template coordinate to the coordinate
    frame in which the Cartesian SDF grid is axis aligned.  This permits the
    affine template preprocessing step to center/rotate its visualization and
    quadrature mesh without resampling the signed-distance field.
    """

    origin: np.ndarray
    spacing: float
    shape: np.ndarray
    values: np.ndarray
    template_to_grid: np.ndarray = None
    template_to_grid_offset: np.ndarray = None

    def __post_init__(self):
        origin = _as_float_array(self.origin, (3,), "origin")
        spacing = _as_positive_scalar(self.spacing, "spacing")
        shape = np.asarray(self.shape, dtype=np.int32)
        if shape.shape != (3,) or np.any(shape < 2):
            raise ValueError("shape must contain three grid sizes >= 2")
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        expected = int(np.prod(shape, dtype=np.int64))
        if values.size != expected:
            raise ValueError(f"values contains {values.size} nodes, expected {expected}")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite")
        linear = (
            np.eye(3, dtype=np.float64)
            if self.template_to_grid is None
            else _as_float_array(self.template_to_grid, (3, 3), "template_to_grid")
        )
        offset = (
            np.zeros(3, dtype=np.float64)
            if self.template_to_grid_offset is None
            else _as_float_array(
                self.template_to_grid_offset,
                (3,),
                "template_to_grid_offset",
            )
        )
        determinant = float(np.linalg.det(linear))
        if abs(determinant) <= 1.0e-12:
            raise ValueError("template_to_grid must be nonsingular")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "shape", np.ascontiguousarray(shape, dtype=np.int32))
        object.__setattr__(self, "values", np.ascontiguousarray(values, dtype=np.float64))
        object.__setattr__(self, "template_to_grid", linear)
        object.__setattr__(self, "template_to_grid_offset", offset)

    @property
    def upper(self):
        return self.origin + self.spacing * (self.shape - 1)

    @classmethod
    def from_object_grid(
        cls,
        grid,
        *,
        template_to_grid=None,
        template_to_grid_offset=None,
    ):
        return cls(
            origin=np.asarray(grid.start_point, dtype=np.float64),
            spacing=float(grid.grid_space),
            shape=np.asarray(grid.gnum, dtype=np.int32),
            values=np.asarray(grid.distance_field, dtype=np.float64),
            template_to_grid=template_to_grid,
            template_to_grid_offset=template_to_grid_offset,
        )


@dataclass
class NonpenetrationResult:
    controls: np.ndarray
    translations: np.ndarray
    initial_minimum_gap: float
    minimum_gap: float
    iterations: int
    continuation_stages: int
    success: bool
    message: str
    objective: float


__all__ = ["TrilinearLevelSet", "NonpenetrationResult"]
