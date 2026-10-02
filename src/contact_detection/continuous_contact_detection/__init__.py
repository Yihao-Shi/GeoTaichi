"""Public continuous contact-detection primitives."""

from .CCD import (
    ccd_mode_parameters,
    edge_edge_ccd,
    linear_gap_ccd,
    point_edge_ccd,
    point_point_ccd,
    point_triangle_ccd,
)
from .AdditiveCCD import (
    edge_edge_accd,
    linear_gap_accd,
    point_edge_accd,
    point_nurbs_accd_increment,
    point_point_accd,
    point_triangle_accd,
)
from .PolynomialCCD import (
    deformation_gradient_ccd,
    point_point_quadratic_ccd,
    real_cubic_roots,
)

__all__ = [
    "ccd_mode_parameters",
    "deformation_gradient_ccd",
    "edge_edge_accd",
    "edge_edge_ccd",
    "linear_gap_accd",
    "linear_gap_ccd",
    "point_edge_accd",
    "point_edge_ccd",
    "point_nurbs_accd_increment",
    "point_point_accd",
    "point_point_ccd",
    "point_point_quadratic_ccd",
    "real_cubic_roots",
    "point_triangle_ccd",
    "point_triangle_accd",
]
