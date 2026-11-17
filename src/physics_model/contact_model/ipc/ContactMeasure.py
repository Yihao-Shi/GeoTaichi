"""Geometry-only quadrature measures for IPC contact stencils.

The routines in this module intentionally do not include IPC time-step,
barrier-width, or material coefficients.  A contact record stores a geometric
measure and the constitutive model applies its own scaling exactly once.
"""

from __future__ import annotations

import numpy as np
import taichi as ti


def reference_point_measure(volume, dimension):
    """Fallback boundary measure inferred from a material-point volume.

    Explicit boundary quadrature or user-provided measures are preferable.
    This helper is kept explicit so callers cannot accidentally confuse a
    volume with a surface measure.
    """
    volume = float(volume)
    dimension = int(dimension)
    if not np.isfinite(volume) or volume < 0.0:
        raise ValueError("reference point volume must be finite and non-negative")
    if dimension not in (2, 3):
        raise ValueError("reference_point_measure supports dimension 2 or 3")
    return volume ** ((dimension - 1.0) / dimension)


@ti.func
def symmetric_contact_measure(measure_a, measure_b):
    """Symmetric quadrature weight for a pair of sampled primitives."""
    return 0.5 * (measure_a + measure_b)


@ti.func
def point_contact_measure(measure):
    """Clamp round-off-only negative point measures without hiding bad input."""
    return ti.max(measure, 0.0)


@ti.func
def curve_quadrature_measure(tangent, parameter_weight):
    """Physical IGA curve measure ``||dx/du|| du``."""
    return ti.max(parameter_weight, 0.0) * tangent.norm()


@ti.func
def surface_quadrature_measure(tangent_u, tangent_v, parameter_weight):
    """Physical IGA surface measure ``||dx/du x dx/dv|| du dv``."""
    return ti.max(parameter_weight, 0.0) * tangent_u.cross(tangent_v).norm()
