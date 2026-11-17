"""Shared model parameters for fem_affine_mixed_triaxial_25."""

from __future__ import annotations

PARTICLE_COUNT = 27

SOFT_PERCENT = 25.0

LOWER = 0.277

UPPER = 0.523

PLATE_SPAN = 0.236

CONTACT_DHAT = 0.002

CONTACT_KAPPA = 4.0e4

FEM_PT_STENCILS_PER_COMPONENT_PAIR = 128

FEM_EE_STENCILS_PER_COMPONENT_PAIR = 256


def plate_specs():
    inner = LOWER + 0.008
    return (
        ("left", (LOWER - 0.020, inner, inner), (0.020, PLATE_SPAN, PLATE_SPAN)),
        ("right", (UPPER, inner, inner), (0.020, PLATE_SPAN, PLATE_SPAN)),
        ("front", (inner, LOWER - 0.020, inner), (PLATE_SPAN, 0.020, PLATE_SPAN)),
        ("back", (inner, UPPER, inner), (PLATE_SPAN, 0.020, PLATE_SPAN)),
        ("bottom", (inner, inner, LOWER - 0.020), (PLATE_SPAN, PLATE_SPAN, 0.020)),
        ("top", (inner, inner, UPPER), (PLATE_SPAN, PLATE_SPAN, 0.020)),
    )
