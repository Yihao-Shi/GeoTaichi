"""Allocation of global contact pair buffers from a primitive count."""

import math


def validate_contact_coordination_number(value, name="contact_coordination_number"):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def contact_pair_capacity(primitive_count, coordination_number):
    return max(1, math.ceil(primitive_count * coordination_number))


def reject_contact_pair_limits(settings, *keys):
    for key in settings:
        if str(key).lower() in keys:
            raise ValueError(f"{key} was removed. Set a contact coordination number instead.")
