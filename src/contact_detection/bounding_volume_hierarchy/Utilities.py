from __future__ import annotations
import numpy as np
import operator


def validate_bvh_rebuild_interval(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("bvh_rebuild_interval must be a positive integer")
    try:
        interval = operator.index(value)
    except TypeError as exc:
        raise ValueError("bvh_rebuild_interval must be a positive integer") from exc
    if interval <= 0:
        raise ValueError("bvh_rebuild_interval must be a positive integer")
    return interval
