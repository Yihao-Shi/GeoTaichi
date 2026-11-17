from __future__ import annotations
import numpy as np


def _nodes(mesh, nodes):
    if isinstance(nodes, str):
        return mesh.select_nodes(node_set=nodes)
    values = np.unique(np.asarray(nodes, dtype=np.int32).reshape(-1))
    if values.size and (values[0] < 0 or values[-1] >= mesh.number_of_nodes):
        raise ValueError("boundary condition contains an invalid node index")
    return values


def _evaluate(value, time, coordinates):
    if callable(value):
        evaluated = value(float(time), np.asarray(coordinates))
    else:
        evaluated = value
    return np.asarray(evaluated, dtype=np.float64)
