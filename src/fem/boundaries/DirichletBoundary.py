from __future__ import annotations
from src.fem.boundaries.BoundaryCondition import DirichletEntry
from src.fem.boundaries.Utilities import _evaluate
from src.fem.boundaries.Utilities import _nodes
import numpy as np


class DirichletBoundary:
    """Prescribed displacement constraints.

    ``nodes`` may be node indices or a named mesh node set. ``value`` may be
    scalar, a vector, one value per node, or a callable ``value(time, X)``.
    """

    @staticmethod
    def _components(components, dimension=3):
        if components is None or str(components).lower() == "all":
            return np.arange(dimension, dtype=np.int32)
        if isinstance(components, str):
            names = {"x": 0, "y": 1, "z": 2}
            try:
                values = [names[component] for component in components.lower() if component.strip()]
            except KeyError as exc:
                raise ValueError("components must contain only x, y, and z") from exc
        else:
            values = np.asarray(components, dtype=np.int32).reshape(-1)
        values = np.unique(np.asarray(values, dtype=np.int32))
        if values.size == 0 or values[0] < 0 or values[-1] >= dimension:
            raise ValueError(f"components must lie in [0, {dimension})")
        return values

    def __init__(self):
        self.entries = []

    @property
    def time_dependent(self):
        """Whether prescribed values must be reevaluated during integration."""
        return any(callable(entry.value) for entry in self.entries)

    def add(self, nodes, components="all", value=0.0):
        self.entries.append(DirichletEntry(nodes, components, value))
        return self

    append = add

    def values(self, mesh, time=0.0):
        constrained = {}
        for entry in self.entries:
            node_ids = _nodes(mesh, entry.nodes)
            component_ids = DirichletBoundary._components(entry.components)
            coordinates = mesh.points[node_ids]
            value = _evaluate(entry.value, time, coordinates)
            if value.ndim == 0:
                value = np.full((node_ids.size, component_ids.size), float(value))
            elif value.ndim == 1:
                if value.size == component_ids.size:
                    value = np.repeat(value[None, :], node_ids.size, axis=0)
                elif component_ids.size == 1 and value.size == node_ids.size:
                    value = value[:, None]
                elif value.size == 3:
                    value = np.repeat(value[component_ids][None, :], node_ids.size, axis=0)
                else:
                    raise ValueError("Dirichlet value shape does not match its nodes/components")
            elif value.shape == (node_ids.size, 3):
                value = value[:, component_ids]
            if value.shape != (node_ids.size, component_ids.size):
                raise ValueError("Dirichlet value shape does not match its nodes/components")
            for local_node, node_id in enumerate(node_ids):
                for local_component, component in enumerate(component_ids):
                    dof = 3 * int(node_id) + int(component)
                    prescribed = float(value[local_node, local_component])
                    if dof in constrained and not np.isclose(constrained[dof], prescribed):
                        raise ValueError(f"conflicting prescribed values for FEM degree of freedom {dof}")
                    constrained[dof] = prescribed
        if not constrained:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
        dofs = np.asarray(sorted(constrained), dtype=np.int32)
        return dofs, np.asarray([constrained[dof] for dof in dofs], dtype=np.float64)
