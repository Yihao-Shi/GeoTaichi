"""Dirichlet and Neumann boundary conditions for FEM meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _nodes(mesh, nodes):
    if isinstance(nodes, str):
        return mesh.select_nodes(node_set=nodes)
    values = np.unique(np.asarray(nodes, dtype=np.int32).reshape(-1))
    if values.size and (values[0] < 0 or values[-1] >= mesh.number_of_nodes):
        raise ValueError("boundary condition contains an invalid node index")
    return values


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


def _evaluate(value, time, coordinates):
    if callable(value):
        evaluated = value(float(time), np.asarray(coordinates))
    else:
        evaluated = value
    return np.asarray(evaluated, dtype=np.float64)


@dataclass
class _DirichletEntry:
    nodes: object
    components: object
    value: object


class DirichletBoundary:
    """Prescribed displacement constraints.

    ``nodes`` may be node indices or a named mesh node set. ``value`` may be
    scalar, a vector, one value per node, or a callable ``value(time, X)``.
    """

    def __init__(self):
        self.entries = []

    @property
    def time_dependent(self):
        """Whether prescribed values must be reevaluated during integration."""
        return any(callable(entry.value) for entry in self.entries)

    def add(self, nodes, components="all", value=0.0):
        self.entries.append(_DirichletEntry(nodes, components, value))
        return self

    append = add

    def values(self, mesh, time=0.0):
        constrained = {}
        for entry in self.entries:
            node_ids = _nodes(mesh, entry.nodes)
            component_ids = _components(entry.components)
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


@dataclass
class _LoadEntry:
    kind: str
    value: object
    nodes: object = None
    selector: object = None
    total: bool = False
    pressure: bool = False


class NeumannBoundary:
    """Nodal forces, surface/edge tractions, and pressure loads."""

    def __init__(self):
        self.entries = []
        # Surface topology, selectors, reference normals, and measures are
        # time independent.  Cache them by mesh/configuration so a transient
        # load updates only its values instead of rebuilding boundary facets
        # on the host at every explicit step.
        self._prepared_geometry = {}

    @property
    def time_dependent(self):
        """Whether load values must be reevaluated during integration."""
        return any(callable(entry.value) for entry in self.entries)

    def _invalidate_geometry(self):
        self._prepared_geometry.clear()

    def add_nodal_force(self, nodes, value, total=False):
        self.entries.append(_LoadEntry("nodal", value, nodes=nodes, total=bool(total)))
        self._invalidate_geometry()
        return self

    def add_traction(self, value, selector=None):
        """Add dead traction on volume boundary facets or membrane faces."""
        self.entries.append(_LoadEntry("surface", value, selector=selector))
        self._invalidate_geometry()
        return self

    def add_pressure(self, value, selector=None):
        """Add positive outward dead pressure on reference facets/faces."""
        self.entries.append(_LoadEntry("surface", value, selector=selector, pressure=True))
        self._invalidate_geometry()
        return self

    def add_edge_traction(self, value, selector=None):
        """Add force per reference length on membrane boundary edges."""
        self.entries.append(_LoadEntry("edge", value, selector=selector))
        self._invalidate_geometry()
        return self

    def append(self, nodes, value):
        return self.add_nodal_force(nodes, value)

    @staticmethod
    def _selected(selector, centroids):
        if selector is None:
            return np.ones(centroids.shape[0], dtype=bool)
        selected = np.asarray(selector(centroids))
        if selected.ndim == 0:
            selected = np.full(centroids.shape[0], bool(selected))
        if selected.shape != (centroids.shape[0],):
            raise ValueError("traction selector must return one boolean per candidate facet")
        return selected.astype(bool)

    @staticmethod
    def _vectors(value, time, centroids, normals, pressure):
        evaluated = _evaluate(value, time, centroids)
        return NeumannBoundary._vectors_from_evaluated(evaluated, centroids, normals, pressure)

    @staticmethod
    def _vectors_from_evaluated(evaluated, centroids, normals, pressure):
        count = centroids.shape[0]
        if pressure:
            if evaluated.ndim == 0:
                return float(evaluated) * normals
            evaluated = evaluated.reshape(-1)
            if evaluated.size != count:
                raise ValueError("pressure callable must return one scalar per selected facet")
            return evaluated[:, None] * normals
        if evaluated.ndim == 1 and evaluated.size == 3:
            return np.repeat(evaluated[None, :], count, axis=0)
        if evaluated.shape != (count, 3):
            raise ValueError("traction must be a 3-vector or one 3-vector per selected facet")
        return evaluated

    def _prepare_geometry(self, mesh, axisymmetric, axis_offset):
        key = (id(mesh), bool(axisymmetric), float(axis_offset), len(self.entries))
        cached = self._prepared_geometry.get(key)
        if cached is not None:
            return cached
        prepared = []
        for entry in self.entries:
            if entry.kind == "nodal":
                node_ids = _nodes(mesh, entry.nodes)
                prepared.append(
                    {
                        "node_ids": node_ids,
                        "coordinates": mesh.points[node_ids],
                    }
                )
                continue
            if entry.kind == "surface":
                if axisymmetric:
                    facets, centroids, normals, measure = self._axisymmetric_edge_geometry(mesh, axis_offset)
                elif mesh.is_membrane:
                    facets, centroids, normals, measure = self._membrane_geometry(mesh)
                else:
                    facets, owners = mesh.boundary_facets()
                    centroids, normals, measure = self._facet_geometry(mesh, facets, owners)
            elif entry.kind == "edge":
                if not mesh.is_membrane:
                    raise ValueError("edge traction is only defined for membrane meshes")
                if axisymmetric:
                    facets, centroids, normals, measure = self._axisymmetric_edge_geometry(mesh, axis_offset)
                else:
                    facets, _ = mesh.boundary_facets()
                    coordinates = mesh.rest_shape[facets]
                    centroids = np.mean(coordinates, axis=1)
                    tangent = coordinates[:, 1] - coordinates[:, 0]
                    measure = np.linalg.norm(tangent, axis=1)
                    normals = np.zeros_like(centroids)
            selected = self._selected(entry.selector, centroids)
            selected_facets = facets[selected]
            selected_centroids = centroids[selected]
            selected_normals = normals[selected]
            selected_measure = measure[selected]
            geometry = {
                "facets": selected_facets,
                "centroids": selected_centroids,
                "normals": selected_normals,
                "measure": selected_measure,
            }
            if entry.pressure:
                unit_nodal_force = np.zeros((mesh.number_of_nodes, 3), dtype=np.float64)
                unit_values = selected_normals * (selected_measure / selected_facets.shape[1])[:, None]
                for local_node in range(selected_facets.shape[1]):
                    np.add.at(
                        unit_nodal_force,
                        selected_facets[:, local_node],
                        unit_values,
                    )
                geometry["unit_pressure_nodal_force"] = unit_nodal_force
            prepared.append(geometry)
        self._prepared_geometry = {key: prepared}
        return prepared

    @staticmethod
    def _facet_geometry(mesh, facets, owners):
        coordinates = mesh.rest_shape[facets]
        centroids = np.mean(coordinates, axis=1)
        cell_centroids = np.mean(mesh.rest_shape[mesh.cells[owners]], axis=1)
        if facets.shape[1] == 3:
            area_vectors = 0.5 * np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0])
        else:
            area_vectors = 0.5 * (
                np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0])
                + np.cross(coordinates[:, 2] - coordinates[:, 0], coordinates[:, 3] - coordinates[:, 0])
            )
        inward = np.einsum("ij,ij->i", area_vectors, centroids - cell_centroids) < 0.0
        area_vectors[inward] *= -1.0
        area = np.linalg.norm(area_vectors, axis=1)
        normal = area_vectors / area[:, None]
        return centroids, normal, area

    @staticmethod
    def _membrane_geometry(mesh):
        facets = mesh.cells
        coordinates = mesh.rest_shape[facets]
        centroids = np.mean(coordinates, axis=1)
        area_vectors = 0.5 * np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0])
        area = np.linalg.norm(area_vectors, axis=1)
        return facets, centroids, area_vectors / area[:, None], area

    @staticmethod
    def _axisymmetric_edge_geometry(mesh, axis_offset):
        facets, owners = mesh.boundary_facets()
        coordinates = mesh.rest_shape[facets]
        centroids = np.mean(coordinates, axis=1)
        cell_centroids = np.mean(mesh.rest_shape[mesh.cells[owners]], axis=1)
        tangent = coordinates[:, 1, :2] - coordinates[:, 0, :2]
        length = np.linalg.norm(tangent, axis=1)
        normals = np.zeros((facets.shape[0], 3), dtype=np.float64)
        normals[:, 0] = tangent[:, 1]
        normals[:, 1] = -tangent[:, 0]
        normals /= length[:, None]
        inward = np.einsum("ij,ij->i", normals, centroids - cell_centroids) < 0.0
        normals[inward] *= -1.0
        radius = centroids[:, 0] - float(axis_offset)
        if np.any(radius <= 0.0):
            raise ValueError("axisymmetric boundary quadrature requires radius > axis_offset")
        measure = 2.0 * np.pi * radius * length
        return facets, centroids, normals, measure

    def force(self, mesh, time=0.0, axisymmetric=False, axis_offset=0.0):
        force = np.zeros((mesh.number_of_nodes, 3), dtype=np.float64)
        prepared = self._prepare_geometry(mesh, axisymmetric, axis_offset)
        for entry, geometry in zip(self.entries, prepared):
            if entry.kind == "nodal":
                node_ids = geometry["node_ids"]
                values = _evaluate(entry.value, time, geometry["coordinates"])
                if values.ndim == 1 and values.size == 3:
                    values = np.repeat(values[None, :], node_ids.size, axis=0)
                if values.shape != (node_ids.size, 3):
                    raise ValueError("nodal force must be a 3-vector or one 3-vector per node")
                if entry.total and node_ids.size:
                    values = values / node_ids.size
                np.add.at(force, node_ids, values)
                continue
            selected_facets = geometry["facets"]
            selected_centroids = geometry["centroids"]
            selected_normals = geometry["normals"]
            selected_measure = geometry["measure"]
            evaluated = _evaluate(entry.value, time, selected_centroids)
            if entry.pressure and evaluated.ndim == 0:
                force += float(evaluated) * geometry["unit_pressure_nodal_force"]
                continue
            values = self._vectors_from_evaluated(
                evaluated,
                selected_centroids,
                selected_normals,
                entry.pressure,
            )
            nodal_values = values * (selected_measure / selected_facets.shape[1])[:, None]
            for local_node in range(selected_facets.shape[1]):
                np.add.at(force, selected_facets[:, local_node], nodal_values)
        return force


__all__ = ["DirichletBoundary", "NeumannBoundary"]
