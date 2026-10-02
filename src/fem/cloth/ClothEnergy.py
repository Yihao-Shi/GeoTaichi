"""Preprocessing for optional cloth energies.

The resulting arrays are immutable device-input data.  Energy, gradient and
Hessian evaluation is implemented by :mod:`src.fem.cloth.ClothAssembler`.
"""

from dataclasses import dataclass

import numpy as np

from src.physics_model.consititutive_model.finite_strain.Cloth import (
    normalize_cloth_bending_model,
)


def normalize_bending_model(value):
    return normalize_cloth_bending_model(value)


def _resolve_nodes(mesh, selection):
    if selection is None or (isinstance(selection, str) and selection.strip().lower() == "all"):
        nodes = np.arange(mesh.number_of_nodes, dtype=np.int32)
    elif isinstance(selection, str):
        if selection not in mesh.node_sets:
            raise KeyError(f"unknown FEM node set {selection!r}")
        nodes = np.asarray(mesh.node_sets[selection], dtype=np.int32).reshape(-1)
    else:
        nodes = np.asarray(selection, dtype=np.int32).reshape(-1)
    if np.any(nodes < 0) or np.any(nodes >= mesh.number_of_nodes):
        raise ValueError("cloth energy contains an out-of-range node index")
    return np.ascontiguousarray(nodes)


def _broadcast_vectors(value, count, name):
    values = np.asarray(value, dtype=np.float64)
    if values.shape == (3,):
        values = np.broadcast_to(values, (count, 3)).copy()
    if values.shape != (count, 3):
        raise ValueError(f"{name} must have shape (3,) or ({count}, 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    return np.ascontiguousarray(values)


def _broadcast_scalars(value, count, name, *, positive=True):
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(count, float(values), dtype=np.float64)
    values = values.reshape(-1)
    if values.size != count:
        raise ValueError(f"{name} must be scalar or contain {count} values")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"{name} must be positive")
    return np.ascontiguousarray(values)


def lumped_surface_area(mesh):
    area = np.zeros(mesh.number_of_nodes, dtype=np.float64)
    for triangle in np.asarray(mesh.cells, dtype=np.int32):
        points = mesh.rest_shape[triangle]
        value = np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) / 6.0
        area[triangle] += value
    return area


@dataclass(frozen=True)
class ClothEnergyData:
    stitch_nodes: np.ndarray
    stitch_ratio: np.ndarray
    stitch_stiffness: np.ndarray
    spring_nodes: np.ndarray
    spring_target: np.ndarray
    spring_stiffness: np.ndarray
    sdf_nodes: np.ndarray
    sdf_target: np.ndarray
    sdf_normal: np.ndarray
    sdf_stiffness: np.ndarray
    sdf_dhat: np.ndarray
    node_area: np.ndarray


def prepare_cloth_energies(mesh, specifications):
    stitch_nodes = []
    stitch_ratio = []
    stitch_stiffness = []
    spring_nodes = []
    spring_target = []
    spring_stiffness = []
    sdf_nodes = []
    sdf_target = []
    sdf_normal = []
    sdf_stiffness = []
    sdf_dhat = []

    for specification in specifications or ():
        values = dict(specification)
        kind = str(values.pop("type", values.pop("energy", "")))
        key = kind.strip().replace("_", "").replace("-", "").lower()
        if key in ("stitch", "garmentstitch"):
            stencil_values = values.pop("stitches", values.pop("stencils", None))
            if stencil_values is None:
                raise ValueError("garment stitch requires a stitches array")
            stencils = np.asarray(stencil_values, dtype=np.int32)
            if stencils.size == 0:
                continue
            stencils = stencils.reshape(-1, 3)
            if np.any(stencils < 0) or np.any(stencils >= mesh.number_of_nodes):
                raise ValueError("garment stitch contains an out-of-range node")
            count = stencils.shape[0]
            ratio = values.pop("ratios", values.pop("ratio", None))
            if ratio is None:
                x = mesh.rest_shape[stencils[:, 0]]
                y0 = mesh.rest_shape[stencils[:, 1]]
                edge = mesh.rest_shape[stencils[:, 2]] - y0
                denominator = np.einsum("ij,ij->i", edge, edge)
                if np.any(denominator <= 1.0e-28):
                    raise ValueError("garment stitch contains a zero-length edge")
                ratio = np.einsum("ij,ij->i", x - y0, edge) / denominator
            ratio = _broadcast_scalars(ratio, count, "stitch ratio", positive=False)
            stiffness = _broadcast_scalars(values.pop("stiffness"), count, "stitch stiffness")
            if values:
                raise TypeError(f"unexpected stitch parameters: {', '.join(values)}")
            stitch_nodes.append(np.ascontiguousarray(stencils))
            stitch_ratio.append(ratio)
            stitch_stiffness.append(stiffness)
        elif key in ("spring", "nodespring", "targetspring"):
            nodes = _resolve_nodes(mesh, values.pop("nodes", values.pop("node", None)))
            count = nodes.size
            target = values.pop("targets", values.pop("target", None))
            if target is None:
                target = mesh.rest_shape[nodes]
            target = _broadcast_vectors(target, count, "spring target")
            stiffness = _broadcast_scalars(values.pop("stiffness"), count, "spring stiffness")
            if values:
                raise TypeError(f"unexpected spring parameters: {', '.join(values)}")
            spring_nodes.append(nodes)
            spring_target.append(target)
            spring_stiffness.append(stiffness)
        elif key in ("sdf", "sdfspring", "springsdf"):
            nodes = _resolve_nodes(mesh, values.pop("nodes", values.pop("node", None)))
            count = nodes.size
            target = values.pop("targets", values.pop("target", None))
            normal = values.pop("normals", values.pop("normal", None))
            sdf = values.pop("sdf", None)
            if (target is None) != (normal is None):
                raise ValueError("SDF target and normal arrays must be supplied together")
            if target is None:
                if sdf is None:
                    raise ValueError("SDF energy requires target/normal arrays or an sdf object")
                points = np.asarray(mesh.points[nodes], dtype=np.float64)
                distance = np.asarray(sdf(points), dtype=np.float64).reshape(-1)
                if distance.size != count:
                    raise ValueError("sdf evaluation returned the wrong number of values")
                if not hasattr(sdf, "_normal"):
                    raise TypeError("sdf object must provide normal evaluation")
                normal = np.asarray(sdf._normal(points), dtype=np.float64)
                normal = _broadcast_vectors(normal, count, "SDF normal")
                target = points - distance[:, None] * normal
            elif sdf is not None:
                raise ValueError("SDF energy accepts either target/normal arrays or an sdf object, not both")
            target = _broadcast_vectors(target, count, "SDF target")
            normal = _broadcast_vectors(normal, count, "SDF normal")
            lengths = np.linalg.norm(normal, axis=1)
            if np.any(lengths <= 1.0e-14):
                raise ValueError("SDF normals must be nonzero")
            normal = np.ascontiguousarray(normal / lengths[:, None])
            stiffness = _broadcast_scalars(values.pop("stiffness"), count, "SDF stiffness")
            dhat = _broadcast_scalars(
                values.pop("dhat", values.pop("activation_distance", None)),
                count,
                "SDF activation distance",
            )
            if values:
                raise TypeError(f"unexpected SDF parameters: {', '.join(values)}")
            sdf_nodes.append(nodes)
            sdf_target.append(target)
            sdf_normal.append(normal)
            sdf_stiffness.append(stiffness)
            sdf_dhat.append(dhat)
        else:
            raise ValueError(f"unsupported cloth energy type {kind!r}")

    def concatenate(values, shape, dtype):
        if not values:
            return np.empty(shape, dtype=dtype)
        return np.ascontiguousarray(np.concatenate(values, axis=0), dtype=dtype)

    return ClothEnergyData(
        concatenate(stitch_nodes, (0, 3), np.int32),
        concatenate(stitch_ratio, (0,), np.float64),
        concatenate(stitch_stiffness, (0,), np.float64),
        concatenate(spring_nodes, (0,), np.int32),
        concatenate(spring_target, (0, 3), np.float64),
        concatenate(spring_stiffness, (0,), np.float64),
        concatenate(sdf_nodes, (0,), np.int32),
        concatenate(sdf_target, (0, 3), np.float64),
        concatenate(sdf_normal, (0, 3), np.float64),
        concatenate(sdf_stiffness, (0,), np.float64),
        concatenate(sdf_dhat, (0,), np.float64),
        lumped_surface_area(mesh),
    )


__all__ = [
    "ClothEnergyData",
    "lumped_surface_area",
    "normalize_bending_model",
    "prepare_cloth_energies",
]
