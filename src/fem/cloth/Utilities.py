from __future__ import annotations
from src.utils.linalg import _compute_cotangent, _signed_dihedral_angle
from src.fem.cloth.ClothEnergy import ClothEnergyData
import numpy as np
from src.fem.cloth.ClothElement import ClothElement


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


def _build_bending_operator(mesh, bending_modulus):
    if bending_modulus <= 0.0:
        return (
            np.empty((0, 4), dtype=np.int32),
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    half_edges = {}
    connectivity = []
    stiffness = []
    edge_length = []
    height = []
    rest_angle = []
    reference = mesh.rest_shape
    for triangle in mesh.cells:
        for local_id in range(3):
            first = int(triangle[local_id])
            second = int(triangle[(local_id + 1) % 3])
            opposite = int(triangle[(local_id + 2) % 3])
            reverse = (second, first)
            if reverse not in half_edges:
                half_edges[(first, second)] = opposite
                continue
            other = half_edges.pop(reverse)
            bending_nodes = np.asarray((opposite, first, second, other), dtype=np.int32)
            x0, x1, x2, x3 = reference[bending_nodes]
            edge0 = x2 - x1
            edge1 = x0 - x1
            edge2 = x3 - x1
            edge3 = x0 - x2
            edge4 = x3 - x2
            area0 = 0.5 * np.linalg.norm(np.cross(edge0, edge1))
            area1 = 0.5 * np.linalg.norm(np.cross(edge0, edge2))
            if area0 <= 0.0 or area1 <= 0.0:
                raise ValueError("cloth bending element contains a zero-area reference triangle")
            c01 = _compute_cotangent(edge0, edge1)
            c02 = _compute_cotangent(edge0, edge2)
            c03 = _compute_cotangent(-edge0, edge3)
            c04 = _compute_cotangent(-edge0, edge4)
            coefficients = np.asarray(
                (-c01 - c03, c03 + c04, c01 + c02, -c02 - c04),
                dtype=np.float64,
            )
            factor = bending_modulus * 3.0 / (2.0 * (area0 + area1))
            connectivity.append(bending_nodes)
            stiffness.append(factor * np.outer(coefficients, coefficients))
            hinge_length = np.linalg.norm(edge0)
            edge_length.append(hinge_length)
            height.append((2.0 * area0 + 2.0 * area1) / (hinge_length * 6.0))
            rest_angle.append(_signed_dihedral_angle((x0, x1, x2, x3)))
    if not connectivity:
        return (
            np.empty((0, 4), dtype=np.int32),
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.ascontiguousarray(connectivity),
        np.ascontiguousarray(stiffness),
        np.ascontiguousarray(edge_length),
        np.ascontiguousarray(height),
        np.ascontiguousarray(rest_angle),
    )


def create_cloth_element(mesh, material):
    """Construct membrane integration data and both bending operators."""
    if mesh.cell_type != "triangle":
        raise ValueError("cloth FEM requires a TRI3 surface mesh")
    count = mesh.number_of_cells
    inverse_jacobian = np.empty((count, 2, 2), dtype=np.float64)
    inverse_metric = np.empty((count, 2, 2), dtype=np.float64)
    weights = np.empty(count, dtype=np.float64)
    lengths = np.empty(count, dtype=np.float64)
    for element_id, connectivity in enumerate(mesh.cells):
        material_triangle = mesh.material_points[mesh.material_cells[element_id]]
        edge01 = material_triangle[1] - material_triangle[0]
        edge02 = material_triangle[2] - material_triangle[0]
        length01 = np.linalg.norm(edge01)
        double_area = np.linalg.norm(np.cross(edge01, edge02))
        scale = max(length01, np.linalg.norm(edge02), 1.0)
        if length01 <= 1.0e-13 * scale or double_area <= 1.0e-13 * scale**2:
            raise ValueError(f"cloth triangle {element_id} has degenerate material coordinates")
        metric = np.asarray(
            (
                (np.dot(edge01, edge01), np.dot(edge01, edge02)),
                (np.dot(edge01, edge02), np.dot(edge02, edge02)),
            )
        )
        edge_matrix = np.asarray(
            (
                (length01, np.dot(edge01, edge02) / length01),
                (0.0, double_area / length01),
            )
        )
        inverse_jacobian[element_id] = np.linalg.inv(edge_matrix)
        inverse_metric[element_id] = np.linalg.inv(metric)
        weights[element_id] = 0.5 * double_area * material.thickness
        lengths[element_id] = max(length01, np.linalg.norm(edge02), np.linalg.norm(edge02 - edge01))
    (
        bending_connectivity,
        bending_stiffness,
        bending_edge_length,
        bending_height,
        bending_rest_angle,
    ) = _build_bending_operator(mesh, material.quadratic_bending_modulus)
    return ClothElement(
        np.ascontiguousarray(mesh.cells, dtype=np.int32),
        np.ascontiguousarray(inverse_jacobian),
        np.ascontiguousarray(inverse_metric),
        np.ascontiguousarray(weights),
        np.ascontiguousarray(lengths),
        bending_connectivity,
        bending_stiffness,
        bending_edge_length,
        bending_height,
        bending_rest_angle,
    )
