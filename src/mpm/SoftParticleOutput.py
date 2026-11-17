from __future__ import annotations

import numpy as np


def stress_to_tensor3(stress):
    stress = np.asarray(stress)
    if stress.ndim == 3 and stress.shape[1:] == (3, 3):
        return stress.astype(np.float64, copy=False)
    if stress.ndim == 2 and stress.shape[1] == 9:
        return stress.reshape((-1, 3, 3)).astype(np.float64, copy=False)
    if stress.ndim == 2 and stress.shape[1] == 6:
        tensor = np.zeros((stress.shape[0], 3, 3), dtype=np.float64)
        tensor[:, 0, 0] = stress[:, 0]
        tensor[:, 1, 1] = stress[:, 1]
        tensor[:, 2, 2] = stress[:, 2]
        tensor[:, 0, 1] = tensor[:, 1, 0] = stress[:, 3]
        tensor[:, 1, 2] = tensor[:, 2, 1] = stress[:, 4]
        tensor[:, 0, 2] = tensor[:, 2, 0] = stress[:, 5]
        return tensor
    raise ValueError(f"Unsupported stress array shape: {stress.shape}")


def pk1_to_cauchy(first_piola, deformation_gradient):
    first_piola = stress_to_tensor3(first_piola)
    deformation_gradient = np.asarray(deformation_gradient, dtype=np.float64)
    if deformation_gradient.shape != first_piola.shape:
        raise ValueError(
            "First Piola stress and deformation gradient must have matching "
            f"(n, 3, 3) shapes, got {first_piola.shape} and "
            f"{deformation_gradient.shape}"
        )
    jacobian = np.linalg.det(deformation_gradient)
    numerator = np.einsum("nij,nkj->nik", first_piola, deformation_gradient)
    cauchy = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = np.abs(jacobian) > np.finfo(np.float64).eps
    np.divide(
        numerator,
        jacobian[:, None, None],
        out=cauchy,
        where=valid[:, None, None],
    )
    return np.ascontiguousarray(cauchy), np.ascontiguousarray(jacobian)


def von_mises_stress(cauchy_stress):
    tensor = stress_to_tensor3(cauchy_stress)
    symmetric = 0.5 * (tensor + np.swapaxes(tensor, 1, 2))
    mean = np.trace(symmetric, axis1=1, axis2=2) / 3.0
    deviator = symmetric.copy()
    deviator[:, 0, 0] -= mean
    deviator[:, 1, 1] -= mean
    deviator[:, 2, 2] -= mean
    j2 = 0.5 * np.einsum("nij,nij->n", deviator, deviator)
    return np.ascontiguousarray(np.sqrt(np.maximum(3.0 * j2, 0.0)))


def vtk_vector(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] not in (2, 3):
        raise ValueError(f"VTK vector data must have shape (n, 2) or (n, 3), got {values.shape}")
    components = [np.ascontiguousarray(values[:, axis]) for axis in range(values.shape[1])]
    if values.shape[1] == 2:
        components.append(np.zeros(values.shape[0], dtype=values.dtype))
    return tuple(components)
