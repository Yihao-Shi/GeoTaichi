"""Independent NumPy cloth responses for numerical test oracles only."""

import numpy as np
from src.utils.errors import InvertedElementError


def _surface_jacobian(deformation_gradient, minimum_jacobian):
    metric = deformation_gradient.T @ deformation_gradient
    determinant = float(np.linalg.det(metric))
    if determinant <= float(minimum_jacobian) ** 2:
        raise InvertedElementError("cloth triangle is inverted or collapsed")
    return np.sqrt(determinant), metric


def _spectral_arap_response(deformation_gradient, stretch_stiffness, compression_stiffness):
    """Return ARAP energy, PK1 and its analytic Frechet derivative."""
    metric = deformation_gradient.T @ deformation_gradient
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    eigenvalues = np.maximum(eigenvalues, np.finfo(np.float64).tiny)
    stretches = np.sqrt(eigenvalues)
    stiffnesses = np.where(stretches <= 1.0, compression_stiffness, stretch_stiffness)
    offsets = stretches - 1.0
    energy = float(np.dot(stiffnesses, offsets * offsets))
    responses = 2.0 * stiffnesses * offsets / stretches
    response_derivatives = stiffnesses / (stretches**3)
    response_matrix = (eigenvectors * responses[None, :]) @ eigenvectors.T
    first_piola = deformation_gradient @ response_matrix

    tangent = np.empty((3, 2, 3, 2), dtype=np.float64)
    eigenvalue_gap = eigenvalues[0] - eigenvalues[1]
    if abs(eigenvalue_gap) > 1.0e-12 * max(float(eigenvalues[-1]), 1.0):
        divided_difference = (responses[0] - responses[1]) / eigenvalue_gap
    else:
        divided_difference = 0.5 * np.sum(response_derivatives)
    for spatial_column in range(3):
        for material_column in range(2):
            increment = np.zeros((3, 2), dtype=np.float64)
            increment[spatial_column, material_column] = 1.0
            metric_increment = increment.T @ deformation_gradient + deformation_gradient.T @ increment
            local_increment = eigenvectors.T @ metric_increment @ eigenvectors
            response_increment = np.empty((2, 2), dtype=np.float64)
            response_increment[0, 0] = response_derivatives[0] * local_increment[0, 0]
            response_increment[1, 1] = response_derivatives[1] * local_increment[1, 1]
            response_increment[0, 1] = divided_difference * local_increment[0, 1]
            response_increment[1, 0] = divided_difference * local_increment[1, 0]
            tangent[:, :, spatial_column, material_column] = (
                increment @ response_matrix + deformation_gradient @ eigenvectors @ response_increment @ eigenvectors.T
            )
    return energy, first_piola, tangent


def arap_response(material, deformation_gradient, need_tangent=True):
    deformation_gradient = np.asarray(deformation_gradient, dtype=np.float64)
    if deformation_gradient.shape != (3, 2):
        raise ValueError("ClothARAP requires a TRI3 surface deformation gradient " "with shape (3, 2)")
    _surface_jacobian(deformation_gradient, material.minimum_jacobian)
    energy, first_piola, tangent = _spectral_arap_response(
        deformation_gradient,
        material.stretch_stiffness,
        material.compression_stiffness,
    )
    return energy, first_piola, tangent if need_tangent else None


def neo_hookean_response(material, deformation_gradient, need_tangent=True):
    deformation_gradient = np.asarray(deformation_gradient, dtype=np.float64)
    if deformation_gradient.shape != (3, 2):
        raise ValueError("ClothNeoHookean requires a TRI3 surface deformation " "gradient with shape (3, 2)")
    jacobian, metric = _surface_jacobian(deformation_gradient, material.minimum_jacobian)
    log_jacobian = np.log(jacobian)
    inverse_metric = np.linalg.inv(metric)
    energy = (
        0.5 * material.mu_ * (np.trace(metric) - 2.0 - 2.0 * log_jacobian) + 0.5 * material.lambda_ * log_jacobian**2
    )
    metric_gradient = deformation_gradient @ inverse_metric
    first_piola = (
        material.mu_ * deformation_gradient + (-material.mu_ + material.lambda_ * log_jacobian) * metric_gradient
    )
    tangent = None
    if need_tangent:
        projection = metric_gradient @ deformation_gradient.T
        tangent = np.empty((3, 2, 3, 2), dtype=np.float64)
        coefficient = -material.mu_ + material.lambda_ * log_jacobian
        for i in range(3):
            for material_i in range(2):
                for k in range(3):
                    for material_k in range(2):
                        tangent[i, material_i, k, material_k] = (
                            material.mu_ * float(i == k and material_i == material_k)
                            + material.lambda_ * metric_gradient[i, material_i] * metric_gradient[k, material_k]
                            + coefficient
                            * (
                                (float(i == k) - projection[i, k]) * inverse_metric[material_k, material_i]
                                - metric_gradient[i, material_k] * metric_gradient[k, material_i]
                            )
                        )
    return float(energy), first_piola, tangent


def cloth_response(material, deformation_gradient, need_tangent=True):
    response = arap_response if material.cloth_model == "arap" else neo_hookean_response
    return response(material, deformation_gradient, need_tangent)
