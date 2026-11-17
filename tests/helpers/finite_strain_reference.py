"""Independent finite-strain NumPy oracles; never imported by production code."""

import numpy as np
from src.utils.errors import InvertedElementError


def neo_hookean_response(model, deformation_gradient, need_tangent=True):
    deformation_gradient = np.asarray(deformation_gradient, dtype=np.float64)
    if deformation_gradient.ndim != 2 or (deformation_gradient.shape[0] != deformation_gradient.shape[1]):
        raise ValueError("NeoHookeanModel requires a square volume deformation gradient")
    dimension = deformation_gradient.shape[0]
    jacobian = float(np.linalg.det(deformation_gradient))
    if jacobian <= model.minimum_jacobian:
        raise InvertedElementError(f"Neo-Hookean deformation Jacobian {jacobian:.3e} is not positive")
    lame_lambda, shear = model.lame_parameters
    log_jacobian = np.log(jacobian)
    inverse_transpose = np.linalg.inv(deformation_gradient).T
    first_piola = shear * (deformation_gradient - inverse_transpose) + lame_lambda * log_jacobian * inverse_transpose
    energy = (
        0.5 * shear * (np.sum(deformation_gradient * deformation_gradient) - dimension)
        - shear * log_jacobian
        + 0.5 * lame_lambda * log_jacobian**2
    )
    tangent = None
    if need_tangent:
        tangent = np.empty(
            (dimension, dimension, dimension, dimension),
            dtype=np.float64,
        )
        for i in range(dimension):
            for j in range(dimension):
                for k in range(dimension):
                    for ell in range(dimension):
                        tangent[i, j, k, ell] = (
                            shear * (1.0 if i == k and j == ell else 0.0)
                            + (shear - lame_lambda * log_jacobian) * inverse_transpose[i, ell] * inverse_transpose[k, j]
                            + lame_lambda * inverse_transpose[i, j] * inverse_transpose[k, ell]
                        )
    return float(energy), first_piola, tangent


def stvk_response(model, deformation_gradient, need_tangent=True):
    deformation_gradient = np.asarray(deformation_gradient, dtype=np.float64)
    if deformation_gradient.ndim != 2:
        raise ValueError("deformation_gradient must be a matrix")
    spatial_dimension, material_dimension = deformation_gradient.shape
    lame_lambda, shear = model.lame_parameters(material_dimension)
    identity = np.eye(material_dimension)
    strain = 0.5 * (deformation_gradient.T @ deformation_gradient - identity)
    second_piola = lame_lambda * np.trace(strain) * identity + 2.0 * shear * strain
    first_piola = deformation_gradient @ second_piola
    energy = 0.5 * lame_lambda * np.trace(strain) ** 2 + shear * np.sum(strain * strain)
    tangent = None
    if need_tangent:
        left_cauchy = deformation_gradient @ deformation_gradient.T
        tangent = np.empty(
            (
                spatial_dimension,
                material_dimension,
                spatial_dimension,
                material_dimension,
            ),
            dtype=np.float64,
        )
        for i in range(spatial_dimension):
            for j in range(material_dimension):
                for k in range(spatial_dimension):
                    for ell in range(material_dimension):
                        tangent[i, j, k, ell] = (
                            (1.0 if i == k else 0.0) * second_piola[ell, j]
                            + lame_lambda * deformation_gradient[i, j] * deformation_gradient[k, ell]
                            + shear * deformation_gradient[i, ell] * deformation_gradient[k, j]
                            + shear * (1.0 if j == ell else 0.0) * left_cauchy[i, k]
                        )
    return float(energy), first_piola, tangent


def material_response(model, deformation_gradient, need_tangent=True):
    response = stvk_response if type(model).__name__ == "StVenantKirchhoffModel" else neo_hookean_response
    return response(model, deformation_gradient, need_tangent)


def make_device_evaluator(model, shape):
    import taichi as ti

    spatial, material = shape
    energy = ti.field(ti.f64, shape=())
    piola = ti.Matrix.field(spatial, material, ti.f64, shape=())
    tangent = ti.Matrix.field(spatial * material, spatial * material, ti.f64, shape=())

    @ti.kernel
    def evaluate_kernel(values: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        deformation = ti.Matrix.zero(ti.f64, spatial, material)
        for i, j in ti.static(ti.ndrange(spatial, material)):
            deformation[i, j] = values[i, j]
        if ti.static(spatial != material):
            energy[None] = model.surface_strain_energy_density(deformation)
            piola[None] = model.surface_first_piola_stress(deformation)
            tangent[None] = model.surface_first_piola_tangent(deformation)
        else:
            energy[None] = model.strain_energy_density(deformation)
            piola[None] = model.first_piola_stress(deformation)
            tangent[None] = model.first_piola_tangent(deformation)

    def evaluate(values, need_tangent=True):
        evaluate_kernel(np.ascontiguousarray(values, dtype=np.float64))
        return (
            float(energy[None]),
            piola.to_numpy()[()],
            (
                tangent.to_numpy()[()].reshape(material, spatial, material, spatial).transpose(1, 0, 3, 2)
                if need_tangent
                else None
            ),
        )

    return evaluate
