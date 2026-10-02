import numpy as np
import pytest

from src.physics_model.contact_model.ipc.LevelSetAffine import (
    AdjointIPCNonpenetration,
    AffineLevelSetBody,
    AffineLevelSetIPC,
    TrilinearLevelSet,
    affine_basis,
    affine_levelset_gap,
    continued_ipc_barrier_distance_terms,
    point_affine_levelset_gap,
)
from src.physics_model.contact_model.ipc.IPC import (
    ipc_barrier_distance_terms_py,
)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.dem,
    pytest.mark.ipc,
    pytest.mark.contact,
    pytest.mark.cpu,
]


def _grid_values(shape, origin, spacing, function):
    axes = [
        origin[d] + spacing * np.arange(shape[d], dtype=np.float64)
        for d in range(3)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    # GeoTaichi's level-set grids use x-fast linearization.
    return function(x, y, z).flatten(order="F")


def _sphere_levelset(radius=0.3, spacing=0.04, node_count=33):
    shape = np.full(3, node_count, dtype=np.int32)
    # Keep symmetry axes away from trilinear cell boundaries so central
    # differences probe one smooth interpolation cell.
    origin = -0.5 * spacing * (shape - 1) - 0.013
    values = _grid_values(
        shape,
        origin,
        spacing,
        lambda x, y, z: np.sqrt(x * x + y * y + z * z) - radius,
    )
    return TrilinearLevelSet(origin, spacing, shape, values)


def _identity_controls(center):
    return np.array(
        [
            center,
            np.asarray(center) + [1.0, 0.0, 0.0],
            np.asarray(center) + [0.0, 1.0, 0.0],
            np.asarray(center) + [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _sphere_quadrature(radius=0.3):
    points = radius * np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    basis = np.asarray([affine_basis(point) for point in points])
    weights = np.full(points.shape[0], 4.0 * np.pi * radius**2 / 6.0)
    return basis, weights


def _two_sphere_contact(center_distance, *, hessian_mode="exact"):
    grid = _sphere_levelset()
    basis, weights = _sphere_quadrature()
    bodies = [
        AffineLevelSetBody(
            _identity_controls([-0.5 * center_distance, 0.0, 0.0]),
            1.0,
            grid,
            basis,
            weights,
        ),
        AffineLevelSetBody(
            _identity_controls([0.5 * center_distance, 0.0, 0.0]),
            1.0,
            grid,
            basis,
            weights,
        ),
    ]
    return AffineLevelSetIPC(
        bodies,
        dhat=0.12,
        kappa=2.5,
        hessian_mode=hessian_mode,
    )


def test_trilinear_levelset_value_gradient_hessian_are_exact():
    shape = np.array([7, 8, 9], dtype=np.int32)
    origin = np.array([-0.6, -0.7, -0.8])
    spacing = 0.2
    coefficients = np.array([0.4, -0.7, 0.2, 0.9, 1.1, -0.3, 0.6, -0.8])

    def polynomial(x, y, z):
        a, bx, by, bz, bxy, bxz, byz, bxyz = coefficients
        return (
            a
            + bx * x
            + by * y
            + bz * z
            + bxy * x * y
            + bxz * x * z
            + byz * y * z
            + bxyz * x * y * z
        )

    grid = TrilinearLevelSet(
        origin,
        spacing,
        shape,
        _grid_values(shape, origin, spacing, polynomial),
    )
    point = np.array([0.13, -0.17, 0.29])
    value, gradient, hessian, inside = grid.sample(point)
    a, bx, by, bz, bxy, bxz, byz, bxyz = coefficients
    expected_gradient = np.array(
        [
            bx + bxy * point[1] + bxz * point[2] + bxyz * point[1] * point[2],
            by + bxy * point[0] + byz * point[2] + bxyz * point[0] * point[2],
            bz + bxz * point[0] + byz * point[1] + bxyz * point[0] * point[1],
        ]
    )
    expected_hessian = np.array(
        [
            [0.0, bxy + bxyz * point[2], bxz + bxyz * point[1]],
            [bxy + bxyz * point[2], 0.0, byz + bxyz * point[0]],
            [bxz + bxyz * point[1], byz + bxyz * point[0], 0.0],
        ]
    )

    assert inside
    assert value == pytest.approx(polynomial(*point), abs=2.0e-14)
    np.testing.assert_allclose(gradient, expected_gradient, atol=2.0e-14)
    np.testing.assert_allclose(hessian, expected_hessian, atol=2.0e-13)


def test_affine_levelset_gap_gradient_and_hessian_match_finite_difference():
    grid = _sphere_levelset()
    source = _identity_controls([0.553, 0.023, 0.017])
    target = _identity_controls([0.0, 0.0, 0.0])
    basis = affine_basis([-0.347, 0.031, 0.019])
    reference = affine_levelset_gap(
        source, target, basis, 1.0, grid
    )
    packed = np.concatenate((source.reshape(-1), target.reshape(-1)))
    eye = np.eye(packed.size)
    epsilon = 2.0e-6

    def evaluate(values):
        return affine_levelset_gap(
            values[:12].reshape(4, 3),
            values[12:].reshape(4, 3),
            basis,
            1.0,
            grid,
        )

    fd_gradient = np.array(
        [
            (
                evaluate(packed + epsilon * direction).gap
                - evaluate(packed - epsilon * direction).gap
            )
            / (2.0 * epsilon)
            for direction in eye
        ]
    )
    fd_hessian = np.column_stack(
        [
            (
                evaluate(packed + epsilon * direction).gradient
                - evaluate(packed - epsilon * direction).gradient
            )
            / (2.0 * epsilon)
            for direction in eye
        ]
    )

    np.testing.assert_allclose(
        reference.gradient, fd_gradient, rtol=2.0e-7, atol=2.0e-9
    )
    np.testing.assert_allclose(
        reference.hessian, fd_hessian, rtol=2.0e-6, atol=3.0e-8
    )
    np.testing.assert_allclose(
        reference.hessian, reference.hessian.T, atol=1.0e-13
    )


def test_point_affine_levelset_gap_derivatives_match_finite_difference():
    grid = _sphere_levelset()
    point = np.array([0.316, -0.071, 0.039])
    target = _identity_controls([0.0, 0.0, 0.0])
    packed = np.concatenate((point, target.reshape(-1)))
    reference = point_affine_levelset_gap(
        point, target, 1.0, grid
    )
    epsilon = 2.0e-6
    eye = np.eye(packed.size)

    def evaluate(values):
        return point_affine_levelset_gap(
            values[:3],
            values[3:].reshape(4, 3),
            1.0,
            grid,
        )

    fd_gradient = np.array(
        [
            (
                evaluate(packed + epsilon * direction).gap
                - evaluate(packed - epsilon * direction).gap
            )
            / (2.0 * epsilon)
            for direction in eye
        ]
    )
    fd_hessian = np.column_stack(
        [
            (
                evaluate(packed + epsilon * direction).gradient
                - evaluate(packed - epsilon * direction).gradient
            )
            / (2.0 * epsilon)
            for direction in eye
        ]
    )

    np.testing.assert_allclose(
        reference.gradient, fd_gradient, rtol=2.0e-7, atol=2.0e-9
    )
    np.testing.assert_allclose(
        reference.hessian, fd_hessian, rtol=2.0e-6, atol=3.0e-8
    )


def test_levelset_ipc_energy_derivatives_match_finite_difference():
    contact = _two_sphere_contact(0.657, hessian_mode="exact")
    packed = contact.pack_controls()
    reference = contact.assemble(packed, need_hessian=True)
    assert reference.feasible
    assert 0.0 < reference.minimum_gap < contact.dhat
    assert reference.active_contacts == 2
    epsilon = 1.0e-6
    eye = np.eye(packed.size)

    fd_gradient = np.array(
        [
            (
                contact.assemble(
                    packed + epsilon * direction, need_hessian=False
                ).energy
                - contact.assemble(
                    packed - epsilon * direction, need_hessian=False
                ).energy
            )
            / (2.0 * epsilon)
            for direction in eye
        ]
    )
    fd_hessian = np.column_stack(
        [
            (
                contact.assemble(
                    packed + epsilon * direction, need_hessian=False
                ).gradient
                - contact.assemble(
                    packed - epsilon * direction, need_hessian=False
                ).gradient
            )
            / (2.0 * epsilon)
            for direction in eye
        ]
    )

    np.testing.assert_allclose(
        reference.gradient, fd_gradient, rtol=2.0e-6, atol=2.0e-8
    )
    np.testing.assert_allclose(
        reference.hessian, fd_hessian, rtol=2.0e-5, atol=2.0e-6
    )


def test_continued_ipc_is_c2_and_finite_for_overlap():
    dhat = 0.12
    delta = 0.02
    kappa = 3.0
    exact = ipc_barrier_distance_terms_py(delta, dhat, kappa=kappa)
    continued = continued_ipc_barrier_distance_terms(
        delta,
        dhat,
        kappa=kappa,
        continuation_distance=delta,
    )
    np.testing.assert_allclose(continued, exact, rtol=0.0, atol=1.0e-14)
    overlap = continued_ipc_barrier_distance_terms(
        -0.2,
        dhat,
        kappa=kappa,
        continuation_distance=delta,
    )
    assert np.isfinite(overlap).all()
    assert overlap[1] < 0.0
    assert overlap[2] > 0.0


def test_adjoint_initializer_pushes_overlap_to_strict_feasibility():
    contact = _two_sphere_contact(0.4, hessian_mode="exact")
    initializer = AdjointIPCNonpenetration(
        contact,
        anchor_stiffness=1.0,
        continuation_ratio=0.15,
        feasibility_tolerance=1.0e-6,
        maximum_stages=8,
    )
    result = initializer.solve(maximum_iterations=300)

    assert result.success, result.message
    assert result.initial_minimum_gap < 0.0
    assert result.minimum_gap >= 1.0e-6
    np.testing.assert_allclose(
        result.translations[0],
        -result.translations[1],
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    strict = contact.assemble(need_hessian=False)
    assert strict.feasible
    assert np.isfinite(strict.energy)

    translations = result.translations.reshape(-1)
    loss_gradient = translations.copy()
    adjoint_gradient = initializer.adjoint_anchor_gradient(loss_gradient)
    epsilon = 1.0e-5
    for anchor_dof in (0, 3):
        losses = []
        for sign in (-1.0, 1.0):
            anchor = np.zeros(6)
            anchor[anchor_dof] = sign * epsilon
            perturbed = initializer.solve(
                anchor=anchor, maximum_iterations=300
            )
            losses.append(
                0.5 * float(np.sum(perturbed.translations**2))
            )
        finite_difference = (losses[1] - losses[0]) / (2.0 * epsilon)
        assert adjoint_gradient[anchor_dof] == pytest.approx(
            finite_difference, rel=2.0e-5, abs=2.0e-7
        )
