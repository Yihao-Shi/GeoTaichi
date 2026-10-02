"""Scale and degeneracy checks for BuildTriplet's cached block Jacobi."""

import numpy as np
import pytest
import taichi as ti


from src.linear_solver.BuildTriplet import BuildTriplet


_TRIPLETS = {}


def _ensure_taichi():
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)


def _packed_symmetric(matrix):
    if matrix.shape == (2, 2):
        return np.asarray(
            [matrix[0, 0], matrix[1, 1], 0.0, matrix[0, 1], 0.0, 0.0],
            dtype=np.float64,
        )
    return np.asarray(
        [
            matrix[0, 0],
            matrix[1, 1],
            matrix[2, 2],
            matrix[0, 1],
            matrix[1, 2],
            matrix[0, 2],
        ],
        dtype=np.float64,
    )


def _cached_inverse(matrix, *, symmetric):
    _ensure_taichi()
    dimension = int(matrix.shape[0])
    key = (dimension, bool(symmetric))
    if key not in _TRIPLETS:
        _TRIPLETS[key] = BuildTriplet(
            dim=dimension,
            max_pairs_num=1,
            max_nonzeros=1,
            max_active_nodes=1,
            symmetric=symmetric,
            device_reduction=False,
        )
    triplet = _TRIPLETS[key]
    packed = (
        _packed_symmetric(matrix)
        if symmetric
        else np.asarray(matrix, dtype=np.float64).reshape(-1)
    )
    triplet.diag.from_numpy(packed.reshape((1, -1)))
    triplet._build_block_jacobi(1)
    return np.asarray(triplet.diag_inverse.to_numpy()[0]).reshape(
        (dimension, dimension)
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray([[4.0, 0.4], [-0.2, 2.5]]),
        np.asarray(
            [[4.0, 0.4, -0.2], [-0.1, 3.0, 0.3], [0.2, -0.4, 2.0]]
        ),
        np.asarray(
            [
                [4.0, 0.4, -0.2, 0.1],
                [-0.1, 3.0, 0.3, -0.2],
                [0.2, -0.4, 2.5, 0.35],
                [0.1, 0.2, -0.15, 1.8],
            ]
        ),
    ],
    ids=["dense2", "dense3", "dense4"],
)
def test_dense_block_jacobi_is_scale_invariant(matrix):
    expected = np.linalg.inv(matrix)
    for scale in (1.0e-12, 1.0, 1.0e12):
        inverse = _cached_inverse(scale * matrix, symmetric=False)
        assert np.isfinite(inverse).all()
        assert np.allclose(
            scale * inverse, expected, rtol=2.0e-12, atol=2.0e-12
        )


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray([[3.0, 0.35], [0.35, 1.75]]),
        np.asarray(
            [[3.0, 0.35, -0.15], [0.35, 2.25, 0.2], [-0.15, 0.2, 1.5]]
        ),
    ],
    ids=["symmetric2", "symmetric3"],
)
def test_symmetric_block_jacobi_is_scale_invariant(matrix):
    expected = np.linalg.inv(matrix)
    for scale in (1.0e-12, 1.0, 1.0e12):
        inverse = _cached_inverse(scale * matrix, symmetric=True)
        assert np.isfinite(inverse).all()
        assert np.allclose(
            scale * inverse, expected, rtol=2.0e-12, atol=2.0e-12
        )
        assert np.allclose(inverse, inverse.T, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dimension", [2, 3, 4])
@pytest.mark.parametrize("kind", ["zero", "rank_one", "nearly_singular"])
def test_dense_block_jacobi_pathological_fallback_is_finite(dimension, kind):
    if kind == "zero":
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
    elif kind == "rank_one":
        vector = np.arange(1.0, dimension + 1.0)
        matrix = np.outer(vector, vector)
    else:
        diagonal = np.ones(dimension, dtype=np.float64)
        diagonal[-1] = 1.0e-18
        matrix = np.diag(diagonal)

    reference = None
    for scale in (1.0e-12, 1.0, 1.0e12):
        inverse = _cached_inverse(scale * matrix, symmetric=False)
        assert np.isfinite(inverse).all()
        assert np.linalg.norm(inverse, ord=np.inf) > 0.0
        # A zero block has no physical scale; its identity fallback is meant to
        # keep the Krylov residual honest, not to approximate an inverse.
        if kind != "zero":
            scaled = scale * inverse
            if reference is None:
                reference = scaled
            else:
                assert np.allclose(
                    scaled, reference, rtol=1.0e-12, atol=1.0e-12
                )


@pytest.mark.parametrize("dimension", [2, 3])
def test_symmetric_block_jacobi_pathological_fallback_is_finite(dimension):
    vector = np.arange(1.0, dimension + 1.0)
    matrix = np.outer(vector, vector)
    for scale in (1.0e-12, 1.0, 1.0e12):
        inverse = _cached_inverse(scale * matrix, symmetric=True)
        assert np.isfinite(inverse).all()
        assert np.linalg.norm(inverse, ord=np.inf) > 0.0
        assert np.allclose(inverse, inverse.T, rtol=0.0, atol=0.0)


def test_tiny_spd_block_pcg_does_not_false_converge():
    _ensure_taichi()
    scale = 1.0e-12
    block = scale * np.asarray(
        [[3.0, 0.25, -0.1], [0.25, 2.0, 0.15], [-0.1, 0.15, 1.5]]
    )
    expected = np.asarray([0.75, -1.25, 0.5])
    rhs = block @ expected
    triplet = BuildTriplet(
        dim=3,
        max_pairs_num=1,
        max_nonzeros=1,
        max_active_nodes=1,
        symmetric=True,
        solver="PCG",
        device_reduction=False,
    )
    triplet.diag.from_numpy(_packed_symmetric(block).reshape((1, -1)))
    result = triplet.solve(
        rhs=rhs.reshape((1, 3)),
        active_nodes=1,
        tol=1.0e-14,
        maxiter=8,
        return_solution=True,
    )
    assert result["converged"]
    assert np.allclose(result["x"][0], expected, rtol=1.0e-11, atol=1.0e-11)
