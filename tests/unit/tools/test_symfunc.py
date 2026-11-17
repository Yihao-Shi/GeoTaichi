"""Contracts for the optional symbolic derivation helpers."""

import pytest


try:
    import sympy
except ModuleNotFoundError:
    sympy = None

pytestmark = pytest.mark.skipif(
    sympy is None,
    reason="symbolic derivation tools require optional SymPy",
)

if sympy is not None:
    Matrix = sympy.Matrix
    symbols = sympy.symbols
    from tools.derivations.symfunc import (
        dot_prod,
        double_dot,
        make_matrix,
        make_scalar,
        make_vector,
        matrix_matrix_prod,
        matrix_vector_prod,
        outer_prod,
    )


def test_symbol_factories_preserve_requested_shapes_and_names():
    assert str(make_scalar("alpha")) == "alpha"
    assert make_vector("v", 3) == Matrix(
        symbols("v_0 v_1 v_2", real=True)
    )
    assert make_matrix("A", 2, 3).shape == (2, 3)


def test_vector_products_support_rectangular_outer_product():
    left = Matrix([1, 2])
    right = Matrix([3, 4, 5])

    assert outer_prod(left, right) == Matrix([[3, 4, 5], [6, 8, 10]])
    assert dot_prod(left, Matrix([6, 7])) == 20


def test_matrix_products_use_standard_dimension_contracts():
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    vector = Matrix([2, -1, 3])

    assert matrix_vector_prod(matrix, vector) == Matrix([9, 21])
    assert matrix_matrix_prod(matrix, Matrix([[1], [0], [-1]])) == Matrix(
        [-2, -2]
    )


def test_double_dot_matches_frobenius_inner_product():
    left = Matrix([[1, 2], [3, 4]])
    right = Matrix([[5, 6], [7, 8]])

    assert double_dot(left, right) == 70
