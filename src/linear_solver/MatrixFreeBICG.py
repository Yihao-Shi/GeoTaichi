"""Backward-compatible name for the matrix-free BiCGSTAB implementation."""

from src.linear_solver.MatrixFreeBICGSTAB import MatrixFreeBICGSTAB


class MatrixFreeBICG(MatrixFreeBICGSTAB):
    pass
