"""Optional SciPy factorization recovery must not suppress unrelated failures."""

import pytest
from scipy.sparse import csc_matrix
import scipy.sparse.linalg

from src.mpm.engines.direct.StaticTwoPhaseULMPM import StaticTwoPhaseULMPM

pytestmark = [pytest.mark.unit, pytest.mark.mpm, pytest.mark.cpu]


def test_singular_ilu_can_use_unpreconditioned_solver():
    solver = StaticTwoPhaseULMPM.__new__(StaticTwoPhaseULMPM)
    solver.iterative_preconditioner = "ilu"
    assert solver.build_preconditioner(csc_matrix((2, 2))) is None


@pytest.mark.parametrize("error", [MemoryError, ValueError])
def test_unrelated_factorization_errors_propagate(monkeypatch, error):
    solver = StaticTwoPhaseULMPM.__new__(StaticTwoPhaseULMPM)
    solver.iterative_preconditioner = "ilu"

    def fail(_matrix):
        raise error("invalid factorization input or allocation")

    monkeypatch.setattr(scipy.sparse.linalg, "spilu", fail)
    with pytest.raises(error):
        solver.build_preconditioner(csc_matrix((2, 2)))
