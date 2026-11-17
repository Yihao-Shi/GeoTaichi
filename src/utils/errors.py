"""Shared failure types; this module must remain free of solver dependencies."""


class InvertedElementError(RuntimeError):
    """An element map is collapsed, orientation reversing, or nonfinite."""


class RuntimeNotInitializedError(RuntimeError):
    """A device operation requires an initialized Taichi runtime."""


class LinearSolveError(RuntimeError):
    """A linear solve failed convergence or true-residual verification."""


class BlockJacobiError(LinearSolveError):
    """A block Jacobi factorization has an invalid or singular pivot."""


class LineSearchError(RuntimeError):
    """A nonlinear search has no admissible descent step."""


class NewtonConvergenceError(RuntimeError):
    """A FEM or coupled Newton solve failed before accepting a step."""


class IGAConvergenceError(RuntimeError):
    """A recoverable failure occurred before accepting an implicit IGA step."""


class MPMConvergenceError(RuntimeError):
    """A recoverable failure occurred before accepting a Direct MPM step."""


class IPCInfeasibleContactState(RuntimeError):
    """A trial configuration crossed the IPC minimum-distance boundary."""


class CapacityError(RuntimeError):
    """A configured field, buffer, contact, or sparse capacity was exceeded."""


class ConfigurationError(RuntimeError):
    """The selected solver, model, or option combination is unsupported."""


class SimulationStateError(RuntimeError):
    """An operation requires a different initialized or accepted simulation state."""


class GeometryError(RuntimeError):
    """Input or current geometry is invalid for the requested operation."""


class NonfiniteStateError(RuntimeError):
    """A numerical state, residual, or correction is not finite."""


class NonlinearSolveError(RuntimeError):
    """A nonlinear solve failed its convergence contract."""


class SparseAssemblyError(RuntimeError):
    """Sparse assembly or a fixed sparsity pattern is inconsistent."""


class BackendError(RuntimeError):
    """The selected numerical backend or required device feature is unavailable."""


class RecordingError(RuntimeError):
    """A trajectory, recorder, or replay lifecycle is inconsistent."""
