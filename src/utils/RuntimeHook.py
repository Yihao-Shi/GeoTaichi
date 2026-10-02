"""Optional runtime callbacks used by external orchestration layers.

The numerical solvers call :func:`runtime_checkpoint` only after a complete
step.  In an ordinary GeoTaichi process the function is a very small no-op.
The MCP task worker installs a callback that drains live execution requests,
without making the solver depend on FastMCP or on the MCP package itself.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional


RuntimeCheckpoint = Callable[[], int]

_lock = threading.RLock()
_checkpoint: Optional[RuntimeCheckpoint] = None


def set_runtime_checkpoint(callback: RuntimeCheckpoint) -> None:
    """Install the process-wide callback invoked at solver safe points."""
    if not callable(callback):
        raise TypeError("runtime checkpoint must be callable")
    global _checkpoint
    with _lock:
        _checkpoint = callback


def clear_runtime_checkpoint(callback: Optional[RuntimeCheckpoint] = None) -> None:
    """Clear the callback, optionally only when it matches *callback*."""
    global _checkpoint
    with _lock:
        if callback is None or _checkpoint is callback:
            _checkpoint = None


def runtime_checkpoint() -> int:
    """Run the installed callback and return the number of handled requests."""
    with _lock:
        callback = _checkpoint
    if callback is None:
        return 0
    return callback()
