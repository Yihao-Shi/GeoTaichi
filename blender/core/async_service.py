"""Single-flight background execution for Blender-facing job I/O."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class OperationResult:
    name: str
    value: Any = None
    error: Optional[BaseException] = None


class SingleFlightService:
    """Run at most one blocking operation away from Blender's main thread."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="geotaichi-blender")
        self._future: Optional[Future] = None
        self._name = ""
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._future is not None

    def submit(self, name: str, function: Callable[[], Any]) -> bool:
        with self._lock:
            if self._future is not None:
                return False
            self._name = name
            self._future = self._executor.submit(function)
            return True

    def poll(self) -> Optional[OperationResult]:
        with self._lock:
            future = self._future
            if future is None or not future.done():
                return None
            name = self._name
            self._future = None
            self._name = ""
        try:
            return OperationResult(name=name, value=future.result())
        except BaseException as exc:
            return OperationResult(name=name, error=exc)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["OperationResult", "SingleFlightService"]
