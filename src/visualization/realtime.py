"""Real-time visualization for small GeoTaichi simulations.

Numerical models remain Taichi device programs.  Visualization uses -- PyRender, Pyglet, and PyOpenGL -- and deliberately
does not import or create a Taichi GGUI window.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable, Mapping, Sequence
from src.visualization.sources import RenderSource
from collections import deque
import time
from typing import Any, Mapping, Protocol, Sequence, Tuple, runtime_checkable

Color = Tuple[float, float, float]
Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class CameraOptions:
    position: Vector3 = (1.55, 1.25, 1.35)
    look_at: Vector3 = (0.5, 0.5, 0.5)
    up: Vector3 = (0.0, 0.0, 1.0)
    fov: float = 45.0
    movement_speed: float = 0.03

    def __post_init__(self) -> None:
        from src.visualization.Utilities import _finite_vector3

        object.__setattr__(self, "position", _finite_vector3("camera position", self.position))
        object.__setattr__(self, "look_at", _finite_vector3("camera look_at", self.look_at))
        object.__setattr__(self, "up", _finite_vector3("camera up", self.up))
        if not np.isfinite(self.fov) or not 1.0 <= self.fov <= 179.0:
            raise ValueError("camera fov must be finite and in [1, 179]")
        if not np.isfinite(self.movement_speed) or self.movement_speed <= 0.0:
            raise ValueError("camera movement_speed must be positive")
        position = np.asarray(self.position)
        look_at = np.asarray(self.look_at)
        up = np.asarray(self.up)
        view = look_at - position
        if np.linalg.norm(view) <= np.finfo(float).eps:
            raise ValueError("camera position and look_at must differ")
        if np.linalg.norm(np.cross(view, up)) <= np.finfo(float).eps:
            raise ValueError("camera up must not be parallel to the view")


@runtime_checkable
class RealtimeSceneModel(Protocol):
    name: str
    render_sources: Sequence[RenderSource]
    domain_min: Vector3
    domain_max: Vector3
    step_index: int
    simulation_time: float
    can_reset: bool

    def step(self) -> None:
        """Advance exactly one model time step."""

    def reset(self) -> bool:
        """Restore the initial state, returning whether reset was available."""

    def statistics(self) -> Mapping[str, object]:
        """Return inexpensive diagnostic values."""


class PlaybackController:
    """Display-independent playback state."""

    def __init__(self, *, paused: bool, steps_per_frame: int) -> None:
        self.paused = bool(paused)
        self.steps_per_frame = 1
        self._pending_single_steps = 0
        self.set_steps_per_frame(steps_per_frame)

    def set_steps_per_frame(self, value: int) -> None:
        value = int(value)
        if value < 1:
            raise ValueError("steps_per_frame must be positive")
        self.steps_per_frame = value

    def toggle(self) -> None:
        self.paused = not self.paused

    def request_single_step(self, count: int = 1) -> None:
        count = int(count)
        if count < 1:
            raise ValueError("single-step count must be positive")
        self.paused = True
        self._pending_single_steps += count

    def clear_pending_steps(self) -> None:
        self._pending_single_steps = 0

    def consume_step_budget(self) -> int:
        if not self.paused:
            return self.steps_per_frame
        budget = self._pending_single_steps
        self._pending_single_steps = 0
        return budget


class PerformanceTracker:
    def __init__(self, sample_count: int = 30) -> None:
        self._frame_times: deque[float] = deque(maxlen=sample_count)
        self._last_frame_end: float | None = None
        self.step_seconds = 0.0

    @property
    def fps(self) -> float:
        if not self._frame_times:
            return 0.0
        mean = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / mean if mean > 0.0 else 0.0

    def record_step_time(self, seconds: float) -> None:
        self.step_seconds = max(float(seconds), 0.0)

    def finish_frame(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else float(now)
        if self._last_frame_end is not None:
            elapsed = now - self._last_frame_end
            if elapsed > 0.0:
                self._frame_times.append(elapsed)
        self._last_frame_end = now


@dataclass
class MeshBinding:
    source: RenderSource
    node: Any
    mesh: Any
    primitive: Any
    kind: str
    item_count: int


@dataclass(frozen=True)
class RealtimeViewerOptions:
    title: str = "GeoTaichi Real-time"
    resolution: tuple[int, int] = (1280, 720)
    vsync: bool = True
    fps_limit: int = 60
    start_paused: bool = False
    steps_per_frame: int = 4
    maximum_steps_per_frame: int = 64
    particle_scale: float = 1.0
    background_color: Color = (0.025, 0.035, 0.055)
    ambient_light: Color = (0.38, 0.38, 0.42)
    point_light_position: Vector3 = (1.5, 1.5, 2.0)
    point_light_color: Color = (0.85, 0.85, 0.85)
    point_light_intensity: float = 4.0
    domain_color: Color = (0.95, 0.62, 0.18)
    sphere_subdivisions: int = 1
    camera: CameraOptions = CameraOptions()

    def __post_init__(self) -> None:
        from src.visualization.Utilities import _finite_vector3

        if not self.title.strip():
            raise ValueError("viewer title must not be empty")
        if len(self.resolution) != 2 or any(int(size) != size or size <= 0 for size in self.resolution):
            raise ValueError("resolution must contain two positive integers")
        if self.fps_limit <= 0:
            raise ValueError("fps_limit must be positive")
        if self.maximum_steps_per_frame < 1:
            raise ValueError("maximum_steps_per_frame must be positive")
        if not 1 <= self.steps_per_frame <= self.maximum_steps_per_frame:
            raise ValueError("steps_per_frame must be in [1, maximum_steps_per_frame]")
        if not np.isfinite(self.particle_scale) or self.particle_scale <= 0.0:
            raise ValueError("particle_scale must be positive")
        if self.sphere_subdivisions not in (0, 1, 2, 3):
            raise ValueError("sphere_subdivisions must be in [0, 3]")
        if not np.isfinite(self.point_light_intensity) or self.point_light_intensity <= 0.0:
            raise ValueError("point_light_intensity must be positive")
        for name in (
            "background_color",
            "ambient_light",
            "point_light_color",
            "domain_color",
        ):
            color = _finite_vector3(name, getattr(self, name))
            if any(component < 0.0 or component > 1.0 for component in color):
                raise ValueError(f"{name} components must be in [0, 1]")
            object.__setattr__(self, name, color)
        object.__setattr__(
            self,
            "point_light_position",
            _finite_vector3("point_light_position", self.point_light_position),
        )


class CallbackSceneModel:
    """Attach existing solver callbacks and geometry sources to the viewer."""

    def __init__(
        self,
        *,
        name: str,
        render_sources: Sequence[RenderSource],
        time_step: float,
        step_callback: Callable[[], None],
        reset_callback: Callable[[], None] | None = None,
        domain_min: Vector3 = (0.0, 0.0, 0.0),
        domain_max: Vector3 = (1.0, 1.0, 1.0),
        statistics_callback: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        from src.visualization.Utilities import _finite_vector3

        if not name.strip():
            raise ValueError("model name must not be empty")
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("model time_step must be positive")
        if not render_sources:
            raise ValueError("a real-time model needs at least one render source")
        self.name = name
        self.render_sources = tuple(render_sources)
        self.time_step = float(time_step)
        self.domain_min = _finite_vector3("domain_min", domain_min)
        self.domain_max = _finite_vector3("domain_max", domain_max)
        if any(upper <= lower for lower, upper in zip(self.domain_min, self.domain_max)):
            raise ValueError("domain_max must be greater than domain_min")
        self._step_callback = step_callback
        self._reset_callback = reset_callback
        self._statistics_callback = statistics_callback
        self.step_index = 0
        self.simulation_time = 0.0
        self.can_reset = reset_callback is not None

    def step(self) -> None:
        self._step_callback()
        self.step_index += 1
        self.simulation_time = self.step_index * self.time_step

    def reset(self) -> bool:
        if self._reset_callback is None:
            return False
        self._reset_callback()
        self.step_index = 0
        self.simulation_time = 0.0
        return True

    def statistics(self) -> Mapping[str, object]:
        if self._statistics_callback is None:
            return {}
        return self._statistics_callback()
