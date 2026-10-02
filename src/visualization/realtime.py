"""Real-time visualization for small GeoTaichi simulations.

Numerical models remain Taichi device programs.  Visualization uses -- PyRender, Pyglet, and PyOpenGL -- and deliberately
does not import or create a Taichi GGUI window.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import (
    Any,
    Callable,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
from src.utils.RuntimeHook import runtime_checkpoint

from .sources import (
    ParticleSnapshot,
    RenderSource,
    TriangleMeshSnapshot,
)


Color = Tuple[float, float, float]
Vector3 = Tuple[float, float, float]


def _finite_vector3(name: str, value: Sequence[float]) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    result = tuple(float(component) for component in value)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class CameraOptions:
    position: Vector3 = (1.55, 1.25, 1.35)
    look_at: Vector3 = (0.5, 0.5, 0.5)
    up: Vector3 = (0.0, 0.0, 1.0)
    fov: float = 45.0
    movement_speed: float = 0.03

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", _finite_vector3("camera position", self.position)
        )
        object.__setattr__(
            self, "look_at", _finite_vector3("camera look_at", self.look_at)
        )
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
        if not self.title.strip():
            raise ValueError("viewer title must not be empty")
        if (
            len(self.resolution) != 2
            or any(int(size) != size or size <= 0 for size in self.resolution)
        ):
            raise ValueError("resolution must contain two positive integers")
        if self.fps_limit <= 0:
            raise ValueError("fps_limit must be positive")
        if self.maximum_steps_per_frame < 1:
            raise ValueError("maximum_steps_per_frame must be positive")
        if not 1 <= self.steps_per_frame <= self.maximum_steps_per_frame:
            raise ValueError(
                "steps_per_frame must be in [1, maximum_steps_per_frame]"
            )
        if not np.isfinite(self.particle_scale) or self.particle_scale <= 0.0:
            raise ValueError("particle_scale must be positive")
        if self.sphere_subdivisions not in (0, 1, 2, 3):
            raise ValueError("sphere_subdivisions must be in [0, 3]")
        if (
            not np.isfinite(self.point_light_intensity)
            or self.point_light_intensity <= 0.0
        ):
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
            _finite_vector3(
                "point_light_position", self.point_light_position
            ),
        )


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
        if any(
            upper <= lower
            for lower, upper in zip(self.domain_min, self.domain_max)
        ):
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


class _PerformanceTracker:
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


def _camera_pose(options: CameraOptions) -> np.ndarray:
    eye = np.asarray(options.position, dtype=np.float64)
    target = np.asarray(options.look_at, dtype=np.float64)
    up_hint = np.asarray(options.up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _particle_poses(
    positions: np.ndarray, radii: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("render positions must have shape (particle_count, 3)")
    if not np.all(np.isfinite(positions)):
        raise FloatingPointError("render positions contain non-finite values")
    radii = np.asarray(radii, dtype=np.float32).reshape(-1)
    if radii.shape != (positions.shape[0],):
        raise ValueError("render radii must have one value per particle")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("render radii must be finite and positive")
    poses = np.repeat(
        np.eye(4, dtype=np.float32)[None, :, :], positions.shape[0], axis=0
    )
    poses[:, :3, 3] = positions
    poses[:, 0, 0] = radii * scale
    poses[:, 1, 1] = radii * scale
    poses[:, 2, 2] = radii * scale
    return poses


@dataclass
class _MeshBinding:
    source: RenderSource
    node: Any
    mesh: Any
    primitive: Any
    kind: str
    item_count: int


class RealtimeViewer:
    """PyRender/Pyglet viewer with camera interaction."""

    backend_name = "PyRender + Pyglet + PyOpenGL"

    def __init__(
        self,
        model: RealtimeSceneModel,
        options: RealtimeViewerOptions | None = None,
    ) -> None:
        self.model = model
        self.options = options or RealtimeViewerOptions()
        if not model.render_sources:
            raise ValueError("a real-time model needs at least one render source")
        self.controller = PlaybackController(
            paused=self.options.start_paused,
            steps_per_frame=self.options.steps_per_frame,
        )
        self.particle_scale = self.options.particle_scale
        self._performance = _PerformanceTracker()
        self._bindings: list[_MeshBinding] = []
        self._reset_requested = False
        self._scene = None

    def _build_scene(self) -> Any:
        from third_party import pyrender
        import trimesh

        scene = pyrender.Scene(
            bg_color=(*self.options.background_color, 1.0),
            ambient_light=self.options.ambient_light,
        )
        camera = pyrender.PerspectiveCamera(
            yfov=np.deg2rad(self.options.camera.fov),
            znear=1.0e-3,
            zfar=100.0,
        )
        scene.add(camera, pose=_camera_pose(self.options.camera))
        scene.add(
            pyrender.PointLight(
                color=self.options.point_light_color,
                intensity=self.options.point_light_intensity,
            ),
            pose=np.array(
                [
                    [1.0, 0.0, 0.0, self.options.point_light_position[0]],
                    [0.0, 1.0, 0.0, self.options.point_light_position[1]],
                    [0.0, 0.0, 1.0, self.options.point_light_position[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

        self._bindings.clear()
        for source in self.model.render_sources:
            snapshot = source.snapshot()
            self._bindings.append(
                self._add_snapshot(scene, source, snapshot)
            )

        lower = np.asarray(self.model.domain_min, dtype=np.float64)
        upper = np.asarray(self.model.domain_max, dtype=np.float64)
        domain_mesh = trimesh.creation.box(extents=upper - lower)
        domain_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(*self.options.domain_color, 1.0),
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        domain_pose = np.eye(4)
        domain_pose[:3, 3] = 0.5 * (lower + upper)
        scene.add(
            pyrender.Mesh.from_trimesh(
                domain_mesh,
                material=domain_material,
                wireframe=True,
                smooth=False,
            ),
            pose=domain_pose,
        )
        self._scene = scene
        return scene

    def _particle_mesh(self, snapshot: ParticleSnapshot) -> Any:
        from third_party import pyrender
        import trimesh

        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(*snapshot.color, 1.0),
            metallicFactor=0.05,
            roughnessFactor=0.62,
        )
        unit_sphere = trimesh.creation.icosphere(
            subdivisions=self.options.sphere_subdivisions,
            radius=1.0,
        )
        return pyrender.Mesh.from_trimesh(
            unit_sphere,
            material=material,
            poses=_particle_poses(
                snapshot.positions,
                snapshot.radii,
                self.particle_scale,
            ),
            smooth=True,
        )

    @staticmethod
    def _triangle_mesh(snapshot: TriangleMeshSnapshot) -> Any:
        from third_party import pyrender
        import trimesh

        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(*snapshot.color, 1.0),
            metallicFactor=0.02,
            roughnessFactor=0.72,
        )
        triangle_mesh = trimesh.Trimesh(
            vertices=snapshot.vertices,
            faces=snapshot.faces,
            process=False,
        )
        return pyrender.Mesh.from_trimesh(
            triangle_mesh,
            material=material,
            wireframe=snapshot.wireframe,
            smooth=snapshot.smooth,
        )

    def _add_snapshot(
        self, scene: Any, source: RenderSource, snapshot: Any
    ) -> _MeshBinding:
        if isinstance(snapshot, ParticleSnapshot):
            item_count = snapshot.positions.shape[0]
            kind = "particles"
            mesh = self._particle_mesh(snapshot) if item_count else None
        elif isinstance(snapshot, TriangleMeshSnapshot):
            item_count = snapshot.faces.shape[0]
            kind = "triangles"
            mesh = self._triangle_mesh(snapshot) if item_count else None
        else:
            raise TypeError(
                f"unsupported render snapshot from {source.name}: "
                f"{type(snapshot).__name__}"
            )
        node = scene.add(mesh) if mesh is not None else None
        primitive = mesh.primitives[0] if mesh is not None else None
        return _MeshBinding(
            source=source,
            node=node,
            mesh=mesh,
            primitive=primitive,
            kind=kind,
            item_count=item_count,
        )

    @staticmethod
    def _upload_poses(primitive: Any, poses: np.ndarray) -> None:
        primitive.poses = poses
        if not primitive._in_context():
            return
        if len(primitive._buffers) < 2:
            raise RuntimeError("PyRender primitive has no instance buffer")

        from OpenGL.GL import (
            GL_ARRAY_BUFFER,
            glBindBuffer,
            glBufferSubData,
        )

        pose_data = np.ascontiguousarray(
            np.transpose(poses, (0, 2, 1)).reshape(-1), dtype=np.float32
        )
        glBindBuffer(GL_ARRAY_BUFFER, primitive._buffers[1])
        glBufferSubData(GL_ARRAY_BUFFER, 0, pose_data.nbytes, pose_data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _synchronize_scene(self) -> None:
        if self._scene is None:
            raise RuntimeError("render scene has not been built")
        for index, binding in enumerate(self._bindings):
            snapshot = binding.source.snapshot()
            if isinstance(snapshot, ParticleSnapshot):
                item_count = snapshot.positions.shape[0]
                if binding.kind == "particles" and item_count == binding.item_count:
                    if item_count:
                        poses = _particle_poses(
                            snapshot.positions,
                            snapshot.radii,
                            self.particle_scale,
                        )
                        self._upload_poses(binding.primitive, poses)
                    continue
            elif isinstance(snapshot, TriangleMeshSnapshot):
                item_count = snapshot.faces.shape[0]
            else:
                raise TypeError(
                    f"unsupported render snapshot: {type(snapshot).__name__}"
                )

            if binding.node is not None and self._scene.has_node(binding.node):
                self._scene.remove_node(binding.node)
            self._bindings[index] = self._add_snapshot(
                self._scene, binding.source, snapshot
            )

    def _advance_model(self) -> None:
        step_count = self.controller.consume_step_budget()
        start = time.perf_counter()
        for _ in range(step_count):
            self.model.step()
            runtime_checkpoint()
        self._performance.record_step_time(time.perf_counter() - start)

    @staticmethod
    def _set_message(viewer: Any, text: str) -> None:
        viewer._message_text = text
        viewer._message_opac = 1.0 + viewer._ticks_till_fade

    def _registered_keys(self) -> Mapping[str, Callable[[Any], None]]:
        def toggle(viewer: Any) -> None:
            self.controller.toggle()
            status = "Paused" if self.controller.paused else "Running"
            self._set_message(viewer, status)

        def single_step(viewer: Any) -> None:
            self.controller.request_single_step()
            self._set_message(viewer, "Single step")

        def reset(viewer: Any) -> None:
            if self.model.can_reset:
                self._reset_requested = True
                self._set_message(viewer, "Simulation reset")
            else:
                self._set_message(viewer, "Reset unavailable")

        def fewer_steps(viewer: Any) -> None:
            value = max(1, self.controller.steps_per_frame - 1)
            self.controller.set_steps_per_frame(value)
            self._set_message(viewer, f"{value} steps / frame")

        def more_steps(viewer: Any) -> None:
            maximum = self.options.maximum_steps_per_frame
            value = min(maximum, self.controller.steps_per_frame + 1)
            self.controller.set_steps_per_frame(value)
            self._set_message(viewer, f"{value} steps / frame")

        return {
            " ": toggle,
            ".": single_step,
            "b": reset,
            "[": fewer_steps,
            "]": more_steps,
        }

    def _tick(self, _elapsed: float) -> None:
        if self._reset_requested:
            if self.model.reset():
                self.controller.clear_pending_steps()
            self._reset_requested = False
        self._advance_model()
        self._synchronize_scene()
        self._performance.finish_frame()

    def run(self) -> None:
        """Open the viewer and run it on the main thread."""

        import pyglet

        from third_party import pyrender

        if self.model.can_reset:
            self.model.reset()
        self.controller.clear_pending_steps()
        scene = self._build_scene()
        interval = 1.0 / self.options.fps_limit
        pyglet.clock.schedule_interval(self._tick, interval)
        try:
            pyrender.Viewer(
                scene,
                viewport_size=self.options.resolution,
                registered_keys=self._registered_keys(),
                run_in_thread=False,
                render_flags={
                    "shadows": False,
                    "cull_faces": True,
                },
                viewer_flags={
                    "window_title": self.options.title,
                    "refresh_rate": float(self.options.fps_limit),
                    "view_center": np.asarray(
                        self.options.camera.look_at, dtype=np.float64
                    ),
                    "use_raymond_lighting": False,
                    "use_direct_lighting": False,
                },
            )
        finally:
            pyglet.clock.unschedule(self._tick)
