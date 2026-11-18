from __future__ import annotations
from src.visualization.realtime import RealtimeViewerOptions
from src.visualization.realtime import CameraOptions
from src.visualization.realtime import PlaybackController
from src.visualization.realtime import RealtimeSceneModel
from src.visualization.realtime import MeshBinding
from src.visualization.realtime import PerformanceTracker
from src.utils.errors import SimulationStateError
import time
from typing import Any, Callable, Mapping
import numpy as np
from src.utils.RuntimeHook import runtime_checkpoint
from src.visualization.sources import ParticleSnapshot, RenderSource, TriangleMeshSnapshot


class RealtimeViewer:
    """PyRender/Pyglet viewer with camera interaction."""

    @staticmethod
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

    @staticmethod
    def _particle_poses(positions: np.ndarray, radii: np.ndarray, scale: float = 1.0) -> np.ndarray:
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
        poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], positions.shape[0], axis=0)
        poses[:, :3, 3] = positions
        poses[:, 0, 0] = radii * scale
        poses[:, 1, 1] = radii * scale
        poses[:, 2, 2] = radii * scale
        return poses

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
        self._performance = PerformanceTracker()
        self._bindings: list[MeshBinding] = []
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
        scene.add(camera, pose=RealtimeViewer._camera_pose(self.options.camera))
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
            self._bindings.append(self._add_snapshot(scene, source, snapshot))

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
            poses=RealtimeViewer._particle_poses(
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

    def _add_snapshot(self, scene: Any, source: RenderSource, snapshot: Any) -> MeshBinding:
        if isinstance(snapshot, ParticleSnapshot):
            item_count = snapshot.positions.shape[0]
            kind = "particles"
            mesh = self._particle_mesh(snapshot) if item_count else None
        elif isinstance(snapshot, TriangleMeshSnapshot):
            item_count = snapshot.faces.shape[0]
            kind = "triangles"
            mesh = self._triangle_mesh(snapshot) if item_count else None
        else:
            raise TypeError(f"unsupported render snapshot from {source.name}: " f"{type(snapshot).__name__}")
        node = scene.add(mesh) if mesh is not None else None
        primitive = mesh.primitives[0] if mesh is not None else None
        return MeshBinding(
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
            raise SimulationStateError("PyRender primitive has no instance buffer")

        from OpenGL.GL import (
            GL_ARRAY_BUFFER,
            glBindBuffer,
            glBufferSubData,
        )

        pose_data = np.ascontiguousarray(np.transpose(poses, (0, 2, 1)).reshape(-1), dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, primitive._buffers[1])
        glBufferSubData(GL_ARRAY_BUFFER, 0, pose_data.nbytes, pose_data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _synchronize_scene(self) -> None:
        if self._scene is None:
            raise SimulationStateError("render scene has not been built")
        for index, binding in enumerate(self._bindings):
            snapshot = binding.source.snapshot()
            if isinstance(snapshot, ParticleSnapshot):
                item_count = snapshot.positions.shape[0]
                if binding.kind == "particles" and item_count == binding.item_count:
                    if item_count:
                        poses = RealtimeViewer._particle_poses(
                            snapshot.positions,
                            snapshot.radii,
                            self.particle_scale,
                        )
                        self._upload_poses(binding.primitive, poses)
                    continue
            elif isinstance(snapshot, TriangleMeshSnapshot):
                item_count = snapshot.faces.shape[0]
            else:
                raise TypeError(f"unsupported render snapshot: {type(snapshot).__name__}")

            if binding.node is not None and self._scene.has_node(binding.node):
                self._scene.remove_node(binding.node)
            self._bindings[index] = self._add_snapshot(self._scene, binding.source, snapshot)

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
                    "view_center": np.asarray(self.options.camera.look_at, dtype=np.float64),
                    "use_raymond_lighting": False,
                    "use_direct_lighting": False,
                },
            )
        finally:
            pyglet.clock.unschedule(self._tick)
