"""Host-side geometry descriptions consumed by the real-time viewer.

The classes in this module never allocate Taichi fields.  They read existing
solver fields only when a frame is synchronized with PyRender.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Any, Callable, Optional, Protocol, Tuple, Union

Color = Tuple[float, float, float]
ArrayProvider = Union[Any, Callable[[], Any]]
CountProvider = Optional[Union[int, Callable[[], int]]]


@dataclass(frozen=True)
class ParticleSnapshot:
    positions: np.ndarray
    radii: np.ndarray
    color: Color


@dataclass(frozen=True)
class TriangleMeshSnapshot:
    vertices: np.ndarray
    faces: np.ndarray
    color: Color
    wireframe: bool
    smooth: bool


class RenderSource(Protocol):
    name: str

    def snapshot(self) -> GeometrySnapshot:
        """Return one immutable host-side rendering snapshot."""


Vector3 = tuple[float, float, float]

GeometrySnapshot = Union[ParticleSnapshot, TriangleMeshSnapshot]


@dataclass
class ParticleRenderSource:
    """Particle positions plus CPU-side visualization radii.

    ``radii`` may be a scalar, NumPy array, callable, or an existing Taichi
    member field.  This source itself creates no radius field on the GPU.
    """

    positions: ArrayProvider
    radii: ArrayProvider
    count: CountProvider = None
    active: ArrayProvider | None = None
    color: Color = (0.22, 0.58, 0.95)
    name: str = "particles"

    def __post_init__(self) -> None:
        from src.visualization.Utilities import _validate_color

        self.color = _validate_color(self.color)

    def snapshot(self) -> ParticleSnapshot:
        from src.visualization.Utilities import _positions_3d
        from src.visualization.Utilities import _resolve
        from src.visualization.Utilities import _resolve_count

        raw_positions = np.asarray(_resolve(self.positions))
        if raw_positions.ndim != 2:
            raise ValueError("particle positions must be a two-dimensional array")
        count = _resolve_count(self.count, raw_positions.shape[0])
        positions = _positions_3d(raw_positions, count)

        raw_radii = _resolve(self.radii)
        if np.isscalar(raw_radii):
            radii = np.full(count, float(raw_radii), dtype=np.float32)
        else:
            radii = np.asarray(raw_radii, dtype=np.float32).reshape(-1)[:count]
            if radii.shape != (count,):
                raise ValueError("particle radii must provide one value per particle")

        if self.active is not None:
            active = np.asarray(_resolve(self.active)).reshape(-1)[:count]
            if active.shape != (count,):
                raise ValueError("particle active mask has the wrong size")
            keep = active.astype(bool)
            positions = positions[keep]
            radii = radii[keep]

        if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError("particle visualization radii must be finite and positive")
        return ParticleSnapshot(
            positions=np.ascontiguousarray(positions),
            radii=np.ascontiguousarray(radii),
            color=self.color,
        )


@dataclass
class TriangleMeshRenderSource:
    """A dynamic triangle mesh backed by host arrays or solver fields."""

    vertices: ArrayProvider
    faces: ArrayProvider
    vertex_count: CountProvider = None
    face_count: CountProvider = None
    color: Color = (0.70, 0.72, 0.76)
    wireframe: bool = False
    smooth: bool = False
    name: str = "triangles"

    def __post_init__(self) -> None:
        from src.visualization.Utilities import _validate_color

        self.color = _validate_color(self.color)

    def snapshot(self) -> TriangleMeshSnapshot:
        from src.visualization.Utilities import _positions_3d
        from src.visualization.Utilities import _resolve
        from src.visualization.Utilities import _resolve_count

        raw_vertices = np.asarray(_resolve(self.vertices))
        if raw_vertices.ndim != 2:
            raise ValueError("triangle vertices must be a two-dimensional array")
        vertex_count = _resolve_count(self.vertex_count, raw_vertices.shape[0])
        vertices = _positions_3d(raw_vertices, vertex_count)

        raw_faces = np.asarray(_resolve(self.faces))
        if raw_faces.ndim == 1:
            if raw_faces.size % 3:
                raise ValueError("flat triangle connectivity must be divisible by 3")
            raw_faces = raw_faces.reshape(-1, 3)
        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
            raise ValueError("triangle faces must have shape (face_count, 3)")
        face_count = _resolve_count(self.face_count, raw_faces.shape[0])
        faces = np.ascontiguousarray(raw_faces[:face_count], dtype=np.int32)
        if faces.size and (np.min(faces) < 0 or np.max(faces) >= vertices.shape[0]):
            raise ValueError("triangle connectivity references a missing vertex")
        return TriangleMeshSnapshot(
            vertices=vertices,
            faces=faces,
            color=self.color,
            wireframe=bool(self.wireframe),
            smooth=bool(self.smooth),
        )
