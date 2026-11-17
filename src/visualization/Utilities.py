from __future__ import annotations
from src.visualization.sources import ParticleRenderSource
from src.visualization.sources import TriangleMeshRenderSource
from src.visualization.sources import Vector3
from typing import Sequence
import numpy as np
from src.visualization.sources import ArrayProvider
from src.visualization.sources import Color
from src.visualization.sources import CountProvider
from typing import Any, Sequence
from src.visualization.adapters import LevelSetDEMRenderSource
from typing import Any


def _finite_vector3(name: str, value: Sequence[float]) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    result = tuple(float(component) for component in value)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _resolve(provider: ArrayProvider) -> Any:
    value = provider() if callable(provider) else provider
    converter = getattr(value, "to_numpy", None)
    return converter() if converter is not None else value


def _resolve_count(provider: CountProvider, capacity: int) -> int:
    if provider is None:
        return capacity
    count = provider() if callable(provider) else provider
    count = int(count)
    if count < 0 or count > capacity:
        raise ValueError(f"render count {count} is outside field capacity {capacity}")
    return count


def _positions_3d(value: Any, count: int | None = None) -> np.ndarray:
    positions = np.asarray(_resolve(value), dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] not in (2, 3):
        raise ValueError("positions must have shape (count, 2) or (count, 3)")
    if count is not None:
        positions = positions[:count]
    if positions.shape[1] == 2:
        positions = np.column_stack((positions, np.zeros(positions.shape[0], dtype=np.float32)))
    if not np.all(np.isfinite(positions)):
        raise FloatingPointError("render positions contain non-finite values")
    return np.ascontiguousarray(positions)


def _validate_color(color: Sequence[float]) -> Color:
    if len(color) != 3:
        raise ValueError("render color must contain exactly three components")
    result = tuple(float(component) for component in color)
    if not np.all(np.isfinite(result)) or any(component < 0.0 or component > 1.0 for component in result):
        raise ValueError("render color components must be finite and in [0, 1]")
    return result


def dem_particle_source(
    scene: Any,
    *,
    color=(0.22, 0.58, 0.95),
) -> ParticleRenderSource:
    """Render ordinary DEM particles using their existing physical radius."""

    return ParticleRenderSource(
        name="DEM particles",
        positions=scene.particle.x,
        radii=scene.particle.rad,
        active=scene.particle.active,
        count=lambda: int(scene.particleNum[0]),
        color=color,
    )


def _mpm_host_radii(scene: Any) -> np.ndarray:
    """Return visualization radii from the existing CPU particle-size array."""

    count = int(scene.particleNum[0])
    psize = np.asarray(scene.psize[:count], dtype=np.float64)
    if psize.ndim == 1:
        radii = np.abs(psize)
    elif psize.ndim == 2 and psize.shape[1] in (2, 3):
        radii = np.linalg.norm(psize, axis=1)
    else:
        raise ValueError("MPM scene.psize must contain one 2D/3D half-size per particle")
    return np.maximum(radii, 1.0e-12).astype(np.float32)


def mpm_particle_source(
    scene: Any,
    *,
    color=(0.22, 0.58, 0.95),
) -> ParticleRenderSource:
    """Render MPM particles without allocating a GPU radius field."""

    return ParticleRenderSource(
        name="MPM material points",
        positions=scene.particle.x,
        radii=lambda: _mpm_host_radii(scene),
        active=scene.particle.active,
        count=lambda: int(scene.particleNum[0]),
        color=color,
    )


def dem_triangle_wall_source(
    scene: Any,
    *,
    color=(0.66, 0.68, 0.72),
) -> TriangleMeshRenderSource:
    """Render the current DEM facet/patch walls as triangles."""

    def vertices():
        count = int(scene.wallNum[0])
        first = scene.wall.vertice1.to_numpy()[:count]
        second = scene.wall.vertice2.to_numpy()[:count]
        third = scene.wall.vertice3.to_numpy()[:count]
        return np.stack((first, second, third), axis=1).reshape(-1, 3)

    def faces():
        count = int(scene.wallNum[0])
        return np.arange(3 * count, dtype=np.int32).reshape(-1, 3)

    return TriangleMeshRenderSource(
        name="DEM triangle walls",
        vertices=vertices,
        faces=faces,
        color=color,
        smooth=False,
    )


def dem_render_sources(sims: Any, scene: Any):
    """Select the appropriate render sources for a built DEM scene."""

    sources = []
    if sims.scheme in ("LSDEM", "LSMPM", "PolySuperEllipsoid", "PolySuperQuadrics"):
        sources.append(LevelSetDEMRenderSource(sims, scene))
    elif scene.particle is not None:
        sources.append(
            dem_particle_source(
                scene,
                color=getattr(sims, "particle_color", (0.22, 0.58, 0.95)),
            )
        )
    if getattr(sims, "wall_type", None) in (1, 2) and scene.wall is not None:
        sources.append(dem_triangle_wall_source(scene))
    return tuple(sources)
