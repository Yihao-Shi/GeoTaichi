"""Rendering adapters for GeoTaichi solver data structures."""

from __future__ import annotations
from src.utils.errors import SimulationStateError
from dataclasses import dataclass
from typing import Any
import numpy as np
from src.visualization.sources import TriangleMeshSnapshot


@dataclass
class LevelSetDEMRenderSource:
    """Dynamic LSDEM/LSMPM surface using the scene's canonical triangulation."""

    sims: Any
    scene: Any
    color: tuple[float, float, float] = (0.30, 0.66, 0.86)
    wireframe: bool = False
    smooth: bool = True
    name: str = "level-set DEM surface"

    def __post_init__(self) -> None:
        from src.visualization.Utilities import _validate_color

        self.color = _validate_color(self.color)

    def snapshot(self) -> TriangleMeshSnapshot:
        if self.scene.visualzie_surface_node is None:
            raise SimulationStateError(
                "level-set real-time rendering requires surface visualization "
                "to be activated while building the DEM scene"
            )
        _, vertices = self.scene.visualize_surface(self.sims)
        vertex_count = int(self.scene.surfaceNum[0])
        vertices = np.ascontiguousarray(vertices[:vertex_count], dtype=np.float32)
        faces = np.ascontiguousarray(self.scene.connectivity, dtype=np.int32)
        if faces.ndim == 1:
            faces = faces.reshape(-1, 3)
        return TriangleMeshSnapshot(
            vertices=vertices,
            faces=faces,
            color=tuple(float(value) for value in self.color),
            wireframe=bool(self.wireframe),
            smooth=bool(self.smooth),
        )


@dataclass
class IGARenderSource:
    """Sample current IGA control points into renderable boundary triangles."""

    @staticmethod
    def _grid_triangles(nu: int, nv: int, offset: int = 0) -> np.ndarray:
        faces = np.empty((2 * max(nu - 1, 0) * max(nv - 1, 0), 3), dtype=np.int32)
        cursor = 0
        for i in range(nu - 1):
            for j in range(nv - 1):
                n0 = offset + i * nv + j
                n1 = n0 + nv
                faces[cursor] = (n0, n1, n1 + 1)
                faces[cursor + 1] = (n0, n1 + 1, n0 + 1)
                cursor += 2
        return faces

    patch: Any
    resolution: int = 16
    color: tuple[float, float, float] = (0.30, 0.72, 0.56)
    wireframe: bool = False
    smooth: bool = True
    name: str = "IGA surface"

    def __post_init__(self) -> None:
        from src.visualization.Utilities import _validate_color

        if int(self.resolution) < 2:
            raise ValueError("IGA render resolution must be at least two")
        self.resolution = int(self.resolution)
        self.color = _validate_color(self.color)

    @staticmethod
    def _sample_patch_2d(primitive, control_points, resolution):
        from src.nurbs.NurbsBasis import nurbs_basis_interpolations2d

        values = np.linspace(0.0, 1.0, resolution)
        points = np.empty((resolution, resolution, primitive.dimension))
        for i, u in enumerate(values):
            for j, v in enumerate(values):
                points[i, j] = nurbs_basis_interpolations2d(
                    u,
                    v,
                    *primitive.degree,
                    primitive.knot_vector_u,
                    primitive.knot_vector_v,
                    control_points,
                    primitive.weights,
                )
        return points.reshape(-1, primitive.dimension), IGARenderSource._grid_triangles(resolution, resolution)

    @staticmethod
    def _sample_patch_3d(primitive, control_points, resolution):
        from src.nurbs.NurbsBasis import nurbs_basis_interpolations3d

        values = np.linspace(0.0, 1.0, resolution)
        vertices = []
        faces = []
        for fixed_axis in range(3):
            free_axes = [axis for axis in range(3) if axis != fixed_axis]
            for fixed_value in (0.0, 1.0):
                surface = np.empty((resolution, resolution, 3))
                for i, first in enumerate(values):
                    for j, second in enumerate(values):
                        coords = [0.0, 0.0, 0.0]
                        coords[fixed_axis] = fixed_value
                        coords[free_axes[0]] = first
                        coords[free_axes[1]] = second
                        surface[i, j] = nurbs_basis_interpolations3d(
                            *coords,
                            *primitive.degree,
                            primitive.knot_vector_u,
                            primitive.knot_vector_v,
                            primitive.knot_vector_w,
                            control_points,
                            primitive.weights,
                        )
                offset = sum(block.shape[0] for block in vertices)
                block = surface.reshape(-1, 3)
                vertices.append(block)
                triangles = IGARenderSource._grid_triangles(resolution, resolution, offset)
                if fixed_value == 0.0:
                    triangles = triangles[:, ::-1]
                faces.append(triangles)
        return np.concatenate(vertices), np.concatenate(faces)

    def snapshot(self) -> TriangleMeshSnapshot:
        all_control_points = self.patch.control_points.to_numpy()
        vertices = []
        faces = []
        for index, (_, metadata) in enumerate(self.patch.primitive.body.items()):
            primitive = metadata["primitive"]
            begin = int(self.patch.prefix_total_num_ctrlpts[index])
            count = int(self.patch.total_num_ctrlpts[index + 1])
            control_points = np.ascontiguousarray(all_control_points[begin : begin + count])
            if primitive.dimension == 2:
                patch_vertices, patch_faces = self._sample_patch_2d(primitive, control_points, self.resolution)
            elif primitive.dimension == 3:
                patch_vertices, patch_faces = self._sample_patch_3d(primitive, control_points, self.resolution)
            else:
                raise ValueError("IGA rendering supports only 2D and 3D patches")
            patch_faces = patch_faces + sum(block.shape[0] for block in vertices)
            vertices.append(patch_vertices)
            faces.append(patch_faces)

        if not vertices:
            raise ValueError("IGA patch contains no primitives")
        combined_vertices = np.concatenate(vertices)
        if combined_vertices.shape[1] == 2:
            combined_vertices = np.column_stack(
                (
                    combined_vertices,
                    np.zeros(combined_vertices.shape[0], dtype=np.float64),
                )
            )
        return TriangleMeshSnapshot(
            vertices=np.ascontiguousarray(combined_vertices, dtype=np.float32),
            faces=np.ascontiguousarray(np.concatenate(faces), dtype=np.int32),
            color=tuple(float(value) for value in self.color),
            wireframe=bool(self.wireframe),
            smooth=bool(self.smooth),
        )
