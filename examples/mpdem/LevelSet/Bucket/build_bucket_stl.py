#!/usr/bin/env python3
"""
Build an open STL surface mesh for the bucket using the geometry inferred from
bucket.txt and generate a structured triangle mesh with adjustable mesh size.

Geometry used here matches the accepted shape:
- lower cylinder: center (72, 11), radius 11, z = 11 -> 31, full circle
- upper cylinder: center (72, 35), radius 11, z = 11 -> 31, full circle
- lower outer wall: center (72, 11), radius 11, z = 31 -> 73, outer arc only
- upper outer wall: center (72, 35), radius 11, z = 31 -> 73, outer arc only
- middle outer wall: center (72, 23), radius 9, z = 31 -> 73, right arc only

No horizontal caps are generated, so the openings remain open.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]


def read_z_range(path: Path) -> tuple[float, float]:
    z_min = None
    z_max = None

    with path.open("r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                z = float(parts[2])
            except ValueError as exc:
                raise ValueError(f"invalid numeric data at line {lineno}") from exc

            z_min = z if z_min is None else min(z_min, z)
            z_max = z if z_max is None else max(z_max, z)

    if z_min is None or z_max is None:
        raise ValueError(f"no valid z values found in {path}")

    return z_min, z_max


def circle_intersection_angles() -> tuple[float, float, float]:
    """
    Return the key angles in degrees for the accepted piecewise model.

    lower/upper radius = 11
    middle radius = 9
    center distance = 12
    """
    alpha = math.degrees(math.acos((11.0**2 + 12.0**2 - 9.0**2) / (2.0 * 11.0 * 12.0)))
    beta = math.degrees(math.acos((9.0**2 + 12.0**2 - 11.0**2) / (2.0 * 9.0 * 12.0)))
    lower_cut = 90.0 - alpha
    middle_cut = beta
    return lower_cut, 180.0 - lower_cut, middle_cut


def patch_counts(
    radius: float, theta0_deg: float, theta1_deg: float, z0: float, z1: float, mesh_size: float
) -> tuple[int, int]:
    arc_angle_rad = math.radians(theta1_deg - theta0_deg)
    arc_length = abs(radius * arc_angle_rad)
    nz = max(1, math.ceil(abs(z1 - z0) / mesh_size))
    nt = max(8, math.ceil(arc_length / mesh_size))
    return nt, nz


def cylindrical_patch(
    cx: float,
    cy: float,
    radius: float,
    theta0_deg: float,
    theta1_deg: float,
    z0: float,
    z1: float,
    mesh_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    nt, nz = patch_counts(radius, theta0_deg, theta1_deg, z0, z1, mesh_size)
    theta = np.linspace(math.radians(theta0_deg), math.radians(theta1_deg), nt + 1)
    z = np.linspace(z0, z1, nz + 1)

    vertices = np.zeros(((nt + 1) * (nz + 1), 3), dtype=float)
    for j, zz in enumerate(z):
        row = j * (nt + 1)
        vertices[row : row + nt + 1, 0] = cx + radius * np.cos(theta)
        vertices[row : row + nt + 1, 1] = cy + radius * np.sin(theta)
        vertices[row : row + nt + 1, 2] = zz

    faces = []
    row_stride = nt + 1
    for j in range(nz):
        for i in range(nt):
            a = j * row_stride + i
            b = a + 1
            d = a + row_stride
            c = d + 1
            if (i + j) % 2 == 0:
                faces.append((a, b, c))
                faces.append((a, c, d))
            else:
                faces.append((a, b, d))
                faces.append((b, c, d))

    return vertices, np.asarray(faces, dtype=int)


def compute_normal(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    normal = np.cross(p1 - p0, p2 - p0)
    length = np.linalg.norm(normal)
    if length <= 1e-14:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    return normal / length


def write_ascii_stl(path: Path, vertices: np.ndarray, faces: np.ndarray, solid_name: str) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write(f"solid {solid_name}\n")
        for tri in faces:
            p0, p1, p2 = vertices[tri]
            normal = compute_normal(p0, p1, p2)
            f.write(f"  facet normal {normal[0]:.8e} {normal[1]:.8e} {normal[2]:.8e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {p0[0]:.8e} {p0[1]:.8e} {p0[2]:.8e}\n")
            f.write(f"      vertex {p1[0]:.8e} {p1[1]:.8e} {p1[2]:.8e}\n")
            f.write(f"      vertex {p2[0]:.8e} {p2[1]:.8e} {p2[2]:.8e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def combine_meshes(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    vertices_all = []
    faces_all = []
    offset = 0
    for vertices, faces in parts:
        vertices_all.append(vertices)
        faces_all.append(faces + offset)
        offset += len(vertices)
    return np.vstack(vertices_all), np.vstack(faces_all)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a structured STL mesh for the bucket.")
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "assets/data/MPDEM/Bucket/bucket.txt"),
        help="input point file used to read z range",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "assets/mesh/MPDEM/bucket.stl"),
        help="output STL path",
    )
    parser.add_argument(
        "--mesh-size",
        type=float,
        default=1.0,
        help="target triangle edge size in model units",
    )
    parser.add_argument(
        "--z-split",
        type=float,
        default=31.0,
        help="z level where the middle cylinder starts",
    )
    args = parser.parse_args()

    if args.mesh_size <= 0.0:
        raise ValueError("--mesh-size must be positive")

    z_min, z_max = read_z_range(Path(args.input))
    z_split = args.z_split
    if not (z_min < z_split < z_max):
        raise ValueError(f"--z-split must satisfy {z_min} < z-split < {z_max}")

    lower_cut0, lower_cut1, middle_cut = circle_intersection_angles()

    parts = [
        cylindrical_patch(72.0, 11.0, 11.0, 0.0, 360.0, z_min, z_split, args.mesh_size),
        cylindrical_patch(72.0, 35.0, 11.0, 0.0, 360.0, z_min, z_split, args.mesh_size),
        cylindrical_patch(72.0, 11.0, 11.0, lower_cut1, 360.0 + lower_cut0, z_split, z_max, args.mesh_size),
        cylindrical_patch(72.0, 35.0, 11.0, -lower_cut0, 360.0 - lower_cut1, z_split, z_max, args.mesh_size),
        cylindrical_patch(72.0, 23.0, 9.0, -0.5 * middle_cut, 0.5 * middle_cut, z_split, z_max, args.mesh_size),
        cylindrical_patch(
            72.0, 23.0, 9.0, 180.0 - 0.5 * middle_cut, 180.0 + 0.5 * middle_cut, z_split, z_max, args.mesh_size
        ),
    ]

    vertices, faces = combine_meshes(parts)
    write_ascii_stl(Path(args.output), vertices, faces, "bucket")

    print(f"STL written to: {args.output}")
    print(f"z-range: {z_min} -> {z_max}")
    print(f"middle starts at z: {z_split}")
    print(f"mesh size: {args.mesh_size}")
    print(f"vertices: {len(vertices)}")
    print(f"triangles: {len(faces)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
