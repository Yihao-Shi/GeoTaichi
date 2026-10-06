"""GeoTaichi analogue of Newton's cloth-rollers demonstration.

The mesh is a rolled TRI3 strip.  The inner seam is prescribed to rotate and
two frozen cylindrical SDF-spring frames approximate the roller surfaces.  The
SDF frames are intentionally static; use the generated case as a cloth
kinematics/contact regression, not as a replacement for a moving rigid-body
contact benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def rolled_mesh(nu, nv, length=24.0, width=5.0, radius=2.0, center=(-4.0, 0.0, 3.0)):
    points = []
    for i in range(nu):
        u = length * i / (nu - 1)
        theta = u / radius
        radial = radius + 0.02 * theta
        for j in range(nv):
            y = width * (j / (nv - 1) - 0.5)
            points.append([center[0] + radial * np.cos(theta), y, center[2] + radial * np.sin(theta)])
    cells = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j
            b = (i + 1) * nv + j
            cells.extend(((a, b, a + 1), (b, b + 1, a + 1)))
    return np.asarray(points, dtype=np.float64), np.asarray(cells, dtype=np.int32)


def roller_frames(points, centers, radii):
    targets = np.empty_like(points)
    normals = np.empty_like(points)
    for node, point in enumerate(points):
        distances = []
        frames = []
        for (cx, _, cz), radius in zip(centers, radii):
            radial = np.asarray([point[0] - cx, point[2] - cz], dtype=np.float64)
            norm = max(float(np.linalg.norm(radial)), 1.0e-12)
            normal = np.asarray([radial[0] / norm, 0.0, radial[1] / norm])
            target = np.asarray([cx + radius * normal[0], point[1], cz + radius * normal[2]])
            distances.append(abs(norm - radius))
            frames.append((target, normal))
        targets[node], normals[node] = frames[int(np.argmin(distances))]
    return targets, normals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gpu")
    parser.add_argument("--nu", type=int, default=48)
    parser.add_argument("--nv", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument(
        "--output-dir", default=str((Path(__file__).parents[1] / "OutputData") / "newton_cloth_rollers")
    )
    args = parser.parse_args()
    if args.nu < 4 or args.nv < 3 or args.steps <= 0 or args.dt <= 0.0:
        parser.error("nu >= 4, nv >= 3, steps > 0, and dt > 0 are required")

    import geotaichi as gt
    from src.fem import DirichletBoundary, FEM
    from src.fem.generator import FEMMesh

    gt.init(arch=args.arch, default_fp="float64", log=True)
    points, cells = rolled_mesh(args.nu, args.nv)
    centers = ((-4.0, 0.0, 3.0), (6.0, 0.0, 1.5))
    radii = (2.0, 2.8)
    targets, normals = roller_frames(points, centers, radii)
    cloth = FEM(title="Newton cloth rollers", log=True)
    cloth.set_configuration(dimension=3, solver_type="Implicit")
    cloth.add_mesh(FEMMesh(points, cells, "TRI3"))

    angular_speed = 2.0 * np.pi / 4.0

    def seam_rotation(time, coordinates):
        theta = angular_speed * float(time)
        cosine, sine = np.cos(theta), np.sin(theta)
        result = coordinates.copy()
        cx, cz = centers[0][0], centers[0][2]
        x = coordinates[:, 0] - cx
        z = coordinates[:, 2] - cz
        result[:, 0] = cx + cosine * x - sine * z
        result[:, 2] = cz + sine * x + cosine * z
        return result - coordinates

    seam = np.arange(args.nv, dtype=np.int32)
    cloth.add_boundary_condition(dirichlet=DirichletBoundary().add(seam, "all", seam_rotation))
    cloth.add_material(
        "ClothARAP",
        stretch_stiffness=1.0e3,
        compression_stiffness=1.0e3,
        density=0.02,
        thickness=2.0e-2,
        bending_stiffness=5.0e-2,
        bending_poisson_ratio=0.3,
        bending_model="Dihedral",
    )
    cloth.add_sdf(nodes="all", stiffness=2.0e2, dhat=0.5, targets=targets, normals=normals)
    cloth.set_solver(
        dt=args.dt,
        step=args.steps,
        gravity=(0.0, 0.0, 0.0),
        damping=0.1,
        max_iterations=40,
        residual_tolerance=1.0e-7,
        correction_velocity_tolerance=1.0e-3,
        line_search=True,
        assemble_type="HashTriplet",
        linear_solver="PCG",
        project_pd=True,
        output_interval=max(1, args.steps // 8),
        path=args.output_dir,
    )
    result = cloth.run(verbose=False)
    summary = {
        "case": "newton_cloth_rollers",
        "nodes": int(points.shape[0]),
        "triangles": int(cells.shape[0]),
        "steps": int(args.steps),
        "final_time": float(result.time),
        "converged": bool(result.converged),
        "finite_positions": bool(np.isfinite(result.positions).all()),
        "seam_rotation_radians": float(angular_speed * args.dt * args.steps),
    }
    output = Path(args.output_dir)
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if (
        not summary["converged"]
        or not summary["finite_positions"]
        or not np.isclose(summary["final_time"], args.steps * args.dt)
    ):
        raise RuntimeError(summary)


if __name__ == "__main__":
    main()
