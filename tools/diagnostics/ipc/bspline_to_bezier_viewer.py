"""Inspect quadratic B-spline knot insertion used for Bézier extraction."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_surface(resolution):
    import numpy as np

    from src.nurbs.BSplinePrimitives import BSplineSurface
    from src.nurbs.Fitting import compute_uniform_knot_vector

    surface = BSplineSurface()
    surface.degree_u = 2
    surface.degree_v = 2
    surface.knot_vector_u = compute_uniform_knot_vector(2, resolution)
    surface.knot_vector_v = compute_uniform_knot_vector(2, resolution)
    coordinates = np.linspace(0.0, 1.0, resolution)
    x, y, z = np.meshgrid(coordinates, coordinates, [0.0], indexing="xy")
    surface.control_points = np.stack(
        [x.ravel(), y.ravel(), z.ravel()], axis=1
    )
    surface.closed = False
    surface.parent_indices = None
    surface.activate_boundary()
    return surface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=5)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    surface = build_surface(args.resolution)
    if args.show:
        surface.visualize()
    surface.insert_knot((1.0 / 3.0, 2.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0))
    print(f"control_points={len(surface.control_points)}")
    print(f"u_knots={surface.knot_vector_u}")
    print(f"v_knots={surface.knot_vector_v}")
    if args.show:
        surface.visualize()


if __name__ == "__main__":
    main()
