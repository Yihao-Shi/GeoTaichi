"""Mesh an implicit sphere and place tetrahedral Gauss points.

``pygalmesh`` and ``pyvista`` are optional and imported only by ``main``.
Importing this module is therefore safe in test discovery and documentation.
"""

import argparse

import numpy as np


def tetrahedron_gauss_rule(order):
    if order == 1:
        return np.asarray([[0.25, 0.25, 0.25]]), np.asarray([1.0])
    if order == 4:
        a = 0.58541020
        b = 0.13819660
        return (
            np.asarray([[a, b, b], [b, a, b], [b, b, a], [b, b, b]]),
            np.full(4, 0.25),
        )
    raise ValueError(
        "only the one- and four-point tetrahedron rules are supported"
    )


def map_reference_points(reference_points, vertices):
    reference_points = np.asarray(reference_points, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (4, 3):
        raise ValueError("vertices must have shape (4, 3)")
    barycentric0 = 1.0 - np.sum(reference_points, axis=1, keepdims=True)
    return (
        barycentric0 * vertices[0]
        + reference_points[:, 0:1] * vertices[1]
        + reference_points[:, 1:2] * vertices[2]
        + reference_points[:, 2:3] * vertices[3]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--cell-size", type=float, default=0.2)
    parser.add_argument("--order", type=int, choices=(1, 4), default=4)
    parser.add_argument("--mesh-output")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    import pygalmesh

    class SphereDomain(pygalmesh.DomainBase):
        def eval(self, point):
            return float(np.linalg.norm(point) - args.radius)

        def get_bounding_sphere_squared_radius(self):
            return float((1.1 * args.radius) ** 2)

    mesh = pygalmesh.generate_mesh(
        SphereDomain(), max_cell_circumradius=args.cell_size
    )
    if args.mesh_output:
        mesh.write(args.mesh_output)

    tetrahedra = next(
        block.data for block in mesh.cells if block.type == "tetra"
    )
    reference_points, _ = tetrahedron_gauss_rule(args.order)
    quadrature_points = np.vstack(
        [
            map_reference_points(reference_points, mesh.points[cell])
            for cell in tetrahedra
        ]
    )
    print(
        f"vertices={len(mesh.points)} tetrahedra={len(tetrahedra)} "
        f"quadrature_points={len(quadrature_points)}"
    )

    if args.show:
        import pyvista as pv

        grid = pv.UnstructuredGrid(
            {pv.CellType.TETRA: tetrahedra}, np.asarray(mesh.points)
        )
        plotter = pv.Plotter()
        plotter.add_mesh(grid, show_edges=True, opacity=0.3)
        plotter.add_points(quadrature_points, color="red", point_size=5)
        plotter.show()


if __name__ == "__main__":
    main()
