import numpy as np
from scipy.spatial import ConvexHull
import alphashape

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class BoundaryExtractor:
    def __init__(self):
        self.points = None
        self.boundary_indices = None

    def load_points(self, points_array):
        """Load point cloud data"""
        self.points = np.array(points_array)

    def method_geometric_boundary(self, tolerance=1e-6):
        """Method 1: Geometric boundary detection"""
        if self.points is None:
            raise ValueError("Please load point cloud data first")

        if self.points.ndim != 2 or self.points.shape[1] not in (2, 3):
            raise ValueError("BoundaryExtractor expects an N x 2 or N x 3 point array")
        minimum = np.min(self.points, axis=0)
        maximum = np.max(self.points, axis=0)
        on_boundary = np.any(
            (np.abs(self.points - minimum) <= tolerance) | (np.abs(self.points - maximum) <= tolerance),
            axis=1,
        )
        self.boundary_indices = np.flatnonzero(on_boundary).tolist()
        return np.asarray(self.boundary_indices)

    def method_convex_hull_plus_edges(self, edge_tolerance=1e-6):
        """Method 4: Convex hull plus edge points detection"""
        if self.points is None:
            raise ValueError("Please load point cloud data first")

        # Compute convex hull
        hull = ConvexHull(self.points)
        boundary_indices = set(hull.vertices)

        # For each convex hull edge, find all points on that edge
        for simplex in hull.simplices:
            p1_idx, p2_idx = simplex
            p1, p2 = self.points[p1_idx], self.points[p2_idx]

            # Check all points to see if they lie on this edge
            for i, point in enumerate(self.points):
                if i in boundary_indices:
                    continue

                # Check if point lies on the line segment
                if self._point_on_line_segment(point, p1, p2, edge_tolerance):
                    boundary_indices.add(i)

        self.boundary_indices = sorted(list(boundary_indices))
        return np.asarray(self.boundary_indices)

    def method_alpha_shape(self, alpha=None):
        if alpha is None:
            from scipy.spatial import cKDTree

            tree = cKDTree(self.points)
            dists, _ = tree.query(self.points, k=8)
            mean_dist = dists[:, 1:].mean()  # 忽略自己
            alpha = 1.5 * mean_dist
        alpha_shape = alphashape.alphashape(self.points, alpha)
        surface_points = set()
        surface_faces = []
        try:
            for simplex in alpha_shape.facets:
                surface_faces.append(simplex)
                surface_points.update(simplex)
        except AttributeError:
            coords = np.array(alpha_shape.exterior.coords)
            for c in coords:
                idx = np.argmin(np.linalg.norm(self.points - c, axis=1))
                surface_points.add(idx)
        self.boundary_indices = sorted(list(surface_points))
        return np.asarray(self.boundary_indices)

    def _point_on_line_segment(self, point, seg_start, seg_end, tolerance=1e-6):
        """Check if a point lies on a line segment"""
        # Vector calculations
        seg_vec = seg_end - seg_start
        point_vec = point - seg_start

        # If segment has zero length
        if np.allclose(seg_vec, [0, 0], atol=tolerance):
            return np.allclose(point, seg_start, atol=tolerance)

        # Calculate distance from point to line
        seg_length = np.linalg.norm(seg_vec)
        seg_unit = seg_vec / seg_length

        # Projection length
        proj_length = np.dot(point_vec, seg_unit)

        # Check if projection point is within segment bounds
        if proj_length < -tolerance or proj_length > seg_length + tolerance:
            return False

        # Calculate perpendicular distance
        proj_point = seg_start + proj_length * seg_unit
        perp_dist = np.linalg.norm(point - proj_point)

        return perp_dist <= tolerance

    def visualize_results(self, method_name="", save_path=None):
        """Visualize the results"""
        if self.points is None or self.boundary_indices is None:
            print("Please load data and compute boundary points first")
            return

        plt.figure(figsize=(12, 8))

        # Plot all points
        plt.scatter(
            self.points[:, 0],
            self.points[:, 1],
            c="lightblue",
            s=50,
            alpha=0.7,
            label=f"All points ({len(self.points)})",
        )

        # Plot boundary points
        if self.boundary_indices:
            boundary_points = self.points[self.boundary_indices]
            plt.scatter(
                boundary_points[:, 0],
                boundary_points[:, 1],
                c="red",
                s=80,
                alpha=0.9,
                label=f"Boundary points ({len(self.boundary_indices)})",
            )

            # Annotate boundary point IDs
            for idx in self.boundary_indices:
                point = self.points[idx]
                plt.annotate(
                    str(idx),
                    (point[0], point[1]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                    color="red",
                    fontweight="bold",
                )

        # Plot internal points
        internal_indices = [i for i in range(len(self.points)) if i not in self.boundary_indices]
        if internal_indices:
            internal_points = self.points[internal_indices]
            plt.scatter(
                internal_points[:, 0],
                internal_points[:, 1],
                c="green",
                s=30,
                alpha=0.5,
                label=f"Internal points ({len(internal_indices)})",
            )

        plt.title(f"{method_name}\nBoundary point IDs: {self.boundary_indices}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis("equal")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def compare_methods(self):
        """Compare both methods"""
        if self.points is None:
            raise ValueError("Please load point cloud data first")

        methods = [
            ("Geometric Boundary Detection", self.method_geometric_boundary),
            ("Convex Hull + Edge Points", self.method_convex_hull_plus_edges),
        ]

        results = {}

        print("Comparing boundary extraction methods:")
        print("=" * 60)

        for name, method in methods:
            try:
                boundary_indices = method()
                results[name] = boundary_indices
                print(f"{name}: {len(boundary_indices)} boundary points")
                print(f"  Boundary point IDs: {boundary_indices}")
                print()
            except Exception as e:
                print(f"{name}: Failed - {e}")
                print()

        # Visualize comparison
        if len(results) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            for i, (name, boundary_indices) in enumerate(results.items()):
                if i >= 2:
                    break

                ax = axes[i]

                # Plot all points
                ax.scatter(self.points[:, 0], self.points[:, 1], c="lightblue", s=40, alpha=0.7, label="All points")

                if boundary_indices:
                    # Plot boundary points
                    boundary_points = self.points[boundary_indices]
                    ax.scatter(
                        boundary_points[:, 0],
                        boundary_points[:, 1],
                        c="red",
                        s=70,
                        alpha=0.9,
                        label=f"Boundary ({len(boundary_indices)})",
                    )

                    # Annotate IDs
                    for idx in boundary_indices:
                        point = self.points[idx]
                        ax.annotate(
                            str(idx),
                            (point[0], point[1]),
                            xytext=(2, 2),
                            textcoords="offset points",
                            fontsize=8,
                            color="red",
                            fontweight="bold",
                        )

                ax.set_title(f"{name}\n{len(boundary_indices) if boundary_indices else 0} boundary points")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_aspect("equal")

            plt.tight_layout()
            plt.show()

        return results

    def get_boundary_summary(self):
        """Get boundary point summary information"""
        if self.boundary_indices is None:
            return "No boundary points computed yet"

        info = []
        info.append(f"Total points: {len(self.points)}")
        info.append(f"Boundary points: {len(self.boundary_indices)}")
        info.append(f"Internal points: {len(self.points) - len(self.boundary_indices)}")
        info.append(f"Boundary point IDs: {self.boundary_indices}")
        info.append("\nBoundary point coordinates:")

        for idx in self.boundary_indices:
            point = self.points[idx]
            info.append(f"  Point {idx}: ({point[0]:.3f}, {point[1]:.3f})")

        return "\n".join(info)

    def save_results(self, filename):
        """Save results to file"""
        if self.boundary_indices is None:
            raise ValueError("No boundary points computed yet")

        with open(filename, "w") as f:
            f.write("Rectangle Boundary Extraction Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(self.get_boundary_summary())

        print(f"Results saved to {filename}")
