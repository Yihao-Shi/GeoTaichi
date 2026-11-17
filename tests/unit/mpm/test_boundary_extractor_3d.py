import numpy as np

from src.mpm.generator.BoundaryExtractor import BoundaryExtractor


def test_geometric_boundary_includes_z_faces():
    extractor = BoundaryExtractor()
    extractor.load_points(
        np.array(
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 1.0],
                [0.0, 0.5, 0.5],
                [1.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 1.0, 0.5],
            ]
        )
    )

    np.testing.assert_array_equal(extractor.method_geometric_boundary(), np.arange(1, 7))
