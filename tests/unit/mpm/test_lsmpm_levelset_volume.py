import numpy as np


def _cube_distance(axis):
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    qx = np.abs(x) - 0.5
    qy = np.abs(y) - 0.5
    qz = np.abs(z) - 0.5
    outside = np.sqrt(
        np.maximum(qx, 0.0) ** 2
        + np.maximum(qy, 0.0) ** 2
        + np.maximum(qz, 0.0) ** 2
    )
    inside = np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
    return outside + inside


def test_tetrahedral_volume_is_levelset_scale_invariant():
    from research.LSMPM.scripts.run_v1_sdf_transport import (
        regularized_cell_volume,
    )

    spacing = 0.05
    axis = np.arange(-0.8, 0.8 + 0.5 * spacing, spacing)
    phi = _cube_distance(axis)
    gnum = np.asarray(phi.shape[::-1], dtype=np.int64)
    reference = regularized_cell_volume(
        phi.ravel(), gnum, spacing, spacing**3
    )
    scaled = regularized_cell_volume(
        (3.7 * phi).ravel(), gnum, spacing, spacing**3
    )

    np.testing.assert_allclose(scaled, reference, rtol=2.0e-14, atol=1.0e-14)


def test_tetrahedral_volume_detects_zero_contour_shift():
    from research.LSMPM.scripts.run_v1_sdf_transport import (
        regularized_cell_volume,
    )

    spacing = 0.05
    axis = np.arange(-0.8, 0.8 + 0.5 * spacing, spacing)
    phi = _cube_distance(axis)
    gnum = np.asarray(phi.shape[::-1], dtype=np.int64)
    reference = regularized_cell_volume(
        phi.ravel(), gnum, spacing, spacing**3
    )
    contracted = regularized_cell_volume(
        (phi + 0.2 * spacing).ravel(),
        gnum,
        spacing,
        spacing**3,
    )

    assert contracted < reference
    assert (reference - contracted) / reference > 0.01
