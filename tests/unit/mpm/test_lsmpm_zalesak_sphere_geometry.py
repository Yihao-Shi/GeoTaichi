import math

import numpy as np

from research.LSMPM.scripts.run_v1_zalesak_sphere import (
    analytical_field,
    zalesak_sphere_signed_distance,
)


def test_zalesak_slot_is_open_and_extruded_through_z():
    x = np.asarray([0.0, 0.03, 0.0])
    y = np.asarray([0.15, 0.15, 0.15])
    z = np.asarray([0.0, 0.0, 0.05])

    phi = zalesak_sphere_signed_distance(x, y, z)

    assert phi[0] > 0.0
    assert phi[1] < 0.0
    assert phi[2] > 0.0


def test_zalesak_sphere_returns_after_one_revolution():
    coordinates = np.linspace(-0.4, 0.4, 17)
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")

    initial = analytical_field(x, y, z, 0.0)
    returned = analytical_field(x, y, z, 2.0 * math.pi)

    np.testing.assert_allclose(returned, initial, atol=2.0e-15, rtol=0.0)
