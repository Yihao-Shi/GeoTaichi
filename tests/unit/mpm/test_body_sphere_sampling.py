import numpy as np

from src.mpm.generator.Body import Body


def test_add_sphere_seed_is_reproducible_and_volume_uniform():
    first = Body()
    second = Body()
    kwargs = {
        "center": [0.25, -0.5, 0.75],
        "radius": 1.2,
        "n_particles": 3000,
        "name": "sphere",
        "seed": 104729,
    }
    first.add_sphere(**kwargs)
    second.add_sphere(**kwargs)

    a = first.bodies["sphere"]
    b = second.bodies["sphere"]
    assert a["points"].shape == (3000, 3)
    assert np.array_equal(a["points"], b["points"])
    radius = np.linalg.norm(a["points"] - np.asarray(kwargs["center"])[None, :], axis=1)
    assert np.max(radius) <= kwargs["radius"]
    expected_volume = 4.0 * np.pi * kwargs["radius"] ** 3 / 3.0
    assert np.isclose(3000 * a["volume"], expected_volume)


def test_add_ring_packs_one_body_and_preserves_direct_grid_metadata():
    body = Body()

    body.add_ring(
        center=[0.5, -0.25],
        r_in=0.2,
        r_out=0.6,
        n_particles=32,
        init_v=[1.0, -2.0],
        name="annulus",
        grid_size=0.05,
        xmin=[0.0, -1.0],
        xmax=[1.5, 0.75],
    )

    ring = body.bodies["annulus"]
    assert body.particle_counter == 32
    assert ring["poffset"] == 0
    assert ring["points"].shape == (32, 2)
    assert ring["grid_size"] == 0.05
    np.testing.assert_array_equal(ring["init_v"], [1.0, -2.0])
    np.testing.assert_array_equal(ring["xmin"], [0.0, -1.0])
    np.testing.assert_array_equal(ring["xmax"], [1.5, 0.75])


def test_add_ring_honors_radial_and_circumferential_counts():
    body = Body()
    body.add_ring([0.0, 0.0], 0.8, 1.0, spacing=[4, 12], name="ring")

    assert body.bodies["ring"]["points"].shape == (48, 2)
    assert body.bodies["ring"]["boundary_ids"].shape == (12,)
