import math

import numpy as np


def dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def sub(a, b):
    return tuple(ai - bi for ai, bi in zip(a, b))


def scale(s, a):
    return tuple(s * ai for ai in a)


def add(a, b):
    return tuple(ai + bi for ai, bi in zip(a, b))


def assert_vec_close(actual, expected, tol=1.0e-12):
    assert len(actual) == len(expected)
    for ai, ei in zip(actual, expected):
        assert math.isclose(ai, ei, rel_tol=tol, abs_tol=tol)


def cut_cell_face_velocity(open_fraction, fluid_velocity, solid_velocity):
    return open_fraction * fluid_velocity + (1.0 - open_fraction) * solid_velocity


def pressure_face_force(pressure, face_area, side, normal, open_fraction):
    solid_fraction = 1.0 - open_fraction
    return scale(solid_fraction * pressure * face_area * side, normal)


def paper_volume_fraction_ibm_force(
    fluid_velocity, solid_velocity, solid_fraction, rho_f, rho_s, dt, div_sigma, cell_volume
):
    rho = (1.0 - solid_fraction) * rho_f + solid_fraction * rho_s
    weighted_solid_fraction = solid_fraction * rho_s / rho
    corrected_velocity = add(
        scale(1.0 - weighted_solid_fraction, fluid_velocity), scale(weighted_solid_fraction, solid_velocity)
    )
    f_ib = scale(rho / dt, sub(corrected_velocity, fluid_velocity))
    return scale(
        cell_volume, sub(scale(weighted_solid_fraction, div_sigma), scale(1.0 - weighted_solid_fraction, f_ib))
    )


def test_cut_cell_flux_uses_open_and_blocked_velocity():
    assert math.isclose(cut_cell_face_velocity(0.25, 2.0, -1.0), -0.25)
    assert math.isclose(cut_cell_face_velocity(1.0, 2.0, -1.0), 2.0)
    assert math.isclose(cut_cell_face_velocity(0.0, 2.0, -1.0), -1.0)


def test_pressure_force_uses_cut_cell_blocked_area():
    force = pressure_face_force(pressure=10.0, face_area=0.125, side=-1.0, normal=(0.0, 1.0, 0.0), open_fraction=0.2)
    assert_vec_close(force, (0.0, -1.0, 0.0))


def test_volume_fraction_ibm_force_matches_paper_eq_19_21_28():
    fluid_velocity = (3.0, 0.0, 0.0)
    solid_velocity = (1.0, 0.0, 0.0)
    paper_force = paper_volume_fraction_ibm_force(
        fluid_velocity=fluid_velocity,
        solid_velocity=solid_velocity,
        solid_fraction=0.25,
        rho_f=1000.0,
        rho_s=2500.0,
        dt=0.1,
        div_sigma=(0.5, 0.0, 0.0),
        cell_volume=0.2,
    )

    assert_vec_close(paper_force, (1363.6818181818182, 0.0, 0.0))
    assert paper_force[0] > 0.0


def test_volume_fraction_ibm_exchange_obeys_action_reaction():
    solid_fraction, rho_f, rho_s, dt, cell_volume = 0.35, 1000.0, 1800.0, 0.02, 0.004
    fluid_velocity = np.array([0.2, -0.1, 0.0])
    solid_velocity = np.array([-0.3, 0.4, 0.1])
    mixed_density = (1.0 - solid_fraction) * rho_f + solid_fraction * rho_s
    weighted_solid_fraction = solid_fraction * rho_s / mixed_density
    velocity_change = weighted_solid_fraction * (solid_velocity - fluid_velocity)
    ibm_force_density = mixed_density * velocity_change / dt
    fluid_impulse = (1.0 - solid_fraction) * rho_f * velocity_change * cell_volume
    body_impulse = -(1.0 - weighted_solid_fraction) * ibm_force_density * cell_volume * dt

    np.testing.assert_allclose(body_impulse + fluid_impulse, 0.0, atol=1.0e-15)


def test_viscous_shear_enters_through_stress_divergence():
    viscosity = 2.0
    velocity_laplacian = (4.0, -2.0, 0.0)
    viscous_divergence = scale(viscosity, velocity_laplacian)
    force = paper_volume_fraction_ibm_force(
        fluid_velocity=(1.0, 1.0, 0.0),
        solid_velocity=(1.0, 1.0, 0.0),
        solid_fraction=0.5,
        rho_f=1000.0,
        rho_s=1000.0,
        dt=0.05,
        div_sigma=viscous_divergence,
        cell_volume=0.25,
    )

    assert_vec_close(force, (1.0, -0.5, 0.0))


def test_affine_force_mapping_preserves_resultant_and_moment():
    controls = np.array([[0.1, 0.2, 0.3], [1.1, 0.2, 0.3], [0.1, 1.2, 0.3], [0.1, 0.2, 1.3]])
    material = np.array([0.2, 0.3, 0.1])
    weights = np.r_[1.0 - material.sum(), material]
    force = np.array([2.0, -3.0, 4.0])
    generalized = weights[:, None] * force
    point = weights @ controls
    origin = np.array([-0.4, 0.6, 0.2])

    np.testing.assert_allclose(generalized.sum(axis=0), force)
    np.testing.assert_allclose(
        np.cross(controls - origin, generalized).sum(axis=0),
        np.cross(point - origin, force),
    )


if __name__ == "__main__":
    test_cut_cell_flux_uses_open_and_blocked_velocity()
    test_pressure_force_uses_cut_cell_blocked_area()
    test_volume_fraction_ibm_force_matches_paper_eq_19_21_28()
    test_viscous_shear_enters_through_stress_divergence()
    print("incompressible IBM coupling formula tests passed")
