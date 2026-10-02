from pathlib import Path

import numpy as np
import pytest

from examples.mpm.Contact.CPT2D import cpt_reference as reference
from examples.mpm.Contact.CPT2D.coupled_cpt import (
    axisymmetric_initial_deformation_gradient,
    axisymmetric_penetrator_mesh,
    axisymmetric_particle_count,
    axisymmetric_surface_pressure_load,
    dp_direct_material,
    dp_native_material,
    explicit_contact_parameters,
    ipc_contact_parameters,
    native_particle_capacity,
    penetrator_tetrahedra,
    penetrator_vertices,
    realized_grid_size,
)

ROOT = Path(__file__).resolve().parents[3]


def test_reference_profile_matches_standard_pile_file():
    pile = np.loadtxt(ROOT / "examples" / "mpm" / "Contact" / "CPT2D" / "pile.txt")
    np.testing.assert_allclose(pile[:, :2], np.asarray(reference.PILE_PROFILE))


def test_reference_parameters_are_the_standard_axisymmetric_cpt_values():
    assert reference.DOMAIN == (0.6, 2.508)
    assert reference.SOIL_SIZE == (0.6, 1.5)
    assert reference.GRID_SIZE == pytest.approx(0.006)
    assert reference.PARTICLES_PER_CELL == 2
    assert reference.TIMESTEP == pytest.approx(1.0e-5)
    assert reference.SIMULATION_TIME == pytest.approx(10.0)
    assert reference.PILE_SPEED == pytest.approx(0.1)
    assert reference.SURFACE_PRESSURE == pytest.approx(150.0e3)
    assert reference.INITIAL_STRESS == pytest.approx((-75.0e3, -150.0e3, -75.0e3, 0.0, 0.0, 0.0))
    assert reference.SOIL_MATERIAL == {
        "MaterialID": 1,
        "Density": 1600.0,
        "YoungModulus": 60.0e6,
        "PoissonRatio": 0.30,
        "e0": 0.62,
        "e_Tao": 0.90,
        "lambda_c": 0.119,
        "ksi": 0.23,
        "nd": 1.70,
        "nf": 2.68,
        "fai_c": 30.0,
        "Cohesion": 3000.0,
    }


def test_coupled_dp_changes_only_the_documented_constitutive_fields():
    native = dp_native_material()
    direct = dp_direct_material()
    for material in (native, direct):
        density = material.get("Density", material.get("density"))
        young = material.get("YoungModulus", material.get("young_modulus"))
        poisson = material.get("PoissonRatio", material.get("poisson_ratio"))
        cohesion = material.get("Cohesion")
        assert density == reference.SOIL_MATERIAL["Density"]
        assert young == reference.SOIL_MATERIAL["YoungModulus"]
        assert poisson == reference.SOIL_MATERIAL["PoissonRatio"]
        assert cohesion == reference.SOIL_MATERIAL["Cohesion"]
    assert native["Friction"] == reference.SOIL_MATERIAL["fai_c"]
    assert direct["FrictionAngle"] == reference.SOIL_MATERIAL["fai_c"]
    assert native["Dilation"] == direct["DilationAngle"]
    assert direct["DilationAngle"] == reference.SOIL_MATERIAL["fai_c"]


def test_extruded_penetrator_is_a_positive_complete_tetrahedralization():
    points = penetrator_vertices()
    volumes = []
    for a, b, c, d in penetrator_tetrahedra():
        matrix = np.column_stack((points[b] - points[a], points[c] - points[a], points[d] - points[a]))
        volumes.append(np.linalg.det(matrix) / 6.0)
    assert np.all(np.asarray(volumes) > 0.0)
    profile = np.asarray(reference.PILE_PROFILE)
    polygon_area = 0.5 * abs(
        np.dot(profile[:, 0], np.roll(profile[:, 1], -1)) - np.dot(profile[:, 1], np.roll(profile[:, 0], -1))
    )
    assert sum(volumes) == pytest.approx(polygon_area * reference.COUPLED_SLICE_THICKNESS)


def test_solver_specific_contact_controls_are_separate_and_grid_scaled():
    assert explicit_contact_parameters("fempm") == reference.FEMPM_EXPLICIT_CONTACT
    assert explicit_contact_parameters("igampm") == reference.IGAMPM_EXPLICIT_CONTACT
    grid_size = realized_grid_size(4.0)
    ipc = ipc_contact_parameters(grid_size)
    assert ipc["dhat"] == pytest.approx(0.5 * grid_size)
    assert ipc["dmin"] == pytest.approx(0.1 * grid_size)
    assert ipc["kappa"] == reference.SOIL_MATERIAL["YoungModulus"]
    with pytest.raises(ValueError, match="family"):
        explicit_contact_parameters("mpm")


def test_extruded_native_capacity_accounts_for_three_dimensional_ppc():
    # 100 x 8 x 250 cells, with 2^3 points per cell and 10% headroom.
    assert native_particle_capacity(reference.GRID_SIZE) == 1_760_000


def test_axisymmetric_discretization_and_surface_pressure_are_physical():
    assert axisymmetric_particle_count(reference.GRID_SIZE) == 100_000
    force = axisymmetric_surface_pressure_load(reference.GRID_SIZE)
    assert force.shape == (101,)
    assert np.sum(force) == pytest.approx(-reference.SURFACE_PRESSURE * np.pi * reference.SOIL_SIZE[0] ** 2)


def test_axisymmetric_direct_preload_reproduces_reference_hencky_stress():
    deformation = axisymmetric_initial_deformation_gradient()
    log_stretch = np.log(np.diag(deformation))
    material = reference.COUPLED_DP_MATERIAL
    shear = material["young_modulus"] / (2.0 * (1.0 + material["poisson_ratio"]))
    bulk = material["young_modulus"] / (3.0 * (1.0 - 2.0 * material["poisson_ratio"]))
    stress = 2.0 * shear * (log_stretch - np.mean(log_stretch)) + bulk * np.sum(log_stretch)
    np.testing.assert_allclose(stress, reference.INITIAL_STRESS[:3], rtol=1.0e-13)
    assert np.linalg.det(deformation) > 0.0


def test_axisymmetric_fem_penetrator_mesh_is_refined_and_matches_profile():
    points, cells = axisymmetric_penetrator_mesh()
    assert points.shape == (4 * 34, 2)
    assert cells.shape == (2 * 3 * 33, 3)
    edge1 = points[cells[:, 1]] - points[cells[:, 0]]
    edge2 = points[cells[:, 2]] - points[cells[:, 0]]
    signed_areas = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    assert np.all(signed_areas > 0.0)
    np.testing.assert_allclose(points[:4, 1], np.linspace(1.5, 1.5312, 4))
    np.testing.assert_allclose(points[-4:, 1], 2.5)


@pytest.mark.parametrize("value", ("nan", "inf", "0", "-1"))
def test_environment_override_rejects_nonfinite_or_nonpositive_values(value):
    with pytest.raises(ValueError, match="finite and positive"):
        reference.environment_float({"DT": value}, "DT", 1.0)
