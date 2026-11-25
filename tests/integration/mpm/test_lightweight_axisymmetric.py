"""End-to-end checks for the no-swirl Lightweight transfers."""

import numpy as np
import pytest

pytestmark = [pytest.mark.serial, pytest.mark.isolated_dimension(2)]


def _make_model(tmp_path, shape, projection, stabilize=None, *, contact=False, traction=None, sparse_grid=False):
    from src.mpm.mainMPM import MPM

    mpm = MPM(log=False)
    mpm.set_configuration(
        dimension=2,
        domain=[0.6, 0.6],
        gravity=[0, 0],
        mode="Lightweight",
        axisymmetric=True,
        shape_function=shape,
        velocity_projection=projection,
        stabilize=stabilize,
        sparse_grid=sparse_grid,
        alphaPIC=0.6,
        visualize=False,
        log=False,
    )
    mpm.set_solver(
        solver={"Timestep": 1e-4, "SimulationTime": 1e-4, "SaveInterval": 1e-4, "SavePath": str(tmp_path)},
        log=False,
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 16,
            "max_constraint_number": {"max_particle_traction_constraint": (3 if contact else 1) if traction else 0},
        },
        log=False,
    )
    if contact:
        mpm.add_contact("MPMContact", friction=0.0)
    mpm.add_material(
        model="LinearElastic",
        material={"MaterialID": 1, "Density": 1800, "YoungModulus": 2e4, "PoissonRatio": 0.3},
    )
    mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": [0.1, 0.1]})
    mpm.add_region(
        region={
            "Name": "ring",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.21, 0.21],
            "BoundingBoxSize": [0.08, 0.08],
        }
    )
    for body_id in range(2 if contact else 1):
        template = {"RegionName": "ring", "nParticlesPerCell": 1, "BodyID": body_id, "MaterialID": 1}
        if traction is not None:
            template["Traction"] = {"Pressure": traction}
        mpm.add_body(body={"Template": template})
    mpm.add_boundary_condition()
    mpm.select_save_data(particle=True, grid=False, object=False)
    assert mpm.scene.particleNum[0] == (2 if contact else 1)
    return mpm


@pytest.mark.parametrize(
    "shape,projection,stabilize",
    [
        ("Linear", None, None),
        ("Linear", "Taylor", None),
        ("GIMP", None, None),
        ("GIMP", "Affine", None),
        ("CubicBSpline", None, None),
        ("QuadBSpline", "Affine", None),
        ("Linear", "Affine", "B-Bar Method"),
        ("GIMP", None, "F-Bar Method"),
        ("QuadBSpline", "Affine", "F-Bar Method"),
    ],
)
def test_axisymmetric_hoop_force_and_strain(tmp_path, taichi_runtime, shape, projection, stabilize):
    mpm = _make_model(tmp_path, shape, projection, stabilize)
    particle = mpm.scene.particle
    position = np.array([0.25, 0.25])
    initial_radius = float(particle[0].x[0])
    particle[0].x = position
    particle[0].v = [0.2, 0.1]
    stress = np.array([0.0, 0.0, 3000.0, 0.0, 0.0, 0.0])
    particle[0].stress = stress
    radius = position[0]
    volume = float(particle[0].vol)
    mass = float(particle[0].m)
    dt = 1e-4
    assert volume == pytest.approx(initial_radius * 0.1**2)
    assert mass == pytest.approx(1800 * volume)

    mpm.run()

    # The pre-force grid velocity sets the hoop strain. The updated stress and volume set the force.
    hoop_rate = 0.2 / radius
    updated_volume = volume * (1.0 + dt * hoop_rate)
    young, poisson = 2e4, 0.3
    shear = young / (2.0 * (1.0 + poisson))
    lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    expected_stress = stress + dt * hoop_rate * np.array([lame, lame, lame + 2.0 * shear, 0.0, 0.0, 0.0])
    acceleration = -updated_volume * expected_stress[2] / (mass * radius)
    velocity = np.array([0.2 + dt * acceleration, 0.1])
    np.testing.assert_allclose(particle[0].x, position + dt * velocity, atol=1e-8, rtol=0)
    np.testing.assert_allclose(particle[0].v, velocity, atol=1e-8, rtol=0)
    assert particle[0].velocity_gradient[2, 2] == pytest.approx(hoop_rate, abs=1e-7)
    if projection != "Affine":
        np.testing.assert_allclose(particle[0].velocity_gradient, np.diag([0.0, 0.0, hoop_rate]), atol=1e-7, rtol=0)
    assert particle[0].vol == pytest.approx(updated_volume, rel=1e-8)
    np.testing.assert_allclose(particle[0].stress, expected_stress, atol=2e-6, rtol=0)
    node = mpm.scene.node
    np.testing.assert_allclose(node.m.to_numpy().sum(), mass, atol=1e-7, rtol=0)
    resultant_force = (node.m.to_numpy()[..., None] * node.force.to_numpy()).sum(axis=(0, 1))
    np.testing.assert_allclose(resultant_force, [mass * acceleration, 0.0], atol=2e-6, rtol=0)


def test_axisymmetric_rejects_zero_radius(tmp_path, taichi_runtime):
    from src.utils.errors import SimulationStateError

    mpm = _make_model(tmp_path, "Linear", None, None)
    mpm.scene.particle[0].x = [0.0, 0.25]
    with pytest.raises(SimulationStateError, match="positive particle radii"):
        mpm.run()


def test_axisymmetric_particle_traction_uses_area_per_radian(tmp_path, taichi_runtime):
    pressure = 1800.0
    mpm = _make_model(tmp_path, "GIMP", None, traction=[0.0, pressure])
    particle = mpm.scene.particle
    radius = float(particle[0].x[0])
    mass = float(particle[0].m)
    dt = 1e-4
    mpm.run()
    # The axial load acts on a radial strip of width h at the particle radius.
    expected_force = pressure * 0.1 * radius
    np.testing.assert_allclose(particle[0].v, [0.0, dt * expected_force / mass], atol=1e-8, rtol=0)
    node = mpm.scene.node
    nodal_force = (node.m.to_numpy()[..., None] * node.force.to_numpy()).sum(axis=(0, 1))
    np.testing.assert_allclose(nodal_force, [0.0, expected_force], atol=2e-6, rtol=0)


def test_axisymmetric_grid_contact_conserves_radial_momentum(tmp_path, taichi_runtime):
    mpm = _make_model(tmp_path, "Linear", None, contact=True)
    particle = mpm.scene.particle
    radii = np.array([0.25, 0.35])
    particle[0].x, particle[1].x = [radii[0], 0.25], [radii[1], 0.25]
    particle[0].v, particle[1].v = [1.0, 0.0], [-1.0, 0.0]
    initial_momentum = float(particle[0].m - particle[1].m)
    mpm.run()
    hoop_impulse = -1e-4 * np.sum(particle.vol.to_numpy()[:2] * particle.stress.to_numpy()[:2, 2] / radii)
    assert particle[0].v[0] < 1.0
    assert particle[1].v[0] > -1.0
    assert float(particle[0].m * particle[0].v[0] + particle[1].m * particle[1].v[0]) == pytest.approx(
        initial_momentum + hoop_impulse, abs=1e-7
    )


def test_axisymmetric_fbar_smooths_three_dimensional_jacobian(tmp_path, taichi_runtime):
    mpm = _make_model(tmp_path, "Linear", None, "F-Bar Method")
    mpm.add_body(body={"Template": {"RegionName": "ring", "nParticlesPerCell": 1, "BodyID": 0, "MaterialID": 1}})
    particle = mpm.scene.particle
    radii = np.array([0.24, 0.28])
    for p, radius in enumerate(radii):
        particle[p].x = [radius, 0.25]
        particle[p].v = [0.2, 0.0]
    masses = particle.m.to_numpy()[:2]
    volumes = particle.vol.to_numpy()[:2]
    dt = 1e-4
    jacobian = 1.0 + dt * 0.2 / radii
    # Both particles share the same z coordinate, so the axial weights cancel.
    weights = np.column_stack(((0.3 - radii) / 0.1, (radii - 0.2) / 0.1))
    nodal_jacobian = ((masses * jacobian) @ weights) / (masses @ weights)
    fraction = mpm.sims.fbar_fraction
    jacobian_bar = fraction * (weights @ nodal_jacobian) + (1.0 - fraction) * jacobian
    stretches = np.column_stack((np.ones((2, 2)), jacobian)) * (jacobian_bar / jacobian)[:, None] ** (1.0 / 3.0)

    mpm.run()

    np.testing.assert_allclose(particle.vol.to_numpy()[:2], volumes * jacobian_bar, atol=1e-12, rtol=0)
    young, poisson = 2e4, 0.3
    shear = young / (2.0 * (1.0 + poisson))
    lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    strain = stretches - 1.0
    expected_stress = 2.0 * shear * strain + lame * strain.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(particle.stress.to_numpy()[:2, :3], expected_stress, atol=2e-6, rtol=0)


@pytest.mark.parametrize(
    "shape,projection,stabilize,contact",
    [
        ("Linear", None, "B-Bar Method", False),
        ("GIMP", "Taylor", "F-Bar Method", True),
        ("QuadBSpline", "Affine", "F-Bar Method", False),
        ("CubicBSpline", None, None, False),
    ],
)
def test_axisymmetric_blockscan_matches_dense(tmp_path, taichi_runtime, shape, projection, stabilize, contact):
    ti = taichi_runtime
    if ti.lang.impl.current_cfg().arch not in (ti.cpu, ti.cuda):
        pytest.skip("BlockScan requires Taichi CPU or CUDA")

    reference = None
    for sparse_grid in (False, {"Backend": "BlockScan", "BlockSize": 2, "MaxActiveBlocks": 12}):
        mpm = _make_model(
            tmp_path / ("blockscan" if sparse_grid else "dense"),
            shape,
            projection,
            stabilize,
            contact=contact,
            traction=[0.0, 1800.0],
            sparse_grid=sparse_grid,
        )
        assert mpm.sims.mapping == "USF"
        if not contact:
            mpm.add_body(
                body={"Template": {"RegionName": "ring", "nParticlesPerCell": 1, "BodyID": 0, "MaterialID": 1}}
            )
        mpm.sims.set_simulation_time(3e-4)
        particle = mpm.scene.particle
        # Nonzero boundary types on the outer grid expose a confused physical/compact node ID.
        particle[0].x, particle[1].x = [0.45, 0.48], [0.54, 0.48]
        particle[0].v, particle[1].v = [0.2, 0.1], [-0.1, -0.2]
        particle[0].stress, particle[1].stress = [0, 0, 3000, 0, 0, 0], [0, 0, 1000, 0, 0, 0]
        mpm.run()

        state = [getattr(particle, name).to_numpy()[:2] for name in ("x", "v", "stress", "vol", "velocity_gradient")]
        for value in state:
            assert np.isfinite(value).all()
        assert np.max(np.abs(state[-1][:, 2, 2])) > 0
        if reference is None:
            reference = state
        else:
            grid = mpm.scene.sparse_grid
            assert 0 < grid.get_active_blocks() < grid.total_blocks
            assert grid.node_capacity < mpm.scene.element.gridSum
            assert mpm.scene.element.LnID is None
            for actual, expected in zip(state, reference):
                np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
