import numpy as np
import pytest
import taichi as ti

from geotaichi import init
from src.mpm.engines.TwoPhaseDoubleLayerKernel import kernel_mac_p2g_double_layer2d, kernel_mac_p2g_double_layer3d
from src.mpm.structs.Particle import ParticleCloudTwoPhase, ParticleCloudTwoPhase2D


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
def test_q2_cell_volume_projection_uses_the_velocity_basis_and_conserves_mass(dimension):
    init(dim=dimension, arch="cpu", default_fp="float64", cpu_max_num_threads=2, offline_cache=False, log=False)
    n, dx = 6, 0.125
    shape = (n,) * dimension
    faces = []
    for _ in range(5):
        fields = []
        for axis in range(dimension):
            extent = list(shape)
            extent[axis] += 1
            fields.append(ti.field(float, shape=extent))
        faces.extend(fields)
    mass, solid_mass, fraction = [ti.field(float, shape=shape) for _ in range(3)]
    solid_velocity, fluid_velocity = [ti.Vector.field(dimension, float, shape=shape) for _ in range(2)]
    particle = (ParticleCloudTwoPhase2D if dimension == 2 else ParticleCloudTwoPhase).field(shape=1)
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.phase.fill(2)
    particle.mf.fill(1)
    particle.ms.fill(2)
    particle.vol.fill(0.5 * dx**dimension)
    particle.porosity.fill(0.4)
    particle.x.from_numpy(np.full((1, dimension), 2.5 * dx))
    lengths = ti.Vector.field(dimension, float, shape=1)
    lengths.fill(dx / 2)
    scatter = kernel_mac_p2g_double_layer2d if dimension == 2 else kernel_mac_p2g_double_layer3d

    def project():
        scatter(
            1,
            ti.Vector([dx] * dimension),
            2,
            3,
            0,
            *faces,
            mass,
            solid_mass,
            fraction,
            solid_velocity,
            fluid_velocity,
            particle,
            lengths,
        )

    project()
    assert mass[(2,) * dimension] == pytest.approx(0.75**dimension)
    assert mass[(3,) + (2,) * (dimension - 1)] == pytest.approx(0.125 * 0.75 ** (dimension - 1))
    assert mass.to_numpy().sum() == pytest.approx(1)
    particle.phase.fill(1)
    project()
    assert solid_mass.to_numpy().sum() == pytest.approx(2)
    assert fraction[(2,) * dimension] == pytest.approx(0.3 * 0.75**dimension, abs=1e-7)


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
def test_domain_wall_images_preserve_a_uniform_water_volume(dimension):
    from src.mpm.engines.TwoPhaseDoubleLayerKernel import kernel_extend_double_layer_cell_support_at_solid_walls

    init(dim=dimension, arch="cpu", default_fp="float64", cpu_max_num_threads=2, offline_cache=False, log=False)
    n, dx = 4, 0.125
    shape = (n,) * dimension
    positions = (
        np.stack(np.meshgrid(*([np.arange(0.25, n, 0.5)] * dimension), indexing="ij"), axis=-1).reshape(-1, dimension)
        * dx
    )
    particle = (ParticleCloudTwoPhase2D if dimension == 2 else ParticleCloudTwoPhase).field(shape=len(positions))
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.phase.fill(2)
    particle.x.from_numpy(positions)
    particle.porosity.fill(1)
    particle.vol.fill((dx / 2) ** dimension)
    particle.mf.fill(1000 * (dx / 2) ** dimension)
    faces = []
    for _ in range(5):
        for axis in range(dimension):
            extent = list(shape)
            extent[axis] += 1
            faces.append(ti.field(float, shape=extent))
    mass, solid_mass, fraction, density = [ti.field(float, shape=shape) for _ in range(4)]
    solid_velocity, fluid_velocity = [ti.Vector.field(dimension, float, shape=shape) for _ in range(2)]
    lengths = ti.Vector.field(dimension, float, shape=1)
    lengths.fill(dx / 2)
    scatter = kernel_mac_p2g_double_layer2d if dimension == 2 else kernel_mac_p2g_double_layer3d
    scatter(
        len(positions),
        ti.Vector([dx] * dimension),
        2,
        3,
        0,
        *faces,
        mass,
        solid_mass,
        fraction,
        solid_velocity,
        fluid_velocity,
        particle,
        lengths,
    )
    expected = 1000 * dx**dimension
    assert mass[(0,) * dimension] < 0.9 * expected
    fraction.fill(1)
    density.fill(1000)
    cells = ti.field(int, shape=shape)
    cells.fill(1)
    kernel_extend_double_layer_cell_support_at_solid_walls(
        len(positions), ti.Vector([dx] * dimension), particle, mass, density, fraction, cells, 2
    )
    np.testing.assert_allclose(mass.to_numpy(), expected, rtol=1e-7)
    np.testing.assert_allclose(particle.mf.to_numpy(), 1000 * (dx / 2) ** dimension)
