from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.EngineKernel import (
    calculate_cell_volume_weighted,
    init_boundary,
    kernel_build_fluid_sdf_from_volume_fraction,
    kernel_fill_enclosed_fluid_cells,
    kernel_deactivate_particles_in_solid_sdf,
    kernel_find_fluid_domain_by_volume,
    kernel_mass_momentum_mac_cell_p2g,
    resolve_solid_normal_velocity,
)
from src.mpm.generator.InsertionKernel import kernel_rebulid_incompressible_particle_2D
from src.mpm.structs.Particle import ParticleCloudIncompressible2D
from src.mpm.structs.StaggeredGrid import StaggeredGrid
import src.utils.GlobalVariable as GlobalVariable
from examples.mpm.IncompressibleFluid.wavemaker_tank_3d import (
    enforce_moving_piston_particles,
    piston_displacement,
    piston_velocity,
    set_moving_piston_velocity,
)
from examples.mmpm.TwoPhaseWavemaker3D.two_layer_two_phase_wavemaker_3d import (
    set_moving_piston_mac_boundary,
)

pytestmark = [pytest.mark.unit, pytest.mark.mpm]


@ti.kernel
def seed_dense_incompressible_particles(
    particles: ti.template(),
    particles_per_axis: int,
    grid_size: ti.types.vector(2, float),
):
    particles_per_cell = particles_per_axis**2
    for particle_id in particles:
        cell_id = particle_id // particles_per_cell
        local_id = particle_id - cell_id * particles_per_cell
        cell = ti.Vector([2 + cell_id % 4, 2 + cell_id // 4])
        local = ti.Vector(
            [
                (local_id % particles_per_axis + 0.5) / particles_per_axis,
                (local_id // particles_per_axis + 0.5) / particles_per_axis,
            ]
        )
        position = (cell + local) * grid_size
        velocity = ti.Vector([0.2 + 1.0e-4 * particle_id, 0.1 + 2.0e-4 * particle_id])
        velocity_gradient = ti.Matrix([[0.005, 0.0], [0.015, 0.01]])
        particles[particle_id].particleID = particle_id
        particles[particle_id].bodyID = ti.u8(0)
        particles[particle_id].materialID = ti.u8(1)
        particles[particle_id].active = ti.u8(1)
        particles[particle_id].m = 0.003 + 1.0e-6 * particle_id
        particles[particle_id].vol = particles[particle_id].m / 1000.0
        particles[particle_id].x = position
        particles[particle_id].v = velocity
        particles[particle_id].velocity_gradient = velocity_gradient


@ti.kernel
def copy_offset_field_to_dense(
    source: ti.template(),
    destination: ti.template(),
    source_offset: ti.types.vector(2, int),
):
    for index in ti.grouped(destination):
        destination[index] = source[index + source_offset]


def offset_field_to_numpy(field, ghost_cell):
    """Export an offset field without Taichi 1.6/1.7 ``to_numpy`` aborts."""
    dense = ti.field(dtype=float, shape=field.shape)
    copy_offset_field_to_dense(field, dense, ti.Vector([-ghost_cell, -ghost_cell]))
    return dense.to_numpy()


def snapshot_mac_grid(node, ghost_cell):
    return [
        (
            offset_field_to_numpy(node.m[axis], ghost_cell),
            offset_field_to_numpy(node.velocity[axis], ghost_cell),
        )
        for axis in range(2)
    ]


def clear_mac_grid(node):
    for fields in (node.m, node.velocity, node.force):
        for field in fields:
            field.fill(0.0)


@ti.kernel
def count_nonzero_cells(cell_type: ti.template()) -> int:
    count = 0
    for index in ti.grouped(cell_type):
        if cell_type[index] != 0:
            count += 1
    return count


@ti.kernel
def seed_crossing_incompressible_particle(particle: ti.template(), grid_size: ti.types.vector(2, float)):
    particle[0].active = ti.u8(1)
    particle[0].materialID = ti.u8(1)
    particle[0].x = ti.Vector([grid_size[0], 1.5 * grid_size[1]])
    particle[0].vol = grid_size[0] * grid_size[1]


@ti.kernel
def seed_upper_boundary_particle(particle: ti.template(), domain: ti.types.vector(2, float)):
    particle[0].active = ti.u8(1)
    particle[0].materialID = ti.u8(1)
    particle[0].x = ti.Vector([domain[0], 0.5 * domain[1]])
    particle[0].vol = 0.01


@ti.kernel
def seed_sdf_test_particle(particle: ti.template(), position: ti.types.vector(2, float)):
    particle[0].active = ti.u8(1)
    particle[0].x = position


@ti.kernel
def seed_piston_test_particles(particle: ti.template()):
    for p in range(2):
        particle[p].active = ti.u8(1)
        particle[p].materialID = ti.u8(1)
        particle[p].x = ti.Vector([0.09 + 0.02 * p, 0.1])
        particle[p].v = ti.Vector([0.05, 0.02])


@pytest.mark.cpu
def test_init_boundary_accepts_2d_integer_cnum(taichi_runtime):
    """Regression for kernels imported before geotaichi.init(dim=2)."""
    cell_type = ti.field(dtype=ti.i32, shape=(6, 5), offset=(-1, -1))
    cell_type.fill(2)

    init_boundary(1, ti.Vector([4, 3]), cell_type)

    # cnum - 2 * ghost_cell = [2, 1], so only (0, 0) and (1, 0)
    # remain in the active domain.
    assert count_nonzero_cells(cell_type) == 2


@pytest.mark.cpu
def test_weighted_cell_volume_keeps_support_on_both_sides_of_grid_crossing(taichi_runtime):
    cnum = ti.Vector([4, 4])
    grid_size = ti.Vector([0.2, 0.25])
    particle = ParticleCloudIncompressible2D.field(shape=1)
    cell_volume_fraction = ti.field(float, shape=16)

    seed_crossing_incompressible_particle(particle, grid_size)
    calculate_cell_volume_weighted(
        cell_volume_fraction,
        1,
        0,
        particle,
        grid_size[0] * grid_size[1],
        grid_size,
        cnum,
        False,
    )

    volume_fraction = cell_volume_fraction.to_numpy().reshape(4, 4, order="F")
    assert volume_fraction[0, 1] == pytest.approx(0.5)
    assert volume_fraction[1, 1] == pytest.approx(0.5)


@pytest.mark.cpu
def test_volume_classification_keeps_particle_occupied_cell_fluid(taichi_runtime):
    cnum = ti.Vector([6, 5])
    grid_size = ti.Vector([0.2, 0.25])
    particle = ParticleCloudIncompressible2D.field(shape=1)
    cell_volume_fraction = ti.field(float, shape=30)
    cell_type = ti.field(dtype=ti.i32, shape=(6, 5), offset=(-1, -1))

    seed_crossing_incompressible_particle(particle, grid_size)
    cell_volume_fraction.fill(0.0)
    cell_type.fill(0)
    kernel_find_fluid_domain_by_volume(
        1,
        1,
        cnum,
        1.0 / grid_size,
        0.2,
        cell_volume_fraction,
        cell_type,
        particle,
    )

    assert cell_type[1, 1] == 1


@pytest.mark.cpu
def test_volume_classification_maps_closed_upper_boundary_to_last_cell(taichi_runtime):
    cnum = ti.Vector([6, 5])
    grid_size = ti.Vector([0.2, 0.25])
    particle = ParticleCloudIncompressible2D.field(shape=1)
    cell_volume_fraction = ti.field(float, shape=30)
    cell_type = ti.field(dtype=ti.i32, shape=(6, 5), offset=(-1, -1))

    seed_upper_boundary_particle(particle, ti.Vector([0.8, 0.75]))
    kernel_find_fluid_domain_by_volume(
        1,
        1,
        cnum,
        1.0 / grid_size,
        0.2,
        cell_volume_fraction,
        cell_type,
        particle,
    )

    assert cell_type[3, 1] == 1


@pytest.mark.cpu
def test_fluid_classification_fills_only_fully_enclosed_air_cell(taichi_runtime):
    cnum = ti.Vector([7, 7])
    cell_type = ti.field(dtype=ti.i32, shape=(7, 7), offset=(-1, -1))
    cell_type.fill(2)
    for i, j in np.ndindex(3, 3):
        cell_type[i + 1, j + 1] = 1
    cell_type[2, 2] = 0
    cell_type[1, 2] = 0
    cell_type[4, 2] = 0

    kernel_fill_enclosed_fluid_cells(1, cnum, cell_type)

    assert cell_type[1, 2] == 0
    assert cell_type[2, 2] == 0
    assert cell_type[4, 2] == 0

    cell_type[1, 2] = 1
    kernel_fill_enclosed_fluid_cells(1, cnum, cell_type)
    assert cell_type[2, 2] == 1


@pytest.mark.cpu
def test_wavemaker_velocity_is_applied_to_the_moving_piston_faces(taichi_runtime):
    velocity = ti.field(float, shape=(3, 3, 3))

    set_moving_piston_velocity(0.25, 0.05, 0.2, 0.2, ti.Vector([0.1, 0.1, 0.1]), velocity)

    values = velocity.to_numpy()
    assert values[0, 0, 0] == pytest.approx(0.25)
    assert values[1, 0, 0] == pytest.approx(0.25)
    assert np.all(values[2] == 0.0)


@pytest.mark.cpu
def test_analytic_piston_plane_prevents_sdf_tunneling(taichi_runtime):
    particle = ParticleCloudIncompressible2D.field(shape=2)
    seed_piston_test_particles(particle)

    enforce_moving_piston_particles(2, 0.1, 0.2, 0.01, particle)

    position = particle.x.to_numpy()
    velocity = particle.v.to_numpy()
    assert position[0, 0] == pytest.approx(0.100001)
    assert velocity[0, 0] == pytest.approx(0.2)
    np.testing.assert_allclose(position[1], [0.11, 0.1])
    np.testing.assert_allclose(velocity[1], [0.05, 0.02])


@pytest.mark.cpu
def test_moving_sdf_wall_uses_relative_normal_velocity(taichi_runtime):
    result = ti.Vector.field(3, float, shape=2)

    @ti.kernel
    def resolve():
        normal = ti.Vector([1.0, 0.0, 0.0])
        result[0] = resolve_solid_normal_velocity(ti.Vector([0.05, 0.02, 0.0]), ti.Vector([0.20, 0.0, 0.0]), normal)
        result[1] = resolve_solid_normal_velocity(ti.Vector([0.05, 0.02, 0.0]), ti.Vector([-0.20, 0.0, 0.0]), normal)

    resolve()
    np.testing.assert_allclose(result.to_numpy(), [[0.20, 0.02, 0.0], [0.05, 0.02, 0.0]])


@pytest.mark.cpu
def test_two_layer_piston_moves_the_solid_cell_boundary(taichi_runtime):
    cell_type = ti.field(ti.i32, shape=(4, 2, 2))
    solid_velocity_x = ti.field(float, shape=(5, 2, 2))

    set_moving_piston_mac_boundary(
        0.2,
        0.151,
        0.2,
        0.2,
        ti.Vector([0.1, 0.1, 0.1]),
        cell_type,
        solid_velocity_x,
    )

    np.testing.assert_array_equal(cell_type.to_numpy()[:, 0, 0], [2, 2, 0, 0])
    np.testing.assert_allclose(solid_velocity_x.to_numpy()[:3, 0, 0], 0.2)
    assert solid_velocity_x[3, 0, 0] == 0.0


def test_wavemaker_ramp_is_derivative_of_zero_net_displacement():
    frequency = 1.0
    amplitude = 0.03
    ramp_time = 1.0
    time = np.linspace(0.0, ramp_time, 10001)
    velocity = np.array([piston_velocity(t, frequency, amplitude, ramp_time) for t in time])

    assert velocity[0] == pytest.approx(0.0)
    assert velocity[-1] == pytest.approx(amplitude)
    assert np.trapezoid(velocity, time) == pytest.approx(0.0, abs=1.0e-10)

    probe = 0.37
    epsilon = 1.0e-6
    derivative = (
        piston_displacement(probe + epsilon, frequency, amplitude, ramp_time)
        - piston_displacement(probe - epsilon, frequency, amplitude, ramp_time)
    ) / (2.0 * epsilon)
    assert derivative == pytest.approx(piston_velocity(probe, frequency, amplitude, ramp_time), rel=1.0e-8)


@pytest.mark.cpu
def test_volume_fraction_level_set_preserves_subcell_surface_distance(taichi_runtime):
    cnum = ti.Vector([5, 5])
    grid_size = ti.Vector([0.1, 0.1])
    cell_volume_fraction = ti.field(float, shape=25)
    cell_type = ti.field(dtype=ti.i32, shape=(5, 5), offset=(-1, -1))
    fluid_sdf = ti.field(float, shape=(5, 5), offset=(-1, -1))
    cell_volume_fraction.fill(0.0)
    cell_type.fill(0)
    cell_type[1, 1] = 1
    cell_type[2, 1] = 1
    cell_volume_fraction[1 + 1 * 5] = 0.9
    cell_volume_fraction[1 + 2 * 5] = 0.2
    cell_volume_fraction[2 + 1 * 5] = 0.4
    cell_volume_fraction[2 + 2 * 5] = 0.49

    kernel_build_fluid_sdf_from_volume_fraction(1, cnum, grid_size, cell_volume_fraction, cell_type, fluid_sdf)

    assert fluid_sdf[1, 1] == pytest.approx(-0.04)
    assert fluid_sdf[1, 2] == pytest.approx(0.03)
    assert fluid_sdf[2, 1] == pytest.approx(-0.01)
    assert fluid_sdf[2, 2] == pytest.approx(0.01)


@pytest.mark.cpu
def test_sdf_initialization_does_not_delete_particle_from_fluid_side_corner_interpolation(taichi_runtime):
    cnum = ti.Vector([4, 4])
    grid_size = ti.Vector([0.1, 0.1])
    solid_sdf = ti.field(float, shape=(4, 4), offset=(-1, -1))
    particle = ParticleCloudIncompressible2D.field(shape=1)
    solid_sdf.fill(-0.1)
    solid_sdf[0, 0] = 0.05
    seed_sdf_test_particle(particle, ti.Vector([0.025, 0.025]))

    deactivated = kernel_deactivate_particles_in_solid_sdf(1, 1, cnum, grid_size, solid_sdf, particle)

    assert deactivated == 0
    assert particle[0].active == 1

    solid_sdf[0, 0] = -0.1
    deactivated = kernel_deactivate_particles_in_solid_sdf(1, 1, cnum, grid_size, solid_sdf, particle)
    assert deactivated == 1
    assert particle[0].active == 0


@pytest.mark.cpu
def test_atomic_p2g_repeatable_and_conserves_mass_2d(taichi_runtime):
    """Small float64, single-thread 2D precision check for production P2G."""
    GlobalVariable.SHAPEFUNCTION = 2
    GlobalVariable.INFLUENCENODE = 3
    GlobalVariable.APIC = True
    GlobalVariable.TPIC = False

    particles_per_axis = 5
    host_cell_count = 12
    particle_count = host_cell_count * particles_per_axis**2
    cnum = ti.Vector([10, 8])
    grid_size = ti.Vector([0.01, 0.01])
    inverse_grid_size = 1.0 / grid_size
    ghost_cell = 1

    particle = ParticleCloudIncompressible2D.field(shape=particle_count)
    particle_lengths = ti.Vector.field(2, dtype=float, shape=1)
    particle_lengths.fill(0.5 * grid_size)
    cell_volume_fraction = ti.field(float, shape=80)
    node = StaggeredGrid(SimpleNamespace(dimension=2, coupling=False), cnum, ghost_cell)
    seed_dense_incompressible_particles(particle, particles_per_axis, grid_size)

    def run_and_snapshot():
        clear_mac_grid(node)
        cell_volume_fraction.fill(0.0)
        kernel_mass_momentum_mac_cell_p2g(
            9,
            particle_count,
            ghost_cell,
            cnum,
            grid_size,
            inverse_grid_size,
            node,
            particle,
            particle_lengths,
            cell_volume_fraction,
            grid_size[0] * grid_size[1],
            False,
        )
        ti.sync()
        return (
            snapshot_mac_grid(node, ghost_cell),
            cell_volume_fraction.to_numpy().copy(),
        )

    first, first_volume_fraction = run_and_snapshot()
    second, second_volume_fraction = run_and_snapshot()
    particle_mass = particle.m.to_numpy()
    expected_mass = float(np.sum(particle_mass, dtype=np.float64))
    expected_volume = float(np.sum(particle.vol.to_numpy(), dtype=np.float64))

    conservation_rtol = 2.0e-13
    if particle_mass.dtype == np.float32:
        conservation_rtol = 1.0e-7

    for (first_mass, first_momentum), (second_mass, second_momentum) in zip(first, second):
        assert np.all(np.isfinite(first_mass))
        assert np.all(np.isfinite(first_momentum))
        np.testing.assert_allclose(second_mass, first_mass, rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(second_momentum, first_momentum, rtol=0.0, atol=1.0e-14)
        assert np.sum(first_mass) == pytest.approx(expected_mass, rel=conservation_rtol, abs=2.0e-14)
    np.testing.assert_allclose(
        second_volume_fraction,
        first_volume_fraction,
        rtol=0.0,
        atol=1.0e-14,
    )
    expected_volume_fraction = expected_volume / (grid_size[0] * grid_size[1])
    assert np.sum(first_volume_fraction) == pytest.approx(expected_volume_fraction, rel=conservation_rtol, abs=2.0e-14)


@pytest.mark.cpu
def test_incompressible_restart_restores_coupling_and_affine_gradient(taichi_runtime):
    particle = ParticleCloudIncompressible2D.field(shape=1)
    is_rigid = ti.field(dtype=int, shape=1)
    gradient = np.array([[[1.0, 2.0], [3.0, 4.0]]])

    kernel_rebulid_incompressible_particle_2D(
        1,
        particle,
        is_rigid,
        np.array([7]),
        np.array([0]),
        np.array([1]),
        np.array([1]),
        np.array([1], dtype=np.uint8),
        np.array([2.0]),
        np.array([[0.25, 0.75]]),
        np.array([[0.5, -0.5]]),
        np.array([0.002]),
        np.array([3.0]),
        gradient,
        np.array([[0, 1]], dtype=np.uint8),
    )

    assert int(particle[0].coupling) == 1
    np.testing.assert_allclose(particle[0].velocity_gradient.to_numpy(), gradient[0])
