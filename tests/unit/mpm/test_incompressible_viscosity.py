from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.EngineKernel import (
    enforce_boundary_cut_cell,
    density_projection_cell_is_active,
    particle_shifting_is_active_fluid,
    shift_incompressible_particle,
    kernel_apply_mac_viscous_delta,
    kernel_compute_mac_viscous_delta,
    kernel_kinemaitc_mac_cell_g2p,
    interpolate_mac_transport_velocity,
)
from src.mpm.structs.Particle import ParticleCloudIncompressible2D
from src.mpm.structs.StaggeredGrid import StaggeredGrid
import src.utils.GlobalVariable as GlobalVariable

pytestmark = [pytest.mark.unit, pytest.mark.mpm, pytest.mark.cpu]


class Fluid:
    density = 1.0
    viscosity = 0.01


@pytest.mark.parametrize("linear_profile", (False, True))
def test_moving_wall_viscosity_matches_startup_and_steady_couette(taichi_runtime, linear_profile):
    GlobalVariable.DIMENSION = 2
    count, dx, dt_value = 6, 1.0 / 6, 0.01
    node = StaggeredGrid(SimpleNamespace(dimension=2, coupling=False), [count, count])
    cell_type = ti.field(int, shape=(count + 2, count + 2), offset=(-1, -1))
    types = np.full((count + 2, count + 2), 2, dtype=np.int32)
    types[1:-1, 1:-1] = 1
    cell_type.from_numpy(types)
    wall = [ti.field(float, shape=field.shape, offset=(-1, -1)) for field in node.velocity]
    delta = [ti.field(float, shape=field.shape, offset=(-1, -1)) for field in node.velocity]
    node.m[0].fill(1.0)
    node.m[1].fill(1.0)
    moving_wall = np.zeros(wall[0].shape)
    moving_wall[:, -1] = 1.0
    wall[0].from_numpy(moving_wall)
    if linear_profile:
        profile = np.broadcast_to((np.arange(count + 2) - 0.5) * dx, node.velocity[0].shape).copy()
        node.velocity[0].from_numpy(profile)
    dt = ti.field(float, shape=())
    dt[None] = dt_value
    cnum = ti.Vector([count + 2, count + 2])
    # Default remains free slip, even when a tangential wall speed was supplied.
    enforce_boundary_cut_cell(1, cnum, cell_type, wall[0], wall[1], wall[1], node, False)
    assert node.velocity[0][2, count] == pytest.approx(node.velocity[0][2, count - 1])
    enforce_boundary_cut_cell(1, cnum, cell_type, wall[0], wall[1], wall[1], node, True)
    assert node.velocity[0][2, count] + node.velocity[0][2, count - 1] == pytest.approx(2.0)
    before = float(node.velocity[0][2, count - 1])
    kernel_compute_mac_viscous_delta(
        1e-12,
        1,
        cnum,
        ti.Vector([dx, dx]),
        dt,
        Fluid(),
        cell_type,
        cell_type,
        cell_type,
        0,
        delta[0],
        delta[1],
        delta[1],
        node,
        True,
    )
    kernel_apply_mac_viscous_delta(delta[0], delta[1], delta[1], node)
    expected_delta = 0.0 if linear_profile else 2.0 * Fluid.viscosity * dt_value / dx**2
    assert float(node.velocity[0][2, count - 1]) - before == pytest.approx(expected_delta, abs=1e-12)
    assert float(delta[0][2, 2]) == pytest.approx(0.0, abs=1e-12)


def test_no_slip_option_requires_incompressible_cut_cell_boundaries(taichi_runtime):
    from src.mpm.Simulation import Simulation

    sims = Simulation()
    assert not sims.fluid_wall_no_slip
    sims.fluid_wall_no_slip = True
    with pytest.raises(RuntimeError, match="fluid_wall_no_slip requires"):
        sims.validate_configuration(require_solver_parameters=False)


def test_particle_regularization_excludes_free_surface_not_solid_wall(taichi_runtime):
    GlobalVariable.DIMENSION = 2
    types = ti.field(int, shape=(4, 4), offset=(-1, -1))
    types.fill(2)
    types[0, 0] = 1
    types[1, 0] = 1
    types[0, 1] = 1

    @ti.kernel
    def active() -> int:
        return int(
            particle_shifting_is_active_fluid(ti.Vector([0.5, 0.5]), 1, ti.Vector([4, 4]), ti.Vector([1.0, 1.0]), types)
        )

    @ti.kernel
    def pressure_active() -> int:
        return int(density_projection_cell_is_active(ti.Vector([0, 0]), 1, ti.Vector([4, 4]), types, True))

    assert active() == 1
    assert pressure_active() == 0
    types[1, 0] = 0
    assert active() == 0


def test_position_regularization_preserves_affine_velocity_and_fixed_components(taichi_runtime):
    GlobalVariable.DIMENSION = 2
    particle = ti.types.struct(
        x=ti.types.vector(2, float),
        v=ti.types.vector(2, float),
        velocity_gradient=ti.types.matrix(2, 2, float),
        fix_v=ti.types.vector(2, ti.u8),
    ).field(shape=1)
    gradient = np.array([[1.0, 2.0], [-3.0, 4.0]])
    origin, offset, shift = np.array([0.3, 0.4]), np.array([0.1, -0.2]), np.array([0.02, -0.01])
    particle[0].x = origin
    particle[0].v = gradient @ origin + offset
    particle[0].velocity_gradient = gradient

    @ti.kernel
    def move():
        shift_incompressible_particle(0, ti.Vector([0.02, -0.01]), particle)

    move()
    np.testing.assert_allclose(particle[0].v.to_numpy(), gradient @ (origin + shift) + offset, atol=1e-12)
    particle[0].fix_v = [0, 1]
    fixed_velocity = float(particle[0].v[1])
    move()
    assert float(particle[0].v[1]) == fixed_velocity


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("shape,stencil,length_ratio", [(0, 2, 0.0), (1, 3, 0.25), (2, 3, 0.5), (3, 4, 1.0)])
def test_fdm_shifting_volume_is_conservative_and_mirror_symmetric(
    taichi_runtime, monkeypatch, dimension, periodic, shape, stencil, length_ratio
):
    from src.mpm.engines import EngineKernel as kernels
    from src.mpm.structs.Particle import ParticleCloudIncompressible3D

    GlobalVariable.DIMENSION = dimension
    GlobalVariable.SHAPEFUNCTION, GlobalVariable.INFLUENCENODE = shape, stencil
    for name in ("MPMXPBC", "MPMYPBC", "MPMZPBC"):
        monkeypatch.setattr(GlobalVariable, name, periodic)
    count, ghost, dx = 8, 1, 0.125
    gnum = count + 1 + 2 * ghost
    struct = ParticleCloudIncompressible2D if dimension == 2 else ParticleCloudIncompressible3D
    # SceneManager adds this workspace member when particle shifting is enabled.
    particle = ti.types.struct(**dict(struct.members, grad_E2=ti.types.vector(dimension, float))).field(shape=1)
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.vol.fill(0.125)
    lengths = ti.Vector.field(dimension, float, shape=1)
    lengths[0] = [length_ratio * dx] * dimension
    volume = ti.field(float, shape=(gnum**dimension, 1))
    boundary = ti.Vector.field(dimension, ti.u8, shape=(gnum**dimension, 1))
    index = np.stack(np.unravel_index(np.arange(gnum**dimension), (gnum,) * dimension, order="F"), axis=-1)
    side = np.minimum(2, index) - np.minimum(gnum - 1 - index, 2)
    boundary.from_numpy(np.where(side < 0, side + 3, np.where(side > 0, side + 2, 0)).astype(np.uint8)[:, None, :])
    reference = ti.field(float, shape=volume.shape)
    kernels.kernel_fdm_shifting_reference_volume(
        ghost, ti.Vector([gnum] * dimension), ti.Vector([dx] * dimension), lengths, boundary, reference
    )
    reference_array = reference.to_numpy()[:, 0].reshape((gnum,) * dimension, order="F")
    assert reference_array.sum() == pytest.approx((count * dx) ** dimension, abs=1e-12)
    center_index = (ghost + count // 2,) * dimension
    assert reference_array[center_index] == pytest.approx(dx**dimension, abs=1e-12)
    if not periodic:
        wall_index = (ghost,) + center_index[1:]
        assert reference_array[wall_index] == pytest.approx(0.5 * dx**dimension, abs=1e-12)
    deposit = getattr(kernels, f"kernel_volume_p2g_fdm_shifting_on_the_fly_{dimension}d")

    def mapped_volume(position):
        particle[0].x = position
        deposit(
            stencil,
            1,
            ghost,
            ti.Vector([dx] * dimension),
            ti.Vector([1 / dx] * dimension),
            ti.Vector([gnum] * dimension),
            volume,
            particle,
            lengths,
            boundary,
        )
        return volume.to_numpy()[:, 0].reshape((gnum,) * dimension, order="F")

    fields = []
    for x in (0.01, 0.99):
        values = mapped_volume([x] + [0.5] * (dimension - 1))
        assert values.sum() == pytest.approx(0.125, abs=1e-12)
        if periodic:
            values = values[(slice(ghost, ghost + count),) * dimension]
        fields.append(values)
    reflected = np.flip(fields[0], axis=0)
    if periodic:
        reflected = np.roll(reflected, 1, axis=0)
    np.testing.assert_allclose(fields[1], reflected, atol=1e-12, rtol=0)

    # The correction must differentiate precisely the same deposited energy.
    position = np.array([0.01] + [0.47] * (dimension - 1))
    mapped_volume(position)
    types = ti.field(int, shape=(count + 2 * ghost,) * dimension, offset=(-ghost,) * dimension)
    types.fill(1)
    arguments = (
        stencil,
        1,
        ghost,
        ti.Vector([count + 2 * ghost] * dimension),
        ti.Vector([dx] * dimension),
        ti.Vector([1 / dx] * dimension),
        ti.Vector([gnum] * dimension),
    )
    if dimension == 2:
        kernels.kernel_particle_shifting_delta_correction_fdm_on_the_fly_2d(
            *arguments, 0.05, volume, reference, types, particle, lengths, boundary
        )
    else:
        kernels.kernel_compute_particle_shifting_gradient_fdm_on_the_fly_3d(
            *arguments, reference, volume, types, particle, lengths
        )
    actual = particle.grad_E2.to_numpy()[0]
    expected = []
    epsilon = dx / 2**16
    for axis in range(dimension):
        offset = np.eye(dimension)[axis] * epsilon
        high = np.sum(np.maximum(mapped_volume(position + offset) - reference_array, 0.0) ** 2)
        low = np.sum(np.maximum(mapped_volume(position - offset) - reference_array, 0.0) ** 2)
        expected.append((high - low) / (2 * epsilon))
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-6)


@pytest.mark.parametrize(
    "dimension,coupled",
    [
        (2, False),
        pytest.param(3, False, marks=pytest.mark.isolated_dimension(3)),
        pytest.param(3, True, marks=pytest.mark.isolated_dimension(3)),
    ],
)
def test_mac_midpoint_rotation_transport_preserves_momentum_update(taichi_runtime, dimension, coupled):
    from src.mpm.structs.Particle import ParticleCloudIncompressible3D, ImplicitParticleCoupling

    GlobalVariable.DIMENSION = dimension
    GlobalVariable.SHAPEFUNCTION = 2
    GlobalVariable.INFLUENCENODE = 3
    # Binary-exact spacing isolates trajectory error from the legacy MAC
    # kernel's import-time vector argument precision.
    count, dx, dt_value = 16, 1.0 / 16, 0.02
    node = StaggeredGrid(SimpleNamespace(dimension=dimension, coupling=False), [count] * dimension)
    gradient = np.zeros((dimension, dimension))
    gradient[0, 1], gradient[1, 0] = -1.0, 1.0
    center = np.full(dimension, 0.5)
    for d, field in enumerate(node.velocity):
        indices = np.moveaxis(np.indices(field.shape), 0, -1) - 1
        stagger = 0.5 * (1 - np.eye(dimension)[d])
        values = ((indices + stagger) * dx - center) @ gradient[d]
        field.from_numpy(values)
    struct = (
        (ImplicitParticleCoupling if coupled else ParticleCloudIncompressible3D)
        if dimension == 3
        else ParticleCloudIncompressible2D
    )
    particle = struct.field(shape=2)
    positions = np.full((2, dimension), 0.45)
    positions[:, 0] = [0.3, 0.6]
    particle.x.from_numpy(positions)
    particle.active.fill(1)
    particle.materialID.fill(1)
    if coupled:
        particle.coupling.fill(1)
    lengths = ti.Vector.field(dimension, float, shape=1)
    lengths[0] = [0.5 * dx] * dimension
    dt = ti.field(float, shape=())
    dt[None] = dt_value
    initial_velocity = (positions - center) @ gradient.T
    kernel_kinemaitc_mac_cell_g2p(
        3**dimension,
        1.0,
        dt,
        2,
        1,
        ti.Vector([count + 2] * dimension),
        ti.Vector([dx] * dimension),
        ti.Vector([1 / dx] * dimension),
        node,
        particle,
        lengths,
    )
    expected_displacement = dt_value * initial_velocity + 0.5 * dt_value**2 * (initial_velocity @ gradient.T)
    np.testing.assert_allclose(particle.x.to_numpy(), positions + expected_displacement, atol=1e-12, rtol=0)
    np.testing.assert_allclose(particle.v.to_numpy(), initial_velocity, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        particle.velocity_gradient.to_numpy(), np.broadcast_to(gradient, (2, dimension, dimension)), atol=1e-12
    )
    if coupled:
        np.testing.assert_allclose(particle.verletDisp.to_numpy(), expected_displacement, atol=1e-12, rtol=0)

    # A fixed component must guide both midpoint construction and displacement,
    # while free components retain the original mixed PIC/FLIP velocity update.
    particle.x.from_numpy(positions)
    old_velocity = np.broadcast_to(0.25 / 2.0 ** np.arange(dimension), positions.shape).copy()
    particle.v.from_numpy(old_velocity)
    fixed = np.zeros_like(positions, dtype=np.uint8)
    fixed[0, 0] = 1
    particle.fix_v.from_numpy(fixed)
    if coupled:
        particle.verletDisp.fill(0)
    for field in node.force:
        field.fill(0.125)
    kernel_kinemaitc_mac_cell_g2p(
        3**dimension,
        0.25,
        dt,
        2,
        1,
        ti.Vector([count + 2] * dimension),
        ti.Vector([dx] * dimension),
        ti.Vector([1 / dx] * dimension),
        node,
        particle,
        lengths,
    )
    free = 1 - fixed
    predictor = initial_velocity * free + old_velocity * fixed
    transport = initial_velocity + 0.5 * dt_value * (predictor @ gradient.T)
    displacement = dt_value * (transport * free + old_velocity * fixed)
    momentum_velocity = (0.25 * initial_velocity + 0.75 * (old_velocity + 0.125)) * free + old_velocity * fixed
    np.testing.assert_allclose(particle.x.to_numpy(), positions + displacement, atol=1e-12, rtol=0)
    np.testing.assert_allclose(particle.v.to_numpy(), momentum_velocity, atol=1e-12, rtol=0)
    if coupled:
        np.testing.assert_allclose(particle.verletDisp.to_numpy(), displacement, atol=1e-12, rtol=0)


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
@pytest.mark.parametrize("periodic", [False, True])
def test_mac_transport_commutes_with_divergence_including_box_boundary(
    taichi_runtime, monkeypatch, dimension, periodic
):
    GlobalVariable.DIMENSION = dimension
    for name in ("MPMXPBC", "MPMYPBC", "MPMZPBC"):
        monkeypatch.setattr(GlobalVariable, name, periodic)
    count, dx = 16, 1.0 / 16
    node = StaggeredGrid(SimpleNamespace(dimension=dimension, coupling=False), [count] * dimension)
    for d, field in enumerate(node.velocity):
        index = np.moveaxis(np.indices(field.shape), 0, -1) - 1
        x, y = index[..., 0] * dx, index[..., 1] * dx
        z_factor = np.cos(np.pi * (index[..., 2] + 0.5) * dx) if dimension == 3 else 1.0
        if d == 0:
            values = np.sin(np.pi * x) ** 2 * (np.sin(2 * np.pi * (y + dx)) ** 2 - np.sin(2 * np.pi * y) ** 2) / dx
        elif d == 1:
            values = -(np.sin(np.pi * (x + dx)) ** 2 - np.sin(np.pi * x) ** 2) * np.sin(2 * np.pi * y) ** 2 / dx
        else:
            values = np.zeros(field.shape)
        field.from_numpy(values * z_factor)
    # Sample both the interior and outer half-cells, where the old isotropic
    # interpolation does not inherit the discrete curl field's zero divergence.
    locations = np.random.default_rng(481).uniform(0.001, 0.999, (96, dimension))
    locations[:32, 0] = 0.01
    locations[32:64, 1] = 0.99
    points = ti.Vector.field(dimension, float, shape=len(locations))
    points.from_numpy(locations)
    divergence = ti.field(float, shape=len(locations))
    wall_normal_velocity = ti.field(float, shape=2 * dimension)

    @ti.kernel
    def measure():
        for p in points:
            value = 0.0
            for d in ti.static(range(dimension)):
                offset = 1e-6 * ti.Vector.unit(dimension, d)
                upper = interpolate_mac_transport_velocity(
                    points[p] + offset, 1, ti.Vector([count + 2] * dimension), ti.Vector([dx] * dimension), node
                )
                lower = interpolate_mac_transport_velocity(
                    points[p] - offset, 1, ti.Vector([count + 2] * dimension), ti.Vector([dx] * dimension), node
                )
                value += (upper[d] - lower[d]) / 2e-6
            divergence[p] = value
        for d in ti.static(range(dimension)):
            for side in ti.static((0, 1)):
                point = ti.Vector([0.37] * dimension)
                point[d] = float(side)
                velocity = interpolate_mac_transport_velocity(
                    point, 1, ti.Vector([count + 2] * dimension), ti.Vector([dx] * dimension), node
                )
                wall_normal_velocity[2 * d + side] = velocity[d]

    measure()
    np.testing.assert_allclose(divergence.to_numpy(), 0.0, atol=2e-8, rtol=0)
    if not periodic:
        np.testing.assert_allclose(wall_normal_velocity.to_numpy(), 0.0, atol=1e-12)
