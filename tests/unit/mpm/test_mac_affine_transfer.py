"""MAC APIC moments must remain distinct from Taylor velocity gradients."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.EngineKernel import (
    kernel_compute_mac_grid_acceleration,
    kernel_compute_mac_grid_velocity,
    kernel_compute_mac_grid_velocity_gravity,
    kernel_kinemaitc_mac_cell_g2p,
    kernel_mass_momentum_mac_cell_p2g,
)
from src.mpm.structs.Particle import ParticleCloudIncompressible2D, ParticleCloudIncompressible3D
from src.mpm.structs.StaggeredGrid import StaggeredGrid
from src.mpm.Simulation import Simulation
import src.utils.GlobalVariable as G

pytestmark = [pytest.mark.unit, pytest.mark.mpm]
DIMENSIONS = [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))]


def setup_transfer(monkeypatch, dimension, scheme, spacing, position):
    monkeypatch.setattr(G, "DIMENSION", dimension)
    monkeypatch.setattr(G, "SHAPEFUNCTION", 2)
    monkeypatch.setattr(G, "INFLUENCENODE", 3)
    monkeypatch.setattr(G, "Affine", scheme == "Affine")
    monkeypatch.setattr(G, "Taylor", scheme == "Taylor")
    for name in ("MPMXPBC", "MPMYPBC", "MPMZPBC"):
        monkeypatch.setattr(G, name, False)
    cnum = ti.Vector([14] * dimension)
    spacing = ti.Vector(spacing)
    node = StaggeredGrid(SimpleNamespace(dimension=dimension, coupling=False), cnum, 1)
    struct = ParticleCloudIncompressible2D if dimension == 2 else ParticleCloudIncompressible3D
    particle = struct.field(shape=1)
    particle[0].active = 1
    particle[0].materialID = 1
    particle[0].m = 2.0
    particle[0].vol = 0.002
    particle[0].x = position
    lengths = ti.Vector.field(dimension, float, shape=1)
    lengths[0] = 0.5 * spacing
    dt = ti.field(float, shape=())

    def g2p(alpha=1.0):
        kernel_kinemaitc_mac_cell_g2p(
            3**dimension, alpha, dt, 1, 1, cnum, spacing, 1.0 / spacing, node, particle, lengths
        )

    return node, particle, lengths, cnum, spacing, g2p


@ti.kernel
def fill_affine(node: ti.template(), spacing: ti.template(), matrix: ti.template(), offset: ti.template()):
    for d in ti.static(range(G.DIMENSION)):
        for I in ti.grouped(node.velocity[d]):
            stagger = 0.5 * (1 - ti.Vector.unit(G.DIMENSION, d))
            position = (I + stagger) * spacing
            velocity = matrix @ position + offset
            node.velocity[d][I] = velocity[d]


@ti.kernel
def fill_nonlinear(node: ti.template(), spacing: ti.template(), all_components: ti.template()):
    for d in ti.static(range(G.DIMENSION)):
        for I in ti.grouped(node.velocity[d]):
            if ti.static(all_components):
                tangent = ti.static((d + 1) % G.DIMENSION)
                stagger = 0.5 * (1 - ti.Vector.unit(G.DIMENSION, d))
                position = (I + stagger) * spacing
                node.velocity[d][I] = (position[tangent] / spacing[tangent] - 4.3) ** 2 + 0.1 * position[d] / spacing[d]
            elif ti.static(d == 0):
                node.velocity[d][I] = (I[1] - 4.0) ** 2


@pytest.mark.parametrize("dimension", DIMENSIONS)
@pytest.mark.parametrize("scheme", ["Affine", "Taylor", None])
def test_mac_reconstruction_selects_moments_or_gradient(taichi_runtime, monkeypatch, dimension, scheme):
    position = [4.5, 4.75, 4.5][:dimension]
    node, particle, _, _, spacing, g2p = setup_transfer(monkeypatch, dimension, scheme, [1.0] * dimension, position)
    fill_nonlinear(node, spacing, False)
    g2p()
    matrix = particle.velocity_gradient.to_numpy()[0]
    expected = np.zeros((dimension, dimension))
    # For xp=.25, QBSpline weights=(1,22,9)/32 and D=1/4.
    # Sum(w*v*dx)/D=11/16, whereas Sum(v*grad(w))=1/2.
    expected[0, 1] = 0.6875 if scheme == "Affine" else 0.5
    np.testing.assert_allclose(matrix, expected, atol=2e-12, rtol=0)
    np.testing.assert_allclose(particle.v.to_numpy()[0], [0.3125] + [0.0] * (dimension - 1), atol=2e-12)


@pytest.mark.parametrize("dimension", DIMENSIONS)
@pytest.mark.parametrize("scheme", ["Affine", "Taylor"])
@pytest.mark.parametrize("near_wall,scale", [(False, 1.0), (True, 1e-6)])
def test_mac_reconstruction_reproduces_affine_fields(taichi_runtime, monkeypatch, dimension, scheme, near_wall, scale):
    spacing = scale * np.array([1.0, 1.3, 0.8][:dimension])
    position = spacing * (
        np.array([0.05, 0.07, 0.09][:dimension]) if near_wall else np.array([4.25, 4.75, 4.1][:dimension])
    )
    matrix = np.array([[0.1, 0.2, -0.1], [-0.3, 0.4, 0.2], [0.2, -0.1, 0.3]])[:dimension, :dimension]
    offset = scale * np.array([-0.3, 0.2, 0.1][:dimension])
    node, particle, _, _, spacing, g2p = setup_transfer(monkeypatch, dimension, scheme, spacing, position)
    fill_affine(node, spacing, ti.Matrix(matrix), ti.Vector(offset))
    g2p()
    np.testing.assert_allclose(particle.v.to_numpy()[0], matrix @ position + offset, atol=2e-12 * scale)
    np.testing.assert_allclose(particle.velocity_gradient.to_numpy()[0], matrix, atol=2e-12, rtol=0)


@pytest.mark.parametrize("dimension", DIMENSIONS)
@pytest.mark.parametrize("scheme", ["Affine", "Taylor"])
@pytest.mark.parametrize("gravity_predictor", [False, True])
def test_mac_cutoff_preserves_translation_and_zero_flip_increment(
    taichi_runtime, monkeypatch, dimension, scheme, gravity_predictor
):
    dx, clearance, cutoff = 0.0025, 1e-8, 3.6015625e-11
    position = [3.31 * dx] * dimension
    position[-1] = 12 * dx - clearance
    node, particle, lengths, cnum, spacing, g2p = setup_transfer(
        monkeypatch, dimension, scheme, [dx] * dimension, position
    )
    velocity = np.arange(1, dimension + 1, dtype=float)
    particle.m[0], particle.v[0] = 2.5e-5, velocity
    fractions = ti.field(float, shape=14**dimension)
    kernel_mass_momentum_mac_cell_p2g(
        3**dimension, 1, 1, cnum, spacing, 1.0 / spacing, node, particle, lengths, fractions, dx**dimension, False
    )
    assert any(np.any((m.to_numpy() > 0) & (m.to_numpy() <= cutoff)) for m in node.m)
    if gravity_predictor:
        for axis, mass in enumerate(node.m):
            values = mass.to_numpy()
            node.force[axis].from_numpy(((values > 0) & (values <= cutoff)).astype(float))
        dt = ti.field(float, shape=())
        dt[None] = 1e-5
        kernel_compute_mac_grid_velocity_gravity(cutoff, ti.Vector([0.0] * dimension), dt, node)
    else:
        kernel_compute_mac_grid_velocity(cutoff, node)
    kernel_compute_mac_grid_acceleration(cutoff, node)
    for alpha in (0.0, 1.0):
        particle.v[0] = velocity
        g2p(alpha)
        np.testing.assert_allclose(particle.v.to_numpy()[0], velocity, atol=2e-12, rtol=0)
        # The wall-normal APIC moment has condition number O(dx / clearance).
        tolerance = 32 * np.finfo(float).eps * np.max(velocity) / clearance
        np.testing.assert_allclose(particle.velocity_gradient.to_numpy()[0], 0, atol=tolerance, rtol=0)


@ti.kernel
def grid_momenta(node: ti.template(), spacing: ti.template(), result: ti.template()):
    for d in ti.static(range(G.DIMENSION)):
        for I in ti.grouped(node.velocity[d]):
            position = (I + 0.5 * (1 - ti.Vector.unit(G.DIMENSION, d))) * spacing
            momentum = node.velocity[d][I]
            result[d] += momentum
            result[G.DIMENSION + d] += node.m[d][I]
            if ti.static(G.DIMENSION == 2):
                result[4] += (-position[1] if d == 0 else position[0]) * momentum
            else:
                angular = position.cross(ti.Vector.unit(3, d) * momentum)
                for k in ti.static(range(3)):
                    result[6 + k] += angular[k]


def quadratic_weight(distance):
    distance = abs(distance)
    return 0.75 - distance**2 if distance < 0.5 else 0.5 * max(0.0, 1.5 - distance) ** 2


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_mac_apic_round_trip_conserves_linear_and_angular_momentum(taichi_runtime, monkeypatch, dimension):
    spacing = np.array([0.2, 0.3, 0.15][:dimension])
    position = spacing * np.array([4.25, 4.75, 4.1][:dimension])
    node, particle, lengths, cnum, grid_size, g2p = setup_transfer(monkeypatch, dimension, "Affine", spacing, position)
    fill_nonlinear(node, grid_size, True)
    g2p()
    expected = np.zeros(5 if dimension == 2 else 9)
    for d in range(dimension):
        stagger = 0.5 * (1 - np.eye(dimension)[d])
        base = np.floor(position / spacing - 0.5 - stagger).astype(int)
        for offset in product(range(3), repeat=dimension):
            coord = (base + offset + stagger) * spacing
            weight = np.prod([quadratic_weight(x) for x in (position - coord) / spacing])
            tangent = (d + 1) % dimension
            velocity = (coord[tangent] / spacing[tangent] - 4.3) ** 2 + 0.1 * coord[d] / spacing[d]
            momentum = 2.0 * weight * velocity
            expected[d] += momentum
            expected[dimension + d] += 2.0 * weight
            if dimension == 2:
                expected[4] += (-coord[1] if d == 0 else coord[0]) * momentum
            else:
                expected[6:] += np.cross(coord, np.eye(3)[d] * momentum)
    node.grid_reset(0.0)
    volume_fraction = ti.field(float, shape=14**dimension)
    kernel_mass_momentum_mac_cell_p2g(
        3**dimension,
        1,
        1,
        cnum,
        grid_size,
        1.0 / grid_size,
        node,
        particle,
        lengths,
        volume_fraction,
        float(np.prod(spacing)),
        False,
    )
    result = ti.field(float, shape=len(expected))
    grid_momenta(node, grid_size, result)
    np.testing.assert_allclose(result.to_numpy(), expected, atol=3e-12, rtol=2e-12)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_mac_apic_singular_boundary_moment_is_finite(taichi_runtime, monkeypatch, dimension):
    node, particle, _, _, spacing, g2p = setup_transfer(
        monkeypatch, dimension, "Affine", [1e-6] * dimension, [0.0] * dimension
    )
    fill_affine(node, spacing, ti.Matrix(np.zeros((dimension, dimension))), ti.Vector([1.0] * dimension))
    g2p()
    np.testing.assert_allclose(particle.v.to_numpy()[0], 1.0, atol=2e-12)
    # Match the shared affine solver's zero-matrix policy for singular support.
    np.testing.assert_array_equal(particle.velocity_gradient.to_numpy()[0], np.zeros((dimension, dimension)))


@pytest.mark.parametrize("dimension", DIMENSIONS)
@pytest.mark.parametrize("alpha,fixed", [(0.0, False), (0.01, False), (1.0, False), (0.0, True)])
def test_mac_affine_blend_uses_moments_independently_of_alpha(taichi_runtime, monkeypatch, dimension, alpha, fixed):
    node, particle, _, _, spacing, g2p = setup_transfer(
        monkeypatch, dimension, "Affine", [1.0] * dimension, [4.5, 4.75, 4.5][:dimension]
    )
    sims = Simulation()
    sims.set_alpha(alpha)
    sims.set_velocity_projection_scheme("Affine")
    assert sims.alphaPIC == alpha and G.Affine and not G.Taylor
    sims.solver_type, sims.material_type, sims.discretization = "Implicit", "Fluid", "FDM"
    sims.linear_solver = "MGPCG"
    sims.validate_configuration()
    fill_nonlinear(node, spacing, False)
    initial = np.arange(dimension, dtype=float) + 2.0
    increments = np.arange(dimension, dtype=float) + 0.25
    particle[0].v = initial
    particle[0].fix_v = [int(fixed)] + [0] * (dimension - 1)
    for d in range(dimension):
        node.force[d].fill(increments[d])
    g2p(sims.alphaPIC)
    expected = alpha * np.array([0.3125] + [0.0] * (dimension - 1)) + (1 - alpha) * (initial + increments)
    if fixed:
        expected[0] = initial[0]
    np.testing.assert_allclose(particle.v.to_numpy()[0], expected, atol=2e-12, rtol=0)
    matrix = np.zeros((dimension, dimension))
    matrix[0, 1] = 0.6875
    np.testing.assert_allclose(particle.velocity_gradient.to_numpy()[0], matrix, atol=2e-12, rtol=0)


def test_mac_apic_smooth_mode_has_the_expected_transfer_filter(taichi_runtime, monkeypatch):
    node, _, lengths, cnum, spacing, _ = setup_transfer(monkeypatch, 2, "Affine", [1.0, 1.0], [4.5, 4.75])
    positions = np.array(list(product(np.arange(0.25, 12.0, 0.5), repeat=2)))
    count = len(positions)
    particles = ParticleCloudIncompressible2D.field(shape=count)
    particles.active.fill(1)
    particles.materialID.fill(1)
    particles.m.fill(0.25)
    particles.vol.fill(0.25)
    particles.x.from_numpy(positions)
    dt = ti.field(float, shape=())
    wavelength = 16.0

    @ti.kernel
    def fill_mode():
        for I in ti.grouped(node.velocity[0]):
            node.velocity[0][I] = ti.cos(2.0 * np.pi * (I[1] + 0.5) / wavelength)

    fill_mode()
    kernel_kinemaitc_mac_cell_g2p(9, 1.0, dt, count, 1, cnum, spacing, 1.0 / spacing, node, particles, lengths)
    node.grid_reset(0.0)
    fractions = ti.field(float, shape=14**2)
    kernel_mass_momentum_mac_cell_p2g(
        9, count, 1, cnum, spacing, 1.0 / spacing, node, particles, lengths, fractions, 1.0, False
    )
    # Uniform two-particle quadrature in each direction, away from boundaries.
    # This is W.T W + 4 (W*r).T (W*r), normalized by face mass.
    phase = 2.0 * np.pi / wavelength
    offsets = np.array([-1.25, -0.25, 0.75])
    weights = np.array([quadratic_weight(x) for x in offsets])
    sample = np.exp(1j * phase * offsets)
    gain = abs(weights @ sample) ** 2 + 4.0 * abs((weights * offsets) @ sample) ** 2
    assert 0.0 < gain < 1.0
    for j in range(3, 9):
        actual = node.velocity[0][4, j] / node.m[0][4, j]
        expected = gain * np.cos(phase * (j + 0.5))
        assert actual == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize("alpha", [0.0, 0.001, 0.01, 1.0])
def test_mac_pic_blend_has_the_expected_shear_mode_damping(taichi_runtime, monkeypatch, alpha):
    node, _, lengths, cnum, spacing, _ = setup_transfer(monkeypatch, 2, None, [1.0, 1.0], [4.5, 4.5])
    positions = np.array(list(product(np.arange(0.5, 12.0), repeat=2)))
    count = len(positions)
    particles = ParticleCloudIncompressible2D.field(shape=count)
    particles.active.fill(1)
    particles.materialID.fill(1)
    particles.m.fill(1.0)
    particles.vol.fill(1.0)
    particles.x.from_numpy(positions)
    phase = 2 * np.pi / 16
    initial = np.column_stack((np.cos(phase * positions[:, 1]), np.zeros(count)))
    particles.v.from_numpy(initial)
    fractions = ti.field(float, shape=14**2)
    node.grid_reset(0.0)
    kernel_mass_momentum_mac_cell_p2g(
        9, count, 1, cnum, spacing, 1.0 / spacing, node, particles, lengths, fractions, 1.0, False
    )

    @ti.kernel
    def normalize_grid():
        for d in ti.static(range(2)):
            for I in ti.grouped(node.velocity[d]):
                if node.m[d][I] > 0:
                    node.velocity[d][I] /= node.m[d][I]

    normalize_grid()
    dt = ti.field(float, shape=())
    kernel_kinemaitc_mac_cell_g2p(9, alpha, dt, count, 1, cnum, spacing, 1.0 / spacing, node, particles, lengths)
    # The centered quadratic B-spline stencil is (1/8, 3/4, 1/8).
    # A transfer round trip squares its Fourier gain; FLIP has no force increment here.
    gain = (1 - alpha) + alpha * (0.75 + 0.25 * np.cos(phase)) ** 2
    interior = np.all((positions > 3) & (positions < 9), axis=1)
    np.testing.assert_allclose(particles.v.to_numpy()[interior], gain * initial[interior], atol=2e-12, rtol=0)
