from types import SimpleNamespace

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.mpdem.fluid_dynamics.DragForceModel import DragForce
from src.mpdem.Engine import cache_dem_external_load, restore_dem_external_load
from src.mpdem.fluid_dynamics.IncompressibleCoupling import (
    cell_pressure_gradient_3d,
    estimate_cell_solid_fraction_3d,
    lsdem_volume_fraction_ibm_force_density,
    partitioned_body_share,
)
from src.mpdem.fluid_dynamics.IncompressibleCouplingKernel import kernel_ibm_velocity_l2_error
from src.mpdem.fluid_dynamics.IncompressibleSemiResolved import (
    expanded_domain_gaussian,
    kernel_apply_cell_drag_force_to_mac_velocity,
    kernel_apply_integrated_added_mass,
    kernel_apply_sphere_plane_wall_lubrication,
    kernel_compute_incompressible_sphere_drag,
    kernel_update_sphere_cell_solid_fraction,
    pressure_gradient_3d,
)
from src.mpm.engines.AssembleMatrixKernel import (
    kernel_assemble_poisson_equation_coupled_3d,
    kernel_poisson_equation_cg_coupled_3d,
)
from src.mpm.engines.EngineKernel import (
    kernel_apply_incompressible_ibm_mac_source,
    kernel_apply_mac_viscous_delta,
    kernel_assemble_incompressible_mg_A_level0_coupled_3d,
    kernel_compute_mac_viscous_delta,
    kernel_correct_velocity_coupled_3d,
)
from src.mpm.structs.StaggeredGrid import StaggeredGrid
import src.utils.GlobalVariable as GlobalVariable

pytestmark = [pytest.mark.verification, pytest.mark.mpdem, pytest.mark.cpu]


@ti.data_oriented
class FluidProperties:
    def __init__(self, density=1000.0, viscosity=1.0e-3, atmospheric_pressure=0.0):
        self.density = density
        self.viscosity = viscosity
        self.atmospheric_pressure = atmospheric_pressure


@ti.dataclass
class AnalyticSphereSDF:
    radius: float

    @ti.func
    def _in_box(self, point):
        return True

    @ti.func
    def distance(self, point, levelset_grid):
        return point.norm() - self.radius


@ti.dataclass
class SphereParticleFixture:
    x: ti.types.vector(3, float)
    rad: float
    m: float
    v: ti.types.vector(3, float)
    contact_force: ti.types.vector(3, float)
    contact_torque: ti.types.vector(3, float)


@ti.dataclass
class PlaneWallFixture:
    active: ti.u8
    point: ti.types.vector(3, float)
    norm: ti.types.vector(3, float)


def test_dem_external_load_cache_restores_only_active_bodies(taichi_runtime):
    bodies = SphereParticleFixture.field(shape=2)
    force = ti.Vector.field(3, float, shape=2)
    torque = ti.Vector.field(3, float, shape=2)
    bodies[0].contact_force = [1.0, 2.0, 3.0]
    bodies[0].contact_torque = [-1.0, 0.0, 4.0]
    bodies[1].contact_force = [9.0, 9.0, 9.0]
    cache_dem_external_load(1, bodies, force, torque)
    bodies[0].contact_force = [0.0, 0.0, 0.0]
    bodies[0].contact_torque = [0.0, 0.0, 0.0]
    restore_dem_external_load(1, bodies, force, torque)
    np.testing.assert_array_equal(bodies.contact_force.to_numpy(), [[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]])
    np.testing.assert_array_equal(bodies[0].contact_torque.to_numpy(), [-1.0, 0.0, 4.0])


def test_integrated_added_mass_changes_inertia_without_changing_weight(taichi_runtime):
    particle = SphereParticleFixture.field(shape=1)
    sphere = SphereIndexFixture.field(shape=1)
    radius, particle_density, fluid_density, coefficient = 0.1, 1200.0, 1000.0, 2.0
    volume = 4.0 / 3.0 * np.pi * radius**3
    particle[0].rad = radius
    particle[0].m = particle_density * volume
    particle[0].contact_force = [3.0, -2.0, 5.0]
    sphere[0].sphereIndex = 0
    gravity = np.array([0.0, 0.0, -9.81])
    original_total = particle[0].contact_force.to_numpy() + particle[0].m * gravity

    kernel_apply_integrated_added_mass(1, coefficient, fluid_density, gravity, particle, sphere)

    transformed_total = particle[0].contact_force.to_numpy() + particle[0].m * gravity
    expected_ratio = particle_density / (particle_density + coefficient * fluid_density)
    np.testing.assert_allclose(transformed_total, expected_ratio * original_total, rtol=2e-7, atol=3e-6)


def test_plane_wall_lubrication_opposes_unresolved_normal_motion(taichi_runtime):
    particle = SphereParticleFixture.field(shape=1)
    sphere = SphereIndexFixture.field(shape=1)
    wall = PlaneWallFixture.field(shape=1)
    particle[0].x = [0.0, 0.0, 0.75]
    particle[0].rad = 0.5
    particle[0].v = [0.0, 0.0, -0.2]
    sphere[0].sphereIndex = 0
    wall[0].active = 1
    wall[0].point = [0.0, 0.0, 0.0]
    wall[0].norm = [0.0, 0.0, 1.0]

    kernel_apply_sphere_plane_wall_lubrication(1, 1, 0.1, 0.5, 0.01, particle, sphere, wall)

    expected = 6.0 * np.pi * 0.1 * 0.5**2 * 0.2 * (1.0 / 0.25 - 1.0 / 0.5)
    np.testing.assert_allclose(particle[0].contact_force.to_numpy(), [0.0, 0.0, expected], rtol=2e-7)


@ti.dataclass
class SphereIndexFixture:
    sphereIndex: int


@ti.kernel
def evaluate_drag(model: ti.template(), result: ti.template()):
    result[0] = model.drag_law(
        0.6,
        1000.0,
        1.0e-3,
        0.05,
        ti.Vector([0.0, 0.0, 0.0]),
    )


@ti.kernel
def evaluate_expanded_gaussian(result: ti.template()):
    radius = 0.1
    support_size = 6
    center_weight = expanded_domain_gaussian(radius, ti.Vector([0.0, 0.0, 0.0]), support_size)
    result[0] = (
        expanded_domain_gaussian(radius, ti.Vector([support_size * radius, 0.0, 0.0]), support_size) / center_weight
    )


@ti.kernel
def evaluate_eq28_force_density(result: ti.template()):
    result[0] = lsdem_volume_fraction_ibm_force_density(
        0.25,
        1000.0,
        2500.0,
        ti.Vector([0.5, 0.0, 0.0]),
        ti.Vector([-12500.0, 0.0, 0.0]),
    )


@ti.kernel
def evaluate_body_partition(result: ti.template()):
    result[0] = partitioned_body_share(0.8, 1.5)
    result[1] = partitioned_body_share(0.7, 1.5)


@ti.kernel
def evaluate_free_surface_pressure_gradient(
    pressure: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    result: ti.template(),
):
    result[0] = cell_pressure_gradient_3d(
        ti.Vector([1, 1, 1]),
        ti.Vector([3, 3, 3]),
        ti.Vector([1.0, 1.0, 1.0]),
        0.0,
        pressure,
        surface_tension,
        cell_type,
        fluid_sdf,
        True,
    )


@ti.kernel
def initialize_linear_pressure(n: int, dx: float, pressure: ti.template(), cell_type: ti.template()):
    pressure_gradient = ti.Vector([-2.0, 1.0, -0.5])
    for i, j, k in ti.ndrange((0, n), (0, n), (0, n)):
        cell = ti.Vector([i, j, k])
        pressure[cell] = pressure_gradient.dot((cell.cast(float) + 0.5) * dx)
        cell_type[cell] = 1


@ti.kernel
def evaluate_semi_resolved_pressure_gradient(
    atmospheric_pressure: float,
    pressure: ti.template(),
    cell_type: ti.template(),
    result: ti.template(),
):
    result[0] = pressure_gradient_3d(
        ti.Vector([1, 1, 1]),
        ti.Vector([3, 3, 3]),
        ti.Vector([1.0, 1.0, 1.0]),
        atmospheric_pressure,
        pressure,
        cell_type,
    )


@ti.kernel
def integrate_sphere_sdf_and_eq28_force(
    n: int,
    dx: float,
    box: ti.template(),
    levelset_grid: ti.template(),
    pressure: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    result: ti.template(),
):
    grid_size = ti.Vector([dx, dx, dx])
    mass_center = ti.Vector([0.5, 0.5, 0.5])
    rotate_matrix = ti.Matrix.identity(float, 3)
    cell_volume = dx * dx * dx
    for i, j, k in ti.ndrange((0, n), (0, n), (0, n)):
        cell = ti.Vector([i, j, k])
        fraction = estimate_cell_solid_fraction_3d(
            cell, grid_size, dx, mass_center, rotate_matrix, 0, box, levelset_grid
        )
        pressure_gradient = cell_pressure_gradient_3d(
            cell,
            ti.Vector([n, n, n]),
            grid_size,
            0.0,
            pressure,
            surface_tension,
            cell_type,
            fluid_sdf,
            False,
        )
        force_density = lsdem_volume_fraction_ibm_force_density(
            fraction,
            1000.0,
            1000.0,
            -pressure_gradient,
            ti.Vector([0.0, 0.0, 0.0]),
        )
        ti.atomic_add(result[0], fraction * cell_volume)
        for d in ti.static(range(3)):
            ti.atomic_add(result[d + 1], force_density[d] * cell_volume)


def offset_scalar_field(shape):
    return ti.field(dtype=float, shape=shape, offset=(-1, -1, -1))


def make_mac_grid(active_cnum, ghost_cell=0):
    return StaggeredGrid(SimpleNamespace(dimension=3, coupling=False), active_cnum, ghost_cell)


def test_zero_slip_drag_is_exactly_finite_zero(taichi_runtime):
    GlobalVariable.DIMENSION = 3
    result = ti.Vector.field(3, dtype=float, shape=1)
    evaluate_drag(DragForce({}), result)

    np.testing.assert_array_equal(result.to_numpy()[0], np.zeros(3))


def test_drag_closures_match_difelice_exponent_and_gidaspow_dilute_limit(taichi_runtime):
    model = DragForce({"DragForceModel": "SchillerNaumannModel"})
    result = ti.Vector.field(3, float, shape=5)

    @ti.kernel
    def evaluate():
        result[0] = model.quadratic_drag_law(0.5, 1.0, 1.0, 0.5, ti.Vector([64.0, 0.0, 0.0]))
        result[1] = model.linear_drag_law(1.0, 1.0, 1.0, 0.5, ti.Vector([32.0, 0.0, 0.0]))
        result[2] = model.linear_drag_law(1.0 - 1e-8, 1.0, 1.0, 0.5, ti.Vector([32.0, 0.0, 0.0]))
        result[3] = model.linear_drag_law(0.9, 1.0, 1.0, 0.5, ti.Vector([2000.0, 0.0, 0.0]))
        result[4] = ti.Vector([model.EmpiricalModel(32.0), 0.0, 0.0])

    evaluate()
    cd = 24 / 32 * (1 + 0.15 * 32**0.687)
    exponent = 3.7 - 0.65 * np.exp(-0.5 * (1.5 - np.log10(32)) ** 2)
    expected_quadratic = np.pi / 8 * cd * 0.5 ** (2 - exponent) * 64**2
    expected_isolated = np.pi / 8 * cd * 32**2
    expected_high_re = np.pi / 8 * 0.44 * 0.9 ** (-1.65) * 2000**2
    values = result.to_numpy()[:, 0]
    np.testing.assert_allclose(values[[0, 1, 3]], [expected_quadratic, expected_isolated, expected_high_re], rtol=1e-6)
    assert values[2] == pytest.approx(values[1], rel=1e-7)
    assert values[4] == pytest.approx((0.63 + 4.8 / np.sqrt(32)) ** 2, rel=1e-6)


def test_semi_resolved_background_kernel_uses_expanded_bandwidth(taichi_runtime):
    result = ti.field(dtype=float, shape=1)
    evaluate_expanded_gaussian(result)

    assert float(result[0]) == pytest.approx(np.exp(-1.0))


def test_semi_resolved_pressure_distinguishes_solid_and_air_cells(taichi_runtime):
    pressure = ti.field(float, shape=(3, 3, 3))
    cell_type = ti.field(int, shape=(3, 3, 3))
    result = ti.Vector.field(3, float, shape=1)
    cell_type.fill(2)
    cell_type[1, 1, 1] = 1
    pressure[1, 1, 1] = 5.0

    evaluate_semi_resolved_pressure_gradient(0.0, pressure, cell_type, result)
    np.testing.assert_allclose(result.to_numpy()[0], 0.0)

    cell_type[2, 1, 1] = 0
    evaluate_semi_resolved_pressure_gradient(0.0, pressure, cell_type, result)
    np.testing.assert_allclose(result.to_numpy()[0], [-5.0, 0.0, 0.0])

    pressure[1, 1, 1] = 12.0
    evaluate_semi_resolved_pressure_gradient(7.0, pressure, cell_type, result)
    np.testing.assert_allclose(result.to_numpy()[0], [-5.0, 0.0, 0.0])


def test_staggered_grid_reset_clears_inactive_face_state(taichi_runtime):
    node = make_mac_grid([1, 1, 1])
    node.velocity[0][0, 0, 0] = 7.0
    node.force[0][0, 0, 0] = -3.0

    node.grid_reset(1.0e-12)

    assert float(node.velocity[0][0, 0, 0]) == 0.0
    assert float(node.force[0][0, 0, 0]) == 0.0


def test_ibm_velocity_error_kernel_is_callable_from_benchmarks(taichi_runtime):
    node = make_mac_grid([1, 1, 1])
    for d, value in enumerate((1.0, 2.0, -1.0)):
        node.velocity[d].fill(value)
    solid_fraction = ti.field(dtype=float, shape=(1, 1, 1))
    solid_velocity = ti.Vector.field(3, dtype=float, shape=(1, 1, 1))
    solid_fraction.fill(1.0)
    solid_velocity[0, 0, 0] = [0.5, 2.0, -1.0]

    error = kernel_ibm_velocity_l2_error(ti.Vector([1, 1, 1]), solid_fraction, solid_velocity, node)

    assert float(error) == pytest.approx(0.5)


def test_semi_resolved_drag_mapping_conserves_porosity_weighted_impulse(taichi_runtime):
    GlobalVariable.DIMENSION = 3
    node = make_mac_grid([1, 1, 1])
    node.m[0].fill(2.0)
    node.m[0][0, 0, 0] = 0.0
    solid_fraction = ti.field(dtype=float, shape=(1, 1, 1))
    solid_fraction.fill(0.4)
    cell_force = ti.Vector.field(3, dtype=float, shape=(1, 1, 1))
    cell_force[0, 0, 0] = [12.0, 0.0, 0.0]
    dt = ti.field(dtype=float, shape=())
    dt[None] = 0.25

    kernel_apply_cell_drag_force_to_mac_velocity(
        1.0e-12,
        dt,
        ti.Vector([1, 1, 1]),
        node,
        solid_fraction,
        cell_force,
    )

    right_velocity = float(node.velocity[0][1, 0, 0])
    effective_mass = 2.0 * (1.0 - 0.4)
    assert effective_mass * right_velocity == pytest.approx(12.0 * 0.25)
    assert float(node.velocity[0][0, 0, 0]) == 0.0


def test_semi_resolved_drag_samples_centered_affine_mac_field(taichi_runtime):
    node = make_mac_grid([4, 4, 4])
    gradient = np.array([[1.0, 2.0, -1.0], [-2.0, 3.0, 1.0], [1.0, -1.0, -4.0]])
    center = np.full(3, 0.5)
    for d, field in enumerate(node.velocity):
        indices = np.moveaxis(np.indices(field.shape), 0, -1)
        stagger = 0.5 * (1 - np.eye(3)[d])
        field.from_numpy(((indices + stagger) * 0.25 - center) @ gradient[d])
    particle = SphereParticleFixture.field(shape=1)
    particle[0].x = center
    particle[0].rad = 0.125
    sphere = SphereIndexFixture.field(shape=1)
    fraction = ti.field(float, shape=(4, 4, 4))
    fraction.fill(0.2)
    force = ti.Vector.field(3, float, shape=(4, 4, 4))
    velocity = ti.Vector.field(3, float, shape=1)
    fluid_fraction = ti.field(float, shape=1)
    weights = ti.field(float, shape=1)
    kernel_compute_incompressible_sphere_drag(
        1,
        6,
        3,
        ti.Vector([4, 4, 4]),
        ti.Vector([0.25] * 3),
        ti.Vector([4.0] * 3),
        FluidProperties(),
        particle,
        sphere,
        node,
        fraction,
        force,
        velocity,
        fluid_fraction,
        weights,
        DragForce({}),
    )
    # Symmetric averaging of this divergence-free affine field is zero. A
    # nearest-face sample shifts each component by half a cell instead.
    np.testing.assert_allclose(velocity.to_numpy()[0] / weights[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(particle.contact_force.to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(force.to_numpy(), 0.0, atol=1e-12)


def test_dense_semi_resolved_gaussian_fraction_preserves_sphere_volume(taichi_runtime):
    GlobalVariable.DIMENSION = 3
    radius = 0.08
    dx = 0.02
    sphere_count = 8
    particle = SphereParticleFixture.field(shape=sphere_count)
    sphere = SphereIndexFixture.field(shape=sphere_count)
    for index, center in enumerate(([x, y, z] for x in (0.24, 0.40) for y in (0.24, 0.40) for z in (0.24, 0.40))):
        particle[index].x = center
        particle[index].rad = radius
        sphere[index].sphereIndex = index

    solid_fraction = ti.field(dtype=float, shape=(32, 32, 32))
    kernel_volume = ti.field(dtype=float, shape=sphere_count)
    kernel_update_sphere_cell_solid_fraction(
        sphere_count,
        3,
        ti.Vector([32, 32, 32]),
        ti.Vector([dx, dx, dx]),
        ti.Vector([1.0 / dx, 1.0 / dx, 1.0 / dx]),
        particle,
        sphere,
        solid_fraction,
        kernel_volume,
    )

    fractions = solid_fraction.to_numpy()
    expected_volume = sphere_count * 4.0 * np.pi * radius**3 / 3.0
    mapped_volume = np.sum(fractions) * dx**3
    assert np.min(fractions) >= 0.0
    assert np.max(fractions) <= 1.0
    assert np.max(fractions) > 0.5
    assert mapped_volume == pytest.approx(expected_volume, rel=1.0e-6)


def test_semi_resolved_poisson_rhs_contains_porosity_time_derivative(taichi_runtime):
    GlobalVariable.DIMENSION = 3
    cnum = ti.Vector([3, 3, 3])
    node = make_mac_grid([3, 3, 3], ghost_cell=1)
    cell_type = ti.field(dtype=int, shape=(3, 3, 3), offset=(-1, -1, -1))
    cell_type.fill(0)
    cell_type[0, 0, 0] = 1
    solid_fraction = offset_scalar_field((3, 3, 3))
    previous_solid_fraction = offset_scalar_field((3, 3, 3))
    solid_density = offset_scalar_field((3, 3, 3))
    solid_fraction.fill(0.0)
    previous_solid_fraction.fill(0.0)
    solid_fraction[0, 0, 0] = 0.4
    previous_solid_fraction[0, 0, 0] = 0.2
    surface_tension = offset_scalar_field((3, 3, 3))
    fluid_sdf = offset_scalar_field((3, 3, 3))
    flag = ti.field(dtype=int, shape=1)
    flag[0] = 0
    rhs = ti.field(dtype=float, shape=1)
    dt = ti.field(dtype=float, shape=())
    dt[None] = 0.1
    props = FluidProperties()

    kernel_assemble_poisson_equation_coupled_3d(
        1,
        cnum,
        ti.Vector([1.0, 1.0, 1.0]),
        dt,
        node,
        flag,
        surface_tension,
        cell_type,
        fluid_sdf,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        previous_solid_fraction,
        solid_density,
        props,
        1,
        False,
        False,
        rhs,
    )

    assert float(rhs[0]) == pytest.approx(20.0)


def test_fully_resolved_projection_uses_face_mixed_density(taichi_runtime):
    GlobalVariable.DIMENSION = 3
    cnum = ti.Vector([4, 3, 3])
    cell_type = ti.field(dtype=int, shape=(4, 3, 3), offset=(-1, -1, -1))
    cell_type.fill(2)
    cell_type[0, 0, 0] = 1
    cell_type[1, 0, 0] = 1
    solid_fraction = ti.field(dtype=float, shape=(4, 3, 3), offset=(-1, -1, -1))
    solid_density = ti.field(dtype=float, shape=(4, 3, 3), offset=(-1, -1, -1))
    solid_fraction.fill(0.0)
    solid_density.fill(3000.0)
    solid_fraction[0, 0, 0] = 0.5
    flag = ti.field(dtype=int, shape=2)
    flag[0] = 0
    flag[1] = 1
    pressure = ti.field(dtype=float, shape=2)
    pressure[0] = 1.0
    pressure[1] = 0.0
    product = ti.field(dtype=float, shape=2)
    fluid_sdf = ti.field(dtype=float, shape=(4, 3, 3), offset=(-1, -1, -1))
    surface_tension = ti.field(dtype=float, shape=(4, 3, 3), offset=(-1, -1, -1))
    props = FluidProperties()

    kernel_poisson_equation_cg_coupled_3d(
        1,
        cnum,
        ti.Vector([1.0, 1.0, 1.0]),
        flag,
        cell_type,
        fluid_sdf,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        solid_density,
        props,
        2,
        False,
        False,
        pressure,
        product,
    )

    expected_coefficient = 1.0 / 1500.0
    np.testing.assert_allclose(product.to_numpy(), [expected_coefficient, -expected_coefficient], rtol=1.0e-12)

    node = make_mac_grid([4, 3, 3], ghost_cell=1)
    pressure_cells = ti.field(dtype=float, shape=(4, 3, 3), offset=(-1, -1, -1))
    pressure_cells[0, 0, 0] = 1.0
    dt = ti.field(dtype=float, shape=())
    dt[None] = 0.2
    kernel_correct_velocity_coupled_3d(
        1,
        cnum,
        ti.Vector([1.0, 1.0, 1.0]),
        dt,
        props,
        pressure_cells,
        surface_tension,
        cell_type,
        fluid_sdf,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        solid_density,
        2,
        False,
        False,
        node,
    )
    assert float(node.velocity[0][1, 0, 0]) == pytest.approx(0.2 / 1500.0)


def test_fully_resolved_mg_level_zero_uses_same_mixed_density(taichi_runtime):
    GlobalVariable.DIMENSION = 3
    grid_type = ti.field(dtype=int, shape=(2, 1, 1))
    grid_type.fill(1)
    cell_type = ti.field(dtype=int, shape=(4, 3, 3), offset=(-1, -1, -1))
    cell_type.fill(2)
    cell_type[0, 0, 0] = 1
    cell_type[1, 0, 0] = 1
    solid_fraction = ti.field(dtype=float, shape=(4, 3, 3), offset=(-1, -1, -1))
    solid_density = ti.field(dtype=float, shape=(4, 3, 3), offset=(-1, -1, -1))
    solid_density.fill(3000.0)
    solid_fraction[0, 0, 0] = 0.5
    fluid_sdf = ti.field(dtype=float, shape=(4, 3, 3), offset=(-1, -1, -1))
    diagonal = ti.field(dtype=float, shape=(2, 1, 1))
    positive_edges = ti.Vector.field(3, dtype=float, shape=(2, 1, 1))
    dt = ti.field(dtype=float, shape=())
    dt[None] = 0.2

    kernel_assemble_incompressible_mg_A_level0_coupled_3d(
        dt,
        ti.Vector([1.0, 1.0, 1.0]),
        FluidProperties(),
        grid_type,
        cell_type,
        fluid_sdf,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        solid_fraction,
        solid_density,
        2,
        False,
        False,
        diagonal,
        positive_edges,
    )

    expected_coefficient = 0.2 / 1500.0
    assert float(positive_edges[0, 0, 0][0]) == pytest.approx(-expected_coefficient)
    assert float(diagonal[0, 0, 0]) == pytest.approx(expected_coefficient)
    assert float(diagonal[1, 0, 0]) == pytest.approx(expected_coefficient)


def test_fully_resolved_ibm_source_uses_light_solid_density(taichi_runtime):
    GlobalVariable.DIMENSION = 3
    node = make_mac_grid([3, 3, 3], ghost_cell=1)
    solid_fraction = offset_scalar_field((3, 3, 3))
    solid_density = offset_scalar_field((3, 3, 3))
    solid_velocity = ti.Vector.field(3, dtype=float, shape=(3, 3, 3), offset=(-1, -1, -1))
    ibm_force = ti.Vector.field(3, dtype=float, shape=(3, 3, 3), offset=(-1, -1, -1))
    cell_type = ti.field(dtype=int, shape=(3, 3, 3), offset=(-1, -1, -1))
    cell_type[0, 0, 0] = 1
    solid_fraction[0, 0, 0] = 0.5
    solid_density[0, 0, 0] = 500.0
    solid_velocity[0, 0, 0] = [3.0, 0.0, 0.0]
    dt = ti.field(dtype=float, shape=())
    dt[None] = 0.1

    kernel_apply_incompressible_ibm_mac_source(
        1,
        ti.Vector([3, 3, 3]),
        dt,
        FluidProperties(),
        cell_type,
        solid_fraction,
        solid_density,
        solid_velocity,
        ibm_force,
        node,
    )

    np.testing.assert_allclose(ibm_force[0, 0, 0], [7500.0, 0.0, 0.0], rtol=1.0e-12)
    assert float(node.velocity[0][0, 0, 0]) == pytest.approx(1.0)
    assert float(node.velocity[0][1, 0, 0]) == pytest.approx(1.0)

    cell_type[0, 0, 0] = 0
    for field in node.velocity:
        field.fill(0.0)
    kernel_apply_incompressible_ibm_mac_source(
        1,
        ti.Vector([3, 3, 3]),
        dt,
        FluidProperties(),
        cell_type,
        solid_fraction,
        solid_density,
        solid_velocity,
        ibm_force,
        node,
    )
    np.testing.assert_array_equal(ibm_force[0, 0, 0], [0.0, 0.0, 0.0])
    assert float(node.velocity[0][0, 0, 0]) == 0.0
    assert float(node.velocity[0][1, 0, 0]) == 0.0


def test_fully_resolved_viscosity_uses_mixed_density(taichi_runtime):
    GlobalVariable.DIMENSION = 3
    node = make_mac_grid([5, 3, 3], ghost_cell=1)
    node.m[0][1, 0, 0] = 1.0
    node.velocity[0][1, 0, 0] = 1.0
    cell_type = ti.field(dtype=int, shape=(5, 3, 3), offset=(-1, -1, -1))
    cell_type.fill(2)
    for i in range(3):
        cell_type[i, 0, 0] = 1
    solid_fraction = ti.field(dtype=float, shape=(5, 3, 3), offset=(-1, -1, -1))
    solid_density = ti.field(dtype=float, shape=(5, 3, 3), offset=(-1, -1, -1))
    solid_fraction.fill(0.5)
    solid_density.fill(3000.0)
    delta = [ti.field(dtype=float, shape=node.velocity[d].shape, offset=(-1, -1, -1)) for d in range(3)]
    dt = ti.field(dtype=float, shape=())
    dt[None] = 0.1

    kernel_compute_mac_viscous_delta(
        1.0e-12,
        1,
        ti.Vector([5, 3, 3]),
        ti.Vector([1.0, 1.0, 1.0]),
        dt,
        FluidProperties(viscosity=2.0),
        cell_type,
        solid_fraction,
        solid_density,
        2,
        delta[0],
        delta[1],
        delta[2],
        node,
        False,
    )
    kernel_apply_mac_viscous_delta(delta[0], delta[1], delta[2], node)

    assert float(node.velocity[0][1, 0, 0]) == pytest.approx(1.0 - 0.0002)


def test_fully_resolved_eq28_device_formula_includes_stress_and_ibm_terms(taichi_runtime):
    GlobalVariable.DIMENSION = 3
    result = ti.Vector.field(3, dtype=float, shape=1)
    evaluate_eq28_force_density(result)

    np.testing.assert_allclose(result.to_numpy()[0], [6818.409090909091, 0.0, 0.0], rtol=1.0e-12)


def test_multi_body_force_partition_sums_to_one_after_total_fraction_clips(taichi_runtime):
    result = ti.field(dtype=float, shape=2)
    evaluate_body_partition(result)

    shares = result.to_numpy()
    np.testing.assert_allclose(shares, [0.8 / 1.5, 0.7 / 1.5], rtol=1.0e-12)
    assert float(shares.sum()) == pytest.approx(1.0)


def test_fully_resolved_pressure_gradient_uses_level_set_surface_distance(taichi_runtime):
    pressure = ti.field(dtype=float, shape=(3, 3, 3))
    surface_tension = ti.field(dtype=float, shape=(3, 3, 3))
    cell_type = ti.field(dtype=int, shape=(3, 3, 3))
    fluid_sdf = ti.field(dtype=float, shape=(3, 3, 3))
    result = ti.Vector.field(3, dtype=float, shape=1)
    cell_type.fill(2)
    cell_type[0, 1, 1] = 1
    cell_type[1, 1, 1] = 1
    cell_type[2, 1, 1] = 0
    pressure[0, 1, 1] = 5.0
    pressure[1, 1, 1] = 1.0
    fluid_sdf[1, 1, 1] = -0.25
    fluid_sdf[2, 1, 1] = 0.75

    evaluate_free_surface_pressure_gradient(pressure, surface_tension, cell_type, fluid_sdf, result)

    np.testing.assert_allclose(result.to_numpy()[0], [-4.0, 0.0, 0.0], rtol=1.0e-12)


def test_sdf_fraction_and_eq28_resultant_converge_for_sphere(taichi_runtime):
    GlobalVariable.DIMENSION = 3
    radius = 0.25
    exact_volume = 4.0 * np.pi * radius**3 / 3.0
    exact_force = exact_volume * np.array([2.0, -1.0, 0.5])
    box = AnalyticSphereSDF.field(shape=1)
    box[0].radius = radius
    levelset_grid = ti.field(dtype=float, shape=1)
    pressure = ti.field(dtype=float, shape=(32, 32, 32))
    surface_tension = ti.field(dtype=float, shape=(32, 32, 32))
    cell_type = ti.field(dtype=int, shape=(32, 32, 32))
    fluid_sdf = ti.field(dtype=float, shape=(32, 32, 32))
    result = ti.field(dtype=float, shape=4)
    volume_errors = []
    force_errors = []

    for n in (8, 16, 32):
        result.fill(0.0)
        cell_type.fill(0)
        initialize_linear_pressure(n, 1.0 / n, pressure, cell_type)
        integrate_sphere_sdf_and_eq28_force(
            n,
            1.0 / n,
            box,
            levelset_grid,
            pressure,
            surface_tension,
            cell_type,
            fluid_sdf,
            result,
        )
        values = result.to_numpy()
        volume_errors.append(abs(values[0] - exact_volume) / exact_volume)
        force_errors.append(np.linalg.norm(values[1:] - exact_force) / np.linalg.norm(exact_force))

    assert volume_errors[2] < volume_errors[1] < volume_errors[0]
    assert force_errors[2] < force_errors[1] < force_errors[0]
    assert volume_errors[-1] < 0.02
    assert force_errors[-1] < 0.02
