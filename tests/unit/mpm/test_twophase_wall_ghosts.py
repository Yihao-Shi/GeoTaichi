import numpy as np
import pytest
import taichi as ti

from geotaichi import init
from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    FLUID_CELL,
    SOLID_CELL,
    MAC_SHAPE_QUAD_BSPLINE,
    MAC_SHAPE_LINEAR,
    _sample_mac_velocity,
    _sample_mac_velocity3d,
    _sample_mac_velocity_gradient,
    _sample_mac_velocity_gradient3d,
    kernel_correct_double_layer_velocity2d,
    kernel_correct_double_layer_velocity3d,
    kernel_extrapolate_double_layer_ghost_faces,
    kernel_g2p_double_layer2d,
    kernel_g2p_double_layer3d,
)
from src.mpm.structs.GridNode import NodeTwoPhase, NodeTwoPhase2D
from src.mpm.structs.Particle import ParticleCloudTwoPhase, ParticleCloudTwoPhase2D
from src.physics_model.consititutive_model.infinitesimal_strain.LinearElastic import LinearElasticModel


@ti.dataclass
class _FixedSolidState:
    estress: float


@ti.kernel
def _sample_wall_field(
    dimension: ti.template(), velocity: ti.template(), result: ti.template(), coordinate: ti.template()
):
    v = ti.Vector.zero(float, dimension)
    gradient = ti.Matrix.zero(float, dimension, dimension)
    if ti.static(dimension == 2):
        position = ti.Vector(coordinate)
        spacing = ti.Vector([0.2, 0.2])
        size = ti.Vector([0.05, 0.05])
        v = _sample_mac_velocity(position, size, MAC_SHAPE_QUAD_BSPLINE, 3, spacing, *velocity)
        gradient = _sample_mac_velocity_gradient(position, size, MAC_SHAPE_QUAD_BSPLINE, 3, spacing, *velocity, v)
    else:
        position = ti.Vector(coordinate)
        spacing = ti.Vector([0.2, 0.2, 0.2])
        size = ti.Vector([0.05, 0.05, 0.05])
        v = _sample_mac_velocity3d(position, size, MAC_SHAPE_QUAD_BSPLINE, 3, spacing, *velocity)
        gradient = _sample_mac_velocity_gradient3d(position, size, MAC_SHAPE_QUAD_BSPLINE, 3, spacing, *velocity, v)
    result[0] = v.norm()
    result[1] = gradient.norm()


@pytest.mark.parametrize("dimension", [2, 3])
def test_hydrostatic_wall_ghosts_do_not_reinject_gravity_into_apic(dimension):
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    shape = (6,) * dimension
    cell_type = ti.field(int, shape=shape)
    types = np.full(shape, SOLID_CELL, dtype=np.int32)
    types[:2] = FLUID_CELL
    cell_type.from_numpy(types)
    pressure = ti.field(float, shape=shape)
    pressure.from_numpy(1000 * 9.81 * (1.2 - (np.indices(shape)[-1] + 0.5) * 0.2))
    sdf = ti.field(float, shape=shape)
    sdf.fill(-0.2)
    dt = ti.field(float, shape=())
    dt[None] = 0.005
    fields = [[] for _ in range(5)]
    for axis in range(dimension):
        face_shape = tuple(n + (axis == d) for d, n in enumerate(shape))
        for group, fill in zip(fields, [1.0, -9.81 * 0.005 if axis == dimension - 1 else 0.0, 0.0, 1000.0, 0.0]):
            field = ti.field(float, shape=face_shape)
            field.fill(fill)
            group.append(field)
    mass, velocity, acceleration, density, solid = fields
    correction = kernel_correct_double_layer_velocity2d if dimension == 2 else kernel_correct_double_layer_velocity3d
    correction(
        1.0e-12,
        ti.Vector([0.2] * dimension),
        dt,
        cell_type,
        pressure,
        sdf,
        *mass,
        *velocity,
        *acceleration,
        *density,
        None,
    )
    result = ti.field(float, shape=2)
    coordinate = (0.395,) + (0.5,) * (dimension - 1)
    _sample_wall_field(dimension, tuple(velocity), result, coordinate)
    assert result[0] > 0.01
    kernel_extrapolate_double_layer_ghost_faces(
        cell_type, tuple(velocity), tuple(acceleration), tuple(solid), tuple(mass), 1e-12, 2, dt
    )
    _sample_wall_field(dimension, tuple(velocity), result, coordinate)
    np.testing.assert_allclose(result.to_numpy(), 0.0, atol=2e-6)


@pytest.mark.parametrize("dimension", [2, 3])
def test_wall_ghosts_use_corrected_tangent_velocity_without_changing_active_faces(dimension):
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    shape = (6,) * dimension
    cell_type = ti.field(int, shape=shape)
    types = np.full(shape, FLUID_CELL, dtype=np.int32)
    types[2:] = SOLID_CELL
    cell_type.from_numpy(types)
    velocity, acceleration, solid_velocity, mass = [], [], [], []
    for axis in range(dimension):
        face_shape = tuple(n + (axis == d) for d, n in enumerate(shape))
        v = ti.field(float, shape=face_shape)
        a = ti.field(float, shape=face_shape)
        s = ti.field(float, shape=face_shape)
        m = ti.field(float, shape=face_shape)
        m.fill(1.0)
        v.fill(2.0 if axis else 0.5)
        a.fill(0.0)
        s.fill(0.5 if axis == 0 else 0.0)
        values = v.to_numpy()
        values[3 if axis == 0 else 2 :] -= 0.04905
        v.from_numpy(values)
        if axis != 0:
            increments = a.to_numpy()
            increments[2:] = -9.81
            a.from_numpy(increments)
        velocity.append(v)
        acceleration.append(a)
        solid_velocity.append(s)
        mass.append(m)
    active_before = [v.to_numpy()[:2].copy() for v in velocity]
    dt = ti.field(float, shape=())
    dt[None] = 0.005
    kernel_extrapolate_double_layer_ghost_faces(
        cell_type, tuple(velocity), tuple(acceleration), tuple(solid_velocity), tuple(mass), 1e-12, 3, dt
    )
    for axis in range(dimension):
        np.testing.assert_array_equal(velocity[axis].to_numpy()[:2], active_before[axis])
        np.testing.assert_allclose(velocity[axis].to_numpy()[2 if axis else 3 : 4], 2.0 if axis else 0.5)
        np.testing.assert_allclose(acceleration[axis].to_numpy(), 0.0)


@pytest.mark.parametrize("dimension", [2, 3])
def test_free_surface_ghosts_do_not_reinject_gravity_into_apic(dimension):
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    shape = (6,) * dimension
    cell_type = ti.field(int, shape=shape)
    types = np.zeros(shape, dtype=int)
    types[(slice(None),) * (dimension - 1) + (slice(0, 3),)] = FLUID_CELL
    cell_type.from_numpy(types)
    vertical_coordinate = (np.indices(shape)[-1] + 0.5) * 0.2
    pressure = ti.field(float, shape=shape)
    pressure.from_numpy(np.maximum(0, 1000 * 9.81 * (0.6 - vertical_coordinate)))
    sdf = ti.field(float, shape=shape)
    sdf.from_numpy(vertical_coordinate - 0.6)
    dt = ti.field(float, shape=())
    dt[None] = 0.005
    fields = [[] for _ in range(5)]
    for axis in range(dimension):
        face_shape = tuple(n + (axis == d) for d, n in enumerate(shape))
        for group, fill in zip(fields, [1.0, -9.81 * 0.005 if axis == dimension - 1 else 0.0, 0.0, 1000.0, 0.0]):
            field = ti.field(float, shape=face_shape)
            field.fill(fill)
            group.append(field)
    mass, velocity, acceleration, density, solid = fields
    correction = kernel_correct_double_layer_velocity2d if dimension == 2 else kernel_correct_double_layer_velocity3d
    correction(
        1e-12,
        ti.Vector([0.2] * dimension),
        dt,
        cell_type,
        pressure,
        sdf,
        *mass,
        *velocity,
        *acceleration,
        *density,
        None,
    )
    result = ti.field(float, shape=2)
    coordinate = (0.5,) * (dimension - 1) + (0.595,)
    _sample_wall_field(dimension, tuple(velocity), result, coordinate)
    assert result[0] > 0.001
    kernel_extrapolate_double_layer_ghost_faces(
        cell_type, tuple(velocity), tuple(acceleration), tuple(solid), tuple(mass), 1e-12, 2, dt
    )
    _sample_wall_field(dimension, tuple(velocity), result, coordinate)
    np.testing.assert_allclose(result.to_numpy(), 0, atol=2e-6)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("fluid_below", [True, False])
def test_normal_wall_ghosts_reflect_linear_velocity_and_preserve_moving_wall(dimension, fluid_below):
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    shape = (8,) * dimension
    axis = dimension - 1
    wall_index = 4
    types = np.full(shape, SOLID_CELL, dtype=np.int32)
    types[(slice(None),) * axis + (slice(None, wall_index) if fluid_below else slice(wall_index, None),)] = FLUID_CELL
    cell_type = ti.field(int, shape=shape)
    cell_type.from_numpy(types)
    velocity, acceleration, solid, mass = [], [], [], []
    for component in range(dimension):
        face_shape = tuple(n + (component == d) for d, n in enumerate(shape))
        for group, fill in zip(
            (velocity, acceleration, solid, mass), (0.0, 0.0, 0.3 if component == axis else 0.0, 1.0)
        ):
            field = ti.field(float, shape=face_shape)
            field.fill(fill)
            group.append(field)
    coordinates = np.indices(velocity[axis].shape)[axis]
    valid = coordinates <= wall_index if fluid_below else coordinates >= wall_index
    values = np.where(valid, 0.3 + 0.2 * (coordinates - wall_index), 99.0)
    increments = np.where(valid, 0.4 * (coordinates - wall_index), 99.0)
    velocity[axis].from_numpy(values)
    acceleration[axis].from_numpy(increments)
    dt = ti.field(float, shape=())
    dt[None] = 0.005
    kernel_extrapolate_double_layer_ghost_faces(
        cell_type, tuple(velocity), tuple(acceleration), tuple(solid), tuple(mass), 1e-12, 2, dt
    )
    selected = [slice(2, 6)] * dimension
    selected[axis] = slice(2, 7)
    np.testing.assert_allclose(
        velocity[axis].to_numpy()[tuple(selected)], (0.3 + 0.2 * (coordinates - wall_index))[tuple(selected)], atol=2e-7
    )
    np.testing.assert_allclose(
        acceleration[axis].to_numpy()[tuple(selected)], (0.4 * (coordinates - wall_index))[tuple(selected)], atol=2e-7
    )
    result = ti.field(float, shape=2)
    coordinate = (0.7,) * axis + (0.795 if fluid_below else 0.805,)
    _sample_wall_field(dimension, tuple(velocity), result, coordinate)
    np.testing.assert_allclose(result.to_numpy(), [0.3 + coordinate[-1] - 0.8, 1.0], atol=2e-6)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("wall", [False, True])
def test_ghosts_preserve_tangential_affine_velocity_when_acceleration_is_zero(dimension, wall):
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    shape = (7,) * dimension
    types = ti.field(int, shape=shape)
    if wall:
        values = np.zeros(shape, dtype=int)
        values[(slice(None),) * (dimension - 1) + (slice(0, 3),)] = SOLID_CELL
        types.from_numpy(values)
    types[(3,) * dimension] = FLUID_CELL
    velocity, acceleration, solid, mass = [], [], [], []
    for axis in range(dimension):
        face_shape = tuple(n + (axis == d) for d, n in enumerate(shape))
        for group, fill in ((velocity, 0), (acceleration, 0), (solid, 0), (mass, 1)):
            field = ti.field(float, shape=face_shape)
            field.fill(fill)
            group.append(field)
        velocity[axis].from_numpy(1 + np.indices(face_shape).sum(axis=0) * 0.2)
    dt = ti.field(float, shape=())
    dt[None] = 0.01
    ghost = (3,) * (dimension - 1) + (2,) if wall else (5,) + (3,) * (dimension - 1)
    before = velocity[0].to_numpy()
    kernel_extrapolate_double_layer_ghost_faces(
        types, tuple(velocity), tuple(acceleration), tuple(solid), tuple(mass), 1e-12, 2, dt
    )
    assert velocity[0][ghost] == pytest.approx(before[ghost])
    assert velocity[0][(3,) * dimension] == pytest.approx(before[(3,) * dimension])


def test_closed_pressure_gauge_does_not_switch_for_one_cell_wall_void():
    from src.mpm.Recorder import _closed_domain_pressure_offset

    types = np.full((5, 5, 5, 1), 2, dtype=np.int32)
    types[1:-1, 1:-1, 1:-1] = 1
    types[1, 1, 2] = 0
    pressure = np.full(types.shape, -73.5)
    assert _closed_domain_pressure_offset(pressure, types) == pytest.approx(-73.5)
    types[1, 1, 3] = 0
    assert _closed_domain_pressure_offset(pressure, types) == 0
    types[0, 1, 2] = 0
    assert _closed_domain_pressure_offset(pressure, types) == 0


@pytest.mark.parametrize("dimension", [2, 3])
def test_incompressible_wall_void_is_not_an_atmospheric_pressure_boundary(dimension):
    from src.mpm.engines.EngineKernel import kernel_fill_enclosed_fluid_cells

    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    shape = (7,) * dimension
    cell_type = ti.field(int, shape=shape, offset=(-1,) * dimension)
    types = np.full(shape, SOLID_CELL, dtype=np.int32)
    types[(slice(1, -1),) * dimension] = FLUID_CELL
    types[(1,) * dimension] = 0
    cell_type.from_numpy(types)
    kernel_fill_enclosed_fluid_cells(1, ti.Vector(shape), cell_type)
    assert cell_type.to_numpy()[(1,) * dimension] == FLUID_CELL


@pytest.mark.parametrize("dimension", [2, 3])
def test_fully_fixed_skeleton_does_not_retain_a_moving_apic_gradient(dimension):
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    node = (NodeTwoPhase2D if dimension == 2 else NodeTwoPhase).field(shape=(1, 1))
    particle = (ParticleCloudTwoPhase2D if dimension == 2 else ParticleCloudTwoPhase).field(shape=1)
    state = _FixedSolidState.field(shape=1)
    material = LinearElasticModel(material_type="TwoPhaseDoubleLayer")
    material.initialize_coupling()
    material.model_initialize({"YoungModulus": 3e6, "Porosity": 0.4, "FluidBulkModulus": 2.2e8, "Permeability": 1e-3})
    node.momentums.from_numpy(np.full((1, 1, dimension), 0.2))
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.phase.fill(1)
    particle.x.from_numpy(np.full((1, dimension), 0.5))
    mapping, lnid, count = [ti.field(int, shape=1) for _ in range(3)]
    count.fill(1)
    shape_fn = ti.field(float, shape=1)
    shape_fn.fill(1.0)
    dshape = ti.Vector.field(dimension, float, shape=1)
    dshape.from_numpy(np.ones((1, dimension)))
    lengths = ti.Vector.field(dimension, float, shape=1)
    lengths.from_numpy(np.full((1, dimension), 0.05))
    dt = ti.field(float, shape=())
    dt[None] = 0.001
    velocity, acceleration = [], []
    for axis in range(dimension):
        face_shape = tuple(2 + (axis == d) for d in range(dimension))
        velocity.append(ti.field(float, shape=face_shape))
        acceleration.append(ti.field(float, shape=face_shape))
    types = ti.field(int, shape=(2,) * dimension)
    pressure, sdf = [ti.field(float, shape=(2,) * dimension) for _ in range(2)]
    types.fill(FLUID_CELL)
    g2p = kernel_g2p_double_layer2d if dimension == 2 else kernel_g2p_double_layer3d
    for fixed in ([0] * dimension, [1] * dimension, [1] + [0] * (dimension - 1)):
        particle.fix_v.from_numpy(np.asarray([fixed], dtype=np.uint8))
        particle.vs.fill(0.0)
        g2p(
            1,
            0,
            1,
            1.0,
            ti.Vector([1.0] * dimension),
            ti.Vector([0.2] * dimension),
            MAC_SHAPE_LINEAR,
            2,
            1,
            dt,
            material,
            state,
            node,
            particle,
            mapping,
            lengths,
            lnid,
            shape_fn,
            dshape,
            count,
            *velocity,
            *acceleration,
            types,
            pressure,
            sdf,
            1,
            0,
            1,
        )
        gradient = particle.solid_velocity_gradient.to_numpy()[0]
        np.testing.assert_allclose(gradient, 0.0 if all(fixed) else 0.2, atol=1e-7)
        np.testing.assert_allclose(particle.vs.to_numpy()[0], np.where(fixed, 0.0, 0.2), atol=1e-7)
