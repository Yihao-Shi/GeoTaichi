import numpy as np
import pytest
import taichi as ti

from geotaichi import init
from src.mpm.engines.EngineKernel import kernel_fdm_shifting_reference_volume
from src.utils import GlobalVariable


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
@pytest.mark.parametrize("corner", [False, True])
def test_shifting_integrates_solid_walls_but_does_not_close_air(dimension, corner):
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    GlobalVariable.SHAPEFUNCTION = 2
    count, dx = 8, 0.125
    gnum = count + 1
    types = np.zeros((count,) * dimension, dtype=np.int32)
    axes = range(dimension) if corner else [dimension - 1]
    for axis in axes:
        slices = [slice(None)] * dimension
        slices[axis] = slice(4, None)
        types[tuple(slices)] = 2
    cells = ti.field(int, shape=types.shape)
    cells.from_numpy(types)
    lengths = ti.Vector.field(dimension, float, shape=1)
    lengths.fill(dx / 2)
    reference = ti.field(float, shape=(gnum**dimension, 1))
    boundary = ti.Vector.field(dimension, ti.u8, shape=reference.shape)
    indices = np.stack(np.unravel_index(np.arange(gnum**dimension), (gnum,) * dimension, order="F"), axis=-1)
    side = np.minimum(2, indices) - np.minimum(gnum - 1 - indices, 2)
    boundary.from_numpy(np.where(side < 0, side + 3, np.where(side > 0, side + 2, 0)).astype(np.uint8)[:, None, :])

    def integrate():
        kernel_fdm_shifting_reference_volume(
            0, ti.Vector([gnum] * dimension), ti.Vector([dx] * dimension), lengths, boundary, reference, cells
        )
        return reference.to_numpy()[:, 0].reshape((gnum,) * dimension, order="F")

    result = integrate()
    assert result[(4,) * dimension] == pytest.approx(dx**dimension * 0.5 ** len(axes), abs=1e-12)
    assert result.sum() == pytest.approx(np.count_nonzero(types != 2) * dx**dimension, abs=1e-12)
    if not corner:
        below, above = [4] * dimension, [4] * dimension
        below[-1], above[-1] = 3, 5
        assert result[tuple(below)] == pytest.approx(dx**dimension * 47 / 48, abs=1e-12)
        assert result[tuple(above)] == pytest.approx(dx**dimension / 48, abs=1e-12)
    types[types == 0] = 1
    cells.from_numpy(types)
    np.testing.assert_array_equal(integrate(), result)


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
@pytest.mark.parametrize("corner", [False, True])
def test_uniform_bed_porosity_at_internal_solid_walls(dimension, corner):
    from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
        kernel_p2g_double_layer_mass2d,
        kernel_p2g_double_layer_mass3d,
        kernel_normalize_double_layer_nodes2d,
        kernel_normalize_double_layer_nodes3d,
    )
    from src.mpm.structs.GridNode import NodeTwoPhase, NodeTwoPhase2D
    from src.mpm.structs.Particle import ParticleCloudTwoPhase, ParticleCloudTwoPhase2D

    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    GlobalVariable.SHAPEFUNCTION = 2
    count, dx, porosity = 8, 0.125, 0.4
    axes = list(range(dimension)) if corner else [dimension - 1]
    points = np.stack(np.meshgrid(*([np.arange(0.25, count, 0.5)] * dimension), indexing="ij"), axis=-1).reshape(
        -1, dimension
    )
    points = points[np.all(points[:, axes] < 4, axis=1)]
    distance = np.abs(points - 4)
    weights = np.where(
        distance < 0.5, 0.75 - distance**2, np.where(distance < 1.5, 0.5 * (1.5 - distance) ** 2, 0.0)
    ).prod(axis=1)
    num, gnum = len(points), count + 1
    target = sum(4 * gnum**d for d in range(dimension))
    particle = (ParticleCloudTwoPhase2D if dimension == 2 else ParticleCloudTwoPhase).field(shape=num)
    node = (NodeTwoPhase2D if dimension == 2 else NodeTwoPhase).field(shape=(gnum**dimension, 1))
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.phase.fill(1)
    particle.porosity.fill(porosity)
    particle.vol.fill((dx / 2) ** dimension)
    particle.ms.fill(2650 * (1 - porosity) * (dx / 2) ** dimension)
    particle.x.from_numpy(points * dx)
    lnid = ti.field(int, shape=num)
    lnid.fill(target)
    shape = ti.field(float, shape=num)
    shape.from_numpy(weights)
    size = ti.field(ti.u8, shape=num)
    size.fill(1)
    cells = ti.field(int, shape=(count,) * dimension)
    types = np.zeros(cells.shape, dtype=np.int32)
    for axis in axes:
        slices = [slice(None)] * dimension
        slices[axis] = slice(4, None)
        types[tuple(slices)] = 2
    cells.from_numpy(types)
    reference = ti.field(float, shape=node.shape)
    lengths = ti.Vector.field(dimension, float, shape=1)
    lengths.fill(dx / 2)
    boundary = ti.Vector.field(dimension, ti.u8, shape=node.shape)
    kernel_fdm_shifting_reference_volume(
        0, ti.Vector([gnum] * dimension), ti.Vector([dx] * dimension), lengths, boundary, reference, cells
    )
    assert reference[target, 0] == pytest.approx(dx**dimension * 0.5 ** len(axes), abs=1e-12)
    scatter = kernel_p2g_double_layer_mass2d if dimension == 2 else kernel_p2g_double_layer_mass3d
    normalize = kernel_normalize_double_layer_nodes2d if dimension == 2 else kernel_normalize_double_layer_nodes3d
    scatter(1, num, ti.Vector([dx] * dimension), ti.Vector([gnum] * dimension), 0, node, particle, lnid, shape, size)
    normalize(1e-12, node, reference)
    assert node.porosity[target, 0] == pytest.approx(porosity, abs=1e-6)


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
def test_shifting_cutoff_is_independent_of_length_units(dimension):
    from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
        kernel_delta_correct_double_layer_fluid2d,
        kernel_delta_correct_double_layer_fluid3d,
    )
    from src.mpm.structs.Particle import ParticleCloudTwoPhase, ParticleCloudTwoPhase2D

    init(dim=dimension, arch="cpu", default_fp="float64", cpu_max_num_threads=2, offline_cache=False, log=False)
    node = ti.Struct.field({"vol": float, "porosity": float}, shape=(1, 1))
    node.porosity.fill(1.0)
    reference = ti.field(float, shape=node.shape)
    volume_error = ti.field(float, shape=node.shape)
    template = ParticleCloudTwoPhase2D if dimension == 2 else ParticleCloudTwoPhase
    particle = ti.Struct.field({**template.members, "grad_E2": ti.types.vector(dimension, float)}, shape=1)
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.phase.fill(2)
    particle.porosity.fill(1.0)
    particle.fluid_velocity_gradient.from_numpy(np.eye(dimension)[None])
    lnid = ti.field(int, shape=1)
    size = ti.field(ti.u8, shape=1)
    size.fill(1)
    gradient = ti.Vector.field(dimension, float, shape=1)
    cells = ti.field(int, shape=(8,) * dimension)
    cells.fill(1)
    correct = kernel_delta_correct_double_layer_fluid2d if dimension == 2 else kernel_delta_correct_double_layer_fluid3d
    for dx in (0.125, 0.005):
        volume = dx**dimension
        particle.vol.fill(0.5 * volume)
        reference.fill(volume)
        for error, derivative in ((1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (1e-4, 1.0)):
            position = np.array([[3.5 * dx] * dimension])
            particle.x.from_numpy(position)
            particle.v.fill(0)
            particle.vf.fill(0)
            node.vol.fill(volume * (1 + error))
            gradient.from_numpy(np.array([[derivative / dx] + [0.0] * (dimension - 1)]))
            correct(
                1,
                1,
                ti.Vector([8 * dx] * dimension),
                ti.Vector([dx] * dimension),
                ti.Vector([8] * dimension),
                1.0,
                cells,
                node,
                particle,
                lnid,
                gradient,
                size,
                reference,
                volume_error,
            )
            expected = np.zeros((1, dimension))
            if derivative:
                expected[0, 0] = -min(2 * error, 0.05) * dx
            tolerance = max(1e-12, np.finfo(particle.x.to_numpy().dtype).eps * 8 * dx)
            np.testing.assert_allclose(particle.x.to_numpy() - position, expected, rtol=1e-7, atol=tolerance)
            np.testing.assert_allclose(particle.vf.to_numpy(), np.zeros_like(expected), atol=tolerance)


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
def test_shifting_corrects_interior_deficits_without_attracting_to_air(dimension):
    from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
        kernel_delta_correct_double_layer_fluid2d,
        kernel_delta_correct_double_layer_fluid3d,
    )
    from src.mpm.structs.Particle import ParticleCloudTwoPhase, ParticleCloudTwoPhase2D

    init(dim=dimension, arch="cpu", default_fp="float64", cpu_max_num_threads=2, offline_cache=False, log=False)
    dx, count = 0.125, 4
    target = sum(2 * (count + 1) ** d for d in range(dimension))
    node = ti.Struct.field({"vol": float, "porosity": float}, shape=((count + 1) ** dimension, 1))
    node.porosity.fill(1.0)
    reference = ti.field(float, shape=node.shape)
    volume_error = ti.field(float, shape=node.shape)
    reference[target, 0] = dx**dimension
    node.vol[target, 0] = 0.9999 * dx**dimension
    template = ParticleCloudTwoPhase2D if dimension == 2 else ParticleCloudTwoPhase
    particle = ti.Struct.field({**template.members, "grad_E2": ti.types.vector(dimension, float)}, shape=1)
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.phase.fill(2)
    particle.porosity.fill(1.0)
    particle.vol.fill(0.5 * dx**dimension)
    particle.mf.fill(0.25)
    lnid = ti.field(int, shape=1)
    lnid.fill(target)
    size = ti.field(ti.u8, shape=1)
    size.fill(1)
    gradient = ti.Vector.field(dimension, float, shape=1)
    gradient.from_numpy(np.array([[1 / dx] + [0.0] * (dimension - 1)]))
    cells = ti.field(int, shape=(count,) * dimension)
    cells.fill(1)
    correct = kernel_delta_correct_double_layer_fluid2d if dimension == 2 else kernel_delta_correct_double_layer_fluid3d
    position = np.array([[2 * dx] * dimension])
    for at_surface, unmovable_deficit in ((False, False), (False, True), (True, True)):
        particle.x.from_numpy(position)
        reference[0, 0] = dx**dimension if unmovable_deficit else 0.0
        if at_surface:
            cells[(1,) * dimension] = 0
        correct(
            1,
            1,
            ti.Vector([count * dx] * dimension),
            ti.Vector([dx] * dimension),
            ti.Vector([count] * dimension),
            1.0,
            cells,
            node,
            particle,
            lnid,
            gradient,
            size,
            reference,
            volume_error,
        )
        expected = np.zeros((1, dimension))
        if not at_surface:
            expected[0, 0] = 0.0002 * dx
        tolerance = max(1e-12, np.finfo(particle.x.to_numpy().dtype).eps * count * dx)
        np.testing.assert_allclose(particle.x.to_numpy() - position, expected, atol=tolerance)
        assert particle.mf[0] == 0.25


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
def test_two_phase_shifting_reference_follows_moving_wall(dimension):
    from types import SimpleNamespace
    from src.mpm.engines.ULSemiImplicitTwoPhaseDoubleLayerEngine import ULSemiImplicitTwoPhaseDoubleLayerEngine

    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    GlobalVariable.SHAPEFUNCTION = 2
    count, dx = 8, 0.125
    gnum = count + 1
    engine = object.__new__(ULSemiImplicitTwoPhaseDoubleLayerEngine)
    engine.shifting_reference_ready = False
    engine.update_external_mac_boundary = lambda *_: None
    engine.cell_type = ti.field(int, shape=(count,) * dimension)
    engine.shifting_reference_volume = ti.field(float, shape=(gnum**dimension, 1))
    lengths = ti.Vector.field(dimension, float, shape=1)
    lengths.fill(dx / 2)
    boundary = ti.Vector.field(dimension, ti.u8, shape=engine.shifting_reference_volume.shape)
    indices = np.stack(np.unravel_index(np.arange(gnum**dimension), (gnum,) * dimension, order="F"), axis=-1)
    side = np.minimum(2, indices) - np.minimum(gnum - 1 - indices, 2)
    boundary.from_numpy(np.where(side < 0, side + 3, np.where(side > 0, side + 2, 0)).astype(np.uint8)[:, None, :])
    scene = SimpleNamespace(
        element=SimpleNamespace(
            gnum=ti.Vector([gnum] * dimension),
            grid_size=ti.Vector([dx] * dimension),
            calLength=lengths,
            boundary_type=boundary,
        )
    )
    # A piston advances and retreats; the target volume excludes its current cells.
    for wall_cells in (0, 3, 2):
        kinds = np.ones(engine.cell_type.shape, dtype=np.int32)
        kinds[:wall_cells] = 2
        engine.cell_type.from_numpy(kinds)
        engine._apply_external_mac_boundary(None, scene)
        engine._prepare_shifting_reference_volume(scene)
        expected = np.count_nonzero(kinds != 2) * dx**dimension
        assert engine.shifting_reference_volume.to_numpy().sum() == pytest.approx(expected, abs=1e-12)


@pytest.mark.isolated_dimension(3)
@pytest.mark.parametrize(
    "case_name",
    [
        "examples.mmpm.TwoPhaseWavemaker3D.two_layer_two_phase_wavemaker_3d",
        "examples.mmpm.TwoPhaseLSDEMCoupling.wavemaker_lsdem_particles_3d.wavemaker_lsdem_particles_3d",
    ],
)
def test_wavemaker_shifting_capacity_follows_physical_piston(case_name):
    import importlib
    from types import SimpleNamespace
    from src.mpm.engines.ULSemiImplicitTwoPhaseDoubleLayerEngine import ULSemiImplicitTwoPhaseDoubleLayerEngine

    init(dim=3, arch="cpu", default_fp="float64", cpu_max_num_threads=2, offline_cache=False, log=False)
    GlobalVariable.SHAPEFUNCTION = 2
    count, dx = 8, 0.125
    gnum = count + 1
    engine = object.__new__(ULSemiImplicitTwoPhaseDoubleLayerEngine)
    engine.shifting_reference_ready = False
    engine.cell_type = ti.field(int, shape=(count,) * 3)
    engine.shifting_reference_volume = ti.field(float, shape=(gnum**3, 1))
    lengths = ti.Vector.field(3, float, shape=1)
    lengths.fill(dx / 2)
    boundary = ti.Vector.field(3, ti.u8, shape=engine.shifting_reference_volume.shape)
    indices = np.stack(np.unravel_index(np.arange(gnum**3), (gnum,) * 3, order="F"), axis=-1)
    side = np.minimum(2, indices) - np.minimum(gnum - 1 - indices, 2)
    boundary.from_numpy(np.where(side < 0, side + 3, np.where(side > 0, side + 2, 0)).astype(np.uint8)[:, None, :])
    scene = SimpleNamespace(
        element=SimpleNamespace(
            gnum=ti.Vector([gnum] * 3),
            grid_size=ti.Vector([dx] * 3),
            calLength=lengths,
            boundary_type=boundary,
        )
    )
    update = importlib.import_module(case_name).update_piston_shifting_reference
    voxel_reference = ti.field(float, shape=engine.shifting_reference_volume.shape)
    for wall in (2 * dx, 2.5 * dx - 1e-7, 2.5 * dx + 1e-7, 3 * dx, 2 * dx):
        kinds = np.ones(engine.cell_type.shape, dtype=np.int32)
        kinds[(np.arange(count) + 0.5) * dx <= wall] = 2
        kinds[:, 0, :] = kinds[:, -1, :] = kinds[:, :, 0] = kinds[-1, :, :] = 2
        engine.cell_type.from_numpy(kinds)
        calls = []

        def callback(sims, scene):
            calls.append(1)
            update(engine, scene, wall, [1.0] * 3, dx)

        engine.update_external_mac_boundary = callback
        engine._apply_external_mac_boundary(None, scene)
        engine._prepare_shifting_reference_volume(scene)
        result = engine.shifting_reference_volume.to_numpy()
        assert calls == [1]
        assert result.sum() == pytest.approx((1 - dx - wall) * (1 - 2 * dx) * (1 - dx), abs=1e-12)
        assert np.isfinite(result).all() and np.all(result >= 0)
        if wall in (2 * dx, 3 * dx):
            kernel_fdm_shifting_reference_volume(
                0,
                scene.element.gnum,
                scene.element.grid_size,
                lengths,
                boundary,
                voxel_reference,
                engine.cell_type,
            )
            np.testing.assert_allclose(result, voxel_reference.to_numpy(), atol=1e-14)
