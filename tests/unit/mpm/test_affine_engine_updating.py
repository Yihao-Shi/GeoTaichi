"""Affine dispatch and transfer use the final grid field at the interpolation position."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.mpm.Simulation import Simulation
from src.mpm.engines.ULExplicitEngine import ULExplicitEngine
from src.mpm.engines.TLExplicitEngine import TLExplicitEngine
from src.mpm.engines.ULImplicitEngine import ImplicitEngine
from src.mpm.engines.IncompressibleEngine import IncompressibleEngine
from src.mpm.engines.ULExplicitTwoPhaseEngine import ULExplicitTwoPhaseEngine
from src.mpm.engines.ULSemiImplicitTwoPhaseEngine import ULSemiImplicitTwoPhaseEngine
from src.mpm.engines.ULSemiImplicitTwoPhaseEngine_u_p import ULSemiImplicitTwoPhaseEngine_u_p
from src.mpm.engines.ULSemiImplicitTwoPhaseDoubleLayerEngine import ULSemiImplicitTwoPhaseDoubleLayerEngine

pytestmark = [pytest.mark.unit, pytest.mark.mpm, pytest.mark.cpu, pytest.mark.serial]


@pytest.mark.parametrize("scheme", [None, "Affine", "Taylor"])
@pytest.mark.parametrize("requested,expected", [(None, 0.0), (0.0, 0.0), (0.01, 0.01), (1.0, 1.0)])
def test_velocity_projection_keeps_independent_alpha(taichi_runtime, scheme, requested, expected):
    from src.mpm.mainMPM import MPM
    import src.utils.GlobalVariable as global_variable

    mpm = MPM(log=False)
    options = dict(
        domain=[1.0, 1.0],
        dimension=2,
        velocity_projection=scheme,
        material_type="Fluid",
        solver_type="Implicit",
        discretization="FDM",
    )
    if requested is not None:
        options["alphaPIC"] = requested
    mpm.set_configuration(log=False, **options)
    assert mpm.sims.alphaPIC == expected
    assert global_variable.Affine == (scheme == "Affine")
    assert global_variable.Taylor == (scheme == "Taylor")


def test_velocity_projection_defaults_to_none(taichi_runtime):
    from src.mpm.mainMPM import MPM

    mpm = MPM(log=False)
    mpm.set_configuration(log=False, dimension=2, domain=[1.0, 1.0])
    assert mpm.sims.velocity_projection_scheme is None and mpm.sims.alphaPIC == 0.0


@pytest.mark.parametrize("alpha", [0.0, 0.01, 1.0])
def test_velocity_projection_switches_flags_without_changing_alpha(taichi_runtime, alpha):
    import src.utils.GlobalVariable as global_variable

    sims = Simulation()
    sims.set_alpha(alpha)
    for scheme in ("Affine", "Taylor", None):
        sims.set_velocity_projection_scheme(scheme)
        assert sims.alphaPIC == alpha
        assert global_variable.Affine == (scheme == "Affine")
        assert global_variable.Taylor == (scheme == "Taylor")


@pytest.mark.parametrize("alpha", [0.0, 0.01, 1.0])
def test_mls_affine_selection_preserves_requested_alpha(taichi_runtime, alpha):
    sims = Simulation()
    sims.set_alpha(alpha)
    sims.set_moving_least_square(True)
    assert sims.velocity_projection_scheme == "Affine" and sims.alphaPIC == alpha


@pytest.mark.parametrize("scheme", ["Affine", "Taylor"])
@pytest.mark.parametrize("alpha", [0.0, 0.01, 1.0])
def test_fdm_taylor_keeps_affine_reconstruction_and_requested_blend(taichi_runtime, scheme, alpha):
    from src.mpm.engines.EngineKernel import mac_face_mass_momentum
    import src.utils.GlobalVariable as global_variable

    sims = Simulation()
    sims.set_alpha(alpha)
    sims.set_velocity_projection_scheme(scheme)
    global_variable.SHAPEFUNCTION = 0
    result = ti.Vector.field(2, float, shape=())

    @ti.kernel
    def transfer():
        mass, momentum = mac_face_mass_momentum(
            0,
            ti.Vector([2, 2]),
            ti.Vector([8, 8]),
            ti.Vector([1.0, 1.0]),
            ti.Vector([1.0, 1.0]),
            1.0,
            ti.Vector([2.25, 2.25]),
            ti.Vector([1.0, 0.0]),
            ti.Matrix([[2.0, 3.0], [0.0, 0.0]]),
            ti.Vector([0.0, 0.0]),
        )
        result[None] = ti.Vector([mass, momentum])

    transfer()
    mass, momentum = result.to_numpy()
    assert mass > 0
    assert momentum / mass == pytest.approx(1.0 + 2.0 * (-0.25) + 3.0 * 0.25)
    assert sims.alphaPIC == alpha
    sims.discretization, sims.linear_solver = "FDM", "PCG"
    engine = IncompressibleEngine(sims)
    engine.choose_engine(sims)
    assert engine.compute == (engine.affine_updating if scheme == "Affine" else engine.fdm_discretization_pcg)


@pytest.mark.parametrize(
    "engine_type",
    [
        ULExplicitEngine,
        TLExplicitEngine,
        ImplicitEngine,
        IncompressibleEngine,
        ULExplicitTwoPhaseEngine,
        ULSemiImplicitTwoPhaseEngine,
        ULSemiImplicitTwoPhaseEngine_u_p,
        ULSemiImplicitTwoPhaseDoubleLayerEngine,
    ],
)
@pytest.mark.parametrize("mapping", ["USF", "USL", "MUSL", "G2P2G"])
def test_affine_dispatch_ignores_mapping(taichi_runtime, engine_type, mapping):
    sims = Simulation()
    sims.dimension = 2
    sims.mapping = mapping
    sims.linear_solver = "PCG"
    sims.set_alpha(1.0)
    sims.set_velocity_projection_scheme("Affine")
    sims.set_shape_function("QuadBSpline")
    sims.set_moving_least_square(False)
    engine = engine_type(sims)
    engine.choose_engine(sims)
    assert engine.compute == engine.affine_updating
    assert sims.alphaPIC == 1.0
    assert not sims.mls


@pytest.mark.parametrize("stabilize", [None, "Displacement F-Bar Method"])
def test_explicit_affine_saves_matrix_after_stress_before_particle_motion(stabilize):
    engine = object.__new__(ULExplicitEngine)
    events = []
    callbacks = [
        "calculate_interpolation",
        "compute_nodal_kinematic",
        "compute_grid_velcity",
        "compute_forces",
        "apply_particle_traction_constraints",
        "apply_traction_constraints",
        "apply_absorbing_constraints",
        "compute_grid_kinematic",
        "pre_contact_calculate",
        "apply_kinematic_constraints",
        "apply_adaptive_constraints",
        "apply_dirichlet_constraints",
        "compute_contact_force_",
        "compute_velocity_gradient",
        "compute_stress_strains",
        "compute_affine_velocity_gradient",
        "compute_particle_kinematic",
        "pressure_smoothing_",
        "update_grid_refinement_profiled",
    ]
    for name in callbacks:
        setattr(engine, name, lambda *args, name=name: events.append(name))
    sims = SimpleNamespace(
        mode="Normal", stabilize=stabilize, timer=SimpleNamespace(begin=lambda *a: None, end=lambda *a: None)
    )
    engine.affine_updating(sims, SimpleNamespace())
    if stabilize == "Displacement F-Bar Method":
        callbacks.insert(callbacks.index("compute_affine_velocity_gradient"), "calculate_interpolation")
    assert events == callbacks


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("dimension", [2, 3])
def test_affine_matrix_reproduces_final_field_on_small_anisotropic_grid(taichi_runtime, dimension, reference):
    from src.mpm.engines.EngineKernel import (
        kernel_update_velocity_gradient_affine,
        kernel_update_velocity_gradient_affine_2D,
    )

    n = 7
    count = n**dimension
    h = np.arange(1, dimension + 1) * 0.0005
    coords = np.stack([(np.arange(count) // n**d) % n for d in range(dimension)], axis=1)
    position = 3.2 * h
    distances = np.abs(coords - 3.2)
    weights = np.where(
        distances < 0.5, 0.75 - distances**2, np.where(distances < 1.5, 0.5 * (1.5 - distances) ** 2, 0.0)
    ).prod(axis=1)
    gradient = np.arange(dimension**2).reshape(dimension, dimension) * 0.1 + np.eye(dimension)
    particle = ti.Struct.field(
        dict(
            active=ti.i32,
            bodyID=ti.i32,
            x=ti.types.vector(dimension, ti.f64),
            velocity_gradient=ti.types.matrix(dimension, dimension, ti.f64),
        ),
        shape=1,
    )
    node = ti.Struct.field(dict(momentum=ti.types.vector(dimension, ti.f64)), shape=(count, 1))
    ids = ti.field(ti.i32, shape=count)
    shape = ti.field(ti.f64, shape=count)
    size = ti.field(ti.i32, shape=1)
    ids.from_numpy(np.arange(count, dtype=np.int32))
    shape.from_numpy(weights)
    size[0] = count
    particle[0] = dict(
        active=1, bodyID=0, x=position + (0.01 if reference else 0.0), velocity_gradient=np.zeros_like(gradient)
    )
    node.momentum.from_numpy(((coords * h - position) @ gradient.T + 0.3)[:, None, :])
    kernel = kernel_update_velocity_gradient_affine if dimension == 3 else kernel_update_velocity_gradient_affine_2D
    kernel(count, 1, ti.Vector([n] * dimension), ti.Vector(h), node, particle, ids, shape, size, reference)
    np.testing.assert_allclose(particle.velocity_gradient.to_numpy()[0], gradient, rtol=1e-5, atol=1e-5)


@pytest.mark.isolated_dimension(2)
@pytest.mark.parametrize("sparse", [False, True])
def test_implicit_affine_p2g_preserves_acceleration_and_recovers_final_velocity(taichi_runtime, sparse):
    from src.mpm.structs.GridNode import ImplicitNodes2D
    from src.mpm.structs.Particle import ImplicitParticle2D

    sims = Simulation()
    sims.dimension = 2
    sims.configuration = "ULMPM"
    sims.set_alpha(1.0)
    sims.set_velocity_projection_scheme("Affine")
    sims.set_gravity([0.0, 0.0])
    particle = ImplicitParticle2D.field(shape=1)
    n = 8 if sparse else 7
    count_nodes = n * n
    position = np.array([1.12 if sparse else 0.32, 0.32])
    gradient = np.array([[0.7, 0.2], [-0.1, 0.3]])
    particle[0].active = 1
    particle[0].m = 2.0
    particle[0].x = position
    particle[0].v = [0.2, -0.1]
    particle[0].a = [0.4, 0.5]
    particle[0].velocity_gradient = gradient
    node = ImplicitNodes2D.field(shape=(count_nodes, 1))
    coords = np.column_stack((np.arange(count_nodes) % n + (8 if sparse else 0), np.arange(count_nodes) // n)) * 0.1
    d = np.abs((coords - position) / 0.1)
    weights = np.where(d < 0.5, 0.75 - d * d, np.where(d < 1.5, 0.5 * (1.5 - d) ** 2, 0.0)).prod(axis=1)
    ids, shape, size = (
        ti.field(ti.i32, shape=count_nodes),
        ti.field(ti.f64, shape=count_nodes),
        ti.field(ti.i32, shape=1),
    )
    ids.from_numpy(np.arange(count_nodes, dtype=np.int32))
    shape.from_numpy(weights)
    size[0] = count_nodes
    count = ti.field(ti.i32, shape=1)
    count[0] = 1
    sparse_grid = None
    if sparse:
        blocks = ti.field(ti.i32, shape=1)
        blocks[0] = 1
        from src.mpm.sparse_grid.BlockSparseGrid import BlockSparseGrid

        sparse_grid = object.__new__(BlockSparseGrid)
        sparse_grid.block_count, sparse_grid.block_size, sparse_grid.block_volume = ti.Vector([2, 2]), 8, 64
        sparse_grid.active_block_ids = blocks
    scene = SimpleNamespace(
        particle=particle,
        node=node,
        particleNum=count,
        sparse_grid=sparse_grid,
        element=SimpleNamespace(
            grid_nodes=count_nodes,
            gnum=ti.Vector([n, n]),
            grid_size=ti.Vector([0.1, 0.1]),
            LnID=ids,
            shape_fn=shape,
            node_size=size,
        ),
    )
    engine = ImplicitEngine(sims)
    engine.compute_nodal_kinematic(sims, scene)
    mass = node.m.to_numpy()[:, 0]
    active = mass > 0
    np.testing.assert_allclose(
        node.momentum.to_numpy()[active, 0] / mass[active, None],
        (coords[active] - position) @ gradient.T + [0.2, -0.1],
        atol=2e-6,
    )
    np.testing.assert_allclose(
        node.inertia.to_numpy()[active, 0] / mass[active, None], [[0.4, 0.5]] * active.sum(), atol=2e-6
    )
    final_gradient = gradient + np.array([[0.1, -0.3], [0.2, 0.4]])
    node.momentum.from_numpy(((coords - position) @ final_gradient.T + [0.2, -0.1])[:, None, :])
    node.displacement.fill(0.0)
    engine.compute_affine_velocity_gradient(sims, scene)
    np.testing.assert_allclose(particle.velocity_gradient.to_numpy()[0], final_gradient, atol=2e-6)


@pytest.mark.parametrize("mixture", [False, True])
def test_twophase_affine_matrix_orientation_and_mixture_field(taichi_runtime, mixture):
    from src.mpm.engines.EngineKernel import kernel_compute_affine_matrix_APIC_twophase2D

    position = np.array([0.32, 0.32])
    coords = np.column_stack((np.arange(49) % 7, np.arange(49) // 7)) * 0.1
    d = np.abs(coords / 0.1 - 3.2)
    weights = np.where(d < 0.5, 0.75 - d * d, np.where(d < 1.5, 0.5 * (1.5 - d) ** 2, 0.0)).prod(axis=1)
    gradients = [
        np.array([[0.7, 0.2], [-0.1, 0.3]]),
        np.array([[0.5, -0.3], [0.4, 0.2]]),
        np.array([[0.2, 0.6], [-0.5, 0.9]]),
    ]
    node = ti.Struct.field(
        {name: ti.types.vector(2, ti.f64) for name in ["momentums", "momentumf", "momentum"]}, shape=(49, 1)
    )
    for name, gradient in zip(["momentums", "momentumf", "momentum"], gradients):
        getattr(node, name).from_numpy(((coords - position) @ gradient.T + 0.1)[:, None, :])
    particle = ti.Struct.field(
        dict(
            active=ti.i32,
            materialID=ti.i32,
            bodyID=ti.i32,
            x=ti.types.vector(2, ti.f64),
            solid_velocity_gradient=ti.types.matrix(2, 2, ti.f64),
            fluid_velocity_gradient=ti.types.matrix(2, 2, ti.f64),
        ),
        shape=1,
    )
    particle[0].active, particle[0].materialID = 1, 1
    particle[0].x = position
    ids, shape, size = ti.field(ti.i32, shape=49), ti.field(ti.f64, shape=49), ti.field(ti.i32, shape=1)
    ids.from_numpy(np.arange(49, dtype=np.int32))
    shape.from_numpy(weights)
    size[0] = 49
    kernel_compute_affine_matrix_APIC_twophase2D(
        49, 1, node, particle, ids, shape, size, ti.Vector([0.1, 0.1]), ti.Vector([7, 7]), mixture
    )
    for name, expected in [
        ("solid_velocity_gradient", gradients[2] if mixture else gradients[0]),
        ("fluid_velocity_gradient", gradients[2] if mixture else gradients[1]),
    ]:
        np.testing.assert_allclose(getattr(particle, name).to_numpy()[0], expected, atol=2e-6)
