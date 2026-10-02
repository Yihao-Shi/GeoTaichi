from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.linear_solver.MultiGridPCG_mixture import MGPCGMixPoissonSolver
from src.linear_solver.MultiGridPCG_mixture_Axi import MGPCGMixPoissonSolver_Axi
from src.mpm.elements.QuadrilateralElement4Nodes import QuadrilateralElement4Nodes
from src.mpm.elements.QuadrilateralKernel import estimate_active_dofs_poisson
from src.mpm.engines.EngineKernel import (
    kernel_assemble_A,
    kernel_clamp_twophase_particle_pressure,
    kernel_correct_grid_kinematic_semitwophase,
    kernel_correct_grid_kinematic_semitwophase_2DAxisy,
    kernel_correct_grid_kinematic_semitwophase_u_p,
    kernel_correct_grid_kinematic_semitwophase_mg,
    kernel_correct_grid_kinematic_semitwophase_mg_3D,
    kernel_update_nodal_pressure_2D,
    kernel_kinemaitc_g2p_semitwophase2D,
    kernel_kinemaitc_g2p_semitwophase2D_FIC,
    kernel_kinemaitc_g2p_semitwophase2D_u_p,
    kernel_force_p2g_semitwophase2D,
    kernel_force_p2g_semitwophase2D_u_p,
    kernel_single_point_porosity_p2g,
    kernel_compute_grid_velocity_twophase,
    kernel_mass_momentum_p2g_twophase_u_p,
    kernel_pressure_tpic_p2g_correction_2D,
)
from src.mpm.engines.AssembleMatrixKernel import (
    kernel_eliminate_pressure_increment_dirichlet,
    kernel_assemble_residual_poisson_2D,
    kernel_assemble_residual_poisson_FIC_2D,
    kernel_assemble_residual_poisson_2D_u_p,
    kernel_assemble_local_stiffness_poisson2D_u_p,
)
from src.mpm.structs.GridNode import NodeTwoPhase, NodeTwoPhase2D
from src.mpm.structs.Particle import ParticleCloudTwoPhase2D

pytestmark = [pytest.mark.unit, pytest.mark.mpm, pytest.mark.assembly, pytest.mark.cpu]


@ti.kernel
def update_single_point_twophase_particle(dt: ti.template(), particle: ti.template()):
    particle[0]._update_particle_state(
        dt,
        1.0,
        ti.Vector([9.0, 9.0]),
        ti.Vector([0.0, 0.0]),
        ti.Vector([1.0, 2.0]),
        ti.Vector([0.0, 0.0]),
        ti.Vector([3.0, 4.0]),
        ti.Vector([0.0, 0.0]),
    )


def test_single_point_twophase_particle_follows_solid_skeleton(taichi_runtime):
    dt = ti.field(float, shape=())
    dt[None] = 0.25
    particle = ParticleCloudTwoPhase2D.field(shape=1)
    particle.x[0] = ti.Vector([0.5, 0.5])

    update_single_point_twophase_particle(dt, particle)

    np.testing.assert_allclose(particle.x.to_numpy()[0], [0.75, 1.0])
    np.testing.assert_allclose(particle.v.to_numpy()[0], [1.0, 2.0])
    np.testing.assert_allclose(particle.vs.to_numpy()[0], [1.0, 2.0])
    np.testing.assert_allclose(particle.vf.to_numpy()[0], [3.0, 4.0])


def test_up_usf_applies_velocity_boundary_before_stress():
    from src.mpm.engines.ULSemiImplicitTwoPhaseEngine_u_p import ULSemiImplicitTwoPhaseEngine_u_p

    calls = []
    stages = (
        "calculate_interpolation",
        "compute_nodal_kinematics",
        "free_surface_detections",
        "compute_grid_velcity",
        "apply_dirichlet_constraints",
        "compute_stress_strains",
        "compute_forces",
        "apply_particle_traction_constraints",
        "apply_traction_constraints",
        "apply_absorbing_constraints",
        "compute_prediction_grid_kinematic",
        "apply_kinematic_constraints",
        "compute_Poisson_equation_implicit",
        "compute_correction_grid_kinematic",
        "compute_particle_kinematic",
    )
    engine = SimpleNamespace(**{name: lambda *args, name=name: calls.append(name) for name in stages})
    ULSemiImplicitTwoPhaseEngine_u_p.usf_updating(engine, None, None, None)
    assert calls.index("compute_grid_velcity") < calls.index("apply_dirichlet_constraints")
    assert calls.index("apply_dirichlet_constraints") < calls.index("compute_stress_strains")


@pytest.mark.parametrize("node_shape,expected", [((3, 2), 6), ((100, 1), 8)])
@pytest.mark.parametrize("up", (False, True))
def test_pressure_capacity_covers_moving_particle_support(taichi_runtime, node_shape, expected, up):
    from src.mpm.engines.PoissonEquation import MatrixFree

    sims = SimpleNamespace(
        poisson_equation=not up,
        poisson_equation_u_p=up,
        assemble_type="MatrixFree",
        calculate_reaction_force=False,
    )
    scene = SimpleNamespace(
        node=SimpleNamespace(shape=node_shape),
        particleNum=[2],
        element=SimpleNamespace(influenced_dofs=4),
    )
    matrix = MatrixFree()
    matrix.set_matrix_vector(1, sims, scene)  # Only one initial active node.
    assert matrix.right_hand_vector.shape == (expected,)
    assert matrix.cg.r.shape == (expected,)
    matrix.operator = SimpleNamespace(active_dofs=expected)
    assert matrix.checked_active_dofs() == expected
    matrix.operator.active_dofs = expected + 1
    for solve in (matrix.run_poisson, matrix.run_poisson_coo):
        with pytest.raises(RuntimeError, match="exceed allocated capacity"):
            solve(sims, scene)  # Must reject before any assembly kernel runs.


@pytest.mark.parametrize("beta", (0.0, 0.5, 1.0))
@pytest.mark.parametrize(
    "transfer",
    (
        kernel_kinemaitc_g2p_semitwophase2D,
        kernel_kinemaitc_g2p_semitwophase2D_FIC,
        kernel_kinemaitc_g2p_semitwophase2D_u_p,
    ),
)
def test_single_layer_pressure_transfer_matches_formulation(taichi_runtime, beta, transfer):
    dt = ti.field(float, shape=())
    dt[None] = 0.1
    particle = ParticleCloudTwoPhase2D.field(shape=1)
    particle.active.fill(1)
    particle.pressure.fill(99.0)
    node = NodeTwoPhase2D.field(shape=(2, 1))
    node.pressure.from_numpy(np.array([[0.0], [10.0]]))
    node.dpressure.from_numpy(np.array([[0.0], [2.0]]))
    node_size = ti.field(int, shape=1)
    node_size.fill(2)
    node_ids = ti.field(int, shape=2)
    node_ids.from_numpy(np.array([0, 1], dtype=np.int32))
    shape = ti.field(float, shape=2)
    shape.from_numpy(np.array([0.25, 0.75]))

    transfer(2, 1.0, beta, dt, 1, node, particle, node_ids, shape, node_size)

    # Transfer the solved total pressure, including prescribed nodal values.
    # u-p retains its particle history in the storage RHS, not a second time here.
    assert particle.pressure[0] == pytest.approx(beta * 0.75 * 10.0 + 0.75 * 2.0)


@pytest.mark.parametrize("beta", (0.0, 0.5, 1.0))
@pytest.mark.parametrize("old_particle_pressure", (3.5, 7.0))
def test_up_pressure_split_has_consistent_darcy_and_storage_terms(taichi_runtime, beta, old_particle_pressure):
    dt = ti.field(float, shape=())
    dt[None] = 0.1
    particle = ParticleCloudTwoPhase2D.field(shape=1)
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.vol.fill(1.0)
    particle.porosity.fill(0.4)
    particle.pressure.fill(old_particle_pressure)
    material_mapping = ti.field(int, shape=1)
    material = namedtuple("Water", "solid_density fluid_density permeability fluid_unit_weight fluid_bulk")(
        2650.0,
        1000.0,
        0.001,
        4900.0,
        2.2e8,
    )
    node = NodeTwoPhase2D.field(shape=(2, 1))
    node.pressure.from_numpy(np.array([[2.0], [4.0]]))
    node.dof.from_numpy(np.array([[0], [1]], dtype=np.int32))
    node_size = ti.field(int, shape=1)
    node_size.fill(2)
    node_ids = ti.field(int, shape=2)
    node_ids.from_numpy(np.array([0, 1], dtype=np.int32))
    shape = ti.field(float, shape=2)
    shape.from_numpy(np.array([0.25, 0.75]))
    dshape = ti.Vector.field(2, float, shape=2)
    dshape.from_numpy(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    rhs = ti.field(float, shape=2)
    stiffness = ti.field(float, shape=(1, 2, 2))

    kernel_assemble_residual_poisson_2D_u_p(
        2,
        0,
        1,
        particle,
        material_mapping,
        node_size,
        node_ids,
        node,
        material,
        ti.Vector([0.0, 0.0, 0.0]),
        dshape,
        shape,
        rhs,
        dt,
        beta,
    )
    kernel_assemble_local_stiffness_poisson2D_u_p(
        2,
        0,
        1,
        particle,
        material_mapping,
        shape,
        dshape,
        node_size,
        material,
        stiffness,
        dt,
    )

    mobility = material.permeability / material.fluid_unit_weight
    # Struct scalar fields can be bound to f32 at module import, before init.
    porosity = float(particle.porosity[0])
    storage = porosity / material.fluid_bulk / dt[None]
    weights = np.array([0.25, 0.75])
    gradient = np.array([-1.0, 1.0])
    expected_rhs = -mobility * beta * 2.0 * gradient + storage * (old_particle_pressure - beta * 3.5) * weights
    density = (1.0 - porosity) * material.solid_density + porosity * material.fluid_density
    expected_matrix = (mobility + dt[None] / density) * np.outer(gradient, gradient)
    expected_matrix += storage * np.outer(weights, weights)
    np.testing.assert_allclose(rhs.to_numpy(), expected_rhs, rtol=1.0e-12)
    np.testing.assert_allclose(stiffness.to_numpy()[0], expected_matrix, rtol=1.0e-12)


@pytest.mark.parametrize("formulation", ("up", "uvp", "fic"))
@pytest.mark.parametrize("relative_velocity", (0.0, 3.0))
def test_single_point_pressure_predictor_and_rigid_translation(taichi_runtime, formulation, relative_velocity):
    # Unequal particle quadrature weights: integration by parts cannot be
    # assumed exact after material points move through a partially filled cell.
    dt = ti.field(float, shape=())
    dt[None] = 0.1
    particle = ParticleCloudTwoPhase2D.field(shape=1)
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.vol.fill(1.0)
    particle.porosity.fill(0.4)
    node = NodeTwoPhase2D.field(shape=(2, 1))
    node.m.fill(2.0)
    node.ms.fill(1.0)
    node.mf.fill(1.0)
    node.dof.from_numpy(np.array([[0], [1]], dtype=np.int32))
    pressure = np.array([1.0, 3.0])
    solid_velocity = np.array([[2.0, 0.0], [4.0, 0.0]])
    fluid_velocity = np.array([[3.0, 0.0], [5.0, 0.0]])
    node.pressure.from_numpy(pressure[:, None])
    node.dpressure.from_numpy(pressure[:, None])
    node.momentums.from_numpy(solid_velocity[:, None, :])
    node.momentumf.from_numpy(fluid_velocity[:, None, :])
    node_size = ti.field(int, shape=1)
    node_size.fill(2)
    node_ids = ti.field(int, shape=2)
    node_ids.from_numpy(np.array([0, 1], dtype=np.int32))
    shape = ti.field(float, shape=2)
    shape.from_numpy(np.array([0.25, 0.75]))
    dshape = ti.Vector.field(2, float, shape=2)
    dshape.from_numpy(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    rhs = ti.field(float, shape=2)

    kernel_correct_grid_kinematic_semitwophase(
        2,
        1,
        particle,
        node_size,
        node_ids,
        node,
        dshape,
        shape,
        dt,
        1.0e-12,
    )

    # The old-pressure predictor must use the same gradient as the increment
    # correction; otherwise changing the pressure split changes the forces.
    total_pressure_force = node.forces.to_numpy() + node.forcef.to_numpy()
    fluid_pressure_force = node.forcef.to_numpy().copy()
    particle.pressure.fill(float(shape.to_numpy() @ pressure))
    particle.permeability.fill(1.0e-3)
    for predictor in (kernel_force_p2g_semitwophase2D, kernel_force_p2g_semitwophase2D_u_p):
        node.force.fill(0.0)
        node.forcef.fill(0.0)
        arguments = (
            2,
            1,
            ti.Vector([0.0, 0.0, 0.0]),
            node,
            particle,
            node_ids,
            shape,
            dshape,
            node_size,
            1.0,
        )
        if predictor is kernel_force_p2g_semitwophase2D:
            predictor(*arguments, True)
        else:
            predictor(*arguments)
        np.testing.assert_allclose(node.force.to_numpy(), total_pressure_force, rtol=1.0e-12)
        if predictor is kernel_force_p2g_semitwophase2D:
            np.testing.assert_allclose(node.forcef.to_numpy(), fluid_pressure_force, rtol=1.0e-12)

    # On a partially filled cell the pressure test function does not vanish
    # on the moving material surface. Dropping its boundary-flux term creates
    # a spurious pressure source even for rigid translation of both phases.
    for velocity in (node.momentum, node.momentums, node.momentumf):
        velocity.from_numpy(np.array([[[2.0, 0.0]], [[2.0, 0.0]]]))
    node.pressure.fill(0.0)
    particle.pressure.fill(0.0)
    node.momentumf.from_numpy(np.array([[[2.0 + relative_velocity, 0.0]]] * 2))
    kernel_single_point_porosity_p2g(2, 1, particle, node, node_ids, shape, node_size)
    np.testing.assert_allclose(node.porosity.to_numpy(), particle.porosity[0])
    np.testing.assert_allclose(node.pressure.to_numpy(), 0.0)
    if formulation == "uvp":
        kernel_assemble_residual_poisson_2D(2, 1, particle, node_size, node_ids, node, dshape, shape, rhs)
    else:
        mapping = ti.field(int, shape=1)
        material = namedtuple(
            "Water", "solid_density fluid_density permeability fluid_unit_weight fluid_bulk young poisson"
        )(
            2650.0,
            1000.0,
            0.001,
            9800.0,
            2.2e8,
            1.0e6,
            0.3,
        )
        if formulation == "up":
            kernel_assemble_residual_poisson_2D_u_p(
                2,
                0,
                1,
                particle,
                mapping,
                node_size,
                node_ids,
                node,
                material,
                ti.Vector([0.0, 0.0, 0.0]),
                dshape,
                shape,
                rhs,
                dt,
                1.0,
            )
        else:
            kernel_assemble_residual_poisson_FIC_2D(
                2,
                0,
                1,
                particle,
                mapping,
                node_size,
                node_ids,
                node,
                material,
                dshape,
                shape,
                rhs,
                dt,
                ti.Vector([1.0, 1.0]),
                1.0,
            )
    np.testing.assert_allclose(rhs.to_numpy(), 0.0, atol=1.0e-14)

    if formulation != "up":
        # Nonuniform porosity must still contribute div(n*(vf-vs)); merely
        # treating each particle's n as spatially constant loses this term.
        node.porosity.from_numpy(np.array([[0.2], [0.6]]))
        rhs.fill(0.0)
        if formulation == "uvp":
            kernel_assemble_residual_poisson_2D(2, 1, particle, node_size, node_ids, node, dshape, shape, rhs)
        else:
            kernel_assemble_residual_poisson_FIC_2D(
                2,
                0,
                1,
                particle,
                mapping,
                node_size,
                node_ids,
                node,
                material,
                dshape,
                shape,
                rhs,
                dt,
                ti.Vector([1.0, 1.0]),
                1.0,
            )
        expected = -(float(node.porosity[1, 0]) - float(node.porosity[0, 0])) * relative_velocity * shape.to_numpy()
        np.testing.assert_allclose(rhs.to_numpy(), expected, rtol=1.0e-12, atol=1.0e-14)


@pytest.mark.parametrize(
    "correction_kernel",
    (
        kernel_correct_grid_kinematic_semitwophase,
        kernel_correct_grid_kinematic_semitwophase_2DAxisy,
        kernel_correct_grid_kinematic_semitwophase_u_p,
    ),
)
def test_particle_pressure_correction_ignores_stale_inactive_particles(taichi_runtime, correction_kernel):
    dt = ti.field(float, shape=())
    dt[None] = 0.1
    particle = ParticleCloudTwoPhase2D.field(shape=2)
    particle.active.from_numpy(np.array([1, 0], dtype=np.uint8))
    particle.vol.fill(1.0)
    particle.porosity.fill(0.4)
    particle.x.from_numpy(np.array([[1.0, 0.5], [1.0, 0.5]]))

    node = NodeTwoPhase2D.field(shape=(2, 1))
    node.m.fill(2.0)
    node.ms.fill(1.0)
    node.mf.fill(1.0)
    node.pressure.fill(2.0)
    node.dpressure.from_numpy(np.array([[0.0], [10.0]]))

    node_size = ti.field(int, shape=2)
    node_size.from_numpy(np.array([2, 2], dtype=np.int32))
    node_ids = ti.field(int, shape=4)
    node_ids.from_numpy(np.array([0, 1, 0, 1], dtype=np.int32))
    shape = ti.field(float, shape=4)
    shape.fill(0.5)
    dshape = ti.Vector.field(2, float, shape=4)
    dshape.from_numpy(np.array([[-1.0, 0.0], [1.0, 0.0]] * 2))

    def run(particle_count):
        for value in (
            node.force,
            node.forces,
            node.forcef,
            node.momentum,
            node.momentums,
            node.momentumf,
            node.extra_stabilize,
        ):
            value.fill(0.0)
        arguments = (2, particle_count, particle, node_size, node_ids, node, dshape, shape, dt, 1.0e-12)
        correction_kernel(*arguments)
        return np.concatenate(
            [
                node.force.to_numpy().ravel(),
                node.forces.to_numpy().ravel(),
                node.forcef.to_numpy().ravel(),
                node.momentum.to_numpy().ravel(),
                node.momentums.to_numpy().ravel(),
                node.momentumf.to_numpy().ravel(),
                node.extra_stabilize.to_numpy().ravel(),
            ]
        )

    active_only = run(1)
    active_plus_stale_inactive = run(2)
    assert np.linalg.norm(active_only) > 0.0
    np.testing.assert_allclose(active_plus_stale_inactive, active_only)


def test_standard_pressure_operator_scales_with_timestep(taichi_runtime):
    dt = ti.field(float, shape=())
    cell_porosity = ti.field(float, shape=(3, 3))
    cell_phi = ti.field(float, shape=(3, 3))
    grid_type = ti.field(int, shape=(3, 3))
    diagonal = ti.field(float, shape=(3, 3))
    off_diagonal = ti.Vector.field(2, float, shape=(3, 3))
    material = namedtuple("MaterialFixture", "solid_density fluid_density")(2650.0, 1000.0)

    grid_type[1, 1] = 1
    cell_porosity[1, 1] = 0.4

    def assemble(timestep):
        dt[None] = timestep
        diagonal.fill(0.0)
        off_diagonal.fill(0.0)
        element_size = ti.Vector([0.1, 0.2])
        kernel_assemble_A(
            0,
            dt,
            material,
            element_size,
            cell_porosity,
            cell_phi,
            grid_type,
            diagonal,
            off_diagonal,
        )
        return diagonal.to_numpy().copy(), off_diagonal.to_numpy().copy()

    diagonal_1, off_diagonal_1 = assemble(1.0e-4)
    diagonal_2, off_diagonal_2 = assemble(2.0e-4)
    mobility = 0.6 / material.solid_density + 0.4 / material.fluid_density
    assert diagonal_1[1, 1] == pytest.approx(-2.0e-4 * mobility * (1.0 / 0.1**2 + 1.0 / 0.2**2))
    np.testing.assert_allclose(diagonal_2, 2.0 * diagonal_1)
    np.testing.assert_allclose(off_diagonal_2, 2.0 * off_diagonal_1)


@pytest.mark.isolated_dimension(3)
def test_standard_pressure_operator_has_3d_laplacian_units(taichi_runtime):
    dt = ti.field(float, shape=())
    dt[None] = 1.0e-4
    cell_porosity = ti.field(float, shape=(3, 3, 3))
    cell_phi = ti.field(float, shape=(3, 3, 3))
    grid_type = ti.field(int, shape=(3, 3, 3))
    diagonal = ti.field(float, shape=(3, 3, 3))
    off_diagonal = ti.Vector.field(3, float, shape=(3, 3, 3))
    material = namedtuple("MaterialFixture", "solid_density fluid_density")(2650.0, 1000.0)
    spacing = ti.Vector([0.1, 0.2, 0.4])

    grid_type[1, 1, 1] = 1
    cell_porosity[1, 1, 1] = 0.4
    kernel_assemble_A(
        0,
        dt,
        material,
        spacing,
        cell_porosity,
        cell_phi,
        grid_type,
        diagonal,
        off_diagonal,
    )

    mobility = 0.6 / material.solid_density + 0.4 / material.fluid_density
    expected = -2.0e-4 * mobility * sum(1.0 / float(h) ** 2 for h in spacing)
    assert diagonal[1, 1, 1] == pytest.approx(expected)


@pytest.mark.isolated_dimension(3)
def test_3d_pressure_correction_uses_each_axis_spacing(taichi_runtime):
    dt = ti.field(float, shape=())
    dt[None] = 0.1
    node = NodeTwoPhase.field(shape=(27, 1))
    cell_type = ti.field(int, shape=(2, 2, 2))
    cell_porosity = ti.field(float, shape=(2, 2, 2))
    cell_dpressure = ti.field(float, shape=(2, 2, 2))
    cell_phi = ti.field(float, shape=(2, 2, 2))
    is_rigid = ti.field(int, shape=1)
    material = namedtuple("MaterialFixture", "solid_density fluid_density")(2000.0, 1000.0)
    spacing = np.array([0.1, 0.2, 0.4])

    pressure = np.zeros((2, 2, 2))
    for index in np.ndindex(pressure.shape):
        pressure[index] = material.solid_density * np.dot([1.0, 2.0, 3.0], spacing * index)
    cell_type.fill(1)
    cell_dpressure.from_numpy(pressure)
    kernel_correct_grid_kinematic_semitwophase_mg_3D(
        0,
        node,
        dt,
        1.0e-12,
        ti.Vector(spacing.tolist()),
        ti.Vector([3, 3, 3]),
        cell_type,
        cell_porosity,
        cell_dpressure,
        cell_phi,
        material,
        is_rigid,
    )

    np.testing.assert_allclose(node.forces.to_numpy()[13, 0], [-1.0, -2.0, -3.0])


@pytest.mark.parametrize("solver_type", ("cartesian", "axisymmetric"))
def test_mixture_mgpcg_supports_negative_definite_operator_and_reports_exhaustion(
    taichi_runtime,
    solver_type,
):
    if solver_type == "cartesian":
        solver = MGPCGMixPoissonSolver(2, (3, 3), n_mg_levels=1, bottom_smoothing=2)
    else:
        solver = MGPCGMixPoissonSolver_Axi(2, (3, 3), 1.0, n_mg_levels=1, bottom_smoothing=2)
    solver.grid_type[0][1, 1] = solver.FLUID
    solver.Adiag[0][1, 1] = -1.0
    solver.b[1, 1] = 1.0

    assert solver.solve(max_iters=2)
    assert np.isclose(solver.x[1, 1], -1.0)

    solver.initialize()
    solver.grid_type[0][1, 1] = solver.FLUID
    solver.Adiag[0][1, 1] = -1.0
    solver.b[1, 1] = 1.0
    assert not solver.solve(max_iters=0)
    assert solver.breakdown_reason == "max_iterations"


@pytest.mark.parametrize("solver_type", ("cartesian", "axisymmetric"))
def test_mixture_multigrid_coarsening_handles_odd_grid_shapes(taichi_runtime, solver_type):
    if solver_type == "cartesian":
        solver = MGPCGMixPoissonSolver(2, (5, 3), n_mg_levels=2)
    else:
        solver = MGPCGMixPoissonSolver_Axi(2, (5, 3), 1.0, n_mg_levels=2)
    solver.grid_type[0].fill(solver.SOLID)
    solver.grid_type[0][4, 2] = solver.FLUID

    solver.init_gridtype(solver.grid_type[0], solver.grid_type[1])

    expected = np.full((3, 2), solver.SOLID, dtype=np.int32)
    expected[2, 1] = solver.FLUID
    np.testing.assert_array_equal(solver.grid_type[1].to_numpy(), expected)


def test_mg_pressure_correction_treats_solid_faces_as_neumann(taichi_runtime):
    dt = ti.field(float, shape=())
    dt[None] = 1.0e-4
    node = NodeTwoPhase2D.field(shape=(16, 1))
    cell_type = ti.field(int, shape=(3, 3))
    cell_porosity = ti.field(float, shape=(3, 3))
    cell_dpressure = ti.field(float, shape=(3, 3))
    cell_phi = ti.field(float, shape=(3, 3))
    is_rigid = ti.field(int, shape=1)
    material = namedtuple("MaterialFixture", "solid_density fluid_density")(2650.0, 1000.0)

    cell_type.fill(2)
    cell_type[1, 1] = 1
    cell_dpressure[1, 1] = 1000.0
    kernel_correct_grid_kinematic_semitwophase_mg(
        0,
        node,
        dt,
        1.0e-12,
        ti.Vector([0.1, 0.1]),
        ti.Vector([4, 4]),
        cell_type,
        cell_porosity,
        cell_dpressure,
        cell_phi,
        material,
        is_rigid,
    )

    nodal = node.to_numpy()
    assert np.allclose(nodal["momentums"], 0.0)
    assert np.allclose(nodal["momentumf"], 0.0)


def test_mg_pressure_correction_treats_outside_as_solid(taichi_runtime):
    dt = ti.field(float, shape=())
    dt[None] = 0.1
    node = NodeTwoPhase2D.field(shape=(6, 1))
    cell_type = ti.field(int, shape=(2, 2))
    cell_porosity = ti.field(float, shape=(2, 2))
    cell_dpressure = ti.field(float, shape=(2, 2))
    cell_phi = ti.field(float, shape=(2, 2))
    is_rigid = ti.field(int, shape=1)
    material = namedtuple("MaterialFixture", "solid_density fluid_density")(2.0, 4.0)

    cell_type.from_numpy(np.array([[1, 0], [1, 0]], dtype=np.int32))
    cell_phi.from_numpy(np.array([[-0.5, 0.5], [-0.5, 0.5]]))
    cell_dpressure[1, 0] = 8.0
    kernel_correct_grid_kinematic_semitwophase_mg(
        0,
        node,
        dt,
        1.0e-12,
        ti.Vector([1.0, 1.0]),
        ti.Vector([3, 2]),
        cell_type,
        cell_porosity,
        cell_dpressure,
        cell_phi,
        material,
        is_rigid,
    )

    nodal = node.to_numpy()
    np.testing.assert_allclose(nodal["forces"][[1, 4], 0, 0], [-4.0, -2.0])
    np.testing.assert_allclose(nodal["forcef"][[1, 4], 0, 0], [-2.0, -1.0])


def test_pressure_dof_estimate_counts_active_nodes(taichi_runtime):
    node = NodeTwoPhase2D.field(shape=(4, 1))
    node[0, 0].m = 1.0
    node[2, 0].m = 2.0
    assert estimate_active_dofs_poisson(1.0e-12, node) == 2


def test_tpic_pressure_transfer_preserves_affine_field(taichi_runtime):
    particle = ParticleCloudTwoPhase2D.field(shape=1)
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.m.fill(3.0)
    particle.ms.fill(2.0)
    particle.mf.fill(1.0)
    particle.x[0] = ti.Vector([0.25, 0.0])
    particle.pressure.fill(10.0)
    particle.pressure_gradient[0] = ti.Vector([3.0, 0.0])
    node = NodeTwoPhase2D.field(shape=(2, 1))
    node_ids = ti.field(int, shape=2)
    node_ids.from_numpy(np.array([0, 1], dtype=np.int32))
    node_size = ti.field(int, shape=1)
    node_size.fill(2)
    shape = ti.field(float, shape=2)
    shape.from_numpy(np.array([0.75, 0.25]))
    is_rigid = ti.field(int, shape=1)

    kernel_mass_momentum_p2g_twophase_u_p(
        2,
        1,
        node,
        particle,
        node_ids,
        shape,
        node_size,
        0,
        is_rigid,
    )
    kernel_pressure_tpic_p2g_correction_2D(
        2,
        1,
        ti.Vector([1.0, 1.0]),
        ti.Vector([2, 1]),
        node,
        particle,
        node_ids,
        shape,
        node_size,
    )
    kernel_compute_grid_velocity_twophase(1.0e-12, node)

    np.testing.assert_allclose(node.pressure.to_numpy()[:, 0], [9.25, 12.25])


def test_gimp_pressure_boundary_does_not_erase_an_extra_cell_layer(taichi_runtime):
    element = QuadrilateralElement4Nodes("Q4N2D", 1, 0)
    element.gridSum = 16
    element.flag = ti.field(int, shape=16)
    element.ti_nodal_coords = ti.Vector.field(2, float, shape=16)
    node = NodeTwoPhase2D.field(shape=(16, 1))
    node.m.fill(1.0)
    cell_volume_fraction = ti.field(float, shape=9)
    cell_volume_fraction.fill(1.0)
    cell_volume_fraction[4] = 0.0
    is_rigid = ti.field(int, shape=1)

    active_dofs = element.find_active_nodes_poisson(
        1.0e-12,
        node,
        cell_volume_fraction,
        ti.Vector([3, 3]),
        is_rigid,
        shape_function="GIMP",
    )

    assert active_dofs == 12


@pytest.mark.parametrize("minimum_pressure", (0.0, -10.0))
def test_twophase_pressure_respects_configured_cavitation_limit(taichi_runtime, minimum_pressure):
    particle = ParticleCloudTwoPhase2D.field(shape=3)
    material_mapping = ti.field(int, shape=3)
    material_mapping.from_numpy(np.array([0, 1, 2], dtype=np.int32))
    particle.active.fill(1)
    particle.pressure.from_numpy(np.array([-5.0, 5.0, 5.0]))
    particle[2].free_surface = 1

    kernel_clamp_twophase_particle_pressure(0, 3, material_mapping, minimum_pressure, particle)

    # A near-surface flag does not place an interior material point on p=0.
    np.testing.assert_allclose(particle.pressure.to_numpy(), [max(-5.0, minimum_pressure), 5.0, 5.0])


@pytest.mark.parametrize("formulation", ("up", "uvp"))
def test_single_layer_free_surface_pressure_is_not_cavitation_pressure(taichi_runtime, formulation):
    from src.mpm.engines.ULSemiImplicitTwoPhaseEngine import ULSemiImplicitTwoPhaseEngine
    from src.mpm.engines.ULSemiImplicitTwoPhaseEngine_u_p import ULSemiImplicitTwoPhaseEngine_u_p

    engine_type = ULSemiImplicitTwoPhaseEngine_u_p if formulation == "up" else ULSemiImplicitTwoPhaseEngine
    engine = object.__new__(engine_type)
    engine._iter_twophase_materials = lambda scene: [(1, 0, 1, SimpleNamespace(cavitation_pressure=-1.0e30))]
    node = NodeTwoPhase2D.field(shape=(1, 1))
    node.m.fill(1.0)
    node.pressure.fill(5.0)
    node.dof.fill(-1)
    engine.matrix_free = SimpleNamespace(unknow_vector=ti.field(float, shape=1))

    engine.update_nodal_pressure_2D(
        SimpleNamespace(pressure_beta=1.0), SimpleNamespace(mass_cut_off=1.0e-12, node=node)
    )

    assert node.dpressure[0, 0] == pytest.approx(-5.0)


@pytest.mark.parametrize("minimum_pressure", (0.0, -10.0))
def test_cavitation_limits_pressure_increment_before_velocity_correction(taichi_runtime, minimum_pressure):
    from src.mpm.engines.EngineKernel import (
        kernel_limit_twophase_cell_pressure_increment,
        kernel_limit_twophase_nodal_pressure_increment,
    )

    node = NodeTwoPhase2D.field(shape=(3, 1))
    node.m.from_numpy(np.array([[1.0], [1.0], [0.0]]))
    node.pressure.from_numpy(np.array([[2.0], [2.0], [2.0]]))
    node.dpressure.from_numpy(np.array([[-5.0], [1.0], [-5.0]]))

    kernel_limit_twophase_nodal_pressure_increment(1.0e-12, 1.0, minimum_pressure, node)

    expected_increment = max(-5.0, minimum_pressure - 2.0)
    np.testing.assert_allclose(node.dpressure.to_numpy().ravel(), [expected_increment, 1.0, -5.0])

    cell_type = ti.field(ti.u8, shape=2)
    cell_pressure = ti.field(float, shape=2)
    cell_dpressure = ti.field(float, shape=2)
    cell_type.from_numpy(np.array([1, 0], dtype=np.uint8))
    cell_pressure.from_numpy(np.array([2.0, 2.0]))
    cell_dpressure.from_numpy(np.array([-5.0, -5.0]))

    kernel_limit_twophase_cell_pressure_increment(1.0, minimum_pressure, cell_type, cell_pressure, cell_dpressure)

    np.testing.assert_allclose(cell_dpressure.to_numpy(), [expected_increment, -5.0])


def test_free_surface_prescribes_pressure_increment_and_eliminates_rhs(taichi_runtime):
    node = NodeTwoPhase2D.field(shape=(2, 1))
    node.m.fill(1.0)
    node.pressure.from_numpy(np.array([[7.0], [5.0]]))
    node.dof.from_numpy(np.array([[0], [-1]], dtype=np.int32))
    unknown = ti.field(float, shape=1)
    unknown[0] = 3.0

    kernel_update_nodal_pressure_2D(1.0e-12, 0.5, 1.0, node, unknown)
    np.testing.assert_allclose(node.dpressure.to_numpy().ravel(), [3.0, -1.5])

    particle = ParticleCloudTwoPhase2D.field(shape=1)
    particle[0].active = 1
    particle[0].materialID = 1
    particle[0].bodyID = 0
    node_size = ti.field(int, shape=1)
    node_size[0] = 2
    node_ids = ti.field(int, shape=2)
    node_ids.from_numpy(np.array([0, 1], dtype=np.int32))
    local_stiffness = ti.field(float, shape=(1, 2, 2))
    local_stiffness.from_numpy(np.array([[[4.0, -2.0], [-2.0, 4.0]]]))
    rhs = ti.field(float, shape=1)
    rhs[0] = 4.0

    kernel_eliminate_pressure_increment_dirichlet(
        2,
        1,
        particle,
        node_size,
        node_ids,
        node,
        local_stiffness,
        1.0e-12,
        0.5,
        1.0,
        rhs,
    )
    assert rhs[0] == pytest.approx(1.0)


def test_mg_single_layer_requires_nonincremental_pressure_projection(taichi_runtime):
    from src.mpm.Simulation import Simulation

    sims = Simulation()
    sims.dimension = 2
    sims.solver_type = "SemiImplicit"
    sims.material_type = "TwoPhaseSingleLayer"
    sims.mapping = "USL"
    sims.pressure_solver = "MGPCG"
    sims.linear_solver = "MGPCG"
    sims.pressure_beta = 1.0
    with pytest.raises(RuntimeError, match="pressure_beta=0"):
        sims.validate_configuration()

    sims.pressure_beta = 0.0
    sims.validate_configuration()


@pytest.mark.parametrize("beta", (0.0, 0.5, 1.0))
def test_up_recovers_darcy_velocity_without_changing_solid_update(taichi_runtime, beta):
    from src.mpm.engines.ULSemiImplicitTwoPhaseEngine_u_p import ULSemiImplicitTwoPhaseEngine_u_p

    particle = ParticleCloudTwoPhase2D.field(shape=3)
    particle.active.from_numpy(np.array([1, 1, 0], dtype=np.uint8))
    particle.materialID.from_numpy(np.array([2, 1, 0], dtype=np.uint8))
    particle.porosity.from_numpy(np.array([0.25, 0.5, 0.5]))
    particle.vf.from_numpy(np.array([[7.0, 8.0], [7.0, 8.0], [7.0, 8.0]]))
    node = NodeTwoPhase2D.field(shape=(3, 1))
    node.momentum.from_numpy(np.array([[[0.3, -0.1]], [[0.3, -0.1]], [[0.3, -0.1]]]))
    node.pressure.from_numpy(np.array([[0.0], [40.0], [80.0]]))
    node.dpressure.from_numpy(np.array([[200.0], [300.0], [0.0]]) - beta * node.pressure.to_numpy())
    node_size = ti.field(int, shape=3)
    node_size.fill(3)
    node_ids = ti.field(int, shape=9)
    node_ids.from_numpy(np.tile(np.arange(3, dtype=np.int32), 3))
    shape = ti.field(float, shape=9)
    shape.from_numpy(np.tile([0.5, 0.2, 0.3], 3))
    dshape = ti.Vector.field(2, float, shape=9)
    dshape.from_numpy(np.tile([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], (3, 1)))
    material_mapping = ti.field(int, shape=2)
    material_mapping.from_numpy(np.array([1, 0], dtype=np.int32))
    props = namedtuple("DarcyWater", "fluid_density permeability fluid_unit_weight cavitation_pressure")
    materials = [props(1000.0, 0.5, 5000.0, -1e30), props(1000.0, 1.0, 5000.0, -1e30)]
    engine = object.__new__(ULSemiImplicitTwoPhaseEngine_u_p)
    engine._iter_twophase_materials = lambda _: [(1, 0, 1, materials[0]), (2, 1, 2, materials[1])]
    dt = ti.field(float, shape=())
    dt[None] = 0.01
    sims = SimpleNamespace(alphaPIC=1.0, pressure_beta=beta, dt=dt, gravity=ti.Vector([0.0, -0.2, 0.0]))
    scene = SimpleNamespace(
        particleNum=[3],
        particle=particle,
        node=node,
        material=SimpleNamespace(materialID=material_mapping),
        element=SimpleNamespace(grid_nodes=3, LnID=node_ids, shape_fn=shape, dshape_fn=dshape, node_size=node_size),
    )
    engine.compute_particle_kinematics_twophase2D(sims, scene)

    # grad(p)=(100,-200), rho*g=(0,-200): only horizontal Darcy flow.
    np.testing.assert_allclose(particle.vf.to_numpy()[:2], [[0.3 - 0.08, -0.1], [0.3 - 0.02, -0.1]], atol=1e-12)
    np.testing.assert_allclose(particle.vs.to_numpy()[:2], [[0.3, -0.1], [0.3, -0.1]], atol=1e-12)
    np.testing.assert_allclose(particle.x.to_numpy()[:2], [[0.003, -0.001], [0.003, -0.001]], atol=1e-12)
    np.testing.assert_allclose(particle.pressure.to_numpy()[:2], [160.0, 160.0], atol=1e-12)
    np.testing.assert_array_equal(particle.vf.to_numpy()[2], [7.0, 8.0])

    materials[1] = materials[1]._replace(permeability=0.0)
    engine.compute_particle_kinematics_twophase2D(sims, scene)
    np.testing.assert_allclose(particle.vf.to_numpy()[0], [0.3, -0.1], atol=1e-12)
    node.dpressure.from_numpy(np.array([[200.0], [200.0], [0.0]]) - beta * node.pressure.to_numpy())
    engine.compute_particle_kinematics_twophase2D(sims, scene)
    np.testing.assert_allclose(particle.vf.to_numpy()[:2], particle.vs.to_numpy()[:2], atol=1e-12)


@pytest.mark.parametrize("densities", ((2400.0, 800.0), (1000.0, 1000.0)))
@pytest.mark.parametrize("beta", (0.0, 0.5, 1.0))
def test_fic_gradient_projection_matches_poisson_mobility(taichi_runtime, densities, beta):
    from src.mpm.engines.ULSemiImplicitTwoPhaseEngine import ULSemiImplicitTwoPhaseEngine

    particle = ParticleCloudTwoPhase2D.field(shape=1)
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.vol.fill(1.0)
    particle.porosity.fill(0.25)
    node = NodeTwoPhase2D.field(shape=(2, 1))
    weights = np.array([0.25, 0.75])
    rho_s, rho_f = densities
    node.ms.from_numpy((weights * 0.75 * rho_s)[:, None])
    node.mf.from_numpy((weights * 0.25 * rho_f)[:, None])
    node.m.from_numpy(node.ms.to_numpy() + node.mf.to_numpy())
    node.weight.from_numpy(weights[:, None])
    node.pressure.from_numpy(np.array([[4.0], [14.0]]))
    node_size = ti.field(int, shape=1)
    node_size.fill(2)
    node_ids = ti.field(int, shape=2)
    node_ids.from_numpy(np.array([0, 1], dtype=np.int32))
    shape = ti.field(float, shape=2)
    shape.from_numpy(weights)
    dshape = ti.Vector.field(2, float, shape=2)
    dshape.from_numpy(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    dt = ti.field(float, shape=())
    dt[None] = 0.01
    mapping = ti.field(int, shape=1)
    material = namedtuple("FICWater", "solid_density fluid_density young poisson")(rho_s, rho_f, 1e6, 0.25)
    engine = object.__new__(ULSemiImplicitTwoPhaseEngine)
    engine._iter_twophase_materials = lambda _: [(1, 0, 1, material)]
    scene = SimpleNamespace(
        node=node,
        particle=particle,
        material=SimpleNamespace(materialID=mapping),
        element=SimpleNamespace(grid_nodes=2, LnID=node_ids, shape_fn=shape, dshape_fn=dshape, node_size=node_size),
    )
    node.extra_stabilize.fill(1234.0)  # No stale history, including new nodes/restarts.
    engine.project_fic_pressure_gradient(scene)
    mobility = 0.75 / rho_s + 0.25 / rho_f
    np.testing.assert_allclose(node.extra_stabilize.to_numpy()[:, 0], [[-10 * mobility, 0.0]] * 2, atol=1e-12)

    node.dof.from_numpy(np.array([[0], [1]], dtype=np.int32))
    rhs = ti.field(float, shape=2)
    kernel_assemble_residual_poisson_FIC_2D(
        2,
        0,
        1,
        particle,
        mapping,
        node_size,
        node_ids,
        node,
        material,
        dshape,
        shape,
        rhs,
        dt,
        ti.Vector([1.0, 1.0]),
        beta,
    )
    wave_speed = np.sqrt(
        material.young * (1 - material.poisson) / ((1 + material.poisson) * (1 - 2 * material.poisson) * rho_s)
    )
    tau = 0.1 / wave_speed
    # For affine p, the stabilization vanishes when dp=(1-beta)*p_old.
    expected = (1 - beta) * tau * mobility * 10 * np.array([-1.0, 1.0])
    np.testing.assert_allclose(rhs.to_numpy(), expected, atol=1e-14)
