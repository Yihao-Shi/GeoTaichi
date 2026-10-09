import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.mpdem.engines.SoftAffineIPCEngine import SoftAffineIPCEngine
from src.mpm.MaterialManager import SoftParticleMaterialManager
from src.mpdem.engines.SoftAffineIPCOperator import (
    SoftAffineIPCOperator,
    _validate_soft_affine_lagged_friction_configuration,
    soft_affine_friction_capabilities,
)
from src.dem.engines.AffineBodyOperator import TaichiAffineBodyOperator
from src.dem.engines.AffineBodyState import AffineBodyState
from src.linear_solver.BuildTriplet import BuildTriplet
from src.physics_model.contact_model.ipc.ContactAssembly import pullback_dense
from src.physics_model.contact_model.ipc.IPC import (
    ipc_barrier_distance_terms_py,
    ipc_toolkit_barrier_distance2_terms,
    ipc_toolkit_barrier_distance_terms,
)
from src.physics_model.contact_model.ipc.LevelSetAffine import (
    TrilinearLevelSet,
    affine_basis,
    point_affine_levelset_gap,
)
from src.utils.PrefixSum import PrefixSumExecutor


def test_soft_particle_material_manager_rejects_plastic_models():
    manager = SoftParticleMaterialManager()
    for model in ("DruckerPrager", "VonMises"):
        with pytest.raises(ValueError, match="hyperelastic-only"):
            manager._normalize_model_name(model, allow_plastic=True)


class _ArrayField:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def from_numpy(self, values):
        self.values = np.asarray(values, dtype=np.float64).copy()

    def to_numpy(self):
        return self.values.copy()

    def fill(self, value):
        self.values.fill(value)


class _ScalarField:
    def __init__(self, value=0.0):
        self.value = value

    def __getitem__(self, _index):
        return self.value

    def __setitem__(self, _index, value):
        self.value = value


def test_soft_grid_dof_scan_rejects_inactive_nodes():
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    grid_type = ti.types.struct(m=float)
    operator = object.__new__(SoftAffineIPCOperator)
    operator.soft_grid_num = 4097
    operator.scene = SimpleNamespace(soft_grid=grid_type.field(shape=operator.soft_grid_num))
    operator.soft_node2dof = ti.field(ti.i32, shape=operator.soft_grid_num)
    operator.soft_dof2node = ti.field(ti.i32, shape=operator.soft_grid_num)
    operator.soft_node_fixed = ti.field(ti.i32, shape=operator.soft_grid_num)
    operator.soft_node_fixed.fill(0)
    active_nodes = np.arange(1, operator.soft_grid_num, 3)
    masses = np.zeros(operator.soft_grid_num, dtype=np.float64)
    masses[active_nodes] = 1.0
    operator.scene.soft_grid.m.from_numpy(masses)
    scan = np.cumsum(masses > 0.0, dtype=np.int32)
    operator.soft_node2dof.from_numpy(scan)

    active = operator._fill_soft_dof()

    assert active == active_nodes.size
    expected = scan.copy()
    expected[masses == 0.0] = -1
    np.testing.assert_array_equal(operator.soft_node2dof.to_numpy(), expected)
    np.testing.assert_array_equal(operator.soft_dof2node.to_numpy()[:active], active_nodes)


def test_soft_affine_masks_existing_mechanical_grid_constraints():
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    operator = object.__new__(SoftAffineIPCOperator)
    operator.soft_grid_num = 5
    operator.soft_velocity_constraint_num = 2
    operator.soft_node_fixed = ti.field(ti.i32, shape=5)
    fixed = ti.field(ti.i32, shape=2)
    fixed.from_numpy(np.asarray([1, 4], dtype=np.int32))
    operator.scene = SimpleNamespace(soft_velocity_constraint=fixed)

    operator._initialize_soft_fixed_nodes()

    np.testing.assert_array_equal(operator.soft_node_fixed.to_numpy(), [0, 1, 0, 0, 1])


def test_soft_affine_levelset_contact_uses_implicit_mixed_assembler():
    operator = object.__new__(SoftAffineIPCOperator)
    events = []
    operator.mixed_pair_num = _ScalarField(1)
    operator.affine = SimpleNamespace(levelset_contact=True)
    operator.fully_implicit = True
    operator._update_mixed_levelset_transforms = lambda: events.append(("transforms",))
    operator._assemble_mixed_levelset_contact = lambda need_matrix: events.append(("levelset", need_matrix))
    operator._count_mixed_contact_types = lambda: pytest.fail(
        "triangle feature classification used for Level Set contact"
    )

    SoftAffineIPCOperator._assemble_mixed_contact(operator, need_matrix=True)

    assert events == [("transforms",), ("levelset", True)]


def test_soft_affine_empty_mixed_broadphase_skips_contact_kernels():
    operator = object.__new__(SoftAffineIPCOperator)
    operator.mixed_pair_num = _ScalarField(0)
    operator.affine = SimpleNamespace(levelset_contact=True)
    operator._assemble_mixed_levelset_contact = lambda _need_matrix: pytest.fail(
        "empty broad phase launched the Level Set contact kernel"
    )
    operator._assemble_mixed_levelset_lagged_friction = lambda _need_matrix: pytest.fail(
        "empty broad phase launched the Level Set friction kernel"
    )

    SoftAffineIPCOperator._assemble_mixed_contact(operator, need_matrix=True)


def test_soft_affine_levelset_mixed_kernel_matches_reference_gradient():
    operator, levelset, controls = _make_levelset_mixed_kernel_harness()
    _clear_coupled_contact_system(operator)

    operator._assemble_mixed_contact(need_matrix=True)

    point = np.array([0.316, 0.011, 0.007])
    reference = point_affine_levelset_gap(point, controls, 1.0, levelset)
    barrier = ipc_barrier_distance_terms_py(reference.gap, 0.1, kappa=1.0e4)
    coefficient = operator.scale * 0.7
    expected_gradient = coefficient * barrier[1] * reference.gradient
    gradient = operator.global_grad.to_numpy()
    matrix_upper = operator.hash_triplet.values.to_numpy()
    matrix = matrix_upper + matrix_upper.T - np.diag(np.diag(matrix_upper))

    assert 0.0 < reference.gap < 0.1
    assert operator.energy[None] == pytest.approx(coefficient * barrier[0], rel=2.0e-11)
    np.testing.assert_allclose(
        gradient[:12],
        expected_gradient[3:],
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        gradient[12:15],
        expected_gradient[:3],
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        gradient[:12].reshape(4, 3).sum(axis=0) + gradient[12:15],
        0.0,
        atol=2.0e-11,
    )
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-14)
    assert np.linalg.eigvalsh(matrix).min() >= -1.0e-9


def test_soft_affine_levelset_semi_ipc_assembles_and_updates_multiplier():
    operator, _levelset, _controls = _make_levelset_mixed_kernel_harness()
    _enable_semi_ipc(operator)
    _clear_coupled_contact_system(operator)

    operator._assemble_mixed_contact(need_matrix=True)
    operator.accept_semi_update_device()

    assert np.isfinite(float(operator.energy[None]))
    assert int(operator.mixed_levelset_contact_count[None]) == 1
    assert int(operator.semi_count[None]) == 1
    assert np.max(operator.semi_multiplier.to_numpy()) > 0.0


def test_soft_affine_timestep_update_reaches_compiled_contact_kernel():
    operator, _levelset, _controls = _make_levelset_mixed_kernel_harness()
    _clear_coupled_contact_system(operator)
    operator._assemble_mixed_contact(need_matrix=False)
    original_energy = float(operator.energy[None])

    operator.set_timestep(0.5 * operator.dt)
    _clear_coupled_contact_system(operator)
    operator._assemble_mixed_contact(need_matrix=False)

    assert float(operator.energy[None]) == pytest.approx(0.25 * original_energy, rel=2.0e-11)
    assert float(operator.dt_device[None]) == pytest.approx(5.0e-3)
    assert float(operator.affine.dt_device[None]) == pytest.approx(5.0e-3)


def test_soft_affine_levelset_lagged_friction_matches_energy_derivatives():
    operator, _levelset, controls = _make_levelset_mixed_kernel_harness()
    # Avoid placing the query exactly on grid-cell faces; the production
    # trilinear sampler treats its interpolation box as half-open.
    point = np.array([0.316, 0.011, 0.007], dtype=np.float64)
    operator.scene.soft_point[0].x = ti.Vector(point)
    operator.mixed_levelset_frozen_point.from_numpy(np.vstack((point, np.zeros(3, dtype=np.float64))))
    operator.affine.y.from_numpy(controls)
    operator.affine.hat_y.from_numpy(controls)
    operator.affine.pp_mu[0, 0] = 0.4
    operator.affine._freeze_levelset_lagged_friction_geometry()
    operator._update_mixed_levelset_transforms()
    operator.soft_hat_x.from_numpy(
        np.vstack(
            (
                point - np.array([0.0, 4.0e-6, 0.0]),
                np.zeros(3, dtype=np.float64),
            )
        )
    )

    def evaluate(displacement_y, need_matrix=False):
        displacement = np.zeros(6, dtype=np.float64)
        displacement[1] = displacement_y
        operator.soft_disp.from_numpy(displacement)
        operator._update_mixed_levelset_transforms()
        _clear_coupled_contact_system(operator)
        operator._assemble_mixed_levelset_lagged_friction(need_matrix)
        return (
            float(operator.energy[None]),
            operator.global_grad.to_numpy(),
            operator.hash_triplet.values.to_numpy(),
        )

    energy, gradient, matrix_upper = evaluate(0.0, need_matrix=True)
    step = 1.0e-8
    energy_plus, gradient_plus, _ = evaluate(step)
    energy_minus, gradient_minus, _ = evaluate(-step)
    row = operator.affine_dof + 1
    matrix = matrix_upper + matrix_upper.T - np.diag(np.diag(matrix_upper))

    assert energy > 0.0
    assert int(operator.mixed_friction_count[None]) == 1
    assert gradient[row] == pytest.approx(
        (energy_plus - energy_minus) / (2.0 * step),
        rel=3.0e-6,
        abs=1.0e-10,
    )
    assert matrix[row, row] == pytest.approx(
        (gradient_plus[row] - gradient_minus[row]) / (2.0 * step),
        rel=3.0e-5,
        abs=1.0e-8,
    )
    np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-14)
    assert np.linalg.eigvalsh(matrix).min() >= -1.0e-9


def test_soft_affine_levelset_ccd_catches_transient_pass_through():
    operator, levelset, controls = _make_levelset_mixed_kernel_harness()
    start = np.array([-0.36, 0.011, 0.007])
    direction = np.array([0.72, 0.0, 0.0])
    operator.scene.soft_point[0].x = ti.Vector(start)
    soft_direction = np.zeros(6, dtype=np.float64)
    soft_direction[:3] = direction
    operator.soft_direction.from_numpy(soft_direction)
    operator.affine.direction_y.fill(0.0)
    operator.ccd_alpha[None] = 1.0

    operator._compute_mixed_levelset_ccd_alpha(0.2, 64)

    alpha = float(operator.ccd_alpha[None])
    end_gap = point_affine_levelset_gap(start + direction, controls, 1.0, levelset).gap
    accepted_gap = point_affine_levelset_gap(start + alpha * direction, controls, 1.0, levelset).gap
    assert end_gap > 0.0
    assert 0.0 < alpha < 0.5
    assert accepted_gap > 0.0


def test_soft_affine_residual_probe_does_not_reset_gpu_triplets():
    operator = SoftAffineIPCOperator.__new__(SoftAffineIPCOperator)
    operator.cuda_hot_loop = False
    reset_calls = []
    operator.hash_triplet = SimpleNamespace(reset_system=lambda: reset_calls.append("reset"))
    operator._profile_stage = lambda _stage, tick: tick
    operator._assemble_affine_self = lambda _value, _need_matrix: (2.5, np.asarray([1.25]))
    operator.max_dof = 3
    operator.affine_dof = 1
    operator.total_dof = 3
    operator.global_grad = _ArrayField(np.zeros(3))
    operator.energy = _ScalarField()
    operator._assemble_soft_energy_gradient = lambda _need_matrix, _project_spd: None
    operator._build_mixed_pairs = lambda _swept: None
    operator._assemble_soft_soft_contact = lambda _need_matrix: None
    operator._assemble_mixed_contact = lambda _need_matrix: None
    operator.fully_implicit = True
    operator.sims = SimpleNamespace(
        affine_fully_implicit_jacobian_shift=0.0,
        affine_hessian_shift=0.0,
    )
    operator._add_hessian_shift = lambda _shift: None
    operator._raise_hash_triplet_overflow = lambda _stage: None

    energy, residual = SoftAffineIPCOperator.assemble(operator, np.zeros(1), need_matrix=False)
    assert reset_calls == []
    assert energy == pytest.approx(2.5)
    np.testing.assert_array_equal(residual, [1.25, 0.0, 0.0])


@pytest.mark.parametrize("plastic", (False, True))
@pytest.mark.parametrize("joint_num", (0, 1))
def test_soft_affine_adjoint_requests_the_physical_lagged_tangent(plastic, joint_num):
    operator = object.__new__(SoftAffineIPCOperator)
    operator.is_semi = False
    operator.fully_implicit = False
    operator.total_dof = 6
    operator.max_dof = 9
    operator.affine = SimpleNamespace(
        levelset_contact=False,
        wall_num=0,
        joint_num=joint_num,
        contact_damping_stiffness=0.0,
        friction_contact_count=_ScalarField(1),
    )
    operator.soft_material = SimpleNamespace(
        matProps=SimpleNamespace(
            is_finite_strain_plastic=plastic,
            model=(type("FiniteStrainVonMisesModel", (), {})() if plastic else None),
        )
    )
    operator.soft_friction_count = _ScalarField(1)
    operator.mixed_friction_count = _ScalarField(1)
    operator.adjoint_rhs = _ArrayField(np.zeros(9))
    operator.adjoint_solution = _ArrayField(np.zeros(9))
    operator.sims = SimpleNamespace(
        affine_linear_tolerance=1.0e-10,
        affine_linear_max_iteration=50,
    )
    calls = []
    operator.restore_lagged_friction_for_adjoint_device = lambda: calls.append("restore")
    operator.assemble_device = lambda **kwargs: calls.append(kwargs)

    def solve(rhs, solution, **kwargs):
        calls.append((kwargs, operator.hash_triplet.solver))
        solution.from_numpy(np.arange(9, dtype=np.float64))
        return {"converged": True, "residual": 0.0, "iterations": 2}

    operator.hash_triplet = SimpleNamespace(
        solver="PCG",
        matrix_symmetric=True,
        finalize_taichi_assembly=lambda: calls.append("finalize"),
        solve_flat_system=solve,
    )

    solve = operator.solve_plastic_equilibrium_adjoint if plastic else operator.solve_elastic_adjoint
    result = solve(np.ones(6))

    assert calls[0] == "restore"
    assert calls[1] == {
        "need_matrix": True,
        "project_spd": False,
        "solver_shift": False,
    }
    assert calls[2] == "finalize"
    solve_kwargs, solve_backend = calls[3]
    assert solve_kwargs["active_nodes"] == 2
    assert solve_kwargs["transpose"] is False
    assert solve_backend == "PCG"
    assert solve_kwargs["fallback_to_bicgstab"] is True
    assert operator.hash_triplet.solver == "PCG"
    np.testing.assert_array_equal(result.to_numpy(), np.arange(9))


def test_soft_affine_differentiation_rejects_fully_implicit_friction():
    operator = object.__new__(SoftAffineIPCOperator)
    operator.is_semi = False
    operator.fully_implicit = True
    operator.affine = SimpleNamespace(levelset_contact=False)
    with pytest.raises(ValueError, match="fully implicit friction"):
        operator.solve_elastic_adjoint(np.ones(3))


def test_soft_affine_elastic_gravity_vjp_combines_affine_and_soft_dofs():
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    operator = object.__new__(SoftAffineIPCOperator)
    operator.affine = SimpleNamespace(
        body_num=1,
        mass=ti.field(float, shape=(1, 4, 4)),
    )
    operator.affine.mass.from_numpy(
        np.asarray(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0],
                ]
            ],
            dtype=np.float64,
        )
    )
    operator.affine_dof = 12
    operator.soft_grid_num = 2
    operator.soft_node2dof = ti.field(ti.i32, shape=2)
    operator.soft_node2dof.from_numpy(np.asarray([1, 2], dtype=np.int32))
    grid_type = ti.types.struct(m=float)
    operator.scene = SimpleNamespace(soft_grid=grid_type.field(shape=2))
    operator.scene.soft_grid.m.from_numpy(np.asarray([5.0, 7.0]))
    operator.scale_device = ti.field(float, shape=())
    operator.scale_device[None] = 0.25
    operator.adjoint_solution = ti.field(float, shape=18)
    adjoint = np.arange(1.0, 19.0, dtype=np.float64)
    operator.adjoint_solution.from_numpy(adjoint)
    operator.gravity_vjp = ti.Vector.field(3, float, shape=())

    operator._differentiate_elastic_gravity()

    expected = np.zeros(3)
    for control, lumped in enumerate((1.0, 2.0, 3.0, 4.0)):
        expected += 0.25 * lumped * adjoint[3 * control : 3 * control + 3]
    expected += 0.25 * 5.0 * adjoint[12:15]
    expected += 0.25 * 7.0 * adjoint[15:18]
    np.testing.assert_allclose(np.asarray(operator.gravity_vjp[None]), expected)


def test_soft_affine_single_material_young_vjp_matches_neo_hookean_force():
    from src.mpm.MaterialManager import SoftParticleSingleMaterialAdapter
    from src.physics_model.consititutive_model.finite_strain.NeoHookean import (
        NeoHookeanModel,
    )

    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    young = 200.0
    model = NeoHookeanModel()
    model.add_material(1.0, young, 0.3)
    point_type = ti.types.struct(
        active=ti.i32,
        materialID=ti.i32,
        vol0=float,
        F=ti.types.matrix(3, 3, float),
    )
    scene = SimpleNamespace(
        soft_point=point_type.field(shape=1),
        soft_shape_count=ti.field(ti.i32, shape=1),
        soft_shape_node=ti.field(ti.i32, shape=(1, 1)),
        soft_dshape=ti.Vector.field(3, float, shape=(1, 1)),
        soft_support_shared=False,
    )
    deformation_gradient = np.diag([1.1, 0.95, 1.02])
    shape_gradient = np.asarray([0.7, -0.2, 0.1])
    scene.soft_point[0].active = 1
    scene.soft_point[0].materialID = 0
    scene.soft_point[0].vol0 = 0.4
    scene.soft_point[0].F = ti.Matrix(deformation_gradient)
    scene.soft_shape_count[0] = 1
    scene.soft_shape_node[0, 0] = 0
    scene.soft_dshape[0, 0] = ti.Vector(shape_gradient)

    operator = object.__new__(SoftAffineIPCOperator)
    operator.scene = scene
    operator.soft_material = SimpleNamespace(matProps=SoftParticleSingleMaterialAdapter(model))
    operator.soft_point_num = 1
    operator.soft_grid_num = 1
    operator.soft_node2dof = ti.field(ti.i32, shape=1)
    operator.soft_node2dof[0] = 1
    operator.soft_disp = ti.field(float, shape=3)
    operator.soft_disp.fill(0.0)
    operator.affine_dof = 0
    operator.scale_device = ti.field(float, shape=())
    operator.scale_device[None] = 0.25
    operator.adjoint_solution = ti.field(float, shape=3)
    adjoint = np.asarray([0.3, -0.2, 0.1])
    operator.adjoint_solution.from_numpy(adjoint)
    operator.soft_young_vjp = ti.field(float, shape=())

    operator._differentiate_soft_young(young)

    _, first_piola, _ = model.evaluate(deformation_gradient, need_tangent=False)
    residual_derivative = 0.4 * 0.25 * (first_piola @ shape_gradient) / young
    expected = -adjoint @ residual_derivative
    assert float(operator.soft_young_vjp[None]) == pytest.approx(expected, rel=2.0e-12, abs=1.0e-14)


def test_soft_affine_plastic_history_vjp_matches_material_force_difference():
    from src.mpm.MaterialManager import SoftParticleSingleMaterialAdapter
    from src.physics_model.consititutive_model.finite_strain.VonMises import (
        FiniteStrainVonMisesModel,
    )

    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    model = FiniteStrainVonMisesModel().initialize_from_kwargs(
        density=7800.0,
        young_modulus=2.0e5,
        poisson_ratio=0.3,
        YieldStress=1200.0,
        HardeningModulus=5000.0,
    )
    model.allocate_state(1)
    plastic_inverse = np.asarray(
        [[1.03, 0.02, 0.0], [-0.01, 0.97, 0.0], [0.0, 0.0, 1.01]],
        dtype=np.float64,
    )
    model.plastic_deformation_inverse.from_numpy(plastic_inverse[None])
    model.equivalent_plastic_strain[0] = 0.15

    point_type = ti.types.struct(
        active=ti.i32,
        materialID=ti.i32,
        vol0=float,
        F=ti.types.matrix(3, 3, float),
    )
    scene = SimpleNamespace(
        soft_point=point_type.field(shape=1),
        soft_shape_count=ti.field(ti.i32, shape=1),
        soft_shape_node=ti.field(ti.i32, shape=(1, 1)),
        soft_dshape=ti.Vector.field(3, float, shape=(1, 1)),
        soft_support_shared=False,
    )
    total_deformation = np.asarray(
        [[1.12, 0.04, 0.0], [0.01, 0.94, 0.03], [0.0, -0.01, 0.96]],
        dtype=np.float64,
    )
    shape_gradient = np.asarray([0.7, -0.2, 0.1], dtype=np.float64)
    adjoint = np.asarray([0.3, -0.2, 0.1], dtype=np.float64)
    scene.soft_point[0].active = 1
    scene.soft_point[0].materialID = 0
    scene.soft_point[0].vol0 = 0.4
    scene.soft_point[0].F = ti.Matrix(total_deformation)
    scene.soft_shape_count[0] = 1
    scene.soft_shape_node[0, 0] = 0
    scene.soft_dshape[0, 0] = ti.Vector(shape_gradient)

    operator = object.__new__(SoftAffineIPCOperator)
    operator.scene = scene
    operator.soft_material = SimpleNamespace(matProps=SoftParticleSingleMaterialAdapter(model))
    operator.soft_point_num = 1
    operator.soft_grid_num = 1
    operator.soft_node2dof = ti.field(ti.i32, shape=1)
    operator.soft_node2dof[0] = 1
    operator.soft_disp = ti.field(float, shape=3)
    operator.soft_disp.fill(0.0)
    operator.affine_dof = 0
    operator.scale_device = ti.field(float, shape=())
    operator.scale_device[None] = 0.25
    operator.adjoint_solution = ti.field(float, shape=3)
    operator.adjoint_solution.from_numpy(adjoint)
    operator.plastic_inverse_vjp = ti.Matrix.field(3, 3, float, shape=1)
    operator.plastic_equivalent_strain_vjp = ti.field(float, shape=1)
    operator.plastic_volumetric_strain_vjp = ti.field(float, shape=1)
    operator.plastic_deformation_vjp = ti.Matrix.field(3, 3, float, shape=1)

    objective = ti.field(float, shape=())

    @ti.kernel
    def evaluate_objective():
        stress = model.total_first_piola_stress_at(0, scene.soft_point[0].F)
        force = 0.4 * 0.25 * stress @ scene.soft_dshape[0, 0]
        objective[None] = -ti.Vector([0.3, -0.2, 0.1]).dot(force)

    def evaluate():
        evaluate_objective()
        return float(objective[None])

    operator._differentiate_soft_plastic_input_history()
    history_vjp = operator.plastic_inverse_vjp.to_numpy()[0]
    hardening_vjp = float(operator.plastic_equivalent_strain_vjp[0])
    deformation_vjp = operator.plastic_deformation_vjp.to_numpy()[0]

    step = 2.0e-7
    plus = plastic_inverse.copy()
    minus = plastic_inverse.copy()
    plus[0, 1] += step
    minus[0, 1] -= step
    model.plastic_deformation_inverse.from_numpy(plus[None])
    plus_value = evaluate()
    model.plastic_deformation_inverse.from_numpy(minus[None])
    minus_value = evaluate()
    assert history_vjp[0, 1] == pytest.approx((plus_value - minus_value) / (2.0 * step), rel=4.0e-4, abs=2.0e-6)

    model.plastic_deformation_inverse.from_numpy(plastic_inverse[None])
    plus = total_deformation.copy()
    minus = total_deformation.copy()
    plus[0, 1] += step
    minus[0, 1] -= step
    scene.soft_point[0].F = ti.Matrix(plus)
    plus_value = evaluate()
    scene.soft_point[0].F = ti.Matrix(minus)
    minus_value = evaluate()
    scene.soft_point[0].F = ti.Matrix(total_deformation)
    assert deformation_vjp[0, 1] == pytest.approx((plus_value - minus_value) / (2.0 * step), rel=4.0e-4, abs=2.0e-6)

    model.plastic_deformation_inverse.from_numpy(plastic_inverse[None])
    model.equivalent_plastic_strain[0] = 0.15 + step
    plus_value = evaluate()
    model.equivalent_plastic_strain[0] = 0.15 - step
    minus_value = evaluate()
    assert hardening_vjp == pytest.approx((plus_value - minus_value) / (2.0 * step), rel=4.0e-4, abs=2.0e-6)
    assert float(operator.plastic_volumetric_strain_vjp[0]) == 0.0

    model.plastic_deformation_inverse.from_numpy(plastic_inverse[None])
    model.equivalent_plastic_strain[0] = 0.15
    operator.plastic_output_deformation_vjp = ti.Matrix.field(3, 3, float, shape=1)
    operator.plastic_output_inverse_vjp = ti.Matrix.field(3, 3, float, shape=1)
    operator.plastic_output_equivalent_vjp = ti.field(float, shape=1)
    operator.plastic_output_volumetric_vjp = ti.field(float, shape=1)
    operator.plastic_commit_deformation_vjp = ti.Matrix.field(3, 3, float, shape=1)
    operator.plastic_commit_inverse_vjp = ti.Matrix.field(3, 3, float, shape=1)
    operator.plastic_commit_equivalent_vjp = ti.field(float, shape=1)
    operator.plastic_commit_volumetric_vjp = ti.field(float, shape=1)
    operator.plastic_commit_grid_vjp = ti.field(float, shape=3)
    terminal_deformation_seed = np.asarray(
        [[0.12, -0.03, 0.05], [0.02, 0.08, -0.04], [-0.01, 0.06, 0.09]],
        dtype=np.float64,
    )
    operator.plastic_output_deformation_vjp.from_numpy(terminal_deformation_seed[None])
    operator._differentiate_soft_plastic_commit_state()
    np.testing.assert_allclose(
        operator.plastic_commit_deformation_vjp.to_numpy()[0],
        terminal_deformation_seed,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        operator.plastic_commit_grid_vjp.to_numpy(),
        terminal_deformation_seed @ shape_gradient,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_soft_affine_elastic_parameter_vjp_reuses_affine_rigidity_kernel():
    events = []
    operator = object.__new__(SoftAffineIPCOperator)
    adjoint = _ArrayField(np.arange(6.0))
    operator.solve_elastic_adjoint = lambda seed: (events.append(("solve", np.asarray(seed).copy())) or adjoint)
    operator._single_soft_young_modulus = lambda: 8.0
    operator._differentiate_elastic_gravity = lambda: events.append(("gravity", None))
    operator._copy_affine_adjoint = lambda: events.append(("copy", None))
    operator._differentiate_soft_young = lambda young: events.append(("soft_young", young))
    operator.gravity_vjp = _ScalarField(np.asarray([1.0, 2.0, 3.0]))
    operator.soft_young_vjp = _ScalarField(6.0)
    operator.affine = SimpleNamespace(
        body_num=2,
        young_vjp=_ArrayField(np.asarray([4.0, 5.0, 99.0])),
        _differentiate_young_parameter=lambda: events.append(("young", None)),
    )

    result = operator.differentiate_elastic_parameters(np.ones(6))

    assert [event[0] for event in events] == [
        "solve",
        "gravity",
        "copy",
        "young",
        "soft_young",
    ]
    np.testing.assert_array_equal(result["gravity"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result["affine_young_modulus"], [4.0, 5.0])
    assert result["soft_young_modulus"] == pytest.approx(6.0)
    assert result["adjoint"] is adjoint


def test_soft_affine_plastic_parameter_vjp_reuses_coupled_adjoint():
    events = []
    operator = object.__new__(SoftAffineIPCOperator)
    adjoint = _ArrayField(np.arange(6.0))
    operator.solve_plastic_equilibrium_adjoint = lambda seed: (
        events.append(("solve", np.asarray(seed).copy())) or adjoint
    )
    operator._differentiate_elastic_gravity = lambda: events.append("gravity")
    operator._copy_affine_adjoint = lambda: events.append("copy")
    operator._differentiate_soft_plastic_input_history = lambda: events.append("plastic")
    operator.gravity_vjp = _ScalarField(np.asarray([1.0, 2.0, 3.0]))
    operator.soft_point_num = 1
    operator.plastic_inverse_vjp = _ArrayField(np.ones((1, 3, 3)))
    operator.plastic_equivalent_strain_vjp = _ArrayField(np.asarray([4.0]))
    operator.plastic_volumetric_strain_vjp = _ArrayField(np.asarray([0.0]))
    operator.plastic_deformation_vjp = _ArrayField(np.zeros((1, 3, 3)))
    operator.affine = SimpleNamespace(
        body_num=1,
        young_vjp=_ArrayField(np.asarray([5.0])),
        _differentiate_young_parameter=lambda: events.append("young"),
    )

    result = operator.differentiate_plastic_equilibrium_parameters(np.ones(6))

    assert [event[0] if isinstance(event, tuple) else event for event in events] == [
        "solve",
        "gravity",
        "copy",
        "young",
        "plastic",
    ]
    assert result["plastic_history"] == "input_vjp"
    np.testing.assert_array_equal(result["equivalent_plastic_strain"], [4.0])
    np.testing.assert_array_equal(result["affine_young_modulus"], [5.0])
    assert result["adjoint"] is adjoint


def test_soft_affine_step_runs_elastic_adjoint_before_commit():
    events = []
    engine = SoftAffineIPCEngine()
    engine.pending_elastic_adjoint_seed = np.asarray([1.0, 2.0, 3.0])
    expected = {"gravity": np.asarray([4.0, 5.0, 6.0])}
    engine.operator = SimpleNamespace(
        differentiate_elastic_parameters=lambda seed: (events.append(("differentiate", seed.copy())) or expected),
        accept_step_device=lambda: events.append(("commit", None)),
    )

    engine._accept_lagged_solution_device()

    assert events[0][0] == "differentiate"
    np.testing.assert_array_equal(events[0][1], [1.0, 2.0, 3.0])
    assert events[1] == ("commit", None)
    assert engine.last_elastic_differentiation is expected


def test_soft_affine_failed_precommit_adjoint_rolls_back_without_commit():
    events = []
    engine = SoftAffineIPCEngine()
    engine.pending_elastic_adjoint_seed = np.ones(3)

    def fail(_seed):
        events.append("differentiate")
        raise RuntimeError("adjoint failed")

    engine.operator = SimpleNamespace(
        differentiate_elastic_parameters=fail,
        rollback_step_device=lambda: events.append("rollback"),
        accept_step_device=lambda: pytest.fail("failed adjoint was committed"),
    )

    with pytest.raises(RuntimeError, match="adjoint failed"):
        engine._accept_lagged_solution_device()

    assert events == ["differentiate", "rollback"]


def test_soft_affine_fully_implicit_pending_adjoint_failure_rolls_back():
    events = []
    engine = SoftAffineIPCEngine()
    engine.pending_elastic_adjoint_seed = np.asarray([1.0, 2.0, 3.0])

    def reject(_seed):
        events.append("differentiate")
        raise ValueError("fully implicit friction is not implemented")

    engine.operator = SimpleNamespace(
        _assert_cuda_device_residency=lambda: events.append("residency"),
        begin_step_device=lambda: events.append("begin"),
        initialize_fully_implicit_velocity_predictor_device=lambda _sims: events.append("predict"),
        assemble_device=lambda need_matrix=True: 1.0,
        residual_inf_norm_device=lambda: 0.0,
        differentiate_elastic_parameters=reject,
        accept_step_device=lambda: pytest.fail("rejected adjoint was committed"),
        rollback_step_device=lambda: events.append("rollback"),
    )
    engine._solve_fully_implicit_newton_device = lambda _sims: 2
    engine.last_inner_converged = True

    with pytest.raises(ValueError, match="fully implicit friction"):
        engine._step_fully_implicit_device(SimpleNamespace())

    assert events[-2:] == ["differentiate", "rollback"]


def test_differentiable_soft_affine_rejects_fully_implicit_friction():
    from src.mpdem.engines.DifferentiableSoftAffine import (
        DifferentiableSoftAffine,
    )

    engine = SoftAffineIPCEngine()
    engine.operator = SimpleNamespace(
        soft_material=SimpleNamespace(matProps=SimpleNamespace(model=type("FiniteStrainVonMisesModel", (), {})())),
        affine=SimpleNamespace(levelset_contact=False),
        is_semi=False,
        fully_implicit=True,
    )

    with pytest.raises(ValueError, match="fully implicit friction"):
        DifferentiableSoftAffine(engine, SimpleNamespace(), None, 2)


def test_soft_affine_precommit_dispatches_plastic_equilibrium_adjoint():
    engine = SoftAffineIPCEngine()
    engine.pending_elastic_adjoint_seed = np.ones(3)
    engine.pending_adjoint_mode = "plastic"
    expected = {"plastic_history": "input_vjp"}
    engine.operator = SimpleNamespace(differentiate_plastic_equilibrium_parameters=lambda _seed: expected)

    engine._differentiate_before_commit()

    assert engine.last_elastic_differentiation is expected


def test_soft_affine_precommit_dispatches_accepted_plastic_state_vjp():
    engine = SoftAffineIPCEngine()
    engine.pending_elastic_adjoint_seed = np.ones(3)
    engine.pending_adjoint_mode = "plastic_state"
    engine.pending_plastic_state_vjp = {"equivalent_plastic_strain": [1.0]}
    expected = {"plastic_history": "accepted_step_input_vjp"}
    engine.operator = SimpleNamespace(
        differentiate_plastic_step_parameters=lambda seed, state: (
            expected if np.array_equal(seed, np.ones(3)) and state is engine.pending_plastic_state_vjp else None
        )
    )

    engine._differentiate_before_commit()

    assert engine.last_elastic_differentiation is expected


def test_soft_affine_precommit_dispatches_device_trajectory_pullback():
    engine = SoftAffineIPCEngine()
    engine.pending_elastic_adjoint_seed = object()
    engine.pending_adjoint_mode = "trajectory"
    calls = []
    engine.operator = SimpleNamespace(pullback_coupled_step_device=lambda: calls.append("pullback"))

    engine._differentiate_before_commit()

    assert calls == ["pullback"]
    assert engine.last_elastic_differentiation is True


def _toolkit_barrier_distance2_oracle(distance2, active_distance2, kappa):
    distance2 = float(distance2)
    active_distance2 = float(active_distance2)
    if distance2 >= active_distance2:
        return 0.0, 0.0, 0.0
    diff = distance2 - active_distance2
    log_term = np.log(distance2 / active_distance2)
    energy = -kappa * diff * diff * log_term
    gradient = -kappa * (2.0 * diff * log_term + diff * diff / distance2)
    hessian = -kappa * (2.0 * log_term + 4.0 * diff / distance2 - diff * diff / (distance2 * distance2))
    return energy, gradient, hessian


@ti.data_oriented
class _MaterialTable:
    def __init__(self):
        self.levelset_contact = False
        self.pp_dhat = ti.field(float, shape=(1, 1))
        self.pp_kappa = ti.field(float, shape=(1, 1))
        self.pp_mu = ti.field(float, shape=(1, 1))
        self.friction_scale = ti.field(float, shape=1)
        self.friction_scale_vjp = ti.field(float, shape=())
        self.epsv = 1.0e-3
        self.pp_dhat[0, 0] = 0.1
        self.pp_kappa[0, 0] = 1.0e4
        self.pp_mu[0, 0] = 0.4
        self.friction_scale[0] = 1.0
        self.edge_type_count = ti.field(ti.i32, shape=9)
        self.edge_mollifier_type_count = ti.field(ti.i32, shape=9)
        self.body_material = ti.field(ti.i32, shape=1)
        self.body_material[0] = 0
        self.face_num = 1
        self.face2body = ti.field(ti.i32, shape=1)
        self.node2body = ti.field(ti.i32, shape=3)
        self.faces = ti.Vector.field(3, ti.i32, shape=1)
        self.x = ti.Vector.field(3, float, shape=3)
        self.hat_x = ti.Vector.field(3, float, shape=3)
        self.basis = ti.field(float, shape=(3, 4))
        self.face2body[0] = 0
        self.node2body.fill(0)
        self.faces[0] = ti.Vector([0, 1, 2])
        triangle = np.array(
            [[-1.0, -1.0, -0.05], [1.0, -1.0, -0.05], [0.0, 1.0, -0.05]],
            dtype=np.float64,
        )
        self.x.from_numpy(triangle)
        self.hat_x.from_numpy(triangle)
        basis = np.zeros((3, 4), dtype=np.float64)
        basis[0, 0] = 1.0
        basis[1, 1] = 1.0
        basis[2, 2] = 1.0
        self.basis.from_numpy(basis)

    def set_timestep(self, _dt):
        pass

    @ti.func
    def _material_id(self, material):
        return material

    @ti.func
    def _ipc_barrier_distance2(self, distance2, active_distance2, kappa):
        return ipc_toolkit_barrier_distance2_terms(distance2, active_distance2, kappa)

    @ti.func
    def _ipc_barrier_gap(self, gap, active_gap, kappa):
        return ipc_toolkit_barrier_distance_terms(gap, active_gap, kappa)


@ti.data_oriented
class _SymmetricScalarSink:
    def __init__(self, size):
        self.values = ti.field(float, shape=(size, size))
        self.symmetric = True

    def reset_system(self):
        self.values.fill(0.0)

    @ti.func
    def add_scalar_entry(self, row, column, value, _row_component, _column_component):
        ti.atomic_add(self.values[row, column], value)

    @ti.func
    def add_block_entry(self, block_i, block_j, block):
        for i, j in ti.static(ti.ndrange(3, 3)):
            row = 3 * block_i + i
            column = 3 * block_j + j
            if ti.static(not self.symmetric) or row <= column:
                ti.atomic_add(self.values[row, column], block[i, j])


def _make_bilateral_kernel_harness():
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    soft_body_type = ti.types.struct(
        surfacePointStart=ti.i32,
        surfacePointEnd=ti.i32,
    )
    soft_point_type = ti.types.struct(
        x=ti.types.vector(3, float),
        active=ti.i32,
        materialID=ti.i32,
        bodyID=ti.i32,
        surface_weight=float,
        vol0=float,
    )
    rigid_type = ti.types.struct(softID=ti.i32)
    scene = SimpleNamespace(
        soft=soft_body_type.field(shape=2),
        soft_point=soft_point_type.field(shape=2),
        rigid=rigid_type.field(shape=2),
        soft_surface_point_id=ti.field(ti.i32, shape=2),
        soft_shape_count=ti.field(ti.i32, shape=2),
        soft_shape_node=ti.field(ti.i32, shape=(2, 1)),
        soft_shape=ti.field(float, shape=(2, 1)),
        soft_support_shared=False,
    )
    scene.soft[0].surfacePointStart = 0
    scene.soft[0].surfacePointEnd = 1
    scene.soft[1].surfacePointStart = 1
    scene.soft[1].surfacePointEnd = 2
    scene.soft_point[0].x = ti.Vector([0.0, 0.0, 0.0])
    scene.soft_point[1].x = ti.Vector([0.05, 0.0, 0.0])
    for point in range(2):
        scene.soft_point[point].active = 1
        scene.soft_point[point].materialID = 0
        scene.soft_point[point].bodyID = point
        scene.rigid[point].softID = point
        scene.soft_point[point].surface_weight = 1.0
        scene.soft_point[point].vol0 = 1.0
        scene.soft_surface_point_id[point] = point
        scene.soft_shape_count[point] = 1
        scene.soft_shape_node[point, 0] = point
        scene.soft_shape[point, 0] = 1.0

    operator = object.__new__(SoftAffineIPCOperator)
    operator.friction_mode = "lagged"
    operator.fully_implicit = False
    operator.cuda_hot_loop = False
    operator.stable_lagged_contacts = False
    operator.fully_mu_dynamic = -1.0
    operator.fully_mu_static = -1.0
    operator.fully_mu_viscous = 0.0
    operator.fully_stribeck_velocity = 1.0e-2
    operator.fully_profile_id = 0
    operator.scene = scene
    operator.soft_num = 2
    operator.soft_surface_point_num = 2
    operator.affine = _MaterialTable()
    operator.dt = 1.0e-2
    operator.scale = operator.dt * operator.dt
    operator.dt_device = ti.field(float, shape=())
    operator.scale_device = ti.field(float, shape=())
    operator.dt_device[None] = operator.dt
    operator.scale_device[None] = operator.scale
    operator.affine_dof = 12
    operator.total_dof = 18
    operator.soft_node2dof = ti.field(ti.i32, shape=2)
    operator.soft_node2dof.from_numpy(np.array([1, 2], dtype=np.int32))
    operator.soft_disp = ti.field(float, shape=6)
    operator.soft_hat_x = ti.Vector.field(3, float, shape=2)
    operator.soft_hat_x.from_numpy(np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64))
    operator.mixed_levelset_frozen_point = ti.Vector.field(3, float, shape=2)
    operator.global_grad = ti.field(float, shape=18)
    operator.adjoint_solution = ti.field(float, shape=18)
    operator.energy = ti.field(float, shape=())
    operator.hash_triplet = _SymmetricScalarSink(18)
    operator.soft_pair_num = ti.field(ti.i32, shape=())
    operator.soft_pair = ti.Vector.field(2, ti.i32, shape=1)
    operator.soft_pair_num[None] = 1
    operator.soft_pair[0] = ti.Vector([0, 1])
    operator.soft_pair_start = ti.field(ti.i32, shape=2)
    operator.soft_pair_end = ti.field(ti.i32, shape=2)
    operator.soft_pair_start.from_numpy(np.array([0, 1], dtype=np.int32))
    operator.soft_pair_end.from_numpy(np.array([1, 1], dtype=np.int32))
    operator.soft_friction_capacity = 2
    operator.soft_friction_count = ti.field(ti.i32, shape=())
    operator.soft_friction_overflow = ti.field(ti.i32, shape=())
    operator.soft_friction_points = ti.Vector.field(2, ti.i32, shape=2)
    operator.soft_friction_normal = ti.Vector.field(3, float, shape=2)
    operator.soft_friction_coeff = ti.field(float, shape=2)
    operator.mixed_friction_count = ti.field(ti.i32, shape=())
    operator.mixed_friction_overflow = ti.field(ti.i32, shape=())
    operator.mixed_friction_capacity = 2
    operator.mixed_friction_point = ti.field(ti.i32, shape=2)
    operator.mixed_friction_face = ti.field(ti.i32, shape=2)
    operator.mixed_friction_bary = ti.Vector.field(3, float, shape=2)
    operator.mixed_friction_normal = ti.Vector.field(3, float, shape=2)
    operator.mixed_friction_coeff = ti.field(float, shape=2)
    operator.mixed_pair_num = ti.field(ti.i32, shape=())
    operator.mixed_pair = ti.Vector.field(2, ti.i32, shape=1)
    operator.mixed_pair_num[None] = 1
    operator.mixed_pair[0] = ti.Vector([0, 0])
    operator.mixed_pair_start = ti.field(ti.i32, shape=2)
    operator.mixed_pair_end = ti.field(ti.i32, shape=2)
    operator.mixed_pair_start.from_numpy(np.array([0, 1], dtype=np.int32))
    operator.mixed_pair_end.from_numpy(np.array([1, 1], dtype=np.int32))
    operator.mixed_contact_type_count = ti.field(ti.i32, shape=7)
    return operator


def _make_levelset_mixed_kernel_harness():
    operator = _make_bilateral_kernel_harness()
    radius = 0.3
    spacing = 0.05
    shape = np.array([17, 17, 17], dtype=np.int32)
    origin = np.full(3, -0.4, dtype=np.float64)
    axes = [origin[d] + spacing * np.arange(shape[d], dtype=np.float64) for d in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    levelset = TrilinearLevelSet(
        origin,
        spacing,
        shape,
        (np.sqrt(x * x + y * y + z * z) - radius).flatten(order="F"),
    )
    reference_points = radius * np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int32,
    )
    controls = np.array(
        [
            [-0.25, -0.25, -0.25],
            [0.75, -0.25, -0.25],
            [-0.25, 0.75, -0.25],
            [-0.25, -0.25, 0.75],
        ],
        dtype=np.float64,
    )
    body = {
        "contact_representation": "LevelSet",
        "levelset": levelset,
        "scale": 1.0,
        "faces": faces,
        "basis": np.asarray([affine_basis(point) for point in reference_points]),
        "volume": 4.0 * np.pi * radius**3 / 3.0,
        "mass_matrix": np.eye(4, dtype=np.float64),
        "young": 0.0,
        "mu": 0.0,
        "materialID": 0,
        "groupID": 0,
        "y": controls,
        "v_y": np.zeros((4, 3), dtype=np.float64),
    }
    affine_state = AffineBodyState([body], np.zeros(3, dtype=np.float64))
    affine_sims = SimpleNamespace(
        max_material_num=1,
        dt=_ScalarField(1.0e-2),
        affine_dhat=0.1,
        affine_barrier_stiffness=1.0e4,
        affine_contact_damping_stiffness=0.0,
        affine_friction_epsv=1.0e-3,
        affine_hessian_shift=0.0,
        wall_coordination_number=0,
        search="LinkedCell",
        domain=np.array([2.0, 2.0, 2.0]),
    )
    affine_scene = SimpleNamespace(
        wall=None,
        affine_contact_properties={},
    )
    operator.affine_state = affine_state
    operator.affine = TaichiAffineBodyOperator(affine_state, affine_sims, affine_scene)
    operator.mixed_levelset_inverse = ti.field(float, shape=(1, 3, 3))
    operator.mixed_levelset_inverse_valid = ti.field(ti.i32, shape=1)
    operator.mixed_levelset_contact_count = ti.field(ti.i32, shape=())
    operator.mixed_levelset_contact_active = ti.field(ti.i32, shape=2)
    operator.mixed_levelset_contact_material = ti.Vector.field(3, float, shape=2)
    operator.mixed_levelset_contact_phi = ti.field(float, shape=2)
    operator.mixed_levelset_contact_phi_gradient = ti.Vector.field(3, float, shape=2)
    operator.mixed_levelset_contact_world_normal = ti.Vector.field(3, float, shape=2)
    operator.mixed_levelset_contact_barrier = ti.Vector.field(3, float, shape=2)
    operator.mixed_levelset_contact_coefficient = ti.field(float, shape=2)
    operator.mixed_levelset_cell_base = ti.field(ti.i32, shape=(2, 3))
    operator.mixed_levelset_cell_fraction = ti.field(float, shape=(2, 3))
    operator.mixed_contact_capacity = 2
    operator.mixed_levelset_friction_inverse = ti.field(float, shape=(1, 3, 3))
    operator.mixed_levelset_friction_inverse_valid = ti.field(ti.i32, shape=1)
    operator.mixed_levelset_current_point = ti.Vector.field(3, float, shape=2)
    operator.mixed_levelset_friction_active = ti.field(ti.i32, shape=2)
    operator.mixed_levelset_friction_material = ti.Vector.field(3, float, shape=2)
    operator.mixed_levelset_friction_phi = ti.field(float, shape=2)
    operator.mixed_levelset_friction_phi_gradient = ti.Vector.field(3, float, shape=2)
    operator.mixed_levelset_friction_unit_normal = ti.Vector.field(3, float, shape=2)
    operator.mixed_levelset_friction_coefficient = ti.field(float, shape=2)
    operator.soft_point_num = 2
    operator.affine.y.from_numpy(controls)
    operator.soft_direction = ti.field(float, shape=6)
    operator.ccd_alpha = ti.field(float, shape=())
    operator.scene.soft_point[0].x = ti.Vector([0.316, 0.011, 0.007])
    operator.scene.soft_point[0].surface_weight = 0.7
    operator.mixed_pair_num[None] = 1
    operator.mixed_pair[0] = ti.Vector([0, 0])
    return operator, levelset, controls


def _enable_stable_contact_compaction(operator):
    operator.soft_pair_capacity = int(operator.soft_pair.shape[0])
    operator.mixed_pair_capacity = int(operator.mixed_pair.shape[0])
    operator.soft_contact_prefix_sum = PrefixSumExecutor(max(operator.soft_surface_point_num, 1))
    operator.mixed_contact_prefix_sum = PrefixSumExecutor(max(operator.soft_surface_point_num, 1))
    operator.soft_contact_prefix = ti.field(
        ti.i32,
        shape=max(
            operator.soft_surface_point_num,
            operator.soft_contact_prefix_sum.get_length(),
            1,
        ),
    )
    operator.mixed_contact_prefix = ti.field(
        ti.i32,
        shape=max(
            operator.soft_surface_point_num,
            operator.mixed_contact_prefix_sum.get_length(),
            1,
        ),
    )
    operator.stable_lagged_contacts = True


def _make_swept_ccd_harness():
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    triangle = np.array(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    basis = np.zeros((3, 4), dtype=np.float64)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    basis[2, 2] = 1.0
    controls = np.zeros((4, 3), dtype=np.float64)
    controls[:3] = triangle
    body = {
        "faces": np.array([[0, 1, 2]], dtype=np.int32),
        "basis": basis,
        "volume": 1.0,
        "mass_matrix": np.eye(4, dtype=np.float64),
        "young": 0.0,
        "materialID": 0,
        "groupID": 0,
        "y": controls,
        "v_y": np.zeros((4, 3), dtype=np.float64),
    }
    affine_state = AffineBodyState([body], np.zeros(3, dtype=np.float64))
    affine_sims = SimpleNamespace(
        max_material_num=1,
        dt=_ScalarField(1.0e-2),
        affine_dhat=0.1,
        affine_barrier_stiffness=1.0e4,
        affine_contact_damping_stiffness=0.0,
        affine_friction_epsv=1.0e-3,
        affine_hessian_shift=0.0,
        wall_coordination_number=0,
        search="LinkedCell",
        domain=np.array([10.0, 10.0, 10.0]),
    )
    affine_scene = SimpleNamespace(
        wall=None,
        affine_contact_properties={},
    )
    affine = TaichiAffineBodyOperator(affine_state, affine_sims, affine_scene)
    affine.y.from_numpy(controls)
    affine.direction_y.fill(0.0)
    affine._reconstruct_vertices()
    affine._reconstruct_vertex_directions()

    soft_body_type = ti.types.struct(
        startIndex=ti.i32,
        endIndex=ti.i32,
        surfacePointStart=ti.i32,
        surfacePointEnd=ti.i32,
    )
    soft_point_type = ti.types.struct(
        x=ti.types.vector(3, float),
        active=ti.i32,
        bodyID=ti.i32,
        m=float,
        materialID=ti.i32,
        surface_weight=float,
        vol0=float,
    )
    rigid_type = ti.types.struct(softID=ti.i32)
    scene = SimpleNamespace(
        soft=soft_body_type.field(shape=2),
        soft_point=soft_point_type.field(shape=2),
        rigid=rigid_type.field(shape=2),
        soft_surface_point_id=ti.field(ti.i32, shape=2),
        soft_shape_count=ti.field(ti.i32, shape=2),
        soft_shape_node=ti.field(ti.i32, shape=(2, 1)),
        soft_shape=ti.field(float, shape=(2, 1)),
        soft_support_shared=False,
    )
    for body_id in range(2):
        scene.soft[body_id].startIndex = body_id
        scene.soft[body_id].endIndex = body_id + 1
        scene.soft[body_id].surfacePointStart = body_id
        scene.soft[body_id].surfacePointEnd = body_id + 1
        scene.soft_point[body_id].active = 1
        scene.soft_point[body_id].bodyID = body_id
        scene.rigid[body_id].softID = body_id
        scene.soft_point[body_id].m = 1.0
        scene.soft_point[body_id].materialID = 0
        scene.soft_point[body_id].surface_weight = 1.0
        scene.soft_point[body_id].vol0 = 1.0e-6
        scene.soft_surface_point_id[body_id] = body_id
        scene.soft_shape_count[body_id] = 1
        scene.soft_shape_node[body_id, 0] = body_id
        scene.soft_shape[body_id, 0] = 1.0

    operator = object.__new__(SoftAffineIPCOperator)
    operator.friction_mode = "lagged"
    operator.fully_implicit = False
    operator.scene = scene
    operator.affine_state = affine_state
    operator.affine = affine
    operator.soft_num = 2
    operator.soft_point_num = 2
    operator.soft_surface_point_num = 2
    operator.soft_node2dof = ti.field(ti.i32, shape=2)
    operator.soft_node2dof.from_numpy(np.array([1, 2], dtype=np.int32))
    operator.soft_disp = ti.field(float, shape=6)
    operator.soft_direction = ti.field(float, shape=6)
    operator.soft_min = ti.Vector.field(3, float, shape=2)
    operator.soft_max = ti.Vector.field(3, float, shape=2)
    operator.soft_center_disp = ti.Vector.field(3, float, shape=2)
    operator.soft_center_direction = ti.Vector.field(3, float, shape=2)
    operator.soft_body_mass = ti.field(float, shape=2)
    operator.affine_min = ti.Vector.field(3, float, shape=1)
    operator.affine_max = ti.Vector.field(3, float, shape=1)
    operator.soft_pair_num = ti.field(ti.i32, shape=())
    operator.soft_pair = ti.Vector.field(2, ti.i32, shape=1)
    operator.soft_pair_start = ti.field(ti.i32, shape=2)
    operator.soft_pair_end = ti.field(ti.i32, shape=2)
    operator.mixed_pair_num = ti.field(ti.i32, shape=())
    operator.mixed_pair = ti.Vector.field(2, ti.i32, shape=2)
    operator.mixed_pair_start = ti.field(ti.i32, shape=2)
    operator.mixed_pair_end = ti.field(ti.i32, shape=2)
    operator.ccd_alpha = ti.field(float, shape=())
    from src.fem.contact.BVHBroadPhase import DynamicBVHBroadPhase

    operator.mixed_bvh_node_count = 5
    operator.mixed_bvh_position = ti.Vector.field(3, float, shape=5)
    operator.mixed_bvh_end_position = ti.Vector.field(3, float, shape=5)
    operator.mixed_candidate_capacity = 2
    operator.mixed_bvh = DynamicBVHBroadPhase(
        np.array([[2, 3, 4]], dtype=np.int32),
        np.empty((0, 2), dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        np.zeros(5),
        np.zeros(0),
        np.vstack((np.zeros((2, 3)), triangle)),
        max_point_triangle_pairs=2,
        max_edge_edge_pairs=1,
    )
    return operator


def _set_soft_motion(operator, positions, directions, active=(1, 1)):
    positions = np.asarray(positions, dtype=np.float64).reshape((2, 3))
    directions = np.asarray(directions, dtype=np.float64).reshape((2, 3))
    for point in range(2):
        operator.scene.soft_point[point].x = ti.Vector(positions[point])
        operator.scene.soft_point[point].active = int(active[point])
    operator.soft_disp.fill(0.0)
    operator.soft_direction.from_numpy(directions.reshape(-1))


def _enable_fully_implicit_friction(operator):
    operator.friction_mode = "fully_implicit"
    operator.fully_implicit = True
    operator.fully_mu_dynamic = 0.31
    operator.fully_mu_static = 0.67
    operator.fully_mu_viscous = 0.015
    operator.fully_stribeck_velocity = 1.25
    operator.fully_profile_id = 0
    operator.hash_triplet.symmetric = False


def _clear_coupled_contact_system(operator):
    operator.global_grad.fill(0.0)
    operator.energy[None] = 0.0
    operator.hash_triplet.values.fill(0.0)


def _enable_semi_ipc(operator, capacity=16):
    operator.is_semi = True
    operator.affine.is_semi = True
    operator.semi_capacity = capacity
    operator.semi_state = ti.field(ti.i32, shape=capacity)
    operator.semi_key = ti.Vector.field(4, ti.i32, shape=capacity)
    operator.semi_multiplier = ti.field(float, shape=capacity)
    operator.semi_count = ti.field(ti.i32, shape=())
    operator.semi_overflow = ti.field(ti.i32, shape=())
    operator.semi_constraint_violation = ti.field(float, shape=())
    operator.semi_state.fill(2)


def _prepare_production_assembly_harness(operator, friction_mode):
    """Keep only non-contact subsystems synthetic around production contact."""
    if friction_mode == "fully_implicit":
        _enable_fully_implicit_friction(operator)
    else:
        operator.friction_mode = "lagged"
        operator.fully_implicit = False

    operator.max_dof = operator.total_dof
    operator.max_soft_dof = operator.total_dof - operator.affine_dof
    operator.soft_direction = ti.field(float, shape=max(operator.max_soft_dof, 1))
    operator.soft_disp_base = ti.field(float, shape=max(operator.max_soft_dof, 1))
    operator.cuda_hot_loop = False
    operator.sims = SimpleNamespace(
        affine_hessian_shift=0.0,
        affine_fully_implicit_jacobian_shift=0.0,
    )
    operator._reject_cuda_host_vector_path = lambda _operation: None
    operator._profile_stage = lambda _stage, tick: tick
    operator._assemble_affine_self = lambda _values, _need_matrix: (
        0.0,
        np.zeros(operator.affine_dof, dtype=np.float64),
    )
    operator._assemble_soft_energy_gradient = lambda _need_matrix, _project_spd: None
    # The fixture already contains the exact one soft-soft and one mixed body
    # pair.  Bypassing only broad phase keeps this test small while every
    # narrow-phase, constitutive, scatter, and mode-dispatch path remains the
    # production implementation.
    operator._build_mixed_pairs = lambda _swept: None
    operator._raise_hash_triplet_overflow = lambda _stage: None
    return operator


def _record_production_contact_dispatch(operator, events):
    soft_barrier = operator._assemble_soft_soft_barrier
    mixed_barrier = operator._assemble_mixed_contact_type
    if operator.fully_implicit:
        soft_friction = operator._assemble_soft_fully_implicit_friction
        mixed_friction = operator._assemble_mixed_fully_implicit_friction
        friction_label = "fully_implicit"
    else:
        soft_friction = operator._assemble_soft_lagged_friction
        mixed_friction = operator._assemble_mixed_lagged_friction
        friction_label = "lagged"

    def assemble_soft_barrier(need_matrix, project_spd=None):
        events.append(("soft_barrier", bool(need_matrix)))
        return soft_barrier(need_matrix, project_spd)

    def assemble_mixed_barrier(contact_type, need_matrix, project_spd):
        events.append(("mixed_barrier", int(contact_type), bool(need_matrix)))
        return mixed_barrier(contact_type, need_matrix, project_spd)

    def assemble_soft_friction(need_matrix):
        events.append((f"soft_{friction_label}_friction", bool(need_matrix)))
        return soft_friction(need_matrix)

    def assemble_mixed_friction(need_matrix):
        events.append((f"mixed_{friction_label}_friction", bool(need_matrix)))
        return mixed_friction(need_matrix)

    operator._assemble_soft_soft_barrier = assemble_soft_barrier
    operator._assemble_mixed_contact_type = assemble_mixed_barrier
    if operator.fully_implicit:
        operator._assemble_soft_fully_implicit_friction = assemble_soft_friction
        operator._assemble_mixed_fully_implicit_friction = assemble_mixed_friction
    else:
        operator._assemble_soft_lagged_friction = assemble_soft_friction
        operator._assemble_mixed_lagged_friction = assemble_mixed_friction


@pytest.mark.parametrize("iterations", (1, 2, 7, -1))
def test_soft_affine_accepts_official_lagged_outer_iteration_options(iterations):
    sims = SimpleNamespace(
        affine_friction_mode="lagged",
        affine_friction_iterations=iterations,
    )
    assert _validate_soft_affine_lagged_friction_configuration(sims) == ("lagged", iterations)


def test_soft_affine_fully_implicit_capability_is_enabled():
    capabilities = soft_affine_friction_capabilities()
    assert capabilities["lagged"]["outer_fixed_point"]
    assert capabilities["fully_implicit"]["enabled"]
    assert capabilities["fully_implicit"]["full_nonsymmetric_jacobian"]
    assert _validate_soft_affine_lagged_friction_configuration(
        SimpleNamespace(
            affine_friction_mode="fully_implicit",
            affine_friction_iterations=1,
        )
    ) == ("fully_implicit", 0)


@pytest.mark.parametrize("iterations", (0, 2, -1, 1.5, "two"))
def test_soft_affine_fully_implicit_rejects_outer_friction_iterations(iterations):
    with pytest.raises(RuntimeError, match="fully implicit friction_iterations must be 1"):
        _validate_soft_affine_lagged_friction_configuration(
            SimpleNamespace(
                affine_friction_mode="fully_implicit",
                affine_friction_iterations=iterations,
            )
        )


@pytest.mark.parametrize(
    ("profile", "expected"),
    (("c1", 0), ("cinfinity", 1), ("c_infinity", 1)),
)
def test_soft_affine_fully_implicit_accepts_profile_aliases(profile, expected):
    operator = object.__new__(SoftAffineIPCOperator)
    operator.affine = SimpleNamespace(epsv=1.0e-3)
    operator.fully_implicit = True
    operator._configure_fully_implicit_friction(
        SimpleNamespace(
            affine_friction_profile=profile,
        )
    )
    assert operator.fully_profile_id == expected
    assert operator.fully_stribeck_velocity == pytest.approx(1.0e-2)


def test_soft_affine_fully_implicit_rejects_negative_stribeck_non_sentinel():
    operator = object.__new__(SoftAffineIPCOperator)
    operator.affine = SimpleNamespace(epsv=1.0e-3)
    operator.fully_implicit = True
    with pytest.raises(ValueError, match="non-negative or -1"):
        operator._configure_fully_implicit_friction(
            SimpleNamespace(
                affine_stribeck_velocity=-2.0,
            )
        )


@pytest.mark.parametrize("coefficient", ("dynamic", "static"))
def test_soft_affine_friction_rejects_negative_coefficient_non_sentinel(coefficient):
    parameter = {f"affine_{coefficient}_friction": -0.5}
    with pytest.raises(RuntimeError, match="non-negative or -1"):
        _validate_soft_affine_lagged_friction_configuration(
            SimpleNamespace(
                affine_friction_mode="lagged",
                **parameter,
            )
        )

    operator = object.__new__(SoftAffineIPCOperator)
    operator.affine = SimpleNamespace(epsv=1.0e-3)
    operator.fully_implicit = True
    with pytest.raises(ValueError, match="non-negative or -1"):
        operator._configure_fully_implicit_friction(
            SimpleNamespace(
                **parameter,
            )
        )


def test_soft_affine_fully_implicit_triplet_capacity_covers_full_mixed_pt(monkeypatch):
    monkeypatch.setenv("GT_SOFT_AFFINE_HASH_TRIPLET_SAFETY", "1.0")
    operator = object.__new__(SoftAffineIPCOperator)
    operator.soft_grid_num = 1
    operator.soft_point_num = 1
    operator.soft_num = 1
    operator.affine = SimpleNamespace(max_hash_triplets=1, face_num=1)
    operator.soft_shape_nodes = 27
    operator.soft_friction_capacity = 1
    operator.soft_contact_capacity = 1
    operator.mixed_friction_capacity = 3
    operator.mixed_contact_capacity = 3
    sims = SimpleNamespace(
        soft_affine_hash_triplet_capacity=0,
    )
    operator.fully_implicit = False
    lagged_capacity = operator._estimate_triplet_capacity(sims)
    operator.fully_implicit = True
    fully_capacity = operator._estimate_triplet_capacity(sims)
    mixed_stencil = 27 + 12
    mixed_barrier_and_friction_upper_bound = operator.mixed_friction_capacity * 2 * mixed_stencil * (mixed_stencil - 1)
    assert fully_capacity >= mixed_barrier_and_friction_upper_bound
    assert fully_capacity >= lagged_capacity

    operator.mixed_contact_capacity += 2
    enlarged_capacity = operator._estimate_triplet_capacity(sims)
    assert enlarged_capacity - fully_capacity == 2 * 2 * mixed_stencil * (mixed_stencil - 1)


def test_soft_affine_lagged_triplet_capacity_covers_two_symmetric_stencils(monkeypatch):
    monkeypatch.setenv("GT_SOFT_AFFINE_HASH_TRIPLET_SAFETY", "1.0")
    operator = object.__new__(SoftAffineIPCOperator)
    operator.soft_grid_num = 1
    operator.soft_point_num = 2
    operator.soft_num = 2
    operator.soft_shape_nodes = 2
    operator.affine = SimpleNamespace(max_hash_triplets=1, face_num=1)
    operator.soft_friction_capacity = 3
    operator.soft_contact_capacity = 5
    operator.mixed_friction_capacity = 7
    operator.mixed_contact_capacity = 11
    operator.fully_implicit = False
    sims = SimpleNamespace(soft_affine_hash_triplet_capacity=0)

    capacity = operator._estimate_triplet_capacity(sims)
    soft_l = 2 * 2
    mixed_l = 2 + 12
    expected_contacts = (3 + 5) * (soft_l * (soft_l - 1) // 2) + (7 + 11) * (mixed_l * (mixed_l - 1) // 2)
    assert capacity >= expected_contacts


def test_soft_affine_reduced_capacity_uses_structured_grid_adjacency():
    operator = object.__new__(SoftAffineIPCOperator)
    operator.sims = SimpleNamespace(soft_grid_type_id=0)
    operator.max_dof = 3 * (56 + 8 * 19**3)
    operator.soft_grid_num = 8 * 19**3
    operator.soft_point_num = 8 * 9_925
    operator.soft_num = 8
    operator.soft_shape_nodes = 27
    operator.affine = SimpleNamespace(control_num=56)
    operator.soft_friction_capacity = 1
    operator.soft_contact_capacity = 1
    operator.mixed_friction_capacity = 1
    operator.mixed_contact_capacity = 1
    operator.fully_implicit = False

    raw_capacity = operator.soft_point_num * operator.soft_shape_nodes**2
    reduced = operator._estimate_reduced_triplet_capacity(raw_capacity)
    structured_soft_bound = operator.soft_grid_num * (5**3 - 1) // 2
    assert reduced >= structured_soft_bound
    assert reduced < raw_capacity // 10


@pytest.mark.parametrize("safety", ("0.99", "nan", "inf"))
def test_soft_affine_triplet_safety_rejects_unsafe_values(monkeypatch, safety):
    monkeypatch.setenv("GT_SOFT_AFFINE_HASH_TRIPLET_SAFETY", safety)
    operator = object.__new__(SoftAffineIPCOperator)
    operator.soft_grid_num = 1
    operator.soft_point_num = 1
    operator.soft_num = 1
    operator.affine = SimpleNamespace(max_hash_triplets=1, face_num=1)
    operator.soft_shape_nodes = 1
    operator.soft_friction_capacity = 1
    operator.soft_contact_capacity = 1
    operator.mixed_friction_capacity = 1
    operator.mixed_contact_capacity = 1
    operator.fully_implicit = False
    with pytest.raises(ValueError, match="finite and >= 1"):
        operator._estimate_triplet_capacity(SimpleNamespace(soft_affine_hash_triplet_capacity=0))


def test_soft_affine_barrier_capacity_fails_before_scatter():
    operator = object.__new__(SoftAffineIPCOperator)
    operator.soft_barrier_count = _ScalarField(0)
    operator.soft_contact_capacity = 2
    operator._count_soft_barrier_contacts = lambda: setattr(operator.soft_barrier_count, "value", 3)
    with pytest.raises(RuntimeError, match="required 3, capacity 2"):
        operator._validate_soft_barrier_contact_capacity()

    events = []
    operator.stable_lagged_contacts = True
    operator.affine = SimpleNamespace(levelset_contact=False)
    operator.mixed_pair_num = _ScalarField(1)
    operator.fully_implicit = False
    operator.mixed_contact_capacity = 4
    operator._count_mixed_contact_types = lambda: events.append("count")
    operator._mixed_active_contact_count = lambda: 5
    operator._mixed_contact_dispatch_mask = lambda: pytest.fail("barrier scatter started after capacity overflow")
    with pytest.raises(RuntimeError, match="required 5, capacity 4"):
        operator._assemble_mixed_contact(True)
    assert events == ["count"]


def test_soft_affine_matrix_assembly_reports_triplet_overflow():
    operator = object.__new__(SoftAffineIPCOperator)
    operator.hash_triplet = SimpleNamespace(
        overflow=np.array([1], dtype=np.int32),
        raw_non_diag_count=np.array([19], dtype=np.int32),
        non_diag=SimpleNamespace(max_pairs_num=8),
    )
    with pytest.raises(RuntimeError, match="used 19, capacity 8"):
        operator._raise_hash_triplet_overflow("unit-test assembly")


def test_soft_affine_cuda_rejects_host_hash_reduction():
    operator = object.__new__(SoftAffineIPCOperator)
    operator.cuda_hot_loop = True
    operator.fully_implicit = True
    operator.hash_triplet = SimpleNamespace(
        solver="BiCGSTAB",
        matrix_symmetric=False,
        non_diag=SimpleNamespace(device_reduction=False),
    )
    with pytest.raises(RuntimeError, match="device reduction"):
        operator._assert_cuda_device_residency()


def test_soft_affine_lagged_requires_official_projected_spd_pcg():
    operator = object.__new__(SoftAffineIPCOperator)
    operator.fully_implicit = False
    operator.cuda_hot_loop = False
    operator.hash_triplet = SimpleNamespace(
        solver="PCG",
        matrix_symmetric=True,
        non_diag=SimpleNamespace(device_reduction=False),
    )
    operator._assert_cuda_device_residency()

    operator.hash_triplet.solver = "BiCGSTAB"
    with pytest.raises(RuntimeError, match="PCG"):
        operator._assert_cuda_device_residency()


def test_soft_affine_cuda_rejects_cpu_full_vector_entry_points():
    operator = object.__new__(SoftAffineIPCOperator)
    operator.cuda_hot_loop = True
    with pytest.raises(RuntimeError, match="CPU reference path"):
        operator.assemble(np.zeros(1), need_matrix=False)
    with pytest.raises(RuntimeError, match="CPU reference path"):
        operator.solve_direction(SimpleNamespace(), np.zeros(1))
    with pytest.raises(RuntimeError, match="CPU reference path"):
        operator.apply_jacobian(np.zeros(1))


def test_soft_affine_cuda_hot_loop_has_no_nonlinear_vector_numpy_roundtrip():
    methods = (
        SoftAffineIPCOperator._initialize_affine_contact_damping_device,
        SoftAffineIPCOperator._assemble_affine_self_fields,
        SoftAffineIPCOperator._assemble_mixed_contact,
        SoftAffineIPCOperator.assemble_device,
        SoftAffineIPCOperator.solve_direction_device,
        SoftAffineIPCOperator.init_step_size_device,
        SoftAffineIPCEngine._step_lagged_device,
        SoftAffineIPCEngine._step_fully_implicit_device,
        SoftAffineIPCEngine._solve_fully_implicit_newton_device,
        SoftAffineIPCEngine._fully_implicit_line_search_device,
        SoftAffineIPCEngine._solve_lagged_inner_device,
        SoftAffineIPCEngine._line_search_device,
    )
    for method in methods:
        source = inspect.getsource(method)
        assert ".to_numpy(" not in source, method.__qualname__
        assert ".from_numpy(" not in source, method.__qualname__


def test_soft_affine_fully_implicit_predictor_rejects_zero_ccd_step():
    operator = SimpleNamespace(
        _build_affine_velocity_predictor_device=lambda: None,
        _build_fully_implicit_velocity_predictor=lambda: None,
        init_step_size_device=lambda **_kwargs: 0.0,
        store_coupled_base_device=lambda: pytest.fail("zero-step predictor was accepted"),
        set_coupled_trial_device=lambda _alpha: pytest.fail("zero-step predictor was applied"),
    )
    sims = SimpleNamespace(
        affine_ccd=True,
        affine_ccd_type="ccd",
        affine_ccd_eta=0.2,
        affine_accd_tolerance=1.0e-7,
        affine_ccd_max_iteration=100,
    )
    with pytest.raises(RuntimeError, match="positive CCD step"):
        SoftAffineIPCOperator.initialize_fully_implicit_velocity_predictor_device(operator, sims)


def test_soft_affine_lagged_device_zero_correction_is_convergence():
    engine = SoftAffineIPCEngine()
    engine.operator = SimpleNamespace(
        dt=1.0e-2,
        solve_direction_device=lambda _sims: {
            "solution_inf_norm": 0.0,
            "unclamped_solution_inf_norm": 0.0,
        },
    )
    engine._line_search_device = lambda *_args: pytest.fail("a converged Newton probe must not be applied")
    sims = SimpleNamespace(
        affine_newton_tolerance=1.0e-8,
        affine_max_newton_iteration=2,
    )
    energy, iterations = engine._solve_lagged_inner_device(sims, energy=0.0)
    assert energy == 0.0
    assert iterations == 0
    assert engine.last_inner_converged
    assert engine.last_inner_failure_reason == ""


def test_soft_affine_lagged_device_applies_first_nonzero_small_correction():
    engine = SoftAffineIPCEngine()
    results = iter(
        (
            {
                "solution_inf_norm": 5.0e-10,
                "unclamped_solution_inf_norm": 5.0e-10,
            },
            {
                "solution_inf_norm": 0.0,
                "unclamped_solution_inf_norm": 0.0,
            },
        )
    )
    engine.operator = SimpleNamespace(
        dt=1.0e-2,
        solve_direction_device=lambda _sims: next(results),
    )
    applied = []
    engine._line_search_device = lambda _sims, energy: applied.append(energy) or energy - 1.0
    sims = SimpleNamespace(
        affine_newton_tolerance=1.0e-7,
        affine_max_newton_iteration=3,
    )

    energy, iterations = engine._solve_lagged_inner_device(sims, energy=2.0)

    assert applied == [2.0]
    assert energy == 1.0
    assert iterations == 1
    assert engine.last_inner_converged
    np.testing.assert_allclose(engine.last_inner_correction_history, [5.0e-8, 0.0])


def test_soft_affine_lagged_cpu_uses_raw_correction_velocity_not_gradient():
    engine = SoftAffineIPCEngine()
    directions = iter(
        (
            np.asarray([2.0e-9, -1.0e-9]),
            np.asarray([5.0e-10, 0.0]),
        )
    )
    applied_affine_directions = []

    def solve_direction(_sims, _gradient):
        direction = next(directions)
        return direction[:1], direction[1:], direction

    engine.operator = SimpleNamespace(
        dt=1.0e-2,
        affine_dof=1,
        total_dof=2,
        max_soft_dof=1,
        soft_direction=_ArrayField(np.zeros(1)),
        solve_direction=solve_direction,
    )

    def line_search(_sims, _y, energy, gradient, affine_direction):
        applied_affine_directions.append(np.asarray(affine_direction).copy())
        full_direction = np.concatenate((affine_direction, engine.operator.soft_direction.to_numpy()))
        return 1.0, energy, gradient, affine_direction, full_direction

    engine._line_search = line_search
    sims = SimpleNamespace(
        affine_newton_tolerance=1.0e-7,
        affine_max_newton_iteration=3,
        # The first raw correction is 2e-9 m, or 2e-7 m/s, and therefore is
        # not converged.  Its applied copy is truncated to 1e-10 m; using the
        # truncated direction would incorrectly report convergence.
        affine_max_step=1.0e-10,
    )
    initial_y = np.zeros(1)
    large_gradient = np.full(2, 1.0e12)

    y, energy, gradient, iterations = engine._solve_lagged_inner(sims, initial_y, 3.0, large_gradient)

    assert iterations == 1
    assert engine.last_inner_converged
    np.testing.assert_allclose(y, [1.0e-10], atol=0.0)
    np.testing.assert_allclose(applied_affine_directions, [[1.0e-10]], atol=0.0)
    assert energy == 3.0
    np.testing.assert_array_equal(gradient, large_gradient)


def test_soft_affine_lagged_device_uses_raw_correction_velocity():
    engine = SoftAffineIPCEngine()
    results = iter(
        (
            {
                "solution_inf_norm": 1.0e-10,
                "unclamped_solution_inf_norm": 2.0e-9,
            },
            {
                "solution_inf_norm": 5.0e-10,
                "unclamped_solution_inf_norm": 5.0e-10,
            },
        )
    )
    line_search_calls = []
    engine.operator = SimpleNamespace(
        dt=1.0e-2,
        solve_direction_device=lambda _sims: next(results),
    )
    engine._line_search_device = lambda _sims, energy: line_search_calls.append(energy) or (energy - 1.0)
    sims = SimpleNamespace(
        affine_newton_tolerance=1.0e-7,
        affine_max_newton_iteration=3,
    )

    energy, iterations = engine._solve_lagged_inner_device(sims, energy=4.0)

    assert iterations == 1
    assert energy == 3.0
    assert line_search_calls == [4.0]
    assert engine.last_inner_converged


def test_soft_affine_lagged_device_rejects_nonfinite_energy_and_slope():
    engine = SoftAffineIPCEngine()
    engine.operator = SimpleNamespace(
        dt=1.0e-2,
        residual_inf_norm_device=lambda: 1.0,
        energy_directional_derivative_device=lambda: np.nan,
    )
    sims = SimpleNamespace(
        affine_newton_tolerance=1.0e-8,
        affine_max_newton_iteration=2,
    )
    with pytest.raises(RuntimeError, match="non-finite IPC energy"):
        engine._solve_lagged_inner_device(sims, energy=np.inf)
    with pytest.raises(RuntimeError, match="non-finite energy derivative"):
        engine._line_search_device(sims, energy=0.0)


def test_soft_affine_lagged_line_search_uses_monotone_energy_cpu_and_device():
    sims = SimpleNamespace(
        affine_line_search_max_iteration=3,
        affine_ccd=False,
    )
    base_energy = 1.0
    trial_energy = 0.99995

    cpu_engine = SoftAffineIPCEngine()
    cpu_engine.operator = SimpleNamespace(
        total_dof=1,
        affine_dof=1,
        soft_direction=_ArrayField(np.zeros(0)),
        store_soft_base=lambda: None,
        set_soft_trial=lambda _alpha: None,
        assemble=lambda _trial, need_matrix: (
            trial_energy,
            np.zeros(1),
        ),
    )
    alpha, accepted, _, _, _ = cpu_engine._line_search(
        sims,
        np.zeros(1),
        base_energy,
        np.ones(1),
        -np.ones(1),
    )
    assert alpha == 1.0
    assert accepted == trial_energy

    device_engine = SoftAffineIPCEngine()
    trial_alphas = []
    device_engine.operator = SimpleNamespace(
        energy_directional_derivative_device=lambda: -1.0,
        store_coupled_base_device=lambda: None,
        set_coupled_trial_device=lambda alpha: trial_alphas.append(alpha),
        assemble_device=lambda need_matrix: trial_energy,
    )
    accepted_device = device_engine._line_search_device(sims, base_energy)
    assert accepted_device == trial_energy
    assert trial_alphas == [1.0]


def test_soft_affine_device_dispatch_masks_and_rollback_stay_in_fields():
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    operator = object.__new__(SoftAffineIPCOperator)
    edge_type_count = ti.field(ti.i32, shape=9)
    edge_mollifier_type_count = ti.field(ti.i32, shape=9)
    operator.affine = SimpleNamespace(
        control_num=2,
        edge_type_count=edge_type_count,
        edge_mollifier_type_count=edge_mollifier_type_count,
        y=ti.Vector.field(3, float, shape=2),
    )
    operator.mixed_contact_type_count = ti.field(ti.i32, shape=7)
    operator.affine_y_step_start = ti.Vector.field(3, float, shape=2)
    operator.max_soft_dof = 3
    operator.soft_disp = ti.field(float, shape=3)
    operator.soft_disp_base = ti.field(float, shape=3)

    edge_type_count[1] = 2
    edge_type_count[8] = 1
    edge_mollifier_type_count[3] = 4
    assert int(operator._affine_edge_dispatch_mask()) == ((1 << 1) | (1 << 8) | (1 << (3 + 9)))
    operator.mixed_contact_type_count[0] = 1
    operator.mixed_contact_type_count[6] = 3
    assert int(operator._mixed_contact_dispatch_mask()) == ((1 << 0) | (1 << 6))

    initial_y = np.array([[0.1, -0.2, 0.3], [1.0, 2.0, 3.0]], dtype=np.float64)
    operator.affine.y.from_numpy(initial_y)
    operator._store_affine_step_start_device()
    operator.affine.y.fill(9.0)
    operator.soft_disp.fill(2.0)
    operator.soft_disp_base.fill(-3.0)
    operator.rollback_step_device()
    np.testing.assert_allclose(operator.affine.y.to_numpy(), initial_y)
    np.testing.assert_allclose(operator.soft_disp.to_numpy(), 0.0)
    np.testing.assert_allclose(operator.soft_disp_base.to_numpy(), 0.0)


@pytest.mark.parametrize(
    ("mode", "device_method", "expected"),
    (
        ("lagged", "_step_lagged_device", "lagged-device"),
        ("fully_implicit", "_step_fully_implicit_device", "fully-device"),
    ),
)
def test_soft_affine_cuda_step_cannot_route_to_cpu_driver(mode, device_method, expected):
    engine = SoftAffineIPCEngine()
    events = []
    engine.operator = SimpleNamespace(
        cuda_hot_loop=True,
        _assert_cuda_device_residency=lambda: events.append("residency-check"),
    )
    engine._step_lagged_device = lambda _sims, _outer: (
        engine.operator._assert_cuda_device_residency() or "lagged-device"
    )
    engine._step_fully_implicit_device = lambda _sims, _outer=None: (
        engine.operator._assert_cuda_device_residency() or "fully-device"
    )
    engine._step_fully_implicit = lambda _sims: pytest.fail("CUDA routed to the CPU fully implicit driver")
    engine._solve_lagged_inner = lambda *_args: pytest.fail("CUDA routed to the CPU lagged driver")
    engine.advance = getattr(engine, device_method)
    engine.requested_friction_iterations = 1
    sims = SimpleNamespace(
        affine_friction_mode=mode,
        affine_friction_iterations=1,
        dt=_ScalarField(1.0e-2),
        current_time=0.0,
        current_step=0,
        delta=1.0e-2,
    )
    assert engine.step(sims, scene=None) == expected
    assert events == ["residency-check"]


def test_soft_affine_merit_jacobian_removes_solver_diagonal_shift():
    operator = object.__new__(SoftAffineIPCOperator)
    operator.cuda_hot_loop = False
    operator.total_dof = 3
    operator.fully_implicit = True
    operator.sims = SimpleNamespace(affine_fully_implicit_jacobian_shift=0.7)
    physical_jacobian = np.array(
        [[2.0, -0.4, 0.1], [0.3, 1.5, -0.2], [0.05, 0.2, 1.1]],
        dtype=np.float64,
    )
    x = _ArrayField(np.zeros((1, 3), dtype=np.float64))
    Ax = _ArrayField(np.zeros((1, 3), dtype=np.float64))

    def matvec(active_nodes, _nnz, input_field, output_field):
        direction = input_field.to_numpy()[:active_nodes].reshape(-1)
        shifted = physical_jacobian + 0.7 * np.eye(3 * active_nodes)
        output_field.from_numpy((shifted @ direction).reshape((-1, 3)))

    operator.hash_triplet = SimpleNamespace(
        x=x,
        Ax=Ax,
        non_diag=SimpleNamespace(element_pair_num=np.array([0], dtype=np.int32)),
        _pad_vector_array=lambda values: np.asarray(values, dtype=np.float64),
        matvec=matvec,
    )
    direction = np.array([0.2, -0.6, 0.4], dtype=np.float64)
    np.testing.assert_allclose(
        operator.apply_jacobian(direction),
        physical_jacobian @ direction,
        atol=1.0e-14,
    )


def test_soft_affine_device_newton_vectors_and_jp_stay_in_taichi_fields():
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    operator = object.__new__(SoftAffineIPCOperator)
    operator.cuda_hot_loop = True
    operator.max_dof = 6
    operator.total_dof = 6
    operator.affine_dof = 3
    operator.max_soft_dof = 3
    operator.fully_implicit = True
    operator.profile = False
    operator.global_grad = ti.field(float, shape=6)
    residual = np.array([1.0, -2.0, 0.5, -0.75, 1.25, -0.2], dtype=np.float64)
    operator.soft_direction = ti.field(float, shape=3)
    operator.device_status = ti.field(ti.i32, shape=())
    operator.energy = ti.field(float, shape=())
    operator.affine = SimpleNamespace(
        control_num=1,
        y=ti.Vector.field(3, float, shape=1),
        direction_y=ti.Vector.field(3, float, shape=1),
        grad=ti.Vector.field(3, float, shape=1),
        energy=ti.field(float, shape=()),
    )
    operator.affine_y_base = ti.Vector.field(3, float, shape=1)
    operator.soft_disp = ti.field(float, shape=3)
    operator.soft_disp_base = ti.field(float, shape=3)
    operator.hash_triplet = BuildTriplet(
        dim=3,
        max_pairs_num=8,
        max_nonzeros=8,
        max_active_nodes=2,
        symmetric=False,
        solver="BiCGSTAB",
        matrix_symmetric=False,
    )
    physical_diagonal = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
    shift = 0.7
    diagonal_blocks = np.zeros((2, 9), dtype=np.float64)
    for block in range(2):
        diagonal_blocks[block, [0, 4, 8]] = physical_diagonal[3 * block : 3 * block + 3] + shift
    operator.hash_triplet.diag.from_numpy(diagonal_blocks)
    operator.sims = SimpleNamespace(affine_fully_implicit_jacobian_shift=shift)
    sims = SimpleNamespace(
        affine_linear_tolerance=1.0e-13,
        affine_linear_max_iteration=50,
        affine_max_step=0.0,
    )

    operator.affine.grad.from_numpy(np.array([[0.2, -0.3, 0.4]], dtype=np.float64))
    operator.affine.energy[None] = 1.25
    operator._seed_coupled_residual_from_affine()
    assert operator.energy[None] == pytest.approx(1.25)
    np.testing.assert_allclose(operator.global_grad.to_numpy(), [0.2, -0.3, 0.4, 0.0, 0.0, 0.0])
    operator.global_grad.from_numpy(residual)
    assert operator.residual_inf_norm_device() == pytest.approx(2.0)
    assert operator.residual_merit_device() == pytest.approx(0.5 * float(np.dot(residual, residual)))

    result = operator.solve_direction_device(sims)
    expected_direction = -residual / (physical_diagonal + shift)
    np.testing.assert_allclose(
        operator.hash_triplet.x.to_numpy().reshape(-1),
        expected_direction,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        operator.affine.direction_y.to_numpy().reshape(-1),
        expected_direction[:3],
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        operator.soft_direction.to_numpy(),
        expected_direction[3:],
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert result["solution_inf_norm"] == pytest.approx(np.linalg.norm(expected_direction, ord=np.inf))
    assert result["unclamped_solution_inf_norm"] == pytest.approx(np.linalg.norm(expected_direction, ord=np.inf))

    expected_slope = float(
        np.dot(
            residual,
            physical_diagonal * expected_direction,
        )
    )
    assert operator.residual_merit_directional_derivative_device() == pytest.approx(
        expected_slope, rel=1.0e-11, abs=1.0e-12
    )

    affine_base = np.array([[0.2, -0.4, 0.8]], dtype=np.float64)
    soft_base = np.array([0.3, -0.6, 0.15], dtype=np.float64)
    operator.affine.y.from_numpy(affine_base)
    operator.soft_disp.from_numpy(soft_base)
    operator.store_coupled_base_device()
    operator.set_coupled_trial_device(0.25)
    np.testing.assert_allclose(
        operator.affine.y.to_numpy().reshape(-1),
        affine_base.reshape(-1) + 0.25 * expected_direction[:3],
    )
    np.testing.assert_allclose(
        operator.soft_disp.to_numpy(),
        soft_base + 0.25 * expected_direction[3:],
    )
    operator.restore_coupled_base_device()
    np.testing.assert_allclose(operator.affine.y.to_numpy(), affine_base)
    np.testing.assert_allclose(operator.soft_disp.to_numpy(), soft_base)

    sims.affine_max_step = 5.0e-2
    clamped_result = operator.solve_direction_device(sims)
    assert clamped_result["solution_inf_norm"] == pytest.approx(5.0e-2)
    assert clamped_result["unclamped_solution_inf_norm"] == pytest.approx(
        np.linalg.norm(expected_direction, ord=np.inf)
    )


def test_soft_affine_lagged_rejects_unrepresented_friction_laws():
    base = dict(
        affine_friction_mode="lagged",
        affine_friction_iterations=1,
    )
    with pytest.raises(RuntimeError, match="static_friction differs"):
        _validate_soft_affine_lagged_friction_configuration(
            SimpleNamespace(**base, affine_dynamic_friction=0.3, affine_static_friction=0.6)
        )
    with pytest.raises(RuntimeError, match="viscous_friction"):
        _validate_soft_affine_lagged_friction_configuration(SimpleNamespace(**base, affine_viscous_friction=0.01))
    with pytest.raises(RuntimeError, match="quadratic IPC C1"):
        _validate_soft_affine_lagged_friction_configuration(
            SimpleNamespace(**base, affine_friction_profile="stabilized")
        )
    assert _validate_soft_affine_lagged_friction_configuration(
        SimpleNamespace(**base, affine_dynamic_friction=0.4, affine_static_friction=0.4)
    ) == ("lagged", 1)


def test_soft_affine_rejects_mode_change_after_operator_initialization():
    engine = SoftAffineIPCEngine()
    engine.operator = SimpleNamespace(friction_mode="lagged")
    engine.soft_material = object()
    scene = SimpleNamespace(
        softNum=np.array([1], dtype=np.int32),
        affine_bodies=[object()],
    )
    sims = SimpleNamespace(
        affine_friction_mode="fully_implicit",
        affine_assemble_type="HashTriplet",
    )
    with pytest.raises(RuntimeError, match="cannot be changed"):
        engine.initialize(sims, scene)


def test_soft_affine_swept_body_pairs_catch_fast_soft_and_mixed_ccd():
    operator = _make_swept_ccd_harness()

    # The two point bodies start much farther apart than dhat, but the first
    # point reaches the second within one Newton step.  Current AABBs are
    # disjoint; swept AABBs must still feed the real PP CCD kernel.
    _set_soft_motion(
        operator,
        positions=[[-2.0, 0.0, 3.0], [2.0, 0.0, 3.0]],
        directions=[[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    operator._build_mixed_pairs(0)
    assert int(operator.soft_pair_num[None]) == 0
    soft_alpha = operator._soft_soft_ccd_step_size(ccd_type="ccd", eta=0.2, max_iteration=1000)
    assert int(operator.soft_pair_num[None]) == 1
    assert 0.0 < soft_alpha < 1.0

    # The active point starts well above the affine triangle, which translates
    # through it during the proposed affine step.  This separately exercises
    # the affine ``x + dx`` endpoint of the mixed swept box.
    # An inactive point closer to the triangle may enter the broad phase,
    # but must not restrict the accepted CCD step.
    _set_soft_motion(
        operator,
        positions=[[0.0, 0.0, 2.0], [0.0, 0.0, 0.2]],
        directions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        active=(1, 0),
    )
    operator._build_mixed_pairs(0)
    assert int(operator.mixed_pair_num[None]) == 0
    mixed_alpha = operator._mixed_ccd_step_size(
        operator.affine_state.pack(),
        np.tile([0.0, 0.0, 3.0], operator.affine.control_num),
        ccd_type="ccd",
        eta=0.2,
        max_iteration=1000,
    )
    assert int(operator.mixed_bvh.point_triangle_count[None]) == 2
    assert 0.0 < mixed_alpha < 1.0
    operator.scene.soft_point[1].active = 1
    operator.ccd_alpha[None] = 1.0
    operator._compute_mixed_ccd_alpha(0.2, 0.0, 1000, False)
    assert float(operator.ccd_alpha[None]) == pytest.approx(0.1 * mixed_alpha)


def test_bilateral_pp_pullback_is_equal_opposite_and_symmetric():
    # Two material points, each interpolated from two generalized nodes.  This
    # is the same linear PP stencil used by SoftAffine's Taichi scatter.
    weight_p = np.array([0.25, 0.75])
    weight_q = np.array([0.6, 0.4])
    interpolation = np.zeros((6, 12), dtype=np.float64)
    for component in range(3):
        interpolation[component, component] = weight_p[0]
        interpolation[component, 3 + component] = weight_p[1]
        interpolation[3 + component, 6 + component] = weight_q[0]
        interpolation[3 + component, 9 + component] = weight_q[1]

    relative_gradient = np.array([1.25, -0.5, 0.75])
    local_gradient = np.r_[relative_gradient, -relative_gradient]
    relative_hessian = np.array([[3.0, 0.2, -0.1], [0.2, 2.0, 0.4], [-0.1, 0.4, 1.5]])
    local_hessian = np.block([[relative_hessian, -relative_hessian], [-relative_hessian, relative_hessian]])
    gradient, hessian = pullback_dense(local_gradient, local_hessian, interpolation)

    nodal_gradient = gradient.reshape((-1, 3))
    np.testing.assert_allclose(nodal_gradient.sum(axis=0), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(hessian, hessian.T, atol=1.0e-14)
    translation = np.tile(np.array([0.2, -0.4, 0.7]), 4)
    np.testing.assert_allclose(hessian @ translation, 0.0, atol=1.0e-14)


def test_soft_affine_official_barrier_and_friction_contact_weights():
    operator = _make_bilateral_kernel_harness()
    operator.scene.soft_point[0].surface_weight = 2.0
    operator.scene.soft_point[1].surface_weight = 4.0
    # An explicit geometric surface measure must not be silently replaced by
    # the volume-based fallback, even when the latter is larger.
    operator.scene.soft_point[0].vol0 = 1.0e6
    operator.scene.soft_point[1].vol0 = 1.0e6

    dhat = float(operator.affine.pp_dhat[0, 0])
    kappa = float(operator.affine.pp_kappa[0, 0])
    mu = float(operator.affine.pp_mu[0, 0])
    distance2 = 0.05**2
    barrier, db, _ = _toolkit_barrier_distance2_oracle(distance2, dhat * dhat, kappa)
    normal_force = -2.0 * db * np.sqrt(distance2)
    vv_measure = 0.5 * (2.0 + 4.0)
    fv_measure = 0.25 * 2.0

    operator.global_grad.fill(0.0)
    operator.energy[None] = 0.0
    operator._assemble_soft_soft_barrier(False)
    assert float(operator.energy[None]) == pytest.approx(operator.scale * vv_measure * barrier, rel=1.0e-12)

    operator.global_grad.fill(0.0)
    operator.energy[None] = 0.0
    operator._assemble_mixed_contact_type(6, False, True)
    assert float(operator.energy[None]) == pytest.approx(operator.scale * fv_measure * barrier, rel=1.0e-12)

    operator._reset_lagged_friction_cache()
    operator._initialize_soft_lagged_friction()
    operator._initialize_mixed_lagged_friction()
    assert float(operator.soft_friction_coeff[0]) == pytest.approx(
        operator.scale * vv_measure * mu * normal_force, rel=1.0e-12
    )
    assert float(operator.mixed_friction_coeff[0]) == pytest.approx(
        operator.scale * fv_measure * mu * normal_force, rel=1.0e-12
    )


def test_soft_affine_production_bilateral_pp_kernels_and_frozen_cache():
    operator = _make_bilateral_kernel_harness()
    operator._reset_lagged_friction_cache()
    operator._initialize_soft_lagged_friction()
    assert int(operator.soft_friction_count[None]) == 1
    frozen_normal = operator.soft_friction_normal.to_numpy()[0].copy()
    frozen_coeff = float(operator.soft_friction_coeff[0])

    operator.global_grad.fill(0.0)
    operator.energy[None] = 0.0
    operator.hash_triplet.values.fill(0.0)
    operator._assemble_soft_soft_barrier(True)
    barrier_gradient = operator.global_grad.to_numpy()[operator.affine_dof :]
    np.testing.assert_allclose(barrier_gradient.reshape((2, 3)).sum(axis=0), 0.0, atol=1.0e-10)
    upper = operator.hash_triplet.values.to_numpy()[operator.affine_dof :, operator.affine_dof :]
    barrier_hessian = upper + upper.T - np.diag(np.diag(upper))
    np.testing.assert_allclose(barrier_hessian, barrier_hessian.T, atol=1.0e-12)
    barrier_eigenvalues = np.linalg.eigvalsh(barrier_hessian)
    assert barrier_eigenvalues[0] >= (-1.0e-10 * max(1.0, float(np.max(np.abs(barrier_eigenvalues)))))
    np.testing.assert_allclose(barrier_hessian @ np.tile([0.3, -0.2, 0.1], 2), 0.0, rtol=1.0e-9, atol=1.0e-8)

    displacement = np.zeros(6, dtype=np.float64)
    displacement[1] = 2.0e-4
    operator.soft_disp.from_numpy(displacement)
    operator.global_grad.fill(0.0)
    operator.energy[None] = 0.0
    operator.hash_triplet.values.fill(0.0)
    operator._assemble_soft_lagged_friction(True)
    friction_gradient = operator.global_grad.to_numpy()[operator.affine_dof :]
    assert float(operator.energy[None]) > 0.0
    np.testing.assert_allclose(friction_gradient.reshape((2, 3)).sum(axis=0), 0.0, atol=1.0e-10)
    upper = operator.hash_triplet.values.to_numpy()[operator.affine_dof :, operator.affine_dof :]
    friction_hessian = upper + upper.T - np.diag(np.diag(upper))
    friction_eigenvalues = np.linalg.eigvalsh(friction_hessian)
    assert friction_eigenvalues[0] >= (-1.0e-10 * max(1.0, float(np.max(np.abs(friction_eigenvalues)))))
    np.testing.assert_allclose(operator.soft_friction_normal.to_numpy()[0], frozen_normal, atol=0.0)
    assert float(operator.soft_friction_coeff[0]) == pytest.approx(frozen_coeff)


def test_soft_affine_coupled_lagged_friction_scale_vjp():
    operator = _make_bilateral_kernel_harness()
    operator._reset_lagged_friction_cache()
    operator._initialize_soft_lagged_friction()
    operator._initialize_mixed_lagged_friction()
    displacement = np.zeros(6, dtype=np.float64)
    displacement[1] = 2.0e-4
    operator.soft_disp.from_numpy(displacement)
    adjoint = np.zeros(18, dtype=np.float64)
    adjoint[12:] = np.array([0.2, -0.3, 0.1, -0.1, 0.4, -0.2])
    operator.adjoint_solution.from_numpy(adjoint)

    def residual(scale):
        operator.affine.friction_scale[0] = scale
        operator.global_grad.fill(0.0)
        operator._assemble_soft_lagged_friction(False)
        operator._assemble_mixed_lagged_friction(False)
        return operator.global_grad.to_numpy()

    epsilon = 1.0e-6
    expected = -adjoint @ ((residual(1.0 + epsilon) - residual(1.0 - epsilon)) / (2.0 * epsilon))
    operator.affine.friction_scale[0] = 1.0
    operator.affine.friction_scale_vjp[None] = 0.0
    operator._differentiate_coupled_friction_scale_parameter()

    assert float(operator.affine.friction_scale_vjp[None]) == pytest.approx(expected, rel=2.0e-10, abs=1.0e-12)


def test_soft_affine_frozen_contact_compaction_is_stable_across_refreshes():
    operator = _make_bilateral_kernel_harness()
    _enable_stable_contact_compaction(operator)
    operator.soft_barrier_count = ti.field(ti.i32, shape=())
    operator.soft_contact_capacity = 2
    operator.mixed_contact_capacity = 2

    operator._count_soft_barrier_contacts()
    operator._count_mixed_contact_types()
    assert int(operator.soft_barrier_count[None]) == 1
    assert int(operator._mixed_active_contact_count()) == 1

    snapshots = []
    for _ in range(2):
        operator._reset_lagged_friction_cache()
        operator._initialize_soft_lagged_friction()
        operator._initialize_mixed_lagged_friction()
        soft_count = int(operator.soft_friction_count[None])
        mixed_count = int(operator.mixed_friction_count[None])
        snapshots.append(
            (
                soft_count,
                mixed_count,
                operator.soft_friction_points.to_numpy()[:soft_count].copy(),
                operator.mixed_friction_point.to_numpy()[:mixed_count].copy(),
                operator.mixed_friction_face.to_numpy()[:mixed_count].copy(),
            )
        )

    assert snapshots[0][0:2] == snapshots[1][0:2] == (1, 1)
    for first, second in zip(snapshots[0][2:], snapshots[1][2:]):
        np.testing.assert_array_equal(first, second)


def test_soft_affine_fi_barrier_retains_exact_negative_curvature():
    lagged = _make_bilateral_kernel_harness()
    _clear_coupled_contact_system(lagged)
    lagged._assemble_soft_soft_barrier(True)
    upper = lagged.hash_triplet.values.to_numpy()[lagged.affine_dof :, lagged.affine_dof :]
    lagged_hessian = upper + upper.T - np.diag(np.diag(upper))

    fully_implicit = _make_bilateral_kernel_harness()
    _enable_fully_implicit_friction(fully_implicit)
    _clear_coupled_contact_system(fully_implicit)
    fully_implicit._assemble_soft_soft_barrier(True)
    exact_hessian = fully_implicit.hash_triplet.values.to_numpy()[
        fully_implicit.affine_dof :, fully_implicit.affine_dof :
    ]

    adjoint = _make_bilateral_kernel_harness()
    _clear_coupled_contact_system(adjoint)
    adjoint._assemble_soft_soft_barrier(True, project_spd=False)
    upper = adjoint.hash_triplet.values.to_numpy()[adjoint.affine_dof :, adjoint.affine_dof :]
    adjoint_hessian = upper + upper.T - np.diag(np.diag(upper))

    lagged_scale = max(1.0, np.linalg.norm(lagged_hessian, ord=2))
    exact_scale = max(1.0, np.linalg.norm(exact_hessian, ord=2))
    assert np.linalg.eigvalsh(lagged_hessian).min() >= -1.0e-10 * lagged_scale
    # A PP squared-distance barrier has negative tangential curvature inside
    # dhat.  FI must retain it exactly rather than reusing lagged make-PD.
    assert np.linalg.eigvalsh(0.5 * (exact_hessian + exact_hessian.T)).min() < -1.0e-8 * exact_scale
    np.testing.assert_allclose(adjoint_hessian, exact_hessian, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        fully_implicit.global_grad.to_numpy(),
        lagged.global_grad.to_numpy(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_soft_affine_unprojected_mixed_barrier_matches_residual_fd():
    operator = _make_bilateral_kernel_harness()
    triangle = operator.affine.x.to_numpy()
    displacement = np.zeros(6, dtype=np.float64)
    active = np.r_[np.arange(12, 15), np.arange(0, 9)]
    _clear_coupled_contact_system(operator)
    operator._assemble_mixed_contact_type(6, True, False)
    upper = operator.hash_triplet.values.to_numpy()
    exact = (upper + upper.T - np.diag(np.diag(upper)))[np.ix_(active, active)]

    step = 1.0e-7
    finite_difference = np.zeros((12, 12), dtype=np.float64)
    for column in range(12):
        residuals = []
        for sign in (-1.0, 1.0):
            trial_triangle = triangle.copy()
            trial_displacement = displacement.copy()
            if column < 3:
                trial_displacement[column] += sign * step
            else:
                local = column - 3
                trial_triangle[local // 3, local % 3] += sign * step
            operator.affine.x.from_numpy(trial_triangle)
            operator.soft_disp.from_numpy(trial_displacement)
            _clear_coupled_contact_system(operator)
            operator._assemble_mixed_contact_type(6, False, False)
            residuals.append(operator.global_grad.to_numpy()[active])
        finite_difference[:, column] = (residuals[1] - residuals[0]) / (2.0 * step)

    np.testing.assert_allclose(exact, finite_difference, rtol=2.0e-6, atol=2.0e-5)


def test_soft_affine_runtime_dispatch_masks_match_scalar_oracles():
    operator = _make_bilateral_kernel_harness()
    edge_counts = np.array([0, 2, 0, 1, 0, 0, 3, 0, 1], dtype=np.int32)
    mollifier_counts = np.array([1, 0, 0, 4, 0, 2, 0, 0, 5], dtype=np.int32)
    operator.affine.edge_type_count.from_numpy(edge_counts)
    operator.affine.edge_mollifier_type_count.from_numpy(mollifier_counts)
    expected_edge_mask = sum((1 << edge_type) if edge_counts[edge_type] > 0 else 0 for edge_type in range(9)) + sum(
        (1 << (edge_type + 9)) if mollifier_counts[edge_type] > 0 else 0 for edge_type in range(9)
    )
    assert int(operator._affine_edge_dispatch_mask()) == expected_edge_mask

    mixed_counts = np.array([3, 0, 2, 0, 0, 4, 1], dtype=np.int32)
    operator.mixed_contact_type_count.from_numpy(mixed_counts)
    assert int(operator._mixed_active_contact_count()) == int(mixed_counts.sum())
    expected_mixed_mask = sum((1 << contact_type) if mixed_counts[contact_type] > 0 else 0 for contact_type in range(7))
    assert int(operator._mixed_contact_dispatch_mask()) == expected_mixed_mask


def test_soft_affine_fully_implicit_residual_only_matches_matrix_path():
    operator = _make_bilateral_kernel_harness()
    _enable_fully_implicit_friction(operator)
    displacement = np.zeros(6, dtype=np.float64)
    displacement[1] = 2.0e-4
    operator.soft_disp.from_numpy(displacement)

    def evaluate(assemble, need_matrix):
        _clear_coupled_contact_system(operator)
        assemble(need_matrix)
        return (
            operator.global_grad.to_numpy().copy(),
            operator.hash_triplet.values.to_numpy().copy(),
        )

    for assemble in (operator._assemble_soft_fully_implicit_friction, operator._assemble_mixed_fully_implicit_friction):
        residual_only, residual_only_matrix = evaluate(assemble, False)
        matrix_residual, jacobian = evaluate(assemble, True)
        np.testing.assert_allclose(residual_only, matrix_residual, rtol=2.0e-13, atol=2.0e-13)
        np.testing.assert_array_equal(
            residual_only_matrix,
            np.zeros_like(residual_only_matrix),
        )
        assert np.linalg.norm(residual_only) > 0.0
        assert np.linalg.norm(jacobian) > 0.0

    def assemble_mixed_barrier(need_matrix):
        operator._assemble_mixed_contact_type(6, need_matrix, True)

    residual_only, residual_only_matrix = evaluate(assemble_mixed_barrier, False)
    matrix_residual, barrier_hessian = evaluate(assemble_mixed_barrier, True)
    np.testing.assert_allclose(residual_only, matrix_residual, rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_array_equal(
        residual_only_matrix,
        np.zeros_like(residual_only_matrix),
    )
    assert np.linalg.norm(residual_only) > 0.0
    assert np.linalg.norm(barrier_hessian) > 0.0

    residual_source = inspect.getsource(SoftAffineIPCOperator._fully_implicit_local_friction_residual)
    assert "12, 12" not in residual_source
    for assembler in (
        SoftAffineIPCOperator._assemble_soft_fully_implicit_friction,
        SoftAffineIPCOperator._assemble_mixed_fully_implicit_friction,
        SoftAffineIPCOperator._assemble_mixed_contact_type,
    ):
        source = inspect.getsource(assembler)
        compact_source = " ".join(source.split())
        assert "if ti.static(need_matrix):" in compact_source
        matrix_markers = [
            compact_source.index(marker)
            for marker in (
                "12, 12",
                "point_triangle_distance_grad_hess",
            )
            if marker in compact_source
        ]
        assert matrix_markers
        assert compact_source.index("if ti.static(need_matrix):") < min(matrix_markers)


def test_soft_soft_fully_implicit_friction_full_jacobian_matches_fd():
    operator = _make_bilateral_kernel_harness()
    _enable_fully_implicit_friction(operator)

    hats = np.array(
        [[0.047, 0.012, 0.021], [0.002, -0.001, 0.0]],
        dtype=np.float64,
    )
    displacement = np.array(
        [[0.003, -0.002, 0.004], [-0.001, 0.003, -0.002]],
        dtype=np.float64,
    )
    for point in range(2):
        operator.scene.soft_point[point].x = ti.Vector(hats[point])
    operator.soft_hat_x.from_numpy(hats)
    operator.soft_disp.from_numpy(displacement.reshape(-1))

    _clear_coupled_contact_system(operator)
    operator._assemble_soft_fully_implicit_friction(True)
    active = np.arange(operator.affine_dof, operator.affine_dof + 6)
    residual = operator.global_grad.to_numpy()[active]
    jacobian = operator.hash_triplet.values.to_numpy()[np.ix_(active, active)]

    # Bilateral generalized forces obey action-reaction even though their
    # fully coupled Jacobian is generally nonsymmetric.
    np.testing.assert_allclose(residual.reshape((2, 3)).sum(axis=0), 0.0, atol=1.0e-11)
    assert np.linalg.norm(jacobian - jacobian.T) > 1.0e-8

    h = 2.0e-7
    finite_difference = np.zeros((6, 6), dtype=np.float64)
    for column in range(6):
        values = displacement.reshape(-1).copy()
        values[column] += h
        operator.soft_disp.from_numpy(values)
        _clear_coupled_contact_system(operator)
        operator._assemble_soft_fully_implicit_friction(False)
        plus = operator.global_grad.to_numpy()[active]
        values[column] -= 2.0 * h
        operator.soft_disp.from_numpy(values)
        _clear_coupled_contact_system(operator)
        operator._assemble_soft_fully_implicit_friction(False)
        minus = operator.global_grad.to_numpy()[active]
        finite_difference[:, column] = (plus - minus) / (2.0 * h)
    operator.soft_disp.from_numpy(displacement.reshape(-1))
    np.testing.assert_allclose(jacobian, finite_difference, rtol=2.0e-5, atol=2.0e-6)


def test_mixed_fully_implicit_pt_scattered_full_jacobian_matches_fd():
    operator = _make_bilateral_kernel_harness()
    _enable_fully_implicit_friction(operator)
    operator.fully_stribeck_velocity = 5.0
    operator.affine.pp_kappa[0, 0] = 80.0

    triangle = np.array(
        [[-0.9, -0.8, 0.10], [1.1, -0.65, 0.22], [-0.15, 1.0, -0.12]],
        dtype=np.float64,
    )
    edge01 = triangle[1] - triangle[0]
    edge02 = triangle[2] - triangle[0]
    normal = np.cross(edge01, edge02)
    normal /= np.linalg.norm(normal)
    edge_out = np.cross(edge01, normal)
    edge_out /= np.linalg.norm(edge_out)
    if np.dot(edge_out, triangle[2] - triangle[0]) > 0.0:
        edge_out *= -1.0
    vertex_out = -(
        (triangle[1] - triangle[0]) / np.linalg.norm(triangle[1] - triangle[0])
        + (triangle[2] - triangle[0]) / np.linalg.norm(triangle[2] - triangle[0])
    )
    vertex_out -= normal * np.dot(vertex_out, normal)
    vertex_out /= np.linalg.norm(vertex_out)
    # Current triangle is rotated/skewed, while distinct previous vertex
    # positions produce a non-rigid affine velocity/deformation field.
    affine_delta = np.array(
        [[0.0020, -0.0010, 0.0010], [-0.0010, 0.0020, -0.0010], [0.0010, 0.0015, 0.0020]],
        dtype=np.float64,
    )
    point_delta = np.array([0.0040, -0.0030, 0.0020])
    active = np.r_[np.arange(12, 15), np.arange(0, 9)]
    h = 1.0e-7

    # Reuse one Taichi operator for vertex, edge, and interior cases.  The
    # dynamic fully implicit kernel is therefore compiled only once while all
    # three closest-feature branches still receive a full-column FD oracle.
    for contact_type in (0, 3, 6):
        if contact_type == 6:
            point = 0.25 * triangle[0] + 0.35 * triangle[1] + 0.40 * triangle[2] + 0.05 * normal
        elif contact_type == 3:
            point = 0.45 * triangle[0] + 0.55 * triangle[1]
            point += 0.025 * edge_out + 0.025 * normal
        else:
            point = triangle[0] + 0.025 * vertex_out + 0.025 * normal

        soft_disp = np.zeros(6, dtype=np.float64)
        soft_disp[:3] = point_delta
        operator.affine.x.from_numpy(triangle)
        operator.affine.hat_x.from_numpy(triangle - affine_delta)
        operator.scene.soft_point[0].x = ti.Vector(point - point_delta)
        operator.soft_disp.from_numpy(soft_disp)
        soft_hats = operator.soft_hat_x.to_numpy()
        soft_hats[0] = point - point_delta
        operator.soft_hat_x.from_numpy(soft_hats)

        operator._count_mixed_contact_types()
        counts = operator.mixed_contact_type_count.to_numpy()
        assert counts[contact_type] == 1

        _clear_coupled_contact_system(operator)
        operator._assemble_mixed_contact_type(contact_type, True, False)
        # Fully implicit friction is a single dynamic-feature pass after the
        # type-split barrier assembly, never once per barrier type.
        operator._assemble_mixed_fully_implicit_friction(True)
        residual = operator.global_grad.to_numpy()[active]
        np.testing.assert_allclose(residual.reshape((4, 3)).sum(axis=0), 0.0, atol=1.0e-10)
        full_matrix = operator.hash_triplet.values.to_numpy()
        jacobian = full_matrix[np.ix_(active, active)]
        assert np.linalg.norm(jacobian - jacobian.T) > 1.0e-8

        finite_difference = np.zeros((12, 12), dtype=np.float64)
        for column in range(12):

            def evaluate(offset):
                trial_triangle = triangle.copy()
                trial_soft_disp = soft_disp.copy()
                if column < 3:
                    trial_soft_disp[column] += offset
                else:
                    local = column - 3
                    trial_triangle[local // 3, local % 3] += offset
                operator.affine.x.from_numpy(trial_triangle)
                operator.soft_disp.from_numpy(trial_soft_disp)
                _clear_coupled_contact_system(operator)
                operator._assemble_mixed_contact_type(contact_type, False, False)
                operator._assemble_mixed_fully_implicit_friction(False)
                return operator.global_grad.to_numpy()[active]

            finite_difference[:, column] = (evaluate(h) - evaluate(-h)) / (2.0 * h)
        operator.affine.x.from_numpy(triangle)
        operator.soft_disp.from_numpy(soft_disp)
        relative_error = np.linalg.norm(jacobian - finite_difference, ord=np.inf) / max(
            np.linalg.norm(jacobian, ord=np.inf),
            np.linalg.norm(finite_difference, ord=np.inf),
            1.0e-14,
        )
        assert relative_error < 1.0e-4
        np.testing.assert_allclose(jacobian, finite_difference, rtol=7.0e-5, atol=2.0e-3)

        if contact_type == 6:
            # Contact measure belongs to lambda_N, not to the additive
            # viscous coefficient.  Subtracting the zero-viscosity response
            # isolates that term and must therefore be area-independent.
            def viscous_increment(surface_weight):
                operator.scene.soft_point[0].surface_weight = surface_weight
                operator.fully_mu_viscous = 0.015
                _clear_coupled_contact_system(operator)
                operator._assemble_mixed_fully_implicit_friction(False)
                with_viscosity = operator.global_grad.to_numpy()[active]
                operator.fully_mu_viscous = 0.0
                _clear_coupled_contact_system(operator)
                operator._assemble_mixed_fully_implicit_friction(False)
                without_viscosity = operator.global_grad.to_numpy()[active]
                return with_viscosity - without_viscosity

            np.testing.assert_allclose(
                viscous_increment(1.0),
                viscous_increment(3.7),
                rtol=1.0e-11,
                atol=1.0e-11,
            )
            operator.scene.soft_point[0].surface_weight = 1.0
            operator.fully_mu_viscous = 0.015


def test_soft_affine_production_mixed_pt_uses_frozen_bilateral_stencil():
    operator = _make_bilateral_kernel_harness()
    operator._reset_lagged_friction_cache()
    operator._initialize_mixed_lagged_friction()
    assert int(operator.mixed_friction_count[None]) == 1
    frozen_bary = operator.mixed_friction_bary.to_numpy()[0].copy()
    frozen_normal = operator.mixed_friction_normal.to_numpy()[0].copy()
    frozen_coeff = float(operator.mixed_friction_coeff[0])

    displacement = np.zeros(6, dtype=np.float64)
    displacement[1] = 2.0e-4
    operator.soft_disp.from_numpy(displacement)
    operator.global_grad.fill(0.0)
    operator.energy[None] = 0.0
    operator.hash_triplet.values.fill(0.0)
    operator._assemble_mixed_lagged_friction(True)

    gradient = operator.global_grad.to_numpy()
    assert float(operator.energy[None]) > 0.0
    np.testing.assert_allclose(gradient.reshape((-1, 3)).sum(axis=0), 0.0, atol=1.0e-10)
    upper = operator.hash_triplet.values.to_numpy()
    hessian = upper + upper.T - np.diag(np.diag(upper))
    np.testing.assert_allclose(hessian, hessian.T, atol=1.0e-12)
    eigenvalues = np.linalg.eigvalsh(hessian)
    assert eigenvalues[0] >= (-1.0e-10 * max(1.0, float(np.max(np.abs(eigenvalues)))))
    np.testing.assert_allclose(hessian @ np.tile([0.3, -0.2, 0.1], 6), 0.0, rtol=1.0e-8, atol=1.0e-7)
    np.testing.assert_allclose(operator.mixed_friction_bary.to_numpy()[0], frozen_bary, atol=0.0)
    np.testing.assert_allclose(operator.mixed_friction_normal.to_numpy()[0], frozen_normal, atol=0.0)
    assert float(operator.mixed_friction_coeff[0]) == pytest.approx(frozen_coeff)


@pytest.mark.parametrize("friction_mode", ("lagged", "fully_implicit"))
def test_soft_affine_solver_entry_runs_production_contact_path(friction_mode):
    """Exercise solver dispatch through the real coupled contact assembly."""
    operator = _prepare_production_assembly_harness(_make_bilateral_kernel_harness(), friction_mode)

    # Give both PP and mixed PT contacts a nonzero tangential increment, while
    # preserving their strictly feasible normal gaps inside dhat.
    displacement = np.zeros(operator.max_soft_dof, dtype=np.float64)
    displacement[1] = 2.0e-4
    operator.soft_disp.from_numpy(displacement)

    events = []
    _record_production_contact_dispatch(operator, events)
    snapshots = []
    accepted = []
    affine_iterate = np.zeros(operator.affine_dof, dtype=np.float64)
    operator.affine_state = SimpleNamespace(pack=lambda: affine_iterate.copy())
    operator.begin_step = lambda: events.append(("begin_step",))

    def refresh_lagged_friction(_values):
        events.append(("refresh_lagged_friction",))
        operator._reset_lagged_friction_cache()
        operator._initialize_soft_lagged_friction()
        operator._initialize_mixed_lagged_friction()
        assert int(operator.soft_friction_count[None]) == 1
        assert int(operator.mixed_friction_count[None]) == 1

    operator.refresh_lagged_friction = refresh_lagged_friction
    operator.initialize_fully_implicit_velocity_predictor = lambda _sims, values: np.asarray(
        values, dtype=np.float64
    ).copy()
    production_assemble = operator.assemble

    def assemble(values, need_matrix=True):
        energy, residual = production_assemble(values, need_matrix)
        snapshots.append(
            (
                float(energy),
                np.asarray(residual, dtype=np.float64).copy(),
                bool(need_matrix),
            )
        )
        return energy, residual

    operator.assemble = assemble
    operator.solve_direction = lambda _sims, _residual: (
        np.zeros(operator.affine_dof, dtype=np.float64),
        np.zeros(operator.max_soft_dof, dtype=np.float64),
        np.zeros(operator.total_dof, dtype=np.float64),
    )
    operator.accept_step = lambda values: accepted.append(np.asarray(values, dtype=np.float64).copy())

    engine = SoftAffineIPCEngine()
    engine.operator = operator
    # The fixture is already an initialized production operator; avoid
    # rebuilding the complete LSMPM scene around this narrow integration test.
    if friction_mode == "fully_implicit":
        engine.advance = lambda step_sims, _outer: engine._step_fully_implicit(step_sims)
    else:
        engine.advance = engine._step_lagged
    engine.requested_friction_iterations = 1
    sims = SimpleNamespace(
        affine_friction_mode=friction_mode,
        affine_friction_iterations=1,
        affine_friction_tolerance=1.0e-12,
        affine_max_newton_iteration=2,
        # Fully implicit enters the production residual assembly and terminates
        # at its first Newton convergence check.  Local derivative fidelity is
        # covered separately by the full-column FD tests above.
        affine_newton_tolerance=1.0e30,
        affine_fully_implicit_force_atol=1.0e30,
        affine_fully_implicit_force_rtol=0.0,
        affine_max_step=1.0,
        affine_ccd=False,
        affine_ccd_type="ccd",
        affine_ccd_eta=0.2,
        affine_accd_tolerance=1.0e-7,
        affine_ccd_max_iteration=100,
        affine_line_search_max_iteration=2,
        dt=_ScalarField(operator.dt),
        current_time=0.0,
        current_step=0,
        delta=operator.dt,
    )

    engine.step(sims, SimpleNamespace())

    assert accepted and len(accepted) == 1
    assert snapshots
    matrix_snapshots = [item for item in snapshots if item[2]]
    assert matrix_snapshots
    energy, residual, unused_need_matrix = matrix_snapshots[-1]
    assert np.isfinite(energy) and energy > 0.0
    assert np.all(np.isfinite(residual))
    assert np.linalg.norm(residual[: operator.affine_dof]) > 0.0
    assert np.linalg.norm(residual[operator.affine_dof :]) > 0.0
    np.testing.assert_allclose(
        residual.reshape((-1, 3)).sum(axis=0),
        0.0,
        rtol=1.0e-10,
        atol=1.0e-9,
    )
    assert np.linalg.norm(operator.hash_triplet.values.to_numpy()) > 0.0

    event_names = [event[0] for event in events]
    assert "soft_barrier" in event_names
    assert ("mixed_barrier", 6, True) in events
    assert f"soft_{friction_mode}_friction" in event_names
    assert f"mixed_{friction_mode}_friction" in event_names
    if friction_mode == "lagged":
        assert event_names.count("refresh_lagged_friction") == 2
        assert engine.last_friction_converged
        assert engine.last_friction_residual == 0.0
    else:
        assert "refresh_lagged_friction" not in event_names
        assert engine.last_inner_converged
        assert engine.last_friction_iterations == 1


def test_soft_affine_outer_refresh_happens_only_between_complete_inner_solves():
    engine = SoftAffineIPCEngine()
    events = []
    accepted = []
    state = SimpleNamespace(pack=lambda: np.zeros(3, dtype=np.float64))
    operator = SimpleNamespace(
        dt=1.0e-2,
        affine_state=state,
        begin_step=lambda: events.append(("begin",)),
        refresh_lagged_friction=lambda y: events.append(("refresh", float(np.asarray(y)[0]))),
        accept_step=lambda y: accepted.append(np.asarray(y).copy()),
    )
    gradients = iter((np.ones(3), np.full(3, 0.5), np.full(3, 1.0e-10)))

    def assemble(y, need_matrix=True):
        events.append(("assemble", float(np.asarray(y)[0]), need_matrix))
        return 1.0, next(gradients)

    def solve_direction(_sims, gradient):
        events.append(("probe", float(np.asarray(gradient)[0])))
        direction = -np.asarray(gradient)
        return direction, np.empty(0), direction

    operator.assemble = assemble
    operator.solve_direction = solve_direction
    engine.operator = operator

    def inner(_sims, y, energy, gradient):
        events.append(("inner", float(np.asarray(y)[0]), float(gradient[0])))
        return np.asarray(y) + 1.0, energy, gradient, 3

    engine._solve_lagged_inner = inner
    sims = SimpleNamespace(
        affine_friction_mode="lagged",
        affine_friction_iterations=4,
        affine_friction_max_iterations=10,
        # Matching defaults use one velocity tolerance for both inner Newton
        # and the outer fixed point.
        affine_newton_tolerance=1.0e-7,
        affine_friction_tolerance=1.0e-7,
    )
    engine._step_lagged(sims, requested_outer=4)

    assert [event[0] for event in events] == [
        "begin",
        "refresh",
        "assemble",
        "inner",
        "refresh",
        "assemble",
        "probe",
        "inner",
        "refresh",
        "assemble",
        "probe",
    ]
    assert engine.last_friction_iterations == 2
    assert engine.last_newton_iterations == 6
    assert engine.last_friction_converged
    assert engine.last_friction_residual == pytest.approx(1.0e-8)
    # The refreshed-system correction is only a convergence probe.  Applying
    # the final 1e-10 m probe would have changed the accepted outer iterate.
    np.testing.assert_allclose(accepted[0], 2.0)


def test_soft_affine_device_outer_probe_is_unclamped_unapplied_velocity():
    engine = SoftAffineIPCEngine()
    events = []
    operator = SimpleNamespace(
        dt=1.0e-2,
        _assert_cuda_device_residency=lambda: events.append("residency"),
        begin_step_device=lambda: events.append("begin"),
        refresh_lagged_friction_device=lambda: events.append("refresh"),
        backup_lagged_friction_for_adjoint_device=lambda: events.append("backup"),
        assemble_device=lambda need_matrix=True: (events.append(("assemble", need_matrix)) or 2.0),
        solve_direction_device=lambda _sims, clamp_direction=True: (
            events.append(("probe", clamp_direction)) or {"solution_inf_norm": 2.0e-9}
        ),
        accept_step_device=lambda: events.append("accept"),
    )
    engine.operator = operator
    engine._solve_lagged_inner_device = lambda _sims, energy: (energy - 0.5, 2)
    sims = SimpleNamespace(
        affine_friction_max_iterations=4,
        affine_friction_tolerance=3.0e-7,
    )

    engine._step_lagged_device(sims, requested_outer=1)

    assert engine.last_friction_converged
    assert engine.last_friction_residual == pytest.approx(2.0e-7)
    assert engine.last_friction_iterations == 1
    assert engine.last_newton_iterations == 2
    assert events.index("backup") < events.index(("probe", False))
    assert ("probe", False) in events
    assert events[-1] == "accept"


def test_soft_affine_unbounded_outer_mode_raises_at_safety_cap_without_accepting():
    engine = SoftAffineIPCEngine()
    accepted = []
    state = SimpleNamespace(pack=lambda: np.zeros(3, dtype=np.float64))
    operator = SimpleNamespace(
        dt=2.5e-1,
        affine_state=state,
        begin_step=lambda: None,
        refresh_lagged_friction=lambda _y: None,
        accept_step=lambda y: accepted.append(np.asarray(y).copy()),
        assemble=lambda _y, need_matrix=True: (1.0, np.ones(3)),
        solve_direction=lambda _sims, gradient: (
            -np.asarray(gradient),
            np.empty(0),
            -np.asarray(gradient),
        ),
    )
    engine.operator = operator
    engine._solve_lagged_inner = lambda _sims, y, energy, gradient: (np.asarray(y) + 1.0, energy, gradient, 1)
    sims = SimpleNamespace(
        affine_friction_mode="lagged",
        affine_friction_iterations=-1,
        affine_friction_max_iterations=2,
        # The public outer tolerance remains an explicit extension and is
        # intentionally distinct from the inner Newton tolerance here.
        affine_newton_tolerance=1.0e3,
        affine_friction_tolerance=1.0e-12,
    )

    with pytest.raises(RuntimeError, match="safety cap 2"):
        engine._step_lagged(sims, requested_outer=-1)

    assert engine.last_friction_iterations == 2
    assert engine.last_friction_terminated_by_cap
    assert accepted == []


def test_soft_affine_fully_implicit_newton_uses_residual_armijo():
    engine = SoftAffineIPCEngine()
    matrix = np.array([[2.0, 0.7], [-0.4, 1.6]], dtype=np.float64)
    target = np.array([0.12, -0.08], dtype=np.float64)
    accepted = []

    class Operator:
        def __init__(self):
            self.dt = 1.0e-2
            self.affine_dof = 1
            self.total_dof = 2
            self.max_soft_dof = 1
            self.soft_value = 0.0
            self.soft_base_value = 0.0
            self.soft_direction = _ArrayField(np.zeros(1))
            self.soft_disp = _ArrayField(np.zeros(1))
            self.soft_disp_base = _ArrayField(np.zeros(1))
            self.affine_state = SimpleNamespace(pack=lambda: np.zeros(1, dtype=np.float64))
            self.affine = SimpleNamespace(
                control_num=1,
                y=_ArrayField(np.zeros((1, 3), dtype=np.float64)),
            )
            self.trial_rebuilds = 0

        def begin_step(self):
            return

        def initialize_fully_implicit_velocity_predictor(self, _sims, values):
            return np.asarray(values, dtype=np.float64).copy()

        def assemble(self, y, need_matrix=True):
            self.trial_rebuilds += 1
            state = np.array([float(np.asarray(y)[0]), self.soft_value])
            return 0.0, matrix @ (state - target)

        def solve_direction(self, _sims, residual):
            direction = np.linalg.solve(matrix, -np.asarray(residual))
            self.soft_direction.from_numpy(direction[1:])
            return direction[:1], direction[1:], direction

        def apply_jacobian(self, direction):
            return matrix @ np.asarray(direction)

        def init_step_size(self, *_args, **_kwargs):
            return 1.0

        def store_soft_base(self):
            self.soft_base_value = self.soft_value

        def set_soft_trial(self, alpha):
            self.soft_value = self.soft_base_value + alpha * self.soft_direction.to_numpy()[0]

        def restore_soft_base(self):
            self.soft_value = self.soft_base_value

        def accept_step(self, y):
            accepted.append(np.array([y[0], self.soft_value]))

    operator = Operator()
    engine.operator = operator
    sims = SimpleNamespace(
        affine_newton_tolerance=1.0e-12,
        affine_fully_implicit_force_atol=1.0e-12,
        affine_fully_implicit_force_rtol=0.0,
        affine_max_newton_iteration=4,
        affine_max_step=1.0,
        affine_ccd=False,
        affine_fully_implicit_armijo=1.0e-4,
        affine_fully_implicit_line_search_contraction=0.5,
        affine_line_search_max_iteration=8,
    )
    engine._step_fully_implicit(sims)

    np.testing.assert_allclose(accepted[0], target, atol=1.0e-14)
    assert engine.last_inner_converged
    assert engine.last_newton_iterations == 1
    # Initial residual, trial residual and accepted-state matrix rebuild.
    assert operator.trial_rebuilds >= 3


def test_soft_affine_fully_implicit_small_step_is_stagnation_not_convergence():
    engine = SoftAffineIPCEngine()
    engine.operator = SimpleNamespace(
        dt=1.0,
        solve_direction=lambda _sims, _residual: (np.zeros(1), np.zeros(1), np.zeros(2)),
    )
    sims = SimpleNamespace(
        affine_newton_tolerance=1.0e-8,
        affine_fully_implicit_force_atol=1.0e-8,
        affine_fully_implicit_force_rtol=0.0,
        affine_max_newton_iteration=2,
        affine_max_step=1.0,
    )
    with pytest.raises(RuntimeError, match="Newton solve stagnated"):
        engine._solve_fully_implicit_newton(sims, np.zeros(1), np.ones(2))
    assert not engine.last_inner_converged
    assert engine.last_inner_failure_reason == "newton_stagnation"


def test_soft_affine_fully_implicit_rejects_non_descent_merit_direction():
    engine = SoftAffineIPCEngine()
    engine.operator = SimpleNamespace(
        dt=1.0,
        solve_direction=lambda _sims, _residual: (np.ones(1), np.ones(1), np.ones(2)),
        apply_jacobian=lambda direction: np.asarray(direction),
    )
    sims = SimpleNamespace(
        affine_newton_tolerance=1.0e-8,
        affine_fully_implicit_force_atol=1.0e-8,
        affine_fully_implicit_force_rtol=0.0,
        affine_max_newton_iteration=2,
        affine_max_step=10.0,
    )
    with pytest.raises(RuntimeError, match="not a descent direction"):
        engine._solve_fully_implicit_newton(sims, np.zeros(1), np.ones(2))
    assert engine.last_inner_failure_reason == "non_descent_merit_direction"


def test_soft_affine_fully_implicit_stagnation_rolls_back_trial_state():
    engine = SoftAffineIPCEngine()
    initial_y = np.arange(3, dtype=np.float64)
    accepted = []
    operator = SimpleNamespace(
        dt=1.0,
        affine_dof=1,
        total_dof=2,
        max_soft_dof=1,
        affine_state=SimpleNamespace(pack=lambda: initial_y.copy()),
        affine=SimpleNamespace(
            control_num=1,
            y=_ArrayField(np.full((1, 3), -4.0)),
        ),
        soft_direction=_ArrayField(np.zeros(1)),
        soft_disp=_ArrayField(np.ones(1)),
        soft_disp_base=_ArrayField(np.ones(1)),
        begin_step=lambda: None,
        initialize_fully_implicit_velocity_predictor=lambda _sims, values: np.asarray(values, dtype=np.float64).copy(),
        assemble=lambda _y, need_matrix=True: (0.0, np.ones(2, dtype=np.float64)),
        solve_direction=lambda _sims, _residual: (np.zeros(1), np.zeros(1), np.zeros(2)),
        accept_step=lambda _y: accepted.append(True),
    )
    engine.operator = operator
    sims = SimpleNamespace(
        affine_newton_tolerance=1.0e-8,
        affine_fully_implicit_force_atol=1.0e-8,
        affine_fully_implicit_force_rtol=0.0,
        affine_max_newton_iteration=2,
        affine_max_step=1.0,
    )
    with pytest.raises(RuntimeError, match="Newton solve stagnated"):
        engine._step_fully_implicit(sims)
    np.testing.assert_allclose(operator.soft_disp.to_numpy(), 0.0)
    np.testing.assert_allclose(operator.soft_disp_base.to_numpy(), 0.0)
    np.testing.assert_allclose(operator.affine.y.to_numpy(), initial_y.reshape((1, 3)))
    assert accepted == []
    assert engine.last_inner_failure_reason == "newton_stagnation"


def test_soft_affine_fully_implicit_failure_rolls_back_trial_fields():
    engine = SoftAffineIPCEngine()
    accepted = []
    initial_y = np.arange(6, dtype=np.float64)
    operator = SimpleNamespace(
        affine_state=SimpleNamespace(pack=lambda: initial_y.copy()),
        affine=SimpleNamespace(
            control_num=2,
            y=_ArrayField(np.full((2, 3), -9.0)),
        ),
        soft_disp=_ArrayField(np.ones(3)),
        soft_disp_base=_ArrayField(np.ones(3)),
        begin_step=lambda: None,
        initialize_fully_implicit_velocity_predictor=lambda _sims, values: np.asarray(values, dtype=np.float64).copy(),
        assemble=lambda _y, need_matrix=True: (_ for _ in ()).throw(RuntimeError("synthetic assembly failure")),
        accept_step=lambda _y: accepted.append(True),
    )
    engine.operator = operator

    with pytest.raises(RuntimeError, match="synthetic assembly failure"):
        engine._step_fully_implicit(SimpleNamespace())

    np.testing.assert_allclose(operator.soft_disp.to_numpy(), 0.0)
    np.testing.assert_allclose(operator.soft_disp_base.to_numpy(), 0.0)
    np.testing.assert_allclose(operator.affine.y.to_numpy(), initial_y.reshape((2, 3)))
    assert accepted == []


def test_coupled_affine_subsystem_includes_all_affine_contact_terms():
    events = []

    def record(name, result=None):
        def call(*_args, **_kwargs):
            events.append(name)
            return result

        return call

    neighbor = SimpleNamespace(
        update=record("candidates", 3),
        candidate_count="pt_count",
        candidate_vertex="pt_vertex",
        candidate_face="pt_face",
        edge_candidate_count="ee_count",
        candidate_edge0="ee0",
        candidate_edge1="ee1",
    )
    affine = SimpleNamespace(
        control_num=4,
        levelset_contact=False,
        y=_ArrayField(np.zeros((4, 3))),
        tilde_y=_ArrayField(np.zeros((4, 3))),
        hat_y=_ArrayField(np.zeros((4, 3))),
        grad=_ArrayField(np.zeros((4, 3))),
        energy=_ScalarField(2.0),
        state=SimpleNamespace(gravity=np.zeros(3)),
        neighbor=neighbor,
        x="x",
        dx="dx",
        faces="faces",
        edges="edges",
        node2body="node2body",
        face2body="face2body",
        edge2body="edge2body",
        dhat=0.1,
        scale=1.0,
        contact_damping_stiffness=1.0,
        _clear_system=record("clear"),
        _reconstruct_vertices=record("reconstruct"),
        _assemble_inertia=record("inertia"),
        _assemble_body_force_device=record("body_force"),
        _assemble_local_damping=record("local_damping"),
        _assemble_rigidity=record("rigidity"),
        _assemble_joints=record("joints"),
        _assemble_particle_contacts=record("pt_barrier"),
        _assemble_edge_contacts=record("ee_barrier"),
        _assemble_body_pair_barrier_hessian=record("body_pair_barrier_hessian"),
        _assemble_lagged_friction=record("friction"),
        _assemble_wall_contacts=record("walls"),
        _assemble_contact_damping=record("contact_damping"),
    )
    operator = object.__new__(SoftAffineIPCOperator)
    operator.affine = affine
    operator.profile = False
    operator.fully_implicit = False
    operator._affine_edge_dispatch_mask = lambda: 0
    operator.affine_state = SimpleNamespace(
        tilde_y=np.zeros((1, 4, 3)),
        hat_y=np.zeros((1, 4, 3)),
    )

    energy, gradient = SoftAffineIPCOperator._assemble_affine_self(operator, np.zeros(12), need_matrix=True)

    assert energy == pytest.approx(2.0)
    assert gradient.shape == (12,)
    for required in (
        "pt_barrier",
        "ee_barrier",
        "body_pair_barrier_hessian",
        "joints",
        "friction",
        "walls",
        "contact_damping",
    ):
        assert required in events
