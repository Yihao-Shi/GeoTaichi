from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("taichi")

from examples.fedem.FEMAffineMixedTriaxial25.fem_affine_mixed_triaxial_25 import (
    fem_contact_pair_capacities,
)
from src.fedem.AffineIPCEngine import FEMAffineIPCEngine
from src.fedem.Simulation import FEDEMSimulation
from src.fedem.contact.AffineIPCAssembler import FEMAffineIPCAssembler
from src.fem.engines.ClassicalAssembler import ClassicalAssembler
from src.fempm.ImplicitEngine import FEMPMImplicitEngine
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix
from src.mpdem.engines.DirectAffineIPCAssembler import (
    DirectAffineIPCAssembler,
)
from src.mpdem.engines.DirectAffineIPCSystem import DirectAffineIPCSystem
from src.mpdem.Simulation import Simulation as MPDEMSimulation


def test_triaxial_fem_contact_capacity_covers_all_component_pairs():
    assert fem_contact_pair_capacities(13) == (16384, 32768)


def test_mpdem_accepts_three_dimensional_direct_mpm_affine_ipc():
    coupling = object.__new__(MPDEMSimulation)
    coupling.delta = 1.0e-3
    coupling.dem_timestep = 1.0e-3
    coupling.coupling_scheme = "MPDEM"
    coupling.validate_configuration = lambda: None
    mpm = SimpleNamespace(
        dimension=3,
        solver_type="Implicit",
        ipc_contact=True,
        is_direct_backend=lambda: True,
    )
    dem = SimpleNamespace(scheme="AffineBody")

    coupling.validate_coupling_configuration(mpm, dem)

    mpm.dimension = 2
    with pytest.raises(RuntimeError, match="supports 3D only"):
        coupling.validate_coupling_configuration(mpm, dem)


def test_fem_affine_coo_capacity_expands_only_dense_block_sources():
    affine_controls = 4
    fem_nodes = 68_921
    affine_raw = 432
    fem_raw = 384_000 * 4 * 3
    contact_raw = 2 * (169 + 100) * 4_096

    capacity = FEMAffineIPCEngine._coo_scalar_capacity(
        affine_controls,
        fem_nodes,
        affine_raw,
        fem_raw,
        contact_raw,
    )

    node_count = affine_controls + fem_nodes
    expected = (
        9 * (affine_controls + fem_nodes + node_count) + 9 * (affine_raw + fem_raw + contact_raw) + 3 * node_count
    )
    assert capacity == expected
    assert capacity == 62_756_145


def test_fem_affine_line_search_passes_accepted_step_to_fem_hook():
    engine = object.__new__(FEMAffineIPCEngine)
    events = []
    engine._split_direction = lambda: None
    engine._reduce_metrics = lambda: None
    engine.directional_derivative = {None: -1.0}
    engine.affine = SimpleNamespace(
        init_step_size_device=lambda **_kwargs: 1.0,
        device_backup_line_search_base=lambda: None,
        device_set_line_search_trial=lambda alpha: None,
        device_restore_line_search_base=lambda: None,
    )
    state = SimpleNamespace(
        position=object(),
        direction=object(),
        set_trial_position=lambda alpha: None,
        accept_trial_position=lambda: None,
    )
    engine.fem = SimpleNamespace(
        state=state,
        minimum_jacobian=0.0,
        _maximum_admissible_step_device=lambda: 1.0,
        _minimum_jacobian_ratio_device=lambda _position: 1.0,
        _after_nonlinear_update_device=lambda alpha: events.append(("fem", alpha)),
    )
    engine.contact = SimpleNamespace(
        ccd_eta=0.2,
        ccd_max_iterations=20,
        maximum_step=lambda *_args: 1.0,
        accept_update=lambda _position: events.append(("contact",)),
    )
    engine._total_energy = lambda: 0.0
    engine.line_search_max_backtracks = 2
    engine.line_search_minimum_step = 1.0e-8
    engine.line_search_c1 = 1.0e-4
    engine.line_search_reduction = 0.5

    alpha, backtracks, energy = engine._line_search(1.0)

    assert (alpha, backtracks, energy) == (1.0, 0, 0.0)
    assert events == [("fem", 1.0), ("contact",)]


def test_affine_pressure_servo_moves_inward_when_pressure_is_low_and_reverses_when_high():
    normal = np.asarray([1.0, 0.0, 0.0])
    low_velocity, low_force = FEMAffineIPCEngine._pressure_servo_velocity(
        np.asarray([-20.0, 0.0, 0.0]), normal, 100.0, 0.01, 0.02
    )
    high_velocity, high_force = FEMAffineIPCEngine._pressure_servo_velocity(
        np.asarray([-150.0, 0.0, 0.0]), normal, 100.0, 0.01, 0.02
    )

    assert low_force == 20.0
    assert high_force == 150.0
    assert low_velocity[0] > 0.0
    assert high_velocity[0] < 0.0


def test_affine_source_capacity_uses_fixed_primitive_pair_capacities():
    calls = []

    def estimate(_sims, **kwargs):
        calls.append(kwargs)
        return 8_192

    engine = object.__new__(FEMAffineIPCEngine)
    engine.dem_wrapper = SimpleNamespace(
        sims=SimpleNamespace(
            max_point_triangle_pairs=123,
            max_edge_edge_pairs=456,
        )
    )
    engine.affine = SimpleNamespace(
        body_num=1,
        vertex_num=162,
        edge_num=480,
        _estimate_hash_triplet_capacity=estimate,
    )

    assert engine._affine_source_raw_capacity() == 8_192
    assert calls == [
        {
            "vf_candidate_capacity": 123,
            "ee_candidate_capacity": 456,
            "full_symmetric_input": True,
            "safety_factor": 1.0,
            "minimum_capacity": 1,
        }
    ]


def test_fedem_primitive_pair_defaults_scale_with_surface_triangles(
    taichi_runtime,
):
    simulation = FEDEMSimulation()
    dem = SimpleNamespace(max_particle_num=100, max_material_num=2)
    simulation.configure_memory(
        {
            "point_triangle_coordination_number": 2.5,
            "edge_edge_coordination_number": 7.25,
        },
        dem,
        face_count=12,
        body_count=1,
    )

    assert simulation.max_contact_pairs == 1_600
    assert simulation.max_point_triangle_pairs == 30
    assert simulation.max_edge_edge_pairs == 87


def test_fedem_absolute_primitive_capacities_override_coordination(
    taichi_runtime,
):
    simulation = FEDEMSimulation()
    dem = SimpleNamespace(max_particle_num=100, max_material_num=2)
    simulation.configure_memory(
        {
            "max_point_triangle_pairs": 123,
            "max_edge_edge_pairs": 456,
        },
        dem,
        face_count=12,
        body_count=1,
    )

    assert simulation.max_point_triangle_pairs == 123
    assert simulation.max_edge_edge_pairs == 456


def test_mixed_contact_capacity_uses_declared_pair_capacity_and_friction():
    engine = object.__new__(FEMAffineIPCEngine)
    engine.simulation = SimpleNamespace(max_contact_pairs=4_096)
    engine.simulation.max_point_triangle_pairs = 4_096
    engine.simulation.max_edge_edge_pairs = 4_096
    engine.contact = SimpleNamespace(activate_friction=True)
    assert engine._configured_contact_raw_capacity() == 2 * (169 + 100) * 4_096

    engine.contact.activate_friction = False
    assert engine._configured_contact_raw_capacity() == (169 + 100) * 4_096


def test_direct_mpm_affine_contact_capacity_covers_basis_pullback():
    assembler = object.__new__(DirectAffineIPCAssembler)
    assembler.mpm = SimpleNamespace(shape_func=SimpleNamespace(max_node_per_particle=8))
    assembler.friction_count = 3
    assembler.activate_friction = True
    assembler.max_point_triangle_pairs = 11

    assert assembler.contact_block_capacity(5) == 8 * 20**2
    assert assembler.configured_contact_block_capacity() == 2 * 11 * 20**2


def test_direct_mpm_affine_gradient_uses_grid_shape_and_abd_basis(
    taichi_runtime,
):
    import taichi as ti

    assembler = object.__new__(DirectAffineIPCAssembler)
    assembler.surface_count = 1
    assembler.affine_control_count = 4
    basis = ti.field(ti.f64, shape=(3, 4))
    basis_values = np.asarray([[1.0, 0.2, 0.0, 0.0], [1.0, 0.0, 0.3, 0.0], [1.0, 0.0, 0.0, 0.4]])
    basis.from_numpy(basis_values)
    node2body = ti.field(ti.i32, shape=3)
    node2body.fill(0)
    scale = ti.field(ti.f64, shape=())
    scale[None] = 1.0
    assembler.affine = SimpleNamespace(
        basis=basis,
        node2body=node2body,
        scale_device=scale,
    )

    offset = ti.field(ti.i32, shape=1)
    offset[0] = 2
    local_nodes = ti.field(ti.i32, shape=(1, 2))
    local_nodes.from_numpy(np.asarray([[0, 1]], dtype=np.int32))
    node2dof = ti.field(ti.i32, shape=2)
    node2dof.from_numpy(np.asarray([1, 2], dtype=np.int32))
    shape = ti.field(ti.f64, shape=(1, 2))
    shape.from_numpy(np.asarray([[0.25, 0.75]], dtype=np.float64))
    assembler.mpm = SimpleNamespace(
        offset=offset,
        LnID=local_nodes,
        node2dof=node2dof,
        shape=shape,
    )
    rhs = ti.field(ti.f64, shape=18)
    gradient_values = np.arange(1.0, 13.0)

    @ti.kernel
    def scatter():
        assembler._scatter_gradient(
            ti.Vector([0, 1, 2, 3]),
            0,
            ti.Vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
            rhs,
        )

    scatter()
    expected = np.zeros((6, 3))
    triangle_gradient = gradient_values[3:].reshape(3, 3)
    expected[:4] = -(basis_values.T @ triangle_gradient)
    expected[4] = -0.25 * gradient_values[:3]
    expected[5] = -0.75 * gradient_values[:3]
    np.testing.assert_allclose(rhs.to_numpy().reshape(6, 3), expected)


def test_direct_mpm_affine_ccd_maps_control_direction_to_vertices(
    taichi_runtime,
):
    import taichi as ti

    assembler = object.__new__(DirectAffineIPCAssembler)
    assembler.surface_count = 1
    assembler.fem_node_count = 3
    assembler.real_type = ti.f64
    assembler.end_positions = ti.Vector.field(3, ti.f64, shape=4)

    basis_values = np.asarray([[1.0, 0.2, 0.0, 0.0], [1.0, 0.0, 0.3, 0.0], [1.0, 0.0, 0.0, 0.4]])
    basis = ti.field(ti.f64, shape=(3, 4))
    basis.from_numpy(basis_values)
    node2body = ti.field(ti.i32, shape=3)
    node2body.fill(0)
    assembler.affine = SimpleNamespace(basis=basis, node2body=node2body)

    surface_id = ti.field(ti.i32, shape=1)
    particle = ti.Struct.field({"x": ti.types.vector(3, ti.f64)}, shape=1)
    particle[0].x = [0.1, 0.2, 0.3]
    offset = ti.field(ti.i32, shape=1)
    offset[0] = 1
    local_nodes = ti.field(ti.i32, shape=(1, 1))
    local_nodes[0, 0] = 0
    node2dof = ti.field(ti.i32, shape=1)
    node2dof[0] = 1
    shape = ti.field(ti.f64, shape=(1, 1))
    shape[0, 0] = 0.5
    assembler.mpm = SimpleNamespace(
        surface_id=surface_id,
        particle=particle,
        offset=offset,
        LnID=local_nodes,
        node2dof=node2dof,
        shape=shape,
    )

    affine_position = ti.Vector.field(3, ti.f64, shape=3)
    affine_position.from_numpy(np.arange(9, dtype=np.float64).reshape(3, 3))
    affine_direction = ti.Vector.field(3, ti.f64, shape=4)
    affine_direction_values = np.arange(12, dtype=np.float64).reshape(4, 3) / 10.0
    affine_direction.from_numpy(affine_direction_values)
    mpm_displacement = ti.field(ti.f64, shape=3)
    mpm_displacement.from_numpy(np.asarray([1.0, 2.0, 3.0]))
    mpm_direction = ti.field(ti.f64, shape=3)
    mpm_direction.from_numpy(np.asarray([0.2, 0.4, 0.6]))

    assembler.build_end_positions(
        affine_position,
        affine_direction,
        mpm_displacement,
        mpm_direction,
    )

    expected = np.empty((4, 3))
    expected[0] = [0.1, 0.2, 0.3] + 0.5 * np.asarray([1.2, 2.4, 3.6])
    expected[1:] = affine_position.to_numpy() + basis_values @ affine_direction_values
    np.testing.assert_allclose(assembler.end_positions.to_numpy(), expected)


def test_direct_mpm_affine_linearization_merges_each_raw_source_once():
    class Matrix:
        def __init__(self, name):
            self.name = name
            self.events = []

        def reset_system(self):
            self.events.append(("reset",))

        def append_raw_from(self, source, **kwargs):
            self.events.append(("append", source.name, kwargs))

        def canonicalize_full_symmetric_input(self):
            self.events.append(("canonicalize",))

        def finalize_taichi_assembly(self):
            self.events.append(("finalize",))

    system = object.__new__(DirectAffineIPCSystem)
    system.affine_controls = 4
    system.affine_source = Matrix("affine")
    system.mixed_source = Matrix("mixed")
    system.mpm_sources = [Matrix("material"), Matrix("self_contact")]
    system.matrix = Matrix("global")
    system.rhs = object()
    events = []
    system.affine = SimpleNamespace(
        x=object(),
        scale_device={None: 0.25},
        assemble_device=lambda **kwargs: events.append(("affine", kwargs)),
        bind_hash_triplet=lambda *args, **kwargs: events.append(("bind", args, kwargs)),
    )
    system.mpm = SimpleNamespace(
        grid_disp=object(),
        dirichlet=SimpleNamespace(num=0),
    )
    system.ipc = SimpleNamespace(
        assemble_current_sources=lambda *args, **kwargs: (events.append(("mpm", args, kwargs)) or 6)
    )
    system.mixed = SimpleNamespace(
        activate_friction=True,
        friction_count=2,
        prepare=lambda *args: events.append(("prepare", args)) or 3,
        candidate_field=lambda: "candidates",
        assemble=lambda *args, **kwargs: events.append(("mixed", args, kwargs)),
    )
    system._load_rhs = lambda active: events.append(("rhs", active))
    system._copy_physical_rhs = lambda active: events.append(("physical_rhs", active))

    result = system.assemble_linearization_device()

    assert result["active_mpm_dof"] == 6
    assert result["active_nodes"] == 6
    assert result["active_dof"] == 18
    assert result["contact_count"] == 3
    assert ("physical_rhs", 18) in events
    assert system.matrix.events == [
        ("reset",),
        ("append", "affine", {"active_nodes": 4}),
        (
            "append",
            "material",
            {"active_nodes": 2, "block_offset": 4, "scale": 0.25},
        ),
        (
            "append",
            "self_contact",
            {"active_nodes": 2, "block_offset": 4, "scale": 0.25},
        ),
        ("append", "mixed", {"active_nodes": 6}),
        ("canonicalize",),
        ("finalize",),
    ]


def test_direct_mpm_affine_adjoint_reassembles_exact_unshifted_jacobian():
    class Field:
        def from_numpy(self, values):
            self.values = values.copy()

        def fill(self, value):
            self.value = value

    class Matrix:
        solver = "PCG"
        matrix_symmetric = True

        def solve_flat_system(self, *args, **kwargs):
            assert self.solver == "PCG"
            assert kwargs["transpose"] is False
            assert kwargs["fallback_to_bicgstab"] is True
            return {"converged": True, "residual": 0.0}

    system = object.__new__(DirectAffineIPCSystem)
    system.affine_controls = 4
    system.dof_capacity = 15
    system.rhs = Field()
    system.correction = Field()
    system.matrix = Matrix()
    system.mpm = SimpleNamespace(
        active_dof=3,
        linear_solver_tolerance=1.0e-10,
        linear_solver_max_iters=100,
    )
    events = []
    system.restore_lagged_friction_for_adjoint_device = lambda: events.append("restore")
    system.assemble_linearization_device = lambda **kwargs: (events.append(kwargs) or {"active_mpm_dof": 3})
    system._scatter_adjoint = lambda active: events.append(("scatter", active))

    system.solve_adjoint_device(np.ones(15), exact_plastic_tangent=True)

    assert events == [
        "restore",
        {
            "project_spd": False,
            "exact_plastic_tangent": True,
            "solver_shift": False,
        },
        ("scatter", 3),
    ]
    assert system.matrix.solver == "PCG"


def test_direct_mpm_affine_solution_scatter_stays_on_device(taichi_runtime):
    import taichi as ti

    system = object.__new__(DirectAffineIPCSystem)
    system.affine_controls = 4
    system.dof_capacity = 18
    system.correction = ti.field(ti.f64, shape=18)
    system.correction.from_numpy(np.arange(18, dtype=np.float64))
    affine_direction = ti.Vector.field(3, ti.f64, shape=4)
    mpm_direction = ti.field(ti.f64, shape=6)
    mpm_displacement = ti.field(ti.f64, shape=6)
    mpm_trial = ti.field(ti.f64, shape=6)
    mpm_displacement.fill(-1.0)
    mpm_trial.from_numpy(np.arange(6, dtype=np.float64))
    system.affine = SimpleNamespace(direction_y=affine_direction)
    system.mpm = SimpleNamespace(
        degree_of_freedom=6,
        incre_resolution=mpm_direction,
        grid_disp=mpm_displacement,
        grid_disp_temp=mpm_trial,
    )

    system._scatter_correction(3)

    np.testing.assert_allclose(affine_direction.to_numpy(), np.arange(12).reshape(4, 3))
    np.testing.assert_allclose(mpm_direction.to_numpy(), [12.0, 13.0, 14.0, 0.0, 0.0, 0.0])
    system._accept_mpm_trial(3)
    np.testing.assert_allclose(mpm_displacement.to_numpy(), [0.0, 1.0, 2.0, -1.0, -1.0, -1.0])


def test_direct_mpm_affine_restores_accepted_mixed_friction_cache(
    taichi_runtime,
):
    import taichi as ti

    assembler = object.__new__(DirectAffineIPCAssembler)
    assembler.activate_friction = True
    assembler.friction_capacity = 2
    assembler.friction_count = 1
    assembler.friction_candidate = ti.Vector.field(4, ti.i32, shape=2)
    assembler.friction_weight = ti.Vector.field(4, ti.f64, shape=2)
    assembler.friction_normal = ti.Vector.field(3, ti.f64, shape=2)
    assembler.friction_normal_force = ti.field(ti.f64, shape=2)
    assembler.adjoint_friction_count = ti.field(ti.i32, shape=())
    assembler.adjoint_friction_valid = ti.field(ti.i32, shape=())
    assembler.adjoint_friction_candidate = ti.Vector.field(4, ti.i32, shape=2)
    assembler.adjoint_friction_weight = ti.Vector.field(4, ti.f64, shape=2)
    assembler.adjoint_friction_normal = ti.Vector.field(3, ti.f64, shape=2)
    assembler.adjoint_friction_normal_force = ti.field(ti.f64, shape=2)
    assembler.friction_candidate.from_numpy(np.array([[1, 2, 3, 4], [0, 0, 0, 0]], dtype=np.int32))
    assembler.friction_weight.from_numpy(np.array([[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]]))
    assembler.friction_normal.from_numpy(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    assembler.friction_normal_force.from_numpy(np.array([7.0, 0.0]))

    assembler.backup_lagged_friction_for_adjoint_device()
    assembler.friction_count = 0
    assembler.friction_candidate.fill(0)
    assembler.friction_weight.fill(0.0)
    assembler.friction_normal.fill(0.0)
    assembler.friction_normal_force.fill(0.0)
    assembler.restore_lagged_friction_for_adjoint_device()

    assert assembler.friction_count == 1
    np.testing.assert_array_equal(assembler.friction_candidate.to_numpy()[0], [1, 2, 3, 4])
    np.testing.assert_allclose(assembler.friction_weight.to_numpy()[0], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(assembler.friction_normal.to_numpy()[0], [0.0, 1.0, 0.0])
    assert assembler.friction_normal_force[0] == pytest.approx(7.0)


def test_direct_mpm_affine_ccd_uses_all_three_contact_systems():
    system = object.__new__(DirectAffineIPCSystem)
    calls = []
    system.affine = SimpleNamespace(
        x="affine_position",
        direction_y="affine_direction",
        init_step_size_device=lambda **kwargs: (calls.append(("affine", kwargs)) or 0.8),
    )
    system.mpm = SimpleNamespace(
        active_dof=6,
        grid_disp="mpm_displacement",
        incre_resolution="mpm_direction",
    )
    system.ipc = SimpleNamespace(ccd=lambda active: calls.append(("mpm", active)) or 0.6)
    system.mixed = SimpleNamespace(maximum_step=lambda *args: calls.append(("mixed", args)) or 0.7)

    step = system.maximum_step_device(ccd_eta=0.1, ccd_max_iterations=42)

    assert step == pytest.approx(0.6)
    assert calls == [
        (
            "affine",
            {
                "ccd_type": "ccd",
                "eta": 0.1,
                "accd_tolerance": 1.0e-7,
                "max_iteration": 42,
            },
        ),
        ("mpm", 6),
        (
            "mixed",
            (
                "affine_position",
                "affine_direction",
                "mpm_displacement",
                "mpm_direction",
            ),
        ),
    ]


def test_direct_mpm_affine_energy_uses_abd_time_scale():
    class Field:
        def __init__(self, value=None):
            self.value = value

        def __getitem__(self, key):
            assert key is None
            return self.value

        def fill(self, value):
            self.value = value

    system = object.__new__(DirectAffineIPCSystem)
    system.affine = SimpleNamespace(
        x="affine_position",
        scale_device=Field(0.25),
        assemble_device=lambda **kwargs: 3.0,
    )
    system.mpm = SimpleNamespace(grid_disp="mpm_displacement")
    system.ipc = SimpleNamespace(total_energy=lambda displacement: 4.0)
    system.rhs = Field()
    system.mixed_source = object()
    system.mixed = SimpleNamespace(
        activate_friction=True,
        friction_count=2,
        total_energy=Field(5.0),
        prepare=lambda *args: 3,
        candidate_field=lambda: "candidates",
        assemble=lambda *args: None,
    )

    assert system.total_energy_device() == pytest.approx(5.25)


def test_direct_mpm_affine_lagged_driver_reuses_device_primitives():
    class Field:
        def fill(self, value):
            raise AssertionError(f"unexpected rollback to {value}")

    system = object.__new__(DirectAffineIPCSystem)
    system.affine = SimpleNamespace(
        is_semi=False,
        device_restore_step_start=lambda: pytest.fail("unexpected rollback"),
    )
    system.mpm = SimpleNamespace(dt=1.0, grid_disp=Field())
    events = []
    energies = iter((10.0, 9.0, 9.0))
    directions = iter((1.0, 0.0, 0.0))
    system.begin_step_device = lambda dt: events.append(("begin", dt))
    system.total_energy_device = lambda displacement=None: (events.append(("energy", displacement)) or next(energies))
    system.assemble_linearization_device = lambda **kwargs: (
        events.append(("assemble", kwargs)) or {"active_mpm_dof": 3}
    )
    system.solve_direction_device = lambda active: events.append(("solve", active))
    system.direction_inf_norm = lambda active: next(directions)
    system.scale_direction_device = lambda scale, active: events.append(("scale", scale, active))
    system.gradient_direction_dot = lambda active: -1.0
    system.maximum_step_device = lambda **kwargs: 0.75
    system.begin_line_search_device = lambda: events.append(("line",))
    system.set_line_search_trial_device = lambda alpha, active: (events.append(("trial", alpha, active)) or "trial")
    system.accept_line_search_trial_device = lambda active: events.append(("accept", active))
    system.refresh_lagged_friction_device = lambda: events.append(("refresh",))
    sims = SimpleNamespace(
        affine_friction_iterations=1,
        affine_friction_max_iterations=5,
        affine_max_newton_iteration=3,
        affine_newton_tolerance=1.0e-6,
        affine_friction_tolerance=1.0e-6,
        affine_max_step=0.5,
        affine_ccd=True,
        affine_ccd_type="ccd",
        affine_ccd_eta=0.2,
        affine_accd_tolerance=1.0e-7,
        affine_ccd_max_iteration=10,
        affine_line_search_max_iteration=4,
    )

    result = system.solve_lagged_equilibrium_device(sims)

    assert result == {
        "newton_iterations": 1,
        "friction_iterations": 1,
        "friction_residual": 0.0,
        "friction_converged": True,
        "energy": 9.0,
        "active_mpm_dof": 3,
    }
    assert ("scale", 0.5, 3) in events
    assert ("trial", 0.75, 3) in events
    assert events.count(("solve", 3)) == 3


def test_direct_mpm_affine_semi_convergence_requires_all_contacts():
    system = object.__new__(DirectAffineIPCSystem)
    system.affine = SimpleNamespace(semi_contact_converged=lambda: True)
    system.ipc = SimpleNamespace(contact_converged=lambda: False)
    system.mixed = SimpleNamespace(contact_converged=lambda: True)

    assert system.contact_converged() is False


def test_fem_contact_capacity_uses_declared_pair_caps_and_friction():
    engine = object.__new__(FEMAffineIPCEngine)
    first = SimpleNamespace(
        activate_friction=False,
        max_point_triangle_pairs=11,
        max_edge_edge_pairs=13,
    )
    second = SimpleNamespace(
        activate_friction=True,
        max_point_triangle_pairs=17,
        max_edge_edge_pairs=19,
    )
    engine.fem_contact = SimpleNamespace(assemblers=(first, second))

    assert engine._configured_fem_contact_raw_capacity() == 12 * (11 + 13) + 24 * (17 + 19)


def test_fem_affine_matrix_dispatches_only_present_contact_types():
    assembler = object.__new__(FEMAffineIPCAssembler)
    events = []
    assembler.is_semi = False
    assembler.activate_friction = False
    assembler._reset_contact_terms = lambda: None
    assembler._contact_feature_mask = lambda _pt, _ee: (1 << 2) | (1 << (7 + 8))
    assembler._assemble_point_triangle_barrier_direct_type = (
        lambda _count, contact_type, _matrix, _rhs, _need_matrix, project_spd: events.append(
            ("pt", contact_type, project_spd)
        )
    )
    assembler._assemble_edge_edge_barrier_direct_type = (
        lambda _count, contact_type, _matrix, _rhs, _need_matrix, project_spd: events.append(
            ("ee", contact_type, project_spd)
        )
    )

    assembler.assemble(3, 4, None, None, True, project_spd=False)

    assert events == [("pt", 2, False), ("ee", 8, False)]


def test_fem_affine_matrix_skips_empty_friction_kernels():
    assembler = object.__new__(FEMAffineIPCAssembler)
    assembler.is_semi = False
    assembler.activate_friction = True
    assembler.friction_pt_count = 0
    assembler.friction_ee_count = 0
    assembler._reset_contact_terms = lambda: None
    assembler._contact_feature_mask = lambda _pt, _ee: 0
    fail = lambda *_args: (_ for _ in ()).throw(AssertionError("empty friction kernel launched"))
    assembler._compute_lagged_friction_gradient = fail
    assembler._scatter_lagged_friction_gradient = fail
    assembler._compute_lagged_friction_hessian = fail
    assembler._scatter_lagged_friction_hessian = fail

    assembler.assemble(0, 0, None, None, True)


def test_coordinate_sparse_matrix_rejects_capacity_above_taichi_int32_limit():
    with pytest.raises(ValueError, match="dense SNode int32 limit"):
        CoordinateSparseMatrix(
            int(np.iinfo(np.int32).max) + 1,
            degree_of_freedom=1,
            linear_solver=False,
        )


def test_fempm_coo_capacity_counts_each_source_diagonal():
    nodes = 100
    raw_blocks = 1_234
    dofs = 3 * nodes
    assert FEMPMImplicitEngine._coo_scalar_capacity(nodes, raw_blocks, dofs) == 18 * nodes + 9 * raw_blocks + dofs


def test_classical_fem_reduced_capacity_uses_unique_mesh_edges():
    # The two tetrahedra share the triangle (1, 2, 3).  Their 12 raw
    # undirected element edges therefore reduce to nine unique edges, or 18
    # directed off-diagonal block coordinates.
    assembler = object.__new__(ClassicalAssembler)
    assembler.mesh = SimpleNamespace(cells=np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32))
    assembler.node_count = 5
    assembler.cell_count = 2
    assembler.nodes_per_cell = 4
    assembler.allocate_hessian = True
    assembler._stiffness_unique_block_pair_count = None
    assert assembler.stiffness_block_pair_count == 24
    assert assembler.stiffness_unique_block_pair_count == 18
